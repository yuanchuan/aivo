//! Responses API ↔ Chat Completions conversion logic
//!
//! Converts between OpenAI Responses API format and Chat Completions format.
//! Used by the ResponsesToChatRouter and ServeRouter to bridge clients that
//! speak the Responses API with providers that only support Chat Completions.
use anyhow::{Context, Result};

use crate::services::codex_model_map::map_model_for_codex_cli;
use crate::services::http_utils::{self, SseLineBuffer, current_unix_ts, gen_id, sse_event};
use crate::services::model_names::select_model_for_provider_attempt;
use crate::services::openai_models::{
    OpenAIChatRequest, ResponsesResponse,
    convert_chat_to_responses_request as convert_typed_chat_to_responses,
    convert_responses_to_chat_response as convert_typed_responses_to_chat,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::tool_call_accumulator::{StreamedToolCall, accumulate_tool_call_deltas};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

/// Qualified chat tool name → (`namespace`, original tool name) for tools
/// advertised inside a Responses API `namespace` group. Chat Completions has
/// no namespace concept, so namespaced tools are flattened to a single
/// qualified function name (`namespace__tool`); this map reverses that
/// flattening when the model's call is converted back into a Responses
/// `function_call` item (codex's tool router matches on both fields).
pub type ToolNamespaceMap = HashMap<String, (String, String)>;

/// Codex's default namespace for top-level function/custom tools. Tools in
/// this namespace keep their plain names — codex resolves them via the
/// default namespace, so no qualification is needed (or wanted: the model
/// must keep calling e.g. `exec_command`, not `functions__exec_command`).
const DEFAULT_FUNCTIONS_NAMESPACE: &str = "functions";

/// Builds the chat-visible name for a tool inside a non-default namespace:
/// `{namespace}__{tool}`, sanitized to the character set Chat Completions
/// function names allow (`[A-Za-z0-9_-]`), so picky upstreams don't 400.
fn qualified_namespace_tool_name(namespace: &str, tool: &str) -> String {
    format!("{namespace}__{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Walks every tool a Responses request advertises — top-level `tools` plus
/// `additional_tools` input items (codex ≥0.143 sol) — expanding `namespace`
/// groups. The visitor receives each leaf tool and its namespace (`None` for
/// top-level tools and the default `functions` namespace).
fn for_each_responses_tool<'a>(body: &'a Value, mut visit: impl FnMut(&'a Value, Option<&'a str>)) {
    fn walk<'a>(
        tool: &'a Value,
        namespace: Option<&'a str>,
        visit: &mut impl FnMut(&'a Value, Option<&'a str>),
    ) {
        if tool.get("type").and_then(|v| v.as_str()) == Some("namespace") {
            let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let nested = if name.is_empty() || name == DEFAULT_FUNCTIONS_NAMESPACE {
                None
            } else {
                Some(name)
            };
            for tool in tool
                .get("tools")
                .and_then(|t| t.as_array())
                .into_iter()
                .flatten()
            {
                walk(tool, nested, visit);
            }
        } else {
            visit(tool, namespace);
        }
    }
    for tool in body
        .get("tools")
        .and_then(|t| t.as_array())
        .into_iter()
        .flatten()
    {
        walk(tool, None, &mut visit);
    }
    if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            if item.get("type").and_then(|v| v.as_str()) == Some("additional_tools")
                && let Some(tools) = item.get("tools").and_then(|t| t.as_array())
            {
                for tool in tools {
                    walk(tool, None, &mut visit);
                }
            }
        }
    }
}

/// The subset of a router's config request conversion reads (model selection,
/// token capping — not transport), keeping conversion above the router layer.
pub struct ResponsesToChatConversionConfig {
    pub requires_reasoning_content: bool,
    pub target_base_url: String,
    pub target_protocol: ProviderProtocol,
    /// Whether the upstream uses a Copilot token (affects model selection).
    pub is_copilot: bool,
    pub model_prefix: Option<String>,
    pub actual_model: Option<String>,
    pub max_tokens_cap: Option<u64>,
}

/// Returns true if the body uses OpenAI Responses API format
/// (has "input" array, no "messages" array)
pub fn is_responses_api_format(body: &Value) -> bool {
    // `input` may be an item array or (rarely) a bare prompt string.
    body.get("input")
        .is_some_and(|v| v.is_array() || v.is_string())
        && body.get("messages").is_none()
}

pub(crate) fn cap_token_value(v: &Value, cap: Option<u64>) -> Value {
    if let Some(limit) = cap {
        http_utils::parse_token_u64(v)
            .map(|n| {
                if n == 0 {
                    json!(n)
                } else {
                    json!(n.min(limit))
                }
            })
            .unwrap_or(v.clone())
    } else {
        v.clone()
    }
}

pub(crate) fn apply_max_tokens_cap_to_fields(body: &mut Value, cap: Option<u64>, fields: &[&str]) {
    for field in fields {
        if let Some(v) = body.get(*field).cloned() {
            body[*field] = cap_token_value(&v, cap);
        }
    }
}

/// Cap `reasoning.effort` values most models don't support (`xhigh` → `high`),
/// unless the model's snapshot publishes the level; unknown models stay clamped.
pub(crate) fn cap_reasoning_effort(body: &mut Value) {
    let xhigh_published = body
        .get("model")
        .and_then(|m| m.as_str())
        .and_then(crate::services::model_metadata::snapshot_limits)
        .is_some_and(|l| l.reasoning_efforts.iter().any(|e| e == "xhigh"));
    if xhigh_published {
        return;
    }
    if let Some(effort) = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|e| e.as_str())
    {
        if effort.eq_ignore_ascii_case("xhigh") {
            body["reasoning"]["effort"] = json!("high");
        }
    } else if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str())
        && effort.eq_ignore_ascii_case("xhigh")
    {
        body["reasoning_effort"] = json!("high");
    }
}

/// Ensure every text-bearing content part in `input` messages has a `text` field.
///
/// The Responses API rejects `output_text` and `input_text` parts that are
/// missing `text`.  Codex CLI can echo back content parts from a previous
/// response where `text` was absent or null; this guard adds an empty string
/// so the upstream API accepts the request.
pub(crate) fn sanitize_input_content(body: &mut Value) {
    let Some(input) = body.get_mut("input").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for item in input.iter_mut() {
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let Some(parts) = item.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for part in parts.iter_mut() {
            let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match part_type {
                "output_text" | "input_text" | ""
                    if !part.get("text").is_some_and(|t| t.is_string()) =>
                {
                    part["text"] = json!("");
                }
                _ => {}
            }
        }
    }
}

/// Converts an OpenAI Responses API request body to Chat Completions format.
///
/// Handles all input item types:
/// - `message` → role/content message
/// - `function_call` → assistant message with tool_calls
/// - `function_call_output` → tool message
///
/// Also converts tool format (Responses API has no `function` wrapper;
/// Chat Completions requires `{type, function: {name, description, parameters}}`).
pub fn convert_responses_to_chat_request(
    body: &Value,
    config: &ResponsesToChatConversionConfig,
) -> Value {
    let mut messages: Vec<Value> = vec![];

    // System message from "instructions" field
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str())
        && !instructions.is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }

    // A DeepSeek thinking-mode turn arrives split across Responses items
    // (reasoning + message + one function_call per parallel call); re-merge
    // them into ONE Chat assistant message, else strict upstreams reject the
    // split (tool results detached from their call, reasoning_content dropped).
    // `current_assistant` indexes the open turn; non-assistant items close it.
    let mut pending_reasoning: String = String::new();
    let mut current_assistant: Option<usize> = None;

    // Convert "input" array items
    if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("reasoning") => {
                    let text = extract_reasoning_text(item);
                    if !text.is_empty() {
                        if pending_reasoning.is_empty() {
                            pending_reasoning = text;
                        } else {
                            pending_reasoning.push('\n');
                            pending_reasoning.push_str(&text);
                        }
                    }
                }
                Some("message") => {
                    // Only valid chat roles; Responses "developer" → "system".
                    let role = match item.get("role").and_then(|v| v.as_str()) {
                        Some("developer") => "system",
                        Some(r @ ("system" | "user" | "assistant" | "tool")) => r,
                        _ => "user",
                    };
                    // Vision/file inputs (input_image, input_file) must be
                    // preserved when bridging to Chat Completions; falling back
                    // to text-only here silently dropped them. The helper
                    // collapses to a string when no non-text part is present so
                    // the wire format is unchanged for the common case.
                    let content = convert_responses_content_to_chat(item.get("content"));
                    if role == "assistant" {
                        // Fold text emitted alongside tool calls into the open
                        // turn instead of splitting it into its own message.
                        let idx = open_assistant_turn(&mut messages, &mut current_assistant);
                        fold_assistant_content(&mut messages[idx]["content"], content);
                        flush_reasoning_into(
                            &mut messages[idx],
                            &mut pending_reasoning,
                            item,
                            config.requires_reasoning_content,
                        );
                    } else {
                        current_assistant = None;
                        messages.push(json!({"role": role, "content": content}));
                    }
                }
                Some(item_type @ ("function_call" | "custom_tool_call")) => {
                    // Use call_id as the Chat Completions tool_calls[].id so it matches
                    // the corresponding *_call_output.call_id → tool_call_id.
                    // Fall back to id only if call_id is absent.
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("call_0");
                    let raw_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    // A call into a non-default namespace went to the model
                    // under its qualified `namespace__tool` name; re-qualify
                    // history the same way so the model recognizes its own
                    // earlier calls. `name` already carrying the namespace
                    // prefix passes through unchanged.
                    let name = match item.get("namespace").and_then(|v| v.as_str()) {
                        Some(namespace)
                            if !namespace.is_empty()
                                && namespace != DEFAULT_FUNCTIONS_NAMESPACE =>
                        {
                            let prefix = qualified_namespace_tool_name(namespace, "");
                            if raw_name.starts_with(&prefix) {
                                raw_name.to_string()
                            } else {
                                qualified_namespace_tool_name(namespace, raw_name)
                            }
                        }
                        _ => raw_name.to_string(),
                    };
                    // custom_tool_call input is freeform; mirror the {"input": …} wrapping.
                    let arguments = if item_type == "custom_tool_call" {
                        json!({"input": item.get("input").and_then(|v| v.as_str()).unwrap_or("")})
                            .to_string()
                    } else {
                        item.get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}")
                            .to_string()
                    };
                    let tool_call = json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                    // Append to the open turn so parallel calls and post-narration calls share one message.
                    let idx = open_assistant_turn(&mut messages, &mut current_assistant);
                    let msg = &mut messages[idx];
                    match msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                        Some(arr) => arr.push(tool_call),
                        None => msg["tool_calls"] = json!([tool_call]),
                    }
                    flush_reasoning_into(
                        msg,
                        &mut pending_reasoning,
                        item,
                        config.requires_reasoning_content,
                    );
                }
                Some("function_call_output" | "custom_tool_call_output") => {
                    current_assistant = None;
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    // `output` may be a plain string or an array of content
                    // parts; collapsing non-strings to "" loses tool results.
                    let output = match item.get("output") {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Array(parts)) => {
                            let text = parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("\n");
                            if text.is_empty() {
                                serde_json::to_string(parts).unwrap_or_default()
                            } else {
                                text
                            }
                        }
                        Some(other) if !other.is_null() => other.to_string(),
                        _ => String::new(),
                    };
                    messages
                        .push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
                }
                None => {
                    // Simple string input
                    if let Some(s) = item.as_str() {
                        current_assistant = None;
                        messages.push(json!({"role": "user", "content": s}));
                    }
                }
                _ => {}
            }
        }
    } else if let Some(s) = body.get("input").and_then(|v| v.as_str()) {
        // String-form `input` — the whole prompt as one user message.
        if !s.is_empty() {
            messages.push(json!({"role": "user", "content": s}));
        }
    }

    let tools: Vec<Value> = responses_request_tools(body)
        .iter()
        .filter_map(convert_responses_tool_to_chat)
        .collect();

    // Apply model name transform (e.g. openai/ prefix for OpenRouter)
    // Skip transform when using Copilot — model names pass through unchanged
    // If actual_model is set, use that (it was set by environment injector)
    // No catalog handle on this sync path → host-based transform only.
    let selected_model = select_model_for_provider_attempt(
        None,
        &config.target_base_url,
        body.get("model").and_then(|v| v.as_str()),
        config.actual_model.as_deref(),
        config.target_protocol,
    );
    let model = if !config.is_copilot {
        if config.target_protocol == ProviderProtocol::Openai {
            Value::String(super::responses_to_chat_router::transform_model_str(
                &selected_model,
                &config.target_base_url,
                config.model_prefix.as_deref(),
            ))
        } else {
            Value::String(selected_model)
        }
    } else {
        Value::String(selected_model)
    };

    let mut chat = json!({
        "model": model,
        "messages": messages,
        "stream": false,  // request non-streaming for simpler response handling
    });

    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
    }
    if let Some(v) = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        chat["max_tokens"] = cap_token_value(v, config.max_tokens_cap);
    }
    // Dropped for models that reject sampling params (o-series etc.) —
    // forwarding them turns into upstream 400s.
    let rejects_sampling = chat["model"]
        .as_str()
        .is_some_and(crate::services::model_metadata::rejects_temperature);
    if !rejects_sampling {
        for field in ["temperature", "top_p"] {
            if let Some(v) = body.get(field) {
                chat[field] = v.clone();
            }
        }
    }

    // Copy reasoning fields
    if let Some(reasoning) = body.get("reasoning").and_then(|r| r.as_object()) {
        if let Some(effort) = reasoning.get("effort").and_then(|e| e.as_str()) {
            chat["reasoning_effort"] = json!(effort);
        }
    } else if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str()) {
        chat["reasoning_effort"] = json!(effort);
    }
    cap_reasoning_effort(&mut chat);

    // tool_choice: the forced-function shape differs ({type:"function",name}
    // vs {type:"function",function:{name}}); hosted-tool choices drop with
    // their tools. Both fields gated on surviving tools — upstreams 400 on
    // tool_choice / parallel_tool_calls without tools.
    if chat.get("tools").is_some() {
        match body.get("tool_choice") {
            Some(Value::String(s)) if matches!(s.as_str(), "auto" | "none" | "required") => {
                chat["tool_choice"] = json!(s);
            }
            Some(tc) if tc.get("type").and_then(|t| t.as_str()) == Some("function") => {
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    let name = match tc.get("namespace").and_then(|n| n.as_str()) {
                        Some(namespace)
                            if !namespace.is_empty()
                                && namespace != DEFAULT_FUNCTIONS_NAMESPACE =>
                        {
                            qualified_namespace_tool_name(namespace, name)
                        }
                        _ => name.to_string(),
                    };
                    chat["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
        if let Some(ptc) = body.get("parallel_tool_calls") {
            chat["parallel_tool_calls"] = ptc.clone();
        }
    }
    // Responses structured output lives in `text.format`; Chat providers
    // expect `response_format`.
    if let Some(format) = body.get("text").and_then(|t| t.get("format")) {
        match format.get("type").and_then(|t| t.as_str()) {
            Some("json_schema") => {
                let mut js = json!({});
                for field in ["name", "schema", "strict"] {
                    if let Some(v) = format.get(field) {
                        js[field] = v.clone();
                    }
                }
                chat["response_format"] = json!({"type": "json_schema", "json_schema": js});
            }
            Some("json_object") => {
                chat["response_format"] = json!({"type": "json_object"});
            }
            _ => {}
        }
    }

    chat
}

/// Copies `reasoning_content` from a source item onto a Chat Completions message.
/// Falls back to a single-space sentinel when the provider requires a non-empty value.
fn attach_reasoning_content(msg: &mut Value, source: &Value, requires: bool) {
    let rc = source
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| requires.then(|| " ".to_string()));
    if let Some(rc) = rc {
        msg["reasoning_content"] = json!(rc);
    }
}

/// Returns the index of the open assistant turn, opening a fresh assistant
/// message (null content) if none is active.
fn open_assistant_turn(messages: &mut Vec<Value>, current: &mut Option<usize>) -> usize {
    if let Some(idx) = *current {
        return idx;
    }
    messages.push(json!({"role": "assistant", "content": null}));
    let idx = messages.len() - 1;
    *current = Some(idx);
    idx
}

/// Drains buffered standalone-`reasoning` text onto an open assistant turn.
/// Appends to reasoning already collected for the turn (so a reasoning item
/// arriving mid-turn isn't carried to a later message) and upgrades a
/// single-space sentinel to real text. With nothing buffered, falls back to
/// the item's own non-standard `reasoning_content` field (or the sentinel the
/// provider requires).
fn flush_reasoning_into(msg: &mut Value, pending: &mut String, source: &Value, requires: bool) {
    if pending.is_empty() {
        if msg.get("reasoning_content").is_none() {
            attach_reasoning_content(msg, source, requires);
        }
        return;
    }
    let rc = std::mem::take(pending);
    match msg.get("reasoning_content").and_then(|v| v.as_str()) {
        Some(existing) if !existing.is_empty() && existing != " " => {
            msg["reasoning_content"] = json!(format!("{existing}\n{rc}"));
        }
        _ => msg["reasoning_content"] = json!(rc),
    }
}

/// Folds assistant text into an open turn's `content`: two strings append with
/// a newline, and a null/empty existing value adopts the addition. Assistant
/// turns here are text-only, so the exotic cases where either side is a
/// multimodal array keep the existing value and drop the addition.
fn fold_assistant_content(existing: &mut Value, addition: Value) {
    match existing {
        Value::String(s) if !s.is_empty() => {
            if let Some(add) = addition.as_str().filter(|a| !a.is_empty()) {
                *existing = Value::String(format!("{s}\n{add}"));
            }
        }
        Value::Null | Value::String(_) => *existing = addition,
        _ => {}
    }
}

/// Extract reasoning text from a standard Responses-API `type:"reasoning"` item.
/// Canonical shape is `summary: [{type:"summary_text", text}]`; some upstreams
/// place the trace in `content[*].text` or a bare `text` field, so accept both.
/// `encrypted_content` is opaque and provider-specific — skip it.
fn extract_reasoning_text(item: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    let collect = |arr: &Value, out: &mut Vec<String>| {
        if let Some(items) = arr.as_array() {
            for part in items {
                if let Some(s) = part.as_str() {
                    if !s.is_empty() {
                        out.push(s.to_string());
                    }
                } else if let Some(text) = part.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(text.to_string());
                } else if let Some(text) = part.get("reasoning").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(text.to_string());
                }
            }
        }
    };
    if let Some(summary) = item.get("summary") {
        collect(summary, &mut parts);
    }
    if let Some(content) = item.get("content") {
        collect(content, &mut parts);
    }
    if let Some(text) = item.get("text").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        parts.push(text.to_string());
    }
    parts.join("\n")
}

/// Convert a Responses-API content value (string or array of `input_text` /
/// `input_image` / `input_file` parts, etc.) into a Chat Completions content
/// value. Returns a `String` when every part is text (preserves the existing
/// wire format), and an array of `{type: ...}` content parts when any
/// multimodal part is present.
///
/// Recognised Responses-API parts:
/// - `input_text` / bare text → `{type: "text", text}`
/// - `input_image` with `image_url` (string or {url}) → `{type: "image_url",
///   image_url: {url, detail?}}`. Both http(s) URLs and `data:` URIs are
///   passed through unchanged — Chat Completions accepts both.
/// - `input_file` → inlined as `{type: "text", text: "[attached file: <name>]"}`
///   so the model still gets a turn-shaped reference instead of a silent drop.
///   (Chat Completions has no native file part. Plan default — see review notes.)
pub fn convert_responses_content_to_chat(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(responses_content_part_to_chat_part)
                .collect();
            if converted
                .iter()
                .all(|p| p.get("type").and_then(|v| v.as_str()) == Some("text"))
            {
                let joined = converted
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                Value::String(joined)
            } else {
                Value::Array(converted)
            }
        }
        Some(Value::Object(obj)) => Value::String(
            obj.get("text")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string(),
        ),
        _ => Value::String(String::new()),
    }
}

fn responses_content_part_to_chat_part(part: &Value) -> Option<Value> {
    if let Some(s) = part.as_str() {
        return Some(json!({"type": "text", "text": s}));
    }
    let part_type = part.get("type").and_then(|v| v.as_str());
    match part_type {
        // Most Responses-API text variants funnel through `text`/`content`.
        Some("input_text") | Some("text") | Some("output_text") | None => part
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| part.get("content").and_then(|v| v.as_str()))
            .map(|t| json!({"type": "text", "text": t})),
        Some("input_image") => {
            // Responses API: image_url is a string OR { url, detail? }.
            let (url, detail) = match part.get("image_url") {
                Some(Value::String(s)) => (Some(s.as_str()), None),
                Some(Value::Object(o)) => (
                    o.get("url").and_then(|v| v.as_str()),
                    o.get("detail").cloned(),
                ),
                _ => (None, None),
            };
            url.map(|u| {
                let mut iu = serde_json::Map::new();
                iu.insert("url".to_string(), Value::String(u.to_string()));
                if let Some(d) = detail {
                    iu.insert("detail".to_string(), d);
                }
                json!({"type": "image_url", "image_url": Value::Object(iu)})
            })
        }
        Some("input_file") => {
            // Chat Completions has no native file content part, so inline a
            // text reference. Default per fix-plan; can be revisited.
            let name = part
                .get("filename")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("name").and_then(|v| v.as_str()))
                .unwrap_or("file");
            Some(json!({"type": "text", "text": format!("[attached file: {name}]")}))
        }
        _ => part
            .get("text")
            .and_then(|v| v.as_str())
            .map(|t| json!({"type": "text", "text": t})),
    }
}

/// Extracts text from a content value (handles string, array of content parts)
pub fn extract_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                _ => p
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| p.get("content").and_then(|v| v.as_str()))
                    .map(String::from),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Object(obj)) => obj
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("content").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Top-level `tools` plus tools from `additional_tools` input items (codex
/// ≥0.143 sol), with `namespace` groups (codex ≥0.147, e.g. MCP servers like
/// `mcp__chrome_devtools`) flattened: tools in the default `functions`
/// namespace keep their plain names; tools in any other namespace are
/// renamed to their qualified `namespace__tool` form so the chat model can
/// call them and the response path can map the call back onto the namespace
/// (see `collect_namespace_tool_names`).
fn responses_request_tools(body: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    for_each_responses_tool(body, |tool, namespace| {
        let mut tool = tool.clone();
        if let Some(namespace) = namespace
            && let Some(name) = tool.get("name").and_then(|v| v.as_str())
        {
            tool["name"] = json!(qualified_namespace_tool_name(namespace, name));
        }
        out.push(tool);
    });
    out
}

/// Maps every qualified chat tool name produced by `responses_request_tools`
/// back to its (`namespace`, original tool name) pair, so a model call to
/// e.g. `mcp__chrome_devtools__navigate_page` can be re-emitted as a
/// Responses `function_call` with `namespace: "mcp__chrome_devtools"` and
/// `name: "navigate_page"` — the exact pair codex's tool registry keys on.
pub fn collect_namespace_tool_names(body: &Value) -> ToolNamespaceMap {
    let mut map = ToolNamespaceMap::new();
    for_each_responses_tool(body, |tool, namespace| {
        if let Some(namespace) = namespace
            && let Some(name) = tool.get("name").and_then(|v| v.as_str())
        {
            map.insert(
                qualified_namespace_tool_name(namespace, name),
                (namespace.to_string(), name.to_string()),
            );
        }
    });
    map
}

/// Names of `custom` (freeform-input) tools, e.g. codex's sol `exec`. The
/// emitter must answer these with `custom_tool_call` items — codex rejects a
/// `function_call` for a custom tool.
pub fn collect_custom_tool_names(body: &Value) -> HashSet<String> {
    responses_request_tools(body)
        .into_iter()
        .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("custom"))
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(str::to_string))
        .collect()
}

/// `function` maps directly; `custom` wraps as a one-string-arg (`input`)
/// function; hosted tools drop (no chat equivalent). `namespace` groups
/// never reach here — `responses_request_tools` already flattened them.
fn convert_responses_tool_to_chat(tool: &Value) -> Option<Value> {
    match tool.get("type").and_then(|v| v.as_str()) {
        Some("function") => Some(convert_tool_to_chat_format(tool)),
        Some("custom") => {
            let name = tool.get("name").and_then(|v| v.as_str())?;
            let mut description = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            description.push_str("\n\nPass the raw input text as the single `input` argument.");
            Some(json!({"type": "function", "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "The raw tool input text."}},
                    "required": ["input"]
                }
            }}))
        }
        _ => None,
    }
}

/// Converts a tool from Responses API format to Chat Completions format.
///
/// Responses API: `{type, name, description, parameters}`
/// Chat Completions: `{type, function: {name, description, parameters}}`
pub fn convert_tool_to_chat_format(tool: &Value) -> Value {
    // Already in Chat Completions format (has "function" wrapper)
    if tool.get("function").is_some() {
        return tool.clone();
    }
    let mut func = serde_json::Map::new();
    for field in ["name", "description", "parameters", "strict"] {
        if let Some(v) = tool.get(field) {
            func.insert(field.to_string(), v.clone());
        }
    }
    json!({"type": "function", "function": Value::Object(func)})
}

/// Parses a provider HTTP response body as either a JSON chat completion
/// (stream:false) or an SSE chat completion stream (stream:true).
/// Returns a unified non-streaming chat completion JSON.
pub fn parse_provider_response(text: &str) -> anyhow::Result<Value> {
    // Try JSON first (non-streaming response)
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return Ok(v);
    }
    // Fallback: provider returned SSE despite stream:false
    Ok(accumulate_chat_sse(text))
}

/// Reads an SSE chat completions stream and returns a synthesized non-streaming response.
pub fn accumulate_chat_sse(text: &str) -> Value {
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls_acc: Vec<StreamedToolCall> = Vec::new();
    let mut finish_reason = String::from("stop");

    for line in text.lines() {
        if let Some(data) = http_utils::sse_data_payload(line) {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                let choice = &chunk["choices"][0];
                let delta = &choice["delta"];

                if let Some(c) = delta["content"].as_str() {
                    content.push_str(c);
                }
                if let Some(rc) = delta["reasoning_content"].as_str() {
                    reasoning_content.push_str(rc);
                }
                if let Some(tcs) = delta["tool_calls"].as_array() {
                    accumulate_tool_call_deltas(tcs, &mut tool_calls_acc);
                }
                if let Some(fr) = choice["finish_reason"].as_str()
                    && !fr.is_empty()
                {
                    finish_reason = fr.to_string();
                }
            }
        }
    }

    if !tool_calls_acc.is_empty() {
        let tcs: Vec<Value> = tool_calls_acc
            .iter()
            .enumerate()
            .map(|(i, call)| {
                json!({
                    "id": if call.id.is_empty() { format!("call_{}", i) } else { call.id.clone() },
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })
            })
            .collect();
        // Keep accumulated assistant text alongside the tool calls — dropping
        // it loses the model's preamble ("Let me check that file.").
        let content_value = if content.is_empty() {
            Value::Null
        } else {
            json!(content)
        };
        let mut msg = json!({"role": "assistant", "content": content_value, "tool_calls": tcs});
        if !reasoning_content.is_empty() {
            msg["reasoning_content"] = json!(reasoning_content);
        }
        // Preserve "length": masking a truncation as tool_calls would hide
        // the incomplete status (and let a half-emitted tool call execute).
        let fr = if finish_reason == "length" {
            "length"
        } else {
            "tool_calls"
        };
        json!({"choices": [{"message": msg, "finish_reason": fr}]})
    } else {
        let mut msg = json!({"role": "assistant", "content": content});
        if !reasoning_content.is_empty() {
            msg["reasoning_content"] = json!(reasoning_content);
        }
        json!({"choices": [{"message": msg, "finish_reason": finish_reason}]})
    }
}

/// Converts a Chat Completions non-streaming response to Responses API SSE events.
///
/// Codex CLI parses these SSE events to display output and handle tool calls.
/// Handles both text responses and tool call responses.
///
/// Key correctness requirements from the OpenAI Responses API spec:
/// - `object` must be "response" (not "realtime.response")
/// - All sub-events must include `response_id`
/// - Function call items need a `call_id` (= Chat Completions tc.id) separate
///   from `id` (a fresh item identifier); Codex puts `call_id` in the
///   follow-up `function_call_output.call_id` field
pub fn convert_chat_response_to_responses_sse(
    chat: &Value,
    requires_reasoning_content: bool,
    original_model: &str,
    custom_tools: &HashSet<String>,
    tool_namespaces: &ToolNamespaceMap,
) -> String {
    let (content, tool_calls, reasoning_content) = extract_chat_response_payload(chat);

    // Replay the buffered payload as a single synthetic delta chunk through
    // the streaming converter, so buffered and streamed responses come from
    // one emitter and can't drift apart again.
    let mut delta = json!({});
    if !reasoning_content.is_empty() {
        delta["reasoning_content"] = json!(reasoning_content);
    }
    if !content.is_empty() {
        delta["content"] = json!(content);
    }
    if !tool_calls.is_empty() {
        // Re-index and default-fill: without a distinct `index` per call the
        // streaming ingester would collapse them all into slot 0.
        let tcs: Vec<Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(i, tc)| {
                json!({
                    "index": i,
                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0"),
                    "function": {
                        "name": tc["function"]["name"].as_str().unwrap_or(""),
                        "arguments": tc["function"]["arguments"].as_str().unwrap_or("{}")
                    }
                })
            })
            .collect();
        delta["tool_calls"] = json!(tcs);
    }

    let mut chunk = json!({"choices": [{"delta": delta}]});
    if let Some(fr) = chat["choices"][0]["finish_reason"].as_str() {
        chunk["choices"][0]["finish_reason"] = json!(fr);
    }
    if let Some(usage) = chat.get("usage") {
        chunk["usage"] = usage.clone();
    }

    let mut converter = ResponsesStreamConverter::new(original_model, requires_reasoning_content)
        .with_custom_tools(custom_tools.clone())
        .with_tool_namespaces(tool_namespaces.clone());
    let mut sse = String::new();
    converter.ensure_created(&mut sse);
    converter.process_chunk(&chunk, &mut sse);
    sse.push_str(&converter.finish());
    sse
}

/// Unwraps `{"input": …}` back to raw text; raw (non-JSON) args pass through.
fn custom_input_from_args(args: &str) -> String {
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(map)) => match map.get("input") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => args.to_string(),
        },
        Ok(Value::String(s)) => s,
        _ => args.to_string(),
    }
}

/// Incrementally converts a streaming Chat Completions response (SSE chunks with
/// `delta.content` / `delta.reasoning_content` / `delta.tool_calls`) into
/// Responses API SSE events, so output reaches Codex as it's produced instead of
/// arriving in one blob after the turn finishes.
///
/// This is the single chat→Responses emitter: the buffered
/// `convert_chat_response_to_responses_sse` replays its payload through this
/// converter, and serve's /v1/responses streaming path drives it directly.
pub struct ResponsesStreamConverter {
    buf: SseLineBuffer,
    resp_id: String,
    created_at: u64,
    codex_model: String,
    requires_reasoning_content: bool,
    created_emitted: bool,
    finished: bool,
    next_output_index: usize,
    reasoning: Option<StreamItem>,
    message: Option<StreamItem>,
    tools: Vec<StreamToolCall>,
    usage: Option<Value>,
    /// Upstream sent finish_reason "length" — surface status "incomplete".
    truncated: bool,
    /// `custom`-tool names — calls go out as `custom_tool_call` items.
    custom_tools: HashSet<String>,
    /// Qualified chat names of namespace-flattened tools → (`namespace`,
    /// original name); the emitted `function_call` / `custom_tool_call`
    /// items must carry both fields for codex's tool router to match.
    tool_namespaces: ToolNamespaceMap,
}

struct StreamItem {
    id: String,
    output_index: usize,
    text: String,
}

struct StreamToolCall {
    chat_index: u64,
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    args: String,
    /// Buffered; emitted as one `custom_tool_call` item at finish.
    is_custom: bool,
    /// `response.output_item.added` already emitted.
    announced: bool,
}

impl ResponsesStreamConverter {
    pub fn new(original_model: &str, requires_reasoning_content: bool) -> Self {
        Self {
            buf: SseLineBuffer::new(),
            resp_id: gen_id("resp"),
            created_at: current_unix_ts(),
            codex_model: map_model_for_codex_cli(original_model),
            requires_reasoning_content,
            created_emitted: false,
            finished: false,
            next_output_index: 0,
            reasoning: None,
            message: None,
            tools: Vec::new(),
            usage: None,
            truncated: false,
            custom_tools: HashSet::new(),
            tool_namespaces: ToolNamespaceMap::new(),
        }
    }

    /// See `collect_custom_tool_names`.
    pub fn with_custom_tools(mut self, names: HashSet<String>) -> Self {
        self.custom_tools = names;
        self
    }

    /// See `collect_namespace_tool_names`.
    pub fn with_tool_namespaces(mut self, map: ToolNamespaceMap) -> Self {
        self.tool_namespaces = map;
        self
    }

    /// Splits a qualified chat tool name back into the (name, namespace)
    /// pair codex's tool registry keys on; unqualified names pass through
    /// with no namespace.
    fn resolve_tool_call_name(&self, name: &str) -> (String, Option<String>) {
        match self.tool_namespaces.get(name) {
            Some((namespace, original)) => (original.clone(), Some(namespace.clone())),
            None => (name.to_string(), None),
        }
    }

    /// Feeds a network chunk of the upstream SSE body, returning any Responses API
    /// SSE to forward to the client. Buffers partial lines across calls.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        let mut out = String::new();
        self.ensure_created(&mut out);
        for line in self.buf.push_chunk(chunk)? {
            self.process_line(&line, &mut out);
        }
        Ok(out)
    }

    /// Flushes any buffered trailing line and emits the closing `.done` events
    /// plus `response.completed`.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        self.ensure_created(&mut out);
        if let Some(tail) = self.buf.take_tail() {
            self.process_line(&tail, &mut out);
        }
        if self.finished {
            return out;
        }
        self.finished = true;

        // Match the buffered converter: when neither text nor tool calls were
        // produced, still emit an (empty) message item so Codex sees a turn.
        if self.message.is_none() && self.tools.is_empty() {
            self.start_message(&mut out);
        }

        let reasoning_text = self.reasoning.as_ref().map(|r| r.text.clone());
        let reasoning_for_tool = match &reasoning_text {
            Some(t) if !t.is_empty() => t.clone(),
            _ if self.requires_reasoning_content => {
                let msg_text = self.message.as_ref().map(|m| m.text.as_str()).unwrap_or("");
                if msg_text.is_empty() {
                    " ".to_string()
                } else {
                    msg_text.to_string()
                }
            }
            _ => String::new(),
        };

        let mut output_items: Vec<(usize, Value)> = Vec::new();

        if let Some(reasoning) = &self.reasoning {
            let text = reasoning.text.clone();
            out.push_str(&sse_event(
                "response.reasoning_summary_text.done",
                &json!({
                    "type": "response.reasoning_summary_text.done",
                    "response_id": self.resp_id, "item_id": reasoning.id,
                    "output_index": reasoning.output_index, "summary_index": 0,
                    "text": text
                }),
            ));
            out.push_str(&sse_event(
                "response.reasoning_summary_part.done",
                &json!({
                    "type": "response.reasoning_summary_part.done",
                    "response_id": self.resp_id, "item_id": reasoning.id,
                    "output_index": reasoning.output_index, "summary_index": 0,
                    "part": {"type": "summary_text", "text": text}
                }),
            ));
            let item = json!({
                "id": reasoning.id, "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}]
            });
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": reasoning.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((reasoning.output_index, item));
        }

        if let Some(message) = &self.message {
            let text = message.text.clone();
            out.push_str(&sse_event(
                "response.output_text.done",
                &json!({
                    "type": "response.output_text.done",
                    "response_id": self.resp_id, "item_id": message.id,
                    "output_index": message.output_index, "content_index": 0, "text": text
                }),
            ));
            out.push_str(&sse_event(
                "response.content_part.done",
                &json!({
                    "type": "response.content_part.done",
                    "response_id": self.resp_id, "item_id": message.id,
                    "output_index": message.output_index, "content_index": 0,
                    "part": {"type": "output_text", "text": text}
                }),
            ));
            // Reasoning stays in the standalone reasoning item; a `reasoning`
            // part in `message.content` makes Codex.app reject the message.
            let content_parts =
                vec![json!({"type": "output_text", "text": text, "annotations": []})];
            let item = json!({
                "id": message.id, "type": "message", "status": "completed",
                "role": "assistant", "content": content_parts
            });
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": message.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((message.output_index, item));
        }

        for tool in &self.tools {
            if tool.is_custom {
                // item.done + response.completed suffice — codex needs no input.delta events.
                let (name, namespace) = self.resolve_tool_call_name(&tool.name);
                let mut item = json!({
                    "id": tool.item_id, "call_id": tool.call_id,
                    "type": "custom_tool_call", "status": "completed",
                    "name": name, "input": custom_input_from_args(&tool.args)
                });
                if let Some(namespace) = namespace {
                    item["namespace"] = json!(namespace);
                }
                out.push_str(&sse_event(
                    "response.output_item.done",
                    &json!({
                        "type": "response.output_item.done",
                        "response_id": self.resp_id, "output_index": tool.output_index,
                        "item": item.clone()
                    }),
                ));
                output_items.push((tool.output_index, item));
                continue;
            }
            if !tool.announced {
                // Deferred announcement whose name never arrived.
                let (name, namespace) = self.resolve_tool_call_name(&tool.name);
                let mut added = json!({
                    "id": tool.item_id, "call_id": tool.call_id,
                    "type": "function_call", "status": "in_progress",
                    "name": name, "arguments": ""
                });
                if let Some(namespace) = namespace {
                    added["namespace"] = json!(namespace);
                }
                out.push_str(&sse_event(
                    "response.output_item.added",
                    &json!({
                        "type": "response.output_item.added",
                        "response_id": self.resp_id, "output_index": tool.output_index,
                        "item": added
                    }),
                ));
            }
            out.push_str(&sse_event(
                "response.function_call_arguments.done",
                &json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": self.resp_id, "output_index": tool.output_index,
                    "item_id": tool.item_id, "arguments": tool.args
                }),
            ));
            let (name, namespace) = self.resolve_tool_call_name(&tool.name);
            let mut item = json!({
                "id": tool.item_id, "call_id": tool.call_id,
                "type": "function_call", "status": "completed",
                "name": name, "arguments": tool.args
            });
            if let Some(namespace) = namespace {
                item["namespace"] = json!(namespace);
            }
            if !reasoning_for_tool.is_empty() {
                item["reasoning_content"] = json!(reasoning_for_tool.clone());
            }
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": tool.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((tool.output_index, item));
        }

        output_items.sort_by_key(|(idx, _)| *idx);
        let output: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
        let mut response = json!({
            "id": self.resp_id, "object": "response",
            "model": self.codex_model,
            "created_at": self.created_at,
            "status": if self.truncated { "incomplete" } else { "completed" },
            "output": output
        });
        if self.truncated {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        if let Some(usage) = &self.usage {
            response["usage"] = usage.clone();
        }
        out.push_str(&sse_event(
            "response.completed",
            &json!({"type": "response.completed", "response": response}),
        ));
        out
    }

    fn ensure_created(&mut self, out: &mut String) {
        if self.created_emitted {
            return;
        }
        self.created_emitted = true;
        out.push_str(&sse_event(
            "response.created",
            &json!({
                "type": "response.created",
                "response": {
                    "id": self.resp_id, "object": "response",
                    "model": self.codex_model,
                    "created_at": self.created_at, "status": "in_progress", "output": []
                }
            }),
        ));
    }

    fn process_line(&mut self, line: &str, out: &mut String) {
        let Some(data) = http_utils::sse_data_payload(line) else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return;
        };
        self.process_chunk(&chunk, out);
    }

    fn process_chunk(&mut self, chunk: &Value, out: &mut String) {
        if chunk.get("usage").is_some_and(|u| !u.is_null())
            && let Some(usage) = chat_usage_to_responses_usage(chunk)
        {
            self.usage = Some(usage);
        }
        let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
            return;
        };
        for choice in choices {
            if choice.get("finish_reason").and_then(|v| v.as_str()) == Some("length") {
                self.truncated = true;
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str())
                && !reasoning.is_empty()
            {
                self.push_reasoning_delta(reasoning, out);
            }
            if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                && !content.is_empty()
            {
                self.push_content_delta(content, out);
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    self.push_tool_call_delta(tc, out);
                }
            }
        }
    }

    fn push_reasoning_delta(&mut self, delta: &str, out: &mut String) {
        if self.reasoning.is_none() {
            let id = gen_id("rs");
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            out.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": self.resp_id, "output_index": output_index,
                    "item": {"id": id, "type": "reasoning", "summary": []}
                }),
            ));
            out.push_str(&sse_event(
                "response.reasoning_summary_part.added",
                &json!({
                    "type": "response.reasoning_summary_part.added",
                    "response_id": self.resp_id, "item_id": id,
                    "output_index": output_index, "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""}
                }),
            ));
            self.reasoning = Some(StreamItem {
                id,
                output_index,
                text: String::new(),
            });
        }
        let Some(reasoning) = self.reasoning.as_mut() else {
            return;
        };
        reasoning.text.push_str(delta);
        out.push_str(&sse_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "response_id": self.resp_id, "item_id": reasoning.id,
                "output_index": reasoning.output_index, "summary_index": 0,
                "delta": delta
            }),
        ));
    }

    fn start_message(&mut self, out: &mut String) {
        if self.message.is_some() {
            return;
        }
        let id = gen_id("msg");
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        out.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": self.resp_id, "output_index": output_index,
                "item": {
                    "id": id, "type": "message",
                    "status": "in_progress", "role": "assistant", "content": []
                }
            }),
        ));
        out.push_str(&sse_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": self.resp_id, "item_id": id,
                "output_index": output_index, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        self.message = Some(StreamItem {
            id,
            output_index,
            text: String::new(),
        });
    }

    fn push_content_delta(&mut self, delta: &str, out: &mut String) {
        self.start_message(out);
        let Some(message) = self.message.as_mut() else {
            return;
        };
        message.text.push_str(delta);
        out.push_str(&sse_event(
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "response_id": self.resp_id, "item_id": message.id,
                "output_index": message.output_index, "content_index": 0, "delta": delta
            }),
        ));
    }

    fn push_tool_call_delta(&mut self, tc: &Value, out: &mut String) {
        let chat_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let name_fragment = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str());
        let id_fragment = tc.get("id").and_then(|v| v.as_str());
        let args_fragment = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str());

        let pos = self.tools.iter().position(|t| t.chat_index == chat_index);
        let pos = match pos {
            Some(p) => {
                // Late-arriving id/name fragments still update the slot.
                if let Some(id) = id_fragment.filter(|s| !s.is_empty()) {
                    self.tools[p].call_id = id.to_string();
                }
                if let Some(name) = name_fragment.filter(|s| !s.is_empty()) {
                    self.tools[p].name = name.to_string();
                    // Announced slots already went out as function_call.
                    if !self.tools[p].announced {
                        self.tools[p].is_custom = self.custom_tools.contains(name);
                    }
                }
                p
            }
            None => {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let item_id = gen_id("fc");
                let call_id = id_fragment.filter(|s| !s.is_empty()).unwrap_or("call_0");
                let name = name_fragment.unwrap_or("");
                let is_custom = !name.is_empty() && self.custom_tools.contains(name);
                self.tools.push(StreamToolCall {
                    chat_index,
                    output_index,
                    item_id,
                    call_id: call_id.to_string(),
                    name: name.to_string(),
                    args: String::new(),
                    is_custom,
                    announced: false,
                });
                self.tools.len() - 1
            }
        };

        if let Some(args) = args_fragment.filter(|s| !s.is_empty()) {
            self.tools[pos].args.push_str(args);
        }

        // Custom calls buffer until finish; with custom tools declared,
        // announcing waits for the name so the item type can't flip post-.added.
        let tool = &self.tools[pos];
        if tool.is_custom || (!self.custom_tools.is_empty() && tool.name.is_empty()) {
            return;
        }
        if !self.tools[pos].announced {
            self.tools[pos].announced = true;
            let (name, namespace) = self.resolve_tool_call_name(&self.tools[pos].name);
            let tool = &self.tools[pos];
            let mut added = json!({
                "id": tool.item_id, "call_id": tool.call_id,
                "type": "function_call", "status": "in_progress",
                "name": name, "arguments": ""
            });
            if let Some(namespace) = namespace {
                added["namespace"] = json!(namespace);
            }
            out.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": self.resp_id, "output_index": tool.output_index,
                    "item": added
                }),
            ));
            // Flush arguments buffered while the announcement was deferred.
            let tool = &self.tools[pos];
            if !tool.args.is_empty() {
                out.push_str(&sse_event(
                    "response.function_call_arguments.delta",
                    &json!({
                        "type": "response.function_call_arguments.delta",
                        "response_id": self.resp_id, "output_index": tool.output_index,
                        "item_id": tool.item_id, "delta": tool.args
                    }),
                ));
            }
            return;
        }

        if let Some(args) = args_fragment.filter(|s| !s.is_empty()) {
            let tool = &self.tools[pos];
            let item_id = tool.item_id.clone();
            let output_index = tool.output_index;
            out.push_str(&sse_event(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": self.resp_id, "output_index": output_index,
                    "item_id": item_id, "delta": args
                }),
            ));
        }
    }
}

fn chat_usage_to_responses_usage(chat: &Value) -> Option<Value> {
    let usage = chat.get("usage")?;

    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let total_tokens = usage.get("total_tokens").cloned().unwrap_or_else(|| {
        let input = input_tokens.as_u64().unwrap_or(0);
        let output = output_tokens.as_u64().unwrap_or(0);
        json!(input.saturating_add(output))
    });

    // Map OpenAI chat-completion's `prompt_tokens_details.cached_tokens` (and
    // Anthropic's `cache_read_input_tokens`) to the Responses API shape.
    // Some clients (recent OpenAI SDKs) crash on `usage.input_tokens_details.
    // cached_tokens` being absent, so emit a zeroed object even when the
    // upstream didn't return cache info.
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .or_else(|| usage.get("cache_read_input_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));

    let mut response_usage = json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
        "total_tokens": total_tokens
    });

    if let Some(value) = usage.get("cache_read_input_tokens").cloned() {
        response_usage["cache_read_input_tokens"] = value;
    }
    if let Some(value) = usage.get("cache_creation_input_tokens").cloned() {
        response_usage["cache_creation_input_tokens"] = value;
    }

    Some(response_usage)
}

/// Extracts assistant text, tool calls, and reasoning content from provider chat completion payloads.
/// Handles multi-choice payloads and common non-standard envelopes.
fn extract_chat_response_payload(chat: &Value) -> (String, Vec<Value>, String) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();

    if let Some(choices) = chat.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
            let text = extract_message_text(&message);
            if !text.is_empty() {
                text_parts.push(text);
            }
            // Extract reasoning_content if present (Moonshot, etc.)
            if let Some(reasoning) = message.get("reasoning_content").and_then(|r| r.as_str())
                && !reasoning.is_empty()
            {
                reasoning_parts.push(reasoning.to_string());
            }
            if let Some(tcs) = message.get("tool_calls").and_then(|t| t.as_array()) {
                tool_calls.extend(tcs.iter().cloned());
            }
        }
    }

    // Fallback: Responses API-style output payloads from some providers.
    if text_parts.is_empty() && tool_calls.is_empty() {
        let output_items = chat
            .get("output")
            .or_else(|| chat.get("response").and_then(|r| r.get("output")))
            .and_then(|v| v.as_array());

        if let Some(items) = output_items {
            for item in items {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("message") => {
                        let text = extract_content_text(item.get("content"));
                        if !text.is_empty() {
                            text_parts.push(text);
                        }
                    }
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()))
                            .unwrap_or("call_0");
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}");
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                            }
                        }));
                    }
                    Some("output_text") => {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Fallback envelopes seen from some OpenAI-compatible providers
    if text_parts.is_empty() {
        if let Some(text) = chat
            .get("result")
            .and_then(|r| r.get("response"))
            .and_then(|v| v.as_str())
        {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("response").and_then(|v| v.as_str()) {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("output_text").and_then(|v| v.as_str()) {
            text_parts.push(text.to_string());
        }
    }

    (
        text_parts.join("\n"),
        tool_calls,
        reasoning_parts.join("\n"),
    )
}

fn extract_message_text(message: &Value) -> String {
    extract_content_text(message.get("content"))
}

// =============================================================================
// CHAT COMPLETIONS → RESPONSES API CONVERSION
// =============================================================================

/// Converts an OpenAI Chat Completions request body to a Responses API request body.
/// Delegates to the typed converter in `openai_models` to avoid duplicating conversion logic.
/// Every caller re-streams to its own client, so the upstream request is always
/// non-streaming. `cache_control` is an Anthropic-only field the Anthropic→Chat
/// step preserves for caching-aware gateways; it has no place in the Responses
/// schema and strict backends (ChatGPT Codex) 400 on it.
pub fn try_convert_chat_to_responses_request(body: &Value) -> Result<Value> {
    let typed: OpenAIChatRequest = serde_json::from_value(body.clone())
        .context("failed to parse openai chat request for responses conversion")?;
    let mut resp = serde_json::to_value(convert_typed_chat_to_responses(&typed))
        .context("failed to serialize responses request")?;
    resp["stream"] = json!(false);
    strip_key_recursive(&mut resp, "cache_control");
    Ok(resp)
}

/// Infallible variant of [`try_convert_chat_to_responses_request`] for paths
/// that must always produce a body to forward.
pub fn convert_chat_to_responses_request(body: &Value) -> Value {
    try_convert_chat_to_responses_request(body)
        .unwrap_or_else(|_| json!({"model": "gpt-4o", "input": [], "stream": false}))
}

/// Removes every occurrence of `key` anywhere in the JSON tree.
fn strip_key_recursive(v: &mut Value, key: &str) {
    match v {
        Value::Object(map) => {
            map.remove(key);
            for val in map.values_mut() {
                strip_key_recursive(val, key);
            }
        }
        Value::Array(arr) => {
            for val in arr.iter_mut() {
                strip_key_recursive(val, key);
            }
        }
        _ => {}
    }
}

/// Converts a Responses API JSON response to Chat Completions format.
/// Delegates to the typed converter in `openai_models` to avoid duplicating conversion logic.
/// Accepts both a direct response object and the wrapped `{"response": ...}` shape.
pub fn try_convert_responses_json_to_chat(resp: &Value) -> Result<Value> {
    let inner = resp
        .get("response")
        .filter(|r| r.is_object())
        .unwrap_or(resp);
    let typed: ResponsesResponse =
        serde_json::from_value(inner.clone()).context("failed to parse responses API response")?;
    serde_json::to_value(convert_typed_responses_to_chat(&typed))
        .context("failed to serialize openai chat response")
}

/// Infallible variant of [`try_convert_responses_json_to_chat`] for paths that
/// must always produce a body to forward.
pub fn convert_responses_json_to_chat(resp: &Value) -> Value {
    try_convert_responses_json_to_chat(resp).unwrap_or_else(|_| json!({"choices": [], "usage": {}}))
}

/// Streaming inverse of [`ResponsesStreamConverter`]: feeds upstream Responses
/// API SSE and emits Chat Completions SSE, so a Chat Completions client (e.g.
/// omp) can drive a model that only accepts `/v1/responses` (gpt-5.x with
/// reasoning + tools) and still get incremental tokens. Buffers partial lines.
pub struct ResponsesToChatStreamConverter {
    buf: SseLineBuffer,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    role_emitted: bool,
    finished: bool,
    /// function_call item_id → its chat `tool_calls` array index.
    tool_index: std::collections::HashMap<String, u64>,
    next_tool_index: u64,
    finish_reason: &'static str,
    usage: Option<Value>,
}

impl ResponsesToChatStreamConverter {
    pub fn new(original_model: &str, include_usage: bool) -> Self {
        Self {
            buf: SseLineBuffer::new(),
            id: gen_id("chatcmpl"),
            created: current_unix_ts(),
            model: original_model.to_string(),
            include_usage,
            role_emitted: false,
            finished: false,
            tool_index: std::collections::HashMap::new(),
            next_tool_index: 0,
            finish_reason: "stop",
            usage: None,
        }
    }

    /// Feed a network chunk of the upstream Responses SSE; returns Chat
    /// Completions SSE to forward. Partial trailing lines buffer across calls.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        let mut out = String::new();
        for line in self.buf.push_chunk(chunk)? {
            self.process_line(&line, &mut out);
        }
        Ok(out)
    }

    /// Flush any buffered line, then emit the terminal `finish_reason` chunk, an
    /// optional usage-only chunk, and `data: [DONE]`.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if let Some(tail) = self.buf.take_tail() {
            self.process_line(&tail, &mut out);
        }
        if self.finished {
            return out;
        }
        self.finished = true;
        out.push_str(&self.chunk(json!({}), Some(self.finish_reason)));
        if self.include_usage
            && let Some(usage) = &self.usage
        {
            out.push_str(&data_line(&json!({
                "id": self.id, "object": "chat.completion.chunk",
                "created": self.created, "model": self.model,
                "choices": [], "usage": usage,
            })));
        }
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn process_line(&mut self, line: &str, out: &mut String) {
        let Some(data) = http_utils::sse_data_payload(line) else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(ev) = serde_json::from_str::<Value>(data) else {
            return;
        };
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "response.output_text.delta" => {
                if let Some(d) = ev
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    self.emit_delta(json!({ "content": d }), out);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(d) = ev
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    self.emit_delta(json!({ "reasoning_content": d }), out);
                }
            }
            "response.output_item.added" => {
                if let Some(item) = ev.get("item")
                    && item.get("type").and_then(|t| t.as_str()) == Some("function_call")
                {
                    self.start_tool_call(item, out);
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = ev.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(&idx), Some(d)) = (
                    self.tool_index.get(item_id),
                    ev.get("delta").and_then(|v| v.as_str()),
                ) {
                    self.emit_delta(
                        json!({ "tool_calls": [{ "index": idx, "function": { "arguments": d } }] }),
                        out,
                    );
                }
            }
            "response.completed" => {
                self.usage = ev
                    .get("response")
                    .and_then(|r| r.get("usage"))
                    .filter(|u| !u.is_null())
                    .map(responses_usage_to_chat_usage);
            }
            _ => {}
        }
    }

    fn start_tool_call(&mut self, item: &Value, out: &mut String) {
        let item_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let call_id = item
            .get("call_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| item.get("id").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
        let idx = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_index.insert(item_id, idx);
        self.finish_reason = "tool_calls";
        self.emit_delta(
            json!({ "tool_calls": [{
                "index": idx, "id": call_id, "type": "function",
                "function": { "name": name, "arguments": args }
            }] }),
            out,
        );
    }

    /// Emit one chat chunk carrying `delta`, prefixing the assistant role on the
    /// first chunk (mirrors OpenAI's stream).
    fn emit_delta(&mut self, mut delta: Value, out: &mut String) {
        if !self.role_emitted {
            self.role_emitted = true;
            if let Some(obj) = delta.as_object_mut() {
                obj.insert("role".to_string(), json!("assistant"));
            }
        }
        out.push_str(&self.chunk(delta, None));
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        data_line(&json!({
            "id": self.id, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        }))
    }
}

fn data_line(v: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(v).unwrap_or_default())
}

/// Map a Responses API `usage` object to the Chat Completions shape.
fn responses_usage_to_chat_usage(usage: &Value) -> Value {
    let num = |obj: &Value, key: &str| obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    let input = num(usage, "input_tokens");
    let output = num(usage, "output_tokens");
    let total = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| input.saturating_add(output));
    let cached = usage
        .get("input_tokens_details")
        .map(|d| num(d, "cached_tokens"))
        .unwrap_or(0);
    let reasoning = usage
        .get("output_tokens_details")
        .map(|d| num(d, "reasoning_tokens"))
        .unwrap_or(0);
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": total,
        "prompt_tokens_details": { "cached_tokens": cached },
        "completion_tokens_details": { "reasoning_tokens": reasoning },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_to_responses_strips_anthropic_cache_control() {
        let chat = json!({
            "model": "gpt-5.5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]
            }]
        });
        let resp = convert_chat_to_responses_request(&chat);
        let dumped = resp.to_string();
        assert!(!dumped.contains("cache_control"), "leaked: {dumped}");
        assert!(dumped.contains("hi"));
    }

    // ── is_responses_api_format ────────────────────────────────────────────────

    #[test]
    fn test_is_responses_api_format_detected() {
        assert!(is_responses_api_format(
            &json!({"input": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_chat_completions_not_detected() {
        assert!(!is_responses_api_format(
            &json!({"messages": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_has_both_not_detected() {
        // If both "input" and "messages" present, treat as Chat Completions
        assert!(!is_responses_api_format(&json!({
            "input": [],
            "messages": []
        })));
    }

    // ── extract_content_text ───────────────────────────────────────────────────

    #[test]
    fn test_extract_content_text_string() {
        assert_eq!(
            extract_content_text(Some(&json!("hello world"))),
            "hello world"
        );
    }

    #[test]
    fn test_extract_content_text_parts_array() {
        let content = json!([
            {"type": "input_text", "text": "list"},
            {"type": "input_text", "text": "files"}
        ]);
        assert_eq!(extract_content_text(Some(&content)), "list\nfiles");
    }

    #[test]
    fn test_extract_content_text_none() {
        assert_eq!(extract_content_text(None), "");
    }

    #[test]
    fn test_extract_content_text_empty_array() {
        assert_eq!(extract_content_text(Some(&json!([]))), "");
    }

    #[test]
    fn test_extract_content_text_null() {
        assert_eq!(extract_content_text(Some(&json!(null))), "");
    }

    // ── convert_tool_to_chat_format ────────────────────────────────────────────

    #[test]
    fn test_convert_tool_format_responses_api_to_chat() {
        let tool = json!({
            "type": "function",
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {}}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["type"], "function");
        assert_eq!(converted["function"]["name"], "shell");
        assert_eq!(converted["function"]["description"], "Run a shell command");
        assert!(converted.get("name").is_none()); // moved into "function" wrapper
    }

    #[test]
    fn test_convert_tool_format_already_chat_format() {
        let tool = json!({
            "type": "function",
            "function": {"name": "shell", "description": "..."}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["function"]["name"], "shell");
    }

    // ── convert_responses_to_chat_request ─────────────────────────────────────

    fn openai_router_config() -> ResponsesToChatConversionConfig {
        ResponsesToChatConversionConfig {
            target_base_url: "https://api.example.com/v1".to_string(),
            target_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
        }
    }

    #[test]
    fn test_convert_request_forwards_tool_choice_and_parallel() {
        let body = json!({
            "model": "gpt-4",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "tool_choice": "required",
            "parallel_tool_calls": false
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["parallel_tool_calls"], false);
    }

    #[test]
    fn test_convert_request_translates_forced_function_tool_choice() {
        let body = json!({
            "model": "gpt-4",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "shell"}
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        assert_eq!(
            chat["tool_choice"],
            json!({"type": "function", "function": {"name": "shell"}})
        );
    }

    #[test]
    fn test_convert_request_drops_tool_choice_when_no_tools_survive() {
        // Hosted-only tool list: every tool is filtered out, so forwarding
        // tool_choice / parallel_tool_calls would 400 on OpenAI upstreams.
        let body = json!({
            "model": "gpt-4",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": "required",
            "parallel_tool_calls": false
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        assert!(chat.get("tools").is_none());
        assert!(chat.get("tool_choice").is_none());
        assert!(chat.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn test_accumulate_chat_sse_preserves_length_over_tool_calls() {
        // A truncated tool call must keep finish_reason "length" so the
        // Responses conversion reports status "incomplete".
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"cm\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_convert_request_text_format_becomes_response_format() {
        let body = json!({
            "model": "gpt-4",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "text": {"format": {"type": "json_schema", "name": "out", "strict": true,
                                "schema": {"type": "object", "properties": {}}}}
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        assert_eq!(chat["response_format"]["type"], "json_schema");
        assert_eq!(chat["response_format"]["json_schema"]["name"], "out");
        assert_eq!(chat["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            chat["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn test_convert_request_simple_message() {
        let body = json!({
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://ai-gateway.vercel.sh/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );

        assert_eq!(chat["model"], "gpt-5.2-codex");
        assert_eq!(chat["stream"], false);
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "list files");
    }

    #[test]
    fn test_convert_request_instructions_become_system_message() {
        let body = json!({
            "model": "gpt-4",
            "instructions": "You are a helpful assistant.",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_convert_request_tool_call_items() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "list files"},
                {"type": "function_call", "id": "fc_item_1", "call_id": "call_abc", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_abc", "output": "file1.txt\nfile2.txt"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_abc");
        assert_eq!(msgs[2]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn test_convert_request_parallel_function_calls_coalesce_into_one_assistant_message() {
        // Codex emits parallel tool calls back as multiple consecutive
        // `function_call` items followed by their `function_call_output`s.
        // Chat Completions requires a single assistant message carrying all
        // parallel tool_calls, immediately followed by one tool message per
        // tool_call_id — otherwise OpenAI strict validators reject with
        // "An assistant message with 'tool_calls' must be followed by tool
        // messages responding to each 'tool_call_id'."
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "do two things"},
                {"type": "function_call", "call_id": "call_a", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call", "call_id": "call_b", "name": "shell", "arguments": "{\"cmd\":\"pwd\"}"},
                {"type": "function_call_output", "call_id": "call_a", "output": "files"},
                {"type": "function_call_output", "call_id": "call_b", "output": "/tmp"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4, "user + 1 assistant + 2 tool messages");
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        let tool_calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(
            tool_calls.len(),
            2,
            "parallel calls share one assistant msg"
        );
        assert_eq!(tool_calls[0]["id"], "call_a");
        assert_eq!(tool_calls[1]["id"], "call_b");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_a");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_b");
    }

    fn default_test_config() -> ResponsesToChatConversionConfig {
        ResponsesToChatConversionConfig {
            target_base_url: "https://example.com/v1".to_string(),
            target_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
        }
    }

    #[test]
    fn test_convert_request_reasoning_item_attaches_to_following_function_call() {
        // Codex emits `type:"reasoning"` items immediately before the
        // function_call they belong to. The converter must lift the summary
        // text onto the assistant tool_call message as `reasoning_content`,
        // or deepseek-thinking 400s with "must be passed back to the API".
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "step by step plan"}]
                },
                {"type": "function_call", "call_id": "call_x", "name": "shell", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_x");
        assert_eq!(msgs[0]["reasoning_content"], "step by step plan");
    }

    #[test]
    fn test_convert_request_strips_sampling_for_rejecting_models() {
        // o3 rejects temperature/top_p — forwarding them 400s upstream.
        let body = json!({
            "model": "o3",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        assert!(chat.get("temperature").is_none());
        assert!(chat.get("top_p").is_none());

        // Normal models keep them.
        let body = json!({
            "model": "deepseek-chat",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.2
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        assert_eq!(chat["temperature"], 0.2);
    }

    #[test]
    fn test_convert_request_reasoning_item_attaches_to_following_assistant_message() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoned"}]
                },
                {"type": "message", "role": "assistant", "content": "final answer"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], "final answer");
        assert_eq!(msgs[0]["reasoning_content"], "reasoned");
    }

    #[test]
    fn test_convert_request_multiple_reasoning_items_join_with_newline() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "first\nsecond");
    }

    #[test]
    fn test_convert_request_reasoning_only_attaches_to_first_following_turn() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "trace"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"type": "message", "role": "assistant", "content": "follow up"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        // First assistant turn carries reasoning; later assistant turn must not inherit it.
        assert_eq!(msgs[0]["reasoning_content"], "trace");
        assert_eq!(msgs[2]["role"], "assistant");
        assert!(msgs[2].get("reasoning_content").is_none());
    }

    #[test]
    fn test_convert_request_reasoning_falls_back_to_content_array() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "fallback"}]
                },
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "fallback");
    }

    #[test]
    fn test_convert_request_reasoning_attaches_to_first_of_parallel_tool_calls() {
        // Parallel function_calls coalesce into one assistant message. The
        // buffered reasoning attaches to the coalesced message via the first
        // function_call only — subsequent appends must not overwrite it.
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "shared trace"}]},
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        let tool_calls = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(msgs[0]["reasoning_content"], "shared trace");
    }

    #[test]
    fn test_convert_request_merges_assistant_text_alongside_tool_call() {
        // Codex replays a content+tool_calls turn as separate message and
        // function_call items; they must re-merge so strict upstreams accept it.
        let body = json!({
            "model": "deepseek-thinking",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                {"type": "message", "role": "assistant", "content": "Checking that file."},
                {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "one assistant turn + one tool result");
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], "Checking that file.");
        assert_eq!(msgs[0]["reasoning_content"], "why");
        let tool_calls = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "c1");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "c1");
    }

    #[test]
    fn test_convert_request_does_not_coalesce_tool_calls_across_user_boundary() {
        // A user message between two tool calls closes the first assistant turn;
        // the second call must start a fresh assistant message.
        let body = json!({
            "model": "deepseek-thinking",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "r"},
                {"type": "message", "role": "user", "content": "next"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        let assistants: Vec<_> = msgs.iter().filter(|m| m["role"] == "assistant").collect();
        assert_eq!(assistants.len(), 2);
        assert_eq!(assistants[0]["tool_calls"][0]["id"], "a");
        assert_eq!(assistants[1]["tool_calls"][0]["id"], "b");
    }

    #[test]
    fn test_convert_response_sse_emits_standard_reasoning_item_before_tool_calls() {
        // Codex CLI parses output items with typed structs; reasoning must
        // travel as a standalone `type:"reasoning"` item so it survives the
        // round-trip. The legacy `function_call.reasoning_content` field is
        // still emitted (Codex ignores it), driving aivo's own self-bridged
        // path when no separate reasoning item is read.
        let chat = json!({
            "id": "chatcmpl-r",
            "choices": [{
                "message": {
                    "content": null,
                    "reasoning_content": "let me think...",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "run", "arguments": "{}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });

        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "deepseek-reasoner",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );

        // The first `output_item.added` is a reasoning item with summary text,
        // at output_index 0. function_call comes after at output_index 1.
        assert!(
            sse.contains("\"type\":\"reasoning\""),
            "expected standalone reasoning item in SSE stream"
        );
        assert!(sse.contains("response.reasoning_summary_text.delta"));
        assert!(sse.contains("response.reasoning_summary_text.done"));
        assert!(sse.contains("\"text\":\"let me think...\""));

        // function_call follows at output_index 1
        assert!(
            sse.contains("\"output_index\":1") && sse.contains("\"type\":\"function_call\""),
            "function_call must appear at output_index 1 after the reasoning item"
        );
    }

    #[test]
    fn test_convert_response_sse_no_reasoning_item_when_empty() {
        let chat = json!({
            "id": "chatcmpl-nr",
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(
            !sse.contains("response.reasoning_summary_text"),
            "no reasoning events expected when message has no reasoning_content"
        );
        // Text message item still appears at output_index 0
        assert!(sse.contains("\"output_index\":0") && sse.contains("\"type\":\"message\""));
    }

    #[test]
    fn test_convert_response_sse_message_content_excludes_reasoning_part() {
        // A `reasoning` part inside message.content makes Codex.app drop the
        // whole message; reasoning must only ride the standalone reasoning item.
        let chat = json!({
            "choices": [{"message": {
                "content": "Once upon a time.",
                "reasoning_content": "the user wants a story"
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "deepseek-reasoner",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("response.reasoning_summary_text.delta"));

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        let message = output.iter().find(|i| i["type"] == "message").unwrap();
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "output_text");
        assert!(!content.iter().any(|p| p["type"] == "reasoning"));
    }

    #[test]
    fn test_convert_request_function_call_without_call_id_falls_back_to_id() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "function_call", "id": "call_legacy", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_legacy", "output": "ok"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_legacy");
        assert_eq!(msgs[1]["tool_call_id"], "call_legacy");
    }

    #[test]
    fn test_convert_request_filters_non_function_tools() {
        let body = json!({
            "model": "gpt-4",
            "input": [],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {}},
                {"type": "computer_use"},
                {"type": "web_search"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "shell");
    }

    #[test]
    fn test_convert_request_hoists_additional_tools_and_wraps_custom() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JavaScript code",
                     "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}},
                    {"type": "function", "name": "wait", "description": "Wait",
                     "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}}}},
                    {"type": "namespace", "name": "collaboration", "tools": [
                        {"type": "function", "name": "spawn_agent", "parameters": {}}
                    ]}
                ]},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());

        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3, "namespace leaf tools must survive");
        assert_eq!(tools[0]["function"]["name"], "exec");
        let params = &tools[0]["function"]["parameters"];
        assert_eq!(params["properties"]["input"]["type"], "string");
        assert_eq!(params["required"][0], "input");
        assert!(
            tools[0]["function"]["description"]
                .as_str()
                .unwrap()
                .starts_with("Run JavaScript code")
        );
        assert_eq!(tools[1]["function"]["name"], "wait");
        assert_eq!(tools[2]["function"]["name"], "collaboration__spawn_agent");

        // additional_tools must not leak into messages.
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");

        let custom = collect_custom_tool_names(&body);
        assert_eq!(custom.len(), 1);
        assert!(custom.contains("exec"));
    }

    #[test]
    fn test_convert_request_flattens_functions_namespace() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [
                    {"type": "namespace", "name": "functions", "description": "", "tools": [
                        {"type": "custom", "name": "exec", "description": "Run JavaScript code",
                         "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}},
                        {"type": "function", "name": "wait", "description": "Wait",
                         "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}}}}
                    ]},
                    {"type": "namespace", "name": "collaboration", "tools": [
                        {"type": "function", "name": "spawn_agent", "parameters": {}}
                    ]}
                ]},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());

        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3, "all namespace leaf tools must survive");
        assert_eq!(tools[0]["function"]["name"], "exec");
        assert_eq!(tools[1]["function"]["name"], "wait");
        assert_eq!(tools[2]["function"]["name"], "collaboration__spawn_agent");

        let custom = collect_custom_tool_names(&body);
        assert_eq!(custom.len(), 1);
        assert!(custom.contains("exec"));

        let namespaces = collect_namespace_tool_names(&body);
        assert_eq!(
            namespaces.get("collaboration__spawn_agent"),
            Some(&("collaboration".to_string(), "spawn_agent".to_string()))
        );
    }

    #[test]
    fn test_convert_request_qualifies_namespaced_history_and_tool_choice() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "function_call", "call_id": "call_1",
                 "namespace": "mcp.chrome", "name": "mcp.chrome_status", "arguments": "{}"}
            ],
            "tools": [{"type": "namespace", "name": "mcp.chrome", "tools": [
                {"type": "function", "name": "mcp.chrome_status", "parameters": {}}
            ]}],
            "tool_choice": {"type": "function", "namespace": "mcp.chrome", "name": "mcp.chrome_status"}
        });

        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        let qualified = "mcp_chrome__mcp_chrome_status";
        assert_eq!(chat["tools"][0]["function"]["name"], qualified);
        assert_eq!(
            chat["messages"][0]["tool_calls"][0]["function"]["name"],
            qualified
        );
        assert_eq!(chat["tool_choice"]["function"]["name"], qualified);
    }

    #[test]
    fn test_convert_request_custom_tool_call_history_round_trips() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "message", "role": "user", "content": "create the file"},
                {"type": "custom_tool_call", "status": "completed", "call_id": "call_1",
                 "name": "exec", "input": "await tools.exec_command({cmd: \"touch x\"})"},
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "Script completed"},
                    {"type": "input_text", "text": "Output:"}
                ]}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        let messages = chat["messages"].as_array().unwrap();

        assert_eq!(messages[1]["role"], "assistant");
        let tc = &messages[1]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "exec");
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            args["input"],
            "await tools.exec_command({cmd: \"touch x\"})"
        );

        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[2]["content"], "Script completed\nOutput:");
    }

    #[test]
    fn test_convert_request_developer_role_maps_to_system() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "message", "role": "developer", "content": "harness instructions"},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "harness instructions");
    }

    #[test]
    fn test_convert_response_sse_emits_custom_tool_call_item() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "exec",
                            "arguments": "{\"input\":\"await tools.exec_command({cmd: \\\"ls\\\"})\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let custom: HashSet<String> = ["exec".to_string()].into();
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-5.6-sol",
            &custom,
            &ToolNamespaceMap::new(),
        );

        assert!(sse.contains("\"type\":\"custom_tool_call\""), "{sse}");
        assert!(sse.contains("\"call_id\":\"call_9\""));
        assert!(
            sse.contains("\"input\":\"await tools.exec_command({cmd: \\\"ls\\\"})\""),
            "{sse}"
        );
        assert!(!sse.contains("\"type\":\"function_call\""), "{sse}");
        assert!(!sse.contains("function_call_arguments"), "{sse}");
        let completed = sse
            .lines()
            .filter_map(crate::services::http_utils::sse_data_payload)
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .find(|v| v["type"] == "response.completed")
            .unwrap();
        assert_eq!(
            completed["response"]["output"][0]["type"],
            "custom_tool_call"
        );
        assert_eq!(completed["response"]["output"][0]["name"], "exec");
    }

    #[test]
    fn test_convert_response_sse_custom_call_raw_args_pass_through() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_raw", "type": "function",
                    "function": {"name": "exec", "arguments": "await tools.wait()"}
                }]},
                "finish_reason": "tool_calls"
            }]
        });
        let custom: HashSet<String> = ["exec".to_string()].into();
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-5.6-sol",
            &custom,
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"input\":\"await tools.wait()\""), "{sse}");
    }

    #[test]
    fn test_streaming_converter_buffers_custom_calls_until_finish() {
        let custom: HashSet<String> = ["exec".to_string()].into();
        let mut converter =
            ResponsesStreamConverter::new("gpt-5.6-sol", false).with_custom_tools(custom);

        let chunk1 = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_s\",\"function\":{\"name\":\"exec\",\"arguments\":\"{\\\"inp\"}}]},\"finish_reason\":null}]}\n";
        let chunk2 = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ut\\\":\\\"1+1\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n";
        let mut out = converter.push_bytes(chunk1.as_bytes()).unwrap();
        out.push_str(&converter.push_bytes(chunk2.as_bytes()).unwrap());
        assert!(!out.contains("output_item.added"), "{out}");
        assert!(!out.contains("function_call_arguments"), "{out}");

        let tail = converter.finish();
        assert!(tail.contains("\"type\":\"custom_tool_call\""), "{tail}");
        assert!(tail.contains("\"input\":\"1+1\""), "{tail}");
    }

    #[test]
    fn test_streaming_converter_function_calls_unaffected_by_custom_set() {
        let custom: HashSet<String> = ["exec".to_string()].into();
        let mut converter =
            ResponsesStreamConverter::new("gpt-5.6-sol", false).with_custom_tools(custom);

        let chunk = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_w\",\"function\":{\"name\":\"wait\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n";
        let out = converter.push_bytes(chunk.as_bytes()).unwrap();
        assert!(out.contains("output_item.added"), "{out}");
        assert!(out.contains("\"type\":\"function_call\""), "{out}");

        let tail = converter.finish();
        assert!(tail.contains("function_call_arguments.done"), "{tail}");
        assert!(!tail.contains("custom_tool_call"), "{tail}");
    }

    #[test]
    fn test_convert_request_openrouter_transforms_model() {
        let body = json!({"model": "gpt-5.2-codex", "input": []});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://openrouter.ai/api/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        assert_eq!(chat["model"], "openai/gpt-5.2-codex");
    }

    #[test]
    fn test_convert_request_caps_max_output_tokens() {
        let body = json!({
            "model": "gpt-4o",
            "input": [],
            "max_output_tokens": 12000
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
            },
        );
        assert_eq!(chat["max_tokens"], 8192);
    }

    #[test]
    fn test_convert_request_caps_string_max_output_tokens() {
        let body = json!({
            "model": "gpt-4o",
            "input": [],
            "max_output_tokens": "12000"
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
            },
        );
        assert_eq!(chat["max_tokens"], 8192);
    }

    #[test]
    fn test_apply_max_tokens_cap_to_fields_caps_chat_completions_fields() {
        let mut body = json!({
            "max_tokens": 10000,
            "max_output_tokens": 9000
        });
        apply_max_tokens_cap_to_fields(&mut body, Some(8192), &["max_tokens", "max_output_tokens"]);
        assert_eq!(body["max_tokens"], 8192);
        assert_eq!(body["max_output_tokens"], 8192);
    }

    #[test]
    fn test_apply_max_tokens_cap_to_fields_caps_numeric_string_fields() {
        let mut body = json!({
            "max_tokens": "10000",
            "max_output_tokens": "9000"
        });
        apply_max_tokens_cap_to_fields(&mut body, Some(8192), &["max_tokens", "max_output_tokens"]);
        assert_eq!(body["max_tokens"], 8192);
        assert_eq!(body["max_output_tokens"], 8192);
    }

    // ── convert_chat_response_to_responses_sse ─────────────────────────────────

    #[test]
    fn test_convert_response_text_contains_required_events() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 90
            },
            "choices": [{"message": {"role": "assistant", "content": "Here are your files."}}]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("event: response.created\n"));
        assert!(sse.contains("event: response.output_text.delta\n"));
        assert!(sse.contains("event: response.output_text.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("Here are your files."));
        assert!(sse.contains("\"cache_read_input_tokens\":90"));
    }

    #[test]
    fn test_convert_response_tool_call_contains_required_events() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("event: response.output_item.added\n"));
        assert!(sse.contains("event: response.function_call_arguments.delta\n"));
        assert!(sse.contains("event: response.function_call_arguments.done\n"));
        assert!(sse.contains("event: response.output_item.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("call_abc"));
        assert!(sse.contains("shell"));
    }

    #[test]
    fn test_convert_response_restores_namespaced_tool_identity() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_ns", "type": "function",
                    "function": {"name": "mcp__browser__navigate", "arguments": "{\"url\":\"https://example.com\"}"}
                }]},
                "finish_reason": "tool_calls"
            }]
        });
        let namespaces: ToolNamespaceMap = [(
            "mcp__browser__navigate".to_string(),
            ("mcp__browser".to_string(), "navigate".to_string()),
        )]
        .into();

        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-5.6-sol",
            &HashSet::new(),
            &namespaces,
        );
        let events: Vec<Value> = sse
            .lines()
            .filter_map(crate::services::http_utils::sse_data_payload)
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect();
        let added = events
            .iter()
            .find(|event| event["type"] == "response.output_item.added")
            .unwrap();
        assert_eq!(added["item"]["namespace"], "mcp__browser");
        assert_eq!(added["item"]["name"], "navigate");
        let done = events
            .iter()
            .find(|event| event["type"] == "response.output_item.done")
            .unwrap();
        assert_eq!(done["item"]["namespace"], "mcp__browser");
        assert_eq!(done["item"]["name"], "navigate");
    }

    #[test]
    fn test_convert_response_empty_content_no_delta_event() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(!sse.contains("response.output_text.delta"));
        assert!(sse.contains("response.output_text.done"));
    }

    #[test]
    fn test_convert_response_joins_text_from_multiple_choices() {
        let chat = json!({
            "choices": [
                {"message": {"role": "assistant", "content": "Hello"}},
                {"message": {"role": "assistant", "content": "world"}}
            ]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("Hello\\nworld"));
    }

    #[test]
    fn test_convert_response_supports_content_array_parts() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]
                }
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("Hello\\nworld"));
    }

    #[test]
    fn test_convert_response_supports_result_response_envelope() {
        let chat = json!({
            "result": {"response": "Hello from envelope"}
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("Hello from envelope"));
    }

    #[test]
    fn test_convert_response_supports_responses_output_message() {
        let chat = json!({
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello from output"}]
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("Hello from output"));
    }

    #[test]
    fn test_convert_response_supports_responses_output_function_call() {
        let chat = json!({
            "response": {
                "output": [{
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"ls\"}"
                }]
            }
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"call_id\":\"call_123\""));
        assert!(sse.contains("\"name\":\"shell\""));
    }

    #[test]
    fn test_convert_response_uses_correct_object_type() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"object\":\"response\""));
        assert!(!sse.contains("realtime.response"));
    }

    #[test]
    fn test_convert_response_includes_response_id() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"response_id\""));
    }

    #[test]
    fn test_convert_response_tool_call_has_call_id() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_abc123", "type": "function",
                                    "function": {"name": "shell", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"call_id\":\"call_abc123\""));
    }

    #[test]
    fn test_convert_response_length_reports_incomplete_status() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "truncat"},
                "finish_reason": "length"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("\"status\":\"incomplete\""), "{sse}");
        assert!(sse.contains("\"reason\":\"max_output_tokens\""), "{sse}");
    }

    #[test]
    fn stream_converter_length_reports_incomplete_status() {
        let mut c = ResponsesStreamConverter::new("gpt-4o", false);
        let mut sse = String::new();
        sse.push_str(&c.push_bytes(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}]}\n",
        ).unwrap());
        sse.push_str(&c.finish());
        assert!(sse.contains("\"status\":\"incomplete\""), "{sse}");
    }

    #[test]
    fn test_convert_request_function_call_output_array_form() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "function_call_output", "call_id": "call_1",
                 "output": [{"type": "output_text", "text": "line one"},
                            {"type": "output_text", "text": "line two"}]}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["content"], "line one\nline two");
    }

    #[test]
    fn test_convert_request_string_input() {
        let body = json!({"model": "gpt-4", "input": "hello there"});
        assert!(is_responses_api_format(&body));
        let chat = convert_responses_to_chat_request(&body, &openai_router_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello there");
    }

    #[test]
    fn test_convert_response_text_only_emits_single_message_item() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        let completed = sse
            .lines()
            .find(|l| l.contains("\"type\":\"response.completed\""))
            .and_then(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str::<Value>(d).unwrap())
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
    }

    #[test]
    fn test_convert_response_keeps_text_alongside_tool_calls() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check that file.",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "read", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        // Preamble text must survive as a message item, before the call.
        assert!(
            sse.contains("\"text\":\"Let me check that file.\""),
            "{sse}"
        );
        assert!(sse.contains("\"call_id\":\"call_1\""));
        let completed = sse
            .lines()
            .find(|l| l.contains("\"type\":\"response.completed\""))
            .and_then(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str::<Value>(d).unwrap())
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "Let me check that file.");
        assert_eq!(output[1]["type"], "function_call");
    }

    #[test]
    fn test_convert_response_tool_calls_without_text_skip_message_item() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "read", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        let completed = sse
            .lines()
            .find(|l| l.contains("\"type\":\"response.completed\""))
            .and_then(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str::<Value>(d).unwrap())
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
    }

    // ── SSE accumulator ────────────────────────────────────────────────────────

    #[test]
    fn test_accumulate_chat_sse_text_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_accumulate_chat_sse_tool_call_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        let tcs = result["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs[0]["id"], "call_x");
        assert_eq!(tcs[0]["function"]["name"], "shell");
        // No content deltas arrived → content stays null.
        assert!(result["choices"][0]["message"]["content"].is_null());
        assert!(
            tcs[0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("ls")
        );
    }

    #[test]
    fn test_accumulate_chat_sse_keeps_content_with_tool_calls() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Let me check.\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        let msg = &result["choices"][0]["message"];
        assert_eq!(msg["content"], "Let me check.");
        assert_eq!(msg["tool_calls"][0]["id"], "call_x");
    }

    #[test]
    fn test_parse_provider_response_json() {
        let json_text = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        let result = parse_provider_response(json_text).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_parse_provider_response_sse_fallback() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\ndata: [DONE]\n";
        let result = parse_provider_response(sse).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_convert_request_copilot_skips_model_transform() {
        let body = json!({"model": "gpt-4o", "input": []});
        let config = ResponsesToChatConversionConfig {
            target_base_url: String::new(),
            target_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
        };
        let chat = convert_responses_to_chat_request(&body, &config);
        assert_eq!(chat["model"], "gpt-4o");
    }

    #[test]
    fn test_convert_response_sse_empty_choices_no_panic() {
        let chat = json!({"choices": []});
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_response_sse_missing_choices_no_panic() {
        let chat = json!({});
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_request_missing_model_uses_default() {
        let body = json!({"input": [{"type": "message", "role": "user", "content": "hi"}]});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        assert!(chat.get("model").is_some());
    }

    #[test]
    fn test_convert_request_empty_input() {
        let body = json!({"model": "gpt-4o", "input": []});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatConversionConfig {
                target_base_url: "https://example.com/v1".to_string(),
                target_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_extract_chat_response_payload_null_message() {
        let chat = json!({"choices": [{"message": null}]});
        let (text, tool_calls, reasoning) = extract_chat_response_payload(&chat);
        assert!(text.is_empty());
        assert!(tool_calls.is_empty());
        assert!(reasoning.is_empty());
    }

    #[test]
    fn test_extract_chat_response_payload_output_text_item() {
        let chat = json!({
            "output": [{"type": "output_text", "text": "hello from output_text"}]
        });
        let (text, tool_calls, _) = extract_chat_response_payload(&chat);
        assert_eq!(text, "hello from output_text");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_accumulate_chat_sse_empty_input() {
        let result = accumulate_chat_sse("");
        assert!(
            result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn test_accumulate_chat_sse_only_done() {
        let result = accumulate_chat_sse("data: [DONE]\n");
        assert!(
            result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn test_parse_provider_response_empty_string() {
        let result = parse_provider_response("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_provider_response_malformed_json() {
        let result = parse_provider_response("{not valid json}");
        assert!(result.is_err() || result.unwrap().is_object());
    }

    #[test]
    fn test_chat_usage_to_responses_usage_missing() {
        let chat = json!({"choices": []});
        assert!(chat_usage_to_responses_usage(&chat).is_none());
    }

    #[test]
    fn test_chat_usage_to_responses_usage_present() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 8
            }
        });
        let usage = chat_usage_to_responses_usage(&chat).unwrap();
        assert_eq!(usage["input_tokens"], 10);
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["cache_read_input_tokens"], 8);
    }

    #[test]
    fn test_accumulate_chat_sse_malformed_json_skipped() {
        let sse = "data: {invalid json!!!}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: not even close to json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_convert_chat_response_to_responses_sse_null_usage() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": null,
                "total_tokens": null
            }
        });
        let sse = convert_chat_response_to_responses_sse(
            &chat,
            false,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        );
        assert!(sse.contains("event: response.created\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("\"input_tokens\""));
        assert!(sse.contains("\"output_tokens\""));
        assert!(sse.contains("hi"));
    }

    #[test]
    fn test_convert_responses_to_chat_actual_model_override() {
        let body = json!({
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "hello"}]
        });
        let config = ResponsesToChatConversionConfig {
            target_base_url: "https://example.com/v1".to_string(),
            target_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: Some("kimi-k2.5".to_string()),
            max_tokens_cap: None,
        };
        let chat = convert_responses_to_chat_request(&body, &config);
        assert_eq!(chat["model"], "kimi-k2.5");
    }

    #[test]
    fn test_extract_chat_response_payload_no_choices_no_output() {
        let chat = json!({"id": "chatcmpl-123", "object": "chat.completion"});
        let (text, tool_calls, reasoning) = extract_chat_response_payload(&chat);
        assert!(
            text.is_empty(),
            "text should be empty when no choices/output"
        );
        assert!(
            tool_calls.is_empty(),
            "tool_calls should be empty when no choices/output"
        );
        assert!(
            reasoning.is_empty(),
            "reasoning should be empty when no choices/output"
        );
    }

    #[test]
    fn test_chat_usage_to_responses_usage_null_tokens() {
        let chat = json!({
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": 5,
                "total_tokens": 5
            }
        });
        let usage = chat_usage_to_responses_usage(&chat).expect("usage should be Some");
        assert!(usage["input_tokens"].is_null());
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["total_tokens"], 5);
    }

    // ── convert_chat_to_responses_request ─────────────────────────────────────

    #[test]
    fn chat_to_responses_simple_message() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024
        });
        let req = convert_chat_to_responses_request(&body);
        assert_eq!(req["model"], "gpt-4o");
        assert_eq!(req["instructions"], "You are helpful.");
        assert_eq!(req["max_output_tokens"], 1024);
        assert_eq!(req["stream"], false);
        let input = req["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn chat_to_responses_tool_calls() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"loc\":\"SF\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}
            ]
        });
        let req = convert_chat_to_responses_request(&body);
        let input = req["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["name"], "get_weather");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["output"], "Sunny");
    }

    // ── cap_reasoning_effort ──────────────────────────────────────────────────

    #[test]
    fn cap_reasoning_effort_clamps_xhigh_chat() {
        let mut body = json!({"reasoning_effort": "xhigh"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_clamps_xhigh_responses() {
        let mut body = json!({"reasoning": {"effort": "xhigh"}});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_passes_through_high() {
        let mut body = json!({"reasoning_effort": "high"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_spares_models_publishing_xhigh() {
        // kimi-k2.6 publishes xhigh in the embedded snapshot.
        let mut body = json!({"model": "kimi-k2.6", "reasoning_effort": "xhigh"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "xhigh");

        let mut body = json!({"model": "kimi-k2.6", "reasoning": {"effort": "xhigh"}});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn cap_reasoning_effort_still_clamps_models_without_xhigh() {
        // deepseek-v4-flash publishes only high,max.
        let mut body = json!({"model": "deepseek-v4-flash", "reasoning_effort": "xhigh"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_noop_when_absent() {
        let mut body = json!({"model": "x"});
        cap_reasoning_effort(&mut body);
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn chat_to_responses_tools_converted() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "shell", "description": "Run cmd", "parameters": {}}}
            ]
        });
        let req = convert_chat_to_responses_request(&body);
        let tools = req["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "shell");
    }

    // ── convert_responses_json_to_chat ────────────────────────────────────────

    #[test]
    fn responses_json_to_chat_text() {
        let resp = json!({
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello!"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["id"], "resp_123");
        assert_eq!(chat["model"], "gpt-4o");
        assert_eq!(chat["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        assert_eq!(chat["usage"]["prompt_tokens"], 10);
        assert_eq!(chat["usage"]["completion_tokens"], 5);
    }

    #[test]
    fn responses_json_to_chat_tool_calls() {
        let resp = json!({
            "id": "resp_456",
            "model": "gpt-4o",
            "output": [
                {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{\"path\":\"test.rs\"}"}
            ]
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["choices"][0]["finish_reason"], "tool_calls");
        let tcs = chat["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "c1");
        assert_eq!(tcs[0]["function"]["name"], "read_file");
    }

    #[test]
    fn responses_json_to_chat_wrapped_response() {
        let resp = json!({
            "response": {
                "id": "resp_789",
                "model": "gpt-4o",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}
                ]
            }
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["id"], "resp_789");
        assert_eq!(chat["choices"][0]["message"]["content"], "Hi");
    }

    #[test]
    fn responses_content_to_chat_text_only_collapses_to_string() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"}
        ])));
        assert_eq!(v, Value::String("hello\nworld".to_string()));
    }

    #[test]
    fn responses_content_to_chat_string_passthrough() {
        let v = convert_responses_content_to_chat(Some(&json!("plain string")));
        assert_eq!(v, Value::String("plain string".to_string()));
    }

    #[test]
    fn responses_content_to_chat_input_image_data_uri_preserved() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgo=",
                "detail": "high"
            }}
        ])));
        let arr = v.as_array().expect("array shape when image present");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "what is this?");
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(
            arr[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
        assert_eq!(arr[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn responses_content_to_chat_input_image_string_url_accepted() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_image", "image_url": "https://example.com/x.jpg"}
        ])));
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["type"], "image_url");
        assert_eq!(arr[0]["image_url"]["url"], "https://example.com/x.jpg");
    }

    #[test]
    fn responses_content_to_chat_input_file_inlined_as_text_reference() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "look at this:"},
            {"type": "input_file", "filename": "report.pdf"}
        ])));
        // Both parts are text after conversion (file collapses to a text
        // reference) so the output collapses to a single string.
        assert_eq!(
            v,
            Value::String("look at this:\n[attached file: report.pdf]".to_string())
        );
    }

    // ── ResponsesStreamConverter ───────────────────────────────────────────────

    /// Collects every `event:` line emitted across the chunks + finish, in order.
    fn collect_events(sse: &str) -> Vec<String> {
        sse.lines()
            .filter_map(|l| l.strip_prefix("event: ").map(str::to_string))
            .collect()
    }

    fn chat_chunk_line(delta: Value) -> Vec<u8> {
        format!(
            "data: {}\n\n",
            json!({"choices": [{"index": 0, "delta": delta}]})
        )
        .into_bytes()
    }

    #[test]
    fn stream_converter_emits_reasoning_then_text_in_order() {
        let mut c = ResponsesStreamConverter::new("deepseek-reasoner", false);
        let mut sse = String::new();
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({"reasoning_content": "thin"})))
                .unwrap(),
        );
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({"reasoning_content": "king"})))
                .unwrap(),
        );
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({"content": "Hel"})))
                .unwrap(),
        );
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({"content": "lo"})))
                .unwrap(),
        );
        sse.push_str(&c.finish());

        let events = collect_events(&sse);
        // Opening event first, completed last.
        assert_eq!(events.first().unwrap(), "response.created");
        assert_eq!(events.last().unwrap(), "response.completed");
        // Reasoning item is opened before the message item.
        let reasoning_added = events
            .iter()
            .position(|e| e == "response.output_item.added")
            .unwrap();
        let first_text_delta = events
            .iter()
            .position(|e| e == "response.output_text.delta")
            .unwrap();
        let reasoning_delta = events
            .iter()
            .position(|e| e == "response.reasoning_summary_text.delta")
            .unwrap();
        assert!(reasoning_added < reasoning_delta);
        assert!(reasoning_delta < first_text_delta);

        // Deltas are streamed (two of each), not collapsed into one blob.
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.reasoning_summary_text.delta")
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.output_text.delta")
                .count(),
            2
        );

        // response.completed carries the assembled text + reasoning items.
        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "thinking");
        assert_eq!(output[1]["type"], "message");
        // Message content is output_text only — a reasoning part would make
        // Codex.app drop the message.
        let content = output[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "Hello");
        assert!(!content.iter().any(|p| p["type"] == "reasoning"));
    }

    #[test]
    fn stream_converter_accepts_spaceless_data_prefix() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let mut sse = String::new();
        sse.push_str(
            &c.push_bytes(
                b"data:{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n",
            )
            .unwrap(),
        );
        sse.push_str(&c.push_bytes(b"data:[DONE]\n").unwrap());
        sse.push_str(&c.finish());
        assert!(sse.contains("\"delta\":\"hi\""), "{sse}");
    }

    #[test]
    fn stream_converter_streams_tool_call_arguments_incrementally() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let mut sse = String::new();
        // First fragment carries id + name; later fragments only arguments.
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({
                "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"ci"}}]
            })))
            .unwrap(),
        );
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({
                "tool_calls": [{"index": 0, "function": {"arguments": "ty\":\"SF\"}"}}]
            })))
            .unwrap(),
        );
        sse.push_str(&c.finish());

        let events = collect_events(&sse);
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.function_call_arguments.delta")
                .count(),
            2,
            "argument fragments should stream as separate deltas"
        );
        // Exactly one function_call item opened.
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.output_item.added")
                .count(),
            1
        );

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let item = &completed["response"]["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_abc");
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["arguments"], "{\"city\":\"SF\"}");
    }

    #[test]
    fn stream_converter_handles_split_data_lines_across_chunks() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let line = chat_chunk_line(json!({"content": "hi"}));
        // Split the SSE line mid-way to exercise the pending-buffer reassembly.
        let (a, b) = line.split_at(line.len() / 2);
        let mut sse = String::new();
        sse.push_str(&c.push_bytes(a).unwrap());
        sse.push_str(&c.push_bytes(b).unwrap());
        sse.push_str(&c.finish());

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "hi"
        );
    }

    #[test]
    fn stream_converter_maps_usage_into_completed_event() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let mut sse = String::new();
        sse.push_str(
            &c.push_bytes(&chat_chunk_line(json!({"content": "x"})))
                .unwrap(),
        );
        // Trailing usage-only chunk (stream_options.include_usage).
        sse.push_str(&c.push_bytes(
            format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
            )
            .as_bytes(),
        ).unwrap());
        sse.push_str(&c.push_bytes(b"data: [DONE]\n\n").unwrap());
        sse.push_str(&c.finish());

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let usage = &completed["response"]["usage"];
        assert_eq!(usage["input_tokens"], 10);
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["total_tokens"], 15);
    }

    // ── ResponsesToChatStreamConverter ─────────────────────────────────────────

    /// Round-trip through both streaming converters: a chat SSE stream → Responses
    /// SSE (existing converter) → chat SSE (new converter) must preserve content,
    /// reasoning, tool calls, and usage. The existing converter is the oracle.
    #[test]
    fn responses_to_chat_stream_roundtrip() {
        let chat_sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        // chat → responses (oracle), then responses → chat (unit under test).
        let mut to_resp = ResponsesStreamConverter::new("gpt-5.4", false);
        let mut responses_sse = to_resp.push_bytes(chat_sse.as_bytes()).unwrap();
        responses_sse.push_str(&to_resp.finish());

        let mut to_chat = ResponsesToChatStreamConverter::new("gpt-5.4", true);
        let mut back = to_chat.push_bytes(responses_sse.as_bytes()).unwrap();
        back.push_str(&to_chat.finish());

        assert!(
            back.trim_end().ends_with("data: [DONE]"),
            "must terminate the stream"
        );
        let acc = accumulate_chat_sse(&back);
        let msg = &acc["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "think");
        assert_eq!(acc["choices"][0]["finish_reason"], "tool_calls");
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "shell");
        assert_eq!(tc["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(tc["id"], "call_1");

        // usage rides the dedicated trailing chunk (include_usage = true).
        let usage_line = back
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .find(|c| c.get("usage").is_some_and(|u| !u.is_null()))
            .expect("a usage chunk");
        assert_eq!(usage_line["usage"]["prompt_tokens"], 10);
        assert_eq!(usage_line["usage"]["completion_tokens"], 5);
        assert_eq!(usage_line["usage"]["total_tokens"], 15);
    }

    /// A plain text turn yields content + a `stop` finish and a clean terminator.
    #[test]
    fn responses_to_chat_stream_text_only() {
        let responses_sse = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
        );
        let mut c = ResponsesToChatStreamConverter::new("gpt-5.4", false);
        let mut out = c.push_bytes(responses_sse.as_bytes()).unwrap();
        out.push_str(&c.finish());
        let acc = accumulate_chat_sse(&out);
        assert_eq!(acc["choices"][0]["message"]["content"], "hi");
        assert_eq!(acc["choices"][0]["finish_reason"], "stop");
        // include_usage = false → no usage chunk emitted.
        assert!(!out.contains("\"usage\""));
    }
}
