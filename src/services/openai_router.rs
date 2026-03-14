/**
 * OpenAI-Compatible Router Service
 *
 * Acts as an HTTP proxy that intercepts Claude Code requests (Anthropic format)
 * and routes them to OpenAI-compatible providers (like Cloudflare Workers AI),
 * handling all necessary API transformations.
 *
 * Flow:
 * Claude Code (Anthropic /v1/messages) → Router → OpenAI /v1/chat/completions → Cloudflare
 */
use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, convert_anthropic_to_openai_request,
};
use crate::services::anthropic_chat_response::{
    OpenAIToAnthropicConfig, UsageValueMode, convert_openai_to_anthropic_message,
};
use crate::services::http_utils::{self, router_http_client};
use crate::services::model_names::{
    infer_provider_name_from_model, is_gateway_style_endpoint, select_model_for_protocol,
    should_preserve_cross_protocol_model,
};
use crate::services::openai_anthropic_bridge::convert_openai_chat_response_to_sse;
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    convert_gemini_to_openai_chat_response, convert_openai_chat_to_gemini_request,
    openai_chat_model,
};
use crate::services::provider_protocol::{ProviderProtocol, fallback_protocols, is_protocol_mismatch};

#[derive(Clone)]
pub struct OpenAIRouterConfig {
    /// The target OpenAI-compatible provider base URL (e.g., Cloudflare)
    pub target_base_url: String,
    /// API key for the target provider
    pub target_api_key: String,
    /// The upstream protocol spoken by the provider.
    pub target_protocol: ProviderProtocol,
    /// Optional model prefix to add (e.g., "@cf/" for Cloudflare)
    pub model_prefix: Option<String>,
    /// Whether the provider requires `reasoning_content` on assistant tool-call turns (e.g., Moonshot)
    pub requires_reasoning_content: bool,
    /// Cap applied to `max_tokens` before forwarding to the provider.
    /// Use for providers with hard limits (e.g., DeepSeek: 8192).
    pub max_tokens_cap: Option<u64>,
}

pub struct OpenAIRouter {
    config: OpenAIRouterConfig,
}

struct OpenAIRouterState {
    config: Arc<OpenAIRouterConfig>,
    client: reqwest::Client,
    active_protocol: Arc<AtomicU8>,
}

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
}

impl OpenAIRouter {
    pub fn new(config: OpenAIRouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let state = OpenAIRouterState {
            config: Arc::new(self.config.clone()),
            client: router_http_client(),
            active_protocol: Arc::new(AtomicU8::new(self.config.target_protocol.to_u8())),
        };
        let handle = tokio::spawn(async move { run_router(listener, state).await });
        Ok((port, handle))
    }
}

async fn run_router(listener: tokio::net::TcpListener, state: OpenAIRouterState) -> Result<()> {
    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = state.config.clone();
        let client = state.client.clone();
        let active_protocol = state.active_protocol.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(err) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
            };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();

            if !http_utils::is_post_path(&request, &["/v1/messages", "/messages"]) {
                let not_found =
                    http_utils::http_response(404, "application/json", "{\"error\":\"Not found\"}");
                let _ = socket.write_all(not_found.as_bytes()).await;
                return;
            }

            let response = match handle_anthropic_to_upstream(&request, &config, &client, &active_protocol).await {
                Ok(response) => response,
                Err(e) => {
                    let error = http_utils::http_error_response(500, &e.to_string());
                    let _ = socket.write_all(error.as_bytes()).await;
                    return;
                }
            };

            let _ = write_router_response(&mut socket, response).await;
        });
    }
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_response_head(status, &content_type, body.len());
            socket.write_all(headers.as_bytes()).await?;
            socket.write_all(&body).await?;
        }
    }

    Ok(())
}

/// Apply an optional prefix to a model name, skipping if the prefix is already present.
fn apply_model_prefix(model: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !model.starts_with(p) => format!("{}{}", p, model),
        _ => model.to_string(),
    }
}

/// Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions
async fn handle_anthropic_to_upstream(
    request: &str,
    config: &Arc<OpenAIRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
) -> Result<RouterResponse> {
    let passthrough_headers = http_utils::extract_passthrough_headers(request)?;
    let body_str = http_utils::extract_request_body(request)?;

    let body: Value = serde_json::from_str(body_str)?;
    let mut simplified = anthropic_to_openai(&body, config.requires_reasoning_content);
    cap_max_tokens_field(&mut simplified, config.max_tokens_cap);
    let requested_stream = simplified
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let current = ProviderProtocol::from_u8(active_protocol.load(Ordering::Relaxed));
    let mut candidates = std::iter::once(current)
        .chain(fallback_protocols(current, &config.target_base_url))
        .collect::<Vec<_>>();

    let mut last_response: Option<RouterResponse> = None;

    for (attempt, protocol) in candidates.drain(..).enumerate() {
        let mut req_body = simplified.clone();
        let mut attempt_headers = passthrough_headers.clone();
        prepare_gateway_model_metadata(&mut req_body, &mut attempt_headers, config, protocol);

        // Apply model prefix for OpenAI protocol
        if protocol == ProviderProtocol::Openai
            && let Some(model) = req_body.get_mut("model")
            && let Some(model_str) = model.as_str()
        {
            *model = Value::String(apply_model_prefix(
                model_str,
                config.model_prefix.as_deref(),
            ));
        }

        let (status_code, response_opt) = match protocol {
            ProviderProtocol::Google => {
                req_body["stream"] = json!(false);
                let model = openai_chat_model(&req_body, "gemini-2.5-pro");
                let google_body = convert_openai_chat_to_gemini_request(
                    &req_body,
                    &OpenAIToGeminiConfig {
                        default_model: "gemini-2.5-pro",
                    },
                );
                let url = build_google_generate_content_url(&config.target_base_url, &model);
                let response = client
                    .post(&url)
                    .headers(attempt_headers)
                    .header("x-goog-api-key", config.target_api_key.as_str())
                    .header("Content-Type", "application/json")
                    .header("User-Agent", "aivo-router/1.0")
                    .json(&google_body)
                    .send()
                    .await?;

                let status_code = response.status().as_u16();
                let response_body = response.text().await?;
                if is_protocol_mismatch(status_code) {
                    (status_code, None)
                } else if status_code >= 400 {
                    let r = RouterResponse::Buffered {
                        status: status_code,
                        content_type: "application/json".to_string(),
                        body: response_body.into_bytes(),
                    };
                    (status_code, Some(r))
                } else {
                    let google_response: Value = serde_json::from_str(&response_body)?;
                    let openai_response =
                        convert_gemini_to_openai_chat_response(&google_response, &model);
                    let r = if requested_stream {
                        let openai_sse = convert_openai_chat_response_to_sse(&openai_response);
                        let anthropic_sse = convert_openai_sse_to_anthropic(&openai_sse, 200)?;
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "text/event-stream".to_string(),
                            body: anthropic_sse.into_bytes(),
                        }
                    } else {
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "application/json".to_string(),
                            body: convert_openai_to_anthropic(&openai_response.to_string(), 200)?
                                .into_bytes(),
                        }
                    };
                    (status_code, Some(r))
                }
            }
            _ => {
                // OpenAI or Anthropic — use chat completions endpoint
                let url = http_utils::build_chat_completions_url(&config.target_base_url);
                let response = client
                    .post(&url)
                    .headers(attempt_headers)
                    .header("Authorization", format!("Bearer {}", config.target_api_key))
                    .header("Content-Type", "application/json")
                    .header("User-Agent", "aivo-router/1.0")
                    .json(&req_body)
                    .send()
                    .await?;

                let status_code = response.status().as_u16();
                if is_protocol_mismatch(status_code) {
                    (status_code, None)
                } else {
                    let is_streaming = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .map(|ct| ct.contains("text/event-stream"))
                        .unwrap_or(false);
                    let response_body = response.text().await?;
                    let r = if status_code == 200
                        && (is_streaming || response_body.starts_with("data:"))
                    {
                        let anthropic_sse =
                            convert_openai_sse_to_anthropic(&response_body, status_code)?;
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "text/event-stream".to_string(),
                            body: anthropic_sse.into_bytes(),
                        }
                    } else {
                        let anthropic_response =
                            convert_openai_to_anthropic(&response_body, status_code)?;
                        RouterResponse::Buffered {
                            status: status_code,
                            content_type: "application/json".to_string(),
                            body: anthropic_response.into_bytes(),
                        }
                    };
                    (status_code, Some(r))
                }
            }
        };

        if let Some(r) = response_opt {
            // Success (not a protocol mismatch)
            if attempt > 0 {
                active_protocol.store(protocol.to_u8(), Ordering::Relaxed);
                eprintln!("  • Protocol auto-switched to {}", protocol.as_str());
            }
            return Ok(r);
        }
        // Protocol mismatch — try next candidate
        last_response = Some(RouterResponse::Buffered {
            status: status_code,
            content_type: "application/json".to_string(),
            body: format!("{{\"error\":\"Protocol mismatch ({})\"}}", status_code).into_bytes(),
        });
    }

    // All candidates exhausted — return last error
    Ok(last_response.unwrap_or(RouterResponse::Buffered {
        status: 503,
        content_type: "application/json".to_string(),
        body: b"{\"error\":\"No compatible protocol found\"}".to_vec(),
    }))
}

fn anthropic_to_openai(body: &Value, requires_reasoning_content: bool) -> Value {
    convert_anthropic_to_openai_request(
        body,
        &AnthropicToOpenAIConfig {
            default_model: "gpt-4o",
            preserve_stream: true,
            model_transform: None,
            include_reasoning_content: true,
            require_non_empty_reasoning_content: requires_reasoning_content,
            stringify_other_tool_result_content: true,
            fallback_tool_arguments_json: "{}",
        },
    )
}

fn prepare_gateway_model_metadata(
    simplified: &mut Value,
    passthrough_headers: &mut HeaderMap,
    config: &OpenAIRouterConfig,
    protocol: ProviderProtocol,
) {
    let requested_model = simplified
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let selected_model = if should_preserve_cross_protocol_model(
        &config.target_base_url,
        &requested_model,
        protocol,
    ) {
        requested_model.clone()
    } else {
        select_model_for_protocol(Some(&requested_model), None, protocol)
    };
    simplified["model"] = Value::String(selected_model);

    if is_gateway_style_endpoint(&config.target_base_url)
        && !passthrough_headers.contains_key("x-provider")
        && let Some(provider) = infer_provider_name_from_model(&requested_model)
        && let Ok(value) = HeaderValue::from_str(&provider)
    {
        passthrough_headers.insert("x-provider", value);
    }
}

fn cap_max_tokens_field(body: &mut Value, cap: Option<u64>) {
    let Some(limit) = cap else {
        return;
    };
    if let Some(mt) = body.get("max_tokens").and_then(http_utils::parse_token_u64)
        && mt > limit
    {
        body["max_tokens"] = json!(limit);
    }
}

/// Convert OpenAI /v1/chat/completions response to Anthropic /v1/messages format
fn convert_openai_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    // If error status, return as-is
    if status_code >= 400 {
        return Ok(response_body.to_string());
    }

    let openai_resp: Value = serde_json::from_str(response_body)?;
    let anthropic_resp = convert_openai_to_anthropic_message(
        &openai_resp,
        &OpenAIToAnthropicConfig {
            fallback_id: "msg_default",
            model: openai_resp
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown"),
            include_created: true,
            usage_value_mode: UsageValueMode::CoerceU64,
        },
    );

    Ok(anthropic_resp.to_string())
}

#[derive(Default)]
struct StreamToolBlock {
    anthropic_idx: usize,
    id: String,
    name: String,
    opened: bool,
    pending_args: String,
}

fn append_sse_event(output: &mut String, event: &str, data: Value) {
    output.push_str(&format!("event: {event}\ndata: {data}\n\n"));
}

fn ensure_message_start(
    output: &mut String,
    started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
) {
    if *started {
        return;
    }
    append_sse_event(
        output,
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0
                }
            }
        }),
    );
    *started = true;
}

#[allow(clippy::too_many_arguments)]
fn emit_tool_delta(
    output: &mut String,
    block_count: &mut usize,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    openai_idx: usize,
    id: Option<&str>,
    name: Option<&str>,
    args_fragment: Option<&str>,
    saw_tool_use: &mut bool,
) {
    let block = tool_blocks.entry(openai_idx).or_insert_with(|| {
        let idx = *block_count;
        *block_count += 1;
        StreamToolBlock {
            anthropic_idx: idx,
            ..Default::default()
        }
    });

    if let Some(v) = id
        && !v.is_empty()
    {
        block.id = v.to_string();
    }
    if let Some(v) = name
        && !v.is_empty()
    {
        block.name = v.to_string();
    }

    if let Some(fragment) = args_fragment
        && !fragment.is_empty()
    {
        if block.opened {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragment
                    }
                }),
            );
        } else {
            block.pending_args.push_str(fragment);
        }
    }

    if !block.opened && !block.name.is_empty() {
        if block.id.is_empty() {
            block.id = format!("toolu_{}", uuid_simple());
        }
        append_sse_event(
            output,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": block.anthropic_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name
                }
            }),
        );
        block.opened = true;
        *saw_tool_use = true;

        if !block.pending_args.is_empty() {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": block.pending_args
                    }
                }),
            );
            block.pending_args.clear();
        }
    }
}

fn map_openai_finish_reason(reason: &str) -> &'static str {
    match reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_stream_message(
    output: &mut String,
    message_started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    text_block_idx: &mut Option<usize>,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    stop_reason: &str,
) {
    ensure_message_start(output, message_started, message_id, model, input_tokens);

    if let Some(idx) = text_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    let mut ordered_tool_idxs = tool_blocks
        .values()
        .filter(|b| b.opened)
        .map(|b| b.anthropic_idx)
        .collect::<Vec<_>>();
    ordered_tool_idxs.sort_unstable();
    for idx in ordered_tool_idxs {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    append_sse_event(
        output,
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": {
                "output_tokens": output_tokens
            }
        }),
    );
    append_sse_event(
        output,
        "message_stop",
        json!({
            "type": "message_stop"
        }),
    );
}

struct OpenAIStreamConverter {
    pending: Vec<u8>,
    message_started: bool,
    finished: bool,
    block_count: usize,
    text_block_idx: Option<usize>,
    tool_blocks: HashMap<usize, StreamToolBlock>,
    message_id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    saw_tool_use: bool,
}

impl OpenAIStreamConverter {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            message_started: false,
            finished: false,
            block_count: 0,
            text_block_idx: None,
            tool_blocks: HashMap::new(),
            message_id: "msg".to_string(),
            model: "claude".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            saw_tool_use: false,
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.extend_from_slice(chunk);

        let mut output = String::new();
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.pending[..pos]).into_owned();
            self.pending.drain(..=pos);
            self.process_line(line.trim_end_matches('\r'), &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        if !self.pending.is_empty() {
            let line = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.process_line(line.trim_end_matches('\r'), &mut output)?;
        }

        if !self.finished && self.message_started {
            let fallback_stop = if self.saw_tool_use {
                "tool_use"
            } else {
                "end_turn"
            };
            finalize_stream_message(
                &mut output,
                &mut self.message_started,
                &self.message_id,
                &self.model,
                self.input_tokens,
                self.output_tokens,
                &mut self.text_block_idx,
                &mut self.tool_blocks,
                fallback_stop,
            );
            self.finished = true;
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let fallback_stop = if self.saw_tool_use {
                    "tool_use"
                } else {
                    "end_turn"
                };
                finalize_stream_message(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.output_tokens,
                    &mut self.text_block_idx,
                    &mut self.tool_blocks,
                    fallback_stop,
                );
                self.finished = true;
            }
            return Ok(());
        }

        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };

        if let Some(v) = chunk.get("id").and_then(|v| v.as_str())
            && !v.is_empty()
        {
            self.message_id = v.to_string();
        }
        if let Some(v) = chunk.get("model").and_then(|v| v.as_str())
            && !v.is_empty()
        {
            self.model = v.to_string();
        }
        if let Some(usage) = chunk.get("usage") {
            if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens = v;
            }
            if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = v;
            }
        }

        let choices = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        for choice in choices {
            let delta = choice.get("delta").cloned().unwrap_or(json!({}));

            if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                );
                if self.text_block_idx.is_none() {
                    let idx = self.block_count;
                    self.block_count += 1;
                    self.text_block_idx = Some(idx);
                    append_sse_event(
                        output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "text",
                                "text": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.text_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "text_delta",
                            "text": text
                        }
                    }),
                );
            }

            if let Some(function_call) = delta.get("function_call") {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                );
                emit_tool_delta(
                    output,
                    &mut self.block_count,
                    &mut self.tool_blocks,
                    0,
                    function_call.get("id").and_then(|v| v.as_str()),
                    function_call.get("name").and_then(|v| v.as_str()),
                    function_call.get("arguments").and_then(|v| v.as_str()),
                    &mut self.saw_tool_use,
                );
            }

            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                );
                for tc in tool_calls {
                    let openai_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    emit_tool_delta(
                        output,
                        &mut self.block_count,
                        &mut self.tool_blocks,
                        openai_idx,
                        tc.get("id").and_then(|v| v.as_str()),
                        tc.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str()),
                        tc.get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str()),
                        &mut self.saw_tool_use,
                    );
                }
            }

            if !self.finished
                && let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str())
                && !finish_reason.is_empty()
            {
                finalize_stream_message(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.output_tokens,
                    &mut self.text_block_idx,
                    &mut self.tool_blocks,
                    map_openai_finish_reason(finish_reason),
                );
                self.finished = true;
            }
        }

        Ok(())
    }
}

/// Convert OpenAI SSE streaming response to Anthropic SSE format.
fn convert_openai_sse_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    if status_code >= 400 {
        return Ok(format!("data: {}\n\ndata: [DONE]\n\n", response_body));
    }

    let mut converter = OpenAIStreamConverter::new();
    let mut sse_output = converter.push_bytes(response_body.as_bytes())?;
    sse_output.push_str(&converter.finish()?);
    Ok(sse_output)
}

/// Generate a collision-resistant unique ID using a monotonic counter + timestamp.
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:x}{:x}",
        duration.as_secs(),
        duration.subsec_nanos(),
        count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_to_anthropic_uses_response_model_and_created() {
        let openai_resp = r#"{
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let result = convert_openai_to_anthropic(openai_resp, 200).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-123");
        assert_eq!(parsed["model"], "gpt-4");
        assert_eq!(parsed["created"], 1700000000);
        assert_eq!(parsed["usage"]["input_tokens"], 10);
        assert_eq!(parsed["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_anthropic_to_openai_preserves_fields_and_tools() {
        let body = json!({
            "model": "gpt-4o-mini",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "read_file"},
            "stream": true
        });

        let req = anthropic_to_openai(&body, false);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(req["model"], "gpt-4o-mini");
        assert_eq!(req["stream"], true);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
        assert_eq!(req["max_tokens"], 128);
        assert_eq!(req["temperature"], 0.2);
        assert_eq!(req["top_p"], 0.9);
        assert_eq!(req["stop"][0], "END");
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_cap_max_tokens_field_caps_numeric_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 12000});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 8192);
    }

    #[test]
    fn test_cap_max_tokens_field_caps_numeric_string_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": "12000"});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 8192);
    }

    #[test]
    fn test_cap_max_tokens_field_ignores_non_numeric_string_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": "oops"});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], "oops");
    }

    #[test]
    fn test_anthropic_to_openai_maps_thinking_to_reasoning_content_for_tool_calls() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect files first."},
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(
            messages[0]["reasoning_content"],
            "Need to inspect files first."
        );
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_anthropic_to_openai_sets_reasoning_content_for_assistant_tool_calls_without_thinking() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], " ");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_prepare_gateway_model_metadata_preserves_gateway_claude_model() {
        let config = OpenAIRouterConfig {
            target_base_url: "https://api.ai.unilake.net/endpoint".to_string(),
            target_api_key: "test".to_string(),
            target_protocol: ProviderProtocol::Openai,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(&mut body, &mut headers, &config, config.target_protocol);

        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(
            headers.get("x-provider").and_then(|v| v.to_str().ok()),
            Some("anthropic")
        );
    }

    #[test]
    fn test_prepare_gateway_model_metadata_remaps_plain_openai_endpoint() {
        let config = OpenAIRouterConfig {
            target_base_url: "https://api.openai.com/v1".to_string(),
            target_api_key: "test".to_string(),
            target_protocol: ProviderProtocol::Openai,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(&mut body, &mut headers, &config, config.target_protocol);

        assert_eq!(body["model"], "gpt-4o");
        assert!(headers.get("x-provider").is_none());
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_text() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("event: message_start"));
        assert!(result.contains("\"type\":\"text_delta\""));
        assert!(result.contains("\"text\":\"hello \""));
        assert!(result.contains("\"text\":\"world\""));
        assert!(result.contains("\"stop_reason\":\"end_turn\""));
        assert!(result.contains("event: message_stop"));
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_split_tool_calls() {
        let sse = "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"list_files\"}}]},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("\"type\":\"tool_use\""));
        assert!(result.contains("\"id\":\"call_1\""));
        assert!(result.contains("\"name\":\"list_files\""));
        assert!(result.contains("\"type\":\"input_json_delta\""));
        assert!(result.contains("\"partial_json\":\"{\\\"path\\\":\\\".\\\"}\""));
        assert!(result.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_openai_stream_converter_handles_split_chunks() {
        let mut converter = OpenAIStreamConverter::new();
        let mut output = String::new();

        output.push_str(
            &converter
                .push_bytes(b"data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hel")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"lo\"},\"finish_reason\":null}]}\n")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n")
                .unwrap(),
        );
        output.push_str(&converter.push_bytes(b"data: [DONE]\n").unwrap());
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"text\":\"hello\""));
        assert!(output.contains("\"text\":\" world\""));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert_eq!(output.matches("event: message_stop").count(), 1);
    }

    #[test]
    fn test_model_prefix() {
        // Test the production helper directly to catch regressions in the real code path
        assert_eq!(
            apply_model_prefix("glm-4.7-flash", Some("@cf/")),
            "@cf/glm-4.7-flash"
        );
        // Prefix already present — must not double-add
        assert_eq!(
            apply_model_prefix("@cf/llama-3.1-8b", Some("@cf/")),
            "@cf/llama-3.1-8b"
        );
        // No prefix configured
        assert_eq!(apply_model_prefix("llama-3.1-8b", None), "llama-3.1-8b");
    }
}
