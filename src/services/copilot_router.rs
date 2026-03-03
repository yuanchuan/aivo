//! CopilotRouter: HTTP proxy for routing Claude Code requests through GitHub Copilot.
//!
//! Receives Anthropic Messages API requests from Claude Code, converts them to
//! OpenAI Chat Completions format, forwards to the Copilot API, and converts
//! the response back to Anthropic format.

use anyhow::Result;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::services::copilot_auth::{
    CopilotTokenManager, COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, COPILOT_OPENAI_INTENT,
};

#[derive(Clone)]
pub struct CopilotRouterConfig {
    pub github_token: String,
}

pub struct CopilotRouter {
    config: CopilotRouterConfig,
}

impl CopilotRouter {
    pub fn new(config: CopilotRouterConfig) -> Self {
        Self { config }
    }

    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let token_manager = CopilotTokenManager::new(self.config.github_token.clone());
        let client = reqwest::Client::new();
        let handle = tokio::spawn(async move { run_router(listener, token_manager, client).await });
        Ok((port, handle))
    }
}

async fn run_router(
    listener: tokio::net::TcpListener,
    token_manager: CopilotTokenManager,
    client: reqwest::Client,
) -> Result<()> {
    let token_manager = Arc::new(token_manager);
    let client = Arc::new(client);
    loop {
        let (mut socket, _) = listener.accept().await?;
        let tm = token_manager.clone();
        let client = client.clone();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let request_bytes = match read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(_) => return,
            };
            let request = String::from_utf8_lossy(&request_bytes);
            let response = if request.contains("POST /v1/messages") {
                match handle_messages(&request, &tm, &client).await {
                    Ok(r) => r,
                    Err(e) => error_response(500, &e.to_string()),
                }
            } else {
                error_response(404, "Not found")
            };
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}

async fn handle_messages(
    request: &str,
    tm: &Arc<CopilotTokenManager>,
    client: &reqwest::Client,
) -> Result<String> {
    let body_str = extract_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Smart mode: use cheaper Haiku model (0.33x) instead of default Sonnet/Opus.
    let smart_mode = std::env::var("AIVO_SMART_MODE").is_ok();

    // Convert Anthropic Messages → OpenAI Chat Completions.
    // smart_mode enables tool schema simplification.
    let openai_req = anthropic_to_openai(&body, smart_mode);

    // Get the model name for the response wrapper.
    let model = openai_req
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-sonnet-4-20250514")
        .to_string();

    // Get a valid Copilot token
    let (copilot_token, api_endpoint) = tm.get_token().await?;

    // Forward to Copilot API
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", copilot_token))
        .header("Content-Type", "application/json")
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .header("Openai-Intent", COPILOT_OPENAI_INTENT)
        .json(&openai_req)
        .send()
        .await?;

    let status = resp.status().as_u16();
    let resp_body = resp.text().await?;

    if status != 200 {
        return Ok(error_response(status, &resp_body));
    }

    let openai_resp: Value = serde_json::from_str(&resp_body)?;

    if is_streaming {
        // Convert to Anthropic SSE format
        let sse = openai_to_anthropic_sse(&openai_resp, &model);
        Ok(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{}",
            sse
        ))
    } else {
        // Convert to Anthropic Messages response
        let anthropic_resp = openai_to_anthropic(&openai_resp, &model);
        let json = serde_json::to_string(&anthropic_resp)?;
        Ok(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            json.len(),
            json
        ))
    }
}

// --- Model name mapping ---

/// Converts Anthropic/Claude Code model IDs to Copilot model IDs.
///
/// Claude Code sends names like `claude-sonnet-4-6-20250603` or `claude-sonnet-4-6`.
/// Copilot API expects names like `claude-sonnet-4.6` (dots for minor versions).
///
/// Steps:
///   1. Strip trailing date suffix `-YYYYMMDD`
///   2. Convert `claude-{family}-{major}-{minor}` → `claude-{family}-{major}.{minor}`
fn copilot_model_name(model: &str) -> String {
    // Strip trailing -YYYYMMDD date suffix
    let base = if model.len() > 9 {
        let (prefix, suffix) = model.split_at(model.len() - 9);
        if suffix.starts_with('-') && suffix[1..].chars().all(|c| c.is_ascii_digit()) {
            prefix
        } else {
            model
        }
    } else {
        model
    };

    // Convert hyphenated version to dotted: claude-sonnet-4-6 → claude-sonnet-4.6
    // Pattern: claude-{family}-{major}-{minor} where major/minor are digits
    if let Some(stripped) = base.strip_prefix("claude-") {
        let parts: Vec<&str> = stripped.split('-').collect();
        // e.g. ["sonnet", "4", "6"] or ["sonnet", "4"] or ["haiku", "4", "5"]
        if parts.len() >= 3 {
            let family = parts[0]; // sonnet, haiku, opus
            let major = parts[1]; // "4"
            let minor = parts[2]; // "6", "5"
            if major.chars().all(|c| c.is_ascii_digit())
                && minor.chars().all(|c| c.is_ascii_digit())
            {
                // Rejoin any remaining parts (e.g. "-thinking") after the version
                let rest = if parts.len() > 3 {
                    format!("-{}", parts[3..].join("-"))
                } else {
                    String::new()
                };
                return format!("claude-{}-{}.{}{}", family, major, minor, rest);
            }
        }
    }

    base.to_string()
}

// --- Request conversion: Anthropic Messages → OpenAI Chat Completions ---

fn anthropic_to_openai(body: &Value, smart_mode: bool) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // System message
    if let Some(system) = body.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            messages.push(json!({"role": "system", "content": system_text}));
        }
    }

    // Convert messages
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match content {
                Some(Value::String(text)) => {
                    messages.push(json!({"role": role, "content": text}));
                }
                Some(Value::Array(blocks)) => {
                    convert_content_blocks(blocks, role, &mut messages);
                }
                _ => {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
    }

    let raw_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-sonnet-4-20250514");
    let model = copilot_model_name(raw_model);

    let mut req = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });

    // max_tokens
    if let Some(mt) = body.get("max_tokens") {
        req["max_tokens"] = mt.clone();
    }

    // temperature
    if let Some(t) = body.get("temperature") {
        req["temperature"] = t.clone();
    }

    // top_p
    if let Some(tp) = body.get("top_p") {
        req["top_p"] = tp.clone();
    }

    // stop_sequences → stop
    if let Some(ss) = body.get("stop_sequences") {
        req["stop"] = ss.clone();
    }

    // tools - with optimization: simplify schemas to reduce token count
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                let name = tool.get("name").cloned().unwrap_or_default();
                let description = tool.get("description").cloned().unwrap_or(json!(""));
                let empty = json!({});
                let params = tool.get("input_schema").unwrap_or(&empty);

                // Apply tool schema simplification to reduce token usage
                // Enabled with --smart flag or AIVO_SMART_MODE env var
                let simplified_params = if smart_mode {
                    simplify_parameters(params)
                } else {
                    params.clone()
                };

                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": simplified_params,
                    }
                })
            })
            .collect();
        if !openai_tools.is_empty() {
            req["tools"] = Value::Array(openai_tools);
        }
    }

    // tool_choice: convert Anthropic format → OpenAI format
    if let Some(tc) = body.get("tool_choice") {
        match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => {
                req["tool_choice"] = json!("auto");
            }
            Some("any") => {
                req["tool_choice"] = json!("required");
            }
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
    }

    req
}

/// Simplifies JSON Schema parameters to reduce token count.
///
/// This optimization reduces Copilot credit usage by stripping verbose metadata
/// from tool definitions that Claude Code sends. Native Copilot agents use
/// simpler tool schemas, so we strip: title, description, examples, default,
/// $schema, $defs, etc. while preserving: type, properties, required, enum.
///
/// This can reduce tool definition size by 30-50%, significantly saving credits.
fn simplify_parameters(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut simplified = serde_json::Map::new();

            // Always keep type and properties
            if let Some(t) = map.get("type") {
                simplified.insert("type".to_string(), t.clone());
            }

            if let Some(props) = map.get("properties") {
                // Recursively simplify each property
                let simplified_props = match props {
                    Value::Object(props_map) => {
                        let mut new_props = serde_json::Map::new();
                        for (key, val) in props_map {
                            new_props.insert(key.clone(), simplify_parameters(val));
                        }
                        Value::Object(new_props)
                    }
                    _ => props.clone(),
                };
                simplified.insert("properties".to_string(), simplified_props);
            }

            // Keep required as-is (array of strings, usually small)
            if let Some(req) = map.get("required") {
                simplified.insert("required".to_string(), req.clone());
            }

            // Keep enum if present (reduces ambiguity)
            if let Some(en) = map.get("enum") {
                simplified.insert("enum".to_string(), en.clone());
            }

            // Keep items for array types
            if let Some(items) = map.get("items") {
                simplified.insert("items".to_string(), simplify_parameters(items));
            }

            // Strip these verbose fields:
            // - title, description (descriptions already on tool itself)
            // - examples, default (not needed for API)
            // - $schema, $id, definitions, $defs (metadata)
            // - const, readOnly, writeOnly (rarely needed)

            Value::Object(simplified)
        }
        // For arrays, simplify each element
        Value::Array(arr) => Value::Array(arr.iter().map(simplify_parameters).collect()),
        _ => schema.clone(),
    }
}

/// Converts Anthropic content blocks to OpenAI messages.
fn convert_content_blocks(blocks: &[Value], role: &str, messages: &mut Vec<Value>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or(json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input).unwrap_or_default(),
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let content = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                tool_results.push((tool_use_id, content));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        // Tool results → individual tool messages
        for (tool_use_id, content) in tool_results {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }));
        }
    } else if !tool_calls.is_empty() {
        // Assistant message with tool calls
        let content = if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n"))
        };
        let mut msg = json!({"role": role, "tool_calls": tool_calls});
        if !content.is_null() {
            msg["content"] = content;
        }
        messages.push(msg);
    } else {
        // Plain text message
        messages.push(json!({"role": role, "content": text_parts.join("\n")}));
    }
}

// --- Response conversion: OpenAI Chat Completions → Anthropic Messages ---

fn openai_to_anthropic(resp: &Value, model: &str) -> Value {
    let choices = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    // Merge all choices: Copilot may split text and tool_calls across multiple choices
    let mut content: Vec<Value> = Vec::new();
    let mut final_finish_reason = "stop";

    for choice in &choices {
        let message = choice.get("message").cloned().unwrap_or(json!({}));
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .unwrap_or("stop");

        // tool_calls finish_reason takes priority
        if finish_reason == "tool_calls" {
            final_finish_reason = "tool_calls";
        } else if final_finish_reason != "tool_calls" {
            final_finish_reason = finish_reason;
        }

        // Text content
        if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                content.push(json!({"type": "text", "text": text}));
            }
        }

        // Tool calls
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let args_str = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                }));
            }
        }
    }

    let stop_reason = match final_finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "end_turn",
        _ => "end_turn",
    };

    // If no content at all, add empty text block
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let mut result = json!({
        "id": resp.get("id").and_then(|i| i.as_str()).unwrap_or("msg_copilot"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
    });

    // Usage
    if let Some(usage) = resp.get("usage") {
        result["usage"] = json!({
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
        });
    } else {
        result["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    }

    result
}

/// Converts an OpenAI response to Anthropic SSE event stream.
fn openai_to_anthropic_sse(resp: &Value, model: &str) -> String {
    let anthropic = openai_to_anthropic(resp, model);
    let mut events = String::new();

    let input_tokens = anthropic["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let output_tokens = anthropic["usage"]["output_tokens"].as_i64().unwrap_or(0);

    // message_start
    events.push_str(&format!(
        "event: message_start\ndata: {}\n\n",
        json!({
            "type": "message_start",
            "message": {
                "id": anthropic["id"],
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        })
    ));

    // Emit each content block
    if let Some(content) = anthropic.get("content").and_then(|c| c.as_array()) {
        for (idx, block) in content.iter().enumerate() {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");

            match block_type {
                "text" => {
                    let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    // content_block_start
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                    ));
                    // content_block_delta
                    if !text.is_empty() {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text}})
                        ));
                    }
                    // content_block_stop
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                "tool_use" => {
                    // content_block_start
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {}
                            }
                        })
                    ));
                    // content_block_delta with input_json_delta
                    let input_str = serde_json::to_string(&block["input"]).unwrap_or_default();
                    if input_str != "{}" {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": input_str}})
                        ));
                    }
                    // content_block_stop
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                _ => {}
            }
        }
    }

    // message_delta
    events.push_str(&format!(
        "event: message_delta\ndata: {}\n\n",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic["stop_reason"], "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        })
    ));

    // message_stop
    events.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");

    events
}

// --- HTTP utilities ---

async fn read_full_request(socket: &mut tokio::net::TcpStream) -> Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::with_capacity(16384);
    let mut tmp = vec![0u8; 4096];
    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let content_length = headers
                .lines()
                .find(|l| l.to_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body_read = buf.len() - (header_end + 4);
            if body_read < content_length {
                let remaining = content_length - body_read;
                let mut body_buf = vec![0u8; remaining];
                socket.read_exact(&mut body_buf).await?;
                buf.extend_from_slice(&body_buf);
            }
            break;
        }
    }
    Ok(buf)
}

fn extract_body(request: &str) -> Result<&str> {
    let pos = request
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request"))?;
    Ok(request[pos + 4..].trim_end_matches('\0').trim())
}

fn error_response(status: u16, message: &str) -> String {
    let body = json!({"error": {"message": message}}).to_string();
    format!(
        "HTTP/1.1 {} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_model_name_strips_date_and_converts_dots() {
        assert_eq!(
            copilot_model_name("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
        assert_eq!(
            copilot_model_name("claude-sonnet-4-6-20250603"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-opus-4-6-20250210"),
            "claude-opus-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-haiku-4-5-20250501"),
            "claude-haiku-4.5"
        );
    }

    #[test]
    fn test_copilot_model_name_converts_dots() {
        assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
        assert_eq!(copilot_model_name("claude-haiku-4-5"), "claude-haiku-4.5");
        assert_eq!(copilot_model_name("claude-opus-4-5"), "claude-opus-4.5");
        assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_simplify_parameters_removes_verbose_fields() {
        let verbose_schema = json!({
            "type": "object",
            "title": "My Object",
            "description": "A verbose description",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name",
                    "description": "The name field",
                    "examples": ["test", "example"],
                    "default": "unknown"
                },
                "count": {
                    "type": "integer",
                    "default": 0
                }
            },
            "required": ["name"],
            "default": {}
        });

        let simplified = simplify_parameters(&verbose_schema);
        let simplified_obj = simplified.as_object().unwrap();

        // Should keep type, properties, required
        assert_eq!(simplified_obj.get("type").unwrap(), "object");
        assert!(simplified_obj.get("properties").is_some());
        assert!(simplified_obj.get("required").is_some());

        // Should strip verbose fields
        assert!(!simplified_obj.contains_key("title"));
        assert!(!simplified_obj.contains_key("description"));
        assert!(!simplified_obj.contains_key("default"));
        assert!(!simplified_obj.contains_key("examples"));

        // Property should be simplified too
        let props = simplified_obj
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        let name_prop = props.get("name").unwrap().as_object().unwrap();
        assert!(!name_prop.contains_key("title"));
        assert!(!name_prop.contains_key("description"));
        assert!(!name_prop.contains_key("examples"));
        assert!(!name_prop.contains_key("default"));
    }

    #[test]
    fn test_anthropic_to_openai_basic() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
        });
        let result = anthropic_to_openai(&body, false);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 4); // system + 3 messages
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "Hi!");
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["stream"], false);
    }

    #[test]
    fn test_anthropic_to_openai_system_array() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "System prompt."}],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = anthropic_to_openai(&body, false);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages[0]["content"], "System prompt.");
    }

    #[test]
    fn test_anthropic_to_openai_tool_use() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 72°F"}
                ]}
            ]
        });
        let result = anthropic_to_openai(&body, false);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // Assistant message with tool calls
        assert_eq!(messages[1]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(messages[1]["content"], "Let me check.");
        // Tool result
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "toolu_1");
        assert_eq!(messages[2]["content"], "Sunny, 72°F");
    }

    #[test]
    fn test_anthropic_to_openai_tools() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
            }]
        });
        let result = anthropic_to_openai(&body, false);
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn test_anthropic_to_openai_stop_sequences() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\nHuman:"]
        });
        let result = anthropic_to_openai(&body, false);
        assert_eq!(result["stop"][0], "\n\nHuman:");
    }

    #[test]
    fn test_openai_to_anthropic_text() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let result = openai_to_anthropic(&resp, "claude-sonnet-4");
        assert_eq!(result["type"], "message");
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], 5);
        assert_eq!(result["usage"]["output_tokens"], 3);
    }

    #[test]
    fn test_openai_to_anthropic_tool_calls() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let result = openai_to_anthropic(&resp, "claude-sonnet-4");
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["id"], "call_abc");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["content"][0]["input"]["location"], "SF");
    }

    #[test]
    fn test_openai_to_anthropic_length_finish() {
        let resp = json!({
            "choices": [{"message": {"content": "..."}, "finish_reason": "length"}],
        });
        let result = openai_to_anthropic(&resp, "claude-sonnet-4");
        assert_eq!(result["stop_reason"], "max_tokens");
    }

    #[test]
    fn test_openai_to_anthropic_sse() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4");
        assert!(sse.contains("event: message_start"));
        assert!(sse.contains("event: content_block_start"));
        assert!(sse.contains("event: content_block_delta"));
        assert!(sse.contains("\"text\":\"Hi!\""));
        assert!(sse.contains("event: content_block_stop"));
        assert!(sse.contains("event: message_delta"));
        assert!(sse.contains("event: message_stop"));
    }

    #[test]
    fn test_openai_to_anthropic_sse_tool_use() {
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"test.rs\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4");
        assert!(sse.contains("\"type\":\"tool_use\""));
        assert!(sse.contains("\"name\":\"read_file\""));
        assert!(sse.contains("input_json_delta"));
    }

    #[test]
    fn test_extract_body() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(extract_body(req).unwrap(), "{\"key\":\"val\"}");
    }

    #[test]
    fn test_openai_to_anthropic_multi_choice() {
        // Copilot splits text and tool_calls into separate choices
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"content": "Let me check:", "role": "assistant"}
                },
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{\"path\":\"test.rs\"}"}
                        }]
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        });
        let result = openai_to_anthropic(&resp, "claude-sonnet-4.6");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check:");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "read_file");
        assert_eq!(result["stop_reason"], "tool_use");
    }

    #[test]
    fn test_openai_to_anthropic_sse_multi_choice() {
        // SSE should also handle multi-choice correctly
        let resp = json!({
            "id": "chatcmpl-xxx",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"content": "Checking...", "role": "assistant"}
                },
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{\"cmd\":\"ls\"}"}
                        }]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let sse = openai_to_anthropic_sse(&resp, "claude-sonnet-4.6");
        assert!(sse.contains("Checking..."));
        assert!(sse.contains("\"type\":\"tool_use\""));
        assert!(sse.contains("\"name\":\"exec\""));
        assert!(sse.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_extract_body_missing_separator() {
        assert!(extract_body("POST /v1/messages HTTP/1.1").is_err());
    }

    #[test]
    fn test_error_response() {
        let resp = error_response(500, "test error");
        assert!(resp.contains("500"));
        assert!(resp.contains("test error"));
    }

    #[test]
    fn test_tool_choice_auto() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "auto"}
        });
        let req = anthropic_to_openai(&body, false);
        assert_eq!(req["tool_choice"], json!("auto"));
    }

    #[test]
    fn test_tool_choice_any() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "any"}
        });
        let req = anthropic_to_openai(&body, false);
        assert_eq!(req["tool_choice"], json!("required"));
    }

    #[test]
    fn test_tool_choice_specific_tool() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read_file", "description": "read", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "read_file"}
        });
        let req = anthropic_to_openai(&body, false);
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_tool_choice_not_present() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req = anthropic_to_openai(&body, false);
        assert!(req.get("tool_choice").is_none());
    }
}
