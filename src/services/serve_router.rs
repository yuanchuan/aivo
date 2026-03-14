//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::commands::models::fetch_models;
use crate::services::codex_router::{
    CodexRouterConfig, convert_chat_response_to_responses_sse, convert_responses_to_chat_request,
};
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils::{self, current_unix_ts, router_http_client, sse_data_payload};
use crate::services::model_names::{copilot_model_name, transform_model_for_openrouter};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    build_google_stream_generate_content_url, convert_gemini_to_openai_chat_response,
    convert_openai_chat_to_gemini_request,
};
use crate::services::provider_protocol::{ProviderProtocol, fallback_protocols, is_protocol_mismatch};
use crate::services::session_store::ApiKey;

pub struct ServeRouterConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub upstream_protocol: ProviderProtocol,
    pub is_copilot: bool,
    pub is_openrouter: bool,
}

pub struct ServeRouter {
    config: ServeRouterConfig,
    key: ApiKey,
}

struct ServeState {
    config: Arc<ServeRouterConfig>,
    client: reqwest::Client,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
    active_protocol: Arc<AtomicU8>,
}

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    Streaming {
        status: u16,
        content_type: String,
        body: Box<StreamingBody>,
    },
}

enum StreamingBody {
    Upstream(reqwest::Response),
    Anthropic {
        upstream: reqwest::Response,
        converter: AnthropicToOpenAIStreamConverter,
    },
    Gemini {
        upstream: reqwest::Response,
        converter: GeminiToOpenAIStreamConverter,
    },
    Responses {
        source: Box<StreamingBody>,
        converter: OpenAIToResponsesStreamConverter,
    },
}

#[derive(Default)]
struct AnthropicToolCallState {
    id: String,
    name: String,
}

struct AnthropicToOpenAIStreamConverter {
    pending: String,
    id: String,
    model: String,
    fallback_model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    tool_calls: HashMap<usize, AnthropicToolCallState>,
}

struct GeminiToOpenAIStreamConverter {
    pending: String,
    id: String,
    model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    next_tool_index: usize,
}

struct OpenAIToResponsesStreamConverter {
    pending: String,
    response_id: String,
    created_at: u64,
    model: String,
    started: bool,
    completed: bool,
    text_item: Option<ResponsesTextItemState>,
    tool_calls: HashMap<usize, ResponsesToolCallState>,
    next_output_index: usize,
}

struct ResponsesTextItemState {
    item_id: String,
    output_index: usize,
    content: String,
}

struct ResponsesToolCallState {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    output_index: usize,
    started: bool,
}

enum ResponsesOutputItem {
    Message {
        item_id: String,
        output_index: usize,
        content: String,
    },
    FunctionCall {
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
        output_index: usize,
    },
}

static RESPONSES_ID_COUNTER: AtomicU64 = AtomicU64::new(0);


impl ServeRouter {
    pub fn new(config: ServeRouterConfig, key: ApiKey) -> Self {
        Self { config, key }
    }

    /// Binds to the port eagerly (propagates "address already in use" immediately),
    /// then spawns the accept loop in the background and returns the join handle.
    pub async fn start_background(self, port: u16) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

        let copilot_tokens = if self.config.is_copilot {
            Some(Arc::new(CopilotTokenManager::new(
                self.config.upstream_api_key.clone(),
            )))
        } else {
            None
        };

        let initial_protocol = self.config.upstream_protocol;

        let state = Arc::new(ServeState {
            config: Arc::new(self.config),
            client: router_http_client(),
            key: self.key,
            copilot_tokens,
            active_protocol: Arc::new(AtomicU8::new(initial_protocol.to_u8())),
        });

        Ok(tokio::spawn(run_accept_loop(listener, state)))
    }
}

async fn run_accept_loop(listener: tokio::net::TcpListener, state: Arc<ServeState>) -> Result<()> {
    loop {
        let (mut socket, _) = listener.accept().await?;
        let state = state.clone();

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
            let path = http_utils::extract_request_path(&request);
            let path_no_query = path.split('?').next().unwrap_or(&path);

            let result = match path_no_query {
                "/v1/models" | "/models" => handle_models(&state).await,
                "/v1/chat/completions" => {
                    if !request.starts_with("POST ") {
                        Ok(buffered_response(
                            405,
                            "application/json",
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_chat(&request, &state).await
                    }
                }
                "/v1/responses" | "/responses" => {
                    if !request.starts_with("POST ") {
                        Ok(buffered_response(
                            405,
                            "application/json",
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_responses(&request, &state).await
                    }
                }
                _ => Ok(buffered_response(
                    404,
                    "application/json",
                    br#"{"error":{"message":"Not found"}}"#.to_vec(),
                )),
            };

            match result {
                Ok(response) => {
                    let _ = write_router_response(&mut socket, response).await;
                }
                Err(e) => {
                    let _ = socket
                        .write_all(http_utils::http_error_response(500, &e.to_string()).as_bytes())
                        .await;
                }
            }
        });
    }
}

async fn handle_models(state: &ServeState) -> Result<RouterResponse> {
    let models = fetch_models(&state.client, &state.key).await?;
    let data: Vec<Value> = models
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "owned_by": "aivo"}))
        .collect();
    let resp = json!({"object": "list", "data": data});
    Ok(buffered_response(
        200,
        "application/json",
        resp.to_string().into_bytes(),
    ))
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    handle_chat_body(body, state).await
}

async fn handle_responses(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;
    let original_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o")
        .to_string();
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Use `actual_model` to pin the model name to the raw user-supplied value.  The config's
    // `target_protocol` is snapshotted here, before `handle_chat_body` runs the fallback loop;
    // if the loop switches protocol, any protocol-based model-name transformation done by
    // `convert_responses_to_chat_request` would have used the wrong protocol.  Setting
    // `actual_model` causes `select_model_for_protocol` to return it verbatim, so the model
    // field in `chat_body` is always the original string and `handle_chat_body` transforms it
    // for the protocol that is actually selected.
    let mut config = responses_router_config(state);
    config.actual_model = Some(original_model.clone());
    let mut chat_body = convert_responses_to_chat_request(&body, &config);
    chat_body["stream"] = json!(client_wants_stream);
    let chat_response = handle_chat_body(chat_body, state).await?;

    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(buffered_response(status, &content_type, body));
            }

            if client_wants_stream {
                let sse = if content_type.contains("text/event-stream") {
                    convert_chat_sse_to_responses_sse(std::str::from_utf8(&body)?, &original_model)?
                } else {
                    let chat_json: Value = serde_json::from_slice(&body)?;
                    convert_chat_response_to_responses_sse(&chat_json, false, &original_model)
                };
                Ok(buffered_response(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let chat_json: Value = serde_json::from_slice(&body)?;
                let response_json =
                    convert_chat_response_to_responses_json(&chat_json, &original_model)?;
                Ok(buffered_response(
                    200,
                    "application/json",
                    serde_json::to_vec(&response_json)?,
                ))
            }
        }
        RouterResponse::Streaming {
            status,
            content_type: _,
            body,
        } => {
            if !client_wants_stream {
                anyhow::bail!(
                    "internal error: responses route received streaming body for non-streaming request"
                );
            }

            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Responses {
                    source: body,
                    converter: OpenAIToResponsesStreamConverter::new(&original_model),
                }),
            })
        }
    }
}

async fn handle_chat_body(body: Value, state: &ServeState) -> Result<RouterResponse> {
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Skip fallback for copilot/openrouter — these have fixed protocols
    if state.config.is_copilot || state.config.is_openrouter {
        let mut body = body;
        return match ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed)) {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body, client_wants_stream, state).await
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body, client_wants_stream, state).await
            }
            ProviderProtocol::Openai => {
                handle_chat_openai(&mut body, client_wants_stream, state).await
            }
        };
    }

    let current = ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed));
    let candidates: Vec<ProviderProtocol> = std::iter::once(current)
        .chain(fallback_protocols(current, &state.config.upstream_base_url))
        .collect();

    let mut last_response: Option<RouterResponse> = None;
    for (attempt, protocol) in candidates.into_iter().enumerate() {
        let mut body_clone = body.clone();
        let response = match protocol {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Openai => {
                handle_chat_openai(&mut body_clone, client_wants_stream, state).await?
            }
        };

        let status = match &response {
            RouterResponse::Buffered { status, .. } => *status,
            // Streaming is only produced when the upstream returned 200 (see each handle_chat_* handler);
            // a protocol mismatch (404/405/415) always results in a Buffered error response.
            RouterResponse::Streaming { .. } => 200,
        };

        if is_protocol_mismatch(status) {
            last_response = Some(response);
            continue;
        }

        // Not a mismatch — return this response
        if attempt > 0 {
            state
                .active_protocol
                .store(protocol.to_u8(), Ordering::Relaxed);
            eprintln!("  \u{2022} Protocol auto-switched to {}", protocol.as_str());
        }
        return Ok(response);
    }

    Ok(last_response.unwrap_or(buffered_response(
        503,
        "application/json",
        br#"{"error":{"message":"No compatible protocol found"}}"#.to_vec(),
    )))
}

fn responses_router_config(state: &ServeState) -> CodexRouterConfig {
    CodexRouterConfig {
        target_base_url: state.config.upstream_base_url.clone(),
        api_key: state.config.upstream_api_key.clone(),
        target_protocol: ProviderProtocol::from_u8(state.active_protocol.load(Ordering::Relaxed)),
        copilot_token_manager: state.copilot_tokens.clone(),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let fallback_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("claude-sonnet-4-5")
        .to_string();

    let mut anthropic_req = convert_openai_chat_to_anthropic_request(
        body,
        &OpenAIToAnthropicChatConfig {
            default_model: "claude-sonnet-4-5",
        },
    );
    anthropic_req["stream"] = json!(client_wants_stream);

    let url = http_utils::build_target_url(&state.config.upstream_base_url, "/v1/messages");
    let response = state
        .client
        .post(&url)
        .header("x-api-key", state.config.upstream_api_key.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&anthropic_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Anthropic {
                upstream: response,
                converter: AnthropicToOpenAIStreamConverter::new(&fallback_model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_anthropic_to_openai_chat_response(&anthropic_resp, &fallback_model);

    if client_wants_stream {
        Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(buffered_response(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    let gemini_req = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&state.config.upstream_base_url, &model)
    } else {
        build_google_generate_content_url(&state.config.upstream_base_url, &model)
    };
    let response = state
        .client
        .post(&url)
        .header("x-goog-api-key", state.config.upstream_api_key.as_str())
        .header("Content-Type", "application/json")
        .header("User-Agent", "aivo-serve/1.0")
        .json(&gemini_req)
        .send()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Gemini {
                upstream: response,
                converter: GeminiToOpenAIStreamConverter::new(&model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_gemini_to_openai_chat_response(&gemini_resp, &model);

    if client_wants_stream {
        Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ))
    } else {
        Ok(buffered_response(
            200,
            "application/json",
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn handle_chat_openai(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    if state.config.is_openrouter {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(transform_model_for_openrouter);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    } else if state.config.is_copilot {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(copilot_model_name);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    }

    let url = http_utils::build_chat_completions_url(&state.config.upstream_base_url);
    let req = http_utils::authorized_openai_post(
        &state.client,
        &url,
        state.config.upstream_api_key.as_str(),
        state.copilot_tokens.as_deref(),
    )
    .await?;

    let response = req.json(&*body).send().await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(buffered_response(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type,
            body: Box::new(StreamingBody::Upstream(response)),
        });
    }

    let resp_body = response.text().await?;

    if client_wants_stream && let Ok(openai_resp) = serde_json::from_str::<Value>(&resp_body) {
        return Ok(buffered_response(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp).into_bytes(),
        ));
    }

    Ok(buffered_response(
        status,
        &content_type,
        resp_body.into_bytes(),
    ))
}

fn buffered_response(status: u16, content_type: &str, body: Vec<u8>) -> RouterResponse {
    RouterResponse::Buffered {
        status,
        content_type: content_type.to_string(),
        body,
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
        RouterResponse::Streaming {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_chunked_response_head(status, &content_type);
            socket.write_all(headers.as_bytes()).await?;

            match *body {
                StreamingBody::Upstream(mut upstream) => {
                    while let Some(chunk) = upstream.chunk().await? {
                        write_chunk(socket, &chunk).await?;
                    }
                }
                StreamingBody::Anthropic {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
                StreamingBody::Gemini {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
                StreamingBody::Responses {
                    source,
                    mut converter,
                } => {
                    match *source {
                        StreamingBody::Upstream(mut upstream) => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let mapped = converter.push_bytes(&chunk)?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Anthropic {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes())?;
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes())?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Gemini {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes())?;
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes())?;
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Responses { .. } => {
                            anyhow::bail!("nested responses stream sources are not supported");
                        }
                    }

                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
            }

            socket.write_all(b"0\r\n\r\n").await?;
        }
    }

    Ok(())
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, chunk: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    if chunk.is_empty() {
        return Ok(());
    }

    let chunk_len = format!("{:X}\r\n", chunk.len());
    socket.write_all(chunk_len.as_bytes()).await?;
    socket.write_all(chunk).await?;
    socket.write_all(b"\r\n").await?;
    Ok(())
}

impl AnthropicToOpenAIStreamConverter {
    fn new(fallback_model: &str) -> Self {
        Self {
            pending: String::new(),
            id: "chatcmpl-aivo".to_string(),
            model: String::new(),
            fallback_model: fallback_model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            tool_calls: HashMap::new(),
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "message_start" => {
                if let Some(message) = event.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str())
                        && !id.is_empty()
                    {
                        self.id = id.to_string();
                    }
                    if let Some(model) = message.get("model").and_then(|v| v.as_str())
                        && !model.is_empty()
                    {
                        self.model = model.to_string();
                    }
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if event
                    .get("content_block")
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    let block = event
                        .get("content_block")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| format!("call_{index}"));
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.tool_calls.insert(
                        index,
                        AnthropicToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                        },
                    );
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        self.model_name(),
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name
                                }
                            }]
                        }),
                        Value::Null,
                    ));
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").cloned().unwrap_or_else(|| json!({}));
                match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                self.model_name(),
                                json!({ "content": text }),
                                Value::Null,
                            ));
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) =
                            delta.get("partial_json").and_then(|v| v.as_str())
                        {
                            let (id, name) = {
                                let tool = self.tool_calls.entry(index).or_default();
                                let id = if tool.id.is_empty() {
                                    format!("call_{index}")
                                } else {
                                    tool.id.clone()
                                };
                                (id, tool.name.clone())
                            };
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                self.model_name(),
                                json!({
                                    "tool_calls": [{
                                        "index": index,
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": partial_json
                                        }
                                    }]
                                }),
                                Value::Null,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(stop_reason) = event
                    .get("delta")
                    .and_then(|v| v.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    self.emit_finish(output, map_anthropic_stop_reason(stop_reason));
                }
            }
            "message_stop" => {
                if !self.finished {
                    let finish_reason = if self.saw_tool_call {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    self.emit_finish(output, finish_reason);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            self.model_name(),
            json!({ "role": "assistant" }),
            Value::Null,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            self.model_name(),
            json!({}),
            json!(finish_reason),
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }

    fn model_name(&self) -> &str {
        if self.model.is_empty() {
            &self.fallback_model
        } else {
            &self.model
        }
    }
}

impl GeminiToOpenAIStreamConverter {
    fn new(model: &str) -> Self {
        Self {
            pending: String::new(),
            id: "chatcmpl-aivo".to_string(),
            model: model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            next_tool_index: 0,
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        if let Some(id) = event.get("responseId").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            self.id = id.to_string();
        }

        let candidate = event
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| json!({}));

        if let Some(parts) = candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({ "content": text }),
                        Value::Null,
                    ));
                }

                if let Some(function_call) = part.get("functionCall") {
                    let index = self.next_tool_index;
                    self.next_tool_index += 1;
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": function_call
                                    .get("id")
                                    .cloned()
                                    .unwrap_or_else(|| json!(format!("call_{index}"))),
                                "type": "function",
                                "function": {
                                    "name": function_call.get("name").cloned().unwrap_or_else(|| json!("")),
                                    "arguments": serde_json::to_string(
                                        &function_call.get("args").cloned().unwrap_or_else(|| json!({}))
                                    ).unwrap_or_else(|_| "{}".to_string())
                                }
                            }]
                        }),
                        Value::Null,
                    ));
                }
            }
        }

        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str())
            && !reason.is_empty()
        {
            self.emit_finish(output, map_gemini_finish_reason(reason, self.saw_tool_call));
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({ "role": "assistant" }),
            Value::Null,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({}),
            json!(finish_reason),
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }
}

fn openai_sse_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Value,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        })
    )
}

fn map_anthropic_stop_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    }
}

fn map_gemini_finish_reason(finish_reason: &str, saw_tool_call: bool) -> &'static str {
    if saw_tool_call {
        return "tool_calls";
    }

    match finish_reason {
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        _ => "stop",
    }
}

impl OpenAIToResponsesStreamConverter {
    fn new(original_model: &str) -> Self {
        Self {
            pending: String::new(),
            response_id: next_responses_id("resp"),
            created_at: current_unix_ts(),
            model: original_model.to_string(),
            started: false,
            completed: false,
            text_item: None,
            tool_calls: HashMap::new(),
            next_output_index: 0,
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.completed {
            self.finalize(&mut output);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.completed {
                self.finalize(output);
            }
            return Ok(());
        }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        if let Some(model) = chunk.get("model").and_then(|v| v.as_str())
            && !model.is_empty()
            && self.model.is_empty()
        {
            self.model = model.to_string();
        }

        let choice = chunk
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| json!({}));
        let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));

        if !delta.is_null() {
            self.ensure_started(output);
        }

        if let Some(text) = delta.get("content").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            self.ensure_text_item(output);
            if let Some(text_item) = self.text_item.as_mut() {
                text_item.content.push_str(text);
                output.push_str(&responses_sse_event(
                    "response.output_text.delta",
                    json!({
                        "type": "response.output_text.delta",
                        "response_id": self.response_id,
                        "item_id": text_item.item_id,
                        "output_index": text_item.output_index,
                        "content_index": 0,
                        "delta": text
                    }),
                ));
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let index = tool_call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if !self.tool_calls.contains_key(&index) {
                    let output_index = self.take_output_index();
                    self.tool_calls.insert(
                        index,
                        ResponsesToolCallState {
                            item_id: next_responses_id("fc"),
                            call_id: tool_call
                                .get("id")
                                .and_then(|v| v.as_str())
                                .filter(|v| !v.is_empty())
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("call_{index}")),
                            name: String::new(),
                            arguments: String::new(),
                            output_index,
                            started: false,
                        },
                    );
                }
                let state = self
                    .tool_calls
                    .get_mut(&index)
                    .expect("tool state inserted");

                if let Some(call_id) = tool_call.get("id").and_then(|v| v.as_str())
                    && !call_id.is_empty()
                {
                    state.call_id = call_id.to_string();
                }
                if let Some(name) = tool_call
                    .get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    && !name.is_empty()
                {
                    state.name = name.to_string();
                }

                if !state.started {
                    output.push_str(&responses_sse_event(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": state.output_index,
                            "item": {
                                "id": state.item_id,
                                "call_id": state.call_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "name": state.name,
                                "arguments": state.arguments
                            }
                        }),
                    ));
                    state.started = true;
                }

                if let Some(arguments) = tool_call
                    .get("function")
                    .and_then(|v| v.get("arguments"))
                    .and_then(|v| v.as_str())
                    && !arguments.is_empty()
                {
                    state.arguments.push_str(arguments);
                    output.push_str(&responses_sse_event(
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "response_id": self.response_id,
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "delta": arguments
                        }),
                    ));
                }
            }
        }

        if let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str())
            && !finish_reason.is_empty()
        {
            self.finalize(output);
        }

        Ok(())
    }

    fn ensure_started(&mut self, output: &mut String) {
        if self.started {
            return;
        }
        self.started = true;
        output.push_str(&responses_sse_event(
            "response.created",
            json!({
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "model": self.model,
                    "created_at": self.created_at,
                    "status": "in_progress",
                    "output": []
                }
            }),
        ));
    }

    fn ensure_text_item(&mut self, output: &mut String) {
        if self.text_item.is_some() {
            return;
        }

        let output_index = self.take_output_index();
        let item_id = next_responses_id("msg");
        output.push_str(&responses_sse_event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            }),
        ));
        output.push_str(&responses_sse_event(
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        self.text_item = Some(ResponsesTextItemState {
            item_id,
            output_index,
            content: String::new(),
        });
    }

    fn finalize(&mut self, output: &mut String) {
        if self.completed {
            return;
        }

        self.ensure_started(output);

        if self.text_item.is_none() && self.tool_calls.is_empty() {
            self.ensure_text_item(output);
        }

        if let Some(text_item) = self.text_item.as_ref() {
            output.push_str(&responses_sse_event(
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "response_id": self.response_id,
                    "item_id": text_item.item_id,
                    "output_index": text_item.output_index,
                    "content_index": 0,
                    "text": text_item.content
                }),
            ));
            output.push_str(&responses_sse_event(
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": text_item.item_id,
                    "output_index": text_item.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_item.content}
                }),
            ));
            output.push_str(&responses_sse_event(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "output_index": text_item.output_index,
                    "item": {
                        "id": text_item.item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": text_item.content,
                            "annotations": []
                        }]
                    }
                }),
            ));
        }

        let mut tool_indexes: Vec<usize> = self.tool_calls.keys().copied().collect();
        tool_indexes.sort_unstable();
        for index in tool_indexes {
            if let Some(tool_call) = self.tool_calls.get(&index) {
                output.push_str(&responses_sse_event(
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "response_id": self.response_id,
                        "output_index": tool_call.output_index,
                        "item_id": tool_call.item_id,
                        "arguments": tool_call.arguments
                    }),
                ));
                output.push_str(&responses_sse_event(
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "response_id": self.response_id,
                        "output_index": tool_call.output_index,
                        "item": {
                            "id": tool_call.item_id,
                            "call_id": tool_call.call_id,
                            "type": "function_call",
                            "status": "completed",
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                        }
                    }),
                ));
            }
        }

        let output_items = self.output_items();
        output.push_str(&responses_sse_event(
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "model": self.model,
                    "created_at": self.created_at,
                    "status": "completed",
                    "output": output_items
                }
            }),
        ));

        self.completed = true;
    }

    fn take_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn output_items(&self) -> Vec<Value> {
        let mut items = Vec::new();

        if let Some(text_item) = self.text_item.as_ref() {
            items.push(ResponsesOutputItem::Message {
                item_id: text_item.item_id.clone(),
                output_index: text_item.output_index,
                content: text_item.content.clone(),
            });
        }

        let mut tool_items: Vec<ResponsesOutputItem> = self
            .tool_calls
            .values()
            .map(|tool_call| ResponsesOutputItem::FunctionCall {
                item_id: tool_call.item_id.clone(),
                call_id: tool_call.call_id.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
                output_index: tool_call.output_index,
            })
            .collect();
        items.append(&mut tool_items);
        items.sort_by_key(|item| match item {
            ResponsesOutputItem::Message { output_index, .. } => *output_index,
            ResponsesOutputItem::FunctionCall { output_index, .. } => *output_index,
        });

        items
            .into_iter()
            .map(|item| match item {
                ResponsesOutputItem::Message {
                    item_id, content, ..
                } => json!({
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": content,
                        "annotations": []
                    }]
                }),
                ResponsesOutputItem::FunctionCall {
                    item_id,
                    call_id,
                    name,
                    arguments,
                    ..
                } => json!({
                    "id": item_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": name,
                    "arguments": arguments
                }),
            })
            .collect()
    }
}

fn convert_chat_response_to_responses_json(chat: &Value, original_model: &str) -> Result<Value> {
    let sse = convert_chat_response_to_responses_sse(chat, false, original_model);
    extract_completed_response_from_sse(&sse)
        .ok_or_else(|| anyhow::anyhow!("failed to synthesize responses JSON payload"))
}

fn convert_chat_sse_to_responses_sse(chat_sse: &str, original_model: &str) -> Result<String> {
    let mut converter = OpenAIToResponsesStreamConverter::new(original_model);
    let mut output = converter.push_bytes(chat_sse.as_bytes())?;
    output.push_str(&converter.finish()?);
    Ok(output)
}

fn responses_sse_event(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

fn next_responses_id(prefix: &str) -> String {
    let count = RESPONSES_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{}_{count}", current_unix_ts())
}

fn extract_completed_response_from_sse(sse: &str) -> Option<Value> {
    let mut saw_completed_event = false;

    for line in sse.lines() {
        if let Some(event) = line.strip_prefix("event:") {
            saw_completed_event = event.trim() == "response.completed";
            continue;
        }

        if saw_completed_event && let Some(data) = sse_data_payload(line) {
            let payload: Value = serde_json::from_str(data).ok()?;
            return payload.get("response").cloned();
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_stream_converter_emits_openai_sse() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let input = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":12}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"shell\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hello\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"toolu_1\""));
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_gemini_stream_converter_emits_openai_sse() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = concat!(
            "data: {\"responseId\":\"resp_1\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call_1\",\"name\":\"shell\",\"args\":{\"cmd\":\"ls\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hel\""));
        assert!(output.contains("\"content\":\"lo\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"call_1\""));
        assert!(output.contains("\"index\":0"));
        assert!(output.contains("\"name\":\"shell\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_convert_chat_response_to_responses_json_text() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from responses"}
            }]
        });

        let response = convert_chat_response_to_responses_json(&chat, "gpt-4o").unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "gpt-4o");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "Hello from responses"
        );
    }

    #[test]
    fn test_convert_chat_response_to_responses_json_tool_call() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let response = convert_chat_response_to_responses_json(&chat, "gpt-4o").unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "call_123");
        assert_eq!(response["output"][0]["name"], "shell");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_text() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "gpt-4o").unwrap();

        assert!(responses_sse.contains("event: response.created"));
        assert!(responses_sse.contains("event: response.output_text.delta"));
        assert!(responses_sse.contains("\"delta\":\"Hel\""));
        assert!(responses_sse.contains("\"delta\":\"lo\""));
        assert!(responses_sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_tool_call() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "gpt-4o").unwrap();

        assert!(responses_sse.contains("event: response.output_item.added"));
        assert!(responses_sse.contains("event: response.function_call_arguments.delta"));
        assert!(responses_sse.contains("\"call_id\":\"call_abc\""));
        assert!(responses_sse.contains("\"delta\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(responses_sse.contains("event: response.completed"));
    }
}
