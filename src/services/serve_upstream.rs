use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_route_pipeline::inject_anthropic_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::effort::gpt5_chat_completions_rejects_tools_with_none_reasoning;
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::model_names::{
    copilot_model_name, requires_max_completion_tokens, transform_model_for_openrouter,
};
use crate::services::openai_anthropic_bridge::convert_openai_chat_response_to_sse;
use crate::services::openai_gemini_bridge::{
    build_google_generate_content_url, build_google_stream_generate_content_url,
};
use crate::services::opencode_session;
use crate::services::wire_format::{
    RequestOptions, ResponseOptions, StreamAdapter, StreamOptions, stream_adapter,
    translate_request, translate_response,
};

#[derive(Clone)]
pub(crate) struct UpstreamRequestContext {
    pub(crate) client: reqwest::Client,
    pub(crate) upstream_base_url: String,
    pub(crate) upstream_api_key: String,
    pub(crate) is_copilot: bool,
    pub(crate) is_openrouter: bool,
    pub(crate) is_starter: bool,
    pub(crate) copilot_tokens: Option<Arc<CopilotTokenManager>>,
    pub(crate) grok_tokens: Option<Arc<crate::services::grok_oauth::GrokTokenManager>>,
    pub(crate) kimi_tokens: Option<Arc<crate::services::kimi_oauth::KimiTokenManager>>,
    pub(crate) codex_tokens: Option<Arc<crate::services::codex_oauth::CodexTokenManager>>,
    /// Usage accounting is on — streamed OpenAI requests must ask for the
    /// trailing usage chunk or the sniffer records zero for the turn.
    pub(crate) accounting: bool,
}

impl UpstreamRequestContext {
    /// Conditionally attaches device fingerprint headers for starter endpoint requests.
    fn with_device_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if !self.is_starter {
            return builder;
        }
        device_fingerprint::with_starter_headers(builder)
    }
}

pub(crate) enum RouterResponse {
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

pub(crate) enum StreamingBody {
    Upstream(reqwest::Response),
    /// Upstream SSE converted to client shape via wire-format adapters.
    Converted {
        upstream: reqwest::Response,
        adapter: Box<dyn StreamAdapter + Send>,
    },
}

impl RouterResponse {
    pub(crate) fn buffered(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Self::Buffered {
            status,
            content_type: content_type.to_string(),
            body,
        }
    }
}

pub(crate) async fn send_anthropic_chat(
    body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let (fallback_model, anthropic_req) = build_anthropic_request(body, client_wants_stream);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/messages");
    let response = context
        .with_device_headers(opencode_session::with_session_header(
            context
                .client
                .post(&url)
                .header("x-api-key", context.upstream_api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", CONTENT_TYPE_JSON),
            &url,
            Some(&anthropic_req),
        ))
        .json(&anthropic_req)
        .send_logged()
        .await?;

    finalize_anthropic_response(response, client_wants_stream, &fallback_model).await
}

pub(crate) async fn send_gemini_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    let gemini_req = translate_request(
        body,
        &RequestOptions::ChatToGemini {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&context.upstream_base_url, &model)
    } else {
        build_google_generate_content_url(&context.upstream_base_url, &model)
    };
    let response = context
        .with_device_headers(
            context
                .client
                .post(&url)
                .header("x-goog-api-key", context.upstream_api_key.as_str())
                .header("Content-Type", CONTENT_TYPE_JSON),
        )
        .json(&gemini_req)
        .send_logged()
        .await?;

    finalize_gemini_response(response, client_wants_stream, &model).await
}

/// Posts an already-Gemini-shaped body, returning the RAW Gemini response — the
/// direct `Anthropic → Gemini` edge converts it itself; streaming passes through.
pub(crate) async fn send_gemini_native(
    gemini_req: &Value,
    model: &str,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&context.upstream_base_url, model)
    } else {
        build_google_generate_content_url(&context.upstream_base_url, model)
    };
    let response = context
        .with_device_headers(
            context
                .client
                .post(&url)
                .header("x-goog-api-key", context.upstream_api_key.as_str())
                .header("Content-Type", CONTENT_TYPE_JSON),
        )
        .json(gemini_req)
        .send_logged()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }
    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Upstream(response)),
        });
    }
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        response.bytes().await?.to_vec(),
    ))
}

/// Posts an already-Anthropic-shaped body to `/v1/messages`, returning the RAW
/// response — the reverse direct edge converts it to Gemini shape itself.
pub(crate) async fn send_anthropic_native(
    anthropic_req: &Value,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let mut anthropic_req = anthropic_req.clone();
    anthropic_req["stream"] = json!(false);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/messages");
    let response = context
        .with_device_headers(opencode_session::with_session_header(
            context
                .client
                .post(&url)
                .header("x-api-key", context.upstream_api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", CONTENT_TYPE_JSON),
            &url,
            Some(&anthropic_req),
        ))
        .json(&anthropic_req)
        .send_logged()
        .await?;

    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    Ok(RouterResponse::buffered(
        status,
        &content_type,
        response.bytes().await?.to_vec(),
    ))
}

/// Forwards a Claude Code request verbatim (headers included — the edge
/// inspects them) to the Anthropic native backend, authenticating with the
/// stored subscription OAuth token.
pub(crate) async fn send_claude_oauth_passthrough(
    request: &str,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let token = match crate::services::claude_oauth::ClaudeOAuthCredential::from_json(
        &context.upstream_api_key,
    ) {
        Ok(creds) => creds.token,
        Err(_) => {
            return Ok(RouterResponse::buffered(
                500,
                CONTENT_TYPE_JSON,
                br#"{"error":{"message":"Claude OAuth credential unreadable - re-run `aivo keys add claude`"}}"#
                    .to_vec(),
            ));
        }
    };

    // Raw target keeps the query string (`/v1/messages?beta=true`).
    let target = http_utils::extract_request_path(request);
    let url = http_utils::build_target_url(&context.upstream_base_url, &target);

    let mut headers = http_utils::extract_passthrough_headers(request)?;
    ensure_oauth_beta(&mut headers);
    if !headers.contains_key("anthropic-version") {
        headers.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static("2023-06-01"),
        );
    }

    let is_post = request.starts_with("POST ");
    let mut builder = if is_post {
        context.client.post(&url)
    } else {
        context.client.get(&url)
    };
    builder = builder
        .headers(headers)
        .header("Authorization", format!("Bearer {token}"));
    if is_post {
        builder = builder
            .header("Content-Type", CONTENT_TYPE_JSON)
            .body(http_utils::extract_request_body(request)?.to_string());
    }

    let response = builder.send_logged().await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    if status < 400 && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Upstream(response)),
        });
    }
    Ok(RouterResponse::buffered(
        status,
        &content_type,
        response.bytes().await?.to_vec(),
    ))
}

/// Merges the OAuth beta flag into `anthropic-beta` — Claude Code omits it
/// when the subscription is only a tier behind an API-key main.
fn ensure_oauth_beta(headers: &mut reqwest::header::HeaderMap) {
    use crate::services::claude_oauth::ANTHROPIC_OAUTH_BETA;
    let merged = match headers.get("anthropic-beta").and_then(|v| v.to_str().ok()) {
        Some(existing) => {
            if existing
                .split(',')
                .any(|b| b.trim() == ANTHROPIC_OAUTH_BETA)
            {
                return;
            }
            format!("{ANTHROPIC_OAUTH_BETA},{existing}")
        }
        None => ANTHROPIC_OAUTH_BETA.to_string(),
    };
    if let Ok(value) = reqwest::header::HeaderValue::from_str(&merged) {
        headers.insert("anthropic-beta", value);
    }
}

pub(crate) async fn send_openai_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    normalize_openai_request_model(body, context.is_openrouter, context.is_copilot);
    http_utils::normalize_developer_role_to_system(body);
    migrate_max_tokens_for_reasoning_models(body);
    strip_non_function_tools(body);
    inject_include_usage_for_accounting(body, context.accounting);
    // Surface OpenAI's GPT-5.4 Chat Completions restriction (no tools when
    // reasoning_effort is "none") with a clear local 400 instead of letting
    // the upstream reject and producing a generic error the user has to
    // decode. The Responses API lifts this restriction; only Chat
    // Completions is affected here.
    if gpt5_chat_completions_rejects_tools_with_none_reasoning(body) {
        let body = serde_json::to_vec(&serde_json::json!({
            "error": {
                "message": "GPT-5.4+ Chat Completions does not support tools with reasoning_effort: \"none\". Switch to a higher effort or use the Responses API.",
                "type": "invalid_request_error",
                "code": "tools_require_reasoning_effort"
            }
        }))?;
        return Ok(RouterResponse::buffered(400, "application/json", body));
    }

    // Inception Mercury doesn't reliably stream `tool_calls`; the model narrates
    // tool intent in `delta.content` instead. Force a non-streamed upstream call
    // when tools are present — `finalize_openai_response` will buffer the JSON
    // and re-emit it as SSE so the inbound client still sees a stream.
    disable_stream_for_inception_with_tools(body, &context.upstream_base_url);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/chat/completions");
    let initiator = if context.is_copilot {
        Some(http_utils::copilot_initiator_from_openai(body))
    } else {
        None
    };

    // Grok's proxy routes by the request model via `x-grok-model-override`.
    let grok_model = context.grok_tokens.as_ref().and_then(|_| {
        body.get("model")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
    });

    let response = {
        let response =
            send_openai_chat_once(context, &url, body, initiator, grok_model.as_deref()).await?;
        // Standard SuperGrok tiers can 403; latch the XAI_API_KEY fallback and
        // retry once.
        if response.status().as_u16() == 403
            && let Some(gtm) = context.grok_tokens.as_ref()
            && gtm.mark_gated().await
        {
            send_openai_chat_once(context, &url, body, initiator, grok_model.as_deref()).await?
        } else {
            response
        }
    };
    finalize_openai_response(response, client_wants_stream).await
}

/// Builds and sends a single OpenAI-chat upstream request. Split out so the
/// SuperGrok 403→API-key path can retry the identical send.
async fn send_openai_chat_once(
    context: &UpstreamRequestContext,
    url: &str,
    body: &Value,
    initiator: Option<&str>,
    grok_model: Option<&str>,
) -> Result<reqwest::Response> {
    let mut req = http_utils::authorized_openai_post(
        &context.client,
        url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        context.grok_tokens.as_deref(),
        context.kimi_tokens.as_deref(),
        initiator,
    )
    .await?;
    if context.is_copilot && http_utils::body_requests_vision(body) {
        req = req.header("Copilot-Vision-Request", "true");
    }
    if let Some(model) = grok_model {
        req = req.header(crate::services::grok_oauth::MODEL_OVERRIDE_HEADER, model);
    }
    req = opencode_session::with_session_header(req, url, Some(body));
    Ok(context
        .with_device_headers(req)
        .json(body)
        .send_logged()
        .await?)
}

/// When usage accounting is on, streamed OpenAI upstreams only emit the
/// trailing usage chunk if asked; without this the sniffer records zero
/// tokens for the turn. Client-provided stream_options win.
fn inject_include_usage_for_accounting(body: &mut Value, accounting: bool) {
    if accounting
        && body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        && body.get("stream_options").is_none()
    {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
}

/// True when a Copilot `/chat/completions` error demands the Responses API.
pub(crate) fn copilot_requires_responses_api(body: &[u8]) -> bool {
    let s = String::from_utf8_lossy(body).to_lowercase();
    s.contains("unsupported_api_for_model")
        || (s.contains("/responses") && s.contains("chat/completions"))
}

/// Sends a Copilot chat request as a non-streamed `/responses` call, converting
/// the result back to Chat Completions (re-emitting SSE when a stream was asked).
/// Self-contained: normalizes the model itself, so callers may pass a raw body.
pub(crate) async fn send_copilot_responses(
    chat_body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let mut chat_body = chat_body.clone();
    normalize_openai_request_model(&mut chat_body, false, true);
    strip_non_function_tools(&mut chat_body);
    let responses_body = translate_request(&chat_body, &RequestOptions::ChatToResponses);
    // Bare path, not a URL: Copilot's endpoint comes from the token exchange, and
    // a URL built from the "copilot" sentinel base wouldn't parse to `/responses`.
    let initiator = http_utils::copilot_initiator_from_openai(&chat_body);
    let mut req = http_utils::authorized_openai_post(
        &context.client,
        "/v1/responses",
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        None,
        None,
        Some(initiator),
    )
    .await?;
    let responses_value = serde_json::to_value(&responses_body)?;
    if http_utils::body_requests_vision(&responses_value) {
        req = req.header("Copilot-Vision-Request", "true");
    }

    let response = context
        .with_device_headers(req)
        .json(&responses_body)
        .send_logged()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    let text = response.text().await?;
    if status != 200 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            text.into_bytes(),
        ));
    }

    let responses_json: Value = serde_json::from_str(&text)?;
    let chat_json = translate_response(&responses_json, &ResponseOptions::ChatToResponses)?;
    if client_wants_stream {
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&chat_json)?.into_bytes(),
        ));
    }
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        serde_json::to_vec(&chat_json)?,
    ))
}

/// Sends a chat request to the ChatGPT Codex Responses-API backend with the
/// Codex OAuth token, converting request/response via the wire-format registry.
/// Cousin of `send_copilot_responses` for the ChatGPT host + auth.
pub(crate) async fn send_codex_responses(
    chat_body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    use crate::services::codex_oauth;

    let Some(ctm) = context.codex_tokens.as_ref() else {
        return Ok(RouterResponse::buffered(
            500,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"codex token manager missing"}}"#.to_vec(),
        ));
    };
    let auth = ctm.authorize().await?;

    let mut chat_body = chat_body.clone();
    normalize_codex_request_model(&mut chat_body);
    strip_non_function_tools(&mut chat_body);
    let model = chat_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(codex_oauth::DEFAULT_CODEX_MODEL)
        .to_string();

    let mut responses_body = serde_json::to_value(translate_request(
        &chat_body,
        &RequestOptions::ChatToResponses,
    ))?;
    // Backend requires store:false + streaming, and rejects max_output_tokens.
    // (`cache_control` is stripped upstream in the Chat→Responses converter.)
    responses_body["store"] = json!(false);
    responses_body["stream"] = json!(true);
    merge_reasoning_include(&mut responses_body);
    if let Some(obj) = responses_body.as_object_mut() {
        obj.remove("max_output_tokens");
    }
    // Cache affinity (see `process_session_id`): a per-request id measured 5 of 23
    // full prompt-cache misses across one 80k-token agent turn.
    let session_id = codex_oauth::process_session_id();
    responses_body["prompt_cache_key"] = json!(codex_cache_key(&chat_body, session_id));

    let mut req = context
        .client
        .post(codex_oauth::CHATGPT_RESPONSES_URL)
        .header("Authorization", format!("Bearer {}", auth.access_token))
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Accept", "text/event-stream")
        .header(
            codex_oauth::OPENAI_BETA_HEADER,
            codex_oauth::OPENAI_BETA_VALUE,
        )
        .header(
            codex_oauth::ORIGINATOR_HEADER,
            codex_oauth::ORIGINATOR_VALUE,
        )
        .header(codex_oauth::SESSION_ID_HEADER, session_id)
        .header("User-Agent", codex_oauth::CODEX_USER_AGENT);
    if let Some(account_id) = auth.account_id.as_deref() {
        req = req.header(codex_oauth::ACCOUNT_ID_HEADER, account_id);
    }

    let response = req.json(&responses_body).send_logged().await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    // Always an SSE stream (we force stream:true), though the backend mislabels
    // it `application/json` — so don't gate on content-type.
    if client_wants_stream {
        return Ok(RouterResponse::Streaming {
            status: 200,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Converted {
                upstream: response,
                adapter: stream_adapter(StreamOptions::ChatToResponses {
                    model: &model,
                    include_usage: context.accounting,
                }),
            }),
        });
    }

    let text = response.text().await?;
    let responses_json = responses_object_from_body(&text)?;
    let chat_json = translate_response(&responses_json, &ResponseOptions::ChatToResponses)?;
    if client_wants_stream {
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&chat_json)?.into_bytes(),
        ));
    }
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        serde_json::to_vec(&chat_json)?,
    ))
}

/// Passes through `gpt-*` slugs (not bare `gpt-5`, which 400s); maps foreign
/// slots to the default codex model.
fn normalize_codex_request_model(body: &mut Value) {
    let keep = body
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.starts_with("gpt-") && m != "gpt-5");
    if !keep {
        body["model"] = json!(crate::services::codex_oauth::DEFAULT_CODEX_MODEL);
    }
}

/// The client's own `prompt_cache_key` when it sent one, else the process id.
fn codex_cache_key(chat_body: &Value, fallback: &str) -> String {
    chat_body
        .get("prompt_cache_key")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// Idempotently adds `reasoning.encrypted_content` to `include` for stateless
/// multi-turn reasoning continuity.
fn merge_reasoning_include(body: &mut Value) {
    const NEEDLE: &str = "reasoning.encrypted_content";
    let arr = body.as_object_mut().and_then(|o| {
        o.entry("include")
            .or_insert_with(|| json!([]))
            .as_array_mut()
    });
    if let Some(arr) = arr
        && !arr.iter().any(|v| v.as_str() == Some(NEEDLE))
    {
        arr.push(json!(NEEDLE));
    }
}

/// Reduces a codex SSE stream (or direct JSON) to one Responses object. Under
/// `store:false` the `response.completed` event's `output` is empty, so rebuild
/// it from the incremental `response.output_item.done` events.
fn responses_object_from_body(body: &str) -> Result<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(body)
        && (v.get("output").is_some() || v.get("type").is_some())
    {
        return Ok(v);
    }
    let mut completed: Option<Value> = None;
    let mut items: Vec<Value> = Vec::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match event.get("type").and_then(|t| t.as_str()) {
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    items.push(item.clone());
                }
            }
            Some("response.completed") => {
                if let Some(response) = event.get("response") {
                    completed = Some(response.clone());
                }
            }
            _ => {}
        }
    }
    let mut response = completed.ok_or_else(|| {
        anyhow::anyhow!("codex response stream missing a response.completed event")
    })?;
    let output_empty = response
        .get("output")
        .and_then(|o| o.as_array())
        .is_none_or(|a| a.is_empty());
    if output_empty && !items.is_empty() {
        response["output"] = Value::Array(items);
    }
    Ok(response)
}

pub(crate) async fn send_openai_embeddings(
    body: &Value,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/embeddings");
    let req = http_utils::authorized_openai_post(
        &context.client,
        &url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        None,
        None,
        None,
    )
    .await?;
    let req = opencode_session::with_session_header(req, &url, None);

    let response = context
        .with_device_headers(req)
        .json(body)
        .send_logged()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    let body_bytes = response.bytes().await?.to_vec();
    Ok(RouterResponse::buffered(status, &content_type, body_bytes))
}

fn build_anthropic_request(body: &Value, client_wants_stream: bool) -> (String, Value) {
    let fallback_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("claude-sonnet-4-5")
        .to_string();

    let mut anthropic_req = translate_request(
        body,
        &RequestOptions::ChatToAnthropic {
            default_model: "claude-sonnet-4-5",
        },
    );
    // Only inject cache_control for Claude models — other providers don't
    // honor it (e.g. Gemini uses a different caching model) and strict ones
    // reject the unknown field outright.
    if body
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("claude"))
    {
        inject_anthropic_cache_control(&mut anthropic_req);
    }
    if crate::services::trace::enabled("anthropic") {
        crate::services::trace::line(
            "anthropic",
            &format!(
                "event=cache_breakpoints {}",
                describe_cache_breakpoints(&anthropic_req)
            ),
        );
    }
    if crate::services::trace::enabled("anthropic-body") {
        crate::services::trace::line(
            "anthropic-body",
            &serde_json::to_string(&anthropic_req).unwrap_or_default(),
        );
    }
    anthropic_req["stream"] = json!(client_wants_stream);

    (fallback_model, anthropic_req)
}

/// Coordinates of every `cache_control` breakpoint, for the `anthropic` trace scope.
fn describe_cache_breakpoints(req: &Value) -> String {
    let mut spots = Vec::new();
    if let Some(blocks) = req.get("system").and_then(Value::as_array) {
        for (j, b) in blocks.iter().enumerate() {
            if b.get("cache_control").is_some() {
                spots.push(format!("system[{j}]"));
            }
        }
    }
    if let Some(msgs) = req.get("messages").and_then(Value::as_array) {
        for (i, m) in msgs.iter().enumerate() {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
            if let Some(blocks) = m.get("content").and_then(Value::as_array) {
                for (j, b) in blocks.iter().enumerate() {
                    if b.get("cache_control").is_some() {
                        let ty = b.get("type").and_then(Value::as_str).unwrap_or("?");
                        spots.push(format!("messages[{i}:{role}].content[{j}:{ty}]"));
                    }
                }
            }
        }
    }
    let n = req
        .get("messages")
        .and_then(Value::as_array)
        .map_or(0, |m| m.len());
    format!("messages={n} breakpoints={}", spots.join(","))
}

/// Rename the legacy `max_tokens` field to `max_completion_tokens` when the
/// target model is in OpenAI's reasoning family (o-series / GPT-5+ / Codex).
/// The Chat Completions API rejects `max_tokens` on those models with a 400.
/// If `max_completion_tokens` is already present, the legacy field is removed
/// to avoid the upstream rejecting both being set.
fn migrate_max_tokens_for_reasoning_models(body: &mut Value) {
    let model = match body.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return,
    };
    if !requires_max_completion_tokens(&model) {
        return;
    }
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let legacy = obj.remove("max_tokens");
    if obj.contains_key("max_completion_tokens") {
        return;
    }
    if let Some(value) = legacy {
        obj.insert("max_completion_tokens".to_string(), value);
    }
}

/// OpenAI Chat Completions only accepts `tools[].type == "function"`. Server
/// tools like `{type:"web_search"}` (Anthropic/Responses-native) reach this
/// passthrough when a model is served over an OpenAI-compatible gateway — e.g.
/// a `claude-*` model on a third-party endpoint — and 400 ("expected function").
/// Drop them so the request succeeds; the Anthropic/Gemini bridges (separate
/// paths) still translate these server tools natively.
pub(crate) fn strip_non_function_tools(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
    }
    if body
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|a| a.is_empty())
        && let Some(obj) = body.as_object_mut()
    {
        obj.remove("tools");
        // A `tool_choice` with no tools 400s the OpenAI Chat upstream.
        obj.remove("tool_choice");
    }
}

pub(crate) fn disable_stream_for_inception_with_tools(body: &mut Value, upstream_base_url: &str) {
    let url_matches = upstream_base_url.contains("inceptionlabs.ai");
    let model_matches = body
        .get("model")
        .and_then(|v| v.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("mercury"));
    if !url_matches && !model_matches {
        return;
    }
    let has_tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    if !has_tools {
        return;
    }
    body["stream"] = json!(false);
    if let Some(obj) = body.as_object_mut() {
        obj.remove("stream_options");
    }
}

fn normalize_openai_request_model(body: &mut Value, is_openrouter: bool, is_copilot: bool) {
    if is_openrouter {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(transform_model_for_openrouter);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    } else if is_copilot {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(copilot_model_name);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    }
}

async fn finalize_anthropic_response(
    response: reqwest::Response,
    client_wants_stream: bool,
    fallback_model: &str,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Converted {
                upstream: response,
                adapter: stream_adapter(StreamOptions::ChatToAnthropic {
                    model: fallback_model,
                }),
            }),
        });
    }

    let resp_body = response.text().await?;
    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = translate_response(
        &anthropic_resp,
        &ResponseOptions::ChatToAnthropic {
            model: fallback_model,
        },
    )?;

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            CONTENT_TYPE_JSON,
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn finalize_gemini_response(
    response: reqwest::Response,
    client_wants_stream: bool,
    model: &str,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Converted {
                upstream: response,
                adapter: stream_adapter(StreamOptions::ChatToGemini { model }),
            }),
        });
    }

    let resp_body = response.text().await?;
    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = translate_response(&gemini_resp, &ResponseOptions::ChatToGemini { model })?;

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            CONTENT_TYPE_JSON,
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn finalize_openai_response(
    response: reqwest::Response,
    client_wants_stream: bool,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
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
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ));
    }

    Ok(RouterResponse::buffered(
        status,
        &content_type,
        resp_body.into_bytes(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::anthropic_route_pipeline::cache_control_breakpoint_count;
    use http::Response as HttpResponse;
    use serde_json::json;

    fn mock_response(
        status: u16,
        content_type: &str,
        body: impl Into<String>,
    ) -> reqwest::Response {
        HttpResponse::builder()
            .status(status)
            .header("content-type", content_type)
            .body(body.into())
            .unwrap()
            .into()
    }

    #[test]
    fn include_usage_injected_only_for_accounted_streaming() {
        let mut body = json!({"model": "m", "stream": true});
        inject_include_usage_for_accounting(&mut body, true);
        assert_eq!(body["stream_options"], json!({"include_usage": true}));

        // Client-provided stream_options win.
        let mut body =
            json!({"model": "m", "stream": true, "stream_options": {"include_usage": false}});
        inject_include_usage_for_accounting(&mut body, true);
        assert_eq!(body["stream_options"], json!({"include_usage": false}));

        // No accounting or no stream → untouched.
        let mut body = json!({"model": "m", "stream": true});
        inject_include_usage_for_accounting(&mut body, false);
        assert!(body.get("stream_options").is_none());
        let mut body = json!({"model": "m", "stream": false});
        inject_include_usage_for_accounting(&mut body, true);
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn copilot_requires_responses_api_detects_gpt5_and_codex_redirects() {
        // Exact gpt-5.4 tools + reasoning_effort rejection from Copilot.
        let gpt5 = br#"{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.4 in /v1/chat/completions. Please use /v1/responses instead.","code":"invalid_request_body"}}"#;
        assert!(copilot_requires_responses_api(gpt5));
        // Codex-family "not accessible" / unsupported_api_for_model.
        let codex = br#"{"error":{"message":"model 'gpt-5.3-codex' is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}"#;
        assert!(copilot_requires_responses_api(codex));
        // Unrelated 400s must not trigger the fallback.
        assert!(!copilot_requires_responses_api(
            br#"{"error":{"message":"invalid request: missing model"}}"#
        ));
        assert!(!copilot_requires_responses_api(b"model not found"));
    }

    fn sample_openai_chat_response() -> String {
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from upstream"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        })
        .to_string()
    }

    fn sample_anthropic_response() -> String {
        json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello from anthropic"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        })
        .to_string()
    }

    fn sample_gemini_response() -> String {
        json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello from gemini"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}
        })
        .to_string()
    }

    #[test]
    fn normalize_openai_request_model_rewrites_openrouter_model_names() {
        let mut body = json!({"model": "claude-sonnet-4-6"});
        normalize_openai_request_model(&mut body, true, false);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.6");
    }

    #[test]
    fn normalize_openai_request_model_rewrites_copilot_model_names() {
        let mut body = json!({"model": "claude-sonnet-4-6-20250603"});
        normalize_openai_request_model(&mut body, false, true);
        assert_eq!(body["model"], "claude-sonnet-4.6");
    }

    #[test]
    fn ensure_oauth_beta_inserts_when_absent() {
        let mut headers = reqwest::header::HeaderMap::new();
        ensure_oauth_beta(&mut headers);
        assert_eq!(headers.get("anthropic-beta").unwrap(), "oauth-2025-04-20");
    }

    #[test]
    fn ensure_oauth_beta_merges_with_existing() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "claude-code-20250219,extended-cache-ttl-2025-04-11"
                .parse()
                .unwrap(),
        );
        ensure_oauth_beta(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").unwrap(),
            "oauth-2025-04-20,claude-code-20250219,extended-cache-ttl-2025-04-11"
        );
    }

    #[test]
    fn ensure_oauth_beta_noop_when_already_present() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "oauth-2025-04-20,claude-code-20250219".parse().unwrap(),
        );
        ensure_oauth_beta(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").unwrap(),
            "oauth-2025-04-20,claude-code-20250219"
        );
    }

    #[test]
    fn normalize_codex_model_passes_codex_slugs_defaults_foreign() {
        let mut body = json!({"model": "gpt-5.5"});
        normalize_codex_request_model(&mut body);
        assert_eq!(body["model"], "gpt-5.5");

        let mut body = json!({"model": "gpt-5.4-mini"});
        normalize_codex_request_model(&mut body);
        assert_eq!(body["model"], "gpt-5.4-mini");

        // Bare gpt-5 and foreign slots map to the default.
        let mut body = json!({"model": "gpt-5"});
        normalize_codex_request_model(&mut body);
        assert_eq!(
            body["model"],
            crate::services::codex_oauth::DEFAULT_CODEX_MODEL
        );

        let mut body = json!({"model": "claude-sonnet-4-6"});
        normalize_codex_request_model(&mut body);
        assert_eq!(
            body["model"],
            crate::services::codex_oauth::DEFAULT_CODEX_MODEL
        );
    }

    #[test]
    fn merge_reasoning_include_is_idempotent() {
        let mut body = json!({"model": "gpt-5.5"});
        merge_reasoning_include(&mut body);
        merge_reasoning_include(&mut body);
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));

        let mut body = json!({"include": ["file_search_call.results"]});
        merge_reasoning_include(&mut body);
        assert_eq!(
            body["include"],
            json!(["file_search_call.results", "reasoning.encrypted_content"])
        );
    }

    #[test]
    fn codex_cache_key_prefers_client_key_else_session_id() {
        assert_eq!(
            codex_cache_key(&json!({"prompt_cache_key": "conv-1"}), "sid"),
            "conv-1"
        );
        assert_eq!(
            codex_cache_key(&json!({"prompt_cache_key": "  "}), "sid"),
            "sid"
        );
        assert_eq!(codex_cache_key(&json!({}), "sid"), "sid");
    }

    #[test]
    fn responses_object_from_body_rebuilds_output_from_item_done_events() {
        // The ChatGPT backend leaves `response.completed.output` empty under
        // `store:false`; the object must be rebuilt from `output_item.done`.
        let sse = "event: response.output_item.done\n\
             data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"pong\"}]}}\n\n\
             event: response.completed\n\
             data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.5\",\"output\":[]}}\n\n";
        let obj = responses_object_from_body(sse).unwrap();
        assert_eq!(obj["id"], "resp_1");
        assert_eq!(obj["output"][0]["type"], "message");
        assert_eq!(obj["output"][0]["content"][0]["text"], "pong");

        // A non-empty completed `output`, and a direct JSON object, pass through.
        let sse2 = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r2\",\"output\":[{\"type\":\"message\"}]}}\n\n";
        assert_eq!(
            responses_object_from_body(sse2).unwrap()["output"][0]["type"],
            "message"
        );

        let direct = r#"{"id":"resp_2","output":[]}"#;
        assert_eq!(responses_object_from_body(direct).unwrap()["id"], "resp_2");

        assert!(responses_object_from_body("data: {}\n").is_err());
    }

    #[test]
    fn build_anthropic_request_sets_stream_flag_and_model() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"}
            ]
        });

        let (fallback_model, request) = build_anthropic_request(&body, true);

        assert_eq!(fallback_model, "claude-sonnet-4-5");
        assert_eq!(request["model"], "claude-sonnet-4-5");
        assert_eq!(request["stream"], true);
        assert_eq!(request["system"][0]["text"], "Be precise.");
        assert_eq!(request["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(request["messages"][0]["content"][0]["text"], "Hello");
        assert_eq!(
            request["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    /// Not on the turn's prompt, or the tool results behind it never cache.
    #[test]
    fn build_anthropic_request_marks_the_tail_not_the_user_prompt() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "read the file"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "c1", "content": "file body"}
            ]
        });

        let (_, request) = build_anthropic_request(&body, false);

        let messages = request["messages"].as_array().unwrap();
        let tail = messages.last().unwrap();
        let blocks = tail["content"].as_array().unwrap();
        assert_eq!(
            blocks.last().unwrap()["cache_control"]["type"],
            "ephemeral",
            "the tail carries the breakpoint: {request:#}"
        );
        assert_eq!(request["system"][0]["cache_control"]["type"], "ephemeral");
        // System + tail only — Anthropic allows at most four.
        assert_eq!(cache_control_breakpoint_count(&request), 2);
    }

    /// Caching behind the engine's reminder turn would store a prefix the next
    /// request replaces.
    #[test]
    fn build_anthropic_request_keeps_the_breakpoint_off_an_ephemeral_tail() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "user", "content": "read the file"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "c1", "content": "file body"},
                {"role": "user", "content": "<system-reminder>Plan mode is still active</system-reminder>"}
            ]
        });

        let (_, request) = build_anthropic_request(&body, false);

        let blocks = request["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_array()
            .unwrap()
            .clone();
        let marked = blocks
            .iter()
            .position(|b| b.get("cache_control").is_some())
            .expect("a breakpoint was placed");
        assert_eq!(
            blocks[marked]["type"], "tool_result",
            "the breakpoint must sit on durable content: {blocks:#?}"
        );
        assert!(
            blocks[marked + 1..]
                .iter()
                .any(|b| b["text"].as_str().is_some_and(|t| t.contains("reminder"))),
            "the ephemeral block stays past the breakpoint: {blocks:#?}"
        );
        assert_eq!(cache_control_breakpoint_count(&request), 1);
    }

    #[test]
    fn build_anthropic_request_skips_cache_control_for_non_claude_models() {
        let body = json!({
            "model": "MiniMax-M1",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"}
            ]
        });

        let (_, request) = build_anthropic_request(&body, false);

        assert_eq!(
            request["system"],
            json!([{"type": "text", "text": "Be precise."}])
        );
        assert_eq!(
            request["messages"][0]["content"],
            json!([{"type": "text", "text": "Hello"}])
        );
        assert!(
            request["system"][0].get("cache_control").is_none()
                && request["messages"][0]["content"][0]
                    .get("cache_control")
                    .is_none(),
            "non-claude models must not get cache_control"
        );
    }

    #[tokio::test]
    async fn finalize_openai_response_converts_json_to_sse_when_streaming_requested() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_openai_chat_response());

        let result = finalize_openai_response(response, true).await.unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("\"content\":\"Hello from upstream\""));
                assert!(sse.contains("data: [DONE]"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[tokio::test]
    async fn finalize_openai_response_buffers_errors() {
        let response = mock_response(404, CONTENT_TYPE_JSON, r#"{"error":"missing"}"#);

        let result = finalize_openai_response(response, false).await.unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                assert_eq!(status, 404);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(String::from_utf8(body).unwrap(), r#"{"error":"missing"}"#);
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered error"),
        }
    }

    #[tokio::test]
    async fn finalize_anthropic_response_converts_json_to_openai_chat() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_anthropic_response());

        let result = finalize_anthropic_response(response, false, "claude-sonnet-4-5")
            .await
            .unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let json: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(
                    json["choices"][0]["message"]["content"],
                    "Hello from anthropic"
                );
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered response"),
        }
    }

    #[tokio::test]
    async fn finalize_gemini_response_converts_json_to_sse_when_streaming_requested() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_gemini_response());

        let result = finalize_gemini_response(response, true, "gemini-2.5-pro")
            .await
            .unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("\"content\":\"Hello from gemini\""));
                assert!(sse.contains("data: [DONE]"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[tokio::test]
    async fn finalize_openai_response_preserves_upstream_event_streams() {
        let response = mock_response(200, "text/event-stream", "data: [DONE]\n\n");

        let result = finalize_openai_response(response, true).await.unwrap();

        match result {
            RouterResponse::Streaming {
                status,
                content_type,
                ..
            } => {
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
            }
            RouterResponse::Buffered { .. } => panic!("expected streaming response"),
        }
    }

    #[test]
    fn build_anthropic_request_missing_model_uses_default() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let (fallback_model, request) = build_anthropic_request(&body, true);

        assert_eq!(fallback_model, "claude-sonnet-4-5");
        assert_eq!(request["model"], "claude-sonnet-4-5");
    }

    #[test]
    fn build_anthropic_request_non_stream() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let (_fallback_model, request) = build_anthropic_request(&body, false);

        assert_eq!(request["stream"], false);
    }

    #[test]
    fn normalize_openai_request_model_no_op_when_neither_flag() {
        let mut body = json!({"model": "claude-sonnet-4-6"});
        normalize_openai_request_model(&mut body, false, false);
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn normalize_openai_request_model_no_op_missing_model_field() {
        let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        normalize_openai_request_model(&mut body, true, false);
        // No crash, and no model field is inserted
        assert!(body.get("model").is_none());
    }

    #[test]
    fn migrate_max_tokens_renames_for_reasoning_model() {
        let mut body = json!({"model": "gpt-5", "max_tokens": 4096});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 4096);
    }

    #[test]
    fn migrate_max_tokens_renames_for_o_series() {
        let mut body = json!({"model": "o3-mini", "max_tokens": 2048});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 2048);
    }

    #[test]
    fn migrate_max_tokens_preserves_non_reasoning_field() {
        let mut body = json!({"model": "gpt-4o", "max_tokens": 4096});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert_eq!(body["max_tokens"], 4096);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn migrate_max_tokens_does_not_overwrite_existing_new_field() {
        let mut body = json!({
            "model": "gpt-5",
            "max_tokens": 4096,
            "max_completion_tokens": 8192,
        });
        migrate_max_tokens_for_reasoning_models(&mut body);
        // Drop the legacy field, keep the explicit new field intact.
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 8192);
    }

    #[test]
    fn migrate_max_tokens_handles_prefixed_reasoning_model_name() {
        let mut body = json!({"model": "openai/gpt-5-codex", "max_tokens": 1024});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert_eq!(body["max_completion_tokens"], 1024);
    }

    #[test]
    fn migrate_max_tokens_no_op_when_field_absent() {
        let mut body = json!({"model": "gpt-5"});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn strip_non_function_tools_drops_server_tools() {
        // `{type:"web_search"}` alongside function tools 400s an OpenAI Chat
        // endpoint ("expected function"); it must be dropped, function tools kept.
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "web_search"}
            ]
        });
        strip_non_function_tools(&mut body);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "read_file");
    }

    #[test]
    fn strip_non_function_tools_removes_empty_tools_key() {
        let mut body = json!({"model": "x", "tools": [{"type": "web_search"}]});
        strip_non_function_tools(&mut body);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn strip_non_function_tools_drops_orphaned_tool_choice() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required"
        });
        strip_non_function_tools(&mut body);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn strip_non_function_tools_keeps_tool_choice_when_function_tools_survive() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "web_search"}
            ],
            "tool_choice": "auto"
        });
        strip_non_function_tools(&mut body);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn disable_stream_for_inception_flips_stream_when_tools_present() {
        let mut body = json!({
            "model": "mercury-2",
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], false);
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn disable_stream_for_inception_no_op_for_other_providers() {
        let mut body = json!({
            "model": "gpt-4o",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://api.openai.com/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_no_op_when_no_tools_field() {
        let mut body = json!({"model": "mercury-2", "stream": true});
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_no_op_when_tools_empty() {
        let mut body = json!({"model": "mercury-2", "stream": true, "tools": []});
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_matches_by_model_name() {
        let mut body = json!({
            "model": "inception/mercury",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://openrouter.ai/api/v1/");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn disable_stream_for_inception_matches_mercury_edit() {
        let mut body = json!({
            "model": "Mercury-Coder-Small",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://example.com/v1/");
        assert_eq!(body["stream"], false);
    }
}
