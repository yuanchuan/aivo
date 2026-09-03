//! Built-in Anthropic-compatible Router service
//!
//! Acts as an HTTP proxy that intercepts Claude requests and routes them
//! to OpenRouter, handling all necessary API transformations.
use anyhow::Result;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_chat_request::hoist_anthropic_system_messages;
use crate::services::anthropic_route_pipeline::{RequestContext, RouterPipeline};
use crate::services::device_fingerprint;
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils::{self, router_http_model_client};
use crate::services::opencode_session;
use crate::services::token_usage::{
    StreamUsageSniffer, TokenUsage, UsageAccounting, parse_token_usage,
};

#[derive(Clone)]
pub struct AnthropicRouterConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub is_starter: bool,
}

pub struct AnthropicRouter {
    config: AnthropicRouterConfig,
    /// Per-launch loopback token; `Some` rejects requests without it so other
    /// local processes can't spend the key through this router. On the router
    /// (not config) so existing config literals stay untouched.
    expected_token: Option<String>,
    /// When set, 2xx token usage is recorded against the launching key.
    usage: Option<UsageAccounting>,
}

struct AnthropicRouterState {
    config: Arc<AnthropicRouterConfig>,
    expected_token: Option<String>,
    usage: Option<UsageAccounting>,
    client: reqwest::Client,
    /// Set to true when the provider rejects `anthropic-beta` headers.
    /// Once learned, the header is stripped from all future requests.
    beta_header_rejected: Arc<AtomicBool>,
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
        upstream: reqwest::Response,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnthropicRoute {
    Messages,
    CountTokens,
    ChatCompletions,
}

impl AnthropicRouter {
    pub fn new(config: AnthropicRouterConfig) -> Self {
        Self {
            config,
            expected_token: None,
            usage: None,
        }
    }

    /// Requires loopback clients to present this token (Bearer/x-api-key).
    pub fn with_auth_token(mut self, token: String) -> Self {
        self.expected_token = Some(token);
        self
    }

    /// Record 2xx token usage against `key_id` in stats, attributed to `tool`.
    pub fn with_usage_accounting(
        mut self,
        store: crate::services::session_store::SessionStore,
        key_id: String,
        tool: String,
    ) -> Self {
        self.usage = Some(UsageAccounting {
            store,
            key_id,
            tool,
            run_tally: None,
        });
        self
    }

    /// Fold accounted usage into a per-run tally; no-op without accounting.
    pub fn with_run_tally(
        mut self,
        tally: std::sync::Arc<crate::services::usage_stats_store::RunTokenTally>,
    ) -> Self {
        if let Some(usage) = self.usage.as_mut() {
            usage.run_tally = Some(tally);
        }
        self
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let state = AnthropicRouterState {
            config: Arc::new(self.config.clone()),
            expected_token: self.expected_token.clone(),
            usage: self.usage.clone(),
            client: router_http_model_client(),
            beta_header_rejected: Arc::new(AtomicBool::new(false)),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_streaming_router(listener, Arc::new(state), handle_router_request).await
        });
        Ok((port, handle))
    }
}

async fn handle_router_request(
    request: String,
    state: Arc<AnthropicRouterState>,
    mut socket: tokio::net::TcpStream,
) {
    use tokio::io::AsyncWriteExt;

    if let Some(expected) = state.expected_token.as_deref()
        && !http_utils::request_loopback_authorized(&request, expected)
    {
        let response = http_utils::http_error_response(
            401,
            "Invalid or missing auth token (expected Authorization: Bearer or x-api-key)",
        );
        let _ = socket.write_all(response.as_bytes()).await;
        return;
    }

    let path = http_utils::extract_request_path(&request);
    let path = path.split('?').next().unwrap_or("");

    let route = if request.starts_with("POST ") {
        AnthropicRoute::from_request_path(path)
    } else {
        None
    };
    let Some(route) = route else {
        let not_found =
            http_utils::http_response(404, CONTENT_TYPE_JSON, "{\"error\":\"Not found\"}");
        let _ = socket.write_all(not_found.as_bytes()).await;
        return;
    };

    match forward_request(
        &request,
        &state.config,
        &state.client,
        route,
        &state.beta_header_rejected,
    )
    .await
    {
        Ok(resp) => {
            let usage = write_router_response(&mut socket, resp, state.usage.is_some())
                .await
                .unwrap_or(None);
            if let (Some(acct), Some(u)) = (&state.usage, usage) {
                let model = http_utils::extract_request_body(&request)
                    .ok()
                    .and_then(|b| serde_json::from_str::<Value>(b).ok())
                    .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(str::to_string));
                acct.record(model.as_deref(), &u).await;
            }
        }
        Err(e) => {
            let err =
                http_utils::http_error_response(500, &format!("Internal Server Error: {e:#}"));
            let _ = socket.write_all(err.as_bytes()).await;
        }
    }
}

impl AnthropicRoute {
    fn from_request_path(path: &str) -> Option<Self> {
        match path {
            "/v1/messages" | "/messages" => Some(Self::Messages),
            "/v1/messages/count_tokens" | "/messages/count_tokens" => Some(Self::CountTokens),
            "/v1/chat/completions" | "/chat/completions" => Some(Self::ChatCompletions),
            _ => None,
        }
    }

    fn endpoint(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::CountTokens => "messages/count_tokens",
            Self::ChatCompletions => "chat/completions",
        }
    }

    fn patch_route(self) -> &'static str {
        self.endpoint()
    }
}

fn build_endpoint_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/{}", base, endpoint.trim_start_matches('/'))
    } else {
        format!("{}/v1/{}", base, endpoint.trim_start_matches('/'))
    }
}

async fn classify_upstream_response(response: reqwest::Response) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if content_type.contains("text/event-stream") {
        Ok(RouterResponse::Streaming {
            status,
            content_type,
            upstream: response,
        })
    } else {
        let body = response.bytes().await?.to_vec();
        Ok(RouterResponse::Buffered {
            status,
            content_type,
            body,
        })
    }
}

/// Writes the response to the socket; with `sniff_usage`, returns 2xx token
/// usage parsed from a buffered body or sniffed off the forwarded stream.
async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
    sniff_usage: bool,
) -> Result<Option<TokenUsage>> {
    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let usage = (sniff_usage && (200..300).contains(&status))
                .then(|| parse_token_usage(&body))
                .flatten();
            http_utils::write_buffered_response(socket, status, &content_type, &body).await?;
            Ok(usage)
        }
        RouterResponse::Streaming {
            status,
            content_type,
            upstream,
        } => {
            let mut sniffer = StreamUsageSniffer::new(sniff_usage && (200..300).contains(&status));
            http_utils::write_streaming_response_with_prefix(
                socket,
                status,
                &content_type,
                &[],
                upstream,
                |chunk| sniffer.observe(chunk),
            )
            .await?;
            Ok(sniffer.finish())
        }
    }
}

async fn forward_request(
    request: &str,
    config: &Arc<AnthropicRouterConfig>,
    client: &reqwest::Client,
    route: AnthropicRoute,
    beta_header_rejected: &AtomicBool,
) -> Result<RouterResponse> {
    let mut passthrough_headers = http_utils::extract_passthrough_headers(request)?;
    if beta_header_rejected.load(Ordering::Relaxed) {
        http_utils::strip_beta_headers(&mut passthrough_headers);
    }
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;
    // Strict Anthropic upstreams 400 on a `role:"system"` entry inside
    // `messages` (only user/assistant are valid there). Hoist any such message
    // into the top-level `system` field before patching/forwarding so the
    // request validates regardless of which client produced it.
    hoist_anthropic_system_messages(&mut body);
    // Catalog (when cached) lets ModelNamePatch snap to the exact advertised id.
    let catalog = crate::services::models_cache::ModelsCache::shared()
        .model_ids(&config.upstream_base_url)
        .await;
    let ctx = RequestContext::new(&config.upstream_base_url).with_catalog(catalog.as_deref());
    let pipeline = RouterPipeline::for_openrouter();
    pipeline.patch_json(route.patch_route(), &mut body, &ctx)?;

    let url = build_endpoint_url(&config.upstream_base_url, route.endpoint());
    let mut headers = passthrough_headers;
    let auth_value = format!("Bearer {}", config.upstream_api_key);
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_JSON));
    pipeline.patch_headers(route.patch_route(), &mut headers, &ctx)?;
    opencode_session::insert_session_header(&mut headers, &url, Some(&body));

    let is_starter = config.is_starter;
    let response = device_fingerprint::maybe_with_starter_headers(
        client.post(&url).headers(headers).json(&body),
        is_starter,
    )
    .send_logged()
    .await?;

    // Detect beta header rejection: on 400, check if the provider rejected
    // anthropic-beta headers, learn to strip them, and retry immediately.
    if response.status() == 400 && !beta_header_rejected.load(Ordering::Relaxed) {
        let content_type = http_utils::response_content_type(&response);
        let response_body = response.bytes().await?.to_vec();

        if http_utils::is_beta_header_rejection(&String::from_utf8_lossy(&response_body)) {
            beta_header_rejected.store(true, Ordering::Relaxed);
            eprintln!("  • Provider rejected anthropic-beta header — retrying without it");

            let mut retry_headers = http_utils::extract_passthrough_headers(request)?;
            http_utils::strip_beta_headers(&mut retry_headers);
            let auth_value = format!("Bearer {}", config.upstream_api_key);
            retry_headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
            retry_headers.insert(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_JSON));
            pipeline.patch_headers(route.patch_route(), &mut retry_headers, &ctx)?;
            opencode_session::insert_session_header(&mut retry_headers, &url, Some(&body));

            let retry_response = device_fingerprint::maybe_with_starter_headers(
                client.post(&url).headers(retry_headers).json(&body),
                is_starter,
            )
            .send_logged()
            .await?;

            return classify_upstream_response(retry_response).await;
        }

        // Not a beta rejection — return the original 400 as-is
        return Ok(RouterResponse::Buffered {
            status: 400,
            content_type,
            body: response_body,
        });
    }

    classify_upstream_response(response).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_names::{
        normalize_claude_version, transform_model_for_openrouter, transform_model_for_provider,
    };

    #[test]
    fn test_transform_openrouter_adds_prefix_and_normalizes() {
        let url = "https://openrouter.ai/api/v1";
        assert_eq!(
            transform_model_for_provider(None, url, "claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_provider(None, url, "claude-opus-4-6"),
            "anthropic/claude-opus-4.6"
        );
        assert_eq!(
            transform_model_for_provider(None, url, "claude-haiku-4-5"),
            "anthropic/claude-haiku-4.5"
        );
    }

    #[test]
    fn test_transform_openrouter_date_suffix_preserved() {
        assert_eq!(
            transform_model_for_provider(
                None,
                "https://openrouter.ai/api/v1",
                "claude-haiku-4-5-20251001"
            ),
            "anthropic/claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_transform_other_provider_passthrough() {
        // Non-OpenRouter providers: model names pass through unchanged
        assert_eq!(
            transform_model_for_provider(
                None,
                "https://ai-gateway.vercel.sh/v1",
                "claude-sonnet-4-6"
            ),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            transform_model_for_provider(None, "https://api.example.com/v1", "claude-opus-4-6"),
            "claude-opus-4-6"
        );
    }

    #[test]
    fn test_transform_already_prefixed() {
        assert_eq!(
            transform_model_for_openrouter("anthropic/claude-sonnet-4.6"),
            "anthropic/claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_transform_non_claude_model() {
        assert_eq!(transform_model_for_openrouter("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_normalize_claude_version() {
        assert_eq!(
            normalize_claude_version("claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_claude_version("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_extract_request_body_normal() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"key\":\"val\"}"
        );
    }

    #[test]
    fn test_extract_request_body_missing_separator_returns_error() {
        let req = "POST /v1/messages HTTP/1.1";
        assert!(http_utils::extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short_request_no_panic() {
        // A request shorter than 4 bytes must not panic
        assert!(http_utils::extract_request_body("AB").is_err());
    }

    #[test]
    fn test_endpoint_path_matching() {
        assert_eq!(
            AnthropicRoute::from_request_path("/v1/messages"),
            Some(AnthropicRoute::Messages)
        );
        assert_eq!(
            AnthropicRoute::from_request_path("/v1/chat/completions"),
            Some(AnthropicRoute::ChatCompletions)
        );
        assert_eq!(
            AnthropicRoute::from_request_path("/v1/messages/count_tokens"),
            Some(AnthropicRoute::CountTokens)
        );
        assert_eq!(AnthropicRoute::from_request_path("/v1/unknown"), None);
    }

    #[test]
    fn test_route_metadata_matches_expected_endpoints() {
        assert_eq!(AnthropicRoute::Messages.endpoint(), "messages");
        assert_eq!(
            AnthropicRoute::CountTokens.endpoint(),
            "messages/count_tokens"
        );
        assert_eq!(
            AnthropicRoute::ChatCompletions.endpoint(),
            "chat/completions"
        );
        assert_eq!(
            AnthropicRoute::CountTokens.patch_route(),
            "messages/count_tokens"
        );
    }

    #[test]
    fn test_build_endpoint_url() {
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api/v1", "messages"),
            "https://openrouter.ai/api/v1/messages"
        );
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api", "chat/completions"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            build_endpoint_url("https://openrouter.ai/api/v1/", "messages/count_tokens"),
            "https://openrouter.ai/api/v1/messages/count_tokens"
        );
    }

    #[test]
    fn route_from_path_returns_none_for_unknown() {
        assert_eq!(AnthropicRoute::from_request_path("/v1/completions"), None);
        assert_eq!(AnthropicRoute::from_request_path("/v1/embeddings"), None);
        assert_eq!(AnthropicRoute::from_request_path("/v1/models"), None);
        assert_eq!(AnthropicRoute::from_request_path("/health"), None);
        assert_eq!(AnthropicRoute::from_request_path(""), None);
    }

    #[test]
    fn route_from_path_bare_chat_completions() {
        // /chat/completions without /v1 prefix should still route correctly
        let route = AnthropicRoute::from_request_path("/chat/completions");
        assert_eq!(route, Some(AnthropicRoute::ChatCompletions));
    }

    #[test]
    fn build_endpoint_url_no_trailing_slash() {
        let url = build_endpoint_url("https://api.example.com/v1", "messages");
        assert_eq!(url, "https://api.example.com/v1/messages");
        // Ensure no double slashes
        assert!(!url.contains("//messages"));
    }

    #[test]
    fn build_endpoint_url_with_trailing_slash() {
        let url = build_endpoint_url("https://api.example.com/v1/", "messages");
        assert_eq!(url, "https://api.example.com/v1/messages");
        // Trailing slash on base URL should not produce double slashes
        assert!(!url.contains("//messages"));
    }

    #[test]
    fn route_endpoint_and_patch_route_consistent() {
        // endpoint() and patch_route() should return the same value for every variant
        let routes = [
            AnthropicRoute::Messages,
            AnthropicRoute::CountTokens,
            AnthropicRoute::ChatCompletions,
        ];
        for route in routes {
            assert_eq!(
                route.endpoint(),
                route.patch_route(),
                "endpoint() and patch_route() must match for {:?}",
                route
            );
        }
    }
}
