//! Anthropic-to-OpenAI router service
//!
//! Acts as an HTTP proxy that accepts Anthropic-format requests and routes them
//! to OpenAI-compatible providers (like Cloudflare Workers AI), handling the
//! required request and response transformations.
//!
//! Flow:
//! Anthropic /v1/messages → Router → OpenAI /v1/chat/completions
use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::device_fingerprint;

use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, hoist_anthropic_system_messages,
};
use crate::services::anthropic_chat_response::{
    OpenAIStreamConverter, convert_openai_sse_to_anthropic, convert_openai_to_anthropic,
};
use crate::services::anthropic_route_pipeline::{
    CacheControlPatch, RequestContext, RequestPatch, ThinkingNormalizationPatch,
    inject_chat_completions_cache_control,
};
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils::{self, router_http_model_client};
use crate::services::model_names::{
    infer_provider_name_from_model, is_gateway_style_endpoint, select_model_for_provider_attempt,
    strip_context_suffix,
};
use crate::services::openai_anthropic_bridge::convert_openai_chat_response_to_sse;
use crate::services::openai_gemini_bridge::{build_google_generate_content_url, openai_chat_model};
use crate::services::openai_models::{
    OpenAIChatRequest, stringify_message_content as stringify_typed_message_content,
};
use crate::services::opencode_session;
use crate::services::protocol_fallback::{
    AttemptOutcome, FirstError, MismatchDirective, QuirkRetryState, classify_attempt,
    commit_protocol_switch, mismatch_directive, protocol_candidates, record_slot_outcome,
};
use crate::services::provider_profile::is_direct_openai_base;
use crate::services::provider_protocol::{
    PathVariant, ProviderProtocol, classify_failed_attempt, decode_route, is_endpoint_missing,
    is_protocol_mismatch, is_terminal_upstream_error,
};
use crate::services::responses_chat_conversion::{
    try_convert_chat_to_responses_request, try_convert_responses_json_to_chat,
};
use crate::services::route_cache::{PersistedRoute, RouteCache};
use crate::services::serve_upstream::disable_stream_for_inception_with_tools;
use crate::services::token_usage::{StreamUsageSniffer, UsageAccounting, parse_token_usage};
use crate::services::wire_format::{
    RequestOptions, ResponseOptions, translate_request, translate_response,
};

#[derive(Clone)]
pub struct AnthropicToOpenAIRouterConfig {
    /// The target OpenAI-compatible provider base URL (e.g., Cloudflare)
    pub target_base_url: String,
    /// API key for the target provider
    pub target_api_key: String,
    /// Per-model routes learned for `claude` (`""` = default). Seeds the
    /// `RouteCache`; absent a route, the tool-native Anthropic prior applies.
    pub seed_routes: BTreeMap<String, PersistedRoute>,
    /// When `true`, strip Anthropic-specific `cache_control` keys from the
    /// request before forwarding (Bedrock-style shims reject them) and skip
    /// the inject step that would otherwise add them.
    pub strip_cache_control: bool,
    /// Optional model prefix to add (e.g., "@cf/" for Cloudflare)
    pub model_prefix: Option<String>,
    /// Whether the provider requires `reasoning_content` on assistant tool-call turns (e.g., Moonshot)
    pub requires_reasoning_content: bool,
    /// Cap applied to `max_tokens` before forwarding to the provider,
    /// for providers that reject values above a fixed output ceiling.
    pub max_tokens_cap: Option<u64>,
    /// Whether this is the aivo starter provider (requires device fingerprint headers).
    pub is_starter: bool,
}

pub struct AnthropicToOpenAIRouter {
    config: AnthropicToOpenAIRouterConfig,
    /// Per-launch loopback token; `Some` rejects requests without it so other
    /// local processes can't spend the key through this router. On the router
    /// (not config) so existing config literals stay untouched.
    expected_token: Option<String>,
    /// When set, 2xx token usage is recorded against the launching key.
    usage: Option<UsageAccounting>,
}

struct AnthropicToOpenAIRouterState {
    config: Arc<AnthropicToOpenAIRouterConfig>,
    expected_token: Option<String>,
    usage: Option<UsageAccounting>,
    client: reqwest::Client,
    /// Per-model learned routes; each request resolves its slot and runs the
    /// cascade against it. A confirmed slot persists on exit via `dirty_routes`.
    route_cache: Arc<RouteCache>,
    probe: ProbeState,
    /// Flipped to `true` when a cascade attempt sees an upstream error envelope
    /// matching the `requires_reasoning_content` quirk. `persist_runtime_discoveries`
    /// reads this and writes `ApiKey::requires_reasoning_content = Some(true)`,
    /// so subsequent launches enable strict mode for this key without growing
    /// the hardcoded substring list in `ProviderQuirks::for_base_url`.
    learned_requires_reasoning: Arc<AtomicBool>,
}

/// Learned state for the native Anthropic probe, cloned into each request
/// handler so it can be mutated across concurrent requests.
/// After this many consecutive non-terminal upstream errors from the native
/// `/v1/messages` probe (e.g. repeated 400/422 shape rejections), give up and
/// mark the probe Failed so subsequent requests skip the probe and go straight
/// to the chat/completions fallback. Terminal errors (5xx/auth/rate-limit) are
/// not counted — those are surfaced directly via `Terminal` and the user is
/// expected to keep retrying the same path.
const PROBE_UPSTREAM_ERROR_LIMIT: u8 = 3;

#[derive(Clone)]
struct ProbeState {
    anthropic_outcome: Arc<AtomicU8>,
    consecutive_upstream_errors: Arc<AtomicU8>,
    /// Set to true when the provider rejects `anthropic-beta` headers (e.g. Bedrock, Vertex AI).
    /// Once learned, the header is stripped from all future requests.
    beta_header_rejected: Arc<AtomicBool>,
}

impl ProbeState {
    fn new() -> Self {
        Self {
            anthropic_outcome: Arc::new(AtomicU8::new(ProbeOutcome::Unlearned as u8)),
            consecutive_upstream_errors: Arc::new(AtomicU8::new(0)),
            beta_header_rejected: Arc::new(AtomicBool::new(false)),
        }
    }

    fn outcome(&self) -> ProbeOutcome {
        ProbeOutcome::from_u8(self.anthropic_outcome.load(Ordering::Relaxed))
    }

    fn set_outcome(&self, outcome: ProbeOutcome) {
        self.anthropic_outcome
            .store(outcome as u8, Ordering::Relaxed);
    }

    fn reset_upstream_error_streak(&self) {
        self.consecutive_upstream_errors.store(0, Ordering::Relaxed);
    }

    /// Increment the streak and return true once it reaches the limit.
    fn record_upstream_error(&self) -> bool {
        let prev = self
            .consecutive_upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        prev + 1 >= PROBE_UPSTREAM_ERROR_LIMIT
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeOutcome {
    Unlearned = 0,
    UseRoot = 1,
    UsePrefixed = 2,
    Failed = 3,
}

impl ProbeOutcome {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::UseRoot,
            2 => Self::UsePrefixed,
            3 => Self::Failed,
            _ => Self::Unlearned,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AnthropicProbePath {
    Root,
    Prefixed,
}

impl AnthropicProbePath {
    fn to_outcome(self) -> ProbeOutcome {
        match self {
            Self::Root => ProbeOutcome::UseRoot,
            Self::Prefixed => ProbeOutcome::UsePrefixed,
        }
    }
}

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    /// Already streamed to socket — nothing to write.
    AlreadyStreamed,
}

/// Outcome of a single native `/v1/messages` send attempt. Distinguishes
/// "endpoint truly missing" (safe to flip the protocol pin to Openai) from
/// "endpoint exists but errored" — and within errored, separates
/// terminal/authoritative responses (5xx/401/403/429) from transient ones.
enum SendNativeOutcome {
    Success(RouterResponse),
    EndpointMissing,
    /// Path answered with an authoritative auth error (401/403). Ambiguous:
    /// could be a real auth failure or a host that doesn't serve the
    /// Anthropic shape — the caller flips the pin and lets chat decide.
    Terminal(RouterResponse),
    /// Path answered with a transient error (429/5xx): the endpoint exists
    /// and works, so the protocol pin must NOT flip.
    Transient(RouterResponse),
    UpstreamError,
}

/// Aggregated outcome across all candidate paths in `try_native_anthropic`.
/// `EndpointMissing` is reported only when *every* candidate produced an
/// endpoint-missing status; a single `UpstreamError` upgrades the aggregate
/// to `UpstreamError` because we can't conclude the endpoint is absent.
enum NativeAnthropicResult {
    Success(RouterResponse),
    EndpointMissing,
    Terminal(RouterResponse),
    Transient(RouterResponse),
    UpstreamError,
}

impl AnthropicToOpenAIRouter {
    pub fn new(config: AnthropicToOpenAIRouterConfig) -> Self {
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
        tally: Arc<crate::services::usage_stats_store::RunTokenTally>,
    ) -> Self {
        if let Some(usage) = self.usage.as_mut() {
            usage.run_tally = Some(tally);
        }
        self
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(
        &self,
    ) -> Result<(
        u16,
        Arc<RouteCache>,
        Arc<AtomicBool>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        // Tool-native protocol for `aivo claude`: try Anthropic `/v1/messages`
        // first for any model without a learned route.
        let route_cache = Arc::new(RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            self.config.seed_routes.clone(),
        ));
        let learned_requires_reasoning = Arc::new(AtomicBool::new(false));
        let state = AnthropicToOpenAIRouterState {
            config: Arc::new(self.config.clone()),
            expected_token: self.expected_token.clone(),
            usage: self.usage.clone(),
            client: router_http_model_client(),
            route_cache: route_cache.clone(),
            probe: ProbeState::new(),
            learned_requires_reasoning: learned_requires_reasoning.clone(),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_streaming_router(listener, Arc::new(state), handle_router_request).await
        });
        Ok((port, route_cache, learned_requires_reasoning, handle))
    }
}

async fn handle_router_request(
    request: String,
    state: Arc<AnthropicToOpenAIRouterState>,
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

    if !http_utils::is_post_path(&request, &["/v1/messages", "/messages"]) {
        let not_found =
            http_utils::http_response(404, CONTENT_TYPE_JSON, "{\"error\":\"Not found\"}");
        let _ = socket.write_all(not_found.as_bytes()).await;
        return;
    }

    let mut sniffer = StreamUsageSniffer::new(state.usage.is_some());
    let response = match handle_anthropic_to_upstream(
        &request,
        &state.config,
        &state.client,
        &state.route_cache,
        &state.probe,
        &state.learned_requires_reasoning,
        &mut socket,
        &mut sniffer,
    )
    .await
    {
        Ok(response) => response,
        Err(e) => {
            let error = http_utils::http_error_response(500, &format!("{e:#}"));
            let _ = socket.write_all(error.as_bytes()).await;
            return;
        }
    };

    // Peek usage before the body moves; record after the write so the
    // stats-file lock never delays the response.
    let usage = if state.usage.is_some() {
        match &response {
            RouterResponse::Buffered { status, body, .. } if (200..300).contains(status) => {
                parse_token_usage(body)
            }
            RouterResponse::AlreadyStreamed => sniffer.finish(),
            _ => None,
        }
    } else {
        None
    };

    let _ = write_router_response(&mut socket, response).await;

    if let (Some(acct), Some(u)) = (&state.usage, usage) {
        let model = http_utils::extract_request_body(&request)
            .ok()
            .and_then(|b| serde_json::from_str::<Value>(b).ok())
            .and_then(|v| {
                v.get("model")
                    .and_then(|m| m.as_str())
                    .map(|s| strip_context_suffix(s).to_string())
            });
        acct.record(model.as_deref(), &u).await;
    }
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
) -> Result<()> {
    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            http_utils::write_buffered_response(socket, status, &content_type, &body).await?;
        }
        RouterResponse::AlreadyStreamed => {}
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

/// Build the standard headers for native Anthropic requests.
fn build_native_anthropic_headers(
    passthrough_headers: &HeaderMap,
    api_key: &str,
) -> Result<HeaderMap> {
    let mut headers = passthrough_headers.clone();
    headers.insert("x-api-key", HeaderValue::from_str(api_key)?);
    headers.insert("Content-Type", HeaderValue::from_static(CONTENT_TYPE_JSON));
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
    Ok(headers)
}

/// Build the native `/v1/messages` URL, optionally under a sub-path prefix
/// (e.g. `/anthropic` for DeepSeek). With a prefix, strips a trailing `/v1`
/// from the base so the prefix can slot in before the version segment,
/// producing `{base}{prefix}/v1/messages`.
fn build_anthropic_messages_url(base_url: &str, prefix: Option<&str>) -> String {
    let prefix = prefix
        .map(|p| p.trim_end_matches('/').trim_start_matches('/'))
        .filter(|p| !p.is_empty());
    match prefix {
        Some(p) => {
            let base = base_url.trim_end_matches('/');
            let base = base.strip_suffix("/v1").unwrap_or(base);
            format!("{}/{}/v1/messages", base, p)
        }
        None => http_utils::build_target_url(base_url, "/v1/messages"),
    }
}

/// Generic fallback sub-path used by the `Prefixed` probe when the provider's
/// profile didn't configure one — common convention for OpenAI-first hosts
/// that also expose an Anthropic-compatible endpoint.
const DEFAULT_ANTHROPIC_SUB_PATH: &str = "/anthropic";

/// Candidate paths to try, ordered, for a given learned state.
///
/// A `None` return means the probe has given up for this router lifetime —
/// the caller should bail out without making a request.
///
/// Unlearned hosts always try `Root` then `Prefixed`. Hosts that mount
/// Anthropic at `/anthropic/v1/messages` (e.g. DeepSeek, MiniMax) self-discover
/// via the trailing fallback on first launch and pin the result to the key's
/// `learned.*_path_variant`, so subsequent launches go straight to the winning
/// path.
fn probe_paths(outcome: ProbeOutcome) -> Option<&'static [AnthropicProbePath]> {
    match outcome {
        ProbeOutcome::UseRoot => Some(&[AnthropicProbePath::Root]),
        ProbeOutcome::UsePrefixed => Some(&[AnthropicProbePath::Prefixed]),
        ProbeOutcome::Failed => None,
        ProbeOutcome::Unlearned => Some(&[AnthropicProbePath::Root, AnthropicProbePath::Prefixed]),
    }
}

/// Send a single native `/v1/messages` attempt. Distinguishes
/// `EndpointMissing` (404/405/415/501 — safe to conclude the path doesn't
/// exist) from `UpstreamError` (any other 4xx/5xx — endpoint may exist, error
/// is auth/rate/transient).
async fn send_native_anthropic(
    url: &str,
    native_body: &Value,
    config: &AnthropicToOpenAIRouterConfig,
    client: &reqwest::Client,
    passthrough_headers: &HeaderMap,
    beta_header_rejected: &AtomicBool,
) -> Result<SendNativeOutcome> {
    let mut headers = build_native_anthropic_headers(passthrough_headers, &config.target_api_key)?;
    opencode_session::insert_session_header(&mut headers, url, Some(native_body));
    let is_starter = config.is_starter;
    let response = device_fingerprint::maybe_with_starter_headers(
        client.post(url).headers(headers).json(native_body),
        is_starter,
    )
    .send_logged()
    .await?;

    let status_code = response.status().as_u16();

    // Beta-header learning runs before the generic mismatch check so a
    // correctable native response isn't discarded in favor of OpenAI.
    if status_code == 400 && !beta_header_rejected.load(Ordering::Relaxed) {
        let response_body = response.bytes().await?;
        let body_str = String::from_utf8_lossy(&response_body);

        if http_utils::is_beta_header_rejection(&body_str) {
            beta_header_rejected.store(true, Ordering::Relaxed);
            eprintln!("  • Provider rejected anthropic-beta header — retrying without it");

            let mut retry_headers =
                build_native_anthropic_headers(passthrough_headers, &config.target_api_key)?;
            http_utils::strip_beta_headers(&mut retry_headers);
            opencode_session::insert_session_header(&mut retry_headers, url, Some(native_body));

            let retry_response = device_fingerprint::maybe_with_starter_headers(
                client.post(url).headers(retry_headers).json(native_body),
                is_starter,
            )
            .send_logged()
            .await?;

            let retry_status = retry_response.status().as_u16();
            if is_protocol_mismatch(retry_status) {
                return classify_native_failure(retry_status, retry_response).await;
            }

            let retry_ct = http_utils::response_content_type(&retry_response);
            let retry_body = retry_response.bytes().await?;
            return Ok(SendNativeOutcome::Success(RouterResponse::Buffered {
                status: retry_status,
                content_type: retry_ct,
                body: retry_body.to_vec(),
            }));
        }

        // 400 with a non-beta-rejection body is a real validation failure.
        return Ok(SendNativeOutcome::UpstreamError);
    }

    if is_protocol_mismatch(status_code) {
        return classify_native_failure(status_code, response).await;
    }

    let content_type = http_utils::response_content_type(&response);
    let response_body = response.bytes().await?;
    Ok(SendNativeOutcome::Success(RouterResponse::Buffered {
        status: status_code,
        content_type,
        body: response_body.to_vec(),
    }))
}

async fn classify_native_failure(
    status: u16,
    response: reqwest::Response,
) -> Result<SendNativeOutcome> {
    if is_endpoint_missing(status) {
        return Ok(SendNativeOutcome::EndpointMissing);
    }
    if is_terminal_upstream_error(status) {
        let content_type = http_utils::response_content_type(&response);
        let body = response.bytes().await?;
        let buffered = RouterResponse::Buffered {
            status,
            content_type,
            body: body.to_vec(),
        };
        return Ok(if native_terminal_flips_pin(status) {
            SendNativeOutcome::Terminal(buffered)
        } else {
            SendNativeOutcome::Transient(buffered)
        });
    }
    Ok(SendNativeOutcome::UpstreamError)
}

/// Only 401/403 from `/v1/messages` may flip the protocol pin: a host that
/// doesn't serve the Anthropic shape can answer auth-like errors (Cloudflare
/// AI Gateway), so chat fallback must arbitrate. 429/5xx mean the endpoint
/// exists and hit transient trouble — flipping on those would permanently
/// abandon the native path (and prompt caching) after a single blip.
fn native_terminal_flips_pin(status: u16) -> bool {
    matches!(status, 401 | 403)
}

/// Try sending the request in native Anthropic format to the upstream's `/v1/messages`.
/// Iterates candidate paths (`/v1/messages`, `{prefix}/v1/messages`) based on
/// the configured prefix and the path learned on earlier requests. Aggregates
/// per-path outcomes: `EndpointMissing` only when every candidate path
/// returned 404-style; a single `UpstreamError` (auth/rate/transient) prevents
/// the aggregate from being EndpointMissing so the caller doesn't poison the
/// protocol pin.
async fn try_native_anthropic(
    body: &Value,
    config: &AnthropicToOpenAIRouterConfig,
    client: &reqwest::Client,
    passthrough_headers: &HeaderMap,
    probe: &ProbeState,
) -> Result<NativeAnthropicResult> {
    let Some(candidates) = probe_paths(probe.outcome()) else {
        // Probe was previously marked Failed (all paths confirmed missing);
        // surface that as EndpointMissing so the caller flips to Openai.
        return Ok(NativeAnthropicResult::EndpointMissing);
    };

    let mut native_body = body.clone();
    let ctx = RequestContext::new(&config.target_base_url);
    CacheControlPatch.patch_json("messages", &mut native_body, &ctx)?;
    ThinkingNormalizationPatch.patch_json("messages", &mut native_body, &ctx)?;

    let mut saw_upstream_error = false;
    for &path in candidates {
        let sub_path = match path {
            AnthropicProbePath::Root => None,
            AnthropicProbePath::Prefixed => Some(DEFAULT_ANTHROPIC_SUB_PATH),
        };
        let url = build_anthropic_messages_url(&config.target_base_url, sub_path);

        match send_native_anthropic(
            &url,
            &native_body,
            config,
            client,
            passthrough_headers,
            &probe.beta_header_rejected,
        )
        .await?
        {
            SendNativeOutcome::Success(response) => {
                probe.set_outcome(path.to_outcome());
                probe.reset_upstream_error_streak();
                return Ok(NativeAnthropicResult::Success(response));
            }
            SendNativeOutcome::Terminal(response) => {
                // Path is real — keep probing on retries (don't bump the streak).
                probe.reset_upstream_error_streak();
                return Ok(NativeAnthropicResult::Terminal(response));
            }
            SendNativeOutcome::Transient(response) => {
                probe.reset_upstream_error_streak();
                return Ok(NativeAnthropicResult::Transient(response));
            }
            SendNativeOutcome::EndpointMissing => continue,
            SendNativeOutcome::UpstreamError => {
                saw_upstream_error = true;
                continue;
            }
        }
    }

    if saw_upstream_error {
        // Endpoint may exist but produced a non-terminal error (400/422).
        // After enough consecutive failures, mark Failed so the probe stops
        // wasting time on subsequent requests.
        if probe.record_upstream_error() {
            probe.set_outcome(ProbeOutcome::Failed);
        }
        Ok(NativeAnthropicResult::UpstreamError)
    } else {
        // Every candidate confirmed the path doesn't exist.
        probe.set_outcome(ProbeOutcome::Failed);
        Ok(NativeAnthropicResult::EndpointMissing)
    }
}

/// Probe native `/v1/messages` only while this model's route is still Anthropic
/// (claude's tool-native default); once it's moved off, re-probing would
/// short-circuit to `EndpointMissing` and undo the learned pin.
fn should_try_native_anthropic(active_protocol: &AtomicU8) -> bool {
    decode_route(active_protocol.load(Ordering::Relaxed)).0 == ProviderProtocol::Anthropic
}

/// Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions
#[allow(clippy::too_many_arguments)]
async fn handle_anthropic_to_upstream(
    request: &str,
    config: &Arc<AnthropicToOpenAIRouterConfig>,
    client: &reqwest::Client,
    route_cache: &Arc<RouteCache>,
    probe: &ProbeState,
    learned_requires_reasoning: &Arc<AtomicBool>,
    socket: &mut tokio::net::TcpStream,
    sniffer: &mut StreamUsageSniffer,
) -> Result<RouterResponse> {
    let mut passthrough_headers = http_utils::extract_passthrough_headers(request)?;
    if probe.beta_header_rejected.load(Ordering::Relaxed) {
        http_utils::strip_beta_headers(&mut passthrough_headers);
    }
    let body_str = http_utils::extract_request_body(request)?;

    let mut body: Value = serde_json::from_str(body_str)?;
    // Hoist any stray `role:"system"` message into the top-level `system`
    // field before conversion/forwarding. Covers both sub-paths from here: the
    // Anthropic→OpenAI bridge (which would otherwise emit a second,
    // mid-conversation system message) and the native-Anthropic forward (which
    // a strict upstream would 400). See hoist_anthropic_system_messages.
    hoist_anthropic_system_messages(&mut body);
    // Drop the `[1m]`/`[2m]` UI-hint suffix Claude Code carries through from
    // its env vars. The upstream (real Anthropic, OpenAI Chat, Gemini) doesn't
    // know about it; native Anthropic is in Direct mode where the bridge
    // isn't in the path, so seeing the suffix here means we're forwarding to
    // something that won't recognize it.
    if let Some(model) = body.get_mut("model")
        && let Some(s) = model.as_str()
    {
        let stripped = strip_context_suffix(s);
        if stripped.len() != s.len() {
            *model = Value::String(stripped.to_string());
        }
    }

    let model_is_claude = body
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("claude"));

    // Per-model route slot: the cascade reads/writes its atom (not a shared
    // pin) so models on one key don't clobber each other; `confirm()` persists.
    let req_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_string();
    let slot = route_cache.resolve(&req_model);
    let active_protocol: &AtomicU8 = slot.route_atom();

    // Stashed Terminal response from the native-Anthropic preflight. Some
    // hosts (e.g., Cloudflare's AI gateway, OpenAI-only proxies) reject
    // /v1/messages with 401/403 and a host-shaped error envelope rather than
    // 404, so we can't tell "auth failed" from "endpoint missing in disguise"
    // until we see whether the OpenAI Chat fallback succeeds. Forwarded as the
    // surfaced error only if every chat candidate also fails.
    let mut native_anthropic_terminal: Option<RouterResponse> = None;

    if should_try_native_anthropic(active_protocol) {
        match try_native_anthropic(&body, config, client, &passthrough_headers, probe).await? {
            NativeAnthropicResult::Success(response) => {
                slot.confirm();
                record_slot_outcome(&slot, true);
                return Ok(response);
            }
            NativeAnthropicResult::Terminal(response) => {
                // Don't return immediately: a 401 from /v1/messages on a host
                // that doesn't actually serve the Anthropic shape (Cloudflare
                // AI Gateway, etc.) looks identical to a real auth failure.
                // Pre-pin Openai (same as EndpointMissing) and try chat
                // fallback. If chat fails too, the chat candidates carry more
                // diagnostic body shapes, so prefer them over this.
                native_anthropic_terminal = Some(response);
                if decode_route(active_protocol.load(Ordering::Relaxed)).0
                    == ProviderProtocol::Anthropic
                {
                    active_protocol.store(ProviderProtocol::Openai.to_u8(), Ordering::Relaxed);
                }
            }
            NativeAnthropicResult::EndpointMissing => {
                // Pre-pin Openai so a fallback success at attempt==0 (where
                // `commit_protocol_switch` is a no-op) still persists. Guard
                // against clobbering a learned non-Anthropic pin.
                if decode_route(active_protocol.load(Ordering::Relaxed)).0
                    == ProviderProtocol::Anthropic
                {
                    active_protocol.store(ProviderProtocol::Openai.to_u8(), Ordering::Relaxed);
                }
            }
            NativeAnthropicResult::Transient(response) => {
                // Endpoint exists (429/5xx); surface the error via the chat
                // cascade's first_error seed but leave the pin alone.
                native_anthropic_terminal = Some(response);
            }
            NativeAnthropicResult::UpstreamError => {
                // Endpoint may exist; don't flip the pin on transient errors.
            }
        }
    }

    // OR in the runtime-learned flag so requests after a successful in-cascade
    // recovery skip the wasted first attempt: without this, every request in
    // the same launch pays one 400 + retry round-trip until the process exits
    // and `persist_runtime_discoveries` writes the quirk back to the key.
    let effective_requires_reasoning =
        config.requires_reasoning_content || learned_requires_reasoning.load(Ordering::Relaxed);
    let mut simplified = build_simplified_openai_body(
        &body,
        effective_requires_reasoning,
        model_is_claude,
        config.strip_cache_control,
        config.max_tokens_cap,
    )?;
    // GPT-5.4+ at api.openai.com accepts `reasoning_effort: "xhigh"`, but
    // generic OpenAI-compat providers (Cloudflare Workers AI, OpenRouter, …)
    // strictly validate against the spec enum `low|medium|high` and 400 on
    // anything else. Clamp here so the same Max-tier request reaches both.
    if !is_direct_openai_base(&config.target_base_url) {
        crate::services::responses_chat_conversion::cap_reasoning_effort(&mut simplified);
    }
    let requested_stream = simplified
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let candidates = router_candidates(active_protocol, &config.target_base_url);
    // Seed first_error with the native-Anthropic Terminal response (if any) so
    // a chat fallback that also exhausts surfaces *some* error to the client.
    // The chat-loop's authoritative errors overwrite this with the more
    // diagnostic chat-shaped response when one is available.
    let mut first_error: FirstError<RouterResponse> = FirstError::seeded(native_anthropic_terminal);
    let mut quirk = QuirkRetryState::new(learned_requires_reasoning, effective_requires_reasoning);

    // Catalog (when cached) snaps the model name to the exact advertised id —
    // this is what makes Claude work on slug-style gateways (Requesty, Vercel).
    let catalog = crate::services::models_cache::ModelsCache::shared()
        .model_ids(&config.target_base_url)
        .await;

    let mut idx = 0;
    while idx < candidates.len() {
        let (protocol, variant) = candidates[idx];
        let attempt = idx;
        let mut req_body = simplified.clone();
        let mut attempt_headers = passthrough_headers.clone();
        prepare_gateway_model_metadata(
            &mut req_body,
            &mut attempt_headers,
            config,
            protocol,
            catalog.as_deref(),
        );
        // Keyed off the base: every protocol branch forwards these headers to it.
        opencode_session::insert_session_header(
            &mut attempt_headers,
            &config.target_base_url,
            Some(&req_body),
        );

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

        let outcome: AttemptOutcome<RouterResponse> = match protocol {
            ProviderProtocol::Google => {
                req_body["stream"] = json!(false);
                let model = openai_chat_model(&req_body, "gemini-2.5-pro");
                let google_body = translate_request(
                    &req_body,
                    &RequestOptions::ChatToGemini {
                        default_model: "gemini-2.5-pro",
                    },
                );
                let url = build_google_generate_content_url(&config.target_base_url, &model);
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("x-goog-api-key", config.target_api_key.as_str())
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&google_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;

                let status_code = response.status().as_u16();
                let response_body = response.text().await?;
                let parsed = if is_protocol_mismatch(status_code) {
                    None
                } else {
                    let google_response: Value = serde_json::from_str(&response_body)?;
                    let openai_response = translate_response(
                        &google_response,
                        &ResponseOptions::ChatToGemini { model: &model },
                    )?;
                    Some(openai_chat_response_to_anthropic_router(
                        &openai_response,
                        requested_stream,
                    )?)
                };
                classify_attempt(status_code, response_body, parsed)
            }
            ProviderProtocol::ResponsesApi => {
                let responses_body = try_convert_chat_to_responses_request(&req_body)?;
                let url = build_responses_url(&config.target_base_url, variant);
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("Authorization", format!("Bearer {}", config.target_api_key))
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&responses_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;

                let status_code = response.status().as_u16();
                let response_body = response.text().await?;
                let parsed = if is_protocol_mismatch(status_code) {
                    None
                } else {
                    let resp: Value = serde_json::from_str(&response_body)?;
                    let openai_response = try_convert_responses_json_to_chat(&resp)?;
                    Some(openai_chat_response_to_anthropic_router(
                        &openai_response,
                        requested_stream,
                    )?)
                };
                classify_attempt(status_code, response_body, parsed)
            }
            _ => {
                // OpenAI or Anthropic — use chat completions endpoint
                let url = http_utils::build_target_url(
                    &config.target_base_url,
                    variant.apply("/v1/chat/completions"),
                );
                disable_stream_for_inception_with_tools(&mut req_body, &config.target_base_url);
                let mut response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("Authorization", format!("Bearer {}", config.target_api_key))
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&req_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;

                let status_code = response.status().as_u16();
                if is_protocol_mismatch(status_code) {
                    let body = response.text().await.unwrap_or_default();
                    AttemptOutcome::Mismatch {
                        status: status_code,
                        body,
                    }
                } else {
                    let is_streaming = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .map(|ct| ct.contains("text/event-stream"))
                        .unwrap_or(false);

                    // Stream OpenAI SSE → Anthropic SSE directly to socket
                    if status_code == 200 && is_streaming {
                        use tokio::io::AsyncWriteExt;
                        let headers =
                            http_utils::http_chunked_response_head(200, "text/event-stream");
                        socket.write_all(headers.as_bytes()).await?;
                        let mut converter = OpenAIStreamConverter::new("claude");
                        while let Some(chunk) = response.chunk().await? {
                            // Raw SSE carries usage — include_usage is always set.
                            sniffer.observe(&chunk);
                            let converted = converter.push_bytes(&chunk)?;
                            if !converted.is_empty() {
                                let formatted = http_utils::format_http_chunk(converted.as_bytes());
                                socket.write_all(&formatted).await?;
                            }
                        }
                        let tail = converter.finish()?;
                        if !tail.is_empty() {
                            let formatted = http_utils::format_http_chunk(tail.as_bytes());
                            socket.write_all(&formatted).await?;
                        }
                        socket.write_all(b"0\r\n\r\n").await?;
                        commit_protocol_switch(active_protocol, protocol, variant, attempt);
                        slot.confirm();
                        record_slot_outcome(&slot, true);
                        return Ok(RouterResponse::AlreadyStreamed);
                    }

                    let response_body = response.text().await?;
                    let r = if status_code == 200 && response_body.starts_with("data:") {
                        let anthropic_sse =
                            convert_openai_sse_to_anthropic(&response_body, status_code)?;
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "text/event-stream".to_string(),
                            body: anthropic_sse.into_bytes(),
                        }
                    } else if status_code == 200 && requested_stream {
                        // Upstream returned JSON (Inception with tools forces
                        // stream:false) but the inbound client asked for SSE —
                        // re-frame as a one-shot stream.
                        let openai_resp: Value = serde_json::from_str(&response_body)?;
                        openai_chat_response_to_anthropic_router(&openai_resp, true)?
                    } else {
                        let anthropic_response =
                            convert_openai_to_anthropic(&response_body, status_code)?;
                        RouterResponse::Buffered {
                            status: status_code,
                            content_type: CONTENT_TYPE_JSON.to_string(),
                            body: anthropic_response.into_bytes(),
                        }
                    };
                    AttemptOutcome::Success(r)
                }
            }
        };

        match outcome {
            AttemptOutcome::Success(r) => {
                commit_protocol_switch(active_protocol, protocol, variant, attempt);
                slot.confirm();
                record_slot_outcome(&slot, true);
                return Ok(r);
            }
            AttemptOutcome::Mismatch {
                status,
                body: response_body,
            } => {
                let classification = classify_failed_attempt(status, &response_body);
                first_error.record_with(&classification, || RouterResponse::Buffered {
                    status,
                    content_type: CONTENT_TYPE_JSON.to_string(),
                    body: response_body.into_bytes(),
                });
                match mismatch_directive(
                    attempt,
                    &classification,
                    &slot,
                    protocol,
                    variant,
                    Some(&mut quirk),
                ) {
                    MismatchDirective::RetrySameCandidate => {
                        // Same-launch recovery: the upstream told us it needs
                        // `reasoning_content`; rebuild `simplified` with strict
                        // mode and retry the same (protocol, variant) so the
                        // *current* request succeeds without a relaunch.
                        if let Ok(mut strict) = build_simplified_openai_body(
                            &body,
                            true,
                            model_is_claude,
                            config.strip_cache_control,
                            config.max_tokens_cap,
                        ) {
                            if !is_direct_openai_base(&config.target_base_url) {
                                crate::services::responses_chat_conversion::cap_reasoning_effort(
                                    &mut strict,
                                );
                            }
                            simplified = strict;
                            continue; // re-do the SAME idx with strict body
                        }
                        // Rebuild failed — treat as a plain semantic bail.
                        commit_protocol_switch(active_protocol, protocol, variant, attempt);
                        break;
                    }
                    MismatchDirective::Bail => break,
                    MismatchDirective::NextCandidate => {}
                }
            }
        }
        idx += 1;
    }

    // Failure-streak valve: after CONSECUTIVE_FAILURES_BEFORE_RESET exhausted
    // requests the learned pin resets, so a long-lived session can recover
    // from a stale route without a process restart.
    record_slot_outcome(&slot, false);
    Ok(first_error.take().unwrap_or(RouterResponse::Buffered {
        status: 503,
        content_type: CONTENT_TYPE_JSON.to_string(),
        body: b"{\"error\":\"No compatible protocol found\"}".to_vec(),
    }))
}

/// Apply the full Anthropic → OpenAI transform pipeline including the
/// post-conversion adjustments (cache_control injection / strip,
/// reasoning_effort mapping, max_tokens cap). Extracted as a function so
/// the cascade can rebuild the body with `requires_reasoning_content=true`
/// and re-run the same pipeline when an upstream signals it requires the
/// quirk — see the in-cascade retry below.
fn build_simplified_openai_body(
    anthropic_body: &Value,
    requires_reasoning_content: bool,
    model_is_claude: bool,
    strip_cache_control: bool,
    max_tokens_cap: Option<u64>,
) -> Result<Value> {
    let mut simplified = anthropic_to_openai(anthropic_body, requires_reasoning_content)?;
    // Only inject cache_control for Claude models — other providers don't support it
    // and strict ones (e.g. Cloudflare Workers AI) reject unknown fields / array content.
    if model_is_claude && !strip_cache_control {
        inject_chat_completions_cache_control(&mut simplified);
    }
    // Bedrock-style shims reject `cache_control` even when it just passes
    // through from a Claude-shaped client; remove it before forwarding.
    if strip_cache_control {
        crate::services::anthropic_route_pipeline::strip_cache_control(&mut simplified);
    }
    if !simplified
        .as_object()
        .is_some_and(|m| m.contains_key("reasoning_effort"))
        && let Some(effort) = crate::services::effort::extract_anthropic_effort(anthropic_body)
    {
        simplified["reasoning_effort"] = json!(effort.to_openai_effort());
    }
    cap_max_tokens_field(&mut simplified, max_tokens_cap);
    Ok(simplified)
}

fn anthropic_to_openai(body: &Value, requires_reasoning_content: bool) -> Result<Value> {
    let mut req = translate_request(
        body,
        &RequestOptions::AnthropicToChat(&AnthropicToOpenAIConfig {
            default_model: "gpt-4o",
            preserve_stream: true,
            model_transform: None,
            include_reasoning_content: true,
            require_non_empty_reasoning_content: requires_reasoning_content,
            stringify_other_tool_result_content: true,
            tool_result_supports_multimodal: true,
            fallback_tool_arguments_json: "{}",
        }),
    );
    let mut typed_req: OpenAIChatRequest = serde_json::from_value(req)
        .context("failed to convert anthropic request to typed OpenAI request")?;
    stringify_typed_message_content(&mut typed_req);
    req = serde_json::to_value(typed_req).context("failed to serialize typed OpenAI request")?;
    Ok(req)
}

/// Flatten any array-valued `content` fields to plain strings.
/// Strict OpenAI-compatible providers (e.g. Cloudflare Workers AI) reject
/// the multi-part content arrays that the standard OpenAI API accepts.
#[cfg(test)]
fn stringify_message_content(req: &mut Value) {
    let Ok(mut typed_req) = serde_json::from_value::<OpenAIChatRequest>(req.clone()) else {
        return;
    };
    stringify_typed_message_content(&mut typed_req);
    *req = serde_json::to_value(typed_req).expect("typed openai request should serialize");
}

/// Build /v1/responses (or /responses, when stripped) URL from a base URL.
fn build_responses_url(base_url: &str, variant: PathVariant) -> String {
    http_utils::build_target_url(base_url, variant.apply("/v1/responses"))
}

/// Wrap an OpenAI Chat Completions response as a buffered Anthropic-format
/// `RouterResponse`, emitting SSE when `streaming` is true and JSON otherwise.
fn openai_chat_response_to_anthropic_router(
    openai_response: &Value,
    streaming: bool,
) -> Result<RouterResponse> {
    if streaming {
        let openai_sse = convert_openai_chat_response_to_sse(openai_response)?;
        let anthropic_sse = convert_openai_sse_to_anthropic(&openai_sse, 200)?;
        Ok(RouterResponse::Buffered {
            status: 200,
            content_type: "text/event-stream".to_string(),
            body: anthropic_sse.into_bytes(),
        })
    } else {
        Ok(RouterResponse::Buffered {
            status: 200,
            content_type: CONTENT_TYPE_JSON.to_string(),
            body: convert_openai_to_anthropic(&openai_response.to_string(), 200)?.into_bytes(),
        })
    }
}

/// Builds the per-request fallback cascade for this router.
///
/// Drops two candidate kinds before the cascade runs:
/// - `Anthropic`: would just hit /v1/chat/completions again (byte-identical
///   to the corresponding `Openai` candidate handled by the catch-all arm).
///   Native Anthropic forwarding lives in `try_native_anthropic`, not here.
/// - `ResponsesApi` against non-OpenAI hosts: `/responses` is OpenAI-specific.
///   Other vendors that expose that route (e.g. Cloudflare Workers AI)
///   implement a different schema, so posting OpenAI-Responses bodies there
///   just produces a misleading second 400 that masks the real
///   `/chat/completions` error.
fn router_candidates(
    active_protocol: &AtomicU8,
    target_base_url: &str,
) -> Vec<(ProviderProtocol, PathVariant)> {
    let allow_responses_fallback = is_direct_openai_base(target_base_url);
    protocol_candidates(active_protocol)
        .into_iter()
        .filter(|(proto, _)| *proto != ProviderProtocol::Anthropic)
        .filter(|(proto, _)| allow_responses_fallback || *proto != ProviderProtocol::ResponsesApi)
        .collect()
}

fn prepare_gateway_model_metadata(
    simplified: &mut Value,
    passthrough_headers: &mut HeaderMap,
    config: &AnthropicToOpenAIRouterConfig,
    protocol: ProviderProtocol,
    catalog: Option<&[String]>,
) {
    let requested_model = simplified
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let selected_model = select_model_for_provider_attempt(
        catalog,
        &config.target_base_url,
        Some(&requested_model),
        None,
        protocol,
    );
    warn_on_model_substitution(&requested_model, &selected_model);
    simplified["model"] = Value::String(selected_model);

    if is_gateway_style_endpoint(&config.target_base_url)
        && !passthrough_headers.contains_key("x-provider")
        && let Some(provider) = infer_provider_name_from_model(&requested_model)
        && let Ok(value) = HeaderValue::from_str(&provider)
    {
        passthrough_headers.insert("x-provider", value);
    }
}

/// Surface a cross-vendor model swap once per pair — substitution is legitimate
/// for single-vendor upstreams, but must never be silent.
fn warn_on_model_substitution(requested: &str, selected: &str) {
    if requested.is_empty() || requested == selected {
        return;
    }
    // Snapping a name to its catalog id is the same model; only vendor changes matter.
    let (Some(from), Some(to)) = (
        infer_provider_name_from_model(requested),
        infer_provider_name_from_model(selected),
    ) else {
        return;
    };
    if from == to {
        return;
    }

    static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let Ok(mut seen) = seen.lock() else {
        return;
    };
    if !seen.insert((requested.to_string(), selected.to_string())) {
        return;
    }
    eprintln!(
        "  • Upstream can't serve '{requested}' over this protocol — sending '{selected}' instead"
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_try_native_anthropic_when_route_is_anthropic() {
        // Tool-native default for `aivo claude`: an Anthropic slot route probes
        // /v1/messages first.
        let active = AtomicU8::new(ProviderProtocol::Anthropic.to_u8());
        assert!(should_try_native_anthropic(&active));
    }

    #[test]
    fn should_skip_native_anthropic_when_route_moved_off_anthropic() {
        // Once a model's route is learned/pinned to a non-Anthropic protocol,
        // re-probing /v1/messages would short-circuit to EndpointMissing and
        // undo the learned pin.
        for pin in [
            ProviderProtocol::Openai,
            ProviderProtocol::ResponsesApi,
            ProviderProtocol::Google,
        ] {
            let active = AtomicU8::new(pin.to_u8());
            assert!(!should_try_native_anthropic(&active), "pin={pin:?}");
        }
    }

    #[test]
    fn build_anthropic_messages_url_without_prefix() {
        assert_eq!(
            build_anthropic_messages_url("https://api.deepseek.com", None),
            "https://api.deepseek.com/v1/messages",
        );
        assert_eq!(
            build_anthropic_messages_url("https://api.deepseek.com/v1", None),
            "https://api.deepseek.com/v1/messages",
        );
        assert_eq!(
            build_anthropic_messages_url("https://api.deepseek.com/v1/", None),
            "https://api.deepseek.com/v1/messages",
        );
    }

    #[test]
    fn build_anthropic_messages_url_with_prefix_normalises_slashes() {
        for base in [
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com/v1/",
        ] {
            for prefix in ["/anthropic", "anthropic", "/anthropic/"] {
                assert_eq!(
                    build_anthropic_messages_url(base, Some(prefix)),
                    "https://api.deepseek.com/anthropic/v1/messages",
                    "base={base} prefix={prefix}",
                );
            }
        }
    }

    #[test]
    fn probe_paths_falls_back_to_anthropic_prefix_for_unlearned_host() {
        // Unlearned host → probe root first, then /anthropic. Hosts that mount
        // Anthropic only at /anthropic/v1/messages self-discover via the trailing
        // fallback and pin the result, so subsequent launches skip the probe.
        assert_eq!(
            probe_paths(ProbeOutcome::Unlearned),
            Some(&[AnthropicProbePath::Root, AnthropicProbePath::Prefixed][..]),
        );
    }

    #[test]
    fn probe_paths_locks_in_after_first_success() {
        assert_eq!(
            probe_paths(ProbeOutcome::UseRoot),
            Some(&[AnthropicProbePath::Root][..]),
        );
        assert_eq!(
            probe_paths(ProbeOutcome::UsePrefixed),
            Some(&[AnthropicProbePath::Prefixed][..]),
        );
    }

    #[test]
    fn probe_paths_returns_none_when_failed() {
        assert_eq!(probe_paths(ProbeOutcome::Failed), None);
    }

    #[test]
    fn probe_state_counts_consecutive_upstream_errors() {
        let probe = ProbeState::new();
        assert!(!probe.record_upstream_error()); // 1
        assert!(!probe.record_upstream_error()); // 2
        // Third tick reaches the limit (PROBE_UPSTREAM_ERROR_LIMIT = 3).
        assert!(probe.record_upstream_error());
    }

    #[test]
    fn probe_state_resets_streak_on_success() {
        let probe = ProbeState::new();
        let _ = probe.record_upstream_error();
        let _ = probe.record_upstream_error();
        probe.reset_upstream_error_streak();
        // Counter is back to 0; need three more errors to reach the limit.
        assert!(!probe.record_upstream_error());
        assert!(!probe.record_upstream_error());
        assert!(probe.record_upstream_error());
    }

    #[test]
    fn anthropic_to_openai_preserves_tool_result_images_through_typed_roundtrip() {
        // Regression for P0-1 Tier B: the typed round-trip in
        // `anthropic_to_openai` (OpenAIChatRequest → stringify →
        // OpenAIChatRequest) used to strip image_url parts. Images in
        // tool_result should now reach the upstream request intact.
        let body = json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_screenshot",
                    "content": [
                        {"type": "text", "text": "Screenshot of the home page."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}}
                    ]
                }]
            }]
        });
        let req = anthropic_to_openai(&body, false).expect("conversion succeeds");
        let tool_msg = &req["messages"][0];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["tool_call_id"], "toolu_screenshot");
        let content = tool_msg["content"]
            .as_array()
            .expect("multimodal content array survives round-trip");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Screenshot of the home page.");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo"
        );
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

        let req = anthropic_to_openai(&body, false).unwrap();
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
    fn test_anthropic_to_openai_strips_sampling_for_rejecting_models() {
        // o3 rejects temperature/top_p — forwarding them 400s upstream.
        let body = json!({
            "model": "o3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9
        });
        let req = anthropic_to_openai(&body, false).unwrap();
        assert!(req.get("temperature").is_none());
        assert!(req.get("top_p").is_none());
        assert_eq!(req["max_tokens"], 128);
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

        let req = anthropic_to_openai(&body, true).unwrap();
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

        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], " ");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_anthropic_to_openai_sets_reasoning_content_for_plain_assistant_text_without_thinking() {
        let body = json!({
            "model": "aivo/starter",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "OK, continuing."}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "OK, continuing.");
        assert_eq!(messages[0]["reasoning_content"], "OK, continuing.");
        assert!(messages[0].get("tool_calls").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_sets_placeholder_reasoning_for_empty_assistant_text() {
        let body = json!({
            "model": "aivo/starter",
            "messages": [{
                "role": "assistant",
                "content": []
            }]
        });

        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], " ");
    }

    #[test]
    fn test_anthropic_to_openai_sets_reasoning_content_for_string_content_assistant_in_strict_mode()
    {
        // An assistant turn with `content: "..."` (string form, not array
        // of blocks) bypassed the strict-mode fallback. This is the exact
        // shape DeepSeek's interleaved-thinking session sends back for brief
        // acknowledgements — and it triggers the 400 even after learning the
        // per-key quirk if the bridge doesn't add the field.
        let body = json!({
            "model": "deepseek-reasoner",
            "messages": [{
                "role": "assistant",
                "content": "OK, continuing."
            }]
        });
        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "OK, continuing.");
        assert_eq!(messages[0]["reasoning_content"], "OK, continuing.");
    }

    #[test]
    fn test_anthropic_to_openai_does_not_set_reasoning_for_user_string_content_in_strict_mode() {
        // The strict fallback only applies to assistant turns. User string
        // content must not get a `reasoning_content` field.
        let body = json!({
            "model": "deepseek-reasoner",
            "messages": [{
                "role": "user",
                "content": "hello"
            }]
        });
        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert!(messages[0].get("reasoning_content").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_omits_reasoning_for_plain_assistant_text_when_not_required() {
        let body = json!({
            "model": "aivo/starter",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "OK, continuing."}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "OK, continuing.");
        assert!(messages[0].get("reasoning_content").is_none());
    }

    #[test]
    fn test_prepare_gateway_model_metadata_preserves_gateway_claude_model() {
        let config = AnthropicToOpenAIRouterConfig {
            target_base_url: "https://api.ai.example-gateway.net/endpoint".to_string(),
            target_api_key: "test".to_string(),
            seed_routes: BTreeMap::new(),
            strip_cache_control: false,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(
            &mut body,
            &mut headers,
            &config,
            ProviderProtocol::Openai,
            None,
        );

        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(
            headers.get("x-provider").and_then(|v| v.to_str().ok()),
            Some("anthropic")
        );
    }

    #[test]
    fn test_prepare_gateway_model_metadata_remaps_plain_openai_endpoint() {
        let config = AnthropicToOpenAIRouterConfig {
            target_base_url: "https://api.openai.com/v1".to_string(),
            target_api_key: "test".to_string(),
            seed_routes: BTreeMap::new(),
            strip_cache_control: false,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(
            &mut body,
            &mut headers,
            &config,
            ProviderProtocol::Openai,
            None,
        );

        assert_eq!(body["model"], "gpt-4o");
        assert!(headers.get("x-provider").is_none());
    }

    #[test]
    fn native_pin_flip_only_on_auth_statuses() {
        assert!(native_terminal_flips_pin(401));
        assert!(native_terminal_flips_pin(403));
        // Transient statuses must never flip a working native-Anthropic pin.
        for status in [429, 500, 502, 503, 504] {
            assert!(!native_terminal_flips_pin(status), "status {status}");
        }
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

    #[test]
    fn test_anthropic_to_openai_keeps_content_on_tool_call_messages() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();

        // content must be present (null) for strict providers like Cloudflare Workers AI
        assert!(
            messages[0].get("content").is_some(),
            "assistant tool_call message must retain content field"
        );
        assert!(messages[0]["tool_calls"].is_array());
    }

    #[test]
    fn test_stringify_message_content_flattens_arrays() {
        let mut req = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
                {"role": "assistant", "content": "already a string"},
                {"role": "user", "content": ["plain", "strings"]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}
            ]
        });

        stringify_message_content(&mut req);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["content"], "hello\nworld");
        assert_eq!(messages[1]["content"], "already a string");
        assert_eq!(messages[2]["content"], "plain\nstrings");
        // Array carrying a non-text kind stays as an array — flattening
        // would silently erase the payload, and strict providers should
        // fail loudly rather than receive corrupted data.
        assert!(messages[3]["content"].is_array());
        assert_eq!(messages[3]["content"][0]["type"], "image_url");
    }

    #[test]
    fn test_anthropic_to_openai_empty_messages() {
        let body = json!({
            "model": "gpt-4o",
            "messages": []
        });
        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_cap_max_tokens_field_no_cap() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 12000});
        cap_max_tokens_field(&mut req, None);
        assert_eq!(req["max_tokens"], 12000);
    }

    #[test]
    fn test_cap_max_tokens_field_under_cap() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 4096});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 4096);
    }

    #[test]
    fn test_stringify_message_content_null_content() {
        let mut req = json!({
            "messages": [{"role": "assistant", "content": null}]
        });
        stringify_message_content(&mut req);
        // null content should remain unchanged (not crash)
        let messages = req["messages"].as_array().unwrap();
        assert!(messages[0]["content"].is_null());
    }

    #[test]
    fn build_responses_url_with_v1_suffix() {
        let url = build_responses_url("https://api.openai.com/v1", PathVariant::Default);
        assert_eq!(url, "https://api.openai.com/v1/responses");
        // Must not produce /v1/v1/responses
        assert!(!url.contains("/v1/v1"));
    }

    #[test]
    fn build_responses_url_without_v1_suffix() {
        let url = build_responses_url("https://api.example.com", PathVariant::Default);
        assert_eq!(url, "https://api.example.com/v1/responses");
    }

    #[test]
    fn build_responses_url_stripped_variant() {
        let url = build_responses_url("https://api.example.com", PathVariant::Stripped);
        assert_eq!(url, "https://api.example.com/responses");
    }

    #[test]
    fn router_candidates_drops_anthropic_always() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        let cands = router_candidates(&active, "https://api.openai.com/v1");
        assert!(!cands.iter().any(|(p, _)| *p == ProviderProtocol::Anthropic));
    }

    #[test]
    fn router_candidates_keeps_responses_for_openai() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        let cands = router_candidates(&active, "https://api.openai.com/v1");
        assert!(
            cands
                .iter()
                .any(|(p, _)| *p == ProviderProtocol::ResponsesApi)
        );
    }

    #[test]
    fn router_candidates_drops_responses_for_non_openai() {
        let active = AtomicU8::new(ProviderProtocol::Openai.to_u8());
        let cands = router_candidates(
            &active,
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1",
        );
        assert!(
            !cands
                .iter()
                .any(|(p, _)| *p == ProviderProtocol::ResponsesApi)
        );
        // Other fallbacks (Openai variants, Google) still present.
        assert!(cands.iter().any(|(p, _)| *p == ProviderProtocol::Google));
    }
}
