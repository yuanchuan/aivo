//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, ensure_assistant_reasoning_content_in_chat_request,
};
use crate::services::anthropic_chat_response::{OpenAIToAnthropicConfig, UsageValueMode};
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils::{self, router_http_client_with_timeout};
use crate::services::log_store::{LogEvent, LogStore};
use crate::services::model_catalog::fetch_models;
use crate::services::model_list_response;
use crate::services::model_names::strip_context_suffix;
use crate::services::openai_anthropic_bridge::convert_openai_chat_response_to_sse;
use crate::services::openai_gemini_bridge::convert_openai_chat_to_gemini_sse;
use crate::services::protocol_fallback::{
    FirstError, MismatchDirective, commit_protocol_switch, mismatch_directive, protocol_candidates,
    record_request_outcome,
};
use crate::services::provider_protocol::{
    PathVariant, ProviderProtocol, classify_failed_attempt, is_protocol_mismatch,
};
use crate::services::request_log::RequestLogger;
use crate::services::responses_to_chat_router::{
    ResponsesToChatRouterConfig, ToolNamespaceMap, collect_custom_tool_names,
    collect_namespace_tool_names, convert_chat_response_to_responses_sse,
};
use crate::services::route_cache::{RouteCache, RouteSlot};
use crate::services::serve_responses::convert_chat_sse_to_responses_sse;
use crate::services::serve_upstream::{
    RouterResponse, StreamingBody, UpstreamRequestContext, copilot_requires_responses_api,
    send_anthropic_chat, send_anthropic_native, send_claude_oauth_passthrough,
    send_codex_responses, send_copilot_responses, send_gemini_chat, send_gemini_native,
    send_openai_chat, send_openai_embeddings,
};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::token_usage::{StreamUsageSniffer, TokenUsage, parse_token_usage};
use crate::services::usage_stats_store::RunTokenTally;
use crate::services::wire_format::{
    Chain, RequestOptions, ResponseOptions, StreamOptions, WireFormat, stream_adapter,
    translate_request, translate_response,
};

use std::sync::LazyLock;

static HEALTH_RESPONSE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    json!({"status": "ok", "version": crate::version::VERSION})
        .to_string()
        .into_bytes()
});

/// A random 32-char alphanumeric bearer token for a serve/endpoint instance.
pub(crate) fn random_auth_token() -> String {
    use rand::Rng;
    rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

pub struct ServeRouterConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub upstream_protocol: ProviderProtocol,
    pub is_copilot: bool,
    /// `upstream_api_key` holds the grok credential JSON, not a bearer.
    pub is_grok: bool,
    /// `XAI_API_KEY` fallback for grok 403s; `None` if unset.
    pub grok_fallback_api_key: Option<String>,
    /// `upstream_api_key` holds the Kimi OAuth credential JSON, not a bearer.
    pub is_kimi: bool,
    /// `upstream_api_key` holds the Codex OAuth credential JSON, not a bearer.
    pub is_codex: bool,
    /// `upstream_api_key` holds the Claude OAuth credential JSON; requests
    /// forward verbatim to the native backend.
    pub is_claude_native_oauth: bool,
    pub is_openrouter: bool,
    pub is_starter: bool,
    /// Upstream requires `reasoning_content` on assistant turns (deepseek/moonshot);
    /// injected to avoid a 400. From the provider profile at startup.
    pub requires_reasoning_content: bool,
    pub cors: bool,
    pub timeout: u64,
    pub auth_token: Option<String>,
    /// Snapshot of model aliases taken at startup. The router rewrites
    /// `body["model"]` against this map before any protocol-specific handler
    /// runs; clients that POST `{"model": "<alias>", ...}` reach the real
    /// upstream model. Edits to aliases after launch require a restart.
    pub aliases: HashMap<String, String>,
}

impl ServeRouterConfig {
    /// Build the config from a resolved key, deriving the upstream URL +
    /// protocol/provider flags from its provider profile. The caller supplies
    /// the serving knobs (cors/timeout/auth/aliases). Shared by `aivo serve` and
    /// the plugin loopback endpoint.
    pub(crate) fn from_key(
        key: &ApiKey,
        cors: bool,
        timeout: u64,
        auth_token: Option<String>,
        aliases: HashMap<String, String>,
    ) -> Self {
        use crate::services::provider_profile::{
            provider_profile_for_key, resolve_starter_base_url,
        };
        let profile = provider_profile_for_key(key);
        // The claude-oauth sentinel has no provider profile; force Anthropic.
        let is_claude_native_oauth = key.is_claude_oauth();
        Self {
            upstream_base_url: resolve_starter_base_url(&key.base_url),
            upstream_api_key: key.key.as_str().to_string(),
            upstream_protocol: if is_claude_native_oauth {
                ProviderProtocol::Anthropic
            } else {
                profile.default_protocol
            },
            is_copilot: profile.serve_flags.is_copilot,
            is_grok: key.is_grok_oauth(),
            grok_fallback_api_key: None,
            is_kimi: key.is_kimi_oauth(),
            is_codex: key.is_codex_oauth(),
            is_claude_native_oauth,
            is_openrouter: profile.serve_flags.is_openrouter,
            is_starter: profile.serve_flags.is_starter,
            requires_reasoning_content: profile.quirks.requires_reasoning_content,
            cors,
            timeout,
            auth_token,
            aliases,
        }
    }

    /// Attaches the `XAI_API_KEY` 403 fallback. No-op for non-grok configs.
    pub(crate) fn with_grok_fallback(mut self, api_key: Option<String>) -> Self {
        if self.is_grok {
            self.grok_fallback_api_key = api_key;
        }
        self
    }
}

pub struct ServeRouter {
    config: ServeRouterConfig,
    key: ApiKey,
    log_store: LogStore,
    logger: Option<RequestLogger>,
    failover_keys: Vec<ApiKey>,
    /// Per-model upstream keys `(model, key)`; each becomes a routable upstream
    /// at startup. Powers per-tier routing.
    model_upstreams: Vec<(String, ApiKey)>,
    /// When set, buffered 2xx responses have their token usage recorded against
    /// `key` in stats. Off by default (plain `aivo serve` doesn't account); the
    /// plugin endpoint opts in via `with_usage_accounting`.
    usage_sink: Option<SessionStore>,
    /// Tool label for accounted requests (the plugin name); `None` → "serve".
    usage_tool: Option<String>,
    /// Per-run token tally for the plugin endpoint, so the run's finished log row
    /// carries timestamped tokens (windowable by `aivo stats --since`). Fed at the
    /// same point as `usage_sink`; `None` for plain `aivo serve`.
    run_tally: Option<Arc<RunTokenTally>>,
    /// Suppress the router's progress lines (protocol auto-switch, failover) on
    /// stderr. `aivo code` runs this router in-process behind a raw-mode TUI, so
    /// stray `eprintln!`s would corrupt the screen / land in the prompt box.
    quiet: bool,
    /// Caller-owned route cache for the primary upstream. When set, the serve
    /// learns into it instead of a throwaway one, so the owner can seed a known
    /// protocol and read confirmed routes back. `aivo code` shares one across its
    /// per-turn serves.
    seed_route_cache: Option<Arc<RouteCache>>,
    /// Persists provider-OAuth (grok/kimi/codex) refresh-token rotations;
    /// ignored for other upstreams.
    oauth_persist_store: Option<SessionStore>,
}

struct ServeState {
    config: Arc<ServeRouterConfig>,
    client: reqwest::Client,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
    grok_tokens: Option<Arc<crate::services::grok_oauth::GrokTokenManager>>,
    kimi_tokens: Option<Arc<crate::services::kimi_oauth::KimiTokenManager>>,
    codex_tokens: Option<Arc<crate::services::codex_oauth::CodexTokenManager>>,
    /// Per-model learned protocol routes (in-memory only — `aivo serve` doesn't
    /// persist routes yet). Replaces the old single per-process pin so a
    /// multi-model gateway key learns a route per model instead of thrashing
    /// one scalar.
    route_cache: Arc<RouteCache>,
    log_store: LogStore,
    logger: Option<RequestLogger>,
    failover_keys: Arc<Vec<FailoverEntry>>,
    /// Per-model upstreams keyed by (suffix-stripped) model name. A matching
    /// `body["model"]` swaps the request to that upstream. Empty for single-key.
    model_upstreams: Arc<HashMap<String, Arc<FailoverEntry>>>,
    shutdown: Arc<tokio::sync::Notify>,
    usage_sink: Option<SessionStore>,
    usage_tool: Option<String>,
    run_tally: Option<Arc<RunTokenTally>>,
    /// Mirror of `ServeRouter::quiet` — suppresses stderr progress lines.
    quiet: bool,
}

struct FailoverEntry {
    config: Arc<ServeRouterConfig>,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
    grok_tokens: Option<Arc<crate::services::grok_oauth::GrokTokenManager>>,
    kimi_tokens: Option<Arc<crate::services::kimi_oauth::KimiTokenManager>>,
    codex_tokens: Option<Arc<crate::services::codex_oauth::CodexTokenManager>>,
    /// Shared across failover attempts so a route learned during one failover
    /// carries to the next request instead of being re-probed every time.
    route_cache: Arc<RouteCache>,
}

impl ServeRouter {
    pub fn new(config: ServeRouterConfig, key: ApiKey, log_store: LogStore) -> Self {
        Self {
            config,
            key,
            log_store,
            logger: None,
            failover_keys: Vec::new(),
            model_upstreams: Vec::new(),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
            seed_route_cache: None,
            oauth_persist_store: None,
        }
    }

    /// Persist provider-OAuth (grok/kimi/codex) refresh-token rotations.
    /// No-op for other upstreams.
    pub fn with_oauth_persist(mut self, store: SessionStore) -> Self {
        self.oauth_persist_store = Some(store);
        self
    }

    /// Use a caller-owned route cache so the learned protocol can be seeded and
    /// read back. Used by `aivo code` to share one across its per-turn serves.
    pub fn with_route_cache(mut self, cache: Arc<RouteCache>) -> Self {
        self.seed_route_cache = Some(cache);
        self
    }

    pub fn with_logger(mut self, logger: Option<RequestLogger>) -> Self {
        self.logger = logger;
        self
    }

    /// Silence the router's stderr progress lines (protocol auto-switch,
    /// failover). Set by `aivo code`, whose TUI owns the terminal.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    pub fn with_failover_keys(mut self, keys: Vec<ApiKey>) -> Self {
        self.failover_keys = keys;
        self
    }

    /// Route requests naming `model` to `key` instead of the base (per-tier
    /// routing). Model matched with its `[1m]`/`[2m]` suffix stripped.
    pub fn with_model_upstreams(mut self, upstreams: Vec<(String, ApiKey)>) -> Self {
        self.model_upstreams = upstreams;
        self
    }

    /// Record token usage of buffered 2xx responses against the bound key,
    /// labeling them with `tool` in logs. Used by the plugin endpoint so a
    /// coding-agent plugin routing through the loopback gets token/cost stats.
    pub fn with_usage_accounting(mut self, store: SessionStore, tool: String) -> Self {
        self.usage_sink = Some(store);
        self.usage_tool = Some(tool);
        self
    }

    /// Also fold accounted usage into a per-run tally, so the plugin run's
    /// finished log row carries timestamped tokens for `aivo stats --since`.
    pub fn with_run_tally(mut self, tally: Arc<RunTokenTally>) -> Self {
        self.run_tally = Some(tally);
        self
    }

    /// Binds to the port eagerly (propagates "address already in use" immediately),
    /// then spawns the accept loop in the background and returns the join handle
    /// and a shutdown notifier.
    pub async fn start_background(
        self,
        host: &str,
        port: u16,
    ) -> Result<(
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
    )> {
        let listener = bind_serve_listener(host, port).await?;
        Ok(self.spawn_on(listener))
    }

    /// Like `start_background`, but also returns the actually-bound port — for
    /// `port: 0` (OS-assigned), which the plugin endpoint uses to avoid clashing
    /// with a user's own `aivo serve`.
    pub async fn start_background_with_addr(
        self,
        host: &str,
        port: u16,
    ) -> Result<(
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
        u16,
    )> {
        let listener = bind_serve_listener(host, port).await?;
        let bound = listener.local_addr()?.port();
        let (handle, shutdown) = self.spawn_on(listener);
        Ok((handle, shutdown, bound))
    }

    /// Build the shared state and spawn the accept loop over an already-bound
    /// listener.
    fn spawn_on(
        self,
        listener: tokio::net::TcpListener,
    ) -> (
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
    ) {
        let copilot_tokens = if self.config.is_copilot {
            Some(Arc::new(CopilotTokenManager::new(
                self.config.upstream_api_key.clone(),
            )))
        } else {
            None
        };
        let grok_tokens = build_grok_tokens(&self.config, self.oauth_persist_store.clone());
        let kimi_tokens = build_kimi_tokens(&self.config, self.oauth_persist_store.clone());
        let codex_tokens = build_codex_tokens(&self.config, self.oauth_persist_store.clone());

        let initial_protocol = self.config.upstream_protocol;
        let timeout = self.config.timeout;
        // A caller-owned cache (shared across `aivo code` turns) wins; otherwise
        // each run gets a fresh throwaway cache seeded at the upstream protocol.
        let route_cache = self.seed_route_cache.unwrap_or_else(|| {
            Arc::new(RouteCache::new(
                "serve",
                initial_protocol,
                std::collections::BTreeMap::new(),
            ))
        });

        // Failover handlers re-parse the RAW request, so each failover config
        // needs the alias map too — an aliased model name would otherwise
        // reach the failover upstream unresolved and 400.
        let failover_aliases = self.config.aliases.clone();
        let failover_entries: Vec<FailoverEntry> = self
            .failover_keys
            .into_iter()
            .map(|fk| {
                let profile = crate::services::provider_profile::provider_profile_for_key(&fk);
                let is_copilot = profile.serve_flags.is_copilot;
                let protocol = profile.default_protocol;
                let ct = if is_copilot {
                    Some(Arc::new(CopilotTokenManager::new(
                        fk.key.as_str().to_string(),
                    )))
                } else {
                    None
                };
                FailoverEntry {
                    config: Arc::new(ServeRouterConfig {
                        upstream_base_url:
                            crate::services::provider_profile::resolve_starter_base_url(
                                &fk.base_url,
                            ),
                        upstream_api_key: fk.key.as_str().to_string(),
                        upstream_protocol: protocol,
                        is_copilot,
                        // Failover keys never carry grok/kimi/codex/claude-oauth state.
                        is_grok: false,
                        grok_fallback_api_key: None,
                        is_kimi: false,
                        is_codex: false,
                        is_claude_native_oauth: false,
                        is_openrouter: profile.serve_flags.is_openrouter,
                        is_starter: profile.serve_flags.is_starter,
                        requires_reasoning_content: profile.quirks.requires_reasoning_content,
                        cors: false,
                        timeout,
                        auth_token: None,
                        aliases: failover_aliases.clone(),
                    }),
                    key: fk,
                    copilot_tokens: ct,
                    grok_tokens: None,
                    kimi_tokens: None,
                    codex_tokens: None,
                    route_cache: Arc::new(RouteCache::new(
                        "serve",
                        protocol,
                        std::collections::BTreeMap::new(),
                    )),
                }
            })
            .collect();

        // Each tier key → a full upstream context, keyed by suffix-stripped model.
        let model_upstreams: HashMap<String, Arc<FailoverEntry>> = self
            .model_upstreams
            .into_iter()
            .map(|(model, key)| {
                let entry = build_upstream_entry(key, timeout, self.oauth_persist_store.as_ref());
                (strip_context_suffix(&model).to_string(), Arc::new(entry))
            })
            .collect();

        let shutdown = Arc::new(tokio::sync::Notify::new());

        let state = Arc::new(ServeState {
            config: Arc::new(self.config),
            client: router_http_client_with_timeout(timeout),
            key: self.key,
            copilot_tokens,
            grok_tokens,
            kimi_tokens,
            codex_tokens,
            route_cache,
            log_store: self.log_store,
            logger: self.logger,
            failover_keys: Arc::new(failover_entries),
            model_upstreams: Arc::new(model_upstreams),
            shutdown: shutdown.clone(),
            usage_sink: self.usage_sink,
            usage_tool: self.usage_tool,
            run_tally: self.run_tally,
            quiet: self.quiet,
        });

        (tokio::spawn(run_accept_loop(listener, state)), shutdown)
    }
}

/// `port == 0` goes through `bind_concrete_ephemeral` (reachable under WSL
/// VirtioProxy, issue #22); an explicit port binds directly.
pub(crate) async fn bind_serve_listener(host: &str, port: u16) -> Result<tokio::net::TcpListener> {
    if port == 0 {
        crate::services::http_utils::bind_concrete_ephemeral(host).await
    } else {
        Ok(tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?)
    }
}

async fn run_accept_loop(listener: tokio::net::TcpListener, state: Arc<ServeState>) -> Result<()> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(100));
    let cors = state.config.cors;
    let cors_extra = if cors {
        http_utils::cors_header_block()
    } else {
        ""
    };
    // Accept both forms Claude Code sends: `Authorization: Bearer <token>`
    // (when ANTHROPIC_AUTH_TOKEN is set) and `x-api-key: <token>` (when
    // ANTHROPIC_API_KEY is set); Google SDKs send `x-goog-api-key` / `?key=`.
    let expected_token: Option<Arc<str>> = state
        .config
        .auth_token
        .as_ref()
        .map(|t| Arc::from(t.as_str()));

    loop {
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = state.shutdown.notified() => {
                // Wait for in-flight requests to finish (max 5s)
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    semaphore.acquire_many(100),
                ).await;
                return Ok(());
            }
        };
        // Transient accept errors (ECONNABORTED on client reset, EMFILE under
        // fd pressure) must not kill the server; back off briefly and keep
        // accepting.
        let (mut socket, peer_addr) = match accept {
            Ok(pair) => pair,
            Err(e) => {
                if !state.quiet {
                    eprintln!("  \u{26a0} accept error (continuing): {e}");
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        };
        let peer_ip = peer_addr.ip().to_string();
        let state = state.clone();
        let permit = match semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => continue, // semaphore closed during shutdown
        };
        let expected_token = expected_token.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let _permit = permit;
            let read_result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                http_utils::read_full_request(&mut socket),
            )
            .await;

            let request_bytes = match read_result {
                Ok(Ok(b)) => b,
                Ok(Err(err)) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
                Err(_) => {
                    let _ = socket
                        .write_all(
                            http_utils::http_error_response(408, "Request read timed out")
                                .as_bytes(),
                        )
                        .await;
                    return;
                }
            };

            let request = String::from_utf8_lossy(&request_bytes).into_owned();

            // Handle OPTIONS preflight for CORS
            if cors && request.starts_with("OPTIONS ") {
                let head =
                    http_utils::http_response_head_with_extra(204, "text/plain", 0, cors_extra);
                let _ = socket.write_all(head.as_bytes()).await;
                return;
            }

            let path = http_utils::extract_request_path(&request);
            let path_no_query = path.split('?').next().unwrap_or(&path);

            // Auth check (skip /health): accept any client-native auth form.
            if let Some(token) = &expected_token
                && path_no_query != "/health"
                && !http_utils::request_loopback_authorized(&request, token)
            {
                let _ = socket
                    .write_all(
                        http_utils::http_error_response(
                            401,
                            "Invalid or missing auth token (expected Authorization: Bearer, x-api-key, x-goog-api-key, or ?key=)",
                        )
                        .as_bytes(),
                    )
                    .await;
                return;
            }

            let request_start = std::time::Instant::now();

            // Extract model from request body for logging (best-effort)
            let log_model = http_utils::extract_request_body(&request)
                .ok()
                .and_then(|body| serde_json::from_str::<Value>(body).ok())
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from));

            let model_state = select_model_upstream(&state, log_model.as_deref())
                .map(|entry| failover_state(entry, &state.client, &state.log_store, state.quiet));
            let active = model_state.as_ref().unwrap_or(&state);

            let result = match path_no_query {
                "/health" => Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
                    HEALTH_RESPONSE.clone(),
                )),
                "/v1/models" | "/models" => {
                    if active.config.is_claude_native_oauth {
                        // Subscriptions get the native catalog, not the local one.
                        send_claude_oauth_passthrough(&request, &upstream_context(active)).await
                    } else {
                        handle_models(active).await
                    }
                }
                "/v1/chat/completions" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_chat_with_failover(&request, active).await
                    }
                }
                "/v1/responses" | "/responses" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_responses_with_failover(&request, active).await
                    }
                }
                "/v1/messages" | "/messages" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_messages_with_failover(&request, active).await
                    }
                }
                p if gemini_generate_target(p).is_some() => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_gemini_with_failover(&request, active).await
                    }
                }
                "/v1/embeddings" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_embeddings_with_failover(&request, active).await
                    }
                }
                // Other upstreams keep the historical 404 for count_tokens.
                "/v1/messages/count_tokens" | "/messages/count_tokens"
                    if active.config.is_claude_native_oauth =>
                {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        send_claude_oauth_passthrough(&request, &upstream_context(active)).await
                    }
                }
                _ => Ok(RouterResponse::buffered(
                    404,
                    CONTENT_TYPE_JSON,
                    br#"{"error":{"message":"Not found"}}"#.to_vec(),
                )),
            };

            let response_status = match &result {
                Ok(RouterResponse::Buffered { status, .. }) => *status,
                Ok(RouterResponse::Streaming { status, .. }) => *status,
                Err(_) => 500,
            };

            let accounting = state.usage_sink.is_some();
            // Peek token usage off a buffered 2xx body before it's moved to the
            // socket; streaming bodies are sniffed as they're forwarded below.
            let buffered_usage = if accounting {
                match &result {
                    Ok(RouterResponse::Buffered { status, body, .. })
                        if (200..300).contains(status) =>
                    {
                        parse_token_usage(body)
                    }
                    _ => None,
                }
            } else {
                None
            };

            let stream_usage = match result {
                Ok(response) => {
                    write_router_response(&mut socket, response, cors_extra, accounting)
                        .await
                        .unwrap_or(None)
                }
                Err(e) => {
                    let _ = socket
                        .write_all(http_utils::http_error_response(500, &e.to_string()).as_bytes())
                        .await;
                    None
                }
            };
            let usage = buffered_usage.or(stream_usage);

            if let (Some(store), Some(u)) = (&state.usage_sink, &usage) {
                let _ = store
                    .record_tokens(
                        // Attribute to the upstream that served (tier or base).
                        &active.key.id,
                        state.usage_tool.as_deref(),
                        log_model.as_deref(),
                        u.prompt_tokens,
                        u.completion_tokens,
                        u.cache_read_input_tokens,
                        u.cache_creation_input_tokens,
                    )
                    .await;
                // Same totals into the per-run tally, so the finished log row is
                // windowable by `aivo stats --since` (lifetime stats aren't).
                if let Some(tally) = &state.run_tally {
                    tally.add(
                        u.prompt_tokens,
                        u.completion_tokens,
                        u.cache_read_input_tokens,
                        u.cache_creation_input_tokens,
                    );
                }
            }

            // Log request (non-blocking, non-fatal)
            let latency_ms = request_start.elapsed().as_millis();
            let method = request
                .split_whitespace()
                .next()
                .unwrap_or("GET")
                .to_string();

            if let Some(ref logger) = state.logger {
                logger
                    .log(crate::services::request_log::RequestLogEntry {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        method: method.clone(),
                        path: path_no_query.to_string(),
                        model: log_model.clone(),
                        status: response_status,
                        latency_ms: latency_ms as u64,
                        ip: peer_ip.clone(),
                    })
                    .await;
            }

            let _ = state
                .log_store
                .append(LogEvent {
                    source: "serve".to_string(),
                    kind: "serve_request".to_string(),
                    key_id: Some(active.key.id.clone()),
                    key_name: Some(active.key.display_name().to_string()),
                    base_url: Some(active.key.base_url.clone()),
                    tool: Some(
                        state
                            .usage_tool
                            .clone()
                            .unwrap_or_else(|| "serve".to_string()),
                    ),
                    model: log_model,
                    status_code: Some(response_status as i64),
                    duration_ms: Some(latency_ms as i64),
                    input_tokens: usage.as_ref().map(|u| u.prompt_tokens as i64),
                    output_tokens: usage.as_ref().map(|u| u.completion_tokens as i64),
                    cache_read_input_tokens: usage
                        .as_ref()
                        .map(|u| u.cache_read_input_tokens as i64),
                    cache_creation_input_tokens: usage
                        .as_ref()
                        .map(|u| u.cache_creation_input_tokens as i64),
                    title: Some(format!("{method} {path_no_query}")),
                    payload_json: Some(json!({
                        "method": method,
                        "path": path_no_query,
                        "ip": peer_ip,
                    })),
                    ..Default::default()
                })
                .await;
        });
    }
}

async fn handle_models(state: &ServeState) -> Result<RouterResponse> {
    let models = fetch_models(&state.client, &state.key).await?;
    // Local cache instance: lazy one-time disk read, and this endpoint
    // already pays a network fetch per call.
    let cache = crate::services::models_cache::ModelsCache::new();
    let cache_base = crate::services::model_catalog::model_cache_key_for_key(&state.key);
    let mut entries = Vec::with_capacity(models.len() + state.config.aliases.len());
    for id in models {
        let limits =
            crate::services::model_metadata::resolve_limits(&cache, Some(&cache_base), &id).await;
        entries.push(model_list_response::ModelListEntry {
            id,
            owned_by: "aivo".to_string(),
            limits,
        });
    }
    let mut alias_names: Vec<&String> = state.config.aliases.keys().collect();
    alias_names.sort();
    for name in alias_names {
        // Aliases inherit the limits of the model they resolve to.
        let limits = match state.config.aliases.get(name) {
            Some(target) => {
                crate::services::model_metadata::resolve_limits(&cache, Some(&cache_base), target)
                    .await
            }
            None => Default::default(),
        };
        entries.push(model_list_response::ModelListEntry {
            id: name.clone(),
            owned_by: "aivo-alias".to_string(),
            limits,
        });
    }
    let resp = model_list_response::build_models_response_body(entries);
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        resp.to_string().into_bytes(),
    ))
}

fn parse_json_body(body_str: &str) -> std::result::Result<Value, RouterResponse> {
    serde_json::from_str(body_str).map_err(|e| {
        RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            json!({"error":{"message": format!("Invalid JSON: {}", e)}})
                .to_string()
                .into_bytes(),
        )
    })
}

/// Rewrites `body["model"]` against the alias snapshot, in place. No-op when
/// the field is missing, non-string, empty, or not an alias. Cycles are
/// detected by `resolve_alias_in_memory` and leave the original value.
fn apply_alias(body: &mut Value, aliases: &HashMap<String, String>) {
    if aliases.is_empty() {
        return;
    }
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return;
    };
    if let Some(resolved) =
        crate::cli_args::resolve_alias_in_memory(aliases, Some(model.to_string()))
        && resolved != model
    {
        body["model"] = Value::String(resolved);
    }
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    if !body.get("messages").is_some_and(|v| v.is_array()) {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: messages"}}"#.to_vec(),
        ));
    }

    apply_alias(&mut body, &state.config.aliases);
    handle_chat_body(body, state).await
}

async fn handle_responses(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    if body.get("input").is_none() {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: input"}}"#.to_vec(),
        ));
    }
    apply_alias(&mut body, &state.config.aliases);
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
    let mut config = responses_router_config(state, resolve_slot(&body, state).current().0);
    config.actual_model = Some(original_model.clone());
    let custom_tools = collect_custom_tool_names(&body);
    let tool_namespaces = collect_namespace_tool_names(&body);
    let mut chat_body = translate_request(
        &body,
        &RequestOptions::ResponsesToChat(&config.conversion_config()),
    );
    chat_body["stream"] = json!(client_wants_stream);
    let chat_response = handle_chat_body(chat_body, state).await?;
    convert_chat_response_for_responses_route(
        chat_response,
        client_wants_stream,
        &original_model,
        custom_tools,
        tool_namespaces,
    )
}

/// Anthropic `/v1/messages` inbound: pivot through Chat, or take the direct
/// Gemini edge when the upstream is confirmed Gemini (`handle_messages_gemini_direct`).
async fn handle_messages(request: &str, state: &ServeState) -> Result<RouterResponse> {
    if state.config.is_claude_native_oauth {
        return send_claude_oauth_passthrough(request, &upstream_context(state)).await;
    }
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    if !body.get("messages").is_some_and(|v| v.is_array()) {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: messages"}}"#.to_vec(),
        ));
    }
    apply_alias(&mut body, &state.config.aliases);
    let original_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("claude-sonnet-4-5")
        .to_string();
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Confirmed Gemini upstream: take the direct edge (one conversion, not two)
    // to preserve the thinking budget + thought-signature round-trip.
    if WireFormat::from(resolve_slot(&body, state).current().0) == WireFormat::Gemini {
        return handle_messages_gemini_direct(&body, &original_model, client_wants_stream, state)
            .await;
    }

    let config = AnthropicToOpenAIConfig {
        default_model: &original_model,
        preserve_stream: true,
        model_transform: None,
        include_reasoning_content: true,
        require_non_empty_reasoning_content: false,
        stringify_other_tool_result_content: true,
        tool_result_supports_multimodal: true,
        fallback_tool_arguments_json: "{}",
    };
    let mut chat_body = translate_request(&body, &RequestOptions::AnthropicToChat(&config));
    chat_body["stream"] = json!(client_wants_stream);
    let chat_response = handle_chat_body(chat_body, state).await?;
    convert_chat_response_for_messages_route(chat_response, client_wants_stream, &original_model)
}

fn convert_chat_response_for_messages_route(
    chat_response: RouterResponse,
    client_wants_stream: bool,
    original_model: &str,
) -> Result<RouterResponse> {
    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }

            if client_wants_stream {
                let chat_sse = if content_type.contains("text/event-stream") {
                    String::from_utf8(body)?
                } else {
                    let chat_json: Value = serde_json::from_slice(&body)?;
                    convert_openai_chat_response_to_sse(&chat_json)?
                };
                let mut adapter = stream_adapter(StreamOptions::AnthropicToChat {
                    fallback_model: original_model,
                });
                let mut sse = adapter.push_bytes(chat_sse.as_bytes())?;
                sse.push_str(&adapter.finish()?);
                Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let chat_json: Value = serde_json::from_slice(&body)?;
                let anthropic = translate_response(
                    &chat_json,
                    &ResponseOptions::AnthropicToChat(&OpenAIToAnthropicConfig {
                        fallback_id: "msg_default",
                        model: chat_json
                            .get("model")
                            .and_then(|m| m.as_str())
                            .unwrap_or(original_model),
                        include_created: true,
                        usage_value_mode: UsageValueMode::CoerceU64,
                    }),
                )?;
                Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
                    serde_json::to_vec(&anthropic)?,
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
                    "internal error: messages route received streaming body for non-streaming request"
                );
            }

            let converter = stream_adapter(StreamOptions::AnthropicToChat {
                fallback_model: original_model,
            });
            let body = match *body {
                StreamingBody::Upstream(upstream) => StreamingBody::Converted {
                    upstream,
                    adapter: converter,
                },
                StreamingBody::Converted { upstream, adapter } => StreamingBody::Converted {
                    upstream,
                    adapter: Box::new(Chain::new(adapter, converter)),
                },
            };
            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(body),
            })
        }
    }
}

/// Direct `Anthropic → Gemini` edge for `/v1/messages`: one conversion each way
/// instead of two Chat hops, so the thinking budget and thought signature survive.
async fn handle_messages_gemini_direct(
    body: &Value,
    original_model: &str,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let gemini_req = translate_request(
        body,
        &RequestOptions::AnthropicToGemini {
            default_model: original_model,
        },
    );
    let response = send_gemini_native(
        &gemini_req,
        original_model,
        client_wants_stream,
        &upstream_context(state),
    )
    .await?;

    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }
            let gemini_json: Value = serde_json::from_slice(&body)?;
            if client_wants_stream {
                // Client wants a stream but upstream returned non-SSE; emulate via the
                // stream adapter as one Gemini event → Anthropic SSE.
                let mut adapter = stream_adapter(StreamOptions::GeminiToAnthropic {
                    model: original_model,
                });
                let event = format!("data: {}\n\n", serde_json::to_string(&gemini_json)?);
                let mut sse = adapter.push_bytes(event.as_bytes())?;
                sse.push_str(&adapter.finish()?);
                return Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ));
            }
            let anthropic = translate_response(
                &gemini_json,
                &ResponseOptions::GeminiToAnthropic {
                    model: original_model,
                },
            )?;
            Ok(RouterResponse::buffered(
                200,
                CONTENT_TYPE_JSON,
                serde_json::to_vec(&anthropic)?,
            ))
        }
        RouterResponse::Streaming { status, body, .. } => {
            let adapter = stream_adapter(StreamOptions::GeminiToAnthropic {
                model: original_model,
            });
            let body = match *body {
                StreamingBody::Upstream(upstream) => StreamingBody::Converted { upstream, adapter },
                // send_gemini_native only ever returns a raw Upstream stream.
                other => other,
            };
            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(body),
            })
        }
    }
}

/// Model + stream flag from a Gemini `generateContent`/`streamGenerateContent` path.
fn gemini_generate_target(path: &str) -> Option<(&str, bool)> {
    let path = path.split('?').next().unwrap_or(path);
    let rest = path
        .strip_prefix("/v1beta/models/")
        .or_else(|| path.strip_prefix("/v1/models/"))?;
    if let Some(model) = rest.strip_suffix(":streamGenerateContent") {
        return Some((model, true));
    }
    rest.strip_suffix(":generateContent").map(|m| (m, false))
}

/// Gemini `generateContent` inbound. Streaming is emulated — no incremental
/// adapter for this edge, so the reply ships buffered as one Gemini SSE event.
async fn handle_gemini(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let path = http_utils::extract_request_path(request);
    let Some((model, client_wants_stream)) = gemini_generate_target(&path) else {
        return Ok(RouterResponse::buffered(
            404,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Not found"}}"#.to_vec(),
        ));
    };
    let body_str = http_utils::extract_request_body(request)?;
    let body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };
    if body.get("contents").is_none() {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: contents"}}"#.to_vec(),
        ));
    }

    // Resolve aliases before the route decision so it uses the real model, not the alias.
    let resolved_model =
        crate::cli_args::resolve_alias_in_memory(&state.config.aliases, Some(model.to_string()))
            .unwrap_or_else(|| model.to_string());
    let model = resolved_model.as_str();

    // Confirmed Anthropic upstream → direct reverse edge. Model lives in the
    // path, so resolve the route by it, not the body.
    let upstream = WireFormat::from(state.route_cache.resolve(model).current().0);
    if upstream == WireFormat::Anthropic {
        return handle_gemini_anthropic_direct(&body, model, client_wants_stream, state).await;
    }

    let mut chat_body = translate_request(
        &body,
        &RequestOptions::GeminiToChat {
            model,
            requires_reasoning_content: state.config.requires_reasoning_content,
            max_tokens_cap: None,
        },
    );
    chat_body["stream"] = json!(false);
    let chat_response = handle_chat_body(chat_body, state).await?;

    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }
            let chat_json: Value = serde_json::from_slice(&body)?;
            if client_wants_stream {
                let sse = convert_openai_chat_to_gemini_sse(&chat_json);
                Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let gemini = translate_response(&chat_json, &ResponseOptions::GeminiToChat)?;
                Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
                    serde_json::to_vec(&gemini)?,
                ))
            }
        }
        RouterResponse::Streaming { .. } => {
            anyhow::bail!("internal error: gemini route received streaming body")
        }
    }
}

/// Direct reverse edge for `generateContent` on a confirmed Anthropic upstream:
/// Gemini → Anthropic → Gemini. Streaming is emulated as one SSE event.
async fn handle_gemini_anthropic_direct(
    body: &Value,
    model: &str,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    let anthropic_req = translate_request(body, &RequestOptions::GeminiToAnthropic { model });
    let response = send_anthropic_native(&anthropic_req, &upstream_context(state)).await?;

    let RouterResponse::Buffered {
        status,
        content_type,
        body,
    } = response
    else {
        anyhow::bail!("internal error: send_anthropic_native returned a streaming body");
    };
    if status >= 400 {
        return Ok(RouterResponse::buffered(status, &content_type, body));
    }

    let anthropic_json: Value = serde_json::from_slice(&body)?;
    let gemini = translate_response(&anthropic_json, &ResponseOptions::AnthropicToGemini)?;
    if client_wants_stream {
        let sse = format!("data: {}\n\n", serde_json::to_string(&gemini)?);
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            sse.into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            CONTENT_TYPE_JSON,
            serde_json::to_vec(&gemini)?,
        ))
    }
}

/// Returns true if the status code should trigger failover.
/// - 401/403: auth failure (key revoked, expired, or lacks model access)
/// - 429: rate limited
/// - 5xx: server errors
fn is_failover_status(status: u16) -> bool {
    matches!(status, 401 | 403 | 429) || (500..600).contains(&status)
}

/// Builds a temporary ServeState from a FailoverEntry, sharing the client.
/// Logger is intentionally omitted — failover attempts are not individually logged.
fn failover_state(
    entry: &FailoverEntry,
    client: &reqwest::Client,
    log_store: &LogStore,
    quiet: bool,
) -> ServeState {
    ServeState {
        config: entry.config.clone(), // Arc clone — O(1) atomic increment
        client: client.clone(),
        key: entry.key.clone(),
        copilot_tokens: entry.copilot_tokens.clone(),
        grok_tokens: entry.grok_tokens.clone(),
        kimi_tokens: entry.kimi_tokens.clone(),
        codex_tokens: entry.codex_tokens.clone(),
        route_cache: entry.route_cache.clone(),
        log_store: log_store.clone(),
        logger: None,
        failover_keys: Arc::new(Vec::new()),
        model_upstreams: Arc::new(HashMap::new()),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        usage_sink: None,
        usage_tool: None,
        run_tally: None,
        quiet,
    }
}

/// Builds a full upstream context (config/key/tokens/route-cache) from a key,
/// for a per-model tier upstream. `from_key` handles any provider.
fn build_upstream_entry(
    key: ApiKey,
    timeout: u64,
    oauth_persist: Option<&SessionStore>,
) -> FailoverEntry {
    let config = ServeRouterConfig::from_key(&key, false, timeout, None, HashMap::new());
    let copilot_tokens = if config.is_copilot {
        Some(Arc::new(CopilotTokenManager::new(
            config.upstream_api_key.clone(),
        )))
    } else {
        None
    };
    let grok_tokens = build_grok_tokens(&config, oauth_persist.cloned());
    let kimi_tokens = build_kimi_tokens(&config, oauth_persist.cloned());
    let codex_tokens = build_codex_tokens(&config, oauth_persist.cloned());
    let route_cache = Arc::new(RouteCache::new(
        "serve",
        config.upstream_protocol,
        std::collections::BTreeMap::new(),
    ));
    FailoverEntry {
        config: Arc::new(config),
        key,
        copilot_tokens,
        grok_tokens,
        kimi_tokens,
        codex_tokens,
        route_cache,
    }
}

/// The per-model upstream for a request's (suffix-stripped) model, if any;
/// `None` → base upstream.
fn select_model_upstream<'a>(
    state: &'a ServeState,
    model: Option<&str>,
) -> Option<&'a Arc<FailoverEntry>> {
    if state.model_upstreams.is_empty() {
        return None;
    }
    let model = model?;
    state
        .model_upstreams
        .get(strip_context_suffix(model))
        .or_else(|| state.model_upstreams.get(model))
}

/// Generates a failover wrapper around a handler function.
/// Tries the primary handler, then falls through to failover keys on 429/5xx
/// buffered responses. Streaming responses are never retried.
macro_rules! impl_with_failover {
    ($name:ident, $handler:ident) => {
        async fn $name(request: &str, state: &ServeState) -> Result<RouterResponse> {
            let response = $handler(request, state).await?;
            if state.failover_keys.is_empty() {
                return Ok(response);
            }

            let status = match &response {
                RouterResponse::Buffered { status, .. } => *status,
                RouterResponse::Streaming { .. } => return Ok(response),
            };

            if !is_failover_status(status) {
                return Ok(response);
            }

            if !state.quiet {
                eprintln!(
                    "  \u{21bb} Primary key returned {}; trying failover keys...",
                    status
                );
            }
            for entry in state.failover_keys.iter() {
                let fstate = failover_state(entry, &state.client, &state.log_store, state.quiet);
                if let Ok(resp) = $handler(request, &fstate).await {
                    let s = match &resp {
                        RouterResponse::Buffered { status, .. } => *status,
                        RouterResponse::Streaming { .. } => 200,
                    };
                    if !is_failover_status(s) {
                        if !state.quiet {
                            eprintln!(
                                "  \u{2713} Failover to {} succeeded",
                                entry.key.display_name()
                            );
                        }
                        return Ok(resp);
                    }
                }
            }
            Ok(response)
        }
    };
}

impl_with_failover!(handle_chat_with_failover, handle_chat);
impl_with_failover!(handle_responses_with_failover, handle_responses);
impl_with_failover!(handle_messages_with_failover, handle_messages);
impl_with_failover!(handle_gemini_with_failover, handle_gemini);
impl_with_failover!(handle_embeddings_with_failover, handle_embeddings);

async fn handle_embeddings(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    apply_alias(&mut body, &state.config.aliases);
    let protocol = resolve_slot(&body, state).current().0;
    if !matches!(
        protocol,
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
    ) {
        return Ok(RouterResponse::buffered(
            501,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Embeddings not supported with this provider"}}"#.to_vec(),
        ));
    }

    send_openai_embeddings(&body, &upstream_context(state)).await
}

/// Resolve the per-model route slot for a request body's `model` field.
fn resolve_slot(body: &Value, state: &ServeState) -> Arc<RouteSlot> {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
    state.route_cache.resolve(model)
}

async fn handle_chat_body(body: Value, state: &ServeState) -> Result<RouterResponse> {
    // Reject clearly rather than surface a cryptic upstream 401.
    if state.config.is_claude_native_oauth {
        return Ok(RouterResponse::buffered(
            501,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Claude subscription upstream only serves the Anthropic wire (/v1/messages)"}}"#.to_vec(),
        ));
    }
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let slot = resolve_slot(&body, state);

    // Copilot has a fixed OpenAI protocol, but its reasoning models (gpt-5.x)
    // reject tools + reasoning_effort on /chat/completions and require /responses.
    // Pin that switch once it's observed so later turns skip the wasted 400.
    if state.config.is_copilot {
        let mut body = body;
        let (protocol, variant) = slot.current();
        let ctx = upstream_context(state);
        if protocol == ProviderProtocol::ResponsesApi {
            return send_copilot_responses(&body, client_wants_stream, &ctx).await;
        }
        let result = handle_chat_openai(&mut body, client_wants_stream, state).await?;
        let redirected = matches!(&result, RouterResponse::Buffered { status: 400, body: e, .. }
            if copilot_requires_responses_api(e));
        if redirected {
            commit_protocol_switch(
                slot.route_atom(),
                ProviderProtocol::ResponsesApi,
                variant,
                1,
            );
            return send_copilot_responses(&body, client_wants_stream, &ctx).await;
        }
        return Ok(result);
    }

    // Skip fallback for openrouter — fixed protocol.
    if state.config.is_openrouter {
        let mut body = body;
        return match slot.current().0 {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body, client_wants_stream, state).await
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body, client_wants_stream, state).await
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body, client_wants_stream, state).await
            }
        };
    }

    // Grok and Kimi speak only OpenAI chat completions; pin them so the
    // protocol-probe loop never tries the Anthropic/Gemini handlers.
    if state.config.is_grok || state.config.is_kimi {
        let mut body = body;
        return handle_chat_openai(&mut body, client_wants_stream, state).await;
    }

    // Codex speaks Responses API only; send there directly, no protocol probing.
    if state.config.is_codex {
        return send_codex_responses(&body, client_wants_stream, &upstream_context(state)).await;
    }

    // Protocol-only candidates: serve's upstream senders own their URLs, so
    // path variants don't apply, and ResponsesApi shares the OpenAI chat
    // handler — keep only the first candidate per handler family.
    let mut seen_openai_family = false;
    let candidates: Vec<ProviderProtocol> = protocol_candidates(slot.route_atom())
        .into_iter()
        .filter(|(_, variant)| *variant == PathVariant::Default)
        .map(|(protocol, _)| protocol)
        .filter(|protocol| {
            let openai_family = matches!(
                protocol,
                ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
            );
            !openai_family || !std::mem::replace(&mut seen_openai_family, true)
        })
        .collect();

    let mut first_error: FirstError<RouterResponse> = FirstError::new();
    let mut success: Option<RouterResponse> = None;
    for (attempt, protocol) in candidates.into_iter().enumerate() {
        let mut body_clone = body.clone();
        let response = match protocol {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body_clone, client_wants_stream, state).await?
            }
        };

        let status = match &response {
            RouterResponse::Buffered { status, .. } => *status,
            // Streaming is only produced when the upstream returned 200 (see each handle_chat_* handler);
            // a protocol mismatch (404/405/415) always results in a Buffered error response.
            RouterResponse::Streaming { .. } => 200,
        };

        if !is_protocol_mismatch(status) {
            commit_protocol_switch(slot.route_atom(), protocol, PathVariant::Default, attempt);
            slot.confirm();
            if attempt > 0 && !state.quiet {
                eprintln!("  \u{2022} Protocol auto-switched to {}", protocol.as_str());
            }
            success = Some(response);
            break;
        }

        let body_text = match &response {
            RouterResponse::Buffered { body, .. } => String::from_utf8_lossy(body).into_owned(),
            RouterResponse::Streaming { .. } => String::new(),
        };
        let classification = classify_failed_attempt(status, &body_text);
        first_error.record_with(&classification, || response);
        match mismatch_directive(
            attempt,
            &classification,
            &slot,
            protocol,
            PathVariant::Default,
            None,
        ) {
            MismatchDirective::Bail => break,
            MismatchDirective::RetrySameCandidate | MismatchDirective::NextCandidate => {}
        }
    }

    let (seed_protocol, seed_variant) = slot.seed_route();
    record_request_outcome(
        slot.route_atom(),
        slot.failures_atom(),
        seed_protocol,
        seed_variant,
        success.is_some(),
    );
    if let Some(response) = success {
        return Ok(response);
    }
    Ok(first_error.take().unwrap_or(RouterResponse::buffered(
        503,
        CONTENT_TYPE_JSON,
        br#"{"error":{"message":"No compatible protocol found"}}"#.to_vec(),
    )))
}

fn responses_router_config(
    state: &ServeState,
    target_protocol: ProviderProtocol,
) -> ResponsesToChatRouterConfig {
    ResponsesToChatRouterConfig {
        target_base_url: state.config.upstream_base_url.clone(),
        api_key: state.config.upstream_api_key.clone(),
        target_protocol,
        target_path_variant: None,
        copilot_token_manager: state.copilot_tokens.clone(),
        model_prefix: None,
        requires_reasoning_content: state.config.requires_reasoning_content,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: state.config.is_starter,
        aivo_prefix_models: Vec::new(),
    }
}

fn upstream_context(state: &ServeState) -> UpstreamRequestContext {
    UpstreamRequestContext {
        client: state.client.clone(),
        upstream_base_url: state.config.upstream_base_url.clone(),
        upstream_api_key: state.config.upstream_api_key.clone(),
        is_copilot: state.config.is_copilot,
        is_openrouter: state.config.is_openrouter,
        is_starter: state.config.is_starter,
        copilot_tokens: state.copilot_tokens.clone(),
        grok_tokens: state.grok_tokens.clone(),
        kimi_tokens: state.kimi_tokens.clone(),
        codex_tokens: state.codex_tokens.clone(),
        accounting: state.usage_sink.is_some(),
    }
}

/// Finds a stored `xai` API key (on `api.x.ai`) for the grok 403 fallback.
pub(crate) async fn resolve_grok_fallback(session_store: &SessionStore) -> Option<String> {
    session_store
        .get_keys()
        .await
        .ok()?
        .into_iter()
        .find(|k| !k.is_grok_oauth() && k.base_url.contains("api.x.ai"))
        .map(|k| k.key.as_str().to_string())
}

/// Starts a universal loopback ServeRouter for a provider-OAuth credential
/// (Codex/Kimi/Grok) and returns its port; `None` if `creds_json` is none of
/// those, so the caller falls back to its static-key path. Probes codex and
/// kimi before grok — both also parse as the looser grok shape. Refreshed
/// tokens persist back to the store. `pub` for sibling-binary plugins.
pub async fn start_provider_oauth_loopback_router(
    creds_json: &str,
    auth_token: Option<String>,
    session_store: &SessionStore,
) -> Result<Option<u16>> {
    use crate::services::codex_oauth::{CODEX_OAUTH_SENTINEL, CodexOAuthCredential};
    use crate::services::grok_oauth::{GROK_OAUTH_SENTINEL, GrokOAuthCredential};
    use crate::services::kimi_oauth::{KIMI_OAUTH_SENTINEL, KimiOAuthCredential};

    let (provider, sentinel) = if CodexOAuthCredential::from_json(creds_json).is_ok() {
        ("codex", CODEX_OAUTH_SENTINEL)
    } else if KimiOAuthCredential::from_json(creds_json).is_ok() {
        ("kimi", KIMI_OAUTH_SENTINEL)
    } else if GrokOAuthCredential::from_json(creds_json).is_ok() {
        ("grok", GROK_OAUTH_SENTINEL)
    } else {
        return Ok(None);
    };

    let key = ApiKey::new_with_protocol(
        provider.to_string(),
        provider.to_string(),
        sentinel.to_string(),
        None,
        creds_json.to_string(),
    );
    let fallback = if key.is_grok_oauth() {
        resolve_grok_fallback(session_store).await
    } else {
        None
    };
    let config = ServeRouterConfig::from_key(&key, false, 300, auth_token, HashMap::new())
        .with_grok_fallback(fallback);
    let (handle, _shutdown, port) = ServeRouter::new(config, key, session_store.logs())
        .with_oauth_persist(session_store.clone())
        .start_background_with_addr("127.0.0.1", 0)
        .await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: {provider} serve router exited unexpectedly: {e}");
        }
    });
    Ok(Some(port))
}

/// Starts a bearer-gated multi-upstream loopback ServeRouter: base = `main_key`
/// (already decrypted), each `(model, key_id)` tier dispatched by
/// suffix-stripped model name to its own key. Tier keys are resolved and
/// decrypted from the store here, so only ids cross the caller boundary and
/// secrets never travel in env. `pub`: sibling-binary plugins (aivo-amp
/// per-mode keys) link it.
pub async fn start_tier_loopback_router(
    main_key: ApiKey,
    tiers: &[(String, String)],
    auth_token: Option<String>,
    session_store: &SessionStore,
) -> Result<u16> {
    // Load once; per-key `get_key_by_id` would re-read config.json each time.
    // `get_keys` returns secrets encrypted, so decrypt only what we reference.
    let all_keys = session_store.get_keys().await?;

    let mut model_upstreams: Vec<(String, ApiKey)> = Vec::with_capacity(tiers.len());
    for (model, key_id) in tiers {
        let mut key = all_keys
            .iter()
            .find(|k| k.id == *key_id)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("per-tier routing: key '{key_id}' for model '{model}' not found")
            })?;
        SessionStore::decrypt_key_secret(&mut key)?;
        // Fail at launch, not with per-request 500s later.
        if key.is_claude_oauth() {
            crate::services::claude_oauth::ClaudeOAuthCredential::from_json(key.key.as_str())
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Claude OAuth credential for key '{}' is unreadable — re-run `aivo keys add claude` ({e})",
                        key.display_name()
                    )
                })?;
        }
        model_upstreams.push((model.clone(), key));
    }

    let fallback = if main_key.is_grok_oauth() {
        resolve_grok_fallback(session_store).await
    } else {
        None
    };
    let config = ServeRouterConfig::from_key(&main_key, false, 300, auth_token, HashMap::new())
        .with_grok_fallback(fallback);
    let (handle, _shutdown, port) = ServeRouter::new(config, main_key, session_store.logs())
        .with_oauth_persist(session_store.clone())
        .with_model_upstreams(model_upstreams)
        .start_background_with_addr("127.0.0.1", 0)
        .await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: multi-upstream serve router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Builds the grok token manager from a grok-oauth config. `None` for non-grok
/// configs or an unparseable credential.
fn build_grok_tokens(
    config: &ServeRouterConfig,
    persist_store: Option<SessionStore>,
) -> Option<Arc<crate::services::grok_oauth::GrokTokenManager>> {
    use crate::services::grok_oauth::{GrokOAuthCredential, GrokTokenManager};
    if !config.is_grok {
        return None;
    }
    match GrokOAuthCredential::from_json(&config.upstream_api_key) {
        Ok(creds) => {
            let mut mgr = GrokTokenManager::new(creds, config.grok_fallback_api_key.clone());
            if let Some(store) = persist_store {
                mgr = mgr.with_persist_store(store);
            }
            Some(Arc::new(mgr))
        }
        Err(_) => None,
    }
}

/// Builds the kimi token manager from a kimi-oauth config. `None` for non-kimi
/// configs or an unparseable credential.
fn build_kimi_tokens(
    config: &ServeRouterConfig,
    persist_store: Option<SessionStore>,
) -> Option<Arc<crate::services::kimi_oauth::KimiTokenManager>> {
    use crate::services::kimi_oauth::{KimiOAuthCredential, KimiTokenManager};
    if !config.is_kimi {
        return None;
    }
    match KimiOAuthCredential::from_json(&config.upstream_api_key) {
        Ok(creds) => {
            let mut mgr = KimiTokenManager::new(creds);
            if let Some(store) = persist_store {
                mgr = mgr.with_persist_store(store);
            }
            Some(Arc::new(mgr))
        }
        Err(_) => None,
    }
}

fn build_codex_tokens(
    config: &ServeRouterConfig,
    persist_store: Option<SessionStore>,
) -> Option<Arc<crate::services::codex_oauth::CodexTokenManager>> {
    use crate::services::codex_oauth::{CodexOAuthCredential, CodexTokenManager};
    if !config.is_codex {
        return None;
    }
    match CodexOAuthCredential::from_json(&config.upstream_api_key) {
        Ok(creds) => {
            let mut manager = CodexTokenManager::new(creds);
            if let Some(store) = persist_store {
                manager = manager.with_persist_store(store);
            }
            Some(Arc::new(manager))
        }
        Err(_) => None,
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_anthropic_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_gemini_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_openai(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    // DeepSeek/Moonshot thinking mode 400s when a replayed assistant message
    // lacks `reasoning_content`; clients strip it, so stub it on passthrough.
    if state.config.requires_reasoning_content {
        ensure_assistant_reasoning_content_in_chat_request(body);
    }
    send_openai_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
    extra_headers: &str,
    sniff_usage: bool,
) -> Result<Option<TokenUsage>> {
    use tokio::io::AsyncWriteExt;

    let mut sniffer = StreamUsageSniffer::new(sniff_usage);
    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_response_head_with_extra(
                status,
                &content_type,
                body.len(),
                extra_headers,
            );
            socket.write_all(headers.as_bytes()).await?;
            socket.write_all(&body).await?;
        }
        RouterResponse::Streaming {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_chunked_response_head_with_extra(
                status,
                &content_type,
                extra_headers,
            );
            socket.write_all(headers.as_bytes()).await?;

            match *body {
                StreamingBody::Upstream(mut upstream) => {
                    while let Some(chunk) = upstream.chunk().await? {
                        sniffer.observe(&chunk);
                        write_chunk(socket, &chunk).await?;
                    }
                }
                StreamingBody::Converted {
                    mut upstream,
                    mut adapter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        sniffer.observe(&chunk);
                        let mapped = adapter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = adapter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
            }

            socket.write_all(b"0\r\n\r\n").await?;
        }
    }

    Ok(sniffer.finish())
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, chunk: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let formatted = http_utils::format_http_chunk(chunk);
    if formatted.is_empty() {
        return Ok(());
    }
    socket.write_all(&formatted).await?;
    Ok(())
}

fn convert_chat_response_for_responses_route(
    chat_response: RouterResponse,
    client_wants_stream: bool,
    original_model: &str,
    custom_tools: HashSet<String>,
    tool_namespaces: ToolNamespaceMap,
) -> Result<RouterResponse> {
    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }

            if client_wants_stream {
                let sse = if content_type.contains("text/event-stream") {
                    convert_chat_sse_to_responses_sse(
                        std::str::from_utf8(&body)?,
                        original_model,
                        &custom_tools,
                        &tool_namespaces,
                    )?
                } else {
                    let chat_json: Value = serde_json::from_slice(&body)?;
                    convert_chat_response_to_responses_sse(
                        &chat_json,
                        false,
                        original_model,
                        &custom_tools,
                        &tool_namespaces,
                    )
                };
                Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let chat_json: Value = serde_json::from_slice(&body)?;
                let response_json = translate_response(
                    &chat_json,
                    &ResponseOptions::ResponsesToChat {
                        model: original_model,
                        custom_tools: &custom_tools,
                        tool_namespaces: &tool_namespaces,
                    },
                )?;
                Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
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

            let converter = stream_adapter(StreamOptions::ResponsesToChat {
                model: original_model,
                requires_reasoning_content: false,
                custom_tools,
                tool_namespaces,
            });
            let body = match *body {
                StreamingBody::Upstream(upstream) => StreamingBody::Converted {
                    upstream,
                    adapter: converter,
                },
                StreamingBody::Converted { upstream, adapter } => StreamingBody::Converted {
                    upstream,
                    adapter: Box::new(Chain::new(adapter, converter)),
                },
            };
            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(body),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ApiKey;
    use http::Response as HttpResponse;
    use serde_json::json;

    #[test]
    fn usage_sniffer_disables_on_oversized_partial_line() {
        let mut sniffer = StreamUsageSniffer::new(true);
        // A newline-less stream larger than the cap must disable sniffing
        // instead of buffering forever.
        let big = vec![b'x'; http_utils::MAX_SSE_PENDING_BYTES + 1];
        sniffer.observe(&big);
        assert!(!sniffer.is_enabled());
        sniffer.observe(b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n");
        assert!(sniffer.finish().is_none());
    }

    fn test_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://example.com/v1".to_string(),
            None,
            "secret".to_string(),
        )
    }

    #[test]
    fn from_key_maps_claude_oauth_sentinel_to_native_passthrough() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        let key = ApiKey::new_with_protocol(
            "id".to_string(),
            "sub".to_string(),
            CLAUDE_OAUTH_SENTINEL.to_string(),
            None,
            r#"{"token":"sk-ant-oat01-TEST","created_at":"2026-01-01T00:00:00Z"}"#.to_string(),
        );
        let config = ServeRouterConfig::from_key(&key, false, 300, None, HashMap::new());
        assert!(config.is_claude_native_oauth);
        assert_eq!(config.upstream_base_url, "https://api.anthropic.com");
        assert_eq!(config.upstream_protocol, ProviderProtocol::Anthropic);
        assert!(config.upstream_api_key.contains("sk-ant-oat01-TEST"));
    }

    fn test_state(protocol: ProviderProtocol) -> ServeState {
        ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://example.com/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: protocol,
                is_copilot: false,
                is_grok: false,
                grok_fallback_api_key: None,
                is_kimi: false,
                is_codex: false,
                is_claude_native_oauth: false,
                is_openrouter: false,
                is_starter: false,
                requires_reasoning_content: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            client: http_utils::router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            grok_tokens: None,
            kimi_tokens: None,
            codex_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                protocol,
                std::collections::BTreeMap::new(),
            )),
            log_store: LogStore::new(std::env::temp_dir()),
            logger: None,
            failover_keys: Arc::new(Vec::new()),
            model_upstreams: Arc::new(HashMap::new()),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
        }
    }

    #[test]
    fn select_model_upstream_matches_by_name_and_suffix() {
        let mut state = test_state(ProviderProtocol::Openai);
        assert!(select_model_upstream(&state, Some("glm-4.6")).is_none());

        let tier_key = ApiKey::new_with_protocol(
            "tierid".to_string(),
            "tier".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-tier".to_string(),
        );
        let entry = Arc::new(build_upstream_entry(tier_key, 300, None));
        let mut map = HashMap::new();
        map.insert("glm-4.6".to_string(), entry);
        state.model_upstreams = Arc::new(map);

        let hit = select_model_upstream(&state, Some("glm-4.6")).expect("mapped");
        assert_eq!(hit.config.upstream_base_url, "https://openrouter.ai/api/v1");
        assert_eq!(hit.key.id, "tierid");
        assert!(select_model_upstream(&state, Some("glm-4.6[1m]")).is_some());
        assert!(select_model_upstream(&state, Some("gpt-4o")).is_none());
        assert!(select_model_upstream(&state, None).is_none());
    }

    fn mock_reqwest_response(
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
    fn convert_chat_response_for_responses_route_maps_buffered_json() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from router"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, serde_json::to_vec(&chat).unwrap()),
            false,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let json: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(json["object"], "response");
                assert_eq!(json["output"][0]["content"][0]["text"], "Hello from router");
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered response"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_maps_streaming_sse() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, "text/event-stream", chat_sse.as_bytes().to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("event: response.created"));
                assert!(sse.contains("\"delta\":\"Hel\""));
                assert!(sse.contains("event: response.completed"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_rejects_streaming_non_stream_requests() {
        let response = convert_chat_response_for_responses_route(
            RouterResponse::Streaming {
                status: 200,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Upstream(mock_reqwest_response(
                    200,
                    "text/event-stream",
                    "data: [DONE]\n\n",
                ))),
            },
            false,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        );

        assert!(response.is_err());
    }

    #[test]
    fn responses_router_config_uses_slot_protocol() {
        let state = test_state(ProviderProtocol::Google);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);

        assert_eq!(config.target_protocol, ProviderProtocol::Google);
        assert_eq!(config.target_base_url, "https://example.com/v1");
        assert_eq!(config.api_key, "secret");
    }

    #[test]
    fn upstream_context_copies_router_flags() {
        let state = ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://openrouter.ai/api/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_grok: false,
                grok_fallback_api_key: None,
                is_kimi: false,
                is_codex: false,
                is_claude_native_oauth: false,
                is_openrouter: true,
                is_starter: false,
                requires_reasoning_content: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            client: http_utils::router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            grok_tokens: None,
            kimi_tokens: None,
            codex_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
            log_store: LogStore::new(std::env::temp_dir()),
            logger: None,
            failover_keys: Arc::new(Vec::new()),
            model_upstreams: Arc::new(HashMap::new()),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
        };

        let context = upstream_context(&state);
        assert!(context.is_openrouter);
        assert!(!context.is_copilot);
        assert_eq!(context.upstream_base_url, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn format_http_chunk_adds_hex_prefix_and_trailer() {
        assert_eq!(http_utils::format_http_chunk(b"hello"), b"5\r\nhello\r\n");
        assert!(http_utils::format_http_chunk(b"").is_empty());
    }

    #[test]
    fn convert_chat_response_for_responses_route_passes_error_status_through() {
        let error_body = br#"{"error":{"message":"rate limited"}}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(429, CONTENT_TYPE_JSON, error_body.to_vec()),
            false,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, body, .. } => {
                assert_eq!(status, 429);
                assert_eq!(body, error_body);
            }
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_passes_500_through() {
        let error_body = br#"{"error":"internal"}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(500, CONTENT_TYPE_JSON, error_body.to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, .. } => assert_eq!(status, 500),
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn format_http_chunk_large_payload() {
        let data = vec![b'x'; 256];
        let chunk = http_utils::format_http_chunk(&data);
        // 256 = 0x100
        assert!(chunk.starts_with(b"100\r\n"));
        assert!(chunk.ends_with(b"\r\n"));
    }

    #[test]
    fn format_http_chunk_single_byte() {
        let chunk = http_utils::format_http_chunk(b"a");
        assert_eq!(chunk, b"1\r\na\r\n");
    }

    #[test]
    fn responses_router_config_anthropic_protocol() {
        let state = test_state(ProviderProtocol::Anthropic);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);
        assert_eq!(config.target_protocol, ProviderProtocol::Anthropic);
    }

    #[test]
    fn convert_chat_response_for_responses_route_buffered_json_to_stream() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "streamed text"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        });

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, serde_json::to_vec(&chat).unwrap()),
            true,
            // client wants stream
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                let sse = String::from_utf8(body).unwrap();
                assert!(sse.contains("event: response.created"));
                assert!(sse.contains("streamed text"));
                assert!(sse.contains("event: response.completed"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    // ── Failover tests ────────────────────────────────────────────────────

    #[test]
    fn is_failover_status_triggers_on_auth_errors() {
        assert!(is_failover_status(401));
        assert!(is_failover_status(403));
    }

    #[test]
    fn is_failover_status_triggers_on_rate_limit() {
        assert!(is_failover_status(429));
    }

    #[test]
    fn is_failover_status_triggers_on_server_errors() {
        assert!(is_failover_status(500));
        assert!(is_failover_status(502));
        assert!(is_failover_status(503));
        assert!(is_failover_status(504));
        assert!(is_failover_status(599));
    }

    #[test]
    fn is_failover_status_does_not_trigger_on_success() {
        assert!(!is_failover_status(200));
        assert!(!is_failover_status(201));
        assert!(!is_failover_status(204));
    }

    #[test]
    fn is_failover_status_does_not_trigger_on_client_errors() {
        // Client errors that indicate a bad request — retrying with a different
        // key won't help.
        assert!(!is_failover_status(400));
        assert!(!is_failover_status(404));
        assert!(!is_failover_status(405));
        assert!(!is_failover_status(422));
    }

    #[test]
    fn failover_state_builds_from_entry() {
        let entry = FailoverEntry {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://backup.example.com/v1".to_string(),
                upstream_api_key: "backup-key".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_grok: false,
                grok_fallback_api_key: None,
                is_kimi: false,
                is_codex: false,
                is_claude_native_oauth: false,
                is_openrouter: false,
                is_starter: false,
                requires_reasoning_content: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            key: test_key(),
            copilot_tokens: None,
            grok_tokens: None,
            kimi_tokens: None,
            codex_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        assert_eq!(
            state.config.upstream_base_url,
            "https://backup.example.com/v1"
        );
        assert_eq!(state.config.upstream_api_key, "backup-key");
        assert!(state.logger.is_none());
        assert!(state.failover_keys.is_empty());
    }

    #[test]
    fn failover_state_shares_arc_config() {
        let config = Arc::new(ServeRouterConfig {
            upstream_base_url: "https://backup.example.com/v1".to_string(),
            upstream_api_key: "key".to_string(),
            upstream_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            is_grok: false,
            grok_fallback_api_key: None,
            is_kimi: false,
            is_codex: false,
            is_claude_native_oauth: false,
            is_openrouter: false,
            is_starter: false,
            requires_reasoning_content: false,
            cors: false,
            timeout: 300,
            auth_token: None,
            aliases: HashMap::new(),
        });

        let entry = FailoverEntry {
            config: config.clone(),
            key: test_key(),
            copilot_tokens: None,
            grok_tokens: None,
            kimi_tokens: None,
            codex_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        // Arc should be a clone of the same allocation, not a new copy
        assert!(Arc::ptr_eq(&entry.config, &state.config));
    }

    #[test]
    fn failover_state_does_not_cascade() {
        let entry = FailoverEntry {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://backup.example.com/v1".to_string(),
                upstream_api_key: "key".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_grok: false,
                grok_fallback_api_key: None,
                is_kimi: false,
                is_codex: false,
                is_claude_native_oauth: false,
                is_openrouter: false,
                is_starter: false,
                requires_reasoning_content: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            key: test_key(),
            copilot_tokens: None,
            grok_tokens: None,
            kimi_tokens: None,
            codex_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        // Failover state should have no failover keys (no cascading)
        assert!(state.failover_keys.is_empty());
        // Logger should be disabled on failover attempts
        assert!(state.logger.is_none());
    }

    #[test]
    fn health_response_is_valid_json() {
        let json: Value = serde_json::from_slice(&HEALTH_RESPONSE).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["version"].is_string());
    }

    #[test]
    fn health_response_is_stable() {
        // LazyLock should return the same bytes every time
        let a = HEALTH_RESPONSE.clone();
        let b = HEALTH_RESPONSE.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn is_failover_status_boundary_499() {
        // 499 is a non-standard client error — should NOT trigger failover
        assert!(!is_failover_status(499));
    }

    #[test]
    fn is_failover_status_boundary_600() {
        // 600 is outside the 5xx range — should NOT trigger failover
        assert!(!is_failover_status(600));
    }

    #[test]
    fn convert_chat_response_for_responses_route_malformed_json_body() {
        // A non-JSON body with status 200 should fail to parse and return an error
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, b"not valid json".to_vec()),
            false,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        );
        assert!(response.is_err());
    }

    #[test]
    fn convert_chat_response_for_responses_route_empty_body_non_stream() {
        // An empty body with status 200 should fail to parse and return an error
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, Vec::new()),
            false,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        );
        assert!(response.is_err());
    }

    #[test]
    fn convert_chat_response_for_responses_route_error_stream_passthrough() {
        // A 400 error response passes through unchanged even when client wants stream
        let error_body = br#"{"error":{"message":"bad request"}}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(400, CONTENT_TYPE_JSON, error_body.to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
            ToolNamespaceMap::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, body, .. } => {
                assert_eq!(status, 400);
                assert_eq!(body, error_body);
            }
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn responses_router_config_openai_protocol() {
        let state = test_state(ProviderProtocol::Openai);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);

        assert_eq!(config.target_protocol, ProviderProtocol::Openai);
        assert_eq!(config.target_base_url, "https://example.com/v1");
        assert_eq!(config.api_key, "secret");
        assert!(config.copilot_token_manager.is_none());
        assert!(config.model_prefix.is_none());
        assert!(!config.requires_reasoning_content);
        assert!(config.actual_model.is_none());
        assert!(config.max_tokens_cap.is_none());
        assert!(config.responses_api_supported.is_none());
    }

    #[test]
    fn apply_alias_rewrites_known_alias() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);
        let mut body = json!({"model": "fast", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "gpt-4o-mini");
    }

    #[test]
    fn apply_alias_passes_through_unknown_model() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);
        let mut body = json!({"model": "claude-sonnet-4-6", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn apply_alias_follows_chain() {
        let aliases = HashMap::from([
            ("fast".to_string(), "quick".to_string()),
            ("quick".to_string(), "gpt-4o-mini".to_string()),
        ]);
        let mut body = json!({"model": "fast"});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "gpt-4o-mini");
    }

    #[test]
    fn apply_alias_no_op_when_field_missing_or_empty() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);

        let mut body = json!({"messages": []});
        apply_alias(&mut body, &aliases);
        assert!(body.get("model").is_none());

        let mut body = json!({"model": "", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "");
    }

    #[test]
    fn apply_alias_no_op_when_alias_map_empty() {
        let mut body = json!({"model": "fast"});
        apply_alias(&mut body, &HashMap::new());
        assert_eq!(body["model"], "fast");
    }
}
