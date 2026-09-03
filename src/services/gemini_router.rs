use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_chat_request::ensure_assistant_reasoning_content_in_chat_request;
use crate::services::anthropic_route_pipeline::inject_anthropic_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::model_names::select_model_for_provider_attempt;
use crate::services::openai_gemini_bridge::{
    build_google_generate_content_url, convert_openai_chat_to_gemini_sse, openai_chat_model,
};
use crate::services::opencode_session;
use crate::services::protocol_fallback::{
    AttemptOutcome, FirstError, MismatchDirective, QuirkRetryState, classify_attempt,
    commit_protocol_switch, mismatch_directive, protocol_candidates, record_slot_outcome,
};
use crate::services::provider_protocol::{PathVariant, ProviderProtocol, classify_failed_attempt};
use crate::services::responses_chat_conversion::{
    try_convert_chat_to_responses_request, try_convert_responses_json_to_chat,
};
use crate::services::route_cache::{PersistedRoute, RouteCache, RouteSlot};
use crate::services::token_usage::{UsageAccounting, parse_http_response_usage};
use crate::services::wire_format::{
    RequestOptions, ResponseOptions, translate_request, translate_response,
};

#[derive(Clone)]
pub struct GeminiRouterConfig {
    pub target_base_url: String,
    pub api_key: String,
    pub upstream_protocol: ProviderProtocol,
    /// When set, overrides the model name extracted from the URL path (used for Copilot mode
    /// since Gemini model names like `gemini-2.0-flash` are not available on Copilot).
    pub forced_model: Option<String>,
    /// When Some, use Copilot token auth instead of api_key
    pub copilot_token_manager: Option<Arc<CopilotTokenManager>>,
    /// Whether the provider requires `reasoning_content` on assistant tool-call turns
    pub requires_reasoning_content: bool,
    /// Cap applied to `max_tokens` before forwarding to the provider
    pub max_tokens_cap: Option<u64>,
    /// Whether this is the aivo starter provider (requires device fingerprint headers).
    pub is_starter: bool,
}

pub struct GeminiRouter {
    config: GeminiRouterConfig,
    /// Per-model routes learned for `gemini` (`""` = default), seeding the
    /// `RouteCache`. On the router (not config) so test literals stay untouched.
    seed_routes: BTreeMap<String, PersistedRoute>,
    /// Per-launch loopback token; `Some` rejects requests without it so other
    /// local processes can't spend the key through this router.
    expected_token: Option<String>,
    /// When set, 2xx token usage is recorded against the launching key.
    usage: Option<UsageAccounting>,
}

enum ForwardResult {
    Success(Value),
    /// All protocol candidates failed; carries the last upstream error.
    Exhausted {
        status: u16,
        body: String,
    },
}

struct GeminiRouterState {
    config: Arc<GeminiRouterConfig>,
    expected_token: Option<String>,
    usage: Option<UsageAccounting>,
    client: Arc<reqwest::Client>,
    /// Per-(model) learned routes; the cascade reads/writes the resolved slot's
    /// atom and `slot.confirm()` marks authoritative outcomes for write-behind.
    route_cache: Arc<RouteCache>,
    /// Set by the cascade when an upstream returned an error envelope matching
    /// the `requires_reasoning_content` quirk. Persisted to `ApiKey` so future
    /// launches inject strict mode without hardcoding the host.
    learned_requires_reasoning: Arc<AtomicBool>,
}

impl GeminiRouter {
    pub fn new(config: GeminiRouterConfig) -> Self {
        Self {
            config,
            seed_routes: BTreeMap::new(),
            expected_token: None,
            usage: None,
        }
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

    pub fn with_seed_routes(mut self, seed_routes: BTreeMap<String, PersistedRoute>) -> Self {
        self.seed_routes = seed_routes;
        self
    }

    /// Requires loopback clients to present this token
    /// (Bearer/x-api-key/x-goog-api-key/?key=).
    pub fn with_auth_token(mut self, token: String) -> Self {
        self.expected_token = Some(token);
        self
    }

    fn build_seed(&self) -> BTreeMap<String, PersistedRoute> {
        let mut seed = self.seed_routes.clone();
        seed.entry(String::new()).or_insert_with(|| {
            PersistedRoute::from_route(self.config.upstream_protocol, PathVariant::Default)
        });
        seed
    }

    pub async fn start_background(
        &self,
    ) -> Result<(
        u16,
        Arc<RouteCache>,
        Arc<AtomicBool>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        // Tool-native protocol for `aivo gemini`: prefer Google.
        let route_cache = Arc::new(RouteCache::new(
            "gemini",
            ProviderProtocol::Google,
            self.build_seed(),
        ));
        let learned_requires_reasoning = Arc::new(AtomicBool::new(false));
        let state = GeminiRouterState {
            config: Arc::new(self.config.clone()),
            expected_token: self.expected_token.clone(),
            usage: self.usage.clone(),
            client: Arc::new(http_utils::router_http_model_client()),
            route_cache: route_cache.clone(),
            learned_requires_reasoning: learned_requires_reasoning.clone(),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_text_router(listener, Arc::new(state), handle_router_request).await
        });
        Ok((port, route_cache, learned_requires_reasoning, handle))
    }
}

async fn handle_router_request(request: String, state: Arc<GeminiRouterState>) -> String {
    if let Some(expected) = state.expected_token.as_deref()
        && !http_utils::request_loopback_authorized(&request, expected)
    {
        return http_utils::http_error_response(
            401,
            "Invalid or missing auth token (expected Authorization: Bearer or x-api-key)",
        );
    }
    let response = match handle_request(
        &request,
        &state.config,
        &state.client,
        &state.route_cache,
        &state.learned_requires_reasoning,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => http_utils::http_error_response(500, &e.to_string()),
    };
    if let Some(acct) = &state.usage
        && let Some(u) = parse_http_response_usage(&response)
    {
        // Model rides in the Gemini URL path, not the body.
        let path = http_utils::extract_request_path(&request);
        let model = parse_gemini_path(path.split('?').next().unwrap_or("")).map(|(m, _)| m);
        acct.record(model.as_deref(), &u).await;
    }
    response
}

async fn handle_request(
    request: &str,
    config: &std::sync::Arc<GeminiRouterConfig>,
    client: &Arc<reqwest::Client>,
    route_cache: &Arc<RouteCache>,
    learned_requires_reasoning: &Arc<AtomicBool>,
) -> Result<String> {
    let path = http_utils::extract_request_path(request);

    match parse_gemini_path(&path) {
        Some((extracted_model, is_streaming)) => {
            let model = config.forced_model.clone().unwrap_or(extracted_model);
            let slot = route_cache.resolve(&model);
            let body: Value = serde_json::from_str(http_utils::extract_request_body(request)?)?;
            let tool_schemas = extract_tool_schemas(&body);
            // OR in the runtime-learned flag so requests after a successful
            // in-cascade recovery skip the wasted first attempt.
            let effective_requires_reasoning = config.requires_reasoning_content
                || learned_requires_reasoning.load(Ordering::Relaxed);
            let openai_req = translate_request(
                &body,
                &RequestOptions::GeminiToChat {
                    model: &model,
                    requires_reasoning_content: effective_requires_reasoning,
                    max_tokens_cap: config.max_tokens_cap,
                },
            );
            // openai_req already has the model from the Gemini request body — don't pre-select here;
            // select_model_for_protocol is applied per-attempt inside forward_to_provider.
            match forward_to_provider(
                openai_req,
                config,
                client,
                &slot,
                learned_requires_reasoning,
            )
            .await?
            {
                ForwardResult::Success(openai_response) => {
                    slot.confirm();
                    let openai_response = repair_tool_call_args(openai_response, &tool_schemas);
                    if is_streaming {
                        let sse = convert_openai_chat_to_gemini_sse(&openai_response);
                        Ok(http_utils::http_response(200, "text/event-stream", &sse))
                    } else {
                        let gemini =
                            translate_response(&openai_response, &ResponseOptions::GeminiToChat)?;
                        let json = serde_json::to_string(&gemini)?;
                        Ok(http_utils::http_json_response(200, &json))
                    }
                }
                ForwardResult::Exhausted { status, body } => {
                    let wrapped = wrap_upstream_error_as_json(status, &body);
                    Ok(http_utils::http_response(
                        status,
                        CONTENT_TYPE_JSON,
                        &wrapped,
                    ))
                }
            }
        }
        None if is_interactions_path(&path) => {
            // 501 + JSON error envelope (not a bare 404) so a client that migrated to
            // the unsupported Interactions API gets a parseable, honest reason.
            let body = serde_json::json!({
                "error": {
                    "code": 501,
                    "message": "aivo's gemini loopback bridges the stateless generateContent API only; \
                                Google's Interactions API (previous_interaction_id/background) is not supported yet",
                    "status": "UNIMPLEMENTED",
                }
            })
            .to_string();
            Ok(http_utils::http_response(501, CONTENT_TYPE_JSON, &body))
        }
        None => Ok(http_utils::http_error_response(404, "not found")),
    }
}

/// Detects Google's stateful Interactions API endpoints (REST `/v1beta/interactions...`).
fn is_interactions_path(path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path.split('/').any(|seg| seg == "interactions")
}

/// Wraps plaintext/HTML error bodies in a Google-native error envelope so
/// gemini-cli's JSON.parse doesn't crash. JSON bodies pass through unchanged.
fn wrap_upstream_error_as_json(status: u16, body: &str) -> String {
    if serde_json::from_str::<Value>(body).is_ok() {
        return body.to_string();
    }
    serde_json::json!({
        "error": {
            "code": status as i64,
            "message": body,
            "status": "UNKNOWN",
        }
    })
    .to_string()
}

async fn forward_to_provider(
    openai_req: Value,
    config: &std::sync::Arc<GeminiRouterConfig>,
    client: &Arc<reqwest::Client>,
    slot: &RouteSlot,
    learned_requires_reasoning: &Arc<AtomicBool>,
) -> Result<ForwardResult> {
    let active_protocol = slot.route_atom();
    let candidates = protocol_candidates(active_protocol);
    let mut first_error: FirstError<(u16, String)> = FirstError::new();
    let original_openai_req = openai_req;
    let mut body_for_attempts = original_openai_req.clone();
    // Snapshot once at the top: the caller in `handle_request` already used the
    // same OR'd value when building `original_openai_req`, so this matches the
    // strictness that's actually on the wire and prevents a wasted retry when
    // the upstream rejects an already-strict body.
    let effective_requires_reasoning =
        config.requires_reasoning_content || learned_requires_reasoning.load(Ordering::Relaxed);
    let mut quirk = QuirkRetryState::new(learned_requires_reasoning, effective_requires_reasoning);
    // Catalog (when cached) snaps the model name to the exact advertised id.
    let catalog = crate::services::models_cache::ModelsCache::shared()
        .model_ids(&config.target_base_url)
        .await;
    let mut idx = 0;

    while idx < candidates.len() {
        let (protocol, variant) = candidates[idx];
        let attempt = idx;
        // Select the right model name for this protocol attempt.
        let mut req_body = body_for_attempts.clone();
        let selected_model = select_model_for_provider_attempt(
            catalog.as_deref(),
            &config.target_base_url,
            req_body.get("model").and_then(|v| v.as_str()),
            None,
            protocol,
        );
        req_body["model"] = serde_json::json!(selected_model);

        let initiator = config
            .copilot_token_manager
            .as_ref()
            .map(|_| http_utils::copilot_initiator_from_openai(&req_body));

        let (status, body_text, parsed) = match protocol {
            ProviderProtocol::Anthropic => {
                let mut anthropic_req = translate_request(
                    &req_body,
                    &RequestOptions::ChatToAnthropic {
                        default_model: "claude-sonnet-4-5",
                    },
                );
                // Only inject cache_control for Claude models — other providers
                // don't honor it (e.g. Gemini has a different caching model) and
                // strict ones reject the unknown field outright.
                if req_body
                    .get("model")
                    .and_then(|m| m.as_str())
                    .is_some_and(|m| m.to_ascii_lowercase().contains("claude"))
                {
                    inject_anthropic_cache_control(&mut anthropic_req);
                }
                anthropic_req["stream"] = serde_json::json!(false);
                let target_url = http_utils::build_target_url(
                    &config.target_base_url,
                    variant.apply("/v1/messages"),
                );
                // Anthropic /v1/messages uses `x-api-key` only; an extra
                // `Authorization: Bearer` makes some gateways (opencode-go's
                // qwen) reject as InvalidApiKey. Mirrors the claude router.
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&target_url)
                        .header("x-api-key", config.api_key.as_str())
                        .header("anthropic-version", "2023-06-01")
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&anthropic_req),
                    config.is_starter,
                )
                .send_logged()
                .await?;
                let status = response.status().as_u16();
                let body_text = response.text().await?;
                let parsed = if status == 200 {
                    let anthropic_response: Value = serde_json::from_str(&body_text)?;
                    Some(translate_response(
                        &anthropic_response,
                        &ResponseOptions::ChatToAnthropic {
                            model: req_body
                                .get("model")
                                .and_then(|v| v.as_str())
                                .unwrap_or("gemini-2.5-pro"),
                        },
                    )?)
                } else {
                    None
                };
                (status, body_text, parsed)
            }
            ProviderProtocol::Google => {
                let google_body = translate_request(
                    &req_body,
                    &RequestOptions::ChatToGemini {
                        default_model: "gemini-2.5-pro",
                    },
                );
                let model = openai_chat_model(&req_body, "gemini-2.5-pro");
                let target_url = build_google_generate_content_url(&config.target_base_url, &model);
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&target_url)
                        .header("x-goog-api-key", config.api_key.as_str())
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&google_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;
                let status = response.status().as_u16();
                let body_text = response.text().await?;
                let parsed = if status == 200 {
                    let google_response: Value = serde_json::from_str(&body_text)?;
                    Some(translate_response(
                        &google_response,
                        &ResponseOptions::ChatToGemini { model: &model },
                    )?)
                } else {
                    None
                };
                (status, body_text, parsed)
            }
            ProviderProtocol::Openai => {
                let target_url = http_utils::build_target_url(
                    &config.target_base_url,
                    variant.apply("/v1/chat/completions"),
                );
                let req = http_utils::authorized_openai_post(
                    client.as_ref(),
                    &target_url,
                    &config.api_key,
                    config.copilot_token_manager.as_deref(),
                    None,
                    None,
                    initiator,
                )
                .await?;
                let req = opencode_session::with_session_header(req, &target_url, Some(&req_body));
                let response = device_fingerprint::maybe_with_starter_headers(
                    req.json(&req_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;
                let status = response.status().as_u16();
                let body_text = response.text().await?;
                let parsed = if status == 200 {
                    Some(serde_json::from_str(&body_text)?)
                } else {
                    None
                };
                (status, body_text, parsed)
            }
            ProviderProtocol::ResponsesApi => {
                let responses_body = try_convert_chat_to_responses_request(&req_body)?;
                let target_url = http_utils::build_target_url(
                    &config.target_base_url,
                    variant.apply("/v1/responses"),
                );
                let req = http_utils::authorized_openai_post(
                    client.as_ref(),
                    &target_url,
                    &config.api_key,
                    config.copilot_token_manager.as_deref(),
                    None,
                    None,
                    initiator,
                )
                .await?;
                let req =
                    opencode_session::with_session_header(req, &target_url, Some(&responses_body));
                let response = device_fingerprint::maybe_with_starter_headers(
                    req.json(&responses_body),
                    config.is_starter,
                )
                .send_logged()
                .await?;
                let status = response.status().as_u16();
                let body_text = response.text().await?;
                let parsed = if status == 200 {
                    Some(try_convert_responses_json_to_chat(&serde_json::from_str(
                        &body_text,
                    )?)?)
                } else {
                    None
                };
                (status, body_text, parsed)
            }
        };

        match classify_attempt(status, body_text, parsed) {
            AttemptOutcome::Success(result) => {
                commit_protocol_switch(active_protocol, protocol, variant, attempt);
                slot.confirm();
                record_slot_outcome(slot, true);
                return Ok(ForwardResult::Success(result));
            }
            AttemptOutcome::Mismatch { status, body } => {
                let classification = classify_failed_attempt(status, &body);
                first_error.record_with(&classification, || (status, body));
                match mismatch_directive(
                    attempt,
                    &classification,
                    slot,
                    protocol,
                    variant,
                    Some(&mut quirk),
                ) {
                    MismatchDirective::RetrySameCandidate => {
                        body_for_attempts = original_openai_req.clone();
                        ensure_assistant_reasoning_content_in_chat_request(&mut body_for_attempts);
                        continue;
                    }
                    MismatchDirective::Bail => break,
                    MismatchDirective::NextCandidate => {}
                }
            }
        }
        idx += 1;
    }

    // Failure-streak valve: resets a stale learned pin after repeated
    // exhausted requests so long-lived sessions recover without restart.
    record_slot_outcome(slot, false);
    let (status, body) = first_error.take().unwrap_or_default();
    Ok(ForwardResult::Exhausted { status, body })
}

/// Parses a Gemini API request path and extracts (model_name, is_streaming).
///
/// Examples:
/// - "/v1beta/models/gemini-2.0-flash:generateContent" → Some(("gemini-2.0-flash", false))
/// - "/v1beta/models/google/gemini-2.0-flash:streamGenerateContent?alt=sse" → Some(("google/gemini-2.0-flash", true))
/// - "/v1/chat/completions" → None
pub fn parse_gemini_path(path: &str) -> Option<(String, bool)> {
    // Strip query string
    let path = path.split('?').next().unwrap_or(path);

    let is_streaming = path.ends_with(":streamGenerateContent");
    let is_generate = path.ends_with(":generateContent");

    if !is_streaming && !is_generate {
        return None;
    }

    // Find "models/" prefix
    let models_prefix = path.find("/models/")?;
    let after_models = &path[models_prefix + "/models/".len()..];

    // Strip the trailing method suffix
    let method_suffix = if is_streaming {
        ":streamGenerateContent"
    } else {
        ":generateContent"
    };
    let model = after_models.strip_suffix(method_suffix)?;

    Some((model.to_string(), is_streaming))
}

/// Extracts tool parameter schemas from a Gemini request body.
/// Returns a map of function name → parameters schema.
fn extract_tool_schemas(body: &Value) -> std::collections::HashMap<String, Value> {
    let mut schemas = std::collections::HashMap::new();
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        for tg in tools {
            if let Some(decls) = tg.get("functionDeclarations").and_then(|fd| fd.as_array()) {
                for decl in decls {
                    if let (Some(name), Some(params)) = (
                        decl.get("name").and_then(|n| n.as_str()),
                        decl.get("parameters"),
                    ) {
                        schemas.insert(name.to_string(), params.clone());
                    }
                }
            }
        }
    }
    schemas
}

/// Repairs tool call arguments in an OpenAI response before converting to Gemini format.
///
/// Fixes two common model mistakes:
/// 1. Wrong parameter name (fuzzy rename: `path` → `file_path`)
/// 2. Missing required parameter with a sensible default (path-like strings → `"."`)
fn repair_tool_call_args(
    mut response: Value,
    schemas: &std::collections::HashMap<String, Value>,
) -> Value {
    if let Some(choices) = response["choices"].as_array_mut() {
        for choice in choices.iter_mut() {
            if let Some(tool_calls) = choice["message"]["tool_calls"].as_array_mut() {
                for tc in tool_calls.iter_mut() {
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    let schema = schemas.get(&name).or_else(|| {
                        schemas
                            .iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case(&name))
                            .map(|(_, v)| v)
                    });
                    if let Some(schema) = schema {
                        repair_single_tool_call(tc, schema);
                    }
                }
            }
        }
    }
    response
}

fn repair_single_tool_call(tc: &mut Value, schema: &Value) {
    let required: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if required.is_empty() {
        return;
    }

    // Parse current args (handle both string-encoded and object forms)
    let mut args: serde_json::Map<String, Value> = match &tc["function"]["arguments"] {
        Value::String(s) => serde_json::from_str(s).unwrap_or_default(),
        Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    let existing_keys: Vec<String> = args.keys().cloned().collect();

    for req in &required {
        // Treat null values the same as missing — remove so default logic applies
        if args.get(req).is_some_and(|v| v.is_null()) {
            args.remove(req);
        }
        if args.contains_key(req) {
            continue;
        }

        // 1. Fuzzy rename: find an existing key whose name overlaps with the required one
        let similar_key = existing_keys.iter().find(|k| {
            let k_lower = k.to_lowercase();
            let r_lower = req.to_lowercase();
            k_lower == r_lower || k_lower.contains(&r_lower) || r_lower.contains(&k_lower)
        });
        if let Some(old_key) = similar_key
            && let Some(val) = args.remove(old_key)
        {
            args.insert(req.clone(), val);
            continue;
        }

        // 2. Default: path-like string params default to current directory
        if is_path_like_param(req) && schema_param_accepts_string(schema, req) {
            args.insert(req.clone(), Value::String(".".to_string()));
        }
    }

    tc["function"]["arguments"] = Value::String(
        serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".to_string()),
    );
}

fn is_path_like_param(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n == "path"
        || n == "dir"
        || n.ends_with("_path")
        || n.ends_with("_dir")
        || n.contains("dir_path")
}

fn schema_param_accepts_string(schema: &Value, name: &str) -> bool {
    let prop = schema.get("properties").and_then(|p| p.get(name));
    let Some(prop) = prop else {
        // If schema doesn't expose the property shape, still repair path-like params.
        return true;
    };
    if prop
        .get("type")
        .and_then(|t| t.as_str())
        .map(|t| t.eq_ignore_ascii_case("string"))
        .unwrap_or(false)
    {
        return true;
    }
    if prop
        .get("type")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter().any(|v| {
                v.as_str()
                    .map(|s| s.eq_ignore_ascii_case("string"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
    {
        return true;
    }
    for key in ["anyOf", "oneOf"] {
        if prop
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().any(|item| {
                    item.get("type")
                        .and_then(|t| t.as_str())
                        .map(|s| s.eq_ignore_ascii_case("string"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_upstream_error_wraps_plaintext_body() {
        let out = wrap_upstream_error_as_json(401, "Authentication Fails (governor)");
        let v: Value = serde_json::from_str(&out).expect("must be valid JSON");
        assert_eq!(v["error"]["code"], 401);
        assert_eq!(v["error"]["message"], "Authentication Fails (governor)");
    }

    #[test]
    fn wrap_upstream_error_passes_through_existing_json() {
        let body = r#"{"error":{"code":401,"message":"invalid_api_key"}}"#;
        assert_eq!(wrap_upstream_error_as_json(401, body), body);
    }

    #[test]
    fn wrap_upstream_error_handles_empty_body() {
        let out = wrap_upstream_error_as_json(502, "");
        let v: Value = serde_json::from_str(&out).expect("must be valid JSON");
        assert_eq!(v["error"]["code"], 502);
    }

    #[test]
    fn test_parse_gemini_path_generate_content() {
        let result = parse_gemini_path("/v1beta/models/gemini-2.0-flash:generateContent");
        assert_eq!(result, Some(("gemini-2.0-flash".to_string(), false)));
    }

    #[test]
    fn test_parse_gemini_path_stream_generate_content() {
        let result = parse_gemini_path(
            "/v1beta/models/google/gemini-2.0-flash:streamGenerateContent?alt=sse",
        );
        assert_eq!(result, Some(("google/gemini-2.0-flash".to_string(), true)));
    }

    #[test]
    fn test_parse_gemini_path_unrecognized() {
        assert_eq!(parse_gemini_path("/v1/chat/completions"), None);
        assert_eq!(parse_gemini_path("/health"), None);
        assert_eq!(parse_gemini_path(""), None);
    }

    #[test]
    fn test_is_interactions_path() {
        assert!(is_interactions_path("/v1beta/interactions"));
        assert!(is_interactions_path("/v1beta/interactions/abc123"));
        assert!(is_interactions_path("/v1beta/interactions?store=false"));
        // generateContent and unrelated paths must not be misclassified
        assert!(!is_interactions_path(
            "/v1beta/models/gemini-2.5-pro:generateContent"
        ));
        assert!(!is_interactions_path("/v1/chat/completions"));
    }

    #[test]
    fn test_parse_gemini_path_simple_model() {
        let result = parse_gemini_path("/v1beta/models/gemini-2.5-pro:generateContent");
        assert_eq!(result, Some(("gemini-2.5-pro".to_string(), false)));
    }

    #[test]
    fn test_gemini_config_forced_model_field() {
        let config = GeminiRouterConfig {
            target_base_url: String::new(),
            api_key: String::new(),
            upstream_protocol: ProviderProtocol::Openai,
            forced_model: Some("gpt-4o".to_string()),
            copilot_token_manager: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        assert_eq!(config.forced_model, Some("gpt-4o".to_string()));
        assert!(config.copilot_token_manager.is_none());
    }

    #[test]
    fn test_gemini_config_no_copilot() {
        let config = GeminiRouterConfig {
            target_base_url: "https://example.com".to_string(),
            api_key: "sk-test".to_string(),
            upstream_protocol: ProviderProtocol::Openai,
            forced_model: None,
            copilot_token_manager: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        assert!(config.copilot_token_manager.is_none());
        assert!(config.forced_model.is_none());
    }

    #[test]
    fn test_repair_single_tool_call_fills_required_dir_path_for_anyof_string() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["dir_path"],
            "properties": {
                "dir_path": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                }
            }
        });
        let mut tc = serde_json::json!({
            "function": {
                "name": "ReadFolder",
                "arguments": "{}"
            }
        });
        repair_single_tool_call(&mut tc, &schema);
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| serde_json::json!({}));
        assert_eq!(args["dir_path"], ".");
    }

    #[test]
    fn test_repair_single_tool_call_fills_required_dir_path_when_property_schema_missing() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["dir_path"]
        });
        let mut tc = serde_json::json!({
            "function": {
                "name": "ReadFolder",
                "arguments": "{}"
            }
        });
        repair_single_tool_call(&mut tc, &schema);
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| serde_json::json!({}));
        assert_eq!(args["dir_path"], ".");
    }

    #[test]
    fn test_repair_tool_call_args_matches_schema_name_case_insensitively() {
        let mut schemas = std::collections::HashMap::new();
        schemas.insert(
            "readfolder".to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["dir_path"],
                "properties": {"dir_path": {"type": "string"}}
            }),
        );
        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "ReadFolder",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        });
        let repaired = repair_tool_call_args(response, &schemas);
        let args = repaired["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap_or("{}");
        let args: Value = serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({}));
        assert_eq!(args["dir_path"], ".");
    }

    #[test]
    fn test_parse_gemini_path_missing_model() {
        // Empty model name still parses (model is "")
        let result = parse_gemini_path("/v1beta/models/:generateContent");
        assert_eq!(result.unwrap().0, "");
    }

    #[test]
    fn test_repair_single_tool_call_non_json_arguments() {
        // Non-JSON arguments string should not panic; should produce valid JSON output
        let schema = serde_json::json!({
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {"type": "string"}
            }
        });
        let mut tc = serde_json::json!({
            "function": {
                "name": "ReadFile",
                "arguments": "this is not json at all !!!"
            }
        });
        repair_single_tool_call(&mut tc, &schema);
        // Should not panic and arguments should be valid JSON
        let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
        let args: Value =
            serde_json::from_str(args_str).expect("repaired arguments should be valid JSON");
        // The required file_path param should be filled with "." since it's path-like
        assert_eq!(args["file_path"], ".");
    }
}
