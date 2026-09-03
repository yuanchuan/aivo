//! Provider model-catalog core: fetch + cache of model lists and per-model
//! metadata (context window, max output, pricing) for every provider family.
//! Pure service layer — pickers, spinners, and table rendering stay in
//! `commands::models`.
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, CopilotTokenManager,
};
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::http_utils::normalize_base_url;
use crate::services::models_cache::{ModelMetadata, ModelsCache, full_catalog_key};
use crate::services::provider_profile::{
    ModelListingStrategy, cloudflare_ai_base, is_aivo_starter_base, provider_profile_for_key,
};
use crate::services::session_store::{ApiKey, SessionStore};

/// Rich model information for display. Fields are populated from API metadata
/// when the provider returns them.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Raw context window; not exposed via `--json`.
    #[serde(skip)]
    pub context_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output: Option<String>,
    /// Raw max-output tokens; not exposed via `--json`.
    #[serde(skip)]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    /// Marked deprecated by models.dev; display-only, never cached.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub deprecated: bool,
    /// Reasoning-effort levels advertised by `/v1/models`; drives `/effort` for
    /// catalog-only models (e.g. `aivo/starter`) the snapshot lacks.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
    /// Live catalog `image_input` when advertised. `None` = unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_input: Option<bool>,
}

impl ModelInfo {
    pub(crate) fn id_only(id: String) -> Self {
        Self {
            id,
            context: None,
            context_tokens: None,
            max_output: None,
            max_output_tokens: None,
            input_price: None,
            output_price: None,
            multiplier: None,
            deprecated: false,
            reasoning_efforts: Vec::new(),
            image_input: None,
        }
    }
}

#[derive(Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

/// Loosely-parsed model entry. Only `id` is required; everything else is
/// extracted by searching field names for known patterns so we adapt to
/// any provider's response shape (OpenRouter, Vercel, OpenAI, etc.).
#[derive(Deserialize)]
struct OpenAIModel {
    id: String,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

impl OpenAIModel {
    fn into_model_info(self, price_scale: f64) -> ModelInfo {
        let context_tokens = self.find_context();
        let context = context_tokens.map(format_token_count);
        let max_output_tokens = self.find_max_output();
        let max_output = max_output_tokens.map(format_token_count);
        let (input_price, output_price) = self.find_pricing(price_scale);
        let multiplier = self.find_multiplier();
        let reasoning_efforts = self.find_reasoning_efforts();
        let image_input = self.find_image_input();
        ModelInfo {
            id: self.id,
            context,
            context_tokens,
            max_output,
            max_output_tokens,
            input_price,
            output_price,
            multiplier,
            deprecated: false,
            reasoning_efforts,
            image_input,
        }
    }

    /// Levels from a `reasoning_efforts` string array (exact key), lowercased;
    /// empty when absent or not a string array.
    fn find_reasoning_efforts(&self) -> Vec<String> {
        self.extra
            .get("reasoning_efforts")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Exact `image_input` boolean when the catalog advertises it.
    fn find_image_input(&self) -> Option<bool> {
        self.extra.get("image_input")?.as_bool()
    }

    /// Search for a Copilot-style billing multiplier (`billing.multiplier`).
    fn find_multiplier(&self) -> Option<f64> {
        let billing = self.extra.get("billing")?.as_object()?;
        billing.get("multiplier").and_then(|v| v.as_f64())
    }

    /// Search for a context-window field: context_length, context_window,
    /// context_size, max_context_length, max_input_tokens, etc.
    fn find_context(&self) -> Option<u64> {
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if k.contains("context")
                && !k.contains("price")
                && !k.contains("cost")
                && let Some(n) = http_utils::parse_token_u64(val)
                && n > 0
            {
                return Some(n);
            }
        }
        // Fallback: direct lookup
        self.extra
            .get("max_input_tokens")
            .and_then(http_utils::parse_token_u64)
            .filter(|&n| n > 0)
    }

    /// Search for a max-output field: max_tokens, max_output_tokens,
    /// max_completion_tokens (top-level or nested in sub-objects).
    fn find_max_output(&self) -> Option<u64> {
        // Top-level fields
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if (k == "max_tokens"
                || k == "max_output_tokens"
                || k == "max_completion_tokens"
                || k == "max_output")
                && !k.contains("price")
                && let Some(n) = http_utils::parse_token_u64(val)
                && n > 0
            {
                return Some(n);
            }
        }
        // Nested in sub-objects (e.g. top_provider.max_completion_tokens)
        for (_key, val) in &self.extra {
            if let Some(obj) = val.as_object() {
                for (k2, v2) in obj {
                    let k = k2.to_ascii_lowercase();
                    if k.contains("max")
                        && (k.contains("output") || k.contains("completion") || k.contains("token"))
                        && let Some(n) = http_utils::parse_token_u64(v2)
                        && n > 0
                    {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    fn find_pricing(&self, price_scale: f64) -> (Option<String>, Option<String>) {
        let (input, output) = self.raw_pricing();
        (
            input.and_then(|v| format_scaled_price(v, price_scale)),
            output.and_then(|v| format_scaled_price(v, price_scale)),
        )
    }

    /// Raw (input, output) prices from a nested "pricing"/"price" object.
    fn raw_pricing(&self) -> (Option<f64>, Option<f64>) {
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if (k == "pricing" || k == "price" || k == "prices")
                && let Some(obj) = val.as_object()
            {
                let input =
                    find_price_value(obj, &["input", "prompt", "input_cost", "input_price"]);
                let output = find_price_value(
                    obj,
                    &["output", "completion", "output_cost", "output_price"],
                );
                return (input, output);
            }
        }
        (None, None)
    }
}

/// Search a pricing object for a field matching one of the candidate names.
fn find_price_value(obj: &serde_json::Map<String, Value>, candidates: &[&str]) -> Option<f64> {
    for candidate in candidates {
        if let Some(val) = obj.get(*candidate) {
            if let Some(s) = val.as_str() {
                return s.trim().parse().ok();
            }
            if let Some(n) = val.as_f64() {
                return Some(n);
            }
        }
    }
    None
}

#[derive(Deserialize)]
struct GeminiModelsResponse {
    models: Vec<GeminiModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModel {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
    #[serde(default)]
    input_token_limit: Option<u64>,
    #[serde(default)]
    output_token_limit: Option<u64>,
}

#[derive(Deserialize)]
struct CloudflareModelsResponse {
    #[serde(default)]
    result: Vec<CloudflareModel>,
    result_info: Option<CloudflareResultInfo>,
}

#[derive(Deserialize)]
struct CloudflareModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct CloudflareResultInfo {
    total_pages: Option<u32>,
}

/// Returns just the scheme + host + port of a URL, e.g. "https://api.example.com".
/// Used to probe the root when the base URL includes a path segment like /endpoint.
fn url_origin(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let mut origin = format!("{}://{}", parsed.scheme(), parsed.host_str()?);
    if let Some(port) = parsed.port() {
        origin.push_str(&format!(":{}", port));
    }
    Some(origin)
}

/// Candidate models endpoints in probe order: exact base-URL paths first,
/// origin as fallback — origin-first broke path-suffixed keys (issue #24).
fn openai_models_candidates(base: &str) -> Vec<String> {
    let model_endpoints = |b: &str| [format!("{}/v1/models", b), format!("{}/models", b)];
    let mut candidates = Vec::new();
    candidates.extend(model_endpoints(base));
    if let Some(origin) = url_origin(base)
        && origin != base
    {
        candidates.extend(model_endpoints(&origin));
    }
    candidates
}

/// Returns true if the model is suitable for text chat. Single source of
/// truth lives in `services::model_compat::text_chat_incompat_reason`; this
/// is the boolean flip used by upstream filters (Copilot, hoisted
/// `chat_only` gate in `fetch_models_detailed_filtered`).
pub(crate) fn is_text_chat_model(id: &str) -> bool {
    crate::services::model_compat::text_chat_incompat_reason(id).is_none()
}

/// Copilot's Claude/OpenAI chat routing uses the chat completions API.
/// Exclude clearly responses-only Codex models that the endpoint rejects.
fn is_copilot_chat_model(id: &str) -> bool {
    is_text_chat_model(id) && !id.to_lowercase().contains("codex")
}

fn cloudflare_model_name(model: CloudflareModel) -> String {
    model
        .name
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(model.id)
}

/// Format a token count as a compact display string: 1M, 200K, 128K, etc.
fn format_token_count(n: u64) -> String {
    if n >= 1_000_000 {
        let m = n / 1_000_000;
        let remainder = (n % 1_000_000) / 100_000;
        if remainder > 0 {
            format!("{}.{}M", m, remainder)
        } else {
            format!("{}M", m)
        }
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

/// Every candidate 404'd — the provider has no models endpoint (e.g. Codex
/// ChatGPT OAuth); explicit pickers soft-fall-back instead of surfacing it.
#[derive(Debug)]
pub(crate) struct NoModelsEndpoint;

impl std::fmt::Display for NoModelsEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider has no models endpoint")
    }
}

impl std::error::Error for NoModelsEndpoint {}

pub(crate) fn is_no_models_endpoint(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<NoModelsEndpoint>())
}

/// Build a friendly error for a non-2xx response from a model-listing endpoint.
/// Leads with the user's intent ("No models found" for 404, "Could not fetch
/// models" otherwise), then explains why — extracting JSON `error.message`
/// when present and dropping HTML payloads. Replaces the raw multi-KB HTML
/// page that providers return for missing endpoints.
fn friendly_api_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    friendly_api_error_with_hint(status, body, status_hint(status))
}

/// `friendly_api_error` variant for OAuth endpoints: 401/403 points at `aivo
/// keys reauth` instead of the API-key hint `status_hint` gives.
fn friendly_oauth_api_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    let hint = match status.as_u16() {
        401 | 403 => Some("sign-in expired or revoked — run `aivo keys reauth` to sign in again"),
        _ => status_hint(status),
    };
    friendly_api_error_with_hint(status, body, hint)
}

fn friendly_api_error_with_hint(
    status: reqwest::StatusCode,
    body: &str,
    hint: Option<&str>,
) -> anyhow::Error {
    let lead = if status == reqwest::StatusCode::NOT_FOUND {
        "No models found"
    } else {
        "Could not fetch models"
    };
    let detail = extract_error_detail(body);
    let reason = match (detail, hint) {
        (Some(d), Some(h)) => format!("{} ({})", d, h),
        (Some(d), None) => d,
        (None, Some(h)) => h.to_string(),
        (None, None) => format!("server returned {}", status),
    };
    let message = format!("{} — {}", lead, reason);
    match status.as_u16() {
        401 | 403 => anyhow::Error::new(crate::errors::CLIError::new(
            message,
            crate::errors::ErrorCategory::Auth,
            None::<String>,
            None::<String>,
        )),
        _ => anyhow::anyhow!(message),
    }
}

/// Pull a one-line message out of an upstream error body. Recognizes common
/// JSON shapes (OpenAI/Anthropic `error.message`, Cloudflare `errors[0].message`,
/// generic `message`). Returns `None` for HTML payloads or empty bodies so the
/// caller can fall back to a status-only hint.
fn extract_error_detail(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        if let Some(msg) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value.get("error").and_then(|e| e.as_str()) {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value
            .get("errors")
            .and_then(|e| e.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("message"))
            .and_then(|m| m.as_str())
        {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value.get("message").and_then(|m| m.as_str()) {
            return Some(truncate_message(msg));
        }
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("<!doctype") || lower.starts_with("<html") || lower.starts_with("<?xml") {
        return None;
    }
    Some(truncate_message(trimmed))
}

fn truncate_message(s: &str) -> String {
    let s = s.trim();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= 200 {
            out.push('…');
            return out;
        }
        out.push(c);
    }
    out
}

fn status_hint(status: reqwest::StatusCode) -> Option<&'static str> {
    match status.as_u16() {
        401 => Some("authentication failed — check the API key with `aivo keys`"),
        403 => Some("this API key may not have permission to list models"),
        404 => {
            Some("the base URL may be wrong, or this provider doesn't expose a model listing API")
        }
        405 => Some("this provider does not expose a model listing API"),
        429 => Some("rate limited — try again shortly"),
        500..=599 => Some("the provider returned a server error — try again later"),
        _ => None,
    }
}

/// Per-1M multipliers for the quoting conventions seen in the wild:
/// per-1M (publicai), per-1K (some relay gateways), per-token (OpenRouter).
const PRICE_SCALES: [f64; 3] = [1.0, 1_000.0, 1_000_000.0];

/// Dollars-per-1M band that real model prices land in.
const SANE_PER_M: std::ops::RangeInclusive<f64> = 0.01..=1_000.0;

/// A lone price is ambiguous (0.0003: per-token $300/1M or per-1K $0.30/1M),
/// so every price in the response votes for each scale landing it in-band;
/// ties take the smallest multiplier (budget per-1K far outnumbers premium
/// per-token in practice).
fn infer_price_scale(models: &[OpenAIModel]) -> f64 {
    let mut votes = [0u32; PRICE_SCALES.len()];
    for m in models {
        let (input, output) = m.raw_pricing();
        for v in [input, output].into_iter().flatten() {
            if v <= 0.0 {
                continue;
            }
            for (slot, scale) in votes.iter_mut().zip(PRICE_SCALES) {
                if SANE_PER_M.contains(&(v * scale)) {
                    *slot += 1;
                }
            }
        }
    }
    let best = votes.iter().copied().max().unwrap_or(0);
    if best == 0 {
        return 1.0;
    }
    let winner = votes.iter().position(|&v| v == best).unwrap_or(0);
    PRICE_SCALES[winner]
}

fn format_scaled_price(raw: f64, scale: f64) -> Option<String> {
    if raw <= 0.0 {
        return None;
    }
    let per_m = raw * scale;
    Some(format!("${per_m:.2}"))
}

pub(crate) async fn fetch_models(client: &Client, key: &ApiKey) -> Result<Vec<String>> {
    fetch_models_detailed(client, key)
        .await
        .map(|v| v.into_iter().map(|m| m.id).collect())
}

/// Fills missing context/max-output columns from the embedded models.dev
/// snapshot. Provider-reported values always win; never feed the result back
/// into the models cache.
pub(crate) fn enrich_from_snapshot(model: &mut ModelInfo) {
    let Some(snap) = crate::services::model_metadata::snapshot_limits(&model.id) else {
        return;
    };
    model.deprecated = snap.deprecated;
    if model.context_tokens.is_none()
        && let Some(context) = snap.context
    {
        model.context_tokens = Some(context);
        model.context = Some(format_token_count(context));
    }
    if model.max_output_tokens.is_none()
        && let Some(output) = snap.output
    {
        model.max_output_tokens = Some(output);
        model.max_output = Some(format_token_count(output));
    }
}

/// Pulls every displayed column out of a detailed model list so the cache can
/// reproduce the table on a warm read. Skips models that returned no
/// metadata at all (id-only entries from providers like Cloudflare).
pub(crate) fn build_metadata_map(models: &[ModelInfo]) -> HashMap<String, ModelMetadata> {
    models
        .iter()
        .filter_map(|m| {
            let meta = ModelMetadata {
                context_window: m.context_tokens,
                max_output: m.max_output.clone(),
                max_output_tokens: m.max_output_tokens,
                input_price: m.input_price.clone(),
                output_price: m.output_price.clone(),
                multiplier: m.multiplier,
                reasoning_efforts: m.reasoning_efforts.clone(),
                image_input: m.image_input,
            };
            let any = meta.context_window.is_some()
                || meta.max_output.is_some()
                || meta.input_price.is_some()
                || meta.output_price.is_some()
                || meta.multiplier.is_some()
                || !meta.reasoning_efforts.is_empty()
                || meta.image_input.is_some();
            any.then(|| (m.id.clone(), meta))
        })
        .collect()
}

pub(crate) fn model_cache_key_for_key(key: &ApiKey) -> String {
    if key.is_cursor_acp() {
        crate::services::cursor_acp::cursor_models_cache_identity(key)
    } else {
        key.base_url.clone()
    }
}

pub(crate) fn full_catalog_cache_key_for_key(key: &ApiKey) -> String {
    if key.is_cursor_acp() {
        // Cursor's `models` listing returns chat-capable ids only — there's no
        // image/audio/embedding catalog to separate. Collapse the `#all`
        // namespace into the bare key so `aivo models cursor`, the model
        // picker, and the in-router cache (see `cursor_bridge::cached_models`)
        // all hit a single shared entry on disk.
        return model_cache_key_for_key(key);
    }
    full_catalog_key(&model_cache_key_for_key(key))
}

/// Kimi `/models` rows → display rows; the endpoint reports context inline.
pub(crate) fn kimi_model_infos(
    models: Vec<crate::services::kimi_oauth::KimiModel>,
) -> Vec<ModelInfo> {
    models
        .into_iter()
        .map(|m| ModelInfo {
            context: m.context_length.map(format_token_count),
            context_tokens: m.context_length,
            image_input: None,
            ..ModelInfo::id_only(m.id)
        })
        .collect()
}

/// Rebuilds the `aivo models` row list from a cached id list and metadata
/// map. Models present in `ids` but missing from `metadata` (e.g. Cloudflare
/// id-only entries) render as plain rows.
pub(crate) fn models_from_cache(
    ids: Vec<String>,
    metadata: HashMap<String, ModelMetadata>,
) -> Vec<ModelInfo> {
    ids.into_iter()
        .map(|id| {
            let m = metadata.get(&id).cloned().unwrap_or_default();
            ModelInfo {
                id,
                context: m.context_window.map(format_token_count),
                context_tokens: m.context_window,
                max_output: m.max_output,
                max_output_tokens: m.max_output_tokens,
                input_price: m.input_price,
                output_price: m.output_price,
                multiplier: m.multiplier,
                deprecated: false,
                reasoning_efforts: m.reasoning_efforts,
                image_input: m.image_input,
            }
        })
        .collect()
}

/// Best-effort warm of the per-model metadata cache (context window, pricing)
/// for a key's full catalog, so `model_metadata::resolve_limits` can answer
/// from cache without a picker/`aivo models` run first. Used by `aivo code` to
/// populate the footer context-utilization stat on the `-m <model>` path, which
/// otherwise never fetches the catalog. Silently no-ops on fetch failure.
pub(crate) async fn warm_full_catalog_metadata(client: &Client, key: &ApiKey, cache: &ModelsCache) {
    if key.is_any_oauth() || crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return;
    }
    let Ok(models) = fetch_models_detailed_filtered(client, key, false).await else {
        return;
    };
    let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
    let metadata = build_metadata_map(&models);
    cache
        .set_with_metadata(&full_catalog_cache_key_for_key(key), ids, metadata)
        .await;
}

/// Cache-first full catalog fetch; stores metadata so `resolve_limits` finds the
/// context window (else Pi/opencode fall back to a 128k default).
pub(crate) async fn fetch_all_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    if !bypass_cache && let Some(cached) = full_catalog_cached(key, cache).await {
        return Ok(cached);
    }
    let detailed = fetch_models_detailed_filtered(client, key, false).await?;
    let models: Vec<String> = detailed.iter().map(|m| m.id.clone()).collect();
    if !crate::services::provider_profile::is_ollama_base(&key.base_url) {
        cache
            .set_with_metadata(
                &full_catalog_cache_key_for_key(key),
                models.clone(),
                build_metadata_map(&detailed),
            )
            .await;
    }
    Ok(models)
}

/// Peek the cached full catalog, honoring the same skips as
/// `fetch_all_models_cached`: Ollama never caches (lists locally, instantly),
/// and a cursor entry poisoned by the old parser bug counts as a miss.
pub(crate) async fn full_catalog_cached(key: &ApiKey, cache: &ModelsCache) -> Option<Vec<String>> {
    if crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return None;
    }
    let cached = cache.get(&full_catalog_cache_key_for_key(key)).await?;
    (!cursor_cache_looks_corrupt(key, &cached)).then_some(cached)
}

/// Whether the key's cached catalog is still within its TTL. OAuth and Ollama
/// keys can't be harvested (`warm_full_catalog_metadata` no-ops), so they count
/// as fresh to skip a pointless warm.
pub(crate) async fn full_catalog_metadata_fresh(key: &ApiKey, cache: &ModelsCache) -> bool {
    if key.is_any_oauth() || crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return true;
    }
    full_catalog_cached(key, cache).await.is_some()
}

/// Force-refetch when a cursor cache entry was populated by the older
/// parser bug — historical builds turned cursor-agent's logged-out
/// "No models available for this account." into the model id `No`, and
/// shipped that into `~/.config/aivo/models-cache.json`.
///
/// The earlier "must contain a digit / hyphen / slash" heuristic was too
/// broad: cursor's own catalog includes `auto` (no digit, no hyphen, no
/// slash), so every cache entry was treated as corrupt and refetched on
/// every call. Match the actual bad string instead.
pub(crate) fn cursor_cache_looks_corrupt(key: &ApiKey, cached: &[String]) -> bool {
    key.is_cursor_acp() && cached.iter().any(|id| id.trim() == "No")
}

/// Pure decision for `starter_model_still_available`: given a freshly-fetched
/// catalog, decide whether `model` is still listed. An empty catalog is
/// treated as a transient hiccup and passes through rather than flagging
/// every user's model as gone.
fn model_present_in_catalog(catalog: &[String], model: &str) -> bool {
    catalog.is_empty() || catalog.iter().any(|m| m == model)
}

/// Whether `model` is still listed for the aivo-starter server. `true` (skip)
/// for non-starter keys, the `aivo/starter` sentinel, `MODEL_DEFAULT_PLACEHOLDER`,
/// and an un-cached catalog.
///
/// Non-blocking by design: reads only the last-known catalog and never fetches,
/// so the chat/run/start launch never waits on `/v1/models` (that cost ~1s every
/// launch). Reads both starter cache spellings — the picker/`aivo models` key by
/// the sentinel, chat's background warm by the post-swap real URL — which are
/// never unified (the routers depend on the real-URL entry). The warm keeps it
/// fresh; a removal shows up a launch late, with the first request's error
/// bridging the gap.
pub(crate) async fn starter_model_still_available(
    key: &ApiKey,
    cache: &ModelsCache,
    model: &str,
) -> bool {
    if !is_aivo_starter_base(&key.base_url) {
        return true;
    }
    if model == crate::constants::AIVO_STARTER_MODEL
        || model == crate::constants::MODEL_DEFAULT_PLACEHOLDER
    {
        return true;
    }
    let mut catalog = cache
        .model_ids(crate::constants::AIVO_STARTER_SENTINEL)
        .await
        .unwrap_or_default();
    if let Some(more) = cache
        .model_ids(crate::constants::AIVO_STARTER_REAL_URL)
        .await
    {
        catalog.extend(more);
    }
    model_present_in_catalog(&catalog, model)
}

/// Fetch models with full metadata from the API where available.
/// Providers like OpenRouter/Vercel return context window, pricing, and max output.
/// Google returns inputTokenLimit and outputTokenLimit.
/// Other providers return just IDs.
pub(crate) async fn fetch_models_detailed(client: &Client, key: &ApiKey) -> Result<Vec<ModelInfo>> {
    fetch_models_detailed_filtered(client, key, true).await
}

/// Implementation behind `fetch_models_detailed` / `fetch_all_models_cached`. When
/// `chat_only` is true (the default), applies `is_text_chat_model` to the
/// OpenAI-compatible / Anthropic / CloudflareSearch branches so chat pickers
/// don't surface image, audio, or embedding models. Set to false for the
/// image command and `aivo models`, which show the full catalog.
pub(crate) async fn fetch_models_detailed_filtered(
    client: &Client,
    key: &ApiKey,
    chat_only: bool,
) -> Result<Vec<ModelInfo>> {
    // Grok is the exception: its OAuth token lists models on the CLI proxy, so
    // every picker/warm path (`aivo code`, `start`, ai_launcher) can enumerate
    // it. Handle it before the generic OAuth bail. id-only here; enrichment
    // fills limits downstream, matching the `aivo models` handler.
    if key.is_grok_oauth() {
        let mut key = key.clone();
        SessionStore::decrypt_key_secret(&mut key)?;
        let mut creds = crate::services::grok_oauth::GrokOAuthCredential::from_json(&key.key)?;
        let ids = crate::services::grok_oauth::fetch_model_ids(&mut creds, None).await?;
        return Ok(ids.into_iter().map(ModelInfo::id_only).collect());
    }
    if key.is_kimi_oauth() {
        let mut key = key.clone();
        SessionStore::decrypt_key_secret(&mut key)?;
        let mut creds = crate::services::kimi_oauth::KimiOAuthCredential::from_json(&key.key)?;
        let models = crate::services::kimi_oauth::fetch_models(&mut creds, None).await?;
        return Ok(kimi_model_infos(models));
    }
    if key.is_codex_oauth() {
        let ids = crate::services::codex_oauth::known_model_ids();
        return Ok(ids.into_iter().map(ModelInfo::id_only).collect());
    }
    // Native `/v1/models` accepts the OAuth bearer (no oauth beta needed) and
    // returns live token limits.
    if key.is_claude_oauth() {
        let mut key = key.clone();
        SessionStore::decrypt_key_secret(&mut key)?;
        let token =
            crate::services::claude_oauth::ClaudeOAuthCredential::from_json(&key.key)?.token;
        let url = format!(
            "{}/v1/models?limit=1000",
            crate::services::claude_oauth::upstream_base_url().trim_end_matches('/')
        );
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("anthropic-version", "2023-06-01")
            .send_logged()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(friendly_oauth_api_error(status, &body));
        }
        let resp: OpenAIModelsResponse = response.json().await?;
        let scale = infer_price_scale(&resp.data);
        let mut models: Vec<ModelInfo> = resp
            .data
            .into_iter()
            .map(|m| m.into_model_info(scale))
            .collect();
        if chat_only {
            models.retain(|m| is_text_chat_model(&m.id));
        }
        return Ok(models);
    }
    if key.is_any_oauth() {
        anyhow::bail!(
            "Key '{}' is an OAuth credential with no model listing API. Use `{}` to launch directly, or switch to a regular API key with `aivo use`.",
            key.display_name(),
            key.oauth_tool_hint()
        );
    }
    let base = normalize_base_url(&key.base_url);
    let profile = provider_profile_for_key(key);

    let raw: Vec<ModelInfo> = match profile.model_listing_strategy {
        ModelListingStrategy::AivoStarter => {
            let starter_base = crate::constants::AIVO_STARTER_REAL_URL;
            let url = format!("{}/v1/models", starter_base.trim_end_matches('/'));
            let response = crate::services::device_fingerprint::with_starter_headers(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
            )
            .send_logged()
            .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            let scale = infer_price_scale(&resp.data);
            Ok::<_, anyhow::Error>(
                resp.data
                    .into_iter()
                    .map(|m| m.into_model_info(scale))
                    .collect(),
            )
        }
        ModelListingStrategy::Ollama => {
            crate::services::ollama::ensure_ready().await?;
            let ids = crate::services::ollama::list_models().await?;
            Ok(ids.into_iter().map(ModelInfo::id_only).collect())
        }
        ModelListingStrategy::Copilot => {
            let tm = CopilotTokenManager::new(key.key.as_str().to_string());
            let (copilot_token, api_endpoint) = tm.get_token().await?;
            let url = format!("{}/models", api_endpoint.trim_end_matches('/'));
            let response = client
                .get(&url)
                .header("Authorization", format!("Bearer {}", copilot_token))
                .header("Editor-Version", COPILOT_EDITOR_VERSION)
                .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
                .header("X-GitHub-Api-Version", "2025-10-01")
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            let scale = infer_price_scale(&resp.data);
            Ok(resp
                .data
                .into_iter()
                .filter(|m| is_copilot_chat_model(&m.id))
                .map(|m| m.into_model_info(scale))
                .collect())
        }
        ModelListingStrategy::CursorAcp => {
            let ids = crate::services::cursor_acp::list_cursor_models(key).await?;
            Ok(ids.into_iter().map(ModelInfo::id_only).collect())
        }
        ModelListingStrategy::Google => {
            let url = build_google_models_url(base);
            let response = client
                .get(&url)
                .header("x-goog-api-key", key.key.as_str())
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: GeminiModelsResponse = response.json().await?;
            Ok(resp
                .models
                .into_iter()
                .filter(|m| {
                    m.supported_generation_methods
                        .iter()
                        .any(|method| method == "generateContent")
                })
                .map(|m| {
                    let id = m
                        .name
                        .strip_prefix("models/")
                        .unwrap_or(&m.name)
                        .to_string();
                    ModelInfo {
                        context: m.input_token_limit.map(format_token_count),
                        context_tokens: m.input_token_limit,
                        max_output: m.output_token_limit.map(format_token_count),
                        max_output_tokens: m.output_token_limit,
                        input_price: None,
                        output_price: None,
                        multiplier: None,
                        deprecated: false,
                        reasoning_efforts: Vec::new(),
                        image_input: None,
                        id,
                    }
                })
                .collect())
        }
        ModelListingStrategy::Anthropic => {
            let url = build_anthropic_models_url(&key.base_url);
            let response = client
                .get(&url)
                .header("x-api-key", key.key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            let scale = infer_price_scale(&resp.data);
            Ok(resp
                .data
                .into_iter()
                .map(|m| m.into_model_info(scale))
                .collect())
        }
        ModelListingStrategy::CloudflareSearch => {
            let cloudflare_base = cloudflare_ai_base(base)
                .ok_or_else(|| anyhow::anyhow!("Failed to normalize Cloudflare AI base URL"))?;
            let auth = format!("Bearer {}", key.key.as_str());
            let mut page = 1u32;
            let mut seen = HashSet::new();
            let mut models = Vec::new();

            loop {
                let url = format!(
                    "{}/models/search?hide_experimental=true&page={}&per_page=100",
                    cloudflare_base, page
                );
                let response = client
                    .get(&url)
                    .header("Authorization", &auth)
                    .send_logged()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(friendly_api_error(status, &body));
                }

                let resp: CloudflareModelsResponse = response.json().await?;
                for model in resp.result.into_iter().map(cloudflare_model_name) {
                    if seen.insert(model.clone()) {
                        models.push(ModelInfo::id_only(model));
                    }
                }

                let total_pages = resp
                    .result_info
                    .and_then(|info| info.total_pages)
                    .unwrap_or(page);
                if page >= total_pages {
                    break;
                }
                page += 1;
            }

            Ok(models)
        }
        ModelListingStrategy::OpenAiCompatible => {
            let candidates = openai_models_candidates(base);
            let auth = format!("Bearer {}", key.key.as_str());

            let mut last_err = String::new();
            let mut missing_everywhere = true;
            let mut success: Option<Vec<ModelInfo>> = None;
            for url in &candidates {
                let response = crate::services::opencode_session::with_session_header(
                    client.get(url).header("Authorization", &auth),
                    url,
                    None,
                )
                .send_logged()
                .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    let error = friendly_api_error(status, &body);
                    // Ambiguous 4xx and garbage 200s may be "wrong path" on a
                    // proxy — keep probing (issue #24); only auth/rate/5xx bail.
                    let code = status.as_u16();
                    if crate::services::provider_protocol::is_terminal_upstream_error(code) {
                        return Err(error);
                    }
                    missing_everywhere &=
                        crate::services::provider_protocol::is_endpoint_missing(code);
                    last_err = error.to_string();
                    continue;
                }

                let body = response.text().await.unwrap_or_default();
                match serde_json::from_str::<OpenAIModelsResponse>(&body) {
                    Ok(resp) => {
                        let scale = infer_price_scale(&resp.data);
                        success = Some(
                            resp.data
                                .into_iter()
                                .map(|m| m.into_model_info(scale))
                                .collect(),
                        );
                        break;
                    }
                    Err(e) => {
                        missing_everywhere = false;
                        last_err = format!("Invalid models response from {}: {}", url, e);
                    }
                }
            }

            match success {
                Some(v) => Ok(v),
                None if missing_everywhere => {
                    Err(anyhow::Error::new(NoModelsEndpoint).context(last_err))
                }
                None => anyhow::bail!("{}", last_err),
            }
        }
    }?
    .into_iter()
    .filter(|m| !m.id.is_empty())
    .collect();

    Ok(if chat_only {
        raw.into_iter()
            .filter(|m| is_text_chat_model(&m.id))
            .collect()
    } else {
        raw
    })
}

fn build_google_models_url(base_url: &str) -> String {
    let base = normalize_base_url(base_url).trim_end_matches('/');
    if base.ends_with("/v1beta") || base.ends_with("/v1") {
        format!("{}/models", base)
    } else if base.ends_with("/models") {
        base.to_string()
    } else {
        format!("{}/v1beta/models", base)
    }
}

fn build_anthropic_models_url(base_url: &str) -> String {
    http_utils::build_target_url(base_url, "/v1/models")
}

/// Cache-aware wrapper around `fetch_models`.
/// Returns cached result if present and not expired (unless `bypass_cache` is true).
/// On cache miss, fetches from the network and writes the result to the cache.
pub(crate) async fn fetch_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    // Ollama lists local models instantly — skip cache entirely.
    let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
    let cache_key = model_cache_key_for_key(key);
    if !bypass_cache
        && !is_ollama
        && let Some(cached) = cache.get(&cache_key).await
    {
        return Ok(cached);
    }
    let models = fetch_models(client, key).await?;
    if !is_ollama {
        cache.set(&cache_key, models.clone()).await;
    }
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::models_cache::ModelsCache;
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::time::{Duration, timeout};

    fn make_key(url: &str) -> ApiKey {
        use zeroize::Zeroizing;
        ApiKey {
            id: "1".to_string(),
            name: "test".to_string(),
            base_url: url.to_string(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            protocol_routes: Default::default(),
            routing_schema_version: 0,
            key: Zeroizing::new("sk-test".to_string()),
            created_at: "2026-01-01".to_string(),
        }
    }

    #[test]
    fn openai_models_candidates_probe_base_path_before_origin() {
        assert_eq!(
            openai_models_candidates("https://proxy.example.com/stg"),
            vec![
                "https://proxy.example.com/stg/v1/models",
                "https://proxy.example.com/stg/models",
                "https://proxy.example.com/v1/models",
                "https://proxy.example.com/models",
            ]
        );
        assert_eq!(
            openai_models_candidates("https://api.example.com"),
            vec![
                "https://api.example.com/v1/models",
                "https://api.example.com/models",
            ]
        );
    }

    #[tokio::test]
    async fn openai_models_auth_error_does_not_probe_fallback_path() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (paths_tx, mut paths_rx) = tokio::sync::mpsc::unbounded_channel();

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let Ok(Ok((mut socket, _))) =
                    timeout(Duration::from_secs(2), listener.accept()).await
                else {
                    return;
                };
                let mut request = vec![0u8; 4096];
                let size = socket.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or_default()
                    .to_string();
                let _ = paths_tx.send(path);

                let body = r#"{"error":{"message":"Invalid token","type":"api_error"}}"#;
                let response = format!(
                    "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let key = make_key(&format!("http://{}", address));
        let client = crate::services::http_utils::router_http_client_loopback();
        let error = fetch_models_detailed_filtered(&client, &key, false)
            .await
            .expect_err("an invalid token must fail model listing");

        assert!(error.to_string().contains("Invalid token"));
        assert_eq!(paths_rx.recv().await.as_deref(), Some("/v1/models"));
        assert!(
            timeout(Duration::from_millis(200), paths_rx.recv())
                .await
                .is_err(),
            "an auth error must not trigger the fallback /models request"
        );
        server.abort();
    }

    async fn spawn_sequence_server(
        responses: Vec<(u16, String)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut paths = Vec::new();
            for (status, body) in responses {
                let Ok(Ok((mut socket, _))) =
                    timeout(Duration::from_secs(2), listener.accept()).await
                else {
                    break;
                };
                let mut request = vec![0u8; 4096];
                let size = socket.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                paths.push(
                    request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or_default()
                        .to_string(),
                );
                let response = format!(
                    "HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            paths
        });
        (format!("http://{}", address), handle)
    }

    #[tokio::test]
    async fn openai_models_ambiguous_4xx_falls_through_to_next_candidate() {
        let (base, handle) = spawn_sequence_server(vec![
            (400, r#"{"error":{"message":"bad path"}}"#.to_string()),
            (200, r#"{"data":[{"id":"listed-model"}]}"#.to_string()),
        ])
        .await;
        let key = make_key(&base);
        let client = crate::services::http_utils::router_http_client_loopback();
        let models = fetch_models_detailed_filtered(&client, &key, false)
            .await
            .expect("a 400 on /v1/models must not abort the candidate cascade");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "listed-model");
        assert_eq!(handle.await.unwrap(), vec!["/v1/models", "/models"]);
    }

    #[tokio::test]
    async fn openai_models_catch_all_200_falls_through_to_next_candidate() {
        let (base, handle) = spawn_sequence_server(vec![
            (200, "<html>catch-all index</html>".to_string()),
            (200, r#"{"data":[{"id":"listed-model"}]}"#.to_string()),
        ])
        .await;
        let key = make_key(&base);
        let client = crate::services::http_utils::router_http_client_loopback();
        let models = fetch_models_detailed_filtered(&client, &key, false)
            .await
            .expect("a non-JSON 200 must not abort the candidate cascade");

        assert_eq!(models.len(), 1);
        assert_eq!(handle.await.unwrap(), vec!["/v1/models", "/models"]);
    }

    #[tokio::test]
    async fn openai_models_all_404_marks_no_models_endpoint() {
        let (base, _handle) =
            spawn_sequence_server(vec![(404, String::new()), (404, String::new())]).await;
        let key = make_key(&base);
        let client = crate::services::http_utils::router_http_client_loopback();
        let error = fetch_models_detailed_filtered(&client, &key, false)
            .await
            .expect_err("exhausted candidates must fail");

        assert!(
            is_no_models_endpoint(&error),
            "all-404 exhaustion must carry the NoModelsEndpoint marker: {error:#}"
        );
    }

    #[tokio::test]
    async fn openai_models_mixed_4xx_is_not_no_models_endpoint() {
        let (base, _handle) =
            spawn_sequence_server(vec![(400, String::new()), (404, String::new())]).await;
        let key = make_key(&base);
        let client = crate::services::http_utils::router_http_client_loopback();
        let error = fetch_models_detailed_filtered(&client, &key, false)
            .await
            .expect_err("exhausted candidates must fail");

        assert!(
            !is_no_models_endpoint(&error),
            "a non-404 failure means the endpoint may exist: {error:#}"
        );
    }

    #[test]
    fn friendly_api_error_auth_status_maps_to_auth_exit_code() {
        use crate::errors::{ExitCode, exit_code_for_error};
        let err = friendly_api_error(reqwest::StatusCode::UNAUTHORIZED, "");
        assert_eq!(exit_code_for_error(&err), ExitCode::AuthError);
        let err = friendly_api_error(reqwest::StatusCode::NOT_FOUND, "");
        assert_eq!(exit_code_for_error(&err), ExitCode::UserError);
    }

    #[test]
    fn cursor_cache_corrupt_predicate_rejects_logged_out_marker_only() {
        // Regression: the previous heuristic ("must contain digit / hyphen /
        // slash") rejected every cursor cache entry because `auto` is a real
        // cursor model id with none of those characters. That defeated the
        // disk cache entirely — every `aivo models` and every cursor router
        // /v1/models hit refetched from cursor-agent.
        let mut cursor_key = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        cursor_key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let real_models = vec![
            "auto".to_string(),
            "composer-2.5".to_string(),
            "claude-sonnet-4-6".to_string(),
        ];
        assert!(
            !cursor_cache_looks_corrupt(&cursor_key, &real_models),
            "`auto` is a real cursor model and must not flag the cache as corrupt"
        );
        // The original sentinel that the predicate exists to catch.
        let bad_models = vec!["No".to_string()];
        assert!(cursor_cache_looks_corrupt(&cursor_key, &bad_models));
        // Non-cursor keys are unaffected.
        let other = make_key("https://api.deepseek.com");
        assert!(!cursor_cache_looks_corrupt(&other, &bad_models));
    }

    #[test]
    fn full_catalog_cache_key_collapses_cursor_into_shared_namespace() {
        // Cursor listings are chat-only, so `aivo models cursor` and the
        // picker must hit a single cache entry. Other providers keep the
        // `#all` namespace to separate the broad catalog from the chat
        // picker's filtered list.
        let mut cursor_key = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        cursor_key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        assert_eq!(
            full_catalog_cache_key_for_key(&cursor_key),
            model_cache_key_for_key(&cursor_key),
        );
        // Non-cursor keys retain the `#all` suffix so chat pickers don't
        // inherit image/audio/embedding entries from the broad fetch.
        let other = make_key("https://api.deepseek.com");
        assert_ne!(
            full_catalog_cache_key_for_key(&other),
            model_cache_key_for_key(&other),
        );
        assert!(full_catalog_cache_key_for_key(&other).ends_with("#all"));
    }

    #[tokio::test]
    async fn full_catalog_metadata_fresh_tracks_cache_ttl() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key("https://api.getaivo.dev");

        // No entry yet → stale.
        assert!(!full_catalog_metadata_fresh(&key, &cache).await);

        // Just written → fresh.
        cache
            .set(
                &full_catalog_cache_key_for_key(&key),
                vec!["aivo/starter".to_string()],
            )
            .await;
        assert!(full_catalog_metadata_fresh(&key, &cache).await);
    }

    #[tokio::test]
    async fn full_catalog_metadata_fresh_reports_expired_entry_as_stale() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        // fetched_at = 0 is past the TTL.
        let key = make_key("https://api.getaivo.dev");
        let entry = serde_json::json!({
            full_catalog_cache_key_for_key(&key): {
                "models": ["aivo/starter"],
                "fetched_at": 0u64
            }
        });
        tokio::fs::write(&path, serde_json::to_string(&entry).unwrap())
            .await
            .unwrap();
        let cache = ModelsCache::with_path(path);
        assert!(!full_catalog_metadata_fresh(&key, &cache).await);
    }

    #[test]
    fn test_is_text_chat_model_keeps_chat_models() {
        assert!(is_text_chat_model("gpt-4o"));
        assert!(is_text_chat_model("gpt-4o-mini"));
        assert!(is_text_chat_model("claude-sonnet-4-6"));
        assert!(is_text_chat_model("gpt-3.5-turbo"));
        assert!(is_text_chat_model("o1"));
        assert!(is_text_chat_model("o3-mini"));
        assert!(is_text_chat_model("gpt-4o-audio-preview"));
        assert!(is_text_chat_model("gemini-1.5-pro"));
        assert!(is_text_chat_model("gemini-2.0-flash"));
    }

    #[test]
    fn test_is_text_chat_model_filters_embeddings() {
        assert!(!is_text_chat_model("text-embedding-3-small"));
        assert!(!is_text_chat_model("text-embedding-3-large"));
        assert!(!is_text_chat_model("text-embedding-ada-002"));
        assert!(!is_text_chat_model("embedding-001"));
        assert!(!is_text_chat_model("text-embeddings-inference"));
    }

    #[test]
    fn test_is_text_chat_model_filters_image_and_audio() {
        assert!(!is_text_chat_model("dall-e-2"));
        assert!(!is_text_chat_model("dall-e-3"));
        assert!(!is_text_chat_model("tts-1"));
        assert!(!is_text_chat_model("tts-1-hd"));
        assert!(!is_text_chat_model("whisper-1"));
        assert!(!is_text_chat_model("gpt-image-1"));
        // Catalog-confirmed image-output chat models pass the `-image` name rule.
        assert!(is_text_chat_model("google/gemini-3.1-flash-image-preview"));
        assert!(is_text_chat_model("gemini-2.5-flash-image"));
    }

    #[test]
    fn test_is_copilot_chat_model_filters_codex_models() {
        assert!(is_copilot_chat_model("gpt-4o"));
        assert!(is_copilot_chat_model("claude-sonnet-4"));
        assert!(!is_copilot_chat_model("gpt-5.1-codex-mini"));
        assert!(!is_copilot_chat_model("gpt-5.3-codex"));
        assert!(!is_copilot_chat_model("openai/gpt-5.1-codex-mini"));
    }

    #[test]
    fn cloudflare_ai_base_normalizes_v1_suffix() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_accepts_ai_root() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_rejects_non_cloudflare() {
        assert_eq!(cloudflare_ai_base("https://api.openai.com/v1"), None);
    }

    #[test]
    fn cloudflare_model_name_prefers_name_over_id() {
        let model: CloudflareModel = serde_json::from_str(
            r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301","name":"@cf/meta/llama-3.1-8b-instruct"}"#,
        )
        .unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "@cf/meta/llama-3.1-8b-instruct".to_string()
        );
    }

    #[test]
    fn cloudflare_model_name_falls_back_to_id() {
        let model: CloudflareModel =
            serde_json::from_str(r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301"}"#).unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "01564c52-8717-47dc-8efd-907a2ca18301".to_string()
        );
    }

    #[test]
    fn build_anthropic_models_url_preserves_v1_path() {
        assert_eq!(
            build_anthropic_models_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            build_anthropic_models_url("https://api.minimax.io/anthropic"),
            "https://api.minimax.io/anthropic/v1/models"
        );
    }

    #[tokio::test]
    async fn cached_models_returned_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let models = vec!["model-a".to_string()];
        cache.set("https://api.example.com", models.clone()).await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With a valid cache, fetch_models_cached should return cached list
        // without making a network call (network call would fail with this fake key)
        let result = fetch_models_cached(&client, &key, &cache, false).await;
        assert_eq!(result.unwrap(), models);
    }

    #[tokio::test]
    async fn bypass_cache_ignores_warm_cache() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        // Seed cache with stale data
        cache
            .set("https://api.example.com", vec!["stale-model".to_string()])
            .await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With bypass_cache=true, the function should NOT return the cached value.
        // It will try a network call (which will fail with a fake key) — that's fine,
        // we just verify it didn't return the cached stale data.
        let result = fetch_models_cached(&client, &key, &cache, true).await;
        // Network call will fail (fake key) — result should be Err, not the stale cached value
        assert!(
            result.is_err(),
            "Expected network error, not cached stale data"
        );
    }

    // Tests for `starter_model_still_available`. Short-circuit paths (no
    // network) are exercised through the full helper; catalog-content
    // decisions are tested through the pure `model_present_in_catalog` so
    // they don't depend on HTTP fixtures.

    #[tokio::test]
    async fn starter_validation_passes_non_starter_keys_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key("https://api.example.com");
        assert!(starter_model_still_available(&key, &cache, "any-model").await);
    }

    #[tokio::test]
    async fn starter_validation_reads_cached_catalog_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);

        // Empty cache → can't judge → passes (never blocks a fresh install).
        assert!(starter_model_still_available(&key, &cache, "model-a").await);

        // The chat warm keys the catalog by the post-swap real URL; validation
        // must read it even though the key's base_url is the sentinel.
        cache
            .set(
                &full_catalog_key(crate::constants::AIVO_STARTER_REAL_URL),
                vec!["model-a".to_string()],
            )
            .await;
        assert!(starter_model_still_available(&key, &cache, "model-a").await);
        assert!(!starter_model_still_available(&key, &cache, "model-gone").await);
    }

    #[tokio::test]
    async fn starter_validation_passes_sentinel_model_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);
        // `aivo/starter` is the stable sentinel — the helper must short-circuit.
        assert!(
            starter_model_still_available(&key, &cache, crate::constants::AIVO_STARTER_MODEL,)
                .await
        );
    }

    #[tokio::test]
    async fn starter_validation_passes_default_placeholder_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);
        assert!(
            starter_model_still_available(
                &key,
                &cache,
                crate::constants::MODEL_DEFAULT_PLACEHOLDER,
            )
            .await
        );
    }

    #[test]
    fn model_present_when_catalog_contains_it() {
        let catalog = vec!["a".to_string(), "b".to_string()];
        assert!(model_present_in_catalog(&catalog, "a"));
        assert!(model_present_in_catalog(&catalog, "b"));
    }

    #[test]
    fn model_absent_when_catalog_does_not_contain_it() {
        let catalog = vec!["a".to_string(), "b".to_string()];
        assert!(!model_present_in_catalog(&catalog, "removed"));
    }

    #[test]
    fn empty_catalog_passes_through() {
        // A `200 OK` with `data: []` is almost always a transient server bug,
        // not "every model was removed". Pass through rather than flagging
        // every user's persisted model as gone.
        assert!(model_present_in_catalog(&[], "anything"));
    }

    #[test]
    fn extract_error_detail_openai_shape() {
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        assert_eq!(
            extract_error_detail(body).as_deref(),
            Some("Invalid API key")
        );
    }

    #[test]
    fn extract_error_detail_cloudflare_shape() {
        let body = r#"{"success":false,"errors":[{"code":7000,"message":"No route for the URI"}],"messages":[],"result":null}"#;
        assert_eq!(
            extract_error_detail(body).as_deref(),
            Some("No route for the URI")
        );
    }

    #[test]
    fn extract_error_detail_simple_error_string() {
        let body = r#"{"error":"unauthorized"}"#;
        assert_eq!(extract_error_detail(body).as_deref(), Some("unauthorized"));
    }

    #[test]
    fn extract_error_detail_drops_html() {
        let body = "<!DOCTYPE html><html><body><h1>404 Not Found</h1></body></html>";
        assert!(extract_error_detail(body).is_none());
        let body = "<html><head></head><body>oops</body></html>";
        assert!(extract_error_detail(body).is_none());
    }

    #[test]
    fn extract_error_detail_handles_empty() {
        assert!(extract_error_detail("").is_none());
        assert!(extract_error_detail("   \n\t  ").is_none());
    }

    #[test]
    fn extract_error_detail_truncates_plain_text() {
        let body = "x".repeat(500);
        let out = extract_error_detail(&body).unwrap();
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 201);
    }

    #[test]
    fn status_hint_known_codes() {
        assert!(
            status_hint(reqwest::StatusCode::UNAUTHORIZED)
                .unwrap()
                .contains("API key")
        );
        assert!(
            status_hint(reqwest::StatusCode::FORBIDDEN)
                .unwrap()
                .contains("permission")
        );
        assert!(
            status_hint(reqwest::StatusCode::NOT_FOUND)
                .unwrap()
                .contains("base URL")
        );
        assert!(
            status_hint(reqwest::StatusCode::TOO_MANY_REQUESTS)
                .unwrap()
                .contains("rate")
        );
        assert!(
            status_hint(reqwest::StatusCode::INTERNAL_SERVER_ERROR)
                .unwrap()
                .contains("server")
        );
        assert!(status_hint(reqwest::StatusCode::IM_A_TEAPOT).is_none());
    }

    #[test]
    fn friendly_api_error_404_leads_with_no_models_found() {
        let err = friendly_api_error(
            reqwest::StatusCode::NOT_FOUND,
            "<!doctype html><html><body>404 page</body></html>",
        );
        let s = err.to_string();
        assert!(s.starts_with("No models found"));
        assert!(s.contains("base URL may be wrong"));
        assert!(!s.contains("<html"));
        assert!(!s.contains("<body"));
        assert!(!s.contains("404"));
    }

    #[test]
    fn friendly_api_error_401_includes_extracted_message() {
        let err = friendly_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Invalid API key"}}"#,
        );
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("Invalid API key"));
        assert!(s.contains("aivo keys"));
        assert!(!s.contains("401"));
    }

    #[test]
    fn friendly_oauth_api_error_401_points_at_reauth() {
        let err = friendly_oauth_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"OAuth token expired"}}"#,
        );
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("OAuth token expired"));
        assert!(s.contains("aivo keys reauth"));
        assert!(!s.contains("check the API key"));
    }

    #[test]
    fn friendly_oauth_api_error_non_auth_status_keeps_generic_hint() {
        let err = friendly_oauth_api_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "");
        let s = err.to_string();
        assert!(s.contains("server error"));
        assert!(!s.contains("reauth"));
    }

    #[test]
    fn friendly_api_error_unknown_status_falls_back_to_status_text() {
        let err = friendly_api_error(reqwest::StatusCode::IM_A_TEAPOT, "");
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("server returned 418"));
    }

    #[test]
    fn friendly_api_error_500_uses_server_error_hint() {
        let err = friendly_api_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "");
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("server error"));
    }

    #[test]
    fn models_from_cache_round_trips_full_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "openrouter/sonnet".to_string(),
            ModelMetadata {
                context_window: Some(200_000),
                max_output: Some("32K".to_string()),
                max_output_tokens: Some(32_000),
                input_price: Some("$3".to_string()),
                output_price: Some("$15".to_string()),
                ..Default::default()
            },
        );
        metadata.insert(
            "copilot/gpt-5".to_string(),
            ModelMetadata {
                multiplier: Some(0.5),
                ..Default::default()
            },
        );

        let ids = vec![
            "openrouter/sonnet".to_string(),
            "copilot/gpt-5".to_string(),
            "id-only-model".to_string(),
        ];
        let models = models_from_cache(ids, metadata);

        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "openrouter/sonnet");
        assert_eq!(models[0].context.as_deref(), Some("200K"));
        assert_eq!(models[0].context_tokens, Some(200_000));
        assert_eq!(models[0].max_output.as_deref(), Some("32K"));
        assert_eq!(models[0].input_price.as_deref(), Some("$3"));
        assert_eq!(models[0].output_price.as_deref(), Some("$15"));

        assert_eq!(models[1].multiplier, Some(0.5));
        assert!(models[1].context.is_none());

        // Models without metadata reconstruct as id-only rows.
        assert_eq!(models[2].id, "id-only-model");
        assert!(models[2].context.is_none());
        assert!(models[2].input_price.is_none());
    }

    #[test]
    fn build_metadata_map_skips_id_only_models() {
        let models = vec![
            ModelInfo {
                id: "rich".to_string(),
                context: Some("128K".to_string()),
                context_tokens: Some(128_000),
                max_output: Some("8K".to_string()),
                max_output_tokens: Some(8_000),
                input_price: Some("$1".to_string()),
                output_price: Some("$2".to_string()),
                multiplier: None,
                deprecated: false,
                reasoning_efforts: Vec::new(),
                image_input: None,
            },
            ModelInfo::id_only("bare".to_string()),
        ];
        let map = build_metadata_map(&models);
        assert!(map.contains_key("rich"));
        assert!(!map.contains_key("bare"));
        let rich = map.get("rich").unwrap();
        assert_eq!(rich.context_window, Some(128_000));
        assert_eq!(rich.max_output.as_deref(), Some("8K"));
    }

    #[test]
    fn harvests_image_input_boolean_from_catalog() {
        let resp: OpenAIModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {"id": "aivo/starter", "image_input": false},
                {"id": "vision", "image_input": true},
                {"id": "mystery"}
            ]
        }))
        .unwrap();
        let infos: Vec<ModelInfo> = resp
            .data
            .into_iter()
            .map(|m| m.into_model_info(1.0))
            .collect();
        assert_eq!(infos[0].image_input, Some(false));
        assert_eq!(infos[1].image_input, Some(true));
        assert_eq!(infos[2].image_input, None);

        let map = build_metadata_map(&infos);
        assert_eq!(
            map.get("aivo/starter").and_then(|m| m.image_input),
            Some(false)
        );
        assert_eq!(map.get("vision").and_then(|m| m.image_input), Some(true));
        assert!(
            !map.contains_key("mystery"),
            "id-only rows still skip the metadata map"
        );
    }

    #[test]
    fn cursor_cache_key_separates_shadow_accounts_and_api_keys() {
        let mut shadow_a = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        shadow_a.key = zeroize::Zeroizing::new(format!(
            "{}aaaa1111",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let mut shadow_b = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        shadow_b.key = zeroize::Zeroizing::new(format!(
            "{}bbbb2222",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let a = model_cache_key_for_key(&shadow_a);
        let b = model_cache_key_for_key(&shadow_b);
        assert!(a.starts_with("cursor#shadow-"));
        assert_ne!(a, b);

        let mut api = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        api.key = zeroize::Zeroizing::new("sk-cursor".to_string());
        let cache_key = model_cache_key_for_key(&api);
        assert!(cache_key.starts_with("cursor#"));
        assert!(!cache_key.starts_with("cursor#shadow-"));
        assert!(!cache_key.contains("sk-cursor"));
    }

    fn priced_models(prices: &[(f64, f64)]) -> Vec<OpenAIModel> {
        prices
            .iter()
            .enumerate()
            .map(|(i, (input, output))| {
                serde_json::from_value(serde_json::json!({
                    "id": format!("m{i}"),
                    "pricing": {"prompt": input, "completion": output},
                }))
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn price_scale_per_token_openrouter_style() {
        let models = priced_models(&[(0.0000005, 0.0000015), (0.000003, 0.000015)]);
        let scale = infer_price_scale(&models);
        assert_eq!(scale, 1_000_000.0);
        assert_eq!(
            format_scaled_price(0.000003, scale).as_deref(),
            Some("$3.00")
        );
        // A premium model ($200/1M) rides the response-wide vote.
        let models = priced_models(&[(0.000003, 0.000015), (0.00004, 0.0002)]);
        assert_eq!(infer_price_scale(&models), 1_000_000.0);
    }

    #[test]
    fn price_scale_per_thousand_relay_style() {
        // 0.0003/0.0012 straddled the old >=0.001 threshold ($300.00/$0.00).
        let models = priced_models(&[
            (0.00014, 0.00028),
            (0.0003, 0.0012),
            (0.0006, 0.003),
            (0.0013, 0.0043),
        ]);
        let scale = infer_price_scale(&models);
        assert_eq!(scale, 1_000.0);
        assert_eq!(
            format_scaled_price(0.00014, scale).as_deref(),
            Some("$0.14")
        );
        assert_eq!(format_scaled_price(0.0012, scale).as_deref(), Some("$1.20"));
        assert_eq!(format_scaled_price(0.0043, scale).as_deref(), Some("$4.30"));
    }

    #[test]
    fn price_scale_per_million_publicai_style() {
        let models = priced_models(&[(0.15, 0.2), (2.92, 200.0)]);
        let scale = infer_price_scale(&models);
        assert_eq!(scale, 1.0);
        assert_eq!(format_scaled_price(0.15, scale).as_deref(), Some("$0.15"));
        assert_eq!(
            format_scaled_price(200.0, scale).as_deref(),
            Some("$200.00")
        );
    }

    #[test]
    fn price_scale_defaults_without_votes() {
        assert_eq!(infer_price_scale(&[]), 1.0);
        assert_eq!(infer_price_scale(&priced_models(&[(0.0, 0.0)])), 1.0);
    }

    #[test]
    fn format_price_rejects_zero_and_negative() {
        assert_eq!(format_scaled_price(0.0, 1.0), None);
        assert_eq!(format_scaled_price(-1.0, 1_000_000.0), None);
    }

    #[test]
    fn zero_context_and_max_tokens_treated_as_absent() {
        // Vercel reports 0 for video/embedding models; "0/0" is meaningless.
        let m: OpenAIModel = serde_json::from_value(serde_json::json!({
            "id": "wan-t2v",
            "context_window": 0,
            "max_tokens": 0,
        }))
        .unwrap();
        assert_eq!(m.find_context(), None);
        assert_eq!(m.find_max_output(), None);
    }

    #[test]
    fn raw_pricing_parses_string_values() {
        let m: OpenAIModel = serde_json::from_value(serde_json::json!({
            "id": "m",
            "pricing": {"prompt": "0.000003", "completion": "free"},
        }))
        .unwrap();
        assert_eq!(m.raw_pricing(), (Some(0.000003), None));
    }
}
