//! ModelsCommand handler for listing available models from the active provider,
//! plus the interactive model pickers. The fetch/cache core lives in
//! `services::model_catalog`; this module owns rendering and prompting only.
use anyhow::Result;
use reqwest::Client;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::errors::ExitCode;
use crate::services::ai_launcher::AIToolType;
use crate::services::http_utils;
use crate::services::model_catalog::{
    ModelInfo, build_metadata_map, cursor_cache_looks_corrupt, enrich_from_snapshot,
    fetch_all_models_cached, fetch_models_cached, fetch_models_detailed_filtered,
    full_catalog_cache_key_for_key, full_catalog_cached, kimi_model_infos, models_from_cache,
};
use crate::services::models_cache::ModelsCache;
use crate::services::provider_profile::is_aivo_starter_base;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ModelsCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl ModelsCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub async fn execute(
        &self,
        key_override: Option<ApiKey>,
        refresh: bool,
        search: Option<String>,
        json: bool,
    ) -> ExitCode {
        match self
            .execute_internal(key_override, refresh, search, json)
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{}", style::yellow(format!("{e:#}")));
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(
        &self,
        key_override: Option<ApiKey>,
        refresh: bool,
        search: Option<String>,
        json: bool,
    ) -> Result<ExitCode> {
        let mut key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key_info().await? {
                Some(k) => k,
                None => {
                    crate::commands::print_no_key_error();
                    return Ok(ExitCode::AuthError);
                }
            },
        };

        // Grok/Codex/Claude tokens can enumerate models; other OAuth can't.
        if key.is_any_oauth() && !key.is_provider_oauth() && !key.is_claude_oauth() {
            eprintln!(
                "{} Key '{}' is an OAuth credential — it doesn't have a model listing API.",
                style::red("Error:"),
                key.display_name()
            );
            eprintln!(
                "  {} Use `{}` to launch the tool directly, or switch to a regular API key with `aivo use`.",
                style::dim("hint:"),
                key.oauth_tool_hint()
            );
            return Ok(ExitCode::UserError);
        }

        if key.is_cursor_acp() {
            SessionStore::decrypt_key_secret(&mut key)?;
        }

        let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
        let all_cache_key = full_catalog_cache_key_for_key(&key);

        // `aivo models` shows the provider's full catalog, including image,
        // audio, and embedding models. Chat pickers filter/annotate at their
        // own call sites. Serve from cache whenever the entry is fresh — the
        // TTL handles staleness, and id-only providers (minimax, cloudflare)
        // round-trip as id-only rows just like a fresh fetch would.
        let cached_entry = if refresh || is_ollama {
            None
        } else {
            self.cache
                .get_with_metadata(&all_cache_key)
                .await
                .filter(|(ids, _)| !cursor_cache_looks_corrupt(&key, ids))
        };

        let mut models = if key.is_grok_oauth() {
            // Grok lists models via the CLI proxy (token-authed); id-only here,
            // enrichment below fills limits.
            SessionStore::decrypt_key_secret(&mut key)?;
            let mut creds = crate::services::grok_oauth::GrokOAuthCredential::from_json(&key.key)?;
            let ids =
                crate::services::grok_oauth::fetch_model_ids(&mut creds, Some(&self.session_store))
                    .await?;
            ids.into_iter().map(ModelInfo::id_only).collect()
        } else if key.is_kimi_oauth() {
            SessionStore::decrypt_key_secret(&mut key)?;
            let mut creds = crate::services::kimi_oauth::KimiOAuthCredential::from_json(&key.key)?;
            let models =
                crate::services::kimi_oauth::fetch_models(&mut creds, Some(&self.session_store))
                    .await?;
            kimi_model_infos(models)
        } else if key.is_codex_oauth() {
            crate::services::codex_oauth::known_model_ids()
                .into_iter()
                .map(ModelInfo::id_only)
                .collect()
        } else if let Some((ids, meta)) = cached_entry {
            models_from_cache(ids, meta)
        } else {
            SessionStore::decrypt_key_secret(&mut key)?;
            let client = http_utils::router_http_client();
            let started_at = Instant::now();
            let (spinning, spinner_handle) = style::start_spinner(Some(" Fetching models..."));
            let result = fetch_models_detailed_filtered(&client, &key, false).await;
            let min_visible = Duration::from_millis(350);
            if let Some(remaining) = min_visible.checked_sub(started_at.elapsed()) {
                tokio::time::sleep(remaining).await;
            }
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            let fresh = result?;

            // Cache the full list (including image/audio/embed) under the `#all`
            // namespace so `aivo image` and future broad pickers can share it.
            // Persist every column the table prints so the next call can be
            // satisfied without a network roundtrip.
            if !is_ollama {
                let ids: Vec<String> = fresh.iter().map(|m| m.id.clone()).collect();
                let metadata = build_metadata_map(&fresh);
                self.cache
                    .set_with_metadata(&all_cache_key, ids, metadata)
                    .await;
            }
            fresh
        };

        // Fill missing limit columns from the embedded models.dev snapshot so
        // id-only providers (starter, minimax, cloudflare) still show
        // context/output. Display-only: the cache write above stores what the
        // provider actually returned.
        for model in &mut models {
            enrich_from_snapshot(model);
        }

        let is_starter = is_aivo_starter_base(&key.base_url);
        // First-party key shows the account plan label, not the raw `aivo-starter` sentinel.
        let starter_account = is_starter
            .then(crate::services::account_store::load)
            .flatten();
        models.sort_by(|a, b| {
            if is_starter {
                let a_s = a.id.ends_with("/starter");
                let b_s = b.id.ends_with("/starter");
                if a_s != b_s {
                    return if a_s {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    };
                }
            }
            a.id.cmp(&b.id)
        });

        let searching = if let Some(ref query) = search {
            let q = query.trim().to_lowercase();
            if q.is_empty() {
                anyhow::bail!("Search query cannot be empty");
            }
            models.retain(|m| m.id.to_lowercase().contains(&q));
            true
        } else {
            false
        };

        let label = if searching { "matches" } else { "models" };
        let provider_cell = if is_starter {
            let plan_label = crate::commands::starter_provider_label(
                starter_account.as_ref().and_then(|a| a.plan.as_deref()),
                starter_account
                    .as_ref()
                    .and_then(|a| a.plan_label.as_deref()),
            );
            style::dim(&plan_label)
        } else {
            style::dim(&key.base_url)
        };
        eprintln!(
            "{} {} {} via {}",
            style::success_symbol(),
            models.len(),
            label,
            provider_cell
        );

        if json {
            let mut payload = serde_json::json!({
                "provider": key.base_url,
                "models": models,
            });
            if let Some(plan) = starter_account.as_ref().and_then(|a| a.plan.as_deref()) {
                payload["plan"] = serde_json::json!(plan);
            }
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let widths = ColumnWidths::from_models(&models);
            // Stdout's LineWriter flushes on every '\n', so a per-row println!
            // becomes one syscall per row. Batch into a single write_all.
            let mut buf = String::with_capacity(models.len() * 80);
            for model in &models {
                buf.push_str(&format_model_line(model, &widths));
                buf.push('\n');
            }
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            let _ = handle.write_all(buf.as_bytes());
        }

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!(
            "{} aivo models [key::][search] [options]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("List available models from the active API key's provider.")
        );
        println!(
            "{}",
            style::dim("A `key::` prefix selects the key; a search term filters by substring.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-k, --key <id|name>", "Select API key by ID or name");
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("-s, --search <query>", "Filter models by substring match");
        print_opt("--json", "Output model list as JSON instead of a table");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo models"));
        println!("  {}", style::dim("aivo models -s sonnet"));
        println!("  {}", style::dim("aivo models openrouter::"));
        println!("  {}", style::dim("aivo models openrouter::glm"));
        println!("  {}", style::dim("aivo models --json | jq '.models[].id'"));
    }
}

#[derive(Default)]
struct ColumnWidths {
    name: usize,
    context: usize,
    max_output: usize,
}

impl ColumnWidths {
    fn from_models(models: &[ModelInfo]) -> Self {
        let mut w = Self::default();
        for m in models {
            w.name = w.name.max(m.id.len());
            if let Some(c) = &m.context {
                w.context = w.context.max(c.len());
            }
            if let Some(o) = &m.max_output {
                w.max_output = w.max_output.max(o.len());
            }
        }
        w
    }
}

fn format_model_line(model: &ModelInfo, widths: &ColumnWidths) -> String {
    let has_price = model.input_price.is_some() && model.output_price.is_some();
    let has_meta = model.context.is_some() || model.max_output.is_some() || has_price;
    if !has_meta && model.multiplier.is_none() && !model.deprecated {
        return model.id.clone();
    }

    let mut line = format!("{:<width$}", model.id, width = widths.name);
    let cw = widths.context.max(1);
    let ow = widths.max_output.max(1);

    // Compact ctx/out cell, e.g. "128K/16K"; a missing half is "-", a priced row
    // with no context is "-/-" so the price column still lines up.
    if has_meta {
        let ctx = model.context.as_deref().unwrap_or("-");
        let out = model.max_output.as_deref().unwrap_or("-");
        let cell = format!("{ctx}/{out}");
        // Pad to full width only when something follows, to avoid trailing space.
        let cell = if has_price || model.multiplier.is_some() || model.deprecated {
            format!("{cell:<width$}", width = cw + 1 + ow)
        } else {
            cell
        };
        line.push_str(&format!(" {}", style::dim(cell)));
    }

    // Left-aligned "$in/$out" per 1M — right-alignment leaves wide gaps ("$1.2/$6").
    if let (Some(input), Some(output)) = (&model.input_price, &model.output_price) {
        line.push_str(&format!(" {}", style::dim(format!("{input}/{output}"))));
    }
    if let Some(mult) = model.multiplier {
        line.push_str(&format!(" {}", style::dim(format_multiplier(mult))));
    }
    if model.deprecated {
        line.push_str(&format!(" {}", style::dim("deprecated")));
    }
    line
}

/// Build a compact picker label (id plus `Nx` suffix when a multiplier is known).
pub(crate) fn picker_label(model: &ModelInfo) -> String {
    match model.multiplier {
        Some(mult) => format!("{}  {}", model.id, format_multiplier(mult)),
        None => model.id.clone(),
    }
}

/// Format a Copilot premium-request multiplier as e.g. `1x`, `0.33x`, `7.5x`.
fn format_multiplier(m: f64) -> String {
    if m == m.trunc() {
        format!("{}x", m as i64)
    } else {
        let s = format!("{:.2}", m);
        format!("{}x", s.trim_end_matches('0').trim_end_matches('.'))
    }
}

/// Cache-first catalog fetch for the interactive model picker, shared by the
/// `run`/`start`/`chat` pickers. Returns instantly on a cache hit; shows a
/// "Fetching models…" spinner only while a genuine network fetch runs — a hit
/// must stay instant, since `stop_spinner` sleeps 100ms and would flash a frame.
/// Fetch errors are returned so explicit picker requests can show the actual
/// provider error instead of reducing it to an empty model list.
pub(crate) async fn fetch_all_models_for_picker(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    refresh: bool,
) -> anyhow::Result<Vec<String>> {
    if !refresh && let Some(cached) = full_catalog_cached(key, cache).await {
        return Ok(cached);
    }
    let (spinning, handle) = style::start_spinner(Some(" Fetching models..."));
    let result = fetch_all_models_cached(client, key, cache, true).await;
    style::stop_spinner(&spinning);
    let _ = handle.await;
    result
}

/// Fetches the model list (cache-first) with a spinner for network fetches,
/// filtered to text-chat models only. Used by chat and run commands for the
/// interactive model picker.
pub(crate) async fn fetch_models_for_select(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
) -> Vec<String> {
    fetch_models_cached(client, key, cache, false)
        .await
        .unwrap_or_default()
}

/// Determines whether the "(leave it to the tool)" default option should be
/// shown in the model picker for the given tool type and model list.
///
/// - Pi, Opencode, Grok: never (these tools require explicit model selection;
///   grok's built-in default model 400s on non-xAI keys)
/// - Claude: only if the list contains Claude-family models
/// - Codex: only if the list contains gpt- prefixed models
/// - Gemini: only if the list contains Gemini-family models
pub(crate) fn tool_supports_default_model(tool: AIToolType, models: &[String]) -> bool {
    match tool {
        AIToolType::Pi | AIToolType::Opencode | AIToolType::Grok => false,
        AIToolType::Claude => models
            .iter()
            .any(|m| m.to_ascii_lowercase().contains("claude")),
        AIToolType::Codex | AIToolType::CodexApp => models
            .iter()
            .any(|m| m.to_ascii_lowercase().starts_with("gpt-")),
        AIToolType::Gemini => models
            .iter()
            .any(|m| m.to_ascii_lowercase().contains("gemini")),
    }
}

/// Folds the positional (`key::`, `key::search`, bare search term) into the
/// `-k`/`-s` slots. Same policy as the run path's `-k` vs `-m key::…`: a half
/// given twice with agreeing values collapses; differing values are an error,
/// not a silent drop.
pub(crate) fn merge_models_spec(
    spec: Option<String>,
    key: Option<String>,
    search: Option<String>,
) -> Result<(Option<String>, Option<String>), String> {
    let Some(spec) = spec else {
        return Ok((key, search));
    };
    let (spec_key, model) = crate::cli_args::split_tier_spec(&spec);
    let model = model.trim();
    let spec_search = (!model.is_empty()).then(|| model.to_string());
    if let (Some(k), Some(f)) = (&spec_key, &key)
        && k != f
    {
        return Err(format!(
            "-k '{f}' conflicts with the key in '{k}::…' — pick the key once."
        ));
    }
    if let (Some(q), Some(f)) = (&spec_search, &search)
        && q != f
    {
        return Err(format!(
            "-s '{f}' conflicts with the positional search '{q}' — pick the search once."
        ));
    }
    Ok((spec_key.or(key), spec_search.or(search)))
}

/// Shows an interactive model picker. The "(leave it to the tool)" default option
/// is conditionally shown based on whether the provider has models compatible with
/// the selected tool. When `tool` is `None` (e.g. chat mode), the default option
/// is hidden since a concrete model is required.
/// Returns `Some(MODEL_DEFAULT_PLACEHOLDER)` if the default is chosen,
/// `Some(model_name)` for a real model, or `None` if cancelled.
/// Outcome of picker-style model resolution. Distinguishes "user cancelled the
/// picker" (exit success, don't launch) from "no fetchable model list, fall back
/// to the tool's own default" (launch anyway, no injected model). Shared by
/// `aivo run` and the plugin endpoint so both resolve models identically.
/// `pub`: fat plugins (aivo-amp) link this for their own bare-flag pickers.
pub enum ModelOutcome {
    /// User picked a model, or `--model <value>` was passed.
    Model(String),
    /// No `--model` flag, or the picker fetched an empty list. Launch with the
    /// tool's own default.
    UseDefault,
    /// Picker shown and cancelled (Ctrl-C / Esc) — caller should not launch.
    Cancelled,
}

/// Resolve the model to launch with. `--model <value>` → use as-is; bare
/// `--model` (`Some("")`) → fuzzy picker; **no `--model` (`None`) → `UseDefault`**
/// (let the tool use its own default — never a forced picker, so a bare launch
/// or a `-k`-only launch doesn't pop a dialog). On a non-TTY or an empty catalog
/// it also falls back to the default. `tool` colors the picker's "(leave it to
/// the tool)" row; pass `None` for a generic client (plugins).
#[allow(clippy::too_many_arguments)]
pub async fn resolve_model_outcome(
    client: &Client,
    key: &ApiKey,
    flag_model: Option<String>,
    explicit_model_flag: bool,
    refresh: bool,
    tool: Option<AIToolType>,
    cache: &ModelsCache,
    prompt: &str,
) -> Result<ModelOutcome> {
    match flag_model {
        None => return Ok(ModelOutcome::UseDefault),
        Some(ref m) if !m.is_empty() => return Ok(ModelOutcome::Model(m.clone())),
        Some(_) => {}
    }

    // The picker raw-modes stdin and renders to stderr; without both TTYs it
    // can't run (a piped stdin reads as a cancel). Bail before the network fetch
    // so piped invocations don't pay for a catalog they can't show. Only explain
    // when the user explicitly asked for a picker.
    if !crate::tui::picker_interactive() {
        if explicit_model_flag {
            crate::commands::print_no_model_list_hint();
        }
        return Ok(ModelOutcome::UseDefault);
    }

    let models_list = match fetch_all_models_for_picker(client, key, cache, refresh).await {
        Ok(models) => models,
        Err(e)
            if explicit_model_flag
                && !crate::services::model_catalog::is_no_models_endpoint(&e) =>
        {
            return Err(e);
        }
        Err(_) => Vec::new(),
    };
    if models_list.is_empty() {
        if explicit_model_flag {
            crate::commands::print_no_model_list_hint();
        }
        return Ok(ModelOutcome::UseDefault);
    }

    let annotations = crate::services::model_compat::text_chat_annotations(&models_list);
    match prompt_model_picker(models_list, tool, annotations, prompt) {
        Some(m) => Ok(ModelOutcome::Model(m)),
        None => Ok(ModelOutcome::Cancelled),
    }
}

/// Shows a fuzzy model picker with per-row annotations. `annotations[i] =
/// Some(reason)` disables the matching model and renders `reason` dim at the
/// end of the row (parallels the key picker). Pass `vec![]` for a vanilla
/// picker with every row selectable.
pub(crate) fn prompt_model_picker(
    models: Vec<String>,
    tool: Option<AIToolType>,
    annotations: Vec<Option<String>>,
    prompt: &str,
) -> Option<String> {
    use crate::constants;
    use crate::tui::FuzzySelect;

    let show_default = tool
        .map(|t| tool_supports_default_model(t, &models))
        .unwrap_or(false);

    let mut items = Vec::with_capacity(models.len() + show_default as usize);
    let mut row_annotations: Vec<Option<String>> = Vec::with_capacity(items.capacity());
    if show_default {
        items.push(constants::MODEL_DEFAULT_DISPLAY.to_string());
        row_annotations.push(None);
    }
    items.extend(models);
    // Align annotations with the model rows (after the optional default row).
    // A shorter/empty `annotations` means "no annotations" for those rows.
    for i in 0..items.len() - row_annotations.len() {
        row_annotations.push(annotations.get(i).cloned().flatten());
    }

    let selected = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&items)
        .annotations(row_annotations)
        .default(0)
        .interact_opt()
        .ok()
        .flatten()?;

    if show_default && selected == 0 {
        Some(constants::MODEL_DEFAULT_PLACEHOLDER.to_string())
    } else {
        Some(items[selected].clone())
    }
}

/// Converts the `__default__` placeholder to `None` for passing to tools.
pub(crate) fn resolve_model_placeholder(model: Option<String>) -> Option<String> {
    match model.as_deref() {
        Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER) => None,
        _ => model,
    }
}

/// Returns a display-friendly model string, converting `__default__` and `None` to "(tool default)".
pub(crate) fn model_display_label(model: Option<&str>) -> &str {
    match model {
        Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER) | None => "(tool default)",
        Some(m) => m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_models_spec_folds_positional_into_flags() {
        let s = |v: &str| Some(v.to_string());
        assert_eq!(
            merge_models_spec(None, s("gapnet"), s("glm")),
            Ok((s("gapnet"), s("glm")))
        );
        assert_eq!(
            merge_models_spec(s("gapnet::"), None, None),
            Ok((s("gapnet"), None))
        );
        assert_eq!(
            merge_models_spec(s("gapnet::glm"), None, None),
            Ok((s("gapnet"), s("glm")))
        );
        assert_eq!(
            merge_models_spec(s("sonnet"), None, None),
            Ok((None, s("sonnet")))
        );
        assert_eq!(
            merge_models_spec(s("gapnet::"), None, s("glm")),
            Ok((s("gapnet"), s("glm")))
        );
        assert_eq!(
            merge_models_spec(s("glm"), s("gapnet"), None),
            Ok((s("gapnet"), s("glm")))
        );
        assert_eq!(merge_models_spec(s("::"), None, None), Ok((None, None)));
    }

    #[test]
    fn merge_models_spec_doubled_halves_collapse_or_error() {
        let s = |v: &str| Some(v.to_string());
        // Agreeing values collapse (run-path policy for -k vs -m key::…).
        assert_eq!(
            merge_models_spec(s("gapnet::"), s("gapnet"), None),
            Ok((s("gapnet"), None))
        );
        assert_eq!(
            merge_models_spec(s("glm"), None, s("glm")),
            Ok((None, s("glm")))
        );
        assert!(merge_models_spec(s("gapnet::"), s("other"), None).is_err());
        // Bare `-k` ("" = picker request) is a differing value, not absence.
        assert!(merge_models_spec(s("gapnet::"), s(""), None).is_err());
        assert!(merge_models_spec(s("gapnet::glm"), None, s("kimi")).is_err());
        assert!(merge_models_spec(s("glm"), None, s("kimi")).is_err());
    }

    #[test]
    fn test_tool_supports_default_pi_always_false() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Pi, &models));
    }

    #[test]
    fn test_tool_supports_default_grok_always_false() {
        let models = vec!["grok-4".to_string()];
        assert!(!tool_supports_default_model(AIToolType::Grok, &models));
    }

    #[test]
    fn test_tool_supports_default_opencode_always_false() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Opencode, &models));
    }

    #[test]
    fn test_tool_supports_default_claude_with_claude_models() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(tool_supports_default_model(AIToolType::Claude, &models));
    }

    #[test]
    fn test_tool_supports_default_claude_without_claude_models() {
        let models = vec!["gpt-4o".into(), "gemini-2.5-pro".into()];
        assert!(!tool_supports_default_model(AIToolType::Claude, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_with_gpt_models() {
        let models = vec!["gpt-4o".into(), "claude-sonnet-4-6".into()];
        assert!(tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_only_matches_gpt_prefix() {
        // o-series and chatgpt should NOT match for Codex
        let models = vec!["o3-mini".into(), "o4-preview".into(), "chatgpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_without_gpt_models() {
        let models = vec!["claude-sonnet-4-6".into(), "gemini-2.5-pro".into()];
        assert!(!tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_gemini_with_gemini_models() {
        let models = vec!["gemini-2.5-pro".into(), "gpt-4o".into()];
        assert!(tool_supports_default_model(AIToolType::Gemini, &models));
    }

    #[test]
    fn test_tool_supports_default_gemini_without_gemini_models() {
        let models = vec!["gpt-4o".into(), "claude-sonnet-4-6".into()];
        assert!(!tool_supports_default_model(AIToolType::Gemini, &models));
    }

    #[test]
    fn test_tool_supports_default_empty_model_list() {
        assert!(!tool_supports_default_model(AIToolType::Claude, &[]));
        assert!(!tool_supports_default_model(AIToolType::Codex, &[]));
        assert!(!tool_supports_default_model(AIToolType::Gemini, &[]));
        assert!(!tool_supports_default_model(AIToolType::Pi, &[]));
        assert!(!tool_supports_default_model(AIToolType::Opencode, &[]));
    }

    #[test]
    fn format_model_line_shows_price_without_context() {
        // A price-only model (no ctx/max_output) still renders its price.
        let mut m = ModelInfo::id_only("allenai/Olmo-3.1-32B-Think".to_string());
        m.input_price = Some("$0.05".to_string());
        m.output_price = Some("$0.20".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&m));
        let line = format_model_line(&m, &widths);
        assert!(line.contains("$0.05/$0.20"), "price missing: {line:?}");
    }

    #[test]
    fn format_model_line_bare_id_when_no_info() {
        let m = ModelInfo::id_only("some/model".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&m));
        assert_eq!(format_model_line(&m, &widths), "some/model");
    }

    #[test]
    fn format_model_line_dashes_missing_meta_values() {
        // A missing context/max-output half renders as "-" in the compact cell.
        let mut ctx_only = ModelInfo::id_only("m".to_string());
        ctx_only.context = Some("512".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&ctx_only));
        let line = console::strip_ansi_codes(&format_model_line(&ctx_only, &widths)).into_owned();
        assert!(line.contains("512/-"), "expected '512/-', got {line:?}");

        // A priced row with no context shows "-/-" so the price column aligns.
        let mut price_only = ModelInfo::id_only("m".to_string());
        price_only.input_price = Some("$0.10".to_string());
        price_only.output_price = Some("$0.20".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&price_only));
        let line = console::strip_ansi_codes(&format_model_line(&price_only, &widths)).into_owned();
        assert!(line.contains("-/-"), "expected '-/-', got {line:?}");
        assert!(line.contains("$0.10/$0.20"), "price missing: {line:?}");
    }

    #[test]
    fn format_model_line_price_column_aligns_with_and_without_context() {
        // The price column starts at the same offset with or without a context.
        let mut with_ctx = ModelInfo::id_only("aaaa".to_string());
        with_ctx.context = Some("128K".to_string());
        with_ctx.max_output = Some("8K".to_string());
        with_ctx.input_price = Some("$0.15".to_string());
        with_ctx.output_price = Some("$0.20".to_string());
        let mut no_ctx = ModelInfo::id_only("b".to_string());
        no_ctx.input_price = Some("$0.05".to_string());
        no_ctx.output_price = Some("$0.20".to_string());

        let models = [with_ctx, no_ctx];
        let widths = ColumnWidths::from_models(&models);
        let l0 = console::strip_ansi_codes(&format_model_line(&models[0], &widths)).into_owned();
        let l1 = console::strip_ansi_codes(&format_model_line(&models[1], &widths)).into_owned();

        let col = |line: &str, needle: &str| line.find(needle).map(|b| line[..b].chars().count());
        assert_eq!(
            col(&l0, "0.15"),
            col(&l1, "0.05"),
            "price columns misaligned:\n{l0:?}\n{l1:?}"
        );
    }
}
