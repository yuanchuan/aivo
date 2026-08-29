//! CodeCommand handler. Interactive sessions launch the full-screen TUI
//! (code_tui). One-shot queries (-x flag) stream directly to stdout using
//! OpenAI-compatible /v1/chat/completions, falling back through the shared
//! protocol router when the upstream rejects that wire format.
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::constants::CONTENT_TYPE_JSON;
use anyhow::{Context, Result};
use chrono::Utc;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INITIATOR_HEADER, COPILOT_INTEGRATION_ID,
    COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::cursor_acp::{self, CursorChunk};
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils::copilot_initiator_from_openai;
use crate::services::http_utils::{SseLineBuffer, sse_data_payload};
use crate::services::huggingface;
use crate::services::model_names;
use crate::services::models_cache::ModelsCache;
use crate::services::provider_profile::{provider_profile_for_key, resolve_starter_base_url};
use crate::services::provider_protocol::{PathVariant, ProviderProtocol};
use crate::services::responses_to_chat_router::{
    ResponsesToChatRouterConfig, forward_chat_completions_with_fallback,
};
use crate::services::route_cache::{PersistedRoute, canonical_model};
use crate::services::session_store::{
    ApiKey, AttachmentStorage, MessageAttachment, SessionStore, SessionTokens, StoredChatMessage,
};
use crate::services::stdin_io::read_stdin_if_piped;
use crate::style;

use super::code_request_builder::{
    build_anthropic_request, build_google_request, build_openai_chat_request,
    build_responses_request,
};
use super::code_response_parser::{
    ChatResponseChunk, ChatTurnResult, capture_model, extract_anthropic_usage,
    extract_google_message, extract_google_model, extract_google_usage, extract_openai_message,
    extract_openai_usage, extract_response_model, extract_responses_message,
    extract_responses_usage, merge_token_usage, normalize_reasoning_content, parse_anthropic_chunk,
    parse_anthropic_model_chunk, parse_anthropic_usage_chunk, parse_google_chunk,
    parse_google_usage_chunk, parse_openai_model_chunk, parse_openai_usage_chunk,
    parse_responses_chunk, parse_responses_model_chunk, parse_responses_usage_chunk,
    parse_sse_chunk,
};

// Re-export for submodules (code_tui_format uses TokenUsage)
pub(crate) use super::code_response_parser::TokenUsage;
pub(crate) use code_tui_format::format_time_ago_short;

#[path = "code_tui.rs"]
pub(crate) mod code_tui;
// `code_tui_format` is declared at the parent (`commands/mod.rs`) so other
// commands (notably `aivo logs` / `--resume`) can reuse its time/text
// formatters. Re-exported here so the chat module references it without
// `super::`.
use super::code_agent_oneshot;
use super::code_tui_format;

/// Max entries in the composer's per-directory recall view (the up-arrow list
/// for the current launch dir). Recall is cwd-filtered like Claude Code's
/// `project` history and Grok's per-cwd history.
const MAX_DRAFT_HISTORY: usize = 100;
/// Max entries across ALL dirs in the global on-disk file; oldest dropped first.
const MAX_DRAFT_HISTORY_TOTAL: usize = 2000;
/// Retry budget for transient HTTP failures.
const MAX_REQUEST_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing, skip_deserializing)]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<MessageAttachment>,
    /// Producing model (assistant turns only); `None` on pre-feature sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Which API format the provider speaks
#[derive(Debug, Clone, PartialEq)]
enum ChatFormat {
    /// OpenAI-compatible: POST /v1/chat/completions
    OpenAI,
    /// Anthropic native: POST /v1/messages
    Anthropic,
    /// OpenAI Responses API: POST /v1/responses
    Responses,
    /// Google Gemini native: POST /models/{model}:streamGenerateContent
    Google,
}

/// Lazily create (or reuse) a per-process sandbox directory under
/// `$TMPDIR/aivo-chat-<pid>/`. Returned as an absolute path. Empty
/// directory; backends that auto-index the workspace see nothing.
fn chat_sandbox_dir() -> Result<std::path::PathBuf> {
    use std::sync::OnceLock;
    static SANDBOX: OnceLock<std::path::PathBuf> = OnceLock::new();
    if let Some(path) = SANDBOX.get() {
        return Ok(path.clone());
    }
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("aivo-chat-{pid}"));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("create chat sandbox dir at {}", path.display()))?;
    let _ = SANDBOX.set(path.clone());
    Ok(path)
}

/// Human-readable endpoint for the `--dry-run` preview: the sentinel base URLs
/// (`copilot`, `ollama`, the aivo-starter sentinel) are resolved to a friendly
/// label / real URL so the preview doesn't leak internal placeholders.
fn friendly_base_url(base_url: &str) -> String {
    match base_url {
        "copilot" => "GitHub Copilot".to_string(),
        "ollama" => "Ollama (local)".to_string(),
        u if u == crate::constants::AIVO_STARTER_SENTINEL => {
            crate::constants::AIVO_STARTER_REAL_URL.to_string()
        }
        u => u.to_string(),
    }
}

fn detect_initial_chat_format(base_url: &str) -> ChatFormat {
    use crate::services::provider_protocol::detect_provider_protocol;
    match detect_provider_protocol(base_url) {
        ProviderProtocol::Google => ChatFormat::Google,
        ProviderProtocol::Anthropic => ChatFormat::Anthropic,
        _ => ChatFormat::OpenAI,
    }
}

/// Lossless `ChatFormat` → protocol for the route store. Unlike
/// `chat_format_for_router_protocol`, `ResponsesApi` stays distinct.
fn chat_format_protocol(format: &ChatFormat) -> ProviderProtocol {
    match format {
        ChatFormat::Anthropic => ProviderProtocol::Anthropic,
        ChatFormat::Google => ProviderProtocol::Google,
        ChatFormat::Responses => ProviderProtocol::ResponsesApi,
        ChatFormat::OpenAI => ProviderProtocol::Openai,
    }
}

fn chat_format_from_protocol(protocol: ProviderProtocol) -> ChatFormat {
    match protocol {
        ProviderProtocol::Anthropic => ChatFormat::Anthropic,
        ProviderProtocol::Google => ChatFormat::Google,
        ProviderProtocol::ResponsesApi => ChatFormat::Responses,
        ProviderProtocol::Openai => ChatFormat::OpenAI,
    }
}

/// Seed the turn's wire format from a persisted `(chat, key, model)` route,
/// else URL detection — mirrors how launch routers seed their `RouteCache`.
fn seeded_chat_format(key: &ApiKey, raw_model: &str) -> ChatFormat {
    match persisted_chat_protocol(key, raw_model) {
        Some(protocol) => chat_format_from_protocol(protocol),
        None => detect_initial_chat_format(&key.base_url),
    }
}

fn persisted_chat_protocol(key: &ApiKey, raw_model: &str) -> Option<ProviderProtocol> {
    if raw_model.trim().is_empty() {
        return None;
    }
    let routes = key.routes_for_tool("code");
    let model = canonical_model(raw_model);
    routes
        .get(&model)
        .or_else(|| routes.get(""))
        .and_then(|route| ProviderProtocol::parse(&route.protocol))
}

/// The `(model, route)` to persist after a turn, or `None` if it matches what's
/// stored — the launch path's "confirmed + changed" rule, so only discoveries write.
fn chat_route_to_persist(
    key: &ApiKey,
    raw_model: &str,
    format: &ChatFormat,
) -> Option<(String, PersistedRoute)> {
    // Cursor ACP keys skip the wire-format cascade; `format` is just a placeholder.
    if raw_model.trim().is_empty() || key.is_cursor_acp() {
        return None;
    }
    let model = canonical_model(raw_model);
    let route = PersistedRoute::from_route(chat_format_protocol(format), PathVariant::Default);
    // Compare packed bytes so a default variant ("" vs "default") isn't a false diff.
    if key
        .routes_for_tool("code")
        .get(&model)
        .and_then(PersistedRoute::to_byte)
        == route.to_byte()
    {
        return None;
    }
    Some((model, route))
}

/// CodeCommand provides an interactive REPL for chatting with AI models
pub struct CodeCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl CodeCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    /// Resolves the model to use:
    /// --model flag > persisted per-key > last_selection > None (show picker)
    async fn resolve_model(
        &self,
        key: &ApiKey,
        flag_model: Option<String>,
    ) -> Result<Option<String>> {
        match flag_model {
            // --model with no value → force picker (bypass persisted model)
            Some(ref m) if m.is_empty() => Ok(None),
            // --model <value> → use it and save
            Some(model) => {
                let current = self.session_store.get_code_model(&key.id).await?;
                if current.as_deref() != Some(&model) {
                    self.session_store.set_code_model(&key.id, &model).await?;
                }
                Ok(Some(model))
            }
            None => {
                if let Some(m) = self.session_store.get_code_model(&key.id).await? {
                    if self.starter_model_valid(key, &m).await {
                        return Ok(Some(m));
                    }
                    return Ok(None);
                }
                if let Ok(Some(sel)) = self.session_store.get_last_selection().await
                    && sel.key_id == key.id
                    && let Some(ref m) = sel.model
                    && m != crate::constants::MODEL_DEFAULT_PLACEHOLDER
                {
                    if self.starter_model_valid(key, m).await {
                        return Ok(Some(m.clone()));
                    }
                    return Ok(None);
                }
                Ok(None)
            }
        }
    }

    /// Wraps `starter_model_still_available`, printing a notice when the
    /// persisted model has been removed from the aivo-starter catalog so the
    /// caller knows why the picker is about to open.
    async fn starter_model_valid(&self, key: &ApiKey, model: &str) -> bool {
        if crate::services::model_catalog::starter_model_still_available(key, &self.cache, model)
            .await
        {
            return true;
        }
        eprintln!(
            "{} Model '{}' is no longer available on aivo-starter. Pick another:",
            style::yellow("Note:"),
            model
        );
        false
    }

    /// Transforms model names for OpenRouter compatibility
    /// OpenRouter uses dots in version numbers: 4.6 instead of 4-6
    fn transform_model_for_provider(base_url: &str, model: &str) -> String {
        // No catalog needed: the picker already yields a provider-correct name.
        model_names::transform_model_for_provider(None, base_url, model)
    }

    /// Read-only preview for `aivo code --dry-run`: resolves the key, model,
    /// endpoint, sandbox dir, and mode the way a real launch would — but performs
    /// no HTTP, persistence, model picker, or server spawn. HF refs show a
    /// placeholder endpoint (llama-server is not started).
    #[allow(clippy::too_many_arguments)]
    async fn print_code_dry_run(
        &self,
        model_input: Option<String>,
        attachments: &[String],
        key_override: Option<ApiKey>,
        one_shot: Option<&str>,
        initial_prompt: bool,
        agent_mode: bool,
        max_context: Option<u64>,
    ) -> Result<ExitCode> {
        let hf = model_input.as_deref().is_some_and(|m| {
            huggingface::is_hf_or_local_gguf(m) || huggingface::is_bare_hf_picker_trigger(m)
        });

        println!("{}", style::bold("Dry run — aivo code would start with:"));
        println!();

        if hf {
            let label = model_input.clone().unwrap_or_default();
            println!(
                "{} {} {}",
                style::bold("Key:   "),
                style::cyan("hf-local"),
                style::dim("(local llama-server — not spawned in dry run)"),
            );
            println!("{} {}", style::bold("Model: "), label);
        } else {
            let key = match key_override {
                Some(k) => Some(k),
                None => self.session_store.get_active_key().await.ok().flatten(),
            };
            match &key {
                Some(k) => println!(
                    "{} {} {}",
                    style::bold("Key:   "),
                    style::cyan(k.display_name()),
                    style::dim(format!("({})", friendly_base_url(&k.base_url))),
                ),
                None => println!(
                    "{} {}",
                    style::bold("Key:   "),
                    style::dim("(none — run `aivo keys add`)"),
                ),
            }
            // Model precedence, read-only: explicit `-m` > per-key persisted
            // model > last selection > picker (shown as a hint).
            let model = match model_input.as_deref() {
                Some(m) if !m.is_empty() => Some(m.to_string()),
                _ => {
                    let mut m: Option<String> = None;
                    if let Some(k) = &key {
                        m = self
                            .session_store
                            .get_code_model(&k.id)
                            .await
                            .ok()
                            .flatten();
                        if m.is_none()
                            && let Ok(Some(sel)) = self.session_store.get_last_selection().await
                            && sel.key_id == k.id
                        {
                            m = sel
                                .model
                                .filter(|x| x != crate::constants::MODEL_DEFAULT_PLACEHOLDER);
                        }
                    }
                    m
                }
            };
            println!(
                "{} {}",
                style::bold("Model: "),
                model.as_deref().unwrap_or("(model picker)"),
            );
        }

        println!(
            "{} {}",
            style::bold("Mode:  "),
            if one_shot.is_some() {
                if agent_mode {
                    "one-shot agent (-e)"
                } else {
                    "one-shot (-p)"
                }
            } else if initial_prompt {
                "interactive TUI (positional prompt sent first)"
            } else {
                "interactive TUI"
            },
        );
        if let Some(window) = max_context {
            println!(
                "{} {} {}",
                style::bold("Context:"),
                window,
                style::dim("(--max-context override, this session)"),
            );
        }
        if let Ok(dir) = chat_sandbox_dir() {
            println!(
                "{} {}",
                style::bold("Sandbox:"),
                style::dim(dir.to_string_lossy()),
            );
        }
        if !attachments.is_empty() {
            println!("{} {}", style::bold("Attach:"), attachments.join(", "),);
        }
        Ok(ExitCode::Success)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        model: Option<String>,
        one_shot: Option<String>,
        initial_prompt: Option<String>,
        attachments: Vec<String>,
        refresh: bool,
        key_override: Option<ApiKey>,
        key_explicit: bool,
        json: bool,
        resume: Option<String>,
        max_context: Option<String>,
        dry_run: bool,
        share: bool,
        agent_mode: bool,
        output_format: Option<String>,
        max_steps: Option<u32>,
        max_output_tokens: Option<u64>,
        max_cost: Option<f64>,
        auto_approve: bool,
        vision_model: Option<String>,
        image_model: Option<String>,
    ) -> ExitCode {
        match self
            .execute_internal(
                model,
                one_shot,
                initial_prompt,
                attachments,
                refresh,
                key_override,
                key_explicit,
                json,
                resume,
                max_context,
                dry_run,
                share,
                agent_mode,
                output_format,
                max_steps,
                max_output_tokens,
                max_cost,
                auto_approve,
                vision_model,
                image_model,
            )
            .await
        {
            Ok(code) => code,
            Err(e) => crate::errors::report(&e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_internal(
        &self,
        model_flag: Option<String>,
        one_shot: Option<String>,
        initial_prompt: Option<String>,
        attachments: Vec<String>,
        refresh: bool,
        key_override: Option<ApiKey>,
        key_explicit: bool,
        json: bool,
        resume: Option<String>,
        max_context: Option<String>,
        dry_run: bool,
        share: bool,
        agent_mode: bool,
        output_format: Option<String>,
        max_steps: Option<u32>,
        max_output_tokens: Option<u64>,
        max_cost: Option<f64>,
        auto_approve: bool,
        vision_model: Option<String>,
        image_model: Option<String>,
    ) -> Result<ExitCode> {
        if (max_steps.is_some() || max_output_tokens.is_some()) && !agent_mode {
            anyhow::bail!("--max-steps and --max-output-tokens may only be used with -e/--exec");
        }
        if auto_approve && one_shot.is_some() && !agent_mode {
            anyhow::bail!("--auto-approve has no effect with -p (no tools run)");
        }

        // Validate `--max-context` up front so a malformed value fails fast.
        let max_context: Option<u64> = match max_context.as_deref() {
            Some(s) => Some(
                crate::services::model_metadata::parse_context_size(s).ok_or_else(|| {
                    anyhow::anyhow!("invalid --max-context '{s}' (use e.g. 200k, 1m, or 128000)")
                })?,
            ),
            None => None,
        };

        // `--dry-run` previews resolution and exits before any side effect
        // (no HTTP, persistence, model picker, llama-server spawn, or TUI).
        // Placed first so it also skips the over-broad-workspace prompt below.
        if dry_run {
            return self
                .print_code_dry_run(
                    model_flag,
                    &attachments,
                    key_override,
                    one_shot.as_deref(),
                    initial_prompt.is_some(),
                    agent_mode,
                    max_context,
                )
                .await;
        }

        // Bare `hf:` opens a picker; rewrite to a concrete ref.
        let model_flag = match model_flag.as_deref() {
            Some(m) if huggingface::is_bare_hf_picker_trigger(m) => {
                match huggingface::pick_cached_short_ref() {
                    Some(short) => Some(short),
                    None => return Ok(ExitCode::Success),
                }
            }
            _ => model_flag,
        };

        // HF takeover: bypass key resolution + last_selection persistence.
        let (mut key, model_flag, hf_active) = match model_flag.as_deref() {
            Some(m) if huggingface::is_hf_or_local_gguf(m) => {
                let hf_ref = huggingface::parse_hf_ref(m)?;
                let port = huggingface::ensure_ready(&hf_ref).await?;
                let synthetic = ApiKey::new_with_protocol(
                    "aivo-hf-local".to_string(),
                    format!("hf:{}", hf_ref.repo),
                    huggingface::local_openai_base_url(port),
                    None,
                    "huggingface".to_string(),
                );
                (synthetic, Some(hf_ref.display_model_name()), true)
            }
            _ => {
                let key = match key_override {
                    Some(k) => k,
                    None => match self.session_store.get_active_key().await? {
                        Some(k) => k,
                        None => {
                            crate::commands::print_no_key_error();
                            return Ok(ExitCode::AuthError);
                        }
                    },
                };
                (key, model_flag, false)
            }
        };

        // Reject CLI-only OAuth for `aivo code`; provider OAuth (grok/codex) is ok.
        if !hf_active && key.is_any_oauth() && !key.is_provider_oauth() {
            key = match crate::commands::keys::swap_incompatible_key(
                &self.session_store,
                &key,
                crate::services::key_compat::KeyCompatContext::Chat,
                "aivo code",
            )
            .await?
            {
                Some(k) => k,
                None => return Ok(ExitCode::UserError),
            };
        }

        if !hf_active
            && key.is_cursor_acp()
            && cursor_acp::cursor_oauth_shadow_signed_out(&key).await
        {
            eprintln!(
                "{} {}",
                style::red("Error:"),
                cursor_acp::cursor_login_reauth_hint(&key.id)
            );
            return Ok(ExitCode::AuthError);
        }

        let client = crate::services::http_utils::router_http_client();
        // `aivo code` always runs in an isolated sandbox dir so backends
        // that accept a `cwd` (cursor ACP today) can't auto-pull
        // surrounding project files into the conversation. Chat is a
        // pure conversation surface; if you need project access, use
        // `aivo run claude` / `aivo run codex` / etc.
        let cwd = chat_sandbox_dir()?.to_string_lossy().into_owned();

        // HF mode already has the model set; skip `resolve_model` to
        // avoid persisting it under the ephemeral synthetic key id.
        let resolved = if hf_active {
            Some(model_flag.clone().unwrap_or_default())
        } else {
            // Precedence: explicit `--model` > the per-key persisted model >
            // last selection > picker.
            let effective = model_flag.clone();
            self.resolve_model(&key, effective).await?
        };
        let raw_model = match resolved {
            Some(m) => m,
            None => {
                ensure_picker_terminal("model", "--model <name>")?;
                // Fetch the full catalog and annotate non-chat models as
                // disabled rather than hiding them — users see why image /
                // audio / embedding entries aren't selectable.
                let models_list = crate::commands::models::fetch_all_models_for_picker(
                    &client,
                    &key,
                    &self.cache,
                    refresh,
                )
                .await;

                if models_list.is_empty() {
                    anyhow::bail!(
                        "No model configured and could not fetch model list. Use --model <name> to specify one."
                    );
                }

                let annotations =
                    crate::services::model_compat::text_chat_annotations(&models_list);
                match crate::commands::models::prompt_model_picker(
                    models_list,
                    None,
                    annotations,
                    "Select model",
                ) {
                    Some(selected) => {
                        self.session_store
                            .set_code_model(&key.id, &selected)
                            .await?;
                        selected
                    }
                    None => return Ok(ExitCode::Success),
                }
            }
        };

        // chat is a first-class coding agent like the native tools, so it
        // records itself as the last tool (the native launch does the same in
        // run.rs). A later bare `aivo run` then recalls "code". Skipped for HF —
        // its synthetic key is ephemeral and shouldn't be remembered.
        if !hf_active {
            let _ = self
                .session_store
                .set_last_selection(&key, "code", Some(&raw_model))
                .await;
        }

        let model = Self::transform_model_for_provider(&key.base_url, &raw_model);
        let pending_attachments = build_pending_attachments(&attachments)?;

        // Resolve sentinel base URLs to actual URLs before any HTTP calls.
        if key.base_url == "ollama" {
            crate::services::ollama::ensure_ready().await?;
            crate::services::ollama::ensure_model(&raw_model).await?;
            key.base_url = crate::services::ollama::ollama_openai_base_url();
        } else if key.base_url == crate::constants::AIVO_STARTER_SENTINEL {
            key.base_url = crate::constants::AIVO_STARTER_REAL_URL.to_string();
        }

        // Create once so its token cache is reused across messages in the session.
        let copilot_tm = if key.base_url == "copilot" {
            Some(Arc::new(CopilotTokenManager::new(
                key.key.as_str().to_string(),
            )))
        } else {
            None
        };

        // `--resume <id>` pre-resolution, fidelity ladder: a saved aivo session
        // (prefix → full id), an importable claude/codex/pi session (left for
        // the TUI/one-shot import path), or — for sources with no importer
        // (gemini, opencode, out-of-window natives) — a rendered context
        // digest into a fresh session. A selector that matches nothing is a
        // hard failure. Bare/`last` selectors go straight through.
        let mut resume = resume;
        let mut injected_context: Option<String> = None;
        let mut injected_context_summary: Option<String> = None;
        if let Some(sel) = resume
            .clone()
            .filter(|s| !s.trim().is_empty() && s != "last")
        {
            use crate::commands::run::ContextResolution;
            let mut inject_digest = |thread: &crate::services::project_id::Thread| {
                let rendered = crate::services::context_render::render_for_aivo_code(thread);
                let summary = format!(
                    "{} (context digest — no {} importer)",
                    crate::commands::run::context_injection_summary(&rendered, thread),
                    thread.cli,
                );
                eprintln!("  {} {}", crate::style::arrow_symbol(), summary);
                injected_context = Some(rendered.text);
                injected_context_summary = Some(summary);
            };
            // Cheap logs.db probe first: resolving a plugin run here skips
            // both expensive session walks (import scan + context ingest).
            match crate::commands::run::try_plugin_run_resume(&self.session_store, &sel).await {
                Some(ContextResolution::Selected(thread)) => {
                    inject_digest(&thread);
                    resume = None;
                }
                Some(ContextResolution::Cancelled) => return Ok(ExitCode::Success),
                Some(ContextResolution::Unavailable(msg)) => anyhow::bail!(msg),
                None => {
                    use crate::services::session_import::{ResumeTarget, resolve_resume_target};
                    let cwd = crate::services::system_env::current_dir().unwrap_or_default();
                    match resolve_resume_target(
                        &self.session_store,
                        std::path::Path::new(&cwd),
                        &sel,
                    )
                    .await
                    {
                        // Resolvable: keep the user's selector — the TUI/one-shot
                        // re-resolve it through the same ladder, and the LOAD path
                        // owns the fork-first collapse plus the source-diverged
                        // notice. This block only vets Ambiguous/Unknown up front.
                        ResumeTarget::AivoSession(_) | ResumeTarget::Foreign(_) => {}
                        ResumeTarget::Ambiguous(msg) => anyhow::bail!(msg),
                        ResumeTarget::Unknown => {
                            match crate::commands::run::resolve_context_thread(
                                &self.session_store,
                                &sel,
                            )
                            .await
                            {
                                ContextResolution::Selected(thread) => {
                                    inject_digest(&thread);
                                    resume = None;
                                }
                                // Unreachable today (explicit selectors never open a
                                // picker), but a cancel means don't open at all.
                                ContextResolution::Cancelled => return Ok(ExitCode::Success),
                                ContextResolution::Unavailable(msg) => anyhow::bail!(msg),
                            }
                        }
                    }
                }
            }
        }

        if let Some(input) = one_shot {
            let one_shot_input = if input.is_empty() {
                sanitize_one_shot_message(read_one_shot_message_from_stdin()?)?
            } else {
                let input = sanitize_one_shot_message(input)?;
                let stdin_context = read_stdin_if_piped()?;
                compose_one_shot_prompt(&input, stdin_context.as_deref())
            };
            // -e runs the agent (serve-reachable keys only); -p falls through to plain.
            if agent_mode {
                if !code_agent_oneshot::key_is_agent_capable(&key) {
                    anyhow::bail!(
                        "-e/--exec needs a standard API key; this key can't run the in-process agent"
                    );
                }
                let one_shot_attachments = materialize_attachments(&pending_attachments).await?;
                self.session_store
                    .record_selection(&key.id, "code", Some(&raw_model))
                    .await?;
                return code_agent_oneshot::run_one_shot_agent(
                    &self.session_store,
                    &self.cache,
                    &key,
                    &raw_model,
                    one_shot_input,
                    one_shot_attachments,
                    injected_context,
                    max_context,
                    code_agent_oneshot::OutputFormat::parse(output_format.as_deref()),
                    code_agent_oneshot::OneShotAgentLimits {
                        max_steps,
                        max_output_tokens,
                        max_cost,
                    },
                    auto_approve,
                    resume,
                    model_flag.is_some(),
                    image_model,
                )
                .await;
            }

            // Plain `-p` has no system-prompt hook; prepend the context block.
            let one_shot_input = match injected_context.as_deref() {
                Some(ctx) => format!("{ctx}\n\n{one_shot_input}"),
                None => one_shot_input,
            };

            let one_shot_attachments = materialize_attachments(&pending_attachments).await?;
            let history = vec![ChatMessage {
                model: None,
                role: "user".to_string(),
                content: one_shot_input,
                reasoning_content: None,
                attachments: one_shot_attachments,
            }];
            let mut format = seeded_chat_format(&key, &raw_model);
            self.session_store
                .record_selection(&key.id, "code", Some(&raw_model))
                .await?;
            let (spinning, spinner_handle) = style::start_spinner(None);
            let mut current_section: Option<&'static str> = None;
            let mut on_chunk = |chunk| {
                if json {
                    return Ok(());
                }
                match chunk {
                    ChatResponseChunk::Reasoning(_) | ChatResponseChunk::Usage(_) => {}
                    ChatResponseChunk::Content(text) => {
                        if current_section != Some("answer") {
                            if current_section.is_some() {
                                print!("\n\n");
                            }
                            current_section = Some("answer");
                        }
                        print!("{text}");
                    }
                }
                io::stdout().flush()?;
                Ok(())
            };
            // Install a Ctrl+C handler so SIGINT cancels the in-flight request
            // cleanly: dropping the `send_message_turn` future closes the HTTP
            // connection before the process exits. Without this branch the
            // default SIGINT terminates the process abruptly, leaving the
            // server to keep generating.
            let result = if key.is_cursor_acp() {
                let prompt_text = history[0].content.clone();
                let attachments = history[0].attachments.clone();
                let spinning_for_cursor = spinning.clone();
                let mut forward = |chunk: CursorChunk<'_>| -> Result<()> {
                    style::stop_spinner(&spinning_for_cursor);
                    let mapped = match chunk {
                        CursorChunk::Content(t) => ChatResponseChunk::Content(t.to_string()),
                        CursorChunk::Reasoning(t) => ChatResponseChunk::Reasoning(t.to_string()),
                        // Tool steps go to stderr (not stdout, which scripts
                        // parse) so the user sees cursor touched the workspace.
                        CursorChunk::ToolCall { name, .. } => {
                            eprintln!("· cursor tool: {name}");
                            return Ok(());
                        }
                        CursorChunk::ToolUpdate { failed, .. } => {
                            if failed {
                                eprintln!("· cursor tool: failed");
                            }
                            return Ok(());
                        }
                        CursorChunk::Usage { .. } => return Ok(()),
                    };
                    on_chunk(mapped)
                };
                tokio::select! {
                    res = cursor_acp::run_cursor_acp_turn(
                        &key,
                        Some(&raw_model),
                        &cwd,
                        &prompt_text,
                        &attachments,
                        &mut forward,
                    ) => res.map(|cur| ChatTurnResult {
                        content: cur.content,
                        usage: cur.usage.map(cursor_acp::CursorUsage::to_token_usage),
                        model: cur.model,
                        raw_body: None,
                    }),
                    _ = tokio::signal::ctrl_c() => {
                        style::stop_spinner(&spinning);
                        let _ = spinner_handle.await;
                        eprintln!();
                        return Ok(ExitCode::ToolExit(130));
                    }
                }
            } else {
                tokio::select! {
                    res = send_message_turn(
                        &client,
                        &key,
                        copilot_tm.as_deref(),
                        &model,
                        &history,
                        &mut format,
                        &spinning,
                        json,
                        &mut on_chunk,
                    ) => res,
                    _ = tokio::signal::ctrl_c() => {
                        style::stop_spinner(&spinning);
                        let _ = spinner_handle.await;
                        eprintln!();
                        return Ok(ExitCode::ToolExit(130));
                    }
                }
            };
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;

            match result {
                Ok(turn) => {
                    if let Some((model_key, route)) =
                        chat_route_to_persist(&key, &raw_model, &format)
                    {
                        let _ = self
                            .session_store
                            .merge_routes(&key.id, "code", &[(model_key, route)])
                            .await;
                    }
                    let prompt_text: String = history.iter().map(|m| m.content.as_str()).collect();
                    let usage = turn.usage_or_estimate(&prompt_text);
                    let billed_model = turn.model.as_deref();
                    let stats_model = billed_model.unwrap_or(&raw_model);
                    self.session_store
                        .record_tokens(
                            &key.id,
                            Some("code"),
                            Some(stats_model),
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.cache_read_input_tokens,
                            usage.cache_creation_input_tokens,
                        )
                        .await?;
                    let chat_session_id = new_code_session_id();
                    let _ = log_chat_turn(
                        &self.session_store,
                        &key,
                        &raw_model,
                        Some(&cwd),
                        Some(&chat_session_id),
                        &history[0],
                        &turn.content,
                        None,
                        &usage,
                    )
                    .await;
                    let (stored, title, preview) = build_one_shot_persist_inputs(
                        &history,
                        turn.content.clone(),
                        None,
                        &raw_model,
                    );
                    let session_tokens = SessionTokens {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        cache_read_tokens: usage.cache_read_input_tokens,
                        cache_write_tokens: usage.cache_creation_input_tokens,
                    };
                    let cost = crate::services::model_metadata::model_pricing(&raw_model)
                        .or_else(|| {
                            billed_model.and_then(crate::services::model_metadata::model_pricing)
                        })
                        .and_then(|p| p.cost_usd(&session_tokens))
                        .unwrap_or(0.0);
                    let _ = self
                        .session_store
                        .save_code_session_with_id(
                            &key.id,
                            &key.base_url,
                            &cwd,
                            &chat_session_id,
                            &raw_model,
                            billed_model,
                            &stored,
                            &title,
                            &preview,
                            session_tokens,
                            cost,
                        )
                        .await;
                    if json {
                        let body = turn.raw_body.ok_or_else(|| {
                            anyhow::anyhow!(
                                "provider did not return a non-streaming body; --json cannot serialize partial streaming output"
                            )
                        })?;
                        println!("{}", serde_json::to_string_pretty(&body)?);
                    } else {
                        println!();
                    }
                    return Ok(ExitCode::Success);
                }
                Err(e) => return Err(e),
            }
        }

        refuse_nested_tui()?;
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            anyhow::bail!(
                "The interactive coding agent uses a full-screen TUI. Run it in a terminal, or go non-interactive: -e/--exec runs the agent (tools), -p/--prompt gets a plain reply."
            );
        }

        // `--share` needs a linked account; fail fast rather than launch unshared.
        if share && !crate::commands::share::device_linked().await {
            return Err(crate::commands::share::not_linked_error());
        }

        let initial_session = new_code_session_id();
        let initial_history = Vec::new();

        let startup_notice = attachment_notice(&pending_attachments);

        self.session_store
            .record_selection(&key.id, "code", Some(&raw_model))
            .await?;

        code_tui::run_chat_tui(code_tui::CodeTuiParams {
            session_store: self.session_store.clone(),
            cache: self.cache.clone(),
            client,
            key,
            copilot_tm,
            cwd,
            raw_model,
            model,
            initial_session,
            initial_history,
            initial_draft_attachments: pending_attachments,
            startup_notice,
            initial_resume: resume,
            model_explicit: model_flag.is_some(),
            key_explicit,
            initial_prompt,
            injected_context,
            injected_context_summary,
            max_context,
            share,
            auto_approve,
            vision_model,
            image_model,
        })
        .await?;

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!(
            "{} aivo code [\"<text>\"|REF] [--model <model>] [-p [prompt]] [--attach <path> ...]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("Interactive coding agent TUI, or one-shot from the shell.")
        );
        println!(
            "{}",
            style::dim("Type /help inside the TUI for slash commands and keybindings.")
        );
        println!();
        println!("{}", style::bold("Modes:"));
        let print_mode = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_mode(
            "aivo code [text]",
            "Full-screen TUI; positional text is sent as the first message",
        );
        print_mode(
            "-e \"task\"",
            "Run the agent headlessly: tools + file edits, then exit",
        );
        print_mode(
            "-p \"prompt\"",
            "One plain reply: no tools, no project access",
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<26}", "mcp")),
            style::dim("Manage MCP servers (aivo code mcp --help)")
        );
        println!(
            "  {}{}",
            style::cyan(format!("{:<26}", "skills")),
            style::dim("Manage agent skills (aivo code skills --help)")
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
        print_opt("-m, --model <model>", "Model to use (saved for next time)");
        print_opt("-k, --key <id|name>", "API key by id/name (bare = picker)");
        print_opt(
            "key::model",
            "Shorthand for -k key -m model (as -m value or bare arg)",
        );
        print_opt(
            "-p, --prompt [prompt]",
            "One prompt, plain reply, exit (no tools)",
        );
        print_opt(
            "-e, --exec [prompt]",
            "One prompt, run the agent, exit (tools)",
        );
        print_opt("--max-steps <N>", "Max -e agent steps (0 disables)");
        print_opt(
            "--max-output-tokens <N>",
            "Max -e output tokens (0 disables)",
        );
        print_opt("--max-cost <usd>", "Max estimated -e spend in USD");
        print_opt(
            "--add-dir <dir>",
            "Extra writable workspace root (repeatable)",
        );
        print_opt(
            "--sandbox <profile>",
            "Shell sandbox: off, workspace, read-only, strict",
        );
        print_opt(
            "--best-of-n <N>",
            "Run -e N ways in parallel, return the best (N ≤ 10; read-only sandbox)",
        );
        print_opt(
            "--json-schema <schema>",
            "Constrain the -e answer to a JSON Schema (inline or @file)",
        );
        print_opt("-r, --refresh", "Refresh the model list (skip cache)");
        print_opt(
            "--resume [last|id]",
            "Resume a saved session, or another agent's (claude/codex/pi id from `aivo logs`; works with -e)",
        );
        print_opt("--share", "Share this session live (needs `aivo login`)");
        print_opt("--attach <path>", "Attach a file or image to the message");
        print_opt(
            "--vision-model [[key::]m]",
            "Vision model that describes images for text-only models (bare/key:: = picker)",
        );
        print_opt(
            "--image-model [key::]m",
            "Image model backing the agent's generate_image tool (session-only)",
        );
        print_opt("--json", "Raw provider JSON (with -p)");
        print_opt(
            "--output-format <fmt>",
            "-e output: text (default), json, or stream-json",
        );
        print_opt(
            "--max-context <size>",
            "Override context window (e.g. 200k)",
        );
        print_opt(
            "--dry-run",
            "Show resolved key/model/endpoint, don't connect",
        );
        print_opt(
            "--auto-approve",
            "Start in auto-approve mode (catastrophic still confirms)",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!(
            "  {}",
            style::dim("aivo code \"Make the failing test pass\"")
        );
        println!(
            "  {}",
            style::dim("aivo code -p \"Explain Rust lifetimes\"")
        );
        println!(
            "  {}",
            style::dim("aivo code -e \"make the failing test pass\"")
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_message_turn<F>(
    client: &Client,
    key: &ApiKey,
    copilot_tm: Option<&CopilotTokenManager>,
    model: &str,
    history: &[ChatMessage],
    format: &mut ChatFormat,
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    if let Some(tm) = copilot_tm {
        match send_copilot_request(
            client,
            tm,
            model,
            history,
            spinning,
            non_streaming,
            on_chunk,
        )
        .await
        {
            ok @ Ok(_) => return ok,
            Err(e) if is_responses_api_required(&e) => {
                match send_copilot_responses_request(
                    client,
                    tm,
                    model,
                    history,
                    spinning,
                    non_streaming,
                    on_chunk,
                )
                .await
                {
                    Ok(content) => {
                        *format = ChatFormat::Responses;
                        return Ok(content);
                    }
                    Err(responses_err) => return Err(responses_err),
                }
            }
            Err(e) => return Err(e),
        }
    }

    match format {
        ChatFormat::OpenAI => {
            match send_chat_request(
                client,
                key,
                model,
                history,
                spinning,
                non_streaming,
                on_chunk,
            )
            .await
            {
                ok @ Ok(_) => ok,
                Err(e) if is_format_mismatch(&e) => {
                    match send_chat_via_shared_fallback(
                        client,
                        key,
                        model,
                        history,
                        ProviderProtocol::Anthropic,
                        spinning,
                        non_streaming,
                        on_chunk,
                    )
                    .await
                    {
                        Ok((content, protocol)) => {
                            *format = chat_format_for_router_protocol(protocol);
                            Ok(content)
                        }
                        Err(_) => Err(e),
                    }
                }
                Err(e) if is_responses_api_required(&e) => {
                    // Model requires the Responses API instead of Chat Completions
                    match send_responses_request(
                        client,
                        key,
                        model,
                        history,
                        spinning,
                        non_streaming,
                        on_chunk,
                    )
                    .await
                    {
                        Ok(content) => {
                            *format = ChatFormat::Responses;
                            Ok(content)
                        }
                        Err(responses_err) => Err(responses_err),
                    }
                }
                Err(e) => Err(e),
            }
        }
        ChatFormat::Anthropic => {
            send_anthropic_request(
                client,
                key,
                model,
                history,
                spinning,
                non_streaming,
                on_chunk,
            )
            .await
        }
        ChatFormat::Responses => {
            send_responses_request(
                client,
                key,
                model,
                history,
                spinning,
                non_streaming,
                on_chunk,
            )
            .await
        }
        ChatFormat::Google => {
            match send_google_request(
                client,
                key,
                model,
                history,
                spinning,
                non_streaming,
                on_chunk,
            )
            .await
            {
                ok @ Ok(_) => ok,
                Err(e) if is_format_mismatch(&e) => {
                    match send_chat_via_shared_fallback(
                        client,
                        key,
                        model,
                        history,
                        ProviderProtocol::Openai,
                        spinning,
                        non_streaming,
                        on_chunk,
                    )
                    .await
                    {
                        Ok((content, protocol)) => {
                            *format = chat_format_for_router_protocol(protocol);
                            Ok(content)
                        }
                        Err(_) => Err(e),
                    }
                }
                Err(e) => Err(e),
            }
        }
    }
}

fn chat_format_for_router_protocol(protocol: ProviderProtocol) -> ChatFormat {
    match protocol {
        ProviderProtocol::Anthropic => ChatFormat::Anthropic,
        ProviderProtocol::Google => ChatFormat::Google,
        // The shared Chat Completions fallback currently treats ResponsesApi as
        // an OpenAI-chat wire route, not as `/v1/responses`.
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => ChatFormat::OpenAI,
    }
}

fn chat_path_variant_for_protocol(key: &ApiKey, protocol: ProviderProtocol) -> Option<PathVariant> {
    let value = match protocol {
        ProviderProtocol::Google => key.gemini_path_variant.as_deref(),
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi | ProviderProtocol::Anthropic => {
            key.claude_path_variant.as_deref()
        }
    };
    value.and_then(PathVariant::parse)
}

fn shared_chat_fallback_config(
    key: &ApiKey,
    initial_protocol: ProviderProtocol,
) -> ResponsesToChatRouterConfig {
    let profile = provider_profile_for_key(key);
    let mut requires_reasoning_content = profile.quirks.requires_reasoning_content;
    requires_reasoning_content |= key.requires_reasoning_content == Some(true);

    ResponsesToChatRouterConfig {
        target_base_url: resolve_starter_base_url(&key.base_url),
        api_key: key.key.as_str().to_string(),
        target_protocol: initial_protocol,
        target_path_variant: chat_path_variant_for_protocol(key, initial_protocol),
        copilot_token_manager: None,
        model_prefix: profile.quirks.model_prefix.map(str::to_string),
        requires_reasoning_content,
        actual_model: None,
        max_tokens_cap: profile.quirks.max_tokens_cap,
        responses_api_supported: key.responses_api_supported,
        is_starter: profile.serve_flags.is_starter,
        aivo_prefix_models: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
async fn send_chat_via_shared_fallback<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    initial_protocol: ProviderProtocol,
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<(ChatTurnResult, ProviderProtocol)>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let body = build_openai_chat_request(model, messages, !non_streaming, None)?;
    let requested_stream = body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let config = shared_chat_fallback_config(key, initial_protocol);
    let (body, protocol) =
        forward_chat_completions_with_fallback(&body, config, client, requested_stream).await?;
    let result = emit_openai_chat_value(body, spinning, on_chunk)?;
    Ok((result, protocol))
}

fn read_one_shot_message_from_stdin() -> Result<String> {
    if io::stdin().is_terminal() {
        eprintln!(
            "{}",
            style::dim("Enter message, then press Ctrl-D to send.")
        );
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn compose_one_shot_prompt(prompt: &str, stdin_context: Option<&str>) -> String {
    match stdin_context.map(str::trim).filter(|c| !c.is_empty()) {
        Some(ctx) => format!("{prompt}\n\nContext from stdin:\n{ctx}"),
        None => prompt.to_string(),
    }
}

fn sanitize_one_shot_message(message: String) -> Result<String> {
    if message.trim().is_empty() {
        anyhow::bail!("Prompt for -p/--prompt cannot be empty");
    }
    Ok(message)
}

/// Refuse a second interactive TUI inside a running `aivo code`. The `!cmd` PTY
/// passes the TTY check, so the env marker is the only reliable signal; one-shot
/// `-p`/`-e` runs return before this and stay nestable.
fn refuse_nested_tui() -> Result<()> {
    if std::env::var_os(crate::constants::CODE_ACTIVE_ENV).is_some() {
        anyhow::bail!(
            "Already inside an `aivo code` session — nested TUIs aren't supported. Use /new for a fresh task, /resume to reopen a session, or a subagent for parallel work."
        );
    }
    Ok(())
}

fn ensure_picker_terminal(kind: &str, explicit_flag: &str) -> Result<()> {
    if io::stderr().is_terminal() {
        return Ok(());
    }

    anyhow::bail!(
        "Cannot open {kind} picker without a terminal. Run in a terminal or pass {explicit_flag}."
    );
}

fn attachment_notice(attachments: &[MessageAttachment]) -> Option<String> {
    if attachments.is_empty() {
        None
    } else {
        Some(format!(
            "{} attachment{} queued. Press Enter to send or use /attach to add more.",
            attachments.len(),
            if attachments.len() == 1 { "" } else { "s" }
        ))
    }
}

fn build_pending_attachments(paths: &[String]) -> Result<Vec<MessageAttachment>> {
    paths
        .iter()
        .map(|path| build_pending_attachment(path))
        .collect()
}

fn build_pending_attachment(path: &str) -> Result<MessageAttachment> {
    let expanded = crate::services::system_env::expand_tilde(path);
    let file_path = expanded.as_path();
    ensure_attachment_exists(file_path)?;
    let mime_type = guess_attachment_mime_type(file_path)?;
    Ok(MessageAttachment {
        name: attachment_name(file_path),
        mime_type,
        storage: AttachmentStorage::FileRef {
            path: expanded.to_string_lossy().into_owned(),
        },
    })
}

fn ensure_attachment_exists(path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| anyhow::anyhow!("Failed to read attachment '{}': {err}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Attachment '{}' is not a file", path.display());
    }
    Ok(())
}

fn attachment_name(path: &Path) -> String {
    match path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
    {
        Some(name) => name.to_string(),
        None => path.to_string_lossy().into_owned(),
    }
}

fn guess_attachment_mime_type(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let mime = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "json" => CONTENT_TYPE_JSON,
        "md" => "text/markdown",
        "html" => "text/html",
        "css" => "text/css",
        "csv" => "text/csv",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "toml" => "application/toml",
        "" => "text/plain",
        _ => "text/plain",
    };
    Ok(mime.to_string())
}

async fn materialize_attachments(
    attachments: &[MessageAttachment],
) -> Result<Vec<MessageAttachment>> {
    let mut resolved = Vec::with_capacity(attachments.len());
    for attachment in attachments {
        resolved.push(materialize_attachment(attachment).await?);
    }
    Ok(resolved)
}

/// Whether this MIME type represents a binary document that should be base64 encoded.
pub fn is_document_mime(mime: &str) -> bool {
    mime == "application/pdf"
}

async fn materialize_attachment(attachment: &MessageAttachment) -> Result<MessageAttachment> {
    if attachment.is_image() {
        return materialize_image_attachment(attachment).await;
    }
    match &attachment.storage {
        AttachmentStorage::Inline { .. } => Ok(attachment.clone()),
        AttachmentStorage::FileRef { path } => {
            let storage = if is_document_mime(&attachment.mime_type) {
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(|err| anyhow::anyhow!("Failed to read '{}': {err}", path))?;
                AttachmentStorage::Inline {
                    data: BASE64.encode(bytes),
                }
            } else {
                match tokio::fs::read_to_string(path).await {
                    Ok(text) => AttachmentStorage::Inline { data: text },
                    // base64 labeled text/plain would send a meaningless blob.
                    Err(_) => anyhow::bail!(
                        "'{}' looks like a binary file — attach images, PDFs, or text",
                        attachment.name
                    ),
                }
            };
            Ok(MessageAttachment {
                name: attachment.name.clone(),
                mime_type: attachment.mime_type.clone(),
                storage,
            })
        }
    }
}

async fn materialize_image_attachment(attachment: &MessageAttachment) -> Result<MessageAttachment> {
    use crate::services::image_optimize::{optimize_base64, optimize_image};
    let mime = attachment.mime_type.clone();
    let task = match &attachment.storage {
        AttachmentStorage::Inline { data } => {
            let data = data.clone();
            tokio::task::spawn_blocking(move || match optimize_base64(&data) {
                Some((mime_type, data)) => (mime_type, data),
                None => (mime, data),
            })
        }
        AttachmentStorage::FileRef { path } => {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|err| anyhow::anyhow!("Failed to read '{}': {err}", path))?;
            tokio::task::spawn_blocking(move || match optimize_image(&bytes) {
                Some(image) => (image.mime_type, BASE64.encode(image.bytes)),
                None => (mime, BASE64.encode(bytes)),
            })
        }
    };
    let (mime_type, data) = task
        .await
        .map_err(|err| anyhow::anyhow!("image processing failed: {err}"))?;
    Ok(MessageAttachment {
        name: attachment.name.clone(),
        mime_type,
        storage: AttachmentStorage::Inline { data },
    })
}

/// Returns `(stored_messages, title, preview)` for a completed one-shot
/// turn. The session_id is generated by the caller and threaded through both
/// `log_chat_turn` (so logs.db rows carry the session linkage that
/// `aivo logs share <event-id>` needs) and `save_code_session_with_id`.
fn build_one_shot_persist_inputs(
    user_history: &[ChatMessage],
    assistant_content: String,
    assistant_reasoning: Option<String>,
    raw_model: &str,
) -> (Vec<StoredChatMessage>, String, String) {
    let mut full_history = user_history.to_vec();
    full_history.push(ChatMessage {
        model: (!raw_model.is_empty()).then(|| raw_model.to_string()),
        role: "assistant".to_string(),
        content: assistant_content,
        reasoning_content: assistant_reasoning.and_then(normalize_reasoning_content),
        attachments: vec![],
    });
    let stored = to_stored_messages(&full_history);
    let title = code_tui::session_title_from_messages(&full_history, raw_model);
    let preview = code_tui::session_preview_text_from_messages(&full_history, raw_model);
    (stored, title, preview)
}

fn to_stored_messages(history: &[ChatMessage]) -> Vec<StoredChatMessage> {
    history
        .iter()
        .map(|message| StoredChatMessage {
            model: message.model.clone(),
            role: message.role.clone(),
            content: message.content.clone(),
            reasoning_content: message.reasoning_content.clone(),
            id: Some(new_code_session_id()),
            timestamp: Some(Utc::now().to_rfc3339()),
            attachments: (!message.attachments.is_empty()).then(|| message.attachments.clone()),
        })
        .collect()
}

pub(crate) fn new_code_session_id() -> String {
    let mut bytes = [0u8; 16];
    crate::services::rng::fill(&mut bytes);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u16::from_be_bytes([bytes[4], bytes[5]]),
        u16::from_be_bytes([bytes[6], bytes[7]]),
        u16::from_be_bytes([bytes[8], bytes[9]]),
        u64::from_be_bytes([
            0, 0, bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ]),
    )
}

fn should_retry_status(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
}

fn should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request() || err.is_body()
}

fn retry_delay(attempt: usize, retry_after: Option<&reqwest::header::HeaderValue>) -> Duration {
    if let Some(seconds) = retry_after
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        return Duration::from_secs(seconds.min(30));
    }
    let exp = 250u64.saturating_mul(1u64 << (attempt.saturating_sub(1).min(4)));
    Duration::from_millis(exp.min(4000))
}

#[allow(clippy::too_many_arguments)]
async fn log_chat_turn(
    session_store: &SessionStore,
    key: &ApiKey,
    raw_model: &str,
    cwd: Option<&str>,
    session_id: Option<&str>,
    user_message: &ChatMessage,
    assistant_content: &str,
    reasoning_content: Option<&str>,
    usage: &TokenUsage,
) -> Result<()> {
    let attachments = user_message
        .attachments
        .iter()
        .map(|attachment| {
            serde_json::json!({
                "name": attachment.name,
                "mime_type": attachment.mime_type,
                "storage": attachment_storage_label(&attachment.storage),
            })
        })
        .collect::<Vec<_>>();

    session_store
        .logs()
        .append(crate::services::log_store::LogEvent {
            source: "code".to_string(),
            kind: "code_turn".to_string(),
            key_id: Some(key.id.clone()),
            key_name: Some(key.display_name().to_string()),
            base_url: Some(key.base_url.clone()),
            tool: Some("code".to_string()),
            model: Some(raw_model.to_string()),
            cwd: cwd.map(str::to_string),
            session_id: session_id.map(str::to_string),
            input_tokens: Some(usage.prompt_tokens as i64),
            output_tokens: Some(usage.completion_tokens as i64),
            cache_read_input_tokens: Some(usage.cache_read_input_tokens as i64),
            cache_creation_input_tokens: Some(usage.cache_creation_input_tokens as i64),
            // A skill invocation's content is the whole inlined SKILL.md body;
            // log the compact `/name args` the user typed (so `aivo logs` shows
            // the real input, e.g. `/baidu-search 歌曲`) and keep the full
            // expanded body in the searchable body_text for debugging.
            title: Some(log_title(
                &code_tui::skill_invocation_label(&user_message.content)
                    .unwrap_or_else(|| user_message.content.clone()),
            )),
            body_text: Some(format!(
                "User:\n{}\n\nAssistant:\n{}",
                user_message.content, assistant_content
            )),
            payload_json: Some(serde_json::json!({
                "user": {
                    "content": user_message.content,
                    "attachments": attachments,
                },
                "assistant": {
                    "content": assistant_content,
                    "reasoning_content": reasoning_content,
                }
            })),
            ..Default::default()
        })
        .await?;
    Ok(())
}

fn log_title(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(empty turn)");
    let mut truncated = line.chars().take(120).collect::<String>();
    if line.chars().count() > 120 {
        truncated.push_str("...");
    }
    truncated
}

fn attachment_storage_label(storage: &AttachmentStorage) -> &'static str {
    match storage {
        AttachmentStorage::Inline { .. } => "inline",
        AttachmentStorage::FileRef { .. } => "file_ref",
    }
}

fn direct_base_url(key: &ApiKey) -> String {
    let resolved = resolve_starter_base_url(&key.base_url);
    normalize_base_url(&resolved).to_string()
}

/// Conditionally adds auth headers to a request. Skips when the key is empty
/// (e.g. the free aivo starter provider needs no authentication).
fn with_auth(builder: reqwest::RequestBuilder, key: &ApiKey) -> reqwest::RequestBuilder {
    if key.key.is_empty() {
        crate::services::device_fingerprint::with_starter_headers(builder)
    } else {
        builder.header("Authorization", format!("Bearer {}", key.key.as_str()))
    }
}

/// Adds native Anthropic auth headers. Some Anthropic-compatible gateways reject
/// a simultaneous OpenAI-style Authorization bearer, so send only `x-api-key`.
fn with_auth_anthropic(builder: reqwest::RequestBuilder, key: &ApiKey) -> reqwest::RequestBuilder {
    if key.key.is_empty() {
        crate::services::device_fingerprint::with_starter_headers(builder)
    } else {
        builder.header("x-api-key", key.key.as_str())
    }
}

/// Adds the `x-goog-api-key` header for Google Gemini native API.
fn with_auth_google(builder: reqwest::RequestBuilder, key: &ApiKey) -> reqwest::RequestBuilder {
    if key.key.is_empty() {
        crate::services::device_fingerprint::with_starter_headers(builder)
    } else {
        builder.header("x-goog-api-key", key.key.as_str())
    }
}

async fn send_with_retry<F>(mut build_request: F) -> Result<reqwest::Response>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=MAX_REQUEST_ATTEMPTS {
        match build_request().send_logged().await {
            Ok(response) => {
                if should_retry_status(response.status()) && attempt < MAX_REQUEST_ATTEMPTS {
                    let delay = retry_delay(
                        attempt,
                        response.headers().get(reqwest::header::RETRY_AFTER),
                    );
                    let _ = response.bytes().await;
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Ok(response);
            }
            Err(err) => {
                if should_retry_error(&err) && attempt < MAX_REQUEST_ATTEMPTS {
                    tokio::time::sleep(retry_delay(attempt, None)).await;
                    continue;
                }
                last_err = Some(err.into());
                break;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Request failed")))
}

/// The request's injected `reasoning_effort` slot (chat or Responses shape).
fn effort_field_mut(request: &mut serde_json::Value) -> Option<&mut serde_json::Value> {
    if request.get("reasoning_effort").is_some() {
        return request.get_mut("reasoning_effort");
    }
    request
        .get_mut("reasoning")
        .and_then(|r| r.get_mut("effort"))
}

/// [`send_with_retry`] plus a one-shot heal for a 400 rejecting the injected
/// `reasoning_effort`: stale catalogs advertise levels some routes refuse, so
/// retry once with the error's own lowest allowed level.
async fn send_with_effort_heal<F>(
    request: &mut serde_json::Value,
    mut build_request: F,
) -> Result<reqwest::Response>
where
    F: FnMut(&serde_json::Value) -> reqwest::RequestBuilder,
{
    let mut healed = false;
    loop {
        let response = send_with_retry(|| build_request(request)).await?;
        if healed
            || response.status() != reqwest::StatusCode::BAD_REQUEST
            || effort_field_mut(request).is_none()
        {
            return Ok(response);
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        match crate::services::model_metadata::effort_rejection_fallback(&body) {
            Some(level) => {
                *effort_field_mut(request).expect("checked above") = serde_json::json!(level);
                healed = true;
            }
            // Body consumed — surface the same error the caller would have.
            None => anyhow::bail!("API returned {} — {}", status, body),
        }
    }
}

/// Sends a chat completion request and prints the response.
/// Tries streaming first; falls back to non-streaming if the server returns a 5xx error.
/// Returns the full assistant message content.
#[allow(clippy::too_many_arguments)]
async fn send_chat_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let base = direct_base_url(key);
    let url = format!("{}/v1/chat/completions", base);

    if non_streaming {
        return send_non_streaming(client, &url, key, model, messages, spinning, on_chunk).await;
    }

    // Try streaming first; fall back to non-streaming on server errors
    let mut request = build_openai_chat_request(model, messages, true, None)?;

    let mut response = send_with_effort_heal(&mut request, |req| {
        with_auth(client.post(&url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(req)
    })
    .await?;

    // If the server can't handle streaming, fall back to non-streaming.
    // Note: 404 is NOT included here — it means wrong endpoint, not streaming unsupported.
    // The caller detects 404 and switches to a different API format instead.
    if response.status().is_server_error() {
        return send_non_streaming(client, &url, key, model, messages, spinning, on_chunk).await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut usage = None;
    let mut response_model: Option<String> = None;
    let mut line_buf = SseLineBuffer::new();
    let mut done = false;

    while !done {
        let chunk_result = response.chunk().await;
        let Some(chunk) = (match chunk_result {
            Ok(c) => c,
            Err(_) if !full_content.is_empty() || !full_reasoning.is_empty() => {
                // Stream error after content was received — use what we have.
                break;
            }
            Err(e) => return Err(e.into()),
        }) else {
            break;
        };
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                if let Some(tokens) = parse_openai_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                capture_model(&mut response_model, parse_openai_model_chunk, data);
                if let Some(chunk) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(content) => full_content.push_str(content),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                        ChatResponseChunk::Usage(_) => {}
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                merge_token_usage(&mut usage, tokens);
            }
            capture_model(&mut response_model, parse_openai_model_chunk, data);
            if data.trim() != "[DONE]"
                && let Some(chunk) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                match &chunk {
                    ChatResponseChunk::Content(content) => full_content.push_str(content),
                    ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                    ChatResponseChunk::Usage(_) => {}
                }
                on_chunk(chunk)?;
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            response_model = response_model.or_else(|| extract_response_model(&resp));
            let response = extract_openai_message(&resp);
            if !response.content.is_empty() || response.reasoning_content.is_some() {
                style::stop_spinner(spinning);
                if let Some(reasoning) = response.reasoning_content.clone() {
                    on_chunk(ChatResponseChunk::Reasoning(reasoning.clone()))?;
                    full_reasoning = reasoning;
                }
                if !response.content.is_empty() {
                    on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
                    full_content = response.content;
                }
            }
        }
    }

    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_non_streaming(client, &url, key, model, messages, spinning, on_chunk).await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        model: response_model,
        raw_body: None,
    })
}

/// Non-streaming fallback for gateways that don't support SSE streaming.
async fn send_non_streaming<F>(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let mut request = build_openai_chat_request(model, messages, false, None)?;

    let response = send_with_effort_heal(&mut request, |req| {
        with_auth(client.post(url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(req)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    emit_openai_chat_value(body, spinning, on_chunk)
}

fn emit_openai_chat_value<F>(
    body: serde_json::Value,
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let response = extract_openai_message(&body);
    let usage = extract_openai_usage(&body);
    let response_model = extract_response_model(&body);

    if response.content.is_empty() && response.reasoning_content.is_none() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    if let Some(reasoning) = response.reasoning_content.clone() {
        on_chunk(ChatResponseChunk::Reasoning(reasoning))?;
    }
    if !response.content.is_empty() {
        on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
    }

    Ok(ChatTurnResult {
        content: response.content,
        usage,
        model: response_model,
        raw_body: Some(body),
    })
}

/// Finalize a non-streaming Responses-API body into a turn result: extract the
/// message / usage / model, fail on empty, emit the content, and carry the raw
/// body for `--json`. Shared by the direct and Copilot Responses senders (the
/// Responses counterpart to [`emit_openai_chat_value`]).
fn emit_responses_chat_value<F>(
    body: serde_json::Value,
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let response = extract_responses_message(&body);
    let usage = extract_responses_usage(&body);
    let response_model = extract_response_model(&body);

    if response.content.is_empty() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    on_chunk(ChatResponseChunk::Content(response.content.clone()))?;

    Ok(ChatTurnResult {
        content: response.content,
        usage,
        model: response_model,
        raw_body: Some(body),
    })
}

/// Sends a chat request via GitHub Copilot (token exchange + Copilot API).
#[allow(clippy::too_many_arguments)]
async fn send_copilot_request<F>(
    client: &Client,
    tm: &CopilotTokenManager,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let (copilot_token, api_endpoint) = tm.get_token().await?;
    let url = format!("{}/chat/completions", api_endpoint.trim_end_matches('/'));

    if non_streaming {
        return send_copilot_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    let mut request = build_openai_chat_request(model, messages, true, None)?;
    let initiator = copilot_initiator_from_openai(&request);

    let mut response = send_with_effort_heal(&mut request, |req| {
        client
            .post(&url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .header(COPILOT_INITIATOR_HEADER, initiator)
            .json(req)
    })
    .await?;

    if response.status().is_server_error() {
        return send_copilot_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut usage = None;
    let mut response_model: Option<String> = None;
    let mut line_buf = SseLineBuffer::new();
    let mut done = false;

    while !done {
        let chunk_result = response.chunk().await;
        let Some(chunk) = (match chunk_result {
            Ok(c) => c,
            Err(_) if !full_content.is_empty() || !full_reasoning.is_empty() => {
                // Stream error after content was received — use what we have.
                break;
            }
            Err(e) => return Err(e.into()),
        }) else {
            break;
        };
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                if let Some(tokens) = parse_openai_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                capture_model(&mut response_model, parse_openai_model_chunk, data);
                if let Some(chunk) = parse_sse_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(content) => full_content.push_str(content),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                        ChatResponseChunk::Usage(_) => {}
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty() {
        if let Some(data) = sse_data_payload(tail) {
            if let Some(tokens) = parse_openai_usage_chunk(data) {
                merge_token_usage(&mut usage, tokens);
            }
            capture_model(&mut response_model, parse_openai_model_chunk, data);
            if data.trim() != "[DONE]"
                && let Some(chunk) = parse_sse_chunk(data)
            {
                style::stop_spinner(spinning);
                match &chunk {
                    ChatResponseChunk::Content(content) => full_content.push_str(content),
                    ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                    ChatResponseChunk::Usage(_) => {}
                }
                on_chunk(chunk)?;
            }
        } else if full_content.is_empty()
            && let Ok(resp) = serde_json::from_str::<serde_json::Value>(tail)
        {
            response_model = response_model.or_else(|| extract_response_model(&resp));
            let response = extract_openai_message(&resp);
            if !response.content.is_empty() || response.reasoning_content.is_some() {
                style::stop_spinner(spinning);
                if let Some(reasoning) = response.reasoning_content.clone() {
                    on_chunk(ChatResponseChunk::Reasoning(reasoning.clone()))?;
                    full_reasoning = reasoning;
                }
                if !response.content.is_empty() {
                    on_chunk(ChatResponseChunk::Content(response.content.clone()))?;
                    full_content = response.content;
                }
            }
        }
    }

    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_copilot_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        model: response_model,
        raw_body: None,
    })
}

async fn send_copilot_non_streaming<F>(
    client: &Client,
    url: &str,
    copilot_token: &str,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let mut request = build_openai_chat_request(model, messages, false, None)?;
    let initiator = copilot_initiator_from_openai(&request);

    let response = send_with_effort_heal(&mut request, |req| {
        client
            .post(url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .header(COPILOT_INITIATOR_HEADER, initiator)
            .json(req)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    emit_openai_chat_value(body, spinning, on_chunk)
}

/// Sends a chat request via GitHub Copilot using the Responses API.
#[allow(clippy::too_many_arguments)]
async fn send_copilot_responses_request<F>(
    client: &Client,
    tm: &CopilotTokenManager,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let (copilot_token, api_endpoint) = tm.get_token().await?;
    let url = format!("{}/responses", api_endpoint.trim_end_matches('/'));

    if non_streaming {
        return send_copilot_responses_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    let mut request = build_responses_request(model, messages, true)?;

    let mut response = send_with_effort_heal(&mut request, |req| {
        client
            .post(&url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .json(req)
    })
    .await?;

    if response.status().is_server_error() {
        return send_copilot_responses_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut response_model: Option<String> = None;
    let mut line_buf = SseLineBuffer::new();

    while let Some(chunk) = response.chunk().await? {
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if let Some(tokens) = parse_responses_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                capture_model(&mut response_model, parse_responses_model_chunk, data);
                if let Some(chunk) = parse_responses_chunk(data) {
                    style::stop_spinner(spinning);
                    if let ChatResponseChunk::Content(ref content) = chunk {
                        full_content.push_str(content);
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty()
        && let Some(data) = sse_data_payload(tail)
    {
        if let Some(tokens) = parse_responses_usage_chunk(data) {
            merge_token_usage(&mut usage, tokens);
        }
        capture_model(&mut response_model, parse_responses_model_chunk, data);
        if let Some(chunk) = parse_responses_chunk(data) {
            style::stop_spinner(spinning);
            if let ChatResponseChunk::Content(ref content) = chunk {
                full_content.push_str(content);
            }
            on_chunk(chunk)?;
        }
    }

    if full_content.is_empty() {
        return send_copilot_responses_non_streaming(
            client,
            &url,
            &copilot_token,
            model,
            messages,
            spinning,
            on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        model: response_model,
        raw_body: None,
    })
}

/// Non-streaming fallback for Copilot Responses API.
async fn send_copilot_responses_non_streaming<F>(
    client: &Client,
    url: &str,
    copilot_token: &str,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let mut request = build_responses_request(model, messages, false)?;

    let response = send_with_effort_heal(&mut request, |req| {
        client
            .post(url)
            .header("Authorization", format!("Bearer {}", copilot_token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .json(req)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    emit_responses_chat_value(body, spinning, on_chunk)
}

/// Returns true when the error indicates the endpoint doesn't exist,
/// meaning we should try a different API format.
fn is_format_mismatch(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("404")
        || msg.contains("405")
        || (msg.contains("not found")
            && (msg.contains("endpoint") || msg.contains("route") || msg.contains("path")))
        || (msg.contains("method not allowed")
            && (msg.contains("endpoint") || msg.contains("route") || msg.contains("path")))
        || crate::services::provider_protocol::is_format_unsupported_error(&msg)
}

/// True when a failed `/chat/completions` attempt should be retried on the
/// Responses API: an explicit "use /responses" signal, or a bare 400 that might
/// just be the wrong endpoint. A 400 carrying a structured error envelope (e.g.
/// "does not support image inputs") is a real rejection — retrying would mask it
/// behind a 404 from a `/responses` path the provider may not even serve.
fn is_responses_api_required(e: &anyhow::Error) -> bool {
    let msg = e.to_string();
    let lower = msg.to_lowercase();
    if lower.contains("unsupported_api_for_model")
        || (lower.contains("not accessible") && lower.contains("/chat/completions"))
    {
        return true;
    }
    lower.contains("400 bad request")
        && !crate::services::provider_protocol::is_request_error_envelope(error_response_body(&msg))
}

/// The `{body}` half of an `API returned {status} — {body}` sender error.
fn error_response_body(msg: &str) -> &str {
    msg.split_once(" — ").map(|(_, body)| body).unwrap_or(msg)
}

/// Sends a chat request via the OpenAI Responses API (/v1/responses).
/// Tries streaming first; falls back to non-streaming on server errors.
#[allow(clippy::too_many_arguments)]
async fn send_responses_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let base = direct_base_url(key);
    let url = format!("{}/v1/responses", base);

    if non_streaming {
        return send_responses_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    let mut request = build_responses_request(model, messages, true)?;

    let mut response = send_with_effort_heal(&mut request, |req| {
        with_auth(client.post(&url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(req)
    })
    .await?;

    if response.status().is_server_error() {
        return send_responses_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut response_model: Option<String> = None;
    let mut line_buf = SseLineBuffer::new();

    while let Some(chunk) = response.chunk().await? {
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if let Some(tokens) = parse_responses_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                capture_model(&mut response_model, parse_responses_model_chunk, data);
                if let Some(chunk) = parse_responses_chunk(data) {
                    style::stop_spinner(spinning);
                    if let ChatResponseChunk::Content(ref content) = chunk {
                        full_content.push_str(content);
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty()
        && let Some(data) = sse_data_payload(tail)
    {
        if let Some(tokens) = parse_responses_usage_chunk(data) {
            merge_token_usage(&mut usage, tokens);
        }
        capture_model(&mut response_model, parse_responses_model_chunk, data);
        if let Some(chunk) = parse_responses_chunk(data) {
            style::stop_spinner(spinning);
            if let ChatResponseChunk::Content(ref content) = chunk {
                full_content.push_str(content);
            }
            on_chunk(chunk)?;
        }
    }

    if full_content.is_empty() {
        return send_responses_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        model: response_model,
        raw_body: None,
    })
}

/// Non-streaming fallback for OpenAI Responses API.
async fn send_responses_non_streaming<F>(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let mut request = build_responses_request(model, messages, false)?;

    let response = send_with_effort_heal(&mut request, |req| {
        with_auth(client.post(url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(req)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    emit_responses_chat_value(body, spinning, on_chunk)
}

/// Sends a request using Anthropic's native /v1/messages API.
/// Tries streaming first; falls back to non-streaming on server errors.
#[allow(clippy::too_many_arguments)]
async fn send_anthropic_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let base = direct_base_url(key);
    let url = format!("{}/v1/messages", base);

    if non_streaming {
        return send_anthropic_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    let request = build_anthropic_request(model, messages, true)?;

    let mut response = send_with_retry(|| {
        with_auth_anthropic(client.post(&url), key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&request)
    })
    .await?;

    if response.status().is_server_error() || response.status() == reqwest::StatusCode::NOT_FOUND {
        return send_anthropic_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut usage = None;
    let mut response_model: Option<String> = None;
    let mut line_buf = SseLineBuffer::new();

    while let Some(chunk) = response.chunk().await? {
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if let Some(tokens) = parse_anthropic_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                capture_model(&mut response_model, parse_anthropic_model_chunk, data);
                if let Some(chunk) = parse_anthropic_chunk(data) {
                    style::stop_spinner(spinning);
                    match &chunk {
                        ChatResponseChunk::Content(text) => full_content.push_str(text),
                        ChatResponseChunk::Reasoning(reasoning) => {
                            full_reasoning.push_str(reasoning);
                        }
                        ChatResponseChunk::Usage(_) => {}
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    // Drain a final, non-newline-terminated SSE line left in the buffer (a
    // truncated stream). Process it whenever present — NOT only when no content
    // arrived — so the last fragment of a truncated reply isn't dropped. The
    // other five senders already do this; gating on `full_content.is_empty()`
    // here was the lone divergence and silently lost that fragment.
    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty()
        && let Some(data) = sse_data_payload(tail)
    {
        if let Some(tokens) = parse_anthropic_usage_chunk(data) {
            merge_token_usage(&mut usage, tokens);
        }
        capture_model(&mut response_model, parse_anthropic_model_chunk, data);
        if let Some(chunk) = parse_anthropic_chunk(data) {
            style::stop_spinner(spinning);
            match &chunk {
                ChatResponseChunk::Content(text) => full_content.push_str(text),
                ChatResponseChunk::Reasoning(reasoning) => full_reasoning.push_str(reasoning),
                ChatResponseChunk::Usage(_) => {}
            }
            on_chunk(chunk)?;
        }
    }

    // If streaming produced no content, fall back to non-streaming
    if full_content.is_empty() && full_reasoning.is_empty() {
        return send_anthropic_non_streaming(
            client, &url, key, model, messages, spinning, on_chunk,
        )
        .await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        model: response_model,
        raw_body: None,
    })
}

/// Non-streaming fallback for Anthropic-format providers.
async fn send_anthropic_non_streaming<F>(
    client: &Client,
    url: &str,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    let request = build_anthropic_request(model, messages, false)?;

    let response = send_with_retry(|| {
        with_auth_anthropic(client.post(url), key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&request)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let usage = extract_anthropic_usage(&body);
    let response_model = extract_response_model(&body);

    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    for block in body["content"].as_array().into_iter().flatten() {
        match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    content_parts.push(text.to_string());
                }
            }
            "thinking" => {
                if let Some(reasoning) = block
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .or_else(|| block.get("text").and_then(|v| v.as_str()))
                {
                    reasoning_parts.push(reasoning.to_string());
                }
            }
            _ => {}
        }
    }

    let content = content_parts.concat();
    let reasoning_content = normalize_reasoning_content(reasoning_parts.join(""));

    if content.is_empty() && reasoning_content.is_none() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    if !content.is_empty() {
        on_chunk(ChatResponseChunk::Content(content.clone()))?;
    }

    Ok(ChatTurnResult {
        content,
        usage,
        model: response_model,
        raw_body: Some(body),
    })
}

/// Sends a request using Google Gemini's native API with streaming.
/// Falls back to non-streaming on server errors.
#[allow(clippy::too_many_arguments)]
async fn send_google_request<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    non_streaming: bool,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    use crate::services::model_names::google_native_model_name;
    use crate::services::openai_gemini_bridge::build_google_stream_generate_content_url;

    if non_streaming {
        return send_google_non_streaming(client, key, model, messages, spinning, on_chunk).await;
    }

    let native_model = google_native_model_name(model);
    let url = build_google_stream_generate_content_url(&key.base_url, native_model);

    let request = build_google_request(model, messages)?;

    let mut response = send_with_retry(|| {
        with_auth_google(client.post(&url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&request)
    })
    .await?;

    if response.status().is_server_error() {
        return send_google_non_streaming(client, key, model, messages, spinning, on_chunk).await;
    }

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let mut full_content = String::new();
    let mut usage = None;
    let mut line_buf = SseLineBuffer::new();

    while let Some(chunk) = response.chunk().await? {
        for line in line_buf.push_chunk(&chunk)? {
            if let Some(data) = sse_data_payload(&line) {
                if let Some(tokens) = parse_google_usage_chunk(data) {
                    merge_token_usage(&mut usage, tokens);
                    if let Some(usage) = usage {
                        on_chunk(ChatResponseChunk::Usage(usage))?;
                    }
                }
                if let Some(chunk) = parse_google_chunk(data) {
                    style::stop_spinner(spinning);
                    if let ChatResponseChunk::Content(ref content) = chunk {
                        full_content.push_str(content);
                    }
                    on_chunk(chunk)?;
                }
            }
        }
    }

    // Process any remaining data in the buffer
    let tail_str = line_buf.take_tail().unwrap_or_default();
    let tail = tail_str.trim();
    if !tail.is_empty()
        && let Some(data) = sse_data_payload(tail)
    {
        if let Some(tokens) = parse_google_usage_chunk(data) {
            merge_token_usage(&mut usage, tokens);
        }
        if let Some(chunk) = parse_google_chunk(data) {
            style::stop_spinner(spinning);
            if let ChatResponseChunk::Content(ref content) = chunk {
                full_content.push_str(content);
            }
            on_chunk(chunk)?;
        }
    }

    // If streaming produced no content, fall back to non-streaming
    if full_content.is_empty() {
        return send_google_non_streaming(client, key, model, messages, spinning, on_chunk).await;
    }

    Ok(ChatTurnResult {
        content: full_content,
        usage,
        // Google's stream chunks don't include the model name; the request
        // model was already substituted into the URL. Recording falls back
        // to the user-typed alias.
        model: None,
        raw_body: None,
    })
}

/// Non-streaming fallback for Google Gemini native API.
async fn send_google_non_streaming<F>(
    client: &Client,
    key: &ApiKey,
    model: &str,
    messages: &[ChatMessage],
    spinning: &Arc<AtomicBool>,
    on_chunk: &mut F,
) -> Result<ChatTurnResult>
where
    F: FnMut(ChatResponseChunk) -> Result<()>,
{
    use crate::services::model_names::google_native_model_name;
    use crate::services::openai_gemini_bridge::build_google_generate_content_url;

    let native_model = google_native_model_name(model);
    let url = build_google_generate_content_url(&key.base_url, native_model);

    let request = build_google_request(model, messages)?;

    let response = send_with_retry(|| {
        with_auth_google(client.post(&url), key)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&request)
    })
    .await?;

    if !response.status().is_success() {
        style::stop_spinner(spinning);
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("API returned {} — {}", status, body);
    }

    let body: serde_json::Value = response.json().await?;
    let google_response = extract_google_message(&body);
    let usage = extract_google_usage(&body);
    let response_model = extract_google_model(&body);

    if google_response.content.is_empty() {
        style::stop_spinner(spinning);
        anyhow::bail!("Provider returned an empty response");
    }

    style::stop_spinner(spinning);
    on_chunk(ChatResponseChunk::Content(google_response.content.clone()))?;

    Ok(ChatTurnResult {
        content: google_response.content,
        usage,
        model: response_model,
        raw_body: Some(body),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn headless_limits_require_exec_mode() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let command = CodeCommand::new(store, cache);

        let code = command
            .execute(
                None,
                Some("hi".to_string()),
                None,
                Vec::new(),
                false,
                None,
                false,
                false,
                None,
                None,
                false,
                false,
                false,
                None,
                Some(1),
                None,
                None,
                false,
                None,
                None,
            )
            .await;

        assert_eq!(code, ExitCode::UserError);
    }

    #[test]
    fn build_one_shot_persist_inputs_includes_assistant_turn() {
        let user_history = vec![ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "hello world".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }];
        let (stored, title, preview) = build_one_shot_persist_inputs(
            &user_history,
            "hi there".to_string(),
            Some("brief greeting".to_string()),
            "gpt-test",
        );

        assert_eq!(stored.len(), 2, "user message + assistant response stored");
        assert_eq!(stored[0].role, "user");
        assert_eq!(stored[0].content, "hello world");
        assert_eq!(stored[1].role, "assistant");
        assert_eq!(stored[1].content, "hi there");
        assert_eq!(
            stored[1].reasoning_content.as_deref(),
            Some("brief greeting")
        );

        // Title comes from the user prompt (first non-empty line); preview
        // composes the last two non-empty messages. Both must be derived
        // from history, not fall back to the raw model name.
        assert_eq!(title, "hello world");
        assert!(
            preview.contains("hello world") && preview.contains("hi there"),
            "preview should compose user + assistant snippets, got: {preview}"
        );
    }

    #[test]
    fn build_one_shot_persist_inputs_drops_empty_reasoning() {
        let user_history = vec![ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "ping".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }];
        let (stored, _, _) = build_one_shot_persist_inputs(
            &user_history,
            "pong".to_string(),
            Some("   \n\t  ".to_string()),
            "gpt-test",
        );
        assert_eq!(stored[1].reasoning_content, None);
    }

    #[test]
    fn test_compose_one_shot_prompt_without_stdin() {
        let out = compose_one_shot_prompt("Summarize in one sentence", None);
        assert_eq!(out, "Summarize in one sentence");
    }

    #[test]
    fn test_friendly_base_url_resolves_sentinels() {
        // Sentinel base URLs must not leak into the --dry-run preview.
        assert_eq!(friendly_base_url("copilot"), "GitHub Copilot");
        assert_eq!(friendly_base_url("ollama"), "Ollama (local)");
        assert_eq!(
            friendly_base_url(crate::constants::AIVO_STARTER_SENTINEL),
            crate::constants::AIVO_STARTER_REAL_URL
        );
        // A real URL passes through untouched.
        assert_eq!(
            friendly_base_url("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1"
        );
    }

    #[test]
    fn test_compose_one_shot_prompt_with_stdin_context() {
        let out = compose_one_shot_prompt("Summarize in one sentence", Some("diff --git a b"));
        assert!(out.contains("Summarize in one sentence"));
        assert!(out.contains("Context from stdin:"));
        assert!(out.contains("diff --git a b"));
    }

    #[test]
    fn test_compose_one_shot_prompt_ignores_empty_stdin() {
        let out = compose_one_shot_prompt("Summarize", Some("   \n  "));
        assert_eq!(out, "Summarize");
    }

    #[test]
    fn test_sanitize_one_shot_message_rejects_whitespace() {
        let err = sanitize_one_shot_message(" \n\t ".to_string()).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_sanitize_one_shot_message_preserves_content() {
        let out = sanitize_one_shot_message("hello\nworld\n".to_string()).unwrap();
        assert_eq!(out, "hello\nworld\n");
    }

    #[test]
    fn test_should_retry_status() {
        assert!(should_retry_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(should_retry_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(!should_retry_status(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_with_auth_anthropic_uses_x_api_key_only() {
        let key = ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            "https://opencode.ai/zen/go/v1".to_string(),
            None,
            "sk-test".to_string(),
        );
        let request = with_auth_anthropic(
            reqwest::Client::new().post("http://127.0.0.1/v1/messages"),
            &key,
        )
        .build()
        .unwrap();
        let headers = request.headers();

        assert_eq!(headers.get("x-api-key").unwrap(), "sk-test");
        assert!(headers.get(reqwest::header::AUTHORIZATION).is_none());
    }

    /// End-to-end smoke of the OpenAI streaming sender against a fake SSE server,
    /// exercising the trailing-tail path: the final `data:` line is NOT
    /// newline-terminated, so it stays in the leftover buffer and must be drained
    /// by the post-loop tail handling (content + usage), not the main loop. Also
    /// covers model capture and cross-chunk content accumulation. (The senders
    /// otherwise had no end-to-end coverage — only the parse helpers did.)
    #[tokio::test]
    async fn send_chat_request_accumulates_stream_including_trailing_tail() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::AtomicBool;

        let body = "data: {\"model\":\"deepseek-v4\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });

        let key = ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            format!("http://127.0.0.1:{port}"),
            None,
            "sk-test".to_string(),
        );
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let spinning = Arc::new(AtomicBool::new(false));
        let messages = [ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "hi".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }];
        let mut streamed = String::new();
        let result = send_chat_request(
            &client,
            &key,
            "deepseek-v4",
            &messages,
            &spinning,
            false,
            &mut |chunk| {
                if let ChatResponseChunk::Content(c) = &chunk {
                    streamed.push_str(c);
                }
                Ok(())
            },
        )
        .await
        .unwrap();

        // " world" (tail line) is folded in alongside the loop's "Hello", and the
        // tail also carried the usage — proving the leftover-buffer path runs.
        assert_eq!(result.content, "Hello world");
        assert_eq!(streamed, "Hello world");
        assert_eq!(result.model.as_deref(), Some("deepseek-v4"));
        assert_eq!(result.usage.map(|u| u.total_tokens()), Some(12));
    }

    /// Regression: a truncated Anthropic stream whose final content_block_delta
    /// lacks a trailing newline (arriving AFTER content already streamed) must
    /// still have that fragment recovered from the leftover buffer. It used to be
    /// dropped because the tail was gated on `full_content.is_empty()` — the lone
    /// sender that diverged from the others' `!tail.is_empty()`.
    #[tokio::test]
    async fn send_anthropic_request_recovers_trailing_content_fragment() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::AtomicBool;

        let body = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude-x\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });

        let key = ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            format!("http://127.0.0.1:{port}"),
            None,
            "sk-test".to_string(),
        );
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let spinning = Arc::new(AtomicBool::new(false));
        let messages = [ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "hi".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }];
        let result = send_anthropic_request(
            &client,
            &key,
            "claude-x",
            &messages,
            &spinning,
            false,
            &mut |_chunk| Ok(()),
        )
        .await
        .unwrap();

        // " world" (the un-terminated tail line) is recovered even though "Hello"
        // already streamed — pre-fix the tail was skipped once content existed.
        assert_eq!(result.content, "Hello world");
        assert_eq!(result.model.as_deref(), Some("claude-x"));
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "hello".to_string(),
            reasoning_content: Some("hidden".to_string()),
            attachments: vec![],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));
        assert!(!json.contains("reasoning_content"));
    }

    #[test]
    fn test_is_format_mismatch_404() {
        let e = anyhow::anyhow!("API returned 404 Not Found — endpoint missing");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_405() {
        let e = anyhow::anyhow!("API returned 405 Method Not Allowed");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_endpoint_text() {
        let e = anyhow::anyhow!("route not found for requested endpoint");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_unsupported_format_wording() {
        // A 401 status whose body is really a format-routing error, not auth.
        let e = anyhow::anyhow!(
            r#"API returned 401 Unauthorized — {{"type":"error","error":{{"type":"ModelError","message":"Model qwen3.7-max is not supported for format oa-compat"}}}}"#
        );
        assert!(is_format_mismatch(&e));
        let e = anyhow::anyhow!("API returned 403 Forbidden — unsupported format");
        assert!(is_format_mismatch(&e));
        let e = anyhow::anyhow!("API returned 400 Bad Request — unsupported format oa-compat");
        assert!(is_format_mismatch(&e));
    }

    #[test]
    fn test_is_format_mismatch_other_errors() {
        let e = anyhow::anyhow!("API returned 401 Unauthorized");
        assert!(!is_format_mismatch(&e));
        let e = anyhow::anyhow!("API returned 429 Too Many Requests");
        assert!(!is_format_mismatch(&e));
    }

    #[test]
    fn test_is_responses_api_required_unsupported_code() {
        let e = anyhow::anyhow!(
            r#"API returned 400 Bad Request — {{"error":{{"message":"model \"gpt-5.4\" is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}}}"#
        );
        assert!(is_responses_api_required(&e));
    }

    #[test]
    fn test_is_responses_api_required_not_accessible() {
        let e = anyhow::anyhow!("model is not accessible via the /chat/completions endpoint");
        assert!(is_responses_api_required(&e));
    }

    #[test]
    fn test_is_responses_api_required_generic_400() {
        let e = anyhow::anyhow!("API returned 400 Bad Request — invalid something");
        assert!(is_responses_api_required(&e));
    }

    #[test]
    fn test_is_responses_api_required_unrelated_error() {
        let e = anyhow::anyhow!("API returned 401 Unauthorized");
        assert!(!is_responses_api_required(&e));
    }

    #[test]
    fn test_is_responses_api_required_skips_structured_400() {
        // A 400 with an error envelope is a real rejection (here: no image
        // support) — surface it, don't retry on /responses. The opencode-zen bug.
        let e = anyhow::anyhow!(
            r#"API returned 400 Bad Request — {{"error":{{"object":"error","type":"invalid_request_error","code":"invalid_request_error","message":"Error from provider: This model does not support image inputs"}}}}"#
        );
        assert!(!is_responses_api_required(&e));
    }

    #[test]
    fn test_error_response_body_extracts_after_separator() {
        assert_eq!(
            error_response_body("API returned 400 Bad Request — {\"error\":1}"),
            "{\"error\":1}"
        );
        // No separator → whole string.
        assert_eq!(error_response_body("plain message"), "plain message");
    }

    #[test]
    fn test_detect_initial_chat_format() {
        // Generic / unknown bases default to OpenAI-compatible.
        for (base_url, expected) in [
            ("https://api.anthropic.com", ChatFormat::Anthropic),
            (
                "https://generativelanguage.googleapis.com/v1beta",
                ChatFormat::Google,
            ),
            ("https://openrouter.ai/api/v1", ChatFormat::OpenAI),
            ("http://localhost:8080", ChatFormat::OpenAI),
        ] {
            assert_eq!(detect_initial_chat_format(base_url), expected, "{base_url}");
        }
    }

    fn key_with_chat_route(model: &str, protocol: &str) -> ApiKey {
        let mut key = ApiKey::new_with_protocol(
            "id".to_string(),
            "name".to_string(),
            // Generic host → URL detection picks OpenAI, so a persisted route must win.
            "https://api.example.com/v1".to_string(),
            None,
            "sk-test".to_string(),
        );
        let mut tool = std::collections::BTreeMap::new();
        tool.insert(
            model.to_string(),
            PersistedRoute {
                protocol: protocol.to_string(),
                path_variant: String::new(),
            },
        );
        key.protocol_routes.insert("code".to_string(), tool);
        key
    }

    #[test]
    fn direct_base_url_resolves_the_starter_sentinel() {
        let mut key = ApiKey::new_with_protocol(
            "id".to_string(),
            crate::constants::AIVO_STARTER_KEY_NAME.to_string(),
            crate::constants::AIVO_STARTER_SENTINEL.to_string(),
            None,
            crate::constants::AIVO_STARTER_EMPTY_SECRET.to_string(),
        );
        assert_eq!(
            direct_base_url(&key),
            crate::constants::AIVO_STARTER_REAL_URL
        );
        key.base_url = "https://api.example.com/v1/".to_string();
        assert_eq!(direct_base_url(&key), "https://api.example.com");
    }

    #[test]
    fn chat_format_protocol_round_trips() {
        for f in [
            ChatFormat::OpenAI,
            ChatFormat::Anthropic,
            ChatFormat::Responses,
            ChatFormat::Google,
        ] {
            assert_eq!(
                chat_format_from_protocol(chat_format_protocol(&f)),
                f,
                "{f:?}"
            );
        }
    }

    #[test]
    fn seeded_format_prefers_persisted_route_over_url() {
        let key = key_with_chat_route("gpt-oss", "anthropic");
        assert_eq!(seeded_chat_format(&key, "gpt-oss"), ChatFormat::Anthropic);
    }

    #[test]
    fn seeded_format_round_trips_responses() {
        let key = key_with_chat_route("o3", "responses");
        assert_eq!(seeded_chat_format(&key, "o3"), ChatFormat::Responses);
    }

    #[test]
    fn seeded_format_falls_back_to_url_when_unrouted() {
        let key = key_with_chat_route("other-model", "anthropic");
        assert_eq!(seeded_chat_format(&key, "gpt-oss"), ChatFormat::OpenAI);
    }

    #[test]
    fn seeded_format_canonicalizes_context_suffix() {
        let key = key_with_chat_route("claude-sonnet", "anthropic");
        assert_eq!(
            seeded_chat_format(&key, "claude-sonnet[1m]"),
            ChatFormat::Anthropic
        );
    }

    #[test]
    fn persist_skips_unchanged_route() {
        let key = key_with_chat_route("gpt-oss", "anthropic");
        assert!(chat_route_to_persist(&key, "gpt-oss", &ChatFormat::Anthropic).is_none());
    }

    #[test]
    fn persist_emits_changed_route() {
        let key = key_with_chat_route("gpt-oss", "anthropic");
        assert_eq!(
            chat_route_to_persist(&key, "gpt-oss", &ChatFormat::OpenAI),
            Some((
                "gpt-oss".to_string(),
                PersistedRoute::from_route(ProviderProtocol::Openai, PathVariant::Default)
            ))
        );
    }

    #[test]
    fn persist_emits_first_route_for_new_model() {
        let key = key_with_chat_route("gpt-oss", "anthropic");
        assert!(chat_route_to_persist(&key, "claude-3", &ChatFormat::Anthropic).is_some());
    }

    #[test]
    fn persist_skips_empty_model() {
        let key = key_with_chat_route("gpt-oss", "anthropic");
        assert!(chat_route_to_persist(&key, "", &ChatFormat::OpenAI).is_none());
    }

    #[test]
    fn persist_skips_cursor_acp_key() {
        let mut key = key_with_chat_route("gpt-oss", "anthropic");
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        assert!(key.is_cursor_acp());
        assert!(chat_route_to_persist(&key, "gpt-oss", &ChatFormat::OpenAI).is_none());
    }

    fn noisy_png_base64() -> (String, usize) {
        let png = crate::services::image_optimize::test_noise_png(1400, 1400);
        assert!(png.len() > crate::services::image_optimize::BYTE_BUDGET);
        let len = png.len();
        (BASE64.encode(png), len)
    }

    #[tokio::test]
    async fn materialize_optimizes_inline_clipboard_image() {
        let (data, original_len) = noisy_png_base64();
        let attachment = MessageAttachment {
            name: "clipboard-test.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline { data },
        };
        let out = materialize_attachment(&attachment).await.unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        let AttachmentStorage::Inline { data } = &out.storage else {
            panic!("expected inline storage");
        };
        assert!(BASE64.decode(data).unwrap().len() < original_len);
    }

    #[tokio::test]
    async fn materialize_optimizes_file_ref_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        let (data, original_len) = noisy_png_base64();
        std::fs::write(&path, BASE64.decode(data).unwrap()).unwrap();
        let attachment = MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::FileRef {
                path: path.to_string_lossy().into_owned(),
            },
        };
        let out = materialize_attachment(&attachment).await.unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        let AttachmentStorage::Inline { data } = &out.storage else {
            panic!("expected inline storage");
        };
        assert!(BASE64.decode(data).unwrap().len() < original_len);
    }

    #[tokio::test]
    async fn materialize_keeps_small_inline_image_verbatim() {
        let pixel_png = crate::services::image_optimize::test_tiny_png();
        let attachment = MessageAttachment {
            name: "dot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: BASE64.encode(&pixel_png),
            },
        };
        let out = materialize_attachment(&attachment).await.unwrap();
        assert_eq!(out.mime_type, "image/png");
        let AttachmentStorage::Inline { data } = &out.storage else {
            panic!("expected inline storage");
        };
        assert_eq!(BASE64.decode(data).unwrap(), pixel_png);
    }

    #[test]
    fn nested_tui_guard_trips_only_on_marker() {
        // SAFETY: no other test touches CODE_ACTIVE_ENV; cleared first so this
        // passes when `cargo test` itself runs inside an aivo code session.
        unsafe { std::env::remove_var(crate::constants::CODE_ACTIVE_ENV) };
        assert!(refuse_nested_tui().is_ok());
        unsafe { std::env::set_var(crate::constants::CODE_ACTIVE_ENV, "1") };
        let err = refuse_nested_tui().unwrap_err();
        assert!(err.to_string().contains("/new"), "got: {err}");
        unsafe { std::env::remove_var(crate::constants::CODE_ACTIVE_ENV) };
        assert!(refuse_nested_tui().is_ok());
    }
}
