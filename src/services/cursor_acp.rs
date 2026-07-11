//! Cursor ACP provider glue: model discovery (Phase 1) and chat turn driver
//! (Phase 2). The driver speaks ACP/JSON-RPC over stdio against the hidden
//! `cursor-agent acp` subcommand via [`crate::services::acp_client::AcpClient`].

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::process::Command;

use crate::agent::protocol::Decision;
use crate::services::acp_client::{
    AcpClient, ExtMethodFn, PermissionDecision, PermissionFn, PromptEvent, PromptStream,
};
use crate::services::cursor_home_shadow::CursorShadow;
use crate::services::session_store::{ApiKey, AttachmentStorage, MessageAttachment};

pub const CURSOR_ACP_SENTINEL: &str = "cursor";
/// `key.key` prefix that marks a shadow-isolated cursor account. The slug
/// after the colon is the [`CursorShadow::account_id`].
pub const CURSOR_SHADOW_PREFIX: &str = "cursor-shadow:";
/// Pre-shadow sentinel that delegated to the user's real ~/.cursor login.
/// Retired — any key still carrying it is asked to re-add so cursor-agent
/// is never accidentally pointed at the user's global cursor home.
const LEGACY_CURSOR_LOGIN_SENTINEL: &str = "cursor-login";
const CURSOR_AGENT_BIN: &str = "cursor-agent";

/// Detects pre-shadow cursor keys that haven't been re-added since the
/// switch to per-account isolation. The launcher refuses to use them so
/// cursor-agent never reads/writes the user's real ~/.cursor through aivo.
pub fn is_legacy_cursor_login_secret(secret: &str) -> bool {
    secret == LEGACY_CURSOR_LOGIN_SENTINEL
}

/// Env var that HARD-OVERRIDES Cursor's tool-execution policy for
/// `session/request_permission`, for scripted / non-interactive use. Set to `0`,
/// `false`, `no`, or `reject` to force conversation-only; any other value forces
/// allow. When UNSET, an interactive chat session follows its live auto-approve
/// toggle (Shift+Tab) instead — see [`cursor_permission_decision`].
pub const CURSOR_ALLOW_TOOLS_ENV: &str = "AIVO_CURSOR_ALLOW_TOOLS";

/// Explicit `AIVO_CURSOR_ALLOW_TOOLS` override, else `None` (fall back to the
/// live toggle). Both directions match EXPLICITLY — an unrecognized value is
/// NOT treated as allow (that would fail open on a security setting); it's
/// ignored with a one-time warning.
fn cursor_allow_tools_env_override() -> Option<PermissionDecision> {
    const DENY: &[&str] = &[
        "0", "false", "no", "off", "reject", "deny", "never", "disable", "none",
    ];
    const ALLOW: &[&str] = &["1", "true", "yes", "on", "allow", "always"];
    let raw = std::env::var(CURSOR_ALLOW_TOOLS_ENV).ok()?;
    let v = raw.trim();
    if v.is_empty() {
        return None;
    }
    if DENY.iter().any(|d| v.eq_ignore_ascii_case(d)) {
        Some(PermissionDecision::Reject)
    } else if ALLOW.iter().any(|a| v.eq_ignore_ascii_case(a)) {
        Some(PermissionDecision::Allow)
    } else {
        warn_unrecognized_allow_tools_once(v);
        None
    }
}

/// Warn once when `AIVO_CURSOR_ALLOW_TOOLS` carries an unrecognized value.
fn warn_unrecognized_allow_tools_once(value: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "aivo: ignoring unrecognized {CURSOR_ALLOW_TOOLS_ENV}={value:?} \
             (expected 1/true/yes/allow or 0/false/no/reject); \
             using the interactive tool-approval toggle instead."
        );
    }
}

/// Decide a cursor `session/request_permission`. The env override wins when set;
/// otherwise the live auto-approve toggle governs — ON lets cursor's tools run
/// in the real workspace, OFF rejects them. Cursor's tools run out-of-process,
/// so aivo can't surface its own permission card for them; "off" therefore means
/// conversation-only, which keeps the displayed safety state honest instead of
/// silently auto-running tools behind an "off" indicator. `None` (no interactive
/// toggle, e.g. a one-shot turn) keeps the historical allow-by-default.
fn cursor_permission_decision(auto_approve: Option<&AtomicBool>) -> PermissionDecision {
    if let Some(decision) = cursor_allow_tools_env_override() {
        return decision;
    }
    match auto_approve {
        Some(flag) if !flag.load(Ordering::Relaxed) => PermissionDecision::Reject,
        _ => PermissionDecision::Allow,
    }
}

/// What the user is asked to approve for one cursor tool call — built from the
/// ACP `session/request_permission` params and shown on aivo's permission card.
pub struct CursorPermissionRequest {
    /// Tool key for the card heading (see `permission_heading`).
    pub tool: String,
    /// Human description of what cursor wants to do.
    pub preview: String,
}

/// Interactive front-end hook: ask the user to approve one cursor tool call and
/// return the decision. Supplied by the chat TUI; `None` falls back to the
/// toggle/env policy (the loopback routers and one-shot turns pass `None`).
pub type CursorPermissionPrompt = Arc<
    dyn Fn(CursorPermissionRequest) -> futures::future::BoxFuture<'static, Decision> + Send + Sync,
>;

/// A cursor `cursor/ask_question` question. `options` are `(id, label)`; the
/// answer comes back as selected option ids (cursor carries no free text here).
pub struct CursorAskQuestion {
    pub id: String,
    pub prompt: String,
    pub options: Vec<(String, String)>,
    pub allow_multiple: bool,
}

pub struct CursorAskRequest {
    pub questions: Vec<CursorAskQuestion>,
}

pub struct CursorAskAnswer {
    pub question_id: String,
    pub selected_option_ids: Vec<String>,
}

pub enum CursorAskOutcome {
    Answered(Vec<CursorAskAnswer>),
    Cancelled,
}

/// TUI hook for `cursor/ask_question`; `None` leaves it unimplemented.
pub type CursorAskQuestionPrompt = Arc<
    dyn Fn(CursorAskRequest) -> futures::future::BoxFuture<'static, CursorAskOutcome> + Send + Sync,
>;

/// Params: `{toolCallId, title, questions:[{id, prompt, options:[{id,label}], allowMultiple}]}`.
fn parse_cursor_ask_request(params: &Value) -> CursorAskRequest {
    let questions = params
        .get("questions")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|q| CursorAskQuestion {
                    id: str_field(q, "id"),
                    prompt: str_field(q, "prompt"),
                    options: q
                        .get("options")
                        .and_then(Value::as_array)
                        .map(|opts| {
                            opts.iter()
                                .map(|o| (str_field(o, "id"), str_field(o, "label")))
                                .collect()
                        })
                        .unwrap_or_default(),
                    allow_multiple: q
                        .get("allowMultiple")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                })
                .collect()
        })
        .unwrap_or_default();
    CursorAskRequest { questions }
}

fn str_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Result: `{outcome:{outcome:"answered",answers:[{questionId,selectedOptionIds}]}}` or `…"cancelled"`.
fn cursor_ask_outcome_to_result(outcome: CursorAskOutcome) -> Value {
    match outcome {
        CursorAskOutcome::Answered(answers) => {
            let answers: Vec<Value> = answers
                .into_iter()
                .map(|a| {
                    json!({"questionId": a.question_id, "selectedOptionIds": a.selected_option_ids})
                })
                .collect();
            json!({"outcome": {"outcome": "answered", "answers": answers}})
        }
        CursorAskOutcome::Cancelled => json!({"outcome": {"outcome": "cancelled"}}),
    }
}

/// A `cursor/update_todos` item; `status` is cursor's raw string.
#[derive(Clone)]
pub struct CursorTodo {
    pub id: String,
    pub content: String,
    pub status: String,
}

/// A `cursor/update_todos` push; `merge` folds `todos` in by `id`, else replaces.
pub struct CursorTodosUpdate {
    pub todos: Vec<CursorTodo>,
    pub merge: bool,
}

/// Non-blocking sink for `cursor/update_todos` (returns immediately).
pub type CursorTodosSink = Arc<dyn Fn(CursorTodosUpdate) + Send + Sync>;

/// A `cursor/task` notice — the only frame carrying the real task description
/// (the `tool_call` title is a generic `Task: Subagent task`).
pub struct CursorTaskNotice {
    pub tool_call_id: String,
    pub description: String,
    pub prompt: String,
    pub subagent_type: String,
}

/// Non-blocking sink for `cursor/task` (returns immediately).
pub type CursorTaskSink = Arc<dyn Fn(CursorTaskNotice) + Send + Sync>;

pub struct CursorPlanPhase {
    pub name: String,
    pub todos: Vec<CursorTodo>,
}

/// A `cursor/create_plan` plan; `plan` is the model's markdown body.
pub struct CursorPlanRequest {
    pub name: Option<String>,
    pub overview: Option<String>,
    pub plan: String,
    pub todos: Vec<CursorTodo>,
    pub phases: Vec<CursorPlanPhase>,
}

pub enum CursorPlanOutcome {
    Accepted,
    Rejected(String),
    Cancelled,
}

/// TUI hook for `cursor/create_plan`; `None` leaves it unimplemented.
pub type CursorPlanPrompt = Arc<
    dyn Fn(CursorPlanRequest) -> futures::future::BoxFuture<'static, CursorPlanOutcome>
        + Send
        + Sync,
>;

/// TUI-supplied hooks for cursor's `cursor/*` extension methods; each optional.
#[derive(Default, Clone)]
pub struct CursorInteractionHooks {
    pub ask_question: Option<CursorAskQuestionPrompt>,
    pub update_todos: Option<CursorTodosSink>,
    pub create_plan: Option<CursorPlanPrompt>,
    pub task: Option<CursorTaskSink>,
}

impl CursorInteractionHooks {
    fn is_empty(&self) -> bool {
        self.ask_question.is_none()
            && self.update_todos.is_none()
            && self.create_plan.is_none()
            && self.task.is_none()
    }
}

/// One [`ExtMethodFn`] dispatching each `cursor/*` method to its hook; `None` when unset.
fn build_ext_method_fn(hooks: CursorInteractionHooks) -> Option<ExtMethodFn> {
    if hooks.is_empty() {
        return None;
    }
    Some(Arc::new(move |method: String, params: Value| {
        let hooks = hooks.clone();
        Box::pin(async move {
            match method.as_str() {
                "cursor/ask_question" => {
                    let prompt = hooks.ask_question.as_ref()?;
                    let outcome = prompt(parse_cursor_ask_request(&params)).await;
                    Some(cursor_ask_outcome_to_result(outcome))
                }
                "cursor/update_todos" => {
                    let sink = hooks.update_todos.as_ref()?;
                    sink(parse_cursor_todos_update(&params));
                    Some(json!({}))
                }
                "cursor/create_plan" => {
                    let prompt = hooks.create_plan.as_ref()?;
                    let outcome = prompt(parse_cursor_plan_request(&params)).await;
                    Some(cursor_plan_outcome_to_result(outcome))
                }
                "cursor/task" => {
                    let sink = hooks.task.as_ref()?;
                    sink(parse_cursor_task_notice(&params));
                    Some(json!({}))
                }
                _ => None,
            }
        })
    }))
}

/// Parse a `cursor/task` notice (`{toolCallId, description, prompt, subagentType, …}`).
fn parse_cursor_task_notice(params: &Value) -> CursorTaskNotice {
    let s = |k: &str| {
        params
            .get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string()
    };
    CursorTaskNotice {
        tool_call_id: s("toolCallId"),
        description: s("description"),
        prompt: s("prompt"),
        subagent_type: s("subagentType"),
    }
}

/// Parse a `[{id, content, status}]` array, dropping items missing id/content.
fn parse_cursor_todo_array(value: Option<&Value>) -> Vec<CursorTodo> {
    value
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    Some(CursorTodo {
                        id: t.get("id").and_then(Value::as_str)?.to_string(),
                        content: t.get("content").and_then(Value::as_str)?.to_string(),
                        status: t
                            .get("status")
                            .and_then(Value::as_str)
                            .unwrap_or("pending")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Params: `{toolCallId, todos:[{id, content, status}], merge}`.
fn parse_cursor_todos_update(params: &Value) -> CursorTodosUpdate {
    CursorTodosUpdate {
        todos: parse_cursor_todo_array(params.get("todos")),
        merge: params
            .get("merge")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

/// Params: `{toolCallId, name?, overview?, plan, todos:[…], isProject?, phases:[{name, todos:[…]}]}`.
fn parse_cursor_plan_request(params: &Value) -> CursorPlanRequest {
    let opt_str = |k: &str| {
        params
            .get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let phases = params
        .get("phases")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|p| CursorPlanPhase {
                    name: str_field(p, "name"),
                    todos: parse_cursor_todo_array(p.get("todos")),
                })
                .collect()
        })
        .unwrap_or_default();
    CursorPlanRequest {
        name: opt_str("name"),
        overview: opt_str("overview"),
        plan: str_field(params, "plan"),
        todos: parse_cursor_todo_array(params.get("todos")),
        phases,
    }
}

/// Result: `{outcome:{outcome:"accepted"|"rejected"|"cancelled", reason?}}` (no planUri → cursor saves its own).
fn cursor_plan_outcome_to_result(outcome: CursorPlanOutcome) -> Value {
    match outcome {
        CursorPlanOutcome::Accepted => json!({"outcome": {"outcome": "accepted"}}),
        CursorPlanOutcome::Rejected(reason) => {
            json!({"outcome": {"outcome": "rejected", "reason": reason}})
        }
        CursorPlanOutcome::Cancelled => json!({"outcome": {"outcome": "cancelled"}}),
    }
}

/// Turn an ACP `session/request_permission` `params` object into a card request.
/// cursor carries the human description in `toolCall.title` (with a `kind` like
/// `execute`/`edit`); degrade gracefully when a field is missing.
fn build_cursor_permission_request(params: &Value) -> CursorPermissionRequest {
    let tool_call = params.get("toolCall").unwrap_or(params);
    let title = tool_call
        .get("title")
        .and_then(Value::as_str)
        .or_else(|| params.get("title").and_then(Value::as_str))
        .unwrap_or("a tool")
        .trim();
    let kind = tool_call.get("kind").and_then(Value::as_str);
    let preview = match kind {
        Some(k) if !k.is_empty() => format!("Cursor wants to run a {k} tool:\n{title}"),
        _ => format!("Cursor wants to run:\n{title}"),
    };
    CursorPermissionRequest {
        tool: "cursor".to_string(),
        preview,
    }
}

/// Resolve one cursor `session/request_permission`. An `AIVO_CURSOR_ALLOW_TOOLS`
/// override hard-wins; otherwise auto-approve-on allows silently; otherwise, when
/// an interactive `prompt` is wired, ask the user (with "always" flipping
/// `auto` on for the rest of the session); with no prompt, fall back to the
/// toggle policy (off = conversation-only reject, no flag = allow-by-default).
async fn resolve_cursor_permission(
    params: &Value,
    auto: Option<&AtomicBool>,
    prompt: Option<&CursorPermissionPrompt>,
) -> PermissionDecision {
    if let Some(decision) = cursor_allow_tools_env_override() {
        return decision;
    }
    if auto.is_some_and(|f| f.load(Ordering::Relaxed)) {
        return PermissionDecision::Allow;
    }
    match prompt {
        Some(ask) => match ask(build_cursor_permission_request(params)).await {
            Decision::Allow => PermissionDecision::Allow,
            Decision::AlwaysAllow => {
                if let Some(flag) = auto {
                    flag.store(true, Ordering::Relaxed);
                }
                PermissionDecision::Allow
            }
            Decision::Deny => PermissionDecision::Reject,
        },
        None => cursor_permission_decision(auto),
    }
}

pub fn is_cursor_acp_base(base_url: &str) -> bool {
    base_url == CURSOR_ACP_SENTINEL
}

/// Resolved cursor-agent path. PATH is walked at most once per process; a
/// missing binary caches `None` so the second call doesn't re-stat.
fn resolved_cursor_agent_path() -> Option<&'static PathBuf> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let dirs = crate::services::path_search::collect_path_dirs();
            crate::services::path_search::find_in_dirs(CURSOR_AGENT_BIN, &dirs)
        })
        .as_ref()
}

pub fn ensure_cursor_agent_installed() -> Result<()> {
    if resolved_cursor_agent_path().is_some() {
        Ok(())
    } else {
        anyhow::bail!(
            "`cursor-agent` was not found on PATH. Install Cursor CLI support first, then run `aivo keys add cursor`."
        )
    }
}

/// Bare cursor-agent Command with no env applied. Prefer
/// [`cursor_agent_command_for_key`] so the shadow env is wired in.
fn cursor_agent_command_bare() -> Command {
    match resolved_cursor_agent_path() {
        Some(path) => Command::new(path),
        None => Command::new(CURSOR_AGENT_BIN),
    }
}

/// Build a Command for spawning cursor-agent against the given key. Wires
/// in the shadow's HOME/XDG/APPDATA + CURSOR_CONFIG_DIR/CURSOR_DATA_DIR
/// when the key is a shadow account; otherwise falls back to a bare spawn
/// (used by raw `--key sk-…` API-key keys).
pub fn cursor_agent_command_for_key(key: &ApiKey) -> Result<Command> {
    // Refuse pre-shadow keys at the single open chokepoint: a legacy key has
    // no shadow/auth env, so cursor-agent would run against the real ~/.cursor.
    if is_legacy_cursor_login_secret(key.key.as_str()) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Run `aivo keys rm {0}` then `aivo keys add cursor` to recreate it as an isolated account.",
            key.id
        );
    }
    let mut cmd = cursor_agent_command_bare();
    apply_shadow_env_to_command(&mut cmd, key)?;
    apply_color_env_to_command(&mut cmd);
    if let Some((name, value)) = cursor_auth_env(key) {
        cmd.env(name, value);
    } else {
        cmd.env_remove("CURSOR_API_KEY");
    }
    Ok(cmd)
}

/// Same as [`cursor_agent_command_for_key`] but takes an explicit shadow.
/// Used by the `aivo keys add cursor` flow before any aivo key exists.
pub fn cursor_agent_command_for_shadow(shadow: &CursorShadow) -> Command {
    let mut cmd = cursor_agent_command_bare();
    for (name, value) in shadow.env_block() {
        cmd.env(name, value);
    }
    apply_color_env_to_command(&mut cmd);
    cmd.env_remove("CURSOR_API_KEY");
    cmd
}

fn apply_shadow_env_to_command(cmd: &mut Command, key: &ApiKey) -> Result<()> {
    if let Some(shadow) = cursor_shadow_for_key(key)? {
        ensure_shadow_once(&shadow)?;
        for (name, value) in shadow.env_block() {
            cmd.env(name, value);
        }
    }
    Ok(())
}

/// Runs `shadow.ensure()` at most once per `(process, account_id)`. The
/// first call repairs broken pre-fix shadows (`set-keychain-settings`,
/// missing `Library/Keychains/login.keychain-db`); subsequent spawns reuse
/// the result without re-shelling `/usr/bin/security`.
fn ensure_shadow_once(shadow: &CursorShadow) -> Result<()> {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().contains(&shadow.account_id) {
        return Ok(());
    }
    shadow.ensure()?;
    seen.lock().unwrap().insert(shadow.account_id.clone());
    Ok(())
}

fn apply_color_env_to_command(cmd: &mut Command) {
    cmd.env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env("CLICOLOR_FORCE", "0");
}

/// Parsed shape of a cursor `key.key` secret. The wire format is
/// extensible — additional `:<kind>:<value>` segments can be added later
/// (e.g. `:auth_token:`, `:base_url:`) without breaking older callers.
///
/// - `cursor-shadow:<id>`              → OAuth login, no embedded creds
/// - `cursor-shadow:<id>:api:<key>`    → API-key auth, key embedded
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorSecret<'a> {
    pub account_id: &'a str,
    pub api_key: Option<&'a str>,
}

pub fn parse_cursor_shadow_secret(secret: &str) -> Option<CursorSecret<'_>> {
    let rest = secret.strip_prefix(CURSOR_SHADOW_PREFIX)?;
    let mut parts = rest.splitn(3, ':');
    let account_id = parts.next().filter(|s| !s.is_empty())?;
    match (parts.next(), parts.next()) {
        (None, _) => Some(CursorSecret {
            account_id,
            api_key: None,
        }),
        (Some("api"), Some(key)) if !key.is_empty() => Some(CursorSecret {
            account_id,
            api_key: Some(key),
        }),
        _ => None,
    }
}

/// Build the secret string for a freshly-allocated OAuth-login shadow.
pub fn build_cursor_oauth_secret(account_id: &str) -> String {
    format!("{CURSOR_SHADOW_PREFIX}{account_id}")
}

/// Build the secret string for an API-key shadow account. Caller must
/// have validated the api_key (no `:`, non-empty).
pub fn build_cursor_apikey_secret(account_id: &str, api_key: &str) -> String {
    format!("{CURSOR_SHADOW_PREFIX}{account_id}:api:{api_key}")
}

/// Extract the shadow account id encoded in `key.key`, if any.
pub fn cursor_account_id(key: &ApiKey) -> Option<&str> {
    parse_cursor_shadow_secret(key.key.as_str()).map(|s| s.account_id)
}

/// Resolve the on-disk shadow for a cursor key. Returns `Ok(None)` for
/// non-shadow keys (those don't exist after the multi-account refactor;
/// kept for forward compatibility).
pub fn cursor_shadow_for_key(key: &ApiKey) -> Result<Option<CursorShadow>> {
    match cursor_account_id(key) {
        Some(id) => Ok(Some(CursorShadow::for_account_id(id.to_string())?)),
        None => Ok(None),
    }
}

pub fn cursor_auth_env(key: &ApiKey) -> Option<(&'static str, OsString)> {
    let secret = key.key.as_str();
    if let Some(parsed) = parse_cursor_shadow_secret(secret) {
        return parsed
            .api_key
            .map(|k| ("CURSOR_API_KEY", OsString::from(k)));
    }
    if secret.is_empty() || is_legacy_cursor_login_secret(secret) {
        None
    } else {
        Some(("CURSOR_API_KEY", OsString::from(secret)))
    }
}

pub fn cursor_models_cache_identity(key: &ApiKey) -> String {
    let secret = key.key.as_str();
    if secret.is_empty() {
        return CURSOR_ACP_SENTINEL.to_string();
    }
    // Shadow keys: cache per account id. Two `aivo keys add cursor` runs
    // are independent accounts even if they happen to log in as the same
    // upstream user, so each id deserves its own cache slot. The cache
    // key only carries the account id — embedded API keys (if any) are
    // sensitive and don't belong in plain-text cache filenames.
    if let Some(parsed) = parse_cursor_shadow_secret(secret) {
        return format!("{CURSOR_ACP_SENTINEL}#shadow-{}", parsed.account_id);
    }
    let digest = Sha256::digest(secret.as_bytes());
    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        hex.push_str(&format!("{byte:02x}"));
    }
    format!("{CURSOR_ACP_SENTINEL}#{hex}")
}

pub async fn list_cursor_models(key: &ApiKey) -> Result<Vec<String>> {
    ensure_cursor_agent_installed()?;

    if is_legacy_cursor_login_secret(key.key.as_str()) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Run `aivo keys rm {0}` then `aivo keys add cursor` to recreate it as an isolated account.",
            key.id
        );
    }

    // OAuth-login shadow keys: `cursor-agent status` is authoritative.
    // (API-key shadows can't use status — cursor-agent only inspects
    // auth.json there, ignoring CURSOR_API_KEY, so a valid API-key key
    // would report "unauthenticated".)
    if let Some(parsed) = parse_cursor_shadow_secret(key.key.as_str())
        && parsed.api_key.is_none()
        && !cursor_status_authenticated_for_key(key)
            .await
            .unwrap_or(false)
    {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth {0}` to sign in again.",
            key.id
        );
    }

    let primary = run_cursor_agent(["models"], key).await;
    let output = match primary {
        Ok(output) => output,
        Err(primary_err) => run_cursor_agent(["--list-models"], key)
            .await
            .with_context(|| {
                format!(
                    "`cursor-agent models` failed ({primary_err}); fallback `cursor-agent --list-models` also failed"
                )
            })?,
    };

    if looks_like_no_models_message(&output) {
        let parsed = parse_cursor_shadow_secret(key.key.as_str());
        let hint = match parsed {
            Some(p) if p.api_key.is_some() => format!(
                "Run `aivo keys reauth {0}` and paste a valid Cursor API key.",
                key.id
            ),
            Some(_) => format!(
                "Run `aivo keys reauth {0}` to sign in again — the saved cursor account has been signed out.",
                key.id
            ),
            _ => "Re-run `aivo keys add cursor` to set up a fresh isolated cursor account."
                .to_string(),
        };
        anyhow::bail!("Cursor returned no models for this account. {hint}");
    }

    let models = parse_cursor_models(&output)?;
    if models.is_empty() {
        anyhow::bail!("Cursor returned no model IDs");
    }
    Ok(models)
}

/// Recognize cursor-agent's prose responses for the unauthenticated /
/// no-quota cases. The CLI exits 0 in these states, so we can't lean on
/// the exit code — only the body text.
fn looks_like_no_models_message(output: &str) -> bool {
    let stripped = crate::services::ansi::strip_ansi(output);
    let lower = stripped.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    lower.starts_with("no models")
        || lower.contains("not logged in")
        || lower.contains("not authenticated")
        || lower.contains("please log in")
        || lower.contains("please sign in")
}

async fn cursor_status_authenticated(mut cmd: Command) -> Result<bool> {
    let output = cmd
        .args(["status", "--format", "json"])
        .stdin(Stdio::null())
        .output()
        .await
        .context("failed to run `cursor-agent status --format json`")?;
    if !output.status.success() {
        return Ok(false);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_cursor_status_authenticated(&stdout).unwrap_or(false))
}

/// `cursor-agent status --format json` against the given key's shadow
/// (or the user's global cursor home if the key isn't shadow-backed).
pub async fn cursor_status_authenticated_for_key(key: &ApiKey) -> Result<bool> {
    ensure_cursor_agent_installed()?;
    cursor_status_authenticated(cursor_agent_command_for_key(key)?).await
}

/// True when `key` is an OAuth-login shadow (no API key) whose `cursor-agent`
/// session is signed out — the signal to bail before spawning a router that
/// would hand over a dead endpoint. API-key shadows always report
/// unauthenticated (`status` only inspects auth.json), so they're excluded and
/// left for the first upstream request to validate.
pub async fn cursor_oauth_shadow_signed_out(key: &ApiKey) -> bool {
    let is_oauth_shadow =
        parse_cursor_shadow_secret(key.key.as_str()).is_some_and(|parsed| parsed.api_key.is_none());
    is_oauth_shadow
        && !cursor_status_authenticated_for_key(key)
            .await
            .unwrap_or(false)
}

/// Status check against an explicit shadow — used by `aivo keys add
/// cursor` after the login flow finishes and before an `ApiKey` exists.
pub async fn cursor_status_authenticated_for_shadow(shadow: &CursorShadow) -> Result<bool> {
    ensure_cursor_agent_installed()?;
    cursor_status_authenticated(cursor_agent_command_for_shadow(shadow)).await
}

/// Run `cursor-agent login` into the given shadow. Inherits stdio so the
/// device-code prompt and browser-open hint reach the user.
pub async fn run_cursor_login_for_shadow(shadow: &CursorShadow) -> Result<()> {
    ensure_cursor_agent_installed()?;
    let status = cursor_agent_command_for_shadow(shadow)
        .arg("login")
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("failed to run `cursor-agent login`")?;
    if !status.success() {
        anyhow::bail!("`cursor-agent login` exited with {status}");
    }
    Ok(())
}

async fn run_cursor_agent<const N: usize>(args: [&str; N], key: &ApiKey) -> Result<String> {
    let mut cmd = cursor_agent_command_for_key(key)?;
    cmd.args(args).stdin(Stdio::null());

    let output = cmd.output().await.context("failed to run cursor-agent")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            anyhow::bail!("cursor-agent exited with {}", output.status);
        }
        anyhow::bail!("cursor-agent exited with {}: {}", output.status, detail);
    }

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&output.stderr).to_string();
    }
    Ok(text)
}

pub fn parse_cursor_models(output: &str) -> Result<Vec<String>> {
    let stripped = crate::services::ansi::strip_ansi(output);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let mut out = Vec::new();
        collect_model_ids_from_json(&value, &mut out);
        out.sort();
        out.dedup();
        return Ok(out);
    }

    let mut out = Vec::new();
    for line in stripped.lines() {
        if let Some(model) = parse_cursor_model_line(line) {
            out.push(model);
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_model_ids_from_json(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_model_ids_from_json(item, out);
            }
        }
        Value::Object(map) => {
            let mut recursed = false;
            for key in ["models", "data", "items", "result"] {
                if let Some(child) = map.get(key) {
                    collect_model_ids_from_json(child, out);
                    recursed = true;
                }
            }
            if recursed {
                return;
            }
            for key in ["id", "model", "name"] {
                if let Some(id) = map.get(key).and_then(Value::as_str)
                    && is_plausible_model_id(id)
                {
                    out.push(id.trim().to_string());
                    return;
                }
            }
        }
        Value::String(s) if is_plausible_model_id(s) => out.push(s.trim().to_string()),
        _ => {}
    }
}

fn parse_cursor_model_line(line: &str) -> Option<String> {
    let mut s = line.trim();
    if s.is_empty() {
        return None;
    }

    let lower = s.to_ascii_lowercase();
    let heading = lower.ends_with(':')
        || matches!(
            lower.as_str(),
            "models" | "available models" | "cursor models" | "model"
        );
    if heading {
        return None;
    }

    // Prose lines (multi-word sentences ending in punctuation) aren't model
    // ids even when their first word happens to be a plausible identifier —
    // e.g. cursor-agent emits "No models available for this account." when
    // signed out, and the unguarded parser used to serve "No" as a model.
    if s.split_whitespace().count() > 1 && matches!(s.chars().last(), Some('.' | '!' | '?')) {
        return None;
    }

    s = s.trim_start_matches(['*', '-', '+', '•', '·']).trim_start();
    s = s.trim_start_matches(['>', '✓', '✔', '●', '○']).trim_start();
    for marker in [
        "(current)",
        "[current]",
        "(default)",
        "[default]",
        "current:",
        "default:",
    ] {
        if let Some(rest) = s.strip_prefix(marker) {
            s = rest.trim_start();
        }
    }

    let first = s
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches([',', ';', ':']);
    if is_plausible_model_id(first) {
        Some(first.to_string())
    } else {
        None
    }
}

fn is_plausible_model_id(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.len() > 160 {
        return false;
    }
    if s.contains("://") || s.contains(char::is_whitespace) {
        return false;
    }
    if !s.bytes().any(|b| b.is_ascii_alphanumeric()) {
        return false;
    }
    s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'/' | b':' | b'@'))
}

/// Reasoning tiers Cursor bakes into model-id suffixes (`none` = off); it ships
/// one id per tier rather than a `reasoning_effort` param.
const CURSOR_EFFORT_TIERS: [&str; 6] = ["none", "low", "medium", "high", "xhigh", "max"];

/// A Cursor model id split into underlying model + baked-in effort/mode:
/// `<base>[-thinking]-<tier>[-fast]` (thinking on either side of the tier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorModelParts {
    pub base: String,
    pub tier: Option<&'static str>,
    pub thinking: bool,
    pub fast: bool,
}

impl CursorModelParts {
    /// Footer label like `max · thinking · fast`; `None` if no effort/mode hint.
    pub fn effort_label(&self) -> Option<String> {
        let label = [
            self.tier,
            self.thinking.then_some("thinking"),
            self.fast.then_some("fast"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ");
        (!label.is_empty()).then_some(label)
    }

    /// The underlying model's context window (`None` if unknown): Cursor-native
    /// ids, then snapshot, then the reordered version-first Claude spelling.
    pub fn context_window(&self) -> Option<u64> {
        use crate::services::model_metadata::snapshot_limits;
        // First, so a coincidental snapshot id (a bare `auto` row) can't shadow it.
        if let Some(ctx) = cursor_native_context_window(&self.base) {
            return Some(ctx);
        }
        if let Some(ctx) = snapshot_limits(&self.base).and_then(|l| l.context) {
            return Some(ctx);
        }
        reorder_claude_version_first(&self.base)
            .and_then(|alt| snapshot_limits(&alt).and_then(|l| l.context))
    }
}

/// Windows for Cursor's own ids, absent from models.dev. Exact-match (not
/// prefix) so an unseen future id stays unknown rather than wrongly numbered.
fn cursor_native_context_window(base: &str) -> Option<u64> {
    match base {
        "composer-2.5" => Some(200_000),
        "auto" => Some(2_000_000),
        _ => None,
    }
}

/// Decompose a Cursor model id. Strips at most one each of `-fast`/tier/
/// `-thinking` so a family ending in a tier word (`gpt-5.1-codex-max`) survives.
pub fn parse_cursor_model(id: &str) -> CursorModelParts {
    let mut s = id;
    let mut fast = false;
    if let Some(prefix) = s.strip_suffix("-fast") {
        s = prefix;
        fast = true;
    }
    let mut tier = None;
    let mut thinking = false;
    // Two passes cover `-thinking-<tier>` and `-<tier>-thinking`; each token once.
    for _ in 0..2 {
        if tier.is_none()
            && let Some((prefix, matched)) = CURSOR_EFFORT_TIERS.iter().find_map(|&t| {
                s.strip_suffix(t)
                    .and_then(|p| p.strip_suffix('-'))
                    .map(|p| (p, t))
            })
        {
            s = prefix;
            tier = Some(matched);
            continue;
        }
        if !thinking && let Some(prefix) = s.strip_suffix("-thinking") {
            s = prefix;
            thinking = true;
            continue;
        }
        break;
    }
    CursorModelParts {
        base: s.to_string(),
        tier,
        thinking,
        fast,
    }
}

/// Swap Cursor's version-first `claude-4.6-opus` to the snapshot's
/// `claude-opus-4.6`; `None` if not shaped that way.
fn reorder_claude_version_first(base: &str) -> Option<String> {
    let rest = base.strip_prefix("claude-")?;
    for family in ["opus", "sonnet", "haiku"] {
        if let Some(version) = rest.strip_suffix(family).and_then(|p| p.strip_suffix('-')) {
            return Some(format!("claude-{family}-{version}"));
        }
    }
    None
}

fn parse_cursor_status_authenticated(output: &str) -> Option<bool> {
    let value: Value = serde_json::from_str(output.trim()).ok()?;
    status_value_authenticated(&value)
}

fn status_value_authenticated(value: &Value) -> Option<bool> {
    match value {
        Value::Object(map) => {
            for key in [
                "authenticated",
                "isAuthenticated",
                "loggedIn",
                "logged_in",
                "signedIn",
                "signed_in",
            ] {
                if let Some(v) = map.get(key).and_then(Value::as_bool) {
                    return Some(v);
                }
            }
            for key in ["status", "auth", "login"] {
                if let Some(s) = map.get(key).and_then(Value::as_str) {
                    let lower = s.to_ascii_lowercase();
                    if [
                        "authenticated",
                        "logged_in",
                        "logged-in",
                        "logged in",
                        "signed_in",
                    ]
                    .contains(&lower.as_str())
                    {
                        return Some(true);
                    }
                    if ["unauthenticated", "logged_out", "logged-out", "logged out"]
                        .contains(&lower.as_str())
                    {
                        return Some(false);
                    }
                }
            }
            if map
                .get("user")
                .or_else(|| map.get("account"))
                .is_some_and(|v| !v.is_null())
            {
                return Some(true);
            }
            for child in map.values() {
                if let Some(v) = status_value_authenticated(child) {
                    return Some(v);
                }
            }
            None
        }
        _ => None,
    }
}

/// Reject image attachments before they reach `cursor-agent acp`.
///
/// Why: ACP defines an `image` content block, but as of writing
/// `cursor-agent`'s ACP server does not advertise the `image` prompt
/// capability — confirmed both by the spec
/// ([agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content),
/// which gates image blocks on `promptCapabilities.image`) and by the
/// `raphaelluethy/cursor-acp` adapter, which flattens images to text to
/// work around it. Bail with a clear message; once a session reports
/// `image: true` via [`PromptCapabilities`], callers can switch to
/// [`ensure_image_attachments_supported`] for the conditional path.
pub fn ensure_no_image_attachments(attachments: &[MessageAttachment]) -> Result<()> {
    if let Some(att) = first_image_attachment(attachments) {
        anyhow::bail!(
            "cursor: image attachments are not supported yet (attachment '{}', {})",
            att.name,
            att.mime_type
        );
    }
    Ok(())
}

/// Capability-conditional variant: only bails when the live session
/// advertises `promptCapabilities.image == false`. Use this once the caller
/// has an open [`CursorAcpSession`] and wants the behavior to follow what
/// cursor-agent actually reports.
pub fn ensure_image_attachments_supported(
    capabilities: &PromptCapabilities,
    attachments: &[MessageAttachment],
) -> Result<()> {
    if capabilities.image {
        return Ok(());
    }
    ensure_no_image_attachments(attachments)
}

fn first_image_attachment(attachments: &[MessageAttachment]) -> Option<&MessageAttachment> {
    attachments
        .iter()
        .find(|a| a.mime_type.starts_with("image/"))
}

/// Build the `session/prompt` content blocks for one user turn.
///
/// Image attachments become ACP `image` blocks (`{type, mimeType, data}`)
/// per [agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content);
/// the user text becomes a trailing `text` block. Non-image attachments are
/// dropped here — the chat layer only materializes image and text/document
/// attachments today, and document attachments aren't yet wired through
/// (they'd need either a `resource` block or a text-flatten path).
///
/// Caller is responsible for the capability check (see
/// [`ensure_image_attachments_supported`]) before sending.
pub fn build_prompt_blocks(text: &str, attachments: &[MessageAttachment]) -> Result<Vec<Value>> {
    let mut image_blocks = Vec::with_capacity(attachments.len());
    for att in attachments {
        if att.mime_type.starts_with("image/") {
            image_blocks.push(image_block_from_attachment(att)?);
        }
        // Non-image attachments are silently skipped for now; the chat layer
        // shouldn't be queuing types we don't handle yet, so this is mostly
        // defensive. Future work: text/document flattening + ACP `resource`
        // blocks once the agent advertises `embeddedContext: true`.
    }
    Ok(assemble_prompt_blocks(text, image_blocks))
}

/// Combine pre-shaped ACP image content blocks with the user text into one
/// `session/prompt` block list. Image blocks come first; the trailing text
/// block is omitted when the text is empty and at least one image block is
/// present, so an `/attach foo.png` + Enter with no draft text sends just
/// the image. A fully-empty submit still yields one empty text block so the
/// wire shape stays valid.
///
/// Router callers use this directly: they extract image blocks from the
/// inbound HTTP request (OpenAI/Anthropic/Responses/Gemini shapes) and
/// pair them with the reduced text prompt. Chat callers go through
/// [`build_prompt_blocks`] which does the [`MessageAttachment`] → image
/// block conversion first.
pub fn assemble_prompt_blocks(text: &str, mut image_blocks: Vec<Value>) -> Vec<Value> {
    if !text.is_empty() || image_blocks.is_empty() {
        image_blocks.push(text_block(text));
    }
    image_blocks
}

pub(crate) fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Build an ACP `image` content block from an inline-encoded base64 string
/// plus a MIME type. Exposed `pub(crate)` for the router, which already
/// holds the decoded `(mime, data)` pair after parsing protocol-specific
/// image parts and doesn't have a `MessageAttachment` to hand.
pub(crate) fn image_block_from_inline(mime_type: &str, data_base64: &str) -> Value {
    json!({
        "type": "image",
        "mimeType": mime_type,
        "data": data_base64,
    })
}

fn image_block_from_attachment(att: &MessageAttachment) -> Result<Value> {
    let data = match &att.storage {
        AttachmentStorage::Inline { data } => data.as_str(),
        AttachmentStorage::FileRef { .. } => {
            anyhow::bail!(
                "cursor: image attachment '{}' was not materialized before send",
                att.name
            )
        }
    };
    Ok(image_block_from_inline(&att.mime_type, data))
}

/// Result of one ACP `session/prompt` round-trip.
#[derive(Debug, Default)]
pub struct CursorTurnResult {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub stop_reason: Option<String>,
    pub model: Option<String>,
}

/// Streaming chunk delivered to callers of [`run_cursor_acp_turn`]. Text deltas
/// are borrowed to avoid allocating per delta on hot paths; tool steps (rare
/// relative to deltas) carry owned data.
#[derive(Debug)]
pub enum CursorChunk<'a> {
    Content(&'a str),
    Reasoning(&'a str),
    /// cursor-agent invoked a tool. `name`/`args` are normalized into aivo's tool
    /// vocabulary (`read_file`/`grep`/`run_bash`/…) so the transcript renders them
    /// like the in-process agent — and coalesces runs of the same kind. `id` is
    /// cursor's `toolCallId`, used to correlate the later [`CursorChunk::ToolUpdate`]
    /// that fills in the resolved target/result.
    ToolCall {
        id: Option<String>,
        name: String,
        args: serde_json::Value,
    },
    /// A `tool_call_update` for an earlier [`CursorChunk::ToolCall`] (matched by
    /// `id`). The initial call event often lacks the real target (path/pattern) —
    /// it arrives here in `rawInput`/`locations`, alongside a compact `result` and
    /// the `failed` status. Lets the transcript enrich the call line in place.
    ToolUpdate {
        id: String,
        args: Option<serde_json::Value>,
        result: Option<String>,
        failed: bool,
    },
}

/// How to resolve a user-facing model name against cursor-agent's encoded
/// `modelId` catalog. Chat prefers non-thinking variants; bridged routes keep
/// the default first-match behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelPickPreference {
    #[default]
    Default,
    PreferNoThinking,
}

/// Live `cursor-agent acp` connection scoped to a single ACP session.
///
/// Reuses one spawned child for many prompts so the TUI doesn't pay Node.js
/// startup latency per turn. Dropping the session kills the child (via
/// `AcpClient`'s `kill_on_drop`), so callers can simply forget the value to
/// terminate cleanly.
pub struct CursorAcpSession {
    client: Arc<AcpClient>,
    session_id: String,
    model_id: Option<String>,
    models: Value,
    /// `session/new` `modes` (`availableModes` + `currentModeId`); `Null` if none.
    modes: Value,
    /// Active mode id (`agent`/`plan`/`ask`); lets `set_mode` skip no-op switches.
    current_mode: Option<String>,
    prompt_capabilities: PromptCapabilities,
    model_pick_preference: ModelPickPreference,
}

/// Parsed `agentCapabilities.promptCapabilities` from the ACP `initialize`
/// response. Per spec ([agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content)),
/// these flags gate which content-block types the agent will accept in
/// `session/prompt`. cursor-agent's current ACP server appears to advertise
/// text-only — capturing the raw flags lets us flip image/audio guards from
/// blanket bails to capability-conditional once that changes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptCapabilities {
    pub image: bool,
    pub audio: bool,
    pub embedded_context: bool,
}

impl PromptCapabilities {
    pub fn from_init_response(init: &Value) -> Self {
        let prompt = init
            .get("agentCapabilities")
            .and_then(|c| c.get("promptCapabilities"));
        let flag = |name: &str| -> bool {
            prompt
                .and_then(|p| p.get(name))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        Self {
            image: flag("image"),
            audio: flag("audio"),
            embedded_context: flag("embeddedContext"),
        }
    }
}

impl CursorAcpSession {
    /// Spawn `cursor-agent acp`, run the initialize/authenticate/session/new
    /// handshake, and optionally lock in a starting model. Uses the
    /// env-var-controlled default permission policy (allow by default;
    /// see [`CURSOR_ALLOW_TOOLS_ENV`]).
    pub async fn open(
        key: &ApiKey,
        requested_model: Option<&str>,
        workspace_cwd: &str,
    ) -> Result<Self> {
        Self::open_with_options(
            key,
            requested_model,
            workspace_cwd,
            None,
            ModelPickPreference::Default,
            None,
            None,
            CursorInteractionHooks::default(),
        )
        .await
    }

    /// Variant of [`Self::open`] that registers an MCP server in
    /// `session/new`'s `mcpServers` array. The cursor router uses this to
    /// expose claude-cli's `/v1/messages` tools to the cursor model via the
    /// [`cursor_bridge::mcp`](crate::services::cursor_bridge::mcp) HTTP server.
    pub async fn open_with_mcp(
        key: &ApiKey,
        requested_model: Option<&str>,
        workspace_cwd: &str,
        mcp_url: Option<&str>,
    ) -> Result<Self> {
        Self::open_with_options(
            key,
            requested_model,
            workspace_cwd,
            mcp_url,
            ModelPickPreference::Default,
            None,
            None,
            CursorInteractionHooks::default(),
        )
        .await
    }

    /// Most general open: lets callers register an MCP server and pick a model
    /// preference. Tool execution follows the env-var default permission policy
    /// (allow by default; set `AIVO_CURSOR_ALLOW_TOOLS=0` for conversation-only —
    /// see [`CURSOR_ALLOW_TOOLS_ENV`]).
    #[allow(clippy::too_many_arguments)]
    pub async fn open_with_options(
        key: &ApiKey,
        requested_model: Option<&str>,
        workspace_cwd: &str,
        mcp_url: Option<&str>,
        model_pick_preference: ModelPickPreference,
        auto_approve: Option<Arc<AtomicBool>>,
        permission_prompt: Option<CursorPermissionPrompt>,
        hooks: CursorInteractionHooks,
    ) -> Result<Self> {
        ensure_cursor_agent_installed()?;
        let mut cmd = cursor_agent_command_for_key(key)?;
        if let Some(model) = requested_model.filter(|s| !s.is_empty()) {
            cmd.args(["--model", model]);
        }
        cmd.arg("acp");

        // cursor runs its tools out-of-process; each arrives as a
        // `session/request_permission`. An env override hard-wins; else if
        // auto-approve is on we allow silently; else when an interactive prompt
        // is wired we surface aivo's own permission card (allow once / always /
        // deny) instead of a blanket reject — "always" flips auto-approve on for
        // the rest of the session. With no prompt (routers, one-shot) we keep the
        // historical toggle/allow-by-default. The closure holds the shared toggle
        // flag, so a mid-session Shift+Tab is reflected on the very next request.
        let permission_fn: PermissionFn = Arc::new(move |params: Value| {
            let auto = auto_approve.clone();
            let prompt = permission_prompt.clone();
            Box::pin(async move {
                resolve_cursor_permission(&params, auto.as_deref(), prompt.as_ref()).await
            })
        });
        let ext_method_fn = build_ext_method_fn(hooks);
        let client =
            Arc::new(AcpClient::spawn_with_handlers(cmd, permission_fn, ext_method_fn).await?);

        let init = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": 1,
                    "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}},
                }),
            )
            .await
            .context("cursor-agent ACP initialize failed")?;

        // Skip the `authenticate` round-trip: cursor-agent is already
        // authed via CURSOR_API_KEY or on-disk shadow login. Calling it
        // would pop a macOS keychain-unlock dialog mid-chat; session/new
        // below surfaces a clean error if creds are actually missing.
        let _ = init.get("authMethods");

        let prompt_capabilities = PromptCapabilities::from_init_response(&init);

        let mcp_servers: Vec<Value> = mcp_url
            .map(|url| {
                vec![json!({
                    "type": "http",
                    "name": "aivo-cursor-bridge",
                    "url": url,
                    "headers": [],
                })]
            })
            .unwrap_or_default();
        let new_session = client
            .request(
                "session/new",
                json!({"cwd": workspace_cwd, "mcpServers": mcp_servers}),
            )
            .await
            .with_context(|| {
                format!(
                    "cursor-agent ACP session/new failed — check the cursor key with `aivo keys reauth {0}`",
                    key.id
                )
            })?;

        let session_id = new_session
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("cursor-agent session/new omitted sessionId: {new_session}"))?
            .to_string();

        let models = new_session.get("models").cloned().unwrap_or(Value::Null);
        let active_model = models
            .get("currentModelId")
            .and_then(Value::as_str)
            .map(str::to_string);

        let modes = new_session.get("modes").cloned().unwrap_or(Value::Null);
        let current_mode = modes
            .get("currentModeId")
            .and_then(Value::as_str)
            .map(str::to_string);

        Ok(Self {
            client,
            session_id,
            model_id: active_model,
            models,
            modes,
            current_mode,
            prompt_capabilities,
            model_pick_preference,
        })
    }

    /// Capability flags from this session's ACP `initialize`. Useful for
    /// gating the image bail (see [`ensure_no_image_attachments`]) on what
    /// the agent actually advertises, instead of a blanket reject.
    pub fn prompt_capabilities(&self) -> &PromptCapabilities {
        &self.prompt_capabilities
    }

    pub fn supports_image_prompts(&self) -> bool {
        self.prompt_capabilities.image
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    pub fn client_handle(&self) -> Arc<AcpClient> {
        self.client.clone()
    }

    /// Switch the active model. `requested` accepts the user-facing name
    /// (`composer-2.5`) or encoded modelId (`composer-2.5[fast=true]`).
    /// Names absent from `availableModels` no-op rather than forwarding —
    /// cursor-agent rejects them with JSON-RPC `-32602`, notably `"auto"`.
    /// Short-circuits matching ids so router requests don't trigger a
    /// `session/set_model` round trip on every call.
    /// Switch the live session's model. `Ok(false)` when `requested` matched no
    /// catalog entry (e.g. `"auto"`, which cursor lists but rejects in
    /// `session/set_model`) — a no-op callers must surface, else the UI lies.
    pub async fn set_model(&mut self, requested: &str) -> Result<bool> {
        let Some(model_id) =
            pick_model_id_from_models(&self.models, Some(requested), self.model_pick_preference)
        else {
            return Ok(false);
        };
        if self.model_id.as_deref() == Some(model_id.as_str()) {
            return Ok(true);
        }
        self.client
            .request(
                "session/set_model",
                json!({"sessionId": self.session_id, "modelId": model_id}),
            )
            .await
            .context("cursor-agent session/set_model failed")?;
        self.model_id = Some(model_id);
        Ok(true)
    }

    /// Switch cursor's mode (`agent`/`plan`/`ask`) via `session/set_mode`. Plan
    /// mode is what makes cursor emit `cursor/create_plan`. `Ok(false)` when
    /// `mode_id` isn't advertised — a no-op the caller must surface, else the UI lies.
    pub async fn set_mode(&mut self, mode_id: &str) -> Result<bool> {
        if !mode_available(&self.modes, mode_id) {
            return Ok(false);
        }
        if self.current_mode.as_deref() == Some(mode_id) {
            return Ok(true);
        }
        self.client
            .request(
                "session/set_mode",
                json!({"sessionId": self.session_id, "modeId": mode_id}),
            )
            .await
            .context("cursor-agent session/set_mode failed")?;
        self.current_mode = Some(mode_id.to_string());
        Ok(true)
    }

    /// Send a `session/prompt` and return a stream of session updates.
    pub async fn prompt(&self, text: &str) -> Result<PromptStream> {
        self.prompt_with_blocks(vec![text_block(text)]).await
    }

    /// Lower-level variant for callers that need to send multi-block prompts
    /// (image + text, etc.). The caller is responsible for honoring
    /// [`PromptCapabilities`] — see [`build_prompt_blocks`] for the
    /// canonical text+attachments → blocks conversion.
    pub async fn prompt_with_blocks(&self, blocks: Vec<Value>) -> Result<PromptStream> {
        self.client.start_prompt(&self.session_id, blocks).await
    }

    /// Best-effort `session/cancel` notification. The TUI fires this when the
    /// user aborts so the agent stops generating; the spawned task is also
    /// aborted independently, so a failure here is non-fatal.
    pub async fn cancel(&self) -> Result<()> {
        self.client
            .notify("session/cancel", json!({"sessionId": self.session_id}))
            .await
    }
}

/// Drive a single ACP turn against `cursor-agent`. Spawns the child, runs the
/// initialize/authenticate/session/new/session/prompt sequence, and streams
/// agent text/thought deltas through `on_chunk`. The child is killed when the
/// returned future resolves or is dropped.
pub async fn run_cursor_acp_turn<F>(
    key: &ApiKey,
    requested_model: Option<&str>,
    workspace_cwd: &str,
    prompt_text: &str,
    attachments: &[MessageAttachment],
    on_chunk: &mut F,
) -> Result<CursorTurnResult>
where
    F: FnMut(CursorChunk<'_>) -> Result<()>,
{
    // One-shot (`aivo code -p`) is non-interactive: default cursor's tools to
    // conversation-only (always-off toggle) so `-p` is fail-closed like the
    // native agent. `AIVO_CURSOR_ALLOW_TOOLS=1` still opts in (checked first).
    let one_shot_tools_off = Arc::new(AtomicBool::new(false));
    let session = CursorAcpSession::open_with_options(
        key,
        requested_model,
        workspace_cwd,
        None,
        ModelPickPreference::PreferNoThinking,
        Some(one_shot_tools_off),
        None,
        CursorInteractionHooks::default(),
    )
    .await?;
    ensure_image_attachments_supported(session.prompt_capabilities(), attachments)?;
    let blocks = build_prompt_blocks(prompt_text, attachments)?;
    let mut stream = session.prompt_with_blocks(blocks).await?;

    let mut out = CursorTurnResult {
        model: session.model_id().map(str::to_string),
        ..Default::default()
    };
    let mut reasoning_buf = String::new();
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                consume_session_update(&value, &mut out, &mut reasoning_buf, on_chunk)?;
            }
            PromptEvent::Done(result) => {
                let value = result
                    .map_err(|e| anyhow!(e))
                    .context("cursor-agent ACP session/prompt failed")?;
                out.stop_reason = value
                    .get("stopReason")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                break;
            }
        }
    }
    if !reasoning_buf.is_empty() {
        out.reasoning_content = Some(reasoning_buf);
    }
    Ok(out)
}

/// Folds one `session/update` value into the running turn result and emits a
/// chunk through `on_chunk` when the payload is plain text. Exposed
/// `pub(crate)` so the chat TUI can share the parsing rules.
pub(crate) fn consume_session_update<F>(
    value: &Value,
    out: &mut CursorTurnResult,
    reasoning_buf: &mut String,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(CursorChunk<'_>) -> Result<()>,
{
    let Some(update) = value.get("update") else {
        return Ok(());
    };
    let kind = update.get("sessionUpdate").and_then(Value::as_str);
    let text = || {
        update
            .get("content")
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
    };
    match kind {
        Some("agent_message_chunk") => {
            if let Some(t) = text() {
                out.content.push_str(t);
                on_chunk(CursorChunk::Content(t))?;
            }
        }
        Some("agent_thought_chunk") => {
            if let Some(t) = text() {
                reasoning_buf.push_str(t);
                on_chunk(CursorChunk::Reasoning(t))?;
            }
        }
        // A tool starting: surface a normalized call card. The resolved target
        // and result land later in a `tool_call_update` (below), correlated by id.
        Some("tool_call") => {
            let (name, args) = normalize_tool_call(update);
            let id = update
                .get("toolCallId")
                .and_then(Value::as_str)
                .map(str::to_string);
            on_chunk(CursorChunk::ToolCall { id, name, args })?;
        }
        // A tool progressing/finishing: carries the resolved input (real path /
        // pattern, which the start event usually omits) and the result. Emitted
        // only when it adds something — an enrichment for the call line.
        Some("tool_call_update") => {
            if let Some(id) = update.get("toolCallId").and_then(Value::as_str) {
                let args = update_target_args(update);
                let (result, failed) = summarize_tool_outcome(update);
                if args.is_some() || result.is_some() || failed {
                    on_chunk(CursorChunk::ToolUpdate {
                        id: id.to_string(),
                        args,
                        result,
                        failed,
                    })?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Map an ACP `tool_call` update onto aivo's tool vocabulary so cursor's steps
/// render (and coalesce) like the in-process agent's. The salient target — file
/// path, search pattern, or command — comes from `locations`/`rawInput`, falling
/// back to the human `title`.
fn normalize_tool_call(update: &Value) -> (String, Value) {
    let kind = update.get("kind").and_then(Value::as_str).unwrap_or("");
    let title = update.get("title").and_then(Value::as_str).unwrap_or("");
    // The path/pattern is left empty when absent rather than falling back to the
    // generic title ("Read File", "grep") — cursor sends those titles but no real
    // target (rawInput is empty, no locations), so the title is pure noise in the
    // call line. Execute is the exception: cursor's execute *title is* the command.
    let path = location_path(update)
        .or_else(|| raw_input_str(update, PATH_KEYS))
        .unwrap_or("");
    match kind {
        "read" => ("read_file".into(), serde_json::json!({ "path": path })),
        "edit" => ("edit_file".into(), serde_json::json!({ "path": path })),
        "delete" => ("delete_file".into(), serde_json::json!({ "path": path })),
        "search" => {
            let pattern = raw_input_str(update, PATTERN_KEYS).unwrap_or("");
            ("grep".into(), serde_json::json!({ "pattern": pattern }))
        }
        "execute" => {
            let command = raw_input_str(update, COMMAND_KEYS).unwrap_or(title);
            ("run_bash".into(), serde_json::json!({ "command": command }))
        }
        _ => {
            // A delegation (`Task: <desc>` title / "task" kind) renders as a
            // delegation, not raw-title jargon like `running Task: Subagent task`.
            let task_desc = title
                .strip_prefix("Task:")
                .or_else(|| (kind == "task" || title == "Task").then_some(""));
            if let Some(desc) = task_desc {
                let label = raw_input_str(update, TASK_LABEL_KEYS).unwrap_or(desc.trim());
                return (
                    "subagent".to_string(),
                    serde_json::json!({ "label": label }),
                );
            }
            let name = if title.is_empty() { "tool" } else { title };
            (name.to_string(), Value::Null)
        }
    }
}

/// cursor's `rawInput` parameter names aren't stable across tools (a read's path
/// may be `path`, `target_file`, `file_path`, …), so each salient field is keyed
/// off a candidate list rather than a single name.
const PATH_KEYS: &[&str] = &[
    "path",
    "target_file",
    "file_path",
    "relative_workspace_path",
    "file",
    "filename",
];
const PATTERN_KEYS: &[&str] = &["pattern", "query", "regex", "search"];
const COMMAND_KEYS: &[&str] = &["command", "cmd"];
const TASK_LABEL_KEYS: &[&str] = &["description", "label", "task"];

/// First string value among `keys` in the event's `rawInput`.
fn raw_input_str<'a>(update: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let raw = update.get("rawInput")?;
    keys.iter()
        .find_map(|k| raw.get(*k).and_then(Value::as_str))
}

/// The first `locations[].path` (ACP's standard place for a tool's file target).
fn location_path(update: &Value) -> Option<&str> {
    update
        .get("locations")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(|l| l.get("path"))
        .and_then(Value::as_str)
}

/// The resolved target from a `tool_call_update` — the real path/pattern/command
/// the start event lacked — as aivo-vocabulary args, or `None` if it carried no
/// new input. Keyed generically so the consumer overwrites whichever field its
/// tool reads.
fn update_target_args(update: &Value) -> Option<Value> {
    let mut map = serde_json::Map::new();
    if let Some(p) = location_path(update).or_else(|| raw_input_str(update, PATH_KEYS)) {
        map.insert("path".into(), Value::String(p.to_string()));
    }
    if let Some(p) = raw_input_str(update, PATTERN_KEYS) {
        map.insert("pattern".into(), Value::String(p.to_string()));
    }
    if let Some(c) = raw_input_str(update, COMMAND_KEYS) {
        map.insert("command".into(), Value::String(c.to_string()));
    }
    (!map.is_empty()).then_some(Value::Object(map))
}

/// A compact, one-line result for a `tool_call_update` plus whether it failed.
/// cursor reports the outcome in `rawOutput` — a match/result count, the read
/// content, or an `error` string (with status still `completed`) — so that's
/// checked first; ACP's standard `content` blocks are the fallback for other
/// tools/versions.
fn summarize_tool_outcome(update: &Value) -> (Option<String>, bool) {
    let mut failed = matches!(
        update.get("status").and_then(Value::as_str),
        Some("failed" | "error")
    );
    let mut result = None;
    if let Some(out) = update.get("rawOutput") {
        if let Some(err) = out.get("error").and_then(Value::as_str) {
            failed = true;
            result = Some(compact_result(err));
        } else if let Some(n) = out.get("totalMatches").and_then(Value::as_u64) {
            result = Some(format!("{n} match{}", if n == 1 { "" } else { "es" }));
        } else if let Some(n) = out.get("resultCount").and_then(Value::as_u64) {
            result = Some(format!("{n} result{}", if n == 1 { "" } else { "s" }));
        } else if let Some(content) = out.get("content").and_then(Value::as_str) {
            result = Some(compact_result(content));
        }
    }
    if result.is_none() {
        let text = collect_content_text(update);
        if !text.trim().is_empty() {
            result = Some(compact_result(&text));
        }
    }
    (result, failed)
}

/// Concatenate the text from ACP `content` blocks (`[{content:{text}}]` or flat
/// `[{text}]`) — the standard place for tool output, used as a fallback when
/// cursor's `rawOutput` digest is absent.
fn collect_content_text(update: &Value) -> String {
    let mut text = String::new();
    if let Some(blocks) = update.get("content").and_then(Value::as_array) {
        for block in blocks {
            let piece = block
                .get("content")
                .and_then(|c| c.get("text"))
                .and_then(Value::as_str)
                .or_else(|| block.get("text").and_then(Value::as_str));
            if let Some(piece) = piece {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(piece);
            }
        }
    }
    text
}

/// One-line digest of tool output: the line itself when short, else `N lines`.
fn compact_result(text: &str) -> String {
    let lines = text.lines().filter(|l| !l.trim().is_empty()).count();
    if lines <= 1 {
        let one = text.trim();
        if one.chars().count() > 50 {
            format!("{}…", one.chars().take(50).collect::<String>())
        } else {
            one.to_string()
        }
    } else {
        format!("{lines} lines")
    }
}

/// True when `mode_id` is in `modes.availableModes`. No advertised list → allow
/// (let cursor accept/reject) rather than block on a missing field.
fn mode_available(modes: &Value, mode_id: &str) -> bool {
    match modes.get("availableModes").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .any(|m| m.get("id").and_then(Value::as_str) == Some(mode_id)),
        None => true,
    }
}

/// Look up the encoded modelId for `requested` against an ACP `models` object
/// (the `models` field of a `session/new` response). Match by user-facing
/// `name` (what `aivo models` shows). With `requested=None` returns
/// `currentModelId`; with an unknown `requested` returns `None` so callers can
/// distinguish "catalog miss" from "no change needed" — the previous
/// fall-back-to-current behavior silently no-op'd `set_model` whenever the
/// picker offered a name that wasn't in `session/new`'s `availableModels`.
fn pick_model_id_from_models(
    models: &Value,
    requested: Option<&str>,
    preference: ModelPickPreference,
) -> Option<String> {
    let Some(requested) = requested.filter(|s| !s.is_empty()) else {
        return models
            .get("currentModelId")
            .and_then(Value::as_str)
            .map(str::to_string);
    };

    let available = models.get("availableModels").and_then(Value::as_array)?;
    if preference == ModelPickPreference::PreferNoThinking
        && let Some(id) = pick_prefer_no_thinking(available, requested)
    {
        return Some(id);
    }
    for entry in available {
        let name = entry.get("name").and_then(Value::as_str);
        let id = entry.get("modelId").and_then(Value::as_str);
        if name == Some(requested) || id == Some(requested) {
            return id.map(str::to_string);
        }
    }
    None
}

fn model_id_prefers_no_thinking(model_id: &str) -> bool {
    !model_id.contains("[thinking=true]")
}

fn pick_prefer_no_thinking(list: &[Value], requested: &str) -> Option<String> {
    let mut fallback = None;
    for entry in list {
        let name = entry.get("name").and_then(Value::as_str);
        // Skip — not abort — entries missing a modelId: cursor's catalog has
        // included partially-formed rows in the past, and `?` here would
        // short-circuit the whole search and silently degrade to the
        // first-match (thinking-variant) path.
        let Some(id) = entry.get("modelId").and_then(Value::as_str) else {
            continue;
        };
        if name != Some(requested) && id != requested {
            continue;
        }
        if model_id_prefers_no_thinking(id) {
            return Some(id.to_string());
        }
        if fallback.is_none() {
            fallback = Some(id.to_string());
        }
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroizing;

    #[test]
    fn mode_available_matches_session_new_modes_shape() {
        let modes = json!({
            "availableModes": [
                {"id": "agent", "name": "Agent"},
                {"id": "plan", "name": "Plan"},
                {"id": "ask", "name": "Ask"}
            ],
            "currentModeId": "agent"
        });
        assert!(mode_available(&modes, "plan"));
        assert!(mode_available(&modes, "agent"));
        assert!(!mode_available(&modes, "multitask"));
    }

    #[test]
    fn mode_available_allows_switch_when_no_modes_advertised() {
        assert!(mode_available(&Value::Null, "plan"));
        assert!(mode_available(&json!({}), "agent"));
    }

    #[test]
    fn parse_cursor_ask_request_reads_questions_options_and_multi() {
        // Wire shape cursor-agent's ACP ask-question handler sends.
        let params = json!({
            "toolCallId": "tc_1",
            "title": "Clarifying Questions",
            "questions": [{
                "id": "q1",
                "prompt": "Which paths should the rename touch?",
                "options": [
                    {"id": "o1", "label": "src/ only"},
                    {"id": "o2", "label": "everything"}
                ],
                "allowMultiple": true
            }]
        });
        let req = parse_cursor_ask_request(&params);
        assert_eq!(req.questions.len(), 1);
        let q = &req.questions[0];
        assert_eq!(q.id, "q1");
        assert_eq!(q.prompt, "Which paths should the rename touch?");
        assert!(q.allow_multiple);
        assert_eq!(
            q.options,
            vec![
                ("o1".to_string(), "src/ only".to_string()),
                ("o2".to_string(), "everything".to_string())
            ]
        );
    }

    #[test]
    fn parse_cursor_ask_request_tolerates_missing_fields() {
        // No questions array, sparse question object: never panics, fills blanks.
        assert!(parse_cursor_ask_request(&json!({})).questions.is_empty());
        let req = parse_cursor_ask_request(&json!({"questions": [{"prompt": "hi"}]}));
        assert_eq!(req.questions.len(), 1);
        assert_eq!(req.questions[0].prompt, "hi");
        assert!(req.questions[0].id.is_empty());
        assert!(req.questions[0].options.is_empty());
        assert!(!req.questions[0].allow_multiple);
    }

    #[test]
    fn cursor_ask_outcome_answered_serializes_to_selected_option_ids() {
        let result =
            cursor_ask_outcome_to_result(CursorAskOutcome::Answered(vec![CursorAskAnswer {
                question_id: "q1".into(),
                selected_option_ids: vec!["o2".into()],
            }]));
        assert_eq!(result["outcome"]["outcome"], "answered");
        assert_eq!(result["outcome"]["answers"][0]["questionId"], "q1");
        assert_eq!(
            result["outcome"]["answers"][0]["selectedOptionIds"],
            json!(["o2"])
        );
    }

    #[test]
    fn cursor_ask_outcome_cancelled_serializes_to_cancelled() {
        let result = cursor_ask_outcome_to_result(CursorAskOutcome::Cancelled);
        assert_eq!(result["outcome"]["outcome"], "cancelled");
        assert!(result["outcome"].get("answers").is_none());
    }

    #[test]
    fn parse_cursor_todos_update_reads_items_and_merge() {
        // Wire shape cursor-agent's update-todos notification sends.
        let params = json!({
            "toolCallId": "tc_1",
            "todos": [
                {"id": "t1", "content": "wire the bridge", "status": "in_progress"},
                {"id": "t2", "content": "add tests", "status": "pending"}
            ],
            "merge": true
        });
        let update = parse_cursor_todos_update(&params);
        assert!(update.merge);
        assert_eq!(update.todos.len(), 2);
        assert_eq!(update.todos[0].id, "t1");
        assert_eq!(update.todos[0].content, "wire the bridge");
        assert_eq!(update.todos[0].status, "in_progress");
    }

    #[test]
    fn parse_cursor_todos_update_defaults_merge_and_drops_malformed() {
        let update = parse_cursor_todos_update(&json!({
            "todos": [{"id": "t1", "content": "ok"}, {"content": "no id"}]
        }));
        assert!(!update.merge); // absent → false (replace)
        assert_eq!(update.todos.len(), 1); // the id-less item is dropped
        assert_eq!(update.todos[0].status, "pending"); // absent status → pending
    }

    #[test]
    fn parse_cursor_task_notice_reads_fields() {
        let notice = parse_cursor_task_notice(&json!({
            "toolCallId": "tool_60759ef1",
            "description": "Audit the auth flow",
            "prompt": "You are auditing…",
            "subagentType": "explore",
            "model": "composer-2.5",
        }));
        assert_eq!(notice.tool_call_id, "tool_60759ef1");
        assert_eq!(notice.description, "Audit the auth flow");
        assert_eq!(notice.prompt, "You are auditing…");
        assert_eq!(notice.subagent_type, "explore");
        // A failed spawn sends empty strings — parsed, not panicked.
        let empty =
            parse_cursor_task_notice(&json!({"toolCallId": "t", "subagentType": "unspecified"}));
        assert!(empty.description.is_empty() && empty.prompt.is_empty());
    }

    #[tokio::test]
    async fn ext_method_fn_dispatches_cursor_task_to_sink() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink_seen = seen.clone();
        let hooks = CursorInteractionHooks {
            task: Some(Arc::new(move |notice: CursorTaskNotice| {
                sink_seen.lock().unwrap().push(notice.description);
            })),
            ..Default::default()
        };
        let f = build_ext_method_fn(hooks).expect("task hook set → fn built");
        let reply = f(
            "cursor/task".to_string(),
            json!({"toolCallId": "t1", "description": "Map the engine", "prompt": "…"}),
        )
        .await;
        assert_eq!(reply, Some(json!({}))); // non-blocking ack
        assert_eq!(*seen.lock().unwrap(), vec!["Map the engine".to_string()]);
    }

    #[test]
    fn parse_cursor_plan_request_reads_fields_todos_and_phases() {
        let params = json!({
            "toolCallId": "tc_1",
            "name": "Rename the bridge",
            "overview": "Split the ACP glue",
            "plan": "Step through each call site.",
            "todos": [{"id": "t1", "content": "grep sites", "status": "completed"}],
            "isProject": false,
            "phases": [{
                "name": "Phase 1",
                "todos": [{"id": "p1", "content": "wire types", "status": "in_progress"}]
            }]
        });
        let req = parse_cursor_plan_request(&params);
        assert_eq!(req.name.as_deref(), Some("Rename the bridge"));
        assert_eq!(req.overview.as_deref(), Some("Split the ACP glue"));
        assert_eq!(req.plan, "Step through each call site.");
        assert_eq!(req.todos.len(), 1);
        assert_eq!(req.phases.len(), 1);
        assert_eq!(req.phases[0].name, "Phase 1");
        assert_eq!(req.phases[0].todos[0].content, "wire types");
    }

    #[test]
    fn cursor_plan_outcome_serializes_each_variant() {
        assert_eq!(
            cursor_plan_outcome_to_result(CursorPlanOutcome::Accepted)["outcome"]["outcome"],
            "accepted"
        );
        let rejected = cursor_plan_outcome_to_result(CursorPlanOutcome::Rejected("nope".into()));
        assert_eq!(rejected["outcome"]["outcome"], "rejected");
        assert_eq!(rejected["outcome"]["reason"], "nope");
        assert_eq!(
            cursor_plan_outcome_to_result(CursorPlanOutcome::Cancelled)["outcome"]["outcome"],
            "cancelled"
        );
    }

    #[test]
    fn parse_cursor_model_decomposes_effort_suffixes() {
        let p = parse_cursor_model("claude-opus-4-8-high");
        assert_eq!(p.base, "claude-opus-4-8");
        assert_eq!(p.tier, Some("high"));
        assert!(!p.thinking && !p.fast);

        let p = parse_cursor_model("claude-opus-4-8-max-fast");
        assert_eq!(p.base, "claude-opus-4-8");
        assert_eq!(p.tier, Some("max"));
        assert!(p.fast && !p.thinking);

        // thinking on either side of the tier resolves the same.
        for id in [
            "claude-opus-4-8-thinking-high-fast",
            "claude-4.5-opus-high-thinking",
        ] {
            let p = parse_cursor_model(id);
            assert_eq!(p.tier, Some("high"), "{id}");
            assert!(p.thinking, "{id}");
        }

        // `-xhigh` not mis-stripped to `-high` (the `-` boundary guards it).
        assert_eq!(
            parse_cursor_model("claude-opus-4-8-xhigh").tier,
            Some("xhigh")
        );

        // A family ending in a tier word keeps its name (one tier stripped).
        let p = parse_cursor_model("gpt-5.1-codex-max-high");
        assert_eq!(p.base, "gpt-5.1-codex-max");
        assert_eq!(p.tier, Some("high"));

        for id in ["auto", "claude-4.5-sonnet"] {
            let p = parse_cursor_model(id);
            assert_eq!(p.base, id);
            assert_eq!(p.tier, None);
            assert_eq!(p.effort_label(), None, "{id}");
        }
    }

    #[test]
    fn cursor_effort_label_joins_tier_and_modes() {
        let p = parse_cursor_model("claude-opus-4-8-thinking-max-fast");
        assert_eq!(p.effort_label().as_deref(), Some("max · thinking · fast"));
        assert_eq!(
            parse_cursor_model("claude-4-sonnet-thinking")
                .effort_label()
                .as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn cursor_model_context_window_recovers_underlying_window() {
        assert_eq!(
            parse_cursor_model("claude-opus-4-8-max").context_window(),
            Some(1_000_000)
        );
        // Version-first spelling resolves via the reorder fallback.
        assert_eq!(
            parse_cursor_model("claude-4.6-opus-high-thinking").context_window(),
            Some(1_000_000)
        );
        assert_eq!(
            parse_cursor_model("claude-4-sonnet").context_window(),
            Some(1_000_000)
        );
        assert_eq!(
            parse_cursor_model("claude-4.5-opus-high").context_window(),
            Some(200_000)
        );
        // Cursor-native, absent from models.dev (`-fast` strips to the same base).
        assert_eq!(
            parse_cursor_model("composer-2.5").context_window(),
            Some(200_000)
        );
        assert_eq!(
            parse_cursor_model("composer-2.5-fast").context_window(),
            Some(200_000)
        );
        assert_eq!(parse_cursor_model("auto").context_window(), Some(2_000_000));
        assert_eq!(
            parse_cursor_model("zzz-unknown-model-9").context_window(),
            None
        );
    }

    fn key(secret: &str) -> ApiKey {
        ApiKey {
            id: "abc".to_string(),
            name: "cursor".to_string(),
            base_url: CURSOR_ACP_SENTINEL.to_string(),
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
            key: Zeroizing::new(secret.to_string()),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn attachment(name: &str, mime: &str) -> MessageAttachment {
        MessageAttachment {
            name: name.to_string(),
            mime_type: mime.to_string(),
            storage: crate::services::session_store::AttachmentStorage::Inline {
                data: String::new(),
            },
        }
    }

    #[test]
    fn ensure_no_image_attachments_rejects_images_only() {
        assert!(ensure_no_image_attachments(&[]).is_ok());
        assert!(ensure_no_image_attachments(&[attachment("notes.txt", "text/plain")]).is_ok());
        assert!(ensure_no_image_attachments(&[attachment("doc.pdf", "application/pdf")]).is_ok());

        let err = ensure_no_image_attachments(&[attachment("shot.png", "image/png")])
            .expect_err("image rejection");
        let msg = format!("{err}");
        assert!(msg.contains("cursor"), "{msg}");
        assert!(msg.contains("shot.png"), "{msg}");
        assert!(msg.contains("image/png"), "{msg}");
    }

    #[test]
    fn prompt_capabilities_parses_flags_and_defaults_to_false() {
        let none = PromptCapabilities::from_init_response(&json!({}));
        assert_eq!(none, PromptCapabilities::default());

        let partial = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {"image": true}},
        }));
        assert!(partial.image);
        assert!(!partial.audio);
        assert!(!partial.embedded_context);

        let full = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {
                "image": true,
                "audio": true,
                "embeddedContext": true,
            }},
        }));
        assert!(full.image && full.audio && full.embedded_context);

        // Non-bool / missing fields fall back to false rather than panicking.
        let weird = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {"image": "yes"}},
        }));
        assert!(!weird.image);
    }

    fn inline_attachment(name: &str, mime: &str, data: &str) -> MessageAttachment {
        MessageAttachment {
            name: name.to_string(),
            mime_type: mime.to_string(),
            storage: AttachmentStorage::Inline {
                data: data.to_string(),
            },
        }
    }

    #[test]
    fn build_prompt_blocks_emits_text_only_when_no_attachments() {
        let blocks = build_prompt_blocks("hi", &[]).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": "hi"})]);
    }

    #[test]
    fn build_prompt_blocks_image_only_send_omits_empty_text_block() {
        // /attach foo.png + Enter with no draft text: send the image alone,
        // no trailing empty text block. If cursor-agent turns out to require
        // a text peer for image blocks, flip this in `build_prompt_blocks`.
        let img = inline_attachment("a.png", "image/png", "AAA=");
        let blocks = build_prompt_blocks("", std::slice::from_ref(&img)).unwrap();
        assert_eq!(
            blocks,
            vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
        );
    }

    #[test]
    fn build_prompt_blocks_empty_input_with_no_attachments_yields_one_text_block() {
        // Pure empty submit should still produce a sendable block list.
        let blocks = build_prompt_blocks("", &[]).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": ""})]);
    }

    #[test]
    fn build_prompt_blocks_orders_images_before_text() {
        let img = inline_attachment("shot.png", "image/png", "QUJD");
        let blocks = build_prompt_blocks("describe", std::slice::from_ref(&img)).unwrap();
        assert_eq!(
            blocks,
            vec![
                json!({"type": "image", "mimeType": "image/png", "data": "QUJD"}),
                json!({"type": "text", "text": "describe"}),
            ]
        );
    }

    #[test]
    fn build_prompt_blocks_drops_unsupported_attachment_types() {
        let doc = inline_attachment("notes.pdf", "application/pdf", "JVBE");
        let blocks = build_prompt_blocks("read this", std::slice::from_ref(&doc)).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": "read this"})]);
    }

    #[test]
    fn build_prompt_blocks_rejects_unmaterialized_images() {
        let img = MessageAttachment {
            name: "x.png".into(),
            mime_type: "image/png".into(),
            storage: AttachmentStorage::FileRef {
                path: "/tmp/x.png".into(),
            },
        };
        let err = build_prompt_blocks("hi", std::slice::from_ref(&img)).unwrap_err();
        assert!(format!("{err}").contains("not materialized"));
    }

    #[test]
    fn ensure_image_attachments_supported_follows_capability_flag() {
        let img = attachment("shot.png", "image/png");
        let caps_off = PromptCapabilities::default();
        let caps_on = PromptCapabilities {
            image: true,
            ..PromptCapabilities::default()
        };

        assert!(ensure_image_attachments_supported(&caps_off, std::slice::from_ref(&img)).is_err());
        assert!(ensure_image_attachments_supported(&caps_on, std::slice::from_ref(&img)).is_ok());
        // No attachments => always ok.
        assert!(ensure_image_attachments_supported(&caps_off, &[]).is_ok());
    }

    #[test]
    fn permission_decision_env_overrides_then_follows_toggle() {
        // Serialize against other env-var tests in the binary by saving the
        // prior value (if any) and restoring it on exit.
        let prior = std::env::var(CURSOR_ALLOW_TOOLS_ENV).ok();
        let on = AtomicBool::new(true);
        let off = AtomicBool::new(false);
        // SAFETY: tests run single-threaded by default for this binary; the
        // restore in the guard below keeps state consistent for any parallel
        // peers that may read this env var afterwards.
        unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) };
        // Env unset: no toggle => legacy allow-by-default; toggle governs when set.
        assert_eq!(cursor_permission_decision(None), PermissionDecision::Allow);
        assert_eq!(
            cursor_permission_decision(Some(&on)),
            PermissionDecision::Allow,
            "auto-approve ON allows cursor tools"
        );
        assert_eq!(
            cursor_permission_decision(Some(&off)),
            PermissionDecision::Reject,
            "auto-approve OFF rejects cursor tools (honest 'off' state)"
        );
        // Env set wins over the toggle, both directions.
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "1") };
        assert_eq!(
            cursor_permission_decision(Some(&off)),
            PermissionDecision::Allow,
            "explicit allow override beats an off toggle"
        );
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "reject") };
        assert_eq!(
            cursor_permission_decision(Some(&on)),
            PermissionDecision::Reject,
            "explicit reject override beats an on toggle"
        );
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "no") };
        assert_eq!(cursor_permission_decision(None), PermissionDecision::Reject);
        match prior {
            Some(v) => unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, v) },
            None => unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) },
        }
    }

    #[test]
    fn cursor_permission_request_reads_tool_call_title_and_kind() {
        let req = build_cursor_permission_request(&json!({
            "toolCall": { "title": "Run `git commit -m hi`", "kind": "execute" }
        }));
        assert_eq!(req.tool, "cursor");
        assert!(req.preview.contains("execute"), "preview: {}", req.preview);
        assert!(
            req.preview.contains("git commit"),
            "preview: {}",
            req.preview
        );

        // Missing fields degrade gracefully rather than panicking.
        let bare = build_cursor_permission_request(&json!({}));
        assert_eq!(bare.tool, "cursor");
        assert!(bare.preview.contains("a tool"));
    }

    #[tokio::test]
    async fn resolve_cursor_permission_prompts_when_off_and_honors_decision() {
        let prior = std::env::var(CURSOR_ALLOW_TOOLS_ENV).ok();
        // SAFETY: this binary's tests run single-threaded; restored below.
        unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) };

        let allow: CursorPermissionPrompt = Arc::new(|_| Box::pin(async { Decision::Allow }));
        let deny: CursorPermissionPrompt = Arc::new(|_| Box::pin(async { Decision::Deny }));
        let off = AtomicBool::new(false);

        assert_eq!(
            resolve_cursor_permission(&json!({}), Some(&off), Some(&allow)).await,
            PermissionDecision::Allow,
            "approved card → allow"
        );
        assert_eq!(
            resolve_cursor_permission(&json!({}), Some(&off), Some(&deny)).await,
            PermissionDecision::Reject,
            "denied card → reject"
        );

        // "Always" allows AND flips auto-approve on for the rest of the session.
        let always: CursorPermissionPrompt =
            Arc::new(|_| Box::pin(async { Decision::AlwaysAllow }));
        let flag = AtomicBool::new(false);
        assert_eq!(
            resolve_cursor_permission(&json!({}), Some(&flag), Some(&always)).await,
            PermissionDecision::Allow
        );
        assert!(
            flag.load(Ordering::Relaxed),
            "'always' turns auto-approve on"
        );

        // Auto-approve ON short-circuits without ever invoking the prompt.
        let called = Arc::new(AtomicBool::new(false));
        let probe = called.clone();
        let never: CursorPermissionPrompt = Arc::new(move |_| {
            probe.store(true, Ordering::Relaxed);
            Box::pin(async { Decision::Deny })
        });
        let on = AtomicBool::new(true);
        assert_eq!(
            resolve_cursor_permission(&json!({}), Some(&on), Some(&never)).await,
            PermissionDecision::Allow
        );
        assert!(!called.load(Ordering::Relaxed), "auto-on skips the prompt");

        // No interactive prompt → falls back to the toggle policy (off = reject).
        assert_eq!(
            resolve_cursor_permission(&json!({}), Some(&off), None).await,
            PermissionDecision::Reject
        );

        match prior {
            Some(v) => unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, v) },
            None => unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) },
        }
    }

    #[test]
    fn sentinel_detection_is_exact() {
        assert!(is_cursor_acp_base("cursor"));
        assert!(!is_cursor_acp_base(" cursor"));
        assert!(!is_cursor_acp_base("cursor/"));
        assert!(!is_cursor_acp_base("https://cursor.com"));
    }

    #[test]
    fn auth_env_skips_oauth_shadows_extracts_apikey_shadows_and_falls_back_for_raw_keys() {
        let oauth = key(&build_cursor_oauth_secret("abcd1234"));
        assert!(
            cursor_auth_env(&oauth).is_none(),
            "OAuth shadow keys must not leak CURSOR_API_KEY"
        );

        let api_shadow = key(&build_cursor_apikey_secret("abcd1234", "key_real_value"));
        let (name, value) = cursor_auth_env(&api_shadow).unwrap();
        assert_eq!(name, "CURSOR_API_KEY");
        assert_eq!(value, OsString::from("key_real_value"));

        let raw = key("sk-cursor");
        let (name, value) = cursor_auth_env(&raw).unwrap();
        assert_eq!(name, "CURSOR_API_KEY");
        assert_eq!(value, OsString::from("sk-cursor"));

        assert!(cursor_auth_env(&key(LEGACY_CURSOR_LOGIN_SENTINEL)).is_none());
    }

    #[test]
    fn parse_cursor_shadow_secret_round_trips() {
        let oauth = build_cursor_oauth_secret("abcd1234");
        let parsed = parse_cursor_shadow_secret(&oauth).unwrap();
        assert_eq!(parsed.account_id, "abcd1234");
        assert_eq!(parsed.api_key, None);

        let with_key = build_cursor_apikey_secret("abcd1234", "key_aabb");
        let parsed = parse_cursor_shadow_secret(&with_key).unwrap();
        assert_eq!(parsed.account_id, "abcd1234");
        assert_eq!(parsed.api_key, Some("key_aabb"));

        // Unknown extension keyword → reject rather than silently misroute
        // the secret. Defends against future format drift.
        assert!(parse_cursor_shadow_secret("cursor-shadow:abc:future:val").is_none());
        assert!(parse_cursor_shadow_secret("cursor-shadow:").is_none());
        assert!(parse_cursor_shadow_secret("sk-cursor").is_none());
    }

    #[test]
    fn cache_identity_separates_accounts_and_api_keys() {
        let a = cursor_models_cache_identity(&key(&format!("{CURSOR_SHADOW_PREFIX}aaaa1111")));
        let b = cursor_models_cache_identity(&key(&format!("{CURSOR_SHADOW_PREFIX}bbbb2222")));
        assert!(a.starts_with("cursor#shadow-"));
        assert_ne!(a, b);

        let x = cursor_models_cache_identity(&key("sk-x"));
        let y = cursor_models_cache_identity(&key("sk-y"));
        assert!(x.starts_with("cursor#"));
        assert!(!x.starts_with("cursor#shadow-"));
        assert_ne!(x, y);
        assert!(!x.contains("sk-x"));
    }

    #[test]
    fn cursor_account_id_round_trip() {
        let shadow = key(&format!("{CURSOR_SHADOW_PREFIX}abcd1234"));
        assert_eq!(cursor_account_id(&shadow), Some("abcd1234"));
        assert_eq!(cursor_account_id(&key("sk-cursor")), None);
    }

    #[test]
    fn parses_json_array_forms() {
        let ids = parse_cursor_models(r#"["composer-2.5","gpt-5"]"#).unwrap();
        assert_eq!(ids, vec!["composer-2.5", "gpt-5"]);
    }

    #[test]
    fn parses_json_object_forms() {
        let ids = parse_cursor_models(
            r#"{"models":[{"id":"composer-2.5"},{"model":"claude-sonnet-4.6"},{"name":"gpt-5"}]}"#,
        )
        .unwrap();
        assert_eq!(ids, vec!["claude-sonnet-4.6", "composer-2.5", "gpt-5"]);
    }

    #[test]
    fn parses_plain_line_output() {
        let ids = parse_cursor_models("composer-2.5\ngpt-5\n").unwrap();
        assert_eq!(ids, vec!["composer-2.5", "gpt-5"]);
    }

    #[test]
    fn rejects_no_models_prose_when_logged_out() {
        let ids = parse_cursor_models("No models available for this account.\n").unwrap();
        assert!(
            ids.is_empty(),
            "prose-style sentences must not parse as model ids, got {ids:?}"
        );

        assert!(looks_like_no_models_message(
            "No models available for this account."
        ));
        assert!(looks_like_no_models_message("Not logged in. Run login."));
        assert!(!looks_like_no_models_message("composer-2.5\ngpt-5"));
    }

    #[test]
    fn parses_headings_bullets_current_markers_and_blanks() {
        let ids = parse_cursor_models(
            "\nAvailable models:\n  * composer-2.5 (current)\n  - claude-sonnet-4.6\n  > gpt-5\n  default: o3\n",
        )
        .unwrap();
        assert_eq!(
            ids,
            vec!["claude-sonnet-4.6", "composer-2.5", "gpt-5", "o3"]
        );
    }

    #[test]
    fn status_parser_accepts_common_shapes() {
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"authenticated":true}"#),
            Some(true)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"auth":{"loggedIn":false}}"#),
            Some(false)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"status":"logged in"}"#),
            Some(true)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"user":{"email":"a@example.com"}}"#),
            Some(true)
        );
    }

    #[test]
    fn pick_model_id_matches_user_facing_name() {
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
                {"modelId": "claude-sonnet-4-6[thinking=true]", "name": "claude-sonnet-4-6"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(
                &models,
                Some("claude-sonnet-4-6"),
                ModelPickPreference::Default
            )
            .as_deref(),
            Some("claude-sonnet-4-6[thinking=true]")
        );
    }

    #[test]
    fn pick_model_id_prefers_non_thinking_for_chat() {
        let models = serde_json::json!({
            "currentModelId": "claude-sonnet-4-6[thinking=true]",
            "availableModels": [
                {"modelId": "claude-sonnet-4-6[thinking=true]", "name": "claude-sonnet-4-6"},
                {"modelId": "claude-sonnet-4-6", "name": "claude-sonnet-4-6"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(
                &models,
                Some("claude-sonnet-4-6"),
                ModelPickPreference::PreferNoThinking
            )
            .as_deref(),
            Some("claude-sonnet-4-6")
        );
    }

    #[test]
    fn pick_model_id_prefers_non_thinking_skips_entries_missing_model_id() {
        // Regression: an earlier version of `pick_prefer_no_thinking` used `?`
        // on the modelId lookup, which short-circuited the whole search when
        // any catalog row was missing the field — silently degrading
        // PreferNoThinking back to the first-match (thinking) behavior.
        let models = serde_json::json!({
            "currentModelId": "claude-sonnet-4-6[thinking=true]",
            "availableModels": [
                {"name": "partially-formed-row"},
                {"modelId": "claude-sonnet-4-6[thinking=true]", "name": "claude-sonnet-4-6"},
                {"modelId": "claude-sonnet-4-6", "name": "claude-sonnet-4-6"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(
                &models,
                Some("claude-sonnet-4-6"),
                ModelPickPreference::PreferNoThinking
            )
            .as_deref(),
            Some("claude-sonnet-4-6")
        );
    }

    #[test]
    fn pick_model_id_accepts_encoded_id_directly() {
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(
                &models,
                Some("composer-2.5[fast=true]"),
                ModelPickPreference::Default
            )
            .as_deref(),
            Some("composer-2.5[fast=true]")
        );
    }

    #[test]
    fn pick_model_id_returns_none_for_unknown_name_but_current_for_no_request() {
        // `"auto"` is the real-world catalog miss: `cursor-agent models`
        // lists it but ACP `session/set_model` rejects it with `-32602`.
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(&models, Some("totally-bogus"), ModelPickPreference::Default),
            None
        );
        assert_eq!(
            pick_model_id_from_models(&models, Some("auto"), ModelPickPreference::Default),
            None
        );
        assert_eq!(
            pick_model_id_from_models(&models, None, ModelPickPreference::Default).as_deref(),
            Some("composer-2.5[fast=true]")
        );
    }

    #[test]
    fn consume_session_update_splits_content_and_thought() {
        let mut out = CursorTurnResult::default();
        let mut reasoning = String::new();
        let mut chunks: Vec<(String, String)> = Vec::new();
        let mut on_chunk = |chunk: CursorChunk<'_>| -> Result<()> {
            match chunk {
                CursorChunk::Content(t) => chunks.push(("content".into(), t.to_string())),
                CursorChunk::Reasoning(t) => chunks.push(("reasoning".into(), t.to_string())),
                CursorChunk::ToolCall { name, .. } => chunks.push(("tool_call".into(), name)),
                CursorChunk::ToolUpdate { id, .. } => chunks.push(("tool_update".into(), id)),
            }
            Ok(())
        };

        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "hmm "}},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();
        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hello"}},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();
        // Unknown update kinds (plan, available_commands_update) are dropped.
        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "available_commands_update", "commands": []},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();

        assert_eq!(out.content, "Hello");
        assert_eq!(reasoning, "hmm ");
        assert_eq!(
            chunks,
            vec![
                ("reasoning".to_string(), "hmm ".to_string()),
                ("content".to_string(), "Hello".to_string()),
            ]
        );
    }

    #[test]
    fn consume_session_update_surfaces_tool_calls() {
        let mut out = CursorTurnResult::default();
        let mut reasoning = String::new();
        let mut chunks: Vec<(String, String)> = Vec::new();
        let mut on_chunk = |chunk: CursorChunk<'_>| -> Result<()> {
            match chunk {
                CursorChunk::Content(t) => chunks.push(("content".into(), t.to_string())),
                CursorChunk::Reasoning(t) => chunks.push(("reasoning".into(), t.to_string())),
                CursorChunk::ToolCall { id, name, args } => chunks.push((
                    "tool_call".into(),
                    format!("{}|{name}|{args}", id.unwrap_or_default()),
                )),
                CursorChunk::ToolUpdate {
                    id,
                    args,
                    result,
                    failed,
                } => chunks.push((
                    "tool_update".into(),
                    format!(
                        "{id}|{}|{}|{failed}",
                        args.map(|a| a.to_string()).unwrap_or_default(),
                        result.unwrap_or_default(),
                    ),
                )),
            }
            Ok(())
        };

        // A read tool. The start event carries cursor's own param name
        // (`target_file`), not ACP's `path`; normalization still resolves it.
        let call = serde_json::json!({
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "c1",
                "title": "Read File",
                "kind": "read",
                "status": "pending",
                "rawInput": {"target_file": "src/main.rs"},
            },
        });
        consume_session_update(&call, &mut out, &mut reasoning, &mut on_chunk).unwrap();

        // The update enriches the call: resolved path (from `locations`) + a
        // compact result, surfaced as a ToolUpdate keyed by the same id.
        let done = serde_json::json!({
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "c1",
                "status": "completed",
                "locations": [{"path": "src/main.rs"}],
                "content": [{"type": "content", "content": {"type": "text", "text": "fn main() {}"}}],
            },
        });
        consume_session_update(&done, &mut out, &mut reasoning, &mut on_chunk).unwrap();

        assert_eq!(
            chunks,
            vec![
                (
                    "tool_call".to_string(),
                    "c1|read_file|{\"path\":\"src/main.rs\"}".to_string()
                ),
                (
                    "tool_update".to_string(),
                    "c1|{\"path\":\"src/main.rs\"}|fn main() {}|false".to_string()
                ),
            ]
        );
        // Tool I/O is not assistant prose — it must not leak into the reply text.
        assert!(out.content.is_empty());
    }

    #[test]
    fn normalize_tool_call_maps_kinds_and_targets() {
        let search = serde_json::json!({
            "kind": "search", "title": "Grep", "rawInput": {"pattern": "hover"},
        });
        assert_eq!(
            normalize_tool_call(&search),
            ("grep".to_string(), serde_json::json!({"pattern": "hover"}))
        );

        let exec = serde_json::json!({
            "kind": "execute", "title": "git show ab12",
        });
        assert_eq!(
            normalize_tool_call(&exec),
            (
                "run_bash".to_string(),
                serde_json::json!({"command": "git show ab12"})
            )
        );

        // Unknown kind keeps the human title.
        let other = serde_json::json!({"kind": "fetch", "title": "Fetch URL"});
        assert_eq!(
            normalize_tool_call(&other),
            ("Fetch URL".to_string(), Value::Null)
        );

        // Delegation titles map onto the `subagent` vocabulary.
        let task = serde_json::json!({"kind": "other", "title": "Task: Subagent task"});
        assert_eq!(
            normalize_tool_call(&task),
            (
                "subagent".to_string(),
                serde_json::json!({"label": "Subagent task"})
            )
        );
        // rawInput's description (when cursor sends one) beats the title suffix.
        let task = serde_json::json!({
            "kind": "task", "title": "Task",
            "rawInput": {"description": "audit the auth flow"},
        });
        assert_eq!(
            normalize_tool_call(&task),
            (
                "subagent".to_string(),
                serde_json::json!({"label": "audit the auth flow"})
            )
        );

        // Real cursor shape: empty rawInput + a generic title → no fake target
        // (the title is noise). The result comes from the update's `rawOutput`.
        let real = serde_json::json!({
            "kind": "read", "title": "Read File", "rawInput": {},
        });
        assert_eq!(
            normalize_tool_call(&real),
            ("read_file".to_string(), serde_json::json!({"path": ""}))
        );
    }

    #[test]
    fn summarize_tool_outcome_reads_raw_output() {
        // Verified against real cursor-agent frames: the outcome lives in
        // `rawOutput` (counts / read content / error), not ACP `content`.
        let matches = serde_json::json!({
            "status": "completed", "rawOutput": {"totalMatches": 18, "truncated": false},
        });
        assert_eq!(
            summarize_tool_outcome(&matches),
            (Some("18 matches".to_string()), false)
        );

        let results = serde_json::json!({
            "status": "completed", "rawOutput": {"resultCount": 1},
        });
        assert_eq!(
            summarize_tool_outcome(&results),
            (Some("1 result".to_string()), false)
        );

        let read = serde_json::json!({
            "status": "completed", "rawOutput": {"content": "line one\nline two\nline three"},
        });
        assert_eq!(
            summarize_tool_outcome(&read),
            (Some("3 lines".to_string()), false)
        );

        // An error rides in `rawOutput.error` with status still `completed`.
        let err = serde_json::json!({
            "status": "completed",
            "rawOutput": {"error": "Glob pattern \"**/*\" is not allowed."},
        });
        let (result, failed) = summarize_tool_outcome(&err);
        assert!(failed);
        assert!(result.unwrap().starts_with("Glob pattern"));
    }
}
