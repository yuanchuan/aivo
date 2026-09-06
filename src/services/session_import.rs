//! Import another coding agent's session into aivo `code`. Full-fidelity reconstruct
//! of the WHOLE conversation (turns + tool calls/results) for `/resume`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use serde_json::{Value, json};

use crate::agent::protocol::{AssistantMessage, ToolCall};
use crate::agent::request::assistant_to_openai;
use crate::services::context_ingest::{
    extract_codex_message_text, extract_pi_text, ingest_project_headlines,
};
use crate::services::session_store::StoredChatMessage;

/// Shown in place of an image block (image reconstruction is deferred).
const IMAGE_PLACEHOLDER: &str = "[image omitted on import]";
/// A tool call whose result never appeared in the transcript (torn/cut session).
const MISSING_RESULT: &str = "[no tool result recorded on import]";
/// Engine-only closer so a resumed transcript ends on an assistant turn and
/// keeps user/assistant alternation (Anthropic 400s otherwise). Never displayed.
const RESUMED_MARK: &str = "[resumed from imported session]";
/// Engine-only opener when a transcript's first surviving turn isn't a user
/// message (Anthropic requires the first message to be `user`). Never displayed.
const IMPORT_LEAD: &str = "[imported session]";

/// Provenance of a session imported from another coding agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionOrigin {
    /// `"claude"` | `"codex"` | `"pi"`.
    pub cli: String,
    /// The source agent's own session id.
    pub foreign_id: String,
    /// Absolute path to the source `.jsonl` transcript.
    pub source_path: String,
}

/// A foreign session discovered for the current cwd, offered in `/resume`.
#[derive(Clone, Debug)]
pub struct ImportableSession {
    pub origin: SessionOrigin,
    /// Human title (first user turn), for the picker row.
    pub title: String,
    pub updated_at: DateTime<Utc>,
    /// The deterministic aivo session id this imports to.
    pub aivo_id: String,
}

/// One reconstructed foreign transcript, ready to persist as an aivo session.
pub struct ImportedTranscript {
    /// Display transcript (`user`/`assistant`/`tool_call`/`tool_result` rows).
    pub messages: Vec<StoredChatMessage>,
    /// Agent-engine wire log (OpenAI chat format, no system message).
    pub engine_messages: Vec<Value>,
    /// The source agent's model (e.g. `claude-sonnet-4-…`), for a provenance badge.
    pub origin_model: Option<String>,
    pub updated_at: DateTime<Utc>,
    pub fidelity: ImportFidelity,
}

/// Loss accounting for one conversion: every point where the importer drops,
/// synthesizes, or flattens content increments a counter, so a resume can state
/// its fidelity instead of degrading silently.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImportFidelity {
    pub user_turns: usize,
    pub assistant_turns: usize,
    pub tool_calls: usize,
    /// Tool results paired with their call.
    pub results_paired: usize,
    /// Calls whose result never appeared — a placeholder was synthesized.
    pub results_missing: usize,
    /// Results whose call was never seen — dropped (unmatched `tool` messages 400).
    pub results_orphaned: usize,
    /// Unparseable transcript lines (torn/partial writes).
    pub torn_lines: usize,
    /// Claude subagent (sidechain) lines not imported; the Task tool's summary
    /// survives on the main thread, so these don't degrade the tier.
    pub sidechain_lines: usize,
    /// Image blocks flattened to a text placeholder.
    pub images_omitted: usize,
}

/// Qualitative fidelity: `Full` = nothing lost, `High` = placeholders stand in
/// for tool results/images but the conversation is complete, `Partial` = source
/// content was actually dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FidelityTier {
    Full,
    High,
    Partial,
}

impl FidelityTier {
    pub fn label(self) -> &'static str {
        match self {
            FidelityTier::Full => "full",
            FidelityTier::High => "high",
            FidelityTier::Partial => "partial",
        }
    }
}

impl ImportFidelity {
    pub fn tier(&self) -> FidelityTier {
        if self.torn_lines > 0 || self.results_orphaned > 0 {
            FidelityTier::Partial
        } else if self.results_missing > 0 || self.images_omitted > 0 {
            FidelityTier::High
        } else {
            FidelityTier::Full
        }
    }

    /// One-line resume announcement:
    /// `Imported from Claude · fidelity full · 42 turns · 20/20 tool calls`.
    pub fn summary(&self, source: &str) -> String {
        let mut s = format!(
            "Imported from {source} · fidelity {} · {} turns",
            self.tier().label(),
            self.user_turns + self.assistant_turns,
        );
        if self.tool_calls > 0 {
            s.push_str(&format!(
                " · {}/{} tool calls",
                self.results_paired, self.tool_calls
            ));
        }
        s
    }

    /// Short degradation phrases, one per non-zero counter — empty at full
    /// fidelity.
    pub fn notes(&self) -> Vec<String> {
        fn count(n: usize, noun: &str, detail: &str) -> String {
            let s = if n == 1 { "" } else { "s" };
            format!("{n} {noun}{s}{detail}")
        }
        let mut out = Vec::new();
        if self.results_missing > 0 {
            out.push(count(
                self.results_missing,
                "tool result",
                " missing (placeholder inserted)",
            ));
        }
        if self.results_orphaned > 0 {
            out.push(count(
                self.results_orphaned,
                "orphaned tool result",
                " dropped",
            ));
        }
        if self.torn_lines > 0 {
            out.push(count(self.torn_lines, "unparseable line", " skipped"));
        }
        if self.images_omitted > 0 {
            out.push(count(self.images_omitted, "image", " omitted"));
        }
        if self.sidechain_lines > 0 {
            out.push(count(
                self.sidechain_lines,
                "subagent line",
                " not imported (task summaries retained)",
            ));
        }
        out
    }
}

/// The deterministic aivo session id a foreign session imports to: the source
/// tool tag over a short digest of the source id (`pi-019f15e1`). Stable so a
/// re-listed session dedupes against its already-imported copy.
pub fn import_session_id(cli: &str, foreign_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(foreign_id.as_bytes());
    let short: String = digest.iter().take(4).map(|b| format!("{b:02x}")).collect();
    format!("{cli}-{short}")
}

/// Split a fork's session id into `(lowercase source tag, remainder)`, accepting
/// both the `<cli>-<hash>` form and the legacy `import-<cli>-…` sentinel. `None`
/// for a native id (a UUID — its leading hex never matches a source tag).
pub(crate) fn split_fork_id(session_id: &str) -> Option<(&str, &str)> {
    let head = session_id.strip_prefix("import-").unwrap_or(session_id);
    let (cli, rest) = head.split_once('-')?;
    matches!(cli, "claude" | "codex" | "pi").then_some((cli, rest))
}

/// Display label for a fork's session id (`Claude`/`Codex`/`Pi`), or `None` for a
/// native session — lets the footer, provenance line, and overlay label a fork
/// from its id alone.
pub fn import_source_label(session_id: &str) -> Option<&'static str> {
    split_fork_id(session_id).map(|(cli, _)| source_label(cli))
}

/// What a `--resume <selector>` names. Resolution order is fidelity order:
/// a saved aivo session (exact id, then unique prefix, globally — a named
/// session resolves regardless of where it ran), else an importable foreign
/// session from this directory (matched by aivo import id or the source
/// tool's own id, prefix). `Unknown` leaves the selector to lower rungs
/// (digest injection / the filtered picker).
pub enum ResumeTarget {
    AivoSession(String),
    Foreign(Box<ImportableSession>),
    Ambiguous(String),
    Unknown,
}

/// Resolve a non-empty, non-`last` resume selector. Shared by the
/// `aivo code --resume` launch path and the TUI's `/resume <id>`.
pub async fn resolve_resume_target(
    store: &crate::services::session_store::SessionStore,
    project_root: &Path,
    selector: &str,
) -> ResumeTarget {
    let sel = selector.trim();
    if sel.is_empty() {
        return ResumeTarget::Unknown;
    }
    let (index, imports) = tokio::join!(
        store.all_chat_sessions(),
        list_importable_sessions(project_root),
    );
    let aivo_ids = index
        .unwrap_or_default()
        .into_iter()
        .map(|entry| entry.session_id);
    match_resume_candidates(aivo_ids, imports, sel)
}

/// Pure matching half of [`resolve_resume_target`]. An exact aivo id wins
/// outright (a full id must never read as ambiguous with sessions it
/// prefixes); an importable whose fork already exists as a saved aivo session
/// collapses into that session (same conversation, higher fidelity).
fn match_resume_candidates(
    aivo_ids: impl Iterator<Item = String>,
    imports: Vec<ImportableSession>,
    sel: &str,
) -> ResumeTarget {
    let mut aivo_hits: Vec<String> = aivo_ids.filter(|id| id.starts_with(sel)).collect();
    if aivo_hits.iter().any(|id| id == sel) {
        return ResumeTarget::AivoSession(sel.to_string());
    }
    let mut import_hits: Vec<ImportableSession> = imports
        .into_iter()
        .filter(|s| s.aivo_id.starts_with(sel) || s.origin.foreign_id.starts_with(sel))
        .collect();
    import_hits.retain(|s| !aivo_hits.contains(&s.aivo_id));
    match (aivo_hits.len(), import_hits.len()) {
        (1, 0) => ResumeTarget::AivoSession(aivo_hits.remove(0)),
        (0, 1) => ResumeTarget::Foreign(Box::new(import_hits.into_iter().next().unwrap())),
        (0, 0) => ResumeTarget::Unknown,
        (a, i) => ResumeTarget::Ambiguous(format!(
            "Session id prefix '{sel}' is ambiguous ({} matches). Use more characters.",
            a + i
        )),
    }
}

/// Per-source cap: the picker's working set, not an archive.
const MAX_PER_SOURCE: usize = 40;

/// List Claude Code / Codex / Pi sessions in `project_root`, newest-first.
pub async fn list_importable_sessions(project_root: &Path) -> Vec<ImportableSession> {
    let opts = crate::services::context_ingest::IngestOptions {
        max_per_source: Some(MAX_PER_SOURCE),
        ..Default::default()
    };
    ingest_project_headlines(project_root, opts)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|t| matches!(t.cli.as_str(), "claude" | "codex" | "pi"))
        .map(|t| {
            let aivo_id = import_session_id(&t.cli, &t.session_id);
            let title = if t.topic.is_empty() {
                format!("{} session", source_label(&t.cli))
            } else {
                t.topic
            };
            ImportableSession {
                aivo_id,
                title,
                updated_at: t.updated_at,
                origin: SessionOrigin {
                    cli: t.cli,
                    foreign_id: t.session_id,
                    source_path: t.source_path,
                },
            }
        })
        .collect()
}

/// Reconstruct a foreign session's transcript (no persistence). Used by the
/// `/resume` path — both the preview pane and resuming a foreign session in
/// memory (which persists as an aivo fork only once a real turn is taken).
pub async fn convert_foreign(origin: &SessionOrigin) -> Result<ImportedTranscript> {
    let path = Path::new(&origin.source_path);
    match origin.cli.as_str() {
        "claude" => import_claude_session(path).await,
        "codex" => import_codex_session(path).await,
        "pi" => import_pi_session(path).await,
        other => Err(anyhow!("unsupported import source: {other}")),
    }
}

/// Divergence slack absorbing file-mtime vs save-clock jitter.
const SOURCE_NEWER_SLACK_SECS: i64 = 5;

/// Outcome of [`resume_foreign`]: the saved fork when one exists, else a
/// fresh conversion of the source transcript.
#[allow(clippy::large_enum_variant)]
pub enum ForeignResume {
    Fork {
        state: crate::services::session_store::CodeSessionState,
        /// The source gained messages after the fork's last save — loading
        /// the fork must not be silent. Heuristic: a fork turn taken after
        /// the source's last message masks it (exact detection needs a
        /// persisted import watermark).
        source_newer: bool,
    },
    Fresh(ImportedTranscript),
}

/// Fork-first foreign resume — the single copy of the policy shared by the
/// TUI and the one-shot: a saved fork (which may hold aivo-side turns) wins
/// over a fresh conversion; an empty source transcript is an error.
pub async fn resume_foreign(
    store: &crate::services::session_store::SessionStore,
    origin: &SessionOrigin,
    source_updated_at: Option<DateTime<Utc>>,
) -> Result<ForeignResume> {
    let aivo_id = import_session_id(&origin.cli, &origin.foreign_id);
    if let Some(state) = store.get_code_session(&aivo_id).await? {
        let source_newer = match (
            source_updated_at,
            DateTime::parse_from_rfc3339(&state.updated_at),
        ) {
            (Some(source), Ok(fork)) => {
                source.signed_duration_since(fork)
                    > chrono::Duration::seconds(SOURCE_NEWER_SLACK_SECS)
            }
            _ => false,
        };
        return Ok(ForeignResume::Fork {
            state,
            source_newer,
        });
    }
    let transcript = convert_foreign(origin)
        .await
        .with_context(|| format!("could not import {} session", origin.cli))?;
    if transcript.messages.is_empty() {
        anyhow::bail!("nothing to import from this session");
    }
    Ok(ForeignResume::Fresh(transcript))
}

/// Display label for a source cli (`claude` → `Claude`).
pub fn source_label(cli: &str) -> &'static str {
    match cli {
        "claude" => "Claude",
        "codex" => "Codex",
        "pi" => "Pi",
        _ => "Imported",
    }
}

// ---------------------------------------------------------------------------
// Shared transcript model. Each source parses into a flat event stream, which a
// common builder shapes into display + engine messages with the invariants the
// engine requires (see `sanitize_alternation`).
// ---------------------------------------------------------------------------

enum ImportEvent {
    User(String),
    Assistant {
        text: Option<String>,
        thinking: Option<String>,
        calls: Vec<ToolCall>,
        model: Option<String>,
    },
    ToolResult {
        id: String,
        content: String,
    },
}

/// Merge adjacent same-role turns (a source may split one turn across lines) so
/// the builder never emits two consecutive `user`/`assistant` engine messages.
/// A `ToolResult` breaks a run, so an assistant's tool calls stay bound to it.
fn coalesce(events: Vec<ImportEvent>) -> Vec<ImportEvent> {
    let mut out: Vec<ImportEvent> = Vec::with_capacity(events.len());
    for ev in events {
        match out.last_mut() {
            Some(ImportEvent::User(prev)) if matches!(ev, ImportEvent::User(_)) => {
                if let ImportEvent::User(text) = ev {
                    push_joined(prev, &text);
                }
            }
            Some(ImportEvent::Assistant {
                text: pt,
                thinking: ptk,
                calls: pc,
                model: pm,
            }) if matches!(ev, ImportEvent::Assistant { .. }) => {
                if let ImportEvent::Assistant {
                    text,
                    thinking,
                    calls,
                    model,
                } = ev
                {
                    merge_opt(pt, text);
                    merge_opt(ptk, thinking);
                    if pm.is_none() {
                        *pm = model;
                    }
                    pc.extend(calls);
                }
            }
            _ => out.push(ev),
        }
    }
    out
}

fn merge_opt(dst: &mut Option<String>, src: Option<String>) {
    if let Some(s) = src {
        match dst {
            Some(existing) => push_joined(existing, &s),
            None => *dst = Some(s),
        }
    }
}

fn push_joined(dst: &mut String, src: &str) {
    if src.is_empty() {
        return;
    }
    if !dst.is_empty() {
        dst.push('\n');
    }
    dst.push_str(src);
}

/// Advance `updated_at` to a line's rfc3339 `timestamp`; the last one wins, so a
/// transcript's newest line sets its mtime-equivalent.
fn capture_timestamp(v: &Value, updated_at: &mut Option<DateTime<Utc>>) {
    if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
        && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
    {
        *updated_at = Some(parsed.with_timezone(&Utc));
    }
}

#[derive(Default)]
struct TranscriptBuilder {
    display: Vec<StoredChatMessage>,
    engine: Vec<Value>,
    /// Unanswered tool-call ids from the most recent assistant, in order.
    pending: Vec<String>,
    /// Builder-stage loss counters; importers seed the parse-stage ones.
    fidelity: ImportFidelity,
}

impl TranscriptBuilder {
    /// Synthesize a placeholder result for any tool call the transcript left
    /// unanswered before a role change, so `assistant(tool_calls)` is always
    /// balanced by `tool` messages (unbalanced → provider 400 on replay).
    fn flush_pending(&mut self) {
        for id in std::mem::take(&mut self.pending) {
            self.fidelity.results_missing += 1;
            self.engine
                .push(json!({"role": "tool", "tool_call_id": id, "content": MISSING_RESULT}));
            self.display.push(display_row(
                "tool_result",
                MISSING_RESULT.to_string(),
                None,
                None,
            ));
        }
    }

    fn push_user(&mut self, text: String) {
        self.flush_pending();
        self.fidelity.user_turns += 1;
        self.engine
            .push(json!({"role": "user", "content": text.clone()}));
        self.display.push(display_row("user", text, None, None));
    }

    fn push_assistant(
        &mut self,
        text: Option<String>,
        thinking: Option<String>,
        calls: Vec<ToolCall>,
        model: Option<String>,
    ) {
        self.flush_pending();
        let text_present = text.as_ref().is_some_and(|t| !t.is_empty());
        if calls.is_empty() {
            if !text_present && thinking.is_none() {
                return; // empty turn — nothing to replay or show
            }
            self.fidelity.assistant_turns += 1;
            let content = text.unwrap_or_default();
            self.engine
                .push(json!({"role": "assistant", "content": content.clone()}));
            self.display
                .push(display_row("assistant", content, thinking, model));
            return;
        }
        self.fidelity.assistant_turns += 1;
        self.fidelity.tool_calls += calls.len();
        // Assistant with tool calls → one engine message (via `assistant_to_openai`
        // so `arguments` is stringified exactly as the live engine emits).
        let am = AssistantMessage {
            content: text.filter(|t| !t.is_empty()),
            tool_calls: calls,
            usage: None,
            truncated: false,
            model: None,
            images: Vec::new(),
        };
        self.engine.push(assistant_to_openai(&am));
        let AssistantMessage {
            content,
            tool_calls,
            ..
        } = am;
        if text_present || thinking.is_some() {
            self.display.push(display_row(
                "assistant",
                content.unwrap_or_default(),
                thinking,
                model,
            ));
        }
        for call in tool_calls {
            self.pending.push(call.id.clone());
            let payload = json!({"name": call.name, "args": call.arguments, "id": call.id});
            self.display
                .push(display_row("tool_call", payload.to_string(), None, None));
        }
    }

    fn push_tool_result(&mut self, id: String, content: String) {
        // Answer only a currently-pending call; drop orphans (e.g. a result whose
        // call was a filtered sidechain) — an unmatched `tool` message 400s.
        if let Some(pos) = self.pending.iter().position(|p| *p == id) {
            self.pending.remove(pos);
            self.fidelity.results_paired += 1;
            self.engine
                .push(json!({"role": "tool", "tool_call_id": id, "content": content.clone()}));
            self.display
                .push(display_row("tool_result", content, None, None));
        } else {
            self.fidelity.results_orphaned += 1;
        }
    }

    fn finish(
        mut self,
        origin_model: Option<String>,
        updated_at: DateTime<Utc>,
    ) -> ImportedTranscript {
        self.flush_pending();
        ImportedTranscript {
            messages: self.display,
            engine_messages: sanitize_alternation(self.engine),
            origin_model,
            updated_at,
            fidelity: self.fidelity,
        }
    }
}

fn display_row(
    role: &str,
    content: String,
    reasoning: Option<String>,
    model: Option<String>,
) -> StoredChatMessage {
    StoredChatMessage {
        role: role.to_string(),
        content,
        reasoning_content: reasoning.filter(|r| !r.trim().is_empty()),
        id: None,
        timestamp: None,
        attachments: None,
        model,
    }
}

/// Enforce the engine's replay invariants on the wire log: first message is
/// `user`, no `tool` message is directly followed by `user` (the bridge maps a
/// tool result to a user turn → two consecutive users otherwise), and the log
/// ends on an `assistant` turn. Consecutive same-role text is already prevented
/// by `coalesce`. Injected messages are engine-only (never shown).
fn sanitize_alternation(engine: Vec<Value>) -> Vec<Value> {
    if engine.is_empty() {
        return engine;
    }
    let role_of = |m: &Value| {
        m.get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string()
    };
    let mut out: Vec<Value> = Vec::with_capacity(engine.len() + 2);
    if role_of(&engine[0]) != "user" {
        out.push(json!({"role": "user", "content": IMPORT_LEAD}));
    }
    for msg in engine {
        if role_of(&msg) == "user" && out.last().map(role_of).as_deref() == Some("tool") {
            out.push(json!({"role": "assistant", "content": RESUMED_MARK}));
        }
        out.push(msg);
    }
    if out.last().map(role_of).as_deref() != Some("assistant") {
        out.push(json!({"role": "assistant", "content": RESUMED_MARK}));
    }
    out
}

/// `None` for blank separators (uncounted) and torn/partial lines (counted as lost).
fn parse_transcript_line(line: &str, fidelity: &mut ImportFidelity) -> Option<Value> {
    if line.trim().is_empty() {
        return None;
    }
    match serde_json::from_str(line) {
        Ok(v) => Some(v),
        Err(_) => {
            fidelity.torn_lines += 1;
            None
        }
    }
}

/// `fidelity` seeds the parse-stage counters; the builder adds its own.
fn build_transcript(
    events: Vec<ImportEvent>,
    fidelity: ImportFidelity,
    origin_model: Option<String>,
    updated_at: Option<DateTime<Utc>>,
) -> ImportedTranscript {
    let mut builder = TranscriptBuilder {
        fidelity,
        ..Default::default()
    };
    apply_events(&mut builder, events);
    builder.finish(origin_model, updated_at.unwrap_or_else(Utc::now))
}

// ---------------------------------------------------------------------------
// Claude Code: ~/.claude/projects/<enc-cwd>/*.jsonl
// ---------------------------------------------------------------------------

/// Reconstruct a Claude Code transcript. Walks lines in file order (the
/// `parentUuid` chain is linear once sidechains are removed), skipping
/// sidechain/meta rows and non user/assistant events.
pub async fn import_claude_session(path: &Path) -> Result<ImportedTranscript> {
    let data = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    let mut events: Vec<ImportEvent> = Vec::new();
    let mut origin_model: Option<String> = None;
    let mut updated_at: Option<DateTime<Utc>> = None;
    let mut fidelity = ImportFidelity::default();

    for line in data.lines() {
        let Some(v) = parse_transcript_line(line, &mut fidelity) else {
            continue;
        };
        // Sidechains are subagent transcripts (counted — the Task summary on the
        // main thread survives); meta lines are harness noise (not counted).
        if flag(&v, "isSidechain") {
            fidelity.sidechain_lines += 1;
            continue;
        }
        if flag(&v, "isMeta") {
            continue;
        }
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }
        capture_timestamp(&v, &mut updated_at);
        let message = v.get("message");
        match kind {
            "assistant" => {
                let model = message
                    .and_then(|m| m.get("model"))
                    .and_then(|m| m.as_str())
                    .map(str::to_string);
                if origin_model.is_none() {
                    origin_model = model.clone();
                }
                let (text, thinking, calls) = parse_assistant_blocks(
                    message.and_then(|m| m.get("content")),
                    "tool_use",
                    "input",
                );
                if text.is_some() || thinking.is_some() || !calls.is_empty() {
                    events.push(ImportEvent::Assistant {
                        text,
                        thinking,
                        calls,
                        model,
                    });
                }
            }
            "user" => parse_claude_user(message, &mut events, &mut fidelity),
            _ => {}
        }
    }

    Ok(build_transcript(events, fidelity, origin_model, updated_at))
}

/// Parse an assistant message's content into `(text, thinking, tool_calls)`.
/// `text`/`thinking` blocks accumulate; tool blocks (type `tool_type`, args
/// under `args_field`) become `ToolCall`s. Claude also allows plain-string
/// content; Pi content is always an array — hence the per-source parameters.
fn parse_assistant_blocks(
    content: Option<&Value>,
    tool_type: &str,
    args_field: &str,
) -> (Option<String>, Option<String>, Vec<ToolCall>) {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut calls = Vec::new();
    if let Some(s) = content.and_then(|c| c.as_str()) {
        text.push_str(s);
    } else if let Some(arr) = content.and_then(|c| c.as_array()) {
        for block in arr {
            match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "text" => append_block_text(&mut text, block, "text"),
                "thinking" => append_block_text(&mut thinking, block, "thinking"),
                kind if kind == tool_type => {
                    let id = block.get("id").and_then(|s| s.as_str()).unwrap_or("");
                    if !id.is_empty() {
                        calls.push(ToolCall {
                            id: id.to_string(),
                            name: block
                                .get("name")
                                .and_then(|s| s.as_str())
                                .unwrap_or("tool")
                                .to_string(),
                            arguments: block.get(args_field).cloned().unwrap_or(Value::Null),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    (
        (!text.is_empty()).then_some(text),
        (!thinking.is_empty()).then_some(thinking),
        calls,
    )
}

/// A Claude `user` line is either a real user turn (string content) or the
/// carrier for tool outputs (an array of `tool_result` blocks) — often both.
fn parse_claude_user(
    message: Option<&Value>,
    out: &mut Vec<ImportEvent>,
    fidelity: &mut ImportFidelity,
) {
    let Some(content) = message.and_then(|m| m.get("content")) else {
        return;
    };
    if let Some(s) = content.as_str() {
        let s = s.trim();
        if !s.is_empty() {
            out.push(ImportEvent::User(s.to_string()));
        }
        return;
    }
    let Some(arr) = content.as_array() else {
        return;
    };
    let mut text = String::new();
    for block in arr {
        match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "text" => append_block_text(&mut text, block, "text"),
            "image" => {
                fidelity.images_omitted += 1;
                push_joined(&mut text, IMAGE_PLACEHOLDER);
            }
            "tool_result" => {
                let id = block
                    .get("tool_use_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                out.push(ImportEvent::ToolResult {
                    id,
                    content: claude_result_text(block.get("content"), fidelity),
                });
            }
            _ => {}
        }
    }
    let text = text.trim();
    if !text.is_empty() {
        out.push(ImportEvent::User(text.to_string()));
    }
}

/// A `tool_result.content` is a string or an array of text/image blocks.
fn claude_result_text(content: Option<&Value>, fidelity: &mut ImportFidelity) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            let mut buf = String::new();
            for block in arr {
                match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "text" => append_block_text(&mut buf, block, "text"),
                    "image" => {
                        fidelity.images_omitted += 1;
                        push_joined(&mut buf, IMAGE_PLACEHOLDER);
                    }
                    _ => {}
                }
            }
            buf
        }
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn append_block_text(dst: &mut String, block: &Value, field: &str) {
    if let Some(t) = block.get(field).and_then(|t| t.as_str()) {
        push_joined(dst, t);
    }
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(|b| b.as_bool()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Codex: ~/.codex/sessions/YYYY/MM/DD/*.jsonl
// ---------------------------------------------------------------------------

/// Reconstruct a Codex transcript from its `response_item` stream. `message`
/// items are user/assistant text; `function_call`/`function_call_output` map to
/// assistant tool calls + tool results; `reasoning` folds into the next
/// assistant turn (via `coalesce`).
pub async fn import_codex_session(path: &Path) -> Result<ImportedTranscript> {
    let data = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    let mut events: Vec<ImportEvent> = Vec::new();
    let mut origin_model: Option<String> = None;
    let mut updated_at: Option<DateTime<Utc>> = None;
    let mut fidelity = ImportFidelity::default();

    for line in data.lines() {
        let Some(v) = parse_transcript_line(line, &mut fidelity) else {
            continue;
        };
        capture_timestamp(&v, &mut updated_at);
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
            && origin_model.is_none()
        {
            origin_model = payload
                .get("model")
                .and_then(|m| m.as_str())
                .map(str::to_string);
        }
        if kind != "response_item" {
            continue;
        }
        let Some(payload) = v.get("payload") else {
            continue;
        };
        match payload.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "message" => {
                let role = payload.get("role").and_then(|s| s.as_str()).unwrap_or("");
                let text = extract_codex_message_text(payload).unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                match role {
                    "user" => events.push(ImportEvent::User(text)),
                    "assistant" => events.push(ImportEvent::Assistant {
                        text: Some(text),
                        thinking: None,
                        calls: vec![],
                        model: origin_model.clone(),
                    }),
                    _ => {}
                }
            }
            "reasoning" => {
                if let Some(t) = codex_reasoning_text(payload) {
                    events.push(ImportEvent::Assistant {
                        text: None,
                        thinking: Some(t),
                        calls: vec![],
                        model: origin_model.clone(),
                    });
                }
            }
            item_type @ ("function_call" | "custom_tool_call") => {
                let id = codex_call_id(payload);
                if !id.is_empty() {
                    // custom_tool_call carries freeform `input`, not an
                    // `arguments` JSON string; wrap it as the wire path does.
                    let arguments = if item_type == "custom_tool_call" {
                        json!({"input": payload.get("input").and_then(|s| s.as_str()).unwrap_or("")})
                    } else {
                        payload
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .and_then(|s| serde_json::from_str(s).ok())
                            .or_else(|| payload.get("arguments").cloned())
                            .unwrap_or(Value::Null)
                    };
                    events.push(ImportEvent::Assistant {
                        text: None,
                        thinking: None,
                        calls: vec![ToolCall {
                            id,
                            name: payload
                                .get("name")
                                .and_then(|s| s.as_str())
                                .unwrap_or("tool")
                                .to_string(),
                            arguments,
                        }],
                        model: origin_model.clone(),
                    });
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                let id = codex_call_id(payload);
                events.push(ImportEvent::ToolResult {
                    id,
                    content: codex_output_text(payload.get("output")),
                });
            }
            _ => {}
        }
    }

    Ok(build_transcript(events, fidelity, origin_model, updated_at))
}

fn codex_call_id(payload: &Value) -> String {
    payload
        .get("call_id")
        .or_else(|| payload.get("id"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string()
}

fn codex_output_text(output: Option<&Value>) -> String {
    match output {
        Some(Value::String(s)) => s.clone(),
        // Codex wraps some outputs as `{ "output": "...", "metadata": {...} }`.
        Some(Value::Object(map)) => map
            .get("output")
            .and_then(|o| o.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| Value::Object(map.clone()).to_string()),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn codex_reasoning_text(payload: &Value) -> Option<String> {
    let mut buf = String::new();
    for key in ["summary", "content"] {
        if let Some(arr) = payload.get(key).and_then(|v| v.as_array()) {
            for block in arr {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    push_joined(&mut buf, t);
                }
            }
        }
    }
    (!buf.is_empty()).then_some(buf)
}

// ---------------------------------------------------------------------------
// Pi: ~/.pi/agent/sessions/--<cwd-dashes>--/*.jsonl
// ---------------------------------------------------------------------------

/// Reconstruct a Pi transcript. `message` lines carry role `user`/`assistant`/
/// `toolResult`; an assistant's content array holds `text`/`thinking`/`toolCall`
/// blocks (`toolCall.arguments` is already an object), and a `toolResult`
/// references its call via `toolCallId`.
pub async fn import_pi_session(path: &Path) -> Result<ImportedTranscript> {
    let data = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;

    let mut events: Vec<ImportEvent> = Vec::new();
    let mut origin_model: Option<String> = None;
    let mut updated_at: Option<DateTime<Utc>> = None;
    let mut fidelity = ImportFidelity::default();

    for line in data.lines() {
        let Some(v) = parse_transcript_line(line, &mut fidelity) else {
            continue;
        };
        capture_timestamp(&v, &mut updated_at);
        if v.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let Some(message) = v.get("message") else {
            continue;
        };
        match message.get("role").and_then(|r| r.as_str()).unwrap_or("") {
            "user" => {
                if let Some(text) = extract_pi_text(message) {
                    let text = text.trim();
                    if !text.is_empty() {
                        events.push(ImportEvent::User(text.to_string()));
                    }
                }
            }
            "assistant" => {
                let model = message
                    .get("model")
                    .and_then(|m| m.as_str())
                    .map(str::to_string);
                if origin_model.is_none() {
                    origin_model = model.clone();
                }
                let (text, thinking, calls) =
                    parse_assistant_blocks(message.get("content"), "toolCall", "arguments");
                if text.is_some() || thinking.is_some() || !calls.is_empty() {
                    events.push(ImportEvent::Assistant {
                        text,
                        thinking,
                        calls,
                        model,
                    });
                }
            }
            "toolResult" => {
                let id = message
                    .get("toolCallId")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                events.push(ImportEvent::ToolResult {
                    id,
                    content: extract_pi_text(message).unwrap_or_default(),
                });
            }
            _ => {}
        }
    }

    Ok(build_transcript(events, fidelity, origin_model, updated_at))
}

fn apply_events(builder: &mut TranscriptBuilder, events: Vec<ImportEvent>) {
    for ev in coalesce(events) {
        match ev {
            ImportEvent::User(text) => builder.push_user(text),
            ImportEvent::Assistant {
                text,
                thinking,
                calls,
                model,
            } => builder.push_assistant(text, thinking, calls, model),
            ImportEvent::ToolResult { id, content } => builder.push_tool_result(id, content),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn role(m: &Value) -> &str {
        m.get("role").and_then(|r| r.as_str()).unwrap_or("")
    }

    fn imp(cli: &str, foreign_id: &str) -> ImportableSession {
        ImportableSession {
            origin: SessionOrigin {
                cli: cli.into(),
                foreign_id: foreign_id.into(),
                source_path: "/tmp/x.jsonl".into(),
            },
            title: "t".into(),
            updated_at: Utc::now(),
            aivo_id: import_session_id(cli, foreign_id),
        }
    }

    fn ids(v: &[&str]) -> std::vec::IntoIter<String> {
        v.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn match_unique_aivo_prefix_resolves_full_id() {
        match match_resume_candidates(ids(&["sess-1234", "other-99"]), vec![], "sess-12") {
            ResumeTarget::AivoSession(id) => assert_eq!(id, "sess-1234"),
            _ => panic!("expected AivoSession"),
        }
    }

    #[test]
    fn match_foreign_by_source_or_import_id_prefix() {
        let one = imp("claude", "049faa11-2222");
        // The source tool's own id (what `aivo logs` shows)…
        match match_resume_candidates(ids(&[]), vec![one.clone()], "049fa") {
            ResumeTarget::Foreign(f) => assert_eq!(f.origin.foreign_id, "049faa11-2222"),
            _ => panic!("expected Foreign via source id"),
        }
        // …and the deterministic aivo import id both match.
        let by_import = one.aivo_id.clone();
        match match_resume_candidates(ids(&[]), vec![one], &by_import) {
            ResumeTarget::Foreign(f) => assert_eq!(f.aivo_id, by_import),
            _ => panic!("expected Foreign via import id"),
        }
    }

    #[test]
    fn match_already_forked_import_collapses_to_saved_session() {
        // The fork exists as a saved aivo session AND the source is still
        // listed — same conversation, so this must not read as ambiguous.
        let one = imp("claude", "feedbeef-4242");
        let fork_id = one.aivo_id.clone();
        match match_resume_candidates(ids(&[fork_id.as_str()]), vec![one], &fork_id) {
            ResumeTarget::AivoSession(id) => assert_eq!(id, fork_id),
            _ => panic!("expected the saved fork to win"),
        }
    }

    #[test]
    fn match_ambiguous_across_sources_and_unknown() {
        let foreign = imp("codex", "abc-junction");
        match match_resume_candidates(ids(&["abc-native"]), vec![foreign], "abc") {
            ResumeTarget::Ambiguous(msg) => assert!(msg.contains("2 matches")),
            _ => panic!("expected Ambiguous"),
        }
        assert!(matches!(
            match_resume_candidates(ids(&["sess-1"]), vec![], "zzz"),
            ResumeTarget::Unknown
        ));
    }

    #[tokio::test]
    async fn resolve_resume_target_store_rungs() {
        let dir = tempfile::tempdir().unwrap();
        let store =
            crate::services::session_store::SessionStore::with_path(dir.path().join("cfg.json"));
        let key_id = store
            .add_key_with_protocol("k", "https://api.example.com", None, "sk-test")
            .await
            .unwrap();
        store
            .save_code_session_with_id(
                &key_id,
                "https://api.example.com",
                "/tmp/demo",
                "sess-abcd1234",
                "m1",
                None,
                &[],
                "t",
                "p",
                crate::services::session_store::SessionTokens::default(),
                0.0,
            )
            .await
            .unwrap();

        // Exact id, unique prefix, and no-match — against a real store.
        for (sel, want_hit) in [("sess-abcd1234", true), ("sess-abc", true), ("nope", false)] {
            match resolve_resume_target(&store, dir.path(), sel).await {
                ResumeTarget::AivoSession(id) if want_hit => assert_eq!(id, "sess-abcd1234"),
                ResumeTarget::Unknown if !want_hit => {}
                _ => panic!("unexpected resolution for {sel}"),
            }
        }
    }

    async fn import_claude_str(jsonl: &str) -> ImportedTranscript {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        import_claude_session(&path).await.unwrap()
    }

    #[tokio::test]
    async fn claude_full_turn_reconstructs_tools() {
        let jsonl = r#"
{"type":"user","sessionId":"s1","cwd":"/p","message":{"role":"user","content":"read main.rs"}}
{"type":"assistant","sessionId":"s1","message":{"role":"assistant","model":"claude-sonnet-4","content":[{"type":"text","text":"On it."},{"type":"tool_use","id":"tu_1","name":"read_file","input":{"path":"main.rs"}}]}}
{"type":"user","sessionId":"s1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"fn main() {}"}]}}
{"type":"assistant","sessionId":"s1","message":{"role":"assistant","model":"claude-sonnet-4","content":[{"type":"text","text":"It's empty."}]}}
"#;
        let t = import_claude_str(jsonl).await;

        // No system message; user-first; ends on assistant.
        assert_eq!(role(&t.engine_messages[0]), "user");
        assert!(t.engine_messages.iter().all(|m| role(m) != "system"));
        assert_eq!(role(t.engine_messages.last().unwrap()), "assistant");

        // Assistant tool call: arguments is a STRING; tool_call_id matches.
        let asst = t
            .engine_messages
            .iter()
            .find(|m| m.get("tool_calls").is_some())
            .unwrap();
        let call = &asst["tool_calls"][0];
        assert_eq!(call["id"], "tu_1");
        assert!(call["function"]["arguments"].is_string());
        let tool = t
            .engine_messages
            .iter()
            .find(|m| role(m) == "tool")
            .unwrap();
        assert_eq!(tool["tool_call_id"], "tu_1");
        assert_eq!(tool["content"], "fn main() {}");

        // Display roles in order.
        let roles: Vec<&str> = t.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool_call", "tool_result", "assistant"]
        );
        // Provenance model carried on assistant rows.
        assert_eq!(t.origin_model.as_deref(), Some("claude-sonnet-4"));
        assert_eq!(t.messages[1].model.as_deref(), Some("claude-sonnet-4"));

        let f = &t.fidelity;
        assert_eq!(f.tier(), FidelityTier::Full);
        assert_eq!(
            (
                f.user_turns,
                f.assistant_turns,
                f.tool_calls,
                f.results_paired
            ),
            (1, 2, 1, 1)
        );
        assert!(f.notes().is_empty());
        assert_eq!(
            f.summary("Claude"),
            "Imported from Claude · fidelity full · 3 turns · 1/1 tool calls"
        );
    }

    #[tokio::test]
    async fn claude_skips_sidechain_and_torn_lines() {
        let jsonl = "\n{ this is a torn line\n{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hi there friend\"}}\n{\"type\":\"assistant\",\"isSidechain\":true,\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"subagent noise\"}]}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}}\n";
        let t = import_claude_str(jsonl).await;
        assert!(!t.engine_messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("subagent noise"))
        }));
        let roles: Vec<&str> = t.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant"]);
        // Torn line = real loss → partial; sidechain is counted but informational.
        assert_eq!(t.fidelity.torn_lines, 1);
        assert_eq!(t.fidelity.sidechain_lines, 1);
        assert_eq!(t.fidelity.tier(), FidelityTier::Partial);
    }

    #[tokio::test]
    async fn claude_orphan_tool_result_dropped() {
        // A tool_result with no matching prior tool_use must not become an
        // unmatched `tool` message (would 400 on replay).
        let jsonl = r#"
{"type":"user","message":{"role":"user","content":"go do the thing please"}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"ghost","content":"orphan output"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}
"#;
        let t = import_claude_str(jsonl).await;
        assert!(!t.engine_messages.iter().any(|m| role(m) == "tool"));
        // Consecutive users are coalesced into one.
        assert_eq!(
            t.engine_messages
                .iter()
                .filter(|m| role(m) == "user")
                .count(),
            1
        );
        assert_eq!(t.fidelity.results_orphaned, 1);
        assert_eq!(t.fidelity.tier(), FidelityTier::Partial);
        assert_eq!(t.fidelity.notes(), vec!["1 orphaned tool result dropped"]);
    }

    #[tokio::test]
    async fn claude_missing_result_synthesized_before_next_turn() {
        // Two calls, only one answered, then a user turn → the unanswered call
        // gets a placeholder result so the assistant turn stays balanced.
        let jsonl = r#"
{"type":"user","message":{"role":"user","content":"do two things at once"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"read_file","input":{}},{"type":"tool_use","id":"b","name":"read_file","input":{}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}}
{"type":"user","message":{"role":"user","content":"never mind"}}
"#;
        let t = import_claude_str(jsonl).await;
        let tool_ids: Vec<&str> = t
            .engine_messages
            .iter()
            .filter(|m| role(m) == "tool")
            .map(|m| m["tool_call_id"].as_str().unwrap())
            .collect();
        assert!(tool_ids.contains(&"a") && tool_ids.contains(&"b"));
        assert_eq!(role(t.engine_messages.last().unwrap()), "assistant");
        // One call answered, one placeholder → high (conversation complete).
        let f = &t.fidelity;
        assert_eq!(
            (f.tool_calls, f.results_paired, f.results_missing),
            (2, 1, 1)
        );
        assert_eq!(f.tier(), FidelityTier::High);
        assert_eq!(
            f.summary("Claude"),
            "Imported from Claude · fidelity high · 3 turns · 1/2 tool calls"
        );
    }

    #[tokio::test]
    async fn fidelity_counts_images_and_notes_read_cleanly() {
        let jsonl = r#"
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"what is in this screenshot"},{"type":"image","source":{"type":"base64","data":"…"}}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"read_file","input":{}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"text","text":"ok"},{"type":"image","source":{"type":"base64","data":"…"}}]}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"a diagram"}]}}
"#;
        let t = import_claude_str(jsonl).await;
        let f = &t.fidelity;
        // One image in the user turn, one inside the tool result.
        assert_eq!(f.images_omitted, 2);
        assert_eq!(f.tier(), FidelityTier::High);
        assert_eq!(f.notes(), vec!["2 images omitted"]);
        // Torn/orphan loss outranks placeholder-only degradation.
        let worse = ImportFidelity {
            torn_lines: 1,
            ..f.clone()
        };
        assert_eq!(worse.tier(), FidelityTier::Partial);
    }

    #[tokio::test]
    async fn empty_session_has_no_messages() {
        let t = import_claude_str("{\"type\":\"summary\",\"summary\":\"x\"}\n").await;
        assert!(t.messages.is_empty());
        assert!(t.engine_messages.is_empty());
    }

    #[tokio::test]
    async fn codex_pairs_function_call_and_output() {
        let jsonl = r#"
{"type":"session_meta","payload":{"id":"c1","cwd":"/p","model":"gpt-5.5"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"list files"}]}}
{"type":"response_item","payload":{"type":"function_call","name":"list_dir","arguments":"{\"path\":\".\"}","call_id":"fc_1"}}
{"type":"response_item","payload":{"type":"function_call_output","call_id":"fc_1","output":"a.rs\nb.rs"}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Two files."}]}}
"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let t = import_codex_session(&path).await.unwrap();

        assert_eq!(role(&t.engine_messages[0]), "user");
        let asst = t
            .engine_messages
            .iter()
            .find(|m| m.get("tool_calls").is_some())
            .unwrap();
        assert_eq!(asst["tool_calls"][0]["id"], "fc_1");
        assert!(asst["tool_calls"][0]["function"]["arguments"].is_string());
        let tool = t
            .engine_messages
            .iter()
            .find(|m| role(m) == "tool")
            .unwrap();
        assert_eq!(tool["tool_call_id"], "fc_1");
        assert_eq!(tool["content"], "a.rs\nb.rs");
        assert_eq!(t.origin_model.as_deref(), Some("gpt-5.5"));
    }

    #[tokio::test]
    async fn codex_pairs_custom_tool_call_and_output() {
        let jsonl = r#"
{"type":"session_meta","payload":{"id":"c2","cwd":"/p","model":"gpt-5.5"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"fix it"}]}}
{"type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch\n*** End Patch","call_id":"ct_1"}}
{"type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"ct_1","output":"Done!"}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Patched."}]}}
"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c2.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let t = import_codex_session(&path).await.unwrap();

        let asst = t
            .engine_messages
            .iter()
            .find(|m| m.get("tool_calls").is_some())
            .unwrap();
        assert_eq!(asst["tool_calls"][0]["id"], "ct_1");
        assert_eq!(asst["tool_calls"][0]["function"]["name"], "apply_patch");
        let args = asst["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(args.contains("Begin Patch"), "got: {args}");
        let tool = t
            .engine_messages
            .iter()
            .find(|m| role(m) == "tool")
            .unwrap();
        assert_eq!(tool["tool_call_id"], "ct_1");
        assert_eq!(tool["content"], "Done!");
    }

    #[tokio::test]
    async fn pi_pairs_toolcall_and_result() {
        let jsonl = r#"
{"type":"session","id":"p1"}
{"type":"message","message":{"role":"user","content":[{"type":"text","text":"list files"}]}}
{"type":"message","message":{"role":"assistant","model":"gemma-4","content":[{"type":"thinking","thinking":"let me look"},{"type":"text","text":"Sure."},{"type":"toolCall","id":"call_1","name":"bash","arguments":{"command":"ls"}}]}}
{"type":"message","message":{"role":"toolResult","toolCallId":"call_1","toolName":"bash","content":[{"type":"text","text":"a.rs\nb.rs"}]}}
{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Two files."}]}}
"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let t = import_pi_session(&path).await.unwrap();

        assert_eq!(role(&t.engine_messages[0]), "user");
        let asst = t
            .engine_messages
            .iter()
            .find(|m| m.get("tool_calls").is_some())
            .unwrap();
        assert_eq!(asst["tool_calls"][0]["id"], "call_1");
        assert!(asst["tool_calls"][0]["function"]["arguments"].is_string());
        let tool = t
            .engine_messages
            .iter()
            .find(|m| role(m) == "tool")
            .unwrap();
        assert_eq!(tool["tool_call_id"], "call_1");
        assert_eq!(tool["content"], "a.rs\nb.rs");
        assert_eq!(t.origin_model.as_deref(), Some("gemma-4"));
        // Thinking is preserved on the display row (reasoning), not the wire.
        assert!(t.messages.iter().any(|m| {
            m.reasoning_content
                .as_deref()
                .is_some_and(|r| r.contains("let me look"))
        }));
    }

    #[test]
    fn import_id_is_short_stable_and_distinct() {
        let id = import_session_id("claude", "3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c");
        // `<cli>-<8 hex>` regardless of how long/messy the source id is.
        assert!(id.starts_with("claude-"));
        let hash = id.strip_prefix("claude-").unwrap();
        assert_eq!(hash.len(), 8);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        // Deterministic (dedupes a re-list) and per-source distinct.
        assert_eq!(
            id,
            import_session_id("claude", "3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c")
        );
        assert_ne!(id, import_session_id("claude", "other-id"));
        // The leading tool tag stays parseable for the source label.
        assert_eq!(import_source_label(&id), Some("Claude"));
        // A native UUID (leading hex) is not a fork; the legacy `import-` form is.
        assert_eq!(import_source_label("3f2a1b4c-5d6e-7f8a"), None);
        assert_eq!(import_source_label("import-pi-019f15e1-a4bd"), Some("Pi"));
    }

    /// Real-data smoke test against the developer's own `~/.claude` / `~/.codex`.
    /// Ignored in CI (environment-dependent); run with
    /// `cargo test ... -- --ignored --nocapture` from the repo root.
    #[tokio::test]
    #[ignore = "reads the developer's real ~/.claude and ~/.codex"]
    async fn lists_real_sessions_for_this_cwd() {
        let cwd = std::env::current_dir().unwrap();
        let found = list_importable_sessions(&cwd).await;
        eprintln!("importable sessions for {}: {}", cwd.display(), found.len());
        for s in found.iter().take(8) {
            eprintln!("  [{}] {} — {}", s.origin.cli, s.aivo_id, s.title);
        }
        assert!(!found.is_empty(), "expected foreign sessions for this repo");
    }
}
