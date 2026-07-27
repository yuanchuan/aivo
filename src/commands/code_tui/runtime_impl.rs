use super::*;

use crate::agent::engine::RewindOutcome;
use crate::agent::protocol::Decision;
use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CursorAcpSession, CursorChunk, CursorTurnResult};
use crate::services::vision_describe;
use anyhow::Context;

/// Default cap on total `/goal` turns (override: `AIVO_GOAL_MAX_ITERS`).
const GOAL_DEFAULT_MAX_ITERS: usize = 20;
/// Framing prepended to the first `/goal` turn so the agent knows the contract.
const GOAL_PREAMBLE: &str = "[Goal mode] Work autonomously toward this objective, doing as many \
steps as it takes — build directly without pausing to confirm the plan first. When the objective \
is FULLY achieved, reply with exactly `GOAL COMPLETE` on its own line. If anything remains, keep \
going. Use `take_note` for key decisions and dead-ends so they survive context compaction.";
/// Self-checking continuation sent between goal turns. Restates the objective
/// each round so it survives context compaction on long runs.
fn goal_continue_message(objective: &str) -> String {
    format!(
        "Continue toward the goal: {objective}\n\nIf the objective is now fully met, reply with \
exactly `GOAL COMPLETE` and nothing else; otherwise do the next step."
    )
}

/// The `/review` directive: a read-only, line-by-line review of a diff.
/// Twin: `agent/builtin_agents/evaluate.md` (the delegatable review sub-agent) —
/// keep review-policy changes in sync.
const REVIEW_PREAMBLE: &str = "[Code review] Review the changes below as a senior engineer \
would before a merge. This is READ-ONLY: do not modify, create, or delete any file.\n\
\n\
1. Establish the diff. With no target given, review the working diff: `git status --short` \
plus `git diff HEAD` (staged + unstaged). If the working tree is clean, review the last \
commit (`git show HEAD`). If a target is given and names a git ref, review \
`git diff <target>...HEAD` (plus the working diff); if it names a path or topic, restrict \
the review to that scope.\n\
2. For every changed hunk, read enough surrounding code to judge it in context — follow \
callers and callees a change could break. Never judge from the diff alone.\n\
3. Report only findings that matter: correctness bugs, edge cases, races, security issues, \
API misuse, behavior changes callers don't expect, dead or duplicated logic, missing tests \
for risky changes. Skip style nits a formatter or linter would catch.\n\
4. Present the review as:\n\
   - One finding per bullet: `file:line — [P0|P1|P2] summary`, then 1-3 sentences of why \
it's wrong (with a concrete failure scenario) and a suggested fix. P0 = must fix before \
merge, P1 = should fix, P2 = polish. Order by severity.\n\
   - A closing verdict paragraph: overall quality, whether it's safe to merge, and what \
you checked but found sound.\n\
   - If there are no findings, say so explicitly and list what you verified.";

/// Bare-`/plan` kick-off: interview for an objective. `ask_user` keeps the turn
/// alive — a prose question would end it and be stamped as a drafted plan.
pub(super) const PLAN_KICKOFF_MESSAGE: &str = "The user entered plan mode without saying what \
to plan. Interview them for the objective before any planning:\n\
1. Orient briefly (git status/log, project layout) — only as far as it yields concrete \
suggestions.\n\
2. Call `ask_user` asking what they want to build, fix, or change — offer candidates you \
noticed as options; they can also type their own.\n\
3. With the objective clear, investigate the code it touches and call `exit_plan_mode` with \
the complete plan.\n\
Never call `exit_plan_mode` before the user states an objective; if their answer is ambiguous, \
ask one focused follow-up.";

/// The `/plan go` message — the plan is already in engine history, so only the
/// go-ahead + guidance is sent.
pub(super) fn plan_go_message(guidance: &str) -> String {
    let mut msg = "The plan above is approved — implement it now.".to_string();
    if !guidance.is_empty() {
        msg.push_str("\n\nAdditional guidance: ");
        msg.push_str(guidance);
    }
    msg
}

/// The `/plan resume` kick-off: this engine has never seen the plan (drafted in
/// another session), so the text rides along for re-validation before approval.
pub(super) fn plan_carryover_message(draft: &str) -> String {
    format!(
        "An implementation plan carried over from a previous session is below. You are in plan \
mode. Review it against the CURRENT state of the code — it may have drifted since the plan was \
written, and parts may already be done. Revise as needed (investigate with read-only tools), \
then call `exit_plan_mode` with the final plan.\n\n<carried-over-plan>\n{draft}\n</carried-over-plan>"
    )
}

/// Bare `/plan resume` for THIS session's interrupted plan (Esc mid-execution):
/// the transcript is already in context, so only a continue nudge goes out —
/// with the checklist re-stated so completed marks survive context trimming.
pub(super) fn plan_interrupted_message(steps: &str) -> String {
    format!(
        "Execution of the approved plan below was interrupted. Verify the steps marked \
\"completed\" still hold, then continue from the first non-completed step. Call `update_plan` \
with this checklist (keeping the completed marks) as you resume, and keep it updated as you \
go.\n\n<interrupted-checklist>\n{steps}\n</interrupted-checklist>"
    )
}

/// The `/plan resume` continuation for a mid-execution plan: the checklist rides
/// along; already approved in its session, so execution continues — no re-approval.
pub(super) fn plan_continue_message(steps: &str) -> String {
    format!(
        "An execution checklist carried over from a previous session is below — that session \
approved the plan and completed the steps marked \"completed\" before running out of context. \
Quickly verify the completed steps still hold in the current code, then continue from the first \
non-completed step. Call `update_plan` with this checklist (keeping the completed marks) as you \
start, and keep it updated as you go.\n\n<carried-over-checklist>\n{steps}\n</carried-over-checklist>"
    )
}

fn goal_max_iterations() -> usize {
    crate::services::system_env::env_parse("AIVO_GOAL_MAX_ITERS")
        .filter(|n| *n > 0)
        .unwrap_or(GOAL_DEFAULT_MAX_ITERS)
}

/// Whole-line match so prose mentioning the marker doesn't end the loop; trims
/// markdown wrapping (the prompts show the marker in backticks, models echo
/// them — some quote or blockquote it instead).
fn signals_goal_complete(text: &str) -> bool {
    text.lines().any(|line| {
        line.trim()
            .trim_matches(|c: char| matches!(c, '`' | '*' | '_' | '.' | '!' | ' ' | '"' | '>'))
            .eq_ignore_ascii_case("GOAL COMPLETE")
    })
}

/// `AIVO_AGENT_SELF_CORRECT=1` — post-edit verification in interactive turns. Opt-in
/// here (headless defaults on): a full-suite run stalls a watched turn.
fn env_self_correct() -> bool {
    crate::services::system_env::env_flag("AIVO_AGENT_SELF_CORRECT").unwrap_or(false)
}

/// Goal mode verifies at declared-done by default; `AIVO_GOAL_VERIFY=0` opts out.
fn goal_verify_enabled() -> bool {
    crate::services::system_env::env_flag("AIVO_GOAL_VERIFY").unwrap_or(true)
}

/// Build the message a `/<skill> [args]` invocation sends to the model. If the
/// body uses the `$ARGUMENTS` placeholder, the args are substituted in place;
/// otherwise the body is sent under a short directive with the args (if any)
/// appended as input. The full instructions are inlined so the skill runs
/// deterministically regardless of whether the model would call the `skill` tool.
pub(super) fn expand_skill_invocation(
    skill: &crate::agent::skills::Skill,
    args: Option<&str>,
) -> String {
    let args = args.unwrap_or("").trim();
    let body = skill.instructions();
    if body.contains("$ARGUMENTS") {
        return body.replace("$ARGUMENTS", args);
    }
    let mut out = format!(
        "Use the \"{}\" skill. Follow these instructions:\n\n{}",
        skill.name, body
    );
    if !args.is_empty() {
        out.push_str(&format!("\n\nInput: {args}"));
    }
    out
}

/// Inverse of [`expand_skill_invocation`], for DISPLAY and LOGGING only: if
/// `content` is an expanded skill invocation, recover the compact `/name args`
/// the user typed — so the transcript and `aivo logs` show `/baidu-search 歌曲`
/// instead of the whole inlined `SKILL.md` body (the model still receives the
/// full body via `content`). Returns `None` for ordinary messages and for
/// `$ARGUMENTS`-style skills, which substitute in place and leave no recoverable
/// wrapper. The first-line marker is a fixed string we emit and skill names are
/// `[A-Za-z0-9_-]` (no quotes), so the match is unambiguous.
pub(crate) fn skill_invocation_label(content: &str) -> Option<String> {
    let name = content
        .lines()
        .next()?
        .strip_prefix("Use the \"")?
        .strip_suffix("\" skill. Follow these instructions:")?;
    if !crate::agent::skills::is_valid_skill_name(name) {
        return None;
    }
    // Args, when present, were appended as a single trailing `\n\nInput: <args>`
    // line; the body may itself contain that marker, so take the LAST one and
    // require a single line (a multi-line tail means there were no args).
    let args = content
        .rsplit_once("\n\nInput: ")
        .map(|(_, rest)| rest.trim())
        .filter(|rest| !rest.is_empty() && !rest.contains('\n'));
    Some(match args {
        Some(args) => format!("/{name} {args}"),
        None => format!("/{name}"),
    })
}

impl CodeTuiApp {
    pub(super) async fn submit_draft(&mut self) -> Result<bool> {
        let action = match self.prepare_submit_action() {
            Ok(action) => action,
            Err(err) => {
                self.notice = Some((ERROR(), err.to_string()));
                return Ok(false);
            }
        };
        let Some(action) = action else {
            return Ok(false);
        };

        // A running `!cmd` owns the transcript tail and the spinner; serialize
        // submissions behind it (esc stops it) rather than overlapping a second.
        if self.local_command.is_some() {
            self.notice = Some((
                MUTED(),
                "A command is running — press esc to stop it".to_string(),
            ));
            return Ok(false);
        }

        match action {
            SubmitAction::Send(input) => {
                if self.sending {
                    // A turn is in flight — queue this message instead of sending
                    // now; it goes out when the current turn finishes.
                    self.queue_message(input);
                } else if let Err(err) = self.send_user_message(input).await {
                    self.notice = Some((ERROR(), err.to_string()));
                }
                Ok(false)
            }
            SubmitAction::Command(command) => {
                // Record the typed `/command` so up-arrow recalls it, the same as
                // a normal message or `!cmd`. Skills and `/create-skill` record
                // their own normalized `/name args` form inside their handlers
                // (see `run_skill_command`), so skip them here to avoid a
                // duplicate entry. Capture the draft before it's cleared below.
                let recordable = !matches!(
                    command,
                    SlashCommand::Skill { .. } | SlashCommand::CreateSkill(_)
                );
                let typed = self.draft.trim().to_string();
                match self.execute_slash_command(command).await {
                    Ok(should_exit) => {
                        if recordable {
                            self.record_draft_history(&typed);
                        }
                        self.draft.clear();
                        self.cursor = 0;
                        self.command_menu.reset();
                        self.draft_history_index = None;
                        self.draft_history_stash = None;
                        Ok(should_exit)
                    }
                    Err(err) => {
                        self.notice = Some((ERROR(), err.to_string()));
                        Ok(false)
                    }
                }
            }
            SubmitAction::Shell(command) => {
                // Don't run a local command on top of a model turn; interrupt it
                // (esc) first. Keeps the draft so the command can be retried.
                if self.sending {
                    self.notice = Some((
                        MUTED(),
                        "Interrupt the current turn (esc) before running a command".to_string(),
                    ));
                    return Ok(false);
                }
                self.record_draft_history(&format!("!{command}"));
                self.start_local_command(command);
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
                self.draft_history_index = None;
                self.draft_history_stash = None;
                Ok(false)
            }
        }
    }

    pub(super) fn prepare_submit_action(&self) -> Result<Option<SubmitAction>> {
        let trimmed = self.draft.trim();
        if trimmed.is_empty() {
            return if self.draft_attachments.is_empty() {
                Ok(None)
            } else {
                Ok(Some(SubmitAction::Send(String::new())))
            };
        }
        if self.draft.contains('\n') {
            return Ok(Some(SubmitAction::Send(trimmed.to_string())));
        }
        if let Some(escaped) = trimmed.strip_prefix("//") {
            return Ok(Some(SubmitAction::Send(format!("/{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('/') {
            return Ok(Some(SubmitAction::Command(
                self.resolve_slash_command(command)?,
            )));
        }
        if let Some(escaped) = trimmed.strip_prefix("!!") {
            return Ok(Some(SubmitAction::Send(format!("!{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('!') {
            let command = command.trim();
            if command.is_empty() {
                anyhow::bail!("Type a command after '!'");
            }
            // `!cmd` forwards no keystrokes (see `system.rs`), so most interactive
            // programs would hang until esc/the 120s cap; refuse those before spawning
            // (bail keeps the draft). `tail -f`/`watch` still stream, so they're allowed.
            if let Some(blocker) = crate::agent::tools::interactive_block_reason(command)
                && blocker.blocks_bang_cmd()
            {
                anyhow::bail!("{}", blocker.user_message());
            }
            return Ok(Some(SubmitAction::Shell(command.to_string())));
        }
        Ok(Some(SubmitAction::Send(trimmed.to_string())))
    }

    /// Parse `input` (the text after the leading `/`) into a [`SlashCommand`]. A
    /// name that isn't a built-in but matches a discovered skill resolves to
    /// [`SlashCommand::Skill`] (the `/repo-study` case); everything else falls back
    /// to [`parse_slash_command`], so a built-in always wins over a same-named skill
    /// and a true typo still errors with "Unknown command".
    pub(super) fn resolve_slash_command(&self, input: &str) -> Result<SlashCommand> {
        let name = input.split_whitespace().next().unwrap_or("");
        let is_builtin = SLASH_COMMANDS.iter().any(|c| c.name == name);
        if !is_builtin && self.skill_commands.iter().any(|s| s.name == name) {
            let argument = input
                .split_once(char::is_whitespace)
                .map(|(_, rest)| rest.trim())
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            return Ok(SlashCommand::Skill {
                name: name.to_string(),
                argument,
            });
        }
        parse_slash_command(input)
    }

    pub(super) async fn send_user_message(&mut self, input: String) -> Result<()> {
        let record = input.clone();
        self.dispatch_user_message(input, Some(record)).await
    }

    /// Send a skill invocation: the model receives `content` (the expanded skill
    /// body), but the draft history records `typed` (`/name args`) so up-arrow
    /// recalls the re-runnable command, not the expanded body.
    pub(super) async fn send_skill_message(
        &mut self,
        content: String,
        typed: String,
    ) -> Result<()> {
        self.dispatch_user_message(content, Some(typed)).await
    }

    pub(super) async fn dispatch_user_message(
        &mut self,
        input: String,
        record: Option<String>,
    ) -> Result<()> {
        self.dispatch_user_message_shown(input, record, None).await
    }

    /// Like [`dispatch_user_message`], but the transcript shows `display`
    /// (e.g. `/review main`) while the model receives the full `input`.
    pub(super) async fn dispatch_user_message_shown(
        &mut self,
        input: String,
        record: Option<String>,
        display: Option<String>,
    ) -> Result<()> {
        // A known text-only model would 400 on image bytes. The fallback describes
        // them instead; anything it can't take falls closed to the refusal, keeping
        // draft + attachment so the user can switch and resend.
        let mut vision_shim: Option<DescriberSource> = None;
        if self.model_image_input == Some(false) {
            let has_image_draft = self.draft_attachments.iter().any(|a| a.is_image());
            let has_other_draft = self.draft_attachments.iter().any(|a| !a.is_image());
            if has_image_draft && has_other_draft {
                // The shim only covers the agent route, which needs all images.
                self.notice = Some((ERROR(), self.image_refusal_base()));
                return Ok(());
            }
            if has_image_draft || self.history_has_image() {
                match self.resolve_describer().await {
                    Ok(src) => vision_shim = Some(src),
                    Err(refusal) if has_image_draft => {
                        self.notice = Some((ERROR(), refusal));
                        return Ok(());
                    }
                    Err(_) => {}
                }
            }
        }
        // An outgoing turn implicitly defers the `/new` continue prompt — the
        // plan stays for `/plan resume`. (The `y` path takes the card first.)
        self.cards.plan_continue = None;
        let attachments = materialize_attachments(&self.draft_attachments).await?;
        if self.key.is_cursor_acp()
            && let Some(session) = self.cursor_acp_session.as_ref()
        {
            // Existing session: capabilities are already known, fail fast
            // without paying a session-open round trip. Cold-open path runs
            // the same check post-open inside `spawn_cursor_turn`.
            cursor_acp::ensure_image_attachments_supported(
                session.prompt_capabilities(),
                &attachments,
            )?;
        }
        if let Some(record) = record {
            self.record_draft_history(&record);
        }
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.overlay = Overlay::None;
        self.notice = None;
        self.last_usage = None;
        self.live_usage = None;
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        // Reset per-turn status state (label can't flash before the first tool).
        self.last_tool_action = None;
        self.subagent_rows.clear();
        self.turn_output_tokens = 0;
        self.turn_stream_chars = 0;
        self.turn_stream_chars_measured = 0;
        self.subagent_token_base = 0;
        self.retrying = false;
        // Fresh stall clock — a stale stamp would flag a "stall" at turn start.
        self.last_stream_activity = Some(Instant::now());
        self.wait_tick = None;
        self.pending_submit = Some(PendingSubmission {
            content: input.clone(),
            attachments: attachments.clone(),
        });
        self.request_started_at = Some(Instant::now());
        // A new message starts (possibly) new work — drop a stale plan card so it
        // doesn't linger above the composer into an unrelated task.
        self.clear_stale_plan();
        self.history.push(ChatMessage {
            model: None,
            role: "user".to_string(),
            content: display.unwrap_or_else(|| input.clone()),
            reasoning_content: None,
            attachments: attachments.clone(),
        });
        // A new turn rebuilds the transcript rows and snaps to the bottom, so any
        // prior selection would point at the wrong content — drop it.
        self.clear_transcript_selection();
        // Fresh turn: a stale goal-stop arm must not interrupt it.
        self.goal_stop_confirm_pending = false;
        self.sending = true;
        self.turn_model = (!self.raw_model.is_empty()).then(|| self.raw_model.clone());
        self.follow_output = true;

        let conversation_has_image = self.history_has_image();
        let all_images = !attachments.is_empty() && attachments.iter().all(|a| a.is_image());
        let shim_active = vision_shim.is_some();
        // Route images to the agent on a model we KNOW reads them, or with the
        // vision shim covering (images become described text); unknown models
        // keep the plain-chat route (which has 400-recovery).
        let agent_vision_ok = all_images && (self.model_image_input == Some(true) || shim_active);
        // Images that accrued on plain chat must keep re-sending there — unless
        // the shim substitutes them, which keeps the agent path (tools stay on).
        let stay_plain_for_vision =
            conversation_has_image && self.model_image_input != Some(true) && !shim_active;
        let route_agent = self.agent_capable()
            && ((attachments.is_empty() && !stay_plain_for_vision) || agent_vision_ok);
        if self.key.is_cursor_acp() {
            self.spawn_cursor_turn(input, attachments);
        } else if route_agent {
            self.spawn_agent_turn(input, attachments, vision_shim).await;
        } else {
            if self.agent_capable() && !attachments.is_empty() && !all_images {
                self.notice = Some((
                    MUTED(),
                    "Attachment sent as plain chat — agent tools are off for this message"
                        .to_string(),
                ));
            }
            // Plain chat's finish path never auto-continues — a goal falling back
            // here would strand the loop, so disarm it and say why.
            if self.goal_mode.take().is_some() {
                self.notice = Some((
                    ERROR(),
                    "Goal mode stopped — this message went out as plain chat (no agent tools)"
                        .to_string(),
                ));
            }
            self.spawn_http_turn();
        }
        Ok(())
    }

    /// Stash a message typed mid-turn: an engine run steers it, anything else
    /// queues for turn end.
    fn queue_message(&mut self, input: String) {
        if input.trim().is_empty() {
            return;
        }
        self.record_draft_history(&input);
        // Serve up + not `/compact` (which runs no batches) = a steerable run.
        if self.agent_serve.is_some() && self.compact_before.is_none() {
            self.steering_queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(input);
            self.notice = Some((
                MUTED(),
                "Queued — the agent picks it up after the current tool step".to_string(),
            ));
        } else {
            self.queued_messages.push(input);
            self.notice = Some((MUTED(), self.queued_notice()));
        }
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    /// Notice text for the queue, reflecting how many are waiting.
    fn queued_notice(&self) -> String {
        match self.queued_messages.len() {
            0 | 1 => "Queued — sends when the current turn finishes".to_string(),
            n => format!("Queued ({n} waiting) — sent one per turn, in order"),
        }
    }

    pub(super) fn clear_steering_queue(&mut self) {
        self.steering_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// Move unconsumed interjections to the front of the follow-up queue.
    pub(super) fn reclaim_unsent_steering(&mut self) {
        let drained: Vec<String> = self
            .steering_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect();
        for (i, message) in drained.into_iter().enumerate() {
            self.queued_messages.insert(i, message);
        }
    }

    /// After a turn ends, send the oldest message queued mid-turn (if any). One
    /// per turn-end, so each queued message becomes its own user turn in order.
    /// Records nothing — queue sites already recorded the recallable form, and a
    /// queued skill's expanded body must not land in ↑/↓ recall.
    pub(super) async fn drain_queued_message(&mut self) -> Result<()> {
        if !self.sending && !self.queued_messages.is_empty() {
            let queued = self.queued_messages.remove(0);
            self.dispatch_user_message(queued, None).await?;
        }
        Ok(())
    }

    /// Stash a slash command typed mid-turn that needs the engine idle; it runs
    /// when the turn finishes (see `drain_queued_commands`).
    fn queue_command(&mut self, command: SlashCommand, label: &str) {
        self.queued_commands.push(command);
        self.notice = Some((
            MUTED(),
            format!("{label} queued — runs when the current turn finishes"),
        ));
    }

    /// After a turn ends, run the commands queued mid-turn, in order. One that
    /// starts a new turn (`/goal`, `/plan`, `/compact`) flips `sending`, ending
    /// the loop; the rest re-queue themselves behind it via their mid-turn gates.
    pub(super) async fn drain_queued_commands(&mut self) {
        while !self.sending && !self.queued_commands.is_empty() {
            let command = self.queued_commands.remove(0);
            if let Err(err) = self.execute_slash_command(command).await {
                self.notice = Some((ERROR(), err.to_string()));
            }
        }
    }

    /// True when any history message carries an image attachment.
    pub(super) fn history_has_image(&self) -> bool {
        self.history
            .iter()
            .any(|m| m.attachments.iter().any(|a| a.is_image()))
    }

    /// True when the current key can drive the in-process agent (see `key_is_agent_capable`).
    pub(super) fn agent_capable(&self) -> bool {
        crate::commands::code_agent_oneshot::key_is_agent_capable(&self.key)
    }

    pub(super) fn describer_status(&self) -> DescriberStatus {
        use crate::services::session_store::VisionFallbackMode;
        if !self.agent_capable() {
            return DescriberStatus::NotAgentCapable;
        }
        match &self.vision_fallback {
            VisionFallbackMode::Off => DescriberStatus::Off,
            VisionFallbackMode::Gateway
                if crate::services::vision_describe::describe_exhausted() =>
            {
                DescriberStatus::Exhausted
            }
            VisionFallbackMode::Gateway => DescriberStatus::Gateway,
            VisionFallbackMode::Custom => match &self.vision_fallback_custom {
                Some((key_id, model)) if *model != self.model => DescriberStatus::OwnKey {
                    key_id: key_id.clone(),
                    model: model.clone(),
                },
                _ => DescriberStatus::Unconfigured,
            },
        }
    }

    /// The custom key is resolved + decrypted here so the spawned turn just
    /// calls it.
    pub(super) async fn resolve_describer(&self) -> std::result::Result<DescriberSource, String> {
        let status = self.describer_status();
        match &status {
            DescriberStatus::Gateway => return Ok(DescriberSource::Gateway),
            DescriberStatus::OwnKey { key_id, model } => {
                if let Ok(Some(mut key)) = self.session_store.get_key_by_id(key_id).await
                    && crate::services::session_store::SessionStore::decrypt_key_secret(&mut key)
                        .is_ok()
                {
                    return Ok(DescriberSource::OwnKey {
                        model: model.clone(),
                        key: Box::new(key),
                    });
                }
            }
            _ => {}
        }
        Err(self.vision_refusal_message(&status))
    }

    fn image_refusal_base(&self) -> String {
        format!(
            "{} can't read images — switch to a vision model (e.g. /model) and resend.",
            self.model
        )
    }

    /// The image refusal, hinted by WHY the fallback didn't cover. `OwnKey` lands
    /// here only when the stored key vanished or wouldn't decrypt.
    fn vision_refusal_message(&self, status: &DescriberStatus) -> String {
        let base = self.image_refusal_base();
        match status {
            DescriberStatus::Off => format!("{base} Or turn on Vision fallback in /config."),
            DescriberStatus::Exhausted => format!("image-describe quota used up — {base}"),
            DescriberStatus::Unconfigured | DescriberStatus::OwnKey { .. } => {
                format!("{base} Or re-pick the describer: /config → Vision fallback → custom.")
            }
            DescriberStatus::Gateway | DescriberStatus::NotAgentCapable => base,
        }
    }

    /// Refresh the cached git branch for `display_cwd`, throttled so the footer's
    /// `.git/HEAD` read happens at most every couple of seconds rather than on
    /// every (≈60fps) frame. A checkout — by the user in another terminal or by
    /// the agent via run_bash — is reflected on the next refresh. Cheap file read,
    /// no subprocess; `None` when the dir isn't a git work tree.
    pub(super) fn refresh_git_branch(&mut self) {
        const REFRESH_AFTER: std::time::Duration = std::time::Duration::from_secs(2);
        let due = self
            .git_branch_checked_at
            .is_none_or(|at| at.elapsed() >= REFRESH_AFTER);
        if !due {
            return;
        }
        let cwd = self.display_cwd().to_string();
        self.git_branch = git_branch_for(&cwd);
        self.git_branch_checked_at = Some(std::time::Instant::now());
    }

    /// Directory shown in the header/footer: the real launch dir for any chat
    /// that actually runs there (the in-process agent and the cursor ACP backend,
    /// where files are edited — a safety signal), else chat's sandbox (OAuth relay).
    pub(super) fn display_cwd(&self) -> &str {
        if (self.agent_capable() || self.key.is_cursor_acp()) && !self.real_cwd.is_empty() {
            &self.real_cwd
        } else {
            &self.cwd
        }
    }

    /// Directory key for persistence / logs / resume: the real launch dir, which
    /// is stable across runs. `self.cwd` is the per-pid ephemeral sandbox, so
    /// keying on it hides every session from `/resume` and `aivo logs` on the
    /// next launch. Falls back to the sandbox only when the real dir is unknown.
    pub(super) fn persist_cwd(&self) -> &str {
        if self.real_cwd.is_empty() {
            &self.cwd
        } else {
            &self.real_cwd
        }
    }

    fn spawn_http_turn(&mut self) {
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = self.key.clone();
        let model = self.model.clone();
        // Drop agent tool / local-command steps — they're display-only and aren't
        // valid provider turns on the plain-chat path.
        let history: Vec<ChatMessage> = self
            .history
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .cloned()
            .collect();
        let copilot_tm = self.copilot_tm.clone();
        let mut format = self.format.clone();

        self.response_task = Some(tokio::spawn(async move {
            let spinning = Arc::new(AtomicBool::new(false));
            let result = send_message_turn(
                &client,
                &key,
                copilot_tm.as_deref(),
                &model,
                &history,
                &mut format,
                &spinning,
                false, // TUI always streams for live rendering
                &mut |chunk| {
                    tx.send(RuntimeEvent::Delta(chunk)).ok();
                    Ok(())
                },
            )
            .await;
            let result = result.map_err(|err| err.to_string());

            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
    }

    /// Get (or lazily build) the per-key route cache the agent serves share.
    /// Seeded from the key's `chat` routes + the provider default, so a known
    /// model starts confirmed (no re-probe). Rebuilt when the key changes.
    fn agent_route_cache(&mut self) -> std::sync::Arc<crate::services::route_cache::RouteCache> {
        if let Some((key_id, cache)) = &self.agent_route_cache
            && *key_id == self.key.id
        {
            return cache.clone();
        }
        let protocol =
            crate::services::provider_profile::provider_profile_for_key(&self.key).default_protocol;
        let cache = std::sync::Arc::new(crate::services::route_cache::RouteCache::new(
            "code",
            protocol,
            self.key.routes_for_tool("code"),
        ));
        self.agent_route_cache = Some((self.key.id.clone(), cache.clone()));
        cache
    }

    /// Run one agent turn: (re)build the in-process engine, start a per-turn
    /// loopback serve, then drive `engine.run_turn` on a background task that
    /// streams text/tool-steps and permission requests back as `RuntimeEvent`s.
    async fn spawn_agent_turn(
        &mut self,
        input: String,
        attachments: Vec<MessageAttachment>,
        vision_shim: Option<DescriberSource>,
    ) {
        use crate::agent::engine::{AgentEngine, TurnCtx};

        // Self-verify at declared-done: opt-in normally, default-on under goal mode.
        let self_correct =
            env_self_correct() || (self.goal_mode.is_some() && goal_verify_enabled());
        // Sync a reused engine to the session mode + self-correct toggle before the turn.
        let unsend_pending = std::mem::take(&mut self.agent_unsend_pending);
        if let Some(session) = self.agent_engine.as_ref() {
            let mut engine = session.engine.lock().await;
            // An Esc-unsent turn whose async un-send hasn't run yet: drop the stale
            // user tail now, before this turn would merge into it.
            if unsend_pending {
                engine.unsend_last_user_turn();
            }
            engine.set_plan_mode(self.plan_mode);
            engine.set_self_correct(self_correct);
        }
        self.plan_exit_pending = false;
        // Clear any stale live exit request; the mode sync above is authoritative.
        self.plan_exit_flag
            .store(false, std::sync::atomic::Ordering::Relaxed);
        // The agent works in the real launch directory — NOT chat's sandbox
        // (`self.cwd`). It reads/edits the user's actual project.
        let real_cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };

        // Rebuild the engine when absent or when the key/model changed; otherwise
        // reuse it so multi-turn context carries over.
        let need_new = self
            .agent_engine
            .as_ref()
            .is_none_or(|s| s.key_id != self.key.id || s.model != self.model);
        if need_new {
            // Snapshot the outgoing engine's transcript before replacing it, so a
            // model switch rebuilds from the exact prior messages (ids intact), not
            // the lossy display seed. Empty after /new or key switch (they clear
            // agent_engine); skipped when a resume payload is pending (it wins).
            // Rewind state rides along: verbatim restore keeps checkpoint indices valid.
            let mut prior_rewind_state = None;
            // A same-key rebuild stays on the same provider: carry the prefix-cache
            // latch so the preventive snip doesn't silently disarm on a model switch.
            let mut prior_prefix_cache = false;
            let prior_engine_messages: Option<Vec<serde_json::Value>> =
                if self.pending_agent_messages.is_some() {
                    None
                } else if let Some(prev) = self.agent_engine.as_ref() {
                    let mut prev_engine = prev.engine.lock().await;
                    prior_prefix_cache =
                        prev.key_id == self.key.id && prev_engine.prefix_cache_seen;
                    let msgs = prev_engine.export_conversation();
                    if msgs.is_empty() {
                        None
                    } else {
                        prior_rewind_state = Some(prev_engine.take_rewind_state());
                        Some(msgs)
                    }
                } else {
                    None
                };
            let date = chrono::Local::now().format("%Y-%m-%d").to_string();
            let guides = crate::agent::system_prompt::discover_project_guides(
                std::path::Path::new(&real_cwd),
            );
            // Discovered skills minus `/skills`-disabled, plus the create-agent
            // builtin (natural-language subagent authoring via the `skill` tool).
            let disabled: std::collections::HashSet<String> = self
                .session_store
                .get_disabled_skills()
                .await
                .unwrap_or_default()
                .into_iter()
                .collect();
            let skills =
                crate::agent::skills::engine_skills(std::path::Path::new(&real_cwd), &disabled);
            // A `--max-context` override wins; otherwise resolve from catalog/snapshot.
            let context_window = match self.context_window_override {
                Some(w) => w,
                None => crate::services::model_metadata::resolve_limits(
                    &self.cache,
                    Some(&self.key.base_url),
                    &self.model,
                )
                .await
                .context
                .unwrap_or(0),
            }
            .min(u32::MAX as u64) as u32;
            let mut engine = AgentEngine::new(
                &real_cwd,
                &self.model,
                &date,
                &guides,
                &skills,
                context_window,
                0,
            );
            engine.prefix_cache_seen = prior_prefix_cache;
            // The bundled aivo-starter provider is first-party: brand the agent so it
            // presents as aivo's assistant instead of disclosing the upstream model.
            // BYOK keys stay honest (no branding).
            if crate::services::provider_profile::is_aivo_starter_base(&self.key.base_url) {
                engine.set_first_party();
            }
            if let Some(ctx) = self.injected_context.as_deref() {
                engine.append_system_context(ctx);
            }
            // Enable `/rewind` tree-checkpointing (top-level chat only — sub-engines
            // never call this, so they don't pay the git cost).
            engine.enable_rewind_checkpoints(&real_cwd);
            // Interactive: the agent confirms before a big build. (Plan mode is
            // applied LAST, below, so it strips from the fully assembled tool list.)
            engine.set_confirm_before_build();
            engine.set_self_correct(self_correct);
            // Interactive chat only — headless (`-e`) and sub-agents build engines elsewhere.
            engine.set_chat_session_context(crate::agent::engine::ChatSessionContext {
                model_label: self.raw_model.clone(),
                provider_label: self.key.display_name().to_string(),
                effort: self.effective_reasoning_effort(),
                effort_levels: self.model_reasoning_efforts.clone(),
            });
            // Share the live thinking toggle so the engine requests reasoning (on
            // reasoning-capable models) only while thinking is on.
            // Named specialist sub-agents (project `.aivo/agents`/`.claude/agents`,
            // then `~/.config/aivo/agents`); the model delegates via `agent`.
            let subagents = crate::agent::subagents::discover_subagents(
                std::path::Path::new(&real_cwd),
                self.session_store.config_dir(),
            );
            engine.set_subagents(&subagents);
            // Delegations re-resolve profiles from disk, so one authored or edited
            // mid-turn runs correctly even before the advert refreshes.
            engine.set_agents_dir(self.session_store.config_dir());
            // Persistent grant store so "always allow"s survive across sessions.
            engine.set_grants_path(self.session_store.config_dir());
            // Durable sub-agent reports under this session's artifacts dir (survive compaction).
            engine.set_artifacts_dir(self.session_store.session_artifacts_dir(&self.session_id));
            engine.set_jobs(self.jobs.clone());
            // LSP diagnostics-after-edit (default on; AIVO_AGENT_LSP=0 opts out).
            engine.maybe_enable_lsp(std::path::Path::new(&real_cwd));
            // User lifecycle hooks (~/.config/aivo/hooks.json).
            engine.set_hooks(std::sync::Arc::new(
                crate::agent::hooks::HookSet::load_default(),
            ));
            // Carry prior conversation in, best fidelity first: a resumed session's
            // durable transcript, else the outgoing engine's messages on a model
            // switch (both verbatim), else the lossy text seed of display history.
            // The just-pushed user turn is excluded — run_turn re-adds it.
            if let Some(conversation) = self.pending_agent_messages.take() {
                engine.restore_conversation(conversation);
            } else if let Some(conversation) = prior_engine_messages {
                engine.restore_conversation(conversation);
                if let Some((store, checkpoints)) = prior_rewind_state.take() {
                    engine.adopt_rewind_state(store, checkpoints);
                }
            } else {
                let prior = self.history.len().saturating_sub(1);
                let seed = agent_seed_turns(&self.history[..prior], &self.vision_descriptions);
                engine.seed_history(seed);
            }
            // Offer any configured MCP servers' tools. Connect in the BACKGROUND
            // (not inline) so a slow server — an `npx` download on first use can
            // take seconds — can't freeze the UI on the first turn. Tools arrive via
            // `McpConnected`, which rebuilds the engine (re-seeding from history) to
            // advertise them. Cached after the first connect, so this turn just uses
            // whatever's ready; the common no-mcp.json case is empty + instant.
            let connected = match &self.mcp_client {
                Some(client) if client.has_tools() => Some(client.clone()),
                _ => None,
            };
            if let Some(client) = connected {
                // Advertise minus the Ctrl+T-disabled tools. Prefs are the source
                // of truth (not the UI cache), so a toggle always lands on the
                // next engine build even if `/mcp` was never opened this session.
                let disabled: std::collections::HashSet<String> = self
                    .session_store
                    .get_disabled_mcp_tools()
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                if disabled.is_empty() {
                    engine.set_external_tools(client);
                } else {
                    engine.set_external_tools(std::sync::Arc::new(
                        crate::agent::mcp::FilteredTools::new(client, disabled),
                    ));
                }
            } else if self.mcp_client.is_none() {
                // Connect the configured servers the user hasn't disabled. A repo's
                // project `.mcp.json` STDIO servers run local code, so they're held
                // back behind a one-time consent card (user + HTTP servers connect
                // freely); see `connect_mcp_with_consent`.
                let disabled = self.effective_disabled_mcp_servers().await;
                self.connect_mcp_with_consent(real_cwd.clone(), disabled)
                    .await;
            }
            // Re-enter plan mode on a rebuilt engine, last — so it strips from the
            // fully assembled tool list (`set_subagents`/MCP add tools above).
            if self.plan_mode {
                engine.set_plan_mode(true);
            }
            self.agent_engine = Some(AgentSession {
                key_id: self.key.id.clone(),
                model: self.model.clone(),
                engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
            });
        }
        let engine = self.agent_engine.as_ref().unwrap().engine.clone();

        // Rides this turn's loopback: same auth, same usage accounting.
        let describer_upstream = match &vision_shim {
            Some(DescriberSource::OwnKey { model, key }) => {
                Some((model.clone(), key.as_ref().clone()))
            }
            _ => None,
        };
        let (base, auth) = match self.start_agent_serve(describer_upstream).await {
            Ok(t) => t,
            Err(e) => {
                self.notice = Some((ERROR(), format!("agent serve failed to start: {e}")));
                self.sending = false;
                self.request_started_at = None;
                self.pending_submit = None;
                return;
            }
        };

        let tx = self.tx.clone();
        let cwd = real_cwd;
        // Clone the shared LIVE flags (not snapshots) so a mid-turn Shift+Tab
        // toggle takes effect on this running turn's permission gate.
        let auto_approve = self.auto_approve_flag.clone();
        let review_edits = self.review_edits_flag.clone();
        let plan_exit = self.plan_exit_flag.clone();
        let steering = self.steering_queue.clone();
        // The context window may have resolved AFTER the engine was built (a model
        // only known via the background catalog warm). Carry the latest value in
        // so the engine can fill a still-missing window and start compacting,
        // instead of letting a long conversation overflow uncompacted.
        let context_window = self.context_window.min(u32::MAX as u64) as u32;
        // Effective reasoning effort, carried in per turn like the window so a
        // mid-session change applies next turn without locking the engine. `None`
        // leaves the engine's own model default in place.
        let reasoning_effort = self.effective_reasoning_effort();
        // Thinking on/off, carried in per turn like the effort; off makes the engine
        // emit the provider-correct disable signal.
        let thinking_enabled = self.thinking_enabled;
        // `/config` "use aivo web search" toggle, applied like thinking each turn.
        let web_search_enabled = self.web_search_enabled;
        let agent_tools_enabled = self.agent_tools_enabled;
        // The model's catalog effort levels, so the engine's disable path only sends
        // a level the provider accepts (e.g. `aivo/starter` has no `none`).
        let reasoning_efforts = self.model_reasoning_efforts.clone();
        // Build the multimodal message up front so an encoding error surfaces here, not
        // inside the spawned turn. Empty unless the agent-vision path was chosen.
        let multimodal: Option<serde_json::Value> = if attachments.is_empty() {
            None
        } else {
            let msg = ChatMessage {
                model: None,
                role: "user".to_string(),
                content: input.clone(),
                reasoning_content: None,
                attachments,
            };
            match crate::commands::code_request_builder::build_openai_message(&msg) {
                Ok(v) => v.get("content").cloned(),
                Err(e) => {
                    self.notice = Some((ERROR(), format!("couldn't attach image: {e}")));
                    self.sending = false;
                    self.request_started_at = None;
                    self.pending_submit = None;
                    return;
                }
            }
        };
        // Flag the user row as engine-dispatched for the `/rewind` picker match —
        // after every early return, so a turn that never ran stays unflagged.
        // Checkpoint by the row's DISPLAYED content — the picker matches by
        // transcript text, which can diverge from the expanded prompt (`/review`).
        let checkpoint_prompt = match self.history.last() {
            Some(m) if m.role == "user" => {
                self.agent_turn_indices.insert(self.history.len() - 1);
                m.content.clone()
            }
            _ => input.clone(),
        };
        let vision_descriptions = match &vision_shim {
            Some(_) => self.vision_descriptions.clone(),
            None => std::collections::HashMap::new(),
        };
        self.response_task = Some(tokio::spawn(async move {
            let client = crate::services::http_utils::router_http_client();
            let ctx = TurnCtx {
                client: &client,
                serve_base: &base,
                auth: Some(&auth),
                cwd: std::path::Path::new(&cwd),
                yes: false,
                auto_approve_all: false, // the live toggle carries the mode
                auto_approve: Some(&auto_approve),
                review_edits: Some(&review_edits),
                plan_exit: Some(&plan_exit),
            };
            let mut ui = ChatAgentUi {
                tx,
                cwd: std::path::PathBuf::from(&cwd),
                steering,
            };
            let mut engine = engine.lock().await;
            engine.set_context_window(context_window);
            engine.set_thinking_enabled(thinking_enabled);
            engine.set_web_search_enabled(web_search_enabled);
            engine.set_agent_tools_enabled(agent_tools_enabled);
            engine.set_reasoning_efforts(reasoning_efforts);
            if let Some(effort) = reasoning_effort {
                engine.set_reasoning_effort(effort);
            }
            // Before the first request: a failure returns here, where the engine
            // hasn't consumed the turn, so the composer restores cleanly.
            engine.set_image_substitution(vision_shim.is_some());
            if let Some(src) = &vision_shim {
                engine.merge_image_descriptions(&vision_descriptions);
                let todo = engine.undescribed_images(multimodal.as_ref());
                // Independent calls, so pay one round trip rather than N.
                let (client, base, auth) = (&client, &base, &auth);
                let described =
                    futures::future::join_all(todo.iter().map(|(hash, url)| async move {
                        (
                            hash,
                            vision_describe::describe(src, client, base, auth, url).await,
                        )
                    }))
                    .await;
                let mut failure = None;
                for (hash, result) in described {
                    match result {
                        // Cache even when a sibling failed, so the retry only
                        // re-describes what actually broke.
                        Ok(desc) => {
                            let text = vision_describe::format_described_image(&desc);
                            engine.insert_image_description(hash.clone(), text.clone());
                            let _ = ui.tx.send(RuntimeEvent::ImageDescribed {
                                hash: hash.clone(),
                                text,
                            });
                        }
                        Err(message) => failure = failure.or(Some(message)),
                    }
                }
                if let Some(message) = failure {
                    let _ = ui.tx.send(RuntimeEvent::DescribeFailed { message });
                    return;
                }
            }
            // run_turn ends by calling ui.footer → AgentFinished commits the turn.
            let content = multimodal.unwrap_or(serde_json::Value::String(input));
            engine
                .run_turn_with_content(&ctx, &mut ui, content, checkpoint_prompt)
                .await
        }));
    }

    /// After aborting an agent turn, close its checkpoint's open segment in the
    /// background: the abort skips the turn-end record, and diffing only at
    /// `/rewind` would sweep the user's post-interrupt hand-edits into the revert
    /// set. Idempotent, so racing a completed turn is benign.
    fn finalize_interrupted_checkpoint(&self) {
        let Some(session) = &self.agent_engine else {
            return;
        };
        let engine = session.engine.clone();
        tokio::spawn(async move {
            engine.lock().await.record_turn_changes().await;
        });
    }

    /// Tear down the per-turn agent serve (shutdown notify + abort accept loop).
    pub(super) fn stop_agent_serve(&mut self) {
        if let Some((handle, shutdown)) = self.agent_serve.take() {
            shutdown.notify_one();
            handle.abort();
        }
    }

    /// Loopback serve router + auth token (sole egress, usage under "code").
    /// `model_upstream` routes one extra model name (the vision-fallback
    /// describer) to another key through the same loopback.
    async fn build_agent_serve_router(
        key: &ApiKey,
        session_store: &crate::services::session_store::SessionStore,
        model_upstream: Option<(String, ApiKey)>,
    ) -> (crate::services::serve_router::ServeRouter, String) {
        use crate::services::serve_router::{
            ServeRouter, ServeRouterConfig, random_auth_token, resolve_grok_fallback,
        };
        let auth = random_auth_token();
        let grok_fallback = if key.is_grok_oauth() {
            resolve_grok_fallback(session_store).await
        } else {
            None
        };
        let config = ServeRouterConfig::from_key(
            key,
            false,
            300,
            Some(auth.clone()),
            std::collections::HashMap::new(),
        )
        .with_grok_fallback(grok_fallback);
        let mut router = ServeRouter::new(config, key.clone(), session_store.logs())
            .with_oauth_persist(session_store.clone())
            .with_usage_accounting(session_store.clone(), "code".to_string())
            .quiet(true);
        if let Some((model, upstream_key)) = model_upstream {
            router = router.with_model_upstreams(vec![(model, upstream_key)]);
        }
        (router, auth)
    }

    /// Start this turn's loopback serve; sets `self.agent_serve`, returns `(base, auth)`.
    async fn start_agent_serve(
        &mut self,
        model_upstream: Option<(String, ApiKey)>,
    ) -> Result<(String, String)> {
        let (router, auth) =
            Self::build_agent_serve_router(&self.key, &self.session_store, model_upstream).await;
        let router = router.with_route_cache(self.agent_route_cache());
        let (handle, shutdown, port) = router.start_background_with_addr("127.0.0.1", 0).await?;
        self.agent_serve = Some((handle, shutdown));
        Ok((format!("http://127.0.0.1:{port}"), auth))
    }

    /// `/compact` folds older turns via the LLM; `/compact fast` clears stale output.
    /// Both queue mid-turn (the running turn holds the engine lock) and no-op with
    /// no agent conversation.
    pub(super) async fn run_compact_command(&mut self, fast: bool) {
        if self.sending {
            self.queue_command(
                SlashCommand::Compact { fast },
                if fast { "/compact fast" } else { "/compact" },
            );
            return;
        }
        let Some(session) = self.agent_engine.as_ref() else {
            // Post-resume/rebuild the conversation re-seeds on the next turn —
            // nothing to fold NOW; don't claim there's nothing to compact.
            let msg = if self.history.is_empty() && self.pending_agent_messages.is_none() {
                "nothing to compact yet"
            } else {
                "no live conversation to compact — send a message first, then /compact"
            };
            self.notice = Some((MUTED(), msg.to_string()));
            return;
        };
        let engine = session.engine.clone();
        if fast {
            let (before, after) = engine.lock().await.compact_now_local();
            let freed = before.saturating_sub(after) as usize;
            self.context_tokens = after;
            self.context_is_estimate = true;
            self.notice = Some(freed_notice(freed, "cleared stale output"));
            return;
        }
        // LLM summary path: skip a pointless round-trip when nothing is foldable.
        let before = {
            let engine = engine.lock().await;
            if !engine.has_compactable_history() {
                self.notice = Some((
                    MUTED(),
                    "already compact — nothing older to fold".to_string(),
                ));
                return;
            }
            engine.estimated_context_tokens()
        };
        self.spawn_compact_turn(engine, before).await;
    }

    /// Run a manual LLM compaction on a background task; finishes via `ui.footer` →
    /// `finish_agent_turn` (which tears the serve down and reports the freed delta).
    async fn spawn_compact_turn(
        &mut self,
        engine: std::sync::Arc<tokio::sync::Mutex<crate::agent::engine::AgentEngine>>,
        before: u64,
    ) {
        use crate::agent::engine::TurnCtx;

        let (base, auth) = match self.start_agent_serve(None).await {
            Ok(t) => t,
            Err(e) => {
                self.notice = Some((ERROR(), format!("compact serve failed to start: {e}")));
                return;
            }
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let tx = self.tx.clone();
        // Turn state: block the composer, show status; flag the freed-delta report.
        self.sending = true;
        self.turn_model = (!self.raw_model.is_empty()).then(|| self.raw_model.clone());
        self.request_started_at = Some(Instant::now());
        self.turn_output_tokens = 0;
        self.turn_stream_chars = 0;
        self.turn_stream_chars_measured = 0;
        self.subagent_token_base = 0;
        self.compact_before = Some(before);
        self.response_task = Some(tokio::spawn(async move {
            let client = crate::services::http_utils::router_http_client();
            let ctx = TurnCtx {
                client: &client,
                serve_base: &base,
                auth: Some(&auth),
                cwd: std::path::Path::new(&cwd),
                yes: false,
                auto_approve_all: false, // compaction runs no tools
                auto_approve: None,
                review_edits: None,
                plan_exit: None,
            };
            let mut ui = ChatAgentUi {
                tx,
                cwd: std::path::PathBuf::from(&cwd),
                steering: SteeringQueue::default(),
            };
            let started = Instant::now();
            let mut engine = engine.lock().await;
            engine
                .compact_now(&ctx, &mut ui, started.elapsed().as_secs())
                .await;
        }));
    }

    /// Permission-prompt hook: surface cursor's tool requests on aivo's own
    /// permission card (reusing the agent's AgentPermission channel; "always"
    /// flips auto-approve on). Shared by the live-turn open and the prewarm.
    fn cursor_permission_prompt(&self) -> cursor_acp::CursorPermissionPrompt {
        let tx = self.tx.clone();
        std::sync::Arc::new(move |req: cursor_acp::CursorPermissionRequest| {
            let tx = tx.clone();
            Box::pin(async move {
                let (reply, rx) = tokio::sync::oneshot::channel();
                if tx
                    .send(RuntimeEvent::AgentPermission {
                        tool: req.tool,
                        preview: Some(req.preview),
                        once_only: false,
                        reply,
                    })
                    .is_err()
                {
                    return Decision::Deny;
                }
                rx.await.unwrap_or(Decision::Deny)
            })
        })
    }

    /// `cursor/ask_question`: show each question on the `ask_user` card (free
    /// text off) and map the picked label(s) back to cursor option ids.
    fn cursor_ask_question_prompt(&self) -> cursor_acp::CursorAskQuestionPrompt {
        let tx = self.tx.clone();
        std::sync::Arc::new(move |req: cursor_acp::CursorAskRequest| {
            let tx = tx.clone();
            Box::pin(async move {
                let mut answers = Vec::new();
                for q in req.questions {
                    // No options → answerable only as free text, which cursor can't carry.
                    if q.options.is_empty() {
                        continue;
                    }
                    let options: Vec<crate::agent::ask::AskOption> = q
                        .options
                        .iter()
                        .map(|(_, label)| crate::agent::ask::AskOption {
                            label: label.clone(),
                            description: None,
                        })
                        .collect();
                    let (reply, rx) = tokio::sync::oneshot::channel();
                    if tx
                        .send(RuntimeEvent::AgentAskUser {
                            question: q.prompt.clone(),
                            options,
                            allow_free_text: false,
                            multi_select: q.allow_multiple,
                            reply,
                        })
                        .is_err()
                    {
                        return cursor_acp::CursorAskOutcome::Cancelled;
                    }
                    match rx.await {
                        Ok(Ok(answer)) => {
                            let selected_option_ids =
                                select_ids_for_answer(&answer, &q.options, q.allow_multiple);
                            answers.push(cursor_acp::CursorAskAnswer {
                                question_id: q.id.clone(),
                                selected_option_ids,
                            });
                        }
                        _ => return cursor_acp::CursorAskOutcome::Cancelled, // dismissed/dropped
                    }
                }
                if answers.is_empty() {
                    cursor_acp::CursorAskOutcome::Cancelled
                } else {
                    cursor_acp::CursorAskOutcome::Answered(answers)
                }
            })
        })
    }

    /// Interactive hooks for cursor's `cursor/*` methods, built fresh per open
    /// (each carries its own state, e.g. the todo merge buffer).
    fn cursor_interaction_hooks(&self) -> cursor_acp::CursorInteractionHooks {
        cursor_acp::CursorInteractionHooks {
            ask_question: Some(self.cursor_ask_question_prompt()),
            update_todos: Some(self.cursor_todos_sink()),
            create_plan: Some(self.cursor_plan_prompt()),
            task: Some(self.cursor_task_sink()),
        }
    }

    /// `cursor/task`: enrich the matching call entry with the real task
    /// description the generic `Task: Subagent task` title lacks.
    fn cursor_task_sink(&self) -> cursor_acp::CursorTaskSink {
        let tx = self.tx.clone();
        std::sync::Arc::new(move |notice: cursor_acp::CursorTaskNotice| {
            if notice.tool_call_id.is_empty()
                || (notice.description.is_empty() && notice.prompt.is_empty())
            {
                return;
            }
            let mut args = serde_json::Map::new();
            if !notice.description.is_empty() {
                args.insert("label".into(), notice.description.into());
            }
            if !notice.prompt.is_empty() {
                args.insert("task".into(), notice.prompt.into());
            }
            // Attribute specialists (`explore — <task>`); default types are noise.
            if !matches!(
                notice.subagent_type.as_str(),
                "" | "unspecified" | "generalPurpose"
            ) {
                args.insert("agent".into(), notice.subagent_type.into());
            }
            let _ = tx.send(RuntimeEvent::AgentToolUpdate {
                id: notice.tool_call_id,
                args: Some(serde_json::Value::Object(args)),
                result: None,
                failed: false,
            });
        })
    }

    /// `cursor/create_plan`: render the plan as markdown, reuse the
    /// `AgentPlanApproval` card, map the verdict to cursor's outcome.
    fn cursor_plan_prompt(&self) -> cursor_acp::CursorPlanPrompt {
        let tx = self.tx.clone();
        std::sync::Arc::new(move |req: cursor_acp::CursorPlanRequest| {
            let tx = tx.clone();
            Box::pin(async move {
                let markdown = render_cursor_plan_markdown(&req);
                let (reply, rx) = tokio::sync::oneshot::channel();
                if tx
                    .send(RuntimeEvent::AgentPlanApproval {
                        plan: markdown,
                        reply,
                    })
                    .is_err()
                {
                    return cursor_acp::CursorPlanOutcome::Cancelled;
                }
                use crate::agent::protocol::PlanDecision;
                match rx.await {
                    Ok(Ok(PlanDecision::Approve)) => cursor_acp::CursorPlanOutcome::Accepted,
                    // Feedback rides back as the rejection reason so the model can revise.
                    Ok(Ok(PlanDecision::KeepPlanning { feedback })) => {
                        cursor_acp::CursorPlanOutcome::Rejected(
                            feedback.unwrap_or_else(|| "Keep planning".to_string()),
                        )
                    }
                    _ => cursor_acp::CursorPlanOutcome::Cancelled, // discard/dismiss/drop
                }
            })
        })
    }

    /// `cursor/update_todos`: merge pushes into a closure-local buffer (resets on
    /// re-open) and render through the existing `AgentPlan` plan-card pipeline.
    fn cursor_todos_sink(&self) -> cursor_acp::CursorTodosSink {
        let tx = self.tx.clone();
        let buffer =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::<cursor_acp::CursorTodo>::new()));
        std::sync::Arc::new(move |update: cursor_acp::CursorTodosUpdate| {
            let items = {
                let mut list = buffer.lock().unwrap();
                if !update.merge {
                    list.clear();
                }
                for todo in update.todos {
                    match list.iter_mut().find(|t| t.id == todo.id) {
                        Some(existing) => *existing = todo,
                        None => list.push(todo),
                    }
                }
                list.iter()
                    .map(|t| {
                        serde_json::json!({
                            "step": t.content,
                            "status": map_cursor_todo_status(&t.status),
                        })
                    })
                    .collect::<Vec<_>>()
            };
            let _ = tx.send(RuntimeEvent::AgentPlan(serde_json::Value::Array(items)));
        })
    }

    /// The dir cursor-agent runs its tools in: the real launch dir, else the
    /// sandbox (tests).
    fn cursor_workspace_cwd(&self) -> String {
        if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        }
    }

    /// Start opening the cursor ACP session in the background so its connect
    /// overlaps the user typing. The first turn consumes this in-flight open
    /// (see [`Self::spawn_cursor_turn`]) rather than starting its own, so
    /// exactly one session is created — no duplicate cursor-agent, no adoption
    /// race. No-op unless it's a cursor key with no session/prewarm yet.
    pub(super) fn prewarm_cursor_session(&mut self) {
        if !self.key.is_cursor_acp()
            || self.cursor_acp_session.is_some()
            || self.cursor_prewarm.is_some()
        {
            return;
        }
        let key = self.key.clone();
        let requested_model = (!self.raw_model.is_empty()).then(|| self.raw_model.clone());
        let cwd = self.cursor_workspace_cwd();
        let auto_approve = self.auto_approve_flag.clone();
        let permission_prompt = self.cursor_permission_prompt();
        let hooks = self.cursor_interaction_hooks();
        self.cursor_prewarm = Some(tokio::spawn(async move {
            open_cursor_session(
                key,
                requested_model,
                cwd,
                auto_approve,
                permission_prompt,
                hooks,
            )
            .await
            .map_err(|e| e.to_string())
        }));
    }

    fn spawn_cursor_turn(&mut self, input: String, attachments: Vec<MessageAttachment>) {
        // Existing session: clone handles cheaply and skip the open step.
        let existing = self.cursor_acp_session.as_ref().map(|session| {
            (
                session.client_handle(),
                session.session_id().to_string(),
                session.model_id().map(str::to_string),
                session.prompt_capabilities().clone(),
            )
        });
        let key = self.key.clone();
        let requested_model = (!self.raw_model.is_empty()).then(|| self.raw_model.clone());
        // A fresh open (prewarm/`/new`) comes up in `agent`; re-apply plan below.
        let want_plan_mode = self.cursor_plan_mode;
        let cwd = self.cursor_workspace_cwd();
        let tx = self.tx.clone();
        let format = self.format.clone();
        let cursor_auto_approve = self.auto_approve_flag.clone();
        let permission_prompt = self.cursor_permission_prompt();
        let hooks = self.cursor_interaction_hooks();
        // Consume the startup prewarm so we reuse its open, not a second one.
        let prewarm = self.cursor_prewarm.take();

        // Open + prompt happen inside the spawned task so the TUI event loop
        // keeps polling input. The Node.js startup + 3 RPC roundtrips on a
        // first-message cold open used to block keyboard handling.
        self.response_task = Some(tokio::spawn(async move {
            let (client, session_id, model_id, capabilities) = match existing {
                Some(handles) => handles,
                None => {
                    // Reuse a *successful* prewarm (awaiting it costs only the
                    // remaining connect time). A panicked or failed-to-connect
                    // one falls through to a fresh open with bounded connect
                    // retry, so a flaky link at prewarm time doesn't surface as a
                    // dead turn on the first miss.
                    let prewarmed = match prewarm {
                        Some(handle) => handle.await.ok().and_then(Result::ok),
                        None => None,
                    };
                    let opened = match prewarmed {
                        Some(session) => Ok(session),
                        None => {
                            open_cursor_session_with_retry(
                                key,
                                requested_model.clone(),
                                cwd,
                                cursor_auto_approve,
                                permission_prompt,
                                hooks,
                                Some(tx.clone()),
                            )
                            .await
                        }
                    };
                    match opened {
                        Ok(mut session) => {
                            // Re-apply the current model — the prewarm may have
                            // opened on one the user has since changed.
                            if let Some(m) = &requested_model {
                                let _ = session.set_model(m).await;
                            }
                            if want_plan_mode {
                                let _ = session.set_mode("plan").await;
                            }
                            let handles = (
                                session.client_handle(),
                                session.session_id().to_string(),
                                session.model_id().map(str::to_string),
                                session.prompt_capabilities().clone(),
                            );
                            // Hand the live session to the event loop so future
                            // turns reuse it. The clone above keeps the Arc alive
                            // for this task even if the event loop drops it.
                            tx.send(RuntimeEvent::CursorSessionOpened(session)).ok();
                            handles
                        }
                        Err(e) => {
                            tx.send(RuntimeEvent::Finished {
                                result: Err(e.to_string()),
                                format,
                            })
                            .ok();
                            return;
                        }
                    }
                }
            };

            if let Err(e) =
                cursor_acp::ensure_image_attachments_supported(&capabilities, &attachments)
            {
                tx.send(RuntimeEvent::Finished {
                    result: Err(e.to_string()),
                    format,
                })
                .ok();
                return;
            }

            let result = drive_cursor_turn(client, session_id, model_id, input, attachments, &tx)
                .await
                .map_err(|err| err.to_string());
            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
    }

    pub(super) fn queue_attachment(&mut self, path: String) -> Result<()> {
        let attachment = build_pending_attachment(&path)?;
        let name = attachment.name.clone();
        let kind = attachment_kind_label(&attachment);
        self.draft_attachments.push(attachment);
        self.notice = Some((MUTED(), format!("Queued {kind}: {name}")));
        Ok(())
    }

    pub(super) fn detach_attachment(&mut self, index: usize) -> Result<()> {
        if index == 0 {
            anyhow::bail!("Usage: /detach <n> where n starts at 1");
        }
        let remove_at = index - 1;
        if remove_at >= self.draft_attachments.len() {
            anyhow::bail!(
                "No queued attachment #{index}. There {} {} queued.",
                if self.draft_attachments.len() == 1 {
                    "is"
                } else {
                    "are"
                },
                self.draft_attachments.len()
            );
        }
        let attachment = self.draft_attachments.remove(remove_at);
        let kind = attachment_kind_label(&attachment);
        self.notice = Some((MUTED(), format!("Removed {kind}: {}", attachment.name)));
        Ok(())
    }

    pub(super) async fn execute_slash_command(&mut self, command: SlashCommand) -> Result<bool> {
        match command {
            SlashCommand::New => {
                // A mid-execution plan never auto-continues: a prompt asks
                // first (continue / discard / Esc keeps it). A draft only hints.
                let prior_session_id = self.session_id.clone();
                if let Some(steps) = self.start_new_chat().await {
                    self.offer_plan_continue(prior_session_id, steps);
                }
                Ok(false)
            }
            SlashCommand::Exit => Ok(true),
            SlashCommand::Resume(query) => {
                self.open_resume_picker(query).await?;
                Ok(false)
            }
            SlashCommand::Model(query) => {
                // A named `/model <name>` sets the model directly — no picker, so
                // it works on providers with no `/v1/models` listing. Bare
                // `/model` opens the picker.
                match query
                    .map(|q| q.trim().to_string())
                    .filter(|q| !q.is_empty())
                {
                    Some(name) => self.set_model_direct(name).await?,
                    None => self.open_model_picker(None, ModelSelectionTarget::CurrentChat, false),
                }
                Ok(false)
            }
            SlashCommand::Key(query) => {
                self.open_or_switch_key(query).await?;
                Ok(false)
            }
            SlashCommand::Attach(path) => {
                self.queue_attachment(path)?;
                Ok(false)
            }
            SlashCommand::Detach(index) => {
                self.detach_attachment(index)?;
                Ok(false)
            }
            SlashCommand::Copy(n) => {
                self.copy_reply_to_clipboard(n)?;
                Ok(false)
            }
            SlashCommand::Skills(arg) => {
                self.run_skills_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Agents(arg) => {
                self.run_agents_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Mcp(arg) => {
                self.run_mcp_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Goal(arg) => {
                self.run_goal_command(arg).await;
                Ok(false)
            }
            SlashCommand::Plan(arg) => {
                self.run_plan_command(arg).await;
                Ok(false)
            }
            SlashCommand::Review(arg) => {
                self.run_review_command(arg).await;
                Ok(false)
            }
            SlashCommand::Memory { dream } => {
                if dream {
                    self.run_memory_dream_command().await;
                } else {
                    self.run_memory_command();
                }
                Ok(false)
            }
            SlashCommand::Effort(arg) => {
                self.run_effort_command(arg).await;
                Ok(false)
            }
            SlashCommand::CreateSkill(arg) => {
                self.run_create_skill_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Skill { name, argument } => {
                self.run_skill_command(name, argument).await?;
                Ok(false)
            }
            SlashCommand::Rewind => {
                self.open_rewind_picker().await?;
                Ok(false)
            }
            SlashCommand::Config => {
                self.open_config_overlay();
                Ok(false)
            }
            SlashCommand::Compact { fast } => {
                self.run_compact_command(fast).await;
                Ok(false)
            }
            SlashCommand::Context => {
                self.open_context_overlay().await;
                Ok(false)
            }
            SlashCommand::Session => {
                self.open_session_overlay();
                Ok(false)
            }
            SlashCommand::Share(arg) => {
                self.run_share_command(arg).await;
                Ok(false)
            }
            SlashCommand::Login => {
                self.run_login_command().await;
                Ok(false)
            }
            SlashCommand::Logout => {
                self.run_logout_command().await;
                Ok(false)
            }
            SlashCommand::Usage => {
                self.run_usage_command().await;
                Ok(false)
            }
            SlashCommand::Help => {
                self.open_help_overlay();
                Ok(false)
            }
        }
    }

    /// Re-discover the skills available for the working dir, drop any disabled in
    /// `/skills`, and cache them as slash commands so the `/` menu suggests them
    /// (`/repo-study`). Cheap dir reads; called at startup and after skill changes.
    pub(super) async fn refresh_skill_commands(&mut self) {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let mut skills = crate::agent::skills::discover_skills(std::path::Path::new(&cwd));
        if let Ok(disabled) = self.session_store.get_disabled_skills().await {
            let disabled: std::collections::HashSet<String> = disabled.into_iter().collect();
            skills.retain(|s| !disabled.contains(&s.name));
        }
        let next: Vec<SkillCommand> = skills
            .into_iter()
            .map(|s| SkillCommand {
                description: crate::agent::skills::advert_description(&s.description),
                name: s.name,
            })
            .collect();
        // Auto-reload: when the available skill set changes mid-session — the agent
        // just wrote a new `SKILL.md`, or one was created/renamed/removed/disabled —
        // drop the cached engine so the next turn re-advertises the new set to the
        // model (and the `skill` tool's enum updates). The exact conversation is
        // preserved across the rebuild. Body-only edits don't change this list; the
        // `/name` path re-reads the body fresh regardless.
        //
        // Same for subagents, comparing FULL profiles: delegation re-resolves from
        // disk (`set_agents_dir`), but the system-prompt advert and the `agent`
        // enum are baked at build — any change (a new specialist, a retuned
        // description) drops the engine so the next turn re-advertises.
        let next_subagents = crate::agent::subagents::discover_subagents(
            std::path::Path::new(&cwd),
            self.session_store.config_dir(),
        );
        if next != self.skill_commands || next_subagents != self.last_subagents {
            self.reset_engine_preserving_conversation();
        }
        self.skill_commands = next;
        self.last_subagents = next_subagents;
    }

    /// Drop the cached agent engine while keeping its exact transcript, so the next
    /// turn rebuilds (re-reading skills/tools/guides) without losing tool history.
    /// If a turn is mid-flight the lock is held and we fall back to the lossy
    /// history seed on rebuild — same as the MCP-tools rebuild path. Caller must be
    /// off the sending path for the lossless case to take effect.
    pub(super) fn reset_engine_preserving_conversation(&mut self) {
        let Some(session) = self.agent_engine.take() else {
            return;
        };
        if let Ok(engine) = session.engine.try_lock() {
            let conversation = engine.export_conversation();
            if !conversation.is_empty() {
                self.pending_agent_messages = Some(conversation);
            }
        }
    }

    /// `/repo-study [args]`: invoke a discovered skill as a slash command. Loads the
    /// skill's instructions fresh (so an edited `SKILL.md` is honored), expands them
    /// with the user's args, and sends them as a turn — a deterministic manual
    /// trigger rather than hoping the model elects to call the `skill` tool.
    pub(super) async fn run_skill_command(
        &mut self,
        name: String,
        argument: Option<String>,
    ) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let Some(skill) = crate::agent::skills::discover_skills(std::path::Path::new(&cwd))
            .into_iter()
            .find(|s| s.name == name)
        else {
            anyhow::bail!("no skill named `{name}`");
        };
        let content = expand_skill_invocation(&skill, argument.as_deref());
        let typed = match argument.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
            Some(args) => format!("/{name} {args}"),
            None => format!("/{name}"),
        };
        if self.sending {
            // A turn is in flight — queue the expanded prompt; record the typed
            // form so up-arrow recalls the command.
            self.record_draft_history(&typed);
            self.queued_messages.push(content);
            self.notice = Some((MUTED(), self.queued_notice()));
        } else {
            self.send_skill_message(content, typed).await?;
        }
        Ok(())
    }

    /// `/create-skill [intent]`: the built-in create-skill command. Unlike a
    /// discovered skill it ships in the binary (no folder, never in `/skills`); it
    /// just dispatches its embedded instructions as a turn so the agent walks the
    /// user through creating or improving a skill. The optional argument is the
    /// initial intent. Reuses the skill-invocation plumbing, so the transcript and
    /// logs show the compact `/create-skill …` (see `skill_invocation_label`).
    pub(super) async fn run_create_skill_command(
        &mut self,
        argument: Option<String>,
    ) -> Result<()> {
        let skill = crate::agent::skills::create_skill_builtin();
        // Build the prompt directly rather than via `expand_skill_invocation`:
        // create-skill's body documents the literal `$ARGUMENTS` token, which that
        // helper would otherwise substitute away. Keep the same wrapper so the
        // transcript recognizer renders the compact `/create-skill …`.
        let args = argument.as_deref().map(str::trim).filter(|a| !a.is_empty());
        let mut content = format!(
            "Use the \"{}\" skill. Follow these instructions:\n\n{}",
            skill.name, skill.body
        );
        if let Some(args) = args {
            content.push_str(&format!("\n\nInput: {args}"));
        }
        let typed = match args {
            Some(args) => format!("/create-skill {args}"),
            None => "/create-skill".to_string(),
        };
        if self.sending {
            self.record_draft_history(&typed);
            self.queued_messages.push(content);
            self.notice = Some((MUTED(), self.queued_notice()));
        } else {
            self.send_skill_message(content, typed).await?;
        }
        Ok(())
    }

    /// `/rewind`: open the picker listing every past user turn (newest first), so
    /// the user can jump back to one. Selecting a turn (see [`rewind_to_turn`])
    /// truncates the conversation there, reverts the agent's file edits made since,
    /// and restores that turn's prompt to the composer for edit/resend. Rows without
    /// their own checkpoint revert newer turns' edits through the nearest newer one;
    /// "conversation only" marks rows where no file revert is possible.
    pub(super) async fn open_rewind_picker(&mut self) -> Result<()> {
        if self.sending {
            self.queue_command(SlashCommand::Rewind, "/rewind");
            return Ok(());
        }
        let user_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "user")
            .map(|(i, _)| i)
            .collect();
        if user_indices.is_empty() {
            self.notice = Some((MUTED(), "Nothing to rewind to".to_string()));
            return Ok(());
        }
        let turn_count = user_indices.len();
        // Engine checkpoints, in order: (opening prompt, file-revertible). `sending`
        // is guarded above, so `lock().await` is uncontended — `try_lock` could
        // miss transiently and mark every turn conversation-only.
        let (targets, revert_block): (Vec<(String, bool)>, Option<&'static str>) =
            if let Some(session) = self.agent_engine.as_ref() {
                let engine = session.engine.lock().await;
                (engine.rewind_targets(), engine.rewind_file_revert_block())
            } else {
                (Vec::new(), None)
            };
        // Map display turns onto checkpoints by prompt text from the newest
        // backward, over engine-dispatched rows only (a plain-chat/ACP row with
        // identical text must not steal an engine turn's checkpoint). A flagged
        // turn that doesn't match the next checkpoint (trimmed/compacted away, or
        // predating the engine) is conversation-only and consumes no checkpoint.
        // Robust to trimming, compaction, and rebuilds — unlike positional
        // arithmetic, which restored the wrong tree when the lists drifted.
        let mut row_ordinal: Vec<Option<usize>> = vec![None; turn_count];
        let mut remaining = targets.len();
        for turn_idx in (0..turn_count).rev() {
            let history_index = user_indices[turn_idx];
            let content = &self.history[history_index].content;
            if remaining > 0
                && self.agent_turn_indices.contains(&history_index)
                && targets[remaining - 1].0 == *content
            {
                remaining -= 1;
                row_ordinal[turn_idx] = Some(remaining);
            }
        }
        // Rows without their own checkpoint (pre-resume, plain-chat/ACP, compacted
        // away) still revert newer turns' edits via the nearest newer checkpoint.
        let mut row_boundary: Vec<Option<usize>> = vec![None; turn_count];
        let mut nearest_newer: Option<usize> = None;
        for turn_idx in (0..turn_count).rev() {
            row_boundary[turn_idx] = row_ordinal[turn_idx].or(nearest_newer);
            if row_ordinal[turn_idx].is_some() {
                nearest_newer = row_ordinal[turn_idx];
            }
        }
        let mut items = Vec::with_capacity(turn_count);
        for (turn_idx, &history_index) in user_indices.iter().enumerate() {
            let ordinal = row_boundary[turn_idx];
            // The engine transcript can only truncate to a turn it checkpointed
            // itself; other rows revert files, then drop and re-seed the engine.
            let keep_engine = row_ordinal[turn_idx].is_some();
            // Files won't revert if no checkpoint applies or none from it has a tree.
            let conversation_only = match ordinal {
                Some(ord) => !targets[ord].1,
                None => true,
            };
            let prompt = rewind_excerpt(&self.history[history_index].content);
            let suffix = rewind_label_suffix(conversation_only);
            items.push(PickerEntry {
                label: format!("{}. {prompt}{suffix}", turn_idx + 1),
                search_text: self.history[history_index].content.clone(),
                value: PickerValue::RewindTurn {
                    history_index,
                    ordinal,
                    keep_engine,
                },
            });
        }
        // Newest turn at the top — the common rewind target.
        items.reverse();
        // Home/root launch dirs never snapshot — explain once in the title.
        let title = match revert_block {
            Some("home directory") => "Rewind to turn — file revert off (launched in home dir)",
            Some("filesystem root") => "Rewind to turn — file revert off (launched at fs root)",
            Some(_) => "Rewind to turn — file revert off here",
            None => "Rewind to turn",
        };
        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            title,
            String::new(),
            items,
            PickerKind::Rewind,
        )));
        Ok(())
    }

    /// Apply a `/rewind` to the picked turn: truncate history there, restore its
    /// prompt + attachments to the composer, and revert file edits through
    /// `ordinal` when a checkpoint applies. `keep_engine: false` drops the engine
    /// after the revert; no checkpoint at all means a lossy conversation-only reset.
    pub(super) async fn rewind_to_turn(
        &mut self,
        history_index: usize,
        ordinal: Option<usize>,
        keep_engine: bool,
    ) -> Result<()> {
        if self.sending {
            self.notice = Some((
                MUTED(),
                "Can't rewind while a turn is in progress".to_string(),
            ));
            return Ok(());
        }
        if history_index >= self.history.len() {
            // Stale selection (history changed under the overlay) — ignore.
            return Ok(());
        }
        let removed = self.history[history_index].clone();
        self.history.truncate(history_index);
        // Surviving indices are unchanged by the truncation — keep them.
        self.agent_turn_indices.retain(|&i| i < history_index);
        // Indices shift on truncation — drop inline-expand state and durations so
        // they can't point at the wrong block.
        self.expanded_thinking.clear();
        self.expanded_output.clear();
        self.local_outputs.clear();
        self.reasoning_durations.clear();
        self.turn_durations.clear();
        // The plan reply may be truncated away; the in-memory plan stays usable.
        self.plan_card_idx = None;
        self.draft = removed.content;
        self.cursor = self.draft.len();
        self.draft_attachments = removed.attachments;
        self.clear_transcript_selection();
        self.cursor_acp_session = None;
        self.follow_output = true;

        let notice = match ordinal {
            Some(ord) if self.agent_engine.is_some() => {
                // Rewind through the engine: truncates the conversation, reverts the
                // rewound turns' files, keeps earlier checkpoints. Lock is
                // uncontended (`sending` guarded above).
                let session = self.agent_engine.as_ref().expect("engine present");
                let mut engine = session.engine.lock().await;
                let outcome = engine.rewind_to(ord).await;
                drop(engine);
                if !keep_engine {
                    // The engine transcript can't truncate to this un-checkpointed
                    // target: drop it and clear the durable transcript, or a
                    // resume restores the pre-rewind conversation.
                    self.agent_engine = None;
                    self.pending_agent_messages = None;
                    let _ = self
                        .session_store
                        .save_agent_messages(&self.session_id, &[])
                        .await;
                }
                rewind_notice(&outcome)
            }
            _ => {
                // No matching checkpoint (predates the engine, trimmed/compacted, or
                // non-agent): drop the engine so the next turn re-seeds from the
                // trimmed history, and clear the durable transcript — persisted AND
                // pending, or a resumed session restores the pre-rewind conversation.
                self.agent_engine = None;
                self.pending_agent_messages = None;
                let _ = self
                    .session_store
                    .save_agent_messages(&self.session_id, &[])
                    .await;
                "Rewound (conversation only — file edits not reverted)".to_string()
            }
        };
        // The measured fill described the truncated turns — re-estimate, or the
        // footer and `/context` keep anchoring to the stale total.
        self.context_tokens = self.estimated_context_used().await;
        self.context_is_estimate = true;
        self.last_usage = None;
        self.notice = Some((MUTED(), notice));
        self.persist_history().await?;
        Ok(())
    }

    /// `!cmd`: run a shell command locally in the agent's working dir, streaming
    /// its output live into the transcript. Purely local — the command and its
    /// output are never sent to the model (a display-only escape hatch; the agent
    /// path already has a model-driven bash tool for that). Returns immediately
    /// after spawning the reader task; output arrives via `LocalCommandLine`
    /// events and the run is committed to history on `LocalCommandDone`.
    pub(super) fn start_local_command(&mut self, command: String) {
        if self.local_command.is_some() {
            self.notice = Some((MUTED(), "A command is already running".to_string()));
            return;
        }
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        // Spawn the command (PTY on Unix for live line-buffered streaming, plain pipes
        // on Windows where ConPTY never EOFs); hold its killer so esc can stop it. The
        // blocking read runs on a worker.
        let shell = match spawn_local_shell(&command, std::path::Path::new(&cwd)) {
            Ok(shell) => shell,
            Err(err) => {
                self.notice = Some((ERROR(), format!("Failed to run command: {err}")));
                return;
            }
        };
        let killer = shell.killer_handle();
        let tx = self.tx.clone();
        let task = tokio::spawn(async move {
            let _ = tokio::task::spawn_blocking(move || run_local_to_completion(shell, tx)).await;
        });
        self.local_command = Some(LocalCommandRun {
            task,
            killer,
            started_at: Instant::now(),
            command,
            stdout: String::new(),
            stderr: String::new(),
        });
        self.follow_output = true;
        self.notice = None;
    }

    /// Esc while a `!cmd` is running: kill the child (via the reader task's
    /// `kill_on_drop`) and commit whatever it produced so far as an interrupted
    /// `local_command` entry.
    pub(super) async fn interrupt_local_command(&mut self) -> Result<()> {
        let Some(mut run) = self.local_command.take() else {
            return Ok(());
        };
        // Kill the PTY child (the blocking read can't be cancelled by aborting the
        // task alone); the reader then hits EOF and the worker winds down.
        let _ = run.killer.kill();
        run.task.abort();
        self.record_local_output(run.command, run.stdout, run.stderr, -1, false, true);
        self.notice = Some((MUTED(), "Command interrupted".to_string()));
        self.persist_history().await?;
        Ok(())
    }

    /// `/copy [n]`: copy the Nth-latest assistant reply (default most recent) to
    /// the system clipboard.
    pub(super) fn copy_reply_to_clipboard(&mut self, n: Option<usize>) -> Result<()> {
        if n == Some(0) {
            anyhow::bail!("Usage: /copy [n] — n counts back from the latest reply, starting at 1");
        }
        let nth = n.unwrap_or(1);
        let replies = self
            .history
            .iter()
            .rev()
            .filter(|m| m.role == "assistant" && !m.content.trim().is_empty());
        let reply = replies.clone().nth(nth - 1).map(|m| m.content.clone());
        let Some(reply) = reply else {
            let count = replies.count();
            if count == 0 {
                anyhow::bail!("No assistant reply to copy yet");
            }
            anyhow::bail!(
                "Only {count} repl{} to copy",
                if count == 1 { "y" } else { "ies" }
            );
        };
        write_system_clipboard(&reply)?;
        let label = if nth == 1 {
            "Copied the latest reply".to_string()
        } else {
            format!("Copied reply #{nth}")
        };
        self.notice = Some((MUTED(), label));
        Ok(())
    }

    pub(super) fn push_newline(&mut self) {
        self.leave_history_navigation();
        self.insert_char_at_cursor('\n');
    }

    pub(super) fn reset_composer(&mut self) {
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    /// `/memory`: audit surface for `remember` — the saved facts as a card,
    /// with the file path for hand edits. Local, no model call.
    pub(super) fn run_memory_command(&mut self) {
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        let path = crate::agent::memory::project_memory_path(std::path::Path::new(&cwd));
        let entries = crate::agent::memory::load_entries(&path);
        let global_path = crate::agent::memory::global_memory_path();
        let global = crate::agent::memory::load_entries(&global_path);
        if entries.is_empty() && global.is_empty() {
            self.notice = Some((
                MUTED(),
                "No memory yet — the agent saves durable facts with its `remember` tool"
                    .to_string(),
            ));
            return;
        }
        let mut content = String::new();
        if !entries.is_empty() {
            content.push_str(&format!(
                "{} project fact(s) injected into every session here:\n\n",
                entries.len()
            ));
            for e in &entries {
                content.push_str(&format!("- {e}\n"));
            }
            content.push_str(&format!("\nEdit or delete lines: `{}`\n", path.display()));
        }
        if !global.is_empty() {
            if !entries.is_empty() {
                content.push('\n');
            }
            content.push_str(&format!(
                "{} global fact(s) injected in every project:\n\n",
                global.len()
            ));
            for e in &global {
                content.push_str(&format!("- {e}\n"));
            }
            content.push_str(&format!(
                "\nEdit or delete lines: `{}`",
                global_path.display()
            ));
        }
        self.history.push(ChatMessage {
            model: None,
            role: "memory".to_string(),
            content,
            reasoning_content: None,
            attachments: vec![],
        });
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
    }

    /// `/memory dream`: consolidate the session log into curated memory now, bypassing the gate.
    pub(super) async fn run_memory_dream_command(&mut self) {
        if self.sending {
            self.notice = Some((
                MUTED(),
                "busy — run /memory dream after the current turn".to_string(),
            ));
            return;
        }
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        let cwd = std::path::Path::new(&cwd);
        let Some(input) = crate::agent::memory::build_dream_input(cwd) else {
            self.notice = Some((MUTED(), "no session history to consolidate yet".to_string()));
            return;
        };
        let (base, auth) = match self.start_agent_serve(None).await {
            Ok(t) => t,
            Err(e) => {
                self.notice = Some((ERROR(), format!("memory dream serve failed: {e}")));
                return;
            }
        };
        let client = crate::services::http_utils::router_http_client();
        let outcome = Self::drive_dream(cwd, input, &client, &base, Some(&auth), &self.model).await;
        self.stop_agent_serve();
        self.notice = Some(match outcome {
            Some(o) => (
                MUTED(),
                format!(
                    "memory consolidated — {} curated fact(s); {} session line(s) folded in",
                    o.entries, o.cleared
                ),
            ),
            None => (MUTED(), "memory dream produced nothing to save".to_string()),
        });
    }

    /// Opt-in (`AIVO_AGENT_MEMORY_DREAM`) background consolidation at session start.
    pub(super) fn spawn_startup_dream(&self) {
        use crate::agent::memory::{DreamGate, dream_gate};
        if crate::services::system_env::env_flag("AIVO_AGENT_MEMORY_DREAM") != Some(true) {
            return;
        }
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        if !matches!(dream_gate(std::path::Path::new(&cwd)), DreamGate::Open(_)) {
            return;
        }
        tokio::spawn(Self::run_dream_with_serve(
            self.key.clone(),
            self.session_store.clone(),
            self.model.clone(),
            cwd,
        ));
    }

    async fn run_dream_with_serve(
        key: ApiKey,
        session_store: crate::services::session_store::SessionStore,
        model: String,
        cwd: String,
    ) {
        let cwd = std::path::Path::new(&cwd);
        let Some(input) = crate::agent::memory::build_dream_input(cwd) else {
            return;
        };
        let (router, auth) = Self::build_agent_serve_router(&key, &session_store, None).await;
        let Ok((handle, shutdown, port)) = router.start_background_with_addr("127.0.0.1", 0).await
        else {
            return;
        };
        let base = format!("http://127.0.0.1:{port}");
        let client = crate::services::http_utils::router_http_client();
        let _ = Self::drive_dream(cwd, input, &client, &base, Some(&auth), &model).await;
        shutdown.notify_one();
        handle.abort();
    }

    async fn drive_dream(
        cwd: &std::path::Path,
        input: (String, Vec<String>),
        client: &reqwest::Client,
        base: &str,
        auth: Option<&str>,
        model: &str,
    ) -> Option<crate::agent::memory::DreamOutcome> {
        let (existing, consumed) = input;
        let request = crate::agent::memory::build_dream_request(model, &existing, &consumed);
        let msg = crate::agent::serve_client::complete(client, base, auth, &request, &mut |_| {})
            .await
            .ok()?;
        crate::agent::memory::apply_dream_result(cwd, &msg.content.unwrap_or_default(), &consumed)
            .ok()
    }

    /// `/review [ref|scope]`: one agent turn under the review directive — no
    /// mode state, the directive travels in the message.
    pub(super) async fn run_review_command(&mut self, arg: Option<String>) {
        if self.sending {
            self.queue_command(SlashCommand::Review(arg), "/review");
            return;
        }
        if self.plan_mode {
            self.notice = Some((
                ERROR(),
                "Plan mode is active — approve the plan or /plan stop before /review".to_string(),
            ));
            return;
        }
        if !self.agent_capable() {
            self.notice = Some((
                ERROR(),
                "/review needs the native agent (an API key or Copilot — not OAuth or cursor)"
                    .to_string(),
            ));
            return;
        }
        let target = arg.as_deref().map(str::trim).unwrap_or("");
        let (prompt, typed) = if target.is_empty() {
            (
                format!("{REVIEW_PREAMBLE}\n\nReview target: the current working diff."),
                "/review".to_string(),
            )
        } else {
            (
                format!("{REVIEW_PREAMBLE}\n\nReview target: `{target}`"),
                format!("/review {target}"),
            )
        };
        if let Err(e) = self
            .dispatch_user_message_shown(prompt, None, Some(typed))
            .await
        {
            self.notice = Some((ERROR(), e.to_string()));
        }
    }

    /// `/goal`: autonomous goal mode. `<objective>` starts it; bare shows
    /// status; `stop` ends it. The loop is driven by `maybe_continue_goal`.
    pub(super) async fn run_goal_command(&mut self, arg: Option<String>) {
        match arg.as_deref().map(str::trim) {
            None | Some("") | Some("status") => {
                let msg = match &self.goal_mode {
                    Some(g) => {
                        let mut obj: String = g.objective.chars().take(48).collect();
                        if g.objective.chars().count() > 48 {
                            obj.push('…');
                        }
                        format!(
                            "Goal: \"{}\" (turn {}/{}) — /goal stop to end",
                            obj, g.iteration, g.max
                        )
                    }
                    None => {
                        "Usage: /goal <objective> — work autonomously until done; /goal stop to end"
                            .to_string()
                    }
                };
                self.notice = Some((MUTED(), msg));
            }
            Some("stop") | Some("off") | Some("cancel") => {
                let msg = if self.goal_mode.take().is_some() {
                    "Goal mode stopped"
                } else {
                    "Goal mode wasn't active"
                };
                self.notice = Some((MUTED(), msg.to_string()));
            }
            Some(objective) => {
                if self.sending {
                    self.queue_command(SlashCommand::Goal(Some(objective.to_string())), "/goal");
                    return;
                }
                if self.plan_mode {
                    self.notice = Some((
                        ERROR(),
                        "Plan mode is read-only — approve the plan or /plan stop before /goal"
                            .to_string(),
                    ));
                    return;
                }
                if !self.agent_capable() {
                    self.notice = Some((
                        ERROR(),
                        "Goal mode needs the native agent (an API key or Copilot — not OAuth or cursor)"
                            .to_string(),
                    ));
                    return;
                }
                self.goal_mode = Some(GoalState {
                    objective: objective.to_string(),
                    iteration: 1,
                    max: goal_max_iterations(),
                    msg_floor: self.history.len(),
                });
                // Fresh objective: no stale guard-stop from a prior loop.
                self.goal_guard_stop = None;
                let first = format!("{GOAL_PREAMBLE}\n\nObjective: {objective}");
                let typed = format!("/goal {objective}");
                if let Err(e) = self
                    .dispatch_user_message_shown(first, None, Some(typed))
                    .await
                {
                    self.goal_mode = None;
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                // Dispatch can disarm the goal (plain-chat route) or decline to
                // send (image guard); an armed goal with nothing in flight stalls.
                if self.goal_mode.is_none() {
                    return;
                }
                if !self.sending {
                    self.goal_mode = None;
                    return;
                }
                // `send_user_message` clears the notice; hint about unattended runs after.
                if !self.agent_auto_approve {
                    self.notice = Some((
                        MUTED(),
                        "Goal mode on — press Shift+Tab to auto-approve tools so it runs unattended"
                            .to_string(),
                    ));
                }
            }
        }
    }

    /// Drive the active `/goal` loop after a turn: stop on the completion marker,
    /// an errored turn, or the turn cap; otherwise auto-send the continuation.
    pub(super) async fn maybe_continue_goal(&mut self) -> Result<()> {
        let Some(floor) = self.goal_mode.as_ref().map(|g| g.msg_floor) else {
            return Ok(());
        };
        // Checked even mid-queued-turn: the rev-find skips the newer user message.
        // Rows below the floor predate this goal — a stale reply must not end it.
        let last_reply = self
            .history
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.role == "assistant")
            .filter(|(i, _)| *i >= floor)
            .map(|(_, m)| m.content.clone())
            .unwrap_or_default();
        if signals_goal_complete(&last_reply) {
            let turns = self.goal_mode.take().map(|g| g.iteration).unwrap_or(0);
            let s = if turns == 1 { "" } else { "s" };
            self.notice = Some((MUTED(), format!("Goal complete (in {turns} turn{s})")));
            return Ok(());
        }
        // An errored turn must not auto-repeat — stop and keep the error visible.
        // The durable `error` transcript row is the signal; an incidental ERROR
        // notice (say, a failed /copy mid-turn) must not kill an unattended loop.
        let errored = self
            .history
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| matches!(m.role.as_str(), "assistant" | "error"))
            .is_some_and(|(i, m)| m.role == "error" && i >= floor);
        if errored {
            self.goal_mode = None;
            match self.notice.as_mut() {
                Some((color, msg)) if *color == ERROR() => msg.push_str(" — goal mode stopped"),
                _ => {
                    self.notice =
                        Some((ERROR(), "Goal mode stopped — the turn errored".to_string()));
                }
            }
            return Ok(());
        }
        if self.sending {
            // A queued message drives the next turn; drop the superseded guard-stop.
            self.goal_guard_stop = None;
            return Ok(());
        }
        let Some(goal) = self.goal_mode.as_mut() else {
            return Ok(());
        };
        goal.iteration += 1;
        if goal.iteration > goal.max {
            let max = goal.max;
            self.goal_mode = None;
            self.notice = Some((
                MUTED(),
                format!(
                    "Goal mode stopped at the {max}-turn cap (/goal <objective> to keep going)"
                ),
            ));
            return Ok(());
        }
        let objective = goal.objective.clone();
        // An early-stopped turn steers the next one instead of blindly continuing.
        let continuation = match self.goal_guard_stop.take() {
            Some(stop) => {
                use crate::agent::engine::TurnStop;
                let steer = match stop {
                    TurnStop::NoProgress => {
                        "it repeated the same action without progress. Do NOT retry the same \
approach — pick a different one, and record the dead end with take_note"
                    }
                    TurnStop::ToolFailureLoop => {
                        "a tool call kept failing the same way. Do NOT retry the same call — \
fix the input or pick another route, and record the dead end with take_note"
                    }
                    TurnStop::StepLimit => {
                        "it ran out of steps mid-work. Break the remaining work into smaller \
pieces and keep going"
                    }
                };
                format!(
                    "[Previous turn stopped early: {steer}.]\n\n{}",
                    goal_continue_message(&objective)
                )
            }
            None => goal_continue_message(&objective),
        };
        // Machine text: record nothing (↑/↓ recall) and stash the composer so the
        // dispatch can't wipe a mid-turn draft or attach the user's staged files.
        let draft = std::mem::take(&mut self.draft);
        let cursor = self.cursor;
        let attachments = std::mem::take(&mut self.draft_attachments);
        let sent = self
            .dispatch_user_message_shown(continuation, None, Some("/goal — continue".to_string()))
            .await;
        self.draft = draft;
        self.cursor = cursor;
        self.draft_attachments = attachments;
        self.sync_command_menu_state();
        match sent {
            // Propagating would abort the event loop over one bad dispatch.
            Err(e) => {
                self.goal_mode = None;
                self.notice = Some((ERROR(), format!("{e} — goal mode stopped")));
            }
            // Dispatch declined to send; an armed goal with nothing in flight stalls.
            Ok(()) => {
                if !self.sending {
                    self.goal_mode = None;
                }
            }
        }
        Ok(())
    }

    /// Enter plan mode: quiet the other modes (exclusive, incl. `/goal`), restrict
    /// a live engine in place (a not-yet-built one enters at build time). `false`
    /// when the key can't drive the native agent. No toast — callers word their own.
    pub(super) async fn enter_plan_mode(&mut self) -> bool {
        if !self.agent_capable() {
            return false;
        }
        self.set_auto_quiet(false);
        self.set_review_quiet(false);
        // A goal would auto-continue into the read-only engine; mirror /goal's gate.
        self.goal_mode = None;
        self.plan_mode = true;
        self.plan_exit_pending = false;
        // A stale live exit request must not cancel the fresh plan mode.
        self.plan_exit_flag
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.pending_plan = None;
        // While sending, the turn task holds the lock — the restriction lands via
        // the mode sync at the next `spawn_agent_turn`.
        if !self.sending
            && let Some(session) = self.agent_engine.as_ref()
        {
            session.engine.lock().await.set_plan_mode(true);
        }
        self.persist_plan_state().await;
        true
    }

    /// Leave plan mode, restoring the engine's tools in place (no rebuild). Mid-turn
    /// the turn task holds the lock, so the exit rides the live flag instead. No
    /// toast — callers word their own.
    pub(super) async fn leave_plan_mode(&mut self, discard_draft: bool) {
        self.plan_mode = false;
        if discard_draft {
            self.pending_plan = None;
            self.plan_card_idx = None;
        }
        if self.sending {
            self.request_plan_exit_live();
        } else {
            if let Some(session) = self.agent_engine.as_ref() {
                session.engine.lock().await.set_plan_mode(false);
            }
            self.plan_exit_pending = false;
        }
        self.persist_plan_state().await;
    }

    /// Mid-turn plan exit: flip TUI state and signal the running turn via the live
    /// flag. `plan_exit_pending` is the fallback for a turn that makes no further
    /// tool calls (applied at turn end).
    pub(super) fn request_plan_exit_live(&mut self) {
        self.plan_mode = false;
        self.plan_exit_pending = true;
        self.plan_exit_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// `/plan resume`: pick an unfinished plan from another session in this
    /// directory and carry it into the current one. The current session is skipped
    /// — its plan is already armed.
    pub(super) async fn open_plan_resume_picker(&mut self, query: String) {
        let scope_cwd = (!self.real_cwd.is_empty()).then(|| self.real_cwd.clone());
        let mut entries = match self.session_store.all_chat_sessions().await {
            Ok(entries) => entries,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Couldn't list sessions: {e}")));
                return;
            }
        };
        entries.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        let mut items = Vec::new();
        // Suffixes of sessions that made a row (plus this one) — their managed
        // save files would be duplicates, so exclude them below.
        let mut rowed_suffixes = vec![plan_file_suffix(&self.session_id)];
        // THIS session first: an interrupted checklist continues right here;
        // without a row `/plan resume` would deny a plan sitting in the transcript.
        if let Some(steps) = self.unfinished_plan_steps() {
            items.push(PickerEntry {
                label: format!(
                    "this session — {} · next: {}",
                    plan_steps_progress(&steps),
                    plan_next_step(&steps)
                ),
                search_text: format!("this session {steps}"),
                value: PickerValue::PlanResume(PlanCarry::Interrupted(steps.to_string())),
            });
        }
        for entry in entries {
            if entry.session_id == self.session_id
                || scope_cwd.as_deref().is_some_and(|dir| entry.cwd != dir)
            {
                continue;
            }
            let Ok(Some(state)) = self.session_store.get_code_session(&entry.session_id).await
            else {
                continue;
            };
            let Some(plan) = state.plan_state else {
                continue;
            };
            // Mid-execution wins over a stale draft: continuing beats re-approving.
            if let Some(steps) = plan.steps {
                let progress = plan_steps_progress(&steps);
                let next = plan_next_step(&steps);
                items.push(PickerEntry {
                    label: format!("{} — {progress} · next: {next}", entry.title),
                    search_text: format!("{} {} {steps}", entry.title, entry.session_id),
                    value: PickerValue::PlanResume(PlanCarry::Continue(steps.to_string())),
                });
                rowed_suffixes.push(plan_file_suffix(&entry.session_id));
                continue;
            }
            // Mode-only planning (no draft yet) has nothing to carry over.
            let Some(draft) = plan.draft.filter(|draft| !draft.trim().is_empty()) else {
                continue;
            };
            let first_line = draft
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("")
                .trim()
                .to_string();
            items.push(PickerEntry {
                label: format!("{} — {}", entry.title, first_line),
                search_text: format!("{} {} {}", entry.title, entry.session_id, draft),
                value: PickerValue::PlanResume(PlanCarry::Draft(draft)),
            });
            rowed_suffixes.push(plan_file_suffix(&entry.session_id));
        }
        let session_rows = items.len();
        items.extend(self.saved_plan_file_entries(&rowed_suffixes));
        if items.is_empty() {
            self.notice = Some((
                MUTED(),
                "No unfinished plans in this directory — a plan stays with its session (and /plan save) until finished or discarded"
                    .to_string(),
            ));
            return;
        }
        // One candidate and no filter: skip the picker and carry it over.
        // Session rows only — a lone saved FILE may come from another project
        // (the plans dir is global), so show the picker rather than surprise.
        if items.len() == 1 && session_rows == 1 && query.is_empty() {
            let PickerValue::PlanResume(carry) = items.remove(0).value else {
                unreachable!("plan picker rows are PlanResume");
            };
            self.resume_plan_from_session(carry).await;
            return;
        }
        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Unfinished plans",
            query,
            items,
            PickerKind::PlanResume,
        )));
    }

    /// Picker rows for managed `/plan save` files whose session made no row of
    /// its own (deleted, other directory, hand-dropped). They carry as drafts,
    /// newest first.
    fn saved_plan_file_entries(&self, rowed_suffixes: &[String]) -> Vec<PickerEntry> {
        let dir = crate::services::paths::plans_dir(self.session_store.config_dir());
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
            .flatten()
            .filter(|e| {
                let name = e.file_name();
                let Some(name) = name.to_str() else {
                    return false;
                };
                name.ends_with(".md") && !rowed_suffixes.iter().any(|s| name.ends_with(s.as_str()))
            })
            .map(|e| {
                let modified = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                (modified, e.path())
            })
            .collect();
        files.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
        files
            .into_iter()
            .filter_map(|(_, path)| {
                let contents = std::fs::read_to_string(&path).ok()?;
                if contents.trim().is_empty() {
                    return None;
                }
                let title = plan_title(&contents).to_string();
                Some(PickerEntry {
                    label: format!("{title} — saved plan"),
                    search_text: format!("{title} {} {contents}", path.display()),
                    value: PickerValue::PlanResume(PlanCarry::Draft(contents)),
                })
            })
            .collect()
    }

    /// Carry a picked plan into this session. A draft enters plan mode, arms
    /// `/plan go`, and sends a re-validation kick-off ending in the approval card;
    /// a mid-execution checklist continues straight into execution.
    pub(super) async fn resume_plan_from_session(&mut self, carry: PlanCarry) {
        if self.sending {
            self.notice = Some((
                MUTED(),
                "A turn is running — /plan resume again when it finishes".to_string(),
            ));
            return;
        }
        match carry {
            PlanCarry::Continue(steps) => {
                let sent = self
                    .dispatch_preserving_composer(
                        plan_continue_message(&steps),
                        Some("/plan resume".to_string()),
                    )
                    .await;
                if let Err(e) = sent {
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                self.notice = Some((
                    MUTED(),
                    "Continuing the carried-over plan — completed steps get a quick re-check first"
                        .to_string(),
                ));
            }
            PlanCarry::Interrupted(steps) => {
                let sent = self
                    .dispatch_preserving_composer(
                        plan_interrupted_message(&steps),
                        Some("/plan resume".to_string()),
                    )
                    .await;
                if let Err(e) = sent {
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                self.notice = Some((
                    MUTED(),
                    "Continuing this session's plan — completed steps get a quick re-check first"
                        .to_string(),
                ));
            }
            PlanCarry::Draft(draft) => {
                if !self.enter_plan_mode().await {
                    self.notice = Some((
                        ERROR(),
                        "Plan mode needs the native agent (an API key or Copilot — not OAuth or cursor)"
                            .to_string(),
                    ));
                    return;
                }
                // `enter_plan_mode` cleared any local draft; re-arm now so
                // `/plan go` works before the model re-frames the plan.
                self.pending_plan = Some(draft.clone());
                self.plan_card_idx = None;
                self.persist_plan_state().await;
                let sent = self
                    .dispatch_preserving_composer(
                        plan_carryover_message(&draft),
                        Some("/plan resume".to_string()),
                    )
                    .await;
                if let Err(e) = sent {
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                self.notice = Some((
                    MUTED(),
                    "Plan carried over — checking it against the current code; approve on the card (or /plan go)"
                        .to_string(),
                ));
            }
        }
    }

    /// Arm the `/new` continue-the-plan prompt for the outgoing session's
    /// mid-execution checklist.
    pub(super) fn offer_plan_continue(
        &mut self,
        prior_session_id: String,
        steps: serde_json::Value,
    ) {
        self.cards.plan_continue = Some(PlanContinuePrompt {
            steps,
            prior_session_id,
        });
    }

    /// Resolve the continue-the-plan prompt: `y`/`Enter` carries the checklist
    /// into this session, `n` discards it (clears the old session's planState
    /// and managed save file), `Esc` keeps it for `/plan resume`. Returns `true`
    /// when consumed as a decision; `y`/`n`/`Enter` only act on an empty composer
    /// (mid-draft they belong to the message, like the permission card).
    pub(super) async fn handle_plan_continue_key(&mut self, key: KeyEvent) -> bool {
        // Esc is never message content: it always resolves to "later".
        if matches!(key.code, KeyCode::Esc) {
            if let Some(prompt) = self.cards.plan_continue.take() {
                self.notice = Some((
                    MUTED(),
                    format!(
                        "Plan kept aside ({}) — /plan resume picks it back up",
                        plan_steps_progress(&prompt.steps)
                    ),
                ));
            }
            return true;
        }
        if !self.draft.is_empty() {
            return false;
        }
        let cont = match key.code {
            KeyCode::Char('y' | 'Y') | KeyCode::Enter => true,
            KeyCode::Char('n' | 'N') => false,
            _ => return false,
        };
        let Some(prompt) = self.cards.plan_continue.take() else {
            return false;
        };
        if cont {
            self.resume_plan_from_session(PlanCarry::Continue(prompt.steps.to_string()))
                .await;
            return true;
        }
        let _ = self
            .session_store
            .set_plan_state(&prompt.prior_session_id, None)
            .await;
        remove_managed_plan_files(
            self.session_store.config_dir(),
            &prompt.prior_session_id,
            None,
        );
        self.notice = Some((MUTED(), "Unfinished plan discarded".to_string()));
        true
    }

    /// Dispatch a machine-generated turn, stashing the composer so it can't swallow
    /// a draft or attachment. `shown` = the compact command shown instead of the
    /// machine text (`None` shows the text).
    async fn dispatch_preserving_composer(
        &mut self,
        message: String,
        shown: Option<String>,
    ) -> Result<()> {
        let stash = std::mem::take(&mut self.draft);
        let cursor = self.cursor;
        let attachments = std::mem::take(&mut self.draft_attachments);
        let sent = self.dispatch_user_message_shown(message, None, shown).await;
        self.draft = stash;
        self.cursor = cursor;
        self.draft_attachments = attachments;
        self.sync_command_menu_state();
        sent
    }

    /// `/plan`: `[objective]` enters plan mode; bare also sends a kick-off turn
    /// so the agent interviews for the objective; `go [guidance]` approves a
    /// drafted plan and executes it in the same session; `save [file]` writes
    /// the plan to a markdown file; `resume` continues this session's
    /// interrupted plan, else carries one over from a saved session (or a
    /// `save`d file); `stop` leaves.
    pub(super) async fn run_plan_command(&mut self, arg: Option<String>) {
        let arg = arg.as_deref().map(str::trim).unwrap_or("");
        // First word = action; the rest is `go`'s optional guidance.
        let (head, rest) = match arg.split_once(char::is_whitespace) {
            Some((h, r)) => (h, r.trim()),
            None => (arg, ""),
        };
        // Cursor runs its own agent: `/plan` maps to cursor's ACP mode, not the
        // in-process engine's plan machinery.
        if self.key.is_cursor_acp() {
            self.run_cursor_plan_command(head, rest, arg).await;
            return;
        }
        match head {
            "go" | "run" | "execute" => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some(arg.to_string())), "/plan go");
                    return;
                }
                if self.pending_plan.is_none() {
                    self.notice = Some((
                        MUTED(),
                        "No plan yet — /plan <objective> to draft one first (or approve on the plan card)"
                            .to_string(),
                    ));
                    return;
                }
                // Mirrors the approval card: `-y` = auto-approve, default = review.
                let auto = rest == "-y" || rest.starts_with("-y ");
                let guidance = rest.strip_prefix("-y").map(str::trim).unwrap_or(rest);
                // Same session — the plan is already in the engine's history.
                self.leave_plan_mode(true).await;
                self.set_auto_quiet(auto);
                self.set_review_quiet(!auto);
                if let Err(e) = self
                    .dispatch_preserving_composer(plan_go_message(guidance), None)
                    .await
                {
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                self.notice = Some((
                    MUTED(),
                    if auto {
                        "Executing the approved plan with auto-approve".to_string()
                    } else {
                        "Executing the approved plan — each edit shows a diff to approve"
                            .to_string()
                    },
                ));
            }
            // Allowed mid-turn: saving only reads state and writes a file.
            "save" => self.save_plan_to_file(rest).await,
            "resume" | "list" => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some(arg.to_string())), "/plan resume");
                    return;
                }
                // An interrupted checklist is a picker row, but a local DRAFT
                // isn't — it's already armed, so point at the approval path.
                if head == "resume"
                    && rest.is_empty()
                    && self.pending_plan.is_some()
                    && self.unfinished_plan_steps().is_none()
                {
                    self.notice = Some((
                        MUTED(),
                        "A plan is already drafted here — approve it on the card or /plan go"
                            .to_string(),
                    ));
                    return;
                }
                // An existing file resumes as a draft; anything else filters the picker.
                if !rest.is_empty() {
                    let path = self.resolve_user_path(rest);
                    if path.is_file() {
                        match std::fs::read_to_string(&path) {
                            Ok(contents) if !contents.trim().is_empty() => {
                                self.resume_plan_from_session(PlanCarry::Draft(contents))
                                    .await;
                            }
                            Ok(_) => {
                                self.notice = Some((
                                    MUTED(),
                                    format!("{} is empty — nothing to resume", path.display()),
                                ));
                            }
                            Err(e) => {
                                self.notice = Some((
                                    ERROR(),
                                    format!("Couldn't read {}: {e}", path.display()),
                                ));
                            }
                        }
                        return;
                    }
                }
                self.open_plan_resume_picker(rest.to_string()).await;
            }
            "stop" | "cancel" | "discard" | "off" => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some("stop".to_string())), "/plan stop");
                    return;
                }
                let was_on = self.plan_mode || self.pending_plan.is_some();
                let had_plan = self.pending_plan.is_some();
                self.leave_plan_mode(true).await;
                let msg = match (was_on, had_plan) {
                    (true, true) => "Plan mode off — plan discarded",
                    (true, false) => "Plan mode off",
                    (false, _) => "Plan mode isn't on",
                };
                self.notice = Some((MUTED(), msg.to_string()));
            }
            "" => {
                if self.plan_mode {
                    self.notice = Some((
                        MUTED(),
                        if self.pending_plan.is_some() {
                            "Plan mode is on — approve the plan card (or /plan go), or /plan stop to leave"
                        } else {
                            "Plan mode is on — describe what to plan, or /plan stop to leave"
                        }
                        .to_string(),
                    ));
                    return;
                }
                if self.sending {
                    self.queue_command(SlashCommand::Plan(None), "/plan");
                    return;
                }
                let goal_stopped = self.goal_mode.is_some();
                if !self.enter_plan_mode().await {
                    self.notice = Some((
                        ERROR(),
                        "Plan mode needs the native agent (an API key or Copilot — not OAuth or cursor)"
                            .to_string(),
                    ));
                    return;
                }
                if let Err(e) = self
                    .dispatch_preserving_composer(
                        PLAN_KICKOFF_MESSAGE.to_string(),
                        Some("/plan".to_string()),
                    )
                    .await
                {
                    self.notice = Some((ERROR(), e.to_string()));
                    return;
                }
                self.notice = Some((
                    MUTED(),
                    if goal_stopped {
                        "Goal mode stopped — plan mode is read-only until you approve the plan"
                    } else {
                        "Plan mode — read-only until you approve the plan"
                    }
                    .to_string(),
                ));
            }
            _ => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some(arg.to_string())), "/plan");
                    return;
                }
                let goal_stopped = self.goal_mode.is_some();
                if !self.enter_plan_mode().await {
                    self.notice = Some((
                        ERROR(),
                        "Plan mode needs the native agent (an API key or Copilot — not OAuth or cursor)"
                            .to_string(),
                    ));
                    return;
                }
                // Bare objective (the directive lives in the system prompt);
                // record: None keeps it out of ↑/↓ recall. Keep plan mode on the
                // error path so the flag and engine restriction stay consistent.
                if let Err(e) = self.dispatch_user_message(arg.to_string(), None).await {
                    self.notice = Some((ERROR(), e.to_string()));
                } else if goal_stopped {
                    self.notice =
                        Some((MUTED(), "Goal mode stopped — planning instead".to_string()));
                }
            }
        }
    }

    /// `/plan save [path]`: write the plan (draft plus execution checklist as
    /// markdown) to a file and checkpoint planState now instead of at turn end.
    /// Bare saves go to the session's managed file under `<config>/plans/`
    /// (updated in place, found by `/plan resume`); an explicit path is the
    /// user's own file.
    async fn save_plan_to_file(&mut self, path_arg: &str) {
        let draft = self.pending_plan.clone();
        let steps = self.unfinished_plan_steps();
        if draft.is_none() && steps.is_none() {
            self.notice = Some((
                MUTED(),
                "No plan to save — /plan <objective> drafts one".to_string(),
            ));
            return;
        }
        let mut md = String::new();
        if let Some(draft) = &draft {
            md.push_str(draft.trim_end());
            md.push('\n');
        } else {
            md.push_str("# Plan\n");
        }
        if let Some(steps) = &steps {
            md.push_str("\n## Progress\n\n");
            for item in steps.as_array().into_iter().flatten() {
                let mark = todo_checkbox(plan_status(item));
                let label = item.get("step").and_then(|s| s.as_str()).unwrap_or("…");
                md.push_str(&format!("- {mark} {label}\n"));
            }
        }
        let managed = path_arg.is_empty();
        let path = if managed {
            managed_plan_path(
                self.session_store.config_dir(),
                draft.as_deref(),
                &self.session_id,
            )
        } else {
            self.resolve_user_path(path_arg)
        };
        let write = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&path, &md));
        match write {
            Ok(()) => {
                if managed {
                    // A changed draft title changes the slug — one file per session.
                    remove_managed_plan_files(
                        self.session_store.config_dir(),
                        &self.session_id,
                        Some(&path),
                    );
                }
                self.persist_plan_state().await;
                self.notice = Some((
                    MUTED(),
                    if managed {
                        "Plan saved — /plan resume restores it, here or in another session"
                            .to_string()
                    } else {
                        format!("Plan saved to {}", path.display())
                    },
                ));
            }
            Err(e) => {
                self.notice = Some((ERROR(), format!("Couldn't write {}: {e}", path.display())));
            }
        }
    }

    /// A user-typed path: `~` expands, relatives resolve against the working dir.
    fn resolve_user_path(&self, raw: &str) -> PathBuf {
        let path = crate::services::system_env::expand_tilde(raw);
        if path.is_relative() && !self.real_cwd.is_empty() {
            return Path::new(&self.real_cwd).join(path);
        }
        path
    }

    /// `/plan` for cursor: drives cursor's ACP `plan` mode via `session/set_mode`.
    /// No drafted-plan buffer — cursor owns its plan file and raises its own
    /// `cursor/create_plan` approval card.
    async fn run_cursor_plan_command(&mut self, head: &str, rest: &str, full: &str) {
        match head {
            "save" => {
                self.notice = Some((
                    MUTED(),
                    "Cursor keeps its own plan file — /plan save needs the native agent"
                        .to_string(),
                ));
            }
            "stop" | "cancel" | "discard" | "off" => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some("stop".to_string())), "/plan stop");
                    return;
                }
                if !self.cursor_plan_mode {
                    self.notice = Some((MUTED(), "Plan mode isn't on".to_string()));
                    return;
                }
                self.set_cursor_mode(false).await;
                self.notice = Some((
                    MUTED(),
                    "Plan mode off — cursor is back in agent mode".to_string(),
                ));
            }
            "go" | "run" | "execute" => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some(full.to_string())), "/plan go");
                    return;
                }
                self.set_cursor_mode(false).await;
                if rest.is_empty() {
                    self.notice = Some((
                        MUTED(),
                        "Back in agent mode — send the go-ahead to execute the plan".to_string(),
                    ));
                } else if let Err(e) = self.dispatch_user_message(rest.to_string(), None).await {
                    self.notice = Some((ERROR(), e.to_string()));
                }
            }
            "" | "on" => {
                if self.cursor_plan_mode {
                    self.notice = Some((
                        MUTED(),
                        "Plan mode is on — describe what to plan, or /plan stop to leave"
                            .to_string(),
                    ));
                    return;
                }
                if self.sending {
                    self.queue_command(SlashCommand::Plan(None), "/plan");
                    return;
                }
                if self.set_cursor_mode(true).await {
                    self.notice = Some((
                        MUTED(),
                        "Plan mode — cursor plans read-only and asks you to approve".to_string(),
                    ));
                }
            }
            _ => {
                if self.sending {
                    self.queue_command(SlashCommand::Plan(Some(full.to_string())), "/plan");
                    return;
                }
                if !self.cursor_plan_mode && !self.set_cursor_mode(true).await {
                    return;
                }
                if let Err(e) = self.dispatch_user_message(full.to_string(), None).await {
                    self.notice = Some((ERROR(), e.to_string()));
                }
            }
        }
    }

    /// Switch the cursor session to `plan`/`agent`. Records the desired mode (a
    /// later prewarm/`/new` open re-applies it) and applies it now if a session is
    /// live. `false` only when a live session reports the mode unavailable.
    async fn set_cursor_mode(&mut self, plan: bool) -> bool {
        self.cursor_plan_mode = plan;
        let mode = if plan { "plan" } else { "agent" };
        let Some(session) = self.cursor_acp_session.as_mut() else {
            return true;
        };
        match session.set_mode(mode).await {
            Ok(true) => true,
            Ok(false) => {
                self.cursor_plan_mode = false;
                self.notice = Some((
                    ERROR(),
                    format!("cursor didn't offer a '{mode}' mode for this session"),
                ));
                false
            }
            Err(e) => {
                self.notice = Some((ERROR(), format!("Couldn't switch cursor mode: {e}")));
                true
            }
        }
    }

    /// After a plan-mode turn that ended WITHOUT an approval, stash the agent's
    /// last reply as the drafted plan (for `/plan go`) and frame it as the plan
    /// card. Plan mode and the read-only engine persist. No-op otherwise.
    pub(super) fn capture_plan_draft(&mut self) {
        if !self.plan_mode || self.sending {
            return;
        }
        let plan_at = self.history.iter().rposition(|m| m.role == "assistant");
        let plan = plan_at
            .map(|i| self.history[i].content.clone())
            .unwrap_or_default();
        // An empty reply (all tool calls / interrupted) leaves any prior draft as-is.
        if plan.trim().is_empty() {
            return;
        }
        self.pending_plan = Some(plan);
        self.plan_card_idx = plan_at;
        self.notice = Some((
            MUTED(),
            "Plan drafted — approve on the card or /plan go; keep refining, or /plan stop to leave"
                .to_string(),
        ));
    }

    /// Returns the outgoing session's mid-execution checklist, if any, for the
    /// `/new` handler to offer continuing (that prompt owns the messaging — no
    /// notice for it here). A draft only leaves the `/plan resume` hint.
    pub(super) async fn start_new_chat(&mut self) -> Option<serde_json::Value> {
        // Scan for a leftover plan before the clears below.
        let handoff_steps = (!self.history.is_empty())
            .then(|| self.unfinished_plan_steps())
            .flatten();
        let draft_hint =
            handoff_steps.is_none() && !self.history.is_empty() && self.pending_plan.is_some();
        self.discard_resume_state();
        // The share is pinned to the current session; a new chat swaps it out.
        self.stop_live_share();
        self.cancel_inflight_request(CancelKind::Discard);
        // /new abandons this conversation — persist it first so `/resume` and
        // `/plan resume` can find it (otherwise a mid-turn `/new` orphans the
        // transcript and the executing checklist). Before the
        // `pristine_import_len` reset so a merely-viewed import doesn't fork.
        if !self.history.is_empty() {
            let _ = self.persist_history().await;
        }
        self.overlay = Overlay::None;
        self.pristine_import_len = None;
        self.import_fidelity = None;
        self.history.clear();
        // After the persist above — the next session's file must not inherit these.
        self.vision_descriptions.clear();
        self.agent_turn_indices.clear();
        self.expanded_thinking.clear();
        self.expanded_output.clear();
        self.local_outputs.clear();
        self.reasoning_durations.clear();
        self.turn_durations.clear();
        self.reasoning_started_at = None;
        self.reasoning_elapsed_ms = None;
        self.clear_transcript_selection();
        self.reset_composer();
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.sending = false;
        self.clear_tool_output();
        self.request_started_at = None;
        self.session_id = new_code_session_id();
        // Fresh session file → no planState on disk yet.
        self.plan_state_written.set(false);
        // New session → re-root NEW jobs' logs (running jobs keep their absolute paths).
        self.jobs.set_logs_root(
            self.session_store
                .session_artifacts_dir(&self.session_id)
                .join("jobs"),
        );
        self.format = seeded_chat_format(&self.key, &self.raw_model);
        self.last_usage = None;
        self.context_tokens = 0;
        // Fresh session → fresh token tally (the index entry starts at zero).
        self.session_tokens = crate::services::session_store::SessionTokens::default();
        self.session_cost_usd = 0.0;
        self.context_is_estimate = true;
        self.follow_output = true;
        self.plan_mode = false;
        self.plan_exit_pending = false;
        self.pending_plan = None;
        self.plan_card_idx = None;
        // A leftover continue prompt belongs to the session it was armed in.
        self.cards.plan_continue = None;
        self.notice = draft_hint.then(|| {
            (
                MUTED(),
                "The previous session left an unfinished plan (unapproved draft) — /plan resume continues it here"
                    .to_string(),
            )
        });
        // Drop the cursor session (no context bleed across /new), then
        // re-prewarm so the next message's connect overlaps typing.
        self.cursor_acp_session = None;
        self.cursor_plan_mode = false; // fresh session opens in `agent`
        self.prewarm_cursor_session();
        // Drop the agent engine + serve so a fresh chat starts with no context.
        self.agent_engine = None;
        // A fresh chat must not inherit a resumed session's pending transcript.
        self.pending_agent_messages = None;
        self.cards.clear_agent_cards();
        self.stop_agent_serve();
        handoff_steps
    }

    /// `Unsend` (ESC before anything streamed) puts the cancelled submission back
    /// in the composer; `Discard` (ESC / resume / `/new`) un-sends it instead,
    /// leaving the composer empty — recallable via ↑.
    pub(super) fn cancel_inflight_request(&mut self, kind: CancelKind) {
        let was_sending = self.sending;
        // Cancelling (interrupt path 1, /new, resume, key switch) also exits any
        // autonomous /goal loop, so it can't auto-continue after the dropped turn.
        // The interrupt-with-partial path clears it separately, before this runs.
        self.goal_mode = None;
        self.goal_guard_stop = None;
        // A cancelled /compact must not mark the NEXT turn as a compact.
        self.compact_before = None;
        // Plan mode persists across an interrupt (it's a session mode). Apply a
        // deferred discard-exit opportunistically; if the turn task still holds
        // the lock, the flag stays set and the next dispatch/turn-end applies it.
        if self.plan_exit_pending
            && let Some(session) = self.agent_engine.as_ref()
            && let Ok(mut engine) = session.engine.try_lock()
        {
            engine.set_plan_mode(false);
            self.plan_exit_pending = false;
        }
        // An in-process agent turn is in flight when its per-turn serve is up. The
        // engine has ALREADY consumed this turn (and may have run side-effecting
        // tools — file writes, shell commands), and it keeps its own conversation
        // record. So un-sending the turn from the transcript would hide that work
        // and diverge the display from the engine; keep the user turn instead.
        let was_agent_turn = self.agent_serve.is_some();
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        if was_agent_turn {
            self.finalize_interrupted_checkpoint();
        }
        // Tear down the agent turn's serve and drop any pending permission card
        // (the dropped reply makes the engine's awaiting tool fail closed).
        self.stop_agent_serve();
        self.cards.clear_agent_cards();
        let discarded = self.discard_queued_input();
        if was_sending && let Some(session) = self.cursor_acp_session.as_ref() {
            // Fire-and-forget session/cancel so the agent stops generating
            // even though our task already dropped the prompt stream.
            let client = session.client_handle();
            let sid = session.session_id().to_string();
            tokio::spawn(async move {
                let _ = client
                    .notify("session/cancel", serde_json::json!({"sessionId": sid}))
                    .await;
            });
        }
        if matches!(kind, CancelKind::Unsend) {
            restore_cancelled_submission(
                &mut self.history,
                &mut self.draft,
                &mut self.draft_attachments,
                &mut self.pending_submit,
            );
            // Un-send from the engine too, or the resent (possibly edited) text
            // merges with the stale copy. Async because the aborted turn task may
            // still hold the engine lock; the pending flag re-applies at next
            // dispatch as the ordering backstop.
            if was_agent_turn && let Some(session) = &self.agent_engine {
                self.agent_unsend_pending = true;
                let engine = session.engine.clone();
                tokio::spawn(async move { engine.lock().await.unsend_last_user_turn() });
            }
        } else if was_agent_turn {
            // Keep the user turn in the transcript; just drop the restore buffer so
            // it can't be resurrected by a later non-agent cancel.
            self.pending_submit = None;
        } else {
            self.pending_submit = None;
            if self
                .history
                .last()
                .is_some_and(|message| message.role == "user")
            {
                self.history.pop();
            }
        }
        // Either pop above may have removed a flagged user row.
        self.agent_turn_indices.retain(|&i| i < self.history.len());
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        self.follow_output = true;
        self.notice = Some((MUTED(), with_discarded("Request cancelled", discarded)));
    }

    pub(super) async fn interrupt_inflight_request(&mut self) -> Result<()> {
        // Interrupting ends any autonomous /goal loop (both interrupt paths route
        // through here; the partial-text path below doesn't call
        // `cancel_inflight_request`, so clear it up front for both).
        let goal_was_active = self.goal_mode.take().is_some();
        // Same for /compact (the partial-text path skips `cancel_inflight_request`).
        self.compact_before = None;
        // Reveal any buffered text so the full received reply is kept, and drop
        // a deferred finish — we're committing the partial turn ourselves.
        self.drain_incoming_buffer();
        self.pending_finish = None;
        if self.pending_response.is_empty() {
            // Still "just pending" (nothing streamed, no tool/reasoning row after the
            // user turn) → return the message to the composer. Goal-mode
            // continuations are synthetic, so leave those on the discard path.
            let nothing_produced = !goal_was_active
                && self
                    .history
                    .last()
                    .is_some_and(|message| message.role == "user");
            self.cancel_inflight_request(if nothing_produced {
                CancelKind::Unsend
            } else {
                CancelKind::Discard
            });
            if goal_was_active {
                self.notice = Some((MUTED(), "Goal mode stopped".to_string()));
            }
            return Ok(());
        }

        let was_agent_turn = self.agent_serve.is_some();
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        if was_agent_turn {
            self.finalize_interrupted_checkpoint();
        }
        // Tear down an agent turn's serve / permission card if this was one.
        self.stop_agent_serve();
        self.cards.clear_agent_cards();
        let discarded = self.discard_queued_input();

        let partial = std::mem::take(&mut self.pending_response);
        // Keep the reasoning shown for this partial reply (the user saw it); the
        // empty-response interrupt above already returned without committing.
        let reasoning_content = (!self.pending_reasoning.is_empty())
            .then(|| std::mem::take(&mut self.pending_reasoning));
        self.pending_submit = None;
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.follow_output = true;
        self.history.push(ChatMessage {
            model: self.turn_model.clone(),
            role: "assistant".to_string(),
            content: partial,
            reasoning_content,
            attachments: vec![],
        });
        self.context_tokens = self.estimated_context_used().await;
        self.context_is_estimate = true;
        self.last_usage = None;
        self.persist_history().await?;
        self.notice = Some((
            MUTED(),
            with_discarded(
                if goal_was_active {
                    "Response interrupted — goal mode stopped"
                } else {
                    "Response interrupted"
                },
                discarded,
            ),
        ));
        Ok(())
    }

    /// Drop unconsumed mid-turn input, returning the count so the interrupt
    /// notice can say so instead of losing it silently.
    pub(super) fn discard_queued_input(&mut self) -> usize {
        let count = self.queued_messages.len()
            + self.queued_commands.len()
            + self
                .steering_queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len();
        self.queued_messages.clear();
        self.clear_steering_queue();
        self.queued_commands.clear();
        self.queue_focus = None;
        count
    }

    pub(super) fn record_draft_history(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }
        // Drop consecutive duplicates (shell `ignoredups`), judged against the
        // current dir's view. Non-adjacent repeats are kept.
        if self.draft_history.last().map(String::as_str) != Some(input) {
            self.draft_history.push(input.to_string());
            let overflow = self.draft_history.len().saturating_sub(MAX_DRAFT_HISTORY);
            if overflow > 0 {
                self.draft_history.drain(..overflow);
            }
            // Mirror into the global list, tagged with the launch dir.
            self.draft_history_all.push(DraftHistoryEntry {
                cwd: self.real_cwd.clone(),
                text: input.to_string(),
            });
            let overflow = self
                .draft_history_all
                .len()
                .saturating_sub(MAX_DRAFT_HISTORY_TOTAL);
            if overflow > 0 {
                self.draft_history_all.drain(..overflow);
            }
        }
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    pub(super) fn history_prev(&mut self) {
        if self.draft_history.is_empty() {
            return;
        }

        let next_index = match self.draft_history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.draft_history_stash = Some(self.draft.clone());
                self.draft_history.len().saturating_sub(1)
            }
        };

        self.draft_history_index = Some(next_index);
        self.draft = self.draft_history[next_index].clone();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn history_next(&mut self) {
        let Some(index) = self.draft_history_index else {
            return;
        };

        if index + 1 < self.draft_history.len() {
            let next_index = index + 1;
            self.draft_history_index = Some(next_index);
            self.draft = self.draft_history[next_index].clone();
            self.cursor = self.draft.len();
            self.sync_command_menu_state();
            return;
        }

        self.draft_history_index = None;
        self.draft = self.draft_history_stash.take().unwrap_or_default();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn leave_history_navigation(&mut self) {
        if self.draft_history_index.is_some() && self.draft_history_stash.is_none() {
            self.draft_history_stash = Some(self.draft.clone());
        }
        self.draft_history_index = None;
    }
}

/// Build the (role, content) turns seeded into a freshly (re)built agent engine
/// on resume or a key/model switch. User/assistant text carries over verbatim;
/// tool steps — which `seed_history` can't replay as real tool messages (no call
/// IDs) — are folded into compact assistant notes (`[used read_file: a.rs → 3
/// lines]`) so the engine remembers what it already did instead of going amnesiac
/// about its own prior work. Image attachments (which the seed can't carry as
/// pixels) ride along as their cached description, else a placeholder note.
pub(super) fn agent_seed_turns(
    history: &[ChatMessage],
    descriptions: &std::collections::HashMap<String, String>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < history.len() {
        let m = &history[i];
        match m.role.as_str() {
            "user" | "assistant" if !m.content.trim().is_empty() => {
                let mut content = m.content.clone();
                for a in &m.attachments {
                    if !a.is_image() {
                        continue;
                    }
                    let described = match &a.storage {
                        AttachmentStorage::Inline { data } => {
                            descriptions.get(&crate::services::vision_describe::image_hash(data))
                        }
                        AttachmentStorage::FileRef { .. } => None,
                    };
                    match described {
                        Some(text) => content.push_str(&format!("\n\n{text}")),
                        None => content.push_str(&format!(
                            "\n\n[image attached: {} — not available to this model]",
                            a.name
                        )),
                    }
                }
                out.push((m.role.clone(), content));
            }
            "tool_call" => {
                let (name, args) = decode_tool_call(&m.content);
                let target = tool_call_target(&name, &args);
                let mut note = if target.is_empty() {
                    format!("[used {name}]")
                } else {
                    format!("[used {name}: {target}]")
                };
                // Fold the immediately-following result's first line in as the outcome.
                if let Some(next) = history.get(i + 1).filter(|n| n.role == "tool_result") {
                    let summary: String = next
                        .content
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(120)
                        .collect();
                    if !summary.trim().is_empty() {
                        note.push_str(&format!(" → {}", summary.trim()));
                    }
                    i += 1; // consume the result
                }
                out.push(("assistant".to_string(), note));
            }
            _ => {} // plan / other display-only roles aren't seeded
        }
        i += 1;
    }
    out
}

/// One-line, length-capped excerpt of a turn's prompt for the `/rewind` picker.
fn rewind_excerpt(text: &str) -> String {
    const MAX: usize = 56;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.is_empty() {
        "(no text)".to_string()
    } else if line.chars().count() > MAX {
        let mut s: String = line.chars().take(MAX - 1).collect();
        s.push('…');
        s
    } else {
        line.to_string()
    }
}

/// Trailing annotation on a `/rewind` picker row: a "conversation only" marker
/// when no checkpoint from the turn onward has a tree snapshot (file revert
/// unavailable); otherwise empty (the exact file impact is reported in the
/// notice after applying).
fn rewind_label_suffix(conversation_only: bool) -> String {
    if conversation_only {
        "  · conversation only".to_string()
    } else {
        String::new()
    }
}

/// Human-readable notice summarizing what a completed `/rewind` did.
fn rewind_notice(outcome: &RewindOutcome) -> String {
    if let Some(err) = &outcome.error {
        return format!("Rewound the conversation; file revert failed: {err}");
    }
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let mut parts = Vec::new();
    if outcome.restored > 0 {
        parts.push(format!(
            "restored {} file{}",
            outcome.restored,
            plural(outcome.restored)
        ));
    }
    if outcome.deleted > 0 {
        parts.push(format!(
            "removed {} file{}",
            outcome.deleted,
            plural(outcome.deleted)
        ));
    }
    if parts.is_empty() {
        "Rewound".to_string()
    } else {
        format!("Rewound — {}", parts.join(", "))
    }
}

/// Start line of each diff pair, found from the pre-edit `old` text — so this
/// must run before the edit applies. `None` when the file can't be read or `old`
/// isn't unique (a wrong number is worse than none).
fn compute_line_starts(
    cwd: &std::path::Path,
    name: &str,
    args: &serde_json::Value,
) -> Vec<Option<usize>> {
    let diffs = edit_diffs(name, args);
    if diffs.is_empty() {
        return vec![];
    }
    let mut cache: std::collections::HashMap<String, Option<String>> =
        std::collections::HashMap::new();
    diffs
        .iter()
        .map(|d| {
            // A pure insertion (apply_patch "Add File") begins the file at line 1.
            if d.old.is_empty() {
                return (!d.new.is_empty()).then_some(1);
            }
            let content = cache
                .entry(d.path.clone())
                .or_insert_with(|| {
                    let full = crate::agent::tools::resolve(cwd, &d.path);
                    std::fs::metadata(&full).ok().filter(|m| m.is_file())?;
                    std::fs::read_to_string(full).ok()
                })
                .as_ref()?;
            let mut hits = content.match_indices(&d.old);
            let offset = hits.next()?.0;
            // Ambiguous match → don't risk numbering the wrong site.
            if hits.next().is_some() {
                return None;
            }
            Some(1 + content[..offset].matches('\n').count())
        })
        .collect()
}

/// Interrupt notice + how many queued messages it threw away.
fn with_discarded(base: &str, discarded: usize) -> String {
    match discarded {
        0 => base.to_string(),
        1 => format!("{base} — 1 queued message discarded"),
        n => format!("{base} — {n} queued messages discarded"),
    }
}

/// A `write_file`'s pre-write snapshot for the transcript diff card, captured
/// before the write applies (by commit time the old content is gone). `None`
/// (missing/non-UTF8/oversized) keeps the all-additions card; the byte cap also
/// bounds what the session file persists per write.
fn capture_pre_write(
    cwd: &std::path::Path,
    name: &str,
    args: &serde_json::Value,
) -> Option<String> {
    const MAX_BYTES: u64 = 256 * 1024;
    if name != "write_file" {
        return None;
    }
    let path = args.get("path").and_then(|v| v.as_str())?;
    let abs = crate::agent::tools::resolve(cwd, path);
    // `is_file`: a FIFO reports len 0 but would block the read.
    let meta = std::fs::metadata(&abs).ok()?;
    if !meta.is_file() || meta.len() > MAX_BYTES {
        return None;
    }
    std::fs::read_to_string(abs).ok()
}

/// Bridges the in-process `AgentEngine` to the chat TUI: engine callbacks become
/// `RuntimeEvent`s the event loop renders, and a permission request round-trips
/// through the loop's permission card via a oneshot.
struct ChatAgentUi {
    tx: UnboundedSender<RuntimeEvent>,
    /// Workspace root, for resolving an edit's `path` in the pre-edit probe.
    cwd: std::path::PathBuf,
    steering: SteeringQueue,
}

/// Bridges parallel sub-agent progress to the event loop's per-delegate rows.
struct ChatSubagentSink {
    tx: UnboundedSender<RuntimeEvent>,
}

impl crate::agent::engine::SubagentSink for ChatSubagentSink {
    fn begin(&self, labels: &[String]) {
        self.tx
            .send(RuntimeEvent::AgentSubBegin {
                labels: labels.to_vec(),
            })
            .ok();
    }

    fn activity(
        &self,
        slot: usize,
        agent: &str,
        tool: &str,
        args: &serde_json::Value,
        step: usize,
    ) {
        self.tx
            .send(RuntimeEvent::AgentSubSlot {
                slot,
                agent: agent.to_string(),
                tool: tool.to_string(),
                args: args.clone(),
                step,
            })
            .ok();
    }

    fn denied(&self, slot: usize, tool: &str) {
        self.tx
            .send(RuntimeEvent::AgentSubDenied {
                slot,
                tool: tool.to_string(),
            })
            .ok();
    }

    fn tokens(&self, slot: usize, output: u64) {
        self.tx
            .send(RuntimeEvent::AgentSubTokens {
                slot,
                tokens: output,
            })
            .ok();
    }

    fn done(&self, slot: usize, ok: bool, steps: usize, tokens: u64) {
        self.tx
            .send(RuntimeEvent::AgentSubDone {
                slot,
                ok,
                steps,
                tokens,
            })
            .ok();
    }

    fn finish(&self) {
        self.tx.send(RuntimeEvent::AgentSubFinish).ok();
    }
}

impl crate::agent::engine::AgentUi for ChatAgentUi {
    fn assistant_text(&mut self, delta: &str) {
        self.tx
            .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
                delta.to_string(),
            )))
            .ok();
    }

    fn assistant_reasoning(&mut self, delta: &str) {
        self.tx
            .send(RuntimeEvent::Delta(ChatResponseChunk::Reasoning(
                delta.to_string(),
            )))
            .ok();
    }

    fn discard_streamed_segment(&mut self) {
        self.tx.send(RuntimeEvent::AgentDiscardSegment).ok();
    }

    fn drain_steering(&mut self) -> Vec<String> {
        let drained: Vec<String> = self
            .steering
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect();
        // Same channel as tool results, so the transcript commit lands in order.
        for text in &drained {
            self.tx.send(RuntimeEvent::AgentSteered(text.clone())).ok();
        }
        drained
    }

    fn context_usage(&mut self, tokens: u64, measured: bool) {
        self.tx
            .send(RuntimeEvent::AgentContext { tokens, measured })
            .ok();
    }

    fn turn_tokens(&mut self, output: u64) {
        self.tx.send(RuntimeEvent::AgentTurnTokens(output)).ok();
    }

    fn subagent_activity(
        &mut self,
        agent: &str,
        tool: &str,
        args: &serde_json::Value,
        step: usize,
    ) {
        self.tx
            .send(RuntimeEvent::AgentSubActivity {
                agent: agent.to_string(),
                tool: tool.to_string(),
                args: args.clone(),
                step,
            })
            .ok();
    }

    fn subagent_sink(&mut self) -> Option<std::sync::Arc<dyn crate::agent::engine::SubagentSink>> {
        Some(std::sync::Arc::new(ChatSubagentSink {
            tx: self.tx.clone(),
        }))
    }

    fn plan_updated(&mut self, items: &[crate::agent::plan::PlanItem]) {
        let value = serde_json::to_value(items).unwrap_or(serde_json::Value::Null);
        self.tx.send(RuntimeEvent::AgentPlan(value)).ok();
    }

    fn tool_start(&mut self, name: &str, args: &serde_json::Value) {
        // Runs on the engine thread before the edit applies, so the probe sees
        // the pre-edit file.
        let line_starts = compute_line_starts(&self.cwd, name, args);
        let old_content = capture_pre_write(&self.cwd, name, args);
        self.tx
            .send(RuntimeEvent::AgentToolCall {
                id: None,
                name: name.to_string(),
                args: args.clone(),
                line_starts,
                old_content,
            })
            .ok();
    }

    fn tool_output(&mut self, _name: &str, chunk: &str) {
        self.tx
            .send(RuntimeEvent::AgentToolOutput {
                chunk: chunk.to_string(),
            })
            .ok();
    }

    fn tool_result(&mut self, _name: &str, result: &Result<String, String>) {
        let content = match result {
            Ok(s) => s.clone(),
            Err(e) => format!("error: {e}"),
        };
        self.tx.send(RuntimeEvent::AgentToolResult { content }).ok();
    }

    fn notify(&mut self, text: &str) {
        self.tx
            .send(RuntimeEvent::AgentNotice(text.to_string()))
            .ok();
    }

    fn notify_error(&mut self, text: &str) {
        self.tx
            .send(RuntimeEvent::AgentError(text.to_string()))
            .ok();
    }

    fn turn_stopped(&mut self, stop: crate::agent::engine::TurnStop) {
        self.tx.send(RuntimeEvent::AgentTurnStop(stop)).ok();
    }

    fn footer(
        &mut self,
        _summary: Option<&str>,
        steps: usize,
        tokens: u64,
        context_tokens: u64,
        _elapsed: u64,
    ) {
        self.tx
            .send(RuntimeEvent::AgentFinished {
                steps,
                tokens,
                context_tokens,
            })
            .ok();
    }

    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        preview: Option<&'a str>,
        once_only: bool,
    ) -> futures::future::BoxFuture<'a, Decision> {
        let tx = self.tx.clone();
        let tool = tool.to_string();
        let preview = preview.map(str::to_string);
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentPermission {
                    tool,
                    preview,
                    once_only,
                    reply,
                })
                .is_err()
            {
                return Decision::Deny;
            }
            rx.await.unwrap_or(Decision::Deny)
        })
    }

    fn switch_chat_model<'a>(
        &'a mut self,
        model: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<String, String>> {
        let tx = self.tx.clone();
        let model = model.to_string();
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentSwitchModel { model, reply })
                .is_err()
            {
                return Err("session is no longer running".to_string());
            }
            rx.await
                .unwrap_or_else(|_| Err("session is no longer running".to_string()))
        })
    }

    fn set_chat_effort<'a>(
        &'a mut self,
        level: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<String, String>> {
        let tx = self.tx.clone();
        let level = level.to_string();
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentSetEffort { level, reply })
                .is_err()
            {
                return Err("session is no longer running".to_string());
            }
            rx.await
                .unwrap_or_else(|_| Err("session is no longer running".to_string()))
        })
    }

    fn ask_user<'a>(
        &'a mut self,
        question: &'a str,
        options: &'a [crate::agent::ask::AskOption],
        allow_free_text: bool,
        multi_select: bool,
    ) -> futures::future::BoxFuture<'a, Result<String, String>> {
        let tx = self.tx.clone();
        let question = question.to_string();
        let options = options.to_vec();
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentAskUser {
                    question,
                    options,
                    allow_free_text,
                    multi_select,
                    reply,
                })
                .is_err()
            {
                return Err("session is no longer running".to_string());
            }
            // A dropped card (interrupt / session end) reads as a dismissal.
            rx.await
                .unwrap_or_else(|_| Err(crate::agent::ask::DISMISSED_DIRECTIVE.to_string()))
        })
    }

    fn approve_plan<'a>(
        &'a mut self,
        plan: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<crate::agent::protocol::PlanDecision, String>> {
        let tx = self.tx.clone();
        let plan = plan.to_string();
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentPlanApproval { plan, reply })
                .is_err()
            {
                return Err("session is no longer running".to_string());
            }
            // A dropped card (interrupt / session end) reads as a dismissal.
            rx.await.unwrap_or_else(|_| {
                Err(crate::agent::plan_mode::PLAN_APPROVAL_DISMISSED.to_string())
            })
        })
    }

    fn review_edits<'a>(
        &'a mut self,
        items: &'a [crate::agent::review::ReviewItem],
    ) -> futures::future::BoxFuture<'a, crate::agent::review::ReviewDecision> {
        let tx = self.tx.clone();
        let items = items.to_vec();
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentReviewEdits { items, reply })
                .is_err()
            {
                return crate::agent::review::ReviewDecision::Reject;
            }
            // A dropped card (interrupt / session end) reads as a rejection (fail-closed).
            rx.await
                .unwrap_or(crate::agent::review::ReviewDecision::Reject)
        })
    }
}

/// Open a cursor ACP session with the TUI's standard options. Shared by
/// `spawn_cursor_turn`'s cold-open and `prewarm_cursor_session`.
async fn open_cursor_session(
    key: ApiKey,
    requested_model: Option<String>,
    cwd: String,
    auto_approve: std::sync::Arc<std::sync::atomic::AtomicBool>,
    permission_prompt: cursor_acp::CursorPermissionPrompt,
    hooks: cursor_acp::CursorInteractionHooks,
) -> Result<CursorAcpSession> {
    CursorAcpSession::open_with_options(
        &key,
        requested_model.as_deref(),
        &cwd,
        None,
        cursor_acp::ModelPickPreference::PreferNoThinking,
        Some(auto_approve),
        Some(permission_prompt),
        hooks,
    )
    .await
}

/// Bounded retry-with-backoff around [`open_cursor_session`]. cursor-agent
/// already retries its own live HTTP/2 link (the "keepalive ping timed out …
/// retry" you see in its TUI), so aivo only sees a hard failure when that's
/// exhausted or the child dies during connect — a single miss on a flaky link
/// (e.g. a VPN/proxy tunnel) would otherwise kill the turn outright. Retry the
/// *connect* (idempotent, no conversation to lose yet) a few times with backoff,
/// emitting a `retrying` notice through `tx` when present so the reconnect is
/// visible (matching the native engine's retry UX). Permanent failures — missing
/// binary, legacy key — return immediately; retrying them can't help.
async fn open_cursor_session_with_retry(
    key: ApiKey,
    requested_model: Option<String>,
    cwd: String,
    auto_approve: std::sync::Arc<std::sync::atomic::AtomicBool>,
    permission_prompt: cursor_acp::CursorPermissionPrompt,
    hooks: cursor_acp::CursorInteractionHooks,
    tx: Option<UnboundedSender<RuntimeEvent>>,
) -> Result<CursorAcpSession> {
    const MAX_ATTEMPTS: u32 = 3;
    let mut attempt = 1;
    loop {
        let result = open_cursor_session(
            key.clone(),
            requested_model.clone(),
            cwd.clone(),
            auto_approve.clone(),
            permission_prompt.clone(),
            hooks.clone(),
        )
        .await;
        match result {
            Ok(session) => return Ok(session),
            Err(e) if attempt >= MAX_ATTEMPTS || is_permanent_cursor_open_error(&e) => {
                return Err(e);
            }
            Err(_) => {
                // "retrying" is the marker the event loop keys on to show the
                // reconnect state (and to auto-clear the notice on recovery).
                if let Some(tx) = &tx {
                    tx.send(RuntimeEvent::AgentNotice(format!(
                        "Cursor connection failed — retrying ({}/{MAX_ATTEMPTS})…",
                        attempt + 1
                    )))
                    .ok();
                }
                // 400ms, 800ms: long enough to ride out a brief tunnel hiccup,
                // short enough to stay responsive.
                let backoff = std::time::Duration::from_millis(400u64 << (attempt - 1));
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
        }
    }
}

/// Whether a cursor open error is permanent — retrying it is pointless because
/// the cause won't clear on its own. Everything else (spawn, `initialize`,
/// `session/new`) can be a transient network/handshake failure worth another
/// attempt. Matches the full error chain so wrapping context doesn't hide the
/// root marker.
fn is_permanent_cursor_open_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("was not found on PATH") || msg.contains("predates per-account isolation")
}

/// Compose cursor's structured plan (title/overview/body + todo & phase
/// checklists) into a markdown body for the approval card.
fn render_cursor_plan_markdown(req: &cursor_acp::CursorPlanRequest) -> String {
    let mut out = String::new();
    if let Some(name) = &req.name {
        out.push_str(&format!("# {name}\n\n"));
    }
    if let Some(overview) = &req.overview {
        out.push_str(&format!("{overview}\n\n"));
    }
    if !req.plan.trim().is_empty() {
        out.push_str(req.plan.trim());
        out.push_str("\n\n");
    }
    let checklist = |todos: &[cursor_acp::CursorTodo]| -> String {
        todos
            .iter()
            .map(|t| format!("- {} {}\n", todo_checkbox(&t.status), t.content))
            .collect()
    };
    if !req.todos.is_empty() {
        out.push_str(&checklist(&req.todos));
        out.push('\n');
    }
    for phase in &req.phases {
        out.push_str(&format!("## {}\n\n", phase.name));
        out.push_str(&checklist(&phase.todos));
        out.push('\n');
    }
    if out.trim().is_empty() {
        out.push_str("(cursor proposed a plan with no details)");
    }
    out.trim_end().to_string()
}

fn todo_checkbox(status: &str) -> &'static str {
    match map_cursor_todo_status(status) {
        "in_progress" => "[~]",
        "completed" => "[x]",
        _ => "[ ]",
    }
}

/// Map a cursor todo status to aivo's 3-state plan status; `cancelled`→`completed`.
fn map_cursor_todo_status(status: &str) -> &'static str {
    match status {
        "in_progress" => "in_progress",
        "completed" | "cancelled" => "completed",
        _ => "pending",
    }
}

/// Map an ask-card answer (one label, or labels joined by ", " when multi) back
/// to cursor option ids by exact label match.
fn select_ids_for_answer(
    answer: &str,
    options: &[(String, String)],
    multi_select: bool,
) -> Vec<String> {
    let parts: Vec<&str> = if multi_select {
        answer.split(", ").collect()
    } else {
        vec![answer]
    };
    parts
        .into_iter()
        .filter_map(|part| {
            let part = part.trim();
            options
                .iter()
                .find(|(_, label)| label == part)
                .map(|(id, _)| id.clone())
        })
        .collect()
}

async fn drive_cursor_turn(
    client: std::sync::Arc<crate::services::acp_client::AcpClient>,
    session_id: String,
    model_id: Option<String>,
    user_input: String,
    attachments: Vec<MessageAttachment>,
    tx: &UnboundedSender<RuntimeEvent>,
) -> Result<ChatTurnResult> {
    let blocks = cursor_acp::build_prompt_blocks(&user_input, &attachments)?;
    let mut stream = client.start_prompt(&session_id, blocks).await?;

    let mut turn_result = CursorTurnResult::default();
    let mut reasoning_buf = String::new();
    let mut forward = |chunk: CursorChunk<'_>| -> Result<()> {
        // Tool calls reuse the in-process agent's tool-call card (the renderer
        // coalesces runs); text/reasoning stream as deltas.
        let event = match chunk {
            CursorChunk::Content(t) => {
                RuntimeEvent::Delta(ChatResponseChunk::Content(t.to_string()))
            }
            CursorChunk::Reasoning(t) => {
                RuntimeEvent::Delta(ChatResponseChunk::Reasoning(t.to_string()))
            }
            CursorChunk::ToolCall { id, name, args } => RuntimeEvent::AgentToolCall {
                id,
                name,
                args,
                // Cursor edits carry no file offset to number.
                line_starts: vec![],
                old_content: None,
            },
            CursorChunk::ToolUpdate {
                id,
                args,
                result,
                failed,
            } => RuntimeEvent::AgentToolUpdate {
                id,
                args,
                result,
                failed,
            },
        };
        tx.send(event).ok();
        Ok(())
    };

    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                cursor_acp::consume_session_update(
                    &value,
                    &mut turn_result,
                    &mut reasoning_buf,
                    &mut forward,
                )?;
            }
            PromptEvent::Done(result) => {
                result
                    .map_err(|e| anyhow::anyhow!(e))
                    .context("cursor-agent ACP session/prompt failed")?;
                break;
            }
        }
    }
    // `reasoning_buf` is required by `consume_session_update`'s signature, but the
    // chat TUI doesn't read it: cursor reasoning reaches the UI live via
    // `CursorChunk::Reasoning` → `pending_reasoning` (committed at turn finish like
    // every other provider), so there's no `reasoning_content` on `ChatTurnResult`
    // to populate here.
    let _ = &reasoning_buf;

    Ok(ChatTurnResult {
        content: turn_result.content,
        usage: None,
        model: model_id,
        raw_body: None,
    })
}

#[cfg(test)]
mod ask_id_tests {
    use super::{map_cursor_todo_status, render_cursor_plan_markdown, select_ids_for_answer};
    use crate::services::cursor_acp::{CursorPlanPhase, CursorPlanRequest, CursorTodo};

    #[test]
    fn cursor_plan_markdown_includes_title_todos_and_phases() {
        let req = CursorPlanRequest {
            name: Some("Do the thing".into()),
            overview: Some("A short overview".into()),
            plan: "The plan body.".into(),
            todos: vec![CursorTodo {
                id: "t1".into(),
                content: "first step".into(),
                status: "in_progress".into(),
            }],
            phases: vec![CursorPlanPhase {
                name: "Phase 1".into(),
                todos: vec![CursorTodo {
                    id: "p1".into(),
                    content: "phase step".into(),
                    status: "completed".into(),
                }],
            }],
        };
        let md = render_cursor_plan_markdown(&req);
        assert!(md.contains("# Do the thing"));
        assert!(md.contains("A short overview"));
        assert!(md.contains("The plan body."));
        assert!(md.contains("- [~] first step")); // in_progress glyph
        assert!(md.contains("## Phase 1"));
        assert!(md.contains("- [x] phase step")); // completed glyph
    }

    #[test]
    fn cursor_plan_markdown_handles_empty_plan() {
        let req = CursorPlanRequest {
            name: None,
            overview: None,
            plan: String::new(),
            todos: vec![],
            phases: vec![],
        };
        assert!(render_cursor_plan_markdown(&req).contains("no details"));
    }

    #[test]
    fn cursor_todo_status_maps_to_three_state_plan() {
        assert_eq!(map_cursor_todo_status("pending"), "pending");
        assert_eq!(map_cursor_todo_status("in_progress"), "in_progress");
        assert_eq!(map_cursor_todo_status("completed"), "completed");
        // cancelled folds into completed; unknown → pending.
        assert_eq!(map_cursor_todo_status("cancelled"), "completed");
        assert_eq!(map_cursor_todo_status("weird"), "pending");
    }

    fn opts() -> Vec<(String, String)> {
        vec![
            ("o1".into(), "Just this file".into()),
            ("o2".into(), "Whole crate".into()),
            ("o3".into(), "Everything".into()),
        ]
    }

    #[test]
    fn single_select_maps_label_to_one_id() {
        assert_eq!(
            select_ids_for_answer("Whole crate", &opts(), false),
            vec!["o2"]
        );
    }

    #[test]
    fn multi_select_splits_and_maps_each_label() {
        assert_eq!(
            select_ids_for_answer("Just this file, Everything", &opts(), true),
            vec!["o1", "o3"]
        );
    }

    #[test]
    fn unmatched_labels_are_dropped() {
        // "none" (empty multi-select sentinel) and typos map to no id.
        assert!(select_ids_for_answer("none", &opts(), true).is_empty());
        assert!(select_ids_for_answer("nonexistent", &opts(), false).is_empty());
    }
}

#[cfg(test)]
mod cursor_open_retry_tests {
    use super::is_permanent_cursor_open_error;

    #[test]
    fn missing_binary_is_permanent() {
        let err = anyhow::anyhow!(
            "`cursor-agent` was not found on PATH. Install Cursor CLI support first."
        );
        assert!(is_permanent_cursor_open_error(&err));
    }

    #[test]
    fn legacy_key_is_permanent() {
        let err = anyhow::anyhow!("This cursor key predates per-account isolation. Run …");
        assert!(is_permanent_cursor_open_error(&err));
    }

    #[test]
    fn permanent_marker_survives_context_wrapping() {
        // The marker must be found even when nested under added `.context(...)`.
        let err = anyhow::anyhow!("`cursor-agent` was not found on PATH.")
            .context("opening cursor session");
        assert!(is_permanent_cursor_open_error(&err));
    }

    #[test]
    fn handshake_and_network_failures_are_transient() {
        // These are the ones worth retrying — a flaky link at connect.
        for msg in [
            "cursor-agent ACP initialize failed",
            "cursor-agent ACP session/new failed — check the cursor key with `aivo keys reauth abc`",
            "ACP child stdout closed",
            "connection reset by peer",
        ] {
            let err = anyhow::anyhow!(msg.to_string());
            assert!(
                !is_permanent_cursor_open_error(&err),
                "should retry transient: {msg}"
            );
        }
    }
}
