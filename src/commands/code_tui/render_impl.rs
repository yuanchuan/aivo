use super::*;

/// The text width a markdown table is laid out to fit, given the full transcript
/// column width: drop the accent gutter. The transcript reserves no scrollbar
/// column, so the full remaining width is available in every case.
fn table_layout_width(area_width: u16) -> u16 {
    area_width.saturating_sub(ACCENT_GUTTER_WIDTH)
}

/// Replace every control-char cell (tab, ESC, …) with a space, keeping its
/// style — a raw `\t` (unicode-width 1) desyncs the terminal's cell grid. Run on
/// the finished frame so no widget can poison the grid, whatever its source.
fn scrub_control_cells(buffer: &mut ratatui::buffer::Buffer) {
    for cell in &mut buffer.content {
        if cell.symbol().chars().any(char::is_control) {
            cell.set_symbol(" ");
        }
    }
}

/// The inner content rect of a centered overlay — mirrors the `Margin` every
/// overlay insets its body by, so the screen selection can be confined to it.
fn overlay_content_rect(area: Rect) -> Rect {
    area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 2,
    })
}

/// Draw a `↓ Jump to bottom` pill — dark text on a light chip — centered on the
/// bottom row of `area`. Returns its rect, or `None` when even the short form won't
/// fit; falls back to a compact label on a narrow transcript.
fn render_jump_to_bottom(frame: &mut Frame<'_>, area: Rect) -> Option<Rect> {
    if area.height == 0 {
        return None;
    }
    // A warm off-white chip with dark ink (brand --text-primary on dark) reads
    // cleanly against the warm-dark transcript.
    let style = Style::default()
        .fg(Color::Rgb(26, 23, 18))
        .bg(Color::Rgb(231, 227, 219));
    let label = [" ↓ Jump to bottom ", " ↓ bottom "]
        .into_iter()
        .find(|l| l.chars().count() as u16 + 2 <= area.width)?;
    let width = label.chars().count() as u16;
    let rect = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height - 1,
        width,
        height: 1,
    };
    frame.render_widget(Paragraph::new(Span::styled(label, style)), rect);
    Some(rect)
}

impl CodeTuiApp {
    pub(super) fn is_transcript_empty(&self) -> bool {
        self.history.is_empty()
            && self.pending_response.is_empty()
            && self.incoming_buffer.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
            && self.local_command.is_none()
    }

    /// The full transcript including the live spinner status line. The body is
    /// memoized across frames (see [`build_transcript_body`]); the spinner is
    /// volatile so it is appended fresh here. Used directly by tests and as the
    /// single source of truth for the cached render path.
    pub(super) fn build_transcript(&self) -> RenderedTranscript {
        let body = self.build_transcript_body();
        let mut lines = body.lines;
        let mut bar_colors = body.bar_colors;
        self.append_spinner_status(&mut lines, &mut bar_colors);
        RenderedTranscript::new(lines, bar_colors)
    }

    /// The transcript body: intro, history, the streamed reply, and any notice —
    /// everything except the per-frame spinner status line. Composed from the
    /// memoized history prefix plus the volatile tail; the result is byte-for-byte
    /// what the single-pass build produced, so tests and `max_scroll` are
    /// unaffected. The render path uses the two pieces separately (cached prefix +
    /// fresh tail) so a growing stream never re-renders the whole history.
    pub(super) fn build_transcript_body(&self) -> RenderedTranscript {
        // `transcript_width` is the last-rendered text-area width (already gutter-
        // adjusted) — the width tables should fit into.
        let text_width = self.transcript_width;
        let body = self.build_transcript_history_body(text_width);
        let (tail_lines, tail_bars) = self.volatile_tail_blocks(text_width);
        if tail_lines.is_empty() {
            return body;
        }
        let mut lines = body.lines;
        let mut bars = body.bar_colors;
        // The prefix is already compacted (no trailing blank) and the tail leads
        // with exactly one spacing blank, so the concatenation is canonical —
        // identical to the old single-pass `compact` over the whole body.
        lines.extend(tail_lines);
        bars.extend(tail_bars);
        RenderedTranscript::new(lines, bars)
    }

    /// The memoizable transcript prefix: intro + committed history, with no
    /// dependence on the live stream or notice. This is the expensive part
    /// (markdown parsing, tool decoding) and what
    /// [`ensure_transcript_cache`](Self::ensure_transcript_cache) caches — so it
    /// is rebuilt at most once per *history* change, never per streamed token.
    pub(super) fn build_transcript_history_body(&self, text_width: u16) -> RenderedTranscript {
        let mut lines = Vec::new();
        // Bar color per logical line, kept in lockstep with `lines`. Chrome
        // (intro, spacing) is `None`; each message block paints its role color.
        let mut bars: Vec<Option<Color>> = Vec::new();
        let mut previous_role: Option<&str> = None;
        // Last stamped assistant model; unstamped (pre-feature) turns don't reset it.
        let mut previous_model: Option<&str> = None;

        if self.is_transcript_empty() {
            push_styled_line(&mut lines, "", Style::default());
            bars.push(None);
            return RenderedTranscript::new(lines, bars);
        }

        push_transcript_intro(&mut lines, text_width);
        // Tip stays pinned above the conversation; frozen once non-empty, so safe
        // to memoize.
        lines.extend(self.welcome_status_lines());
        push_message_spacing(&mut lines);
        bars.resize(lines.len(), None);

        // The agent's working dir — tool paths render relative to it (the footer
        // already shows the cwd, so absolute paths are just noise that hides the
        // basename). Falls back to the chat sandbox when the real dir is unknown.
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.as_str()
        } else {
            self.real_cwd.as_str()
        };

        // In-process emits separate `tool_result` lines (which carry per-call
        // targets); cursor enriches the call in place. With separate results, a
        // coalesced call line drops its target list to avoid repeating it.
        let separate_results = self.history.iter().any(|m| m.role == "tool_result");

        // Hide the trailing `tool_call` run while any of it is in flight — the
        // status line (and batch rows) names the work instead. Cursor resolves
        // entries out of order, so the cards wait for the whole run.
        let mut render_len = self.history.len();
        if self.sending {
            let mut start = render_len;
            while start > 0 && self.history[start - 1].role == "tool_call" {
                start -= 1;
            }
            let live = self.history[start..render_len].iter().any(|m| {
                let (result, failed) = decode_tool_outcome(&m.content);
                result.is_none() && !failed
            });
            if live {
                render_len = start;
            }
        }

        let mut idx = 0;
        while idx < render_len {
            let message = &self.history[idx];
            // The plan/task list is pinned in its own panel above the composer
            // (see `render_plan_panel`), not rendered inline — so it stays visible
            // instead of scrolling away under later tool calls. Skip it here
            // without touching `previous_role`, so spacing reads as if it weren't
            // present.
            if message.role == "plan" {
                idx += 1;
                continue;
            }
            if should_add_message_spacing(previous_role, message.role.as_str()) {
                push_message_spacing(&mut lines);
                bars.resize(lines.len(), None);
            }
            // Model-switch divider; mirrors the share viewer's phase divider.
            if message.role == "assistant"
                && let Some(model) = message.model.as_deref()
            {
                if previous_model.is_some_and(|prev| prev != model) {
                    push_styled_line(
                        &mut lines,
                        format!("model → {model}"),
                        Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                    );
                    bars.push(None);
                    push_styled_line(&mut lines, String::new(), Style::default());
                    bars.push(None);
                }
                previous_model = Some(model);
            }
            let mut block = Vec::new();
            let mut advance = 1;
            match message.role.as_str() {
                "user" => {
                    // A skill invocation stores the full expanded SKILL.md as the
                    // user message (the model needs it), but the transcript should
                    // show the compact `/name args` the user actually typed.
                    let label = super::skill_invocation_label(&message.content);
                    let shown = label.as_deref().unwrap_or(&message.content);
                    render_user_message(&mut block, shown, &message.attachments);
                }
                "assistant" => {
                    let reasoning = self
                        .thinking_enabled
                        .then_some(message.reasoning_content.as_deref())
                        .flatten();
                    // Windowed by default; expanded only for turns the user clicked.
                    let expanded = self.expanded_thinking.contains(&idx);
                    let view = reasoning.map(|text| ReasoningView { text, expanded });
                    if self.plan_card_idx == Some(idx) {
                        push_plan_card(
                            &mut lines,
                            &mut bars,
                            view,
                            &message.content,
                            text_width,
                            Some("approve on the card · /plan go [guidance] · /plan stop to leave"),
                        );
                    } else {
                        push_assistant_blocks(
                            &mut lines,
                            &mut bars,
                            view,
                            &message.content,
                            text_width,
                            role_bar_color("assistant"),
                        );
                    }
                }
                "tool_call" => {
                    let (name, args) = decode_tool_call(&message.content);
                    // Render the plan payload as a card, not an opaque tool row.
                    if name == "exit_plan_mode" {
                        let plan = args.get("plan").and_then(|v| v.as_str()).unwrap_or("");
                        push_plan_card(&mut lines, &mut bars, None, plan, text_width, None);
                        previous_role = Some(message.role.as_str());
                        idx += 1;
                        continue;
                    }
                    // Coalesce a run of adjacent same-verb calls into one line (see
                    // `tool_group_key`). Subagents are the exception — each is a
                    // distinct unit of work, so render it on its own line with its
                    // task visible, never an opaque `subagent ×N`.
                    let run = if name == "subagent" {
                        1
                    } else {
                        self.tool_call_run_len(idx, &name)
                    };
                    // Don't coalesce into the hidden in-flight tail.
                    let run = run.min(render_len - idx);
                    if run >= 2 {
                        let targets: Vec<String> = self.history[idx..idx + run]
                            .iter()
                            .map(|m| {
                                let (n, a) = decode_tool_call(&m.content);
                                let target = tool_call_target_display(&n, &a, cwd);
                                // cursor gives no path/pattern, so show the per-call
                                // result (e.g. `18 matches`) in the detail slot.
                                if target.is_empty() {
                                    decode_tool_outcome(&m.content).0.unwrap_or_default()
                                } else {
                                    target
                                }
                            })
                            .collect();
                        let failed = self.history[idx..idx + run]
                            .iter()
                            .filter(|m| decode_tool_outcome(&m.content).1)
                            .count();
                        let header_targets: &[String] =
                            if separate_results { &[] } else { &targets };
                        render_tool_call_group(&mut block, &name, run, header_targets, failed);
                        advance = run;
                    } else {
                        let (result, failed) = decode_tool_outcome(&message.content);
                        let line_starts = decode_line_starts(&message.content);
                        let old_content = decode_old_content(&message.content);
                        render_tool_call(
                            &mut block,
                            &name,
                            &args,
                            result.as_deref(),
                            failed,
                            cwd,
                            &line_starts,
                            old_content.as_deref(),
                        );
                    }
                }
                "tool_result" => {
                    // `tool` fixes the unit (files/entries/matches); a detached
                    // call's target tags the result (see `tool_result_source`).
                    let (tool, label) = match self.tool_result_source(idx, cwd) {
                        Some((name, target, detached)) => (
                            Some(name),
                            detached.then_some(target).filter(|t| !t.is_empty()),
                        ),
                        None => (None, None),
                    };
                    render_tool_result(
                        &mut block,
                        &message.content,
                        cwd,
                        tool.as_deref(),
                        label.as_deref(),
                        self.expanded_output.contains(&idx),
                    );
                }
                "local_command" => {
                    // Expanded renders the in-memory output (persisted preview after a
                    // resume) in place; folded shows the preview + clickable expander.
                    let view = if self.expanded_output.contains(&idx) {
                        OutputView::Expanded {
                            full: self.local_outputs.get(&idx),
                        }
                    } else {
                        OutputView::Collapsed
                    };
                    render_local_command(&mut block, &message.content, view);
                }
                "plan" => render_plan(&mut block, &message.content),
                "error" => render_error_message(&mut block, &message.content),
                other => render_system_message(&mut block, other, &message.content, text_width),
            }
            let bar = role_bar_color(message.role.as_str());
            push_block(&mut lines, &mut bars, block, Some(bar));
            // The `✶ Done in …` marker for a turn stamped on its last entry (which
            // may sit inside a coalesced block, so scan the block's index range).
            if let Some((i, &ms)) =
                (idx..idx + advance).find_map(|i| self.turn_durations.get(&i).map(|ms| (i, ms)))
            {
                push_styled_line(&mut lines, String::new(), Style::default());
                bars.push(None);
                // Trailing per-turn tokens/cost note, when the finish recorded one.
                let note = self
                    .turn_notes
                    .get(&i)
                    .map(|n| format!(" · {n}"))
                    .unwrap_or_default();
                push_styled_line(
                    &mut lines,
                    format!(
                        "✶ Done in {}{note}",
                        format_request_elapsed(std::time::Duration::from_millis(ms))
                    ),
                    Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                );
                bars.push(None);
            }
            previous_role = Some(message.role.as_str());
            idx += advance;
        }

        compact_lines_and_bars(&mut lines, &mut bars);
        RenderedTranscript::new(lines, bars)
    }

    /// The volatile blocks that follow the committed history — the live streamed
    /// reply and any notice — each with its leading spacing blank. Kept OUT of the
    /// memoized body so a growing stream doesn't re-render and re-wrap the whole
    /// history every frame; composed fresh per frame, like the spinner. Empty in
    /// the empty-state (`build_transcript_history_body` shows neither then).
    ///
    /// The leading blank is unconditional because both blocks always follow
    /// preceding content here: a streamed reply follows the history/intro (and
    /// `should_add_message_spacing(_, "assistant")` is always true), and a notice
    /// always gets its separator — matching the old single-pass spacing exactly.
    pub(super) fn volatile_tail_blocks(
        &self,
        text_width: u16,
    ) -> (Vec<StyledLine>, Vec<Option<Color>>) {
        let mut lines: Vec<StyledLine> = Vec::new();
        let mut bars: Vec<Option<Color>> = Vec::new();
        if self.is_transcript_empty() {
            return (lines, bars);
        }
        if self.pending_response.is_empty() {
            // Thinking-only phase: stream the reasoning as a rolling window so the
            // user watches it think (the spinner carries elapsed/tokens).
            if self.thinking_enabled && reasoning_is_substantive(&self.pending_reasoning) {
                lines.push(blank_line());
                bars.push(None);
                let mut block = Vec::new();
                render_reasoning_window(&mut block, &self.pending_reasoning, text_width);
                push_block(&mut lines, &mut bars, block, None);
            }
        } else {
            // Answer started: show the same window above the streaming reply.
            let live_reasoning = (self.thinking_enabled
                && reasoning_is_substantive(&self.pending_reasoning))
            .then_some(self.pending_reasoning.as_str());
            lines.push(blank_line());
            bars.push(None);
            push_assistant_blocks(
                &mut lines,
                &mut bars,
                live_reasoning.map(|text| ReasoningView {
                    text,
                    expanded: false,
                }),
                &self.pending_response,
                text_width,
                ACCENT,
            );
        }
        // A running `!cmd` streams its output here (not into history) so the
        // memoized history body stays put while lines arrive; it's committed to
        // history once it finishes.
        if let Some(run) = &self.local_command {
            lines.push(blank_line());
            bars.push(None);
            let mut block = Vec::new();
            // Serialize only a bounded preview (+ the true total) rather than the
            // whole streamed buffer: a long-running command can accumulate megabytes,
            // and this runs every frame. Only the first MAX_OUTPUT_LINES ever show.
            let total = run.stdout.lines().count() + run.stderr.lines().count();
            let content = serde_json::json!({
                "command": run.command,
                "stdout": first_lines(&run.stdout, MAX_PERSISTED_OUTPUT_LINES),
                "stderr": first_lines(&run.stderr, MAX_PERSISTED_OUTPUT_LINES),
                "total_lines": total,
                "running": true,
            })
            .to_string();
            render_local_command(&mut block, &content, OutputView::Live);
            push_block(&mut lines, &mut bars, block, Some(SHELL));
        }
        if let Some((color, _)) = notice_display(self.notice.as_ref()) {
            lines.push(blank_line());
            bars.push(None);
            let mut block = Vec::new();
            if let Some(spans) = notice_spans(self.notice.as_ref()) {
                block.push(line_with_plain(spans));
            }
            push_block(&mut lines, &mut bars, block, Some(color));
        }
        (lines, bars)
    }

    /// Length of the run of consecutive `tool_call` entries starting at `start`
    /// that share `name`'s coalescing verb (≥1; see `tool_group_key`).
    fn tool_call_run_len(&self, start: usize, name: &str) -> usize {
        let key = tool_group_key(name);
        self.history[start..]
            .iter()
            .take_while(|m| {
                m.role == "tool_call" && tool_group_key(&decode_tool_call(&m.content).0) == key
            })
            .count()
    }

    /// The `(tool name, target, detached)` for the `tool_result` at `idx`. Results
    /// are emitted after the whole batch in call order, so the j-th result pairs
    /// with the j-th call in the preceding call run (not `idx-1`). `detached` —
    /// the call isn't immediately before the result — means the result carries its
    /// own target, since no adjacent call line shows it.
    fn tool_result_source(&self, idx: usize, cwd: &str) -> Option<(String, String, bool)> {
        // Offset of this result within its contiguous run of results.
        let mut res_start = idx;
        while res_start > 0 && self.history[res_start - 1].role == "tool_result" {
            res_start -= 1;
        }
        let j = idx - res_start;
        // The matching calls are the contiguous tool_call run just before them.
        let mut call_start = res_start;
        while call_start > 0 && self.history[call_start - 1].role == "tool_call" {
            call_start -= 1;
        }
        let call_idx = call_start + j;
        if call_idx >= res_start {
            return None;
        }
        let (name, args) = decode_tool_call(&self.history[call_idx].content);
        let detached = call_idx + 1 != idx;
        Some((
            name.clone(),
            tool_call_target_display(&name, &args, cwd),
            detached,
        ))
    }

    /// The live status line (spinner + activity + elapsed + this turn's tokens),
    /// or `None` when idle. Rebuilt per frame and appended after the cached body
    /// so animation never invalidates the cache.
    pub(super) fn spinner_status_line(&self) -> Option<StyledLine> {
        // A background skill install drives the spinner when no turn or `!cmd`
        // owns it. Suppressed while a skills modal is open — its row narrates,
        // and the status must show in exactly one place.
        if !self.sending
            && self.local_command.is_none()
            && !matches!(self.overlay, Overlay::Skills(_) | Overlay::SkillInstall(_))
            && let Some(progress) = &self.installing_skill
        {
            let mut block = Vec::new();
            render_pending_status(
                &mut block,
                self.frame_tick,
                self.reduce_motion,
                progress.started.elapsed(),
                None,
                &progress.status_text(),
                "",
            );
            return block.into_iter().next();
        }
        let started_at = if self.sending {
            self.request_started_at
        } else if let Some(run) = &self.local_command {
            Some(run.started_at)
        } else {
            return None;
        };
        // Throttled label; fall back to the live one before the first tick.
        let activity = match &self.status_display {
            Some((label, _)) => label.clone(),
            None => self.desired_status(),
        };
        // Tokens (measured, else ~chars/4; 0 omitted) · queued count · esc hint.
        let tail = if self.sending {
            let mut parts: Vec<String> = Vec::new();
            let (used, is_estimate) = if self.turn_output_tokens > 0 {
                (self.turn_output_tokens, false)
            } else {
                let streamed = self.pending_response.len()
                    + self.incoming_buffer.len()
                    + self.pending_reasoning.len();
                (streamed as u64 / 4, true)
            };
            if used > 0 {
                let approx = if is_estimate { "~" } else { "" };
                parts.push(format!("{approx}{} tokens", format_token_count_value(used)));
            }
            let queued = self.queued_input_count();
            if queued > 0 {
                parts.push(format!("{queued} queued"));
            }
            parts.push("esc to interrupt".to_string());
            parts.join(" · ")
        } else {
            "esc to interrupt".to_string()
        };
        // A named tool step is timed by its own runtime, not the whole turn's —
        // else a fast read reads "20m" when a sibling subagent dominates the turn.
        let (elapsed, deadline) = if self.current_action_label().is_some() {
            self.last_tool_action
                .as_ref()
                .map(|(_, since, budget)| {
                    (since.elapsed(), budget.map(std::time::Duration::from_secs))
                })
                .unwrap_or_default()
        } else {
            (
                started_at
                    .map(|started_at| started_at.elapsed())
                    .unwrap_or_default(),
                None,
            )
        };
        let mut block = Vec::new();
        render_pending_status(
            &mut block,
            self.frame_tick,
            self.reduce_motion,
            elapsed,
            deadline,
            &activity,
            &tail,
        );
        block.into_iter().next()
    }

    /// Input typed mid-turn still waiting to run: steering + follow-ups + commands.
    pub(super) fn queued_input_count(&self) -> usize {
        let steering = self
            .steering_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        steering + self.queued_messages.len() + self.queued_commands.len()
    }

    /// The status label right now, pre-throttle: a decision card names the wait,
    /// a tool step names itself, else "Working"/"Thinking" (or a stall's "waiting").
    pub(super) fn desired_status(&self) -> String {
        if !self.sending {
            return "running command".to_string();
        }
        // Blocked on the user — "running rm -rf build (38s)" would imply the
        // command is already executing.
        if self.agent_permission.is_some() {
            return "waiting for your approval".to_string();
        }
        if self.agent_ask.is_some() {
            return "waiting for your answer".to_string();
        }
        if self.agent_review.is_some() {
            return "waiting for your review".to_string();
        }
        if self.agent_plan_approval.is_some() {
            return "waiting for plan approval".to_string();
        }
        // A parallel sub-agent batch owns the headline while its rows are live.
        if !self.subagent_rows.is_empty() {
            let done = self
                .subagent_rows
                .iter()
                .filter(|r| r.done.is_some())
                .count();
            return format!(
                "running {} sub-agents ({done} done)",
                self.subagent_rows.len()
            );
        }
        // A bridged parallel batch: count the trailing run rather than naming
        // only the newest call.
        if let Some((start, total, done)) = self.trailing_tool_batch()
            && total >= 2
        {
            let all_delegates = self.history[start..].iter().all(|m| {
                super::render::canonical_tool_name(&decode_tool_call(&m.content).0) == "subagent"
            });
            let noun = if all_delegates {
                "sub-agents"
            } else {
                "parallel steps"
            };
            return format!("running {total} {noun} ({done} done)");
        }
        if let Some(action) = self.current_action_label() {
            return action;
        }
        // Streaming/retrying → "Working", else "Thinking" — unless silent long
        // enough to look like a stall.
        const STALL_AFTER: std::time::Duration = std::time::Duration::from_secs(10);
        if let Some(last) = self.last_stream_activity
            && last.elapsed() >= STALL_AFTER
        {
            return "waiting".to_string();
        }
        if self.retrying || !self.pending_response.is_empty() || !self.incoming_buffer.is_empty() {
            return "Working".to_string();
        }
        "Thinking".to_string()
    }

    /// Advance the throttled status label (once per loop iteration): adopt the
    /// new label only after the current one has shown for `STATUS_MIN_DURATION`,
    /// so fast steps don't flicker. Called per frame.
    pub(super) fn tick_status_throttle(&mut self) {
        if !self.sending && self.local_command.is_none() {
            self.status_display = None;
            return;
        }
        let desired = self.desired_status();
        match &self.status_display {
            // Unchanged — keep the original timestamp so it can still age out.
            Some((label, _)) if *label == desired => {}
            Some((_, since)) if since.elapsed() < STATUS_MIN_DURATION => {}
            _ => self.status_display = Some((desired, Instant::now())),
        }
    }

    /// Appends the live spinner status line (with its leading spacing blank) to a
    /// freshly built body. The body is already compacted (no trailing blank), so
    /// one blank + the spinner keeps "what's happening" pinned to the bottom of
    /// the transcript without a double gap.
    fn append_spinner_status(&self, lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
        let Some(spinner) = self.spinner_status_line() else {
            return;
        };
        if !lines.is_empty() {
            lines.push(blank_line());
            bars.push(None);
        }
        lines.push(spinner);
        // No accent bar — the status line is chrome, not a message.
        bars.push(None);
        for row in self.subagent_status_rows() {
            lines.push(row);
            bars.push(None);
        }
        for row in self.tool_output_tail_rows() {
            lines.push(row);
            bars.push(None);
        }
    }

    /// Live parallel-batch rows, styled like the spinner they sit under. Empty
    /// when idle so an interrupted batch can't leave ghost rows. Without sink
    /// rows (a bridged engine), a trailing parallel run derives its rows from
    /// the history entries instead.
    fn subagent_status_rows(&self) -> Vec<StyledLine> {
        if !self.sending {
            return Vec::new();
        }
        let style = Style::default().fg(MUTED).add_modifier(Modifier::ITALIC);
        if !self.subagent_rows.is_empty() {
            return self
                .subagent_rows
                .iter()
                .map(|row| line_plain(super::render::subagent_row_text(row), style))
                .collect();
        }
        let Some((start, total, _)) = self.trailing_tool_batch() else {
            return Vec::new();
        };
        if total < 2 {
            return Vec::new();
        }
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.as_str()
        } else {
            self.real_cwd.as_str()
        };
        self.history[start..]
            .iter()
            .map(|m| {
                let (name, args) = decode_tool_call(&m.content);
                let outcome = decode_tool_outcome(&m.content);
                line_plain(
                    super::render::parallel_call_row_text(&name, &args, outcome, cwd),
                    style,
                )
            })
            .collect()
    }

    /// Live `run_bash` tail rows under the spinner; empty when idle so an
    /// interrupted turn can't leave ghost output.
    fn tool_output_tail_rows(&self) -> Vec<StyledLine> {
        if !self.sending {
            return Vec::new();
        }
        let style = Style::default().fg(MUTED);
        let mut rows: Vec<StyledLine> = self
            .tool_output_tail
            .iter()
            .map(|line| line_plain(super::render::tool_tail_row_text(line), style))
            .collect();
        let partial = self.tool_output_partial.trim_end();
        if !partial.trim().is_empty() {
            rows.push(line_plain(
                super::render::tool_tail_row_text(partial),
                style,
            ));
        }
        rows
    }

    /// A cheap O(1) fingerprint of everything the cached *history body* depends
    /// on: intro + committed history (length + the immutable first/last entries).
    /// History entries are *mostly* append-only, so length plus the endpoints
    /// identifies the body without hashing all of it every frame; the one
    /// exception — in-place enrichment of a cursor tool-call entry — bumps
    /// `transcript_revision`, which is mixed in here so those edits still
    /// invalidate. The streamed reply, notice, and spinner are deliberately
    /// EXCLUDED: they live in the volatile tail (composed fresh each frame), so a
    /// growing stream must not bust this cache and force a full-history re-render.
    /// `is_transcript_empty` (which reads the stream/sending) flips at most once
    /// per turn — when streaming starts/ends — so it stays stable mid-stream.
    fn transcript_body_fp(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.is_transcript_empty().hash(&mut hasher);
        self.transcript_revision.hash(&mut hasher);
        // The committed history renders the folded reasoning summary only while
        // this is on, so a toggle must invalidate the memoized body.
        self.thinking_enabled.hash(&mut hasher);
        // The in-flight card hide depends on `sending`, so a flip must rebuild.
        self.sending.hash(&mut hasher);
        self.history.len().hash(&mut hasher);
        if let Some(first) = self.history.first() {
            first.role.hash(&mut hasher);
            first.content.len().hash(&mut hasher);
            first.attachments.len().hash(&mut hasher);
        }
        if let Some(last) = self.history.last() {
            last.role.hash(&mut hasher);
            last.content.len().hash(&mut hasher);
            last.attachments.len().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Rebuilds the cached transcript history body (and its char-wrap height
    /// estimate) only when the history fingerprint or the terminal width changed.
    /// The expensive markdown render and tool decoding happen here, at most once
    /// per *history* change — not on every animation frame, keystroke, or streamed
    /// token (the live reply and notice are the volatile tail, composed outside).
    fn ensure_transcript_cache(&mut self, area_width: u16) {
        let fp = self.transcript_body_fp();
        let fresh = self
            .transcript_cache
            .as_ref()
            .is_some_and(|cache| cache.fp == fp && cache.area_width == area_width);
        if fresh {
            return;
        }
        let body = self.build_transcript_history_body(table_layout_width(area_width));
        let plain_width = area_width.saturating_sub(ACCENT_GUTTER_WIDTH).max(1);
        let plain_prepass = wrap_plain_lines(&body.plain_lines, plain_width).len();
        self.transcript_cache = Some(TranscriptCache {
            fp,
            area_width,
            body,
            plain_prepass,
            styled_width: 0,
            wrapped: None,
        });
    }

    /// Word-wraps the cached body to `text_width`, reusing the previous wrap when
    /// the width is unchanged. Must run after [`ensure_transcript_cache`].
    fn ensure_transcript_wrap(&mut self, text_width: u16) {
        let cache = self
            .transcript_cache
            .as_mut()
            .expect("ensure_transcript_cache runs before ensure_transcript_wrap");
        if cache.wrapped.is_some() && cache.styled_width == text_width {
            return;
        }
        let wrapped = wrap_transcript(&cache.body.lines, &cache.body.bar_colors, text_width);
        cache.styled_width = text_width;
        cache.wrapped = Some(wrapped);
    }

    /// Cheap O(1) fingerprint of the volatile tail's inputs — the streamed reply,
    /// a running `!cmd`'s buffered output, and any notice. All grow append-only
    /// (or are short), so byte lengths + the notice identify the rendered tail
    /// without hashing it. The spinner is deliberately EXCLUDED (it animates every
    /// frame and is wrapped fresh at compose time), so a pure animation tick — the
    /// 60fps redraw that drove the O(reply²) re-parse — hits the cache.
    pub(super) fn volatile_tail_fp(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.is_transcript_empty().hash(&mut hasher);
        self.pending_response.len().hash(&mut hasher);
        // `pending_reasoning.len()` keys the live thinking summary's line-count
        // fallback; `thinking_enabled` gates whether it renders, so a /config
        // toggle mid-turn must invalidate too.
        self.pending_reasoning.len().hash(&mut hasher);
        self.thinking_enabled.hash(&mut hasher);
        match &self.local_command {
            Some(run) => {
                run.command.hash(&mut hasher);
                run.stdout.len().hash(&mut hasher);
                run.stderr.len().hash(&mut hasher);
            }
            None => 0usize.hash(&mut hasher),
        }
        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            text.as_ref().hash(&mut hasher);
            format!("{color:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Renders the volatile tail (markdown → styled lines) only when its
    /// fingerprint or the render width changed — not on every animation frame or
    /// every streamed token of an unchanged reply. The expensive markdown parse +
    /// table layout (the O(reply) cost the spinner redraw used to repeat every
    /// frame) happen here, at most once per *content* change.
    fn ensure_volatile_tail(&mut self, render_width: u16) {
        let fp = self.volatile_tail_fp();
        let fresh = self
            .volatile_tail_cache
            .as_ref()
            .is_some_and(|cache| cache.fp == fp && cache.render_width == render_width);
        if fresh {
            return;
        }
        let (lines, bars) = self.volatile_tail_blocks(render_width);
        self.volatile_tail_cache = Some(VolatileTailCache {
            fp,
            render_width,
            lines,
            bars,
            plain_width: 0,
            prepass: 0,
            styled_width: 0,
            wrapped: None,
        });
    }

    /// Char-wrapped row count of the cached tail at `plain_width`, for the pane-
    /// height prepass — memoized on the width so a streamed reply isn't re-counted
    /// every frame. Must run after [`ensure_volatile_tail`].
    fn volatile_tail_prepass(&mut self, plain_width: u16) -> usize {
        let cache = self
            .volatile_tail_cache
            .as_mut()
            .expect("ensure_volatile_tail runs before volatile_tail_prepass");
        if cache.lines.is_empty() {
            return 0;
        }
        if cache.plain_width == plain_width {
            return cache.prepass;
        }
        let plain: Vec<String> = cache.lines.iter().map(|l| l.plain.clone()).collect();
        let rows = wrap_plain_lines(&plain, plain_width).len();
        cache.plain_width = plain_width;
        cache.prepass = rows;
        rows
    }

    /// Word-wraps the cached tail to `text_width`, reusing the previous wrap when
    /// the width is unchanged. Leaves `wrapped = None` when the tail is empty (so
    /// it contributes no rows). Must run after [`ensure_volatile_tail`].
    fn ensure_volatile_tail_wrap(&mut self, text_width: u16) {
        let cache = self
            .volatile_tail_cache
            .as_mut()
            .expect("ensure_volatile_tail runs before ensure_volatile_tail_wrap");
        if cache.lines.is_empty() {
            cache.wrapped = None;
            cache.styled_width = text_width;
            return;
        }
        if cache.wrapped.is_some() && cache.styled_width == text_width {
            return;
        }
        cache.wrapped = Some(wrap_transcript(&cache.lines, &cache.bars, text_width));
        cache.styled_width = text_width;
    }

    /// Composes the final wrapped transcript for this frame: the cached history
    /// body wrap, the cached volatile-tail wrap (streamed reply + running `!cmd` +
    /// notice), and the freshly wrapped spinner. Only the spinner — two lines that
    /// animate — is wrapped per frame; the history and tail wraps are reused from
    /// their caches, so a growing stream costs O(tail-delta), not O(history+reply)
    /// every frame. The rest are shallow clones.
    fn composed_transcript_rows(
        &self,
        spinner: Option<&StyledLine>,
        text_width: u16,
    ) -> (Text<'static>, Vec<String>, Vec<Option<Color>>) {
        let wrapped = self
            .transcript_cache
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
            .expect("ensure_transcript_wrap runs before composed_transcript_rows");
        let mut text_lines: Vec<Line<'static>> = wrapped.text.lines.clone();
        let mut rows = wrapped.rows.clone();
        let mut bars = wrapped.bars.clone();

        // The volatile tail already carries its own leading spacing blanks (see
        // `volatile_tail_blocks`); reuse its cached wrap instead of re-wrapping the
        // streamed reply every frame. `wrapped` is `None` exactly when the tail is
        // empty, so this adds no spurious rows.
        if let Some(tail) = self
            .volatile_tail_cache
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
        {
            text_lines.extend(tail.text.lines.iter().cloned());
            rows.extend(tail.rows.iter().cloned());
            bars.extend(tail.bars.iter().cloned());
        }

        // The spinner mirrors `append_spinner_status` — a leading blank (whenever
        // anything precedes it, which it always does) then the status line. Wrapped
        // fresh here, never cached, since it animates every frame.
        if let Some(spinner) = spinner {
            let mut tail: Vec<StyledLine> = Vec::new();
            let mut tail_bars: Vec<Option<Color>> = Vec::new();
            if !rows.is_empty() {
                tail.push(blank_line());
                tail_bars.push(None);
            }
            tail.push(spinner.clone());
            tail_bars.push(None);
            for row in self.subagent_status_rows() {
                tail.push(row);
                tail_bars.push(None);
            }
            for row in self.tool_output_tail_rows() {
                tail.push(row);
                tail_bars.push(None);
            }
            let wrapped_tail = wrap_transcript(&tail, &tail_bars, text_width);
            text_lines.extend(wrapped_tail.text.lines);
            rows.extend(wrapped_tail.rows);
            bars.extend(wrapped_tail.bars);
        }

        (Text::from(text_lines), rows, bars)
    }

    pub(super) fn transcript_intro_lines(&self, width: u16) -> Vec<String> {
        // Plain mirror of the banner for height reservation; derived from the
        // styled builder to stay in lockstep with `render_empty_state`.
        brand_wordmark_lines(width)
            .into_iter()
            .map(|line| line.plain)
            .collect()
    }

    pub(super) fn render(&mut self, frame: &mut Frame<'_>) {
        self.tick_selection_flash();
        self.refresh_git_branch();
        self.check_live_share_health();
        let outer = frame.area();
        self.picker_hitbox = None;
        self.transcript_hitbox = None;
        self.screen_region = None;
        self.overlay_detail_area = None;
        let composer_area = self.render_main(frame, outer);
        if let Some(menu) = self.visible_command_menu() {
            let (area, placement) = command_menu_area(
                composer_area,
                outer,
                menu.entries.len(),
                self.command_menu.placement,
            );
            self.command_menu.placement = Some(placement);
            self.render_command_menu(frame, area, &menu);
        }
        let body = outer;

        match self.overlay.clone() {
            Overlay::Picker(picker) => {
                // Only the session picker gets the split (preview) layout.
                let (area, split) = match picker.kind {
                    PickerKind::Session => split_overlay_area(body, 86, 80, 68, 72),
                    _ => (centered_rect(68, 72, body), false),
                };
                let out = self.render_picker(frame, area, &picker, split);
                self.overlay_detail_area = out.detail_area;
                if let (Some(clamped), Overlay::Picker(p)) = (out.detail_scroll, &mut self.overlay)
                {
                    p.preview_scroll = clamped;
                    p.preview_scroll_for = out.scroll_for;
                }
            }
            Overlay::Help { scroll } => {
                let area = centered_rect(72, 88, body);
                self.screen_region = Some(overlay_content_rect(area));
                let clamped = self.render_help_overlay(frame, area, scroll);
                if let Overlay::Help { scroll } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Context { report, scroll } => {
                let area = centered_rect(72, 88, body);
                self.screen_region = Some(overlay_content_rect(area));
                let clamped = self.render_context_overlay(frame, area, &report, scroll);
                if let Overlay::Context { scroll, .. } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Skills(skills) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.screen_region = Some(overlay_content_rect(area));
                let out = self.render_skills_overlay(frame, area, &skills, split);
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Skills(s) = &mut self.overlay {
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    // Canonicalize a drill-in that a resize carried into split mode.
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::Agents(agents) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.screen_region = Some(overlay_content_rect(area));
                let out = self.render_agents_overlay(frame, area, &agents, split);
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Agents(s) = &mut self.overlay {
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::SkillInstall(pick) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.screen_region = Some(overlay_content_rect(area));
                let out = self.render_skill_install_overlay(frame, area, &pick, split);
                self.overlay_detail_area = out.detail_area;
                if let Overlay::SkillInstall(s) = &mut self.overlay {
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::Mcp(mcp) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.screen_region = Some(overlay_content_rect(area));
                let out = self.render_mcp_overlay(frame, area, &mcp, split);
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Mcp(s) = &mut self.overlay {
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::McpTools(tools) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                self.render_mcp_tools_overlay(frame, area, &tools);
            }
            Overlay::McpPaste(paste) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                self.render_mcp_paste_overlay(frame, area, &paste);
            }
            Overlay::Config(config) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                self.render_config_overlay(frame, area, &config);
            }
            Overlay::None => {}
        }

        if self.pending_mcp_consent.is_some() {
            self.render_mcp_consent_card(frame, composer_area, outer);
        } else if self.pending_logout.is_some() {
            self.render_logout_confirm_card(frame, composer_area, outer);
        } else if self.pending_key_switch.is_some() {
            self.render_key_switch_confirm_card(frame, composer_area, outer);
        } else if self.agent_permission.is_some() {
            self.render_permission_card(frame, composer_area, outer);
        } else if self.agent_ask.is_some() {
            self.render_ask_user_card(frame, composer_area, outer);
        } else if self.agent_plan_approval.is_some() {
            let clamped = self.render_plan_approval_card(frame, composer_area, outer);
            if let (Some(s), Some(p)) = (clamped, self.agent_plan_approval.as_mut()) {
                p.scroll = s;
            }
        } else if self.agent_review.is_some() {
            let clamped = self.render_review_card(frame, composer_area, outer);
            if let (Some(s), Some(r)) = (clamped, self.agent_review.as_mut()) {
                r.scroll = s;
            }
        } else if self.account_login.is_some() {
            // Last: passive status — decision cards win the slot.
            self.render_login_card(frame, composer_area, outer);
        }

        // Snapshot the finished screen so a drag can copy from anywhere on it,
        // then wash the full-screen selection over whatever now sits there.
        self.capture_screen_surface(frame);
        self.render_screen_selection_highlight(frame);

        self.render_toast(frame, outer);
        scrub_control_cells(frame.buffer_mut());
    }

    /// Captures the rendered screen into `screen_surface` for full-screen drag-copy.
    /// Confined to `screen_region` (a modal's content rect) when one is open.
    fn capture_screen_surface(&mut self, frame: &mut Frame<'_>) {
        let full = frame.area();
        let area = self
            .screen_region
            .map(|region| region.intersection(full))
            .unwrap_or(full);
        let buffer = frame.buffer_mut();
        let mut rows = Vec::with_capacity(usize::from(area.height));
        for y in area.y..area.y.saturating_add(area.height) {
            let mut row = String::with_capacity(usize::from(area.width));
            for x in area.x..area.x.saturating_add(area.width) {
                if let Some(cell) = buffer.cell((x, y)) {
                    // Wide-char spacers carry an empty symbol, so the row's display
                    // width stays equal to its column count.
                    row.push_str(cell.symbol());
                }
            }
            rows.push(row);
        }
        self.screen_surface = Some(ScreenSurface { area, rows });
    }

    /// Screen-coordinate twin of `render_transcript_selection_highlight`.
    fn render_screen_selection_highlight(&self, frame: &mut Frame<'_>) {
        let Some(selection) = self
            .screen_selection
            .filter(|selection| !selection.is_empty())
        else {
            return;
        };
        let Some(surface) = &self.screen_surface else {
            return;
        };

        let wash = if self.selection_flash_until.is_some() {
            SELECT_FLASH
        } else {
            SELECT_WASH
        };
        let (start, end) = normalized_selection(selection);
        let area = surface.area;
        let row_end = end.row.min(surface.rows.len().saturating_sub(1));
        if start.row > row_end {
            return;
        }

        let buffer = frame.buffer_mut();
        for row in start.row..=row_end {
            // Clamp to the row's real text (trailing blanks excluded), matching copy.
            let text_width = surface
                .rows
                .get(row)
                .map(|line| row_display_width(line.trim_end()))
                .unwrap_or(0);
            let start_col = if row == start.row { start.column } else { 0 };
            let end_col = if row == end.row {
                end.column
            } else {
                text_width
            };
            let start_col = start_col.min(text_width);
            let end_col = end_col.min(text_width);
            if start_col >= end_col {
                continue;
            }
            let y = area.y.saturating_add(row as u16);
            for column in start_col..end_col {
                if let Some(cell) = buffer.cell_mut((area.x + column, y)) {
                    cell.set_bg(wash);
                }
            }
        }
    }

    /// Card asking whether to spawn a repo's project `.mcp.json` stdio servers
    /// (the local code-execution surface). Lists each server's exact command so
    /// the risk is visible, then color-coded y/a/n keys. Anchored above the
    /// composer like the permission card.
    fn render_mcp_consent_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) {
        let Some(prompt) = self.pending_mcp_consent.as_ref() else {
            return;
        };
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);
        // Fixed chrome: 2 borders + heading + note + blank-before-keys + keys = 6.
        // Whatever rows remain list the servers (trimmed if the screen is short).
        let chrome = 6usize;
        let list_budget = usize::from(max_total).saturating_sub(chrome);

        let n = prompt.servers.len();
        let heading = format!(
            "Run {n} MCP server{} from this repo's .mcp.json?",
            if n == 1 { "" } else { "s" }
        );
        let note = "These commands run locally on your machine.";
        let keys = mcp_consent_keys_line();

        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        let mut content_w = display_width(&heading).max(display_width(note)).max(keys_w);
        for (name, cmd) in &prompt.servers {
            content_w = content_w.max(display_width(&format!("{name}  {cmd}")));
        }
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))];
        let shown = n.min(list_budget);
        for (name, cmd) in prompt.servers.iter().take(shown) {
            let room = inner_width.saturating_sub(display_width(name) + 2).max(1);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{name}  "),
                    Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_for_display_width(cmd, room),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        if shown < n {
            lines.push(Line::from(Span::styled(
                format!("…and {} more", n - shown),
                Style::default().fg(MUTED),
            )));
        }
        lines.push(Line::from(Span::styled(
            note.to_string(),
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(WARNING))
            .title(Span::styled(
                " mcp servers ",
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// The `/logout` y/n confirm card (owns the keyboard, like MCP consent).
    fn render_logout_confirm_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) {
        let Some(account) = self.pending_logout.as_ref() else {
            return;
        };
        let lines = vec![
            Line::from(vec![
                Span::styled("Unlink this device from ", Style::default().fg(TEXT)),
                Span::styled(
                    account.clone(),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
                Span::styled("?", Style::default().fg(TEXT)),
            ]),
            Line::from(Span::styled(
                "This device drops to the free tier until you sign in again.",
                Style::default().fg(MUTED),
            )),
            Line::from(""),
            account_keys_line(&[("y", ASSISTANT, "sign out"), ("n", ERROR, "cancel")]),
        ];
        render_account_card(
            frame,
            composer_area,
            frame_area,
            "sign out of aivo",
            WARNING,
            lines,
        );
    }

    /// `/key` provider-switch confirm card.
    fn render_key_switch_confirm_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) {
        let Some(target) = self.pending_key_switch.as_ref() else {
            return;
        };
        let lines = vec![
            Line::from(vec![
                Span::styled("Switch to ", Style::default().fg(TEXT)),
                Span::styled(
                    target.display_name().to_string(),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
                Span::styled("?", Style::default().fg(TEXT)),
            ]),
            Line::from(Span::styled(
                "It's a different provider, so this starts a new session.",
                Style::default().fg(MUTED),
            )),
            Line::from(Span::styled(
                "The current session is saved — /resume brings it back.",
                Style::default().fg(MUTED),
            )),
            Line::from(""),
            account_keys_line(&[
                ("y", ASSISTANT, "new session"),
                ("n", ERROR, "keep current"),
            ]),
        ];
        render_account_card(
            frame,
            composer_area,
            frame_area,
            "switch key",
            WARNING,
            lines,
        );
    }

    /// The `/login` status card: code + URL + waiting state. Passive — it never
    /// owns the keyboard (see `handle_login_card_key`), so typing stays live.
    fn render_login_card(&self, frame: &mut Frame<'_>, composer_area: Rect, frame_area: Rect) {
        let Some(card) = self.account_login.as_ref() else {
            return;
        };
        let lines = vec![
            Line::from(vec![
                Span::styled(
                    "Confirm this code in your browser:  ",
                    Style::default().fg(TEXT),
                ),
                Span::styled(
                    card.user_code.clone(),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                card.open_url.clone(),
                Style::default().fg(LINK),
            )),
            Line::from(Span::styled(
                "Waiting for approval…",
                Style::default().fg(MUTED),
            )),
            Line::from(""),
            account_keys_line(&[
                ("Enter", ASSISTANT, "open browser"),
                ("Esc", ERROR, "cancel"),
            ]),
        ];
        render_account_card(
            frame,
            composer_area,
            frame_area,
            "sign in to aivo",
            ACCENT,
            lines,
        );
    }

    /// Card asking the user to approve a mutating agent tool. Anchored directly
    /// above the composer (where the eye and cursor already are) rather than
    /// floating mid-screen, so the decision sits right next to the input. Shows
    /// the action, a preview (diff / command / path), and color-coded y/a/n keys.
    fn render_permission_card(&self, frame: &mut Frame<'_>, composer_area: Rect, frame_area: Rect) {
        let Some(pending) = self.agent_permission.as_ref() else {
            return;
        };
        // Flag a run_bash card: locally destructive, or a remote mutation.
        // Destructive wins the label when both apply.
        let cmd = if pending.tool == "run_bash" {
            pending.preview.as_deref()
        } else {
            None
        };
        let flag_line = if cmd.is_some_and(crate::agent::tools::bash_looks_destructive) {
            Some("⚠ looks destructive")
        } else if cmd.is_some_and(crate::agent::tools::bash_mutates_remote) {
            Some("⚠ remote side effect")
        } else {
            None
        };
        let destructive = flag_line.is_some();

        // The card rests just above the composer's divider line (a narrower card
        // would otherwise leave that full-width rule poking out past its right
        // edge). Cap its height to the rows above so it never runs off the top.
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);

        // Fixed chrome: 2 borders + heading + blank-before-keys + keys (+ the
        // destructive flag line). Whatever rows remain feed the preview block
        // (its own leading blank + the preview lines), bottom-trimmed first so
        // the keys line is always the last thing the user sees.
        let chrome = 5 + usize::from(destructive);
        let preview_budget = usize::from(max_total).saturating_sub(chrome);
        // Expand tabs up front so width sizing and the cell grid agree.
        let preview: Vec<String> = pending
            .preview
            .as_deref()
            .map(|p| {
                p.lines()
                    .take(12)
                    .map(|l| expand_tabs(l).into_owned())
                    .collect()
            })
            .unwrap_or_default();
        let preview_take = if preview.is_empty() {
            0
        } else {
            preview.len().min(preview_budget.saturating_sub(1))
        };

        // Size the card to its widest visible line rather than the whole input
        // row, so a short confirm reads as a compact card; never wider than the
        // composer. +4 = 2 borders + 1 col of padding on each side.
        let heading = permission_heading(&pending.tool);
        let keys = permission_keys_line(&pending.tool, !self.draft.is_empty());
        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        let mut content_w = display_width(&heading).max(keys_w);
        if let Some(flag) = flag_line {
            content_w = content_w.max(display_width(flag));
        }
        for raw in preview.iter().take(preview_take) {
            content_w = content_w.max(display_width(raw));
        }
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))];
        if preview_take > 0 {
            lines.push(Line::from(""));
            for raw in preview.iter().take(preview_take) {
                let trimmed = raw.trim_start();
                let style = if trimmed.starts_with("+ ") {
                    Style::default().fg(ASSISTANT)
                } else if trimmed.starts_with("- ") {
                    Style::default().fg(ERROR)
                } else if pending.tool == "run_bash" {
                    Style::default().fg(if destructive { WARNING } else { TEXT })
                } else {
                    Style::default().fg(MUTED)
                };
                lines.push(Line::from(Span::styled(
                    truncate_for_display_width(raw, inner_width),
                    style,
                )));
            }
        }
        if let Some(flag) = flag_line {
            lines.push(Line::from(Span::styled(
                flag.to_string(),
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " permission ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// The `ask_user` card: question, numbered pick-list (`❯` = highlighted), key
    /// hint. Floats above the composer like the permission card, clamped to the
    /// rows above.
    fn render_ask_user_card(&self, frame: &mut Frame<'_>, composer_area: Rect, frame_area: Rect) {
        let Some(ask) = self.agent_ask.as_ref() else {
            return;
        };
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let inner_cap = usize::from(max_width.saturating_sub(4)).max(1);

        // Question wraps to at most 3 lines; options render as "N. label — desc".
        let mut q_lines = super::overlay_render_impl::wrap_chars(&ask.question, inner_cap);
        q_lines.truncate(3);
        let opt_plain: Vec<String> = ask
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| match &o.description {
                Some(d) => format!("{}. {} — {}", i + 1, o.label, d),
                None => format!("{}. {}", i + 1, o.label),
            })
            .collect();

        // Size to the widest visible line (question / option+marker / keys).
        let keys = ask_user_keys_line(ask.allow_free_text, ask.multi_select);
        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        // Multi-select prefixes each option with a "[✓] " checkbox.
        let box_w = if ask.multi_select { 4 } else { 0 };
        let mut content_w = keys_w;
        for l in &q_lines {
            content_w = content_w.max(display_width(l));
        }
        for s in &opt_plain {
            content_w = content_w.max(display_width(s) + 2 + box_w);
        }
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        // Assemble lines; trim the option list from the bottom if the card would
        // overrun the space above the composer (keys/question stay visible).
        let mut lines: Vec<Line<'static>> = Vec::new();
        for l in &q_lines {
            lines.push(Line::from(Span::styled(
                truncate_for_display_width(l, inner_width),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        // Fixed chrome after the options: blank + keys + 2 borders.
        let chrome_after = 3usize;
        let option_budget = usize::from(max_total)
            .saturating_sub(lines.len() + chrome_after)
            .max(1);
        let shown = ask.options.len().min(option_budget);
        for (i, opt) in ask.options.iter().enumerate().take(shown) {
            let selected = i == ask.selected;
            let marker_style = if selected {
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(FAINT)
            };
            let label_style = if selected {
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            let mut spans = vec![Span::styled(
                if selected { "❯ " } else { "  " },
                marker_style,
            )];
            if ask.multi_select {
                let checked = ask.checked.get(i).copied().unwrap_or(false);
                let box_style = if checked {
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(FAINT)
                };
                spans.push(Span::styled(
                    if checked { "[✓] " } else { "[ ] " },
                    box_style,
                ));
            }
            spans.push(Span::styled(
                format!("{}. ", i + 1),
                Style::default().fg(MUTED),
            ));
            // The description (FAINT) only shows if it still fits after the label.
            let label =
                truncate_for_display_width(&opt.label, inner_width.saturating_sub(3 + box_w));
            let used = display_width(&label) + 3 + box_w;
            spans.push(Span::styled(label, label_style));
            if let Some(desc) = opt
                .description
                .as_deref()
                .filter(|_| used + 3 < inner_width)
            {
                let room = inner_width - used - 3;
                spans.push(Span::styled(
                    format!(" — {}", truncate_for_display_width(desc, room)),
                    Style::default().fg(FAINT),
                ));
            }
            lines.push(Line::from(spans));
        }
        if shown < ask.options.len() {
            lines.push(Line::from(Span::styled(
                format!("  …{} more", ask.options.len() - shown),
                Style::default().fg(FAINT),
            )));
        }
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " question ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// The edit-review card: heading, the scrollable precomputed diff, and y/n keys.
    /// Floats above the composer; returns the clamped scroll for write-back.
    fn render_review_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) -> Option<u16> {
        let review = self.agent_review.as_ref()?;
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let inner_width = usize::from(max_width.saturating_sub(4)).max(1);

        let heading = format!(
            "review {} edit{} before writing",
            review.count,
            if review.count == 1 { "" } else { "s" }
        );
        let keys = review_keys_line();

        // heading + 2 blanks + keys + 2 borders + reserved overflow-marker row.
        let chrome = 7u16;
        let body_budget = usize::from(max_total.saturating_sub(chrome)).max(1);
        let overflow = review.body.len() > body_budget;
        // Last-full-page clamp: keeps the card height stable at the bottom.
        let scroll = usize::from(review.scroll).min(review.body.len().saturating_sub(body_budget));
        let visible = review.body.len().saturating_sub(scroll).min(body_budget);

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(Span::styled(
            truncate_for_display_width(&heading, inner_width),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for line in review.body.iter().skip(scroll).take(visible) {
            lines.push(line.clone());
        }
        let remaining = review.body.len().saturating_sub(scroll + visible);
        if remaining > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … +{remaining} more (↑↓ scroll)"),
                Style::default().fg(FAINT),
            )));
        } else if overflow {
            lines.push(Line::from(Span::styled(
                "  … end of diff",
                Style::default().fg(FAINT),
            )));
        }
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width: max_width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " review edits ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
        Some(u16::try_from(scroll).unwrap_or(u16::MAX))
    }

    /// The plan-approval card (`exit_plan_mode`): heading, the scrollable rendered
    /// plan, and the three verdicts. Floats above the composer like the review card;
    /// returns the clamped scroll for write-back.
    fn render_plan_approval_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) -> Option<u16> {
        let pending = self.agent_plan_approval.as_ref()?;
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let inner_width = usize::from(max_width.saturating_sub(4)).max(1);

        // heading + blank + (plan…) + blank + 3 options + blank + keys + 2 borders;
        // one more row is reserved for the "+N more" marker when the plan overflows.
        let chrome = 10u16;
        let body_budget = usize::from(max_total.saturating_sub(chrome)).max(1);
        let overflow = pending.body.len() > body_budget;
        let scroll =
            usize::from(pending.scroll).min(pending.body.len().saturating_sub(body_budget));
        let visible = pending.body.len().saturating_sub(scroll).min(body_budget);

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(Span::styled(
            truncate_for_display_width("Implementation plan — ready for review", inner_width),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for line in pending.body.iter().skip(scroll).take(visible) {
            lines.push(line.clone());
        }
        let remaining = pending.body.len().saturating_sub(scroll + visible);
        if remaining > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … +{remaining} more (PgUp/PgDn scroll)"),
                Style::default().fg(FAINT),
            )));
        } else if overflow {
            lines.push(Line::from(Span::styled(
                "  … end of plan",
                Style::default().fg(FAINT),
            )));
        }
        lines.push(Line::from(""));
        const OPTIONS: [&str; 3] = [
            "Approve — execute with auto-approve",
            "Approve — review each edit first",
            "Keep planning — type feedback below",
        ];
        for (i, opt) in OPTIONS.iter().enumerate() {
            let selected = i == pending.selected;
            let (marker_style, label_style) = if selected {
                (
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            } else {
                (Style::default().fg(FAINT), Style::default().fg(TEXT))
            };
            lines.push(Line::from(vec![
                Span::styled(if selected { "❯ " } else { "  " }, marker_style),
                Span::styled(format!("{}. ", i + 1), Style::default().fg(MUTED)),
                Span::styled(
                    truncate_for_display_width(opt, inner_width.saturating_sub(5)),
                    label_style,
                ),
            ]));
        }
        lines.push(Line::from(""));
        lines.push(plan_approval_keys_line());

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width: max_width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " plan ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
        Some(u16::try_from(scroll).unwrap_or(u16::MAX))
    }

    pub(super) fn render_main(&mut self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        let composer_height = self.composer_height(area.width);
        // The footer is a single fixed row: just the status line. No hint bar, so
        // the layout never shifts up or down as turns start and finish.
        let footer_height = 1u16;
        // The pinned plan/task-list panel sits between the transcript and the
        // composer (a faint top rule + the wrapped checklist), so progress stays
        // visible instead of scrolling away under later tool calls. Sized from the
        // current plan, capped so the transcript keeps a usable minimum.
        let plan_lines = self.plan_panel_lines();
        let plan_panel_height =
            self.plan_panel_height(&plan_lines, area, composer_height, footer_height);
        // Clamp queue focus each frame — the engine or a turn-end drain may
        // have emptied the rows it selects since the last event.
        let queue_rows = self.queued_rows();
        match (&mut self.queue_focus, queue_rows.len()) {
            (focus @ Some(_), 0) => *focus = None,
            (Some(sel), n) => *sel = (*sel).min(n - 1),
            (None, _) => {}
        }
        let queue_lines = self.queued_panel_lines(&queue_rows, area.width);
        let queue_panel_height = self.queued_panel_height(
            &queue_lines,
            area,
            composer_height,
            footer_height,
            plan_panel_height,
        );
        let max_transcript_height = area
            .height
            .saturating_sub(
                composer_height + footer_height + plan_panel_height + queue_panel_height,
            )
            .max(1);
        let is_empty = self.is_transcript_empty();
        // Memoize the heavy history body build + wrap AND the volatile tail
        // (streamed reply + running !cmd + notice); only the per-frame spinner is
        // rebuilt fresh (see `ensure_transcript_cache` / `ensure_volatile_tail`).
        self.ensure_transcript_cache(area.width);
        // Render the volatile tail at most once per content change — its markdown
        // parse + wrap are reused across animation frames of an unchanged reply.
        self.ensure_volatile_tail(table_layout_width(area.width));
        let spinner = self.spinner_status_line();
        let plain_width = area.width.saturating_sub(ACCENT_GUTTER_WIDTH).max(1);
        // The volatile tail's char-wrap height, sized like the body's estimate so
        // the pane grows to fit the streamed reply (which left the cached body).
        let volatile_prepass = self.volatile_tail_prepass(plain_width);
        // Spinner blank + status line + any live sub-agent rows, sized the same
        // way the body's char-wrap height estimate is, so the pane height matches.
        let spinner_prepass = spinner
            .as_ref()
            .map(|line| {
                let mut plain = vec![String::new(), line.plain.clone()];
                plain.extend(self.subagent_status_rows().into_iter().map(|r| r.plain));
                wrap_plain_lines(&plain, plain_width).len()
            })
            .unwrap_or(0);
        let prepass_rows = self
            .transcript_cache
            .as_ref()
            .map(|cache| cache.plain_prepass)
            .unwrap_or(1)
            + volatile_prepass
            + spinner_prepass;
        let min_transcript_height = self
            .empty_state_height(area.width.max(1))
            .clamp(1, max_transcript_height);
        let transcript_height = if is_empty {
            min_transcript_height
        } else {
            (prepass_rows as u16).clamp(min_transcript_height, max_transcript_height)
        };
        let stack_height = transcript_height
            .saturating_add(plan_panel_height)
            .saturating_add(queue_panel_height)
            .saturating_add(composer_height)
            .saturating_add(footer_height)
            .min(area.height.max(1));

        let stack = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(stack_height), Constraint::Min(0)])
            .split(area);
        // transcript / [plan panel] / [queue panel] / composer / footer — the
        // plan and queue rows are present only when non-empty, so the indices
        // below shift accordingly.
        let mut constraints = vec![Constraint::Length(transcript_height)];
        if plan_panel_height > 0 {
            constraints.push(Constraint::Length(plan_panel_height));
        }
        if queue_panel_height > 0 {
            constraints.push(Constraint::Length(queue_panel_height));
        }
        constraints.push(Constraint::Length(composer_height));
        constraints.push(Constraint::Length(footer_height));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(stack[0]);

        let transcript_area = chunks[0];
        let mut chunk_idx = 1usize;
        let plan_panel_area = if plan_panel_height > 0 {
            let a = chunks[chunk_idx];
            chunk_idx += 1;
            Some(a)
        } else {
            None
        };
        let queue_panel_area = if queue_panel_height > 0 {
            let a = chunks[chunk_idx];
            chunk_idx += 1;
            Some(a)
        } else {
            None
        };
        let composer_outer = chunks[chunk_idx];
        let footer_outer = chunks[chunk_idx + 1];
        // The composer reserves its own blank spacing row above the divider (see
        // the composer layout below), so the transcript fills its whole area here
        // — no extra bottom padding carved from within, in overflow or otherwise.
        let transcript_view_area = transcript_area;
        let view_height = transcript_view_area.height.max(1);
        // The transcript uses its full width — no column reserved for a scrollbar.
        let transcript_content_area = transcript_view_area;
        // Reserve a left gutter for per-role accent bars. Content wraps and
        // renders into the inset text area; the bars are painted separately so
        // they never bleed into copied/selected text.
        let transcript_text_area = Rect {
            x: transcript_content_area
                .x
                .saturating_add(ACCENT_GUTTER_WIDTH),
            y: transcript_content_area.y,
            width: transcript_content_area
                .width
                .saturating_sub(ACCENT_GUTTER_WIDTH)
                .max(1),
            height: transcript_content_area.height,
        };
        // Word-wrap ourselves to the text width so our row model (scroll, gutter,
        // selection) exactly matches the rendered rows; we render with wrap OFF.
        // The body and volatile-tail wraps are cached; only the spinner is wrapped
        // per frame.
        self.ensure_transcript_wrap(transcript_text_area.width);
        self.ensure_volatile_tail_wrap(transcript_text_area.width);
        let (wrapped_text, visual_rows, visual_bars) =
            self.composed_transcript_rows(spinner.as_ref(), transcript_text_area.width);
        let transcript_total_lines = visual_rows.len();
        self.transcript_width = transcript_text_area.width.max(1);
        self.transcript_view_height = view_height;
        let max_scroll = transcript_total_lines.saturating_sub(usize::from(view_height));
        // Cache the exact value so the scroll handlers don't rebuild the whole
        // transcript per wheel event (see `effective_max_scroll`).
        self.last_max_scroll = Some(max_scroll);
        if self.follow_output {
            self.transcript_scroll = max_scroll;
        } else {
            self.transcript_scroll = self.transcript_scroll.min(max_scroll);
        }
        self.transcript_hitbox = Some(TranscriptHitbox {
            area: transcript_text_area,
            first_row: self.transcript_scroll,
            // Last use of `visual_rows`; move it in rather than re-cloning the
            // whole-transcript row vector on every repaint.
            rows: visual_rows,
        });

        frame.render_widget(Clear, chunks[0]);

        if is_empty {
            // Inset by the accent gutter so the brand banner sits at the same
            // column as the transcript content does once a message arrives — without
            // this the banner jumps 2 cols right when the first message lands.
            self.render_empty_state(frame, transcript_text_area);
            self.jump_to_bottom_hit = None;
        } else {
            // Pre-wrapped above → render with wrap OFF so rendered rows match.
            // ratatui's `Paragraph` does NOT virtualize: `.scroll((y, 0))` still
            // runs EVERY line through its reflow/LineComposer and writes cells
            // before discarding the scrolled-past rows, so a full-transcript
            // Paragraph costs O(total rows) per frame. While a turn streams, the
            // spinner forces a ~60fps repaint, and on the single-thread runtime
            // that O(n) draw starves the streaming task — a long session makes a
            // subagent crawl. Slice to the visible window and render at scroll 0
            // → O(visible rows). The full row model (gutter, selection, hitbox)
            // below is unchanged, so geometry stays exact.
            let view_start = self.transcript_scroll.min(wrapped_text.lines.len());
            let view_end = view_start
                .saturating_add(usize::from(transcript_text_area.height))
                .min(wrapped_text.lines.len());
            let visible_text = Text::from(wrapped_text.lines[view_start..view_end].to_vec());
            let transcript_widget = Paragraph::new(visible_text).style(Style::default().fg(TEXT));
            frame.render_widget(transcript_widget, transcript_text_area);
            self.paint_accent_gutter(
                frame,
                transcript_content_area.x,
                transcript_text_area.y,
                transcript_text_area.height,
                &visual_bars,
            );
            self.render_transcript_selection_highlight(frame, transcript_text_area);
            // Clickable jump-to-bottom pill (like Ctrl+End), only while scrolled up.
            self.jump_to_bottom_hit = if self.transcript_scroll < max_scroll {
                render_jump_to_bottom(frame, transcript_view_area)
            } else {
                None
            };
        }

        if let Some(plan_panel_area) = plan_panel_area {
            self.render_plan_panel(frame, plan_panel_area, &plan_lines);
        }

        if let Some(queue_panel_area) = queue_panel_area {
            self.render_queued_panel(frame, queue_panel_area, &queue_lines);
        }

        // A blank spacing row, then the divider rule, then the input. The blank
        // row gives the prompt one line of breathing room above its divider —
        // matching the single blank line the transcript leaves between blocks —
        // instead of the rule pressing directly against the last message.
        let composer_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
            ])
            .split(composer_outer);

        frame.render_widget(Clear, composer_chunks[0]);
        frame.render_widget(
            Paragraph::new(self.composer_rule_line(composer_chunks[1].width.max(1))),
            composer_chunks[1],
        );

        let composer_area = composer_chunks[2];
        // Record the area + scroll the draft so the cursor row stays on-screen,
        // then render the (pre-wrapped) visible rows. We wrap ourselves into
        // hanging-indent rows and render with wrap OFF, so rendering, cursor
        // placement, and mouse hit-testing all share one geometry.
        self.composer_text_area = Some(composer_area);
        self.update_composer_scroll(composer_area);
        let composer = Paragraph::new(self.render_composer_text());
        frame.render_widget(composer, composer_area);

        if self.should_show_input_cursor()
            && let Some((cursor_x, cursor_y)) = self.composer_cursor_screen(composer_area)
        {
            frame.set_cursor_position((cursor_x, cursor_y));
        }

        self.render_footer(frame, footer_outer);
        composer_area
    }

    /// The divider rule above the composer. It always carries the auto-approve
    /// mode badge, inset near the right end — amber "on" when active, a faint
    /// "⇧⇥ … off" (naming the toggle key) when not — so the mode and how to
    /// change it are always discoverable. It gets this fixed home above the input
    /// rather than the hint bar, which drops its right-hand items on narrow
    /// terminals.
    pub(super) fn composer_rule_line(&self, width: u16) -> Line<'static> {
        let width = usize::from(width);
        // In `!cmd` shell mode the prompt's top line picks up the magenta shell
        // hue; in plan mode the whole rule tints ACCENT so the read-only session is
        // unmistakable. Shell wins (the tinted draft must match).
        // Plan mode from either backend: in-process engine or cursor's ACP mode.
        let plan_mode = self.plan_mode || self.cursor_plan_mode;
        let rule_style = if self.draft_is_shell_command() {
            Style::default().fg(SHELL)
        } else if plan_mode {
            Style::default().fg(ACCENT)
        } else {
            Style::default().fg(FAINT)
        };
        // The mode badge — one slot, since the four modes are exclusive.
        let (badge, badge_style) = if plan_mode {
            (
                "◇ plan",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else if self.agent_auto_approve {
            ("⚡ auto-approve", Style::default().fg(WARNING))
        } else if self.agent_review_edits {
            ("✎ review", Style::default().fg(TOOL))
        } else {
            ("normal", Style::default().fg(MUTED))
        };
        const CYCLE_HINT: &str = " (shift+tab)";
        // Left title on the rule. While recalling input history, show
        // `History {pos}/{total}` — the newest entry reads as total/total and
        // counts down as you scroll further back — preceded by two rule dashes
        // so it reads as a titled divider (matching the recall affordance).
        // Otherwise, a live `/goal` step indicator pinned to the very left so an
        // unattended loop is always visible (not just in a transient notice).
        // The two never coincide in one frame: history recall is a foreground
        // composer action.
        let (left_text, left_style, left_lead) = if let Some(index) = self.draft_history_index {
            (
                format!(" History {}/{} ", index + 1, self.draft_history.len()),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                2usize,
            )
        } else if let Some(goal) = self.goal_mode.as_ref() {
            (
                format!(" ◎ goal {}/{} ", goal.iteration, goal.max),
                Style::default().fg(ACCENT),
                0usize,
            )
        } else {
            (String::new(), Style::default(), 0usize)
        };
        // Left-cluster badge for running jobs (count cached + reaped per event-loop tick).
        let jobs_running = self.jobs_running;
        let jobs_text = if jobs_running > 0 {
            let s = if jobs_running == 1 { "" } else { "s" };
            format!(" ✦ {jobs_running} job{s} ")
        } else {
            String::new()
        };
        let jobs_w = display_width(&jobs_text);
        let trailing = 2usize;
        // Badge + faint cycle hint, one space of padding each side.
        let badge_w = display_width(badge) + display_width(CYCLE_HINT) + 2;
        let left_w = if left_text.is_empty() {
            0
        } else {
            left_lead + display_width(&left_text)
        };
        if width <= left_w + jobs_w + badge_w + trailing + 2 {
            // Too narrow to inset it all — keep just the mode badge.
            return Line::from(Span::styled(badge.to_string(), badge_style));
        }
        let fill = width - left_w - jobs_w - badge_w - trailing;
        let mut spans = Vec::with_capacity(7);
        if !left_text.is_empty() {
            if left_lead > 0 {
                spans.push(Span::styled("─".repeat(left_lead), rule_style));
            }
            spans.push(Span::styled(left_text, left_style));
        }
        if !jobs_text.is_empty() {
            spans.push(Span::styled(jobs_text, Style::default().fg(TOOL)));
        }
        spans.push(Span::styled("─".repeat(fill), rule_style));
        spans.push(Span::styled(format!(" {badge}"), badge_style));
        spans.push(Span::styled(
            format!("{CYCLE_HINT} "),
            Style::default().fg(FAINT),
        ));
        spans.push(Span::styled("─".repeat(trailing), rule_style));
        Line::from(spans)
    }

    /// The pinned plan/task-list panel's content lines (the `Plan N/M done`
    /// header plus one line per step), or empty when there's no plan or the plan
    /// is fully done. Built fresh each frame — it's small, and the plan changes
    /// rarely.
    fn plan_panel_lines(&self) -> Vec<StyledLine> {
        let Some(content) = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "plan")
            .map(|m| m.content.as_str())
        else {
            return Vec::new();
        };
        // A finished plan is hidden (clutter, and reads as false "done" on error).
        if plan_all_completed(content) {
            return Vec::new();
        }
        let mut lines = Vec::new();
        render_plan(&mut lines, content);
        lines
    }

    /// Rows the pinned plan panel will occupy (0 when there's no plan): a top rule
    /// plus the wrapped checklist, capped so the transcript keeps a usable minimum
    /// and a long plan can't dominate the screen (it scrolls instead).
    fn plan_panel_height(
        &self,
        lines: &[StyledLine],
        area: Rect,
        composer_height: u16,
        footer_height: u16,
    ) -> u16 {
        if lines.is_empty() {
            return 0;
        }
        let body_width = area.width.saturating_sub(2).max(1);
        let plain: Vec<String> = lines.iter().map(|l| l.plain.clone()).collect();
        let content_rows = wrap_plain_lines(&plain, body_width).len() as u16;
        let reserved = composer_height
            .saturating_add(footer_height)
            .saturating_add(PLAN_PANEL_MIN_TRANSCRIPT);
        let max_body = area
            .height
            .saturating_sub(reserved)
            .min(area.height / 3)
            .max(1);
        content_rows.clamp(1, max_body).saturating_add(1) // + top rule
    }

    /// Paint the pinned plan panel: a faint top rule (fencing it off from the
    /// transcript, mirroring the composer's divider) over the wrapped checklist.
    /// When the plan overflows the panel, scroll so the active (`in_progress`)
    /// step stays on screen.
    fn render_plan_panel(&self, frame: &mut Frame<'_>, area: Rect, lines: &[StyledLine]) {
        if area.height == 0 || lines.is_empty() {
            return;
        }
        frame.render_widget(Clear, area);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "─".repeat(usize::from(area.width.max(1))),
                Style::default().fg(FAINT),
            ))),
            Rect {
                x: area.x,
                y: area.y,
                width: area.width,
                height: 1,
            },
        );
        let body = Rect {
            x: area.x.saturating_add(1),
            y: area.y.saturating_add(1),
            width: area.width.saturating_sub(2).max(1),
            height: area.height.saturating_sub(1),
        };
        if body.height == 0 {
            return;
        }
        let bars = vec![None; lines.len()];
        let wrapped = wrap_transcript(lines, &bars, body.width);
        // Keep the in-progress step visible when the plan is taller than the panel.
        let scroll = wrapped
            .rows
            .iter()
            .position(|r| r.contains('▸'))
            .filter(|&i| i >= usize::from(body.height))
            .map(|i| (i + 1 - usize::from(body.height)) as u16)
            .unwrap_or(0);
        frame.render_widget(Paragraph::new(wrapped.text).scroll((scroll, 0)), body);
    }

    /// Queued-input panel lines: a blank spacer, one line per item (windowed
    /// around the selection with `… +k` indicators), a hint line while focused.
    fn queued_panel_lines(&self, rows: &[QueuedRow], width: u16) -> Vec<Line<'static>> {
        if rows.is_empty() {
            return Vec::new();
        }
        let selected = self.queue_focus;
        let start = selected
            .map(|sel| sel.saturating_sub(QUEUE_PANEL_MAX_ROWS - 1))
            .unwrap_or(0)
            .min(rows.len().saturating_sub(QUEUE_PANEL_MAX_ROWS));
        let end = (start + QUEUE_PANEL_MAX_ROWS).min(rows.len());
        let mut lines = vec![Line::default()];
        if start > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … +{start} earlier"),
                Style::default().fg(FAINT),
            )));
        }
        for (i, row) in rows.iter().enumerate().take(end).skip(start) {
            let is_selected = selected == Some(i);
            let marker = if is_selected { "▸ " } else { "  " };
            let prefix = match row.segment {
                QueueSegment::Steering => "» ",
                QueueSegment::Command => "",
                QueueSegment::Message => "· ",
            };
            let room = usize::from(width).saturating_sub(3 + prefix.chars().count());
            let (marker_style, text_style) = if is_selected {
                (Style::default().fg(ACCENT), Style::default().fg(TEXT))
            } else {
                (Style::default().fg(MUTED), Style::default().fg(MUTED))
            };
            lines.push(Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(
                    format!("{prefix}{}", truncate_for_display_width(&row.display, room)),
                    text_style,
                ),
            ]));
        }
        if end < rows.len() {
            lines.push(Line::from(Span::styled(
                format!("  … +{} more", rows.len() - end),
                Style::default().fg(FAINT),
            )));
        }
        if selected.is_some() {
            lines.push(Line::from(Span::styled(
                "  enter edit · ctrl+d remove · alt+↑↓ move · esc back",
                Style::default().fg(FAINT),
            )));
        }
        lines
    }

    /// Panel height, clamped so the transcript keeps a usable minimum.
    fn queued_panel_height(
        &self,
        lines: &[Line<'static>],
        area: Rect,
        composer_height: u16,
        footer_height: u16,
        plan_panel_height: u16,
    ) -> u16 {
        if lines.is_empty() {
            return 0;
        }
        let reserved = composer_height
            .saturating_add(footer_height)
            .saturating_add(plan_panel_height)
            .saturating_add(PLAN_PANEL_MIN_TRANSCRIPT);
        (lines.len() as u16).min(area.height.saturating_sub(reserved))
    }

    fn render_queued_panel(&self, frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>]) {
        if area.height == 0 || lines.is_empty() {
            return;
        }
        frame.render_widget(Clear, area);
        frame.render_widget(Paragraph::new(Text::from(lines.to_vec())), area);
    }

    pub(super) fn empty_state_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(1).max(1);
        let mut height = if let Some(loading) = &self.loading_resume {
            let mut rows = vec![
                "Loading saved session…".to_string(),
                loading.preview.title.clone(),
                plain_text_from_spans(&resume_metadata_spans(
                    &loading.preview,
                    content_width.saturating_sub(1).max(1),
                )),
                self.display_cwd().to_string(),
            ];
            rows.extend(self.notice_plain_lines(content_width));
            rows.extend(self.spinner_status_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        } else {
            // Inset width, matching what `render_empty_state` draws into.
            let mut rows = self.transcript_intro_lines(width.saturating_sub(ACCENT_GUTTER_WIDTH));
            // Reserve the chip + tip height too, matching `render_empty_state`.
            rows.extend(self.welcome_status_lines().into_iter().map(|sl| sl.plain));
            rows.extend(self.notice_plain_lines(content_width));
            rows.extend(self.spinner_status_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        };
        height = height
            .saturating_add(EMPTY_STATE_TOP_GAP)
            .saturating_add(EMPTY_STATE_BOTTOM_GAP);
        height.max(1)
    }

    fn notice_plain_lines(&self, width: u16) -> Vec<String> {
        notice_display(self.notice.as_ref())
            .map(|(_, text)| {
                let mut lines = vec![String::new()];
                lines.extend(wrap_plain_lines(&[text.into_owned()], width));
                lines
            })
            .unwrap_or_default()
    }

    /// Height-side twin of the spinner line `render_empty_state` appends.
    fn spinner_status_plain_lines(&self, width: u16) -> Vec<String> {
        self.spinner_status_line()
            .map(|line| {
                let mut lines = vec![String::new()];
                lines.extend(wrap_plain_lines(&[line.plain], width));
                lines
            })
            .unwrap_or_default()
    }

    /// Paint a `▌` accent bar in the reserved gutter column for each visible
    /// transcript row, colored by the role of the block that owns that row.
    fn paint_accent_gutter(
        &self,
        frame: &mut Frame<'_>,
        gutter_x: u16,
        top_y: u16,
        view_height: u16,
        bars: &[Option<Color>],
    ) {
        let buffer = frame.buffer_mut();
        for offset in 0..view_height {
            let row_index = self.transcript_scroll + usize::from(offset);
            let Some(Some(color)) = bars.get(row_index).copied() else {
                continue;
            };
            if let Some(cell) = buffer.cell_mut((gutter_x, top_y.saturating_add(offset))) {
                cell.set_symbol("▌");
                cell.set_fg(color);
            }
        }
    }

    /// Auto-clears the selection once the post-copy flash window elapses, so a
    /// just-copied selection briefly lights up then disappears (amp-style).
    pub(super) fn tick_selection_flash(&mut self) {
        if let Some(until) = self.selection_flash_until
            && Instant::now() >= until
        {
            self.selection_flash_until = None;
            self.transcript_selection = None;
            self.screen_selection = None;
        }
    }

    fn render_transcript_selection_highlight(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(selection) = self
            .transcript_selection
            .filter(|selection| !selection.is_empty())
        else {
            return;
        };
        let Some(hitbox) = &self.transcript_hitbox else {
            return;
        };

        let wash = if self.selection_flash_until.is_some() {
            SELECT_FLASH
        } else {
            SELECT_WASH
        };
        let (start, end) = normalized_selection(selection);
        let visible_start = hitbox.first_row;
        let visible_end = visible_start.saturating_add(usize::from(area.height));
        let row_start = start.row.max(visible_start);
        let row_end = end.row.min(visible_end.saturating_sub(1));
        if row_start > row_end {
            return;
        }

        let buffer = frame.buffer_mut();
        for row in row_start..=row_end {
            let local_y = row.saturating_sub(visible_start) as u16;
            // Clamp the wash to the row's real text so we never paint the blank
            // cells past a line's end — keeps the highlight matching what copy
            // actually yields (trailing space is trimmed on copy).
            let text_width = hitbox
                .rows
                .get(row)
                .map(|line| row_display_width(line))
                .unwrap_or(0);
            let start_col = if row == start.row { start.column } else { 0 };
            let end_col = if row == end.row {
                end.column
            } else {
                text_width
            };
            let start_col = start_col.min(area.width);
            let end_col = end_col.min(text_width).min(area.width);
            if start_col >= end_col {
                continue;
            }

            for column in start_col..end_col {
                if let Some(cell) = buffer.cell_mut((area.x + column, area.y + local_y)) {
                    cell.set_bg(wash);
                }
            }
        }
    }

    pub(super) fn render_toast(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let Some(toast) = self.toast.clone() else {
            return;
        };
        let now = Instant::now();
        if now >= toast.expires_at {
            self.toast = None;
            return;
        }

        let text_width = display_width(&toast.text).min(usize::from(area.width));
        let toast_width = (text_width as u16).saturating_add(4).min(area.width.max(1));
        // Anchor bottom-right: float on the last transcript row, just above the
        // composer rule, so the confirmation appears near where the user acted
        // without clobbering the divider, composer, or footer. Falls back to the
        // top edge before the first layout records the composer area.
        let anchor_y = self
            .composer_text_area
            .map(|c| c.y.saturating_sub(2))
            .unwrap_or(area.y);
        let toast_area = Rect {
            x: area
                .x
                .saturating_add(area.width.saturating_sub(toast_width)),
            y: anchor_y,
            width: toast_width,
            height: 1,
        };
        let color = if now.duration_since(toast.created_at) >= TOAST_FADE_AFTER {
            FAINT
        } else {
            ACCENT
        };
        frame.render_widget(Clear, toast_area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(&toast.text, Style::default().fg(color)),
                Span::raw(" "),
            ]))
            .style(Style::default().bg(Color::Rgb(24, 21, 17))),
            toast_area,
        );
    }

    pub(super) fn render_empty_state(&self, frame: &mut Frame<'_>, area: Rect) {
        let content_area = Rect {
            x: area.x,
            y: area.y.saturating_add(EMPTY_STATE_TOP_GAP),
            width: area.width,
            height: area
                .height
                .saturating_sub(EMPTY_STATE_TOP_GAP)
                .saturating_sub(EMPTY_STATE_BOTTOM_GAP),
        };

        let lines = if let Some(loading) = &self.loading_resume {
            vec![
                Line::from(vec![
                    Span::styled(
                        "Loading",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " saved session…",
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    loading.preview.title.clone(),
                    Style::default().fg(TEXT),
                )),
                Line::from(resume_metadata_spans(
                    &loading.preview,
                    area.width.max(1).saturating_sub(2),
                )),
                Line::from(Span::styled(self.display_cwd(), Style::default().fg(FAINT))),
            ]
        } else {
            // `area` is the gutter-inset column, driving the full/narrow choice.
            brand_wordmark_lines(area.width)
                .into_iter()
                .map(|sl| sl.line)
                .collect()
        };

        let mut lines = lines;
        // Chip + tip on the fresh welcome only, never the resume-loading screen.
        if self.loading_resume.is_none() {
            lines.extend(self.welcome_status_lines().into_iter().map(|sl| sl.line));
        }
        if let Some(spans) = notice_spans(self.notice.as_ref()) {
            lines.push(Line::from(""));
            lines.push(Line::from(spans));
        }
        // The empty state replaces the transcript's spinner tail, so a fetch on
        // a fresh chat with no overlay open must narrate here.
        if let Some(spinner) = self.spinner_status_line() {
            lines.push(Line::from(""));
            lines.push(spinner.line);
        }

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            content_area,
        );
    }

    /// The `N skills · M MCP` capability chip, or `None` when neither is configured.
    pub(super) fn welcome_capabilities_label(&self) -> Option<String> {
        let skills = self.skill_commands.len();
        let mcp = self.mcp_configured_count;
        let mut parts: Vec<String> = Vec::new();
        if skills > 0 {
            let noun = if skills == 1 { "skill" } else { "skills" };
            parts.push(format!("{skills} {noun}"));
        }
        if mcp > 0 {
            parts.push(format!("{mcp} MCP"));
        }
        (!parts.is_empty()).then(|| parts.join(" · "))
    }

    /// Blank spacer, optional capability chip, then the rotating tip. Shared by
    /// the empty state, the transcript intro, and `empty_state_height` (kept in
    /// lockstep, measuring the same lines).
    fn welcome_status_lines(&self) -> Vec<StyledLine> {
        let mut lines = vec![blank_line()];
        if let Some(chip) = self.welcome_capabilities_label() {
            lines.push(line_plain(chip, Style::default().fg(MUTED)));
        }
        let tip = WELCOME_TIPS[self.welcome_tip_index % WELCOME_TIPS.len()];
        lines.push(line_with_plain(vec![
            // MUTED hint (up from FAINT) so the tip reads on dim terminals.
            Span::styled("✶ Tip  ", Style::default().fg(ACCENT)),
            Span::styled(tip.to_string(), Style::default().fg(MUTED)),
        ]));
        lines
    }

    pub(super) fn render_composer_text(&self) -> Text<'static> {
        let prompt = if self.draft_history_index.is_some() {
            Span::styled(
                "^ ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else if self.draft_is_shell_command() {
            // `!cmd` shell mode tints the prompt itself in the magenta shell hue too.
            Span::styled(
                "> ",
                Style::default().fg(SHELL).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("> ", Style::default().fg(USER).add_modifier(Modifier::BOLD))
        };
        let mut lines = composer_attachment_lines(&self.draft_attachments);
        if self.draft.is_empty() {
            let placeholder = if self.loading_resume.is_some() {
                Span::styled("Resume loading…", Style::default().fg(FAINT))
            } else if self.sending {
                Span::styled(
                    " Type to queue your next message…",
                    Style::default().fg(FAINT),
                )
            } else {
                Span::styled(
                    " Ask, plan, or build · / for commands",
                    Style::default().fg(FAINT),
                )
            };
            lines.push(Line::from(vec![prompt, placeholder]));
            return Text::from(lines);
        }

        // Ghost hint trailing a bare slash command (Claude-Code style), e.g.
        // `> /mcp [add … | rm <name>]`. Only set when the draft is a single line.
        let ghost = self.composer_command_hint();
        // A `!cmd` draft is tinted in the magenta shell hue, so shell mode reads at
        // a glance — the prompt and its top divider pick up the same magenta color.
        let draft_color = if self.draft_is_shell_command() {
            SHELL
        } else {
            TEXT
        };
        // Pre-wrapped visual rows: row 0 gets the `> ` prompt; every other row
        // gets a matching 2-col hanging indent so wrapped text aligns under the
        // first character. Rendered with wrap OFF (see `render_main`).
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let last = rows.len().saturating_sub(1);
        for (index, &(start, end)) in rows.iter().enumerate().skip(self.composer_scroll) {
            let prefix = if index == 0 {
                prompt.clone()
            } else {
                Span::raw("  ")
            };
            let mut spans = vec![
                prefix,
                Span::styled(
                    self.draft[start..end].to_string(),
                    Style::default().fg(draft_color),
                ),
            ];
            if index == last
                && let Some(hint) = ghost
            {
                spans.push(Span::styled(format!(" {hint}"), Style::default().fg(FAINT)));
            }
            lines.push(Line::from(spans));
        }

        Text::from(lines)
    }

    /// Scroll the draft within the composer so the cursor's visual row stays
    /// inside the visible window. Attachment lines sit above the draft and aren't
    /// scrolled; only the draft rows scroll. Recomputed each render.
    pub(super) fn update_composer_scroll(&mut self, area: Rect) {
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let attach = self.draft_attachments.len();
        let visible = usize::from(area.height).saturating_sub(attach).max(1);
        let (cursor_row, _) = composer_cursor_rowcol(&self.draft, self.cursor, &rows);
        if cursor_row < self.composer_scroll {
            self.composer_scroll = cursor_row;
        } else if cursor_row >= self.composer_scroll + visible {
            self.composer_scroll = cursor_row + 1 - visible;
        }
        self.composer_scroll = self.composer_scroll.min(rows.len().saturating_sub(visible));
    }

    /// Absolute terminal `(x, y)` for the input cursor, in the composer's
    /// hanging-indent wrap model, accounting for attachment rows and scroll.
    /// `None` when the cursor row is scrolled out of view.
    pub(super) fn composer_cursor_screen(&self, area: Rect) -> Option<(u16, u16)> {
        let (x_rel, row) = cursor_position(
            &self.draft,
            self.cursor,
            area.width.max(1),
            COMPOSER_PREFIX_WIDTH,
        );
        let row = usize::from(row);
        if row < self.composer_scroll {
            return None;
        }
        let attach = self.draft_attachments.len() as u16;
        let y = area.y + attach + (row - self.composer_scroll) as u16;
        let x = area.x + x_rel;
        let max_x = area.x + area.width.saturating_sub(1);
        let max_y = area.y + area.height.saturating_sub(1);
        Some((x.min(max_x), y.min(max_y)))
    }

    pub(super) fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        // Right side: the context meter (which warms toward the window limit) and,
        // when thinking is on, the effort tier. The effort is a static setting, so
        // it stays quiet MUTED — only the meter's warning/error warmth draws the eye.
        let (meter_label, meter_color) = self.footer_status_label();
        let mut right_spans: Vec<Span<'static>> =
            vec![Span::styled(meter_label, Style::default().fg(meter_color))];
        if let Some(effort) = self.footer_effort_label() {
            right_spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
            right_spans.push(Span::styled(effort, Style::default().fg(MUTED)));
        }
        let right_label_width: u16 = right_spans
            .iter()
            .map(|s| display_width(s.content.as_ref()) as u16)
            .sum();
        let left_width = if right_label_width == 0 {
            area.width
        } else {
            area.width.saturating_sub(right_label_width + 1)
        };
        // Reserve columns for the model-line badges so the text truncates to fit them.
        let live = self.live_share.is_some();
        let plain_chat = !self.agent_tools_enabled;
        let glue = 3u16; // " · " between the model and each badge
        let badge_w = if live {
            display_width(LIVE_BADGE) as u16 + glue
        } else {
            0
        } + if plain_chat {
            display_width(PLAIN_CODE_BADGE) as u16 + glue
        } else {
            0
        };
        let left_text = build_footer_text(
            &self.raw_model,
            &self.key.base_url,
            &self.key.name,
            self.display_cwd(),
            self.git_branch.as_deref(),
            left_width.saturating_sub(badge_w),
        );
        // Status-line: the model name and host/cwd context share one MUTED hue,
        // with the ` · ` glue receding to FAINT between segments.
        let mut spans: Vec<Span<'static>> = Vec::new();
        for (index, segment) in left_text.split(" · ").enumerate() {
            if index > 0 {
                spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
            }
            spans.push(Span::styled(
                segment.to_string(),
                Style::default().fg(MUTED),
            ));
            // Badges sit right after the model (first segment).
            if index == 0 {
                if live {
                    spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
                    spans.push(Span::styled(LIVE_BADGE, Style::default().fg(LIVE)));
                }
                if plain_chat {
                    spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
                    spans.push(Span::styled(PLAIN_CODE_BADGE, Style::default().fg(USER)));
                }
            }
        }
        let left_len = display_width(&left_text) as u16 + badge_w;
        let pad = left_width.saturating_sub(left_len);
        if right_label_width > 0 {
            spans.push(Span::raw(" ".repeat(usize::from(pad) + 1)));
            spans.extend(right_spans);
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Effort tier for the status line: Cursor's from the model id, else the engine's
    /// effective level (only while thinking is on, so the two can't disagree).
    pub(super) fn footer_effort_label(&self) -> Option<String> {
        if let Some(label) = self.cursor_effort_label.as_deref() {
            Some(label.to_string())
        } else if self.thinking_enabled && self.model_supports_thinking {
            self.effective_reasoning_effort()
        } else {
            None
        }
    }

    /// Present-tense label for the in-flight tool step (e.g. `running grep`), or
    /// `None`. Uses the same in-flight test that hides the tool's card (trailing
    /// `tool_call` run with a pending result), so the status and the card never
    /// both show.
    pub(super) fn current_action_label(&self) -> Option<String> {
        self.trailing_tool_batch()?;
        self.last_tool_action
            .as_ref()
            .map(|(label, _, _)| label.clone())
    }

    /// The trailing contiguous `tool_call` run while the model is between
    /// replies: `(start index, total, resolved)`. `None` when idle, once reply
    /// text is streaming, or when every call has its outcome — cursor resolves
    /// entries in place, so a batch stays live until its last member resolves.
    pub(super) fn trailing_tool_batch(&self) -> Option<(usize, usize, usize)> {
        if !self.sending || !self.pending_response.is_empty() || !self.incoming_buffer.is_empty() {
            return None;
        }
        let mut start = self.history.len();
        while start > 0 && self.history[start - 1].role == "tool_call" {
            start -= 1;
        }
        let total = self.history.len() - start;
        let done = self.history[start..]
            .iter()
            .filter(|m| {
                let (result, failed) = decode_tool_outcome(&m.content);
                result.is_some() || failed
            })
            .count();
        (done < total).then_some((start, total, done))
    }

    /// Tokens to show in the footer fill right now, and whether the figure is a
    /// chars/4 estimate rather than provider-measured. During an in-flight turn we
    /// prefer the live measured usage (Anthropic streams it from `message_start`);
    /// until that lands we grow a chars/4 estimate of the transcript plus the text
    /// streamed so far, so the fill still moves for providers that only report
    /// usage at the end of the turn. Idle: the last turn's measured total.
    pub(super) fn context_fill(&self) -> (u64, bool) {
        if self.sending {
            if let Some(usage) = self.live_usage {
                return (usage.total_tokens(), false);
            }
            // No measured usage yet this turn: grow from the best known baseline —
            // the prior turn's fill, or the transcript estimate when larger (a fresh
            // chat with no prior turn) — plus the text streamed so far. Taking the
            // max avoids the footer dropping at turn start when the prior fill was a
            // measured total (which exceeds the chars/4 transcript estimate).
            let streamed = (self.pending_response.len()
                + self.incoming_buffer.len()
                + self.pending_reasoning.len()) as u64
                / 4;
            let baseline = self
                .context_tokens
                .max(estimate_context_tokens(&self.history));
            return (baseline + streamed, true);
        }
        match self.last_usage {
            Some(usage) => (usage.total_tokens(), false),
            None => (self.context_tokens, self.context_is_estimate),
        }
    }

    pub(super) fn footer_status_label(&self) -> (String, Color) {
        let (used, is_estimate) = self.context_fill();
        if self.context_window == 0 {
            // No known window: just the count. Use the live measurement while a
            // turn is in flight so the figure tracks the stream, else the last
            // turn's; a `None` makes `format_token_count` flag it `~` (estimate).
            let usage = if self.sending {
                self.live_usage
            } else {
                self.last_usage
            };
            return (format_token_count(used, usage), MUTED);
        }
        // Fresh session: show the window size, not an empty `0 / 1M` meter.
        if used == 0 {
            return (
                format!("{} context", format_token_count_value(self.context_window)),
                MUTED,
            );
        }
        // Percent isn't shown (the used/window pair already implies it) but still
        // drives the meter color.
        let pct = (used.saturating_mul(100) / self.context_window).min(100);
        // Mark estimate-only figures (cursor ACP / agents without reported usage):
        // aivo's tracked transcript is a fraction of the model's real context, so
        // the number understates the true fill — `~` flags it as approximate.
        let approx = if is_estimate && used > 0 { "~" } else { "" };
        let label = format!(
            "{approx}{}/{}{}",
            format_token_count_value(used),
            format_token_count_value(self.context_window),
            self.session_cost_label(),
        );
        (label, context_fill_color(pct))
    }

    /// ` · ~$X.XX` session-spend suffix; empty only without any recorded spend.
    /// Always `~`: snapshot list prices × parsed usage is an estimate, not a bill.
    pub(super) fn session_cost_label(&self) -> String {
        if self.session_cost_usd <= 0.0 {
            return String::new();
        }
        format!(" · ~${}", format_usd(self.session_cost_usd))
    }

    pub(super) fn composer_height(&self, width: u16) -> u16 {
        // Count wrapped *visual* rows, not logical lines, so a long line that
        // wraps grows the box (and keeps the cursor on-screen) instead of being
        // clipped. The clamp caps growth at 7 text rows; longer drafts scroll
        // within the box (see `composer_scroll`).
        let draft_rows = if self.draft.is_empty() {
            1
        } else {
            let text_width = usize::from(width)
                .saturating_sub(usize::from(COMPOSER_PREFIX_WIDTH))
                .max(1);
            composer_visual_rows(&self.draft, text_width).len()
        };
        let lines = (draft_rows + self.draft_attachments.len()) as u16;
        // +3 reserves the leading blank spacing row, the divider rule, and one
        // trailing row below the input; the rest is draft text (capped, then it
        // scrolls within the box).
        (lines + 3).clamp(4, 10)
    }

    /// Wrap width available to the composer's draft text (the rendered composer
    /// width minus the per-row prompt indent). Falls back to a sane default
    /// before the first render has recorded the area.
    pub(super) fn composer_text_width(&self) -> usize {
        self.composer_text_area
            .map(|area| usize::from(area.width))
            .unwrap_or(80)
            .saturating_sub(usize::from(COMPOSER_PREFIX_WIDTH))
            .max(1)
    }
}

/// A human-readable question for the permission card heading. Known mutating
/// tools get a plain-language phrase; anything else (e.g. an MCP tool) falls
/// back to its raw name.
fn permission_heading(tool: &str) -> String {
    match tool {
        "run_bash" => "Run a command?".to_string(),
        "run_bash_unsandboxed" => "Run outside the workspace sandbox?".to_string(),
        "cursor" => "Allow Cursor to run this?".to_string(),
        "write_file" => "Write a file?".to_string(),
        "edit_file" | "multi_edit" => "Edit a file?".to_string(),
        other => format!("Allow {other}?"),
    }
}

/// Content-sized bordered card for the account flows. Placed below the
/// composer's footer when the dead space there fits it (a short session must
/// not cover the banner/transcript while empty rows sit unused); a full screen
/// falls back to the above-the-composer permission-card slot.
fn render_account_card(
    frame: &mut Frame<'_>,
    composer_area: Rect,
    frame_area: Rect,
    title: &str,
    border: Color,
    mut lines: Vec<Line<'static>>,
) {
    let needed = lines.len() as u16 + 2;
    let anchor = composer_area.y.saturating_sub(1);
    let above_budget = anchor.saturating_sub(frame_area.y).max(1);
    // +3: the 2-row footer (status + hint bar) plus a blank gap.
    let below_top = composer_area
        .y
        .saturating_add(composer_area.height)
        .saturating_add(3);
    let below_budget = frame_area
        .y
        .saturating_add(frame_area.height)
        .saturating_sub(below_top);
    let below = below_budget >= needed;
    let budget = if below { below_budget } else { above_budget };
    // Tight slot: shed spacer rows before clipping content (key hints).
    if lines.len() as u16 + 2 > budget {
        lines.retain(|l| l.width() > 0);
    }
    let content_w = lines.iter().map(Line::width).max().unwrap_or(0);
    let max_width = composer_area.width.min(frame_area.width).max(1);
    let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
    let height = (lines.len() as u16 + 2).min(budget);
    let y = if below {
        below_top
    } else {
        anchor.saturating_sub(height).max(frame_area.y)
    };
    let card = Rect {
        x: composer_area.x,
        y,
        width,
        height,
    };
    frame.render_widget(Clear, card);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(border).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(card).inner(ratatui::layout::Margin {
        vertical: 0,
        horizontal: 1,
    });
    frame.render_widget(block, card);
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

/// Color-coded `key label` hint row for the account cards.
fn account_keys_line(keys: &[(&'static str, Color, &'static str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (key, color, label)) in keys.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("    ".to_string(), Style::default().fg(FAINT)));
        }
        spans.push(Span::styled(
            key.to_string(),
            Style::default().fg(*color).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {label}"),
            Style::default().fg(MUTED),
        ));
    }
    Line::from(spans)
}

/// The project-MCP consent choices row: run once / always (this repo) / deny,
/// color-coded like the permission card's traffic light.
fn mcp_consent_keys_line() -> Line<'static> {
    let keycap = |key: &str, color: Color| {
        Span::styled(
            key.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    Line::from(vec![
        keycap("y", ASSISTANT),
        label(" run once"),
        gap(),
        keycap("a", WARNING),
        label(" always (this repo)"),
        gap(),
        keycap("n", ERROR),
        label(" deny"),
    ])
}

/// The choices row: color-coded keycaps reading like a traffic light —
/// green allow, amber always (it arms auto-approve), red deny. `tool` selects
/// the "always" scope wording (a Cursor card's "always" is session-wide, unlike
/// the native engine's, which is scoped to this one command/path), and
/// `composing` swaps in a hint when a queued-message draft is in progress —
/// there the letter keys type into the draft instead of deciding (see
/// `handle_permission_key`).
fn permission_keys_line(tool: &str, composing: bool) -> Line<'static> {
    let keycap = |key: &str, color: Color| {
        Span::styled(
            key.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    if composing {
        // A draft is in progress, so y/a/n flow into the message, not the card.
        // Only Esc (deny) and Shift+Tab (allow + auto-approve) act on the card.
        return Line::from(vec![
            keycap("⇧⇥", WARNING),
            label(" allow"),
            gap(),
            keycap("esc", ERROR),
            label(" deny"),
            gap(),
            label("y/a/n type into your message"),
        ]);
    }
    // Cursor's "always" turns on auto-approve for the rest of the session (its
    // out-of-process tools can't be remembered per-action), so spell that out;
    // the native engine's "always" is scoped to this command/path and reads as
    // expected without a qualifier.
    let always_label = if tool == "cursor" {
        " always (this session)"
    } else {
        " always"
    };
    Line::from(vec![
        keycap("y", ASSISTANT),
        label(" allow once"),
        gap(),
        keycap("a", WARNING),
        label(always_label),
        gap(),
        keycap("n", ERROR),
        label(" deny"),
    ])
}

/// The `ask_user` card's key-hint row: "space toggle · ↵ confirm" in multi-select,
/// otherwise "↵ select" (with a "type your own" note when free text is allowed).
fn ask_user_keys_line(allow_free_text: bool, multi_select: bool) -> Line<'static> {
    let keycap = |key: &str| {
        Span::styled(
            key.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    let mut spans = vec![keycap("↑↓"), label(" move")];
    if multi_select {
        spans.push(gap());
        spans.push(keycap("space"));
        spans.push(label(" toggle"));
        spans.push(gap());
        spans.push(keycap("↵"));
        spans.push(label(" confirm"));
    } else {
        spans.push(gap());
        spans.push(keycap("↵"));
        spans.push(label(" select"));
        if allow_free_text {
            spans.push(gap());
            spans.push(label("type your own"));
        }
    }
    spans.push(gap());
    spans.push(keycap("esc"));
    spans.push(label(" dismiss"));
    Line::from(spans)
}

/// The plan-approval card's key-hint row.
fn plan_approval_keys_line() -> Line<'static> {
    let keycap = |key: &str| {
        Span::styled(
            key.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    Line::from(vec![
        keycap("↑↓"),
        label(" choose"),
        gap(),
        keycap("↵"),
        label(" confirm"),
        gap(),
        keycap("⇞⇟"),
        label(" scroll"),
        gap(),
        label("type feedback"),
        gap(),
        keycap("esc"),
        label(" dismiss"),
    ])
}

/// The edit-review card's key-hint row.
fn review_keys_line() -> Line<'static> {
    let keycap = |key: &str| {
        Span::styled(
            key.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    Line::from(vec![
        keycap("y"),
        label(" approve"),
        gap(),
        keycap("n"),
        label(" reject"),
        gap(),
        keycap("↑↓"),
        label(" scroll"),
        gap(),
        keycap("esc"),
        label(" reject"),
    ])
}

#[cfg(test)]
mod render_impl_tests {
    use super::scrub_control_cells;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn scrub_replaces_control_cells_with_spaces() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        buf.cell_mut((0, 0)).unwrap().set_symbol("a");
        buf.cell_mut((1, 0)).unwrap().set_symbol("\t");
        buf.cell_mut((2, 0)).unwrap().set_symbol("\u{1b}");
        buf.cell_mut((3, 0)).unwrap().set_symbol("界");
        scrub_control_cells(&mut buf);
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "a");
        assert_eq!(buf.cell((1, 0)).unwrap().symbol(), " ");
        assert_eq!(buf.cell((2, 0)).unwrap().symbol(), " ");
        // Non-control symbols (incl. wide chars) are untouched.
        assert_eq!(buf.cell((3, 0)).unwrap().symbol(), "界");
    }
}
