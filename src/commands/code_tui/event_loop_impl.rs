use super::*;
use crate::services::route_cache::PersistedRoute;

impl CodeTuiApp {
    /// Drains queued runtime events; `true` if any were handled (caller repaints).
    pub(super) async fn handle_runtime_events(&mut self) -> Result<bool> {
        let mut handled = false;
        while let Ok(event) = self.rx.try_recv() {
            self.handle_runtime_event(event).await?;
            handled = true;
        }
        if handled {
            // Any runtime event is proof of life for the stall detector.
            self.last_stream_activity = Some(Instant::now());
        }
        Ok(handled)
    }

    async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<()> {
        match event {
            RuntimeEvent::Delta(delta) => self.apply_runtime_delta(delta),
            RuntimeEvent::Finished { result, format } => {
                // Hold the finish until the typewriter has revealed the whole
                // reply, so the tail types out instead of snapping in.
                if self.incoming_buffer.is_empty() {
                    self.finish_response(result, format).await?;
                } else {
                    self.pending_finish = Some(DeferredFinish::Chat { result, format });
                }
            }
            RuntimeEvent::ModelsLoaded(result) => {
                self.apply_loaded_models(result).await?;
            }
            RuntimeEvent::CatalogWarmed => {
                self.refresh_context_window().await;
            }
            RuntimeEvent::ResumeLoaded { request_id, result } => {
                self.apply_resume_load_result(request_id, result).await?;
            }
            RuntimeEvent::SessionPreviewLoaded { session_id, entry } => {
                // Always cache — an entry for a no-longer-selected row is still correct.
                if self.session_preview_cache.len() >= PREVIEW_CACHE_CAP {
                    self.session_preview_cache.clear();
                }
                self.session_preview_cache.insert(session_id.clone(), entry);
                if self
                    .session_preview_task
                    .as_ref()
                    .is_some_and(|(sid, _)| *sid == session_id)
                {
                    self.session_preview_task = None;
                }
            }
            RuntimeEvent::CursorSessionOpened(session) => {
                // Defensive: a /new or /key switch could land between the task
                // starting `open()` and this event arriving. Only adopt the
                // session if we still need one for the current cursor key.
                if self.key.is_cursor_acp() && self.cursor_acp_session.is_none() {
                    self.cursor_acp_session = Some(session);
                }
            }
            RuntimeEvent::AgentContext { tokens, measured } => {
                self.apply_agent_context(tokens, measured);
            }
            RuntimeEvent::AgentTurnTokens(output) => {
                self.turn_output_tokens = output;
            }
            RuntimeEvent::AgentToolCall {
                id,
                name,
                args,
                line_starts,
                old_content,
            } => self.apply_agent_tool_call(id, name, args, line_starts, old_content),
            RuntimeEvent::AgentSubActivity {
                agent,
                tool,
                args,
                step,
            } => self.apply_subagent_activity(agent, tool, args, step),
            RuntimeEvent::AgentSubBegin { labels } => self.apply_subagent_begin(labels),
            RuntimeEvent::AgentSubSlot {
                slot,
                agent,
                tool,
                args,
                step,
            } => self.apply_subagent_slot(slot, agent, tool, args, step),
            RuntimeEvent::AgentSubDenied { slot, tool } => self.apply_subagent_denied(slot, tool),
            RuntimeEvent::AgentSubDone {
                slot,
                ok,
                steps,
                tokens,
            } => {
                if let Some(agent) = self.apply_subagent_done(slot, ok, steps, tokens) {
                    self.record_agent_run(agent, ok, steps, tokens);
                }
            }
            RuntimeEvent::AgentSubFinish => self.subagent_rows.clear(),
            RuntimeEvent::AgentToolUpdate {
                id,
                args,
                result,
                failed,
            } => self.apply_agent_tool_update(id, args, result, failed),
            RuntimeEvent::AgentToolOutput { chunk } => self.push_tool_output(&chunk),
            RuntimeEvent::AgentToolResult { content } => self.apply_agent_tool_result(content),
            RuntimeEvent::AgentSteered(text) => self.apply_agent_steered(text),
            RuntimeEvent::AgentDiscardSegment => self.discard_streamed_segment(),
            RuntimeEvent::McpConnected { client, generation } => {
                // Drop a connect that started before a `/mcp` toggle changed the
                // server set; only the current generation's result is applied.
                if generation == self.mcp_connect_gen {
                    self.apply_mcp_connected(client);
                }
            }
            RuntimeEvent::McpServerProgress {
                name,
                status,
                health,
                generation,
            } => {
                // One server resolved mid-connect: stash its status and repaint the
                // open overlay so that row flips now. Stale-generation events (a
                // connect superseded by a toggle) are dropped.
                if generation == self.mcp_connect_gen {
                    self.mcp_connect_progress.insert(name, (status, health));
                    self.refresh_mcp_overlay_status();
                }
            }
            RuntimeEvent::McpAuthorizeUrl { url } => {
                self.notice = Some((MUTED, format!("Authorize in your browser: {url}")));
            }
            RuntimeEvent::McpAuthorized { name, result } => match result {
                Ok(cred) => match crate::services::mcp_token_store::save(&name, &cred).await {
                    Ok(()) => {
                        self.notice = Some((MUTED, format!("Authorized `{name}` — reconnecting…")));
                        // Reconnect so the now-authorized server's tools appear
                        // (live if the /mcp overlay is open, else next turn).
                        self.reset_mcp_after_config_change();
                        self.restart_mcp_connect_for_overlay();
                    }
                    Err(e) => {
                        self.notice = Some((
                            ERROR,
                            format!("Authorized `{name}` but couldn't save the token: {e}"),
                        ));
                    }
                },
                Err(e) => {
                    self.notice = Some((ERROR, format!("Authorization for `{name}` failed: {e}")));
                }
            },
            RuntimeEvent::AgentPlan(items) => self.apply_agent_plan(items),
            RuntimeEvent::AgentNotice(text) => {
                // A connection-retry notice means we're recovering, not thinking.
                self.retrying = text.contains("retrying");
                self.notice = Some((MUTED, text));
            }
            // Typed early-stop (guard stop / step limit) — remembered for /goal steering.
            RuntimeEvent::AgentTurnStop(stop) => self.goal_guard_stop = Some(stop),
            RuntimeEvent::AgentError(text) => self.apply_agent_error(text),
            RuntimeEvent::AgentPermission {
                tool,
                preview,
                reply,
            } => {
                self.agent_permission = Some(PendingPermission {
                    tool,
                    preview,
                    reply,
                });
                // The card floats above the composer (drawn every frame
                // regardless of scroll), so don't yank the transcript to the
                // bottom — a user reading earlier output keeps their place.
            }
            RuntimeEvent::AgentSwitchModel { model, reply } => {
                let result = self.agent_switch_model(model).await;
                let _ = reply.send(result);
            }
            RuntimeEvent::AgentSetEffort { level, reply } => {
                let result = self.agent_set_effort(level).await;
                let _ = reply.send(result);
            }
            RuntimeEvent::AgentAskUser {
                question,
                options,
                allow_free_text,
                multi_select,
                reply,
            } => {
                let checked = vec![false; options.len()];
                self.agent_ask = Some(PendingAskUser {
                    question,
                    options,
                    allow_free_text,
                    multi_select,
                    checked,
                    selected: 0,
                    reply,
                });
            }
            RuntimeEvent::AgentReviewEdits { items, reply } => {
                // Precompute the diff body now (has the live cwd) so rendering stays pure.
                let cwd = if self.real_cwd.is_empty() {
                    self.cwd.clone()
                } else {
                    self.real_cwd.clone()
                };
                let body = super::render::review_body_lines(&items, std::path::Path::new(&cwd));
                self.agent_review = Some(PendingReview {
                    count: items.len(),
                    body,
                    scroll: 0,
                    reply,
                });
            }
            RuntimeEvent::AgentPlanApproval { plan, reply } => {
                // Precompute the rendered plan so the frame render stays pure.
                let width = self.transcript_width.saturating_sub(6).max(40);
                let body: Vec<ratatui::text::Line<'static>> =
                    super::render::render_markdown_lines(&plan, width)
                        .into_iter()
                        .map(|l| l.line)
                        .collect();
                self.agent_plan_approval = Some(PendingPlanApproval {
                    body,
                    scroll: 0,
                    selected: 0,
                    reply,
                });
            }
            RuntimeEvent::AgentFinished {
                steps,
                tokens,
                context_tokens,
            } => {
                // Same deferral as the chat path: let the final assistant text
                // finish typing out before the turn commits.
                if self.incoming_buffer.is_empty() {
                    self.finish_agent_turn(steps, tokens, context_tokens)
                        .await?;
                } else {
                    self.pending_finish = Some(DeferredFinish::Agent {
                        steps,
                        tokens,
                        context_tokens,
                    });
                }
            }
            RuntimeEvent::LocalCommandLine { is_err, line } => {
                self.apply_local_command_line(is_err, line)
            }
            RuntimeEvent::LocalCommandDone {
                exit_code,
                truncated,
            } => self.finish_local_command(exit_code, truncated).await?,
            RuntimeEvent::SkillInstalled {
                source,
                project,
                result,
            } => self.apply_skill_installed(source, project, result).await?,
            RuntimeEvent::SkillInstallPick {
                source,
                project,
                staged,
            } => {
                self.apply_skill_install_pick(source, project, staged)
                    .await?
            }
            RuntimeEvent::LiveShareReady { share_gen, result } => {
                self.apply_live_share_ready(share_gen, result)
            }
            RuntimeEvent::AccountLoginPrompt {
                account_gen,
                result,
            } => self.apply_account_login_prompt(account_gen, result),
            RuntimeEvent::AccountLoginDone {
                account_gen,
                result,
            } => self.apply_account_login_done(account_gen, result).await,
            RuntimeEvent::AccountLogoutDone {
                account_gen,
                result,
            } => self.apply_account_logout_done(account_gen, result).await,
        }
        Ok(())
    }

    /// Append one streamed `!cmd` output line to the in-progress run. The output
    /// lives on `local_command` (not history) while running, so it renders in the
    /// volatile transcript tail without busting the memoized history body.
    fn apply_local_command_line(&mut self, is_err: bool, line: String) {
        let Some(run) = self.local_command.as_mut() else {
            return;
        };
        let buf = if is_err {
            &mut run.stderr
        } else {
            &mut run.stdout
        };
        buf.push_str(&line);
        buf.push('\n');
    }

    /// Commit a finished `!cmd` run: stash its full output for the ctrl+o pager and
    /// push a bounded preview to history, then point the user at the pager when the
    /// transcript elided lines.
    async fn finish_local_command(&mut self, exit_code: i64, truncated: bool) -> Result<()> {
        let Some(run) = self.local_command.take() else {
            return Ok(());
        };
        let total = self.record_local_output(
            run.command,
            run.stdout,
            run.stderr,
            exit_code,
            truncated,
            false,
        );
        if truncated {
            self.notice = Some((
                MUTED,
                format!("Output truncated at the capture cap ({total} lines)"),
            ));
        }
        self.persist_history().await?;
        Ok(())
    }

    /// Stash a finished/interrupted `!cmd` run's full output in `local_outputs` keyed
    /// by history index, and push a bounded preview to history as a `local_command`
    /// entry, so the persisted session stays small while the output stays viewable.
    /// The true line count rides along as `total_lines` so "+N more" stays honest.
    /// Returns that total.
    pub(super) fn record_local_output(
        &mut self,
        command: String,
        stdout: String,
        stderr: String,
        exit_code: i64,
        truncated: bool,
        interrupted: bool,
    ) -> usize {
        let total = stdout.lines().count() + stderr.lines().count();

        let mut entry = serde_json::json!({
            "command": command.clone(),
            "stdout": first_lines(&stdout, MAX_PERSISTED_OUTPUT_LINES),
            "stderr": first_lines(&stderr, MAX_PERSISTED_OUTPUT_LINES),
            "exit_code": exit_code,
            "total_lines": total,
        });
        if truncated {
            entry["truncated"] = serde_json::Value::Bool(true);
        }
        if interrupted {
            entry["interrupted"] = serde_json::Value::Bool(true);
        }
        self.history.push(ChatMessage {
            model: None,
            role: "local_command".to_string(),
            content: entry.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });

        // Retain only what an expander can show (bounded by `MAX_EXPANDED_OUTPUT_LINES`)
        // — and only when there's more than the fold preview already reveals.
        if total > MAX_OUTPUT_LINES {
            let idx = self.history.len() - 1;
            self.local_outputs.insert(
                idx,
                LocalCommandOutput {
                    stdout: first_lines(&stdout, MAX_EXPANDED_OUTPUT_LINES),
                    stderr: first_lines(&stderr, MAX_EXPANDED_OUTPUT_LINES),
                },
            );
        }
        self.follow_output = true;
        total
    }

    /// Drop the in-flight segment's streamed text (typed + buffered) before it
    /// commits; the engine flagged it as a tool call written as text. Reasoning is
    /// left to commit with the retry's segment.
    pub(super) fn discard_streamed_segment(&mut self) {
        self.incoming_buffer.clear();
        self.pending_response.clear();
    }

    /// Commit any streamed assistant text into a history entry. Called before a
    /// tool step (so prose precedes the call) and at turn end.
    pub(super) fn flush_pending_assistant(&mut self) {
        // Reveal any buffered text before committing so a tool step never lands
        // ahead of prose the typewriter hadn't shown yet.
        self.drain_incoming_buffer();
        let content = std::mem::take(&mut self.pending_response);
        self.commit_assistant_segment(content);
    }

    /// Commit an assistant segment to history with its thinking duration (for the
    /// folded `▸ thought for Ns` summary). A reasoning-only segment (thought, then a
    /// tool call with no prose) still commits — the empty-content message renders
    /// just the thinking summary before the tool step. Resets the thinking clock so
    /// the next segment times from its own first reasoning.
    fn commit_assistant_segment(&mut self, content: String) {
        let reasoning_content = (!self.pending_reasoning.is_empty())
            .then(|| std::mem::take(&mut self.pending_reasoning));
        let duration_ms = reasoning_content.as_ref().and(self.segment_reasoning_ms());
        if !content.is_empty() || reasoning_content.is_some() {
            self.history.push(ChatMessage {
                model: self.turn_model.clone(),
                role: "assistant".to_string(),
                content,
                reasoning_content,
                attachments: vec![],
            });
            if let Some(ms) = duration_ms {
                self.reasoning_durations.insert(self.history.len() - 1, ms);
            }
        }
        self.reasoning_started_at = None;
        self.reasoning_elapsed_ms = None;
    }

    /// This segment's thinking duration (ms): the value frozen when the answer
    /// started, else the live elapsed since the first reasoning chunk. `None` if
    /// no reasoning has streamed this segment.
    pub(super) fn segment_reasoning_ms(&self) -> Option<u64> {
        self.reasoning_elapsed_ms.or_else(|| {
            self.reasoning_started_at
                .map(|t| t.elapsed().as_millis() as u64)
        })
    }

    /// Live context-fill from the agent engine. A measured step total flows
    /// through `live_usage` so the footer shows it exactly (no `~`, and without
    /// re-adding streamed text on top); a pre-usage estimate updates the baseline
    /// `context_tokens` so the footer's estimate counts the real request (system
    /// prompt + tool schemas), not just the visible transcript.
    pub(super) fn apply_agent_context(&mut self, tokens: u64, measured: bool) {
        if measured {
            self.live_usage = Some(TokenUsage {
                prompt_tokens: tokens,
                ..Default::default()
            });
        } else {
            self.live_usage = None;
            self.context_tokens = tokens;
        }
    }

    /// Drop the sandbox-escalation ack on the agent's next output. Scoped to that
    /// exact notice so an unrelated one sharing the slot survives.
    fn clear_sandbox_escalation_notice(&mut self) {
        if self
            .notice
            .as_ref()
            .is_some_and(|(_, text)| text == crate::agent::engine::SANDBOX_ESCALATION_NOTICE)
        {
            self.notice = None;
        }
    }

    /// Drop a "…retrying…" notice once the turn recovers or ends so it can't stay
    /// stuck. Scoped to retry text so a real error notice survives.
    fn clear_retry_notice(&mut self) {
        if self
            .notice
            .as_ref()
            .is_some_and(|(_, text)| text.contains("retrying"))
        {
            self.notice = None;
        }
    }

    pub(super) fn apply_agent_tool_call(
        &mut self,
        id: Option<String>,
        name: String,
        args: serde_json::Value,
        line_starts: Vec<Option<usize>>,
        old_content: Option<String>,
    ) {
        self.clear_sandbox_escalation_notice();
        self.flush_pending_assistant();
        self.clear_tool_output();
        // Stamp the status-line action label for the in-flight step.
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        // Timeout budget for the status clock (mirrors the engine's resolution).
        let deadline = (name == "run_bash").then(|| {
            args.get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(120)
                .min(600)
        });
        self.last_tool_action = Some((
            super::render::tool_action_label(&name, &args, &cwd),
            Instant::now(),
            deadline,
        ));
        let mut obj = serde_json::Map::new();
        obj.insert("name".to_string(), serde_json::Value::String(name.clone()));
        obj.insert("args".to_string(), args);
        if let Some(id) = id {
            obj.insert("id".to_string(), serde_json::Value::String(id));
        }
        // Carry the pre-edit line numbers so the diff card can number rows.
        if !line_starts.is_empty() {
            obj.insert("line_starts".to_string(), serde_json::json!(line_starts));
        }
        // Carry a write_file's pre-write snapshot so the card shows a real diff.
        if let Some(old) = old_content.filter(|o| !o.is_empty()) {
            obj.insert("old_content".to_string(), serde_json::Value::String(old));
        }
        let content = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or(name);
        self.history.push(ChatMessage {
            model: None,
            role: "tool_call".to_string(),
            content,
            reasoning_content: None,
            attachments: vec![],
        });
        // Don't force-follow: if the user scrolled up to read earlier output,
        // a streamed tool step shouldn't yank the view back to the bottom. The
        // render already follows new output while `follow_output` is set, and
        // scrolling back to the bottom re-arms it.
    }

    /// Nested sub-agent step → the status-line label only; the parent's own
    /// `subagent` tool-call card still owns the transcript.
    pub(super) fn apply_subagent_activity(
        &mut self,
        agent: String,
        tool: String,
        args: serde_json::Value,
        step: usize,
    ) {
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        let who: &str = if agent.is_empty() { "subagent" } else { &agent };
        // Empty tool = child thinking between calls.
        let inner = if tool.is_empty() {
            "working".to_string()
        } else {
            super::render::tool_action_label(&tool, &args, &cwd)
        };
        let label = if step > 0 {
            format!("↳ {who}: {inner} · step {step}")
        } else {
            format!("↳ {who}: {inner}")
        };
        self.last_tool_action = Some((label, Instant::now(), None));
    }

    /// Unnamed delegates get numbered so every row stays distinguishable.
    pub(super) fn apply_subagent_begin(&mut self, labels: Vec<String>) {
        let now = Instant::now();
        self.subagent_rows = labels
            .into_iter()
            .enumerate()
            .map(|(i, name)| super::shared::SubagentRow {
                name: if name.is_empty() {
                    format!("sub-agent {}", i + 1)
                } else {
                    name
                },
                action: "starting".to_string(),
                step: 0,
                started: now,
                denied: None,
                done: None,
            })
            .collect();
    }

    /// The action label is precomputed here so rendering stays pure.
    pub(super) fn apply_subagent_slot(
        &mut self,
        slot: usize,
        agent: String,
        tool: String,
        args: serde_json::Value,
        step: usize,
    ) {
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        let Some(row) = self.subagent_rows.get_mut(slot) else {
            return;
        };
        if !agent.is_empty() {
            row.name = agent;
        }
        row.action = if tool.is_empty() {
            "working".to_string()
        } else {
            super::render::tool_action_label(&tool, &args, &cwd)
        };
        row.step = step;
    }

    /// Mark the row and explain in a notice, so the deny doesn't read as
    /// delegate sloppiness.
    pub(super) fn apply_subagent_denied(&mut self, slot: usize, tool: String) {
        let Some(row) = self.subagent_rows.get_mut(slot) else {
            return;
        };
        let name = row.name.clone();
        row.denied = Some(tool.clone());
        self.notice = Some((
            WARNING,
            format!(
                "{name}: `{tool}` auto-denied — parallel delegates can't prompt; run it solo to approve"
            ),
        ));
    }

    /// Returns the profile name to attribute in lifetime stats when the finished
    /// row names a discovered subagent; generic/labeled delegates return `None`
    /// so they never pollute per-agent stats.
    pub(super) fn apply_subagent_done(
        &mut self,
        slot: usize,
        ok: bool,
        steps: usize,
        tokens: u64,
    ) -> Option<String> {
        let row = self.subagent_rows.get_mut(slot)?;
        row.done = Some((ok, steps, tokens, row.started.elapsed()));
        let name = row.name.clone();
        self.last_subagents
            .iter()
            .any(|s| s.name == name)
            .then_some(name)
    }

    /// Persist one finished named delegation under the active key. Fire-and-forget:
    /// a stats write must never block or fail the turn.
    fn record_agent_run(&self, agent: String, ok: bool, steps: usize, tokens: u64) {
        let session_store = self.session_store.clone();
        let key_id = self.key.id.clone();
        tokio::spawn(async move {
            let _ = session_store
                .record_agent_run(&key_id, &agent, ok, steps as u64, tokens)
                .await;
        });
    }

    /// Enrich the matching tool-call entry in place (cursor reports the resolved
    /// path/pattern and the result in a later `tool_call_update`, keyed by id):
    /// swap in the real args and attach a compact result / failed flag. Bumps the
    /// transcript revision so the memoized body re-renders.
    pub(super) fn apply_agent_tool_update(
        &mut self,
        id: String,
        args: Option<serde_json::Value>,
        result: Option<String>,
        failed: bool,
    ) {
        let Some(entry) = self.history.iter_mut().rev().find(|m| {
            m.role == "tool_call"
                && serde_json::from_str::<serde_json::Value>(&m.content)
                    .ok()
                    .and_then(|v| v.get("id").and_then(|x| x.as_str()).map(str::to_string))
                    .as_deref()
                    == Some(id.as_str())
        }) else {
            return;
        };
        let mut obj = serde_json::from_str::<serde_json::Value>(&entry.content)
            .unwrap_or_else(|_| serde_json::json!({}));
        let args_updated = args.is_some();
        if let Some(args) = args {
            obj["args"] = args;
        }
        if let Some(result) = result {
            obj["result"] = serde_json::Value::String(result);
        }
        if failed {
            obj["failed"] = serde_json::Value::Bool(true);
        }
        entry.content = obj.to_string();
        // Enriched args on a live call refresh the status label too
        // (a `cursor/task` notice delivers the real task after the call frame).
        let relabel = (args_updated && decode_tool_outcome(&entry.content).0.is_none() && !failed)
            .then(|| super::render::decode_tool_call(&entry.content));
        if let Some((name, new_args)) = relabel {
            let cwd = if self.real_cwd.is_empty() {
                self.cwd.clone()
            } else {
                self.real_cwd.clone()
            };
            if let Some((label, _, _)) = self.last_tool_action.as_mut() {
                *label = super::render::tool_action_label(&name, &new_args, &cwd);
            }
        }
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
    }

    /// A background MCP connect resolved: cache the client and, if it brought
    /// tools, arrange for the engine to advertise them. The engine is rebuilt (not
    /// mutated in place) because it's behind a mutex and may be mid-turn; rebuild
    /// re-seeds from history, so the conversation survives. Deferred until the
    /// turn finishes when one is in flight.
    pub(super) fn apply_mcp_connected(
        &mut self,
        client: std::sync::Arc<crate::agent::mcp::McpClient>,
    ) {
        self.mcp_connecting = false;
        // The full client now answers every status query; the interim per-server
        // map is superseded.
        self.mcp_connect_progress.clear();
        // Surface connect failures so a mis-configured server isn't a silent no-op.
        // Don't raise a scary "failed" notice for servers that merely need OAuth
        // — the /mcp roster shows those as "needs authorization", and a freshly
        // added one auto-authorizes below. Config-file parse errors (whose source
        // is a filename, not a server) and genuine failures still surface.
        let hard_errors: Vec<&(String, String)> = client
            .errors()
            .iter()
            .filter(|(source, _)| !client.needs_auth(source))
            .collect();
        if let Some((source, reason)) = hard_errors.first() {
            let msg = if hard_errors.len() == 1 {
                format!("MCP server failed — {source}: {reason}")
            } else {
                format!(
                    "{} MCP servers failed — {source}: {reason}; …",
                    hard_errors.len()
                )
            };
            self.notice = Some((WARNING, msg));
        }
        let has_tools = client.has_tools();
        self.mcp_client = Some(client);
        // If the `/mcp` overlay is open, refresh its rows from the now-resolved
        // client so each "connecting…" flips to the real tool count or failure
        // live (done before the no-tools early-return so failures still update).
        self.refresh_mcp_overlay_status();
        // A freshly-added HTTP server that came back 401 auto-starts its OAuth
        // flow (one-step add), so the browser opens without a separate Ctrl+O.
        // One that connected fine — or failed for another reason — is just
        // dropped from the queue. Done before the no-tools return: a needs-auth
        // server has no tools.
        if !self.pending_mcp_auth.is_empty() {
            let pending = std::mem::take(&mut self.pending_mcp_auth);
            let to_auth: Vec<(String, String)> = pending
                .into_iter()
                .filter(|(name, _)| self.mcp_client.as_ref().is_some_and(|c| c.needs_auth(name)))
                .collect();
            for (name, url) in to_auth {
                self.start_mcp_authorize(name, url);
            }
        }
        if !has_tools {
            return; // no servers / no mcp.json — nothing to attach
        }
        self.request_engine_rebuild();
    }

    /// Drop the engine so the next turn rebuilds with the changed tool set —
    /// deferred mid-turn (an immediate drop skips `take_turn_usage` + persist).
    pub(super) fn request_engine_rebuild(&mut self) {
        if self.sending {
            self.engine_rebuild_pending = true;
        } else {
            self.agent_engine = None;
        }
    }

    /// The deferred half of [`Self::request_engine_rebuild`], run after a turn
    /// finishes.
    pub(super) fn maybe_apply_engine_rebuild(&mut self) {
        if self.engine_rebuild_pending {
            self.engine_rebuild_pending = false;
            self.agent_engine = None;
        }
    }

    /// A turn-level agent error: the transient notice, plus a durable `error`
    /// transcript entry — else a failed turn reads as a prompt with no reply
    /// after scrolling or resume. Display-only; `agent_seed_turns` skips it.
    pub(super) fn apply_agent_error(&mut self, text: String) {
        self.flush_pending_assistant();
        self.notice = Some((ERROR, text.clone()));
        self.history.push(ChatMessage {
            model: None,
            role: "error".to_string(),
            content: text,
            reasoning_content: None,
            attachments: vec![],
        });
    }

    pub(super) fn apply_agent_tool_result(&mut self, content: String) {
        self.clear_tool_output();
        self.history.push(ChatMessage {
            model: None,
            role: "tool_result".to_string(),
            content,
            reasoning_content: None,
            attachments: vec![],
        });
        // Same as the tool-call append: leave `follow_output` alone so a user
        // reading scrolled-up output isn't snapped to the bottom each step.
    }

    /// Fold a live output chunk into the tail; a bare \r (progress redraw)
    /// counts as a line break.
    pub(super) fn push_tool_output(&mut self, chunk: &str) {
        const TAIL_LINES: usize = super::render::STREAM_TAIL_LINES;
        const PARTIAL_CAP: usize = 512;
        self.tool_output_partial.push_str(chunk);
        while let Some(pos) = self.tool_output_partial.find(['\n', '\r']) {
            let rest = self.tool_output_partial.split_off(pos + 1);
            let line = std::mem::replace(&mut self.tool_output_partial, rest);
            let line = line.trim_end_matches(['\n', '\r']);
            if !line.trim().is_empty() {
                if self.tool_output_tail.len() >= TAIL_LINES {
                    self.tool_output_tail.pop_front();
                }
                self.tool_output_tail.push_back(line.to_string());
            }
        }
        if self.tool_output_partial.len() > PARTIAL_CAP {
            let mut cut = PARTIAL_CAP;
            while !self.tool_output_partial.is_char_boundary(cut) {
                cut -= 1;
            }
            self.tool_output_partial.truncate(cut);
        }
    }

    pub(super) fn clear_tool_output(&mut self) {
        self.tool_output_tail.clear();
        self.tool_output_partial.clear();
    }

    /// Commit a consumed interjection at its injection point (events arrive in
    /// engine order).
    pub(super) fn apply_agent_steered(&mut self, text: String) {
        self.flush_pending_assistant();
        self.history.push(ChatMessage {
            model: None,
            role: "user".to_string(),
            content: text,
            reasoning_content: None,
            attachments: vec![],
        });
        self.notice = Some((MUTED, "Interjection delivered".to_string()));
    }

    /// Render an `update_plan` call as a SINGLE checklist card. The model resends
    /// the full plan on every call, so the transcript keeps just one card: each
    /// update drops the previous one and re-appends the latest at the current
    /// point of work. This keeps the plan current and near the live cursor instead
    /// of stacking a near-identical copy after every batch of tool calls.
    pub(super) fn apply_agent_plan(&mut self, items: serde_json::Value) {
        self.flush_pending_assistant();
        let content = items.to_string();
        // Drop the prior card (with index-map fixup), re-append the latest below.
        self.drop_plan_entries();
        self.history.push(ChatMessage {
            model: None,
            role: "plan".to_string(),
            content,
            reasoning_content: None,
            attachments: vec![],
        });
        // Removing the prior card can leave history length and the last entry
        // unchanged (e.g. a status-only edit), so bump the revision unconditionally
        // to invalidate the transcript render cache.
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        // Don't force-follow on a plan refresh; respect the user's scroll position.
    }

    /// Pricing for the session's model, else the upstream-echoed `billed_model`
    /// (an alias like `aivo/starter` has no snapshot row).
    pub(super) fn session_pricing(&self) -> Option<crate::services::model_metadata::Pricing> {
        crate::services::model_metadata::model_pricing(&self.model).or_else(|| {
            self.billed_model
                .as_deref()
                .and_then(crate::services::model_metadata::model_pricing)
        })
    }

    async fn finish_agent_turn(
        &mut self,
        _steps: usize,
        tokens: u64,
        context_tokens: u64,
    ) -> Result<()> {
        self.flush_pending_assistant();
        // A `/compact` turn has no assistant reply: report freed context, not a marker.
        let compact_before = self.compact_before.take();
        // Marker index, so the per-turn usage below can note tokens/cost on it.
        let mut done_idx = None;
        if let Some(before) = compact_before {
            let freed = before.saturating_sub(context_tokens) as usize;
            self.notice = Some(freed_notice(freed, "summarized older turns"));
        } else {
            // `✶ Done in …` marker — skipped under 1s and on an errored turn. Attach
            // to the last VISIBLE entry: a trailing plan renders in its own panel, so
            // stamping it there hides/misplaces the marker once the plan clears.
            let errored = self.notice.as_ref().is_some_and(|(c, _)| *c == ERROR);
            if let Some(started) = self.request_started_at
                && !errored
            {
                let elapsed = started.elapsed();
                if elapsed.as_secs() >= 1
                    && let Some(idx) = self.history.iter().rposition(|m| m.role != "plan")
                {
                    self.turn_durations.insert(idx, elapsed.as_millis() as u64);
                    done_idx = Some(idx);
                }
            }
        }
        self.retrying = false;
        // A retry that recovered on the final step has no later chunk to clear it.
        self.clear_retry_notice();
        self.sending = false;
        self.subagent_rows.clear();
        self.clear_tool_output();
        self.request_started_at = None;
        self.response_task = None;
        self.pending_submit = None;
        self.agent_permission = None;
        self.agent_ask = None;
        self.agent_review = None;
        self.agent_plan_approval = None;
        self.stop_agent_serve();
        // Adopt + persist the protocol the serve negotiated this turn.
        self.persist_agent_route().await;
        // A compact reports a calibrated estimate; a real turn prefers the provider-
        // measured fill, falling back to the chars/4 estimate when no usage was reported.
        if compact_before.is_some() {
            self.context_tokens = context_tokens;
            self.context_is_estimate = true;
        } else if context_tokens > 0 {
            self.context_tokens = context_tokens;
            self.context_is_estimate = false;
        } else {
            self.context_tokens = self.estimated_context_used().await;
            self.context_is_estimate = true;
        }
        self.last_usage = None;
        self.live_usage = None;
        // Fold this turn's real provider-measured split into the session's running
        // total BEFORE a possible MCP rebuild drops the engine, so the chat index
        // entry (and thus `aivo stats --since`) carries actual chat tokens.
        if let Some(session) = self.agent_engine.as_ref() {
            let (turn, reported_cost, billed) = {
                let mut engine = session.engine.lock().await;
                (
                    engine.take_turn_usage(),
                    engine.take_turn_cost_usd(),
                    engine.billed_model().map(str::to_string),
                )
            };
            if billed.is_some() {
                self.billed_model = billed;
            }
            // Provider-reported spend beats list-price × usage when available.
            let cost =
                reported_cost.or_else(|| self.session_pricing().and_then(|p| p.cost_usd(&turn)));
            if let Some(cost) = cost {
                self.session_cost_usd += cost;
            }
            // The spinner's live token count vanishes at turn end — keep the
            // final figure on the `✶ Done` line.
            if let Some(idx) = done_idx
                && turn.completion_tokens > 0
            {
                let note = format!(
                    "{} tokens",
                    format_token_count_value(turn.completion_tokens)
                );
                self.turn_notes.insert(idx, note);
            }
            self.session_tokens = self.session_tokens.merge(turn);
        }
        // If the tool set changed mid-turn, drop the engine now so the next turn
        // rebuilds with it (must happen while not sending).
        self.maybe_apply_engine_rebuild();
        self.persist_history().await?;
        // A compact adds no user/assistant message; logging would duplicate the prior row.
        if compact_before.is_none() {
            self.log_agent_turn(tokens).await;
        }
        // Pick up skills created/edited during the turn (e.g. via `/create-skill`):
        // refresh the `/` menu and, if the set changed, rebuild the engine next turn
        // so the model sees the new skills. Runs while not sending, so the engine
        // reset stays lossless.
        self.refresh_skill_commands().await;
        // The turn is over — apply a deferred plan-discard tool restore.
        if self.plan_exit_pending {
            if let Some(session) = self.agent_engine.as_ref() {
                session.engine.lock().await.set_plan_mode(false);
            }
            self.plan_exit_pending = false;
        }
        // Before a queued message can flip `sending` and skip the capture.
        self.capture_plan_draft();
        self.reclaim_unsent_steering();
        // Commands queued mid-turn run first (a queued `/plan go` needs the plan
        // captured above; a queued `/compact` should fold before the next message).
        self.drain_queued_commands().await;
        self.drain_queued_message().await?;
        // Autonomous `/goal` loop: if active (and a queued message didn't already
        // start the next turn), continue toward the goal or stop on completion/cap.
        self.maybe_continue_goal().await?;
        Ok(())
    }

    /// Record the agent turn in `aivo logs`. The per-turn loopback serve only
    /// logs low-level, cwd-less `serve_request` rows; this adds the same
    /// `chat_turn` entry the HTTP path writes (prompt title, conversation body)
    /// under the real cwd so the turn shows in the project's logs. The accurate
    /// per-tool token split lives in `aivo stats` (the serve's usage accounting);
    /// here we only have the turn total.
    pub(super) async fn log_agent_turn(&self, tokens: u64) {
        let Some(user_message) = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .cloned()
        else {
            return;
        };
        let assistant_content = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let usage = TokenUsage {
            completion_tokens: tokens,
            ..Default::default()
        };
        let _ = log_chat_turn(
            &self.session_store,
            &self.key,
            &self.raw_model,
            Some(self.persist_cwd()),
            Some(&self.session_id),
            &user_message,
            &assistant_content,
            None,
            &usage,
        )
        .await;
    }

    fn apply_runtime_delta(&mut self, delta: ChatResponseChunk) {
        // Any chunk is progress — a prior connection retry has recovered.
        self.retrying = false;
        self.clear_retry_notice();
        match delta {
            // Buffer the chunk; `tick_typewriter` reveals it into the displayed
            // reply over the next frames so output reads as fast typing instead
            // of arriving in network-sized bursts.
            ChatResponseChunk::Content(text) => {
                // Answer starting ends this segment's thinking — freeze the duration
                // so the folded `▸ thought for Ns` excludes answer-streaming time.
                if self.reasoning_started_at.is_some() && self.reasoning_elapsed_ms.is_none() {
                    self.reasoning_elapsed_ms = self.segment_reasoning_ms();
                }
                self.clear_sandbox_escalation_notice();
                self.incoming_buffer.push_str(&text);
            }
            // Live provider-measured usage — the footer's context-fill reads this
            // while `sending` so the stat grows during the turn, not just at the end.
            ChatResponseChunk::Usage(usage) => {
                // Plain chat is a single round, so its completion IS the turn output.
                self.turn_output_tokens = usage.completion_tokens;
                self.live_usage = Some(usage);
            }
            // Accumulate the model's reasoning unconditionally; `thinking_enabled`
            // gates only the *render* (so toggling /config reveals/hides it
            // instantly). Committed onto the assistant message at turn end. The
            // first chunk starts this segment's thinking clock.
            ChatResponseChunk::Reasoning(text) => {
                if self.reasoning_started_at.is_none() {
                    self.reasoning_started_at = Some(Instant::now());
                }
                self.pending_reasoning.push_str(&text);
            }
        }
    }

    /// Reveals the next slice of buffered stream text into the displayed reply.
    /// Returns true if anything was revealed (caller repaints). Paced by the
    /// animating frame cadence; see [`TYPEWRITER_MIN_CHARS`] for the rate.
    pub(super) fn tick_typewriter(&mut self) -> bool {
        if self.incoming_buffer.is_empty() {
            return false;
        }
        let remaining = self.incoming_buffer.chars().count();
        let step = TYPEWRITER_MIN_CHARS
            .max(remaining / TYPEWRITER_CATCHUP_DIVISOR)
            .min(remaining);
        // Cut on a char boundary so multi-byte glyphs are never split.
        let byte_idx = self
            .incoming_buffer
            .char_indices()
            .nth(step)
            .map_or(self.incoming_buffer.len(), |(idx, _)| idx);
        let revealed: String = self.incoming_buffer.drain(..byte_idx).collect();
        self.pending_response.push_str(&revealed);
        true
    }

    /// Reveals all remaining buffered text at once. Used when a boundary needs
    /// the full reply now — committing a turn, a tool step, an interrupt, or
    /// exit — so no received text is lost or left to type out of order.
    pub(super) fn drain_incoming_buffer(&mut self) {
        if !self.incoming_buffer.is_empty() {
            let rest = std::mem::take(&mut self.incoming_buffer);
            self.pending_response.push_str(&rest);
        }
    }

    /// Runs a finish event that was deferred until the typewriter caught up,
    /// once the buffer is empty. Returns true if a finish ran (caller repaints).
    pub(super) async fn run_deferred_finish_if_ready(&mut self) -> Result<bool> {
        if !self.incoming_buffer.is_empty() {
            return Ok(false);
        }
        match self.pending_finish.take() {
            Some(DeferredFinish::Chat { result, format }) => {
                self.finish_response(result, format).await?;
                Ok(true)
            }
            Some(DeferredFinish::Agent {
                steps,
                tokens,
                context_tokens,
            }) => {
                self.finish_agent_turn(steps, tokens, context_tokens)
                    .await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Watchdog for a response task that ended WITHOUT delivering its terminal
    /// event: a panic in `run_turn` (or the chat/cursor task) never sends
    /// `AgentFinished`/`Finished`, leaving `sending` stuck true forever — a frozen
    /// composer and a stalled `/goal` loop. Salvage partial text, reset the turn,
    /// and surface the failure. Returns true if it recovered a dead turn.
    pub(super) async fn recover_dead_response_task(&mut self) -> Result<bool> {
        // Only suspect a still-sending turn whose finish isn't deferred behind
        // the typewriter.
        if !self.sending || self.pending_finish.is_some() || !self.incoming_buffer.is_empty() {
            return Ok(false);
        }
        if !self.response_task.as_ref().is_some_and(|t| t.is_finished()) {
            return Ok(false);
        }
        // A normal finish sends its terminal event just BEFORE the task returns,
        // so drain once more — don't mistake a clean finish for a crash.
        self.handle_runtime_events().await?;
        if !self.sending {
            return Ok(false);
        }
        // Still sending with a finished task ⇒ it died without finishing.
        let Some(task) = self.response_task.take() else {
            return Ok(false);
        };
        let detail = match task.await {
            Err(e) if e.is_panic() => "the agent turn crashed",
            _ => "the agent turn ended unexpectedly",
        };
        self.drain_incoming_buffer();
        let partial = std::mem::take(&mut self.pending_response);
        self.commit_assistant_segment(partial);
        // Reset the turn, fail-closed like an interrupt.
        self.sending = false;
        self.subagent_rows.clear();
        self.clear_tool_output();
        self.request_started_at = None;
        self.pending_submit = None;
        // A dead compact turn must not mark the NEXT turn as a compact.
        self.compact_before = None;
        self.agent_permission = None;
        self.agent_ask = None;
        self.agent_review = None;
        self.agent_plan_approval = None;
        self.queued_messages.clear();
        self.queued_commands.clear();
        self.stop_agent_serve();
        self.follow_output = true;
        // A crash mid-loop must NOT auto-continue the goal into a likely repeat.
        let goal_stopped = self.goal_mode.take().is_some();
        let mut msg = detail.to_string();
        if goal_stopped {
            msg.push_str(" — goal mode stopped");
        }
        self.notice = Some((ERROR, msg));
        if !self.history.is_empty() {
            let _ = self.persist_history().await;
        }
        Ok(true)
    }

    async fn finish_response(
        &mut self,
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    ) -> Result<()> {
        self.sending = false;
        self.subagent_rows.clear();
        self.clear_tool_output();
        self.request_started_at = None;
        self.response_task = None;
        // The turn's format belongs to the model that ran it — don't clobber a
        // mid-turn /model switch's freshly seeded format.
        if self
            .turn_model
            .as_deref()
            .is_none_or(|m| m == self.raw_model)
        {
            self.format = format;
        }

        match result {
            Ok(turn) => self.finish_successful_response(turn).await?,
            Err(err) => self.finish_failed_response(err),
        }

        // Keep the `/` menu in sync with any skills added/edited during the turn
        // (parity with the agent path's `finish_agent_turn`).
        self.refresh_skill_commands().await;
        self.drain_queued_commands().await;
        self.drain_queued_message().await?;
        Ok(())
    }

    /// After an agent turn: point `self.format` at the wire the serve proved for
    /// the current model (so an attachment turn's plain-chat fallback follows it),
    /// then persist every confirmed route the shared cache learned — including
    /// subagent models — so they survive across launches.
    async fn persist_agent_route(&mut self) {
        let cache = match &self.agent_route_cache {
            Some((key_id, cache)) if *key_id == self.key.id => cache.clone(),
            _ => return,
        };
        let slot = cache.resolve(&self.model);
        if slot.is_confirmed() {
            self.format = chat_format_from_protocol(slot.current().0);
        }
        // Skip routes already on the key so a long session doesn't rewrite config.
        let tool = cache.tool();
        let stored = self.key.routes_for_tool(tool);
        let fresh: Vec<_> = cache
            .dirty_routes()
            .into_iter()
            .filter(|(model, route)| {
                stored.get(model).and_then(PersistedRoute::to_byte) != route.to_byte()
            })
            .collect();
        self.apply_chat_routes(tool, fresh).await;
    }

    /// Persist the plain-chat turn's learned route, if new/changed.
    async fn persist_chat_route(&mut self) {
        if let Some(route) = chat_route_to_persist(&self.key, &self.raw_model, &self.format) {
            self.apply_chat_routes("code", vec![route]).await;
        }
    }

    /// Write learned routes to the key (disk + in-memory key); no-op when empty.
    async fn apply_chat_routes(&mut self, tool: &str, routes: Vec<(String, PersistedRoute)>) {
        if routes.is_empty() {
            return;
        }
        if self
            .session_store
            .merge_routes(&self.key.id, tool, &routes)
            .await
            .is_ok()
        {
            let entry = self
                .key
                .protocol_routes
                .entry(tool.to_string())
                .or_default();
            for (model, route) in routes {
                entry.insert(model, route);
            }
        }
    }

    async fn finish_successful_response(&mut self, turn: ChatTurnResult) -> Result<()> {
        self.persist_chat_route().await;

        // History already holds everything sent to the model this turn (the user
        // input plus any assistant/tool segments flushed while streaming); capture
        // it for the usage estimate before appending the final reply.
        let prompt_text: String = self.history.iter().map(|m| m.content.as_str()).collect();

        // Streaming paths accumulate the reply in `pending_response`; a
        // non-streaming HTTP reply arrives in `turn.content`. When a turn ends on
        // a tool step (cursor agents), the prose was already flushed as earlier
        // entries and `turn.content` holds the full accumulation — re-pushing it
        // would duplicate the transcript, so emit nothing.
        let ended_on_tool = matches!(
            self.history.last().map(|m| m.role.as_str()),
            Some("tool_call" | "tool_result")
        );
        let content = if !self.pending_response.is_empty() {
            self.pending_response.clone()
        } else if ended_on_tool {
            String::new()
        } else {
            turn.content.clone()
        };
        self.pending_submit = None;
        self.pending_response.clear();
        // Carry the turn's reasoning onto the committed message (with its thinking
        // duration for the folded summary) so it stays in the transcript instead of
        // vanishing when the live `pending_reasoning` clears.
        self.commit_assistant_segment(content);

        let usage = turn.usage_or_estimate(&prompt_text);
        // Cache for subsequent heartbeat saves, which run without a turn.
        if let Some(ref model) = turn.model {
            self.billed_model = Some(model.clone());
        }
        let stats_model = self.billed_model.as_deref().unwrap_or(&self.raw_model);
        self.session_store
            .record_tokens(
                &self.key.id,
                Some("code"),
                Some(stats_model),
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cache_read_input_tokens,
                usage.cache_creation_input_tokens,
            )
            .await?;
        // Fold the same split into the session's running total so the chat index
        // entry feeds `aivo stats --since` (the non-agent / cursor path).
        let split = crate::services::session_store::SessionTokens {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cache_read_tokens: usage.cache_read_input_tokens,
            cache_write_tokens: usage.cache_creation_input_tokens,
        };
        if let Some(cost) = self.session_pricing().and_then(|p| p.cost_usd(&split)) {
            self.session_cost_usd += cost;
        }
        self.session_tokens = self.session_tokens.merge(split);
        self.context_tokens = if turn.usage.is_some() {
            usage.total_tokens()
        } else {
            self.estimated_context_used().await
        };
        // cursor ACP returns no usage → the figure is a transcript estimate.
        self.context_is_estimate = turn.usage.is_none();
        self.last_usage = turn.usage;
        self.live_usage = None;

        // The turn's reply for the log: the most recent assistant entry (the final
        // text, or the last flushed segment when the turn ended on a tool step).
        let assistant_content = self
            .history
            .iter()
            .rev()
            .find(|message| message.role == "assistant")
            .map(|message| message.content.clone())
            .unwrap_or_default();
        let user_message = self
            .history
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .cloned();
        if let Some(user_message) = user_message {
            let _ = log_chat_turn(
                &self.session_store,
                &self.key,
                &self.raw_model,
                Some(self.persist_cwd()),
                Some(&self.session_id),
                &user_message,
                &assistant_content,
                None,
                &usage,
            )
            .await;
        }

        self.persist_history().await?;
        self.notice = None;
        Ok(())
    }

    fn finish_failed_response(&mut self, err: String) {
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        restore_cancelled_submission(
            &mut self.history,
            &mut self.draft,
            &mut self.draft_attachments,
            &mut self.pending_submit,
        );
        self.notice = Some((ERROR, reframe_image_input_error(err, &self.model)));
    }

    async fn apply_loaded_models(
        &mut self,
        result: std::result::Result<Vec<ModelChoice>, String>,
    ) -> Result<()> {
        match result {
            Ok(models) => {
                if let Some(index) = self.populate_model_picker(models) {
                    self.activate_picker_selection(index).await?;
                }
            }
            Err(err) => {
                self.overlay = Overlay::None;
                self.notice = Some((ERROR, err));
            }
        }
        Ok(())
    }

    fn populate_model_picker(&mut self, models: Vec<ModelChoice>) -> Option<usize> {
        let Overlay::Picker(picker) = &mut self.overlay else {
            return None;
        };
        if !matches!(picker.kind, PickerKind::Model { .. }) {
            return None;
        }

        picker.items = models
            .into_iter()
            .map(|m| PickerEntry {
                search_text: m.id.clone(),
                label: m.label,
                value: PickerValue::Model(m.id),
            })
            .collect();
        picker.loading = false;
        picker.selected = 0;
        picker.exact_match_index()
    }

    async fn apply_resume_load_result(
        &mut self,
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    ) -> Result<()> {
        let Some(loading) = &self.loading_resume else {
            return Ok(());
        };
        if loading.request_id != request_id {
            return Ok(());
        }

        self.resume_task = None;
        match result {
            Ok(session) => {
                self.apply_loaded_session(session).await?;
                self.loading_resume = None;
                self.resume_restore_state = None;
                self.notice = None;
            }
            Err(err) => {
                self.loading_resume = None;
                if let Some(state) = self.resume_restore_state.take() {
                    self.restore_resume_state(state);
                }
                self.notice = Some((ERROR, err));
            }
        }

        Ok(())
    }

    pub(super) async fn flush_for_exit(&mut self) {
        // Drain any runtime events that landed between the last poll and the
        // exit keypress (e.g. a Finished that completed while the user was
        // pressing Ctrl-C) so a just-finished turn is captured in history.
        let _ = self.handle_runtime_events().await;

        // Reveal anything still buffered so the full received reply (not just
        // the typed-out prefix) is what we salvage below.
        self.drain_incoming_buffer();

        // If the response was still streaming at exit, salvage the partial
        // assistant text the same way an explicit interrupt does — otherwise
        // the user's prompt and any visible reply would be lost.
        if self.sending && !self.pending_response.is_empty() {
            let partial = std::mem::take(&mut self.pending_response);
            self.pending_reasoning.clear();
            self.history.push(ChatMessage {
                model: self.turn_model.clone(),
                role: "assistant".to_string(),
                content: partial,
                reasoning_content: None,
                attachments: vec![],
            });
        }

        // Persist whatever history we have so /resume can find this session
        // even when the user exits before a successful Finished event.
        if !self.history.is_empty() {
            let _ = self.persist_history().await;
        }

        // Searchable session topic for `memory_search` — user text only
        // (shell commands can embed secrets).
        if !self.real_cwd.is_empty()
            && let Some(topic) = self.history.iter().find(|m| m.role == "user")
        {
            let date = chrono::Local::now().format("%Y-%m-%d").to_string();
            crate::agent::memory::record_session_summary(
                std::path::Path::new(&self.real_cwd),
                &topic.content,
                &date,
            );
        }
    }

    /// Debounced `/resume` preview loader: after the selection rests on an
    /// un-cached (or stale) session, spawn a background decrypt; `true` = repaint.
    pub(super) fn tick_session_preview(&mut self) -> bool {
        // Split session picker only (last-frame rect; the debounce absorbs the lag).
        let wanted = match (&self.overlay, self.overlay_detail_area) {
            (Overlay::Picker(picker), Some(_))
                if matches!(picker.kind, PickerKind::Session) && !picker.loading =>
            {
                picker
                    .filtered_items()
                    .get(picker.selected)
                    .and_then(|(_, item)| match &item.value {
                        PickerValue::Session(preview) => {
                            Some((preview.session_id.clone(), preview.updated_at.clone()))
                        }
                        _ => None,
                    })
            }
            _ => None,
        };
        let Some((session_id, updated_at)) = wanted else {
            self.session_preview_pending = None;
            return false;
        };
        // Valid cache entry (same index updated_at) → nothing to load.
        if self
            .session_preview_cache
            .get(&session_id)
            .is_some_and(|entry| entry.updated_at == updated_at)
        {
            self.session_preview_pending = None;
            return false;
        }
        let due = match &self.session_preview_pending {
            Some((sid, due)) if *sid == session_id => Instant::now() >= *due,
            _ => {
                // (Re)arm the debounce whenever the selection lands on a new row.
                self.session_preview_pending =
                    Some((session_id, Instant::now() + PREVIEW_DEBOUNCE));
                return false;
            }
        };
        if !due
            || self
                .session_preview_task
                .as_ref()
                .is_some_and(|(sid, task)| *sid == session_id && !task.is_finished())
        {
            return false;
        }
        self.session_preview_pending = None;

        let session_store = self.session_store.clone();
        let tx = self.tx.clone();
        let sid = session_id.clone();
        let task = tokio::spawn(async move {
            let entry = match load_session_preview(&session_store, &sid, PREVIEW_MAX_MESSAGES).await
            {
                Ok((messages, truncated)) => PreviewEntry {
                    updated_at,
                    messages,
                    truncated,
                    error: None,
                },
                Err(err) => PreviewEntry {
                    updated_at,
                    messages: Vec::new(),
                    truncated: false,
                    error: Some(err),
                },
            };
            let _ = tx.send(RuntimeEvent::SessionPreviewLoaded {
                session_id: sid,
                entry,
            });
        });
        self.session_preview_task = Some((session_id, task));
        true
    }

    pub(super) async fn run(&mut self) -> Result<()> {
        let mut terminal = setup_terminal(chat_mouse_enabled())?;
        // Open the cursor session now (no-op for other keys) so its connect
        // overlaps the user typing their first message.
        self.prewarm_cursor_session();
        // Repaint only on change; an idle chat draws nothing.
        let mut needs_redraw = true;
        let run_result = loop {
            match self.handle_runtime_events().await {
                Ok(true) => needs_redraw = true,
                Ok(false) => {}
                Err(err) => break Err(err),
            }

            // Deferred `--share` start, once the session has settled.
            if self.maybe_start_live_share().await {
                needs_redraw = true;
            }

            // Keep the selection growing while a drag rests on the top/bottom edge.
            if self.tick_drag_autoscroll() {
                needs_redraw = true;
            }

            // Reveal buffered stream text a slice at a time (typewriter), then
            // run any finish that was waiting for the buffer to drain.
            if self.tick_typewriter() {
                needs_redraw = true;
            }
            // `break Err` (not `?`) so the post-loop teardown (kill_all, restore_terminal) still runs.
            match self.run_deferred_finish_if_ready().await {
                Ok(true) => needs_redraw = true,
                Ok(false) => {}
                Err(err) => break Err(err),
            }
            // Watchdog: recover a turn left stuck "sending" by a task that died silently.
            match self.recover_dead_response_task().await {
                Ok(true) => needs_redraw = true,
                Ok(false) => {}
                Err(err) => break Err(err),
            }

            self.tick_decision_wait();
            self.tick_status_throttle();

            // Refresh the cached job count (this reaps) at most every 250ms — a badge
            // needs no tick-rate waitpid sweeps, and the input-repaint tick runs at ~1ms.
            if self.last_jobs_poll.elapsed() >= Duration::from_millis(250) {
                self.last_jobs_poll = std::time::Instant::now();
                let jobs_running = self.jobs.running_count();
                if jobs_running != self.jobs_running {
                    self.jobs_running = jobs_running;
                    needs_redraw = true;
                }
            }

            // Rotate the welcome tip on its interval (cheap at the idle cadence).
            if self.tick_welcome_tip() {
                needs_redraw = true;
            }

            // Debounced `/resume` preview load once the selection rests on a row.
            if self.tick_session_preview() {
                needs_redraw = true;
            }

            // Animations repaint without input.
            if self.is_animating() {
                needs_redraw = true;
            }

            if needs_redraw {
                if let Err(err) = terminal.draw(|frame| self.render(frame)) {
                    break Err(err.into());
                }
                needs_redraw = false;
            }

            // Drain every buffered input event in one pass before the next
            // repaint. Processing one event per tick caps consumption at the
            // idle cadence (~40/s), far below the rate a fast drag emits, so the
            // selection would otherwise trail the cursor by a growing backlog.
            // The draw above already reset `needs_redraw`, so it is true here iff
            // this pass handled input — the cue to repaint promptly below.
            match self.drain_input(&mut needs_redraw).await {
                Ok(true) => break Ok(()),
                Ok(false) => {}
                Err(err) => break Err(err),
            }

            if std::mem::take(&mut self.pending_external_edit) {
                self.edit_draft_in_external_editor(&mut terminal).await;
                needs_redraw = true;
            }

            // Spinner advances only while animating.
            if self.is_animating() {
                self.frame_tick = self.frame_tick.wrapping_add(1);
            }

            // Non-blocking nap, never a blocking poll — that would freeze the
            // streaming task on the current-thread runtime. When this pass
            // handled input, nap only briefly so the scroll/keystroke repaints
            // near-instantly and in fine increments instead of trailing the idle
            // cadence; the short sleep still yields so streaming keeps flowing.
            let nap = if needs_redraw {
                INPUT_REPAINT_INTERVAL
            } else if self.is_animating() {
                ANIMATING_FRAME_INTERVAL
            } else {
                IDLE_POLL_INTERVAL
            };
            tokio::time::sleep(nap).await;
        };

        self.flush_for_exit().await;

        // Abort in-flight tasks and await them so their futures are actually
        // dropped (closing any open HTTP connections) before we return. On the
        // current-thread runtime, `abort()` alone only schedules cancellation;
        // without the explicit `await` the task stays alive until the runtime
        // itself shuts down at process exit.
        let response_task = self.response_task.take();
        let resume_task = self.resume_task.take();
        let preview_task = self.session_preview_task.take();
        let local_command = self.local_command.take();
        self.loading_resume = None;
        self.resume_restore_state = None;
        self.session_preview_pending = None;
        if let Some(task) = response_task {
            task.abort();
            let _ = task.await;
        }
        if let Some(task) = resume_task {
            task.abort();
            let _ = task.await;
        }
        if let Some((_, task)) = preview_task {
            task.abort();
            let _ = task.await;
        }
        if let Some(mut run) = local_command {
            let _ = run.killer.kill();
            run.task.abort();
            let _ = run.task.await;
        }
        // Background jobs must not outlive the TUI.
        let _ = self.jobs.kill_all().await;
        restore_terminal(terminal)?;
        run_result
    }

    /// Consumes all input events currently buffered, handling each in arrival
    /// order, and returns `Ok(true)` when one of them asks the app to exit.
    /// Sets `needs_redraw` if anything was handled. Bounded by
    /// [`MAX_INPUT_EVENTS_PER_TICK`] so a flood can't starve the repaint —
    /// the remainder is picked up on the next loop pass.
    async fn drain_input(&mut self, needs_redraw: &mut bool) -> Result<bool> {
        let mut drained = 0usize;
        // Reassemble mouse reports that crossterm split at the ESC byte. A fast
        // SGR mouse report (`\x1b[<b;x;y` then `M`/`m`) whose leading ESC lands in
        // a separate read surfaces as a bare `Esc` key (which would spuriously
        // close an overlay) followed by its tail `[<…M` as literal `Char`s typed
        // into the composer. We withhold a bare Esc for one event to see whether
        // that tail follows in the same burst; if so the Esc is dropped, the tail
        // swallowed, and the scroll the user meant is re-synthesized.
        let mut esc = EscReassembly::Idle;
        while event::poll(Duration::from_millis(0))? {
            let event = event::read()?;
            *needs_redraw = true;
            drained += 1;

            let event = match self.step_esc_reassembly(&mut esc, event).await? {
                EscStep::Consumed => {
                    if drained >= MAX_INPUT_EVENTS_PER_TICK {
                        break;
                    }
                    continue;
                }
                EscStep::Exit => return Ok(true),
                EscStep::Passthrough(event) => event,
            };

            if let Some(true) = self.handle_terminal_event(event).await? {
                return Ok(true);
            }
            if drained >= MAX_INPUT_EVENTS_PER_TICK {
                break;
            }
        }
        // Burst ended: flush whatever we were still holding — a real lone Esc, or
        // an incomplete fragment that never resolved into a mouse report.
        if self.flush_esc_reassembly(esc).await? {
            return Ok(true);
        }
        Ok(false)
    }

    /// Ctrl+X Ctrl+E: suspend the TUI, edit the draft in `$VISUAL`/`$EDITOR`, load it back.
    /// The terminal is restored even on failure; errors become a notice, the draft is kept.
    async fn edit_draft_in_external_editor(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) {
        let editor = external_editor_command();
        let _ = disable_raw_mode();
        let _ = execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableBracketedPaste,
            DisableMouseCapture
        );
        let _ = terminal.show_cursor();

        let result = self.run_editor_on_draft(&editor).await;

        // Mirror `setup_terminal` on the live handle.
        let _ = enable_raw_mode();
        let _ = execute!(
            terminal.backend_mut(),
            EnterAlternateScreen,
            EnableBracketedPaste
        );
        if chat_mouse_enabled() {
            let _ = execute!(terminal.backend_mut(), EnableMouseCapture);
        }
        let _ = terminal.clear();

        if let Err(err) = result {
            self.notice = Some((ERROR, format!("External edit failed: {err:#}")));
        }
    }

    async fn run_editor_on_draft(&mut self, editor: &[String]) -> Result<()> {
        use anyhow::Context as _;
        let file = tempfile::Builder::new()
            .prefix("aivo-draft-")
            .suffix(".md")
            .tempfile()
            .context("create temp file")?;
        std::fs::write(file.path(), &self.draft).context("write draft")?;
        let status = tokio::process::Command::new(&editor[0])
            .args(&editor[1..])
            .arg(file.path())
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
            .await
            .with_context(|| format!("launch `{}` (set $VISUAL or $EDITOR)", editor[0]))?;
        if !status.success() {
            anyhow::bail!("`{}` exited with {status}; draft kept", editor[0]);
        }
        let text = std::fs::read_to_string(file.path()).context("read edited draft")?;
        // Editors append a final newline and may save CRLF; neither belongs in the composer.
        let text = text.replace("\r\n", "\n");
        let text = text.trim_end_matches('\n');
        self.leave_history_navigation();
        self.draft = text.to_string();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        Ok(())
    }

    /// Feeds one freshly-read event through the [`EscReassembly`] state machine.
    /// Returns whether it was [`EscStep::Consumed`] (folded into a held Esc or a
    /// mouse fragment), should [`EscStep::Exit`], or is a plain
    /// [`EscStep::Passthrough`] event the caller should handle normally. Any
    /// held input that turns out to be real text is replayed inside here so it is
    /// never lost.
    pub(super) async fn step_esc_reassembly(
        &mut self,
        esc: &mut EscReassembly,
        event: Event,
    ) -> Result<EscStep> {
        match esc {
            EscReassembly::Idle => {
                if is_bare_esc(&event) {
                    *esc = EscReassembly::PendingEsc;
                    return Ok(EscStep::Consumed);
                }
                Ok(EscStep::Passthrough(event))
            }
            EscReassembly::PendingEsc => {
                // `\x1b[` is the CSI lead of a split mouse report — start swallowing.
                if char_press(&event) == Some('[') {
                    *esc = EscReassembly::Sgr("[".to_string());
                    return Ok(EscStep::Consumed);
                }
                // A genuine lone Esc: deliver it, then let this event through.
                *esc = EscReassembly::Idle;
                if self.deliver_esc().await? {
                    return Ok(EscStep::Exit);
                }
                Ok(EscStep::Passthrough(event))
            }
            EscReassembly::Sgr(buf) => {
                let Some(c) = char_press(&event) else {
                    // A non-character event interrupted the fragment: replay what
                    // we held as text, then handle this event normally.
                    let buffered = std::mem::take(buf);
                    *esc = EscReassembly::Idle;
                    if self.replay_held(&buffered).await? {
                        return Ok(EscStep::Exit);
                    }
                    return Ok(EscStep::Passthrough(event));
                };
                buf.push(c);
                match sgr_mouse_frag_step(buf) {
                    FragStep::Continue => Ok(EscStep::Consumed),
                    FragStep::Final => {
                        let frag = std::mem::take(buf);
                        *esc = EscReassembly::Idle;
                        if self.dispatch_leaked_scroll(&frag).await? {
                            return Ok(EscStep::Exit);
                        }
                        Ok(EscStep::Consumed)
                    }
                    FragStep::Invalid => {
                        // Not a mouse report after all; the just-pushed char is
                        // part of a plainly textual run, so replay the whole held
                        // sequence (the suppressed Esc plus the buffer) as text.
                        let buffered = std::mem::take(buf);
                        *esc = EscReassembly::Idle;
                        if self.replay_held(&buffered).await? {
                            return Ok(EscStep::Exit);
                        }
                        Ok(EscStep::Consumed)
                    }
                }
            }
        }
    }

    /// Flushes input held by the reassembler when the input burst ends.
    pub(super) async fn flush_esc_reassembly(&mut self, esc: EscReassembly) -> Result<bool> {
        match esc {
            EscReassembly::Idle => Ok(false),
            EscReassembly::PendingEsc => self.deliver_esc().await,
            EscReassembly::Sgr(buf) => self.replay_held(&buf).await,
        }
    }

    /// Delivers a synthesized `Esc` keypress (the held bare Esc was real).
    async fn deliver_esc(&mut self) -> Result<bool> {
        self.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .await
    }

    /// Replays a withheld Esc followed by `buffered` as ordinary keystrokes — the
    /// path taken when a suspected mouse fragment turns out to be real text.
    async fn replay_held(&mut self, buffered: &str) -> Result<bool> {
        if self.deliver_esc().await? {
            return Ok(true);
        }
        for c in buffered.chars() {
            if self
                .handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
                .await?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Re-synthesizes the wheel scroll from a reassembled SGR mouse fragment so a
    /// split report still scrolls instead of being silently dropped. Non-scroll
    /// reports (clicks, drags) are discarded — rebuilding those is not worth it.
    async fn dispatch_leaked_scroll(&mut self, frag: &str) -> Result<bool> {
        match parse_sgr_scroll(frag) {
            Some(mouse) => self.handle_mouse(mouse).await,
            None => Ok(false),
        }
    }

    async fn handle_terminal_event(&mut self, event: Event) -> Result<Option<bool>> {
        match event {
            // On Windows, crossterm emits both Press and Release events for
            // every keystroke; Unix only emits the press equivalent. Process
            // Press only so typed characters aren't doubled on Windows.
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                Ok(Some(self.handle_key(key).await?))
            }
            Event::Key(_) => Ok(None),
            Event::Mouse(mouse) => Ok(Some(self.handle_mouse(mouse).await?)),
            Event::Resize(_, _) => Ok(None),
            Event::Paste(text) => {
                if !self.overlay_paste(&text) && !self.overlay.blocks_input() && !self.is_busy() {
                    self.insert_pasted_text(&text);
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    pub(super) async fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<bool> {
        if let Some(should_exit) = self.handle_overlay_mouse(mouse).await? {
            return Ok(should_exit);
        }

        match mouse.kind {
            MouseEventKind::ScrollUp if self.mouse_over_transcript(mouse) => {
                self.scroll_up_lines(self.scroll_speed)
            }
            MouseEventKind::ScrollDown if self.mouse_over_transcript(mouse) => {
                self.scroll_down_lines(self.scroll_speed)
            }
            MouseEventKind::Down(MouseButton::Left) => {
                // The jump-to-bottom pill claims the click before composer/selection.
                if !self.overlay.blocks_input()
                    && let Some(hit) = self.jump_to_bottom_hit
                    && rect_contains(hit, (mouse.column, mouse.row))
                {
                    self.scroll_to_bottom();
                    return Ok(false);
                }
                // A press in the composer also drops the caret there; a drag still selects.
                if self.should_show_input_cursor()
                    && !self.overlay.blocks_input()
                    && let Some(offset) = self.composer_offset_for_mouse(mouse)
                {
                    self.cursor = offset;
                }
                let Some((surface, point)) = self.selection_target(mouse, false) else {
                    return Ok(false);
                };
                let clicks = self.register_click(point);
                self.selection_flash_until = None;
                match clicks {
                    // Double-click selects the word under the cursor; triple-click
                    // selects the whole visual row. Both copy + flash, matching the
                    // drag-to-copy model. They fall through to a caret drag when the
                    // click lands on blank space (no word/row to grab).
                    2 if self.select_word_on(surface, point) => {
                        self.end_drag();
                        self.copy_selection_to_clipboard();
                    }
                    3 if self.select_line_on(surface, point) => {
                        self.end_drag();
                        self.copy_selection_to_clipboard();
                    }
                    // A single click on a `▸`/`▾` fold marker (thinking header or
                    // `!cmd` output expander) toggles that block's inline expansion.
                    // Anything else starts a drag-select.
                    1 if matches!(surface, SelectionSurface::Transcript)
                        && (self.toggle_thinking_at_row(point.row)
                            || self.toggle_output_at_row(point.row)) => {}
                    _ => self.begin_drag(surface, point),
                }
            }
            MouseEventKind::Drag(MouseButton::Left) if self.transcript_drag_active => {
                self.set_drag_focus(mouse);
                self.update_drag_autoscroll(mouse);
            }
            MouseEventKind::Drag(MouseButton::Left) if self.screen_drag_active => {
                self.set_screen_drag_focus(mouse);
            }
            MouseEventKind::Up(MouseButton::Left) if self.transcript_drag_active => {
                self.transcript_drag_active = false;
                self.drag_autoscroll = None;
                self.set_drag_focus(mouse);
                self.copy_selection_to_clipboard();
            }
            MouseEventKind::Up(MouseButton::Left) if self.screen_drag_active => {
                self.screen_drag_active = false;
                self.set_screen_drag_focus(mouse);
                self.copy_selection_to_clipboard();
            }
            _ => {}
        }

        Ok(false)
    }

    /// Records a left-click and returns how many consecutive clicks it forms
    /// (1, 2, or 3) — the basis for word/line selection.
    pub(super) fn register_click(&mut self, point: TranscriptPoint) -> u8 {
        let now = Instant::now();
        let count = match self.last_click {
            Some(prev)
                if now.duration_since(prev.at) <= MULTI_CLICK_INTERVAL
                    && prev.point.row == point.row
                    && prev.point.column.abs_diff(point.column) <= 1 =>
            {
                (prev.count + 1).min(3)
            }
            _ => 1,
        };
        self.last_click = Some(ClickTracker {
            at: now,
            point,
            count,
        });
        count
    }

    /// The selectable rows backing `surface`.
    fn surface_rows(&self, surface: SelectionSurface) -> Option<&[String]> {
        match surface {
            SelectionSurface::Transcript => {
                self.transcript_hitbox.as_ref().map(|h| h.rows.as_slice())
            }
            SelectionSurface::Screen => self.screen_surface.as_ref().map(|s| s.rows.as_slice()),
        }
    }

    /// Writes `selection` to `surface`, clearing the other so only one is live.
    fn set_selection(&mut self, surface: SelectionSurface, selection: TranscriptSelection) {
        match surface {
            SelectionSurface::Transcript => {
                self.screen_selection = None;
                self.transcript_selection = Some(selection);
            }
            SelectionSurface::Screen => {
                self.transcript_selection = None;
                self.screen_selection = Some(selection);
            }
        }
    }

    /// Selects the word at `point` on `surface`; false on whitespace/past the row.
    pub(super) fn select_word_on(
        &mut self,
        surface: SelectionSurface,
        point: TranscriptPoint,
    ) -> bool {
        let Some((start, end)) = self
            .surface_rows(surface)
            .and_then(|rows| rows.get(point.row))
            .and_then(|row| word_bounds_at(row, point.column))
        else {
            return false;
        };
        self.set_selection(
            surface,
            TranscriptSelection {
                anchor: TranscriptPoint {
                    row: point.row,
                    column: start,
                },
                focus: TranscriptPoint {
                    row: point.row,
                    column: end,
                },
            },
        );
        true
    }

    /// Selects the visual row at `point` on `surface` (trailing blanks excluded);
    /// false for empty rows.
    pub(super) fn select_line_on(
        &mut self,
        surface: SelectionSurface,
        point: TranscriptPoint,
    ) -> bool {
        let width = self
            .surface_rows(surface)
            .and_then(|rows| rows.get(point.row))
            .map(|row| row_display_width(row.trim_end()))
            .unwrap_or(0);
        if width == 0 {
            return false;
        }
        self.set_selection(
            surface,
            TranscriptSelection {
                anchor: TranscriptPoint {
                    row: point.row,
                    column: 0,
                },
                focus: TranscriptPoint {
                    row: point.row,
                    column: width,
                },
            },
        );
        true
    }

    /// Picks the surface + mapped point for a press/drag: the transcript when the
    /// pointer is over it with no overlay, else the flat screen. `clamp` pins an
    /// off-surface drag to the edge instead of returning `None`.
    pub(super) fn selection_target(
        &self,
        mouse: MouseEvent,
        clamp: bool,
    ) -> Option<(SelectionSurface, TranscriptPoint)> {
        // The empty state is drawn by `render_empty_state`, not the transcript row
        // model, so its hitbox rows don't match the screen — select from the flat
        // screen surface instead (which snapshots the rendered cells).
        if !self.overlay.blocks_input()
            && self.mouse_over_transcript(mouse)
            && !self.is_transcript_empty()
        {
            return self
                .transcript_point_for_mouse(mouse, clamp)
                .map(|point| (SelectionSurface::Transcript, point));
        }
        self.screen_point_for_mouse(mouse, clamp)
            .map(|point| (SelectionSurface::Screen, point))
    }

    /// Anchors a fresh drag-selection at `point` on `surface`.
    fn begin_drag(&mut self, surface: SelectionSurface, point: TranscriptPoint) {
        self.set_selection(
            surface,
            TranscriptSelection {
                anchor: point,
                focus: point,
            },
        );
        match surface {
            SelectionSurface::Transcript => {
                self.screen_drag_active = false;
                self.transcript_drag_active = true;
            }
            SelectionSurface::Screen => {
                self.transcript_drag_active = false;
                self.drag_autoscroll = None;
                self.screen_drag_active = true;
            }
        }
    }

    /// Ends any in-progress drag (after a word/line click finalizes).
    fn end_drag(&mut self) {
        self.transcript_drag_active = false;
        self.screen_drag_active = false;
        self.drag_autoscroll = None;
    }

    /// Extends the live transcript selection to the dragged position.
    fn set_drag_focus(&mut self, mouse: MouseEvent) {
        if let Some(point) = self.transcript_point_for_mouse(mouse, true)
            && let Some(selection) = &mut self.transcript_selection
        {
            selection.focus = point;
        }
    }

    /// Extends the live screen selection to the dragged position.
    fn set_screen_drag_focus(&mut self, mouse: MouseEvent) {
        if let Some(point) = self.screen_point_for_mouse(mouse, true)
            && let Some(selection) = &mut self.screen_selection
        {
            selection.focus = point;
        }
    }

    /// Arms or disarms edge auto-scroll based on the drag position. The
    /// transcript sits flush with the top of the screen, so there is no room
    /// *above* it for the pointer — scroll-up therefore arms on the top edge
    /// *row* of the viewport. Scroll-down keeps requiring the pointer to cross
    /// *below* the transcript (into the composer), so resting on the last
    /// visible line never steals it from a normal selection. A step only fires
    /// when there is hidden content that way, so arming with nothing left to
    /// reveal is a no-op.
    pub(super) fn update_drag_autoscroll(&mut self, mouse: MouseEvent) {
        self.drag_autoscroll = self.transcript_hitbox.as_ref().and_then(|hitbox| {
            let area = hitbox.area;
            if area.height == 0 {
                return None;
            }
            let max_x = area.x.saturating_add(area.width.saturating_sub(1));
            let column = mouse.column.clamp(area.x, max_x).saturating_sub(area.x);
            if mouse.row <= area.y {
                Some(DragAutoscroll { dir: -1, column })
            } else if mouse.row >= area.y.saturating_add(area.height) {
                Some(DragAutoscroll { dir: 1, column })
            } else {
                None
            }
        });
    }

    /// Drives one throttled auto-scroll step while a drag sits at an edge, then
    /// re-anchors the selection focus to the newly exposed row. Returns true if
    /// the view moved (caller repaints).
    pub(super) fn tick_drag_autoscroll(&mut self) -> bool {
        let Some(auto) = self.drag_autoscroll else {
            return false;
        };
        if !self.transcript_drag_active {
            self.drag_autoscroll = None;
            return false;
        }
        let now = Instant::now();
        if let Some(last) = self.last_autoscroll
            && now.duration_since(last) < DRAG_AUTOSCROLL_INTERVAL
        {
            return false;
        }
        self.last_autoscroll = Some(now);

        let before = self.transcript_scroll;
        if auto.dir < 0 {
            self.scroll_up_lines(1);
        } else {
            self.scroll_down_lines(1);
        }
        if self.transcript_scroll == before {
            return false; // already at the top/bottom — nothing exposed
        }

        let row_count = self
            .transcript_hitbox
            .as_ref()
            .map(|hitbox| hitbox.rows.len())
            .unwrap_or(0);
        let view_height = usize::from(self.transcript_view_height);
        let focus_row = if auto.dir < 0 {
            self.transcript_scroll
        } else {
            self.transcript_scroll + view_height.saturating_sub(1)
        }
        .min(row_count.saturating_sub(1));
        if let Some(selection) = &mut self.transcript_selection {
            selection.focus = TranscriptPoint {
                row: focus_row,
                column: auto.column,
            };
        }
        true
    }

    /// Copies the current selection to the clipboard, toasts, and lights the
    /// brief flash. An empty selection is cleared instead.
    fn copy_selection_to_clipboard(&mut self) {
        match self.selected_any_text().filter(|text| !text.is_empty()) {
            Some(selected) => match write_system_clipboard(&selected) {
                Ok(()) => {
                    let chars = selected.chars().count();
                    let lines = selected.lines().count().max(1);
                    let char_label = if chars == 1 { "char" } else { "chars" };
                    let mut detail = format!("Copied {chars} {char_label}");
                    if lines > 1 {
                        detail.push_str(&format!(" · {lines} lines"));
                    }
                    self.show_toast(detail);
                    self.selection_flash_until = Some(Instant::now() + SELECTION_FLASH_DURATION);
                }
                Err(err) => {
                    self.notice = Some((ERROR, format!("Copy failed: {err}")));
                }
            },
            None => {
                self.transcript_selection = None;
                self.screen_selection = None;
            }
        }
    }

    /// Flash a brief, self-expiring toast bottom-right (copy confirmations, mode
    /// toggles). Unlike `notice`, it fades on its own instead of lingering until
    /// the next turn.
    pub(super) fn show_toast(&mut self, text: impl Into<String>) {
        let created_at = Instant::now();
        self.toast = Some(Toast {
            text: text.into(),
            created_at,
            expires_at: created_at + TOAST_DURATION,
        });
    }

    fn mouse_over_transcript(&self, mouse: MouseEvent) -> bool {
        self.transcript_hitbox
            .as_ref()
            .is_some_and(|hitbox| rect_contains(hitbox.area, (mouse.column, mouse.row)))
    }

    /// Byte offset in the draft for a click inside the composer, or `None` when
    /// the click misses it. Maps the click row (minus the attachment rows above
    /// the draft, plus the draft scroll) and column to the draft via the shared
    /// wrap model, so a click lands exactly where the caret renders.
    pub(super) fn composer_offset_for_mouse(&self, mouse: MouseEvent) -> Option<usize> {
        let area = self.composer_text_area?;
        if !rect_contains(area, (mouse.column, mouse.row)) {
            return None;
        }
        if self.draft.is_empty() {
            return Some(0);
        }
        let attach = self.draft_attachments.len() as u16;
        let rel_y = mouse.row.saturating_sub(area.y);
        if rel_y < attach {
            // Clicked an attachment row above the draft → caret to the start.
            return Some(0);
        }
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let row = (usize::from(rel_y - attach) + self.composer_scroll).min(rows.len() - 1);
        let target_col = usize::from(mouse.column.saturating_sub(area.x));
        Some(composer_offset_for_col(&self.draft, &rows, row, target_col))
    }

    fn transcript_point_for_mouse(
        &self,
        mouse: MouseEvent,
        clamp_to_hitbox: bool,
    ) -> Option<TranscriptPoint> {
        let hitbox = self.transcript_hitbox.as_ref()?;
        let point = (mouse.column, mouse.row);
        if !clamp_to_hitbox && !rect_contains(hitbox.area, point) {
            return None;
        }

        let max_x = hitbox
            .area
            .x
            .saturating_add(hitbox.area.width.saturating_sub(1));
        let max_y = hitbox
            .area
            .y
            .saturating_add(hitbox.area.height.saturating_sub(1));
        let column = mouse
            .column
            .clamp(hitbox.area.x, max_x)
            .saturating_sub(hitbox.area.x);
        let visible_row = mouse
            .row
            .clamp(hitbox.area.y, max_y)
            .saturating_sub(hitbox.area.y);
        Some(TranscriptPoint {
            row: hitbox.first_row + usize::from(visible_row),
            column,
        })
    }

    /// History indices of assistant turns that render a `▸ thought` header, in
    /// display order — the Nth header row maps to the Nth entry (each such turn
    /// renders exactly one header row). The live streaming summary is absent: it has
    /// no history index and isn't expandable inline. Empty when thinking is off.
    fn reasoning_message_indices(&self) -> Vec<usize> {
        if !self.thinking_enabled {
            return Vec::new();
        }
        self.history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "assistant")
            .filter(|(_, m)| {
                m.reasoning_content
                    .as_deref()
                    .is_some_and(|r| !r.trim().is_empty())
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// History index behind each expander marker row, in display order. Must
    /// repeat an index exactly as many times as the render emits markers for
    /// it — one per block, two for a long expanded tool result.
    fn expandable_output_indices(&self) -> Vec<usize> {
        self.history
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                let markers = match m.role.as_str() {
                    "local_command" => {
                        usize::from(local_command_total_lines(&m.content) > MAX_OUTPUT_LINES)
                    }
                    "tool_result" => {
                        let count = m.content.lines().count();
                        if count <= 1 {
                            0
                        } else if self.expanded_output.contains(&i) && count > MAX_OUTPUT_LINES {
                            2
                        } else {
                            1
                        }
                    }
                    _ => 0,
                };
                std::iter::repeat_n(i, markers)
            })
            .collect()
    }

    /// While a decision card is up, push the step + turn clocks forward by each
    /// waiting interval — decision time must not read as tool runtime or
    /// inflate `✶ Done in …`. Called per frame beside `tick_status_throttle`.
    pub(super) fn tick_decision_wait(&mut self) {
        let waiting = self.sending
            && (self.agent_permission.is_some()
                || self.agent_ask.is_some()
                || self.agent_review.is_some()
                || self.agent_plan_approval.is_some());
        if !waiting {
            self.wait_tick = None;
            return;
        }
        let now = Instant::now();
        let Some(dt) = self.wait_tick.replace(now).map(|prev| now - prev) else {
            return;
        };
        if let Some((_, since, _)) = &mut self.last_tool_action {
            *since += dt;
        }
        if let Some(started) = &mut self.request_started_at {
            *started += dt;
        }
    }

    /// Toggle the most recent expandable output block (Ctrl+O) — the keyboard
    /// path to the `▸ +N lines` click. Returns false when there is none.
    pub(super) fn toggle_latest_output(&mut self) -> bool {
        let Some(&idx) = self.expandable_output_indices().last() else {
            return false;
        };
        if !self.expanded_output.insert(idx) {
            self.expanded_output.remove(&idx);
        }
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        true
    }

    /// If transcript `row` is a `▸ +N more lines` / `▾ collapse` output marker, toggle
    /// that `local_command` block's inline expansion and return `true`. Mirrors
    /// [`Self::toggle_thinking_at_row`]: counts marker rows from the top for the
    /// block's ordinal, then maps it to a history index via
    /// [`Self::expandable_output_indices`]. A click on a live run's marker (ordinal
    /// past the committed blocks) is ignored.
    pub(super) fn toggle_output_at_row(&mut self, row: usize) -> bool {
        let ordinal = {
            let Some(hitbox) = self.transcript_hitbox.as_ref() else {
                return false;
            };
            if !hitbox.rows.get(row).is_some_and(|r| is_output_expander(r)) {
                return false;
            }
            hitbox.rows[..=row]
                .iter()
                .filter(|r| is_output_expander(r))
                .count()
        };
        let Some(&idx) = self.expandable_output_indices().get(ordinal - 1) else {
            return false;
        };
        if !self.expanded_output.insert(idx) {
            self.expanded_output.remove(&idx);
        }
        // The memoized body keys on `transcript_revision`; bump so the flip repaints.
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        true
    }

    /// If transcript `row` is a `▸`/`▾ thought` header, toggle that block's inline
    /// expansion and return `true`. Counts header rows from the top to get the
    /// block's ordinal, then maps it to a committed message via
    /// [`Self::reasoning_message_indices`] — they stay in lockstep because each
    /// committed block renders exactly one header row. A click on the live
    /// streaming summary (ordinal past the committed blocks) is ignored.
    pub(super) fn toggle_thinking_at_row(&mut self, row: usize) -> bool {
        let ordinal = {
            let Some(hitbox) = self.transcript_hitbox.as_ref() else {
                return false;
            };
            if !hitbox.rows.get(row).is_some_and(|r| is_thinking_header(r)) {
                return false;
            }
            hitbox.rows[..=row]
                .iter()
                .filter(|r| is_thinking_header(r))
                .count()
        };
        let Some(&idx) = self.reasoning_message_indices().get(ordinal - 1) else {
            return false;
        };
        if !self.expanded_thinking.insert(idx) {
            self.expanded_thinking.remove(&idx);
        }
        // The memoized body keys on `transcript_revision`; bump so the flip repaints.
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        true
    }

    /// Maps a mouse position to a `screen_surface` point (absolute coordinates).
    /// `clamp` pins an off-screen drag to the edge instead of returning `None`.
    fn screen_point_for_mouse(&self, mouse: MouseEvent, clamp: bool) -> Option<TranscriptPoint> {
        let surface = self.screen_surface.as_ref()?;
        let area = surface.area;
        if !clamp && !rect_contains(area, (mouse.column, mouse.row)) {
            return None;
        }
        let max_x = area.x.saturating_add(area.width.saturating_sub(1));
        let max_y = area.y.saturating_add(area.height.saturating_sub(1));
        let column = mouse.column.clamp(area.x, max_x).saturating_sub(area.x);
        let row = mouse.row.clamp(area.y, max_y).saturating_sub(area.y);
        Some(TranscriptPoint {
            row: usize::from(row),
            column,
        })
    }

    async fn handle_overlay_mouse(&mut self, mouse: MouseEvent) -> Result<Option<bool>> {
        // A left press/drag/release over a non-picker overlay falls through to the
        // screen selection, so the help / skills / mcp bodies are selectable; wheel +
        // picker clicks stay handled below.
        if matches!(
            mouse.kind,
            MouseEventKind::Down(MouseButton::Left)
                | MouseEventKind::Drag(MouseButton::Left)
                | MouseEventKind::Up(MouseButton::Left)
        ) && !matches!(self.overlay, Overlay::Picker(_) | Overlay::None)
        {
            return Ok(None);
        }
        match (&self.overlay, mouse.kind) {
            (Overlay::Help { .. }, MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                if let Overlay::Help { scroll } = &mut self.overlay {
                    *scroll = wheel_scroll(*scroll, up);
                }
                Ok(Some(false))
            }
            (Overlay::Help { .. }, _) => Ok(Some(false)),
            (Overlay::Context { .. }, MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                if let Overlay::Context { scroll, .. } = &mut self.overlay {
                    *scroll = wheel_scroll(*scroll, up);
                }
                Ok(Some(false))
            }
            (Overlay::Context { .. }, _) => Ok(Some(false)),
            // Wheel: detail scroll in a drill-in or over the split's right pane,
            // else selection move; add-input ignores it.
            (Overlay::Skills(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                let over_detail = self
                    .overlay_detail_area
                    .is_some_and(|area| rect_contains(area, (mouse.column, mouse.row)));
                if let Overlay::Skills(state) = &mut self.overlay {
                    if state.viewing.is_some() || over_detail {
                        state.detail_scroll = wheel_scroll(state.detail_scroll, up);
                    } else if state.adding.is_none() {
                        if up {
                            state.select_prev();
                        } else {
                            state.select_next();
                        }
                    }
                }
                Ok(Some(false))
            }
            (Overlay::Skills(_), _) => Ok(Some(false)),
            (Overlay::Agents(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                let over_detail = self
                    .overlay_detail_area
                    .is_some_and(|area| rect_contains(area, (mouse.column, mouse.row)));
                if let Overlay::Agents(state) = &mut self.overlay {
                    if state.viewing.is_some() || over_detail {
                        state.detail_scroll = wheel_scroll(state.detail_scroll, up);
                    } else if up {
                        state.select_prev();
                    } else {
                        state.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::Agents(_), _) => Ok(Some(false)),
            (Overlay::SkillInstall(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                let over_detail = self
                    .overlay_detail_area
                    .is_some_and(|area| rect_contains(area, (mouse.column, mouse.row)));
                if let Overlay::SkillInstall(state) = &mut self.overlay {
                    if state.viewing.is_some() || over_detail {
                        state.detail_scroll = wheel_scroll(state.detail_scroll, up);
                    } else if up {
                        state.select_prev();
                    } else {
                        state.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::SkillInstall(_), _) => Ok(Some(false)),
            (Overlay::Mcp(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                let over_detail = self
                    .overlay_detail_area
                    .is_some_and(|area| rect_contains(area, (mouse.column, mouse.row)));
                if let Overlay::Mcp(state) = &mut self.overlay {
                    if state.viewing.is_some() || over_detail {
                        state.detail_scroll = wheel_scroll(state.detail_scroll, up);
                    } else if state.adding.is_none() {
                        if up {
                            state.select_prev();
                        } else {
                            state.select_next();
                        }
                    }
                }
                Ok(Some(false))
            }
            (Overlay::Mcp(_), _) => Ok(Some(false)),
            (Overlay::McpTools(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                if let Overlay::McpTools(state) = &mut self.overlay {
                    if up {
                        state.select_prev();
                    } else {
                        state.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::McpTools(_), _) => Ok(Some(false)),
            (Overlay::McpPaste(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                if let Overlay::McpPaste(state) = &mut self.overlay {
                    if up {
                        state.select_prev();
                    } else {
                        state.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::McpPaste(_), _) => Ok(Some(false)),
            (Overlay::Config(_), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown) => {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                if let Overlay::Config(state) = &mut self.overlay {
                    if up {
                        state.select_prev();
                    } else {
                        state.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::Config(_), _) => Ok(Some(false)),
            (Overlay::Picker(picker), MouseEventKind::ScrollUp | MouseEventKind::ScrollDown)
                if !picker.loading =>
            {
                let up = matches!(mouse.kind, MouseEventKind::ScrollUp);
                // Over the preview pane the wheel pages history (up = older); else selection.
                let over_preview = self
                    .overlay_detail_area
                    .is_some_and(|area| rect_contains(area, (mouse.column, mouse.row)));
                if let Overlay::Picker(picker) = &mut self.overlay {
                    if over_preview && matches!(picker.kind, PickerKind::Session) {
                        let sid =
                            picker
                                .filtered_items()
                                .get(picker.selected)
                                .and_then(|(_, item)| match &item.value {
                                    PickerValue::Session(preview) => {
                                        Some(preview.session_id.clone())
                                    }
                                    _ => None,
                                });
                        picker.preview_scroll = wheel_scroll(picker.preview_scroll, !up);
                        picker.preview_scroll_for = sid;
                    } else if up {
                        picker.select_prev();
                    } else {
                        picker.select_next();
                    }
                }
                Ok(Some(false))
            }
            (Overlay::Picker(picker), MouseEventKind::Down(MouseButton::Left))
                if !picker.loading =>
            {
                self.handle_picker_click(mouse).await
            }
            (Overlay::Picker(_), _) => Ok(Some(false)),
            (Overlay::None, _) => Ok(None),
        }
    }

    async fn handle_picker_click(&mut self, mouse: MouseEvent) -> Result<Option<bool>> {
        let Some(hitbox) = &self.picker_hitbox else {
            return Ok(Some(false));
        };

        let point = (mouse.column, mouse.row);
        if rect_contains(hitbox.list_area, point) {
            let row = usize::from(mouse.row.saturating_sub(hitbox.list_area.y));
            if let Some(Some(filtered_index)) = hitbox.row_to_filtered_index.get(row) {
                return self
                    .activate_picker_selection(*filtered_index)
                    .await
                    .map(Some);
            }
        } else if !rect_contains(hitbox.overlay_area, point) {
            self.overlay = Overlay::None;
        }

        Ok(Some(false))
    }
}

/// State for stitching a mouse report that crossterm split at its leading ESC
/// back together (see `drain_input`). `Idle` is the common case; `PendingEsc`
/// holds a bare Esc whose fate is undecided; `Sgr` accumulates the `[<…`
/// fragment once the CSI lead is confirmed.
pub(super) enum EscReassembly {
    Idle,
    PendingEsc,
    Sgr(String),
}

/// Outcome of feeding one event through the reassembler.
pub(super) enum EscStep {
    /// The event was folded into held state; nothing more to do.
    Consumed,
    /// Handling held input asked the app to exit.
    Exit,
    /// A plain event the caller should handle as usual.
    Passthrough(Event),
}

/// How an SGR mouse fragment (always starting `[`) grows one character at a time.
pub(super) enum FragStep {
    /// A valid-so-far prefix of `[<{params}` — keep accumulating.
    Continue,
    /// A complete `[<{params}{M|m}` report.
    Final,
    /// The run can't be an SGR mouse report; treat what we held as text.
    Invalid,
}

/// Steps an overlay scroll offset by one mouse-wheel notch (`up` decreases it).
/// Clamping is left to the renderer, so over-scrolling past the end is harmless.
fn wheel_scroll(offset: u16, up: bool) -> u16 {
    const WHEEL_LINES: u16 = 3;
    if up {
        offset.saturating_sub(WHEEL_LINES)
    } else {
        offset.saturating_add(WHEEL_LINES)
    }
}

/// `$VISUAL` then `$EDITOR` (shlex-split so `code --wait` works), else a platform default.
fn external_editor_command() -> Vec<String> {
    for var in ["VISUAL", "EDITOR"] {
        if let Ok(value) = env::var(var)
            && let Some(parts) = shlex::split(&value)
            && !parts.is_empty()
        {
            return parts;
        }
    }
    let default = if cfg!(windows) { "notepad" } else { "vi" };
    vec![default.to_string()]
}

/// `true` when `event` is an unmodified `Esc` keypress.
fn is_bare_esc(event: &Event) -> bool {
    matches!(event, Event::Key(k)
        if k.kind == KeyEventKind::Press && k.code == KeyCode::Esc && k.modifiers.is_empty())
}

/// Lead an image-rejection 400 with an actionable line, for models the snapshot
/// didn't know were text-only. The provider wording is the cross-vendor signal.
pub(super) fn reframe_image_input_error(err: String, model: &str) -> String {
    if err.to_ascii_lowercase().contains("image input") {
        format!(
            "{model} can't read images — switch to a vision model (e.g. /model) and resend.\n{err}"
        )
    } else {
        err
    }
}

/// The character of a `Char` keypress, else `None` (ignores modifiers — the
/// leaked fragment bytes arrive as plain unmodified characters).
fn char_press(event: &Event) -> Option<char> {
    match event {
        Event::Key(k) if k.kind == KeyEventKind::Press => match k.code {
            KeyCode::Char(c) => Some(c),
            _ => None,
        },
        _ => None,
    }
}

/// Classifies how `buf` (the accumulated fragment, always beginning with `[`)
/// fits the SGR mouse grammar `[<{digits and ';'}{M|m}`.
pub(super) fn sgr_mouse_frag_step(buf: &str) -> FragStep {
    let Some(body) = buf.strip_prefix("[<") else {
        // Only `[` so far: still waiting for the `<` that marks SGR mouse.
        return if buf == "[" {
            FragStep::Continue
        } else {
            FragStep::Invalid
        };
    };
    let Some(last) = body.chars().last() else {
        return FragStep::Continue; // exactly `[<`
    };
    if last == 'M' || last == 'm' {
        let params = &body[..body.len() - last.len_utf8()];
        if !params.is_empty() && params.chars().all(|c| c.is_ascii_digit() || c == ';') {
            FragStep::Final
        } else {
            FragStep::Invalid
        }
    } else if last.is_ascii_digit() || last == ';' {
        FragStep::Continue
    } else {
        FragStep::Invalid
    }
}

/// Parses a reassembled SGR fragment (`[<{button};{col};{row}{M|m}`) into a
/// wheel-scroll `MouseEvent`. Returns `None` for non-scroll buttons or malformed
/// input. SGR coordinates are 1-based; crossterm's `MouseEvent` is 0-based.
pub(super) fn parse_sgr_scroll(frag: &str) -> Option<MouseEvent> {
    let body = frag.strip_prefix("[<")?;
    let body = body.strip_suffix('M').or_else(|| body.strip_suffix('m'))?;
    let mut parts = body.split(';');
    let button: u16 = parts.next()?.parse().ok()?;
    let col: u16 = parts.next()?.parse().ok()?;
    let row: u16 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let kind = match button {
        64 => MouseEventKind::ScrollUp,
        65 => MouseEventKind::ScrollDown,
        _ => return None,
    };
    Some(MouseEvent {
        kind,
        column: col.saturating_sub(1),
        row: row.saturating_sub(1),
        modifiers: KeyModifiers::NONE,
    })
}
