use super::*;
use crate::commands::models::fetch_models_for_select;
use crate::services::model_metadata::effort_level_is_off;

impl CodeTuiApp {
    pub(super) fn open_model_picker(
        &mut self,
        query: Option<String>,
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    ) {
        let query = query.unwrap_or_default();
        let title = match &target {
            ModelSelectionTarget::VisionDescriber(_) => "Describer model (reads images)",
            ModelSelectionTarget::ImageGenerator(_) => "Generator model (makes images)",
            _ => "Select model",
        };
        self.overlay = Overlay::Picker(Box::new(PickerState::loading(
            title,
            query,
            PickerKind::Model {
                target,
                auto_accept_exact,
            },
        )));
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = match self.current_model_picker_key() {
            Some(key) => key,
            None => return,
        };
        let cache = self.cache.clone();

        tokio::spawn(async move {
            let choices = load_model_choices(&client, &key, &cache).await;
            if choices.is_empty() {
                tx.send(RuntimeEvent::ModelsLoaded(Err(
                    "No models available for this provider".to_string(),
                )))
                .ok();
            } else {
                tx.send(RuntimeEvent::ModelsLoaded(Ok(choices))).ok();
            }
        });
    }

    /// Resolve and cache the active model's context window for the footer
    /// utilization stat. Cheap (in-memory cache) — call after any model/key
    /// change. 0 when the model isn't in the live catalog or snapshot.
    pub(super) async fn refresh_context_window(&mut self) {
        let limits = crate::services::model_metadata::resolve_limits(
            &self.cache,
            Some(&self.key.base_url),
            &self.model,
        )
        .await;
        // A `--max-context` override wins over the resolved window.
        self.context_window = self.context_window_override.or(limits.context).unwrap_or(0);
        // Cursor bakes the effort tier into bare ids; resolve the window from the
        // underlying model + surface the tier. Unknown → footer token-count.
        self.cursor_effort_label = if self.key.is_cursor_acp() {
            let parts = crate::services::cursor_acp::parse_cursor_model(&self.model);
            if self.context_window_override.is_none() {
                self.context_window = parts.context_window().unwrap_or(0);
            }
            parts.effort_label()
        } else {
            None
        };
        // Valid `/effort` levels: live catalog (e.g. aivo/starter) or snapshot.
        self.model_reasoning_efforts = limits.reasoning_efforts.clone();
        // Reasoning-capable per the snapshot, or implied by advertised levels.
        self.model_supports_thinking =
            limits.caps.is_some_and(|c| c.reasoning) || !self.model_reasoning_efforts.is_empty();
        // Snapshot vision support (None when absent); live catalog wins.
        self.model_image_input = limits.image_input;
        // This model's remembered effort, dropped if no longer a valid level.
        // Off-grade values drop too — off lives in the thinking toggle, never
        // in the level.
        self.reasoning_effort = match self
            .session_store
            .get_chat_reasoning_effort(&self.model)
            .await
        {
            Some(level)
                if self.model_reasoning_efforts.contains(&level)
                    && !effort_level_is_off(&level) =>
            {
                Some(level)
            }
            _ => None,
        };
    }

    /// Set the reasoning effort: remember it (per-model) and persist it. The
    /// engine picks it up at the start of the next turn (carried in like the
    /// context window), so we never lock the engine here — that would block the
    /// event loop on an in-flight turn's guard. Choosing an effort implies you
    /// want the model to think, so this also turns thinking on if it was off.
    pub(super) async fn apply_reasoning_effort(&mut self, level: String) {
        // Under the unified scale an off-grade value IS the off switch — e.g.
        // the agent's `set_effort` tool asking for "none".
        if effort_level_is_off(&level) {
            self.set_thinking_enabled(false).await;
            return;
        }
        // A stale picker can apply a level from a model the agent switched away
        // from mid-turn — refuse instead of 400ing later turns.
        if !self.model_reasoning_efforts.is_empty()
            && !self.model_reasoning_efforts.contains(&level)
        {
            self.notice = Some((
                ERROR(),
                format!(
                    "'{level}' isn't a level for {} (choose: {})",
                    self.model,
                    self.model_reasoning_efforts.join(", ")
                ),
            ));
            return;
        }
        self.reasoning_effort = Some(level.clone());
        let _ = self
            .session_store
            .set_chat_reasoning_effort(&self.model, Some(&level))
            .await;
        if !self.thinking_enabled {
            self.set_thinking_enabled(true).await;
        }
        self.notice = Some((MUTED(), format!("reasoning effort: {level}")));
    }

    /// gpt-5.4+ refuses tools + an off-grade effort, so with agent tools on
    /// these models have NO off state at all (see `gpt_rejects_none_with_tools`;
    /// the engine substitutes on the wire, this keeps the UI in step).
    pub(super) fn thinking_off_unavailable(&self) -> bool {
        self.agent_tools_enabled
            && crate::services::model_metadata::gpt_rejects_none_with_tools(&self.model)
    }

    /// The Thinking scale's real levels: the catalog minus off-grade values —
    /// those are spellings of "off" and surface as the unified `off` pill, not
    /// as levels (see `thinking_off_selectable`).
    pub(super) fn displayed_reasoning_levels(&self) -> Vec<String> {
        self.model_reasoning_efforts
            .iter()
            .filter(|l| !effort_level_is_off(l))
            .cloned()
            .collect()
    }

    /// Whether the unified `off` pill exists for this model right now: only
    /// where the engine's off translation (`thinking_off_wire`) goes genuinely
    /// below the lowest real level — an off-grade effort value or the
    /// `thinking:{disabled}` struct. o-series/codex map off to plain `low`
    /// (a pill that duplicates `low` would lie about saving reasoning), and
    /// gpt-5.4+ with tools on refuses off outright.
    pub(super) fn thinking_off_selectable(&self) -> bool {
        if self.thinking_off_unavailable() {
            return false;
        }
        match crate::services::model_metadata::thinking_off_wire(
            &self.model,
            &self.model_reasoning_efforts,
        ) {
            (Some(level), _) => effort_level_is_off(level),
            (None, disable) => disable,
        }
    }

    /// Ctrl+T: step the thinking scale — off → lowest level → … → highest →
    /// off. One ring with the `/config` Thinking row; models whose off isn't
    /// selectable ring over the levels alone. A thinking-capable model without
    /// catalog levels just toggles on/off.
    pub(super) async fn cycle_reasoning_effort(&mut self) {
        let levels = self.displayed_reasoning_levels();
        if levels.is_empty() {
            if self.model_supports_thinking {
                let on = self.thinking_enabled;
                self.set_thinking_enabled(!on).await;
                self.flash_thinking_state();
            } else {
                self.show_center_toast(format!("{} has no thinking to cycle", self.model));
            }
            return;
        }
        if !self.thinking_enabled {
            let level = levels[0].clone();
            self.cycle_to_effort(level).await;
            return;
        }
        let pos = self
            .effective_reasoning_effort()
            .and_then(|cur| levels.iter().position(|l| *l == cur));
        match pos {
            Some(idx) if idx + 1 < levels.len() => {
                let level = levels[idx + 1].clone();
                self.cycle_to_effort(level).await;
            }
            // Highest level (or an unknown one) wraps around — through off
            // where it exists, else straight back to the lowest level.
            _ if self.thinking_off_selectable() => {
                self.set_thinking_enabled(false).await;
                self.flash_thinking_state();
            }
            _ => {
                let level = levels[0].clone();
                self.cycle_to_effort(level).await;
            }
        }
    }

    /// A Ctrl+T step onto `level`: the fading flash replaces the lingering
    /// notice `apply_reasoning_effort` leaves.
    async fn cycle_to_effort(&mut self, level: String) {
        self.apply_reasoning_effort(level).await;
        self.notice = None;
        self.flash_thinking_state();
    }

    /// Center-toast the thinking state the cycle just landed on.
    fn flash_thinking_state(&mut self) {
        let text = if !self.thinking_enabled {
            "Thinking off".to_string()
        } else {
            match self.effective_reasoning_effort() {
                Some(level) => format!("Thinking: {level}"),
                None => "Thinking on".to_string(),
            }
        };
        self.show_center_toast(text);
    }

    /// The effort the engine will request while thinking is on: the user's
    /// choice, else the default (env or `medium`) if valid, else the middle
    /// real level; `None` when the model has no levels. Keeps the footer badge
    /// in step with what's sent and lets catalog-only models reason by default.
    pub(super) fn effective_reasoning_effort(&self) -> Option<String> {
        if let Some(level) = &self.reasoning_effort {
            return Some(level.clone());
        }
        // Runs every footer render: borrow the levels, clone only the pick.
        let levels: Vec<&String> = self
            .model_reasoning_efforts
            .iter()
            .filter(|l| !effort_level_is_off(l))
            .collect();
        if levels.is_empty() {
            return None;
        }
        let default = crate::agent::engine::default_reasoning_effort_level();
        if levels.iter().any(|l| **l == default) {
            return Some(default);
        }
        Some(levels[(levels.len() - 1) / 2].clone())
    }

    /// `/model <name>`: apply the named model verbatim, no catalog lookup.
    pub(super) async fn set_model_direct(&mut self, name: String) -> Result<()> {
        let applied = self.apply_model(name.clone()).await?;
        let msg = if !applied {
            // Live cursor session rejected the name (e.g. "auto") and kept its own.
            format!(
                "cursor kept its current model — \"{name}\" isn't selectable on the live session"
            )
        } else if self.sending {
            format!("Now using {name} — applies from the next turn")
        } else {
            format!("Now using {name}")
        };
        self.notice = Some((MUTED(), msg));
        Ok(())
    }

    /// Apply a model. Returns `false` only when a live cursor session rejected
    /// the name and kept its own (the picker's `"auto"`) — callers surface it.
    pub(super) async fn apply_model(&mut self, raw_model: String) -> Result<bool> {
        self.persist_model_selection(&raw_model).await?;

        self.raw_model = raw_model.clone();
        self.model = CodeCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.refresh_context_window().await;
        // Routes are per-model — re-seed the format for the new model.
        self.format = seeded_chat_format(&self.key, &raw_model);
        self.billed_model = None;
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.notice = None;

        // If we have a live cursor ACP session, switch its model in place so
        // the conversation context is preserved. Drop the session on failure
        // so the next turn opens a fresh one with the new model.
        let switch = if let Some(session) = self.cursor_acp_session.as_mut() {
            Some(session.set_model(&raw_model).await)
        } else {
            None
        };
        let applied = match switch {
            // Errored — drop so the next turn reopens on the new model.
            Some(Err(_)) => {
                self.cursor_acp_session = None;
                true
            }
            // No catalog match (e.g. "auto"): session kept its model — report it.
            Some(Ok(false)) => false,
            Some(Ok(true)) | None => true,
        };

        if !self.history.is_empty() {
            self.persist_history().await?;
        }
        Ok(applied)
    }

    /// Back the agent's `switch_model` tool: resolve `requested` against the catalog and
    /// `apply_model` it (takes effect next turn). Err = why not, so the agent can guide to `/model`.
    pub(super) async fn agent_switch_model(
        &mut self,
        requested: String,
    ) -> std::result::Result<String, String> {
        let requested = requested.trim().to_string();
        if requested.is_empty() {
            return Err("no model given".to_string());
        }
        if self.raw_model.eq_ignore_ascii_case(&requested) {
            return Ok(format!("Already using {}.", self.raw_model));
        }
        let choices = load_model_choices(&self.client, &self.key, &self.cache).await;
        let resolved = resolve_model_request(&requested, &choices)?;
        self.apply_model(resolved.clone())
            .await
            .map_err(|e| format!("couldn't switch model: {e}"))?;
        Ok(format!(
            "Switched to {resolved}. It takes effect on the user's next message; the conversation \
is preserved."
        ))
    }

    /// Back the agent's `switch_key` tool: move the session onto `requested`, landing on `model`
    /// (else that key's last-used one). Applies next turn, like `switch_model`.
    ///
    /// Deliberately NOT `complete_key_switch`: this runs mid-turn with the engine lock held, so
    /// its `reset_engine_preserving_conversation` would fall back to the lossy history seed. The
    /// next-turn rebuild fires on the changed `key_id` and replays the conversation verbatim.
    pub(super) async fn agent_switch_key(
        &mut self,
        requested: String,
        model: Option<String>,
    ) -> std::result::Result<String, String> {
        let keys = self
            .session_store
            .get_keys()
            .await
            .map_err(|e| format!("couldn't read the saved keys: {e}"))?;
        let mut key = resolve_key_request(requested.trim(), &keys)?;
        SessionStore::decrypt_key_secret(&mut key)
            .map_err(|e| format!("couldn't unlock key {}: {e}", key.display_name()))?;

        let raw_model = match model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
            Some(m) => {
                let choices = load_model_choices(&self.client, &key, &self.cache).await;
                resolve_model_request(m, &choices)?
            }
            None => match self.session_store.get_code_model(&key.id).await {
                Ok(Some(m)) => m,
                _ => {
                    let choices = load_model_choices(&self.client, &key, &self.cache).await;
                    return Err(pick_a_model_error(&key, &choices));
                }
            },
        };

        if key.id == self.key.id && self.raw_model == raw_model {
            return Ok(format!(
                "Already using {} with {raw_model}.",
                key.display_name()
            ));
        }

        self.key = key;
        self.raw_model = raw_model.clone();
        self.model = CodeCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.billed_model = None;
        self.copilot_tm = copilot_token_manager_for_key(&self.key);
        self.format = seeded_chat_format(&self.key, &raw_model);
        // Auth changed — a live cursor session can't carry across it.
        self.cursor_acp_session = None;
        self.persist_model_selection(&raw_model)
            .await
            .map_err(|e| format!("couldn't save the key switch: {e}"))?;
        self.refresh_context_window().await;

        let name = self.key.display_name().to_string();
        self.notice = Some((
            MUTED(),
            format!("Agent switched to {name} / {raw_model} — applies from the next turn"),
        ));
        Ok(format!(
            "Switched to {name} on {raw_model}. It takes effect on the user's next message; the \
conversation is preserved."
        ))
    }

    /// Back the agent's `set_effort` tool: validate `level` against the model's levels and apply it.
    pub(super) async fn agent_set_effort(
        &mut self,
        level: String,
    ) -> std::result::Result<String, String> {
        let level = level.trim().to_ascii_lowercase();
        if self.model_reasoning_efforts.is_empty() {
            return Err(format!(
                "{} has no reasoning-effort levels to set.",
                self.raw_model
            ));
        }
        if !self.model_reasoning_efforts.contains(&level) {
            return Err(format!(
                "'{level}' isn't a valid effort for {}. Options: {}.",
                self.raw_model,
                self.model_reasoning_efforts.join(", ")
            ));
        }
        self.apply_reasoning_effort(level.clone()).await;
        Ok(format!("Reasoning effort set to {level}."))
    }

    pub(super) async fn complete_key_switch(
        &mut self,
        key: ApiKey,
        raw_model: String,
    ) -> Result<()> {
        let cross_provider = !same_wire_provider(&self.key.base_url, &key.base_url);

        self.key = key;
        self.raw_model = raw_model.clone();
        self.model = CodeCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.billed_model = None;
        self.copilot_tm = copilot_token_manager_for_key(&self.key);
        self.persist_model_selection(&raw_model).await?;
        self.refresh_context_window().await;

        if self.history.is_empty() {
            self.start_new_chat().await;
            return Ok(());
        }

        // Keep the conversation across the switch — even to a different provider.
        // The engine's transcript is stored as provider-agnostic OpenAI-wire
        // messages and replayed on the new provider through aivo serve (the same
        // path a foreign import resumes on). Carry it across the rebuild; drop the
        // cursor session (auth changed).
        self.format = seeded_chat_format(&self.key, &raw_model);
        self.reset_engine_preserving_conversation();
        self.cursor_acp_session = None;
        self.persist_history().await?;
        self.notice = Some((
            MUTED(),
            if cross_provider {
                format!(
                    "Switched to {} — conversation kept, continues on the new provider",
                    self.key.display_name()
                )
            } else {
                format!(
                    "Switched key to {} — session preserved",
                    self.key.display_name()
                )
            },
        ));
        Ok(())
    }

    pub(super) async fn open_or_switch_key(&mut self, query: Option<String>) -> Result<()> {
        if let Some(query) = query {
            if let Some(key) = self.resolve_key_exact(&query).await? {
                self.begin_key_switch(key).await?;
                return Ok(());
            }
            self.open_key_picker(Some(query), None).await?;
            return Ok(());
        }

        self.open_key_picker(None, None).await
    }

    /// Always via the model picker (saved model focused) so an old choice is
    /// confirmed, not silently reused; the switch lands on pick, Esc cancels.
    pub(super) async fn begin_key_switch(&mut self, mut key: ApiKey) -> Result<()> {
        SessionStore::decrypt_key_secret(&mut key)?;
        let prior_model = self.session_store.get_code_model(&key.id).await?;
        self.open_model_picker(
            None,
            ModelSelectionTarget::KeySwitch { key, prior_model },
            false,
        );
        Ok(())
    }

    /// `focus` overrides the active-key default (Esc-back refocuses the drilled key).
    pub(super) async fn open_key_picker(
        &mut self,
        query: Option<String>,
        focus: Option<String>,
    ) -> Result<()> {
        let keys = self.session_store.get_keys().await?;
        if keys.is_empty() {
            self.notice = Some((ERROR(), "No saved keys".to_string()));
            return Ok(());
        }

        let mut picker = PickerState::ready(
            "Keys",
            query.unwrap_or_default(),
            key_picker_items(keys),
            PickerKind::Key {
                target: KeySelectionTarget::Switch,
            },
        );
        let focus_id = focus.as_deref().unwrap_or(&self.key.id);
        picker.selected = picker
            .filtered_items()
            .iter()
            .position(
                |(_, item)| matches!(&item.value, PickerValue::Key(key) if key.id == focus_id),
            )
            .unwrap_or(0);
        self.overlay = Overlay::Picker(Box::new(picker));
        Ok(())
    }

    pub(super) async fn open_vision_picker_for_key(&mut self, query: &str) {
        match super::resolve_describer_key(&self.session_store, query).await {
            Ok(key) => self.open_describer_model_picker(key),
            Err(message) => self.notice = Some((ERROR(), message)),
        }
    }

    /// Picker items carry encrypted secrets (`get_keys`), but the live model
    /// fetch has to authenticate with the real one.
    fn open_describer_model_picker(&mut self, mut key: ApiKey) {
        if let Err(e) = SessionStore::decrypt_key_secret(&mut key) {
            self.set_config_error(
                ConfigSetting::VisionFallback,
                format!("couldn't unlock key: {e}"),
            );
            return;
        }
        self.open_model_picker(None, ModelSelectionTarget::VisionDescriber(key), false);
    }

    /// One eligible key skips straight to the model stage; the stored describer
    /// pre-selects.
    pub(super) async fn open_vision_key_picker(&mut self) {
        let keys = match self.session_store.get_keys().await {
            Ok(keys) => keys,
            Err(e) => {
                self.set_config_error(
                    ConfigSetting::VisionFallback,
                    format!("couldn't read keys: {e}"),
                );
                return;
            }
        };
        let mut eligible: Vec<ApiKey> = keys
            .into_iter()
            .filter(crate::commands::code_agent_oneshot::key_is_agent_capable)
            .collect();
        if eligible.is_empty() {
            self.set_config_error(
                ConfigSetting::VisionFallback,
                "No key can serve as a describer (needs an API key, not launch-bound OAuth/ACP)",
            );
            return;
        }
        if eligible.len() == 1 {
            self.open_describer_model_picker(eligible.remove(0));
            return;
        }
        let stored_key = self
            .vision_fallback_custom
            .as_ref()
            .and_then(|(id, _)| eligible.iter().position(|k| &k.id == id))
            .unwrap_or(0);
        let mut picker = PickerState::ready(
            "Describer key",
            String::new(),
            key_picker_items(eligible),
            PickerKind::Key {
                target: KeySelectionTarget::VisionDescriber,
            },
        );
        picker.selected = stored_key;
        self.overlay = Overlay::Picker(Box::new(picker));
    }

    /// One eligible key skips the key stage; the stored generator pre-selects.
    pub(super) async fn open_image_gen_key_picker(&mut self) {
        let keys = match self.session_store.get_keys().await {
            Ok(keys) => keys,
            Err(e) => {
                self.set_config_error(ConfigSetting::ImageGen, format!("couldn't read keys: {e}"));
                return;
            }
        };
        let mut eligible: Vec<ApiKey> = keys
            .into_iter()
            .filter(crate::commands::code_agent_oneshot::key_is_agent_capable)
            .collect();
        if eligible.is_empty() {
            self.set_config_error(
                ConfigSetting::ImageGen,
                "No key can serve as a generator (needs an API key, not launch-bound OAuth/ACP)",
            );
            return;
        }
        if eligible.len() == 1 {
            self.open_image_generator_model_picker(eligible.remove(0));
            return;
        }
        let stored_key = self
            .image_gen_custom
            .as_ref()
            .and_then(|(id, _)| eligible.iter().position(|k| &k.id == id))
            .unwrap_or(0);
        let mut picker = PickerState::ready(
            "Generator key",
            String::new(),
            key_picker_items(eligible),
            PickerKind::Key {
                target: KeySelectionTarget::ImageGenerator,
            },
        );
        picker.selected = stored_key;
        self.overlay = Overlay::Picker(Box::new(picker));
    }

    /// Picker items carry encrypted secrets, but the live model fetch has to
    /// authenticate with the real one.
    fn open_image_generator_model_picker(&mut self, mut key: ApiKey) {
        if let Err(e) = SessionStore::decrypt_key_secret(&mut key) {
            self.set_config_error(ConfigSetting::ImageGen, format!("couldn't unlock key: {e}"));
            return;
        }
        self.open_model_picker(None, ModelSelectionTarget::ImageGenerator(key), false);
    }

    pub(super) async fn open_resume_picker(&mut self, query: Option<String>) -> Result<()> {
        // Scoped to the launch dir; an empty real_cwd can't scope → all.
        let scope_cwd = (!self.real_cwd.is_empty()).then(|| self.real_cwd.clone());
        let mut sessions = load_resume_snapshots(&self.session_store, scope_cwd.as_deref()).await?;
        if !self.history.is_empty()
            && !sessions
                .iter()
                .any(|session| session.session_id == self.session_id)
        {
            self.persist_history().await?;
            sessions = load_resume_snapshots(&self.session_store, scope_cwd.as_deref()).await?;
        }

        // Fold in this dir's importable foreign sessions (Claude Code / Codex /
        // Pi), skipping any already imported (their id is already a native row).
        // Not for `--resume last` — it only jumps to a native session, so it need
        // not pay the foreign-directory scan.
        if query.as_deref() != Some("last")
            && let Some(cwd) = scope_cwd.as_deref()
        {
            let native: std::collections::HashSet<String> =
                sessions.iter().map(|s| s.session_id.clone()).collect();
            for preview in load_importable_previews(cwd).await {
                if !native.contains(&preview.session_id) {
                    sessions.push(preview);
                }
            }
            sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        }

        // `--resume last`: jump to the most recent session in this dir. In-session
        // the current chat was just persisted and sorts newest, so skip it.
        if query.as_deref() == Some("last") {
            // `last` means the most recent aivo session — never silently import a
            // foreign one (that's an explicit pick in the bare `/resume` picker).
            let pick = sessions
                .iter()
                .find(|session| session.origin.is_none() && session.session_id != self.session_id);
            match pick {
                Some(snapshot) => self.begin_resume_load(snapshot.clone()),
                None => {
                    self.notice = Some((MUTED(), "No saved session in this directory".to_string()))
                }
            }
            return Ok(());
        }

        // Explicit id: exact hit in the loaded rows first, then the shared
        // resolver — unique prefix over saved aivo sessions (globally — a named
        // session resolves regardless of where it ran) or this dir's
        // importables (aivo import id or the source tool's own id).
        if let Some(query) = &query {
            if let Some(snapshot) = sessions.iter().find(|session| session.session_id == *query) {
                self.begin_resume_load(snapshot.clone());
                return Ok(());
            }
            use crate::services::session_import::{ResumeTarget, resolve_resume_target};
            match resolve_resume_target(
                &self.session_store,
                std::path::Path::new(&self.real_cwd),
                query,
            )
            .await
            {
                ResumeTarget::AivoSession(full) => {
                    let snapshot = match sessions.iter().find(|s| s.session_id == full) {
                        Some(s) => Some(s.clone()),
                        None => load_resume_snapshots(&self.session_store, None)
                            .await?
                            .into_iter()
                            .find(|s| s.session_id == full),
                    };
                    if let Some(snapshot) = snapshot {
                        self.begin_resume_load(snapshot);
                        return Ok(());
                    }
                }
                ResumeTarget::Foreign(imp) => {
                    self.begin_resume_load(SessionPreview::from_importable(*imp));
                    return Ok(());
                }
                ResumeTarget::Ambiguous(msg) => {
                    self.notice = Some((MUTED(), msg));
                    return Ok(());
                }
                // Fall through to the filtered picker.
                ResumeTarget::Unknown => {}
            }
        }

        // Empty scoped list → notice, not an unfilterable empty picker.
        if sessions.is_empty() {
            self.notice = Some((
                MUTED(),
                "No saved sessions in this directory to resume".to_string(),
            ));
            return Ok(());
        }

        let items = sessions
            .into_iter()
            .map(|session| PickerEntry {
                label: session.title.clone(),
                search_text: session.search_text(),
                value: PickerValue::Session(session),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            query.unwrap_or_default(),
            items,
            PickerKind::Session,
        )));
        Ok(())
    }

    pub(super) fn open_help_overlay(&mut self) {
        self.overlay = Overlay::Help { scroll: 0 };
    }

    /// `/session` (or clicking the footer id): the session-detail overlay.
    pub(super) fn open_session_overlay(&mut self) {
        self.overlay = Overlay::Session { scroll: 0 };
    }

    /// `/context`: the context-window breakdown viewer.
    pub(super) async fn open_context_overlay(&mut self) {
        let report = self.compute_context_report().await;
        self.overlay = Overlay::Context {
            report: Box::new(report),
            scroll: 0,
        };
    }

    /// The live engine's report when built and idle; else a preview. `try_lock` (not
    /// `lock`) because `/context` runs mid-turn and the response task holds the engine.
    async fn compute_context_report(&self) -> crate::agent::engine::ContextReport {
        if let Some(session) = self.agent_engine.as_ref()
            && let Ok(engine) = session.engine.try_lock()
        {
            return engine.context_report();
        }
        self.preview_context_report().await
    }

    /// A throwaway engine mirroring the send path's ingredients, for before the first
    /// turn (or while the live one is busy). Read-only: only already-connected MCP counts.
    async fn preview_context_report(&self) -> crate::agent::engine::ContextReport {
        use crate::agent::engine::AgentEngine;
        let real_cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let cwd = std::path::Path::new(&real_cwd);
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let guides = crate::agent::system_prompt::discover_project_guides(cwd);
        // Same assembler as the send path, so the report reflects the real prompt.
        let disabled: std::collections::HashSet<String> = self
            .session_store
            .get_disabled_skills()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        let skills = crate::agent::skills::engine_skills(cwd, &disabled);
        let window = self.context_window.min(u32::MAX as u64) as u32;
        let mut engine =
            AgentEngine::new(&real_cwd, &self.model, &date, &guides, &skills, window, 0);
        let subagents =
            crate::agent::subagents::discover_subagents(cwd, self.session_store.config_dir());
        engine.set_subagents(&subagents);
        if let Some(ctx) = self.injected_context.as_deref() {
            engine.append_system_context(ctx);
        }
        // Only already-connected servers are in context; mirror the Ctrl+T filter.
        if let Some(client) = &self.mcp_client
            && client.has_tools()
        {
            let disabled: std::collections::HashSet<String> = self
                .session_store
                .get_disabled_mcp_tools()
                .await
                .unwrap_or_default()
                .into_iter()
                .collect();
            if disabled.is_empty() {
                engine.set_external_tools(client.clone());
            } else {
                engine.set_external_tools(std::sync::Arc::new(
                    crate::agent::mcp::FilteredTools::new(client.clone(), disabled),
                ));
            }
        }
        // A resumed transcript (pending until the first send) beats the lossy display seed.
        if let Some(conversation) = self.pending_agent_messages.clone() {
            engine.restore_conversation(conversation);
        } else {
            engine.seed_history(super::runtime_impl::agent_seed_turns(
                &self.history,
                &self.vision_descriptions,
            ));
        }
        engine.context_report()
    }

    /// Idle footer fill: the exact total `/context` shows, via the same
    /// `compute_context_report`, so the two never disagree.
    pub(super) async fn estimated_context_used(&self) -> u64 {
        self.compute_context_report().await.used()
    }

    /// `/config`: a fixed list of chat-preference segmented switches, seeded live.
    pub(super) fn open_config_overlay(&mut self) {
        let items = vec![
            ConfigRow {
                setting: ConfigSetting::Theme,
                label: "Theme",
            },
            ConfigRow {
                setting: ConfigSetting::InlineImages,
                label: "Inline images",
            },
            ConfigRow {
                setting: ConfigSetting::Thinking,
                label: "Thinking",
            },
            ConfigRow {
                setting: ConfigSetting::FooterTps,
                label: "Footer tok/s",
            },
            ConfigRow {
                setting: ConfigSetting::FooterCacheHit,
                label: "Footer cache hit",
            },
            ConfigRow {
                setting: ConfigSetting::FooterPrice,
                label: "Footer price",
            },
            ConfigRow {
                setting: ConfigSetting::Approval,
                label: "Mode",
            },
            ConfigRow {
                setting: ConfigSetting::UseWebSearch,
                label: "Web search",
            },
            ConfigRow {
                setting: ConfigSetting::AgentTools,
                label: "Agent tools",
            },
            ConfigRow {
                setting: ConfigSetting::VisionFallback,
                label: "Vision fallback",
            },
            ConfigRow {
                setting: ConfigSetting::ImageGen,
                label: "Image generation",
            },
        ];
        let selected = self
            .last_config_setting
            .and_then(|setting| items.iter().position(|item| item.setting == setting))
            .unwrap_or(0);
        self.overlay = Overlay::Config(ConfigOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            items,
            selected,
            error: None,
            detail_scroll: 0,
        });
    }

    /// Open `/config` with `setting` focused — picker drill-in entry and Esc/error exit.
    pub(super) fn open_config_overlay_at(&mut self, setting: ConfigSetting) {
        self.open_config_overlay();
        if let Overlay::Config(state) = &mut self.overlay
            && let Some(row) = state.items.iter().position(|i| i.setting == setting)
        {
            state.selected = row;
            self.last_config_setting = Some(setting);
        }
    }

    /// A drill-in failure belongs on the setting, not the transcript notice
    /// (the modal sits far below the transcript, so the two never read together).
    pub(super) fn set_config_error(&mut self, setting: ConfigSetting, message: impl Into<String>) {
        self.notice = None;
        self.open_config_overlay_at(setting);
        if let Overlay::Config(state) = &mut self.overlay {
            state.error = Some((setting, message.into()));
        }
    }

    /// What the focused setting's *live* value means — one paragraph, not a
    /// glossary of every alternative (those sit in the segmented control).
    pub(super) fn config_active_help(&self, setting: ConfigSetting) -> String {
        let segs = self.config_segments(setting);
        let opt = segs.options.get(segs.active).map_or("", String::as_str);
        match (setting, opt) {
            (ConfigSetting::Theme, "light") => {
                "Light palette for the whole TUI. Applies immediately.".to_string()
            }
            (ConfigSetting::Theme, _) => {
                "Dark palette for the whole TUI. Applies immediately.".to_string()
            }
            (ConfigSetting::InlineImages, "on") if !self.detected_graphics_caps.enabled() => {
                "On, but image previews aren't available in this terminal — nothing will render."
                    .to_string()
            }
            (ConfigSetting::InlineImages, "on") => {
                "Images in the transcript render inline.".to_string()
            }
            (ConfigSetting::InlineImages, _) => {
                "No inline previews; images appear as text mentions only.".to_string()
            }
            (ConfigSetting::Thinking, "on") if self.model_supports_thinking => {
                "The model reasons before answering. Thinking is shown folded.".to_string()
            }
            (ConfigSetting::Thinking, "on") => {
                "This model has no thinking; the toggle is remembered for models that do."
                    .to_string()
            }
            (ConfigSetting::Thinking, "off") => {
                "Answers directly, without reasoning first.".to_string()
            }
            (ConfigSetting::Thinking, level) if self.thinking_off_unavailable() => {
                format!(
                    "The model reasons at {level} effort. It can't turn thinking off while agent tools are on."
                )
            }
            (ConfigSetting::Thinking, level) => {
                format!("The model reasons at {level} effort before answering. Thinking is shown folded.")
            }
            (ConfigSetting::Approval, "auto-approve") => {
                "Tools run unattended. Plan mode is /plan.".to_string()
            }
            (ConfigSetting::Approval, "review") => {
                "Approve each edit before it's applied.".to_string()
            }
            (ConfigSetting::Approval, _) => {
                "Risky actions ask first. Plan mode is /plan, not a standing setting.".to_string()
            }
            (ConfigSetting::UseWebSearch, "on") => {
                "The agent can search the web through aivo (daily quota).".to_string()
            }
            (ConfigSetting::UseWebSearch, _) => {
                "The agent won't search via aivo.".to_string()
            }
            (ConfigSetting::AgentTools, "on") => {
                "The agent can edit files, run commands, and use tools.".to_string()
            }
            (ConfigSetting::AgentTools, _) => {
                "Plain chat: no tools, no system prompt.".to_string()
            }
            (ConfigSetting::FooterTps, "on") => {
                "Footer shows the generation rate — live during a turn, the last turn's when idle."
                    .to_string()
            }
            (ConfigSetting::FooterTps, _) => "No tok/s figure in the footer.".to_string(),
            (ConfigSetting::FooterCacheHit, "on") => {
                "Footer shows the last turn's prompt-cache hit rate, when the provider reports cache usage."
                    .to_string()
            }
            (ConfigSetting::FooterCacheHit, _) => {
                "No cache-hit figure in the footer.".to_string()
            }
            (ConfigSetting::FooterPrice, "on") => {
                "Footer shows the session's estimated spend (~$) next to the context meter."
                    .to_string()
            }
            (ConfigSetting::FooterPrice, _) => "No spend figure in the footer.".to_string(),
            (ConfigSetting::VisionFallback, "custom") => {
                "The picked model describes images the chat model can't read.".to_string()
            }
            (ConfigSetting::VisionFallback, "off") => {
                "Text-only models refuse images.".to_string()
            }
            (ConfigSetting::VisionFallback, _) => {
                "When the chat model can't read images, aivo describes them (daily quota)."
                    .to_string()
            }
            (ConfigSetting::ImageGen, "custom") => {
                "The agent can generate images with this model.".to_string()
            }
            (ConfigSetting::ImageGen, "aivo") => {
                "The agent generates images through aivo (daily quota).".to_string()
            }
            (ConfigSetting::ImageGen, _) => {
                "The agent can't generate images. Switch to aivo, or to custom to pick a key with an image-output model."
                    .to_string()
            }
        }
    }

    /// The live custom model for vision/image, if that mode is on.
    pub(super) fn config_current_model(&self, setting: ConfigSetting) -> Option<&str> {
        self.config_current_pick(setting).map(|(_, model)| model)
    }

    pub(super) fn config_current_pick(&self, setting: ConfigSetting) -> Option<(&str, &str)> {
        match setting {
            ConfigSetting::VisionFallback => {
                match (&self.vision_fallback, &self.vision_fallback_custom) {
                    (
                        crate::services::session_store::VisionFallbackMode::Custom,
                        Some((key, model)),
                    ) => Some((key.as_str(), model.as_str())),
                    _ => None,
                }
            }
            ConfigSetting::ImageGen => match (&self.image_gen, &self.image_gen_custom) {
                (crate::services::session_store::ImageGenMode::Custom, Some((key, model))) => {
                    Some((key.as_str(), model.as_str()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// List-row value: the live model when a custom pick is on, else the segment label.
    pub(super) fn config_list_value(&self, setting: ConfigSetting) -> String {
        if let Some(model) = self.config_current_model(setting) {
            return model.to_string();
        }
        let segs = self.config_segments(setting);
        segs.options.get(segs.active).cloned().unwrap_or_default()
    }

    pub(super) fn config_has_drill_in(&self, setting: ConfigSetting) -> bool {
        match setting {
            ConfigSetting::VisionFallback => {
                self.vision_fallback == crate::services::session_store::VisionFallbackMode::Custom
            }
            ConfigSetting::ImageGen => {
                self.image_gen == crate::services::session_store::ImageGenMode::Custom
            }
            _ => false,
        }
    }

    /// Segmented values for `setting` and which is live — read fresh so the
    /// rendered row can't drift from its flag.
    pub(super) fn config_segments(&self, setting: ConfigSetting) -> ConfigSegments {
        let owned = |options: &[&str], active: usize| ConfigSegments {
            options: options.iter().map(|s| s.to_string()).collect(),
            active,
        };
        let switch = |on: bool| owned(&["on", "off"], if on { 0 } else { 1 });
        match setting {
            ConfigSetting::Theme => owned(
                &["dark", "light"],
                match self.theme {
                    UiTheme::Dark => 0,
                    UiTheme::Light => 1,
                },
            ),
            ConfigSetting::InlineImages => switch(self.inline_images_enabled),
            // One unified scale: `off` (when selectable, see
            // `thinking_off_selectable`) then the levels in catalog order;
            // without levels a plain toggle.
            ConfigSetting::Thinking => {
                let levels = self.displayed_reasoning_levels();
                if levels.is_empty() {
                    switch(self.thinking_enabled)
                } else {
                    let off = self.thinking_off_selectable();
                    let offset = usize::from(off);
                    let active = if !self.thinking_enabled && off {
                        0
                    } else {
                        // Thinking on — or off unrepresentable: what actually ships.
                        self.effective_reasoning_effort()
                            .and_then(|cur| levels.iter().position(|l| *l == cur))
                            .map_or(0, |i| i + offset)
                    };
                    let mut options = Vec::with_capacity(levels.len() + 1);
                    if off {
                        options.push("off".to_string());
                    }
                    options.extend(levels);
                    ConfigSegments { options, active }
                }
            }
            ConfigSetting::Approval => {
                const OPTIONS: &[&str] = &["normal", "auto-approve", "review"];
                let label = self.approval_mode_label();
                owned(
                    OPTIONS,
                    OPTIONS.iter().position(|o| *o == label).unwrap_or(0),
                )
            }
            ConfigSetting::UseWebSearch => switch(self.web_search_enabled),
            ConfigSetting::AgentTools => switch(self.agent_tools_enabled),
            ConfigSetting::FooterTps => switch(self.footer_tps_enabled),
            ConfigSetting::FooterCacheHit => switch(self.footer_cache_enabled),
            ConfigSetting::FooterPrice => switch(self.footer_price_enabled),
            ConfigSetting::VisionFallback => {
                use crate::services::session_store::VisionFallbackMode;
                owned(
                    &["aivo", "custom", "off"],
                    match self.vision_fallback {
                        VisionFallbackMode::Gateway => 0,
                        VisionFallbackMode::Custom => 1,
                        VisionFallbackMode::Off => 2,
                    },
                )
            }
            ConfigSetting::ImageGen => {
                use crate::services::session_store::ImageGenMode;
                owned(
                    &["aivo", "custom", "off"],
                    match self.image_gen {
                        ImageGenMode::Gateway => 0,
                        ImageGenMode::Custom => 1,
                        ImageGenMode::Off => 2,
                    },
                )
            }
        }
    }

    /// The live standing mode as an Approval label; plan mode clears both flags → `normal`.
    fn approval_mode_label(&self) -> &'static str {
        if self.agent_auto_approve {
            "auto-approve"
        } else if self.agent_review_edits {
            "review"
        } else {
            "normal"
        }
    }

    fn config_row_setting(&self, row: usize) -> Option<ConfigSetting> {
        match &self.overlay {
            Overlay::Config(state) => state.items.get(row).map(|item| item.setting),
            _ => None,
        }
    }

    /// ←/→: move the active segment by `dir`, clamped to the ends.
    pub(super) async fn step_config_setting(&mut self, row: usize, dir: i32) {
        let Some(setting) = self.config_row_setting(row) else {
            return;
        };
        let segs = self.config_segments(setting);
        let last = segs.options.len().saturating_sub(1) as i32;
        let next = (segs.active as i32 + dir).clamp(0, last) as usize;
        self.set_config_segment(setting, next).await;
    }

    /// Advance to the next segment, wrapping. Enter on an already-active `custom`
    /// re-opens the describer picker — the way to CHANGE the pick.
    pub(super) async fn cycle_config_setting(&mut self, row: usize, dir: CycleDir) {
        let Some(setting) = self.config_row_setting(row) else {
            return;
        };
        if dir == CycleDir::Enter
            && setting == ConfigSetting::VisionFallback
            && self.vision_fallback == crate::services::session_store::VisionFallbackMode::Custom
        {
            self.open_vision_key_picker().await;
            return;
        }
        if dir == CycleDir::Enter
            && setting == ConfigSetting::ImageGen
            && self.image_gen == crate::services::session_store::ImageGenMode::Custom
        {
            self.open_image_gen_key_picker().await;
            return;
        }
        let segs = self.config_segments(setting);
        let len = segs.options.len();
        if len == 0 {
            return;
        }
        let step = if dir == CycleDir::Prev { len - 1 } else { 1 };
        self.set_config_segment(setting, (segs.active + step) % len)
            .await;
    }

    /// Mouse: pick a specific segment. Clicking the live `custom` pill re-opens
    /// the key/model picker — same as Enter on that row.
    pub(super) async fn pick_config_segment(&mut self, option: usize) {
        let Overlay::Config(state) = &self.overlay else {
            return;
        };
        let row = state.selected;
        let Some(setting) = self.config_row_setting(row) else {
            return;
        };
        let segs = self.config_segments(setting);
        if segs.active == option {
            if self.config_has_drill_in(setting) {
                self.cycle_config_setting(row, CycleDir::Enter).await;
            }
            return;
        }
        self.set_config_segment(setting, option).await;
    }

    /// Apply `setting`'s `target` segment live (+persist); no-op if already active.
    async fn set_config_segment(&mut self, setting: ConfigSetting, target: usize) {
        let segs = self.config_segments(setting);
        if segs.active == target {
            return;
        }
        match setting {
            ConfigSetting::Theme => {
                let next = if target == 1 {
                    UiTheme::Light
                } else {
                    UiTheme::Dark
                };
                self.set_theme(next).await;
            }
            ConfigSetting::InlineImages => self.set_inline_images_enabled(target == 0).await,
            ConfigSetting::Thinking => {
                let levels = self.displayed_reasoning_levels();
                if levels.is_empty() {
                    self.set_thinking_enabled(target == 0).await;
                } else {
                    let off = self.thinking_off_selectable();
                    if off && target == 0 {
                        self.set_thinking_enabled(false).await;
                    } else if let Some(level) = levels.get(target - usize::from(off)).cloned() {
                        self.apply_reasoning_effort(level).await;
                    }
                }
            }
            ConfigSetting::Approval => {
                let mode = segs.options.get(target).map_or("normal", String::as_str);
                self.set_approval_mode(mode).await;
            }
            ConfigSetting::UseWebSearch => self.set_web_search_enabled(target == 0).await,
            ConfigSetting::AgentTools => self.set_agent_tools_enabled(target == 0).await,
            ConfigSetting::FooterTps => self.set_footer_tps_enabled(target == 0).await,
            ConfigSetting::FooterCacheHit => self.set_footer_cache_enabled(target == 0).await,
            ConfigSetting::FooterPrice => self.set_footer_price_enabled(target == 0).await,
            ConfigSetting::VisionFallback => {
                use crate::services::session_store::VisionFallbackMode;
                let mode =
                    VisionFallbackMode::parse(segs.options.get(target).map_or("", String::as_str));
                match mode {
                    // Unconfigured `custom` opens the key→model picker; the mode
                    // only flips once something is picked (Esc leaves it as-is).
                    VisionFallbackMode::Custom if self.vision_fallback_custom.is_none() => {
                        self.open_vision_key_picker().await
                    }
                    mode => self.set_vision_fallback_mode(mode).await,
                }
            }
            ConfigSetting::ImageGen => {
                use crate::services::session_store::ImageGenMode;
                // `parse` reads the PERSISTED vocabulary ("gateway"); the row
                // shows `aivo`, which would parse to the Off default.
                let mode = match segs.options.get(target).map_or("", String::as_str) {
                    "aivo" => ImageGenMode::Gateway,
                    "custom" => ImageGenMode::Custom,
                    _ => ImageGenMode::Off,
                };
                match mode {
                    ImageGenMode::Custom if self.image_gen_custom.is_none() => {
                        self.open_image_gen_key_picker().await
                    }
                    mode => self.set_image_gen_mode(mode).await,
                }
            }
        }
    }

    /// Set the standing mode by label (`normal`/`auto-approve`/`review`); leaves plan mode first.
    pub(super) async fn set_approval_mode(&mut self, mode: &str) {
        if self.plan_mode {
            self.leave_plan_mode(false).await;
        }
        match mode {
            "auto-approve" => {
                self.set_review_quiet(false);
                self.set_auto_quiet(true);
                self.show_toast("Auto-approve mode — tools run without asking");
            }
            "review" => {
                self.set_auto_quiet(false);
                self.set_review_quiet(true);
                self.show_toast("Review mode — approve each edit");
            }
            _ => {
                self.set_auto_quiet(false);
                self.set_review_quiet(false);
                self.show_toast("Normal mode — risky actions ask first");
            }
        }
    }

    /// Set + persist the theme (idempotent). Bumps `transcript_revision` so
    /// color-memoized spans rebuild against the new palette.
    pub(super) async fn set_theme(&mut self, next: UiTheme) {
        if self.theme == next {
            return;
        }
        self.theme = next;
        set_ui_theme(next);
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.show_toast(format!("Theme: {}", next.label()));
        let persisted = match next {
            UiTheme::Dark => crate::services::session_store::ChatTheme::Dark,
            UiTheme::Light => crate::services::session_store::ChatTheme::Light,
        };
        let _ = self.session_store.set_chat_theme(persisted).await;
    }

    /// Set the thinking on/off flag and persist it (best-effort). Both transcript
    /// fingerprints (history body and volatile tail) key on `thinking_enabled`, so
    /// the flip invalidates the memoized render on its own — no revision bump needed.
    /// The engine reads the flag at the start of the next turn, so a mid-session
    /// toggle applies next turn.
    pub(super) async fn set_thinking_enabled(&mut self, on: bool) {
        if self.thinking_enabled == on {
            return;
        }
        self.thinking_enabled = on;
        self.show_toast(if on { "Thinking on" } else { "Thinking off" });
        let _ = self.session_store.set_chat_thinking_enabled(on).await;
    }

    pub(super) async fn set_footer_tps_enabled(&mut self, on: bool) {
        if self.footer_tps_enabled == on {
            return;
        }
        self.footer_tps_enabled = on;
        self.show_toast(if on {
            "Footer tok/s on"
        } else {
            "Footer tok/s off"
        });
        let _ = self.session_store.set_chat_footer_tps_enabled(on).await;
    }

    pub(super) async fn set_footer_cache_enabled(&mut self, on: bool) {
        if self.footer_cache_enabled == on {
            return;
        }
        self.footer_cache_enabled = on;
        self.show_toast(if on {
            "Footer cache hit on"
        } else {
            "Footer cache hit off"
        });
        let _ = self.session_store.set_chat_footer_cache_enabled(on).await;
    }

    pub(super) async fn set_footer_price_enabled(&mut self, on: bool) {
        if self.footer_price_enabled == on {
            return;
        }
        self.footer_price_enabled = on;
        self.show_toast(if on {
            "Footer price on"
        } else {
            "Footer price off"
        });
        let _ = self.session_store.set_chat_footer_price_enabled(on).await;
    }

    /// `custom` is only ever set with a describer pair in hand (picker flow /
    /// `--vision-model`), so no pairless guard is needed here.
    pub(super) async fn set_vision_fallback_mode(
        &mut self,
        mode: crate::services::session_store::VisionFallbackMode,
    ) {
        use crate::services::session_store::VisionFallbackMode;
        if self.vision_fallback == mode {
            return;
        }
        self.vision_fallback = mode;
        self.clear_config_error_if(ConfigSetting::VisionFallback);
        let toast = match mode {
            VisionFallbackMode::Gateway => {
                "Vision fallback: aivo describes images (daily quota)".to_string()
            }
            VisionFallbackMode::Custom => match &self.vision_fallback_custom {
                Some((_, model)) => format!("Vision fallback: {model} describes images"),
                None => "Vision fallback: custom describer".to_string(),
            },
            VisionFallbackMode::Off => {
                "Vision fallback off — text-only models refuse images".to_string()
            }
        };
        self.show_toast(toast);
        let _ = self
            .session_store
            .set_chat_vision_fallback(mode, None)
            .await;
    }

    pub(super) async fn apply_vision_describer(&mut self, key: ApiKey, model: String) {
        use crate::services::session_store::VisionFallbackMode;
        // Live-over-snapshot: `image_input: false` rejects without a snapshot row.
        let image_input = crate::services::model_metadata::resolve_limits(
            &self.cache,
            Some(&key.base_url),
            &model,
        )
        .await
        .image_input;
        if let Err(message) = super::validate_describer_model(&model, &self.model, image_input) {
            self.set_config_error(ConfigSetting::VisionFallback, message);
            return;
        }
        self.vision_fallback_custom = Some((key.id.clone(), model.clone()));
        self.vision_fallback = VisionFallbackMode::Custom;
        let _ = self
            .session_store
            .set_chat_vision_fallback(VisionFallbackMode::Custom, Some((&key.id, &model)))
            .await;
        self.open_config_overlay_at(ConfigSetting::VisionFallback);
        self.show_toast(format!(
            "Vision fallback: {model} via {}",
            key.display_name()
        ));
    }

    /// `custom` is only ever set with a generator pair in hand (picker flow /
    /// `--image-model`), so no pairless guard is needed here.
    pub(super) async fn set_image_gen_mode(
        &mut self,
        mode: crate::services::session_store::ImageGenMode,
    ) {
        use crate::services::session_store::ImageGenMode;
        if self.image_gen == mode {
            return;
        }
        self.image_gen = mode;
        self.clear_config_error_if(ConfigSetting::ImageGen);
        let toast = match mode {
            ImageGenMode::Gateway => {
                "Image generation: aivo generates images (daily quota)".to_string()
            }
            ImageGenMode::Custom => match &self.image_gen_custom {
                Some((_, model)) => format!("Image generation: {model} (generate_image tool)"),
                None => "Image generation: custom generator".to_string(),
            },
            ImageGenMode::Off => "Image generation off".to_string(),
        };
        self.show_toast(toast);
        let _ = self.session_store.set_chat_image_gen(mode, None).await;
    }

    pub(super) async fn apply_image_generator(&mut self, key: ApiKey, model: String) {
        use crate::services::session_store::ImageGenMode;
        if let Err(message) = super::validate_generator_model(&model, &self.model) {
            self.set_config_error(ConfigSetting::ImageGen, message);
            return;
        }
        self.image_gen_custom = Some((key.id.clone(), model.clone()));
        self.image_gen = ImageGenMode::Custom;
        let _ = self
            .session_store
            .set_chat_image_gen(ImageGenMode::Custom, Some((&key.id, &model)))
            .await;
        self.open_config_overlay_at(ConfigSetting::ImageGen);
        self.show_toast(format!(
            "Image generation: {model} via {}",
            key.display_name()
        ));
    }

    /// Drop a row's overlay error once that setting actually changes.
    fn clear_config_error_if(&mut self, setting: ConfigSetting) {
        if let Overlay::Config(state) = &mut self.overlay
            && state.error.as_ref().is_some_and(|(s, _)| *s == setting)
        {
            state.error = None;
        }
    }

    /// Set the aivo-web-search flag and persist it; the engine applies it next turn.
    pub(super) async fn set_web_search_enabled(&mut self, on: bool) {
        if self.web_search_enabled == on {
            return;
        }
        self.web_search_enabled = on;
        self.show_toast(if on {
            "aivo web search on"
        } else {
            "aivo web search off — the agent won't search via aivo"
        });
        let _ = self.session_store.set_chat_web_search_enabled(on).await;
    }

    pub(super) async fn set_inline_images_enabled(&mut self, on: bool) {
        if self.inline_images_enabled == on {
            return;
        }
        self.inline_images_enabled = on;
        if on {
            self.pending_inline_image_cleanup = false;
            self.inline_images.caps = self.detected_graphics_caps;
        } else {
            // Kitty deletes need the terminal writer — the event loop finishes
            // the disable; the full repaint wipes sixel/half-block rows.
            self.pending_inline_image_cleanup = true;
            self.inline_images.desired.clear();
            self.inline_images.placed.clear();
        }
        self.transcript_revision = self.transcript_revision.wrapping_add(1);
        self.pending_full_repaint = true;
        let _ = self.session_store.set_chat_inline_images_enabled(on).await;
    }

    pub(super) async fn set_agent_tools_enabled(&mut self, on: bool) {
        if self.agent_tools_enabled == on {
            return;
        }
        self.agent_tools_enabled = on;
        self.show_toast(if on {
            "Agent tools on"
        } else {
            "Agent tools off — plain chat (no tools, no system prompt)"
        });
        let _ = self.session_store.set_chat_agent_tools_enabled(on).await;
    }

    /// `/skills`: discover the agent skills available for the working dir and show
    /// them in a toggle overlay, seeded with each skill's persisted enabled state.
    /// Discovery is on-demand (cheap dir reads) so the list reflects skills added
    /// since launch.
    pub(super) async fn open_skills_overlay(&mut self) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let cwd_path = std::path::Path::new(&cwd);
        let disabled: std::collections::HashSet<String> = self
            .session_store
            .get_disabled_skills()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mut items: Vec<SkillToggle> = crate::agent::skills::discover_skills(cwd_path)
            .into_iter()
            .map(|skill| SkillToggle {
                enabled: !disabled.contains(&skill.name),
                // Discovery leaves the body empty (lazy); the detail pane needs it.
                body: skill.instructions().into_owned(),
                scope: crate::agent::skills::skill_scope(&skill.dir, cwd_path),
                description: skill.description,
                dir: skill.dir,
                name: skill.name,
            })
            .collect();
        sort_skill_rows(&mut items);
        // Keep the `/`-menu skill commands in sync with the freshly discovered set
        // (add/install/remove all reopen this overlay, so it's the convergence point).
        self.skill_commands = enabled_skill_commands(&items);
        self.overlay = Overlay::Skills(SkillsOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            items,
            selected: 0,
            query: String::new(),
            adding: None,
            pending_delete: None,
            viewing: None,
            detail_scroll: 0,
        });
        Ok(())
    }

    /// `/agents`: discover the named sub-agents for the working dir and show them
    /// in a filterable list with a detail pane. Discovery is on-demand, so the
    /// list reflects profiles the agent authored since launch.
    pub(super) fn open_agents_overlay(&mut self) {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let config_dir = self.session_store.config_dir().to_path_buf();
        let items: Vec<AgentRow> =
            crate::agent::subagents::discover_subagents(std::path::Path::new(&cwd), &config_dir)
                .into_iter()
                .map(|sa| AgentRow {
                    scope: if sa.is_builtin() {
                        "builtin"
                    } else if sa.repo_local {
                        "repo"
                    } else {
                        "user"
                    },
                    tools: sa.resolved_tools(),
                    name: sa.name,
                    description: sa.description,
                    source: sa.source,
                    model: sa.model,
                    isolation_worktree: sa.isolation_worktree,
                    body: sa.body,
                })
                .collect();
        self.overlay = Overlay::Agents(AgentsOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            items,
            selected: 0,
            query: String::new(),
            pending_delete: None,
            viewing: None,
            detail_scroll: 0,
        });
    }

    /// `/agents` from the composer: bare opens the overlay; `rm <name>` deletes a
    /// profile without opening it. There is no `add` — creation is conversational.
    pub(super) async fn run_agents_command(&mut self, arg: Option<String>) -> Result<()> {
        let Some(arg) = arg.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
            self.open_agents_overlay();
            return Ok(());
        };
        match arg.split_once(char::is_whitespace) {
            Some(("remove" | "rm", rest)) if !rest.trim().is_empty() => {
                self.remove_agent_named(rest.trim()).await
            }
            _ => {
                self.notice = Some((
                    ERROR(),
                    "Usage: /agents [rm <name>] — to create one, just ask (\"make me a code-reviewer subagent\")"
                        .to_string(),
                ));
                Ok(())
            }
        }
    }

    /// Delete a sub-agent's file by name (repo or user scope only). Rebuilds the
    /// engine so the advert and the `agent` enum drop the name next turn.
    pub(super) async fn remove_agent_named(&mut self, name: &str) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let config_dir = self.session_store.config_dir().to_path_buf();
        let Some(sa) =
            crate::agent::subagents::discover_subagents(std::path::Path::new(&cwd), &config_dir)
                .into_iter()
                .find(|s| s.name == name)
        else {
            self.notice = Some((ERROR(), format!("No sub-agent named “{name}”")));
            return Ok(());
        };
        if sa.is_builtin() {
            self.notice = Some((
                ERROR(),
                format!("“{name}” is built into aivo — shadow it with your own {name}.md instead"),
            ));
            return Ok(());
        }
        match std::fs::remove_file(&sa.source) {
            Ok(()) => {
                self.notice = Some((MUTED(), format!("Removed sub-agent “{name}”")));
                self.request_engine_rebuild();
                // Refresh the open overlay so the row disappears immediately.
                if matches!(self.overlay, Overlay::Agents(_)) {
                    self.open_agents_overlay();
                }
            }
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to remove “{name}”: {e}")));
            }
        }
        Ok(())
    }

    /// Ctrl+D-confirmed delete of the `/agents` row at `index`.
    pub(super) async fn remove_agent(&mut self, index: usize) -> Result<()> {
        let Overlay::Agents(state) = &self.overlay else {
            return Ok(());
        };
        let Some(name) = state.items.get(index).map(|it| it.name.clone()) else {
            return Ok(());
        };
        self.remove_agent_named(&name).await
    }

    /// `/skills` from the composer: bare opens the overlay; `add <name>
    /// [description]` scaffolds a new skill and `remove|rm <name>` deletes one,
    /// without opening the overlay first.
    pub(super) async fn run_skills_command(&mut self, arg: Option<String>) -> Result<()> {
        let Some(arg) = arg.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
            return self.open_skills_overlay().await;
        };
        let (verb, rest) = match arg.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (arg.as_str(), ""),
        };
        match verb {
            "add" => self.submit_skill_add(rest.to_string()).await,
            "remove" | "rm" if rest.is_empty() => {
                self.notice = Some((ERROR(), "Usage: /skills rm <name>".to_string()));
                Ok(())
            }
            "remove" | "rm" => self.remove_skill_named(rest).await,
            "update" => {
                let name = (!rest.is_empty()).then(|| rest.to_string());
                self.update_skills_command(name).await
            }
            _ => {
                self.notice = Some((
                    ERROR(),
                    "Usage: /skills [add [-p] <name>|<github:owner/repo> …] [rm <name>] [update [name]]"
                        .to_string(),
                ));
                Ok(())
            }
        }
    }

    /// Flip the enabled state of the skill at `index` in the open `/skills`
    /// overlay, persist it, and drop the engine so the next turn rebuilds with
    /// the new skill set.
    pub(super) async fn toggle_skill(&mut self, index: usize) -> Result<()> {
        let Some((name, enabled)) = (match &mut self.overlay {
            Overlay::Skills(state) => state.items.get_mut(index).map(|item| {
                item.enabled = !item.enabled;
                (item.name.clone(), item.enabled)
            }),
            _ => None,
        }) else {
            return Ok(());
        };
        self.session_store.set_skill_enabled(&name, enabled).await?;
        self.request_engine_rebuild();
        // A disabled skill drops out of the `/` menu; an enabled one returns.
        if let Overlay::Skills(state) = &self.overlay {
            self.skill_commands = enabled_skill_commands(&state.items);
        }
        Ok(())
    }

    /// Handle the `/skills` add-input. A `-p`/`--project` flag (leading or
    /// trailing) targets the repo's `.agents/skills` instead of
    /// `~/.config/aivo/skills`. A first token that isn't a bare skill name
    /// (a `github:owner/repo`, a github.com URL, or a local path) is INSTALLED
    /// from that source (the rest is an optional `<skill-name>` filter or `*`);
    /// otherwise `name [description…]` SCAFFOLDS a template. Either way the
    /// overlay reopens.
    pub(super) async fn submit_skill_add(&mut self, input: String) -> Result<()> {
        let (input, project) = split_project_flag(&input);
        let (first, rest) = match input.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (input.as_str(), ""),
        };
        // A flag-like token that survived `split_project_flag` is a typo or a
        // mid-line `-p`: reject it up front — before a download (a bad filter
        // used to surface only after the fetch) and before scaffolding a
        // folder literally named `--foo` (dashes are valid name chars).
        let stray_flag = if first.starts_with('-') {
            Some(first)
        } else {
            rest.split_whitespace()
                .next()
                .filter(|t| t.starts_with('-'))
        };
        if let Some(flag) = stray_flag {
            self.notice = Some((
                ERROR(),
                format!(
                    "Unknown option `{flag}` — only -p/--project is supported (at the start or end)"
                ),
            ));
            return Ok(());
        }
        // A scaffold name is `[A-Za-z0-9_-]`; anything else (a `/`, `:`, `.`, URL)
        // is an install source.
        if !first.is_empty() && !crate::agent::skills::is_valid_skill_name(first) {
            let only = (!rest.is_empty()).then(|| rest.to_string());
            return self
                .install_skill_from_source(first.to_string(), only, project)
                .await;
        }

        let (name, description) = match parse_skill_add_input(&input) {
            Ok(parsed) => parsed,
            Err(msg) => {
                self.notice = Some((ERROR(), msg));
                return Ok(());
            }
        };
        let root = match self.skills_dest_root(project) {
            Ok(root) => root,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to add skill: {e}")));
                return Ok(());
            }
        };
        match crate::agent::skills::scaffold_skill_at(&root, &name, &description) {
            Ok(path) => {
                // A freshly scaffolded skill starts enabled, clearing any stale
                // disabled flag left by a same-name skill removed earlier.
                self.session_store.set_skill_enabled(&name, true).await.ok();
                self.request_engine_rebuild();
                let mut notice = skill_add_success_notice(&name, &description, &path);
                if project {
                    notice.push_str(PROJECT_SKILL_NOTE);
                }
                self.notice = Some((MUTED(), notice));
            }
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to add skill: {e}")));
                return Ok(());
            }
        }
        self.open_skills_overlay().await
    }

    /// Resolve where `/skills add` writes: the user-global dir, or with
    /// `-p/--project` the repo's tool-neutral `.agents/skills`.
    fn skills_dest_root(&self, project: bool) -> std::result::Result<std::path::PathBuf, String> {
        if project {
            let cwd = if self.real_cwd.is_empty() {
                "."
            } else {
                self.real_cwd.as_str()
            };
            Ok(crate::agent::skills::project_skills_dir(
                std::path::Path::new(cwd),
            ))
        } else {
            Ok(crate::agent::skills::user_skills_dir())
        }
    }

    /// Install skill(s) from an online/local source into `~/.config/aivo/skills`
    /// (or the repo's `.agents/skills` when `project`), following the
    /// `skills/*/SKILL.md` convention. `only` is an optional skill-name filter
    /// (`*` = all). A multi-skill source with no filter lists the names so the
    /// user can re-run with one (or `*`).
    pub(super) async fn install_skill_from_source(
        &mut self,
        source: String,
        only: Option<String>,
        project: bool,
    ) -> Result<()> {
        if self.installing_skill.is_some() {
            self.notice = Some((WARNING(), "A skill install is already running".to_string()));
            return Ok(());
        }
        let dest_root = match self.skills_dest_root(project) {
            Ok(root) => root,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to install skill: {e}")));
                return Ok(());
            }
        };
        // Fetch on a background task; an unambiguous choice installs there and
        // arrives as `SkillInstalled`, a multi-skill source as `SkillInstallPick`.
        let progress = SkillInstallProgress::new(source.clone(), "Fetching");
        let bytes = progress.bytes.clone();
        let overlay_source = source.clone();
        self.installing_skill = Some(progress);
        self.notice = None;
        let tx = self.tx.clone();
        tokio::spawn(async move {
            use crate::agent::skills::InstallOrStage;
            let event = match crate::agent::skills::install_or_stage_into(
                &dest_root,
                &source,
                only.as_deref(),
                Some(bytes),
            )
            .await
            {
                Ok(InstallOrStage::Installed(report)) => RuntimeEvent::SkillInstalled {
                    source,
                    project,
                    result: Ok(report),
                },
                Ok(InstallOrStage::Pick(staged)) => RuntimeEvent::SkillInstallPick {
                    source,
                    project,
                    staged,
                },
                Err(e) => RuntimeEvent::SkillInstalled {
                    source,
                    project,
                    result: Err(e),
                },
            };
            tx.send(event).ok();
        });
        // Open the install modal in its loading state right away; the pick/
        // installed events replace it in place.
        if matches!(
            self.overlay,
            Overlay::None | Overlay::Skills(_) | Overlay::SkillInstall(_)
        ) {
            self.overlay = Overlay::SkillInstall(SkillInstallOverlay {
                source: overlay_source,
                project,
                ..Default::default()
            });
        }
        Ok(())
    }

    /// Apply a background install's outcome: enable the skills, drop the engine,
    /// and reopen the `/skills` overlay unless another overlay is open.
    pub(super) async fn apply_skill_installed(
        &mut self,
        source: String,
        project: bool,
        result: std::result::Result<crate::agent::skills::InstallReport, String>,
    ) -> Result<()> {
        self.installing_skill = None;
        match result {
            Ok(report) => {
                for name in &report.installed {
                    self.session_store.set_skill_enabled(name, true).await.ok();
                }
                if !report.installed.is_empty() || !report.updated.is_empty() {
                    self.request_engine_rebuild();
                }
                self.notice = Some(install_report_notice(&source, project, &report));
            }
            Err(e) => self.notice = Some((ERROR(), format!("Failed to install skill: {e}"))),
        }
        if matches!(
            self.overlay,
            Overlay::None | Overlay::Skills(_) | Overlay::SkillInstall(_)
        ) {
            self.open_skills_overlay().await?;
        }
        Ok(())
    }

    /// A fetched source held several skills: open the install picker over its
    /// staged tree. If an unrelated overlay took over meanwhile, don't barge in.
    pub(super) async fn apply_skill_install_pick(
        &mut self,
        source: String,
        project: bool,
        staged: crate::agent::skills::StagedInstall,
    ) -> Result<()> {
        self.installing_skill = None;
        if !matches!(
            self.overlay,
            Overlay::None | Overlay::Skills(_) | Overlay::SkillInstall(_)
        ) {
            self.notice = Some((
                WARNING(),
                format!(
                    "`{source}` has {} skills — run `/skills add {source}` again to pick",
                    staged.skills.len()
                ),
            ));
            return Ok(());
        }
        let dest_root = match self.skills_dest_root(project) {
            Ok(root) => root,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to install skill: {e}")));
                return Ok(());
            }
        };
        let installed = staged.already_installed_in(&dest_root);
        let items: Vec<InstallPickItem> = staged
            .skills
            .iter()
            .zip(&installed)
            .map(|(skill, &installed)| InstallPickItem {
                name: skill.name.clone(),
                description: skill.description.clone(),
                // Read now — the staged tree is gone once the pick resolves.
                body: skill.instructions().into_owned(),
                checked: false,
                installed,
            })
            .collect();
        self.staged_skill_install = Some((staged, project));
        self.notice = None;
        self.overlay = Overlay::SkillInstall(SkillInstallOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            source,
            project,
            items,
            selected: 0,
            query: String::new(),
            viewing: None,
            detail_scroll: 0,
        });
        Ok(())
    }

    /// The picker's Enter: copy the chosen skills out of the staged tree on a
    /// blocking task (it can be large) and report via `SkillInstalled`.
    pub(super) async fn install_staged_skills(&mut self, names: Vec<String>) -> Result<()> {
        let source = match &self.overlay {
            Overlay::SkillInstall(state) => state.source.clone(),
            _ => String::new(),
        };
        self.overlay = Overlay::None;
        let Some((staged, project)) = self.staged_skill_install.take() else {
            return self.open_skills_overlay().await;
        };
        let dest_root = match self.skills_dest_root(project) {
            Ok(root) => root,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to install skill: {e}")));
                return self.open_skills_overlay().await;
            }
        };
        self.installing_skill = Some(SkillInstallProgress::new(source.clone(), "Installing"));
        let tx = self.tx.clone();
        tokio::task::spawn_blocking(move || {
            // update_existing: a marked installed row is an explicit replace ask.
            let result = staged.install_into(&dest_root, &names, true);
            tx.send(RuntimeEvent::SkillInstalled {
                source,
                project,
                result,
            })
            .ok();
        });
        // Land back on /skills, where the spinner row shows until the copy ends.
        self.open_skills_overlay().await
    }

    /// Esc: discard the stage and fall back to `/skills` — except from the
    /// loading state, which closes to the composer the user came from.
    pub(super) async fn cancel_skill_install(&mut self) -> Result<()> {
        let loading = matches!(&self.overlay, Overlay::SkillInstall(s) if s.items.is_empty());
        self.staged_skill_install = None;
        self.overlay = Overlay::None;
        if loading {
            return Ok(());
        }
        self.open_skills_overlay().await
    }

    /// `/skills update [name]`: re-fetch from the recorded source and replace
    /// in place; bare form updates everything with provenance, across both
    /// install roots (the repo's `.agents/skills` and the user dir).
    pub(super) async fn update_skills_command(&mut self, name: Option<String>) -> Result<()> {
        if self.installing_skill.is_some() {
            self.notice = Some((WARNING(), "A skill install is already running".to_string()));
            return Ok(());
        }
        // Project root first, matching discovery precedence: a name shadowed by
        // the repo updates the copy the agent actually sees.
        let mut roots = Vec::new();
        if let Ok(root) = self.skills_dest_root(true) {
            roots.push(root);
        }
        if let Ok(root) = self.skills_dest_root(false) {
            roots.push(root);
        }
        let label = name
            .clone()
            .unwrap_or_else(|| "installed skills".to_string());
        let progress = SkillInstallProgress::new(label.clone(), "Updating");
        let bytes = progress.bytes.clone();
        self.installing_skill = Some(progress);
        self.notice = None;
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result =
                crate::agent::skills::update_installed_skills(&roots, name.as_deref(), Some(bytes))
                    .await;
            tx.send(RuntimeEvent::SkillInstalled {
                source: label,
                project: false,
                result,
            })
            .ok();
        });
        if matches!(self.overlay, Overlay::None | Overlay::Skills(_)) {
            self.open_skills_overlay().await?;
        }
        Ok(())
    }

    /// Delete the skill at `index` (the overlay's confirmed `d`); resolves its
    /// name/dir/scope from the row then defers to the shared remover.
    pub(super) async fn remove_skill(&mut self, index: usize) -> Result<()> {
        let Some((name, dir, scope)) = (match &self.overlay {
            Overlay::Skills(state) => state
                .items
                .get(index)
                .map(|i| (i.name.clone(), i.dir.clone(), i.scope)),
            _ => None,
        }) else {
            return Ok(());
        };
        self.remove_scoped_skill(&name, &dir, scope).await
    }

    /// Delete a skill by name (`/skills rm <name>`); resolves its dir+scope from
    /// discovery, erroring if there's no such skill.
    pub(super) async fn remove_skill_named(&mut self, name: &str) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let cwd_path = std::path::Path::new(&cwd);
        let Some(skill) = crate::agent::skills::discover_skills(cwd_path)
            .into_iter()
            .find(|s| s.name == name)
        else {
            self.notice = Some((MUTED(), format!("No skill named `{name}`")));
            return Ok(());
        };
        let scope = crate::agent::skills::skill_scope(&skill.dir, cwd_path);
        self.remove_scoped_skill(name, &skill.dir, scope).await
    }

    /// Shared removal: a `Project` skill (in the repo's `.agents`/`.aivo` skills)
    /// is left alone with a notice — that's the repo's to manage; a `User` skill
    /// has its folder deleted and the overlay refreshed so it disappears.
    async fn remove_scoped_skill(
        &mut self,
        name: &str,
        dir: &std::path::Path,
        scope: crate::agent::skills::SkillScope,
    ) -> Result<()> {
        if scope == crate::agent::skills::SkillScope::Project {
            self.notice = Some((
                WARNING(),
                format!(
                    "`{name}` is a project skill ({}) — delete that folder to remove it",
                    dir.display()
                ),
            ));
            return Ok(());
        }
        match crate::agent::skills::remove_skill_dir(dir) {
            Ok(()) => {
                // Clear any leftover disabled flag so a re-add isn't stuck off.
                self.session_store.set_skill_enabled(name, true).await.ok();
                self.request_engine_rebuild();
                self.notice = Some((MUTED(), format!("Removed skill `{name}`")));
            }
            Err(e) => self.notice = Some((ERROR(), format!("Failed to remove skill: {e}"))),
        }
        self.open_skills_overlay().await
    }

    /// The MCP servers NOT to connect because the user explicitly turned them off
    /// in `/mcp` (`disabled_mcp_servers`). This is the *base* opt-out set; project
    /// `.mcp.json` STDIO servers are additionally held back until the user grants
    /// consent (see [`Self::connect_mcp_with_consent`]) since they run local code.
    pub(super) async fn effective_disabled_mcp_servers(&self) -> std::collections::HashSet<String> {
        self.session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Connect MCP, gating a repo's project `.mcp.json` servers behind a one-time
    /// consent — stdio spawns local commands, `url` hands the conversation to a
    /// repo-picked endpoint. User-scope servers connect freely. Until the user
    /// approves, project servers are held back; a consent card shows the targets.
    pub(super) async fn connect_mcp_with_consent(
        &mut self,
        cwd: String,
        base_disabled: std::collections::HashSet<String>,
    ) {
        let gated = crate::agent::mcp::project_gated_servers(std::path::Path::new(&cwd));
        if gated.is_empty() {
            self.start_mcp_connect(cwd, base_disabled);
            return;
        }
        // Seed the session decision from the persistent per-repo allow-list once —
        // but only if the stored approval matches the CURRENT server set (digest),
        // so a changed `.mcp.json` re-prompts instead of silently reusing consent.
        if self.project_mcp_consent == ProjectMcpConsent::Unknown {
            let dir_key = canonical_dir_key(&cwd);
            let digest = project_mcp_digest(&gated);
            if self
                .session_store
                .get_project_mcp_approved(&dir_key, &digest)
                .await
            {
                self.project_mcp_consent = ProjectMcpConsent::Allowed;
            }
        }
        if self.project_mcp_consent == ProjectMcpConsent::Allowed {
            self.start_mcp_connect(cwd, base_disabled);
            return;
        }
        // Unknown or Denied → hold the project servers back from the connect
        // (user-scope servers still come up).
        let mut held = base_disabled.clone();
        held.extend(gated.iter().map(|(name, _)| name.clone()));
        // Unknown → surface the consent card (unless one is already up).
        if self.project_mcp_consent == ProjectMcpConsent::Unknown
            && self.cards.mcp_consent.is_none()
        {
            self.cards.mcp_consent = Some(McpConsentPrompt {
                servers: gated,
                cwd: cwd.clone(),
                base_disabled,
            });
        }
        self.start_mcp_connect(cwd, held);
    }

    /// Resolve the project-MCP consent card. `y` runs the servers once (this
    /// session), `a` also remembers the approval for this repo, `n`/`Esc` denies
    /// (they stay held back). On approval the cached client/engine are dropped so
    /// the reconnect includes the previously-held project servers.
    pub(super) async fn handle_mcp_consent_key(&mut self, key: KeyEvent) {
        let always = matches!(key.code, KeyCode::Char('a' | 'A'));
        let allow = always || matches!(key.code, KeyCode::Char('y' | 'Y'));
        let deny = matches!(key.code, KeyCode::Char('n' | 'N') | KeyCode::Esc);
        if !allow && !deny {
            return; // unrecognized key: leave the card up
        }
        let Some(prompt) = self.cards.mcp_consent.take() else {
            return;
        };
        if deny {
            self.project_mcp_consent = ProjectMcpConsent::Denied;
            self.show_toast("Project MCP servers not started");
            return;
        }
        self.project_mcp_consent = ProjectMcpConsent::Allowed;
        if always {
            let dir_key = canonical_dir_key(&prompt.cwd);
            let digest = project_mcp_digest(&prompt.servers);
            if let Err(e) = self
                .session_store
                .set_project_mcp_approved(&dir_key, &digest)
                .await
            {
                self.notice = Some((ERROR(), format!("Couldn't remember the approval: {e}")));
            }
        }
        // Drop the cached client/engine so the reconnect picks up the servers
        // held back on the first connect.
        self.reset_mcp_after_config_change();
        self.start_mcp_connect(prompt.cwd, prompt.base_disabled);
        self.refresh_mcp_overlay_status();
        self.show_toast(if always {
            "Project MCP servers approved for this repo"
        } else {
            "Project MCP servers started for this session"
        });
    }

    /// `/mcp`: list the configured MCP servers with their live connection status
    /// (a snapshot of the current client) and per-server enabled state. Kicks off
    /// a background connect when nothing is cached yet so a reopened overlay shows
    /// real status instead of a perpetual "not connected".
    pub(super) async fn open_mcp_overlay(&mut self) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let disabled = self.effective_disabled_mcp_servers().await;
        // Refresh the Ctrl+T tool opt-outs so row statuses count `· N off` right.
        self.disabled_mcp_tools = self
            .session_store
            .get_disabled_mcp_tools()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        // Kick the connect off first so the snapshot below reads "connecting…".
        // Project servers are gated (consent card) the same as on a turn.
        self.connect_mcp_with_consent(cwd.clone(), disabled.clone())
            .await;
        let mut items: Vec<McpServerRow> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|srv| {
                    let enabled = !disabled.contains(&srv.name);
                    let (status, health) = self.mcp_server_status(&srv.name, enabled);
                    McpServerRow {
                        name: srv.name,
                        status,
                        health,
                        enabled,
                        scope: srv.scope,
                        command: srv.command,
                        remote: srv.remote,
                    }
                })
                .collect();
        sort_mcp_rows(&mut items);
        // Refresh the welcome chip's MCP count — every add/remove reopens here.
        self.mcp_configured_count = items.iter().filter(|i| i.enabled).count();
        self.overlay = Overlay::Mcp(McpOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            items,
            selected: 0,
            query: String::new(),
            adding: None,
            pending_delete: None,
            viewing: None,
            detail_scroll: 0,
        });
        Ok(())
    }

    /// Parse the `/mcp` add-input (`command [args…]`, shell-quoted), DERIVE the
    /// server name from the command (the explicit name was redundant), write the
    /// server to the user `mcp.json` — or the repo `.mcp.json` with `-p` — and
    /// refresh the overlay so the agent picks it up. A parse problem stays in
    /// the overlay as a notice.
    pub(super) async fn submit_mcp_add(&mut self, input: String) -> Result<()> {
        // `-p`/`--project` at either edge targets the repo `.mcp.json` (same
        // flag shape as `/skills add`).
        let (input, project) = split_project_flag(&input);
        // A pasted `mcpServers` JSON block (Ctrl+V in the add field) — the form
        // every README hands you — is parsed directly; the name comes from the
        // JSON key (env and extra fields preserved).
        let trimmed = input.trim();
        if trimmed.starts_with('{') {
            return self.submit_mcp_add_json(input.clone(), project).await;
        }
        // A bare http(s) URL is a remote Streamable HTTP server — wrap it as a
        // `{url}` config (no JSON typing needed) and route through the same path,
        // so naming, dedup, and auto-authorize are shared.
        if let Some(json) = bare_url_to_config(trimmed) {
            return self.submit_mcp_add_json(json, project).await;
        }
        let (command, args) = match parse_mcp_add_input(&input) {
            Ok(parsed) => parsed,
            Err(msg) => {
                self.notice = Some((ERROR(), msg));
                return Ok(());
            }
        };
        // Derive a name from the command and de-duplicate against existing servers
        // (user + project), so two `filesystem` servers become `filesystem`/`-2`.
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let existing: std::collections::HashSet<String> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|s| s.name)
                .collect();
        let name = dedupe_name(
            crate::agent::mcp::derive_server_name(&command, &args),
            &existing,
        );
        let write = if project {
            crate::agent::mcp::add_project_server_value(
                std::path::Path::new(&cwd),
                &name,
                &serde_json::json!({"command": command, "args": args}),
            )
            .await
        } else {
            crate::agent::mcp::add_user_server(&name, &command, &args).await
        };
        if let Err(e) = write {
            self.notice = Some((ERROR(), format!("Failed to add MCP server: {e}")));
            return Ok(());
        }
        // A freshly added server starts enabled, even if a same-name one had been
        // disabled before.
        self.session_store
            .set_mcp_server_enabled(&name, true)
            .await
            .ok();
        if project {
            self.allow_self_added_project_server();
            self.notice = Some((
                MUTED(),
                format!("Added MCP server `{name}` → ./.mcp.json (project — commit it to share)"),
            ));
        } else {
            self.notice = Some((MUTED(), format!("Added MCP server `{name}`")));
        }
        self.reset_mcp_after_config_change();
        self.open_mcp_overlay().await
    }

    /// The user just typed a project server into `/mcp add -p` — that IS
    /// the consent, so grant the run-once session approval (like pressing `y`).
    /// Never persisted: the digest guard still re-prompts other sessions, and a
    /// later hand-edit of `.mcp.json` re-prompts as usual.
    fn allow_self_added_project_server(&mut self) {
        self.project_mcp_consent = ProjectMcpConsent::Allowed;
        self.cards.mcp_consent = None;
    }

    /// Add server(s) from a pasted `mcpServers` JSON block, preserving each
    /// entry's `env`/extra fields and taking the name from the JSON key (or
    /// deriving it for a bare `{command,…}`), de-duplicating against existing.
    async fn submit_mcp_add_json(&mut self, input: String, project: bool) -> Result<()> {
        let parsed = match crate::agent::mcp::parse_mcp_json(&input) {
            Ok(p) => p,
            Err(e) => {
                self.notice = Some((ERROR(), format!("Couldn't parse MCP config: {e}")));
                return Ok(());
            }
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        // Two or more servers in one paste → a pick overlay instead of adding
        // everything: new names arrive prechecked, an already-configured name
        // needs an explicit mark and then REPLACES that entry (re-pasting a
        // README block no longer mints `github-2` duplicates).
        if parsed.len() >= 2 {
            self.open_mcp_paste_picker(parsed, project, &cwd);
            return Ok(());
        }
        let mut existing: std::collections::HashSet<String> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|s| s.name)
                .collect();
        let mut added = Vec::new();
        let mut added_gated = false;
        for (name_opt, value) in parsed {
            let name = dedupe_name(
                name_opt.unwrap_or_else(|| crate::agent::mcp::derive_name_from_value(&value)),
                &existing,
            );
            let write = if project {
                crate::agent::mcp::add_project_server_value(
                    std::path::Path::new(&cwd),
                    &name,
                    &value,
                )
                .await
            } else {
                crate::agent::mcp::add_user_server_value(&name, &value).await
            };
            if let Err(e) = write {
                self.notice = Some((ERROR(), format!("Failed to add `{name}`: {e}")));
                self.reset_mcp_after_config_change();
                return self.open_mcp_overlay().await;
            }
            self.session_store
                .set_mcp_server_enabled(&name, true)
                .await
                .ok();
            // A url server may need OAuth — queue it to auto-authorize if its
            // connect comes back 401, so the user needn't press Ctrl+O.
            if let Some(url) = value.get("url").and_then(|u| u.as_str()) {
                self.pending_mcp_auth.insert(name.clone(), url.to_string());
            }
            added_gated |= value.get("command").is_some() || value.get("url").is_some();
            existing.insert(name.clone());
            added.push(name);
        }
        if project && added_gated {
            self.allow_self_added_project_server();
        }
        let label = if added.len() == 1 {
            "MCP server"
        } else {
            "MCP servers"
        };
        let suffix = if project {
            " → ./.mcp.json (project)"
        } else {
            ""
        };
        self.notice = Some((
            MUTED(),
            format!("Added {label}: {}{suffix}", added.join(", ")),
        ));
        self.reset_mcp_after_config_change();
        self.open_mcp_overlay().await
    }

    /// Stage a ≥2-server JSON paste as a pick overlay. Names come from the
    /// JSON keys (unique by construction — multi-entry pastes are maps), so
    /// there's nothing to dedupe: an existing name is a *replace* candidate.
    fn open_mcp_paste_picker(
        &mut self,
        parsed: Vec<(Option<String>, serde_json::Value)>,
        project: bool,
        cwd: &str,
    ) {
        let existing: std::collections::HashSet<String> =
            crate::agent::mcp::configured_servers(std::path::Path::new(cwd))
                .into_iter()
                .map(|s| s.name)
                .collect();
        let items: Vec<McpPasteRow> = parsed
            .into_iter()
            .map(|(name_opt, config)| {
                let name =
                    name_opt.unwrap_or_else(|| crate::agent::mcp::derive_name_from_value(&config));
                let exists = existing.contains(&name);
                McpPasteRow {
                    display: crate::agent::mcp_import::display_of(&config),
                    checked: !exists,
                    exists,
                    name,
                    config,
                }
            })
            .collect();
        let parent = match std::mem::replace(&mut self.overlay, Overlay::None) {
            Overlay::Mcp(state) => Some(Box::new(state)),
            other => {
                // Composer-initiated paste: nothing to restore on Esc.
                drop(other);
                None
            }
        };
        self.overlay = Overlay::McpPaste(McpPasteOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            parent,
            project,
            items,
            selected: 0,
            query: String::new(),
        });
    }

    /// Enter in the paste picker: write every checked row (an existing name is
    /// overwritten in place), then land back on `/mcp` with a summary notice.
    pub(super) async fn apply_mcp_paste(&mut self) -> Result<()> {
        let (rows, project) = match &self.overlay {
            Overlay::McpPaste(state) => {
                let rows: Vec<McpPasteRow> =
                    state.items.iter().filter(|i| i.checked).cloned().collect();
                (rows, state.project)
            }
            _ => return Ok(()),
        };
        if rows.is_empty() {
            self.notice = Some((
                WARNING(),
                "Nothing marked — Space marks a server to add".to_string(),
            ));
            return Ok(());
        }
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let mut added = Vec::new();
        let mut replaced = Vec::new();
        let mut added_gated = false;
        for row in rows {
            let write = if project {
                crate::agent::mcp::add_project_server_value(
                    std::path::Path::new(&cwd),
                    &row.name,
                    &row.config,
                )
                .await
            } else {
                crate::agent::mcp::add_user_server_value(&row.name, &row.config).await
            };
            if let Err(e) = write {
                self.notice = Some((ERROR(), format!("Failed to add `{}`: {e}", row.name)));
                self.reset_mcp_after_config_change();
                return self.open_mcp_overlay().await;
            }
            self.session_store
                .set_mcp_server_enabled(&row.name, true)
                .await
                .ok();
            // A url server may need OAuth — queue the auto-authorize.
            if let Some(url) = row.config.get("url").and_then(|u| u.as_str()) {
                self.pending_mcp_auth
                    .insert(row.name.clone(), url.to_string());
            }
            added_gated |= row.config.get("command").is_some() || row.config.get("url").is_some();
            if row.exists {
                replaced.push(row.name);
            } else {
                added.push(row.name);
            }
        }
        if project && added_gated {
            self.allow_self_added_project_server();
        }
        let mut parts = Vec::new();
        if !added.is_empty() {
            parts.push(format!("Added {}", added.join(", ")));
        }
        if !replaced.is_empty() {
            parts.push(format!("replaced {}", replaced.join(", ")));
        }
        let suffix = if project {
            " → ./.mcp.json (project)"
        } else {
            ""
        };
        self.notice = Some((MUTED(), format!("{}{suffix}", parts.join(" · "))));
        self.reset_mcp_after_config_change();
        self.open_mcp_overlay().await
    }

    /// `/mcp` from the composer: bare opens the overlay; `add <command> [args…]`
    /// (name derived) and `remove|rm <name>` manage servers without opening it.
    pub(super) async fn run_mcp_command(&mut self, arg: Option<String>) -> Result<()> {
        let Some(arg) = arg.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
            return self.open_mcp_overlay().await;
        };
        let (verb, rest) = match arg.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (arg.as_str(), ""),
        };
        match verb {
            "add" => self.submit_mcp_add(rest.to_string()).await,
            "remove" | "rm" if rest.is_empty() => {
                self.notice = Some((ERROR(), "Usage: /mcp rm <name>".to_string()));
                Ok(())
            }
            "remove" | "rm" => self.remove_mcp_server_named(rest).await,
            _ => {
                self.notice = Some((
                    ERROR(),
                    "Usage: /mcp [add <command> …] [rm <name>]".to_string(),
                ));
                Ok(())
            }
        }
    }

    /// Remove the server at `index` (the overlay's `d`); resolves the name+scope
    /// from the row then defers to the shared remover.
    pub(super) async fn remove_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, scope)) = (match &self.overlay {
            Overlay::Mcp(state) => state.items.get(index).map(|i| (i.name.clone(), i.scope)),
            _ => None,
        }) else {
            return Ok(());
        };
        self.remove_scoped_mcp_server(&name, scope).await
    }

    /// Remove a server by name (`/mcp rm <name>`); resolves its scope from config.
    pub(super) async fn remove_mcp_server_named(&mut self, name: &str) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let Some(scope) = crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
            .into_iter()
            .find(|s| s.name == name)
            .map(|s| s.scope)
        else {
            self.notice = Some((MUTED(), format!("No MCP server named `{name}`")));
            return Ok(());
        };
        self.remove_scoped_mcp_server(name, scope).await
    }

    /// Shared removal: a `Project` server (from a repo `.mcp.json`) is left alone
    /// with a notice — that file is the repo's to edit; a `User` server is removed
    /// from the user `mcp.json` and the overlay refreshed so its tools drop.
    async fn remove_scoped_mcp_server(
        &mut self,
        name: &str,
        scope: crate::agent::mcp::ServerScope,
    ) -> Result<()> {
        if scope == crate::agent::mcp::ServerScope::Project {
            self.notice = Some((
                WARNING(),
                format!("`{name}` is defined in .mcp.json — edit that file to remove it"),
            ));
            return Ok(());
        }
        match crate::agent::mcp::remove_user_server(name).await {
            Ok(true) => {
                // Clear any leftover disabled flag so a re-add isn't stuck off.
                self.session_store
                    .set_mcp_server_enabled(name, true)
                    .await
                    .ok();
                self.notice = Some((MUTED(), format!("Removed MCP server `{name}`")));
                self.reset_mcp_after_config_change();
            }
            Ok(false) => self.notice = Some((MUTED(), format!("`{name}` was not in mcp.json"))),
            Err(e) => self.notice = Some((ERROR(), format!("Failed to remove MCP server: {e}"))),
        }
        self.open_mcp_overlay().await
    }

    /// After the configured server set changes, invalidate any in-flight connect
    /// and drop the cached client + engine so the next turn (or overlay reopen)
    /// reconnects with the new set.
    pub(super) fn reset_mcp_after_config_change(&mut self) {
        self.mcp_connect_gen = self.mcp_connect_gen.wrapping_add(1);
        self.mcp_client = None;
        self.mcp_connecting = false;
        self.mcp_connect_progress.clear();
        self.request_engine_rebuild();
    }

    /// Status + health for one server, read from the current client snapshot.
    /// A disabled server (user- or project-scoped) reads "off".
    fn mcp_server_status(&self, name: &str, enabled: bool) -> (String, McpHealth) {
        if !enabled {
            return ("off".to_string(), McpHealth::Disabled);
        }
        if let Some(client) = &self.mcp_client {
            // A poisoned transport still carries its tool snapshot — surface the
            // crash instead of a healthy-looking tool count.
            if client.is_dead(name) {
                return ("failed: connection lost".to_string(), McpHealth::Failed);
            }
            if let Some(n) = client.tool_count(name) {
                let off = client
                    .tool_names(name)
                    .map(|tools| {
                        tools
                            .iter()
                            .filter(|t| {
                                self.disabled_mcp_tools
                                    .contains(&crate::agent::mcp::qualified_name(name, t))
                            })
                            .count()
                    })
                    .unwrap_or(0);
                let on = n.saturating_sub(off);
                let plural = if on == 1 { "" } else { "s" };
                let status = if off > 0 {
                    format!("{on} tool{plural} · {off} off")
                } else {
                    format!("{on} tool{plural}")
                };
                return (status, McpHealth::Connected);
            }
            // A 401 isn't a hard failure — the server just needs OAuth.
            if client.needs_auth(name) {
                return ("needs authorization".to_string(), McpHealth::NeedsAuth);
            }
            if let Some(err) = client.error_for(name) {
                return (format!("failed: {err}"), McpHealth::Failed);
            }
        }
        // Mid-connect: this server's handshake may have already resolved even
        // though the whole set (and `mcp_client`) hasn't — show its real status
        // instead of a blanket "connecting…".
        if let Some((status, health)) = self.mcp_connect_progress.get(name) {
            return (status.clone(), *health);
        }
        if self.mcp_connecting {
            return ("connecting…".to_string(), McpHealth::Idle);
        }
        ("not connected".to_string(), McpHealth::Idle)
    }

    /// Flip the enabled state of the server at `index` in the open `/mcp` overlay,
    /// persist it, and drop + re-establish the MCP connection for the new server
    /// set. Reconnecting live (rather than deferring to the next turn) means the
    /// overlay updates from "connecting…" to the real tool count / failure while
    /// it's still open, so the toggle has visible feedback.
    pub(super) async fn toggle_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, enabled)) = (match &mut self.overlay {
            Overlay::Mcp(state) => state.items.get_mut(index).map(|item| {
                item.enabled = !item.enabled;
                (item.name.clone(), item.enabled)
            }),
            _ => None,
        }) else {
            return Ok(());
        };
        // Both user- and project-scoped servers use the global opt-out list; a
        // project `.mcp.json` server is enabled by default (the user owns their
        // own repo) and toggling it off just adds it to the opt-outs.
        self.session_store
            .set_mcp_server_enabled(&name, enabled)
            .await?;
        // Reuse the live client so the servers that *aren't* being toggled keep
        // their connection and status — only the toggled one changes — instead of
        // tearing down and reconnecting the whole set.
        self.reconnect_mcp_preserving_for_overlay();
        // Toggle doesn't reopen the overlay, so refresh the welcome chip count here.
        if let Overlay::Mcp(state) = &self.overlay {
            self.mcp_configured_count = state.items.iter().filter(|i| i.enabled).count();
        }
        Ok(())
    }

    /// Open the Ctrl+T tool-toggle drill-in for the server at `index` in the
    /// open `/mcp` overlay. Needs the server connected (tools are unknown
    /// otherwise); keeps the `/mcp` state to restore on Esc.
    pub(super) fn open_mcp_tools(&mut self, index: usize) {
        let name = match &self.overlay {
            Overlay::Mcp(state) => match state.items.get(index) {
                Some(row) => row.name.clone(),
                None => return,
            },
            _ => return,
        };
        let details: Option<Vec<(String, String)>> = self
            .mcp_client
            .as_ref()
            .and_then(|c| c.tool_details(&name))
            .map(|v| {
                v.into_iter()
                    .map(|(t, d)| (t.to_string(), d.to_string()))
                    .collect()
            });
        let Some(details) = details else {
            self.notice = Some((
                WARNING(),
                format!("`{name}` isn't connected — its tools are unknown"),
            ));
            return;
        };
        if details.is_empty() {
            self.notice = Some((MUTED(), format!("`{name}` exposes no tools")));
            return;
        }
        let items: Vec<McpToolRow> = details
            .into_iter()
            .map(|(tool, desc)| McpToolRow {
                enabled: !self
                    .disabled_mcp_tools
                    .contains(&crate::agent::mcp::qualified_name(&name, &tool)),
                name: tool,
                description: desc,
            })
            .collect();
        let parent = match std::mem::replace(&mut self.overlay, Overlay::None) {
            Overlay::Mcp(state) => Box::new(state),
            other => {
                self.overlay = other;
                return;
            }
        };
        self.overlay = Overlay::McpTools(McpToolsOverlay {
            list_scroll: 0,
            scroll_selected: usize::MAX,
            server: name,
            parent,
            items,
            selected: 0,
            query: String::new(),
        });
    }

    /// Flip one tool in the Ctrl+T drill-in, persist it, and drop the engine so
    /// the next turn advertises the filtered set. No reconnect — the server
    /// connection is untouched; only what's advertised changes.
    pub(super) async fn toggle_mcp_tool(&mut self, index: usize) -> Result<()> {
        let Some((server, tool, enabled)) = (match &self.overlay {
            Overlay::McpTools(state) => state
                .items
                .get(index)
                .map(|item| (state.server.clone(), item.name.clone(), !item.enabled)),
            _ => None,
        }) else {
            return Ok(());
        };
        let qualified = crate::agent::mcp::qualified_name(&server, &tool);
        // Persist FIRST — a store failure must not leave the row, the cache,
        // and the engine disagreeing about the tool's state.
        self.session_store
            .set_mcp_tool_enabled(&qualified, enabled)
            .await?;
        if let Overlay::McpTools(state) = &mut self.overlay
            && let Some(item) = state.items.get_mut(index)
        {
            item.enabled = enabled;
        }
        if enabled {
            self.disabled_mcp_tools.remove(&qualified);
        } else {
            self.disabled_mcp_tools.insert(qualified);
        }
        // Specs are baked in at set_external_tools — rebuild on the next turn.
        self.request_engine_rebuild();
        Ok(())
    }

    /// Retry the failed servers in the open `/mcp` overlay (Ctrl+R). Reuses the
    /// preserve-live reconnect: connected servers are carried over verbatim, so
    /// only failed / crashed / needs-auth ones actually reconnect.
    pub(super) fn retry_mcp_failed(&mut self) {
        let any_failed = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .any(|i| matches!(i.health, McpHealth::Failed)),
            _ => return,
        };
        if !any_failed {
            self.notice = Some((MUTED(), "No failed MCP servers to retry".to_string()));
            return;
        }
        self.reconnect_mcp_preserving_for_overlay();
    }

    /// Kick a fresh background connect for the server set shown in the open `/mcp`
    /// overlay (derived from each row's enabled flag) and repaint every row to its
    /// interim status, so a toggle shows "connecting…"/"off" immediately and the
    /// real status lands when `McpConnected` resolves. A no-op when the overlay
    /// isn't the MCP one.
    pub(super) fn restart_mcp_connect_for_overlay(&mut self) {
        let disabled: std::collections::HashSet<String> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .filter(|i| !i.enabled)
                .map(|i| i.name.clone())
                .collect(),
            _ => return,
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        self.start_mcp_connect(cwd, disabled);
        self.refresh_mcp_overlay_status();
    }

    /// Recompute each `/mcp` row's status + health from the current client
    /// snapshot. Called when a background connect resolves so an open overlay
    /// updates live instead of stalling on a stale "connecting…" until reopened.
    pub(super) fn refresh_mcp_overlay_status(&mut self) {
        // Snapshot (name, enabled) first so the immutable status lookups don't
        // overlap the later mutable borrow of the overlay.
        let rows: Vec<(String, bool)> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .map(|i| (i.name.clone(), i.enabled))
                .collect(),
            _ => return,
        };
        let statuses: Vec<(String, McpHealth)> = rows
            .iter()
            .map(|(name, enabled)| self.mcp_server_status(name, *enabled))
            .collect();
        if let Overlay::Mcp(state) = &mut self.overlay {
            for (item, (status, health)) in state.items.iter_mut().zip(statuses) {
                item.status = status;
                item.health = health;
            }
        }
    }

    /// Start a background MCP connect, idempotently: a no-op when a client is
    /// already cached or a connect is in flight. The result returns as
    /// `RuntimeEvent::McpConnected`. Shared by the first agent turn and `/mcp`.
    pub(super) fn start_mcp_connect(
        &mut self,
        cwd: String,
        disabled: std::collections::HashSet<String>,
    ) {
        if self.mcp_client.is_some() || self.mcp_connecting {
            return;
        }
        self.mcp_connecting = true;
        self.mcp_connect_progress.clear();
        self.spawn_mcp_connect(cwd, disabled, self.mcp_connect_gen, None);
    }

    /// Reconnect for a changed server set in the open `/mcp` overlay (a toggle),
    /// reusing the live client so a server that's still enabled keeps its
    /// connection and its displayed status instead of every row flashing
    /// "connecting…". Only the changed server actually (re)connects. `mcp_client`
    /// is deliberately kept (not nulled) so status stays live during the connect;
    /// the engine is dropped so the next turn rebuilds with the new tool set.
    pub(super) fn reconnect_mcp_preserving_for_overlay(&mut self) {
        let disabled: std::collections::HashSet<String> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .filter(|i| !i.enabled)
                .map(|i| i.name.clone())
                .collect(),
            _ => return,
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        self.mcp_connect_gen = self.mcp_connect_gen.wrapping_add(1);
        self.mcp_connecting = true;
        self.mcp_connect_progress.clear();
        self.request_engine_rebuild();
        self.spawn_mcp_connect(cwd, disabled, self.mcp_connect_gen, self.mcp_client.clone());
        self.refresh_mcp_overlay_status();
    }

    /// Spawn the background connect task: report each server's status as its
    /// handshake resolves (`McpServerProgress`) and the assembled client at the
    /// end (`McpConnected`), both tagged with `generation` so a stale result is
    /// dropped. When `reuse` is given, still-enabled servers already live there
    /// are carried over instead of being reconnected.
    fn spawn_mcp_connect(
        &self,
        cwd: String,
        disabled: std::collections::HashSet<String>,
        generation: u64,
        reuse: Option<std::sync::Arc<crate::agent::mcp::McpClient>>,
    ) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let progress_tx = tx.clone();
            // Report each server's status as its handshake resolves so an open
            // `/mcp` overlay flips that row immediately, rather than every row
            // sitting on "connecting…" until the slowest server finishes.
            let report = move |name, status| {
                let (status, health) = mcp_status_from_connect(status);
                progress_tx
                    .send(RuntimeEvent::McpServerProgress {
                        name,
                        status,
                        health,
                        generation,
                    })
                    .ok();
            };
            let path = std::path::Path::new(&cwd);
            let client = std::sync::Arc::new(match &reuse {
                Some(prev) => {
                    prev.reconnect_enabled_with_progress(path, &disabled, report)
                        .await
                }
                None => {
                    crate::agent::mcp::McpClient::connect_enabled_with_progress(
                        path, &disabled, report,
                    )
                    .await
                }
            });
            tx.send(RuntimeEvent::McpConnected { client, generation })
                .ok();
        });
    }

    /// Start the interactive OAuth flow for the HTTP server at `index` in the open
    /// `/mcp` overlay. Runs on a background task (so the TUI keeps painting): it
    /// auto-opens the browser, emits the authorize URL as a notice, and reports
    /// the outcome via `RuntimeEvent::McpAuthorized`. A stdio server is rejected —
    /// OAuth only applies to `url` servers.
    pub(super) async fn authorize_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, target, remote)) = (match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .get(index)
                .map(|i| (i.name.clone(), i.command.clone(), i.remote)),
            _ => None,
        }) else {
            return Ok(());
        };
        if !remote {
            self.notice = Some((
                WARNING(),
                format!("`{name}` is a stdio server — OAuth applies only to HTTP (url) servers"),
            ));
            return Ok(());
        }
        self.start_mcp_authorize(name, target);
        Ok(())
    }

    /// Spawn the interactive OAuth flow for an HTTP MCP server `(name, url)` on a
    /// background task (the TUI keeps painting): it auto-opens the browser, emits
    /// the authorize URL as a notice, and reports the outcome via
    /// `RuntimeEvent::McpAuthorized`. Shared by the explicit Ctrl+O action and the
    /// auto-authorize that follows adding a server that turns out to need OAuth.
    pub(super) fn start_mcp_authorize(&mut self, name: String, url: String) {
        // A config url may carry `${VAR}` refs — resolve them like connect does.
        let url = match crate::agent::mcp::expand_env_refs(&url) {
            Ok(u) => u,
            Err(e) => {
                self.notice = Some((WARNING(), format!("`{name}`: {e}")));
                return;
            }
        };
        self.notice = Some((
            MUTED(),
            format!("Authorizing `{name}` — a browser window should open…"),
        ));
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let url_tx = tx.clone();
            let result = crate::services::mcp_oauth::authorize(&url, None, |authorize_url| {
                url_tx
                    .send(RuntimeEvent::McpAuthorizeUrl {
                        url: authorize_url.to_string(),
                    })
                    .ok();
            })
            .await;
            tx.send(RuntimeEvent::McpAuthorized {
                name,
                result: result.map_err(|e| e.to_string()),
            })
            .ok();
        });
    }

    /// Clear the stored OAuth credential for the server at `index` and reconnect,
    /// so an authorized HTTP server drops back to "needs authorization".
    pub(super) async fn sign_out_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some(name) = (match &self.overlay {
            Overlay::Mcp(state) => state.items.get(index).map(|i| i.name.clone()),
            _ => None,
        }) else {
            return Ok(());
        };
        match crate::services::mcp_token_store::remove(&name).await {
            Ok(true) => {
                self.notice = Some((MUTED(), format!("Signed out of `{name}`")));
                self.reset_mcp_after_config_change();
                self.restart_mcp_connect_for_overlay();
            }
            Ok(false) => {
                self.notice = Some((MUTED(), format!("`{name}` had no stored credentials")));
            }
            Err(e) => {
                self.notice = Some((ERROR(), format!("Failed to sign out of `{name}`: {e}")));
            }
        }
        Ok(())
    }

    pub(super) async fn activate_picker_selection(
        &mut self,
        filtered_index: usize,
    ) -> Result<bool> {
        let (kind, value) = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((original_index, _)) = picker.filtered_items().get(filtered_index).copied()
            else {
                return Ok(false);
            };
            (
                picker.kind.clone(),
                picker.items[original_index].value.clone(),
            )
        };

        self.overlay = Overlay::None;

        match (kind, value) {
            (PickerKind::Model { target, .. }, PickerValue::Model(model)) => match target {
                ModelSelectionTarget::CurrentChat => {
                    // Mid-turn is fine: the running turn keeps its model (same as
                    // the agent's `switch_model` tool); the new one applies next turn.
                    self.apply_model(model.clone()).await?;
                    let msg = if self.sending {
                        format!("Now using {model} — applies from the next turn")
                    } else {
                        format!("Now using {model}")
                    };
                    self.notice = Some((MUTED(), msg));
                }
                ModelSelectionTarget::KeySwitch { key, .. } => {
                    self.complete_key_switch(key, model).await?
                }
                ModelSelectionTarget::VisionDescriber(key) => {
                    self.apply_vision_describer(key, model).await;
                }
                ModelSelectionTarget::ImageGenerator(key) => {
                    self.apply_image_generator(key, model).await;
                }
            },
            (
                PickerKind::Key {
                    target: KeySelectionTarget::Switch,
                },
                PickerValue::Key(key),
            ) => {
                self.begin_key_switch(key).await?;
            }
            (
                PickerKind::Key {
                    target: KeySelectionTarget::VisionDescriber,
                },
                PickerValue::Key(key),
            ) => {
                self.open_describer_model_picker(key);
            }
            (
                PickerKind::Key {
                    target: KeySelectionTarget::ImageGenerator,
                },
                PickerValue::Key(key),
            ) => {
                self.open_image_generator_model_picker(key);
            }
            (PickerKind::Session, PickerValue::Session(session)) => {
                self.begin_resume_load(session);
            }
            (
                PickerKind::Rewind,
                PickerValue::RewindTurn {
                    history_index,
                    ordinal,
                    keep_engine,
                },
            ) => {
                self.rewind_to_turn(history_index, ordinal, keep_engine)
                    .await?;
            }
            (PickerKind::PlanResume, PickerValue::PlanResume(carry)) => {
                self.resume_plan_from_session(carry).await;
            }
            _ => {}
        }

        Ok(false)
    }

    pub(super) async fn delete_picker_selection(&mut self, filtered_index: usize) -> Result<bool> {
        let session = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((_, item)) = picker.filtered_items().get(filtered_index).copied() else {
                return Ok(false);
            };
            match &item.value {
                PickerValue::Session(session) => session.clone(),
                _ => return Ok(false),
            }
        };

        let removed = self
            .session_store
            .delete_chat_session(&session.session_id)
            .await?;
        if !removed {
            self.notice = Some((ERROR(), "Saved session no longer exists".to_string()));
            return Ok(false);
        }

        if let Overlay::Picker(picker) = &mut self.overlay {
            picker.clear_pending_delete();
            picker.items.retain(|item| {
                !matches!(
                    &item.value,
                    PickerValue::Session(existing)
                        if existing.key_id == session.key_id && existing.session_id == session.session_id
                )
            });

            let filtered_len = picker.filtered_items().len();
            if filtered_len == 0 {
                self.overlay = Overlay::None;
                self.notice = Some((MUTED(), "Saved session deleted".to_string()));
                return Ok(false);
            }

            picker.selected = picker.selected.min(filtered_len.saturating_sub(1));
        }

        self.notice = Some((MUTED(), "Saved session deleted".to_string()));
        Ok(false)
    }

    pub(super) async fn resolve_key_exact(&self, query: &str) -> Result<Option<ApiKey>> {
        let keys = self.session_store.get_keys().await?;

        if let Some(key) = keys.iter().find(|key| key.id == query).cloned() {
            return Ok(Some(key));
        }

        let name_matches = keys
            .into_iter()
            .filter(|key| key.name == query)
            .collect::<Vec<_>>();

        if name_matches.len() == 1 {
            Ok(name_matches.into_iter().next())
        } else {
            Ok(None)
        }
    }

    pub(super) fn current_model_picker_key(&self) -> Option<ApiKey> {
        let Overlay::Picker(picker) = &self.overlay else {
            return None;
        };
        match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::CurrentChat,
                ..
            } => Some(self.key.clone()),
            PickerKind::Model {
                target: ModelSelectionTarget::KeySwitch { key, .. },
                ..
            }
            | PickerKind::Model {
                target: ModelSelectionTarget::VisionDescriber(key),
                ..
            }
            | PickerKind::Model {
                target: ModelSelectionTarget::ImageGenerator(key),
                ..
            } => Some(key.clone()),
            _ => None,
        }
    }

    pub(super) async fn persist_history(&self) -> Result<()> {
        // A freshly-opened foreign import lives in memory only — don't persist an
        // aivo copy just for viewing it. Once a real turn grows the history past
        // the resume baseline, this gate opens and it persists as a fork.
        if let Some(baseline) = self.pristine_import_len
            && self.history.len() <= baseline
        {
            return Ok(());
        }
        let stored = to_stored_messages(&self.history);
        let title = session_title_from_messages(&self.history, &self.raw_model);
        let preview = session_preview_text_from_messages(&self.history, &self.raw_model);
        // `session_tokens` is the running cumulative for this session (each turn
        // folds in the provider-measured split); the index entry stores the total
        // so `aivo stats --since` can attribute windowed chat usage per model.
        self.session_store
            .save_code_session_with_id(
                &self.key.id,
                &self.key.base_url,
                self.persist_cwd(),
                &self.session_id,
                &self.raw_model,
                self.billed_model.as_deref(),
                &stored,
                &title,
                &preview,
                self.session_tokens,
                self.session_cost_usd,
            )
            .await?;
        // Durable resume: also persist the agent engine's exact conversation
        // (assistant tool_calls + tool results with ids) so a resume restores tool
        // history verbatim. Best-effort and non-blocking: `try_lock` skips the
        // export when a turn is mid-flight (the text save above preserved the prior
        // blob), and a non-agent chat has no engine. Done after the text save so
        // the session file exists for `save_agent_messages` to update.
        if let Some(session) = &self.agent_engine
            && let Ok(engine) = session.engine.try_lock()
        {
            let conversation = engine.export_conversation();
            drop(engine);
            // Persist even when empty: a `/rewind` to the first turn empties the
            // engine, and storing that (which clears the blob — see
            // `save_agent_messages`) keeps resume from restoring the stale
            // pre-rewind transcript. `try_lock` already skips the mid-turn case.
            let _ = self
                .session_store
                .save_agent_messages(&self.session_id, &conversation)
                .await;
        }
        // After the text save (the setter no-ops without a session file);
        // write-once, so every later persist is a no-op.
        if let Some(fidelity) = &self.import_fidelity {
            let _ = self
                .session_store
                .set_import_fidelity(&self.session_id, fidelity)
                .await;
        }
        if !self.vision_descriptions.is_empty() {
            let _ = self
                .session_store
                .set_image_descriptions(&self.session_id, &self.vision_descriptions)
                .await;
        }
        self.persist_plan_state().await;
        Ok(())
    }

    /// Push the live plan-mode state into the session file so a resume picks the
    /// plan back up. Turn ends reach this via `persist_history`; `/plan` enter/leave
    /// call it directly for idle-time changes. Best-effort; no-ops before the
    /// session first persists.
    pub(super) fn current_plan_state(&self) -> Option<crate::services::session_store::PlanState> {
        let steps = self.unfinished_plan_steps();
        (self.plan_mode || self.pending_plan.is_some() || steps.is_some()).then(|| {
            crate::services::session_store::PlanState {
                mode: self.plan_mode,
                draft: self.pending_plan.clone(),
                steps,
            }
        })
    }

    pub(super) async fn persist_plan_state(&self) {
        let plan = self.current_plan_state();
        // Nothing to write or clear: skip the round-trip (`set_plan_state` loads
        // the whole file before its own no-op check, and this runs every turn end).
        if plan.is_none() && !self.plan_state_written.get() {
            return;
        }
        // Plan finished or discarded: retire the managed `/plan save` file too
        // (explicit-path saves are the user's own).
        if plan.is_none() {
            remove_managed_plan_files(self.session_store.config_dir(), &self.session_id, None);
        }
        let _ = self
            .session_store
            .set_plan_state(&self.session_id, plan.as_ref())
            .await;
        self.plan_state_written.set(plan.is_some());
    }

    /// The latest `update_plan` checklist from history, when steps are still open.
    /// `None` for no checklist or an all-done one — a finished plan isn't resumable.
    pub(super) fn unfinished_plan_steps(&self) -> Option<serde_json::Value> {
        let content = &self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "plan")?
            .content;
        let value: serde_json::Value = serde_json::from_str(content).ok()?;
        let open = value
            .as_array()?
            .iter()
            .any(|item| plan_status(item) != "completed");
        open.then_some(value)
    }

    pub(super) fn begin_resume_load(&mut self, preview: SessionPreview) {
        self.discard_resume_state();
        // The share is pinned to the current session; resume swaps it out.
        self.stop_live_share();
        self.overlay = Overlay::None;
        if self.sending {
            self.cancel_inflight_request(CancelKind::Discard);
        }

        self.resume_restore_state = Some(ResumeRestoreState::capture(self));
        self.clear_for_resume_loading();
        // The new session id will come from storage; drop any live cursor ACP
        // session since cursor doesn't know about the resumed session.
        self.cursor_acp_session = None;
        self.resume_request_id = self.resume_request_id.wrapping_add(1);
        let request_id = self.resume_request_id;
        self.loading_resume = Some(LoadingResume {
            request_id,
            preview: preview.clone(),
            continue_plan: false,
        });

        let session_store = self.session_store.clone();
        let tx = self.tx.clone();
        // A foreign (import) row is reconstructed in memory using the live
        // key/model; it persists as a fork only once a real turn is taken.
        let key_id = self.key.id.clone();
        let model = self.raw_model.clone();
        let task = tokio::spawn(async move {
            let result =
                load_or_import_resume_session(&session_store, &preview, &key_id, &model).await;
            let _ = tx.send(RuntimeEvent::ResumeLoaded { request_id, result });
        });
        self.resume_task = Some(task);
    }

    pub(super) async fn apply_loaded_session(&mut self, session: LoadedSession) -> Result<()> {
        let mut key_fallback_notice = None;
        if !self.key_explicit && self.key.id != session.key_id {
            match self.session_store.get_key_by_id(&session.key_id).await? {
                Some(key) => {
                    self.key = key;
                    self.copilot_tm = copilot_token_manager_for_key(&self.key);
                }
                // Saved key was removed — keep the live key so the conversation
                // still opens; the next save re-stamps it.
                None => {
                    key_fallback_notice = Some(
                        "This session's saved key was removed — continuing with the current key"
                            .to_string(),
                    );
                }
            }
        }

        self.overlay = Overlay::None;
        // Drop any live agent engine/serve/permission so the resumed
        // conversation re-seeds its context from `session.messages` on the next
        // turn. Reusing a same-key/model engine would continue the PREVIOUS
        // session's thread (`/new` and key/model switches reset it the same way).
        self.agent_engine = None;
        self.cards.clear_agent_cards();
        // Plan/goal modes belong to the OLD conversation — a leaked plan card
        // indexes the replaced history and `/plan go` would run the old plan.
        self.plan_mode = false;
        self.plan_exit_pending = false;
        self.pending_plan = None;
        self.plan_card_idx = None;
        self.goal_mode = None;
        self.stop_agent_serve();
        self.session_id = session.session_id;
        // Re-root NEW background-job logs under the resumed session's artifacts dir.
        self.jobs.set_logs_root(
            self.session_store
                .session_artifacts_dir(&self.session_id)
                .join("jobs"),
        );
        // Re-seed the running totals from the stored entry so further turns
        // accumulate on top of them (the index save overwrites with the cumulative).
        let (tokens, stored_billed, stored_cost) = self
            .session_store
            .chat_session_billing(&self.session_id)
            .await;
        self.session_tokens = tokens;
        self.history = session.messages;
        // Resumed rows never map to live checkpoints (the store is session-scoped).
        self.agent_turn_indices.clear();
        self.acp_checkpoints.clear();
        self.acp_checkpoint_store = None;
        self.expanded_thinking.clear();
        self.expanded_output.clear();
        self.expanded_step_folds.clear();
        self.local_outputs.clear();
        self.reasoning_durations.clear();
        self.turn_durations.clear();
        self.turn_notes.clear();
        // Restore the exact agent transcript (tool calls + results with ids) into
        // the next engine build instead of the lossy text seed. `None` for
        // non-agent or pre-feature sessions → falls back to the text seed.
        self.pending_agent_messages = session.engine_messages;
        self.vision_descriptions = session.image_descriptions.unwrap_or_default();
        // A freshly-opened foreign import lives in memory only until a real turn
        // grows the transcript past this baseline (then it persists as a fork).
        self.pristine_import_len = session.pristine_import.then_some(self.history.len());
        self.import_fidelity = session.import_fidelity;
        // An explicit launch model must win over the model stored in the
        // resumed session. Bare `--resume` keeps the session's original model.
        let resume_model = if self.model_explicit {
            self.raw_model.clone()
        } else {
            session.raw_model.clone()
        };
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_submit = None;
        self.format = seeded_chat_format(&self.key, &resume_model);
        self.last_usage = None;
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.raw_model = resume_model.clone();
        self.model = CodeCommand::transform_model_for_provider(&self.key.base_url, &resume_model);
        self.billed_model = stored_billed;
        // Stored spend survives resume verbatim (it may include provider-reported
        // figures); legacy entries without one fall back to a list-price estimate.
        self.session_cost_usd = if stored_cost > 0.0 {
            stored_cost
        } else {
            self.session_pricing()
                .and_then(|p| p.cost_usd(&self.session_tokens))
                .unwrap_or(0.0)
        };
        self.refresh_context_window().await;
        // After model/window are set, so the preview mirrors the resumed session.
        self.context_tokens = self.estimated_context_used().await;
        self.context_is_estimate = true;
        // An unfinished plan saved with the session comes back: draft re-armed (its
        // card re-framed against the RESTORED history), plan mode restored — the
        // next turn's mode sync puts the rebuilt engine into read-only planning.
        // Seed the persist guard from the file's state so a later persist can clear
        // a snapshot the restore skipped.
        self.plan_state_written.set(session.plan_state.is_some());
        let restored_plan = match session.plan_state.filter(|_| self.agent_capable()) {
            Some(plan) => {
                self.plan_mode = plan.mode;
                self.pending_plan = plan.draft.filter(|d| !d.trim().is_empty());
                self.plan_card_idx = self.pending_plan.as_deref().and_then(|p| {
                    self.history
                        .iter()
                        .rposition(|m| m.role == "assistant" && m.content == p)
                });
                self.plan_mode || self.pending_plan.is_some()
            }
            None => false,
        };
        if session.source_newer {
            let label = crate::services::session_import::import_source_label(&self.session_id)
                .unwrap_or("source");
            self.notice = Some((
                MUTED(),
                format!(
                    "This fork is behind its {label} session — newer messages there were never imported here"
                ),
            ));
        } else if let Some(msg) = key_fallback_notice {
            self.notice = Some((MUTED(), msg));
        } else if session.pristine_import
            && let Some(fidelity) = &self.import_fidelity
        {
            // Announce fidelity on first open only; a saved fork re-opens silently.
            let label = crate::services::session_import::import_source_label(&self.session_id)
                .unwrap_or("import");
            self.notice = Some((MUTED(), fidelity.summary(label)));
        } else if restored_plan {
            self.notice = Some((
                MUTED(),
                if self.pending_plan.is_some() {
                    "Resumed with an unfinished plan — approve on the card or /plan go; /plan exit to discard"
                } else {
                    "Resumed in plan mode — read-only until you approve a plan (/plan exit to leave)"
                }
                .to_string(),
            ));
        } else if let Some(steps) = self.unfinished_plan_steps() {
            // The checklist and engine transcript both came back, so plain
            // "continue" resumes execution in place.
            self.notice = Some((
                MUTED(),
                format!(
                    "Resumed with an in-progress plan ({}) — say continue to keep executing",
                    plan_steps_progress(&steps)
                ),
            ));
        }
        // Session-local: no persist, so viewing an old chat can't reset the key's
        // default model. Only explicit `/model` and `/key` persist.
        Ok(())
    }

    async fn persist_model_selection(&self, raw_model: &str) -> Result<()> {
        self.session_store
            .set_code_model(&self.key.id, raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "code", Some(raw_model))
            .await?;
        self.update_last_selection(raw_model).await;
        Ok(())
    }

    /// Keep the global "selected key & model" in sync on explicit `/key` / `/model`
    /// so `aivo run`/`start`/`info` recall what chat is using. Resume is excluded
    /// (session-local — see `apply_loaded_session`).
    ///
    /// Preserves the existing launchable tool (so `aivo run` with no tool still
    /// recalls the last *launchable* tool, not "code"), skips the ephemeral HF
    /// synthetic key, and writes the *stored* key's canonical `base_url`: the
    /// live `self.key` may carry a sentinel resolved to a real URL (ollama,
    /// aivo-starter), and a resolved URL would fail `get_last_selection`'s
    /// validity check against the saved record and be pruned as stale.
    /// Best-effort — a failed write never aborts the switch.
    async fn update_last_selection(&self, raw_model: &str) {
        if self.key.id == crate::services::huggingface::HF_LOCAL_KEY_ID {
            return;
        }
        let Ok(Some(stored_key)) = self.session_store.get_key_by_id(&self.key.id).await else {
            return;
        };
        let existing_tool = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|sel| sel.tool);
        let _ = self
            .session_store
            .set_last_selection(
                &stored_key,
                existing_tool.as_deref().unwrap_or("code"),
                Some(raw_model),
            )
            .await;
    }

    pub(super) fn scroll_up(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.effective_max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(step.max(1));
    }

    pub(super) fn scroll_down(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.effective_max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + step.max(1)).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_up_lines(&mut self, lines: usize) {
        let max_scroll = self.effective_max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(lines);
    }

    pub(super) fn scroll_down_lines(&mut self, lines: usize) {
        let max_scroll = self.effective_max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + lines).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_to_top(&mut self) {
        self.transcript_scroll = 0;
        self.follow_output = false;
    }

    pub(super) fn scroll_to_bottom(&mut self) {
        self.transcript_scroll = self.effective_max_scroll();
        self.follow_output = true;
    }

    /// Max scroll for the hot wheel/key handlers: reuse the value the last
    /// render computed when it's available (cheap — no rebuild), else recompute.
    /// The render re-clamps `transcript_scroll` every pass, so a one-frame-stale
    /// value only ever causes a transient that the next frame corrects.
    pub(super) fn effective_max_scroll(&self) -> usize {
        self.last_max_scroll.unwrap_or_else(|| self.max_scroll())
    }

    pub(super) fn max_scroll(&self) -> usize {
        let transcript = self.build_transcript();
        // Word-wrap to match the render's row count (char-wrap under-counts).
        let wrapped = wrap_transcript(
            &transcript.lines,
            &transcript.bar_colors,
            self.transcript_width,
        );
        wrapped
            .rows
            .len()
            .saturating_sub(usize::from(self.transcript_view_height))
    }

    pub(super) fn selected_transcript_text(&self) -> Option<String> {
        let selection = self.transcript_selection?;
        let hitbox = self.transcript_hitbox.as_ref()?;
        // Materialize only the selected row range; copy never runs per frame.
        let (start, end) = normalized_selection(selection);
        if start.row >= hitbox.rows_len() {
            return None;
        }
        let rows: Vec<String> = hitbox
            .rows()
            .skip(start.row)
            .take(end.row.saturating_sub(start.row) + 1)
            .map(str::to_string)
            .collect();
        let rebased = TranscriptSelection {
            anchor: TranscriptPoint {
                row: 0,
                column: start.column,
            },
            focus: TranscriptPoint {
                row: end.row - start.row,
                column: end.column,
            },
        };
        selected_text_from_rows(&rows, rebased)
    }

    /// Text under the full-screen selection, from the captured [`ScreenSurface`].
    pub(super) fn selected_screen_text(&self) -> Option<String> {
        let selection = self.screen_selection?;
        let rows = &self.screen_surface.as_ref()?.rows;
        selected_text_from_rows(rows, selection)
    }

    /// Text under whichever selection is live — transcript or screen.
    pub(super) fn selected_any_text(&self) -> Option<String> {
        self.selected_transcript_text()
            .or_else(|| self.selected_screen_text())
    }
}

// The add-line parsing and name dedup moved to `crate::agent::mcp` so
// `aivo code mcp add` shares them; re-exported for the call sites and tests here.
pub(super) use crate::agent::mcp::{bare_url_to_config, dedupe_name, parse_mcp_add_input};

pub(super) use crate::agent::mcp::canonical_dir_key;

/// Appended to a `-p/--project` add/install notice: repo-local skills are
/// advertised to the model inside `<untrusted>` (the dir changes under `git
/// pull`), which would surprise a user who just installed the skill themselves.
/// Inline (` · `), not `\n` — the notice renders as one line and scrubs control
/// chars, so a newline would mash the two sentences together.
pub(super) const PROJECT_SKILL_NOTE: &str =
    " · project skill — commit it to share; the agent advertises repo skills as untrusted";

/// Split a leading or trailing `-p`/`--project` off a `/skills add` line. Only
/// the edges are checked so a scaffold description containing a literal `-p`
/// mid-sentence survives.
pub(super) fn split_project_flag(input: &str) -> (String, bool) {
    let trimmed = input.trim();
    for flag in ["--project", "-p"] {
        if let Some(rest) = trimmed.strip_prefix(flag)
            && (rest.is_empty() || rest.starts_with(char::is_whitespace))
        {
            return (rest.trim().to_string(), true);
        }
        if let Some(rest) = trimmed.strip_suffix(flag)
            && (rest.is_empty() || rest.ends_with(char::is_whitespace))
        {
            return (rest.trim().to_string(), true);
        }
    }
    (trimmed.to_string(), false)
}

/// Parse a `/skills add` line into `(name, description)`: the first whitespace-
/// delimited token is the (folder-safe) name, the rest is a free-text one-line
/// description (empty is fine — a placeholder is templated in). `Err` is a
/// user-facing message.
pub(super) fn parse_skill_add_input(input: &str) -> std::result::Result<(String, String), String> {
    let input = input.trim();
    let (name, description) = match input.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (input, ""),
    };
    if !crate::agent::skills::is_valid_skill_name(name) {
        return Err(
            "Usage: <name> [description] — name is letters, digits, '-' or '_'".to_string(),
        );
    }
    Ok((name.to_string(), description.to_string()))
}

pub(super) fn skill_add_success_notice(
    name: &str,
    description: &str,
    path: &std::path::Path,
) -> String {
    let used_placeholder = description.trim().is_empty();
    let description = if used_placeholder {
        crate::agent::skills::PLACEHOLDER_DESCRIPTION
    } else {
        description.trim()
    };
    let advert = crate::agent::skills::advert_description(description);
    let warnings = crate::agent::skills::description_advert_warnings(description, used_placeholder);

    let mut notice = format!(
        "Created skill `{name}` — edit {}\nAdvert: {advert}",
        path.display()
    );
    for warning in warnings {
        notice.push_str("\nWarning: ");
        notice.push_str(&warning);
    }
    notice
}

/// One-line notice for a finished install; warning-hued when nothing changed.
/// A `project` install appends where it landed and the untrusted caveat.
pub(super) fn install_report_notice(
    source: &str,
    project: bool,
    report: &crate::agent::skills::InstallReport,
) -> (ratatui::style::Color, String) {
    let installed = &report.installed;
    let updated = &report.updated;
    let skipped = &report.skipped_existing;
    if installed.is_empty() && updated.is_empty() && skipped.is_empty() {
        return (WARNING(), format!("No skills found in `{source}`"));
    }
    if installed.is_empty() && updated.is_empty() {
        return (
            WARNING(),
            format!("Already installed: {}", skipped.join(", ")),
        );
    }
    let label = |names: &[String]| if names.len() == 1 { "skill" } else { "skills" };
    let mut parts: Vec<String> = Vec::new();
    if !installed.is_empty() {
        parts.push(format!(
            "Installed {}: {}",
            label(installed),
            installed.join(", ")
        ));
    }
    if !updated.is_empty() {
        parts.push(format!(
            "Updated {}: {}",
            label(updated),
            updated.join(", ")
        ));
    }
    let mut msg = parts.join(" · ");
    if project {
        msg.push_str(" → ./.agents/skills");
    }
    if !skipped.is_empty() {
        msg.push_str(&format!(" (already installed: {})", skipped.join(", ")));
    }
    if project {
        msg.push_str(PROJECT_SKILL_NOTE);
    }
    (MUTED(), msg)
}

/// Map a connect-time per-server outcome to the overlay's `(status, health)`
/// display pair — the same vocabulary `mcp_server_status` uses once the full
/// client lands, so an incrementally-updated row reads identically to a finished
/// one.
pub(super) fn mcp_status_from_connect(
    status: crate::agent::mcp::ServerConnectStatus,
) -> (String, McpHealth) {
    use crate::agent::mcp::ServerConnectStatus;
    match status {
        ServerConnectStatus::Connected { tools } => {
            let plural = if tools == 1 { "" } else { "s" };
            (format!("{tools} tool{plural}"), McpHealth::Connected)
        }
        ServerConnectStatus::NeedsAuth => ("needs authorization".to_string(), McpHealth::NeedsAuth),
        ServerConnectStatus::Failed(msg) => (format!("failed: {msg}"), McpHealth::Failed),
    }
}

/// Order `/mcp` rows problems-first: failed servers float to the top (act on
/// them), then connected/connecting, with disabled sunk to the bottom;
/// alphabetical within each group. Applied once at open (not on the live status
/// refresh) so rows don't jump under the cursor mid-session.
pub(super) fn sort_mcp_rows(rows: &mut [McpServerRow]) {
    rows.sort_by(|a, b| {
        mcp_health_rank(a.health)
            .cmp(&mcp_health_rank(b.health))
            .then_with(|| a.name.cmp(&b.name))
    });
}

fn mcp_health_rank(health: McpHealth) -> u8 {
    match health {
        McpHealth::Failed => 0,
        // Actionable (authorize with Ctrl+O) — surface near the top.
        McpHealth::NeedsAuth => 1,
        McpHealth::Connected => 2,
        McpHealth::Idle => 3,
        McpHealth::Disabled => 4,
    }
}

/// Order `/skills` rows: enabled first, disabled sunk to the bottom; alphabetical
/// within each group.
pub(super) fn sort_skill_rows(rows: &mut [SkillToggle]) {
    rows.sort_by(|a, b| b.enabled.cmp(&a.enabled).then_with(|| a.name.cmp(&b.name)));
}

/// The enabled skills from an open `/skills` overlay, as `/`-menu slash commands.
pub(super) fn enabled_skill_commands(rows: &[SkillToggle]) -> Vec<SkillCommand> {
    rows.iter()
        .filter(|i| i.enabled)
        .map(|i| SkillCommand {
            name: i.name.clone(),
            description: crate::agent::skills::advert_description(&i.description),
        })
        .collect()
}

/// Resolve the agent's requested model against `choices`: exact id > unique substring >
/// ambiguity/miss error. An empty catalog (provider lists nothing) accepts the raw string.
pub(super) fn resolve_model_request(
    requested: &str,
    choices: &[ModelChoice],
) -> std::result::Result<String, String> {
    if choices.is_empty() {
        return Ok(requested.to_string());
    }
    if let Some(c) = choices
        .iter()
        .find(|c| c.id.eq_ignore_ascii_case(requested))
    {
        return Ok(c.id.clone());
    }
    let needle = requested.to_ascii_lowercase();
    let matches: Vec<&ModelChoice> = choices
        .iter()
        .filter(|c| c.id.to_ascii_lowercase().contains(&needle))
        .collect();
    match matches.as_slice() {
        [one] => Ok(one.id.clone()),
        [] => Err(format!(
            "no model matches '{requested}'. Ask the user to run /model to pick, or give an exact id."
        )),
        many => {
            let sample: Vec<&str> = many.iter().take(6).map(|c| c.id.as_str()).collect();
            Err(format!(
                "'{requested}' is ambiguous — matches: {}. Ask the user which one, or suggest /model.",
                sample.join(", ")
            ))
        }
    }
}

/// Resolve the agent's requested key against the saved keys: exact id > exact name >
/// unique case-insensitive substring. Ambiguity/miss names the candidates back.
pub(super) fn resolve_key_request(
    requested: &str,
    keys: &[ApiKey],
) -> std::result::Result<ApiKey, String> {
    if keys.is_empty() {
        return Err("there are no saved keys — the user has to run `aivo keys add`.".to_string());
    }
    let needle = requested.to_ascii_lowercase();
    let name = |k: &ApiKey| k.display_name().to_ascii_lowercase();
    if let Some(k) = keys.iter().find(|k| k.id == requested) {
        return Ok(k.clone());
    }
    let exact: Vec<&ApiKey> = keys.iter().filter(|k| name(k) == needle).collect();
    let matches = if exact.is_empty() {
        keys.iter().filter(|k| name(k).contains(&needle)).collect()
    } else {
        exact
    };
    match matches.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(format!(
            "no saved key matches '{requested}'. Saved keys: {}.",
            key_names(keys.iter())
        )),
        many => Err(format!(
            "'{requested}' is ambiguous — matches: {}. Ask the user which one, or suggest /key.",
            key_names(many.iter().copied())
        )),
    }
}

fn key_names<'a>(keys: impl Iterator<Item = &'a ApiKey>) -> String {
    keys.take(12)
        .map(|k| k.display_name())
        .collect::<Vec<_>>()
        .join(", ")
}

fn pick_a_model_error(key: &ApiKey, choices: &[ModelChoice]) -> String {
    let name = key.display_name();
    if choices.is_empty() {
        return format!("{name} has no remembered model and its model list couldn't be fetched.");
    }
    format!(
        "{name} has no remembered model — call switch_key again with `model` set. Available: {}.",
        choices
            .iter()
            .take(10)
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Two base URLs resolve to the same provider (same wire protocol). The starter
/// sentinel resolves to its real URL first; other sentinels compare literally.
pub(super) fn same_wire_provider(a: &str, b: &str) -> bool {
    use crate::services::provider_profile::resolve_starter_base_url;
    resolve_starter_base_url(a) == resolve_starter_base_url(b)
}

/// Fetch model metadata for the picker, falling back to cached IDs on error.
/// On a successful detailed fetch we also seed the IDs cache so other commands
/// stay warm.
async fn load_model_choices(
    client: &reqwest::Client,
    key: &ApiKey,
    cache: &crate::services::ModelsCache,
) -> Vec<ModelChoice> {
    match crate::services::model_catalog::fetch_models_detailed(client, key).await {
        Ok(infos) => {
            let ids: Vec<String> = infos.iter().map(|m| m.id.clone()).collect();
            let cache_key = crate::services::model_catalog::model_cache_key_for_key(key);
            cache.set(&cache_key, ids).await;
            infos
                .into_iter()
                .map(|m| ModelChoice {
                    label: crate::commands::models::picker_label(&m),
                    image_input: m.image_input,
                    id: m.id,
                })
                .collect()
        }
        Err(_) => fetch_models_for_select(client, key, cache)
            .await
            .into_iter()
            .map(|id| ModelChoice {
                label: id.clone(),
                id,
                image_input: None,
            })
            .collect(),
    }
}
