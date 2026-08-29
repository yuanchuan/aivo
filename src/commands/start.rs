use std::io::IsTerminal;

use anyhow::Result;
use console::{Key, Term};

use crate::cli::parse_env_vars;
use crate::commands::CodeCommand;
use crate::commands::keys::prompt_pick_key_without_activation;
use crate::commands::models::{model_display_label, resolve_model_placeholder};
use crate::commands::print_launch_preview;
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::http_utils;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, LastSelection, SessionStore};
use crate::style;
use crate::tui::FuzzySelect;

#[derive(Debug, Clone)]
pub struct StartFlowArgs {
    pub model: Option<String>,
    pub key: Option<String>,
    pub tool: Option<String>,
    pub dry_run: bool,
    pub refresh: bool,
    pub yes: bool,
    pub envs: Vec<String>,
}

struct Resolved<T> {
    value: T,
    interactive: bool,
}

/// Marker error for a user-cancelled picker: `execute` renders it as the
/// standard dim "Cancelled." notice with exit 0 — backing out isn't an error
/// (same contract as the keys/login flows and the run-path pickers).
#[derive(Debug)]
struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Cancelled.")
    }
}

impl std::error::Error for Cancelled {}

/// No-key launch failure: auth exit code (3), matching `code`/`serve`/`models`,
/// with the standard add-key CTA.
fn no_key_error() -> anyhow::Error {
    crate::errors::CLIError::new(
        "No API key configured.",
        crate::errors::ErrorCategory::Auth,
        None::<String>,
        Some(crate::commands::add_key_cta()),
    )
    .into()
}

/// A picked launch target: a native tool, aivo's chat agent, or a plugin.
enum StartTool {
    Native(AIToolType),
    Code,
    Plugin(String),
}

impl StartTool {
    fn name(&self) -> &str {
        match self {
            StartTool::Native(tool) => tool.as_str(),
            StartTool::Code => "code",
            StartTool::Plugin(name) => name,
        }
    }
}

pub struct StartCommand {
    session_store: SessionStore,
    ai_launcher: AILauncher,
    cache: ModelsCache,
}

impl StartCommand {
    pub fn new(session_store: SessionStore, ai_launcher: AILauncher, cache: ModelsCache) -> Self {
        Self {
            session_store,
            ai_launcher,
            cache,
        }
    }

    pub async fn execute(&self, args: StartFlowArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) if e.is::<Cancelled>() => {
                eprintln!("{}", style::dim("Cancelled."));
                ExitCode::Success
            }
            Err(e) => crate::errors::report(&e),
        }
    }

    async fn execute_internal(&self, args: StartFlowArgs) -> Result<ExitCode> {
        // Use global last selection for defaults
        let last_sel = self.session_store.get_last_selection().await?;

        if last_sel.is_none() {
            eprintln!(
                "{}",
                style::dim("No saved selection yet. I'll help you pick one.")
            );
        }

        let key_explicit = args.key.is_some();
        let model_explicit = args.model.is_some();

        let tool = self.resolve_tool(args.tool.as_deref(), last_sel.as_ref(), args.yes)?;
        let tool = match tool.value {
            // Plugins resolve their own key/model and write the shared last selection.
            StartTool::Plugin(name) => return self.dispatch_plugin(&name, &args).await,
            // The chat agent owns its model picker, sandbox, and conversation loop.
            StartTool::Code => return self.dispatch_code(&args).await,
            StartTool::Native(t) => Resolved {
                value: t,
                interactive: tool.interactive,
            },
        };

        let key = self
            .resolve_key(args.key.as_deref(), last_sel.as_ref())
            .await?;

        // Determine model: if -k was explicit, force picker; otherwise use last selection.
        let model_arg = if model_explicit {
            args.model
        } else if key_explicit {
            // -k used without -m → force model picker
            Some(String::new())
        } else {
            // No -k, no -m → check last selection
            match last_sel.as_ref() {
                Some(sel) if sel.key_id == key.value.id => {
                    sel.model.clone().or(Some(String::new()))
                }
                _ => None, // will trigger picker below
            }
        };
        let model_arg = if model_explicit || key_explicit {
            model_arg
        } else {
            self.refresh_starter_model_arg(&key.value, model_arg).await
        };

        let model = self
            .resolve_model(model_arg, last_sel.as_ref(), &key, args.refresh, tool.value)
            .await?;

        // HF models run against a synthetic local key; skip persisting selection.
        let model_is_hf = model
            .value
            .as_deref()
            .is_some_and(crate::services::huggingface::is_huggingface_ref);
        if !model_is_hf {
            let _ = self
                .session_store
                .set_last_selection(&key.value, tool.value.as_str(), model.value.as_deref())
                .await;
        }

        let launch_model = resolve_model_placeholder(model.value.clone());

        crate::cli::warn_malformed_env(&args.envs);
        let env = parse_env_vars(&args.envs);
        let skip_confirm =
            last_sel.is_some() || (key.interactive && tool.interactive && model.interactive);

        if args.dry_run {
            let plan = self
                .ai_launcher
                .prepare_launch(&LaunchOptions {
                    tool: tool.value,
                    args: Vec::new(),
                    model: launch_model,
                    claude_overrides: Default::default(),
                    env: (!env.is_empty()).then_some(env),
                    key_override: Some(key.value),
                    codex_effort: None,
                    codex_slots: Default::default(),
                })
                .await?;
            print_launch_preview(&plan);
            return Ok(ExitCode::Success);
        }

        if !args.yes {
            let provider = super::truncate_url_for_display(&key.value.base_url, 50);
            eprintln!(
                "{}{}",
                style::cyan(tool.value.as_str()),
                style::dim(format!(
                    " · {} · {}",
                    provider,
                    model_display_label(model.value.as_deref())
                ))
            );
            if !skip_confirm && !confirm("Run?")? {
                return Ok(ExitCode::Success);
            }
        }

        let exit_code = self
            .ai_launcher
            .launch(&LaunchOptions {
                tool: tool.value,
                args: Vec::new(),
                model: launch_model,
                claude_overrides: Default::default(),
                env: (!env.is_empty()).then_some(env),
                key_override: Some(key.value),
                codex_effort: None,
                codex_slots: Default::default(),
            })
            .await?;

        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    async fn resolve_key(
        &self,
        key_arg: Option<&str>,
        last_sel: Option<&LastSelection>,
    ) -> Result<Resolved<ApiKey>> {
        if matches!(key_arg, Some("")) {
            return self.prompt_select_key(last_sel).await;
        }

        if let Some(key_id_or_name) = key_arg {
            let matches = self
                .session_store
                .find_keys_by_id_or_name(key_id_or_name)
                .await?;
            let key = match matches.len() {
                0 | 1 => {
                    self.session_store
                        .resolve_key_by_id_or_name(key_id_or_name)
                        .await?
                }
                _ => {
                    if !std::io::stderr().is_terminal() {
                        self.session_store
                            .resolve_key_by_id_or_name(key_id_or_name)
                            .await?
                    } else {
                        eprintln!(
                            "{} Multiple keys match {}:",
                            crate::style::yellow("Note:"),
                            crate::style::cyan(key_id_or_name)
                        );
                        let prompt = format!("Select key '{}'", key_id_or_name);
                        match prompt_pick_key_without_activation(&matches, &[], &prompt, 0)? {
                            Some(key) => key,
                            None => return Err(Cancelled.into()),
                        }
                    }
                }
            };
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        // Try last selection
        if let Some(sel) = last_sel
            && let Some(key) = self.session_store.get_key_by_id(&sel.key_id).await?
        {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        // Fallback to active key
        if let Some(key) = self.session_store.get_active_key().await? {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => Err(no_key_error()),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => match prompt_pick_key_without_activation(&keys, &[], "Select key", 0)? {
                Some(key) => Ok(Resolved {
                    value: key,
                    interactive: true,
                }),
                None => Err(Cancelled.into()),
            },
        }
    }

    async fn prompt_select_key(
        &self,
        last_sel: Option<&LastSelection>,
    ) -> Result<Resolved<ApiKey>> {
        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => Err(no_key_error()),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => {
                let active_key_id = self
                    .session_store
                    .get_active_key_info()
                    .await?
                    .map(|active_key| active_key.id);
                let default_idx = last_sel
                    .and_then(|sel| keys.iter().position(|key| key.id == sel.key_id))
                    .or_else(|| {
                        active_key_id
                            .as_ref()
                            .and_then(|active_id| keys.iter().position(|key| &key.id == active_id))
                    })
                    .unwrap_or(0);
                match prompt_pick_key_without_activation(&keys, &[], "Select key", default_idx)? {
                    Some(key) => Ok(Resolved {
                        value: key,
                        interactive: true,
                    }),
                    None => Err(Cancelled.into()),
                }
            }
        }
    }

    fn resolve_tool(
        &self,
        tool_arg: Option<&str>,
        last_sel: Option<&LastSelection>,
        yes: bool,
    ) -> Result<Resolved<StartTool>> {
        let plugins = crate::plugin::launchable_coding_agents();
        let parse = |name: &str| -> Option<StartTool> {
            if name.eq_ignore_ascii_case("code") || name.eq_ignore_ascii_case("chat") {
                Some(StartTool::Code)
            } else if let Some(tool) = AIToolType::parse(name) {
                Some(StartTool::Native(tool))
            } else if plugins.iter().any(|p| p == name) {
                Some(StartTool::Plugin(name.to_string()))
            } else {
                None
            }
        };

        if let Some(tool) = tool_arg {
            return Ok(Resolved {
                value: parse(tool).ok_or_else(|| anyhow::anyhow!("Unknown AI tool '{}'", tool))?,
                interactive: false,
            });
        }

        // The remembered tool: replayed verbatim under `-y` or headless;
        // otherwise it pre-selects the picker row so Enter replays it.
        let remembered = last_sel.and_then(|sel| parse(&sel.tool));
        let remembered_name = remembered.as_ref().map(|t| t.name().to_string());
        if (yes || !std::io::stderr().is_terminal())
            && let Some(tool) = remembered
        {
            return Ok(Resolved {
                value: tool,
                interactive: false,
            });
        }

        let plugin_details = crate::plugin::coding_agent_descriptions();
        let mut entries = builtin_tool_entries();
        entries.extend(plugins.iter().map(|name| {
            let detail = plugin_details.get(name).cloned().unwrap_or_default();
            (name.clone(), detail)
        }));
        let items = render_tool_rows(&entries);
        let default_idx = remembered_name
            .and_then(|name| entries.iter().position(|(item, _)| *item == name))
            .unwrap_or(0);
        let selected = FuzzySelect::new()
            .with_prompt("Select tool")
            .items(&items)
            .default(default_idx)
            .interact_opt()
            .ok()
            .flatten()
            .ok_or(Cancelled)?;
        Ok(Resolved {
            value: parse(&entries[selected].0).expect("picker items are parseable"),
            interactive: true,
        })
    }

    /// Hand a coding-agent plugin to the standard plugin dispatch, which owns
    /// key/model resolution, consent gates, the endpoint, and accounting.
    /// `-k`/`-m`/`--dry-run` forward as the flags `aivo <plugin>` accepts.
    async fn dispatch_plugin(&self, name: &str, args: &StartFlowArgs) -> Result<ExitCode> {
        let mut plugin_args: Vec<String> = Vec::new();
        match args.key.as_deref() {
            Some("") => plugin_args.push("-k".to_string()),
            Some(key) => plugin_args.push(format!("--key={key}")),
            None => {}
        }
        match args.model.as_deref() {
            Some("") => plugin_args.push("-m".to_string()),
            Some(model) => plugin_args.push(format!("--model={model}")),
            None => {}
        }
        if args.dry_run {
            plugin_args.push("--dry-run".to_string());
        }
        if !args.envs.is_empty() {
            eprintln!(
                "{} `-e` overrides are not passed to plugins; ignoring.",
                style::yellow("Note:")
            );
        }
        let code = crate::plugin::dispatch_installed(name, &plugin_args, &self.session_store)
            .await
            .ok_or_else(|| anyhow::anyhow!("Plugin `aivo-{name}` is no longer installed."))?;
        Ok(match code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    /// Launch aivo's chat agent. Resolves key (honoring `-k`, last selection, active key).
    async fn dispatch_code(&self, args: &StartFlowArgs) -> Result<ExitCode> {
        // HF models run against a synthetic local key; skip resolution.
        let model_is_hf = args
            .model
            .as_deref()
            .is_some_and(crate::services::huggingface::is_hf_or_local_gguf);
        let key_override = if model_is_hf {
            None
        } else {
            let last_sel = self.session_store.get_last_selection().await?;
            Some(
                self.resolve_key(args.key.as_deref(), last_sel.as_ref())
                    .await?
                    .value,
            )
        };
        // `-k` without `-m` forces the model picker; otherwise hand model through.
        let model = match args.model.clone() {
            Some(m) => Some(m),
            None if args.key.is_some() => Some(String::new()),
            None => None,
        };
        let command = CodeCommand::new(self.session_store.clone(), self.cache.clone());
        Ok(command
            .execute(
                model,
                None,
                None,
                Vec::new(),
                args.refresh,
                key_override,
                args.key.is_some(),
                false,
                None,
                None,
                false,
                false,
                false,
                None,
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .await)
    }

    async fn resolve_model(
        &self,
        model_arg: Option<String>,
        last_sel: Option<&LastSelection>,
        key: &Resolved<ApiKey>,
        refresh: bool,
        tool: AIToolType,
    ) -> Result<Resolved<Option<String>>> {
        // Only use last_sel model when the key matches
        let matching_sel = last_sel.filter(|sel| sel.key_id == key.value.id);
        let explicit_picker = model_arg.as_ref().is_some_and(|value| value.is_empty());
        let should_prompt = explicit_picker || (model_arg.is_none() && matching_sel.is_none());

        if should_prompt {
            return self
                .prompt_select_model(&key.value, refresh, tool, explicit_picker)
                .await;
        }

        match model_arg {
            Some(value) => Ok(Resolved {
                value: Some(value),
                interactive: false,
            }),
            None => Ok(Resolved {
                value: matching_sel.and_then(|sel| sel.model.clone()),
                interactive: false,
            }),
        }
    }

    async fn refresh_starter_model_arg(
        &self,
        key: &ApiKey,
        model_arg: Option<String>,
    ) -> Option<String> {
        let model = model_arg?;
        if model.is_empty() {
            return Some(model);
        }

        if crate::services::model_catalog::starter_model_still_available(key, &self.cache, &model)
            .await
        {
            return Some(model);
        }

        eprintln!(
            "{} Model '{}' is no longer available on aivo-starter. Pick another:",
            style::yellow("Note:"),
            model
        );
        Some(String::new())
    }

    async fn prompt_select_model(
        &self,
        key: &ApiKey,
        refresh: bool,
        tool: AIToolType,
        explicit_picker: bool,
    ) -> Result<Resolved<Option<String>>> {
        // Same rationale as run.rs `resolve_model`: the picker spins on a
        // non-TTY. Bail before the network fetch.
        if !std::io::stderr().is_terminal() {
            if explicit_picker {
                crate::commands::print_no_model_list_hint();
            }
            return Ok(Resolved {
                value: None,
                interactive: false,
            });
        }

        let client = http_utils::router_http_client();
        // Full catalog + per-row annotations: non-chat models show as
        // disabled with a reason, rather than being silently stripped.
        let models = match crate::commands::models::fetch_all_models_for_picker(
            &client,
            key,
            &self.cache,
            refresh,
        )
        .await
        {
            Ok(models) => models,
            Err(e)
                if explicit_picker
                    && !crate::services::model_catalog::is_no_models_endpoint(&e) =>
            {
                return Err(e);
            }
            Err(_) => Vec::new(),
        };
        if models.is_empty() {
            // No fetchable model list (common for providers without a public
            // /v1/models endpoint — e.g. Codex ChatGPT OAuth). Skip the
            // picker and let the tool use its own default rather than
            // blocking the launch. Only explain this when the user
            // explicitly asked for a picker; on the implicit "no prior
            // selection" path the launch just proceeds silently.
            if explicit_picker {
                crate::commands::print_no_model_list_hint();
            }
            return Ok(Resolved {
                value: None,
                interactive: false,
            });
        }

        let annotations = crate::services::model_compat::text_chat_annotations(&models);
        match crate::commands::models::prompt_model_picker(
            models,
            Some(tool),
            annotations,
            "Select model",
        ) {
            Some(selected) => Ok(Resolved {
                value: Some(selected),
                interactive: true,
            }),
            None => Err(Cancelled.into()),
        }
    }
}

/// The built-in picker rows as `(name, detail)` pairs: aivo's own in-process
/// chat agent first (no install, shares the same keys as the tools below), then
/// every native tool supported on this platform. Plugins are appended by the
/// caller (they need runtime discovery). Standalone so the headline ordering is
/// unit-tested.
fn builtin_tool_entries() -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = vec![(
        "code".to_string(),
        "aivo's built-in coding agent (no install needed).".to_string(),
    )];
    entries.extend(
        AIToolType::all()
            .iter()
            .filter(|t| t.supported_on_current_platform())
            .map(|t| {
                let mut detail = t.description().to_string();
                if !t.looks_installed() {
                    detail.push_str(" (not installed)");
                }
                (t.as_str().to_string(), detail)
            }),
    );
    entries
}

/// Aligned `<name>  <detail>` picker rows; detail is painted dim so the name
/// stays the visual anchor (same convention as `format_key_choice`).
fn render_tool_rows(entries: &[(String, String)]) -> Vec<String> {
    let name_w = entries
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0);
    entries
        .iter()
        .map(|(name, detail)| {
            if detail.is_empty() {
                name.clone()
            } else {
                format!("{name:<name_w$}  {}", style::dim(detail))
            }
        })
        .collect()
}

fn confirm(prompt: &str) -> std::io::Result<bool> {
    let term = Term::stdout();
    term.write_str(&format!("{prompt} [Y/n] "))?;

    loop {
        match term.read_key()? {
            Key::Enter | Key::Char('y') | Key::Char('Y') => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Running..."))?;
                return Ok(true);
            }
            Key::Char('n') | Key::Char('N') | Key::Escape => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Cancelled."))?;
                return Ok(false);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::render_tool_rows;
    use crate::services::ai_launcher::AIToolType;

    fn plain(rows: Vec<String>) -> Vec<String> {
        rows.iter()
            .map(|r| console::strip_ansi_codes(r).into_owned())
            .collect()
    }

    #[test]
    fn rows_align_names_to_longest() {
        let entries = vec![
            ("claude".to_string(), "Claude Code · Anthropic".to_string()),
            ("codex-app".to_string(), "Codex Desktop App".to_string()),
        ];
        let rows = plain(render_tool_rows(&entries));
        assert_eq!(rows[0], "claude     Claude Code · Anthropic");
        assert_eq!(rows[1], "codex-app  Codex Desktop App");
    }

    #[test]
    fn empty_detail_renders_bare_name() {
        let entries = vec![
            ("someplugin".to_string(), String::new()),
            ("pi".to_string(), "Pi · Earendil".to_string()),
        ];
        let rows = plain(render_tool_rows(&entries));
        assert_eq!(rows[0], "someplugin");
        assert_eq!(rows[1], "pi          Pi · Earendil");
    }

    #[test]
    fn codex_app_offered_only_where_the_desktop_app_ships() {
        let offered = AIToolType::all()
            .iter()
            .filter(|t| t.supported_on_current_platform())
            .any(|t| *t == AIToolType::CodexApp);
        assert_eq!(offered, cfg!(any(target_os = "macos", windows)));
    }

    #[test]
    fn code_leads_the_builtin_picker_rows() {
        let entries = super::builtin_tool_entries();
        // aivo's own agent is the headline row, ahead of the native tools.
        assert_eq!(entries[0].0, "code");
        assert!(!entries[0].1.is_empty(), "code row carries a description");
        // The native tools still follow it.
        assert!(entries.iter().any(|(name, _)| name == "claude"));
        assert!(entries.iter().any(|(name, _)| name == "codex"));
    }
}
