//! Top-level CLI dispatch.

use std::process;

use clap::{Command, CommandFactory, Parser};
use serde_json::{Map, Value, json};

use crate::cli::{AccountSubcommand, Cli, CodeArgs, Commands};
use crate::cli_args::{
    extract_aivo_flags, lift_context_suffix, needs_bundle_lookup, parse_context_token,
    rewrite_cli_args,
};
use crate::commands::{
    self, AccountCommand, AliasCommand, CodeCommand, InfoCommand, KeysCommand, LoginCommand,
    LogoutCommand, LogsCommand, ModelsCommand, PluginsCommand, RunCommand, ServeCommand,
    ServeParams, ShareCommand, StartCommand, StartFlowArgs, StatsCommand, UpdateCommand,
};
use crate::errors::ExitCode;
use crate::key_resolution::{
    KeyLookupMode, KeyResolution, key_or_exit, resolve_key_override, resolve_key_override_info,
};
use crate::plugin;
use crate::services::ai_launcher::AIToolType;
use crate::services::environment_injector::ClaudeSlotFlags;
use crate::services::key_compat::KeyCompatContext;
use crate::services::session_store::{AliasValue, BundleAlias};
use crate::services::{self, AILauncher, EnvironmentInjector, SessionStore};
use crate::{style, version};

/// Binary built with `__internal_test_fast_crypto` uses reduced PBKDF2 iterations.
/// It cannot decrypt keys encrypted by a normal aivo binary.
/// Override with `AIVO_TEST_FAST_CRYPTO_OK=1`.
#[cfg(feature = "__internal_test_fast_crypto")]
fn fast_crypto_guard() {
    if std::env::var_os("AIVO_TEST_FAST_CRYPTO_OK").is_some() {
        return;
    }
    eprintln!(
        "{} This aivo binary was built with the `__internal_test_fast_crypto` feature,\n       which uses reduced PBKDF2 iterations for fast tests.\n       It cannot decrypt keys encrypted by a normal aivo binary.\n\n       Rebuild without the feature: {}\n       Or, to override intentionally, set {}",
        style::red("error:"),
        style::cyan("cargo build"),
        style::dim("AIVO_TEST_FAST_CRYPTO_OK=1"),
    );
    process::exit(ExitCode::UserError.code());
}

#[cfg(not(feature = "__internal_test_fast_crypto"))]
fn fast_crypto_guard() {}

/// If `--debug` was passed on the CLI, resolve its path (default vs explicit)
/// and initialize the global HTTP debug logger so subsequent `.send_logged()`
/// calls capture request/response details. Prints the resolved log path to
/// stderr; on open failure, warns and continues without logging.
/// Outer-dispatcher predicate for skipping key resolution; the HF flow
/// synthesizes its own loopback `ApiKey`.
fn is_hf_takeover(model: Option<&str>) -> bool {
    use services::huggingface;
    match model {
        Some(m) => huggingface::is_hf_or_local_gguf(m) || huggingface::is_bare_hf_picker_trigger(m),
        None => false,
    }
}

#[derive(Debug, PartialEq)]
enum CodePositionalReject {
    /// Bare `[a-z0-9-]` word — a (typo'd) subcommand, not text.
    NotACommand(String),
    /// Text alongside a flag that already owns the prompt slot.
    Conflicts(String, &'static str),
}

/// Classifies `aivo code <positional>`: a model ref stays in `reference`, a
/// bare `-` lifts into the empty prompt (stdin, matching bare `aivo code -p`),
/// and free text returns `Ok(Some(_))` — sent as the TUI's first message
/// (unlike top-level `aivo "..."`, which stays a one-shot `-p`).
fn take_code_initial_prompt(args: &mut CodeArgs) -> Result<Option<String>, CodePositionalReject> {
    let Some(raw) = args.reference.clone() else {
        return Ok(None);
    };
    if is_hf_takeover(Some(raw.as_str())) {
        return Ok(None);
    }
    // A `<key>::<model>` spec is a model ref, not prompt text (coexists with -p).
    if raw.contains("::") {
        return Ok(None);
    }
    if raw == "-" {
        args.prompt.get_or_insert(String::new());
        args.reference = None;
        return Ok(None);
    }
    if raw.is_empty() || crate::cli_args::is_subcommand_shaped(&raw) {
        return Err(CodePositionalReject::NotACommand(raw));
    }
    if args.prompt.is_some() {
        return Err(CodePositionalReject::Conflicts(raw, "-p/--prompt"));
    }
    if args.exec.is_some() {
        return Err(CodePositionalReject::Conflicts(raw, "-e/--exec"));
    }
    if args.resume.is_some() {
        return Err(CodePositionalReject::Conflicts(raw, "--resume"));
    }
    args.reference = None;
    Ok(Some(raw))
}

async fn maybe_init_http_debug(value: &Option<String>) {
    let Some(raw) = value else {
        return;
    };
    let path = if raw.is_empty() {
        services::http_debug::default_log_path()
    } else {
        std::path::PathBuf::from(raw)
    };
    match services::http_debug::init(path).await {
        Ok(p) => eprintln!("[aivo] HTTP debug log → {}", p.display()),
        Err(e) => {
            eprintln!("[aivo] failed to open debug log: {e}; HTTP requests will not be logged")
        }
    }
}

/// CLI entry point. Parses argv, routes to the matching command handler, and
/// exits the process with the resulting code. Never returns.
pub async fn run() -> ! {
    // Internal re-exec for the agent's `run_bash` sandbox: the binary launches
    // itself as `aivo __agent-sandbox --workspace <cwd> -- <shell…>` so it can
    // install a Landlock ruleset (Linux) in a fresh process before running the
    // shell. Handle it FIRST — before clap (which would reject the unknown
    // subcommand), the fast-crypto guard, and all service init — since it needs
    // none of them. (Linux-only; macOS uses the external `sandbox-exec` wrapper
    // and never emits this subcommand.)
    #[cfg(target_os = "linux")]
    {
        let raw_args: Vec<String> = std::env::args().collect();
        if raw_args.get(1).map(String::as_str) == Some("__agent-sandbox") {
            crate::agent::sandbox::run_sandbox_child(&raw_args);
        }
        // Past the dispatch above: this is the real CLI process, which DOES handle
        // the `__agent-sandbox` subcommand, so the agent's Landlock sandbox is
        // cleared to relaunch it. Test/embedding binaries never reach here, so
        // they leave it off and run the bare shell instead of relaunching themselves.
        crate::agent::sandbox::enable_landlock_relaunch();
    }

    fast_crypto_guard();
    // Must run before any reqwest client is built or `spawn_blocking` is
    // launched — reqwest snapshots proxy env at construction, and env
    // mutation is only race-free while the current-thread runtime is idle.
    services::launch_runtime::ensure_loopback_no_proxy_in_process_env();

    // One-shot layout migration (flat config dir → state/secrets/cache/logs/run
    // split). Cheap once migrated (a dozen stats); must precede every store.
    services::paths::migrate_layout(&services::paths::config_dir());

    let raw_args: Vec<String> = std::env::args().collect();

    // Detached background update checker; handle before clap/service init.
    if raw_args.get(1).map(String::as_str) == Some("__update-check") {
        services::update_check::run_check_and_exit().await;
    }

    // Initialize services. Bundle aliases need to be loaded *before* CLI
    // parsing because `aivo <bundle>` and `aivo run <bundle>` are expanded by
    // `rewrite_cli_args` ahead of clap.
    let session_store = SessionStore::new();
    let bundle_index = if needs_bundle_lookup(&raw_args) {
        load_bundle_index(&session_store).await
    } else {
        std::collections::HashMap::new()
    };

    // Route an unowned `aivo <name>` / `aivo run <name>` to its `aivo-<name>`
    // plugin before clap can reject it (see `crate::plugin`).
    if let Some(code) = plugin::try_dispatch(&raw_args, &bundle_index, &session_store).await {
        process::exit(code);
    }

    // `mcp`/`skills` live under `aivo code`; reject the bare top-level forms.
    if let Some(name @ ("mcp" | "skills")) = raw_args.get(1).map(String::as_str) {
        eprintln!(
            "{} unknown command `{name}` — use `aivo code {name}`",
            style::red("Error:")
        );
        process::exit(1);
    }

    let rewritten = rewrite_cli_args(raw_args, &bundle_index);
    let args = match Cli::try_parse_from(&rewritten) {
        Ok(args) => args,
        Err(err) => {
            // Nested `InvalidSubcommand` (`aivo account frobnicate`) shares this
            // kind — only intercept when the rejected token is argv[1].
            if err.kind() == clap::error::ErrorKind::InvalidSubcommand
                && let Some(name) = rewritten.get(1)
                && crate::cli_args::is_subcommand_shaped(name)
                && err
                    .get(clap::error::ContextKind::InvalidSubcommand)
                    .is_some_and(|v| matches!(v, clap::error::ContextValue::String(s) if s == name))
            {
                print_unknown_top_level(name);
                process::exit(ExitCode::UserError.code());
            }
            err.exit();
        }
    };

    // Handle --version and subcommand --help early, before any further service initialization.
    if args.version {
        print_version();
        process::exit(0);
    }

    if args.help_json {
        print_help_json();
        process::exit(0);
    }

    let models_cache = services::ModelsCache::new();

    if args.help
        && let Some(cmd) = &args.command
    {
        match cmd {
            Commands::Run(run_args) => RunCommand::print_help(run_args.tool.as_deref()),
            Commands::Keys(keys_args) => KeysCommand::print_help(keys_args.action.as_deref()),
            Commands::Account(a) => {
                let sub = a.command.as_ref().map(|c| match c {
                    AccountSubcommand::Login(_) => "login",
                    AccountSubcommand::Logout(_) => "logout",
                    AccountSubcommand::Info(_) => "info",
                    AccountSubcommand::Usage(_) => "usage",
                    AccountSubcommand::Open(_) => "open",
                });
                AccountCommand::print_help(sub)
            }
            Commands::Login(_) => LoginCommand::print_help(),
            Commands::Logout(_) => LogoutCommand::print_help(),
            Commands::Code(_) => CodeCommand::print_help(),
            Commands::Models(_) => ModelsCommand::print_help(),
            Commands::Serve(_) => ServeCommand::print_help(),
            Commands::Alias(_) => AliasCommand::print_help(),
            Commands::Info(_) => InfoCommand::print_help(),
            Commands::Logs(logs_args) => LogsCommand::print_help(logs_args.action.as_deref()),
            Commands::Stats(_) => StatsCommand::print_help(),
            Commands::Update(_) => UpdateCommand::print_help(),
            Commands::Hf(_) => crate::commands::hf::HfCommand::print_help(),
            Commands::Plugins(_) => PluginsCommand::print_help(),
            Commands::Mcp(_) => crate::commands::mcp::McpCommand::print_help(),
            Commands::Skills(_) => crate::commands::skills::SkillsCommand::print_help(),
            Commands::Share(_) => ShareCommand::print_help(),
            Commands::Guide => commands::guide::print_guide(),
        }
        process::exit(0);
    }

    // Ensure the free starter key exists for all users.
    // For new users (no keys), also activate it.
    if let Some((starter, is_new_user)) = session_store.ensure_starter_key().await
        && is_new_user
    {
        let _ = session_store.set_active_key(&starter.id).await;
    }

    if args.help {
        print_help();
        print_active_selection(&session_store).await;
        process::exit(0);
    }

    // Get the command or show help if none provided
    let command = match args.command {
        Some(cmd) => cmd,
        None => {
            print_help();
            print_active_selection(&session_store).await;
            process::exit(0);
        }
    };

    // Skip the background check + notice when the user is already updating.
    let is_update_cmd = matches!(command, Commands::Update(_));
    if !is_update_cmd {
        services::update_check::maybe_spawn_background_check();
    }

    // Route to command handler
    let exit_code = match command {
        Commands::Alias(alias_args) => {
            let command = AliasCommand::new(session_store);
            command.execute(alias_args).await
        }

        Commands::Keys(keys_args) => {
            let command = KeysCommand::new(session_store);
            command.execute(keys_args).await
        }

        Commands::Account(account_args) => {
            let command = AccountCommand::new(session_store);
            command.execute(account_args).await
        }

        Commands::Login(login_args) => {
            let command = LoginCommand::new(session_store);
            command.execute(login_args).await
        }

        Commands::Logout(logout_args) => {
            let command = LogoutCommand::new();
            command.execute(logout_args).await
        }

        Commands::Code(mut code_args) => {
            maybe_init_http_debug(&code_args.debug).await;
            // Roots must exist (a typo'd root would silently allow nothing).
            for dir in &code_args.add_dir {
                let path = crate::services::system_env::expand_tilde(dir);
                if !path.is_dir() {
                    eprintln!("{} --add-dir {dir}: not a directory", style::red("Error:"));
                    process::exit(ExitCode::UserError.code());
                }
                crate::agent::sandbox::add_extra_write_root(path);
            }
            if let Some(profile) = code_args
                .sandbox
                .as_deref()
                .and_then(crate::agent::sandbox::SandboxProfile::parse)
            {
                crate::agent::sandbox::set_sandbox_profile(profile);
            }
            if let Some(n) = code_args.best_of_n {
                crate::commands::code_agent_oneshot::set_best_of_n(n as usize);
                // Candidates share the working tree; default to read-only so
                // parallel runs can't race writes (--sandbox/env overrides).
                if n >= 2 {
                    crate::agent::sandbox::set_read_only_unless_configured();
                }
            }
            if let Some(spec) = code_args.json_schema.take() {
                // `@path` loads the schema from a file; otherwise it's inline JSON.
                let schema = match spec.strip_prefix('@') {
                    Some(path) => match std::fs::read_to_string(
                        crate::services::system_env::expand_tilde(path),
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("{} --json-schema {path}: {e}", style::red("Error:"));
                            process::exit(ExitCode::UserError.code());
                        }
                    },
                    None => spec,
                };
                if serde_json::from_str::<serde_json::Value>(&schema).is_err() {
                    eprintln!("{} --json-schema: not valid JSON", style::red("Error:"));
                    process::exit(ExitCode::UserError.code());
                }
                crate::commands::code_agent_oneshot::set_json_schema(schema);
            }
            let initial_prompt = match take_code_initial_prompt(&mut code_args) {
                Ok(prompt) => prompt,
                Err(CodePositionalReject::NotACommand(raw)) => {
                    eprintln!(
                        "{} `aivo code` got an unexpected argument: {raw:?}",
                        style::red("Error:"),
                    );
                    eprintln!(
                        "  {}",
                        style::dim(
                            "Expected a subcommand (mcp, skills), a model ref (`hf:<owner>/<repo>` or `https://huggingface.co/...`), or quoted text to send as the first TUI message."
                        ),
                    );
                    eprintln!(
                        "  {}",
                        style::dim(
                            "A bare word reads as a subcommand — use -e \"...\" (agent), -p \"...\" (plain reply), or `aivo code \"...\"` to open the TUI."
                        ),
                    );
                    process::exit(ExitCode::UserError.code());
                }
                Err(CodePositionalReject::Conflicts(raw, flag)) => {
                    eprintln!(
                        "{} `aivo code {raw:?}` can't be combined with {flag} — pick one way to start the session.",
                        style::red("Error:"),
                    );
                    process::exit(ExitCode::UserError.code());
                }
            };
            // Positional lifts into model; explicit -m still wins.
            let model_input = code_args
                .model
                .clone()
                .or_else(|| code_args.reference.clone());
            let have_model_input = model_input.is_some();
            // Split `[key::]model`; the model half is alias-expanded (may itself
            // carry `key::`). Empty half (`aivo::`) is key-only → picker.
            let (model_key_ref, model_half) = match model_input {
                Some(v) => {
                    let aliases = session_store.get_aliases().await.unwrap_or_default();
                    let (key_ref, model) = crate::cli_args::split_tier_spec(&v);
                    crate::cli_args::resolve_alias_with_tier(&aliases, key_ref, model)
                }
                None => (None, String::new()),
            };
            if let (Some(k), Some(kr)) = (code_args.key.as_deref(), model_key_ref.as_deref())
                && k != kr
            {
                eprintln!(
                    "{} -k '{k}' conflicts with the provider in '{kr}::…' — pass just one.",
                    style::red("Error:")
                );
                process::exit(ExitCode::UserError.code());
            }
            let key_explicit = code_args.key.is_some() || model_key_ref.is_some();
            let effective_key = code_args.key.clone().or(model_key_ref);
            let expanded_model = have_model_input.then_some(model_half);
            let key_override = if is_hf_takeover(expanded_model.as_deref()) {
                None
            } else {
                key_or_exit(
                    resolve_key_override(
                        &session_store,
                        effective_key.as_deref(),
                        KeyLookupMode::RequireActiveOrPrompt,
                        KeyCompatContext::Chat,
                    )
                    .await,
                )
            };
            // -k (or a `key::` prefix) without a model forces the picker.
            let model = if have_model_input {
                expanded_model
            } else if key_explicit {
                Some(String::new())
            } else {
                None
            };
            // -e runs the agent (auto-approved tools); -p a plain completion (clap-exclusive).
            let agent_mode = code_args.exec.is_some();
            let one_shot = code_args.exec.take().or_else(|| code_args.prompt.take());
            let command = CodeCommand::new(session_store, models_cache.clone());
            command
                .execute(
                    model,
                    one_shot,
                    initial_prompt,
                    code_args.attachments,
                    code_args.refresh,
                    key_override,
                    key_explicit,
                    code_args.json,
                    code_args.resume,
                    // `--1m`/`--2m` shorthands collapse into max_context, same
                    // as the `run` path; chat takes it as a raw window size.
                    code_args
                        .max_context
                        .or_else(|| code_args.one_m.then(|| "1m".to_string()))
                        .or_else(|| code_args.two_m.then(|| "2m".to_string())),
                    code_args.dry_run,
                    code_args.share,
                    agent_mode,
                    code_args.output_format,
                    code_args.max_steps,
                    code_args.max_output_tokens,
                    code_args.max_cost,
                    code_args.auto_approve,
                    code_args.vision_model,
                    code_args.image_model,
                )
                .await
        }

        Commands::Run(run_args) => {
            let env_injector = EnvironmentInjector::new();
            let ai_launcher =
                AILauncher::new(session_store.clone(), env_injector, models_cache.clone());

            // Re-extract aivo flags from passthrough args that clap's trailing_var_arg
            // may have swallowed (e.g. `aivo run claude --agent-name foo --model opus`
            // puts --model into args instead of parsing it as an aivo flag).
            let mut extracted = extract_aivo_flags(
                run_args.model,
                ClaudeSlotFlags {
                    fable: run_args.fable_model,
                    subagent: run_args.subagent_model,
                    haiku: run_args.haiku_model,
                    sonnet: run_args.sonnet_model,
                    opus: run_args.opus_model,
                },
                run_args.key,
                run_args.debug,
                run_args.dry_run,
                run_args.refresh,
                run_args.relogin,
                run_args.envs,
                // `--1m`/`--2m` shorthands collapsed into max_context up
                // front so every downstream consumer sees a single signal.
                run_args
                    .max_context
                    .or_else(|| run_args.one_m.then(|| "1m".to_string()))
                    .or_else(|| run_args.two_m.then(|| "2m".to_string())),
                run_args.effort,
                run_args.review_model,
                &run_args.args,
            );
            // Pi defaults to the router; `--transparent` opts back into native.
            let is_pi =
                run_args.tool.as_deref().and_then(AIToolType::parse) == Some(AIToolType::Pi);
            let transparent = run_args.transparent || extracted.transparent;
            let transform_on = run_args.transform || extracted.transform || is_pi;
            services::transform_mode::set_active(transform_on && !transparent);
            services::transform_mode::set_transparent(transparent);
            // After extract_aivo_flags so `--debug` after the tool name
            // (recovered from passthrough) activates the logger too.
            maybe_init_http_debug(&extracted.debug).await;
            // A bare leading `<key>::<model>` acts like `-m` — but only if `key`
            // names a stored key, else it's left as passthrough for the tool.
            if extracted.model.is_none()
                && extracted
                    .remaining_args
                    .first()
                    .is_some_and(|a| crate::cli_args::looks_like_key_model_spec(a))
            {
                let (key_ref, _) = crate::cli_args::split_tier_spec(&extracted.remaining_args[0]);
                let key_is_real = match &key_ref {
                    Some(kr) => session_store
                        .find_keys_by_id_or_name_info(kr)
                        .await
                        .map(|m| !m.is_empty())
                        .unwrap_or(false),
                    None => false,
                };
                if key_is_real {
                    extracted.model = Some(extracted.remaining_args.remove(0));
                } else if let Some(kr) = key_ref {
                    // Passed through, but flag it — a silent forward reads as "it worked".
                    let tool = run_args.tool.as_deref().unwrap_or("the tool");
                    eprintln!(
                        "{} '{kr}' isn't a known key — forwarding {:?} to {tool} as a plain argument.",
                        style::yellow("Note:"),
                        extracted.remaining_args[0],
                    );
                    eprintln!(
                        "  {}",
                        style::dim(
                            "For a <key>::<model> spec use a real key; run `aivo keys` to list them."
                        )
                    );
                }
            }
            // Resolve aliases for main + 6 slot models against a single
            // in-memory snapshot of the alias map, instead of paying one disk
            // read per call (worst case 7).
            let aliases = session_store.get_aliases().await.unwrap_or_default();
            // `-m <keyRef>::<model>`: split off the provider; the model half
            // (or an alias expanding to `key::model`) is resolved after.
            let (main_key_ref, resolved_model) = match extracted.model {
                Some(v) => {
                    let (key_ref, model) = crate::cli_args::split_tier_spec(&v);
                    let (key_ref, model) =
                        crate::cli_args::resolve_alias_with_tier(&aliases, key_ref, model);
                    (key_ref, Some(model))
                }
                None => (None, None),
            };
            // Raw: `resolve_claude_overrides` splits the `<key>::` prefix and
            // alias-resolves the model half itself.
            let slots = extracted.slots;
            // Normalize `-m foo[1m]`/`-m foo[2m]` (and any alias that expands
            // to one) into the same internal state as `-m foo --1m`/`--2m`.
            // Without this, mixing the two — `-m foo[1m] --1m` — would
            // double-append the suffix and the env injector would emit
            // `foo[1m][1m]`.
            let (model, max_context) = lift_context_suffix(resolved_model, extracted.max_context);
            // Validate shape (must be `<digits>m`) and tool restriction before
            // any picker UI runs — otherwise the user picks a key + model and
            // *then* gets told their flag is malformed. The tier itself
            // (1m/2m/12m/…) isn't validated; aivo passes whatever the user
            // gave through to Claude as a `[<N>m]` model-name suffix.
            // Claude takes `--max-context` as its `1m`/`2m` beta tag; every other
            // tool takes it as a manual window size, set via the resolve_limits
            // override.
            let parsed_tool = run_args.tool.as_deref().and_then(AIToolType::parse);
            let max_context = match max_context.as_deref() {
                Some(value) if parsed_tool == Some(AIToolType::Claude) => {
                    let Some(canonical) = parse_context_token(value) else {
                        eprintln!(
                            "{} --max-context expects a value like '1m' or '12m' (got {:?}).",
                            style::red("Error:"),
                            value
                        );
                        process::exit(ExitCode::UserError.code());
                    };
                    Some(canonical)
                }
                Some(value) => {
                    let Some(tokens) = services::model_metadata::parse_context_size(value) else {
                        eprintln!(
                            "{} --max-context expects a size like '200k', '1m', or '128000' (got {:?}).",
                            style::red("Error:"),
                            value
                        );
                        process::exit(ExitCode::UserError.code());
                    };
                    services::model_metadata::set_context_window_override(tokens);
                    None
                }
                None => None,
            };
            // `-k` wins; a conflicting `-m <key>::` is a user error, not a silent drop.
            if let (Some(k), Some(kr)) = (extracted.key_flag.as_deref(), main_key_ref.as_deref())
                && k != kr
            {
                eprintln!(
                    "{} -k '{k}' conflicts with the provider in -m '{kr}::…' — set the default model's provider once.",
                    style::red("Error:")
                );
                process::exit(ExitCode::UserError.code());
            }
            let key_flag = extracted.key_flag.or(main_key_ref);
            let dry_run = extracted.dry_run;
            let refresh = extracted.refresh;
            let relogin = extracted.relogin;
            // Recovered from passthrough — `--resume` is not a clap arg (see `RunArgs`).
            let resume_selector = extracted.resume;
            let effort = extracted.effort;
            let review_model = extracted.review_model;
            let env_strings = extracted.env_strings;
            let remaining_args = extracted.remaining_args;

            // Bare `aivo run` opens the `start` tool picker; `aivo run code`
            // names aivo's own in-process agent. Both route through the start
            // flow — `code` isn't an external `AIToolType`, so it can't take the
            // launcher pipeline below; the picker dispatches it to `aivo code`.
            // `chat` stays accepted as the pre-rename alias.
            let code_selected = matches!(run_args.tool.as_deref(), Some("code") | Some("chat"));
            if run_args.tool.is_none() || code_selected {
                if !remaining_args.is_empty() {
                    if code_selected {
                        eprintln!(
                            "{} `aivo run code` does not accept passthrough args",
                            style::red("Error:")
                        );
                        eprintln!(
                            "  {}",
                            style::dim(
                                "Use `aivo code -p \"<prompt>\"` for a one-shot message, or `aivo code` for the TUI."
                            )
                        );
                    } else {
                        eprintln!(
                            "{} `aivo run` without a tool does not accept passthrough args",
                            style::red("Error:")
                        );
                        eprintln!(
                            "  {}",
                            style::dim("Use `aivo run <tool> ...` for passthrough flags.")
                        );
                    }
                    process::exit(ExitCode::UserError.code());
                }

                // The start flow has no resume hook; warn rather than drop it silently.
                if resume_selector.is_some() {
                    eprintln!(
                        "  {} --resume is ignored here; use `aivo code --resume` or `aivo <tool> --resume=<id>`",
                        style::yellow("!")
                    );
                }
                if effort.is_some() {
                    eprintln!(
                        "  {} --effort is ignored here; use `aivo codex --effort <level>`",
                        style::yellow("!")
                    );
                }
                if review_model.is_some() {
                    eprintln!(
                        "  {} --review-model is ignored here; use `aivo codex --review-model <model>`",
                        style::yellow("!")
                    );
                }
                let command = StartCommand::new(session_store, ai_launcher, models_cache);
                command
                    .execute(StartFlowArgs {
                        model,
                        key: key_flag,
                        tool: run_args.tool.clone(),
                        dry_run,
                        refresh,
                        yes: run_args.yes,
                        envs: env_strings,
                    })
                    .await
            } else {
                let command =
                    RunCommand::new(session_store.clone(), ai_launcher, models_cache.clone());

                let key_explicit = key_flag.is_some();
                let compat = run_args
                    .tool
                    .as_deref()
                    .and_then(AIToolType::parse)
                    .map(KeyCompatContext::Tool)
                    .unwrap_or(KeyCompatContext::None);
                // Bare `--model` alongside tier flags picks its provider like
                // the tiers do, instead of reusing the active key.
                let force_default_provider = key_flag.is_none()
                    && slots.any_set()
                    && matches!(model.as_deref(), Some(""))
                    && !dry_run;
                let key_flag = if force_default_provider {
                    Some("")
                } else {
                    key_flag.as_deref()
                };
                let key_override = if is_hf_takeover(model.as_deref()) {
                    None
                } else {
                    key_or_exit(
                        resolve_key_override(
                            &session_store,
                            key_flag,
                            KeyLookupMode::RequireActiveOrPrompt,
                            compat,
                        )
                        .await,
                    )
                };

                // --relogin: drive the OAuth flow up front so the launch sees
                // a fresh credential. Done after key resolution (so the user
                // sees the picker if they didn't pass `-k`) and before any
                // model picker (so a stale token doesn't lead to a model list
                // the user can't actually use).
                let key_override = if relogin {
                    let Some(key) = key_override else {
                        eprintln!(
                            "{} --relogin requires a key — none selected.",
                            style::red("Error:")
                        );
                        process::exit(ExitCode::UserError.code());
                    };
                    match services::oauth_relogin::relogin_key(&session_store, &key).await {
                        Ok(updated) => Some(updated),
                        Err(e) => {
                            process::exit(crate::errors::report(&e).code());
                        }
                    }
                } else {
                    key_override
                };

                // Resolve model using last selection when no explicit flags given.
                // When -k is used without -m, normally force the model picker —
                // except under --dry-run, where the user just wants to see what
                // would launch without being forced into an interactive prompt.
                let model_flag_explicit = model.is_some();
                let last_sel_for_key = session_store
                    .get_last_selection()
                    .await
                    .ok()
                    .flatten()
                    .filter(|sel| key_override.as_ref().is_some_and(|k| k.id == sel.key_id));
                // First-run-only fallback: when the user has just installed
                // aivo and hasn't added any of their own keys yet (only the
                // auto-created `aivo-starter` key exists, and it's active),
                // skip the model picker and use the seeded chat model
                // (`aivo/starter` from `ensure_starter_key`). Once the user
                // adds another key, the fallback no longer applies and the
                // picker resumes its normal behavior.
                let persisted_model_for_key =
                    match last_sel_for_key.as_ref().and_then(|sel| sel.model.clone()) {
                        Some(m) => Some(m),
                        None => {
                            if let Some(k) = key_override.as_ref()
                                && services::provider_profile::is_aivo_starter_base(&k.base_url)
                                && session_store
                                    .get_keys()
                                    .await
                                    .map(|keys| keys.len() == 1)
                                    .unwrap_or(false)
                            {
                                session_store.get_code_model(&k.id).await.ok().flatten()
                            } else {
                                None
                            }
                        }
                    };
                // Drop a persisted starter model that's been removed from the
                // server catalog so the picker reopens with the current list.
                // Skipped when --model was explicit — we trust what the user
                // typed and let upstream surface any mismatch.
                let persisted_model_for_key = if model_flag_explicit {
                    persisted_model_for_key
                } else {
                    match (persisted_model_for_key, key_override.as_ref()) {
                        (Some(m), Some(k))
                            if services::provider_profile::is_aivo_starter_base(&k.base_url) =>
                        {
                            if services::model_catalog::starter_model_still_available(
                                k,
                                &models_cache,
                                &m,
                            )
                            .await
                            {
                                Some(m)
                            } else {
                                eprintln!(
                                    "{} Model '{}' is no longer available on aivo-starter. Pick another:",
                                    style::yellow("Note:"),
                                    m
                                );
                                None
                            }
                        }
                        (other, _) => other,
                    }
                };
                let model = if model.is_some() {
                    model
                } else if dry_run {
                    // --dry-run never opens a picker. Reuse persisted model for
                    // this key if any; otherwise let the tool's default surface
                    // in the dry-run output.
                    persisted_model_for_key
                } else if key_explicit {
                    // -k used without -m → force model picker
                    Some(String::new())
                } else {
                    // Same key as last time → reuse saved model (could be
                    // __default__); otherwise empty string triggers the picker.
                    persisted_model_for_key.or(Some(String::new()))
                };

                let env = if !env_strings.is_empty() {
                    crate::cli::warn_malformed_env(&env_strings);
                    Some(crate::cli::parse_env_vars(&env_strings))
                } else {
                    None
                };

                command
                    .execute(
                        run_args.tool.as_deref(),
                        remaining_args,
                        dry_run,
                        refresh,
                        model,
                        model_flag_explicit,
                        slots,
                        env,
                        key_override,
                        resume_selector,
                        max_context,
                        effort,
                        review_model,
                    )
                    .await
            }
        }

        Commands::Models(models_args) => {
            let (key_flag, search) = match commands::models::merge_models_spec(
                models_args.spec,
                models_args.key,
                models_args.search,
            ) {
                Ok(merged) => merged,
                Err(msg) => {
                    eprintln!("{} {msg}", style::red("Error:"));
                    process::exit(ExitCode::UserError.code());
                }
            };
            let key_override = key_or_exit(
                resolve_key_override_info(
                    &session_store,
                    key_flag.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                    KeyCompatContext::None,
                )
                .await,
            );
            let command = ModelsCommand::new(session_store, models_cache);
            command
                .execute(key_override, models_args.refresh, search, models_args.json)
                .await
        }

        Commands::Serve(serve_args) => {
            // Reject non-HF positionals up-front so a typo doesn't fall
            // through to "no key" mode.
            if let Some(ref raw) = serve_args.reference
                && !is_hf_takeover(Some(raw.as_str()))
            {
                eprintln!(
                    "{} `aivo serve <REF>` only accepts a HuggingFace ref (`hf:<owner>/<repo>` or `https://huggingface.co/...`). Got: {}",
                    style::red("Error:"),
                    raw,
                );
                eprintln!(
                    "  {}",
                    style::dim("For a remote provider, omit the positional and use `-k <key>`."),
                );
                process::exit(ExitCode::UserError.code());
            }
            let hf_mode = serve_args.reference.is_some();
            if hf_mode {
                // Warn before download starts so it lands above the spinner.
                if serve_args.key.is_some() {
                    eprintln!(
                        "  {} -k / --key is ignored when a local model is specified",
                        style::yellow("!"),
                    );
                }
                if serve_args.failover {
                    eprintln!(
                        "  {} --failover is ignored when a local model is specified",
                        style::yellow("!"),
                    );
                }
            }
            let hf_takeover_key = if hf_mode {
                let raw = serve_args.reference.as_deref().unwrap_or("");
                let hf_ref = match services::huggingface::parse_hf_ref(raw) {
                    Ok(r) => r,
                    Err(e) => {
                        process::exit(crate::errors::report(&e).code());
                    }
                };
                match services::huggingface::ensure_ready(&hf_ref).await {
                    Ok(port) => {
                        let base_url = services::huggingface::local_openai_base_url(port);
                        Some(services::session_store::ApiKey::new_with_protocol(
                            "aivo-hf-local".to_string(),
                            format!("hf:{}", hf_ref.repo),
                            base_url,
                            None,
                            "huggingface".to_string(),
                        ))
                    }
                    Err(e) => {
                        eprintln!("{} {:#}", style::red("Error:"), e);
                        // ensure_ready may have stored the child in
                        // SERVER_CHILD before its health-check failed;
                        // `process::exit` skips destructors so we have
                        // to kill the orphan explicitly.
                        services::huggingface::stop_if_we_started();
                        process::exit(ExitCode::UserError.code());
                    }
                }
            } else {
                None
            };

            let key_override = if let Some(key) = hf_takeover_key {
                Some(key)
            } else {
                match resolve_key_override(
                    &session_store,
                    serve_args.key.as_deref(),
                    KeyLookupMode::PreferActiveAllowNone,
                    KeyCompatContext::None,
                )
                .await
                {
                    Ok(KeyResolution::Selected(key)) => Some(key),
                    Ok(KeyResolution::Cancelled) => process::exit(ExitCode::Success.code()),
                    Ok(KeyResolution::MissingAuth) => None,
                    Err(e) => {
                        process::exit(crate::errors::report(&e).code());
                    }
                }
            };
            let failover_keys = if serve_args.failover && !hf_mode {
                let mut all_keys = session_store.get_keys().await.unwrap_or_default();
                // Decrypt and exclude the primary key
                let primary_id = key_override.as_ref().map(|k| k.id.clone());
                all_keys.retain(|k| primary_id.as_deref() != Some(&k.id) && !k.is_any_oauth());
                all_keys.iter_mut().for_each(|k| {
                    let _ = SessionStore::decrypt_key_secret(k);
                });
                all_keys
            } else {
                Vec::new()
            };
            // Snapshot aliases at startup; edits made while serve is running
            // require a restart (matches `aivo run`'s behavior).
            let aliases = session_store.get_aliases().await.unwrap_or_default();
            let command = ServeCommand::new(session_store);
            command
                .execute(ServeParams {
                    port: serve_args.port,
                    host: serve_args.host,
                    key_override,
                    log: serve_args.log,
                    failover_keys,
                    cors: serve_args.cors,
                    timeout: serve_args.timeout,
                    auth_token: serve_args.auth_token,
                    aliases,
                })
                .await
        }

        Commands::Info(info_args) => {
            let command = InfoCommand::new(session_store);
            command.execute(info_args.ping, info_args.json).await
        }

        Commands::Logs(logs_args) => {
            let command = LogsCommand::new(session_store);
            command.execute(logs_args).await
        }

        Commands::Stats(stats_args) => {
            let command = StatsCommand::new(session_store);
            command.execute(stats_args).await
        }

        Commands::Hf(hf_args) => {
            let command = crate::commands::hf::HfCommand::new();
            command.execute(hf_args).await
        }

        Commands::Plugins(plugin_args) => {
            let command = PluginsCommand::new();
            command.execute(plugin_args).await
        }

        Commands::Mcp(mcp_args) => {
            let command = crate::commands::mcp::McpCommand::new();
            command.execute(mcp_args).await
        }

        Commands::Skills(skills_args) => {
            let command = crate::commands::skills::SkillsCommand::new();
            command.execute(skills_args).await
        }

        Commands::Share(share_args) => {
            let command = ShareCommand::new(session_store);
            command.execute(share_args).await
        }

        Commands::Guide => {
            commands::guide::print_guide();
            ExitCode::Success
        }

        Commands::Update(update_args) if update_args.rollback => {
            commands::update::execute_rollback().await
        }

        Commands::Update(update_args) => match UpdateCommand::new() {
            Ok(command) => {
                if update_args.sync_model_data {
                    command.execute_sync_model_data().await
                } else {
                    command
                        .execute(update_args.force, update_args.sudo_elevated)
                        .await
                }
            }
            Err(e) => {
                eprintln!(
                    "{} Failed to initialize update command: {}",
                    style::red("Error:"),
                    e
                );
                crate::errors::exit_code_for_error(&e)
            }
        },
    };

    // Stop Ollama if aivo auto-started it during this session.
    services::ollama::stop_if_we_started();
    // Stop llama-server if aivo auto-started it for a HuggingFace run.
    services::huggingface::stop_if_we_started();
    // `process::exit` below skips `kill_on_drop`; cursor-agent ignores stdin EOF.
    services::acp_client::kill_all_acp_children().await;

    // Nudge after the command (TUI screen restored) if a newer release is cached.
    if !is_update_cmd {
        services::update_check::maybe_print_notice(crate::version::VERSION);
    }

    process::exit(exit_code.code());
}

/// Snapshot of Bundle aliases for `rewrite_cli_args`. Errors during load are
/// swallowed — a startup-time alias read failure must never stop the user
/// from invoking a non-bundle command.
async fn load_bundle_index(
    session_store: &SessionStore,
) -> std::collections::HashMap<String, BundleAlias> {
    session_store
        .list_alias_values()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, v)| match v {
            AliasValue::Bundle(b) => Some((k, b)),
            AliasValue::Model(_) => None,
        })
        .collect()
}

fn unknown_top_level_recovery(name: &str) -> [String; 2] {
    [
        format!("one-shot:  aivo -p {name}"),
        "TUI:       aivo code".to_string(),
    ]
}

fn print_unknown_top_level(name: &str) {
    eprintln!("{} unknown command `{name}`", style::red("Error:"));
    for line in unknown_top_level_recovery(name) {
        eprintln!("  {}", style::dim(line));
    }
}

/// Prints help information
fn print_help() {
    println!(
        "{} {} — CLI for AI coding assistants",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION)),
    );
    println!();
    println!("{} aivo <command> [options]", style::bold("Usage:"));
    println!();

    println!("{}", style::bold("Get started:"));
    let print_start = |cmd: &str, desc: &str| {
        println!("  {}  {}", style::cyan(format!("{:<12}", cmd)), desc);
    };
    print_start("aivo code", "built-in agent");
    print_start(
        "aivo claude",
        "launch Claude Code (same for codex/gemini/…)",
    );
    print_start("aivo run", "pick a tool");
    println!();

    println!("{}", style::bold("Commands:"));
    let print_cmd = |name: &str, desc: &str| {
        println!("  {}  {}", style::cyan(format!("{:<8}", name)), desc);
    };
    print_cmd("run", "Launch an AI tool, or open the tool picker");
    print_cmd("keys", "Manage API keys");
    print_cmd(
        "account",
        "Manage your account (info, usage, login, logout)",
    );
    print_cmd("code", "Start the interactive coding agent");
    print_cmd("models", "List available models from the active provider");
    print_cmd("serve", "Start a local OpenAI-compatible API server");
    print_cmd("alias", "Create, list, or remove model aliases");
    print_cmd("hf", "Manage cached HuggingFace GGUF files");
    print_cmd("logs", "Show recent local logs from code, run, and serve");
    print_cmd("stats", "Show usage statistics");
    print_cmd("plugins", "Install, list, or remove plugins");
    print_cmd("update", "Update to the latest version");
    print_cmd("guide", "Print the built-in usage guide");
    println!();

    println!("{}", style::bold("Shortcuts:"));
    let shortcuts: &[(&str, &str, &str)] = &[
        ("use", "keys use", "aivo keys use --help"),
        ("ping", "keys ping", "aivo keys ping --help"),
        ("share", "logs share", "aivo logs share --help"),
        (
            "\"text\"",
            "code -p",
            "one-shot reply (must contain a space)",
        ),
        ("hf:/url", "code <ref>", "open code with a local HF model"),
        (
            "key::model",
            "-k key -m model",
            "pick key + model in one token",
        ),
        (
            "<tool>",
            "run <tool>",
            "claude/codex/gemini/opencode/pi/grok",
        ),
    ];
    let alias_width = shortcuts.iter().map(|(a, _, _)| a.len()).max().unwrap_or(0);
    let expansion_width = shortcuts.iter().map(|(_, e, _)| e.len()).max().unwrap_or(0);
    for (alias, expansion, hint) in shortcuts {
        let alias_col = style::cyan(format!("{alias:<alias_width$}"));
        let hint_col = style::dim(format!("({hint})"));
        println!("  {alias_col}  {expansion:<expansion_width$}  {hint_col}");
    }
    println!();

    println!("{}", style::bold("Examples:"));
    for cmd in [
        "aivo code",
        "aivo \"tell me a short story\"",
        "aivo pi -k openrouter",
        "git diff | aivo \"summarize changes\"",
        "aivo hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ] {
        println!("  {}", style::dim(cmd));
    }
    println!();

    println!("{}", style::bold("Options:"));
    let print_opt = |flag: &str, desc: &str| {
        println!("  {}  {}", style::bold(format!("{:<13}", flag)), desc);
    };
    print_opt("-h, --help", "Display help information");
    print_opt("--help-json", "Dump the full command tree as JSON");
    print_opt("-v, --version", "Display the current version");
}

/// Prints the active selection (key, tool, model) at the bottom of help output.
///
/// One config load resolves both the last-selection record and the key's
/// display name (no PBKDF2). Skips the stale-record self-healing that
/// `get_last_selection` does — that path will heal on the next command
/// that actually touches selection state.
async fn print_active_selection(session_store: &SessionStore) {
    let Some(config) = session_store.load().await.ok() else {
        return;
    };

    // Prefer the full last_selection (key + model). Fall back to the bare
    // active_key_id so this footer agrees with what `aivo keys` marks active —
    // that list also falls back to active_key_id when last_selection is empty
    // (e.g. right after `aivo keys add`, which clears last_selection).
    let selection = config
        .last_selection
        .as_ref()
        .and_then(|sel| {
            // Treat a missing or moved key as "no active selection" — mirrors
            // `get_last_selection`'s stale check without rewriting the config.
            let key = config.api_keys.iter().find(|k| k.id == sel.key_id)?;
            (key.base_url == sel.base_url).then_some((key, sel.model.as_deref()))
        })
        .or_else(|| {
            let id = config.active_key_id.as_deref()?;
            config
                .api_keys
                .iter()
                .find(|k| k.id == id)
                .map(|k| (k, None))
        });
    let Some((key, model)) = selection else {
        return;
    };

    let model_display = commands::models::model_display_label(model);
    // HF models bypass the API key entirely (local llama-server with a
    // synthetic loopback key). Showing `key  hf:...` implies a coupling that
    // doesn't exist at runtime, so swap the line out for an HF-specific one.
    let model_is_hf = model.is_some_and(services::huggingface::is_huggingface_ref);

    println!();
    println!("{}", style::bold("Active key:"));
    if model_is_hf {
        println!(
            "  {} {}  {}  {}",
            style::bullet_symbol(),
            style::dim(model_display),
            style::dim("(local, no API key)"),
            style::dim("(change with: aivo use)"),
        );
    } else {
        println!(
            "  {} {}  {}  {}",
            style::bullet_symbol(),
            key.display_name(),
            style::dim(model_display),
            style::dim("(change with: aivo use)"),
        );
    }
    println!("  {}", style::dim("ready — try aivo code"));
}

/// Dump the entire CLI command tree (commands, flags, descriptions, env hints)
/// as JSON on stdout. Intended for AI agents / tooling that needs reliable
/// machine-readable command discovery — human help text is easy to misparse.
fn print_help_json() {
    let cmd = Cli::command();
    let tree = serialize_command(&cmd);
    // Enrich each plugin entry with its cached manifest (version / roles / caps).
    let registry = plugin::registry::load().plugins;
    let plugins = plugin::installed_plugin_names()
        .into_iter()
        .map(|name| {
            let mut obj = Map::new();
            obj.insert("name".into(), json!(name));
            obj.insert("binary".into(), json!(format!("aivo-{name}")));
            if let Some(m) = registry.get(&name).and_then(|r| r.manifest.as_ref()) {
                obj.insert("version".into(), json!(m.version));
                if !m.roles.is_empty() {
                    obj.insert("roles".into(), json!(m.roles));
                }
                if !m.capabilities.is_empty() {
                    obj.insert("capabilities".into(), json!(m.capabilities));
                }
            }
            Value::Object(obj)
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "name": "aivo",
        "version": version::VERSION,
        "shortcuts": [
            { "alias": "use", "expands_to": ["keys", "use"] },
            { "alias": "ping", "expands_to": ["keys", "ping"] },
            { "alias": "share", "expands_to": ["logs", "share"] },
            { "alias": "code mcp", "expands_to": ["code mcp"], "note": "MCP servers are managed under `aivo code`; resolves to the hidden command named \"code mcp\"" },
            { "alias": "code skills", "expands_to": ["code skills"], "note": "Skills are managed under `aivo code`; resolves to the hidden command named \"code skills\"" },
            { "alias": "-p", "expands_to": ["code", "-p"] },
            { "alias": "<text>", "expands_to": ["code", "-p", "<text>"], "note": "Top-level arg that can't be a command name (whitespace, uppercase, punctuation, non-ASCII) → one-shot code prompt; bare [a-z0-9-] words fall through as subcommands" },
            { "alias": "hf:<ref> | http(s)://<url>", "expands_to": ["code", "<ref>"], "note": "Top-level HF/URL arg → code with that model" },
            { "alias": "claude", "expands_to": ["run", "claude"] },
            { "alias": "codex", "expands_to": ["run", "codex"] },
            { "alias": "codex-app", "expands_to": ["run", "codex-app"] },
            { "alias": "gemini", "expands_to": ["run", "gemini"] },
            { "alias": "opencode", "expands_to": ["run", "opencode"] },
            { "alias": "pi", "expands_to": ["run", "pi"] },
            { "alias": "grok", "expands_to": ["run", "grok"] }
        ],
        "environment": [
            { "name": "AIVO_REDUCE_MOTION", "desc": "Disable code TUI motion effects (=1)" },
            { "name": "AIVO_PREVIEW", "desc": "Force-disable (=0) or force-enable (=1) terminal image preview" },
            { "name": "AIVO_CODE_DISABLE_MOUSE", "desc": "Disable mouse capture in code TUI (=1; auto-off under Termux, =0 re-enables; legacy AIVO_CHAT_DISABLE_MOUSE still honored)" },
            { "name": "AIVO_CODE_SCROLL_SPEED", "desc": "Lines scrolled per wheel tick in code TUI (default 3; legacy AIVO_CHAT_SCROLL_SPEED still honored)" },
            { "name": "AIVO_CODE_SWIPE_SCROLL", "desc": "Up/Down arrows scroll the transcript instead of draft history (=1; auto-on under Termux for touch swipes, =0 disables; legacy AIVO_CHAT_SWIPE_SCROLL still honored)" },
            { "name": "AIVO_PATH", "desc": "Override the install path detected by `aivo update`" },
            { "name": "AIVO_SHARE_BASE_URL", "desc": "Override the public tunnel endpoint used by `aivo logs share`" },
            { "name": "AIVO_DEBUG", "desc": "Surface upstream HTTP request/response detail in some flows (=1)" }
        ],
        "plugins": plugins,
        "tree": tree,
    });
    match serde_json::to_string_pretty(&payload) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialize help: {e}"),
    }
}

/// Recursively serialize a clap `Command` into JSON.
fn serialize_command(cmd: &Command) -> Value {
    let mut obj = Map::new();
    obj.insert("name".into(), json!(cmd.get_name()));
    if let Some(about) = cmd.get_about() {
        obj.insert("about".into(), json!(about.to_string()));
    }
    if let Some(long) = cmd.get_long_about() {
        obj.insert("long_about".into(), json!(long.to_string()));
    }
    let aliases: Vec<String> = cmd
        .get_visible_aliases()
        .map(|s| s.to_string())
        .chain(cmd.get_all_aliases().map(|s| s.to_string()))
        .collect();
    if !aliases.is_empty() {
        obj.insert("aliases".into(), json!(aliases));
    }
    if cmd.is_hide_set() {
        obj.insert("hidden".into(), json!(true));
    }

    let mut args = Vec::new();
    for arg in cmd.get_arguments() {
        if arg.is_hide_set() {
            continue;
        }
        let mut a = Map::new();
        a.insert("id".into(), json!(arg.get_id().as_str()));
        if let Some(s) = arg.get_short() {
            a.insert("short".into(), json!(format!("-{s}")));
        }
        if let Some(l) = arg.get_long() {
            a.insert("long".into(), json!(format!("--{l}")));
        }
        if let Some(h) = arg.get_help().or_else(|| arg.get_long_help()) {
            a.insert("help".into(), json!(h.to_string()));
        }
        let names: Vec<String> = arg
            .get_value_names()
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        if !names.is_empty() {
            a.insert("value_names".into(), json!(names));
        }
        a.insert("takes_value".into(), json!(arg.get_action().takes_values()));
        a.insert("positional".into(), json!(arg.is_positional()));
        args.push(Value::Object(a));
    }
    if !args.is_empty() {
        obj.insert("args".into(), Value::Array(args));
    }

    let subs: Vec<Value> = cmd.get_subcommands().map(serialize_command).collect();
    if !subs.is_empty() {
        obj.insert("subcommands".into(), Value::Array(subs));
    }
    Value::Object(obj)
}

/// Prints version information
fn print_version() {
    println!(
        "{} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn code_args(argv: &[&str]) -> CodeArgs {
        match Cli::try_parse_from(argv).unwrap().command {
            Some(Commands::Code(a)) => a,
            other => panic!("expected chat args, got {other:?}"),
        }
    }

    #[test]
    fn free_text_positional_becomes_initial_tui_prompt() {
        let mut a = code_args(&["aivo", "chat", "hello world"]);
        assert_eq!(
            take_code_initial_prompt(&mut a),
            Ok(Some("hello world".to_string()))
        );
        assert_eq!(a.reference, None);
        assert_eq!(a.prompt, None);
    }

    #[test]
    fn bare_word_positional_is_rejected() {
        let mut a = code_args(&["aivo", "chat", "mpc"]);
        assert_eq!(
            take_code_initial_prompt(&mut a),
            Err(CodePositionalReject::NotACommand("mpc".to_string()))
        );
        assert_eq!(a.prompt, None);
    }

    #[test]
    fn hf_positional_stays_a_model_ref() {
        let mut a = code_args(&["aivo", "chat", "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"]);
        assert_eq!(take_code_initial_prompt(&mut a), Ok(None));
        assert_eq!(
            a.reference.as_deref(),
            Some("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        );
        assert_eq!(a.prompt, None);
    }

    #[test]
    fn key_model_positional_stays_a_model_ref_even_with_prompt() {
        // The spec lifts into the model slot, so it coexists with -p.
        let mut a = code_args(&["aivo", "chat", "alt::gpt-4o", "-p", "hello"]);
        assert_eq!(take_code_initial_prompt(&mut a), Ok(None));
        assert_eq!(a.reference.as_deref(), Some("alt::gpt-4o"));
        assert_eq!(a.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn key_only_positional_stays_a_model_ref() {
        let mut a = code_args(&["aivo", "chat", "alt::"]);
        assert_eq!(take_code_initial_prompt(&mut a), Ok(None));
        assert_eq!(a.reference.as_deref(), Some("alt::"));
    }

    #[test]
    fn positional_conflicts_with_explicit_prompt() {
        let mut a = code_args(&["aivo", "chat", "Fix the tests", "-p", "PFLAG"]);
        assert_eq!(
            take_code_initial_prompt(&mut a),
            Err(CodePositionalReject::Conflicts(
                "Fix the tests".to_string(),
                "-p/--prompt"
            ))
        );
    }

    #[test]
    fn positional_conflicts_with_resume() {
        let mut a = code_args(&["aivo", "chat", "Fix the tests", "--resume", "last"]);
        assert_eq!(
            take_code_initial_prompt(&mut a),
            Err(CodePositionalReject::Conflicts(
                "Fix the tests".to_string(),
                "--resume"
            ))
        );
    }

    #[test]
    fn bare_dash_positional_reads_from_stdin() {
        let mut a = code_args(&["aivo", "chat", "-"]);
        assert_eq!(take_code_initial_prompt(&mut a), Ok(None));
        assert_eq!(a.prompt.as_deref(), Some(""));
        assert_eq!(a.reference, None);
    }

    #[test]
    fn explicit_prompt_wins_over_bare_dash() {
        let mut a = code_args(&["aivo", "chat", "-", "-p", "PFLAG"]);
        assert_eq!(take_code_initial_prompt(&mut a), Ok(None));
        assert_eq!(a.prompt.as_deref(), Some("PFLAG"));
        assert_eq!(a.reference, None);
    }

    #[test]
    fn bare_word_is_an_invalid_subcommand() {
        let err = Cli::try_parse_from(["aivo", "hello"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(
            matches!(
                err.get(clap::error::ContextKind::InvalidSubcommand),
                Some(clap::error::ContextValue::String(s)) if s == "hello"
            ),
            "context names the rejected token"
        );
    }

    #[test]
    fn nested_invalid_subcommand_names_the_nested_token() {
        let err = Cli::try_parse_from(["aivo", "account", "frobnicate"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(
            matches!(
                err.get(clap::error::ContextKind::InvalidSubcommand),
                Some(clap::error::ContextValue::String(s)) if s == "frobnicate"
            ),
            "context names the nested token, not `account`"
        );
    }

    #[test]
    fn unknown_top_level_recovery_points_at_oneshot_and_tui() {
        let [oneshot, tui] = unknown_top_level_recovery("hello");
        assert_eq!(oneshot, "one-shot:  aivo -p hello");
        assert_eq!(tui, "TUI:       aivo code");
    }
}
