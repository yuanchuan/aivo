/**
 * Main entry point for the aivo CLI.
 * Initializes services with dependency injection and routes commands to handlers.
 */
use std::process;

use clap::Parser;

mod cli;
mod commands;
mod errors;
mod services;
mod style;
mod tui;
mod version;

use cli::{Cli, Commands};
use commands::{ChatCommand, KeysCommand, ModelsCommand, RunCommand, UpdateCommand};
use errors::ExitCode;
use services::session_store::ApiKey;
use services::{AILauncher, EnvironmentInjector, SessionStore};

/// Known AI tool names that can be used as shortcut aliases for `run`.
const TOOL_ALIASES: &[&str] = &["claude", "codex", "gemini", "opencode"];

/// Main entry point for the CLI
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Rewrite `aivo <tool> ...` to `aivo run <tool> ...` for known tool aliases
    let raw_args: Vec<String> = std::env::args().collect();
    let args = if raw_args.len() > 1 && TOOL_ALIASES.contains(&raw_args[1].as_str()) {
        let mut rewritten = vec![raw_args[0].clone(), "run".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        Cli::parse_from(rewritten)
    } else if raw_args.len() > 1 && raw_args[1] == "use" {
        let mut rewritten = vec![raw_args[0].clone(), "keys".to_string(), "use".to_string()];
        rewritten.extend_from_slice(&raw_args[2..]);
        Cli::parse_from(rewritten)
    } else {
        Cli::parse()
    };

    // Initialize services early so we can show active key in help
    let session_store = SessionStore::new();
    let models_cache = services::ModelsCache::new();

    // Handle help and version flags at the top level
    if args.help {
        match &args.command {
            Some(Commands::Run(_)) => {
                RunCommand::print_help();
            }
            Some(Commands::Keys(_)) => {
                KeysCommand::print_help();
            }
            Some(Commands::Chat(_)) => {
                ChatCommand::print_help();
            }
            Some(Commands::Models(_)) => {
                ModelsCommand::print_help();
            }
            Some(Commands::Update(_)) => {
                UpdateCommand::print_help();
            }
            None => {
                print_help();
                print_active_key(&session_store).await;
            }
        }
        process::exit(0);
    }

    if args.version {
        print_version();
        process::exit(0);
    }

    // Get the command or show help if none provided
    let command = match args.command {
        Some(cmd) => cmd,
        None => {
            print_help();
            print_active_key(&session_store).await;
            process::exit(0);
        }
    };

    // Route to command handler
    let exit_code = match command {
        Commands::Keys(keys_args) => {
            let command = KeysCommand::new(session_store);
            let action = keys_args.action.as_deref();
            let args: Vec<_> = keys_args.args.iter().map(|s| s.as_str()).collect();
            command.execute(action, Some(&args)).await
        }

        Commands::Chat(chat_args) => {
            // Handle --key flag: resolve key for temporary use (not persisted)
            let key_override = if let Some(ref key_id_or_name) = chat_args.key {
                match session_store
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await
                {
                    Ok(key) => Some(key),
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        process::exit(ExitCode::UserError.code());
                    }
                }
            } else {
                match resolve_active_key_or_prompt(&session_store).await {
                    Some(key) => Some(key),
                    None => process::exit(ExitCode::AuthError.code()),
                }
            };
            let command = ChatCommand::new(session_store, models_cache.clone());
            command.execute(chat_args.model, key_override).await
        }

        Commands::Run(run_args) => {
            let env_injector = EnvironmentInjector::new();
            let ai_launcher =
                AILauncher::new(session_store.clone(), env_injector, models_cache.clone());
            let command = RunCommand::new(ai_launcher, models_cache);

            // Re-extract aivo flags from passthrough args that clap's trailing_var_arg
            // may have swallowed (e.g. `aivo run claude --agent-name foo --model opus`
            // puts --model into args instead of parsing it as an aivo flag).
            let mut model = run_args.model;
            let mut key_flag = run_args.key;
            let mut debug = run_args.debug;
            let mut env_strings = run_args.envs;
            let mut remaining_args = Vec::new();
            let mut i = 0;
            let args = &run_args.args;
            while i < args.len() {
                let arg = &args[i];
                if let Some(value) = arg.strip_prefix("--model=") {
                    if !value.is_empty() && model.is_none() {
                        model = Some(value.to_string());
                    } else {
                        remaining_args.push(arg.clone());
                    }
                } else if (arg == "--model" || arg == "-m") && model.is_none() {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        model = Some(args[i + 1].clone());
                        i += 1;
                    } else {
                        // --model with no value in passthrough args → trigger picker
                        model = Some(String::new());
                    }
                } else if let Some(value) = arg.strip_prefix("--key=") {
                    if !value.is_empty() && key_flag.is_none() {
                        key_flag = Some(value.to_string());
                    } else {
                        remaining_args.push(arg.clone());
                    }
                } else if (arg == "--key" || arg == "-k") && key_flag.is_none() {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        key_flag = Some(args[i + 1].clone());
                        i += 1;
                    } else {
                        remaining_args.push(arg.clone());
                    }
                } else if arg == "--debug" {
                    debug = true;
                } else if let Some(value) = arg
                    .strip_prefix("--env=")
                    .or_else(|| arg.strip_prefix("-e="))
                {
                    if !value.is_empty() {
                        env_strings.push(value.to_string());
                    }
                } else if (arg == "--env" || arg == "-e") && i + 1 < args.len() {
                    env_strings.push(args[i + 1].clone());
                    i += 1;
                } else {
                    remaining_args.push(arg.clone());
                }
                i += 1;
            }

            // Handle --key flag: resolve key for temporary use (not persisted)
            let key_override = if let Some(ref key_id_or_name) = key_flag {
                match session_store
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await
                {
                    Ok(key) => Some(key),
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        process::exit(ExitCode::UserError.code());
                    }
                }
            } else {
                match resolve_active_key_or_prompt(&session_store).await {
                    Some(key) => Some(key),
                    None => process::exit(ExitCode::AuthError.code()),
                }
            };

            let env = if !env_strings.is_empty() {
                let mut map = std::collections::HashMap::new();
                for env_str in &env_strings {
                    if let Some((key, value)) = env_str.split_once('=') {
                        map.insert(key.to_string(), value.to_string());
                    } else {
                        eprintln!(
                            "{} Ignoring malformed env value '{}' (expected KEY=VALUE format)",
                            style::yellow("Warning:"),
                            env_str
                        );
                    }
                }
                Some(map)
            } else {
                None
            };

            command
                .execute(
                    run_args.tool.as_deref(),
                    remaining_args,
                    debug,
                    model,
                    env,
                    key_override,
                    run_args.no_optimize,
                )
                .await
        }

        Commands::Models(models_args) => {
            let key_override = if let Some(ref key_id_or_name) = models_args.key {
                match session_store
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await
                {
                    Ok(key) => Some(key),
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        process::exit(ExitCode::UserError.code());
                    }
                }
            } else {
                match resolve_active_key_or_prompt(&session_store).await {
                    Some(key) => Some(key),
                    None => process::exit(ExitCode::AuthError.code()),
                }
            };
            let command = ModelsCommand::new(session_store, models_cache);
            command.execute(key_override, models_args.refresh).await
        }

        Commands::Update(update_args) => match UpdateCommand::new() {
            Ok(command) => command.execute(update_args.force).await,
            Err(e) => {
                eprintln!(
                    "{} Failed to initialize update command: {}",
                    style::red("Error:"),
                    e
                );
                ExitCode::UserError
            }
        },
    };

    process::exit(exit_code.code());
}

/// When no active key is set, prompts the user to select one (if keys exist)
/// or to add one (if no keys exist). Returns the selected key on success.
async fn resolve_active_key_or_prompt(session_store: &SessionStore) -> Option<ApiKey> {
    // Already have an active key — nothing to do
    if let Ok(Some(key)) = session_store.get_active_key().await {
        return Some(key);
    }

    let all_keys = match session_store.get_keys().await {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            return None;
        }
    };

    if all_keys.is_empty() {
        eprintln!("{} No API keys configured.", style::yellow("Note:"));
        eprintln!();
        eprintln!("  Run {} to add one.", style::cyan("aivo keys add"));
        return None;
    }

    // Keys exist but none is active — let the user pick
    eprintln!(
        "{} No active API key. Select one to continue:",
        style::yellow("Note:")
    );
    eprintln!();

    match commands::keys::prompt_select_key(session_store, &all_keys, "Select a key", 0).await {
        Ok(Some(key)) => {
            eprintln!();
            Some(key)
        }
        Ok(None) => {
            eprintln!("{}", style::dim("Cancelled."));
            None
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            None
        }
    }
}

/// Prints the active key info.
/// Only prints if there is an active key configured.
async fn print_active_key(session_store: &SessionStore) {
    let active_key = match session_store.get_active_key().await.ok().flatten() {
        Some(key) => key,
        None => return,
    };

    println!();
    println!("{}", style::bold("Active key:"));
    let id_padded = format!("{:<4}", active_key.id);
    println!(
        "  {} {}  {}  {}",
        style::bullet_symbol(),
        style::cyan(&id_padded),
        active_key.name,
        style::dim(&active_key.base_url)
    );
}

/// Prints help information
fn print_help() {
    println!(
        "{} {} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION)),
        style::dim("— CLI for AI coding assistants")
    );
    println!();
    println!("{} aivo <command> [options]", style::bold("Usage:"));
    println!();
    println!("{}", style::bold("Commands:"));
    println!(
        "  {}      {}",
        style::cyan("run <tool>"),
        style::dim("Launch AI tool with local API keys")
    );
    println!(
        "  {}  {}",
        style::cyan("chat [--model]"),
        style::dim("Start an interactive chat REPL")
    );
    println!(
        "  {}   {}",
        style::cyan("keys [action]"),
        style::dim("Manage API keys (list, use, rm, add, cat)")
    );
    println!(
        "  {}      {}",
        style::cyan("use [name]"),
        style::dim("Switch active API key")
    );
    println!(
        "  {}          {}",
        style::cyan("models"),
        style::dim("List available models from the active provider")
    );
    println!(
        "  {}          {}",
        style::cyan("update"),
        style::dim("Update to the latest version")
    );
    println!();
    println!(
        "{} {}",
        style::bold("Shortcuts:"),
        style::dim("aivo claude/codex/gemini/opencode")
    );
    println!();
    println!("{}", style::bold("Options:"));
    println!(
        "  {}   Display help information",
        style::dim("-h, --help   ")
    );
    println!(
        "  {}   Display the current version",
        style::dim("-v, --version")
    );
}

/// Prints version information
fn print_version() {
    println!(
        "{} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
}
