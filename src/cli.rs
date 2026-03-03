/**
 * CLI argument parsing and command routing.
 * Uses clap for argument parsing.
 */
use clap::builder::NonEmptyStringValueParser;
use clap::{Args, Parser, Subcommand};
use std::collections::HashMap;

fn non_empty() -> NonEmptyStringValueParser {
    NonEmptyStringValueParser::new()
}

/// The aivo CLI - unified access to AI coding assistants
#[derive(Parser, Debug)]
#[command(
    name = "aivo",
    about = "CLI tool for unified access to AI coding assistants (Claude, Codex, Gemini, OpenCode)",
    version = crate::version::VERSION,
    author = "yuanchuan",
    disable_help_flag = true,
    disable_version_flag = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Display help information
    #[arg(short, long, global = true, help = "Display help information")]
    pub help: bool,

    /// Display the current version
    #[arg(short, long, global = true, help = "Display the current version")]
    pub version: bool,
}

/// Available commands for the CLI
#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Run AI tools (claude, codex, gemini, opencode) - all args passed through
    Run(RunArgs),

    /// Manage API keys (list, use <id|name>, rm <id|name>, add, cat, edit)
    Keys(KeysArgs),

    /// Start an interactive chat REPL
    Chat(ChatArgs),

    /// List available models from the active provider
    Models(ModelsArgs),

    /// Update the CLI tool to the latest version
    Update(UpdateArgs),
}

/// Arguments for the keys command
#[derive(Args, Debug, Clone)]
pub struct KeysArgs {
    /// The action to perform (list, use, rm, add, cat)
    #[arg(
        value_name = "ACTION",
        help = "Action to perform: list, use, rm, add, cat, edit"
    )]
    pub action: Option<String>,

    /// Additional arguments for the action (e.g., key ID or name)
    #[arg(value_name = "ARGS", help = "Additional arguments for the action")]
    pub args: Vec<String>,
}

/// Arguments for the run command
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// The AI tool to run (claude, codex, gemini, opencode)
    #[arg(
        value_name = "TOOL",
        help = "AI tool to run: claude, codex, gemini, or opencode"
    )]
    pub tool: Option<String>,

    /// Specify AI model to use
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(short = 'k', long, value_name = "ID|NAME", value_parser = non_empty())]
    pub key: Option<String>,

    /// Enable debug output
    #[arg(long)]
    pub debug: bool,

    /// Enable smart routing: auto-select cheap models, cache responses, trim context
    #[arg(long)]
    pub smart: bool,

    /// Inject environment variable (KEY=VALUE)
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    pub envs: Vec<String>,

    /// Additional arguments to pass through to the AI tool
    #[arg(
        value_name = "ARGS",
        help = "Arguments to pass through to the AI tool",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub args: Vec<String>,
}

/// Arguments for the models command
#[derive(Args, Debug, Clone)]
pub struct ModelsArgs {
    /// Select API key by ID or name
    #[arg(short = 'k', long, value_name = "ID|NAME", value_parser = non_empty())]
    pub key: Option<String>,

    /// Bypass cache and fetch fresh model list from the provider
    #[arg(short = 'r', long)]
    pub refresh: bool,
}

/// Arguments for the update command
#[derive(Args, Debug, Clone)]
pub struct UpdateArgs {
    /// Force update even if installed via a package manager
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for the chat command
#[derive(Args, Debug, Clone)]
pub struct ChatArgs {
    /// Specify AI model to use (remembered across sessions)
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(short = 'k', long, value_name = "ID|NAME", value_parser = non_empty())]
    pub key: Option<String>,
}

/// Parse environment variable strings in the format KEY=VALUE
#[allow(dead_code)]
pub fn parse_env_vars(env_strings: &[String]) -> HashMap<String, String> {
    let mut env_map = HashMap::new();

    for env_str in env_strings {
        if let Some((key, value)) = env_str.split_once('=') {
            env_map.insert(key.to_string(), value.to_string());
        }
    }

    env_map
}

/// Get the list of valid commands
#[allow(dead_code)]
pub fn get_valid_commands() -> Vec<&'static str> {
    vec!["update", "keys", "run", "chat", "models", "use"]
}

/// Check if a command is a passthrough command (passes all args to underlying tool)
#[allow(dead_code)]
pub fn is_passthrough_command(command: &str) -> bool {
    command == "run"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_vars() {
        let env_strings = vec![
            "KEY1=value1".to_string(),
            "KEY2=value2".to_string(),
            "KEY3=nested=value".to_string(),
        ];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(env_map.get("KEY2"), Some(&"value2".to_string()));
        assert_eq!(env_map.get("KEY3"), Some(&"nested=value".to_string()));
    }

    #[test]
    fn test_parse_env_vars_invalid() {
        let env_strings = vec!["NO_EQUALS".to_string(), "VALID=key".to_string()];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.len(), 1);
        assert_eq!(env_map.get("VALID"), Some(&"key".to_string()));
    }

    #[test]
    fn test_get_valid_commands() {
        let commands = get_valid_commands();
        assert!(commands.contains(&"update"));
        assert!(commands.contains(&"keys"));
        assert!(commands.contains(&"run"));
    }

    #[test]
    fn test_is_passthrough_command() {
        assert!(is_passthrough_command("run"));
        assert!(!is_passthrough_command("keys"));
    }

    #[test]
    fn test_run_args_debug_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--debug"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            assert!(run_args.debug);
            assert!(!run_args.args.contains(&"--debug".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_flag() {
        // --model value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // --model=value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model=gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // -m value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-m", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // --model with no value → triggers picker (Some(""))
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_env_flag() {
        let cli =
            Cli::try_parse_from(["aivo", "run", "claude", "--env", "FOO=bar", "-e", "BAZ=qux"])
                .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.envs, vec!["FOO=bar", "BAZ=qux"]);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough() {
        // Arguments not matching aivo flags should be passed through
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--debug",
            "--",
            "--some-tool-flag",
            "value",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert!(run_args.debug);
            assert!(run_args.args.contains(&"--some-tool-flag".to_string()));
            assert!(run_args.args.contains(&"value".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough_claude_teammate_flags() {
        // Real-world usage: claude flags mixed with aivo --model flag.
        // When unknown flags appear before --model, clap's trailing_var_arg swallows
        // --model into args. main.rs re-extracts aivo flags from args at runtime.
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--agent-name",
            "senior-engineer",
            "--team-name",
            "ai-gateway-team",
            "--agent-color",
            "blue",
            "--parent-session-id",
            "df205d21-e955-421c-b2b9-5ff42c900cb6",
            "--agent-type",
            "general-purpose",
            "--dangerously-skip-permissions",
            "--model",
            "claude-opus-4-6",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            // --model gets swallowed into args by trailing_var_arg; main.rs re-extracts it
            assert!(run_args.args.contains(&"--model".to_string()));
            assert!(run_args.args.contains(&"claude-opus-4-6".to_string()));
            // All unknown flags pass through
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(run_args.args.contains(&"senior-engineer".to_string()));
            assert!(run_args.args.contains(&"--team-name".to_string()));
            assert!(run_args.args.contains(&"ai-gateway-team".to_string()));
            assert!(run_args
                .args
                .contains(&"--dangerously-skip-permissions".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_before_unknown_flags() {
        // When --model comes before unknown flags, clap parses it directly
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--model",
            "claude-opus-4-6",
            "--agent-name",
            "senior-engineer",
            "--dangerously-skip-permissions",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("claude-opus-4-6".to_string()));
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(run_args
                .args
                .contains(&"--dangerously-skip-permissions".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    /// Helper to simulate the alias rewriting done in main.rs
    fn rewrite_alias(args: &[&str]) -> Vec<String> {
        let aliases = ["claude", "codex", "gemini", "opencode"];
        let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        if raw.len() > 1 && aliases.contains(&raw[1].as_str()) {
            let mut rewritten = vec![raw[0].clone(), "run".to_string()];
            rewritten.extend_from_slice(&raw[1..]);
            rewritten
        } else {
            raw
        }
    }

    #[test]
    fn test_tool_alias_claude() {
        let args = rewrite_alias(&["aivo", "claude"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_codex_with_args() {
        let args = rewrite_alias(&["aivo", "codex", "--model", "o4-mini", "file.ts"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("codex".to_string()));
            assert_eq!(run_args.model, Some("o4-mini".to_string()));
            assert!(run_args.args.contains(&"file.ts".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_gemini_with_debug() {
        let args = rewrite_alias(&["aivo", "gemini", "--debug"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("gemini".to_string()));
            assert!(run_args.debug);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_opencode() {
        let args = rewrite_alias(&["aivo", "opencode"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("opencode".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_non_alias_not_rewritten() {
        let args = rewrite_alias(&["aivo", "keys"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        assert!(matches!(cli.command, Some(Commands::Keys(_))));
    }

    /// Helper to simulate the 'use' alias rewriting done in main.rs
    fn rewrite_use_alias(args: &[&str]) -> Vec<String> {
        let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        if raw.len() > 1 && raw[1] == "use" {
            let mut rewritten = vec![raw[0].clone(), "keys".to_string(), "use".to_string()];
            rewritten.extend_from_slice(&raw[2..]);
            rewritten
        } else {
            raw
        }
    }

    #[test]
    fn test_use_alias_with_key_name() {
        let args = rewrite_use_alias(&["aivo", "use", "my-key"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Keys(keys_args)) = cli.command {
            assert_eq!(keys_args.action.as_deref(), Some("use"));
            assert_eq!(keys_args.args, vec!["my-key"]);
        } else {
            panic!("Expected Keys command");
        }
    }

    #[test]
    fn test_use_alias_no_arg() {
        let args = rewrite_use_alias(&["aivo", "use"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Keys(keys_args)) = cli.command {
            assert_eq!(keys_args.action.as_deref(), Some("use"));
            assert!(keys_args.args.is_empty());
        } else {
            panic!("Expected Keys command");
        }
    }

    #[test]
    fn test_get_valid_commands_includes_use() {
        let commands = get_valid_commands();
        assert!(commands.contains(&"use"));
    }

    #[test]
    fn test_chat_command_no_model() {
        let cli = Cli::try_parse_from(["aivo", "chat"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, None);
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_with_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--model", "gpt-4o"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_model_no_value() {
        // --model with no value → triggers picker (Some(""))
        let cli = Cli::try_parse_from(["aivo", "chat", "--model"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_with_short_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-m", "claude-sonnet-4-5"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("claude-sonnet-4-5".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_run_args_key_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--key", "my-key"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_key_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-k", "a1b2"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("a1b2".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_key_equals_syntax() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--key=my-key"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_chat_args_key_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--key", "my-key"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_key_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "a1b2"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("a1b2".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_key_with_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "my-key", "-m", "gpt-4o"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }
}
