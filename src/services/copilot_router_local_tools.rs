//! Local tool execution for Copilot Router.
//!
//! In smart mode, intercepts read-only tool calls (glob, ls, read_file, grep)
//! and executes them locally without forwarding to Copilot.

use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;

/// Execute a local tool (glob/ls/read_file/grep) without forwarding to Copilot.
/// Returns the tool result as an Anthropic tool_result block.
pub async fn execute_local_tool(
    tool_name: &str,
    tool_input: &Value,
    cwd: &PathBuf,
) -> Result<Value> {
    match tool_name {
        "glob" => execute_glob(tool_input, cwd).await,
        "ls" => execute_ls(tool_input, cwd).await,
        "read_file" => execute_read_file(tool_input, cwd).await,
        "grep" => execute_grep(tool_input, cwd).await,
        _ => Err(anyhow::anyhow!("Unsupported local tool: {}", tool_name)),
    }
}

async fn execute_glob(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // TODO: implement
    Err(anyhow::anyhow!("Not implemented"))
}

async fn execute_ls(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // TODO: implement
    Err(anyhow::anyhow!("Not implemented"))
}

async fn execute_read_file(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // TODO: implement
    Err(anyhow::anyhow!("Not implemented"))
}

async fn execute_grep(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // TODO: implement
    Err(anyhow::anyhow!("Not implemented"))
}
