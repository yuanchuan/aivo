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
    let pattern = input
        .get("pattern")
        .and_then(|p| p.as_str())
        .unwrap_or("*");

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");

    let base = cwd.join(path);
    let mut results = Vec::new();

    walkdir(&base, pattern, &mut results)?;

    Ok(Value::Array(results.into_iter().map(Value::String).collect()))
}

fn walkdir(base: &PathBuf, pattern: &str, results: &mut Vec<String>) -> Result<()> {
    use std::fs;

    if !base.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip hidden files
        if name.starts_with('.') {
            continue;
        }

        let relative = path.strip_prefix(base)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        if matches_glob(name, pattern) {
            results.push(relative);
        }

        if path.is_dir() {
            // Recurse for **
            if pattern.starts_with("**") || pattern.contains("/**") {
                walkdir(&path, pattern, results)?;
            }
        }
    }

    Ok(())
}

fn matches_glob(name: &str, pattern: &str) -> bool {
    // Simple glob matching
    if pattern == "*" {
        return true;
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{}", ext));
    }
    if pattern.ends_with("*") {
        let prefix = pattern.trim_end_matches('*');
        return name.starts_with(prefix);
    }
    name == pattern
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_glob_simple() {
        let input = serde_json::json!({"pattern": "*.rs"});
        let cwd = PathBuf::from(".");
        let result = execute_glob(&input, &cwd).await;
        assert!(result.is_ok());
        let files = result.unwrap();
        assert!(files.is_array());
    }
}
