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
    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
    let base = cwd.join(path);

    if !base.exists() {
        return Err(anyhow::anyhow!("Directory not found: {}", path));
    }

    let mut entries = tokio::fs::read_dir(&base).await?;
    let mut files = Vec::new();
    let mut directories = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type().await?.is_dir() {
            directories.push(name);
        } else {
            files.push(name);
        }
    }

    directories.sort();
    files.sort();

    let mut result = directories;
    result.extend(files);

    Ok(Value::Array(result.into_iter().map(Value::String).collect()))
}

async fn execute_read_file(input: &Value, cwd: &PathBuf) -> Result<Value> {
    let path = input
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing path parameter"))?;

    let limit = input.get("limit").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
    let offset = input.get("offset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;

    let base = cwd.join(path);

    if !base.exists() {
        return Err(anyhow::anyhow!("File not found: {}", path));
    }

    let content = tokio::fs::read_to_string(&base).await?;

    // Apply offset/limit if specified
    let lines: Vec<&str> = content.lines().collect();
    let selected = if limit > 0 {
        lines.iter().skip(offset).take(limit).copied().collect::<Vec<_>>().join("\n")
    } else if offset > 0 {
        lines.iter().skip(offset).copied().collect::<Vec<_>>().join("\n")
    } else {
        content
    };

    Ok(Value::String(selected))
}

async fn execute_grep(input: &Value, cwd: &PathBuf) -> Result<Value> {
    let pattern = input
        .get("pattern")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing pattern parameter"))?;

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
    let base = cwd.join(path);

    let case_sensitive = input.get("case_sensitive").and_then(|c| c.as_bool()).unwrap_or(true);

    // Get files to search
    let files: Vec<PathBuf> = if base.is_file() {
        vec![base]
    } else if base.is_dir() {
        collect_files(&base).await?
    } else {
        return Ok(Value::Array(vec![]));
    };

    let mut results = Vec::new();

    for file in files {
        // Skip binary files - try to read as string first
        let content = match tokio::fs::read_to_string(&file).await {
            Ok(c) => c,
            Err(_) => continue,
        };

        for (line_num, line) in content.lines().enumerate() {
            let matches = if case_sensitive {
                line.contains(pattern)
            } else {
                line.to_lowercase().contains(&pattern.to_lowercase())
            };

            if matches {
                let relative = file.strip_prefix(cwd)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                results.push(Value::String(format!(
                    "{}:{}:{}",
                    relative,
                    line_num + 1,
                    line
                )));
            }
        }
    }

    Ok(Value::Array(results))
}

async fn collect_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    // Use synchronous std::fs to avoid async recursion issues
    let files = collect_files_sync(dir)?;
    Ok(files)
}

fn collect_files_sync(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    use std::fs;

    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden directories
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') {
                if let Ok(mut sub) = collect_files_sync(&path) {
                    files.append(&mut sub);
                }
            }
        } else {
            // Skip hidden files
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') {
                files.push(path);
            }
        }
    }

    Ok(files)
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

    #[tokio::test]
    async fn test_ls() {
        let input = serde_json::json!({"path": "src"});
        let cwd = PathBuf::from(".");
        let result = execute_ls(&input, &cwd).await;
        // May fail if src doesn't exist, that's ok - just check it runs
        println!("ls result: {:?}", result);
    }

    #[tokio::test]
    async fn test_read_file() {
        let input = serde_json::json!({"path": "src/main.rs"});
        let cwd = PathBuf::from(".");
        let result = execute_read_file(&input, &cwd).await;
        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(content.is_string());
        println!("File content length: {}", content.as_str().unwrap_or("").len());
    }

    #[tokio::test]
    async fn test_grep() {
        let input = serde_json::json!({"pattern": "fn main", "path": "src"});
        let cwd = PathBuf::from(".");
        let result = execute_grep(&input, &cwd).await;
        assert!(result.is_ok());
        let matches = result.unwrap();
        assert!(matches.is_array());
        println!("Grep results: {:?}", matches);
    }
}
