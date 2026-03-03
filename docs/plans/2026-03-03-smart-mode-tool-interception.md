# Smart Mode Tool Interception Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** In `--smart` mode, intercept read-only tool calls (glob, ls, read_file, grep), execute locally without forwarding to Copilot, saving request quota.

**Architecture:** Add tool interception logic in CopilotRouter's `handle_messages()` function. When a request contains tool_use for glob/ls/read_file/grep, execute locally and return result directly instead of forwarding to Copilot.

**Tech Stack:** Rust, tokio for async file operations, existing CopilotRouter structure

---

## Task 1: Understand Current Request Parsing

**Files:**
- Modify: `src/services/copilot_router.rs:70-120`

**Step 1: Read current handle_messages implementation**

Run: `cat -n src/services/copilot_router.rs | head -120`
Goal: Understand how requests are parsed and where tool calls are detected

---

## Task 2: Add Tool Execution Module

**Files:**
- Create: `src/services/copilot_router_local_tools.rs`
- Modify: `src/services/mod.rs` (export new module)

**Step 1: Create local tools module with execute function**

```rust
// src/services/copilot_router_local_tools.rs

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
    // Implementation: parse pattern, walk directory, return matching files
    todo!()
}

async fn execute_ls(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // Implementation: list directory contents
    todo!()
}

async fn execute_read_file(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // Implementation: read file content
    todo!()
}

async fn execute_grep(input: &Value, cwd: &PathBuf) -> Result<Value> {
    // Implementation: search file contents
    todo!()
}
```

**Step 2: Export module in mod.rs**

Run: `echo "pub mod copilot_router_local_tools;" >> src/services/mod.rs`

---

## Task 3: Implement glob Tool

**Files:**
- Modify: `src/services/copilot_router_local_tools.rs`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_glob_simple() {
        let input = json!({"pattern": "*.rs"});
        let cwd = PathBuf::from(".");
        let result = execute_glob(&input, &cwd).await;
        assert!(result.is_ok());
    }
}
```

Run: `cargo test copilot_router_local_tools -- --nocapture`
Expected: FAIL - function not implemented

**Step 2: Implement glob**

```rust
async fn execute_glob(input: &Value, cwd: &PathBuf) -> Result<Value> {
    let pattern = input
        .get("pattern")
        .and_then(|p| p.as_str())
        .unwrap_or("*");

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");

    let base = cwd.join(path);
    let mut results = Vec::new();

    // Simple glob implementation supporting * and **
    // For complex patterns, return empty (let Copilot handle)
    walkdir(&base, pattern, &mut results).await?;

    Ok(json!(results))
}

async fn walkdir(base: &PathBuf, pattern: &str, results: &mut Vec<String>) -> Result<()> {
    use tokio::fs;

    if !base.exists() {
        return Ok(());
    }

    let mut entries = fs::read_dir(base).await?;
    while let Some(entry) = entries.next_entry().await? {
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
                walkdir(&path, pattern, results).await?;
            }
        }
    }

    Ok(())
}

fn matches_glob(name: &str, pattern: &str) -> bool {
    // Simplified: just check if pattern matches
    // For "*.rs", check if ends with .rs
    if pattern == "*" {
        return true;
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{}", ext));
    }
    name == pattern
}
```

**Step 3: Run test**

Run: `cargo test copilot_router_local_tools -- --nocapture`
Expected: PASS

---

## Task 4: Implement ls Tool

**Files:**
- Modify: `src/services/copilot_router_local_tools.rs`

**Step 1: Write failing test**

```rust
#[tokio::test]
async fn test_ls() {
    let input = json!({"path": "src"});
    let cwd = PathBuf::from(".");
    let result = execute_ls(&input, &cwd).await;
    assert!(result.is_ok());
}
```

**Step 2: Implement ls**

```rust
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

    Ok(json!(result))
}
```

**Step 3: Run test**

Run: `cargo test copilot_router_local_tools -- --nocapture`
Expected: PASS

---

## Task 5: Implement read_file Tool

**Files:**
- Modify: `src/services/copilot_router_local_tools.rs`

**Step 1: Write failing test**

**Step 2: Implement read_file**

```rust
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

    Ok(json!(selected))
}
```

**Step 3: Run test**

---

## Task 6: Implement grep Tool

**Files:**
- Modify: `src/services/copilot_router_local_tools.rs`

**Step 1: Write failing test**

**Step 2: Implement grep**

```rust
async fn execute_grep(input: &Value, cwd: &PathBuf) -> Result<Value> {
    let pattern = input
        .get("pattern")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing pattern parameter"))?;

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
    let base = cwd.join(path);

    let is_regex = input.get("regex").and_then(|r| r.as_bool()).unwrap_or(true);
    let case_sensitive = input.get("case_sensitive").and_then(|c| c.as_bool()).unwrap_or(true);

    // Get files to search
    let files: Vec<PathBuf> = if base.is_dir() {
        // Search all files in directory
        collect_files(&base).await?
    } else {
        vec![base]
    };

    let mut results = Vec::new();

    for file in files {
        if let Ok(content) = tokio::fs::read_to_string(&file).await {
            for (line_num, line) in content.lines().enumerate() {
                let matches = if is_regex {
                    // Simple regex match (or use regex crate)
                    line.contains(pattern)
                } else {
                    if case_sensitive {
                        line.contains(pattern)
                    } else {
                        line.to_lowercase().contains(&pattern.to_lowercase())
                    }
                };

                if matches {
                    let relative = file.strip_prefix(cwd)
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default();

                    results.push(format!(
                        "{}:{}:{}",
                        relative,
                        line_num + 1,
                        line
                    ));
                }
            }
        }
    }

    Ok(json!(results))
}

async fn collect_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut entries = tokio::fs::read_dir(dir).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            if let Ok(mut sub) = collect_files(&path).await {
                files.append(&mut sub);
            }
        } else {
            files.push(path);
        }
    }

    Ok(files)
}
```

**Step 3: Run test**

---

## Task 7: Integrate Tool Interception in Router

**Files:**
- Modify: `src/services/copilot_router.rs:70-150`

**Step 1: Read current handle_messages**

Run: `cat -n src/services/copilot_router.rs | sed -n '70,150p'`

**Step 2: Add tool interception logic**

In `handle_messages()`, after parsing body but before sending to Copilot:

```rust
// Check if this is a tool call that can be executed locally
let smart_mode = std::env::var("AIVO_SMART_MODE").is_ok();

if smart_mode {
    if let Some(tool_result) = try_local_tool_execution(&body, &cwd).await {
        return Ok(tool_result);
    }
}
```

Add new function:

```rust
async fn try_local_tool_execution(body: &Value, cwd: &PathBuf) -> Option<String> {
    // Extract tool_use from messages
    let messages = body.get("messages")?.as_array()?;
    let last_msg = messages.last()?;

    let content = last_msg.get("content")?.as_array()?;

    for block in content {
        if block.get("type")?.as_str()? == "tool_use" {
            let name = block.get("name")?.as_str()?;
            let input = block.get("input")?;
            let id = block.get("id")?.as_str()?;

            // Only intercept read-only tools
            let local_tools = ["glob", "ls", "read_file", "grep"];
            if local_tools.contains(&name) {
                match execute_local_tool(name, input, cwd).await {
                    Ok(result) => {
                        // Convert to Anthropic tool_result format
                        let response = json!({
                            "type": "message",
                            "id": format!("msg_local_{}", id),
                            "role": "assistant",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": result.as_str().unwrap_or("")
                            }]
                        });
                        return Some(response.to_string());
                    }
                    Err(e) => {
                        // Return error to model
                        let response = json!({
                            "type": "message",
                            "id": format!("msg_local_{}", id),
                            "role": "assistant",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": format!("Error: {}", e)
                            }]
                        });
                        return Some(response.to_string());
                    }
                }
            }
        }
    }

    None
}
```

**Step 3: Add imports**

```rust
use crate::services::copilot_router_local_tools::execute_local_tool;
```

---

## Task 8: Test End-to-End

**Step 1: Build and install**

Run: `cargo build --release && cargo install --path .`

**Step 2: Test with smart mode**

Run: `aivo run claude --smart "list files in src"`

Expected: Should intercept glob/ls and return local results without Copilot request

---

## Task 9: Commit

```bash
git add -A
git commit -m "feat(copilot): add local tool execution in smart mode

Intercept glob/ls/read_file/grep in --smart mode and execute locally
to save Copilot request quota.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
