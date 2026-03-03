//! Local tool execution for Copilot Router.
//!
//! In smart mode, intercepts read-only tool calls (glob, ls, read_file, grep)
//! and executes them locally without forwarding to Copilot.

use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Execute a local tool (glob/ls/read_file/grep) without forwarding to Copilot.
/// Returns the tool output as a JSON value on success.
pub async fn execute_local_tool(tool_name: &str, tool_input: &Value, cwd: &Path) -> Result<Value> {
    match tool_name {
        "glob" => execute_glob(tool_input, cwd).await,
        "ls" => execute_ls(tool_input, cwd).await,
        "read_file" => execute_read_file(tool_input, cwd).await,
        "grep" => execute_grep(tool_input, cwd).await,
        _ => Err(anyhow::anyhow!("Unsupported local tool: {}", tool_name)),
    }
}

// ---------------------------------------------------------------------------
// Path safety helpers
// ---------------------------------------------------------------------------

/// Normalize a path by resolving `.` and `..` components without requiring
/// the path to exist (unlike `canonicalize`).
fn normalize_path(path: &Path) -> PathBuf {
    let mut components: Vec<std::path::Component> = Vec::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Join `cwd` with `path` and verify the result stays inside `cwd`.
/// Resolves symlinks when the path exists to prevent symlink escapes.
fn safe_join(cwd: &Path, path: &str) -> Result<PathBuf> {
    let joined = normalize_path(&cwd.join(path));
    let cwd_norm = normalize_path(cwd);

    // If path exists, canonicalize both to resolve symlinks
    if joined.exists() {
        let canonical = joined
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Failed to resolve path: {}", e))?;
        let cwd_canonical = cwd
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Failed to resolve working directory: {}", e))?;
        if !canonical.starts_with(&cwd_canonical) {
            return Err(anyhow::anyhow!(
                "Access denied: path escapes working directory"
            ));
        }
        return Ok(canonical);
    }

    // For non-existent paths, check normalized form
    if !joined.starts_with(&cwd_norm) {
        return Err(anyhow::anyhow!(
            "Access denied: path escapes working directory"
        ));
    }

    Ok(joined)
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB
const MAX_FILES_TO_SCAN: usize = 10_000; // Prevent memory exhaustion

async fn execute_glob(input: &Value, cwd: &Path) -> Result<Value> {
    let pattern = input.get("pattern").and_then(|p| p.as_str()).unwrap_or("*");

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");

    let base = safe_join(cwd, path)?;
    let pattern = pattern.to_string();
    let results = tokio::task::spawn_blocking(move || {
        let mut results = Vec::new();
        walkdir(&base, &base, &pattern, &mut results)?;
        Ok::<Vec<String>, anyhow::Error>(results)
    })
    .await??;

    Ok(Value::Array(
        results.into_iter().map(Value::String).collect(),
    ))
}

/// Recursively walk `base`, recording paths relative to `root` that match `pattern`.
fn walkdir(root: &Path, base: &Path, pattern: &str, results: &mut Vec<String>) -> Result<()> {
    use std::fs;

    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip hidden files/dirs
        if name.starts_with('.') {
            continue;
        }

        let relative = path
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        if path.is_dir() {
            // Always recurse for ** patterns or when pattern could match inside
            if pattern.starts_with("**") || pattern.contains("/") {
                walkdir(root, &path, pattern, results)?;
            }
        } else if matches_glob(&relative, pattern) {
            results.push(relative);
        }
    }

    Ok(())
}

fn matches_glob(path: &str, pattern: &str) -> bool {
    if pattern == "*" || pattern == "**" {
        return true;
    }
    // Handle **/*.ext or **/name
    if let Some(suffix_pat) = pattern.strip_prefix("**/") {
        let filename = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path);
        return matches_glob(filename, suffix_pat);
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return path.ends_with(&format!(".{}", ext));
    }
    if pattern.ends_with('*') {
        let prefix = pattern.trim_end_matches('*');
        return path.starts_with(prefix);
    }
    // Exact match or basename match for bare patterns
    path == pattern || Path::new(path).file_name().and_then(|n| n.to_str()) == Some(pattern)
}

async fn execute_ls(input: &Value, cwd: &Path) -> Result<Value> {
    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
    let base = safe_join(cwd, path)?;

    let mut entries = tokio::fs::read_dir(&base).await?;
    let mut files = Vec::new();
    let mut directories = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().to_string();
        // Skip hidden entries
        if name.starts_with('.') {
            continue;
        }
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

    Ok(Value::Array(
        result.into_iter().map(Value::String).collect(),
    ))
}

async fn execute_read_file(input: &Value, cwd: &Path) -> Result<Value> {
    let path = input
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing path parameter"))?;

    let limit = input.get("limit").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
    let offset = input.get("offset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;

    let base = safe_join(cwd, path)?;

    // Guard against huge files before reading
    let metadata = tokio::fs::metadata(&base).await?;
    if metadata.len() > MAX_FILE_SIZE {
        return Err(anyhow::anyhow!(
            "File too large: {} bytes (max {})",
            metadata.len(),
            MAX_FILE_SIZE
        ));
    }

    let content = tokio::fs::read_to_string(&base).await?;

    // Apply offset/limit if specified
    let lines: Vec<&str> = content.lines().collect();
    let selected = if limit > 0 {
        lines
            .iter()
            .skip(offset)
            .take(limit)
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
    } else if offset > 0 {
        lines
            .iter()
            .skip(offset)
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        content
    };

    Ok(Value::String(selected))
}

async fn execute_grep(input: &Value, cwd: &Path) -> Result<Value> {
    let pattern = input
        .get("pattern")
        .and_then(|p| p.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing pattern parameter"))?;

    let path = input.get("path").and_then(|p| p.as_str()).unwrap_or(".");
    let base = safe_join(cwd, path)?;

    let case_sensitive = input
        .get("case_sensitive")
        .and_then(|c| c.as_bool())
        .unwrap_or(true);

    // Compile regex (fall back to literal match on invalid regex)
    let re = if case_sensitive {
        Regex::new(pattern).ok()
    } else {
        Regex::new(&format!("(?i){}", pattern)).ok()
    };

    // Get files to search
    let files: Vec<PathBuf> = if base.is_file() {
        vec![base]
    } else if base.is_dir() {
        let base_clone = base.clone();
        tokio::task::spawn_blocking(move || collect_files_sync(&base_clone)).await??
    } else {
        return Ok(Value::Array(vec![]));
    };

    let mut results = Vec::new();

    for file in files {
        // Skip files that are too large
        if let Ok(meta) = tokio::fs::metadata(&file).await {
            if meta.len() > MAX_FILE_SIZE {
                continue;
            }
        }

        // Skip binary files
        let content = match tokio::fs::read_to_string(&file).await {
            Ok(c) => c,
            Err(_) => continue,
        };

        for (line_num, line) in content.lines().enumerate() {
            let matches = match &re {
                Some(re) => re.is_match(line),
                None => {
                    // Invalid regex: fall back to literal match
                    if case_sensitive {
                        line.contains(pattern)
                    } else {
                        line.to_lowercase().contains(&pattern.to_lowercase())
                    }
                }
            };

            if matches {
                let relative = file
                    .strip_prefix(cwd)
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

fn collect_files_sync(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_recursive(dir, &mut files)?;
    Ok(files)
}

fn collect_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    use std::fs;

    // Prevent unbounded recursion and memory exhaustion
    if files.len() >= MAX_FILES_TO_SCAN {
        return Err(anyhow::anyhow!(
            "Too many files to scan (max {})",
            MAX_FILES_TO_SCAN
        ));
    }

    for entry in match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(()),
    } {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if name.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            collect_files_recursive(&path, files)?;
        } else {
            files.push(path);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_temp_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn make_temp_dir(dir: &TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    // --- safe_join / path traversal ---

    #[test]
    fn test_path_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let result = safe_join(dir.path(), "../../etc/passwd");
        assert!(result.is_err(), "traversal should be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Access denied"), "error message: {}", msg);
    }

    #[test]
    fn test_safe_join_valid_path() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, "hello.rs", "fn main() {}");
        let result = safe_join(dir.path(), "hello.rs");
        assert!(result.is_ok());
    }

    // --- glob ---

    #[tokio::test]
    async fn test_glob_finds_rs_files() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, "foo.rs", "");
        make_temp_file(&dir, "bar.rs", "");
        make_temp_file(&dir, "baz.txt", "");

        let input = serde_json::json!({"pattern": "*.rs", "path": "."});
        let result = execute_glob(&input, dir.path()).await.unwrap();
        let files: Vec<_> = result
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        assert!(
            files.iter().any(|f| f.contains("foo.rs")),
            "files: {:?}",
            files
        );
        assert!(
            files.iter().any(|f| f.contains("bar.rs")),
            "files: {:?}",
            files
        );
        assert!(
            !files.iter().any(|f| f.contains("baz.txt")),
            "files: {:?}",
            files
        );
    }

    #[tokio::test]
    async fn test_glob_pattern_star_star() {
        let dir = tempfile::tempdir().unwrap();
        let sub = make_temp_dir(&dir, "sub");
        std::fs::File::create(sub.join("nested.rs")).unwrap();
        make_temp_file(&dir, "top.rs", "");

        let input = serde_json::json!({"pattern": "**/*.rs", "path": "."});
        let result = execute_glob(&input, dir.path()).await.unwrap();
        let files: Vec<_> = result
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        assert!(
            files.iter().any(|f| f.contains("nested.rs")),
            "should find nested .rs file, got: {:?}",
            files
        );
    }

    // --- read_file ---

    #[tokio::test]
    async fn test_read_file_content() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, "hello.txt", "line1\nline2\nline3\n");

        let input = serde_json::json!({"path": "hello.txt"});
        let result = execute_read_file(&input, dir.path()).await.unwrap();
        let content = result.as_str().unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line3"));
    }

    #[tokio::test]
    async fn test_read_file_offset_limit() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, "data.txt", "a\nb\nc\nd\ne\n");

        // offset=1, limit=2 → lines b, c
        let input = serde_json::json!({"path": "data.txt", "offset": 1, "limit": 2});
        let result = execute_read_file(&input, dir.path()).await.unwrap();
        let content = result.as_str().unwrap();
        assert_eq!(content, "b\nc");
    }

    // --- ls ---

    #[tokio::test]
    async fn test_ls_excludes_hidden() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, ".hidden", "secret");
        make_temp_file(&dir, "visible.txt", "hello");

        let input = serde_json::json!({"path": "."});
        let result = execute_ls(&input, dir.path()).await.unwrap();
        let entries: Vec<_> = result
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        assert!(
            entries.contains(&"visible.txt".to_string()),
            "entries: {:?}",
            entries
        );
        assert!(
            !entries.iter().any(|e| e.starts_with('.')),
            "entries: {:?}",
            entries
        );
    }

    // --- grep ---

    #[tokio::test]
    async fn test_grep_regex_match() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(
            &dir,
            "code.rs",
            "fn main() {}\npub fn helper() {}\nlet x = 1;\n",
        );

        let input = serde_json::json!({"pattern": r"fn\s+\w+", "path": "code.rs"});
        let result = execute_grep(&input, dir.path()).await.unwrap();
        let matches = result.as_array().unwrap();
        assert!(!matches.is_empty(), "should match fn declarations");
        // Should find both fn main and fn helper
        assert_eq!(matches.len(), 2, "matches: {:?}", matches);
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        make_temp_file(&dir, "readme.txt", "Hello World\nhello again\nGoodbye\n");

        let input =
            serde_json::json!({"pattern": "hello", "path": "readme.txt", "case_sensitive": false});
        let result = execute_grep(&input, dir.path()).await.unwrap();
        let matches = result.as_array().unwrap();
        assert_eq!(
            matches.len(),
            2,
            "should match Hello and hello: {:?}",
            matches
        );
    }
}
