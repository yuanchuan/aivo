//! The built-in agent tools, executed locally (file I/O, search, a sandboxed
//! shell, and a read-only web fetch). Pure execution — no terminal I/O, no
//! permission prompts (the engine confirms only `is_dangerous` calls before
//! `execute`). Outputs are capped (Finding 2). glob/grep are zero-dep (std walk;
//! grep shells to rg/grep when present, else a literal-substring fallback).
//! (`skill` and `update_plan` are engine-handled, not dispatched here.)

use crate::agent::protocol::ToolSpec;
use crate::agent::subagents;
use serde_json::{Value, json};
use std::io::Read;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

mod bash;
pub use bash::*;
mod files;
pub(crate) use files::*;
mod remote;
pub use remote::*;
pub mod registry;
mod safety;
pub use safety::*;
mod search;
use search::*;
mod specs;
pub use specs::*;
mod web;
pub(crate) use web::*;

/// Max bytes / lines returned from any single tool result before truncation
/// (whichever is hit first). Borrowed from pi's bounded-output approach.
const MAX_OUTPUT: usize = 30_000;
const MAX_OUTPUT_LINES: usize = 2_000;

/// Default / hard cap on `read_file` lines when no limit is given.
const DEFAULT_READ_LIMIT: usize = 2_000;

/// Cap on bytes slurped by `read_file` so a giant log can't exhaust memory.
pub(crate) const MAX_READ_BYTES: u64 = 10 * 1024 * 1024;

/// Max paths returned from `glob`.
const GLOB_CAP: usize = 500;

/// Max entries a glob/grep-fallback walk visits — GLOB_CAP bounds matches, not the tree walked.
const WALK_VISIT_CAP: usize = 100_000;

const BASH_DEFAULT_TIMEOUT: u64 = 120;

const BASH_MAX_TIMEOUT: u64 = 600;

/// `web_fetch`: request timeout, hard byte cap on the body slurped, and the
/// ceiling on `max_chars` (so a huge value can't flood the model).
const WEB_FETCH_TIMEOUT: u64 = 30;

const WEB_FETCH_MAX_BYTES: usize = 5 * 1024 * 1024;

const WEB_FETCH_CHAR_CEIL: usize = 100_000;

/// Redirects are followed manually so each hop is SSRF-checked; cap the chain.
const WEB_FETCH_MAX_REDIRECTS: usize = 5;

/// `web_search`: default and ceiling result count requested from the gateway.
const WEB_SEARCH_DEFAULT_RESULTS: usize = 8;

const WEB_SEARCH_MAX_RESULTS: usize = 20;

/// Directories never descended into by glob/grep walks.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    ".venv",
    "dist",
    "build",
    ".next",
    "__pycache__",
];

/// Human-readable preview for the permission card: diff for edit, command for
/// bash, path+size for write. `None` for read-only tools.
pub fn preview(name: &str, args: &Value) -> Option<String> {
    match name {
        "write_file" => {
            let path = args.get("path")?.as_str()?;
            let lines = args
                .get("content")
                .and_then(|v| v.as_str())
                .map(|c| c.lines().count())
                .unwrap_or(0);
            Some(format!("{path}  ({lines} lines)"))
        }
        "edit_file" => {
            let path = args.get("path")?.as_str()?;
            let old = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let new = args
                .get("new_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mut d = format!("{path}\n");
            for l in old.lines() {
                d.push_str(&format!("  - {l}\n"));
            }
            for l in new.lines() {
                d.push_str(&format!("  + {l}\n"));
            }
            Some(d)
        }
        "multi_edit" => {
            let path = args.get("path")?.as_str()?;
            let n = args
                .get("edits")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let plural = if n == 1 { "edit" } else { "edits" };
            Some(format!("{path}  ({n} {plural})"))
        }
        "apply_patch" => {
            let paths = crate::agent::apply_patch::target_paths(args.get("input")?.as_str()?);
            if paths.is_empty() {
                Some("apply_patch".to_string())
            } else {
                Some(format!("patch: {}", paths.join(", ")))
            }
        }
        "run_bash" => Some(args.get("command")?.as_str()?.to_string()),
        _ => None,
    }
}

/// Execute a tool. Returns Ok(result) or Err(message); errors are fed back to
/// the model as a tool result so it can self-correct (they don't abort the loop).
pub async fn execute(name: &str, args: &Value, cwd: &Path) -> Result<String, String> {
    // Normalize known aliases (e.g. "shell" / "bash" → "run_bash") before
    // dispatching, so external APIs that use different tool names still work.
    let name = match subagents::normalize_tool_name(name) {
        Some(n) => n,
        None => name,
    };
    // The OS sandbox confines only the shell; refuse in-process edits here too.
    refuse_writes_in_read_only(name)?;
    match name {
        "read_file" => read_file(args, cwd),
        "list_dir" => list_dir(args, cwd),
        "glob" => glob(args, cwd).await,
        "grep" => grep(args, cwd).await,
        "write_file" => write_file(args, cwd),
        "edit_file" => edit_file(args, cwd),
        "multi_edit" => multi_edit(args, cwd),
        "apply_patch" => crate::agent::apply_patch::apply(arg_str(args, "input")?, cwd),
        "web_fetch" => web_fetch(args).await,
        "web_search" => web_search(args).await,
        "run_bash" => run_bash(args, cwd).await,
        other => Err(format!(
            "unknown tool `{other}` (available: {})",
            registry::workspace_exec_names()
        )),
    }
}

/// Execute a write tool with confinement waived (approved escalation only).
/// The read-only profile still refuses.
pub async fn execute_write_unconfined(
    name: &str,
    args: &Value,
    cwd: &Path,
) -> Result<String, String> {
    let name = subagents::normalize_tool_name(name).unwrap_or(name);
    refuse_writes_in_read_only(name)?;
    match name {
        "write_file" => write_file_confined(args, cwd, false),
        "edit_file" => edit_file_confined(args, cwd, false),
        "multi_edit" => multi_edit_confined(args, cwd, false),
        "apply_patch" => {
            crate::agent::apply_patch::apply_confined(arg_str(args, "input")?, cwd, false)
        }
        other => Err(format!("`{other}` is not a write tool")),
    }
}

fn refuse_writes_in_read_only(name: &str) -> Result<(), String> {
    if crate::agent::file_tracker::is_write_tool(name)
        && crate::agent::sandbox::current_profile()
            == crate::agent::sandbox::SandboxProfile::ReadOnly
    {
        return Err(format!(
            "{name}: refused — the read-only sandbox profile is active, so no files may be written."
        ));
    }
    Ok(())
}

// --- argument helpers ---

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing required string argument `{key}`"))
}

fn arg_str_opt<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

pub(crate) fn resolve(cwd: &Path, p: &str) -> PathBuf {
    // Expand `~` to $HOME — the tools advertise it and run_bash's shell expands it;
    // else `list_dir ~/.ssh` ENOENTs on `cwd/~/.ssh` and the model reads it as sandboxed.
    if (p == "~" || p.starts_with("~/") || (cfg!(windows) && p.starts_with("~\\")))
        && let Some(home) = crate::services::system_env::home_dir()
    {
        let rest = p[1..].trim_start_matches(['/', '\\']);
        return if rest.is_empty() {
            home
        } else {
            home.join(rest)
        };
    }
    let pb = Path::new(p);
    if pb.is_absolute() {
        pb.to_path_buf()
    } else {
        cwd.join(pb)
    }
}

/// Effective `grep` context lines (clamped). Shared with `read_dedupe_key`.
fn grep_context(args: &Value) -> u64 {
    arg_u64(args, "context").unwrap_or(0).min(100)
}

/// Effective `web_fetch` char cap (default + ceiling). Shared with `read_dedupe_key`.
fn web_fetch_max_chars(args: &Value) -> usize {
    arg_u64(args, "max_chars")
        .map(|n| n as usize)
        .unwrap_or(MAX_OUTPUT)
        .min(WEB_FETCH_CHAR_CEIL)
}

/// Truncate to ≤ `max` bytes on a UTF-8 boundary; returns whether anything was cut.
fn truncate_on_char_boundary(s: &mut String, max: usize) -> bool {
    if s.len() <= max {
        return false;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    true
}

/// Cap keeping the HEAD — for file reads / listings, where the start matters.
fn cap_head(s: String) -> String {
    cap_head_resume(s, |_| "narrow the request to see the rest".to_string())
}

/// Like [`cap_head`], but the caller turns the count of whole lines kept into a
/// runnable continuation — the notice hands over the next call, not arithmetic.
fn cap_head_resume(s: String, resume: impl FnOnce(usize) -> String) -> String {
    let total_lines = s.lines().count();
    let total_bytes = s.len();
    let mut truncated = total_lines > MAX_OUTPUT_LINES;
    let mut out = if truncated {
        s.lines()
            .take(MAX_OUTPUT_LINES)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        s
    };
    let byte_cut = truncate_on_char_boundary(&mut out, MAX_OUTPUT);
    truncated |= byte_cut;
    if truncated {
        let kept_lines = out.lines().count();
        let kept_bytes = out.len();
        // A byte cut leaves a partial last line; drop it so a resume re-reads
        // that line instead of skipping its tail.
        let whole_lines = kept_lines - usize::from(byte_cut && !out.ends_with('\n'));
        out.push_str(&format!(
            "\n… (output truncated: showing the first {kept_lines} of {total_lines} lines, \
{kept_bytes} of {total_bytes} bytes — {})",
            resume(whole_lines)
        ));
    }
    out
}

/// Cap keeping the TAIL — for shell output, where the error/result is at the end
/// (pi's truncateTail). Dropping the head would hide the very thing you need.
fn cap_tail(s: String) -> String {
    cap_tail_with(s, MAX_OUTPUT, MAX_OUTPUT_LINES)
}

/// Best-effort spill of untruncated shell output to a temp log.
fn spill_full_output(out: &str) -> Option<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("aivo-bash-{}-{id}.log", std::process::id()));
    std::fs::write(&path, out).ok()?;
    Some(path)
}

/// [`cap_tail`] with explicit limits, so background-job log tails can keep a smaller
/// window than foreground `run_bash`.
pub(crate) fn cap_tail_with(s: String, max_bytes: usize, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let total_lines = lines.len();
    let total_bytes = s.len();
    let start = lines.len().saturating_sub(max_lines);
    let mut out = lines[start..].join("\n");
    let mut truncated = start > 0;
    if out.len() > max_bytes {
        let mut from = out.len() - max_bytes;
        while !out.is_char_boundary(from) {
            from += 1;
        }
        out = out[from..].to_string();
        truncated = true;
    }
    if truncated {
        let kept_lines = out.lines().count();
        let kept_bytes = out.len();
        out = format!(
            "… (earlier output truncated: showing the last {kept_lines} of {total_lines} lines, \
{kept_bytes} of {total_bytes} bytes)\n{out}"
        );
    }
    out
}

// --- read-only tools ---

/// Refuse anything but a regular file: reading a FIFO or device (e.g. /dev/tty)
/// blocks forever and wedges the single runtime thread. Stats before open — a
/// FIFO with no writer blocks at open(), not read().
pub(crate) fn regular_file_metadata(full: &Path) -> Result<std::fs::Metadata, String> {
    let meta = std::fs::metadata(full).map_err(|e| e.to_string())?;
    if meta.is_dir() {
        return Err("is a directory (use list_dir)".to_string());
    }
    if !meta.is_file() {
        return Err("not a regular file (fifo/device/socket)".to_string());
    }
    Ok(meta)
}

#[cfg(test)]
mod tests;
