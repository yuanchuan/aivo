//! File tools: read/list/write and the edit family with no-match hints.

use super::*;

/// Effective `read_file` range (offset/limit aliases + defaults) — shared with
/// `read_dedupe_key` so the dedupe identity can't drift from the read.
pub(crate) fn read_file_range(args: &Value) -> (u64, u64) {
    let offset = arg_u64(args, "offset")
        .or_else(|| arg_u64(args, "start_line"))
        .unwrap_or(1)
        .max(1);
    let limit = arg_u64(args, "limit")
        .or_else(|| arg_u64(args, "end_line").map(|end| end.saturating_sub(offset - 1)))
        .unwrap_or(DEFAULT_READ_LIMIT as u64);
    (offset, limit)
}

/// Repeat-supersedable tools; checked before paying to parse a call's arguments.
pub(crate) fn is_dedupe_eligible(name: &str) -> bool {
    matches!(
        name,
        "read_file" | "list_dir" | "glob" | "grep" | "web_fetch"
    )
}

/// Canonical identity of a repeatable read-only call (`None` = ineligible).
/// Paths resolve against `cwd` and args normalize through the tools' own
/// helpers, so two calls share a key iff the tool would return the same content.
pub(crate) fn read_dedupe_key(name: &str, args: &Value, cwd: &Path) -> Option<String> {
    let path_of = |default: Option<&str>| -> Option<String> {
        let p = arg_str_opt(args, "path").or(default)?;
        Some(
            lexical_normalize(&resolve(cwd, p))
                .to_string_lossy()
                .into_owned(),
        )
    };
    match name {
        "read_file" => {
            let path = path_of(None)?;
            let (offset, limit) = read_file_range(args);
            Some(format!("read_file\u{0}{path}\u{0}{offset}\u{0}{limit}"))
        }
        "list_dir" => Some(format!("list_dir\u{0}{}", path_of(Some("."))?)),
        "glob" => {
            let pattern = arg_str_opt(args, "pattern")?;
            Some(format!("glob\u{0}{}\u{0}{pattern}", path_of(Some("."))?))
        }
        "grep" => {
            let pattern = arg_str_opt(args, "pattern")?;
            Some(format!(
                "grep\u{0}{}\u{0}{pattern}\u{0}{}",
                path_of(Some("."))?,
                grep_context(args)
            ))
        }
        "web_fetch" => {
            let url = arg_str_opt(args, "url")?;
            Some(format!(
                "web_fetch\u{0}{url}\u{0}{}",
                web_fetch_max_chars(args)
            ))
        }
        _ => None,
    }
}

pub(super) fn read_file(args: &Value, cwd: &Path) -> Result<String, String> {
    let path = arg_str(args, "path")?;
    let full = resolve(cwd, path);
    let meta = regular_file_metadata(&full).map_err(|e| format!("read {path}: {e}"))?;
    let file = std::fs::File::open(&full).map_err(|e| format!("read {path}: {e}"))?;
    // Slurp at most MAX_READ_BYTES so a multi-GB file can't OOM the process;
    // the model can page further with offset/limit or fall back to run_bash.
    let oversize = meta.len() > MAX_READ_BYTES;
    let mut bytes = Vec::new();
    file.take(MAX_READ_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {path}: {e}"))?;
    // A NUL byte means binary — line/offset semantics don't apply, and dumping
    // it would flood the model with garbage.
    if bytes.contains(&0) {
        return Err(format!("read {path}: appears to be a binary file"));
    }
    let content = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    // Accept `start_line`/`end_line` as aliases — ignoring them re-paged line 1 forever.
    let (offset, limit) = read_file_range(args);
    let (offset, limit) = (offset as usize, limit as usize);
    let start = offset - 1;
    let mut out = String::new();
    for (i, line) in lines.iter().skip(start).take(limit).enumerate() {
        out.push_str(&format!("{:>6}\t{}\n", start + i + 1, line));
    }
    // `saturating_add`: a model-supplied offset/limit near `usize::MAX` would
    // otherwise overflow `start + limit` — a panic in debug builds (where
    // overflow checks are on) — before this comparison even runs.
    let end = start.saturating_add(limit);
    if end < total {
        out.push_str(&format!(
            "… ({} more lines; continue with offset={})\n",
            total - end,
            end + 1
        ));
    }
    if oversize {
        out.push_str(
            "… (file exceeds 10 MB; only the first 10 MB was read — use run_bash for more)\n",
        );
    }
    if out.is_empty() {
        out.push_str("(empty or past end of file)");
    }
    Ok(cap_head_resume(out, |whole_lines| {
        format!(
            "continue with offset={}",
            offset.saturating_add(whole_lines)
        )
    }))
}

pub(super) fn list_dir(args: &Value, cwd: &Path) -> Result<String, String> {
    let path = arg_str_opt(args, "path").unwrap_or(".");
    let full = resolve(cwd, path);
    let rd = std::fs::read_dir(&full).map_err(|e| format!("list {path}: {e}"))?;
    let mut entries: Vec<String> = rd
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() {
                format!("{name}/")
            } else {
                name
            }
        })
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Ok("(empty directory)".to_string());
    }
    Ok(cap_head(entries.join("\n")))
}

/// Runs `program args` in `cwd` and returns combined stdout (+stderr if stdout
/// is empty). `None` only when the program can't be spawned (not installed).
pub(super) async fn run_capture(program: &str, args: &[&str], cwd: &Path) -> Option<String> {
    let output = tokio::process::Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()
        .await
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if !stderr.trim().is_empty() {
            return Some(stderr);
        }
    }
    Some(stdout)
}

// --- mutating tools ---

pub(super) fn write_file(args: &Value, cwd: &Path) -> Result<String, String> {
    write_file_confined(args, cwd, true)
}

pub(super) fn write_file_confined(
    args: &Value,
    cwd: &Path,
    confine: bool,
) -> Result<String, String> {
    let path = arg_str(args, "path")?;
    let content = arg_str(args, "content")?;
    if confine {
        confine_write_path(path, cwd)?;
    }
    let full = resolve(cwd, path);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create dir for {path}: {e}"))?;
    }
    atomic_write(&full, content).map_err(|e| format!("write {path}: {e}"))?;
    Ok(format!("wrote {path} ({} lines)", content.lines().count()))
}

/// Apply one `old`→`new` replacement to `content`. Without `replace_all` the
/// match must be unique (the safe default — an ambiguous match could edit the
/// wrong site); with it, every occurrence is replaced. Returns the new content
/// and the replacement count. `path` is only for error messages.
pub(super) fn apply_one_edit(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
    path: &str,
) -> Result<(String, usize), String> {
    if old.is_empty() {
        return Err(format!("old_string must not be empty in {path}"));
    }
    if old == new {
        return Err(format!(
            "old_string and new_string are identical in {path} (no change)"
        ));
    }
    // Match literally first. If that fails only because the file uses CRLF while
    // the tool args use LF (the common cross-platform case), retry with both
    // strings converted to the file's line ending — so the edit lands and the
    // inserted text keeps the file's existing style instead of corrupting it.
    let (crlf_old, crlf_new);
    let (old, new) =
        if content.matches(old).count() == 0 && content.contains("\r\n") && !old.contains('\r') {
            crlf_old = to_crlf(old);
            crlf_new = to_crlf(new);
            (crlf_old.as_str(), crlf_new.as_str())
        } else {
            (old, new)
        };
    // The constant head must fill failure_signature's 80-char window, so
    // per-attempt hint text can't split one flailing-edit streak into many.
    match content.matches(old).count() {
        0 => Err(format!(
            "old_string not found in {path} — the given text does not appear verbatim in the \
file's current contents{}",
            no_match_hint(content, old)
        )),
        n if n > 1 && !replace_all => Err(format!(
            "old_string matches {n} times in {path}; make it unique or set replace_all"
        )),
        n => {
            let updated = if replace_all {
                content.replace(old, new)
            } else {
                content.replacen(old, new, 1)
            };
            Ok((updated, n))
        }
    }
}

/// Normalize any line endings in `s` to CRLF (used to match/insert against a
/// CRLF file): collapse existing CRLF to LF first so no `\r\r\n` is produced.
pub(super) fn to_crlf(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\n', "\r\n")
}

/// Max lines / bytes of file text echoed back in a no-match hint.
pub(super) const HINT_SNIPPET_LINES: usize = 8;

pub(super) const HINT_SNIPPET_BYTES: usize = 700;

/// Past this many file lines the collapse+window scan is skipped (cost cap).
pub(super) const HINT_SCAN_MAX_LINES: usize = 5_000;

/// Diagnose a failed exact match with the file's exact text at the closest
/// region, so the model can re-anchor instead of retrying blind.
pub(super) fn no_match_hint(content: &str, old: &str) -> String {
    // Most common miss: `old_string` pasted with read_file's `NNN\t` prefixes.
    if let Some(stripped) = strip_line_number_prefixes(old)
        && content.contains(&stripped)
    {
        return ". old_string includes read_file's line-number prefixes — resend the exact \
text without them"
            .to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() > HINT_SCAN_MAX_LINES {
        return ". The file may have changed since you read it — re-read the region and copy \
the exact text"
            .to_string();
    }
    let norm: Vec<String> = lines.iter().map(|l| collapse_ws(l)).collect();
    let mut old_norm: Vec<String> = old.lines().map(collapse_ws).collect();
    // Blank boundary lines collapse to "" and would match ANY line — drop them.
    while old_norm.first().is_some_and(|l| l.is_empty()) {
        old_norm.remove(0);
    }
    while old_norm.last().is_some_and(|l| l.is_empty()) {
        old_norm.pop();
    }
    if old_norm.is_empty() || lines.is_empty() {
        return String::new();
    }
    // Whitespace-insensitive window scan, gated on one distinctive line — a
    // brace-only window matches all over a file and would name the wrong block.
    let distinctive = old_norm.iter().any(|l| l.len() >= 4);
    let starts: Vec<usize> = if distinctive {
        (0..lines.len().saturating_sub(old_norm.len() - 1))
            .filter(|&s| window_matches_ws(&norm, &old_norm, s))
            .collect()
    } else {
        Vec::new()
    };
    if let Some(&s) = starts.first() {
        let n = old_norm.len();
        let many = if starts.len() > 1 {
            format!(" ({} such regions; the first is shown)", starts.len())
        } else {
            String::new()
        };
        return format!(
            ". Lines {}\u{2013}{} differ only in whitespace/indentation{many}; the file's exact \
text is:\n{}\nRetry with old_string copied exactly from that",
            s + 1,
            s + n,
            hint_snippet(&lines, s, n)
        );
    }
    // Anchor on old_string's most distinctive line to point at the closest region.
    let anchor = old_norm
        .iter()
        .enumerate()
        .filter(|(_, l)| l.len() >= 12)
        .max_by_key(|(_, l)| l.len());
    if let Some((old_idx, anchor)) = anchor
        && let Some(hit) = norm.iter().position(|l| l == anchor)
    {
        // Keep the anchor inside the capped snippet.
        let s = hit
            .saturating_sub(old_idx)
            .max(hit.saturating_sub(HINT_SNIPPET_LINES - 1));
        return format!(
            ". Closest match is near line {}:\n{}\nCopy old_string exactly from the file's \
current text",
            hit + 1,
            hint_snippet(&lines, s, old_norm.len())
        );
    }
    ". No similar text found — the file may have changed since you read it; re-read it and retry"
        .to_string()
}

/// Whitespace-collapsed form of a line: trimmed, inner runs squeezed to one space.
pub(super) fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Does `old_norm` match the collapsed file lines at `start`? First/last lines
/// match as suffix/prefix so an `old_string` cut mid-line still anchors.
pub(super) fn window_matches_ws(norm: &[String], old_norm: &[String], start: usize) -> bool {
    let n = old_norm.len();
    if n == 1 {
        return !old_norm[0].is_empty() && norm[start].contains(old_norm[0].as_str());
    }
    norm[start].ends_with(old_norm[0].as_str())
        && norm[start + n - 1].starts_with(old_norm[n - 1].as_str())
        && (1..n - 1).all(|k| norm[start + k] == old_norm[k])
}

/// Verbatim file lines `[start, start+len)` for a hint, capped so the error stays small.
pub(super) fn hint_snippet(lines: &[&str], start: usize, len: usize) -> String {
    let end = (start + len).min(lines.len());
    let shown = (end - start).min(HINT_SNIPPET_LINES);
    let mut out = lines[start..start + shown].join("\n");
    let mut capped = shown < end - start;
    capped |= truncate_on_char_boundary(&mut out, HINT_SNIPPET_BYTES);
    if capped {
        out.push_str("\n… (snippet truncated)");
    }
    out
}

/// Strip `read_file`-style `NNN\t` prefixes when every non-blank line has one.
/// Tab-only: a `NNN:` form would misfire on numeric-key literals (`200: "OK",`).
pub(super) fn strip_line_number_prefixes(s: &str) -> Option<String> {
    let mut out = Vec::new();
    let mut stripped = false;
    for line in s.lines() {
        if line.trim().is_empty() {
            out.push(line.to_string());
            continue;
        }
        let t = line.trim_start();
        let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits == 0 {
            return None;
        }
        let rest = t[digits..].strip_prefix('\t')?;
        out.push(rest.to_string());
        stripped = true;
    }
    stripped.then(|| out.join("\n"))
}

/// Write `content` to `full` atomically: stage it in a sibling temp file, then
/// rename over the target. A crash or error mid-write leaves the original file
/// intact rather than a truncated/partial one. The temp shares the target's
/// parent directory so the rename stays on one filesystem (and is atomic).
pub(crate) fn atomic_write(full: &Path, content: &str) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let parent = full.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = parent.unwrap_or_else(|| Path::new("."));
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".aivo-tmp-{}-{}", std::process::id(), seq));
    if let Err(e) = std::fs::write(&tmp, content) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, full) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

pub(super) fn edit_file(args: &Value, cwd: &Path) -> Result<String, String> {
    edit_file_confined(args, cwd, true)
}

pub(super) fn edit_file_confined(
    args: &Value,
    cwd: &Path,
    confine: bool,
) -> Result<String, String> {
    let path = arg_str(args, "path")?;
    if confine {
        confine_write_path(path, cwd)?;
    }
    let old = arg_str(args, "old_string")?;
    let new = arg_str(args, "new_string")?;
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let full = resolve(cwd, path);
    regular_file_metadata(&full).map_err(|e| format!("read {path}: {e}"))?;
    let content = std::fs::read_to_string(&full).map_err(|e| format!("read {path}: {e}"))?;
    let (updated, n) = apply_one_edit(&content, old, new, replace_all, path)?;
    atomic_write(&full, &updated).map_err(|e| format!("write {path}: {e}"))?;
    if n > 1 {
        Ok(format!("edited {path} ({n} replacements)"))
    } else {
        Ok(format!("edited {path}"))
    }
}

/// Apply several edits to one file atomically: each runs against the result of
/// the previous, and the file is written only if all match — so a later failure
/// never leaves a half-edited file (Claude's MultiEdit semantics).
pub(super) fn multi_edit(args: &Value, cwd: &Path) -> Result<String, String> {
    multi_edit_confined(args, cwd, true)
}

pub(super) fn multi_edit_confined(
    args: &Value,
    cwd: &Path,
    confine: bool,
) -> Result<String, String> {
    let path = arg_str(args, "path")?;
    if confine {
        confine_write_path(path, cwd)?;
    }
    let edits = args
        .get("edits")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing required array argument `edits`".to_string())?;
    if edits.is_empty() {
        return Err("`edits` must contain at least one edit".to_string());
    }
    let full = resolve(cwd, path);
    regular_file_metadata(&full).map_err(|e| format!("read {path}: {e}"))?;
    let mut content = std::fs::read_to_string(&full).map_err(|e| format!("read {path}: {e}"))?;
    let mut replacements = 0usize;
    for (i, edit) in edits.iter().enumerate() {
        let old = edit
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("edit #{} missing `old_string`", i + 1))?;
        let new = edit
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("edit #{} missing `new_string`", i + 1))?;
        let replace_all = edit
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let (updated, n) = apply_one_edit(&content, old, new, replace_all, path)
            .map_err(|e| format!("edit #{}: {e}", i + 1))?;
        content = updated;
        replacements += n;
    }
    atomic_write(&full, &content).map_err(|e| format!("write {path}: {e}"))?;
    Ok(format!(
        "edited {path} ({} edits, {replacements} replacements)",
        edits.len()
    ))
}
