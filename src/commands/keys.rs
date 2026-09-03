//! KeysCommand handler for managing API keys.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use zeroize::Zeroizing;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::cli::KeysArgs;
use crate::commands::keys_ui;
use crate::commands::{starter_provider_label, truncate_url_for_display};
use crate::tui::{FuzzyOutcome, FuzzySelect, MultiSelect};

use crate::errors::ExitCode;
use crate::services::account_store;
use crate::services::models_cache::ModelsCache;
use crate::services::provider_profile::is_aivo_starter_base;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

#[allow(clippy::large_enum_variant)]
enum KeySelection {
    Key(ApiKey),
    Cancelled,
    Empty,
    NotFound,
}

// Force cooked (canonical + echo) mode on stdin. The `console` crate's FuzzySelect
// can leave termios flags off on macOS, and a previously-crashed `aivo` invocation
// may have left the terminal corrupted. `crossterm::disable_raw_mode` is a no-op
// when crossterm didn't enable raw mode itself, so we shell out to `stty sane`.
// Call this at entry of an interactive flow and after any FuzzySelect exits.
#[cfg(unix)]
fn restore_cooked_mode() {
    let _ = std::process::Command::new("stty")
        .arg("sane")
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn restore_cooked_mode() {
    let _ = crossterm::terminal::disable_raw_mode();
}

// Reads a line from stdin, flushing the prompt first so it appears before blocking.
fn term_read_line(prompt: &str) -> std::io::Result<String> {
    term_edit_line(prompt, "")
}

// Single-line editor with cursor movement: ←/→ Home/End (and Ctrl-A/E/B/F) move,
// Backspace/Delete edit at the cursor, Ctrl-U/K/W kill, and bracketed paste inserts
// (control chars stripped). `initial` pre-fills the buffer so edit prompts can offer
// the current value for in-place editing. Falls back to a plain cooked read when
// stdin isn't a TTY (piped input, CI) — there `initial` is ignored and empty input
// lets the caller keep its default.
// Windows has no bracketed paste, so a pasted newline arrives as a bare Enter.
// Peek past key-release records: real input right behind the Enter marks it as
// pasted; a consumed event lands in `pending` for the caller's loop.
fn enter_is_paste_newline(pending: &mut Option<crossterm::event::Event>) -> bool {
    use crossterm::event::{self, Event, KeyEventKind};
    if !cfg!(windows) {
        return false;
    }
    while event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
        let Ok(next) = event::read() else { break };
        if let Event::Key(k) = &next
            && k.kind == KeyEventKind::Release
        {
            continue;
        }
        *pending = Some(next);
        return true;
    }
    false
}

fn term_edit_line(prompt: &str, initial: &str) -> std::io::Result<String> {
    term_edit_line_masked(prompt, initial, |_| false)
}

// `term_edit_line` that echoes `*` per char whenever `should_mask(buf)` holds.
fn term_edit_line_masked(
    prompt: &str,
    initial: &str,
    should_mask: impl Fn(&[char]) -> bool,
) -> std::io::Result<String> {
    use std::io::Write;

    if !crate::tui::prompt_interactive() {
        use std::io::BufRead;
        print!("{}", prompt);
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().lock().read_line(&mut input)?;
        return Ok(input.trim().to_string());
    }

    use crossterm::cursor::MoveToColumn;
    use crossterm::event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    };
    use crossterm::terminal::{self, Clear, ClearType};
    use crossterm::{execute, queue};
    use unicode_width::UnicodeWidthChar;

    // Repaint the input region in place: jump to the input's start column, clear to
    // the line end, write the buffer, then park the cursor at its logical position.
    fn render(
        stdout: &mut std::io::Stdout,
        start_col: u16,
        buf: &[char],
        cursor: usize,
        masked: bool,
    ) -> std::io::Result<()> {
        let cursor_w: u16 = if masked {
            cursor as u16
        } else {
            buf[..cursor]
                .iter()
                .map(|c| UnicodeWidthChar::width(*c).unwrap_or(0) as u16)
                .sum()
        };
        queue!(
            stdout,
            MoveToColumn(start_col),
            Clear(ClearType::UntilNewLine)
        )?;
        if masked {
            write!(stdout, "{}", "*".repeat(buf.len()))?;
        } else {
            write!(stdout, "{}", buf.iter().collect::<String>())?;
        }
        queue!(stdout, MoveToColumn(start_col.saturating_add(cursor_w)))?;
        stdout.flush()
    }

    print!("{}", prompt);
    std::io::stdout().flush()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    let _ = execute!(stdout, EnableBracketedPaste);
    // Anchor for redraws: the column just past the prompt where input begins.
    let start_col = crossterm::cursor::position().map(|(c, _)| c).unwrap_or(0);

    let mut buf: Vec<char> = initial.chars().collect();
    let mut cursor = buf.len();
    let _ = render(&mut stdout, start_col, &buf, cursor, should_mask(&buf));

    let mut pending: Option<Event> = None;
    let result = loop {
        match pending.take().map(Ok).unwrap_or_else(event::read) {
            // On Windows, crossterm emits Press and Release for every key —
            // process Press only.
            Ok(Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press,
                ..
            })) => {
                let ctrl = modifiers.contains(KeyModifiers::CONTROL);
                match code {
                    KeyCode::Enter if enter_is_paste_newline(&mut pending) => continue,
                    KeyCode::Enter => {
                        let _ = write!(stdout, "\r\n");
                        let _ = stdout.flush();
                        break Ok(buf.iter().collect::<String>().trim().to_string());
                    }
                    KeyCode::Char('c') if ctrl => {
                        let _ = write!(stdout, "\r\n");
                        let _ = stdout.flush();
                        break Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "interrupted",
                        ));
                    }
                    // Ctrl-D = readline delete-char: drop the char at the cursor;
                    // on an empty line it's EOF, submitting so the caller keeps
                    // its default.
                    KeyCode::Char('d') if ctrl => {
                        if buf.is_empty() {
                            let _ = write!(stdout, "\r\n");
                            let _ = stdout.flush();
                            break Ok(String::new());
                        }
                        if cursor < buf.len() {
                            buf.remove(cursor);
                        }
                    }
                    KeyCode::Left => cursor = cursor.saturating_sub(1),
                    KeyCode::Char('b') if ctrl => cursor = cursor.saturating_sub(1),
                    KeyCode::Right if cursor < buf.len() => cursor += 1,
                    KeyCode::Char('f') if ctrl && cursor < buf.len() => cursor += 1,
                    KeyCode::Home => cursor = 0,
                    KeyCode::End => cursor = buf.len(),
                    KeyCode::Char('a') if ctrl => cursor = 0,
                    KeyCode::Char('e') if ctrl => cursor = buf.len(),
                    KeyCode::Char('u') if ctrl => {
                        buf.drain(0..cursor);
                        cursor = 0;
                    }
                    KeyCode::Char('k') if ctrl => buf.truncate(cursor),
                    KeyCode::Char('w') if ctrl => {
                        let end = cursor;
                        while cursor > 0 && buf[cursor - 1].is_whitespace() {
                            cursor -= 1;
                        }
                        while cursor > 0 && !buf[cursor - 1].is_whitespace() {
                            cursor -= 1;
                        }
                        buf.drain(cursor..end);
                    }
                    KeyCode::Backspace if cursor > 0 => {
                        cursor -= 1;
                        buf.remove(cursor);
                    }
                    KeyCode::Delete if cursor < buf.len() => {
                        buf.remove(cursor);
                    }
                    KeyCode::Char(c) if !ctrl && !modifiers.contains(KeyModifiers::ALT) => {
                        buf.insert(cursor, c);
                        cursor += 1;
                    }
                    _ => continue,
                }
                let _ = render(&mut stdout, start_col, &buf, cursor, should_mask(&buf));
            }
            // Insert pasted text at the cursor, stripping control chars (newlines,
            // tabs) so a multi-line paste doesn't submit early or smuggle bytes in.
            Ok(Event::Paste(data)) => {
                for c in data.chars().filter(|c| !c.is_control()) {
                    buf.insert(cursor, c);
                    cursor += 1;
                }
                let _ = render(&mut stdout, start_col, &buf, cursor, should_mask(&buf));
            }
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = execute!(stdout, DisableBracketedPaste);
    let _ = terminal::disable_raw_mode();
    result
}

// Edit prompt for a key's display name: pre-fills the current name for in-place
// editing when set, or shows a descriptive placeholder when unnamed. Empty input
// keeps the current value.
fn edit_name(key: &ApiKey) -> std::io::Result<String> {
    let input = if key.name.is_empty() {
        term_read_line(&style::dim(format!(
            "Name [unnamed; shown as {}]: ",
            key.short_id()
        )))?
    } else {
        term_edit_line(&style::dim("Name: "), &key.name)?
    };
    Ok(if input.is_empty() {
        key.name.clone()
    } else {
        input
    })
}

// Reads a line from stdin with masked echo (prints '*' per character) for secrets.
fn term_read_secret(prompt: &str) -> std::io::Result<String> {
    use crossterm::event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    };
    use crossterm::{execute, terminal};
    use std::io::Write;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    // Bracketed paste delivers a multi-line paste as one Event::Paste instead of
    // per-char key events; without it an embedded '\n' reads as Enter and submits
    // the secret early, dumping the rest of the paste onto the next prompt.
    let _ = execute!(stdout, EnableBracketedPaste);
    let mut input = String::new();
    let mut pending: Option<Event> = None;
    let result = loop {
        match pending.take().map(Ok).unwrap_or_else(event::read) {
            // On Windows, crossterm emits Press and Release for every key —
            // process Press only so secrets aren't doubled.
            Ok(Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press,
                ..
            })) => match code {
                KeyCode::Enter if enter_is_paste_newline(&mut pending) => continue,
                KeyCode::Enter => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Ok(input);
                }
                KeyCode::Backspace if !input.is_empty() => {
                    input.pop();
                    let _ = write!(stdout, "\x08 \x08");
                    let _ = stdout.flush();
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "interrupted",
                    ));
                }
                // Ctrl-D = readline delete-char. A masked secret keeps the cursor
                // at the end, so there's nothing ahead to delete; it only acts as
                // EOF on an empty line (caller keeps its default).
                KeyCode::Char('d')
                    if modifiers.contains(KeyModifiers::CONTROL) && input.is_empty() =>
                {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Ok(input);
                }
                KeyCode::Char(c) if !modifiers.contains(KeyModifiers::CONTROL) => {
                    input.push(c);
                    let _ = write!(stdout, "*");
                    let _ = stdout.flush();
                }
                _ => {}
            },
            // Strip control chars (newlines, tabs) so a pasted secret with line
            // breaks doesn't submit early or smuggle control bytes into the key.
            Ok(Event::Paste(data)) => {
                for c in data.chars().filter(|c| !c.is_control()) {
                    input.push(c);
                    let _ = write!(stdout, "*");
                }
                let _ = stdout.flush();
            }
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = execute!(stdout, DisableBracketedPaste);
    let _ = terminal::disable_raw_mode();
    result
}

// Reads a confirmation from stdin (y/yes for true, anything else for false).
fn confirm(prompt: &str) -> std::io::Result<bool> {
    let input = term_read_line(&format!("{} [y/N]: ", prompt))?;
    Ok(matches!(input.to_ascii_lowercase().as_str(), "y" | "yes"))
}

/// Per-loop cap; empty and mismatch retries compose to at most 3×3 prompts.
const PASSWORD_ATTEMPTS: usize = 3;

fn read_password_once(from_stdin: bool, label: &str) -> Result<Zeroizing<String>> {
    if from_stdin {
        use std::io::BufRead;
        use zeroize::Zeroize;

        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        let trimmed_len = line.trim_end_matches(['\n', '\r']).len();
        line.truncate(trimmed_len);
        if line.is_empty() {
            line.zeroize();
            return Err(anyhow::anyhow!("Password from stdin was empty"));
        }
        Ok(Zeroizing::new(line))
    } else {
        // Re-prompt on empty; bounded so an EOF-fed pty can't spin forever.
        for attempt in 0..PASSWORD_ATTEMPTS {
            let pw = term_read_secret(&format!("{}: ", label))?;
            if pw.is_empty() {
                if attempt + 1 < PASSWORD_ATTEMPTS {
                    println!(
                        "{}",
                        style::dim("Password must not be empty — try again (Ctrl+C cancels).")
                    );
                }
                continue;
            }
            return Ok(Zeroizing::new(pw));
        }
        Err(anyhow::anyhow!("Password must not be empty"))
    }
}

fn read_password_twice(from_stdin: bool, label: &str) -> Result<Zeroizing<String>> {
    if from_stdin {
        return read_password_once(true, label);
    }
    for attempt in 0..PASSWORD_ATTEMPTS {
        let pw = read_password_once(false, label)?;
        let confirmation = Zeroizing::new(term_read_secret("Confirm password: ")?);
        if pw.as_str() == confirmation.as_str() {
            return Ok(pw);
        }
        if attempt + 1 < PASSWORD_ATTEMPTS {
            println!("{}", style::dim("Passwords didn't match — try again."));
        }
    }
    Err(anyhow::anyhow!("Passwords did not match"))
}

// Ctrl+C at a prompt surfaces as io::ErrorKind::Interrupted.
fn prompt_interrupted(err: &anyhow::Error) -> bool {
    err.downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == std::io::ErrorKind::Interrupted)
}

/// Outcome of the guided key-selection step of `keys export`.
enum ExportPick {
    Keys(Vec<String>),
    /// Flow ends here; the message has already been printed.
    Stop(ExitCode),
}

fn print_no_keys_to_export(skipped_oauth: usize, skipped_starter: usize) {
    eprintln!("{} No keys to export.", style::red("Error:"));
    if skipped_oauth > 0 {
        eprintln!(
            "{}",
            style::dim("Login sessions never export — sign in again on the target machine.")
        );
    }
    if skipped_starter > 0 {
        eprintln!(
            "{}",
            style::dim("The aivo-starter key never exports — it only works on this machine.")
        );
    }
}

// Comma-joins names into width-bounded lines; never splits a name.
fn wrap_names(names: &[&str], width: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthStr;
    let width = width.max(20);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w = 0usize;
    for (i, name) in names.iter().enumerate() {
        let piece = if i + 1 < names.len() {
            format!("{name},")
        } else {
            (*name).to_string()
        };
        let piece_w = UnicodeWidthStr::width(piece.as_str());
        if current.is_empty() {
            current = piece;
            current_w = piece_w;
        } else if current_w + 1 + piece_w <= width {
            current.push(' ');
            current.push_str(&piece);
            current_w += 1 + piece_w;
        } else {
            lines.push(std::mem::take(&mut current));
            current = piece;
            current_w = piece_w;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn default_export_filename() -> String {
    format!("aivo-keys-{}.aivo", chrono::Local::now().format("%Y-%m-%d"))
}

fn refuse_overwrite(path: &std::path::Path) -> anyhow::Error {
    anyhow::anyhow!(
        "Refusing to overwrite existing file at {}. Pass --yes to override.",
        path.display()
    )
}

/// Directory always errors, symlink errors unless `force`; returns whether
/// a regular file exists — the overwrite decision is the caller's.
fn vet_export_target(path: &std::path::Path, force: bool) -> Result<bool> {
    // `symlink_metadata` doesn't follow symlinks; `exists()` would silently
    // accept a dangling symlink and write through it.
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return Ok(false);
    };
    let ft = meta.file_type();
    if ft.is_dir() {
        return Err(anyhow::anyhow!(
            "{} is a directory — pass a file path.",
            path.display()
        ));
    }
    if ft.is_symlink() {
        if !force {
            return Err(anyhow::anyhow!(
                "Refusing to write through symlink at {}. Pass --yes to override.",
                path.display()
            ));
        }
        return Ok(false);
    }
    Ok(true)
}

fn write_export_file(path: &std::path::Path, data: &[u8], force: bool) -> Result<()> {
    if vet_export_target(path, force)? && !force {
        return Err(refuse_overwrite(path));
    }

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {:?}", parent))?;
    }

    crate::services::atomic_write::atomic_write_secure_blocking(path, data)
}

// Caps a hostile file from OOMing the importer before the password check runs.
const MAX_EXPORT_FILE_BYTES: u64 = 16 * 1024 * 1024;

fn read_export_file_capped(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to read export file: {}", path.display()))?;
    let mut buf = String::new();
    file.by_ref()
        .take(MAX_EXPORT_FILE_BYTES + 1)
        .read_to_string(&mut buf)?;
    if buf.len() as u64 > MAX_EXPORT_FILE_BYTES {
        return Err(anyhow::anyhow!(
            "Export file is larger than {} MiB; refusing to load.",
            MAX_EXPORT_FILE_BYTES / (1024 * 1024)
        ));
    }
    Ok(buf)
}

fn is_http_url(s: &str) -> bool {
    url::Url::parse(s)
        .map(|u| matches!(u.scheme(), "http" | "https"))
        .unwrap_or(false)
}

// Chunks the response so a lying Content-Length can't bypass the size cap.
async fn fetch_export_from_url(url: &str) -> Result<String> {
    let client = crate::services::http_utils::aivo_http_client_builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("failed to build HTTP client")?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to fetch export from {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(anyhow::anyhow!("server returned {} for {}", status, url));
    }
    if let Some(len) = resp.content_length()
        && len > MAX_EXPORT_FILE_BYTES
    {
        return Err(anyhow::anyhow!(
            "Remote export advertises {} bytes; exceeds {} MiB cap.",
            len,
            MAX_EXPORT_FILE_BYTES / (1024 * 1024)
        ));
    }
    let mut total: Vec<u8> = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        if total.len() as u64 + chunk.len() as u64 > MAX_EXPORT_FILE_BYTES {
            return Err(anyhow::anyhow!(
                "Remote export is larger than {} MiB; refusing to load.",
                MAX_EXPORT_FILE_BYTES / (1024 * 1024)
            ));
        }
        total.extend_from_slice(&chunk);
    }
    String::from_utf8(total).context("export file body is not valid UTF-8")
}

/// Sizes the echo erase in `mask_echoed_secret` — keep plain ASCII.
const DUAL_CREDENTIAL_PROMPT: &str = "Base URL or API key: ";

/// Scheme-less URL (`host[:port][/path]`) rather than an API key — such input
/// gets the "must start with http(s)://" error instead of being stored as a secret.
fn looks_like_schemeless_url(input: &str) -> bool {
    if !input
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '/' | ':'))
    {
        return false;
    }
    let host = input.split(['/', ':']).next().unwrap_or("");
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let labels: Vec<&str> = host.split('.').collect();
    if labels.len() < 2 || labels.iter().any(|l| l.is_empty()) {
        return false;
    }
    let ipv4 = labels.len() == 4 && labels.iter().all(|l| l.chars().all(|c| c.is_ascii_digit()));
    let last = labels.last().unwrap();
    ipv4 || (last.len() >= 2 && last.chars().all(|c| c.is_ascii_alphabetic()))
}

/// Masked ⟺ the input would be classified as an API key on submit. Kicks in
/// after 5 chars so short input and URLs being typed stay readable.
fn credential_should_mask(buf: &[char]) -> bool {
    if buf.len() <= 5 {
        return false;
    }
    let s: String = buf.iter().collect();
    !["http://", "https://"]
        .iter()
        .any(|p| s.starts_with(p) || p.starts_with(&s))
        && !looks_like_schemeless_url(&s)
}

/// Erases the dual prompt's echoed rows, leaving only a masked key preview.
fn mask_echoed_secret(key: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout();
    if crate::tui::prompt_interactive() {
        use crossterm::cursor::{MoveToColumn, MoveUp};
        use crossterm::execute;
        use crossterm::terminal::{self, Clear, ClearType};
        use unicode_width::UnicodeWidthStr;
        let cols = match terminal::size() {
            Ok((c, _)) if c > 0 => c as usize,
            _ => 80,
        };
        let width = DUAL_CREDENTIAL_PROMPT.len() + UnicodeWidthStr::width(key);
        // Round down at exact column multiples (xenl terminals defer the wrap)
        // and cap at the cursor row so a paste taller than the screen can't
        // clear unrelated lines above.
        let rows = ((width.saturating_sub(1) / cols + 1) as u16)
            .min(crossterm::cursor::position().map_or(u16::MAX, |(_, row)| row));
        if rows > 0 {
            let _ = execute!(
                stdout,
                MoveUp(rows),
                MoveToColumn(0),
                Clear(ClearType::FromCursorDown)
            );
        }
    }
    let _ = writeln!(
        stdout,
        "{} {}",
        style::dim("API Key:"),
        style::cyan(key_preview(key))
    );
    let _ = stdout.flush();
}

// Creates a safe preview of an API key, handling short keys without panicking.
fn key_preview(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 10 {
        let prefix: String = chars.iter().take(3).collect();
        format!("{prefix}...")
    } else {
        let prefix: String = chars.iter().take(6).collect();
        let suffix: String = chars.iter().skip(chars.len() - 4).collect();
        format!("{prefix}...{suffix}")
    }
}

fn display_secret(key: &ApiKey) -> String {
    key.credential_label()
        .map(str::to_string)
        .unwrap_or_else(|| key_preview(&key.key))
}

pub struct KeysCommand {
    session_store: SessionStore,
    models_cache: ModelsCache,
}

#[derive(Clone, Copy, Debug, Default)]
struct AddKeyOptions<'a> {
    name: Option<&'a str>,
    base_url: Option<&'a str>,
    key: Option<&'a str>,
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Outcome of prompting the user about a conflicting existing key.
enum ReplaceDecision {
    NoExisting,
    Replace(String),
    Abort,
}

fn print_ping_result(result: &PingResult, max_name_len: usize) {
    let icon = match &result.status {
        PingStatus::Ok => style::green(result.status.icon()),
        _ => style::red(result.status.icon()),
    };
    let latency = result
        .latency
        .map(|d| format!("{}ms", d.as_millis()))
        .unwrap_or_default();
    let name_padded = format!("{:<width$}", result.name, width = max_name_len);
    let message = result.status.message();
    println!(
        " {} {}  {}  {:>6}  {}",
        icon,
        name_padded,
        style::dim(truncate_url_for_display(&result.url, 40)),
        style::dim(latency),
        match &result.status {
            PingStatus::Ok => style::green(&message),
            _ => style::red(&message),
        }
    );
}

fn detect_base_url(name: &str) -> Option<&str> {
    crate::services::known_providers::find_by_name_substring(name).map(|p| p.base_url.as_str())
}

/// Placeholder used in `providers.json` for Cloudflare Workers AI account ID.
const CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER: &str = "${CLOUDFLARE_ACCOUNT_ID}";

/// Placeholder used in `providers.json` for the Amazon Bedrock region.
const AWS_REGION_PLACEHOLDER: &str = "${AWS_REGION}";

/// Validates a user-supplied base URL. Returns `Err` with a user-facing message
/// if the URL is malformed. Accepts `http(s)://host[/path]` with no whitespace.
fn validate_base_url(url: &str) -> Result<(), &'static str> {
    let rest = if let Some(r) = url.strip_prefix("https://") {
        r
    } else if let Some(r) = url.strip_prefix("http://") {
        r
    } else {
        return Err("URL must start with http:// or https://");
    };
    if url.chars().any(char::is_whitespace) {
        return Err("URL must not contain whitespace");
    }
    let host = rest.split('/').next().unwrap_or("");
    if host.is_empty() {
        return Err("URL must include a host");
    }
    // API paths are glued on by string concat, so a query would end up mid-URL.
    if url.contains('?') || url.contains('#') {
        return Err("URL must not contain a query string or fragment (secrets go in the API key)");
    }
    Ok(())
}
const CLOUDFLARE_PROVIDER_ID: &str = "cloudflare-workers-ai";
const PICKER_LABEL_WIDTH: usize = 36;
const PICKER_URL_MAX_LEN: usize = 50;

/// Provider info lines shown after picker selection / during shortcut flows.
/// Defined once so the interactive and shortcut paths stay in sync.
const COPILOT_INFO: (&str, &str) = ("GitHub Copilot", "device login: github.com/login/device");
const CODEX_OAUTH_INFO: (&str, &str) = (
    "OpenAI Codex (ChatGPT)",
    "sign in to your ChatGPT account — multiple accounts supported",
);
const CLAUDE_OAUTH_INFO: (&str, &str) = (
    "Claude Code (Anthropic)",
    "sign in to your Anthropic account — multiple accounts supported",
);
const CURSOR_INFO: (&str, &str) = (
    "Cursor",
    "uses cursor-agent login or CURSOR_API_KEY for model discovery",
);
const GROK_OAUTH_INFO: (&str, &str) = (
    "SuperGrok (xAI)",
    "sign in with your SuperGrok / X Premium+ subscription — usable by any coding agent",
);
const KIMI_OAUTH_INFO: (&str, &str) = (
    "Kimi Code (OAuth)",
    "sign in with your Kimi membership — usable by any coding agent",
);
const OLLAMA_INFO: (&str, &str) = ("Ollama", "install: ollama.com/download");
const STARTER_INFO: (&str, &str) = ("aivo starter", "free shared key, no signup needed");

fn format_picker_choice(label: &str, hint: &str) -> String {
    format!(
        "{:<width$} {}",
        label,
        style::dim(hint),
        width = PICKER_LABEL_WIDTH
    )
}

/// Provider column for list/picker rows (labels specials instead of sentinels).
fn key_provider_hint(key: &ApiKey) -> String {
    if key.is_copilot() {
        return "GitHub Copilot".to_string();
    }
    if key.base_url == "ollama" {
        return "Ollama (local)".to_string();
    }
    if key.is_cursor_acp() {
        return "Cursor".to_string();
    }
    if is_aivo_starter_base(&key.base_url) {
        return "aivo starter".to_string();
    }
    if let Some(label) = key.credential_label() {
        return label
            .trim_start_matches('<')
            .trim_end_matches('>')
            .to_string();
    }
    truncate_url_for_display(&key.base_url, 50)
}

/// AWS regions where Amazon Bedrock is available. Pairs are (region, city) so
/// the fuzzy filter can match either form.
const BEDROCK_REGIONS: &[(&str, &str)] = &[
    ("us-east-1", "N. Virginia"),
    ("us-east-2", "Ohio"),
    ("us-west-1", "N. California"),
    ("us-west-2", "Oregon"),
    ("af-south-1", "Cape Town"),
    ("ap-east-1", "Hong Kong"),
    ("ap-east-2", "Taipei"),
    ("ap-northeast-1", "Tokyo"),
    ("ap-northeast-2", "Seoul"),
    ("ap-northeast-3", "Osaka"),
    ("ap-south-1", "Mumbai"),
    ("ap-south-2", "Hyderabad"),
    ("ap-southeast-1", "Singapore"),
    ("ap-southeast-2", "Sydney"),
    ("ap-southeast-3", "Jakarta"),
    ("ap-southeast-4", "Melbourne"),
    ("ap-southeast-5", "Malaysia"),
    ("ap-southeast-7", "Thailand"),
    ("ca-central-1", "Canada Central"),
    ("ca-west-1", "Calgary"),
    ("eu-central-1", "Frankfurt"),
    ("eu-central-2", "Zurich"),
    ("eu-west-1", "Ireland"),
    ("eu-west-2", "London"),
    ("eu-west-3", "Paris"),
    ("eu-north-1", "Stockholm"),
    ("eu-south-1", "Milan"),
    ("eu-south-2", "Spain"),
    ("il-central-1", "Tel Aviv"),
    ("me-central-1", "UAE"),
    ("me-south-1", "Bahrain"),
    ("mx-central-1", "Mexico Central"),
    ("sa-east-1", "São Paulo"),
    ("us-gov-east-1", "GovCloud East"),
    ("us-gov-west-1", "GovCloud West"),
];

/// Picks an AWS region for the Bedrock provider. Returns `Ok(None)` if the
/// user cancels the picker (Esc / Ctrl-C). If the user types a query that
/// matches no list entry and presses Enter, the typed query is accepted as a
/// custom literal — `parse_aws_region` normalizes recognized inputs (bare
/// regions or Bedrock URLs); URL-shaped strings that don't match a known
/// Bedrock host are refused so they can't corrupt the URL template.
///
/// Prints a `Region: <id>  <city>` confirmation line after a successful pick
/// so the chosen region stays visible once the picker UI clears.
///
/// `current` is the region to pre-select (only meaningful when editing an
/// existing Bedrock key). When `Some` and the value matches a known region,
/// the picker opens with that row highlighted; otherwise it defaults to
/// the first entry.
fn pick_bedrock_region(current: Option<&str>) -> Result<Option<String>> {
    let labels: Vec<String> = BEDROCK_REGIONS
        .iter()
        .map(|(region, city)| format_picker_choice(region, city))
        .collect();

    let default_idx = current
        .and_then(|cur| BEDROCK_REGIONS.iter().position(|(r, _)| *r == cur))
        .unwrap_or(0);

    let outcome = FuzzySelect::new()
        .with_prompt("AWS Region")
        .items(&labels)
        .default(default_idx)
        .interact_outcome()?;
    restore_cooked_mode();

    let (region, hint) = match outcome {
        FuzzyOutcome::Selected(idx) => {
            let (r, city) = BEDROCK_REGIONS[idx];
            (r.to_string(), Some(city.to_string()))
        }
        FuzzyOutcome::Cancelled => return Ok(None),
        FuzzyOutcome::Query(q) => {
            // `Query` is only emitted when the filter has zero matches, which
            // requires a non-empty query (an empty query matches every item),
            // so `trimmed` here is guaranteed non-empty.
            let trimmed = q.trim();
            if let Some(r) = crate::services::provider_profile::parse_aws_region(trimmed) {
                let city = BEDROCK_REGIONS
                    .iter()
                    .find(|(reg, _)| *reg == r)
                    .map(|(_, c)| (*c).to_string());
                (r, city)
            } else if trimmed.contains('/') || trimmed.contains(':') || trimmed.contains('.') {
                eprintln!(
                    "{} '{}' isn't a recognized AWS region or Bedrock URL",
                    style::red("Error:"),
                    trimmed
                );
                return Ok(None);
            } else {
                (trimmed.to_string(), Some("custom".to_string()))
            }
        }
    };

    match &hint {
        Some(h) => println!(
            "{} {}  {}",
            style::dim("Region:"),
            style::cyan(&region),
            style::dim(format!("({h})")),
        ),
        None => println!("{} {}", style::dim("Region:"), style::cyan(&region)),
    }
    Ok(Some(region))
}

/// Prompts for a secret. If the user enters nothing, asks whether to save
/// without one; loops until a value is provided or confirmation is given.
fn prompt_secret(prompt: &str, secret_noun: &str) -> std::io::Result<String> {
    loop {
        let input = term_read_secret(&style::dim(prompt))?;
        if !input.is_empty() {
            return Ok(input);
        }
        let confirm_prompt = style::yellow(format!("Save without {}?", secret_noun));
        if confirm(&confirm_prompt)? {
            return Ok(String::new());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PingStatus {
    Ok,
    AuthError,
    Unreachable,
    Timeout,
    Error(String),
}

impl PingStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            PingStatus::Ok => "✓",
            PingStatus::AuthError => "✗",
            PingStatus::Unreachable => "✗",
            PingStatus::Timeout => "✗",
            PingStatus::Error(_) => "✗",
        }
    }

    pub fn message(&self) -> String {
        match self {
            PingStatus::Ok => "ok".to_string(),
            PingStatus::AuthError => "auth failed".to_string(),
            PingStatus::Unreachable => "unreachable".to_string(),
            PingStatus::Timeout => "timeout".to_string(),
            PingStatus::Error(msg) => msg.clone(),
        }
    }

    /// Exit status this ping contributes to, so scripts can gate on `keys ping`.
    pub fn exit_code(&self) -> ExitCode {
        match self {
            PingStatus::Ok => ExitCode::Success,
            PingStatus::AuthError => ExitCode::AuthError,
            PingStatus::Unreachable | PingStatus::Timeout => ExitCode::NetworkError,
            PingStatus::Error(_) => ExitCode::UserError,
        }
    }

    /// Machine-readable identifier used in structured output.
    pub fn json_key(&self) -> &'static str {
        match self {
            PingStatus::Ok => "ok",
            PingStatus::AuthError => "auth_error",
            PingStatus::Unreachable => "unreachable",
            PingStatus::Timeout => "timeout",
            PingStatus::Error(_) => "error",
        }
    }

    pub fn from_http_status(status: u16) -> Self {
        match status {
            200..=299 => PingStatus::Ok,
            401 | 403 => PingStatus::AuthError,
            // 404/405 on probe endpoints means reachable but wrong path — still ok for ping
            404 | 405 => PingStatus::Ok,
            _ => PingStatus::Error(format!("HTTP {}", status)),
        }
    }
}

#[derive(Debug)]
pub struct PingResult {
    pub name: String,
    pub url: String,
    pub status: PingStatus,
    pub latency: Option<Duration>,
}

/// Builds a JSON metadata object for a key. Never includes the secret.
pub(crate) fn key_metadata_json(key: &ApiKey, selected_id: Option<&str>) -> Value {
    json!({
        "id": key.id,
        "name": key.name,
        "base_url": key.base_url,
        "active": selected_id == Some(key.id.as_str()),
        "created_at": key.created_at,
        "learned": learned_routing_json(key),
    })
}

/// Per-tool routing the runtime has discovered for this key (`null` when not
/// yet learned). Surfacing these in `aivo info` is the only way users can see
/// what protocol/path was pinned after fallback resolved — otherwise the
/// pin is invisible until something breaks.
fn learned_routing_json(key: &ApiKey) -> Value {
    json!({
        // Learned upstream protocol per (tool, model); "" = the tool's default.
        "protocol_routes": key.protocol_routes,
        "codex_mode": key.codex_mode.map(|m| m.as_str()),
        "opencode_mode": key.opencode_mode.map(|m| m.as_str()),
        "pi_mode": key.pi_mode.map(|m| m.as_str()),
    })
}

/// Converts a PingResult into a JSON object for structured output.
pub(crate) fn ping_result_json(result: &PingResult) -> Value {
    json!({
        "ok": matches!(result.status, PingStatus::Ok),
        "status": result.status.json_key(),
        "message": result.status.message(),
        "latency_ms": result.latency.map(|d| d.as_millis() as u64),
    })
}

const PING_TIMEOUT: Duration = Duration::from_secs(5);

const PING_MAX_RETRIES: u32 = 3;

pub async fn ping_key(key: ApiKey) -> PingResult {
    let name = if key.name.is_empty() {
        key.short_id().to_string()
    } else {
        key.name.clone()
    };
    let url = key.base_url.clone();

    let start = Instant::now();
    let status = ping_with_retries(&key).await;
    let latency = Some(start.elapsed());

    PingResult {
        name,
        url,
        status,
        latency,
    }
}

/// Pings keys concurrently and calls `on_result(key_id, result)` for each as it resolves.
/// Decrypt failures are reported immediately; successful decrypts are spawned and awaited in order.
pub async fn ping_keys_streaming(keys: Vec<ApiKey>, mut on_result: impl FnMut(&str, &PingResult)) {
    let mut handles = Vec::new();
    for mut key in keys {
        let id = key.id.clone();
        if SessionStore::decrypt_key_secret(&mut key).is_err() {
            on_result(
                &id,
                &PingResult {
                    name: key.display_name().to_string(),
                    url: key.base_url.clone(),
                    status: PingStatus::Error("decrypt failed".to_string()),
                    latency: None,
                },
            );
            continue;
        }
        handles.push(tokio::spawn(async move {
            let result = ping_key(key).await;
            (id, result)
        }));
    }
    for handle in handles {
        if let Ok((id, result)) = handle.await {
            on_result(&id, &result);
        }
    }
}

async fn ping_with_retries(key: &ApiKey) -> PingStatus {
    let mut last_status = PingStatus::Unreachable;
    for _ in 0..PING_MAX_RETRIES {
        last_status = match tokio::time::timeout(PING_TIMEOUT, probe_key(key)).await {
            Ok(Ok(s)) => s,
            Ok(Err(_)) => PingStatus::Unreachable,
            Err(_) => PingStatus::Timeout,
        };
        if matches!(last_status, PingStatus::Ok) {
            return last_status;
        }
    }
    last_status
}

async fn probe_key(key: &ApiKey) -> Result<PingStatus> {
    use crate::services::provider_profile::{ModelListingStrategy, provider_profile_for_key};

    // OAuth keys have no reachable REST endpoint; tokens are consumed by the
    // native CLI against the provider's subscription backend. Report "Ok" so
    // the list doesn't look scary; the real health check is `aivo run <tool>`.
    if key.is_any_oauth() {
        return Ok(PingStatus::Ok);
    }

    let profile = provider_profile_for_key(key);
    let client = crate::services::http_utils::aivo_http_client_builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(PING_TIMEOUT)
        .build()?;

    match profile.model_listing_strategy {
        ModelListingStrategy::Ollama => match client.get("http://localhost:11434/").send().await {
            Ok(_) => Ok(PingStatus::Ok),
            Err(_) => Ok(PingStatus::Unreachable),
        },
        ModelListingStrategy::Copilot => {
            use crate::services::copilot_auth::CopilotTokenManager;
            let tm = CopilotTokenManager::new(key.key.as_str().to_string());
            match tm.get_token().await {
                Ok(_) => Ok(PingStatus::Ok),
                Err(_) => Ok(PingStatus::AuthError),
            }
        }
        ModelListingStrategy::CursorAcp => {
            if crate::services::cursor_acp::is_legacy_cursor_login_secret(key.key.as_str()) {
                return Ok(PingStatus::AuthError);
            }
            match crate::services::cursor_acp::parse_cursor_shadow_secret(key.key.as_str()) {
                // OAuth shadow → `cursor-agent status` is authoritative.
                Some(parsed) if parsed.api_key.is_none() => {
                    match crate::services::cursor_acp::cursor_status_authenticated_for_key(key)
                        .await
                    {
                        Ok(true) => Ok(PingStatus::Ok),
                        Ok(false) => Ok(PingStatus::AuthError),
                        Err(_) => Ok(PingStatus::Unreachable),
                    }
                }
                // API-key shadow → `status` can't validate the key, so
                // we just confirm cursor-agent is installed. The key is
                // exercised on the first real request.
                Some(_) => match crate::services::cursor_acp::ensure_cursor_agent_installed() {
                    Ok(()) => Ok(PingStatus::Ok),
                    Err(_) => Ok(PingStatus::Unreachable),
                },
                None => match crate::services::cursor_acp::ensure_cursor_agent_installed() {
                    Ok(()) => Ok(PingStatus::Ok),
                    Err(_) => Ok(PingStatus::Unreachable),
                },
            }
        }
        ModelListingStrategy::Google => {
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models?key={}",
                key.key.as_str()
            );
            match client.get(&url).send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::Anthropic => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            match client
                .get(&url)
                .header("x-api-key", key.key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send()
                .await
            {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::AivoStarter => {
            let url = format!(
                "{}/v1/models",
                crate::constants::AIVO_STARTER_REAL_URL.trim_end_matches('/')
            );
            let req = crate::services::device_fingerprint::with_starter_headers(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
            );
            match req.send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::CloudflareSearch | ModelListingStrategy::OpenAiCompatible => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            let req = crate::services::opencode_session::with_session_header(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
                &url,
                None,
            );
            match req.send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
    }
}

/// Warms the model cache for a newly added key so subsequent commands are instant.
fn sync_models_in_background(key_id: &str, base_url: &str) {
    if crate::services::provider_profile::is_ollama_base(base_url) {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };
    let _ = std::process::Command::new(exe)
        .args(["models", "--refresh", "--key", key_id])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

impl KeysCommand {
    /// Creates a new KeysCommand instance
    pub fn new(session_store: SessionStore) -> Self {
        Self {
            session_store,
            models_cache: ModelsCache::new(),
        }
    }

    #[cfg(test)]
    fn with_models_cache(session_store: SessionStore, models_cache: ModelsCache) -> Self {
        Self {
            session_store,
            models_cache,
        }
    }

    /// Returns the active key ID from last_selection or active_key_id.
    async fn selected_key_id(&self) -> Option<String> {
        if let Some(id) = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|s| s.key_id)
        {
            return Some(id);
        }
        self.session_store
            .get_active_key_info()
            .await
            .ok()
            .flatten()
            .map(|k| k.id)
    }

    /// Executes the keys command with the specified action
    pub async fn execute(&self, keys_args: KeysArgs) -> ExitCode {
        match self.execute_internal(&keys_args).await {
            Ok(code) => code,
            // Ctrl+C at any prompt reads as a cancel, not a failure.
            Err(e) if prompt_interrupted(&e) => {
                println!("{}", style::dim("Cancelled."));
                ExitCode::Success
            }
            Err(e) => crate::errors::report(&e),
        }
    }

    async fn execute_internal(&self, keys_args: &KeysArgs) -> Result<ExitCode> {
        let action = keys_args.action.as_deref();
        let args: Vec<_> = keys_args.args.iter().map(|s| s.as_str()).collect();
        let first = args.first().copied();
        let add_options = AddKeyOptions {
            name: keys_args.name.as_deref(),
            base_url: keys_args.base_url.as_deref(),
            key: keys_args.key.as_deref(),
        };

        match action {
            None | Some("list" | "ls") if keys_args.ping => {
                self.list_keys_with_ping(keys_args.json).await
            }
            None | Some("list" | "ls") => self.list_keys(keys_args.json).await,
            Some("add") => self.add_key(first, add_options).await,
            Some("rm" | "remove") => self.remove_key(first, keys_args.yes).await,
            Some("use") => self.use_key(first).await,
            Some("cat") => self.cat_key(first).await,
            Some("edit") => self.edit_key(first).await,
            Some("reauth") => self.reauth_key(first).await,
            Some("ping") => self.ping_keys(first, keys_args.all).await,
            Some("reset-route") => self.reset_route(first).await,
            Some("export") => self.export_keys_action(first, keys_args).await,
            Some("import") => self.import_keys_action(first, keys_args).await,
            Some(action) => {
                eprintln!(
                    "{} Unknown action '{}'. Valid actions: list, use, add, rm, cat, edit, reauth, ping, reset-route, export, import.",
                    style::red("Error:"),
                    action
                );
                eprintln!("Run `aivo keys --help` for details.");
                Ok(ExitCode::UserError)
            }
        }
    }

    /// Health-checks API keys.
    async fn ping_keys(&self, key_id_or_name: Option<&str>, ping_all: bool) -> Result<ExitCode> {
        let keys: Vec<ApiKey> = if ping_all {
            let all_keys = self.session_store.get_keys().await?;
            if all_keys.len() > 1 {
                let confirmed = confirm(&format!("Ping all {} keys?", all_keys.len()))?;
                if !confirmed {
                    println!("{}", style::dim("Cancelled."));
                    return Ok(ExitCode::Success);
                }
            }
            all_keys
        } else if let Some(filter) = key_id_or_name {
            let all_keys = self.session_store.get_keys().await?;
            let matched: Vec<ApiKey> = all_keys
                .into_iter()
                .filter(|k| {
                    k.id == filter || k.short_id() == filter || k.name.eq_ignore_ascii_case(filter)
                })
                .collect();
            if matched.is_empty() {
                eprintln!("{} API key \"{}\" not found", style::red("Error:"), filter);
                return Ok(ExitCode::UserError);
            }
            matched
        } else {
            match self.session_store.get_active_key().await? {
                Some(key) => vec![key],
                None => {
                    eprintln!(
                        "{} No active API key. Run 'aivo keys use' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::UserError);
                }
            }
        };

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            println!("  {}", crate::commands::add_key_cta());
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys
            .iter()
            .map(|k| k.display_name().len())
            .max()
            .unwrap_or(0);

        // Worst result across keys, so any failure yields a non-zero exit.
        let mut worst = ExitCode::Success;
        ping_keys_streaming(keys, |_id, result| {
            print_ping_result(result, max_name_len);
            worst = worst.worse_of(result.status.exit_code());
        })
        .await;

        Ok(worst)
    }

    /// Lists all API keys
    async fn list_keys(&self, json: bool) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let selected_key_id = self.selected_key_id().await;
        let cached_account = account_store::load();
        let cached_plan = cached_account.as_ref().and_then(|a| a.plan.as_deref());
        let cached_label = cached_account
            .as_ref()
            .and_then(|a| a.plan_label.as_deref());
        let plan_json = serde_json::to_value(cached_plan).unwrap_or(Value::Null);

        if json {
            let payload: Vec<Value> = keys
                .iter()
                .map(|k| {
                    let mut obj = key_metadata_json(k, selected_key_id.as_deref());
                    if is_aivo_starter_base(&k.base_url) {
                        obj["plan"] = plan_json.clone();
                    }
                    obj
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            println!("  {}", crate::commands::add_key_cta());
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys.iter().map(|k| k.name.len()).max().unwrap_or(0);

        for key in &keys {
            let is_selected = selected_key_id.as_deref() == Some(key.id.as_str());
            let active_indicator = if is_selected {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<3}", key.short_id());
            let name_padded = format!("{:<width$}", key.name, width = max_name_len);
            // First-party aivo key: tint only the name; plan label sits dim in the URL column.
            let starter = is_aivo_starter_base(&key.base_url)
                .then(|| starter_provider_label(cached_plan, cached_label));
            let name_col = match &starter {
                Some(_) => style::magenta(&name_padded),
                None => name_padded,
            };
            let url_col = match &starter {
                Some(label) => style::dim(label),
                None => style::dim(key_provider_hint(key)),
            };
            println!(
                "{} {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                name_col,
                url_col
            );
        }

        Ok(ExitCode::Success)
    }

    /// Lists all API keys with live ping status, streaming results as they complete.
    async fn list_keys_with_ping(&self, json: bool) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let selected_key_id = self.selected_key_id().await;
        let cached_account = account_store::load();
        let cached_plan = cached_account.as_ref().and_then(|a| a.plan.as_deref());
        let cached_label = cached_account
            .as_ref()
            .and_then(|a| a.plan_label.as_deref());
        let plan_json = serde_json::to_value(cached_plan).unwrap_or(Value::Null);

        if json {
            let mut ping_by_id: HashMap<String, Value> = HashMap::new();
            let spinner =
                (!keys.is_empty()).then(|| style::start_spinner(Some(" Pinging keys...")));
            ping_keys_streaming(keys.clone(), |id, result| {
                ping_by_id.insert(id.to_string(), ping_result_json(result));
            })
            .await;
            if let Some((spinning, handle)) = spinner {
                style::stop_spinner(&spinning);
                let _ = handle.await;
            }
            let payload: Vec<Value> = keys
                .iter()
                .map(|k| {
                    let mut obj = key_metadata_json(k, selected_key_id.as_deref());
                    if let Some(p) = ping_by_id.remove(&k.id) {
                        obj["ping"] = p;
                    }
                    if is_aivo_starter_base(&k.base_url) {
                        obj["plan"] = plan_json.clone();
                    }
                    obj
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            println!("  {}", crate::commands::add_key_cta());
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys.iter().map(|k| k.name.len()).max().unwrap_or(0);
        let url_display_width = 42;

        for mut key in keys {
            let is_selected = selected_key_id.as_deref() == Some(key.id.as_str());
            let active_indicator = if is_selected {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<3}", key.short_id());
            let name_padded = format!("{:<width$}", key.name, width = max_name_len);
            let starter = is_aivo_starter_base(&key.base_url)
                .then(|| starter_provider_label(cached_plan, cached_label));
            let name_col = match &starter {
                Some(_) => style::magenta(&name_padded),
                None => name_padded,
            };

            let ping_status = if SessionStore::decrypt_key_secret(&mut key).is_ok() {
                let start = Instant::now();
                let status = ping_with_retries(&key).await;
                let ms = start.elapsed().as_millis();
                let icon = match &status {
                    PingStatus::Ok => style::green(status.icon()),
                    _ => style::red(status.icon()),
                };
                let msg = match &status {
                    PingStatus::Ok => style::green(status.message()),
                    _ => style::red(status.message()),
                };
                format!("{} {:>5}ms {}", icon, ms, msg)
            } else {
                format!("{} {}", style::red("✗"), style::red("decrypt failed"))
            };

            let url_col = match &starter {
                Some(label) => {
                    let padded = format!("{:<width$}", label, width = url_display_width);
                    style::dim(&padded)
                }
                None => {
                    let hint = key_provider_hint(&key);
                    let url_display = if hint.len() > url_display_width {
                        truncate_url_for_display(&key.base_url, url_display_width)
                    } else {
                        hint
                    };
                    let url_padded = format!("{:<width$}", url_display, width = url_display_width);
                    style::dim(&url_padded)
                }
            };
            println!(
                "{} {}  {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                name_col,
                url_col,
                ping_status
            );
        }

        Ok(ExitCode::Success)
    }

    /// Activates a specific API key by ID or name
    /// Clears all learned routing state for a key so the next launch re-probes:
    /// `claude_protocol`, `gemini_protocol`, `claude_path_variant`,
    /// `gemini_path_variant`, `responses_api_supported`, plus the key's cached
    /// model list. Useful when an upstream provider gains/loses support for a
    /// protocol or changes its catalog and the cached state is stale.
    async fn reset_route(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let selection = self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to reset routing for",
                "No API keys found.",
            )
            .await?;
        let key = match selection {
            KeySelection::Key(k) => k,
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };
        // Drop learned routes (and any pre-v2 scalar pins) so the next launch
        // re-learns. Each call no-ops if the key was removed.
        let _ = self.session_store.clear_protocol_routes(&key.id).await;
        let _ = self
            .session_store
            .set_key_claude_protocol(&key.id, None)
            .await;
        let _ = self
            .session_store
            .set_key_gemini_protocol(&key.id, None)
            .await;
        let _ = self
            .session_store
            .set_key_claude_path_variant(&key.id, None)
            .await;
        let _ = self
            .session_store
            .set_key_gemini_path_variant(&key.id, None)
            .await;
        let _ = self
            .session_store
            .set_key_responses_api_supported(&key.id, None)
            .await;
        // A stale route usually means the endpoint changed; drop its cached
        // model list too so pickers re-fetch instead of serving the old catalog.
        self.models_cache
            .remove(&crate::services::model_catalog::model_cache_key_for_key(
                &key,
            ))
            .await;
        println!(
            "{} Cleared learned routing state for {}.",
            style::success_symbol(),
            style::bold(key.display_name())
        );
        println!(
            "{}",
            style::dim("The next launch will re-probe protocol/path automatically.")
        );
        Ok(ExitCode::Success)
    }

    /// Starter rows never appear; login rows show disabled — hiding them
    /// would read as a lost key.
    async fn prompt_pick_export_keys(&self) -> Result<ExportPick> {
        use crate::services::provider_profile::is_aivo_starter_base;

        // Metadata-only load — the picker never needs decrypted secrets.
        let (keys, _) = self.session_store.get_keys_and_active_id_info().await?;
        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            println!("  {}", crate::commands::add_key_cta());
            return Ok(ExportPick::Stop(ExitCode::UserError));
        }
        let mut skipped_starter = 0usize;
        let mut choices: Vec<ApiKey> = Vec::new();
        let mut items: Vec<String> = Vec::new();
        let mut annotations: Vec<Option<String>> = Vec::new();
        for key in keys {
            if is_aivo_starter_base(&key.base_url) {
                skipped_starter += 1;
                continue;
            }
            let login = crate::services::api_key_store::is_login_session_record(&key);
            items.push(format_key_choice(&key));
            annotations.push(login.then(|| "login session — never exports".to_string()));
            choices.push(key);
        }
        let selectable = annotations.iter().filter(|a| a.is_none()).count();
        if selectable == 0 {
            print_no_keys_to_export(choices.len(), skipped_starter);
            return Ok(ExportPick::Stop(ExitCode::UserError));
        }

        match MultiSelect::new()
            .with_prompt("Select keys to export")
            .items(&items)
            .checked(&vec![true; items.len()])
            .annotations(annotations)
            .interact_opt()?
        {
            None => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExportPick::Stop(ExitCode::Success))
            }
            Some(picked) if picked.is_empty() => {
                println!("{}", style::dim("Nothing selected."));
                Ok(ExportPick::Stop(ExitCode::Success))
            }
            Some(picked) => {
                // The picker erases itself — leave a one-line record.
                let echo = if picked.len() <= 3 {
                    picked
                        .iter()
                        .map(|&i| choices[i].display_name())
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    format!("{} of {} selected", picked.len(), selectable)
                };
                println!("{} {}", style::dim("Keys:"), style::cyan(echo));
                Ok(ExportPick::Keys(
                    picked.iter().map(|&i| choices[i].id.clone()).collect(),
                ))
            }
        }
    }

    async fn export_keys_action(
        &self,
        file_arg: Option<&str>,
        keys_args: &KeysArgs,
    ) -> Result<ExitCode> {
        if keys_args.include_oauth {
            eprintln!(
                "{} --include-oauth was removed — login sessions never export. Sign in again on the target machine.",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }
        // `--password-stdin` reserves stdin for the password, so no prompting then.
        let interactive = !keys_args.password_stdin && crate::tui::prompt_interactive();

        // Step chips number only the prompts this invocation will show.
        let picker_step =
            keys_args.ids.is_empty() && interactive && crate::tui::picker_interactive();
        let dest_step = interactive && file_arg.is_none_or(str::is_empty);
        let total_steps = usize::from(picker_step) + usize::from(dest_step) + 1;

        let picked_ids: Option<Vec<String>> = if picker_step {
            keys_ui::step_header(
                1,
                total_steps,
                "Select keys",
                "type to filter, space toggles, enter confirms",
            );
            match self.prompt_pick_export_keys().await? {
                ExportPick::Keys(ids) => Some(ids),
                ExportPick::Stop(code) => return Ok(code),
            }
        } else {
            None
        };

        let id_filter: Option<&[String]> = picked_ids
            .as_deref()
            .or((!keys_args.ids.is_empty()).then_some(keys_args.ids.as_slice()));

        // Fail on an empty selection or bad --ids before any prompt.
        let (records, filter_report) = self.session_store.export_keys(id_filter).await?;
        if records.is_empty() {
            print_no_keys_to_export(filter_report.skipped_oauth, filter_report.skipped_starter);
            return Ok(ExitCode::UserError);
        }
        let n = records.len();

        // Review point before the password prompt; the picker path already
        // echoed, and scripts have no decision left.
        if picked_ids.is_none() && interactive {
            use unicode_width::UnicodeWidthStr;
            let names: Vec<&str> = records.iter().map(|k| k.display_name()).collect();
            let width = console::Term::stdout().size().1 as usize;
            let head = format!("{} {} key{}:", style::dim("Exporting"), n, plural(n));
            let joined = names.join(", ");
            if console::measure_text_width(&head) + 1 + joined.width() <= width {
                println!("{} {}", head, style::dim(joined));
            } else {
                println!("{head}");
                for line in wrap_names(&names, width.saturating_sub(2)) {
                    println!("  {}", style::dim(line));
                }
            }
        }
        if filter_report.skipped_oauth > 0 {
            println!(
                "{} Skipped {} login session{} — they never export; sign in again on the target machine.",
                style::dim("·"),
                filter_report.skipped_oauth,
                plural(filter_report.skipped_oauth),
            );
        }

        let file = match file_arg {
            Some(p) if !p.is_empty() => crate::services::system_env::expand_tilde(p),
            _ if interactive => {
                // Destination always precedes the last step (password).
                keys_ui::step_header(
                    total_steps - 1,
                    total_steps,
                    "Destination",
                    "where to write the encrypted file",
                );
                let input = term_edit_line(&style::dim("Export to: "), &default_export_filename())?;
                if input.is_empty() {
                    println!("{}", style::dim("Cancelled."));
                    return Ok(ExitCode::Success);
                }
                crate::services::system_env::expand_tilde(&input)
            }
            _ => {
                Self::print_help(Some("export"));
                return Ok(ExitCode::Success);
            }
        };

        // Vet the target before the user types a password twice.
        let mut force = keys_args.yes;
        if vet_export_target(&file, force)? && !force {
            if !interactive {
                return Err(refuse_overwrite(&file));
            }
            if !confirm(&format!("Overwrite {}?", file.display()))? {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            force = true;
        }

        if interactive {
            keys_ui::step_header(
                total_steps,
                total_steps,
                "Password",
                "protects the file; input is hidden",
            );
        }
        let password = read_password_twice(keys_args.password_stdin, "Encryption password")?;

        let payload =
            serde_json::to_vec(&records).context("failed to serialise keys for export")?;
        let envelope = crate::services::export_crypto::encrypt_export(&payload, &password)?;
        let json = serde_json::to_string_pretty(&envelope)
            .context("failed to serialise export envelope")?;

        write_export_file(&file, json.as_bytes(), force)?;

        println!(
            "{} Exported {} key{} to {}",
            style::success_symbol(),
            n,
            plural(n),
            style::cyan(file.display().to_string())
        );
        // Keep scripted output tight — the onboarding hint is for humans.
        if interactive {
            println!(
                "{}",
                style::dim(format!(
                    "Import on another machine: aivo keys import {}",
                    file.display()
                ))
            );
        }
        println!(
            "{}",
            style::dim(
                "Keep this file private — anyone with the file and the password can read every key."
            )
        );
        Ok(ExitCode::Success)
    }

    async fn import_keys_action(
        &self,
        file_arg: Option<&str>,
        keys_args: &KeysArgs,
    ) -> Result<ExitCode> {
        use crate::services::api_key_store::ImportPolicy;

        let arg = match file_arg {
            Some(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "{} `aivo keys import` requires a file path or URL.",
                    style::red("Error:")
                );
                Self::print_help(Some("import"));
                return Ok(ExitCode::UserError);
            }
        };

        let raw = if is_http_url(arg) {
            println!("{} {}", style::dim("Fetching"), style::cyan(arg));
            fetch_export_from_url(arg).await?
        } else {
            let path = crate::services::system_env::expand_tilde(arg);
            read_export_file_capped(&path)?
        };
        let envelope: crate::services::export_crypto::ExportEnvelope =
            serde_json::from_str(&raw)
                .context("export file is not a valid aivo keys export (failed to parse JSON)")?;
        envelope.validate_header()?;

        let password = read_password_once(keys_args.password_stdin, "Decryption password")?;
        let plaintext = Zeroizing::new(crate::services::export_crypto::decrypt_export(
            &envelope, &password,
        )?);

        let records: Vec<ApiKey> = serde_json::from_slice(&plaintext)
            .context("export file decrypted but payload is not a valid key list")?;
        if records.is_empty() {
            println!("{}", style::dim("Export contained no keys."));
            return Ok(ExitCode::Success);
        }

        let policy = if keys_args.overwrite {
            ImportPolicy::Overwrite
        } else if keys_args.rename {
            ImportPolicy::Rename
        } else {
            ImportPolicy::Skip
        };

        let report = self.session_store.import_keys(records, policy).await?;
        let check = style::success_symbol();
        let dot = style::dim("·");
        let (i, o, r, s) = (
            report.imported.len(),
            report.overwritten.len(),
            report.renamed.len(),
            report.skipped.len(),
        );
        if i > 0 {
            println!("{check} Imported {i} new key{}.", plural(i));
        }
        if o > 0 {
            println!("{check} Overwrote {o} existing key{}.", plural(o));
        }
        if r > 0 {
            println!(
                "{check} Renamed {r} conflicting key{} (kept existing).",
                plural(r)
            );
        }
        if s > 0 {
            println!(
                "{dot} Skipped {s} conflicting key{}. Re-run with {} or {} to merge.",
                plural(s),
                style::cyan("--overwrite"),
                style::cyan("--rename")
            );
        }
        if report.skipped_starter > 0 {
            println!(
                "{dot} {}",
                style::dim(
                    "Skipped the export's aivo-starter key — it only works on its original machine."
                )
            );
        }
        if report.skipped_oauth > 0 {
            println!(
                "{dot} {}",
                style::dim(
                    "Skipped the export's login sessions — sign in on this machine instead."
                )
            );
        }
        Ok(ExitCode::Success)
    }

    async fn use_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to activate",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                self.activate_key(&key).await?;
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Activates a key and prints confirmation
    async fn activate_key(&self, key: &ApiKey) -> Result<()> {
        self.session_store.set_active_key(&key.id).await?;
        // Update global last-selection: keep existing tool, clear model since
        // the new key may have a different provider/model set.
        if let Some(existing_tool) = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|s| s.tool)
        {
            let _ = self
                .session_store
                .set_last_selection(key, &existing_tool, None)
                .await;
        }
        let preview = display_secret(key);
        println!(
            "{} Activated key: {} {}",
            style::success_symbol(),
            style::cyan(key.display_name()),
            style::dim(&preview)
        );
        Ok(())
    }

    /// Activates a newly added key and prints the post-add status.
    async fn finalize_add(
        &self,
        id: &str,
        name: &str,
        detail: &str,
        next_hint: Option<(&str, &str)>,
    ) -> Result<()> {
        self.session_store.set_active_key(id).await?;
        // Clear last_selection so key resolution picks the new active key
        // instead of a stale last_selection (e.g. aivo-starter).
        let _ = self.session_store.clear_last_selection().await;

        let display_name = style::cyan(if name.is_empty() { id } else { name });
        println!();
        println!(
            "{} Added and activated key: {}",
            style::success_symbol(),
            display_name
        );
        println!("  {}", style::dim(format!("ID: {}", id)));
        println!("  {}", style::dim(detail));
        println!();
        if let Some((cmd, desc)) = next_hint {
            if desc.is_empty() {
                println!("{} {}", style::yellow("Next:"), style::bold(cmd));
            } else {
                println!(
                    "{} {} {}",
                    style::yellow("Next:"),
                    style::bold(cmd),
                    style::dim(desc)
                );
            }
        }
        Ok(())
    }

    /// Displays details for a specific API key
    async fn cat_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to inspect",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                self.display_key_details(&key);
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Displays key details
    fn display_key_details(&self, key: &ApiKey) {
        println!("Name:     {}", style::cyan(key.display_name()));
        println!("Base URL: {}", style::blue(&key.base_url));
        if key.key.is_empty() {
            println!("API Key:  {}", style::dim("(none)"));
        } else if let Some(label) = key.credential_label() {
            // Dumping OAuth bundles / Copilot device tokens would leak live
            // access/refresh tokens that the user can't re-enter anywhere.
            println!("API Key:  {}", style::dim(label));
        } else {
            println!("API Key:  {}", style::yellow(&*key.key));
        }
    }

    /// Interactively edits an API key
    async fn edit_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key = match self
            .resolve_key_selection(key_id_or_name, "Select a key to edit", "No API keys found.")
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                key
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        println!("{}", style::bold("Edit API Key"));
        println!();
        println!("Press Enter to keep the current value.");

        // Bedrock keys: dedicated edit flow that mirrors the add flow's
        // region picker instead of forcing the user to hand-edit the
        // `bedrock-runtime.<region>.amazonaws.com` URL string. Detected
        // by URL pattern (mantle or runtime hosts).
        if let Some(current_region) =
            crate::services::provider_profile::parse_aws_region(&key.base_url)
        {
            return self.edit_bedrock_key(key, &current_region).await;
        }

        // OAuth/cursor: name only — refresh credentials with `keys reauth`.
        if key.is_any_oauth() || key.is_cursor_acp() {
            eprintln!(
                "  {} To refresh login credentials, run {}",
                style::dim("tip:"),
                style::cyan(format!("aivo keys reauth {}", key.display_name())),
            );
            keys_ui::step_header(1, 1, "Name", "a short label for this key");
            let name = edit_name(&key)?;

            if name == key.name {
                println!("{}", style::dim("No changes."));
                return Ok(ExitCode::Success);
            }

            let updated = self
                .session_store
                .update_key(
                    &key.id,
                    &name,
                    &key.base_url,
                    key.claude_protocol,
                    key.key.as_str(),
                )
                .await?;
            if updated {
                println!(
                    "  {} renamed to {}",
                    style::success_symbol(),
                    style::cyan(&name)
                );
            }
            return Ok(ExitCode::Success);
        }

        // Name — pre-filled with the current value for in-place editing.
        keys_ui::step_header(1, 3, "Name", "a short label for this key");
        let name = edit_name(&key)?;

        // Base URL — pre-filled so the user can tweak a long URL in place.
        keys_ui::step_header(2, 3, "Base URL", "where requests are sent");
        let base_url = loop {
            let input = term_edit_line(&style::dim("Base URL: "), &key.base_url)?;
            let value = if input.is_empty() {
                key.base_url.clone()
            } else {
                input
            };
            if value == "copilot"
                || value == "ollama"
                || value == "aivo-starter"
                || value == "aivo starter"
            {
                break value;
            }
            match validate_base_url(&value) {
                Ok(()) => break value,
                Err(msg) if value.starts_with("http://") || value.starts_with("https://") => {
                    eprintln!("{} {msg}", style::red("Error:"));
                }
                Err(_) => {
                    eprintln!(
                        "{} URL must start with http:// or https:// (or enter 'copilot' / 'ollama' for special providers)",
                        style::red("Error:")
                    );
                }
            }
        };

        // API Key
        keys_ui::step_header(3, 3, "API Key", "input is hidden");
        let api_key = loop {
            let preview = display_secret(&key);
            let input = term_read_secret(&style::dim(format!("API Key [{}]: ", preview)))?;
            let value = if input.is_empty() {
                key.key.as_str().to_string()
            } else {
                input
            };
            if value.is_empty() {
                let prompt = style::yellow("Save without an API key?");
                if confirm(&prompt)? {
                    break String::new();
                }
            } else {
                break value;
            }
        };

        println!();

        let updated = self
            .session_store
            .update_key(
                &key.id,
                &name,
                &base_url,
                if base_url == key.base_url {
                    key.claude_protocol
                } else {
                    None
                },
                &api_key,
            )
            .await?;

        if updated && base_url != key.base_url {
            // A new upstream invalidates everything learned for the old one.
            let _ = self.session_store.clear_protocol_routes(&key.id).await;
            let _ = self
                .session_store
                .set_key_gemini_protocol(&key.id, None)
                .await?;
            let _ = self.session_store.set_key_codex_mode(&key.id, None).await?;
            let _ = self
                .session_store
                .set_key_opencode_mode(&key.id, None)
                .await?;
        }

        if !updated {
            eprintln!("{} Key no longer exists", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!(
            "{} Updated key: {}",
            style::success_symbol(),
            style::cyan(if name.is_empty() {
                key.short_id()
            } else {
                &name
            })
        );

        Ok(ExitCode::Success)
    }

    /// Re-login OAuth/Cursor credentials in place. Plain API keys: `keys edit`.
    /// Copilot: remove and re-add.
    async fn reauth_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key = match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to re-authenticate",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(mut k) => {
                SessionStore::decrypt_key_secret(&mut k)?;
                k
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        if key.is_any_oauth() {
            match crate::services::oauth_relogin::relogin_key(&self.session_store, &key).await {
                Ok(_) => {
                    println!(
                        "{} Re-authenticated: {}",
                        style::success_symbol(),
                        style::cyan(key.display_name())
                    );
                    Ok(ExitCode::Success)
                }
                Err(e) => {
                    eprintln!("{} {e:#}", style::red("Error:"));
                    Ok(ExitCode::UserError)
                }
            }
        } else if key.is_cursor_acp() {
            self.reauth_cursor_key(&key).await
        } else {
            eprintln!(
                "{} `keys reauth` only applies to OAuth or cursor keys. Use `keys edit` to rotate a plain API key.",
                style::red("Error:")
            );
            Ok(ExitCode::UserError)
        }
    }

    /// Cursor-specific reauth — preserves the shadow account id, so the
    /// aivo key id (and every downstream reference) keeps working.
    async fn reauth_cursor_key(&self, key: &ApiKey) -> Result<ExitCode> {
        use crate::services::cursor_acp;
        use crate::services::cursor_home_shadow::CursorShadow;

        let Some(parsed) = cursor_acp::parse_cursor_shadow_secret(key.key.as_str()) else {
            eprintln!(
                "{} This cursor key isn't shadow-backed (legacy or raw). Remove it (`aivo keys rm`) and re-add (`aivo keys add cursor`).",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        };

        let shadow = CursorShadow::for_account_id(parsed.account_id.to_string())?;
        // ensure() is idempotent — re-creates dirs / keychain if the
        // shadow got partially deleted while the aivo key survived.
        shadow.ensure()?;

        let new_secret = if parsed.api_key.is_some() {
            let entered = term_read_line(&style::dim("New Cursor API key: "))?;
            let trimmed = entered.trim();
            if trimmed.is_empty() {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            if trimmed.contains(':') {
                eprintln!(
                    "{} Cursor API keys must not contain ':'. Double-check what you pasted.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }
            cursor_acp::build_cursor_apikey_secret(&shadow.account_id, trimmed)
        } else {
            if let Err(e) = cursor_acp::run_cursor_login_for_shadow(&shadow).await {
                eprintln!("{} {e:#}", style::red("Error:"));
                return Ok(ExitCode::UserError);
            }
            if !cursor_acp::cursor_status_authenticated_for_shadow(&shadow)
                .await
                .unwrap_or(false)
            {
                eprintln!(
                    "{} Cursor login was not confirmed by `cursor-agent status`.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }
            cursor_acp::build_cursor_oauth_secret(&shadow.account_id)
        };

        self.session_store
            .update_key(
                &key.id,
                &key.name,
                &key.base_url,
                key.claude_protocol,
                &new_secret,
            )
            .await?;
        println!(
            "{} Re-authenticated: {}",
            style::success_symbol(),
            style::cyan(key.display_name())
        );
        Ok(ExitCode::Success)
    }

    /// Edit a Bedrock key via region picker (ESC keeps the current region).
    async fn edit_bedrock_key(&self, key: ApiKey, current_region: &str) -> Result<ExitCode> {
        // Name (same shape as the generic edit flow).
        keys_ui::step_header(1, 3, "Name", "a short label for this key");
        let name = edit_name(&key)?;

        // Region picker, defaulted to current.
        keys_ui::step_header(2, 3, "Region", "Bedrock AWS region");
        let region = match pick_bedrock_region(Some(current_region))? {
            Some(r) => r,
            None => {
                println!("{} {}", style::dim("Region:"), style::cyan(current_region),);
                current_region.to_string()
            }
        };

        // Substitute the region into the existing URL — preserves
        // runtime vs mantle form. `replacen(.., 1)` is defensive: a
        // pathological stored URL with the region appearing twice (a
        // path component) shouldn't get its tail rewritten.
        let base_url = if region == current_region {
            key.base_url.clone()
        } else {
            key.base_url.replacen(current_region, &region, 1)
        };

        // API key with masked preview (same shape as generic edit flow).
        keys_ui::step_header(3, 3, "API Key", "input is hidden");
        let api_key = loop {
            let preview = display_secret(&key);
            let input = term_read_secret(&style::dim(format!("API Key [{}]: ", preview)))?;
            let value = if input.is_empty() {
                key.key.as_str().to_string()
            } else {
                input
            };
            if value.is_empty() {
                let prompt = style::yellow("Save without an API key?");
                if confirm(&prompt)? {
                    break String::new();
                }
            } else {
                break value;
            }
        };

        println!();

        if name == key.name && base_url == key.base_url && api_key == key.key.as_str() {
            println!("{}", style::dim("No changes."));
            return Ok(ExitCode::Success);
        }

        let updated = self
            .session_store
            .update_key(&key.id, &name, &base_url, key.claude_protocol, &api_key)
            .await?;

        if !updated {
            eprintln!("{} Key no longer exists", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!(
            "{} Updated key: {}",
            style::success_symbol(),
            style::cyan(if name.is_empty() {
                key.short_id()
            } else {
                &name
            })
        );

        Ok(ExitCode::Success)
    }

    /// Shows the provider picker, then routes to the appropriate add flow.
    /// `name` is the pre-collected key name (may be empty — each arm supplies
    /// its own default when empty).
    async fn interactive_add(&self, name: &str) -> Result<ExitCode> {
        #[derive(Clone, Copy, PartialEq)]
        enum ProviderChoice {
            Known(usize),
            Copilot,
            CodexOAuth,
            ClaudeOAuth,
            GrokOAuth,
            KimiOAuth,
            Cursor,
            Ollama,
            Starter,
            Custom,
        }

        let providers = crate::services::known_providers::all();
        let existing_keys = self.session_store.get_keys().await?;
        let has_starter = existing_keys
            .iter()
            .any(|k| is_aivo_starter_base(&k.base_url));

        let ollama_base_url = crate::services::ollama::ollama_openai_base_url();

        let label_for = |choice: ProviderChoice| -> (&str, String) {
            match choice {
                ProviderChoice::Known(i) => {
                    let p = &providers[i];
                    (
                        p.name.as_str(),
                        truncate_url_for_display(&p.base_url, PICKER_URL_MAX_LEN),
                    )
                }
                ProviderChoice::Ollama => ("Ollama", ollama_base_url.clone()),
                ProviderChoice::Copilot => ("GitHub Copilot", "device login".to_string()),
                ProviderChoice::CodexOAuth => (
                    "OpenAI Codex (ChatGPT)",
                    "browser login — multi-account".to_string(),
                ),
                ProviderChoice::ClaudeOAuth => (
                    "Claude Code (Anthropic)",
                    "browser login — multi-account".to_string(),
                ),
                ProviderChoice::GrokOAuth => (
                    "SuperGrok (xAI)",
                    "device login — any coding agent".to_string(),
                ),
                ProviderChoice::KimiOAuth => (
                    "Kimi Code (OAuth)",
                    "device login — any coding agent".to_string(),
                ),
                ProviderChoice::Cursor => ("Cursor", "cursor-agent login/API key".to_string()),
                ProviderChoice::Starter => ("aivo starter", "free".to_string()),
                ProviderChoice::Custom => ("Custom URL", "enter manually".to_string()),
            }
        };

        let mut choices: Vec<ProviderChoice> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        let mut preselected: Option<usize> = None;
        let push = |choices: &mut Vec<ProviderChoice>,
                    labels: &mut Vec<String>,
                    choice: ProviderChoice| {
            let (label, hint) = label_for(choice);
            labels.push(format_picker_choice(label, &hint));
            choices.push(choice);
        };

        push(&mut choices, &mut labels, ProviderChoice::Custom);

        let detected_indices: Vec<usize> = if name.is_empty() {
            Vec::new()
        } else {
            crate::services::known_providers::find_all_by_name_substring(name)
                .into_iter()
                .filter_map(|m| providers.iter().position(|p| p.base_url == m.base_url))
                .collect()
        };

        let hoisted_special: Option<ProviderChoice> = if name.is_empty() {
            None
        } else {
            match name.trim().to_ascii_lowercase().as_str() {
                "ollama" => Some(ProviderChoice::Ollama),
                "copilot" => Some(ProviderChoice::Copilot),
                "codex" => Some(ProviderChoice::CodexOAuth),
                "claude" => Some(ProviderChoice::ClaudeOAuth),
                "grok" | "supergrok" => Some(ProviderChoice::GrokOAuth),
                "kimi" | "kimi-code" => Some(ProviderChoice::KimiOAuth),
                "cursor" => Some(ProviderChoice::Cursor),
                _ => None,
            }
        };

        // Hoist matched entries right after Custom URL so users can pick among
        // ambiguous matches (e.g. "bedrock" → Mantle + Runtime) without
        // scrolling. An exact special-name match ("kimi" → Kimi Code OAuth)
        // outranks substring provider matches ("Kimi For Coding" API key);
        // both are offered, special first and preselected.
        let hoisted = usize::from(hoisted_special.is_some()) + detected_indices.len();
        if hoisted > 0 {
            if let Some(special) = hoisted_special {
                push(&mut choices, &mut labels, special);
            }
            for &di in &detected_indices {
                push(&mut choices, &mut labels, ProviderChoice::Known(di));
            }
            preselected = Some(labels.len() - hoisted);
        }

        for (i, _) in providers.iter().enumerate() {
            if detected_indices.contains(&i) {
                continue;
            }
            push(&mut choices, &mut labels, ProviderChoice::Known(i));
        }

        for choice in [
            ProviderChoice::Ollama,
            ProviderChoice::Copilot,
            ProviderChoice::CodexOAuth,
            ProviderChoice::ClaudeOAuth,
            ProviderChoice::GrokOAuth,
            ProviderChoice::KimiOAuth,
            ProviderChoice::Cursor,
        ] {
            if hoisted_special == Some(choice) {
                continue;
            }
            push(&mut choices, &mut labels, choice);
        }

        // Starter is a singleton — hide the picker entry once one is already set up.
        if !has_starter {
            push(&mut choices, &mut labels, ProviderChoice::Starter);
        }

        keys_ui::step_header(2, 3, "Provider", "preset or custom URL");

        let selection = FuzzySelect::new()
            .with_prompt("Provider")
            .items(&labels)
            .default(preselected.unwrap_or(0))
            .interact_opt()?;

        let Some(idx) = selection else {
            println!("{}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        };

        // FuzzySelect's `console` crate can leave termios flags off.
        restore_cooked_mode();

        let picked = choices[idx];
        let picked_label = label_for(picked).0;
        let picked_url: Option<String> = match picked {
            ProviderChoice::Known(i) => Some(providers[i].base_url.clone()),
            ProviderChoice::Ollama => Some(ollama_base_url.clone()),
            _ => None,
        };
        match picked_url {
            Some(url) => println!(
                "{} {}  {}",
                style::dim("Provider:"),
                style::cyan(picked_label),
                style::dim(url),
            ),
            None => println!("{} {}", style::dim("Provider:"), style::cyan(picked_label)),
        }

        match picked {
            ProviderChoice::Known(i) => self.add_known_provider(name, &providers[i]).await,
            ProviderChoice::Copilot => self.add_copilot_interactive(name).await,
            ProviderChoice::CodexOAuth => self.add_codex_oauth_interactive(name).await,
            ProviderChoice::ClaudeOAuth => self.add_claude_oauth_interactive(name).await,
            ProviderChoice::GrokOAuth => self.add_grok_oauth_interactive(name).await,
            ProviderChoice::KimiOAuth => self.add_kimi_oauth_interactive(name).await,
            ProviderChoice::Cursor => self.add_cursor_interactive(name, None).await,
            ProviderChoice::Ollama => self.add_ollama_interactive(name).await,
            ProviderChoice::Starter => self.add_starter_interactive(name).await,
            ProviderChoice::Custom => self.add_custom_interactive(name).await,
        }
    }

    async fn add_known_provider(
        &self,
        name: &str,
        provider: &crate::services::known_providers::KnownProvider,
    ) -> Result<ExitCode> {
        let name = if name.is_empty() {
            self.unique_key_name(provider.id.clone()).await
        } else {
            name.to_string()
        };
        let is_cloudflare = provider.id == CLOUDFLARE_PROVIDER_ID;

        let base_url = if provider
            .base_url
            .contains(CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER)
        {
            keys_ui::step_header(
                3,
                3,
                "Credentials",
                "Cloudflare Account ID, then auth token",
            );
            let account_id = loop {
                let input = term_read_line(&style::dim("Cloudflare Account ID: "))?;
                if !input.is_empty() {
                    break input;
                }
                eprintln!("{} Account ID is required", style::red("Error:"));
            };
            provider
                .base_url
                .replace(CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER, &account_id)
        } else if provider.base_url.contains(AWS_REGION_PLACEHOLDER) {
            keys_ui::step_header(3, 3, "Credentials", "AWS region, then Bedrock API key");
            let Some(region) = pick_bedrock_region(None)? else {
                return Ok(ExitCode::Success);
            };
            provider.base_url.replace(AWS_REGION_PLACEHOLDER, &region)
        } else {
            keys_ui::step_header(3, 3, "API Key", "input is hidden");
            provider.base_url.clone()
        };

        let (key_prompt, secret_noun) = if is_cloudflare {
            ("Auth Token: ", "an auth token")
        } else {
            ("API Key: ", "an API key")
        };
        let key = prompt_secret(key_prompt, secret_noun)?;

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            &name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;
        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    async fn add_copilot_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(COPILOT_INFO.0, COPILOT_INFO.1);

        let decision = self.confirm_replace_existing("copilot", "Copilot").await?;
        if matches!(decision, ReplaceDecision::Abort) {
            return Ok(ExitCode::Success);
        }

        keys_ui::step_header(3, 3, "Device login", "follow the code shown below");
        let token = crate::services::copilot_auth::device_flow_login().await?;
        if let ReplaceDecision::Replace(old_id) = decision {
            self.session_store.delete_key(&old_id).await?;
        }

        let name = if name.is_empty() { "copilot" } else { name };
        let id = self
            .session_store
            .add_key_with_protocol(name, "copilot", None, &token)
            .await?;
        self.finalize_add(
            &id,
            name,
            "Provider: GitHub Copilot",
            Some(("aivo run claude", "(uses Copilot subscription)")),
        )
        .await?;
        sync_models_in_background(&id, "copilot");
        Ok(ExitCode::Success)
    }

    /// Prompt to adopt an existing codex sign-in (default yes; non-TTY auto-yes).
    fn offer_codex_import(&self, email: Option<&str>) -> Result<bool> {
        use std::io::IsTerminal;
        let who = match email {
            Some(e) => format!("Found an existing codex sign-in for {e}."),
            None => "Found an existing codex sign-in.".to_string(),
        };
        eprintln!("  {}", style::dim(&who));
        if !std::io::stdin().is_terminal() {
            return Ok(true);
        }
        let input = term_read_line("  Import it (skip browser login)? [Y/n]: ")?;
        Ok(!matches!(
            input.trim().to_ascii_lowercase().as_str(),
            "n" | "no"
        ))
    }

    /// Interactive Codex ChatGPT OAuth sign-in (multi-account). Prefers adopting
    /// an existing native `codex` session when present.
    async fn add_codex_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::codex_oauth::{
            CODEX_OAUTH_SENTINEL, import_from_codex_home, interactive_login,
        };

        keys_ui::provider_info(CODEX_OAUTH_INFO.0, CODEX_OAUTH_INFO.1);

        // Adopt an existing codex sign-in when present, else browser login.
        let existing = import_from_codex_home().await.unwrap_or(None);
        let creds = match existing {
            Some(creds) if self.offer_codex_import(creds.email.as_deref())? => {
                keys_ui::step_header(3, 3, "Import", "adopting your existing codex sign-in");
                creds
            }
            _ => {
                keys_ui::step_header(
                    3,
                    3,
                    "Sign in",
                    "follow the URL below — the browser opens automatically if possible",
                );
                interactive_login().await?
            }
        };
        let derived_name = creds.email.clone().unwrap_or_else(|| "codex".to_string());
        let final_name = if name.is_empty() { &derived_name } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, CODEX_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        let email_line = match creds.email.as_deref() {
            Some(email) => format!("Signed in as {}", email),
            None => "Signed in to Codex".to_string(),
        };
        self.finalize_add(
            &id,
            final_name,
            &email_line,
            Some(("aivo run codex", "(launches codex with this account)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Interactive Claude Code OAuth sign-in. Unlike Copilot we allow MULTIPLE
    /// accounts — each login produces a fresh key entry. Shells out to
    /// `claude setup-token`, which drives the browser OAuth flow itself; we
    /// capture the opaque token it prints on stdout and store it encrypted.
    async fn add_claude_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::claude_oauth::{
            CLAUDE_OAUTH_SENTINEL, SetupTokenError, spawn_setup_token_and_capture,
        };

        keys_ui::provider_info(CLAUDE_OAUTH_INFO.0, CLAUDE_OAUTH_INFO.1);
        keys_ui::step_header(
            3,
            3,
            "Sign in",
            "running `claude setup-token` — follow the browser prompt",
        );

        let creds = match spawn_setup_token_and_capture().await {
            Ok(c) => c,
            Err(SetupTokenError::ClaudeNotFound) => {
                eprintln!(
                    "{} The `claude` CLI wasn't found on PATH.",
                    style::red("Error:")
                );
                eprintln!(
                    "  {} Install Claude Code first: {}",
                    style::dim("hint:"),
                    style::cyan("npm i -g @anthropic-ai/claude-code")
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::EmptyOutput) => {
                eprintln!(
                    "{} `claude setup-token` exited without printing a token.",
                    style::red("Error:")
                );
                eprintln!(
                    "  {} If the login was cancelled, try again. Otherwise report the output above.",
                    style::dim("hint:"),
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::NonZeroExit { status }) => {
                eprintln!(
                    "{} `claude setup-token` failed ({}).",
                    style::red("Error:"),
                    status
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::Other(e)) => return Err(e),
        };

        let final_name = if name.is_empty() { "claude" } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, CLAUDE_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        self.finalize_add(
            &id,
            final_name,
            "Signed in to Claude Code",
            Some(("aivo run claude", "(launches claude with this account)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Interactive SuperGrok / X Premium+ OAuth sign-in (device-code flow). The
    /// stored credential is a provider bearer usable by any coding agent.
    async fn add_grok_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::grok_oauth::{GROK_OAUTH_SENTINEL, interactive_login};

        keys_ui::provider_info(GROK_OAUTH_INFO.0, GROK_OAUTH_INFO.1);
        keys_ui::step_header(3, 3, "Sign in", "enter the code shown below at the URL");

        let creds = interactive_login().await?;
        let derived_name = creds
            .account_label
            .clone()
            .unwrap_or_else(|| "grok".to_string());
        let final_name = if name.is_empty() { &derived_name } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, GROK_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        self.finalize_add(
            &id,
            final_name,
            "Signed in to SuperGrok",
            Some(("aivo code", "(any coding agent can use this subscription)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Interactive Kimi Code OAuth sign-in (device-code flow). The stored
    /// credential is a provider bearer usable by any coding agent.
    async fn add_kimi_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::kimi_oauth::{KIMI_OAUTH_SENTINEL, interactive_login};

        keys_ui::provider_info(KIMI_OAUTH_INFO.0, KIMI_OAUTH_INFO.1);
        keys_ui::step_header(3, 3, "Sign in", "enter the code shown below at the URL");

        let creds = interactive_login().await?;
        let final_name = if name.is_empty() { "kimi" } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, KIMI_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        self.finalize_add(
            &id,
            final_name,
            "Signed in to Kimi Code",
            Some(("aivo code", "(any coding agent can use this subscription)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    async fn add_cursor_interactive(
        &self,
        name: &str,
        explicit_key: Option<&str>,
    ) -> Result<ExitCode> {
        keys_ui::provider_info(CURSOR_INFO.0, CURSOR_INFO.1);

        // Cursor keys are multi-account — each `aivo keys add cursor`
        // produces a fresh isolated shadow account. No replace prompt; if
        // the user wants to swap accounts, they remove + add.

        keys_ui::step_header(3, 3, "Credentials", "checking cursor-agent");

        // Always allocate a shadow first so cursor-agent's per-account
        // state (auth.json, cli-config.json, projects/, chats/,
        // acp-sessions/) stays out of the user's real ~/.cursor regardless
        // of auth mode.
        let shadow = crate::services::cursor_home_shadow::CursorShadow::create_new()?;

        let secret = match self.resolve_cursor_auth(explicit_key, &shadow).await {
            Ok(Some(s)) => s,
            Ok(None) => {
                let _ = shadow.delete();
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            Err(code) => {
                let _ = shadow.delete();
                return Ok(code);
            }
        };
        let shadow_to_cleanup_on_abort = Some(shadow);

        let final_name = if name.is_empty() { "cursor" } else { name };
        let id = match self
            .session_store
            .add_key_with_protocol(final_name, "cursor", None, &secret)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                if let Some(shadow) = shadow_to_cleanup_on_abort {
                    let _ = shadow.delete();
                }
                return Err(e);
            }
        };
        self.finalize_add(
            &id,
            final_name,
            "Provider: Cursor",
            Some(("aivo models", "(list Cursor models)")),
        )
        .await?;
        sync_models_in_background(&id, final_name);
        Ok(ExitCode::Success)
    }

    /// Decide how the new cursor key authenticates. Returns:
    /// - `Ok(Some(secret))` — secret to store in `key.key`.
    /// - `Ok(None)` — user cancelled; caller cleans up.
    /// - `Err(code)` — non-recoverable error; caller cleans up and exits
    ///   with the given code.
    async fn resolve_cursor_auth(
        &self,
        explicit_key: Option<&str>,
        shadow: &crate::services::cursor_home_shadow::CursorShadow,
    ) -> std::result::Result<Option<String>, ExitCode> {
        use crate::services::cursor_acp;
        use std::io::IsTerminal;

        if let Some(key) = explicit_key {
            return Ok(Some(cursor_acp::build_cursor_apikey_secret(
                &shadow.account_id,
                key,
            )));
        }
        if let Ok(key) = std::env::var("CURSOR_API_KEY")
            && !key.trim().is_empty()
        {
            println!("{}", style::dim("Using CURSOR_API_KEY from environment."));
            return Ok(Some(cursor_acp::build_cursor_apikey_secret(
                &shadow.account_id,
                key.trim(),
            )));
        }

        if !std::io::stderr().is_terminal() {
            eprintln!(
                "{} Cursor needs an interactive terminal for sign-in. Pass `--api-key sk-…` or set CURSOR_API_KEY for non-interactive setup.",
                style::red("Error:")
            );
            return Err(ExitCode::UserError);
        }

        let chose_api_key = match self.prompt_cursor_auth_mode() {
            Ok(Some(v)) => v,
            Ok(None) => return Ok(None),
            Err(_) => return Err(ExitCode::UserError),
        };

        if chose_api_key {
            let entered = match term_read_line(&style::dim("Cursor API key: ")) {
                Ok(s) => s,
                Err(_) => return Err(ExitCode::UserError),
            };
            let trimmed = entered.trim();
            if trimmed.is_empty() {
                println!("{}", style::dim("Cancelled."));
                return Ok(None);
            }
            // Reject `:` because the on-disk secret format uses it as a
            // separator (`cursor-shadow:<id>:api:<key>`). Cursor's keys
            // don't contain `:`, so this is safe and protective.
            if trimmed.contains(':') {
                eprintln!(
                    "{} Cursor API keys must not contain ':'. Double-check what you pasted.",
                    style::red("Error:")
                );
                return Err(ExitCode::UserError);
            }
            return Ok(Some(cursor_acp::build_cursor_apikey_secret(
                &shadow.account_id,
                trimmed,
            )));
        }

        // OAuth login into the shadow.
        if let Err(e) = cursor_acp::run_cursor_login_for_shadow(shadow).await {
            eprintln!("{} {:#}", style::red("Error:"), e);
            return Err(ExitCode::UserError);
        }
        if !cursor_acp::cursor_status_authenticated_for_shadow(shadow)
            .await
            .unwrap_or(false)
        {
            eprintln!(
                "{} Cursor login was not confirmed by `cursor-agent status`.",
                style::red("Error:")
            );
            return Err(ExitCode::UserError);
        }
        Ok(Some(cursor_acp::build_cursor_oauth_secret(
            &shadow.account_id,
        )))
    }

    /// Two-branch picker for cursor sign-in. Returns:
    /// - `Ok(Some(true))` → API key (paste a Cursor API key)
    /// - `Ok(Some(false))` → OAuth login (sign in via browser)
    /// - `Ok(None)` → user cancelled (Esc / Ctrl-C)
    fn prompt_cursor_auth_mode(&self) -> Result<Option<bool>> {
        let items = vec![
            "Sign in with browser  —  cursor-agent login (recommended)".to_string(),
            "Paste a Cursor API key  —  from cursor.com → Settings → API Keys".to_string(),
        ];
        let selection = FuzzySelect::new()
            .with_prompt("Authentication")
            .items(&items)
            .default(0)
            .interact_opt()?;
        restore_cooked_mode();
        Ok(selection.map(|idx| idx == 1))
    }

    async fn add_ollama_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(OLLAMA_INFO.0, OLLAMA_INFO.1);

        let decision = self.confirm_replace_existing("ollama", "Ollama").await?;
        if matches!(decision, ReplaceDecision::Abort) {
            return Ok(ExitCode::Success);
        }

        keys_ui::step_header(3, 3, "Verify local install", "checking ollama is reachable");
        crate::services::ollama::ensure_ready().await?;
        if let ReplaceDecision::Replace(old_id) = decision {
            self.session_store.delete_key(&old_id).await?;
        }

        let name = if name.is_empty() { "ollama" } else { name };
        let id = self
            .session_store
            .add_key_with_protocol(name, "ollama", None, "ollama-local")
            .await?;
        self.finalize_add(
            &id,
            name,
            "Provider: Ollama (local)",
            Some(("aivo models", "(list local models)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    async fn add_starter_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(STARTER_INFO.0, STARTER_INFO.1);

        if let Some(code) = self.notify_if_starter_already_added().await? {
            return Ok(code);
        }

        let _ = self.session_store.set_starter_key_dismissed(false).await;

        let (starter, _) = self
            .session_store
            .ensure_starter_key()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to create aivo starter key"))?;
        let display_name = if name.is_empty() {
            "aivo-starter"
        } else {
            name
        };
        self.finalize_add(
            &starter.id,
            display_name,
            "Provider: aivo starter (free)",
            Some(("aivo code", "(start coding)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Prints a notice and returns `Some(Success)` if an aivo-starter key is
    /// already configured; returns `None` otherwise so the caller can create one.
    async fn notify_if_starter_already_added(&self) -> Result<Option<ExitCode>> {
        let Some(existing) = self
            .session_store
            .get_keys()
            .await?
            .into_iter()
            .find(|k| k.base_url == crate::constants::AIVO_STARTER_SENTINEL)
        else {
            return Ok(None);
        };
        println!(
            "{} aivo starter is already added as {} (ID: {}). Run {} to use it.",
            style::yellow("Note:"),
            style::cyan(&existing.name),
            style::dim(&existing.id),
            style::cyan("aivo code"),
        );
        Ok(Some(ExitCode::Success))
    }

    async fn add_custom_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::step_header(3, 3, "Credentials", "base URL and API key, either order");
        // Either credential may come first, so a key already on the clipboard
        // isn't lost to copying the URL.
        let mut pasted_key: Option<String> = None;
        let base_url = loop {
            let input = if pasted_key.is_none() {
                term_edit_line_masked(
                    &style::dim(DUAL_CREDENTIAL_PROMPT),
                    "",
                    credential_should_mask,
                )?
            } else {
                term_read_line(&style::dim("Base URL: "))?
            };
            if input.is_empty() {
                continue;
            }
            if input.starts_with("http://") || input.starts_with("https://") {
                match validate_base_url(&input) {
                    Ok(()) => break input,
                    Err(msg) => {
                        eprintln!("{} {}", style::red("Error:"), msg);
                    }
                }
            } else if pasted_key.is_none() && !looks_like_schemeless_url(&input) {
                mask_echoed_secret(&input);
                pasted_key = Some(input);
            } else {
                eprintln!(
                    "{} URL must start with http:// or https://",
                    style::red("Error:")
                );
            }
        };

        let key = match pasted_key {
            Some(key) => key,
            None => prompt_secret("API Key: ", "an API key")?,
        };

        let name = if name.is_empty() {
            self.derived_key_name(&base_url).await
        } else {
            name.to_string()
        };

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            &name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;
        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    /// Prompts to replace any existing key with the given `base_url`.
    /// - Returns `Ok(Some(id))` — caller must delete `id` before adding the new key.
    /// - Returns `Ok(None)` when there was no existing key (caller proceeds).
    /// - Returns `Err` on IO error; aborts on user decline by returning early via this type.
    async fn confirm_replace_existing(
        &self,
        base_url: &str,
        label: &str,
    ) -> Result<ReplaceDecision> {
        let existing_keys = self.session_store.get_keys().await?;
        let Some(existing) = existing_keys.iter().find(|k| k.base_url == base_url) else {
            return Ok(ReplaceDecision::NoExisting);
        };
        let answer = term_read_line(&style::dim(format!(
            "{} {} key '{}' (ID: {}) already exists. Replace it? [y/N] ",
            style::yellow("Warning:"),
            label,
            existing.name,
            existing.id
        )))?;
        if matches!(answer.to_lowercase().as_str(), "y" | "yes") {
            Ok(ReplaceDecision::Replace(existing.id.clone()))
        } else {
            println!("Aborted.");
            Ok(ReplaceDecision::Abort)
        }
    }

    /// Append `-2`, `-3`, … to `base` until it no longer collides with an
    /// existing key name, keeping an auto-suggested name unambiguous.
    async fn unique_key_name(&self, base: String) -> String {
        let existing = self.session_store.get_keys().await.unwrap_or_default();
        let taken = |candidate: &str| {
            existing
                .iter()
                .any(|k| k.name.eq_ignore_ascii_case(candidate))
        };
        if !taken(&base) {
            return base;
        }
        (2..=99)
            .map(|n| format!("{base}-{n}"))
            .find(|candidate| !taken(candidate))
            .unwrap_or(base)
    }

    /// A short, unique name derived from a base URL for a `keys add` without
    /// `--name`. Empty (→ short-id display) when nothing sensible can be derived.
    async fn derived_key_name(&self, base_url: &str) -> String {
        match crate::services::known_providers::suggest_name_from_url(base_url) {
            Some(base) => self.unique_key_name(base).await,
            None => String::new(),
        }
    }

    /// Interactively adds an API key
    async fn add_key(
        &self,
        provided_name: Option<&str>,
        add_options: AddKeyOptions<'_>,
    ) -> Result<ExitCode> {
        use std::io;

        fn read_line(prompt: &str) -> io::Result<String> {
            term_read_line(&style::dim(prompt))
        }

        if provided_name.is_some() && add_options.name.is_some() {
            eprintln!(
                "{} Specify the key name either positionally or with --name",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }
        if add_options.base_url == Some(crate::services::cursor_acp::CURSOR_ACP_SENTINEL) {
            eprintln!(
                "{} Cursor uses the cursor-agent flow. Use `aivo keys add cursor` instead of `--base-url cursor`.",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }

        // Defensive: a previously-crashed invocation may have left the terminal
        // in raw mode, which breaks backspace in the first prompt.
        restore_cooked_mode();
        let mut name_was_prompted = false;
        let name = if let Some(n) = add_options.name.or(provided_name) {
            n.to_string()
        } else if add_options.base_url.is_some() && add_options.key.is_some() {
            String::new()
        } else {
            name_was_prompted = true;
            keys_ui::step_header(1, 3, "Name", "a short label for this key");
            read_line("Name (optional): ")?
        };

        let is_starter_name = name == "aivo-starter" || name == "aivo starter";
        // `aivo keys add cursor --api-key sk-…` is the non-interactive shortcut
        // into the cursor flow. Without `--api-key`, "cursor" is treated as a
        // plain key name — users who want OAuth / API-key sub-pickers run
        // `aivo keys add` (no args) and pick Cursor from the provider list.
        let is_cursor_shortcut = name.eq_ignore_ascii_case("cursor")
            && add_options.base_url.is_none()
            && add_options.key.is_some();
        if is_cursor_shortcut {
            return self.add_cursor_interactive(&name, add_options.key).await;
        }
        let interactive =
            add_options.base_url.is_none() && add_options.key.is_none() && !is_starter_name;

        if interactive {
            // Echo the name when it came from arg/flag so the user sees what
            // was used before the provider picker takes over.
            if !name_was_prompted && !name.is_empty() {
                println!("{} {}", style::dim("Name:"), style::cyan(&name));
            }
            return self.interactive_add(&name).await;
        }

        let base_url = if is_starter_name {
            crate::constants::AIVO_STARTER_SENTINEL.to_string()
        } else {
            let detected_url = detect_base_url(&name);
            let mut provided_base_url = add_options.base_url.map(str::to_string);
            loop {
                let value = if let Some(value) = provided_base_url.take() {
                    value
                } else {
                    let prompt = match detected_url {
                        Some(default) => format!("Base URL [{}]: ", default),
                        None => "Base URL: ".to_string(),
                    };
                    let input = read_line(&prompt)?;
                    if input.is_empty() {
                        detected_url.unwrap_or("").to_string()
                    } else {
                        input
                    }
                };
                let picker_hint =
                    "run 'aivo keys add' (no flags) and pick it from the provider list";
                let picker_rejections: &[(&str, &str)] = &[
                    ("copilot", "GitHub Copilot login needs the device flow"),
                    ("ollama", "Ollama setup needs a local installation check"),
                    (
                        crate::services::cursor_acp::CURSOR_ACP_SENTINEL,
                        "Cursor setup needs the cursor-agent flow",
                    ),
                    (
                        crate::services::codex_oauth::CODEX_OAUTH_SENTINEL,
                        "Codex ChatGPT login needs browser auth",
                    ),
                    (
                        crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL,
                        "Claude Code login needs browser auth",
                    ),
                    (
                        crate::services::grok_oauth::GROK_OAUTH_SENTINEL,
                        "SuperGrok login needs the device flow",
                    ),
                    (
                        crate::services::kimi_oauth::KIMI_OAUTH_SENTINEL,
                        "Kimi Code login needs the device flow",
                    ),
                ];
                let reject = |msg: String| -> Option<ExitCode> {
                    eprintln!("{} {msg}", style::red("Error:"));
                    add_options
                        .base_url
                        .is_some()
                        .then_some(ExitCode::UserError)
                };
                if let Some((_, reason)) = picker_rejections.iter().find(|(s, _)| value == *s) {
                    if let Some(code) = reject(format!("{reason} — {picker_hint}.")) {
                        return Ok(code);
                    }
                    continue;
                }
                if value == "aivo-starter" || value == "aivo starter" {
                    if let Some(code) =
                        reject("Use 'aivo keys add aivo-starter' instead.".to_string())
                    {
                        return Ok(code);
                    }
                    continue;
                }
                match validate_base_url(&value) {
                    Ok(()) => break value,
                    Err(msg) => {
                        if let Some(code) = reject(msg.to_string()) {
                            return Ok(code);
                        }
                    }
                }
            }
        };

        // GitHub Copilot: use device flow instead of manual key entry
        if base_url == "copilot" {
            if add_options.key.is_some() {
                eprintln!(
                    "{} GitHub Copilot uses device login; run 'aivo keys add' (no flags) and pick GitHub Copilot.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }

            keys_ui::provider_info(COPILOT_INFO.0, COPILOT_INFO.1);

            // Check for an existing Copilot key and prompt to replace
            let existing_keys = self.session_store.get_keys().await?;
            let existing_copilot_id =
                if let Some(existing) = existing_keys.iter().find(|k| k.base_url == "copilot") {
                    let answer = read_line(&format!(
                        "{} Copilot key '{}' (ID: {}) already exists. Replace it? [y/N] ",
                        style::yellow("Warning:"),
                        existing.name,
                        existing.id
                    ))?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        println!("Aborted.");
                        return Ok(ExitCode::Success);
                    }
                    Some(existing.id.clone())
                } else {
                    None
                };

            let token = crate::services::copilot_auth::device_flow_login().await?;

            // Device flow succeeded — now safe to remove the old key
            if let Some(old_id) = existing_copilot_id {
                self.session_store.delete_key(&old_id).await?;
            }

            let id = self
                .session_store
                .add_key_with_protocol(&name, "copilot", None, &token)
                .await?;
            self.finalize_add(
                &id,
                &name,
                "Provider: GitHub Copilot",
                Some(("aivo run claude", "(uses Copilot subscription)")),
            )
            .await?;

            sync_models_in_background(&id, &base_url);
            return Ok(ExitCode::Success);
        }

        // Ollama: verify installation, no API key needed
        if base_url == "ollama" {
            if add_options.key.is_some() {
                eprintln!(
                    "{} Ollama runs locally without authentication. Do not pass --api-key.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }

            keys_ui::provider_info(OLLAMA_INFO.0, OLLAMA_INFO.1);

            crate::services::ollama::ensure_ready().await?;

            // Check for an existing Ollama key and prompt to replace
            let existing_keys = self.session_store.get_keys().await?;
            let existing_ollama_id =
                if let Some(existing) = existing_keys.iter().find(|k| k.base_url == "ollama") {
                    let answer = read_line(&format!(
                        "{} Ollama key '{}' (ID: {}) already exists. Replace it? [y/N] ",
                        style::yellow("Warning:"),
                        existing.name,
                        existing.id
                    ))?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        println!("Aborted.");
                        return Ok(ExitCode::Success);
                    }
                    Some(existing.id.clone())
                } else {
                    None
                };

            if let Some(old_id) = existing_ollama_id {
                self.session_store.delete_key(&old_id).await?;
            }

            let id = self
                .session_store
                .add_key_with_protocol(&name, "ollama", None, "ollama-local")
                .await?;
            self.finalize_add(
                &id,
                &name,
                "Provider: Ollama (local)",
                Some(("aivo models", "(list local models)")),
            )
            .await?;

            return Ok(ExitCode::Success);
        }

        // Aivo starter: free provider, no API key needed
        if base_url == crate::constants::AIVO_STARTER_SENTINEL {
            keys_ui::provider_info(STARTER_INFO.0, STARTER_INFO.1);

            if let Some(code) = self.notify_if_starter_already_added().await? {
                return Ok(code);
            }

            // Clear the dismissed flag so ensure_starter_key works again
            let _ = self.session_store.set_starter_key_dismissed(false).await;

            let (starter, _) = self
                .session_store
                .ensure_starter_key()
                .await
                .ok_or_else(|| anyhow::anyhow!("Failed to create aivo starter key"))?;
            self.finalize_add(
                &starter.id,
                &name,
                "Provider: aivo starter (free)",
                Some(("aivo code", "(start coding)")),
            )
            .await?;

            return Ok(ExitCode::Success);
        }

        let key = if let Some(key) = add_options.key {
            key.to_string()
        } else {
            loop {
                let input = term_read_secret(&style::dim("API Key: "))?;
                if !input.is_empty() {
                    break input;
                }

                let prompt = style::yellow("Save without an API key?");
                if confirm(&prompt)? {
                    break String::new();
                }
            }
        };

        // No name given: derive one from the URL instead of an opaque short id.
        let name = if name.is_empty() {
            self.derived_key_name(&base_url).await
        } else {
            name
        };

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            &name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;

        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    /// Removes an API key by ID or name
    async fn remove_key(&self, key_id_or_name: Option<&str>, yes: bool) -> Result<ExitCode> {
        let key_to_remove = match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to remove",
                "No keys to remove.",
            )
            .await?
        {
            KeySelection::Key(key) => key,
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        // Show confirmation — display full stored ID (not short form) before a destructive action
        println!("ID:  {}", style::cyan(&key_to_remove.id));
        println!("URL: {}", style::dim(&key_to_remove.base_url));
        println!();

        let confirmed = yes || confirm(&format!("Remove \"{}\"?", key_to_remove.display_name()))?;

        if !confirmed {
            println!("{}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        }

        // Resolve any cursor shadow account before the key is gone, so the
        // shadow dir can be removed once deletion succeeds.
        let cursor_shadow_to_delete =
            if crate::services::cursor_acp::is_cursor_acp_base(&key_to_remove.base_url) {
                self.session_store
                    .get_key_by_id(&key_to_remove.id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|key| crate::services::cursor_acp::cursor_shadow_for_key(&key).ok())
                    .flatten()
            } else {
                None
            };

        if self.session_store.delete_key(&key_to_remove.id).await? {
            let _ = self.session_store.remove_key_stats(&key_to_remove.id).await;
            if let Some(shadow) = cursor_shadow_to_delete {
                let _ = shadow.delete();
            }
            // Remember if the user dismissed the aivo-starter key
            if key_to_remove.base_url == crate::constants::AIVO_STARTER_SENTINEL {
                let _ = self.session_store.set_starter_key_dismissed(true).await;
            }
            println!(
                "{} Removed key: {}",
                style::success_symbol(),
                style::cyan(key_to_remove.display_name())
            );
            Ok(ExitCode::Success)
        } else {
            eprintln!("{} Failed to remove key", style::red("Error:"));
            Ok(ExitCode::UserError)
        }
    }

    async fn resolve_key_selection(
        &self,
        key_id_or_name: Option<&str>,
        prompt: &str,
        empty_message: &str,
    ) -> Result<KeySelection> {
        // Load without decrypting — only metadata is needed for selection.
        let (all_keys, active_key_id) = self.session_store.get_keys_and_active_id_info().await?;

        if all_keys.is_empty() {
            println!("{}", style::dim(empty_message));
            println!("  {}", crate::commands::add_key_cta());
            return Ok(KeySelection::Empty);
        }

        let selected = if let Some(key_id_or_name) = key_id_or_name {
            if let Some(key) = all_keys
                .iter()
                .find(|k| k.id == key_id_or_name || k.short_id() == key_id_or_name)
            {
                Some(key.clone())
            } else {
                let name_matches: Vec<ApiKey> = all_keys
                    .iter()
                    .filter(|k| k.name == key_id_or_name)
                    .cloned()
                    .collect();

                match name_matches.len() {
                    0 => {
                        eprintln!(
                            "{} API key \"{}\" not found",
                            style::red("Error:"),
                            key_id_or_name
                        );
                        eprintln!();
                        eprintln!("{}", style::dim("Run 'aivo keys' to see available keys."));
                        return Ok(KeySelection::NotFound);
                    }
                    1 => Some(name_matches[0].clone()),
                    _ => {
                        println!(
                            "{} Multiple keys found with name \"{}\":",
                            style::yellow("Note:"),
                            key_id_or_name
                        );
                        prompt_pick_key(&name_matches, &[], prompt, 0)?
                    }
                }
            }
        } else {
            let default_idx = active_key_id
                .and_then(|id| all_keys.iter().position(|k| k.id == id))
                .unwrap_or(0);
            prompt_pick_key(&all_keys, &[], prompt, default_idx)?
        };

        match selected {
            Some(key) => Ok(KeySelection::Key(key)),
            None => Ok(KeySelection::Cancelled),
        }
    }

    // Shows usage information.
    pub fn print_help(action: Option<&str>) {
        match action {
            Some("use") => print_help_use(),
            Some("add") => print_help_add(),
            Some("rm" | "remove") => print_help_rm(),
            Some("cat") => print_help_cat(),
            Some("edit") => print_help_edit(),
            Some("reauth") => print_help_reauth(),
            Some("ping") => print_help_ping(),
            Some("reset-route") => print_help_reset_route(),
            Some("export") => print_help_export(),
            Some("import") => print_help_import(),
            _ => print_help_overview(),
        }
    }
}

fn keys_help_row(label: &str, desc: &str) {
    println!(
        "  {}{}",
        style::cyan(format!("{:<24}", label)),
        style::dim(desc)
    );
}

fn print_help_overview() {
    println!("{} aivo keys [action]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Manage API keys: add, remove, activate, and inspect.")
    );
    println!();
    println!("{}", style::bold("Commands:"));
    keys_help_row("list", "List all API keys (default when omitted)");
    keys_help_row("use [id|name]", "Activate a specific API key");
    keys_help_row("cat [id|name]", "Display details for a key");
    keys_help_row("rm [id|name]", "Remove an API key");
    keys_help_row("add [name]", "Add an API key");
    keys_help_row("edit [id|name]", "Edit an API key");
    keys_help_row(
        "reauth [id|name]",
        "Re-authenticate (OAuth re-login or rotate API key)",
    );
    keys_help_row("ping [id|name]", "Health-check API keys (or: aivo ping)");
    keys_help_row(
        "reset-route [id|name]",
        "Reset cached provider routing for a key",
    );
    keys_help_row("export [file]", "Write keys to a password-encrypted file");
    keys_help_row("import <file>", "Merge keys from a password-encrypted file");
    println!();
    println!(
        "{}",
        style::dim(
            "Flags: --ping/--json (list), --name/--base-url/--api-key (add). See <cmd> --help."
        )
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys"));
    println!("  {}", style::dim("aivo keys use openrouter"));
    println!(
        "  {}",
        style::dim("aivo keys add --name abc --base-url https://example.io --api-key sk-...")
    );
}

fn print_help_use() {
    println!("{} aivo keys use [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Activate an API key as the default for run/code/serve (bare opens the picker)."
        )
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys use"));
    println!("  {}", style::dim("aivo keys use openrouter"));
    println!("  {}", style::dim("aivo use openrouter        # shortcut"));
}

fn print_help_add() {
    println!("{} aivo keys add [NAME] [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Add a new API key. With no flags, prompts interactively.")
    );
    println!();
    println!("{}", style::bold("Interactive picker includes:"));
    keys_help_row(
        "Presets",
        "OpenRouter, Anthropic, Groq, Bedrock, … (API key)",
    );
    keys_help_row("OAuth", "Codex, Claude Code, SuperGrok, Kimi Code");
    keys_help_row("Other", "GitHub Copilot, Cursor, Ollama, aivo starter");
    keys_help_row("Custom URL", "Any OpenAI-compatible base URL + API key");
    println!();
    println!("{}", style::bold("Options:"));
    keys_help_row("--name <name>", "Display name (skips the name prompt)");
    keys_help_row(
        "--base-url <url>",
        "Provider base URL (e.g. https://openrouter.ai/api/v1)",
    );
    keys_help_row("--api-key <key>", "Provider API key value");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys add"));
    println!("  {}", style::dim("aivo keys add openrouter"));
    println!("  {}", style::dim("aivo keys add cursor --api-key key_…"));
    println!(
        "  {}",
        style::dim("aivo keys add --name abc --base-url https://example.io --api-key sk-...")
    );
}

fn print_help_rm() {
    println!("{} aivo keys rm [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Remove an API key. Bare `aivo keys rm` opens the picker.")
    );
    println!();
    println!("{}", style::bold("Flags:"));
    keys_help_row("-y, --yes", "Skip the confirmation prompt");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys rm"));
    println!("  {}", style::dim("aivo keys rm openrouter"));
}

fn print_help_cat() {
    println!("{} aivo keys cat [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Show details for a key (name, base URL, masked secret).")
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys cat"));
    println!("  {}", style::dim("aivo keys cat openrouter"));
}

fn print_help_edit() {
    println!("{} aivo keys edit [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Edit an existing key (name, base URL, secret) interactively.")
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys edit"));
    println!("  {}", style::dim("aivo keys edit openrouter"));
}

fn print_help_reauth() {
    println!("{} aivo keys reauth [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Re-authenticate a stored key without removing it: OAuth re-login (codex/claude/grok/kimi) or Cursor login/API key. Bare opens the picker."
        )
    );
    println!();
    println!(
        "{}",
        style::dim("Copilot: remove and re-add. Plain API keys: use `aivo keys edit` to rotate.")
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys reauth"));
    println!("  {}", style::dim("aivo keys reauth codex"));
}

fn print_help_ping() {
    println!(
        "{} aivo keys ping [ID|NAME] [OPTIONS]",
        style::bold("Usage:")
    );
    println!();
    println!(
        "{}",
        style::dim("Health-check API keys against their providers.")
    );
    println!();
    println!("{}", style::bold("Options:"));
    keys_help_row("--all", "Ping every saved key");
    keys_help_row("--json", "Output ping results as JSON");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys ping"));
    println!("  {}", style::dim("aivo keys ping openrouter"));
    println!("  {}", style::dim("aivo ping --all          # shortcut"));
}

fn print_help_reset_route() {
    println!("{} aivo keys reset-route [ID|NAME]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Clear the cached provider routing for a key so the next request re-probes.")
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys reset-route"));
    println!("  {}", style::dim("aivo keys reset-route openrouter"));
}

fn print_help_export() {
    println!("{} aivo keys export [FILE]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Export keys to a password-encrypted file. Run bare for a guided flow.")
    );
    println!();
    println!("{}", style::bold("Flags:"));
    keys_help_row("--ids <a,b,c>", "Export only the given keys (ids or names)");
    keys_help_row("--password-stdin", "Read password from stdin (no prompt)");
    keys_help_row("-y, --yes", "Overwrite an existing file without asking");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys export"));
    println!(
        "  {}",
        style::dim("aivo keys export keys.aivo --ids abc,def")
    );
    println!(
        "  {}",
        style::dim("printf '%s' \"$PW\" | aivo keys export keys.aivo --password-stdin -y")
    );
}

fn print_help_import() {
    println!("{} aivo keys import <FILE|URL>", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Decrypt a keys export and merge into local config. Accepts a local file path or an http(s) URL. Conflicts skip by default."
        )
    );
    println!();
    println!("{}", style::bold("Flags:"));
    keys_help_row("--password-stdin", "Read password from stdin (no prompt)");
    keys_help_row("--overwrite", "Replace existing keys on conflict");
    keys_help_row("--rename", "Keep existing; import conflicts under new ids");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo keys import ~/aivo-backup.aivo"));
    println!(
        "  {}",
        style::dim("aivo keys import https://gist.example.com/raw/abc.aivo")
    );
    println!("  {}", style::dim("aivo keys import keys.aivo --overwrite"));
}

// Formats an API key as a choice string for interactive selectors.
pub(crate) fn format_key_choice(key: &ApiKey) -> String {
    format!(
        "{}  {}  {}",
        style::cyan(format!("{:<3}", key.short_id())),
        key.display_name(),
        style::dim(key_provider_hint(key))
    )
}

// Prompts the user to select a key from the given list. `annotations`
// parallels `keys`: `Some(reason)` disables that row and shows the reason
// as a dim suffix. Pass `&[]` for no annotations.
fn prompt_pick_key(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    let choices: Vec<String> = keys.iter().map(format_key_choice).collect();
    let mut picker = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&choices)
        .default(default);
    if !annotations.is_empty() {
        picker = picker.annotations(annotations.to_vec());
    }
    let selection = picker.interact_opt()?;
    Ok(selection.map(|idx| keys[idx].clone()))
}

pub fn prompt_pick_key_without_activation(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            SessionStore::decrypt_key_secret(&mut key)?;
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

// Picks a key from `keys` and activates it. Returns `Ok(None)` if cancelled.
pub(crate) async fn prompt_select_key(
    session_store: &SessionStore,
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            SessionStore::decrypt_key_secret(&mut key)?;
            session_store.set_active_key(&key.id).await?;
            let preview = display_secret(&key);
            eprintln!(
                "{} Activated key: {} {}",
                style::success_symbol(),
                style::cyan(key.display_name()),
                style::dim(&preview)
            );
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

/// Offers a picker of compatible keys when `bad_key` is an OAuth credential
/// the current command can't use. OAuth keys stay visible but are disabled
/// with an inline reason. `context_phrase` is the user-visible command name
/// inserted into messages (e.g. `"aivo code"` or `"aivo run codex"`).
///
/// Returns `Ok(Some(new_key))` when the user picks a replacement; `Ok(None)`
/// when there's no TTY, no eligible key, or the user cancelled — callers
/// should exit with `ExitCode::UserError`.
pub(crate) async fn swap_incompatible_key(
    session_store: &SessionStore,
    bad_key: &ApiKey,
    compat: crate::services::key_compat::KeyCompatContext,
    context_phrase: &str,
) -> Result<Option<ApiKey>> {
    use std::io::IsTerminal;

    let all_keys = session_store.get_keys().await?;
    let annotations = compat.annotations_for(&all_keys);
    let has_eligible = annotations.iter().any(Option::is_none);

    if !has_eligible || !std::io::stderr().is_terminal() {
        eprintln!(
            "{} Key '{}' is a {} OAuth account — `{}` can't use it.",
            style::red("Error:"),
            bad_key.display_name(),
            bad_key.oauth_kind_label(),
            context_phrase,
        );
        eprintln!(
            "  {} Use `{}` or select a regular API key.",
            style::dim("hint:"),
            bad_key.oauth_tool_hint(),
        );
        return Ok(None);
    }

    eprintln!(
        "{} Key '{}' is a {} OAuth account — pick a regular API key for `{}`.",
        style::yellow("Note:"),
        bad_key.display_name(),
        bad_key.oauth_kind_label(),
        context_phrase,
    );

    prompt_pick_key_without_activation(&all_keys, &annotations, "Select a key", 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::KeysArgs;

    #[test]
    fn wrap_names_breaks_between_names_within_width() {
        let names = ["alpha", "beta", "gamma", "delta"];
        let lines = wrap_names(&names, 20);
        assert_eq!(lines, vec!["alpha, beta, gamma,", "delta"]);
        assert!(lines.iter().all(|l| l.len() <= 20));
    }

    #[test]
    fn wrap_names_single_line_when_it_fits() {
        let lines = wrap_names(&["a", "b"], 80);
        assert_eq!(lines, vec!["a, b"]);
    }

    #[test]
    fn wrap_names_overlong_name_gets_own_line() {
        let long = "x".repeat(40);
        let lines = wrap_names(&["short", &long, "tail"], 20);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1], format!("{long},"));
    }

    #[test]
    fn wrap_names_counts_wide_chars_as_two_columns() {
        // 6 CJK chars = 12 columns per name; two plus separators exceed 20,
        // so they split.
        let lines = wrap_names(&["智谱清言模型", "月之暗面模型"], 20);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn is_http_url_accepts_http_and_https() {
        assert!(is_http_url("http://example.com/x"));
        assert!(is_http_url("https://example.com/x"));
        assert!(is_http_url("HTTPS://example.com/x"));
    }

    #[test]
    fn is_http_url_rejects_other_schemes_and_paths() {
        assert!(!is_http_url("file:///tmp/x"));
        assert!(!is_http_url("ftp://example.com/x"));
        assert!(!is_http_url("./relative.aivo-keys"));
        assert!(!is_http_url("/abs/path"));
        assert!(!is_http_url("~/path"));
        assert!(!is_http_url(""));
    }

    fn keys_args(action: Option<&str>, args: &[&str]) -> KeysArgs {
        KeysArgs {
            action: action.map(str::to_string),
            args: args.iter().map(|s| s.to_string()).collect(),
            name: None,
            base_url: None,
            key: None,
            all: false,
            ping: false,
            json: false,
            ids: Vec::new(),
            password_stdin: false,
            overwrite: false,
            rename: false,
            include_oauth: false,
            yes: false,
        }
    }

    #[test]
    fn test_keys_command_creation() {
        let session_store = SessionStore::new();
        let _command = KeysCommand::new(session_store);
    }

    #[tokio::test]
    async fn test_edit_key_missing_id() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_edit_key_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        store
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &["nonexistent"])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_use_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        // No keys stored — should succeed (prints "No API keys found.")
        let code = cmd.execute(keys_args(Some("use"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_cat_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("cat"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_remove_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("rm"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_remove_key_yes_skips_confirmation() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        store
            .add_key_with_protocol("doomed", "https://api.example.com", None, "sk-1")
            .await
            .unwrap();
        let cmd = KeysCommand::new(store.clone());

        let mut args = keys_args(Some("rm"), &["doomed"]);
        args.yes = true;
        let code = cmd.execute(args).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
        assert!(store.get_keys().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_reset_route_clears_all_routing_fields() {
        use crate::services::session_store::{
            ClaudeProviderProtocol, GeminiProviderProtocol, SessionStore,
        };

        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let key_id = store
            .add_key_with_protocol(
                "myKey",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-1",
            )
            .await
            .unwrap();
        // Populate every routing field that reset-route should clear.
        store
            .set_key_gemini_protocol(&key_id, Some(GeminiProviderProtocol::Google))
            .await
            .unwrap();
        store
            .set_key_claude_path_variant(&key_id, Some("stripped".to_string()))
            .await
            .unwrap();
        store
            .set_key_gemini_path_variant(&key_id, Some("stripped".to_string()))
            .await
            .unwrap();
        store
            .set_key_responses_api_supported(&key_id, Some(true))
            .await
            .unwrap();
        // The v2 per-model routes are the primary thing reset-route must clear.
        store
            .merge_routes(
                &key_id,
                "claude",
                &[(
                    "qwen3.7-max".to_string(),
                    crate::services::route_cache::PersistedRoute {
                        protocol: "anthropic".to_string(),
                        path_variant: String::new(),
                    },
                )],
            )
            .await
            .unwrap();

        let cmd = KeysCommand::new(store.clone());
        let code = cmd
            .execute(keys_args(Some("reset-route"), &[key_id.as_str()]))
            .await;
        assert_eq!(code, crate::errors::ExitCode::Success);

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.claude_protocol, None);
        assert_eq!(reloaded.gemini_protocol, None);
        assert_eq!(reloaded.claude_path_variant, None);
        assert_eq!(reloaded.gemini_path_variant, None);
        assert_eq!(reloaded.responses_api_supported, None);
        assert!(reloaded.protocol_routes.is_empty());
    }

    #[tokio::test]
    async fn test_reset_route_clears_models_cache() {
        use crate::services::models_cache::{ModelsCache, full_catalog_key};
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));
        let base_url = "https://api.example.com";
        let key_id = store
            .add_key_with_protocol(
                "myKey",
                base_url,
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-1",
            )
            .await
            .unwrap();

        let cache = ModelsCache::with_path(temp_dir.path().join("models-cache.json"));
        cache.set(base_url, vec!["gpt-4o".to_string()]).await;
        cache
            .set(&full_catalog_key(base_url), vec!["gpt-4o".to_string()])
            .await;

        let cmd = KeysCommand::with_models_cache(store, cache.clone());
        let code = cmd
            .execute(keys_args(Some("reset-route"), &[key_id.as_str()]))
            .await;
        assert_eq!(code, crate::errors::ExitCode::Success);

        assert!(cache.get(base_url).await.is_none());
        assert!(cache.get(&full_catalog_key(base_url)).await.is_none());
    }

    #[tokio::test]
    async fn test_keys_list_and_ls_actions_list() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        for action in ["list", "ls"] {
            let code = cmd.execute(keys_args(Some(action), &[])).await;
            assert_eq!(code, crate::errors::ExitCode::Success);
        }
    }

    #[tokio::test]
    async fn test_keys_remove_synonym_matches_rm() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        // Empty store: both verbs reach remove_key and fail identically.
        let rm = cmd.execute(keys_args(Some("rm"), &[])).await;
        let remove = cmd.execute(keys_args(Some("remove"), &[])).await;
        assert_eq!(rm, remove);
    }

    #[tokio::test]
    async fn test_add_key_with_flags() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: Some("minimax".to_string()),
                base_url: Some("https://api.minimax.io/anthropic".to_string()),
                key: Some("sk-minimax-test".to_string()),
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "minimax");
        assert_eq!(keys[0].base_url, "https://api.minimax.io/anthropic");
        assert_eq!(keys[0].claude_protocol, None);

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);
        assert_eq!(active.key.as_str(), "sk-minimax-test");
    }

    #[tokio::test]
    async fn test_add_cursor_with_key_stores_shadow_apikey_secret() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: vec!["cursor".to_string()],
                name: None,
                base_url: None,
                key: Some("key_cursor_test".to_string()),
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "cursor");
        assert_eq!(keys[0].base_url, "cursor");

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);

        // Secret encodes a fresh shadow account + the embedded API key so
        // launch_runtime can drive cursor-agent without re-prompting.
        let parsed = crate::services::cursor_acp::parse_cursor_shadow_secret(active.key.as_str())
            .expect("cursor secret must parse as a shadow secret");
        assert!(!parsed.account_id.is_empty());
        assert_eq!(parsed.api_key, Some("key_cursor_test"));

        // Test leaves an empty shadow dir behind under
        // $HOME/.config/aivo/cursor-accounts/<id>/. Clean it up to avoid
        // polluting the developer's machine when tests run repeatedly.
        if let Ok(Some(shadow)) = crate::services::cursor_acp::cursor_shadow_for_key(&active) {
            let _ = shadow.delete();
        }
    }

    #[tokio::test]
    async fn test_add_key_rejects_manual_cursor_base_url() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: Some("manual-cursor".to_string()),
                base_url: Some("cursor".to_string()),
                key: Some("sk-cursor-test".to_string()),
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_rejects_conflicting_name_sources() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: vec!["positional-name".to_string()],
                name: Some("flag-name".to_string()),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                key: Some("sk-or-v1-test".to_string()),
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    fn add_args(base_url: &str, key: &str) -> KeysArgs {
        KeysArgs {
            action: Some("add".to_string()),
            args: Vec::new(),
            name: None,
            base_url: Some(base_url.to_string()),
            key: Some(key.to_string()),
            all: false,
            ping: false,
            json: false,
            ids: Vec::new(),
            password_stdin: false,
            overwrite: false,
            rename: false,
            include_oauth: false,
            yes: false,
        }
    }

    #[tokio::test]
    async fn test_add_key_without_name_derives_name_from_url() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(add_args("https://openrouter.ai/api/v1", "sk-or-1"))
            .await;
        assert_eq!(code, crate::errors::ExitCode::Success);
        let code = cmd
            .execute(add_args("https://openrouter.ai/api/v1", "sk-or-2"))
            .await;
        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].name, "openrouter");
        assert_eq!(keys[0].display_name(), "openrouter");
        assert_eq!(keys[1].name, "openrouter-2");
    }

    #[tokio::test]
    async fn test_add_key_without_name_ip_host_stays_empty() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(add_args("http://127.0.0.1:11434/v1", "sk-local"))
            .await;
        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "");
        assert_eq!(keys[0].display_name(), keys[0].short_id());
    }

    #[tokio::test]
    async fn test_add_key_rejects_ollama_base_url_without_ollama_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("ollama".to_string()),
                key: None,
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_rejects_copilot_base_url_without_copilot_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("copilot".to_string()),
                key: None,
                all: false,
                ping: false,
                json: false,
                ids: Vec::new(),
                password_stdin: false,
                overwrite: false,
                rename: false,
                include_oauth: false,
                yes: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[test]
    fn test_detect_base_url_exact_match() {
        assert_eq!(
            detect_base_url("openrouter"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("deepseek"),
            Some("https://api.deepseek.com/v1")
        );
        assert_eq!(
            detect_base_url("groq"),
            Some("https://api.groq.com/openai/v1")
        );
        assert_eq!(
            detect_base_url("mistral"),
            Some("https://api.mistral.ai/v1")
        );
        assert_eq!(detect_base_url("xai"), Some("https://api.x.ai/v1"));
        assert_eq!(
            detect_base_url("fireworks-ai"),
            Some("https://api.fireworks.ai/inference/v1/")
        );
        assert_eq!(
            detect_base_url("moonshotai"),
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(detect_base_url("minimax"), Some("https://api.minimax.io"));
        assert_eq!(
            detect_base_url("minimax-cn"),
            Some("https://api.minimax.com")
        );
        assert_eq!(
            detect_base_url("vercel"),
            Some("https://ai-gateway.vercel.sh/v1")
        );
    }

    #[test]
    fn test_detect_base_url_case_insensitive() {
        assert_eq!(
            detect_base_url("OpenRouter"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("GROQ"),
            Some("https://api.groq.com/openai/v1")
        );
        assert_eq!(
            detect_base_url("DeepSeek"),
            Some("https://api.deepseek.com/v1")
        );
    }

    #[test]
    fn test_detect_base_url_substring() {
        assert_eq!(
            detect_base_url("my-openrouter-key"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("work_groq"),
            Some("https://api.groq.com/openai/v1")
        );
    }

    #[test]
    fn test_detect_base_url_no_match() {
        assert_eq!(detect_base_url("random"), None);
        assert_eq!(detect_base_url(""), None);
    }

    #[test]
    fn test_validate_base_url() {
        assert!(validate_base_url("https://api.example.com").is_ok());
        assert!(validate_base_url("http://localhost:8080").is_ok());
        assert!(validate_base_url("https://api.example.com/v1/path").is_ok());

        assert!(validate_base_url("ftp://example.com").is_err());
        assert!(validate_base_url("example.com").is_err());
        assert!(validate_base_url("https://").is_err());
        assert!(validate_base_url("https:// example.com").is_err());
        assert!(validate_base_url("https://example.com/path with space").is_err());
        assert!(validate_base_url("https://example.com/v1?key=sk-secret").is_err());
        assert!(validate_base_url("https://example.com/v1#frag").is_err());
    }

    #[test]
    fn test_ping_status_from_http_status_ok() {
        assert_eq!(PingStatus::from_http_status(200), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(204), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_from_http_status_auth_error() {
        assert_eq!(PingStatus::from_http_status(401), PingStatus::AuthError);
        assert_eq!(PingStatus::from_http_status(403), PingStatus::AuthError);
    }

    #[test]
    fn test_ping_status_from_http_status_other() {
        assert_eq!(
            PingStatus::from_http_status(500),
            PingStatus::Error("HTTP 500".to_string())
        );
        assert_eq!(
            PingStatus::from_http_status(429),
            PingStatus::Error("HTTP 429".to_string())
        );
    }

    #[test]
    fn test_ping_status_from_http_status_reachable_wrong_path() {
        assert_eq!(PingStatus::from_http_status(404), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(405), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_icons_and_messages() {
        assert_eq!(PingStatus::Ok.icon(), "✓");
        assert_eq!(PingStatus::Ok.message(), "ok");
        assert_eq!(PingStatus::AuthError.icon(), "✗");
        assert_eq!(PingStatus::AuthError.message(), "auth failed");
        assert_eq!(PingStatus::Unreachable.icon(), "✗");
        assert_eq!(PingStatus::Unreachable.message(), "unreachable");
        assert_eq!(PingStatus::Timeout.icon(), "✗");
        assert_eq!(PingStatus::Timeout.message(), "timeout");
    }

    #[test]
    fn test_ping_status_exit_code() {
        use crate::errors::ExitCode;
        assert_eq!(PingStatus::Ok.exit_code(), ExitCode::Success);
        assert_eq!(PingStatus::AuthError.exit_code(), ExitCode::AuthError);
        assert_eq!(PingStatus::Unreachable.exit_code(), ExitCode::NetworkError);
        assert_eq!(PingStatus::Timeout.exit_code(), ExitCode::NetworkError);
        assert_eq!(
            PingStatus::Error("boom".into()).exit_code(),
            ExitCode::UserError
        );
    }

    #[test]
    fn test_ping_result_empty_name_uses_short_id() {
        let result = PingResult {
            name: "abc".to_string(),
            url: "https://api.openai.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        assert_eq!(result.name, "abc");
        assert_eq!(result.status, PingStatus::Ok);
    }

    #[test]
    fn test_format_key_choice_uses_id_for_unnamed_keys() {
        let key = ApiKey::new_with_protocol(
            "a2b".to_string(),
            String::new(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-test".to_string(),
        );

        let choice = format_key_choice(&key);

        assert!(choice.contains("a2b"));
        assert!(choice.contains("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn key_provider_hint_labels_specials() {
        use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;

        let copilot = ApiKey::new_with_protocol(
            "id".into(),
            "copilot".into(),
            "copilot".into(),
            None,
            "tok".into(),
        );
        assert_eq!(key_provider_hint(&copilot), "GitHub Copilot");

        let oauth = ApiKey::new_with_protocol(
            "id".into(),
            "codex".into(),
            CODEX_OAUTH_SENTINEL.into(),
            None,
            "{}".into(),
        );
        assert_eq!(key_provider_hint(&oauth), "Codex OAuth");

        let url = ApiKey::new_with_protocol(
            "id".into(),
            "groq".into(),
            "https://api.groq.com/openai/v1".into(),
            None,
            "sk".into(),
        );
        assert!(key_provider_hint(&url).contains("api.groq.com"));
    }

    #[test]
    fn key_preview_redacts_by_length() {
        for (input, expected) in [
            ("sk-abc", "sk-..."),
            ("1234567890", "123..."),
            ("sk-abcdefghijklmnop", "sk-abc...mnop"),
            ("", "..."),
        ] {
            assert_eq!(key_preview(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn schemeless_url_detection_splits_urls_from_keys() {
        for url in [
            "api.example.com",
            "api.example.com/v1",
            "www.Example.COM/openai/v1",
            "localhost",
            "localhost:11434/v1",
            "127.0.0.1:8080",
            "192.168.1.10/v1",
            "host.docker.internal:11434",
        ] {
            assert!(looks_like_schemeless_url(url), "should be URL: {url:?}");
        }
        for key in [
            "sk-proj-abc123DEF456ghi789",
            "sk-ant-api03-AbC_dEf-123",
            "gsk_abc123def456",
            "AIzaSyA-1234567890_abcdef",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123XYZ",
            "token+with/slash",
            "example.com.",
        ] {
            assert!(!looks_like_schemeless_url(key), "should be key: {key:?}");
        }
    }

    #[test]
    fn credential_mask_kicks_in_after_five_non_url_chars() {
        let mask = |s: &str| credential_should_mask(&s.chars().collect::<Vec<_>>());
        assert!(!mask("sk-ab"));
        assert!(!mask(""));
        assert!(!mask("https:"));
        assert!(!mask("http:/"));
        assert!(!mask("https://api.example.com"));
        assert!(!mask("api.example.com"));
        assert!(!mask("localhost:11434"));
        assert!(mask("sk-abc"));
        assert!(mask("sk-ant-api03-AbC_dEf"));
        assert!(mask("gsk_abc123"));
        assert!(mask("httpsX"));
    }

    #[test]
    fn key_preview_unicode_safe() {
        // Multi-byte chars: prove we slice on char boundaries, not bytes.
        let key = "🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑";
        let out = key_preview(key);
        assert!(out.contains("..."));
        assert!(out.starts_with("🔑"));
    }

    #[test]
    fn display_secret_labels_oauth_and_copilot() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
        use crate::services::cursor_acp::{CURSOR_ACP_SENTINEL, CURSOR_SHADOW_PREFIX};

        let cursor_shadow_secret = format!("{CURSOR_SHADOW_PREFIX}testaccount1");
        let cases: [(&str, &str, &str); 4] = [
            (
                CLAUDE_OAUTH_SENTINEL,
                "must-not-leak-this-credential-blob",
                "<Claude OAuth>",
            ),
            (
                CODEX_OAUTH_SENTINEL,
                "must-not-leak-this-credential-blob",
                "<Codex OAuth>",
            ),
            ("copilot", "must-not-leak-this-credential-blob", "<Copilot>"),
            (
                CURSOR_ACP_SENTINEL,
                cursor_shadow_secret.as_str(),
                "<Cursor login>",
            ),
        ];
        for (base_url, secret, expected) in cases {
            let key = ApiKey::new_with_protocol(
                "id".to_string(),
                "name".to_string(),
                base_url.to_string(),
                None,
                secret.to_string(),
            );
            let out = display_secret(&key);
            assert_eq!(out, expected, "base_url: {base_url}");
            assert!(!out.contains("must-not-leak"), "base_url: {base_url}");
        }
    }

    #[test]
    fn display_secret_falls_back_to_preview_for_api_keys() {
        let key = ApiKey::new_with_protocol(
            "id".to_string(),
            "name".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-abcdefghijklmnop".to_string(),
        );
        assert_eq!(display_secret(&key), "sk-abc...mnop");
    }

    #[test]
    fn key_metadata_json_excludes_secret() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-must-not-leak".to_string(),
        );
        let payload = key_metadata_json(&key, Some("abc"));
        let s = serde_json::to_string(&payload).unwrap();
        assert!(!s.contains("sk-must-not-leak"));
        assert!(s.contains("\"active\":true"));
        assert_eq!(payload["id"], "abc");
        assert_eq!(payload["name"], "test");
        assert_eq!(payload["base_url"], "https://api.example.com");
    }

    #[test]
    fn key_metadata_json_inactive() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk".to_string(),
        );
        let payload = key_metadata_json(&key, Some("xyz"));
        assert_eq!(payload["active"], false);
        let payload = key_metadata_json(&key, None);
        assert_eq!(payload["active"], false);
    }

    #[test]
    fn key_metadata_json_learned_routing_null_when_unset() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk".to_string(),
        );
        let payload = key_metadata_json(&key, None);
        let learned = &payload["learned"];
        assert_eq!(learned["claude_protocol"], serde_json::Value::Null);
        assert_eq!(learned["gemini_protocol"], serde_json::Value::Null);
        assert_eq!(learned["codex_mode"], serde_json::Value::Null);
        assert_eq!(learned["responses_api_supported"], serde_json::Value::Null);
    }

    #[test]
    fn key_metadata_json_learned_routing_populated() {
        let mut key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk".to_string(),
        );
        key.protocol_routes
            .entry("claude".to_string())
            .or_default()
            .insert(
                "qwen3.7-max".to_string(),
                crate::services::route_cache::PersistedRoute {
                    protocol: "anthropic".to_string(),
                    path_variant: String::new(),
                },
            );
        key.codex_mode = Some(crate::services::session_store::OpenAICompatibilityMode::Router);
        let payload = key_metadata_json(&key, None);
        assert_eq!(
            payload["learned"]["protocol_routes"]["claude"]["qwen3.7-max"]["protocol"],
            "anthropic"
        );
        assert_eq!(payload["learned"]["codex_mode"], "router");
    }

    #[test]
    fn ping_result_json_ok_includes_latency() {
        let result = PingResult {
            name: "test".to_string(),
            url: "https://api.example.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        let payload = ping_result_json(&result);
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["status"], "ok");
        assert_eq!(payload["latency_ms"], 42);
    }

    #[test]
    fn ping_result_json_error_status_keys() {
        for (status, expected_key) in [
            (PingStatus::AuthError, "auth_error"),
            (PingStatus::Unreachable, "unreachable"),
            (PingStatus::Timeout, "timeout"),
            (PingStatus::Error("boom".into()), "error"),
        ] {
            let result = PingResult {
                name: "n".to_string(),
                url: "u".to_string(),
                status,
                latency: None,
            };
            let payload = ping_result_json(&result);
            assert_eq!(payload["ok"], false);
            assert_eq!(payload["status"], expected_key);
            assert!(payload["latency_ms"].is_null());
        }
    }
}
