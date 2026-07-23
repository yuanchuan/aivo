//! Wraps `claude setup-token` to capture a long-lived Claude Code OAuth
//! token per account. The token is the only stored secret; Anthropic
//! rotates it server-side, so aivo doesn't track refresh/expiry.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::Command;

/// Sentinel stored in `ApiKey.base_url` for Claude Code OAuth entries.
pub const CLAUDE_OAUTH_SENTINEL: &str = "claude-oauth";

/// Beta flag Anthropic requires on OAuth-authenticated inference requests.
pub const ANTHROPIC_OAUTH_BETA: &str = "oauth-2025-04-20";

/// Native upstream for Claude-subscription passthrough requests.
/// `AIVO_CLAUDE_OAUTH_UPSTREAM` overrides it (test seam).
pub fn upstream_base_url() -> String {
    std::env::var("AIVO_CLAUDE_OAUTH_UPSTREAM")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaudeOAuthCredential {
    pub token: String,
    pub created_at: DateTime<Utc>,
}

impl ClaudeOAuthCredential {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize ClaudeOAuthCredential")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse ClaudeOAuthCredential JSON")
    }
}

/// Extracts the Claude Code OAuth token from the captured stdout of
/// `claude setup-token`. The output format isn't a published contract, so
/// the parser strips ANSI and picks the last line matching a token shape
/// (alphanumeric + `-` / `_`, byte length ≥ 20).
pub fn extract_token_from_setup_output(stdout: &str) -> Option<String> {
    let cleaned = crate::services::ansi::strip_ansi(stdout);
    cleaned
        .lines()
        .rev()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .find(|l| looks_like_token(l))
        .map(|l| l.to_string())
}

fn looks_like_token(s: &str) -> bool {
    s.len() >= 20
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[derive(Debug)]
pub enum SetupTokenError {
    ClaudeNotFound,
    // stderr was inherited during the spawn, so the user has already seen it.
    NonZeroExit { status: String },
    EmptyOutput,
    Other(anyhow::Error),
}

impl std::fmt::Display for SetupTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetupTokenError::ClaudeNotFound => write!(
                f,
                "`claude` CLI not found on PATH — install Claude Code first (npm i -g @anthropic-ai/claude-code)"
            ),
            SetupTokenError::NonZeroExit { status } => {
                write!(f, "`claude setup-token` exited with {status}")
            }
            SetupTokenError::EmptyOutput => write!(
                f,
                "`claude setup-token` produced no parseable token on stdout"
            ),
            SetupTokenError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SetupTokenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SetupTokenError::Other(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<anyhow::Error> for SetupTokenError {
    fn from(e: anyhow::Error) -> Self {
        SetupTokenError::Other(e)
    }
}

/// Spawns `claude setup-token` and captures the token from its stdout.
/// stdin/stderr are inherited so the user can complete the browser OAuth;
/// the spawn is Enter-gated on a TTY so the browser doesn't steal focus.
pub async fn spawn_setup_token_and_capture() -> Result<ClaudeOAuthCredential, SetupTokenError> {
    prompt_before_browser_open();
    spawn_setup_token_with_binary("claude").await
}

fn prompt_before_browser_open() {
    use std::io::{BufRead, IsTerminal, Write as _};
    if !std::io::stdin().is_terminal() {
        return;
    }
    eprintln!();
    eprint!(
        "Press {} to run `claude setup-token` (a browser tab will open) ",
        crate::style::keycap(" Enter ")
    );
    let _ = std::io::stderr().flush();
    let mut buf = String::new();
    let _ = std::io::stdin().lock().read_line(&mut buf);
}

/// Resolve to a spawnable path so Windows finds npm's `claude.cmd` shim
/// (`CreateProcess` only auto-appends `.exe`); bare-name fallback preserves
/// the `ClaudeNotFound` path.
fn resolve_program(binary: &str, dirs: &[std::path::PathBuf]) -> std::ffi::OsString {
    crate::services::path_search::find_in_dirs(binary, dirs)
        .map(|p| p.into_os_string())
        .unwrap_or_else(|| binary.into())
}

async fn spawn_setup_token_with_binary(
    binary: &str,
) -> Result<ClaudeOAuthCredential, SetupTokenError> {
    let program = resolve_program(binary, &crate::services::path_search::collect_path_dirs());
    let mut cmd = Command::new(program);
    cmd.arg("setup-token")
        .stdin(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdout(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(SetupTokenError::ClaudeNotFound);
        }
        Err(e) => {
            return Err(SetupTokenError::Other(
                anyhow::Error::new(e).context(format!("spawn `{} setup-token`", binary)),
            ));
        }
    };

    let stdout_handle = child.stdout.take().ok_or_else(|| {
        SetupTokenError::Other(anyhow::anyhow!("claude setup-token stdout pipe missing"))
    })?;
    let stdout_task = tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        let mut reader = stdout_handle;
        reader.read_to_end(&mut buf).await.map(|_| buf)
    });

    let status = child.wait().await.map_err(|e| {
        SetupTokenError::Other(anyhow::Error::new(e).context("wait on claude setup-token"))
    })?;

    let stdout_bytes = stdout_task
        .await
        .map_err(|e| SetupTokenError::Other(anyhow::anyhow!("join stdout reader: {e}")))?
        .map_err(|e| {
            SetupTokenError::Other(anyhow::Error::new(e).context("read claude setup-token stdout"))
        })?;
    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();

    if !status.success() {
        return Err(SetupTokenError::NonZeroExit {
            status: format!("{status}"),
        });
    }

    let token = extract_token_from_setup_output(&stdout).ok_or(SetupTokenError::EmptyOutput)?;
    Ok(ClaudeOAuthCredential {
        token,
        created_at: Utc::now(),
    })
}

/// Test seam: lets tests pass a stub binary path instead of real `claude`.
///
/// Retries on `SetupTokenError::Other` to absorb the Linux ETXTBSY race where
/// fork+execve can fail because another concurrent test thread inherited a
/// writable fd at fork time (kernel reports os error 26). Production callers
/// don't hit this — they exec a long-installed `claude` binary, not a stub
/// we wrote microseconds ago.
#[cfg(test)]
pub async fn spawn_setup_token_with_binary_for_test(
    binary: &std::path::Path,
) -> Result<ClaudeOAuthCredential, SetupTokenError> {
    let path = binary.to_str().expect("binary path is utf-8");
    let mut backoff_ms = 25;
    for _ in 0..5 {
        match spawn_setup_token_with_binary(path).await {
            Err(SetupTokenError::Other(_)) => {
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 2;
                continue;
            }
            other => return other,
        }
    }
    spawn_setup_token_with_binary(path).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    static SPAWN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[cfg(unix)]
    fn write_executable_stub(path: &std::path::Path, body: &[u8]) {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(body).unwrap();
        f.sync_all().unwrap();
        drop(f);
        let mut perms = std::fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).unwrap();
    }

    #[test]
    fn credential_json_roundtrip() {
        let c = ClaudeOAuthCredential {
            token: "sk-ant-oat01-fake".into(),
            created_at: Utc::now(),
        };
        let json = c.to_json().unwrap();
        let back = ClaudeOAuthCredential::from_json(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn sentinel_value_is_stable() {
        assert_eq!(CLAUDE_OAUTH_SENTINEL, "claude-oauth");
    }

    #[test]
    fn extracts_token_from_clean_output() {
        let stdout = "Please visit https://...\n\nToken:\nsk-ant-oat01-abcdefghijklmnop\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn extracts_token_with_ansi_codes() {
        let stdout = "\x1b[32mSuccess!\x1b[0m\n\x1b[1msk-ant-oat01-abcdefghijklmnop\x1b[0m\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn extracts_token_with_trailing_whitespace() {
        let stdout = "sk-ant-oat01-abcdefghijklmnop   \n\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn ignores_human_readable_lines_after_token() {
        let stdout = "sk-ant-oat01-abcdefghijklmnop\nSave this token somewhere safe.\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn returns_none_when_no_plausible_token() {
        assert!(extract_token_from_setup_output("").is_none());
        assert!(extract_token_from_setup_output("Error: login cancelled\n").is_none());
        assert!(extract_token_from_setup_output("tooshort\n").is_none());
    }

    #[test]
    fn extracts_token_after_osc_title_sequence() {
        // Ink emits `ESC ] 0 ; claude BEL` around the terminal title.
        let stdout = "\x1b]0;claude\x07sk-ant-oat01-abcdefghijklmnop\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn tolerates_two_byte_escape_inline_with_token() {
        // Inline placement forces strip_ansi to consume both bytes of the
        // escape, not just ESC — otherwise `=` would glue to the token.
        let stdout = "\x1b=sk-ant-oat01-abcdefghijklmnop\n";
        let token = extract_token_from_setup_output(stdout).unwrap();
        assert_eq!(token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[test]
    fn length_boundary_at_twenty() {
        let at_19 = "a".repeat(19);
        let at_20 = "a".repeat(20);
        assert!(extract_token_from_setup_output(&at_19).is_none());
        assert_eq!(
            extract_token_from_setup_output(&at_20).as_deref(),
            Some(at_20.as_str())
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_returns_claude_not_found_for_missing_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let err = spawn_setup_token_with_binary_for_test(&missing)
            .await
            .unwrap_err();
        assert!(matches!(err, SetupTokenError::ClaudeNotFound));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_extracts_token_from_stub_stdout() {
        let _guard = SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let stub = tmp.path().join("claude-stub.sh");
        write_executable_stub(
            &stub,
            b"#!/usr/bin/env sh\nprintf 'Follow the browser prompt...\\nsk-ant-oat01-abcdefghijklmnop\\n'\n",
        );

        let cred = spawn_setup_token_with_binary_for_test(&stub).await.unwrap();
        assert_eq!(cred.token, "sk-ant-oat01-abcdefghijklmnop");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_errors_on_non_zero_exit() {
        let _guard = SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let stub = tmp.path().join("claude-fail.sh");
        write_executable_stub(&stub, b"#!/usr/bin/env sh\nexit 7\n");

        let err = spawn_setup_token_with_binary_for_test(&stub)
            .await
            .unwrap_err();
        assert!(matches!(err, SetupTokenError::NonZeroExit { .. }));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_errors_on_empty_stdout() {
        let _guard = SPAWN_LOCK.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let stub = tmp.path().join("claude-empty.sh");
        write_executable_stub(&stub, b"#!/usr/bin/env sh\nexit 0\n");

        let err = spawn_setup_token_with_binary_for_test(&stub)
            .await
            .unwrap_err();
        assert!(matches!(err, SetupTokenError::EmptyOutput));
    }

    #[test]
    fn resolve_program_falls_back_to_bare_name_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let got = resolve_program("claude", &[dir.path().to_path_buf()]);
        assert_eq!(got, std::ffi::OsString::from("claude"));
    }

    #[cfg(windows)]
    #[test]
    fn resolve_program_prefers_cmd_shim_on_windows() {
        // CreateProcess can only spawn the `.cmd`, not npm's extensionless shim.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("claude"), "bash shim").unwrap();
        let cmd = dir.path().join("claude.cmd");
        std::fs::write(&cmd, "@echo off\r\n").unwrap();
        let got = resolve_program("claude", &[dir.path().to_path_buf()]);
        // PATHEXT casing varies by runner; compare case-folded.
        assert_eq!(
            std::path::Path::new(&got).to_string_lossy().to_lowercase(),
            cmd.to_string_lossy().to_lowercase(),
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn spawn_extracts_token_from_cmd_stub() {
        let tmp = tempfile::tempdir().unwrap();
        let stub = tmp.path().join("claude-stub.cmd");
        std::fs::write(&stub, "@echo off\r\necho sk-ant-oat01-abcdefghijklmnop\r\n").unwrap();
        let cred = spawn_setup_token_with_binary_for_test(&stub).await.unwrap();
        assert_eq!(cred.token, "sk-ant-oat01-abcdefghijklmnop");
    }
}
