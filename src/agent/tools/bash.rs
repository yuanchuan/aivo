//! Bash execution: confined/unconfined runners, interactive-command refusal,
//! process-group kill, output streaming.

use super::*;

/// Outcome of a confined `run_bash`: the tool result plus whether the OS
/// sandbox blocked a file write (EPERM/EACCES while a sandbox is active).
/// `sandbox_blocked` lets the engine offer an in-session escape hatch —
/// re-running the command outside the sandbox on approval — instead of
/// surfacing what looks like an ordinary failure. See [`crate::agent::sandbox`].
pub struct BashOutcome {
    pub result: Result<String, String>,
    pub sandbox_blocked: bool,
}

/// Live `run_bash` output chunks for the UI; never changes the final result.
pub type BashProgress = tokio::sync::mpsc::UnboundedSender<String>;

/// Run a shell command with file writes confined to the workspace sandbox.
pub(super) async fn run_bash(args: &Value, cwd: &Path) -> Result<String, String> {
    run_bash_confined(args, cwd, None).await.result
}

/// Like [`run_bash`], but also reports whether the sandbox blocked a write so
/// the engine can offer to escalate (see [`run_bash_unconfined`]).
pub async fn run_bash_confined(
    args: &Value,
    cwd: &Path,
    progress: Option<BashProgress>,
) -> BashOutcome {
    run_bash_inner(args, cwd, true, progress).await
}

/// Run a shell command WITHOUT the workspace sandbox. Reserved for the
/// user-approved escalation of a command the sandbox blocked.
pub async fn run_bash_unconfined(
    args: &Value,
    cwd: &Path,
    progress: Option<BashProgress>,
) -> Result<String, String> {
    run_bash_inner(args, cwd, false, progress).await.result
}

pub(super) fn is_shell_operator(tok: &str) -> bool {
    matches!(tok, "|" | "||" | "&&" | ";" | "&" | "|&" | ";;")
}

pub(super) fn program_basename(prog: &str) -> &str {
    prog.rsplit(['/', '\\']).next().unwrap_or(prog)
}

/// Whether ssh/sftp/telnet carries a remote command (an operand past the
/// destination), which makes it non-interactive.
pub(super) fn ssh_has_remote_command(args: &[String]) -> bool {
    const VALUE_FLAGS: &[&str] = &[
        "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O", "-o", "-p",
        "-Q", "-R", "-S", "-W", "-w",
    ];
    let mut seen_dest = false;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with('-') && a.len() > 1 {
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 1;
            }
            i += 1;
            continue;
        }
        if seen_dest {
            return true;
        }
        seen_dest = true;
        i += 1;
    }
    false
}

/// A command that stalls a shell with no interactive input — the agent's `run_bash`
/// (no TTY) or the chat's `!cmd` (a PTY, but no keystrokes forwarded). One detector,
/// two messages: `agent_message` steers to tools/flags, `user_message` to a terminal.
#[derive(Debug, PartialEq)]
pub(crate) enum InteractiveBlocker {
    Editor(String),
    InteractiveRemote(String),
    Sudo,
    FullScreenMonitor(String),
    Watch,
    TailFollow,
    GitRebaseInteractive,
    GitAddPatch,
    ContainerTty(String),
}

impl InteractiveBlocker {
    pub(crate) fn agent_message(&self) -> String {
        match self {
            Self::Editor(prog) => format!(
                "`{prog}` opens an interactive editor, which can't run under the agent's \
                 non-interactive shell. Edit files with the write/edit tools instead, or ask the \
                 user to run it."
            ),
            Self::InteractiveRemote(prog) => format!(
                "`{prog}` with no remote command opens an interactive session that will block \
                 here. Run a non-interactive form like `{prog} <host> '<command>'`, or ask the \
                 user to run it."
            ),
            Self::Sudo => "`sudo` may prompt for a password on the terminal and block here. Use \
                 `sudo -n` (it fails fast when credentials aren't cached), or ask the user to run \
                 it."
            .into(),
            Self::FullScreenMonitor(prog) => format!(
                "`{prog}` is a full-screen monitor that never exits on its own. Use a one-shot \
                 snapshot like `ps aux` (or `top -b -n1`)."
            ),
            Self::Watch => {
                "`watch` reruns a command forever and never exits. Run the command once instead."
                    .into()
            }
            Self::TailFollow => "`tail -f` follows the file forever and never exits. Read a \
                 bounded slice with `tail -n <N>`, or background it."
                .into(),
            Self::GitRebaseInteractive => "`git rebase -i` opens an interactive editor and can't \
                 run here. Script the rebase non-interactively, or ask the user to run it."
                .into(),
            Self::GitAddPatch => "`git add -p/-i` is interactive and can't run here. Stage paths \
                 explicitly (`git add <path>`) instead."
                .into(),
            Self::ContainerTty(prog) => format!(
                "`{prog}` with an interactive TTY (`-it`) opens a session that will block here. \
                 Drop `-t` and pass the command non-interactively, or ask the user to run it."
            ),
        }
    }

    pub(crate) fn user_message(&self) -> String {
        match self {
            Self::Editor(prog) => format!(
                "`{prog}` is an interactive editor, but `!cmd` forwards no keystrokes to it — it'd \
                 just hang. Run it in a separate terminal."
            ),
            Self::InteractiveRemote(prog) => format!(
                "`{prog}` with no remote command opens an interactive session, which `!cmd` can't \
                 drive (it forwards no keystrokes). Add a command (`{prog} <host> '<cmd>'`), or \
                 run it in a separate terminal."
            ),
            Self::Sudo => "`sudo`'s password prompt can't be answered under `!cmd` (it forwards \
                 no keystrokes). Use `sudo -n`, or run it in a separate terminal."
                .into(),
            Self::FullScreenMonitor(prog) => format!(
                "`{prog}` is a full-screen monitor that never exits on its own. Use a snapshot \
                 like `ps aux` (or `top -b -n1`)."
            ),
            Self::Watch => {
                "`watch` reruns forever and never exits. Run the command once instead.".into()
            }
            Self::TailFollow => "`tail -f` follows forever and never exits under `!cmd`. Use \
                 `tail -n <N>`, or run it in a separate terminal."
                .into(),
            Self::GitRebaseInteractive => "`git rebase -i` opens an interactive editor, which \
                 `!cmd` can't drive. Run it in a separate terminal."
                .into(),
            Self::GitAddPatch => "`git add -p/-i` is interactive, which `!cmd` can't drive. Stage \
                 paths explicitly (`git add <path>`)."
                .into(),
            Self::ContainerTty(prog) => format!(
                "`{prog} -it` opens an interactive session `!cmd` can't drive. Drop `-t` for a \
                 one-shot command, or run it in a separate terminal."
            ),
        }
    }

    /// Whether the chat's `!cmd` should refuse this too. Unlike the agent, `!cmd`
    /// streams PTY output live and esc stops it, so an endless-but-streaming monitor
    /// (`tail -f`, `watch`) is a legit in-chat use there; only the rest (editors,
    /// full-screen TUIs, prompts we can't answer) genuinely can't work.
    pub(crate) fn blocks_bang_cmd(&self) -> bool {
        !matches!(self, Self::Watch | Self::TailFollow)
    }
}

/// Interactive `git` subcommands (`git commit` w/o `-m` is handled by `GIT_EDITOR`).
pub(super) fn blocking_git(args: &[String]) -> Option<InteractiveBlocker> {
    let sub = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .map(String::as_str);
    let has = |flags: &[&str]| args.iter().any(|a| flags.contains(&a.as_str()));
    match sub {
        Some("rebase") if has(&["-i", "--interactive"]) => {
            Some(InteractiveBlocker::GitRebaseInteractive)
        }
        Some("add") if has(&["-p", "--patch", "-i", "--interactive"]) => {
            Some(InteractiveBlocker::GitAddPatch)
        }
        _ => None,
    }
}

pub(super) fn blocking_program(prog: &str, args: &[String]) -> Option<InteractiveBlocker> {
    let has = |flags: &[&str]| args.iter().any(|a| flags.contains(&a.as_str()));
    match prog {
        "vim" | "vi" | "nvim" | "nano" | "pico" | "emacs" | "joe" | "mcedit" => {
            Some(InteractiveBlocker::Editor(prog.to_string()))
        }
        "ssh" | "sftp" | "telnet" if !ssh_has_remote_command(args) => {
            Some(InteractiveBlocker::InteractiveRemote(prog.to_string()))
        }
        "sudo" if !has(&["-n", "--non-interactive", "-A", "--askpass"]) => {
            Some(InteractiveBlocker::Sudo)
        }
        "top" | "htop" if !has(&["-b", "--batch"]) => {
            Some(InteractiveBlocker::FullScreenMonitor(prog.to_string()))
        }
        "watch" => Some(InteractiveBlocker::Watch),
        "tail" if has(&["-f", "-F", "--follow"]) => Some(InteractiveBlocker::TailFollow),
        "git" => blocking_git(args),
        "docker" | "podman" | "kubectl"
            if has(&["-it", "-ti"]) || (has(&["-i", "--interactive"]) && has(&["-t", "--tty"])) =>
        {
            Some(InteractiveBlocker::ContainerTty(prog.to_string()))
        }
        _ => None,
    }
}

/// The blocker if any pipeline segment of `command` would stall a no-input shell.
/// Conservative + argument-aware, so ordinary slow commands still run to the timeout.
pub(crate) fn interactive_block_reason(command: &str) -> Option<InteractiveBlocker> {
    let tokens = shlex::split(command)?;
    for seg in tokens.split(|t| is_shell_operator(t)) {
        let Some((prog, rest)) = seg.split_first() else {
            continue;
        };
        if let Some(blocker) = blocking_program(program_basename(prog), rest) {
            return Some(blocker);
        }
    }
    None
}

/// Tree-kills the child on drop unless disarmed — covers timeout, wait error,
/// and Esc-cancel. `kill_on_drop` alone would orphan grandchildren.
pub(super) struct GroupKillGuard(Option<u32>);

impl GroupKillGuard {
    fn new(pid: Option<u32>) -> Self {
        Self(pid)
    }
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for GroupKillGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            // Child leads its own group, so pid == pgid.
            #[cfg(unix)]
            crate::agent::jobs::signal_group(pid as i32, libc::SIGKILL);
            #[cfg(windows)]
            crate::agent::jobs::taskkill_tree_detached(pid);
            #[cfg(not(any(unix, windows)))]
            let _ = pid;
        }
    }
}

/// Read a child pipe to EOF, forwarding complete-UTF-8 chunks to `progress`
/// (a split code point carries over rather than rendering as �).
pub(super) async fn drain_pipe<R: tokio::io::AsyncRead + Unpin>(
    reader: Option<R>,
    progress: Option<&BashProgress>,
) -> Vec<u8> {
    let mut collected = Vec::new();
    let Some(mut reader) = reader else {
        return collected;
    };
    use tokio::io::AsyncReadExt;
    let mut chunk = [0u8; 8192];
    let mut sent = 0; // bytes of `collected` already forwarded
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                collected.extend_from_slice(&chunk[..n]);
                let Some(tx) = progress else { continue };
                let pending = &collected[sent..];
                let valid = match std::str::from_utf8(pending) {
                    Ok(s) => s,
                    Err(e) => {
                        let (valid, _) = pending.split_at(e.valid_up_to());
                        std::str::from_utf8(valid).unwrap_or("")
                    }
                };
                if !valid.is_empty() {
                    let _ = tx.send(valid.to_string());
                    sent += valid.len();
                }
            }
        }
    }
    collected
}

pub(super) async fn run_bash_inner(
    args: &Value,
    cwd: &Path,
    confined: bool,
    progress: Option<BashProgress>,
) -> BashOutcome {
    let early = |result| BashOutcome {
        result,
        sandbox_blocked: false,
    };
    let command = match arg_str(args, "command") {
        Ok(c) => c,
        Err(e) => return early(Err(e)),
    };
    // Refuse blockers up front, so they don't burn the whole timeout as dead air.
    if let Some(blocker) = interactive_block_reason(command) {
        return early(Err(blocker.agent_message()));
    }
    let timeout = arg_u64(args, "timeout")
        .unwrap_or(BASH_DEFAULT_TIMEOUT)
        .min(BASH_MAX_TIMEOUT);
    // Confine file writes to the workspace (where supported); reads and network
    // stay open. The unconfined path runs the bare shell — reserved for the
    // user-approved escalation of a blocked command. See agent::sandbox.
    let spawn = if confined {
        crate::agent::sandbox::wrap_shell(command, cwd)
    } else {
        crate::agent::sandbox::bare_shell(command)
    };
    // Std builder first so the anti-hang hardening is shared with background jobs
    // (one drift-proof site); the tokio conversion preserves args/env/stdio.
    let mut builder = std::process::Command::new(&spawn.program);
    builder.args(&spawn.args).current_dir(cwd);
    crate::agent::sandbox::harden_headless(&mut builder);
    builder.stdout(Stdio::piped()).stderr(Stdio::piped());
    // Own pgroup: a descendant probing /dev/tty (ssh passphrase, pinentry) stops
    // on SIGTTIN/SIGTTOU instead of stealing the TUI's keystrokes or raw mode,
    // and the tree is killable as `-pgid` (leader pgid == pid, as in jobs.rs).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        builder.process_group(0);
    }
    let mut builder = tokio::process::Command::from(builder);
    builder.kill_on_drop(true);
    let mut child = match builder.spawn() {
        Ok(c) => c,
        Err(e) => return early(Err(format!("spawn shell: {e}"))),
    };
    let mut tree_kill = GroupKillGuard::new(child.id());
    // Drain pipes concurrently with the wait — no deadlock on a full pipe.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let run = async {
        let (out_bytes, err_bytes, status) = tokio::join!(
            drain_pipe(stdout_pipe, progress.as_ref()),
            drain_pipe(stderr_pipe, progress.as_ref()),
            child.wait(),
        );
        (status, out_bytes, err_bytes)
    };
    let (status, out_bytes, err_bytes) =
        match tokio::time::timeout(Duration::from_secs(timeout), run).await {
            Ok((Ok(status), out_bytes, err_bytes)) => (status, out_bytes, err_bytes),
            Ok((Err(e), ..)) => return early(Err(format!("run command: {e}"))),
            Err(_) => {
                // `tree_kill` fires on the early return below, killing the subtree.
                return early(Err(format!(
                    "command timed out after {timeout}s and was killed. If it was waiting on \
                     interactive input (a password, prompt, or editor), a REPL, or a \
                     long-running server/watcher that never exits on its own, it can't run \
                     under the agent's non-interactive shell — use a non-interactive form, \
                     re-run it with `background: true` and poll it with `check_job`, or ask \
                     the user to run it. If it's just slow, retry with a larger `timeout` \
                     (max {BASH_MAX_TIMEOUT}).",
                )));
            }
        };
    // Completed normally: leave deliberately-detached survivors alone.
    tree_kill.disarm();
    let mut out = String::new();
    let stdout = String::from_utf8_lossy(&out_bytes);
    let stderr = String::from_utf8_lossy(&err_bytes);
    if !stdout.trim().is_empty() {
        out.push_str(&stdout);
    }
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&stderr);
    }
    let code = status.code().unwrap_or(-1);
    let mut sandbox_blocked = false;
    if code != 0 {
        out.push_str(&format!("\n[exit {code}]"));
        // A blocked write surfaces as EPERM ("Operation not permitted", macOS
        // seatbelt) or EACCES/EPERM ("Permission denied", Linux Landlock). Flag
        // it so the engine can offer to re-run the command outside the sandbox on
        // approval, and tell the model this was a confinement block — not a real
        // failure — so it doesn't give up and ask the user to run it by hand.
        if confined
            && crate::agent::sandbox::active()
            && (out.contains("Operation not permitted") || out.contains("Permission denied"))
        {
            sandbox_blocked = true;
            out.push_str(
                "\n[note: blocked by the workspace write-sandbox, not a real command \
failure — it wrote outside the agent's workspace. The user can approve re-running it \
outside the sandbox; don't fall back to telling the user to run it by hand. To drop \
confinement for the whole session, relaunch aivo with AIVO_AGENT_NO_SANDBOX=1.]",
            );
        }
    }
    if out.is_empty() {
        out.push_str("(no output)");
    }
    let result = if out.len() > MAX_OUTPUT || out.lines().count() > MAX_OUTPUT_LINES {
        let spilled = spill_full_output(&out);
        let mut capped = cap_tail(out);
        if let Some(path) = spilled {
            capped.push_str(&format!("\n[full output: {}]", path.display()));
        }
        capped
    } else {
        out
    };
    BashOutcome {
        result: Ok(result),
        sandbox_blocked,
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn disarm_clears_the_armed_pid() {
        let mut g = GroupKillGuard::new(Some(1234));
        assert_eq!(g.0, Some(1234));
        g.disarm();
        assert_eq!(g.0, None, "disarm must clear the pid");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn guard_tree_kills_child_on_drop() {
        let child = tokio::process::Command::new("cmd")
            .args(["/C", "ping", "-n", "60", "127.0.0.1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn child");
        let pid = child.id().expect("child pid");
        assert!(crate::services::system_env::is_pid_alive(pid));
        drop(GroupKillGuard::new(Some(pid)));
        // Detached taskkill is async; poll for death rather than fixed-sleeping.
        let mut dead = false;
        for _ in 0..50 {
            if !crate::services::system_env::is_pid_alive(pid) {
                dead = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(dead, "guard drop must tree-kill the child");
        let _ = child;
    }
}

// --- web tool ---
