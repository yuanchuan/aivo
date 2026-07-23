//! Best-effort OS sandbox for the agent's `run_bash` tool. The goal is narrow:
//! confine a shell command's file WRITES to the workspace (cwd) plus temp dirs
//! and the common dev-tool caches, while leaving reads, process exec, and the
//! network open — the agent is still expected to fetch live data and inspect the
//! system (see the engine's system prompt). This is the safety counterpart to
//! the heuristic destructive-command gate: the heuristic catches `rm -rf` inside
//! the workspace (which the sandbox allows), the sandbox catches a stray write
//! to `/etc` or `~/.ssh` (which the heuristic misses).
//!
//! Backends:
//! - **macOS** via `sandbox-exec` (Apple seatbelt) — an external wrapper binary
//!   spawned around the shell.
//! - **Linux** via Landlock (kernel 5.13+). Landlock has no external wrapper —
//!   the ruleset must be installed by syscall *in the process* before running
//!   the shell — so `wrap_shell` re-executes the aivo binary as a hidden
//!   `__agent-sandbox` subcommand (dispatched in `run::run`) which installs the
//!   ruleset and then spawns the shell (Landlock confinement is inherited by
//!   children). Degrades to no confinement on kernels without Landlock.
//! - **Windows**: no-op for now (writes are NOT confined). Windows has no
//!   path-allowlist write-confinement primitive comparable to seatbelt/Landlock —
//!   restricted tokens / integrity levels gate by ACL not by path (and would
//!   break ordinary writes), Job Objects govern CPU/memory not the filesystem,
//!   and AppContainer (the nearest fit) is heavyweight and brittle for an
//!   arbitrary PowerShell command. The heuristic destructive-command gate still
//!   applies. AppContainer is the eventual path if pursued.
//!
//! Default-on where supported; opt out everywhere with `AIVO_AGENT_NO_SANDBOX=1`.

use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};

/// Extra writable roots from `--add-dir`, set once at CLI startup. Process-wide
/// because confinement is process-level anyway (seatbelt profile / Landlock argv).
static EXTRA_WRITE_ROOTS: OnceLock<Vec<PathBuf>> = OnceLock::new();

/// Register `--add-dir` roots (first caller wins; callers validate existence).
pub fn set_extra_write_roots(dirs: Vec<PathBuf>) {
    let _ = EXTRA_WRITE_ROOTS.set(dirs);
}

pub fn extra_write_roots() -> &'static [PathBuf] {
    EXTRA_WRITE_ROOTS.get().map(Vec::as_slice).unwrap_or(&[])
}

/// Confinement level for the agent's shell (`run_bash` writes/network).
/// Under `ReadOnly`, `tools::execute` also refuses the in-process edit tools.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SandboxProfile {
    /// No confinement (legacy `AIVO_AGENT_NO_SANDBOX=1`).
    Off,
    /// Shell writes confined to workspace + temp + dev caches (default).
    #[default]
    Workspace,
    /// No writes anywhere (temp scratch only); child network blocked.
    ReadOnly,
    /// Writes confined to workspace + temp (no caches); child network blocked.
    Strict,
}

impl SandboxProfile {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Some(Self::Off),
            "workspace" | "default" | "on" => Some(Self::Workspace),
            "read-only" | "readonly" | "ro" => Some(Self::ReadOnly),
            "strict" => Some(Self::Strict),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Workspace => "workspace",
            Self::ReadOnly => "read-only",
            Self::Strict => "strict",
        }
    }

    /// The accepted `--sandbox` values, for CLI help/validation.
    pub const VALUES: &'static [&'static str] = &["off", "workspace", "read-only", "strict"];

    /// Child-network denial is macOS-only (Landlock can't gate network), so the
    /// only non-test caller is the macOS SBPL builder.
    #[cfg(any(test, target_os = "macos"))]
    fn deny_network(self) -> bool {
        matches!(self, Self::ReadOnly | Self::Strict)
    }
}

static SANDBOX_PROFILE: OnceLock<SandboxProfile> = OnceLock::new();

/// Register the `--sandbox` profile (first caller wins).
pub fn set_sandbox_profile(profile: SandboxProfile) {
    let _ = SANDBOX_PROFILE.set(profile);
}

/// Force read-only unless the user picked a profile via flag or env — used by
/// `--best-of-n`, whose parallel candidates share the working tree and would
/// otherwise race each other's writes (losers' edits are never rolled back).
pub fn set_read_only_unless_configured() {
    if std::env::var_os("AIVO_AGENT_SANDBOX").is_none() && !legacy_no_sandbox() {
        let _ = SANDBOX_PROFILE.set(SandboxProfile::ReadOnly);
    }
}

/// Resolution order: `--sandbox` flag > `AIVO_AGENT_SANDBOX` >
/// legacy `AIVO_AGENT_NO_SANDBOX` (→ `Off`) > `Workspace`.
pub fn current_profile() -> SandboxProfile {
    if let Some(p) = SANDBOX_PROFILE.get() {
        return *p;
    }
    if let Ok(v) = std::env::var("AIVO_AGENT_SANDBOX")
        && let Some(p) = SandboxProfile::parse(&v)
    {
        return p;
    }
    if legacy_no_sandbox() {
        return SandboxProfile::Off;
    }
    SandboxProfile::Workspace
}

/// A note when this platform can't confine writes; `None` where a sandbox
/// backend exists or the user opted out (`Off`). Windows enforces no profile,
/// so it warns — more sharply for `read-only`/`strict` (network too).
pub fn confinement_notice() -> Option<&'static str> {
    confinement_notice_for(current_profile())
}

/// Split from `confinement_notice` so the mapping is testable without the global.
fn confinement_notice_for(profile: SandboxProfile) -> Option<&'static str> {
    #[cfg(windows)]
    {
        match profile {
            SandboxProfile::Off => None,
            SandboxProfile::Workspace => Some(
                "Windows: shell writes are not sandbox-confined — the destructive-command \
confirm is the only write guard",
            ),
            SandboxProfile::ReadOnly | SandboxProfile::Strict => Some(
                "Windows: --sandbox is not enforced — the shell can write anywhere and reach \
the network; the destructive-command confirm is the only write guard",
            ),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = profile;
        None
    }
}

/// Set by `run::run` in the real CLI binary to signal that this process
/// dispatches the hidden `__agent-sandbox` subcommand, so `wrap_shell` may
/// relaunch it for Landlock confinement. Stays false in test / embedding
/// binaries that never run that entrypoint — relaunching one of *them* as
/// `__agent-sandbox` (which they don't handle) makes every Linux `run_bash`
/// fail with the harness's own "Unrecognized option: 'workspace'".
#[cfg(target_os = "linux")]
static RELAUNCH_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable the Landlock self-relaunch for this process. Called once, early, by
/// `run::run` (the entrypoint that handles the `__agent-sandbox` subcommand).
#[cfg(target_os = "linux")]
pub fn enable_landlock_relaunch() {
    RELAUNCH_ENABLED.store(true, Ordering::Relaxed);
}

/// The program + args `run_bash` should actually spawn for a given shell command
/// — either a sandbox wrapper around the shell, or the bare shell when no
/// sandbox applies.
pub struct ShellInvocation {
    pub program: String,
    pub args: Vec<String>,
}

/// Anti-hang hardening shared by every agent shell spawn: stdin closed,
/// prompt-capable tools forced non-interactive. Grow it here, not per call site.
pub(crate) fn harden_headless(cmd: &mut std::process::Command) {
    cmd.env("GIT_TERMINAL_PROMPT", "0")
        .stdin(std::process::Stdio::null());
    // Unix-only: `true`/`cat` aren't launchable as editor/pager on Windows.
    #[cfg(unix)]
    cmd.env("GIT_EDITOR", "true").env("PAGER", "cat");
}

/// Whether a write-confining sandbox is active for this process. Used by
/// `run_bash` to add a hint when a command likely failed because the sandbox
/// blocked a write.
pub fn active() -> bool {
    if disabled() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        Path::new(SANDBOX_EXEC).exists()
    }
    #[cfg(target_os = "linux")]
    {
        landlock_supported()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

fn disabled() -> bool {
    current_profile() == SandboxProfile::Off
}

/// Legacy `AIVO_AGENT_NO_SANDBOX` opt-out (any value other than empty/`0`).
fn legacy_no_sandbox() -> bool {
    std::env::var("AIVO_AGENT_NO_SANDBOX")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Build the spawn target for `command`, applying the OS sandbox when available.
/// `cwd` is the workspace root writes are confined to.
pub fn wrap_shell(command: &str, cwd: &Path) -> ShellInvocation {
    #[cfg(target_os = "macos")]
    if active() {
        return ShellInvocation {
            program: SANDBOX_EXEC.to_string(),
            args: vec![
                "-p".to_string(),
                macos_profile(cwd, current_profile()),
                "sh".to_string(),
                "-c".to_string(),
                command.to_string(),
            ],
        };
    }

    // Linux: relaunch ourselves as the hidden `__agent-sandbox` subcommand, which
    // installs a Landlock ruleset (confining writes to `cwd` + caches) and then
    // runs the shell — but only when the real CLI enabled it (see
    // `RELAUNCH_ENABLED`); a test/embedding binary doesn't handle the subcommand,
    // so relaunching it would just fail. Falls through to the bare shell when off
    // or `current_exe` can't be resolved.
    #[cfg(target_os = "linux")]
    if active()
        && RELAUNCH_ENABLED.load(Ordering::Relaxed)
        && let Ok(exe) = std::env::current_exe()
    {
        return landlock_relaunch(
            exe.to_string_lossy().into_owned(),
            cwd,
            command,
            current_profile(),
        );
    }

    // `cwd` is consulted only by the macOS/Linux sandbox backends above; on other
    // targets (Windows) there's no path-confinement, so it's intentionally unused.
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let _ = cwd;
    bare_shell(command)
}

/// The relaunch invocation that runs `command`'s shell under Landlock: start this
/// binary again as the hidden `__agent-sandbox` subcommand, which installs the
/// ruleset then `sh -c`'s the command. Split out so the wiring is unit-testable
/// without touching the global relaunch flag (which concurrent tests share).
#[cfg(target_os = "linux")]
fn landlock_relaunch(
    exe: String,
    cwd: &Path,
    command: &str,
    profile: SandboxProfile,
) -> ShellInvocation {
    // First `--workspace` = cwd, extras are `--add-dir` roots; roots + profile
    // must ride the argv (the OnceLock isn't inherited across the re-exec).
    let mut args = vec![
        "__agent-sandbox".to_string(),
        "--profile".to_string(),
        profile.as_str().to_string(),
        "--workspace".to_string(),
        cwd.to_string_lossy().into_owned(),
    ];
    for root in extra_write_roots() {
        args.push("--workspace".to_string());
        args.push(root.to_string_lossy().into_owned());
    }
    args.extend([
        "--".to_string(),
        "sh".to_string(),
        "-c".to_string(),
        command.to_string(),
    ]);
    ShellInvocation { program: exe, args }
}

/// The plain shell invocation with no sandbox wrapper. Used by `wrap_shell` when
/// no sandbox applies, and by `run_bash`'s escalation path when the user
/// approves re-running a sandbox-blocked command outside the workspace.
pub fn bare_shell(command: &str) -> ShellInvocation {
    // PowerShell on Windows, not `cmd`: the agent (and the model driving it) leans
    // on POSIX-ish commands, and PowerShell's aliases/cmdlets (`ls`, `cat`,
    // `Select-String`) cover far more of them than `cmd` does. The engine's system
    // prompt tells the model which shell it has via `shell_label`. POSIX `sh`
    // everywhere else.
    if cfg!(windows) {
        ShellInvocation {
            program: "powershell.exe".to_string(),
            args: vec![
                "-NoProfile".to_string(),
                "-Command".to_string(),
                command.to_string(),
            ],
        }
    } else {
        ShellInvocation {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), command.to_string()],
        }
    }
}

/// Human name of the shell `run_bash` (and the TUI `!cmd`) spawn commands through
/// on this platform, injected into the engine's system prompt so the model writes
/// commands in the right syntax. Must stay in sync with [`bare_shell`].
pub fn shell_label() -> &'static str {
    if cfg!(windows) {
        "PowerShell"
    } else {
        "POSIX sh"
    }
}

// ---------------------------------------------------------------------------
// macOS (seatbelt) backend
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Extra writable roots when `cwd` is a LINKED worktree: its `.git` is a file into
/// the parent repo's `.git/worktrees/<name>`, and git writes land there + the shared
/// object store, both outside `cwd` — so `git add`/`commit` inside an isolated
/// worktree would hit EPERM. A normal repo's `.git` dir is under `cwd` → empty.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn git_metadata_roots(cwd: &Path) -> Vec<PathBuf> {
    let mut dir = Some(cwd);
    for _ in 0..64 {
        let Some(d) = dir else { break };
        let dotgit = d.join(".git");
        let Ok(meta) = std::fs::symlink_metadata(&dotgit) else {
            dir = d.parent();
            continue;
        };
        if !meta.is_file() {
            return Vec::new(); // real `.git` dir = repo boundary, nothing extra
        }
        // Grant the pointed-to worktree gitdir and its `.git` common dir.
        let Some(gitdir) = std::fs::read_to_string(&dotgit).ok().and_then(|t| {
            t.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("gitdir:")
                    .map(|p| PathBuf::from(p.trim()))
            })
        }) else {
            return Vec::new();
        };
        let common = gitdir
            .ancestors()
            .find(|a| a.file_name().is_some_and(|n| n == ".git"))
            .map(Path::to_path_buf)
            .unwrap_or_else(|| gitdir.clone());
        return vec![gitdir, common];
    }
    Vec::new()
}

/// A seatbelt (SBPL) profile: allow everything, optionally deny network, then
/// deny all file writes and re-allow the profile's writable set (last matching
/// rule wins). Reads / exec stay open from `(allow default)`.
#[cfg(target_os = "macos")]
fn macos_profile(cwd: &Path, profile: SandboxProfile) -> String {
    // Temp scratch stays writable under every profile — many read-only tools
    // still write to $TMPDIR / /dev/null.
    let mut writable: Vec<String> = vec![
        "/tmp".into(),
        "/private/tmp".into(),
        "/var/folders".into(),
        "/private/var/folders".into(),
        "/dev".into(),
    ];
    if profile == SandboxProfile::Workspace {
        // Package-manager prefixes so `brew`, etc. keep working.
        writable.push("/usr/local".into());
        writable.push("/opt/homebrew".into());
    }
    // Seatbelt matches the resolved path, so a symlinked cwd needs both forms.
    if profile != SandboxProfile::ReadOnly {
        writable.push(cwd.to_string_lossy().into_owned());
        if let Ok(canon) = cwd.canonicalize() {
            writable.push(canon.to_string_lossy().into_owned());
        }
        // A linked worktree's git metadata lives under the parent repo (see helper).
        for root in git_metadata_roots(cwd) {
            writable.push(root.to_string_lossy().into_owned());
            if let Ok(canon) = root.canonicalize() {
                writable.push(canon.to_string_lossy().into_owned());
            }
        }
        for root in extra_write_roots() {
            writable.push(root.to_string_lossy().into_owned());
            if let Ok(canon) = root.canonicalize() {
                writable.push(canon.to_string_lossy().into_owned());
            }
        }
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        writable.push(tmp.to_string_lossy().into_owned());
    }
    if profile == SandboxProfile::Workspace
        && let Some(home) = crate::services::system_env::home_dir()
    {
        // Dev-tool caches, deliberately NOT `~/.config`: that holds aivo's own
        // encrypted key store (`~/.config/aivo`) and every app's config, which
        // the agent shouldn't be able to silently rewrite. A command that
        // genuinely needs to write there hits the escalation prompt instead.
        for sub in [
            ".cache",
            ".cargo",
            ".rustup",
            ".npm",
            ".gradle",
            ".m2",
            ".cocoapods",
            "Library/Caches",
        ] {
            writable.push(home.join(sub).to_string_lossy().into_owned());
        }
    }

    let mut profile_str = String::from("(version 1)\n(allow default)\n");
    if profile.deny_network() {
        // The agent's own LLM/web calls are in-process, unaffected by this.
        profile_str.push_str("(deny network*)\n");
    }
    profile_str.push_str("(deny file-write*)\n(allow file-write*\n");
    for path in writable {
        let trimmed = path.trim_end_matches('/');
        if trimmed.is_empty() {
            continue;
        }
        profile_str.push_str(&format!("    (subpath \"{}\")\n", sbpl_escape(trimmed)));
    }
    profile_str.push_str(")\n");
    profile_str
}

/// Escape a path for an SBPL double-quoted string literal.
#[cfg(target_os = "macos")]
fn sbpl_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// Linux (Landlock) backend
// ---------------------------------------------------------------------------

/// The same write-allowlist as the macOS profile, but with Linux paths and
/// filtered to entries that actually exist (Landlock errors on a rule for a
/// missing path). Pure function — unit-testable without the kernel feature.
#[cfg(target_os = "linux")]
fn linux_writable_paths(cwd: &Path, profile: SandboxProfile) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = vec![
        PathBuf::from("/tmp"),
        PathBuf::from("/var/tmp"),
        PathBuf::from("/dev"),
    ];
    if profile == SandboxProfile::Workspace {
        // Package-manager prefix.
        candidates.push(PathBuf::from("/usr/local"));
    }
    if profile != SandboxProfile::ReadOnly {
        candidates.push(cwd.to_path_buf());
        if let Ok(canon) = cwd.canonicalize() {
            candidates.push(canon);
        }
        // A linked worktree's git metadata lives under the parent repo (see helper).
        candidates.extend(git_metadata_roots(cwd));
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        candidates.push(PathBuf::from(tmp));
    }
    // Per-user runtime dir (usually /run/user/<uid>).
    if let Some(run) = std::env::var_os("XDG_RUNTIME_DIR") {
        candidates.push(PathBuf::from(run));
    }
    if profile == SandboxProfile::Workspace
        && let Some(home) = crate::services::system_env::home_dir()
    {
        // Dev-tool caches, deliberately NOT `~/.config`: that holds aivo's own
        // encrypted key store (`~/.config/aivo`) and every app's config, which
        // the agent shouldn't be able to silently rewrite. A command that
        // genuinely needs to write there hits the escalation prompt instead.
        for sub in [
            ".cache",
            ".cargo",
            ".rustup",
            ".npm",
            ".gradle",
            ".m2",
            ".local/share",
        ] {
            candidates.push(home.join(sub));
        }
    }
    // Landlock add_rule fails on a non-existent path; only keep present ones.
    candidates.retain(|p| p.exists());
    candidates
}

/// Whether the running kernel supports Landlock. Probes by *creating* a ruleset
/// fd with a hard compatibility requirement (so an unsupported kernel reports
/// failure rather than a silent best-effort no-op); creating the fd does NOT
/// restrict this process — only `restrict_self` would — so the probe is safe.
/// Cached: the kernel capability can't change within a process lifetime.
#[cfg(target_os = "linux")]
fn landlock_supported() -> bool {
    use std::sync::OnceLock;
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        use landlock::{ABI, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr};
        Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_all(ABI::V1))
            .and_then(|r| r.create())
            .is_ok()
    })
}

/// Install a Landlock ruleset on the current process confining file writes to
/// `cwd` + the cache allowlist. Best-effort: the highest ABI the kernel supports
/// is negotiated and unsupported rights are dropped; on any failure it returns
/// `false` (caller degrades to running unconfined). Only WRITE accesses are
/// handled, so reads and exec stay open; network is never restricted.
#[cfg(target_os = "linux")]
fn apply_landlock(workspaces: &[String], profile: SandboxProfile) -> bool {
    use landlock::{
        ABI, AccessFs, CompatLevel, Compatible, PathBeneath, PathFd, Ruleset, RulesetAttr,
        RulesetCreatedAttr, RulesetStatus,
    };
    let abi = ABI::V1;
    let write = AccessFs::from_write(abi);
    let created = Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(write)
        .and_then(|r| r.create());
    let mut created = match created {
        Ok(c) => c,
        Err(_) => return false,
    };
    // First workspace = cwd (carries the standard writable set); the rest are
    // `--add-dir` roots.
    let cwd = workspaces.first().map(String::as_str).unwrap_or(".");
    let mut paths = linux_writable_paths(Path::new(cwd), profile);
    paths.extend(workspaces.iter().skip(1).map(PathBuf::from));
    for path in paths {
        let Ok(fd) = PathFd::new(&path) else {
            continue; // skip a path that vanished between the exists() check and now
        };
        created = match created.add_rule(PathBeneath::new(fd, write)) {
            Ok(c) => c,
            Err(_) => return false,
        };
    }
    matches!(
        created.restrict_self(),
        Ok(status) if !matches!(status.ruleset, RulesetStatus::NotEnforced)
    )
}

/// Split the `__agent-sandbox` argv into the workspace path and the shell argv
/// after `--`. Factored out for unit testing (the entry point below diverges).
/// `raw_args` is the full process argv (`[exe, "__agent-sandbox", …]`).
#[cfg(target_os = "linux")]
fn parse_sandbox_child_args(raw_args: &[String]) -> (SandboxProfile, Vec<String>, Vec<String>) {
    let mut profile = SandboxProfile::Workspace;
    let mut workspaces = Vec::new();
    let mut rest = Vec::new();
    let mut i = 2; // skip exe + subcommand
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--profile" => {
                if let Some(p) = raw_args.get(i + 1).and_then(|v| SandboxProfile::parse(v)) {
                    profile = p;
                }
                i += 2;
            }
            "--workspace" => {
                if let Some(w) = raw_args.get(i + 1) {
                    workspaces.push(w.clone());
                }
                i += 2;
            }
            "--" => {
                rest = raw_args[i + 1..].to_vec();
                break;
            }
            _ => i += 1,
        }
    }
    (profile, workspaces, rest)
}

/// Entry point for the hidden `aivo __agent-sandbox` re-exec (dispatched in
/// `run::run` before clap). Installs the Landlock ruleset (best-effort, degrades
/// silently) and runs the shell as a child — Landlock confinement is inherited,
/// so the child shell is confined — then exits with the shell's status. Never
/// returns.
#[cfg(target_os = "linux")]
pub fn run_sandbox_child(raw_args: &[String]) -> ! {
    let (profile, workspaces, rest) = parse_sandbox_child_args(raw_args);
    if rest.is_empty() {
        eprintln!("aivo: __agent-sandbox: no command after `--`");
        std::process::exit(127);
    }
    let cwd = workspaces
        .first()
        .cloned()
        .unwrap_or_else(|| ".".to_string());
    // Best-effort confinement; if Landlock is unavailable we still run the shell.
    let _ = apply_landlock(&workspaces, profile);
    let status = std::process::Command::new(&rest[0])
        .args(&rest[1..])
        .current_dir(&cwd)
        .status();
    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("aivo: __agent-sandbox: failed to run {}: {e}", rest[0]);
            std::process::exit(127);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_os = "macos"))]
mod macos_tests {
    use super::*;

    #[test]
    fn profile_confines_writes_to_workspace() {
        let profile = macos_profile(Path::new("/Users/x/proj"), SandboxProfile::Workspace);
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("(subpath \"/Users/x/proj\")"));
        // Temp is always writable (macOS $TMPDIR lives under /var/folders).
        assert!(profile.contains("/private/var/folders"));
        // Reads/exec/network are not denied.
        assert!(profile.contains("(allow default)"));
        assert!(!profile.contains("(deny file-read"));
        assert!(!profile.contains("(deny network"));
        // Package prefixes writable under Workspace only.
        assert!(profile.contains("/opt/homebrew"));
    }

    #[test]
    fn read_only_profile_denies_workspace_writes_and_network() {
        let profile = macos_profile(Path::new("/Users/x/proj"), SandboxProfile::ReadOnly);
        // Workspace is NOT re-allowed for writing.
        assert!(!profile.contains("(subpath \"/Users/x/proj\")"));
        // Network is denied for the shell's children.
        assert!(profile.contains("(deny network*)"));
        // Temp scratch stays writable so read-only tools can still run.
        assert!(profile.contains("/private/var/folders"));
        // Reads still open.
        assert!(profile.contains("(allow default)"));
        assert!(!profile.contains("(deny file-read"));
    }

    #[test]
    fn strict_profile_confines_to_workspace_but_denies_network_and_caches() {
        let profile = macos_profile(Path::new("/Users/x/proj"), SandboxProfile::Strict);
        // Workspace is writable (untrusted code still needs to build in-tree).
        assert!(profile.contains("(subpath \"/Users/x/proj\")"));
        // Network denied; package prefixes NOT writable.
        assert!(profile.contains("(deny network*)"));
        assert!(!profile.contains("/opt/homebrew"));
    }

    #[test]
    fn sbpl_escape_handles_quotes_and_backslashes() {
        assert_eq!(sbpl_escape(r#"/a/b"c\d"#), r#"/a/b\"c\\d"#);
    }

    #[test]
    fn wrap_shell_uses_sandbox_exec_when_active() {
        // Only assert the wrapper shape when the sandbox is actually active in
        // this environment (it can be disabled via env).
        let inv = wrap_shell("echo hi", Path::new("/tmp"));
        if active() {
            assert_eq!(inv.program, SANDBOX_EXEC);
            assert_eq!(inv.args[0], "-p");
            assert_eq!(inv.args[2], "sh");
            assert_eq!(inv.args.last().unwrap(), "echo hi");
        } else {
            assert_eq!(inv.program, "sh");
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod linux_tests {
    use super::*;

    #[test]
    fn writable_paths_include_present_workspace_and_temp_only() {
        // /tmp always exists; assert it's present and macOS-only paths aren't.
        let paths = linux_writable_paths(Path::new("/tmp"), SandboxProfile::Workspace);
        assert!(paths.iter().any(|p| p == Path::new("/tmp")));
        assert!(!paths.iter().any(|p| p.starts_with("/private")));
        assert!(!paths.iter().any(|p| p == Path::new("/opt/homebrew")));
        // Every returned path exists (the filter held).
        assert!(paths.iter().all(|p| p.exists()));
    }

    #[test]
    fn read_only_profile_excludes_workspace_from_writable() {
        // Workspace cwd must not become writable; temp scratch stays.
        let paths = linux_writable_paths(Path::new("/usr"), SandboxProfile::ReadOnly);
        assert!(!paths.iter().any(|p| p == Path::new("/usr")));
        assert!(paths.iter().any(|p| p == Path::new("/tmp")));
    }

    #[test]
    fn parse_child_args_extracts_workspace_and_command() {
        let raw: Vec<String> = [
            "aivo",
            "__agent-sandbox",
            "--workspace",
            "/x",
            "--",
            "sh",
            "-c",
            "echo hi",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (profile, ws, rest) = parse_sandbox_child_args(&raw);
        assert_eq!(profile, SandboxProfile::Workspace);
        assert_eq!(ws, vec!["/x"]);
        assert_eq!(rest, vec!["sh", "-c", "echo hi"]);
    }

    #[test]
    fn parse_child_args_reads_profile() {
        let raw: Vec<String> = [
            "aivo",
            "__agent-sandbox",
            "--profile",
            "strict",
            "--workspace",
            "/x",
            "--",
            "sh",
            "-c",
            "echo hi",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (profile, ws, _) = parse_sandbox_child_args(&raw);
        assert_eq!(profile, SandboxProfile::Strict);
        assert_eq!(ws, vec!["/x"]);
    }

    #[test]
    fn landlock_relaunch_has_subcommand_shape() {
        // The relaunch invocation's shape, tested directly (no global flag, so it
        // can't race concurrent run_bash tests into relaunching the test binary).
        let inv = landlock_relaunch(
            "/usr/bin/aivo".to_string(),
            Path::new("/tmp"),
            "echo hi",
            SandboxProfile::Strict,
        );
        assert_eq!(inv.program, "/usr/bin/aivo");
        assert_eq!(inv.args[0], "__agent-sandbox");
        assert!(inv.args.iter().any(|a| a == "--workspace"));
        assert!(
            inv.args
                .windows(2)
                .any(|w| w[0] == "--profile" && w[1] == "strict")
        );
        assert!(inv.args.iter().any(|a| a == "--"));
        assert_eq!(inv.args.last().unwrap(), "echo hi");
    }

    #[test]
    fn wrap_shell_is_bare_until_relaunch_is_enabled() {
        // The relaunch flag is off in the test harness (only `run::run` sets it),
        // so `wrap_shell` must NOT relaunch the test binary — it falls through to
        // the bare shell. This is the guard that keeps `run_bash` working on Linux
        // CI, where Landlock is active but the binary isn't the real CLI.
        let inv = wrap_shell("echo hi", Path::new("/tmp"));
        assert_eq!(inv.program, "sh");
        assert_eq!(inv.args, vec!["-c".to_string(), "echo hi".to_string()]);
    }
}

#[cfg(test)]
mod shell_tests {
    use super::*;

    /// `bare_shell` always passes the command through as its LAST arg, and
    /// `shell_label` names the program it picks — the two must agree per platform
    /// (the system prompt relies on the label matching the real shell).
    #[test]
    fn bare_shell_and_label_agree_for_this_platform() {
        let inv = bare_shell("echo hi");
        assert_eq!(inv.args.last().unwrap(), "echo hi");
        if cfg!(windows) {
            assert_eq!(inv.program, "powershell.exe");
            assert_eq!(shell_label(), "PowerShell");
            assert!(inv.args.iter().any(|a| a == "-Command"));
        } else {
            assert_eq!(inv.program, "sh");
            assert_eq!(shell_label(), "POSIX sh");
            assert_eq!(inv.args[0], "-c");
        }
    }

    #[test]
    fn profile_parse_accepts_aliases_and_rejects_junk() {
        assert_eq!(SandboxProfile::parse("off"), Some(SandboxProfile::Off));
        assert_eq!(SandboxProfile::parse("none"), Some(SandboxProfile::Off));
        assert_eq!(
            SandboxProfile::parse(" Workspace "),
            Some(SandboxProfile::Workspace)
        );
        assert_eq!(
            SandboxProfile::parse("readonly"),
            Some(SandboxProfile::ReadOnly)
        );
        assert_eq!(
            SandboxProfile::parse("read-only"),
            Some(SandboxProfile::ReadOnly)
        );
        assert_eq!(
            SandboxProfile::parse("STRICT"),
            Some(SandboxProfile::Strict)
        );
        assert_eq!(SandboxProfile::parse("bogus"), None);
        // Round-trips through as_str for the canonical spellings.
        for v in SandboxProfile::VALUES {
            assert_eq!(SandboxProfile::parse(v).unwrap().as_str(), *v);
        }
    }

    #[test]
    fn only_confining_profiles_deny_network() {
        assert!(SandboxProfile::ReadOnly.deny_network());
        assert!(SandboxProfile::Strict.deny_network());
        assert!(!SandboxProfile::Workspace.deny_network());
        assert!(!SandboxProfile::Off.deny_network());
    }

    #[cfg(windows)]
    #[test]
    fn confinement_notice_is_profile_aware_on_windows() {
        assert!(confinement_notice_for(SandboxProfile::Off).is_none());
        assert!(confinement_notice_for(SandboxProfile::Workspace).is_some());
        assert!(
            confinement_notice_for(SandboxProfile::ReadOnly)
                .unwrap()
                .contains("network")
        );
        assert!(
            confinement_notice_for(SandboxProfile::Strict)
                .unwrap()
                .contains("network")
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn confinement_notice_is_silent_off_windows() {
        assert!(confinement_notice_for(SandboxProfile::Strict).is_none());
        assert!(confinement_notice_for(SandboxProfile::Off).is_none());
    }
}
