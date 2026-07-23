//! Background shell jobs: `run_bash {"background": true}` detaches a long-running
//! command (dev server, watcher) into a [`JobTable`] and returns a job id + log file;
//! `check_job` polls or tree-kills it. The table is app-owned (survives engine rebuilds)
//! and torn down at exit — `kill_on_drop(false)` makes `kill`/`kill_all` the one cleanup
//! path. Confinement matches foreground `run_bash`; no escalation flow (a detached spawn
//! returns before a sandbox block is observable — it shows in the log instead).

use crate::agent::protocol::ToolSpec;
use crate::agent::sandbox;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// Cap on concurrently RUNNING jobs — a runaway-loop backstop; a spawn past it errs.
const MAX_RUNNING_JOBS: usize = 16;
/// Finished jobs kept queryable before the oldest are pruned (bounds table growth).
const MAX_FINISHED_JOBS: usize = 32;
const TAIL_BYTES: usize = 8_000;
const TAIL_LINES: usize = 200;
const TERM_GRACE: Duration = Duration::from_millis(1_500);
const KILL_WAIT: Duration = Duration::from_millis(1_000);
const KILL_POLL: Duration = Duration::from_millis(50);

/// Shared handle to the process table; cloned into each engine via `set_jobs`.
pub type SharedJobs = Arc<JobTable>;

pub enum JobStatus {
    Running,
    /// Exited on its own — `Some(code)`, or `None` for signal death.
    Exited(Option<i32>),
    Killed,
}

struct Job {
    id: String,
    command: String,
    /// Absolute, frozen at spawn (survives a `set_logs_root` re-root of NEW jobs).
    log_path: PathBuf,
    started_at: Instant,
    ended_at: Option<Instant>,
    child: tokio::process::Child,
    /// Process-group id (== pid via `process_group(0)`) for tree-kill.
    #[cfg(unix)]
    pgid: i32,
    pid: u32,
    status: JobStatus,
    /// The model has seen this job finish — gates `drain_finished_notices`.
    notified: bool,
}

impl Job {
    fn runtime(&self) -> Duration {
        self.ended_at
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.started_at)
    }
}

struct Inner {
    jobs: Vec<Job>,
    next_id: u64,
    logs_root: PathBuf,
}

impl Inner {
    /// Reap every finished-but-still-Running job (never overwrites `Killed`).
    fn reap_all(&mut self) {
        for job in &mut self.jobs {
            reap(job);
        }
    }

    fn running_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Running))
            .count()
    }

    /// Drop the oldest finished jobs beyond [`MAX_FINISHED_JOBS`]; running jobs stay.
    fn prune_finished(&mut self) {
        let finished = self
            .jobs
            .iter()
            .filter(|j| !matches!(j.status, JobStatus::Running))
            .count();
        if finished <= MAX_FINISHED_JOBS {
            return;
        }
        let mut to_drop = finished - MAX_FINISHED_JOBS;
        self.jobs.retain(|j| {
            if to_drop > 0 && !matches!(j.status, JobStatus::Running) {
                to_drop -= 1;
                false
            } else {
                true
            }
        });
    }

    /// Unknown-id error listing the known jobs, so the model can self-correct.
    fn unknown_id_error(&self, id: &str) -> String {
        if self.jobs.is_empty() {
            format!("no background job '{id}' — no jobs have been started.")
        } else {
            let known: Vec<String> = self
                .jobs
                .iter()
                .map(|j| format!("{} ({})", j.id, status_word(&j.status)))
                .collect();
            format!(
                "no background job '{id}'. Known jobs: {}.",
                known.join(", ")
            )
        }
    }
}

/// All critical sections are non-blocking syscalls under a std mutex, never held across
/// an await — so the sync render path reads `running_count` without `block_on`.
pub struct JobTable {
    inner: Mutex<Inner>,
}

impl JobTable {
    /// `logs_root` `None` → `temp_dir()/aivo-jobs-<pid>`; created lazily at first spawn.
    pub fn new(logs_root: Option<PathBuf>) -> SharedJobs {
        Arc::new(JobTable {
            inner: Mutex::new(Inner {
                jobs: Vec::new(),
                next_id: 1,
                logs_root: logs_root.unwrap_or_else(default_logs_root),
            }),
        })
    }

    /// Re-root where NEW jobs log (`/new` + resume); running jobs keep their absolute paths.
    pub fn set_logs_root(&self, root: PathBuf) {
        self.inner.lock().unwrap().logs_root = root;
    }

    /// The current logs dir (for one-shot temp cleanup).
    pub fn logs_root(&self) -> PathBuf {
        self.inner.lock().unwrap().logs_root.clone()
    }

    /// Running-job count; reaps as it counts (the render tick doubles as the reaper).
    pub fn running_count(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        inner.reap_all();
        inner.running_count()
    }

    /// One `id: command` line per still-running job (reaps first); folded into each
    /// request so the model doesn't claim a live server stopped.
    pub fn running_snapshot(&self) -> Vec<String> {
        let mut inner = self.inner.lock().unwrap();
        inner.reap_all();
        inner
            .jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Running))
            .map(|j| format!("{}: {}", j.id, j.command))
            .collect()
    }

    /// Spawn `command` detached; `Ok` is the formatted tool result (id + log + hint).
    pub fn spawn(&self, command: &str, cwd: &Path) -> Result<String, String> {
        let mut inner = self.inner.lock().unwrap();
        inner.reap_all();
        inner.prune_finished();
        let running = inner.running_count();
        if running >= MAX_RUNNING_JOBS {
            return Err(format!(
                "too many background jobs running ({running}/{MAX_RUNNING_JOBS}) — stop one with \
check_job {{\"id\": \"…\", \"kill\": true}} before starting another."
            ));
        }
        std::fs::create_dir_all(&inner.logs_root)
            .map_err(|e| format!("create job log dir: {e}"))?;
        let seq = inner.next_id;
        let id = format!("j{seq}");
        let log_path = inner.logs_root.join(format!("{id}.log"));
        let log_file =
            std::fs::File::create(&log_path).map_err(|e| format!("create job log: {e}"))?;
        let log_err = log_file
            .try_clone()
            .map_err(|e| format!("clone job log handle: {e}"))?;

        // Skips `interactive_block_reason`: servers/watchers are the point, and with
        // stdin null a stray prompt hits EOF and exits rather than hanging.
        let inv = sandbox::wrap_shell(command, cwd);
        let mut cmd = std::process::Command::new(&inv.program);
        cmd.args(&inv.args).current_dir(cwd);
        sandbox::harden_headless(&mut cmd);
        cmd.stdout(log_file) // shared file description → offset-safe interleave
            .stderr(log_err);
        #[cfg(unix)]
        cmd.process_group(0); // leader pgid == pid → tree-kill via -pgid
        let mut cmd = tokio::process::Command::from(cmd);
        cmd.kill_on_drop(false); // detached BY DESIGN — kill/kill_all is the one cleanup path
        let child = cmd
            .spawn()
            .map_err(|e| format!("spawn background job: {e}"))?;
        let pid = child.id().unwrap_or(0);

        inner.next_id += 1;
        inner.jobs.push(Job {
            id: id.clone(),
            command: command.to_string(),
            log_path: log_path.clone(),
            started_at: Instant::now(),
            ended_at: None,
            child,
            #[cfg(unix)]
            pgid: pid as i32,
            pid,
            status: JobStatus::Running,
            notified: false,
        });

        Ok(format!(
            "started background job {id} (pid {pid}): {command}\n\
log: {}\n\
Poll it with check_job {{\"id\": \"{id}\"}}; stop it with check_job {{\"id\": \"{id}\", \"kill\": true}}. \
You'll also get a notice at a later step when it finishes — no need to busy-poll.\n\
Note: file writes are sandbox-confined to the workspace like any run_bash; if the log shows \
\"Operation not permitted\"/\"Permission denied\", re-run in the foreground to be offered the \
unsandboxed escalation.",
            log_path.display()
        ))
    }

    /// Status + runtime + a bounded log tail; unknown id → Err listing knowns.
    pub fn check(&self, id: &str) -> Result<String, String> {
        let mut inner = self.inner.lock().unwrap();
        inner.reap_all();
        let Some(job) = inner.jobs.iter_mut().find(|j| j.id == id) else {
            return Err(inner.unknown_id_error(id));
        };
        if !matches!(job.status, JobStatus::Running) {
            job.notified = true; // the model just observed the finish
        }
        let job = &*job;
        let mut out = status_line(job);
        out.push_str(&format!("\nlog: {}", job.log_path.display()));
        out.push_str("\n--- log tail ---\n");
        let tail = read_tail(&job.log_path);
        if tail.trim().is_empty() {
            // Empty doesn't mean stuck: stdout can be block-buffered with the
            // startup line still unflushed, so say so rather than imply failure.
            out.push_str(match job.status {
                JobStatus::Running => {
                    "(no output captured yet — the program may be buffering stdout; a server \
can be up before its startup line flushes. Probe it directly, e.g. curl the port, to confirm.)"
                }
                _ => "(no output)",
            });
        } else {
            out.push_str(&tail);
        }
        Ok(out)
    }

    /// Finished jobs the model hasn't seen yet, marked seen as taken — the engine
    /// folds these in at step boundaries so the model needn't busy-poll `check_job`.
    pub fn drain_finished_notices(&self) -> Vec<String> {
        let mut inner = self.inner.lock().unwrap();
        inner.reap_all();
        inner
            .jobs
            .iter_mut()
            .filter(|j| !matches!(j.status, JobStatus::Running) && !j.notified)
            .map(|j| {
                j.notified = true;
                format!("{} (log: {})", status_line(j), j.log_path.display())
            })
            .collect()
    }

    /// Tree-kill job `id` (TERM → grace → KILL), returning as soon as it dies;
    /// idempotent on a finished job. A job that exits on its own during the grace
    /// keeps its real `Exited(code)` — only a forced/unreaped job is marked `Killed`.
    pub async fn kill(&self, id: &str) -> Result<String, String> {
        let command = {
            let mut inner = self.inner.lock().unwrap();
            inner.reap_all();
            let Some(idx) = inner.jobs.iter().position(|j| j.id == id) else {
                return Err(inner.unknown_id_error(id));
            };
            if !matches!(inner.jobs[idx].status, JobStatus::Running) {
                inner.jobs[idx].notified = true; // the model just observed the finish
                return Ok(format!(
                    "job {id} already finished ({}).",
                    status_word(&inner.jobs[idx].status)
                ));
            }
            first_kill_signal(&inner.jobs[idx]);
            inner.jobs[idx].command.clone()
        };

        if !self.wait_for_exit(id, TERM_GRACE).await {
            {
                let mut inner = self.inner.lock().unwrap();
                if let Some(idx) = inner.jobs.iter().position(|j| j.id == id)
                    && matches!(inner.jobs[idx].status, JobStatus::Running)
                {
                    hard_kill_signal(&mut inner.jobs[idx]);
                }
            }
            self.wait_for_exit(id, KILL_WAIT).await;
        }

        let mut inner = self.inner.lock().unwrap();
        let Some(idx) = inner.jobs.iter().position(|j| j.id == id) else {
            return Ok(format!("killed job {id} ({command})."));
        };
        let job = &mut inner.jobs[idx];
        finalize_killed(job);
        job.notified = true; // finish observed via the kill's own result
        let ran = fmt_dur(job.runtime());
        Ok(match &job.status {
            JobStatus::Exited(Some(c)) => {
                format!("job {id} ({command}) stopped — exited with code {c} after {ran}.")
            }
            _ => format!("killed job {id} ({command}) after {ran}."),
        })
    }

    /// TERM every running group, ONE shared (polled) grace, KILL survivors; returns how many were running.
    pub async fn kill_all(&self) -> usize {
        // Capture the ids WE signal — only those may be relabeled `Killed`; a job that
        // signal-died before this call keeps its own `Exited(None)`.
        let signaled: Vec<String> = {
            let mut inner = self.inner.lock().unwrap();
            inner.reap_all();
            let running: Vec<String> = inner
                .jobs
                .iter()
                .filter(|j| matches!(j.status, JobStatus::Running))
                .map(|j| j.id.clone())
                .collect();
            for job in inner.jobs.iter().filter(|j| running.contains(&j.id)) {
                first_kill_signal(job);
            }
            running
        };
        if signaled.is_empty() {
            return 0;
        }

        if !self.wait_all_exited(TERM_GRACE).await {
            {
                let mut inner = self.inner.lock().unwrap();
                for job in &mut inner.jobs {
                    reap(job);
                    if matches!(job.status, JobStatus::Running) {
                        hard_kill_signal(job);
                    }
                }
            }
            self.wait_all_exited(KILL_WAIT).await;
        }

        let mut inner = self.inner.lock().unwrap();
        for job in &mut inner.jobs {
            if signaled.contains(&job.id) {
                finalize_killed(job);
            }
        }
        signaled.len()
    }

    /// Poll job `id` up to `budget` (reaping under the lock); `true` once terminal/gone.
    async fn wait_for_exit(&self, id: &str, budget: Duration) -> bool {
        self.poll_until(budget, |inner| {
            match inner.jobs.iter_mut().find(|j| j.id == id) {
                Some(job) => {
                    reap(job);
                    !matches!(job.status, JobStatus::Running)
                }
                None => true,
            }
        })
        .await
    }

    /// Poll all jobs up to `budget`; `true` once none are running.
    async fn wait_all_exited(&self, budget: Duration) -> bool {
        self.poll_until(budget, |inner| {
            inner.reap_all();
            inner.running_count() == 0
        })
        .await
    }

    /// Re-lock + `done` each poll tick until it's true or `budget` runs out; the sleep
    /// clamp keeps the final tick from overshooting the budget.
    async fn poll_until(&self, budget: Duration, mut done: impl FnMut(&mut Inner) -> bool) -> bool {
        let start = Instant::now();
        loop {
            if done(&mut self.inner.lock().unwrap()) {
                return true;
            }
            let elapsed = start.elapsed();
            if elapsed >= budget {
                return false;
            }
            tokio::time::sleep(KILL_POLL.min(budget - elapsed)).await;
        }
    }
}

/// `args.background == true` — routes a `run_bash` call to [`JobTable::spawn`].
pub fn wants_background(args: &Value) -> bool {
    args.get("background")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Advertised only when a job table is wired (see `set_jobs`).
pub fn check_job_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "check_job".to_string(),
        description:
            "Check a background job started with run_bash `background: true`: returns its \
status (running / exited with code / killed), runtime, and the tail of its log. Pass `kill: true` \
to terminate the job and its whole process tree."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The job id run_bash returned (e.g. \"j1\")."},
                "kill": {"type": "boolean", "description": "Terminate the job and its whole process tree."}
            },
            "required": ["id"]
        }),
    }
}

/// `try_wait` a Running job; freeze `ended_at` + status on exit. Skips finished jobs so
/// it can't overwrite `Killed`.
fn reap(job: &mut Job) {
    if !matches!(job.status, JobStatus::Running) {
        return;
    }
    if let Ok(Some(status)) = job.child.try_wait() {
        job.ended_at.get_or_insert_with(Instant::now);
        job.status = JobStatus::Exited(status.code());
    }
}

/// Only call on jobs WE signaled: a clean self-exit keeps its code, anything
/// else (our signal or a still-unreaped job) is recorded as `Killed`.
fn finalize_killed(job: &mut Job) {
    reap(job);
    if !matches!(job.status, JobStatus::Exited(Some(_))) {
        job.ended_at.get_or_insert_with(Instant::now);
        job.status = JobStatus::Killed;
    }
}

fn first_kill_signal(job: &Job) {
    #[cfg(unix)]
    signal_group(job.pgid, libc::SIGTERM);
    // Detached: runs under the jobs mutex; must not block the runtime.
    #[cfg(windows)]
    taskkill_tree_detached(job.pid);
}

fn hard_kill_signal(job: &mut Job) {
    #[cfg(unix)]
    signal_group(job.pgid, libc::SIGKILL);
    #[cfg(windows)]
    if !taskkill_tree(job.pid) {
        let _ = job.child.start_kill();
    }
}

/// Signal a whole process group (`-pgid`); best-effort (the target may have just exited).
#[cfg(unix)]
pub(crate) fn signal_group(pgid: i32, sig: i32) {
    if pgid > 1 {
        // SAFETY: kill takes integers only; a negative pid targets the group.
        unsafe {
            libc::kill(-pgid, sig);
        }
    }
}

/// `/T` kills the whole tree (`child.kill()` orphans grandchildren); detached so
/// it never blocks the caller.
#[cfg(windows)]
pub(crate) fn taskkill_tree_detached(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Blocking variant; reports whether `taskkill` ran so the hard kill can fall
/// back to `child.kill()`.
#[cfg(windows)]
pub(crate) fn taskkill_tree(pid: u32) -> bool {
    std::process::Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

fn read_tail(path: &Path) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(TAIL_BYTES as u64);
    let _ = f.seek(SeekFrom::Start(start));
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    crate::agent::tools::cap_tail_with(text, TAIL_BYTES, TAIL_LINES)
}

fn status_word(s: &JobStatus) -> String {
    match s {
        JobStatus::Running => "running".to_string(),
        JobStatus::Exited(Some(c)) => format!("exited {c}"),
        JobStatus::Exited(None) => "exited (signal)".to_string(),
        JobStatus::Killed => "killed".to_string(),
    }
}

fn status_line(job: &Job) -> String {
    let ran = fmt_dur(job.runtime());
    match &job.status {
        JobStatus::Running => format!(
            "job {}: running ({ran}, pid {}): {}",
            job.id, job.pid, job.command
        ),
        JobStatus::Exited(Some(c)) => {
            format!(
                "job {}: exited with code {c} (ran {ran}): {}",
                job.id, job.command
            )
        }
        JobStatus::Exited(None) => {
            format!(
                "job {}: exited (signal) (ran {ran}): {}",
                job.id, job.command
            )
        }
        JobStatus::Killed => format!("job {}: killed (ran {ran}): {}", job.id, job.command),
    }
}

/// Compact duration: `12s`, `2m14s`, `1h3m`.
fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3_600 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        format!("{}h{}m", s / 3_600, (s % 3_600) / 60)
    }
}

fn default_logs_root() -> PathBuf {
    std::env::temp_dir().join(format!("aivo-jobs-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("aivo-jobs-test-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn table(logs: &Path) -> SharedJobs {
        JobTable::new(Some(logs.join("jobs")))
    }

    #[cfg(unix)]
    async fn wait_for(jobs: &SharedJobs, id: &str, needle: &str) -> String {
        for _ in 0..100 {
            let out = jobs.check(id).unwrap_or_default();
            if out.contains(needle) {
                return out;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        jobs.check(id).unwrap_or_default()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_spawn_returns_id_and_log_path() {
        let dir = tmp();
        let jobs = table(&dir);
        let out = jobs.spawn("echo hello", &dir).unwrap();
        assert!(out.contains("started background job j1"), "got: {out}");
        assert!(out.contains("j1.log"), "log path missing: {out}");
        let check = wait_for(&jobs, "j1", "exited").await;
        assert!(check.contains("hello"), "log tail missing output: {check}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_check_reports_exit_code() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("exit 7", &dir).unwrap();
        let check = wait_for(&jobs, "j1", "exited with code").await;
        assert!(check.contains("exited with code 7"), "got: {check}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_kill_terminates_process_tree() {
        let dir = tmp();
        let jobs = table(&dir);
        let out = jobs.spawn("sleep 30", &dir).unwrap();
        let pid: u32 = out
            .split("pid ")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .and_then(|s| s.trim().parse().ok())
            .unwrap();
        let msg = jobs.kill("j1").await.unwrap();
        assert!(msg.contains("killed job j1"), "got: {msg}");
        assert!(
            !crate::services::system_env::is_pid_alive(pid),
            "process {pid} should be gone after kill"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_kill_escalates_to_sigkill() {
        let dir = tmp();
        let jobs = table(&dir);
        // Ignores TERM → only SIGKILL ends it.
        jobs.spawn("trap '' TERM; sleep 30", &dir).unwrap();
        let start = Instant::now();
        let msg = jobs.kill("j1").await.unwrap();
        assert!(msg.contains("killed job j1"), "got: {msg}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "kill took too long"
        );
        let check = jobs.check("j1").unwrap();
        assert!(check.contains("killed"), "status should be killed: {check}");
    }

    /// kill() returns as soon as a TERM-responsive job dies, not after the full window.
    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_kill_returns_promptly() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("sleep 30", &dir).unwrap(); // dies on the default SIGTERM
        let start = Instant::now();
        let msg = jobs.kill("j1").await.unwrap();
        assert!(msg.contains("killed job j1"), "got: {msg}");
        assert!(
            start.elapsed() < TERM_GRACE + KILL_WAIT,
            "kill should return before the full grace+kill window, took {:?}",
            start.elapsed()
        );
    }

    /// kill() on a self-finished job is idempotent and keeps its real exit code.
    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_kill_on_self_finished_keeps_exit_code() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("exit 5", &dir).unwrap();
        let _ = wait_for(&jobs, "j1", "exited with code").await;
        let msg = jobs.kill("j1").await.unwrap();
        assert!(msg.contains("already finished"), "got: {msg}");
        assert!(msg.contains("exited 5"), "exit code preserved: {msg}");
    }

    /// Finished jobs beyond the cap are pruned so the table can't grow unbounded.
    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_prune_bounds_finished() {
        let dir = tmp();
        let jobs = table(&dir);
        // Spawn well past the finished cap; each exits immediately. Wait for each to
        // be reaped before the next spawn — a fixed sleep flakes under suite load.
        for _ in 0..(MAX_FINISHED_JOBS + 4) {
            jobs.spawn("exit 0", &dir).unwrap();
            for _ in 0..200 {
                if jobs.running_count() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        // The oldest finished job has been pruned; a recent id is still queryable.
        assert!(jobs.check("j1").is_err(), "oldest finished job pruned");
        let recent = format!("j{}", MAX_FINISHED_JOBS + 4);
        assert!(jobs.check(&recent).is_ok(), "recent job still queryable");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_unknown_id_lists_known() {
        let dir = tmp();
        let jobs = table(&dir);
        assert!(
            jobs.check("j9")
                .unwrap_err()
                .contains("no jobs have been started")
        );
        jobs.spawn("exit 0", &dir).unwrap();
        let err = jobs.check("j9").unwrap_err();
        assert!(err.contains("Known jobs: j1"), "got: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_finished_job_stays_queryable() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("exit 0", &dir).unwrap();
        let _ = wait_for(&jobs, "j1", "exited").await;
        assert!(jobs.check("j1").is_ok());
        assert!(jobs.kill("j1").await.unwrap().contains("already finished"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_running_cap_blocks_spawn() {
        let dir = tmp();
        let jobs = table(&dir);
        for _ in 0..MAX_RUNNING_JOBS {
            jobs.spawn("sleep 30", &dir).unwrap();
        }
        let err = jobs.spawn("sleep 30", &dir).unwrap_err();
        assert!(err.contains("too many background jobs"), "got: {err}");
        let _ = jobs.kill_all().await;
    }

    #[test]
    fn jobs_logs_root_falls_back_to_temp() {
        let jobs = JobTable::new(None);
        assert!(
            jobs.logs_root().starts_with(std::env::temp_dir()),
            "None must resolve under temp_dir: {}",
            jobs.logs_root().display()
        );
        assert_eq!(jobs.running_count(), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_check_reads_output_while_running() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("echo ready; sleep 30", &dir).unwrap();
        let check = wait_for(&jobs, "j1", "ready").await;
        assert!(check.contains("running"), "still running: {check}");
        assert!(check.contains("ready"), "live output visible: {check}");
        jobs.kill("j1").await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_empty_tail_mentions_buffering_while_running() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("sleep 30", &dir).unwrap();
        let check = jobs.check("j1").unwrap();
        assert!(
            check.contains("buffering stdout"),
            "buffering hint missing: {check}"
        );
        jobs.kill("j1").await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn jobs_tail_is_bounded() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("for i in $(seq 1 5000); do echo line$i; done", &dir)
            .unwrap();
        let check = wait_for(&jobs, "j1", "exited").await;
        let tail = check.split("--- log tail ---\n").nth(1).unwrap_or("");
        assert!(
            tail.lines().count() <= TAIL_LINES + 1,
            "tail not bounded: {} lines",
            tail.lines().count()
        );
        assert!(
            check.contains("line5000"),
            "tail should keep the END: {check}"
        );
    }

    #[cfg(unix)]
    async fn wait_reaped(jobs: &SharedJobs) {
        for _ in 0..100 {
            if jobs.running_count() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drain_finished_notices_reports_each_finish_once() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("echo done-quickly", &dir).unwrap();
        wait_reaped(&jobs).await;
        let notices = jobs.drain_finished_notices();
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(
            notices[0].contains("job j1") && notices[0].contains("exited"),
            "{notices:?}"
        );
        assert!(
            notices[0].contains("j1.log"),
            "log path missing: {notices:?}"
        );
        assert!(
            jobs.drain_finished_notices().is_empty(),
            "a drained finish must not re-announce"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn checked_or_killed_jobs_are_not_re_announced() {
        let dir = tmp();
        let jobs = table(&dir);
        jobs.spawn("true", &dir).unwrap();
        wait_for(&jobs, "j1", "exited").await; // check() observes the finish
        assert!(
            jobs.drain_finished_notices().is_empty(),
            "check_job already told the model"
        );
        jobs.spawn("sleep 30", &dir).unwrap();
        jobs.kill("j2").await.unwrap();
        assert!(
            jobs.drain_finished_notices().is_empty(),
            "the kill's own result already told the model"
        );
    }

    /// Keeps the Windows runner compiling + exercising the module (Windows-only breakage
    /// surfaces pre-release).
    #[cfg(windows)]
    #[tokio::test]
    async fn jobs_spawn_and_exit_windows() {
        let dir = tmp();
        let jobs = table(&dir);
        let out = jobs.spawn("exit 0", &dir).unwrap();
        assert!(out.contains("started background job j1"), "got: {out}");
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(jobs.check("j1").is_ok());
        let _ = jobs.kill_all().await;
    }
}
