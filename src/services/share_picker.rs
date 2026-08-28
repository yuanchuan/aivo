//! Inline session picker for `aivo logs share` / `aivo logs show` (no id passed).
//!
//! Draws straight from `fetch_unified_rows`, which already excludes `[run]`
//! launch records and `[serve]` HTTP events from its default output — the
//! unified listing is the *session* list, and that's exactly what the
//! picker needs. No source-type filter here; the two views agree by
//! construction.

use std::io::{self, IsTerminal};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::cli::LogsArgs;
use crate::commands::logs::{
    UnifiedRow, compute_orphan_code_ids, fetch_unified_rows, min_unique_id_width,
    picker_detail_width,
};
use crate::services::session_store::SessionStore;
use crate::style;
use crate::tui::FuzzySelect;

/// Upper bound on rows pulled from `fetch_unified_rows` before run/serve
/// filtering. Sized for "user can find what they want without scrolling";
/// older items are still reachable by typing a prefix as an explicit id.
const PICKER_FETCH_LIMIT: usize = 500;

/// Public entrypoint. `Ok(Some(id))` = user selected, `Ok(None)` = cancelled
/// or no items, `Err(_)` = setup or I/O failure. `prompt` is the FuzzySelect
/// header; `cmd_hint` is the example shown in the non-TTY error.
pub async fn pick_session_id(
    session_store: &SessionStore,
    project_root: &Path,
    all: bool,
    prompt: &str,
    cmd_hint: &str,
) -> Result<Option<String>> {
    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        anyhow::bail!(
            "A terminal is required to show the picker. Pass an explicit id, e.g. `{cmd_hint}`."
        );
    }

    let args = build_logs_args(project_root, all);

    // Delayed spinner: cheap loads finish before the spinner shows, slow
    // ones get feedback. Mirrors `aivo logs`'s list path.
    const SPINNER_DELAY: Duration = Duration::from_millis(250);
    let load = load_rows(session_store, &args);
    tokio::pin!(load);
    let (rows, run_meta) = tokio::select! {
        rows = &mut load => rows?,
        _ = tokio::time::sleep(SPINNER_DELAY) => {
            let (spinning, handle) = style::start_spinner(Some(" loading sessions…"));
            let rows = (&mut load).await;
            style::stop_spinner(&spinning);
            let _ = handle.await;
            rows?
        }
    };

    if rows.is_empty() {
        let scope = if all { "any project" } else { "this project" };
        println!("{}", style::dim(format!("No sessions found in {scope}.")));
        return Ok(None);
    }

    let prompt = if all {
        format!("{prompt} (all projects)")
    } else {
        prompt.to_string()
    };

    let id_width = min_unique_id_width(&rows);
    let detail_width = picker_detail_width(console::Term::stdout().size().1 as usize, id_width);
    let orphan_chat_ids = compute_orphan_code_ids(session_store).await;
    let labels: Vec<String> = rows
        .iter()
        .map(|r| r.picker_label(id_width, detail_width, &orphan_chat_ids, &run_meta))
        .collect();
    let ids: Vec<String> = rows.iter().map(UnifiedRow::id).collect();

    // `FuzzySelect::interact_opt` blocks on `event::read()`. The aivo
    // runtime is current-thread, so do it on a blocking thread to keep
    // the runtime free for other futures (e.g. Ctrl+C handling).
    tokio::task::spawn_blocking(move || -> std::io::Result<Option<String>> {
        let selected = FuzzySelect::new()
            .with_prompt(&prompt)
            .items(&labels)
            .default(0)
            .interact_opt()?;
        Ok(selected.map(|idx| ids[idx].clone()))
    })
    .await
    .context("picker thread panicked")?
    .context("picker I/O failed")
}

/// Build a `LogsArgs` for the picker query. Mirrors `aivo logs` defaults
/// (14-day native window, 20-row limit raised here to `PICKER_FETCH_LIMIT`
/// so the user has room to scroll/filter without an immediate "N more below"
/// truncation).
fn build_logs_args(project_root: &Path, all: bool) -> LogsArgs {
    LogsArgs {
        action: None,
        target: None,
        limit: PICKER_FETCH_LIMIT,
        json: false,
        watch: false,
        jsonl: false,
        search: None,
        by: None,
        model: None,
        key: None,
        cwd: if all {
            None
        } else {
            Some(project_root.to_string_lossy().into_owned())
        },
        all,
        since: None,
        until: None,
        errors: false,
        no_redact: false,
        open: false,
        debug_local_only: false,
        yes: false,
    }
}

async fn load_rows(
    store: &SessionStore,
    args: &LogsArgs,
) -> Result<(Vec<UnifiedRow>, crate::commands::logs::RunMetaIndex)> {
    fetch_unified_rows(store, args).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{SessionTokens, StoredChatMessage};
    use tempfile::TempDir;

    #[tokio::test]
    async fn load_rows_includes_a_live_code_checkpoint_before_its_first_turn() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
            .await
            .unwrap();
        let messages = vec![StoredChatMessage {
            role: "user".to_string(),
            content: "/goal finish the task".to_string(),
            model: None,
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }];
        store
            .save_code_session_with_id(
                &key_id,
                "https://api.example.com",
                "/tmp/demo",
                "live-picker-session",
                "claude",
                None,
                &messages,
                "claude",
                "claude",
                SessionTokens::default(),
                0.0,
            )
            .await
            .unwrap();

        let args = build_logs_args(Path::new("/tmp/demo"), false);
        let (rows, _) = load_rows(&store, &args).await.unwrap();
        assert!(
            rows.iter().any(|row| row.id() == "live-picker-session"),
            "a running code session must be visible before its first completed turn"
        );
    }
}
