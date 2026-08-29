use super::super::*;
use super::helpers::*;
use chrono::Duration as ChronoDuration;
use tempfile::TempDir;

#[test]
fn test_resumable_session_id_skips_empty_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "abc-123".to_string();

    // An untouched chat has nothing saved → no resume hint.
    assert!(app.history.is_empty());
    assert_eq!(app.resumable_session_id(), None);

    // Once something has been said, the exit hint points back at this session.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    assert_eq!(app.resumable_session_id(), Some("abc-123"));
}

#[tokio::test]
async fn test_rewind_truncates_history_and_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-1".to_string();
    seed_two_exchanges(&mut app);

    // Rewind to the second user turn (history index 2). No live engine in the
    // test app → conversation-only path (ordinal None).
    app.rewind_to_turn(2, None, false).await.unwrap();

    // That turn and everything after it are gone; the prior exchange stays.
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[0].content, "first question");
    assert_eq!(app.history[1].content, "first answer");
    // The rewound message is restored to the composer with the cursor at the end.
    assert_eq!(app.draft, "second question");
    assert_eq!(app.cursor, app.draft.len());
}

/// A rewind invalidates the measured fill — footer and `/context` must drop
/// back to a flagged estimate of the surviving turns.
#[tokio::test]
async fn test_rewind_reestimates_context_fill() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-ctx".to_string();
    seed_two_exchanges(&mut app);
    app.context_tokens = 100_000;
    app.context_is_estimate = false;
    app.last_usage = Some(crate::commands::code_response_parser::TokenUsage {
        prompt_tokens: 99_000,
        completion_tokens: 1_000,
        ..Default::default()
    });

    app.rewind_to_turn(2, None, false).await.unwrap();

    assert!(app.context_is_estimate, "measured flag must not survive");
    assert_eq!(app.last_usage, None);
    assert!(
        app.context_tokens < 100_000,
        "fill must be re-estimated from the truncated history, got {}",
        app.context_tokens
    );
}

#[tokio::test]
async fn test_rewind_to_first_turn_clears_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-2".to_string();
    seed_two_exchanges(&mut app);

    app.rewind_to_turn(0, None, false).await.unwrap();

    assert!(app.history.is_empty());
    assert_eq!(app.draft, "first question");
}

#[tokio::test]
async fn test_open_rewind_picker_lists_user_turns_newest_first() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert!(matches!(picker.kind, PickerKind::Rewind));
    // One row per user turn, newest first.
    assert_eq!(picker.items.len(), 2);
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
        keep_engine,
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 2);
    // No live engine in the test app → no checkpoints → conversation-only.
    assert!(ordinal.is_none());
    assert!(!keep_engine);
    assert!(picker.items[0].label.contains("conversation only"));
}

#[tokio::test]
async fn test_rewind_picker_ignores_non_agent_row_with_identical_text() {
    // A plain-chat/ACP row with text equal to an earlier engine turn's prompt
    // must not steal that turn's checkpoint (that rewound one turn too far).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for (role, content) in [
        ("user", "continue"),
        ("assistant", "done"),
        ("user", "continue"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    // Only the FIRST "continue" went through the engine.
    app.agent_turn_indices.insert(0);
    let mut engine =
        crate::agent::engine::AgentEngine::new("/tmp", "claude", "2026-06-14", &[], &[], 0, 0);
    engine.checkpoints.push(crate::agent::engine::Checkpoint {
        msg_index: 1,
        prompt: "continue".to_string(),
        tree: Some("abc".to_string()),
        changed: Some(Vec::new()),
        seg_tree: None,
    });
    app.agent_engine = Some(AgentSession {
        key_id: "k".to_string(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert_eq!(picker.items.len(), 2);
    // Newest row: no checkpoint of its own and none newer — conversation-only.
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
        keep_engine,
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 2);
    assert!(
        ordinal.is_none(),
        "a non-engine row must not steal the checkpoint"
    );
    assert!(!keep_engine);
    assert!(picker.items[0].label.contains("conversation only"));
    // Older row = the real engine turn: keeps its checkpoint and file revert.
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
        keep_engine,
    } = &picker.items[1].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 0);
    assert_eq!(*ordinal, Some(0));
    assert!(keep_engine);
    assert!(!picker.items[1].label.contains("conversation only"));
}

#[tokio::test]
async fn test_rewind_picker_boundary_reverts_newer_turns_for_conversation_only_row() {
    // A checkpoint-less row must revert newer edits via the nearest newer checkpoint.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for (role, content) in [
        ("user", "old ask"),
        ("assistant", "old reply"),
        ("user", "new ask"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    app.agent_turn_indices.insert(2);
    let mut engine =
        crate::agent::engine::AgentEngine::new("/tmp", "claude", "2026-06-14", &[], &[], 0, 0);
    engine.checkpoints.push(crate::agent::engine::Checkpoint {
        msg_index: 1,
        prompt: "new ask".to_string(),
        tree: Some("abc".to_string()),
        changed: Some(Vec::new()),
        seg_tree: None,
    });
    app.agent_engine = Some(AgentSession {
        key_id: "k".to_string(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert_eq!(picker.items.len(), 2);
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
        keep_engine,
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!((*history_index, *ordinal, *keep_engine), (2, Some(0), true));
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
        keep_engine,
    } = &picker.items[1].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(
        (*history_index, *ordinal, *keep_engine),
        (0, Some(0), false)
    );
    assert!(!picker.items[1].label.contains("conversation only"));
}

/// Boundary rewind: files revert via the newer checkpoint; engine + transcript drop.
#[tokio::test]
async fn test_rewind_boundary_reverts_files_then_drops_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-boundary".to_string();
    let dir = TempDir::new().unwrap();
    let cwd = dir.path();
    std::fs::write(cwd.join("a.txt"), "v0").unwrap();

    let mut engine = crate::agent::engine::AgentEngine::new(
        &cwd.display().to_string(),
        "claude",
        "2026-06-14",
        &[],
        &[],
        0,
        0,
    );
    engine.enable_rewind_checkpoints(&cwd.display().to_string());
    // Reach the store through the transplant API (private field otherwise).
    let (store, _) = engine.take_rewind_state();
    let mut store = store.unwrap();
    if !store.git_available().await {
        return; // git missing → skip
    }
    let tree = store.snapshot().await;
    assert!(tree.is_some(), "snapshot must succeed");
    engine.adopt_rewind_state(
        Some(store),
        vec![crate::agent::engine::Checkpoint {
            msg_index: 1,
            prompt: "new ask".to_string(),
            tree,
            changed: None,
            seg_tree: None,
        }],
    );
    std::fs::write(cwd.join("a.txt"), "v1").unwrap();

    for (role, content) in [("user", "old ask"), ("assistant", "r"), ("user", "new ask")] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    app.agent_turn_indices.insert(2);
    app.agent_engine = Some(AgentSession {
        key_id: "k".to_string(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });
    app.pending_agent_messages = Some(vec![serde_json::json!({"role": "user", "content": "x"})]);

    app.rewind_to_turn(0, Some(0), false).await.unwrap();

    assert_eq!(
        std::fs::read_to_string(cwd.join("a.txt")).unwrap(),
        "v0",
        "the newer turn's edit reverts through the boundary checkpoint"
    );
    assert!(
        app.agent_engine.is_none(),
        "engine dropped after the revert"
    );
    assert!(app.pending_agent_messages.is_none());
    assert!(app.history.is_empty());
    assert_eq!(app.draft, "old ask");
}

/// Home/root launch dirs disable snapshots permanently — the picker title must say so.
#[tokio::test]
async fn test_rewind_picker_title_explains_permanent_file_revert_off() {
    let Some(home) = crate::services::system_env::home_dir() else {
        return;
    };
    if home.parent().is_none() {
        return; // HOME="/" classifies as root, not home — env-fragile
    }
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);
    let mut engine =
        crate::agent::engine::AgentEngine::new("/tmp", "claude", "2026-06-14", &[], &[], 0, 0);
    engine.enable_rewind_checkpoints(&home.display().to_string());
    app.agent_engine = Some(AgentSession {
        key_id: "k".to_string(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert!(
        picker.title.contains("file revert off"),
        "got title: {}",
        picker.title
    );
}

/// A cursor-ACP turn with a TUI-side checkpoint tree gets file revert — and so
/// does an older row via the nearest-newer rule — without any engine involved.
#[tokio::test]
async fn test_rewind_picker_acp_checkpoint_lights_up_file_revert() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);
    // Both turns ran over cursor ACP; only the newer one snapshotted a tree.
    app.acp_checkpoints.push(AcpCheckpoint {
        history_index: 0,
        prompt: "first question".to_string(),
        tree: None,
        changed: Some(Vec::new()),
    });
    app.acp_checkpoints.push(AcpCheckpoint {
        history_index: 2,
        prompt: "second question".to_string(),
        tree: Some("abc".to_string()),
        changed: Some(vec![std::path::PathBuf::from("a.txt")]),
    });

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert!(
        !picker.items[0].label.contains("conversation only"),
        "own checkpoint tree → file revert; got: {}",
        picker.items[0].label
    );
    assert!(
        !picker.items[1].label.contains("conversation only"),
        "treeless row reverts via the nearest newer tree; got: {}",
        picker.items[1].label
    );
    // ACP rows never map to engine checkpoints — still the drop-and-reseed path.
    let PickerValue::RewindTurn {
        ordinal,
        keep_engine,
        ..
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert!(ordinal.is_none());
    assert!(!keep_engine);
}

/// A drifted checkpoint (row content no longer matches) must not count as
/// revertible — the row degrades to conversation-only.
#[tokio::test]
async fn test_rewind_picker_ignores_drifted_acp_checkpoint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);
    app.acp_checkpoints.push(AcpCheckpoint {
        history_index: 2,
        prompt: "SOMETHING ELSE".to_string(),
        tree: Some("abc".to_string()),
        changed: Some(Vec::new()),
    });

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert!(picker.items[0].label.contains("conversation only"));
    assert!(picker.items[1].label.contains("conversation only"));
}

/// End-to-end ACP rewind: an OPEN checkpoint (interrupted turn) finalizes
/// lazily, the turn's edit reverts, and the checkpoint list truncates.
#[tokio::test]
async fn test_rewind_acp_reverts_files() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-acp".to_string();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.txt"), "v0").unwrap();

    let mut store = crate::agent::checkpoint::CheckpointStore::new(dir.path());
    if !store.git_available().await {
        return; // git missing → skip
    }
    let tree = store.snapshot().await;
    assert!(tree.is_some(), "snapshot must succeed");
    // The cursor turn edits the file after the pre-prompt snapshot.
    std::fs::write(dir.path().join("a.txt"), "v1").unwrap();

    for (role, content) in [
        ("user", "old ask"),
        ("assistant", "r"),
        ("user", "cursor ask"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    app.acp_checkpoints.push(AcpCheckpoint {
        history_index: 2,
        prompt: "cursor ask".to_string(),
        tree,
        changed: None, // interrupted before turn end → finalized at rewind
    });
    app.acp_checkpoint_store = Some(std::sync::Arc::new(tokio::sync::Mutex::new(store)));

    app.rewind_to_turn(2, None, false).await.unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "v0",
        "the cursor turn's edit reverts"
    );
    assert!(app.acp_checkpoints.is_empty());
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.draft, "cursor ask");
    let (_, msg) = app.notice.as_ref().expect("a notice");
    assert!(msg.contains("restored 1 file"), "got: {msg}");
}

#[tokio::test]
async fn test_open_rewind_picker_with_no_turns_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_rewind_picker().await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    let (_, msg) = app.notice.as_ref().expect("a notice");
    assert!(msg.contains("Nothing to rewind to"), "got: {msg}");
}

#[test]
fn test_fork_shows_provenance_line_in_welcome() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Native session: no provenance line.
    app.session_id = "abcdef12-3456-7890-abcd-ef1234567890".to_string();
    let native: Vec<String> = app
        .welcome_status_lines()
        .into_iter()
        .map(|sl| sl.plain)
        .collect();
    assert!(
        !native.iter().any(|l| l.contains("Forked from")),
        "native session should have no provenance line: {native:?}"
    );

    // A fork (imported Claude session) names its source.
    app.session_id = "import-claude-a1b2c3d4".to_string();
    let fork: Vec<String> = app
        .welcome_status_lines()
        .into_iter()
        .map(|sl| sl.plain)
        .collect();
    assert!(
        fork.iter()
            .any(|l| l.contains("Forked from a Claude session")),
        "fork provenance line missing: {fork:?}"
    );
}

#[tokio::test]
async fn opening_foreign_session_resumes_in_memory_without_persisting() {
    use crate::services::session_import::{ImportableSession, SessionOrigin};
    let dir = tempfile::tempdir().unwrap();
    let jsonl = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello there friend\"}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hi back\"}]}}\n";
    let src = dir.path().join("f.jsonl");
    std::fs::write(&src, jsonl).unwrap();
    let preview = SessionPreview::from_importable(ImportableSession {
        origin: SessionOrigin {
            cli: "claude".to_string(),
            foreign_id: "zzz".to_string(),
            source_path: src.to_string_lossy().to_string(),
        },
        title: "hello there friend".to_string(),
        updated_at: Utc::now(),
        aivo_id: "import-claude-zzz".to_string(),
    });
    let store =
        crate::services::session_store::SessionStore::with_path(dir.path().join("config.json"));

    let loaded =
        super::super::storage::load_or_import_resume_session(&store, &preview, "key-1", "gpt-x")
            .await
            .expect("foreign resume should reconstruct in memory");
    // Reconstructed in memory, flagged pristine, and NOT written to the store —
    // merely opening a Claude session creates no aivo copy.
    assert!(loaded.pristine_import);
    // The fork's id is the deterministic digest of the source id (recomputed from
    // origin), not the picker row's hand-set value.
    assert_eq!(
        loaded.session_id,
        crate::services::session_import::import_session_id("claude", "zzz")
    );
    assert!(loaded.messages.iter().any(|m| m.role == "user"));
    assert!(loaded.engine_messages.is_some());
    assert_eq!(store.count_chat_sessions().await, 0);
}

#[tokio::test]
async fn resuming_stale_fork_by_source_id_flags_source_newer() {
    use crate::services::session_import::{ImportableSession, SessionOrigin, import_session_id};
    let dir = tempfile::tempdir().unwrap();
    let store =
        crate::services::session_store::SessionStore::with_path(dir.path().join("config.json"));
    let fork_id = import_session_id("claude", "src-42");
    store
        .save_code_session_with_id(
            "key-1",
            "https://api.example.com",
            "/tmp/demo",
            &fork_id,
            "gpt-x",
            None,
            &one_user_message("imported turn"),
            "imported turn",
            "imported turn",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    let preview_at = |ts: chrono::DateTime<Utc>| {
        SessionPreview::from_importable(ImportableSession {
            origin: SessionOrigin {
                cli: "claude".to_string(),
                foreign_id: "src-42".to_string(),
                source_path: "/tmp/src.jsonl".to_string(),
            },
            title: "imported turn".to_string(),
            updated_at: ts,
            aivo_id: fork_id.clone(),
        })
    };

    // Source file newer than the fork's save → diverged, flagged.
    let loaded = super::super::storage::load_or_import_resume_session(
        &store,
        &preview_at(Utc::now() + ChronoDuration::hours(1)),
        "key-1",
        "gpt-x",
    )
    .await
    .unwrap();
    assert_eq!(loaded.session_id, fork_id);
    assert!(loaded.source_newer, "future source must flag divergence");

    // Source untouched since the import → clean fork-first load.
    let loaded = super::super::storage::load_or_import_resume_session(
        &store,
        &preview_at(Utc::now() - ChronoDuration::hours(1)),
        "key-1",
        "gpt-x",
    )
    .await
    .unwrap();
    assert!(!loaded.source_newer);
}

/// A conversation-only rewind after `/resume` must drop the stashed durable
/// transcript, or the next turn restores the full pre-rewind conversation.
#[tokio::test]
async fn test_conversation_only_rewind_drops_pending_transcript() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first ask".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_agent_messages = Some(vec![
        serde_json::json!({"role": "user", "content": "first ask"}),
        serde_json::json!({"role": "assistant", "content": "rewound-away reply"}),
    ]);

    app.rewind_to_turn(0, None, false).await.unwrap();

    assert!(
        app.pending_agent_messages.is_none(),
        "the pre-rewind transcript must not seed the next engine"
    );
    assert!(app.history.is_empty());
    assert_eq!(app.draft, "first ask", "prompt restored for edit/resend");
}

/// Resuming must not leak the old session's plan/goal modes: a stale plan card
/// would index the replaced history and `/plan go` would run the old plan.
#[tokio::test]
async fn test_resume_with_explicit_model_keeps_launch_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "new-model".to_string();
    app.model = "new-model".to_string();
    app.model_explicit = true;

    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "resumed-explicit-model".to_string(),
        raw_model: "old-model".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert_eq!(app.raw_model, "new-model");
    assert_eq!(app.model, "new-model");
}

#[tokio::test]
async fn test_resume_with_explicit_key_keeps_launch_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key_explicit = true;
    let launch_key_id = app.key.id.clone();
    let launch_base_url = app.key.base_url.clone();

    let session = LoadedSession {
        key_id: "old-session-key".to_string(),
        session_id: "resumed-explicit-key".to_string(),
        raw_model: "old-model".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert_eq!(app.key.id, launch_key_id);
    assert_eq!(app.key.base_url, launch_base_url);
}

#[tokio::test]
async fn test_resume_resets_plan_and_goal_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.plan_mode = true;
    app.plan_exit_pending = true;
    app.pending_plan = Some("old session's plan".to_string());
    app.plan_card_idx = Some(3);
    app.goal_mode = Some(GoalState {
        objective: "old goal".to_string(),
        iteration: 2,
        max: 20,
        msg_floor: 0,
    });

    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert!(!app.plan_mode, "plan mode belongs to the old session");
    assert!(!app.plan_exit_pending);
    assert!(
        app.pending_plan.is_none(),
        "old plan must not be /plan go-able"
    );
    assert!(
        app.plan_card_idx.is_none(),
        "card index points at replaced history"
    );
    assert!(app.goal_mode.is_none());
}

/// The saved describe cache comes back on resume (images stay cache hits);
/// `/new` clears it so the next session's file can't inherit it.
#[tokio::test]
async fn test_resume_restores_image_descriptions_and_new_clears_them() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.vision_descriptions
        .insert("stale".to_string(), "old session's entry".to_string());

    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: Some(std::collections::HashMap::from([(
            "abcd1234".to_string(),
            "[Image] a red button".to_string(),
        )])),
    };
    app.apply_loaded_session(session).await.unwrap();

    assert_eq!(
        app.vision_descriptions.get("abcd1234").map(String::as_str),
        Some("[Image] a red button"),
        "saved cache restored"
    );
    assert!(
        !app.vision_descriptions.contains_key("stale"),
        "old session's cache replaced, not merged"
    );

    app.start_new_chat().await;
    assert!(
        app.vision_descriptions.is_empty(),
        "/new must not carry descriptions into the next session's file"
    );
}

/// A resumed session with a saved unfinished plan picks it back up: plan mode
/// restored, draft `/plan go`-able, and the card re-framed at its message in the
/// restored history.
#[tokio::test]
async fn test_resume_restores_unfinished_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let plan_text = "1. refactor the gate\n2. add tests";
    let msg = |role: &str, content: &str| ChatMessage {
        model: None,
        role: role.to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };
    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![
            msg("user", "plan the refactor"),
            msg("assistant", plan_text),
        ],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: Some(crate::services::session_store::PlanState {
            mode: true,
            draft: Some(plan_text.to_string()),
            steps: None,
        }),
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert!(app.plan_mode, "plan mode restored");
    assert_eq!(app.pending_plan.as_deref(), Some(plan_text));
    assert_eq!(
        app.plan_card_idx,
        Some(1),
        "card re-framed at the plan's message in the restored history"
    );
    let notice = app.notice.as_ref().map(|(_, n)| n.as_str()).unwrap_or("");
    assert!(
        notice.contains("unfinished plan"),
        "restore is announced: {notice:?}"
    );
}

#[tokio::test]
async fn test_fresh_import_announces_fidelity_saved_fork_stays_silent() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let fidelity = crate::services::session_import::ImportFidelity {
        user_turns: 3,
        assistant_turns: 4,
        tool_calls: 2,
        results_paired: 1,
        results_missing: 1,
        ..Default::default()
    };

    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "claude-deadbeef".to_string(),
        raw_model: "m1".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: true,
        source_newer: false,
        import_fidelity: Some(fidelity.clone()),
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    let notice = app.notice.as_ref().map(|(_, n)| n.clone()).unwrap();
    assert_eq!(
        notice,
        "Imported from Claude · fidelity high · 7 turns · 1/2 tool calls"
    );
    assert_eq!(app.import_fidelity, Some(fidelity));

    // A saved fork carries its stamp from disk — no announce on re-open.
    let saved = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "claude-deadbeef".to_string(),
        raw_model: "m1".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: Some(crate::services::session_import::ImportFidelity::default()),
        plan_state: None,
        image_descriptions: None,
    };
    app.notice = None;
    app.apply_loaded_session(saved).await.unwrap();
    assert!(
        app.notice.is_none(),
        "fidelity announces only on first open"
    );
}

#[tokio::test]
async fn test_new_chat_clears_import_fidelity() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.import_fidelity = Some(crate::services::session_import::ImportFidelity::default());

    app.start_new_chat().await;

    assert!(app.import_fidelity.is_none());
}

#[tokio::test]
async fn test_begin_resume_load_clears_transcript_before_result() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "old".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "pending".to_string();
    app.draft = "draft".to_string();
    let preview = SessionPreview {
        key_id: app.key.id.clone(),
        key_name: app.key.display_name().to_string(),
        base_url: app.key.base_url.clone(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
        origin: None,
    };

    app.begin_resume_load(preview.clone());

    assert!(app.history.is_empty());
    assert!(app.pending_response.is_empty());
    assert!(app.draft.is_empty());
    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.title.clone()),
        Some(preview.title)
    );
}

#[tokio::test]
async fn test_resume_last_jumps_to_newest_from_fresh_launch() {
    // `aivo code --resume last` from a fresh process (empty history) reopens the
    // most recent saved chat directly — the exit hint's round-trip.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    store
        .save_code_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "older-sess",
            "claude",
            None,
            &one_user_message("older"),
            "older",
            "older",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();
    // Guarantee a strictly-later updated_at for the second save.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    store
        .save_code_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "newer-sess",
            "claude",
            None,
            &one_user_message("newer"),
            "newer",
            "newer",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.open_resume_picker(Some("last".to_string()))
        .await
        .unwrap();

    assert!(
        matches!(app.overlay, Overlay::None),
        "`last` should resume directly, not open the picker"
    );
    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.session_id.clone()),
        Some("newer-sess".to_string()),
    );
}

#[tokio::test]
async fn test_resume_last_in_session_skips_current_chat() {
    // `/resume last` mid-conversation lands on the PREVIOUS chat, not a reload of
    // the one you're already in (which sorts newest after being persisted).
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    store
        .save_code_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "prev-sess",
            "claude",
            None,
            &one_user_message("previous"),
            "previous",
            "previous",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "current-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "live conversation".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.open_resume_picker(Some("last".to_string()))
        .await
        .unwrap();

    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.session_id.clone()),
        Some("prev-sess".to_string()),
        "in-session `last` should skip the current chat"
    );
}

#[tokio::test]
async fn test_open_resume_picker_saves_current_unsaved_session() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "fresh-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hello from a new chat".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.open_resume_picker(None).await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected session picker");
    };
    assert!(
        picker.items.iter().any(|item| {
            matches!(
                &item.value,
                PickerValue::Session(session) if session.session_id == "fresh-session"
            )
        }),
        "current unsaved session should be listed"
    );

    let saved = store
        .get_code_session("fresh-session")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(saved.session_id, "fresh-session");
}

/// The `/resume` picker only lists the launch dir's sessions, but an explicit
/// id from another dir still resolves via the global fallback.
#[tokio::test]
async fn test_open_resume_picker_scopes_to_cwd_but_id_is_global() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    for (sid, cwd) in [
        ("here-sess", "/home/me/here"),
        ("elsewhere-sess", "/home/me/elsewhere"),
    ] {
        store
            .save_code_session_with_id(
                &key_id,
                &key.base_url,
                cwd,
                sid,
                "claude",
                None,
                &[crate::services::session_store::StoredChatMessage {
                    model: None,
                    role: "user".into(),
                    content: "hi".into(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                sid,
                sid,
                crate::services::session_store::SessionTokens::default(),
                0.0,
            )
            .await
            .unwrap();
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.real_cwd = "/home/me/here".to_string();

    // Bare picker: only the launch dir's session is listed.
    app.open_resume_picker(None).await.unwrap();
    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected session picker");
    };
    let listed: Vec<&str> = picker
        .items
        .iter()
        .filter_map(|item| match &item.value {
            PickerValue::Session(session) => Some(session.session_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(listed, vec!["here-sess"], "got {listed:?}");

    // Explicit id from another dir resolves via the global fallback.
    app.overlay = Overlay::None;
    app.open_resume_picker(Some("elsewhere-sess".to_string()))
        .await
        .unwrap();
    assert!(
        app.loading_resume
            .as_ref()
            .is_some_and(|l| l.preview.session_id == "elsewhere-sess"),
        "explicit cross-dir id should begin a resume load"
    );
}

#[tokio::test]
async fn test_delete_picker_selection_removes_saved_chat() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    store
        .save_code_session_with_id(
            &key_id,
            "https://api.example.com",
            "/tmp/demo",
            "session-1234",
            "claude",
            None,
            &[
                crate::services::session_store::StoredChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                },
                crate::services::session_store::StoredChatMessage {
                    model: None,
                    role: "assistant".to_string(),
                    content: "hi there".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                },
            ],
            "hello",
            "hello · hi there",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    let preview = SessionPreview {
        key_id: key_id.clone(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hello".to_string(),
        preview_text: "hello · hi there".to_string(),
        origin: None,
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.cwd = "/tmp/demo".to_string();
    app.overlay = Overlay::Picker(Box::new(PickerState::ready(
        "Sessions",
        String::new(),
        vec![PickerEntry {
            label: preview.title.clone(),
            search_text: preview.search_text(),
            value: PickerValue::Session(preview),
        }],
        PickerKind::Session,
    )));

    app.delete_picker_selection(0).await.unwrap();

    assert!(matches!(app.overlay, Overlay::None));
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Saved session deleted")
    );
    let saved = app
        .session_store
        .get_code_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_none());
}

#[tokio::test]
async fn test_ctrl_d_requires_confirmation_before_delete() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    store
        .save_code_session_with_id(
            &key_id,
            "https://api.example.com",
            "/tmp/demo",
            "session-1234",
            "claude",
            None,
            &[crate::services::session_store::StoredChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            }],
            "hello",
            "hello",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    let preview = SessionPreview {
        key_id: key_id.clone(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hello".to_string(),
        preview_text: "hello".to_string(),
        origin: None,
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.cwd = "/tmp/demo".to_string();
    app.overlay = Overlay::Picker(Box::new(PickerState::ready(
        "Sessions",
        String::new(),
        vec![PickerEntry {
            label: preview.title.clone(),
            search_text: preview.search_text(),
            value: PickerValue::Session(preview),
        }],
        PickerKind::Session,
    )));

    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let saved = app
        .session_store
        .get_code_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_some());
    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected picker overlay");
    };
    assert!(picker.pending_delete.is_some());

    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let saved = app
        .session_store
        .get_code_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_none());
}

#[tokio::test]
async fn test_resume_loaded_failure_restores_previous_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx.clone(), rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "old".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let preview = SessionPreview {
        key_id: app.key.id.clone(),
        key_name: app.key.display_name().to_string(),
        base_url: app.key.base_url.clone(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
        origin: None,
    };

    app.begin_resume_load(preview);
    let request_id = app.loading_resume.as_ref().unwrap().request_id;
    tx.send(RuntimeEvent::ResumeLoaded {
        request_id,
        result: Err("boom".to_string()),
    })
    .unwrap();

    app.handle_runtime_events().await.unwrap();

    assert_eq!(app.history.len(), 1);
    assert_eq!(app.history[0].content, "old");
    assert!(app.loading_resume.is_none());
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("boom")
    );
}

#[tokio::test]
async fn test_resume_resets_agent_engine() {
    // A resumed conversation must drop any live engine so the next turn re-seeds
    // from the loaded history; reusing it would continue the prior thread.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.key = key.clone();
    app.model = "claude".to_string();

    let engine =
        crate::agent::engine::AgentEngine::new("/tmp", "claude", "2026-06-14", &[], &[], 0, 0);
    app.agent_engine = Some(AgentSession {
        key_id: key.id.clone(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });

    let session = LoadedSession {
        key_id: key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "earlier turn".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }],
        // A durable transcript on the resumed session is stashed for the next
        // engine build to restore verbatim (exact tool history).
        engine_messages: Some(vec![
            serde_json::json!({"role": "user", "content": "earlier turn"}),
            serde_json::json!({"role": "assistant", "content": "earlier reply"}),
        ]),
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert!(
        app.agent_engine.is_none(),
        "resume must drop the prior engine so the next turn re-seeds"
    );
    assert_eq!(app.session_id, "resumed");
    assert_eq!(app.history.len(), 1);
    assert_eq!(
        app.pending_agent_messages.as_ref().map(|m| m.len()),
        Some(2),
        "the durable transcript is stashed for exact restore on the next build"
    );
}

/// The idle footer after `/resume` must estimate from the stashed durable
/// transcript, not the lossy display seed (~10x too small on tool-heavy sessions).
#[tokio::test]
async fn test_resume_footer_estimate_uses_durable_transcript() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.key = key.clone();
    app.model = "claude".to_string();

    let fat = "x".repeat(200_000);
    let session = LoadedSession {
        key_id: key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "earlier turn".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }],
        engine_messages: Some(vec![
            serde_json::json!({"role": "user", "content": "earlier turn"}),
            serde_json::json!({"role": "tool", "tool_call_id": "t1", "content": fat}),
        ]),
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert!(
        app.context_is_estimate,
        "post-resume fill is an estimate until measured"
    );
    assert!(
        app.context_tokens >= 20_000,
        "estimate must reflect the ~25k-token transcript, got {}",
        app.context_tokens
    );
}

/// Resume restores the stored spend and billed model verbatim — an alias with no
/// snapshot pricing must not zero the figure the session already accumulated.
#[tokio::test]
async fn test_resume_restores_session_cost_and_billed_model() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    store
        .save_code_session_with_id(
            &key.id,
            &key.base_url,
            "/tmp/proj",
            "cost-sess",
            "aivo/starter",
            Some("deepseek-v4-flash"),
            &[],
            "t",
            "p",
            SessionTokens {
                prompt_tokens: 5,
                completion_tokens: 176,
                ..Default::default()
            },
            0.000_065,
        )
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.key = key;
    let session = LoadedSession {
        key_id,
        session_id: "cost-sess".to_string(),
        raw_model: "aivo/starter".to_string(),
        messages: vec![],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert_eq!(app.session_cost_usd, 0.000_065);
    assert_eq!(app.billed_model.as_deref(), Some("deepseek-v4-flash"));
}

#[tokio::test]
async fn test_resume_does_not_overwrite_persisted_default_model() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    store.set_code_model(&key_id, "aivo/starter").await.unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.key = key.clone();
    app.raw_model = "aivo/starter".to_string();

    let session = LoadedSession {
        key_id: key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "google/gemma-4-31b-it".to_string(),
        messages: vec![ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "earlier turn".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }],
        engine_messages: None,
        pristine_import: false,
        source_newer: false,
        import_fidelity: None,
        plan_state: None,
        image_descriptions: None,
    };
    app.apply_loaded_session(session).await.unwrap();

    assert_eq!(
        app.raw_model, "google/gemma-4-31b-it",
        "the resumed conversation adopts its own model in memory"
    );
    assert_eq!(
        app.session_store
            .get_code_model(&key_id)
            .await
            .unwrap()
            .as_deref(),
        Some("aivo/starter"),
        "resume must NOT rewrite the persisted per-key default"
    );
    assert!(
        app.session_store
            .get_last_selection()
            .await
            .unwrap()
            .and_then(|sel| sel.model)
            .is_none(),
        "resume must NOT write the global last-selection model"
    );
}

#[tokio::test]
async fn test_resume_snapshots_scope_by_cwd() {
    // Sessions persist under their real launch dir. `/resume` is directory-
    // scoped: `Some(dir)` returns only that dir's sessions, while `None` (the
    // explicit-id fallback) returns every session across all dirs.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    // One session under an old ephemeral sandbox path (saved directly).
    store
        .save_code_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/aivo-chat-old",
            "sandbox-sess",
            "claude",
            None,
            &[crate::services::session_store::StoredChatMessage {
                model: None,
                role: "user".into(),
                content: "older".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            }],
            "older",
            "older",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    // One persisted through the app under the real launch dir.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/aivo-chat-99999".to_string();
    app.real_cwd = "/home/me/project".to_string();
    app.session_id = "real-cwd-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "remember me".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    assert_eq!(app.persist_cwd(), "/home/me/project"); // logs key on real dir
    app.persist_history().await.unwrap();

    // Unscoped (explicit-id fallback): both, newest first.
    let all = load_resume_snapshots(&store, None).await.unwrap();
    let all_ids: Vec<&str> = all.iter().map(|s| s.session_id.as_str()).collect();
    assert!(all_ids.contains(&"real-cwd-sess"), "got {all_ids:?}");
    assert!(all_ids.contains(&"sandbox-sess"), "got {all_ids:?}");

    // Scoped to the launch dir: only that dir's session.
    let scoped = load_resume_snapshots(&store, Some("/home/me/project"))
        .await
        .unwrap();
    let scoped_ids: Vec<&str> = scoped.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(scoped_ids, vec!["real-cwd-sess"], "got {scoped_ids:?}");

    // Scoped to the old sandbox dir: only the sandbox session.
    let sandbox = load_resume_snapshots(&store, Some("/tmp/aivo-chat-old"))
        .await
        .unwrap();
    let sandbox_ids: Vec<&str> = sandbox.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(sandbox_ids, vec!["sandbox-sess"], "got {sandbox_ids:?}");
}

#[tokio::test]
async fn test_resume_snapshots_keep_key_removed_sessions() {
    // A session whose stored key no longer exists (interrupted delete-cascade,
    // legacy data) stays listed — labeled — so /resume agrees with `aivo logs`;
    // resuming falls back to the live key.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    store
        .save_code_session_with_id(
            "ghost-key",
            "https://api.example.com",
            "/tmp/demo",
            "orphaned-sess",
            "claude",
            None,
            &one_user_message("hello"),
            "hello",
            "hello",
            crate::services::session_store::SessionTokens::default(),
            0.0,
        )
        .await
        .unwrap();

    let rows = load_resume_snapshots(&store, Some("/tmp/demo"))
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "key-removed session must stay listed");
    assert_eq!(rows[0].session_id, "orphaned-sess");
    assert_eq!(rows[0].key_name, "key removed");
    assert!(rows[0].key_id.is_empty());
}
