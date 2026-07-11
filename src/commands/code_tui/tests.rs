use super::*;
use chrono::Duration as ChronoDuration;
use tempfile::TempDir;

#[test]
fn test_matches_fuzzy() {
    assert!(matches_fuzzy("g4", "gpt-4o"));
    assert!(matches_fuzzy("", "anything"));
    assert!(!matches_fuzzy("xyz", "gpt-4o"));
}

#[test]
fn test_chat_mouse_enabled_policy() {
    // No override: on everywhere except Termux, where taps must keep toggling
    // the soft keyboard instead of being eaten as mouse events.
    assert!(chat_mouse_enabled_for(None, false));
    assert!(!chat_mouse_enabled_for(None, true));
    // Explicit override wins in both directions, even under Termux.
    assert!(!chat_mouse_enabled_for(Some("1"), false));
    assert!(!chat_mouse_enabled_for(Some("yes"), true));
    assert!(chat_mouse_enabled_for(Some("0"), true));
    assert!(chat_mouse_enabled_for(Some("false"), true));
}

#[test]
fn test_chat_swipe_scroll_policy() {
    // No override: on under Termux (swipes arrive as arrows there), off elsewhere.
    assert!(!chat_swipe_scroll_enabled_for(None, false));
    assert!(chat_swipe_scroll_enabled_for(None, true));
    // Explicit override wins either way.
    assert!(chat_swipe_scroll_enabled_for(Some("1"), false));
    assert!(chat_swipe_scroll_enabled_for(Some("yes"), false));
    assert!(!chat_swipe_scroll_enabled_for(Some("0"), true));
    assert!(!chat_swipe_scroll_enabled_for(Some("false"), true));
}

#[test]
fn test_cursor_position_multiline() {
    assert_eq!(cursor_position("hello", 5, 10, 2), (7, 0));
    assert_eq!(cursor_position("hello\nworld", 11, 10, 2), (7, 1));
    assert_eq!(cursor_position("hello\nworld", 6, 10, 2), (2, 1));
    assert_eq!(cursor_position("hello\nworld", 0, 10, 2), (2, 0));
}

#[test]
fn test_truncate_path_left_uses_display_width_for_cjk() {
    // 4 CJK glyphs = 8 columns but only 4 chars; the whole path is 12 chars yet
    // 16 columns. A char-count check (≤12) would keep it unshortened and overflow
    // the 12-column budget; display width must truncate on a segment boundary.
    let out = truncate_path_left("aaa/bbb/项目目录", 12);
    assert_eq!(out, "…/项目目录");
    assert!(
        display_width(&out) <= 12,
        "overflows {} cols",
        display_width(&out)
    );
}

#[test]
fn test_cursor_position_uses_display_width_for_cjk() {
    assert_eq!(
        cursor_position("最新的软件开发工具", "最新的软件开发工具".len(), 30, 2),
        (20, 0)
    );
}

#[test]
fn test_cursor_position_wraps_after_prefix_width() {
    // width 8, prompt indent 2 → 6 text cols per row. "abcdef" fills row 0, "gh"
    // wraps to row 1. Wrapped rows carry the same 2-col hanging indent, so the
    // cursor after "gh" sits at column 2 + 2 = 4 (not column 2).
    assert_eq!(cursor_position("abcdefgh", 8, 8, 2), (4, 1));
}

#[test]
fn test_composer_cursor_position_offsets_attachment_rows() {
    let (x, y) = cursor_position("hello", 5, 20, 2);
    assert_eq!((x, y.saturating_add(1)), (7, 1));
    let (x, y) = cursor_position("", 0, 20, 2);
    assert_eq!((x, y.saturating_add(2)), (2, 2));
}

#[test]
fn test_composer_visual_rows_wraps_with_hanging_indent() {
    // One word, no break opportunity → hard-wraps mid-word: "ij" spills to row 1.
    let rows = composer_visual_rows("abcdefghij", 8);
    assert_eq!(rows, vec![(0, 8), (8, 10)]);
    // A trailing newline yields a final empty row so the caret can rest there.
    assert_eq!(composer_visual_rows("ab\n", 8), vec![(0, 2), (3, 3)]);
    // An empty draft is a single empty row.
    assert_eq!(composer_visual_rows("", 8), vec![(0, 0)]);
}

#[test]
fn test_composer_visual_rows_wraps_at_word_boundary() {
    // "world" overflows → moves whole to row 1; the space stays on row 0.
    assert_eq!(
        composer_visual_rows("hello world", 8),
        vec![(0, 6), (6, 11)]
    );
    // Word longer than the row breaks at the boundary, then hard-wraps mid-word.
    assert_eq!(
        composer_visual_rows("a bcdefghij", 5),
        vec![(0, 2), (2, 7), (7, 11)]
    );
}

#[test]
fn test_composer_cursor_visual_up_down_on_wrapped_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // width 10 → 8 text cols. "abcdefghij" → row0 (0..8), row1 (8..10).
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();
    app.cursor = 9; // row 1, one char in (display col 3)

    // Up moves to the same display column on row 0 (after 'a').
    assert!(app.composer_cursor_up());
    assert_eq!(app.cursor, 1);
    // Already on the top row → false, so the caller recalls history instead.
    assert!(!app.composer_cursor_up());
    // Down returns to row 1 at the same column.
    assert!(app.composer_cursor_down());
    assert_eq!(app.cursor, 9);
    // Bottom row → false.
    assert!(!app.composer_cursor_down());
}

#[test]
fn test_composer_wrapped_row_renders_hanging_indent() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();

    let text = app.render_composer_text();
    assert_eq!(plain_text_from_spans(&text.lines[0].spans), "> abcdefgh");
    // Wrapped continuation aligns under the text with a 2-col indent.
    assert_eq!(plain_text_from_spans(&text.lines[1].spans), "  ij");
}

#[test]
fn test_composer_highlights_shell_command_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 40, 5));

    // A plain message draft renders in the normal text color.
    app.draft = "hello".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(TEXT));

    // A `!cmd` draft is tinted in the magenta shell hue to signal shell mode.
    app.draft = "!ls -la".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(SHELL));

    // `!!` is the literal-`!` escape (sent to the model), not shell mode.
    app.draft = "!!not a command".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(TEXT));
}

#[test]
fn test_composer_mouse_click_maps_to_cursor_offset() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();

    let click = |column: u16, row: u16| MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Column 4 on row 0 is the 'c' cell ('>'=0, ' '=1, 'a'=2, 'b'=3, 'c'=4) → caret before 'c'.
    assert_eq!(app.composer_offset_for_mouse(click(4, 0)), Some(2));
    // Row 1 holds "  ij"; clicking the 'j' cell (col 3) lands the caret before 'j'.
    assert_eq!(app.composer_offset_for_mouse(click(3, 1)), Some(9));
    // A click outside the composer misses.
    assert_eq!(app.composer_offset_for_mouse(click(40, 40)), None);
}

#[tokio::test]
async fn test_streamed_steps_keep_scroll_when_user_scrolled_up() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // The user scrolled up to read earlier output (follow disengaged).
    app.follow_output = false;

    app.apply_agent_tool_call(
        Some("1".to_string()),
        "read_file".to_string(),
        serde_json::json!({"path": "x"}),
        vec![],
        None,
    );
    assert!(
        !app.follow_output,
        "a tool call must not snap the view to the bottom"
    );
    app.apply_agent_tool_result("ok".to_string());
    assert!(
        !app.follow_output,
        "a tool result must not snap the view to the bottom"
    );
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "pending"}]));
    assert!(
        !app.follow_output,
        "a plan update must not snap the view to the bottom"
    );

    // While following (at the bottom), streamed steps keep following.
    app.follow_output = true;
    app.apply_agent_tool_result("more".to_string());
    assert!(app.follow_output);
}

#[test]
fn test_insert_pasted_text_updates_draft_and_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "abc".to_string();
    app.cursor = 1;

    app.insert_pasted_text("XYZ");

    assert_eq!(app.draft, "aXYZbc");
    assert_eq!(app.cursor, 4);
}

#[test]
fn test_question_mark_is_not_help_shortcut() {
    let question = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE);
    let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
    assert!(!is_help_shortcut(question));
    assert!(is_help_shortcut(f1));
}

#[tokio::test]
async fn test_help_overlay_groups_lists_every_command_and_scrolls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `/help` opens the overlay at the top of its body.
    app.open_help_overlay();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));

    // A tall render shows the top: the section header, every purpose group, and
    // every command label (commands sit before the fold, so they all fit).
    let (top, _) = render_full_screen(&mut app, 90, 70);
    assert!(top.contains("Slash commands"), "missing header:\n{top}");
    for group in [
        "Session",
        "Model & key",
        "Context",
        "Skills & tools",
        "Autonomous",
    ] {
        assert!(top.contains(group), "missing command group {group}:\n{top}");
    }
    for command in SLASH_COMMANDS {
        // Account commands are hidden on this (non-aivo) test key.
        if !app.slash_command_visible(command.name) {
            assert!(
                !top.contains(command.help_label),
                "hidden command {} leaked into help:\n{top}",
                command.help_label
            );
            continue;
        }
        assert!(
            top.contains(command.help_label),
            "command {} missing from help:\n{top}",
            command.help_label
        );
    }
    // The aivo-only account group is absent on a BYOK key.
    assert!(
        !top.contains("aivo account"),
        "account group shown on a non-aivo key:\n{top}"
    );
    // Every visible command is grouped, so the completeness-guard "More" bucket is empty.
    assert!(
        !top.contains("More"),
        "unexpected ungrouped commands:\n{top}"
    );

    // End scrolls to the bottom; the keybindings + text-entry tips are reachable.
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let (bottom, _) = render_full_screen(&mut app, 90, 24);
    let scrolled = match app.overlay {
        Overlay::Help { scroll } => scroll,
        _ => panic!("help overlay closed unexpectedly"),
    };
    assert!(scrolled > 0, "End did not scroll the help body");
    assert!(
        bottom.contains("Keybindings") || bottom.contains("Text entry"),
        "bottom sections not reachable by scrolling:\n{bottom}"
    );
    assert!(
        bottom.contains("shell command"),
        "text-entry tip not reachable:\n{bottom}"
    );

    // Home snaps back to the top; Esc closes.
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_account_commands_gated_to_aivo_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The default test key is BYOK → account commands are hidden and refused.
    assert!(!app.is_aivo_account_key());
    for name in ["login", "logout", "usage"] {
        assert!(
            !app.slash_command_visible(name),
            "/{name} should be hidden on a BYOK key"
        );
    }
    // `/usage` on a BYOK key is a no-op with a hint — no task spawned.
    app.run_usage_command().await;
    assert!(app.account_task.is_none());
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("aivo provider")),
        "expected the aivo-only hint, got {:?}",
        app.notice
    );

    // On the bundled aivo starter key the three commands surface.
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();
    assert!(app.is_aivo_account_key());
    for name in ["login", "logout", "usage"] {
        assert!(
            app.slash_command_visible(name),
            "/{name} should show on the aivo key"
        );
    }
    // The `/` menu now offers them.
    let entries = app.matching_command_entries("login");
    assert!(
        entries.iter().any(|e| e.label() == "/login"),
        "/login missing from the menu on the aivo key"
    );
}

#[tokio::test]
async fn test_account_login_card_flow_and_stale_generation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();

    // Stand in for `run_login_command` (no network poll): notice, no card yet.
    app.account_gen = 7;
    app.notice = Some((MUTED, "Starting sign-in…".to_string()));
    assert!(app.account_login.is_none());

    // The device code + URL arrive → the card appears, notice cleared.
    app.apply_account_login_prompt(
        7,
        Ok((
            "WXYZ-1234".to_string(),
            "https://getaivo.dev/device?code=WXYZ-1234".to_string(),
        )),
    );
    assert!(app.notice.is_none(), "starting notice not cleared");
    let (frame, _) = render_full_screen(&mut app, 80, 24);
    assert!(frame.contains("sign in to aivo"), "title missing:\n{frame}");
    assert!(frame.contains("WXYZ-1234"), "code missing:\n{frame}");
    assert!(
        frame.contains("Waiting for approval…"),
        "status missing:\n{frame}"
    );
    assert!(
        frame.contains("Enter open browser"),
        "key hints missing:\n{frame}"
    );
    // Empty session parks the composer at top → the card takes the space below.
    assert!(
        frame.find("Ask, plan, or build").unwrap() < frame.find("sign in to aivo").unwrap(),
        "card should sit below the parked composer:\n{frame}"
    );

    // A prompt stamped with a stale generation is ignored (card stays).
    app.apply_account_login_prompt(3, Err("boom".to_string()));
    assert!(app.account_login.is_some(), "stale error dropped the card");

    // Esc with a non-empty composer belongs to the draft — the card stays.
    app.draft = "half a thought".to_string();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account_login.is_some(), "Esc stole the draft's key");

    // Esc on an empty composer cancels: card gone, generation bumped.
    app.draft.clear();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account_login.is_none());
    assert_ne!(app.account_gen, 7, "cancel must invalidate the flow");

    // A late success for the cancelled flow is dropped (no login notice).
    app.apply_account_login_done(7, Ok("Logged in as x".to_string()))
        .await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("cancelled")),
        "late result overwrote the cancel notice: {:?}",
        app.notice
    );

    // A current-generation success drops the TUI's starter catalog.
    let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
    app.cache
        .set(sentinel, vec!["aivo/starter".to_string()])
        .await;
    let account_gen = app.account_gen;
    app.apply_account_login_done(account_gen, Ok("Logged in as x".to_string()))
        .await;
    assert!(
        app.cache.model_ids(sentinel).await.is_none(),
        "login left the TUI's starter catalog stale"
    );
}

#[tokio::test]
async fn test_account_usage_runs_the_cli_as_a_local_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();

    // `/usage` runs the CLI itself through the `!` machinery.
    app.run_usage_command().await;
    let run = app
        .local_command
        .as_ref()
        .expect("no local command spawned");
    assert_eq!(run.command, "aivo account usage");
    // Kill it before it does anything — this test is wiring-only.
    app.interrupt_local_command().await.unwrap();
    assert!(app.local_command.is_none());

    // A second `/usage` while one is still streaming is refused like any `!cmd`.
    app.run_usage_command().await;
    assert!(app.local_command.is_some());
    app.run_usage_command().await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("already running")),
        "expected the busy notice, got {:?}",
        app.notice
    );
    app.interrupt_local_command().await.unwrap();
}

#[tokio::test]
async fn test_logout_confirm_card_and_stale_done() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The confirm card owns the keyboard: n dismisses without unlinking.
    app.pending_logout = Some("me@example.com".to_string());
    let (frame, _) = render_full_screen(&mut app, 80, 24);
    assert!(
        frame.contains("sign out of aivo"),
        "title missing:\n{frame}"
    );
    assert!(
        frame.contains("me@example.com"),
        "account missing:\n{frame}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.pending_logout.is_none());
    assert!(app.account_task.is_none(), "deny must not spawn an unlink");

    // A stale unlink result is ignored; the current one lands as a notice.
    let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
    app.cache
        .set(sentinel, vec!["aivo/starter".to_string()])
        .await;
    app.account_gen = 4;
    app.apply_account_logout_done(1, Ok(())).await;
    assert!(
        app.notice
            .as_ref()
            .is_none_or(|(_, m)| !m.contains("Logged out")),
        "stale result produced a notice: {:?}",
        app.notice
    );
    assert!(
        app.cache.model_ids(sentinel).await.is_some(),
        "stale result cleared the catalog"
    );
    app.apply_account_logout_done(4, Ok(())).await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("Logged out")),
        "expected the logout confirmation, got {:?}",
        app.notice
    );
    // The TUI's own catalog (distinct from the shared instance) dropped too.
    assert!(
        app.cache.model_ids(sentinel).await.is_none(),
        "logout left the TUI's starter catalog stale"
    );
}

#[test]
fn test_sgr_mouse_frag_step_classifies_fragment() {
    use super::event_loop_impl::{FragStep, sgr_mouse_frag_step};
    // Valid growing prefixes of `[<{params}`.
    assert!(matches!(sgr_mouse_frag_step("["), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<"), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<6"), FragStep::Continue));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23"),
        FragStep::Continue
    ));
    // Complete reports (press `M` / release `m`).
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23M"),
        FragStep::Final
    ));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23m"),
        FragStep::Final
    ));
    // Not SGR mouse: `[` not followed by `<`, empty params, stray char.
    assert!(matches!(sgr_mouse_frag_step("[h"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<M"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<6x"), FragStep::Invalid));
}

#[test]
fn test_parse_sgr_scroll_only_wheel_buttons() {
    use super::event_loop_impl::parse_sgr_scroll;
    let up = parse_sgr_scroll("[<64;56;23M").unwrap();
    assert!(matches!(up.kind, MouseEventKind::ScrollUp));
    assert_eq!((up.column, up.row), (55, 22)); // SGR is 1-based, MouseEvent 0-based
    let down = parse_sgr_scroll("[<65;1;1m").unwrap();
    assert!(matches!(down.kind, MouseEventKind::ScrollDown));
    assert_eq!((down.column, down.row), (0, 0));
    assert!(parse_sgr_scroll("[<0;5;5M").is_none()); // left-button press, not a wheel
    assert!(parse_sgr_scroll("[<64;5M").is_none()); // missing the row param
    assert!(parse_sgr_scroll("[<64;5;5;5M").is_none()); // an extra param
}

// A fast mouse-wheel report (`\x1b[<65;…M`) that crossterm splits at its ESC must
// not leak its tail into the composer nor spuriously close the open overlay — the
// bare Esc is withheld, the tail swallowed, and the scroll re-synthesized.
#[tokio::test]
async fn test_split_mouse_report_is_reassembled_not_leaked() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    // The leading ESC arrives alone; it is held, not acted on.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "Esc closed overlay early"
    );

    // The tail `[<65;56;23M` (wheel-down) follows as literal chars in the burst.
    for c in "[<65;56;23M".chars() {
        let step = app
            .step_esc_reassembly(
                &mut esc,
                Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
            )
            .await
            .unwrap();
        assert!(
            matches!(step, EscStep::Consumed),
            "char {c:?} leaked through"
        );
    }

    // Nothing typed, overlay still open, and the wheel-down scrolled the body.
    assert_eq!(app.draft, "", "mouse tail leaked into composer");
    match app.overlay {
        Overlay::Help { scroll } => assert_eq!(scroll, 3, "re-synthesized scroll missing"),
        _ => panic!("help overlay was spuriously closed by the split ESC"),
    }
}

#[tokio::test]
async fn test_lone_esc_still_closes_overlay_at_burst_end() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(matches!(app.overlay, Overlay::Help { .. }));

    // Burst ends with the Esc still held: it was real, so flushing closes help.
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_esc_then_non_mouse_text_is_replayed_losslessly() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // No overlay: characters reach the composer.

    let mut esc = EscReassembly::Idle;
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    // `h` breaks the SGR-mouse shape, so the held `[` and this `h` are real text.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert_eq!(app.draft, "[h", "non-mouse run after Esc was not replayed");
}

#[test]
fn test_cursor_movement_basic() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();

    app.cursor_left();
    assert_eq!(app.cursor, 10); // before 'd'

    app.cursor_right();
    assert_eq!(app.cursor, 11); // end

    app.cursor_home();
    assert_eq!(app.cursor, 0);

    app.cursor_end();
    assert_eq!(app.cursor, 11);
}

#[test]
fn test_cursor_insert_delete() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hllo".to_string();
    app.cursor = 1; // after 'h'

    app.insert_char_at_cursor('e');
    assert_eq!(app.draft, "hello");
    assert_eq!(app.cursor, 2);

    app.cursor = app.draft.len();
    app.delete_char_before_cursor();
    assert_eq!(app.draft, "hell");
    assert_eq!(app.cursor, 4);

    app.cursor = 0;
    app.delete_char_at_cursor();
    assert_eq!(app.draft, "ell");
    assert_eq!(app.cursor, 0);
}

#[test]
fn bare_url_in_mcp_add_becomes_url_config() {
    use super::session_impl::bare_url_to_config;
    // A bare http(s) URL → a {url} server config (no JSON typing needed).
    let json = bare_url_to_config("https://mcp.linear.app/mcp").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://mcp.linear.app/mcp");
    assert!(bare_url_to_config("http://127.0.0.1:8080/mcp").is_some());
    // Only the first token is taken (a URL has no spaces).
    let json = bare_url_to_config("https://h/mcp  oops").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://h/mcp");
    // A command line or a JSON block is NOT a bare URL (handled elsewhere).
    assert!(bare_url_to_config("npx -y @scope/server").is_none());
    assert!(bare_url_to_config(r#"{"url":"https://h"}"#).is_none());
}

#[tokio::test]
async fn ctrl_d_deletes_forward_in_prompt() {
    // Ctrl+D in the composer is emacs `delete-char` (forward), not transcript
    // scroll — it must reach the editor, deleting the char under the cursor.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello".to_string();
    app.cursor = 0;
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.draft, "ello");
    assert_eq!(app.cursor, 0);
}

#[test]
fn test_cursor_word_movement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world foo".to_string();
    app.cursor = app.draft.len();

    app.cursor_word_left();
    assert_eq!(app.cursor, 12); // start of 'foo'

    app.cursor_word_left();
    assert_eq!(app.cursor, 6); // start of 'world'

    app.cursor_word_right();
    assert_eq!(app.cursor, 11); // end of 'world'
}

#[test]
fn test_visible_command_menu_filters_and_hides_escaped_slash() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = 3;
    app.sync_command_menu_state();
    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.entries.len(), 2); // "mo" → model + memory
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
    assert!(matches!(
        menu.entries[1],
        ComposerMenuEntry::Command(command) if command.name == "memory"
    ));
    assert_eq!(menu.selected, Some(0));

    app.draft = "//literal".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());

    app.draft = "/model claude".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_filter_slash_commands_prefers_prefix_matches() {
    let matches = filter_slash_commands("m");
    assert_eq!(matches.first().map(|command| command.name), Some("model"));
}

#[test]
fn test_command_menu_area_prefers_below_when_above_space_is_tight() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) = command_menu_area(composer, frame, 6, None);
    assert_eq!(placement, CommandMenuPlacement::Below);
    assert!(area.y >= composer.y + composer.height);
}

#[test]
fn test_command_menu_area_respects_sticky_placement() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) =
        command_menu_area(composer, frame, 6, Some(CommandMenuPlacement::Above));
    assert_eq!(placement, CommandMenuPlacement::Above);
    assert_eq!(area.y, frame.y);
}

#[test]
fn test_command_menu_query_change_keeps_existing_placement_until_reopen() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.command_menu.placement = Some(CommandMenuPlacement::Above);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.command_menu.query = "m".to_string();
    app.command_menu.dismissed = false;

    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Above)
    );

    app.dismiss_command_menu();
    app.draft = "/mod".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(app.command_menu.placement, None);
}

#[test]
fn test_bare_slash_filtering_keeps_detected_placement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    app.command_menu.placement = Some(CommandMenuPlacement::Below);

    app.draft = "/new".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Below)
    );
}

#[test]
fn test_insert_selected_command_uses_argument_space_when_needed() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/model ");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_insert_selected_command_omits_space_for_zero_arg_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/ex".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/exit");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[tokio::test]
async fn test_ctrl_p_navigate_command_menu_before_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["previous prompt".to_string()];
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.selected, Some(menu.entries.len() - 1));
    assert_eq!(app.draft, "/");
    assert!(app.draft_history_index.is_none());
}

#[tokio::test]
async fn test_history_nav_past_recalled_slash_command() {
    // Recalling a `/command` from history must not pop the command menu and
    // steal the arrow keys — ↑ keeps walking history past it.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["older".to_string(), "/help".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    // First ↑ recalls the newest entry (a slash command); the menu stays hidden.
    assert_eq!(app.draft, "/help");
    assert_eq!(app.draft_history_index, Some(1));
    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    // Second ↑ continues up through history rather than navigating the menu.
    assert_eq!(app.draft, "older");
    assert_eq!(app.draft_history_index, Some(0));

    // ↓ steps back down past the slash command, too.
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "/help");
    assert_eq!(app.draft_history_index, Some(1));
}

#[tokio::test]
async fn test_swipe_scroll_arrows_scroll_transcript() {
    // Mobile: a swipe arrives as Up/Down; with an empty composer they scroll.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = true;
    app.draft_history = vec!["older".to_string()];
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.transcript_scroll < 50);
    assert!(!app.follow_output);
    assert!(app.draft.is_empty());
    assert!(app.draft_history_index.is_none());

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);
    assert!(app.draft.is_empty());
}

#[tokio::test]
async fn test_arrows_recall_history_without_swipe_scroll() {
    // Desktop: bare Up still walks draft history, transcript stays pinned.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = false;
    app.draft_history = vec!["older".to_string()];
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "older");
    assert_eq!(app.draft_history_index, Some(0));
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);
}

#[tokio::test]
async fn test_swipe_scroll_yields_to_multiline_cursor() {
    // Swipe-scroll still yields to caret movement in a multi-line draft; it only
    // scrolls at the top edge.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = true;
    app.draft = "line one\nline two".to_string();
    app.cursor = app.draft.len();
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.cursor < app.draft.len());
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.transcript_scroll < 50);
}

#[tokio::test]
async fn test_escape_dismisses_command_menu_until_query_changes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_some());

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    let menu = app.visible_command_menu().unwrap();
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
}

#[tokio::test]
async fn test_enter_executes_selected_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert_eq!(app.cursor, 0);
    assert!(should_exit);
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_render_command_menu_rows_shows_empty_state() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: Vec::new(),
        selected: None,
    };
    let lines = render_command_menu_rows(&menu, 32);
    let plain = plain_text_from_spans(&lines[0].spans);
    assert_eq!(plain, "No matching command");
}

#[test]
fn test_render_command_menu_rows_aligns_description_column() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: vec![
            ComposerMenuEntry::Command(&SLASH_COMMANDS[0]),
            ComposerMenuEntry::Command(&SLASH_COMMANDS[2]),
        ],
        selected: Some(0),
    };
    let lines = render_command_menu_rows(&menu, 48);
    let first = plain_text_from_spans(&lines[0].spans);
    let second = plain_text_from_spans(&lines[1].spans);
    let first_desc = first.find("start a fresh session").unwrap();
    let second_desc = second.find("resume a saved session").unwrap();
    assert_eq!(first_desc, second_desc);
}

#[test]
fn test_collect_attach_path_suggestions_lists_matching_entries() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("alpha.txt"), "hi").unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let entries = collect_attach_path_suggestions(temp_dir.path().to_str().unwrap(), "a");

    assert!(entries.iter().any(|entry| entry.label == "assets/"));
    assert!(entries.iter().any(|entry| entry.label == "alpha.txt"));
}

#[test]
fn test_attach_query_uses_path_menu_and_tab_inserts_selected_path() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = temp_dir.path().to_string_lossy().into_owned();
    app.draft = "/attach a".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.kind, MenuKind::AttachPath);
    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/attach assets/");
    // Menu stays open after tab on a directory so the user can continue navigating.
    assert!(app.visible_command_menu().is_some());
}

#[test]
fn test_delete_word_backward() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();
    app.delete_word_backward();
    assert_eq!(app.draft, "hello ");
    assert_eq!(app.cursor, 6);
}

#[test]
fn test_kill_to_end_of_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello\nworld".to_string();
    app.cursor = 2;
    app.kill_to_end_of_line();
    assert_eq!(app.draft, "he\nworld");
    assert_eq!(app.cursor, 2);
}

#[tokio::test]
async fn test_ctrl_l_clears_prompt_without_exiting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();
    app.draft_attachments.push(MessageAttachment {
        name: "notes.md".to_string(),
        mime_type: "text/markdown".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./notes.md".to_string(),
        },
    });

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft.is_empty());
    assert!(app.draft_attachments.is_empty());
    assert_eq!(app.cursor, 0);
}

#[tokio::test]
async fn test_ctrl_l_clears_attachment_only_prompt() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "image.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./image.png".to_string(),
        },
    });

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft_attachments.is_empty());
}

#[tokio::test]
async fn test_ctrl_l_clears_history_navigation_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["older".to_string(), "newer".to_string()];
    app.history_prev();
    assert!(app.draft_history_index.is_some());
    assert!(!app.draft.is_empty());

    app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert!(app.draft_history_index.is_none());
    assert!(app.draft_history_stash.is_none());
}

#[tokio::test]
async fn test_submit_slash_command_records_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/help".to_string();
    app.cursor = app.draft.len();

    let should_exit = app.submit_draft().await.unwrap();

    // The command ran (help overlay opened) and the draft cleared...
    assert!(!should_exit);
    assert!(app.draft.is_empty());
    assert!(matches!(app.overlay, Overlay::Help { .. }));
    // ...but the typed `/help` is now recallable from input history, just like
    // a normal message or `!cmd`.
    assert_eq!(app.draft_history, vec!["/help".to_string()]);
    app.history_prev();
    assert_eq!(app.draft, "/help");
}

#[tokio::test]
async fn test_record_draft_history_dedupes_consecutive() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.record_draft_history("/help");
    app.record_draft_history("/help"); // consecutive dup → ignored
    app.record_draft_history("/model");
    app.record_draft_history("/help"); // non-adjacent repeat → kept

    assert_eq!(
        app.draft_history,
        vec![
            "/help".to_string(),
            "/model".to_string(),
            "/help".to_string()
        ]
    );
}

#[tokio::test]
async fn test_ctrl_c_requires_confirmation_to_exit() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!should_exit);
    assert!(app.exit_confirm_pending);
    assert!(app.notice.is_some());

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(should_exit);
}

#[tokio::test]
async fn test_ctrl_c_pending_resets_on_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.exit_confirm_pending);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.exit_confirm_pending);
    assert!(app.notice.is_none());
}

#[tokio::test]
async fn test_ctrl_x_ctrl_e_chord_requests_external_edit() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.pending_ctrl_x);
    assert!(!app.pending_external_edit);

    app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!app.pending_ctrl_x);
    assert!(app.pending_external_edit);
}

#[tokio::test]
async fn test_ctrl_x_chord_cancelled_by_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.pending_ctrl_x);

    app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.pending_ctrl_x);
    assert!(!app.pending_external_edit);
    assert_eq!(app.draft, "h");

    app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!app.pending_external_edit);
}

#[tokio::test]
async fn prewarm_cursor_session_noops_for_non_cursor_key() {
    // Non-cursor key => prewarm must not spawn cursor-agent or arm the handle.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.key.is_cursor_acp());
    app.prewarm_cursor_session();
    assert!(app.cursor_prewarm.is_none());
}

fn make_test_app(
    tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) -> CodeTuiApp {
    CodeTuiApp {
        // A unique throwaway store — NEVER the real `~/.config/aivo` (which
        // `SessionStore::new()` points at). Tests that drive a save (persist /
        // flush / turn-finish) would otherwise pollute the user's real config.
        session_store: {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("aivo-test-{}-{n}", std::process::id()));
            SessionStore::with_path(dir.join("config.json"))
        },
        // Same isolation: `new()` points at the real models-cache.json and the
        // account flows write through this instance.
        cache: {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("aivo-test-cache-{}-{n}", std::process::id()));
            ModelsCache::with_path(dir.join("models-cache.json"))
        },
        client: reqwest::Client::new(),
        key: ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            "https://api.anthropic.com".to_string(),
            None,
            String::new(),
        ),
        copilot_tm: None,
        cwd: String::new(),
        real_cwd: String::new(),
        git_branch: None,
        git_branch_checked_at: None,
        raw_model: String::new(),
        model: String::new(),
        billed_model: None,
        turn_model: None,
        format: ChatFormat::OpenAI,
        history: Vec::new(),
        draft: String::new(),
        draft_attachments: Vec::new(),
        cursor: 0,
        command_menu: CommandMenuState::default(),
        skill_commands: Vec::new(),
        last_subagents: Vec::new(),
        mcp_configured_count: 0,
        welcome_tip_index: 0,
        welcome_tip_rotated_at: None,
        draft_history: Vec::new(),
        draft_history_all: Vec::new(),
        draft_history_index: None,
        draft_history_stash: None,
        session_id: String::new(),
        overlay: Overlay::None,
        notice: None,
        pending_response: String::new(),
        incoming_buffer: String::new(),
        pending_finish: None,
        pending_reasoning: String::new(),
        pending_submit: None,
        sending: false,
        request_started_at: None,
        compact_before: None,
        last_tool_action: None,
        wait_tick: None,
        last_stream_activity: None,
        subagent_rows: Vec::new(),
        tool_output_tail: std::collections::VecDeque::new(),
        tool_output_partial: String::new(),
        status_display: None,
        turn_output_tokens: 0,
        retrying: false,
        last_usage: None,
        live_usage: None,
        context_tokens: 0,
        session_tokens: crate::services::session_store::SessionTokens::default(),
        session_cost_usd: 0.0,
        context_window: 0,
        context_window_override: None,
        injected_context: None,
        injected_context_summary: None,
        context_is_estimate: true,
        follow_output: true,
        transcript_revision: 0,
        transcript_scroll: 0,
        transcript_width: 0,
        transcript_view_height: 0,
        last_max_scroll: None,
        transcript_hitbox: None,
        jump_to_bottom_hit: None,
        composer_text_area: None,
        composer_scroll: 0,
        transcript_cache: None,
        volatile_tail_cache: None,
        transcript_selection: None,
        transcript_drag_active: false,
        screen_selection: None,
        screen_drag_active: false,
        screen_surface: None,
        screen_region: None,
        drag_autoscroll: None,
        last_autoscroll: None,
        last_click: None,
        selection_flash_until: None,
        scroll_speed: DEFAULT_CHAT_SCROLL_SPEED,
        swipe_scroll: false,
        toast: None,
        tx,
        rx,
        response_task: None,
        resume_task: None,
        resume_request_id: 0,
        loading_resume: None,
        resume_restore_state: None,
        session_preview_cache: std::collections::HashMap::new(),
        session_preview_pending: None,
        session_preview_task: None,
        reduce_motion: false,
        frame_tick: 0,
        picker_hitbox: None,
        overlay_detail_area: None,
        exit_confirm_pending: false,
        goal_stop_confirm_pending: false,
        pending_ctrl_x: false,
        pending_external_edit: false,
        cursor_acp_session: None,
        cursor_prewarm: None,
        cursor_plan_mode: false,
        pending_agent_messages: None,
        goal_mode: None,
        goal_guard_stop: None,
        plan_mode: false,
        plan_exit_pending: false,
        pending_plan: None,
        plan_card_idx: None,
        agent_engine: None,
        agent_route_cache: None,
        mcp_client: None,
        mcp_connecting: false,
        mcp_connect_progress: std::collections::HashMap::new(),
        disabled_mcp_tools: std::collections::HashSet::new(),
        mcp_connect_gen: 0,
        engine_rebuild_pending: false,
        live_share_gen: 0,
        pending_mcp_auth: std::collections::HashMap::new(),
        agent_serve: None,
        agent_permission: None,
        agent_ask: None,
        agent_review: None,
        agent_plan_approval: None,
        agent_auto_approve: false,
        auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        agent_review_edits: false,
        review_edits_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        thinking_enabled: true,
        web_search_enabled: true,
        agent_tools_enabled: true,
        model_supports_thinking: true,
        model_image_input: None,
        cursor_effort_label: None,
        reasoning_effort: None,
        model_reasoning_efforts: Vec::new(),
        queued_messages: Vec::new(),
        steering_queue: SteeringQueue::default(),
        queued_commands: Vec::new(),
        queue_focus: None,
        project_mcp_consent: ProjectMcpConsent::default(),
        pending_mcp_consent: None,
        local_command: None,
        jobs: crate::agent::jobs::JobTable::new(None),
        last_jobs_poll: std::time::Instant::now(),
        jobs_running: 0,
        local_outputs: std::collections::HashMap::new(),
        expanded_output: std::collections::HashSet::new(),
        expanded_thinking: std::collections::HashSet::new(),
        agent_turn_indices: std::collections::HashSet::new(),
        reasoning_durations: std::collections::HashMap::new(),
        turn_durations: std::collections::HashMap::new(),
        turn_notes: std::collections::HashMap::new(),
        reasoning_started_at: None,
        reasoning_elapsed_ms: None,
        installing_skill: None,
        staged_skill_install: None,
        live_share: None,
        live_share_starting: false,
        live_requested: false,
        account_gen: 0,
        account_task: None,
        account_login: None,
        pending_logout: None,
        pending_key_switch: None,
    }
}

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

fn seed_two_exchanges(app: &mut CodeTuiApp) {
    for (role, content) in [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

#[tokio::test]
async fn test_rewind_truncates_history_and_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-1".to_string();
    seed_two_exchanges(&mut app);

    // Rewind to the second user turn (history index 2). No live engine in the
    // test app → conversation-only path (ordinal None).
    app.rewind_to_turn(2, None).await.unwrap();

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

    app.rewind_to_turn(2, None).await.unwrap();

    assert!(app.context_is_estimate, "measured flag must not survive");
    assert_eq!(app.last_usage, None);
    assert!(
        app.context_tokens < 100_000,
        "fill must be re-estimated from the truncated history, got {}",
        app.context_tokens
    );
}

#[tokio::test]
async fn test_session_pricing_falls_back_to_billed_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "aivo/starter".to_string();
    assert!(app.session_pricing().is_none(), "alias alone is unpriced");
    app.billed_model = Some("claude-opus-4-8".to_string());
    assert!(app.session_pricing().is_some(), "billed model resolves");
}

#[tokio::test]
async fn test_rewind_to_first_turn_clears_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-2".to_string();
    seed_two_exchanges(&mut app);

    app.rewind_to_turn(0, None).await.unwrap();

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
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 2);
    // No live engine in the test app → no checkpoints → conversation-only.
    assert!(ordinal.is_none());
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
    // Newest row = the plain-chat duplicate: no checkpoint, conversation-only.
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 2);
    assert!(
        ordinal.is_none(),
        "a non-engine row must not steal the checkpoint"
    );
    assert!(picker.items[0].label.contains("conversation only"));
    // Older row = the real engine turn: keeps its checkpoint and file revert.
    let PickerValue::RewindTurn {
        history_index,
        ordinal,
    } = &picker.items[1].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 0);
    assert_eq!(*ordinal, Some(0));
    assert!(!picker.items[1].label.contains("conversation only"));
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
fn test_composer_empty_lines_align_with_cursor_position() {
    use ratatui::buffer::Buffer;
    use ratatui::widgets::Widget;

    // Empty lines must use Line::from("") (no whitespace prefix) to avoid
    // ratatui WordWrapper producing extra visual rows for whitespace-only Lines.
    let lines = vec![
        Line::from(vec![
            Span::styled("> ", Style::default().fg(USER)),
            Span::styled("hello", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("hel", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("sadf", Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("dsf", Style::default().fg(TEXT)),
        ]),
    ];
    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    let area = Rect::new(0, 0, 20, 10);
    let mut buf = Buffer::empty(area);
    paragraph.render(area, &mut buf);

    let cell_row = |r: u16| -> String {
        (0..20)
            .map(|x| {
                buf.cell((x, r))
                    .unwrap()
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ')
            })
            .collect()
    };

    assert!(cell_row(0).starts_with("> hello"), "row 0");
    assert!(cell_row(1).starts_with("  hel"), "row 1");
    assert!(cell_row(2).starts_with("  sadf"), "row 2");
    assert!(
        cell_row(5).starts_with("  dsf"),
        "row 5: dsf must align with cursor_position y=5"
    );

    // cursor_position must agree
    let (cx, cy) = cursor_position("hello\nhel\nsadf\n\n\ndsf", 17, 20, 2);
    assert_eq!((cx, cy), (2, 5));
}

#[test]
fn test_markdown_renderer_formats_code_and_lists() {
    let lines = render_markdown_lines("## Title\n\n- one\n- two\n\n```rust\nlet x = 1;\n```", 80);
    let plain = lines
        .into_iter()
        .map(|line| line.plain)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(plain.contains("Title"));
    assert!(plain.contains("• one"));
    assert!(plain.contains("rust"));
    assert!(plain.contains("let x = 1;"));
}

#[test]
fn test_markdown_loose_list_renders_tight() {
    // A "loose" list (blank lines between items in source) used to get a blank
    // line between every rendered item — double-spaced. Items should be adjacent.
    let md = "1. first item\n\n2. second item\n\n3. third item";
    let plain: Vec<String> = render_markdown_lines(md, 80)
        .into_iter()
        .map(|l| l.plain)
        .collect();
    let items: Vec<usize> = plain
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("item"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(items.len(), 3, "{plain:?}");
    // Consecutive item rows with no blank between them.
    assert_eq!(items[1] - items[0], 1, "blank between items 1-2: {plain:?}");
    assert_eq!(items[2] - items[1], 1, "blank between items 2-3: {plain:?}");
    // And no double-blank tail after the list.
    assert!(
        !plain.windows(2).any(|w| w[0].is_empty() && w[1].is_empty()),
        "consecutive blank lines: {plain:?}"
    );
}

#[test]
fn test_render_assistant_streaming_does_not_append_cursor_glyph() {
    let mut lines = Vec::new();
    render_assistant_message(&mut lines, None, "- item", 80);

    let plain = lines
        .into_iter()
        .map(|line| line.plain)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!plain.contains('▋'));
    assert!(plain.contains("• item"));
}

#[test]
fn test_build_transcript_shows_pending_status_without_visible_stream() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(plain.contains("Thinking"));
}

#[test]
fn test_streaming_reasoning_windows_when_answering() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.sending = true;
    // Once the answer starts, the finished thought shows the same bounded window
    // (most recent rows) above the reply — not the whole thing.
    app.pending_reasoning =
        "Inspecting the request\nSecond detail\nThird detail\nExtra detail\nFourth detail"
            .to_string();
    app.pending_response = "Working on it".to_string();

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(
        plain.contains("▸"),
        "windowed-with-more shows the ▸ expand chevron: {plain}"
    );
    assert!(
        plain.contains("Fourth detail"),
        "most recent line in the window"
    );
    assert!(
        !plain.contains("Inspecting the request"),
        "earliest line scrolls off the window"
    );
    assert!(plain.contains("Working on it"));
}

#[test]
fn test_streams_thinking_window_during_thinking_only_phase() {
    // Thinking-only gap: the reasoning streams live as a rolling window.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.sending = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_reasoning = "Working out the approach".to_string();
    assert!(app.pending_response.is_empty(), "no answer text yet");

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✻"), "streaming window marker: {plain}");
    assert!(
        plain.contains("Working out the approach"),
        "live thought is shown: {plain}"
    );
    assert!(
        !plain.contains("▸ thought"),
        "not the committed fold: {plain}"
    );
}

#[test]
fn test_thinking_window_shows_only_recent_lines() {
    // The live window keeps only the most recent THINKING_WINDOW_LINES lines;
    // older ones scroll off so it never grows unbounded.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.sending = true;
    app.pending_reasoning = "aaa\nbbb\nccc\nddd\neee".to_string();
    assert!(app.pending_response.is_empty());

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("bbb")
            && plain.contains("ccc")
            && plain.contains("ddd")
            && plain.contains("eee"),
        "recent lines visible: {plain}"
    );
    assert!(!plain.contains("aaa"), "older lines scroll off: {plain}");
}

#[test]
fn test_thinking_wrapped_lines_hang_indented_under_marker() {
    // A long thought that soft-wraps keeps its continuation rows indented under
    // the `✻` marker (pre-wrapped to the content width so the transcript wrapper
    // leaves the already-fitted rows alone).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 46;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "Here are the changes.".to_string(),
        reasoning_content: Some(
            "The user is asking which files changed — likely in the git working \
directory. I should check git status and recent git log."
                .to_string(),
        ),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let full = app.build_transcript();
    let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
    let think: Vec<&String> = wrapped
        .rows
        .iter()
        .filter(|r| r.contains("git") || r.starts_with("✻"))
        .collect();
    assert!(
        think.len() >= 2,
        "the thought wrapped to multiple rows: {think:?}"
    );
    assert!(think[0].starts_with("✻ "), "first row carries the marker");
    // Every continuation row is indented (two spaces), never flush left.
    for row in &think[1..] {
        assert!(
            row.starts_with("  ") && !row.starts_with("✻"),
            "wrapped row must hang-indent under the marker: {row:?}"
        );
    }
}

#[test]
fn test_build_transcript_hides_streaming_reasoning_when_disabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = false;
    app.sending = true;
    app.pending_reasoning = "Inspecting the request".to_string();
    app.pending_response = "Working on it".to_string();

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(!plain.contains("▸ thought"));
    assert!(!plain.contains("✻"));
    assert!(!plain.contains("Inspecting the request"));
    assert!(plain.contains("Working on it"));
}

#[tokio::test]
async fn effort_command_sets_level_enables_thinking_and_validates() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];
    app.thinking_enabled = false;
    app.reasoning_effort = None;

    // `/effort high` also turns thinking on — picking an effort implies you want the model to reason.
    app.run_effort_command(Some("HIGH".into())).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(app.thinking_enabled, "setting effort must turn thinking on");

    // An unknown level errors and leaves the choice unchanged.
    app.run_effort_command(Some("bogus".into())).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(matches!(app.notice, Some((ERROR, _))));

    // Bare `/effort` opens the picker of the model's levels.
    app.run_effort_command(None).await;
    assert!(
        matches!(&app.overlay, Overlay::Picker(p) if matches!(p.kind, PickerKind::Effort)),
        "bare /effort opens the effort picker"
    );
}

#[tokio::test]
async fn effort_command_noop_when_model_has_no_levels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_reasoning_efforts.clear();
    app.run_effort_command(None).await;
    assert!(
        matches!(app.overlay, Overlay::None),
        "no picker without levels"
    );
    assert!(matches!(app.notice, Some((MUTED, _))));
}

#[tokio::test]
async fn test_config_overlay_toggles_thinking() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    // First row is "Thinking"; its rendered state reads the live flag.
    assert_eq!(state.items[0].setting, ConfigSetting::Thinking);
    assert!(app.config_setting_enabled(ConfigSetting::Thinking));

    // Toggling it flips the live flag (the renderer derives the checkbox from it).
    app.toggle_config_setting(0).await;
    assert!(!app.thinking_enabled);
    assert!(!app.config_setting_enabled(ConfigSetting::Thinking));
}

#[tokio::test]
async fn test_config_overlay_toggles_agent_tools() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.agent_tools_enabled = true;

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::AgentTools)
        .expect("Agent tools row present");
    assert!(app.config_setting_enabled(ConfigSetting::AgentTools));

    app.toggle_config_setting(idx).await;
    assert!(!app.agent_tools_enabled);
    assert!(!app.config_setting_enabled(ConfigSetting::AgentTools));
}

#[test]
fn test_flush_pending_assistant_keeps_reasoning() {
    // Regression: the native agent path (flush_pending_assistant, used before each
    // tool step and at turn end) must carry pending_reasoning onto the committed
    // message, not drop it.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_response = "the answer".to_string();
    app.pending_reasoning = "the chain of thought".to_string();

    app.flush_pending_assistant();

    let last = app.history.last().expect("assistant message committed");
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "the answer");
    assert_eq!(
        last.reasoning_content.as_deref(),
        Some("the chain of thought")
    );
    assert!(app.pending_reasoning.is_empty());
}

#[test]
fn test_flush_pending_assistant_commits_reasoning_only_segment() {
    // A reasoning-only segment (model thought, then a tool call with no prose)
    // still commits, so the thinking isn't lost at the tool boundary.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_reasoning = "thinking before a tool".to_string();

    app.flush_pending_assistant();

    let last = app
        .history
        .last()
        .expect("reasoning-only message committed");
    assert_eq!(last.role, "assistant");
    assert!(last.content.is_empty());
    assert_eq!(
        last.reasoning_content.as_deref(),
        Some("thinking before a tool")
    );
}

#[test]
fn test_volatile_tail_fp_tracks_reasoning_and_toggle() {
    // Regression: the live "Thinking" block lives in the volatile tail, so its
    // fingerprint must change as reasoning streams and when thinking_enabled flips —
    // otherwise the cached tail never repaints during the reasoning-only gap.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.thinking_enabled = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let base = app.volatile_tail_fp();
    app.pending_reasoning.push_str("more thinking");
    let after_reasoning = app.volatile_tail_fp();
    assert_ne!(
        base, after_reasoning,
        "fp must change as reasoning streams (pending_response stays empty)"
    );

    app.thinking_enabled = false;
    let after_toggle = app.volatile_tail_fp();
    assert_ne!(
        after_reasoning, after_toggle,
        "fp must change when thinking_enabled flips so a /config toggle repaints"
    );
}

#[test]
fn test_thinking_block_has_distinct_bar_color() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some("the reasoning".to_string()),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);

    let t = app.build_transcript();
    let bar_for = |needle: &str| {
        t.plain_lines
            .iter()
            .position(|l| l.contains(needle))
            .and_then(|i| t.bar_colors[i])
    };
    // A committed turn shows its thinking in full, marked with `✻`, barless.
    let thinking_bar = bar_for("the reasoning");
    let answer_bar = bar_for("the answer");
    assert_eq!(
        thinking_bar, None,
        "thinking block is barless so it recedes as ephemeral meta"
    );
    assert_eq!(answer_bar, Some(ACCENT), "answer uses the accent bar");
    assert_ne!(thinking_bar, answer_bar);
}

#[test]
fn test_history_reasoning_windows_when_thinking_enabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_width = 80;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some(
            "the gist line\nsecond thought\nthird thought\nfourth thought\nthe private chain of thought"
                .to_string(),
        ),
        attachments: vec![],
    });

    // On: a bounded window shows the most recent lines behind the `▸` expand
    // chevron; earlier lines scroll off (expand to see them).
    app.thinking_enabled = true;
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let shown = app.build_transcript().plain_lines.join("\n");
    assert!(shown.contains("▸"), "windowed-with-more shows ▸: {shown}");
    assert!(
        shown.contains("the private chain of thought"),
        "most recent line is in the window"
    );
    assert!(
        !shown.contains("the gist line"),
        "the earliest line scrolls off the window"
    );
    assert!(shown.contains("the answer"));

    // Expanded (user clicked): the chevron flips to `▾` and every line shows.
    app.expanded_thinking.insert(0);
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let expanded = app.build_transcript().plain_lines.join("\n");
    assert!(expanded.contains("▾"), "expanded shows ▾: {expanded}");
    assert!(
        expanded.contains("the gist line") && expanded.contains("the private chain of thought"),
        "expanding reveals every line: {expanded}"
    );

    app.thinking_enabled = false;
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let hidden = app.build_transcript().plain_lines.join("\n");
    assert!(!hidden.contains("✻"));
    assert!(!hidden.contains("the private chain of thought"));
    assert!(hidden.contains("the answer"));
}

#[test]
fn test_click_thinking_header_toggles_inline_expansion() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    for (content, reasoning) in [
        ("first answer", "FIRST cot"),
        ("second answer", "SECOND cot"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning_content: Some(reasoning.to_string()),
            attachments: vec![],
        });
    }
    // Simulate the rendered rows: each assistant turn is an answer line preceded
    // by its full-thought `✻` header, in display (top→bottom) order.
    let hitbox = |rows: Vec<&str>| {
        Some(TranscriptHitbox {
            area: Rect::new(0, 0, 40, 10),
            first_row: 0,
            rows: rows.into_iter().map(str::to_string).collect(),
        })
    };
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "✻ SECOND cot",
        "second answer",
    ]);

    // Click the second thought's `✻` header → expand block 1.
    assert!(app.toggle_thinking_at_row(2));
    assert!(app.expanded_thinking.contains(&1));
    assert!(!app.expanded_thinking.contains(&0));

    // Expanded now leads with `▾`; clicking it collapses back to the window.
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "▾ SECOND cot",
        "second answer",
    ]);
    assert!(app.toggle_thinking_at_row(2));
    assert!(!app.expanded_thinking.contains(&1));

    // First header → first block (history index 0).
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "✻ SECOND cot",
        "second answer",
    ]);
    assert!(app.toggle_thinking_at_row(0));
    assert!(app.expanded_thinking.contains(&0));

    // Clicking a non-header (answer / indented content) row does nothing.
    assert!(!app.toggle_thinking_at_row(1));

    // With thinking off, no headers render, so clicks never resolve a block.
    app.thinking_enabled = false;
    assert!(!app.toggle_thinking_at_row(0));
}

#[test]
fn test_committed_thinking_windows_and_expands_on_click() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    // Short (≤ window): everything already fits, marked `✻`.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer".to_string(),
        reasoning_content: Some("line one\nline two".to_string()),
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✻"), "thinking marker present: {plain}");
    assert!(
        plain.contains("line one") && plain.contains("line two"),
        "short thought fits entirely in the window: {plain}"
    );

    // Long (> window): the window shows only the most recent rows; earliest scroll
    // off — until the user expands it.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer2".to_string(),
        reasoning_content: Some("alpha\nbeta\ngamma\ndelta\nepsilon".to_string()),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    // Short thought keeps `✻` (nothing hidden); the long one shows `▸` (more to
    // reveal) and is not yet expanded.
    assert!(plain.contains("✻"), "short thought keeps ✻: {plain}");
    assert!(
        plain.contains("▸"),
        "long windowed thought shows ▸: {plain}"
    );
    assert!(!plain.contains("▾"), "nothing expanded yet: {plain}");
    assert!(plain.contains("epsilon"), "most recent line shown: {plain}");
    assert!(
        !plain.contains("alpha"),
        "earliest line scrolls off: {plain}"
    );

    // Expand the long one (index 1): the marker flips to `▾` and every line shows.
    app.expanded_thinking.insert(1);
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("▾"),
        "expanded thought carries the ▾ marker: {plain}"
    );
    assert!(
        plain.contains("alpha") && plain.contains("delta"),
        "expanding reveals every line: {plain}"
    );
}

#[test]
fn test_finished_turn_renders_done_marker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_durations.insert(1, 404_000); // stamped on the last entry; 6m 44s
    app.transcript_revision = app.transcript_revision.wrapping_add(1);

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✶ Done in 6m 44s"), "{plain}");

    // No recorded duration → no marker.
    app.turn_durations.clear();
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("Done in"), "{plain}");
}

#[test]
fn test_commit_records_thinking_duration_and_resets_clock() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Simulate a segment whose thinking was frozen at 2s when the answer began.
    app.pending_reasoning = "the chain of thought".to_string();
    app.pending_response = "the answer".to_string();
    app.reasoning_elapsed_ms = Some(2000);

    app.flush_pending_assistant();

    let idx = app.history.len() - 1;
    assert_eq!(app.history[idx].role, "assistant");
    assert_eq!(app.reasoning_durations.get(&idx), Some(&2000));
    // The clock resets so the next segment times from its own first reasoning.
    assert!(app.reasoning_started_at.is_none());
    assert!(app.reasoning_elapsed_ms.is_none());
}

#[test]
fn test_composer_placeholder_stays_plain_when_history_has_reasoning() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer".to_string(),
        reasoning_content: Some("private reasoning".to_string()),
        attachments: vec![],
    });

    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);

    assert_eq!(plain, ">  Ask, plan, or build · / for commands");
}

#[test]
fn test_normalized_reasoning_lines_trims_and_removes_blank_runs() {
    assert_eq!(
        normalized_reasoning_lines("\nalpha\n\n\nbeta\n\n"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn test_footer_status_label_is_token_count_without_window() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_tokens = 5_120; // window unknown (0) → plain count

    // Idle and while sending: the footer is the token count — processing status
    // lives in the transcript, not the footer corner.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "~5.1k tokens");
    assert_eq!(color, MUTED);

    app.sending = true;
    assert_eq!(app.footer_status_label().0, "~5.1k tokens");
}

#[test]
fn test_footer_status_label_shows_window_on_pristine_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 1_000_000;
    app.context_tokens = 0; // no turn yet → no fill to gauge

    // The welcome screen shows the window size, not an empty `0/1M` meter.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "1M context");
    assert_eq!(color, MUTED);

    // The first tokens flip it to the live gauge.
    app.context_tokens = 50_000;
    app.context_is_estimate = false;
    assert_eq!(app.footer_status_label().0, "50k/1M");
}

#[test]
fn test_footer_status_label_shows_context_utilization() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;
    app.context_is_estimate = false; // a provider-measured fill

    // used/window, quiet until it nears the limit.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "10k/200k");
    assert_eq!(color, MUTED);

    // Warms toward the window limit (compaction territory).
    app.context_tokens = 170_000; // 85%
    assert_eq!(app.footer_status_label().1, WARNING);
    app.context_tokens = 195_000; // 97%
    assert_eq!(app.footer_status_label().1, ERROR);

    // A measured last-turn total wins over the chars/4 estimate.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k/200k");
}

#[test]
fn test_footer_status_label_marks_estimates() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;

    // cursor ACP / agents without reported usage: the chars/4 transcript figure
    // understates the model's real context, so flag it with `~`.
    app.context_is_estimate = true;
    app.last_usage = None;
    assert_eq!(app.footer_status_label().0, "~10k/200k");

    // A provider-measured last-turn total is exact even if the estimate flag
    // lingers from a prior turn — no tilde.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k/200k");
}

#[test]
fn test_footer_shows_plain_chat_badge_when_agent_tools_off() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn footer_text(app: &CodeTuiApp) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 1)).unwrap();
        terminal
            .draw(|frame| app.render_footer(frame, frame.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().to_string())
            .collect()
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // On (default): no badge.
    app.agent_tools_enabled = true;
    assert!(!footer_text(&app).contains("plain chat"));

    // Off: the badge marks plain-chat mode in the footer.
    app.agent_tools_enabled = false;
    assert!(footer_text(&app).contains("plain chat"));
}

#[test]
fn test_footer_status_label_updates_live_during_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 40_000; // prior turn's measured fill
    app.context_is_estimate = false;

    // Turn in flight, no measured usage yet: the footer holds the prior fill as a
    // baseline (no drop at turn start) and grows it as text streams in, flagged `~`.
    app.sending = true;
    app.pending_response = "x".repeat(4_000); // ~1k tokens streamed so far
    assert_eq!(app.footer_status_label().0, "~41k/200k");

    // Provider-measured usage arrives mid-stream (Anthropic message_start/_delta):
    // the live figure replaces the estimate immediately, no `~`.
    app.live_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_000,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52k/200k");

    // Turn ends: the fold into last_usage keeps the measured total on the footer.
    app.sending = false;
    app.live_usage = None;
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_500,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52.5k/200k");
}

#[test]
fn test_agent_context_drives_footer_live() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.sending = true;

    // Pre-usage request estimate (system prompt + tool schemas + conversation):
    // counts the real request the engine sends, not just the visible transcript.
    // Flagged `~` since it is a chars/4 estimate.
    app.apply_agent_context(50_000, false);
    assert_eq!(app.footer_status_label().0, "~50k/200k");

    // The step's measured total arrives → exact figure, no `~`, and streamed text
    // is not re-added on top (it is already in the measured completion).
    app.pending_response = "x".repeat(8_000); // would add ~2k if double-counted
    app.apply_agent_context(60_000, true);
    assert_eq!(app.footer_status_label().0, "60k/200k");
}

#[test]
fn test_current_action_label_reflects_phase() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // No tool in flight → pure model compute, only the Thinking heartbeat.
    assert_eq!(app.current_action_label(), None);
    // A tool call in flight → name the step, present-tense, with its target.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "ls"}),
        vec![],
        None,
    );
    assert_eq!(app.current_action_label().as_deref(), Some("running ls"));
    // Streamed tokens arriving → the step is over, heartbeat only.
    app.pending_response = "partial".to_string();
    assert_eq!(app.current_action_label(), None);
}

#[test]
fn test_tool_action_label_is_present_tense_with_target() {
    use super::render::tool_action_label;
    assert_eq!(
        tool_action_label("read_file", &serde_json::json!({"path": "src/main.rs"}), ""),
        "reading main.rs"
    );
    assert_eq!(
        tool_action_label("grep", &serde_json::json!({"pattern": "parse_expr"}), ""),
        "searching parse_expr"
    );
    assert_eq!(
        tool_action_label("edit_file", &serde_json::json!({"path": "a/b.rs"}), ""),
        "editing b.rs"
    );
    // MCP / unknown tools: named, no target to show.
    assert_eq!(
        tool_action_label("mcp__linear__create_issue", &serde_json::json!({}), ""),
        "running linear/create_issue"
    );
}

#[test]
fn test_tool_action_label_caps_long_command() {
    use super::render::tool_action_label;
    let long = "cd /private/tmp/test/markdown-preview && node server.js > /tmp/server.log & \
                sleep 2 && curl -s http://localhost:3000/ && kill $SERVER_PID";
    let label = tool_action_label("run_bash", &serde_json::json!({ "command": long }), "");
    // Capped to one line: short, ellipsized, single line.
    assert!(
        label.starts_with("running cd /private/tmp"),
        "label: {label:?}"
    );
    assert!(label.ends_with('…'), "ellipsized: {label:?}");
    assert!(label.chars().count() <= 50, "kept short: {label:?}");
    assert!(!label.contains('\n'), "single line: {label:?}");
}

#[test]
fn test_current_action_shows_inline_on_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "parse_expr"}),
        vec![],
        None,
    );
    // The action replaces "Thinking" on the SAME single status line (no extra
    // line), so the layout never shifts as steps come and go.
    let status = app
        .build_transcript()
        .plain_lines
        .into_iter()
        .find(|l| l.contains("searching parse_expr"))
        .expect("action shown inline on the status line");
    // It's the live status line (carries the elapsed clock), not a tool card.
    assert!(status.contains('('), "on the status line: {status:?}");
}

#[test]
fn test_subagent_activity_drives_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // Parent's `subagent` call in flight (no result) — the case that used to
    // freeze the status line on the parent's last tool for the whole delegation.
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"agent": "code-reviewer", "task": "review mcp.rs"}),
        vec![],
        None,
    );
    app.apply_subagent_activity(
        "code-reviewer".to_string(),
        "grep".to_string(),
        serde_json::json!({"pattern": "fn"}),
        7,
    );
    let status = app
        .build_transcript()
        .plain_lines
        .into_iter()
        .find(|l| l.contains("code-reviewer: searching"))
        .expect("nested sub-agent activity shown on the status line");
    assert!(
        status.contains("step 7"),
        "carries the child's step count: {status:?}"
    );
    assert!(
        status.contains('↳'),
        "marked as nested delegation: {status:?}"
    );
}

#[test]
fn test_parallel_subagent_rows_render_under_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // The batch's `subagent` calls are in flight (no results yet).
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"label": "audit auth flow", "task": "…"}),
        vec![],
        None,
    );
    app.apply_subagent_begin(vec!["audit auth flow".to_string(), String::new()]);
    // The unnamed delegate is numbered so its row stays distinguishable.
    assert_eq!(app.subagent_rows[1].name, "sub-agent 2");
    app.apply_subagent_slot(
        0,
        "audit auth flow".to_string(),
        "grep".to_string(),
        serde_json::json!({"pattern": "session"}),
        3,
    );
    app.apply_subagent_denied(1, "run_bash".to_string());
    app.apply_subagent_done(1, true, 8, 1200);
    // Headline counts completions; per-delegate rows sit under it.
    assert_eq!(app.desired_status(), "running 2 sub-agents (1 done)");
    let lines = app.build_transcript().plain_lines;
    let running = lines
        .iter()
        .find(|l| l.contains("audit auth flow — searching session"))
        .expect("live row shows the delegate's current action");
    assert!(running.contains("step 3"), "carries steps: {running:?}");
    let done = lines
        .iter()
        .find(|l| l.contains("✓ sub-agent 2 — done"))
        .expect("finished row flips to ✓ with stats");
    assert!(done.contains("1.2k tokens"), "carries tokens: {done:?}");
    // A denied gated call surfaces as a warning notice, not a silent deny.
    let (_, notice) = app.notice.clone().expect("denial sets a notice");
    assert!(
        notice.contains("run_bash") && notice.contains("auto-denied"),
        "explains the deny: {notice:?}"
    );
    // Slot events past the row list are ignored (defensive).
    app.apply_subagent_slot(9, String::new(), String::new(), serde_json::Value::Null, 1);
    // Batch end retires the rows and hands the headline back.
    app.subagent_rows.clear();
    assert_ne!(app.desired_status(), "running 2 sub-agents (1 done)");
}

#[test]
fn subagent_done_attributes_only_discovered_profiles() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let profile = |name: &str| crate::agent::subagents::Subagent {
        name: name.to_string(),
        description: String::new(),
        model: None,
        tools: None,
        body: String::new(),
        isolation_worktree: false,
        repo_local: false,
        source: std::path::PathBuf::new(),
    };
    app.last_subagents = vec![profile("code-reviewer")];
    app.apply_subagent_begin(vec!["code-reviewer".to_string(), "sub-agent 2".to_string()]);
    // A row named after a discovered profile is attributed to it.
    assert_eq!(
        app.apply_subagent_done(0, true, 6, 900),
        Some("code-reviewer".to_string())
    );
    // A generic/labeled delegate is not (it would pollute per-agent stats).
    assert_eq!(app.apply_subagent_done(1, true, 4, 200), None);
    // An out-of-range slot is a defensive no-op.
    assert_eq!(app.apply_subagent_done(9, true, 1, 1), None);
}

#[test]
fn test_done_marker_stays_above_new_input_after_plan_clear() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A finished turn: a reply, then a completed plan pinned in its panel. The
    // Done marker is stamped on the last VISIBLE entry (the reply, idx 1).
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first task".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the reply".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_durations.insert(1, 78_000);

    // The next user message clears the completed plan, then appends.
    app.clear_stale_plan();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "second task".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines;
    let done = plain.iter().position(|l| l.contains("Done in"));
    let next = plain.iter().position(|l| l.contains("second task"));
    assert!(done.is_some(), "Done marker still shown: {plain:?}");
    assert!(
        done < next,
        "Done marker must stay above the new input: {plain:?}"
    );
}

#[test]
fn test_clear_completed_plan_shifts_index_maps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "a".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "b".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Markers keyed to the entry AFTER the plan (idx 2) must slide to idx 1.
    app.turn_durations.insert(2, 5_000);
    app.expanded_thinking.insert(2);

    app.clear_stale_plan();

    assert_eq!(app.turn_durations.get(&2), None, "stale key dropped");
    assert_eq!(app.turn_durations.get(&1), Some(&5_000), "shifted down one");
    assert!(app.expanded_thinking.contains(&1), "set key shifted too");
}

#[test]
fn test_in_flight_tool_card_hidden_until_result() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "lsof -ti:3000"}),
        vec![],
        None,
    );
    // In flight: only the status names it — the `→ run_bash(…)` card is held back
    // so the same action isn't shown twice (the dup the user reported).
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("run_bash("),
        "card hidden while running: {plain:?}"
    );
    assert!(
        plain.contains("running lsof"),
        "status names the step: {plain:?}"
    );
    // Result lands → the card (with the command) renders.
    app.apply_agent_tool_result("ok".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("run_bash("),
        "card shown after result: {plain:?}"
    );
}

#[test]
fn test_parallel_bridged_batch_counts_and_lists_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for (id, task) in [("1", "Audit auth"), ("2", "Map engine"), ("3", "Scan CLI")] {
        app.apply_agent_tool_call(
            Some(id.to_string()),
            "subagent".to_string(),
            serde_json::json!({"label": task}),
            vec![],
            None,
        );
    }
    assert_eq!(app.desired_status(), "running 3 sub-agents (0 done)");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("↳ Audit auth"), "per-call rows: {plain:?}");
    assert!(plain.contains("↳ Scan CLI"), "per-call rows: {plain:?}");

    // Newest call resolving first: no "Thinking" flip, card held with the batch.
    app.apply_agent_tool_update("3".to_string(), None, Some("12 files".to_string()), false);
    assert_eq!(app.desired_status(), "running 3 sub-agents (1 done)");
    assert!(app.current_action_label().is_some());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("✓ Scan CLI — 12 files"),
        "done row: {plain:?}"
    );
    assert!(
        !plain.contains("Scan CLI ⎿") && !plain.contains("subagent("),
        "cards held until the batch settles: {plain:?}"
    );

    // All resolved → the batch is over: rows gone, cards render.
    app.apply_agent_tool_update("1".to_string(), None, Some("ok".to_string()), false);
    app.apply_agent_tool_update("2".to_string(), None, Some("ok".to_string()), true);
    assert_ne!(app.desired_status(), "running 3 sub-agents (3 done)");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("↳ "), "live rows cleared: {plain:?}");
    assert!(plain.contains("Audit auth"), "cards render: {plain:?}");
}

#[test]
fn test_cursor_task_notice_renames_generic_batch_rows() {
    // A `cursor/task` notice arrives as an args-only AgentToolUpdate.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for id in ["1", "2"] {
        app.apply_agent_tool_call(
            Some(id.to_string()),
            "subagent".to_string(),
            serde_json::json!({"label": "Subagent task"}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_update(
        "2".to_string(),
        Some(serde_json::json!({"label": "Audit the auth flow", "agent": "explore"})),
        None,
        false,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("↳ explore — Audit the auth flow"),
        "row renamed by the notice: {plain:?}"
    );
    assert!(
        plain.contains("↳ Subagent task"),
        "sibling untouched: {plain:?}"
    );
    // Enrichment also refreshes the one-line status label.
    assert_eq!(
        app.current_action_label().as_deref(),
        Some("delegating: Audit the auth flow")
    );
}

#[test]
fn test_parallel_bridged_batch_mixed_tools_noun() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for (id, name, args) in [
        ("1", "grep", serde_json::json!({"pattern": "hover"})),
        ("2", "subagent", serde_json::json!({"label": "Audit"})),
    ] {
        app.apply_agent_tool_call(Some(id.to_string()), name.to_string(), args, vec![], None);
    }
    assert_eq!(app.desired_status(), "running 2 parallel steps (0 done)");
}

#[test]
fn test_status_tail_shows_turn_output_tokens() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.pending_response = "x".repeat(4_000); // ~1k tokens streamed
    // A live ~-flagged estimate, alongside the always-present interrupt hint —
    // stopping a runaway turn must be discoverable mid-turn.
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("tokens"), "estimate shown: {plain:?}");
    assert!(
        plain.contains("esc to interrupt"),
        "esc hint in the agent-turn status: {plain:?}"
    );
    // The engine reports the turn's cumulative generated tokens → exact (no ~),
    // distinct from the prompt-dominated context total (which stays in the footer).
    app.turn_output_tokens = 512;
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("512 tokens"), "turn output shown: {plain:?}");
    assert!(!plain.contains("~512"), "measured, not estimate: {plain:?}");
}

#[test]
fn test_status_tail_counts_queued_input() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages.push("follow-up one".to_string());
    app.queued_messages.push("follow-up two".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("2 queued"), "queued chip missing: {plain:?}");
}

#[test]
fn test_desired_status_names_decision_waits_and_stalls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // A permission card means blocked on the user — the status must not read as
    // the (possibly destructive) command already executing.
    let (reply, _rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });
    assert_eq!(app.desired_status(), "waiting for your approval");
    app.agent_permission = None;

    // A silent stream reads as a stall, not an ever-ticking "Thinking".
    app.last_stream_activity =
        std::time::Instant::now().checked_sub(std::time::Duration::from_secs(15));
    assert_eq!(app.desired_status(), "waiting");
    app.last_stream_activity = Some(std::time::Instant::now());
    assert_eq!(app.desired_status(), "Thinking");
}

#[test]
fn test_decision_wait_freezes_step_and_turn_clocks() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    let t0 = std::time::Instant::now();
    app.last_tool_action = Some(("running rm -rf build".to_string(), t0, None));
    app.request_started_at = Some(t0);
    let (reply, _rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: None,
        reply,
    });
    // Two ticks a real interval apart: the waiting span must be pushed out of
    // both clocks, so their effective elapsed stays near zero.
    app.tick_decision_wait();
    std::thread::sleep(std::time::Duration::from_millis(30));
    app.tick_decision_wait();
    let (_, since, _) = app.last_tool_action.as_ref().unwrap();
    assert!(
        since.elapsed() < std::time::Duration::from_millis(15),
        "tool clock ran during the wait: {:?}",
        since.elapsed()
    );
    assert!(
        app.request_started_at.unwrap().elapsed() < std::time::Duration::from_millis(15),
        "turn clock ran during the wait"
    );
    // Card resolved → clocks run again from here.
    app.agent_permission = None;
    app.tick_decision_wait();
    assert!(app.wait_tick.is_none());
}

#[test]
fn test_discard_queued_input_counts_and_clears() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages.push("a".to_string());
    app.queued_messages.push("b".to_string());
    app.steering_queue.lock().unwrap().push("steer".to_string());
    assert_eq!(app.discard_queued_input(), 3);
    assert!(app.queued_messages.is_empty());
    assert!(app.steering_queue.lock().unwrap().is_empty());
    assert_eq!(app.discard_queued_input(), 0);
}

#[test]
fn test_bash_status_clock_shows_timeout_budget() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cargo test", "timeout": 300}),
        vec![],
        None,
    );
    let (_, _, budget) = app.last_tool_action.as_ref().unwrap();
    assert_eq!(*budget, Some(300));
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains(" / "),
        "deadline missing from clock: {plain:?}"
    );
    // Non-bash steps carry no budget.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "x.rs"}),
        vec![],
        None,
    );
    assert_eq!(app.last_tool_action.as_ref().unwrap().2, None);
}

#[test]
fn test_done_marker_appends_turn_note() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let idx = app.history.len() - 1;
    app.turn_durations.insert(idx, 42_000);
    app.turn_notes.insert(idx, "3.1k tokens".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✶ Done in 42s · 3.1k tokens"), "{plain}");
}

#[tokio::test]
async fn test_agent_error_notice_uses_error_color() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // An engine error (notify_error) lands in the error hue, not the neutral one.
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: 429".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, Some((ERROR, "LLM error: 429".to_string())));
    // An ordinary notice stays neutral.
    app.tx
        .send(RuntimeEvent::AgentNotice("compacting context…".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, Some((MUTED, "compacting context…".to_string())));
}

#[tokio::test]
async fn test_agent_error_persists_in_transcript() {
    // Beyond the transient notice, the error must land as a durable transcript entry.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentError(
            "the provider rejected this API key (upstream 401: bad key)".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    let last = app.history.last().expect("error entry committed");
    assert_eq!(last.role, "error");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("✗ the provider rejected this API key"),
        "{plain}"
    );
    // Never seeded back to the model on an engine rebuild.
    assert!(
        super::runtime_impl::agent_seed_turns(&app.history).is_empty(),
        "error entries must stay display-only"
    );
}

#[tokio::test]
async fn test_done_marker_skipped_on_errored_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.request_started_at =
        std::time::Instant::now().checked_sub(std::time::Duration::from_secs(3));
    // The turn errors, then finishes → no `Done in …` marker (would misrepresent).
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: 429".to_string()))
        .unwrap();
    app.tx
        .send(RuntimeEvent::AgentFinished {
            steps: 1,
            tokens: 0,
            context_tokens: 0,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(
        app.turn_durations.is_empty(),
        "no Done marker on error: {:?}",
        app.turn_durations
    );
}

#[tokio::test]
async fn test_sandbox_escalation_notice_clears_on_next_output() {
    use crate::agent::engine::SANDBOX_ESCALATION_NOTICE;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            SANDBOX_ESCALATION_NOTICE.to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.notice,
        Some((MUTED, SANDBOX_ESCALATION_NOTICE.to_string()))
    );
    app.tx
        .send(RuntimeEvent::AgentToolCall {
            id: None,
            name: "read_file".to_string(),
            args: serde_json::json!({"path": "x"}),
            line_starts: vec![],
            old_content: None,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "next tool clears the ack");

    // Streamed prose clears it too.
    app.tx
        .send(RuntimeEvent::AgentNotice(
            SANDBOX_ESCALATION_NOTICE.to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "done".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "new prose clears the ack");

    // An unrelated notice in the same slot is left alone.
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "Queued — sends later".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::AgentToolCall {
            id: None,
            name: "read_file".to_string(),
            args: serde_json::json!({"path": "y"}),
            line_starts: vec![],
            old_content: None,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.notice,
        Some((MUTED, "Queued — sends later".to_string())),
        "unrelated notice survives"
    );
}

#[tokio::test]
async fn test_retry_notice_clears_on_recovery_not_stuck() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "connection issue — retrying (2/3)…".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.retrying, "retry flag set");
    assert!(app.notice.is_some());

    // Output resumes → recovered → notice + flag clear, not stuck.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "ok".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "retry notice cleared on recovery");
    assert!(!app.retrying);

    // A real error notice must NOT be cleared by later output.
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: boom".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "x".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.notice.is_some(), "unrelated error notice survives");
}

#[test]
fn test_connection_retry_status_reads_working() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    assert_eq!(app.desired_status(), "Thinking");
    app.retrying = true;
    assert_eq!(app.desired_status(), "Working");
}

#[tokio::test]
async fn test_retry_notice_sets_retrying_then_progress_clears_it() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "connection issue — retrying (2/3)…".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.retrying, "retry notice sets the flag");
    // A streamed chunk = recovery → clears it.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "hi".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(!app.retrying, "progress clears the flag");
}

#[test]
fn test_status_label_throttled_to_min_duration() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    // First tick adopts the current label.
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("Thinking")
    );

    // A tool starts, but "Thinking" was just shown → it must hold its second.
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "foo"}),
        vec![],
        None,
    );
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("Thinking"),
        "must hold the prior label for its minimum second"
    );

    // Backdate the display past the minimum → the next tick may switch.
    let old = std::time::Instant::now()
        .checked_sub(STATUS_MIN_DURATION + std::time::Duration::from_millis(50))
        .expect("instant in range");
    app.status_display = Some(("Thinking".to_string(), old));
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("searching foo"),
        "switches once the prior label has had its second"
    );
}

#[test]
fn test_intro_column_stable_from_empty_to_message() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn aivo_col(app: &mut CodeTuiApp) -> u16 {
        let mut terminal = Terminal::new(TestBackend::new(48, 16)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        let area = buf.area;
        // The half-block wordmark's top-left glyphs are "▄▀█" — find that run.
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width.saturating_sub(2) {
                if buf.cell((x, y)).unwrap().symbol() == "▄"
                    && buf.cell((x + 1, y)).unwrap().symbol() == "▀"
                    && buf.cell((x + 2, y)).unwrap().symbol() == "█"
                {
                    return x;
                }
            }
        }
        panic!("AIVO banner not found");
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let empty_col = aivo_col(&mut app);

    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let message_col = aivo_col(&mut app);

    assert_eq!(
        empty_col, message_col,
        "AIVO Chat banner shifts columns between the empty and message states"
    );
}

#[test]
fn test_transcript_intro_is_brand_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "claude-sonnet-4".to_string();
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/tmp/project".to_string();

    // Wide column: full "aivo code" mark, version trailing the baseline.
    let version = format!("  v{}", crate::version::VERSION);
    assert_eq!(
        app.transcript_intro_lines(80),
        vec![
            "▄▀█ █ █░█ █▀█  █▀▀ █▀█ █▀▄ █▀█".to_string(),
            format!("█▀█ █ ▀▄▀ █▄█  █▄▄ █▄█ █▄▀ █▄▄{version}"),
        ]
    );
}

#[test]
fn test_transcript_intro_narrow_falls_back_to_aivo() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);

    // Too slim for the full mark and the version: bare "aivo", no wrap.
    assert_eq!(
        app.transcript_intro_lines(20),
        vec!["▄▀█ █ █░█ █▀█".to_string(), "█▀█ █ ▀▄▀ █▄█".to_string()]
    );
    // Exactly the mark width keeps the full mark (version needs more room).
    assert_eq!(
        app.transcript_intro_lines(30)[0],
        "▄▀█ █ █░█ █▀█  █▀▀ █▀█ █▀▄ █▀█"
    );
    let version = format!("v{}", crate::version::VERSION);
    assert!(
        app.transcript_intro_lines(80)[1].ends_with(&version),
        "version should trail the wide banner"
    );
    assert!(
        !app.transcript_intro_lines(20)
            .iter()
            .any(|l| l.contains(&version)),
        "version should be dropped when the column is too narrow"
    );
}

#[test]
fn test_welcome_shows_capability_chip_and_tip() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![
        skill_command("repo-study", "Study a repo"),
        skill_command("release", "Cut a release"),
    ];
    app.mcp_configured_count = 1;
    app.welcome_tip_index = 0; // "start a line with ! to run a shell command"

    let (screen, _) = render_full_screen(&mut app, 90, 24);
    assert!(
        screen.contains("2 skills · 1 MCP"),
        "missing chip:\n{screen}"
    );
    assert!(
        screen.contains("✶ Tip"),
        "missing tip label + glyph:\n{screen}"
    );
    assert!(
        screen.contains(WELCOME_TIPS[0]),
        "missing tip text:\n{screen}"
    );
    // The bare "Ready" filler is gone.
    assert!(!screen.contains("Ready"), "stale Ready line:\n{screen}");
}

#[test]
fn test_welcome_tip_rotates_on_interval_only_while_visible() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx); // empty transcript, no overlay
    app.welcome_tip_index = 0;

    // First frame just starts the clock — no swap yet.
    assert!(!app.tick_welcome_tip(), "swapped on the first frame");
    assert_eq!(app.welcome_tip_index, 0);
    assert!(app.welcome_tip_rotated_at.is_some(), "clock not started");

    // Before the interval elapses, still no swap.
    assert!(!app.tick_welcome_tip());
    assert_eq!(app.welcome_tip_index, 0);

    // Backdate past the interval → the next tick advances and re-clocks.
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(app.tick_welcome_tip(), "no swap after the interval");
    assert_eq!(app.welcome_tip_index, 1);

    // An overlay pauses rotation and resets the clock.
    app.overlay = Overlay::Help { scroll: 0 };
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(!app.tick_welcome_tip(), "must not rotate under an overlay");
    assert_eq!(app.welcome_tip_index, 1, "index frozen while covered");
    assert!(
        app.welcome_tip_rotated_at.is_none(),
        "clock reset while covered"
    );

    // A draft pauses it too; clearing it restarts the clock.
    app.overlay = Overlay::None;
    app.draft = "hello".to_string();
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(!app.tick_welcome_tip(), "must not rotate while composing");
    assert_eq!(app.welcome_tip_index, 1, "index frozen while composing");
    assert!(
        app.welcome_tip_rotated_at.is_none(),
        "clock reset while composing"
    );
    app.draft.clear();
    assert!(
        !app.tick_welcome_tip(),
        "cleared draft restarts the clock, no swap"
    );
    assert!(
        app.welcome_tip_rotated_at.is_some(),
        "clock restarted after clear"
    );
}

#[test]
fn test_welcome_chip_omits_zero_and_pluralizes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Nothing configured → no chip, but the tip still shows.
    assert_eq!(app.welcome_capabilities_label(), None);

    // One skill, no MCP → singular, MCP segment omitted.
    app.skill_commands = vec![skill_command("repo-study", "Study a repo")];
    assert_eq!(app.welcome_capabilities_label().as_deref(), Some("1 skill"));

    // MCP only → skills segment omitted.
    app.skill_commands.clear();
    app.mcp_configured_count = 3;
    assert_eq!(app.welcome_capabilities_label().as_deref(), Some("3 MCP"));
}

#[test]
fn test_session_picker_item_line_fits_mixed_width_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "deepseek".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hi".to_string(),
        preview_text: "hi · Hi there! ✨ 想聊点什么？还是需要我帮忙呢？ 我随时待命～ 😊🌟"
            .to_string(),
    };

    let line = session_picker_item_lines(&preview, true, false, 64)
        .into_iter()
        .next()
        .unwrap();
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 64);
}

#[test]
fn test_key_picker_item_line_fits_modal_width() {
    let key = ApiKey::new_with_protocol(
        "deepseek".to_string(),
        "deepseek".to_string(),
        "https://api.cloudflare.com/client/v4/accounts/long/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let line = key_picker_item_line(&key, true, 36);
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 36);
    assert!(plain.contains("deepseek"));
}

#[test]
fn test_key_search_text_uses_host_not_full_path() {
    let key = ApiKey::new_with_protocol(
        "testgw".to_string(),
        "testgw".to_string(),
        "https://api.ai.example-gateway.net/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let search = key_search_text(&key);
    assert!(search.contains("testgw"));
    assert!(search.contains("api.ai.example-gateway.net"));
    assert!(!search.contains("/endpoint"));
}

#[test]
fn test_key_filter_does_not_match_across_full_url_path() {
    let unrelated = "groq groq api.groq.com";
    let target = "testgw testgw api.ai.example-gateway.net";

    assert!(matches_fuzzy("gapn", target));
    assert!(!matches_fuzzy("gapn", unrelated));
}

#[test]
fn test_notice_display_prefixes_errors_only() {
    let error = (ERROR, "boom".to_string());
    let info = (MUTED, "ok".to_string());

    let displayed = notice_display(Some(&error)).unwrap();
    assert_eq!(displayed.0, ERROR);
    assert_eq!(displayed.1.as_ref(), "Error: boom");

    let displayed = notice_display(Some(&info)).unwrap();
    assert_eq!(displayed.0, MUTED);
    assert_eq!(displayed.1.as_ref(), "ok");

    assert!(notice_display(None).is_none());
}

#[test]
fn test_picker_visible_items_track_selection_for_single_line_rows() {
    let picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..6)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 4,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    let visible = picker.visible_items(3);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 2);
    assert_eq!(visible[2].0, 4);
}

#[test]
fn test_picker_navigation_wraps() {
    let mut picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..3)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 0,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    picker.select_prev();
    assert_eq!(picker.selected, 2);

    picker.select_next();
    assert_eq!(picker.selected, 0);
}

#[test]
fn test_picker_visible_items_respect_single_line_session_rows() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };
    let picker = PickerState {
        title: "Resume",
        query: String::new(),
        items: vec![
            PickerEntry {
                label: "one".to_string(),
                search_text: "one".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "two".to_string(),
                search_text: "two".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "three".to_string(),
                search_text: "three".to_string(),
                value: PickerValue::Session(preview),
            },
        ],
        loading: false,
        selected: 2,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    let visible = picker.visible_items(4);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 0);
    assert_eq!(visible[2].0, 2);
}

#[test]
fn test_session_picker_header_targets_newest_session() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 8, 48);
    let first = plain_text_from_spans(&lines[0].spans);

    assert_eq!(row_map.first().copied(), Some(Some(0)));
    assert!(!first.contains("Newest chat"));
    assert_eq!(row_map.get(1).copied(), Some(Some(0)));
}

#[test]
fn test_grouped_session_picker_short_view_shows_selected_session_row() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 1, 48);
    let only = plain_text_from_spans(&lines[0].spans);

    assert!(only.contains("Newest chat"));
    assert_eq!(row_map, vec![Some(0)]);
}

#[test]
fn test_rect_contains() {
    let area = Rect::new(10, 4, 8, 3);
    assert!(rect_contains(area, (10, 4)));
    assert!(rect_contains(area, (17, 6)));
    assert!(!rect_contains(area, (18, 6)));
    assert!(!rect_contains(area, (17, 7)));
}

#[test]
fn test_render_main_keeps_composer_near_short_empty_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut composer_area = Rect::default();

    terminal
        .draw(|frame| {
            composer_area = app.render_main(frame, frame.area());
        })
        .unwrap();

    assert!(composer_area.y + composer_area.height < 11);
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.y, 0);
    // 80 cols minus the 2-col accent gutter (no scrollbar column reserved).
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.width, 78);
    assert_eq!(app.transcript_width, 78);
}

#[test]
fn test_markdown_table_renders_aligned_columns() {
    let md = "| Name | ID |\n|------|----|\n| aivo | uz9 |\n| openrouter | 598 |\n";
    let plain: Vec<String> = render_markdown_lines(md, 80)
        .iter()
        .map(|l| l.plain.clone())
        .collect();
    let joined = plain.join("\n");

    // The table is a closed box: top/header-rule/bottom borders all present.
    assert!(
        joined.contains('┌') && joined.contains('┐'),
        "no top border:\n{joined}"
    );
    assert!(joined.contains('┼'), "no header rule:\n{joined}");
    assert!(
        joined.contains('└') && joined.contains('┘'),
        "no bottom border:\n{joined}"
    );

    // Cells are separated by a column divider — NOT concatenated into a blob.
    assert!(joined.contains('│'), "no column divider:\n{joined}");
    assert!(
        !joined.contains("aivouz9") && !joined.contains("NameID"),
        "cells got concatenated:\n{joined}"
    );
    // It fits the width, so nothing is truncated.
    assert!(joined.contains("openrouter") && joined.contains("598"));

    // Every box line is exactly the same display width (a rectangular box) and
    // its `│`/junction columns line up across rows.
    let box_lines: Vec<&String> = plain
        .iter()
        .filter(|l| l.chars().any(|c| "│┌┐└┘├┤┬┴┼".contains(c)))
        .collect();
    assert!(
        box_lines.len() >= 5,
        "expected 4 borders + ≥1 body row:\n{joined}"
    );
    let width = |s: &str| {
        s.chars()
            .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
            .sum::<usize>()
    };
    let w0 = width(box_lines[0]);
    assert!(
        box_lines.iter().all(|l| width(l) == w0),
        "box lines are not the same width: {:?}",
        box_lines.iter().map(|l| width(l)).collect::<Vec<_>>()
    );
    assert!(w0 <= 80, "box overflows the 80-col width: {w0}");
    let bar_cols = |s: &str| -> Vec<usize> {
        let mut col = 0usize;
        let mut cols = Vec::new();
        for c in s.chars() {
            if "│┌┐└┘├┤┬┴┼".contains(c) {
                cols.push(col);
            }
            col += UnicodeWidthChar::width(c).unwrap_or(0);
        }
        cols
    };
    let cols0 = bar_cols(box_lines[0]);
    assert!(
        box_lines.iter().all(|l| bar_cols(l) == cols0),
        "column separators not aligned across rows:\n{joined}"
    );
}

#[test]
fn test_markdown_table_is_responsive_to_width() {
    // A table whose natural width exceeds the pane must shrink to fit (wrapping
    // long cells) rather than overflow — so the transcript wrapper never shears it.
    let md = "| Tool | Description |\n|------|-------------|\n\
        | aivo | A unified CLI for several AI coding assistants with secure key storage |\n";
    for width in [30u16, 50, 72] {
        let lines = render_markdown_lines(md, width);
        let widths: Vec<usize> = lines
            .iter()
            .map(|l| {
                l.plain
                    .chars()
                    .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
                    .sum::<usize>()
            })
            .collect();
        assert!(
            widths.iter().all(|&w| w <= usize::from(width)),
            "table overflows width {width}: {widths:?}"
        );
        // The long description survives — just wrapped across multiple lines.
        let joined: String = lines
            .iter()
            .map(|l| l.plain.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("unified"),
            "content dropped at width {width}"
        );
    }
}

#[test]
fn test_wrap_styled_line_word_wraps_and_hard_breaks() {
    use ratatui::text::Line;

    // Word wrap: every row fits the width and no word is split across rows.
    let line = Line::from("the quick brown fox jumps");
    let rows = wrap_styled_line(&line.spans, 10);
    for r in &rows {
        assert!(
            r.plain.chars().count() <= 10,
            "row exceeds width: {:?}",
            r.plain
        );
    }
    let words: Vec<String> = rows
        .iter()
        .flat_map(|r| r.plain.split_whitespace().map(String::from))
        .collect();
    assert_eq!(words, ["the", "quick", "brown", "fox", "jumps"]);
    assert!(rows.len() >= 3);

    // A word longer than the width hard-breaks, losing no characters.
    let long = Line::from("abcdefghij");
    let rows = wrap_styled_line(&long.spans, 4);
    let joined: String = rows.iter().map(|r| r.plain.as_str()).collect();
    assert_eq!(joined, "abcdefghij");
    for r in &rows {
        assert!(r.plain.chars().count() <= 4);
    }

    // Leading whitespace is a hanging indent: continuation rows re-apply it so
    // wrapped text (e.g. expanded reasoning) stays aligned under the first line.
    let indented = Line::from("  alpha beta gamma");
    let rows = wrap_styled_line(&indented.spans, 9);
    let plains: Vec<&str> = rows.iter().map(|r| r.plain.as_str()).collect();
    assert!(rows.len() >= 2, "should wrap: {plains:?}");
    for r in &rows {
        assert!(r.plain.starts_with("  "), "row lost indent: {:?}", r.plain);
        assert!(
            r.plain.chars().count() <= 9,
            "row exceeds width: {:?}",
            r.plain
        );
    }
}

#[test]
fn test_long_reply_scrolls_to_show_last_line() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Many words → word-wraps to far more rows than the view; ends in a sentinel.
    let mut body = (0..80)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    body.push_str(" TAILWORD");
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: body,
        reasoning_content: None,
        attachments: vec![],
    });
    assert!(app.follow_output);

    let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..10u16 {
        for x in 0..40u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // Following output must scroll to the bottom; the last word is visible. (A
    // char-wrap row count under-counts and would clip it — the original bug.)
    assert!(
        screen.contains("TAILWORD"),
        "last line clipped — did not scroll to bottom:\n{screen}"
    );
}

#[test]
fn test_transcript_cache_reuses_across_frames_until_content_changes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let draw = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
    };

    draw(&mut app, &mut terminal);
    let fp_after_first = app.transcript_cache.as_ref().unwrap().fp;
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();

    // A second frame with identical content must NOT rebuild the body: same
    // fingerprint and the same backing allocation (no re-parse / re-wrap).
    draw(&mut app, &mut terminal);
    assert_eq!(app.transcript_cache.as_ref().unwrap().fp, fp_after_first);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "identical content must reuse the cached body without rebuilding"
    );

    // Appending a message changes the fingerprint → the cache rebuilds.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "another turn".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    draw(&mut app, &mut terminal);
    assert_ne!(
        app.transcript_cache.as_ref().unwrap().fp,
        fp_after_first,
        "new content must invalidate the cache fingerprint"
    );
}

// Render the whole screen (transcript + composer + any card/overlay) to a plain
// string plus the per-row strings, for layout assertions.
#[cfg(test)]
fn render_full_screen(app: &mut CodeTuiApp, w: u16, h: u16) -> (String, Vec<String>) {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut rows = Vec::new();
    for y in 0..h {
        let mut row = String::new();
        for x in 0..w {
            row.push_str(buf[(x, y)].symbol());
        }
        rows.push(row);
    }
    (rows.join("\n"), rows)
}

#[test]
fn test_permission_card_anchored_above_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Some history so the composer is pushed toward the bottom of the screen.
    for _ in 0..4 {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: "working on it".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build/".to_string()),
        reply,
    });

    let (screen, rows) = render_full_screen(&mut app, 60, 20);

    // The friendly heading + command + color-coded keys all render.
    assert!(
        screen.contains("Run a command?"),
        "heading missing:\n{screen}"
    );
    assert!(
        screen.contains("rm -rf build/"),
        "command missing:\n{screen}"
    );
    assert!(
        screen.contains("allow once") && screen.contains("always") && screen.contains("deny"),
        "keys missing:\n{screen}"
    );
    // A destructive command is flagged.
    assert!(
        screen.contains("⚠ looks destructive"),
        "destructive flag missing:\n{screen}"
    );

    // The card hugs the composer: its bottom border sits just above the input's
    // full-width divider rule, which sits directly above the prompt (not floating
    // mid-screen like a centered modal).
    let bottom_border_row = rows
        .iter()
        .position(|r| r.contains('└'))
        .expect("card bottom border");
    let composer_row = rows
        .iter()
        .position(|r| r.trim_start().starts_with('>'))
        .expect("composer prompt row");
    assert_eq!(
        bottom_border_row + 2,
        composer_row,
        "card bottom must sit one row (the divider) above the composer:\n{screen}"
    );
    // The divider directly under the card is the full-width composer rule — now
    // always carrying the mode badge (here "normal", since a card only shows
    // when auto-approve is off) — so the narrower card never leaves it poking
    // out past the card's right edge.
    let divider = &rows[bottom_border_row + 1];
    assert!(
        divider.contains('─') && divider.contains("normal"),
        "full-width composer rule (with the mode badge) must sit under the card:\n{screen}"
    );

    // The card is sized to its content, not stretched across the full 60-col
    // input row — its border must be clearly narrower than the screen.
    let border_w = rows[bottom_border_row].trim_end().chars().count();
    assert!(
        (10..50).contains(&border_w),
        "card should be a compact width, got {border_w}:\n{screen}"
    );
}

#[test]
fn test_permission_card_keys_always_visible_in_short_terminal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    // A long multi-line preview that cannot fully fit above the composer.
    let preview = (0..30)
        .map(|i| format!("line {i} of a very long command"))
        .collect::<Vec<_>>()
        .join("\n");
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some(preview),
        reply,
    });

    // Short screen: the preview must be trimmed so the keys row still shows.
    let (screen, _rows) = render_full_screen(&mut app, 50, 10);
    assert!(
        screen.contains("allow once") && screen.contains("deny"),
        "keys must survive a cramped card:\n{screen}"
    );
}

#[test]
fn test_spinner_animation_does_not_invalidate_transcript_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..60u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    };

    let screen = render_screen(&mut app, &mut terminal);
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();
    assert!(screen.contains("Thinking"), "spinner missing:\n{screen}");

    // Advancing the spinner glyph must not rebuild the body — only the appended
    // status line is volatile, so a long transcript never reparses to animate.
    app.frame_tick = app.frame_tick.wrapping_add(7);
    let screen = render_screen(&mut app, &mut terminal);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "spinner animation must reuse the cached body"
    );
    assert!(screen.contains("Thinking"), "spinner missing:\n{screen}");
}

#[test]
fn test_streaming_tokens_do_not_invalidate_history_body_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Stream".to_string();

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..60u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    };

    let screen = render_screen(&mut app, &mut terminal);
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();
    assert!(
        screen.contains("Stream"),
        "streamed text missing:\n{screen}"
    );

    // Appending streamed tokens must NOT rebuild the cached history body: the live
    // reply lives in the volatile tail (composed fresh each frame), so a long
    // conversation never re-parses/re-wraps its whole history per token.
    app.pending_response.push_str(" more text");
    let screen = render_screen(&mut app, &mut terminal);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "streamed tokens must reuse the cached history body (no full re-render)"
    );
    assert!(
        screen.contains("more text"),
        "new streamed text missing:\n{screen}"
    );
}

/// The split render (cached history body + volatile tail + spinner) must
/// reproduce the single-pass `build_transcript` row-for-row at the same width —
/// scroll math goes through `max_scroll` → `build_transcript`, so any divergence
/// would desync what's painted from where it scrolls. Locks the invariant the
/// streaming-perf split (keeping the live reply out of the cached body) relies on.
#[test]
fn test_streaming_composed_render_matches_full_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "explain the plan in detail please".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content:
            "# Heading\n\nA committed reply with **markdown** and a fairly long line that wraps."
                .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Mid-stream: a partial reply (the volatile tail) + a notice, plus the spinner.
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.pending_response =
        "Streaming this answer now, with another long line that should wrap across the pane."
            .to_string();
    app.notice = Some((MUTED, "compacting context…".to_string()));

    // Render through the split path; the hitbox carries the full composed row set.
    let mut terminal = Terminal::new(TestBackend::new(40, 24)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let composed: Vec<String> = app.transcript_hitbox.as_ref().unwrap().rows.clone();

    // Reference: the single-pass transcript wrapped to the same width.
    let full = app.build_transcript();
    let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
    assert_eq!(
        composed, wrapped.rows,
        "split render diverged from the single-pass build_transcript"
    );
}

/// The volatile-tail cache must invalidate when the streamed reply grows, a
/// notice changes, or the reply clears at turn end. Render across each state and
/// confirm the composed render still matches the single-pass `build_transcript`,
/// so no stale render survives a content change at the same width.
#[test]
fn streaming_reply_cache_invalidates_on_change() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "go".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    let long = "Short reply that has now grown a good deal longer and wraps across the pane.";
    let states: [(&str, Option<&str>); 4] = [
        ("Short", None),
        (long, None),                // grew: cache must re-render
        (long, Some("compacting…")), // notice added at same reply len
        ("", None),                  // reply cleared at turn end: tail collapses
    ];

    let mut terminal = Terminal::new(TestBackend::new(40, 24)).unwrap();
    for (reply, notice) in states {
        app.pending_response = reply.to_string();
        app.notice = notice.map(|t| (MUTED, t.to_string()));
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let composed: Vec<String> = app.transcript_hitbox.as_ref().unwrap().rows.clone();
        let full = app.build_transcript();
        let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
        assert_eq!(
            composed, wrapped.rows,
            "cached render diverged from build_transcript at reply {reply:?} notice {notice:?}"
        );
    }
}

/// Clicking a folded `!cmd` block's `▸ +N more lines` expander reveals the full
/// output inline (the in-process successor to the ctrl+o pager); clicking the
/// `▾ collapse` toggle folds it back to the preview. Drives the real render +
/// hitbox + click-mapping path end to end.
#[test]
fn output_block_expands_inline_on_click() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 250 lines: past the 40-line fold preview, so the block renders an expander.
    let full: String = (1..=250).map(|i| format!("L{i:04}\n")).collect();
    app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    let idx = app.history.len() - 1;

    // The transcript rows the click handler resolves against — built exactly as the
    // real render does (wrap the body, then seed the hitbox).
    let refresh = |app: &mut CodeTuiApp| -> Vec<String> {
        let body = app.build_transcript_history_body(80);
        let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
        app.transcript_hitbox = Some(TranscriptHitbox {
            area: Rect::new(0, 0, 80, 40),
            first_row: 0,
            rows: wrapped.rows.clone(),
        });
        wrapped.rows
    };

    // Folded: the preview stops at the cap, the tail (L0250) is hidden behind a
    // clickable expander.
    let rows = refresh(&mut app);
    assert!(rows.iter().any(|r| r.contains("+210 more lines")));
    assert!(rows.iter().all(|r| !r.contains("L0250")));
    let marker_row = rows
        .iter()
        .position(|r| is_output_expander(r))
        .expect("folded block renders an expander row");

    // Clicking the expander row toggles this block into `expanded_output`.
    assert!(app.toggle_output_at_row(marker_row));
    assert!(app.expanded_output.contains(&idx));

    // Expanded: the full tail is shown in place, over a `▾ collapse` toggle.
    let rows = refresh(&mut app);
    assert!(
        rows.iter().any(|r| r.contains("L0250")),
        "expanded block reveals the elided tail:\n{}",
        rows.join("\n")
    );
    let collapse_row = rows
        .iter()
        .position(|r| r.trim_start().starts_with(OUTPUT_EXPANDED_PREFIX))
        .expect("expanded block renders a collapse toggle");

    // Clicking the collapse toggle folds it back.
    assert!(app.toggle_output_at_row(collapse_row));
    assert!(!app.expanded_output.contains(&idx));
    let rows = refresh(&mut app);
    assert!(rows.iter().all(|r| !r.contains("L0250")));
}

#[test]
fn test_wrap_transcript_carries_bar_color_per_row() {
    use ratatui::text::Line;
    let lines = vec![StyledLine {
        line: Line::from("alpha beta gamma delta"),
        plain: "alpha beta gamma delta".to_string(),
    }];
    let wrapped = wrap_transcript(&lines, &[Some(TOOL)], 8);
    assert!(wrapped.rows.len() >= 3);
    // Every wrapped row inherits the source line's bar color.
    assert!(wrapped.bars.iter().all(|b| *b == Some(TOOL)));
    assert_eq!(wrapped.rows.len(), wrapped.bars.len());
}

#[test]
fn test_wrap_transcript_fills_background_to_full_width() {
    use ratatui::text::Line;
    // A line that ends in a background-colored span (an inline-diff row) is padded
    // so the tint fills the whole row width and a wrap reads as one block.
    let diff = vec![StyledLine {
        line: Line::from(vec![
            Span::styled("  ".to_string(), Style::default()),
            Span::styled(" + ".to_string(), Style::default().bg(DIFF_ADD_BG)),
            Span::styled(
                "let very long added line".to_string(),
                Style::default().bg(DIFF_ADD_BG),
            ),
        ]),
        plain: "   + let very long added line".to_string(),
    }];
    let wrapped = wrap_transcript(&diff, &[None], 12);
    assert!(wrapped.rows.len() >= 2, "long diff line should wrap");
    for row in &wrapped.rows {
        assert_eq!(
            row_display_width(row),
            12,
            "every wrapped diff row should be padded to full width: {row:?}"
        );
    }
    // Each visual row still ends in the add tint, so the block stays contiguous.
    for line in &wrapped.text.lines {
        assert_eq!(
            line.spans.last().and_then(|s| s.style.bg),
            Some(DIFF_ADD_BG)
        );
    }

    // A plain line (no trailing background) is never padded — only opted-in tinted
    // rows gain a background.
    let plain = vec![StyledLine {
        line: Line::from(Span::styled("hi".to_string(), Style::default().fg(TEXT))),
        plain: "hi".to_string(),
    }];
    let wrapped = wrap_transcript(&plain, &[None], 12);
    assert_eq!(wrapped.rows[0], "hi");
}

#[test]
fn test_edit_diff_rows_carry_add_remove_tints() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let lines = app.build_transcript().lines;
    let bg_of = |needle: &str| -> Option<Color> {
        lines
            .iter()
            .find(|l| l.plain.contains(needle))
            .and_then(|l| l.line.spans.iter().find_map(|s| s.style.bg))
    };
    assert_eq!(
        bg_of("- let x = 1;"),
        Some(DIFF_DEL_BG),
        "removed line tint"
    );
    assert_eq!(bg_of("+ let x = 2;"), Some(DIFF_ADD_BG), "added line tint");
}

#[test]
fn test_edit_diff_word_highlight_brightens_only_changed_tokens() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let lines = app.build_transcript().lines;
    // Background of the span carrying exactly `text`, on the line containing `needle`.
    let span_bg = |needle: &str, text: &str| -> Option<Color> {
        lines
            .iter()
            .find(|l| l.plain.contains(needle))
            .and_then(|l| l.line.spans.iter().find(|s| s.content.as_ref() == text))
            .and_then(|s| s.style.bg)
    };
    // Only the differing token jumps to the brighter emphasis tint; the shared
    // run of the line keeps the base tint (the intra-line word diff).
    assert_eq!(
        span_bg("- let x = 1;", "1"),
        Some(DIFF_DEL_HL_BG),
        "changed token emphasised on the removed line"
    );
    assert_eq!(
        span_bg("- let x = 1;", "let x = "),
        Some(DIFF_DEL_BG),
        "common run stays at the base tint"
    );
    assert_eq!(
        span_bg("+ let x = 2;", "2"),
        Some(DIFF_ADD_HL_BG),
        "changed token emphasised on the added line"
    );
    assert_eq!(
        span_bg("+ let x = 2;", "let x = "),
        Some(DIFF_ADD_BG),
        "common run stays at the base tint"
    );
}

#[test]
fn test_edit_diff_numbers_rows_from_line_starts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // `old_string` begins at file line 10 (a=10, b=11, c=12); only the middle
    // line changes. `line_starts` is the pre-edit probe the runtime stamps.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"a.rs","old_string":"a\nb\nc","new_string":"a\nB\nc"},"line_starts":[10]}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("11 - b"),
        "removed line numbered by its old-file offset:\n{plain}"
    );
    assert!(
        plain.contains("11 + B"),
        "added line numbered by its new-file offset:\n{plain}"
    );
    assert!(
        plain.lines().any(|l| l.contains("10") && l.contains('a')),
        "leading context carries its file number:\n{plain}"
    );
    assert!(
        plain.lines().any(|l| l.contains("12") && l.contains('c')),
        "trailing context carries its file number:\n{plain}"
    );
}

#[test]
fn test_edit_diff_shows_context_only_marks_changed_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Only the middle line changes; the first and last are shared context.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"fn a() {\n    let y = 2;\n}","new_string":"fn a() {\n    let y = 20;\n}"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let lines = app.build_transcript().lines;
    let find = |needle: &str| lines.iter().find(|l| l.plain.contains(needle));
    let bg_of =
        |needle: &str| find(needle).and_then(|l| l.line.spans.iter().find_map(|s| s.style.bg));

    // The shared lines render as context: present, but with no +/- and no tint.
    let ctx = find("fn a() {").expect("context line missing");
    assert!(
        !ctx.plain.contains("- ") && !ctx.plain.contains("+ "),
        "context line should carry no diff sign: {:?}",
        ctx.plain
    );
    assert_eq!(bg_of("fn a() {"), None, "context line must not be tinted");
    assert!(find("}").is_some(), "trailing context line missing");

    // Only the genuinely changed line is flagged on each side.
    assert_eq!(bg_of("- "), Some(DIFF_DEL_BG), "old line removed + tinted");
    assert_eq!(bg_of("+ "), Some(DIFF_ADD_BG), "new line added + tinted");
    assert!(find("- ").unwrap().plain.contains("let y = 2;"));
    assert!(find("+ ").unwrap().plain.contains("let y = 20;"));
    // The unchanged lines are NOT duplicated as removed/added.
    assert!(
        !lines
            .iter()
            .any(|l| l.plain.contains("- fn a()") || l.plain.contains("+ fn a()")),
        "shared line should not appear as a change"
    );
}

#[test]
fn test_edit_diff_trims_context_and_collapses_gap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Two far-apart changes (first and last line) with a long unchanged middle.
    // Context is limited to a few lines, so the middle collapses to a `⋯`.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"x.rs","old_string":"A\nc1\nc2\nc3\nMID\nc5\nc6\nc7\nB","new_string":"A2\nc1\nc2\nc3\nMID\nc5\nc6\nc7\nB2"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    // Both edited lines are flagged; the deep-interior context collapses.
    assert!(
        plain.contains("- A") && plain.contains("+ A2"),
        "first change:\n{plain}"
    );
    assert!(
        plain.contains("- B") && plain.contains("+ B2"),
        "last change:\n{plain}"
    );
    assert!(
        plain.contains('⋯'),
        "collapsed gap marker missing:\n{plain}"
    );
    assert!(
        !plain.contains("MID"),
        "context beyond the window should be dropped:\n{plain}"
    );
    // Context adjacent to a change is still shown.
    assert!(
        plain.contains("c1") && plain.contains("c7"),
        "near context kept:\n{plain}"
    );
}

#[test]
fn test_footer_is_single_status_row() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // Render the app, returning just the footer row — a single status line now
    // (there is no hint bar), pinned to the terminal's bottom row.
    fn footer_text(configure: impl Fn(&mut CodeTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        // A long transcript so the footer sits on the terminal's bottom rows.
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: (0..40)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            reasoning_content: None,
            attachments: vec![],
        });
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        // The footer is a single row (the status line) at the terminal's bottom.
        let mut text = String::new();
        for x in 0..80u16 {
            text.push_str(buf[(x, 11)].symbol());
        }
        text
    }

    // The whole rendered screen as one string, for state shown outside the bottom
    // hint-bar row (e.g. the composer-rule mode badge).
    fn full_screen(configure: impl Fn(&mut CodeTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..80u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    }

    // The footer is a single row: the status line (context meter) in every state.
    let idle = footer_text(|_| {});
    assert!(idle.contains("tokens"), "idle status line: {idle:?}");
    // No hint bar — no contextual key hints ever leak into the footer.
    assert!(!idle.contains("send"), "no idle key hints: {idle:?}");
    assert!(
        !footer_text(|a| a.sending = true).contains("interrupt"),
        "no sending hint bar — footer stays status-only"
    );
    // The mode badge rides the composer rule, not the footer, and every mode is
    // shown so the current state + its cycle key stay discoverable.
    // Matched by each badge's unique glyph: a wide glyph splits across buffer
    // cells (breaking adjacent-text matches) and the rotating welcome tip can
    // contain the bare mode words.
    assert!(
        full_screen(|a| a.agent_auto_approve = true).contains('⚡'),
        "auto badge on composer rule"
    );
    assert!(
        full_screen(|a| a.agent_auto_approve = false).contains("normal (shift+tab)"),
        "normal mode shown on composer rule (discoverable)"
    );
    assert!(
        full_screen(|a| a.agent_review_edits = true).contains('✎'),
        "review badge on composer rule"
    );
    assert!(
        !footer_text(|a| a.agent_auto_approve = true).contains('⚡'),
        "the mode badge is not in the footer"
    );
    // Effort tier (bare value) shows on the status line only when thinking is on.
    assert!(
        footer_text(|a| {
            a.thinking_enabled = true;
            a.model_supports_thinking = true;
            a.model_reasoning_efforts = vec!["high".to_string()];
        })
        .contains("high"),
        "effort tier shown when thinking on"
    );
    assert!(
        !footer_text(|a| {
            a.thinking_enabled = false;
            a.model_supports_thinking = true;
            a.model_reasoning_efforts = vec!["high".to_string()];
        })
        .contains("high"),
        "effort tier hidden when thinking off"
    );
}

#[test]
fn test_inline_status_stays_in_transcript_across_phases() {
    // The processing status renders in the transcript in every phase, so its
    // position never jumps to the footer when the reply starts streaming.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    // Model-compute phase: the Thinking heartbeat shows, no streamed text yet —
    // and no round tokens generated, so no "0 tokens" noise.
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("Thinking"), "compute phase: {plain:?}");
    assert!(!plain.contains("tokens"), "no token tail at 0: {plain:?}");

    // Streaming the reply reads "Working"; once output flows the tail shows it.
    app.pending_response = "streaming the answer".to_string();
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("streaming the answer"));
    assert!(plain.contains("Working"), "streaming phase: {plain:?}");
    assert!(plain.contains("tokens"));
}

#[test]
fn test_display_cwd_shows_real_dir_for_agent_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // make_test_app uses a plain API key → agent-capable → show the real dir.
    assert!(app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
}

#[test]
fn test_display_cwd_shows_real_dir_for_cursor_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key = ApiKey::new_with_protocol(
        "cursor".to_string(),
        String::new(),
        "cursor".to_string(),
        None,
        String::new(),
    );
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // cursor-agent now runs as a real agent in the launch dir (not the in-process
    // engine, so still not `agent_capable`), so the footer shows the real dir.
    assert!(app.key.is_cursor_acp());
    assert!(!app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
}

#[tokio::test]
async fn test_cursor_model_refresh_sets_window_and_effort_badge() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key = ApiKey::new_with_protocol(
        "cursor".to_string(),
        String::new(),
        "cursor".to_string(),
        None,
        String::new(),
    );

    // Claude tier → underlying-model window + tier badge.
    app.model = "claude-opus-4-8-max".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 1_000_000);
    assert_eq!(app.cursor_effort_label.as_deref(), Some("max"));

    // Cursor-native windows (not in models.dev): composer 200k, auto 2M.
    app.model = "composer-2.5".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 200_000);
    assert_eq!(app.cursor_effort_label, None);

    app.model = "auto".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 2_000_000);
    assert_eq!(app.cursor_effort_label, None);
}

#[test]
fn test_agent_events_build_display_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Streamed prose arrives, then the engine calls a tool: the prose is
    // committed as an assistant entry *before* the tool-call entry.
    app.pending_response = "I'll read it.".to_string();
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    assert!(app.pending_response.is_empty());
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[0].role, "assistant");
    assert_eq!(app.history[0].content, "I'll read it.");
    assert_eq!(app.history[1].role, "tool_call");
    assert!(app.history[1].content.contains("read_file"));
    assert!(app.history[1].content.contains("a.rs"));

    app.apply_agent_tool_result("128 lines".to_string());
    assert_eq!(app.history[2].role, "tool_result");
    assert_eq!(app.history[2].content, "128 lines");

    // A second tool call with no intervening prose doesn't inject a blank entry.
    app.apply_agent_tool_call(
        None,
        "edit_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    assert_eq!(app.history.len(), 4);
    assert_eq!(app.history[3].role, "tool_call");
}

#[test]
fn test_folded_run_bash_result_keeps_streaming_tail_height() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Short output stays whole under the summary — nothing vanishes.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "make"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("line one\nline two\nline three".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("line one"), "{plain}");
    assert!(plain.contains("line three"), "{plain}");

    // Long output keeps only the last `STREAM_TAIL_LINES` — no collapse, no flood.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "curl"}),
        vec![],
        None,
    );
    let long = (1..=40)
        .map(|n| format!("row {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.apply_agent_tool_result(long);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("row 40"), "last line kept: {plain}");
    assert!(plain.contains("row 37"), "tail window kept: {plain}");
    assert!(
        !plain.contains("row 36"),
        "earlier lines fold away: {plain}"
    );
    assert!(
        !plain.contains("row 20"),
        "earlier lines fold away: {plain}"
    );

    // A non-run_bash tool has no tail, so it stays a summary line only.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("secret-alpha\nsecret-beta".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("secret-alpha"),
        "a non-run_bash result stays folded: {plain}"
    );
}

#[test]
fn test_native_tool_paths_render_relative_to_cwd() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/Users/alice/proj".to_string();

    // Paths under cwd render relative (the absolute prefix is footer noise).
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/alice/proj/src/ui/views/panel.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("128 lines".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("read_file(src/ui/views/panel.rs)"),
        "{plain}"
    );
    assert!(
        !plain.contains("/Users/alice/proj"),
        "absolute path leaked: {plain}"
    );

    // Over-long paths left-truncate on a segment boundary to keep the basename.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/alice/proj/src/module/feature/component/section/detail/view/inner/widget.rs"}),
    vec![], None);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("…/"), "expected left-truncation: {plain}");
    assert!(plain.contains("widget.rs"), "basename lost: {plain}");
}

#[test]
fn test_tool_result_count_units_by_tool() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Each tool reports its multi-line count in the right unit, not bare "lines".
    let cases = [
        ("glob", "a.rs\nb.rs\nc.rs", "3 files"),
        ("list_dir", "src/\ntests/", "2 entries"),
        ("grep", "1: x\n2: y\n3: z\n4: w", "4 matches"),
        ("read_file", "line one\nline two", "2 lines"),
    ];
    for (tool, output, expected) in cases {
        app.apply_agent_tool_call(
            None,
            tool.to_string(),
            serde_json::json!({"path": "x"}),
            vec![],
            None,
        );
        app.apply_agent_tool_result(output.to_string());
        let plain = app.build_transcript().plain_lines.join("\n");
        assert!(
            plain.contains(expected),
            "{tool}: expected {expected:?} in {plain}"
        );
    }
}

#[test]
fn test_batched_tool_results_resolve_unit_and_target_by_position() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Batch of [grep×3] then [r0, r1, r2]: each result must resolve its own call
    // by position, not idx-1 (which left the 2nd/3rd mislabelled "lines").
    for pat in ["alpha", "beta", "gamma"] {
        app.apply_agent_tool_call(
            None,
            "grep".to_string(),
            serde_json::json!({"pattern": pat}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_result("1: x\n2: y".to_string()); // 2 matches
    app.apply_agent_tool_result("1: x\n2: y\n3: z".to_string()); // 3 matches
    app.apply_agent_tool_result("1: x\n2: y\n3: z\n4: w".to_string()); // 4 matches

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("2 matches"), "{plain}");
    assert!(plain.contains("3 matches"), "{plain}");
    assert!(plain.contains("4 matches"), "{plain}");
    assert!(plain.contains("alpha · "), "{plain}");
    assert!(plain.contains("beta · "), "{plain}");
    assert!(plain.contains("gamma · "), "{plain}");
    assert!(plain.contains("searched ×3"), "{plain}");
    assert!(!plain.contains("searched ×3: alpha"), "{plain}");
}

#[test]
fn test_adjacent_search_tools_merge_into_one_group() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 2 globs + 2 greps both read as "searched" → one `searched ×4` header, not
    // two indistinguishable `searched ×2`, with results keeping their own units.
    for (tool, pat) in [
        ("glob", "**/*canary*"),
        ("glob", "**/*gemini*"),
        ("grep", "gemini"),
        ("grep", "canary"),
    ] {
        app.apply_agent_tool_call(
            None,
            tool.to_string(),
            serde_json::json!({"pattern": pat}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_result("a.rs\nb.rs\nc.rs".to_string()); // glob -> 3 files
    app.apply_agent_tool_result("x.rs".to_string());
    app.apply_agent_tool_result("1:a\n2:b".to_string()); // grep -> 2 matches
    app.apply_agent_tool_result("1:a\n2:b\n3:c\n4:d".to_string()); // grep -> 4 matches

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("searched ×4"), "{plain}");
    assert_eq!(plain.matches("searched ×").count(), 1, "{plain}");
    assert!(plain.contains("3 files"), "{plain}");
    assert!(plain.contains("2 matches"), "{plain}");
    assert!(plain.contains("4 matches"), "{plain}");
    assert!(plain.contains("**/*canary* · "), "{plain}");
    assert!(plain.contains("gemini · "), "{plain}");
}

#[test]
fn test_run_bash_label_strips_redundant_cd_prefix() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/Users/alice/project/work/aivo".to_string();
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cd /Users/alice/project/work/aivo && git show f391ffd"}),
        vec![],
        None,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("run_bash(git show f391ffd)"), "{plain}");
    assert!(
        !plain.contains("cd /Users/alice/project/work/aivo"),
        "{plain}"
    );
}

#[test]
fn test_detached_results_in_batch_carry_their_target() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // read_file + grep in one batch: both results land after both calls, so each
    // must name its own target to stay distinguishable.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "src/gemini_router.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "400|sanitize"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("a\nb\nc".to_string()); // read -> 3 lines
    app.apply_agent_tool_result("1:x\n2:y".to_string()); // grep -> 2 matches

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("▸ +3 lines · gemini_router.rs"), "{plain}");
    assert!(plain.contains("▸ +2 matches · 400|sanitize"), "{plain}");
}

#[test]
fn test_failed_tool_result_renders_in_error_hue() {
    // An in-process failure arrives as a single-line `error: …`; the result reads
    // in the error hue, not dim, so a timeout/denial is legible as a failure.
    let mut lines = Vec::new();
    render_tool_result(
        &mut lines,
        "error: command timed out after 120s",
        "",
        Some("run_bash"),
        None,
        false,
    );
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(ERROR))
    );
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .all(|s| s.style.fg != Some(FAINT))
    );

    // A normal multi-line result stays neutral even when a line says "error:".
    let mut ok = Vec::new();
    render_tool_result(
        &mut ok,
        "error: x\nmore\nlines",
        "",
        Some("grep"),
        None,
        false,
    );
    assert!(ok[0].line.spans.iter().any(|s| s.style.fg == Some(FAINT)));
    assert!(ok[0].line.spans.iter().all(|s| s.style.fg != Some(ERROR)));
}

#[test]
fn test_failed_bash_result_shows_exit_code_in_error_hue() {
    // Nonzero exit arrives as Ok(output + "[exit N]") — the summary must still read red.
    let mut lines = Vec::new();
    render_tool_result(
        &mut lines,
        "compiling…\nerror[E0308]: mismatched types\n[exit 101]",
        "",
        Some("run_bash"),
        None,
        false,
    );
    assert!(lines[0].plain.contains("exited 101"), "{}", lines[0].plain);
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(ERROR)),
        "failed bash summary should carry the error hue"
    );

    // The `[exit N]` tail is found past the spill pointer, and a clean run
    // stays neutral.
    assert_eq!(
        bash_exit_code("x\ny\n[exit 2]\n[full output: /tmp/f]"),
        Some(2)
    );
    assert_eq!(bash_exit_code("all good\n[done]"), None);
    let mut ok = Vec::new();
    render_tool_result(&mut ok, "a\nb\nc", "", Some("run_bash"), None, false);
    assert!(
        ok[0].line.spans.iter().all(|s| s.style.fg != Some(ERROR)),
        "clean bash output must not read as a failure"
    );
}

#[test]
fn test_tool_result_expands_inline_via_keyboard_toggle() {
    // Folded run_bash keeps only the streamed tail; Ctrl+O (toggle_latest_output)
    // reveals the full output in place and folds back.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cargo test"}),
        vec![],
        None,
    );
    // Eight lines: folded keeps the tail, so the first lines only appear expanded.
    app.apply_agent_tool_result(
        "test 1 ... ok\ntest 2 ... ok\ntest 3 ... ok\ntest a ... ok\n\
         test b ... FAILED\ntest c ... ok\ntest d ... ok\n[exit 1]"
            .to_string(),
    );

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("▸ +8 lines"), "{plain}");
    assert!(plain.contains("exited 1"), "{plain}");
    // The streamed tail stays put, but earlier lines fold away.
    assert!(plain.contains("[exit 1]"), "tail line kept: {plain}");
    assert!(
        !plain.contains("test 1 ... ok"),
        "folded result hides its early lines: {plain}"
    );

    assert!(app.toggle_latest_output());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("test 1 ... ok"),
        "expanded result must show all its lines: {plain}"
    );
    // The summary stays the (sole) toggle for a short block — no trailing collapse.
    assert!(plain.contains("▾ 8 lines"), "{plain}");
    assert!(!plain.contains(OUTPUT_EXPANDED_PREFIX), "{plain}");

    assert!(app.toggle_latest_output());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("test 1 ... ok"), "{plain}");
}

#[test]
fn expanded_tool_result_refolds_from_its_summary_row() {
    // Regression: expanding flipped the summary to a dead "▾ N lines" row, so a
    // long read_file could only fold from the far end of the block.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "big.rs"}),
        vec![],
        None,
    );
    let content: String = (1..=100).map(|i| format!("{i}: line\n")).collect();
    app.apply_agent_tool_result(content.trim_end().to_string());
    let idx = app.history.len() - 1;

    let refresh = |app: &mut CodeTuiApp| -> Vec<String> {
        let body = app.build_transcript_history_body(80);
        let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
        app.transcript_hitbox = Some(TranscriptHitbox {
            area: Rect::new(0, 0, 80, 40),
            first_row: 0,
            rows: wrapped.rows.clone(),
        });
        wrapped.rows
    };

    let rows = refresh(&mut app);
    let marker = rows.iter().position(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(marker));
    assert!(app.expanded_output.contains(&idx));

    // A long expanded block renders two toggles, both mapping to this entry.
    let rows = refresh(&mut app);
    let markers: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| is_output_expander(r))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(markers.len(), 2, "rows:\n{}", rows.join("\n"));
    assert!(
        rows[markers[0]].contains("▾ 100 lines"),
        "{}",
        rows[markers[0]]
    );
    assert!(rows[markers[1]].contains(OUTPUT_EXPANDED_PREFIX));

    assert!(app.toggle_output_at_row(markers[0]));
    assert!(!app.expanded_output.contains(&idx));

    refresh(&mut app);
    let rows = refresh(&mut app);
    let marker = rows.iter().position(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(marker));
    let rows = refresh(&mut app);
    let bottom = rows.iter().rposition(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(bottom));
    assert!(!app.expanded_output.contains(&idx));
}

#[test]
fn tool_result_expander_click_maps_across_mixed_blocks() {
    // A `!cmd` fold and a tool-result fold interleave; clicking the second
    // marker row must toggle the tool result, not the shell block.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let full: String = (1..=250).map(|i| format!("L{i:04}\n")).collect();
    app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "x.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("alpha\nbeta".to_string());
    let result_idx = app.history.len() - 1;

    let body = app.build_transcript_history_body(80);
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 80, 40),
        first_row: 0,
        rows: wrapped.rows.clone(),
    });
    let marker_rows: Vec<usize> = wrapped
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| is_output_expander(r))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(marker_rows.len(), 2, "rows:\n{}", wrapped.rows.join("\n"));

    assert!(app.toggle_output_at_row(marker_rows[1]));
    assert!(app.expanded_output.contains(&result_idx));
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("beta"), "{plain}");
}

#[test]
fn write_file_snapshot_rides_tool_call_entry() {
    // The tool_start snapshot turns the write_file card into a real diff.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "write_file".to_string(),
        serde_json::json!({"path": "a.rs", "content": "fn keep() {}\nfn renamed() {}\n"}),
        vec![Some(1)],
        Some("fn keep() {}\nfn original() {}\n".to_string()),
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("original"), "old side missing: {plain}");
    assert!(plain.contains("renamed"), "new side missing: {plain}");
    assert!(
        plain.contains(" - ") && plain.contains(" + "),
        "expected real del/ins rows, not all-additions: {plain}"
    );
}

#[test]
fn test_run_bash_label_drops_redirection_noise() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "which aivo 2>/dev/null && aivo --help 2>&1"}),
        vec![],
        None,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("which aivo && aivo --help"), "{plain}");
    assert!(!plain.contains("2>/dev/null"), "{plain}");
    assert!(!plain.contains("2>&1"), "{plain}");
}

#[test]
fn test_punctuation_only_reasoning_renders_no_thought_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some("...".to_string()),
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("▸ thought"), "{plain}");
    assert!(plain.contains("the answer"));
}

#[test]
fn test_subagents_render_individually_not_coalesced() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Two subagents dispatched in parallel arrive as adjacent tool_calls. Unlike
    // the many tiny cursor steps coalescing is meant to fold, each subagent is a
    // distinct unit of work and must keep its task visible — never an opaque
    // `subagent ×2`.
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Review the chat API endpoint for gaps"}),
        vec![],
        None,
    );
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"agent": "reviewer", "task": "Audit the auth flow"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("## Findings\nfirst\nsecond".to_string());

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("subagent ×"),
        "subagents must not coalesce: {plain}"
    );
    // The word "subagent" is jargon and must not appear — each delegated task is
    // shown directly after the arrow, with no `subagent(...)` wrapper.
    assert!(
        !plain.contains("subagent"),
        "the word 'subagent' must not be shown: {plain}"
    );
    assert!(
        plain.contains("→ Review the chat API endpoint for gaps"),
        "first delegated task missing: {plain}"
    );
    // A named delegation leads with the profile name so the row attributes it.
    assert!(
        plain.contains("→ reviewer — Audit the auth flow"),
        "second delegated task missing: {plain}"
    );
    // The result previews the report's first line after the fold toggle — not a
    // bare count alone that says nothing about what the subagent found.
    assert!(
        plain.contains("▸ +3 lines · ## Findings"),
        "subagent result preview missing: {plain}"
    );
}

#[test]
fn test_delegating_spinner_label_drops_subagent_word() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Audit the auth flow"}),
        vec![],
        None,
    );
    let activity = app.current_action_label().unwrap_or_default();
    assert_eq!(
        activity, "delegating",
        "action line should read 'delegating'"
    );
    assert!(
        !activity.contains("subagent"),
        "action line leaked 'subagent'"
    );
}

#[test]
fn test_cursor_tool_update_enriches_call_in_place() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Cursor's start event lacks the real target → the label falls back to the
    // generic title.
    app.apply_agent_tool_call(
        Some("c1".to_string()),
        "read_file".to_string(),
        serde_json::json!({"path": "Read File"}),
        vec![],
        None,
    );
    let rev_before = app.transcript_revision;
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("read_file(Read File)"), "{plain}");

    // The follow-up update resolves the path and carries a compact result.
    app.apply_agent_tool_update(
        "c1".to_string(),
        Some(serde_json::json!({"path": "src/chat.rs"})),
        Some("42 lines".to_string()),
        false,
    );
    // The cache fingerprint must move so the edited (middle) entry re-renders.
    assert!(app.transcript_revision > rev_before);

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("read_file(src/chat.rs)"), "{plain}");
    assert!(plain.contains("⎿ 42 lines"), "{plain}");
    assert!(
        !plain.contains("Read File"),
        "stale title remained: {plain}"
    );

    // A failed update surfaces the error in place.
    app.apply_agent_tool_update(
        "c1".to_string(),
        None,
        Some("permission denied".into()),
        true,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("⎿ permission denied"), "{plain}");
}

#[tokio::test]
async fn test_cursor_turn_ending_on_tool_does_not_duplicate_prose() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    // Cursor streams prose, runs a tool, and ends with no trailing text — the
    // prose is flushed as an assistant entry before the tool steps.
    app.pending_response = "Reading and fixing.".to_string();
    app.apply_agent_tool_call(
        None,
        "edit_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("done".to_string());
    assert!(app.pending_response.is_empty());

    // The turn finishes. `turn.content` carries the full accumulated prose
    // (cursor reports no usage) — re-pushing it would duplicate the reply.
    app.tx
        .send(RuntimeEvent::Finished {
            result: Ok(ChatTurnResult {
                content: "Reading and fixing.".to_string(),
                usage: None,
                model: None,
                raw_body: None,
            }),
            format: ChatFormat::OpenAI,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    let roles: Vec<&str> = app.history.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "tool_call", "tool_result"]);
    assert_eq!(app.history[1].content, "Reading and fixing.");
}

#[test]
fn test_typewriter_reveals_buffer_progressively() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 34 chars: the floor (30) dominates the 1/2 catch-up (17), revealing 30 on
    // the first frame.
    let full = "字".repeat(34);
    app.incoming_buffer = full.clone();
    assert!(app.tick_typewriter());
    assert_eq!(app.pending_response, "字".repeat(30)); // exactly 30 chars, no split glyph
    assert_eq!(app.incoming_buffer.chars().count(), 4);

    // Drains fully over the next frames, never losing or splitting a char.
    let mut guard = 0;
    while app.tick_typewriter() {
        guard += 1;
        assert!(guard < 100, "typewriter should converge");
    }
    assert_eq!(app.pending_response, full);
    assert!(app.incoming_buffer.is_empty());
}

#[tokio::test]
async fn test_finish_deferred_until_typewriter_drains() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // A chunk arrived but hasn't been revealed yet.
    app.incoming_buffer = "the full reply".to_string();

    // Finished lands while text is still buffered → it must be held back.
    app.tx
        .send(RuntimeEvent::Finished {
            result: Ok(ChatTurnResult {
                content: "the full reply".to_string(),
                usage: None,
                model: None,
                raw_body: None,
            }),
            format: ChatFormat::OpenAI,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.pending_finish.is_some(), "finish should be deferred");
    assert!(!app.history.iter().any(|m| m.role == "assistant"));

    // Drive the reveal + deferred finish to completion, as the run loop does.
    let mut guard = 0;
    loop {
        app.tick_typewriter();
        if app.run_deferred_finish_if_ready().await.unwrap() {
            break;
        }
        guard += 1;
        assert!(guard < 100, "deferred finish should fire once drained");
    }
    assert!(app.pending_finish.is_none());
    assert!(app.incoming_buffer.is_empty());
    assert_eq!(
        app.history
            .last()
            .map(|m| (m.role.as_str(), m.content.as_str())),
        Some(("assistant", "the full reply")),
    );
}

#[test]
fn test_build_transcript_renders_tool_steps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"read_file","args":{"path":"src/parser.rs"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_result".to_string(),
        // Realistic read_file output: right-aligned line numbers + tab + content.
        content: "     1\t/**\n     2\t * sum\n     3\t */".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");
    // Tree-style call line with the salient arg, and a clean line-count result —
    // NOT the noisy first numbered line.
    assert!(
        plain.contains("→ read_file(src/parser.rs)"),
        "missing tool-call line in:\n{plain}"
    );
    assert!(
        plain.contains("⎿ ▸ +3 lines"),
        "missing tool-result line in:\n{plain}"
    );
    assert!(
        !plain.contains("/**"),
        "tool-result summary should not leak the noisy first line:\n{plain}"
    );

    // The call and its result hug (no blank line between them) and both carry
    // the cyan TOOL accent bar.
    let call_idx = transcript
        .plain_lines
        .iter()
        .position(|l| l.contains("→ read_file"))
        .unwrap();
    let result_idx = transcript
        .plain_lines
        .iter()
        .position(|l| l.contains("⎿ ▸ +3 lines"))
        .unwrap();
    assert_eq!(result_idx, call_idx + 1, "result should hug its call");
    assert_eq!(transcript.bar_colors[call_idx], Some(TOOL));
    assert_eq!(transcript.bar_colors[result_idx], Some(TOOL));
}

#[test]
fn test_agent_seed_turns_folds_tool_steps() {
    let msg = |role: &str, content: &str| ChatMessage {
        model: None,
        role: role.to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };
    let history = vec![
        msg("user", "what's in a.rs?"),
        msg("assistant", "let me look"),
        msg(
            "tool_call",
            r#"{"name":"read_file","args":{"path":"src/a.rs"}}"#,
        ),
        msg("tool_result", "     1\tfn main() {}"),
        msg("assistant", "it's an empty main"),
    ];
    let seed = super::runtime_impl::agent_seed_turns(&history);
    // user, assistant, [folded tool note], assistant — the tool step is preserved
    // as an assistant note instead of dropped (which caused resume amnesia).
    assert_eq!(seed.len(), 4);
    assert_eq!(seed[0], ("user".to_string(), "what's in a.rs?".to_string()));
    assert_eq!(seed[2].0, "assistant");
    assert!(
        seed[2].1.contains("read_file") && seed[2].1.contains("a.rs"),
        "tool note missing detail: {}",
        seed[2].1
    );
    assert!(
        seed[2].1.contains('→'),
        "tool note missing outcome: {}",
        seed[2].1
    );
    assert_eq!(seed[3].1, "it's an empty main");
}

#[test]
fn test_build_transcript_renders_edit_diff() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_result".to_string(),
        content: "edited src/a.rs".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines.join("\n");
    // The call line, a removed line, an added line, then the confirmation.
    assert!(
        plain.contains("→ edit_file(src/a.rs)"),
        "missing call in:\n{plain}"
    );
    assert!(
        plain.contains("- let x = 1;"),
        "missing removed line in:\n{plain}"
    );
    assert!(
        plain.contains("+ let x = 2;"),
        "missing added line in:\n{plain}"
    );
    assert!(
        plain.contains("edited src/a.rs"),
        "missing result in:\n{plain}"
    );
}

#[test]
fn test_build_transcript_prettifies_mcp_tool_name() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"mcp__filesystem__read_file","args":{}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("→ filesystem/read_file"),
        "mcp tool name not prettified in:\n{plain}"
    );
    assert!(!plain.contains("mcp__"), "raw mcp__ name leaked:\n{plain}");
}

#[test]
fn test_plan_renders_in_pinned_panel_not_inline() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"scan code","status":"completed"},{"step":"write fix","status":"in_progress"},{"step":"run tests","status":"pending"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    // The plan is pinned in its own panel above the composer — it must NOT render
    // inline in the transcript (where it would scroll away under later content).
    let inline = app.build_transcript().plain_lines.join("\n");
    assert!(
        !inline.contains("Plan") && !inline.contains("scan code"),
        "plan leaked into the inline transcript:\n{inline}"
    );

    // Render the full UI: the pinned panel carries the header, every step, and the
    // per-status glyphs.
    let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let screen: String = (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(screen.contains("Plan"), "panel header missing:\n{screen}");
    assert!(
        screen.contains("1/3 done"),
        "panel progress missing:\n{screen}"
    );
    for step in ["scan code", "write fix", "run tests"] {
        assert!(
            screen.contains(step),
            "panel step {step} missing:\n{screen}"
        );
    }
    assert!(screen.contains('✔') && screen.contains('▸') && screen.contains('○'));
}

fn render_screen(app: &mut CodeTuiApp, w: u16, h: u16) -> String {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn test_completed_plan_hidden_from_panel() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"scan code","status":"completed"},{"step":"write fix","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let screen = render_screen(&mut app, 80, 20);
    assert!(
        !screen.contains("Plan") && !screen.contains("scan code"),
        "a fully-done plan must not stay pinned:\n{screen}"
    );
}

#[test]
fn test_long_plan_windows_to_five_with_more_marker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 10 steps: 3 done, step 3 in progress, rest pending.
    let mut plan = Vec::new();
    for i in 0..10 {
        let status = match i {
            0..=2 => "completed",
            3 => "in_progress",
            _ => "pending",
        };
        plan.push(serde_json::json!({"step": format!("step {i}"), "status": status}));
    }
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: serde_json::Value::Array(plan).to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let screen = render_screen(&mut app, 80, 24);
    assert!(
        screen.contains("3/10 done"),
        "full count in header:\n{screen}"
    );
    assert!(
        screen.contains("step 3"),
        "active step must show:\n{screen}"
    );
    assert!(
        screen.contains("more"),
        "hidden steps need a marker:\n{screen}"
    );
    let step_rows = screen
        .lines()
        .filter(|l| l.contains('✔') || l.contains('▸') || l.contains('○'))
        .count();
    assert!(
        step_rows <= 5,
        "at most 5 steps shown, got {step_rows}:\n{screen}"
    );
    assert!(
        !screen.contains("step 0"),
        "collapsed done step leaked:\n{screen}"
    );
}

#[test]
fn test_completed_plan_clears_on_next_user_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plan_count = |a: &CodeTuiApp| a.history.iter().filter(|m| m.role == "plan").count();

    // A finished plan is recorded and stays pinned (nothing clears it on its own).
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(plan_count(&app), 1);

    // The next user message clears it — `send_user_message` runs this before
    // pushing the turn, so a done plan doesn't linger into a new task.
    app.clear_stale_plan();
    assert_eq!(plan_count(&app), 0, "done plan cleared on next message");

    // A mid-execution plan (some pending, some done) is never auto-cleared.
    app.apply_agent_plan(serde_json::json!([
        {"step": "a", "status": "completed"},
        {"step": "b", "status": "pending"},
    ]));
    app.clear_stale_plan();
    assert_eq!(plan_count(&app), 1, "an active plan must not be cleared");

    // An all-pending proposal is stale once the user moves on — cleared on pivot.
    app.apply_agent_plan(serde_json::json!([
        {"step": "a", "status": "pending"},
        {"step": "b", "status": "pending"},
    ]));
    app.clear_stale_plan();
    assert_eq!(
        plan_count(&app),
        0,
        "unstarted proposal cleared on next message"
    );
}

fn dummy_agent_session() -> AgentSession {
    AgentSession {
        key_id: "k".to_string(),
        model: "m".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::engine::AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0),
        )),
    }
}

#[tokio::test]
async fn test_apply_mcp_connected_empty_keeps_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.mcp_connecting = true;
    app.agent_engine = Some(dummy_agent_session());
    // An empty client (no mcp.json in this temp dir) brings no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-empty-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    app.apply_mcp_connected(client);
    assert!(!app.mcp_connecting, "connecting flag should clear");
    assert!(app.mcp_client.is_some(), "client should be cached");
    assert!(
        app.agent_engine.is_some(),
        "an empty MCP result must not drop the engine"
    );
    assert!(!app.engine_rebuild_pending);
}

#[tokio::test]
async fn test_apply_mcp_connected_surfaces_connect_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A config pointing at a non-spawnable command → a connect error, no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-notice-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    assert!(!client.errors().is_empty(), "expected a connect error");

    app.apply_mcp_connected(client);
    let notice = app
        .notice
        .as_ref()
        .expect("a failed MCP connect should notify");
    assert!(notice.1.contains("MCP"), "notice: {}", notice.1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// When a background connect resolves while the `/mcp` overlay is open, its rows
/// must refresh from the new client in place (no close-and-reopen) — here the
/// "connecting…" row flips to the failure once the broken server's error lands.
#[tokio::test]
async fn test_apply_mcp_connected_refreshes_open_overlay() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "broken".to_string(),
            status: "connecting…".to_string(),
            health: McpHealth::Idle,
            enabled: true,
            scope: ServerScope::Project,
            command: "aivo_no_such_binary_zzz".to_string(),
            remote: false,
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-refresh-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );

    app.apply_mcp_connected(client);
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(
            state.items[0].health,
            McpHealth::Failed,
            "open overlay row not refreshed to failed: {}",
            state.items[0].status
        );
        assert!(
            state.items[0].status.contains("failed"),
            "status: {}",
            state.items[0].status
        );
    } else {
        panic!("mcp overlay vanished");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A connect launched before a `/mcp` toggle (older generation) must be dropped,
/// so it can't resurrect a just-disabled server; the current generation applies.
#[tokio::test]
async fn test_stale_mcp_connect_is_dropped() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx2 = tx.clone();
    let mut app = make_test_app(tx, rx);
    // A toggle advanced the generation while a previous connect is in flight.
    app.mcp_connect_gen = 1;
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-stale-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let stale = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: stale,
        generation: 0,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_none(), "stale connect must not be cached");
    assert!(
        app.mcp_connecting,
        "stale connect must not clear the in-flight flag"
    );

    let fresh = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: fresh,
        generation: 1,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_some(), "current-gen connect should cache");
    assert!(!app.mcp_connecting, "current-gen connect clears the flag");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_maybe_apply_engine_rebuild_drops_engine_when_pending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.agent_engine = Some(dummy_agent_session());
    app.engine_rebuild_pending = true;
    app.maybe_apply_engine_rebuild();
    assert!(
        app.agent_engine.is_none(),
        "pending rebuild should drop engine"
    );
    assert!(!app.engine_rebuild_pending, "flag should clear");
    // Not pending → engine left alone.
    app.agent_engine = Some(dummy_agent_session());
    app.maybe_apply_engine_rebuild();
    assert!(app.agent_engine.is_some());
}

#[test]
fn test_apply_agent_plan_keeps_single_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let count_plans = |app: &CodeTuiApp| app.history.iter().filter(|m| m.role == "plan").count();

    // Two updates with nothing between → one card, updated in place.
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "pending"}]));
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(count_plans(&app), 1, "consecutive updates should collapse");
    assert!(app.history.last().unwrap().content.contains("completed"));

    // A plan after real work still keeps ONE card, relocated to the latest point
    // (so the transcript never stacks a near-identical copy after each batch).
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: "{}".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(count_plans(&app), 1, "plan after work stays a single card");
    assert_eq!(
        app.history.last().unwrap().role,
        "plan",
        "the card relocates to the latest position"
    );
}

#[test]
fn test_build_transcript_coalesces_consecutive_tool_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "study the sidebar".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Cursor-style: a run of read_file calls with no interleaved results.
    for path in ["src/sidebar.rs", "src/session.rs", "src/time.rs"] {
        app.history.push(ChatMessage {
            model: None,
            role: "tool_call".to_string(),
            content: format!(r#"{{"name":"read_file","args":{{"path":"{path}"}}}}"#),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    // A different kind right after starts a new run (not merged with the reads).
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"grep","args":{"pattern":"hover"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines.join("\n");
    // The three reads collapse to one line naming their basenames; the lone grep
    // renders on its own.
    assert!(
        plain.contains("→ read 3 files: sidebar.rs, session.rs, time.rs"),
        "missing coalesced read line in:\n{plain}"
    );
    assert!(
        plain.contains("→ grep(hover)"),
        "lone grep should render normally in:\n{plain}"
    );
    // No per-file card survived the coalescing.
    assert!(
        !plain.contains("read_file(src/sidebar.rs)"),
        "individual read cards should be coalesced away:\n{plain}"
    );
}

#[test]
fn test_render_main_paints_per_role_accent_gutter() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "ping".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Long enough to wrap across several rows at width 24.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "alpha beta gamma delta epsilon zeta eta theta iota kappa".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let backend = TestBackend::new(24, 16);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();

    // Collect the gutter column (x = 0): which rows carry a "▌" and in what color.
    let mut user_bar_rows = 0;
    let mut assistant_bar_rows = 0;
    for y in 0..16u16 {
        let cell = &buf[(0, y)];
        if cell.symbol() != "▌" {
            continue;
        }
        if cell.fg == USER {
            user_bar_rows += 1;
        } else if cell.fg == ACCENT {
            assistant_bar_rows += 1;
        }
    }

    // User block is one row; assistant block wraps onto multiple rows and the
    // accent bar must repeat on every wrapped continuation row, not just the
    // first. Content is also inset past the gutter (no "▌" at the text origin).
    assert_eq!(
        user_bar_rows, 1,
        "user block should show one blue accent bar"
    );
    assert!(
        assistant_bar_rows >= 2,
        "assistant bar must repeat on wrapped rows, got {assistant_bar_rows}"
    );
    assert_ne!(
        buf[(ACCENT_GUTTER_WIDTH, 0)].symbol(),
        "▌",
        "transcript text must render past the gutter"
    );
}

#[test]
fn test_render_main_uses_full_height_for_long_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut composer_area = Rect::default();

    terminal
        .draw(|frame| {
            composer_area = app.render_main(frame, frame.area());
        })
        .unwrap();

    // Composer bottom = height (12) minus the footer's single row.
    assert_eq!(composer_area.y + composer_area.height, 11);
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.y, 0);
    // 80 cols minus the 2-col accent gutter; the overflow transcript keeps full
    // width now that no scrollbar column is reserved.
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.width, 78);
    assert_eq!(app.transcript_width, 78);
}

#[tokio::test]
async fn test_mouse_wheel_scrolls_only_inside_transcript_hitbox() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_width = 80;
    app.transcript_view_height = 6;
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 80, 6),
        first_row: 0,
        rows: wrap_plain_lines(&app.build_transcript().plain_lines, 80),
    });
    app.follow_output = false;
    app.scroll_speed = 4;

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 10,
        row: 8,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert_eq!(app.transcript_scroll, 0);

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 10,
        row: 2,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert_eq!(app.transcript_scroll, 4);
}

#[test]
fn test_chat_scroll_speed_clamps_to_safe_range() {
    assert_eq!(DEFAULT_CHAT_SCROLL_SPEED.clamp(1, MAX_CHAT_SCROLL_SPEED), 3);
    assert_eq!(0usize.clamp(1, MAX_CHAT_SCROLL_SPEED), 1);
    assert_eq!(
        999usize.clamp(1, MAX_CHAT_SCROLL_SPEED),
        MAX_CHAT_SCROLL_SPEED
    );
}

#[test]
fn test_selected_text_normalizes_drag_direction_and_preserves_lines() {
    let rows = vec![
        "alpha beta".to_string(),
        "second line".to_string(),
        "third".to_string(),
    ];
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 2, column: 2 },
        focus: TranscriptPoint { row: 0, column: 6 },
    };

    assert_eq!(
        selected_text_from_rows(&rows, selection).unwrap(),
        "beta\nsecond line\nth"
    );
}

#[test]
fn test_zero_length_selection_is_not_rendered_as_selection() {
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 1, column: 4 },
        focus: TranscriptPoint { row: 1, column: 4 },
    };

    assert!(selection.is_empty());
    assert!(selected_text_from_rows(&["alpha".to_string()], selection).is_none());
}

#[test]
fn test_selection_highlight_preserves_rendered_text_and_foreground() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn rendered_cells(app: &mut CodeTuiApp) -> Vec<(String, Color)> {
        let backend = TestBackend::new(48, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut cells = Vec::new();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                let cell = buffer.cell((x, y)).unwrap();
                cells.push((cell.symbol().to_string(), cell.fg));
            }
        }
        cells
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut normal = make_test_app(tx, rx);
    normal.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "A **styled** answer with enough words to wrap across multiple visual lines."
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut selected = make_test_app(tx, rx);
    selected.history = normal.history.clone();
    selected.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 3, column: 2 },
        focus: TranscriptPoint { row: 4, column: 10 },
    });

    assert_eq!(rendered_cells(&mut normal), rendered_cells(&mut selected));
}

#[test]
fn test_copy_toast_expires_without_touching_transcript_notice() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.notice = None;
    app.toast = Some(Toast {
        text: "Copied selection".to_string(),
        created_at: Instant::now() - TOAST_DURATION,
        expires_at: Instant::now() - Duration::from_millis(1),
    });
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|frame| app.render_toast(frame, frame.area()))
        .unwrap();

    assert!(app.toast.is_none());
    assert!(app.notice.is_none());
}

#[test]
fn test_copy_toast_anchors_bottom_right() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.notice = None;
    // Composer text top sits near the bottom of a 12-row frame; the toast should
    // float on the last transcript row (composer top - 2), not the top edge.
    app.composer_text_area = Some(Rect::new(0, 10, 40, 2));
    app.toast = Some(Toast {
        text: "Copied 5 chars".to_string(),
        created_at: Instant::now(),
        expires_at: Instant::now() + TOAST_DURATION,
    });
    let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
    terminal
        .draw(|frame| app.render_toast(frame, frame.area()))
        .unwrap();
    let buf = terminal.backend().buffer();

    let row_text = |y: u16| -> String {
        (0..40)
            .map(|x| buf.cell((x, y)).unwrap().symbol().to_owned())
            .collect()
    };
    // Anchored on the last transcript row (composer top 10 - 2 = row 8).
    assert!(
        row_text(8).contains("Copied 5 chars"),
        "toast should anchor bottom-right: {:?}",
        row_text(8)
    );
    // The top row stays clear — the old top-right placement is gone.
    assert!(
        !row_text(0).contains("Copied"),
        "toast must not render at the top: {:?}",
        row_text(0)
    );
}

#[tokio::test]
async fn test_auto_approve_toggle_shows_toast_not_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // Shift+Tab (BackTab) flips auto-approve. The confirmation must be a
    // self-expiring toast, NOT a persistent transcript notice pinned above the
    // input for the rest of the session.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve);
    assert!(
        app.notice.is_none(),
        "toggle must not pin a lingering notice"
    );
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Auto-approve mode")),
        "toggle should flash a toast"
    );

    // The next press cycles into plan mode — same toast-not-notice contract.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.agent_auto_approve);
    assert!(app.plan_mode, "auto cycles into plan mode");
    assert!(app.notice.is_none());
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Plan mode"))
    );
}

#[tokio::test]
async fn test_mouse_drag_coordinates_map_to_transcript_rows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A populated transcript hitbox only exists with real content; an empty
    // transcript routes selection to the screen surface (see `selection_target`).
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "x".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(4, 2, 20, 4),
        first_row: 10,
        rows: vec!["a".to_string(); 20],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 7,
        row: 3,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 12,
        row: 5,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 11, column: 3 },
            focus: TranscriptPoint { row: 13, column: 8 },
        })
    );
}

#[tokio::test]
async fn test_screen_drag_selects_off_transcript_from_screen_capture() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A press below the transcript (composer/footer band) selects the screen capture.
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 1),
        first_row: 0,
        rows: vec!["transcript".to_string()],
    });
    app.screen_surface = Some(ScreenSurface {
        area: Rect::new(0, 0, 20, 4),
        rows: vec![
            "transcript".to_string(),
            "alpha beta gamma".to_string(),
            "second line here".to_string(),
            String::new(),
        ],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 6,
        row: 1,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 6,
        row: 2,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert!(app.transcript_selection.is_none());
    assert_eq!(
        app.screen_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 1, column: 6 },
            focus: TranscriptPoint { row: 2, column: 6 },
        })
    );
    assert_eq!(
        app.selected_screen_text().as_deref(),
        Some("beta gamma\nsecond")
    );
}

#[tokio::test]
async fn test_open_overlay_left_drag_selects_overlay_text() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // An open overlay covers the transcript: a left drag selects the overlay text.
    app.overlay = Overlay::Help { scroll: 0 };
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 40, 10),
        first_row: 0,
        rows: vec!["hidden transcript".to_string(); 10],
    });
    app.screen_surface = Some(ScreenSurface {
        area: Rect::new(0, 0, 40, 4),
        rows: vec![
            "  command output line one".to_string(),
            "  command output line two".to_string(),
            String::new(),
            String::new(),
        ],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 2,
        row: 0,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 25,
        row: 0,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert!(app.transcript_selection.is_none());
    assert_eq!(
        app.selected_screen_text().as_deref(),
        Some("command output line one")
    );
}

/// The jump-to-bottom pill appears only while scrolled up (content below the
/// viewport), and clicking it pins back to the latest output.
#[tokio::test]
async fn jump_to_bottom_pill_shows_when_scrolled_up_and_clicks_to_latest() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Enough turns to overflow an 80x24 viewport, so the jump-to-bottom pill applies.
    for i in 0..40 {
        app.history.push(ChatMessage {
            model: None,
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("message line {i}"),
            reasoning_content: None,
            attachments: vec![],
        });
    }

    let render = |app: &mut CodeTuiApp| {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    };

    // Pinned to the bottom → no pill.
    app.follow_output = true;
    render(&mut app);
    assert!(
        app.jump_to_bottom_hit.is_none(),
        "pill hidden while following the bottom"
    );

    // Scroll to the top → content below the viewport, pill appears bottom-right.
    app.follow_output = false;
    app.transcript_scroll = 0;
    render(&mut app);
    let hit = app
        .jump_to_bottom_hit
        .expect("pill shown while scrolled up");

    // Click it → pins back to the latest output.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: hit.x,
        row: hit.y,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert!(
        app.follow_output,
        "clicking the pill jumps to the latest output"
    );

    // Following again → the pill is gone.
    render(&mut app);
    assert!(
        app.jump_to_bottom_hit.is_none(),
        "pill hidden after jumping to the bottom"
    );
}

#[test]
fn test_open_modal_confines_screen_selection_to_its_content() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Help { scroll: 0 };

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();

    // A modal is open → the selection surface is its inner content rect, not the
    // whole screen, so a drag/line-select can't grab the full terminal line.
    let region = app
        .screen_region
        .expect("an open modal sets a selection region");
    assert!(
        region.width < 100,
        "region must be narrower than the screen"
    );
    assert!(region.x > 0, "region starts inside the modal border");
    let surface = app.screen_surface.as_ref().unwrap();
    assert_eq!(surface.area, region);
    assert!(
        surface
            .rows
            .iter()
            .all(|r| row_display_width(r) <= region.width),
        "captured rows must not exceed the modal width"
    );
    assert!(
        surface.rows.iter().any(|r| r.contains("Slash commands")),
        "modal content should be captured: {:?}",
        surface.rows
    );
}

#[test]
fn test_word_bounds_at_grabs_contiguous_word() {
    // "alpha beta gamma" → clicking inside "beta" (cols 6..10) selects 6..10.
    let row = "alpha beta gamma";
    assert_eq!(word_bounds_at(row, 7), Some((6, 10)));
    // Click at the first char of "gamma" (col 11) → 11..16.
    assert_eq!(word_bounds_at(row, 11), Some((11, 16)));
    // Click on the space between words → no word.
    assert_eq!(word_bounds_at(row, 5), None);
    // Click past the end of the text → no word.
    assert_eq!(word_bounds_at(row, 40), None);
}

#[test]
fn test_word_bounds_at_handles_wide_chars() {
    // Each CJK char is 2 display columns: "你好 abc" → 你=0..2, 好=2..4, space=4..5.
    let row = "你好 abc";
    assert_eq!(word_bounds_at(row, 0), Some((0, 4)));
    assert_eq!(word_bounds_at(row, 3), Some((0, 4)));
    assert_eq!(word_bounds_at(row, 4), None); // the space
    assert_eq!(word_bounds_at(row, 5), Some((5, 8))); // "abc"
}

#[test]
fn test_row_display_width_counts_wide_chars() {
    assert_eq!(row_display_width("abc"), 3);
    assert_eq!(row_display_width("你好"), 4);
    assert_eq!(row_display_width(""), 0);
}

#[test]
fn test_selected_text_trims_trailing_wrap_padding() {
    let rows = vec![
        "alpha   ".to_string(),
        "beta".to_string(),
        "gamma  ".to_string(),
    ];
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 2, column: 7 },
    };
    assert_eq!(
        selected_text_from_rows(&rows, selection).unwrap(),
        "alpha\nbeta\ngamma"
    );
}

#[test]
fn test_register_click_counts_double_and_triple() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let point = TranscriptPoint { row: 2, column: 4 };
    assert_eq!(app.register_click(point), 1);
    assert_eq!(app.register_click(point), 2);
    assert_eq!(app.register_click(point), 3);
    // Caps at 3.
    assert_eq!(app.register_click(point), 3);
    // A click on a different row resets the run.
    assert_eq!(app.register_click(TranscriptPoint { row: 9, column: 4 }), 1);
}

#[test]
fn test_register_click_resets_after_interval() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let point = TranscriptPoint { row: 1, column: 1 };
    app.last_click = Some(ClickTracker {
        at: Instant::now() - MULTI_CLICK_INTERVAL - Duration::from_millis(10),
        point,
        count: 1,
    });
    // Too slow to chain → counts as a fresh first click.
    assert_eq!(app.register_click(point), 1);
}

#[test]
fn test_select_word_and_line_set_expected_bounds() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 3),
        first_row: 0,
        rows: vec!["alpha beta".to_string(), "x".to_string()],
    });

    assert!(app.select_word_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 7 }
    ));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 6 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    assert!(app.select_line_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 2 }
    ));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 0 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    // A click on whitespace produces no word selection.
    app.transcript_selection = None;
    assert!(!app.select_word_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 5 }
    ));
    assert!(app.transcript_selection.is_none());
}

#[tokio::test]
async fn test_drag_to_bottom_edge_arms_and_advances_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Real content so max_scroll() (which rebuilds from history) leaves room to
    // scroll past the 4-row viewport.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_width = 20;
    app.transcript_view_height = 4;
    app.transcript_scroll = 0;
    app.follow_output = false;
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 0,
        rows: vec!["row".to_string(); 60],
    });
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 0 },
    });
    app.transcript_drag_active = true;

    // Drag below the bottom edge (row 4 == area.y + height) arms downward scroll.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 5,
        row: 4,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(
        app.drag_autoscroll,
        Some(DragAutoscroll { dir: 1, column: 5 })
    );

    // First tick scrolls one line and re-anchors the focus to the exposed row.
    assert!(app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 1);
    let focus = app.transcript_selection.unwrap().focus;
    assert_eq!(focus.column, 5);
    assert_eq!(focus.row, app.transcript_scroll + 3); // bottom visible row

    // An immediate second tick is throttled (no time has passed).
    assert!(!app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 1);
}

#[tokio::test]
async fn test_drag_to_top_edge_arms_and_advances_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_width = 20;
    app.transcript_view_height = 4;
    app.transcript_scroll = 5;
    app.follow_output = false;
    // The transcript is flush with the top of the screen (area.y == 0), so the
    // pointer can never sit *above* it — scroll-up must arm on the top row.
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 5,
        rows: vec!["row".to_string(); 60],
    });
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 8, column: 0 },
        focus: TranscriptPoint { row: 8, column: 0 },
    });
    app.transcript_drag_active = true;

    // Drag onto the top edge row (row 0 == area.y) arms upward scroll.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 3,
        row: 0,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(
        app.drag_autoscroll,
        Some(DragAutoscroll { dir: -1, column: 3 })
    );

    // First tick scrolls up one line and re-anchors the focus to the top row.
    assert!(app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 4);
    let focus = app.transcript_selection.unwrap().focus;
    assert_eq!(focus.column, 3);
    assert_eq!(focus.row, app.transcript_scroll); // top visible row
}

#[tokio::test]
async fn test_drag_inside_viewport_disarms_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 0,
        rows: vec!["row".to_string(); 40],
    });
    app.drag_autoscroll = Some(DragAutoscroll { dir: 1, column: 2 });

    // A drag back inside the viewport clears the arming.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 5,
        row: 2,
        modifiers: KeyModifiers::NONE,
    });
    assert!(app.drag_autoscroll.is_none());
}

#[test]
fn test_selection_flash_auto_clears_after_window() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 4 },
    });

    // Flash still active → selection retained.
    app.selection_flash_until = Some(Instant::now() + Duration::from_secs(5));
    app.tick_selection_flash();
    assert!(app.transcript_selection.is_some());
    assert!(app.selection_flash_until.is_some());

    // Flash elapsed → selection auto-clears.
    app.selection_flash_until = Some(Instant::now() - Duration::from_millis(1));
    app.tick_selection_flash();
    assert!(app.transcript_selection.is_none());
    assert!(app.selection_flash_until.is_none());
}

#[test]
fn test_highlight_does_not_wash_blank_cells_past_text() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Select well past the two-character "hi" line.
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 30 },
    });

    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();

    let buffer = terminal.backend().buffer();
    // The wash must stop at each row's text: a blank cell *past* the last washed
    // character is the bug. Interior spaces (e.g. the gap in "AIVO Chat") are
    // legitimately washed and must not trip this.
    let area = buffer.area;
    let mut trailing_wash = false;
    for y in area.y..area.y + area.height {
        let mut last_text_col: Option<u16> = None;
        let mut washed_cols = Vec::new();
        for x in area.x..area.x + area.width {
            let cell = buffer.cell((x, y)).unwrap();
            if cell.bg == SELECT_WASH {
                washed_cols.push(x);
                if !cell.symbol().trim().is_empty() {
                    last_text_col = Some(x);
                }
            }
        }
        let past_text = match last_text_col {
            Some(last) => washed_cols.iter().any(|&x| x > last),
            None => !washed_cols.is_empty(), // washed cells, none textual → all trailing
        };
        trailing_wash |= past_text;
    }
    assert!(
        !trailing_wash,
        "selection wash should not cover blank cells past the line's text"
    );
}

#[test]
fn test_clipboard_command_candidates_are_platform_specific() {
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Macos)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["pbcopy"]
    );
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Linux)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["wl-copy", "xclip", "xsel"]
    );
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Windows)[0].program,
        "powershell.exe"
    );
    assert!(clipboard_command_candidates(ClipboardOs::Other).is_empty());
}

#[test]
fn test_clipboard_read_candidates_are_platform_specific() {
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Macos)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["pbpaste"]
    );
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Linux)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["wl-paste", "xclip", "xsel"]
    );
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Windows)[0].program,
        "powershell.exe"
    );
    assert!(clipboard_read_candidates(ClipboardOs::Other).is_empty());
}

#[test]
fn test_parse_slash_command_with_argument() {
    assert_eq!(
        parse_slash_command("model claude-sonnet-4").unwrap(),
        SlashCommand::Model(Some("claude-sonnet-4".to_string()))
    );
    assert_eq!(
        parse_slash_command("attach ./README.md").unwrap(),
        SlashCommand::Attach("./README.md".to_string())
    );
    assert_eq!(
        parse_slash_command("resume").unwrap(),
        SlashCommand::Resume(None)
    );
    assert_eq!(
        parse_slash_command("detach 2").unwrap(),
        SlashCommand::Detach(2)
    );
    assert_eq!(
        parse_slash_command("mcp add fs npx -y srv").unwrap(),
        SlashCommand::Mcp(Some("add fs npx -y srv".to_string()))
    );
    assert_eq!(parse_slash_command("mcp").unwrap(), SlashCommand::Mcp(None));
}

#[test]
fn test_parse_slash_share() {
    assert_eq!(
        parse_slash_command("share").unwrap(),
        SlashCommand::Share(None)
    );
    assert_eq!(
        parse_slash_command("share stop").unwrap(),
        SlashCommand::Share(Some("stop".to_string()))
    );
}

#[test]
fn test_parse_slash_compact() {
    assert_eq!(
        parse_slash_command("compact").unwrap(),
        SlashCommand::Compact { fast: false }
    );
    assert_eq!(
        parse_slash_command("compact fast").unwrap(),
        SlashCommand::Compact { fast: true }
    );
    assert_eq!(
        parse_slash_command("compact now").unwrap(),
        SlashCommand::Compact { fast: false }
    );
}

#[test]
fn test_parse_slash_context() {
    assert_eq!(
        parse_slash_command("context").unwrap(),
        SlashCommand::Context
    );
}

/// `/context` always opens the breakdown, folding in the injected `-c` section when present.
#[tokio::test]
async fn test_context_overlay_shows_breakdown() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Nothing injected: the breakdown still opens.
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Context"), "title:\n{screen}");
    assert!(screen.contains("System prompt"), "segments:\n{screen}");
    assert!(screen.contains("Tools"), "segments:\n{screen}");
    assert!(
        !screen.contains("Injected context"),
        "no injection expected:\n{screen}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));

    // With an injected block, the breakdown also carries the injected section + body.
    app.injected_context = Some("# aivo context\n\n**Topic:** prior work".to_string());
    app.injected_context_summary = Some("injected ~9 tokens from claude session abc (2m)".into());
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 44);
    assert!(screen.contains("Injected context"), "section:\n{screen}");
    assert!(
        screen.contains("injected ~9 tokens from claude session abc"),
        "summary header:\n{screen}"
    );
    assert!(screen.contains("Topic:"), "body:\n{screen}");
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_compact_command_no_engine_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_compact_command(true).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("nothing to compact"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.agent_serve.is_none() && app.response_task.is_none());
}

#[tokio::test]
async fn test_share_command_stop_and_usage_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `/share stop` with no active share → informative notice, nothing started.
    app.run_share_command(Some("stop".to_string())).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("Not currently"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.live_share.is_none());

    // Unknown argument → usage notice (no background start).
    app.run_share_command(Some("frobnicate".to_string())).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("Usage"),
        "notice: {:?}",
        app.notice
    );
    assert!(!app.live_share_starting);
}

#[tokio::test]
async fn test_share_command_reshows_url_then_stops() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.live_share = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/v.html?t=zz",
    ));

    // Bare `/share` while already sharing just re-shows the URL — no new start.
    app.run_share_command(None).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("t=zz"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.live_share.is_some());
    assert!(!app.live_share_starting);

    // `/share stop` tears it down.
    app.run_share_command(Some("stop".to_string())).await;
    assert!(app.live_share.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("stopped"));
}

#[test]
fn test_apply_live_share_ready_ok_and_err() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.live_share_starting = true;
    app.apply_live_share_ready(
        app.live_share_gen,
        Ok(LiveShareHandle::for_test(
            "https://s.getaivo.dev/v.html?t=ok",
        )),
    );
    assert!(!app.live_share_starting);
    assert!(app.live_share.is_some());
    assert!(app.notice.as_ref().unwrap().1.contains("t=ok"));

    // Failure: clears the starting flag, surfaces the reason, stores nothing.
    app.live_share = None;
    app.live_share_starting = true;
    app.apply_live_share_ready(app.live_share_gen, Err("no link".to_string()));
    assert!(!app.live_share_starting);
    assert!(app.live_share.is_none());
    assert_eq!(app.notice.as_ref().unwrap().1, "no link");
}

/// A share start outlived by a stop//new//resume must not install its tunnel;
/// a start under the fresh generation still lands.
#[test]
fn test_stale_live_share_ready_is_dropped() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.live_share_starting = true;
    let stale_gen = app.live_share_gen;
    assert!(
        app.stop_live_share(),
        "cancelling a mid-handshake start counts as a stop"
    );
    assert!(!app.live_share_starting);

    app.apply_live_share_ready(stale_gen, Ok(LiveShareHandle::for_test("https://s/old")));
    assert!(app.live_share.is_none(), "stale handle must not install");

    // A new start under the bumped generation works.
    app.live_share_starting = true;
    app.apply_live_share_ready(
        app.live_share_gen,
        Ok(LiveShareHandle::for_test("https://s/new")),
    );
    assert!(app.live_share.is_some());
}

/// A dead tunnel (network drop — no auto-reconnect) clears the badge and its
/// server on the next frame instead of showing a live share that no longer serves.
#[test]
fn test_dead_share_tunnel_clears_badge() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let handle = LiveShareHandle::for_test("https://s/z");
    handle.mark_dead_for_test();
    app.live_share = Some(handle);

    app.check_live_share_health();

    assert!(app.live_share.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("disconnected"));

    // No share → no-op (must not overwrite an unrelated notice).
    app.notice = None;
    app.check_live_share_health();
    assert!(app.notice.is_none());
}

#[test]
fn test_footer_shows_share_badge_only_when_sharing() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // No share → no badge.
    let (screen, _) = render_full_screen(&mut app, 80, 12);
    assert!(
        !screen.contains("● sharing"),
        "share badge shown without an active share:\n{screen}"
    );

    // Active share → the `● sharing` badge appears in the footer.
    app.live_share = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/v.html?t=ab",
    ));
    let (screen, _) = render_full_screen(&mut app, 80, 12);
    assert!(
        screen.contains("● sharing"),
        "no share badge in footer while sharing:\n{screen}"
    );
}

#[tokio::test]
async fn test_maybe_start_live_share_defers_until_session_settles() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // No `--share` request → never starts.
    assert!(!app.maybe_start_live_share().await);

    app.live_requested = true;

    // A pending `--resume` load defers the start (it must pin the resumed session).
    app.loading_resume = Some(LoadingResume {
        request_id: 1,
        preview: SessionPreview {
            key_id: "k".into(),
            key_name: "k".into(),
            base_url: "u".into(),
            session_id: "resumed".into(),
            raw_model: "m".into(),
            updated_at: "t".into(),
            title: "t".into(),
            preview_text: "p".into(),
        },
    });
    assert!(!app.maybe_start_live_share().await);
    assert!(
        app.live_requested,
        "request stays pending while resume loads"
    );
    app.loading_resume = None;

    // An already-running share or an in-flight start are both no-ops.
    app.live_share = Some(LiveShareHandle::for_test("https://x"));
    assert!(!app.maybe_start_live_share().await);
    app.live_share = None;
    app.live_share_starting = true;
    assert!(!app.maybe_start_live_share().await);
    assert!(app.live_requested);
}

#[test]
fn test_empty_state_notice_selects_via_screen_surface() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // `--share` launch state: empty transcript, share-URL notice (the notice draws
    // the URL line; the handle just drives the badge).
    assert!(app.is_transcript_empty());
    app.live_share = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/s/uniqueurlzz",
    ));
    app.notice = Some((
        LIVE,
        format!("{LIVE_NOTICE_PREFIX}https://s.getaivo.dev/s/uniqueurlzz"),
    ));
    let (_, rows) = render_full_screen(&mut app, 80, 16);
    let url_row = rows
        .iter()
        .position(|r| r.contains("uniqueurlzz"))
        .expect("live URL rendered in the empty state") as u16;

    // The press must target the screen surface, not the empty transcript hitbox.
    let mouse = MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 12,
        row: url_row,
        modifiers: KeyModifiers::NONE,
    };
    assert!(
        matches!(
            app.selection_target(mouse, false),
            Some((SelectionSurface::Screen, _))
        ),
        "empty-state notice should select via the screen surface"
    );
}

#[test]
fn test_notice_spans_splits_live_url_from_indicator() {
    // The share notice paints `● Sharing:` red but the URL a calm link color, so
    // the long line doesn't read as an error. Other notices stay a single span.
    let share = (
        LIVE,
        format!("{LIVE_NOTICE_PREFIX}https://s.getaivo.dev/s/abc"),
    );
    let spans = notice_spans(Some(&share)).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].content.as_ref(), LIVE_NOTICE_PREFIX);
    assert_eq!(spans[0].style.fg, Some(LIVE));
    assert_eq!(spans[1].content.as_ref(), "https://s.getaivo.dev/s/abc");
    assert_eq!(spans[1].style.fg, Some(LINK));

    let plain = (MUTED, "just a status".to_string());
    let spans = notice_spans(Some(&plain)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].style.fg, Some(MUTED));

    // ERROR keeps its `Error:` prefix and single span.
    let err = (ERROR, "boom".to_string());
    let spans = notice_spans(Some(&err)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].content.as_ref(), "Error: boom");
}

#[tokio::test]
async fn test_mcp_add_project_flag_writes_repo_config_and_grants_consent() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let dir = std::env::temp_dir().join(format!("aivo-mcp-project-add-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();

    app.submit_mcp_add("-p echo hi".to_string()).await.unwrap();

    // Written to the repo `.mcp.json`, not the user config.
    let root: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(".mcp.json")).unwrap()).unwrap();
    let servers = root["mcpServers"].as_object().unwrap();
    assert!(
        servers.values().any(|v| v["command"] == "echo"),
        "project .mcp.json holds the added server: {root}"
    );
    // Typing the command IS the consent — run-once session approval, like `y`.
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    assert!(app.pending_mcp_consent.is_none());
    let notice = app.notice.as_ref().unwrap().1.clone();
    assert!(notice.contains("./.mcp.json"), "notice: {notice}");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn test_mcp_multi_paste_opens_picker_and_replaces_on_apply() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let dir = std::env::temp_dir().join(format!("aivo-mcp-paste-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();
    // A same-named server is already configured (project scope).
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"github":{"command":"old-cmd"}}}"#,
    )
    .unwrap();

    app.submit_mcp_add(
        r#"-p {"mcpServers":{
            "github":{"command":"echo","args":["new"]},
            "linear":{"url":"http://127.0.0.1:1/mcp"}
        }}"#
        .to_string(),
    )
    .await
    .unwrap();

    // ≥2 servers → picker, not a blind add: the new name is prechecked, the
    // existing one needs an explicit replace mark.
    let github = {
        let Overlay::McpPaste(state) = &app.overlay else {
            panic!("expected the paste picker to open");
        };
        assert!(state.project);
        assert!(
            state.parent.is_none(),
            "composer paste has no /mcp to restore"
        );
        let github = state.items.iter().position(|i| i.name == "github").unwrap();
        let linear = state.items.iter().position(|i| i.name == "linear").unwrap();
        assert!(state.items[github].exists && !state.items[github].checked);
        assert!(!state.items[linear].exists && state.items[linear].checked);
        github
    };
    // Mark the existing row too — that means replace-in-place.
    if let Overlay::McpPaste(state) = &mut app.overlay {
        state.items[github].checked = true;
    }
    app.apply_mcp_paste().await.unwrap();

    let root: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(".mcp.json")).unwrap()).unwrap();
    let servers = root["mcpServers"].as_object().unwrap();
    assert_eq!(servers.len(), 2, "no `github-2` duplicate: {root}");
    assert_eq!(servers["github"]["command"], "echo", "replaced in place");
    assert_eq!(servers["linear"]["url"], "http://127.0.0.1:1/mcp");
    let notice = app.notice.as_ref().unwrap().1.clone();
    assert!(
        notice.contains("Added linear") && notice.contains("replaced github"),
        "notice: {notice}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn test_mcp_add_json_routes_and_reports_parse_error() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A `{`-leading add input routes to the JSON path; malformed JSON surfaces a
    // parse error (and writes nothing — verified by never reaching a real write).
    app.submit_mcp_add("{ not valid json".to_string())
        .await
        .unwrap();
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("Couldn't parse MCP config"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

#[tokio::test]
async fn test_mcp_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_mcp_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Mcp(_)),
        "bare /mcp opens overlay"
    );

    // Unknown subcommand → usage notice, overlay not opened.
    app.overlay = Overlay::None;
    app.run_mcp_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name → usage notice.
    app.run_mcp_command(Some("rm".to_string())).await.unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent server → "No MCP server" notice, no config write.
    app.run_mcp_command(Some("rm __aivo_no_such_server__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No MCP server"),
        "notice: {:?}",
        app.notice
    );
}

#[test]
fn test_parse_slash_command_unknown() {
    let err = parse_slash_command("wat").unwrap_err().to_string();
    assert!(err.contains("Unknown command"));
}

#[test]
fn test_parse_slash_skills() {
    assert_eq!(
        parse_slash_command("skills").unwrap(),
        SlashCommand::Skills(None)
    );
    assert_eq!(
        parse_slash_command("skills add fs Helper").unwrap(),
        SlashCommand::Skills(Some("add fs Helper".to_string()))
    );
    // `/skills` is advertised in the command menu + help listing.
    assert!(SLASH_COMMANDS.iter().any(|c| c.name == "skills"));
}

fn skill_command(name: &str, description: &str) -> SkillCommand {
    SkillCommand {
        name: name.to_string(),
        description: description.to_string(),
    }
}

#[test]
fn test_filter_skill_commands_ranks_prefix_before_fuzzy() {
    let commands = vec![
        skill_command("repo-study", "Study a repo"),
        skill_command("review", "Review a PR"),
        skill_command("deep-research", "Research a topic"),
    ];
    // Empty query returns everything, order preserved.
    assert_eq!(filter_skill_commands(&commands, "").len(), 3);
    // Prefix match wins; the fuzzy `re…h` (deep-research) still shows, but after.
    let names: Vec<String> = filter_skill_commands(&commands, "re")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names.first().map(String::as_str), Some("repo-study"));
    assert!(names.contains(&"review".to_string()));
    // A query matching nothing yields nothing.
    assert!(filter_skill_commands(&commands, "zzz").is_empty());
}

#[test]
fn test_resolve_slash_command_skill_vs_builtin_vs_typo() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![skill_command("repo-study", "Study a repo")];

    // A discovered skill resolves to the Skill variant, with trailing args.
    assert_eq!(
        app.resolve_slash_command("repo-study https://x/y").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: Some("https://x/y".to_string()),
        }
    );
    // No args → None.
    assert_eq!(
        app.resolve_slash_command("repo-study").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: None,
        }
    );
    // A built-in always wins, even if a same-named skill exists.
    app.skill_commands.push(skill_command("model", "shadow"));
    assert_eq!(
        app.resolve_slash_command("model gpt").unwrap(),
        SlashCommand::Model(Some("gpt".to_string()))
    );
    // An unknown name (not a built-in, not a skill) still errors.
    let err = app.resolve_slash_command("nope").unwrap_err().to_string();
    assert!(err.contains("Unknown command"), "{err}");
}

#[test]
fn test_matching_command_entries_includes_skills_after_builtins() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![
        skill_command("repo-study", "Study a repo"),
        // A skill colliding with the built-in `/model` must be dropped.
        skill_command("model", "shadow"),
    ];
    // No built-in starts with "repo", so the skill is the sole match.
    let entries = app.matching_command_entries("repo");
    let labels: Vec<String> = entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(labels, vec!["/repo-study".to_string()]);

    // `/model` resolves to the built-in only — the colliding skill never appears.
    let model_entries = app.matching_command_entries("model");
    let model_labels: Vec<String> = model_entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(
        model_labels.iter().filter(|l| *l == "/model").count(),
        1,
        "a skill must not duplicate or shadow a built-in command"
    );
}

#[tokio::test]
async fn test_refresh_skill_commands_discovers_and_respects_disabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A project skill under the app's working dir, with a unique name so real
    // home-dir skills can't collide with the assertions.
    let proj = std::env::temp_dir().join(format!(
        "aivo-skillcmd-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let name = "zz-unique-skill-cmd";
    let dir = proj.join(".aivo").join("skills").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: A unique test skill.\n---\nBody.\n"),
    )
    .unwrap();
    app.real_cwd = proj.to_string_lossy().into_owned();

    app.refresh_skill_commands().await;
    let found = app.skill_commands.iter().find(|c| c.name == name);
    assert!(found.is_some(), "discovered skill should become a command");
    assert_eq!(found.unwrap().description, "A unique test skill.");

    // Disabling it in `/skills` drops it from the command set.
    app.session_store
        .set_skill_enabled(name, false)
        .await
        .unwrap();
    app.refresh_skill_commands().await;
    assert!(
        !app.skill_commands.iter().any(|c| c.name == name),
        "a disabled skill must not be offered as a command"
    );

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn test_expand_skill_invocation_arguments_and_fallback() {
    use crate::agent::skills::Skill;
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // `$ARGUMENTS` is substituted in place.
    assert_eq!(
        super::runtime_impl::expand_skill_invocation(&placeholder, Some("the repo")),
        "Study the repo now."
    );

    let plain = Skill {
        name: "repo-study".to_string(),
        description: "d".to_string(),
        body: "Follow these steps.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // No placeholder + args → directive + body + appended input.
    let out = super::runtime_impl::expand_skill_invocation(&plain, Some("https://x/y"));
    assert!(out.contains("Use the \"repo-study\" skill"), "{out}");
    assert!(out.contains("Follow these steps."), "{out}");
    assert!(out.ends_with("Input: https://x/y"), "{out}");
    // No placeholder, no args → no trailing input line.
    let bare = super::runtime_impl::expand_skill_invocation(&plain, None);
    assert!(bare.contains("Follow these steps."));
    assert!(!bare.contains("Input:"));
}

/// The display/log recognizer recovers the compact `/name args` the user typed
/// from an expanded invocation, round-tripping with the producer, and declines
/// ordinary messages and `$ARGUMENTS`-style skills (which leave no wrapper).
#[test]
fn test_skill_invocation_label_recovers_typed_command() {
    use super::runtime_impl::{expand_skill_invocation, skill_invocation_label};
    use crate::agent::skills::Skill;
    let skill = Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "Search Baidu.\n\nUse the bundled script.".to_string(),
        dir: std::path::PathBuf::new(),
    };

    // With args (incl. CJK) → `/name args`.
    let expanded = expand_skill_invocation(&skill, Some("歌曲"));
    assert_eq!(
        skill_invocation_label(&expanded).as_deref(),
        Some("/baidu-search 歌曲")
    );
    // No args → bare `/name`.
    let bare = expand_skill_invocation(&skill, None);
    assert_eq!(
        skill_invocation_label(&bare).as_deref(),
        Some("/baidu-search")
    );

    // A body that itself contains an `Input:`-style line must not be mistaken for
    // args when none were passed (multi-line tail ⇒ no args).
    let trappy = Skill {
        name: "trap".to_string(),
        description: "d".to_string(),
        body: "Step.\n\nInput: foo\nmore body".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&trappy, None)).as_deref(),
        Some("/trap")
    );

    // Ordinary user text and a `$ARGUMENTS` skill (no wrapper) → None.
    assert_eq!(skill_invocation_label("just a normal question"), None);
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&placeholder, Some("x"))),
        None
    );
}

/// The transcript shows the compact `/name args` for a skill turn, not the whole
/// inlined SKILL.md body (the verbosity reported in the field).
#[tokio::test]
async fn test_skill_turn_renders_compact_not_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let skill = crate::agent::skills::Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "SEARCH_BAIDU_BODY_MARKER do the thing.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    let expanded = super::runtime_impl::expand_skill_invocation(&skill, Some("歌曲"));
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: expanded,
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The compact command shows (the test backend pads wide CJK glyphs with an
    // inter-cell space, so assert on the ASCII command + each arg glyph rather
    // than the exact contiguous string).
    assert!(
        screen.contains("/baidu-search"),
        "compact label missing:\n{screen}"
    );
    assert!(
        screen.contains('歌') && screen.contains('曲'),
        "skill arg missing from the turn:\n{screen}"
    );
    assert!(
        !screen.contains("SEARCH_BAIDU_BODY_MARKER"),
        "the inlined body leaked into the transcript:\n{screen}"
    );
}

fn skills_overlay_fixture() -> SkillsOverlay {
    use crate::agent::skills::SkillScope;
    SkillsOverlay {
        items: vec![
            SkillToggle {
                name: "brandkit".to_string(),
                description: "Premium brand-kit image generation.".to_string(),
                enabled: true,
                dir: std::path::PathBuf::from("/home/me/.config/aivo/skills/brandkit"),
                scope: SkillScope::User,
                body: "Step 1. Render the boards.".to_string(),
            },
            SkillToggle {
                name: "critique".to_string(),
                description: "Evaluate design effectiveness from a UX perspective.".to_string(),
                enabled: false,
                dir: std::path::PathBuf::from("/repo/.agents/skills/critique"),
                scope: SkillScope::Project,
                body: "Step 1. Score the design.".to_string(),
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

#[test]
fn test_skills_overlay_renders_toggle_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("Skills"), "missing title:\n{screen}");
    assert!(screen.contains("brandkit"), "missing skill name:\n{screen}");
    assert!(
        screen.contains("[✓]"),
        "missing enabled checkbox:\n{screen}"
    );
    assert!(
        screen.contains("[ ]"),
        "missing disabled checkbox:\n{screen}"
    );
    // The name and its description render on separate lines.
    assert!(
        screen.contains("Premium brand-kit"),
        "missing description line:\n{screen}"
    );
    // The on-count badge sits in the top border (1 of 2 on).
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
    // Search placeholder up top, controls along the footer.
    assert!(
        screen.contains("filter skills") && screen.contains("toggle"),
        "missing controls:\n{screen}"
    );
}

/// The detail line shows the selected skill's location, and tags a project
/// skill (which `d` can't delete). Selecting the project-scoped row surfaces
/// the `project` marker that appears nowhere else on screen.
#[test]
fn test_skills_overlay_detail_line_shows_scope_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.selected = 1; // "critique", a project skill
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project"),
        "detail line should tag a project-scoped skill:\n{screen}"
    );
    assert!(
        screen.contains("skills/critique"),
        "detail line should show the skill's path:\n{screen}"
    );
}

#[tokio::test]
async fn test_toggle_skill_persists_and_resets_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Toggle "brandkit" (index 0) from enabled → disabled.
    app.toggle_skill(0).await.unwrap();

    if let Overlay::Skills(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
    } else {
        panic!("skills overlay vanished");
    }
    // Persisted to the store, and the engine is dropped so the next turn rebuilds.
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert_eq!(disabled, vec!["brandkit".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_skill(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_skills()
            .await
            .unwrap()
            .is_empty()
    );
}

/// A name hit must outrank rows whose long description merely subsequence-
/// matches, and typing re-anchors the selection to that top hit.
#[test]
fn test_skills_filter_ranks_name_matches_first() {
    use crate::agent::skills::SkillScope;
    let skill = |name: &str, description: &str| SkillToggle {
        name: name.to_string(),
        description: description.to_string(),
        enabled: false,
        dir: std::path::PathBuf::from("/tmp/x"),
        scope: SkillScope::User,
        body: String::new(),
    };
    let mut overlay = SkillsOverlay {
        // "big old dear" contains b-o-l-d-e-r as a subsequence.
        items: vec![
            skill("alpha", "big old dear"),
            skill("bolder", "amplify designs"),
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    };

    overlay.query = "bolder".to_string();
    overlay.refilter();

    assert_eq!(
        overlay.filtered_indices(),
        vec![1, 0],
        "name match must rank above a description-only match"
    );
    assert_eq!(overlay.selected, 1, "typing re-anchors to the top hit");
}

/// Space is a dead key in a fuzzy filter, so it toggles instead of typing.
#[tokio::test]
async fn test_space_toggles_without_entering_filter() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(s) = &app.overlay {
        assert!(
            !s.items[0].enabled,
            "Space should toggle the selected skill"
        );
        assert!(s.query.is_empty(), "Space must never enter the filter");
    } else {
        panic!("skills overlay vanished");
    }

    // With no visible selection (filter matches nothing) Space is a no-op.
    if let Overlay::Skills(s) = &mut app.overlay {
        s.query = "zzz".to_string();
        s.refilter();
    }
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(s) = &app.overlay {
        assert_eq!(s.query, "zzz");
    }

    app.overlay = Overlay::Mcp(mcp_overlay_fixture());
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(s) = &app.overlay {
        assert!(
            !s.items[0].enabled,
            "Space should toggle the selected server"
        );
        assert!(s.query.is_empty());
    } else {
        panic!("mcp overlay vanished");
    }
}

/// Discovery leaves `Skill::body` empty (lazy) and the advert truncates the
/// description — the overlay must load/keep both in full for the detail pane.
#[tokio::test]
async fn test_open_skills_overlay_loads_full_body_and_description() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let dir = std::env::temp_dir().join(format!("aivo-skill-detail-{}", std::process::id()));
    let skill_dir = dir.join(".agents").join("skills").join("fulltext");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: fulltext\ndescription: First sentence. Second sentence the advert would drop.\n---\nLine one of instructions.\nLine two of instructions.\n",
    )
    .unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();

    app.open_skills_overlay().await.unwrap();
    let Overlay::Skills(state) = &app.overlay else {
        panic!("skills overlay did not open");
    };
    let item = state
        .items
        .iter()
        .find(|i| i.name == "fulltext")
        .expect("skill discovered");
    assert!(
        item.description.contains("Second sentence"),
        "full description should survive into the overlay: {}",
        item.description
    );
    assert!(
        item.body.contains("Line two of instructions"),
        "SKILL.md body should be read at open time: {:?}",
        item.body
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_parse_skill_add_input() {
    use super::session_impl::parse_skill_add_input;
    // First token is the name; the rest is a free-text description.
    let (name, desc) = parse_skill_add_input("changelog Summarize the git log").unwrap();
    assert_eq!(name, "changelog");
    assert_eq!(desc, "Summarize the git log");
    // A bare name (no description) is fine — a placeholder is templated in.
    assert_eq!(
        parse_skill_add_input("solo").unwrap(),
        ("solo".to_string(), String::new())
    );
    // The first token is the name (so a name can't contain a space) and the
    // remainder is the description.
    let (name, desc) = parse_skill_add_input("multi word description here").unwrap();
    assert_eq!(name, "multi");
    assert_eq!(desc, "word description here");
    // A folder-unsafe single-token name is rejected.
    assert!(parse_skill_add_input("a.b desc").is_err(), "dot in name");
    assert!(parse_skill_add_input("").is_err(), "empty");
}

#[test]
fn test_skill_add_success_notice_includes_advert_and_warnings() {
    use super::session_impl::skill_add_success_notice;
    let path = std::path::Path::new("/tmp/aivo-test/skills/deploy/SKILL.md");

    let notice = skill_add_success_notice("deploy", "Deploy safely", path);
    assert!(notice.contains("Created skill `deploy`"), "{notice}");
    assert!(notice.contains("Advert: Deploy safely"), "{notice}");
    assert!(!notice.contains("Warning:"), "{notice}");

    let multi = skill_add_success_notice(
        "deploy",
        "Deploy safely. Use when release or rollback cues appear.",
        path,
    );
    assert!(multi.contains("Advert: Deploy safely."), "{multi}");
    assert!(
        multi.contains("only first sentence is advertised"),
        "{multi}"
    );

    let blank = skill_add_success_notice("deploy", "", path);
    assert!(blank.contains("Advert: One-line summary"), "{blank}");
    assert!(blank.contains("replace placeholder description"), "{blank}");
}

#[tokio::test]
async fn test_skills_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_skills_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Skills(_)),
        "bare /skills opens overlay"
    );

    // An unknown verb is a usage error.
    app.run_skills_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name is a usage error.
    app.run_skills_command(Some("rm".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent skill → "No skill" notice, no deletion.
    app.run_skills_command(Some("rm __aivo_no_such_skill__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No skill"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

#[test]
fn test_parse_goal_command() {
    assert_eq!(
        parse_slash_command("goal").unwrap(),
        SlashCommand::Goal(None)
    );
    assert_eq!(
        parse_slash_command("goal ship the feature").unwrap(),
        SlashCommand::Goal(Some("ship the feature".to_string()))
    );
    assert_eq!(
        parse_slash_command("goal stop").unwrap(),
        SlashCommand::Goal(Some("stop".to_string()))
    );
}

#[test]
fn test_parse_plan_command() {
    assert_eq!(
        parse_slash_command("plan").unwrap(),
        SlashCommand::Plan(None)
    );
    assert_eq!(
        parse_slash_command("plan add a cache layer").unwrap(),
        SlashCommand::Plan(Some("add a cache layer".to_string()))
    );
    assert_eq!(
        parse_slash_command("plan go").unwrap(),
        SlashCommand::Plan(Some("go".to_string()))
    );
}

#[test]
fn test_plan_go_message_appends_guidance() {
    use super::runtime_impl::plan_go_message;
    let bare = plan_go_message("");
    assert!(bare.contains("approved"));
    assert!(!bare.contains("Additional guidance"));
    let steered = plan_go_message("use the existing retry helper");
    assert!(steered.starts_with(&bare));
    assert!(steered.contains("Additional guidance: use the existing retry helper"));
}

/// The plan-card anchor slides down when an earlier history entry is removed
/// (e.g. an `update_plan` checklist card dropped by `drop_plan_entries`).
#[tokio::test]
async fn test_plan_card_idx_shifts_on_removal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let msg = |role: &str, c: &str| ChatMessage {
        model: None,
        role: role.to_string(),
        content: c.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };
    app.history.clear();
    app.history.push(msg("user", "hi")); // 0
    app.history.push(msg("plan", "[]")); // 1 (checklist card — dropped below)
    app.history.push(msg("assistant", "plan body")); // 2
    app.plan_card_idx = Some(2);
    app.drop_plan_entries();
    assert_eq!(
        app.plan_card_idx,
        Some(1),
        "anchor follows the assistant down"
    );
    assert_eq!(app.history[1].role, "assistant");
}

/// Plan-mode state machine without the dispatch paths (which need a serve):
/// a finished plan-mode turn drafts its reply as the pending plan while the
/// MODE PERSISTS; `stop` leaves the mode; bare while on reports status (with
/// vs without a draft); `go` with nothing pending just guides.
#[tokio::test]
async fn test_plan_capture_discard_and_status() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let assistant = |content: &str| ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    // Bare `/plan` in the mode with nothing drafted points at the composer.
    app.plan_mode = true;
    app.run_plan_command(None).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("describe what to plan")
    );

    // A finished plan-mode turn stashes the reply as the draft — and stays in
    // plan mode (persistent until approved or stopped).
    app.history.push(assistant("1. do X\n2. do Y"));
    app.capture_plan_draft();
    assert!(app.plan_mode, "plan mode persists after a draft");
    assert_eq!(app.pending_plan.as_deref(), Some("1. do X\n2. do Y"));
    assert!(app.notice.as_ref().unwrap().1.contains("/plan go"));
    // The captured reply is anchored as the plan card.
    assert_eq!(
        app.plan_card_idx,
        app.history.iter().rposition(|m| m.role == "assistant")
    );

    // Bare `/plan` with a drafted plan points at the approval card instead.
    app.run_plan_command(None).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("approve the plan card")
    );

    // `/plan stop` leaves plan mode, discarding the draft and the card frame.
    app.run_plan_command(Some("stop".to_string())).await;
    assert!(!app.plan_mode);
    assert!(app.pending_plan.is_none());
    assert!(app.plan_card_idx.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("discarded"));

    // `/plan go` with nothing pending guides instead of dispatching.
    app.run_plan_command(Some("go".to_string())).await;
    assert!(app.notice.as_ref().unwrap().1.contains("No plan yet"));

    // `/plan go <guidance>` routes to execute (first word), not a new objective.
    app.run_plan_command(Some("go also add tests".to_string()))
        .await;
    assert!(app.notice.as_ref().unwrap().1.contains("No plan yet"));

    // An empty reply leaves the draft untouched (all-tool-call turns).
    app.plan_mode = true;
    app.history.push(assistant("   "));
    app.capture_plan_draft();
    assert!(app.pending_plan.is_none(), "blank reply isn't a plan");

    // An interrupt keeps plan mode on (regression: the old one-way read-only
    // restriction leaked past the mode when the engine survived a cancel).
    app.cancel_inflight_request(super::CancelKind::Discard);
    assert!(app.plan_mode, "plan mode persists across an interrupt");
    app.run_plan_command(Some("stop".to_string())).await;
    assert!(!app.plan_mode);
}

/// Approval-card verdicts, Claude Code-style: 1 = approve + auto-approve,
/// 2 = approve + manual approval, 3 = keep planning. (Bare `/plan`'s kick-off
/// dispatch is covered in `test_plan_bare_dispatches_kickoff`.)
#[tokio::test]
async fn test_plan_mode_enter_and_approval_verdicts() {
    use crate::agent::protocol::PlanDecision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Enter the mode (no engine yet — build-time entry).
    assert!(app.enter_plan_mode().await);
    assert!(app.plan_mode);

    // Approve & auto-approve: mode off, execution continues unattended.
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.agent_plan_approval = Some(super::PendingPlanApproval {
        body: vec![],
        scroll: 0,
        selected: 0,
        reply,
    });
    app.pick_plan_approval_option(0);
    assert!(!app.plan_mode, "approval exits plan mode");
    assert!(!app.plan_exit_pending);
    assert!(app.agent_auto_approve, "option 1 lands in auto mode");
    assert!(!app.agent_review_edits, "modes are exclusive");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows so the running turn sees it"
    );
    assert_eq!(rx1.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Approve with per-edit review: mode off, review mode on.
    app.plan_mode = true;
    let (reply, mut rx2) = tokio::sync::oneshot::channel();
    app.agent_plan_approval = Some(super::PendingPlanApproval {
        body: vec![],
        scroll: 0,
        selected: 0,
        reply,
    });
    app.pick_plan_approval_option(1);
    assert!(!app.plan_mode);
    assert!(!app.agent_auto_approve, "option 2 lands in review mode");
    assert!(app.agent_review_edits, "each edit will show a diff");
    assert!(
        app.review_edits_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live review flag follows mid-turn"
    );
    assert_eq!(rx2.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Keep planning: mode stays on.
    app.plan_mode = true;
    let (reply, mut rx3) = tokio::sync::oneshot::channel();
    app.agent_plan_approval = Some(super::PendingPlanApproval {
        body: vec![],
        scroll: 0,
        selected: 0,
        reply,
    });
    app.pick_plan_approval_option(2);
    assert!(app.plan_mode, "keep-planning stays in plan mode");
    assert_eq!(
        rx3.try_recv().unwrap(),
        Ok(PlanDecision::KeepPlanning { feedback: None })
    );
}

/// Shift+Tab cycles the agent mode Claude Code-style: default → auto → plan →
/// review → default, with the modes mutually exclusive. Mid-turn the plan step
/// is skipped (the engine can't be restricted while a turn holds it).
#[tokio::test]
async fn test_shift_tab_cycles_agent_modes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mode = |app: &super::CodeTuiApp| {
        (
            app.agent_auto_approve,
            app.plan_mode,
            app.agent_review_edits,
        )
    };
    assert_eq!(mode(&app), (false, false, false), "starts in default");

    // default → auto.
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (true, false, false));

    // auto → plan (auto forced off: exclusive).
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, true, false), "auto cycles into plan");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    );

    // plan → review (the drafted plan would survive; here there is none).
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "plan cycles into review");
    assert!(
        app.review_edits_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live review flag follows"
    );

    // review → default.
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, false), "full circle");

    // Mid-turn: auto → review directly — plan entry is skipped while sending.
    app.cycle_agent_mode().await; // default → auto
    app.sending = true;
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "plan is skipped mid-turn");
    app.cycle_agent_mode().await; // review → default
    app.sending = false;

    // Leaving plan mid-turn defers the engine restore (badge flips now).
    app.plan_mode = true;
    app.sending = true;
    app.cycle_agent_mode().await;
    assert!(!app.plan_mode);
    assert!(app.agent_review_edits, "plan still cycles into review");
    assert!(app.plan_exit_pending, "engine restore deferred to turn end");
}

/// Shift+Tab on a permission card during plan mode allows that one call only —
/// it must NOT silently enable auto-approve (plan mode has no auto-approve).
#[tokio::test]
async fn test_permission_card_shift_tab_in_plan_mode_allows_once() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.plan_mode = true;
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(super::PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("cargo build".to_string()),
        reply,
    });
    let consumed = app.handle_permission_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
    assert!(consumed);
    assert_eq!(rx1.try_recv().unwrap(), Decision::Allow);
    assert!(app.plan_mode, "still planning");
    assert!(!app.agent_auto_approve, "auto-approve NOT enabled");
}

/// An `exit_plan_mode` tool call renders as the plan card (the plan is the
/// payload), not as an opaque `→ exit_plan_mode(…)` row.
#[test]
fn test_exit_plan_mode_call_renders_plan_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content:
            r#"{"name":"exit_plan_mode","args":{"plan":"1. refactor the gate\n2. add tests"}}"#
                .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let lines = app.build_transcript().lines;
    assert!(
        lines.iter().any(|l| l.plain == "Implementation plan"),
        "plan card header shown"
    );
    assert!(
        lines.iter().any(|l| l.plain.contains("refactor the gate")),
        "plan body shown"
    );
    assert!(
        !lines.iter().any(|l| l.plain.contains("exit_plan_mode")),
        "no raw tool-call row"
    );
}

/// The composer rule shows the persistent `◇ plan` indicator while plan mode is
/// on (and not while it's off), carries the cycle hint, and tints the rule ACCENT.
#[tokio::test]
async fn test_plan_badge_on_composer_rule() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plain = |line: &ratatui::text::Line<'_>| -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    };

    let off = app.composer_rule_line(80);
    assert!(!plain(&off).contains("plan"));
    assert!(plain(&off).contains("normal"), "normal mode shown");
    // Every mode carries the cycle hint.
    assert!(plain(&off).contains("(shift+tab)"));

    app.plan_mode = true;
    let on = app.composer_rule_line(80);
    assert!(plain(&on).contains("◇ plan"));
    assert!(plain(&on).contains("(shift+tab)"));
    // The rule dashes tint ACCENT in plan mode (FAINT otherwise).
    let dash_color = |line: &ratatui::text::Line<'_>| {
        line.spans
            .iter()
            .find(|s| s.content.contains('─'))
            .and_then(|s| s.style.fg)
    };
    assert_eq!(dash_color(&on), Some(ACCENT), "plan rule is accent-tinted");
    assert_eq!(dash_color(&off), Some(FAINT), "default rule is faint");
}

/// `/create-skill` is a first-class built-in command (in `SLASH_COMMANDS` and
/// `/help`), parses with an optional intent argument, and dispatches the embedded
/// create-skill instructions as a turn — shown compactly in the transcript.
#[tokio::test]
async fn test_create_skill_is_a_builtin_command() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // It's registered as a built-in command, so `/help` and the `/` menu list it.
    assert!(
        SLASH_COMMANDS.iter().any(|c| c.name == "create-skill"),
        "create-skill must be a built-in command"
    );
    // Parses bare and with an intent argument.
    assert_eq!(
        parse_slash_command("create-skill").unwrap(),
        SlashCommand::CreateSkill(None)
    );
    assert_eq!(
        parse_slash_command("create-skill a git-diff summarizer").unwrap(),
        SlashCommand::CreateSkill(Some("a git-diff summarizer".to_string()))
    );

    // Running it queues/sends the embedded instructions and shows the compact
    // command (not the inlined body) in the transcript.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_create_skill_command(Some("a git-diff summarizer".to_string()))
        .await
        .unwrap();
    let user = app
        .history
        .iter()
        .find(|m| m.role == "user")
        .expect("a user turn was dispatched");
    assert!(
        user.content.contains("create-skill") && user.content.contains("git-diff summarizer"),
        "the model receives the expanded instructions + intent"
    );
    // Regression: the body documents the literal `$ARGUMENTS` token, which must
    // survive verbatim — the command must NOT run it through `$ARGUMENTS`
    // substitution and splice the user's intent into the documentation.
    assert!(
        user.content.contains("$ARGUMENTS"),
        "the `$ARGUMENTS` doc token must not be substituted away"
    );

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("/create-skill"),
        "transcript should show the compact command:\n{screen}"
    );
}

/// Creating a subagent is natural-language only, by design: there is NO
/// `/create-agent` slash command (it would be redundant with the advertised
/// skill and clutter the menu). The workflow is instead exposed to the model as
/// a folderless built-in skill it reaches for on a request like "make a
/// code-reviewer subagent".
#[test]
fn test_create_agent_has_no_slash_command() {
    // Not registered as a typeable command — absent from the menu/help and unknown
    // to the parser.
    assert!(
        !SLASH_COMMANDS.iter().any(|c| c.name == "create-agent"),
        "create-agent must NOT be a slash command — it's natural-language only"
    );
    assert!(
        parse_slash_command("create-agent").is_err(),
        "typing /create-agent is an unknown command, not a builtin"
    );

    // The workflow still exists as a model-facing builtin skill (this is what the
    // send path injects into the engine's skill list to advertise it).
    let sc = crate::agent::skills::create_agent_builtin();
    assert_eq!(sc.name, "create-agent");
    assert!(!sc.body.is_empty());
}

/// `/agents`: registered as a typeable command; bare opens the overlay, `rm`
/// on an unknown name reports instead of erroring, and anything else prints
/// usage (there is no `add` — creation is conversational by design).
#[tokio::test]
async fn test_agents_command_opens_overlay_and_validates_args() {
    assert!(SLASH_COMMANDS.iter().any(|c| c.name == "agents"));
    assert!(matches!(
        parse_slash_command("agents"),
        Ok(SlashCommand::Agents(None))
    ));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_agents_command(None).await.unwrap();
    assert!(matches!(app.overlay, Overlay::Agents(_)));

    app.overlay = Overlay::None;
    app.run_agents_command(Some("rm no-such-agent".to_string()))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    let notice = app.notice.as_ref().expect("notice set").1.clone();
    assert!(notice.contains("no-such-agent"), "{notice}");

    app.run_agents_command(Some("add reviewer".to_string()))
        .await
        .unwrap();
    let notice = app.notice.as_ref().expect("usage notice").1.clone();
    assert!(notice.contains("Usage: /agents"), "{notice}");

    // Built-ins can't be removed — the notice points at shadowing instead.
    app.run_agents_command(Some("rm explorer".to_string()))
        .await
        .unwrap();
    let notice = app.notice.as_ref().expect("builtin notice").1.clone();
    assert!(notice.contains("built into aivo"), "{notice}");
}

/// The `/agents` empty state reads intact on a narrow terminal: the body clips
/// rather than wraps, so every line must be pre-wrapped short enough (~40 cols)
/// — the quoted example must survive whole, not as "make me a cod".
#[test]
fn test_agents_overlay_empty_state_fits_narrow_terminals() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Agents(AgentsOverlay::default());
    let (top, _) = render_full_screen(&mut app, 46, 18);
    // Keep only the modal interior (between the │ borders), then collapse
    // whitespace: wrapping may split lines, but every WORD must survive whole.
    let interior: String = top
        .lines()
        .filter_map(|row| {
            let first = row.find('\u{2502}')?;
            let last = row.rfind('\u{2502}')?;
            (last > first).then(|| row[first + '\u{2502}'.len_utf8()..last].to_string())
        })
        .collect::<Vec<_>>()
        .join(" ");
    let flat = interior.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains("No sub-agents yet"), "{top}");
    assert!(
        flat.contains("\u{201c}make me a code-reviewer subagent\u{201d}"),
        "quoted example clipped:\n{top}"
    );
    assert!(flat.contains("or drop a <name>.md profile in:"), "{top}");
    assert!(flat.contains("~/.config/aivo/agents"), "{top}");
}

/// The `@` mention menu: word-boundary trigger (mid-message ok, emails no),
/// prefix-first filtering over discovered profiles, and completion that inserts
/// `@name ` at the token without submitting the draft.
#[test]
fn test_at_mention_menu_completes_subagent_names() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let profile = |name: &str| crate::agent::subagents::Subagent {
        name: name.to_string(),
        description: format!("{name} does things. Extra sentence."),
        model: None,
        tools: None,
        body: String::new(),
        isolation_worktree: false,
        repo_local: false,
        source: std::path::PathBuf::new(),
    };

    // No discovered profiles → no menu, even on a bare `@`.
    app.draft = "@".to_string();
    app.cursor = 1;
    assert!(app.active_mention_query().is_none());

    app.last_subagents = vec![profile("code-reviewer"), profile("architect")];

    // Mid-message mention at a word boundary; query is the partial after `@`.
    app.draft = "use @co on the diff".to_string();
    app.cursor = 7; // after "use @co"
    let (at, query) = app.active_mention_query().expect("mention active");
    assert_eq!((at, query.as_str()), (4, "co"));
    let menu = app.visible_command_menu().expect("menu visible");
    assert!(matches!(menu.kind, MenuKind::Mention));
    assert_eq!(menu.entries.len(), 1);
    assert_eq!(menu.entries[0].label(), "@code-reviewer");

    // Tab completion replaces just the token and keeps composing (no submit).
    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "use @code-reviewer  on the diff");
    assert_eq!(app.cursor, "use @code-reviewer ".len());

    // A bare `@` lists every profile.
    app.draft = "@".to_string();
    app.cursor = 1;
    app.command_menu.reset();
    let menu = app.visible_command_menu().expect("menu visible");
    assert_eq!(menu.entries.len(), 2);

    // No word boundary (email-style) → no menu; ditto once the token has a space.
    app.draft = "mail me a@b".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());
    app.draft = "@code-reviewer go".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());

    // `/attach` path mode wins over mention parsing…
    app.draft = "/attach @x".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());
    // …but a mention inside an ordinary command ARGUMENT works (`/goal`, `/plan`
    // steering a named agent is a legit composition).
    app.draft = "/goal ship it with @arch".to_string();
    app.cursor = app.draft.len();
    let (_, query) = app.active_mention_query().expect("mention in command arg");
    assert_eq!(query, "arch");
}

/// `/goal` status/stop and the start guards — none of which send a turn.
#[tokio::test]
async fn test_goal_command_status_stop_and_guards() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Bare /goal with nothing active → usage notice, starts nothing.
    app.run_goal_command(None).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // stop when inactive → notice, still inactive.
    app.run_goal_command(Some("stop".to_string())).await;
    assert!(app.goal_mode.is_none());

    // Starting mid-turn queues the command for turn end (no send yet).
    app.sending = true;
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert_eq!(
        app.queued_commands,
        vec![SlashCommand::Goal(Some("do it".to_string()))]
    );
    assert!(app.notice.as_ref().unwrap().1.contains("queued"));
    app.sending = false;
    app.queued_commands.clear();

    // A non-agent key (OAuth) is refused (no send).
    app.key.base_url = "claude-oauth".to_string();
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("native agent"));
}

/// Goal mode sends machine text to the model but must not leak it into ↑/↓
/// recall — `record: None` records nothing; only the typed `/goal <obj>` does.
#[tokio::test]
async fn test_goal_machine_text_never_enters_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.dispatch_user_message("[Goal mode] preamble".to_string(), None)
        .await
        .unwrap();
    assert!(app.draft_history.is_empty());

    app.dispatch_user_message(
        "[Goal mode] preamble".to_string(),
        Some("/goal x".to_string()),
    )
    .await
    .unwrap();
    assert_eq!(app.draft_history, vec!["/goal x".to_string()]);
}

/// The goal loop ends on the completion marker and at the turn cap (both
/// terminal — neither sends another turn). Exercises `signals_goal_complete`.
#[tokio::test]
async fn test_goal_loop_stops_on_marker_and_cap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let assistant = |content: &str| ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    // Marker on its own line → loop ends, complete notice.
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
    });
    app.history.push(assistant("all set.\nGOAL COMPLETE"));
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none(), "marker ends the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("complete"));

    // Prose merely mentioning the marker does NOT count (strict whole-line match).
    app.history.clear();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 20,
        max: 20,
    });
    app.history.push(assistant(
        "I will reply GOAL COMPLETE once everything is finished.",
    ));
    // At the cap with no real marker → loop ends with the cap notice (no send).
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none(), "cap ends the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("cap"));

    // Not in goal mode → no-op.
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none());
}

/// The marker counts when markdown-wrapped (models echo the prompts' backticks);
/// prose around the words still doesn't.
#[tokio::test]
async fn test_goal_marker_tolerates_markdown_wrapping() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let assistant = |content: &str| ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    for reply in [
        "`GOAL COMPLETE`",
        "**GOAL COMPLETE**",
        "All tests pass.\n\ngoal complete.",
        "done\n`GOAL COMPLETE.`",
    ] {
        app.history.clear();
        app.history.push(assistant(reply));
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 2,
            max: 20,
        });
        app.maybe_continue_goal().await.unwrap();
        assert!(app.goal_mode.is_none(), "marker ends the loop: {reply:?}");
    }

    // `sending` blocks the continuation dispatch so only the check runs.
    for reply in [
        "I will reply GOAL COMPLETE once everything is finished.",
        "- [ ] GOAL COMPLETE",
    ] {
        app.history.clear();
        app.history.push(assistant(reply));
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 2,
            max: 20,
        });
        app.sending = true;
        app.maybe_continue_goal().await.unwrap();
        assert!(
            app.goal_mode.is_some(),
            "prose must not end the loop: {reply:?}"
        );
        app.sending = false;
    }
}

/// An errored turn stops the loop instead of replaying to the cap, and the
/// error notice stays visible.
#[tokio::test]
async fn test_goal_stops_on_errored_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "partial work".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.notice = Some((ERROR, "LLM error: insufficient credits".to_string()));

    app.maybe_continue_goal().await.unwrap();

    assert!(app.goal_mode.is_none(), "an errored turn ends the loop");
    assert!(!app.sending, "no continuation was sent");
    let (style, msg) = app.notice.clone().unwrap();
    assert_eq!(style, ERROR);
    assert!(msg.contains("insufficient credits"), "error kept: {msg}");
    assert!(msg.contains("goal mode stopped"), "stop noted: {msg}");
}

/// In `/goal` mode a single Esc only arms a confirm — the loop keeps running;
/// a second consecutive Esc interrupts and stops it.
#[tokio::test]
async fn test_goal_esc_requires_confirmation_to_stop() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
    });
    app.sending = true;

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.goal_stop_confirm_pending, "first Esc arms the confirm");
    assert!(app.goal_mode.is_some(), "loop still armed after one Esc");
    assert!(app.sending, "turn not interrupted by one Esc");
    assert!(app.notice.is_some());

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.goal_stop_confirm_pending);
    assert!(app.goal_mode.is_none(), "second Esc stops the loop");
    assert!(!app.sending, "second Esc interrupts the turn");
}

/// A non-Esc key between the two Esc presses disarms the confirm, so the loop
/// keeps running and the next Esc re-arms rather than stopping.
#[tokio::test]
async fn test_goal_esc_confirm_resets_on_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
    });
    app.sending = true;

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.goal_stop_confirm_pending);

    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.goal_stop_confirm_pending,
        "other key disarms the confirm"
    );
    assert!(app.notice.is_none(), "confirm notice cleared");
    assert!(app.goal_mode.is_some(), "loop untouched");
    assert!(app.sending);
}

/// The goal turn's marker ends the loop even when a queued user message already
/// started the next turn — otherwise the completed goal burns an extra round.
#[tokio::test]
async fn test_goal_completion_detected_while_queued_message_runs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 2,
        max: 20,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done\nGOAL COMPLETE".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // The queued message's user turn is already in flight.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "also rename the module".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;

    app.maybe_continue_goal().await.unwrap();

    assert!(app.goal_mode.is_none(), "marker ends the loop mid-queue");
    assert!(app.notice.as_ref().unwrap().1.contains("Goal complete"));
}

/// The synthetic continuation must not consume a draft typed mid-turn; a
/// plain-chat route also disarms the loop (its finish never auto-continues).
#[tokio::test]
async fn test_goal_continuation_preserves_composer_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.draft = "half-typed reply".to_string();
    app.cursor = 4;

    app.maybe_continue_goal().await.unwrap();

    assert_eq!(app.draft, "half-typed reply", "draft survives the dispatch");
    assert_eq!(app.cursor, 4, "cursor survives the dispatch");
    let last = app.history.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(last.content, "/goal — continue");
    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.starts_with("Continue toward the goal"),
        "the continuation still went out: {}",
        sent.content
    );
    assert!(app.goal_mode.is_none(), "plain-chat route disarms the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("plain chat"));
}

/// A guard-stopped turn steers the next `/goal` continuation: the message tells the
/// model not to retry the dead end, and the stored guard-stop is consumed.
#[tokio::test]
async fn test_goal_guard_stop_enriches_continuation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = "claude-oauth".to_string();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
    });
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::NoProgress);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.maybe_continue_goal().await.unwrap();

    let last = app.history.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(
        last.content, "/goal — continue",
        "compact transcript marker"
    );
    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.starts_with("[Previous turn stopped early:"),
        "guard-stop should enrich the continuation: {}",
        sent.content
    );
    assert!(
        sent.content.contains("Continue toward the goal"),
        "the base continuation still rides along: {}",
        sent.content
    );
    assert!(
        app.goal_guard_stop.is_none(),
        "the guard-stop is consumed, not resent next turn"
    );
}

/// A step-limit stop — which the old text-match silently ignored — also steers the
/// continuation, telling the model to break the work into smaller pieces.
#[tokio::test]
async fn test_goal_step_limit_steers_continuation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = "claude-oauth".to_string();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
    });
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::StepLimit);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.maybe_continue_goal().await.unwrap();

    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.contains("ran out of steps"),
        "step-limit gets its own steering: {}",
        sent.content
    );
}

/// Starting a fresh `/goal` clears any stale guard-stop from a prior loop.
#[tokio::test]
async fn test_goal_guard_stop_cleared_on_new_goal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Force the plain-chat route (image in history, vision unknown) to keep the
    // dispatch lightweight; the guard-stop clear happens before dispatch either way.
    app.model_image_input = None;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::ToolFailureLoop);

    app.run_goal_command(Some("do the thing".to_string())).await;

    assert!(
        app.goal_guard_stop.is_none(),
        "a fresh goal must not carry a stale guard-stop"
    );
}

/// A running background job shows a `✦ N job(s)` badge in the composer rule.
#[cfg(unix)]
#[tokio::test]
async fn jobs_badge_shows_running_count() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let cwd = std::env::temp_dir();
    app.jobs.spawn("sleep 30", &cwd).unwrap();
    // The event loop caches this each tick; refresh it directly for the test.
    app.jobs_running = app.jobs.running_count();
    let line = app.composer_rule_line(120);
    let plain = plain_text_from_spans(&line.spans);
    assert!(plain.contains("✦ 1 job"), "badge missing: {plain}");
    let _ = app.jobs.kill_all().await;
}

/// With no running jobs, the composer rule carries no jobs badge (guards width math).
#[test]
fn jobs_badge_absent_when_idle() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let line = app.composer_rule_line(120);
    let plain = plain_text_from_spans(&line.spans);
    assert!(!plain.contains("✦"), "no jobs → no badge: {plain}");
}

/// `/new` re-roots NEW background-job logs under the fresh session's artifacts dir.
#[cfg(unix)]
#[tokio::test]
async fn new_session_reroots_job_logs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.start_new_chat();
    let new_id = app.session_id.clone();
    assert!(!new_id.is_empty(), "a new session id was minted");
    let out = app.jobs.spawn("echo hi", &std::env::temp_dir()).unwrap();
    assert!(
        out.contains(&new_id),
        "job log should be under the new session dir: {out}"
    );
    let _ = app.jobs.kill_all().await;
}

/// `/goal` where dispatch falls back to plain chat (image in history, vision
/// unconfirmed) must not arm a loop `finish_response` will never drive.
#[tokio::test]
async fn test_goal_start_disarms_on_plain_chat_route() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = None; // vision support unknown → plain-chat fallback
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });

    app.run_goal_command(Some("fix it".to_string())).await;

    assert!(
        app.goal_mode.is_none(),
        "plain-chat route must not arm a loop"
    );
    assert!(
        app.sending,
        "the objective still went out as a plain message"
    );
    assert!(app.notice.as_ref().unwrap().1.contains("plain chat"));
}

/// `/goal` whose first dispatch is refused outright (staged image on a known
/// text-only model) must not leave an armed loop with nothing in flight.
#[tokio::test]
async fn test_goal_start_cleared_when_dispatch_refuses() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false); // snapshot says text-only
    app.draft_attachments.push(MessageAttachment {
        name: "shot.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    });

    app.run_goal_command(Some("describe the screenshot".to_string()))
        .await;

    assert!(
        app.goal_mode.is_none(),
        "refused send must not arm the loop"
    );
    assert!(!app.sending, "nothing went out");
    assert!(app.notice.as_ref().unwrap().1.contains("can't read images"));
    assert_eq!(app.draft_attachments.len(), 1, "attachment kept for resend");
}

/// Entering plan mode stops an active goal loop (mirrors `/goal`'s refusal to
/// start while planning).
#[tokio::test]
async fn test_plan_entry_stops_goal_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "ship".to_string(),
        iteration: 3,
        max: 20,
    });
    // Image in history + unknown vision pins the kick-off to plain chat — an
    // agent engine build would touch real config/git.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });
    app.model_image_input = None;
    app.run_plan_command(None).await;

    assert!(app.plan_mode, "plan mode is on");
    assert!(app.goal_mode.is_none(), "plan entry ends the goal loop");
    assert!(app.sending, "the kick-off turn went out");
    assert!(app.notice.as_ref().unwrap().1.contains("Goal mode stopped"));
}

/// A cancelled or interrupted `/compact` must not mark the NEXT turn as a
/// compact (bogus "freed" notice, skipped logs, corrupted context stats).
#[tokio::test]
async fn test_teardown_clears_compact_before() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // ESC / /new / resume / key-switch route (cancel).
    app.sending = true;
    app.compact_before = Some(50_000);
    app.cancel_inflight_request(CancelKind::Discard);
    assert_eq!(app.compact_before, None, "cancel clears the compact flag");

    // Interrupt-with-partial-text route (skips cancel_inflight_request).
    app.sending = true;
    app.compact_before = Some(50_000);
    app.pending_response = "partial".to_string();
    app.interrupt_inflight_request().await.unwrap();
    assert_eq!(
        app.compact_before, None,
        "interrupt clears the compact flag"
    );
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

    app.rewind_to_turn(0, None).await.unwrap();

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
    });

    let session = LoadedSession {
        key_id: app.key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![],
        engine_messages: None,
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

/// Mid-turn tool-set changes (skill/MCP toggles, async skill installs) defer the
/// engine drop to turn end — an immediate drop loses the turn's usage + transcript.
#[test]
fn test_request_engine_rebuild_defers_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.sending = true;
    app.request_engine_rebuild();
    assert!(
        app.engine_rebuild_pending,
        "deferred while a turn is in flight"
    );

    app.sending = false;
    app.maybe_apply_engine_rebuild();
    assert!(!app.engine_rebuild_pending, "applied at turn end");

    // Idle: applies immediately, no pending flag.
    app.request_engine_rebuild();
    assert!(!app.engine_rebuild_pending);
}

/// A level not offered by the current model is refused at apply time — a stale
/// effort picker across an agent-driven model switch must not 400 later turns.
#[tokio::test]
async fn test_apply_reasoning_effort_rejects_foreign_level() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "m".to_string();
    app.model_reasoning_efforts = vec!["low".to_string(), "high".to_string()];

    app.apply_reasoning_effort("xhigh".to_string()).await;
    assert!(app.reasoning_effort.is_none(), "foreign level refused");
    assert!(app.notice.as_ref().unwrap().1.contains("isn't a level"));

    app.apply_reasoning_effort("high".to_string()).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
}

/// Draining a queued message records nothing — every queue site already recorded
/// the recallable form, and a queued skill's expanded body must not enter ↑/↓.
#[tokio::test]
async fn test_drained_queued_message_not_recorded_in_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.queued_messages
        .push("Use the \"x\" skill. Follow these instructions:\n\n…pages…".to_string());
    app.drain_queued_message().await.unwrap();

    assert!(
        app.draft_history.is_empty(),
        "expanded machine text must not enter recall"
    );
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content.starts_with("Use the")),
        "the queued message still went out"
    );
}

#[tokio::test]
async fn test_mid_turn_message_steers_then_reclaims_or_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.agent_serve = Some((
        tokio::spawn(async { Ok(()) }),
        std::sync::Arc::new(tokio::sync::Notify::new()),
    ));

    app.draft = "actually use tabs".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();
    assert!(
        app.queued_messages.is_empty(),
        "engine turns steer, not queue"
    );
    {
        let steering = app.steering_queue.lock().unwrap();
        assert_eq!(steering.as_slice(), ["actually use tabs".to_string()]);
    }

    app.reclaim_unsent_steering();
    assert_eq!(app.queued_messages, vec!["actually use tabs".to_string()]);
    assert!(app.steering_queue.lock().unwrap().is_empty());

    app.apply_agent_steered("also add a test".to_string());
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "also add a test")
    );
}

/// `/copy 0` is a usage error and out-of-range names the real count, instead of
/// claiming no reply exists.
#[tokio::test]
async fn test_copy_rejects_zero_and_reports_range() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "only reply".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let err = app.copy_reply_to_clipboard(Some(0)).unwrap_err();
    assert!(err.to_string().contains("Usage"), "{err}");

    let err = app.copy_reply_to_clipboard(Some(5)).unwrap_err();
    assert!(err.to_string().contains("Only 1 reply"), "{err}");
}

/// `/plan go` sends machine text — it must not swallow a draft or staged
/// attachment the user prepared mid-planning (same treatment as `/goal`).
#[tokio::test]
async fn test_plan_go_preserves_composer_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();
    app.pending_plan = Some("the plan".to_string());
    app.draft = "note to self".to_string();
    app.cursor = 4;

    app.run_plan_command(Some("go".to_string())).await;

    assert!(app.sending, "the go message went out");
    assert_eq!(app.draft, "note to self", "draft survives the dispatch");
    assert_eq!(app.cursor, 4);
}

/// Bare `/plan` enters the mode AND dispatches the kick-off turn: the model
/// gets the machine text, the transcript shows the compact `/plan`, and
/// neither a composer draft nor ↑ recall picks it up.
#[tokio::test]
async fn test_plan_bare_dispatches_kickoff() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Image in history + unknown vision pins the kick-off to plain chat — an
    // agent engine build would touch real config/git.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });
    app.model_image_input = None;
    app.draft = "note to self".to_string();
    app.cursor = 4;

    app.run_plan_command(None).await;

    assert!(app.plan_mode, "bare /plan enters the mode");
    assert!(app.sending, "the kick-off went out");
    assert_eq!(
        app.pending_submit.as_ref().unwrap().content,
        super::runtime_impl::PLAN_KICKOFF_MESSAGE,
        "the model receives the interview instructions"
    );
    assert_eq!(
        app.history.last().unwrap().content,
        "/plan",
        "the transcript shows the compact command, not the machine text"
    );
    assert_eq!(app.draft, "note to self", "draft survives the dispatch");
    assert_eq!(app.cursor, 4);
    assert!(
        app.draft_history.is_empty(),
        "machine text never enters ↑ recall"
    );
}

/// A non-UTF8 file under a text mime (unknown extension) is refused with a clear
/// error instead of being sent as a base64 blob labeled text/plain.
#[tokio::test]
async fn test_dispatch_refuses_binary_text_attachment() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("blob.bin");
    std::fs::write(&path, [0xffu8, 0xfe, 0x01, 0x00]).unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "blob.bin".to_string(),
        mime_type: "text/plain".to_string(),
        storage: AttachmentStorage::FileRef {
            path: path.to_string_lossy().into_owned(),
        },
    });

    let err = app
        .dispatch_user_message("look at this".to_string(), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("binary"), "{err}");
}

/// The composer rule shows a live `/goal` step indicator (and the auto-approve
/// badge) while a goal loop runs, nothing goal-related when off, within width.
#[test]
fn test_composer_rule_shows_goal_step_indicator() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let off = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!off.contains("goal"), "no goal badge when off: {off:?}");
    assert!(off.contains("normal"), "mode badge shows: {off:?}");

    app.goal_mode = Some(GoalState {
        objective: "ship it".to_string(),
        iteration: 2,
        max: 20,
    });
    let on = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(on.contains("goal 2/20"), "goal step indicator: {on:?}");
    assert!(on.contains("normal"), "mode badge stays: {on:?}");
    assert!(
        display_width(&on) <= 80,
        "rule fits width: {}",
        display_width(&on)
    );
}

/// While recalling input history with up-arrow, the composer rule titles itself
/// `History pos/total` (newest is total/total, counting down going back) and
/// drops the title once navigation ends.
#[tokio::test]
async fn test_composer_rule_shows_history_position_while_navigating() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = (0..100).map(|i| format!("/cmd {i}")).collect();

    // Idle: no history title, just the auto-approve badge.
    let idle = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!idle.contains("History"), "no title when idle: {idle:?}");

    // Up-arrow into the newest entry → `History 100/100`.
    app.history_prev();
    let newest = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(newest.contains("History 100/100"), "newest: {newest:?}");
    assert!(
        display_width(&newest) <= 80,
        "rule fits width: {}",
        display_width(&newest)
    );

    // One step further back → `History 99/100`.
    app.history_prev();
    let back = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(back.contains("History 99/100"), "back one: {back:?}");

    // Leaving navigation drops the title again.
    app.leave_history_navigation();
    let done = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!done.contains("History"), "title gone after exit: {done:?}");
}

/// The input/draft history is bounded to the most recent `MAX_DRAFT_HISTORY`
/// entries, dropping the oldest first so the on-disk recall file can't grow
/// without bound.
#[tokio::test]
async fn test_draft_history_capped_at_max() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Record 50 over the cap; each entry is distinct so none dedupe away.
    for i in 0..(MAX_DRAFT_HISTORY + 50) {
        app.record_draft_history(&format!("/cmd {i}"));
    }
    assert_eq!(app.draft_history.len(), MAX_DRAFT_HISTORY);
    // The 50 oldest were dropped; the newest is retained.
    assert_eq!(app.draft_history.first().unwrap(), "/cmd 50");
    assert_eq!(
        app.draft_history.last().unwrap(),
        &format!("/cmd {}", MAX_DRAFT_HISTORY + 49)
    );
}

#[tokio::test]
async fn test_skills_add_routes_source_not_scaffold() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A token that isn't a bare skill name (has `/` or `.`) routes to install-
    // from-source, not scaffold. A bad local path surfaces an install error
    // (and never scaffolds a literal `./…`-named skill).
    app.submit_skill_add("./aivo_no_such_skill_dir_zzz".to_string())
        .await
        .unwrap();
    // Install runs on a background task now; drain its `SkillInstalled` outcome.
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if app.installing_skill.is_none() && app.notice.is_some() {
            break;
        }
        tokio::task::yield_now().await;
    }
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(
        notice.contains("Failed to install") || notice.contains("not a directory"),
        "expected an install error, got: {notice}"
    );
}

/// A local two-skill install source (`skills/alpha`, `skills/beta`) in a tempdir.
fn write_skill_pack() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let pack = std::env::temp_dir().join(format!(
        "aivo-tui-skill-pack-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    for (name, desc) in [("alpha", "First skill."), ("beta", "Second skill.")] {
        let dir = pack.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {desc}\n---\nBody of {name}.\n"),
        )
        .unwrap();
    }
    pack
}

/// Phase verb + live size readout; the size precedes the source so a clipped
/// URL never hides it.
#[test]
fn test_skill_install_progress_status_text() {
    let progress = SkillInstallProgress::new("github:o/r".to_string(), "Fetching");
    assert_eq!(progress.status_text(), "Fetching github:o/r…");
    progress
        .bytes
        .store(2_621_440, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(progress.status_text(), "Fetching (2.5MB) github:o/r…");
    let copy = SkillInstallProgress::new("github:o/r".to_string(), "Installing");
    assert_eq!(copy.status_text(), "Installing github:o/r…");
}

/// The `/skills` overlay carries the progress row; the transcript line is
/// suppressed while it does.
#[test]
fn test_skills_overlay_shows_fetch_progress_row() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let progress = SkillInstallProgress::new("github:anthropics/skills".to_string(), "Fetching");
    progress
        .bytes
        .store(1_048_576, std::sync::atomic::Ordering::Relaxed);
    app.installing_skill = Some(progress);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Fetching (1.0MB)"),
        "missing progress row with size:\n{screen}"
    );
    assert!(
        app.spinner_status_line().is_none(),
        "status line must be suppressed while /skills shows the progress row"
    );

    app.overlay = Overlay::None;
    let line = app
        .spinner_status_line()
        .expect("status line while fetching");
    let text = plain_text_from_spans(&line.line.spans);
    assert!(
        text.contains("Fetching (1.0MB) github:anthropics/skills"),
        "status line: {text:?}"
    );
}

/// Picker chrome: title, marked badge, source line, checkbox rows, footer.
#[test]
fn test_skill_install_overlay_renders_picker() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:anthropics/skills".to_string(),
        project: false,
        items: vec![
            InstallPickItem {
                name: "alpha".to_string(),
                description: "First skill.".to_string(),
                body: "Body.".to_string(),
                checked: true,
                installed: false,
            },
            InstallPickItem {
                name: "beta".to_string(),
                description: "Second skill.".to_string(),
                body: "Body.".to_string(),
                checked: false,
                installed: true,
            },
        ],
        selected: 0,
        query: String::new(),
        viewing: None,
        detail_scroll: 0,
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Install skills"),
        "missing title:\n{screen}"
    );
    assert!(
        screen.contains("from github:anthropics/skills"),
        "missing source line:\n{screen}"
    );
    assert!(
        screen.contains("into ~/.config/aivo/skills (user)"),
        "missing destination line:\n{screen}"
    );
    assert!(screen.contains("alpha"), "missing skill row:\n{screen}");
    assert!(
        screen.contains("First skill."),
        "missing description:\n{screen}"
    );
    assert!(screen.contains("1/2 marked"), "missing badge:\n{screen}");
    assert!(
        screen.contains("installed — Space to update"),
        "missing installed/update note:\n{screen}"
    );
    // Footer clips in narrow terminals; the mark/install/Esc trio comes first.
    assert!(
        screen.contains("mark") && screen.contains("install") && screen.contains("Esc"),
        "missing footer controls:\n{screen}"
    );
    assert!(
        screen.contains("Enter applies the 1 marked"),
        "missing marked-count detail:\n{screen}"
    );
}

/// A bracketed paste lands in the open overlay's text input (add field first,
/// else filter) instead of being dropped.
#[tokio::test]
async fn test_paste_routes_into_overlay_inputs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some("".to_string());
    app.overlay = Overlay::Skills(overlay);
    assert!(app.overlay_paste("github:anthropics/skills\n"));
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("github:anthropics/skills"));
    } else {
        panic!("skills overlay vanished");
    }

    let mut overlay = skills_overlay_fixture();
    overlay.adding = None;
    app.overlay = Overlay::Skills(overlay);
    assert!(app.overlay_paste("brand"));
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.query, "brand");
    } else {
        panic!("skills overlay vanished");
    }

    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:o/r".to_string(),
        ..Default::default()
    });
    assert!(app.overlay_paste("pdf"));
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert_eq!(state.query, "pdf");
    } else {
        panic!("install picker vanished");
    }

    app.overlay = Overlay::None;
    assert!(!app.overlay_paste("plain text"));
}

/// The loading state narrates the fetch; Esc from it returns to the composer.
#[tokio::test]
async fn test_skill_install_loading_state_renders_and_esc_closes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.installing_skill = Some(SkillInstallProgress::new(
        "github:anthropics/skills".to_string(),
        "Fetching",
    ));
    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:anthropics/skills".to_string(),
        ..Default::default()
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Install skills"),
        "missing modal title:\n{screen}"
    );
    assert!(
        screen.contains("Fetching github:anthropics/skills"),
        "missing loading row:\n{screen}"
    );
    assert!(
        screen.contains("will appear here"),
        "missing loading hint:\n{screen}"
    );
    assert!(
        app.spinner_status_line().is_none(),
        "transcript line must stay quiet while the modal narrates"
    );

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        matches!(app.overlay, Overlay::None),
        "Esc on loading must close the modal, not open /skills"
    );
}

/// A mark on an installed row is an update ask.
#[test]
fn test_skill_install_picker_marks_installed_for_update() {
    let mut state = SkillInstallOverlay {
        source: "github:o/r".to_string(),
        project: false,
        items: vec![
            InstallPickItem {
                name: "fresh".to_string(),
                description: String::new(),
                body: String::new(),
                checked: false,
                installed: false,
            },
            InstallPickItem {
                name: "have".to_string(),
                description: String::new(),
                body: String::new(),
                checked: false,
                installed: true,
            },
        ],
        selected: 1,
        query: String::new(),
        viewing: None,
        detail_scroll: 0,
    };
    // Enter's fallback never implicitly updates an installed row.
    assert!(state.pick_names().is_empty());
    state.items[1].checked = true;
    assert_eq!(state.pick_names(), ["have"]);
    // Mark-all targets only the not-yet-installed rows.
    state.items[1].checked = false;
    state.toggle_all();
    assert!(state.items[0].checked && !state.items[1].checked);
}

/// Multi-skill source: loading modal at once, picker when staged, marking keys,
/// Esc discards the stage.
#[tokio::test]
async fn test_skills_install_multi_source_opens_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(pack.display().to_string())
        .await
        .unwrap();
    // Loading modal opens at once — never the installed-skills list.
    assert!(
        matches!(&app.overlay, Overlay::SkillInstall(s) if s.items.is_empty()),
        "submit must open the install modal right away, before the fetch completes"
    );
    assert!(
        app.installing_skill.is_some(),
        "progress state set from the first frame"
    );
    // The loading modal is already SkillInstall — wait for the staged items.
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if matches!(&app.overlay, Overlay::SkillInstall(s) if !s.items.is_empty()) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let Overlay::SkillInstall(state) = &app.overlay else {
        panic!(
            "multi-skill source must open the install picker: {:?}",
            app.notice
        );
    };
    assert!(
        !state.items.is_empty(),
        "picker never populated: {:?}",
        app.notice
    );
    let names: Vec<&str> = state.items.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(names, ["alpha", "beta"]);
    assert!(state.items.iter().all(|i| !i.checked), "nothing pre-marked");
    assert_eq!(state.source, pack.display().to_string());
    assert!(
        app.staged_skill_install.is_some(),
        "the fetched tree stays staged for the pick"
    );
    assert!(app.installing_skill.is_none(), "spinner cleared");
    assert_eq!(state.items[0].description, "First skill.");
    assert_eq!(state.items[0].body, "Body of alpha.");
    // Nothing marked: Enter targets the highlighted row.
    assert_eq!(state.pick_names(), ["alpha"]);

    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert!(state.items[0].checked, "Space marks");
        assert_eq!(state.pick_names(), ["alpha"]);
    } else {
        panic!("picker vanished on Space");
    }
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert!(
            state.items.iter().all(|i| i.checked),
            "Ctrl+A marks all when any is unmarked"
        );
        assert_eq!(state.pick_names(), ["alpha", "beta"]);
    } else {
        panic!("picker vanished on Ctrl+A");
    }

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.staged_skill_install.is_none(),
        "Esc must discard the staged tree"
    );
    assert!(
        matches!(app.overlay, Overlay::Skills(_)),
        "Esc falls back to the /skills overlay"
    );
    // A local source is never deleted by the discard.
    assert!(pack.join("skills/alpha/SKILL.md").is_file());
    let _ = std::fs::remove_dir_all(&pack);
}

/// `-p` rides the whole install pipeline: flag parse → staged pick → overlay
/// marked as a project install (nothing is written until names are picked).
#[tokio::test]
async fn test_skills_install_project_flag_reaches_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(format!("-p {}", pack.display()))
        .await
        .unwrap();
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if matches!(&app.overlay, Overlay::SkillInstall(s) if !s.items.is_empty()) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let Overlay::SkillInstall(state) = &app.overlay else {
        panic!("picker must open: {:?}", app.notice);
    };
    assert!(state.project, "picker must carry the -p destination");
    assert_eq!(state.source, pack.display().to_string(), "flag stripped");
    assert!(
        matches!(app.staged_skill_install, Some((_, true))),
        "staged pick must remember the project destination"
    );

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    let _ = std::fs::remove_dir_all(&pack);
}

/// Notice wording per report shape.
#[test]
fn test_install_report_notice_wording() {
    use super::session_impl::install_report_notice;
    use crate::agent::skills::InstallReport;
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec![],
            skipped_existing: vec![],
        },
    );
    assert_eq!(msg, "Installed skill: a");
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into(), "b".into()],
            updated: vec![],
            skipped_existing: vec!["c".into()],
        },
    );
    assert_eq!(msg, "Installed skills: a, b (already installed: c)");
    let (color, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec![],
            updated: vec![],
            skipped_existing: vec!["c".into()],
        },
    );
    assert_eq!(msg, "Already installed: c");
    assert_eq!(color, WARNING);
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec!["u".into(), "v".into()],
            skipped_existing: vec![],
        },
    );
    assert_eq!(msg, "Installed skill: a · Updated skills: u, v");
    // `-p/--project`: destination and the untrusted caveat are spelled out.
    let (_, msg) = install_report_notice(
        "src",
        true,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec![],
            skipped_existing: vec![],
        },
    );
    assert!(
        msg.starts_with("Installed skill: a → ./.agents/skills"),
        "{msg}"
    );
    assert!(msg.contains("untrusted"), "{msg}");
    let (color, _) = install_report_notice("src", false, &InstallReport::default());
    assert_eq!(color, WARNING);
}

/// A flag-like token other than `-p/--project` is rejected up front — no
/// download for the install path, no folder named `--foo` for the scaffold path.
#[tokio::test]
async fn test_skill_add_rejects_unknown_options_before_fetch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.submit_skill_add("--foo github:o/r".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `--foo`")),
        "unknown leading flag must error: {:?}",
        app.notice
    );
    assert!(app.installing_skill.is_none(), "no fetch may start");

    // A bad filter used to surface only after the whole source was downloaded.
    app.submit_skill_add("github:o/r --bogus".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `--bogus`")),
        "unknown filter flag must error before the fetch: {:?}",
        app.notice
    );
    assert!(app.installing_skill.is_none(), "no fetch may start");

    // Mid-line `-p` is guided to the edges rather than silently treated as a name.
    app.submit_skill_add("github:o/r -p alpha".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `-p`")),
        "mid-line -p must point at the start/end rule: {:?}",
        app.notice
    );
}

/// `-p`/`--project` is recognized at either edge of the add line, and only there.
#[test]
fn test_split_project_flag() {
    use super::session_impl::split_project_flag;
    assert_eq!(
        split_project_flag("-p github:o/r"),
        ("github:o/r".to_string(), true)
    );
    assert_eq!(
        split_project_flag("github:o/r --project"),
        ("github:o/r".to_string(), true)
    );
    assert_eq!(split_project_flag("--project"), (String::new(), true));
    assert_eq!(
        split_project_flag("github:o/r"),
        ("github:o/r".to_string(), false)
    );
    // Mid-line `-p` belongs to the description, not the flag.
    assert_eq!(
        split_project_flag("deploy pass -p to the deploy script"),
        ("deploy pass -p to the deploy script".to_string(), false)
    );
    // `-project` (one dash) is not the flag and must survive intact.
    assert_eq!(
        split_project_flag("-project x"),
        ("-project x".to_string(), false)
    );
}

/// An unrelated open overlay is not replaced; the stage drops with a hint.
#[tokio::test]
async fn test_skills_install_pick_defers_to_open_overlay() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(pack.display().to_string())
        .await
        .unwrap();
    app.overlay = Overlay::Help { scroll: 0 };
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if app.installing_skill.is_none() && app.notice.is_some() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "an unrelated overlay is not replaced"
    );
    assert!(app.staged_skill_install.is_none(), "stage is discarded");
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("has 2 skills"), "{notice}");
    let _ = std::fs::remove_dir_all(&pack);
}

#[tokio::test]
async fn test_skills_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without scaffolding.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("skills overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[tokio::test]
async fn test_skills_delete_arms_then_cancels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.pending_delete, Some(0), "first Ctrl+D should arm");
    } else {
        panic!("skills overlay vanished on arm");
    }
    // Esc cancels the arm rather than closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.pending_delete.is_none(), "Esc should disarm");
    } else {
        panic!("Esc on an armed delete must not close the overlay");
    }
}

#[tokio::test]
async fn test_shift_tab_cycles_modes_through_handle_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve && !app.plan_mode);
    // Shift+Tab arrives as BackTab — normal → auto-approve.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "Shift+Tab should enable auto-approve"
    );
    // The shared LIVE flag the running agent turn reads tracks the mode.
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve ON"
    );
    // The Tab+SHIFT form some terminals send — auto-approve → plan mode.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert!(app.plan_mode, "second press cycles into plan mode");
    assert!(!app.agent_auto_approve, "modes are mutually exclusive");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve OFF"
    );
    // Third press — plan → review.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.plan_mode && app.agent_review_edits);
    // Fourth press — review → default.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.plan_mode && !app.agent_auto_approve && !app.agent_review_edits);
    // Ctrl+O is no longer an auto-approve alias (Shift+Tab only).
    app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "Ctrl+O no longer toggles auto-approve"
    );
}

#[tokio::test]
async fn test_shift_tab_on_permission_card_approves_and_enables() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // A tool-permission card is up, holding the keyboard.
    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // Shift+Tab on the card must enable auto-approve AND approve THIS request —
    // not get swallowed (the old behavior, where only y/a/n worked).
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve, "toggle enables auto-approve");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the live flag the running turn reads is set"
    );
    assert!(app.agent_permission.is_none(), "the card is dismissed");
    assert_eq!(
        decision_rx.await.unwrap(),
        Decision::Allow,
        "the pending request is approved"
    );
}

/// "Always" on a Cursor permission card flips session-wide auto-approve, so the
/// composer badge and exit-persistence (which read `agent_auto_approve`) must
/// agree with the shared live flag — the badge can't say "off" while Cursor
/// silently allows the rest of the session.
#[tokio::test]
async fn test_always_on_cursor_card_enables_auto_approve() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("Cursor wants to run:\nedit main.rs".to_string()),
        reply,
    });

    // 'a' = always: cursor "always" is session-wide auto-approve.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "cursor 'always' flips the bool the badge/persistence read"
    );
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "and the shared live flag the cursor session reads"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// In-process-engine "always" stays scoped to that (tool,args): it must NOT flip
/// the global auto-approve badge on — only Cursor cards do that.
#[tokio::test]
async fn test_always_on_native_card_leaves_auto_approve_off() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "native 'always' is per-tool, not global auto-approve"
    );
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the global live flag stays off for native 'always'"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// A project `.mcp.json` server is not in the *base* opt-out set (the consent
/// gate that holds stdio servers back lives in `connect_mcp_with_consent`, not
/// here), and toggling a project row off in `/mcp` adds it to the global opt-out
/// list, exactly like a user server.
#[tokio::test]
async fn project_mcp_server_connects_by_default_and_toggles_like_user() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-proj-mcp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"fs":{"command":"echo"}}}"#,
    )
    .unwrap();
    app.real_cwd = repo.to_str().unwrap().to_string();

    // The base opt-out set is empty (the consent gate is applied separately).
    assert!(
        !app.effective_disabled_mcp_servers().await.contains("fs"),
        "a project .mcp.json server is not in the base opt-out set"
    );

    // Toggling a project row off goes to the global opt-out, like a user server.
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "fs".to_string(),
            status: "1 tool".to_string(),
            health: McpHealth::Connected,
            enabled: true,
            scope: ServerScope::Project,
            command: "echo".to_string(),
            remote: false,
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(
        app.session_store.get_disabled_mcp_servers().await.unwrap(),
        vec!["fs".to_string()],
        "toggling a project row off adds it to the user opt-out list"
    );
    assert!(app.effective_disabled_mcp_servers().await.contains("fs"));

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn test_humanize_count() {
    use super::shared::humanize_count;
    assert_eq!(humanize_count(0), "0");
    assert_eq!(humanize_count(999), "999");
    assert_eq!(humanize_count(1234), "1.2k");
    assert_eq!(humanize_count(12345), "12k");
}

#[test]
fn test_sort_mcp_rows_problems_first() {
    use super::session_impl::sort_mcp_rows;
    use crate::agent::mcp::ServerScope;
    let row = |name: &str, health| McpServerRow {
        name: name.to_string(),
        status: String::new(),
        health,
        enabled: !matches!(health, McpHealth::Disabled),
        scope: ServerScope::User,
        command: "x".to_string(),
        remote: false,
    };
    let mut rows = vec![
        row("zeta", McpHealth::Connected),
        row("off-one", McpHealth::Disabled),
        row("beta", McpHealth::Failed),
        row("needs", McpHealth::NeedsAuth),
        row("alpha", McpHealth::Connected),
        row("idle", McpHealth::Idle),
    ];
    sort_mcp_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Failed first, then needs-auth (actionable), then connected (alphabetical),
    // then idle, then disabled last.
    assert_eq!(
        order,
        vec!["beta", "needs", "alpha", "zeta", "idle", "off-one"]
    );
}

#[test]
fn test_sort_skill_rows_enabled_first() {
    use super::session_impl::sort_skill_rows;
    use crate::agent::skills::SkillScope;
    let row = |name: &str, enabled| SkillToggle {
        name: name.to_string(),
        description: String::new(),
        enabled,
        dir: std::path::PathBuf::from(name),
        scope: SkillScope::User,
        body: String::new(),
    };
    let mut rows = vec![
        row("zoff", false),
        row("able", true),
        row("aoff", false),
        row("baker", true),
    ];
    sort_skill_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Enabled first (alphabetical), disabled at the bottom (alphabetical).
    assert_eq!(order, vec!["able", "baker", "aoff", "zoff"]);
}

#[tokio::test]
async fn test_skills_filter_narrows_and_enter_toggles() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // brandkit(on), critique(off)

    // Typing 'c' filters to "critique" (index 1); the selection re-anchors to it.
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.query, "c");
        assert_eq!(
            state.filtered_indices(),
            vec![1],
            "only critique matches 'c'"
        );
        assert_eq!(state.selected, 1, "selection re-anchored to the match");
    } else {
        panic!("overlay vanished");
    }
    // Enter toggles the matched skill (critique off → on).
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert!(
        !disabled.contains(&"critique".to_string()),
        "Enter should have toggled critique on"
    );
    // Backspace clears the one-char filter.
    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.query.is_empty(), "Backspace should clear the filter");
    }
}

#[tokio::test]
async fn test_mcp_filter_narrows_selection() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // filesystem, github

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.filtered_indices(), vec![1], "only github matches 'g'");
        assert_eq!(state.selected, 1);
        assert!(state.has_selection());
    } else {
        panic!("overlay vanished");
    }
    // A query matching nothing leaves no visible selection (so Enter/Tab no-op).
    app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(
            state.filtered_indices().is_empty(),
            "no server matches 'gz'"
        );
        assert!(!state.has_selection());
    }
}

#[test]
fn test_skills_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some("changelog".to_string());
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The `+` prompt marks the add field; the typed buffer and the install hint
    // both show, and the footer offers save/cancel.
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(
        screen.contains("changelog"),
        "missing typed input:\n{screen}"
    );
    assert!(
        screen.contains("github:owner/repo"),
        "missing add hint:\n{screen}"
    );
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

fn mcp_overlay_fixture() -> McpOverlay {
    use crate::agent::mcp::ServerScope;
    McpOverlay {
        items: vec![
            McpServerRow {
                name: "filesystem".to_string(),
                status: "5 tools".to_string(),
                health: McpHealth::Connected,
                enabled: true,
                scope: ServerScope::User,
                command: "npx".to_string(),
                remote: false,
            },
            McpServerRow {
                name: "github".to_string(),
                status: "off".to_string(),
                health: McpHealth::Disabled,
                enabled: false,
                scope: ServerScope::User,
                command: "docker".to_string(),
                remote: false,
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

fn wheel(kind: MouseEventKind) -> MouseEvent {
    MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }
}

#[tokio::test]
async fn test_skills_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection (the list follows it).
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // 2 items, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the body, leaving the selection put.
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 3 && s.selected == 0));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 0));

    // Add-input mode: the wheel is ignored (no selection move, no scroll).
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some(String::new());
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0 && s.detail_scroll == 0));
}

#[tokio::test]
async fn test_mcp_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection.
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // 2 servers, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the tool list.
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Mcp(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.detail_scroll == 3 && s.selected == 0));
}

fn test_screen(terminal: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
    let buf = terminal.backend().buffer().clone();
    let area = *buf.area();
    let mut screen = String::new();
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    screen
}

fn session_picker_fixture() -> (PickerState, SessionPreview) {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "sess-new".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
    };
    let older = SessionPreview {
        session_id: "sess-old".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
        ..newest.clone()
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest.clone()),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );
    (picker, newest)
}

fn preview_chat_message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        model: None,
        role: role.to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    }
}

#[test]
fn test_skills_overlay_splits_on_wide_terminal() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);

    assert!(screen.contains("filter skills"), "missing list:\n{screen}");
    assert!(
        screen.contains("Instructions:") && screen.contains("Render the boards"),
        "missing right-pane detail:\n{screen}"
    );
    assert!(
        app.overlay_detail_area.is_some(),
        "split should record the detail pane rect"
    );
}

#[test]
fn test_skills_overlay_narrow_keeps_single_pane_and_drill_in() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        !screen.contains("Instructions:"),
        "narrow list mode must not show the detail pane:\n{screen}"
    );
    assert!(app.overlay_detail_area.is_none());

    if let Overlay::Skills(state) = &mut app.overlay {
        state.viewing = Some(0);
    }
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("Instructions:") && screen.contains("Esc back"),
        "narrow drill-in should render full-modal:\n{screen}"
    );
}

#[tokio::test]
async fn test_skills_split_page_keys_scroll_detail_and_tab_disabled() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // A draw arms `overlay_detail_area` (the split-active signal for keys).
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    assert!(app.overlay_detail_area.is_some());

    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == DETAIL_PAGE_LINES),
        "PageDown should scroll the detail pane"
    );

    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.viewing.is_none()));

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 1 && s.detail_scroll == 0));
}

#[tokio::test]
async fn test_mcp_split_wheel_routes_by_pane() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let detail = app.overlay_detail_area.expect("split active");

    let mut over_detail = wheel(MouseEventKind::ScrollDown);
    over_detail.column = detail.x + 1;
    over_detail.row = detail.y + 1;
    app.handle_mouse(over_detail).await.unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.detail_scroll == 3 && s.selected == 0));

    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 1 && s.detail_scroll == 0));
}

#[test]
fn test_session_preview_lines_collapses_tool_runs() {
    let messages = vec![
        preview_chat_message("user", "hello there"),
        preview_chat_message("assistant", "**hi** back"),
        preview_chat_message("tool_call", "{\"name\":\"run_bash\"}"),
        preview_chat_message("tool_result", "output"),
        preview_chat_message("tool_call", "{\"name\":\"read_file\"}"),
        preview_chat_message("assistant", "done"),
    ];
    let (lines, bars) = session_preview_lines(&messages, 60, true);
    let plain: Vec<&str> = lines.iter().map(|l| l.plain.as_str()).collect();

    assert!(
        plain[0].contains("earlier messages not shown"),
        "truncation banner missing: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("⚙ 3 tool steps")),
        "tool run should collapse to one line: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("hi back")),
        "assistant markdown missing: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("hello there")),
        "user message missing: {plain:?}"
    );
    let tool_row = plain
        .iter()
        .position(|l| l.contains("⚙ 3 tool steps"))
        .unwrap();
    assert_eq!(bars[tool_row], Some(TOOL));
}

#[tokio::test]
async fn test_session_picker_preview_bottom_anchor_and_clamp() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, newest) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));
    let messages: Vec<ChatMessage> = (0..80)
        .map(|i| {
            preview_chat_message(
                if i % 2 == 0 { "user" } else { "assistant" },
                &format!("message number {i}"),
            )
        })
        .collect();
    app.session_preview_cache.insert(
        newest.session_id.clone(),
        PreviewEntry {
            updated_at: newest.updated_at.clone(),
            messages,
            truncated: false,
            error: None,
        },
    );

    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("message number 79"),
        "preview should anchor to the latest message:\n{screen}"
    );
    assert!(
        !screen.contains("message number 0 "),
        "oldest message should be scrolled out:\n{screen}"
    );

    // Home over-scrolls to u16::MAX; the renderer clamps to the oldest line.
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("message number 0 "),
        "Home should land on the oldest loaded message:\n{screen}"
    );
    assert!(!screen.contains("message number 79"));
    if let Overlay::Picker(p) = &app.overlay {
        assert!(p.preview_scroll < u16::MAX);
        assert_eq!(p.preview_scroll_for.as_deref(), Some("sess-new"));
    } else {
        panic!("picker vanished");
    }
}

#[tokio::test]
async fn test_session_preview_loaded_caches_even_when_not_selected() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, _) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    app.tx
        .send(RuntimeEvent::SessionPreviewLoaded {
            session_id: "sess-old".to_string(),
            entry: PreviewEntry {
                updated_at: "t1".to_string(),
                messages: vec![preview_chat_message("user", "old hello")],
                truncated: false,
                error: None,
            },
        })
        .unwrap();
    assert!(app.handle_runtime_events().await.unwrap());
    assert!(app.session_preview_cache.contains_key("sess-old"));
}

#[tokio::test]
async fn test_tick_session_preview_debounce_and_invalidation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, newest) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    // No split pane rendered yet → nothing scheduled.
    assert!(!app.tick_session_preview());
    assert!(app.session_preview_pending.is_none());

    // Split active: the first tick arms the debounce, the next is not yet due.
    app.overlay_detail_area = Some(Rect::new(0, 0, 40, 20));
    assert!(!app.tick_session_preview());
    assert!(app.session_preview_pending.is_some());
    assert!(!app.tick_session_preview());
    assert!(app.session_preview_task.is_none());

    // Once due, exactly one load task spawns and the pending slot clears.
    if let Some((_, due)) = &mut app.session_preview_pending {
        *due = Instant::now() - Duration::from_millis(1);
    }
    assert!(app.tick_session_preview());
    assert!(app.session_preview_task.is_some());
    assert!(app.session_preview_pending.is_none());

    // A valid cache entry (matching updated_at) suppresses any reload…
    app.session_preview_task = None;
    app.session_preview_cache.insert(
        newest.session_id.clone(),
        PreviewEntry {
            updated_at: newest.updated_at.clone(),
            messages: vec![],
            truncated: false,
            error: None,
        },
    );
    assert!(!app.tick_session_preview());
    assert!(app.session_preview_pending.is_none());

    // …while a stale one (index row updated since) re-arms the debounce.
    app.session_preview_cache
        .get_mut(&newest.session_id)
        .unwrap()
        .updated_at = "stale".to_string();
    assert!(!app.tick_session_preview());
    assert!(app.session_preview_pending.is_some());
}

#[tokio::test]
async fn test_session_picker_split_click_routes_left_rows_only() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, _) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let detail = app.overlay_detail_area.expect("split active");
    let hitbox = app.picker_hitbox.clone().expect("hitbox recorded");

    // A click inside the preview pane neither activates nor closes.
    let mut click = wheel(MouseEventKind::Down(MouseButton::Left));
    click.column = detail.x + 2;
    click.row = detail.y + 2;
    app.handle_mouse(click).await.unwrap();
    assert!(matches!(&app.overlay, Overlay::Picker(_)));

    // A click on a mapped list row resumes that session (picker closes).
    let row = hitbox
        .row_to_filtered_index
        .iter()
        .position(|idx| idx.is_some())
        .expect("a clickable row") as u16;
    let mut click = wheel(MouseEventKind::Down(MouseButton::Left));
    click.column = hitbox.list_area.x + 1;
    click.row = hitbox.list_area.y + row;
    app.handle_mouse(click).await.unwrap();
    assert!(
        !matches!(&app.overlay, Overlay::Picker(_)),
        "row click should activate the session"
    );
    assert!(app.loading_resume.is_some());
}

#[test]
fn test_composer_command_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let set = |app: &mut CodeTuiApp, draft: &str| {
        app.draft = draft.to_string();
        app.cursor = app.draft.len();
    };

    // Bare command (and a trailing space) → ghost hint with the arg syntax.
    set(&mut app, "/mcp");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add [-p] <command>")),
        "typing /mcp should ghost the add/rm syntax"
    );
    set(&mut app, "/mcp ");
    assert!(
        app.composer_command_hint().is_some(),
        "trailing space still bare"
    );
    // Once a real argument is typed, the hint clears.
    set(&mut app, "/mcp add");
    assert!(
        app.composer_command_hint().is_none(),
        "args typed → no ghost"
    );

    // Other argument-taking commands ghost their arg syntax too.
    for (draft, expect) in [
        ("/model", "[name]"),
        ("/key", "[id|name]"),
        ("/resume", "[query]"),
        ("/copy", "[n]"),
        ("/detach", "<n>"),
    ] {
        set(&mut app, draft);
        assert_eq!(
            app.composer_command_hint(),
            Some(expect),
            "{draft} should ghost {expect}"
        );
    }
    // `/skills` ghosts its add/rm subcommand syntax, like `/mcp`.
    set(&mut app, "/skills");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add [-p] <name>")),
        "/skills should ghost the add/rm syntax"
    );

    // `/attach` is deliberately hint-less: typing `/attach ` opens path
    // completion, and a ghost would suppress that menu.
    set(&mut app, "/attach");
    assert!(
        app.composer_command_hint().is_none(),
        "attach must not ghost (path menu owns it)"
    );
    // A no-argument command has no hint either.
    set(&mut app, "/new");
    assert!(app.composer_command_hint().is_none());

    // None for a literal slash or plain text.
    set(&mut app, "//mcp");
    assert!(app.composer_command_hint().is_none());
    set(&mut app, "hello");
    assert!(app.composer_command_hint().is_none());
    // Cursor not at the end → no ghost (e.g. editing mid-line).
    app.draft = "/mcp".to_string();
    app.cursor = 2;
    assert!(app.composer_command_hint().is_none());
}

#[test]
fn test_mcp_ghost_hint_renders_in_composer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mcp".to_string();
    app.cursor = app.draft.len();

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The command and its ghost arg-hint share the composer line.
    assert!(
        screen.contains("/mcp [add [-p] <command>"),
        "composer should show the inline ghost hint:\n{screen}"
    );
}

#[test]
fn test_mcp_overlay_renders_server_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("MCP servers"), "missing title:\n{screen}");
    assert!(
        screen.contains("filesystem"),
        "missing server name:\n{screen}"
    );
    // The status renders on its own line under the server name.
    assert!(screen.contains("5 tools"), "missing status:\n{screen}");
    assert!(
        screen.contains("[✓]") && screen.contains("[ ]"),
        "missing checkboxes:\n{screen}"
    );
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
}

#[test]
fn test_mcp_overlay_renders_detail_line_for_selected() {
    use crate::agent::mcp::ServerScope;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Select a disabled, project-scoped server: with no live client its detail
    // line should still spell out the scope and the (actionable) disabled state.
    let mut overlay = mcp_overlay_fixture();
    overlay.items[1].scope = ServerScope::Project;
    overlay.selected = 1; // "github", off
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project (.mcp.json)"),
        "detail line should tag a project-scoped server:\n{screen}"
    );
    assert!(
        screen.contains("disabled"),
        "detail line should show the disabled state:\n{screen}"
    );
}

/// A per-server progress event mid-connect flips just that server's row to its
/// resolved status; other still-connecting servers keep reading "connecting…".
/// A stale-generation event is ignored.
#[tokio::test]
async fn test_mcp_progress_flips_only_resolved_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture(); // filesystem, github
    // Both enabled and mid-connect.
    for item in &mut overlay.items {
        item.enabled = true;
        item.status = "connecting…".to_string();
        item.health = McpHealth::Idle;
    }
    app.overlay = Overlay::Mcp(overlay);
    app.mcp_connecting = true;
    app.mcp_client = None;

    // "filesystem" resolves; "github" hasn't yet.
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "filesystem".to_string(),
            status: "5 tools".to_string(),
            health: McpHealth::Connected,
            generation: app.mcp_connect_gen,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    let row = |app: &CodeTuiApp, name: &str| -> (String, McpHealth) {
        match &app.overlay {
            Overlay::Mcp(s) => s
                .items
                .iter()
                .find(|i| i.name == name)
                .map(|i| (i.status.clone(), i.health))
                .unwrap(),
            _ => panic!("overlay vanished"),
        }
    };
    assert_eq!(
        row(&app, "filesystem"),
        ("5 tools".to_string(), McpHealth::Connected),
        "resolved server should flip to its tool count"
    );
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "unresolved server should still read connecting"
    );

    // A stale-generation event (a connect superseded by a toggle) is dropped.
    let stale = app.mcp_connect_gen.wrapping_add(7);
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "github".to_string(),
            status: "9 tools".to_string(),
            health: McpHealth::Connected,
            generation: stale,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "a stale-generation progress event must be ignored"
    );
}

#[tokio::test]
async fn test_toggle_mcp_server_persists_and_resets() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Disable "filesystem" (index 0, currently enabled).
    app.toggle_mcp_server(0).await.unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
        assert_eq!(state.items[0].status, "off");
    } else {
        panic!("mcp overlay vanished");
    }
    let disabled = app.session_store.get_disabled_mcp_servers().await.unwrap();
    assert_eq!(disabled, vec!["filesystem".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_mcp_server(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap()
            .is_empty()
    );
}

/// Toggling a server refreshes the welcome chip's MCP count instead of freezing
/// it at the startup value.
#[tokio::test]
async fn test_toggle_mcp_server_updates_welcome_count() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // filesystem on, github off
    app.mcp_configured_count = 42; // stale value the fix must overwrite

    // Disable the one enabled server → 0 enabled.
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(
        app.mcp_configured_count, 0,
        "count not refreshed on disable"
    );

    // Re-enable it → back to 1.
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(app.mcp_configured_count, 1, "count not refreshed on enable");
}

/// Toggling a server keeps the live client (rather than nulling it) so the
/// servers that *aren't* being toggled keep serving their status during the
/// reconnect — and bumps the generation so the reconnect supersedes any in-flight
/// one. (The connection-level reuse itself is covered by mcp's
/// `reconnect_reuses_live_servers`.)
#[tokio::test]
async fn test_toggle_preserves_live_client() {
    let empty_dir = tempfile::tempdir().unwrap();
    let live = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(
            empty_dir.path(),
            &std::collections::HashSet::new(),
        )
        .await,
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());
    app.mcp_client = Some(live);
    let gen0 = app.mcp_connect_gen;

    app.toggle_mcp_server(0).await.unwrap();

    assert!(
        app.mcp_client.is_some(),
        "toggle must keep the live client (for the other servers' status), not null it"
    );
    assert_ne!(
        app.mcp_connect_gen, gen0,
        "generation should advance so the reconnect supersedes any stale one"
    );
    assert!(app.mcp_connecting, "a reconnect should be in flight");
}

#[test]
fn test_parse_mcp_add_input() {
    use super::session_impl::parse_mcp_add_input;
    // No name — the first token is the command; a shell-quoted path survives.
    let (command, args) = parse_mcp_add_input("npx -y srv \"/a b/c\"").unwrap();
    assert_eq!(command, "npx");
    assert_eq!(args, vec!["-y", "srv", "/a b/c"]);
    // A bare command with no args is fine.
    assert_eq!(
        parse_mcp_add_input("my-mcp-binary").unwrap(),
        ("my-mcp-binary".to_string(), Vec::<String>::new())
    );
    // Empty input is a usage error.
    assert!(parse_mcp_add_input("   ").is_err());
}

#[tokio::test]
async fn test_mcp_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without writing config.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("mcp overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[test]
fn test_mcp_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.adding = Some("fs npx".to_string());
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(screen.contains("fs npx"), "missing typed input:\n{screen}");
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

#[tokio::test]
async fn test_mcp_drill_in_tab_then_esc() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Tab drills into the selected server's details.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.viewing, Some(0), "Tab should open the detail view");
    } else {
        panic!("overlay closed on Tab instead of drilling in");
    }
    // Esc backs out to the list, NOT closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.viewing.is_none(), "Esc should back out of detail");
    } else {
        panic!("Esc in detail closed the overlay instead of returning to the list");
    }
}

#[test]
fn test_mcp_drill_in_renders_command_and_footer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0); // "filesystem", command "npx"
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("command:"),
        "detail missing command:\n{screen}"
    );
    assert!(
        screen.contains("npx"),
        "detail missing the command value:\n{screen}"
    );
    assert!(
        screen.contains("Esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// The drill-in tool list stacks each name over its wrapped description (no
/// far-right column): a `•` name line, then `    `-indented description lines,
/// with a blank line separating tools and long descriptions wrapping.
#[test]
fn test_mcp_tool_lines_stacks_name_over_wrapped_desc() {
    use super::overlay_render_impl::mcp_tool_lines;

    let line_text =
        |line: &Line| -> String { line.spans.iter().map(|s| s.content.as_ref()).collect() };
    let tools = [
        ("short_tool", "A brief description.", true),
        (
            "browserslist_compatibility_check",
            "Check web feature compatibility against your browserslist configuration across many supported browsers.",
            true,
        ),
    ];
    let lines: Vec<String> = mcp_tool_lines(&tools, 40).iter().map(line_text).collect();

    // Each name is on its own bulleted line.
    assert!(
        lines.iter().any(|l| l == "  • short_tool"),
        "first tool name not on its own line:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "  • browserslist_compatibility_check"),
        "second tool name not on its own line:\n{lines:#?}"
    );
    // The description sits indented beneath the name (not in a right-hand column).
    assert!(
        lines.iter().any(|l| l == "    A brief description."),
        "description not indented under the name:\n{lines:#?}"
    );
    // A blank line separates the two tools.
    assert!(
        lines.iter().any(|l| l.is_empty()),
        "expected a blank separator between tools:\n{lines:#?}"
    );
    // The long description wraps onto multiple indented lines, none over width.
    let desc_lines = lines
        .iter()
        .filter(|l| l.starts_with("    ") && l.contains("browserslist") || l.contains("supported"))
        .count();
    assert!(
        desc_lines >= 2,
        "long description should wrap to multiple lines:\n{lines:#?}"
    );
    assert!(
        lines.iter().all(|l| display_width(l) <= 40),
        "no rendered line should exceed the width:\n{lines:#?}"
    );
}

#[test]
fn test_skill_drill_in_renders_body_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0); // "brandkit", body "Step 1. Render the boards."
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Instructions:"),
        "detail missing body header:\n{screen}"
    );
    assert!(
        screen.contains("Render the boards"),
        "detail missing the SKILL.md body:\n{screen}"
    );
    assert!(
        screen.contains("Esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// A long SKILL.md body is scrollable in the drill-in: the top hides later lines,
/// End reveals the last line (clamped by the renderer), and Esc resets the scroll.
#[tokio::test]
async fn test_skill_drill_in_scrolls_long_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.items[0].body = (1..=60)
        .map(|i| format!("Line number {i} of the instructions"))
        .collect::<Vec<_>>()
        .join("\n");
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);

    let render_screen = |app: &mut CodeTuiApp| -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut s = String::new();
        for y in 0..20u16 {
            for x in 0..80u16 {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    };

    let top = render_screen(&mut app);
    assert!(
        top.contains("Line number 1 of"),
        "top hides first line:\n{top}"
    );
    assert!(
        !top.contains("Line number 60 of"),
        "last line should be off-screen at the top:\n{top}"
    );
    assert!(
        top.contains("scroll"),
        "a scrollable body shows the scroll hint:\n{top}"
    );

    // End jumps to the bottom (the renderer clamps the offset).
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let bottom = render_screen(&mut app);
    assert!(
        bottom.contains("Line number 60 of"),
        "End reveals the last line:\n{bottom}"
    );
    match &app.overlay {
        Overlay::Skills(s) => assert!(s.detail_scroll > 0, "scroll offset advanced"),
        _ => panic!("overlay vanished"),
    }

    // Esc backs out of the drill-in and resets the scroll.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Skills(s) => {
            assert!(s.viewing.is_none(), "Esc leaves the drill-in");
            assert_eq!(s.detail_scroll, 0, "scroll resets on back-out");
        }
        _ => panic!("overlay vanished"),
    }
}

#[test]
fn test_restore_cancelled_submission_puts_prompt_back() {
    let mut history = vec![ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    }];
    let mut draft = String::new();
    let mut draft_attachments = Vec::new();
    let mut pending_submit = Some(PendingSubmission {
        content: "draft".to_string(),
        attachments: Vec::new(),
    });

    restore_cancelled_submission(
        &mut history,
        &mut draft,
        &mut draft_attachments,
        &mut pending_submit,
    );

    assert!(history.is_empty());
    assert_eq!(draft, "draft");
    assert!(draft_attachments.is_empty());
    assert!(pending_submit.is_none());
}

#[test]
fn test_prepare_submit_action_bang_runs_shell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!ls -la".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Shell(cmd)) if cmd == "ls -la"
    ));
}

#[test]
fn test_prepare_submit_action_double_bang_escapes_to_literal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!!important".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(text)) if text == "!important"
    ));
}

#[test]
fn test_prepare_submit_action_bare_bang_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!   ".to_string();

    assert!(app.prepare_submit_action().is_err());
}

#[test]
fn test_prepare_submit_action_interactive_bang_is_refused() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for draft in ["!vim notes.txt", "!make && vim Cargo.toml", "!top"] {
        app.draft = draft.to_string();
        // `SubmitAction` isn't `Debug`, so match rather than `unwrap_err`.
        match app.prepare_submit_action() {
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains("separate terminal") || err.contains("ps aux"),
                    "{draft}: {err}"
                );
            }
            Ok(_) => panic!("{draft} should be refused"),
        }
    }
    // A non-interactive command in the same family still runs.
    app.draft = "!git add src/".to_string();
    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Shell(cmd)) if cmd == "git add src/"
    ));
    // `tail -f`/`watch` stream live under the PTY (esc stops them), so `!cmd` runs
    // them even though the agent's `run_bash` refuses them.
    for draft in ["!tail -f server.log", "!watch ls"] {
        app.draft = draft.to_string();
        assert!(
            matches!(
                app.prepare_submit_action().unwrap(),
                Some(SubmitAction::Shell(_))
            ),
            "{draft} should run under !cmd"
        );
    }
}

// Unix-only: drives the `!cmd` PTY with POSIX commands (`printf`) that the
// Windows shell (PowerShell) doesn't provide, and a PTY read that blocks until
// the child exits would stall the tokio runtime drop on Windows. The `!cmd`
// feature itself is cross-platform; only this Unix-command assertion is gated.
#[cfg(unix)]
#[tokio::test]
async fn test_run_local_command_is_display_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf hi".to_string());
    run_local_command_to_completion(&mut app).await;

    // A transcript step is recorded for display once the run finishes.
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"command\":\"printf hi\""));
    assert!(step.content.contains("hi"));

    // It is purely local: the `local_command` role is excluded from the model
    // context (only user/assistant turns are sent), so nothing reaches the server.
    let sent_to_model = app
        .history
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .count();
    assert_eq!(sent_to_model, 0);
}

// Unix-only: see `test_run_local_command_is_display_only` — POSIX `printf`
// through the PTY isn't portable to the Windows shell.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf 'a\\nb\\nc\\n'".to_string());
    // The run is live (not yet in history) right after starting.
    assert!(app.local_command.is_some());
    assert!(app.history.is_empty());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once, with the streamed output and a zero exit code.
    assert!(app.local_command.is_none());
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[cfg(unix)]
#[tokio::test]
async fn test_local_command_neutralizes_pager() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // A pager-spawning command (`git diff`) would launch `less` under the PTY and
    // hang waiting for a keypress we never send. The spawn env disables pagers, so
    // the child sees PAGER/GIT_PAGER=cat — verify that reaches it through the PTY.
    app.start_local_command("echo \"p=[$PAGER] g=[$GIT_PAGER]\"".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert!(
        step.content.contains("p=[cat] g=[cat]"),
        "pager env not injected into the PTY child: {}",
        step.content
    );
}

// Unix-only: `yes` (the infinite-flood command this caps) has no PowerShell
// equivalent, and the PTY drive is Unix-oriented like the sibling tests above.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_caps_huge_output() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `yes` floods forever — the reader must cap, kill it, and still finish.
    app.start_local_command("yes aivo".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"truncated\":true"));
    let stdout = serde_json::from_str::<serde_json::Value>(&step.content)
        .unwrap()
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .lines()
        .count();
    // Bounded by the capture cap, not unbounded.
    assert!(stdout <= 1000, "captured {stdout} lines, expected ≤ 1000");

    // A run we killed at the cap must NOT render a scary `[exited -1]` — the
    // "truncated" note explains the stop; the SIGKILL status is ours, not `yes`'s.
    let mut block = Vec::new();
    render_local_command(&mut block, &step.content, OutputView::Collapsed);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !rendered.contains("[exited"),
        "truncated run should not show an exit code:\n{rendered}"
    );
    assert!(
        rendered.contains("truncated"),
        "truncated run should show the truncated note:\n{rendered}"
    );
}

#[tokio::test]
async fn test_local_command_full_output_kept_for_inline_expand() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 250 output lines: past the 40-line display cap AND the persisted preview.
    let full: String = (1..=250).map(|i| format!("{i}\n")).collect();
    let total =
        app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    assert_eq!(total, 250);
    let idx = app.history.len() - 1;

    // The committed transcript entry keeps only a bounded preview…
    let step = app.history.last().expect("history entry");
    let decoded: serde_json::Value = serde_json::from_str(&step.content).unwrap();
    let persisted = decoded["stdout"].as_str().unwrap().lines().count();
    assert!(
        persisted <= MAX_PERSISTED_OUTPUT_LINES,
        "persisted {persisted} lines, expected ≤ {MAX_PERSISTED_OUTPUT_LINES}"
    );
    // …but records the TRUE total, so the transcript's "+N more" stays honest.
    assert_eq!(decoded["total_lines"].as_u64(), Some(250));

    // The full output is retained in memory keyed by the entry's history index (the
    // source an expanded block renders from), never persisted into history.
    let kept = app.local_outputs.get(&idx).expect("full output retained");
    assert_eq!(kept.stdout.lines().count(), 250);
}

#[test]
fn test_render_local_command_marker_counts_total_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let _app = make_test_app(tx, rx);
    // A committed entry stores a small preview but the true total in `total_lines`.
    let content = serde_json::json!({
        "command": "find .",
        "stdout": "a\nb\nc\n",
        "stderr": "",
        "exit_code": 0,
        "total_lines": 41243,
    })
    .to_string();
    let mut block = Vec::new();
    render_local_command(&mut block, &content, OutputView::Collapsed);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    // Marker reflects the true total (41243 − 3 shown), not the 3 persisted lines.
    assert!(
        rendered.contains("+41240 more lines"),
        "marker should count the true total:\n{rendered}"
    );
}

#[test]
fn test_local_command_long_line_wraps_in_full() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long = "./target/release/.fingerprint/typenum-2abcdef0123456789-very-long/lib-typenum.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let width: u16 = 58;
    let body = app.build_transcript_history_body(width);
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, width);

    // No row overflows the width (so ratatui's wrap-OFF render can't clip)…
    for row in &wrapped.rows {
        assert!(
            row_display_width(row) <= width,
            "row exceeds width: {row:?}"
        );
    }
    // …the long path is shown IN FULL — wrapped onto an extra row, not truncated.
    // Stitch every row (the path has no spaces, so dropping spaces rejoins it).
    let stitched: String = wrapped.rows.iter().map(|r| r.replace(' ', "")).collect();
    assert!(
        stitched.contains(long),
        "long path not shown in full across wrapped rows:\n{stitched}"
    );
    assert!(
        !stitched.contains('…'),
        "output should not be per-line truncated with an ellipsis"
    );
    let path_rows = wrapped
        .rows
        .iter()
        .filter(|r| r.contains("target") || r.contains("typenum"))
        .count();
    assert!(
        path_rows >= 2,
        "the long line should wrap onto multiple rows"
    );
}

#[test]
fn test_render_main_local_command_no_clip() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long =
        "./target/release/.fingerprint/typenum-2ebc5dae76d28bAAAAAA/lib-typenum-bbbbbbbbbbbb.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let (w, h) = (60u16, 20u16);
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();

    // Through the FULL render pipeline (cache → wrap → slice → paint), the path is
    // shown in full — wrapped across rows, never truncated (`…`) nor edge-clipped
    // (which would drop the tail). The path has no spaces, so stitching the whole
    // screen with spaces and the accent-bar glyph removed rejoins it intact.
    let mut screen = String::new();
    for y in 0..h {
        for x in 0..w {
            screen.push_str(buf[(x, y)].symbol());
        }
    }
    let compact: String = screen
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '▌')
        .collect();
    assert!(
        compact.contains(long),
        "path not shown in full (clipped/truncated):\n{screen}"
    );
    assert!(
        !screen.contains('…'),
        "output should wrap in full, not truncate with an ellipsis:\n{screen}"
    );
}

/// Drive a started `!cmd` to completion: drain runtime events until the run
/// commits to history (clearing `local_command`). Bounded so a `!cmd` that never
/// finishes (the Windows ConPTY hang this guards against) fails the test instead of
/// hanging the runner forever.
#[cfg(any(unix, windows))]
async fn run_local_command_to_completion(app: &mut CodeTuiApp) {
    for _ in 0..5000 {
        app.handle_runtime_events().await.unwrap();
        if app.local_command.is_none() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    panic!("local command did not finish in time");
}

// Windows counterpart of `test_local_command_streams_then_commits`: `!cmd` runs
// through plain pipes (not ConPTY), whose output EOFs when the child exits — so the
// run must stream output AND commit (clear `local_command`). The bug this guards
// against left every `!cmd` stuck at "running…" forever. Uses a PowerShell command
// since `bare_shell` spawns PowerShell on Windows.
#[cfg(windows)]
#[tokio::test]
async fn test_local_command_pipe_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("Write-Output a; Write-Output b; Write-Output c".to_string());
    assert!(app.local_command.is_some());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once (run finished), with the streamed output and exit 0.
    assert!(app.local_command.is_none(), "the run must finish, not hang");
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[test]
fn test_prepare_submit_action_allows_attachment_only_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "notes.md".to_string(),
        mime_type: "text/markdown".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./notes.md".to_string(),
        },
    });

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(input)) if input.is_empty()
    ));
}

#[test]
fn test_detach_attachment_removes_selected_item() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments = vec![
        MessageAttachment {
            name: "one.txt".to_string(),
            mime_type: "text/plain".to_string(),
            storage: AttachmentStorage::FileRef {
                path: "./one.txt".to_string(),
            },
        },
        MessageAttachment {
            name: "two.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::FileRef {
                path: "./two.png".to_string(),
            },
        },
    ];

    app.detach_attachment(2).unwrap();

    assert_eq!(app.draft_attachments.len(), 1);
    assert_eq!(app.draft_attachments[0].name, "one.txt");
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Removed image: two.png")
    );
}

#[tokio::test]
async fn test_submit_draft_keeps_failed_attach_command_and_shows_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/attach ./definitely-missing-file.txt".to_string();
    app.cursor = app.draft.len();

    let should_exit = app.submit_draft().await.unwrap();

    assert!(!should_exit);
    assert_eq!(app.draft, "/attach ./definitely-missing-file.txt");
    assert!(app.draft_attachments.is_empty());
    assert!(
        app.notice.as_ref().is_some_and(
            |(color, text)| *color == ERROR && text.contains("Failed to read attachment")
        )
    );
}

#[test]
fn test_composer_attachment_lines_show_indices() {
    let lines = composer_attachment_lines(&[MessageAttachment {
        name: "hi.css".to_string(),
        mime_type: "text/css".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./hi.css".to_string(),
        },
    }]);
    let plain = plain_text_from_spans(&lines[0].spans);
    assert_eq!(plain, "· 1. [file] hi.css");
}

/// Opening the model picker mid-turn must NOT cancel the in-flight turn (it
/// used to): the running turn keeps its model and the pick applies next turn,
/// same as the agent's `switch_model` tool.
#[tokio::test]
async fn test_open_model_picker_keeps_inflight_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.open_model_picker(None, ModelSelectionTarget::CurrentChat, false);

    assert!(app.sending, "the in-flight turn must keep running");
    assert_eq!(app.pending_response, "partial");
    assert_eq!(
        app.history.len(),
        1,
        "the user turn stays in the transcript"
    );
    assert!(matches!(app.overlay, Overlay::Picker(_)));
}

/// `/model <name>` applies the name directly, opening no picker.
#[tokio::test]
async fn test_model_command_applies_name_directly() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.set_model_direct("my-model".to_string()).await.unwrap();

    assert_eq!(app.raw_model, "my-model");
    assert!(matches!(app.overlay, Overlay::None));
    let (color, msg) = app.notice.as_ref().expect("a confirmation notice");
    assert_eq!(*color, MUTED);
    assert!(msg.contains("my-model"), "notice names the model: {msg}");
}

#[tokio::test]
async fn test_cancel_keeps_user_turn_for_in_process_agent_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "edit the config".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "edit the config".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    // Mark an in-process agent turn as in flight (its per-turn serve is up).
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.cancel_inflight_request(CancelKind::Discard);

    // The engine already consumed this turn (and may have edited files), so the
    // request stays in the transcript instead of being silently un-sent — unlike
    // the plain-chat Discard path, which drops the dangling user message.
    assert_eq!(app.history.len(), 1, "agent user turn must be kept");
    assert_eq!(app.history[0].content, "edit the config");
    assert!(
        app.draft.is_empty(),
        "an agent turn must not be restored to the composer"
    );
    assert!(app.pending_submit.is_none());
    assert!(!app.sending);
}

#[tokio::test]
async fn test_interrupt_inflight_request_keeps_partial_response() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "session-123".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "draft".to_string(),
        attachments: Vec::new(),
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert!(app.pending_response.is_empty());
    assert!(app.pending_submit.is_none());
    assert!(app.draft.is_empty());
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[1].role, "assistant");
    assert_eq!(app.history[1].content, "partial");
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Response interrupted")
    );
}

/// A leaked tool-call streamed as text must not persist in the scrollback — the
/// engine emits `AgentDiscardSegment` so only the retry's clean answer commits.
#[tokio::test]
async fn test_discard_segment_drops_leaked_markup_from_scrollback() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "<tool_calls>{\"name\":\"read_file\"}</tool_calls>".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    // Engine strips + retries → tells the UI to drop the leaked segment.
    app.tx.send(RuntimeEvent::AgentDiscardSegment).unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.pending_response.is_empty(), "typed reply cleared");
    assert!(app.incoming_buffer.is_empty(), "buffered reply cleared");
    // The retry's real answer streams in fresh.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "done".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.flush_pending_assistant();
    let last = app.history.last().expect("a committed assistant segment");
    assert_eq!(last.content, "done");
    assert!(
        !last.content.contains("<tool_calls>"),
        "leaked markup must never reach the scrollback: {:?}",
        last.content
    );
}

/// Esc on a still-pending request returns the message to the composer, un-sent.
#[tokio::test]
async fn test_interrupt_empty_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first message".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "first message".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert_eq!(
        app.draft, "first message",
        "the pending message returns to the composer"
    );
    assert_eq!(app.cursor, "first message".len());
    assert!(app.pending_submit.is_none());
    assert!(
        app.history.is_empty(),
        "the unanswered user turn is un-sent so resent history stays alternating"
    );

    app.insert_char_at_cursor('!');
    assert_eq!(app.draft, "first message!");
}

/// A draft typed while pending is not clobbered by the un-sent message.
#[tokio::test]
async fn test_interrupt_empty_keeps_typed_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first message".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "first message".to_string(),
        attachments: Vec::new(),
    });
    app.draft = "typed while waiting".to_string();
    app.cursor = app.draft.len();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert_eq!(
        app.draft, "typed while waiting",
        "a freshly typed draft is not overwritten by the cancelled message"
    );
    assert!(
        app.history.is_empty(),
        "the unanswered user turn is un-sent"
    );
}

/// An agent turn that produced nothing is un-sent too (engine merges on resend).
#[tokio::test]
async fn test_interrupt_empty_agent_turn_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "edit the config".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "edit the config".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert_eq!(app.draft, "edit the config");
    assert!(app.pending_submit.is_none());
    assert!(
        app.history.is_empty(),
        "the untouched agent turn is un-sent"
    );
}

/// An agent turn that already ran a tool is kept, not un-sent.
#[tokio::test]
async fn test_interrupt_empty_agent_turn_with_tool_keeps_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "edit the config".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: "{\"name\":\"edit_file\"}".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "edit the config".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert!(
        app.draft.is_empty(),
        "a turn that ran a tool is not restored"
    );
    assert_eq!(app.history.len(), 2, "the user + tool rows are kept");
    assert!(app.pending_submit.is_none());
}

/// Watchdog: a task that finished WITHOUT a terminal event (a `run_turn` panic
/// before `ui.footer`) must not leave the turn stuck "sending"; it salvages
/// partial text, resets, and stops the `/goal` loop. A running turn is untouched.
#[tokio::test]
async fn test_recover_dead_response_task_resets_stuck_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "do it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.goal_mode = Some(GoalState {
        objective: "do it".to_string(),
        iteration: 1,
        max: 20,
    });
    // A finished task that sent NO terminal event (stands in for a panic).
    let dead = tokio::spawn(async {});
    for _ in 0..100 {
        if dead.is_finished() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(dead.is_finished(), "spawned task should have completed");
    app.response_task = Some(dead);

    let recovered = app.recover_dead_response_task().await.unwrap();
    assert!(recovered, "a dead, still-sending turn must be recovered");
    assert!(!app.sending, "sending must be reset");
    assert!(app.response_task.is_none());
    assert!(
        app.goal_mode.is_none(),
        "goal loop must stop, not auto-continue into a likely repeat"
    );
    let last = app.history.last().unwrap();
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "partial");
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("goal mode stopped"), "{notice}");

    // A healthy in-flight turn (task still running) is left strictly alone.
    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
    let mut app2 = make_test_app(tx2, rx2);
    app2.sending = true;
    app2.response_task = Some(tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    }));
    let recovered2 = app2.recover_dead_response_task().await.unwrap();
    assert!(!recovered2, "a running turn must not be touched");
    assert!(app2.sending, "a running turn stays sending");
    if let Some(task) = app2.response_task.take() {
        task.abort();
    }
}

/// Stopping a turn (cancel / empty interrupt / partial interrupt) must exit any
/// `/goal` loop, so `maybe_continue_goal` can't auto-continue after the user stops.
#[tokio::test]
async fn test_stopping_a_turn_clears_goal_mode() {
    let armed = || {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.sending = true;
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 1,
            max: 20,
        });
        app
    };

    let mut app = armed();
    app.cancel_inflight_request(CancelKind::Discard);
    assert!(app.goal_mode.is_none(), "cancel must clear goal mode");

    // Interrupt with NO partial response routes through cancel.
    let mut app = armed();
    app.interrupt_inflight_request().await.unwrap();
    assert!(
        app.goal_mode.is_none(),
        "empty interrupt must clear goal mode"
    );
    assert_eq!(app.notice.as_ref().unwrap().1, "Goal mode stopped");

    // Interrupt WITH a partial response takes the salvage path.
    let mut app = armed();
    app.pending_response = "half a reply".to_string();
    app.interrupt_inflight_request().await.unwrap();
    assert!(
        app.goal_mode.is_none(),
        "partial interrupt must clear goal mode"
    );
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("goal mode stopped"), "{notice}");
}

#[test]
fn test_empty_composer_placeholder_reserves_cursor_cell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);
    assert_eq!(plain, ">  Ask, plan, or build · / for commands");
}

#[test]
fn test_overlay_hides_input_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.should_show_input_cursor());

    app.overlay = Overlay::Picker(Box::new(PickerState::loading(
        "Select model",
        String::new(),
        PickerKind::Model {
            target: ModelSelectionTarget::CurrentChat,
            auto_accept_exact: false,
        },
    )));

    assert!(!app.should_show_input_cursor());
}

#[test]
fn test_persisted_draft_history_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("chat_history");
    let history = vec![
        DraftHistoryEntry {
            cwd: "/work/a".to_string(),
            text: "first".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/b".to_string(),
            text: "second".to_string(),
        },
    ];

    save_persisted_draft_history_to_path(&path, &history).unwrap();

    let loaded = load_persisted_draft_history_from_path(&path);
    let pairs: Vec<(String, String)> = loaded
        .into_iter()
        .map(|entry| (entry.cwd, entry.text))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("/work/a".to_string(), "first".to_string()),
            ("/work/b".to_string(), "second".to_string()),
        ]
    );
}

#[test]
fn test_draft_history_view_filters_by_cwd() {
    let all = vec![
        DraftHistoryEntry {
            cwd: String::new(),
            text: "legacy".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/a".to_string(),
            text: "in-a".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/b".to_string(),
            text: "in-b".to_string(),
        },
    ];

    // Current dir's entries plus the legacy fallback; other dirs filtered out.
    assert_eq!(
        draft_history_view(&all, "/work/a"),
        vec!["legacy".to_string(), "in-a".to_string()]
    );
    assert_eq!(
        draft_history_view(&all, "/work/b"),
        vec!["legacy".to_string(), "in-b".to_string()]
    );
    // A fresh dir sees only the legacy fallback.
    assert_eq!(
        draft_history_view(&all, "/work/new"),
        vec!["legacy".to_string()]
    );
}

#[test]
fn test_legacy_plaintext_history_loads_untagged() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("chat_history");
    // The old writer encrypted raw newline-joined prompt lines (no JSON).
    let blob = crate::services::session_store::encrypt("old one\nold two").unwrap();
    std::fs::write(&path, blob).unwrap();

    let loaded = load_persisted_draft_history_from_path(&path);
    assert_eq!(loaded.len(), 2);
    assert!(loaded.iter().all(|entry| entry.cwd.is_empty()));
    assert_eq!(loaded[0].text, "old one");
    assert_eq!(loaded[1].text, "old two");
    // Untagged entries surface in every dir's view.
    assert_eq!(
        draft_history_view(&loaded, "/anywhere"),
        vec!["old one".to_string(), "old two".to_string()]
    );
}

#[test]
fn test_session_preview_uses_last_user_message() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: session_title_from_messages(
            &[
                ChatMessage {
                    model: None,
                    role: "assistant".to_string(),
                    content: "Hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "What is the deployment status for api gateway?".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
            "claude",
        ),
        preview_text: "What is the deployment status for api gateway?".to_string(),
    };

    assert_eq!(
        preview.title,
        "What is the deployment status for api gateway?".to_string()
    );
}

#[test]
fn test_session_preview_text_uses_two_latest_turns() {
    let preview = session_preview_text_from_messages(
        &[
            ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
            ChatMessage {
                model: None,
                role: "assistant".to_string(),
                content: "hi there".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
        ],
        "claude",
    );

    assert_eq!(preview, "hello · hi there");
}

#[test]
fn test_resume_metadata_spans_drop_labels_and_id() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };

    let plain = plain_text_from_spans(&resume_metadata_spans(&preview, 40));
    assert!(plain.contains("2h"));
    assert!(plain.contains("prod"));
    assert!(plain.contains("claude"));
    assert!(!plain.contains("time"));
    assert!(!plain.contains("key"));
    assert!(!plain.contains("model"));
    assert!(!plain.contains("session-1"));
}

#[test]
fn test_session_picker_item_line_shows_two_turn_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text:
            "What is the deployment status for api gateway after the canary rollout finished?"
                .to_string(),
    };

    let lines = session_picker_item_lines(&preview, false, false, 32);
    let first = plain_text_from_spans(&lines[0].spans);

    assert!(first.contains("What is"));
    assert!(first.chars().any(|ch| ch.is_ascii_digit()));
    assert!(!first.contains("key"));
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

fn one_user_message(content: &str) -> Vec<crate::services::session_store::StoredChatMessage> {
    vec![crate::services::session_store::StoredChatMessage {
        model: None,
        role: "user".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        id: None,
        timestamp: None,
        attachments: None,
    }]
}

#[tokio::test]
async fn test_empty_chat_persists_no_session_on_exit() {
    // Opening `aivo code` and leaving without saying anything must NOT create a
    // session — `flush_for_exit` only persists a non-empty history, so an
    // untouched chat leaves the resume list untouched.
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
    app.session_id = "untouched-sess".to_string();
    app.raw_model = "claude".to_string();
    assert!(app.history.is_empty());

    app.flush_for_exit().await;

    assert_eq!(store.count_chat_sessions().await, 0);
    assert!(
        store
            .get_code_session("untouched-sess")
            .await
            .unwrap()
            .is_none()
    );
    // Nothing to resume, so no exit hint either.
    assert_eq!(app.resumable_session_id(), None);
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

/// `session_tokens` (the running per-session total folded from each turn) is
/// written into the chat index entry, so `aivo stats --since` can attribute
/// windowed chat usage; and resuming re-seeds the running total from it.
#[tokio::test]
async fn test_persist_history_writes_session_tokens_to_index() {
    use crate::services::session_store::SessionTokens;

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
    app.session_id = "tok-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // A turn folded this much real usage into the session total.
    app.session_tokens = SessionTokens {
        prompt_tokens: 100,
        completion_tokens: 20,
        cache_read_tokens: 40,
        cache_write_tokens: 0,
    };
    app.persist_history().await.unwrap();

    // The windowed aggregation (what `aivo stats --since` reads) now sees them.
    let far_past = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let window = store.aggregate_chat_window_since(far_past).await;
    let total = window.total();
    assert_eq!(total.prompt_tokens, 100);
    assert_eq!(total.completion_tokens, 20);
    assert_eq!(total.cache_read_tokens, 40);

    // The getter that re-seeds the running total on resume returns the same.
    let seeded = store.chat_session_tokens("tok-sess").await;
    assert_eq!(seeded, app.session_tokens);
    assert_eq!(
        store.chat_session_tokens("nope").await,
        SessionTokens::default()
    );
}

#[tokio::test]
async fn test_log_agent_turn_records_under_real_cwd() {
    use crate::services::log_store::LogQuery;

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
    app.cwd = "/tmp/aivo-chat-1".to_string();
    app.real_cwd = "/home/me/proj".to_string();
    app.session_id = "agent-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "do the thing".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.log_agent_turn(1234).await;

    // The turn shows in `aivo logs` filtered to the real project dir.
    let rows = store
        .logs()
        .list(LogQuery {
            limit: 100,
            cwd: Some("/home/me/proj".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, "code_turn");
    assert_eq!(rows[0].session_id.as_deref(), Some("agent-sess"));
    assert_eq!(rows[0].output_tokens, Some(1234));
}

#[tokio::test]
async fn test_flush_for_exit_persists_partial_response_when_streaming() {
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
    app.session_id = "exit-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Once upon a time".to_string();

    app.flush_for_exit().await;

    let saved = store
        .get_code_session("exit-session")
        .await
        .unwrap()
        .expect("session should be persisted on exit");
    let messages = saved.messages;
    assert_eq!(messages.len(), 2, "user prompt + partial reply should save");
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, "Once upon a time");
}

#[tokio::test]
async fn test_flush_for_exit_persists_user_only_history() {
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
    app.session_id = "user-only-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.flush_for_exit().await;

    let saved = store
        .get_code_session("user-only-session")
        .await
        .unwrap()
        .expect("session with only a user message should still persist on exit");
    let messages = saved.messages;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
}

#[tokio::test]
async fn test_flush_for_exit_skips_persist_for_empty_history() {
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
    app.session_id = "empty-session".to_string();
    app.raw_model = "claude".to_string();

    app.flush_for_exit().await;

    let saved = store.get_code_session("empty-session").await.unwrap();
    assert!(
        saved.is_none(),
        "empty history should not produce a session"
    );
}

#[tokio::test]
async fn test_apply_model_updates_last_selection_preserving_tool() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Seed a prior launchable selection so we can assert the tool is preserved
    // (a `/model` switch must not overwrite it with "code").
    store
        .set_last_selection(&key, "claude", Some("old-model"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("new-model".to_string()).await.unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_id);
    assert_eq!(sel.model.as_deref(), Some("new-model"));
    assert_eq!(sel.tool, "claude", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_complete_key_switch_updates_last_selection() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();
    store
        .set_last_selection(&key_a_full, "codex", Some("model-a"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_b_id, "switched-to key must be selected");
    assert_eq!(sel.model.as_deref(), Some("model-b"));
    assert_eq!(sel.tool, "codex", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_complete_key_switch_same_provider_preserves_chat() {
    // Same base_url = credential swap → chat survives.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("personal", "https://same.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("work", "https://same.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "keep-me".to_string();
    seed_two_exchanges(&mut app);

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    assert_eq!(app.key.id, key_b_id, "switched to the new key");
    assert_eq!(
        app.session_id, "keep-me",
        "same-provider switch keeps the session"
    );
    assert_eq!(app.history.len(), 4, "conversation is preserved");
}

#[tokio::test]
async fn test_complete_key_switch_different_provider_resets_chat() {
    // Different base_url = different wire format → fresh session.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "old-session".to_string();
    seed_two_exchanges(&mut app);

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    assert_eq!(app.key.id, key_b_id, "switched to the new key");
    assert!(
        app.history.is_empty(),
        "different-provider switch resets the chat"
    );
    assert_ne!(
        app.session_id, "old-session",
        "a fresh session id is minted"
    );
}

#[tokio::test]
async fn test_begin_key_switch_confirms_before_provider_reset() {
    // Different provider + live chat → arm the y/n card, don't switch yet.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "old-session".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();

    assert!(
        app.pending_key_switch.is_some(),
        "the switch is armed, not applied"
    );
    assert_eq!(
        app.key.id, key_a,
        "still on the original key until confirmed"
    );
    assert_eq!(app.history.len(), 4, "conversation untouched while armed");
}

#[tokio::test]
async fn test_key_switch_confirm_yes_resets_chat() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    store.set_code_model(&key_b_id, "model-b").await.unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "old-session".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();
    app.handle_key_switch_confirm_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.pending_key_switch.is_none(), "card cleared");
    assert_eq!(app.key.id, key_b_id, "confirm applies the switch");
    assert!(app.history.is_empty(), "confirm resets the chat");
    assert_ne!(app.session_id, "old-session", "fresh session id");
}

#[tokio::test]
async fn test_key_switch_confirm_no_keeps_chat() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "old-session".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();
    app.handle_key_switch_confirm_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(
        app.pending_key_switch.is_none(),
        "declining clears the card"
    );
    assert_eq!(app.key.id, key_a, "declining keeps the current key");
    assert_eq!(app.history.len(), 4, "declining preserves the conversation");
    assert_eq!(app.session_id, "old-session", "same session id");
}

#[tokio::test]
async fn test_begin_key_switch_same_provider_skips_confirm() {
    // Same provider = credential swap: apply straight through, no card.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("personal", "https://same.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("work", "https://same.example.com", None, "sk-b")
        .await
        .unwrap();
    store.set_code_model(&key_b_id, "model-b").await.unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "keep-me".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();

    assert!(
        app.pending_key_switch.is_none(),
        "same-provider switch needs no confirm"
    );
    assert_eq!(app.key.id, key_b_id, "applied directly");
    assert_eq!(app.session_id, "keep-me", "chat preserved");
    assert_eq!(app.history.len(), 4);
}

#[tokio::test]
async fn test_apply_model_survives_resolved_sentinel_base_url() {
    // The live key may carry a base_url resolved away from a sentinel (ollama,
    // aivo-starter). The persisted selection must use the *stored* key's
    // base_url, or `get_last_selection` prunes it as stale.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("ollama", "ollama", None, "")
        .await
        .unwrap();
    let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Simulate the launch-time sentinel resolution that mutates the live key.
    key.base_url = "http://localhost:11434/v1".to_string();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("llama3".to_string()).await.unwrap();

    let sel = store
        .get_last_selection()
        .await
        .unwrap()
        .expect("selection must survive the sentinel/resolved base_url mismatch");
    assert_eq!(sel.key_id, key_id);
    assert_eq!(
        sel.base_url, "ollama",
        "stored sentinel base_url is persisted"
    );
    assert_eq!(sel.model.as_deref(), Some("llama3"));
}

#[tokio::test]
async fn test_apply_model_skips_last_selection_for_hf_synthetic_key() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = ApiKey::new_with_protocol(
        crate::services::huggingface::HF_LOCAL_KEY_ID.to_string(),
        "hf:demo".to_string(),
        "http://localhost:8080/v1".to_string(),
        None,
        "huggingface".to_string(),
    );

    app.apply_model("hf-model".to_string()).await.unwrap();

    assert!(
        store.get_last_selection().await.unwrap().is_none(),
        "ephemeral HF synthetic key must not be remembered as the selection"
    );
}

/// A permission card must not steal in-flight composer typing. With a queued
/// draft in progress the single-letter decision keys (y/a/n) belong to that
/// message — they type into the draft and leave the card up, so a stray
/// keystroke can't approve a tool by accident. Esc still resolves the card.
#[tokio::test]
async fn test_permission_card_keys_do_not_decide_while_composing() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The user is mid-composing a queued message when a card appears.
    app.draft = "deplo".to_string();
    app.cursor = app.draft.len();
    let (reply, mut decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // 'y' is part of the word "deploy", not an approval.
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "deploy", "the keystroke typed into the draft");
    assert!(
        app.agent_permission.is_some(),
        "the card stays up — the key did not decide"
    );
    assert!(
        matches!(
            decision_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ),
        "no decision was sent to the waiting engine"
    );

    // Esc always denies — it can't be message content.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "Esc resolves the card");
    assert_eq!(decision_rx.await.unwrap(), Decision::Deny);
}

/// With an empty composer (the idle case) the quick single-key decisions still
/// work — 'y' approves immediately.
#[tokio::test]
async fn test_permission_card_y_approves_with_empty_composer() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.draft.is_empty());

    let (reply, decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "the card is resolved");
    assert_eq!(decision_rx.await.unwrap(), Decision::Allow);
}

/// The card swaps its keycap row for a hint while a draft is in progress, so the
/// user knows y/a/n won't decide and how to respond without losing the message.
#[test]
fn test_permission_card_shows_composing_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "next message".to_string();
    app.cursor = app.draft.len();
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("type into your message"),
        "composing hint missing:\n{screen}"
    );
}

/// A Cursor card's "always" is session-wide auto-approve, unlike the native
/// engine's per-action "always"; the keycap label spells that out so the broader
/// scope isn't a surprise.
#[test]
fn test_permission_card_cursor_always_label_says_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("edit main.rs".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always (this session)"),
        "cursor 'always' must disclose its session-wide scope:\n{screen}"
    );
}

/// The native engine's "always" stays scoped to this action, so its label is the
/// plain "always" with no session-wide qualifier.
#[test]
fn test_permission_card_native_always_label_is_plain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always"),
        "native card still offers 'always':\n{screen}"
    );
    assert!(
        !screen.contains("this session"),
        "native 'always' is scoped, not session-wide:\n{screen}"
    );
}

fn ask_options(labels: &[&str]) -> Vec<crate::agent::ask::AskOption> {
    labels
        .iter()
        .map(|l| crate::agent::ask::AskOption {
            label: (*l).to_string(),
            description: None,
        })
        .collect()
}

/// The `ask_user` card: ↓ moves the highlight, Enter picks it, and the chosen
/// label is sent back to the waiting engine task as the answer.
#[tokio::test]
async fn test_ask_card_arrow_then_enter_selects_option() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Add release notes now?".to_string(),
        options: ask_options(&["Yes, I'll write them", "You add them", "No, auto-generate"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none(), "answering resolves the card");
    assert_eq!(answer_rx.await.unwrap(), Ok("You add them".to_string()));
}

/// A digit key jumps straight to that option and picks it (numbered-menu style).
#[tokio::test]
async fn test_ask_card_digit_picks_option() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Pick one".to_string(),
        options: ask_options(&["alpha", "beta", "gamma"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none());
    assert_eq!(answer_rx.await.unwrap(), Ok("gamma".to_string()));
}

/// With free text allowed, typing an answer and pressing Enter submits the draft
/// as the answer (rather than queueing it as a chat message).
#[tokio::test]
async fn test_ask_card_free_text_answer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which version?".to_string(),
        options: ask_options(&["patch", "minor"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.draft = "0.36.0".to_string();
    app.cursor = app.draft.len();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none());
    assert!(app.draft.is_empty(), "the draft is consumed as the answer");
    assert_eq!(answer_rx.await.unwrap(), Ok("0.36.0".to_string()));
}

/// Esc dismisses the card; the engine's `ask_user` future then resolves to an
/// error it surfaces as the tool result.
#[tokio::test]
async fn test_ask_card_esc_dismisses() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Proceed?".to_string(),
        options: ask_options(&["Yes", "No"]),
        allow_free_text: false,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none(), "Esc resolves the card");
    assert!(answer_rx.await.unwrap().is_err());
}

/// The card renders the question, the numbered options, and the nav hint.
#[test]
fn test_ask_card_renders_question_and_options() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Add release notes now?".to_string(),
        options: ask_options(&["You add them", "Auto-generate"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });
    // A transcript pushes the composer to the bottom so the floating card has room
    // above it to render its own key-hint line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("Add release notes now?"),
        "question missing:\n{screen}"
    );
    assert!(screen.contains("You add them"), "option missing:\n{screen}");
    assert!(screen.contains("select"), "nav hint missing:\n{screen}");
}

/// Multi-select: `space` toggles the highlighted box, arrows move, and Enter
/// returns the checked labels joined by ", " (not just the highlighted one).
#[tokio::test]
async fn test_ask_card_multi_select_space_toggles_and_enter_joins() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which checks?".to_string(),
        options: ask_options(&["fmt", "clippy", "test"]),
        allow_free_text: false,
        multi_select: true,
        checked: vec![false; 3],
        selected: 0,
        reply,
    });

    // Check fmt via space, move down twice, check test.
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_ask.is_some(), "space toggles without confirming");
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(
        app.agent_ask.is_none(),
        "Enter resolves the multi-select card"
    );
    assert_eq!(answer_rx.await.unwrap(), Ok("fmt, test".to_string()));
}

/// Multi-select renders checkboxes and the "toggle" hint, and a digit toggles the
/// matching box rather than immediately submitting.
#[tokio::test]
async fn test_ask_card_multi_select_digit_toggles_box() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which checks?".to_string(),
        options: ask_options(&["fmt", "clippy", "test"]),
        allow_free_text: false,
        multi_select: true,
        checked: vec![false; 3],
        selected: 0,
        reply,
    });

    // A transcript pushes the composer to the bottom so the card has room to show
    // its own key-hint line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("[ ]"), "unchecked boxes render:\n{screen}");
    assert!(
        screen.contains("toggle"),
        "multi-select hint missing:\n{screen}"
    );

    // Digit 2 toggles clippy (index 1) without submitting. Assert on state: a
    // slim card can scroll the second option out of view.
    app.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_ask.is_some(),
        "a digit toggles, it does not submit"
    );
    assert_eq!(
        app.agent_ask.as_ref().unwrap().checked,
        vec![false, true, false],
        "digit 2 checks the second box"
    );

    // Toggle the always-visible first option so a [✓] is on screen regardless
    // of card height.
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE))
        .await
        .unwrap();
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("[✓]"),
        "a toggled box shows checked:\n{screen}"
    );

    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(answer_rx.await.unwrap(), Ok("fmt, clippy".to_string()));
}

/// The edit-review card renders the heading, the per-file diff, and the y/n keys.
#[test]
fn test_review_card_renders_diff_and_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let items = vec![crate::agent::review::review_item(
        0,
        "edit_file",
        &serde_json::json!({
            "path": "src/lib.rs",
            "old_string": "let x = 1;",
            "new_string": "let x = 2;",
        }),
    )];
    let body = super::render::review_body_lines(&items, std::path::Path::new("."));
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    // A transcript pushes the composer to the bottom so the card has room to show
    // its own y/n key line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 80, 24);
    assert!(
        screen.contains("review 1 edit"),
        "heading missing:\n{screen}"
    );
    assert!(
        screen.contains("src/lib.rs"),
        "file header missing:\n{screen}"
    );
    assert!(
        screen.contains("approve") && screen.contains("reject"),
        "y/n hints missing:\n{screen}"
    );
}

/// `y`/Enter approve the batch, `n`/Esc reject it — each resolves the card and
/// sends the verdict to the waiting engine task.
#[tokio::test]
async fn test_review_card_keys_resolve_decision() {
    use crate::agent::review::ReviewDecision;
    let cases: [(KeyCode, ReviewDecision); 4] = [
        (KeyCode::Char('y'), ReviewDecision::ApproveAll),
        (KeyCode::Enter, ReviewDecision::ApproveAll),
        (KeyCode::Char('n'), ReviewDecision::Reject),
        (KeyCode::Esc, ReviewDecision::Reject),
    ];
    for (code, expected) in cases {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        let (reply, decision_rx) = tokio::sync::oneshot::channel::<ReviewDecision>();
        app.agent_review = Some(PendingReview {
            count: 1,
            body: vec![ratatui::text::Line::from("diff")],
            scroll: 0,
            reply,
        });
        app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
            .await
            .unwrap();
        assert!(app.agent_review.is_none(), "{code:?} resolves the card");
        assert_eq!(decision_rx.await.unwrap(), expected, "{code:?}");
    }
}

/// Arrow keys scroll the review body without resolving the card.
#[tokio::test]
async fn test_review_card_arrows_scroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..10)
        .map(|i| ratatui::text::Line::from(format!("line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.agent_review.as_ref().unwrap().scroll, 2, "down scrolls");
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(
        app.agent_review.as_ref().unwrap().scroll,
        1,
        "up scrolls back"
    );
    assert!(app.agent_review.is_some(), "scrolling does not resolve");
}

/// History pushes the composer to the bottom, giving cards a real viewport.
fn push_review_test_history(app: &mut CodeTuiApp) {
    for _ in 0..4 {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: "working on it".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

/// A diff taller than the screen must not push the y/n keys off the card.
#[test]
fn test_review_card_long_diff_keeps_keys_visible() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    push_review_test_history(&mut app);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..100)
        .map(|i| ratatui::text::Line::from(format!("diff line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 80, 24);
    assert!(
        screen.contains("approve") && screen.contains("reject"),
        "y/n hints clipped off the overflowing card:\n{screen}"
    );
    assert!(
        screen.contains("more (↑↓ scroll)"),
        "overflow marker missing:\n{screen}"
    );
}

/// Overscrolling past the bottom is a no-op — the card height never changes.
#[tokio::test]
async fn test_review_card_overscroll_clamps_stable_height() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    push_review_test_history(&mut app);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..100)
        .map(|i| ratatui::text::Line::from(format!("diff line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    for _ in 0..200 {
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .await
            .unwrap();
    }
    let (bottom, _rows) = render_full_screen(&mut app, 80, 24);
    let clamped = app.agent_review.as_ref().unwrap().scroll;
    assert!(
        usize::from(clamped) < 99,
        "scroll clamps to the last page, not the last line (got {clamped})"
    );
    assert!(
        bottom.contains("end of diff") && bottom.contains("approve"),
        "bottom keeps the marker row and keys:\n{bottom}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    let (after, _rows) = render_full_screen(&mut app, 80, 24);
    assert_eq!(bottom, after, "scrolling past the bottom changes nothing");
}

/// Messages submitted while a turn is in flight queue in order — a second one
/// must not silently clobber the first (the old single-slot behavior).
#[tokio::test]
async fn test_queued_messages_fifo_no_clobber() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    app.draft = "first".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    app.draft = "second".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    assert_eq!(
        app.queued_messages,
        vec!["first".to_string(), "second".to_string()],
        "both messages are queued in submit order"
    );
    let (_lvl, notice) = app.notice.clone().expect("a queued notice");
    assert!(
        notice.contains("2 waiting"),
        "the notice reflects the queue count: {notice}"
    );
}

/// The `/` menu (dropdown + inline hint) stays available while a turn is in
/// flight — it used to be blanket-hidden by `is_busy()`, which made slash
/// commands look dead mid-turn.
#[test]
fn test_command_menu_visible_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    let menu = app.visible_command_menu().expect("menu shows mid-turn");
    assert!(!menu.entries.is_empty(), "matching commands are listed");
}

/// While the `/` menu is open mid-turn, ↑/↓ navigate the menu instead of
/// scrolling the transcript (which owns bare arrows during a turn otherwise).
#[tokio::test]
async fn test_arrows_navigate_menu_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_some());
    assert_eq!(app.command_menu.selected, 0);

    app.handle_key(KeyEvent::from(KeyCode::Down)).await.unwrap();
    assert_eq!(
        app.command_menu.selected, 1,
        "Down moves the menu selection, not the transcript"
    );
    app.handle_key(KeyEvent::from(KeyCode::Up)).await.unwrap();
    assert_eq!(app.command_menu.selected, 0);
}

/// Commands that need the engine idle queue mid-turn (instead of refusing) and
/// run when the turn finishes.
#[tokio::test]
async fn test_engine_idle_commands_queue_and_drain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    app.open_rewind_picker().await.unwrap();
    assert_eq!(app.queued_commands, vec![SlashCommand::Rewind]);
    let (_lvl, notice) = app.notice.clone().expect("a queued notice");
    assert!(
        notice.contains("/rewind queued"),
        "the notice names the queued command: {notice}"
    );

    app.sending = false;
    app.drain_queued_commands().await;
    assert!(app.queued_commands.is_empty(), "the queue drained");
    // With no history there is nothing to rewind to — the command still ran.
    let (_lvl, notice) = app.notice.clone().expect("the drained command's notice");
    assert!(notice.contains("Nothing to rewind"), "{notice}");
}

/// A queued command is dropped by an interrupt/cancel, like queued messages.
#[tokio::test]
async fn test_queued_commands_cleared_on_cancel() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.run_compact_command(true).await;
    assert_eq!(
        app.queued_commands,
        vec![SlashCommand::Compact { fast: true }]
    );

    app.cancel_inflight_request(CancelKind::Discard);
    assert!(app.queued_commands.is_empty());
}

#[tokio::test]
async fn test_queue_focus_entered_by_up_on_empty_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["first".to_string(), "second".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1), "newest row selected");

    // A non-empty draft blocks entry.
    app.queue_focus = None;
    app.draft = "typing".to_string();
    app.cursor = app.draft.len();
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert_eq!(app.draft, "typing");
}

#[tokio::test]
async fn test_queue_focus_selection_and_down_past_end_exits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(2));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(2));
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None, "↓ past the last row exits");
}

#[tokio::test]
async fn test_queue_focus_enter_recalls_message_into_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["fix login".to_string(), "run tests".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "run tests");
    assert_eq!(app.cursor, app.draft.len());
    assert_eq!(app.queued_messages, vec!["fix login".to_string()]);
    assert_eq!(app.queue_focus, None);
}

#[tokio::test]
async fn test_queue_focus_recalls_steering_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .push("steer it".to_string());

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "steer it");
    assert!(app.steering_queue.lock().unwrap().is_empty());
}

/// Ops on a steering row the engine drained mid-event fail gracefully.
#[test]
fn test_queue_row_ops_validate_after_steering_drain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.steering_queue.lock().unwrap().push("steer".to_string());
    let rows = app.queued_rows();
    assert_eq!(rows.len(), 1);

    // Simulate the engine draining the batch between snapshot and op.
    app.steering_queue.lock().unwrap().clear();
    assert!(!app.queue_row_remove(&rows[0]));
    assert!(app.queue_row_recall(&rows[0]).is_none());
    assert!(!app.queue_row_move(&rows[0], 1));
}

#[tokio::test]
async fn test_queue_focus_delete_clamps_and_exits_when_empty() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string(), "b".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
    assert_eq!(app.queue_focus, Some(0), "selection clamps to the survivor");
    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.queued_messages.is_empty());
    assert_eq!(app.queue_focus, None, "focus exits with the last row");
}

/// Reorder stays within a segment — delivery semantics differ across them.
#[tokio::test]
async fn test_queue_focus_reorder_within_segment_not_across() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .extend(["s1".to_string(), "s2".to_string()]);
    app.queued_commands.push(SlashCommand::Rewind);
    app.queued_messages = vec!["m1".to_string(), "m2".to_string()];
    // Unified rows: [s1, s2, /rewind, m1, m2].

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(4)); // m2
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m2".to_string(), "m1".to_string()]
    );
    assert_eq!(app.queue_focus, Some(3), "selection follows the moved row");

    // No crossing into the command segment.
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m2".to_string(), "m1".to_string()]
    );
    assert_eq!(app.queued_commands, vec![SlashCommand::Rewind]);
    assert_eq!(app.queue_focus, Some(3));

    // Shift is the alias for terminals that don't deliver Alt+arrows.
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m1".to_string(), "m2".to_string()]
    );
    assert_eq!(app.queue_focus, Some(4));
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m1".to_string(), "m2".to_string()]
    );
    assert_eq!(app.queue_focus, Some(4));

    app.queue_focus = Some(1); // s2
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        *app.steering_queue.lock().unwrap(),
        vec!["s2".to_string(), "s1".to_string()]
    );
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        *app.steering_queue.lock().unwrap(),
        vec!["s2".to_string(), "s1".to_string()]
    );
}

#[tokio::test]
async fn test_queue_focus_esc_exits_without_interrupting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert!(app.sending, "Esc in focus mode must not interrupt the turn");
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
}

#[tokio::test]
async fn test_queue_focus_typing_char_exits_and_inserts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert_eq!(app.draft, "h");
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
}

#[test]
fn test_command_recall_text_round_trips() {
    for cmd in [
        SlashCommand::Compact { fast: false },
        SlashCommand::Compact { fast: true },
        SlashCommand::Rewind,
        SlashCommand::Review(None),
        SlashCommand::Review(Some("main".to_string())),
        SlashCommand::Goal(Some("ship the fix".to_string())),
        SlashCommand::Plan(Some("go".to_string())),
    ] {
        let text = queue_impl::command_recall_text(&cmd);
        assert!(text.starts_with('/'), "{text}");
        assert_eq!(parse_slash_command(&text[1..]).unwrap(), cmd, "{text}");
    }
    // Skills aren't in the static parser; they resolve by name at submit time.
    assert_eq!(
        queue_impl::command_recall_text(&SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: Some("this repo".to_string()),
        }),
        "/repo-study this repo"
    );
}

#[test]
fn test_queued_panel_rows_render_with_cap_and_more() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .push("steer msg".to_string());
    app.queued_commands.push(SlashCommand::Rewind);
    app.queued_messages.push("plain msg".to_string());
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("» steer msg"), "steering row:\n{screen}");
    assert!(screen.contains("/rewind"), "command row:\n{screen}");
    assert!(screen.contains("· plain msg"), "message row:\n{screen}");

    // An expanded skill body renders as its compact /name form.
    app.queued_messages.push(
        "Use the \"my-skill\" skill. Follow these instructions:\n\nLong body.\n\nInput: hello"
            .to_string(),
    );
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("/my-skill hello"), "{screen}");
    assert!(!screen.contains("Follow these instructions"), "{screen}");

    // Overflow: 7 messages cap at QUEUE_PANEL_MAX_ROWS + an indicator.
    app.steering_queue.lock().unwrap().clear();
    app.queued_commands.clear();
    app.queued_messages = (1..=7).map(|i| format!("q{i}")).collect();
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("· q1") && screen.contains("· q5"),
        "{screen}"
    );
    assert!(!screen.contains("· q6"), "{screen}");
    assert!(screen.contains("… +2 more"), "{screen}");
    assert!(
        !screen.contains("enter edit"),
        "no hint unfocused:\n{screen}"
    );

    // Focused on the newest row: the window follows the selection.
    app.queue_focus = Some(6);
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("▸ · q7"), "{screen}");
    assert!(screen.contains("… +2 earlier"), "{screen}");
    assert!(screen.contains("enter edit · ctrl+d remove"), "{screen}");
}

#[test]
fn test_queue_focus_cleared_by_discard_queued_input() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages = vec!["a".to_string()];
    app.queue_focus = Some(0);
    app.discard_queued_input();
    assert_eq!(app.queue_focus, None);
}

/// `/mcp` Ctrl+D arms a two-press delete (removal edits the user mcp.json), the
/// same confirm as /skills and the resume picker — the first press only arms and
/// surfaces a confirm prompt; Esc disarms without closing the overlay.
#[tokio::test]
async fn test_mcp_delete_arms_then_esc_disarms() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, Some(0), "first Ctrl+D arms"),
        _ => panic!("the overlay must stay open after the first Ctrl+D"),
    }
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("confirm"),
        "an armed delete shows a confirm prompt:\n{screen}"
    );

    // Esc cancels the arm but must NOT close the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, None, "Esc disarms"),
        _ => panic!("Esc on an armed delete must not close the overlay"),
    }
}

/// A repo whose .mcp.json defines stdio servers must NOT spawn them silently:
/// `connect_mcp_with_consent` raises a consent card listing the exact commands
/// and leaves the decision Unknown until the user answers.
#[tokio::test]
async fn project_mcp_stdio_raises_consent_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"sh","args":["-c","echo hi"]}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;

    let prompt = app.pending_mcp_consent.as_ref().expect("a consent card");
    assert_eq!(
        prompt.servers,
        vec![("x".to_string(), "sh -c echo hi".to_string())],
        "the card lists the exact command to be run"
    );
    assert_eq!(
        app.project_mcp_consent,
        ProjectMcpConsent::Unknown,
        "no decision until the user answers"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// Denying the consent card holds the servers back for the session and clears
/// the card; nothing is persisted.
#[tokio::test]
async fn project_mcp_consent_deny_holds_back() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "sh -c echo hi".to_string())],
        cwd: ".".to_string(),
        base_disabled: Default::default(),
    });
    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await;
    assert!(app.pending_mcp_consent.is_none(), "the card is cleared");
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Denied);
}

/// "always" approves for this repo: it persists to the per-repo allow-list (so a
/// future session in the same dir skips the card) and clears the card.
#[tokio::test]
async fn project_mcp_consent_always_persists() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-always-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    app.pending_mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "echo".to_string())],
        cwd: cwd.clone(),
        base_disabled: Default::default(),
    });

    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await;
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    assert!(app.pending_mcp_consent.is_none());

    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let digest = project_mcp_digest(&[("x".to_string(), "echo".to_string())]);
    assert!(
        app.session_store
            .get_project_mcp_approved(&dir_key, &digest)
            .await,
        "'always' is persisted to the per-repo allow-list, bound to the server digest"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo already on the per-repo allow-list connects its project stdio servers
/// without re-prompting — the consent is seeded from the persistent store.
#[tokio::test]
async fn project_mcp_preapproved_repo_skips_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-pre-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        // Bogus command: the background connect's spawn fails fast, no real process.
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let servers = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&servers))
        .await
        .unwrap();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_none(),
        "a pre-approved repo doesn't prompt"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo previously approved "always" but whose `.mcp.json` then CHANGES (a
/// different command) re-prompts: the stored approval is bound to the server
/// content digest, so a swapped-in command can't ride the old consent.
#[tokio::test]
async fn project_mcp_changed_config_reprompts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-changed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();

    // Approve the ORIGINAL server set.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let orig = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&orig))
        .await
        .unwrap();

    // The author swaps in a DIFFERENT command — the prior approval must not apply.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_evil_binary_zzz"}}}"#,
    )
    .unwrap();
    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_some(),
        "a changed .mcp.json re-prompts instead of reusing the old approval"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Unknown);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo with no project `.mcp.json` (or only HTTP servers) never raises the
/// consent card — there's no local command to gate.
#[tokio::test]
async fn project_mcp_no_stdio_no_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-http-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"remote":{"url":"https://h/mcp"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_none(),
        "HTTP-only project servers aren't gated (no local exec)"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn test_reframe_image_input_error_leads_with_action() {
    use super::event_loop_impl::reframe_image_input_error;
    // The provider's stable wording is reframed with an actionable first line,
    // keeping the raw envelope below for debuggability.
    let raw = r#"API returned 400 Bad Request — {"error":{"message":"Error from provider: This model does not support image inputs"}}"#;
    let out = reframe_image_input_error(raw.to_string(), "glm-5.2");
    assert!(out.starts_with("glm-5.2 can't read images"), "got: {out}");
    assert!(out.contains("/model"));
    assert!(out.contains(raw), "raw envelope retained");

    // Unrelated errors pass through untouched.
    let other = "API returned 500 Bad Gateway".to_string();
    assert_eq!(reframe_image_input_error(other.clone(), "glm-5.2"), other);
}

#[tokio::test]
async fn test_history_has_image_detects_image_attachment() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.history_has_image(), "empty history has no image");

    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "just text".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "notes.txt".to_string(),
            mime_type: "text/plain".to_string(),
            storage: AttachmentStorage::Inline {
                data: "abc".to_string(),
            },
        }],
    });
    assert!(
        !app.history_has_image(),
        "a text attachment is not an image"
    );

    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });
    assert!(app.history_has_image(), "image attachment detected");
}

#[tokio::test]
async fn test_preflight_refuses_image_on_known_text_only_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "glm-5.1".to_string();
    app.model_image_input = Some(false); // snapshot says text-only
    app.draft_attachments.push(MessageAttachment {
        name: "shot.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    });

    app.dispatch_user_message("what's in this".to_string(), None)
        .await
        .unwrap();

    let (style, msg) = app.notice.clone().expect("a refusal notice is shown");
    assert_eq!(style, ERROR);
    assert!(msg.contains("can't read images"), "got: {msg}");
    // The draft + attachment survive so the user can switch models and resend;
    // nothing was sent.
    assert_eq!(app.draft_attachments.len(), 1, "attachment retained");
    assert!(app.history.is_empty(), "no user turn was pushed");
    assert!(!app.sending, "no turn started");
}

/// A `find .`-sized capture (way past the inline cap) expands to at most
/// `MAX_EXPANDED_OUTPUT_LINES` rendered lines — bounding the O(lines) re-wrap so the
/// UI can't freeze — and notes the remainder instead of rendering it.
#[tokio::test]
async fn output_expand_caps_huge_capture_and_notes_remainder() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 10_000 lines captured this session (held in memory, but bounded by the cap).
    let full: String = (1..=10_000).map(|i| format!("L{i:05}\n")).collect();
    app.record_local_output("find .".to_string(), full, String::new(), 0, false, false);
    let idx = app.history.len() - 1;

    // Memory retention is bounded by the inline cap, not the full 10k capture.
    let kept = app.local_outputs.get(&idx).expect("output retained");
    assert!(
        kept.stdout.lines().count() <= MAX_EXPANDED_OUTPUT_LINES,
        "retained {} lines, expected ≤ {MAX_EXPANDED_OUTPUT_LINES}",
        kept.stdout.lines().count()
    );

    app.expanded_output.insert(idx);
    let mut block = Vec::new();
    render_local_command(
        &mut block,
        &app.history[idx].content,
        OutputView::Expanded {
            full: app.local_outputs.get(&idx),
        },
    );
    // The `! command` header + at most the cap of output rows + the overflow note +
    // the `▾ collapse` toggle — never 10k rows.
    let output_rows = block.iter().filter(|l| l.plain.starts_with("  L")).count();
    assert_eq!(output_rows, MAX_EXPANDED_OUTPUT_LINES);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains(&format!(
            "+{} more lines",
            10_000 - MAX_EXPANDED_OUTPUT_LINES
        )),
        "overflow note counts the un-rendered remainder:\n{rendered}"
    );
    assert!(rendered.contains("too long to show inline"));
    assert!(
        !rendered.contains("L10000"),
        "the tail is not rendered inline"
    );
}

#[test]
fn test_render_output_line_collapses_spinner_frames() {
    // The spinner writes "\r{dim frame} Fetching models..." per frame (no newline
    // between frames), then "\r\x1b[2K" + the result; the PTY hands it over as one line.
    let frame = |glyph: &str| format!("\r\u{1b}[2m{glyph}\u{1b}[0m Fetching models...");
    let mut raw = String::new();
    for glyph in ["⠋", "⠙", "⠹", "⠸", "⠼"] {
        raw.push_str(&frame(glyph));
    }
    raw.push_str("\r\u{1b}[2K✓ 2 models via aivo-starter\r"); // trailing \r = PTY's \r\n
    assert_eq!(render_output_line(&raw), "✓ 2 models via aivo-starter");
}

#[test]
fn test_render_output_line_plain_line_unchanged() {
    // Normal lines pass through; the trailing \r (PTY's \r\n) and colour SGR drop out.
    assert_eq!(render_output_line("hello world\r"), "hello world");
    assert_eq!(render_output_line("  \u{1b}[32mOK\u{1b}[0m"), "  OK");
    assert_eq!(render_output_line("plain"), "plain");
}

#[test]
fn test_render_output_line_progress_bar_keeps_final_state() {
    // A progress bar redraws in place with bare \r; only the last state should show.
    let raw = "[#   ] 25%\r[##  ] 50%\r[####] 100%";
    assert_eq!(render_output_line(raw), "[####] 100%");
}

#[test]
fn test_render_output_line_erase_in_line_drops_stale_tail() {
    // Erase-to-end (`\x1b[K`) must drop the stale tail when a shorter string overwrites.
    assert_eq!(render_output_line("longlabel\rhi\u{1b}[K"), "hi");
    // `\x1b[2K` clears the whole line.
    assert_eq!(render_output_line("spam\r\u{1b}[2Kdone"), "done");
}

#[cfg(unix)]
#[test]
fn test_pty_run_collapses_carriage_return_overwrites() {
    // End-to-end through the real PTY reader: a command that redraws one line with
    // bare \r (like the spinner) must commit only its final state, not every frame.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let shell =
        spawn_local_shell("printf '111\\r222\\r333\\n'", &std::env::temp_dir()).expect("spawn pty");
    run_local_to_completion(shell, tx);

    let mut lines = Vec::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            RuntimeEvent::LocalCommandLine { line, .. } => lines.push(line),
            RuntimeEvent::LocalCommandDone {
                exit_code,
                truncated,
            } => {
                assert_eq!(exit_code, 0, "printf should exit 0");
                assert!(!truncated, "a tiny run is not truncated");
            }
            _ => {}
        }
    }
    assert_eq!(
        lines,
        vec!["333".to_string()],
        "carriage-return overwrites collapse to the final state"
    );
}

// ---- agent session-control tools (switch_model / set_effort) ----

fn model_choice(id: &str) -> ModelChoice {
    ModelChoice {
        id: id.to_string(),
        label: id.to_string(),
    }
}

#[test]
fn resolve_model_request_exact_and_unique_substring() {
    let choices = [
        model_choice("anthropic/claude-opus-4-8"),
        model_choice("openai/gpt-5"),
        model_choice("openai/gpt-5-mini"),
    ];
    // exact id wins even though it's also a substring of another
    assert_eq!(
        super::session_impl::resolve_model_request("OPENAI/GPT-5", &choices).unwrap(),
        "openai/gpt-5"
    );
    assert_eq!(
        super::session_impl::resolve_model_request("opus", &choices).unwrap(),
        "anthropic/claude-opus-4-8"
    );
}

#[test]
fn resolve_model_request_ambiguous_and_missing() {
    let choices = [
        model_choice("openai/gpt-5"),
        model_choice("openai/gpt-5-mini"),
    ];
    // substring of both, no exact "gpt-5" id → ambiguous
    let err = super::session_impl::resolve_model_request("gpt-5", &choices).unwrap_err();
    assert!(err.contains("ambiguous"));
    assert!(err.contains("openai/gpt-5") && err.contains("openai/gpt-5-mini"));
    let miss = super::session_impl::resolve_model_request("llama", &choices).unwrap_err();
    assert!(miss.contains("no model matches") && miss.contains("/model"));
    // empty catalog accepts the raw string
    assert_eq!(
        super::session_impl::resolve_model_request("whatever", &[]).unwrap(),
        "whatever"
    );
}

#[tokio::test]
async fn agent_set_effort_validates_against_levels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "gpt-5".to_string();
    app.model = "gpt-5".to_string();
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];

    let ok = app.agent_set_effort("High".to_string()).await.unwrap();
    assert!(ok.contains("high"));
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    // invalid level rejected, effort unchanged
    let err = app.agent_set_effort("turbo".to_string()).await.unwrap_err();
    assert!(err.contains("low, medium, high"));
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    app.model_reasoning_efforts.clear();
    let none = app.agent_set_effort("high".to_string()).await.unwrap_err();
    assert!(none.contains("no reasoning-effort levels"));
}

#[tokio::test]
async fn agent_switch_model_noops_when_already_on_it() {
    // The already-on-it short-circuit returns before any catalog fetch (no network).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "gpt-5".to_string();
    let msg = app.agent_switch_model("GPT-5".to_string()).await.unwrap();
    assert!(msg.contains("Already using gpt-5"));
}

/// Assistant turns are stamped with their dispatch-time model, and the
/// transcript draws a `model →` divider where the stamp changes.
#[test]
fn model_switch_stamps_turns_and_renders_divider() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Turn 1 dispatched on model-a.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first question".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_model = Some("model-a".to_string());
    // Mid-turn switch: the running turn must keep its dispatch-time stamp.
    app.raw_model = "model-b".to_string();
    app.pending_response = "answer one".to_string();
    app.flush_pending_assistant();
    assert_eq!(
        app.history.last().unwrap().model.as_deref(),
        Some("model-a")
    );

    // Turn 2 dispatched on model-b.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "second question".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_model = Some("model-b".to_string());
    app.pending_response = "answer two".to_string();
    app.flush_pending_assistant();
    assert_eq!(
        app.history.last().unwrap().model.as_deref(),
        Some("model-b")
    );

    let body = app.build_transcript_history_body(80);
    let rows = wrap_transcript(&body.lines, &body.bar_colors, 80).rows;
    // One divider at the boundary; none above the first stamped turn.
    assert_eq!(
        rows.iter()
            .filter(|r| r.contains("model → model-b"))
            .count(),
        1
    );
    assert!(rows.iter().all(|r| !r.contains("model → model-a")));
    let first = rows.iter().position(|r| r.contains("answer one")).unwrap();
    let divider = rows
        .iter()
        .position(|r| r.contains("model → model-b"))
        .unwrap();
    let second = rows.iter().position(|r| r.contains("answer two")).unwrap();
    assert!(first < divider && divider < second);
}

/// Unstamped (pre-feature) history renders no divider.
#[test]
fn unstamped_history_renders_no_model_divider() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for (role, content) in [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let body = app.build_transcript_history_body(80);
    let rows = wrap_transcript(&body.lines, &body.bar_colors, 80).rows;
    assert!(rows.iter().all(|r| !r.contains("model →")));
}

/// Dispatch freezes the selected model into `turn_model`.
#[tokio::test]
async fn test_dispatch_captures_turn_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();
    app.raw_model = "model-a".to_string();

    app.dispatch_user_message("hello".to_string(), None)
        .await
        .unwrap();
    assert_eq!(app.turn_model.as_deref(), Some("model-a"));
}
