use super::super::*;
use super::helpers::*;

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

/// An overflowing `/help` body draws the gutter scrollbar like every other
/// modal, and dragging its thumb pans the scroll offset.
#[tokio::test]
async fn test_help_overlay_overflow_draws_draggable_scrollbar() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    // Short terminal: the body overflows, so the thumb + hit must appear.
    render_full_screen(&mut app, 90, 20);
    let hit = app
        .scrollbar_hit
        .expect("overflowing help body renders a scrollbar");
    let (_, _, _, max_start) = hit.thumb();
    let at = |kind, row| MouseEvent {
        kind,
        column: hit.track.x,
        row,
        modifiers: KeyModifiers::NONE,
    };
    app.handle_mouse(at(MouseEventKind::Down(MouseButton::Left), hit.track.y))
        .await
        .unwrap();
    app.handle_mouse(at(
        MouseEventKind::Drag(MouseButton::Left),
        hit.track.y + hit.track.height - 1,
    ))
    .await
    .unwrap();
    let Overlay::Help { scroll } = app.overlay else {
        panic!("help overlay closed by the drag")
    };
    assert_eq!(
        usize::from(scroll),
        max_start,
        "drag to the bottom = max scroll"
    );

    // A tall render fits everything — no scrollbar, and the offset re-clamps to 0.
    render_full_screen(&mut app, 90, 140);
    assert!(
        app.scrollbar_hit.is_none(),
        "fitting body must not draw a scrollbar"
    );
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
    assert!(app.account.task.is_none());
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
    app.account.generation = 7;
    app.notice = Some((MUTED(), "Starting sign-in…".to_string()));
    assert!(app.account.login.is_none());

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
    // The card lives in its reserved slot directly above the composer.
    assert!(
        frame.find("sign in to aivo").unwrap() < frame.find("Ask, plan, or build").unwrap(),
        "card should sit above the composer:\n{frame}"
    );

    // A prompt stamped with a stale generation is ignored (card stays).
    app.apply_account_login_prompt(3, Err("boom".to_string()));
    assert!(app.account.login.is_some(), "stale error dropped the card");

    // Esc with a non-empty composer belongs to the draft — the card stays.
    app.draft = "half a thought".to_string();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account.login.is_some(), "Esc stole the draft's key");

    // Esc on an empty composer cancels: card gone, generation bumped.
    app.draft.clear();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account.login.is_none());
    assert_ne!(app.account.generation, 7, "cancel must invalidate the flow");

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
    let account_gen = app.account.generation;
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
    app.account.pending_logout = Some("me@example.com".to_string());
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
    assert!(app.account.pending_logout.is_none());
    assert!(app.account.task.is_none(), "deny must not spawn an unlink");

    // A stale unlink result is ignored; the current one lands as a notice.
    let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
    app.cache
        .set(sentinel, vec!["aivo/starter".to_string()])
        .await;
    app.account.generation = 4;
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

#[tokio::test]
async fn test_config_overlay_toggles_thinking() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::Thinking)
        .expect("Thinking row present");
    // `on` is segment 0 — the renderer derives the highlighted pill from this.
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 0);

    // Advancing the switch flips the live flag (off is segment 1).
    app.cycle_config_setting(idx, CycleDir::Enter).await;
    assert!(!app.thinking_enabled);
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 1);
}

#[tokio::test]
async fn test_config_overlay_toggles_inline_images() {
    use crate::services::terminal_graphics::{GraphicsCaps, Protocol};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let detected = GraphicsCaps {
        protocol: Protocol::KittyVirtual,
        ..GraphicsCaps::default()
    };
    // No detected protocol (Windows, unsupported terminal): on warns it's inert.
    assert!(
        app.config_active_help(ConfigSetting::InlineImages)
            .contains("aren't available"),
        "undetected caps must warn that the toggle is inert"
    );
    app.detected_graphics_caps = detected;
    app.inline_images.caps = detected;
    assert!(app.inline_images_enabled, "defaults on");
    assert!(
        app.config_active_help(ConfigSetting::InlineImages)
            .contains("render inline"),
        "detected caps get the normal copy"
    );

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::InlineImages)
        .expect("Inline images row present");
    assert_eq!(app.config_segments(ConfigSetting::InlineImages).active, 0);

    // Off: the event loop finishes the disable with the terminal writer.
    app.cycle_config_setting(idx, CycleDir::Enter).await;
    assert!(!app.inline_images_enabled);
    assert!(app.pending_inline_image_cleanup);
    app.finish_inline_image_disable(&mut Vec::new());
    assert!(!app.inline_images.caps.enabled());
    assert_eq!(app.config_segments(ConfigSetting::InlineImages).active, 1);

    // Back on: the startup detection result is restored, not re-probed.
    app.cycle_config_setting(idx, CycleDir::Enter).await;
    assert!(app.inline_images_enabled);
    assert_eq!(app.inline_images.caps, detected);
}

#[tokio::test]
// Serializing the global theme is the point — the guard must span the awaits.
#[allow(clippy::await_holding_lock)]
async fn test_config_overlay_cycles_theme() {
    let _theme = theme_lock();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert_eq!(app.theme, UiTheme::Dark);
    assert_eq!(ui_theme(), UiTheme::Dark);
    assert_eq!(TEXT(), Palette::DARK.text);

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    assert_eq!(state.items[0].setting, ConfigSetting::Theme);

    app.cycle_config_setting(0, CycleDir::Enter).await;
    assert_eq!(app.theme, UiTheme::Light);
    assert_eq!(ui_theme(), UiTheme::Light);
    assert_eq!(TEXT(), Palette::LIGHT.text);

    // Light mode paints the warm-paper canvas across the whole screen so dark ink
    // stays readable even on a dark terminal; dark mode keeps the terminal's own bg.
    let canvas = Palette::LIGHT.canvas.expect("light theme fills the canvas");
    assert!(
        Palette::DARK.canvas.is_none(),
        "dark theme keeps the terminal bg"
    );
    {
        use ratatui::backend::TestBackend;
        let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        // The floating transcript/overlay regions are `Clear`ed and must be
        // repainted with the canvas, not left on the terminal's native bg — so the
        // paper reaches the interior, not just the uncleared margins. A strong
        // majority of cells should carry the canvas fill.
        let cells = terminal.backend().buffer().content();
        let on_canvas = cells.iter().filter(|c| c.bg == canvas).count();
        assert!(
            on_canvas * 2 > cells.len(),
            "light canvas must fill cleared regions ({on_canvas}/{} cells)",
            cells.len()
        );
    }

    app.cycle_config_setting(0, CycleDir::Enter).await;
    assert_eq!(app.theme, UiTheme::Dark);
    assert_eq!(ui_theme(), UiTheme::Dark);
}

#[test]
fn resolve_startup_theme_stored_choice_else_dark() {
    use crate::services::session_store::ChatTheme;
    assert_eq!(
        resolve_startup_theme(Some(ChatTheme::Light)),
        UiTheme::Light
    );
    assert_eq!(resolve_startup_theme(Some(ChatTheme::Dark)), UiTheme::Dark);
    // Unset (first launch) defaults to dark; /config persists a choice.
    assert_eq!(resolve_startup_theme(None), UiTheme::Dark);
}

#[test]
fn brand_palettes_keep_lime_focus_and_theme_aware_destructive_rows() {
    assert_eq!(Palette::DARK.accent, Color::Rgb(222, 252, 9));
    assert_eq!(Palette::DARK.select_accent, Palette::DARK.accent);
    assert_eq!(Palette::LIGHT.faint, Color::Rgb(122, 114, 102));
    assert_eq!(
        Palette::DARK.text,
        Color::Reset,
        "dark body text inherits the terminal's own foreground"
    );
    assert_eq!(
        Palette::LIGHT.canvas,
        Some(Color::Rgb(237, 233, 226)),
        "light canvas is the brand paper — cream, not white"
    );
    assert_ne!(Palette::DARK.select_bg, Palette::DARK.delete_bg);
    assert_ne!(Palette::LIGHT.select_bg, Palette::LIGHT.delete_bg);
    assert_ne!(Palette::DARK.error, Palette::DARK.info);
    assert_ne!(Palette::LIGHT.error, Palette::LIGHT.info);

    let diff_colors = [
        Palette::DARK.diff_add_bg,
        Palette::DARK.diff_del_bg,
        Palette::DARK.diff_add_fg,
        Palette::DARK.diff_del_fg,
        Palette::DARK.diff_add_sign,
        Palette::DARK.diff_del_sign,
        Palette::DARK.diff_add_hl_bg,
        Palette::DARK.diff_del_hl_bg,
        Palette::LIGHT.diff_add_bg,
        Palette::LIGHT.diff_del_bg,
        Palette::LIGHT.diff_add_fg,
        Palette::LIGHT.diff_del_fg,
        Palette::LIGHT.diff_add_sign,
        Palette::LIGHT.diff_del_sign,
        Palette::LIGHT.diff_add_hl_bg,
        Palette::LIGHT.diff_del_hl_bg,
    ];
    assert!(
        diff_colors
            .iter()
            .all(|color| matches!(color, Color::Indexed(_))),
        "diff colors must avoid truecolor SGR sequences for old terminal compatibility"
    );
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
    assert_eq!(app.config_segments(ConfigSetting::AgentTools).active, 0);

    app.cycle_config_setting(idx, CycleDir::Enter).await;
    assert!(!app.agent_tools_enabled);
    assert_eq!(app.config_segments(ConfigSetting::AgentTools).active, 1);
}

#[tokio::test]
async fn test_config_approval_radio_is_mutually_exclusive() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Three standing modes; plan is a transient state, not a segment here.
    assert_eq!(
        app.config_segments(ConfigSetting::Approval).options,
        &["normal", "auto-approve", "review"]
    );

    // Fresh session: normal mode — segment 0 is live.
    assert!(!app.agent_auto_approve && !app.agent_review_edits && !app.plan_mode);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 0);

    // Auto-approve sets exactly one flag.
    app.set_approval_mode("auto-approve").await;
    assert!(app.agent_auto_approve && !app.agent_review_edits);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 1);

    // Switching to Review clears auto-approve — the fold's whole point.
    app.set_approval_mode("review").await;
    assert!(app.agent_review_edits && !app.agent_auto_approve);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 2);

    // Back to Normal leaves every mode off.
    app.set_approval_mode("normal").await;
    assert!(!app.agent_auto_approve && !app.agent_review_edits && !app.plan_mode);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 0);
}

#[tokio::test]
async fn config_overlay_renders_live_values_and_focused_options() {
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.open_config_overlay();

    let render = |app: &mut CodeTuiApp| {
        // 30 tall fits the full stacked list; shorter terminals scroll with the selection.
        let mut terminal = Terminal::new(TestBackend::new(80, 30)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer.cell((x, y)).map_or(" ", |c| c.symbol()))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Theme focused: live values everywhere, alternatives only on that row.
    let text = render(&mut app);
    for label in ["Theme", "Mode", "dark", "light", "normal", "aivo"] {
        assert!(text.contains(label), "{label:?} missing:\n{text}");
    }
    for hidden in ["review", "auto-approve", "gateway"] {
        assert!(!text.contains(hidden), "{hidden:?} leaked:\n{text}");
    }
    assert!(
        !text.contains("on · Esc"),
        "count badge still drawn:\n{text}"
    );
    assert!(
        text.contains("change"),
        "footer missing change hint:\n{text}"
    );

    let Overlay::Config(state) = &mut app.overlay else {
        panic!("expected config overlay");
    };
    while state.items[state.selected].setting != ConfigSetting::Approval {
        state.select_next();
    }
    let text = render(&mut app);
    for label in ["normal", "auto-approve", "review"] {
        assert!(text.contains(label), "option {label:?} missing:\n{text}");
    }
    assert!(
        text.contains("Risky actions"),
        "active-value copy missing:\n{text}"
    );
}

#[tokio::test]
async fn config_overlay_split_shows_inspector_beside_list() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_config_overlay();
    // 84% of 100 = 84 cols, inner 80 >= SPLIT_MIN_INNER_WIDTH (76).
    let (text, _) = render_full_screen(&mut app, 100, 30);
    assert!(text.contains("│"), "split rule missing:\n{text}");
    assert!(
        text.contains("Dark palette") || text.contains("Applies immediately"),
        "Theme inspector missing on the right:\n{text}"
    );
    assert!(text.contains("Image generation"), "list missing:\n{text}");
    for header in ["Appearance", "Behavior", "Media"] {
        assert!(text.contains(header), "group {header:?} missing:\n{text}");
    }
}

#[tokio::test]
async fn config_overlay_click_selects_list_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_config_overlay();
    let _ = render_full_screen(&mut app, 100, 30);
    let hitbox = app
        .picker_hitbox
        .clone()
        .expect("config list hitbox recorded");
    // Visual rows (split, with group gaps): Appearance, Theme, Inline images,
    // Thinking, Footer tok/s, Footer cache hit, Footer price, blank, Behavior,
    // Mode, Web search, Agent tools, blank, Media, Vision fallback,
    // Image generation.
    let image_row = hitbox.list_area.y + 15;
    app.handle_mouse(left_click(hitbox.list_area.x, image_row))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Config(state) => {
            assert_eq!(state.items[state.selected].setting, ConfigSetting::ImageGen)
        }
        _ => panic!("expected config overlay"),
    }
}

#[tokio::test]
async fn config_overlay_click_header_does_not_change_selection() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_config_overlay();
    let _ = render_full_screen(&mut app, 100, 30);
    let hitbox = app
        .picker_hitbox
        .clone()
        .expect("config list hitbox recorded");
    app.handle_mouse(left_click(hitbox.list_area.x, hitbox.list_area.y))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Config(state) => {
            assert_eq!(state.items[state.selected].setting, ConfigSetting::Theme)
        }
        _ => panic!("expected config overlay"),
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn config_overlay_click_segment_sets_theme() {
    let _theme = theme_lock();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert_eq!(app.theme, UiTheme::Dark);
    app.open_config_overlay();
    let _ = render_full_screen(&mut app, 100, 30);
    let hitbox = app
        .picker_hitbox
        .clone()
        .expect("config segment hits recorded");
    let light = hitbox
        .segment_hits
        .iter()
        .find(|(_, option)| *option == 1)
        .map(|(area, _)| *area)
        .expect("light pill hitbox");
    app.handle_mouse(left_click(light.x + 1, light.y))
        .await
        .unwrap();
    assert_eq!(app.theme, UiTheme::Light);
    // Theme is process-global; put Dark back so later tests see the default.
    app.cycle_config_setting(0, CycleDir::Enter).await;
    assert_eq!(app.theme, UiTheme::Dark);
}

#[tokio::test]
async fn config_overlay_reopens_on_last_setting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_config_overlay();
    if let Overlay::Config(state) = &mut app.overlay {
        state.selected = state
            .items
            .iter()
            .position(|i| i.setting == ConfigSetting::ImageGen)
            .unwrap();
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    app.open_config_overlay();
    match &app.overlay {
        Overlay::Config(state) => {
            assert_eq!(state.items[state.selected].setting, ConfigSetting::ImageGen)
        }
        _ => panic!("expected config overlay"),
    }
}

#[tokio::test]
async fn test_clicking_footer_session_id_opens_overlay() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "import-claude-a1b2c3d4".to_string();

    // Render the footer so the id's click rect is recorded.
    let mut terminal = Terminal::new(TestBackend::new(100, 1)).unwrap();
    terminal
        .draw(|frame| app.render_footer(frame, frame.area()))
        .unwrap();
    let hit = app.session_id_hit.expect("session id click rect recorded");

    // A click inside that rect opens the detail overlay.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: hit.x,
        row: hit.y,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert!(matches!(app.overlay, Overlay::Session { .. }));
}

#[tokio::test]
async fn test_overlay_backdrop_click_dismisses_help() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Help { scroll: 0 };

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let hit = app
        .render_cache
        .overlay_hitbox
        .expect("overlay box recorded");

    // Inside the box the press falls through to text selection — overlay stays.
    app.handle_mouse(left_click(hit.x + hit.width / 2, hit.y + hit.height / 2))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Help { .. }));

    // On the backdrop it dismisses, like Esc.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_overlay_backdrop_click_steps_back_like_esc() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(SkillsOverlay {
        list_scroll: 0,
        scroll_selected: 0,
        items: Vec::new(),
        selected: 0,
        query: "abc".to_string(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    assert!(app.render_cache.overlay_hitbox.is_some());

    // First backdrop press clears the filter (Esc's first stage)…
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    match &app.overlay {
        Overlay::Skills(state) => assert!(state.query.is_empty()),
        _ => panic!("overlay closed too early"),
    }

    // …the next one closes the overlay.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_loading_picker_backdrop_click_closes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Picker(Box::new(PickerState::loading(
        "Select model",
        String::new(),
        PickerKind::Key {
            target: KeySelectionTarget::Switch,
        },
    )));

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let list_area = app
        .picker_hitbox
        .as_ref()
        .expect("loading picker box recorded")
        .list_area;

    // A click on the (empty) list while loading is inert.
    app.handle_mouse(left_click(list_area.x, list_area.y))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Picker(_)));

    // A backdrop click closes it, same as a loaded picker.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_session_overlay_shows_full_id_and_provenance() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "import-claude-a1b2c3d4".to_string();
    app.open_session_overlay();
    assert!(matches!(app.overlay, Overlay::Session { .. }));

    // Tall + wide enough that the whole (short) detail box shows without scrolling
    // and the resume command doesn't wrap.
    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut text = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            text.push_str(buf[(x, y)].symbol());
        }
        text.push('\n');
    }
    // Full id (the footer only had room for a handle), plus the fork provenance
    // and the resume command.
    assert!(text.contains("import-claude-a1b2c3d4"), "overlay:\n{text}");
    assert!(text.contains("Claude (forked)"), "overlay:\n{text}");
    assert!(
        text.contains("aivo code --resume import-claude-a1b2c3d4"),
        "overlay:\n{text}"
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
    // Locate the modal from its rounded title row, then keep only those columns.
    // Other UI (including the composer) can have its own vertical borders.
    let title_row = top.lines().find(|row| row.contains("Agents")).unwrap();
    let left = title_row.chars().position(|c| c == '╭').unwrap();
    let right = title_row.chars().position(|c| c == '╮').unwrap();
    let interior: String = top
        .lines()
        .map(|row| {
            row.chars()
                .skip(left + 1)
                .take(right.saturating_sub(left + 1))
                // The margin-column scrollbar thumb isn't part of the copy.
                .filter(|c| !is_scrollbar_thumb_char(*c))
                .collect::<String>()
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

#[tokio::test]
async fn test_config_overlay_toggles_footer_stats_and_persists() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let tps_idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::FooterTps)
        .expect("Footer tok/s row present");
    let cache_idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::FooterCacheHit)
        .expect("Footer cache hit row present");
    let price_idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::FooterPrice)
        .expect("Footer price row present");

    assert_eq!(app.config_segments(ConfigSetting::FooterTps).active, 0);
    app.cycle_config_setting(tps_idx, CycleDir::Enter).await;
    assert!(!app.footer_tps_enabled);
    app.cycle_config_setting(cache_idx, CycleDir::Enter).await;
    assert!(!app.footer_cache_enabled);
    app.cycle_config_setting(price_idx, CycleDir::Enter).await;
    assert!(!app.footer_price_enabled);

    let toggles = app.session_store.get_chat_toggles().await;
    assert!(!toggles.footer_tps_enabled);
    assert!(!toggles.footer_cache_enabled);
    assert!(!toggles.footer_price_enabled);
}
