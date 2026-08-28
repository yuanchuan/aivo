use super::*;

use std::sync::atomic::{AtomicU8, Ordering};

/// Chat TUI color theme. Persisted in `code-prefs.json` as `"theme": "dark"|"light"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UiTheme {
    Dark = 0,
    Light = 1,
}

impl UiTheme {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    pub(super) fn label(self) -> &'static str {
        self.as_str()
    }
}

static CURRENT_THEME: AtomicU8 = AtomicU8::new(UiTheme::Dark as u8);

/// Install the active theme for color accessors. Call at startup and whenever
/// `/config` cycles the theme so render paths pick up the new palette.
pub(super) fn set_ui_theme(theme: UiTheme) {
    CURRENT_THEME.store(theme as u8, Ordering::Relaxed);
}

pub(super) fn ui_theme() -> UiTheme {
    match CURRENT_THEME.load(Ordering::Relaxed) {
        x if x == UiTheme::Light as u8 => UiTheme::Light,
        _ => UiTheme::Dark,
    }
}

/// Semantic colors for the chat TUI. Neutral grays carry the structure (warm
/// paper neutrals in light); acid lime is reserved for the aivo mark, focus,
/// and active choices.
#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub text: Color,
    pub muted: Color,
    pub faint: Color,
    pub accent: Color,
    pub assistant: Color,
    pub user: Color,
    pub tool: Color,
    pub shell: Color,
    pub link: Color,
    pub quote: Color,
    pub error: Color,
    pub warning: Color,
    pub info: Color,
    pub live: Color,
    pub diff_add_bg: Color,
    pub diff_del_bg: Color,
    pub diff_add_fg: Color,
    pub diff_del_fg: Color,
    pub diff_add_sign: Color,
    pub diff_del_sign: Color,
    pub diff_add_hl_bg: Color,
    pub diff_del_hl_bg: Color,
    pub select_bg: Color,
    pub select_text: Color,
    pub select_accent: Color,
    pub select_wash: Color,
    pub select_flash: Color,
    pub delete_bg: Color,
    pub delete_text: Color,
    pub delete_accent: Color,
    /// Full-screen canvas fill. `None` keeps the terminal's default background
    /// (dark theme); light paints the brand warm paper so dark ink stays readable
    /// even on a dark terminal.
    pub canvas: Option<Color>,
    pub code: Color,
    pub toast_bg: Color,
    pub jump_fg: Color,
    pub jump_bg: Color,
}

impl Palette {
    /// Neutral night: terminal black, the terminal's own ink, and the full
    /// acid-lime mark. Body text is `Reset` so it always matches the host
    /// terminal; safe because dark never paints text over its own canvas
    /// (canvas: None) and every filled bar pairs its own fg (select_text,
    /// jump_fg, delete_text). The gray ramp is desaturated to match Reset's
    /// unknown hue — warmth lives only in the brand accents.
    pub const DARK: Self = Self {
        text: Color::Reset,
        muted: Color::Rgb(156, 156, 156),
        faint: Color::Rgb(106, 106, 106),
        accent: Color::Rgb(222, 252, 9), // #DEFC09
        assistant: Color::Rgb(171, 193, 116),
        user: Color::Rgb(193, 193, 193),
        tool: Color::Rgb(142, 164, 159),
        shell: Color::Rgb(215, 172, 80),
        link: Color::Rgb(143, 178, 222),
        quote: Color::Rgb(147, 147, 147),
        error: Color::Rgb(228, 128, 114),
        warning: Color::Rgb(224, 180, 104),
        info: Color::Rgb(126, 184, 212),
        live: Color::Rgb(232, 96, 92),
        // Diff backgrounds use ANSI 256-color indexes rather than RGB. Some
        // older Terminal.app/tmux combinations misparse `48;2;r;g;b` and
        // leave the entire row with a green background.
        diff_add_bg: Color::Indexed(22),
        diff_del_bg: Color::Indexed(52),
        diff_add_fg: Color::Indexed(151),
        diff_del_fg: Color::Indexed(180),
        diff_add_sign: Color::Indexed(108),
        diff_del_sign: Color::Indexed(173),
        diff_add_hl_bg: Color::Indexed(23),
        diff_del_hl_bg: Color::Indexed(88),
        select_bg: Color::Rgb(94, 102, 126),
        select_text: Color::Rgb(240, 240, 240),
        select_accent: Color::Rgb(222, 252, 9),
        // Mouse wash matches the menu selection bar exactly — one selection
        // color everywhere; the flash is the same hue a step brighter.
        select_wash: Color::Rgb(94, 102, 126),
        select_flash: Color::Rgb(122, 132, 160),
        delete_bg: Color::Rgb(82, 42, 36),
        delete_text: Color::Rgb(255, 235, 226),
        delete_accent: Color::Rgb(255, 174, 146),
        canvas: None,
        code: Color::Rgb(154, 205, 185),
        toast_bg: Color::Rgb(21, 21, 21),
        jump_fg: Color::Rgb(22, 22, 22),
        jump_bg: Color::Rgb(228, 228, 228),
    };

    /// Light brand linen: the same warm paper dark uses as ink, now as the
    /// canvas — cream, not white, so it doesn't glare. Olive lime keeps the
    /// brand hue with enough contrast on that paper.
    pub const LIGHT: Self = Self {
        text: Color::Rgb(27, 24, 19),
        muted: Color::Rgb(102, 95, 85),
        faint: Color::Rgb(122, 114, 102),
        accent: Color::Rgb(102, 116, 0),
        assistant: Color::Rgb(73, 103, 36),
        user: Color::Rgb(75, 69, 60),
        tool: Color::Rgb(67, 91, 87),
        shell: Color::Rgb(126, 87, 15),
        link: Color::Rgb(60, 110, 180),
        quote: Color::Rgb(120, 120, 100),
        error: Color::Rgb(185, 72, 46),   // --code-flag
        warning: Color::Rgb(122, 90, 30), // --code-string
        info: Color::Rgb(50, 105, 160),
        live: Color::Rgb(200, 60, 55),
        // Keep diff fills in the indexed palette too; this avoids truecolor
        // parsing bugs on the same terminal path in light mode.
        diff_add_bg: Color::Indexed(253),
        diff_del_bg: Color::Indexed(253),
        diff_add_fg: Color::Indexed(238),
        diff_del_fg: Color::Indexed(131),
        diff_add_sign: Color::Indexed(29),
        diff_del_sign: Color::Indexed(130),
        diff_add_hl_bg: Color::Indexed(151),
        diff_del_hl_bg: Color::Indexed(181),
        select_bg: Color::Rgb(184, 181, 174),
        select_text: Color::Rgb(27, 24, 19),
        select_accent: Color::Rgb(89, 102, 0),
        // Same rule as dark: wash = the bar; flash darkens (paper inverts emphasis).
        select_wash: Color::Rgb(184, 181, 174),
        select_flash: Color::Rgb(162, 158, 150),
        delete_bg: Color::Rgb(226, 206, 198),
        delete_text: Color::Rgb(116, 38, 24),
        delete_accent: Color::Rgb(166, 58, 35),
        canvas: Some(Color::Rgb(237, 233, 226)), // brand paper
        code: Color::Rgb(38, 105, 98),
        toast_bg: Color::Rgb(226, 220, 208),
        jump_fg: Color::Rgb(237, 233, 226),
        jump_bg: Color::Rgb(22, 20, 26),
    };

    pub(super) fn current() -> &'static Self {
        match ui_theme() {
            UiTheme::Light => &Self::LIGHT,
            UiTheme::Dark => &Self::DARK,
        }
    }
}

#[inline]
pub(super) fn palette() -> &'static Palette {
    Palette::current()
}

/// Clear a floating region (overlay, card, toast), then repaint the theme canvas
/// under it. `Clear` resets cells to the terminal's own background; dark theme has
/// no canvas so the element floats on that background as before, but light theme
/// must repaint the paper fill `Clear` wiped or the modal would expose the raw
/// terminal bg — dark ink on a dark terminal, unreadable. Widgets that paint their
/// own background (e.g. a selected row) still draw on top of this.
pub(super) fn clear_to_canvas(frame: &mut Frame<'_>, area: Rect) {
    frame.render_widget(Clear, area);
    if let Some(canvas) = palette().canvas {
        frame.render_widget(Block::default().style(Style::default().bg(canvas)), area);
    }
}

// Accessors keep the historical UPPER_CASE call-site names (`TEXT()`, `MUTED()`, …)
// as zero-arg functions so a theme switch updates every render path at once.
#[allow(non_snake_case)]
#[inline]
pub(super) fn TEXT() -> Color {
    palette().text
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn MUTED() -> Color {
    palette().muted
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn FAINT() -> Color {
    palette().faint
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn ACCENT() -> Color {
    palette().accent
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn CODE() -> Color {
    palette().code
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn ASSISTANT() -> Color {
    palette().assistant
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn USER() -> Color {
    palette().user
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn TOOL() -> Color {
    palette().tool
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SHELL() -> Color {
    palette().shell
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn LINK() -> Color {
    palette().link
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn QUOTE() -> Color {
    palette().quote
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn ERROR() -> Color {
    palette().error
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn WARNING() -> Color {
    palette().warning
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn INFO() -> Color {
    palette().info
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn LIVE() -> Color {
    palette().live
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_ADD_BG() -> Color {
    palette().diff_add_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_DEL_BG() -> Color {
    palette().diff_del_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_ADD_FG() -> Color {
    palette().diff_add_fg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_DEL_FG() -> Color {
    palette().diff_del_fg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_ADD_SIGN() -> Color {
    palette().diff_add_sign
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_DEL_SIGN() -> Color {
    palette().diff_del_sign
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_ADD_HL_BG() -> Color {
    palette().diff_add_hl_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DIFF_DEL_HL_BG() -> Color {
    palette().diff_del_hl_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SELECT_BG() -> Color {
    palette().select_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SELECT_TEXT() -> Color {
    palette().select_text
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SELECT_ACCENT() -> Color {
    palette().select_accent
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SELECT_WASH() -> Color {
    palette().select_wash
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn SELECT_FLASH() -> Color {
    palette().select_flash
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DELETE_BG() -> Color {
    palette().delete_bg
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DELETE_TEXT() -> Color {
    palette().delete_text
}
#[allow(non_snake_case)]
#[inline]
pub(super) fn DELETE_ACCENT() -> Color {
    palette().delete_accent
}

/// Share URL notice prefix; `notice_spans` matches it to color the line.
pub(super) const LIVE_NOTICE_PREFIX: &str = "● Sharing: ";
/// Informational notice; `notice_display` adds the `ⓘ` marker at render time.
pub(super) fn info_notice(text: impl Into<String>) -> (Color, String) {
    (INFO(), text.into())
}
/// Footer badge shown while sharing.
pub(super) const LIVE_BADGE: &str = "● sharing";
/// Footer badge shown when `/config` "Agent tools" is off (plain-chat mode).
pub(super) const PLAIN_CODE_BADGE: &str = "plain chat";
pub(super) const EMPTY_STATE_TOP_GAP: u16 = 1;
/// Max visible rows in the queued-input panel.
pub(super) const QUEUE_PANEL_MAX_ROWS: usize = 5;
// No bottom padding: the composer already reserves its own blank spacing row
// above the divider, so the welcome screen's last line keeps the same single
// blank gap above the prompt as a live conversation does (not a doubled gap).
pub(super) const EMPTY_STATE_BOTTOM_GAP: u16 = 0;
/// First-run welcome tips. A fresh session cycles only these so the first
/// screen teaches `/help` and mode-switching, not power-user shortcuts.
pub(super) const WELCOME_STARTER_TIPS: &[&str] = &[
    "/help lists commands and keybindings",
    "Shift+Tab cycles mode: normal → auto-approve → review",
    "Ctrl+R reopens a past session",
];
/// After the first message, the intro banner shows one of these instead
/// (frozen — rotation only runs on the empty welcome screen).
pub(super) const WELCOME_ADVANCED_TIPS: &[&str] = &[
    "start a line with ! to run a shell command",
    "/rewind undoes the agent's file edits",
    "/goal <task> keeps working on its own until it's done",
    "Esc interrupts the agent mid-turn — in /goal mode, press Esc twice",
    "/share creates a live web link to this session",
    "Ctrl+T changes how hard the model thinks",
    "/skills and /mcp manage the agent's extra tools",
    "ask the agent to create a subagent for a task — a reviewer, an architect …",
    "/compact summarizes older turns to free up context",
    "/plan plans read-only — approve the plan and it builds",
    "/model switches models without losing the thread",
    "/config switches theme and toggles thinking / auto-approve",
    "/copy grabs a past reply to your clipboard",
    "Ctrl+O expands the last collapsed output — clicking ▸ works too",
    "type while the agent works to queue your next message",
    "/new starts a fresh session, keeping your keys",
];

pub(super) fn welcome_tips_for(empty_transcript: bool) -> &'static [&'static str] {
    if empty_transcript {
        WELCOME_STARTER_TIPS
    } else {
        WELCOME_ADVANCED_TIPS
    }
}

// In-box `❯ ` prompt flush to the border (matching the zero vertical padding),
// with a 1-cell gap to the text.
pub(super) const COMPOSER_PREFIX_WIDTH: u16 = 2;
pub(super) const DEFAULT_CHAT_SCROLL_SPEED: usize = 3;
pub(super) const MAX_CHAT_SCROLL_SPEED: usize = 50;
pub(super) const TOAST_DURATION: Duration = Duration::from_secs(3);
pub(super) const TOAST_FADE_AFTER: Duration = Duration::from_secs(2);
// Center toasts overlap the transcript, so they get out of the way faster.
pub(super) const CENTER_TOAST_DURATION: Duration = Duration::from_millis(2000);
pub(super) const CENTER_TOAST_FADE_AFTER: Duration = Duration::from_millis(1200);
/// How long the selection stays lit in the bright "just copied" color before
/// it auto-clears (amp-style flash). Kept short so it reads as a confirmation.
pub(super) const SELECTION_FLASH_DURATION: Duration = Duration::from_millis(550);
/// Max gap between clicks to count as a double/triple click.
pub(super) const MULTI_CLICK_INTERVAL: Duration = Duration::from_millis(400);
/// Minimum delay between auto-scroll steps while dragging at an edge.
pub(super) const DRAG_AUTOSCROLL_INTERVAL: Duration = Duration::from_millis(40);
/// How long each welcome-screen tip stays up before rotating (see `tick_welcome_tip`).
pub(super) const WELCOME_TIP_ROTATE_INTERVAL: Duration = Duration::from_secs(12);
// Tight repaint cadence while animating; slower when idle to cut wakeups.
pub(super) const ANIMATING_FRAME_INTERVAL: Duration = Duration::from_millis(16);
pub(super) const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// Nap after a pass that handled input: short enough that a scroll/keystroke
/// repaints near-instantly and in fine increments (not trailing the idle
/// cadence), but still a real yield so the streaming task keeps progressing on
/// the current-thread runtime.
pub(super) const INPUT_REPAINT_INTERVAL: Duration = Duration::from_millis(1);
/// Minimum time the live status label holds before it may change, so fast steps
/// don't flash by unreadably.
pub(super) const STATUS_MIN_DURATION: Duration = Duration::from_millis(1500);
/// Full-screen rewrite cadence while a reply streams (see `pending_full_repaint`).
pub(super) const STREAM_FULL_REPAINT_INTERVAL: Duration = Duration::from_secs(3);
/// Typewriter reveal rate. Each animation frame reveals at least
/// `TYPEWRITER_MIN_CHARS` of the buffered stream text (a steady floor so a slow
/// trickle still types out) plus `1/TYPEWRITER_CATCHUP_DIVISOR` of whatever
/// backlog remains, so a fast burst catches up in a few frames. At the ~60fps
/// animating cadence the floor alone types ~1800 chars/sec, so even a long reply
/// empties well under a second while still reading as fast typing.
pub(super) const TYPEWRITER_MIN_CHARS: usize = 30;
pub(super) const TYPEWRITER_CATCHUP_DIVISOR: usize = 2;

/// Upper bound on input events drained in a single event-loop pass. A fast
/// mouse drag emits one report per cell crossed, so the whole burst must be
/// consumed before the next repaint — otherwise the selection lags the cursor.
/// The cap keeps a pathological flood (e.g. a giant paste) from starving the
/// repaint; leftover events are drained on the next tick.
pub(super) const MAX_INPUT_EVENTS_PER_TICK: usize = 512;

/// An Enter this deep into one drain burst is a pasted newline (a burst spans
/// ~25ms — no human lands this many keystrokes before an intentional Enter).
pub(super) const PASTE_BURST_MIN_PRIOR_EVENTS: usize = 8;

// One-column message indent so transcript `❯`/`◆` markers line up with the
// composer's in-box prompt (border col 0, glyph col 1) and text columns match.
pub(super) const ACCENT_GUTTER_WIDTH: u16 = 1;

// The welcome header (banner + tip) sits on the message column itself; its
// 1-col left padding then mirrors the 1-row `EMPTY_STATE_TOP_GAP` above it.
pub(super) const HEADER_LEFT_INSET: u16 = 0;

// Right margin so wrapped prose stops short of the terminal edge; chrome (divider,
// composer, footer) stays full-bleed.
pub(super) const TRANSCRIPT_RIGHT_MARGIN: u16 = 2;

// Keep the main UI flush with the terminal's own inset.
pub(super) const APP_LEFT_MARGIN: u16 = 0;

// Keep the main UI flush with the terminal's own top inset.
pub(super) const APP_TOP_MARGIN: u16 = 0;

// Every non-main block (thinking, tool calls/results, shell, meta) nests this many
// columns under the `> `/`◆ ` turns — aligned with their body text.
pub(super) const SUB_BLOCK_INDENT: u16 = 2;

/// The pinned plan panel never crushes the transcript below this many rows, and
/// its body is also capped at a fraction of the screen (see `plan_panel_height`).
pub(super) const PLAN_PANEL_MIN_TRANSCRIPT: u16 = 4;
/// Transcript rows a decision card's slot leaves visible when the screen allows.
pub(super) const CARD_MIN_TRANSCRIPT: u16 = 4;
/// On short screens the slot still claims up to this — a blocking decision
/// outranks scrollback.
pub(super) const CARD_MIN_HEIGHT: u16 = 8;

pub(super) const COMMAND_MENU_MAX_ROWS: usize = 7;
pub(super) const PICKER_ROW_PREFIX_WIDTH: usize = 2;
/// Lines a PageUp/PageDn moves the `/skills` `/mcp` detail drill-in.
pub(super) const DETAIL_PAGE_LINES: u16 = 10;
/// Debounce so holding ↓ in `/resume` doesn't spawn a decrypt per row skimmed.
pub(super) const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(100);
/// Tail of the decrypted history kept for the `/resume` preview pane.
pub(super) const PREVIEW_MAX_MESSAGES: usize = 60;
/// Preview-cache cap; beyond it the cache is cleared (rebuilt on demand).
pub(super) const PREVIEW_CACHE_CAP: usize = 64;
#[derive(Clone, Copy)]
pub(super) struct SlashCommandSpec {
    pub(super) name: &'static str,
    pub(super) help_label: &'static str,
    pub(super) description: &'static str,
    pub(super) takes_argument: bool,
}

impl SlashCommandSpec {
    pub(super) fn command_label(self) -> String {
        format!("/{}", self.name)
    }

    pub(super) fn insertion_text(self) -> String {
        let suffix = if self.takes_argument { " " } else { "" };
        format!("/{}{}", self.name, suffix)
    }
}

/// Hidden `alias -> canonical` map; typing an alias surfaces its canonical
/// command. Keep in sync with the alias arms in `parse_slash_command`.
pub(super) const SLASH_ALIASES: &[(&str, &str)] =
    &[("quit", "exit"), ("undo", "rewind"), ("unwind", "rewind")];

pub(super) const SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "new",
        help_label: "/new",
        description: "start a fresh session",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "exit",
        help_label: "/exit",
        description: "quit this session",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "resume",
        help_label: "/resume [query]",
        description: "resume a saved session",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "model",
        help_label: "/model [name]",
        description: "switch model",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "key",
        help_label: "/key [id|name]",
        description: "switch saved key",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "attach",
        help_label: "/attach <path>",
        description: "attach a file or image",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "copy",
        help_label: "/copy [n]",
        description: "copy a past reply to the clipboard",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "preview",
        help_label: "/preview [path|url|reload|off]",
        description: "pin a live preview pane (image, SVG, HTML)",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "skills",
        help_label: "/skills [add|rm|update …]",
        description: "list, add, or remove agent skills",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "agents",
        help_label: "/agents [rm <name>]",
        description: "list or remove named sub-agents",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "create-skill",
        help_label: "/create-skill [intent]",
        description: "create or improve an agent skill, guided",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "mcp",
        help_label: "/mcp [add|rm …]",
        description: "list, add, or remove MCP servers",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "goal",
        help_label: "/goal <objective>",
        description: "work until the goal is done",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "plan",
        help_label: "/plan [objective]",
        description: "plan read-only, then build",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "rewind",
        help_label: "/rewind",
        description: "undo the agent's file edits",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "config",
        help_label: "/config",
        description: "toggle session settings (theme, thinking, auto-approve)",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "compact",
        help_label: "/compact [fast]",
        description: "summarize older turns to free context",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "context",
        help_label: "/context",
        description: "what's filling the context window",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "session",
        help_label: "/session",
        description: "show this session's id, source, and resume command",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "share",
        help_label: "/share [stop]",
        description: "share a live viewer link",
        takes_argument: true,
    },
    // aivo-provider only — hidden on BYOK keys (see `slash_command_visible`).
    SlashCommandSpec {
        name: "login",
        help_label: "/login",
        description: "sign in to your aivo account",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "logout",
        help_label: "/logout",
        description: "sign out of your aivo account",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "usage",
        help_label: "/usage",
        description: "show your aivo plan and usage",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "help",
        help_label: "/help",
        description: "show commands and keybindings",
        takes_argument: false,
    },
];

/// Ghost-text argument hint trailing a bare slash command in the composer
/// (matched by command name), so `/model` shows `[name]` inline. The menu row
/// already gives the one-line description; this teaches the argument syntax,
/// and the dropdown is suppressed while it shows (see `visible_command_menu`).
/// `[…]` marks an optional argument, `<…>` a required one.
pub(super) fn command_usage_hint(name: &str) -> Option<&'static str> {
    match name {
        // Richer than a placeholder — teaches the subcommand grammar.
        "mcp" => Some("[add [-p] <command> [args…] | rm <name>]"),
        "skills" => Some("[add [-p] <name>|<github:owner/repo> | rm <name>]"),
        "agents" => Some("[rm <name>]"),
        "create-skill" => Some("[what the skill should do]"),
        "goal" => Some("<objective> | stop"),
        // go/resume/save/stop stay as hidden aliases.
        "plan" => Some("[objective] | list | exit"),
        "share" => Some("[stop]"),
        "compact" => Some("[fast]"),
        "model" => Some("[name]"),
        "key" => Some("[id|name]"),
        "resume" => Some("[query]"),
        "copy" => Some("[n]"),
        // `attach`/`preview` deliberately omitted: a `<path>` ghost would
        // suppress the path-completion menu (see `visible_command_menu`).
        _ => None,
    }
}

/// Compact a count: `1234` → `1.2k`, `12345` → `12k`, `<1000` verbatim.
pub(super) fn humanize_count(n: usize) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{}k", n / 1000)
    }
}

/// `/compact` result notice; `kind` names what happened ("cleared stale output" /
/// "summarized older turns").
pub(super) fn freed_notice(freed: usize, kind: &str) -> (Color, String) {
    if freed == 0 {
        (MUTED(), "already compact — nothing to free".to_string())
    } else {
        (
            MUTED(),
            format!("freed ~{} tokens — {kind}", humanize_count(freed)),
        )
    }
}

pub(crate) struct CodeTuiParams {
    pub session_store: SessionStore,
    pub cache: ModelsCache,
    pub client: Client,
    pub key: ApiKey,
    pub copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub cwd: String,
    pub raw_model: String,
    pub model: String,
    pub initial_session: String,
    pub initial_history: Vec<ChatMessage>,
    pub initial_draft_attachments: Vec<MessageAttachment>,
    pub startup_notice: Option<String>,
    /// `--resume` request: `Some("")` opens the session picker at startup,
    /// `Some(id)` jumps straight to that session, `None` starts fresh.
    pub initial_resume: Option<String>,
    /// Positional `aivo code "<text>"`: auto-sent as the first message.
    pub initial_prompt: Option<String>,
    /// Context-digest block (`--resume`'s digest rung), appended to the system
    /// prompt per build. Session-only.
    pub injected_context: Option<String>,
    /// One-line digest summary for the startup notice + `/context` header.
    pub injected_context_summary: Option<String>,
    /// `--max-context <SIZE>` manual context-window override (tokens). Session-only.
    pub max_context: Option<u64>,
    /// `--share`: start live sharing at launch (device-link verified beforehand).
    pub share: bool,
    /// `--auto-approve`: pre-set the toggle at launch (session-only; Shift+Tab reverts).
    pub auto_approve: bool,
    /// `--vision-model [key::]model`: session-only describer, not persisted.
    pub vision_model: Option<String>,
    /// `--image-model [key::]model`: session-only generator, not persisted.
    pub image_model: Option<String>,
}

#[derive(Clone)]
pub(super) struct SessionPreview {
    pub(super) key_id: String,
    pub(super) key_name: String,
    pub(super) base_url: String,
    pub(super) session_id: String,
    pub(super) raw_model: String,
    pub(super) updated_at: String,
    pub(super) title: String,
    pub(super) preview_text: String,
    /// `Some` for a foreign coding-agent session (Claude Code / Codex) that
    /// imports on select; `None` for a native aivo session. When set,
    /// `session_id` is the deterministic id it imports to.
    pub(super) origin: Option<crate::services::session_import::SessionOrigin>,
}

pub(super) fn to_chat_messages(
    messages: Vec<crate::services::session_store::StoredChatMessage>,
) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|m| ChatMessage {
            model: m.model,
            role: m.role,
            content: m.content,
            reasoning_content: m.reasoning_content,
            attachments: m.attachments.unwrap_or_default(),
        })
        .collect()
}

impl SessionPreview {
    /// `key` is `None` when the session's stored key has since been removed —
    /// the row stays listed (the conversation is still valuable) and resuming
    /// it falls back to the live key.
    pub(super) fn from_index_entry(
        entry: crate::services::session_store::SessionIndexEntry,
        key: Option<&ApiKey>,
    ) -> Self {
        Self {
            key_id: key.map(|k| k.id.clone()).unwrap_or_default(),
            key_name: key
                .map(|k| k.display_name().to_string())
                .unwrap_or_else(|| "key removed".to_string()),
            base_url: key.map(|k| k.base_url.clone()).unwrap_or_default(),
            session_id: entry.session_id,
            raw_model: entry.model,
            updated_at: entry.updated_at,
            title: entry.title,
            preview_text: entry.preview,
            origin: None,
        }
    }

    /// A picker row for a foreign session (Claude Code / Codex) not yet imported.
    /// `key_name` doubles as the source badge; `session_id` is the id it will
    /// import to, so it dedupes against an already-imported native row.
    pub(super) fn from_importable(imp: crate::services::session_import::ImportableSession) -> Self {
        let label = crate::services::session_import::source_label(&imp.origin.cli);
        Self {
            key_id: String::new(),
            key_name: label.to_string(),
            base_url: String::new(),
            session_id: imp.aivo_id,
            raw_model: String::new(),
            updated_at: imp.updated_at.to_rfc3339(),
            preview_text: imp.title.clone(),
            title: imp.title,
            origin: Some(imp.origin),
        }
    }

    /// An imported foreign session that's been continued in aivo — a fork of the
    /// (never-modified) original. Because a foreign session persists only after a
    /// real turn (see `pristine_import_len`), any persisted row with a source-tool
    /// id is a genuine fork. A not-yet-opened foreign row (`origin` set) isn't one.
    pub(super) fn is_fork(&self) -> bool {
        self.origin.is_none()
            && crate::services::session_import::import_source_label(&self.session_id).is_some()
    }

    /// Source tag shown as the picker-row prefix: `aivo` for a native session,
    /// `Claude`/`Codex` for an importable foreign one. A foreign session keeps
    /// its tag after import via its deterministic `import-<cli>-…` id, so
    /// continuing it in aivo never silently relabels it `[aivo]`.
    pub(super) fn source_tag(&self) -> &'static str {
        if let Some(origin) = &self.origin {
            return crate::services::session_import::source_label(&origin.cli);
        }
        crate::services::session_import::import_source_label(&self.session_id).unwrap_or("aivo")
    }

    pub(super) fn search_text(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            self.session_id,
            self.title,
            self.preview_text,
            self.key_name,
            self.raw_model,
            self.base_url
        )
    }
}

#[derive(Clone)]
pub(super) struct LoadedSession {
    pub(super) key_id: String,
    pub(super) session_id: String,
    pub(super) raw_model: String,
    pub(super) messages: Vec<ChatMessage>,
    /// The durably-persisted agent-engine transcript (raw OpenAI messages with
    /// tool_calls + results), restored verbatim into the engine on resume for
    /// exact tool history. `None` for non-agent or pre-feature sessions.
    pub(super) engine_messages: Option<Vec<serde_json::Value>>,
    /// True for a foreign session resumed IN MEMORY (not yet persisted): merely
    /// viewing a Claude/Codex session shouldn't create an aivo copy. It persists
    /// (as a fork) only when a real turn grows the transcript past this baseline.
    pub(super) pristine_import: bool,
    /// True when this is a saved fork whose SOURCE session has newer messages
    /// than the fork has seen (the histories diverged) — surfaced as a notice
    /// so loading the older fork is never silent.
    pub(super) source_newer: bool,
    /// From the converter on a first-open import, from the persisted stamp on
    /// a saved fork; `None` for native sessions and pre-feature forks.
    pub(super) import_fidelity: Option<crate::services::session_import::ImportFidelity>,
    /// Unfinished plan-mode snapshot saved with the session (drafted plan +
    /// plan-mode flag), restored on resume so `/plan go` keeps working.
    pub(super) plan_state: Option<crate::services::session_store::PlanState>,
    /// Saved describe cache, restored so resume doesn't re-describe.
    pub(super) image_descriptions: Option<std::collections::HashMap<String, String>>,
}

impl LoadedSession {
    pub(super) fn from_state(state: crate::services::session_store::CodeSessionState) -> Self {
        let mut loaded = Self {
            key_id: state.key_id,
            session_id: state.session_id,
            raw_model: state.model,
            messages: to_chat_messages(state.messages),
            engine_messages: state.engine_messages,
            pristine_import: false,
            source_newer: false,
            import_fidelity: state.import_fidelity,
            plan_state: state.plan_state,
            image_descriptions: state.image_descriptions,
        };
        loaded.normalize_history_images();
        loaded
    }

    pub(super) fn from_import(
        key_id: String,
        session_id: String,
        raw_model: String,
        transcript: crate::services::session_import::ImportedTranscript,
    ) -> Self {
        let mut loaded = Self {
            key_id,
            session_id,
            raw_model,
            messages: to_chat_messages(transcript.messages),
            engine_messages: Some(transcript.engine_messages),
            pristine_import: true,
            source_newer: false,
            import_fidelity: Some(transcript.fidelity),
            plan_state: None,
            image_descriptions: None,
        };
        loaded.normalize_history_images();
        loaded
    }

    /// Shrinks oversized images in persisted history and preserves describe-cache keys.
    pub(super) fn normalize_history_images(&mut self) {
        use crate::services::image_optimize::optimize_base64;
        use crate::services::session_store::AttachmentStorage;
        use crate::services::vision_describe::{image_hash, parse_data_url};

        let mut rewritten: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();

        for message in &mut self.messages {
            for attachment in &mut message.attachments {
                if !attachment.is_image() {
                    continue;
                }
                let AttachmentStorage::Inline { data } = &mut attachment.storage else {
                    continue;
                };
                let (mime, new_b64) = match rewritten.get(data.as_str()) {
                    Some(hit) => hit.clone(),
                    None => match optimize_base64(data) {
                        Some(new) => new,
                        None => continue,
                    },
                };
                let old = std::mem::replace(data, new_b64.clone());
                attachment.mime_type = mime.clone();
                rewritten.insert(old, (mime, new_b64));
            }
        }

        for message in self.engine_messages.iter_mut().flatten() {
            let Some(serde_json::Value::Array(parts)) = message.get_mut("content") else {
                continue;
            };
            for part in parts
                .iter_mut()
                .filter(|p| crate::agent::tokens::is_image_part(p))
            {
                let Some((_, b64)) = part["image_url"]["url"].as_str().and_then(parse_data_url)
                else {
                    continue;
                };
                let (mime, new_b64) = match rewritten.get(b64) {
                    Some(hit) => hit.clone(),
                    None => match optimize_base64(b64) {
                        Some(new) => new,
                        None => continue,
                    },
                };
                let old = b64.to_string();
                part["image_url"]["url"] =
                    serde_json::Value::String(format!("data:{mime};base64,{new_b64}"));
                rewritten.insert(old, (mime, new_b64));
            }
        }

        // Preserve cached descriptions after changing the hashed image data.
        if let Some(descriptions) = self.image_descriptions.as_mut()
            && !descriptions.is_empty()
        {
            for (old_b64, (_, new_b64)) in &rewritten {
                if let Some(text) = descriptions.remove(&image_hash(old_b64)) {
                    descriptions.insert(image_hash(new_b64), text);
                }
            }
        }
    }
}

/// One discovered skill in the `/skills` overlay. `description` is the FULL
/// frontmatter text — rows and the `/` menu advert-truncate it at render time.
#[derive(Clone, Debug)]
pub(super) struct SkillToggle {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) enabled: bool,
    pub(super) dir: std::path::PathBuf,
    pub(super) scope: crate::agent::skills::SkillScope,
    /// Full SKILL.md instructions, read at open time (discovery leaves them empty).
    pub(super) body: String,
}

/// One row in the `/agents` overlay: a discovered sub-agent profile plus the
/// display metadata the panes need (resolved at open time).
#[derive(Clone, Debug)]
pub(super) struct AgentRow {
    pub(super) name: String,
    pub(super) description: String,
    /// "builtin" | "repo" | "user" — where the file lives.
    pub(super) scope: &'static str,
    pub(super) source: std::path::PathBuf,
    pub(super) model: Option<String>,
    /// Resolved tool scope; `None` = unscoped (all tools).
    pub(super) tools: Option<Vec<&'static str>>,
    pub(super) isolation_worktree: bool,
    pub(super) body: String,
}

/// The interactive `/agents` overlay: the `/skills` interaction grammar minus
/// toggle/add — profiles have no enabled state, and creation is conversational
/// (the create-agent workflow) rather than a form.
#[derive(Clone, Debug, Default)]
pub(super) struct AgentsOverlay {
    pub(super) items: Vec<AgentRow>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) pending_delete: Option<usize>,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl AgentsOverlay {
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.description.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    pub(super) fn refilter(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// First Ctrl+D arms the delete, a second on the same row confirms it.
    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        if self.pending_delete == Some(self.selected) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(self.selected);
            false
        }
    }
}

/// The interactive `/skills` overlay: a filterable toggle list. `query` is the
/// type-to-filter text; `selected` is the highlighted item's index INTO `items`
/// (not the filtered view). `adding` holds the in-progress add text; `pending_delete`
/// is the index armed for a two-press Ctrl+D delete; `viewing` is the index whose
/// detail drill-in is open; `detail_scroll` is that drill-in's vertical scroll
/// offset in (already-wrapped) lines.
#[derive(Clone, Debug, Default)]
pub(super) struct SkillsOverlay {
    pub(super) items: Vec<SkillToggle>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) adding: Option<String>,
    pub(super) pending_delete: Option<usize>,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl SkillsOverlay {
    /// Indices of `items` matching `query`, best match first (see
    /// [`ranked_indices`]). An empty query matches everything in list order.
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.description.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    /// Re-anchor to the best-ranked match after the query changed, so
    /// Enter/Space act on the top hit, not a stale survivor further down.
    pub(super) fn refilter(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    /// Whether the highlighted item is currently visible (matches the filter) —
    /// gates Enter/Tab/Ctrl+D so they never act on a filtered-out row.
    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// First Ctrl+D on a row arms the delete (returns `false`); a second on the
    /// same row confirms it (returns `true`). Selecting away clears the arm.
    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        if self.pending_delete == Some(self.selected) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(self.selected);
            false
        }
    }
}

/// A background skill install in flight; `bytes` is written live by the
/// download stream for the size readout.
pub(super) struct SkillInstallProgress {
    pub(super) source: String,
    pub(super) started: Instant,
    /// "Fetching" while the source downloads/stages; "Installing" during a copy.
    pub(super) verb: &'static str,
    pub(super) bytes: crate::agent::skills::DownloadProgress,
}

impl SkillInstallProgress {
    pub(super) fn new(source: String, verb: &'static str) -> Self {
        Self {
            source,
            started: Instant::now(),
            verb,
            bytes: Default::default(),
        }
    }

    /// `Fetching (2.4MB) <source>…` — the size sits BEFORE the source because a
    /// long URL clips at the pane edge; omitted until the first chunk lands.
    pub(super) fn status_text(&self) -> String {
        let bytes = self.bytes.load(std::sync::atomic::Ordering::Relaxed);
        if bytes == 0 {
            format!("{} {}…", self.verb, self.source)
        } else {
            format!(
                "{} ({}) {}…",
                self.verb,
                crate::services::huggingface::human_size(bytes),
                self.source
            )
        }
    }
}

/// One candidate row in the skill-install picker.
#[derive(Clone, Debug)]
pub(super) struct InstallPickItem {
    pub(super) name: String,
    pub(super) description: String,
    /// Read at open time — the staged tree is gone once the pick resolves.
    pub(super) body: String,
    pub(super) checked: bool,
    /// Already installed; a mark on it means update-in-place.
    pub(super) installed: bool,
}

/// The install picker; empty `items` = the fetch is still running. The staged
/// tree lives on the app — the overlay is cloned every render.
#[derive(Clone, Debug, Default)]
pub(super) struct SkillInstallOverlay {
    pub(super) source: String,
    /// `-p/--project`: installing into the repo's `.agents/skills`.
    pub(super) project: bool,
    pub(super) items: Vec<InstallPickItem>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl SkillInstallOverlay {
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.description.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    pub(super) fn refilter(&mut self) {
        self.detail_scroll = 0;
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// The checked set (a checked installed row = update mark), else the
    /// highlighted row — which never implicitly updates an installed one.
    pub(super) fn pick_names(&self) -> Vec<String> {
        let checked: Vec<String> = self
            .items
            .iter()
            .filter(|i| i.checked)
            .map(|i| i.name.clone())
            .collect();
        if !checked.is_empty() {
            return checked;
        }
        self.items
            .get(self.selected)
            .filter(|i| !i.installed && self.has_selection())
            .map(|i| vec![i.name.clone()])
            .unwrap_or_default()
    }

    /// Ctrl+A: check every not-yet-installed skill, or clear all.
    pub(super) fn toggle_all(&mut self) {
        let all_checked = self
            .items
            .iter()
            .filter(|i| !i.installed)
            .all(|i| i.checked);
        for item in self.items.iter_mut().filter(|i| !i.installed) {
            item.checked = !all_checked;
        }
    }
}

/// Coarse health of one MCP server, for coloring its `/mcp` status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum McpHealth {
    Connected,
    Failed,
    /// An HTTP server that answered 401 — it needs OAuth (authorize with Ctrl+O).
    NeedsAuth,
    Idle,
    Disabled,
}

/// One row in the `/mcp` overlay: a configured server, its live status (a snapshot
/// taken when the overlay opened), whether it's enabled, and which file defines
/// it (only `User` servers can be removed in-overlay).
#[derive(Clone, Debug)]
pub(super) struct McpServerRow {
    pub(super) name: String,
    pub(super) status: String,
    pub(super) health: McpHealth,
    pub(super) enabled: bool,
    pub(super) scope: crate::agent::mcp::ServerScope,
    /// The configured launch command, for the Enter drill-in detail view.
    pub(super) command: String,
    /// `true` for a remote (`url`) server — drives the Ctrl+O gate and the
    /// detail label without re-sniffing the display string.
    pub(super) remote: bool,
}

/// The interactive `/mcp` overlay: a filterable toggle list of configured MCP
/// servers. `query` is the type-to-filter text; `selected` is the highlighted
/// item's index INTO `items`; `adding` holds the in-progress add text;
/// `pending_delete` is the index armed for a two-press Ctrl+D delete (removal
/// edits the user `mcp.json`, so it confirms); `viewing` is the index whose
/// detail drill-in is open; `detail_scroll` is that drill-in's vertical scroll
/// offset in (already-wrapped) lines.
#[derive(Clone, Debug, Default)]
pub(super) struct McpOverlay {
    pub(super) items: Vec<McpServerRow>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) adding: Option<String>,
    pub(super) pending_delete: Option<usize>,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl McpOverlay {
    /// Indices of `items` matching `query`, best match first (see
    /// [`ranked_indices`]). An empty query matches everything in list order.
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.status.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    /// Re-anchor to the best-ranked match after the query changed, so
    /// Enter/Space act on the top hit, not a stale survivor further down.
    pub(super) fn refilter(&mut self) {
        self.pending_delete = None;
        self.detail_scroll = 0;
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    /// Whether the highlighted item is currently visible (matches the filter).
    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// First Ctrl+D on a row arms the delete (returns `false`); a second on the
    /// same row confirms it (returns `true`). Selecting away clears the arm.
    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        if self.pending_delete == Some(self.selected) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(self.selected);
            false
        }
    }
}

/// One row in the Ctrl+T per-server tool drill-in: an MCP tool and whether it's
/// offered to the agent.
#[derive(Clone, Debug)]
pub(super) struct McpToolRow {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) enabled: bool,
}

/// The Ctrl+T tool-toggle sub-overlay for one MCP server. `parent` is the
/// `/mcp` overlay state to restore on Esc (its statuses are refreshed then, so
/// the server row's `· N off` count stays current).
#[derive(Clone, Debug)]
pub(super) struct McpToolsOverlay {
    pub(super) server: String,
    pub(super) parent: Box<McpOverlay>,
    pub(super) items: Vec<McpToolRow>,
    pub(super) selected: usize,
    pub(super) query: String,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl McpToolsOverlay {
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.description.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    pub(super) fn refilter(&mut self) {
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }
}

/// One server from a multi-server JSON paste, awaiting the pick.
#[derive(Clone, Debug)]
pub(super) struct McpPasteRow {
    pub(super) name: String,
    /// Listing target: the command line or the URL.
    pub(super) display: String,
    pub(super) config: serde_json::Value,
    pub(super) checked: bool,
    /// A same-named server is already configured; a mark on it replaces that
    /// entry in place (never a `-2` duplicate).
    pub(super) exists: bool,
}

/// The paste picker for a `mcpServers` block defining ≥2 servers: new names
/// arrive prechecked, existing ones need an explicit mark to replace. `parent`
/// is the `/mcp` overlay to restore on Esc (`None` when the paste came from
/// the composer, where no overlay was open).
#[derive(Clone, Debug, Default)]
pub(super) struct McpPasteOverlay {
    pub(super) parent: Option<Box<McpOverlay>>,
    /// `-p/--project`: writing into the repo `.mcp.json`.
    pub(super) project: bool,
    pub(super) items: Vec<McpPasteRow>,
    pub(super) selected: usize,
    pub(super) query: String,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl McpPasteOverlay {
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        ranked_indices(
            &self.query,
            self.items
                .iter()
                .map(|it| (it.name.as_str(), it.display.as_str())),
        )
    }

    pub(super) fn select_prev(&mut self) {
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    pub(super) fn refilter(&mut self) {
        self.selected = self.filtered_indices().first().copied().unwrap_or(0);
    }

    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// Ctrl+A: check every new (non-existing) server, or clear them all.
    /// Existing servers only replace via an explicit Space.
    pub(super) fn toggle_all(&mut self) {
        let all_checked = self.items.iter().filter(|i| !i.exists).all(|i| i.checked);
        for item in self.items.iter_mut().filter(|i| !i.exists) {
            item.checked = !all_checked;
        }
    }
}

/// One `/config` preference, keyed so the handler routes to the right live state
/// without matching label text. Each is a segmented switch, so booleans and
/// multi-value settings share one control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConfigSetting {
    Theme,
    InlineImages,
    Thinking,
    /// Standing permission mode (`normal` / `auto-approve` / `review`), folding the
    /// two former checkboxes into one radio.
    Approval,
    UseWebSearch,
    AgentTools,
    /// Footer generation-rate (tok/s) indicator.
    FooterTps,
    /// Footer prompt-cache hit-rate indicator.
    FooterCacheHit,
    /// Footer session-spend (`~$`) suffix on the context meter.
    FooterPrice,
    /// Describe images for text-only models (`aivo` / `custom` / `off`).
    VisionFallback,
    /// The agent's `generate_image` tool (`custom` / `off`).
    ImageGen,
}

/// `Enter` also drills in: the vision row re-opens the describer picker instead
/// of advancing off an active `custom`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CycleDir {
    Enter,
    Next,
    Prev,
}

/// The segmented values a `/config` row can hold and which one is live. Read live
/// (see `config_segments`) so a row can't drift from the flag it mirrors.
pub(super) struct ConfigSegments {
    /// Owned: the Thinking row's options come from the model's catalog levels.
    pub(super) options: Vec<String>,
    pub(super) active: usize,
}

/// One `/config` row: a labelled preference. Inspector copy is live-read at
/// render (`config_active_help`) so a mid-overlay pick can't go stale.
#[derive(Clone, Debug)]
pub(super) struct ConfigRow {
    pub(super) setting: ConfigSetting,
    pub(super) label: &'static str,
}

/// The `/config` overlay: a fixed list of segmented switches. ↑/↓ move rows, ←/→
/// (or Enter/Space) change the selected value; `selected` indexes `items`.
/// Split layout shows a setting inspector in the right pane.
#[derive(Clone, Debug, Default)]
pub(super) struct ConfigOverlay {
    pub(super) items: Vec<ConfigRow>,
    pub(super) selected: usize,
    /// Picker/apply failure for a row — shown in that row's inspector, not as a
    /// transcript notice (the modal would hide it / sit too far from the setting).
    pub(super) error: Option<(ConfigSetting, String)>,
    /// Right-pane (or stacked detail) scroll; reset when the selection moves.
    pub(super) detail_scroll: u16,
    /// Persisted list-window start (rows), written back after render.
    pub(super) list_scroll: usize,
    /// `selected` at last render; equality marks `list_scroll` as a wheel pan.
    pub(super) scroll_selected: usize,
}

impl ConfigOverlay {
    /// Wraps: the list is short, so cycling beats a dead stop.
    pub(super) fn select_prev(&mut self) {
        if self.items.is_empty() {
            return;
        }
        self.selected = self.selected.checked_sub(1).unwrap_or(self.items.len() - 1);
        self.detail_scroll = 0;
    }

    pub(super) fn select_next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1) % self.items.len();
            self.detail_scroll = 0;
        }
    }
}

/// `/login` waiting for approval: a passive status card (a card, not a modal —
/// the poll can run for minutes and must not block typing), created only once
/// the device code arrives.
pub(super) struct AccountLoginCard {
    pub(super) user_code: String,
    pub(super) open_url: String,
}

/// Filter + rank toggle-overlay rows: name matches (scored by `score_match`:
/// prefix > substring > fuzzy) rank above rows only their detail text rescued,
/// so an exact name hit can't drown under fuzzy description noise. Stable
/// (list order) within equal ranks; empty query keeps everything in list order.
fn ranked_indices<'a>(query: &str, rows: impl Iterator<Item = (&'a str, &'a str)>) -> Vec<usize> {
    // (name-vs-detail tier, name score): lower sorts first.
    type FilterRank = (u8, (u8, usize, usize));
    let mut ranked: Vec<(FilterRank, usize)> = rows
        .enumerate()
        .filter_map(|(index, (name, detail))| {
            if query.is_empty() {
                Some(((0, (0, 0, 0)), index))
            } else if matches_fuzzy(query, name) {
                Some(((0, crate::tui::score_match(query, name)), index))
            } else if matches_fuzzy(query, &format!("{name} {detail}")) {
                Some(((1, (0, 0, 0)), index))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by_key(|&(rank, _)| rank);
    ranked.into_iter().map(|(_, index)| index).collect()
}

/// Move `selected` (an index into the full list) one step (`dir` = -1/+1) within
/// `filtered` (the in-order matching indices), clamped to the filtered ends.
fn move_within(filtered: &[usize], selected: &mut usize, dir: i32) {
    if filtered.is_empty() {
        return;
    }
    let pos = filtered.iter().position(|&i| i == *selected).unwrap_or(0);
    let next = if dir < 0 {
        pos.saturating_sub(1)
    } else {
        (pos + 1).min(filtered.len() - 1)
    };
    *selected = filtered[next];
}

/// Session decision on whether to spawn a repo's project `.mcp.json` STDIO
/// servers — the one piece of project content aivo *executes* (a local child
/// process) rather than reads, so it's gated like aivo's own actions. Seeded
/// from the persistent per-repo allow-list on the first connect; the consent
/// card sets it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum ProjectMcpConsent {
    #[default]
    Unknown,
    Allowed,
    Denied,
}

/// A pending consent card: a repo's `.mcp.json` wants to spawn these stdio
/// servers. Carries what's needed to (re)connect once the user decides.
#[derive(Clone, Debug)]
pub(super) struct McpConsentPrompt {
    /// `(name, "command args…")` for each project stdio server, for display.
    pub(super) servers: Vec<(String, String)>,
    pub(super) cwd: String,
    /// The base opt-out set (the user's `/mcp` toggles) to reconnect with.
    pub(super) base_disabled: std::collections::HashSet<String>,
}

/// The agent mode `/plan exit` returns to (modes are mutually exclusive).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PlanPriorMode {
    #[default]
    Normal,
    Auto,
    Review,
}

/// Content digest of a repo's project `.mcp.json` stdio servers — the exact
pub(super) use crate::agent::mcp::project_mcp_digest;

#[derive(Clone)]
pub(super) enum Overlay {
    None,
    /// `/help` — slash commands, skills, and keybindings. `scroll` is the body's
    /// vertical scroll offset (the content is taller than the box on most
    /// terminals), clamped by the renderer and written back each frame.
    Help {
        scroll: u16,
    },
    /// `/skills` — the agent skills discovered for the working dir, toggleable.
    Skills(SkillsOverlay),
    /// `/agents` — the named sub-agents discovered for the working dir.
    Agents(AgentsOverlay),
    /// A multi-skill install source was staged — pick which skills to copy in.
    SkillInstall(SkillInstallOverlay),
    /// `/mcp` — the configured MCP servers with status, toggleable.
    Mcp(McpOverlay),
    /// Ctrl+T from `/mcp` — one server's tools, individually toggleable.
    McpTools(McpToolsOverlay),
    /// A multi-server `mcpServers` JSON paste — pick which to add/replace.
    McpPaste(McpPasteOverlay),
    /// `/config` — a small fixed list of chat preferences, toggleable.
    Config(ConfigOverlay),
    /// `/context` — the context-window breakdown, over the injected digest text.
    Context {
        report: Box<crate::agent::engine::ContextReport>,
        scroll: u16,
    },
    /// `/session` (or clicking the footer id) — this session's id, provenance,
    /// model, key, and resume command. `scroll` for tiny terminals.
    Session {
        scroll: u16,
    },
    /// Clicking the footer `● sharing` badge — the live share's URL and controls.
    Share {
        scroll: u16,
    },
    Picker(Box<PickerState>),
}

impl Overlay {
    pub(super) fn blocks_input(&self) -> bool {
        !matches!(self, Self::None)
    }
}

pub(super) use crate::services::image_generate::GeneratorSource;
pub(super) use crate::services::vision_describe::DescriberSource;

/// Whether the vision fallback can cover this turn, and if not why — the one
/// place that policy is decided. Resolving `OwnKey` is deferred to dispatch.
pub(super) enum DescriberStatus {
    Gateway,
    OwnKey {
        key_id: String,
        model: String,
    },
    Off,
    /// Gateway quota/auth exhausted for the rest of the session.
    Exhausted,
    /// `custom` selected with no usable pair — unset, or equal to the active
    /// model (which the per-model upstream would route back into the main chat).
    Unconfigured,
    /// Launch-bound OAuth/ACP key: no loopback router for the describe call.
    NotAgentCapable,
}

#[derive(Clone)]
pub(super) enum PickerValue {
    Model(String),
    Key(ApiKey),
    Session(SessionPreview),
    /// A `/rewind` target. `history_index` = truncation point in chat history;
    /// `ordinal` = checkpoint to revert files through — the turn's own or the
    /// nearest newer row's (`None` = conversation-only); `keep_engine` = the turn
    /// has its own checkpoint, so the engine transcript can truncate to it.
    RewindTurn {
        history_index: usize,
        ordinal: Option<usize>,
        keep_engine: bool,
    },
    /// An unfinished plan from a saved session (`/plan resume`); the payload
    /// rides in the row so activation needs no second load.
    PlanResume(PlanCarry),
}

/// A model option ready for the picker: stable id and display label.
#[derive(Clone, Debug)]
pub(super) struct ModelChoice {
    pub(super) id: String,
    pub(super) label: String,
    /// Live catalog `image_input` when advertised; `None` = unknown.
    pub(super) image_input: Option<bool>,
}

#[derive(Clone)]
pub(super) struct PickerEntry {
    pub(super) label: String,
    pub(super) search_text: String,
    pub(super) value: PickerValue,
}

impl PickerEntry {
    pub(super) fn row_height(&self) -> usize {
        1
    }
}

/// What `/plan resume` carries: a draft awaiting approval (re-enters plan mode),
/// another session's mid-execution checklist (continues directly), or THIS
/// session's interrupted checklist (same-session framing).
#[derive(Clone)]
pub(super) enum PlanCarry {
    Draft(String),
    /// Checklist steps stay parsed — display sites read progress directly and
    /// only the machine message stringifies them.
    Continue(serde_json::Value),
    Interrupted(serde_json::Value),
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum ModelSelectionTarget {
    CurrentChat,
    KeySwitch {
        key: ApiKey,
        /// Saved model: picker focus, and the fallback when there's no listing.
        prior_model: Option<String>,
    },
    VisionDescriber(ApiKey),
    ImageGenerator(ApiKey),
}

/// What picking a key is for; the vision/image variants chain into a model picker.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum KeySelectionTarget {
    Switch,
    VisionDescriber,
    ImageGenerator,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum PickerKind {
    Model {
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    },
    Key {
        target: KeySelectionTarget,
    },
    Session,
    Rewind,
    PlanResume,
}

impl PickerKind {
    /// Drill-in from `/config` — Esc or an empty catalog lands on this row.
    pub(super) fn config_home(&self) -> Option<ConfigSetting> {
        match self {
            Self::Key {
                target: KeySelectionTarget::VisionDescriber,
            }
            | Self::Model {
                target: ModelSelectionTarget::VisionDescriber(_),
                ..
            } => Some(ConfigSetting::VisionFallback),
            Self::Key {
                target: KeySelectionTarget::ImageGenerator,
            }
            | Self::Model {
                target: ModelSelectionTarget::ImageGenerator(_),
                ..
            } => Some(ConfigSetting::ImageGen),
            _ => None,
        }
    }

    /// Drill-in from the `/key` picker — Esc re-opens it focused on this key.
    pub(super) fn key_picker_home(&self) -> Option<&str> {
        match self {
            Self::Model {
                target: ModelSelectionTarget::KeySwitch { key, .. },
                ..
            } => Some(&key.id),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(super) struct PickerState {
    pub(super) title: &'static str,
    pub(super) query: String,
    pub(super) items: Vec<PickerEntry>,
    pub(super) loading: bool,
    pub(super) selected: usize,
    pub(super) kind: PickerKind,
    pub(super) pending_delete: Option<DeleteConfirmTarget>,
    /// Session-preview scroll, in lines UP from the bottom (0 = latest).
    pub(super) preview_scroll: u16,
    /// Session the scroll belongs to; a selection mismatch reads as 0 (re-anchor).
    pub(super) preview_scroll_for: Option<String>,
    /// First visible list position (item index generic / row index session),
    /// written back after each render.
    pub(super) scroll_top: usize,
    /// `selected` at last render; equality marks a `scroll_top` change as a
    /// wheel pan that may leave the selection off-screen. Init usize::MAX so
    /// the first frame follows the selection.
    pub(super) scroll_selected: usize,
}

/// Renderer→dispatch report: the clamped scroll to write back (renderers get a
/// clone of the overlay) and the right detail pane's rect while split.
#[derive(Default)]
pub(super) struct OverlayRenderOut {
    pub(super) detail_scroll: Option<u16>,
    pub(super) detail_area: Option<Rect>,
    /// Session id a `/resume` preview-scroll clamp belongs to.
    pub(super) scroll_for: Option<String>,
    /// Config (and similar) list pane: visual row → item index for click-to-select.
    pub(super) list_area: Option<Rect>,
    pub(super) list_row_index: Vec<Option<usize>>,
    /// Inspector segmented-control pills: clickable rect → option index.
    pub(super) segment_hits: Vec<(Rect, usize)>,
    /// Clamped list-window start to persist (`scroll_top`/`list_scroll`).
    pub(super) list_scroll: Option<usize>,
    /// The list scrollbar's geometry from this render, for thumb dragging.
    pub(super) scrollbar: Option<ScrollbarHit>,
}

/// A rendered list scrollbar: the 1-column track plus the window it maps, in
/// the owning list's units. Dragging inverts [`Self::thumb`].
#[derive(Clone, Copy, Debug)]
pub(super) struct ScrollbarHit {
    pub(super) track: Rect,
    pub(super) start: usize,
    pub(super) visible: usize,
    pub(super) total: usize,
}

impl ScrollbarHit {
    /// `(thumb_y, thumb_h, run, max_start)` — `run` is the travel span in rows.
    pub(super) fn thumb(&self) -> (usize, usize, usize, usize) {
        let track_h = usize::from(self.track.height);
        let thumb_h = (track_h * self.visible / self.total).max(1);
        let run = track_h - thumb_h;
        let max_start = self.total - self.visible;
        let thumb_y = (run * self.start.min(max_start) + max_start / 2)
            .checked_div(max_start)
            .unwrap_or(0);
        (thumb_y, thumb_h, run, max_start)
    }

    /// The window start for a drag with the thumb's top at `thumb_y` rows.
    pub(super) fn start_for_thumb_y(&self, thumb_y: usize) -> usize {
        let (_, _, run, max_start) = self.thumb();
        if run == 0 {
            return self.start;
        }
        (thumb_y.min(run) * max_start + run / 2) / run
    }
}

/// The glyphs `render_list_scrollbar` draws the thumb with (half-width bar +
/// quadrant end caps) — tests filter/scan the modal margin column by this set.
#[cfg(test)]
pub(super) fn is_scrollbar_thumb_char(c: char) -> bool {
    matches!(c, '▐' | '▗' | '▝')
}

#[derive(Clone, Default)]
pub(super) struct PickerHitbox {
    pub(super) overlay_area: Rect,
    pub(super) list_area: Rect,
    pub(super) row_to_filtered_index: Vec<Option<usize>>,
    pub(super) segment_hits: Vec<(Rect, usize)>,
}

#[derive(Clone)]
pub(super) struct LoadingResume {
    pub(super) request_id: u64,
    pub(super) preview: SessionPreview,
    /// Bare `/plan` restored this session for its plan: once the load lands,
    /// auto-dispatch the continue turn. Scoped here so it dies with the load.
    pub(super) continue_plan: bool,
}

/// A loaded `/resume` preview: the decrypted tail of one session's history.
/// `updated_at` is the INDEX value captured at request time — not the file's,
/// so index/file skew can't retrigger a load every tick.
#[derive(Clone)]
pub(super) struct PreviewEntry {
    pub(super) updated_at: String,
    pub(super) messages: Vec<ChatMessage>,
    pub(super) truncated: bool,
    pub(super) error: Option<String>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct DeleteConfirmTarget {
    pub(super) key_id: String,
    pub(super) session_id: String,
}

#[derive(Clone)]
pub(super) struct ResumeRestoreState {
    pub(super) key: ApiKey,
    pub(super) copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub(super) raw_model: String,
    pub(super) model: String,
    pub(super) format: ChatFormat,
    pub(super) history: Vec<ChatMessage>,
    pub(super) draft: String,
    pub(super) draft_attachments: Vec<MessageAttachment>,
    pub(super) cursor: usize,
    pub(super) command_menu: CommandMenuState,
    pub(super) draft_history_index: Option<usize>,
    pub(super) draft_history_stash: Option<String>,
    pub(super) session_id: String,
    pub(super) notice: Option<(Color, String)>,
    pub(super) pending_response: String,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) last_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    pub(super) context_window: u64,
    pub(super) context_is_estimate: bool,
    pub(super) follow_output: bool,
    pub(super) transcript_scroll: usize,
}

#[derive(Clone)]
pub(super) struct PendingSubmission {
    pub(super) content: String,
    pub(super) attachments: Vec<MessageAttachment>,
}

#[derive(Clone, Default)]
pub(super) struct CommandMenuState {
    pub(super) kind: Option<MenuKind>,
    pub(super) query: String,
    pub(super) selected: usize,
    pub(super) dismissed: bool,
    pub(super) placement: Option<CommandMenuPlacement>,
}

impl CommandMenuState {
    pub(super) fn reset(&mut self) {
        self.kind = None;
        self.query.clear();
        self.selected = 0;
        self.dismissed = false;
        self.placement = None;
    }
}

#[derive(Clone)]
pub(super) struct PathMenuEntry {
    pub(super) label: String,
    pub(super) is_dir: bool,
    pub(super) description: String,
    pub(super) insertion_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MenuKind {
    Commands,
    /// Path completion for a command's file argument (`/attach`, `/preview`).
    Path,
    /// `@name` sub-agent mentions in the composer.
    Mention,
}

/// A discovered skill surfaced as a user-typeable slash command (`/repo-study`),
/// so the `/` menu suggests it and submitting it invokes the skill. `name` is the
/// skill name; `description` is its one-line advert (already truncated for the
/// menu). The full body is loaded on invocation, not cached here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SkillCommand {
    pub(super) name: String,
    pub(super) description: String,
}

impl SkillCommand {
    pub(super) fn command_label(&self) -> String {
        format!("/{}", self.name)
    }

    /// What lands in the composer when the row is Tab-completed: `/name ` with a
    /// trailing space so the user can type the skill's input right away.
    pub(super) fn insertion_text(&self) -> String {
        format!("/{} ", self.name)
    }
}

/// A discovered sub-agent offered by the `@` mention menu: name + one-line
/// advert. Selecting it inserts `@name ` into the draft (it does not submit —
/// a mention is part of a message still being composed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AgentMention {
    pub(super) name: String,
    pub(super) description: String,
}

#[derive(Clone)]
pub(super) enum ComposerMenuEntry {
    Command(&'static SlashCommandSpec),
    Skill(SkillCommand),
    Path(PathMenuEntry),
    Agent(AgentMention),
}

impl ComposerMenuEntry {
    pub(super) fn label(&self) -> String {
        match self {
            // Help label, not the bare name, so arg-taking commands show their argument.
            Self::Command(command) => command.help_label.to_string(),
            Self::Skill(skill) => skill.command_label(),
            Self::Path(path) => path.label.clone(),
            Self::Agent(agent) => format!("@{}", agent.name),
        }
    }

    pub(super) fn description(&self) -> &str {
        match self {
            Self::Command(command) => command.description,
            Self::Skill(skill) => &skill.description,
            Self::Path(path) => &path.description,
            Self::Agent(agent) => &agent.description,
        }
    }
}

pub(super) struct VisibleCommandMenu {
    pub(super) kind: MenuKind,
    pub(super) entries: Vec<ComposerMenuEntry>,
    pub(super) selected: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CommandMenuPlacement {
    Above,
    Below,
}

impl ResumeRestoreState {
    pub(super) fn capture(app: &CodeTuiApp) -> Self {
        Self {
            key: app.key.clone(),
            copilot_tm: app.copilot_tm.clone(),
            raw_model: app.raw_model.clone(),
            model: app.model.clone(),
            format: app.format.clone(),
            history: app.history.clone(),
            draft: app.draft.clone(),
            draft_attachments: app.draft_attachments.clone(),
            cursor: app.cursor,
            command_menu: app.command_menu.clone(),
            draft_history_index: app.draft_history_index,
            draft_history_stash: app.draft_history_stash.clone(),
            session_id: app.session_id.clone(),
            notice: app.notice.clone(),
            pending_response: app.pending_response.clone(),
            pending_reasoning: app.pending_reasoning.clone(),
            pending_submit: app.pending_submit.clone(),
            last_usage: app.last_usage,
            context_tokens: app.context_tokens,
            context_window: app.context_window,
            context_is_estimate: app.context_is_estimate,
            follow_output: app.follow_output,
            transcript_scroll: app.transcript_scroll,
        }
    }
}

impl PickerState {
    pub(super) fn loading(title: &'static str, query: String, kind: PickerKind) -> Self {
        Self {
            title,
            query,
            items: Vec::new(),
            loading: true,
            selected: 0,
            kind,
            pending_delete: None,
            preview_scroll: 0,
            preview_scroll_for: None,
            scroll_top: 0,
            scroll_selected: usize::MAX,
        }
    }

    pub(super) fn ready(
        title: &'static str,
        query: String,
        items: Vec<PickerEntry>,
        kind: PickerKind,
    ) -> Self {
        Self {
            title,
            query,
            items,
            loading: false,
            selected: 0,
            kind,
            pending_delete: None,
            preview_scroll: 0,
            preview_scroll_for: None,
            scroll_top: 0,
            scroll_selected: usize::MAX,
        }
    }

    pub(super) fn filtered_items(&self) -> Vec<(usize, &PickerEntry)> {
        let mut filtered: Vec<(usize, &PickerEntry)> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches_fuzzy(&self.query, &item.search_text))
            .collect();
        // Rank prefix > substring > subsequence; stable, ties keep list order.
        if !self.query.is_empty() {
            filtered.sort_by_cached_key(|(_, item)| {
                crate::tui::score_match(&self.query, &item.search_text)
            });
        }
        filtered
    }

    pub(super) fn exact_match_index(&self) -> Option<usize> {
        let PickerKind::Model {
            auto_accept_exact, ..
        } = &self.kind
        else {
            return None;
        };
        if !*auto_accept_exact || self.query.is_empty() {
            return None;
        }
        self.filtered_items().iter().position(
            |(_, item)| matches!(&item.value, PickerValue::Model(model) if model == &self.query),
        )
    }

    /// Window at the persisted `scroll_top`, slid only as far as needed to keep
    /// a moved selection visible; unmoved selection = wheel pan owns the offset.
    /// The caller writes the first returned index back to `scroll_top`.
    pub(super) fn visible_items(&self, max_rows: usize) -> Vec<(usize, &PickerEntry)> {
        let filtered = self.filtered_items();
        if filtered.is_empty() || max_rows == 0 {
            return Vec::new();
        }

        let selected = self.selected.min(filtered.len().saturating_sub(1));
        let mut start = if self.scroll_selected == selected {
            self.scroll_top.min(filtered.len() - 1)
        } else {
            // Topmost start that still shows the whole selected row.
            let mut min_start = selected;
            let mut used_rows = filtered[selected].1.row_height();
            while min_start > 0 {
                let next_height = filtered[min_start - 1].1.row_height();
                if used_rows + next_height > max_rows {
                    break;
                }
                used_rows += next_height;
                min_start -= 1;
            }
            self.scroll_top.clamp(min_start, selected)
        };

        let mut end = start;
        let mut used_rows = 0usize;
        while end < filtered.len() {
            let next_height = filtered[end].1.row_height();
            if end > start && used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            end += 1;
        }
        // Fill the last page from above when the tail leaves slack.
        while start > 0 {
            let next_height = filtered[start - 1].1.row_height();
            if used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            start -= 1;
        }

        filtered[start..end]
            .iter()
            .enumerate()
            .map(|(offset, (_, item))| (start + offset, *item))
            .collect()
    }

    pub(super) fn select_prev(&mut self) {
        let len = self.filtered_items().len();
        if len == 0 {
            self.selected = 0;
        } else if self.selected == 0 {
            self.selected = len - 1;
        } else {
            self.selected -= 1;
        }
    }

    pub(super) fn select_next(&mut self) {
        let len = self.filtered_items().len();
        if len > 0 {
            self.selected = if self.selected + 1 >= len {
                0
            } else {
                self.selected + 1
            };
        }
    }

    pub(super) fn clear_pending_delete(&mut self) {
        self.pending_delete = None;
    }

    pub(super) fn selected_delete_target(&self) -> Option<DeleteConfirmTarget> {
        let (_, item) = self.filtered_items().get(self.selected).copied()?;
        match &item.value {
            PickerValue::Session(session) => Some(DeleteConfirmTarget {
                key_id: session.key_id.clone(),
                session_id: session.session_id.clone(),
            }),
            _ => None,
        }
    }

    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        let Some(target) = self.selected_delete_target() else {
            return false;
        };
        if self.pending_delete.as_ref() == Some(&target) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(target);
            false
        }
    }

    pub(super) fn delete_is_armed_for_selected(&self) -> bool {
        self.selected_delete_target()
            .is_some_and(|target| self.pending_delete.as_ref() == Some(&target))
    }

    pub(super) fn delete_is_armed_for_session(&self, preview: &SessionPreview) -> bool {
        self.pending_delete.as_ref().is_some_and(|target| {
            target.key_id == preview.key_id && target.session_id == preview.session_id
        })
    }
}

pub(super) enum SubmitAction {
    Send(String),
    Command(SlashCommand),
    /// `!cmd`: run a shell command locally (Claude Code's bash prefix).
    Shell(String),
}

/// How `cancel_inflight_request` disposes of the pending user turn. Agent turns
/// keep their (engine-consumed) message except under `Unsend`.
pub(super) enum CancelKind {
    Discard,
    /// Nothing produced yet — drop the message from the transcript and back into
    /// the composer, even for an agent turn.
    Unsend,
}

/// An in-flight `!cmd` local shell run. Independent of model turns (`sending`):
/// the background reader task streams output here line by line, rendered live in
/// the transcript's volatile tail and committed to a `local_command` history
/// entry when it finishes. `stdout`/`stderr` are capped by the reader (see
/// `run_shell_streaming`).
pub(super) struct LocalCommandRun {
    pub(super) task: JoinHandle<()>,
    /// Kills the PTY child so `esc` (and app exit) can stop a running command —
    /// the blocking PTY read can't be cancelled by aborting `task` alone.
    pub(super) killer: Box<dyn portable_pty::ChildKiller + Send + Sync>,
    pub(super) started_at: Instant,
    pub(super) command: String,
    pub(super) stdout: String,
    pub(super) stderr: String,
}

/// The full captured stdout/stderr of a finished/interrupted `!cmd` run, kept in
/// memory only (never persisted) so an inline-expanded block can show everything
/// even though the transcript and the on-disk session keep just a bounded preview.
/// Held in `local_outputs` keyed by history index; cleared on `/new` and resume.
/// The command line and exit status aren't stored here — an expanded block reads
/// those from its persisted `local_command` entry, which survives a resume.
#[derive(Clone)]
pub(super) struct LocalCommandOutput {
    pub(super) stdout: String,
    pub(super) stderr: String,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // Attachment only constructed on macOS (clipboard image)
pub(super) enum ClipboardPayload {
    Text(String),
    Attachment(MessageAttachment),
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TranscriptPoint {
    pub(super) row: usize,
    pub(super) column: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TranscriptSelection {
    pub(super) anchor: TranscriptPoint,
    pub(super) focus: TranscriptPoint,
}

impl TranscriptSelection {
    pub(super) fn is_empty(self) -> bool {
        self.anchor == self.focus
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TranscriptHitbox {
    pub(super) area: Rect,
    pub(super) first_row: usize,
    /// Display rows in order (body, tail sections, spinner), `Arc`-shared with
    /// the render caches so a frame publishes them without copying every row.
    pub(super) segments: Vec<std::sync::Arc<Vec<String>>>,
}

impl TranscriptHitbox {
    #[cfg(test)]
    pub(super) fn from_rows(area: Rect, first_row: usize, rows: Vec<String>) -> Self {
        Self {
            area,
            first_row,
            segments: vec![std::sync::Arc::new(rows)],
        }
    }

    pub(super) fn rows_len(&self) -> usize {
        self.segments.iter().map(|seg| seg.len()).sum()
    }

    pub(super) fn row(&self, mut i: usize) -> Option<&str> {
        for seg in &self.segments {
            match seg.get(i) {
                Some(row) => return Some(row),
                None => i -= seg.len(),
            }
        }
        None
    }

    pub(super) fn rows(&self) -> impl Iterator<Item = &str> {
        self.segments
            .iter()
            .flat_map(|seg| seg.iter().map(String::as_str))
    }
}

/// Snapshot of the rendered screen so a drag can copy from anywhere on it, not
/// just the transcript. `rows[i]` is screen row `area.y + i`, one symbol per
/// display column (CJK wide cells included).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ScreenSurface {
    pub(super) area: Rect,
    pub(super) rows: Vec<String>,
}

/// Which surface a drag-selection reads from: the scroll-aware transcript, or the
/// flat [`ScreenSurface`] (overlays, composer, footer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectionSurface {
    Transcript,
    Screen,
}

/// Memoizes the heavy transcript build + word-wrap across frames. The transcript
/// body (intro + history + the streamed reply + notice) is re-rendered and
/// re-wrapped only when its content or the render width actually changes — not on
/// every animation tick. The live spinner status line is deliberately excluded
/// (it changes every frame) and appended fresh at draw time, so spinner
/// animation and idle keystrokes hit the cache instead of reparsing all history.
pub(super) struct TranscriptCache {
    /// Fingerprint of the content that produced `body` (see `transcript_body_fp`).
    pub(super) fp: u64,
    /// Terminal width the `plain_prepass` height estimate was computed for.
    pub(super) area_width: u16,
    /// The logical body lines (no spinner), already compacted.
    pub(super) body: RenderedTranscript,
    /// Char-wrapped row count of `body` at `area_width - ACCENT_GUTTER_WIDTH`,
    /// used to size the transcript pane; with no scrollbar column reserved this
    /// is also the exact text width.
    pub(super) plain_prepass: usize,
    /// Text width `wrapped` is valid for (0 = not wrapped yet).
    pub(super) styled_width: u16,
    /// `body` word-wrapped to `styled_width`.
    pub(super) wrapped: Option<WrappedTranscript>,
}

/// Cross-frame memo of the VOLATILE tail (the streamed reply, a running `!cmd`'s
/// preview, and any notice) — the part `TranscriptCache` deliberately excludes.
/// The reply and command stream append-only, so a cheap fingerprint of their
/// lengths + the notice identifies the rendered output; caching the markdown
/// render AND the word-wrap turns the per-frame O(reply) re-parse and re-wrap
/// (which the 60fps spinner redraw makes O(reply²) over a long answer) into an
/// O(1) lookup whenever no new token arrived. The spinner itself is excluded (it
/// animates every frame) and wrapped fresh at compose time.
/// One display-ordered slice of the volatile tail, with its own wrap and
/// pane-height prepass memos so an unchanged section never re-renders.
pub(super) struct TailSection {
    pub(super) lines: Vec<StyledLine>,
    pub(super) bars: Vec<Option<Color>>,
    /// `lines` wrapped to the cache's `styled_width`; `None` when empty or stale.
    pub(super) wrapped: Option<WrappedTranscript>,
    /// Char-wrapped row count at the cache's `plain_width`.
    pub(super) prepass: Option<usize>,
}

impl TailSection {
    pub(super) fn new(lines: Vec<StyledLine>, bars: Vec<Option<Color>>) -> Self {
        Self {
            lines,
            bars,
            wrapped: None,
            prepass: None,
        }
    }

    pub(super) fn empty() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

/// The tail as display-ordered sections so a streamed token re-renders only
/// the live remainder: `head` (blank + reasoning window), `settled` (reply
/// prefix up to the last markdown-safe boundary, rendered once, immutable),
/// `live` (reply suffix + `!cmd` preview + notice, rebuilt per content change
/// but bounded by the trailing block, not the reply length).
pub(super) struct VolatileTailCache {
    /// Fingerprint of every tail input EXCEPT the reply length (see
    /// `volatile_tail_fp`); any change resets all sections.
    pub(super) fp: u64,
    pub(super) render_width: u16,
    /// `pending_response` byte length the sections currently cover.
    pub(super) reply_len: usize,
    /// `pending_response` bytes covered by `settled` (always a safe boundary).
    pub(super) settled_src: usize,
    /// Whether a settled chunk already placed the `◆ ` reply marker.
    pub(super) settled_marked: bool,
    pub(super) head: TailSection,
    pub(super) settled: Vec<TailSection>,
    pub(super) live: TailSection,
    /// Width the sections' `prepass` counts are valid for (0 = none yet).
    pub(super) plain_width: u16,
    /// Width the sections' `wrapped` forms are valid for (0 = none yet).
    pub(super) styled_width: u16,
}

impl VolatileTailCache {
    pub(super) fn sections(&self) -> impl Iterator<Item = &TailSection> {
        std::iter::once(&self.head)
            .chain(self.settled.iter())
            .chain(std::iter::once(&self.live))
    }

    pub(super) fn sections_mut(&mut self) -> impl Iterator<Item = &mut TailSection> {
        std::iter::once(&mut self.head)
            .chain(self.settled.iter_mut())
            .chain(std::iter::once(&mut self.live))
    }
}

/// Live drag that has reached the top/bottom edge of the transcript: the event
/// loop keeps scrolling in `dir` (−1 up, +1 down) and re-anchoring the focus to
/// `column` so a selection can run past a single screenful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DragAutoscroll {
    pub(super) dir: i8,
    pub(super) column: u16,
}

/// Tracks consecutive left-clicks so double/triple clicks select word/line.
#[derive(Clone, Copy, Debug)]
pub(super) struct ClickTracker {
    pub(super) at: Instant,
    pub(super) point: TranscriptPoint,
    pub(super) count: u8,
}

/// A brief floating confirmation that auto-expires and fades — used for copy
/// confirmations and transient mode toggles (e.g. auto-approve), so they flash
/// and vanish instead of lingering in the transcript.
#[derive(Clone, Debug)]
pub(super) struct Toast {
    pub(super) text: String,
    pub(super) expires_at: Instant,
    pub(super) anchor: ToastAnchor,
}

/// `Corner`: quiet confirmations bottom-right. `Center`: mid-transcript, for
/// state flips the eye must not miss (e.g. the Ctrl+T thinking cycle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToastAnchor {
    Corner,
    Center,
}

/// Blend `from` toward `to` (t: 0→1); non-`Rgb` pairs snap at halfway.
pub(super) fn fade_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    match (from, to) {
        (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
            let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t) as u8;
            Color::Rgb(mix(r1, r2), mix(g1, g2), mix(b1, b2))
        }
        _ if t < 0.5 => from,
        _ => to,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SlashCommand {
    New,
    Exit,
    Resume(Option<String>),
    Model(Option<String>),
    Key(Option<String>),
    Attach(String),
    /// Copy the Nth-latest assistant reply (default 1 = most recent) to the clipboard.
    Copy(Option<usize>),
    /// Side preview pane: `<path|url>` pins, `off` closes, bare toggles off.
    Preview(Option<String>),
    /// Agent skills: bare opens the overlay; `add …` / `rm <name>` manage them.
    Skills(Option<String>),
    /// Named sub-agents: bare opens the overlay; `rm <name>` deletes one.
    Agents(Option<String>),
    /// MCP servers: bare opens the overlay; `add …` / `rm <name>` manage them.
    Mcp(Option<String>),
    /// Goal mode: `<objective>` works autonomously until done; bare shows status,
    /// `stop` ends it.
    Goal(Option<String>),
    /// Plan mode: `<objective>` investigates read-only and drafts an
    /// implementation plan; `go` executes it in a fresh context; bare shows
    /// status, `stop` discards the pending plan.
    Plan(Option<String>),
    /// Built-in `create-skill` command: starts the guided create/improve-a-skill
    /// workflow. The optional argument is the initial intent (what the skill
    /// should do); bare just opens the workflow.
    CreateSkill(Option<String>),
    /// Invoke a discovered skill by name as a slash command (`/repo-study`), the
    /// way Claude Code / grok CLI surface skills. `argument` is the optional text
    /// after the name, passed to the skill (substituted into `$ARGUMENTS` if the
    /// body uses it, else appended as input).
    Skill {
        name: String,
        argument: Option<String>,
    },
    /// Open the rewind picker: jump back to an earlier turn, reverting the file
    /// edits made since and restoring that turn's prompt to the composer.
    Rewind,
    /// Open the `/config` overlay of chat preferences.
    Config,
    /// `/compact` folds older turns via the LLM; `fast` clears stale tool output only.
    Compact {
        fast: bool,
    },
    /// `/context` — read-only viewer of the injected context digest.
    Context,
    /// `/session` — this session's id, provenance, model, key, and resume command.
    Session,
    /// Share this chat: bare/`start` opens a viewer URL (re-shown if already
    /// live); `stop` ends it.
    Share(Option<String>),
    /// aivo account flows (aivo provider only).
    Login,
    Logout,
    Usage,
    Help,
}

/// A turn-finish event captured while the typewriter still has buffered text to
/// reveal. The event loop replays it once the buffer drains so the final reply
/// types out fully before the turn commits. Mirrors the finish-bearing
/// [`RuntimeEvent`] variants.
pub(super) enum DeferredFinish {
    Chat {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    Agent {
        steps: usize,
        tokens: u64,
        context_tokens: u64,
    },
}

/// Composer→engine steering handoff; `std::sync` mutex — never held across an await.
pub(super) type SteeringQueue = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// Owning queue of a unified queue row; variants in delivery order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QueueSegment {
    Steering,
    Command,
    Message,
}

/// One row of the unified queued-input view, snapshotted per key event/frame;
/// ops revalidate `offset`+`recall` against the owning queue before mutating.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct QueuedRow {
    pub(super) segment: QueueSegment,
    pub(super) offset: usize,
    /// Single-line label; width truncation happens at render time.
    pub(super) display: String,
    /// Text a recall puts back into the composer.
    pub(super) recall: String,
}

/// One delegate's live row in a parallel sub-agent batch.
pub(super) struct SubagentRow {
    pub(super) name: String,
    /// Present-tense current action, precomputed at event time so rendering stays pure.
    pub(super) action: String,
    pub(super) step: usize,
    pub(super) started: Instant,
    /// Last gated tool auto-denied for this delegate.
    pub(super) denied: Option<String>,
    /// Latest output-token estimate, summed into the footer's turn total
    /// (`done` pins the final value here; the row itself renders `done` only).
    pub(super) live_tokens: u64,
    /// (produced an answer, steps, tokens, runtime) once finished.
    pub(super) done: Option<(bool, usize, u64, std::time::Duration)>,
}

/// One cursor-ACP turn's `/rewind` checkpoint (see `acp_checkpoints`).
pub(super) struct AcpCheckpoint {
    /// User row in `history`; validated against `prompt` at use time so index
    /// drift degrades to conversation-only, never a wrong revert.
    pub(super) history_index: usize,
    pub(super) prompt: String,
    /// Pre-turn tree snapshot; `None` = pending or failed (non-revertible alone).
    pub(super) tree: Option<String>,
    /// Paths the turn changed; `None` = still open (interrupted turn) →
    /// finalized lazily at rewind time.
    pub(super) changed: Option<Vec<std::path::PathBuf>>,
}

pub(super) enum RuntimeEvent {
    Delta(ChatResponseChunk),
    Finished {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    ModelsLoaded(std::result::Result<Vec<ModelChoice>, String>),
    /// The background catalog warm finished — re-resolve the active model's
    /// limits (window + `/effort` levels) so catalog-advertised efforts surface.
    CatalogWarmed,
    ResumeLoaded {
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    },
    /// Bare-`/plan` scan result: `tip` replaces the notice only while it
    /// still reads `expect` (the entry notice).
    PlanHintReady {
        expect: String,
        tip: String,
    },
    /// A `/resume` preview finished loading. Content-addressed by
    /// `(session_id, updated_at)`: always cached, never "stale".
    SessionPreviewLoaded {
        session_id: String,
        entry: PreviewEntry,
    },
    /// An inline-image preview prep finished; `None` = source undecodable.
    ImagePreviewReady {
        key: u64,
        preview: Option<Box<crate::services::terminal_graphics::EncodedPreview>>,
    },
    /// A cursor-agent ACP session finished opening on a background task. The
    /// event loop stores it on the app so subsequent turns reuse it without
    /// paying the Node.js startup cost again.
    CursorSessionOpened(crate::services::cursor_acp::CursorAcpSession),
    /// A cursor turn's pre-prompt `/rewind` tree snapshot, taken on the turn
    /// task. `turn_seq` (the `cursor_turn_seq` stamp) drops stale events from
    /// an aborted task racing a newer turn.
    AcpSnapshot {
        turn_seq: u64,
        tree: Option<String>,
    },
    /// The paths a cursor turn changed, at turn end. `None` = diff failed →
    /// the turn reads non-revertible instead of "changed nothing".
    AcpChanged {
        turn_seq: u64,
        changed: Option<Vec<std::path::PathBuf>>,
    },
    /// The agent engine invoked a tool — render a `→ verb(args)` transcript step.
    /// `id` (cursor's `toolCallId`) correlates a later `AgentToolUpdate`; `None`
    /// for the in-process agent, which reports results separately.
    AgentToolCall {
        id: Option<String>,
        name: String,
        args: serde_json::Value,
        /// Pre-edit start line of each diff pair (aligned with `edit_diffs`), so
        /// the card can number rows; empty for non-edit and cursor calls.
        line_starts: Vec<Option<usize>>,
        /// A `write_file`'s bounded pre-write snapshot (see `capture_pre_write`)
        /// for a real diff card; `None` for every other tool.
        old_content: Option<String>,
    },
    /// Enriches an earlier `AgentToolCall` (matched by `id`) in place: the
    /// resolved target (real path/pattern) and/or a compact result. Cursor only —
    /// its start event omits the target, which arrives in a `tool_call_update`.
    AgentToolUpdate {
        id: String,
        args: Option<serde_json::Value>,
        result: Option<String>,
        failed: bool,
    },
    /// Live `run_bash` output chunk — feeds the streaming tail, not the transcript.
    AgentToolOutput {
        chunk: String,
    },
    /// The agent engine's tool returned — render the `⎿ result` step.
    AgentToolResult {
        content: String,
    },
    /// The agent's just-streamed output was a tool call written as text: drop the
    /// uncommitted segment so the markup never reaches the scrollback.
    AgentDiscardSegment,
    /// The engine consumed a mid-turn interjection — commit it at the injection point.
    AgentSteered(String),
    /// Peer mail surfaced mid-turn, in `transcript_display` form; committed as a ✉ row.
    AgentSessionMail(String),
    /// A background MCP connect finished. Carries the client (possibly empty) and
    /// the generation it started under; the event loop caches it and rebuilds the
    /// engine if it brought tools, but drops a result from a stale generation.
    McpConnected {
        client: std::sync::Arc<crate::agent::mcp::McpClient>,
        generation: u64,
    },
    /// One MCP server's handshake resolved mid-connect (servers connect
    /// concurrently). Lets the open `/mcp` overlay flip that single row to its real
    /// status while the rest are still connecting. `generation` guards against a
    /// stale connect; `health`/`status` are the already-mapped display values.
    McpServerProgress {
        name: String,
        status: String,
        health: McpHealth,
        generation: u64,
    },
    /// A background OAuth authorize for an HTTP MCP server produced its browser
    /// URL — shown as a notice so the user has it if the auto-opened browser
    /// didn't appear.
    McpAuthorizeUrl {
        url: String,
    },
    /// A background OAuth authorize finished. On `Ok`, the event loop persists the
    /// credential and reconnects so the now-authorized server's tools appear.
    McpAuthorized {
        name: String,
        result: std::result::Result<crate::services::mcp_oauth::McpOAuthCredential, String>,
    },
    /// The agent set/updated its task plan (update_plan tool). Carries the JSON
    /// array of `{step, status}`; rendered as a checklist card in the transcript.
    AgentPlan(serde_json::Value),
    /// Engine status line (compaction, step limit, …) — shown as a notice.
    AgentNotice(String),
    /// The engine ended the turn early (guard stop / step limit) — typed, for /goal steering.
    AgentTurnStop(crate::agent::engine::TurnStop),
    /// A mutating tool needs approval. The event loop shows a permission card
    /// and replies with the decision; the engine task awaits `reply`.
    /// `once_only` = a floor prompt (never remembered); the card hides "always".
    AgentPermission {
        tool: String,
        preview: Option<String>,
        once_only: bool,
        reply: tokio::sync::oneshot::Sender<crate::agent::protocol::Decision>,
    },
    /// The agent's `switch_model`/`set_effort` tools: apply the change and reply to the
    /// waiting engine task (Ok = confirmation, Err = why not). Same oneshot pattern as
    /// [`AgentPermission`](Self::AgentPermission).
    AgentSwitchModel {
        model: String,
        reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
    },
    AgentSetEffort {
        level: String,
        reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
    },
    /// The agent's `ask_user` tool: show the card, reply with the answer. Same
    /// oneshot pattern as [`AgentPermission`](Self::AgentPermission).
    AgentAskUser {
        question: String,
        options: Vec<crate::agent::ask::AskOption>,
        allow_free_text: bool,
        multi_select: bool,
        record_history: bool,
        reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
    },
    /// The agent's edit-review gate: show the pending edits, reply with the verdict.
    AgentReviewEdits {
        items: Vec<crate::agent::review::ReviewItem>,
        reply: tokio::sync::oneshot::Sender<crate::agent::review::ReviewDecision>,
    },
    /// The agent's `exit_plan_mode` tool: show the plan + approval card, reply with
    /// the verdict. Same oneshot pattern as [`AgentPermission`](Self::AgentPermission).
    AgentPlanApproval {
        plan: String,
        reply: tokio::sync::oneshot::Sender<
            std::result::Result<crate::agent::protocol::PlanDecision, String>,
        >,
    },
    /// Live context-window fill from the agent engine mid-turn: `measured` true =
    /// a provider step total (exact), false = a chars/4 request estimate. Moves
    /// the footer's context stat during an agent turn instead of only at the end.
    AgentContext {
        tokens: u64,
        measured: bool,
    },
    /// Cursor's `usage_update` reported the real context window — beats the
    /// id-parsed guess.
    AgentContextWindow(u64),
    /// The turn's cumulative generated (output) tokens so far — drives the live
    /// per-turn counter in the status line.
    AgentTurnTokens(u64),
    /// A delegated sub-agent began a step — updates the parent's status-line label
    /// (`↳ <agent>: <action> · step N`) only, never a transcript card.
    AgentSubActivity {
        agent: String,
        tool: String,
        args: serde_json::Value,
        step: usize,
    },
    /// A parallel sub-agent batch started — one live row per delegate, slot order.
    AgentSubBegin {
        labels: Vec<String>,
    },
    /// Slot-tagged counterpart of [`AgentSubActivity`](Self::AgentSubActivity).
    AgentSubSlot {
        slot: usize,
        agent: String,
        tool: String,
        args: serde_json::Value,
        step: usize,
    },
    /// A parallel delegate's gated tool call was auto-denied.
    AgentSubDenied {
        slot: usize,
        tool: String,
    },
    AgentSubTokens {
        slot: usize,
        tokens: u64,
    },
    /// A parallel delegate finished (`ok` = it produced an answer).
    AgentSubDone {
        slot: usize,
        ok: bool,
        steps: usize,
        tokens: u64,
    },
    /// The batch is over — retire the rows (results land as cards).
    AgentSubFinish,
    /// An agent error (e.g. an LLM/API failure) — shown as an error-hued notice.
    AgentError(String),
    /// The agent turn finished (engine called `footer`) — commit the turn.
    AgentFinished {
        steps: usize,
        tokens: u64,
        /// Last step's real prompt+completion (context-window fill), 0 if the
        /// provider reported no usage.
        context_tokens: u64,
    },
    /// One image described; mirrored into the session cache.
    ImageDescribed {
        hash: String,
        text: String,
    },
    /// Describe failed before the turn's first request (see
    /// `abort_undispatched_turn`).
    DescribeFailed {
        message: String,
    },
    /// A `!cmd` local shell run produced one output line — appended to the live
    /// in-progress run and shown immediately in the transcript tail.
    LocalCommandLine {
        is_err: bool,
        line: String,
    },
    /// A `!cmd` local shell run finished (drained, capped, or timed out). Commits
    /// the captured output as a `local_command` history entry. `truncated` marks
    /// output cut short by the capture caps.
    LocalCommandDone {
        exit_code: i64,
        truncated: bool,
    },
    /// A background `/skills add <source>` install finished, off the event loop.
    SkillInstalled {
        source: String,
        /// `-p/--project`: the install targeted the repo's `.agents/skills`.
        project: bool,
        result: std::result::Result<crate::agent::skills::InstallReport, String>,
    },
    /// A fetch found several skills — the staged tree is handed to the picker.
    SkillInstallPick {
        source: String,
        project: bool,
        staged: crate::agent::skills::StagedInstall,
    },
    /// A `/share` (or `--share`) start finished: `Ok` the handle, `Err` the reason.
    LiveShareReady {
        /// `live_share_gen` at start time; stale (dropped) after a stop//new//resume.
        share_gen: u64,
        result: std::result::Result<crate::services::share_live::LiveShareHandle, String>,
    },
    /// `/login`: device code + verification URL, or failure to start.
    AccountLoginPrompt {
        account_gen: u64,
        result: std::result::Result<(String, String), String>,
    },
    /// `/login` resolved — `Ok` is the ready-to-show notice.
    AccountLoginDone {
        account_gen: u64,
        result: std::result::Result<String, String>,
    },
    /// `/logout`: the server-side unlink resolved.
    AccountLogoutDone {
        account_gen: u64,
        result: std::result::Result<(), String>,
    },
}

/// A live in-process agent for the current chat, keyed by the (key, model) it
/// was built for so a key/model switch rebuilds it. The engine owns the real
/// LLM conversation; chat history is a display log.
pub(super) struct AgentSession {
    pub(super) key_id: String,
    pub(super) model: String,
    pub(super) engine: std::sync::Arc<tokio::sync::Mutex<crate::agent::engine::AgentEngine>>,
}

/// A pending tool-permission prompt: shown as a card until the user answers,
/// then `reply` delivers the decision to the waiting engine task. `once_only`
/// hides "always" (never remembered).
pub(super) struct PendingPermission {
    pub(super) tool: String,
    pub(super) preview: Option<String>,
    pub(super) once_only: bool,
    pub(super) reply: tokio::sync::oneshot::Sender<crate::agent::protocol::Decision>,
}

/// A pending `ask_user` card: `reply` delivers the chosen/typed answer to the
/// waiting engine task (`selected` = highlighted option). Dropping it unreplied
/// resolves the engine's future to an error.
pub(super) struct PendingAskUser {
    pub(super) question: String,
    pub(super) options: Vec<crate::agent::ask::AskOption>,
    pub(super) allow_free_text: bool,
    /// Multi-select (free text off): `checked` is the per-option state, `Enter`
    /// returns the checked labels joined by ", ".
    pub(super) multi_select: bool,
    pub(super) checked: Vec<bool>,
    pub(super) selected: usize,
    pub(super) record_history: bool,
    pub(super) reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
}

/// A pending edit-review card: `count` edits as precomputed scrollable `body` diff
/// lines and `reply` for the verdict. Dropping it unreplied resolves to `Reject`.
pub(super) struct PendingReview {
    pub(super) count: usize,
    pub(super) body: Vec<ratatui::text::Line<'static>>,
    pub(super) scroll: u16,
    pub(super) reply: tokio::sync::oneshot::Sender<crate::agent::review::ReviewDecision>,
}

/// A pending plan-approval card (`exit_plan_mode`): the plan as precomputed
/// scrollable `body`, the highlighted `selected` option, and `reply`. Dropping it
/// unreplied resolves to the dismissal directive (plan mode stays on).
pub(super) struct PendingPlanApproval {
    pub(super) body: Vec<ratatui::text::Line<'static>>,
    pub(super) scroll: u16,
    pub(super) selected: usize,
    pub(super) reply: tokio::sync::oneshot::Sender<
        std::result::Result<crate::agent::protocol::PlanDecision, String>,
    >,
}

/// Active `/goal` autonomous loop: after each agent turn the app auto-continues
/// toward `objective` until the completion marker, the `max` turn cap, an errored
/// turn, or a user interrupt. `iteration` = 1-based turn number. `None` = off.
#[derive(Clone)]
pub(super) struct GoalState {
    pub(super) objective: String,
    pub(super) iteration: usize,
    pub(super) max: usize,
    /// `history.len()` when this goal armed. Completion/error detection ignores
    /// rows below it, so a reply from before the goal can't end a fresh loop
    /// (e.g. a queued `/goal` restart right after a turn that said the marker).
    pub(super) msg_floor: usize,
}

/// One persisted input-history entry, tagged with the launch dir it was typed
/// in. Legacy plain-text lines load with an empty `cwd` (shown everywhere).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct DraftHistoryEntry {
    pub(super) cwd: String,
    pub(super) text: String,
}

pub(super) struct CodeTuiApp {
    pub(super) session_store: SessionStore,
    pub(super) cache: ModelsCache,
    pub(super) client: Client,
    pub(super) key: ApiKey,
    pub(super) copilot_tm: Option<Arc<CopilotTokenManager>>,
    /// Chat's isolated sandbox dir (cursor cwd / display for non-agent chats).
    pub(super) cwd: String,
    /// The real launch dir — where the agent reads/edits files, and what the
    /// header/footer show for agent-capable chats (a safety signal).
    pub(super) real_cwd: String,
    /// Current git branch of `display_cwd` (when it's a repo), shown after the cwd
    /// in the footer. Refreshed on a throttle (see `refresh_git_branch`) so a
    /// checkout — by the user or the agent — shows up without re-reading
    /// `.git/HEAD` every frame. `None` when the dir isn't a git work tree.
    pub(super) git_branch: Option<String>,
    /// When `git_branch` was last refreshed, to throttle the `.git/HEAD` read.
    pub(super) git_branch_checked_at: Option<Instant>,
    pub(super) raw_model: String,
    pub(super) model: String,
    /// Upstream model captured from the most recent turn so heartbeat
    /// saves (no fresh turn in scope) can still write `billed_model` to
    /// the session index. Cleared on key/model switch and resume.
    pub(super) billed_model: Option<String>,
    /// `raw_model` frozen at dispatch — a mid-turn switch applies next turn, so
    /// assistant commits stamp from here, never from live `raw_model`.
    pub(super) turn_model: Option<String>,
    pub(super) format: ChatFormat,
    pub(super) history: Vec<ChatMessage>,
    pub(super) draft: String,
    pub(super) draft_attachments: Vec<MessageAttachment>,
    pub(super) cursor: usize,
    pub(super) command_menu: CommandMenuState,
    /// Discovered skills offered as user-typeable slash commands (`/repo-study`).
    /// Refreshed from `discover_skills` (minus the `/skills` disabled set) at
    /// startup and after any skill mutation; read by the `/` menu and command
    /// resolver. Empty when none; its length feeds the welcome chip.
    pub(super) skill_commands: Vec<SkillCommand>,
    /// Subagent profiles discovered for the working dir, last time the set was
    /// checked (startup + after each turn). Compared post-turn — full structs,
    /// since the engine snapshots profiles at build and never re-reads them — so
    /// a profile the agent just authored or edited (via the create-agent skill)
    /// drops the cached engine and takes effect next turn.
    pub(super) last_subagents: Vec<crate::agent::subagents::Subagent>,
    /// Enabled MCP servers for the welcome chip; refreshed on `/mcp` changes.
    pub(super) mcp_configured_count: usize,
    /// The welcome-tip entry showing now; advanced by `tick_welcome_tip`.
    pub(super) welcome_tip_index: usize,
    /// When the current tip was shown; `None` off the welcome screen so it restarts
    /// with a full interval on return.
    pub(super) welcome_tip_rotated_at: Option<Instant>,
    /// Up-arrow recall list for the current launch dir — the cwd-filtered view
    /// of `draft_history_all` that `history_prev`/`history_next` walk.
    pub(super) draft_history: Vec<String>,
    /// Full global history across every dir; persisted source of truth that the
    /// `draft_history` view is derived from.
    pub(super) draft_history_all: Vec<DraftHistoryEntry>,
    pub(super) draft_history_index: Option<usize>,
    pub(super) draft_history_stash: Option<String>,
    pub(super) session_id: String,
    pub(super) overlay: Overlay,
    /// Last `/config` row the user stood on — restored the next time the modal
    /// opens in this session (not persisted).
    pub(super) last_config_setting: Option<ConfigSetting>,
    pub(super) notice: Option<(Color, String)>,
    pub(super) pending_response: String,
    /// Stream text received but not yet revealed by the typewriter. Deltas land
    /// here; [`tick_typewriter`](CodeTuiApp::tick_typewriter) drips it into
    /// `pending_response` (the displayed reply) over successive frames.
    pub(super) incoming_buffer: String,
    /// A turn-finish event held back until the typewriter has revealed the whole
    /// buffer, so the tail types out instead of snapping in at the end.
    pub(super) pending_finish: Option<DeferredFinish>,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) sending: bool,
    pub(super) request_started_at: Option<Instant>,
    /// Context fill before a manual `/compact` (LLM) turn, so the finish path reports
    /// the freed delta and skips the duration marker. `None` outside a compact.
    pub(super) compact_before: Option<u64>,
    /// Current tool step, present-tense (`running grep`), when it started, and
    /// its timeout budget in seconds (`run_bash` only). Feeds the status label.
    pub(super) last_tool_action: Option<(String, Instant, Option<u64>)>,
    /// Last frame tick seen while a decision card was up — `tick_decision_wait`
    /// pushes the step + turn clocks forward so human decision time never reads
    /// as tool runtime or inflates `✻ Done in …`.
    pub(super) wait_tick: Option<Instant>,
    /// When the turn last produced any runtime event, for the stall label.
    pub(super) last_stream_activity: Option<Instant>,
    /// Live rows under the spinner for a parallel sub-agent batch (slot-indexed);
    /// cleared on batch finish / turn end.
    pub(super) subagent_rows: Vec<SubagentRow>,
    /// Parent turn tokens at batch start; slot totals sum on top.
    pub(super) subagent_token_base: u64,
    /// Streaming tail of the in-flight `run_bash`; cleared when the tool returns.
    pub(super) tool_output_tail: std::collections::VecDeque<String>,
    /// Unterminated last line of the stream (rendered too, for progress output).
    pub(super) tool_output_partial: String,
    /// This turn's cumulative generated tokens (status-line tail). Reset at turn
    /// start; fed by `AgentTurnTokens` (agent) or `Usage` (plain chat).
    pub(super) turn_output_tokens: u64,
    /// Streamed bytes this turn / at the last measured update; the gap ÷4 keeps
    /// the status tail ticking between round-boundary measurements.
    pub(super) turn_stream_chars: u64,
    pub(super) turn_stream_chars_measured: u64,
    /// Tool calls this turn (status-line tail) — the growing count is the
    /// visible proof a long turn is advancing, not wedged.
    pub(super) turn_steps: u64,
    /// A connection retry is in progress → status reads "Working", not
    /// "Thinking". Set on the retry notice, cleared on progress.
    pub(super) retrying: bool,
    /// Last turn's generation rate, frozen by `capture_turn_tps`; not persisted.
    pub(super) last_turn_tps: Option<f64>,
    /// Last turn's cache-read share of the inclusive prompt; `None` = none reported.
    pub(super) last_cache_hit_pct: Option<u8>,
    pub(super) last_usage: Option<TokenUsage>,
    /// Provider-measured usage streamed mid-turn (Anthropic reports it from
    /// `message_start`; OpenAI/Responses/Google only at the end). Drives the
    /// footer's context-fill while `sending`, then folds into `last_usage` at
    /// turn end. `None` outside an in-flight turn.
    pub(super) live_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    /// Cumulative provider-measured tokens for the CURRENT chat session (across
    /// all its turns), persisted into the chat index entry on every save so
    /// `aivo stats --since` can attribute windowed chat usage. Reset on `/new`,
    /// re-seeded from the stored entry on resume.
    pub(super) session_tokens: crate::services::session_store::SessionTokens,
    /// Estimated session spend in USD (snapshot pricing × each turn's measured
    /// usage); 0 when the model has no known pricing. Reset on `/new`.
    pub(super) session_cost_usd: f64,
    /// Active model's context window (tokens), 0 = unknown. Cached on model/key
    /// change for the footer utilization stat; see `refresh_context_window`.
    pub(super) context_window: u64,
    /// `--max-context` manual override (tokens); wins over the resolved window in
    /// `refresh_context_window` and the engine build. Session-only.
    pub(super) context_window_override: Option<u64>,
    /// Context-digest block, re-appended to the system prompt per engine build.
    pub(super) injected_context: Option<String>,
    /// One-line `injected_context` summary, shown as the `/context` header.
    pub(super) injected_context_summary: Option<String>,
    /// `context_tokens` is a chars/4 estimate of the visible transcript, not a
    /// provider-measured count (cursor ACP and agents-without-usage). The footer
    /// marks these with `~` since the model's real context is larger.
    pub(super) context_is_estimate: bool,
    pub(super) follow_output: bool,
    /// Bumped on any in-place edit of a history entry (cursor tool-call
    /// enrichment) so the transcript cache fingerprint invalidates — the
    /// fingerprint otherwise assumes entries are only appended/cleared.
    pub(super) transcript_revision: u64,
    pub(super) transcript_scroll: usize,
    pub(super) transcript_width: u16,
    pub(super) transcript_view_height: u16,
    /// Max scroll offset the last render computed from the composed rows. The
    /// hot scroll handlers reuse it (per wheel event) so a fast scroll skips the
    /// full `build_transcript` rebuild `max_scroll` would do; `None` before the
    /// first render falls back to recomputing. At most one frame stale, and the
    /// render re-clamps `transcript_scroll` each pass, so staleness is benign.
    pub(super) last_max_scroll: Option<usize>,
    pub(super) transcript_hitbox: Option<TranscriptHitbox>,
    /// Click region of the "jump to bottom" pill from the last render; `None` when
    /// it isn't shown (transcript pinned to the bottom, or no overflow).
    pub(super) jump_to_bottom_hit: Option<Rect>,
    /// Click region of the footer session id from the last render; clicking it
    /// opens the session-detail overlay. `None` when the id isn't shown (narrow
    /// terminal or empty id).
    pub(super) session_id_hit: Option<Rect>,
    /// Footer `● sharing` badge click region; `None` while not sharing.
    pub(super) share_badge_hit: Option<Rect>,
    /// Footer effort badge click region — opens `/config` on the Thinking row.
    /// `None` when no effort badge is shown (or it's Cursor's model-id tier).
    pub(super) effort_badge_hit: Option<Rect>,
    /// The open overlay's list scrollbar from the last render, if any.
    pub(super) scrollbar_hit: Option<ScrollbarHit>,
    /// Active thumb drag: grab row-offset inside the thumb.
    pub(super) scrollbar_drag: Option<u16>,
    /// Preview pane `✕` click regions (header + hint); empty while not drawn.
    pub(super) preview_close_hits: Vec<Rect>,
    /// The composer text region from the last render, for mouse cursor-placement
    /// and the key-handler's wrap math (it needs the width before the next frame).
    /// `None` until the first render.
    pub(super) composer_text_area: Option<Rect>,
    /// Vertical scroll (in visual rows) of the draft within the composer, so the
    /// cursor stays visible when a multi-line draft outgrows the composer's rows.
    /// Recomputed each render from the cursor position; never persisted.
    pub(super) composer_scroll: usize,
    /// Per-frame render products and cross-frame memos; written by
    /// `render_impl.rs`, hitboxes read back by mouse handling.
    pub(super) render_cache: RenderCache,
    pub(super) transcript_selection: Option<TranscriptSelection>,
    pub(super) transcript_drag_active: bool,
    /// Full-screen drag selection (overlays, composer, footer), mutually exclusive
    /// with `transcript_selection` — starting either drag clears the other.
    pub(super) screen_selection: Option<TranscriptSelection>,
    pub(super) screen_drag_active: bool,
    pub(super) screen_surface: Option<ScreenSurface>,
    /// Live edge auto-scroll state while dragging a selection.
    pub(super) drag_autoscroll: Option<DragAutoscroll>,
    /// When the last auto-scroll step fired, to throttle the scroll rate.
    pub(super) last_autoscroll: Option<Instant>,
    /// Last left-click, for double/triple-click word/line selection.
    pub(super) last_click: Option<ClickTracker>,
    /// While set (and not expired) the selection is painted in the bright flash
    /// color; on expiry the selection auto-clears.
    pub(super) selection_flash_until: Option<Instant>,
    pub(super) scroll_speed: usize,
    /// Bare Up/Down at the composer edge scroll the transcript instead of walking
    /// draft history — for mobile terminals (Termux) where swipes arrive as arrows.
    pub(super) swipe_scroll: bool,
    pub(super) toast: Option<Toast>,
    pub(super) tx: UnboundedSender<RuntimeEvent>,
    pub(super) rx: UnboundedReceiver<RuntimeEvent>,
    pub(super) response_task: Option<JoinHandle<()>>,
    pub(super) resume_task: Option<JoinHandle<()>>,
    pub(super) resume_request_id: u64,
    pub(super) loading_resume: Option<LoadingResume>,
    pub(super) resume_restore_state: Option<ResumeRestoreState>,
    /// `/resume` picker preview state; only `event_loop_impl.rs` drives it.
    pub(super) session_preview: SessionPreviewState,
    /// Inline transcript image previews (Kitty graphics); see `inline_images.rs`.
    pub(super) inline_images: InlineImageState,
    /// The pinned side preview pane (`/preview`, agent `preview` tool).
    pub(super) preview_pane: Option<PreviewPane>,
    /// `/config` inline-preview toggle; off keeps `inline_images.caps` disabled.
    pub(super) inline_images_enabled: bool,
    /// The kitty deletes on disable need the terminal writer — event loop only.
    pub(super) pending_inline_image_cleanup: bool,
    /// Startup detection result, kept so a mid-session re-enable can restore caps.
    pub(super) detected_graphics_caps: crate::services::terminal_graphics::GraphicsCaps,
    pub(super) reduce_motion: bool,
    pub(super) frame_tick: usize,
    pub(super) picker_hitbox: Option<PickerHitbox>,
    /// Split overlay's right-pane rect from the last render (`None` = single-pane);
    /// the key/mouse "split active" signal — a draw always precedes input.
    pub(super) overlay_detail_area: Option<Rect>,
    pub(super) exit_confirm_pending: bool,
    /// Armed by one Esc during a `/goal` turn; a second consecutive Esc stops the loop.
    pub(super) goal_stop_confirm_pending: bool,
    /// Ctrl+X pressed; the next key completes (Ctrl+E → external editor) or cancels the chord.
    pub(super) pending_ctrl_x: bool,
    /// Chord fired: the event loop opens the draft in $VISUAL/$EDITOR before the next repaint.
    pub(super) pending_external_edit: bool,
    /// The Enter being dispatched is part of a Windows paste burst — insert a
    /// newline instead of submitting (see `drain_input`).
    pub(super) paste_burst_newline: bool,
    /// Live `cursor-agent acp` connection scoped to the current chat session.
    /// `None` outside of cursor keys and before the first turn.
    pub(super) cursor_acp_session: Option<crate::services::cursor_acp::CursorAcpSession>,
    /// Background open of the cursor ACP session, started at TUI startup so
    /// the connect overlaps the user typing. The first turn `take`s and awaits
    /// this instead of opening its own — exactly one session is ever created.
    pub(super) cursor_prewarm: Option<
        tokio::task::JoinHandle<
            std::result::Result<crate::services::cursor_acp::CursorAcpSession, String>,
        >,
    >,
    /// `/rewind` checkpoints for cursor ACP turns — cursor runs its tools
    /// out-of-process, so the engine's lazy pre-edit snapshot never fires.
    /// In-memory, cleared/retained alongside `agent_turn_indices`.
    pub(super) acp_checkpoints: Vec<AcpCheckpoint>,
    /// Shadow-git store backing `acp_checkpoints`. Mutexed because turn tasks
    /// snapshot/diff it off the event loop (a cold snapshot hashes the tree).
    pub(super) acp_checkpoint_store:
        Option<std::sync::Arc<tokio::sync::Mutex<crate::agent::checkpoint::CheckpointStore>>>,
    /// Desired cursor ACP mode: `true` = `plan` (emits `cursor/create_plan`),
    /// `false` = `agent`. Applied via `session/set_mode`, re-applied on re-open.
    /// Cursor keys only.
    pub(super) cursor_plan_mode: bool,
    /// Armed on plan approval: cursor ends its turn after an accepted
    /// `cursor/create_plan`, so turn-end auto-sends the build go-ahead.
    pub(super) cursor_plan_go_pending: bool,
    /// Bumped per cursor prompt turn so the `cursor/update_todos` sink can
    /// tell a turn's first push from later ones.
    pub(super) cursor_turn_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// A resumed session's durable agent transcript (raw OpenAI messages with
    /// tool_calls + results), awaiting the next engine build to be restored
    /// verbatim (exact tool history). Consumed (`take`) on build; `None` otherwise.
    pub(super) pending_agent_messages: Option<Vec<serde_json::Value>>,
    /// `Some(baseline)` when the current session is a foreign import resumed in
    /// memory but not yet persisted — the history length at resume. Persistence
    /// is skipped while `history.len() <= baseline` so merely viewing a
    /// Claude/Codex session creates no aivo copy; the first real turn grows it
    /// past the baseline and it persists as a fork. `None` for every other session.
    pub(super) pristine_import_len: Option<usize>,
    /// Fidelity of the foreign import this session began as, surfaced in
    /// `/session`. Set on resume; cleared on `/new`.
    pub(super) import_fidelity: Option<crate::services::session_import::ImportFidelity>,
    /// Active `/goal` autonomous loop, or `None`. Drives auto-continuation between
    /// agent turns; cleared on completion, the iteration cap, `/goal stop`, an
    /// interrupt, `/new`, resume, or a key/model switch.
    pub(super) goal_mode: Option<GoalState>,
    /// Last turn's engine guard-stop, consumed by the `/goal` continuation to steer past a dead end.
    pub(super) goal_guard_stop: Option<crate::agent::engine::TurnStop>,
    /// Plan mode is on: read-only, persists across turns/interrupts until the plan
    /// is approved or `/plan exit`.
    pub(super) plan_mode: bool,
    /// Captured on plan entry; `/plan exit` returns there.
    pub(super) plan_prior_mode: PlanPriorMode,
    /// Plan exited mid-turn but the live engine's tools are still stripped: restore
    /// them at the next safe async point (turn end / cancel / next dispatch), never
    /// while the model is still running.
    pub(super) plan_exit_pending: bool,
    /// An Esc-unsent agent turn whose engine-side un-send may not have landed yet
    /// (the aborted turn task can still hold the lock): re-apply at next dispatch.
    pub(super) agent_unsend_pending: bool,
    /// A drafted plan (a plan-mode reply that ended without `exit_plan_mode`),
    /// awaiting `/plan go`. Cleared on execute, `/plan exit`, or `/new`.
    pub(super) pending_plan: Option<String>,
    /// History index of the plan reply, framed as the plan card; shifted on
    /// history removal (like `turn_durations`), cleared on execute/discard/`/new`.
    pub(super) plan_card_idx: Option<usize>,
    /// In-process agent for API-key chats (the agent path); `None` until the
    /// first agent turn, rebuilt on key/model switch, cleared on `/new`.
    pub(super) agent_engine: Option<AgentSession>,
    /// `(key_id, cache)` shared across the agent path's per-turn serves so the
    /// learned wire protocol is remembered across turns/launches instead of
    /// re-probed. Seeded from the key's `chat` routes; rebuilt on key switch.
    pub(super) agent_route_cache: Option<(
        String,
        std::sync::Arc<crate::services::route_cache::RouteCache>,
    )>,
    /// Connected MCP servers, shared across engine rebuilds so the servers spawn
    /// once per session. `None` until the first background connect resolves.
    pub(super) mcp_client: Option<std::sync::Arc<crate::agent::mcp::McpClient>>,
    /// A background MCP connect is in flight (don't start a second one).
    pub(super) mcp_connecting: bool,
    /// Per-server interim status while a connect is in flight (server name →
    /// already-mapped `(status, health)`), populated as each server's handshake
    /// resolves so the `/mcp` overlay shows a connected server's real tool count
    /// while slower ones still read "connecting…". Cleared when a new connect
    /// starts; superseded by `mcp_client` once the whole connect lands.
    pub(super) mcp_connect_progress: std::collections::HashMap<String, (String, McpHealth)>,
    /// Qualified `mcp__server__tool` names turned off with Ctrl+T — a UI-side
    /// cache of code-prefs' `disabledMcpTools`, loaded when `/mcp` opens; the
    /// engine re-reads the store on each rebuild, so this is display-only.
    pub(super) disabled_mcp_tools: std::collections::HashSet<String>,
    /// Bumped whenever the configured server set changes (a `/mcp` toggle). A
    /// background connect carries the generation it started under; a result from
    /// an older generation is dropped, so a connect launched before a toggle can't
    /// resurrect a just-disabled server.
    pub(super) mcp_connect_gen: u64,
    /// Tool set changed — rebuild the engine at the next turn. Kept (not
    /// dropped) so the rebuild exports its conversation + rewind checkpoints.
    pub(super) engine_stale: bool,
    /// HTTP MCP servers added this session (name → url) to auto-authorize once
    /// their connect reports a 401 — so adding an OAuth server is one step, not a
    /// separate Ctrl+O. Drained when that connect resolves.
    pub(super) pending_mcp_auth: std::collections::HashMap<String, String>,
    /// Per-turn loopback serve backing the agent engine: (accept-loop handle,
    /// shutdown notify). Torn down when the turn finishes or is cancelled.
    pub(super) agent_serve: Option<(
        JoinHandle<anyhow::Result<()>>,
        std::sync::Arc<tokio::sync::Notify>,
    )>,
    /// Agent decision cards (permission/ask/review/plan/MCP consent); the
    /// engine blocks on each oneshot, so at most one is visible at a time.
    pub(super) cards: AgentCards,
    /// Session decision on spawning a repo's project `.mcp.json` stdio servers.
    pub(super) project_mcp_consent: ProjectMcpConsent,
    /// Session-wide auto-approve (Shift+Tab): when on, the agent runs mutating
    /// tools without a permission card. Off by default (safe).
    pub(super) agent_auto_approve: bool,
    /// The same auto-approve state as a shared atomic, so a running agent turn
    /// consults the LIVE toggle rather than a per-turn snapshot: the native
    /// in-process engine reads it on each tool call, and a long-lived
    /// cursor-agent ACP session reads it on each out-of-process
    /// `request_permission`. Kept in lockstep with `agent_auto_approve`.
    pub(super) auto_approve_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Session-wide edit review (`/config`): when on, an edit batch pauses for a
    /// diff-review card before writing. Off by default (opt-in).
    pub(super) agent_review_edits: bool,
    /// The same state as a shared atomic, read LIVE per batch by the running turn.
    pub(super) review_edits_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Live mid-turn plan-mode exit signal: set when the user leaves plan mode with
    /// a turn in flight; the running engine reads it per tool-call boundary and
    /// restores its tools (`plan_exit_pending` is the turn-end fallback).
    pub(super) plan_exit_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Live mid-turn plan-mode ENTRY signal (Shift+Tab into plan with a turn in
    /// flight); the running engine reads it per tool-call boundary and goes
    /// read-only (the `spawn_agent_turn` mode sync is the turn-end fallback).
    pub(super) plan_enter_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether a planState snapshot is (or may be) in the session file — lets
    /// `persist_plan_state` skip its round-trip for the common no-plan session.
    /// `Cell`: flipped from the `&self` persist path.
    pub(super) plan_state_written: std::cell::Cell<bool>,
    /// Whether the model reasons before answering. On (default): engine requests
    /// reasoning at the effective effort; off: engine sends the family's "off"
    /// floor, which the loopback bridge maps to `thinking:{type:"disabled"}` for
    /// Anthropic upstreams (so off truly stops reasoning, not just hides it).
    /// Toggled in `/config`, remembered across sessions.
    pub(super) thinking_enabled: bool,
    /// aivo's hosted web_search; `/config` toggle, applied to the engine each turn.
    pub(super) web_search_enabled: bool,
    pub(super) agent_tools_enabled: bool,
    /// Footer stat toggles (`/config`); remembered across sessions.
    pub(super) footer_tps_enabled: bool,
    pub(super) footer_cache_enabled: bool,
    pub(super) footer_price_enabled: bool,
    /// Chat TUI color theme (`/config`); remembered across sessions.
    pub(super) theme: UiTheme,
    /// Whether the current model is known to support reasoning/thinking (from the
    /// model-limits snapshot). Cached on each model resolve (see
    /// `refresh_context_window`); gates the footer effort badge so it only shows
    /// where thinking can actually appear. Unknown models → false.
    pub(super) model_supports_thinking: bool,
    /// Snapshot vision support, cached on each model resolve. `Some(false)` =
    /// text-only (image sends refused pre-flight); `None` = unknown (let through).
    pub(super) model_image_input: Option<bool>,
    /// Test-only: force the plain-chat route — production never downgrades,
    /// but tests need a successful dispatch without an agent-engine build.
    #[cfg(test)]
    pub(super) test_force_plain_route: bool,
    /// Image-generation mode (`/config`, `--image-model`); persisted in code-prefs.
    pub(super) image_gen: crate::services::session_store::ImageGenMode,
    /// The picked generator `(key_id, model)`; `Custom` without a pair = unconfigured.
    pub(super) image_gen_custom: Option<(String, String)>,
    /// Vision fallback mode (`/config`, `--vision-model`); persisted in code-prefs.
    pub(super) vision_fallback: crate::services::session_store::VisionFallbackMode,
    /// Custom describer `(key_id, model)` for `VisionFallbackMode::Custom`.
    pub(super) vision_fallback_custom: Option<(String, String)>,
    /// Image-hash → described text, mirrored from the engine so descriptions
    /// survive engine rebuilds (key/model switch, /new).
    pub(super) vision_descriptions: std::collections::HashMap<String, String>,
    /// Parsed Cursor effort tier for the footer badge (`None` for non-cursor or
    /// bare ids); set in `refresh_context_window`.
    pub(super) cursor_effort_label: Option<String>,
    /// `/effort` reasoning level chosen by the user (None = model default);
    /// applied to the engine on build/change and persisted in chat-prefs.
    pub(super) reasoning_effort: Option<String>,
    /// Valid effort levels for the active model (from `caps.reasoning_efforts`),
    /// refreshed on model/key change. Empty = the model exposes none.
    pub(super) model_reasoning_efforts: Vec<String>,
    /// Messages typed while a turn was in flight, in submit order; each is
    /// auto-sent (one per turn) as the preceding turn finishes — a real FIFO so
    /// a second queued message doesn't silently clobber the first.
    pub(super) queued_messages: Vec<String>,
    /// Session-mail messages seen waiting while a turn ran; the "arrives when
    /// this turn ends" notice fires only when this count grows.
    pub(super) mail_waiting_seen: usize,
    /// Claimed mail the outgoing turn should answer; consumed on every dispatch route.
    pub(super) pending_reply_obligation: Option<crate::agent::engine::ReplyObligation>,
    /// Mid-turn steering handoff to the engine task; leftovers reclaim into
    /// `queued_messages` at turn end, cleared with it on interrupt/cancel.
    pub(super) steering_queue: SteeringQueue,
    /// Slash commands typed while a turn was in flight that need the engine idle
    /// (`/compact`, `/rewind`, `/goal`, `/plan`), in submit order; executed as the
    /// turn finishes, before any queued message. Cleared with `queued_messages` on
    /// interrupt/cancel.
    pub(super) queued_commands: Vec<SlashCommand>,
    /// Selected row in the queued-input panel (`queued_rows` order); `None` =
    /// composer focused. Entered by ↑ on an empty composer.
    pub(super) queue_focus: Option<usize>,
    /// An in-flight `!cmd` local shell run streaming output into the transcript,
    /// or `None`. Separate from `sending` (model turns) so the two don't entangle.
    pub(super) local_command: Option<LocalCommandRun>,
    /// App-owned background-job table; never rebuilt (jobs survive `/new`/switches), killed at exit.
    pub(super) jobs: crate::agent::jobs::SharedJobs,
    /// Running-job count; refreshed (and reaped) on a ~250ms throttle; render reads this field.
    pub(super) jobs_running: usize,
    /// Last job-table poll — bounds the reap sweep to ~4Hz (the input-repaint tick is ~1ms).
    pub(super) last_jobs_poll: std::time::Instant,
    /// Throttle for the open-session mailbox poll (`tick_session_mail`).
    pub(super) last_mail_poll: std::time::Instant,
    /// Holds this session's presence record on disk; dropping deregisters
    /// (also on panic-unwind). Replaced when `/resume`//new` switches ids.
    pub(super) mail_presence: Option<crate::services::session_mail::PresenceGuard>,
    /// FULL output of each finished `!cmd`, keyed by the history index of its
    /// `local_command` entry — the source an expanded block renders from (the
    /// transcript and on-disk session keep only a bounded preview). In-memory only,
    /// so after a resume an old block expands to just its persisted preview. Cleared
    /// alongside `expanded_thinking` when history is replaced.
    pub(super) local_outputs: std::collections::HashMap<usize, LocalCommandOutput>,
    /// History indices of `local_command` entries the user expanded inline (full
    /// output shown in place of the folded preview). In-memory only; cleared with
    /// `expanded_thinking`. A toggle bumps `transcript_revision` so the flip repaints.
    pub(super) expanded_output: std::collections::HashSet<usize>,
    /// Start indices of long settled tool-step runs the user expanded back out
    /// of their one-row `▸ N earlier steps` fold. In-memory only; maintained
    /// like `expanded_output` (shifted on removal, cleared on history replace).
    pub(super) expanded_step_folds: std::collections::HashSet<usize>,
    /// History indices of assistant turns the user COLLAPSED (folded to the `▸`
    /// summary). Thoughts show in full by default, so this set holds the exceptions.
    /// In-memory only; cleared when history is replaced (new chat, resume, rewind).
    /// A toggle bumps `transcript_revision`, the body-cache key, so a flip repaints.
    pub(super) expanded_thinking: std::collections::HashSet<usize>,
    /// History indices of user turns dispatched to the agent engine. The `/rewind`
    /// picker matches checkpoints by prompt text; requiring this flag stops a
    /// plain-chat/ACP turn with identical text from stealing an engine turn's
    /// checkpoint. In-memory; cleared on new chat/resume, retained-below on rewind.
    pub(super) agent_turn_indices: std::collections::HashSet<usize>,
    /// Thinking duration (ms) per committed assistant turn, by history index.
    /// Recorded but no longer surfaced (thoughts show content, not timing).
    /// In-memory only, cleared alongside `expanded_thinking`.
    pub(super) reasoning_durations: std::collections::HashMap<usize, u64>,
    /// Wall time (ms) a finished turn took, by the history index of its last entry;
    /// drives the `✻ Done in …` marker. In-memory only, cleared with `expanded_thinking`.
    pub(super) turn_durations: std::collections::HashMap<usize, u64>,
    /// Completion note appended to the `✻ Done in …` marker (this turn's tokens
    /// and estimated cost), keyed and cleared like `turn_durations`.
    pub(super) turn_notes: std::collections::HashMap<usize, String>,
    /// When the current segment's reasoning started streaming (first reasoning
    /// chunk), for the live `▸ thought for Ns` timer. `None` between segments.
    pub(super) reasoning_started_at: Option<Instant>,
    /// The current segment's thinking duration (ms), frozen when the answer began
    /// streaming so the displayed time excludes answer-streaming. `None` until the
    /// answer starts (the live timer runs from `reasoning_started_at` until then).
    pub(super) reasoning_elapsed_ms: Option<u64>,
    /// In-flight skill install; drives the progress row and idle status line.
    pub(super) installing_skill: Option<SkillInstallProgress>,
    /// Held here, not in the (cloned-every-render) overlay, so the temp tree is
    /// deleted exactly once. The bool is the pick's `-p/--project` destination.
    pub(super) staged_skill_install: Option<(crate::agent::skills::StagedInstall, bool)>,
    /// `/share` live-share state; only `live_impl.rs` drives it (the footer
    /// badge reads `handle` presence).
    pub(super) share: LiveShareState,
    /// `/login`–`/logout` flow state; only `account_impl.rs` drives it.
    pub(super) account: AccountFlow,
    /// One-shot: the next draw clears first, healing emulator-corrupted cells
    /// that diff-only painting would never rewrite (macOS Tahoe Terminal.app).
    pub(super) pending_full_repaint: bool,
}

/// The one agent decision card awaiting the user's verdict. Mutually
/// exclusive by construction; at most one can block the agent because every
/// producer awaits its oneshot before the next request (detached subagents
/// deny visibly without a card).
pub(super) enum AgentCard {
    /// Tool-permission card, while the agent waits for the user's y/n/a.
    Permission(PendingPermission),
    /// `ask_user` question card, while the agent waits for the user's pick.
    Ask(PendingAskUser),
    /// Edit-review card, while the agent waits for approve/reject.
    Review(PendingReview),
    /// Plan-approval card (`exit_plan_mode`), while the agent waits for the
    /// verdict.
    PlanApproval(PendingPlanApproval),
}

/// Agent decision cards awaiting the user's verdict.
#[derive(Default)]
pub(super) struct AgentCards {
    /// The active agent card. Setting a new one drops (= denies via closed
    /// oneshot) any predecessor.
    active: Option<AgentCard>,
    /// Pending consent card for project MCP servers (held back until decided).
    /// Independent lifecycle: shown above any agent card and NOT dropped by
    /// [`Self::clear_agent_cards`] at turn teardown.
    pub(super) mcp_consent: Option<McpConsentPrompt>,
}

/// Typed views of `active` so call sites keep their per-card shape:
/// `$get()` / `$get_mut()` borrow when that variant is up, `$take()` resolves
/// it, `$set()` replaces whatever was up.
macro_rules! card_accessors {
    ($get:ident, $take:ident, $set:ident, $variant:ident, $ty:ty) => {
        pub(super) fn $get(&self) -> Option<&$ty> {
            match &self.active {
                Some(AgentCard::$variant(c)) => Some(c),
                _ => None,
            }
        }

        pub(super) fn $take(&mut self) -> Option<$ty> {
            match self.active.take() {
                Some(AgentCard::$variant(c)) => Some(c),
                other => {
                    self.active = other;
                    None
                }
            }
        }

        pub(super) fn $set(&mut self, card: $ty) {
            self.active = Some(AgentCard::$variant(card));
        }
    };
    ($get:ident, $get_mut:ident, $take:ident, $set:ident, $variant:ident, $ty:ty) => {
        card_accessors!($get, $take, $set, $variant, $ty);

        pub(super) fn $get_mut(&mut self) -> Option<&mut $ty> {
            match &mut self.active {
                Some(AgentCard::$variant(c)) => Some(c),
                _ => None,
            }
        }
    };
}

impl AgentCards {
    card_accessors!(
        permission,
        take_permission,
        set_permission,
        Permission,
        PendingPermission
    );
    card_accessors!(ask, ask_mut, take_ask, set_ask, Ask, PendingAskUser);
    card_accessors!(
        review,
        review_mut,
        take_review,
        set_review,
        Review,
        PendingReview
    );
    card_accessors!(
        plan_approval,
        plan_approval_mut,
        take_plan_approval,
        set_plan_approval,
        PlanApproval,
        PendingPlanApproval
    );

    /// True while any agent decision card blocks the turn (`mcp_consent` has
    /// its own lifecycle and doesn't count).
    pub(super) fn any_agent_card(&self) -> bool {
        self.active.is_some()
    }

    /// Turn teardown (finish/error/cancel): drop the active card, denying a
    /// still-pending request via its closed oneshot. Leaves `mcp_consent` up.
    pub(super) fn clear_agent_cards(&mut self) {
        self.active = None;
    }
}

/// Per-frame render products and cross-frame memos, written during render;
/// the hitboxes are read back by mouse handling (backdrop-click dismiss,
/// modal-confined selection).
#[derive(Default)]
pub(super) struct RenderCache {
    /// Cross-frame memo of the built + wrapped transcript body; see
    /// [`TranscriptCache`]. Rebuilt only on content/width change.
    pub(super) transcript: Option<TranscriptCache>,
    /// Cross-frame memo of the volatile tail (streamed reply + running `!cmd` +
    /// notice); see [`VolatileTailCache`]. Keeps a 60fps redraw off the O(reply²)
    /// re-parse/re-wrap path while the answer streams.
    pub(super) volatile_tail: Option<VolatileTailCache>,
    /// The status label on screen + when first shown; throttled by
    /// `tick_status_throttle` so it switches at most once per `STATUS_MIN_DURATION`.
    pub(super) status_display: Option<(String, Instant)>,
    /// Non-picker overlay's full box (borders included) from the last render;
    /// a left press outside it dismisses the overlay like Esc.
    pub(super) overlay_hitbox: Option<Rect>,
    /// Region the screen selection is confined to — a modal's inner content rect
    /// while one is open, so a drag selects inside the modal, not the whole line.
    /// `None` = the full screen.
    pub(super) screen_region: Option<Rect>,
}

/// `/resume` picker preview state (cache + debounce + in-flight load), driven
/// by `event_loop_impl.rs`.
#[derive(Default)]
pub(super) struct SessionPreviewState {
    /// Preview cache: session id → history tail, valid while its
    /// `updated_at` matches the index row's.
    pub(super) cache: std::collections::HashMap<String, PreviewEntry>,
    /// Debounced preview load: spawned once `due` passes with the selection still there.
    pub(super) pending: Option<(String, Instant)>,
    /// In-flight preview load; at most one at a time.
    pub(super) task: Option<(String, JoinHandle<()>)>,
}

/// `/share` live-share state, extracted so the one impl file that drives it
/// (`live_impl.rs`) owns a named cluster instead of loose fields.
#[derive(Default)]
pub(super) struct LiveShareState {
    /// Active share, `None` when not sharing; its presence drives the footer
    /// `● sharing` badge. Stopped on `/share stop`, `/new`, resume, and exit.
    pub(super) handle: Option<crate::services::share_live::LiveShareHandle>,
    /// True between a start and its `LiveShareReady` event; blocks a second start.
    pub(super) starting: bool,
    /// Bumped by `stop_live_share` so an in-flight start's result reads as stale.
    pub(super) generation: u64,
    /// `--share` requested but not yet started — `maybe_start_live_share` defers it
    /// until the session settles so it pins the final session id.
    pub(super) requested: bool,
}

/// `/login`–`/logout` account-flow state, extracted so the one impl file that
/// drives it (`account_impl.rs`) owns a named cluster instead of loose fields.
#[derive(Default)]
pub(super) struct AccountFlow {
    /// Generation — bumped on start/cancel so a superseded flow's late result
    /// is dropped (mirrors `live_share_gen`).
    pub(super) generation: u64,
    /// In-flight login poll / unlink, aborted on cancel so an escaped login
    /// can't write `account.json` after the user walked away.
    pub(super) task: Option<JoinHandle<()>>,
    /// `/login` waiting for approval — the status card.
    pub(super) login: Option<AccountLoginCard>,
    /// `/logout` awaiting its y/n confirm; the account display name.
    pub(super) pending_logout: Option<String>,
}

impl CodeTuiApp {
    /// The one exhaustive field literal: every field gets a neutral, zero-I/O
    /// default. `new()` overwrites the fields it computes; `make_test_app`
    /// injects throwaway stores. Add new fields HERE — tests never change.
    pub(super) fn bare(
        tx: UnboundedSender<RuntimeEvent>,
        rx: UnboundedReceiver<RuntimeEvent>,
        session_store: SessionStore,
        cache: ModelsCache,
        client: Client,
        key: ApiKey,
    ) -> Self {
        Self {
            session_store,
            cache,
            client,
            key,
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
            last_config_setting: None,
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
            subagent_token_base: 0,
            tool_output_tail: std::collections::VecDeque::new(),
            tool_output_partial: String::new(),
            turn_output_tokens: 0,
            turn_stream_chars: 0,
            turn_stream_chars_measured: 0,
            turn_steps: 0,
            retrying: false,
            last_turn_tps: None,
            last_cache_hit_pct: None,
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
            scrollbar_hit: None,
            scrollbar_drag: None,
            preview_close_hits: Vec::new(),
            session_id_hit: None,
            share_badge_hit: None,
            effort_badge_hit: None,
            composer_text_area: None,
            composer_scroll: 0,
            transcript_selection: None,
            transcript_drag_active: false,
            screen_selection: None,
            screen_drag_active: false,
            screen_surface: None,
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
            session_preview: SessionPreviewState::default(),
            inline_images: InlineImageState::default(),
            preview_pane: None,
            inline_images_enabled: true,
            pending_inline_image_cleanup: false,
            detected_graphics_caps: crate::services::terminal_graphics::GraphicsCaps::default(),
            render_cache: RenderCache::default(),
            reduce_motion: false,
            frame_tick: 0,
            picker_hitbox: None,
            overlay_detail_area: None,
            exit_confirm_pending: false,
            goal_stop_confirm_pending: false,
            pending_ctrl_x: false,
            pending_external_edit: false,
            paste_burst_newline: false,
            cursor_acp_session: None,
            cursor_prewarm: None,
            acp_checkpoints: Vec::new(),
            acp_checkpoint_store: None,
            cursor_plan_mode: false,
            cursor_plan_go_pending: false,
            cursor_turn_seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            pending_agent_messages: None,
            pristine_import_len: None,
            import_fidelity: None,
            goal_mode: None,
            goal_guard_stop: None,
            plan_mode: false,
            plan_prior_mode: PlanPriorMode::default(),
            plan_exit_pending: false,
            agent_unsend_pending: false,
            pending_plan: None,
            plan_card_idx: None,
            agent_engine: None,
            agent_route_cache: None,
            mcp_client: None,
            mcp_connecting: false,
            mcp_connect_progress: std::collections::HashMap::new(),
            disabled_mcp_tools: std::collections::HashSet::new(),
            mcp_connect_gen: 0,
            engine_stale: false,
            pending_mcp_auth: std::collections::HashMap::new(),
            agent_serve: None,
            cards: AgentCards::default(),
            agent_auto_approve: false,
            auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            agent_review_edits: false,
            review_edits_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            plan_exit_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            plan_enter_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            plan_state_written: std::cell::Cell::new(false),
            thinking_enabled: true,
            web_search_enabled: true,
            agent_tools_enabled: true,
            footer_tps_enabled: true,
            footer_cache_enabled: true,
            footer_price_enabled: true,
            theme: UiTheme::Dark,
            model_supports_thinking: false,
            model_image_input: None,
            #[cfg(test)]
            test_force_plain_route: false,
            vision_fallback: crate::services::session_store::VisionFallbackMode::default(),
            vision_fallback_custom: None,
            image_gen: crate::services::session_store::ImageGenMode::default(),
            image_gen_custom: None,
            vision_descriptions: std::collections::HashMap::new(),
            cursor_effort_label: None,
            reasoning_effort: None,
            model_reasoning_efforts: Vec::new(),
            queued_messages: Vec::new(),
            mail_waiting_seen: 0,
            pending_reply_obligation: None,
            steering_queue: SteeringQueue::default(),
            queued_commands: Vec::new(),
            queue_focus: None,
            project_mcp_consent: ProjectMcpConsent::default(),
            local_command: None,
            jobs: crate::agent::jobs::JobTable::new(None),
            last_jobs_poll: std::time::Instant::now(),
            last_mail_poll: std::time::Instant::now(),
            mail_presence: None,
            jobs_running: 0,
            local_outputs: std::collections::HashMap::new(),
            expanded_output: std::collections::HashSet::new(),
            expanded_step_folds: std::collections::HashSet::new(),
            expanded_thinking: std::collections::HashSet::new(),
            agent_turn_indices: std::collections::HashSet::new(),
            reasoning_durations: std::collections::HashMap::new(),
            turn_durations: std::collections::HashMap::new(),
            turn_notes: std::collections::HashMap::new(),
            reasoning_started_at: None,
            reasoning_elapsed_ms: None,
            installing_skill: None,
            staged_skill_install: None,
            share: LiveShareState::default(),
            account: AccountFlow::default(),
            pending_full_repaint: false,
        }
    }
}
