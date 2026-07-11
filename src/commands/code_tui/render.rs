use super::*;
use std::borrow::Cow;

#[derive(Clone)]
pub(super) struct StyledLine {
    pub(super) line: Line<'static>,
    pub(super) plain: String,
}

pub(super) struct RenderedTranscript {
    /// The logical (unwrapped) lines. Wrapped to the render width via
    /// [`wrap_transcript`] at draw time so our row model matches what ratatui
    /// actually paints (it word-wraps; a char-wrap count under-scrolls).
    pub(super) lines: Vec<StyledLine>,
    pub(super) plain_lines: Vec<String>,
    /// Per logical line: the accent-bar color for the block it belongs to, or
    /// `None` for chrome (intro, inter-block spacing). Aligned with `lines`.
    pub(super) bar_colors: Vec<Option<Color>>,
}

impl RenderedTranscript {
    pub(super) fn new(lines: Vec<StyledLine>, bar_colors: Vec<Option<Color>>) -> Self {
        let plain_lines = lines.iter().map(|l| l.plain.clone()).collect();
        Self {
            lines,
            plain_lines,
            bar_colors,
        }
    }
}

/// Word-wrap one styled line to `width`, preserving per-character styles and
/// returning one [`StyledLine`] per visual row. Mirrors a terminal word-wrap:
/// break on whitespace, hard-break words longer than the width. Rendering these
/// rows with ratatui's wrap OFF makes our row count exactly match the display.
/// A run of like characters (all whitespace or all non-whitespace) used by the
/// word-wrapper.
struct Token {
    is_space: bool,
    chars: Vec<(char, Style)>,
    width: usize,
}

// `cur_w` is a loop accumulator; its final write (a row break on the last token)
// is intentionally not read again.
#[allow(unused_assignments)]
pub(super) fn wrap_styled_line(spans: &[Span<'static>], width: usize) -> Vec<StyledLine> {
    let chars: Vec<(char, Style)> = spans
        .iter()
        .flat_map(|s| {
            let style = s.style;
            s.content.chars().map(move |c| (c, style))
        })
        .collect();
    if width == 0 || chars.is_empty() {
        return vec![styled_line_from_chars(&chars)];
    }
    let cw = |c: char| UnicodeWidthChar::width(c).unwrap_or(0).max(1);

    // Leading whitespace becomes a hanging indent: it's stripped off, the body is
    // wrapped to the remaining width, and the indent is re-applied to every row so
    // continuation rows align under the first line's text instead of falling back
    // to the gutter (e.g. a wrapped `▾ thought` reasoning line stays indented).
    // Disabled when the indent leaves no room for text. Lines that fit unwrapped
    // are unaffected — the body still fits in one row, so the result is identical.
    let indent_len = chars
        .iter()
        .take_while(|(c, _)| *c == ' ' || *c == '\t')
        .count();
    let indent: &[(char, Style)] = &chars[..indent_len];
    let indent_w: usize = indent.iter().map(|(c, _)| cw(*c)).sum();
    let (indent, body, avail) = if indent_w > 0 && indent_w < width {
        (indent.to_vec(), &chars[indent_len..], width - indent_w)
    } else {
        (Vec::new(), &chars[..], width)
    };

    // Tokenize the body into alternating whitespace / word runs.
    let mut tokens: Vec<Token> = Vec::new();
    for &(c, st) in body {
        let is_space = c == ' ' || c == '\t';
        let w = cw(c);
        match tokens.last_mut() {
            Some(tok) if tok.is_space == is_space => {
                tok.chars.push((c, st));
                tok.width += w;
            }
            _ => tokens.push(Token {
                is_space,
                chars: vec![(c, st)],
                width: w,
            }),
        }
    }

    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0usize;
    for Token {
        is_space,
        chars: buf,
        width: tw,
    } in tokens
    {
        if cur_w + tw <= avail {
            cur.extend(buf);
            cur_w += tw;
        } else if is_space {
            // Whitespace that won't fit ends the row; drop it (no leading space).
            rows.push(std::mem::take(&mut cur));
            cur_w = 0;
        } else if tw <= avail {
            // Word fits on a fresh row.
            if !cur.is_empty() {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            cur = buf;
            cur_w = tw;
        } else {
            // Word longer than the width: hard-break across rows.
            for (c, st) in buf {
                let w = cw(c);
                if cur_w + w > avail && !cur.is_empty() {
                    rows.push(std::mem::take(&mut cur));
                    cur_w = 0;
                }
                cur.push((c, st));
                cur_w += w;
            }
        }
    }
    if !cur.is_empty() || rows.is_empty() {
        rows.push(cur);
    }
    rows.iter()
        .map(|r| {
            if indent.is_empty() {
                styled_line_from_chars(r)
            } else {
                let mut row = indent.clone();
                row.extend_from_slice(r);
                styled_line_from_chars(&row)
            }
        })
        .collect()
}

/// Build a [`StyledLine`] from per-character styles, coalescing runs of the same
/// style into spans.
fn styled_line_from_chars(chars: &[(char, Style)]) -> StyledLine {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut cur = String::new();
    let mut cur_style: Option<Style> = None;
    for &(c, style) in chars {
        plain.push(c);
        if cur_style == Some(style) {
            cur.push(c);
        } else {
            if let Some(s) = cur_style {
                spans.push(Span::styled(std::mem::take(&mut cur), s));
            }
            cur.push(c);
            cur_style = Some(style);
        }
    }
    if let Some(s) = cur_style {
        spans.push(Span::styled(cur, s));
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

/// A transcript wrapped to the render width: the `Text` (rendered with wrap OFF),
/// the per-row plain strings (for selection), and per-row bar colors (the gutter).
pub(super) struct WrappedTranscript {
    pub(super) text: Text<'static>,
    pub(super) rows: Vec<String>,
    pub(super) bars: Vec<Option<Color>>,
}

/// Word-wrap all logical lines to `width`, carrying each line's bar color onto
/// every visual row.
pub(super) fn wrap_transcript(
    lines: &[StyledLine],
    bars: &[Option<Color>],
    width: u16,
) -> WrappedTranscript {
    let width = usize::from(width.max(1));
    let mut text_lines: Vec<Line<'static>> = Vec::new();
    let mut rows: Vec<String> = Vec::new();
    let mut row_bars: Vec<Option<Color>> = Vec::new();
    for (idx, sl) in lines.iter().enumerate() {
        let bar = bars.get(idx).copied().flatten();
        for vrow in wrap_styled_line(&sl.line.spans, width) {
            let vrow = fill_trailing_background(vrow, width);
            text_lines.push(vrow.line);
            rows.push(vrow.plain);
            row_bars.push(bar);
        }
    }
    if text_lines.is_empty() {
        text_lines.push(Line::default());
        rows.push(String::new());
        row_bars.push(None);
    }
    WrappedTranscript {
        text: Text::from(text_lines),
        rows,
        bars: row_bars,
    }
}

/// If a wrapped visual row ends in a background-colored span, extend that
/// background across the rest of the row width. This turns the tinted inline-diff
/// lines (see `render_edit_diff`) into contiguous full-width blocks, so a long
/// changed line that wraps still reads as one shaded region instead of a ragged
/// tail at the left margin. Ordinary rows (no trailing background) are untouched,
/// so nothing else in the transcript gains a background.
fn fill_trailing_background(mut row: StyledLine, width: usize) -> StyledLine {
    let Some(bg) = row.line.spans.last().and_then(|span| span.style.bg) else {
        return row;
    };
    let used = usize::from(row_display_width(&row.plain));
    if used >= width {
        return row;
    }
    let pad = " ".repeat(width - used);
    row.plain.push_str(&pad);
    row.line
        .spans
        .push(Span::styled(pad, Style::default().bg(bg)));
    row
}

/// Accent-bar color for a transcript role: user = lavender, assistant = brand
/// accent (aivo's voice), `!cmd` shell runs = magenta, agent tool steps = cyan,
/// everything else = muted.
pub(super) fn role_bar_color(role: &str) -> Color {
    match role {
        "user" => USER,
        "assistant" => ACCENT,
        "local_command" => SHELL,
        "tool_call" | "tool_result" | "plan" => TOOL,
        "error" => ERROR,
        _ => MUTED,
    }
}

/// A turn-level error persisted into the transcript: `✗ message` in the error hue.
pub(super) fn render_error_message(lines: &mut Vec<StyledLine>, content: &str) {
    for (i, line) in content.lines().enumerate() {
        let prefix = if i == 0 { "✗ " } else { "  " };
        lines.push(line_with_plain(vec![Span::styled(
            format!("{prefix}{line}"),
            Style::default().fg(ERROR),
        )]));
    }
}

fn wrap_one_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    let mut wrapped = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in line.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if current_width > 0 && current_width + ch_width > width {
            wrapped.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }
    wrapped.push(current);
    wrapped
}

/// Greedy word wrap by display width, for prose that must stay readable on
/// narrow terminals — words never split mid-word (an oversized single word
/// falls back to the character wrap).
pub(super) fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let sep = usize::from(!current.is_empty());
        let current_width: usize = current.chars().map(char_cell_width).sum();
        let word_width: usize = word.chars().map(char_cell_width).sum();
        if !current.is_empty() && current_width + sep + word_width > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
        .into_iter()
        .flat_map(|l| wrap_one_line(&l, width))
        .collect()
}

fn char_cell_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0).max(1)
}

pub(super) fn wrap_plain_lines(lines: &[String], width: u16) -> Vec<String> {
    let width = usize::from(width.max(1));
    let mut wrapped = Vec::new();
    for line in lines {
        wrapped.extend(wrap_one_line(line, width));
    }
    if wrapped.is_empty() {
        wrapped.push(String::new());
    }
    wrapped
}

pub(super) fn normalized_selection(
    selection: TranscriptSelection,
) -> (TranscriptPoint, TranscriptPoint) {
    if (selection.anchor.row, selection.anchor.column)
        <= (selection.focus.row, selection.focus.column)
    {
        (selection.anchor, selection.focus)
    } else {
        (selection.focus, selection.anchor)
    }
}

pub(super) fn selected_text_from_rows(
    rows: &[String],
    selection: TranscriptSelection,
) -> Option<String> {
    let (start, end) = normalized_selection(selection);
    if (start.row, start.column) == (end.row, end.column) || start.row >= rows.len() {
        return None;
    }

    let end_row = end.row.min(rows.len().saturating_sub(1));
    let mut selected = Vec::new();
    for (row_index, row) in rows.iter().enumerate().take(end_row + 1).skip(start.row) {
        let text = if start.row == end_row {
            slice_display_columns(row, start.column, end.column)
        } else if row_index == start.row {
            slice_display_columns(row, start.column, u16::MAX)
        } else if row_index == end_row {
            slice_display_columns(row, 0, end.column)
        } else {
            row.clone()
        };
        // Drop wrap-padding spaces so pasted text has no ragged trailing runs.
        selected.push(text.trim_end().to_string());
    }

    Some(selected.join("\n"))
}

/// Display-column width of a visual row (CJK-aware).
pub(super) fn row_display_width(row: &str) -> u16 {
    let mut width = 0usize;
    for ch in row.chars() {
        width += UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
    }
    width.min(u16::MAX as usize) as u16
}

/// Word boundaries (display columns `[start, end)`) around `column` in `row`.
/// Returns `None` when the click lands on whitespace or past the row's end —
/// the caller falls back to a plain caret in that case.
pub(super) fn word_bounds_at(row: &str, column: u16) -> Option<(u16, u16)> {
    let column = usize::from(column);
    // Build (start_col, end_col, is_word) spans per character so we can locate
    // the clicked column and expand over the contiguous word run it sits in.
    let mut spans: Vec<(usize, usize, bool)> = Vec::new();
    let mut col = 0usize;
    for ch in row.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        spans.push((col, col + w, !ch.is_whitespace()));
        col += w;
    }
    let hit = spans
        .iter()
        .position(|&(s, e, is_word)| is_word && column >= s && column < e)?;
    let mut start = hit;
    while start > 0 && spans[start - 1].2 {
        start -= 1;
    }
    let mut end = hit;
    while end + 1 < spans.len() && spans[end + 1].2 {
        end += 1;
    }
    let start_col = spans[start].0.min(u16::MAX as usize) as u16;
    let end_col = spans[end].1.min(u16::MAX as usize) as u16;
    Some((start_col, end_col))
}

pub(super) fn slice_display_columns(text: &str, start: u16, end: u16) -> String {
    let start = usize::from(start);
    let end = usize::from(end);
    let mut result = String::new();
    let mut col = 0usize;
    for ch in text.chars() {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        let next = col + width;
        if next > start && col < end {
            result.push(ch);
        }
        col = next;
        if col >= end {
            break;
        }
    }
    result
}

pub(super) fn push_message_spacing(lines: &mut Vec<StyledLine>) {
    if !lines.is_empty() {
        lines.push(blank_line());
    }
}

/// Append a rendered block under a single accent-bar color (`None` = no bar),
/// keeping `bars` in lockstep with `lines`. Trailing blank lines are trimmed so
/// the only gaps between blocks are the explicit (barless) spacing lines — that
/// keeps the accent bar tight to its content.
pub(super) fn push_block(
    lines: &mut Vec<StyledLine>,
    bars: &mut Vec<Option<Color>>,
    mut block: Vec<StyledLine>,
    bar: Option<Color>,
) {
    while block.last().is_some_and(|l| l.plain.trim().is_empty()) {
        block.pop();
    }
    bars.resize(lines.len() + block.len(), bar);
    lines.extend(block);
}

/// The two-row half-block "aivo code" wordmark, 30 columns wide.
pub(super) const BRAND_WORDMARK: [&str; 2] = [
    "▄▀█ █ █░█ █▀█  █▀▀ █▀█ █▀▄ █▀█",
    "█▀█ █ ▀▄▀ █▄█  █▄▄ █▄█ █▄▀ █▄▄",
];
/// Narrow "aivo" fallback for columns too slim for the full mark.
pub(super) const BRAND_WORDMARK_NARROW: [&str; 2] = ["▄▀█ █ █░█ █▀█", "█▀█ █ ▀▄▀ █▄█"];
pub(super) const BRAND_WORDMARK_WIDTH: u16 = 30;
pub(super) const BRAND_WORDMARK_NARROW_WIDTH: u16 = 13;

fn brand_wordmark_for(width: u16) -> (&'static [&'static str; 2], u16) {
    if width >= BRAND_WORDMARK_WIDTH {
        (&BRAND_WORDMARK, BRAND_WORDMARK_WIDTH)
    } else {
        (&BRAND_WORDMARK_NARROW, BRAND_WORDMARK_NARROW_WIDTH)
    }
}

/// Brand wordmark as accent styled lines, version trailing the baseline when it
/// fits. Shared by the empty state and transcript intro so both start at the
/// same column (see `test_intro_column_stable_*`); `width` is the inset column.
pub(super) fn brand_wordmark_lines(width: u16) -> Vec<StyledLine> {
    let mark_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
    let (mark, mark_width) = brand_wordmark_for(width);
    // Shares the baseline row, so the banner stays two lines tall either way.
    let version = format!("  v{}", crate::version::VERSION);
    let bottom = if width >= mark_width + version.chars().count() as u16 {
        line_with_plain(vec![
            Span::styled(mark[1].to_string(), mark_style),
            Span::styled(version, Style::default().fg(FAINT)),
        ])
    } else {
        line_plain(mark[1].to_string(), mark_style)
    };
    vec![line_plain(mark[0].to_string(), mark_style), bottom]
}

pub(super) fn push_transcript_intro(lines: &mut Vec<StyledLine>, width: u16) {
    // Match the empty-state top gap so the banner doesn't jump when the first
    // message lands.
    for _ in 0..EMPTY_STATE_TOP_GAP {
        lines.push(blank_line());
    }
    lines.extend(brand_wordmark_lines(width));
}

pub(super) fn should_add_message_spacing(previous_role: Option<&str>, next_role: &str) -> bool {
    let Some(prev) = previous_role else {
        return false;
    };
    if next_role.is_empty() {
        return false;
    }
    // Keep a tool result tight against its call, and keep consecutive tool
    // steps grouped — no blank line within an agent tool sequence (the blank
    // before/after the whole sequence is still kept).
    if next_role == "tool_result" {
        return false;
    }
    let prev_is_tool = prev == "tool_call" || prev == "tool_result";
    if prev_is_tool && next_role == "tool_call" {
        return false;
    }
    true
}

pub(super) fn attachment_kind_label(attachment: &MessageAttachment) -> &'static str {
    if attachment.mime_type.starts_with("image/") {
        "image"
    } else {
        "file"
    }
}

pub(super) fn render_user_attachment_lines(
    lines: &mut Vec<StyledLine>,
    attachments: &[MessageAttachment],
) {
    for attachment in attachments {
        push_styled_line(
            lines,
            format!(
                "  [{}] {}",
                attachment_kind_label(attachment),
                attachment.name
            ),
            Style::default().fg(MUTED),
        );
    }
}

pub(super) fn composer_attachment_lines(attachments: &[MessageAttachment]) -> Vec<Line<'static>> {
    attachments
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            Line::from(vec![
                Span::styled("· ", Style::default().fg(ACCENT)),
                Span::styled(
                    format!(
                        "{}. [{}] {}",
                        index + 1,
                        attachment_kind_label(attachment),
                        attachment.name
                    ),
                    Style::default().fg(MUTED),
                ),
            ])
        })
        .collect()
}

/// `/resume` preview transcript: user/assistant render like the main transcript,
/// a tool run collapses to one `⚙ n tool steps` line, reasoning/plan skipped.
pub(super) fn session_preview_lines(
    messages: &[ChatMessage],
    width: u16,
    truncated: bool,
) -> (Vec<StyledLine>, Vec<Option<Color>>) {
    fn spacing(
        lines: &mut Vec<StyledLine>,
        bars: &mut Vec<Option<Color>>,
        prev: Option<&str>,
        next: &str,
    ) {
        if should_add_message_spacing(prev, next) {
            push_message_spacing(lines);
            bars.resize(lines.len(), None);
        }
    }

    let mut lines: Vec<StyledLine> = Vec::new();
    let mut bars: Vec<Option<Color>> = Vec::new();
    if truncated {
        lines.push(line_plain(
            "· earlier messages not shown ·".to_string(),
            Style::default().fg(FAINT),
        ));
        bars.push(None);
    }
    let mut prev_role: Option<&str> = None;
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        match message.role.as_str() {
            "tool_call" | "tool_result" => {
                let mut steps = 0usize;
                while index < messages.len()
                    && matches!(messages[index].role.as_str(), "tool_call" | "tool_result")
                {
                    steps += 1;
                    index += 1;
                }
                spacing(&mut lines, &mut bars, prev_role, "tool_call");
                push_block(
                    &mut lines,
                    &mut bars,
                    vec![line_plain(
                        format!("⚙ {steps} tool step{}", if steps == 1 { "" } else { "s" }),
                        Style::default().fg(MUTED),
                    )],
                    Some(TOOL),
                );
                prev_role = Some("tool_result");
            }
            "user" if !message.content.trim().is_empty() || !message.attachments.is_empty() => {
                spacing(&mut lines, &mut bars, prev_role, "user");
                let mut block = Vec::new();
                render_user_message(&mut block, &message.content, &message.attachments);
                push_block(&mut lines, &mut bars, block, Some(USER));
                prev_role = Some("user");
                index += 1;
            }
            "assistant" if !message.content.trim().is_empty() => {
                spacing(&mut lines, &mut bars, prev_role, "assistant");
                let mut block = Vec::new();
                render_assistant_message(&mut block, None, &message.content, width);
                push_block(&mut lines, &mut bars, block, Some(ACCENT));
                prev_role = Some("assistant");
                index += 1;
            }
            _ => {
                index += 1;
            }
        }
    }
    (lines, bars)
}

pub(super) fn render_user_message(
    lines: &mut Vec<StyledLine>,
    content: &str,
    attachments: &[MessageAttachment],
) {
    // No `> ` marker: the message renders as plain text and the role is carried
    // entirely by the lavender gutter bar, so a user turn reads as clean prose.
    let mut had_line = false;
    for raw_line in content.lines() {
        push_styled_line(lines, raw_line.to_string(), Style::default().fg(TEXT));
        had_line = true;
    }
    if !had_line {
        push_styled_line(lines, String::new(), Style::default().fg(TEXT));
    }
    render_user_attachment_lines(lines, attachments);
}

pub(super) fn extend_without_leading_blank(
    lines: &mut Vec<StyledLine>,
    mut rendered: Vec<StyledLine>,
) {
    while rendered
        .first()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        rendered.remove(0);
    }
    lines.extend(rendered);
}

/// First-row thinking markers: `✻` when it all fits, `▸` windowed (click to
/// expand), `▾` expanded (click to collapse).
pub(super) const THINKING_MARKER: &str = "✻";
pub(super) const THINKING_COLLAPSED_MARKER: &str = "▸";
pub(super) const THINKING_EXPANDED_MARKER: &str = "▾";
/// A thought shows at most this many rows; older rows scroll off (expand for the rest).
pub(super) const THINKING_WINDOW_LINES: usize = 4;

fn thinking_marker(has_more: bool, expanded: bool) -> &'static str {
    match (has_more, expanded) {
        (false, _) => THINKING_MARKER,
        (true, true) => THINKING_EXPANDED_MARKER,
        (true, false) => THINKING_COLLAPSED_MARKER,
    }
}

/// A clickable thinking header — the marker row; the block's later rows are
/// indented so they don't match.
pub(super) fn is_thinking_header(row: &str) -> bool {
    let row = row.trim_start();
    row.starts_with(THINKING_MARKER)
        || row.starts_with(THINKING_COLLAPSED_MARKER)
        || row.starts_with(THINKING_EXPANDED_MARKER)
}

/// Skips bare-punctuation reasoning (some models emit a lone "..." segment).
pub(super) fn reasoning_is_substantive(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

/// A dim reasoning row: `marker` on the block's first row, two-space indent after.
fn push_reasoning_line(lines: &mut Vec<StyledLine>, text: &str, marker: Option<&str>) {
    lines.push(line_with_plain(vec![
        Span::styled(
            match marker {
                Some(m) => format!("{m} "),
                None => "  ".to_string(),
            },
            Style::default().fg(MUTED),
        ),
        Span::styled(
            text.to_string(),
            Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
        ),
    ]));
}

/// Pre-wrap to the content width less the 2-col marker, so wrapped rows hang
/// indented under the marker instead of the transcript wrapper breaking them flush.
fn wrapped_reasoning_rows(reasoning: &str, width: u16) -> Vec<String> {
    let wrap_w = usize::from(width).saturating_sub(2).max(1);
    normalized_reasoning_lines(reasoning)
        .iter()
        .flat_map(|line| super::overlay_render_impl::wrap_chars(line, wrap_w))
        .collect()
}

fn render_reasoning_rows(lines: &mut Vec<StyledLine>, rows: &[String], marker: &str) {
    if rows.is_empty() {
        push_reasoning_line(lines, "", Some(marker));
        return;
    }
    for (i, row) in rows.iter().enumerate() {
        push_reasoning_line(lines, row, (i == 0).then_some(marker));
    }
}

/// Every row of the thought (the expanded state).
pub(super) fn render_reasoning_full(lines: &mut Vec<StyledLine>, reasoning: &str, width: u16) {
    let rows = wrapped_reasoning_rows(reasoning, width);
    let marker = thinking_marker(rows.len() > THINKING_WINDOW_LINES, true);
    render_reasoning_rows(lines, &rows, marker);
}

/// The most recent [`THINKING_WINDOW_LINES`] rows (the default/live view).
pub(super) fn render_reasoning_window(lines: &mut Vec<StyledLine>, reasoning: &str, width: u16) {
    let rows = wrapped_reasoning_rows(reasoning, width);
    let marker = thinking_marker(rows.len() > THINKING_WINDOW_LINES, false);
    let start = rows.len().saturating_sub(THINKING_WINDOW_LINES);
    render_reasoning_rows(lines, &rows[start..], marker);
}

pub(super) fn normalized_reasoning_lines(reasoning: &str) -> Vec<String> {
    let mut lines = Vec::new();

    for raw_line in reasoning.lines() {
        let trimmed = raw_line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }

    lines
}

pub(super) fn render_assistant_message(
    lines: &mut Vec<StyledLine>,
    reasoning: Option<&str>,
    content: &str,
    width: u16,
) {
    if let Some(reasoning) = reasoning.filter(|text| reasoning_is_substantive(text)) {
        render_reasoning_full(lines, reasoning, width);
        if !content.is_empty() {
            push_styled_line(lines, "", Style::default());
        }
    }

    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content, width));
    }
}

/// Reasoning alongside an assistant turn: the text and whether the user expanded
/// it (`false` → the rolling window; `true` → the full thought).
pub(super) struct ReasoningView<'a> {
    pub(super) text: &'a str,
    pub(super) expanded: bool,
}

/// Push an assistant turn as up to two blocks: the thinking block is barless so it
/// recedes as meta; the answer carries `content_bar`. Separate blocks keep the
/// answer's bar from bleeding up into the reasoning.
pub(super) fn push_assistant_blocks(
    lines: &mut Vec<StyledLine>,
    bars: &mut Vec<Option<Color>>,
    reasoning: Option<ReasoningView<'_>>,
    content: &str,
    width: u16,
    content_bar: Color,
) {
    if let Some(view) = reasoning.filter(|v| reasoning_is_substantive(v.text)) {
        let mut block = Vec::new();
        if view.expanded {
            render_reasoning_full(&mut block, view.text, width);
        } else {
            render_reasoning_window(&mut block, view.text, width);
        }
        push_block(lines, bars, block, None);
        if !content.is_empty() {
            lines.push(blank_line());
            bars.push(None);
        }
    }

    if !content.is_empty() {
        let mut block = Vec::new();
        render_assistant_message(&mut block, None, content, width);
        push_block(lines, bars, block, Some(content_bar));
    }
}

pub(super) fn render_pending_status(
    lines: &mut Vec<StyledLine>,
    frame_tick: usize,
    reduce_motion: bool,
    elapsed: Duration,
    deadline: Option<Duration>,
    activity: &str,
    tail: &str,
) {
    let spinner = spinner_frame_indexed(frame_tick, reduce_motion);
    let mut elapsed = format_request_elapsed(elapsed);
    // A step with a timeout budget shows the deadline it will be killed at.
    if let Some(deadline) = deadline {
        elapsed = format!("{elapsed} / {}", format_request_elapsed(deadline));
    }
    // Empty tail → just the clock.
    let text = if tail.is_empty() {
        format!("{spinner} {activity} ({elapsed})")
    } else {
        format!("{spinner} {activity} ({elapsed} • {tail})")
    };
    push_styled_line(
        lines,
        text,
        Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
    );
}

/// Row-name cap — short enough that the delegate's action stays visible.
const SUBAGENT_ROW_NAME_MAX_COLS: usize = 28;

/// One live tail row, truncated so a long line can't re-wrap taller each frame.
pub(super) fn tool_tail_row_text(line: &str) -> String {
    const TOOL_TAIL_MAX_COLS: usize = 96;
    let line = line.replace('\t', "  ");
    format!(
        "    {}",
        truncate_label(line.trim_end(), TOOL_TAIL_MAX_COLS)
    )
}

/// One live row for a parallel delegate: `↳ name — action · step N (12s)`,
/// flipping to `✓ name — done (32s · 8 step(s) · 1.2k tokens)`.
pub(super) fn subagent_row_text(row: &super::shared::SubagentRow) -> String {
    let name = truncate_label(&row.name, SUBAGENT_ROW_NAME_MAX_COLS);
    if let Some((ok, steps, tokens, took)) = &row.done {
        let mark = if *ok { "✓" } else { "✗" };
        let what = if *ok { "done" } else { "no answer" };
        let mut stats = format!("{} · {steps} step(s)", format_request_elapsed(*took));
        if *tokens > 0 {
            stats.push_str(&format!(" · {} tokens", format_token_count_value(*tokens)));
        }
        return format!("  {mark} {name} — {what} ({stats})");
    }
    let action = truncate_label(&row.action, ACTION_TARGET_MAX_COLS);
    let mut line = format!("  ↳ {name} — {action}");
    if row.step > 0 {
        line.push_str(&format!(" · step {}", row.step));
    }
    line.push_str(&format!(
        " ({})",
        format_request_elapsed(row.started.elapsed())
    ));
    if let Some(tool) = &row.denied {
        line.push_str(&format!(" · {tool} denied"));
    }
    line
}

/// One row for a call in a trailing parallel batch: `↳ Audit the auth flow`,
/// flipping to `✓ Audit the auth flow — 34 lines` / `✗ … — <error>`. No
/// per-row clock — history entries carry no start time.
pub(super) fn parallel_call_row_text(
    name: &str,
    args: &serde_json::Value,
    outcome: (Option<String>, bool),
    cwd: &str,
) -> String {
    // A delegate reads as its task — the headline already says "sub-agents".
    let label = if canonical_tool_name(name) == "subagent" {
        let task = tool_arg_summary(name, args, cwd);
        if task.is_empty() {
            "sub-agent".to_string()
        } else {
            task
        }
    } else {
        tool_action_label(name, args, cwd)
    };
    let detail = |text: &str| {
        truncate_label(
            text.lines().next().unwrap_or_default().trim(),
            ACTION_TARGET_MAX_COLS,
        )
    };
    match outcome {
        (result, true) => format!(
            "  ✗ {label} — {}",
            detail(result.as_deref().unwrap_or("failed"))
        ),
        (Some(result), false) => format!("  ✓ {label} — {}", detail(&result)),
        (None, false) => format!("  ↳ {label}"),
    }
}

/// Footer context-stat color by fullness: a quiet signal that warms toward the
/// model's window limit (compaction territory) before it's hit.
pub(super) fn context_fill_color(pct: u64) -> Color {
    if pct >= 95 {
        ERROR
    } else if pct >= 80 {
        WARNING
    } else {
        MUTED
    }
}

pub(super) fn spinner_frame_indexed(frame_tick: usize, reduce_motion: bool) -> &'static str {
    if reduce_motion {
        return spinner_frame(0);
    }
    spinner_frame(frame_tick / 5)
}

pub(super) fn notice_display(notice: Option<&(Color, String)>) -> Option<(Color, Cow<'_, str>)> {
    notice.map(|(color, text)| {
        let formatted = if *color == ERROR {
            Cow::Owned(format!("Error: {text}"))
        } else {
            Cow::Borrowed(text.as_str())
        };
        (*color, formatted)
    })
}

/// Styled spans for the active notice. The share notice splits into a red
/// `● Sharing:` indicator + a link-colored URL (an all-red line reads as an error);
/// everything else is one color.
pub(super) fn notice_spans(notice: Option<&(Color, String)>) -> Option<Vec<Span<'static>>> {
    let (color, text) = notice_display(notice)?;
    if let Some(url) = text.strip_prefix(LIVE_NOTICE_PREFIX) {
        return Some(vec![
            Span::styled(LIVE_NOTICE_PREFIX, Style::default().fg(LIVE)),
            Span::styled(url.to_string(), Style::default().fg(LINK)),
        ]);
    }
    Some(vec![Span::styled(
        text.into_owned(),
        Style::default().fg(color),
    )])
}

pub(super) fn rect_contains(area: Rect, point: (u16, u16)) -> bool {
    let (x, y) = point;
    x >= area.x
        && x < area.x.saturating_add(area.width)
        && y >= area.y
        && y < area.y.saturating_add(area.height)
}

pub(super) fn render_system_message(
    lines: &mut Vec<StyledLine>,
    role: &str,
    content: &str,
    width: u16,
) {
    push_styled_line(
        lines,
        role.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content, width));
    }
}

/// Render an agent tool invocation as a Tree-style `→ verb(args)` line. The verb
/// is yellow for mutating tools (write/edit/bash), dim otherwise. The agent
/// bridge stores these as `tool_call` entries whose `content` is JSON
/// `{"name", "args"}` (chat history is a display log; the engine owns the real
/// LLM context).
#[allow(clippy::too_many_arguments)]
pub(super) fn render_tool_call(
    lines: &mut Vec<StyledLine>,
    name: &str,
    args: &serde_json::Value,
    result: Option<&str>,
    failed: bool,
    cwd: &str,
    line_starts: &[Option<usize>],
    old_content: Option<&str>,
) {
    let name = canonical_tool_name(name);
    let summary = tool_arg_summary(name, args, cwd);
    let verb_style = if crate::agent::tools::is_mutating(name) {
        Style::default().fg(WARNING)
    } else {
        Style::default().fg(MUTED)
    };
    let mut spans = vec![Span::styled("→ ".to_string(), Style::default().fg(TOOL))];
    if name == "subagent" {
        // A delegated task reads as the task itself — "subagent" is jargon and the
        // arrow already marks it as a step. Show the task in place of the verb.
        let label = if summary.is_empty() {
            "delegated a task".to_string()
        } else {
            summary
        };
        spans.push(Span::styled(label, verb_style));
    } else {
        spans.push(Span::styled(tool_display_name(name), verb_style));
        // cursor reports no target for its tools (only a generic title), so the
        // summary is often empty — skip the `()` rather than render `read_file()`.
        if !summary.is_empty() {
            spans.push(Span::styled(
                format!("({summary})"),
                Style::default().fg(MUTED),
            ));
        }
    }
    lines.push(line_with_plain(spans));
    // For edit/write tools, show a compact diff of what changed so the user can
    // review the agent's edit without opening the file (no-op for tools without
    // a textual old/new, e.g. cursor edits).
    render_edit_diff(lines, name, args, line_starts, old_content);
    // Cursor stores its result on the call entry (the in-process agent emits a
    // separate `tool_result` line instead) — surface it as a compact `⎿` line,
    // in the error hue when the tool failed.
    if failed {
        lines.push(line_with_plain(vec![
            Span::styled("  ⎿ ".to_string(), Style::default().fg(ERROR)),
            Span::styled(
                result.unwrap_or("failed").to_string(),
                Style::default().fg(ERROR),
            ),
        ]));
    } else if let Some(result) = result.filter(|r| !r.is_empty()) {
        lines.push(line_with_plain(vec![
            Span::styled("  ⎿ ".to_string(), Style::default().fg(FAINT)),
            Span::styled(result.to_string(), Style::default().fg(FAINT)),
        ]));
    }
}

/// Decode the optional result + failed flag a cursor `tool_call` entry carries
/// after enrichment (the in-process agent leaves both unset — it reports results
/// as separate `tool_result` entries).
pub(super) fn decode_tool_outcome(content: &str) -> (Option<String>, bool) {
    let Some(v) = serde_json::from_str::<serde_json::Value>(content).ok() else {
        return (None, false);
    };
    let result = v.get("result").and_then(|x| x.as_str()).map(str::to_string);
    let failed = v.get("failed").and_then(|x| x.as_bool()).unwrap_or(false);
    (result, failed)
}

/// The per-pair edit start lines a `tool_call` entry carries; empty for non-edit
/// tools and older entries, which then render without a number gutter.
pub(super) fn decode_line_starts(content: &str) -> Vec<Option<usize>> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.get("line_starts").and_then(|x| x.as_array()).cloned())
        .map(|arr| arr.iter().map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default()
}

/// The pre-write snapshot a `write_file` `tool_call` entry carries; absent
/// (new/non-UTF8/oversized/older entries) → all-additions fallback.
pub(super) fn decode_old_content(content: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()?
        .get("old_content")?
        .as_str()
        .map(str::to_string)
}

/// Decode a `tool_call` history entry's JSON `{name, args}` payload.
pub(super) fn decode_tool_call(content: &str) -> (String, serde_json::Value) {
    let decoded = serde_json::from_str::<serde_json::Value>(content).ok();
    let name = decoded
        .as_ref()
        .and_then(|v| v.get("name"))
        .and_then(|x| x.as_str())
        .unwrap_or("tool")
        .to_string();
    let args = decoded
        .as_ref()
        .and_then(|v| v.get("args"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    (name, args)
}

/// Coalescing key: tools sharing a display verb (grep+glob, edit_file+multi_edit)
/// fold into one group, so a mixed batch reads as one `searched ×N` line.
pub(super) fn tool_group_key(name: &str) -> &str {
    match name {
        "grep" | "glob" => "search",
        "edit_file" | "multi_edit" => "edit",
        other => other,
    }
}

/// Render a coalesced run of same-verb tool calls as one `→ verb N: a, b…` line.
/// cursor agents explore in many small steps, so a card per call is noise — the
/// run collapses to its count and the distinguishing targets (file basenames,
/// patterns, commands).
pub(super) fn render_tool_call_group(
    lines: &mut Vec<StyledLine>,
    name: &str,
    count: usize,
    targets: &[String],
    failed: usize,
) {
    let n = count;
    let head = match name {
        "read_file" => format!("read {n} files"),
        "edit_file" | "multi_edit" => format!("edited {n} files"),
        "delete_file" => format!("deleted {n} files"),
        "grep" | "glob" => format!("searched ×{n}"),
        "run_bash" => format!("ran {n} commands"),
        "web_fetch" => format!("fetched {n} URLs"),
        _ => format!("{} ×{n}", tool_display_name(name)),
    };
    let verb_style = if crate::agent::tools::is_mutating(name) {
        Style::default().fg(WARNING)
    } else {
        Style::default().fg(MUTED)
    };
    let mut spans = vec![
        Span::styled("→ ".to_string(), Style::default().fg(TOOL)),
        Span::styled(head, verb_style),
    ];
    let list = join_targets(targets, 56);
    if !list.is_empty() {
        spans.push(Span::styled(
            format!(": {list}"),
            Style::default().fg(MUTED),
        ));
    }
    if failed > 0 {
        spans.push(Span::styled(
            format!(" · {failed} failed"),
            Style::default().fg(ERROR),
        ));
    }
    lines.push(line_with_plain(spans));
}

/// Friendly verb for a tool call: an `mcp__server__tool` name renders as
/// `server/tool` (the raw double-underscore form is ugly); everything else is
/// shown as-is.
fn tool_display_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("mcp__")
        && let Some((server, tool)) = rest.split_once("__")
    {
        return format!("{server}/{tool}");
    }
    name.to_string()
}

/// The raw target a tool acted on (basename / pattern / command). Verbatim for
/// model-facing seed notes; the transcript uses [`tool_call_target_display`].
pub(super) fn tool_call_target(name: &str, args: &serde_json::Value) -> String {
    let pick = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match canonical_tool_name(name) {
        "read_file" | "edit_file" | "multi_edit" | "delete_file" | "write_file" | "list_dir" => {
            basename(pick("path"))
        }
        "grep" | "glob" => pick("pattern").to_string(),
        "run_bash" => pick("command").to_string(),
        "web_fetch" => pick("url").to_string(),
        _ => String::new(),
    }
}

/// Transcript form of [`tool_call_target`]: a `run_bash` command is condensed for
/// display; everything else is identical.
pub(super) fn tool_call_target_display(name: &str, args: &serde_json::Value, cwd: &str) -> String {
    if canonical_tool_name(name) == "run_bash" {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        condense_command(command, cwd)
    } else {
        tool_call_target(name, args)
    }
}

/// Present-tense label for the inline status line — `running grep "foo"`,
/// `reading main.rs`, `delegating to reviewer`. Target capped so it can't wrap.
pub(super) fn tool_action_label(name: &str, args: &serde_json::Value, cwd: &str) -> String {
    let name = canonical_tool_name(name);
    if name == "subagent" {
        // Fallback keys are Claude Code's names for the same fields.
        let pick = |keys: [&str; 2]| {
            keys.into_iter().find_map(|k| {
                args.get(k)
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
            })
        };
        return match (
            pick(["label", "description"]),
            pick(["agent", "subagent_type"]),
        ) {
            (Some(label), _) => format!(
                "delegating: {}",
                truncate_label(label, ACTION_TARGET_MAX_COLS)
            ),
            (None, Some(agent)) => format!("delegating to {agent}"),
            (None, None) => "delegating".to_string(),
        };
    }
    if name == "update_plan" {
        return "updating the plan".to_string();
    }
    if name == "skill" {
        let s = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
        return if s.is_empty() {
            "running a skill".to_string()
        } else {
            format!("running skill {s}")
        };
    }
    if name == "take_note" {
        return "taking a note".to_string();
    }
    if name == "switch_model" {
        let m = args.get("model").and_then(|v| v.as_str()).unwrap_or("");
        return if m.is_empty() {
            "switching model".to_string()
        } else {
            format!("switching model to {m}")
        };
    }
    if name == "set_effort" {
        let l = args.get("level").and_then(|v| v.as_str()).unwrap_or("");
        return if l.is_empty() {
            "setting effort".to_string()
        } else {
            format!("setting effort to {l}")
        };
    }
    let verb = match name {
        "read_file" => "reading",
        "edit_file" | "multi_edit" => "editing",
        "write_file" => "writing",
        "delete_file" => "deleting",
        "list_dir" => "listing",
        "grep" | "glob" => "searching",
        "run_bash" => "running",
        "web_fetch" => "fetching",
        // MCP and any other external tool: name it, no target to show.
        _ => return format!("running {}", tool_display_name(name)),
    };
    let target = truncate_label(
        &tool_call_target_display(name, args, cwd),
        ACTION_TARGET_MAX_COLS,
    );
    if target.is_empty() {
        format!("{verb} {}", tool_display_name(name))
    } else {
        format!("{verb} {target}")
    }
}

/// Condense a shell command for a one-line label: drop a redundant `cd <cwd> &&`
/// prefix and `2>/dev/null` / `2>&1` redirection noise, then collapse whitespace.
fn condense_command(cmd: &str, cwd: &str) -> String {
    let mut out = strip_leading_cd(cmd.trim(), cwd).to_string();
    for noise in [
        " 2>/dev/null",
        " 2> /dev/null",
        " 1>/dev/null",
        " >/dev/null",
        " > /dev/null",
        " 2>&1",
    ] {
        out = out.replace(noise, "");
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strip a leading `cd <cwd> && ` (cwd optionally quoted) — redundant, run_bash
/// already runs there. A `cd` into a different dir is kept.
fn strip_leading_cd<'a>(cmd: &'a str, cwd: &str) -> &'a str {
    let cwd = cwd.trim_end_matches(['/', '\\']);
    if cwd.is_empty() {
        return cmd;
    }
    for (open, close) in [("", ""), ("\"", "\""), ("'", "'")] {
        let prefix = format!("cd {open}{cwd}{close} &&");
        if let Some(rest) = cmd.strip_prefix(prefix.as_str()) {
            return rest.trim_start();
        }
    }
    cmd
}

/// Final path segment (handles both `/` and `\` separators).
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .find(|s| !s.is_empty())
        .unwrap_or(path)
        .to_string()
}

/// Max display columns for a status-line action target (room for verb + tail).
const ACTION_TARGET_MAX_COLS: usize = 40;

/// Cap `s` to `max` display columns, appending `…` on overflow.
fn truncate_label(s: &str, max: usize) -> String {
    if cell_width(s) <= max {
        return s.to_string();
    }
    let budget = max.saturating_sub(1); // room for the ellipsis
    let mut out = String::new();
    for ch in s.chars() {
        if cell_width(&out) + cell_width(&ch.to_string()) > budget {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

/// Join target labels into one line, dropping empties, capped at `max` display
/// columns with a `(+K more)` tail when the rest don't fit.
fn join_targets(targets: &[String], max: usize) -> String {
    let items: Vec<&str> = targets
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let mut out = String::new();
    let mut shown = 0usize;
    for (i, item) in items.iter().enumerate() {
        let piece = if i == 0 {
            (*item).to_string()
        } else {
            format!(", {item}")
        };
        if shown > 0 && cell_width(&out) + cell_width(&piece) > max {
            break;
        }
        out.push_str(&piece);
        shown += 1;
    }
    let remaining = items.len() - shown;
    if remaining > 0 {
        out.push_str(&format!(" (+{remaining} more)"));
    }
    out
}

/// Most output lines a `!cmd` run shows in the transcript; the rest collapse to a
/// "+N more lines" marker. The reader (`run_shell_streaming`) captures somewhat
/// more than this so that count is accurate for moderate output.
pub(super) const MAX_OUTPUT_LINES: usize = 40;

/// Ceiling on what an inline-expanded block renders (and what `local_outputs` keeps
/// for it). Bounds the O(lines) whole-body re-wrap, which runs on every history
/// change — an unbounded 50k-line expand would re-freeze the UI each turn.
pub(super) const MAX_EXPANDED_OUTPUT_LINES: usize = 2_000;

/// Trailing lines a folded `run_bash` result keeps visible — its live streaming
/// window (`push_tool_output`'s tail), so landing doesn't drop them and jump the view.
pub(super) const STREAM_TAIL_LINES: usize = 4;

/// How many output lines a finished `!cmd` persists into its `local_command` history
/// entry (and the on-disk session) — a bounded preview that caps session size and is
/// what an expanded block falls back to after a resume. The true count rides along as
/// `total_lines`; the full in-session output lives in `local_outputs`.
pub(super) const MAX_PERSISTED_OUTPUT_LINES: usize = 200;

/// The first `n` lines of `s`, rejoined with `\n` (no trailing newline). Used to
/// bound what a `!cmd` run persists without disturbing its display line count.
pub(super) fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join("\n")
}

/// Leading marker of a folded `!cmd` output (`▸ +N more lines`).
pub(super) const OUTPUT_COLLAPSED_PREFIX: &str = "▸ +";
/// Leading marker of an expanded `!cmd` output (`▾ collapse`).
pub(super) const OUTPUT_EXPANDED_PREFIX: &str = "▾ collapse";

/// Whether a rendered transcript row is a clickable output expander: folded
/// `▸ +N…`, expanded `⎿ ▾ N lines` summary, or trailing `▾ collapse`.
pub(super) fn is_output_expander(row: &str) -> bool {
    let row = row.trim_start();
    let row = row.strip_prefix("⎿ ").unwrap_or(row);
    row.starts_with(OUTPUT_COLLAPSED_PREFIX)
        || row.starts_with(OUTPUT_EXPANDED_PREFIX)
        || row
            .strip_prefix("▾ ")
            .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
}

/// The true output line count a `local_command` entry carries (its persisted
/// `total_lines`, or the counted preview as a fallback). Only runs whose total
/// exceeds `MAX_OUTPUT_LINES` render an expander, so this keys the click handler's
/// ordinal → history-index mapping.
pub(super) fn local_command_total_lines(content: &str) -> usize {
    let decoded =
        serde_json::from_str::<serde_json::Value>(content).unwrap_or(serde_json::Value::Null);
    decoded
        .get("total_lines")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or_else(|| {
            let pick = |k: &str| {
                decoded
                    .get(k)
                    .and_then(|v| v.as_str())
                    .map(|s| s.lines().count())
                    .unwrap_or(0)
            };
            pick("stdout") + pick("stderr")
        })
}

/// How an existing `!cmd` block should render its output.
pub(super) enum OutputView<'a> {
    /// A still-running command — preview only, with a plain (non-clickable)
    /// `… (+N more lines)` marker, since it isn't committed yet.
    Live,
    /// A committed command folded to its `MAX_OUTPUT_LINES` preview + a clickable
    /// `▸ +N more lines` expander.
    Collapsed,
    /// A committed command expanded in place: the retained in-memory output when
    /// it's still held (`full`, up to `MAX_EXPANDED_OUTPUT_LINES`), else the
    /// persisted ≤200-line preview, + a `▾ collapse` toggle.
    Expanded {
        full: Option<&'a LocalCommandOutput>,
    },
}

/// Render a `!cmd` local shell run: a `! command` header over its output (stdout
/// faint, stderr in the warning hue). A folded block shows only the first
/// `MAX_OUTPUT_LINES` over a clickable `▸ +N more lines` expander; clicking it
/// expands the block to its full output in place. Each shown line is rendered in
/// full — the transcript word-wraps long lines onto extra rows, so nothing is
/// clipped at the pane edge. Stored as a `local_command` entry whose `content` is
/// JSON `{"command", "stdout", "stderr", "exit_code"}` (plus optional
/// `running`/`truncated`/`interrupted` flags).
pub(super) fn render_local_command(
    lines: &mut Vec<StyledLine>,
    content: &str,
    view: OutputView<'_>,
) {
    let decoded =
        serde_json::from_str::<serde_json::Value>(content).unwrap_or(serde_json::Value::Null);
    let pick = |k: &str| decoded.get(k).and_then(|v| v.as_str()).unwrap_or("");
    let flag = |k: &str| decoded.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let command = pick("command");
    let exit_code = decoded
        .get("exit_code")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    // While running there is no exit code yet; `truncated`/`interrupted` annotate
    // a finished run cut short by the capture caps or by esc.
    let running = flag("running");
    let truncated = flag("truncated");
    let interrupted = flag("interrupted");

    lines.push(line_with_plain(vec![
        Span::styled(
            "! ".to_string(),
            Style::default().fg(SHELL).add_modifier(Modifier::BOLD),
        ),
        Span::styled(command.to_string(), Style::default().fg(TEXT)),
    ]));

    // A committed run stores only a bounded preview but carries the true line count
    // in `total_lines`, so "+N more" reflects everything the command produced. A
    // live run has no `total_lines` yet — count its (full) streamed preview.
    let total = decoded
        .get("total_lines")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or_else(|| local_command_total_lines(content));
    // Expanded blocks render the FULL in-memory output when it's still held; folded
    // and live blocks (and a resumed block whose memory is gone) render the persisted
    // preview, capped at `MAX_OUTPUT_LINES` while folded.
    let (stdout, stderr) = match &view {
        OutputView::Expanded { full: Some(o) } => (o.stdout.as_str(), o.stderr.as_str()),
        _ => (pick("stdout"), pick("stderr")),
    };
    let cap = match view {
        OutputView::Expanded { .. } => MAX_EXPANDED_OUTPUT_LINES,
        _ => MAX_OUTPUT_LINES,
    };
    let mut shown = 0usize;
    for (text, color) in stdout
        .lines()
        .map(|l| (l, FAINT))
        .chain(stderr.lines().map(|l| (l, WARNING)))
    {
        if shown >= cap {
            break;
        }
        lines.push(line_with_plain(vec![Span::styled(
            format!("  {text}"),
            Style::default().fg(color),
        )]));
        shown += 1;
    }
    let mut faint = |text: String| {
        lines.push(line_with_plain(vec![Span::styled(
            text,
            Style::default().fg(FAINT),
        )]));
    };
    match view {
        OutputView::Live => {
            if total > shown {
                let suffix = if truncated { ", truncated" } else { "" };
                faint(format!("  … (+{} more lines{suffix})", total - shown));
            } else if truncated {
                faint("  … (output truncated)".to_string());
            }
        }
        OutputView::Collapsed => {
            if total > shown {
                // Clickable expander (the `▸ thought` fold affordance). A truncated
                // capture still notes the cut, since expand shows only what we held.
                let suffix = if truncated { ", truncated" } else { "" };
                lines.push(line_with_plain(vec![Span::styled(
                    format!(
                        "  {OUTPUT_COLLAPSED_PREFIX}{} more lines{suffix}",
                        total - shown
                    ),
                    Style::default().fg(MUTED),
                )]));
            } else if truncated {
                faint("  … (output truncated)".to_string());
            }
        }
        OutputView::Expanded { .. } => {
            // Why anything is still hidden, then the collapse toggle. The inline cap
            // dominates; capture-cap / resume loss only show once we rendered all we held.
            if total > shown && shown >= MAX_EXPANDED_OUTPUT_LINES {
                faint(format!(
                    "  … (+{} more lines — too long to show inline; re-run with `> file` for all)",
                    total - shown
                ));
            } else if truncated {
                faint("  … (output truncated at the capture cap)".to_string());
            } else if total > shown {
                faint(format!(
                    "  … (+{} lines not retained after resume)",
                    total - shown
                ));
            }
            lines.push(line_with_plain(vec![Span::styled(
                format!("  {OUTPUT_EXPANDED_PREFIX}"),
                Style::default().fg(MUTED),
            )]));
        }
    }
    if running {
        lines.push(line_with_plain(vec![Span::styled(
            "  (running…)".to_string(),
            Style::default().fg(FAINT),
        )]));
    } else if total == 0 && exit_code == 0 && !interrupted {
        lines.push(line_with_plain(vec![Span::styled(
            "  (no output)".to_string(),
            Style::default().fg(FAINT),
        )]));
    }
    if interrupted {
        lines.push(line_with_plain(vec![Span::styled(
            "  [interrupted]".to_string(),
            Style::default().fg(ERROR),
        )]));
    } else if !running && !truncated && exit_code != 0 {
        // A truncated run was killed by us (cap/timeout), so its non-zero status is
        // ours, not the command's — the "truncated" note already explains the stop,
        // so don't add a misleading `[exited -1]`.
        lines.push(line_with_plain(vec![Span::styled(
            format!("  [exited {exit_code}]"),
            Style::default().fg(ERROR),
        )]));
    }
}

/// The `[exit N]` tail a nonzero-exit `run_bash` result carries — scan the last
/// few lines, since a sandbox note or spill pointer can follow it.
pub(super) fn bash_exit_code(result: &str) -> Option<i32> {
    result.lines().rev().take(4).find_map(|l| {
        l.trim()
            .strip_prefix("[exit ")
            .and_then(|r| r.strip_suffix(']'))
            .and_then(|n| n.parse().ok())
    })
}

/// Render an agent tool result as a compact `⎿ summary` line under its call.
/// A multi-line summary doubles as the fold toggle (`▸ +N lines`); a nonzero
/// `run_bash` exit reads in the error hue so a broken build can't pass for green.
pub(super) fn render_tool_result(
    lines: &mut Vec<StyledLine>,
    result: &str,
    cwd: &str,
    tool: Option<&str>,
    label: Option<&str>,
    expanded: bool,
) {
    let tool = tool.map(canonical_tool_name);
    let count = result.lines().count();
    let exit = (tool == Some("run_bash"))
        .then(|| bash_exit_code(result))
        .flatten()
        .filter(|&c| c != 0);
    // Multi-line "error:" text stays neutral — only a single-line `error: …`
    // (see `ChatAgentUi`) or a nonzero exit is a real failure.
    let failed = exit.is_some() || (count <= 1 && result.trim_start().starts_with("error:"));
    let summary_color = if failed { ERROR } else { FAINT };
    let mut spans = vec![Span::styled(
        "  ⎿ ".to_string(),
        Style::default().fg(summary_color),
    )];
    if count > 1 {
        // The summary line is the fold toggle in both states.
        let unit = count_unit(tool, count);
        let summary = if expanded {
            format!("▾ {count} {unit}")
        } else {
            format!("{OUTPUT_COLLAPSED_PREFIX}{count} {unit}")
        };
        spans.push(Span::styled(
            summary,
            Style::default().fg(if failed { ERROR } else { MUTED }),
        ));
        if let Some(code) = exit {
            spans.push(Span::styled(
                format!(" · exited {code}"),
                Style::default().fg(ERROR),
            ));
        }
        if let Some(label) = label.filter(|l| !l.is_empty()) {
            spans.push(Span::styled(
                format!(" · {}", truncate_chars(label, 40)),
                Style::default().fg(MUTED),
            ));
        }
        // A search's or subagent's first line is the payoff — preview it.
        if matches!(tool, Some("grep" | "glob" | "subagent")) {
            let first = result
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("");
            spans.push(Span::styled(
                format!(" · {}", truncate_chars(&strip_ansi_and_controls(first), 48)),
                Style::default().fg(summary_color),
            ));
        }
        lines.push(line_with_plain(spans));
        if expanded {
            for line in result.lines().take(MAX_EXPANDED_OUTPUT_LINES) {
                let is_exit_line = line.trim().starts_with("[exit ");
                lines.push(line_with_plain(vec![Span::styled(
                    format!("    {}", strip_ansi_and_controls(line)),
                    Style::default().fg(if is_exit_line { ERROR } else { FAINT }),
                )]));
            }
            if count > MAX_EXPANDED_OUTPUT_LINES {
                lines.push(line_with_plain(vec![Span::styled(
                    format!("    … (+{} more lines)", count - MAX_EXPANDED_OUTPUT_LINES),
                    Style::default().fg(FAINT),
                )]));
            }
            // A long block also folds from its far end.
            if count > MAX_OUTPUT_LINES {
                lines.push(line_with_plain(vec![Span::styled(
                    format!("  {OUTPUT_EXPANDED_PREFIX}"),
                    Style::default().fg(MUTED),
                )]));
            }
        } else if tool == Some("run_bash") {
            // Keep the streamed tail under the summary so landing doesn't jump the view.
            for line in result.lines().skip(count.saturating_sub(STREAM_TAIL_LINES)) {
                let is_exit_line = line.trim().starts_with("[exit ");
                lines.push(line_with_plain(vec![Span::styled(
                    format!("    {}", strip_ansi_and_controls(line)),
                    Style::default().fg(if is_exit_line { ERROR } else { FAINT }),
                )]));
            }
        }
        return;
    }
    if let Some(label) = label.filter(|l| !l.is_empty()) {
        spans.push(Span::styled(
            format!("{} · ", truncate_chars(label, 40)),
            Style::default().fg(MUTED),
        ));
    }
    spans.push(Span::styled(
        tool_result_summary(result, cwd),
        Style::default().fg(summary_color),
    ));
    lines.push(line_with_plain(spans));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffTag {
    Equal,
    Del,
    Ins,
}

/// A word-diff segment: `(changed, text)`, `changed` picking the emphasis tint.
type Seg = (bool, String);

/// A grouped-diff row. `num` is the gutter line number (old-file for `Del`,
/// new-file for `Context`/`Ins`), `None` when the offset is unknown. Changed
/// lines carry per-token segments so the word diff brightens only what moved.
enum DiffRow {
    Context { num: Option<usize>, text: String },
    Del { num: Option<usize>, segs: Vec<Seg> },
    Ins { num: Option<usize>, segs: Vec<Seg> },
    Gap,
}

/// Line-level LCS diff of `old` vs `new`, so only changed lines are flagged and
/// the rest stays context. Falls back to remove-all/add-all past a size cap so a
/// giant rewrite can't trigger the O(n·m) table.
fn diff_lines<'a>(old: &'a str, new: &'a str) -> Vec<(DiffTag, &'a str)> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let (n, m) = (a.len(), b.len());
    if n > 600 || m > 600 {
        let mut ops = Vec::with_capacity(n + m);
        ops.extend(a.iter().map(|l| (DiffTag::Del, *l)));
        ops.extend(b.iter().map(|l| (DiffTag::Ins, *l)));
        return ops;
    }
    lcs_diff(&a, &b)
}

/// Split a line into word-diff tokens: identifier runs and whitespace runs each
/// coalesce; every other char stands alone, so punctuation highlights on its own.
fn tokenize(s: &str) -> Vec<&str> {
    #[derive(PartialEq)]
    enum Class {
        Word,
        Space,
        Other,
    }
    let class = |c: char| {
        if c.is_alphanumeric() || c == '_' {
            Class::Word
        } else if c.is_whitespace() {
            Class::Space
        } else {
            Class::Other
        }
    };
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let cl = class(c);
        // `Other` chars never merge — each is its own token.
        let split = match chars.peek() {
            Some(&(_, next)) => cl == Class::Other || class(next) != cl,
            None => true,
        };
        if split {
            let end = i + c.len_utf8();
            tokens.push(&s[start..end]);
            start = end;
        }
    }
    tokens
}

/// Intra-line word diff of a paired removed/added line into per-side `(changed,
/// text)` runs — common runs keep the base tint, changed runs the emphasis tint.
fn word_segments(old: &str, new: &str) -> (Vec<Seg>, Vec<Seg>) {
    let a = tokenize(old);
    let b = tokenize(new);
    // Past a token cap the O(n·m) table isn't worth it — tint the whole line.
    if a.len() > 400 || b.len() > 400 {
        return (
            vec![(false, old.to_string())],
            vec![(false, new.to_string())],
        );
    }
    let mut del: Vec<Seg> = Vec::new();
    let mut ins: Vec<Seg> = Vec::new();
    let push = |v: &mut Vec<Seg>, changed: bool, tok: &str| match v.last_mut() {
        Some(last) if last.0 == changed => last.1.push_str(tok),
        _ => v.push((changed, tok.to_string())),
    };
    for (tag, tok) in lcs_diff(&a, &b) {
        match tag {
            DiffTag::Equal => {
                push(&mut del, false, tok);
                push(&mut ins, false, tok);
            }
            DiffTag::Del => push(&mut del, true, tok),
            DiffTag::Ins => push(&mut ins, true, tok),
        }
    }
    (del, ins)
}

/// Suffix-LCS diff of two token slices (lines or words), removals before
/// additions within each change run (git order). Shared by [`diff_lines`] and
/// [`word_segments`].
fn lcs_diff<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(DiffTag, &'a str)> {
    let (n, m) = (a.len(), b.len());
    // Suffix LCS-length table: lcs[i][j] = LCS length of a[i..], b[j..].
    let mut lcs = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    let mut raw: Vec<(DiffTag, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            raw.push((DiffTag::Equal, a[i]));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            raw.push((DiffTag::Del, a[i]));
            i += 1;
        } else {
            raw.push((DiffTag::Ins, b[j]));
            j += 1;
        }
    }
    raw.extend(a[i..].iter().map(|l| (DiffTag::Del, *l)));
    raw.extend(b[j..].iter().map(|l| (DiffTag::Ins, *l)));

    // Within each maximal run of changed lines, show removals before additions
    // regardless of how the backtrack interleaved them.
    let mut ops = Vec::with_capacity(raw.len());
    let mut k = 0;
    while k < raw.len() {
        if raw[k].0 == DiffTag::Equal {
            ops.push(raw[k]);
            k += 1;
            continue;
        }
        let start = k;
        while k < raw.len() && raw[k].0 != DiffTag::Equal {
            k += 1;
        }
        ops.extend(raw[start..k].iter().filter(|o| o.0 == DiffTag::Del));
        ops.extend(raw[start..k].iter().filter(|o| o.0 == DiffTag::Ins));
    }
    ops
}

/// One before/after text an edit tool touches. Shared by the diff renderer and
/// the line-number probe so both walk the same pairs.
pub(super) struct EditDiff {
    pub(super) path: String,
    pub(super) old: String,
    pub(super) new: String,
}

/// The pairs an edit tool applies, in render order (one per `edit_file`, per
/// `multi_edit` step, per `apply_patch` hunk, whole content for `write_file`) —
/// so `line_starts` aligns by index. `write_file` defaults to all-additions;
/// the review card ([`review_edit_diffs`]) and the transcript card (via the
/// `old_content` snapshot in [`render_edit_diff`]) upgrade it to a real diff.
pub(super) fn edit_diffs(name: &str, args: &serde_json::Value) -> Vec<EditDiff> {
    let pick = |v: &serde_json::Value, k: &str| {
        v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
    };
    match name {
        "write_file" => {
            let new = pick(args, "content");
            if new.is_empty() {
                vec![]
            } else {
                vec![EditDiff {
                    path: pick(args, "path"),
                    old: String::new(),
                    new,
                }]
            }
        }
        "edit_file" => {
            let (old, new) = (pick(args, "old_string"), pick(args, "new_string"));
            if old.is_empty() && new.is_empty() {
                vec![]
            } else {
                vec![EditDiff {
                    path: pick(args, "path"),
                    old,
                    new,
                }]
            }
        }
        "multi_edit" => {
            let path = pick(args, "path");
            args.get("edits")
                .and_then(|v| v.as_array())
                .map(|edits| {
                    edits
                        .iter()
                        .map(|e| EditDiff {
                            path: path.clone(),
                            old: pick(e, "old_string"),
                            new: pick(e, "new_string"),
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        "apply_patch" => args
            .get("input")
            .and_then(|v| v.as_str())
            .map(|input| {
                crate::agent::apply_patch::diff_blocks(input)
                    .into_iter()
                    .map(|b| EditDiff {
                        path: b.path,
                        old: b.old,
                        new: b.new,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// Like [`edit_diffs`] but also covers `write_file` by reading the current file
/// (read-only). Non-UTF8/oversized files fall back to a byte-count summary.
pub(super) fn review_edit_diffs(
    name: &str,
    args: &serde_json::Value,
    cwd: &std::path::Path,
) -> Vec<EditDiff> {
    if name != "write_file" {
        return edit_diffs(name, args);
    }
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let new = args
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    const MAX_BYTES: u64 = 512 * 1024;
    let abs = cwd.join(path);
    let (old, new) = match std::fs::metadata(&abs) {
        Ok(meta) if meta.len() > MAX_BYTES => (
            format!("<existing file: {} bytes>", meta.len()),
            format!("<new content: {} bytes>", new.len()),
        ),
        Ok(_) => match std::fs::read(&abs).map(String::from_utf8) {
            Ok(Ok(s)) => (s, new),
            Ok(Err(e)) => (
                format!("<binary file: {} bytes>", e.into_bytes().len()),
                format!("<new content: {} bytes>", new.len()),
            ),
            Err(_) => (String::new(), new), // unreadable → treat as new
        },
        Err(_) => (String::new(), new), // missing file → all additions
    };
    if old.is_empty() && new.is_empty() {
        return vec![];
    }
    vec![EditDiff {
        path: path.to_string(),
        old,
        new,
    }]
}

/// The scrollable body of the edit-review card: a filename header + numbered diff
/// per pending edit. Precomputed once when the card opens so the frame render is pure.
pub(super) fn review_body_lines(
    items: &[crate::agent::review::ReviewItem],
    cwd: &std::path::Path,
) -> Vec<Line<'static>> {
    const CONTEXT: usize = 3;
    let mut out: Vec<Line<'static>> = Vec::new();
    for item in items {
        let diffs = review_edit_diffs(&item.tool, &item.args, cwd);
        for d in &diffs {
            out.push(Line::from(Span::styled(
                format!("  {}", d.path),
                Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
            )));
            let rows = build_hunk(&d.old, &d.new, None, CONTEXT);
            if rows.is_empty() {
                out.push(Line::from(Span::styled(
                    "    (no textual change)".to_string(),
                    Style::default().fg(FAINT),
                )));
                continue;
            }
            let numw = diff_num_width(&rows);
            for row in &rows {
                out.push(render_diff_row(row, numw).line);
            }
        }
    }
    out
}

/// Expand tabs to 4 spaces so a raw `\t` (unicode-width 1) can't desync the
/// terminal's cell grid and leave stale ghost cells.
pub(super) fn expand_tabs(s: &str) -> Cow<'_, str> {
    if s.contains('\t') {
        Cow::Owned(s.replace('\t', "    "))
    } else {
        Cow::Borrowed(s)
    }
}

/// Diff `old`→`new` into display rows: trim to changed lines ±`context` (git's
/// `-U`), collapse longer gaps to `⋯`, number kept rows from `start`, and refine
/// paired del/ins lines into word segments. Empty when nothing changed.
fn build_hunk(old: &str, new: &str, start: Option<usize>, context: usize) -> Vec<DiffRow> {
    let (old, new) = (expand_tabs(old), expand_tabs(new));
    let ops = diff_lines(&old, &new);
    if !ops.iter().any(|(t, _)| *t != DiffTag::Equal) {
        return Vec::new();
    }
    let keep: Vec<bool> = (0..ops.len())
        .map(|i| {
            let lo = i.saturating_sub(context);
            let hi = (i + context).min(ops.len() - 1);
            (lo..=hi).any(|j| ops[j].0 != DiffTag::Equal)
        })
        .collect();

    // One row per op in op order, so `keep` indexes `full` directly. Numbers
    // advance on the row's own side; segments come from pairing the k-th del with
    // the k-th ins of a run.
    let mut full: Vec<DiffRow> = Vec::with_capacity(ops.len());
    let mut oldn = start;
    let mut newn = start;
    let bump = |n: &mut Option<usize>| {
        if let Some(v) = n {
            *v += 1;
        }
    };
    let mut i = 0;
    while i < ops.len() {
        if ops[i].0 == DiffTag::Equal {
            full.push(DiffRow::Context {
                num: newn,
                text: ops[i].1.to_string(),
            });
            bump(&mut oldn);
            bump(&mut newn);
            i += 1;
            continue;
        }
        let run_start = i;
        while i < ops.len() && ops[i].0 != DiffTag::Equal {
            i += 1;
        }
        let dels: Vec<&str> = ops[run_start..i]
            .iter()
            .filter(|o| o.0 == DiffTag::Del)
            .map(|o| o.1)
            .collect();
        let inss: Vec<&str> = ops[run_start..i]
            .iter()
            .filter(|o| o.0 == DiffTag::Ins)
            .map(|o| o.1)
            .collect();
        let mut del_segs: Vec<Vec<Seg>> = Vec::with_capacity(dels.len());
        let mut ins_segs: Vec<Vec<Seg>> = Vec::with_capacity(inss.len());
        for k in 0..dels.len().max(inss.len()) {
            match (dels.get(k), inss.get(k)) {
                // Paired removed/added line — refine to word-level segments.
                (Some(d), Some(n)) => {
                    let (d_seg, n_seg) = word_segments(d, n);
                    del_segs.push(d_seg);
                    ins_segs.push(n_seg);
                }
                // Unpaired (pure remove or pure add) — the whole line is the change.
                (Some(d), None) => del_segs.push(vec![(false, d.to_string())]),
                (None, Some(n)) => ins_segs.push(vec![(false, n.to_string())]),
                (None, None) => {}
            }
        }
        for segs in del_segs {
            full.push(DiffRow::Del { num: oldn, segs });
            bump(&mut oldn);
        }
        for segs in ins_segs {
            full.push(DiffRow::Ins { num: newn, segs });
            bump(&mut newn);
        }
    }

    let mut rows = Vec::new();
    let mut prev_kept: Option<usize> = None;
    for (i, row) in full.into_iter().enumerate() {
        if !keep[i] {
            continue;
        }
        if prev_kept.is_some_and(|p| i > p + 1) {
            rows.push(DiffRow::Gap);
        }
        rows.push(row);
        prev_kept = Some(i);
    }
    rows
}

/// Decimal width of the largest gutter line number across `rows`, or 0 when none
/// carries a number — the signal to render without a number column.
fn diff_num_width<'a>(rows: impl IntoIterator<Item = &'a DiffRow>) -> usize {
    let num = |row: &DiffRow| match row {
        DiffRow::Context { num, .. } | DiffRow::Del { num, .. } | DiffRow::Ins { num, .. } => *num,
        DiffRow::Gap => None,
    };
    match rows.into_iter().filter_map(num).max() {
        Some(n) => n.to_string().len(),
        None => 0,
    }
}

/// Render one diff row. `numw` is the shared number-column width (0 = no column,
/// for entries that predate line-number capture).
fn render_diff_row(row: &DiffRow, numw: usize) -> StyledLine {
    let num_span = |num: Option<usize>| -> Option<Span<'static>> {
        (numw > 0).then(|| {
            let text = match num {
                Some(n) => format!("{n:>numw$}"),
                None => " ".repeat(numw),
            };
            Span::styled(text, Style::default().fg(FAINT))
        })
    };
    match row {
        DiffRow::Context { num, text } => {
            let mut spans = Vec::new();
            spans.extend(num_span(*num));
            spans.push(Span::styled(
                format!("   {text}"),
                Style::default().fg(MUTED),
            ));
            line_with_plain(spans)
        }
        DiffRow::Del { num, segs } => render_change_row(
            num_span(*num),
            '-',
            segs,
            DIFF_DEL_BG,
            DIFF_DEL_HL_BG,
            DIFF_DEL_FG,
            DIFF_DEL_SIGN,
        ),
        DiffRow::Ins { num, segs } => render_change_row(
            num_span(*num),
            '+',
            segs,
            DIFF_ADD_BG,
            DIFF_ADD_HL_BG,
            DIFF_ADD_FG,
            DIFF_ADD_SIGN,
        ),
        DiffRow::Gap => {
            let mut spans = Vec::new();
            spans.extend(num_span(None));
            spans.push(Span::styled("   ⋯".to_string(), Style::default().fg(FAINT)));
            line_with_plain(spans)
        }
    }
}

/// A removed/added line: number gutter, bold `+`/`-` sign, then text split into
/// common runs (`base` tint) and changed runs (`hl` tint). The trailing span keeps
/// a background so `fill_trailing_background` extends it across a wrapped row.
fn render_change_row(
    num: Option<Span<'static>>,
    sign: char,
    segs: &[Seg],
    base: Color,
    hl: Color,
    fg: Color,
    sign_fg: Color,
) -> StyledLine {
    let mut spans = Vec::new();
    spans.extend(num);
    spans.push(Span::styled(
        format!(" {sign} "),
        Style::default()
            .fg(sign_fg)
            .bg(base)
            .add_modifier(Modifier::BOLD),
    ));
    if segs.is_empty() {
        // A blank line changed — keep a (zero-width) tinted span so the row fills.
        spans.push(Span::styled(
            String::new(),
            Style::default().fg(fg).bg(base),
        ));
    }
    for (changed, text) in segs {
        let bg = if *changed { hl } else { base };
        spans.push(Span::styled(text.clone(), Style::default().fg(fg).bg(bg)));
    }
    line_with_plain(spans)
}

/// One cap for every edit tool's diff card — the same change shouldn't truncate
/// differently depending on which tool the model picked.
const MAX_DIFF_LINES: usize = 22;

/// The compact diff card under an edit tool call: removed lines red, added green,
/// changed tokens brightened, unchanged lines dim context. Capped so a big rewrite
/// can't flood the transcript; nothing for tools with no textual diff.
/// `old_content` upgrades a `write_file` card to a real diff.
fn render_edit_diff(
    lines: &mut Vec<StyledLine>,
    name: &str,
    args: &serde_json::Value,
    line_starts: &[Option<usize>],
    old_content: Option<&str>,
) {
    if name == "apply_patch" {
        render_patch_diff(lines, args, line_starts);
        return;
    }
    const CONTEXT: usize = 3;
    let mut diffs = edit_diffs(name, args);
    if name == "write_file"
        && let Some(old) = old_content
        && let [diff] = diffs.as_mut_slice()
    {
        diff.old = old.to_string();
    }
    if diffs.is_empty() {
        return;
    }

    // One hunk per edit, separated by a `⋯` in a multi_edit.
    let mut rows: Vec<DiffRow> = Vec::new();
    for (i, d) in diffs.iter().enumerate() {
        let start = line_starts.get(i).copied().flatten();
        let hunk = build_hunk(&d.old, &d.new, start, CONTEXT);
        if hunk.is_empty() {
            continue;
        }
        if !rows.is_empty() {
            rows.push(DiffRow::Gap);
        }
        rows.extend(hunk);
    }
    if rows.is_empty() {
        return;
    }

    // No outer indent so a wrapped changed line stays a single flush block.
    let numw = diff_num_width(&rows);
    for row in rows.iter().take(MAX_DIFF_LINES) {
        lines.push(render_diff_row(row, numw));
    }
    if rows.len() > MAX_DIFF_LINES {
        lines.push(line_with_plain(vec![Span::styled(
            format!("    … (+{} more)", rows.len() - MAX_DIFF_LINES),
            Style::default().fg(FAINT),
        )]));
    }
}

/// Render an `apply_patch` call as a per-file diff: a filename header over the
/// same numbered, word-refined diff used for `edit_file`.
fn render_patch_diff(
    lines: &mut Vec<StyledLine>,
    args: &serde_json::Value,
    line_starts: &[Option<usize>],
) {
    const MAX_LINES: usize = MAX_DIFF_LINES;
    const CONTEXT: usize = 3;
    let diffs = edit_diffs("apply_patch", args);
    if diffs.is_empty() {
        return;
    }

    enum Item {
        Header(String),
        Row(DiffRow),
    }
    let mut items: Vec<Item> = Vec::new();
    let mut last: Option<&str> = None;
    for (i, d) in diffs.iter().enumerate() {
        if last != Some(d.path.as_str()) {
            items.push(Item::Header(d.path.clone()));
            last = Some(&d.path);
        }
        let start = line_starts.get(i).copied().flatten();
        for row in build_hunk(&d.old, &d.new, start, CONTEXT) {
            items.push(Item::Row(row));
        }
    }

    let numw = diff_num_width(items.iter().filter_map(|it| match it {
        Item::Row(r) => Some(r),
        Item::Header(_) => None,
    }));
    for item in items.iter().take(MAX_LINES) {
        lines.push(match item {
            Item::Header(path) => line_with_plain(vec![Span::styled(
                format!("  {path}"),
                Style::default().fg(TOOL),
            )]),
            Item::Row(row) => render_diff_row(row, numw),
        });
    }
    if items.len() > MAX_LINES {
        lines.push(line_with_plain(vec![Span::styled(
            format!("    … (+{} more)", items.len() - MAX_LINES),
            Style::default().fg(FAINT),
        )]));
    }
}

/// A plan as a framed card (header + body + optional footer hint) so it stands
/// apart from an ordinary reply.
pub(super) fn push_plan_card(
    lines: &mut Vec<StyledLine>,
    bars: &mut Vec<Option<Color>>,
    reasoning: Option<ReasoningView<'_>>,
    content: &str,
    width: u16,
    footer: Option<&str>,
) {
    push_styled_line(
        lines,
        "Implementation plan",
        Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
    );
    bars.push(Some(TOOL));
    push_assistant_blocks(lines, bars, reasoning, content, width, TOOL);
    if let Some(footer) = footer {
        push_styled_line(lines, footer, Style::default().fg(FAINT));
        bars.push(Some(TOOL));
    }
}

pub(super) const PLAN_MAX_VISIBLE: usize = 5;

fn plan_status(item: &serde_json::Value) -> &str {
    item.get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("pending")
}

/// `[start, end)` window of steps to show when a plan exceeds `max`: the active
/// step (`in_progress`, else first `pending`) at the top, backfilling upward only
/// when it sits near the end.
fn plan_window(items: &[serde_json::Value], max: usize) -> (usize, usize) {
    let len = items.len();
    if len <= max {
        return (0, len);
    }
    let focus = items
        .iter()
        .position(|i| plan_status(i) == "in_progress")
        .or_else(|| items.iter().position(|i| plan_status(i) == "pending"))
        .unwrap_or(0);
    let start = focus.min(len - max);
    (start, start + max)
}

/// A faint `… N more` marker for a run of hidden plan steps.
fn plan_more_line(n: usize) -> StyledLine {
    line_with_plain(vec![Span::styled(
        format!("  … {n} more"),
        Style::default().fg(FAINT),
    )])
}

/// Render an `update_plan` checklist card: a "Plan N/M done" header over one
/// line per step, each prefixed by a status glyph (done = green ✔, active =
/// teal ▸, pending = muted ○). Completed steps are dimmed and struck through so
/// the eye lands on what's left. Over `PLAN_MAX_VISIBLE` steps windows to the
/// active step (`… N more` for the rest). `content` is the JSON `[{step,status}]`.
pub(super) fn render_plan(lines: &mut Vec<StyledLine>, content: &str) {
    let items = serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    if items.is_empty() {
        return;
    }
    let done = items
        .iter()
        .filter(|i| plan_status(i) == "completed")
        .count();
    lines.push(line_with_plain(vec![
        Span::styled(
            "Plan".to_string(),
            Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {done}/{} done", items.len()),
            Style::default().fg(FAINT),
        ),
    ]));
    let (start, end) = plan_window(&items, PLAN_MAX_VISIBLE);
    if start > 0 {
        lines.push(plan_more_line(start));
    }
    for item in &items[start..end] {
        let step = item.get("step").and_then(|v| v.as_str()).unwrap_or("");
        let (glyph, glyph_color, text_style) = match plan_status(item) {
            "completed" => (
                "✔",
                ASSISTANT,
                Style::default()
                    .fg(FAINT)
                    .add_modifier(Modifier::CROSSED_OUT),
            ),
            "in_progress" => (
                "▸",
                TOOL,
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            _ => ("○", MUTED, Style::default().fg(MUTED)),
        };
        lines.push(line_with_plain(vec![
            Span::styled(format!("  {glyph} "), Style::default().fg(glyph_color)),
            Span::styled(step.to_string(), text_style),
        ]));
    }
    if end < items.len() {
        lines.push(plan_more_line(items.len() - end));
    }
}

/// Whether a stored plan card (`content` is the JSON `[{step,status}]` array) has
/// at least one step and every step is `completed`. A finished plan is hidden
/// from the panel (`plan_panel_lines`) and dropped on the next message
/// (`clear_completed_plan`).
pub(super) fn plan_all_completed(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .is_some_and(|items| {
            !items.is_empty()
                && items
                    .iter()
                    .all(|i| i.get("status").and_then(|s| s.as_str()) == Some("completed"))
        })
}

/// True when no step is past `pending` — a proposed plan not yet executed. Dropped
/// at the next user turn (`clear_stale_plan`); an approved one is re-emitted anyway.
pub(super) fn plan_unstarted(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .is_some_and(|items| {
            !items.is_empty()
                && items.iter().all(|i| {
                    !matches!(
                        i.get("status").and_then(|s| s.as_str()),
                        Some("in_progress") | Some("completed")
                    )
                })
        })
}

/// Display width of a string (sum of per-character terminal widths).
fn cell_width(s: &str) -> usize {
    s.chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum()
}

/// Distribute `budget` display columns across a table's columns. Keeps natural
/// widths when they already fit; otherwise shrinks the widest columns first —
/// down toward each column's longest word, then below it only if still forced —
/// until the total fits. Always returns widths summing to ≤ `budget`, each ≥ 1.
fn fit_column_widths(natural: &[usize], min_word: &[usize], budget: usize) -> Vec<usize> {
    let ncols = natural.len();
    let mut widths: Vec<usize> = natural.to_vec();
    let total: usize = widths.iter().sum();
    if ncols == 0 || total <= budget {
        return widths;
    }

    // Soft floors: avoid shrinking a column below its longest word while another
    // column still has slack — keeps words intact as long as possible. Cap the
    // floor so one giant token can't starve every other column.
    let floor_cap = (budget / ncols).max(1);
    let floors: Vec<usize> = min_word.iter().map(|&w| w.min(floor_cap).max(1)).collect();

    // Shrink the widest column above its floor, one column at a time, until we fit
    // or every column is at its floor; then, if still over, shrink the widest
    // column above a hard minimum of 1 (this hard-breaks words).
    let mut excess = total - budget;
    for floor in [floors.as_slice(), &vec![1usize; ncols]] {
        while excess > 0 {
            let pick = (0..ncols)
                .filter(|&i| widths[i] > floor[i])
                .max_by_key(|&i| widths[i]);
            let Some(i) = pick else { break };
            widths[i] -= 1;
            excess -= 1;
        }
    }
    widths
}

/// Word-wrap a plain table cell to `width` display columns, hard-breaking any
/// word wider than the column. Whitespace runs collapse to a single space.
fn wrap_cell_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for word in text.split_whitespace() {
        let ww = cell_width(word);
        if ww > width {
            // Oversized word: flush the line, then hard-break the word across rows,
            // carrying the trailing partial chunk into the next line.
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            let chunks = wrap_one_line(word, width);
            let split = chunks.len().saturating_sub(1);
            out.extend(chunks[..split].iter().cloned());
            if let Some(tail) = chunks.last() {
                cur = tail.clone();
                cur_w = cell_width(tail);
            }
            continue;
        }
        let sep = usize::from(!cur.is_empty());
        if cur_w + sep + ww > width {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
            cur_w = ww;
        } else {
            if sep == 1 {
                cur.push(' ');
                cur_w += 1;
            }
            cur.push_str(word);
            cur_w += ww;
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

/// Condense a sub-agent `task` for the transcript label: strip the second-person
/// instruction preamble models tend to write ("You are reviewing …", "Your task
/// is to …", "Please …") so the label shows the actual subtask, not boilerplate
/// the model wrote to brief the sub-agent. Leaves a task without a preamble as-is.
fn condense_subagent_task(task: &str) -> String {
    let t = task.trim();
    // Longest / most specific first so a generic prefix doesn't pre-empt a fuller
    // one. Matched case-insensitively against the start (all entries are ASCII, so
    // the byte length is a valid slice boundary on the original-cased string).
    const PREAMBLES: &[&str] = &[
        "you have been asked to ",
        "you are going to ",
        "you are tasked with ",
        "i would like you to ",
        "i'd like you to ",
        "i want you to ",
        "i need you to ",
        "your task is to ",
        "your job is to ",
        "your goal is to ",
        "your task: ",
        "you need to ",
        "you should ",
        "you must ",
        "you will ",
        "you are ",
        "you're ",
        "please ",
        "task: ",
    ];
    let lower = t.to_ascii_lowercase();
    for p in PREAMBLES {
        if let Some(rest) = lower.strip_prefix(p)
            && !rest.trim().is_empty()
        {
            return t[p.len()..].trim().to_string();
        }
    }
    t.to_string()
}

/// Display canonicalization: models sometimes emit Claude Code vocabulary
/// (`Task`, `Read`, …) which would fall through to the raw-name jargon path.
pub(super) fn canonical_tool_name(name: &str) -> &str {
    crate::agent::subagents::normalize_tool_name(name).unwrap_or(name)
}

/// One-line argument summary for a tool's `→ verb(...)` line (the salient field).
fn tool_arg_summary(name: &str, args: &serde_json::Value, cwd: &str) -> String {
    let pick = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match canonical_tool_name(name) {
        // File paths: relative to cwd, left-truncated so the basename survives.
        "read_file" | "list_dir" | "write_file" | "edit_file" | "multi_edit" | "delete_file" => {
            display_path(pick("path"), cwd)
        }
        "glob" | "grep" => truncate_chars(pick("pattern"), 60),
        "run_bash" => truncate_chars(&condense_command(pick("command"), cwd), 60),
        "web_fetch" => truncate_chars(pick("url"), 60),
        "skill" => truncate_chars(pick("name"), 60),
        // The question is the salient detail; the answer lands on the `⎿` result line.
        "ask_user" => truncate_chars(pick("question"), 72),
        // "subagent" is jargon — the short `label`, else the task (preamble
        // stripped), renders as the label itself (see `render_tool_call`).
        // `description`/`prompt` are Claude Code's names for the same args.
        // A named delegation (`agent`/`subagent_type`) leads with the profile
        // name so the transcript attributes the work: `code-reviewer — <task>`.
        "subagent" => {
            let label = [pick("label"), pick("description")]
                .into_iter()
                .find(|s| !s.trim().is_empty());
            let body = match label {
                Some(l) => truncate_chars(l.trim(), 72),
                None => {
                    let task = if pick("task").is_empty() {
                        pick("prompt")
                    } else {
                        pick("task")
                    };
                    truncate_chars(&condense_subagent_task(task), 72)
                }
            };
            match [pick("agent"), pick("subagent_type")]
                .into_iter()
                .find(|s| !s.trim().is_empty())
            {
                Some(agent) if !body.is_empty() => format!("{} — {}", agent.trim(), body),
                Some(agent) => agent.trim().to_string(),
                None => body,
            }
        }
        _ => String::new(),
    }
}

/// A file path for the transcript: made relative to the agent's `cwd` (the footer
/// already shows it), then — if still too wide — truncated from the LEFT on a
/// segment boundary so the basename (what distinguishes sibling files) survives.
fn display_path(path: &str, cwd: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let rel = strip_cwd(path, cwd);
    truncate_path_left(&rel, 56)
}

/// Strip a leading `cwd/` so paths under the working dir render relative; a path
/// equal to the cwd becomes `.`; paths elsewhere are returned unchanged.
fn strip_cwd(path: &str, cwd: &str) -> String {
    let cwd = cwd.trim_end_matches(['/', '\\']);
    if cwd.is_empty() {
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix(cwd) {
        let rest = rest.trim_start_matches(['/', '\\']);
        return if rest.is_empty() {
            ".".to_string()
        } else {
            rest.to_string()
        };
    }
    path.to_string()
}

/// Truncate from the left, keeping whole trailing path segments (and the
/// basename) under `max` columns, with a leading `…/`.
pub(super) fn truncate_path_left(s: &str, max: usize) -> String {
    // `max` is a column budget — measure display width (CJK / wide glyphs are 2
    // columns each), not char count, or a wide path overflows its slot. Mirrors
    // `cell_width` / `truncate_to_width` above.
    if cell_width(s) <= max {
        return s.to_string();
    }
    let segments: Vec<&str> = s.split(['/', '\\']).filter(|p| !p.is_empty()).collect();
    let mut kept = String::new();
    for seg in segments.iter().rev() {
        let candidate = if kept.is_empty() {
            (*seg).to_string()
        } else {
            format!("{seg}/{kept}")
        };
        // +2 leaves room for the leading `…/` (each glyph one column).
        if cell_width(&candidate) + 2 > max {
            break;
        }
        kept = candidate;
    }
    if kept.is_empty() {
        // A single oversized segment (long basename) — hard left-truncate to the
        // rightmost characters that fit in `max - 1` columns (the `…` takes one).
        let budget = max.saturating_sub(1);
        let mut tail: Vec<char> = Vec::new();
        let mut w = 0usize;
        for c in s.chars().rev() {
            let cw = UnicodeWidthChar::width(c).unwrap_or(0);
            if w + cw > budget {
                break;
            }
            tail.push(c);
            w += cw;
        }
        tail.reverse();
        let tail: String = tail.into_iter().collect();
        return format!("…{tail}");
    }
    format!("…/{kept}")
}

/// Compact `⎿` summary: a line count for multi-line output, else the single line
/// (with the agent's cwd stripped so result paths stay short).
/// Strip ANSI escape sequences and other control characters from text about to
/// be shown in the TUI. Captured command output (`git -c color.ui=always …`,
/// `printf '\033[..'`, anything forced to `--color=always`) can carry escape
/// bytes that ratatui mis-measures (zero/unknown width) and paints as garbage;
/// we only ever want the visible characters. Tabs become a single space so words
/// stay separated.
pub(super) fn strip_ansi_and_controls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                // CSI: ESC [ … final byte in 0x40..=0x7e.
                Some('[') => {
                    chars.next();
                    for d in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … terminated by BEL or ST (ESC \).
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\x07' {
                            break;
                        }
                        if d == '\x1b' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Lone ESC or a 2-byte escape — drop the following byte.
                _ => {
                    chars.next();
                }
            }
            continue;
        }
        if c == '\t' {
            out.push(' ');
        } else if !c.is_control() {
            out.push(c);
        }
    }
    out
}

/// Clean one captured `!cmd` line with single-line cursor semantics, so a
/// carriage-return overwrite collapses to its final visible state. The PTY merges
/// stdout+stderr and splits only on `\n`, so a spinner/progress bar (`\r{frame}`
/// repeated) arrives as one newline-less run; stripping each `\r` as a control byte
/// (as `strip_ansi_and_controls` does) would keep every frame — one garbled line.
pub(super) fn render_output_line(raw: &str) -> String {
    let mut line: Vec<char> = Vec::new();
    let mut cursor = 0usize;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => match chars.peek() {
                // CSI: ESC [ … final byte in 0x40..=0x7e. Only `K` (erase-in-line) alters text.
                Some('[') => {
                    chars.next();
                    let mut params = String::new();
                    let mut final_byte = '\0';
                    for d in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            final_byte = d;
                            break;
                        }
                        params.push(d);
                    }
                    if final_byte == 'K' {
                        match params.as_str() {
                            "" | "0" => line.truncate(cursor.min(line.len())),
                            "1" => line.iter_mut().take(cursor).for_each(|slot| *slot = ' '),
                            "2" => line.clear(),
                            _ => {}
                        }
                    }
                }
                // OSC: ESC ] … terminated by BEL or ST (ESC \).
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\x07' {
                            break;
                        }
                        if d == '\x1b' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Lone ESC or a 2-byte escape — drop the following byte.
                _ => {
                    chars.next();
                }
            },
            '\r' => cursor = 0,
            // Match `strip_ansi_and_controls`: one space per tab, advancing one column.
            '\t' => {
                while line.len() < cursor {
                    line.push(' ');
                }
                if cursor < line.len() {
                    line[cursor] = ' ';
                } else {
                    line.push(' ');
                }
                cursor += 1;
            }
            c if c.is_control() => {}
            c => {
                while line.len() < cursor {
                    line.push(' ');
                }
                if cursor < line.len() {
                    line[cursor] = c;
                } else {
                    line.push(c);
                }
                cursor += 1;
            }
        }
    }
    line.into_iter().collect()
}

/// A single-line result is the meaningful outcome ("wrote x", "+1 −1") — show it
/// cwd-stripped and sanitized. Multi-line results fold in [`render_tool_result`].
fn tool_result_summary(s: &str, cwd: &str) -> String {
    let cwd = cwd.trim_end_matches(['/', '\\']);
    let clean = strip_ansi_and_controls(s.trim());
    let stripped = if cwd.is_empty() {
        clean
    } else {
        clean.replace(&format!("{cwd}/"), "")
    };
    truncate_chars(&stripped, 60)
}

/// Unit for a tool's multi-line result count. File listers/searchers count
/// files/entries/matches, not "lines"; everything else (read_file, run_bash
/// output) is line-oriented.
fn count_unit(tool: Option<&str>, count: usize) -> &'static str {
    let one = count == 1;
    match tool {
        Some("list_dir") => {
            if one {
                "entry"
            } else {
                "entries"
            }
        }
        Some("glob") => {
            if one {
                "file"
            } else {
                "files"
            }
        }
        Some("grep") => {
            if one {
                "match"
            } else {
                "matches"
            }
        }
        _ => {
            if one {
                "line"
            } else {
                "lines"
            }
        }
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let t: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{t}…")
}

/// Render markdown to styled transcript lines. `width` is the text-area width
/// tables are laid out to fit (so they never exceed the pane and get sheared by
/// the line word-wrapper); pass 0 for an unconstrained default. Non-table
/// content is wrapped later by [`wrap_transcript`], so `width` only affects tables.
pub(super) fn render_markdown_lines(content: &str, width: u16) -> Vec<StyledLine> {
    let mut options = MdOptions::empty();
    options.insert(MdOptions::ENABLE_STRIKETHROUGH);
    options.insert(MdOptions::ENABLE_TABLES);
    options.insert(MdOptions::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(content, options);
    let mut renderer = MarkdownRenderer::new(width);

    for event in parser {
        renderer.push_event(event);
    }

    renderer.finish()
}

pub(super) struct MarkdownRenderer {
    lines: Vec<StyledLine>,
    current_spans: Vec<Span<'static>>,
    current_plain: String,
    inline_style: InlineStyle,
    heading: Option<HeadingLevel>,
    quote_depth: usize,
    list_stack: Vec<ListState>,
    item_prefix: Option<String>,
    code_block: Option<CodeFence>,
    /// Accumulates a GFM table until its end, then emits aligned columns.
    table: Option<TableAcc>,
    /// Text-area width (display columns) tables are laid out to fit.
    table_width: usize,
}

/// Buffers a markdown table's cells so it can be rendered as aligned columns
/// (the renderer is otherwise streaming, but a table needs all rows to size).
#[derive(Default)]
struct TableAcc {
    rows: Vec<Vec<String>>,
    cur_row: Vec<String>,
    cur_cell: String,
}

impl MarkdownRenderer {
    fn new(width: u16) -> Self {
        // 0 means "unconstrained" — fall back to a comfortable default so a table
        // rendered outside the transcript (e.g. a direct call) still looks sane.
        let table_width = if width == 0 { 80 } else { usize::from(width) };
        Self {
            lines: Vec::new(),
            current_spans: Vec::new(),
            current_plain: String::new(),
            inline_style: InlineStyle::default(),
            heading: None,
            quote_depth: 0,
            list_stack: Vec::new(),
            item_prefix: None,
            code_block: None,
            table: None,
            table_width,
        }
    }

    fn finish(mut self) -> Vec<StyledLine> {
        self.flush_line();
        self.lines
    }

    fn push_event(&mut self, event: MdEvent<'_>) {
        match event {
            MdEvent::Start(tag) => self.start_tag(tag),
            MdEvent::End(tag) => self.end_tag(tag),
            MdEvent::Text(text) => self.push_text(text.as_ref()),
            MdEvent::Code(text) => self.push_inline_code(text.as_ref()),
            MdEvent::SoftBreak | MdEvent::HardBreak => self.flush_line(),
            MdEvent::Rule => {
                self.flush_line();
                self.lines.push(line_plain(
                    "────────────────────────────────".to_string(),
                    Style::default().fg(FAINT),
                ));
            }
            MdEvent::Html(text) | MdEvent::InlineHtml(text) => self.push_text(text.as_ref()),
            MdEvent::FootnoteReference(text) => self.push_text(text.as_ref()),
            MdEvent::TaskListMarker(checked) => {
                self.ensure_prefix();
                let marker = if checked { "☑ " } else { "☐ " };
                self.push_span(marker.to_string(), Style::default().fg(ACCENT));
            }
            _ => {}
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.flush_line();
                self.heading = Some(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth += 1;
            }
            Tag::List(start) => {
                self.flush_line();
                self.list_stack.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_pending();
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::CodeBlock(kind) => {
                self.flush_line();
                self.code_block = Some(CodeFence::new(kind));
            }
            Tag::Table(_) => {
                self.flush_line();
                self.table = Some(TableAcc::default());
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.cur_row.clear();
                }
            }
            Tag::TableCell => {
                if let Some(table) = &mut self.table {
                    table.cur_cell.clear();
                }
            }
            Tag::Emphasis => self.inline_style.emphasis += 1,
            Tag::Strong => self.inline_style.strong += 1,
            Tag::Strikethrough => self.inline_style.strike += 1,
            Tag::Link { .. } => self.inline_style.link += 1,
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_line();
            }
            TagEnd::Heading(_) => {
                self.flush_line();
                self.heading = None;
                self.lines.push(blank_line());
            }
            TagEnd::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.lines.push(blank_line());
            }
            TagEnd::List(_) => {
                // `flush_pending` (not `flush_line`) so the list's content end
                // doesn't add its own blank on top of the explicit one below —
                // that double-blanked the gap after a list.
                self.flush_pending();
                self.list_stack.pop();
                self.lines.push(blank_line());
            }
            TagEnd::Item => {
                // No `flush_line`: a loose list's item content was already flushed
                // at `End(Paragraph)`, so flushing here would emit a blank between
                // every item. `flush_pending` keeps tight lists working (their text
                // is still pending) without spacing loose ones out.
                self.flush_pending();
                self.item_prefix = None;
            }
            TagEnd::CodeBlock => {
                if let Some(block) = self.code_block.take() {
                    self.emit_code_block(block);
                    self.lines.push(blank_line());
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    let cell = table.cur_cell.trim().to_string();
                    table.cur_row.push(cell);
                    table.cur_cell.clear();
                }
            }
            TagEnd::TableHead | TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.cur_row);
                    table.rows.push(row);
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.emit_table(table);
                    self.lines.push(blank_line());
                }
            }
            TagEnd::Emphasis => {
                self.inline_style.emphasis = self.inline_style.emphasis.saturating_sub(1)
            }
            TagEnd::Strong => self.inline_style.strong = self.inline_style.strong.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline_style.strike = self.inline_style.strike.saturating_sub(1)
            }
            TagEnd::Link => self.inline_style.link = self.inline_style.link.saturating_sub(1),
            _ => {}
        }
    }

    fn emit_code_block(&mut self, block: CodeFence) {
        let label = if block.language.is_empty() {
            "code".to_string()
        } else {
            block.language
        };
        self.lines.push(line_plain(
            format!("  {label}"),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));

        let content = if block.content.is_empty() {
            String::new()
        } else {
            block.content
        };
        for raw_line in content.lines() {
            self.lines.push(line_plain(
                format!("  {raw_line}"),
                Style::default().fg(TEXT),
            ));
        }
        if content.is_empty() || content.ends_with('\n') {
            self.lines
                .push(line_plain("  ".to_string(), Style::default().fg(TEXT)));
        }
    }

    /// Render a buffered GFM table as a bordered, width-aware box (Claude-Code
    /// style): columns are sized to fit the available text width, long cells wrap
    /// across multiple lines instead of overflowing, and the whole table never
    /// exceeds the pane — so the transcript word-wrapper can't shear it apart.
    /// A bold header row sits under a top border, a `├─┼─┤` rule splits it from
    /// the body, and a bottom border closes the box.
    fn emit_table(&mut self, table: TableAcc) {
        let rows = table.rows;
        let Some(ncols) = rows.iter().map(Vec::len).max().filter(|n| *n > 0) else {
            return;
        };

        // Natural (unwrapped) width of each column, plus its widest single word —
        // the column can't shrink below that word without hard-breaking it, so the
        // word width is the soft floor the fair-shrink pass tries to respect.
        let mut natural = vec![0usize; ncols];
        let mut min_word = vec![1usize; ncols];
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                natural[i] = natural[i].max(cell_width(cell));
                let longest = cell.split_whitespace().map(cell_width).max().unwrap_or(0);
                min_word[i] = min_word[i].max(longest.max(1));
            }
        }

        // Box chrome per row: a `│` on each side and between every column, plus one
        // space of padding on each side of every cell. The rest is column content.
        let chrome = (ncols + 1) + 2 * ncols;
        let budget = self.table_width.saturating_sub(chrome).max(ncols);
        let widths = fit_column_widths(&natural, &min_word, budget);

        let border = Style::default().fg(FAINT);
        let rule = |left: &str, mid: &str, right: &str| -> StyledLine {
            let segs: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
            line_plain(format!("{left}{}{right}", segs.join(mid)), border)
        };

        let last = rows.len().saturating_sub(1);
        self.lines.push(rule("┌", "┬", "┐"));
        for (ri, row) in rows.iter().enumerate() {
            let header = ri == 0;
            // Wrap each cell to its column width; the row is as tall as its tallest cell.
            let cells: Vec<Vec<String>> = (0..ncols)
                .map(|i| wrap_cell_text(row.get(i).map(String::as_str).unwrap_or(""), widths[i]))
                .collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let cell_style = if header {
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            for k in 0..height {
                let mut spans: Vec<Span<'static>> = vec![Span::styled("│", border)];
                for (i, w) in widths.iter().enumerate() {
                    let text = cells[i].get(k).map(String::as_str).unwrap_or("");
                    let mut padded = String::with_capacity(w + 2);
                    padded.push(' ');
                    padded.push_str(text);
                    padded.push_str(&" ".repeat(w.saturating_sub(cell_width(text))));
                    padded.push(' ');
                    spans.push(Span::styled(padded, cell_style));
                    spans.push(Span::styled("│", border));
                }
                self.lines.push(line_with_plain(spans));
            }
            // Header gets a `├─┼─┤` rule only when a body follows; close with the
            // bottom border after the final row.
            if header && last > 0 {
                self.lines.push(rule("├", "┼", "┤"));
            }
            if ri == last {
                self.lines.push(rule("└", "┴", "┘"));
            }
        }
    }

    fn next_item_prefix(&mut self) -> String {
        if let Some(list) = self.list_stack.last_mut() {
            list.take_prefix()
        } else {
            "• ".to_string()
        }
    }

    fn push_text(&mut self, text: &str) {
        if let Some(block) = &mut self.code_block {
            block.content.push_str(text);
            return;
        }
        if let Some(table) = &mut self.table {
            table.cur_cell.push_str(text);
            return;
        }

        for (idx, part) in text.split('\n').enumerate() {
            if idx > 0 {
                self.flush_line();
            }
            if !part.is_empty() {
                self.ensure_prefix();
                self.push_span(part.to_string(), self.current_style());
            }
        }
    }

    fn push_inline_code(&mut self, text: &str) {
        if let Some(table) = &mut self.table {
            table.cur_cell.push_str(text);
            return;
        }
        self.ensure_prefix();
        // No surrounding padding — the markdown source already supplies spacing
        // and the accent color marks the span; padding caused double spaces.
        self.push_span(
            text.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        );
    }

    fn current_style(&self) -> Style {
        let mut style = Style::default().fg(TEXT);
        if self.inline_style.emphasis > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.inline_style.strong > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.inline_style.strike > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.inline_style.link > 0 {
            style = style.fg(LINK).add_modifier(Modifier::UNDERLINED);
        }
        if let Some(level) = self.heading {
            style = heading_style(level);
        }
        style
    }

    fn ensure_prefix(&mut self) {
        if !self.current_spans.is_empty() || !self.current_plain.is_empty() {
            return;
        }
        if self.quote_depth > 0 {
            let prefix = format!("{} ", "▎".repeat(self.quote_depth));
            self.push_span(prefix, Style::default().fg(QUOTE));
        }
        if let Some(prefix) = self.item_prefix.take() {
            self.push_span(prefix, Style::default().fg(ACCENT));
        }
    }

    fn push_span(&mut self, text: String, style: Style) {
        self.current_plain.push_str(&text);
        self.current_spans.push(Span::styled(text, style));
    }

    fn flush_line(&mut self) {
        if self.current_spans.is_empty() {
            if !self.lines.last().is_some_and(|line| line.plain.is_empty()) {
                self.lines.push(blank_line());
            }
            self.current_plain.clear();
            return;
        }
        self.flush_pending();
    }

    /// Emit any pending inline content as a line, but — unlike [`flush_line`] —
    /// never push a paragraph-break blank when there's nothing pending. Used at
    /// list-item boundaries so a *loose* list (whose items pulldown wraps in
    /// paragraphs, leaving `End(Item)` to flush empty) renders tight, with no
    /// blank line between items, matching a tight list.
    fn flush_pending(&mut self) {
        if self.current_spans.is_empty() {
            self.current_plain.clear();
            return;
        }
        let line = StyledLine {
            line: Line::from(std::mem::take(&mut self.current_spans)),
            plain: std::mem::take(&mut self.current_plain),
        };
        self.lines.push(line);
    }
}

#[derive(Default)]
pub(super) struct InlineStyle {
    emphasis: usize,
    strong: usize,
    strike: usize,
    link: usize,
}

pub(super) struct ListState {
    next_number: Option<u64>,
}

impl ListState {
    fn new(start: Option<u64>) -> Self {
        Self { next_number: start }
    }

    fn take_prefix(&mut self) -> String {
        match self.next_number {
            Some(number) => {
                self.next_number = Some(number + 1);
                format!("{number}. ")
            }
            None => "• ".to_string(),
        }
    }
}

pub(super) struct CodeFence {
    language: String,
    content: String,
}

impl CodeFence {
    fn new(kind: CodeBlockKind<'_>) -> Self {
        let language = match kind {
            CodeBlockKind::Indented => String::new(),
            CodeBlockKind::Fenced(name) => name.to_string(),
        };
        Self {
            language,
            content: String::new(),
        }
    }
}

pub(super) fn heading_style(level: HeadingLevel) -> Style {
    match level {
        HeadingLevel::H1 => Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    }
}

pub(super) fn blank_line() -> StyledLine {
    line_plain(String::new(), Style::default())
}

pub(super) fn line_plain(text: String, style: Style) -> StyledLine {
    StyledLine {
        plain: text.clone(),
        line: Line::from(Span::styled(text, style)),
    }
}

pub(super) fn line_with_plain(spans: Vec<Span<'static>>) -> StyledLine {
    let mut plain = String::new();
    for span in &spans {
        plain.push_str(span.content.as_ref());
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

pub(super) fn push_styled_line(lines: &mut Vec<StyledLine>, text: impl Into<String>, style: Style) {
    lines.push(line_plain(text.into(), style));
}

/// Collapse runs of blank lines (and drop trailing blanks), dropping each
/// line's parallel bar color in lockstep so the two vectors stay aligned.
pub(super) fn compact_lines_and_bars(lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
    let mut out_lines = Vec::with_capacity(lines.len());
    let mut out_bars = Vec::with_capacity(bars.len());
    // Start `false` so a single intentional leading blank survives — the
    // transcript intro opens with `EMPTY_STATE_TOP_GAP` so the banner keeps its
    // top padding once a message lands. Runs of blanks still collapse below.
    let mut last_was_blank = false;

    for (line, bar) in lines.drain(..).zip(bars.drain(..)) {
        let is_blank = line.plain.trim().is_empty();
        if is_blank && last_was_blank {
            continue;
        }
        last_was_blank = is_blank;
        out_lines.push(line);
        out_bars.push(bar);
    }

    while out_lines
        .last()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        out_lines.pop();
        out_bars.pop();
    }

    *lines = out_lines;
    *bars = out_bars;
}

#[cfg(test)]
mod render_tests {
    use super::{
        condense_subagent_task, parallel_call_row_text, render_edit_diff, strip_ansi_and_controls,
        subagent_row_text, tool_arg_summary, tool_result_summary,
    };

    #[test]
    fn edit_diff_expands_tabs() {
        let args = serde_json::json!({
            "path": "main.go",
            "old_string": "\tif len(step, hours) > 12 {\n\t\tstep = 4\n\t}",
            "new_string": "\tif len(hours) > 12 {\n\t\tstep = 4\n\t}",
        });
        let mut lines = Vec::new();
        render_edit_diff(&mut lines, "edit_file", &args, &[None], None);
        assert!(!lines.is_empty());
        let texts: Vec<String> = lines
            .iter()
            .map(|l| {
                l.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(
            texts.iter().all(|t| !t.contains('\t')),
            "raw tab leaked into diff spans: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("    if len(hours) > 12 {")),
            "expanded indent missing: {texts:?}"
        );
    }

    #[test]
    fn write_file_renders_content_as_additions() {
        let args = serde_json::json!({
            "path": "src/new.rs",
            "content": "fn main() {\n    println!(\"hi\");\n}\n",
        });
        let mut lines = Vec::new();
        render_edit_diff(&mut lines, "write_file", &args, &[Some(1)], None);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| {
                l.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("fn main() {")),
            "written content missing from diff: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("println!(\"hi\");")),
            "written content missing from diff: {texts:?}"
        );
    }

    #[test]
    fn write_file_empty_content_renders_nothing() {
        let args = serde_json::json!({ "path": "src/empty.rs", "content": "" });
        let mut lines = Vec::new();
        render_edit_diff(&mut lines, "write_file", &args, &[None], None);
        assert!(lines.is_empty(), "empty write should emit no diff rows");
    }

    #[test]
    fn condense_subagent_task_strips_instruction_preamble() {
        assert_eq!(
            condense_subagent_task("You are reviewing a SvelteKit chat page component"),
            "reviewing a SvelteKit chat page component"
        );
        assert_eq!(
            condense_subagent_task("Your task is to audit the auth flow"),
            "audit the auth flow"
        );
        assert_eq!(
            condense_subagent_task("Please investigate the crash"),
            "investigate the crash"
        );
        // A task with no preamble is left untouched.
        assert_eq!(
            condense_subagent_task("Review the gateway proxy changes"),
            "Review the gateway proxy changes"
        );
        // Never strip down to nothing — a bare preamble stays as-is.
        assert_eq!(condense_subagent_task("You are"), "You are");
    }

    #[test]
    fn tool_action_label_subagent_prefers_label_and_accepts_cc_vocabulary() {
        use super::tool_action_label;
        // A short `label` beats the specialist name in the status line.
        assert_eq!(
            tool_action_label(
                "subagent",
                &serde_json::json!({"label": "audit auth flow", "agent": "reviewer"}),
                ""
            ),
            "delegating: audit auth flow"
        );
        assert_eq!(
            tool_action_label("subagent", &serde_json::json!({"agent": "reviewer"}), ""),
            "delegating to reviewer"
        );
        // Claude Code's `Task` call (description/prompt/subagent_type args)
        // renders as a delegation, not `running Task`.
        assert_eq!(
            tool_action_label(
                "Task",
                &serde_json::json!({"description": "deep-dive engine", "prompt": "…", "subagent_type": "explore"}),
                ""
            ),
            "delegating: deep-dive engine"
        );
        // Other hallucinated Claude Code names canonicalize for display too.
        assert_eq!(
            tool_action_label("Bash", &serde_json::json!({"command": "ls"}), ""),
            "running ls"
        );
    }

    #[test]
    fn tool_arg_summary_subagent_prefers_label_over_task() {
        assert_eq!(
            tool_arg_summary(
                "subagent",
                &serde_json::json!({"label": "audit auth flow", "task": "You are auditing the auth flow in depth"}),
                ""
            ),
            "audit auth flow"
        );
        // Claude Code arg names work as fallbacks.
        assert_eq!(
            tool_arg_summary(
                "Task",
                &serde_json::json!({"prompt": "Please investigate the crash"}),
                ""
            ),
            "investigate the crash"
        );
    }

    #[test]
    fn tool_arg_summary_subagent_leads_with_named_agent() {
        // A named delegation attributes the row to the profile: `agent — task`.
        assert_eq!(
            tool_arg_summary(
                "subagent",
                &serde_json::json!({"agent": "code-reviewer", "label": "audit auth flow"}),
                ""
            ),
            "code-reviewer — audit auth flow"
        );
        // Claude Code's `subagent_type` is the same field; falls back to the task.
        assert_eq!(
            tool_arg_summary(
                "Task",
                &serde_json::json!({"subagent_type": "explorer", "prompt": "Please investigate the crash"}),
                ""
            ),
            "explorer — investigate the crash"
        );
        // Named delegate with no label/task renders the bare agent name.
        assert_eq!(
            tool_arg_summary(
                "subagent",
                &serde_json::json!({"agent": "code-reviewer"}),
                ""
            ),
            "code-reviewer"
        );
        // No agent → unchanged label-only behavior (generic delegate).
        assert_eq!(
            tool_arg_summary(
                "subagent",
                &serde_json::json!({"label": "audit auth flow"}),
                ""
            ),
            "audit auth flow"
        );
    }

    #[test]
    fn subagent_row_text_running_and_done_forms() {
        use std::time::{Duration, Instant};
        let mut row = super::super::shared::SubagentRow {
            name: "audit auth flow".to_string(),
            action: "reading engine.rs".to_string(),
            step: 4,
            started: Instant::now(),
            denied: None,
            done: None,
        };
        let text = subagent_row_text(&row);
        assert!(
            text.starts_with("  ↳ audit auth flow — reading engine.rs · step 4 ("),
            "unexpected running row: {text}"
        );
        row.denied = Some("run_bash".to_string());
        assert!(
            subagent_row_text(&row).ends_with(") · run_bash denied"),
            "denied marker missing: {}",
            subagent_row_text(&row)
        );
        row.done = Some((true, 8, 1200, Duration::from_secs(32)));
        assert_eq!(
            subagent_row_text(&row),
            "  ✓ audit auth flow — done (32s · 8 step(s) · 1.2k tokens)"
        );
        row.done = Some((false, 12, 0, Duration::from_secs(61)));
        assert_eq!(
            subagent_row_text(&row),
            "  ✗ audit auth flow — no answer (1m 1s · 12 step(s))"
        );
    }

    #[test]
    fn parallel_call_row_text_forms() {
        let args = serde_json::json!({"label": "Audit the auth flow"});
        assert_eq!(
            parallel_call_row_text("subagent", &args, (None, false), ""),
            "  ↳ Audit the auth flow"
        );
        assert_eq!(
            parallel_call_row_text(
                "subagent",
                &args,
                (Some("34 lines\nmore".into()), false),
                ""
            ),
            "  ✓ Audit the auth flow — 34 lines"
        );
        assert_eq!(
            parallel_call_row_text("subagent", &args, (Some("boom".into()), true), ""),
            "  ✗ Audit the auth flow — boom"
        );
        assert_eq!(
            parallel_call_row_text("subagent", &serde_json::json!({}), (None, false), ""),
            "  ↳ sub-agent"
        );
        assert_eq!(
            parallel_call_row_text(
                "grep",
                &serde_json::json!({"pattern": "hover"}),
                (None, false),
                ""
            ),
            "  ↳ searching hover"
        );
    }

    #[test]
    fn strip_ansi_removes_csi_osc_and_controls() {
        // CSI color codes gone, visible text kept.
        assert_eq!(strip_ansi_and_controls("\x1b[32mok\x1b[0m"), "ok");
        // OSC (title) sequence terminated by BEL is dropped.
        assert_eq!(strip_ansi_and_controls("\x1b]0;title\x07done"), "done");
        // Bare control chars dropped; tab becomes a space.
        assert_eq!(strip_ansi_and_controls("a\x07b\tc"), "ab c");
        // Plain text is untouched.
        assert_eq!(
            strip_ansi_and_controls("nothing special"),
            "nothing special"
        );
    }

    #[test]
    fn single_line_summary_is_sanitized() {
        // A single-line colored result renders with no escape bytes leaking in.
        let out = tool_result_summary("\x1b[31mfatal: bad ref\x1b[0m", "");
        assert_eq!(out, "fatal: bad ref");
        assert!(!out.contains('\x1b'));
    }
}
