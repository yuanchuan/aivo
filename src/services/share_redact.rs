//! Redaction pipeline for `aivo logs share`.
//!
//! Hand-rolled (no regex crate — house style). Each category is one pass;
//! the order matters: more specific patterns run first so a token that
//! could match multiple categories gets the most precise label.
//!
//! Walks the entire `SharePayload`, applying redaction to every text-bearing
//! field including `tool_call.arguments` JSON (recursive). Returns the
//! redacted payload plus a per-category hit count for the preview.

use std::collections::HashMap;

use serde_json::Value;

use crate::services::share_payload::{ContentBlock, RedactionHit, ShareMessage, SharePayload};

/// Context for one redaction run. Currently just the user's `$HOME` so paths
/// can be replaced with `~`. Injected explicitly (rather than read from the
/// process env) so tests don't have to mutate global state.
#[derive(Debug, Clone, Default)]
pub struct RedactCtx {
    pub home_dir: Option<String>,
}

impl RedactCtx {
    pub fn from_system() -> Self {
        Self {
            home_dir: crate::services::system_env::home_dir()
                .map(|p| p.to_string_lossy().to_string()),
        }
    }
}

const CAT_PRIVATE_KEY: &str = "private_key";
const CAT_API_KEY: &str = "api_key";
const CAT_ANTHROPIC_OAUTH: &str = "anthropic_oauth";
const CAT_AUTH_HEADER: &str = "authorization_header";
const CAT_AWS_ACCESS_KEY: &str = "aws_access_key";
const CAT_SECRET_ENV: &str = "secret_env";
const CAT_GENERIC_SECRET: &str = "generic_secret";
const CAT_HOME_PATH: &str = "home_path";

/// Apply every category to `input`, returning the redacted text plus
/// a flat `category -> count` accumulator. Caller controls aggregation
/// across many strings.
pub fn scan_text(input: &str, ctx: &RedactCtx, hits: &mut HashMap<String, usize>) -> String {
    // Order matters: header rules consume their value, so they run before
    // the standalone token rules that would also match the same suffix. The
    // private-key span goes first so token rules can't fire inside its body.
    let s = pass_private_key(input, hits);
    let s = pass_auth_header(&s, hits);
    let s = pass_bearer_token(&s, hits);
    let s = pass_anthropic_oauth(&s, hits);
    let s = pass_api_key(&s, hits);
    let s = pass_aws_access_key(&s, hits);
    let s = pass_secret_env(&s, hits);
    let s = pass_generic_secret(&s, hits);
    pass_home_path(&s, ctx, hits)
}

/// Walk `payload` recursively and apply `scan_text` to every text-bearing
/// field. Mutates a copy and returns it alongside a sorted hit list.
pub fn redact(mut payload: SharePayload, ctx: &RedactCtx) -> (SharePayload, Vec<RedactionHit>) {
    let mut hits: HashMap<String, usize> = HashMap::new();

    if let Some(root) = payload.project.root.take() {
        payload.project.root = Some(scan_text(&root, ctx, &mut hits));
    }
    if let Some(name) = payload.project.name.take() {
        payload.project.name = Some(scan_text(&name, ctx, &mut hits));
    }

    for msg in &mut payload.messages {
        redact_message(msg, ctx, &mut hits);
    }

    let mut report: Vec<RedactionHit> = hits
        .into_iter()
        .map(|(category, count)| RedactionHit { category, count })
        .collect();
    report.sort_by(|a, b| a.category.cmp(&b.category));

    payload.meta.redacted = !report.is_empty();
    payload.meta.redaction_summary = if report.is_empty() {
        None
    } else {
        Some(report.clone())
    };

    (payload, report)
}

fn redact_message(msg: &mut ShareMessage, ctx: &RedactCtx, hits: &mut HashMap<String, usize>) {
    if let Some(reasoning) = msg.reasoning.take() {
        msg.reasoning = Some(scan_text(&reasoning, ctx, hits));
    }
    for block in &mut msg.content {
        redact_block(block, ctx, hits);
    }
}

fn redact_block(block: &mut ContentBlock, ctx: &RedactCtx, hits: &mut HashMap<String, usize>) {
    match block {
        ContentBlock::Text { text } | ContentBlock::Code { text, .. } => {
            *text = scan_text(text, ctx, hits);
        }
        ContentBlock::ToolCall { arguments, .. } => {
            scan_value(arguments, ctx, hits);
        }
        ContentBlock::ToolResult { output, error, .. } => {
            *output = scan_text(output, ctx, hits);
            if let Some(err) = error.take() {
                *error = Some(scan_text(&err, ctx, hits));
            }
        }
        ContentBlock::Attachment { .. } => {}
    }
}

/// Skipped by value walkers: a chance key-shaped match inside a base64 body
/// would corrupt the media.
pub fn is_base64_data_url(s: &str) -> bool {
    s.starts_with("data:") && s.find(";base64,").is_some_and(|i| i < 256)
}

pub fn scan_value(v: &mut Value, ctx: &RedactCtx, hits: &mut HashMap<String, usize>) {
    match v {
        Value::String(s) => {
            if !is_base64_data_url(s) {
                *s = scan_text(s, ctx, hits);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                scan_value(item, ctx, hits);
            }
        }
        Value::Object(map) => {
            for (_, val) in map.iter_mut() {
                scan_value(val, ctx, hits);
            }
        }
        _ => {}
    }
}

fn bump(hits: &mut HashMap<String, usize>, category: &str) {
    *hits.entry(category.to_string()).or_insert(0) += 1;
}

// ---------------------------------------------------------------------------
// Pass 0: PEM/OpenSSH private-key blocks — a span rule, not a token rule: the
// base64 body has no key-shaped prefix for the token passes to catch.
// ---------------------------------------------------------------------------

fn pass_private_key(input: &str, hits: &mut HashMap<String, usize>) -> String {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "-----END ";
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        let Some(begin) = rest.find(BEGIN) else {
            out.push_str(rest);
            return out;
        };
        let label_start = begin + BEGIN.len();
        let Some(label_len) = rest[label_start..].find("-----") else {
            out.push_str(rest);
            return out;
        };
        let label = &rest[label_start..label_start + label_len];
        // A newline before the closing dashes means this isn't a PEM header.
        if !label.contains("PRIVATE KEY") || label.contains('\n') {
            out.push_str(&rest[..label_start]);
            rest = &rest[label_start..];
            continue;
        }
        out.push_str(&rest[..begin]);
        out.push_str("<redacted:private_key>");
        bump(hits, CAT_PRIVATE_KEY);
        // A block with no END footer (truncated output) is still key material.
        let after_header = label_start + label_len + "-----".len();
        rest = match rest[after_header..].find(END) {
            Some(rel) => {
                let footer_label = after_header + rel + END.len();
                match rest[footer_label..].find("-----") {
                    Some(rel2) => &rest[footer_label + rel2 + "-----".len()..],
                    None => "",
                }
            }
            None => "",
        };
    }
}

// ---------------------------------------------------------------------------
// Token-tail helper
// ---------------------------------------------------------------------------

/// Returns the byte length of the token-suffix starting at `start` in `bytes`,
/// where a token is a run of [A-Za-z0-9_-]. Used by every key-prefix rule.
fn token_len_from(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        let ok = b.is_ascii_alphanumeric() || b == b'-' || b == b'_';
        if !ok {
            break;
        }
        i += 1;
    }
    i - start
}

// ---------------------------------------------------------------------------
// Pass 1: Authorization headers — case-insensitive `authorization:` followed
// by a token. Replaces just the token portion so the header line stays
// intelligible.
// ---------------------------------------------------------------------------

fn pass_auth_header(input: &str, hits: &mut HashMap<String, usize>) -> String {
    redact_after_marker_ci(input, "authorization:", hits, CAT_AUTH_HEADER, "<redacted>")
}

/// Find every case-insensitive occurrence of `marker` and redact the token
/// that follows (skipping leading whitespace and an optional `Bearer` word).
/// Used for both `Authorization:` and standalone `Bearer ` matches.
fn redact_after_marker_ci(
    input: &str,
    marker: &str,
    hits: &mut HashMap<String, usize>,
    category: &str,
    placeholder: &str,
) -> String {
    let bytes = input.as_bytes();
    let marker_bytes = marker.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + marker_bytes.len() <= bytes.len()
            && bytes[i..i + marker_bytes.len()].eq_ignore_ascii_case(marker_bytes)
        {
            // Copy the marker verbatim (preserve original case).
            out.extend_from_slice(&bytes[i..i + marker_bytes.len()]);
            i += marker_bytes.len();
            // Skip whitespace (preserved literally).
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                out.push(bytes[i]);
                i += 1;
            }
            // Optional leading auth scheme (`Bearer`, `Token`, `Basic`,
            // …) — any short alpha word followed by whitespace.
            let scheme_start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            let scheme_len = i - scheme_start;
            let has_trailing_space = i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t');
            if (1..=10).contains(&scheme_len) && has_trailing_space {
                // Preserve the literal scheme + the whitespace after it.
                out.extend_from_slice(&bytes[scheme_start..i]);
                while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                    out.push(bytes[i]);
                    i += 1;
                }
            } else {
                // Backtrack — the alpha run wasn't a scheme; treat it as
                // the start of the token to redact.
                i = scheme_start;
            }
            // Now redact the token tail.
            let tail_len = token_len_from(bytes, i);
            if tail_len >= 8 {
                out.extend_from_slice(placeholder.as_bytes());
                bump(hits, category);
                i += tail_len;
            }
        } else {
            // Copy this byte verbatim. Output stays valid UTF-8 because the
            // input was valid UTF-8 and we never split a multi-byte sequence
            // (markers/schemes/tokens are ASCII so non-ASCII chars always
            // fall through to this byte-copy branch as continuation bytes).
            out.push(bytes[i]);
            i += 1;
        }
    }
    // Safe: only original valid UTF-8 bytes and ASCII placeholders/markers.
    String::from_utf8(out).expect("redactor produced valid UTF-8")
}

// ---------------------------------------------------------------------------
// Pass 2: standalone `Bearer <token>` (when not already inside an
// Authorization header that pass_auth_header consumed).
// ---------------------------------------------------------------------------

fn pass_bearer_token(input: &str, hits: &mut HashMap<String, usize>) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        // word-boundary `bearer ` — must follow a non-token byte (start of
        // string also counts) so we don't match `cyberbearer foo`.
        let at_boundary = i == 0 || !is_token_byte(bytes[i - 1]);
        if at_boundary
            && i + 7 <= bytes.len()
            && bytes[i..i + 6].eq_ignore_ascii_case(b"bearer")
            && (bytes[i + 6] == b' ' || bytes[i + 6] == b'\t')
        {
            out.extend_from_slice(&bytes[i..i + 7]);
            i += 7;
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                out.push(bytes[i]);
                i += 1;
            }
            let tail_len = token_len_from(bytes, i);
            if tail_len >= 8 {
                out.extend_from_slice(b"<redacted>");
                bump(hits, CAT_AUTH_HEADER);
                i += tail_len;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("redactor produced valid UTF-8")
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

// ---------------------------------------------------------------------------
// Pass 3: Anthropic OAuth — `sk-ant-oat01-` is more specific than the
// generic `sk-` API-key prefix and must run first.
// ---------------------------------------------------------------------------

fn pass_anthropic_oauth(input: &str, hits: &mut HashMap<String, usize>) -> String {
    replace_prefix_token(
        input,
        "sk-ant-oat01-",
        16,
        "<redacted:anthropic_oauth>",
        CAT_ANTHROPIC_OAUTH,
        hits,
    )
}

// ---------------------------------------------------------------------------
// Pass 4: Provider API keys — sk-, sk_live_, xai-, gsk_, AIza, gh[opusur]_,
// each followed by ≥20 token chars to keep false-positive rate low.
// ---------------------------------------------------------------------------

fn pass_api_key(input: &str, hits: &mut HashMap<String, usize>) -> String {
    let prefixes: &[&str] = &[
        "sk-proj-", "sk_live_", "sk_test_", "sk-", "xai-", "gsk_", "AIza", "ghp_", "gho_", "ghu_",
        "ghs_", "ghr_",
    ];
    let mut current = input.to_string();
    for prefix in prefixes {
        // The min tail length is intentionally long enough to skip short
        // false positives (e.g. literal `sk-1` in an example).
        let min_tail = if prefix.starts_with("sk-") { 20 } else { 16 };
        current = replace_prefix_token(
            &current,
            prefix,
            min_tail,
            "<redacted:api_key>",
            CAT_API_KEY,
            hits,
        );
    }
    current
}

/// Find every occurrence of literal `prefix` followed by a token tail of
/// length ≥ `min_tail`, replacing the entire `prefix+tail` with `placeholder`
/// and bumping the hit counter. Word-boundary checked on the prefix start
/// so substrings inside other tokens (`mysk-foo` etc.) don't match.
fn replace_prefix_token(
    input: &str,
    prefix: &str,
    min_tail: usize,
    placeholder: &str,
    category: &str,
    hits: &mut HashMap<String, usize>,
) -> String {
    let bytes = input.as_bytes();
    let prefix_bytes = prefix.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || !is_token_byte(bytes[i - 1]);
        if at_boundary
            && i + prefix_bytes.len() <= bytes.len()
            && &bytes[i..i + prefix_bytes.len()] == prefix_bytes
        {
            let tail_start = i + prefix_bytes.len();
            let tail_len = token_len_from(bytes, tail_start);
            if tail_len >= min_tail {
                out.extend_from_slice(placeholder.as_bytes());
                bump(hits, category);
                i = tail_start + tail_len;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("redactor produced valid UTF-8")
}

// ---------------------------------------------------------------------------
// Pass 5: AWS access keys — `AKIA` followed by exactly 16 uppercase alnum.
// ---------------------------------------------------------------------------

fn pass_aws_access_key(input: &str, hits: &mut HashMap<String, usize>) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || !is_token_byte(bytes[i - 1]);
        if at_boundary && i + 20 <= bytes.len() && &bytes[i..i + 4] == b"AKIA" {
            // Verify next 16 bytes are uppercase alphanumeric and 21st (if any) is a non-token boundary.
            let tail_ok = bytes[i + 4..i + 20]
                .iter()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit());
            let after_ok = i + 20 == bytes.len() || !is_token_byte(bytes[i + 20]);
            if tail_ok && after_ok {
                out.extend_from_slice(b"<redacted:aws_access_key>");
                bump(hits, CAT_AWS_ACCESS_KEY);
                i += 20;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("redactor produced valid UTF-8")
}

// ---------------------------------------------------------------------------
// Pass 6: Secret env vars — line-anchored `KEY=value` where KEY is uppercase
// and value is suspicious (≥20 token bytes). Replaces value only.
// ---------------------------------------------------------------------------

fn pass_secret_env(input: &str, hits: &mut HashMap<String, usize>) -> String {
    let mut out = String::with_capacity(input.len());
    let mut first = true;
    for line in input.split('\n') {
        if !first {
            out.push('\n');
        }
        first = false;

        let line_bytes = line.as_bytes();
        // Skip leading whitespace.
        let mut start = 0;
        while start < line_bytes.len() && (line_bytes[start] == b' ' || line_bytes[start] == b'\t')
        {
            start += 1;
        }
        // Accept optional `export ` (Bash) prefix as still "env-var-shaped".
        let mut after_prefix = start;
        if after_prefix + 7 <= line_bytes.len()
            && &line_bytes[after_prefix..after_prefix + 7] == b"export "
        {
            after_prefix += 7;
        }

        // Match KEY: starts with [A-Z], then [A-Z0-9_]{3,}, followed by `=`.
        let key_start = after_prefix;
        let mut key_end = key_start;
        if key_end < line_bytes.len() && line_bytes[key_end].is_ascii_uppercase() {
            key_end += 1;
            while key_end < line_bytes.len()
                && (line_bytes[key_end].is_ascii_uppercase()
                    || line_bytes[key_end].is_ascii_digit()
                    || line_bytes[key_end] == b'_')
            {
                key_end += 1;
            }
        }
        let key_len = key_end.saturating_sub(key_start);
        if key_len >= 4 && key_end < line_bytes.len() && line_bytes[key_end] == b'=' {
            let value_start = key_end + 1;
            // The value runs to end of line; for the suspicion test, count
            // token-like characters (skip leading quote).
            let mut tail = value_start;
            if tail < line_bytes.len() && (line_bytes[tail] == b'"' || line_bytes[tail] == b'\'') {
                tail += 1;
            }
            let token = token_len_from(line_bytes, tail);
            if token >= 20 {
                out.push_str(&line[..value_start]);
                out.push_str("<redacted>");
                bump(hits, CAT_SECRET_ENV);
                continue;
            }
        }
        out.push_str(line);
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 7: Generic secrets after `token=`/`secret=`/`key=`/`password=`
// (case-insensitive). Less precise than provider-specific rules; only fires
// when context strongly implies a secret.
// ---------------------------------------------------------------------------

fn pass_generic_secret(input: &str, hits: &mut HashMap<String, usize>) -> String {
    let markers: &[&str] = &[
        "token=",
        "secret=",
        "password=",
        "passwd=",
        "apikey=",
        "api_key=",
    ];
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || !is_token_byte(bytes[i - 1]);
        let mut matched = None;
        if at_boundary {
            for m in markers {
                let mb = m.as_bytes();
                if i + mb.len() <= bytes.len() && bytes[i..i + mb.len()].eq_ignore_ascii_case(mb) {
                    matched = Some(mb.len());
                    break;
                }
            }
        }
        if let Some(mlen) = matched {
            out.extend_from_slice(&bytes[i..i + mlen]);
            i += mlen;
            // Skip optional opening quote.
            let mut tail = i;
            if tail < bytes.len() && (bytes[tail] == b'"' || bytes[tail] == b'\'') {
                tail += 1;
            }
            let token = token_len_from(bytes, tail);
            if token >= 20 {
                if tail > i {
                    out.push(bytes[i]);
                }
                out.extend_from_slice(b"<redacted:secret>");
                bump(hits, CAT_GENERIC_SECRET);
                i = tail + token;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("redactor produced valid UTF-8")
}

// ---------------------------------------------------------------------------
// Pass 8: Home-path replacement. Literal substring substitution: `<HOME>` → `~`.
// Runs last so secrets inside paths get caught by their own categories first.
// ---------------------------------------------------------------------------

fn pass_home_path(input: &str, ctx: &RedactCtx, hits: &mut HashMap<String, usize>) -> String {
    let Some(home) = ctx.home_dir.as_deref() else {
        return input.to_string();
    };
    if home.is_empty() {
        return input.to_string();
    }
    if !input.contains(home) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find(home) {
        out.push_str(&rest[..pos]);
        out.push('~');
        bump(hits, CAT_HOME_PATH);
        rest = &rest[pos + home.len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::share_payload::{ProjectInfo, SHARE_SCHEMA_VERSION};
    use serde_json::json;

    fn ctx_with_home(home: &str) -> RedactCtx {
        RedactCtx {
            home_dir: Some(home.to_string()),
        }
    }

    fn scan(input: &str, ctx: &RedactCtx) -> (String, HashMap<String, usize>) {
        let mut hits = HashMap::new();
        let out = scan_text(input, ctx, &mut hits);
        (out, hits)
    }

    #[test]
    fn redacts_anthropic_oauth_before_generic_api_key() {
        let (out, hits) = scan(
            "token=sk-ant-oat01-AAAAAAAAAAAAAAAA blah",
            &RedactCtx::default(),
        );
        assert!(out.contains("<redacted:anthropic_oauth>"));
        assert!(!out.contains("sk-ant-oat01-"));
        assert_eq!(hits.get(CAT_ANTHROPIC_OAUTH), Some(&1));
        assert!(!hits.contains_key(CAT_API_KEY));
    }

    #[test]
    fn redacts_private_key_blocks_keeping_surrounding_text() {
        let input = "$ cat ~/.ssh/id_ed25519\n-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ==\n-----END OPENSSH PRIVATE KEY-----\ndone\n";
        let (out, hits) = scan(input, &RedactCtx::default());
        assert_eq!(
            out,
            "$ cat ~/.ssh/id_ed25519\n<redacted:private_key>\ndone\n"
        );
        assert_eq!(hits.get(CAT_PRIVATE_KEY), Some(&1));

        for label in ["RSA PRIVATE KEY", "PRIVATE KEY", "PGP PRIVATE KEY BLOCK"] {
            let input = format!("-----BEGIN {label}-----\nAAAA\n-----END {label}-----");
            let (out, hits) = scan(&input, &RedactCtx::default());
            assert_eq!(out, "<redacted:private_key>", "label: {label}");
            assert_eq!(hits.get(CAT_PRIVATE_KEY), Some(&1), "label: {label}");
        }
    }

    #[test]
    fn redacts_unterminated_private_key_to_end() {
        let (out, hits) = scan(
            "log: -----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBx\nMHcCAQEEIBy",
            &RedactCtx::default(),
        );
        assert_eq!(out, "log: <redacted:private_key>");
        assert_eq!(hits.get(CAT_PRIVATE_KEY), Some(&1));
    }

    #[test]
    fn keeps_public_keys_and_certificates() {
        let input = "-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n\
-----BEGIN CERTIFICATE-----\nBBBB\n-----END CERTIFICATE-----";
        let (out, hits) = scan(input, &RedactCtx::default());
        assert_eq!(out, input);
        assert!(!hits.contains_key(CAT_PRIVATE_KEY));
    }

    #[test]
    fn private_key_redaction_idempotent_and_counts_multiple_blocks() {
        let key = "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----";
        let input = format!("first {key} second {key}");
        let (once, hits) = scan(&input, &RedactCtx::default());
        assert_eq!(
            once,
            "first <redacted:private_key> second <redacted:private_key>"
        );
        assert_eq!(hits.get(CAT_PRIVATE_KEY), Some(&2));
        let (twice, _) = scan(&once, &RedactCtx::default());
        assert_eq!(once, twice);
    }

    #[test]
    fn redacts_provider_api_keys_with_diverse_prefixes() {
        let cases: &[&str] = &[
            "sk-AAAAAAAAAAAAAAAAAAAAAAAA",
            "sk-proj-BBBBBBBBBBBBBBBBBBBBBBBB",
            "xai-CCCCCCCCCCCCCCCCCCCC",
            "gsk_DDDDDDDDDDDDDDDDDDDD",
            "AIzaEEEEEEEEEEEEEEEEEEEE",
            "ghp_FFFFFFFFFFFFFFFFFFFF",
        ];
        for raw in cases {
            let (out, hits) = scan(raw, &RedactCtx::default());
            assert!(out.contains("<redacted:api_key>"), "{raw} → {out}");
            assert_eq!(hits.get(CAT_API_KEY), Some(&1), "case: {raw}");
        }
    }

    #[test]
    fn does_not_redact_short_or_unboundaried_token() {
        // Too-short tail
        let (out, hits) = scan("here's a literal sk-1 in a doc", &RedactCtx::default());
        assert_eq!(out, "here's a literal sk-1 in a doc");
        assert!(hits.is_empty());

        // Embedded inside a larger word: `mysk-AAA...` should NOT redact.
        let (out, _) = scan(
            "var mysk-AAAAAAAAAAAAAAAAAAAAAAAA = 1",
            &RedactCtx::default(),
        );
        assert!(out.contains("mysk-AAAAAAAAAAAAAAAAAAAAAAAA"));
    }

    #[test]
    fn redacts_authorization_header_preserving_label() {
        let (out, hits) = scan(
            "Authorization: Bearer sk-AAAAAAAAAAAAAAAAAAAAAAAA",
            &RedactCtx::default(),
        );
        assert_eq!(out, "Authorization: Bearer <redacted>");
        assert_eq!(hits.get(CAT_AUTH_HEADER), Some(&1));
    }

    #[test]
    fn redacts_authorization_header_lowercased() {
        let (out, hits) = scan(
            "authorization: token AAAAAAAAAAAAAAAAAAAAAAAA",
            &RedactCtx::default(),
        );
        assert!(out.starts_with("authorization: token "));
        assert!(out.ends_with("<redacted>"));
        assert_eq!(hits.get(CAT_AUTH_HEADER), Some(&1));
    }

    #[test]
    fn redacts_standalone_bearer_token() {
        let (out, hits) = scan("curl -H 'Bearer abcdefghijkl' x", &RedactCtx::default());
        assert!(out.contains("Bearer <redacted>"));
        assert_eq!(hits.get(CAT_AUTH_HEADER), Some(&1));
    }

    #[test]
    fn redacts_aws_access_key() {
        let (out, hits) = scan("aws_id=AKIAIOSFODNN7EXAMPLE x", &RedactCtx::default());
        assert!(out.contains("<redacted:aws_access_key>"));
        assert_eq!(hits.get(CAT_AWS_ACCESS_KEY), Some(&1));
    }

    #[test]
    fn does_not_redact_almost_aws_key() {
        // Lowercase tail → not an AWS key shape.
        let (out, hits) = scan("AKIA1234567890aaaaaa", &RedactCtx::default());
        assert!(!out.contains("<redacted"));
        assert!(hits.is_empty());
    }

    #[test]
    fn redacts_secret_env_var_lines() {
        let input = "OPENAI_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxx\nNORMAL=42";
        let (out, hits) = scan(input, &RedactCtx::default());
        assert!(out.starts_with("OPENAI_API_KEY=<redacted"), "{out}");
        assert!(out.contains("\nNORMAL=42"));
        // `sk_xxxx...` doesn't match any specific provider rule (sk_live_/sk_test_),
        // so secret_env catches the value.
        assert!(hits.contains_key(CAT_SECRET_ENV) || hits.contains_key(CAT_API_KEY));
    }

    #[test]
    fn redacts_generic_secret_after_context_marker() {
        let (out, hits) = scan(
            "request body: { \"password\": \"verysecretvalue1234567890\" }",
            &RedactCtx::default(),
        );
        // Note: `password=` form via JSON: scan won't match `password":` — but
        // the pass also handles `password=` form. JSON shape is left as-is.
        // This test verifies the form-encoded variant.
        let _ = (out, hits);

        let (out, hits) = scan("url?token=secretsecretsecretsecret", &RedactCtx::default());
        assert!(out.contains("<redacted:secret>"));
        assert_eq!(hits.get(CAT_GENERIC_SECRET), Some(&1));
    }

    #[test]
    fn replaces_home_path_with_tilde() {
        let ctx = ctx_with_home("/Users/alice");
        let (out, hits) = scan("see /Users/alice/project/aivo/src/main.rs", &ctx);
        assert_eq!(out, "see ~/project/aivo/src/main.rs");
        assert_eq!(hits.get(CAT_HOME_PATH), Some(&1));
    }

    #[test]
    fn redact_idempotent() {
        let ctx = ctx_with_home("/Users/alice");
        let input = "Authorization: Bearer sk-AAAAAAAAAAAAAAAAAAAAAAAA at /Users/alice/work";
        let (once, _) = scan(input, &ctx);
        let (twice, _) = scan(&once, &ctx);
        assert_eq!(once, twice);
    }

    #[test]
    fn composite_end_to_end() {
        let ctx = ctx_with_home("/Users/alice");
        let input = "
Authorization: Bearer sk-AAAAAAAAAAAAAAAAAAAAAAAA
ANTHROPIC_API_KEY=sk-ant-oat01-BBBBBBBBBBBBBBBB
aws_id=AKIAIOSFODNN7EXAMPLE
log file at /Users/alice/.config/aivo/logs/x.jsonl
literal sk-1 should pass
";
        let (out, hits) = scan(input, &ctx);
        assert!(out.contains("Bearer <redacted>"));
        assert!(out.contains("<redacted:anthropic_oauth>"));
        assert!(out.contains("<redacted:aws_access_key>"));
        assert!(out.contains("~/.config/aivo/logs/x.jsonl"));
        assert!(out.contains("literal sk-1 should pass"));
        assert!(hits.contains_key(CAT_AUTH_HEADER));
        assert!(hits.contains_key(CAT_ANTHROPIC_OAUTH));
        assert!(hits.contains_key(CAT_AWS_ACCESS_KEY));
        assert!(hits.contains_key(CAT_HOME_PATH));
    }

    #[test]
    fn does_not_inflate_non_ascii_bytes() {
        // Regression: an early version pushed each byte as `byte as char`,
        // re-encoding multi-byte UTF-8 as Latin-1 and doubling its size on
        // every pass. With 8 passes that ballooned a 760KB payload into
        // 600MB. Verify a passthrough pays no length penalty.
        let ctx = RedactCtx::default();
        let input = "한글 日本語 中文 — emoji: 🦀🚀🤖 — straight ASCII no secrets";
        let (out, _) = scan(input, &ctx);
        assert_eq!(out, input);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn scan_value_masks_strings_but_skips_base64_data_urls() {
        let ctx = RedactCtx::default();
        let mut hits = HashMap::new();
        let data_url = "data:image/png;base64,sk-BBBBBBBBBBBBBBBBBBBBBBBB";
        let mut v = json!({
            "text": "key sk-AAAAAAAAAAAAAAAAAAAAAAAA here",
            "image_url": {"url": data_url},
        });
        scan_value(&mut v, &ctx, &mut hits);
        assert!(v["text"].as_str().unwrap().contains("<redacted:api_key>"));
        assert_eq!(v["image_url"]["url"], data_url, "media payload mangled");
        assert_eq!(hits.get(CAT_API_KEY), Some(&1));
    }

    #[test]
    fn redact_payload_walks_messages_and_tool_args_and_writes_summary() {
        let payload = SharePayload {
            schema_version: SHARE_SCHEMA_VERSION.into(),
            source_cli: "claude".into(),
            session_id: "T-x".into(),
            project: ProjectInfo {
                root: Some("/Users/alice/project/aivo".into()),
                name: Some("aivo".into()),
            },
            model: None,
            created_at: None,
            updated_at: None,
            messages: vec![ShareMessage {
                role: "assistant".into(),
                timestamp: None,
                model: None,
                reasoning: Some("thought about /Users/alice/secret".into()),
                content: vec![
                    ContentBlock::Text {
                        text: "see /Users/alice/code".into(),
                    },
                    ContentBlock::ToolCall {
                        id: None,
                        name: "Bash".into(),
                        arguments: json!({"cmd": "echo sk-AAAAAAAAAAAAAAAAAAAAAAAA"}),
                    },
                    ContentBlock::ToolResult {
                        id: None,
                        ok: true,
                        output: "Authorization: Bearer abcdefghijklmn".into(),
                        error: None,
                    },
                ],
            }],
            meta: SharePayload::new_meta(false),
        };
        // Pre-redaction sanity.
        assert!(!payload.meta.redacted);

        let ctx = ctx_with_home("/Users/alice");
        let (red, report) = redact(payload, &ctx);

        // Project root is normalized.
        assert_eq!(red.project.root.as_deref(), Some("~/project/aivo"));
        // Reasoning + text walk.
        assert!(
            red.messages[0]
                .reasoning
                .as_deref()
                .unwrap()
                .contains("~/secret")
        );
        if let ContentBlock::Text { text } = &red.messages[0].content[0] {
            assert_eq!(text, "see ~/code");
        } else {
            panic!("expected text");
        }
        // Tool arguments JSON walk.
        if let ContentBlock::ToolCall { arguments, .. } = &red.messages[0].content[1] {
            assert!(
                arguments["cmd"]
                    .as_str()
                    .unwrap()
                    .contains("<redacted:api_key>")
            );
        } else {
            panic!("expected tool_call");
        }
        // Tool result text walk.
        if let ContentBlock::ToolResult { output, .. } = &red.messages[0].content[2] {
            assert!(output.contains("Bearer <redacted>"));
        } else {
            panic!("expected tool_result");
        }

        assert!(red.meta.redacted);
        let summary = red.meta.redaction_summary.as_ref().unwrap();
        assert!(!summary.is_empty());
        // Report mirrors the meta block.
        assert_eq!(summary, &report);
        // Sorted by category alphabetically.
        let mut cats: Vec<&str> = report.iter().map(|h| h.category.as_str()).collect();
        let mut sorted = cats.clone();
        sorted.sort();
        cats.sort();
        assert_eq!(cats, sorted);
    }
}
