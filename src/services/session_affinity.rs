//! One stable id per conversation for upstreams that route by it. aivo's routers
//! hold no session, so it hashes the opening user turn — minus the system prompt
//! and reminders, whose dates and env would churn it. Compaction drops that turn,
//! so the id rotates once per compaction.

use serde_json::Value;

const FINGERPRINT_CHARS: usize = 2048;

pub fn conversation_id(body: Option<&Value>) -> String {
    format!("aivo_{}", hex(&fingerprint(body)))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn random_bytes() -> [u8; 16] {
    let mut b = [0u8; 16];
    crate::services::rng::fill(&mut b);
    b
}

fn fingerprint(body: Option<&Value>) -> [u8; 16] {
    static PROCESS: std::sync::LazyLock<[u8; 16]> = std::sync::LazyLock::new(random_bytes);
    let Some(opening) = body.and_then(opening_turn) else {
        return *PROCESS;
    };
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(opening.as_bytes());
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

fn opening_turn(body: &Value) -> Option<String> {
    if let Some(Value::String(text)) = body.get("input") {
        return conversation_text(text);
    }
    let items = ["messages", "input", "contents"]
        .iter()
        .find_map(|key| body.get(*key).and_then(Value::as_array))?;
    items.iter().find_map(|item| {
        // Gemini's first `contents` entry may omit the role; it is the user's.
        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
        (role == "user").then(|| turn_text(item)).flatten()
    })
}

fn turn_text(item: &Value) -> Option<String> {
    match item.get("content").or_else(|| item.get("parts"))? {
        Value::String(text) => conversation_text(text),
        Value::Array(parts) => conversation_text(
            &parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<String>(),
        ),
        _ => None,
    }
}

fn conversation_text(raw: &str) -> Option<String> {
    let text: String = strip_reminders(raw)
        .trim()
        .chars()
        .take(FINGERPRINT_CHARS)
        .collect();
    (!text.is_empty()).then_some(text)
}

fn strip_reminders(text: &str) -> String {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        match rest[start..].find(CLOSE) {
            Some(end) => rest = &rest[start + end + CLOSE.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(text: &str) -> Value {
        json!({"messages": [{"role": "user", "content": text}]})
    }

    #[test]
    fn id_is_stable_across_the_turns_of_one_conversation() {
        let first = json!({"messages": [
            {"role": "system", "content": "you are aivo"},
            {"role": "user", "content": "port the parser to serde"}
        ]});
        let later = json!({"messages": [
            {"role": "system", "content": "you are aivo, today is Tuesday"},
            {"role": "user", "content": "port the parser to serde"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now add tests"}
        ]});
        assert_eq!(conversation_id(Some(&first)), conversation_id(Some(&later)));
        assert!(conversation_id(Some(&first)).starts_with("aivo_"));
        assert_ne!(
            conversation_id(Some(&chat("one"))),
            conversation_id(Some(&chat("two")))
        );
    }

    #[test]
    fn reads_the_opening_turn_from_every_wire_shape() {
        let chat_parts = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": "what is this"}
        ]}]});
        let responses = json!({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "what is this"}
        ]}]});
        let gemini = json!({"contents": [{"parts": [{"text": "what is this"}]}]});
        let bare = json!({"input": "what is this"});
        let expected = conversation_id(Some(&chat("what is this")));
        for body in [chat_parts, responses, gemini, bare] {
            assert_eq!(conversation_id(Some(&body)), expected, "shape: {body}");
        }
    }

    #[test]
    fn ignores_reminders_riding_the_opening_turn() {
        let expected = conversation_id(Some(&chat("plan the port")));
        let tail = chat("plan the port\n\n<system-reminder>Plan mode is active</system-reminder>");
        let part = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "plan the port"},
            {"type": "text", "text": "<system-reminder>Plan mode is active</system-reminder>"}
        ]}]});
        let lead = chat("<system-reminder>CLAUDE.md says hi</system-reminder>\nplan the port");
        for body in [tail, part, lead] {
            assert_eq!(conversation_id(Some(&body)), expected, "body: {body}");
        }
    }

    #[test]
    fn falls_back_to_a_stable_process_id() {
        let listing = conversation_id(None);
        assert_eq!(listing, conversation_id(None));
        assert_eq!(
            listing,
            conversation_id(Some(
                &json!({"messages": [{"role": "assistant", "content": "hi"}]})
            ))
        );
        assert_eq!(listing, conversation_id(Some(&chat("   "))));
        assert_eq!(
            listing,
            conversation_id(Some(&chat("<system-reminder>only</system-reminder>")))
        );
    }
}
