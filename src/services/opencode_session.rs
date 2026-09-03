//! `x-opencode-session` for OpenCode Zen / Go: their backend routes on one id
//! per conversation, and clients that omit it are slated to error.

use crate::services::session_affinity;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

pub const SESSION_HEADER: &str = "x-opencode-session";

/// Base URLs always carry a scheme, so an unparsable target (`copilot`) isn't one.
pub fn is_opencode_url(url: &str) -> bool {
    let Some(host) = reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    host == "opencode.ai" || host.ends_with(".opencode.ai")
}

/// `body` is the outgoing request in any wire shape, `None` when there is none.
pub fn with_session_header(
    builder: reqwest::RequestBuilder,
    target_url: &str,
    body: Option<&Value>,
) -> reqwest::RequestBuilder {
    if !is_opencode_url(target_url) {
        return builder;
    }
    builder.header(SESSION_HEADER, session_affinity::conversation_id(body))
}

/// Passthrough form: a client's own id wins.
pub fn insert_session_header(headers: &mut HeaderMap, target_url: &str, body: Option<&Value>) {
    if !is_opencode_url(target_url) || headers.contains_key(SESSION_HEADER) {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(&session_affinity::conversation_id(body)) {
        headers.insert(HeaderName::from_static(SESSION_HEADER), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recognizes_opencode_urls_only() {
        assert!(is_opencode_url(
            "https://opencode.ai/zen/v1/chat/completions"
        ));
        assert!(is_opencode_url("https://opencode.ai/zen/go/v1/messages"));
        assert!(is_opencode_url("https://OpenCode.ai/zen/v1/models"));
        assert!(is_opencode_url("https://api.opencode.ai/v1/models"));
        assert!(!is_opencode_url(
            "https://api.openai.com/v1/chat/completions"
        ));
        assert!(!is_opencode_url("https://opencode.ai.evil.test/zen/v1"));
        assert!(!is_opencode_url("/v1/responses"));
    }

    #[test]
    fn stamps_the_header_for_opencode_targets_only() {
        let client = reqwest::Client::new();
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let stamp = |url: &str| {
            with_session_header(client.post(url), url, Some(&body))
                .build()
                .unwrap()
                .headers()
                .get(SESSION_HEADER)
                .cloned()
        };
        assert_eq!(
            stamp("https://opencode.ai/zen/go/v1/chat/completions").unwrap(),
            session_affinity::conversation_id(Some(&body)).as_str()
        );
        assert!(stamp("https://api.openai.com/v1/chat/completions").is_none());
    }

    #[test]
    fn header_map_form_keeps_a_client_supplied_id() {
        let url = "https://opencode.ai/zen/v1/messages";
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SESSION_HEADER),
            HeaderValue::from_static("ses_client"),
        );
        insert_session_header(&mut headers, url, None);
        assert_eq!(headers.get(SESSION_HEADER).unwrap(), "ses_client");

        let mut empty = HeaderMap::new();
        insert_session_header(&mut empty, url, None);
        assert_eq!(
            empty.get(SESSION_HEADER).unwrap(),
            session_affinity::conversation_id(None).as_str()
        );

        let mut other = HeaderMap::new();
        insert_session_header(&mut other, "https://api.anthropic.com/v1/messages", None);
        assert!(other.get(SESSION_HEADER).is_none());
    }
}
