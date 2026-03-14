#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProtocol {
    Openai,
    Anthropic,
    Google,
}

impl ProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Openai => 0,
            Self::Anthropic => 1,
            Self::Google => 2,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Anthropic,
            2 => Self::Google,
            _ => Self::Openai,
        }
    }
}

pub fn normalize_protocol_base(base_url: &str) -> &str {
    let trimmed = base_url.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed)
}

pub fn is_anthropic_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    normalized.contains("api.anthropic.com") || normalized.ends_with("/anthropic")
}

pub fn is_google_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    normalized.contains("generativelanguage.googleapis.com")
}

pub fn detect_provider_protocol(base_url: &str) -> ProviderProtocol {
    if is_anthropic_endpoint(base_url) {
        ProviderProtocol::Anthropic
    } else if is_google_endpoint(base_url) {
        ProviderProtocol::Google
    } else {
        ProviderProtocol::Openai
    }
}

/// Returns true if the HTTP status suggests the endpoint path doesn't exist
/// (wrong protocol), as opposed to auth/model/rate errors.
pub fn is_protocol_mismatch(status: u16) -> bool {
    matches!(status, 404 | 405 | 415)
}

/// Returns fallback protocol candidates to try after `current` fails.
/// Excludes Google unless the URL suggests a Google endpoint.
pub fn fallback_protocols(current: ProviderProtocol, base_url: &str) -> Vec<ProviderProtocol> {
    let include_google = is_google_endpoint(base_url);
    [ProviderProtocol::Openai, ProviderProtocol::Anthropic, ProviderProtocol::Google]
        .into_iter()
        .filter(|p| *p != current && (*p != ProviderProtocol::Google || include_google))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_anthropic_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic"),
            ProviderProtocol::Anthropic
        );
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic/v1"),
            ProviderProtocol::Anthropic
        );
    }

    #[test]
    fn detects_google_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://generativelanguage.googleapis.com/v1beta"),
            ProviderProtocol::Google
        );
    }

    #[test]
    fn defaults_to_openai_for_other_endpoints() {
        assert_eq!(
            detect_provider_protocol("https://openrouter.ai/api/v1"),
            ProviderProtocol::Openai
        );
    }

    #[test]
    fn is_protocol_mismatch_returns_true_for_404_405_415() {
        assert!(is_protocol_mismatch(404));
        assert!(is_protocol_mismatch(405));
        assert!(is_protocol_mismatch(415));
    }

    #[test]
    fn is_protocol_mismatch_returns_false_for_other_codes() {
        assert!(!is_protocol_mismatch(200));
        assert!(!is_protocol_mismatch(401));
        assert!(!is_protocol_mismatch(500));
    }

    #[test]
    fn fallback_protocols_excludes_current_and_google_for_generic_url() {
        let result = fallback_protocols(ProviderProtocol::Openai, "https://api.example.com");
        assert_eq!(result, vec![ProviderProtocol::Anthropic]);
    }

    #[test]
    fn fallback_protocols_includes_google_for_google_url() {
        let result = fallback_protocols(
            ProviderProtocol::Openai,
            "https://generativelanguage.googleapis.com/v1beta",
        );
        assert!(result.contains(&ProviderProtocol::Google));
        assert!(result.contains(&ProviderProtocol::Anthropic));
    }
}
