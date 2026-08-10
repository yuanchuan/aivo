//! Wire-format registry: the (client, upstream) protocol pair matrix.
//! Each edge bundles request/response conversion plus an incremental SSE
//! adapter. Hub-and-spoke around OpenAI Chat.

use std::collections::HashSet;

use anyhow::Result;
use serde_json::Value;

use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, convert_anthropic_to_openai_request,
};
use crate::services::anthropic_chat_response::{
    OpenAIStreamConverter, OpenAIToAnthropicConfig, convert_openai_to_anthropic_message,
};
use crate::services::anthropic_gemini_bridge::{
    AnthropicToGeminiConfig, GeminiToAnthropicConfig, GeminiToAnthropicStreamConverter,
    convert_anthropic_to_gemini_request, convert_anthropic_to_gemini_response,
    convert_gemini_to_anthropic_request, convert_gemini_to_anthropic_response,
};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, convert_gemini_to_openai_chat_request,
    convert_gemini_to_openai_chat_response, convert_openai_chat_to_gemini_request,
    convert_openai_chat_to_gemini_response,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::responses_chat_conversion::{
    ResponsesStreamConverter, ResponsesToChatConversionConfig, ResponsesToChatStreamConverter,
    ToolNamespaceMap, convert_chat_to_responses_request, convert_responses_json_to_chat,
    convert_responses_to_chat_request,
};
use crate::services::serve_responses::convert_chat_response_to_responses_json;
use crate::services::serve_stream_converters::{
    AnthropicToOpenAIStreamConverter, GeminiToOpenAIStreamConverter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireFormat {
    OpenAiChat,
    Anthropic,
    Gemini,
    ResponsesApi,
}

impl WireFormat {
    pub const ALL: [WireFormat; 4] = [
        WireFormat::OpenAiChat,
        WireFormat::Anthropic,
        WireFormat::Gemini,
        WireFormat::ResponsesApi,
    ];
}

/// `Google`'s wire format is Gemini.
impl From<ProviderProtocol> for WireFormat {
    fn from(protocol: ProviderProtocol) -> Self {
        match protocol {
            ProviderProtocol::Openai => WireFormat::OpenAiChat,
            ProviderProtocol::Anthropic => WireFormat::Anthropic,
            ProviderProtocol::Google => WireFormat::Gemini,
            ProviderProtocol::ResponsesApi => WireFormat::ResponsesApi,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeCaps {
    pub request: bool,
    pub response: bool,
    /// Incremental SSE adapter exists (else: buffered single-event emulation).
    pub stream: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    pub client: WireFormat,
    pub upstream: WireFormat,
    pub caps: EdgeCaps,
}

const FULL: EdgeCaps = EdgeCaps {
    request: true,
    response: true,
    stream: true,
};

/// The `Anthropic ↔ Gemini` pair is direct (non-hub) for the fidelity the two
/// Chat hops flatten — see `anthropic_gemini_bridge`.
pub const EDGES: [EdgeInfo; 8] = [
    EdgeInfo {
        client: WireFormat::OpenAiChat,
        upstream: WireFormat::Anthropic,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::OpenAiChat,
        upstream: WireFormat::Gemini,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::OpenAiChat,
        upstream: WireFormat::ResponsesApi,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::Anthropic,
        upstream: WireFormat::OpenAiChat,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::Gemini,
        upstream: WireFormat::OpenAiChat,
        caps: EdgeCaps {
            request: true,
            response: true,
            // Emulated: buffered Chat response wrapped in one Gemini SSE event.
            stream: false,
        },
    },
    EdgeInfo {
        client: WireFormat::ResponsesApi,
        upstream: WireFormat::OpenAiChat,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::Anthropic,
        upstream: WireFormat::Gemini,
        caps: FULL,
    },
    EdgeInfo {
        client: WireFormat::Gemini,
        upstream: WireFormat::Anthropic,
        caps: EdgeCaps {
            request: true,
            response: true,
            // Emulated: buffered Anthropic response in one Gemini SSE event.
            stream: false,
        },
    },
];

pub fn edge(client: WireFormat, upstream: WireFormat) -> Option<&'static EdgeInfo> {
    EDGES
        .iter()
        .find(|e| e.client == client && e.upstream == upstream)
}

pub fn has_pair(client: WireFormat, upstream: WireFormat) -> bool {
    client == upstream || edge(client, upstream).is_some()
}

/// Per-edge request options; variant names the edge, payload its config.
pub enum RequestOptions<'a> {
    ChatToAnthropic {
        default_model: &'a str,
    },
    ChatToGemini {
        default_model: &'a str,
    },
    ChatToResponses,
    AnthropicToChat(&'a AnthropicToOpenAIConfig<'a>),
    GeminiToChat {
        model: &'a str,
        requires_reasoning_content: bool,
        max_tokens_cap: Option<u64>,
    },
    ResponsesToChat(&'a ResponsesToChatConversionConfig),
    AnthropicToGemini {
        default_model: &'a str,
    },
    GeminiToAnthropic {
        model: &'a str,
    },
}

impl RequestOptions<'_> {
    pub fn edge(&self) -> (WireFormat, WireFormat) {
        use WireFormat::*;
        match self {
            Self::ChatToAnthropic { .. } => (OpenAiChat, Anthropic),
            Self::ChatToGemini { .. } => (OpenAiChat, Gemini),
            Self::ChatToResponses => (OpenAiChat, ResponsesApi),
            Self::AnthropicToChat(_) => (Anthropic, OpenAiChat),
            Self::GeminiToChat { .. } => (Gemini, OpenAiChat),
            Self::ResponsesToChat(_) => (ResponsesApi, OpenAiChat),
            Self::AnthropicToGemini { .. } => (Anthropic, Gemini),
            Self::GeminiToAnthropic { .. } => (Gemini, Anthropic),
        }
    }
}

pub fn translate_request(body: &Value, opts: &RequestOptions) -> Value {
    match opts {
        RequestOptions::ChatToAnthropic { default_model } => {
            convert_openai_chat_to_anthropic_request(
                body,
                &OpenAIToAnthropicChatConfig { default_model },
            )
        }
        RequestOptions::ChatToGemini { default_model } => {
            convert_openai_chat_to_gemini_request(body, &OpenAIToGeminiConfig { default_model })
        }
        RequestOptions::ChatToResponses => convert_chat_to_responses_request(body),
        RequestOptions::AnthropicToChat(config) => {
            convert_anthropic_to_openai_request(body, config)
        }
        RequestOptions::GeminiToChat {
            model,
            requires_reasoning_content,
            max_tokens_cap,
        } => convert_gemini_to_openai_chat_request(
            body,
            model,
            *requires_reasoning_content,
            *max_tokens_cap,
        ),
        RequestOptions::ResponsesToChat(config) => convert_responses_to_chat_request(body, config),
        RequestOptions::AnthropicToGemini { default_model } => {
            convert_anthropic_to_gemini_request(body, &AnthropicToGeminiConfig { default_model })
        }
        RequestOptions::GeminiToAnthropic { model } => {
            convert_gemini_to_anthropic_request(body, model)
        }
    }
}

pub enum ResponseOptions<'a> {
    ChatToAnthropic {
        model: &'a str,
    },
    ChatToGemini {
        model: &'a str,
    },
    ChatToResponses,
    AnthropicToChat(&'a OpenAIToAnthropicConfig<'a>),
    GeminiToChat,
    ResponsesToChat {
        model: &'a str,
        custom_tools: &'a HashSet<String>,
        tool_namespaces: &'a ToolNamespaceMap,
    },
    GeminiToAnthropic {
        model: &'a str,
    },
    AnthropicToGemini,
}

impl ResponseOptions<'_> {
    pub fn edge(&self) -> (WireFormat, WireFormat) {
        use WireFormat::*;
        match self {
            Self::ChatToAnthropic { .. } => (OpenAiChat, Anthropic),
            Self::ChatToGemini { .. } => (OpenAiChat, Gemini),
            Self::ChatToResponses => (OpenAiChat, ResponsesApi),
            Self::AnthropicToChat(_) => (Anthropic, OpenAiChat),
            Self::GeminiToChat => (Gemini, OpenAiChat),
            Self::ResponsesToChat { .. } => (ResponsesApi, OpenAiChat),
            Self::GeminiToAnthropic { .. } => (Anthropic, Gemini),
            Self::AnthropicToGemini => (Gemini, Anthropic),
        }
    }
}

/// Fallible only on the Anthropic/Responses-client edges (typed parse).
pub fn translate_response(resp: &Value, opts: &ResponseOptions) -> Result<Value> {
    match opts {
        ResponseOptions::ChatToAnthropic { model } => {
            Ok(convert_anthropic_to_openai_chat_response(resp, model))
        }
        ResponseOptions::ChatToGemini { model } => {
            Ok(convert_gemini_to_openai_chat_response(resp, model))
        }
        ResponseOptions::ChatToResponses => Ok(convert_responses_json_to_chat(resp)),
        ResponseOptions::AnthropicToChat(config) => {
            Ok(convert_openai_to_anthropic_message(resp, config)?)
        }
        ResponseOptions::GeminiToChat => Ok(convert_openai_chat_to_gemini_response(resp)),
        ResponseOptions::ResponsesToChat {
            model,
            custom_tools,
            tool_namespaces,
        } => convert_chat_response_to_responses_json(resp, model, custom_tools, tool_namespaces),
        ResponseOptions::GeminiToAnthropic { model } => Ok(convert_gemini_to_anthropic_response(
            resp,
            &GeminiToAnthropicConfig { model },
        )),
        ResponseOptions::AnthropicToGemini => Ok(convert_anthropic_to_gemini_response(resp)),
    }
}

pub trait StreamAdapter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String>;
    fn finish(&mut self) -> Result<String>;
}

impl StreamAdapter for AnthropicToOpenAIStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        AnthropicToOpenAIStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        AnthropicToOpenAIStreamConverter::finish(self)
    }
}

impl StreamAdapter for GeminiToOpenAIStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        GeminiToOpenAIStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        GeminiToOpenAIStreamConverter::finish(self)
    }
}

impl StreamAdapter for OpenAIStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        OpenAIStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        OpenAIStreamConverter::finish(self)
    }
}

impl StreamAdapter for ResponsesStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        ResponsesStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        Ok(ResponsesStreamConverter::finish(self))
    }
}

impl StreamAdapter for ResponsesToChatStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        ResponsesToChatStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        Ok(ResponsesToChatStreamConverter::finish(self))
    }
}

impl StreamAdapter for GeminiToAnthropicStreamConverter {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        GeminiToAnthropicStreamConverter::push_bytes(self, chunk)
    }
    fn finish(&mut self) -> Result<String> {
        Ok(GeminiToAnthropicStreamConverter::finish(self))
    }
}

/// Pipes `first` into `second`; nests since Chain is itself an adapter.
pub struct Chain {
    first: Box<dyn StreamAdapter + Send>,
    second: Box<dyn StreamAdapter + Send>,
}

impl Chain {
    pub fn new(
        first: Box<dyn StreamAdapter + Send>,
        second: Box<dyn StreamAdapter + Send>,
    ) -> Self {
        Self { first, second }
    }
}

impl StreamAdapter for Chain {
    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        let mid = self.first.push_bytes(chunk)?;
        if mid.is_empty() {
            return Ok(String::new());
        }
        self.second.push_bytes(mid.as_bytes())
    }

    fn finish(&mut self) -> Result<String> {
        let mut out = String::new();
        let mid = self.first.finish()?;
        if !mid.is_empty() {
            out.push_str(&self.second.push_bytes(mid.as_bytes())?);
        }
        out.push_str(&self.second.finish()?);
        Ok(out)
    }
}

/// No Gemini-client variant — its `caps.stream` is false (buffered
/// single-event emulation).
pub enum StreamOptions<'a> {
    ChatToAnthropic {
        model: &'a str,
    },
    ChatToGemini {
        model: &'a str,
    },
    ChatToResponses {
        model: &'a str,
        include_usage: bool,
    },
    AnthropicToChat {
        fallback_model: &'a str,
    },
    ResponsesToChat {
        model: &'a str,
        requires_reasoning_content: bool,
        custom_tools: HashSet<String>,
        tool_namespaces: ToolNamespaceMap,
    },
    GeminiToAnthropic {
        model: &'a str,
    },
}

impl StreamOptions<'_> {
    pub fn edge(&self) -> (WireFormat, WireFormat) {
        use WireFormat::*;
        match self {
            Self::ChatToAnthropic { .. } => (OpenAiChat, Anthropic),
            Self::ChatToGemini { .. } => (OpenAiChat, Gemini),
            Self::ChatToResponses { .. } => (OpenAiChat, ResponsesApi),
            Self::AnthropicToChat { .. } => (Anthropic, OpenAiChat),
            Self::ResponsesToChat { .. } => (ResponsesApi, OpenAiChat),
            Self::GeminiToAnthropic { .. } => (Anthropic, Gemini),
        }
    }
}

pub fn stream_adapter(opts: StreamOptions) -> Box<dyn StreamAdapter + Send> {
    match opts {
        StreamOptions::ChatToAnthropic { model } => {
            Box::new(AnthropicToOpenAIStreamConverter::new(model))
        }
        StreamOptions::ChatToGemini { model } => {
            Box::new(GeminiToOpenAIStreamConverter::new(model))
        }
        StreamOptions::ChatToResponses {
            model,
            include_usage,
        } => Box::new(ResponsesToChatStreamConverter::new(model, include_usage)),
        StreamOptions::AnthropicToChat { fallback_model } => {
            Box::new(OpenAIStreamConverter::new(fallback_model))
        }
        StreamOptions::ResponsesToChat {
            model,
            requires_reasoning_content,
            custom_tools,
            tool_namespaces,
        } => Box::new(
            ResponsesStreamConverter::new(model, requires_reasoning_content)
                .with_custom_tools(custom_tools)
                .with_tool_namespaces(tool_namespaces),
        ),
        StreamOptions::GeminiToAnthropic { model } => {
            Box::new(GeminiToAnthropicStreamConverter::new(model))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::anthropic_chat_response::UsageValueMode;
    use WireFormat::*;
    use serde_json::json;

    fn anthropic_request_config() -> AnthropicToOpenAIConfig<'static> {
        AnthropicToOpenAIConfig {
            default_model: "gpt-4o",
            preserve_stream: true,
            model_transform: None,
            include_reasoning_content: true,
            require_non_empty_reasoning_content: false,
            stringify_other_tool_result_content: true,
            tool_result_supports_multimodal: true,
            fallback_tool_arguments_json: "{}",
        }
    }

    fn anthropic_response_config() -> OpenAIToAnthropicConfig<'static> {
        OpenAIToAnthropicConfig {
            fallback_id: "msg_test",
            model: "m",
            include_created: false,
            usage_value_mode: UsageValueMode::CoerceU64,
        }
    }

    fn responses_router_config() -> ResponsesToChatConversionConfig {
        ResponsesToChatConversionConfig {
            requires_reasoning_content: false,
            target_base_url: "https://example.com/v1".to_string(),
            target_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            model_prefix: None,
            actual_model: None,
            max_tokens_cap: None,
        }
    }

    #[test]
    fn wire_matrix_is_exact() {
        let expect = |c, u| match (c, u) {
            (OpenAiChat, Anthropic)
            | (OpenAiChat, Gemini)
            | (OpenAiChat, ResponsesApi)
            | (Anthropic, OpenAiChat)
            | (ResponsesApi, OpenAiChat)
            | (Anthropic, Gemini) => Some(EdgeCaps {
                request: true,
                response: true,
                stream: true,
            }),
            // Gemini-client edges emulate streaming (buffered single event).
            (Gemini, OpenAiChat) | (Gemini, Anthropic) => Some(EdgeCaps {
                request: true,
                response: true,
                stream: false,
            }),
            _ => None,
        };

        for c in WireFormat::ALL {
            for u in WireFormat::ALL {
                if c == u {
                    assert!(has_pair(c, u), "identity pair {c:?} must be supported");
                    assert!(edge(c, u).is_none(), "identity pair {c:?} is not an edge");
                    continue;
                }
                let caps = edge(c, u).map(|e| e.caps);
                assert_eq!(caps, expect(c, u), "edge {c:?} -> {u:?} drifted");
                assert_eq!(has_pair(c, u), caps.is_some());
            }
        }
        assert_eq!(EDGES.len(), 8, "edge count drifted");
    }

    #[test]
    fn wire_format_from_provider_protocol_is_total() {
        assert_eq!(WireFormat::from(ProviderProtocol::Openai), OpenAiChat);
        assert_eq!(WireFormat::from(ProviderProtocol::Anthropic), Anthropic);
        assert_eq!(WireFormat::from(ProviderProtocol::Google), Gemini);
        assert_eq!(
            WireFormat::from(ProviderProtocol::ResponsesApi),
            ResponsesApi
        );
    }

    fn stream_options_for(c: WireFormat, u: WireFormat) -> Option<StreamOptions<'static>> {
        match (c, u) {
            (OpenAiChat, Anthropic) => Some(StreamOptions::ChatToAnthropic {
                model: "test-model",
            }),
            (OpenAiChat, Gemini) => Some(StreamOptions::ChatToGemini {
                model: "test-model",
            }),
            (OpenAiChat, ResponsesApi) => Some(StreamOptions::ChatToResponses {
                model: "test-model",
                include_usage: false,
            }),
            (Anthropic, OpenAiChat) => Some(StreamOptions::AnthropicToChat {
                fallback_model: "test-model",
            }),
            (ResponsesApi, OpenAiChat) => Some(StreamOptions::ResponsesToChat {
                model: "test-model",
                requires_reasoning_content: false,
                custom_tools: HashSet::new(),
                tool_namespaces: ToolNamespaceMap::new(),
            }),
            (Anthropic, Gemini) => Some(StreamOptions::GeminiToAnthropic {
                model: "test-model",
            }),
            _ => None,
        }
    }

    /// The client-shape marker only survives if the frame is fed to the
    /// correct converter, so it catches cross-wiring.
    #[test]
    fn stream_adapters_match_matrix_and_convert() {
        let frame_and_marker = |c: WireFormat, u: WireFormat| match (c, u) {
            (OpenAiChat, Anthropic) => Some((
                concat!(
                    r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"wired-ok"}}"#,
                    "\n\n"
                ),
                "chat.completion.chunk",
            )),
            (OpenAiChat, Gemini) => Some((
                concat!(
                    r#"data: {"candidates":[{"content":{"parts":[{"text":"wired-ok"}],"role":"model"}}]}"#,
                    "\n\n"
                ),
                "chat.completion.chunk",
            )),
            (OpenAiChat, ResponsesApi) => Some((
                concat!(
                    r#"data: {"type":"response.output_text.delta","delta":"wired-ok"}"#,
                    "\n\n"
                ),
                "chat.completion.chunk",
            )),
            // Both Chat-upstream adapters parse this frame; the marker distinguishes them.
            (Anthropic, OpenAiChat) => Some((
                concat!(
                    r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"wired-ok"}}]}"#,
                    "\n\n"
                ),
                "message_start",
            )),
            (ResponsesApi, OpenAiChat) => Some((
                concat!(
                    r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"wired-ok"}}]}"#,
                    "\n\n"
                ),
                "response.output_text.delta",
            )),
            (Anthropic, Gemini) => Some((
                concat!(
                    r#"data: {"candidates":[{"content":{"parts":[{"text":"wired-ok"}],"role":"model"}}]}"#,
                    "\n\n"
                ),
                "content_block_delta",
            )),
            _ => None,
        };

        for c in WireFormat::ALL {
            for u in WireFormat::ALL {
                if c == u {
                    continue;
                }
                let expected = edge(c, u).map(|e| e.caps.stream).unwrap_or(false);
                match stream_options_for(c, u) {
                    Some(opts) => {
                        assert!(expected, "unexpected adapter for {c:?} -> {u:?}");
                        assert_eq!(opts.edge(), (c, u), "options mapped to the wrong edge");
                        let mut adapter = stream_adapter(opts);
                        let (frame, marker) =
                            frame_and_marker(c, u).expect("frame for streaming edge");
                        let mut out = adapter.push_bytes(frame.as_bytes()).expect("push");
                        out.push_str(&adapter.finish().expect("finish"));
                        assert!(
                            out.contains("wired-ok"),
                            "{c:?} -> {u:?} dropped content: {out}"
                        );
                        assert!(
                            out.contains(marker),
                            "{c:?} -> {u:?} missing client marker {marker:?}: {out}"
                        );
                    }
                    None => assert!(!expected, "missing adapter for {c:?} -> {u:?}"),
                }
            }
        }
    }

    #[test]
    fn chain_composes_adapters_across_two_hops() {
        let first = stream_adapter(stream_options_for(OpenAiChat, Anthropic).expect("edge"));
        let second = stream_adapter(stream_options_for(ResponsesApi, OpenAiChat).expect("edge"));
        let mut chain = Chain::new(first, second);

        let frame = concat!(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"wired-ok"}}"#,
            "\n\n"
        );
        let mut out = chain.push_bytes(frame.as_bytes()).expect("push");
        out.push_str(&chain.finish().expect("finish"));
        assert!(out.contains("wired-ok"), "content dropped: {out}");
        assert!(
            out.contains("response.output_text.delta"),
            "missing Responses marker: {out}"
        );
        assert!(
            !out.contains("chat.completion.chunk"),
            "intermediate Chat frames leaked: {out}"
        );
    }

    /// Every EDGES capability has a constructible options variant, and none
    /// beyond the matrix; dispatch totality is already compiler-enforced.
    #[test]
    fn options_variants_cover_matrix() {
        let anth_req = anthropic_request_config();
        let anth_resp = anthropic_response_config();
        let resp_cfg = responses_router_config();
        let custom_tools = HashSet::new();
        let tool_namespaces = ToolNamespaceMap::new();

        let request_opts = |c, u| -> Option<RequestOptions<'_>> {
            match (c, u) {
                (OpenAiChat, Anthropic) => {
                    Some(RequestOptions::ChatToAnthropic { default_model: "m" })
                }
                (OpenAiChat, Gemini) => Some(RequestOptions::ChatToGemini { default_model: "m" }),
                (OpenAiChat, ResponsesApi) => Some(RequestOptions::ChatToResponses),
                (Anthropic, OpenAiChat) => Some(RequestOptions::AnthropicToChat(&anth_req)),
                (Gemini, OpenAiChat) => Some(RequestOptions::GeminiToChat {
                    model: "m",
                    requires_reasoning_content: false,
                    max_tokens_cap: None,
                }),
                (ResponsesApi, OpenAiChat) => Some(RequestOptions::ResponsesToChat(&resp_cfg)),
                (Anthropic, Gemini) => {
                    Some(RequestOptions::AnthropicToGemini { default_model: "m" })
                }
                (Gemini, Anthropic) => Some(RequestOptions::GeminiToAnthropic { model: "m" }),
                _ => None,
            }
        };
        let response_opts = |c, u| -> Option<ResponseOptions<'_>> {
            match (c, u) {
                (OpenAiChat, Anthropic) => Some(ResponseOptions::ChatToAnthropic { model: "m" }),
                (OpenAiChat, Gemini) => Some(ResponseOptions::ChatToGemini { model: "m" }),
                (OpenAiChat, ResponsesApi) => Some(ResponseOptions::ChatToResponses),
                (Anthropic, OpenAiChat) => Some(ResponseOptions::AnthropicToChat(&anth_resp)),
                (Gemini, OpenAiChat) => Some(ResponseOptions::GeminiToChat),
                (ResponsesApi, OpenAiChat) => Some(ResponseOptions::ResponsesToChat {
                    model: "m",
                    custom_tools: &custom_tools,
                    tool_namespaces: &tool_namespaces,
                }),
                (Anthropic, Gemini) => Some(ResponseOptions::GeminiToAnthropic { model: "m" }),
                (Gemini, Anthropic) => Some(ResponseOptions::AnthropicToGemini),
                _ => None,
            }
        };

        for c in WireFormat::ALL {
            for u in WireFormat::ALL {
                if c == u {
                    continue;
                }
                let caps = edge(c, u).map(|e| e.caps);
                assert_eq!(
                    request_opts(c, u).is_some(),
                    caps.map(|x| x.request).unwrap_or(false),
                    "request options coverage drifted for {c:?} -> {u:?}"
                );
                assert_eq!(
                    response_opts(c, u).is_some(),
                    caps.map(|x| x.response).unwrap_or(false),
                    "response options coverage drifted for {c:?} -> {u:?}"
                );
                assert_eq!(
                    stream_options_for(c, u).is_some(),
                    caps.map(|x| x.stream).unwrap_or(false),
                    "stream options coverage drifted for {c:?} -> {u:?}"
                );
                if let Some(o) = request_opts(c, u) {
                    assert_eq!(o.edge(), (c, u), "request options edge() mismapped");
                }
                if let Some(o) = response_opts(c, u) {
                    assert_eq!(o.edge(), (c, u), "response options edge() mismapped");
                }
            }
        }
    }

    #[test]
    fn translate_request_dispatches_all_edges() {
        let chat_body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });

        let anthropic = translate_request(
            &chat_body,
            &RequestOptions::ChatToAnthropic { default_model: "m" },
        );
        assert_eq!(anthropic["messages"][0]["role"], "user");

        let gemini = translate_request(
            &chat_body,
            &RequestOptions::ChatToGemini { default_model: "m" },
        );
        assert_eq!(gemini["contents"][0]["role"], "user");

        let responses = translate_request(&chat_body, &RequestOptions::ChatToResponses);
        assert!(responses.get("input").is_some());

        let anthropic_body = json!({
            "model": "m", "max_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let config = anthropic_request_config();
        let chat = translate_request(&anthropic_body, &RequestOptions::AnthropicToChat(&config));
        assert_eq!(chat["messages"][0]["role"], "user");

        let gemini_body = json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        });
        let chat = translate_request(
            &gemini_body,
            &RequestOptions::GeminiToChat {
                model: "m",
                requires_reasoning_content: false,
                max_tokens_cap: None,
            },
        );
        assert_eq!(chat["messages"][0]["role"], "user");

        let responses_body = json!({
            "model": "m",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}],
        });
        let config = responses_router_config();
        let chat = translate_request(&responses_body, &RequestOptions::ResponsesToChat(&config));
        assert_eq!(chat["messages"][0]["role"], "user");

        let gemini = translate_request(
            &anthropic_body,
            &RequestOptions::AnthropicToGemini { default_model: "m" },
        );
        assert_eq!(gemini["contents"][0]["role"], "user");
        assert_eq!(gemini["generationConfig"]["maxOutputTokens"], 128);

        let gemini_body = json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        });
        let anthropic = translate_request(
            &gemini_body,
            &RequestOptions::GeminiToAnthropic { model: "m" },
        );
        assert_eq!(anthropic["model"], "m");
        assert_eq!(anthropic["messages"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn translate_response_dispatches_all_edges() {
        let anthropic_resp = json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        });
        let chat = translate_response(
            &anthropic_resp,
            &ResponseOptions::ChatToAnthropic { model: "m" },
        )
        .unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hi");

        let gemini_resp = json!({
            "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
        });
        let chat = translate_response(&gemini_resp, &ResponseOptions::ChatToGemini { model: "m" })
            .unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hi");

        let responses_resp = json!({
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}]}],
        });
        let chat = translate_response(&responses_resp, &ResponseOptions::ChatToResponses).unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hi");

        let chat_resp = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop", "index": 0}],
        });
        let config = anthropic_response_config();
        let anthropic =
            translate_response(&chat_resp, &ResponseOptions::AnthropicToChat(&config)).unwrap();
        assert_eq!(anthropic["content"][0]["text"], "hi");

        let gemini = translate_response(&chat_resp, &ResponseOptions::GeminiToChat).unwrap();
        assert_eq!(gemini["candidates"][0]["content"]["parts"][0]["text"], "hi");

        let custom_tools = HashSet::new();
        let tool_namespaces = ToolNamespaceMap::new();
        let responses = translate_response(
            &chat_resp,
            &ResponseOptions::ResponsesToChat {
                model: "m",
                custom_tools: &custom_tools,
                tool_namespaces: &tool_namespaces,
            },
        )
        .unwrap();
        assert!(responses["output"].is_array(), "{responses}");
        assert!(responses.to_string().contains("hi"), "{responses}");

        let gemini_native = json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}},],
        });
        let anthropic = translate_response(
            &gemini_native,
            &ResponseOptions::GeminiToAnthropic { model: "m" },
        )
        .unwrap();
        assert_eq!(anthropic["type"], "message");
        assert_eq!(anthropic["content"][0]["text"], "hi");

        let anthropic_native = json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        });
        let gemini =
            translate_response(&anthropic_native, &ResponseOptions::AnthropicToGemini).unwrap();
        assert_eq!(gemini["candidates"][0]["content"]["parts"][0]["text"], "hi");
        assert_eq!(gemini["candidates"][0]["finishReason"], "STOP");
    }
}
