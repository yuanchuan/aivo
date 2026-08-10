use anyhow::Result;
use serde_json::Value;
use std::collections::HashSet;

use crate::services::http_utils::sse_data_payload;
use crate::services::responses_chat_conversion::{ResponsesStreamConverter, ToolNamespaceMap};
use crate::services::responses_to_chat_router::convert_chat_response_to_responses_sse;

pub(crate) fn convert_chat_response_to_responses_json(
    chat: &Value,
    original_model: &str,
    custom_tools: &HashSet<String>,
    tool_namespaces: &ToolNamespaceMap,
) -> Result<Value> {
    let sse = convert_chat_response_to_responses_sse(
        chat,
        false,
        original_model,
        custom_tools,
        tool_namespaces,
    );
    extract_completed_response_from_sse(&sse)
        .ok_or_else(|| anyhow::anyhow!("failed to synthesize responses JSON payload"))
}

pub(crate) fn convert_chat_sse_to_responses_sse(
    chat_sse: &str,
    original_model: &str,
    custom_tools: &HashSet<String>,
    tool_namespaces: &ToolNamespaceMap,
) -> Result<String> {
    let mut converter = ResponsesStreamConverter::new(original_model, false)
        .with_custom_tools(custom_tools.clone())
        .with_tool_namespaces(tool_namespaces.clone());
    let mut output = converter.push_bytes(chat_sse.as_bytes())?;
    output.push_str(&converter.finish());
    Ok(output)
}

fn extract_completed_response_from_sse(sse: &str) -> Option<Value> {
    let mut saw_completed_event = false;

    for line in sse.lines() {
        if let Some(event) = line.strip_prefix("event:") {
            saw_completed_event = event.trim() == "response.completed";
            continue;
        }

        if saw_completed_event && let Some(data) = sse_data_payload(line) {
            let payload: Value = serde_json::from_str(data).ok()?;
            return payload.get("response").cloned();
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        convert_chat_response_to_responses_json, convert_chat_sse_to_responses_sse,
        extract_completed_response_from_sse,
    };
    use crate::services::responses_chat_conversion::{ResponsesStreamConverter, ToolNamespaceMap};
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn test_convert_chat_response_to_responses_json_text() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 90
            },
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from responses"}
            }]
        });

        let response = convert_chat_response_to_responses_json(
            &chat,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "gpt-4o");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["usage"]["input_tokens"], 10);
        assert_eq!(response["usage"]["output_tokens"], 5);
        assert_eq!(response["usage"]["cache_read_input_tokens"], 90);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            90
        );
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "Hello from responses"
        );
    }

    #[test]
    fn test_convert_chat_response_to_responses_json_tool_call() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let response = convert_chat_response_to_responses_json(
            &chat,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "call_123");
        assert_eq!(response["output"][0]["name"], "shell");
    }

    #[test]
    fn test_push_bytes_survives_utf8_split_across_chunks() {
        // "你好" split mid-character: each chunk alone is invalid UTF-8, so a
        // per-chunk lossy decode would emit U+FFFD instead of the character.
        let line =
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"},\"finish_reason\":null}]}\n";
        let bytes = line.as_bytes();
        let split = line.find("你好").unwrap() + 2; // mid-way through 你 (3 bytes)

        let mut converter = ResponsesStreamConverter::new("gpt-4o", false);
        let mut out = converter.push_bytes(&bytes[..split]).unwrap();
        out.push_str(&converter.push_bytes(&bytes[split..]).unwrap());
        out.push_str(&converter.push_bytes(b"data: [DONE]\n").unwrap());
        out.push_str(&converter.finish());

        assert!(out.contains("你好"), "{out}");
        assert!(!out.contains('\u{FFFD}'), "{out}");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_text() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"cache_read_input_tokens\":90}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();

        assert!(responses_sse.contains("event: response.created"));
        assert!(responses_sse.contains("event: response.output_text.delta"));
        assert!(responses_sse.contains("\"delta\":\"Hel\""));
        assert!(responses_sse.contains("\"delta\":\"lo\""));
        assert!(responses_sse.contains("\"cache_read_input_tokens\":90"));
        assert!(responses_sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_accepts_input_output_usage_aliases() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "grok-4.3",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();

        assert_eq!(response["usage"]["input_tokens"], 15000);
        assert_eq!(response["usage"]["output_tokens"], 42);
        assert_eq!(response["usage"]["total_tokens"], 15042);
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_waits_for_trailing_usage_after_finish_reason() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "grok-4.3",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();

        assert_eq!(response["usage"]["input_tokens"], 15000);
        assert_eq!(response["usage"]["output_tokens"], 42);
        assert_eq!(response["output"][0]["content"][0]["text"], "hi!");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_tool_call() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();

        assert!(responses_sse.contains("event: response.output_item.added"));
        assert!(responses_sse.contains("event: response.function_call_arguments.delta"));
        assert!(responses_sse.contains("\"call_id\":\"call_abc\""));
        assert!(responses_sse.contains("\"delta\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(responses_sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_emits_reasoning_events() {
        // Serve's old converter dropped delta.reasoning_content entirely;
        // the unified converter must surface it as reasoning summary events.
        let chat_sse = concat!(
            "data: {\"id\":\"c\",\"model\":\"deepseek-reasoner\",\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c\",\"model\":\"deepseek-reasoner\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "deepseek-reasoner",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();

        assert!(responses_sse.contains("event: response.reasoning_summary_text.delta"));
        assert!(responses_sse.contains("\"delta\":\"thinking...\""));
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(response["output"][1]["type"], "message");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_length_truncation_surfaces_incomplete() {
        let chat_sse = concat!(
            "data: {\"id\":\"c\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"cut off\"},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(
            chat_sse,
            "gpt-4o",
            &HashSet::new(),
            &ToolNamespaceMap::new(),
        )
        .unwrap();
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();

        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }
}
