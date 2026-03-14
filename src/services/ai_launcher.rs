//! AILauncher service for spawning AI tool processes.
//! Handles process spawning with environment injection and stdio passthrough.

use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;
#[cfg(unix)]
use tokio::signal;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::codex_model_map::map_model_for_codex_cli;
use crate::services::environment_injector::EnvironmentInjector;
use crate::services::models_cache::ModelsCache;
use crate::services::provider_profile::{
    is_copilot_base, is_direct_openai_base, provider_profile_for_base_url,
};
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, OpenAICompatibilityMode, SessionStore,
};

/// Supported AI tool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIToolType {
    Claude,
    Codex,
    Gemini,
    Opencode,
}

impl AIToolType {
    /// Parses a string into an AIToolType
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "gemini" => Some(Self::Gemini),
            "opencode" => Some(Self::Opencode),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
            Self::Opencode => "opencode",
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Claude, Self::Codex, Self::Gemini, Self::Opencode]
    }
}

/// Launch options for AI tools
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub tool: AIToolType,
    pub args: Vec<String>,
    pub debug: bool,
    pub model: Option<String>,
    pub env: Option<HashMap<String, String>>,
    /// Temporary key override for this launch (does not persist to config)
    pub key_override: Option<ApiKey>,
}

/// Tool configuration including command and environment variables
#[derive(Debug, Clone)]
pub struct ToolConfig {
    pub command: String,
    pub env_vars: HashMap<String, String>,
}

/// AILauncher spawns AI tool processes with configured environment and stdio passthrough
#[derive(Debug, Clone)]
pub struct AILauncher {
    session_store: SessionStore,
    env_injector: EnvironmentInjector,
    cache: ModelsCache,
}

impl AILauncher {
    /// Creates a new AILauncher
    pub fn new(
        session_store: SessionStore,
        env_injector: EnvironmentInjector,
        cache: ModelsCache,
    ) -> Self {
        Self {
            session_store,
            env_injector,
            cache,
        }
    }

    /// Spawns an AI tool with configured environment and stdio passthrough
    pub async fn launch(&self, options: &LaunchOptions) -> Result<i32> {
        let mut key = match &options.key_override {
            Some(k) => k.clone(),
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    return Err(CLIError::new(
                        "No API key configured. Please add a key with 'aivo keys add'.",
                        ErrorCategory::Auth,
                        None::<String>,
                        Some("Run 'aivo keys add' to add an API key"),
                    )
                    .into());
                }
            },
        };

        if options.tool == AIToolType::Claude {
            key = self
                .resolve_claude_protocol(
                    key,
                    options.key_override.is_none(),
                    options.model.as_deref(),
                )
                .await?;
        } else if options.tool == AIToolType::Codex {
            key = self
                .resolve_codex_mode(key, options.key_override.is_none())
                .await?;
        } else if options.tool == AIToolType::Gemini {
            key = self
                .resolve_gemini_protocol(key, options.key_override.is_none())
                .await?;
        } else if options.tool == AIToolType::Opencode {
            key = self
                .resolve_opencode_mode(key, options.key_override.is_none())
                .await?;
        }

        self.output_key_info(&key);

        let (model, opencode_models) = if options.tool == AIToolType::Opencode {
            let (selected_model, discovered_models) = self
                .resolve_opencode_model_config(&key, options.model.as_deref())
                .await?;
            (selected_model, Some(discovered_models))
        } else {
            (options.model.clone(), None)
        };
        let tool_config = self.get_tool_config(
            options.tool,
            &key,
            model.as_deref(),
            opencode_models.as_deref(),
        );

        let mut env =
            self.env_injector
                .merge(&tool_config.env_vars, options.env.as_ref(), options.debug);

        // Start AnthropicRouter for OpenRouter + Claude, update ANTHROPIC_BASE_URL with actual port
        if options.tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
            let port = start_anthropic_router(&env).await?;
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start OpenAI router for OpenAI-compatible providers (Cloudflare, etc.), update ANTHROPIC_BASE_URL
        if options.tool == AIToolType::Claude && env.contains_key("AIVO_USE_OPENAI_ROUTER") {
            let port = start_openai_router(&env).await?;
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start CopilotRouter for GitHub Copilot, update ANTHROPIC_BASE_URL with actual port
        if options.tool == AIToolType::Claude && env.contains_key("AIVO_USE_COPILOT_ROUTER") {
            let port = start_copilot_router(&env).await?;
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start CodexRouter for non-OpenAI providers, update OPENAI_BASE_URL with actual port
        if options.tool == AIToolType::Codex && env.contains_key("AIVO_USE_CODEX_ROUTER") {
            let port = start_codex_router(&env).await?;
            env.insert(
                "OPENAI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start CodexRouter with CopilotTokenManager for GitHub Copilot, update OPENAI_BASE_URL
        if options.tool == AIToolType::Codex && env.contains_key("AIVO_USE_CODEX_COPILOT_ROUTER") {
            let port = start_codex_copilot_router(&env).await?;
            env.insert(
                "OPENAI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start GeminiRouter for non-Google providers, update GOOGLE_GEMINI_BASE_URL with actual port
        if options.tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
            let port = start_gemini_router(&env).await?;
            env.insert(
                "GOOGLE_GEMINI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start GeminiRouter (Copilot mode), update GOOGLE_GEMINI_BASE_URL with actual port
        if options.tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER")
        {
            let port = start_gemini_copilot_router(&env).await?;
            env.insert(
                "GOOGLE_GEMINI_BASE_URL".to_string(),
                format!("http://127.0.0.1:{}", port),
            );
        }

        // Start CodexRouter (Copilot mode) for OpenCode, patch OPENCODE_CONFIG_CONTENT with real port
        if options.tool == AIToolType::Opencode
            && env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER")
        {
            let port = start_codex_copilot_router(&env).await?;
            let real_url = format!("http://127.0.0.1:{}", port);
            if let Some(content) = env.get("OPENCODE_CONFIG_CONTENT").cloned() {
                let patched = content.replace("http://127.0.0.1:0", &real_url);
                env.insert("OPENCODE_CONFIG_CONTENT".to_string(), patched);
            }
        }

        // Start CodexRouter for OpenCode when the provider needs compatibility routing
        if options.tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_ROUTER") {
            let port = start_codex_router(&env).await?;
            let real_url = format!("http://127.0.0.1:{}", port);
            if let Some(content) = env.get("OPENCODE_CONFIG_CONTENT").cloned() {
                let patched = content.replace("http://127.0.0.1:0", &real_url);
                env.insert("OPENCODE_CONFIG_CONTENT".to_string(), patched);
            }
        }

        // For Claude, inject --teammate-mode in-process to run in single window
        let args = inject_claude_teammate_mode(options.tool, &options.args);

        // For Codex with routed non-OpenAI providers, optionally provide a local model catalog
        // entry for custom models to avoid fallback metadata mode.
        let use_codex_router = env.contains_key("AIVO_USE_CODEX_ROUTER")
            || env.contains_key("AIVO_USE_CODEX_COPILOT_ROUTER");
        let codex_model_catalog_path = if options.tool == AIToolType::Codex {
            maybe_write_codex_model_catalog(model.as_deref(), use_codex_router).await?
        } else {
            None
        };

        // For Codex, inject -m <model> if model is specified via --model flag
        let args = if options.tool == AIToolType::Codex {
            let args = inject_codex_model(model.as_deref(), &args, use_codex_router);
            inject_codex_model_catalog(codex_model_catalog_path.as_deref(), &args)
        } else {
            args
        };

        let mut child = self.spawn_child(&tool_config.command, &args, env)?;

        let _ = self
            .session_store
            .record_selection(&key.id, options.tool.as_str(), model.as_deref())
            .await;
        if let Some(cwd) = crate::services::system_env::current_dir_string() {
            let _ = self
                .session_store
                .set_directory_start(
                    &cwd,
                    &key.id,
                    &key.base_url,
                    options.tool.as_str(),
                    model.as_deref(),
                )
                .await;
        }

        let result = self.wait_for_process(&mut child).await;

        if let Some(ref path) = codex_model_catalog_path {
            let _ = tokio::fs::remove_file(path).await;
        }

        result
    }

    /// Outputs information about which key is being used
    fn output_key_info(&self, key: &ApiKey) {
        use crate::style;

        eprintln!(
            "  {} Using key: {} {}",
            style::success_symbol(),
            style::cyan(key.display_name()),
            style::dim(format!("({})", key.base_url))
        );
    }

    async fn resolve_claude_protocol(
        &self,
        mut key: ApiKey,
        _persist: bool,
        _model: Option<&str>,
    ) -> Result<ApiKey> {
        let profile = provider_profile_for_base_url(&key.base_url);
        if profile.serve_flags.is_copilot || profile.serve_flags.is_openrouter {
            return Ok(key);
        }
        if key.claude_protocol.is_none() {
            key.claude_protocol = Some(preferred_claude_protocol(&key.base_url));
        }
        Ok(key)
    }

    async fn resolve_codex_mode(&self, mut key: ApiKey, _persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        if key.codex_mode.is_none() {
            key.codex_mode = Some(preferred_codex_mode(&key.base_url));
        }
        Ok(key)
    }

    async fn resolve_gemini_protocol(&self, mut key: ApiKey, _persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        if key.gemini_protocol.is_none() {
            key.gemini_protocol = Some(preferred_gemini_protocol(&key.base_url));
        }
        Ok(key)
    }

    async fn resolve_opencode_mode(&self, mut key: ApiKey, _persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        if key.opencode_mode.is_none() {
            key.opencode_mode = Some(preferred_opencode_mode(&key.base_url));
        }
        Ok(key)
    }

    async fn resolve_opencode_model_config(
        &self,
        key: &ApiKey,
        model: Option<&str>,
    ) -> Result<(Option<String>, Vec<String>)> {
        let requested_model = model.map(|m| m.strip_prefix("aivo/").unwrap_or(m).to_string());
        let client = Client::new();

        // Check cache first — skip the spinner if we get a hit
        let fetch_result = if let Some(cached) = self.cache.get(&key.base_url).await {
            Ok(cached)
        } else {
            // Cache miss: show spinner while fetching from network
            let (spinning, spinner_handle) =
                crate::style::start_spinner(Some(" Fetching models..."));

            // bypass_cache=true: we know it's a miss; fetch_models_cached will still write result to cache
            let result =
                crate::commands::models::fetch_models_cached(&client, key, &self.cache, true).await;

            crate::style::stop_spinner(&spinning);
            let _ = spinner_handle.await;

            result
        };

        let mut models = match fetch_result {
            Ok(models) => models,
            Err(e) => {
                if let Some(requested_model) = requested_model.clone() {
                    return Ok((Some(requested_model.clone()), vec![requested_model]));
                }
                return Err(e).with_context(|| {
                    "Unable to determine an OpenCode model from your provider. Pass --model <provider/model>."
                });
            }
        };
        if let Some(requested_model) = requested_model {
            if !models.contains(&requested_model) {
                models.push(requested_model.clone());
            }
            models.sort();
            models.dedup();
            return Ok((Some(requested_model), models));
        }

        models.sort();
        models.dedup();

        let selected_model = models
            .iter()
            .find(|m| m.contains("claude") && m.contains("sonnet"))
            .or_else(|| models.first())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No models returned by provider. Pass --model <provider/model> for opencode."
                )
            })?;
        Ok((Some(selected_model), models))
    }

    /// Gets tool-specific configuration including command and environment variables
    fn get_tool_config(
        &self,
        tool: AIToolType,
        key: &ApiKey,
        model: Option<&str>,
        opencode_models: Option<&[String]>,
    ) -> ToolConfig {
        let env_vars = match tool {
            AIToolType::Claude => self.env_injector.for_claude(key, model),
            AIToolType::Codex => self.env_injector.for_codex(key, model),
            AIToolType::Gemini => self.env_injector.for_gemini(key, model),
            AIToolType::Opencode => self.env_injector.for_opencode(key, model, opencode_models),
        };

        ToolConfig {
            command: tool.as_str().to_string(),
            env_vars,
        }
    }

    /// Spawns a child process with stdio inheritance and returns its exit code
    fn spawn_child(
        &self,
        command: &str,
        args: &[String],
        env: HashMap<String, String>,
    ) -> Result<tokio::process::Child> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(&env)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn {}", command))?;
        Ok(child)
    }

    /// Waits for a child process while forwarding signals on Unix.
    #[cfg(unix)]
    async fn wait_for_process(&self, child: &mut tokio::process::Child) -> Result<i32> {
        // Get the child PID for signal forwarding
        let child_id = child.id();

        // Set up signal forwarding
        let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;

        // Wait for the child to complete, while also listening for signals
        let result = tokio::select! {
            status = child.wait() => {
                status.map(|s| s.code().unwrap_or(1))
            }
            _ = sigint.recv() => {
                // Forward SIGINT to child
                if let Some(id) = child_id {
                    // SAFETY: `kill` does not dereference pointers; pid/signal values are plain integers.
                    let _ = unsafe { libc::kill(id as i32, libc::SIGINT) };
                }
                child.wait().await.map(|s| s.code().unwrap_or(130)) // 128 + SIGINT (2)
            }
            _ = sigterm.recv() => {
                // Forward SIGTERM to child
                if let Some(id) = child_id {
                    // SAFETY: `kill` does not dereference pointers; pid/signal values are plain integers.
                    let _ = unsafe { libc::kill(id as i32, libc::SIGTERM) };
                }
                child.wait().await.map(|s| s.code().unwrap_or(143)) // 128 + SIGTERM (15)
            }
        };

        result.map_err(|e| e.into())
    }

    /// Waits for a child process and returns its exit code (non-Unix)
    #[cfg(not(unix))]
    async fn wait_for_process(&self, child: &mut tokio::process::Child) -> Result<i32> {
        let status = child.wait().await?;
        Ok(status.code().unwrap_or(1))
    }
}

fn preferred_claude_protocol(base_url: &str) -> ClaudeProviderProtocol {
    match provider_profile_for_base_url(base_url).default_protocol {
        ProviderProtocol::Anthropic => ClaudeProviderProtocol::Anthropic,
        ProviderProtocol::Google => ClaudeProviderProtocol::Google,
        ProviderProtocol::Openai => ClaudeProviderProtocol::Openai,
    }
}

fn preferred_codex_mode(base_url: &str) -> OpenAICompatibilityMode {
    if is_direct_openai_base(base_url) {
        OpenAICompatibilityMode::Direct
    } else {
        OpenAICompatibilityMode::Router
    }
}

fn preferred_gemini_protocol(base_url: &str) -> GeminiProviderProtocol {
    match provider_profile_for_base_url(base_url).default_protocol {
        ProviderProtocol::Google => GeminiProviderProtocol::Google,
        ProviderProtocol::Anthropic => GeminiProviderProtocol::Anthropic,
        ProviderProtocol::Openai => GeminiProviderProtocol::Openai,
    }
}

fn preferred_opencode_mode(base_url: &str) -> OpenAICompatibilityMode {
    if provider_profile_for_base_url(base_url).default_protocol == ProviderProtocol::Openai {
        OpenAICompatibilityMode::Direct
    } else {
        OpenAICompatibilityMode::Router
    }
}

/// Starts the built-in AnthropicRouter and returns the port it bound to
async fn start_anthropic_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{AnthropicRouter, AnthropicRouterConfig};

    let api_key = env
        .get("AIVO_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_BASE_URL"))?
        .clone();

    let config = AnthropicRouterConfig {
        upstream_base_url: base_url,
        upstream_api_key: api_key,
    };

    let router = AnthropicRouter::new(config);
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in OpenAI router for OpenAI-compatible providers (Cloudflare, etc.)
/// Returns the port it bound to
async fn start_openai_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{OpenAIRouter, OpenAIRouterConfig};

    let api_key = env
        .get("AIVO_OPENAI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_OPENAI_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_OPENAI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_OPENAI_ROUTER_BASE_URL"))?
        .clone();

    let model_prefix = env.get("AIVO_OPENAI_ROUTER_MODEL_PREFIX").cloned();
    let requires_reasoning_content = env
        .get("AIVO_OPENAI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_OPENAI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_OPENAI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let config = OpenAIRouterConfig {
        target_base_url: base_url,
        target_api_key: api_key,
        target_protocol,
        model_prefix,
        requires_reasoning_content,
        max_tokens_cap,
    };

    let router = OpenAIRouter::new(config);
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: openai router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in CodexRouter for non-OpenAI providers and returns the port it bound to
async fn start_codex_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CodexRouter, CodexRouterConfig};

    let api_key = env
        .get("AIVO_CODEX_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_CODEX_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_CODEX_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_CODEX_ROUTER_BASE_URL"))?
        .clone();

    let model_prefix = env.get("AIVO_CODEX_ROUTER_MODEL_PREFIX").cloned();
    let requires_reasoning_content = env
        .get("AIVO_CODEX_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let actual_model = env.get("AIVO_CODEX_ROUTER_ACTUAL_MODEL").cloned();
    let max_tokens_cap = env
        .get("AIVO_CODEX_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_CODEX_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));

    let router = CodexRouter::new(CodexRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        copilot_token_manager: None,
        model_prefix,
        requires_reasoning_content,
        actual_model,
        max_tokens_cap,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: codex router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in GeminiRouter for non-Google providers and returns the port it bound to
async fn start_gemini_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let api_key = env
        .get("AIVO_GEMINI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_GEMINI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_BASE_URL"))?
        .clone();

    let requires_reasoning_content = env
        .get("AIVO_GEMINI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_GEMINI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let upstream_protocol = env
        .get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: base_url,
        api_key,
        upstream_protocol,
        forced_model: None,
        copilot_token_manager: None,
        requires_reasoning_content,
        max_tokens_cap,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts a GeminiRouter configured for GitHub Copilot and returns the port it bound to
async fn start_gemini_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{GeminiRouter, GeminiRouterConfig};
    use std::sync::Arc;

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let forced_model = env.get("AIVO_GEMINI_COPILOT_FORCED_MODEL").cloned();

    if forced_model.is_none() {
        eprintln!(
            "  {} Gemini + Copilot: no model specified. Gemini models are not available on \
             Copilot. Pass --model <model> (e.g., --model gpt-4o).",
            crate::style::yellow("Warning:")
        );
    }

    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        upstream_protocol: ProviderProtocol::Openai,
        forced_model,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        requires_reasoning_content: false,
        max_tokens_cap: None,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts the built-in CopilotRouter for GitHub Copilot and returns the port it bound to
async fn start_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CopilotRouter, CopilotRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = CopilotRouter::new(CopilotRouterConfig { github_token });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Starts a CodexRouter configured for GitHub Copilot and returns the port it bound to
async fn start_codex_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{CodexRouter, CodexRouterConfig};
    use std::sync::Arc;

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = CodexRouter::new(CodexRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        target_protocol: ProviderProtocol::Openai,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: codex copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Injects `--teammate-mode in-process` for Claude if not already specified by the user.
/// This ensures Claude runs in a single window instead of using split panels.
fn inject_claude_teammate_mode(tool: AIToolType, args: &[String]) -> Vec<String> {
    if tool != AIToolType::Claude {
        return args.to_vec();
    }

    // Check if the user already specified --teammate-mode
    let has_teammate_mode = args
        .iter()
        .any(|a| a == "--teammate-mode" || a.starts_with("--teammate-mode="));
    if has_teammate_mode {
        return args.to_vec();
    }

    let mut new_args = vec!["--teammate-mode".to_string(), "in-process".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

/// Injects `-m <model>` for Codex if not already specified by the user.
/// Codex CLI requires the model to be passed as a CLI argument, not via env vars.
/// When using a router, passes the original model name (catalog provides metadata).
/// For direct OpenAI connections, maps to a known OpenAI model so Codex CLI finds metadata.
fn inject_codex_model(model: Option<&str>, args: &[String], use_router: bool) -> Vec<String> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return args.to_vec(),
    };

    // Check if the user already specified --model or -m
    let has_model_flag = args
        .iter()
        .any(|a| a == "--model" || a == "-m" || a.starts_with("--model=") || a.starts_with("-m="));
    if has_model_flag {
        return args.to_vec();
    }

    // When using a router, pass the original model name so it matches the catalog entry.
    // For direct OpenAI, map to a known model so Codex CLI finds built-in metadata.
    let codex_model = if use_router {
        model.to_string()
    } else {
        map_model_for_codex_cli(model)
    };
    let mut new_args = vec!["-m".to_string(), codex_model];
    new_args.extend_from_slice(args);
    new_args
}

/// Injects `--config model_catalog_json="<path>"` for Codex unless already provided.
fn inject_codex_model_catalog(path: Option<&str>, args: &[String]) -> Vec<String> {
    let path = match path {
        Some(p) if !p.is_empty() => p,
        _ => return args.to_vec(),
    };

    // Respect user-specified model_catalog_json settings.
    if args.iter().any(|a| a.contains("model_catalog_json")) {
        return args.to_vec();
    }

    let escaped_path = path.replace('\\', "\\\\").replace('"', "\\\"");
    let mut new_args = vec![
        "--config".to_string(),
        format!("model_catalog_json=\"{}\"", escaped_path),
    ];
    new_args.extend_from_slice(args);
    new_args
}

/// Creates a minimal custom Codex model catalog for namespaced non-OpenAI models.
async fn maybe_write_codex_model_catalog(
    model: Option<&str>,
    uses_non_openai_router: bool,
) -> Result<Option<String>> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return Ok(None),
    };

    // Don't write catalog if not using a router (OpenAI official)
    if !uses_non_openai_router {
        return Ok(None);
    }

    // Don't write catalog for standard OpenAI models - they exist in Codex's built-in catalog
    let model_lower = model.to_lowercase();
    let name_only = model_lower.split('/').next_back().unwrap_or(&model_lower);
    if name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
    {
        return Ok(None);
    }

    let catalog_json = build_codex_model_catalog_json(model)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = format!(
        "aivo-codex-model-catalog-{}-{}.json",
        std::process::id(),
        nonce
    );
    let path = std::env::temp_dir().join(file_name);

    tokio::fs::write(&path, catalog_json)
        .await
        .with_context(|| {
            format!(
                "Failed to write Codex model catalog override at {}",
                path.display()
            )
        })?;

    Ok(Some(path.to_string_lossy().to_string()))
}

fn build_codex_model_catalog_json(model: &str) -> Result<String> {
    // Compatible with Codex's ModelsResponse/ModelInfo shape used in tests.
    let catalog = json!({
        "models": [{
            "slug": model,
            "display_name": model,
            "description": format!("Custom model metadata for {}", model),
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "minimal_client_version": [0, 1, 0],
            "supported_in_api": true,
            "priority": 0,
            "upgrade": serde_json::Value::Null,
            "base_instructions": "base instructions",
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": serde_json::Value::Null,
            "apply_patch_tool_type": serde_json::Value::Null,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": false,
            "supports_image_detail_original": false,
            "context_window": 272000,
            "experimental_supported_tools": []
        }]
    });
    Ok(serde_json::to_string(&catalog)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_tool_type_from_str() {
        assert_eq!(AIToolType::parse("claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("Claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("CLAUDE"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("codex"), Some(AIToolType::Codex));
        assert_eq!(AIToolType::parse("gemini"), Some(AIToolType::Gemini));
        assert_eq!(AIToolType::parse("opencode"), Some(AIToolType::Opencode));
        assert_eq!(AIToolType::parse("unknown"), None);
    }

    #[test]
    fn test_inject_claude_teammate_mode_for_claude() {
        let args = vec!["--verbose".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(
            result,
            vec!["--teammate-mode", "in-process", "--verbose", "prompt"]
        );
    }

    #[test]
    fn test_inject_claude_teammate_mode_skips_non_claude() {
        let args = vec!["--verbose".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Codex, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Gemini, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Opencode, &args);
        assert_eq!(result, vec!["--verbose"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_respects_user_flag() {
        // User specified --teammate-mode explicitly
        let args = vec![
            "--teammate-mode".to_string(),
            "split".to_string(),
            "prompt".to_string(),
        ];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "split", "prompt"]);

        // User specified --teammate-mode=value format
        let args = vec!["--teammate-mode=split".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode=split", "prompt"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_empty_args() {
        let args: Vec<String> = vec![];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "in-process"]);
    }

    // Tests for inject_codex_model

    #[test]
    fn test_inject_codex_model_injects_when_provided() {
        let model = Some("o4-mini");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "o4-mini", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_original() {
        let model = Some("kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_namespaced() {
        let model = Some("moonshot/kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "moonshot/kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_when_already_specified() {
        let model = Some("o4-mini");
        let args = vec![
            "--model".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        // Should NOT inject since user already specified --model
        assert_eq!(result, vec!["--model", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_shorthand_flag() {
        let model = Some("o4-mini");
        let args = vec![
            "-m".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_equals_format() {
        let model = Some("o4-mini");
        let args = vec!["--model=gpt-4o".to_string(), "file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["--model=gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_empty_model() {
        let model = Some("");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_none_model() {
        let model: Option<&str> = None;
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_catalog_injects_when_path_provided() {
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(
            result,
            vec![
                "--config",
                "model_catalog_json=\"/tmp/catalog.json\"",
                "file.ts"
            ]
        );
    }

    #[test]
    fn test_inject_codex_model_catalog_skips_when_existing_setting_present() {
        let args = vec![
            "--config".to_string(),
            "model_catalog_json=\"/tmp/custom.json\"".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(result, args);
    }

    #[test]
    fn test_build_codex_model_catalog_json_includes_model_slug() {
        let model = "minimax/minimax-m2.5";
        let json = build_codex_model_catalog_json(model).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["models"][0]["slug"], model);
        assert_eq!(parsed["models"][0]["display_name"], model);
    }

    #[test]
    fn test_preferred_claude_protocol_for_anthropic_urls() {
        assert_eq!(
            preferred_claude_protocol("https://api.anthropic.com/v1"),
            ClaudeProviderProtocol::Anthropic
        );
        assert_eq!(
            preferred_claude_protocol("https://api.minimax.io/anthropic/v1"),
            ClaudeProviderProtocol::Anthropic
        );
    }

    #[test]
    fn test_preferred_claude_protocol_for_openai_compatible_urls() {
        assert_eq!(
            preferred_claude_protocol("https://api.openai.com/v1"),
            ClaudeProviderProtocol::Openai
        );
        assert_eq!(
            preferred_claude_protocol("https://ai-gateway.vercel.sh/v1"),
            ClaudeProviderProtocol::Openai
        );
        assert_eq!(
            preferred_claude_protocol("https://example.com/openai"),
            ClaudeProviderProtocol::Openai
        );
    }

    #[test]
    fn test_preferred_codex_mode() {
        assert_eq!(
            preferred_codex_mode("https://api.openai.com/v1"),
            OpenAICompatibilityMode::Direct
        );
        assert_eq!(
            preferred_codex_mode("https://openrouter.ai/api/v1"),
            OpenAICompatibilityMode::Router
        );
    }

    #[test]
    fn test_preferred_gemini_protocol() {
        assert_eq!(
            preferred_gemini_protocol("https://generativelanguage.googleapis.com/v1beta"),
            GeminiProviderProtocol::Google
        );
        assert_eq!(
            preferred_gemini_protocol("https://api.openai.com/v1"),
            GeminiProviderProtocol::Openai
        );
    }

    #[test]
    fn test_preferred_opencode_mode() {
        assert_eq!(
            preferred_opencode_mode("https://api.openai.com/v1"),
            OpenAICompatibilityMode::Direct
        );
        assert_eq!(
            preferred_opencode_mode("https://openrouter.ai/api/v1"),
            OpenAICompatibilityMode::Direct
        );
    }
}
