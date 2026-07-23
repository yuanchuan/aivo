use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::codex_home_shadow::{AuthDotJson, CodexHomeShadow, tokens_changed};
use crate::services::codex_oauth::{CodexOAuthCredential, REFRESH_SKEW_SECS, ensure_fresh};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::route_cache::{PersistedRoute, RouteCache};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::symlink_util::{symlink_dir, symlink_file};

/// Holds the shadow `CODEX_HOME` dir + metadata needed to sync refreshed
/// tokens back into aivo's store after codex exits.
pub(crate) struct CodexOAuthSync {
    pub(crate) key_id: String,
    pub(crate) shadow: CodexHomeShadow,
    pub(crate) original: CodexOAuthCredential,
}

pub(crate) struct LaunchRuntimeState {
    pub(crate) env: HashMap<String, String>,
    /// Env vars to `env_remove` from the child (vs setting via `env`).
    /// Populated by `prepare_runtime_env` from the
    /// `_AIVO_INTERNAL_ENV_UNSET` carrier emitted by the injector for the
    /// Claude OAuth path — see `environment_injector::AIVO_INTERNAL_ENV_UNSET`.
    pub(crate) env_unset: Vec<String>,
    /// The started router's per-(model) route cache, if any. After the child
    /// exits, `persist_runtime_discoveries` reads `dirty_routes()` and merges
    /// the confirmed routes back into the key under `cache.tool()`.
    pub(crate) route_cache: Option<Arc<RouteCache>>,
    /// Set to `true` by a router after observing a `reasoning_content` semantic
    /// rejection from the upstream. Persisted to the key so subsequent launches
    /// inject `_REQUIRE_REASONING=1` without needing the host in the static
    /// substring list in `ProviderQuirks::for_base_url`.
    pub(crate) learned_requires_reasoning: Option<Arc<AtomicBool>>,
    pub(crate) pi_agent_dir: Option<String>,
    pub(crate) codex_oauth_sync: Option<CodexOAuthSync>,
    /// Holds the temp dir that backs `GEMINI_CLI_SYSTEM_SETTINGS_PATH`
    /// for non-OAuth gemini launches. Dropping it deletes the settings
    /// override file; must outlive the spawned gemini process.
    #[allow(dead_code)] // kept alive solely for its Drop impl
    pub(crate) gemini_system_settings: Option<tempfile::TempDir>,
}

pub(crate) async fn prepare_runtime_env(
    tool: AIToolType,
    mut env: HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<LaunchRuntimeState> {
    let mut route_cache: Option<Arc<RouteCache>> = None;
    let mut learned_requires_reasoning: Option<Arc<AtomicBool>> = None;

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
        let port = start_anthropic_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER") {
        let router_key = env
            .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY")
            .cloned()
            .unwrap_or_default();
        if env.contains_key("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_TIERS_JSON") {
            let port = start_multi_upstream_serve_router(&env, session_store).await?;
            set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
        } else if let Some(port) =
            start_provider_oauth_router(&env, router_key, session_store).await?
        {
            set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
        } else {
            let (port, cache, learned) = start_anthropic_to_openai_router(&env).await?;
            route_cache = Some(cache);
            learned_requires_reasoning = Some(learned);
            set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
        }
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_COPILOT_ROUTER") {
        let port = start_copilot_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if env.contains_key("AIVO_USE_CURSOR_ROUTER") {
        let port = start_cursor_router(&mut env, tool).await?;
        if tool == AIToolType::Pi && env.contains_key("AIVO_PI_MODELS_JSON") {
            // Pi reads its upstream URL from a JSON file, not an env var, so
            // patch the placeholder in AIVO_PI_MODELS_JSON before writing the
            // temp agent dir.
            write_pi_agent_dir(&mut env, Some(port)).await?;
        } else if tool == AIToolType::Opencode && env.contains_key("OPENCODE_CONFIG_CONTENT") {
            // OpenCode reads its upstream from OPENCODE_CONFIG_CONTENT JSON;
            // swap the placeholder URL for the bound cursor-router port.
            patch_opencode_config_content(&mut env, port);
        } else {
            let base_url_env =
                env.remove("AIVO_CURSOR_BASE_URL_ENV")
                    .unwrap_or_else(|| match tool {
                        tool if tool.is_codex_family() => "OPENAI_BASE_URL".to_string(),
                        _ => "ANTHROPIC_BASE_URL".to_string(),
                    });
            set_local_base_url(&mut env, &base_url_env, port);
        }
        // Note: AIVO_USE_CURSOR_ROUTER is left in `env` here on purpose so
        // `build_runtime_args` (called next, after `prepare_runtime_env`
        // returns) can detect cursor mode and route codex through the
        // non-OpenAI catalog/model-passthrough path. The marker is stripped
        // later in `ai_launcher` right before spawn. `AIVO_CURSOR_KEY_SECRET`
        // is already consumed by `start_cursor_router` itself.
    }

    if tool.is_codex_family() && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER") {
        let router_key = env
            .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
            .cloned()
            .unwrap_or_default();
        if let Some(port) = start_provider_oauth_router(&env, router_key, session_store).await? {
            set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
        } else {
            let (port, cache, learned) = start_responses_to_chat_router("codex", &env).await?;
            route_cache = Some(cache);
            learned_requires_reasoning = Some(learned);
            set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
        }
    }

    if tool.is_codex_family() && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
        // Provider OAuth rides the universal ServeRouter (handles Gemini
        // `generateContent` inbound); else the static gemini router.
        let router_key = env
            .get("AIVO_GEMINI_ROUTER_API_KEY")
            .cloned()
            .unwrap_or_default();
        if let Some(port) = start_provider_oauth_router(&env, router_key, session_store).await? {
            set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
        } else {
            let (port, cache, learned) = start_gemini_router(&env).await?;
            route_cache = Some(cache);
            learned_requires_reasoning = Some(learned);
            set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
        }
        clear_node_proxy_env(&mut env);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER") {
        let port = start_gemini_copilot_router(&env).await?;
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
        clear_node_proxy_env(&mut env);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_ROUTER") {
        let router_key = env
            .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
            .cloned()
            .unwrap_or_default();
        if let Some(port) = start_provider_oauth_router(&env, router_key, session_store).await? {
            patch_opencode_config_content(&mut env, port);
        } else {
            let (port, cache, learned) = start_responses_to_chat_router("opencode", &env).await?;
            route_cache = Some(cache);
            learned_requires_reasoning = Some(learned);
            patch_opencode_config_content(&mut env, port);
        }
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_SETUP_PI_AGENT_DIR") {
        // Direct connection — no router needed, just write the temp agent dir.
        write_pi_agent_dir(&mut env, None).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_STARTER_ROUTER") {
        let (port, cache, learned) = start_responses_to_chat_router("pi", &env).await?;
        route_cache = Some(cache);
        learned_requires_reasoning = Some(learned);
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_ROUTER") {
        let router_key = env
            .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
            .cloned()
            .unwrap_or_default();
        if let Some(port) = start_provider_oauth_router(&env, router_key, session_store).await? {
            write_pi_agent_dir(&mut env, Some(port)).await?;
        } else {
            let (port, cache, learned) = start_responses_to_chat_router("pi", &env).await?;
            route_cache = Some(cache);
            learned_requires_reasoning = Some(learned);
            write_pi_agent_dir(&mut env, Some(port)).await?;
        }
    }

    let pi_agent_dir = env.get("PI_CODING_AGENT_DIR").cloned();

    let codex_oauth_sync = if tool.is_codex_family() && env.contains_key("AIVO_CODEX_OAUTH_CREDS") {
        Some(prepare_codex_oauth_shadow(tool, &mut env, session_store).await?)
    } else {
        None
    };

    if tool == AIToolType::CodexApp && codex_oauth_sync.is_none() {
        prepare_codex_app_home_without_auth(&mut env, session_store).await?;
    }

    let gemini_system_settings =
        if tool == AIToolType::Gemini && env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH") {
            Some(prepare_gemini_api_key_settings_override(&mut env).await?)
        } else {
            None
        };

    let env_unset = env
        .remove(crate::services::environment_injector::AIVO_INTERNAL_ENV_UNSET)
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    // The routers above consumed their upstream credentials; strip them (and
    // the token carrier) so the child env never holds a real key — the child
    // authenticates to the loopback router with its injected per-launch token.
    for var in [
        "AIVO_ROUTER_API_KEY",
        "AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY",
        "AIVO_ANTHROPIC_TO_OPENAI_ROUTER_TIERS_JSON",
        "AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MAIN_KEY_ID",
        "AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY",
        "AIVO_GEMINI_ROUTER_API_KEY",
        "AIVO_COPILOT_GITHUB_TOKEN",
        crate::services::environment_injector::AIVO_ROUTER_AUTH_TOKEN,
    ] {
        env.remove(var);
    }

    Ok(LaunchRuntimeState {
        env,
        env_unset,
        route_cache,
        learned_requires_reasoning,
        pi_agent_dir,
        codex_oauth_sync,
        gemini_system_settings,
    })
}

/// Parses `AIVO_CODEX_OAUTH_CREDS` (set by `environment_injector::for_codex`
/// for ChatGPT OAuth keys), refreshes the access token if near expiry, and
/// writes a shadow `CODEX_HOME` temp dir containing a native `auth.json`.
///
/// The placeholder env vars are stripped before codex is spawned; all codex
/// sees is `CODEX_HOME=<shadow>` and the standard model overrides.
async fn prepare_codex_oauth_shadow(
    tool: AIToolType,
    env: &mut HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<CodexOAuthSync> {
    let raw = env
        .remove("AIVO_CODEX_OAUTH_CREDS")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_OAUTH_CREDS"))?;
    let key_id = env
        .remove("AIVO_CODEX_KEY_ID")
        .or_else(|| env.remove("AIVO_CODEX_APP_HOME_KEY"))
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_KEY_ID"))?;
    env.remove("AIVO_CODEX_APP_HOME_KEY");
    let mut creds = CodexOAuthCredential::from_json(&raw)?;

    if tool == AIToolType::CodexApp {
        adopt_persistent_codex_app_tokens(session_store, &key_id, &mut creds).await;
    }

    // Refresh pre-launch so codex starts with a valid access token. If the
    // refresh token has been invalidated — codex was killed before the
    // post-exit sync ran, or another process (native codex, a parallel aivo)
    // rotated it — fall back to an interactive re-login and persist the new
    // creds immediately so the next launch doesn't repeat the recovery.
    if let Err(e) = ensure_fresh(&mut creds, REFRESH_SKEW_SECS).await {
        if !is_oauth_invalid_grant(&e) {
            return Err(e);
        }
        eprintln!(
            "{} Codex refresh token is no longer valid — re-authenticating.",
            crate::style::yellow("aivo:")
        );
        let stale = creds.clone();
        creds = crate::services::codex_oauth::interactive_login()
            .await
            .map_err(|err| err.context("codex re-login after invalid refresh token"))?;
        persist_refreshed_if_needed(session_store, &key_id, &stale, &creds).await;
    }

    let shadow = if tool == AIToolType::CodexApp {
        CodexHomeShadow::create_persistent(&creds, session_store.config_dir(), &key_id).await?
    } else {
        CodexHomeShadow::create(&creds).await?
    };
    env.insert(
        "CODEX_HOME".to_string(),
        shadow.path().to_string_lossy().to_string(),
    );

    Ok(CodexOAuthSync {
        key_id,
        shadow,
        original: creds,
    })
}

async fn prepare_codex_app_home_without_auth(
    env: &mut HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<()> {
    let key_id = env
        .remove("AIVO_CODEX_APP_HOME_KEY")
        .unwrap_or_else(|| "default".to_string());
    let shadow =
        CodexHomeShadow::create_persistent_without_auth(session_store.config_dir(), &key_id)
            .await?;
    env.insert(
        "CODEX_HOME".to_string(),
        shadow.path().to_string_lossy().to_string(),
    );
    Ok(())
}

async fn adopt_persistent_codex_app_tokens(
    session_store: &SessionStore,
    key_id: &str,
    creds: &mut CodexOAuthCredential,
) {
    let auth_path =
        CodexHomeShadow::persistent_path(session_store.config_dir(), key_id).join("auth.json");
    let disk = match CodexHomeShadow::read_auth_path(auth_path).await {
        Ok(Some(disk)) => disk,
        _ => return,
    };
    if !tokens_changed(creds, &disk) {
        return;
    }
    let updated = auth_dot_json_into_credential(disk, creds);
    let original = creds.clone();
    persist_refreshed_if_needed(session_store, key_id, &original, &updated).await;
    *creds = updated;
}

/// Reads the shadow `auth.json` back after codex exits and, if any token
/// changed, persists the rotated credential into aivo's store. Errors are
/// logged but never propagated — the user's codex session has already
/// completed, and a failed sync just means the next launch will refresh
/// again.
pub(crate) async fn finalize_codex_oauth(
    session_store: &SessionStore,
    sync: Option<CodexOAuthSync>,
) {
    let Some(sync) = sync else {
        return;
    };

    let disk = match sync.shadow.read_back().await {
        Ok(Some(v)) => v,
        Ok(None) => {
            // File missing/truncated — codex probably crashed. Keep the
            // pre-launch credential intact. (It was refreshed before
            // launch, so the refresh_token is already up-to-date on disk.)
            persist_refreshed_if_needed(
                session_store,
                &sync.key_id,
                &sync.original,
                &sync.original,
            )
            .await;
            return;
        }
        Err(_) => return,
    };

    let updated = auth_dot_json_into_credential(disk, &sync.original);
    persist_refreshed_if_needed(session_store, &sync.key_id, &sync.original, &updated).await;
}

fn auth_dot_json_into_credential(
    disk: AuthDotJson,
    original: &CodexOAuthCredential,
) -> CodexOAuthCredential {
    disk.into_credential(original.email.clone(), original.expires_at)
}

async fn persist_refreshed_if_needed(
    session_store: &SessionStore,
    key_id: &str,
    original: &CodexOAuthCredential,
    updated: &CodexOAuthCredential,
) {
    if original == updated {
        return;
    }
    let json = match updated.to_json() {
        Ok(j) => j,
        Err(_) => return,
    };
    // base_url / name / protocols are preserved by passing the same values
    // as the existing entry. Pull the current entry first so we don't
    // clobber name changes made mid-session.
    if let Ok(Some(existing)) = session_store.get_key_by_id(key_id).await {
        let _ = session_store
            .update_key(
                key_id,
                &existing.name,
                &existing.base_url,
                existing.claude_protocol,
                &json,
            )
            .await;
    }
}

/// Heuristic for "the refresh server told us our refresh token is bad" vs
/// "transient failure". 4xx from the token endpoint (`invalid_grant`,
/// `invalid_request_error`) is recoverable via an interactive re-login;
/// 5xx and network errors are not. `codex_oauth::refresh` formats its
/// failures as `"refresh failed (NNN): <body>"`, so a substring match is
/// enough.
pub(crate) fn is_oauth_invalid_grant(e: &anyhow::Error) -> bool {
    let s = e.to_string();
    s.contains("refresh failed (400")
        || s.contains("refresh failed (401")
        || s.contains("refresh failed (403")
        || s.contains("refresh failed (404")
}

/// Writes a gemini-cli *system-scope* settings file containing just
/// `security.auth.selectedType = "gemini-api-key"` and points
/// `GEMINI_CLI_SYSTEM_SETTINGS_PATH` at it. The CLI merges system over
/// user over defaults, so this forces the `USE_GEMINI` auth path (which
/// honors `GEMINI_API_KEY` + `GOOGLE_GEMINI_BASE_URL`) even when the
/// user's `~/.gemini/settings.json` has a stale `oauth-personal`
/// selection from a prior Google login. Without this,
/// `configuredAuthType || getAuthTypeFromEnv()` in gemini-cli returns
/// `LOGIN_WITH_GOOGLE` and every request bypasses aivo's router.
///
/// The user's real settings file is never read, copied, or written by
/// aivo; in-session edits (theme, vim mode, MCP server tweaks, model
/// preferences) persist to `~/.gemini/` as usual. Auto-fallbacks via
/// `activateFallbackMode` use `isTemporary=true` in gemini-cli and so
/// skip the user-scope `model.name` write, which is the only automatic
/// write path that could leak an aivo-injected model back into the
/// user's defaults.
async fn prepare_gemini_api_key_settings_override(
    env: &mut HashMap<String, String>,
) -> Result<tempfile::TempDir> {
    use anyhow::Context;
    env.remove("AIVO_GEMINI_FORCE_API_KEY_AUTH");
    let model_config_model = env.remove("AIVO_GEMINI_MODEL_CONFIG_MODEL");

    let dir = tempfile::Builder::new()
        .prefix("aivo-gemini-settings-")
        .tempdir()
        .context("create aivo gemini settings override temp dir")?;
    let path = dir.path().join("settings.json");
    let mut settings = serde_json::json!({
        "security": {
            "auth": {
                "selectedType": "gemini-api-key"
            }
        }
    });
    if let Some(model) = model_config_model.filter(|m| !m.trim().is_empty()) {
        settings["modelConfigs"] = gemini_internal_model_config_override(&model);
    }
    tokio::fs::write(&path, serde_json::to_vec(&settings)?)
        .await
        .context("write aivo gemini system settings override")?;

    env.insert(
        "GEMINI_CLI_SYSTEM_SETTINGS_PATH".to_string(),
        path.to_string_lossy().to_string(),
    );
    Ok(dir)
}

fn gemini_internal_model_config_override(model: &str) -> serde_json::Value {
    let aliases = [
        // Gemini CLI 0.40.x uses these helper aliases for routing,
        // completion, edit correction and summarization. Several default to
        // gemini-2.5-flash-lite, which may be unavailable or irrelevant when
        // aivo is routing the CLI to a non-Google provider.
        "gemini-2.5-flash-lite",
        "classifier",
        "prompt-completion",
        "edit-corrector",
        "summarizer-default",
        "summarizer-shell",
        "chat-compression-2.5-flash-lite",
    ];
    let custom_aliases = aliases
        .into_iter()
        .map(|alias| {
            (
                alias.to_string(),
                serde_json::json!({
                    "modelConfig": {
                        "model": model
                    }
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    serde_json::json!({
        "customAliases": custom_aliases
    })
}

pub(crate) async fn record_launch_state(
    session_store: &SessionStore,
    key: &ApiKey,
    tool: AIToolType,
    model: Option<&str>,
) {
    let _ = session_store
        .record_selection(&key.id, tool.as_str(), model)
        .await;
}

/// One-shot migration of routing fields on `key`, run before any router reads
/// them. Mutates `key` in place and persists the same change (persist failures
/// are ignored — the in-memory mutation still benefits this launch).
///
/// - **v1**: cleared a latched `responses_api_supported = Some(false)`.
/// - **v2**: fold the five per-CLI scalar pins into the per-`(tool, model)`
///   `protocol_routes` map under each tool's `""` default, then drop them
///   (lossless — the scalars were already per-CLI defaults).
pub(crate) async fn migrate_routing_schema_for_key(session_store: &SessionStore, key: &mut ApiKey) {
    use crate::services::session_store::CURRENT_ROUTING_SCHEMA_VERSION;
    if key.routing_schema_version >= CURRENT_ROUTING_SCHEMA_VERSION {
        return;
    }

    let mut migrated: BTreeMap<String, BTreeMap<String, PersistedRoute>> = BTreeMap::new();
    if let Some(p) = key.claude_protocol {
        migrated.entry("claude".to_string()).or_default().insert(
            String::new(),
            PersistedRoute {
                protocol: p.as_str().to_string(),
                path_variant: key
                    .claude_path_variant
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
            },
        );
    }
    if let Some(p) = key.gemini_protocol {
        migrated.entry("gemini".to_string()).or_default().insert(
            String::new(),
            PersistedRoute {
                protocol: p.as_str().to_string(),
                path_variant: key
                    .gemini_path_variant
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
            },
        );
    }
    if let Some(supported) = key.responses_api_supported {
        migrated.entry("codex".to_string()).or_default().insert(
            String::new(),
            PersistedRoute {
                protocol: if supported { "responses" } else { "openai" }.to_string(),
                path_variant: "default".to_string(),
            },
        );
    }

    // Mirror into the in-memory key so this launch seeds the routers correctly,
    // then drop the legacy scalars (existing routes win over migrated defaults).
    for (tool, models) in &migrated {
        let dst = key.protocol_routes.entry(tool.clone()).or_default();
        for (model, route) in models {
            dst.entry(model.clone()).or_insert_with(|| route.clone());
        }
    }
    key.claude_protocol = None;
    key.gemini_protocol = None;
    key.responses_api_supported = None;
    key.claude_path_variant = None;
    key.gemini_path_variant = None;
    key.routing_schema_version = CURRENT_ROUTING_SCHEMA_VERSION;

    let _ = session_store
        .migrate_key_to_routes_v2(&key.id, migrated, CURRENT_ROUTING_SCHEMA_VERSION)
        .await;
}

pub(crate) async fn persist_runtime_discoveries(
    session_store: &SessionStore,
    key: &ApiKey,
    route_cache: Option<Arc<RouteCache>>,
    learned_requires_reasoning: Option<Arc<AtomicBool>>,
) {
    // Persist a learned `requires_reasoning_content` quirk: the upstream's
    // *parseable* error envelope is itself proof the quirk is real, even if no
    // 2xx response was ever seen this session.
    if let Some(flag) = learned_requires_reasoning.as_ref()
        && flag.load(Ordering::Relaxed)
        && key.requires_reasoning_content != Some(true)
    {
        let _ = session_store
            .set_key_requires_reasoning_content(&key.id, Some(true))
            .await;
    }

    // Merge confirmed per-model routes back into the key. `dirty_routes` only
    // returns proven, new-or-changed slots, so a failures-only session can't
    // poison the store and unchanged routes aren't rewritten.
    if let Some(cache) = route_cache {
        let dirty = cache.dirty_routes();
        if !dirty.is_empty() {
            let _ = session_store
                .merge_routes(&key.id, cache.tool(), &dirty)
                .await;
        }
    }
}

/// Walk Pi session JSONL files in the temp agent dir and copy them to
/// `~/.pi/agent/sessions/` for long-term storage.
pub(crate) async fn process_pi_sessions(pi_agent_dir: Option<&str>) {
    let temp_dir = match pi_agent_dir {
        Some(d) => d,
        None => return,
    };

    let temp_sessions = std::path::PathBuf::from(temp_dir).join("sessions");
    let real_sessions = crate::services::system_env::home_dir()
        .map(|h| h.join(".pi").join("agent").join("sessions"));

    let Some(real_sessions) = real_sessions else {
        return;
    };

    if pi_sessions_share_storage(&temp_sessions, &real_sessions).await {
        return;
    }

    copy_pi_session_jsonl_tree(&temp_sessions, &real_sessions).await;
}

pub(crate) async fn cleanup_runtime_artifacts(
    codex_model_catalog_path: Option<&str>,
    claude_settings_pin_path: Option<&str>,
    pi_agent_dir: Option<&str>,
) {
    if let Some(path) = codex_model_catalog_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    if let Some(path) = claude_settings_pin_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    if let Some(dir) = pi_agent_dir {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
}

/// Writes a temporary `PI_CODING_AGENT_DIR` that surfaces the user's real
/// `~/.pi/agent/` customization (packages, MCP servers, rules, themes,
/// settings, auth) while pinning the provider to the aivo entry.
///
/// `models.json` is aivo-only so an explicit `--model anthropic/foo`
/// against an aivo launch errors with "unknown provider" instead of
/// silently routing through the user's real key. Everything else is
/// symlinked so mid-session writes (package installs, login flows)
/// persist back to the real home.
///
/// When `port` is `Some`, the placeholder `PLACEHOLDER_LOOPBACK_URL` in
/// `AIVO_PI_MODELS_JSON` is patched with the real router port.
/// When `port` is `None`, the JSON already contains the real upstream URL.
async fn write_pi_agent_dir(env: &mut HashMap<String, String>, port: Option<u16>) -> Result<()> {
    let real_agent = crate::services::system_env::home_dir().map(|h| h.join(".pi").join("agent"));
    write_pi_agent_dir_with_real(env, port, real_agent.as_deref()).await
}

async fn write_pi_agent_dir_with_real(
    env: &mut HashMap<String, String>,
    port: Option<u16>,
    real_agent: Option<&Path>,
) -> Result<()> {
    let raw = env
        .get("AIVO_PI_MODELS_JSON")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_PI_MODELS_JSON"))?
        .clone();

    let aivo_models_json = match port {
        Some(p) => {
            ensure_loopback_no_proxy(env);
            raw.replace(PLACEHOLDER_LOOPBACK_URL, &format!("http://127.0.0.1:{p}"))
        }
        None => raw,
    };

    let dir = tempfile::Builder::new()
        .prefix("aivo-pi-")
        .tempdir()?
        .keep();

    tokio::try_join!(
        tokio::fs::write(dir.join("models.json"), aivo_models_json.as_bytes()),
        link_or_default(real_agent, "settings.json", &dir),
        link_or_default(real_agent, "auth.json", &dir),
    )?;

    if let Some(real_agent) = real_agent {
        link_pi_agent_state(real_agent, &dir).await;
        populate_pi_bin_dir(&real_agent.join("bin"), &dir.join("bin")).await;
        seed_pi_sessions(Some(real_agent.join("sessions")), &dir).await;
    } else {
        seed_pi_sessions(None, &dir).await;
    }

    env.insert(
        "PI_CODING_AGENT_DIR".to_string(),
        dir.to_string_lossy().to_string(),
    );
    Ok(())
}

/// Tries symlink → hard-link → copy so writes propagate back to `real`
/// when possible. NTFS hard links work without Developer Mode and still
/// propagate writes; copy is the last-resort read-only path. Returns
/// `Err` only if all three fail (e.g., `real` became unreadable).
async fn link_existing_file(real: &Path, dest: &Path) -> std::io::Result<()> {
    if symlink_file(real, dest).await.is_ok() {
        return Ok(());
    }
    if tokio::fs::hard_link(real, dest).await.is_ok() {
        return Ok(());
    }
    let bytes = tokio::fs::read(real).await?;
    tokio::fs::write(dest, bytes).await
}

/// Touches the real file with `{}` if missing, then links it into
/// `dest` via `link_existing_file`. Used for files Pi expects to find
/// (`settings.json`, `auth.json`) — the default keeps Pi happy even on
/// a fresh `~/.pi/agent/`.
async fn link_or_default(
    real_agent: Option<&Path>,
    name: &str,
    dest: &Path,
) -> std::io::Result<()> {
    let dest_path = dest.join(name);
    let Some(real) = real_agent else {
        return tokio::fs::write(&dest_path, "{}").await;
    };
    let real_path = real.join(name);

    if !real_path.is_file() {
        if let Some(parent) = real_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::write(&real_path, "{}").await;
    }

    if link_existing_file(&real_path, &dest_path).await.is_ok() {
        return Ok(());
    }
    // Last-resort: real became unreadable between is_file() check and copy.
    tokio::fs::write(&dest_path, b"{}").await
}

/// Symlinks the user's mutable `~/.pi/agent/` state (rules, tools,
/// prompts, themes, git- and npm-sourced packages, mcp.json) into the
/// temp dir. Dirs are pre-created so a first-time `pi install <pkg>`
/// lands in the real home instead of vanishing with the temp dir.
/// `mcp.json` goes through the same symlink → hard-link → copy chain as the
/// other linked files; dirs use a junction (see `symlink_util`) so both persist
/// writes on Windows without Developer Mode.
async fn link_pi_agent_state(real_agent: &Path, dest: &Path) {
    // Add new pi state dirs here as they appear. `bin/`, `sessions/`,
    // `models.json` are absent on purpose (each handled specially).
    for d in ["rules", "tools", "prompts", "themes", "git", "npm"] {
        let real = real_agent.join(d);
        if tokio::fs::create_dir_all(&real).await.is_ok() {
            let _ = symlink_dir(&real, &dest.join(d)).await;
        }
    }

    let mcp = real_agent.join("mcp.json");
    if mcp.is_file() {
        let _ = link_existing_file(&mcp, &dest.join("mcp.json")).await;
    }
}

/// Ensure `dest_bin` is a writable dir for pi's managed binaries, linking the
/// existing ones from `real_bin` when present. Best effort: failures are
/// silently skipped so pi just re-downloads what's missing.
#[cfg(unix)]
async fn populate_pi_bin_dir(real_bin: &std::path::Path, dest_bin: &std::path::Path) {
    // On first run real_bin doesn't exist yet; a dangling symlink would make
    // pi's own `mkdir <temp>/bin` fail (ENOENT), so fall back to a writable dir.
    if real_bin.is_dir() && symlink_dir(real_bin, dest_bin).await.is_ok() {
        return;
    }
    let _ = tokio::fs::create_dir_all(dest_bin).await;
}

#[cfg(windows)]
async fn populate_pi_bin_dir(real_bin: &std::path::Path, dest_bin: &std::path::Path) {
    // Windows symlinks / junctions need elevation or developer mode; fall
    // back to per-file hard links, then copies. Works for the common case
    // where HOME and the temp dir share a filesystem.
    if tokio::fs::create_dir_all(dest_bin).await.is_err() {
        return;
    }
    let mut entries = match tokio::fs::read_dir(real_bin).await {
        Ok(rd) => rd,
        Err(_) => return,
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let src = entry.path();
        let Some(name) = src.file_name() else {
            continue;
        };
        let dst = dest_bin.join(name);
        if tokio::fs::hard_link(&src, &dst).await.is_ok() {
            continue;
        }
        // hard_link fails across filesystems or on directories; try a
        // plain copy for regular files before giving up. entry.file_type()
        // reuses readdir metadata so we skip a redundant stat syscall.
        if let Ok(ft) = entry.file_type().await
            && ft.is_file()
        {
            let _ = tokio::fs::copy(&src, &dst).await;
        }
    }
}

#[cfg(not(any(unix, windows)))]
async fn populate_pi_bin_dir(_real_bin: &std::path::Path, dest_bin: &std::path::Path) {
    let _ = tokio::fs::create_dir_all(dest_bin).await;
}

async fn seed_pi_sessions(
    real_sessions: Option<std::path::PathBuf>,
    temp_agent_dir: &std::path::Path,
) {
    let temp_sessions = temp_agent_dir.join("sessions");

    let Some(real_sessions) = real_sessions else {
        let _ = tokio::fs::create_dir_all(&temp_sessions).await;
        return;
    };

    if tokio::fs::create_dir_all(&real_sessions).await.is_err() {
        let _ = tokio::fs::create_dir_all(&temp_sessions).await;
        return;
    }

    if link_pi_sessions_dir(&real_sessions, &temp_sessions).await {
        return;
    }

    copy_pi_session_jsonl_tree(&real_sessions, &temp_sessions).await;
}

#[cfg(unix)]
async fn link_pi_sessions_dir(
    real_sessions: &std::path::Path,
    temp_sessions: &std::path::Path,
) -> bool {
    tokio::fs::symlink(real_sessions, temp_sessions)
        .await
        .is_ok()
}

#[cfg(not(unix))]
async fn link_pi_sessions_dir(
    real_sessions: &std::path::Path,
    temp_sessions: &std::path::Path,
) -> bool {
    // Windows: junction it in (write-through). Non-unix/windows Errs → copy fallback.
    symlink_dir(real_sessions, temp_sessions).await.is_ok()
}

async fn pi_sessions_share_storage(
    temp_sessions: &std::path::Path,
    real_sessions: &std::path::Path,
) -> bool {
    let (Ok(temp), Ok(real)) = (
        tokio::fs::canonicalize(temp_sessions).await,
        tokio::fs::canonicalize(real_sessions).await,
    ) else {
        return false;
    };
    temp == real
}

async fn copy_pi_session_jsonl_tree(src_root: &std::path::Path, dst_root: &std::path::Path) {
    let mut dirs = vec![src_root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(rel) = path.strip_prefix(src_root) else {
                continue;
            };
            let dest = dst_root.join(rel);
            if let Some(parent) = dest.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::copy(&path, &dest).await;
        }
    }
}

fn set_local_base_url(env: &mut HashMap<String, String>, key: &str, port: u16) {
    env.insert(key.to_string(), format!("http://127.0.0.1:{port}"));
    ensure_loopback_no_proxy(env);
}

fn patch_opencode_config_content(env: &mut HashMap<String, String>, port: u16) {
    let real_url = format!("http://127.0.0.1:{port}");
    if let Some(content) = env.get("OPENCODE_CONFIG_CONTENT").cloned() {
        let patched = content.replace(PLACEHOLDER_LOOPBACK_URL, &real_url);
        env.insert("OPENCODE_CONFIG_CONTENT".to_string(), patched);
        ensure_loopback_no_proxy(env);
    }
}

/// Clears every HTTP proxy env var the spawned gemini might honor, in
/// both casings. Needed because gemini reads them, installs a global
/// undici `ProxyAgent`, and from then on every `fetch` — including to
/// our `http://127.0.0.1:<port>` router — is sent through the proxy
/// regardless of `NO_PROXY`. Setting the vars to empty strings (rather
/// than removing them) is enough because the lookups use `||` chains
/// that treat `""` as falsy.
///
/// `ALL_PROXY` isn't on gemini-cli's primary lookup path, but the
/// bundled `proxy-from-env` library and various sub-deps (gaxios,
/// googleapis) do consult it, so we clear it too for defense in depth.
fn clear_node_proxy_env(env: &mut HashMap<String, String>) {
    for var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        env.insert(var.to_string(), String::new());
    }
}

/// Ensures the spawned subprocess will bypass any HTTP proxy when talking to
/// the local loopback router. Without this, a user's `HTTP_PROXY`/`HTTPS_PROXY`
/// would route the subprocess's `http://127.0.0.1:<port>` request through the
/// proxy, which cannot reach the user's localhost. We append loopback entries
/// to both the upper- and lower-case variants since different HTTP libraries
/// check different casings.
fn ensure_loopback_no_proxy(env: &mut HashMap<String, String>) {
    for var in NO_PROXY_VAR_NAMES {
        let existing = env.get(*var).cloned().unwrap_or_default();
        env.insert((*var).to_string(), merge_loopback_entries(&existing));
    }
}

/// Same as `ensure_loopback_no_proxy` but mutates the current process env.
///
/// SAFETY: aivo's tokio runtime is `current_thread` (see `main.rs`), so async
/// work shares one OS thread and concurrent env reads can't race. Must not
/// run inside or after a `spawn_blocking` on the env vars being modified —
/// the blocking pool's reads would race the write.
pub fn ensure_loopback_no_proxy_in_process_env() {
    // SAFETY: see fn-level comment.
    unsafe {
        for var in NO_PROXY_VAR_NAMES {
            let existing = std::env::var(var).unwrap_or_default();
            std::env::set_var(var, merge_loopback_entries(&existing));
        }
    }
}

const NO_PROXY_VAR_NAMES: &[&str] = &["NO_PROXY", "no_proxy"];
const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

/// Merges loopback hosts into a comma-separated NO_PROXY value, deduping
/// case-insensitively so a pre-existing `LOCALHOST` or `127.0.0.1` entry
/// doesn't get duplicated.
fn merge_loopback_entries(existing: &str) -> String {
    let mut entries: Vec<String> = existing
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for host in LOOPBACK_HOSTS {
        if !entries.iter().any(|e| e.eq_ignore_ascii_case(host)) {
            entries.push((*host).to_string());
        }
    }
    entries.join(",")
}

/// Starts the built-in AnthropicRouter and returns the port it bound to
/// Per-launch loopback token injected by the environment injector; routers
/// require it from clients so other local processes can't spend the key.
fn loopback_auth_token(env: &HashMap<String, String>) -> Option<String> {
    env.get(crate::services::environment_injector::AIVO_ROUTER_AUTH_TOKEN)
        .cloned()
}

/// Decodes a Claude-subscription key's OAuth token.
fn claude_oauth_token(key: &ApiKey) -> Result<String> {
    crate::services::claude_oauth::ClaudeOAuthCredential::from_json(key.key.as_str())
        .map(|c| c.token)
        .map_err(|e| {
            anyhow::anyhow!(
                "Claude OAuth credential for key '{}' is unreadable — re-run `aivo keys add claude` ({e})",
                key.display_name()
            )
        })
}

/// Provider-OAuth (Codex/Kimi/Grok) `router_key`: starts the universal loopback
/// ServeRouter with this launch's auth token and returns its port; `None`
/// otherwise so the caller falls back to its tool-native static router.
async fn start_provider_oauth_router(
    env: &HashMap<String, String>,
    router_key: String,
    session_store: &SessionStore,
) -> Result<Option<u16>> {
    crate::services::serve_router::start_provider_oauth_loopback_router(
        &router_key,
        loopback_auth_token(env),
        session_store,
    )
    .await
}

/// Multi-upstream ServeRouter for `aivo claude` per-tier routing: base = main
/// key, each tier model → its own key. Key ids come from env; keys are resolved
/// from the store (secrets never travel in env). Returns the loopback port.
async fn start_multi_upstream_serve_router(
    env: &HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<u16> {
    use crate::services::environment_injector::TierUpstream;

    let main_id = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MAIN_KEY_ID")
        .cloned()
        .unwrap_or_default();
    let mut main_key = session_store
        .get_keys()
        .await?
        .into_iter()
        .find(|k| k.id == main_id)
        .ok_or_else(|| anyhow::anyhow!("per-tier routing: main key '{main_id}' not found"))?;
    SessionStore::decrypt_key_secret(&mut main_key)?;

    // Hard error rather than silently fall through to base-only routing.
    let tiers: Vec<TierUpstream> = match env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_TIERS_JSON") {
        Some(s) => serde_json::from_str(s)
            .map_err(|e| anyhow::anyhow!("per-tier routing: invalid tiers spec: {e}"))?,
        None => Vec::new(),
    };
    let tiers: Vec<(String, String)> = tiers.into_iter().map(|t| (t.model, t.key_id)).collect();

    // A Claude-subscription main authenticates to the loopback with its own
    // OAuth bearer, so the gate must expect that token.
    let auth_token = if main_key.is_claude_oauth() {
        Some(claude_oauth_token(&main_key)?)
    } else {
        loopback_auth_token(env)
    };
    crate::services::serve_router::start_tier_loopback_router(
        main_key,
        &tiers,
        auth_token,
        session_store,
    )
    .await
}

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
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    };

    let mut router = AnthropicRouter::new(config);
    if let Some(token) = loopback_auth_token(env) {
        router = router.with_auth_token(token);
    }
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Parse a `*_ROUTES_JSON` env var into a router's per-model seed; empty on a
/// missing/unparseable value (the router then uses its tool-native prior).
fn parse_seed_routes(env: &HashMap<String, String>, var: &str) -> BTreeMap<String, PersistedRoute> {
    env.get(var)
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default()
}

async fn start_anthropic_to_openai_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<RouteCache>, Arc<AtomicBool>)> {
    use crate::services::{AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig};

    let api_key = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let strip_cache_control = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_STRIP_CACHE_CONTROL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let config = AnthropicToOpenAIRouterConfig {
        target_base_url: base_url,
        target_api_key: api_key,
        seed_routes: parse_seed_routes(env, "AIVO_ANTHROPIC_TO_OPENAI_ROUTER_ROUTES_JSON"),
        strip_cache_control,
        model_prefix,
        requires_reasoning_content,
        max_tokens_cap,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    };

    let mut router = AnthropicToOpenAIRouter::new(config);
    if let Some(token) = loopback_auth_token(env) {
        router = router.with_auth_token(token);
    }
    let (port, route_cache, learned_requires_reasoning, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic-to-openai router exited unexpectedly: {e}");
        }
    });
    Ok((port, route_cache, learned_requires_reasoning))
}

async fn start_responses_to_chat_router(
    tool_name: &'static str,
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<RouteCache>, Arc<AtomicBool>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let api_key = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let actual_model = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL")
        .cloned();
    let max_tokens_cap = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let responses_api_supported = match env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_RESPONSES_API")
        .map(|v| v.as_str())
    {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    };

    let aivo_prefix_models = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_AIVO_PREFIX_MODELS")
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        target_path_variant: None,
        copilot_token_manager: None,
        model_prefix,
        requires_reasoning_content,
        actual_model,
        max_tokens_cap,
        responses_api_supported,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
        aivo_prefix_models,
    })
    .with_tool(tool_name)
    .with_seed_routes(parse_seed_routes(
        env,
        "AIVO_RESPONSES_TO_CHAT_ROUTER_ROUTES_JSON",
    ));
    let router = match loopback_auth_token(env) {
        Some(token) => router.with_auth_token(token),
        None => router,
    };
    let (port, route_cache, learned_requires_reasoning, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat router exited unexpectedly: {e}");
        }
    });
    Ok((port, route_cache, learned_requires_reasoning))
}

async fn start_gemini_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<RouteCache>, Arc<AtomicBool>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
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
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    })
    .with_seed_routes(parse_seed_routes(env, "AIVO_GEMINI_ROUTER_ROUTES_JSON"));
    let router = match loopback_auth_token(env) {
        Some(token) => router.with_auth_token(token),
        None => router,
    };
    let (port, route_cache, learned_requires_reasoning, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok((port, route_cache, learned_requires_reasoning))
}

async fn start_gemini_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{GeminiRouter, GeminiRouterConfig};

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
        upstream_protocol: ProviderProtocol::ResponsesApi,
        forced_model,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    });
    let router = match loopback_auth_token(env) {
        Some(token) => router.with_auth_token(token),
        None => router,
    };
    let (port, _route_cache, _learned, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CopilotRouter, CopilotRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let mut router = CopilotRouter::new(CopilotRouterConfig { github_token });
    if let Some(token) = loopback_auth_token(env) {
        router = router.with_auth_token(token);
    }
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_cursor_router(env: &mut HashMap<String, String>, tool: AIToolType) -> Result<u16> {
    use crate::services::cursor_acp::{self, CURSOR_ACP_SENTINEL};
    use crate::services::cursor_bridge::mcp::ToolUseIdStyle;
    use crate::services::cursor_bridge::{CursorModelRouter, CursorRouterConfig};
    use crate::services::session_store::ApiKey;
    use zeroize::Zeroizing;

    let key_secret = env.remove("AIVO_CURSOR_KEY_SECRET").ok_or_else(|| {
        anyhow::anyhow!(
            "Missing AIVO_CURSOR_KEY_SECRET; re-run `aivo keys add cursor` to set up an isolated cursor account."
        )
    })?;

    if cursor_acp::is_legacy_cursor_login_secret(&key_secret) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Remove it (`aivo keys rm cursor`) and re-add (`aivo keys add cursor`) so cursor-agent runs in its own isolated home."
        );
    }

    let key = ApiKey {
        id: "cursor-router".to_string(),
        name: "cursor".to_string(),
        base_url: CURSOR_ACP_SENTINEL.to_string(),
        claude_protocol: None,
        gemini_protocol: None,
        responses_api_supported: None,
        codex_mode: None,
        opencode_mode: None,
        pi_mode: None,
        claude_path_variant: None,
        gemini_path_variant: None,
        requires_reasoning_content: None,
        protocol_routes: Default::default(),
        routing_schema_version: 0,
        key: Zeroizing::new(key_secret),
        created_at: String::new(),
    };

    // Bail before spawning the router when the saved OAuth shadow has been
    // signed out, rather than let the first request surface a dead upstream.
    if cursor_acp::cursor_oauth_shadow_signed_out(&key).await {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth <id>` (or pick `aivo keys reauth` interactively) to sign in again."
        );
    }
    let workspace_cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_else(|| ".".to_string());

    // Every launched tool that targets cursor sends `stream:true` + a tools
    // array on every real turn, so they all flow through
    // `run_*_bridged_fresh`, which opens its own session with the MCP bridge
    // attached at `session/new` time. MCP-less pool prewarms were dead
    // weight — observed in debug captures opening two cursor-agent processes
    // (Claude `prewarm_count=2`) that the bridged path never consumed.
    // Replace the pool prewarm with a single MCP-attached prewarm keyed to
    // the protocol's id-style; the first bridged turn picks it up via
    // `take_mcp_prewarmed`. Claude Code's main+subagent paired burst still
    // works: title-gen subagents short-circuit before hitting cursor, and
    // genuine subagent dispatch is internal to cursor-agent (we only see one
    // `/v1/messages` per main turn).
    let prewarm_count = 0;
    let mcp_prewarm_id_style = Some(match tool {
        AIToolType::Claude => ToolUseIdStyle::Anthropic,
        AIToolType::Gemini => ToolUseIdStyle::Gemini,
        AIToolType::Codex | AIToolType::CodexApp | AIToolType::Opencode | AIToolType::Pi => {
            ToolUseIdStyle::OpenAi
        }
    });
    let router = CursorModelRouter::new(CursorRouterConfig {
        key,
        workspace_cwd,
        models_cache: Some(crate::services::models_cache::ModelsCache::new()),
        prewarm_count,
        mcp_prewarm_id_style,
        // Native-tool launches now inject a per-launch token too, so the
        // bearer gate applies the same as for the plugin endpoint.
        expected_token: loopback_auth_token(env),
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: cursor router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_responses_to_chat_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        target_protocol: ProviderProtocol::Openai,
        target_path_variant: None,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: false,
        aivo_prefix_models: Vec::new(),
    });
    let router = match loopback_auth_token(env) {
        Some(token) => router.with_auth_token(token),
        None => router,
    };
    let (port, _route_cache, _learned, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::{
        clear_node_proxy_env, is_oauth_invalid_grant, patch_opencode_config_content,
        prepare_gemini_api_key_settings_override,
    };
    use crate::services::provider_protocol::ProviderProtocol;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    #[test]
    fn is_oauth_invalid_grant_matches_4xx_refresh_failures() {
        // Real-world wording from the OpenAI token endpoint. The whole point
        // of this branch is to trigger interactive re-login, so the bodies
        // matter less than the status prefix.
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (401): {{\"error\":{{\"message\":\"Your refresh token has already been used\"}}}}"
        )));
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (400): invalid_grant"
        )));
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (403): forbidden"
        )));
        // The appended reauth hint must not break the status-prefix match.
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (400){}: invalid_grant",
            crate::services::oauth_credential::REAUTH_HINT
        )));
    }

    #[test]
    fn is_oauth_invalid_grant_skips_transient_failures() {
        // 5xx, network, and parse errors aren't recoverable via re-login —
        // we don't want to open a browser for a transient outage.
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (500): bad gateway"
        )));
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "POST /oauth/token (refresh_token)"
        )));
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "parse refresh response"
        )));
    }

    #[test]
    fn patch_opencode_config_content_rewrites_placeholder_url() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"http://127.0.0.1:0\"}".to_string(),
        )]);

        patch_opencode_config_content(&mut env, 24860);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"http://127.0.0.1:24860\"}"
        );
    }

    #[test]
    fn patch_opencode_config_content_ignores_missing_payload() {
        let mut env = HashMap::new();
        patch_opencode_config_content(&mut env, 24860);
        assert!(env.is_empty());
    }

    #[test]
    fn set_local_base_url_inserts_loopback_address() {
        use super::set_local_base_url;
        let mut env = HashMap::new();
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:9999"
        );
    }

    #[test]
    fn set_local_base_url_overwrites_existing() {
        use super::set_local_base_url;
        let mut env = HashMap::from([(
            "ANTHROPIC_BASE_URL".to_string(),
            "https://old-url.example.com".to_string(),
        )]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 12345);
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:12345"
        );
    }

    #[test]
    fn patch_opencode_config_content_preserves_non_placeholder() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"https://api.openai.com/v1\"}".to_string(),
        )]);

        patch_opencode_config_content(&mut env, 24860);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"https://api.openai.com/v1\"}"
        );
    }

    #[test]
    fn patch_opencode_config_content_replaces_multiple_occurrences() {
        use crate::constants::PLACEHOLDER_LOOPBACK_URL;

        let content = format!(
            "{{\"url1\":\"{}\",\"url2\":\"{}\"}}",
            PLACEHOLDER_LOOPBACK_URL, PLACEHOLDER_LOOPBACK_URL
        );
        let mut env = HashMap::from([("OPENCODE_CONFIG_CONTENT".to_string(), content)]);

        patch_opencode_config_content(&mut env, 55555);

        let result = env.get("OPENCODE_CONFIG_CONTENT").unwrap();
        assert!(!result.contains(PLACEHOLDER_LOOPBACK_URL));
        assert_eq!(result.matches("http://127.0.0.1:55555").count(), 2);
    }

    #[test]
    fn patch_opencode_config_content_uses_constant() {
        use crate::constants::PLACEHOLDER_LOOPBACK_URL;

        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            format!("{{\"baseUrl\":\"{}\"}}", PLACEHOLDER_LOOPBACK_URL),
        )]);

        patch_opencode_config_content(&mut env, 9876);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"http://127.0.0.1:9876\"}"
        );
    }

    #[test]
    fn set_local_base_url_injects_loopback_no_proxy() {
        use super::set_local_base_url;
        let mut env = HashMap::new();
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").expect("NO_PROXY should be set");
        assert!(no_proxy.contains("127.0.0.1"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("localhost"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("::1"), "NO_PROXY={no_proxy}");

        let no_proxy_lower = env.get("no_proxy").expect("no_proxy should be set");
        assert!(no_proxy_lower.contains("127.0.0.1"));
    }

    #[test]
    fn set_local_base_url_appends_to_existing_no_proxy() {
        use super::set_local_base_url;
        let mut env = HashMap::from([("NO_PROXY".to_string(), "internal.corp".to_string())]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        assert!(no_proxy.contains("internal.corp"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("127.0.0.1"), "NO_PROXY={no_proxy}");
    }

    #[test]
    fn set_local_base_url_does_not_duplicate_existing_loopback_entry() {
        use super::set_local_base_url;
        let mut env = HashMap::from([(
            "NO_PROXY".to_string(),
            "127.0.0.1,localhost,::1".to_string(),
        )]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        assert_eq!(
            no_proxy.matches("127.0.0.1").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
        assert_eq!(
            no_proxy.matches("localhost").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
    }

    #[test]
    fn set_local_base_url_treats_existing_loopback_entries_case_insensitively() {
        use super::set_local_base_url;
        let mut env = HashMap::from([("NO_PROXY".to_string(), "LOCALHOST,127.0.0.1".to_string())]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        // Existing uppercase LOCALHOST should be preserved without a
        // duplicate lowercase `localhost` appended.
        assert!(no_proxy.contains("LOCALHOST"), "NO_PROXY={no_proxy}");
        assert_eq!(
            no_proxy.to_ascii_lowercase().matches("localhost").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
        assert_eq!(
            no_proxy.matches("127.0.0.1").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
    }

    #[test]
    fn patch_opencode_config_content_injects_loopback_no_proxy() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"http://127.0.0.1:0\"}".to_string(),
        )]);
        patch_opencode_config_content(&mut env, 24860);

        let no_proxy = env.get("NO_PROXY").expect("NO_PROXY should be set");
        assert!(no_proxy.contains("127.0.0.1"));
        assert!(no_proxy.contains("localhost"));
    }

    #[test]
    fn clear_node_proxy_env_empties_known_proxy_vars() {
        // Needed for gemini-cli: it builds a global undici ProxyAgent from
        // HTTP(S)_PROXY and then routes every fetch through it, including to
        // our loopback router. NO_PROXY is ignored by ProxyAgent, so the only
        // way to prevent this is to hide the proxy vars from the child. We
        // also clear ALL_PROXY because gemini's bundled `proxy-from-env`
        // lookup consults it as a fallback.
        let mut env = HashMap::from([
            ("HTTP_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("HTTPS_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("ALL_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("http_proxy".to_string(), "http://proxy:8080".to_string()),
            ("https_proxy".to_string(), "http://proxy:8080".to_string()),
            ("all_proxy".to_string(), "http://proxy:8080".to_string()),
            ("PATH".to_string(), "/usr/bin".to_string()),
        ]);
        clear_node_proxy_env(&mut env);
        for var in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ] {
            assert_eq!(env.get(var), Some(&String::new()), "{var} should be empty");
        }
        assert_eq!(
            env.get("PATH"),
            Some(&"/usr/bin".to_string()),
            "unrelated vars untouched"
        );
    }

    #[tokio::test]
    async fn prepare_gemini_api_key_settings_override_pins_selected_type() {
        let mut env = HashMap::from([
            (
                "AIVO_GEMINI_FORCE_API_KEY_AUTH".to_string(),
                "1".to_string(),
            ),
            (
                "AIVO_GEMINI_MODEL_CONFIG_MODEL".to_string(),
                "aivo/starter".to_string(),
            ),
        ]);
        let dir = prepare_gemini_api_key_settings_override(&mut env)
            .await
            .unwrap();

        // Sentinel consumed — must not leak to the spawned child.
        assert!(!env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH"));
        assert!(!env.contains_key("AIVO_GEMINI_MODEL_CONFIG_MODEL"));

        // Child sees a system-scope settings override path. Because
        // system-scope wins over user-scope in gemini-cli's merge, this
        // pins selectedType regardless of any stale `oauth-personal` in
        // the user's real ~/.gemini/settings.json.
        let path = env.get("GEMINI_CLI_SYSTEM_SETTINGS_PATH").unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            parsed["security"]["auth"]["selectedType"].as_str(),
            Some("gemini-api-key")
        );
        assert_eq!(
            parsed["modelConfigs"]["customAliases"]["prompt-completion"]["modelConfig"]["model"]
                .as_str(),
            Some("aivo/starter")
        );
        assert_eq!(
            parsed["modelConfigs"]["customAliases"]["classifier"]["modelConfig"]["model"].as_str(),
            Some("aivo/starter")
        );

        // We deliberately don't redirect GEMINI_CLI_HOME — user's real
        // ~/.gemini/ stays the gemini-cli user-scope root so in-session
        // edits (theme, vim mode, MCP tweaks) persist normally.
        assert!(!env.contains_key("GEMINI_CLI_HOME"));

        drop(dir);
        assert!(!std::path::Path::new(path).exists());
    }

    #[tokio::test]
    async fn copy_pi_session_jsonl_tree_preserves_nested_history() {
        let temp = tempfile::tempdir().unwrap();
        let src = temp.path().join("src");
        let dst = temp.path().join("dst");
        let nested = src.join("--Users-alice-project-work-aivo--");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        tokio::fs::write(nested.join("old.jsonl"), "{\"type\":\"session\"}\n")
            .await
            .unwrap();
        tokio::fs::write(nested.join("ignore.txt"), "nope")
            .await
            .unwrap();

        super::copy_pi_session_jsonl_tree(&src, &dst).await;

        let copied =
            tokio::fs::read_to_string(dst.join("--Users-alice-project-work-aivo--/old.jsonl"))
                .await
                .unwrap();
        assert_eq!(copied, "{\"type\":\"session\"}\n");
        assert!(
            !dst.join("--Users-alice-project-work-aivo--/ignore.txt")
                .exists()
        );
    }

    #[tokio::test]
    async fn seed_pi_sessions_exposes_prior_history_to_temp_agent_dir() {
        let temp = tempfile::tempdir().unwrap();
        let real_sessions = temp.path().join("real-sessions");
        let project_dir = real_sessions.join("--Users-alice-project-work-aivo--");
        tokio::fs::create_dir_all(&project_dir).await.unwrap();
        tokio::fs::write(project_dir.join("prior.jsonl"), "{\"type\":\"session\"}\n")
            .await
            .unwrap();

        let temp_agent = temp.path().join("temp-agent");
        tokio::fs::create_dir_all(&temp_agent).await.unwrap();

        super::seed_pi_sessions(Some(real_sessions), &temp_agent).await;

        let visible = tokio::fs::read_to_string(
            temp_agent.join("sessions/--Users-alice-project-work-aivo--/prior.jsonl"),
        )
        .await
        .unwrap();
        assert_eq!(visible, "{\"type\":\"session\"}\n");
    }

    fn aivo_models_only(api_key: &str) -> String {
        serde_json::json!({
            "providers": {
                "aivo": {
                    "baseUrl": "https://example.invalid",
                    "apiKey": api_key,
                    "api": "openai-completions",
                    "models": [{ "id": "m", "name": "m" }]
                }
            }
        })
        .to_string()
    }

    fn pi_env(models: &str) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("AIVO_PI_MODELS_JSON".to_string(), models.to_string());
        env
    }

    fn temp_agent_dir_from(env: &HashMap<String, String>) -> std::path::PathBuf {
        std::path::PathBuf::from(
            env.get("PI_CODING_AGENT_DIR")
                .expect("PI_CODING_AGENT_DIR set"),
        )
    }

    /// Launches `write_pi_agent_dir_with_real` against a fresh `real` tempdir
    /// and returns the temp dir handle plus the resolved agent path.
    async fn launch_pi_agent(
        real: Option<&std::path::Path>,
    ) -> (HashMap<String, String>, std::path::PathBuf) {
        let mut env = pi_env(&aivo_models_only("k"));
        super::write_pi_agent_dir_with_real(&mut env, None, real)
            .await
            .unwrap();
        let agent = temp_agent_dir_from(&env);
        (env, agent)
    }

    #[tokio::test]
    async fn write_pi_agent_dir_symlinks_settings_and_auth() {
        // Mid-session writes must reach the real file, not vanish with the temp dir.
        let real = tempfile::tempdir().unwrap();
        let real_settings = "{\"packages\":[\"pi-subagents\"],\"defaultThinkingLevel\":\"high\"}";
        let real_auth = "{\"openai\":\"sk-real\"}";
        tokio::fs::write(real.path().join("settings.json"), real_settings)
            .await
            .unwrap();
        tokio::fs::write(real.path().join("auth.json"), real_auth)
            .await
            .unwrap();

        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("settings.json"))
                .await
                .unwrap(),
            real_settings
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("auth.json"))
                .await
                .unwrap(),
            real_auth
        );

        let updated = "{\"packages\":[\"pi-subagents\",\"pi-newpkg\"]}";
        tokio::fs::write(agent.join("settings.json"), updated)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            updated
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_persists_first_time_writes_back_to_real_home() {
        // First-time Pi use: real ~/.pi/agent/ exists, settings.json doesn't yet.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            "{}"
        );

        let installed = "{\"packages\":[\"pi-newpkg\"]}";
        tokio::fs::write(agent.join("settings.json"), installed)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            installed
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_creates_writable_bin_on_first_time_use() {
        // First-time use: real agent dir exists but has no bin/ yet.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        let bin = agent.join("bin");
        assert!(bin.is_dir(), "temp agent bin/ should exist");
        tokio::fs::write(bin.join("fd"), b"#!/bin/sh\n")
            .await
            .unwrap();
        assert!(bin.join("fd").is_file());
    }

    #[tokio::test]
    async fn write_pi_agent_dir_aivo_only_models() {
        // User's custom providers must NOT leak — otherwise --model anthropic/foo
        // silently bypasses the aivo key.
        let real = tempfile::tempdir().unwrap();
        let real_models = serde_json::json!({
            "providers": {
                "user-anthropic": { "baseUrl": "https://anthropic.example", "api": "anthropic-messages" }
            }
        })
        .to_string();
        tokio::fs::write(real.path().join("models.json"), &real_models)
            .await
            .unwrap();

        let mut env = pi_env(&aivo_models_only("fresh-key"));
        super::write_pi_agent_dir_with_real(&mut env, None, Some(real.path()))
            .await
            .unwrap();
        let agent = temp_agent_dir_from(&env);

        let written: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(agent.join("models.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(written["providers"]["aivo"].is_object());
        assert_eq!(written["providers"]["aivo"]["apiKey"], "fresh-key");
        assert!(written["providers"]["user-anthropic"].is_null());
    }

    #[tokio::test]
    async fn write_pi_agent_dir_persists_first_time_git_package_install() {
        // git-sourced packages land under <agentDir>/git/<host>/<path>/ — must
        // survive temp dir cleanup even on fresh ~/.pi/agent.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert!(real.path().join("git").is_dir());

        let installed = agent.join("git/github.com/example/pkg");
        tokio::fs::create_dir_all(&installed).await.unwrap();
        tokio::fs::write(installed.join("package.json"), "{\"name\":\"pkg\"}")
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read_to_string(real.path().join("git/github.com/example/pkg/package.json"))
                .await
                .unwrap(),
            "{\"name\":\"pkg\"}"
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_persists_first_time_npm_package_install() {
        // npm packages land under <agentDir>/npm/; without the link Pi
        // re-installs them every launch (issue #8). Like git packages, an
        // install into the temp dir must persist back to the real home.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert!(real.path().join("npm").is_dir());

        let installed = agent.join("npm/node_modules/pi-newpkg");
        tokio::fs::create_dir_all(&installed).await.unwrap();
        tokio::fs::write(installed.join("package.json"), "{\"name\":\"pi-newpkg\"}")
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read_to_string(real.path().join("npm/node_modules/pi-newpkg/package.json"))
                .await
                .unwrap(),
            "{\"name\":\"pi-newpkg\"}"
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_links_user_customization() {
        let real = tempfile::tempdir().unwrap();
        for d in ["rules", "tools", "prompts", "themes", "git", "npm"] {
            tokio::fs::create_dir_all(real.path().join(d))
                .await
                .unwrap();
        }
        tokio::fs::write(real.path().join("rules/lean-ctx.md"), "rule body")
            .await
            .unwrap();
        tokio::fs::write(real.path().join("mcp.json"), "{\"servers\":{}}")
            .await
            .unwrap();

        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("rules/lean-ctx.md"))
                .await
                .unwrap(),
            "rule body"
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("mcp.json"))
                .await
                .unwrap(),
            "{\"servers\":{}}"
        );
        for d in ["tools", "prompts", "themes", "git", "npm"] {
            assert!(agent.join(d).is_dir(), "{d} missing");
        }

        // mcp.json must persist mid-session writes back to the real home
        // — same persistence guarantee as settings.json. Regressions that
        // copy instead of link (e.g., dropping the symlink/hard-link
        // chain) would silently break `pi mcp add` during aivo sessions.
        let updated = "{\"servers\":{\"new\":{\"command\":\"x\"}}}";
        tokio::fs::write(agent.join("mcp.json"), updated)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("mcp.json"))
                .await
                .unwrap(),
            updated
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_falls_back_when_no_real_home() {
        let (_env, agent) = launch_pi_agent(None).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("settings.json"))
                .await
                .unwrap(),
            "{}"
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("auth.json"))
                .await
                .unwrap(),
            "{}"
        );
        let models: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(agent.join("models.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(models["providers"]["aivo"].is_object());
    }

    #[tokio::test]
    async fn migrate_v2_folds_scalars_into_per_tool_routes() {
        use crate::services::session_store::{
            CURRENT_ROUTING_SCHEMA_VERSION, ClaudeProviderProtocol, SessionStore,
        };

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.anthropic.com",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk",
            )
            .await
            .unwrap();
        store
            .set_key_claude_path_variant(&key_id, Some("stripped".to_string()))
            .await
            .unwrap();
        store
            .set_key_responses_api_supported(&key_id, Some(false))
            .await
            .unwrap();
        // Force a pre-v2 key so the migration runs.
        store
            .set_key_routing_schema_version(&key_id, 0)
            .await
            .unwrap();
        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        super::migrate_routing_schema_for_key(&store, &mut key).await;

        // In-memory key: scalars folded into per-tool "" defaults and cleared.
        assert!(key.claude_protocol.is_none());
        assert!(key.responses_api_supported.is_none());
        assert!(key.claude_path_variant.is_none());
        assert_eq!(key.routing_schema_version, CURRENT_ROUTING_SCHEMA_VERSION);
        assert_eq!(key.protocol_routes["claude"][""].protocol, "anthropic");
        assert_eq!(key.protocol_routes["claude"][""].path_variant, "stripped");
        assert_eq!(key.protocol_routes["codex"][""].protocol, "openai");

        // Persisted to disk likewise; the scalar fields no longer serialize.
        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert!(reloaded.claude_protocol.is_none());
        assert_eq!(reloaded.protocol_routes["claude"][""].protocol, "anthropic");
        assert_eq!(reloaded.protocol_routes["codex"][""].protocol, "openai");
        assert_eq!(
            reloaded.routing_schema_version,
            CURRENT_ROUTING_SCHEMA_VERSION
        );
    }

    #[tokio::test]
    async fn migrate_v2_is_idempotent_when_already_current() {
        use crate::services::session_store::{CURRENT_ROUTING_SCHEMA_VERSION, SessionStore};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("test", "https://openrouter.ai/api/v1", None, "sk")
            .await
            .unwrap();
        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(key.routing_schema_version, CURRENT_ROUTING_SCHEMA_VERSION);
        super::migrate_routing_schema_for_key(&store, &mut key).await;
        assert!(key.protocol_routes.is_empty());
    }

    #[tokio::test]
    async fn persist_merges_confirmed_per_model_route() {
        use crate::services::route_cache::RouteCache;
        use crate::services::session_store::SessionStore;

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("test", "https://opencode-go.example/v1", None, "sk")
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let cache = Arc::new(RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            BTreeMap::new(),
        ));
        // qwen confirmed on the tool-native Anthropic route.
        cache.resolve("qwen3.7-max").confirm();
        super::persist_runtime_discoveries(&store, &key, Some(cache), None).await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.protocol_routes["claude"]["qwen3.7-max"].protocol,
            "anthropic"
        );
    }

    #[tokio::test]
    async fn persist_skips_unconfirmed_route() {
        use crate::services::provider_protocol::{PathVariant, encode_route};
        use crate::services::route_cache::RouteCache;
        use crate::services::session_store::SessionStore;
        use std::sync::atomic::Ordering;

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("test", "https://gw.example/v1", None, "sk-bad")
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let cache = Arc::new(RouteCache::new(
            "claude",
            ProviderProtocol::Anthropic,
            BTreeMap::new(),
        ));
        // Route switched but never confirmed (e.g. a bad-key session) — must not persist.
        let slot = cache.resolve("m");
        slot.route_atom().store(
            encode_route(ProviderProtocol::Openai, PathVariant::Default),
            Ordering::Relaxed,
        );
        super::persist_runtime_discoveries(&store, &key, Some(cache), None).await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert!(reloaded.protocol_routes.is_empty());
    }

    #[tokio::test]
    async fn persist_writes_learned_requires_reasoning() {
        use crate::services::session_store::SessionStore;
        use std::sync::atomic::AtomicBool;

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol("test", "https://gw.example/v1", None, "sk")
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let learned = Arc::new(AtomicBool::new(true));
        super::persist_runtime_discoveries(&store, &key, None, Some(learned)).await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.requires_reasoning_content, Some(true));
    }
}
