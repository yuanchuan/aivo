use aes::Aes256;
use aes_gcm::{
    AesGcm,
    aead::{Aead, KeyInit, consts::U16, generic_array::GenericArray},
};
use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::Utc;
use pbkdf2::pbkdf2_hmac;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
/**
 * SessionStore service for managing credential persistence.
 * Stores credentials in ~/.config/aivo/config.json with AES-256-GCM encryption.
 */
use std::path::PathBuf;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::errors::{CLIError, ErrorCategory};
use crate::services::system_env;

/// Marker to identify encrypted values
pub const ENCRYPTION_MARKER: &str = "enc:";

/// Serde module for serializing/deserializing Zeroizing<String> as regular String
mod zeroizing_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use zeroize::Zeroizing;

    pub fn serialize<S>(value: &Zeroizing<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value.as_str())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Zeroizing<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Zeroizing::new(s))
    }
}

mod encrypted_chat_messages {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::{StoredChatMessage, decrypt, encrypt};

    pub fn serialize<S>(value: &[StoredChatMessage], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let json = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
        let encrypted = encrypt(&json).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&encrypted)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<StoredChatMessage>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let decrypted = decrypt(&value).map_err(serde::de::Error::custom)?;
        serde_json::from_str(&decrypted).map_err(serde::de::Error::custom)
    }
}

const IV_LENGTH: usize = 16;
const SALT_LENGTH: usize = 32;
const KEY_LENGTH: usize = 32;
const KEY_ID_LENGTH: usize = 4;
const KEY_ID_ALPHABET: &[u8] = b"23456789abcdefghijkmnpqrstuvwxyz";

/// Wrapper for encryption keys that automatically zeroizes on drop
#[derive(Zeroize, ZeroizeOnDrop)]
struct SecretKey([u8; KEY_LENGTH]);

impl SecretKey {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

// Use much lower iterations in tests for speed (PBKDF2 is still secure with lower iterations for testing)
#[cfg(any(test, feature = "test-fast-crypto"))]
const ITERATIONS: u32 = 100;
#[cfg(not(any(test, feature = "test-fast-crypto")))]
const ITERATIONS: u32 = 100_000;

/// API key stored on user's machine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeProviderProtocol {
    Anthropic,
    Openai,
    Google,
}

impl ClaudeProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Google => "google",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GeminiProviderProtocol {
    Google,
    Openai,
    Anthropic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OpenAICompatibilityMode {
    Direct,
    Router,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(
        rename = "claudeProtocol",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub claude_protocol: Option<ClaudeProviderProtocol>,
    #[serde(
        rename = "geminiProtocol",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub gemini_protocol: Option<GeminiProviderProtocol>,
    #[serde(rename = "codexMode", default, skip_serializing_if = "Option::is_none")]
    pub codex_mode: Option<OpenAICompatibilityMode>,
    #[serde(
        rename = "opencodeMode",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub opencode_mode: Option<OpenAICompatibilityMode>,
    #[serde(with = "zeroizing_string")]
    pub key: Zeroizing<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

impl ApiKey {
    pub fn new_with_protocol(
        id: String,
        name: String,
        base_url: String,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: String,
    ) -> Self {
        Self {
            id,
            name,
            base_url,
            claude_protocol,
            gemini_protocol: None,
            codex_mode: None,
            opencode_mode: None,
            key: Zeroizing::new(key),
            created_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn display_name(&self) -> &str {
        if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectoryStartRecord {
    #[serde(rename = "keyId")]
    pub key_id: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct UsageCounter {
    #[serde(default, skip_serializing_if = "is_zero")]
    pub selections: u64,
    #[serde(rename = "promptTokens", default, skip_serializing_if = "is_zero")]
    pub prompt_tokens: u64,
    #[serde(rename = "completionTokens", default, skip_serializing_if = "is_zero")]
    pub completion_tokens: u64,
    #[serde(rename = "totalTokens", default, skip_serializing_if = "is_zero")]
    pub total_tokens: u64,
}

impl UsageCounter {
    fn add_tokens(&mut self, prompt_tokens: u64, completion_tokens: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
        self.total_tokens = self
            .total_tokens
            .saturating_add(prompt_tokens.saturating_add(completion_tokens));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct UsageStats {
    #[serde(rename = "totalSelections", default, skip_serializing_if = "is_zero")]
    pub total_selections: u64,
    #[serde(rename = "totalPromptTokens", default, skip_serializing_if = "is_zero")]
    pub total_prompt_tokens: u64,
    #[serde(
        rename = "totalCompletionTokens",
        default,
        skip_serializing_if = "is_zero"
    )]
    pub total_completion_tokens: u64,
    #[serde(rename = "totalTokens", default, skip_serializing_if = "is_zero")]
    pub total_tokens: u64,
    #[serde(
        rename = "keyUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub key_usage: HashMap<String, UsageCounter>,
    #[serde(
        rename = "toolCounts",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub tool_counts: HashMap<String, u64>,
    #[serde(
        rename = "modelUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub model_usage: HashMap<String, UsageCounter>,
}

impl UsageStats {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    fn record_selection(&mut self, key_id: &str, tool: &str, model: Option<&str>) {
        self.total_selections = self.total_selections.saturating_add(1);
        let key_stats = self.key_usage.entry(key_id.to_string()).or_default();
        key_stats.selections = key_stats.selections.saturating_add(1);
        let tool_count = self.tool_counts.entry(tool.to_string()).or_default();
        *tool_count = tool_count.saturating_add(1);

        if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
            let model_stats = self.model_usage.entry(model.to_string()).or_default();
            model_stats.selections = model_stats.selections.saturating_add(1);
        }
    }

    fn record_tokens(
        &mut self,
        key_id: &str,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) {
        self.total_prompt_tokens = self.total_prompt_tokens.saturating_add(prompt_tokens);
        self.total_completion_tokens = self
            .total_completion_tokens
            .saturating_add(completion_tokens);
        self.total_tokens = self
            .total_tokens
            .saturating_add(prompt_tokens.saturating_add(completion_tokens));

        self.key_usage
            .entry(key_id.to_string())
            .or_default()
            .add_tokens(prompt_tokens, completion_tokens);

        if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
            self.model_usage
                .entry(model.to_string())
                .or_default()
                .add_tokens(prompt_tokens, completion_tokens);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatSessionState {
    #[serde(rename = "sessionId", default = "default_chat_session_id")]
    pub session_id: String,
    #[serde(rename = "keyId")]
    pub key_id: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub cwd: String,
    pub model: String,
    #[serde(with = "encrypted_chat_messages")]
    pub messages: Vec<StoredChatMessage>,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn default_chat_session_id() -> String {
    "legacy".to_string()
}

fn chat_session_map_key(key_id: &str, cwd: &str, session_id: &str) -> String {
    format!("{key_id}::{cwd}::{session_id}")
}

fn remove_runtime_state_for_key(config: &mut StoredConfig, key_id: &str) {
    config.chat_models.remove(key_id);
    config
        .directory_starts
        .retain(|_, record| record.key_id != key_id);
    config
        .chat_sessions
        .retain(|_, session| session.key_id != key_id);
}

/// Stored configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredConfig {
    #[serde(rename = "api_keys", default)]
    pub api_keys: Vec<ApiKey>,
    #[serde(rename = "active_key_id")]
    pub active_key_id: Option<String>,
    #[serde(
        rename = "chat_models",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub chat_models: HashMap<String, String>,
    #[serde(
        rename = "directory_starts",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub directory_starts: HashMap<String, DirectoryStartRecord>,
    #[serde(
        rename = "stats",
        default,
        skip_serializing_if = "UsageStats::is_empty"
    )]
    pub stats: UsageStats,
    #[serde(
        rename = "chat_sessions",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub chat_sessions: HashMap<String, ChatSessionState>,
}

impl Default for StoredConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StoredConfig {
    pub fn new() -> Self {
        Self {
            api_keys: Vec::new(),
            active_key_id: None,
            chat_models: HashMap::new(),
            directory_starts: HashMap::new(),
            stats: UsageStats::default(),
            chat_sessions: HashMap::new(),
        }
    }
}

/// Derives an encryption key from machine-specific information.
/// Uses username and home directory to create a consistent key per machine.
fn derive_key() -> SecretKey {
    let username = system_env::username().unwrap_or_default();
    let homedir: String = system_env::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let machine_data = format!("{}:{}", username, homedir);

    // Create a static salt derived from the machine data
    let mut hasher = Sha256::new();
    hasher.update(b"aivo-salt");
    hasher.update(machine_data.as_bytes());
    let salt_full = hasher.finalize();
    let salt = &salt_full[..SALT_LENGTH];

    let iterations = ITERATIONS;
    let mut key = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(machine_data.as_bytes(), salt, iterations, &mut key);

    SecretKey(key)
}

/// Checks if a string is encrypted
pub fn is_encrypted(value: &str) -> bool {
    value.starts_with(ENCRYPTION_MARKER)
}

type Aes256Gcm16 = AesGcm<Aes256, U16, U16>;

/// Encrypts a plaintext string
pub fn encrypt(plaintext: &str) -> Result<String> {
    if plaintext.is_empty() {
        return Ok(plaintext.to_string());
    }

    // Don't double-encrypt
    if is_encrypted(plaintext) {
        return Ok(plaintext.to_string());
    }

    let key = derive_key();
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    let mut iv = [0u8; IV_LENGTH];
    rand::thread_rng().fill_bytes(&mut iv);

    let nonce = GenericArray::from_slice(&iv);

    // Encrypt
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    // Combine: IV + ciphertext (which includes auth tag in aes-gcm)
    let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
    combined.extend_from_slice(&iv);
    combined.extend_from_slice(&ciphertext);

    // Encode as base64 with marker
    Ok(format!("{}{}", ENCRYPTION_MARKER, BASE64.encode(&combined)))
}

/// Decrypts an encrypted string
pub fn decrypt(encrypted_data: &str) -> Result<String> {
    if encrypted_data.is_empty() {
        return Ok(encrypted_data.to_string());
    }

    if !is_encrypted(encrypted_data) {
        return Err(anyhow::anyhow!("Invalid encrypted data: missing marker"));
    }

    let key = derive_key();
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    // Decode from base64
    let data = BASE64
        .decode(&encrypted_data[ENCRYPTION_MARKER.len()..])
        .map_err(|e| anyhow::anyhow!("Base64 decode failed: {}", e))?;

    if data.len() < IV_LENGTH {
        return Err(anyhow::anyhow!("Invalid encrypted data: too short"));
    }

    // Extract IV and ciphertext
    let iv = &data[..IV_LENGTH];
    let ciphertext = &data[IV_LENGTH..];

    let nonce = GenericArray::from_slice(iv);

    // Decrypt
    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("Decryption failed - key may be from different machine"))?;

    String::from_utf8(plaintext)
        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in decrypted data: {}", e))
}

/// SessionStore manages API key persistence in ~/.config/aivo/config.json
#[derive(Debug, Clone)]
pub struct SessionStore {
    config_path: PathBuf,
    config_dir: PathBuf,
}

#[cfg(unix)]
struct ConfigLockGuard {
    _file: std::fs::File,
}

#[cfg(windows)]
struct ConfigLockGuard {
    _file: std::fs::File,
}

#[cfg(not(any(unix, windows)))]
struct ConfigLockGuard;

#[cfg(unix)]
impl Drop for ConfigLockGuard {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        // SAFETY: the file descriptor remains valid for the lifetime of the guard.
        unsafe {
            libc::flock(self._file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(windows)]
impl Drop for ConfigLockGuard {
    fn drop(&mut self) {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::UnlockFile;

        // SAFETY: the handle stays valid for the guard lifetime; UnlockFile is safe to call
        // on a handle previously locked with LockFileEx.
        unsafe {
            UnlockFile(self._file.as_raw_handle(), 0, 0, u32::MAX, u32::MAX);
        }
    }
}

impl SessionStore {
    pub fn new() -> Self {
        let config_dir = system_env::home_dir()
            .map(|p| p.join(".config").join("aivo"))
            .unwrap_or_else(|| PathBuf::from(".config/aivo"));
        let config_path = config_dir.join("config.json");

        Self {
            config_path,
            config_dir,
        }
    }

    /// Creates a new SessionStore with a custom config path (for testing)
    #[allow(dead_code)]
    pub fn with_path(config_path: PathBuf) -> Self {
        let config_dir = config_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        Self {
            config_path,
            config_dir,
        }
    }

    fn lock_path(&self) -> PathBuf {
        self.config_dir.join("config.lock")
    }

    fn acquire_config_lock(&self) -> Result<ConfigLockGuard> {
        if !self.config_dir.as_os_str().is_empty() {
            std::fs::create_dir_all(&self.config_dir).with_context(|| {
                format!("Failed to create config directory: {:?}", self.config_dir)
            })?;
        }

        let lock_path = self.lock_path();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("Failed to open config lock file: {:?}", lock_path))?;

        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;

            loop {
                // SAFETY: the file descriptor stays open for the guard lifetime.
                let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
                if rc == 0 {
                    break;
                }

                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::Interrupted {
                    return Err(err).with_context(|| {
                        format!("Failed to acquire config lock: {:?}", lock_path)
                    });
                }
            }

            Ok(ConfigLockGuard { _file: file })
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = file;
            Ok(ConfigLockGuard)
        }

        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Foundation::BOOL;
            use windows_sys::Win32::Storage::FileSystem::{
                LockFileEx, LOCKFILE_EXCLUSIVE_LOCK,
            };

            let handle = file.as_raw_handle();
            let mut overlapped = unsafe { std::mem::zeroed() };
            // SAFETY: handle is valid; we own `file` for the guard's lifetime.
            let rc: BOOL = unsafe {
                LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, u32::MAX, u32::MAX, &mut overlapped)
            };
            if rc == 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("Failed to acquire config lock: {:?}", lock_path));
            }
            Ok(ConfigLockGuard { _file: file })
        }
    }

    /// Saves config to the config file.
    /// Uses atomic write (write to temp file then rename) to prevent corruption.
    #[allow(dead_code)]
    pub async fn save(&self, config: &StoredConfig) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        self.save_unlocked(config).await
    }

    async fn save_unlocked(&self, config: &StoredConfig) -> Result<()> {
        tokio::fs::create_dir_all(&self.config_dir)
            .await
            .with_context(|| format!("Failed to create config directory: {:?}", self.config_dir))?;

        // Encrypt only the api_keys vec (avoids cloning the full config which may
        // include large chat session history).
        let encrypted_keys = self.encrypt_keys(config)?;
        let mut serializable = serde_json::to_value(config).context("Failed to serialize config")?;
        serializable["api_keys"] =
            serde_json::to_value(&encrypted_keys).context("Failed to serialize encrypted keys")?;
        let data =
            serde_json::to_string_pretty(&serializable).context("Failed to serialize config")?;

        // Write to a temp file first, then atomically rename to prevent partial writes
        let tmp_path = self.config_path.with_extension("json.tmp");

        tokio::fs::write(&tmp_path, &data)
            .await
            .with_context(|| format!("Failed to write temp config file: {:?}", tmp_path))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = tokio::fs::metadata(&tmp_path).await?;
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o600);
            tokio::fs::set_permissions(&tmp_path, permissions).await?;
        }

        tokio::fs::rename(&tmp_path, &self.config_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to rename temp config file to {:?}",
                    self.config_path
                )
            })?;

        Ok(())
    }

    /// Loads config from the config file
    pub async fn load(&self) -> Result<StoredConfig> {
        self.load_unlocked().await
    }

    async fn load_unlocked(&self) -> Result<StoredConfig> {
        let parsed = self.load_raw().await?;
        self.decrypt_keys(&parsed)
    }

    /// Loads config without decrypting API key secrets (fast path for metadata-only reads).
    async fn load_raw(&self) -> Result<StoredConfig> {
        let data = match tokio::fs::read_to_string(&self.config_path).await {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(StoredConfig::new());
            }
            Err(e) => return Err(e.into()),
        };

        match serde_json::from_str(&data) {
            Ok(p) => Ok(p),
            Err(e) => Err(anyhow::anyhow!(
                "config file is corrupted and cannot be read: {e}"
            )),
        }
    }

    /// Adds a new API key with an optional explicit Claude protocol.
    pub async fn add_key_with_protocol(
        &self,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<String> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;

        let existing_ids: HashSet<String> = config.api_keys.iter().map(|k| k.id.clone()).collect();
        let id = generate_key_id(&existing_ids)?;

        config.api_keys.push(ApiKey::new_with_protocol(
            id.clone(),
            name.to_string(),
            base_url.to_string(),
            claude_protocol,
            key.to_string(),
        ));

        self.save_unlocked(&config).await?;
        Ok(id)
    }

    /// Gets all API keys
    pub async fn get_keys(&self) -> Result<Vec<ApiKey>> {
        Ok(self.load().await?.api_keys)
    }

    /// Gets a specific API key by ID
    #[allow(dead_code)]
    pub async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        let keys = self.get_keys().await?;
        Ok(keys.into_iter().find(|k| k.id == id))
    }

    /// Deletes an API key by ID
    pub async fn delete_key(&self, id: &str) -> Result<bool> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        let initial_len = config.api_keys.len();
        config.api_keys.retain(|k| k.id != id);

        if config.api_keys.len() < initial_len {
            if config.active_key_id.as_deref() == Some(id) {
                config.active_key_id = None;
            }
            remove_runtime_state_for_key(&mut config, id);
            self.save_unlocked(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Updates an existing API key's fields by ID. Returns false if not found.
    pub async fn update_key(
        &self,
        id: &str,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<bool> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        if let Some(entry) = config.api_keys.iter_mut().find(|k| k.id == id) {
            let base_url_changed = entry.base_url != base_url;
            entry.name = name.to_string();
            entry.base_url = base_url.to_string();
            entry.claude_protocol = claude_protocol;
            entry.key = Zeroizing::new(key.to_string());
            if base_url_changed {
                remove_runtime_state_for_key(&mut config, id);
            }
            self.save_unlocked(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Internal helper: load config, apply `f` to the key with `id`, save, and return whether
    /// the key was found.
    async fn update_key_field(&self, id: &str, f: impl FnOnce(&mut ApiKey)) -> Result<bool> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        if let Some(entry) = config.api_keys.iter_mut().find(|k| k.id == id) {
            f(entry);
            self.save_unlocked(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Persists the learned Claude protocol for an existing key.
    pub async fn set_key_claude_protocol(
        &self,
        id: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.claude_protocol = claude_protocol)
            .await
    }

    pub async fn set_key_gemini_protocol(
        &self,
        id: &str,
        gemini_protocol: Option<GeminiProviderProtocol>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.gemini_protocol = gemini_protocol)
            .await
    }

    pub async fn set_key_codex_mode(
        &self,
        id: &str,
        codex_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.codex_mode = codex_mode)
            .await
    }

    pub async fn set_key_opencode_mode(
        &self,
        id: &str,
        opencode_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.opencode_mode = opencode_mode)
            .await
    }

    /// Sets the currently active API key
    pub async fn set_active_key(&self, id: &str) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;

        if !config.api_keys.iter().any(|k| k.id == id) {
            return Err(CLIError::new(
                format!("Key {} not found", id),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see available keys"),
            )
            .into());
        }

        config.active_key_id = Some(id.to_string());
        self.save_unlocked(&config).await
    }

    /// Resolves an API key by ID or name.
    /// Tries exact ID match first, then name match.
    /// Returns an error if no match found or multiple names match.
    pub async fn resolve_key_by_id_or_name(&self, id_or_name: &str) -> Result<ApiKey> {
        let keys = self.get_keys().await?;

        // Try exact ID match first
        if let Some(key) = keys.iter().find(|k| k.id == id_or_name) {
            return Ok(key.clone());
        }

        // Try name match
        let name_matches: Vec<_> = keys.iter().filter(|k| k.name == id_or_name).collect();

        match name_matches.len() {
            0 => Err(CLIError::new(
                format!("API key \"{}\" not found", id_or_name),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see available keys"),
            )
            .into()),
            1 => Ok(name_matches[0].clone()),
            _ => Err(CLIError::new(
                format!(
                    "Multiple keys found with name \"{}\". Use the key ID instead.",
                    id_or_name
                ),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see key IDs"),
            )
            .into()),
        }
    }

    /// Gets the currently active API key
    pub async fn get_active_key(&self) -> Result<Option<ApiKey>> {
        let config = self.load().await?;

        match config.active_key_id {
            Some(ref id) => Ok(config.api_keys.into_iter().find(|k| k.id == *id)),
            None => Ok(None),
        }
    }

    /// Gets all keys and the active key ID without decrypting secrets.
    /// Use this for display-only paths (e.g., `aivo keys` list).
    pub async fn get_keys_and_active_id_info(&self) -> Result<(Vec<ApiKey>, Option<String>)> {
        let config = self.load_raw().await?;
        Ok((config.api_keys, config.active_key_id))
    }

    /// Gets the active key's display metadata (id, name, base_url) without decrypting secrets.
    /// Use this when the key value is not needed (e.g., help output).
    pub async fn get_active_key_info(&self) -> Result<Option<ApiKey>> {
        let config = self.load_raw().await?;

        match config.active_key_id {
            Some(ref id) => Ok(config.api_keys.into_iter().find(|k| k.id == *id)),
            None => Ok(None),
        }
    }

    /// Gets the config path
    #[allow(dead_code)]
    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    /// Gets the persisted chat model for a specific API key
    pub async fn get_chat_model(&self, key_id: &str) -> Result<Option<String>> {
        let config = self.load().await?;
        Ok(config.chat_models.get(key_id).cloned())
    }

    /// Saves the chat model for a specific API key
    pub async fn set_chat_model(&self, key_id: &str, model: &str) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        config
            .chat_models
            .insert(key_id.to_string(), model.to_string());
        self.save_unlocked(&config).await
    }

    pub async fn get_directory_start(&self, cwd: &str) -> Result<Option<DirectoryStartRecord>> {
        // Fast path: shared read — load() acquires and releases the lock internally.
        let config = self.load().await?;
        let Some(record) = config.directory_starts.get(cwd).cloned() else {
            return Ok(None);
        };

        let key_is_valid = config
            .api_keys
            .iter()
            .any(|key| key.id == record.key_id && key.base_url == record.base_url);
        if key_is_valid {
            return Ok(Some(record));
        }

        // Slow path: stale record — re-acquire exclusive lock, reload, remove, save.
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        config.directory_starts.remove(cwd);
        self.save_unlocked(&config).await?;
        Ok(None)
    }

    pub async fn set_directory_start(
        &self,
        cwd: &str,
        key_id: &str,
        base_url: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        config.directory_starts.insert(
            cwd.to_string(),
            DirectoryStartRecord {
                key_id: key_id.to_string(),
                base_url: base_url.to_string(),
                tool: tool.to_string(),
                model: model
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string),
                updated_at: Utc::now().to_rfc3339(),
            },
        );
        self.save_unlocked(&config).await
    }

    #[allow(dead_code)]
    pub async fn clear_directory_start(&self, cwd: &str) -> Result<bool> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        let removed = config.directory_starts.remove(cwd).is_some();
        if removed {
            self.save_unlocked(&config).await?;
        }
        Ok(removed)
    }

    pub async fn record_selection(
        &self,
        key_id: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        config.stats.record_selection(key_id, tool, model);
        self.save_unlocked(&config).await
    }

    pub async fn record_tokens(
        &self,
        key_id: &str,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        config
            .stats
            .record_tokens(key_id, model, prompt_tokens, completion_tokens);
        self.save_unlocked(&config).await
    }

    pub async fn get_chat_session(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
    ) -> Result<Option<ChatSessionState>> {
        Ok(self
            .list_chat_sessions(key_id, base_url, cwd)
            .await?
            .into_iter()
            .next())
    }

    pub async fn list_chat_sessions(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
    ) -> Result<Vec<ChatSessionState>> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        let key_is_valid = config
            .api_keys
            .iter()
            .any(|key| key.id == key_id && key.base_url == base_url);

        let mut dirty = false;
        let mut sessions = Vec::new();
        config.chat_sessions.retain(|_, session| {
            let matches = session.key_id == key_id && session.cwd == cwd;
            if !matches {
                return true;
            }

            let keep = key_is_valid && session.base_url == base_url;
            if keep {
                sessions.push(session.clone());
            } else {
                dirty = true;
            }
            keep
        });

        if dirty {
            self.save_unlocked(&config).await?;
        }

        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    pub async fn save_chat_session_with_id(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
        session_id: &str,
        model: &str,
        messages: &[StoredChatMessage],
    ) -> Result<()> {
        let _lock = self.acquire_config_lock()?;
        let mut config = self.load_unlocked().await?;
        let map_key = chat_session_map_key(key_id, cwd, session_id);
        config.chat_sessions.insert(
            map_key,
            ChatSessionState {
                session_id: session_id.to_string(),
                key_id: key_id.to_string(),
                base_url: base_url.to_string(),
                cwd: cwd.to_string(),
                model: model.to_string(),
                messages: messages.to_vec(),
                updated_at: Utc::now().to_rfc3339(),
            },
        );
        self.save_unlocked(&config).await
    }

    /// Encrypts API keys before saving
    /// Encrypts API keys before saving.
    /// Only clones the `api_keys` vec (not the full config) to keep allocation minimal.
    fn encrypt_keys(&self, config: &StoredConfig) -> Result<Vec<ApiKey>> {
        let mut encrypted_keys = config.api_keys.clone();
        for key in &mut encrypted_keys {
            if !is_encrypted(&key.key) {
                key.key = Zeroizing::new(encrypt(&key.key)?);
            }
        }
        Ok(encrypted_keys)
    }

    /// Decrypts API keys after loading
    fn decrypt_keys(&self, config: &StoredConfig) -> Result<StoredConfig> {
        let mut decrypted = config.clone();
        for key in &mut decrypted.api_keys {
            if is_encrypted(&key.key) {
                let plaintext = decrypt(&key.key)
                    .with_context(|| format!("failed to decrypt key '{}'", key.display_name()))?;
                key.key = Zeroizing::new(plaintext);
            }
        }
        Ok(decrypted)
    }
}

fn generate_key_id(existing_ids: &HashSet<String>) -> Result<String> {
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let id: String = (0..KEY_ID_LENGTH)
            .map(|_| {
                let idx = rng.gen_range(0..KEY_ID_ALPHABET.len());
                KEY_ID_ALPHABET[idx] as char
            })
            .collect();

        if !existing_ids.contains(&id) {
            return Ok(id);
        }
    }

    anyhow::bail!(
        "Failed to generate unique key ID after 1000 attempts. Consider removing unused keys."
    );
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_load_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let config = store.load().await.unwrap();
        assert!(config.api_keys.is_empty());
        assert!(config.active_key_id.is_none());
    }

    #[tokio::test]
    async fn test_key_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // Add a key
        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test123")
            .await
            .unwrap();
        assert_eq!(id.len(), 4);

        // Verify it was saved
        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "my-key");
        assert_eq!(keys[0].base_url, "http://localhost:8080");
        assert_eq!(keys[0].claude_protocol, None);

        // Set as active
        store.set_active_key(&id).await.unwrap();
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().id, id);

        // Delete the key
        assert!(store.delete_key(&id).await.unwrap());
        let keys = store.get_keys().await.unwrap();
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn test_key_encryption_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-secret-12345")
            .await
            .unwrap();

        // Verify the file contains encrypted key
        let file_content = tokio::fs::read_to_string(&config_path).await.unwrap();
        assert!(file_content.contains("enc:"));
        assert!(!file_content.contains("sk-secret-12345"));

        // Verify we can still read back the decrypted key
        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.key.as_str(), "sk-secret-12345");
    }

    #[tokio::test]
    async fn test_delete_active_key_clears_selection() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        // Delete the active key
        store.delete_key(&id).await.unwrap();

        // Active key should be cleared
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn test_resolve_key_by_id() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name(&id).await.unwrap();
        assert_eq!(resolved.id, id);
        assert_eq!(resolved.name, "my-key");
    }

    #[tokio::test]
    async fn test_resolve_key_by_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name("my-key").await.unwrap();
        assert_eq!(resolved.id, id);
    }

    #[tokio::test]
    async fn test_resolve_key_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let result = store.resolve_key_by_id_or_name("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_resolve_key_ambiguous_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        store
            .add_key_with_protocol("same-name", "http://localhost:8080", None, "sk-test1")
            .await
            .unwrap();
        store
            .add_key_with_protocol("same-name", "http://localhost:9090", None, "sk-test2")
            .await
            .unwrap();

        let result = store.resolve_key_by_id_or_name("same-name").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Multiple keys found")
        );
    }

    #[tokio::test]
    async fn test_load_corrupted_config_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(&config_path, b"not valid json {{{")
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        let result = store.load().await;
        assert!(result.is_err(), "expected Err on corrupted config, got Ok");
    }

    #[tokio::test]
    async fn test_load_returns_error_on_invalid_encrypted_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        // "enc:" prefix triggers decryption; the payload is not valid ciphertext
        let bad_config = r#"{"api_keys":[{"id":"aaaa","name":"test","baseUrl":"http://example.com","key":"enc:notvalidbase64!!!","createdAt":"2024-01-01T00:00:00Z"}],"active_key_id":null}"#;
        tokio::fs::write(&config_path, bad_config.as_bytes())
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        let result = store.load().await;
        assert!(
            result.is_err(),
            "expected Err on invalid encrypted key, got Ok"
        );
    }

    #[tokio::test]
    async fn test_update_key_fields() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("original", "http://localhost:8080", None, "sk-old")
            .await
            .unwrap();

        let updated = store
            .update_key(
                &id,
                "renamed",
                "https://new.example.com",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-new",
            )
            .await
            .unwrap();
        assert!(updated);

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.name, "renamed");
        assert_eq!(key.base_url, "https://new.example.com");
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
        assert_eq!(key.key.as_str(), "sk-new");
        assert_eq!(key.id, id);
    }

    #[test]
    fn test_api_key_display_name_falls_back_to_id() {
        let key = ApiKey::new_with_protocol(
            "1a2b".to_string(),
            String::new(),
            "https://example.com".to_string(),
            None,
            "sk-test".to_string(),
        );

        assert_eq!(key.display_name(), "1a2b");
    }

    #[tokio::test]
    async fn test_update_key_not_found_returns_false() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let updated = store
            .update_key("nonexistent", "name", "http://example.com", None, "sk-key")
            .await
            .unwrap();
        assert!(!updated);
    }

    #[tokio::test]
    async fn test_update_key_preserves_created_at() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        let before = store.get_key_by_id(&id).await.unwrap().unwrap();

        store
            .update_key(&id, "new-name", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        let after = store.get_key_by_id(&id).await.unwrap().unwrap();

        assert_eq!(before.created_at, after.created_at);
    }

    #[tokio::test]
    async fn test_directory_start_removed_when_key_base_url_changes() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store
            .set_directory_start(
                "/tmp/demo",
                &id,
                "http://localhost",
                "claude",
                Some("model-a"),
            )
            .await
            .unwrap();

        store
            .update_key(&id, "orig", "https://new.example.com", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .get_directory_start("/tmp/demo")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_record_stats_and_chat_session_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        store
            .record_selection(&id, "chat", Some("gpt-4o"))
            .await
            .unwrap();
        store
            .record_tokens(&id, Some("gpt-4o"), 10, 5)
            .await
            .unwrap();
        store
            .save_chat_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "legacy",
                "gpt-4o",
                &[StoredChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
            )
            .await
            .unwrap();

        let stats = store.load().await.unwrap().stats;
        assert_eq!(stats.total_selections, 1);
        assert_eq!(stats.total_tokens, 15);
        assert_eq!(stats.tool_counts.get("chat"), Some(&1));
        assert_eq!(
            stats
                .model_usage
                .get("gpt-4o")
                .map(|usage| usage.total_tokens),
            Some(15)
        );

        let session = store
            .get_chat_session(&id, "http://localhost", "/tmp/demo")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.session_id, "legacy");

        store
            .save_chat_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "session-2",
                "gpt-4o-mini",
                &[StoredChatMessage {
                    role: "user".to_string(),
                    content: "second".to_string(),
                }],
            )
            .await
            .unwrap();

        let sessions = store
            .list_chat_sessions(&id, "http://localhost", "/tmp/demo")
            .await
            .unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(
            sessions
                .iter()
                .any(|session| session.session_id == "session-2")
        );

        let raw = tokio::fs::read_to_string(&config_path).await.unwrap();
        assert!(!raw.contains("\"hello\""));
    }

    #[tokio::test]
    async fn test_clear_directory_start_returns_true_when_removed() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store
            .set_directory_start("/tmp/demo", &id, "http://localhost", "claude", None)
            .await
            .unwrap();

        assert!(store.clear_directory_start("/tmp/demo").await.unwrap());
        assert!(!store.clear_directory_start("/tmp/demo").await.unwrap());
    }

    #[tokio::test]
    async fn test_add_key_with_claude_protocol_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol(
                "minimax",
                "https://api.minimax.io/anthropic",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-test",
            )
            .await
            .unwrap();

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
    }

    #[tokio::test]
    async fn test_generated_key_id_excludes_ambiguous_characters() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert_eq!(id.len(), KEY_ID_LENGTH);
        assert!(!id.contains('0'));
        assert!(!id.contains('1'));
        assert!(!id.contains('l'));
        assert!(!id.contains('o'));
        assert!(id.chars().all(|c| KEY_ID_ALPHABET.contains(&(c as u8))));
    }

    #[tokio::test]
    async fn test_set_key_claude_protocol_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_claude_protocol(&id, Some(ClaudeProviderProtocol::Openai))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Openai));
    }

    #[tokio::test]
    async fn test_set_key_gemini_protocol_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_gemini_protocol(&id, Some(GeminiProviderProtocol::Google))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.gemini_protocol, Some(GeminiProviderProtocol::Google));
    }

    #[tokio::test]
    async fn test_set_key_codex_mode_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_codex_mode(&id, Some(OpenAICompatibilityMode::Router))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.codex_mode, Some(OpenAICompatibilityMode::Router));
    }

    #[tokio::test]
    async fn test_set_key_opencode_mode_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_opencode_mode(&id, Some(OpenAICompatibilityMode::Router))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.opencode_mode, Some(OpenAICompatibilityMode::Router));
    }

    #[tokio::test]
    async fn test_load_legacy_config_without_claude_protocol() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let plaintext = encrypt("sk-test").unwrap();
        let legacy_config = format!(
            r#"{{"api_keys":[{{"id":"aaaa","name":"legacy","baseUrl":"http://example.com","key":"{}","createdAt":"2024-01-01T00:00:00Z"}}],"active_key_id":"aaaa"}}"#,
            plaintext
        );
        tokio::fs::write(&config_path, legacy_config.as_bytes())
            .await
            .unwrap();

        let store = SessionStore::with_path(config_path);
        let key = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(key.name, "legacy");
        assert_eq!(key.claude_protocol, None);
        assert_eq!(key.key.as_str(), "sk-test");
    }

    // Tests moved from tests/encryption_test.rs
    #[test]
    fn test_encryption_format() {
        let plaintext = "test-api-key-12345";
        let encrypted = encrypt(plaintext).unwrap();

        // Should start with enc:
        assert!(encrypted.starts_with(ENCRYPTION_MARKER));

        // Should be base64 after marker
        let data = &encrypted[ENCRYPTION_MARKER.len()..];
        let decoded = BASE64.decode(data).unwrap();

        // Format: 16 byte IV + ciphertext (includes 16 byte auth tag in aes-gcm)
        // Minimum: 16 + 16 = 32 bytes, plus at least some ciphertext
        assert!(
            decoded.len() >= 32,
            "Expected at least 32 bytes (IV + auth tag), got {}",
            decoded.len()
        );
    }

    #[test]
    fn test_encryption_roundtrip() {
        let test_cases = [
            "simple-key",
            "key-with-special-chars-!@#$%",
            "sk-ant-api03-test123",
            "unicode-キー-测试",
        ];

        for plaintext in test_cases {
            let encrypted = encrypt(plaintext).unwrap();
            let decrypted = decrypt(&encrypted).unwrap();
            assert_eq!(decrypted, plaintext);
        }
    }

    #[test]
    fn test_is_encrypted_detection() {
        assert!(is_encrypted("enc:abc123"));
        assert!(!is_encrypted("plain-text"));
        assert!(!is_encrypted(""));
        assert!(!is_encrypted("enc"));
    }

    // Tests moved from tests/encryption_property.rs
    #[test]
    fn test_encryption_never_panics() {
        let inputs = [
            "a",
            "normal-key",
            "key-with-symbols!@#",
            "sk-test123456789",
            "unicode-キー-测试",
        ];

        for input in inputs {
            let encrypted = encrypt(input).expect("encryption should not fail");
            assert!(is_encrypted(&encrypted));

            let decrypted = decrypt(&encrypted).expect("decryption should not fail");
            assert_eq!(decrypted, input);
        }

        // Empty string special case - returns empty without encryption
        assert_eq!(encrypt("").unwrap(), "");
    }

    #[test]
    fn test_double_encryption_idempotent() {
        let key = "my-api-key";
        let encrypted1 = encrypt(key).unwrap();
        let encrypted2 = encrypt(&encrypted1).unwrap();

        // Double encryption should return the same value
        assert_eq!(encrypted1, encrypted2);
    }
}
