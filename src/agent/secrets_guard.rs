//! Two guards over the agent's file access: reading a known secret-bearing path is
//! permission-gated, and key-shaped strings are redacted from every tool result before
//! it enters the transcript (reusing the `share_redact` scanner).

use std::path::Path;

use serde_json::Value;

use crate::services::share_redact::{self, RedactCtx};

/// Whether `path` names a file that usually holds credentials. Narrow on purpose (the
/// redaction pass is the catch-all backstop) so ordinary files don't nag.
pub fn is_secret_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let by_name = name == ".env"
        || name.starts_with(".env.") // .env.local, .env.production
        || name == ".npmrc"
        || name == ".netrc"
        || name == ".pgpass"
        || name == "id_rsa"
        || name == "id_dsa"
        || name == "id_ecdsa"
        || name == "id_ed25519";
    let by_ext = [".pem", ".key", ".pfx", ".p12", ".keystore", ".jks"]
        .iter()
        .any(|ext| name.ends_with(ext));
    let full = path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    let by_store = full.contains("/.aws/credentials")
        || full.contains("/.gnupg/")
        || full.contains("/.config/gcloud/");
    by_name || by_ext || by_store
}

/// Whether a `read_file` call targets a secret path (so the gate confirms it). A
/// `run_bash` `cat` of a secret is caught by [`redact_for_model`] on the way back.
pub fn read_targets_secret(tool: &str, args: &Value, cwd: &Path) -> bool {
    if tool != "read_file" {
        return false;
    }
    match args.get("path").and_then(Value::as_str) {
        Some(path) => is_secret_path(&crate::agent::tools::resolve(cwd, path)),
        None => false,
    }
}

/// Redact key-shaped strings from tool output before it reaches the model. `RedactCtx`
/// has no home dir, so paths stay literal (the `~` rewrite is share-only). Returns the
/// redacted text and the mask count.
pub fn redact_for_model(text: &str) -> (String, usize) {
    let mut hits = std::collections::HashMap::new();
    let out = share_redact::scan_text(text, &RedactCtx::default(), &mut hits);
    (out, hits.values().sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn is_secret_path_flags_credential_files_only() {
        for p in [
            "/home/u/.env",
            "/home/u/.env.production",
            "app/.npmrc",
            "/home/u/.ssh/id_ed25519",
            "certs/server.pem",
            "keys/private.key",
            "/home/u/.aws/credentials",
            "/home/u/.gnupg/secring.gpg",
        ] {
            assert!(is_secret_path(Path::new(p)), "{p} should be secret");
        }
        for p in [
            "src/main.rs",
            "README.md",
            "environment.yml", // not `.env`
            "/home/u/.ssh/config",
            "package.json",
        ] {
            assert!(!is_secret_path(Path::new(p)), "{p} should not be secret");
        }
    }

    #[test]
    fn read_targets_secret_only_for_read_file() {
        let cwd = PathBuf::from("/work");
        assert!(read_targets_secret(
            "read_file",
            &json!({"path": ".env"}),
            &cwd
        ));
        assert!(!read_targets_secret(
            "write_file",
            &json!({"path": ".env"}),
            &cwd
        ));
        assert!(!read_targets_secret(
            "read_file",
            &json!({"path": "src/main.rs"}),
            &cwd
        ));
    }

    #[test]
    fn redact_for_model_masks_catted_private_key() {
        let (out, hits) = redact_for_model(
            "$ cat ~/.ssh/id_ed25519\n-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ==\n-----END OPENSSH PRIVATE KEY-----\n",
        );
        assert!(out.contains("<redacted:private_key>"), "{out}");
        assert!(!out.contains("b3BlbnNzaC1rZXktdjEA"));
        assert!(hits >= 1);
    }

    #[test]
    fn redact_for_model_masks_secrets_but_keeps_paths() {
        let (out, hits) = redact_for_model(
            "read /Users/me/app/.env\nAWS_KEY=AKIAIOSFODNN7EXAMPLE\nOPENAI=sk-AAAAAAAAAAAAAAAAAAAAAAAA",
        );
        assert!(hits >= 2, "expected redactions, got {hits}: {out}");
        assert!(!out.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!out.contains("sk-AAAAAAAAAAAAAAAAAAAAAAAA"));
        assert!(
            out.contains("/Users/me/app/.env"),
            "path was rewritten: {out}"
        );
    }
}
