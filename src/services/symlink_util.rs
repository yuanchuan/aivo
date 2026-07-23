//! Symlink helpers for the codex/gemini/pi home shadows: link the user's real
//! state into a temp dir so the spawned CLI reads it and writes persist back.
//!
//! Windows plain symlinks need Developer Mode, so we use no-privilege,
//! write-through equivalents: a junction for dirs, a hard link for files (each
//! degrading to a Dev-Mode symlink, then a one-way copy). `remove_dir_all` is
//! reparse-point aware, so shadow teardown deletes the junction, not the real
//! home (pinned by `symlink_dir_removes_junction_not_target`).

use std::path::Path;

#[cfg(unix)]
pub async fn symlink_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    tokio::fs::symlink(src, dst).await
}

#[cfg(unix)]
pub async fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    tokio::fs::symlink(src, dst).await
}

#[cfg(windows)]
pub async fn symlink_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    let src = src.to_path_buf();
    let dst = dst.to_path_buf();
    tokio::task::spawn_blocking(move || link_dir_blocking(&src, &dst))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
}

#[cfg(windows)]
pub async fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    let src = src.to_path_buf();
    let dst = dst.to_path_buf();
    tokio::task::spawn_blocking(move || link_file_blocking(&src, &dst))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
}

/// junction (write-through) → Dev-Mode symlink → one-way copy (cross-volume).
#[cfg(windows)]
fn link_dir_blocking(src: &Path, dst: &Path) -> std::io::Result<()> {
    let junction_err = match junction::create(src, dst) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    if std::os::windows::fs::symlink_dir(src, dst).is_ok() {
        return Ok(());
    }
    copy_dir_recursive(src, dst).map_err(|copy_err| {
        std::io::Error::other(format!(
            "junction ({junction_err}) and symlink failed; copy fallback failed: {copy_err}"
        ))
    })
}

/// hard link (write-through) → Dev-Mode symlink → one-way copy.
#[cfg(windows)]
fn link_file_blocking(src: &Path, dst: &Path) -> std::io::Result<()> {
    if std::fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    if std::os::windows::fs::symlink_file(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst).map(|_| ())
}

#[cfg(windows)]
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub async fn symlink_dir(_src: &Path, _dst: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("symlink unsupported"))
}

#[cfg(not(any(unix, windows)))]
pub async fn symlink_file(_src: &Path, _dst: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("symlink unsupported"))
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    #[tokio::test]
    async fn symlink_dir_junction_is_write_through() {
        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("seed.txt"), b"seed").unwrap();
        let shadow = root.path().join("shadow"); // must not pre-exist

        symlink_dir(&real, &shadow).await.unwrap();

        assert_eq!(std::fs::read(shadow.join("seed.txt")).unwrap(), b"seed");
        std::fs::write(shadow.join("new.txt"), b"written").unwrap();
        assert_eq!(std::fs::read(real.join("new.txt")).unwrap(), b"written");
        // Junction reports as a symlink, so replace_with_symlink_dir's idempotency holds.
        assert!(
            std::fs::symlink_metadata(&shadow)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[tokio::test]
    async fn symlink_dir_removes_junction_not_target() {
        // Data-loss guard: shadow teardown must delete the junction, not recurse in.
        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("precious.txt"), b"do not delete").unwrap();
        let shadow_root = root.path().join("shadow");
        std::fs::create_dir(&shadow_root).unwrap();
        let link = shadow_root.join("sessions");

        symlink_dir(&real, &link).await.unwrap();
        std::fs::remove_dir_all(&shadow_root).unwrap();

        assert!(!shadow_root.exists());
        assert_eq!(
            std::fs::read(real.join("precious.txt")).unwrap(),
            b"do not delete",
            "remove_dir_all must not follow the junction into the real home"
        );
    }

    #[tokio::test]
    async fn symlink_file_hard_link_is_write_through() {
        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("history.jsonl");
        std::fs::write(&real, b"line1\n").unwrap();
        let shadow = root.path().join("shadow-history.jsonl");

        symlink_file(&real, &shadow).await.unwrap();

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&shadow)
            .unwrap();
        f.write_all(b"line2\n").unwrap();
        drop(f);
        assert_eq!(std::fs::read(&real).unwrap(), b"line1\nline2\n");
    }
}
