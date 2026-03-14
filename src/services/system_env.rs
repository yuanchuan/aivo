use std::path::PathBuf;

/// Best-effort user home directory lookup using standard environment variables.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                Some(PathBuf::from(format!(
                    "{}{}",
                    drive.to_string_lossy(),
                    path.to_string_lossy()
                )))
            })
    }

    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Best-effort current username lookup.
/// Tries the USER/USERNAME environment variable first, then falls back to the
/// OS user database via libc on Unix (unaffected by sudo USER overrides or
/// environments where USER is unset).
pub fn username() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERNAME").ok().filter(|s| !s.is_empty())
    }

    #[cfg(not(windows))]
    {
        if let Ok(user) = std::env::var("USER")
            && !user.is_empty()
        {
            return Some(user);
        }

        // Fall back to OS user database so key derivation remains consistent
        // when USER is unset (CI, containers, sudo -i, etc.).
        #[cfg(unix)]
        // SAFETY: getpwuid returns a pointer to static thread-local storage valid
        // until the next getpwuid call on this thread. We copy pw_name immediately.
        unsafe {
            let uid = libc::getuid();
            let passwd = libc::getpwuid(uid);
            if !passwd.is_null() {
                let name = std::ffi::CStr::from_ptr((*passwd).pw_name);
                if let Ok(s) = name.to_str()
                    && !s.is_empty()
                {
                    return Some(s.to_string());
                }
            }
        }

        None
    }
}

/// Expands a leading `~` to the user's home directory.
/// Returns the path unchanged (as a `PathBuf`) if expansion is not needed or not possible.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

/// Best-effort current working directory lookup with canonicalization when possible.
pub fn current_dir() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    std::fs::canonicalize(&cwd).ok().or(Some(cwd))
}

pub fn current_dir_string() -> Option<String> {
    current_dir().map(|path| path.to_string_lossy().to_string())
}

/// Returns a hardware-specific machine identifier.
/// - macOS: IOPlatformUUID
/// - Linux: /etc/machine-id
/// - Windows: HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid
pub fn machine_id() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Some(pos) = line.find("IOPlatformUUID")
                && let Some(start) = line[pos..].find('"').map(|i| pos + i + 1)
                && let Some(end) = line[start..].find('"').map(|i| start + i)
            {
                // Format: "IOPlatformUUID" = "XXXXXXXX-..."
                let uuid = line[start..end].trim().to_string();
                if !uuid.is_empty() {
                    return Some(uuid);
                }
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/machine-id")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("MachineGuid") {
                // Format: MachineGuid    REG_SZ    XXXXXXXX-...
                if let Some(guid) = line.split_whitespace().last() {
                    let guid = guid.trim().to_string();
                    if !guid.is_empty() {
                        return Some(guid);
                    }
                }
            }
        }
        None
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}
