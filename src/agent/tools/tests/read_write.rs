use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn resolve_expands_leading_tilde() {
    let Some(home) = crate::services::system_env::home_dir() else {
        return; // no HOME in this environment
    };
    let cwd = Path::new("/tmp/work/aivo");
    assert_eq!(resolve(cwd, "~"), home);
    assert_eq!(resolve(cwd, "~/.ssh"), home.join(".ssh"));
    assert_eq!(resolve(cwd, "~/a/b"), home.join("a/b"));
    #[cfg(windows)]
    assert_eq!(resolve(cwd, "~\\docs"), home.join("docs"));
    // `~` only triggers as the first segment; `foo/~` stays literal under cwd.
    assert_eq!(resolve(cwd, "src"), cwd.join("src"));
    assert_eq!(resolve(cwd, "foo/~"), cwd.join("foo/~"));
    // Absolute path returned unchanged; root form is OS-specific.
    #[cfg(unix)]
    assert_eq!(resolve(cwd, "/etc/hosts"), PathBuf::from("/etc/hosts"));
    #[cfg(windows)]
    assert_eq!(resolve(cwd, "C:\\Windows"), PathBuf::from("C:\\Windows"));
}

#[test]
fn write_then_read_roundtrips() {
    let dir = tmp();
    write_file(&json!({"path":"a.txt","content":"hello\nworld"}), &dir).unwrap();
    let out = read_file(&json!({"path":"a.txt"}), &dir).unwrap();
    assert!(out.contains("hello"));
    assert!(out.contains("     1\t"));
}

#[test]
fn read_file_paging() {
    let dir = tmp();
    let body: String = (1..=10).map(|n| format!("line{n}\n")).collect();
    write_file(&json!({"path":"b.txt","content":body}), &dir).unwrap();
    let out = read_file(&json!({"path":"b.txt","offset":3,"limit":2}), &dir).unwrap();
    assert!(out.contains("line3"));
    assert!(out.contains("line4"));
    assert!(!out.contains("line5"));
    assert!(
        out.contains("6 more lines; continue with offset=5"),
        "{out}"
    );
}

#[test]
fn read_file_cap_notices_carry_resume_offset() {
    let dir = tmp();
    // Line cap: 3000 short lines, so the first 2000 fit under the byte cap.
    let body: String = (1..=3000).map(|n| format!("L{n}\n")).collect();
    write_file(&json!({"path":"lines.txt","content":body}), &dir).unwrap();
    let out = read_file(&json!({"path":"lines.txt"}), &dir).unwrap();
    assert!(out.contains("continue with offset=2001"), "{out}");

    // Byte cap mid-line: 108-byte rows cut at 30000 = 277 whole lines plus a
    // partial 278th, which the resume must re-read.
    let body: String = (1..=400)
        .map(|_| format!("{}\n", "x".repeat(100)))
        .collect();
    write_file(&json!({"path":"wide.txt","content":body}), &dir).unwrap();
    let out = read_file(&json!({"path":"wide.txt"}), &dir).unwrap();
    assert!(out.contains("continue with offset=278"), "{out}");
}

#[test]
fn read_file_accepts_start_line_end_line_aliases() {
    let dir = tmp();
    let body: String = (1..=10).map(|n| format!("line{n}\n")).collect();
    write_file(&json!({"path":"b.txt","content":body}), &dir).unwrap();
    let out = read_file(&json!({"path":"b.txt","start_line":3}), &dir).unwrap();
    assert!(out.contains("line3") && !out.contains("line2"));
    let out = read_file(&json!({"path":"b.txt","start_line":3,"end_line":5}), &dir).unwrap();
    assert!(out.contains("line3") && out.contains("line5") && !out.contains("line6"));
    // Explicit offset/limit win over the aliases.
    let out = read_file(
        &json!({"path":"b.txt","offset":2,"limit":1,"start_line":9}),
        &dir,
    )
    .unwrap();
    assert!(out.contains("line2") && !out.contains("line3"));
}

/// A model-supplied offset near `usize::MAX` must not overflow `start + limit`
/// (a panic in debug builds) — it should read past the end gracefully.
#[test]
fn read_file_huge_offset_does_not_overflow() {
    let dir = tmp();
    write_file(&json!({"path":"h.txt","content":"a\nb\nc\n"}), &dir).unwrap();
    let out = read_file(&json!({"path":"h.txt","offset": u64::MAX}), &dir).unwrap();
    assert!(out.contains("past end of file"), "got: {out}");
    // A huge limit (with a sane offset) must not overflow either.
    let out2 = read_file(&json!({"path":"h.txt","limit": u64::MAX}), &dir).unwrap();
    assert!(
        out2.contains("a") && !out2.contains("more lines"),
        "got: {out2}"
    );
}

#[test]
fn read_file_rejects_binary_and_directory() {
    let dir = tmp();
    std::fs::write(dir.join("bin.dat"), [0x00u8, 0x01, 0x02, b'x']).unwrap();
    let err = read_file(&json!({"path":"bin.dat"}), &dir).unwrap_err();
    assert!(err.contains("binary"), "got: {err}");
    let err = read_file(&json!({"path":"."}), &dir).unwrap_err();
    assert!(err.contains("directory"), "got: {err}");
}

/// A FIFO/device read blocks forever and once froze the whole TUI — these
/// must error fast, never hang.
#[cfg(unix)]
#[test]
fn read_file_refuses_fifo_and_device() {
    let dir = tmp();
    mkfifo(&dir.join("pipe"));
    let err = read_file(&json!({"path":"pipe"}), &dir).unwrap_err();
    assert!(err.contains("not a regular file"), "got: {err}");
    let err = read_file(&json!({"path":"/dev/null"}), &dir).unwrap_err();
    assert!(err.contains("not a regular file"), "got: {err}");
}

#[test]
fn cap_keeps_correct_end() {
    let body: String = (1..=3000).map(|n| format!("L{n}\n")).collect();
    let head = cap_head(body.clone());
    assert!(head.contains("L1\n") && head.contains("truncated") && !head.contains("L3000"));
    let tail = cap_tail(body);
    assert!(tail.contains("L3000") && tail.contains("truncated") && !tail.contains("L1\n"));
}

#[test]
fn read_dedupe_key_normalizes_paths_aliases_and_defaults() {
    let cwd = Path::new("/w");
    let a = read_dedupe_key("read_file", &json!({"path":"src/x.rs"}), cwd);
    let b = read_dedupe_key("read_file", &json!({"path":"./src/x.rs","offset":1}), cwd);
    let c = read_dedupe_key(
        "read_file",
        &json!({"path":"/w/src/x.rs","start_line":1}),
        cwd,
    );
    assert!(a.is_some());
    assert_eq!(a, b, "relative/`./`-prefixed + default offset collide");
    assert_eq!(a, c, "absolute path + start_line alias collide");
    let paged = read_dedupe_key("read_file", &json!({"path":"src/x.rs","offset":100}), cwd);
    assert_ne!(a, paged, "a different page is a different read");
    assert_eq!(
        read_dedupe_key("run_bash", &json!({"command":"ls"}), cwd),
        None,
        "non-repeatable tools are ineligible"
    );
    assert_eq!(
        read_dedupe_key("grep", &json!({"pattern":"foo"}), cwd),
        read_dedupe_key(
            "grep",
            &json!({"pattern":"foo","path":".","context":0}),
            cwd
        ),
        "grep defaults normalize"
    );
    assert_eq!(
        read_dedupe_key("grep", &json!({"pattern":"foo","context":150}), cwd),
        read_dedupe_key("grep", &json!({"pattern":"foo","context":100}), cwd),
        "grep context mirrors the tool's clamp"
    );
    assert_eq!(
        read_dedupe_key("web_fetch", &json!({"url":"https://e.co/d"}), cwd),
        read_dedupe_key(
            "web_fetch",
            &json!({"url":"https://e.co/d","max_chars":30_000}),
            cwd
        ),
        "web_fetch default max_chars mirrors the tool's MAX_OUTPUT"
    );
    assert_ne!(
        read_dedupe_key("web_fetch", &json!({"url":"https://e.co/d"}), cwd),
        read_dedupe_key(
            "web_fetch",
            &json!({"url":"https://e.co/d","max_chars":0}),
            cwd
        ),
        "an explicit tiny cap is a different fetch — must not collide with a full one"
    );
}

/// A write stages through a sibling temp and renames into place, leaving no
/// `.aivo-tmp-*` staging file behind.
#[test]
fn write_is_atomic_and_leaves_no_temp_file() {
    let dir = tmp();
    write_file(&json!({"path":"a.txt","content":"hello"}), &dir).unwrap();
    assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "hello");
    let leftover = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().starts_with(".aivo-tmp-"));
    assert!(!leftover, "a staging temp file was left behind");
}

#[test]
fn write_file_refuses_path_outside_workspace_and_creates_nothing() {
    let dir = tmp();
    let rel = format!(
        "../{}-escape.txt",
        dir.file_name().unwrap().to_string_lossy()
    );
    let err = write_file(&json!({"path": &rel, "content": "PWN"}), &dir).unwrap_err();
    assert!(
        err.contains("outside the workspace"),
        "expected refuse, got: {err}"
    );
    assert!(
        !dir.parent()
            .unwrap()
            .join(rel.trim_start_matches("../"))
            .exists(),
        "a refused write must not create the file"
    );
}

#[test]
fn edit_file_refuses_path_outside_workspace() {
    let dir = tmp();
    let rel = format!(
        "../{}-escape.txt",
        dir.file_name().unwrap().to_string_lossy()
    );
    let err = edit_file(
        &json!({"path": &rel, "old_string": "a", "new_string": "b"}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("outside the workspace"), "got: {err}");
}
