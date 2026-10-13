//! Console output that never panics: write errors on stdout (a closed pipe, a full disk) stop
//! further console output but do not abort the command.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

static STDOUT_CLOSED: AtomicBool = AtomicBool::new(false);

/// Writes `text` to stdout as is; after the first write error every later call is a no-op.
pub fn print(text: &str) {
    if STDOUT_CLOSED.load(Ordering::Relaxed) {
        return;
    }
    let mut out = std::io::stdout().lock();
    if out.write_all(text.as_bytes()).and_then(|()| out.flush()).is_err() {
        STDOUT_CLOSED.store(true, Ordering::Relaxed);
    }
}

/// Writes `text` and a newline to stdout (see [`print()`]).
pub fn println(text: &str) {
    let mut line = String::with_capacity(text.len() + 1);
    line.push_str(text);
    line.push('\n');
    print(&line);
}

/// Writes `text` and a newline to stderr, ignoring write errors.
pub fn eprintln(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{text}");
}
