//! JSON-lines-over-stdio sidecar processes and the sidecar coreference backend.
//!
//! A sidecar is a long-running child process that reads one JSON request per line on stdin and
//! answers each with one JSON line on stdout. Its stderr is forwarded to `tracing` line by line.
//! On Windows the child runs in a job object (it dies with this process) without a console
//! window; on Unix it leads its own process group.
//!
//! A response object with an `"error"` member is reported as [`ModelError::Inference`].
//! A process that exits or times out is restarted on the next request.

use std::ffi::OsString;
use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;

use ce_core::util::lock;
use std::thread;
use std::time::{Duration, Instant};

use process_wrap::std::{ChildWrapper, CommandWrap};
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::traits::{Coref, CorefCluster};

/// Time allowed for a sidecar to exit after its stdin is closed before it is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// How to start a sidecar process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarCommand {
    /// Executable to run.
    pub program: PathBuf,
    /// Arguments.
    pub args: Vec<OsString>,
    /// Extra environment variables (applied after the UTF-8 defaults).
    pub env: Vec<(OsString, OsString)>,
    /// Working directory; `None` inherits the current one.
    pub current_dir: Option<PathBuf>,
    /// Per-request response timeout; `None` waits indefinitely.
    pub timeout: Option<Duration>,
}

impl SidecarCommand {
    /// A command running `program` with no arguments and no timeout.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            current_dir: None,
            timeout: None,
        }
    }

    /// Appends an argument.
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Appends arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Sets an environment variable for the child.
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Sets the working directory.
    pub fn current_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    /// Sets the per-request response timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    fn display_name(&self) -> String {
        self.program.file_stem().map_or_else(
            || self.program.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        )
    }

    fn to_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            // Python sidecars read and write UTF-8 regardless of the console code page.
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = &self.current_dir {
            command.current_dir(dir);
        }
        command
    }
}

/// A running child with its request pipe and response line channel.
struct Running {
    child: Box<dyn ChildWrapper>,
    stdin: Option<ChildStdin>,
    lines: Receiver<Result<String, String>>,
}

impl Running {
    fn spawn(command: &SidecarCommand, name: &str) -> Result<Self, ModelError> {
        let mut wrapped = CommandWrap::from(command.to_command());
        #[cfg(windows)]
        {
            use process_wrap::std::{CreationFlags, JobObject};
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let mut flags = CreationFlags(Default::default());
            flags.0 .0 = CREATE_NO_WINDOW;
            wrapped.wrap(JobObject).wrap(flags);
        }
        #[cfg(unix)]
        {
            wrapped.wrap(process_wrap::std::ProcessGroup::leader());
        }
        let mut child = wrapped.spawn().map_err(|e| ModelError::Load {
            model: name.to_owned(),
            message: format!("failed to start sidecar `{}`: {e}", command.program.display()),
        })?;
        let pipe_error = |pipe: &str| ModelError::Load {
            model: name.to_owned(),
            message: format!("sidecar {pipe} pipe is not available"),
        };
        let stdin = child.stdin().take().ok_or_else(|| pipe_error("stdin"))?;
        let stdout = child.stdout().take().ok_or_else(|| pipe_error("stdout"))?;
        let stderr = child.stderr().take().ok_or_else(|| pipe_error("stderr"))?;

        let (tx, lines) = mpsc::channel();
        thread::Builder::new()
            .name(format!("{name}-stdout"))
            .spawn(move || read_lines(stdout, |line| tx.send(line).is_ok()))?;
        let stderr_name = name.to_owned();
        thread::Builder::new().name(format!("{name}-stderr")).spawn(move || {
            read_lines(stderr, |line| {
                match line {
                    Ok(line) => tracing::info!(target: "ce_models::sidecar", sidecar = %stderr_name, "{line}"),
                    Err(e) => tracing::warn!(target: "ce_models::sidecar", sidecar = %stderr_name, "{e}"),
                }
                true
            });
        })?;
        Ok(Self { child, stdin: Some(stdin), lines })
    }

    /// Closes stdin, waits up to [`SHUTDOWN_GRACE`] for the child to exit, then kills it.
    fn shutdown(mut self) {
        drop(self.stdin.take());
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
                _ => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn exit_description(&mut self) -> String {
        match self.child.try_wait() {
            Ok(Some(status)) => format!("sidecar exited with {status}"),
            _ => "sidecar closed its stdout".to_owned(),
        }
    }
}

/// Reads `reader` line by line (UTF-8, `\n` or `\r\n` terminated) until EOF, an I/O error or
/// `sink` returning `false`.
fn read_lines<R: Read>(reader: R, mut sink: impl FnMut(Result<String, String>) -> bool) {
    let mut reader = BufReader::new(reader);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => return,
            Ok(_) => {
                while matches!(buf.last(), Some(b'\n' | b'\r')) {
                    buf.pop();
                }
                let line = String::from_utf8(std::mem::take(&mut buf))
                    .map_err(|e| format!("sidecar output is not UTF-8: {e}"));
                if !sink(line) {
                    return;
                }
            }
            Err(e) => {
                let _ = sink(Err(format!("failed to read sidecar output: {e}")));
                return;
            }
        }
    }
}

/// A JSON-lines request / response client for a sidecar process. Requests are serialised: one
/// request is in flight at a time.
pub struct SidecarProcess {
    name: String,
    command: SidecarCommand,
    running: Mutex<Option<Running>>,
}

impl fmt::Debug for SidecarProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SidecarProcess")
            .field("name", &self.name)
            .field("command", &self.command)
            .field("running", &lock(&self.running).is_some())
            .finish()
    }
}

impl SidecarProcess {
    /// Starts the sidecar. Fails when the program cannot be spawned.
    pub fn spawn(command: SidecarCommand) -> Result<Self, ModelError> {
        let name = command.display_name();
        let running = Running::spawn(&command, &name)?;
        Ok(Self { name, command, running: Mutex::new(Some(running)) })
    }

    /// The command this sidecar runs.
    pub fn command(&self) -> &SidecarCommand {
        &self.command
    }

    /// Sends `request` as one JSON line and returns the next JSON line of output. Blank output
    /// lines are skipped. A dead sidecar is restarted first.
    pub fn request(&self, request: &Value) -> Result<Value, ModelError> {
        let line = serde_json::to_string(request)
            .map_err(|e| ModelError::Protocol(format!("cannot encode sidecar request: {e}")))?;
        let mut guard = lock(&self.running);
        if guard.is_none() {
            tracing::info!(sidecar = %self.name, "restarting sidecar");
            *guard = Some(Running::spawn(&self.command, &self.name)?);
        }
        let running = guard.as_mut().expect("sidecar is running: started above when absent");
        match self.exchange(running, &line) {
            Ok(value) => Ok(value),
            Err(e) => {
                // The stream may be out of sync; restart on the next request.
                if let Some(running) = guard.take() {
                    running.shutdown();
                }
                Err(e)
            }
        }
    }

    fn exchange(&self, running: &mut Running, line: &str) -> Result<Value, ModelError> {
        let stdin = running
            .stdin
            .as_mut()
            .ok_or_else(|| ModelError::Protocol("sidecar stdin is closed".to_owned()))?;
        let written = stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush());
        if let Err(e) = written {
            return Err(ModelError::Protocol(format!(
                "cannot write to sidecar `{}` ({e}): {}",
                self.name,
                running.exit_description()
            )));
        }
        let deadline = self.command.timeout.map(|t| Instant::now() + t);
        loop {
            let received = match deadline {
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    running.lines.recv_timeout(left)
                }
                None => running.lines.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };
            let output = match received {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => return Err(ModelError::Protocol(e)),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(ModelError::Inference(format!(
                        "sidecar `{}` did not answer within {:?}",
                        self.name, self.command.timeout
                    )))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Give the child a moment to be reaped so the exit status is reported.
                    thread::sleep(Duration::from_millis(50));
                    return Err(ModelError::Protocol(format!(
                        "sidecar `{}` stopped before answering: {}",
                        self.name,
                        running.exit_description()
                    )));
                }
            };
            if output.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&output).map_err(|e| {
                ModelError::Protocol(format!(
                    "sidecar `{}` wrote a non-JSON line ({e}): {output:?}",
                    self.name
                ))
            })?;
            if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
                let message = error.as_str().map_or_else(|| error.to_string(), str::to_owned);
                return Err(ModelError::Inference(format!("sidecar `{}`: {message}", self.name)));
            }
            return Ok(value);
        }
    }

    /// Stops the sidecar (closing stdin, then killing it after a grace period). A later request
    /// starts it again.
    pub fn shutdown(&self) {
        if let Some(running) = lock(&self.running).take() {
            running.shutdown();
        }
    }
}

impl Drop for SidecarProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Coreference through a sidecar speaking the coref protocol:
/// request `{"op": "coref", "text": ...}`, response `{"clusters": [[[start, end], ...], ...]}`
/// with character offsets (fastcoref `get_clusters(as_strings=False)`).
///
/// Spans are filtered like the Python service: malformed spans, `end <= start`, `end` past the
/// text and duplicates within a cluster are dropped. Clusters keep their position in the
/// response even when every span is dropped (they are then empty), so cluster indexes match the
/// Python `<doc_id>_cluster_<idx>` ids; callers skip empty clusters.
#[derive(Debug)]
pub struct SidecarCoref {
    process: SidecarProcess,
}

impl SidecarCoref {
    /// Starts the sidecar described by `command`.
    pub fn new(command: SidecarCommand) -> Result<Self, ModelError> {
        SidecarProcess::spawn(command).map(Self::from_process)
    }

    /// Uses an already running sidecar.
    pub fn from_process(process: SidecarProcess) -> Self {
        Self { process }
    }

    /// The underlying sidecar process.
    pub fn process(&self) -> &SidecarProcess {
        &self.process
    }
}

impl Coref for SidecarCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>, ModelError> {
        let response = self.process.request(&json!({"op": "coref", "text": text}))?;
        parse_coref_response(&response, text.chars().count())
    }
}

/// Parses a coref response for a text of `char_len` characters.
fn parse_coref_response(
    response: &Value,
    char_len: usize,
) -> Result<Vec<CorefCluster>, ModelError> {
    let clusters = response.get("clusters").and_then(Value::as_array).ok_or_else(|| {
        ModelError::Protocol(format!("coref response has no `clusters` array: {response}"))
    })?;
    clusters
        .iter()
        .map(|cluster| {
            let spans = cluster.as_array().ok_or_else(|| {
                ModelError::Protocol(format!("coref cluster is not an array: {cluster}"))
            })?;
            let mut kept: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
            for span in spans {
                let Some([start, end]) = span
                    .as_array()
                    .map(Vec::as_slice)
                    .and_then(|s| <&[Value; 2]>::try_from(s).ok())
                else {
                    continue;
                };
                let (Some(start), Some(end)) = (as_offset(start), as_offset(end)) else {
                    continue;
                };
                if end <= start || end > char_len || kept.contains(&(start, end)) {
                    continue;
                }
                kept.push((start, end));
            }
            Ok(CorefCluster { spans: kept })
        })
        .collect()
}

/// A non-negative integer offset (integral floats accepted, as Python `int()` would).
fn as_offset(value: &Value) -> Option<usize> {
    if let Some(n) = value.as_u64() {
        return usize::try_from(n).ok();
    }
    value
        .as_f64()
        .filter(|f| f.fract() == 0.0 && *f >= 0.0 && *f <= usize::MAX as f64)
        .map(|f| f as usize)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn coref_response_filters_like_python() {
        let response = json!({"clusters": [
            [[0, 4], [10, 12], [10, 12], [5, 5], [7, 3], [8, 99], [1], [-1, 2]],
            [[20, 30]],
            [[2.0, 3.0]]
        ]});
        let clusters = parse_coref_response(&response, 15).expect("valid response");
        assert_eq!(
            clusters,
            vec![
                CorefCluster { spans: vec![(0, 4), (10, 12)] },
                CorefCluster { spans: vec![] },
                CorefCluster { spans: vec![(2, 3)] },
            ]
        );
    }

    #[test]
    fn coref_response_requires_clusters() {
        assert!(matches!(
            parse_coref_response(&json!({"result": []}), 3),
            Err(ModelError::Protocol(_))
        ));
        assert!(matches!(
            parse_coref_response(&json!({"clusters": [5]}), 3),
            Err(ModelError::Protocol(_))
        ));
    }

    #[test]
    fn spawn_failure_is_a_load_error() {
        let err = SidecarProcess::spawn(SidecarCommand::new("ce-models-no-such-program-xyz"))
            .expect_err("missing program");
        assert!(matches!(err, ModelError::Load { .. }), "{err:?}");
    }
}
