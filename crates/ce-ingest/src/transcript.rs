//! Transcript parser (Python `TranscriptParser`): JSON / JSON Lines, CSV and plain-text
//! deposition transcripts into utterances, and their flattening into one document text.

use std::cmp::Ordering;
use std::path::Path;
use std::sync::LazyLock;

use ce_core::Metadata;
use ce_text::normalize::py_strip;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;

use crate::error::{IngestError, Result};
use crate::loader::suffix;
use crate::py;

/// One utterance. Values parsed from JSON or CSV are kept as read (a JSON `speaker` may be any
/// JSON value, a short CSV row yields `null`); the plain-text parser always produces strings
/// and a `line_number`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Utterance {
    /// Speaker label.
    pub speaker: Value,
    /// Speaker role (`unknown` when absent).
    pub role: Value,
    /// Utterance text.
    pub text: Value,
    /// Timestamp, `null` when absent.
    pub timestamp: Value,
    /// 1-based source line where the utterance starts (plain text only; 0 when unknown).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<usize>,
    /// Remaining input fields (JSON / CSV) or the speaker marker kind (plain text).
    pub metadata: Metadata,
}

/// Keys `_normalize_utterance` consumes; everything else goes to `metadata`.
const UTTERANCE_KEYS: [&str; 10] = [
    "speaker",
    "Speaker",
    "role",
    "Role",
    "text",
    "Text",
    "utterance",
    "timestamp",
    "Timestamp",
    "time",
];

/// Python `_normalize_utterance` over a JSON object or CSV row.
fn normalize_utterance(data: &Metadata) -> Utterance {
    let first = |keys: &[&str], default: Value| {
        keys.iter().find_map(|key| data.get(*key).cloned()).unwrap_or(default)
    };
    Utterance {
        speaker: first(&["speaker", "Speaker"], Value::from("unknown")),
        role: first(&["role", "Role"], Value::from("unknown")),
        text: first(&["text", "Text", "utterance"], Value::from("")),
        timestamp: first(&["timestamp", "Timestamp", "time"], Value::Null),
        line_number: None,
        metadata: data
            .iter()
            .filter(|(key, _)| !UTTERANCE_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    }
}

/// Parses a JSON transcript: an array of utterance objects or a single object; when the file
/// is not one JSON document, each non-blank line is parsed as JSON and undecodable lines are
/// skipped (JSON Lines).
///
/// `NaN`, `Infinity` and `-Infinity` literals (which Python's `json.load` accepts) read as the
/// strings `nan`, `inf` and `-inf`, what Python `str()` prints for them; a non-finite speaker
/// is rejected.
///
/// # Errors
/// Read errors, and [`IngestError::InvalidTranscript`] when an utterance is not an object or a
/// speaker is a non-finite number.
pub fn parse_json(path: &Path) -> Result<Vec<Utterance>> {
    let content = py::read_text(path)?;
    let not_object = || IngestError::InvalidTranscript {
        path: path.to_path_buf(),
        reason: "every utterance must be a JSON object".to_owned(),
    };
    let utterance = |value: Value| match value {
        Value::Object(map) => Ok(normalize_utterance(&map)),
        _ => Err(not_object()),
    };
    let mut utterances = match json_loads(&content) {
        Some(Value::Array(items)) => items.into_iter().map(utterance).collect::<Result<_>>()?,
        Some(other) => vec![utterance(other)?],
        None => content
            .split_terminator('\n')
            .map(py_strip)
            .filter(|line| !line.is_empty())
            .filter_map(json_loads)
            .map(utterance)
            .collect::<Result<_>>()?,
    };
    if has_non_finite_literal(&content) {
        for utterance in &mut utterances {
            resolve_non_finite(path, utterance)?;
        }
    }
    Ok(utterances)
}

/// Marks a string that stands for one of Python's non-finite float literals.
const NON_FINITE_TAG: char = '\0';

fn has_non_finite_literal(text: &str) -> bool {
    text.contains("NaN") || text.contains("Infinity")
}

/// Python `json.loads`, which also accepts the `NaN`, `Infinity` and `-Infinity` literals;
/// those become tagged strings that [`resolve_non_finite`] resolves. Returns `None` when the
/// text is not one JSON document.
fn json_loads(text: &str) -> Option<Value> {
    match serde_json::from_str(text) {
        Ok(value) => Some(value),
        Err(_) if has_non_finite_literal(text) => serde_json::from_str(&tag_non_finite(text)).ok(),
        Err(_) => None,
    }
}

/// Replaces `NaN`, `Infinity` and `-Infinity` outside string literals with tagged strings of
/// Python's `str()` spelling (`nan`, `inf`, `-inf`).
fn tag_non_finite(text: &str) -> String {
    const LITERALS: [(&str, &str); 3] =
        [("-Infinity", "-inf"), ("Infinity", "inf"), ("NaN", "nan")];
    let mut out = String::with_capacity(text.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        } else if let Some((literal, spelling)) =
            LITERALS.iter().find(|(literal, _)| rest.starts_with(literal))
        {
            out.push_str(&Value::String(format!("{NON_FINITE_TAG}{spelling}")).to_string());
            rest = &rest[literal.len()..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// The Python spelling behind a tagged non-finite string.
fn non_finite_spelling(value: &Value) -> Option<&str> {
    value.as_str()?.strip_prefix(NON_FINITE_TAG)
}

/// Rejects a non-finite speaker (Python would write a bare `NaN` into `unique_speakers`, which
/// is not JSON, or fail to sort it against string speakers) and turns every other tagged value
/// into its Python `str()` spelling.
fn resolve_non_finite(path: &Path, utterance: &mut Utterance) -> Result<()> {
    if let Some(spelling) = non_finite_spelling(&utterance.speaker) {
        return Err(IngestError::InvalidTranscript {
            path: path.to_path_buf(),
            reason: format!("speaker is the non-finite number {spelling}"),
        });
    }
    untag(&mut utterance.role);
    untag(&mut utterance.text);
    untag(&mut utterance.timestamp);
    utterance.metadata.values_mut().for_each(untag);
    Ok(())
}

fn untag(value: &mut Value) {
    match value {
        Value::String(_) => {
            if let Some(spelling) = non_finite_spelling(value) {
                *value = Value::from(spelling);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(untag),
        Value::Object(map) => map.values_mut().for_each(untag),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Parses a CSV transcript with a header row (`speaker,role,text[,timestamp]`) like Python's
/// `csv.DictReader`: blank lines are skipped, missing trailing fields are `null`, surplus
/// fields are collected as a list under the `null` metadata key.
///
/// # Errors
/// Read and CSV syntax errors.
pub fn parse_csv(path: &Path) -> Result<Vec<Utterance>> {
    let content = py::read_text(path)?;
    let csv_error = |source| IngestError::Csv { path: path.to_path_buf(), source };
    let mut reader =
        csv::ReaderBuilder::new().has_headers(false).flexible(true).from_reader(content.as_bytes());
    let mut records = reader.records();
    // An empty first line is an empty header row for Python's reader; this reader skips it.
    let header: Vec<String> = if content.starts_with('\n') {
        Vec::new()
    } else {
        match records.next() {
            Some(record) => record.map_err(csv_error)?.iter().map(str::to_owned).collect(),
            None => return Ok(Vec::new()),
        }
    };
    let mut utterances = Vec::new();
    for record in records {
        let record = record.map_err(csv_error)?;
        let mut row = Metadata::new();
        for (key, field) in header.iter().zip(record.iter()) {
            row.insert(key.clone(), Value::from(field));
        }
        if record.len() > header.len() {
            let rest: Vec<Value> = record.iter().skip(header.len()).map(Value::from).collect();
            row.insert("null".to_owned(), Value::Array(rest));
        } else {
            for key in &header[record.len()..] {
                row.insert(key.clone(), Value::Null);
            }
        }
        utterances.push(normalize_utterance(&row));
    }
    Ok(utterances)
}

/// Python `\s` (str patterns): Unicode whitespace plus U+001C..U+001F.
const WS: &str = r"[\s\x1C-\x1F]";

fn regex(pattern: &str) -> Regex {
    Regex::new(&pattern.replace(r"\s", WS)).expect("transcript pattern is valid")
}

static SPEAKER_WITH_ROLE_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^([A-Z][A-Z .,'-]+?)(?:\s*\(([^)]+)\))?:\s*(.*)$"));
static QA_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"^(Q|A)\.\s*(.*)$"));
static BY_EXAMINER_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^BY\s+(MR\.|MS\.|MRS\.)\s+([A-Z .'-]+):\s*(.*)$"));
static SIMPLE_SPEAKER_RE: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^([A-Z][A-Z .,'-]+):\s*(.*)$"));
// The `(?=\S)` lookahead of the Python pattern holds whenever the match ends before the end of
// a stripped line; `strip_line_number` checks that.
static LINE_NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| regex(r"^\s*\d{1,4}\s+"));
static HEADER_PATTERNS: LazyLock<[Regex; 3]> = LazyLock::new(|| {
    [
        regex(r"(?i)^Page\s+\d+(\s+of\s+\d+)?$"),
        regex(r"(?i)^(?:Deposition|Hearing) of .+Page \d+$"),
        regex(r"(?i)^Job Number[:\s].*$"),
    ]
});

/// True for page headers / footers (`Page 3 of 9`, `Deposition of ... Page 3`, `Job Number:`)
/// and short all-caps lines without `:` or `.`.
fn looks_like_header_or_footer(line: &str) -> bool {
    if line.is_empty() {
        return false;
    }
    if HEADER_PATTERNS.iter().any(|pattern| pattern.is_match(line)) {
        return true;
    }
    line.chars().count() <= 60 && !line.contains(':') && !line.contains('.') && py::isupper(line)
}

/// Removes a left-margin line number (`^\s*\d{1,4}\s+(?=\S)`) from a stripped line.
fn strip_line_number(line: &str) -> &str {
    match LINE_NUMBER_RE.find(line) {
        Some(m) if m.end() < line.len() => &line[m.end()..],
        _ => line,
    }
}

/// Drops page headers / footers, blanks bare line-number lines and strips margin line numbers.
fn preprocess_lines(content: &str) -> Vec<(usize, &str)> {
    let mut processed = Vec::new();
    for (idx, raw) in content.split_terminator('\n').enumerate() {
        let line_number = idx + 1;
        let stripped = py_strip(raw);
        if py::isdigit(stripped) {
            processed.push((line_number, ""));
            continue;
        }
        if looks_like_header_or_footer(stripped) {
            continue;
        }
        processed.push((line_number, strip_line_number(stripped)));
    }
    processed
}

/// A speaker marker recognised at the start of a line.
struct Marker {
    speaker: String,
    role: String,
    text: String,
    kind: &'static str,
    set_examiner: Option<String>,
    set_witness: Option<String>,
    emit: bool,
}

fn guess_role_from_label(label: &str) -> &'static str {
    let label = label.to_uppercase();
    let has = |tokens: &[&str]| tokens.iter().any(|token| label.contains(token));
    if has(&["JUDGE", "COURT"]) {
        "JUDGE"
    } else if has(&["WITNESS"]) {
        "WITNESS"
    } else if has(&["MR.", "MS.", "MRS.", "ATTORNEY", "COUNSEL"]) {
        "ATTORNEY"
    } else {
        "OTHER"
    }
}

fn non_empty(value: Option<&String>) -> Option<&String> {
    value.filter(|s| !s.is_empty())
}

fn speaker_marker(speaker: &str, role: String, remainder: &str) -> Marker {
    let trimmed = py_strip(speaker).to_owned();
    let role = if role.is_empty() { "UNKNOWN".to_owned() } else { role };
    Marker {
        set_witness: (role == "WITNESS").then(|| trimmed.clone()),
        set_examiner: (role == "ATTORNEY").then(|| trimmed.clone()),
        speaker: trimmed,
        role,
        text: remainder.to_owned(),
        kind: "speaker_line",
        emit: true,
    }
}

fn match_speaker_marker(
    line: &str,
    examining_attorney: Option<&String>,
    witness_name: Option<&String>,
) -> Option<Marker> {
    if line.is_empty() {
        return None;
    }
    if let Some(caps) = BY_EXAMINER_RE.captures(line) {
        let speaker = py_strip(&format!("{} {}", &caps[1], &caps[2])).to_owned();
        let remainder = &caps[3];
        return Some(Marker {
            set_examiner: Some(speaker.clone()),
            speaker,
            role: "ATTORNEY".to_owned(),
            text: remainder.to_owned(),
            kind: "examiner",
            set_witness: None,
            emit: !py_strip(remainder).is_empty(),
        });
    }
    if let Some(caps) = QA_RE.captures(line) {
        let (speaker, role, kind) = if &caps[1] == "Q" {
            let speaker = non_empty(examining_attorney).map_or("QUESTIONER", String::as_str);
            (speaker, "ATTORNEY", "question")
        } else {
            let speaker = non_empty(witness_name).map_or("WITNESS", String::as_str);
            (speaker, "WITNESS", "answer")
        };
        return Some(Marker {
            speaker: speaker.to_owned(),
            role: role.to_owned(),
            text: caps[2].to_owned(),
            kind,
            set_examiner: None,
            set_witness: None,
            emit: true,
        });
    }
    if let Some(caps) = SPEAKER_WITH_ROLE_RE.captures(line) {
        let speaker = &caps[1];
        let role = match caps.get(2) {
            Some(role) if !role.as_str().is_empty() => role.as_str().to_uppercase(),
            _ => guess_role_from_label(speaker).to_owned(),
        };
        return Some(speaker_marker(speaker, role, &caps[3]));
    }
    if let Some(caps) = SIMPLE_SPEAKER_RE.captures(line) {
        let speaker = &caps[1];
        let role = guess_role_from_label(speaker).to_owned();
        return Some(speaker_marker(speaker, role, &caps[2]));
    }
    None
}

/// Accumulates the lines of the utterance being read.
#[derive(Default)]
struct PlainTextState {
    speaker: Option<String>,
    role: Option<String>,
    marker: Option<&'static str>,
    line_number: Option<usize>,
    buffer: Vec<String>,
}

impl PlainTextState {
    fn has_speaker(&self) -> bool {
        non_empty(self.speaker.as_ref()).is_some()
    }

    /// Emits the pending utterance, if any (Python `flush_current`).
    fn flush(&mut self) -> Option<Utterance> {
        if self.buffer.is_empty() && !self.has_speaker() {
            return None;
        }
        let parts: Vec<&str> =
            self.buffer.iter().map(|part| py_strip(part)).filter(|part| !part.is_empty()).collect();
        let text = py_strip(&parts.join(" ")).to_owned();
        self.buffer.clear();
        if text.is_empty() && !self.has_speaker() {
            self.line_number = None;
            self.marker = None;
            return None;
        }
        let mut metadata = Metadata::new();
        if let Some(kind) = self.marker.take() {
            metadata.insert("marker".to_owned(), Value::from(kind));
        }
        let or_unknown = |value: Option<String>| {
            Value::from(value.filter(|s| !s.is_empty()).unwrap_or_else(|| "unknown".to_owned()))
        };
        Some(Utterance {
            speaker: or_unknown(self.speaker.take()),
            role: or_unknown(self.role.take()),
            text: Value::from(text),
            timestamp: Value::Null,
            line_number: Some(self.line_number.take().unwrap_or(0)),
            metadata,
        })
    }
}

/// Parses a plain-text transcript: upper-case speaker labels (`NAME (role):`, `MR. X:`),
/// `BY MR. X:` examiner lines and `Q.` / `A.` markers attributed to the current examiner /
/// witness; page headers, `Job Number` lines, bare line numbers and short all-caps lines are
/// dropped, margin line numbers stripped, and blank lines end an utterance.
///
/// # Errors
/// Read errors.
pub fn parse_plain_text(path: &Path) -> Result<Vec<Utterance>> {
    let content = py::read_text(path)?;
    Ok(parse_plain_text_str(&content))
}

/// [`parse_plain_text`] over text already in memory (lines split on `\n`).
#[must_use]
pub fn parse_plain_text_str(content: &str) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    let mut state = PlainTextState::default();
    let mut examining_attorney: Option<String> = None;
    let mut witness_name: Option<String> = None;

    for (line_number, line) in preprocess_lines(content) {
        if let Some(marker) =
            match_speaker_marker(line, examining_attorney.as_ref(), witness_name.as_ref())
        {
            if let Some(examiner) = marker.set_examiner.filter(|s| !s.is_empty()) {
                examining_attorney = Some(examiner);
            }
            if let Some(witness) = marker.set_witness.filter(|s| !s.is_empty()) {
                witness_name = Some(witness);
            }
            if !marker.emit {
                continue;
            }
            utterances.extend(state.flush());
            if marker.role == "WITNESS"
                && marker.speaker != "unknown"
                && marker.speaker != "UNKNOWN"
            {
                witness_name = Some(marker.speaker.clone());
            }
            state.speaker = Some(marker.speaker);
            state.role = Some(marker.role);
            state.marker = Some(marker.kind);
            state.line_number = Some(line_number);
            let remainder = py_strip(&marker.text);
            if !remainder.is_empty() {
                state.buffer.push(remainder.to_owned());
            }
            continue;
        }

        let stripped = py_strip(line);
        if stripped.is_empty() {
            if !state.buffer.is_empty() || state.has_speaker() {
                utterances.extend(state.flush());
            }
            continue;
        }
        state.line_number.get_or_insert(line_number);
        state.buffer.push(stripped.to_owned());
    }
    utterances.extend(state.flush());
    utterances
}

/// Parses a transcript by extension: `.json`, `.csv`, anything else as plain text.
///
/// # Errors
/// The errors of the selected parser.
pub fn parse_file(path: &Path) -> Result<Vec<Utterance>> {
    match suffix(path).to_lowercase().as_str() {
        ".json" => parse_json(path),
        ".csv" => parse_csv(path),
        _ => parse_plain_text(path),
    }
}

/// A transcript flattened into document text.
#[derive(Debug, Clone, PartialEq)]
pub struct FlattenedTranscript {
    /// One `SPEAKER (role): text` line per utterance (`SPEAKER: text` when the role is falsy
    /// or `unknown`), joined with `\n`.
    pub text: String,
    /// Distinct speakers, sorted.
    pub unique_speakers: Vec<Value>,
}

/// Flattens utterances the way `run_ingestion` does, formatting values with Python `str()`.
///
/// # Errors
/// [`IngestError::InvalidTranscript`] for `path` when a speaker is a list or object
/// (unhashable in Python) or when speakers of different types would have to be sorted together.
pub fn flatten_utterances(path: &Path, utterances: &[Utterance]) -> Result<FlattenedTranscript> {
    flatten(utterances)
        .map_err(|reason| IngestError::InvalidTranscript { path: path.to_path_buf(), reason })
}

fn flatten(utterances: &[Utterance]) -> std::result::Result<FlattenedTranscript, String> {
    let mut lines = Vec::with_capacity(utterances.len());
    let mut speakers: Vec<&Value> = Vec::new();
    for utterance in utterances {
        let speaker = &utterance.speaker;
        if let Value::Array(_) | Value::Object(_) = speaker {
            return Err(format!("unhashable speaker {}", py::py_str(speaker)));
        }
        if !speakers.iter().any(|seen| py_equal(seen, speaker)) {
            speakers.push(speaker);
        }
        let role = &utterance.role;
        let speaker = py::py_str(speaker);
        let text = py::py_str(&utterance.text);
        if py::truthy(role) && role.as_str() != Some("unknown") {
            lines.push(format!("{speaker} ({}): {text}", py::py_str(role)));
        } else {
            lines.push(format!("{speaker}: {text}"));
        }
    }
    Ok(FlattenedTranscript { text: lines.join("\n"), unique_speakers: sort_speakers(speakers)? })
}

/// A number or bool as Python compares it.
#[derive(Clone, Copy)]
enum PyNumber {
    Int(i128),
    Float(f64),
}

fn as_number(value: &Value) -> Option<PyNumber> {
    match value {
        Value::Bool(b) => Some(PyNumber::Int(i128::from(*b))),
        Value::Number(n) => n
            .as_i64()
            .map(|i| PyNumber::Int(i128::from(i)))
            .or_else(|| n.as_u64().map(|u| PyNumber::Int(i128::from(u))))
            .or_else(|| n.as_f64().map(PyNumber::Float)),
        _ => None,
    }
}

fn compare_numbers(a: PyNumber, b: PyNumber) -> Ordering {
    match (a, b) {
        (PyNumber::Int(x), PyNumber::Int(y)) => x.cmp(&y),
        (PyNumber::Int(x), PyNumber::Float(y)) => (x as f64).total_cmp(&y),
        (PyNumber::Float(x), PyNumber::Int(y)) => x.total_cmp(&(y as f64)),
        (PyNumber::Float(x), PyNumber::Float(y)) => x.total_cmp(&y),
    }
}

/// Python `==` between hashable JSON values (`1 == 1.0 == True`).
fn py_equal(a: &Value, b: &Value) -> bool {
    match (as_number(a), as_number(b)) {
        (Some(x), Some(y)) => compare_numbers(x, y) == Ordering::Equal,
        (None, None) => a == b,
        _ => false,
    }
}

/// Python `sorted()` of the distinct speakers: strings by code point, numbers numerically;
/// mixed types fail as they do in Python (a single speaker is never compared).
fn sort_speakers(speakers: Vec<&Value>) -> Result<Vec<Value>, String> {
    let mut speakers: Vec<Value> = speakers.into_iter().cloned().collect();
    if speakers.len() < 2 {
        return Ok(speakers);
    }
    if speakers.iter().all(Value::is_string) {
        speakers.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
        return Ok(speakers);
    }
    let numbers: Option<Vec<PyNumber>> = speakers.iter().map(as_number).collect();
    let Some(numbers) = numbers else {
        return Err("speakers of different types cannot be sorted".to_owned());
    };
    let mut indexed: Vec<(PyNumber, Value)> = numbers.into_iter().zip(speakers).collect();
    indexed.sort_by(|a, b| compare_numbers(a.0, b.0));
    Ok(indexed.into_iter().map(|(_, value)| value).collect())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    /// Writes dedented `content` plus a trailing newline, like the Python test helper.
    fn write_tmp_text(dir: &Path, content: &str, name: &str) -> std::path::PathBuf {
        let lines: Vec<&str> = content.lines().collect();
        let indent = lines
            .iter()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.len() - line.trim_start().len())
            .min()
            .unwrap_or(0);
        let dedented: Vec<&str> =
            lines.iter().map(|line| line.get(indent..).unwrap_or("")).collect();
        let path = dir.join(name);
        std::fs::write(&path, format!("{}\n", dedented.join("\n").trim())).expect("write");
        path
    }

    fn field<'a>(u: &'a Utterance, key: &str) -> &'a str {
        let value = match key {
            "speaker" => &u.speaker,
            "role" => &u.role,
            "text" => &u.text,
            _ => &u.metadata[key],
        };
        value.as_str().expect("string field")
    }

    #[test]
    fn plain_text_parsing_with_speakers_and_q_a() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sample = "
            Page 12
            1   MR. SMITH (ATTORNEY): Please state your name for the record.
            2   THE WITNESS: Bob Smith.
            3   Q. Where were you on Friday night?
            4   A. I was at home.
            5   BY MS. JONES:
            6   Q. Did you speak with anyone after the incident?
            7   A. No.
        ";
        let utterances =
            parse_plain_text(&write_tmp_text(dir.path(), sample, "sample_transcript.txt"))
                .expect("parse");

        assert_eq!(utterances.len(), 6);
        assert_eq!(field(&utterances[0], "speaker"), "MR. SMITH");
        assert_eq!(field(&utterances[0], "role"), "ATTORNEY");
        assert!(field(&utterances[0], "text").contains("Please state your name"));

        assert_eq!(field(&utterances[1], "speaker"), "THE WITNESS");
        assert_eq!(field(&utterances[1], "role"), "WITNESS");
        assert_eq!(field(&utterances[1], "text"), "Bob Smith.");

        // Q uses the most recent attorney speaker.
        assert_eq!(field(&utterances[2], "speaker"), "MR. SMITH");
        assert_eq!(field(&utterances[2], "role"), "ATTORNEY");
        assert_eq!(field(&utterances[2], "marker"), "question");

        assert_eq!(field(&utterances[3], "speaker"), "THE WITNESS");
        assert_eq!(field(&utterances[3], "role"), "WITNESS");
        assert_eq!(field(&utterances[3], "marker"), "answer");

        // The examiner changes via the BY line.
        assert_eq!(field(&utterances[4], "speaker"), "MS. JONES");
        assert_eq!(field(&utterances[4], "role"), "ATTORNEY");
        assert_eq!(field(&utterances[4], "marker"), "question");

        assert_eq!(field(&utterances[5], "speaker"), "THE WITNESS");
        assert_eq!(field(&utterances[5], "role"), "WITNESS");
        assert_eq!(field(&utterances[5], "marker"), "answer");
        assert_eq!(utterances[5].line_number, Some(8));
    }

    #[test]
    fn plain_text_handles_continuation_and_unknown() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sample = "
            1   RANDOM HEADER
            2
            3   MS. CARTER: This is a long
            4   sentence that wraps
            5   across lines.
            6
            7   Unlabeled line continues previous speaker.
        ";
        let utterances =
            parse_plain_text(&write_tmp_text(dir.path(), sample, "t.txt")).expect("parse");

        assert_eq!(utterances.len(), 2);
        assert_eq!(field(&utterances[0], "speaker"), "MS. CARTER");
        assert_eq!(field(&utterances[0], "role"), "ATTORNEY");
        assert_eq!(
            field(&utterances[0], "text"),
            "This is a long sentence that wraps across lines."
        );
        assert_eq!(field(&utterances[1], "speaker"), "unknown");
        assert_eq!(field(&utterances[1], "role"), "unknown");
        assert!(field(&utterances[1], "text").contains("Unlabeled line"));
        assert!(utterances[1].metadata.is_empty());
    }

    #[test]
    fn plain_text_markers_and_headers() {
        let content = "Deposition of Jane Doe   Page 4\nJob Number: 77\n12345 MR. LEE: Long number kept.\n\
                       BY MR. LEE:\nQ. Question one?\nA. Answer.\nJUDGE: Order.\nCOUNSEL FOR X (Defense): Noted.\n\
                       MR. LEE:\n\n";
        let utterances = parse_plain_text_str(content);
        let summary: Vec<(String, String, String, Option<usize>)> = utterances
            .iter()
            .map(|u| {
                (py::py_str(&u.speaker), py::py_str(&u.role), py::py_str(&u.text), u.line_number)
            })
            .collect();
        let expected = [
            ("unknown", "unknown", "12345 MR. LEE: Long number kept.", 3),
            ("MR. LEE", "ATTORNEY", "Question one?", 5),
            ("WITNESS", "WITNESS", "Answer.", 6),
            ("JUDGE", "JUDGE", "Order.", 7),
            ("COUNSEL FOR X", "DEFENSE", "Noted.", 8),
            ("MR. LEE", "ATTORNEY", "", 9),
        ];
        let expected: Vec<(String, String, String, Option<usize>)> = expected
            .iter()
            .map(|(s, r, t, n)| ((*s).to_owned(), (*r).to_owned(), (*t).to_owned(), Some(*n)))
            .collect();
        assert_eq!(summary, expected);
    }

    #[test]
    fn strips_margin_line_numbers() {
        assert_eq!(strip_line_number("12   Text"), "Text");
        assert_eq!(strip_line_number("1234 Text"), "Text");
        assert_eq!(strip_line_number("12345 Text"), "12345 Text");
        assert_eq!(strip_line_number("12Text"), "12Text");
        assert_eq!(strip_line_number("Text 12"), "Text 12");
    }

    #[test]
    fn parse_file_dispatches_json_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.json");
        std::fs::write(
            &path,
            "{\"speaker\": \"MR. SMITH\", \"role\": \"ATTORNEY\", \"text\": \"Q1\", \"page\": 3}\n\
             {\"Speaker\": \"BOB\", \"Role\": \"WITNESS\", \"utterance\": \"A1\", \"time\": \"00:01\"}",
        )
        .expect("write");
        let utterances = parse_file(&path).expect("parse");
        let speakers: Vec<&Value> = utterances.iter().map(|u| &u.speaker).collect();
        assert_eq!(speakers, [&json!("MR. SMITH"), &json!("BOB")]);
        let texts: Vec<&Value> = utterances.iter().map(|u| &u.text).collect();
        assert_eq!(texts, [&json!("Q1"), &json!("A1")]);
        assert_eq!(Value::Object(utterances[0].metadata.clone()), json!({"page": 3}));
        assert_eq!(utterances[1].timestamp, json!("00:01"));
    }

    #[test]
    fn json_variants() {
        let dir = tempfile::tempdir().expect("tempdir");
        let single = dir.path().join("single.json");
        std::fs::write(&single, r#"{"speaker": "Solo", "text": "hi"}"#).expect("write");
        let utterances = parse_json(&single).expect("parse");
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].role, json!("unknown"));
        assert_eq!(utterances[0].timestamp, Value::Null);

        let lines = dir.path().join("lines.json");
        std::fs::write(
            &lines,
            "{\"speaker\": \"A\", \"text\": \"one\"}\r\n\r\nnot json\n{\"speaker\": \"B\", \"text\": \"two\"}\n",
        )
        .expect("write");
        assert_eq!(parse_json(&lines).expect("parse").len(), 2);

        let empty = dir.path().join("empty.json");
        std::fs::write(&empty, "").expect("write");
        assert!(parse_json(&empty).expect("parse").is_empty());

        let scalars = dir.path().join("scalars.json");
        std::fs::write(&scalars, "[1, 2]").expect("write");
        assert!(matches!(parse_json(&scalars), Err(IngestError::InvalidTranscript { .. })));

        let non_finite = dir.path().join("non_finite.json");
        std::fs::write(
            &non_finite,
            "[\n  {\"speaker\": \"A\", \"text\": \"NaN or -Infinity\", \"timestamp\": NaN},\n  \
             {\"speaker\": -Infinity, \"text\": \"bye\", \"page\": Infinity}\n]",
        )
        .expect("write");
        let error = parse_json(&non_finite).expect_err("non-finite speaker");
        assert!(
            matches!(&error, IngestError::InvalidTranscript { reason, .. } if reason.contains("-inf")),
            "{error:?}"
        );

        let non_finite_values = dir.path().join("non_finite_values.json");
        std::fs::write(
            &non_finite_values,
            "[{\"speaker\": \"A\", \"text\": NaN, \"timestamp\": -Infinity, \"page\": [Infinity]}]",
        )
        .expect("write");
        let utterances = parse_json(&non_finite_values).expect("parse");
        assert_eq!(utterances[0].text, json!("nan"));
        assert_eq!(utterances[0].timestamp, json!("-inf"));
        assert_eq!(Value::Object(utterances[0].metadata.clone()), json!({"page": ["inf"]}));
    }

    #[test]
    fn parse_file_dispatches_csv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.csv");
        std::fs::write(&path, "speaker,role,text\nMS. JONES,ATTORNEY,Hello\n").expect("write");
        let utterances = parse_file(&path).expect("parse");
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].speaker, json!("MS. JONES"));
        assert_eq!(utterances[0].text, json!("Hello"));
    }

    #[test]
    fn csv_rows_follow_dict_reader() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("extra.csv");
        std::fs::write(
            &path,
            "speaker,role,text\nA,b,c,d,e\n\n\"multi\r\nline\",r,\"t\r\nx\"\n,,\nshort\n",
        )
        .expect("write");
        let utterances = parse_csv(&path).expect("parse");
        assert_eq!(utterances.len(), 4);
        assert_eq!(Value::Object(utterances[0].metadata.clone()), json!({"null": ["d", "e"]}));
        assert_eq!(utterances[1].speaker, json!("multi\nline"));
        assert_eq!(utterances[1].text, json!("t\nx"));
        assert_eq!(utterances[3].role, Value::Null);
        let flat = flatten(&utterances).expect("flatten");
        assert_eq!(flat.text, "A (b): c\nmulti\nline (r): t\nx\n: \nshort: None");
        assert_eq!(
            flat.unique_speakers,
            [json!(""), json!("A"), json!("multi\nline"), json!("short")]
        );

        let blank_header = dir.path().join("blank.csv");
        std::fs::write(&blank_header, "\nspeaker,role,text\nA,b,c\n").expect("write");
        let flat = flatten(&parse_csv(&blank_header).expect("parse")).expect("flatten");
        assert_eq!(flat.text, "unknown: \nunknown: ");
        assert_eq!(flat.unique_speakers, [json!("unknown")]);

        let bom = dir.path().join("bom.csv");
        std::fs::write(&bom, "\u{feff}speaker,role,text\nA,b,c\n").expect("write");
        let flat = flatten(&parse_csv(&bom).expect("parse")).expect("flatten");
        assert_eq!(flat.text, "A (b): c");
    }

    fn utterance(speaker: Value, role: Value, text: Value) -> Utterance {
        Utterance {
            speaker,
            role,
            text,
            timestamp: Value::Null,
            line_number: None,
            metadata: Metadata::new(),
        }
    }

    // Expected values produced by the Python run_ingestion on the same utterances.
    #[test]
    fn flattens_non_string_values_like_python() {
        let utterances = [
            utterance(json!(2), json!(true), json!(1.5)),
            utterance(json!(1.0), json!("unknown"), json!([1, "a"])),
            utterance(json!(true), Value::Null, json!({"k": null})),
        ];
        let flat = flatten(&utterances).expect("flatten");
        assert_eq!(flat.text, "2 (True): 1.5\n1.0: [1, 'a']\nTrue: {'k': None}");
        assert_eq!(flat.unique_speakers, [json!(1.0), json!(2)]);

        let big = [
            utterance(json!(1e20), Value::Null, json!(1e-7)),
            utterance(json!(123_456_789_012_345_678_u64), Value::Null, json!(0.1)),
        ];
        let flat = flatten(&big).expect("flatten");
        assert_eq!(flat.text, "1e+20: 1e-07\n123456789012345678: 0.1");
        assert_eq!(flat.unique_speakers, [json!(123_456_789_012_345_678_u64), json!(1e20)]);
    }

    #[test]
    fn flatten_rejects_what_python_cannot_sort_or_hash() {
        let mixed = [
            utterance(json!("A"), Value::Null, json!("")),
            utterance(Value::Null, Value::Null, json!("")),
        ];
        assert!(flatten(&mixed).is_err());
        let unhashable = [utterance(json!(["A"]), Value::Null, json!(""))];
        assert!(flatten(&unhashable).is_err());
        let lone_null = [utterance(Value::Null, Value::Null, Value::Null)];
        let flat = flatten(&lone_null).expect("flatten");
        assert_eq!(flat.text, "None: None");
        assert_eq!(flat.unique_speakers, [Value::Null]);
    }

    #[test]
    fn utterance_serializes_in_python_key_order() {
        let utterances = parse_plain_text_str("MR. A: Hi.\n");
        let json = serde_json::to_string(&utterances[0]).expect("serialize");
        assert_eq!(
            json,
            r#"{"speaker":"MR. A","role":"ATTORNEY","text":"Hi.","timestamp":null,"line_number":1,"metadata":{"marker":"speaker_line"}}"#
        );
    }
}
