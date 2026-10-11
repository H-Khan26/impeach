//! Salient value extraction (numbers, money, dates, years, clock times) and the value guard.
//!
//! Python origin: `tests/e2e/evaluate_output.py` (`extract_values`, `_values_agree`,
//! `values_compatible`). The evaluator uses these to reject embedding matches whose key and
//! candidate texts mention disagreeing values.
//!
//! Regex semantics follow Python `re` with these documented differences: digits are ASCII
//! `[0-9]` (Python `\d` also accepts other Unicode decimal digits) and `\b` uses the Rust regex
//! Unicode word definition. Python's look-around in the number pattern is emulated by a
//! hand-written scanner that reproduces its backtracking.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use regex::{Captures, Regex};

use crate::normalize::is_py_whitespace;

/// A float with total equality and ordering, so values can live in sets. Only finite,
/// non-negative values parsed from decimal digits are constructed here.
#[derive(Debug, Clone, Copy)]
pub struct Number(f64);

impl Number {
    /// The wrapped value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Number {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for Number {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// One extracted value; mirrors the Python tuples `("date", year|None, month, day|None)`,
/// `("year", y)`, `("time", h, m, ampm|None)` and `("num", float)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// A calendar date; the month is not range-checked for numeric dates.
    Date {
        /// Four-digit year, or two-digit year plus 2000.
        year: Option<u32>,
        /// Month number.
        month: u32,
        /// Day of month.
        day: Option<u32>,
    },
    /// A bare year 1900-2099.
    Year(u32),
    /// A clock time; `hour` is taken modulo 12.
    Time {
        /// Hour modulo 12.
        hour: u32,
        /// Minutes.
        minute: u32,
        /// Lowercase `'a'` or `'p'` when an a.m./p.m. marker is present.
        meridiem: Option<char>,
    },
    /// Any other number, including money amounts and the number words two to twelve.
    Num(Number),
}

const MONTH: &str = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?";

// Python `\s` on str patterns: Unicode White_Space plus U+001C..U+001F.
macro_rules! ws {
    () => {
        r"[\s\x1C-\x1F]"
    };
}

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).expect("value-guard patterns are valid")
}

static DATE_MDY: LazyLock<Regex> = LazyLock::new(|| {
    compile(&format!(
        concat!(
            r"(?i){}",
            ws!(),
            r"+([0-9]{{1,2}})(?:st|nd|rd|th)?(?:,?",
            ws!(),
            r"*([0-9]{{4}}))?"
        ),
        MONTH
    ))
});
static DATE_MY: LazyLock<Regex> =
    LazyLock::new(|| compile(&format!(concat!(r"(?i){},?", ws!(), r"+([0-9]{{4}})"), MONTH)));
static DATE_NUMERIC: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\b([0-9]{1,2})/([0-9]{1,2})/([0-9]{2,4})\b"));
static TIME_12: LazyLock<Regex> = LazyLock::new(|| {
    compile(concat!(r"(?i)\b([0-9]{1,2})(?::([0-9]{2}))?", ws!(), r"*([ap])\.?", ws!(), r"*m\b\.?"))
});
static TIME_24: LazyLock<Regex> = LazyLock::new(|| compile(r"\b([0-9]{1,2}):([0-9]{2})\b"));
static YEAR: LazyLock<Regex> = LazyLock::new(|| compile(r"\b(19[0-9]{2}|20[0-9]{2})\b"));
static NUMBER_WORD: LazyLock<Regex> = LazyLock::new(|| {
    compile(r"(?i)\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b")
});

fn month_number(name: &str) -> Option<u32> {
    let prefix: String = name.to_lowercase().chars().take(3).collect();
    let month = match prefix.as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    };
    Some(month)
}

fn number_word(word: &str) -> Option<u32> {
    let value = match word.to_lowercase().as_str() {
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        _ => return None,
    };
    Some(value)
}

fn group_u32(caps: &Captures<'_>, index: usize) -> Option<u32> {
    caps.get(index).and_then(|m| m.as_str().parse().ok())
}

/// Runs `regex` over `buf`, adds the value `read` returns for each match, and blanks (replaces
/// with one space per character) every match that produced a value, so later patterns cannot
/// re-read its parts. Matches are collected before blanking, as in the Python
/// `for m in list(regex.finditer(buf))` loops.
fn take_and_blank<F>(buf: &mut String, regex: &Regex, values: &mut BTreeSet<Value>, read: F)
where
    F: Fn(&Captures<'_>) -> Option<Value>,
{
    let mut spans = Vec::new();
    for caps in regex.captures_iter(buf) {
        if let Some(value) = read(&caps) {
            values.insert(value);
            let whole = caps.get(0).expect("group 0 always participates");
            spans.push((whole.start(), whole.end()));
        }
    }
    for (start, end) in spans.into_iter().rev() {
        let blank = " ".repeat(buf[start..end].chars().count());
        buf.replace_range(start..end, &blank);
    }
}

/// Python `\w` on str patterns.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Emulates `(?<![\w.])\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])` anchored at
/// `start`, returning the match end, the whole-number digits (commas removed) and the fraction
/// digits. Candidates are tried in the order a backtracking engine tries them.
fn match_number_at(chars: &[char], start: usize) -> Option<(usize, String, String)> {
    if start > 0 && (is_word(chars[start - 1]) || chars[start - 1] == '.') {
        return None;
    }
    let n = chars.len();
    let mut pos = start;
    if pos < n && chars[pos] == '$' {
        pos += 1;
    }
    if pos < n && is_py_whitespace(chars[pos]) {
        pos += 1;
    }
    let digits_from =
        |from: usize| chars[from.min(n)..].iter().take_while(|c| c.is_ascii_digit()).count();
    let run = digits_from(pos);
    if run == 0 {
        return None;
    }

    // Whole-part end positions in backtracking order.
    let mut whole_ends = Vec::new();
    // Alternative 1: \d{1,3}(?:,\d{3})+ ; shorter leads leave a digit where ',' is required.
    if run <= 3 {
        let mut groups_end = vec![];
        let mut p = pos + run;
        while p < n && chars[p] == ',' && digits_from(p + 1) >= 3 {
            p += 4;
            groups_end.push(p);
        }
        whole_ends.extend(groups_end.into_iter().rev());
    }
    // Alternative 2: \d+ , greedy.
    whole_ends.extend((1..=run).rev().map(|len| pos + len));

    let lookahead_ok = |end: usize| end >= n || !is_word(chars[end]);
    for whole_end in whole_ends {
        let whole: String = chars[pos..whole_end].iter().filter(|c| **c != ',').collect();
        if whole_end < n && chars[whole_end] == '.' {
            let frac_run = digits_from(whole_end + 1);
            for len in (1..=frac_run).rev() {
                let end = whole_end + 1 + len;
                if lookahead_ok(end) {
                    let frac: String = chars[whole_end + 1..end].iter().collect();
                    return Some((end, whole, frac));
                }
            }
        }
        if lookahead_ok(whole_end) {
            return Some((whole_end, whole, String::new()));
        }
    }
    None
}

/// All numbers matched by the evaluator's `_NUMBER_RE`, scanning left to right without
/// overlap like `finditer`.
fn scan_numbers(buf: &str) -> Vec<f64> {
    let chars: Vec<char> = buf.chars().collect();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        if let Some((end, whole, frac)) = match_number_at(&chars, pos) {
            let frac = frac.trim_end_matches('0');
            let literal = if frac.is_empty() { whole } else { format!("{whole}.{frac}") };
            if let Ok(value) = literal.parse::<f64>() {
                out.push(value);
            }
            pos = end;
        } else {
            pos += 1;
        }
    }
    out
}

/// Extracts the salient values of `text`.
///
/// Patterns run in this order, each on the text left after blanking the matches of the
/// previous ones: month-day[-year] dates (day 1-31, else ignored and not blanked), month-year
/// dates, numeric `m/d/y` dates (two-digit years + 2000), 12-hour times, 24-hour times, years
/// 1900-2099; then plain numbers (optional `$`, thousands separators, decimals with trailing
/// zeros dropped) and the number words two to twelve, which are not blanked.
///
/// Python origin: `tests/e2e/evaluate_output.py:extract_values`. A month name Python cannot
/// map (only reachable through Unicode case folding, e.g. `ſep`) raises `KeyError` there and is
/// skipped here.
#[must_use]
pub fn extract_values(text: &str) -> BTreeSet<Value> {
    let mut values = BTreeSet::new();
    if text.is_empty() {
        return values;
    }
    let mut buf = text.to_owned();

    take_and_blank(&mut buf, &DATE_MDY, &mut values, |caps| {
        let month = month_number(caps.get(1)?.as_str())?;
        let day = group_u32(caps, 2)?;
        if !(1..=31).contains(&day) {
            return None;
        }
        Some(Value::Date { year: group_u32(caps, 3), month, day: Some(day) })
    });
    take_and_blank(&mut buf, &DATE_MY, &mut values, |caps| {
        let month = month_number(caps.get(1)?.as_str())?;
        Some(Value::Date { year: group_u32(caps, 2), month, day: None })
    });
    take_and_blank(&mut buf, &DATE_NUMERIC, &mut values, |caps| {
        let year = group_u32(caps, 3)?;
        let year = if year < 100 { year + 2000 } else { year };
        Some(Value::Date { year: Some(year), month: group_u32(caps, 1)?, day: group_u32(caps, 2) })
    });
    take_and_blank(&mut buf, &TIME_12, &mut values, |caps| {
        let meridiem = caps.get(3)?.as_str().chars().next()?.to_ascii_lowercase();
        Some(Value::Time {
            hour: group_u32(caps, 1)? % 12,
            minute: group_u32(caps, 2).unwrap_or(0),
            meridiem: Some(meridiem),
        })
    });
    take_and_blank(&mut buf, &TIME_24, &mut values, |caps| {
        Some(Value::Time {
            hour: group_u32(caps, 1)? % 12,
            minute: group_u32(caps, 2)?,
            meridiem: None,
        })
    });
    take_and_blank(&mut buf, &YEAR, &mut values, |caps| Some(Value::Year(group_u32(caps, 1)?)));

    for number in scan_numbers(&buf) {
        values.insert(Value::Num(Number(number)));
    }
    for caps in NUMBER_WORD.captures_iter(&buf) {
        if let Some(number) = caps.get(1).and_then(|m| number_word(m.as_str())) {
            values.insert(Value::Num(Number(f64::from(number))));
        }
    }
    values
}

/// True when two values can describe the same thing: equal numbers; equal clock times whose
/// a.m./p.m. markers do not conflict; equal years; a year and a date whose year is missing or
/// equal; dates with equal months whose years and days do not conflict. Values of other kind
/// pairs never agree.
///
/// Python origin: `tests/e2e/evaluate_output.py:_values_agree`.
#[must_use]
pub fn values_agree(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (
            Value::Time { hour: h1, minute: m1, meridiem: p1 },
            Value::Time { hour: h2, minute: m2, meridiem: p2 },
        ) => h1 == h2 && m1 == m2 && (p1.is_none() || p2.is_none() || p1 == p2),
        (Value::Year(x), Value::Year(y)) => x == y,
        (Value::Year(y), Value::Date { year, .. }) | (Value::Date { year, .. }, Value::Year(y)) => {
            year.is_none_or(|d| d == *y)
        }
        (
            Value::Date { year: y1, month: m1, day: d1 },
            Value::Date { year: y2, month: m2, day: d2 },
        ) => {
            if m1 != m2 {
                return false;
            }
            if let (Some(y1), Some(y2)) = (y1, y2) {
                if y1 != y2 {
                    return false;
                }
            }
            d1.is_none() || d2.is_none() || d1 == d2
        }
        _ => false,
    }
}

/// True unless both sides carry salient values and none of them agree.
///
/// Python origin: `tests/e2e/evaluate_output.py:values_compatible`.
#[must_use]
pub fn values_compatible(key_values: &BTreeSet<Value>, cand_values: &BTreeSet<Value>) -> bool {
    if key_values.is_empty() || cand_values.is_empty() {
        return true;
    }
    key_values.iter().any(|k| cand_values.iter().any(|c| values_agree(k, c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(value: f64) -> Value {
        Value::Num(Number(value))
    }

    fn date(year: Option<u32>, month: u32, day: Option<u32>) -> Value {
        Value::Date { year, month, day }
    }

    fn set(items: &[Value]) -> BTreeSet<Value> {
        items.iter().copied().collect()
    }

    #[test]
    fn extracts_dates_and_blanks_them() {
        assert_eq!(
            extract_values("On March 3rd, 2021 he paid $1,250.50"),
            set(&[date(Some(2021), 3, Some(3)), num(1250.5)])
        );
        assert_eq!(extract_values("in Sept., 2019"), set(&[date(Some(2019), 9, None)]));
        // The month-day pattern reads "Sept. 20" first, as in Python.
        assert_eq!(extract_values("in Sept. 2019"), set(&[date(None, 9, Some(20)), num(19.0)]));
        assert_eq!(extract_values("on 3/4/21"), set(&[date(Some(2021), 3, Some(4))]));
        assert_eq!(extract_values("May 45"), set(&[num(45.0)]));
    }

    #[test]
    fn extracts_times_years_and_words() {
        assert_eq!(
            extract_values("at 10:30 p.m. in 1999, five cars"),
            set(&[
                Value::Time { hour: 10, minute: 30, meridiem: Some('p') },
                Value::Year(1999),
                num(5.0)
            ])
        );
        assert_eq!(
            extract_values("at 14:05"),
            set(&[Value::Time { hour: 2, minute: 5, meridiem: None }])
        );
        assert_eq!(
            extract_values("at 12 AM"),
            set(&[Value::Time { hour: 0, minute: 0, meridiem: Some('a') }])
        );
    }

    #[test]
    fn number_lookaround_matches_python() {
        assert_eq!(extract_values("1.5a"), set(&[num(1.0)]));
        assert_eq!(extract_values("x12 a1"), set(&[]));
        assert_eq!(extract_values("1,2345"), set(&[num(1.0), num(2345.0)]));
        assert_eq!(extract_values("1,234,5678"), set(&[num(1234.0), num(5678.0)]));
        assert_eq!(extract_values("2.25.15"), set(&[num(2.25)]));
        assert_eq!(extract_values("$ 40.00"), set(&[num(40.0)]));
        assert_eq!(extract_values("v.2"), set(&[]));
    }

    #[test]
    fn empty_text_has_no_values() {
        assert!(extract_values("").is_empty());
        assert!(extract_values("no values here").is_empty());
    }

    #[test]
    fn agreement_rules() {
        let t = |hour, minute, meridiem| Value::Time { hour, minute, meridiem };
        assert!(values_agree(&num(2.0), &num(2.0)));
        assert!(!values_agree(&num(2.0), &num(3.0)));
        assert!(values_agree(&t(3, 0, Some('p')), &t(3, 0, None)));
        assert!(!values_agree(&t(3, 0, Some('p')), &t(3, 0, Some('a'))));
        assert!(values_agree(&Value::Year(2020), &date(None, 1, Some(5))));
        assert!(values_agree(&date(Some(2020), 1, None), &Value::Year(2020)));
        assert!(!values_agree(&Value::Year(2021), &date(Some(2020), 1, None)));
        assert!(values_agree(&date(None, 1, Some(5)), &date(Some(2020), 1, None)));
        assert!(!values_agree(&date(Some(2020), 1, Some(5)), &date(Some(2020), 1, Some(6))));
        assert!(!values_agree(&date(Some(2020), 1, Some(5)), &date(Some(2020), 2, Some(5))));
        assert!(!values_agree(&num(2020.0), &Value::Year(2020)));
    }

    #[test]
    fn compatibility_guard() {
        assert!(values_compatible(&set(&[]), &set(&[num(1.0)])));
        assert!(values_compatible(&set(&[num(1.0), num(2.0)]), &set(&[num(2.0)])));
        assert!(!values_compatible(&set(&[num(1.0)]), &set(&[num(2.0)])));
    }
}
