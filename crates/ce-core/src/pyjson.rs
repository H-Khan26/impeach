//! JSON in the byte format of Python's `json.dumps`: `", "` / `": "` separators, floats as Python
//! `repr` (`1e+20`, `1.0`), and non-ASCII text either kept (`ensure_ascii=False`) or escaped as
//! `\uXXXX` UTF-16 units (`ensure_ascii=True`, the Python default). Field order is the serde
//! order, so records serialise in Python field order.
//!
//! One deliberate difference: non-finite floats are written as `null` (what `serde_json` does),
//! not as Python's `NaN` / `Infinity`, which are not valid JSON.

use std::fmt::Write as _;
use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::{CharEscape, Formatter};

/// `serde_json` formatter reproducing `json.dumps` output.
#[derive(Debug, Clone, Copy, Default)]
pub struct PythonFormatter {
    /// Escape every character outside printable ASCII as `\uXXXX` (Python `ensure_ascii=True`).
    pub ensure_ascii: bool,
}

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        writer.write_all(float_repr(f64::from(value)).as_bytes())
    }

    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii || fragment.bytes().all(|b| (b' '..=b'~').contains(&b)) {
            return writer.write_all(fragment.as_bytes());
        }
        let mut out = String::with_capacity(fragment.len() + 16);
        for c in fragment.chars() {
            if (' '..='~').contains(&c) {
                out.push(c);
            } else {
                let mut units = [0_u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    // Writing to a String cannot fail.
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
        writer.write_all(out.as_bytes())
    }

    fn write_char_escape<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        char_escape: CharEscape,
    ) -> io::Result<()> {
        let escaped: &[u8] = match char_escape {
            CharEscape::Quote => b"\\\"",
            CharEscape::ReverseSolidus => b"\\\\",
            CharEscape::Solidus => b"/",
            CharEscape::Backspace => b"\\b",
            CharEscape::FormFeed => b"\\f",
            CharEscape::LineFeed => b"\\n",
            CharEscape::CarriageReturn => b"\\r",
            CharEscape::Tab => b"\\t",
            CharEscape::AsciiControl(byte) => return write!(writer, "\\u{byte:04x}"),
        };
        writer.write_all(escaped)
    }
}

fn to_string_with<T: Serialize + ?Sized>(
    value: &T,
    formatter: PythonFormatter,
) -> serde_json::Result<String> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    value.serialize(&mut serializer)?;
    String::from_utf8(out).map_err(serde::ser::Error::custom)
}

/// Python `json.dumps(value, ensure_ascii=False)`.
///
/// # Errors
/// The serialisation error of `value`.
pub fn to_python_json<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    to_string_with(value, PythonFormatter { ensure_ascii: false })
}

/// Python `json.dumps(value)` (`ensure_ascii=True`).
///
/// # Errors
/// The serialisation error of `value`.
pub fn to_python_json_ascii<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    to_string_with(value, PythonFormatter { ensure_ascii: true })
}

/// Python `repr()` (and `str()`) of a float, which is also what `json.dumps` writes for finite
/// values: the shortest round-trip digits, positional for decimal exponents in `-4..16`,
/// otherwise scientific with a signed two-digit exponent (`1e+20`, `1.5e-07`); `nan`, `inf`
/// and `-inf` for non-finite values.
#[must_use]
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let Some((sign, digits, exp)) = shortest_digits(value) else {
        return value.to_string();
    };
    let mut out = String::from(sign);
    if !(-4..16).contains(&exp) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exp_sign = if exp < 0 { '-' } else { '+' };
        // Writing to a String cannot fail.
        let _ = write!(out, "e{exp_sign}{:02}", exp.unsigned_abs());
    } else if let Ok(int_len) = usize::try_from(exp).map(|e| e + 1) {
        if digits.len() <= int_len {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', int_len - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..int_len]);
            out.push('.');
            out.push_str(&digits[int_len..]);
        }
    } else {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', exp.unsigned_abs() as usize - 1));
        out.push_str(&digits);
    }
    out
}

/// Sign, significant digits (no leading or trailing zeros) and decimal exponent of the shortest
/// round-trip representation of a finite non-zero `value`, as `ryu` (through `serde_json`)
/// produces it with round-half-even ties like Python's `repr`.
fn shortest_digits(value: f64) -> Option<(&'static str, String, i32)> {
    let shortest = serde_json::Number::from_f64(value)?.to_string();
    let (sign, unsigned) = match shortest.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", shortest.as_str()),
    };
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (unsigned, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all: String = int_part.chars().chain(frac_part.chars()).collect();
    let leading = all.len() - all.trim_start_matches('0').len();
    let digits = all.trim_matches('0').to_owned();
    if digits.is_empty() {
        return None;
    }
    let int_len = i32::try_from(int_part.len()).ok()?;
    let leading = i32::try_from(leading).ok()?;
    Some((sign, digits, exponent + int_len - 1 - leading))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn python_json_format() {
        let value =
            json!({"a": [1, 2.5, 1e20, "é\n\u{1}"], "b": {}, "c": [], "d": null, "e": true});
        assert_eq!(
            to_python_json(&value).expect("serialize"),
            "{\"a\": [1, 2.5, 1e+20, \"é\\n\\u0001\"], \"b\": {}, \"c\": [], \"d\": null, \"e\": true}"
        );
    }

    #[test]
    fn non_finite_floats_are_written_as_null() {
        let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5];
        assert_eq!(to_python_json(&values).expect("serialize"), "[null, null, null, 0.5]");
        assert_eq!(to_python_json_ascii(&f32::NAN).expect("serialize"), "null");
    }

    #[test]
    fn ensure_ascii_escapes_like_python() {
        let value =
            json!({"a": [1, 2.5, null, true], "b": {"c": "Ünïcode \u{1F600} \"q\"\n\u{7f}/"}});
        assert_eq!(
            to_python_json_ascii(&value).expect("serialize"),
            r#"{"a": [1, 2.5, null, true], "b": {"c": "\u00dcn\u00efcode \ud83d\ude00 \"q\"\n\u007f/"}}"#
        );
        assert_eq!(to_python_json_ascii(&json!({})).expect("serialize"), "{}");
        assert_eq!(to_python_json_ascii(&json!("\u{1}")).expect("serialize"), r#""\u0001""#);
    }

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1e20, "1e+20"),
            (1e-7, "1e-07"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1_234_567_890_123_456.0, "1234567890123456.0"),
            (0.0001, "0.0001"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-2.5, "-2.5"),
            (123.456, "123.456"),
            (-0.0, "-0.0"),
            (1.0e300, "1e+300"),
            (2.0e-308, "2e-308"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (2_163_462_279_744_188.0 + 0.25, "2163462279744188.2"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.000123, "0.000123"),
            (123_456_789.0, "123456789.0"),
        ];
        for (value, expected) in cases {
            assert_eq!(float_repr(value), expected, "{value}");
        }
    }
}
