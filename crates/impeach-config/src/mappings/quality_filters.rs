//! Mention quality filters (`quality_filters.yaml`).

use regex::Regex;
use serde::Deserialize;

use super::{build_regex, RegexFlag};
use crate::error::{ConfigError, Result};

/// Thresholds and patterns for dropping low-quality mentions.
///
/// Patterns are compiled when the registry loads and match at the start of the name
/// (Python `re.match` semantics).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QualityFilters {
    /// Minimum mention length in characters; defaults to 3.
    pub min_mention_length: usize,
    /// Names shorter than this need `short_mention_confidence_threshold`; defaults to 6.
    pub short_mention_length: usize,
    /// Minimum score for short names; defaults to 0.75.
    pub short_mention_confidence_threshold: f64,
    /// Truncated-word pattern from chunk boundaries.
    pub partial_word_patterns: Option<PartialWordPattern>,
    /// Further patterns that mark a name as low quality.
    pub additional_filter_patterns: Vec<FilterPattern>,
    /// Collapse whitespace in names; defaults to true.
    pub normalize_names: bool,
    /// Apply the partial-word pattern; defaults to true.
    pub filter_partial_words: bool,
    #[serde(skip)]
    partial_word_regex: Option<Regex>,
    #[serde(skip)]
    additional_regexes: Vec<Regex>,
}

/// `partial_word_patterns:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialWordPattern {
    /// Regex source; an empty pattern disables the filter.
    #[serde(default)]
    pub regex: Option<String>,
    /// `IGNORECASE` or `NONE`.
    #[serde(default)]
    pub flags: Option<RegexFlag>,
    /// Free-text description.
    #[serde(default)]
    pub description: Option<String>,
}

/// One entry of `additional_filter_patterns:`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterPattern {
    /// Regex source.
    pub pattern: String,
    /// `IGNORECASE` or `NONE`.
    #[serde(default)]
    pub flags: Option<RegexFlag>,
    /// Free-text description.
    #[serde(default)]
    pub description: Option<String>,
}

impl Default for QualityFilters {
    fn default() -> Self {
        Self {
            min_mention_length: 3,
            short_mention_length: 6,
            short_mention_confidence_threshold: 0.75,
            partial_word_patterns: None,
            additional_filter_patterns: Vec::new(),
            normalize_names: true,
            filter_partial_words: true,
            partial_word_regex: None,
            additional_regexes: Vec::new(),
        }
    }
}

impl PartialEq for QualityFilters {
    fn eq(&self, other: &Self) -> bool {
        self.min_mention_length == other.min_mention_length
            && self.short_mention_length == other.short_mention_length
            && self.short_mention_confidence_threshold == other.short_mention_confidence_threshold
            && self.partial_word_patterns == other.partial_word_patterns
            && self.additional_filter_patterns == other.additional_filter_patterns
            && self.normalize_names == other.normalize_names
            && self.filter_partial_words == other.filter_partial_words
    }
}

impl QualityFilters {
    /// Compiles the configured patterns; `file` names the source in errors.
    pub(crate) fn compile(&mut self, file: &str) -> Result<()> {
        let regex_error = |name: String, source: regex::Error| ConfigError::Regex {
            file: file.to_owned(),
            name,
            source: Box::new(source),
        };
        self.partial_word_regex = match &self.partial_word_patterns {
            Some(PartialWordPattern { regex: Some(source), flags, .. }) if !source.is_empty() => {
                Some(
                    build_regex(source, *flags, true)
                        .map_err(|e| regex_error("partial_word_patterns".into(), e))?,
                )
            }
            _ => None,
        };
        self.additional_regexes = self
            .additional_filter_patterns
            .iter()
            .enumerate()
            .map(|(i, p)| {
                build_regex(&p.pattern, p.flags, true)
                    .map_err(|e| regex_error(format!("additional_filter_patterns[{i}]"), e))
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Whether a mention should be dropped: too short, a truncated word, short with a low
    /// score, or matching an additional filter pattern. `name` is trimmed first.
    pub fn is_low_quality(&self, name: &str, score: Option<f64>) -> bool {
        let name = name.trim();
        let len = name.chars().count();
        if len < self.min_mention_length {
            return true;
        }
        if self.filter_partial_words
            && self.partial_word_regex.as_ref().is_some_and(|re| re.is_match(name))
        {
            return true;
        }
        if let Some(score) = score {
            if len < self.short_mention_length && score < self.short_mention_confidence_threshold {
                return true;
            }
        }
        self.additional_regexes.iter().any(|re| re.is_match(name))
    }

    /// Collapses whitespace runs to one space and strips the ends (when `normalize_names` is
    /// set), with Python `str.split()` whitespace semantics.
    pub fn normalize_mention_name(&self, name: &str) -> String {
        if !self.normalize_names {
            return name.to_owned();
        }
        impeach_text::normalize_whitespace(name)
    }
}
