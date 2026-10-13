//! Fuzzy string similarity matching `rapidfuzz.fuzz` (0-100) and `difflib` (0-1).
//!
//! The Python prototype calls `difflib.SequenceMatcher(None, a, b).ratio()` in the entity
//! disambiguation services (`services/ed/name_similarity.py`, `services/ed/service.py`); it does
//! not call `rapidfuzz` anywhere. The `rapidfuzz` scorers are provided with their exact
//! semantics for parity with tooling that uses them. `strsim` offers none of these metrics
//! (it has Levenshtein, Jaro, Dice, but no Indel/LCS or Ratcliff/Obershelp), so all are
//! implemented here.
//!
//! All scorers compare Unicode scalar values (Python `str` code points) with no preprocessing:
//! callers lowercase or strip themselves, as `rapidfuzz` 3.x does with `processor=None`.

use std::collections::{HashMap, HashSet};

use crate::normalize::py_split;

/// Bit-parallel longest-common-subsequence matcher for a fixed pattern (Hyyrö 2004), the
/// algorithm rapidfuzz uses for Indel distances.
struct LcsPattern {
    len: usize,
    words: usize,
    masks: HashMap<char, Vec<u64>>,
}

impl LcsPattern {
    fn new(pattern: &[char]) -> Self {
        let words = pattern.len().div_ceil(64);
        let mut masks: HashMap<char, Vec<u64>> = HashMap::new();
        for (i, &c) in pattern.iter().enumerate() {
            masks.entry(c).or_insert_with(|| vec![0; words])[i / 64] |= 1 << (i % 64);
        }
        Self { len: pattern.len(), words, masks }
    }

    /// Length of the longest common subsequence of the pattern and `text`.
    fn lcs(&self, text: &[char]) -> usize {
        if self.len == 0 || text.is_empty() {
            return 0;
        }
        let mut v = vec![u64::MAX; self.words];
        for c in text {
            let Some(mask) = self.masks.get(c) else {
                continue;
            };
            let mut carry = false;
            for (word, &m) in v.iter_mut().zip(mask) {
                let u = *word & m;
                let (sum, c1) = word.overflowing_add(u);
                let (sum, c2) = sum.overflowing_add(u64::from(carry));
                carry = c1 || c2;
                *word = sum | (*word - u);
            }
        }
        let tail_bits = self.len % 64;
        v.iter()
            .enumerate()
            .map(|(i, &word)| {
                let valid = if i + 1 == self.words && tail_bits != 0 {
                    (1_u64 << tail_bits) - 1
                } else {
                    u64::MAX
                };
                (!word & valid).count_ones() as usize
            })
            .sum()
    }

    /// Normalized Indel similarity (0-1) of the pattern and `text`; 1.0 when both are empty.
    fn similarity(&self, text: &[char]) -> f64 {
        let lensum = self.len + text.len();
        if lensum == 0 {
            return 1.0;
        }
        let distance = lensum - 2 * self.lcs(text);
        1.0 - distance as f64 / lensum as f64
    }
}

/// Length of the longest common subsequence of `a` and `b`.
fn lcs_len(a: &[char], b: &[char]) -> usize {
    let (pattern, text) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    LcsPattern::new(pattern).lcs(text)
}

/// Indel distance (insertions and deletions only, no substitutions):
/// `len(a) + len(b) - 2 * lcs(a, b)`.
///
/// Matches `rapidfuzz.distance.Indel.distance`.
#[must_use]
pub fn indel_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    indel_distance_chars(&a, &b)
}

fn indel_distance_chars(a: &[char], b: &[char]) -> usize {
    a.len() + b.len() - 2 * lcs_len(a, b)
}

/// Normalized Indel similarity on a 0-1 scale; 1.0 when both are empty.
fn indel_normalized_similarity(a: &[char], b: &[char]) -> f64 {
    let lensum = a.len() + b.len();
    if lensum == 0 {
        return 1.0;
    }
    1.0 - indel_distance_chars(a, b) as f64 / lensum as f64
}

/// `rapidfuzz.fuzz.ratio`: normalized Indel similarity, 0-100 scale.
///
/// `100 * (1 - indel_distance / (len(a) + len(b)))`, 100 for two empty strings. This is not a
/// Levenshtein ratio: a substitution costs 2. Equals `2 * lcs / (len(a) + len(b)) * 100`.
#[must_use]
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    indel_normalized_similarity(&a, &b) * 100.0
}

/// Best normalized Indel similarity (0-1) of `s1` against the alignments of `s1` in `s2` that
/// can be optimal. Requires `s1.len() <= s2.len()`. Port of
/// `rapidfuzz.fuzz_py._partial_ratio_impl`.
fn partial_ratio_impl(s1: &[char], s2: &[char]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();
    if len1 == 0 {
        return 0.0;
    }
    let chars1: HashSet<char> = s1.iter().copied().collect();
    // Windows growing from the left edge, full-length windows, then windows shrinking towards
    // the right edge of s2. A window is skipped when its moving edge is not in s1.
    let left = (1..len1).filter(|&i| chars1.contains(&s2[i - 1])).map(|i| &s2[..i]);
    let full =
        (0..len2 - len1).filter(|&i| chars1.contains(&s2[i + len1 - 1])).map(|i| &s2[i..i + len1]);
    let right = (len2 - len1..len2).filter(|&i| chars1.contains(&s2[i])).map(|i| &s2[i..]);
    let pattern = LcsPattern::new(s1);
    let mut best = 0.0_f64;
    for window in left.chain(full).chain(right) {
        let score = pattern.similarity(window);
        if score > best {
            best = score;
            if best >= 1.0 {
                return 1.0;
            }
        }
    }
    best
}

/// `rapidfuzz.fuzz.partial_ratio`: the best [`ratio`] of the shorter string against any
/// alignment in the longer one (including alignments hanging off either end), 0-100 scale.
///
/// Both empty gives 100; exactly one empty gives 0. When both have the same length the search
/// runs in both directions and the maximum is kept, as in `rapidfuzz` 3.x.
///
/// Implements rapidfuzz's exhaustive short-needle algorithm for every needle length. For
/// needles longer than 64 characters the rapidfuzz C++ build switches to a heuristic that
/// only tries alignments starting at matching blocks and can score lower than this function;
/// the pure-Python `rapidfuzz.fuzz_py.partial_ratio` matches this function for all lengths.
#[must_use]
pub fn partial_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 100.0;
    }
    let (shorter, longer) = if a.len() <= b.len() { (&a, &b) } else { (&b, &a) };
    let mut best = partial_ratio_impl(shorter, longer);
    if best < 1.0 && a.len() == b.len() {
        best = best.max(partial_ratio_impl(longer, shorter));
    }
    best * 100.0
}

fn sorted_tokens(text: &str) -> Vec<&str> {
    let mut tokens: Vec<&str> = py_split(text).collect();
    tokens.sort_unstable();
    tokens
}

/// `rapidfuzz.fuzz.token_sort_ratio`: [`ratio`] of the whitespace-split tokens sorted and
/// re-joined with single spaces, 0-100 scale. Tokens sort by code point, like Python `sorted`.
#[must_use]
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    ratio(&sorted_tokens(a).join(" "), &sorted_tokens(b).join(" "))
}

fn norm_distance(dist: usize, lensum: usize) -> f64 {
    if lensum == 0 {
        100.0
    } else {
        100.0 - 100.0 * dist as f64 / lensum as f64
    }
}

/// Code-point length of the tokens joined by single spaces.
fn joined_len(tokens: &[&str]) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    tokens.iter().map(|t| t.chars().count()).sum::<usize>() + tokens.len() - 1
}

/// `rapidfuzz.fuzz.token_set_ratio`, 0-100 scale.
///
/// Splits both strings into sets of whitespace tokens. Returns 0 when either set is empty and
/// 100 when they share a token and one set contains the other. Otherwise returns the maximum of
/// the Indel similarity of the two sorted difference strings (each conceptually prefixed by the
/// sorted intersection) and of the intersection against each `intersection + difference`.
#[must_use]
pub fn token_set_ratio(a: &str, b: &str) -> f64 {
    let tokens_a: HashSet<&str> = py_split(a).collect();
    let tokens_b: HashSet<&str> = py_split(b).collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    let intersect: Vec<&str> = tokens_a.intersection(&tokens_b).copied().collect();
    let mut diff_ab: Vec<&str> = tokens_a.difference(&tokens_b).copied().collect();
    let mut diff_ba: Vec<&str> = tokens_b.difference(&tokens_a).copied().collect();
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    diff_ab.sort_unstable();
    diff_ba.sort_unstable();
    let diff_ab_joined: Vec<char> = diff_ab.join(" ").chars().collect();
    let diff_ba_joined: Vec<char> = diff_ba.join(" ").chars().collect();

    let ab_len = diff_ab_joined.len();
    let ba_len = diff_ba_joined.len();
    let sect_len = joined_len(&intersect);
    let separator = usize::from(sect_len != 0);
    let sect_ab_len = sect_len + separator + ab_len;
    let sect_ba_len = sect_len + separator + ba_len;

    let dist = indel_distance_chars(&diff_ab_joined, &diff_ba_joined);
    let result = norm_distance(dist, sect_ab_len + sect_ba_len);
    if sect_len == 0 {
        return result;
    }
    let sect_ab_ratio = norm_distance(separator + ab_len, sect_len + sect_ab_len);
    let sect_ba_ratio = norm_distance(separator + ba_len, sect_len + sect_ba_len);
    result.max(sect_ab_ratio).max(sect_ba_ratio)
}

/// `difflib.SequenceMatcher(None, a, b).ratio()`: Ratcliff/Obershelp similarity `2 * M / T`
/// on a 0-1 scale, where `M` is the total size of the matching blocks and `T` the total
/// length; 1.0 when both are empty. Uses `autojunk=True` (the difflib default): when `b` has
/// at least 200 characters, characters occurring more than `1 + len(b) // 100` times are
/// "popular" and cannot seed a match (they can still extend one). Not symmetric in general.
///
/// Python origin of the call sites: `services/ed/name_similarity.py` (several functions) and
/// `services/ed/service.py:_string_similarity`.
#[must_use]
pub fn sequence_matcher_ratio(a: &str, b: &str) -> f64 {
    sequence_matcher_ratio_with(a, b, true)
}

/// [`sequence_matcher_ratio`] with an explicit `autojunk` flag.
#[must_use]
pub fn sequence_matcher_ratio_with(a: &str, b: &str, autojunk: bool) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let matches = SequenceMatcher::new(&a, &b, autojunk).matching_characters();
    2.0 * matches as f64 / total as f64
}

/// Port of `difflib.SequenceMatcher` with `isjunk=None`, limited to what `ratio()` needs.
struct SequenceMatcher<'a> {
    a: &'a [char],
    b: &'a [char],
    b2j: HashMap<char, Vec<usize>>,
}

impl<'a> SequenceMatcher<'a> {
    fn new(a: &'a [char], b: &'a [char], autojunk: bool) -> Self {
        let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, &elt) in b.iter().enumerate() {
            b2j.entry(elt).or_default().push(i);
        }
        let n = b.len();
        if autojunk && n >= 200 {
            let ntest = n / 100 + 1;
            b2j.retain(|_, indices| indices.len() <= ntest);
        }
        Self { a, b, b2j }
    }

    /// `find_longest_match(alo, ahi, blo, bhi)`. With no junk only the first pair of extension
    /// loops can fire; popular characters are not junk, so they still extend a match.
    fn find_longest_match(
        &self,
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
    ) -> (usize, usize, usize) {
        let (a, b) = (self.a, self.b);
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        let mut new_j2len: HashMap<usize, usize> = HashMap::new();
        for (i, ch) in a.iter().enumerate().take(ahi).skip(alo) {
            new_j2len.clear();
            if let Some(indices) = self.b2j.get(ch) {
                for &j in indices {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let prev = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0);
                    let k = prev + 1;
                    new_j2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            std::mem::swap(&mut j2len, &mut new_j2len);
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    /// Sum of the block sizes of `get_matching_blocks()`.
    fn matching_characters(&self) -> usize {
        let mut queue = vec![(0, self.a.len(), 0, self.b.len())];
        let mut total = 0;
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.find_longest_match(alo, ahi, blo, bhi);
            if k > 0 {
                total += k;
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn rapidfuzz_doc_examples() {
        assert!(close(ratio("this is a test", "this is a test!"), 96.551_724_137_931_03));
        assert!(close(partial_ratio("this is a test", "this is a test!"), 100.0));
        assert!(close(token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"), 100.0));
        assert!(close(token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear"), 100.0));
        assert!(close(
            token_set_ratio("fuzzy was a bear but not a dog", "fuzzy was a bear but not a cat"),
            92.307_692_307_692_3
        ));
    }

    #[test]
    fn empty_inputs() {
        assert!(close(ratio("", ""), 100.0));
        assert!(close(ratio("", "abc"), 0.0));
        assert!(close(partial_ratio("", ""), 100.0));
        assert!(close(partial_ratio("", "abc"), 0.0));
        assert!(close(partial_ratio("abc", ""), 0.0));
        assert!(close(token_set_ratio("", "abc"), 0.0));
        assert!(close(token_sort_ratio("", ""), 100.0));
        assert!(close(sequence_matcher_ratio("", ""), 1.0));
        assert!(close(sequence_matcher_ratio("", "abc"), 0.0));
    }

    #[test]
    fn ratio_is_indel_not_levenshtein() {
        // Levenshtein counts one substitution; Indel counts a deletion plus an insertion.
        assert!(close(ratio("abc", "abd"), 100.0 * (1.0 - 2.0 / 6.0)));
        assert_eq!(indel_distance("kitten", "sitting"), 5);
    }

    #[test]
    fn bit_parallel_lcs_spans_words() {
        let a: Vec<char> = "ab".repeat(70).chars().collect();
        let b: Vec<char> = "ba".repeat(70).chars().collect();
        assert_eq!(lcs_len(&a, &b), 139);
        let c: Vec<char> = "x".repeat(64).chars().collect();
        assert_eq!(lcs_len(&c, &c), 64);
        assert_eq!(lcs_len(&a, &c), 0);
    }

    #[test]
    fn difflib_basic() {
        assert!(close(sequence_matcher_ratio("abcd", "bcde"), 0.75));
        assert!(close(sequence_matcher_ratio("abc", "abc"), 1.0));
    }
}
