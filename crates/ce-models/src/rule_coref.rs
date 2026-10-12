//! Deterministic rule-based coreference: [`RuleCoref`], the `rule` backend of `models.coref`.
//!
//! The resolver works on the mentions passed to [`Coref::resolve_with_mentions`]; without
//! mentions ([`Coref::resolve`]) it finds nothing.
//!
//! 1. **Name matching** between mentions of the same entity type. Names are normalised:
//!    lowercased, `.` and apostrophes removed, other non-alphanumeric characters split words,
//!    leading articles and honorifics (`the`, `mr`, `dr`, ...) and trailing company designators
//!    (`inc`, `llc`, `corp`, ...) dropped. Two names match when their normalised token lists are
//!    equal, or when one is an alias of the other: its token set is a proper subset of the
//!    other's (`Lopez` / `Maria Lopez`) or it is an all-caps acronym of the other's tokens
//!    (`GVL` / `Green Valley Logistics`). An alias joins its longer names only when those names
//!    are nested in each other; an alias of two unrelated names (`Lopez` with `Maria Lopez` and
//!    `Carlos Lopez`) is ambiguous and stays unlinked.
//! 2. **Pronoun linking**: every pronoun in the text that does not overlap a mention, and every
//!    mention whose text is a pronoun, links to the nearest preceding compatible mention (or
//!    already linked pronoun of the same kind) that starts in the same sentence or at most
//!    [`PRONOUN_SENTENCE_WINDOW`] sentences earlier:
//!
//!    | Pronouns | Compatible entity types |
//!    | --- | --- |
//!    | `he him his she her hers` | person (`PER`, `PERSON`) |
//!    | `it its` | anything but person and temporal (`DATE`, `TIME`) |
//!    | `they them their theirs` | anything but person and temporal (groups, organisations) |
//!    | `we us our ours` | not linked (they usually include the speaker) |
//!
//!    Sentences end at `.`, `!` or `?` (plus closing quotes or brackets) followed by
//!    whitespace, except after common abbreviations (`Mr.`, `Inc.`, `No.`, ...) and single
//!    capital initials; a blank line also ends a sentence.
//!
//! Output clusters hold at least two distinct spans sorted by start, ordered by their first span.

use std::collections::{BTreeSet, HashMap};

use crate::error::ModelError;
use crate::traits::{Coref, CorefCluster, CorefMention};

/// Pronouns flagged as such in coreference output (the Python `CorefService` list).
pub const PRONOUNS: [&str; 16] = [
    "he", "she", "they", "them", "him", "her", "his", "hers", "their", "theirs", "it", "its", "we",
    "us", "our", "ours",
];

/// How many sentences before its own a pronoun's antecedent may start in (0 = same sentence).
pub const PRONOUN_SENTENCE_WINDOW: usize = 1;

const ARTICLES: [&str; 3] = ["the", "a", "an"];
const HONORIFICS: [&str; 10] =
    ["mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam", "hon"];
const DESIGNATORS: [&str; 14] = [
    "inc",
    "llc",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "lp",
    "llp",
    "pllc",
    "pc",
    "pa",
];
const ACRONYM_SKIP: [&str; 4] = ["of", "and", "for", "the"];
const ABBREVIATIONS: [&str; 20] = [
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "co", "corp", "ltd", "no", "vs", "v",
    "etc", "approx", "ave", "dept", "hon",
];

/// Rule-based coreference over detected mentions (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleCoref;

impl RuleCoref {
    /// A rule resolver.
    pub fn new() -> Self {
        Self
    }
}

impl Coref for RuleCoref {
    fn resolve(&self, _text: &str) -> Result<Vec<CorefCluster>, ModelError> {
        Ok(Vec::new())
    }

    fn resolve_with_mentions(
        &self,
        text: &str,
        mentions: &[CorefMention<'_>],
    ) -> Result<Vec<CorefCluster>, ModelError> {
        Ok(cluster_mentions(text, mentions))
    }
}

/// Pronoun kinds that link to an antecedent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PronounKind {
    Person,
    Neuter,
    Plural,
}

impl PronounKind {
    /// The kind of a lowercase pronoun; `None` for other words and unlinked pronouns.
    fn of(word: &str) -> Option<Self> {
        match word {
            "he" | "him" | "his" | "she" | "her" | "hers" => Some(Self::Person),
            "it" | "its" => Some(Self::Neuter),
            "they" | "them" | "their" | "theirs" => Some(Self::Plural),
            _ => None,
        }
    }

    fn accepts(self, entity_type: &str) -> bool {
        let is = |names: &[&str]| names.iter().any(|n| entity_type.eq_ignore_ascii_case(n));
        let person = is(&["PER", "PERSON"]);
        let temporal = is(&["DATE", "TIME"]);
        match self {
            Self::Person => person,
            Self::Neuter | Self::Plural => !person && !temporal,
        }
    }
}

/// Whether `word` (any case, surrounding whitespace ignored) is one of [`PRONOUNS`].
pub fn is_pronoun(word: &str) -> bool {
    let word = word.trim().to_lowercase();
    PRONOUNS.contains(&word.as_str())
}

/// Disjoint-set forest over span nodes.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self { parent: (0..len).collect() }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // The smaller root wins so results do not depend on union order.
            let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[drop] = keep;
        }
    }
}

/// Mentions sharing one entity type and normalised name.
struct NameGroup<'a> {
    entity_type: &'a str,
    tokens: Vec<String>,
    set: BTreeSet<String>,
    acronym: Option<String>,
    nodes: Vec<usize>,
}

impl NameGroup<'_> {
    /// Whether this group's name is an alias of `other`'s.
    fn is_alias_of(&self, other: &NameGroup<'_>) -> bool {
        if self.entity_type != other.entity_type {
            return false;
        }
        let subset = self.set.len() < other.set.len()
            && self.set.iter().any(|t| t.chars().count() >= 2)
            && self.set.is_subset(&other.set);
        subset || self.acronym.as_ref().is_some_and(|a| other.matches_acronym(a))
    }

    fn matches_acronym(&self, acronym: &str) -> bool {
        if self.tokens.len() < 2 {
            return false;
        }
        let initials = |skip: bool| -> String {
            self.tokens
                .iter()
                .filter(|t| !skip || !ACRONYM_SKIP.contains(&t.as_str()))
                .filter_map(|t| t.chars().next())
                .collect()
        };
        initials(false) == acronym || initials(true) == acronym
    }
}

/// Normalised name tokens (see the module docs).
fn name_tokens(name: &str) -> Vec<String> {
    let lowered: String = name
        .to_lowercase()
        .chars()
        .filter(|c| !matches!(c, '.' | '\'' | '\u{2019}'))
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let mut tokens: Vec<&str> = lowered.split_whitespace().collect();
    let leading =
        tokens.iter().take_while(|t| ARTICLES.contains(t) || HONORIFICS.contains(t)).count();
    tokens.drain(..leading);
    while tokens.len() > 1 && tokens.last().is_some_and(|t| DESIGNATORS.contains(t)) {
        tokens.pop();
    }
    tokens.into_iter().map(str::to_owned).collect()
}

/// The lowercase acronym spelled by `name` (`GVL`, `G.V.L.`), if it looks like one.
fn acronym(name: &str) -> Option<String> {
    let letters: String = name.trim().chars().filter(|c| *c != '.').collect();
    let len = letters.chars().count();
    ((2..=8).contains(&len) && letters.chars().all(|c| c.is_alphabetic() && c.is_uppercase()))
        .then(|| letters.to_lowercase())
}

/// Start offsets of the sentences of `chars` (the first sentence starts at 0).
fn sentence_starts(chars: &[char]) -> Vec<usize> {
    let n = chars.len();
    let mut starts = vec![0];
    let mut push = |pos: usize| {
        if pos < n && starts.last().is_some_and(|last| *last < pos) {
            starts.push(pos);
        }
    };
    let next_non_space = |mut j: usize| {
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        j
    };
    for i in 0..n {
        match chars[i] {
            '.' | '!' | '?' => {
                let mut j = i + 1;
                while j < n && matches!(chars[j], '"' | '\'' | '\u{201d}' | '\u{2019}' | ')' | ']')
                {
                    j += 1;
                }
                if j < n && !chars[j].is_whitespace() {
                    continue;
                }
                if chars[i] == '.' && is_abbreviation(chars, i) {
                    continue;
                }
                push(next_non_space(j));
            }
            '\n' => {
                let mut j = i + 1;
                while j < n && matches!(chars[j], ' ' | '\t' | '\r') {
                    j += 1;
                }
                if j < n && chars[j] == '\n' {
                    push(next_non_space(j));
                }
            }
            _ => {}
        }
    }
    starts
}

/// Whether the word ending right before the `.` at `dot` is an abbreviation or an initial.
fn is_abbreviation(chars: &[char], dot: usize) -> bool {
    let start = chars[..dot].iter().rposition(|c| !c.is_alphabetic()).map_or(0, |p| p + 1);
    let word = &chars[start..dot];
    match word {
        [] => false,
        [c] if c.is_uppercase() => true,
        _ => {
            let lowered: String = word.iter().flat_map(|c| c.to_lowercase()).collect();
            ABBREVIATIONS.contains(&lowered.as_str())
        }
    }
}

/// Clusters `mentions` of `text` (see the module docs).
fn cluster_mentions(text: &str, mentions: &[CorefMention<'_>]) -> Vec<CorefCluster> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut valid: Vec<&CorefMention<'_>> =
        mentions.iter().filter(|m| m.start < m.end && m.end <= n).collect();
    valid.sort_by_key(|m| (m.start, m.end));

    // Span nodes: named mentions first, then pronouns.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    // (node, entity type) of each named mention, in text order.
    let mut named: Vec<(usize, &str)> = Vec::new();
    let mut groups: Vec<NameGroup<'_>> = Vec::new();
    let mut group_index: HashMap<(&str, String), usize> = HashMap::new();
    // (node, kind) of each pronoun occurrence.
    let mut pronouns: Vec<(usize, PronounKind)> = Vec::new();

    for m in &valid {
        let lowered = m.text.trim().to_lowercase();
        if PRONOUNS.contains(&lowered.as_str()) {
            if let Some(kind) = PronounKind::of(&lowered) {
                spans.push((m.start, m.end));
                pronouns.push((spans.len() - 1, kind));
            }
            continue;
        }
        let tokens = name_tokens(m.text);
        if tokens.is_empty() {
            continue;
        }
        spans.push((m.start, m.end));
        let node = spans.len() - 1;
        named.push((node, m.entity_type));
        let key = (m.entity_type, tokens.join(" "));
        let index = match group_index.get(&key) {
            Some(&index) => index,
            None => {
                groups.push(NameGroup {
                    entity_type: m.entity_type,
                    set: tokens.iter().cloned().collect(),
                    tokens,
                    acronym: None,
                    nodes: Vec::new(),
                });
                group_index.insert(key, groups.len() - 1);
                groups.len() - 1
            }
        };
        let group = &mut groups[index];
        group.nodes.push(node);
        if group.acronym.is_none() {
            group.acronym = acronym(m.text);
        }
    }

    // Pronoun words outside mentions.
    let mut word_start = None;
    for i in 0..=n {
        let alphabetic = i < n && chars[i].is_alphabetic();
        match (alphabetic, word_start) {
            (true, None) => word_start = Some(i),
            (false, Some(start)) => {
                word_start = None;
                let word: String = chars[start..i].iter().flat_map(|c| c.to_lowercase()).collect();
                let Some(kind) = PronounKind::of(&word) else { continue };
                if valid.iter().any(|m| m.start < i && start < m.end) {
                    continue;
                }
                spans.push((start, i));
                pronouns.push((spans.len() - 1, kind));
            }
            _ => {}
        }
    }
    pronouns.sort_by_key(|(node, _)| spans[*node]);

    let mut forest = UnionFind::new(spans.len());
    for group in &groups {
        for pair in group.nodes.windows(2) {
            forest.union(pair[0], pair[1]);
        }
    }
    for (g, group) in groups.iter().enumerate() {
        let longer: Vec<&NameGroup<'_>> = groups
            .iter()
            .enumerate()
            .filter(|(h, other)| *h != g && group.is_alias_of(other))
            .map(|(_, other)| other)
            .collect();
        let nested = longer.iter().enumerate().all(|(i, a)| {
            longer[i + 1..].iter().all(|b| a.set.is_subset(&b.set) || b.set.is_subset(&a.set))
        });
        if !nested {
            continue;
        }
        for other in longer {
            forest.union(group.nodes[0], other.nodes[0]);
        }
    }

    let sentences = sentence_starts(&chars);
    let sentence_of = |pos: usize| sentences.partition_point(|s| *s <= pos).saturating_sub(1);
    let mut linked: Vec<(usize, PronounKind)> = Vec::new();
    for &(node, kind) in &pronouns {
        let (start, _) = spans[node];
        let earliest = sentence_of(start).saturating_sub(PRONOUN_SENTENCE_WINDOW);
        let in_window = |candidate: usize| {
            let (s, e) = spans[candidate];
            e <= start && sentence_of(s) >= earliest
        };
        let names = named
            .iter()
            .filter(|(c, entity_type)| in_window(*c) && kind.accepts(entity_type))
            .map(|(c, _)| *c);
        let earlier_pronouns =
            linked.iter().filter(|(c, k)| *k == kind && in_window(*c)).map(|(c, _)| *c);
        // Nearest: latest end, then the longer span.
        let antecedent = names
            .chain(earlier_pronouns)
            .max_by_key(|c| (spans[*c].1, std::cmp::Reverse(spans[*c].0)));
        if let Some(antecedent) = antecedent {
            forest.union(node, antecedent);
            linked.push((node, kind));
        }
    }

    let mut by_root: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (node, span) in spans.iter().enumerate() {
        by_root.entry(forest.find(node)).or_default().push(*span);
    }
    let mut clusters: Vec<CorefCluster> = by_root
        .into_values()
        .filter_map(|mut members| {
            members.sort_unstable();
            members.dedup();
            (members.len() >= 2).then_some(CorefCluster { spans: members })
        })
        .collect();
    clusters.sort_by_key(|c| c.spans[0]);
    clusters
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Mentions of every occurrence of each `(text, type)` in `text` that does not overlap an
    /// earlier listed one, as char offsets.
    fn mentions_of<'a>(text: &str, names: &[(&'a str, &'a str)]) -> Vec<CorefMention<'a>> {
        let mut out: Vec<CorefMention<'a>> = Vec::new();
        for (name, entity_type) in names {
            for (byte, _) in text.match_indices(name) {
                let start = text[..byte].chars().count();
                let end = start + name.chars().count();
                if out.iter().all(|m| m.end <= start || end <= m.start) {
                    out.push(CorefMention { start, end, text: name, entity_type });
                }
            }
        }
        out
    }

    fn texts(text: &str, clusters: &[CorefCluster]) -> Vec<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        clusters
            .iter()
            .map(|c| c.spans.iter().map(|(s, e)| chars[*s..*e].iter().collect()).collect())
            .collect()
    }

    fn resolve(text: &str, names: &[(&str, &str)]) -> Vec<Vec<String>> {
        let mentions = mentions_of(text, names);
        let clusters = RuleCoref::new().resolve_with_mentions(text, &mentions).expect("rule coref");
        texts(text, &clusters)
    }

    #[test]
    fn normalises_names() {
        assert_eq!(name_tokens("Ms. Lopez"), vec!["lopez"]);
        assert_eq!(
            name_tokens("The Green Valley Logistics, Inc."),
            vec!["green", "valley", "logistics"]
        );
        assert_eq!(name_tokens("O'Brien & Co"), vec!["obrien"]);
        assert_eq!(name_tokens("Inc"), vec!["inc"]);
        assert_eq!(acronym("G.V.L."), Some("gvl".to_owned()));
        assert_eq!(acronym("Reed"), None);
    }

    #[test]
    fn links_names_aliases_and_acronyms() {
        let text =
            "Maria Lopez sued Green Valley Logistics, Inc. Ms. Lopez said GVL was at fault. \
                    Green Valley denied it.";
        let got = resolve(
            text,
            &[
                ("Green Valley Logistics, Inc", "ORG"),
                ("Maria Lopez", "PER"),
                ("Ms. Lopez", "PER"),
                ("GVL", "ORG"),
                ("Green Valley", "ORG"),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec!["Maria Lopez", "Ms. Lopez"],
                vec!["Green Valley Logistics, Inc", "GVL", "Green Valley", "it"],
            ]
        );
    }

    #[test]
    fn ambiguous_alias_stays_unlinked() {
        let text = "Maria Lopez and Carlos Lopez arrived. Lopez left first.";
        let got =
            resolve(text, &[("Maria Lopez", "PER"), ("Carlos Lopez", "PER"), ("Lopez", "PER")]);
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn types_must_match() {
        let text = "Reed drove past Reed Street.";
        let mentions = vec![
            CorefMention { start: 0, end: 4, text: "Reed", entity_type: "PER" },
            CorefMention { start: 16, end: 27, text: "Reed Street", entity_type: "LOC" },
        ];
        let clusters = RuleCoref::new().resolve_with_mentions(text, &mentions).expect("rule");
        assert!(clusters.is_empty());
    }

    #[test]
    fn links_pronouns_within_the_sentence_window() {
        let text = "Maria Lopez was driving. She stopped at the light. Later it rained. \
                    Then her car was hit.";
        let got = resolve(text, &[("Maria Lopez", "PER")]);
        // "She" links to Maria Lopez in the previous sentence; "it" has no neuter antecedent;
        // "her" is two sentences after "She", outside the window.
        assert_eq!(got, vec![vec!["Maria Lopez", "She"]]);
    }

    #[test]
    fn pronoun_kinds_follow_entity_types() {
        let text = "Acme Corp fired Reed after he complained. Its manager said they had no choice.";
        let got = resolve(text, &[("Acme Corp", "ORG"), ("Reed", "PER")]);
        assert_eq!(got, vec![vec!["Acme Corp", "Its", "they"], vec!["Reed", "he"]]);
    }

    #[test]
    fn abbreviations_do_not_end_sentences() {
        let chars: Vec<char> =
            "Mr. Reed met J. Smith. Then they left.\n\nNew part".chars().collect();
        let starts = sentence_starts(&chars);
        assert_eq!(starts, vec![0, 23, 40]);
    }

    #[test]
    fn without_mentions_finds_nothing() {
        assert!(RuleCoref::new().resolve("She said he left.").expect("rule").is_empty());
    }

    #[test]
    fn offsets_are_characters() {
        let text = "Zoë Brandt arrived. Brandt said she was late.";
        let got = resolve(text, &[("Zoë Brandt", "PER"), ("Brandt", "PER")]);
        assert_eq!(got[0], vec!["Zoë Brandt", "Brandt", "she"]);
    }
}
