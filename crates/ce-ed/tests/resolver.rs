//! CoreEdResolver over one indexed chunk and a MemoryStore: intra-document matching, entity
//! creation with cluster aliases, cross-type separation and second-document reuse.

mod ed_support;

use std::sync::Arc;

use ce_config::pipeline::EntityRetrievalStrategy;
use ce_config::prompts::EdPrompt;
use ce_core::{
    Chunk, ChunkWriteResult, CorefClusterContext, EdResult, Mention, MentionId, Metadata, NodeId,
    Span,
};
use ce_ed::{CoreEdResolver, EntityOutcome, ResolveInput};
use ce_models::{Embedder, MockEmbedder};
use ce_store::GraphStore;
use ed_support::{ed_config, entities, mappings, roles, store, DIM};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;
use serde_json::json;

const TEXT: &str = "Maria Lopez sued Mercury Freight. Ms. Lopez said Mercury lied. \
                    Plaintiff was injured at Riverside Arena.";

fn chunk(doc_id: &str, helix_id: &str) -> ChunkWriteResult {
    let chunk = Chunk::new(doc_id, TEXT, 0, 0, TEXT.chars().count());
    ChunkWriteResult { chunk, helix_id: NodeId::new(helix_id), created: true }
}

/// A mention of the first occurrence of `name` at or after char `from`.
fn mention(id: &str, label: &str, name: &str, from: usize, chunk_id: &str) -> Mention {
    let skip: usize = TEXT.chars().take(from).map(char::len_utf8).sum();
    let byte = skip + TEXT[skip..].find(name).expect("name occurs");
    let start = TEXT[..byte].chars().count();
    let mut m = Mention::new(id, label, name, Span::new(start, start + name.chars().count()));
    m.chunk_id = Some(chunk_id.into());
    m
}

fn resolver(create_if_missing: bool) -> CoreEdResolver {
    let mut config = ed_config();
    config.create_if_missing = create_if_missing;
    let prompt = EdPrompt::load(&config.prompt_file).expect("prompt");
    CoreEdResolver::new(&config, mappings(), Some(prompt), Some(roles()))
}

struct Run<'a> {
    store: &'a Arc<dyn GraphStore>,
    embedder: Option<&'a Arc<dyn Embedder>>,
    chunks: &'a [ChunkWriteResult],
    metadata: &'a Metadata,
    clusters: Option<&'a [CorefClusterContext]>,
    canonical: Option<&'a IndexMap<String, MentionId>>,
    all: Option<&'a IndexMap<MentionId, Mention>>,
}

impl Run<'_> {
    fn input(&self) -> ResolveInput<'_> {
        ResolveInput {
            store: self.store,
            embedder: self.embedder,
            llm: None,
            chunks: self.chunks,
            doc_metadata: self.metadata,
            coref_clusters: self.clusters,
            canonical_for_cluster: self.canonical,
            all_mentions_by_id: self.all,
        }
    }
}

fn strategy(results: &IndexMap<MentionId, EdResult>, id: &str) -> String {
    results[&MentionId::from(id)].strategy_used.clone()
}

fn chosen(results: &IndexMap<MentionId, EdResult>, id: &str) -> Option<String> {
    results[&MentionId::from(id)].chosen_entity_id.as_ref().map(ToString::to_string)
}

#[test]
fn resolves_a_document_with_intra_doc_matching_and_creation() {
    let (memory, store) = store();
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(DIM));
    let chunks = [chunk("doc-1", "100")];
    let metadata: Metadata =
        json!({"case_id": "CASE-1", "doc_type": "complaint"}).as_object().cloned().expect("map");
    let mentions = vec![
        mention("m1", "PER", "Maria Lopez", 0, "100"),
        mention("m2", "ORG", "Mercury Freight", 0, "100"),
        mention("m3", "PER", "Ms. Lopez", 0, "100"),
        mention("m4", "PER", "Mercury", 40, "100"),
        mention("m5", "LOC", "Riverside Arena", 0, "100"),
        mention("m6", "PER", "Maria Lopez", 0, "999"),
        mention("m7", "PER", "Lopez", 34, "100"),
    ];
    let run = Run {
        store: &store,
        embedder: Some(&embedder),
        chunks: &chunks,
        metadata: &metadata,
        clusters: None,
        canonical: None,
        all: None,
    };
    let results = resolver(true).resolve(&mentions, &run.input()).expect("resolves").results;

    // m6 is outside the indexed chunks: no result.
    let ids: Vec<&str> = results.keys().map(MentionId::as_str).collect();
    assert_eq!(ids, ["m1", "m2", "m3", "m4", "m5", "m7"]);
    assert_eq!(strategy(&results, "m1"), "hybrid_unresolved_create");
    assert_eq!(strategy(&results, "m2"), "hybrid_unresolved_create");
    // "Ms. Lopez" is not a loose name match of "Maria Lopez": the service's person
    // similarity resolves it; "Lopez" matches within the document.
    assert_eq!(strategy(&results, "m3"), "hybrid_similarity");
    assert_eq!(chosen(&results, "m3"), chosen(&results, "m1"));
    assert_eq!(strategy(&results, "m7"), "intra_doc_fuzzy");
    assert_eq!(chosen(&results, "m7"), chosen(&results, "m1"));
    // PER "Mercury" never matches ORG "Mercury Freight".
    assert_ne!(chosen(&results, "m4"), chosen(&results, "m2"));
    assert!(strategy(&results, "m4").ends_with("_create"), "{}", strategy(&results, "m4"));
    assert_eq!(strategy(&results, "m5"), "hybrid_unresolved_create");
    for (id, result) in &results {
        assert_eq!(result.mention_id.as_ref(), Some(id));
    }
    assert_eq!(entities(&memory).len(), 4);
    let created = memory.embedding(chosen(&results, "m1").expect("chosen").as_str());
    assert!(created.is_some());

    // A second document with the same names reuses every entity through the service.
    let chunks2 = [chunk("doc-2", "200")];
    let mentions2 = vec![
        mention("n1", "PER", "Maria Lopez", 0, "200"),
        mention("n2", "ORG", "Mercury Freight", 0, "200"),
        mention("n5", "LOC", "Riverside Arena", 0, "200"),
    ];
    let run2 = Run { chunks: &chunks2, ..run };
    let results2 = resolver(true).resolve(&mentions2, &run2.input()).expect("resolves").results;
    assert_eq!(chosen(&results2, "n1"), chosen(&results, "m1"));
    assert_eq!(chosen(&results2, "n2"), chosen(&results, "m2"));
    assert_eq!(chosen(&results2, "n5"), chosen(&results, "m5"));
    assert_eq!(strategy(&results2, "n1"), "hybrid_heuristic");
    assert_eq!(entities(&memory).len(), 4);
}

#[test]
fn canonical_mentions_get_cluster_aliases() {
    let (memory, store) = store();
    let chunks = [chunk("doc-1", "100")];
    let metadata = Metadata::new();
    let canonical = mention("m1", "PER", "Maria Lopez", 0, "100");
    let member = mention("m3", "PER", "Ms. Lopez", 0, "100");
    let role = mention("m7", "PER", "Plaintiff", 0, "100");
    let all: IndexMap<MentionId, Mention> =
        [canonical.clone(), member, role].into_iter().map(|m| (m.id.clone(), m)).collect();
    let clusters = vec![CorefClusterContext {
        id: "doc-1_cluster_0".into(),
        member_mention_ids: vec!["m1".into(), "m3".into(), "m7".into()],
        pronoun_spans: vec![],
    }];
    let canonical_map: IndexMap<String, MentionId> =
        [("doc-1_cluster_0".to_owned(), MentionId::from("m1"))].into_iter().collect();
    let run = Run {
        store: &store,
        embedder: None,
        chunks: &chunks,
        metadata: &metadata,
        clusters: Some(&clusters),
        canonical: Some(&canonical_map),
        all: Some(&all),
    };
    let results = resolver(true).resolve(&[canonical], &run.input()).expect("resolves").results;
    assert_eq!(strategy(&results, "m1"), "hybrid_unresolved_create");
    let stored = entities(&memory);
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].node.name, "Maria Lopez");
    assert_eq!(stored[0].node.aliases(), vec!["Ms. Lopez", "Plaintiff"]);
    let metadata: serde_json::Value =
        serde_json::from_str(&stored[0].node.metadata).expect("metadata");
    assert_eq!(metadata["roles"], json!(["PLAINTIFF"]));
}

#[test]
fn without_create_if_missing_mentions_stay_unresolved() {
    let (memory, store) = store();
    let chunks = [chunk("doc-1", "100")];
    let metadata = Metadata::new();
    let run = Run {
        store: &store,
        embedder: None,
        chunks: &chunks,
        metadata: &metadata,
        clusters: None,
        canonical: None,
        all: None,
    };
    let mentions = [mention("m1", "PER", "Maria Lopez", 0, "100")];
    let results = resolver(false).resolve(&mentions, &run.input()).expect("resolves").results;
    assert_eq!(chosen(&results, "m1"), None);
    assert_eq!(strategy(&results, "m1"), "hybrid_unresolved");
    assert!(entities(&memory).is_empty());
}

#[test]
fn intra_doc_matching_can_be_disabled() {
    let (_, store) = store();
    let chunks = [chunk("doc-1", "100")];
    let metadata = Metadata::new();
    let mut config = ed_config();
    config.enable_intra_doc_matching = false;
    let resolver = CoreEdResolver::new(&config, mappings(), None, Some(roles()));
    let run = Run {
        store: &store,
        embedder: None,
        chunks: &chunks,
        metadata: &metadata,
        clusters: None,
        canonical: None,
        all: None,
    };
    let mentions = [
        mention("m1", "PER", "Maria Lopez", 0, "100"),
        mention("m3", "PER", "Ms. Lopez", 0, "100"),
    ];
    let results = resolver.resolve(&mentions, &run.input()).expect("resolves").results;
    assert!(!strategy(&results, "m3").starts_with("intra_doc"), "{}", strategy(&results, "m3"));
    assert_eq!(chosen(&results, "m3"), chosen(&results, "m1"));
}

#[test]
fn all_entities_fuzzy_sees_entities_created_earlier_in_the_document() {
    // With intra-document matching off, the second "Mercury Freight" must find the entity the
    // first one created through the refreshed `all_entities_fuzzy` caches.
    let (memory, store) = store();
    let chunks = [chunk("doc-1", "100")];
    let metadata = Metadata::new();
    let mut config = ed_config();
    config.enable_intra_doc_matching = false;
    config.entity_retrieval_strategy = EntityRetrievalStrategy::AllEntitiesFuzzy;
    // BM25 similarity retrieval would find the entity outside the cache.
    config.enable_similarity_retrieval = false;
    let resolver = CoreEdResolver::new(&config, mappings(), None, Some(roles()));
    let run = Run {
        store: &store,
        embedder: None,
        chunks: &chunks,
        metadata: &metadata,
        clusters: None,
        canonical: None,
        all: None,
    };
    let mentions = [
        mention("m1", "ORG", "Mercury Freight", 0, "100"),
        mention("m2", "ORG", "Mercury Freight", 0, "100"),
    ];
    let resolution = resolver.resolve(&mentions, &run.input()).expect("resolves");
    assert_eq!(entities(&memory).len(), 1);
    assert!(matches!(
        resolution.outcomes.get(&MentionId::from("m1")),
        Some(EntityOutcome::Created(_))
    ));
    // The retriever's refreshed cache resolves it before the entity creator is needed.
    assert_eq!(resolution.outcomes.get(&MentionId::from("m2")), None);
    assert_eq!(strategy(&resolution.results, "m2"), "hybrid_heuristic");
    assert_eq!(chosen(&resolution.results, "m2"), chosen(&resolution.results, "m1"));
}
