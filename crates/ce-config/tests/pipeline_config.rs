//! Pipeline config loading and validation.

use ce_config::assets::{embedded_file, DEFAULT_CONFIG_PATH};
use ce_config::pipeline::{
    ChunkingStrategy, CorefBackend, EntityRetrievalStrategy, ExtractionMode, FactRetrievalStrategy,
    GatingStrategy, HelixMode, JudgeModel, MentionDetector,
};
use ce_config::services::ContradictionCheckConfig;
use ce_config::{load_config_in, ConfigError, PipelineConfig, StageConfig};
use pretty_assertions::assert_eq;

const EXAMPLE_PATH: &str = "contradiction_engine/config/processing/example_pipeline.yaml";

fn embedded_text(path: &str) -> String {
    embedded_file(path)
        .and_then(|f| f.contents_utf8())
        .unwrap_or_else(|| panic!("{path} is embedded"))
        .to_owned()
}

fn v17_text() -> String {
    embedded_text(DEFAULT_CONFIG_PATH)
}

fn parse(text: &str) -> Result<PipelineConfig, ConfigError> {
    PipelineConfig::from_yaml_str(text, "test.yaml")
}

fn replace_once(text: &str, from: &str, to: &str) -> String {
    assert_eq!(text.matches(from).count(), 1, "`{from}` must occur exactly once");
    text.replacen(from, to, 1)
}

fn parse_error(text: &str) -> String {
    match parse(text) {
        Err(err @ ConfigError::Yaml { .. }) => err.to_string(),
        other => panic!("expected a YAML error, got {other:?}"),
    }
}

fn validation_error(text: &str) -> (String, String) {
    match parse(text) {
        Err(ConfigError::Validation { path, message }) => (path, message),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

fn v17() -> PipelineConfig {
    parse(&v17_text()).expect("processing_v17.yaml loads")
}

fn contradiction_stage(
    config: &PipelineConfig,
) -> &ce_config::pipeline::ContradictionCheckStageConfig {
    match config.get_stage_config("ContradictionCheckStage") {
        Some(StageConfig::ContradictionCheck(c)) => c,
        other => panic!("expected ContradictionCheckStage, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Shipped configs
// ---------------------------------------------------------------------------

#[test]
fn processing_v17_loads_and_validates() {
    let config = v17();
    config.validate().expect("valid");
    assert_eq!(config.config_version, "2.0");
    assert_eq!(config.name(), "processing_v17");
    assert_eq!(config.helix_namespace(), "processing_v17");
    let types: Vec<&str> = config.stages.iter().map(StageConfig::type_name).collect();
    assert_eq!(
        types,
        [
            "RAGIndexStage",
            "MentionStage",
            "DefinitionStage",
            "CoreferenceStage",
            "EDStage",
            "RelationStage",
            "FactStage",
            "ContradictionCheckStage",
        ]
    );
    assert!(config.stages.iter().all(StageConfig::enabled));
    assert_eq!(config.stages[4].name(), "ed_resolution");

    let Some(StageConfig::RagIndex(rag)) = config.get_stage_config("RAGIndexStage") else {
        panic!("RAGIndexStage present");
    };
    assert!(!rag.offline);
    assert!(!rag.use_document_cache, "use_document_cache defaults to false");

    let StageConfig::Mention(mention) = &config.stages[1] else { panic!("MentionStage") };
    assert_eq!(mention.min_mention_length, 3);
    assert_eq!(mention.detectors.len(), 3);
    assert_eq!(mention.detectors[0], MentionDetector::Eyecite);
    assert_eq!(mention.detectors[1], MentionDetector::Regex);
    let MentionDetector::Gliner2(gliner) = &mention.detectors[2] else { panic!("gliner2") };
    assert_eq!(gliner.label_set, "legal_entities");
    assert_eq!(gliner.min_score, 0.4);

    let StageConfig::Definition(definition) = &config.stages[2] else { panic!("Definition") };
    assert_eq!(definition.role_config_path, None);

    let StageConfig::Coreference(coref) = &config.stages[3] else { panic!("Coreference") };
    assert_eq!(coref.eligible_types, ["PER", "ORG", "LOC"]);
    assert_eq!(coref.incompatible_type_groups.len(), 3);
    assert!(coref.synthesize_pronoun_mentions);
    assert_eq!(coref.pronoun_confidence, 0.85);

    let StageConfig::Ed(ed) = &config.stages[4] else { panic!("EDStage") };
    assert_eq!(ed.entity_retrieval_strategy, EntityRetrievalStrategy::Bm25);
    assert_eq!(ed.prompt_file, "contradiction_engine/prompts/ed/disambiguation.json");
    assert_eq!(ed.vector_fuzzy_overfetch_multiplier, 3);

    let StageConfig::Fact(fact) = &config.stages[6] else { panic!("FactStage") };
    assert_eq!(fact.extraction_mode, ExtractionMode::Processing);
    assert_eq!(fact.prompt_files.len(), 1);
    assert!(!fact.parallel);
    assert_eq!(fact.max_workers, 1);
    assert!(fact.use_resolved_text_view);

    let check = contradiction_stage(&config);
    let verification = check.verification.as_ref().expect("verification block");
    assert_eq!(verification.judge_model, JudgeModel::Nli);
    assert_eq!(verification.gating_strategy, GatingStrategy::DualPath);
    assert_eq!(check.retrieval.fact_retrieval_strategy, FactRetrievalStrategy::Vector);
    let review = check.review_store.as_ref().expect("review store block");
    assert!(!review.enabled);
    assert!(review.fail_on_sqlite_error);

    assert_eq!(config.models.coref.backend, CorefBackend::Fastcoref);
    assert_eq!(config.models.coref.model, None);
    assert_eq!(config.models.ollama.num_ctx, 16384);
    assert_eq!(config.models.gliner2.as_ref().map(|g| g.threshold), Some(0.4));
    assert_eq!(config.helix.mode, HelixMode::Embedded);
    assert_eq!(config.helix.url, None);
    assert_eq!(config.helix.embedding_dim, 384);
    assert_eq!(config.rag.chunking.strategy, ChunkingStrategy::Sentence);
    assert_eq!(
        config.mappings.entity_mappings,
        "contradiction_engine/config/mappings/entity_types.yaml"
    );
}

#[test]
fn example_pipeline_loads_and_validates() {
    let config = parse(&embedded_text(EXAMPLE_PATH)).expect("example_pipeline.yaml loads");
    assert_eq!(config.name(), "example_pipeline");
    let Some(StageConfig::Fact(fact)) = config.get_stage_config("FactStage") else {
        panic!("FactStage present");
    };
    assert_eq!(fact.prompt_files.len(), 4);
    assert!(fact.parallel);
    let Some(StageConfig::Ed(ed)) = config.get_stage_config("EDStage") else { panic!("EDStage") };
    assert_eq!(ed.entity_retrieval_strategy, EntityRetrievalStrategy::VectorFuzzy);
    let verification = contradiction_stage(&config).verification.clone().expect("verification");
    assert_eq!(verification.judge_model, JudgeModel::Qwen);
    assert_eq!(verification.gating_strategy, GatingStrategy::Permissive);
    assert_eq!(
        verification.qwen_prompt_file.as_deref(),
        Some("contradiction_engine/prompts/processing/qwen_judge.json")
    );
    assert_eq!(config.rag.chunking.strategy, ChunkingStrategy::Recursive);
}

#[test]
fn shipped_configs_load_through_asset_resolution() {
    let empty = tempfile::tempdir().expect("tempdir");
    let v17 = load_config_in(empty.path(), DEFAULT_CONFIG_PATH).expect("embedded v17");
    assert_eq!(v17, self::v17());
    load_config_in(empty.path(), EXAMPLE_PATH).expect("embedded example");
    ce_config::load_default_config().expect("default config");

    let local = tempfile::tempdir().expect("tempdir");
    let on_disk = local.path().join(DEFAULT_CONFIG_PATH);
    std::fs::create_dir_all(on_disk.parent().expect("parent")).expect("mkdir");
    let renamed = replace_once(&v17_text(), "name: \"processing_v17\"", "name: \"local\"");
    std::fs::write(&on_disk, renamed).expect("write");
    let config = load_config_in(local.path(), DEFAULT_CONFIG_PATH).expect("local config");
    assert_eq!(config.name(), "local");
}

#[test]
fn get_stage_config_returns_none_for_missing_type() {
    assert!(v17().get_stage_config("NoSuchStage").is_none());
}

// ---------------------------------------------------------------------------
// Strictness
// ---------------------------------------------------------------------------

#[test]
fn unknown_fields_fail_with_their_path() {
    let text = v17_text();
    let cases = [
        ("config_version: \"2.0\"", "config_version: \"2.0\"\nbogus: 1", "unknown field `bogus`", ""),
        ("  name: \"processing_v17\"", "  name: \"processing_v17\"\n  bogus: 1", "bogus", "pipeline"),
        ("    offline: false", "    offline: false\n    bogus: 1", "bogus", "stages[0]"),
        (
            "        label_set: \"legal_entities\"",
            "        label_set: \"legal_entities\"\n        bogus: 1",
            "bogus",
            "stages[1].detectors[2]",
        ),
        ("      - type: \"regex\"", "      - type: \"regex\"\n        bogus: 1", "bogus", "stages[1].detectors[1]"),
        ("    llm_only: false", "    llm_only: false\n    bogus: 1", "bogus", "stages[4]"),
        ("      bm25_limit: 8", "      bm25_limit: 8\n      bogus: 1", "bogus", "stages[7].retrieval"),
        (
            "      qwen_batch_size: 5",
            "      qwen_batch_size: 5\n      bogus: 1",
            "bogus",
            "stages[7].verification",
        ),
        (
            "      fail_on_sqlite_error: true",
            "      fail_on_sqlite_error: true\n      bogus: 1",
            "bogus",
            "stages[7].review_store",
        ),
        ("    num_ctx: 16384", "    num_ctx: 16384\n    bogus: 1", "bogus", "models.ollama"),
        ("  embedding_dim: 384", "  embedding_dim: 384\n  bogus: 1", "bogus", "helix"),
        (
            "    include_sentence_spans: true",
            "    include_sentence_spans: true\n    bogus: 1",
            "bogus",
            "rag.chunking",
        ),
        (
            "  relationship_types: \"contradiction_engine/config/mappings/relationship_types.yaml\"",
            "  relationship_types: \"contradiction_engine/config/mappings/relationship_types.yaml\"\n  bogus: 1",
            "bogus",
            "mappings",
        ),
    ];
    for (from, to, field, path) in cases {
        let err = parse_error(&replace_once(&text, from, to));
        assert!(err.contains(field), "{err}");
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains(&format!("{path}: ")) || path.is_empty(), "path `{path}` in: {err}");
    }
}

#[test]
fn type_key_may_come_after_other_keys() {
    let text = replace_once(
        &v17_text(),
        "  - type: RAGIndexStage\n    name: rag_index\n",
        "  - name: rag_index\n    type: RAGIndexStage\n",
    );
    let config = parse(&text).expect("loads");
    assert_eq!(config.stages[0].type_name(), "RAGIndexStage");

    let with_bogus = replace_once(&text, "    offline: false", "    offline: false\n    bogus: 1");
    let err = parse_error(&with_bogus);
    assert!(err.contains("stages[0]") && err.contains("unknown field `bogus`"), "{err}");
}

#[test]
fn unknown_stage_and_detector_types_fail() {
    let err = parse_error(&replace_once(&v17_text(), "type: DefinitionStage", "type: NopeStage"));
    assert!(err.contains("stages[2]") && err.contains("unknown variant `NopeStage`"), "{err}");
    let err = parse_error(&replace_once(&v17_text(), "type: \"eyecite\"", "type: \"spacy\""));
    assert!(err.contains("stages[1].detectors[0]") && err.contains("spacy"), "{err}");
}

#[test]
fn missing_and_mistyped_fields_fail() {
    let err = parse_error(&replace_once(&v17_text(), "    llm_only: false\n", ""));
    assert!(err.contains("stages[4]") && err.contains("missing field `llm_only`"), "{err}");
    let err = parse_error(&replace_once(&v17_text(), "  batch_size: 50", "  batch_size: many"));
    assert!(err.contains("helix.batch_size"), "{err}");
    let err = parse_error(&replace_once(&v17_text(), "chunk_overlap: 150", "chunk_overlap: -1"));
    assert!(err.contains("rag.chunking.chunk_overlap"), "{err}");
    let err =
        parse_error(&replace_once(&v17_text(), "judge_model: \"nli\"", "judge_model: \"gpt\""));
    assert!(err.contains("stages[7].verification.judge_model"), "{err}");
    let err = parse_error(&replace_once(&v17_text(), "mode: \"embedded\"", "mode: \"cloud\""));
    assert!(err.contains("helix.mode"), "{err}");
}

#[test]
fn wrong_config_version_fails() {
    let (path, message) = validation_error(&replace_once(
        &v17_text(),
        "config_version: \"2.0\"",
        "config_version: \"1.0\"",
    ));
    assert_eq!(path, "config_version");
    assert_eq!(message, "Unsupported config_version '1.0'. Supported: ['2.0'].");
}

#[test]
fn empty_file_fails() {
    assert!(matches!(parse("# only a comment\n"), Err(ConfigError::Empty { .. })));
    assert!(matches!(parse(""), Err(ConfigError::Empty { .. })));
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

#[test]
fn validators_report_field_paths() {
    let text = v17_text();
    let cases: &[(&str, &str, &str, &str)] = &[
        (
            "min_embedding_similarity: 0.55",
            "min_embedding_similarity: 1.5",
            "stages[4].min_embedding_similarity",
            "between 0 and 1",
        ),
        (
            "hybrid_similarity_threshold: 0.72",
            "hybrid_similarity_threshold: -0.1",
            "stages[4].hybrid_similarity_threshold",
            "Similarity threshold",
        ),
        (
            "reuse_similarity_threshold: 0.72",
            "reuse_similarity_threshold: 2",
            "stages[4].reuse_similarity_threshold",
            "Similarity threshold",
        ),
        (
            "cross_doc_reuse_boost: 0.05",
            "cross_doc_reuse_boost: 0.3",
            "stages[4].cross_doc_reuse_boost",
            "between 0 and 0.2",
        ),
        (
            "vector_fuzzy_overfetch_multiplier: 3",
            "vector_fuzzy_overfetch_multiplier: 0",
            "stages[4].vector_fuzzy_overfetch_multiplier",
            ">= 1",
        ),
        (
            "vector_fuzzy_overfetch_multiplier: 3",
            "vector_fuzzy_overfetch_multiplier: 11",
            "stages[4].vector_fuzzy_overfetch_multiplier",
            "<= 10",
        ),
        (
            "vector_fuzzy_min_similarity: 0.5",
            "vector_fuzzy_min_similarity: 1.1",
            "stages[4].vector_fuzzy_min_similarity",
            "vector_fuzzy_*",
        ),
        (
            "vector_fuzzy_vector_weight: 0.4",
            "vector_fuzzy_vector_weight: 0.5",
            "stages[4]",
            "must sum to 1.0",
        ),
        (
            "      min_score: 0.0",
            "      min_score: 1.5",
            "stages[7].retrieval.min_score",
            "min_score must be between 0 and 1",
        ),
        (
            "      min_text_overlap: 0.15\n      require_entity_overlap",
            "      min_text_overlap: 1.15\n      require_entity_overlap",
            "stages[7].pre_filter.min_text_overlap",
            "threshold",
        ),
        (
            "fuzzy_match_threshold: 0.6",
            "fuzzy_match_threshold: -0.6",
            "stages[7].pre_filter.fuzzy_match_threshold",
            "threshold",
        ),
        (
            "contradiction_threshold: 0.80",
            "contradiction_threshold: 1.0",
            "stages[7].nli.contradiction_threshold",
            "threshold",
        ),
        (
            "review_threshold: 0.60",
            "review_threshold: 0.0",
            "stages[7].nli.review_threshold",
            "threshold",
        ),
        (
            "semantic_proximity_threshold: 0.60",
            "semantic_proximity_threshold: 1.6",
            "stages[7].verification.semantic_proximity_threshold",
            "threshold",
        ),
        (
            "qwen_batch_size: 5",
            "qwen_batch_size: 51",
            "stages[7].verification.qwen_batch_size",
            "between 1 and 50",
        ),
        (
            "namespace: \"processing_v17\"",
            "namespace: \"   \"",
            "helix.namespace",
            "must not be empty",
        ),
        ("embedding_dim: 384", "embedding_dim: 0", "helix.embedding_dim", "must be positive"),
        ("url: null", "url: \"ftp://host\"", "helix.url", "valid HTTP URL"),
        ("mode: \"embedded\"", "mode: \"server\"", "helix", "helix.url is required"),
        (
            "data_dir: \"data/helix\"",
            "data_dir: \"  \"",
            "helix",
            "helix.data_dir must not be empty",
        ),
        ("chunk_size: 1500", "chunk_size: 0", "rag.chunking.chunk_size", "chunk_size must be > 0"),
    ];
    for (from, to, path, message) in cases {
        let (got_path, got_message) = validation_error(&replace_once(&text, from, to));
        assert_eq!(got_path, *path, "{to}");
        assert!(got_message.contains(*message), "{to}: {got_message}");
    }
}

#[test]
fn empty_lists_fail_validation() {
    let text = v17_text();
    let start = text.find("stages:\n").expect("stages");
    let end = text.find("# Model configurations").expect("models");
    let no_stages = format!("{}stages: []\n\n{}", &text[..start], &text[end..]);
    assert_eq!(validation_error(&no_stages).0, "stages");

    let detectors_start = text.find("    detectors:\n").expect("detectors");
    let detectors_end = text.find("  - type: DefinitionStage").expect("definition");
    let no_detectors =
        format!("{}    detectors: []\n\n{}", &text[..detectors_start], &text[detectors_end..]);
    assert_eq!(validation_error(&no_detectors).0, "stages[1].detectors");

    let no_prompts = replace_once(
        &text,
        "    prompt_files:\n      - \"contradiction_engine/prompts/processing/singlepass/processing_single_v3.json\"",
        "    prompt_files: []",
    );
    assert_eq!(
        validation_error(&no_prompts),
        ("stages[6].prompt_files".to_owned(), "prompt_files must not be empty".to_owned())
    );
}

#[test]
fn helix_server_mode_accepts_a_valid_url() {
    let text = replace_once(&v17_text(), "mode: \"embedded\"", "mode: \"server\"");
    let text = replace_once(&text, "url: null", "url: \"http://localhost:6969\"");
    let config = parse(&text).expect("server mode with url");
    assert_eq!(config.helix.mode, HelixMode::Server);
    let memory = replace_once(&v17_text(), "mode: \"embedded\"", "mode: \"memory\"");
    let memory = replace_once(&memory, "data_dir: \"data/helix\"", "data_dir: \"\"");
    parse(&memory).expect("memory mode ignores data_dir");
}

#[test]
fn optional_blocks_use_python_defaults() {
    let text = v17_text();
    let start = text.find("    verification:\n").expect("verification");
    let end = text.find("    # Existing configuration").expect("retrieval comment");
    let without_verification = format!("{}{}", &text[..start], &text[end..]);
    let without_verification = replace_once(
        &without_verification,
        "      min_score: 0.0\n      fact_retrieval_strategy: \"vector\"\n",
        "",
    );
    let config = parse(&without_verification).expect("loads");
    let check = contradiction_stage(&config);
    assert!(check.verification.is_none());
    assert_eq!(check.retrieval.min_score, 0.0);
    assert_eq!(check.retrieval.fact_retrieval_strategy, FactRetrievalStrategy::Both);

    let minimal_verification = format!(
        "{}    verification:\n      judge_model: \"qwen\"\n      gating_strategy: \"permissive\"\n{}",
        &text[..start],
        &text[end..]
    );
    let config = parse(&minimal_verification).expect("loads");
    let v = contradiction_stage(&config).verification.clone().expect("verification");
    assert!(v.allow_text_value_type_mismatch);
    assert!(!v.require_entity_id_match);
    assert_eq!(v.semantic_proximity_threshold, 0.60);
    assert_eq!(v.keyword_overlap_threshold, 0.30);
    assert_eq!(v.qwen_prompt_file, None);
    assert_eq!(v.qwen_batch_size, 5);
}

#[test]
fn rag_index_use_document_cache_can_be_enabled() {
    let text = replace_once(
        &v17_text(),
        "    offline: false",
        "    offline: false\n    use_document_cache: true",
    );
    let Some(StageConfig::RagIndex(rag)) =
        parse(&text).expect("loads").get_stage_config("RAGIndexStage").cloned()
    else {
        panic!("RAGIndexStage present");
    };
    assert!(rag.use_document_cache);
    let err = parse_error(&replace_once(
        &v17_text(),
        "    offline: false",
        "    offline: false\n    document_cache:\n      enabled: true",
    ));
    assert!(err.contains("unknown field `document_cache`"), "{err}");
}

// ---------------------------------------------------------------------------
// Service configs
// ---------------------------------------------------------------------------

#[test]
fn contradiction_service_config_is_built_from_the_stage() {
    let config = v17();
    let service = ContradictionCheckConfig::from(contradiction_stage(&config));
    assert_eq!(service.retrieval.max_candidates, 25);
    assert_eq!(service.retrieval.fact_retrieval_strategy, FactRetrievalStrategy::Vector);
    assert_eq!(service.pre_filter.min_text_overlap, 0.15);
    assert!(service.pre_filter.enable_text_overlap_fallback);
    assert_eq!(service.pre_filter.subject_overlap_threshold, 0.4);
    assert_eq!(service.intra_doc_filter.min_text_overlap, 0.15);
    assert_eq!(service.intra_doc_filter.temporal_relationships.len(), 20);
    assert_eq!(service.contradiction_threshold, 0.80);
    assert_eq!(service.review_threshold, 0.60);
    assert!(service.intra_document_enabled);
    assert!(!service.commit_as_we_go);
    assert_eq!(service.max_intra_doc_pairs, 2000);
}

#[test]
fn service_config_defaults_match_python() {
    let d = ContradictionCheckConfig::default();
    assert_eq!(d.contradiction_threshold, 0.85);
    assert_eq!(d.review_threshold, 0.6);
    assert_eq!(d.retrieval.bm25_limit, 8);
    assert_eq!(d.retrieval.multi_source_boost, 1.2);
    assert_eq!(d.retrieval.fact_retrieval_strategy, FactRetrievalStrategy::Both);
    assert_eq!(d.pre_filter.min_text_overlap, 0.05);
    assert_eq!(d.pre_filter.semantic_similarity_threshold, 0.45);
    assert_eq!(d.pre_filter.predicate_similarity_threshold, 0.7);
    assert!(d.pre_filter.use_assertion_boost);
    assert!(d.intra_doc_filter.require_assertion_overlap);
    assert_eq!(d.intra_doc_filter.quantitative_relationships.len(), 11);
    assert_eq!(d.intra_doc_filter.role_relationships.len(), 13);
}

#[test]
fn mappings_block_is_required_but_its_paths_are_not_read() {
    let text = v17_text()
        .replace(
            "contradiction_engine/config/mappings/entity_types.yaml",
            "does/not/exist/entity_types.yaml",
        )
        .replace(
            "contradiction_engine/config/mappings/relationship_types.yaml",
            "does/not/exist/relationship_types.yaml",
        );
    let config = parse(&text).expect("mapping paths are not resolved at load time");
    config.validate().expect("mapping paths are not validated");
    assert_eq!(config.mappings.entity_mappings, "does/not/exist/entity_types.yaml");

    let without_block = replace_once(
        &v17_text(),
        "  relationship_types: \"contradiction_engine/config/mappings/relationship_types.yaml\"",
        "",
    );
    assert!(parse_error(&without_block).contains("relationship_types"));
}
