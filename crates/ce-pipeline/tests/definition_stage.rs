//! DefinitionStage on a context whose `doc_mentions`, `doc_text` and `doc_id` are set directly.

use std::sync::Arc;

use ce_config::pipeline::DefinitionStageConfig;
use ce_config::{PipelineConfig, StageConfig};
use ce_core::processing::DefinedTerm;
use ce_core::{DocId, Mention, MentionId, Span};
use ce_models::ModelRegistry;
use ce_pipeline::{
    DefinitionStage, PipelineContext, PipelineError, Stage, StageBuildContext, StageRegistry,
};
use ce_store::{GraphStore, MemoryStore};
use pretty_assertions::assert_eq;
use serde_json::json;

const TEXT: &str = r#"Maria Lopez ("Plaintiff") brings this action against Reed Logistics LLC (the "Company"). Plaintiff worked for the Company."#;

fn services() -> StageBuildContext {
    let config: PipelineConfig = ce_config::load_default_config().expect("default config loads");
    let registry = ModelRegistry::from_config(&config);
    let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(384));
    StageBuildContext::new(Arc::new(config), Arc::new(registry), store)
}

fn stage_config(services: &StageBuildContext) -> StageConfig {
    services
        .config
        .stages
        .iter()
        .find(|stage| stage.type_name() == DefinitionStage::TYPE_NAME)
        .cloned()
        .expect("default config has a DefinitionStage")
}

fn definition_config(enabled: bool, role_config_path: Option<String>) -> DefinitionStageConfig {
    DefinitionStageConfig { name: "definition_extraction".into(), enabled, role_config_path }
}

fn mentions() -> Vec<Mention> {
    let mut judge = Mention::new("m4", "PERSON", "Hon. Ruiz", Span::new(130, 139));
    judge.metadata.insert("role".into(), json!("Judge"));
    let mut employee = Mention::new("m5", "PERSON", "Maria", Span::new(140, 145));
    employee.metadata.insert("role_in_doc".into(), json!("Employee"));
    vec![
        Mention::new("m1", "PERSON", "Maria Lopez", Span::new(0, 11)),
        Mention::new("m2", "ORG", "Reed Logistics LLC", Span::new(53, 71)),
        Mention::new("m3", "PERSON", "Plaintiff", Span::new(89, 98)),
        judge,
        employee,
    ]
}

fn context(
    services: &StageBuildContext,
    mentions: Option<Vec<Mention>>,
    text: Option<&str>,
) -> PipelineContext {
    let mut ctx = PipelineContext::new(services);
    ctx.data.doc_mentions = mentions;
    ctx.data.doc_text = text.map(str::to_owned);
    ctx.data.doc_id = Some(DocId::new("doc1"));
    ctx
}

fn run(stage: &dyn Stage, ctx: &mut PipelineContext) -> Option<Vec<DefinedTerm>> {
    stage.run(ctx).expect("stage runs");
    ctx.data.defined_terms.clone()
}

/// (mention id, role, method, span, confidence to two decimals) of each term.
type Summary<'a> = (&'a str, &'a str, &'a str, (usize, usize), String);

fn summary(terms: &[DefinedTerm]) -> Vec<Summary<'_>> {
    terms
        .iter()
        .map(|t| {
            (
                t.entity_mention_id.as_str(),
                t.defined_term.as_str(),
                t.extraction_method.as_str(),
                (t.span.start, t.span.end),
                format!("{:.2}", t.confidence),
            )
        })
        .collect()
}

#[test]
fn writes_pattern_then_metadata_definitions() {
    let services = services();
    let stage = StageRegistry::builtin()
        .build_stage(&stage_config(&services), &services)
        .expect("stage builds from the default config");
    assert_eq!(stage.name(), "definition_extraction");

    let mut ctx = context(&services, Some(mentions()), Some(TEXT));
    let terms = run(stage.as_ref(), &mut ctx).expect("defined_terms written");
    assert_eq!(
        summary(&terms),
        [
            ("m1", "PLAINTIFF", "pattern:parenthetical_quoted", (0, 25), "0.95".to_owned()),
            ("m2", "COMPANY", "pattern:parenthetical_with_article", (53, 87), "0.95".to_owned()),
            ("m5", "EMPLOYEE", "metadata:role", (140, 145), "0.90".to_owned()),
        ]
    );
    assert_eq!(terms[0].original_term, "Plaintiff");
    assert_eq!(terms[0].entity_mention_id, MentionId::new("m1"));
    // Other slots are untouched.
    assert_eq!(ctx.data.doc_mentions.as_ref().map(Vec::len), Some(5));
    assert!(ctx.data.role_map.is_none());
}

#[test]
fn skips_without_mentions_or_text() {
    let services = services();
    let stage = DefinitionStage::new(&definition_config(true, None), &services).expect("builds");
    for (mentions, text) in [
        (None, Some(TEXT)),
        (Some(Vec::new()), Some(TEXT)),
        (Some(mentions()), None),
        (Some(mentions()), Some("")),
    ] {
        let mut ctx = context(&services, mentions, text);
        assert_eq!(run(&stage, &mut ctx), None);
    }
}

#[test]
fn writes_nothing_when_no_definition_is_found() {
    let services = services();
    let stage = DefinitionStage::new(&definition_config(true, None), &services).expect("builds");
    let plain = vec![Mention::new("m1", "PERSON", "Maria Lopez", Span::new(0, 11))];
    let mut ctx = context(&services, Some(plain), Some("Maria Lopez went home."));
    assert_eq!(run(&stage, &mut ctx), None);
}

#[test]
fn missing_doc_id_still_runs() {
    let services = services();
    let stage = DefinitionStage::new(&definition_config(true, None), &services).expect("builds");
    let mut ctx = context(&services, Some(mentions()), Some(TEXT));
    ctx.data.doc_id = None;
    assert_eq!(run(&stage, &mut ctx).map(|terms| terms.len()), Some(3));
}

#[test]
fn disabled_stage_does_nothing() {
    let services = services();
    let stage = DefinitionStage::new(&definition_config(false, None), &services).expect("builds");
    let mut ctx = context(&services, Some(mentions()), Some(TEXT));
    assert_eq!(run(&stage, &mut ctx), None);
}

#[test]
fn custom_role_config_path() {
    let services = services();
    let dir = std::env::temp_dir().join(format!("ce-pipeline-roles-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("roles.yaml");
    std::fs::write(
        &path,
        "role_seed_list:\n  x: [\"Company\"]\ndefinition_patterns:\n  - name: \"article\"\n    pattern: '(?P<entity_name>[A-Z][a-zA-Z ]+?) \\(the \"(?P<defined_term>[A-Z][a-z]+)\"\\)'\n    confidence: 0.5\n",
    )
    .expect("write roles.yaml");
    let config = definition_config(true, Some(path.display().to_string()));
    let stage = DefinitionStage::new(&config, &services).expect("custom roles load");
    let mut ctx = context(&services, Some(mentions()), Some(TEXT));
    let terms = run(&stage, &mut ctx).expect("defined_terms written");
    assert_eq!(
        summary(&terms),
        [("m2", "COMPANY", "pattern:article", (53, 87), "0.50".to_owned())]
    );

    let missing = definition_config(true, Some(dir.join("missing.yaml").display().to_string()));
    let error = DefinitionStage::new(&missing, &services).expect_err("missing roles file");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn construct_rejects_other_stage_types() {
    let services = services();
    let other = services
        .config
        .stages
        .iter()
        .find(|stage| stage.type_name() != DefinitionStage::TYPE_NAME)
        .expect("another stage type")
        .clone();
    let Err(error) = DefinitionStage::construct(&other, &services) else {
        panic!("a {} config must be rejected", other.type_name());
    };
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
}
