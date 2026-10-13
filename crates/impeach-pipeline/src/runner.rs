//! The pipeline runner (Python `pipeline/runner.py`) and its inputs.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use impeach_config::StageConfig;
use impeach_core::{DocId, DocumentPayload, Metadata};
use serde_json::Value;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// What a pipeline runs on.
#[derive(Debug, Clone)]
pub enum PipelineInput {
    /// A document already in memory, used as is.
    Payload(Box<DocumentPayload>),
    /// The first record of a JSONL file, read with [`impeach_ingest::read_first_record`] (the
    /// `run_processing` rules: `text` required, `doc_id` defaults to the file stem, extra
    /// top-level keys move into `metadata`, `file_path` defaults to the JSONL path).
    Jsonl(PathBuf),
    /// A JSON file holding one document object with `doc_id` and `text` (Python
    /// `DocumentLoader` + `RAGPipeline._document_from_dict`): `doc_type` defaults to `UNKNOWN`,
    /// `title` to the `doc_id`, top-level `case_id` / `db_id` are copied into `metadata`
    /// unless already there, other unknown keys are ignored, and `file_path` is the JSON path.
    JsonFile(PathBuf),
}

impl PipelineInput {
    /// [`PipelineInput::Jsonl`] for a `.jsonl` path (case-insensitive), else
    /// [`PipelineInput::JsonFile`].
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let is_jsonl =
            path.extension().is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("jsonl"));
        if is_jsonl {
            Self::Jsonl(path)
        } else {
            Self::JsonFile(path)
        }
    }

    /// Resolves the input to a document.
    ///
    /// # Errors
    /// [`PipelineError::Ingest`] for JSONL read errors; [`PipelineError::Io`] and
    /// [`PipelineError::InvalidDocument`] for JSON files.
    pub fn into_document(self) -> Result<DocumentPayload> {
        match self {
            Self::Payload(document) => Ok(*document),
            Self::Jsonl(path) => Ok(impeach_ingest::read_first_record(&path)?),
            Self::JsonFile(path) => read_json_document(&path),
        }
    }
}

impl From<DocumentPayload> for PipelineInput {
    fn from(document: DocumentPayload) -> Self {
        Self::Payload(Box::new(document))
    }
}

fn read_json_document(path: &Path) -> Result<DocumentPayload> {
    let invalid = |message: &str| PipelineError::InvalidDocument {
        path: path.to_path_buf(),
        message: message.to_owned(),
    };
    let raw = std::fs::read(path)
        .map_err(|source| PipelineError::Io { path: path.to_path_buf(), source })?;
    let raw = raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&raw);
    let value: Value =
        serde_json::from_slice(raw).map_err(|e| invalid(&format!("invalid JSON: {e}")))?;
    let Value::Object(mut data) = value else {
        return Err(invalid("expected a JSON object"));
    };
    if !data.contains_key("doc_id") || !data.contains_key("text") {
        return Err(invalid("JSON document must contain 'doc_id' and 'text'"));
    }
    let doc_id = match data.get("doc_id") {
        Some(Value::String(id)) => id.clone(),
        Some(Value::Number(n)) => n.to_string(),
        _ => return Err(invalid("'doc_id' must be a string or a number")),
    };
    let Some(Value::String(text)) = data.remove("text") else {
        return Err(invalid("'text' must be a string"));
    };
    let mut metadata = match data.remove("metadata") {
        Some(Value::Object(map)) => map,
        None | Some(Value::Null) => Metadata::new(),
        Some(_) => return Err(invalid("'metadata' must be an object")),
    };
    for key in ["case_id", "db_id"] {
        if let Some(value) = data.get(key) {
            if !metadata.contains_key(key) {
                metadata.insert(key.to_owned(), value.clone());
            }
        }
    }
    let non_empty = |key: &str| match data.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let total_pages = match data.get("total_pages") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("'total_pages' must be a non-negative integer"))?,
        ),
    };
    Ok(DocumentPayload {
        doc_type: non_empty("doc_type").unwrap_or_else(|| "UNKNOWN".to_owned()),
        title: non_empty("title").unwrap_or_else(|| doc_id.clone()),
        doc_id: DocId::from(doc_id),
        text,
        file_path: Some(path.to_string_lossy().into_owned()),
        total_pages,
        date_created: data
            .get("date_created")
            .and_then(Value::as_str)
            .and_then(impeach_core::rag::parse_iso_datetime),
        metadata,
        helix_id: None,
    })
}

/// `stage_type` span field of a stage that is not in the pipeline config.
pub const CUSTOM_STAGE_TYPE: &str = "custom";

/// Wall time of one stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTiming {
    /// Stage name.
    pub stage: String,
    /// Time spent in [`Stage::run`].
    pub elapsed: Duration,
}

/// Timings of one pipeline run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunReport {
    /// One entry per stage, in run order.
    pub stage_timings: Vec<StageTiming>,
    /// Time of the whole run, including input loading.
    pub total: Duration,
}

/// An ordered list of stages over shared services (Python `PipelineRunner`).
///
/// The pipeline itself holds no per-run state: [`Pipeline::new_context`] creates a fresh
/// context, the caller may seed it (e.g. `doc_metadata`), and [`Pipeline::run`] threads it
/// through every stage.
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
    /// Config `type` of each stage (`custom` for a stage whose name is not in the config).
    stage_types: Vec<&'static str>,
    services: StageBuildContext,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages", &self.stage_names())
            .field("pipeline", &self.services.config.name())
            .finish_non_exhaustive()
    }
}

impl Pipeline {
    /// A pipeline running `stages` in order against `services`.
    #[must_use]
    pub fn new(stages: Vec<Box<dyn Stage>>, services: StageBuildContext) -> Self {
        let stage_types = stages
            .iter()
            .map(|stage| {
                services
                    .config
                    .stages
                    .iter()
                    .find(|config| config.name() == stage.name())
                    .map_or(CUSTOM_STAGE_TYPE, StageConfig::type_name)
            })
            .collect();
        Self { stages, stage_types, services }
    }

    /// The stages, in run order.
    #[must_use]
    pub fn stages(&self) -> &[Box<dyn Stage>] {
        &self.stages
    }

    /// Stage names, in run order.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|stage| stage.name()).collect()
    }

    /// The services the pipeline was built with.
    #[must_use]
    pub fn services(&self) -> &StageBuildContext {
        &self.services
    }

    /// An empty context over the pipeline's services.
    #[must_use]
    pub fn new_context(&self) -> PipelineContext {
        PipelineContext::new(&self.services)
    }

    /// Resolves `input` into `ctx.data.document` and runs every stage in order. Each stage runs
    /// inside a `tracing` info span `stage` with the fields `name`, `stage_type` and `doc_id`
    /// (target `impeach_pipeline::runner`), and its wall time is logged and returned.
    ///
    /// # Errors
    /// Input errors as in [`PipelineInput::into_document`]; the first stage error, wrapped in
    /// [`PipelineError::Stage`].
    pub fn run(&self, ctx: &mut PipelineContext, input: PipelineInput) -> Result<RunReport> {
        let started = Instant::now();
        let document = input.into_document()?;
        let doc_id = document.doc_id.as_str().to_owned();
        ctx.data.document = Some(document);
        let mut report = RunReport::default();
        for (stage, stage_type) in self.stages.iter().zip(&self.stage_types) {
            let name = stage.name();
            let span = tracing::info_span!(
                "stage",
                name,
                stage_type = *stage_type,
                doc_id = doc_id.as_str()
            );
            let _entered = span.enter();
            tracing::debug!(stage = name, "stage started");
            let stage_started = Instant::now();
            let outcome = stage.run(ctx);
            let elapsed = stage_started.elapsed();
            if let Err(source) = outcome {
                tracing::error!(
                    stage = name,
                    elapsed_ms = elapsed.as_millis(),
                    "stage failed: {source}"
                );
                return Err(PipelineError::Stage {
                    stage: name.to_owned(),
                    source: Box::new(source),
                });
            }
            tracing::info!(
                stage = name,
                elapsed_ms = elapsed.as_secs_f64() * 1000.0,
                "stage finished"
            );
            report.stage_timings.push(StageTiming { stage: name.to_owned(), elapsed });
        }
        report.total = started.elapsed();
        Ok(report)
    }
}
