//! Ollama LLM backend (Python `OllamaService`, which called `/api/generate` through
//! langchain-ollama).
//!
//! The [`Llm`] trait is synchronous, so each [`OllamaLlm`] owns a private current-thread tokio
//! runtime and blocks on the HTTP call. Do not call it from inside an async task.
//!
//! A request that cannot reach the server (connection refused, connect timeout) fails with
//! [`ModelError::Unavailable`] naming the `base_url` and how to start Ollama or run without it;
//! [`OllamaLlm::check_reachable`] performs that check up front.

use std::time::Duration;

use impeach_config::pipeline::OllamaModelConfig;
use ollama_rs::error::OllamaError;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::parameters::{FormatType, JsonStructure};
use ollama_rs::models::ModelOptions;
use ollama_rs::Ollama;
use tokio::runtime::Runtime;

use crate::error::{error_chain, ModelError};
use crate::json::extract_json;
use crate::traits::{Llm, LlmRequest};

/// Time [`OllamaLlm::check_reachable`] waits for the server.
pub const REACHABILITY_TIMEOUT: Duration = Duration::from_secs(5);

/// Settings of an [`OllamaLlm`].
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaOptions {
    /// Server URL, e.g. `http://localhost:11434`.
    pub base_url: String,
    /// Model tag, e.g. `qwen2.5:7b-instruct-q4_K_M`.
    pub model: String,
    /// Default sampling temperature.
    pub temperature: f32,
    /// Default context window.
    pub num_ctx: u32,
    /// HTTP request timeout; `None` waits indefinitely.
    pub timeout: Option<Duration>,
}

impl OllamaOptions {
    /// Options from the `models.ollama` config block.
    pub fn from_config(config: &OllamaModelConfig) -> Self {
        Self {
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            temperature: config.temperature as f32,
            num_ctx: config.num_ctx,
            timeout: None,
        }
    }
}

/// An LLM served by Ollama.
pub struct OllamaLlm {
    client: Ollama,
    runtime: Runtime,
    options: OllamaOptions,
}

impl std::fmt::Debug for OllamaLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaLlm").field("options", &self.options).finish_non_exhaustive()
    }
}

impl OllamaLlm {
    /// A client for `options.base_url`. Does not contact the server.
    pub fn new(options: &OllamaOptions) -> Result<Self, ModelError> {
        let load_error =
            |message: String| ModelError::Load { model: options.model.clone(), message };
        let client = Ollama::try_new(options.base_url.as_str()).map_err(|e| {
            load_error(format!("invalid Ollama base_url `{}`: {e}", options.base_url))
        })?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| load_error(format!("cannot start the tokio runtime: {e}")))?;
        Ok(Self { client, runtime, options: options.clone() })
    }

    /// The configured options.
    pub fn options(&self) -> &OllamaOptions {
        &self.options
    }

    /// Asks the server for its local models, waiting at most [`REACHABILITY_TIMEOUT`].
    ///
    /// # Errors
    /// [`ModelError::Unavailable`] when the server cannot be reached; [`ModelError::Inference`]
    /// for other failures.
    pub fn check_reachable(&self) -> Result<(), ModelError> {
        let call = self.client.list_local_models();
        match self
            .runtime
            .block_on(async { tokio::time::timeout(REACHABILITY_TIMEOUT, call).await })
        {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(self.request_error(&error)),
            Err(_) => Err(unreachable_error(
                &self.options,
                &format!("no answer within {REACHABILITY_TIMEOUT:?}"),
            )),
        }
    }

    /// The error of a failed request: [`unreachable_error`] when the server could not be reached,
    /// else an inference error.
    fn request_error(&self, error: &OllamaError) -> ModelError {
        if let OllamaError::ReqwestError(source) = error {
            if source.is_connect() || source.is_timeout() {
                return unreachable_error(&self.options, &error_chain(error));
            }
        }
        ModelError::Inference(format!(
            "Ollama request to {} (model `{}`) failed: {}",
            self.options.base_url,
            self.options.model,
            error_chain(error)
        ))
    }

    fn generate(
        &self,
        req: &LlmRequest<'_>,
        format: Option<FormatType>,
    ) -> Result<String, ModelError> {
        let model_options = ModelOptions::default()
            .temperature(req.temperature.unwrap_or(self.options.temperature))
            .num_ctx(u64::from(req.num_ctx.unwrap_or(self.options.num_ctx)));
        let mut request =
            GenerationRequest::new(self.options.model.clone(), req.prompt).options(model_options);
        if let Some(system) = req.system {
            request = request.system(system);
        }
        if let Some(format) = format {
            request = request.format(format);
        }
        let call = self.client.generate(request);
        let response = match self.options.timeout {
            Some(timeout) => {
                self.runtime.block_on(async { tokio::time::timeout(timeout, call).await }).map_err(
                    |_| ModelError::Inference(format!("Ollama did not answer within {timeout:?}")),
                )?
            }
            None => self.runtime.block_on(call),
        };
        response.map(|r| r.response).map_err(|e| self.request_error(&e))
    }
}

/// [`ModelError::Unavailable`] for an Ollama server that cannot be reached.
fn unreachable_error(options: &OllamaOptions, cause: &str) -> ModelError {
    ModelError::Unavailable {
        backend: "ollama".to_owned(),
        reason: format!(
            "Ollama is not reachable at {} ({cause}). Start it with `ollama serve` and pull the \
             model with `ollama pull {}`, or set models.ollama.base_url to a running server. To \
             run without an LLM, disable FactStage, set allow_llm: false on EDStage and \
             judge_model: nli on ContradictionCheckStage, or inject a mock LLM \
             (impeach_models::MockLlm) into the model registry",
            options.base_url, options.model
        ),
    }
}

/// The Ollama `format` for a request: the schema as structured output, else plain JSON mode.
fn json_format(schema: Option<&serde_json::Value>) -> Result<FormatType, ModelError> {
    match schema {
        Some(schema) => serde_json::from_value::<JsonStructure>(schema.clone())
            .map(|s| FormatType::StructuredJson(Box::new(s)))
            .map_err(|e| ModelError::Protocol(format!("invalid JSON schema: {e}"))),
        None => Ok(FormatType::Json),
    }
}

impl Llm for OllamaLlm {
    fn complete(&self, req: &LlmRequest<'_>) -> Result<String, ModelError> {
        self.generate(req, None)
    }

    fn complete_json(&self, req: &LlmRequest<'_>) -> Result<serde_json::Value, ModelError> {
        let format = json_format(req.json_schema)?;
        let text = self.generate(req, Some(format))?;
        extract_json(&text)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn schema_becomes_structured_format() {
        let schema = json!({
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"]
        });
        let format = json_format(Some(&schema)).expect("schema");
        assert_eq!(serde_json::to_value(&format).expect("serialise"), schema);
        let plain = json_format(None).expect("json mode");
        assert_eq!(serde_json::to_value(&plain).expect("serialise"), json!("json"));
    }

    #[test]
    fn client_construction_does_not_connect() {
        let options = OllamaOptions {
            base_url: "http://127.0.0.1:9".to_owned(),
            model: "qwen2.5:7b-instruct-q4_K_M".to_owned(),
            temperature: 0.2,
            num_ctx: 16384,
            timeout: Some(Duration::from_secs(5)),
        };
        let llm = OllamaLlm::new(&options).expect("client");
        assert_eq!(llm.options().num_ctx, 16384);
        assert!(
            OllamaLlm::new(&OllamaOptions { base_url: "not a url".to_owned(), ..options }).is_err()
        );
    }
}
