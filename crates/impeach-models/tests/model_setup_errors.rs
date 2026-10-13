//! Error messages for common model misconfigurations: a model missing from an empty cache with
//! the hub unreachable, and an Ollama server that is not running.
//!
//! The test sets `HF_ENDPOINT` to a closed local port and clears `HF_HOME`, so it lives in its
//! own test binary and needs no network access or model files. It needs every model backend
//! feature (the default).
#![cfg(all(feature = "fastembed", feature = "ort-nli", feature = "gliner2", feature = "ollama"))]

use std::net::TcpListener;

use impeach_models::ModelError;

/// A local URL on which nothing listens.
fn closed_local_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    let port = listener.local_addr().expect("local address").port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

fn assert_names(message: &str, parts: &[&str]) {
    for part in parts {
        assert!(message.contains(part), "`{part}` missing from: {message}");
    }
}

#[test]
fn misconfigurations_have_actionable_messages() {
    // Edition 2021: setting variables is safe; this binary holds a single test.
    std::env::set_var("HF_ENDPOINT", closed_local_url());
    std::env::remove_var("HF_HOME");
    let cache = tempfile::tempdir().expect("empty cache dir");
    let cache_text = cache.path().display().to_string();

    {
        use impeach_models::{FastEmbedOptions, FastEmbedder};
        let options = FastEmbedOptions {
            cache_dir: Some(cache.path().to_path_buf()),
            ..FastEmbedOptions::new("BAAI/bge-small-en-v1.5")
        };
        let error = FastEmbedder::new(&options).expect_err("no model files and no hub");
        assert!(matches!(error, ModelError::Download { .. }), "{error:?}");
        assert_names(
            &error.to_string(),
            &["BAAI/bge-small-en-v1.5", &cache_text, "FASTEMBED_CACHE_DIR", "HF_HOME"],
        );
    }

    {
        use impeach_models::{OrtNli, OrtNliOptions};
        let options = OrtNliOptions {
            cache_dir: Some(cache.path().to_path_buf()),
            ..OrtNliOptions::new("cross-encoder/nli-deberta-v3-large")
        };
        let error = OrtNli::new(&options).expect_err("no model files and no hub");
        assert!(matches!(error, ModelError::Download { .. }), "{error:?}");
        assert_names(
            &error.to_string(),
            &["cross-encoder/nli-deberta-v3-large", &cache_text, "HF_HOME", "deberta_nli"],
        );
    }

    {
        use impeach_models::{Gliner2Ner, Gliner2Options};
        let options = Gliner2Options {
            model_dir: Some(cache.path().to_path_buf()),
            ..Gliner2Options::new("fastino/gliner2-base-v1")
        };
        let error = Gliner2Ner::new(&options).expect_err("empty model directory");
        let message = error.to_string();
        for part in ["fastino/gliner2-base-v1", cache_text.as_str(), "tokenizer.json"] {
            assert!(message.contains(part), "`{part}` missing from: {message}");
        }
    }

    {
        use impeach_models::{Llm, LlmRequest, OllamaLlm, OllamaOptions};
        let base_url = closed_local_url();
        let options = OllamaOptions {
            base_url: base_url.clone(),
            model: "qwen2.5:7b-instruct-q4_K_M".to_owned(),
            temperature: 0.2,
            num_ctx: 2048,
            timeout: Some(std::time::Duration::from_secs(30)),
        };
        let llm = OllamaLlm::new(&options).expect("client construction does not connect");
        let error = llm.complete(&LlmRequest::new("hello")).expect_err("nothing listens");
        assert!(
            matches!(&error, ModelError::Unavailable { backend, .. } if backend == "ollama"),
            "{error:?}"
        );
        let message = error.to_string();
        for part in [base_url.as_str(), "not reachable", "ollama serve", "mock"] {
            assert!(message.contains(part), "`{part}` missing from: {message}");
        }
        let error = llm.check_reachable().expect_err("nothing listens");
        assert!(error.to_string().contains("not reachable"), "{error}");
    }
}
