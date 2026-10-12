//! `eval`: score e2e pipeline output against a ground-truth key (Python
//! `tests/e2e/evaluate_output.py`).

use ce_eval::{
    default_report_path, parse_key_file, run_evaluation_with_key, EvalError, EvalOptions,
};
use ce_models::Embedder;

use crate::cli::EvalArgs;
use crate::output::print;

/// Evaluates `args.input` against `args.key`, writes the JSON report and prints the summary.
///
/// # Errors
/// Missing inputs, an empty key, a build without `fastembed`, model, load and write errors.
pub fn execute(args: &EvalArgs) -> anyhow::Result<()> {
    for (path, label) in [(&args.input, "Input"), (&args.key, "Key")] {
        if !path.exists() {
            anyhow::bail!("{label} file not found: {}", path.display());
        }
    }
    // The key is parsed once and checked before the embedding model is loaded.
    let key = parse_key_file(&args.key)?;
    if key.is_empty() {
        return Err(EvalError::EmptyKey { path: args.key.clone() }.into());
    }
    let embedder = load_embedder(&args.model)?;

    let report_path = args.output.clone().unwrap_or_else(|| default_report_path(&args.input));
    let options = EvalOptions {
        threshold: args.threshold,
        use_value_guard: !args.no_value_guard,
        embedding_model: Some(args.model.clone()),
        ..EvalOptions::default()
    };
    let report = run_evaluation_with_key(
        &args.input,
        &key,
        &args.key,
        Some(&report_path),
        embedder.as_ref(),
        &options,
    )?;
    print(&report.summary(Some(&report_path.display().to_string())));
    Ok(())
}

#[cfg(feature = "fastembed")]
fn load_embedder(model: &str) -> anyhow::Result<Box<dyn Embedder>> {
    use ce_models::{FastEmbedOptions, FastEmbedder};

    tracing::info!(model, "loading embedding model");
    Ok(Box::new(FastEmbedder::new(&FastEmbedOptions::new(model))?))
}

#[cfg(not(feature = "fastembed"))]
fn load_embedder(model: &str) -> anyhow::Result<Box<dyn Embedder>> {
    anyhow::bail!(
        "eval needs the embedding model {model}, but this binary was built without the \
         `fastembed` feature; rebuild with `--features fastembed`"
    )
}
