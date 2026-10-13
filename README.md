<p align="center">
  <img src="docs/images/pipeline-animation.svg" alt="Documents are distilled into facts stored in a case-memory graph; new statements are compared against related facts and flagged or committed." width="100%">
</p>

# Impeach

Finds where witness statements and legal documents disagree.

A legal matter accumulates documents that all describe the same events:
incident reports, complaints, deposition transcripts, medical records, expert
reports. A different date, speed, signal colour or dollar figure between two of
them can decide credibility, and finding those differences by hand means
reading everything against everything.

Impeach reads each document once, turns what it says into small structured
facts, works out who and what each fact is about, and keeps them in a case
graph. Every new fact is checked against the facts already in the graph and
against the rest of its own document. Facts that hold up are committed; pairs
that conflict are flagged for a person to review.

Everything runs locally in one binary. No document leaves the machine.

## How it works

Two phases, repeated for every document in a case.

**Build the case memory.** The raw text or transcript is cleaned, split into
chunks and embedded. Names, places, dates and citations are detected, pronouns
and aliases are resolved, and each mention is matched to an entity that may
already exist from an earlier document. A local LLM then extracts the
document's claims as atomic facts linked to those entities.

**Check new testimony (contradiction detection).** Each extracted fact pulls
the related facts out of the graph, through the entities it shares and by
semantic similarity. Candidate pairs are filtered, then a
natural-language-inference model judges whether the two statements can both
be true. Clean facts join the graph. Contradictions and borderline pairs go to
a review queue.

<p align="center">
  <img src="docs/images/model-pipeline.svg" alt="The nine processing stages, the local model behind each, the graph nodes each stage writes, and the review queue fed by the contradiction check." width="100%">
</p>

The stack is a local LLM served by [Ollama](https://ollama.com) for fact
extraction, small Hugging Face encoders for embeddings, entity recognition and
the NLI judge, and [HelixDB](https://github.com/HelixDB/helix-db) embedded
in-process as the graph store.

## Quick start

Requirements: a Rust toolchain (pinned in `rust-toolchain.toml`) and Ollama.
On Windows, building HelixDB needs NASM on the `PATH`.

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
cargo build --release

impeach run doc.pdf --db-path out/review.sqlite3 -o out/doc.json
```

`impeach ingest` and `impeach process` run the two halves separately
(document to JSONL, JSONL to results). Process the documents of one case in
sequence; each is checked against what the earlier ones committed. Run
`impeach store reset` before starting an unrelated case.

`impeach run-e2e --scenario mason` runs a hand-labelled scenario end to end and
scores it against its key in `fixtures/keys/`.

## Pipeline

```text
file (.pdf / .txt / .json, or transcript .json / .csv / .txt)
  |  ingest                                  -> DocumentPayload (JSONL)
  v
RagIndexStage ........ chunk, embed, write Document / Chunk / Span nodes
MentionStage ......... citation + regex + GLiNER2 mentions, merged -> mention Spans
DefinitionStage ...... defined terms ("Plaintiff") from roles.yaml patterns
CoreferenceStage ..... clusters, canonical mention, pronoun mentions
EdStage .............. mentions -> Entity nodes (rules, string / BM25 / vector, LLM tie-break)
RelationStage ........ GLiNER2 relations (context only)
FactStage ............ LLM JSON facts, linked to entities
ContradictionStage ... intra-doc + cross-doc retrieval, prefilter, NLI / LLM judge
  |                    -> clean facts committed to the graph
  v                    -> contradictions to the report (+ SQLite review queue)
ProcessingResult
```

## Workspace layout

```text
Cargo.toml                 workspace; all dependency versions live in [workspace.dependencies]
rust-toolchain.toml        pinned stable toolchain
crates/
  impeach-core/            data records shared by every crate: DocumentPayload, Chunk, Span,
                           Mention, Entity, Fact, ContradictionPair, ContradictionReport, EdResult,
                           typed ids, error types. serde field names match the Python records.
  impeach-config/          strict pipeline config (deny_unknown_fields), mappings registry
                           (entity types, detection patterns, normalization, name matching, roles,
                           fact taxonomy, gliner label sets), prompt loading, asset resolution.
  impeach-text/            pure text utilities: normalization, tokenization, fuzzy similarity,
                           regex helpers, value parsing, retry policy.
  impeach-ingest/          document loaders (pdf, txt, json), transcript parser, JSONL I/O,
                           run_ingestion.
  impeach-store/           GraphStore trait (nodes, edges, cosine vector search, BM25 text search);
                           backends: helix (embedded HelixDB v3) and memory (tests, fallback);
                           SQLite review store (contradiction_review_queue).
  impeach-models/          model traits: Embedder, NliJudge, Llm, NerModel, Coref; ModelRegistry;
                           backends behind cargo features: fastembed (embeddings), ort-nli
                           (DeBERTa cross-encoder), gliner2, ollama (HTTP), sidecar (JSON-lines
                           subprocess), mock (tests).
  impeach-ner/             mention detection: regex rules, legal citation detector, GLiNER2
                           adapter, merging and quality filters, defined-term extraction.
  impeach-coref/           coreference: cluster post-processing, canonical mention selection,
                           pronoun mention synthesis; backends rule / llm / sidecar.
  impeach-ed/              entity disambiguation: candidate retrieval, name matching and
                           similarity, structured keys, rule resolvers, LLM tie-break, entity creation.
  impeach-facts/           LLM fact extraction (single-pass and multi-pass), entity linking,
                           fact indexing.
  impeach-contradictions/  contradiction detection: similar-fact retrieval, pair prefilter, gating
                           strategies, NLI context, LLM judge, check service, review-queue items.
  impeach-pipeline/        Stage trait, PipelineContext, runner, factory (config -> stages),
                           the eight stages, run_processing, e2e runner.
  impeach-eval/            evaluator: parse a markdown key, match detected pairs to key pairs by
                           embedding similarity with a value guard, optimal one-to-one assignment,
                           recall / precision / F1 report.
  impeach-cli/             the `impeach` binary: ingest, process, run, eval, run-e2e, store reset.
assets/
  config/processing/       pipeline YAML
  config/mappings/         vocabulary and rule YAML
  prompts/                 LLM prompt JSON
docs/images/               README diagrams
fixtures/
  input/                   evaluation documents (JSONL) and contradiction test scenarios
  keys/                    hand-labelled contradiction keys
  reference/               outputs produced by the Python prototype, used for parity tests
```

Dependency direction is strictly downward:

```text
impeach-cli -> impeach-pipeline -> { impeach-ner, impeach-coref, impeach-ed, impeach-facts,
                                     impeach-contradictions, impeach-eval }
            -> { impeach-ingest, impeach-store, impeach-models }
            -> { impeach-config, impeach-text } -> impeach-core
```

Domain crates (`impeach-ner` .. `impeach-contradictions`) depend on
`impeach-models` traits and `impeach-store` traits, never on a concrete
backend. Backends are chosen by the factory from the config and gated by cargo
features so that a minimal build (`--no-default-features`) compiles without
ONNX Runtime, candle or HelixDB.

## Design rules

1. The YAML config is the only source of truth; no environment-variable fallbacks.
   Unknown or missing fields fail at load time.
2. Record and config field names match the earlier Python prototype so its
   reference outputs diff cleanly against this implementation.
3. Facts are committed only after the contradiction check; contradicting facts
   never enter the graph.
4. Every model sits behind a trait with a mock implementation; all pipeline
   logic is unit-testable without models, network or a database.
5. Asset paths in config (`impeach/prompts/...`, `impeach/config/mappings/...`)
   resolve against the working directory first, then against the assets
   compiled into the binary, with the `impeach/` prefix mapped to the asset root.

## Stack

| Concern | Crate | Notes |
| --- | --- | --- |
| Graph + vector + BM25 store | `helix-db` (git, `embedded` feature) | In-process HelixDB v3; `memory` backend for tests |
| Embeddings | `fastembed` | `BAAI/bge-small-en-v1.5` (384-d); `all-MiniLM-L6-v2` for the evaluator |
| NLI judge | `ort` + `tokenizers` | `cross-encoder/nli-deberta-v3-large` ONNX from the HF repo |
| NER / relations | `gliner2` | `fastino/gliner2-base-v1` on candle |
| LLM | `ollama-rs` | Structured JSON output; in-process `llama-cpp-2` is a later option |
| Coreference | rule / LLM prompt / sidecar | No Rust model exists; the sidecar protocol is JSON lines over stdio |
| Legal citations | regex port | The citation patterns the pipeline relies on |
| Chunking | `text-splitter` | Character-based sentence / recursive chunking with overlap |
| PDF | `pdf-extract` | Pure Rust, page-separated text |
| Config | `serde` + `yaml_serde` | `deny_unknown_fields` everywhere |
| Review queue | `rusqlite` (bundled) | |

## Build phases

Each phase is one branch and one pull request into `main`, reviewed and
tested before merge.

| Phase | Branch | Delivers |
| --- | --- | --- |
| 1 | `phase-1-foundation` | Workspace, CI, `impeach-core`, `impeach-config`, `impeach-text`, assets, fixtures |
| 2 | `phase-2-ingest-store` | `impeach-ingest`, `impeach-store` (helix + memory + sqlite), ingestion parity |
| 3 | `phase-3-models-pipeline` | `impeach-models`, `impeach-pipeline` runner, `RagIndexStage`, `impeach-eval`, CLI skeleton |
| 4 | `phase-4-mentions-coref` | `impeach-ner`, `impeach-coref`, Mention / Definition / Coreference stages |
| 5 | `phase-5-entity-disambiguation` | `impeach-ed`, Ed and Relation stages |
| 6 | `phase-6-facts` | `impeach-facts`, Fact stage |
| 7 | `phase-7-contradictions` | `impeach-contradictions`, Contradiction stage, `run_processing`, full CLI, e2e runner |
| 8 | `phase-8-parity-hardening` | Stage-by-stage parity against the Python reference outputs, batched writes, perf |

## Building

```bash
cargo build --release                  # all backends
cargo build --no-default-features      # logic only, no model runtimes
cargo test --workspace                 # no models or network needed
cargo test --workspace -- --ignored    # model-backed tests (downloads models on first run)
```

Runtime requirements for the full pipeline: an Ollama server with the model
named in the config (`models.ollama.model`), and internet access on first run
to fetch the embedding, NLI and NER models into the local cache.

## Status

A personal research project under active development; see the phase table.
Accuracy is measured with `impeach eval` / `impeach run-e2e` on two
hand-labelled scenarios with known contradictions (`fixtures/keys/`).

## License

All rights reserved.
