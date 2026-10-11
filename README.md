# contradiction-engine (Rust)

Finds where witness statements and legal documents disagree. A batch pipeline
reads each document once, turns what it says into small structured facts,
works out who and what each fact is about, keeps them in a per-case graph, and
checks every new fact against the graph and the rest of its own document.
Clean facts are committed; conflicting pairs go to a review queue.

Everything runs locally in one binary. This is a from-scratch Rust
reimplementation of an earlier Python research prototype; the two share
configuration files, prompt files, data records and evaluation keys so that
outputs can be compared, but no code.

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
  ce-core/                 data records shared by every crate: DocumentPayload, Chunk, Span,
                           Mention, Entity, Fact, ContradictionPair, ContradictionReport, EdResult,
                           typed ids, error types. serde field names match the Python records.
  ce-config/               strict pipeline config (deny_unknown_fields), mappings registry
                           (entity types, detection patterns, normalization, name matching, roles,
                           fact taxonomy, gliner label sets), prompt loading, asset resolution.
  ce-text/                 pure text utilities: normalization, tokenization, fuzzy similarity,
                           regex helpers, value parsing, retry policy.
  ce-ingest/               document loaders (pdf, txt, json), transcript parser, JSONL I/O,
                           run_ingestion.
  ce-store/                GraphStore trait (nodes, edges, cosine vector search, BM25 text search);
                           backends: helix (embedded HelixDB v3) and memory (tests, fallback);
                           SQLite review store (contradiction_review_queue).
  ce-models/               model traits: Embedder, NliJudge, Llm, NerModel, Coref; ModelRegistry;
                           backends behind cargo features: fastembed (embeddings), ort-nli
                           (DeBERTa cross-encoder), gliner2, ollama (HTTP), sidecar (JSON-lines
                           subprocess), mock (tests).
  ce-ner/                  mention detection: regex rules, legal citation detector, GLiNER2
                           adapter, merging and quality filters, defined-term extraction.
  ce-coref/                coreference: cluster post-processing, canonical mention selection,
                           pronoun mention synthesis; backends rule / llm / sidecar.
  ce-ed/                   entity disambiguation: candidate retrieval, name matching and
                           similarity, structured keys, rule resolvers, LLM tie-break, entity creation.
  ce-facts/                LLM fact extraction (single-pass and multi-pass), entity linking,
                           fact indexing.
  ce-contradictions/       similar-fact retrieval, pair prefilter, gating strategies, NLI context,
                           LLM judge, contradiction check service, review-queue items.
  ce-pipeline/             Stage trait, PipelineContext, runner, factory (config -> stages),
                           the eight stages, run_processing.
  ce-eval/                 evaluator: parse a markdown key, match detected pairs to key pairs by
                           embedding similarity with a value guard, optimal one-to-one assignment,
                           recall / precision / F1 report.
  ce-cli/                  the `contradiction-engine` binary: ingest, process, run, eval, store reset.
assets/
  config/processing/       pipeline YAML (same files as the Python prototype)
  config/mappings/         vocabulary and rule YAML
  prompts/                 LLM prompt JSON
fixtures/
  input/                   evaluation documents (JSONL) and contradiction test scenarios
  keys/                    hand-labelled contradiction keys
  reference/               outputs produced by the Python prototype, used for parity tests
```

Dependency direction is strictly downward:

```text
ce-cli -> ce-pipeline -> { ce-ner, ce-coref, ce-ed, ce-facts, ce-contradictions, ce-eval }
                      -> { ce-ingest, ce-store, ce-models } -> { ce-config, ce-text } -> ce-core
```

Domain crates (`ce-ner` .. `ce-contradictions`) depend on `ce-models` traits and
`ce-store` traits, never on a concrete backend. Backends are chosen by the
factory from the config and gated by cargo features so that a minimal build
(`--no-default-features`) compiles without ONNX Runtime, candle or HelixDB.

## Design rules

1. The YAML config is the only source of truth; no environment-variable fallbacks.
   Unknown or missing fields fail at load time.
2. Record and config field names match the Python prototype so the same files
   load in both and outputs diff cleanly.
3. Facts are committed only after the contradiction check; contradicting facts
   never enter the graph.
4. Every model sits behind a trait with a mock implementation; all pipeline
   logic is unit-testable without models, network or a database.
5. Asset paths in config (`contradiction_engine/prompts/...`,
   `contradiction_engine/config/mappings/...`) resolve against the working
   directory first, then against the assets compiled into the binary, with the
   `contradiction_engine/` prefix mapped to the asset root.

## Stack

| Concern | Crate | Notes |
| --- | --- | --- |
| Graph + vector + BM25 store | `helix-db` (git, `embedded` feature) | In-process HelixDB v3; `memory` backend for tests |
| Embeddings | `fastembed` | `BAAI/bge-small-en-v1.5` (384-d); `all-MiniLM-L6-v2` for the evaluator |
| NLI judge | `ort` + `tokenizers` | `cross-encoder/nli-deberta-v3-large` ONNX from the HF repo |
| NER / relations | `gliner2` | `fastino/gliner2-base-v1` on candle |
| LLM | `ollama-rs` | Structured JSON output; in-process `llama-cpp-2` is a later option |
| Coreference | rule / LLM prompt / sidecar | No Rust model exists; the sidecar protocol is JSON lines over stdio |
| Legal citations | regex port | No Rust eyecite; the patterns the pipeline relies on are ported |
| Chunking | `text-splitter` | Character-based sentence / recursive chunking with overlap |
| PDF | `pdf-extract` | Pure Rust, page-separated text |
| Config | `serde` + `yaml_serde` | `deny_unknown_fields` everywhere |
| Review queue | `rusqlite` (bundled) | |

## Translation phases

Each phase is one branch and one pull request into `main`, reviewed and
tested before merge.

| Phase | Branch | Delivers |
| --- | --- | --- |
| 1 | `phase-1-foundation` | Workspace, CI, `ce-core`, `ce-config`, `ce-text`, assets, fixtures |
| 2 | `phase-2-ingest-store` | `ce-ingest`, `ce-store` (helix + memory + sqlite), ingestion parity |
| 3 | `phase-3-models-pipeline` | `ce-models`, `ce-pipeline` runner, `RagIndexStage`, `ce-eval`, CLI skeleton |
| 4 | `phase-4-mentions-coref` | `ce-ner`, `ce-coref`, Mention / Definition / Coreference stages |
| 5 | `phase-5-entity-disambiguation` | `ce-ed`, Ed and Relation stages |
| 6 | `phase-6-facts` | `ce-facts`, Fact stage |
| 7 | `phase-7-contradictions` | `ce-contradictions`, Contradiction stage, `run_processing`, full CLI, e2e runner |
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

Under construction; see the phase table. Accuracy work (pairing recall, NLI
false positives, model upgrades) continues to be measured with `ce-eval`
against the keys in `fixtures/keys/`.

## License

All rights reserved.
