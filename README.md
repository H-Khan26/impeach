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
