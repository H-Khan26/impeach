//! The ED service (Python `services/ed/service.py`): routes a mention context to the rule-first,
//! hybrid or LLM-first strategy, retrieves candidates (exact, similarity, embedding) and breaks
//! ties with an [`ce_models::Llm`] using `assets/prompts/ed/disambiguation.json`.
