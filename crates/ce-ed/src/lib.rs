//! Entity disambiguation: resolves mentions to `Entity` nodes of the graph store (Python
//! `services/ed/`).
//!
//! - [`name_matching`]: name normalisation, loose name matching and the single entity type
//!   compatibility rule ([`entity_types_compatible`]).
//! - [`name_similarity`]: type-aware name similarity ([`NameSimilarity`]).
//! - [`structured_keys`]: exhibit / statute / case citation keys and their search terms.
//! - [`retrieval`]: the [`EdRetrieval`] contract and [`EdRetriever`] over a
//!   [`ce_store::GraphStore`] (bm25, `vector_fuzzy`, `all_entities_fuzzy`).
//! - [`rule_resolvers`]: deterministic resolvers for rule-first labels.
//! - [`context_builder`]: sentence-window and document contexts of mentions.
//! - [`service`]: strategy routing and the LLM tie-break.
//! - [`entity_creator`]: reuse-before-create and entity creation with role blocking.
//! - [`resolver`]: the per-document resolver used by `EDStage`.

pub mod context_builder;
pub mod entity_creator;
pub mod error;
pub mod name_matching;
pub mod name_similarity;
pub mod resolver;
pub mod retrieval;
pub mod rule_resolvers;
pub mod service;
pub mod structured_keys;

pub use context_builder::ContextBuilder;
pub use entity_creator::{EntityCreator, EntityOutcome};
pub use error::{recover_store, EdError, Result};
pub use name_matching::{entity_types_compatible, names_match, normalize_for_matching};
pub use name_similarity::{date_similarity, money_similarity, NameSimilarity};
pub use resolver::{
    collect_cluster_aliases, CoreEdResolver, EdResolver, IntraDocumentMatcher, ResolveInput,
};
pub use retrieval::{
    candidate_from_entity, EdRetrieval, EdRetriever, RetrieverSettings, StructuredKeyQuery,
};
pub use rule_resolvers::{
    resolve_case_rule, resolve_exhibit_rule, resolve_generic_rule, resolve_statute_rule,
};
pub use service::{EdService, EdServiceSettings};
pub use structured_keys::{CaseCitationKey, ExhibitKey, FederalRuleKey, KeyPatterns, StatuteKey};
