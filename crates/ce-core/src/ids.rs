//! String identifier newtypes.
//!
//! Every id serialises as a plain JSON string, so records keep the Python wire format while Rust
//! code cannot mix up, say, a mention id and an entity id.
//!
//! Conventions: [`FactId`] identifies a fact both before commit (extraction ids such as
//! `STATED_FACT_0`) and after (graph ids of committed facts); [`NodeId`] is the graph store id of a
//! document or chunk node (the Python `helix_id` fields).

use std::borrow::Borrow;
use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Wraps a string as this id type.
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            /// The id as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Unwraps the id into its string.
            #[must_use]
            pub fn into_inner(self) -> String {
                self.0
            }

            /// Whether the id is the empty string.
            #[must_use]
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self(id)
            }
        }

        impl From<&str> for $name {
            fn from(id: &str) -> Self {
                Self(id.to_owned())
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }
    };
}

string_id!(
    /// Document id (`doc_id`).
    DocId
);
string_id!(
    /// Chunk id.
    ChunkId
);
string_id!(
    /// Sentence or mention span id.
    SpanId
);
string_id!(
    /// Mention id (the Python `ExtractedEntity.id`).
    MentionId
);
string_id!(
    /// Resolved entity id.
    EntityId
);
string_id!(
    /// Fact id, either an extraction id or the id of a committed fact.
    FactId
);
string_id!(
    /// Graph store node id of a document or chunk (`helix_id`).
    NodeId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialises_as_plain_string() {
        let id = DocId::new("doc-1");
        assert_eq!(serde_json::to_string(&id).ok().as_deref(), Some("\"doc-1\""));
        let back: DocId = serde_json::from_str("\"doc-1\"").expect("valid JSON string");
        assert_eq!(back, id);
    }

    #[test]
    fn display_and_conversions() {
        let id = FactId::from("STATED_FACT_0");
        assert_eq!(id.to_string(), "STATED_FACT_0");
        assert_eq!(id, "STATED_FACT_0");
        assert_eq!(String::from(id.clone()), "STATED_FACT_0");
        assert_eq!(id.as_str(), "STATED_FACT_0");
        assert!(!id.is_empty());
        assert!(MentionId::default().is_empty());
    }

    #[test]
    fn orders_like_python_strings() {
        assert!(FactId::new("FACT_10") < FactId::new("FACT_2"));
        assert!(FactId::new("a") < FactId::new("b"));
    }
}
