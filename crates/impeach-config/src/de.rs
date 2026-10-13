//! Deserialisation helpers shared by the config modules.

use std::fmt;
use std::marker::PhantomData;

use serde::de::value::MapAccessDeserializer;
use serde::de::{DeserializeOwned, Deserializer, Error as _, MapAccess, Visitor};

use crate::error::{ConfigError, Result};

/// Parses a YAML document, reporting empty documents as [`ConfigError::Empty`].
pub(crate) fn parse_yaml<T: DeserializeOwned>(text: &str, origin: &str) -> Result<T> {
    yaml_serde::from_str(text).map_err(|source| {
        if matches!(yaml_serde::from_str::<yaml_serde::Value>(text), Ok(yaml_serde::Value::Null)) {
            ConfigError::Empty { path: origin.to_owned() }
        } else {
            ConfigError::Yaml { path: origin.to_owned(), source }
        }
    })
}

/// A union discriminated by a `type` key whose remaining keys form the variant's body.
pub(crate) trait TaggedUnion: Sized {
    /// Accepted `type` values, for error messages.
    const VARIANTS: &'static [&'static str];

    /// Deserialises the variant named `tag` from `body` (the mapping without `type`).
    fn from_tagged<'de, D: Deserializer<'de>>(tag: &str, body: D) -> Result<Self, D::Error>;
}

/// Deserialises a [`TaggedUnion`].
///
/// When `type` is the first key, the rest of the mapping streams straight into the variant
/// struct, so nested errors keep their full YAML path (`stages[7].retrieval: unknown field`).
/// Otherwise the mapping is buffered and errors carry the path of the union node only.
pub(crate) fn deserialize_tagged<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: TaggedUnion,
{
    deserializer.deserialize_map(TaggedVisitor(PhantomData))
}

struct TaggedVisitor<T>(PhantomData<T>);

impl<'de, T: TaggedUnion> Visitor<'de> for TaggedVisitor<T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a mapping with a `type` key (one of {:?})", T::VARIANTS)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<T, A::Error> {
        let Some(first) = map.next_key::<String>()? else {
            return Err(A::Error::missing_field("type"));
        };
        if first == "type" {
            let tag: String = map.next_value()?;
            return T::from_tagged(&tag, MapAccessDeserializer::new(map));
        }

        let mut entries = yaml_serde::Mapping::new();
        entries.insert(yaml_serde::Value::String(first), map.next_value()?);
        while let Some(key) = map.next_key::<String>()? {
            entries.insert(yaml_serde::Value::String(key), map.next_value()?);
        }
        let Some(tag) = entries.remove("type") else {
            return Err(A::Error::missing_field("type"));
        };
        let tag: String = yaml_serde::from_value(tag).map_err(A::Error::custom)?;
        T::from_tagged(&tag, yaml_serde::Value::Mapping(entries)).map_err(A::Error::custom)
    }
}

/// Body of a variant that has no fields besides `type`.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoFields {}
