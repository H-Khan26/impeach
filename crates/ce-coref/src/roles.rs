//! Role-term checks for canonical mention selection (Python `RoleConfig.is_valid_role_term`
//! and `RoleConfig.is_hybrid_role_name` over `roles.yaml`).

use std::collections::HashSet;

use ce_config::mappings::Roles;
use ce_text::normalize::{is_py_whitespace, py_split, py_strip};
use indexmap::IndexMap;

/// Known role terms (uppercase) and their normalisation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoleTerms {
    seed_terms: HashSet<String>,
    normalization: IndexMap<String, String>,
}

impl RoleTerms {
    /// Role terms from `roles.yaml` (every `role_seed_list` term, uppercased).
    pub fn from_roles(roles: &Roles) -> Self {
        Self {
            seed_terms: roles.role_seed_list.values().flatten().map(|t| t.to_uppercase()).collect(),
            normalization: roles.role_normalization.clone(),
        }
    }

    /// `term` uppercased and stripped, mapped through `role_normalization`.
    fn normalize(&self, term: &str) -> String {
        let upper = term.to_uppercase();
        let upper = py_strip(&upper);
        self.normalization.get(upper).cloned().unwrap_or_else(|| upper.to_owned())
    }

    /// Whether `name` is exactly a known role (`Plaintiff`, `Defendants`).
    pub fn is_role_term(&self, name: &str) -> bool {
        !name.is_empty() && self.seed_terms.contains(&self.normalize(name))
    }

    /// Whether `name` is a role followed by a name (`Defendant Reed`): it starts with a role
    /// term and a space, and more than two characters follow.
    pub fn is_hybrid_role_name(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let normalized = py_strip(name).to_uppercase();
        self.seed_terms.iter().any(|role| {
            normalized
                .strip_prefix(role.as_str())
                .and_then(|rest| rest.strip_prefix(' '))
                .is_some_and(|rest| {
                    let remainder = rest.trim_matches(is_py_whitespace);
                    py_split(remainder).next().is_some() && remainder.chars().count() > 2
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles() -> RoleTerms {
        let roles = Roles::load(ce_config::mappings::ROLES_PATH).expect("roles.yaml loads");
        RoleTerms::from_roles(&roles)
    }

    #[test]
    fn detects_pure_role_terms() {
        let roles = roles();
        assert!(roles.is_role_term("Plaintiff"));
        assert!(roles.is_role_term("Defendant"));
        assert!(roles.is_role_term("Manager"));
        assert!(roles.is_role_term(" plaintiffs "));
        assert!(!roles.is_role_term("Maria Lopez"));
        assert!(!roles.is_role_term("John Smith"));
        assert!(!roles.is_role_term(""));
    }

    #[test]
    fn detects_hybrid_role_names() {
        let roles = roles();
        assert!(roles.is_hybrid_role_name("Defendant Reed"));
        assert!(roles.is_hybrid_role_name("Plaintiff Smith"));
        assert!(roles.is_hybrid_role_name("Manager Johnson"));
        assert!(!roles.is_hybrid_role_name("Defendant"));
        assert!(!roles.is_hybrid_role_name("Defendant Al"));
        assert!(!roles.is_hybrid_role_name("Maria Lopez"));
        assert!(!roles.is_hybrid_role_name("John Smith"));
    }

    #[test]
    fn empty_roles_detect_nothing() {
        let roles = RoleTerms::default();
        assert!(!roles.is_role_term("Plaintiff"));
        assert!(!roles.is_hybrid_role_name("Defendant Reed"));
    }
}
