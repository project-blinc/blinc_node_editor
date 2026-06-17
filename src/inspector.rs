//! Inspector — config-schema-driven properties panel.
//!
//! Per DIRECTIVE.md §7: the inspector renders a form derived from
//! the selected node's [`NodeTemplate::config_schema`](crate::node::NodeTemplate::config_schema),
//! surfaces port-metadata hints (units, IIP defaults), and emits
//! [`InspectorPatchRequest`] events as the user edits values.
//!
//! ## Scope
//!
//! The strongly-typed [`crate::config::ConfigSchema`] supersedes the
//! opaque `serde_json::Value` schema the editor used previously. This
//! module ships the schema-walking surface hosts can render with:
//!
//! * [`InspectorField`] — a (definition, current value, validation
//!   issues) triple for one property.
//! * [`fields`] — iterator that pairs each property with the current
//!   value from a node's `config` JSON object.
//! * [`apply_patch`] — merges an [`InspectorPatchRequest`] into a
//!   `serde_json::Value::Object` (the host's standard merge path).
//!
//! Hosts implement the actual widgetry against their own UI
//! toolkit (`blinc_cn::input`, custom forms, …). The editor crate
//! itself stays UI-toolkit-agnostic.
//!
//! ## Patch convention
//!
//! [`InspectorPatchRequest::path`] is the property's `key` (the
//! same string from `PropertyMeta::key`). Hosts can use the
//! [`apply_patch`] helper to write the value into the node's
//! `config` map — or implement their own merge if they need
//! dotted-path semantics for nested fields. The editor never
//! mutates `NodeInstance::config` itself.

use crate::config::{validate, ConfigSchema, PropertyDefinition, ValidationIssue};
use crate::node::NodeId;
use serde_json::{Map, Value};

/// Fired when the user edits a config value in the inspector. The
/// host applies the patch to the node's
/// [`config`](crate::node::NodeInstance::config) — typically via
/// [`apply_patch`] — and re-syncs.
#[derive(Debug, Clone)]
pub struct InspectorPatchRequest {
    pub node: NodeId,
    /// Property key — MUST match the top-level
    /// [`PropertyMeta::key`](crate::config::PropertyMeta::key) on
    /// the schema's definition.
    ///
    /// The editor's helpers ([`apply_patch`],
    /// [`crate::NodeEditor::apply_inspector_patch`],
    /// [`crate::NodeEditor::patch_node_config`]) treat `path` as a
    /// flat key into the top-level config object — there is no
    /// dotted-path / nested-merge semantics. Hosts that store
    /// nested config (`config.connection.host`) MUST flatten or
    /// merge to top-level keys before dispatching the patch, or
    /// emit one `InspectorPatchRequest` per leaf and have their
    /// own apply step.
    ///
    /// The same key flows through to
    /// [`crate::config::PropertyRule::triggers`] and
    /// [`crate::EditorEvent::NodeConfigChanged::key`] — all three
    /// surfaces share this flat-key contract.
    pub path: String,
    pub value: Value,
}

/// One property + its current value + any open validation issues.
/// Yielded by [`fields`] and consumed by host-side form renderers.
pub struct InspectorField<'a> {
    pub definition: &'a PropertyDefinition,
    /// `None` when the field is unset in the current config; hosts
    /// fall back to [`PropertyDefinition::default_value`] when
    /// painting placeholders.
    pub current_value: Option<&'a Value>,
    /// Issues touching this field (filtered from a full
    /// [`validate`] sweep against `definition.meta.key`).
    pub issues: Vec<ValidationIssue>,
}

/// Walk a template's schema and produce one [`InspectorField`] per
/// property, paired with the matching slot from `config` (when
/// present). Validation runs once over the full schema; per-field
/// issues are partitioned by key.
///
/// `config` SHOULD be a `Value::Object`; non-objects are treated as
/// "no values set" and every required field surfaces a missing-
/// value issue.
pub fn fields<'a>(schema: &'a ConfigSchema, config: &'a Value) -> Vec<InspectorField<'a>> {
    let issues = validate(schema, config);
    let object = config.as_object();

    schema
        .properties
        .iter()
        .map(|def| {
            let key = &def.meta().key;
            let current_value = object.and_then(|o| o.get(key));
            let field_issues = issues.iter().filter(|i| i.key == *key).cloned().collect();
            InspectorField {
                definition: def,
                current_value,
                issues: field_issues,
            }
        })
        .collect()
}

/// Apply a patch from the inspector to a node's config object.
/// Hosts use this in their `EditorEvent::InspectorPatchRequested`
/// handler when their config matches the schema's flat-key
/// layout 1:1.
///
/// Behaviour:
/// * `request.path` is treated as a TOP-LEVEL key into the config
///   object. No dotted-path / nested-merge semantics — hosts that
///   need those must merge themselves before calling this helper
///   (or implement their own apply path).
/// * If `config` is `Value::Null`, it's promoted to an empty
///   `Value::Object` first.
/// * If `config` is not an object (string / array / etc.) the patch
///   is rejected and the function returns `false`.
/// * `request.value == Value::Null` removes the key.
/// * Any other value writes / overwrites at `request.path`.
///
/// Returns `true` when the config was mutated.
pub fn apply_patch(config: &mut Value, request: &InspectorPatchRequest) -> bool {
    if config.is_null() {
        *config = Value::Object(Map::new());
    }
    let Some(obj) = config.as_object_mut() else {
        return false;
    };
    if request.value.is_null() {
        obj.remove(&request.path).is_some()
    } else {
        obj.insert(request.path.clone(), request.value.clone());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NumberProperty, TextProperty};

    fn schema() -> ConfigSchema {
        ConfigSchema::from(vec![
            TextProperty::new("name", "Name").required(true).into(),
            NumberProperty::new("threshold", "Threshold")
                .default(0.5)
                .range(0.0, 1.0)
                .into(),
        ])
    }

    #[test]
    fn fields_pair_definitions_with_current_values() {
        let schema = schema();
        let config = serde_json::json!({ "name": "alice", "threshold": 0.7 });
        let fields = fields(&schema, &config);
        assert_eq!(fields.len(), 2);

        let name = &fields[0];
        assert_eq!(name.definition.meta().key, "name");
        assert_eq!(name.current_value, Some(&Value::String("alice".into())));
        assert!(name.issues.is_empty());

        let threshold = &fields[1];
        assert_eq!(threshold.definition.meta().key, "threshold");
        assert_eq!(threshold.current_value.and_then(|v| v.as_f64()), Some(0.7));
        assert!(threshold.issues.is_empty());
    }

    #[test]
    fn fields_surface_validation_issues_per_key() {
        let schema = schema();
        let config = serde_json::json!({ "threshold": 5.0 });
        let fields = fields(&schema, &config);

        let name = &fields[0];
        assert_eq!(name.issues.len(), 1, "required name should flag");
        assert_eq!(name.issues[0].key, "name");

        let threshold = &fields[1];
        assert_eq!(threshold.issues.len(), 1, "out-of-range threshold");
        assert_eq!(threshold.issues[0].key, "threshold");
    }

    #[test]
    fn apply_patch_writes_then_clears() {
        let schema = schema();
        let mut config = crate::config::default_config(&schema);

        let req = InspectorPatchRequest {
            node: NodeId::from("n1"),
            path: "name".into(),
            value: Value::String("bob".into()),
        };
        assert!(apply_patch(&mut config, &req));
        assert_eq!(
            config.as_object().unwrap().get("name"),
            Some(&Value::String("bob".into()))
        );

        // Null removes.
        let clear = InspectorPatchRequest {
            node: NodeId::from("n1"),
            path: "name".into(),
            value: Value::Null,
        };
        assert!(apply_patch(&mut config, &clear));
        assert!(!config.as_object().unwrap().contains_key("name"));
    }

    #[test]
    fn apply_patch_promotes_null_to_object() {
        let mut config = Value::Null;
        let req = InspectorPatchRequest {
            node: NodeId::from("n1"),
            path: "k".into(),
            value: Value::Bool(true),
        };
        assert!(apply_patch(&mut config, &req));
        assert!(config.is_object());
    }
}
