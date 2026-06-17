//! Typed schema for a node template's editable configuration.
//!
//! Replaces the opaque `serde_json::Value` the inspector used
//! previously with a strongly-typed enum so the editor (and hosts
//! that render their own inspector chrome) can match on property
//! kind, walk per-variant validation, and emit coherent patches.
//!
//! A `ConfigSchema` is just a list of [`PropertyDefinition`]s. Each
//! property declares its `key` (the field name in
//! [`NodeInstance::config`](crate::node::NodeInstance::config) — a
//! `serde_json::Value::Object`), a human-readable label, optional
//! description, and per-variant validation / defaults.
//!
//! ## Backwards compatibility
//!
//! Templates that need to ship an opaque payload (Zeal-style
//! `propertyRules`, `dataOperations`, custom rule DSLs that the
//! editor doesn't speak) can use [`PropertyDefinition::Custom`] —
//! the inspector skips its own form generation for those slots and
//! hosts render the field with their own widget.
//!
//! ## Example
//!
//! ```ignore
//! use blinc_node_editor::config::*;
//!
//! let schema = ConfigSchema::new()
//!     .with_property(
//!         SelectProperty::new("mode", "Mode")
//!             .option("strict", "Strict")
//!             .option("lenient", "Lenient")
//!             .default("strict"),
//!     )
//!     .with_property(
//!         NumberProperty::new("threshold", "Threshold")
//!             .default(0.5)
//!             .range(0.0, 1.0),
//!     )
//!     // Mode = "lenient" lowers the threshold default + clears any
//!     // user-supplied strict-only field.
//!     .with_rule(
//!         PropertyRule::new()
//!             .trigger("mode")
//!             .when(Predicate::Eq { key: "mode".into(), value: serde_json::json!("lenient") })
//!             .set("threshold", serde_json::json!(0.2)),
//!     );
//! ```
//!
//! ## Rules engine
//!
//! Schemas carry an optional list of [`PropertyRule`]s that drive
//! reactive cascades when config values change. [`cascade_rules`]
//! applies them iteratively (up to [`MAX_RULE_CASCADE_DEPTH`]) until
//! the config stabilises. Predicates ([`Predicate`]) compose with
//! `All` / `Any` / `Not` so a rule can key off arbitrary value
//! combinations; effects ([`PropertyEffect`]) patch the config in
//! turn. Hosts subscribe to the editor's `NodeConfigChanged` event
//! to observe each cascade step.

use serde_json::{Map, Value};

/// Schema for a node template's editable configuration.
///
/// Combines a list of [`PropertyDefinition`]s (the form widgets) with
/// an optional list of [`PropertyRule`]s (reactive cascades). An
/// empty schema means the node has no editable config (no inspector
/// pane is shown).
///
/// Construct via `ConfigSchema::new().with_property(...).with_rule(...)`,
/// or convert directly from a property vector via
/// `ConfigSchema::from(vec![...])` for cases that don't need rules.
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    /// Property definitions in render order.
    pub properties: Vec<PropertyDefinition>,
    /// Optional reactive cascades — fired when a property
    /// matching `triggers` changes. See [`cascade_rules`].
    pub rules: Vec<PropertyRule>,
}

impl ConfigSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a property to the schema.
    pub fn with_property(mut self, property: impl Into<PropertyDefinition>) -> Self {
        self.properties.push(property.into());
        self
    }

    /// Append a reactive rule to the schema.
    pub fn with_rule(mut self, rule: PropertyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Returns `true` when the schema has no properties — the
    /// inspector treats this as "no config pane".
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }
}

impl From<Vec<PropertyDefinition>> for ConfigSchema {
    fn from(properties: Vec<PropertyDefinition>) -> Self {
        Self {
            properties,
            rules: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// PropertyMeta — fields common to every property variant
// ─────────────────────────────────────────────────────────────────────

/// Common metadata every [`PropertyDefinition`] carries.
///
/// `key` is the field name written into
/// [`NodeInstance::config`](crate::node::NodeInstance::config); the
/// host receives it back unchanged in
/// [`InspectorPatchRequest::path`](crate::inspector::InspectorPatchRequest).
#[derive(Debug, Clone)]
pub struct PropertyMeta {
    pub key: String,
    pub label: String,
    pub description: Option<String>,
    /// When `true` the inspector flags missing / empty values during
    /// validation. Defaults to `false`.
    pub required: bool,
}

impl PropertyMeta {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            description: None,
            required: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// PropertyDefinition + variants
// ─────────────────────────────────────────────────────────────────────

/// A single editable field on a node template.
///
/// Variant choice maps to the widget the inspector renders. Hosts
/// that want richer types can extend via
/// [`PropertyDefinition::Custom`] and render their own form widget
/// for that field.
#[derive(Debug, Clone)]
pub enum PropertyDefinition {
    Text(TextProperty),
    Textarea(TextareaProperty),
    Number(NumberProperty),
    Boolean(BooleanProperty),
    Select(SelectProperty),
    Color(ColorProperty),
    File(FileProperty),
    CodeEditor(CodeEditorProperty),
    /// Opaque JSON payload for host-rendered fields. The inspector
    /// surfaces the meta (label / description) and exposes the
    /// `value` to hosts via [`crate::inspector::InspectorField`] —
    /// the host renders its own widget and emits patches against
    /// `meta.key`.
    ///
    /// **Default-seeding caveat**: unlike every other variant
    /// (which has a separate `Option<default>` field),
    /// [`PropertyDefinition::default_value`] returns
    /// `Some(value.clone())` for `Custom` whenever `value` is not
    /// `Value::Null`. So `Custom { ..., value: json!({...}) }`
    /// will be seeded into [`default_config`] output as if the
    /// payload were a default. Set `value` to `Value::Null` when
    /// the field should NOT be auto-seeded, or use a different
    /// variant if you need separate "schema example" + "default
    /// value" semantics.
    Custom { meta: PropertyMeta, value: Value },
}

impl PropertyDefinition {
    /// Borrow the common metadata block.
    pub fn meta(&self) -> &PropertyMeta {
        match self {
            Self::Text(p) => &p.meta,
            Self::Textarea(p) => &p.meta,
            Self::Number(p) => &p.meta,
            Self::Boolean(p) => &p.meta,
            Self::Select(p) => &p.meta,
            Self::Color(p) => &p.meta,
            Self::File(p) => &p.meta,
            Self::CodeEditor(p) => &p.meta,
            Self::Custom { meta, .. } => meta,
        }
    }

    /// Convenience accessor for the property's config key.
    pub fn key(&self) -> &str {
        &self.meta().key
    }

    /// The default value for this property, if any, as a JSON value.
    pub fn default_value(&self) -> Option<Value> {
        match self {
            Self::Text(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::Textarea(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::Number(p) => p.default.map(|n| {
                serde_json::Number::from_f64(n)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }),
            Self::Boolean(p) => p.default.map(Value::Bool),
            Self::Select(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::Color(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::File(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::CodeEditor(p) => p.default.as_ref().map(|s| Value::String(s.clone())),
            Self::Custom { value, .. } => {
                if value.is_null() {
                    None
                } else {
                    Some(value.clone())
                }
            }
        }
    }
}

// ─── Text ──────────────────────────────────────────────────────────────

/// Single-line text field. Renders to a `text_input`.
#[derive(Debug, Clone)]
pub struct TextProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    pub placeholder: Option<String>,
    pub max_length: Option<usize>,
}

impl TextProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            placeholder: None,
            max_length: None,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, value: impl Into<String>) -> Self {
        self.default = Some(value.into());
        self
    }

    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = Some(text.into());
        self
    }

    pub fn max_length(mut self, n: usize) -> Self {
        self.max_length = Some(n);
        self
    }
}

impl From<TextProperty> for PropertyDefinition {
    fn from(p: TextProperty) -> Self {
        Self::Text(p)
    }
}

// ─── Textarea ─────────────────────────────────────────────────────────

/// Multi-line text field. Renders to a `text_area`.
#[derive(Debug, Clone)]
pub struct TextareaProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    pub placeholder: Option<String>,
    /// Visible row count hint for the renderer. `None` lets the
    /// inspector pick.
    pub rows: Option<u32>,
}

impl TextareaProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            placeholder: None,
            rows: None,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, value: impl Into<String>) -> Self {
        self.default = Some(value.into());
        self
    }

    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = Some(text.into());
        self
    }

    pub fn rows(mut self, n: u32) -> Self {
        self.rows = Some(n);
        self
    }
}

impl From<TextareaProperty> for PropertyDefinition {
    fn from(p: TextareaProperty) -> Self {
        Self::Textarea(p)
    }
}

// ─── Number ───────────────────────────────────────────────────────────

/// Numeric field. Renders to a `number_input` (or slider, if a
/// finite range is given). Set [`integer`](Self::integer) to clamp
/// to integers; otherwise the field accepts decimals.
#[derive(Debug, Clone)]
pub struct NumberProperty {
    pub meta: PropertyMeta,
    pub default: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    /// Constrain to integer values. Defaults to `false`.
    pub integer: bool,
    /// Optional unit suffix surfaced next to the input (e.g. "ms",
    /// "px", "%"). Editor convention — hosts can ignore.
    pub unit: Option<String>,
}

impl NumberProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            min: None,
            max: None,
            step: None,
            integer: false,
            unit: None,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, value: f64) -> Self {
        self.default = Some(value);
        self
    }

    pub fn min(mut self, value: f64) -> Self {
        self.min = Some(value);
        self
    }

    pub fn max(mut self, value: f64) -> Self {
        self.max = Some(value);
        self
    }

    pub fn range(self, min: f64, max: f64) -> Self {
        self.min(min).max(max)
    }

    pub fn step(mut self, value: f64) -> Self {
        self.step = Some(value);
        self
    }

    pub fn integer(mut self) -> Self {
        self.integer = true;
        self
    }

    pub fn unit(mut self, suffix: impl Into<String>) -> Self {
        self.unit = Some(suffix.into());
        self
    }
}

impl From<NumberProperty> for PropertyDefinition {
    fn from(p: NumberProperty) -> Self {
        Self::Number(p)
    }
}

// ─── Boolean ──────────────────────────────────────────────────────────

/// Boolean toggle. Renders to a switch / checkbox.
#[derive(Debug, Clone)]
pub struct BooleanProperty {
    pub meta: PropertyMeta,
    pub default: Option<bool>,
}

impl BooleanProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, value: bool) -> Self {
        self.default = Some(value);
        self
    }
}

impl From<BooleanProperty> for PropertyDefinition {
    fn from(p: BooleanProperty) -> Self {
        Self::Boolean(p)
    }
}

// ─── Select ───────────────────────────────────────────────────────────

/// Enumerated value picked from a fixed option list. Renders to a
/// radio group / select / dropdown depending on inspector chrome.
#[derive(Debug, Clone)]
pub struct SelectProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    pub options: Vec<SelectOption>,
    /// When `true`, the selected value is a JSON array of option
    /// values rather than a single string. Defaults to `false`.
    pub multiple: bool,
}

impl SelectProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            options: Vec::new(),
            multiple: false,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, value: impl Into<String>) -> Self {
        self.default = Some(value.into());
        self
    }

    pub fn multiple(mut self) -> Self {
        self.multiple = true;
        self
    }

    pub fn option(mut self, value: impl Into<String>, label: impl Into<String>) -> Self {
        self.options.push(SelectOption {
            value: value.into(),
            label: label.into(),
        });
        self
    }
}

impl From<SelectProperty> for PropertyDefinition {
    fn from(p: SelectProperty) -> Self {
        Self::Select(p)
    }
}

/// A single entry in a [`SelectProperty`].
#[derive(Debug, Clone)]
pub struct SelectOption {
    /// Stored in `NodeInstance::config` under the property's key.
    pub value: String,
    /// Shown to the user.
    pub label: String,
}

// ─── Color ────────────────────────────────────────────────────────────

/// Colour picker. Stored as a hex string ("#rrggbb" or "#rrggbbaa").
#[derive(Debug, Clone)]
pub struct ColorProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    /// When `true` the picker exposes an alpha channel. Defaults to
    /// `false`.
    pub with_alpha: bool,
}

impl ColorProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            with_alpha: false,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, hex: impl Into<String>) -> Self {
        self.default = Some(hex.into());
        self
    }

    pub fn with_alpha(mut self) -> Self {
        self.with_alpha = true;
        self
    }
}

impl From<ColorProperty> for PropertyDefinition {
    fn from(p: ColorProperty) -> Self {
        Self::Color(p)
    }
}

// ─── File ─────────────────────────────────────────────────────────────

/// File path / upload picker. Stored as a string (host-defined —
/// path, URL, content URI).
#[derive(Debug, Clone)]
pub struct FileProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    /// Suggested file extensions (no leading dot) e.g.
    /// `["png", "jpg", "svg"]`. Advisory — hosts MAY enforce.
    pub accept: Vec<String>,
}

impl FileProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            accept: Vec::new(),
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, path: impl Into<String>) -> Self {
        self.default = Some(path.into());
        self
    }

    pub fn accept<I, S>(mut self, exts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.accept = exts.into_iter().map(Into::into).collect();
        self
    }
}

impl From<FileProperty> for PropertyDefinition {
    fn from(p: FileProperty) -> Self {
        Self::File(p)
    }
}

// ─── CodeEditor ───────────────────────────────────────────────────────

/// Multi-line code editor with an optional language hint. Stored as
/// a string.
#[derive(Debug, Clone)]
pub struct CodeEditorProperty {
    pub meta: PropertyMeta,
    pub default: Option<String>,
    /// Language tag for syntax highlighting (e.g. `"rust"`,
    /// `"javascript"`, `"sql"`, `"json"`, `"yaml"`). The inspector
    /// passes this to the host's code widget; the editor itself
    /// doesn't interpret it.
    pub language: Option<String>,
    /// Show line numbers in the gutter. Defaults to `true`.
    pub line_numbers: bool,
    /// Soft-wrap long lines. Defaults to `false`.
    pub line_wrap: bool,
}

impl CodeEditorProperty {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            meta: PropertyMeta::new(key, label),
            default: None,
            language: None,
            line_numbers: true,
            line_wrap: false,
        }
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.meta.description = Some(text.into());
        self
    }

    pub fn required(mut self, required: bool) -> Self {
        self.meta.required = required;
        self
    }

    pub fn default(mut self, source: impl Into<String>) -> Self {
        self.default = Some(source.into());
        self
    }

    pub fn language(mut self, tag: impl Into<String>) -> Self {
        self.language = Some(tag.into());
        self
    }

    pub fn line_numbers(mut self, on: bool) -> Self {
        self.line_numbers = on;
        self
    }

    pub fn line_wrap(mut self, on: bool) -> Self {
        self.line_wrap = on;
        self
    }
}

impl From<CodeEditorProperty> for PropertyDefinition {
    fn from(p: CodeEditorProperty) -> Self {
        Self::CodeEditor(p)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Defaults + validation
// ─────────────────────────────────────────────────────────────────────

/// Build the initial config object for a template — every property
/// with a declared default lands at its `key` in the resulting JSON
/// object. Properties without defaults are omitted (host can decide
/// whether to seed them as `null` or skip until first user edit).
/// Cascading rules run once over the seeded values so the initial
/// config already reflects any rule-driven defaults.
///
/// Returns `Value::Object({})` for an empty schema.
pub fn default_config(schema: &ConfigSchema) -> Value {
    let mut map = Map::new();
    for prop in &schema.properties {
        if let Some(v) = prop.default_value() {
            map.insert(prop.meta().key.clone(), v);
        }
    }
    let mut config = Value::Object(map);
    if !schema.rules.is_empty() {
        let trigger_keys: Vec<String> = schema
            .properties
            .iter()
            .map(|p| p.meta().key.clone())
            .collect();
        cascade_rules(&schema.rules, &mut config, &trigger_keys);
    }
    config
}

/// A single validation issue found by [`validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// The property `key` that failed.
    pub key: String,
    /// Human-readable message.
    pub message: String,
    pub severity: IssueSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Renderer surfaces with an error glyph; host should treat as
    /// invalid.
    Error,
    /// Renderer surfaces with a warning glyph; node is still
    /// considered usable.
    Warning,
}

/// Walk a schema against a current config and surface issues:
/// missing required values, out-of-range numbers, select values
/// outside the declared option list, duplicate keys in the schema
/// itself. The inspector renders error chips next to offending
/// fields; hosts can also block "save" on `Error`-level issues.
pub fn validate(schema: &ConfigSchema, config: &Value) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let object = config.as_object();

    // Duplicate-key detection. Quadratic but schemas are tiny in
    // practice (the largest Zeal templates ship ~20 properties); a
    // HashSet would barely shave runtime.
    for (i, prop) in schema.properties.iter().enumerate() {
        let key = &prop.meta().key;
        if schema.properties[..i].iter().any(|p| &p.meta().key == key) {
            issues.push(ValidationIssue {
                key: key.clone(),
                message: format!("duplicate property key `{key}` in schema"),
                severity: IssueSeverity::Error,
            });
        }
    }

    for prop in &schema.properties {
        let meta = prop.meta();
        let current = object.and_then(|o| o.get(&meta.key));

        // required + empty?
        if meta.required {
            let empty = match current {
                None => true,
                Some(Value::Null) => true,
                Some(Value::String(s)) => s.is_empty(),
                Some(Value::Array(a)) => a.is_empty(),
                _ => false,
            };
            if empty {
                issues.push(ValidationIssue {
                    key: meta.key.clone(),
                    message: format!("`{}` is required", meta.label),
                    severity: IssueSeverity::Error,
                });
                continue;
            }
        }

        match (prop, current) {
            (PropertyDefinition::Number(n), Some(value)) => {
                if let Some(num) = value.as_f64() {
                    if n.integer && num.fract() != 0.0 {
                        issues.push(ValidationIssue {
                            key: meta.key.clone(),
                            message: format!("`{}` must be an integer", meta.label),
                            severity: IssueSeverity::Error,
                        });
                    }
                    if let Some(min) = n.min {
                        if num < min {
                            issues.push(ValidationIssue {
                                key: meta.key.clone(),
                                message: format!("`{}` < {} (min)", meta.label, min),
                                severity: IssueSeverity::Error,
                            });
                        }
                    }
                    if let Some(max) = n.max {
                        if num > max {
                            issues.push(ValidationIssue {
                                key: meta.key.clone(),
                                message: format!("`{}` > {} (max)", meta.label, max),
                                severity: IssueSeverity::Error,
                            });
                        }
                    }
                } else {
                    issues.push(ValidationIssue {
                        key: meta.key.clone(),
                        message: format!("`{}` must be a number", meta.label),
                        severity: IssueSeverity::Error,
                    });
                }
            }
            (PropertyDefinition::Select(s), Some(value)) if !s.options.is_empty() => {
                if s.multiple {
                    if let Some(arr) = value.as_array() {
                        for v in arr {
                            if let Some(text) = v.as_str() {
                                if !s.options.iter().any(|o| o.value == text) {
                                    issues.push(ValidationIssue {
                                        key: meta.key.clone(),
                                        message: format!(
                                            "`{}` contains unknown value `{}`",
                                            meta.label, text
                                        ),
                                        severity: IssueSeverity::Error,
                                    });
                                }
                            }
                        }
                    }
                } else if let Some(text) = value.as_str() {
                    if !s.options.iter().any(|o| o.value == text) {
                        issues.push(ValidationIssue {
                            key: meta.key.clone(),
                            message: format!(
                                "`{}` has unknown value `{}`",
                                meta.label, text
                            ),
                            severity: IssueSeverity::Error,
                        });
                    }
                }
            }
            (PropertyDefinition::Text(t), Some(Value::String(s))) => {
                if let Some(max) = t.max_length {
                    if s.chars().count() > max {
                        issues.push(ValidationIssue {
                            key: meta.key.clone(),
                            message: format!(
                                "`{}` exceeds max length {} characters",
                                meta.label, max
                            ),
                            severity: IssueSeverity::Error,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    issues
}

// ─────────────────────────────────────────────────────────────────────
// Rules engine — declarative reactive cascades
// ─────────────────────────────────────────────────────────────────────

/// Predicate over a config object. Used by [`PropertyRule::when`]
/// to gate effect application. JSON-value comparisons use
/// `serde_json::Value::eq`; numeric comparators coerce via
/// `Value::as_f64` and silently fail (return `false`) on non-numeric
/// values.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Always matches. Pair with explicit triggers when the rule
    /// should fire on every cascade pass without a value test.
    Always,
    /// `config[key] == value`.
    Eq { key: String, value: Value },
    /// `config[key] != value`.
    NotEq { key: String, value: Value },
    /// `config[key]` is in `values`.
    In { key: String, values: Vec<Value> },
    /// JSON-truthy: present, not null, not `false`, not `""`, not
    /// `0`, not empty array, not empty object.
    Truthy(String),
    /// Key exists in the config object (even when its value is
    /// `null`). Distinct from `Truthy` — useful for "user has
    /// touched this field" semantics.
    Exists(String),
    Gt { key: String, value: f64 },
    Lt { key: String, value: f64 },
    Gte { key: String, value: f64 },
    Lte { key: String, value: f64 },
    /// All sub-predicates match. Empty `Vec` returns `true`.
    All(Vec<Predicate>),
    /// Any sub-predicate matches. Empty `Vec` returns `false`.
    Any(Vec<Predicate>),
    /// Negation.
    Not(Box<Predicate>),
}

impl Predicate {
    /// Evaluate against a config value. Non-object configs match
    /// only [`Predicate::Always`] / `Not(Always)` / nested boolean
    /// compositions — every key-based predicate returns `false`.
    pub fn evaluate(&self, config: &Value) -> bool {
        let obj = config.as_object();
        match self {
            Self::Always => true,
            Self::Eq { key, value } => obj
                .and_then(|o| o.get(key))
                .map(|v| values_eq(v, value))
                .unwrap_or(false),
            Self::NotEq { key, value } => match obj.and_then(|o| o.get(key)) {
                Some(v) => !values_eq(v, value),
                // Missing keys are NotEq to any concrete value —
                // matches the symmetry with Eq, which is `false` on
                // a missing key (so NotEq is `true`).
                None => true,
            },
            Self::In { key, values } => obj
                .and_then(|o| o.get(key))
                .map(|v| values.iter().any(|x| values_eq(x, v)))
                .unwrap_or(false),
            Self::Truthy(key) => obj
                .and_then(|o| o.get(key))
                .map(is_json_truthy)
                .unwrap_or(false),
            Self::Exists(key) => obj.map(|o| o.contains_key(key)).unwrap_or(false),
            Self::Gt { key, value } => obj
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .map(|n| n > *value)
                .unwrap_or(false),
            Self::Lt { key, value } => obj
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .map(|n| n < *value)
                .unwrap_or(false),
            Self::Gte { key, value } => obj
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .map(|n| n >= *value)
                .unwrap_or(false),
            Self::Lte { key, value } => obj
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .map(|n| n <= *value)
                .unwrap_or(false),
            Self::All(preds) => preds.iter().all(|p| p.evaluate(config)),
            Self::Any(preds) => preds.iter().any(|p| p.evaluate(config)),
            Self::Not(inner) => !inner.evaluate(config),
        }
    }
}

/// JSON value equality with numeric-bridge: two `Value::Number`s
/// compare via `as_f64` so integer-encoded values (`Number::from(3)`)
/// match float-encoded ones (`Number::from_f64(3.0)`). Without this
/// bridge `Predicate::Eq` silently never matched a default-seeded
/// `NumberProperty` (defaults round-trip through `from_f64`) against
/// a host-supplied `json!(3)`. All other variants fall back to
/// `Value::eq` (structural equality).
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => match (na.as_f64(), nb.as_f64()) {
            (Some(fa), Some(fb)) => fa == fb,
            _ => na == nb,
        },
        _ => a == b,
    }
}

fn is_json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A single config mutation that a [`PropertyRule`] applies when it
/// fires.
#[derive(Debug, Clone)]
pub enum PropertyEffect {
    /// Write `value` at `key`. Overwrites any existing entry.
    Set { key: String, value: Value },
    /// Remove `key` from the config object.
    Clear(String),
}

impl PropertyEffect {
    pub fn key(&self) -> &str {
        match self {
            Self::Set { key, .. } => key,
            Self::Clear(key) => key,
        }
    }
}

/// One reactive rule. Fires when any key in [`triggers`](Self::triggers)
/// changes (or on every cascade pass when `triggers` is empty) AND
/// the [`when`](Self::when) predicate matches the post-change
/// config. Each firing applies every [`PropertyEffect`] in
/// [`effects`](Self::effects); effects that don't actually change
/// the config are no-ops and don't re-trigger the cascade.
#[derive(Debug, Clone)]
pub struct PropertyRule {
    /// Property keys that must appear in the cascade's current
    /// trigger frontier for this rule to evaluate. Empty means
    /// "evaluate on every cascade pass where the rule was reached
    /// by some other rule's effect" — see [`cascade_rules`] for the
    /// frontier semantics.
    ///
    /// **Caveat**: a rule with empty `triggers` does NOT fire on
    /// passes where the frontier itself is empty (which is the
    /// initial state when `cascade_rules` is called with an empty
    /// `initial_changes`). Pair empty triggers with a primary patch
    /// that seeds the frontier; for "fire once on schema init" use
    /// [`default_config`] which seeds every property key.
    pub triggers: Vec<String>,
    pub when: Predicate,
    pub effects: Vec<PropertyEffect>,
}

impl PropertyRule {
    /// Build a rule that fires on any config change (no triggers),
    /// matches every config (`Predicate::Always`), and applies no
    /// effects. Chain the builders below to populate it.
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
            when: Predicate::Always,
            effects: Vec::new(),
        }
    }

    /// Append a trigger key. The rule re-evaluates whenever a
    /// listed key changes.
    pub fn trigger(mut self, key: impl Into<String>) -> Self {
        self.triggers.push(key.into());
        self
    }

    /// Convenience for multiple triggers at once.
    pub fn triggers<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.triggers.extend(keys.into_iter().map(Into::into));
        self
    }

    pub fn when(mut self, predicate: Predicate) -> Self {
        self.when = predicate;
        self
    }

    pub fn set(mut self, key: impl Into<String>, value: Value) -> Self {
        self.effects.push(PropertyEffect::Set {
            key: key.into(),
            value,
        });
        self
    }

    pub fn clear(mut self, key: impl Into<String>) -> Self {
        self.effects.push(PropertyEffect::Clear(key.into()));
        self
    }

    pub fn effect(mut self, effect: PropertyEffect) -> Self {
        self.effects.push(effect);
        self
    }
}

impl Default for PropertyRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum cascade iterations before [`cascade_rules`] bails out
/// with a warning. Protects against cyclic rules (`A->B`, `B->A`)
/// that would otherwise loop forever.
pub const MAX_RULE_CASCADE_DEPTH: usize = 16;

/// Apply `rules` to `config`, cascading until either no rule fires
/// or [`MAX_RULE_CASCADE_DEPTH`] is reached. `initial_changes`
/// seeds the first iteration's trigger frontier — pass the keys
/// the primary patch wrote.
///
/// Returns the effects that actually changed the config (Set
/// operations whose value differed from the prior value, Clear
/// operations that removed an existing entry). The caller emits
/// per-effect notifications from this list.
///
/// ## Ordering guarantee
///
/// Within a single cascade pass, rules evaluate in
/// `schema.rules` declaration order. Effects from earlier rules
/// are visible to later rules in the same pass via the live
/// `config` argument — hosts may declare a rule that reads a
/// value just written by an earlier rule.
///
/// ## Termination
///
/// The cascade terminates naturally when a pass produces an empty
/// next-frontier (no effect actually changed a value). The
/// `apply_effect` no-op short-circuit at the bottom of this file
/// means most ping-pong rule pairs self-terminate after one
/// round-trip — e.g. `A.Set(B=1), B.Set(A=1)` settles in two
/// passes because the second pass's effects all become no-ops.
///
/// A cascade that genuinely changes its targets on every pass —
/// for instance `Rule A: trigger b, when Exists(b), Clear(b)` +
/// `Rule B: trigger b, when !Exists(b), Set(b, …)` — ping-pongs
/// indefinitely. `cascade_rules` caps at [`MAX_RULE_CASCADE_DEPTH`]
/// iterations and logs a `tracing::warn!` if it bails, returning
/// whatever effects landed before the cap.
///
/// ## Initial changes
///
/// When `initial_changes` is empty the cascade returns
/// immediately with no effects — rules that have empty `triggers`
/// fire only when SOMETHING is in the frontier, not on the
/// initial pass. To force a full re-evaluation (e.g. on schema
/// init) seed the trigger set with every property key — see
/// [`default_config`] for the reference pattern.
pub fn cascade_rules(
    rules: &[PropertyRule],
    config: &mut Value,
    initial_changes: &[String],
) -> Vec<PropertyEffect> {
    if rules.is_empty() {
        return Vec::new();
    }
    let mut applied: Vec<PropertyEffect> = Vec::new();
    let mut frontier: Vec<String> = initial_changes.to_vec();
    let mut depth = 0;
    while !frontier.is_empty() && depth < MAX_RULE_CASCADE_DEPTH {
        let mut next_frontier: Vec<String> = Vec::new();
        for rule in rules {
            let triggered = rule.triggers.is_empty()
                || rule.triggers.iter().any(|t| frontier.contains(t));
            if !triggered {
                continue;
            }
            if !rule.when.evaluate(config) {
                continue;
            }
            for effect in &rule.effects {
                if apply_effect(config, effect) {
                    next_frontier.push(effect.key().to_string());
                    applied.push(effect.clone());
                }
            }
        }
        frontier = next_frontier;
        depth += 1;
    }
    if depth == MAX_RULE_CASCADE_DEPTH && !frontier.is_empty() {
        tracing::warn!(
            target: "blinc_node_editor::config",
            depth = MAX_RULE_CASCADE_DEPTH,
            "cascade_rules: hit MAX_RULE_CASCADE_DEPTH — likely a rule cycle"
        );
    }
    applied
}

/// Apply a single effect to a config in place. Returns `true` when
/// the config was actually mutated (skipped when a `Set` overwrites
/// with the same value, or `Clear` removes a non-existent key).
fn apply_effect(config: &mut Value, effect: &PropertyEffect) -> bool {
    if config.is_null() {
        *config = Value::Object(Map::new());
    }
    let Some(obj) = config.as_object_mut() else {
        return false;
    };
    match effect {
        PropertyEffect::Set { key, value } => {
            if obj.get(key) == Some(value) {
                return false;
            }
            obj.insert(key.clone(), value.clone());
            true
        }
        PropertyEffect::Clear(key) => obj.remove(key).is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(props: Vec<PropertyDefinition>) -> ConfigSchema {
        ConfigSchema::from(props)
    }

    #[test]
    fn defaults_populate_object() {
        let s = schema(vec![
            TextProperty::new("name", "Name").default("alice").into(),
            NumberProperty::new("count", "Count").default(3.0).into(),
            BooleanProperty::new("on", "On").default(true).into(),
            // No default → omitted.
            TextProperty::new("untouched", "Untouched").into(),
        ]);

        let config = default_config(&s);
        let obj = config.as_object().unwrap();
        assert_eq!(obj.get("name"), Some(&Value::String("alice".into())));
        assert_eq!(obj.get("count").and_then(|v| v.as_f64()), Some(3.0));
        assert_eq!(obj.get("on"), Some(&Value::Bool(true)));
        assert!(!obj.contains_key("untouched"));
    }

    #[test]
    fn required_field_with_no_value_is_error() {
        let s = schema(vec![TextProperty::new("name", "Name").required(true).into()]);

        let issues = validate(&s, &Value::Object(Map::new()));
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert_eq!(issues[0].key, "name");
    }

    #[test]
    fn number_range_validation() {
        let s = schema(vec![NumberProperty::new("threshold", "Threshold")
            .range(0.0, 1.0)
            .into()]);

        let too_low = serde_json::json!({ "threshold": -0.5 });
        let in_range = serde_json::json!({ "threshold": 0.4 });
        let too_high = serde_json::json!({ "threshold": 1.5 });

        assert_eq!(validate(&s, &too_low).len(), 1);
        assert!(validate(&s, &in_range).is_empty());
        assert_eq!(validate(&s, &too_high).len(), 1);
    }

    #[test]
    fn select_rejects_unknown_value() {
        let s = schema(vec![SelectProperty::new("mode", "Mode")
            .option("warn", "Warn")
            .option("error", "Error")
            .into()]);

        let unknown = serde_json::json!({ "mode": "panic" });
        let issues = validate(&s, &unknown);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
    }

    #[test]
    fn duplicate_keys_flag_error() {
        let s = schema(vec![
            TextProperty::new("name", "Name").into(),
            TextProperty::new("name", "Different label").into(),
        ]);

        let issues = validate(&s, &Value::Object(Map::new()));
        assert!(
            issues.iter().any(|i| i.message.contains("duplicate")),
            "expected duplicate-key error, got {issues:?}"
        );
    }

    #[test]
    fn integer_property_rejects_fractions() {
        let s = schema(vec![NumberProperty::new("count", "Count").integer().into()]);

        let frac = serde_json::json!({ "count": 3.5 });
        let issues = validate(&s, &frac);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].message.contains("integer"));
    }

    #[test]
    fn text_max_length_enforced() {
        let s = schema(vec![TextProperty::new("name", "Name").max_length(5).into()]);

        let over = serde_json::json!({ "name": "abcdefg" });
        let issues = validate(&s, &over);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].message.contains("max length"));
    }

    #[test]
    fn empty_schema_is_valid() {
        let s = ConfigSchema::new();
        assert!(validate(&s, &Value::Null).is_empty());
        assert_eq!(default_config(&s), Value::Object(Map::new()));
    }

    // ─── Rules engine ─────────────────────────────────────────────

    #[test]
    fn predicate_basics() {
        let cfg = serde_json::json!({ "mode": "lenient", "n": 5 });
        assert!(Predicate::Eq {
            key: "mode".into(),
            value: serde_json::json!("lenient")
        }
        .evaluate(&cfg));
        assert!(!Predicate::Eq {
            key: "mode".into(),
            value: serde_json::json!("strict")
        }
        .evaluate(&cfg));
        assert!(Predicate::Truthy("mode".into()).evaluate(&cfg));
        assert!(Predicate::Gt {
            key: "n".into(),
            value: 4.0
        }
        .evaluate(&cfg));
        assert!(!Predicate::Gt {
            key: "n".into(),
            value: 5.0
        }
        .evaluate(&cfg));
        assert!(Predicate::Gte {
            key: "n".into(),
            value: 5.0
        }
        .evaluate(&cfg));
        assert!(Predicate::Exists("mode".into()).evaluate(&cfg));
        assert!(!Predicate::Exists("missing".into()).evaluate(&cfg));
    }

    #[test]
    fn predicate_eq_bridges_integer_and_float_numbers() {
        // Default-seeded NumberProperty rounds through
        // `Number::from_f64` → always a Float. A host emitting
        // `json!(3)` produces an Integer. Without the numeric
        // bridge in `values_eq`, Value::PartialEq would see these
        // as distinct and the predicate would silently never
        // match.
        let int_cfg = serde_json::json!({ "n": 3 });
        let float_cfg = serde_json::json!({ "n": 3.0 });
        let eq_int = Predicate::Eq {
            key: "n".into(),
            value: serde_json::json!(3),
        };
        let eq_float = Predicate::Eq {
            key: "n".into(),
            value: serde_json::json!(3.0),
        };
        assert!(eq_int.evaluate(&int_cfg));
        assert!(eq_int.evaluate(&float_cfg), "int predicate matches float value");
        assert!(eq_float.evaluate(&int_cfg), "float predicate matches int value");
        assert!(eq_float.evaluate(&float_cfg));

        // NotEq follows the same bridge.
        let ne = Predicate::NotEq {
            key: "n".into(),
            value: serde_json::json!(3),
        };
        assert!(!ne.evaluate(&float_cfg));

        // In tolerates int/float mix in both haystack and needle.
        let in_pred = Predicate::In {
            key: "n".into(),
            values: vec![serde_json::json!(1.0), serde_json::json!(3)],
        };
        assert!(in_pred.evaluate(&int_cfg));
        assert!(in_pred.evaluate(&float_cfg));
    }

    #[test]
    fn predicate_noteq_on_missing_key_returns_true() {
        // Symmetric with Predicate::Eq on a missing key returning
        // false: NotEq on a missing key must return true.
        let cfg = serde_json::json!({});
        let ne = Predicate::NotEq {
            key: "absent".into(),
            value: serde_json::json!("anything"),
        };
        assert!(ne.evaluate(&cfg));
    }

    #[test]
    fn predicate_composition() {
        let cfg = serde_json::json!({ "mode": "strict", "n": 3 });
        let pred = Predicate::All(vec![
            Predicate::Eq {
                key: "mode".into(),
                value: serde_json::json!("strict"),
            },
            Predicate::Gt {
                key: "n".into(),
                value: 1.0,
            },
        ]);
        assert!(pred.evaluate(&cfg));

        let pred_or = Predicate::Any(vec![
            Predicate::Eq {
                key: "mode".into(),
                value: serde_json::json!("loose"),
            },
            Predicate::Lt {
                key: "n".into(),
                value: 10.0,
            },
        ]);
        assert!(pred_or.evaluate(&cfg));

        let pred_not = Predicate::Not(Box::new(Predicate::Eq {
            key: "mode".into(),
            value: serde_json::json!("strict"),
        }));
        assert!(!pred_not.evaluate(&cfg));
    }

    #[test]
    fn cascade_applies_single_rule() {
        let rules = vec![PropertyRule::new()
            .trigger("mode")
            .when(Predicate::Eq {
                key: "mode".into(),
                value: serde_json::json!("lenient"),
            })
            .set("threshold", serde_json::json!(0.2))];

        let mut config = serde_json::json!({ "mode": "lenient", "threshold": 0.5 });
        let applied = cascade_rules(&rules, &mut config, &["mode".into()]);
        assert_eq!(applied.len(), 1);
        assert_eq!(
            config.get("threshold").and_then(|v| v.as_f64()),
            Some(0.2)
        );
    }

    #[test]
    fn cascade_chains_through_multiple_rules() {
        // A -> B -> C
        let rules = vec![
            PropertyRule::new()
                .trigger("a")
                .when(Predicate::Truthy("a".into()))
                .set("b", serde_json::json!(true)),
            PropertyRule::new()
                .trigger("b")
                .when(Predicate::Truthy("b".into()))
                .set("c", serde_json::json!("derived")),
        ];

        let mut config = serde_json::json!({ "a": true });
        let applied = cascade_rules(&rules, &mut config, &["a".into()]);
        assert_eq!(applied.len(), 2);
        assert_eq!(config.get("b"), Some(&Value::Bool(true)));
        assert_eq!(
            config.get("c"),
            Some(&Value::String("derived".into()))
        );
    }

    #[test]
    fn cascade_predicate_failure_skips_rule() {
        let rules = vec![PropertyRule::new()
            .trigger("mode")
            .when(Predicate::Eq {
                key: "mode".into(),
                value: serde_json::json!("lenient"),
            })
            .set("threshold", serde_json::json!(0.2))];

        let mut config = serde_json::json!({ "mode": "strict", "threshold": 0.5 });
        let applied = cascade_rules(&rules, &mut config, &["mode".into()]);
        assert!(applied.is_empty(), "predicate didn't match");
        assert_eq!(
            config.get("threshold").and_then(|v| v.as_f64()),
            Some(0.5),
            "threshold untouched"
        );
    }

    #[test]
    fn cascade_no_op_set_is_not_reported() {
        // Effect would set value to its existing value — no change,
        // no entry in the applied list, no re-trigger.
        let rules = vec![PropertyRule::new()
            .trigger("a")
            .when(Predicate::Always)
            .set("b", serde_json::json!(1))];

        let mut config = serde_json::json!({ "a": 1, "b": 1 });
        let applied = cascade_rules(&rules, &mut config, &["a".into()]);
        assert!(applied.is_empty());
    }

    #[test]
    fn cascade_cycle_terminates_at_max_depth() {
        // Two rules that flip each other forever.
        let rules = vec![
            PropertyRule::new()
                .trigger("a")
                .when(Predicate::Always)
                .set("b", serde_json::json!("from-a")),
            PropertyRule::new()
                .trigger("b")
                .when(Predicate::Always)
                .set("a", serde_json::json!("from-b")),
        ];

        let mut config = serde_json::json!({ "a": "seed", "b": "seed" });
        // Doesn't loop forever — capped by MAX_RULE_CASCADE_DEPTH.
        let applied = cascade_rules(&rules, &mut config, &["a".into()]);
        // We can't predict exact count due to alternating triggers
        // but it should NOT exceed 2 effects per depth × 16 depths.
        assert!(
            applied.len() <= 32,
            "cascade hit cap, got {} applied effects",
            applied.len()
        );
    }

    #[test]
    fn cascade_clear_removes_key() {
        let rules = vec![PropertyRule::new()
            .trigger("mode")
            .when(Predicate::Eq {
                key: "mode".into(),
                value: serde_json::json!("clean"),
            })
            .clear("scratch")];

        let mut config = serde_json::json!({ "mode": "clean", "scratch": "old" });
        let applied = cascade_rules(&rules, &mut config, &["mode".into()]);
        assert_eq!(applied.len(), 1);
        assert!(!config.as_object().unwrap().contains_key("scratch"));
    }

    #[test]
    fn default_config_runs_rules_once() {
        // Rule seeds threshold=0.2 when mode default is "lenient".
        let s = ConfigSchema::new()
            .with_property(
                SelectProperty::new("mode", "Mode")
                    .option("strict", "Strict")
                    .option("lenient", "Lenient")
                    .default("lenient"),
            )
            .with_property(NumberProperty::new("threshold", "Threshold").default(0.5))
            .with_rule(
                PropertyRule::new()
                    .trigger("mode")
                    .when(Predicate::Eq {
                        key: "mode".into(),
                        value: serde_json::json!("lenient"),
                    })
                    .set("threshold", serde_json::json!(0.2)),
            );

        let config = default_config(&s);
        assert_eq!(
            config.get("threshold").and_then(|v| v.as_f64()),
            Some(0.2),
            "rule should override the default during default_config"
        );
    }
}
