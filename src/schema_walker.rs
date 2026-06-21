//! Schema → Portal-UI walker (Tier 11).
//!
//! Turns a [`ConfigSchema`](crate::config::ConfigSchema) intermediate
//! representation into a live, two-way-bound Portal-UI form rendered
//! inside a node's content slot. The host describes fields as data
//! (`PropertyDefinition` variants); the walker composes the matching
//! Portal widget for each, binds it through a [`FieldAccess`] bridge,
//! and writes edits back so the host can persist + cascade them.
//!
//! Immediate mode: [`walk_schema`] runs every frame inside the content
//! closure. Each field's value is reseeded from [`FieldAccess::get`]
//! each frame, so external mutations (rule cascades, other clients)
//! show up on the next paint with no reconciliation. Widget identity is
//! kept stable across frames via [`PortalUi::push_id`] keyed on the
//! field key, so caret / drag state survives even if the field list
//! reorders.
//!
//! ```ignore
//! NodeTemplate::new("config", "Config")
//!     .with_config_schema(schema.clone())
//!     .with_content(140.0, move |id, ui| {
//!         let mut access = EditorFieldAccess::new(editor.clone(), id.clone());
//!         walk_schema(ui, &schema, &mut access, &WalkOptions::default());
//!     });
//! ```

use serde_json::Value;

use crate::config::{ConfigSchema, PropertyDefinition};
use crate::node::NodeId;

use blinc_portal_ui::PortalUi;

/// Two-way bridge between a config field (`key` → JSON `Value`) and the
/// host's source of truth. The walker never decides where field data
/// lives; the host wires that through this trait.
///
/// [`EditorFieldAccess`] is the turnkey implementation backed by a
/// [`NodeEditor`](crate::NodeEditor): `get` reads the node's `config`,
/// `set` routes through `patch_node_config` (write + rule cascade +
/// `NodeConfigChanged` event).
pub trait FieldAccess {
    /// Current value for `key`, or [`Value::Null`] when unset.
    fn get(&self, key: &str) -> Value;
    /// The user edited `key` this frame. The host persists it (and may
    /// run rule cascades / emit an event).
    fn set(&mut self, key: &str, value: Value);

    /// The user clicked a select field's trigger. Implementors open a
    /// picker (the editor opens a screen-space overlay menu) listing
    /// `options` (`(value, label)` pairs), anchored to `anchor_content`
    /// (the trigger's rect in canvas-content space), and write the
    /// chosen value back via [`Self::set`]. Default: no-op (the walker
    /// then shows the trigger but the picker never opens).
    fn open_select(
        &mut self,
        _key: &str,
        _options: &[(String, String)],
        _current: &str,
        _anchor_content: blinc_core::layer::Rect,
    ) {
    }
}

/// Layout / behaviour knobs for [`walk_schema`].
#[derive(Debug, Clone)]
pub struct WalkOptions {
    /// Vertical gap (px) inserted between fields.
    pub field_gap: f32,
    /// Render a `*` suffix on labels of `required` properties.
    pub mark_required: bool,
    /// Render [`crate::config::validate`] issues inline under each
    /// field (error / warning glyph + message).
    pub show_validation: bool,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            field_gap: 8.0,
            mark_required: true,
            show_validation: true,
        }
    }
}

/// Walk `schema` and emit one Portal widget per property, bound through
/// `access`. Returns the keys the user mutated this frame (already
/// pushed through `access.set`).
///
/// Flat path: every property renders as a label + widget, stacked
/// vertically (the content closure's default layout direction). Nested
/// layout (sections / rows / conditional groups) lands with
/// `ContentSchema` in a later phase.
pub fn walk_schema(
    ui: &mut PortalUi,
    schema: &ConfigSchema,
    access: &mut dyn FieldAccess,
    opts: &WalkOptions,
) -> Vec<String> {
    // Effective-value snapshot: the current value per key, falling back
    // to the property default when unset. Used both as the widget seed
    // and for validation, so a field with a default isn't falsely
    // flagged "required" before the host seeds `config`.
    let config = effective_config(schema, access);
    let issues = if opts.show_validation {
        crate::config::validate(schema, &config)
    } else {
        Vec::new()
    };

    let mut changed: Vec<String> = Vec::new();
    for (i, prop) in schema.properties.iter().enumerate() {
        if i > 0 {
            ui.spacing(opts.field_gap);
        }
        let key = prop.key().to_string();
        let field_issues: Vec<&crate::config::ValidationIssue> =
            issues.iter().filter(|iss| iss.key == key).collect();
        // `push_id` keeps each field's WidgetId stable across frames so
        // text caret / slider drag state in PortalStorage stays correct
        // even as the field list changes.
        ui.push_id(key.as_str(), |ui| {
            if emit_field(ui, prop, access, &config, false, &field_issues, opts) {
                changed.push(key.clone());
            }
        });
    }
    changed
}

/// Build the effective config object: `access.get(key)` per property,
/// or the property's default when unset. Omits keys with neither.
pub(crate) fn effective_config(schema: &ConfigSchema, access: &dyn FieldAccess) -> Value {
    let mut obj = serde_json::Map::new();
    for prop in &schema.properties {
        let key = prop.key();
        let mut v = access.get(key);
        if v.is_null() {
            if let Some(def) = prop.default_value() {
                v = def;
            }
        }
        if !v.is_null() {
            obj.insert(key.to_string(), v);
        }
    }
    Value::Object(obj)
}

/// Emit the label + widget for one property. Returns `true` if the user
/// edited it this frame (the new value has already been written through
/// `access`).
pub(crate) fn emit_field(
    ui: &mut PortalUi,
    prop: &PropertyDefinition,
    access: &mut dyn FieldAccess,
    config: &Value,
    disabled: bool,
    issues: &[&crate::config::ValidationIssue],
    opts: &WalkOptions,
) -> bool {
    let meta = prop.meta();
    let key = meta.key.as_str();

    // Label line.
    let label = if opts.mark_required && meta.required {
        format!("{} *", meta.label)
    } else {
        meta.label.clone()
    };
    ui.label(&label);

    let cur = config.get(key).cloned().unwrap_or(Value::Null);
    let mut edited = false;

    match prop {
        PropertyDefinition::Text(p) => {
            let mut s = as_string(&cur, p.default.as_deref());
            let mut b = ui.text_input(&mut s).disabled(disabled);
            if let Some(ph) = &p.placeholder {
                b = b.placeholder(ph.clone());
            }
            if b.show().changed {
                access.set(key, Value::String(s));
                edited = true;
            }
        }
        PropertyDefinition::Textarea(p) => {
            let mut s = as_string(&cur, p.default.as_deref());
            let mut b = ui
                .textarea(&mut s)
                .rows(p.rows.unwrap_or(3) as usize)
                .disabled(disabled);
            if let Some(ph) = &p.placeholder {
                b = b.placeholder(ph.clone());
            }
            if b.show().changed {
                access.set(key, Value::String(s));
                edited = true;
            }
        }
        PropertyDefinition::CodeEditor(p) => {
            // No syntax highlighting in portal_ui yet — a multi-line
            // textarea is the honest surface for now.
            let mut s = as_string(&cur, p.default.as_deref());
            if ui.textarea(&mut s).rows(6).disabled(disabled).show().changed {
                access.set(key, Value::String(s));
                edited = true;
            }
        }
        PropertyDefinition::Number(p) => {
            let mut f = as_f32(&cur, p.default);
            let has_range = p.min.is_some() && p.max.is_some();
            let changed = if has_range {
                let lo = p.min.unwrap() as f32;
                let hi = p.max.unwrap() as f32;
                let mut b = ui.slider(&mut f, lo..hi).disabled(disabled);
                if let Some(step) = p.step {
                    b = b.step(step as f32);
                }
                b.show().changed
            } else {
                let mut b = ui.numeric_input(&mut f).disabled(disabled);
                if let Some(step) = p.step {
                    b = b.step(step as f32);
                }
                if p.integer {
                    b = b.integer();
                }
                b.show().changed
            };
            if changed {
                let out = if p.integer { f.round() as f64 } else { f as f64 };
                access.set(key, number(out));
                edited = true;
            }
        }
        PropertyDefinition::Boolean(p) => {
            let mut b = as_bool(&cur, p.default);
            if ui.switch(&mut b).disabled(disabled).show().changed {
                access.set(key, Value::Bool(b));
                edited = true;
            }
        }
        PropertyDefinition::Select(p) => {
            // The portal `select` is a trigger; the options menu is a
            // screen-space overlay the editor opens (NOT a canvas draw).
            // On click we hand the options + the trigger's rect to
            // `open_select`; `EditorFieldAccess` defers the overlay open
            // to the next frame and writes the chosen value back via the
            // normal patch path.
            let cur_v = as_string(&cur, p.default.as_deref());
            let pairs: Vec<(String, String)> = p
                .options
                .iter()
                .map(|o| (o.value.clone(), o.label.clone()))
                .collect();
            let resp = ui.select(&cur_v, &pairs).disabled(disabled).show();
            if resp.clicked && !disabled && !p.options.is_empty() {
                access.open_select(key, &pairs, &cur_v, resp.rect);
            }
        }
        PropertyDefinition::Color(p) => {
            let mut hex = as_string(&cur, p.default.as_deref());
            if ui.color_picker(&mut hex).disabled(disabled).show().changed {
                access.set(key, Value::String(hex));
                edited = true;
            }
        }
        PropertyDefinition::File(p) => {
            // Path text field. A native file dialog is host-owned (a
            // CustomSlot can wire one); for now the path is editable.
            let mut path = as_string(&cur, p.default.as_deref());
            if ui.text_input(&mut path).disabled(disabled).show().changed {
                access.set(key, Value::String(path));
                edited = true;
            }
        }
        PropertyDefinition::Custom { .. } => {
            // Host-rendered escape hatch — the walker emits nothing for
            // the value (the host draws its own widget via a CustomSlot).
            // Label already shown above.
        }
    }

    // Inline validation: one line per issue under the field. `label`
    // has no colour channel, so severity is conveyed by a glyph prefix.
    if opts.show_validation {
        for iss in issues {
            let glyph = match iss.severity {
                crate::config::IssueSeverity::Error => "⚠",
                crate::config::IssueSeverity::Warning => "•",
            };
            ui.label(&format!("{glyph} {}", iss.message));
        }
    }

    edited
}

// ── JSON ⇆ scalar coercion ─────────────────────────────────────────

fn as_string(v: &Value, default: Option<&str>) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => default.unwrap_or("").to_string(),
        other => other.to_string(),
    }
}

fn as_f32(v: &Value, default: Option<f64>) -> f32 {
    v.as_f64().or(default).unwrap_or(0.0) as f32
}

fn as_bool(v: &Value, default: Option<bool>) -> bool {
    v.as_bool().or(default).unwrap_or(false)
}

fn number(n: f64) -> Value {
    serde_json::Number::from_f64(n)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

// ── Turnkey editor-backed FieldAccess ──────────────────────────────

/// [`FieldAccess`] backed by a [`NodeEditor`](crate::NodeEditor): reads
/// the node's `config` and writes through `patch_node_config` (so edits
/// run the rule cascade and emit `NodeConfigChanged`). Holds a cloned
/// editor handle (the editor is `Clone` + interior-mutable), so it can
/// live inside the `'static` content closure.
pub struct EditorFieldAccess<K, N, C, G>
where
    K: crate::port::PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    editor: crate::NodeEditor<K, N, C, G>,
    node: NodeId,
}

impl<K, N, C, G> EditorFieldAccess<K, N, C, G>
where
    K: crate::port::PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    pub fn new(editor: crate::NodeEditor<K, N, C, G>, node: NodeId) -> Self {
        Self { editor, node }
    }
}

impl<K, N, C, G> FieldAccess for EditorFieldAccess<K, N, C, G>
where
    K: crate::port::PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    fn get(&self, key: &str) -> Value {
        self.editor
            .node_config(&self.node)
            .and_then(|cfg| cfg.get(key).cloned())
            .unwrap_or(Value::Null)
    }

    fn set(&mut self, key: &str, value: Value) {
        // Deferred: the walker runs inside the content closure during
        // `render_frame`, which holds the graph read lock. Applying the
        // patch now (graph.write) would deadlock; the editor drains the
        // queue at the top of the next frame. See `queue_config_patch`.
        self.editor.queue_config_patch(&self.node, key, value);
    }

    fn open_select(
        &mut self,
        key: &str,
        options: &[(String, String)],
        _current: &str,
        anchor_content: blinc_core::layer::Rect,
    ) {
        // Deferred for the same reason as `set`. The editor opens a
        // screen-space overlay menu at the top of the next frame.
        self.editor
            .queue_select_overlay(&self.node, key, options.to_vec(), anchor_content);
    }
}
