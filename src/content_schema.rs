//! `ContentSchema` — a thin layout + conditional tree layered over
//! [`ConfigSchema`](crate::config::ConfigSchema) for composable,
//! Zeal-parity node forms.
//!
//! [`ConfigSchema`] stays the flat *leaf vocabulary* (the typed fields).
//! `ContentSchema` adds the *structure*: sections, rows, conditional
//! groups, static labels — referencing properties by key rather than
//! redefining them. [`walk_content`] interprets the tree into Portal UI
//! every frame, reusing the same per-field emit + binding the flat
//! [`walk_schema`](crate::schema_walker::walk_schema) uses.
//!
//! Conditional visibility / enablement is purely a tree property,
//! evaluated each frame against the live config via
//! [`Predicate::evaluate`] — it never mutates the config object (that's
//! what value-cascade [`PropertyRule`](crate::config::PropertyRule)s
//! are for). `When` skips its children; `DisableWhen` renders them
//! disabled.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::config::{ConfigSchema, Predicate, ValidationIssue};
use crate::node::NodeId;
use crate::schema_walker::{effective_config, emit_field, FieldAccess, WalkOptions};

use blinc_portal_ui::PortalUi;

/// Host-rendered escape hatch: closures keyed by slot name, invoked by
/// [`ContentItem::CustomSlot`]. Lets a host compose its own Portal-UI
/// (charts, color wheels, bespoke editors) *inside* the schema tree
/// rather than in a separate hand-written closure.
pub type SlotRegistry = HashMap<String, Arc<dyn Fn(&NodeId, &mut PortalUi) + Send + Sync>>;

/// A structured node form: a tree of [`ContentItem`]s over a flat
/// [`ConfigSchema`] of typed properties.
#[derive(Debug, Clone)]
pub struct ContentSchema {
    pub root: Vec<ContentItem>,
    pub properties: ConfigSchema,
}

impl ContentSchema {
    /// New schema over `properties` with an empty layout tree.
    pub fn new(properties: ConfigSchema) -> Self {
        Self {
            root: Vec::new(),
            properties,
        }
    }

    /// Append a top-level item.
    pub fn with(mut self, item: ContentItem) -> Self {
        self.root.push(item);
        self
    }

    /// Append a top-level [`ContentItem::Field`] by key.
    pub fn field(self, key: impl Into<String>) -> Self {
        self.with(ContentItem::Field(key.into()))
    }
}

/// One node in the content tree. Containers nest; leaves reference a
/// schema property by key.
#[derive(Debug, Clone)]
pub enum ContentItem {
    /// Render one schema property by key (the 95% case).
    Field(String),
    /// Titled section: a label followed by indented children.
    Section {
        title: String,
        children: Vec<ContentItem>,
    },
    /// Lay children out left-to-right.
    Row(Vec<ContentItem>),
    /// Render `children` only while `when` holds (declarative show/hide).
    When {
        when: Predicate,
        children: Vec<ContentItem>,
    },
    /// Render `children` disabled while `when` holds (declarative enable).
    DisableWhen {
        when: Predicate,
        children: Vec<ContentItem>,
    },
    /// Repeat an editable text row per element of the string array at
    /// `config[key]`, plus an "add" button. The minimal dynamic-list
    /// primitive; subform-per-element repeaters (path-scoped binding)
    /// are a future extension.
    Repeater { key: String },
    /// Invoke the host closure registered under `slot` in the
    /// [`SlotRegistry`] — composes arbitrary host Portal-UI into the
    /// tree (charts, color wheels, etc.).
    CustomSlot(String),
    /// Static text line.
    Label(String),
    /// Vertical gap / visual break.
    Separator,
}

/// Walk a [`ContentSchema`] into Portal UI, binding each field through
/// `access`. Returns the keys the user mutated this frame.
pub fn walk_content(
    ui: &mut PortalUi,
    schema: &ContentSchema,
    access: &mut dyn FieldAccess,
    node: &NodeId,
    slots: &SlotRegistry,
    opts: &WalkOptions,
) -> Vec<String> {
    let config = effective_config(&schema.properties, &*access);
    let issues = if opts.show_validation {
        crate::config::validate(&schema.properties, &config)
    } else {
        Vec::new()
    };
    let mut ctx = WalkCtx {
        props: &schema.properties,
        config: &config,
        issues: &issues,
        node,
        slots,
        opts,
        changed: Vec::new(),
    };
    walk_items(ui, &schema.root, false, access, &mut ctx);
    ctx.changed
}

/// Per-walk context — the read-only schema/config/slots plus the
/// accumulating `changed` list. Bundled so the recursion signature
/// stays small.
struct WalkCtx<'a> {
    props: &'a ConfigSchema,
    config: &'a Value,
    issues: &'a [ValidationIssue],
    node: &'a NodeId,
    slots: &'a SlotRegistry,
    opts: &'a WalkOptions,
    changed: Vec<String>,
}

fn walk_items(
    ui: &mut PortalUi,
    items: &[ContentItem],
    disabled: bool,
    access: &mut dyn FieldAccess,
    ctx: &mut WalkCtx,
) {
    let props = ctx.props;
    let config = ctx.config;
    let issues = ctx.issues;
    let opts = ctx.opts;
    let node = ctx.node;
    let slots = ctx.slots;
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            ui.spacing(opts.field_gap);
        }
        match item {
            ContentItem::Field(key) => {
                if let Some(prop) = props.properties.iter().find(|p| p.key() == key) {
                    let field_issues: Vec<&ValidationIssue> =
                        issues.iter().filter(|x| &x.key == key).collect();
                    ui.push_id(key.as_str(), |ui| {
                        if emit_field(ui, prop, access, config, disabled, &field_issues, opts) {
                            ctx.changed.push(key.clone());
                        }
                    });
                }
            }
            ContentItem::Section { title, children } => {
                // Title, then fields FLUSH beneath it (no indent — a
                // form section is a header over left-aligned fields,
                // not a tree node).
                ui.label(title);
                ui.spacing(opts.field_gap * 0.5);
                walk_items(ui, children, disabled, access, ctx);
            }
            ContentItem::Row(children) => {
                ui.horizontal(|ui| {
                    walk_items(ui, children, disabled, access, ctx);
                });
            }
            ContentItem::When { when, children } => {
                if when.evaluate(config) {
                    walk_items(ui, children, disabled, access, ctx);
                }
            }
            ContentItem::DisableWhen { when, children } => {
                let d = disabled || when.evaluate(config);
                walk_items(ui, children, d, access, ctx);
            }
            ContentItem::Repeater { key } => {
                // Read the live array directly (a Repeater key need not
                // be a declared schema property, so it won't be in the
                // effective-config snapshot).
                let mut items_v: Vec<String> = access
                    .get(key)
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|v| v.as_str().unwrap_or("").to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                let mut mutated = false;
                // Deferred removal: a row's delete button records its
                // index; the `remove` runs after the layout pass so we
                // never mutate the Vec mid-iteration.
                let mut remove_idx: Option<usize> = None;
                ui.push_id(key.as_str(), |ui| {
                    for idx in 0..items_v.len() {
                        ui.push_id(idx, |ui| {
                            ui.horizontal(|ui| {
                                // Reserve room for the trailing delete
                                // button so the input doesn't push it
                                // off the row.
                                let (avail, _) = ui.available_size();
                                let input_w = (avail - 34.0).max(48.0);
                                let row = &mut items_v[idx];
                                if ui
                                    .text_input(row)
                                    .width(input_w)
                                    .disabled(disabled)
                                    .show()
                                    .changed
                                {
                                    mutated = true;
                                }
                                // Variant-aware: a delete reads as a
                                // destructive action.
                                if !disabled && ui.button("×").destructive().show().clicked {
                                    remove_idx = Some(idx);
                                }
                            });
                        });
                    }
                    if !disabled && ui.button("+ add").outline().show().clicked {
                        items_v.push(String::new());
                        mutated = true;
                    }
                });
                if let Some(i) = remove_idx {
                    if i < items_v.len() {
                        items_v.remove(i);
                        mutated = true;
                    }
                }
                if mutated {
                    access.set(
                        key,
                        Value::Array(items_v.into_iter().map(Value::String).collect()),
                    );
                    ctx.changed.push(key.clone());
                }
            }
            ContentItem::CustomSlot(slot) => {
                if let Some(render) = slots.get(slot) {
                    render(node, ui);
                }
            }
            ContentItem::Label(text) => {
                ui.label(text);
            }
            ContentItem::Separator => {
                ui.spacing(opts.field_gap);
            }
        }
    }
}
