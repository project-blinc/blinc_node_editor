//! Palette — template browser for inserting new nodes.
//!
//! The palette renders one entry per
//! [`NodeTemplate`](crate::NodeTemplate), grouped by
//! `NodeTemplate.category`, searchable by `display_name`. Drag /
//! click → emits a `PaletteInsertRequest` the host materialises in
//! its model.
//!
//! **Status: scaffold only.** The trait + request types are in
//! place so hosts can write against them; the rendered widget
//! ships later.

use crate::node::NodeTemplate;
use crate::port::PortKind;

/// Filter / sort options the palette renders. Shape is committed
/// now so hosts can build their own palette UI against it; the
/// bundled widget will read the same struct.
#[derive(Debug, Clone, Default)]
pub struct PaletteQuery {
    pub search: String,
    pub category: Option<String>,
}

/// Fired when the user picks a template from the palette. Host
/// creates a fresh [`NodeInstance`](crate::NodeInstance) at the drop
/// point (or canvas centre, if click-to-insert) and re-syncs.
#[derive(Debug, Clone)]
pub struct PaletteInsertRequest<K: PortKind> {
    pub template_component: String,
    /// Drop position in canvas-content coordinates. `None` =
    /// click-to-insert (host picks a sensible default — typically
    /// viewport centre or a fresh open spot).
    pub at: Option<blinc_core::layer::Point>,
    /// Phantom so K participates — needed for downstream nan8
    /// implementations that want type-aware suggestions.
    pub _phantom: std::marker::PhantomData<K>,
}

/// Filter a template registry by [`PaletteQuery`]. Pure helper —
/// the bundled palette widget will use this internally, and hosts
/// that ship their own UI can reuse the same filtering rule.
pub fn filter_templates<'a, K: PortKind>(
    templates: &'a [NodeTemplate<K>],
    query: &PaletteQuery,
) -> Vec<&'a NodeTemplate<K>> {
    let needle = query.search.to_lowercase();
    templates
        .iter()
        .filter(|t| {
            if let Some(cat) = &query.category {
                if t.category != *cat {
                    return false;
                }
            }
            if needle.is_empty() {
                return true;
            }
            t.display_name.to_lowercase().contains(&needle)
                || t.component.to_lowercase().contains(&needle)
        })
        .collect()
}
