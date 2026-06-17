//! `blinc_node_editor` — metadata-driven node-graph editor over [`blinc_canvas_kit`].
//!
//! ## What this is
//!
//! A toolkit for building DAG visual editors on top of Blinc. The crate
//! is intentionally generic over the host's port type, node metadata,
//! connection metadata, and group metadata — so a single editor surface
//! can be reused for reflow graphs (nan8), ML inference DAGs, audio
//! routing graphs, shader nodes, or any other domain that maps to
//! "boxes with ports + curved edges."
//!
//! The reference design lives in
//! [`DIRECTIVE.md`](../DIRECTIVE.md). Conceptual highlights:
//!
//! * **Metadata-driven**: nodes render from a declarative
//!   [`NodeTemplate`] + a generic per-instance metadata bag. The
//!   editor never hardcodes a node's fields — it walks the template.
//! * **Generic over port type**: hosts implement [`PortKind`] for
//!   their port-type enum. The editor delegates compatibility
//!   checks to the host's matcher.
//! * **Host-as-source-of-truth**: the editor is a view. Mutations
//!   (drag-to-connect, marquee-to-group, drag-out-of-group, group
//!   collapse / delete, …) are emitted as request events; the host
//!   updates its model and re-syncs via [`NodeEditor::set_graph`].
//! * **Theme-aware**: chrome colours / radii / spacing derive from
//!   [`blinc_theme::ThemeState`] by default with optional per-token
//!   overrides via [`NodeEditorTheme`].
//! * **Zeal-inspired UX where it counts**: [`NodeShape`],
//!   [`PortPosition`], [`ConnectionState`], group chrome with
//!   collapse + badges, marquee-to-group, drag-out-of-group.
//!
//! ## Module map
//!
//! | Module | Concern |
//! | --- | --- |
//! | [`port`] | Port kind trait, port descriptors, addressing |
//! | [`node`] | Node templates + placed instances |
//! | [`connection`] | Edges, connection state, validation events |
//! | [`group`] | Node groups + status badges |
//! | [`subgraph`] | Exposed boundary ports for subgraph editing |
//! | [`theme`] | Editor theme + resolver |
//! | [`layout`] | Auto-layout strategies (Manual / Layered / Force / Custom) |
//! | [`bezier`] | Cubic bezier helpers (sampling + hit-testing) |
//! | [`render`] | Drawing primitives |
//! | [`interaction`] | Drag-to-connect + group-membership state machines |
//! | [`editor`] | The [`NodeEditor`] widget |
//! | [`palette`] | Template browser (types scaffolded; widget pending) |
//! | [`inspector`] | Properties panel (types scaffolded; widget pending) |
//! | [`event`] | `EditorEvent` + `EditorCommand` channel types |

#![forbid(unsafe_code)]

pub mod bezier;
pub mod config;
pub mod connection;
pub mod editor;
pub mod event;
pub mod group;
pub mod history;
pub mod icon;
pub mod inspector;
pub mod interaction;
pub mod layout;
pub mod node;
pub mod palette;
pub mod port;
pub mod region;
pub mod render;
pub mod slot;
pub mod subgraph;
pub mod theme;

// ── Top-level re-exports ───────────────────────────────────────────

pub use connection::{
    Connection, ConnectionEvent, ConnectionId, ConnectionState, ConnectRequest, ValidationOutcome,
};
pub use editor::{NodeEditor, RenderStats, SearchHit};
pub use event::{
    AlignEdge, ClickModifiers, ContextMenuTarget, DistributeAxis, EditorCommand, EditorEvent,
    FlashKind, HoverTarget,
};
pub use group::{
    AddToGroupRequest, BadgeKind, CreateGroupRequest, DeleteGroupRequest, Group, GroupId,
    RemoveFromGroupRequest, RemoveSource, StatusBadge, ToggleCollapseRequest,
};
pub use history::{CoalesceKey, History, HistoryEntry};
pub use icon::NodeIcon;
pub use interaction::{
    classify_drag_membership_change, group_request_from_selection, DragConnect,
    GroupMembershipChange,
};
pub use layout::{
    apply_layout, CustomLayoutFn, ForceConfig, LayeredConfig, LayoutContext, LayoutEdge,
    LayoutNode, LayoutOrientation, LayoutStrategy,
};
pub use node::{NodeId, NodeInstance, NodeShape, NodeTemplate};
pub use port::{
    Direction, PortAddress, PortCategory, PortDesc, PortId, PortKind, PortMetadata, PortPosition,
};
pub use subgraph::{
    ExposedPort, ExposedPortId, InternalTarget, NavigationCrumb, Subgraph, SubgraphId,
};
pub use theme::{NodeEditorTheme, ThemeResolver};

// ── Prelude ────────────────────────────────────────────────────────

/// One-import surface for common editor types. Mirrors the
/// convention in other Blinc downstreams (`blinc_canvas_kit::prelude`,
/// `blinc_cn::prelude`).
pub mod prelude {
    pub use crate::config::{
        cascade_rules, default_config, validate, BooleanProperty, CodeEditorProperty,
        ColorProperty, ConfigSchema, FileProperty, IssueSeverity, NumberProperty, Predicate,
        PropertyDefinition, PropertyEffect, PropertyMeta, PropertyRule, SelectOption,
        SelectProperty, TextProperty, TextareaProperty, ValidationIssue,
    };
    pub use crate::inspector::{apply_patch, fields, InspectorField, InspectorPatchRequest};
    /// Re-export of [`serde_json::Value`] so hosts that don't carry a
    /// direct `serde_json` dependency can still construct
    /// [`PropertyRule`] effects + predicates.
    pub use serde_json::{Map as JsonMap, Value as JsonValue};
    pub use crate::connection::{
        Connection, ConnectionEvent, ConnectionState, ConnectRequest, ValidationOutcome,
    };
    pub use crate::editor::{NodeEditor, SearchHit};
    pub use crate::event::{
        AlignEdge, ClickModifiers, ContextMenuTarget, DistributeAxis, EditorCommand, EditorEvent,
        FlashKind, HoverTarget,
    };
    pub use crate::group::{BadgeKind, Group, GroupId, StatusBadge};
    pub use crate::history::{CoalesceKey, History};
    pub use crate::icon::NodeIcon;
    pub use crate::layout::LayoutStrategy;
    pub use crate::node::{NodeId, NodeInstance, NodeShape, NodeTemplate};
    pub use crate::port::{Direction, PortAddress, PortDesc, PortId, PortKind, PortPosition};
    pub use crate::region::RegionId;
    pub use crate::subgraph::{Subgraph, SubgraphId};
    pub use crate::theme::NodeEditorTheme;
}
