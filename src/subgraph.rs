//! Subgraph boundary + named-subgraph storage.
//!
//! Two related concepts live here:
//!
//! 1. **Boundary ports** — [`ExposedPort`] declares an inport / outport
//!    rendered at the canvas edge. Per DIRECTIVE.md §2 + §6.6: reflow's
//!    `GraphExport.inports` / `outports` (and the `GraphEdge.expose:
//!    bool` flag) declare a subgraph's *boundary* — ports the subgraph
//!    exposes to its parent. The editor renders these at the left /
//!    right canvas edges so users can author subgraphs visually.
//!
//! 2. **Named subgraphs** — [`Subgraph`] is a stored, navigable graph
//!    referenced by a [`NodeInstance`] via
//!    [`NodeInstance::subgraph_ref`]. The editor stores these alongside
//!    the active view; navigation is host-driven via the
//!    [`SubgraphRequested`](crate::event::EditorEvent::SubgraphRequested)
//!    event the editor emits on double-click. Hosts decide whether to
//!    push a new modal, swap the active graph, open a side sheet, etc.
//!    Mirrors Zeal's `SubgraphNode` pattern: the parent canvas shows
//!    a diamond-shape entry node; the host owns the workspace
//!    navigation policy.
//!
//! ## Subgraph navigation
//!
//! Navigation is **host-driven**: the editor renders one graph
//! slice at a time. To navigate into a child subgraph, the host
//! calls [`crate::NodeEditor::set_graph`] with the child's contents
//! (and the child's exposed-port list — typically fetched via
//! [`crate::NodeEditor::subgraph`]). The host owns the navigation
//! stack; [`NavigationCrumb`] is the suggested shape for breadcrumbs
//! the host might render alongside the editor.
//!
//! A future revision may pull the navigation stack into the editor;
//! deferred until actual usage feedback indicates the host-side
//! shape isn't enough.

use crate::connection::Connection;
use crate::group::Group;
use crate::node::{NodeId, NodeInstance};
use crate::port::{Direction, PortId, PortKind};
use std::sync::Arc;

/// Stable identifier for an exposed boundary port. Lives on the
/// canvas edge, not on a node — has its own id space distinct from
/// per-node `PortId`s.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExposedPortId(pub Arc<str>);

impl ExposedPortId {
    pub fn new(id: impl Into<Arc<str>>) -> Self { Self(id.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl From<&str> for ExposedPortId    { fn from(s: &str)    -> Self { Self(Arc::from(s)) } }
impl From<String> for ExposedPortId  { fn from(s: String)  -> Self { Self(Arc::from(s)) } }
impl From<Arc<str>> for ExposedPortId { fn from(s: Arc<str>) -> Self { Self(s) } }

/// A port exposed on the canvas boundary — the subgraph's interface
/// to its parent. Rendered at the left edge (Input direction) or
/// right edge (Output direction).
///
/// `internal_target` names which inner node-port this boundary port
/// proxies to. The editor draws a "boundary connection" between the
/// canvas-edge port dot and the internal target port; the host's
/// drag-to-connect handler creates these the same way it creates
/// any other [`Connection`](crate::Connection), with the boundary
/// port substituting for a regular port endpoint.
///
/// `K` is the host's port-kind (matches the editor-wide port-kind).
#[derive(Debug, Clone)]
pub struct ExposedPort<K: PortKind> {
    pub id: ExposedPortId,
    pub name: String,
    /// `Input` = exposed inport (drawn on left canvas edge; data
    /// flows from outside into the subgraph).
    /// `Output` = exposed outport (drawn on right canvas edge; data
    /// flows from the subgraph out).
    pub direction: Direction,
    pub kind: K,
    /// The internal node:port this boundary port proxies to. `None`
    /// means "exposed but unconnected" — drawn dimmed.
    pub internal_target: Option<InternalTarget>,
    /// Free-form host metadata (reflow's `GraphEdge.metadata` from
    /// the exposed-port table).
    pub metadata: serde_json::Value,
}

/// An exposed port's internal anchor: which node-port inside the
/// subgraph the boundary forwards to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InternalTarget {
    pub node: NodeId,
    pub port: PortId,
}

impl InternalTarget {
    pub fn new(node: impl Into<NodeId>, port: impl Into<PortId>) -> Self {
        Self { node: node.into(), port: port.into() }
    }
}

impl<K: PortKind> ExposedPort<K> {
    pub fn input(id: impl Into<ExposedPortId>, name: impl Into<String>, kind: K) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            direction: Direction::Input,
            kind,
            internal_target: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn output(id: impl Into<ExposedPortId>, name: impl Into<String>, kind: K) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            direction: Direction::Output,
            kind,
            internal_target: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_internal_target(mut self, target: InternalTarget) -> Self {
        self.internal_target = Some(target);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// NavigationCrumb — host-side breadcrumb hint
// ─────────────────────────────────────────────────────────────────────

/// One entry in a host-maintained subgraph navigation stack. The
/// editor doesn't read these; they're a suggested shape for the
/// host's own breadcrumb widget rendered alongside the editor.
///
/// When the user clicks a SubgraphActor-style node and "zooms in",
/// the host pushes a crumb and calls `set_graph(child_subgraph)`.
/// Clicking the previous crumb pops the stack and re-loads the
/// parent slice.
#[derive(Debug, Clone)]
pub struct NavigationCrumb {
    /// Display label — e.g. the subgraph's name or the parent node's
    /// component name.
    pub label: String,
    /// Host-defined identifier for "which slice is this." Opaque to
    /// the editor; the host uses it to look up the slice on pop.
    pub slice_id: Arc<str>,
}

impl NavigationCrumb {
    pub fn new(label: impl Into<String>, slice_id: impl Into<Arc<str>>) -> Self {
        Self {
            label: label.into(),
            slice_id: slice_id.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// SubgraphId — stable identifier for a named subgraph stored on the
// editor.
// ─────────────────────────────────────────────────────────────────────

/// Stable identifier for a [`Subgraph`] stored alongside the editor's
/// active graph. Cheap to clone (`Arc<str>` backed).
///
/// Hosts construct these via `SubgraphId::new("my-filter")` or
/// `NodeEditor::create_subgraph(...)`; the returned id is the only
/// handle into [`crate::NodeEditor::subgraph`] /
/// [`crate::NodeEditor::with_subgraph_graph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubgraphId(pub Arc<str>);

impl SubgraphId {
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SubgraphId {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl From<String> for SubgraphId {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

impl From<Arc<str>> for SubgraphId {
    fn from(s: Arc<str>) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for SubgraphId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Subgraph — a named graph stored on the editor, navigable via the
// `SubgraphRequested` event.
// ─────────────────────────────────────────────────────────────────────

/// A named subgraph stored on [`crate::NodeEditor`]. Contains the same
/// shape of state the editor's main view holds (nodes / connections /
/// groups / exposed boundary ports), so hosts can swap a subgraph into
/// view via [`crate::NodeEditor::set_graph`] when handling a
/// [`crate::event::EditorEvent::SubgraphRequested`] event.
///
/// `namespace` is the Zeal-style display string —
/// `"workflow_name/subgraph_id"` — rendered as the subtitle on subgraph
/// instance nodes in the parent canvas. The editor itself doesn't parse
/// it; it's a hint for breadcrumbs / search / property panes.
///
/// Generic over the same `K / N / C / G` parameters as the editor so
/// stored subgraphs share the host's port-kind + metadata types.
///
/// ## Identity
///
/// Each subgraph carries a stable [`SubgraphId`]. A
/// [`NodeInstance::subgraph_ref`] field references this id; the editor
/// renders such nodes with [`crate::node::NodeShape::Diamond`] (forced
/// override of the template's default shape) and a `↗` icon hint so
/// users at a glance distinguish "regular node" from "navigable
/// subgraph node."
#[derive(Debug, Clone)]
pub struct Subgraph<K: PortKind, N, C, G> {
    pub id: SubgraphId,
    /// Display name. Editable via [`crate::NodeEditor::rename_subgraph`].
    pub name: String,
    /// Display namespace — typically `"workflow/subgraph_id"`. Used as
    /// the subgraph node's subtitle in the parent canvas.
    pub namespace: String,
    pub nodes: Vec<NodeInstance<N>>,
    pub connections: Vec<Connection<C>>,
    pub groups: Vec<Group<G>>,
    pub exposed: Vec<ExposedPort<K>>,
}

impl<K: PortKind, N, C, G> Subgraph<K, N, C, G> {
    /// Construct an empty subgraph. `namespace` defaults to the id; use
    /// [`Self::with_namespace`] to override.
    pub fn new(id: impl Into<SubgraphId>, name: impl Into<String>) -> Self {
        let id = id.into();
        let namespace = id.as_str().to_string();
        Self {
            id,
            name: name.into(),
            namespace,
            nodes: Vec::new(),
            connections: Vec::new(),
            groups: Vec::new(),
            exposed: Vec::new(),
        }
    }

    /// Override the display namespace. Typical use: include the
    /// workflow name so multiple subgraphs sharing an id across
    /// workflows still read distinctly in the UI.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }
}
