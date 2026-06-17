//! Editor event + command channels for host integration.
//!
//! ## Two channels, opposite directions
//!
//! * [`EditorEvent`] — emitted by the editor into a queue; hosts
//!   subscribe via `editor.events_signal()` and drain with
//!   `editor.drain_events()`. One-shot events: a drag completed, a
//!   connection got accepted, the user asked to group a selection.
//! * [`EditorCommand`] — host → editor. Hosts that prefer queued or
//!   scripted dispatch can package any granular mutation as a single
//!   value and call `editor.dispatch(cmd)`. The same surface is
//!   exposed as direct methods on [`crate::NodeEditor`]; pick whichever
//!   fits the host's architecture.
//!
//! Validators (mid-drag yes/no answers the editor needs *now*) stay as
//! callbacks because signals are asynchronous. See
//! [`NodeEditor::on_connect_request`](crate::NodeEditor::on_connect_request).

use std::time::Duration;

use blinc_core::layer::{Point, Rect};

use crate::connection::{Connection, ConnectionEvent, ConnectionId, ConnectionState};
use crate::group::{
    AddToGroupRequest, CreateGroupRequest, DeleteGroupRequest, Group, GroupId,
    RemoveFromGroupRequest, StatusBadge, ToggleCollapseRequest,
};
use crate::node::{NodeId, NodeInstance};
use crate::port::PortKind;
use crate::subgraph::ExposedPort;

/// Keyboard modifiers carried with click events. Defaults are `false`
/// across the board so hosts that don't capture modifiers can ignore
/// the field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClickModifiers {
    pub shift: bool,
    pub cmd_or_ctrl: bool,
    pub alt: bool,
}

/// Transient highlight kind for [`EditorCommand::FlashNode`] —
/// short-lived overlay used for runtime instrumentation (trace
/// stepping, debug pulses, error highlights).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashKind {
    /// Generic step-through highlight.
    Step,
    /// Trace-event highlight (data flowed through the node).
    Trace,
    /// Error / failure highlight.
    Error,
    /// Success / completed highlight.
    Success,
}

/// What the user just right-clicked. Carried on
/// [`EditorEvent::ContextMenuRequested`] so hosts can dispatch to
/// the right menu shape without doing their own hit-test. `Canvas`
/// fires when the right-click landed on empty space (no region under
/// the cursor) — typical "background" menu (select-all, zoom-to-fit,
/// undo, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextMenuTarget {
    Node(NodeId),
    Edge(ConnectionId),
    Group(GroupId),
    Canvas,
}

/// What the pointer is currently hovering over. `None` when the
/// pointer is over empty canvas or off the editor entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoverTarget {
    Node(NodeId),
    Port(crate::port::PortAddress),
    Edge(ConnectionId),
    Group(GroupId),
    /// Status badge attached to a node. Surfaces the badge's optional
    /// `tooltip` string when the pointer hovers — distinct from
    /// hovering the node body so a tooltip-bearing badge can show
    /// its label without lighting up the whole node.
    NodeBadge(NodeId),
    /// Status badge attached to a group's header (Zeal-style chip).
    /// Mirror of `NodeBadge`.
    GroupBadge(GroupId),
}

/// One-shot events the editor emits as the user interacts. Hosts
/// subscribe by deriving against [`crate::NodeEditor::events_signal`]
/// and draining via [`crate::NodeEditor::drain_events`].
///
/// `K` matches the editor's port-kind generic so typed payloads
/// (e.g. `ConnectionAccepted`) carry the host's port type through.
#[derive(Debug, Clone)]
pub enum EditorEvent<K: PortKind> {
    /// A node drag gesture settled. Fired once per gesture on
    /// pointer-up, NOT per per-frame tick. Host typically writes
    /// `position` back into its model.
    NodeDragged { id: NodeId, position: Point },

    /// A click landed on a node. `modifiers` carries the live
    /// shift/cmd/alt state — empty by default if the host's
    /// `EventRouter` configuration doesn't capture modifiers.
    NodeClicked {
        id: NodeId,
        modifiers: ClickModifiers,
    },

    /// A click landed on an edge / connection curve.
    EdgeClicked { id: ConnectionId },

    /// A drag-to-connect gesture completed on a compatible candidate
    /// port. The host materialises the edge in its model and
    /// re-syncs (typically by appending to a graph signal).
    ConnectionAccepted(ConnectionEvent<K>),

    /// A drag-to-connect gesture ended on a candidate the validator
    /// rejected (incompatible port kinds, cycle attempt, host-rule
    /// violation, etc.). The live preview already conveyed the
    /// rejection visually (red curve tint); this event carries the
    /// reason string the host's `on_connect_request` validator
    /// returned so the host can surface it textually — typically a
    /// toast / banner. The editor itself doesn't render this text —
    /// hosts may have their own toast subsystem (`blinc_cn::toast`
    /// or similar) and prefer to drive it from a single events
    /// channel. `reason` is the empty string when the validator
    /// returned `ValidationOutcome::from(false)` without specifying
    /// a reason; hosts should fall back to a generic "incompatible
    /// connection" message in that case.
    ConnectionRejected {
        from: crate::port::PortAddress,
        to: crate::port::PortAddress,
        reason: String,
    },

    /// The user asked to wrap a selection in a new group (marquee →
    /// "Group these" affordance). Host generates a `GroupId` and
    /// inserts the [`Group`] back via `set_graph` or `insert_group`.
    CreateGroupRequested(CreateGroupRequest),

    /// A node was dragged into the footprint of an existing group.
    AddToGroupRequested(AddToGroupRequest),

    /// A node was dragged outside its parent group's bounds.
    RemoveFromGroupRequested(RemoveFromGroupRequest),

    /// The user clicked a group's collapse / expand chrome.
    ToggleCollapseRequested(ToggleCollapseRequest),

    /// The user deleted a group via chrome / shortcut.
    DeleteGroupRequested(DeleteGroupRequest),

    /// A delete-edge gesture (key, context menu, etc.) fired.
    DeleteConnectionRequested(ConnectionId),

    /// A delete-node gesture (Delete / Backspace on selection, context
    /// menu, etc.) fired. Carries every node id in the current
    /// selection so the host can confirm + drop them atomically. Host
    /// typically also drops the incident connections by calling
    /// [`crate::NodeEditor::remove_node`] for each id (which already
    /// strips incident edges + group membership in one step).
    DeleteNodesRequested(Vec<NodeId>),

    /// `Cmd-D` (`Ctrl-D` on non-macOS) — host duplicates each node id
    /// with a small position offset and re-syncs the graph. Empty
    /// `Vec` is suppressed before emission (we only fire when the
    /// selection contains at least one node), so hosts can treat
    /// every payload as non-empty. Hosts typically also generate
    /// fresh `NodeId`s for the clones — that's a host concern, not
    /// the editor's.
    DuplicateNodesRequested(Vec<NodeId>),

    /// `Cmd-A` (`Ctrl-A`) — host calls
    /// [`crate::NodeEditor::select_all`] (or builds a tailored
    /// selection, e.g. nodes-only). The editor doesn't presume what
    /// "select all" means in a host that has off-canvas siblings or
    /// hidden / locked layers.
    SelectAllRequested,

    /// `Cmd-Z` (`Ctrl-Z`) — host pops its undo stack (e.g.
    /// [`crate::history::History`]) and re-applies the inverse
    /// command. The editor itself doesn't track history; this is
    /// purely a one-shot signal.
    UndoRequested,

    /// `Cmd-Shift-Z` (`Ctrl-Shift-Z`, or `Ctrl-Y` if the host wires
    /// it) — host pops its redo stack.
    RedoRequested,

    /// User right-clicked over a hit-tested target (or empty
    /// canvas). The editor never paints its own menu — hosts open
    /// the contextual surface of their choice (typically
    /// `blinc_cn::context_menu()`) anchored at `anchor_screen`.
    /// `target` carries the entity under the cursor so the host can
    /// switch on the variant and offer the right items.
    ///
    /// Selection side-effect: if `target` references a Node / Edge /
    /// Group that wasn't already in the canvas-kit selection, the
    /// editor replaces the selection with just that one item before
    /// emitting the event — matching the convention every desktop
    /// editor follows (right-click selects + opens menu). Hosts that
    /// want multi-selection-as-target should check the live
    /// selection from inside their menu callbacks, not the `target`
    /// field.
    ContextMenuRequested {
        target: ContextMenuTarget,
        /// Screen-space anchor (canvas-relative). Pass directly to
        /// `cn::context_menu().at(x, y)`.
        anchor_screen: Point,
    },

    /// `apply_layout` finished; host patches per-node positions.
    LayoutApplied(Vec<(NodeId, Point)>),

    /// User just settled a selection containing 2+ nodes (marquee
    /// finalized, or shift-click extended the set past one). Hosts
    /// typically respond by opening a floating mini-toolbar
    /// anchored at `anchor_screen` — group, delete, align, etc.
    ///
    /// Fires once per selection-change event (not per frame), so
    /// the host can pop a fresh toolbar on each settle without
    /// throttling.
    MultiSelectionSettled {
        node_ids: Vec<NodeId>,
        /// Screen-space position from the most recent pointer
        /// event — typically where the user's cursor was at the
        /// moment the selection committed. Use as the cn popover
        /// anchor.
        anchor_screen: Point,
    },

    /// Selection went empty (or narrowed to a single item). Hosts
    /// close any open multi-select toolbar / panel.
    SelectionCleared,

    /// User double-clicked a group's title. Hosts respond by
    /// opening an inline editor (typically a cn::input) anchored
    /// at `anchor_screen` prefilled with `current`. On commit, the
    /// host patches its own model and re-syncs via
    /// [`crate::NodeEditor::insert_group`].
    EditGroupTitleRequested {
        group: GroupId,
        current: String,
        /// Screen-space rect of the title text — use as the
        /// editor overlay's anchor + size hint.
        anchor_screen: Rect,
    },

    /// User double-clicked a group's description. Same contract
    /// as [`Self::EditGroupTitleRequested`] but the editor of
    /// choice is typically a cn::textarea (multi-line). Enter
    /// commits; Shift+Enter inserts a newline.
    EditGroupDescriptionRequested {
        group: GroupId,
        current: String,
        anchor_screen: Rect,
    },

    /// User clicked the group header's edit chip — a combined
    /// "rename + redescribe" affordance. Hosts open a popup with
    /// BOTH a title input and a description textarea pre-filled
    /// with the current values; on commit they patch the host
    /// model and re-sync via [`crate::NodeEditor::insert_group`].
    /// Use the title's screen-space rect as the popover anchor.
    /// Distinct from [`Self::EditGroupTitleRequested`] /
    /// [`Self::EditGroupDescriptionRequested`] so hosts can keep
    /// the focused single-field flows (double-click on text) AND
    /// surface a full form via the chip.
    EditGroupRequested {
        group: GroupId,
        current_title: String,
        current_description: String,
        anchor_screen: Rect,
    },

    /// User wants to navigate into a stored subgraph — fired when the
    /// user double-clicks a [`NodeInstance`] whose `subgraph_ref` is
    /// `Some`, presses `Enter` while one is selected, or calls
    /// [`crate::NodeEditor::request_subgraph_open`] programmatically.
    ///
    /// The editor is intentionally NOT opinionated about how the host
    /// surfaces the subgraph. Typical handlers: open a modal sheet
    /// with a second editor instance, push a route-level page,
    /// swap the active graph in-place via [`crate::NodeEditor::set_graph`]
    /// with the contents fetched from [`crate::NodeEditor::subgraph`],
    /// open a workspace tab, etc.
    ///
    /// `source_node` is the parent-graph node id that requested the
    /// navigation — useful when the host wants to anchor a transition
    /// animation or focus the parent node on exit. `source_anchor` is
    /// the screen-space position the gesture originated from (the
    /// node's centre on double-click / `Enter`, the cursor position
    /// for programmatic calls) — handy for hosts that animate the
    /// new view in from a starting point.
    SubgraphRequested {
        subgraph_id: crate::subgraph::SubgraphId,
        source_node: NodeId,
        source_anchor: Point,
    },

    /// A node's config field changed — either via a direct call to
    /// [`crate::NodeEditor::patch_node_config`] /
    /// [`crate::NodeEditor::set_node_config`], an
    /// [`crate::inspector::InspectorPatchRequest`] forwarded through
    /// [`crate::NodeEditor::apply_inspector_patch`], or a cascading
    /// [`crate::config::PropertyRule`] effect that mutated this key.
    ///
    /// Fired once per actually-changed key (the cascade reports each
    /// step separately so hosts can observe the full trail). Hosts
    /// typically use this to:
    /// * propagate the value into their domain model / runtime,
    /// * translate certain keys into editor commands (e.g.
    ///   `"badge"` → [`EditorCommand::SetNodeBadge`]),
    /// * re-evaluate downstream nodes / connections that depend on
    ///   this config.
    NodeConfigChanged {
        node: NodeId,
        /// Property key (matches `PropertyMeta::key`).
        key: String,
        /// Previous value, or `Value::Null` when the field was
        /// unset / didn't exist in the config object.
        previous: serde_json::Value,
        /// New value, or `Value::Null` when the field was cleared.
        value: serde_json::Value,
        /// Whether this change was applied by a
        /// [`crate::config::PropertyRule`] cascade rather than a
        /// direct user / host patch. Lets hosts distinguish "user
        /// edited X" from "X was bumped because Y changed".
        from_rule: bool,
    },
}

/// Single-entry command dispatch surface mirroring the granular
/// methods on [`crate::NodeEditor`]. Pick the methods when each call
/// site knows exactly what it wants; pick `dispatch(cmd)` when the
/// host's architecture wants a uniform queue / scripting channel /
/// cross-thread sender.
///
/// Hosts can `clone()` a [`crate::NodeEditor`] across threads and call
/// `dispatch` from any of them — all mutations route through the
/// editor's internal locks.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // SetGraph / RestoreSubgraph dominate the size; boxing the heavy variants would force allocs on every command dispatch (drag-coalesced UpdateNodePosition is hot)
pub enum EditorCommand<K: PortKind, N, C, G> {
    // ── Graph mutations ─────────────────────────────────────────────
    InsertNode(NodeInstance<N>),
    RemoveNode(NodeId),
    UpdateNodePosition(NodeId, Point),
    InsertConnection(Connection<C>),
    RemoveConnection(ConnectionId),
    InsertGroup(Group<G>),
    RemoveGroup(GroupId),
    SetGroupMembers(GroupId, Vec<NodeId>),
    SetGroupCollapsed(GroupId, bool),

    // ── Subgraph storage ────────────────────────────────────────────
    /// Create an empty subgraph stored under the given id + display
    /// name. NO-op if the id already exists. Inverse: `DeleteSubgraph`.
    CreateSubgraph {
        id: crate::subgraph::SubgraphId,
        name: String,
    },
    /// Drop a stored subgraph. `previous` carries the dropped state for
    /// history's inverse (re-insert via `RestoreSubgraph`). Hosts that
    /// don't use history can pass `Subgraph::new(...)` and ignore.
    DeleteSubgraph(crate::subgraph::SubgraphId),
    /// Re-insert a previously-deleted subgraph wholesale — the
    /// inverse of `DeleteSubgraph` when paired with a snapshot the
    /// host captured before deletion. Mirrors `InsertNode`'s
    /// "re-insert from inverse" pattern.
    RestoreSubgraph(crate::subgraph::Subgraph<K, N, C, G>),
    /// Rename a stored subgraph in place.
    RenameSubgraph(crate::subgraph::SubgraphId, String),
    /// Override the namespace string for a stored subgraph.
    SetSubgraphNamespace(crate::subgraph::SubgraphId, String),

    // ── Selection ───────────────────────────────────────────────────
    Select(Vec<NodeId>),
    SelectOne(NodeId),
    AddToSelection(NodeId),
    ClearSelection,

    // ── Viewport ────────────────────────────────────────────────────
    FocusOnNode(NodeId),
    ZoomToFit,
    ZoomToSelection,
    SetViewport {
        zoom: f32,
        pan: Point,
    },

    // ── Runtime observability ───────────────────────────────────────
    SetNodeBadge(NodeId, Option<StatusBadge>),
    SetNodeDisabled(NodeId, bool),
    SetGroupDisabled(GroupId, bool),
    SetConnectionState(ConnectionId, ConnectionState),
    FlashNode(NodeId, FlashKind, Duration),

    // ── Bulk ────────────────────────────────────────────────────────
    SetGraph {
        nodes: Vec<NodeInstance<N>>,
        connections: Vec<Connection<C>>,
        groups: Vec<Group<G>>,
        exposed: Vec<ExposedPort<K>>,
    },
    ApplyLayout,

    // ── Align / distribute ──────────────────────────────────────────
    /// Align every node in `ids` to the requested edge along the
    /// alignment's axis. `ids.len() < 2` is a no-op (nothing to align
    /// against). Positions update through the standard
    /// `update_node_position` path so snap-to-grid + the graph
    /// revision bump still apply.
    AlignNodes(Vec<NodeId>, AlignEdge),
    /// Evenly distribute every node in `ids` along the requested
    /// axis. The two end nodes (extremes by centre coord) anchor the
    /// distribution; everything between spaces uniformly by node-
    /// centre. `ids.len() < 3` is a no-op.
    DistributeNodes(Vec<NodeId>, DistributeAxis),

    /// Apply a sequence of commands in order, atomically from the
    /// host's perspective. Used by [`crate::history::History`] to
    /// describe compound inverses (e.g. undo-of-`RemoveNode` =
    /// re-insert the node + every incident connection + restore each
    /// affected group's membership). Hosts can also use it to script
    /// bulk edits.
    Composite(Vec<EditorCommand<K, N, C, G>>),
}

/// Edge to align node positions against. `Left` / `Top` use the
/// minimum-coordinate edge of the bundle; `Right` / `Bottom` the
/// maximum; `CenterX` / `CenterY` average every node's centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignEdge {
    Left,
    Right,
    CenterX,
    Top,
    Bottom,
    CenterY,
}

/// Axis to distribute nodes evenly along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributeAxis {
    Horizontal,
    Vertical,
}
