//! `NodeEditor` — the public-facing editor widget.
//!
//! Wraps [`blinc_canvas_kit::CanvasKit`] with metadata-driven node /
//! port / edge / group rendering and an event-callback surface so
//! hosts can subscribe to mutation requests without owning the
//! editor's interior state.
//!
//! ## Generic parameters
//!
//! * `K` — the host's port-kind. Implements [`PortKind`]; nan8 uses
//!   `reflow_graph::PortType` here.
//! * `N` — per-node metadata. Opaque to the editor.
//! * `C` — per-connection metadata. Opaque to the editor.
//! * `G` — per-group metadata. Opaque to the editor.
//!
//! ## Lifecycle
//!
//! ```ignore
//! let editor = NodeEditor::<MyKind, (), (), ()>::new("editor-1")
//!     .with_templates(my_templates)
//!     .with_theme(NodeEditorTheme::default())
//!     .on_connect_request(|req| /* validate */ ValidationOutcome::Accept)
//!     .on_connect_accepted(|evt| /* host materialises edge */ );
//! editor.set_graph(nodes, connections, groups, exposed);
//! div().child(editor.element())
//! ```

use ahash::AHashMap;
use blinc_core::layer::{Point, Rect};
use blinc_core::reactive::{SignalId, State};
use blinc_core::DrawContext;
use blinc_layout::Div;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use blinc_canvas_kit::{CanvasBackground, CanvasKit, CanvasViewport};

use crate::connection::{
    ConnectRequest, Connection, ConnectionId, ConnectionState, ValidationOutcome,
};
use crate::event::{AlignEdge, DistributeAxis, EditorCommand, EditorEvent, FlashKind, HoverTarget};
use crate::group::{Group, GroupId, StatusBadge};
use crate::interaction::DragConnect;
use crate::layout::{apply_layout, LayoutStrategy};
use crate::node::{NodeId, NodeInstance, NodeTemplate};
use crate::port::{PortAddress, PortKind};
use crate::region::RegionId;
use crate::render::{draw_group, draw_node_at, draw_port, iter_port_positions, PortHoverState};
use crate::slot::{
    compute_group_slots, compute_node_slots, fingerprint_group, fingerprint_node,
    group_inputs_from, node_inputs_from, warm_group_slot_cache, warm_node_slot_cache,
    GroupSlotCache, GroupSlots, NodeFingerprint, NodeSlotCache, NodeSlots,
};
use crate::subgraph::ExposedPort;
use crate::theme::{NodeEditorTheme, ThemeResolver};

// ─────────────────────────────────────────────────────────────────────
// Callback type aliases — only the validator stays as a callback.
//
// Validators answer mid-drag yes/no questions the editor needs *now*
// to drive preview tinting; signals are asynchronous and the host
// couldn't respond in time. Every other host integration goes through
// the events queue (see `EditorEvent` + `drain_events`).
// ─────────────────────────────────────────────────────────────────────

type ValidateFn<K> = Arc<dyn Fn(&ConnectRequest<'_, K>) -> ValidationOutcome + Send + Sync>;
type ContextMenuFn =
    Arc<dyn Fn(crate::event::ContextMenuTarget, blinc_core::layer::Point) + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────
// Graph state — owned by the editor, swapped by the host via set_graph
// ─────────────────────────────────────────────────────────────────────

/// The editor's view of the graph. Hosts swap this wholesale via
/// [`NodeEditor::set_graph`] — the editor doesn't mutate the model;
/// it requests mutations via callbacks and the host re-syncs.
struct GraphState<K: PortKind, N, C, G> {
    nodes: Vec<NodeInstance<N>>,
    connections: Vec<Connection<C>>,
    groups: Vec<Group<G>>,
    exposed: Vec<ExposedPort<K>>,
}

impl<K: PortKind, N, C, G> Default for GraphState<K, N, C, G> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            groups: Vec::new(),
            exposed: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// FrameContext — per-frame snapshot of every lock-guarded source the
// renderer reads. Built by `NodeEditor::begin_frame`.
// ─────────────────────────────────────────────────────────────────────

/// Bundle of read-guards + clones produced by
/// [`NodeEditor::begin_frame`]. Lives for the duration of one render
/// pass; drops release the locks. Holding the guards (instead of
/// snapshotting + dropping) keeps `templates` and `graph` available
/// as `&` references throughout the frame without cloning every
/// node / template — useful for graphs with hundreds of nodes.
struct FrameContext<'a, K: PortKind, N, C, G> {
    graph: std::sync::RwLockReadGuard<'a, GraphState<K, N, C, G>>,
    templates: std::sync::RwLockReadGuard<'a, AHashMap<String, NodeTemplate<K>>>,
    theme_overrides: std::sync::RwLockReadGuard<'a, NodeEditorTheme>,
    selection: blinc_canvas_kit::SelectionState,
    drag: DragConnect,
}

// ─────────────────────────────────────────────────────────────────────
// NodeEditor
// ─────────────────────────────────────────────────────────────────────

/// Standard duration in milliseconds for viewport tweens triggered
/// by focus / fit / search-and-focus. Matches the ease-out-cubic
/// "fly-to" duration most node editors and map UIs converge on
/// (~250 ms feels responsive without flicker; sub-200 ms reads as a
/// jump cut, ~400 ms reads as sluggish).
const VIEWPORT_TWEEN_MS: f32 = 260.0;

/// Per-editor viewport tween state. The scheduler tick callback that
/// drives this reads + advances `elapsed_ms`, lerps from `from_*` to
/// `to_*` with `ease_out_cubic`, and writes back through the kit's
/// `update_viewport`. Cleared (and the tick callback unregistered)
/// once `elapsed_ms >= duration_ms`.
#[derive(Debug, Clone)]
pub(crate) struct ViewportAnimation {
    pub from_pan_x: f32,
    pub from_pan_y: f32,
    pub from_zoom: f32,
    pub to_pan_x: f32,
    pub to_pan_y: f32,
    pub to_zoom: f32,
    pub elapsed_ms: f32,
    pub duration_ms: f32,
}

/// One result returned by [`NodeEditor::search`] — either a node in
/// the active graph or a group on the active canvas. Subgraph
/// matches are projected by `search` onto the diamond
/// [`crate::NodeInstance`] that references them, so a hit list maps
/// 1:1 to canvas entities the host can select / focus / outline.
///
/// Hosts wiring a result list typically pattern-match on the variant
/// to decide which icon / label to render alongside each entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchHit {
    /// A matching node in the active graph.
    Node(crate::node::NodeId),
    /// A matching group on the active canvas.
    Group(crate::group::GroupId),
}

/// Live drag-into / drag-out group hints. Updated per-frame during a
/// node-drag gesture, cleared on drag-end / cancel. Drives both
/// the renderer's group-border tinting AND the per-frame auto-
/// bounds source for any group the dragged node currently belongs
/// to.
#[derive(Debug, Default, Clone)]
pub(crate) struct DragGroupPreview {
    /// Node currently being dragged (mirrors `kit.interaction().active`
    /// for `node:*` regions). Used by the renderer to determine
    /// which member to exclude from the dragged-node's current
    /// group's auto-bounds while Shift is held.
    pub dragged_node: Option<NodeId>,
    /// Group currently being dragged by its chrome (`group:` /
    /// `group_title:` / `group_desc:` regions). Mutually exclusive
    /// with `dragged_node` in practice. Drives the parent-group
    /// shrunk-bounds + tint logic for multi-node drag escapes: the
    /// renderer treats the dragged group's full member set as the
    /// exclusion when computing any enclosing parent's auto-bounds.
    pub dragged_group: Option<GroupId>,
    /// Shift modifier state captured at the most recent drag tick.
    /// While `true`, the renderer renders the dragged node's current
    /// group at its OTHER-members footprint (group visually stays
    /// put / shrinks instead of growing to follow the node — gives
    /// the user a clear "you're tearing this out" affordance).
    pub shift_held: bool,
    /// Group whose footprint the dragged node's centre is currently
    /// inside (and the node isn't already a member). Drawn with
    /// `theme.group_add_target_border()`.
    pub add_target: Option<GroupId>,
    /// Group the dragged node is currently a member of but whose
    /// excluding-bounds (other-members footprint) no longer contains
    /// it. Set only while Shift is held — without Shift, drag-end
    /// can't actually remove anything. Drawn with
    /// `theme.group_remove_target_border()`.
    pub remove_target: Option<GroupId>,
}

/// Transient highlight tracking — kind + expiry instant. Drives
/// [`NodeEditor::flash_node`] / `EditorCommand::FlashNode`.
///
/// Fields are read by `render_frame` once the flash overlay is
/// wired; the storage already lives behind the editor so hosts can
/// start emitting flashes today.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NodeFlash {
    pub kind: FlashKind,
    pub expires_at: web_time::Instant,
}

/// The node-graph editor. Cheap-to-clone (`Arc`-backed); the host
/// typically constructs one per view and observes state reactively
/// via the signal getters.
///
/// ## Reactive surface
///
/// Hosts derive against the signal getters to keep their UI in sync
/// without manual wiring:
///
/// - [`graph_signal`](Self::graph_signal) — bumps on every graph
///   mutation (insert/remove node/connection/group/member changes).
/// - [`drag_state_signal`](Self::drag_state_signal) — bumps on
///   drag-to-connect FSM transitions.
/// - [`hover_signal`](Self::hover_signal) — bumps when hovered
///   target changes.
/// - [`events_signal`](Self::events_signal) — bumps when an event
///   is pushed; pair with `drain_events()` inside an effect to
///   process pending events.
///
/// Selection + viewport signals stay on the underlying
/// [`CanvasKit`]; reach for them via
/// [`canvas_kit`](Self::canvas_kit).
pub struct NodeEditor<K: PortKind, N, C, G> {
    key: String,
    kit: CanvasKit,
    templates: Arc<RwLock<AHashMap<String, NodeTemplate<K>>>>,
    graph: Arc<RwLock<GraphState<K, N, C, G>>>,
    theme: Arc<RwLock<NodeEditorTheme>>,
    /// Bumped on every theme override change. Feeds every slot
    /// fingerprint so a theme swap cleanly invalidates all cached
    /// node + group layouts without flushing the map explicitly.
    theme_revision: Arc<AtomicU64>,
    node_slots: NodeSlotCache,
    group_slots: GroupSlotCache,
    layout_strategy: Arc<RwLock<LayoutStrategy>>,

    // ── Reactive state ─────────────────────────────────────────────
    /// Monotonic counter bumped on every graph mutation; reading code
    /// uses the `signal_id` for `derived`/`stateful().deps()`.
    graph_rev: State<u64>,
    /// Live drag-to-connect FSM. Cloned each frame for rendering.
    drag_state: State<DragConnect>,
    /// What the pointer is currently over. `None` for empty canvas.
    hover_state: State<Option<HoverTarget>>,
    /// Event-queue revision counter — bumps on every event push so
    /// hosts can `effect_with_deps([events_signal()], ...)` and
    /// drain.
    events_rev: State<u64>,
    /// Pending events; drained by [`drain_events`](Self::drain_events).
    events_queue: Arc<Mutex<Vec<EditorEvent<K>>>>,
    /// Active flashes keyed by node id (see `flash_node`).
    flashes: Arc<RwLock<AHashMap<NodeId, NodeFlash>>>,

    /// Last-known auto_bounds per group, captured during render_frame
    /// for non-empty groups. Used as a fallback when a group's
    /// `members` list goes empty (drag-out of last member, explicit
    /// removal) so the group's chrome stays at its previous position
    /// instead of snapping to the (0, 0) auto-fallback origin.
    last_group_auto_bounds: Arc<RwLock<AHashMap<GroupId, Rect>>>,

    /// Live drag-into / drag-out preview hint. See [`DragGroupPreview`].
    drag_group_preview: Arc<RwLock<DragGroupPreview>>,

    /// In-flight viewport tween. `Some` while a search-driven focus
    /// (or any caller of `animate_viewport_to`) is interpolating
    /// from `(from_*)` to `(to_*)`; the registered scheduler tick
    /// callback advances `elapsed_ms` each frame and clears this
    /// slot when the animation settles. `None` means the viewport
    /// is at rest.
    viewport_anim: Arc<Mutex<Option<ViewportAnimation>>>,
    /// Id of the registered scheduler tick callback that advances
    /// `viewport_anim`. Stored so the callback can self-unregister
    /// on settle (the animation scheduler keeps wake-active while
    /// any tick callback is registered; leaving an idle one in
    /// place would peg the scheduler at 60 Hz forever).
    viewport_anim_cb_id: Arc<Mutex<Option<blinc_animation::TickCallbackId>>>,

    /// Latest screen-space pointer position observed by the canvas.
    /// Used as the anchor for [`EditorEvent::MultiSelectionSettled`]
    /// so the host can pop a cn popover where the gesture ended.
    /// `None` before the first pointer event.
    last_screen_pos: Arc<RwLock<Option<Point>>>,

    // ── Validators (the only callback surface) ─────────────────────
    on_validate: Arc<RwLock<Option<ValidateFn<K>>>>,
    /// Synchronous callback fired alongside `EditorEvent::ContextMenuRequested`
    /// the moment the user right-clicks. Designed so hosts can mount
    /// a contextual surface (e.g. `cn::context_menu().show()`)
    /// *inside the same frame* as the right-click event handler —
    /// the event-drain path arrives too late in the frame to catch
    /// the overlay-stack dirty poll, so a drain-mounted overlay
    /// misses the subtree-rebuild + start_all_css_animations on its
    /// first frame and ends up invisible until the next paint
    /// invalidation. Both surfaces co-exist; hosts pick whichever
    /// fits.
    on_context_menu: Arc<RwLock<Option<ContextMenuFn>>>,

    /// Per-node immediate-mode portals for nodes whose template has
    /// a [`crate::NodeContent`] slot. Lazily created on first render
    /// of a content node; dropped when the node leaves the graph.
    /// `Drop` on `Portal` unsubscribes from any signals it read so
    /// removed nodes stop dirtying the canvas.
    portals: Arc<Mutex<blinc_portal_ui::PortalManager<NodeId>>>,

    /// Per-node consumed-height feedback from the portal's previous
    /// frame. Read on this frame's slot-input prep to grow the body
    /// region to fit the closure's actual painting — the template's
    /// `NodeContent.height` acts as a MIN, the consumed value
    /// over-rides it when the closure painted more. Written after
    /// `portal.frame` returns, with the value
    /// [`blinc_portal_ui::Portal::consumed_height`] reports.
    portal_content_heights: Arc<Mutex<AHashMap<NodeId, f32>>>,

    /// Per-node measured natural width from the last portal frame.
    /// Mirror of [`Self::portal_content_heights`] on the horizontal
    /// axis — drives fit-content node width. Value is the quantised
    /// (4 px grid), padded measurement; `apply_portal_width_override`
    /// also enforces the template's `content.min_width` floor at
    /// apply time. Only consulted when `instance.size` is `None` —
    /// explicit host sizing always wins.
    portal_content_widths: Arc<Mutex<AHashMap<NodeId, f32>>>,

    /// Last render frame's cull stats — visible vs total counts for
    /// nodes + edges. Updated atomically at the end of every
    /// `render_frame`; hosts read via [`Self::last_render_stats`] to
    /// surface a "X / Y visible" HUD that demonstrates the frustum
    /// cull's effect as the user pans / zooms. Atomic so the read
    /// can happen from any thread without locking.
    render_stats: Arc<RenderStatsCell>,

    /// Content-space rects of group title / description text the
    /// renderer drew this frame, keyed by region id
    /// (`group_title:{id}` / `group_desc:{id}`). Populated during
    /// the group draw loop; read by the additive click listener
    /// installed in [`Self::new`] when a double-click fires so the
    /// emitted [`crate::EditorEvent::EditGroupTitleRequested`] /
    /// `EditGroupDescriptionRequested` carries a fresh
    /// screen-space anchor for the host's inline editor overlay.
    /// Replaced wholesale each frame so removed groups don't leak.
    group_text_rects: Arc<Mutex<AHashMap<String, Rect>>>,

    /// Per-frame badge rects keyed by canvas-kit region id
    /// (`node_badge:{id}` / `group_badge:{id}`). Populated as
    /// nodes / groups paint, drained in the badge-tooltip render
    /// pass at the end of the frame so the tooltip chip anchors
    /// to the exact rect the badge painted at. Same lifecycle as
    /// `group_text_rects` — replaced wholesale each frame.
    badge_rects: Arc<Mutex<AHashMap<String, Rect>>>,

    /// `(region_id, instant)` of the most recent single-click,
    /// used by the additive click listener to coalesce two
    /// consecutive clicks on the same region into a double-click.
    /// 400 ms window — matches the platform double-click default
    /// without making accidental drags-then-clicks look like
    /// double-clicks. Cleared on a successful match.
    last_click: Arc<Mutex<Option<(String, web_time::Instant)>>>,

    /// Named subgraphs stored alongside the active graph. Hosts
    /// navigate into a subgraph by handling
    /// [`crate::event::EditorEvent::SubgraphRequested`] — fired on
    /// double-click of any [`NodeInstance`] whose `subgraph_ref` is
    /// `Some(id)`. The typical handler fetches the subgraph via
    /// [`Self::subgraph`] and calls [`Self::set_graph`] with its
    /// contents.
    #[allow(clippy::type_complexity)]
    subgraphs:
        Arc<RwLock<AHashMap<crate::subgraph::SubgraphId, crate::subgraph::Subgraph<K, N, C, G>>>>,

    /// Monotonic counter bumped on every subgraph CRUD — used by
    /// [`Self::subgraph_signal`] so hosts can `derive` against the
    /// stored set without polling. Mirrors `graph_rev` for the
    /// subgraph storage.
    subgraph_rev: State<u64>,
}

/// Snapshot of the cull pre-pass + edge / node draw counters from
/// the previous `render_frame`. All fields are `usize` (or `u32` on
/// 32-bit) packed atomically. Read via [`NodeEditor::last_render_stats`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    /// Total node count in the graph at frame start (excluding
    /// nodes inside collapsed groups, which are intentionally hidden
    /// and don't count against the cull metric).
    pub total_nodes: usize,
    /// Node count that survived the frustum cull and was actually
    /// drawn this frame.
    pub visible_nodes: usize,
    /// Total connection count in the graph at frame start.
    pub total_edges: usize,
    /// Connection count drawn this frame (after frustum cull +
    /// collapsed-group folding).
    pub visible_edges: usize,
}

/// Atomic backing for [`RenderStats`] so `render_frame` can write
/// without taking a lock. Each counter sits in its own atomic word;
/// `RenderStats` reads them with `Relaxed` ordering — the four
/// fields are read in close sequence and a slightly-torn snapshot is
/// acceptable for HUD use.
#[derive(Default)]
struct RenderStatsCell {
    total_nodes: AtomicUsize,
    visible_nodes: AtomicUsize,
    total_edges: AtomicUsize,
    visible_edges: AtomicUsize,
}

impl RenderStatsCell {
    fn store(&self, stats: RenderStats) {
        self.total_nodes.store(stats.total_nodes, Ordering::Relaxed);
        self.visible_nodes
            .store(stats.visible_nodes, Ordering::Relaxed);
        self.total_edges.store(stats.total_edges, Ordering::Relaxed);
        self.visible_edges
            .store(stats.visible_edges, Ordering::Relaxed);
    }
    fn snapshot(&self) -> RenderStats {
        RenderStats {
            total_nodes: self.total_nodes.load(Ordering::Relaxed),
            visible_nodes: self.visible_nodes.load(Ordering::Relaxed),
            total_edges: self.total_edges.load(Ordering::Relaxed),
            visible_edges: self.visible_edges.load(Ordering::Relaxed),
        }
    }
}

impl<K: PortKind, N, C, G> Clone for NodeEditor<K, N, C, G> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            kit: self.kit.clone(),
            templates: self.templates.clone(),
            graph: self.graph.clone(),
            theme: self.theme.clone(),
            theme_revision: self.theme_revision.clone(),
            node_slots: self.node_slots.clone(),
            group_slots: self.group_slots.clone(),
            layout_strategy: self.layout_strategy.clone(),
            graph_rev: self.graph_rev.clone(),
            drag_state: self.drag_state.clone(),
            hover_state: self.hover_state.clone(),
            events_rev: self.events_rev.clone(),
            events_queue: self.events_queue.clone(),
            flashes: self.flashes.clone(),
            last_group_auto_bounds: self.last_group_auto_bounds.clone(),
            drag_group_preview: self.drag_group_preview.clone(),
            viewport_anim: self.viewport_anim.clone(),
            viewport_anim_cb_id: self.viewport_anim_cb_id.clone(),
            last_screen_pos: self.last_screen_pos.clone(),
            on_validate: self.on_validate.clone(),
            on_context_menu: self.on_context_menu.clone(),
            portals: self.portals.clone(),
            portal_content_heights: self.portal_content_heights.clone(),
            portal_content_widths: self.portal_content_widths.clone(),
            render_stats: self.render_stats.clone(),
            group_text_rects: self.group_text_rects.clone(),
            badge_rects: self.badge_rects.clone(),
            last_click: self.last_click.clone(),
            subgraphs: self.subgraphs.clone(),
            subgraph_rev: self.subgraph_rev.clone(),
        }
    }
}

impl<K, N, C, G> NodeEditor<K, N, C, G>
where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    /// Construct a new editor keyed by `key` (used by the underlying
    /// canvas_kit for state persistence across rebuilds).
    ///
    /// Ships a sensible default canvas background — a zoom-adaptive
    /// dot grid tinted from `ColorToken::Border` so the workspace
    /// pattern reads correctly against both light + dark themes.
    /// Override via `with_background(...)` (pass
    /// `CanvasBackground::None` to disable).
    pub fn new(key: impl Into<String>) -> Self {
        let key: String = key.into();
        let ctx = blinc_core::context_state::BlincContextState::get();
        let mut kit = CanvasKit::new(&key).with_background(default_canvas_background());
        // Portal widgets read per-frame click events out of the kit
        // via a global timestamp map populated by this hook. Idempotent
        // per kit instance.
        blinc_portal_ui::ui::install_click_hook(&mut kit);

        // Shared per-editor state captured by the double-click
        // listener below. Constructed BEFORE the listener so the
        // listener's `move` closure owns its own Arc clones and the
        // `Self { ... }` body re-uses the same originals — both
        // sides see one map.
        let group_text_rects: Arc<Mutex<AHashMap<String, Rect>>> =
            Arc::new(Mutex::new(AHashMap::new()));
        let badge_rects: Arc<Mutex<AHashMap<String, Rect>>> = Arc::new(Mutex::new(AHashMap::new()));
        let last_click: Arc<Mutex<Option<(String, web_time::Instant)>>> =
            Arc::new(Mutex::new(None));
        let events_queue: Arc<Mutex<Vec<EditorEvent<K>>>> = Arc::new(Mutex::new(Vec::new()));
        let graph_arc: Arc<RwLock<GraphState<K, N, C, G>>> =
            Arc::new(RwLock::new(GraphState::default()));
        let events_rev: State<u64> = ctx.use_state_keyed(&format!("{key}_ne_evrev"), || 0u64);

        // Install the double-click listener on the LOCAL `kit`
        // before it's moved into `Self`. `CanvasKit` is `#[derive(Clone)]`
        // with an owned `click_listeners: Vec<_>`, so installing on
        // a post-move clone would only mutate that clone's vec and
        // be dropped. The portal-ui click hook above uses the same
        // pre-move pattern.
        let kit_for_listener = kit.clone();
        let graph_for_listener = graph_arc.clone();
        let rects_for_listener = group_text_rects.clone();
        let last_click_for_listener = last_click.clone();
        let evt_queue_for_listener = events_queue.clone();
        let evt_rev_for_listener = events_rev.clone();
        kit.add_click_listener(move |evt| {
            let Some(region_id) = evt.region_id.as_ref() else {
                return;
            };
            // Filter to only the three region variants this listener
            // handles. RegionId::parse is the single source of truth
            // for the prefix-to-variant mapping — adding a new region
            // kind anywhere becomes a compile error here if the match
            // arms don't cover it.
            #[derive(Clone)]
            enum DoubleClickTarget {
                Node(NodeId),
                GroupTitle(GroupId),
                GroupDesc(GroupId),
            }
            let target = match RegionId::parse(region_id) {
                Some(RegionId::Node(id)) => DoubleClickTarget::Node(id),
                Some(RegionId::GroupTitle(id)) => DoubleClickTarget::GroupTitle(id),
                Some(RegionId::GroupDesc(id)) => DoubleClickTarget::GroupDesc(id),
                _ => return,
            };
            let now = web_time::Instant::now();
            // The double-click timestamp map is keyed on the raw
            // region string so the HashMap identity is stable across
            // clicks. The RegionId::parse front-door keeps the
            // typed enum honest while preserving the same HashMap
            // semantics.
            let is_double = {
                let mut last = last_click_for_listener.lock().unwrap();
                let matched = last
                    .as_ref()
                    .map(|(prev, t)| {
                        prev == region_id
                            && now.duration_since(*t) < std::time::Duration::from_millis(400)
                    })
                    .unwrap_or(false);
                if matched {
                    *last = None;
                    true
                } else {
                    *last = Some((region_id.clone(), now));
                    false
                }
            };
            if !is_double {
                return;
            }
            match target {
                // ── Node double-click — emit SubgraphRequested when
                // the node carries `subgraph_ref`. Plain node
                // double-clicks are intentionally ignored at the
                // editor layer (hosts wire their own "open inspector
                // / edit title" gesture via the events channel).
                DoubleClickTarget::Node(node_id) => {
                    let (subgraph_id, anchor_content) = {
                        let graph = graph_for_listener.read().unwrap();
                        let Some(node) = graph.nodes.iter().find(|n| n.id == node_id) else {
                            return;
                        };
                        let Some(sub_id) = node.subgraph_ref.clone() else {
                            return;
                        };
                        // Compute the node's centre in content space —
                        // hosts use it as a transition-from anchor
                        // when animating into the subgraph view.
                        let (w, h) = node.size.unwrap_or((180.0, 72.0));
                        let centre =
                            Point::new(node.position.x + w * 0.5, node.position.y + h * 0.5);
                        (sub_id, centre)
                    };
                    let anchor_screen = kit_for_listener.content_to_screen(anchor_content);
                    evt_queue_for_listener
                        .lock()
                        .unwrap()
                        .push(EditorEvent::SubgraphRequested {
                            subgraph_id,
                            source_node: node_id,
                            source_anchor: anchor_screen,
                        });
                    evt_rev_for_listener.update(|n| n.wrapping_add(1));
                    blinc_layout::request_redraw();
                    return;
                }
                DoubleClickTarget::GroupTitle(_) | DoubleClickTarget::GroupDesc(_) => {
                    // Fall through to the shared group-rect lookup
                    // below.
                }
            }
            // ── Group title / desc double-click — inline-editor path.
            let (group_id, is_title) = match &target {
                DoubleClickTarget::GroupTitle(g) => (g.clone(), true),
                DoubleClickTarget::GroupDesc(g) => (g.clone(), false),
                DoubleClickTarget::Node(_) => unreachable!("node arm returned above"),
            };
            let (current, anchor_content) = {
                let graph = graph_for_listener.read().unwrap();
                let Some(grp) = graph.groups.iter().find(|g| g.id == group_id) else {
                    return;
                };
                let current = if is_title {
                    grp.name.clone()
                } else {
                    grp.description.clone().unwrap_or_default()
                };
                let rects = rects_for_listener.lock().unwrap();
                let Some(anchor) = rects.get(region_id).copied() else {
                    return;
                };
                (current, anchor)
            };
            // Convert content-space rect → screen-space via kit's
            // viewport transform so the host can anchor an overlay
            // at the right pixel position.
            let tl = kit_for_listener
                .content_to_screen(Point::new(anchor_content.x(), anchor_content.y()));
            let br = kit_for_listener.content_to_screen(Point::new(
                anchor_content.x() + anchor_content.width(),
                anchor_content.y() + anchor_content.height(),
            ));
            let anchor_screen = Rect::new(
                tl.x.min(br.x),
                tl.y.min(br.y),
                (br.x - tl.x).abs(),
                (br.y - tl.y).abs(),
            );
            let event = if is_title {
                EditorEvent::EditGroupTitleRequested {
                    group: group_id,
                    current,
                    anchor_screen,
                }
            } else {
                EditorEvent::EditGroupDescriptionRequested {
                    group: group_id,
                    current,
                    anchor_screen,
                }
            };
            evt_queue_for_listener.lock().unwrap().push(event);
            evt_rev_for_listener.update(|n| n.wrapping_add(1));
            blinc_layout::request_redraw();
        });

        Self {
            kit,
            templates: Arc::new(RwLock::new(AHashMap::new())),
            graph: graph_arc,
            theme: Arc::new(RwLock::new(NodeEditorTheme::default())),
            theme_revision: Arc::new(AtomicU64::new(0)),
            node_slots: Arc::new(RwLock::new(AHashMap::new())),
            group_slots: Arc::new(RwLock::new(AHashMap::new())),
            layout_strategy: Arc::new(RwLock::new(LayoutStrategy::default())),
            graph_rev: ctx.use_state_keyed(&format!("{key}_ne_graph"), || 0u64),
            drag_state: ctx.use_state_keyed(&format!("{key}_ne_drag"), DragConnect::default),
            hover_state: ctx
                .use_state_keyed::<Option<HoverTarget>, _>(&format!("{key}_ne_hover"), || None),
            events_rev,
            events_queue,
            flashes: Arc::new(RwLock::new(AHashMap::new())),
            last_group_auto_bounds: Arc::new(RwLock::new(AHashMap::new())),
            drag_group_preview: Arc::new(RwLock::new(DragGroupPreview::default())),
            viewport_anim: Arc::new(Mutex::new(None)),
            viewport_anim_cb_id: Arc::new(Mutex::new(None)),
            last_screen_pos: Arc::new(RwLock::new(None)),
            on_validate: Arc::new(RwLock::new(None)),
            on_context_menu: Arc::new(RwLock::new(None)),
            portals: Arc::new(Mutex::new(blinc_portal_ui::PortalManager::new())),
            portal_content_heights: Arc::new(Mutex::new(AHashMap::new())),
            portal_content_widths: Arc::new(Mutex::new(AHashMap::new())),
            render_stats: Arc::new(RenderStatsCell::default()),
            group_text_rects,
            badge_rects,
            last_click,
            subgraphs: Arc::new(RwLock::new(AHashMap::new())),
            subgraph_rev: ctx.use_state_keyed(&format!("{key}_ne_subgraph"), || 0u64),
            key,
        }
    }

    /// Snapshot of the previous render frame's cull counters —
    /// `(visible, total)` for both nodes and edges. Hosts read this
    /// to surface a HUD that visually confirms the frustum
    /// cull is active: pan / zoom out and the visible counts drop
    /// while the totals stay constant. Returns zeroes before the
    /// first frame paints.
    pub fn last_render_stats(&self) -> RenderStats {
        self.render_stats.snapshot()
    }

    /// Patch `inputs.content_height` with the previous frame's
    /// portal-consumed height if it exceeds the template's declared
    /// minimum. The closure paints whatever it wants; this method
    /// is what makes the slot tree (and therefore the node bounds,
    /// the inset background, the portal clip, and the group
    /// auto-bounds) grow to fit on the next frame.
    fn apply_portal_height_override(
        &self,
        node_id: &NodeId,
        inputs: &mut crate::slot::NodeSlotInputs,
    ) {
        if let Some(stored) = self
            .portal_content_heights
            .lock()
            .unwrap()
            .get(node_id)
            .copied()
        {
            // The portal's `consumed_height` measures the closure's
            // emitted content. The slot's body region also needs
            // breathing room equal to the inner padding the renderer
            // adds in `content_slot_rects` (8px top + 8px bottom +
            // 2px / 10px outer pad) — round to 28px total so the
            // closure's content has the same visible margin it had
            // at the template's declared height.
            const OUTER_AND_INNER_PAD: f32 = 28.0;
            let needed = stored + OUTER_AND_INNER_PAD;
            if needed > inputs.content_height {
                inputs.content_height = needed;
            }
        }
    }

    /// Patch `inputs.width` with the previous frame's portal-consumed
    /// width when it exceeds the resolved width (theme default raised
    /// by `template.content.min_width` if any). Caller should gate on
    /// `instance.size.is_none()` — explicit host sizing must win.
    ///
    /// Horizontal counterpart of [`Self::apply_portal_height_override`].
    /// Same model: closure emits whatever it wants; this method grows
    /// the slot's overall width on the next frame so wide widgets
    /// stop clipping. Includes the renderer's per-side body padding
    /// from `content_slot_rects` (10 px outer pad + 8 px inner pad on
    /// each side = 36 px total) so the measured cursor extent still
    /// has the same margin to the chrome.
    fn apply_portal_width_override(
        &self,
        template: &crate::NodeTemplate<K>,
        node_id: &NodeId,
        inputs: &mut crate::slot::NodeSlotInputs,
    ) {
        if let Some(stored) = self
            .portal_content_widths
            .lock()
            .unwrap()
            .get(node_id)
            .copied()
        {
            const OUTER_AND_INNER_PAD_X: f32 = 36.0;
            let content_min_w = template
                .content
                .as_ref()
                .and_then(|c| c.min_width)
                .unwrap_or(0.0);
            let needed = (stored + OUTER_AND_INNER_PAD_X).max(content_min_w);
            if needed > inputs.width {
                inputs.width = needed;
            }
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    /// Replace the editor's template registry. Templates are keyed by
    /// `NodeTemplate::component`; instances reference them via that
    /// same field.
    pub fn with_templates(self, templates: impl IntoIterator<Item = NodeTemplate<K>>) -> Self {
        {
            let mut t = self.templates.write().unwrap();
            t.clear();
            for tpl in templates {
                t.insert(tpl.component.clone(), tpl);
            }
        }
        self
    }

    pub fn add_template(&self, template: NodeTemplate<K>) {
        self.templates
            .write()
            .unwrap()
            .insert(template.component.clone(), template);
    }

    /// Override the editor theme. Pass [`NodeEditorTheme::default()`]
    /// to inherit everything from `blinc_theme::ThemeState`.
    ///
    /// Bumps the theme-revision counter — existing slot-cache entries
    /// fingerprint stale and the next `set_graph` warms a fresh
    /// generation in parallel.
    pub fn with_theme(self, theme: NodeEditorTheme) -> Self {
        *self.theme.write().unwrap() = theme;
        self.theme_revision.fetch_add(1, Ordering::Relaxed);
        self
    }

    /// Set the auto-layout strategy. Default is
    /// [`LayoutStrategy::Manual`] — node positions come from the
    /// host's model.
    pub fn with_layout(self, strategy: LayoutStrategy) -> Self {
        *self.layout_strategy.write().unwrap() = strategy;
        self
    }

    /// Runtime sibling of [`Self::with_layout`] — swap the layout
    /// strategy on a live editor (e.g. when the user picks a new
    /// auto-layout from a menu). Pairs with [`Self::apply_layout`]
    /// or [`EditorCommand::ApplyLayout`] to actually run the swapped
    /// strategy and emit the resulting `LayoutApplied` event.
    pub fn set_layout_strategy(&self, strategy: LayoutStrategy) {
        *self.layout_strategy.write().unwrap() = strategy;
    }

    /// Set the canvas background (grid, dots, solid, etc.). Falls
    /// through to canvas_kit's default if not set.
    pub fn with_background(mut self, bg: CanvasBackground) -> Self {
        self.kit = self.kit.with_background(bg);
        self
    }

    /// Enable snap-to-grid at `spacing` content units. Active for
    /// every node-drag end and every `update_node_position` call:
    /// the position quantises to the nearest grid intersection
    /// before being committed. Pass `None` (or call
    /// [`Self::set_snap_enabled(false)`](Self::set_snap_enabled))
    /// to disable. Wraps [`CanvasKit::with_snap`] so the kit's own
    /// drag-snap path (marquee, etc.) sees the same spacing.
    pub fn with_snap(mut self, spacing: f32) -> Self {
        self.kit = self.kit.with_snap(spacing);
        self
    }

    /// Toggle snap-to-grid without rebuilding the editor. Pairs
    /// with [`Self::with_snap`] for hosts that wire a "snap" toggle
    /// into a toolbar.
    pub fn set_snap_enabled(&mut self, enabled: bool) {
        self.kit.set_snap_enabled(enabled);
    }

    /// True if snap-to-grid is currently active. Mirrors
    /// [`CanvasKit::snap_enabled`].
    pub fn snap_enabled(&self) -> bool {
        self.kit.snap_enabled()
    }

    /// Quantise `pt` to the active snap grid. Returns `pt` unchanged
    /// when snap is disabled. Useful for hosts that want to preview
    /// the snapped position during a drag.
    pub fn snap_point(&self, pt: Point) -> Point {
        self.kit.snap_point(pt)
    }

    /// Borrow the underlying [`CanvasKit`] for advanced configuration
    /// (zoom controller, snap, marquee tool selection, etc.).
    pub fn canvas_kit(&self) -> &CanvasKit {
        &self.kit
    }

    // ─── Graph sync ──────────────────────────────────────────────────

    /// Replace the editor's graph wholesale. The editor is a view —
    /// hosts always call this after handling a mutation request, even
    /// if they only changed one node.
    ///
    /// Triggers an off-render warm of the slot cache for any new node
    /// / group fingerprints. The warm runs serially for small sets
    /// and parallelises via rayon above the threshold (see
    /// [`crate::slot::PARALLEL_THRESHOLD`]).
    pub fn set_graph(
        &self,
        nodes: Vec<NodeInstance<N>>,
        connections: Vec<Connection<C>>,
        groups: Vec<Group<G>>,
        exposed: Vec<ExposedPort<K>>,
    ) {
        {
            let mut g = self.graph.write().unwrap();
            g.nodes = nodes;
            g.connections = connections;
            g.groups = groups;
            g.exposed = exposed;
        }
        self.warm_slot_cache();
        self.bump_graph_rev();
    }

    // ─── Subgraph storage ────────────────────────────────────────────
    //
    // Named subgraphs stored alongside the active graph. The editor
    // never auto-navigates into a subgraph — it emits
    // `EditorEvent::SubgraphRequested` on double-click of a
    // subgraph-ref node and hosts decide the UI (modal / side sheet /
    // route navigation / tab / etc.).

    /// Bumps every time a subgraph is created, removed, renamed, or
    /// its contents mutated via [`Self::with_subgraph_graph_mut`].
    /// Pair with `derive` / `effect_with_deps` to keep host UI in
    /// sync with stored subgraphs.
    pub fn subgraph_signal(&self) -> SignalId {
        self.subgraph_rev.signal_id()
    }

    /// Increment + notify `subgraph_rev`. Centralised so every
    /// subgraph-storage mutation bumps the same counter.
    fn bump_subgraph_rev(&self) {
        let next = self.subgraph_rev.try_get().unwrap_or(0).wrapping_add(1);
        self.subgraph_rev.set(next);
    }

    /// Create an empty subgraph stored under `id`. `namespace`
    /// defaults to `id` — typical Zeal-style format is
    /// `"workflow/subgraph_id"` for cross-workflow uniqueness;
    /// override via [`Self::set_subgraph_namespace`]. Overwrites
    /// any existing subgraph with the same id (callers wanting
    /// "create-only" should check [`Self::has_subgraph`] first).
    pub fn create_subgraph(
        &self,
        id: impl Into<crate::subgraph::SubgraphId>,
        name: impl Into<String>,
    ) -> crate::subgraph::SubgraphId {
        let id = id.into();
        let name = name.into();
        {
            let mut subs = self.subgraphs.write().unwrap();
            subs.insert(id.clone(), crate::subgraph::Subgraph::new(id.clone(), name));
        }
        self.bump_subgraph_rev();
        id
    }

    /// Drop the subgraph with `id` from storage. NO-op if absent.
    /// Note: any [`NodeInstance::subgraph_ref`] still pointing to
    /// `id` becomes dangling — its double-click will still emit
    /// `SubgraphRequested`, but [`Self::subgraph`] returns `None`
    /// and hosts should treat that as "subgraph missing." Hosts
    /// typically clear the orphaned refs separately or accept the
    /// dangling state.
    pub fn delete_subgraph(&self, id: &crate::subgraph::SubgraphId) -> bool {
        let removed = {
            let mut subs = self.subgraphs.write().unwrap();
            subs.remove(id).is_some()
        };
        if removed {
            self.bump_subgraph_rev();
        }
        removed
    }

    /// Rename a subgraph in place. NO-op if `id` doesn't exist.
    pub fn rename_subgraph(
        &self,
        id: &crate::subgraph::SubgraphId,
        name: impl Into<String>,
    ) -> bool {
        let renamed = {
            let mut subs = self.subgraphs.write().unwrap();
            if let Some(sub) = subs.get_mut(id) {
                sub.name = name.into();
                true
            } else {
                false
            }
        };
        if renamed {
            self.bump_subgraph_rev();
        }
        renamed
    }

    /// Override the display namespace for a subgraph (the subtitle
    /// shown on subgraph-ref nodes in the parent canvas). NO-op if
    /// `id` doesn't exist.
    pub fn set_subgraph_namespace(
        &self,
        id: &crate::subgraph::SubgraphId,
        namespace: impl Into<String>,
    ) -> bool {
        let changed = {
            let mut subs = self.subgraphs.write().unwrap();
            if let Some(sub) = subs.get_mut(id) {
                sub.namespace = namespace.into();
                true
            } else {
                false
            }
        };
        if changed {
            self.bump_subgraph_rev();
        }
        changed
    }

    /// True iff a subgraph with `id` is currently stored.
    pub fn has_subgraph(&self, id: &crate::subgraph::SubgraphId) -> bool {
        self.subgraphs.read().unwrap().contains_key(id)
    }

    /// Read-only snapshot of a stored subgraph (cloned). Returns
    /// `None` if the id isn't registered. Hosts typically call this
    /// inside an [`crate::event::EditorEvent::SubgraphRequested`]
    /// handler to fetch the subgraph's contents, then pass them to
    /// [`Self::set_graph`] to switch the view.
    pub fn subgraph(
        &self,
        id: &crate::subgraph::SubgraphId,
    ) -> Option<crate::subgraph::Subgraph<K, N, C, G>>
    where
        N: Clone,
        C: Clone,
        G: Clone,
    {
        self.subgraphs.read().unwrap().get(id).cloned()
    }

    /// Snapshot of every stored subgraph's `(id, name, namespace)`
    /// — useful for hosts rendering a workspace tab bar or a
    /// breadcrumb dropdown without holding the read lock.
    pub fn subgraph_summaries(&self) -> Vec<(crate::subgraph::SubgraphId, String, String)> {
        self.subgraphs
            .read()
            .unwrap()
            .values()
            .map(|s| (s.id.clone(), s.name.clone(), s.namespace.clone()))
            .collect()
    }

    /// Read-only access to a stored subgraph via a closure (no
    /// clone). Returns `None` from the closure when the id is
    /// absent.
    pub fn with_subgraph<R, F>(&self, id: &crate::subgraph::SubgraphId, f: F) -> Option<R>
    where
        F: FnOnce(&crate::subgraph::Subgraph<K, N, C, G>) -> R,
    {
        let subs = self.subgraphs.read().unwrap();
        subs.get(id).map(f)
    }

    /// Mutable access to a stored subgraph's interior via a
    /// closure. Use to edit `nodes` / `connections` / `groups` /
    /// `exposed` in place. Returns `None` from the closure when
    /// the id is absent. Bumps `subgraph_rev` on any access (the
    /// closure was given write access — assume something changed
    /// even if it didn't actually mutate, otherwise callers have
    /// to plumb a "did change" flag through every closure they
    /// write).
    pub fn with_subgraph_graph_mut<R, F>(&self, id: &crate::subgraph::SubgraphId, f: F) -> Option<R>
    where
        F: FnOnce(&mut crate::subgraph::Subgraph<K, N, C, G>) -> R,
    {
        let result = {
            let mut subs = self.subgraphs.write().unwrap();
            subs.get_mut(id).map(f)
        };
        if result.is_some() {
            self.bump_subgraph_rev();
        }
        result
    }

    /// Place a NEW subgraph-reference node in the active graph at
    /// `position` that points to subgraph `id`. The node is forced
    /// to [`crate::NodeShape::Diamond`] by the renderer; subtitle
    /// defaults to the subgraph's namespace.
    ///
    /// Returns the new node's id. Hosts may insert their own
    /// subgraph-ref nodes manually via [`Self::insert_node`] with
    /// [`NodeInstance::with_subgraph_ref`] — this helper is a
    /// convenience for the common case where the host doesn't
    /// already have a node id in mind.
    pub fn instantiate_subgraph(
        &self,
        id: &crate::subgraph::SubgraphId,
        position: blinc_core::layer::Point,
    ) -> Option<NodeId>
    where
        N: Default,
    {
        // Lookup namespace + name under a read lock — needed for the
        // node's subtitle.
        let (namespace, _name) = {
            let subs = self.subgraphs.read().unwrap();
            let sub = subs.get(id)?;
            (sub.namespace.clone(), sub.name.clone())
        };

        // Generate a unique node id. Match the pattern Zeal uses —
        // a typed prefix + monotonic-ish suffix. Hosts that want a
        // specific id should call `insert_node` directly.
        let node_id = NodeId::from(format!(
            "subgraph_{}_{}",
            id.as_str(),
            self.subgraph_rev.try_get().unwrap_or(0).wrapping_add(1),
        ));

        // Templates for subgraph-ref nodes are host-supplied. We
        // pick a sentinel component string `"__subgraph"` that hosts
        // can register; if no template is registered the renderer
        // falls back to a minimal diamond.
        let node: NodeInstance<N> = NodeInstance::new(node_id.clone(), "__subgraph", position)
            .with_shape(crate::node::NodeShape::Diamond)
            .with_subtitle(namespace)
            .with_subgraph_ref(id.clone());
        self.insert_node(node);
        Some(node_id)
    }

    /// Programmatic navigation request — emits
    /// [`EditorEvent::SubgraphRequested`] for `source_node` (which
    /// must be a [`NodeInstance`] with `subgraph_ref` set). NO-op if
    /// the node doesn't exist or carries no subgraph reference.
    /// Useful when the host wants to drive navigation from outside
    /// the canvas (a button in a sidebar, an `Enter` keystroke
    /// handler the host owns, etc.).
    pub fn request_subgraph_open(&self, source_node: &NodeId) {
        let (subgraph_id, anchor) = {
            let graph = self.graph.read().unwrap();
            let Some(node) = graph.nodes.iter().find(|n| n.id == *source_node) else {
                return;
            };
            let Some(sub) = node.subgraph_ref.clone() else {
                return;
            };
            let (w, h) = node.size.unwrap_or((180.0, 72.0));
            let centre =
                blinc_core::layer::Point::new(node.position.x + w * 0.5, node.position.y + h * 0.5);
            (sub, self.kit.content_to_screen(centre))
        };
        self.events_queue
            .lock()
            .unwrap()
            .push(EditorEvent::SubgraphRequested {
                subgraph_id,
                source_node: source_node.clone(),
                source_anchor: anchor,
            });
        let next = self.events_rev.try_get().unwrap_or(0).wrapping_add(1);
        self.events_rev.set(next);
        blinc_layout::request_redraw();
    }

    // ─── Granular graph mutations ────────────────────────────────────
    //
    // Each method holds the graph write lock for the minimum window,
    // mutates in place, releases the lock, then warms the slot cache
    // for any new fingerprints and bumps `graph_rev`. Hosts driving
    // single-entity updates pay an O(touched-fingerprints) warm
    // instead of the O(all-nodes + all-groups) walk `set_graph`
    // triggers.

    /// Append a node to the graph. Duplicate `id` overwrites the
    /// existing instance (host's responsibility to enforce uniqueness
    /// at insertion time if that matters).
    pub fn insert_node(&self, node: NodeInstance<N>) {
        {
            let mut g = self.graph.write().unwrap();
            if let Some(existing) = g.nodes.iter_mut().find(|n| n.id == node.id) {
                *existing = node;
            } else {
                g.nodes.push(node);
            }
        }
        self.warm_slot_cache();
        self.bump_graph_rev();
    }

    /// Remove a node and every connection incident to it.
    pub fn remove_node(&self, id: &NodeId) {
        {
            let mut g = self.graph.write().unwrap();
            g.nodes.retain(|n| n.id != *id);
            g.connections
                .retain(|c| c.from.node != *id && c.to.node != *id);
            for group in &mut g.groups {
                group.members.retain(|m| m != id);
            }
        }
        self.bump_graph_rev();
    }

    /// Move a node to an absolute content-space position. Used by
    /// hosts that compute layouts programmatically; for drag input
    /// the editor handles in-place updates internally.
    pub fn update_node_position(&self, id: &NodeId, position: Point) {
        // Quantise to the active snap grid if enabled. No-op when
        // snap is off — `snap_point` returns the input unchanged.
        let position = self.kit.snap_point(position);
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(n) = g.nodes.iter_mut().find(|n| n.id == *id) {
                if n.position == position {
                    false
                } else {
                    n.position = position;
                    true
                }
            } else {
                false
            }
        };
        if changed {
            self.bump_graph_rev();
        }
    }

    // ─── Node config mutation ────────────────────────────────────────

    /// Patch a single config field on a node. Runs the template's
    /// [`PropertyRule`](crate::config::PropertyRule) cascade so any
    /// dependent fields update atomically, emits one
    /// [`EditorEvent::NodeConfigChanged`] per actually-changed key
    /// (initial patch + every cascading effect, with `from_rule`
    /// distinguishing the two), and bumps the graph signal once for
    /// the whole atomic update.
    ///
    /// Returns the post-cascade validation issues
    /// ([`crate::config::validate`] result). The patch is applied
    /// regardless of validation outcome — hosts decide whether to
    /// surface, accept, or roll back based on the returned issues.
    ///
    /// No-ops + returns `Vec::new()` when:
    /// * The node id isn't present in the graph.
    /// * The new value equals the existing value AND no rule
    ///   cascades fire.
    pub fn patch_node_config(
        &self,
        id: &NodeId,
        key: &str,
        value: serde_json::Value,
    ) -> Vec<crate::config::ValidationIssue> {
        self.update_node_config_inner(id, |config| {
            // Apply primary write; the inspector apply_patch helper
            // handles null-promote + remove-on-null semantics.
            crate::inspector::apply_patch(
                config,
                &crate::inspector::InspectorPatchRequest {
                    node: id.clone(),
                    path: key.to_string(),
                    value: value.clone(),
                },
            );
            vec![key.to_string()]
        })
    }

    /// Replace a node's entire config object. Same cascade + event +
    /// validation semantics as
    /// [`patch_node_config`](Self::patch_node_config) — every key
    /// whose value shifts between previous and new (including
    /// cascading effects) fires its own `NodeConfigChanged` event.
    pub fn set_node_config(
        &self,
        id: &NodeId,
        new_config: serde_json::Value,
    ) -> Vec<crate::config::ValidationIssue> {
        self.update_node_config_inner(id, |config| {
            // Diff prev vs next at the key level. A rule with an
            // empty `when` clause would otherwise re-fire on every
            // set_node_config call for keys whose value didn't
            // change between the two configs.
            let prev_obj = config.as_object();
            let next_obj = new_config.as_object();
            let mut keys: std::collections::HashSet<&str> = std::collections::HashSet::new();
            if let Some(o) = prev_obj {
                keys.extend(o.keys().map(String::as_str));
            }
            if let Some(o) = next_obj {
                keys.extend(o.keys().map(String::as_str));
            }
            let mut touched: Vec<String> = keys
                .into_iter()
                .filter(|k| {
                    let p = prev_obj.and_then(|o| o.get(*k));
                    let n = next_obj.and_then(|o| o.get(*k));
                    p != n
                })
                .map(|s| s.to_string())
                .collect();
            touched.sort(); // deterministic order for downstream tests
            *config = new_config;
            touched
        })
    }

    /// Forward an [`InspectorPatchRequest`](crate::inspector::InspectorPatchRequest)
    /// straight into [`patch_node_config`](Self::patch_node_config).
    /// Convenience for hosts that drain inspector events directly
    /// into the editor.
    pub fn apply_inspector_patch(
        &self,
        request: &crate::inspector::InspectorPatchRequest,
    ) -> Vec<crate::config::ValidationIssue> {
        self.patch_node_config(&request.node, &request.path, request.value.clone())
    }

    /// Shared core of `patch_node_config` / `set_node_config`. The
    /// `mutate` closure performs the primary write on the node's
    /// config and returns the list of keys it touched — those keys
    /// seed the rule cascade's initial trigger set.
    fn update_node_config_inner(
        &self,
        id: &NodeId,
        mutate: impl FnOnce(&mut serde_json::Value) -> Vec<String>,
    ) -> Vec<crate::config::ValidationIssue> {
        // Schema clone runs under the templates read lock only — that
        // lock is independent of the graph lock, so no order-of-
        // acquisition concern.
        let schema = {
            let templates = self.templates.read().unwrap();
            let component = {
                let g = self.graph.read().unwrap();
                g.nodes
                    .iter()
                    .find(|n| n.id == *id)
                    .map(|n| n.component.clone())
            };
            match component {
                Some(c) => templates
                    .get(&c)
                    .map(|t| t.config_schema.clone())
                    .unwrap_or_default(),
                None => return Vec::new(),
            }
        };

        // Hold a single graph.write() across snapshot → mutate →
        // cascade → commit. Splitting it into multiple acquisitions
        // would let a concurrent `patch_node_config` on the same
        // node from another thread write between the snapshot and
        // the commit and lose updates / mis-emit `previous` values.
        // The editor is Send+Sync (EditorCommand dispatches from
        // any thread), so the race is API-permitted.
        //
        // Cascade depth is bounded by MAX_RULE_CASCADE_DEPTH (16),
        // and per-tick predicate evaluation is O(rules), so the
        // critical section stays short even under hostile rule
        // sets.
        let issues;
        let events: Vec<EditorEvent<K>>;
        {
            let mut g = self.graph.write().unwrap();
            let Some(node) = g.nodes.iter_mut().find(|n| n.id == *id) else {
                return Vec::new();
            };

            // Three snapshots so cascade events can quote the
            // post-primary state as their `previous` (matching
            // `NodeConfigChanged`'s "each step separately" contract
            // at event.rs:309-310). Without the mid-state capture
            // hosts can't distinguish "user wrote X" from "rule
            // overrode X to Y": both events would carry the same
            // pre-everything `previous`.
            let pre_config = node.config.clone();
            let mut working = pre_config.clone();
            let primary_keys = mutate(&mut working);
            let post_primary = working.clone();
            let applied = if schema.rules.is_empty() {
                Vec::new()
            } else {
                crate::config::cascade_rules(&schema.rules, &mut working, &primary_keys)
            };

            // Diff three snapshots into per-step events.
            // 1. Primary-write events: pre → post_primary, from_rule=false.
            // 2. Cascade-effect events: post_primary → post-cascade,
            //    from_rule=true. A key that was both primary-written
            //    AND cascade-overridden emits TWO events — first
            //    the user-intent write, then the rule override.
            let mut emitted: Vec<EditorEvent<K>> = Vec::new();
            let pre_obj = pre_config.as_object();
            let mid_obj = post_primary.as_object();
            let final_obj = working.as_object();
            for key in &primary_keys {
                let prev_value = pre_obj
                    .and_then(|o| o.get(key))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let new_value = mid_obj
                    .and_then(|o| o.get(key))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if prev_value != new_value {
                    emitted.push(EditorEvent::NodeConfigChanged {
                        node: id.clone(),
                        key: key.clone(),
                        previous: prev_value,
                        value: new_value,
                        from_rule: false,
                    });
                }
            }
            for effect in &applied {
                let key = effect.key().to_string();
                let prev_value = mid_obj
                    .and_then(|o| o.get(&key))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let new_value = final_obj
                    .and_then(|o| o.get(&key))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if prev_value != new_value {
                    emitted.push(EditorEvent::NodeConfigChanged {
                        node: id.clone(),
                        key,
                        previous: prev_value,
                        value: new_value,
                        from_rule: true,
                    });
                }
            }

            // Commit the final config + run validation under the
            // same lock so concurrent readers see the post-commit
            // state, not the post-primary intermediate.
            node.config = working.clone();
            issues = crate::config::validate(&schema, &working);
            events = emitted;
        }

        // 5. Emit events + bump signal. Only bump when something
        //    actually changed.
        if !events.is_empty() {
            for evt in events {
                self.push_event(evt);
            }
            self.bump_graph_rev();
        }

        issues
    }

    /// Append a connection. Duplicate `id` overwrites.
    pub fn insert_connection(&self, conn: Connection<C>) {
        {
            let mut g = self.graph.write().unwrap();
            if let Some(existing) = g.connections.iter_mut().find(|c| c.id == conn.id) {
                *existing = conn;
            } else {
                g.connections.push(conn);
            }
        }
        self.bump_graph_rev();
    }

    /// Remove a connection by id.
    pub fn remove_connection(&self, id: ConnectionId) {
        let removed = {
            let mut g = self.graph.write().unwrap();
            let before = g.connections.len();
            g.connections.retain(|c| c.id != id);
            g.connections.len() != before
        };
        if removed {
            self.bump_graph_rev();
        }
    }

    /// Append a group. Duplicate `id` overwrites.
    pub fn insert_group(&self, group: Group<G>) {
        {
            let mut g = self.graph.write().unwrap();
            if let Some(existing) = g.groups.iter_mut().find(|gr| gr.id == group.id) {
                *existing = group;
            } else {
                g.groups.push(group);
            }
        }
        self.warm_slot_cache();
        self.bump_graph_rev();
    }

    /// Remove a group (member nodes stay; they just lose their
    /// grouping).
    pub fn remove_group(&self, id: &GroupId) {
        {
            let mut g = self.graph.write().unwrap();
            g.groups.retain(|gr| gr.id != *id);
        }
        self.bump_graph_rev();
    }

    /// Replace a group's member list. No-op if the group is missing.
    pub fn set_group_members(&self, id: &GroupId, members: Vec<NodeId>) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(group) = g.groups.iter_mut().find(|gr| gr.id == *id) {
                if group.members != members {
                    group.members = members;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.warm_slot_cache();
            self.bump_graph_rev();
        }
    }

    /// Set a group's collapsed state. No-op if state already matches.
    pub fn set_group_collapsed(&self, id: &GroupId, collapsed: bool) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(group) = g.groups.iter_mut().find(|gr| gr.id == *id) {
                if group.is_collapsed != collapsed {
                    group.is_collapsed = collapsed;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.warm_slot_cache();
            self.bump_graph_rev();
        }
    }

    // ─── Selection commands ──────────────────────────────────────────
    //
    // Thin wrappers over `kit.set_selection` that translate node-id
    // arguments into the canvas-kit region-id string convention
    // (`"node:{id}"`). Hosts that prefer to drive selection directly
    // can keep using `editor.canvas_kit().set_selection(...)`.

    /// Replace the selection with the given node ids.
    pub fn select(&self, ids: &[NodeId]) {
        let set: HashSet<String> = ids
            .iter()
            .map(|id| RegionId::Node(id.clone()).encode())
            .collect();
        self.kit.set_selection(set);
    }

    /// Select exactly one node (replaces any previous selection).
    pub fn select_one(&self, id: &NodeId) {
        self.select(std::slice::from_ref(id));
    }

    /// Add a node to the current selection (does not clear).
    /// Idempotent — no signal bump if the node is already selected.
    /// Uses canvas-kit's primitive `add_selection` so the path avoids
    /// the clone-mutate-replace round-trip the host would otherwise
    /// pay via `set_selection`.
    pub fn add_to_selection(&self, id: &NodeId) {
        self.kit.add_selection(RegionId::Node(id.clone()).encode());
    }

    /// Clear all selection.
    pub fn clear_selection(&self) {
        self.kit.clear_selection();
    }

    /// Currently-selected node ids (extracted from canvas-kit's
    /// `node:`-prefixed region selection).
    pub fn selected_node_ids(&self) -> Vec<NodeId> {
        self.kit
            .selection()
            .selected
            .iter()
            .filter_map(|s| match RegionId::parse(s)? {
                RegionId::Node(id) => Some(id),
                _ => None,
            })
            .collect()
    }

    /// Currently-selected connection ids (extracted from
    /// `edge:`-prefixed regions).
    pub fn selected_connection_ids(&self) -> Vec<ConnectionId> {
        self.kit
            .selection()
            .selected
            .iter()
            .filter_map(|s| match RegionId::parse(s)? {
                RegionId::Edge(id) => Some(id),
                _ => None,
            })
            .collect()
    }

    // ─── Viewport commands ───────────────────────────────────────────

    /// Pan + zoom so the given node sits at the canvas centre. Uses
    /// the canvas's last-known screen bounds; pre-first-frame this
    /// just sets pan with the current zoom.
    pub fn focus_on_node(&self, id: &NodeId) {
        let Some((centre, _size)) = self.node_centre(id) else {
            return;
        };
        let (sw, sh) = self.kit.screen_bounds();
        let z = self.kit.viewport().zoom;
        // Pan in CONTENT units (see ViewportAnimation comment +
        // CanvasViewport::pan_by). pan = S/zoom - centre puts
        // `centre` at screen midpoint S.
        let target_pan_x = sw * 0.5 / z - centre.x;
        let target_pan_y = sh * 0.5 / z - centre.y;
        self.animate_viewport_to(target_pan_x, target_pan_y, z, VIEWPORT_TWEEN_MS);
    }

    /// Pan + zoom so every node fits the viewport with a small
    /// margin. No-op when the graph is empty or screen bounds aren't
    /// known.
    pub fn zoom_to_fit(&self) {
        // Read the live theme so any host overrides (custom node
        // sizes, padding, badge dimensions) are honoured by the bbox
        // math. Using `NodeEditorTheme::default()` here under-fits
        // when the host enlarged nodes via `with_theme(...)`.
        let theme_overrides = self.theme.read().unwrap();
        let resolver = ThemeResolver::new(&theme_overrides);
        let Some(bounds) = self.union_of_nodes(|_| true, &resolver) else {
            return;
        };
        drop(theme_overrides);
        self.fit_rect_to_viewport(bounds);
    }

    /// Like [`zoom_to_fit`](Self::zoom_to_fit) but only frames
    /// currently-selected nodes.
    pub fn zoom_to_selection(&self) {
        let selected: HashSet<NodeId> = self.selected_node_ids().into_iter().collect();
        if selected.is_empty() {
            return;
        }
        // Same live-theme rationale as `zoom_to_fit`.
        let theme_overrides = self.theme.read().unwrap();
        let resolver = ThemeResolver::new(&theme_overrides);
        let Some(bounds) = self.union_of_nodes(|id| selected.contains(id), &resolver) else {
            return;
        };
        drop(theme_overrides);
        self.fit_rect_to_viewport(bounds);
    }

    // ─── Search ──────────────────────────────────────────────────────

    /// Case-insensitive substring search over the active graph and
    /// the editor's stored subgraphs. Matches against:
    ///
    /// * **Unique ids** — `NodeId`, `GroupId`, `SubgraphId`.
    /// * **Port ids** — each input + output `PortId` on every node's
    ///   template.
    /// * **Title / description** — node template `display_name` and
    ///   `subtitle`, instance `subtitle`, group `name` and
    ///   `description`, subgraph `name` and `namespace`.
    ///
    /// Subgraph matches are projected through to any diamond
    /// `subgraph_ref` instance in the active graph (the focusable
    /// canvas entity for that subgraph); a subgraph with no diamond
    /// reference on the active canvas is silently dropped from the
    /// result set.
    ///
    /// Hits are returned in graph iteration order — nodes first,
    /// then groups, then subgraph-projected nodes — so the host's
    /// result list has a stable, predictable ordering. Empty /
    /// whitespace-only query short-circuits to an empty vec, so the
    /// host can treat that as "no active search."
    pub fn search(&self, query: &str) -> Vec<SearchHit> {
        let q = query.trim();
        if q.is_empty() {
            return Vec::new();
        }
        let q_lower = q.to_lowercase();
        let contains = |s: &str| s.to_lowercase().contains(&q_lower);

        let graph = self.graph.read().unwrap();
        let templates = self.templates.read().unwrap();
        let subgraphs = self.subgraphs.read().unwrap();

        let mut hits: Vec<SearchHit> = Vec::new();

        // Nodes — id, instance subtitle, template display_name +
        // subtitle, port ids on the node's template.
        for n in &graph.nodes {
            let mut matched = contains(n.id.as_str());
            if !matched {
                if let Some(s) = &n.subtitle {
                    matched = contains(s);
                }
            }
            if !matched {
                if let Some(t) = templates.get(&n.component) {
                    matched = contains(&t.display_name)
                        || t.subtitle.as_ref().is_some_and(|s| contains(s));
                    if !matched {
                        for p in t.inputs.iter().chain(t.outputs.iter()) {
                            if contains(p.id.as_str()) {
                                matched = true;
                                break;
                            }
                        }
                    }
                }
            }
            if matched {
                hits.push(SearchHit::Node(n.id.clone()));
            }
        }

        // Groups — id, name, description.
        for g in &graph.groups {
            let matched = contains(g.id.as_str())
                || contains(&g.name)
                || g.description.as_ref().is_some_and(|d| contains(d));
            if matched {
                hits.push(SearchHit::Group(g.id.clone()));
            }
        }

        // Subgraphs — id, name, namespace. Project each match to the
        // diamond node in the active graph that references it. A
        // subgraph with no reference on the active canvas has nothing
        // to focus, so it's intentionally elided from the hit list.
        for sub in subgraphs.values() {
            let matched =
                contains(sub.id.as_str()) || contains(&sub.name) || contains(&sub.namespace);
            if !matched {
                continue;
            }
            for n in &graph.nodes {
                if n.subgraph_ref.as_ref() == Some(&sub.id) {
                    let hit = SearchHit::Node(n.id.clone());
                    if !hits.contains(&hit) {
                        hits.push(hit);
                    }
                }
            }
        }

        hits
    }

    /// One-shot search + viewport focus convenience for hosts wiring
    /// a search box. Pipes the result of [`Self::search`] into a
    /// selection + viewport command per the result count:
    ///
    /// * **0 hits** — clears selection; viewport untouched.
    /// * **1 hit** — selects + zooms tight on the single match (the
    ///   underlying [`Self::zoom_to_selection`] frames a single node
    ///   or group's members within the viewport).
    /// * **N > 1** — selects every match + zooms out to fit the union
    ///   of their bounds inside the viewport.
    ///
    /// Group hits put both the group's outline region (`group:<id>`)
    /// AND every member's `node:<id>` into the selection set so the
    /// member nodes' outlines join the group border AND the bounds
    /// union picks up the members' rects. Node hits add only
    /// `node:<id>` to the selection.
    ///
    /// Returns the same hit list `search` returned so the host can
    /// show the result count, group results by kind, or surface a
    /// "no matches" toast on an empty result.
    pub fn search_and_focus(&self, query: &str) -> Vec<SearchHit> {
        let hits = self.search(query);

        let mut selection: HashSet<String> = HashSet::new();
        {
            let graph = self.graph.read().unwrap();
            for h in &hits {
                match h {
                    SearchHit::Node(id) => {
                        selection.insert(RegionId::Node(id.clone()).encode());
                    }
                    SearchHit::Group(gid) => {
                        selection.insert(RegionId::Group(gid.clone()).encode());
                        if let Some(g) = graph.groups.iter().find(|gg| gg.id == *gid) {
                            for m in &g.members {
                                selection.insert(RegionId::Node(m.clone()).encode());
                            }
                        }
                    }
                }
            }
        }

        // Decide what drives the zoom-target set, separately from the
        // selection-outline set. Rule:
        //   • If there is at least one direct Node hit, those nodes
        //     are the zoom targets. Group hits in the same query are
        //     treated as outline-only — their borders still tint, but
        //     their members do NOT pull the camera. Otherwise a node
        //     like "Formatter" gets dragged into a group-wide framing
        //     because the group's description happens to contain the
        //     query (e.g. "Filter + formatter") and the user-typed
        //     term was a node-name intent.
        //   • If there are only Group hits (no Node hits), zoom uses
        //     the union of every matched group's members so the
        //     camera at least frames the matched group's contents.
        let primary_node_ids: HashSet<NodeId> = hits
            .iter()
            .filter_map(|h| match h {
                SearchHit::Node(id) => Some(id.clone()),
                SearchHit::Group(_) => None,
            })
            .collect();
        let zoom_node_ids: HashSet<NodeId> = if !primary_node_ids.is_empty() {
            primary_node_ids
        } else {
            let graph = self.graph.read().unwrap();
            hits.iter()
                .filter_map(|h| match h {
                    SearchHit::Group(gid) => Some(gid.clone()),
                    _ => None,
                })
                .flat_map(|gid| {
                    graph
                        .groups
                        .iter()
                        .find(|gg| gg.id == gid)
                        .map(|g| g.members.clone())
                        .unwrap_or_default()
                })
                .collect()
        };

        tracing::info!(
            target: "blinc_node_editor::search",
            query = %query,
            hit_count = hits.len(),
            zoom_node_count = zoom_node_ids.len(),
            hits = ?hits.iter().map(|h| match h {
                SearchHit::Node(id) => RegionId::Node(id.clone()).encode(),
                SearchHit::Group(id) => RegionId::Group(id.clone()).encode(),
            }).collect::<Vec<_>>(),
            "search_and_focus dispatch"
        );

        if selection.is_empty() {
            self.clear_selection();
            return hits;
        }
        self.kit.set_selection(selection);

        // Viewport policy keyed on the ZOOM-TARGET count, not hit count:
        //   • 1 zoom-target  → pan + zoom in capped at the single-match
        //                      ceiling so a small ~200×100 node doesn't
        //                      blow up to a near-fullscreen interior shot
        //                      (which reads as "zoomed into empty space"
        //                      — body fill IS most of what's left at 5×).
        //   • N > 1          → pan to centre of the union + zoom OUT only.
        //                      Never zoom in for multi-match; clustered
        //                      hits should not yank the user into a
        //                      tighter view than they had.
        if !zoom_node_ids.is_empty() {
            let theme_overrides = self.theme.read().unwrap();
            let resolver = ThemeResolver::new(&theme_overrides);
            if let Some(bounds) = self.union_of_nodes(|id| zoom_node_ids.contains(id), &resolver) {
                drop(theme_overrides);
                if zoom_node_ids.len() == 1 {
                    /// Single-match cap chosen empirically: small nodes
                    /// (~180–220 px wide) become comfortably readable
                    /// at 1.5× without dominating the viewport. Larger
                    /// matches just fit at < 1.5× anyway.
                    const SEARCH_SINGLE_MATCH_MAX_ZOOM: f32 = 1.5;
                    self.fit_rect_to_viewport_capped(bounds, SEARCH_SINGLE_MATCH_MAX_ZOOM);
                } else {
                    self.fit_rect_to_viewport_zoom_out_only(bounds);
                }
            }
        }

        hits
    }

    /// Pan + zoom the viewport to frame a group's full member union.
    /// Counterpart to [`Self::focus_on_node`] for hosts that want a
    /// "zoom to this group" affordance (right-click menu, palette
    /// command). Falls back silently when the group isn't in the
    /// current graph or has no members yet (a freshly-created empty
    /// group has nothing to frame).
    pub fn focus_on_group(&self, id: &GroupId) {
        let members: HashSet<NodeId> = {
            let graph = self.graph.read().unwrap();
            let Some(group) = graph.groups.iter().find(|g| g.id == *id) else {
                return;
            };
            group.members.iter().cloned().collect()
        };
        if members.is_empty() {
            return;
        }
        let theme_overrides = self.theme.read().unwrap();
        let resolver = ThemeResolver::new(&theme_overrides);
        let Some(bounds) = self.union_of_nodes(|nid| members.contains(nid), &resolver) else {
            return;
        };
        drop(theme_overrides);
        self.fit_rect_to_viewport(bounds);
    }

    /// Read-only accessor for a node's `disabled` flag. Returns
    /// `None` when the id isn't in the current graph. Lets hosts
    /// implement "toggle disable" affordances against the editor's
    /// authoritative state instead of mirroring it.
    pub fn is_node_disabled(&self, id: &NodeId) -> Option<bool> {
        self.graph
            .read()
            .ok()?
            .nodes
            .iter()
            .find(|n| n.id == *id)
            .map(|n| n.disabled)
    }

    /// Counterpart of [`Self::is_node_disabled`] for groups. Returns
    /// the group's own `disabled` flag — does NOT walk up parent
    /// groups (a node inside a disabled group reports `false` here
    /// unless it carries its own disabled flag).
    pub fn is_group_disabled(&self, id: &GroupId) -> Option<bool> {
        self.graph
            .read()
            .ok()?
            .groups
            .iter()
            .find(|g| g.id == *id)
            .map(|g| g.disabled)
    }

    /// Set the viewport directly. Clamps `zoom` to the configured
    /// `min_zoom..=max_zoom`.
    pub fn set_viewport(&self, zoom: f32, pan: Point) {
        self.kit.update_viewport(|vp| {
            vp.set_zoom(zoom);
            vp.pan_x = pan.x;
            vp.pan_y = pan.y;
        });
    }

    /// Current viewport snapshot — re-exported from canvas-kit for
    /// hosts that want a single import.
    pub fn viewport(&self) -> CanvasViewport {
        self.kit.viewport()
    }

    /// Visible canvas-content rect padded by 25 % of the viewport
    /// extent on each axis. Used by the frustum-cull pass to skip
    /// off-screen nodes / edges in the render loop — the slack keeps
    /// elements whose bounds touch the screen edge fully drawn (so
    /// their hit regions still register for incident edges that
    /// stretch into view) and absorbs a frame of camera-pan motion
    /// without popping. Returns a rect that effectively covers the
    /// whole content space when `screen_bounds` is degenerate (zero
    /// pre-paint) so first-frame culling never skips real work.
    fn visible_content_rect_padded(&self) -> Rect {
        let (sw, sh) = self.kit.screen_bounds();
        if sw <= 0.0 || sh <= 0.0 {
            return Rect::new(f32::MIN / 2.0, f32::MIN / 2.0, f32::MAX, f32::MAX);
        }
        let tl = self.kit.screen_to_content(Point::new(0.0, 0.0));
        let br = self.kit.screen_to_content(Point::new(sw, sh));
        let min_x = tl.x.min(br.x);
        let min_y = tl.y.min(br.y);
        let max_x = tl.x.max(br.x);
        let max_y = tl.y.max(br.y);
        let pad_x = (max_x - min_x) * 0.25;
        let pad_y = (max_y - min_y) * 0.25;
        Rect::new(
            min_x - pad_x,
            min_y - pad_y,
            (max_x - min_x) + pad_x * 2.0,
            (max_y - min_y) + pad_y * 2.0,
        )
    }

    // ─── Runtime observability commands ──────────────────────────────

    /// Attach (or clear) a status badge on a node. Triggers a slot
    /// cache invalidation if presence flipped (badge changes
    /// fingerprint).
    pub fn set_node_badge(&self, id: &NodeId, badge: Option<StatusBadge>) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(n) = g.nodes.iter_mut().find(|n| n.id == *id) {
                let fingerprint = |b: &StatusBadge| (b.kind, b.count, b.tooltip.clone());
                if n.badge.is_some() != badge.is_some()
                    || n.badge.as_ref().map(fingerprint) != badge.as_ref().map(fingerprint)
                {
                    n.badge = badge;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.warm_slot_cache();
            self.bump_graph_rev();
        }
    }

    /// Update the runtime state of a connection (Running / Pending /
    /// idle). Avoids the full re-warm `set_graph` triggers for
    /// state-only mutations.
    pub fn set_connection_state(&self, id: ConnectionId, state: ConnectionState) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(c) = g.connections.iter_mut().find(|c| c.id == id) {
                if c.state != state {
                    c.state = state;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.bump_graph_rev();
        }
    }

    /// Flip a group's soft-disabled state. Every member node
    /// inherits the disabled visual (same effect as calling
    /// `set_node_disabled` on each member) AND the group's chrome
    /// dims. Member nodes' OWN `disabled` flag is left untouched —
    /// re-enabling the group restores each member to whatever flag
    /// it had before.
    pub fn set_group_disabled(&self, id: &GroupId, disabled: bool) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(grp) = g.groups.iter_mut().find(|g| g.id == *id) {
                if grp.disabled == disabled {
                    false
                } else {
                    grp.disabled = disabled;
                    true
                }
            } else {
                false
            }
        };
        if changed {
            // `bump_graph_rev` already calls `request_redraw`; no need
            // to repeat it.
            self.bump_graph_rev();
        }
    }

    /// Flip a node's soft-disabled state. The renderer dims the node
    /// and downgrades every incident edge to `Pending` style; no
    /// other editor state changes. Hosts typically wire this to a
    /// 'D' key or a context-menu entry. Bumps the graph revision
    /// so signal observers see the change.
    pub fn set_node_disabled(&self, id: &NodeId, disabled: bool) {
        let changed = {
            let mut g = self.graph.write().unwrap();
            if let Some(n) = g.nodes.iter_mut().find(|n| n.id == *id) {
                if n.disabled == disabled {
                    false
                } else {
                    n.disabled = disabled;
                    true
                }
            } else {
                false
            }
        };
        if changed {
            // `bump_graph_rev` already calls `request_redraw`.
            self.bump_graph_rev();
        }
    }

    /// Show a transient highlight on a node. Hosts call this from
    /// runtime hooks (trace stepping, debug pulses, error markers);
    /// the editor renders the highlight overlay and clears it after
    /// `duration`.
    pub fn flash_node(&self, id: NodeId, kind: FlashKind, duration: Duration) {
        let expires_at = web_time::Instant::now() + duration;
        self.flashes
            .write()
            .unwrap()
            .insert(id, NodeFlash { kind, expires_at });
        self.bump_graph_rev();
    }

    // ─── Single-entry dispatch ───────────────────────────────────────

    /// Apply a command. Mirrors the granular method API in enum form
    /// for hosts that want a uniform / queued / scripted dispatch
    /// surface.
    pub fn dispatch(&self, cmd: EditorCommand<K, N, C, G>) {
        use EditorCommand::*;
        match cmd {
            InsertNode(n) => self.insert_node(n),
            RemoveNode(id) => self.remove_node(&id),
            UpdateNodePosition(id, p) => self.update_node_position(&id, p),
            InsertConnection(c) => self.insert_connection(c),
            RemoveConnection(id) => self.remove_connection(id),
            InsertGroup(g) => self.insert_group(g),
            RemoveGroup(id) => self.remove_group(&id),
            SetGroupMembers(id, members) => self.set_group_members(&id, members),
            SetGroupCollapsed(id, collapsed) => self.set_group_collapsed(&id, collapsed),
            CreateSubgraph { id, name } => {
                let _ = self.create_subgraph(id, name);
            }
            DeleteSubgraph(id) => {
                let _ = self.delete_subgraph(&id);
            }
            RestoreSubgraph(sub) => {
                {
                    let mut subs = self.subgraphs.write().unwrap();
                    subs.insert(sub.id.clone(), sub);
                }
                self.bump_subgraph_rev();
            }
            RenameSubgraph(id, name) => {
                let _ = self.rename_subgraph(&id, name);
            }
            SetSubgraphNamespace(id, ns) => {
                let _ = self.set_subgraph_namespace(&id, ns);
            }
            Select(ids) => self.select(&ids),
            SelectOne(id) => self.select_one(&id),
            AddToSelection(id) => self.add_to_selection(&id),
            ClearSelection => self.clear_selection(),
            FocusOnNode(id) => self.focus_on_node(&id),
            ZoomToFit => self.zoom_to_fit(),
            ZoomToSelection => self.zoom_to_selection(),
            SetViewport { zoom, pan } => self.set_viewport(zoom, pan),
            SetNodeBadge(id, badge) => self.set_node_badge(&id, badge),
            SetNodeDisabled(id, disabled) => self.set_node_disabled(&id, disabled),
            SetGroupDisabled(id, disabled) => self.set_group_disabled(&id, disabled),
            SetConnectionState(id, state) => self.set_connection_state(id, state),
            FlashNode(id, kind, dur) => self.flash_node(id, kind, dur),
            SetGraph {
                nodes,
                connections,
                groups,
                exposed,
            } => self.set_graph(nodes, connections, groups, exposed),
            ApplyLayout => self.apply_layout(),
            AlignNodes(ids, edge) => self.align_nodes(&ids, edge),
            DistributeNodes(ids, axis) => self.distribute_nodes(&ids, axis),
            Composite(cmds) => {
                for c in cmds {
                    self.dispatch(c);
                }
            }
        }
    }

    // ─── Align / distribute ──────────────────────────────────────────

    /// Align every node in `ids` to a shared edge. The bundle's
    /// existing extent (left-most, top-most, etc.) determines the
    /// target coordinate — no node moves toward an arbitrary point.
    /// Each move routes through [`Self::update_node_position`] so
    /// snap-to-grid quantises the result and `graph_signal` bumps
    /// once per moved node. No-op for `ids.len() < 2`.
    pub fn align_nodes(&self, ids: &[NodeId], edge: AlignEdge) {
        if ids.len() < 2 {
            return;
        }
        let theme_overrides = self.theme.read().unwrap();
        let theme = ThemeResolver::new(&theme_overrides);
        // Snapshot bounds + positions before any mutation so the
        // alignment target doesn't drift as we move nodes one at a
        // time.
        let snapshot: Vec<(NodeId, Rect)> = {
            let graph = self.graph.read().unwrap();
            graph
                .nodes
                .iter()
                .filter(|n| ids.contains(&n.id))
                .map(|n| (n.id.clone(), self.node_bounds_for(n, &theme)))
                .collect()
        };
        drop(theme_overrides);
        if snapshot.len() < 2 {
            return;
        }
        for (id, new_pos) in compute_align(&snapshot, edge) {
            self.update_node_position(&id, new_pos);
        }
    }

    /// Evenly space every node in `ids` along `axis`. The two
    /// outermost nodes (by centre coord on the axis) stay put and
    /// anchor the distribution; everything between spaces uniformly
    /// node-centre to node-centre. No-op for `ids.len() < 3`.
    pub fn distribute_nodes(&self, ids: &[NodeId], axis: DistributeAxis) {
        if ids.len() < 3 {
            return;
        }
        let theme_overrides = self.theme.read().unwrap();
        let theme = ThemeResolver::new(&theme_overrides);
        let snapshot: Vec<(NodeId, Rect)> = {
            let graph = self.graph.read().unwrap();
            graph
                .nodes
                .iter()
                .filter(|n| ids.contains(&n.id))
                .map(|n| (n.id.clone(), self.node_bounds_for(n, &theme)))
                .collect()
        };
        drop(theme_overrides);
        if snapshot.len() < 3 {
            return;
        }
        for (id, new_pos) in compute_distribute(&snapshot, axis) {
            self.update_node_position(&id, new_pos);
        }
    }

    // ─── Signal + state getters ──────────────────────────────────────

    /// Bumps on every graph mutation (insert/remove node, connection,
    /// group; member changes; node position update). Use as a
    /// dependency for derived state that reflects graph shape.
    pub fn graph_signal(&self) -> SignalId {
        self.graph_rev.signal_id()
    }

    /// Monotonic revision counter — useful when a host wants to
    /// detect *whether* the graph changed (not just be notified).
    pub fn graph_revision(&self) -> u64 {
        self.graph_rev.try_get().unwrap_or(0)
    }

    /// Read-only snapshot of the editor's current graph. Useful for
    /// hosts re-syncing their own mirror after applying an undo /
    /// redo (the editor just dispatched the inverse command against
    /// its internal graph, so its state is the new source of truth).
    /// Allocates — clone the vectors out. For hot-path reads,
    /// subscribe to [`Self::graph_signal`] instead.
    #[allow(clippy::type_complexity)]
    pub fn graph_snapshot(
        &self,
    ) -> (
        Vec<NodeInstance<N>>,
        Vec<crate::connection::Connection<C>>,
        Vec<crate::group::Group<G>>,
        Vec<crate::subgraph::ExposedPort<K>>,
    )
    where
        N: Clone,
        C: Clone,
        G: Clone,
    {
        let g = self.graph.read().unwrap();
        (
            g.nodes.clone(),
            g.connections.clone(),
            g.groups.clone(),
            g.exposed.clone(),
        )
    }

    /// Bumps on every drag-to-connect FSM transition.
    pub fn drag_state_signal(&self) -> SignalId {
        self.drag_state.signal_id()
    }

    /// Current drag-to-connect state.
    pub fn drag_state(&self) -> DragConnect {
        self.drag_state.try_get().unwrap_or_default()
    }

    /// Bumps when the hovered target changes.
    pub fn hover_signal(&self) -> SignalId {
        self.hover_state.signal_id()
    }

    /// What the pointer is currently over (or `None`).
    pub fn hovered(&self) -> Option<HoverTarget> {
        self.hover_state.try_get().unwrap_or(None)
    }

    /// Bumps every time an [`EditorEvent`] is pushed. Pair with
    /// [`drain_events`](Self::drain_events) inside an effect.
    pub fn events_signal(&self) -> SignalId {
        self.events_rev.signal_id()
    }

    /// Return + clear the pending event queue. Idempotent: a second
    /// call on the same frame returns an empty vec.
    pub fn drain_events(&self) -> Vec<EditorEvent<K>> {
        std::mem::take(&mut *self.events_queue.lock().unwrap())
    }

    /// Bumps every time the canvas selection set changes. Re-exported
    /// from [`CanvasKit`] so hosts can subscribe to selection without
    /// reaching through [`Self::canvas_kit`]. Pair with
    /// [`Self::selected_node_ids`] / [`Self::selected_connection_ids`]
    /// inside an effect.
    pub fn selection_signal(&self) -> SignalId {
        self.kit.selection_signal()
    }

    /// Bumps every time the viewport (zoom or pan) changes. Re-
    /// exported from [`CanvasKit`] for the same one-import ergonomics
    /// as [`Self::viewport`]. Pair with [`Self::viewport`] inside an
    /// effect (or read `viewport().zoom` / `viewport().pan_x/y`
    /// directly).
    pub fn viewport_signal(&self) -> SignalId {
        self.kit.viewport_signal()
    }

    /// Current canvas-kit selection state — re-exported for hosts
    /// that want a single import alongside [`Self::selection_signal`].
    /// For node / connection ids specifically, use
    /// [`Self::selected_node_ids`] / [`Self::selected_connection_ids`];
    /// this returns the raw region-id set.
    pub fn selection(&self) -> blinc_canvas_kit::SelectionState {
        self.kit.selection()
    }

    // ─── Internal helpers ────────────────────────────────────────────

    /// Increment + notify `graph_rev`. Centralised so every mutation
    /// path bumps the same counter and `request_redraw` fires.
    /// Callers MUST NOT issue their own `request_redraw` after this —
    /// the helper already does, and a second call adds nothing but
    /// noise in the trace logs.
    fn bump_graph_rev(&self) {
        let next = self.graph_rev.try_get().unwrap_or(0).wrapping_add(1);
        self.graph_rev.set(next);
        blinc_layout::request_redraw();
    }

    /// Push an event onto the queue and bump `events_rev`. The
    /// editor itself calls this from drag handlers / drag-end /
    /// port-finalise / right-click. Hosts may call it too — e.g. a
    /// context-menu callback that wants to re-use the same event-
    /// drain loop instead of invoking granular methods directly,
    /// or a programmatic / scripting layer that synthesises user
    /// gestures.
    pub fn push_event(&self, evt: EditorEvent<K>) {
        self.events_queue.lock().unwrap().push(evt);
        let next = self.events_rev.try_get().unwrap_or(0).wrapping_add(1);
        self.events_rev.set(next);
    }

    /// Compute the centre point + size of a node by id, using its
    /// `size` field (falling back to the theme default).
    fn node_centre(&self, id: &NodeId) -> Option<(Point, (f32, f32))> {
        let g = self.graph.read().unwrap();
        let node = g.nodes.iter().find(|n| n.id == *id)?;
        let (w, h) = node.size.unwrap_or((180.0, 72.0));
        Some((
            Point::new(node.position.x + w * 0.5, node.position.y + h * 0.5),
            (w, h),
        ))
    }

    /// Union AABB of all nodes matching `pred`. Returns `None` when
    /// no node matches.
    fn union_of_nodes(
        &self,
        pred: impl Fn(&NodeId) -> bool,
        theme: &ThemeResolver<'_>,
    ) -> Option<Rect> {
        let g = self.graph.read().unwrap();
        let mut acc: Option<Rect> = None;
        for n in g.nodes.iter().filter(|n| pred(&n.id)) {
            let b = self.node_bounds_for(n, theme);
            acc = Some(match acc {
                None => b,
                Some(prev) => union_rect(prev, b),
            });
        }
        acc
    }

    /// Pan + zoom the viewport so `rect` fits with a 12% margin.
    /// Drive the viewport from its current `(pan_x, pan_y, zoom)` to
    /// the supplied target over `duration_ms`, using ease-out-cubic.
    /// Replaces the inflight animation in place (no callback churn);
    /// if no animation was running, registers a tick callback with
    /// the global scheduler that advances the lerp each frame and
    /// self-unregisters on settle so the scheduler can park itself
    /// back at the lower idle rate.
    ///
    /// Falls back to a snap (immediate `update_viewport`) when the
    /// global scheduler isn't initialised — keeps headless / test
    /// paths working without a background scheduler thread.
    fn animate_viewport_to(
        &self,
        target_pan_x: f32,
        target_pan_y: f32,
        target_zoom: f32,
        duration_ms: f32,
    ) {
        let vp_now = self.kit.viewport();
        // Skip if already at the target — avoids re-registering a
        // tick callback for a no-op (and keeps the scheduler idle).
        if (vp_now.pan_x - target_pan_x).abs() < 0.001
            && (vp_now.pan_y - target_pan_y).abs() < 0.001
            && (vp_now.zoom - target_zoom).abs() < 0.001
        {
            return;
        }

        // Snap on no-scheduler: e.g. unit tests construct a
        // NodeEditor without the layout scheduler being initialised.
        let Some(scheduler) = blinc_layout::get_global_scheduler() else {
            self.kit.update_viewport(|vp| {
                vp.set_zoom(target_zoom);
                vp.pan_x = target_pan_x;
                vp.pan_y = target_pan_y;
            });
            return;
        };

        let anim = ViewportAnimation {
            from_pan_x: vp_now.pan_x,
            from_pan_y: vp_now.pan_y,
            from_zoom: vp_now.zoom,
            to_pan_x: target_pan_x,
            to_pan_y: target_pan_y,
            to_zoom: target_zoom,
            elapsed_ms: 0.0,
            duration_ms: duration_ms.max(1.0),
        };
        *self.viewport_anim.lock().unwrap() = Some(anim);

        // Tick callback already running? Then the next tick will see
        // the updated `viewport_anim` (replaced above) and pick up
        // the new target from-the-current-state. No re-registration
        // needed — saves scheduler churn when the user types fast
        // search queries (each keystroke triggers a new target).
        if self.viewport_anim_cb_id.lock().unwrap().is_some() {
            return;
        }

        let anim_slot = self.viewport_anim.clone();
        let cb_id_slot = self.viewport_anim_cb_id.clone();
        let kit = self.kit.clone();
        let cb_id = scheduler.register_tick_callback(move |dt_secs| {
            // SchedulerHandle::register_tick_callback passes `dt` in
            // SECONDS (see scheduler.rs ~line 455: `raw_dt = ... .as_secs_f32()`).
            // The springs / timelines that read `dt_ms` internally
            // multiply by 1000 — we have to do the same here.
            let dt_ms = dt_secs * 1000.0;
            let (target_pan_x, target_pan_y, target_zoom, done) = {
                let mut guard = anim_slot.lock().unwrap();
                let Some(a) = guard.as_mut() else {
                    // Animation was cleared externally — let the
                    // self-unregister branch below take care of
                    // removing the callback.
                    return;
                };
                a.elapsed_ms += dt_ms;
                let raw_t = (a.elapsed_ms / a.duration_ms).clamp(0.0, 1.0);
                // Ease-out-cubic — standard camera-tween curve via
                // blinc_animation's canonical Easing enum so the
                // search transitions match every other Blinc
                // animation that uses the same curve.
                let eased = blinc_animation::Easing::EaseOutCubic.apply(raw_t);
                let pan_x = a.from_pan_x + (a.to_pan_x - a.from_pan_x) * eased;
                let pan_y = a.from_pan_y + (a.to_pan_y - a.from_pan_y) * eased;
                let zoom = a.from_zoom + (a.to_zoom - a.from_zoom) * eased;
                let done = raw_t >= 1.0;
                (pan_x, pan_y, zoom, done)
            };

            kit.update_viewport(|vp| {
                vp.set_zoom(target_zoom);
                vp.pan_x = target_pan_x;
                vp.pan_y = target_pan_y;
            });

            if done {
                *anim_slot.lock().unwrap() = None;
                // Self-unregister so the scheduler can return to
                // its idle rate. Safe to call `remove_tick_callback`
                // from inside a tick — the scheduler clones the
                // callback list before invocation, so a same-tick
                // removal doesn't invalidate the iterator.
                let id = cb_id_slot.lock().unwrap().take();
                if let Some(id) = id {
                    if let Some(s) = blinc_layout::get_global_scheduler() {
                        s.remove_tick_callback(id);
                    }
                }
            }

            blinc_layout::request_redraw();
        });
        if let Some(id) = cb_id {
            *self.viewport_anim_cb_id.lock().unwrap() = Some(id);
        }
    }

    fn fit_rect_to_viewport(&self, rect: Rect) {
        let (sw, sh) = self.kit.screen_bounds();
        if sw <= 0.0 || sh <= 0.0 {
            return;
        }
        let margin = 0.12;
        let avail_w = sw * (1.0 - margin * 2.0);
        let avail_h = sh * (1.0 - margin * 2.0);
        let zoom_x = avail_w / rect.width().max(1.0);
        let zoom_y = avail_h / rect.height().max(1.0);
        let zoom = zoom_x.min(zoom_y);
        let cx = rect.x() + rect.width() * 0.5;
        let cy = rect.y() + rect.height() * 0.5;
        // Pan in CONTENT units (see focus_on_node comment).
        let target_pan_x = sw * 0.5 / zoom - cx;
        let target_pan_y = sh * 0.5 / zoom - cy;
        self.animate_viewport_to(target_pan_x, target_pan_y, zoom, VIEWPORT_TWEEN_MS);
    }

    /// Variant of [`fit_rect_to_viewport`] that caps the new zoom at
    /// `max_zoom`. Used by `search_and_focus` on single-match queries
    /// so a small node doesn't get magnified to a near-fullscreen
    /// interior shot — the user expects "focus on the match," not
    /// "show me the body fill at 5×."
    fn fit_rect_to_viewport_capped(&self, rect: Rect, max_zoom: f32) {
        let (sw, sh) = self.kit.screen_bounds();
        tracing::info!(
            target: "blinc_node_editor::search",
            rect_x = rect.x(), rect_y = rect.y(),
            rect_w = rect.width(), rect_h = rect.height(),
            sw, sh, max_zoom,
            "fit_rect_to_viewport_capped called"
        );
        if sw <= 0.0 || sh <= 0.0 {
            tracing::warn!(
                target: "blinc_node_editor::search",
                "fit_rect_to_viewport_capped: screen_bounds is zero, skipping"
            );
            return;
        }
        let margin = 0.12;
        let avail_w = sw * (1.0 - margin * 2.0);
        let avail_h = sh * (1.0 - margin * 2.0);
        let fit_zoom_x = avail_w / rect.width().max(1.0);
        let fit_zoom_y = avail_h / rect.height().max(1.0);
        let new_zoom = fit_zoom_x.min(fit_zoom_y).min(max_zoom).max(0.05);
        let cx = rect.x() + rect.width() * 0.5;
        let cy = rect.y() + rect.height() * 0.5;
        let target_pan_x = sw * 0.5 / new_zoom - cx;
        let target_pan_y = sh * 0.5 / new_zoom - cy;
        tracing::info!(
            target: "blinc_node_editor::search",
            new_zoom, cx, cy,
            target_pan_x,
            target_pan_y,
            "fit_rect_to_viewport_capped → setting viewport"
        );
        self.animate_viewport_to(target_pan_x, target_pan_y, new_zoom, VIEWPORT_TWEEN_MS);
    }

    /// Variant of [`fit_rect_to_viewport`] that only ever zooms OUT,
    /// never IN. Pan always centres the rect; zoom is the MIN of the
    /// natural fit-zoom and the viewport's current zoom. Used by
    /// `search_and_focus` for multi-match queries — the user's
    /// intent is "show me every match in one view," which a hard
    /// fit-zoom can violate when matches happen to cluster close
    /// together (fit-zoom > current_zoom yanks them into a tighter
    /// view than where they started).
    fn fit_rect_to_viewport_zoom_out_only(&self, rect: Rect) {
        let (sw, sh) = self.kit.screen_bounds();
        if sw <= 0.0 || sh <= 0.0 {
            return;
        }
        let margin = 0.12;
        let avail_w = sw * (1.0 - margin * 2.0);
        let avail_h = sh * (1.0 - margin * 2.0);
        let fit_zoom_x = avail_w / rect.width().max(1.0);
        let fit_zoom_y = avail_h / rect.height().max(1.0);
        let fit_zoom = fit_zoom_x.min(fit_zoom_y);
        let current_zoom = self.kit.viewport().zoom;
        let new_zoom = fit_zoom.min(current_zoom);
        let cx = rect.x() + rect.width() * 0.5;
        let cy = rect.y() + rect.height() * 0.5;
        let target_pan_x = sw * 0.5 / new_zoom - cx;
        let target_pan_y = sh * 0.5 / new_zoom - cy;
        self.animate_viewport_to(target_pan_x, target_pan_y, new_zoom, VIEWPORT_TWEEN_MS);
    }

    /// Compute slot tables for every node / group fingerprint in the
    /// current graph that isn't already cached. Called automatically
    /// by `set_graph`; exposed publicly so hosts can re-warm after
    /// theme overrides or other out-of-band invalidations.
    pub fn warm_slot_cache(&self) {
        let theme_overrides = self.theme.read().unwrap();
        let theme = ThemeResolver::new(&theme_overrides);
        let theme_rev = self.theme_revision.load(Ordering::Relaxed);
        let graph = self.graph.read().unwrap();
        let templates = self.templates.read().unwrap();

        // Build the wanted fingerprint set + an inputs lookup table.
        let mut node_inputs_by_fp: AHashMap<NodeFingerprint, crate::slot::NodeSlotInputs> =
            AHashMap::new();
        let mut node_wanted: HashSet<NodeFingerprint> = HashSet::new();
        for node in &graph.nodes {
            let Some(template) = templates.get(&node.component) else {
                continue;
            };
            let mut inputs = node_inputs_from::<K, N>(template, node, &theme);
            if node.size.is_none() {
                self.apply_portal_width_override(template, &node.id, &mut inputs);
            }
            self.apply_portal_height_override(&node.id, &mut inputs);
            let fp = fingerprint_node(template, node, &inputs, theme_rev);
            node_wanted.insert(fp);
            node_inputs_by_fp.entry(fp).or_insert(inputs);
        }

        let mut group_inputs_by_fp: AHashMap<
            crate::slot::GroupFingerprint,
            crate::slot::GroupSlotInputs,
        > = AHashMap::new();
        let mut group_wanted: HashSet<crate::slot::GroupFingerprint> = HashSet::new();
        for group in &graph.groups {
            // Use the auto-bounds width as the group's chrome width;
            // fallback to a small default for empty groups.
            let auto_bounds = self::compute_group_auto_bounds(group, &graph.nodes, |n| {
                self.node_bounds_for(n, &theme)
            });
            let pad = theme.group_padding();
            let width = group
                .bounds
                .map(|b| b.width())
                .unwrap_or(auto_bounds.width() + pad * 2.0);
            let inputs = group_inputs_from(group, width, &theme);
            let fp = fingerprint_group(group, &inputs, theme_rev);
            group_wanted.insert(fp);
            group_inputs_by_fp.entry(fp).or_insert(inputs);
        }

        // Release the graph + templates read locks before the warm
        // call — the warm work doesn't need them and the parallel
        // path holds work for milliseconds at scale.
        drop(graph);
        drop(templates);
        drop(theme_overrides);

        warm_node_slot_cache(&self.node_slots, &node_wanted, |fp| {
            // Defensive: a stale fingerprint shouldn't reach this
            // path, but fall back to a zero-size compute so the
            // warm never panics.
            node_inputs_by_fp
                .get(&fp)
                .cloned()
                .unwrap_or(crate::slot::NodeSlotInputs {
                    width: 0.0,
                    has_subtitle: false,
                    has_badge: false,
                    has_icon: false,
                    icon_size: 16.0,
                    title_font_size: 13.0,
                    subtitle_font_size: 11.0,
                    content_padding: 10.0,
                    content_height: 0.0,
                })
        });

        warm_group_slot_cache(&self.group_slots, &group_wanted, |fp| {
            group_inputs_by_fp
                .get(&fp)
                .cloned()
                .unwrap_or(crate::slot::GroupSlotInputs {
                    width: 0.0,
                    header_height: 28.0,
                    has_description: false,
                    has_badge: false,
                    title_font_size: 13.0,
                    subtitle_font_size: 11.0,
                    content_padding: 10.0,
                    description_lines: 1,
                })
        });
    }

    /// Look up a node's cached slot table; computes synchronously on
    /// cache miss (the warm path may not have completed before the
    /// first paint).
    pub fn node_slots_for(
        &self,
        template: &NodeTemplate<K>,
        instance: &NodeInstance<N>,
        theme: &ThemeResolver<'_>,
    ) -> NodeSlots {
        let mut inputs = node_inputs_from::<K, N>(template, instance, theme);
        if instance.size.is_none() {
            self.apply_portal_width_override(template, &instance.id, &mut inputs);
        }
        self.apply_portal_height_override(&instance.id, &mut inputs);
        let fp = fingerprint_node(
            template,
            instance,
            &inputs,
            self.theme_revision.load(Ordering::Relaxed),
        );
        if let Some(slots) = self.node_slots.read().unwrap().get(&fp).cloned() {
            return slots;
        }
        // Miss — compute now + insert.
        let slots = compute_node_slots(&inputs);
        self.node_slots.write().unwrap().insert(fp, slots.clone());
        slots
    }

    /// Resolve a node's bounding rect by going through the slot
    /// cache. Height comes from taffy via `slots.total_height`;
    /// width is `instance.size.0` (or theme default). Use this
    /// instead of `render::node_bounds` from editor / interaction
    /// code so we never pay the lookup cost more than once per node
    /// per frame (the cache is shared) AND so the height always
    /// matches what the renderer paints. Falls back to a zero-rect
    /// when the node has no registered template — same defensive
    /// behaviour as the rest of the editor.
    pub fn node_bounds_for(&self, instance: &NodeInstance<N>, theme: &ThemeResolver<'_>) -> Rect {
        let templates = self.templates.read().unwrap();
        let Some(template) = templates.get(&instance.component) else {
            return Rect::new(instance.position.x, instance.position.y, 0.0, 0.0);
        };
        let slots = self.node_slots_for(template, instance, theme);
        crate::render::node_bounds(instance, &slots, theme)
    }

    /// Look up a group's cached slot table; computes synchronously on
    /// cache miss.
    pub fn group_slots_for(
        &self,
        group: &Group<G>,
        width: f32,
        theme: &ThemeResolver<'_>,
    ) -> GroupSlots {
        let inputs = group_inputs_from(group, width, theme);
        let fp = fingerprint_group(group, &inputs, self.theme_revision.load(Ordering::Relaxed));
        if let Some(slots) = self.group_slots.read().unwrap().get(&fp).cloned() {
            return slots;
        }
        let slots = compute_group_slots(&inputs);
        self.group_slots.write().unwrap().insert(fp, slots.clone());
        slots
    }

    /// Trigger the configured [`LayoutStrategy`] over the current
    /// graph. Pushes [`EditorEvent::LayoutApplied`] with the new
    /// per-node positions onto the events queue; hosts patch their
    /// own model from that event (and re-sync via `set_graph` /
    /// `update_node_position`).
    pub fn apply_layout(&self) {
        let graph = self.graph.read().unwrap();
        let strategy = self.layout_strategy.read().unwrap();
        let positions = apply_layout(&strategy, &graph.nodes, &graph.connections, &graph.groups);
        drop(strategy);

        let updates: Vec<(NodeId, Point)> = graph
            .nodes
            .iter()
            .zip(positions.iter())
            .map(|(n, p)| (n.id.clone(), *p))
            .collect();
        drop(graph);

        self.push_event(EditorEvent::LayoutApplied(updates));
    }

    // ─── Callback registration (validators only) ─────────────────────

    /// Host-supplied connection validator. Called as the user drags
    /// over a candidate input port; return `Accept` to allow release-
    /// to-connect, `Reject{reason}` to dim the preview and surface
    /// the reason.
    ///
    /// Validators are the only callback surface — everything else
    /// routes through the events queue. See module docs for the
    /// rationale.
    pub fn on_connect_request(
        self,
        cb: impl Fn(&ConnectRequest<'_, K>) -> ValidationOutcome + Send + Sync + 'static,
    ) -> Self {
        *self.on_validate.write().unwrap() = Some(Arc::new(cb));
        self
    }

    /// Register a synchronous handler that fires the moment the user
    /// right-clicks the canvas — BEFORE the matching
    /// `EditorEvent::ContextMenuRequested` lands in the event queue.
    /// Hosts use this to open a `cn::context_menu()` or equivalent
    /// contextual surface inside the SAME frame as the right-click,
    /// so the overlay's subtree rebuild + class-animation start
    /// land on this frame instead of being deferred. (The
    /// drain-then-show path arrives too late in the frame loop — the
    /// new overlay's class-anim init misses
    /// `start_all_css_animations` and the menu can render at its
    /// `from`-keyframe sample until a paint invalidation forces a
    /// re-bake.)
    ///
    /// The event still fires for hosts that want to observe via
    /// `events_signal()`. Pick the surface that fits — this callback
    /// for "render a menu RIGHT NOW", the event for "log / record /
    /// react asynchronously".
    pub fn on_context_menu(
        self,
        cb: impl Fn(crate::event::ContextMenuTarget, blinc_core::layer::Point) + Send + Sync + 'static,
    ) -> Self {
        *self.on_context_menu.write().unwrap() = Some(Arc::new(cb));
        self
    }

    // ─── Element / rendering ─────────────────────────────────────────

    /// Build the editor as a [`Div`]. Mount inside a parent layout
    /// the same way you'd mount any Blinc widget.
    ///
    /// Takes `&mut self` because `CanvasKit::on_element_drag` etc.
    /// require a mutable kit borrow to install callbacks. Typical
    /// usage is `editor.element()` in `build_ui` where the editor
    /// is a local mutable binding.
    ///
    /// Wraps the canvas in a theme-tinted div so hosts never need
    /// to set a workspace background manually — the canvas dots
    /// draw on top of the tinted surface, both pulled from the
    /// active theme.
    pub fn element(&mut self) -> Div {
        use blinc_layout::div as ldiv;

        // Wire drag handlers BEFORE building the canvas widget —
        // `kit.element()` snapshots the kit's callback table into
        // the rendering closure, so any callbacks registered after
        // that call would miss this paint cycle.
        self.install_drag_handlers();

        let editor = self.clone();
        let canvas = self.kit.element(move |ctx, _bounds| {
            editor.render_frame(ctx);
        });

        // Keyboard shortcuts. KEY_DOWN routes only to the focused
        // element — the canvas div takes focus on POINTER_DOWN (see
        // `EventRouter::on_pointer_down`), so attaching the handler
        // here means clicks anywhere on the canvas surface enable
        // shortcuts for the next key press without the host needing
        // to wire focus management.
        let key_editor = self.clone();
        let canvas = canvas.on_key_down(move |evt| {
            // Spacebar = temporary pan override. canvas-kit's
            // `set_force_pan` flips an internal flag the next
            // pointer-down checks; releasing space (KEY_UP arm
            // below) clears it. Mid-drag pan continues regardless
            // since the flag is only consulted at pointer-down.
            let kc = blinc_core::events::KeyCode(evt.key_code);
            if kc == blinc_core::events::KeyCode::SPACE {
                key_editor.kit.set_force_pan(true);
                return;
            }
            let mods = blinc_core::events::Modifiers::new(evt.shift, evt.ctrl, evt.alt, evt.meta);
            key_editor.handle_key_down(kc, mods);
        });
        let key_up_editor = self.clone();
        let canvas = canvas.on_key_up(move |evt| {
            let kc = blinc_core::events::KeyCode(evt.key_code);
            if kc == blinc_core::events::KeyCode::SPACE {
                key_up_editor.kit.set_force_pan(false);
            }
        });
        // Right-click → ContextMenuRequested. Filters POINTER_DOWN
        // on `mouse_button == 2` via `Div::on_right_click`. Hit-tests
        // the cursor's content point, classifies into a
        // `ContextMenuTarget`, replaces the canvas-kit selection
        // when the right-clicked target isn't already in it
        // (standard editor convention), then emits the event so the
        // host can paint the menu (e.g. via `cn::context_menu()`).
        let rc_editor = self.clone();
        let outer = ldiv()
            .w_full()
            .h_full()
            .bg(workspace_background())
            .child(canvas)
            .on_right_click(move |evt| {
                let kit = rc_editor.canvas_kit();
                let vp = kit.viewport();
                let screen_pt = blinc_core::layer::Point::new(evt.local_x, evt.local_y);
                let content_pt = vp.screen_to_content(screen_pt);
                let hit = kit.hit_test(content_pt);
                let target = resolve_context_menu_target(hit.as_deref());
                // Selection update: replace selection with the
                // right-clicked target unless it's already in the
                // selection (so right-clicking inside a multi-select
                // keeps the group). Canvas-blank right-clicks leave
                // selection untouched — hosts who want "click-empty
                // clears selection" can wire that in their handler.
                if let Some(region) = target_region_id(&target) {
                    let mut sel = kit.selection().selected;
                    if !sel.contains(&region) {
                        sel.clear();
                        sel.insert(region);
                        kit.set_selection(sel);
                    }
                }
                // Synchronous callback FIRST. Hosts that supply one
                // can mount their contextual surface in this same
                // frame, before the overlay-stack dirty-poll at
                // windowed.rs:5504 runs. The event still fires
                // immediately after for any host observing via
                // `events_signal()`.
                if let Some(cb) = rc_editor.on_context_menu.read().unwrap().clone() {
                    cb(target.clone(), screen_pt);
                }
                rc_editor.push_event(EditorEvent::ContextMenuRequested {
                    target,
                    anchor_screen: screen_pt,
                });
                blinc_layout::request_redraw();
            });
        // Capture KEY_DOWN + TEXT_INPUT events on the outer canvas
        // wrapper so portal_ui's inline-editable widgets (text_input,
        // future number input) can drain them per-frame. Idempotent
        // against the editor's existing canvas-level keyboard
        // handlers — these are additive Div handlers, not the
        // single-owner kit-level ones.
        blinc_portal_ui::ui::install_kbd_hook(outer)
    }

    /// Register drag + drag-end handlers on the underlying canvas
    /// kit. The drag handler parses region IDs to route deltas
    /// onto either a single node or every member of a group (when
    /// the user grabbed the group's chrome). The drag-end handler
    /// pushes [`EditorEvent::NodeDragged`] once per drag with the
    /// final position, so hosts get a single settled position per
    /// gesture (per-frame fires would saturate a typical host model
    /// with hundreds of writes per drag).
    fn install_drag_handlers(&mut self) {
        // Hover: feed `hover_state` from region-id changes so edge /
        // node / port hover paints can pick it up via `hover_signal`
        // (host can also derive against it). Edge hover lights up the
        // curve via `draw_edge`'s `hovered` arg; future work uses the
        // same signal for port + node tooltips.
        let hover_editor = self.clone();
        self.kit.on_element_hover(move |evt| {
            let next = match evt.region_id.as_deref() {
                Some(r) => parse_hover_target(r),
                None => None,
            };
            if hover_editor.hover_state.try_get().unwrap_or(None) != next {
                hover_editor.hover_state.set(next);
                blinc_layout::request_redraw();
            }
        });

        // Click: route delete-button clicks into the events queue.
        // Selection is already managed by canvas-kit's POINTER_DOWN
        // auto-select; clicking the × button incidentally selects
        // `edge_delete:{id}`, so we restore the original edge
        // selection here so the user can cancel the delete dialog
        // without losing their selection.
        let click_editor = self.clone();
        self.kit.on_element_click(move |evt| {
            let Some(region_id) = evt.region_id.as_deref() else {
                return;
            };
            tracing::debug!(
                target: "blinc_node_editor::click",
                region = region_id,
                "on_element_click"
            );
            // Match once on the typed region id — adding a new
            // region kind anywhere becomes a compile error here
            // until every branch is taught how to handle it.
            let parsed = RegionId::parse(region_id);
            if let Some(RegionId::EdgeDelete(conn_id)) = &parsed {
                // Restore edge selection (replace the accidental
                // edge_delete: select with the edge: id).
                let edge_region = RegionId::Edge(*conn_id).encode();
                let mut sel = click_editor.kit.selection().selected;
                sel.remove(region_id);
                sel.insert(edge_region);
                click_editor.kit.set_selection(sel);
                click_editor.push_event(EditorEvent::DeleteConnectionRequested(*conn_id));
                blinc_layout::request_redraw();
                return;
            }
            // Group header chrome buttons. POINTER_DOWN's
            // auto-selection writes the button's region into the
            // selection set; clear it so the underlying group's
            // selection state isn't polluted by a chrome click.
            if let Some(RegionId::GroupCollapse(group_id)) = &parsed {
                let is_collapsed = click_editor
                    .graph
                    .read()
                    .unwrap()
                    .groups
                    .iter()
                    .find(|g| g.id == *group_id)
                    .map(|g| g.is_collapsed)
                    .unwrap_or(false);
                let mut sel = click_editor.kit.selection().selected;
                sel.remove(region_id);
                click_editor.kit.set_selection(sel);
                click_editor.push_event(EditorEvent::ToggleCollapseRequested(
                    crate::group::ToggleCollapseRequest {
                        group: group_id.clone(),
                        collapsed: !is_collapsed,
                    },
                ));
                blinc_layout::request_redraw();
                return;
            }
            if let Some(RegionId::GroupDelete(group_id)) = &parsed {
                let mut sel = click_editor.kit.selection().selected;
                sel.remove(region_id);
                click_editor.kit.set_selection(sel);
                click_editor.push_event(EditorEvent::DeleteGroupRequested(
                    crate::group::DeleteGroupRequest {
                        group: group_id.clone(),
                    },
                ));
                blinc_layout::request_redraw();
                return;
            }
            if let Some(RegionId::GroupEdit(group_id)) = &parsed {
                // Edit chip opens a combined title + description
                // form. Reuse the title's stashed content-space rect
                // as the popover anchor — double-click on title and
                // double-click on description still emit their
                // focused single-field events; only the chip emits
                // the combined `EditGroupRequested`.
                let group_id = group_id.clone();
                let title_key = RegionId::GroupTitle(group_id.clone()).encode();
                let title_rect = click_editor
                    .group_text_rects
                    .lock()
                    .unwrap()
                    .get(&title_key)
                    .copied();
                let (current_title, current_description) = {
                    let graph = click_editor.graph.read().unwrap();
                    graph
                        .groups
                        .iter()
                        .find(|g| g.id == group_id)
                        .map(|g| (g.name.clone(), g.description.clone().unwrap_or_default()))
                        .unwrap_or_default()
                };
                let mut sel = click_editor.kit.selection().selected;
                sel.remove(region_id);
                click_editor.kit.set_selection(sel);
                if let Some(anchor) = title_rect {
                    let view = click_editor.kit.viewport().transform();
                    let tl = view.transform_point(Point::new(anchor.x(), anchor.y()));
                    let br = view.transform_point(Point::new(
                        anchor.x() + anchor.width(),
                        anchor.y() + anchor.height(),
                    ));
                    let anchor_screen = Rect::new(
                        tl.x.min(br.x),
                        tl.y.min(br.y),
                        (br.x - tl.x).abs(),
                        (br.y - tl.y).abs(),
                    );
                    click_editor.push_event(EditorEvent::EditGroupRequested {
                        group: group_id,
                        current_title,
                        current_description,
                        anchor_screen,
                    });
                }
                blinc_layout::request_redraw();
            }
        });

        // Per-frame drag: mutate the editor's internal graph in
        // place so the next render sees the new positions. We
        // don't fire the host callback here — that lives on
        // drag-end so hosts get a single settled position per
        // gesture (per-frame fires would saturate a typical
        // host model with hundreds of writes per drag).
        //
        // CanvasKit fires the drag callback ONCE PER SELECTED
        // REGION plus once for the active region (if not in
        // selection). For a node editor we want a node-drag to
        // move only the clicked node, even if a previous gesture
        // left other regions selected (e.g. group selected by
        // marquee, then user clicks an outside node). Without
        // this gate, dragging an outside node would also drag
        // the still-selected group's members — visible as the
        // group "snapping back" on drag-end when `set_graph`
        // re-syncs from the host model that only updated the
        // dragged node.
        let drag_editor = self.clone();
        self.kit.on_element_drag(move |evt| {
            let active = drag_editor.kit.interaction().active;
            if active.as_deref() != Some(evt.region_id.as_str()) {
                // Selected-but-not-active fire — skip.
                return;
            }

            // Parse once; reuse for routing + group/node disambiguation.
            let parsed = RegionId::parse(&evt.region_id);

            // Port-to-port drag-connect: update the DragConnect
            // state machine; render_frame paints the live preview
            // edge from the source port to the cursor (with the
            // validator-derived tint).
            if let Some(RegionId::Port(addr)) = parsed.clone() {
                update_port_drag(&drag_editor, addr, evt.content_point);
                blinc_layout::request_redraw();
                return;
            }

            // Resolve "group_title:{id}" / "group_desc:{id}" hit
            // regions onto their parent group's drag target. The
            // text rects sit on top of the group's body hit region
            // for double-click-to-edit detection and would
            // otherwise swallow pointer-down on the header text,
            // leaving the group un-draggable when the user grabs
            // it by title or description. Aliasing here lets the
            // same hit region serve both editing (via the click
            // listener) and dragging.
            let resolved_region_id = match &parsed {
                Some(RegionId::GroupTitle(g)) | Some(RegionId::GroupDesc(g)) => {
                    RegionId::Group(g.clone()).encode()
                }
                _ => evt.region_id.clone(),
            };

            apply_drag_delta(&drag_editor, &resolved_region_id, evt.content_delta);

            // Live drag-into / drag-out preview: when the active
            // region is a node (not a group's chrome), classify
            // current position against group bounds + the modifier
            // state captured at THIS drag tick. The renderer reads
            // the preview state and tints group borders so the user
            // sees the would-be add / remove BEFORE releasing.
            match &parsed {
                Some(RegionId::Node(node_id)) => {
                    update_drag_group_preview(&drag_editor, node_id, evt.modifiers.shift);
                }
                Some(RegionId::Group(g))
                | Some(RegionId::GroupTitle(g))
                | Some(RegionId::GroupDesc(g)) => {
                    // Multi-node sibling: dragging a group container
                    // mid-flight should tint any enclosing parent
                    // group's border the same way a single-node drag
                    // does. See `update_drag_group_preview_for_group`.
                    update_drag_group_preview_for_group(&drag_editor, g, evt.modifiers.shift);
                }
                _ => {}
            }
            blinc_layout::request_redraw();
        });

        // Drag-end: dispatch by what was being dragged.
        let end_editor = self.clone();
        self.kit.on_element_drag_end(move |evt| {
            // Always clear the live drag-into / drag-out preview at
            // drag-end. Stale highlights would otherwise linger on
            // groups until the next drag tick reset them.
            {
                let mut slot = end_editor.drag_group_preview.write().unwrap();
                if *slot != DragGroupPreview::default() {
                    *slot = DragGroupPreview::default();
                    drop(slot);
                    blinc_layout::request_redraw();
                }
            }

            // Port-drag in flight → finalise the connect attempt.
            // Check this BEFORE the node-drag branch so a port-drag
            // started on a node's port doesn't fall through to the
            // node-drag-end host callback. Pass the drag-end's
            // `content_point` so finalise can re-resolve the hover
            // state against the ACTUAL release coordinate — winit's
            // per-frame `Drag` events don't always include a final
            // move at the release position, so reading whatever the
            // last `update_port_drag` tick stored would latch onto a
            // stale candidate (the one the cursor was hovering one
            // frame earlier, possibly a port the user already moved
            // past).
            {
                if end_editor.drag_state().is_active() {
                    finalise_port_drag(&end_editor, evt.content_point);
                    blinc_layout::request_redraw();
                    return;
                }
            }

            let Some(region_id) = evt.region_id.as_deref() else {
                return;
            };
            match RegionId::parse(region_id) {
                Some(RegionId::Node(id)) => {
                    let position = {
                        let graph = end_editor.graph.read().unwrap();
                        graph.nodes.iter().find(|n| n.id == id).map(|n| n.position)
                    };
                    if let Some(pos) = position {
                        // Snap-to-grid quantises the final position before
                        // the host sees it. Per-frame drag deltas still
                        // move freely; only the settled position commits
                        // to the grid. Symmetrically applied inside
                        // `update_node_position` for programmatic moves.
                        let snapped = end_editor.kit.snap_point(pos);
                        if snapped != pos {
                            let mut g = end_editor.graph.write().unwrap();
                            if let Some(n) = g.nodes.iter_mut().find(|n| n.id == id) {
                                n.position = snapped;
                            }
                        }
                        end_editor.push_event(EditorEvent::NodeDragged {
                            id: id.clone(),
                            position: snapped,
                        });
                        detect_membership_changes(&end_editor, &id, evt.modifiers.shift);
                    }
                }
                Some(RegionId::Group(g))
                | Some(RegionId::GroupTitle(g))
                | Some(RegionId::GroupDesc(g)) => {
                    // Shift-drag of a group container: detach its
                    // members from any *enclosing* group whose remaining
                    // (non-dragged) members no longer enclose them. The
                    // per-node detect path can't see this case because
                    // every dragged node moves together so its
                    // self-excluding auto-bounds still tracks the drag.
                    //
                    // `group_title:` / `group_desc:` are aliased here for
                    // the same reason `apply_drag_delta` aliases them
                    // mid-drag: the title/description text rects sit on
                    // top of the group's body for double-click-to-edit
                    // detection, so the user's pointer-down/up may land
                    // on those region ids instead of the bare body.
                    detect_group_drag_membership_changes(&end_editor, &g, evt.modifiers.shift);
                }
                _ => {}
            }
        });

        // Track the latest pointer position. Used as the popover
        // anchor for `MultiSelectionSettled` so the host can pop
        // the floating mini-toolbar where the user's cursor was at
        // the moment selection committed.
        let pos_editor = self.clone();
        self.kit.on_any_event(move |evt| {
            use blinc_core::events::event_types;
            if matches!(
                evt.event_type,
                event_types::POINTER_DOWN | event_types::POINTER_UP | event_types::POINTER_MOVE
            ) {
                *pos_editor.last_screen_pos.write().unwrap() =
                    Some(Point::new(evt.local_x, evt.local_y));
            }
        });

        // Fire MultiSelectionSettled (2+ nodes) or SelectionCleared
        // (empty) on every commit. CanvasKit fires this callback
        // once per gesture-end (marquee finalised, shift-click
        // toggled), so it's the right place to debounce: hosts
        // see one event per user-visible selection change.
        let sel_editor = self.clone();
        self.kit.on_selection_change(move |evt| {
            let nodes: Vec<NodeId> = evt
                .selected
                .iter()
                .filter_map(|s| match RegionId::parse(s)? {
                    RegionId::Node(id) => Some(id),
                    _ => None,
                })
                .collect();
            if nodes.len() >= 2 {
                let anchor = sel_editor
                    .last_screen_pos
                    .read()
                    .unwrap()
                    .unwrap_or(Point::new(0.0, 0.0));
                sel_editor.push_event(EditorEvent::MultiSelectionSettled {
                    node_ids: nodes,
                    anchor_screen: anchor,
                });
            } else if evt.selected.is_empty() {
                sel_editor.push_event(EditorEvent::SelectionCleared);
            }
        });
    }

    /// Snapshot every piece of state the renderer needs into a
    /// single [`FrameContext`]. Centralises the lock acquisition so
    /// `render_frame` (and hosts that drive their own canvas surface)
    /// don't have to spell out the same five reads each time.
    ///
    /// All five reads are intentionally taken every frame: `graph`
    /// and `drag` change at interaction rate; `templates` and
    /// `theme_overrides` rarely change but stale snapshots would
    /// give wrong slot lookups for the duration of a frame.
    /// `selection` is already a cheap internal snapshot. The
    /// lock-contention cost here is negligible — all writers run on
    /// the same UI thread, so the reads are uncontended atomics.
    ///
    /// Future optimization slot: when `templates` / `theme_overrides`
    /// grow to non-trivial size, memoise an `Arc<Snapshot>` keyed off
    /// `theme_revision` + a template-revision counter and short-
    /// circuit the read when the rev hasn't moved.
    fn begin_frame(&self) -> FrameContext<'_, K, N, C, G> {
        FrameContext {
            graph: self.graph.read().unwrap(),
            templates: self.templates.read().unwrap(),
            theme_overrides: self.theme.read().unwrap(),
            selection: self.kit.selection(),
            drag: self.drag_state(),
        }
    }

    /// One-frame render pass. Public so hosts that want a custom
    /// canvas surface (overlay editor, embedded preview) can call it
    /// from their own draw closure.
    pub fn render_frame(&self, ctx: &mut dyn DrawContext) {
        let frame = self.begin_frame();
        let theme = ThemeResolver::new(&frame.theme_overrides);

        // Snapshot the live drag-preview state once for this frame
        // so each group iteration reads a consistent view (the per-
        // frame drag handler may bump the preview between groups
        // otherwise).
        let preview = self.drag_group_preview.read().unwrap().clone();

        // Frustum cull rect — visible canvas-content region with a
        // generous slack margin so off-screen nodes still register
        // hit-regions for incident edges that arc back into view +
        // so a node sliding in from the edge isn't a single-frame
        // pop-in. `frustum_padding` is in content units; the slack
        // is wider at high zoom-out (small content per screen px)
        // so the perceptual margin stays constant.
        let frustum = self.visible_content_rect_padded();

        // Drop the previous frame's title / description rects so a
        // removed group's stale entry can't fire a double-click
        // event against a region the kit no longer routes to. The
        // group draw loop below re-populates this for every live
        // group it paints.
        self.group_text_rects.lock().unwrap().clear();
        self.badge_rects.lock().unwrap().clear();

        // Nodes that belong to a collapsed group are hidden from
        // rendering + hit-test for this frame. Their incident edges
        // are NOT removed — they get merged at the closest point on
        // the group's container perimeter so the user still sees
        // the connection topology (Zeal-style collapse).
        // Build a node→group lookup so the edge loop can map a
        // hidden endpoint back to its container rect.
        let mut node_to_collapsed_group: std::collections::HashMap<NodeId, GroupId> =
            std::collections::HashMap::new();
        for g in &frame.graph.groups {
            if g.is_collapsed {
                for m in &g.members {
                    node_to_collapsed_group.insert(m.clone(), g.id.clone());
                }
            }
        }
        let hidden_nodes: std::collections::HashSet<NodeId> =
            node_to_collapsed_group.keys().cloned().collect();
        // Collected during the group iteration below — keyed by
        // group id so the connection loop can look up the rect on
        // demand. Empty for expanded groups.
        let mut collapsed_group_rects: std::collections::HashMap<GroupId, Rect> =
            std::collections::HashMap::new();

        // Pre-pass: compute every group's preliminary body rect (the
        // chrome-inclusive rect each group would draw on its own),
        // then grow each group's body to enclose any *subset* group's
        // body. Without this, a nested wrapping group's header band
        // can poke above its enclosing group's top edge because the
        // outer group's `compute_group_auto_bounds` only sees member-
        // node positions, not its neighbours' chrome.
        //
        // Subset rule: group A is enclosed by group B iff
        // `A.members ⊆ B.members` AND A ≠ B AND A has fewer members
        // (or same count, breaking ties so A and B don't enclose
        // each other when they happen to share an identical member
        // list). The growth is unioned in; group `bounds` overrides
        // still win at draw time.
        let pad = theme.group_padding();
        let mut preliminary_bodies: std::collections::HashMap<GroupId, Rect> =
            std::collections::HashMap::new();
        for group in &frame.graph.groups {
            let auto = compute_group_auto_bounds(group, &frame.graph.nodes, |n| {
                self.node_bounds_for(n, &theme)
            });
            let width = group
                .bounds
                .map(|b| b.width())
                .unwrap_or(auto.width() + pad * 2.0);
            let slots = self.group_slots_for(group, width, &theme);
            let body = group.bounds.unwrap_or_else(|| {
                if group.is_collapsed {
                    Rect::new(
                        auto.x() - pad,
                        auto.y() - pad - slots.header.height(),
                        auto.width() + pad * 2.0,
                        slots.header.height(),
                    )
                } else {
                    Rect::new(
                        auto.x() - pad,
                        auto.y() - pad - slots.header.height(),
                        auto.width() + pad * 2.0,
                        auto.height() + pad * 2.0 + slots.header.height(),
                    )
                }
            });
            preliminary_bodies.insert(group.id.clone(), body);
        }
        let nested_body_growth: std::collections::HashMap<GroupId, Rect> = {
            use std::collections::{HashMap, HashSet};
            let member_sets: HashMap<GroupId, HashSet<NodeId>> = frame
                .graph
                .groups
                .iter()
                .map(|g| (g.id.clone(), g.members.iter().cloned().collect()))
                .collect();
            let mut grown: HashMap<GroupId, Rect> = HashMap::new();
            for outer in &frame.graph.groups {
                let outer_members = match member_sets.get(&outer.id) {
                    Some(s) if !s.is_empty() => s,
                    _ => continue,
                };
                let outer_body = match preliminary_bodies.get(&outer.id) {
                    Some(r) => *r,
                    None => continue,
                };
                let mut union = outer_body;
                for inner in &frame.graph.groups {
                    if inner.id == outer.id {
                        continue;
                    }
                    let inner_members = match member_sets.get(&inner.id) {
                        Some(s) if !s.is_empty() => s,
                        _ => continue,
                    };
                    let subset = inner_members.len() < outer_members.len()
                        && inner_members.is_subset(outer_members);
                    if !subset {
                        continue;
                    }
                    if let Some(inner_body) = preliminary_bodies.get(&inner.id) {
                        let nx = union.x().min(inner_body.x());
                        let ny = union.y().min(inner_body.y());
                        let nxe =
                            (union.x() + union.width()).max(inner_body.x() + inner_body.width());
                        let nye =
                            (union.y() + union.height()).max(inner_body.y() + inner_body.height());
                        union = Rect::new(nx, ny, nxe - nx, nye - ny);
                    }
                }
                if union != outer_body {
                    grown.insert(outer.id.clone(), union);
                }
            }
            grown
        };

        // 1. Groups — drawn first so they sit behind everything.
        for group in &frame.graph.groups {
            // Auto-bounds source per group:
            //
            // - Shift+drag of a member of THIS group: exclude the
            //   dragged node from the auto-fit calc so the group
            //   visually shrinks to its OTHER members' footprint
            //   instead of growing to follow the node. Reads as
            //   "the node is being torn out" while the gesture is
            //   in flight. Reverts on drag-end when the preview
            //   clears.
            // - Otherwise: live auto-bounds (members union). The
            //   `last_group_auto_bounds` cache keeps the chrome
            //   anchored if the group has temporarily lost all
            //   members (e.g. last member dragged out).
            let live_auto_bounds = compute_group_auto_bounds(group, &frame.graph.nodes, |n| {
                self.node_bounds_for(n, &theme)
            });
            let shift_excl_bounds = if preview.shift_held {
                if let Some(n) = preview.dragged_node.as_ref() {
                    if group.members.contains(n) {
                        compute_group_auto_bounds_excluding(group, &frame.graph.nodes, n, |node| {
                            self.node_bounds_for(node, &theme)
                        })
                    } else {
                        None
                    }
                } else if let Some(dg) = preview.dragged_group.as_ref() {
                    // Multi-node escape: the user is dragging a
                    // group container by its chrome. Exclude EVERY
                    // dragged-group member from the parent's
                    // auto-bounds so the parent visually shrinks
                    // around its remaining (stationary) members,
                    // mirroring the single-node escape's affordance.
                    let dragged_members: Vec<NodeId> = frame
                        .graph
                        .groups
                        .iter()
                        .find(|g| g.id == *dg)
                        .map(|g| g.members.clone())
                        .unwrap_or_default();
                    if !dragged_members.is_empty()
                        && dg != &group.id
                        && dragged_members.iter().any(|m| group.members.contains(m))
                    {
                        compute_group_auto_bounds_excluding_set(
                            group,
                            &frame.graph.nodes,
                            &dragged_members,
                            |node| self.node_bounds_for(node, &theme),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let has_members = group
                .members
                .iter()
                .any(|m| frame.graph.nodes.iter().any(|n| n.id == *m));
            let auto_bounds = if let Some(b) = shift_excl_bounds {
                // Don't cache the shrunk bounds — we want the
                // post-drag cache fallback to reflect the natural
                // members-union footprint, not the transient drag
                // state. Cache the live (full) bounds so a later
                // empty-group state restores correctly.
                if has_members {
                    self.last_group_auto_bounds
                        .write()
                        .unwrap()
                        .insert(group.id.clone(), live_auto_bounds);
                }
                b
            } else if has_members {
                self.last_group_auto_bounds
                    .write()
                    .unwrap()
                    .insert(group.id.clone(), live_auto_bounds);
                live_auto_bounds
            } else {
                self.last_group_auto_bounds
                    .read()
                    .unwrap()
                    .get(&group.id)
                    .copied()
                    .unwrap_or(live_auto_bounds)
            };
            let pad = theme.group_padding();
            let width = group
                .bounds
                .map(|b| b.width())
                .unwrap_or(auto_bounds.width() + pad * 2.0);
            let slots = self.group_slots_for(group, width, &theme);
            // CanvasKit selection contains region IDs (`group:id`),
            // not raw IDs — see the matching `node:` pattern below.
            let region_id = RegionId::Group(group.id.clone()).encode();
            let is_selected = frame.selection.selected.contains(&region_id);
            let border_kind = if preview.remove_target.as_ref() == Some(&group.id) {
                crate::render::GroupBorderKind::RemoveTarget
            } else if preview.add_target.as_ref() == Some(&group.id) {
                crate::render::GroupBorderKind::AddTarget
            } else {
                crate::render::GroupBorderKind::Normal
            };
            let group_badge_rect = draw_group(
                ctx,
                group,
                auto_bounds,
                &slots,
                &theme,
                is_selected,
                border_kind,
            );
            if let (Some(rect), Some(badge)) = (group_badge_rect, group.badge.as_ref()) {
                if badge.tooltip.is_some() {
                    let region = RegionId::GroupBadge(group.id.clone()).encode();
                    self.kit.hit_rect(region.clone(), rect);
                    self.badge_rects.lock().unwrap().insert(region, rect);
                }
            }

            // Header chrome — collapse / expand chevron. Painted on
            // top of the header band; hit region registered after
            // the group's body rect so the chrome wins the
            // canvas-kit reverse-order hit-test against the body
            // drag region.
            //
            // Delete is intentionally not part of the header chrome:
            // group deletion fires from the keyboard handler (when a
            // group region is selected and Delete / Backspace is
            // pressed) AND from `EditorCommand::RemoveGroup` for
            // programmatic / command-palette callers. Keeps the
            // header readable when a group's title or badge would
            // otherwise crowd the chrome.
            let body_rect = group.bounds.unwrap_or_else(|| {
                let natural = if group.is_collapsed {
                    Rect::new(
                        auto_bounds.x() - pad,
                        auto_bounds.y() - pad - slots.header.height(),
                        auto_bounds.width() + pad * 2.0,
                        slots.header.height(),
                    )
                } else {
                    Rect::new(
                        auto_bounds.x() - pad,
                        auto_bounds.y() - pad - slots.header.height(),
                        auto_bounds.width() + pad * 2.0,
                        auto_bounds.height() + pad * 2.0 + slots.header.height(),
                    )
                };
                // Nested-group growth: if a subset group's body rect
                // extends past this one's natural body rect, swap in
                // the grown union so the outer group's footprint
                // encloses the inner group's chrome (header + border)
                // properly. Computed once in the pre-pass above.
                nested_body_growth
                    .get(&group.id)
                    .copied()
                    .map(|grown| {
                        let nx = natural.x().min(grown.x());
                        let ny = natural.y().min(grown.y());
                        let nxe = (natural.x() + natural.width()).max(grown.x() + grown.width());
                        let nye = (natural.y() + natural.height()).max(grown.y() + grown.height());
                        Rect::new(nx, ny, nxe - nx, nye - ny)
                    })
                    .unwrap_or(natural)
            });
            // When `group.accent` is set, the header chrome (glyph
            // + chip outline) needs a contrasting foreground so it
            // stays readable on the accent band. `draw_group` /
            // `draw_group_header_chrome` honour the override
            // automatically when the group carries an accent.
            let chrome_override = group.accent.map(|a| {
                let lum = 0.299 * a.r + 0.587 * a.g + 0.114 * a.b;
                if lum > 0.55 {
                    blinc_core::layer::Color::rgba(0.08, 0.09, 0.12, 1.0)
                } else {
                    blinc_core::layer::Color::rgba(0.96, 0.96, 0.97, 1.0)
                }
            });
            let chrome = crate::render::draw_group_header_chrome(
                ctx,
                group,
                body_rect,
                &slots,
                &theme,
                chrome_override,
            );

            // Hit region must match the DRAWN body rect — `auto_bounds`
            // is just the member-node union (a strict sub-rect of the
            // padded + header-extended body), so registering it as the
            // hit shape leaves the visible chrome (header band +
            // padding ring) as dead space. The pointer falls through
            // to canvas-kit's background drag handler → drags pan the
            // whole canvas instead of moving the group.
            // Re-use the body_rect computed above for the drag-hit
            // region — collapsed groups expose only the header band
            // as the drag affordance; expanded groups use the full
            // members-wrap rect.
            self.kit.hit_rect(region_id, body_rect);

            // Title + description hit regions for double-click-to-edit.
            // The slot rects are NODE-LOCAL — translate to absolute
            // canvas-content coords via the group body's top-left.
            // Stash the absolute rects so the double-click listener
            // installed in `new()` can derive a screen-space anchor
            // for the host's inline editor overlay. Registered AFTER
            // the group body so the reverse-order hit-test picks
            // text first when the click falls inside the title /
            // description bounding box.
            //
            // `body_rect` here is the visible group rect (header +
            // padded body region). The slot tree's `header` rect is
            // relative to that body's origin, so the title / desc
            // rects in absolute canvas-content space are
            // `body_rect.origin + slots.title` / `body_rect.origin +
            // slots.description`.
            let title_rect = Rect::new(
                body_rect.x() + slots.title.x(),
                body_rect.y() + slots.title.y(),
                slots.title.width(),
                slots.title.height(),
            );
            let title_region = RegionId::GroupTitle(group.id.clone()).encode();
            self.kit.hit_rect(title_region.clone(), title_rect);
            self.group_text_rects
                .lock()
                .unwrap()
                .insert(title_region, title_rect);
            if let Some(desc_slot) = slots.description {
                let desc_rect = Rect::new(
                    body_rect.x() + desc_slot.x(),
                    body_rect.y() + desc_slot.y(),
                    desc_slot.width(),
                    desc_slot.height(),
                );
                let desc_region = RegionId::GroupDesc(group.id.clone()).encode();
                self.kit.hit_rect(desc_region.clone(), desc_rect);
                self.group_text_rects
                    .lock()
                    .unwrap()
                    .insert(desc_region, desc_rect);
            }

            // Chrome hit regions MUST be registered after the group's
            // body region. CanvasKit's hit-test scans regions in
            // reverse insertion order (last inserted = topmost) —
            // registering chrome BEFORE the body means the body
            // rect wins for clicks inside the chrome, swallowing the
            // affordance click. (Symptom: click logs land on
            // `group:{id}` instead of `group_collapse:{id}`.) Edit
            // and delete go in alongside the collapse chip — order
            // among them doesn't matter (rects don't overlap), but
            // keeping the visually-leftmost chip registered first
            // mirrors the painted left-to-right flex order.
            self.kit
                .hit_rect(RegionId::GroupEdit(group.id.clone()).encode(), chrome.edit);
            self.kit.hit_rect(
                RegionId::GroupDelete(group.id.clone()).encode(),
                chrome.delete,
            );
            self.kit.hit_rect(
                RegionId::GroupCollapse(group.id.clone()).encode(),
                chrome.collapse,
            );

            // Stash the body rect for collapsed groups so the edge
            // loop can re-route incident connections to the nearest
            // point on the perimeter.
            if group.is_collapsed {
                collapsed_group_rects.insert(group.id.clone(), body_rect);
            }
        }

        // 2. Connections — drawn beneath nodes so endpoints sit on
        //    top, with hover / select highlights.
        let slot_lookup =
            |tpl: &NodeTemplate<K>, n: &NodeInstance<N>| self.node_slots_for(tpl, n, &theme);
        let hovered_edge = match self.hovered() {
            Some(HoverTarget::Edge(id)) => Some(id),
            _ => None,
        };
        // Collect selected edges' midpoints so the delete-button pass
        // at the bottom can draw + register them AFTER every other
        // edge / node / port hit region. CanvasKit's hit-test scans
        // registrations in reverse, so anything registered last wins
        // — registering the delete button per-edge inside the loop
        // let later edges' segment AABBs intercept clicks on the
        // button (visible as the button being unclickable when a
        // crossing edge ran above it).
        // ─── Sliding port layout ─────────────────────────────────
        //
        // 1. For each port, determine its "preferred side" — the
        //    edge of its node that faces the average of its
        //    connection counterparts. Unconnected input ports
        //    default to the LEFT edge, outputs default to RIGHT.
        // 2. Group ports per (node, side). Track each port's
        //    perpendicular coordinate (the counterpart's y for
        //    L/R sides, x for T/B) so we can sort along the side
        //    in a way that minimises crossings.
        // 3. Compute an effective node height that accommodates
        //    the busier of left/right port counts — ports drive
        //    sizing so multi-input nodes get tall enough to
        //    spread without crowding.
        // 4. Distribute ports evenly along the side:
        //    - 1 port → centre of the side (NOT corner)
        //    - N ports → (i + 0.5) / N spacing, centred around
        //      the side midpoint
        // Group by (node, side) + record sort key. The SIDE is
        // pinned to the port's `resolved_position` (input → Left,
        // output → Right by default; per-port `with_position`
        // override wins) so ports never cross from one side to
        // another when a counterpart node moves. Within a side,
        // ports stack in TEMPLATE ORDER (the order
        // `with_input`/`with_output` were chained in). A previous
        // version sorted by the counterpart's perpendicular
        // coordinate to minimise edge crossings — that caused the
        // ports to SWAP positions the instant a single connection
        // landed (the connected port's sort key would jump from `0.0`
        // to the counterpart's y, ending up below an unconnected
        // sibling whose key stayed `0.0`). Users would aim at the
        // top port, the editor would relayout mid-drag, and the
        // connection would silently land on what was now the bottom
        // port. Template order keeps the geometry stable while the
        // user is dragging.
        let mut sides: std::collections::HashMap<(NodeId, PortSide), Vec<(PortAddress, f32)>> =
            std::collections::HashMap::new();
        for node in &frame.graph.nodes {
            if hidden_nodes.contains(&node.id) {
                continue;
            }
            let Some(template) = frame.templates.get(&node.component) else {
                continue;
            };
            for (template_idx, desc) in template
                .inputs
                .iter()
                .chain(template.outputs.iter())
                .enumerate()
            {
                let addr = PortAddress::new(node.id.clone(), desc.id.clone());
                let side = match desc.resolved_position() {
                    crate::port::PortPosition::Left => PortSide::Left,
                    crate::port::PortPosition::Right => PortSide::Right,
                    crate::port::PortPosition::Top => PortSide::Top,
                    crate::port::PortPosition::Bottom => PortSide::Bottom,
                };
                sides
                    .entry((node.id.clone(), side))
                    .or_default()
                    .push((addr, template_idx as f32));
            }
        }
        // Sort each (node, side) bucket by template index so ports
        // keep the order their template declares regardless of which
        // are currently connected. The position *along* a side is
        // still uniform (each port lands at `(i + 0.5)/N`), so the
        // node visually grows to accommodate multiple ports without
        // the slots ever swapping rank.
        for entries in sides.values_mut() {
            entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Per-node effective height: bounds height grows to
        // accommodate the L/R side with the most ports, with
        // enough clearance so ports never land inside the
        // rounded-corner zone. Header layout enforces its own
        // minimum via the slot tree's `total_height`.
        let port_d = theme.port_radius() * 2.0;
        // Gap between adjacent ports on the same side (between
        // edges, not centres). 12 reads as breathing-room
        // without ballooning the node.
        let port_spacing = 12.0_f32;
        let corner_radius = theme.node_corner_radius();
        let mut effective_bounds: std::collections::HashMap<NodeId, Rect> =
            std::collections::HashMap::new();
        for node in &frame.graph.nodes {
            if hidden_nodes.contains(&node.id) {
                continue;
            }
            let base = self.node_bounds_for(node, &theme);
            let l_count = sides
                .get(&(node.id.clone(), PortSide::Left))
                .map(|v| v.len())
                .unwrap_or(0);
            let r_count = sides
                .get(&(node.id.clone(), PortSide::Right))
                .map(|v| v.len())
                .unwrap_or(0);
            let max_lr = l_count.max(r_count) as f32;
            // Required vertical room: port centres distribute at
            // `(i + 0.5)/N` of the safe strip
            // `[corner_radius .. height - corner_radius]`. For
            // N ports the strip must fit N port diameters plus
            // (N - 1) inter-port gaps; adding 2 * corner_radius
            // restores the outer bounds. No extra edge buffer —
            // the corner-clearance inset already provides it.
            let required = if max_lr > 0.0 {
                2.0 * corner_radius + max_lr * port_d + (max_lr - 1.0).max(0.0) * port_spacing
            } else {
                0.0
            };
            let h = base.height().max(required);
            effective_bounds.insert(
                node.id.clone(),
                Rect::new(base.x(), base.y(), base.width(), h),
            );
        }
        // Final port positions — distribute INSIDE the corner-safe
        // strip on each side so ports don't visually clip the
        // node's rounded corner curve. Same idea for top/bottom
        // sides on the horizontal axis.
        let mut port_overrides: std::collections::HashMap<PortAddress, Point> =
            std::collections::HashMap::new();
        for ((node_id, side), entries) in &sides {
            let Some(bounds) = effective_bounds.get(node_id) else {
                continue;
            };
            let count = entries.len();
            // Strip inset = the rounded-corner clear zone. Port
            // centres land at `corner_radius + (i + 0.5)/N * (h - 2cr)`,
            // which keeps the dot's body off the corner curve
            // because the closest port centre to a corner sits at
            // `corner_radius + port_d/2` away from the corner.
            let inset = corner_radius;
            for (i, (addr, _)) in entries.iter().enumerate() {
                let t = (i as f32 + 0.5) / count as f32; // 0..1 within safe strip
                let pt = match side {
                    PortSide::Left => {
                        let strip_top = bounds.y() + inset;
                        let strip_h = (bounds.height() - inset * 2.0).max(0.0);
                        Point::new(bounds.x(), strip_top + strip_h * t)
                    }
                    PortSide::Right => {
                        let strip_top = bounds.y() + inset;
                        let strip_h = (bounds.height() - inset * 2.0).max(0.0);
                        Point::new(bounds.x() + bounds.width(), strip_top + strip_h * t)
                    }
                    PortSide::Top => {
                        let strip_left = bounds.x() + inset;
                        let strip_w = (bounds.width() - inset * 2.0).max(0.0);
                        Point::new(strip_left + strip_w * t, bounds.y())
                    }
                    PortSide::Bottom => {
                        let strip_left = bounds.x() + inset;
                        let strip_w = (bounds.width() - inset * 2.0).max(0.0);
                        Point::new(strip_left + strip_w * t, bounds.y() + bounds.height())
                    }
                };
                port_overrides.insert(addr.clone(), pt);
            }
        }

        // Animation clock for state-driven edge motion (Running
        // shimmer + Pending pulse). Computed once per frame so
        // every connection animates against the same timeline.
        let time_secs = animation_time_secs();
        let mut any_animated_edge = false;
        let mut selected_edge_buttons: Vec<(crate::connection::ConnectionId, Point)> = Vec::new();
        // Soft-disabled node set — looked up per-edge to downgrade
        // incident connections to Pending + per-node to dim the
        // body paint. Includes both directly-flagged nodes and
        // members of any disabled group so a "disable this group"
        // toggle dims the whole subgraph in one shot. A member's
        // own `disabled` flag stays untouched — re-enabling the
        // group restores each node to its previous state.
        let disabled_groups: std::collections::HashSet<&GroupId> = frame
            .graph
            .groups
            .iter()
            .filter(|g| g.disabled)
            .map(|g| &g.id)
            .collect();
        let mut disabled_nodes: std::collections::HashSet<NodeId> = frame
            .graph
            .nodes
            .iter()
            .filter(|n| n.disabled)
            .map(|n| n.id.clone())
            .collect();
        for grp in &frame.graph.groups {
            if disabled_groups.contains(&grp.id) {
                for m in &grp.members {
                    disabled_nodes.insert(m.clone());
                }
            }
        }
        // Cull counters — incremented each time a node or edge
        // survives the frustum check and reaches its draw call.
        // Stored at end of frame via `render_stats.store(...)` so
        // hosts can read `last_render_stats()` to surface a
        // "X / Y visible" HUD.
        let total_nodes = frame
            .graph
            .nodes
            .iter()
            .filter(|n| !hidden_nodes.contains(&n.id))
            .count();
        let total_edges = frame.graph.connections.len();
        let mut visible_nodes = 0_usize;
        let mut visible_edges = 0_usize;
        // Subgraph-reference nodes have no template ports, so a
        // connection terminating at one resolves to `None` via the
        // normal port-point lookup. Mirror the collapsed-group
        // routing convention: such endpoints snap to the closest
        // perimeter point on the subgraph node's bounds rect so the
        // line visually "flows into" the diamond body — the user's
        // signal that the connection routes through the subgraph
        // boundary. Built once per frame so the inner loop just hits
        // a hashset.
        let subgraph_ref_nodes: std::collections::HashSet<NodeId> = frame
            .graph
            .nodes
            .iter()
            .filter(|n| n.subgraph_ref.is_some())
            .map(|n| n.id.clone())
            .collect();
        for conn in &frame.graph.connections {
            // Internal connections — both endpoints in the SAME
            // collapsed group — get folded entirely inside the
            // chip. No external visual.
            let from_grp = node_to_collapsed_group.get(&conn.from.node);
            let to_grp = node_to_collapsed_group.get(&conn.to.node);
            if let (Some(a), Some(b)) = (from_grp, to_grp) {
                if a == b {
                    continue;
                }
            }
            let from_sub = if subgraph_ref_nodes.contains(&conn.from.node) {
                Some(&conn.from.node)
            } else {
                None
            };
            let to_sub = if subgraph_ref_nodes.contains(&conn.to.node) {
                Some(&conn.to.node)
            } else {
                None
            };
            // For visible (non-hidden) endpoints, prefer the sliding
            // `port_overrides` position (closest perimeter point
            // facing the counterpart node centre). Fall back to the
            // slot-based fixed position when the override is missing
            // (e.g. unconnected port, currently unreachable in this
            // loop but kept defensive). For endpoints folded into a
            // collapsed group, defer the point: we need the OTHER
            // endpoint resolved first so we can pick the closest
            // perimeter point on the group's body rect.
            let raw_from = if from_grp.is_some() || from_sub.is_some() {
                None
            } else {
                port_overrides.get(&conn.from).copied().or_else(|| {
                    resolve_port_point(
                        &frame.graph.nodes,
                        &frame.templates,
                        &conn.from,
                        &theme,
                        &slot_lookup,
                    )
                })
            };
            let raw_to = if to_grp.is_some() || to_sub.is_some() {
                None
            } else {
                port_overrides.get(&conn.to).copied().or_else(|| {
                    resolve_port_point(
                        &frame.graph.nodes,
                        &frame.templates,
                        &conn.to,
                        &theme,
                        &slot_lookup,
                    )
                })
            };
            // Merge collapsed endpoints to the closest point on the
            // collapsed group's perimeter, OR — for endpoints that
            // resolve to a subgraph-reference node — to the closest
            // point on that diamond's bounding rect. The latter
            // mirrors the collapsed-group convention: lines visually
            // "flow into" the diamond body, terminating at the side
            // that faces the counterpart endpoint. Falls back to
            // skip if a raw endpoint is missing (stale connection)
            // or rect lookup misses (first-frame edge case).
            let subgraph_rect = |id: &NodeId| effective_bounds.get(id).copied();
            let from_pt = match (raw_from, from_grp, from_sub) {
                (Some(p), _, _) => p,
                (None, Some(g), _) => {
                    let Some(rect) = collapsed_group_rects.get(g) else {
                        continue;
                    };
                    let anchor = raw_to
                        .or_else(|| {
                            to_grp.and_then(|g2| {
                                collapsed_group_rects.get(g2).map(|r| {
                                    Point::new(r.x() + r.width() * 0.5, r.y() + r.height() * 0.5)
                                })
                            })
                        })
                        .or_else(|| {
                            to_sub.and_then(|n| {
                                subgraph_rect(n).map(|r| {
                                    Point::new(r.x() + r.width() * 0.5, r.y() + r.height() * 0.5)
                                })
                            })
                        })
                        .unwrap_or_else(|| {
                            Point::new(rect.x() + rect.width() * 0.5, rect.y() - 20.0)
                        });
                    closest_point_on_rect(*rect, anchor)
                }
                (None, None, Some(node_id)) => {
                    let Some(rect) = subgraph_rect(node_id) else {
                        continue;
                    };
                    let anchor = raw_to
                        .or_else(|| {
                            to_grp.and_then(|g2| {
                                collapsed_group_rects.get(g2).map(|r| {
                                    Point::new(r.x() + r.width() * 0.5, r.y() + r.height() * 0.5)
                                })
                            })
                        })
                        .or_else(|| {
                            to_sub.and_then(|n| {
                                subgraph_rect(n).map(|r| {
                                    Point::new(r.x() + r.width() * 0.5, r.y() + r.height() * 0.5)
                                })
                            })
                        })
                        .unwrap_or_else(|| {
                            Point::new(rect.x() + rect.width() * 0.5, rect.y() - 20.0)
                        });
                    closest_point_on_rect(rect, anchor)
                }
                (None, None, None) => continue,
            };
            let to_pt = match (raw_to, to_grp, to_sub) {
                (Some(p), _, _) => p,
                (None, Some(g), _) => {
                    let Some(rect) = collapsed_group_rects.get(g) else {
                        continue;
                    };
                    let anchor = from_pt;
                    closest_point_on_rect(*rect, anchor)
                }
                (None, None, Some(node_id)) => {
                    let Some(rect) = subgraph_rect(node_id) else {
                        continue;
                    };
                    closest_point_on_rect(rect, from_pt)
                }
                (None, None, None) => continue,
            };
            // Frustum cull edges whose endpoint pair AABB doesn't
            // touch the visible rect. Bezier curves can arc slightly
            // past the from/to bbox, but the 25% slack in
            // `visible_content_rect_padded` more than absorbs typical
            // mid-x control offsets.
            let edge_aabb = Rect::new(
                from_pt.x.min(to_pt.x),
                from_pt.y.min(to_pt.y),
                (from_pt.x - to_pt.x).abs().max(1.0),
                (from_pt.y - to_pt.y).abs().max(1.0),
            );
            if !frustum.intersects(&edge_aabb) {
                continue;
            }

            let region_id = RegionId::Edge(conn.id).encode();
            let is_selected = frame.selection.selected.contains(&region_id);
            let is_hovered = hovered_edge.as_ref() == Some(&conn.id);
            // Soft-disable downgrades incident edges to Pending so
            // the dataflow visually breaks at a disabled node —
            // matching the dimmed node body. Connection's own state
            // stays untouched; we just paint differently.
            let effective_state = if disabled_nodes.contains(&conn.from.node)
                || disabled_nodes.contains(&conn.to.node)
            {
                crate::connection::ConnectionState::Pending
            } else {
                conn.state
            };
            crate::render::draw_edge_with_state(
                ctx,
                effective_state,
                from_pt,
                to_pt,
                &theme,
                is_selected,
                is_hovered,
                time_secs,
            );
            visible_edges += 1;
            if matches!(
                effective_state,
                crate::connection::ConnectionState::Running
                    | crate::connection::ConnectionState::Pending
            ) {
                any_animated_edge = true;
            }

            // Register segment-aabb hit regions along the curve so
            // hover + click + select work on the visible curve, not
            // just a fat single-AABB approximation. The 8-segment
            // sample matches the look-thickness budget the curve
            // already renders against; we share the same region id
            // across every segment so the click handler doesn't need
            // to disambiguate.
            let (c1, c2) = crate::bezier::mid_x_controls(from_pt, to_pt);
            let hit_thickness = theme.edge_thickness() + 8.0;
            for bbox in crate::bezier::segment_bboxes(from_pt, c1, c2, to_pt, 8, hit_thickness) {
                self.kit.hit_rect(region_id.clone(), bbox);
            }

            if is_selected {
                selected_edge_buttons.push((
                    conn.id,
                    crate::bezier::cubic_midpoint(from_pt, c1, c2, to_pt),
                ));
            }
        }

        // 3. Nodes — body + per-side ports + hit regions.
        for node in &frame.graph.nodes {
            // Members of a collapsed group are visually folded into
            // the group chip; skip the draw + hit-region register
            // for them. Position state stays untouched so expanding
            // the group restores them in place.
            if hidden_nodes.contains(&node.id) {
                continue;
            }
            let Some(template) = frame.templates.get(&node.component) else {
                continue;
            };
            // Frustum cull — skip nodes outside the visible content
            // rect. The slot lookup + portal.frame are the heaviest
            // per-node work in this loop, so the early-out saves
            // real cycles at high node counts. Use the unmodified
            // `node_bounds_for` (pre-effective-bounds growth) since
            // we need a cheap bounds before paying for slots; the
            // padding in `visible_content_rect_padded` covers the
            // worst-case extra height from port-driven growth.
            let cull_bounds = self.node_bounds_for(node, &theme);
            if !frustum.intersects(&cull_bounds) {
                continue;
            }
            visible_nodes += 1;
            let slots = self.node_slots_for(template, node, &theme);
            // CanvasKit stores selected REGION IDs (with the
            // `node:` / `group:` prefix), not raw model IDs — match
            // the same prefix here so the editor highlights nodes
            // clicked via canvas-kit's automatic POINTER_DOWN
            // selection.
            let region_id = RegionId::Node(node.id.clone()).encode();
            let is_selected = frame.selection.selected.contains(&region_id);
            // Effective bounds — driven up by port-count when L/R
            // sides hold multiple connections. Stays at base
            // node_bounds otherwise.
            let bounds = effective_bounds
                .get(&node.id)
                .copied()
                .unwrap_or_else(|| self.node_bounds_for(node, &theme));
            // `disabled_nodes` already absorbs group-inherited
            // disable, so the renderer treats group + node disables
            // uniformly.
            let node_disabled = disabled_nodes.contains(&node.id);
            let badge_rect = draw_node_at(
                ctx,
                node,
                template,
                bounds,
                &theme,
                &slots,
                is_selected,
                node_disabled,
            );
            self.kit.hit_rect(region_id, bounds);
            // Register the badge as its own hit region when it
            // carries a tooltip, so the badge-hover branch in the
            // render pass can light up the tooltip chip without
            // promoting every node-with-a-badge into a permanent
            // hovered-tooltip state. The badge region is registered
            // AFTER the body region so canvas-kit's reverse-order
            // hit-test picks the badge first when the pointer is
            // inside the badge chip.
            if let (Some(rect), Some(badge)) = (badge_rect, node.badge.as_ref()) {
                if badge.tooltip.is_some() {
                    let region = RegionId::NodeBadge(node.id.clone()).encode();
                    self.kit.hit_rect(region.clone(), rect);
                    self.badge_rects.lock().unwrap().insert(region, rect);
                }
            }

            // Render the template's content slot via a portal-ui
            // mini-runtime. The body rect is the absolute translation
            // of `slots.body` (which taffy sized from
            // `template.content.height`). The portal owns its own
            // widget storage + signal subscriptions; signal changes
            // anywhere in the host repaint the canvas on the next
            // frame via the portal-ui notifier hook.
            if let Some(content) = template.content.as_ref() {
                // Portal paints into the INNER `portal` rect inside
                // the inset background. `crate::render::draw_node_at`
                // painted the inset itself a moment ago; here we
                // use the same helper to inset the portal a step
                // further so widgets sit cleanly inside the inset.
                let Some(slot_rects) = crate::render::content_slot_rects(bounds, &slots) else {
                    continue;
                };
                let body_rect = slot_rects.portal;
                let mut portals = self.portals.lock().unwrap();
                let portal = portals.get_or_make(node.id.clone());
                let style = blinc_portal_ui::PortalStyle::from_active_theme();
                // Pure pass-through bridge — node-editor's canvas
                // is rendered IN canvas-content space already (the
                // closure receives content-space bounds), so the
                // portal's coords ARE canvas-content coords. Overlay
                // anchoring requires screen-space; the bridge wraps
                // the kit's content_to_screen for that.
                let kit_for_bridge = self.kit.clone();
                let kit_for_inverse = self.kit.clone();
                let host = blinc_portal_ui::HostBridge::from_closures(
                    move |p| kit_for_bridge.content_to_screen(p),
                    move |p| kit_for_inverse.screen_to_content(p),
                );
                let render = content.render.clone();
                let node_id = node.id.clone();
                // Match the inset background's corner radius
                // (`draw_node_at` uses `radius * 0.7`) so the
                // portal clip cuts widget paint along the same
                // curve the inset paints — no glyph or sparkline
                // tail bleeds past the visible inset edge.
                let inset_radius = theme.node_corner_radius() * 0.7;
                let portal_animating = portal
                    .begin(ctx, &self.kit, body_rect)
                    .clip_radius(inset_radius)
                    .style(&style)
                    .host(&host)
                    .run(|ui| render(&node_id, ui))
                    .needs_redraw();
                // Feed the portal's measured content height back
                // to the slot tree for the NEXT frame so the body
                // grows to fit. Triggers a one-frame lag the first
                // time a node's closure overflows the template's
                // declared min height; subsequent frames are
                // stable. Only writes when the value changes
                // meaningfully to avoid pointless cache churn.
                let consumed = portal.consumed_height();
                let mut heights = self.portal_content_heights.lock().unwrap();
                let prev = heights.get(&node.id).copied().unwrap_or(0.0);
                if (consumed - prev).abs() > 0.5 {
                    heights.insert(node.id.clone(), consumed);
                    blinc_layout::request_redraw();
                }
                drop(heights);

                // Same shape on the horizontal axis — drives fit-
                // content node width. Quantise to a 4 px grid so the
                // `NodeSlotCache` fingerprint (keyed on width.to_bits)
                // doesn't bloat with sub-pixel drift, and only write
                // when meaningfully changed.
                let consumed_w_raw = portal.consumed_width();
                let consumed_w = (consumed_w_raw / 4.0).round() * 4.0;
                let mut widths = self.portal_content_widths.lock().unwrap();
                let prev_w = widths.get(&node.id).copied().unwrap_or(0.0);
                if (consumed_w - prev_w).abs() > 0.5 {
                    widths.insert(node.id.clone(), consumed_w);
                    blinc_layout::request_redraw();
                }
                drop(widths);

                if portal_animating {
                    any_animated_edge = true;
                }
            }

            // Ports. Connected ports use the sliding override
            // position (closest perimeter point facing their
            // counterpart); unconnected ports fall back to the
            // fixed slot position so they still appear somewhere
            // sensible. The port circle + hit region anchor at
            // the same point the connection loop terminates at,
            // so click targets stay in sync visually.
            for (_dir, desc, slot_centre) in
                iter_port_positions(node, &template.inputs, &template.outputs, &slots, &theme)
            {
                let addr = PortAddress::new(node.id.clone(), desc.id.clone());
                let centre = port_overrides.get(&addr).copied().unwrap_or(slot_centre);
                let hover_state = drag_port_hover_state(&frame.drag, node, desc);
                draw_port(ctx, desc, centre, &theme, hover_state);

                let r = theme.port_radius() + 4.0;
                let port_rect = Rect::new(centre.x - r, centre.y - r, r * 2.0, r * 2.0);
                let region_id = RegionId::Port(PortAddress {
                    node: node.id.clone(),
                    port: desc.id.clone(),
                })
                .encode();
                self.kit.hit_rect(region_id, port_rect);
            }
        }

        // 4. Drag-to-connect preview.
        if let DragConnect::Dragging { cursor, .. } | DragConnect::Hovering { cursor, .. } =
            &frame.drag
        {
            if let Some(from) = frame.drag.from_port() {
                // Source endpoint must match where the port is
                // actually DRAWN. Connected ports get an override
                // position from `port_overrides` (the sliding
                // perimeter-point shared with `draw_port` at the
                // call site below); the resolved slot centre is
                // only used when the port is unconnected. Falling
                // back to the slot centre for a connected port
                // would snap the drag-preview's source to the
                // port's default position rather than its drawn
                // position — a vertical offset users can see
                // clearly.
                let from_pt = port_overrides.get(from).copied().or_else(|| {
                    resolve_port_point(
                        &frame.graph.nodes,
                        &frame.templates,
                        from,
                        &theme,
                        &slot_lookup,
                    )
                });
                if let Some(from_pt) = from_pt {
                    let compatible = match &frame.drag {
                        DragConnect::Hovering { validation, .. } => Some(validation.is_accept()),
                        _ => None,
                    };
                    crate::render::draw_drag_preview(ctx, from_pt, *cursor, &theme, compatible);
                }
            }
        }

        // 5. Edge delete buttons — drawn + registered LAST so they
        //    sit visually on top of every other edge / node / port
        //    AND their hit regions take priority in canvas-kit's
        //    reverse-order hit-test scan. Without this final pass,
        //    a crossing edge or a node positioned near the cubic
        //    midpoint could swallow the click on the × button.
        for (conn_id, mid) in selected_edge_buttons {
            let btn_rect = crate::render::draw_edge_delete_button(ctx, mid, &theme);
            self.kit
                .hit_rect(RegionId::EdgeDelete(conn_id).encode(), btn_rect);
        }

        // 6. Port tooltip — drawn LAST so it composites above every
        //    node, port, and edge. Anchored to the hovered port's
        //    centre via the existing slot_lookup helper. Suppressed
        //    while a drag-to-connect gesture is in flight (the
        //    candidate-port outline + drag preview already convey
        //    intent; a tooltip on top would just visual-noise the
        //    drag).
        if !frame.drag.is_active() {
            if let Some(HoverTarget::Port(ref addr)) = self.hovered() {
                if let Some(node) = frame.graph.nodes.iter().find(|n| n.id == addr.node) {
                    if let Some(template) = frame.templates.get(&node.component) {
                        let slots = self.node_slots_for(template, node, &theme);
                        for (_dir, desc, port_centre) in iter_port_positions(
                            node,
                            &template.inputs,
                            &template.outputs,
                            &slots,
                            &theme,
                        ) {
                            if desc.id == addr.port {
                                let (sw, sh) = self.kit.screen_bounds();
                                let viewport = if sw > 0.0 && sh > 0.0 {
                                    Some(blinc_core::Rect::new(0.0, 0.0, sw, sh))
                                } else {
                                    None
                                };
                                crate::render::draw_port_tooltip_clamped(
                                    ctx,
                                    desc,
                                    port_centre,
                                    &theme,
                                    viewport,
                                    self.kit.viewport().transform(),
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }

        // 7. Badge tooltip — drawn LAST in the same family as the
        //    port tooltip (above every other chrome). Anchored to
        //    the hovered badge's rect. Pulls the badge from the
        //    hovered node or group; suppressed during drag-connect
        //    so the badge-tooltip chip doesn't compete visually
        //    with the live connection preview.
        if !frame.drag.is_active() {
            let (sw, sh) = self.kit.screen_bounds();
            let viewport = if sw > 0.0 && sh > 0.0 {
                Some(blinc_core::Rect::new(0.0, 0.0, sw, sh))
            } else {
                None
            };
            match self.hovered() {
                Some(HoverTarget::NodeBadge(ref node_id)) => {
                    let region = RegionId::NodeBadge(node_id.clone()).encode();
                    let rect = self.badge_rects.lock().unwrap().get(&region).copied();
                    if let (Some(node), Some(rect)) =
                        (frame.graph.nodes.iter().find(|n| n.id == *node_id), rect)
                    {
                        if let Some(tip) = node.badge.as_ref().and_then(|b| b.tooltip.as_deref()) {
                            crate::render::draw_badge_tooltip(
                                ctx,
                                tip,
                                rect,
                                &theme,
                                viewport,
                                self.kit.viewport().transform(),
                            );
                        }
                    }
                }
                Some(HoverTarget::GroupBadge(ref group_id)) => {
                    let region = RegionId::GroupBadge(group_id.clone()).encode();
                    let rect = self.badge_rects.lock().unwrap().get(&region).copied();
                    if let (Some(group), Some(rect)) =
                        (frame.graph.groups.iter().find(|g| g.id == *group_id), rect)
                    {
                        if let Some(tip) = group.badge.as_ref().and_then(|b| b.tooltip.as_deref()) {
                            crate::render::draw_badge_tooltip(
                                ctx,
                                tip,
                                rect,
                                &theme,
                                viewport,
                                self.kit.viewport().transform(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // Self-sustaining frame loop for state animations. Both
        // signals are needed every frame:
        //
        // * `request_animation_tick` flips a dedicated atomic the
        //   windowed app's end-of-frame redraw chain ORs into
        //   `any_redraw_signal`, so the next vsync is requested
        //   even when nothing else is animating.
        // * `request_redraw` flips the stateful-redraw atomic so
        //   the Frame's start-of-frame `peek_needs_redraw` check
        //   admits the paint when there's no other dirty source.
        //
        // Both auto-clear when `any_animated_edge` flips false: the
        // take-and-clear cycle for `animation_tick_request` runs in
        // windowed.rs every frame, so the chain quiesces the moment
        // the source stops re-asking.
        //
        // CPU cost: this runs the full editor walker at vsync while
        // any Running/Pending edge exists. Hosts that want to cap
        // the frame rate (often acceptable for "background runtime"
        // animation that doesn't need 60 fps) can set
        // `WindowConfig::animation_fps_cap` — that throttles every
        // animation-driven redraw in the app, including this one,
        // without losing the chain.
        if any_animated_edge {
            blinc_layout::request_animation_tick();
            blinc_layout::request_redraw();
        }

        // GC portals for nodes no longer in the graph. `Drop` on
        // `Portal` unsubscribes any signals it was tracking so a
        // removed content-node stops dirtying the canvas. Same
        // pass prunes the consumed-height feedback map so a node
        // re-added later doesn't inherit a stale height.
        {
            let live: HashSet<NodeId> = frame.graph.nodes.iter().map(|n| n.id.clone()).collect();
            self.portals.lock().unwrap().retain(|id| live.contains(id));
            self.portal_content_heights
                .lock()
                .unwrap()
                .retain(|id, _| live.contains(id));
            self.portal_content_widths
                .lock()
                .unwrap()
                .retain(|id, _| live.contains(id));
        }

        // Commit cull stats — `last_render_stats()` returns this
        // snapshot to any host HUD watching the cull effect.
        self.render_stats.store(RenderStats {
            total_nodes,
            visible_nodes,
            total_edges,
            visible_edges,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────

/// Compute the union of all member-node bounding rects for a group.
/// Used when `group.bounds` is `None` (auto-fit mode).
///
/// `bounds_for` resolves each node's CURRENT bounding rect — typically
/// `self.node_bounds_for(node, &theme)`, which reads the slot tree's
/// `total_height`. That means content-slot nodes (whose taffy height
/// includes the portal body) contribute their REAL extent here, not
/// the stale `instance.size` hint the host passed at creation time.
/// Reading from `instance.size` directly would clip the group
/// chrome above a portal-grown node's true bottom when the user
/// dragged the node into the group.
fn compute_group_auto_bounds<N, G>(
    group: &Group<G>,
    nodes: &[NodeInstance<N>],
    bounds_for: impl Fn(&NodeInstance<N>) -> Rect,
) -> Rect {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;

    for node in nodes {
        if !group.members.contains(&node.id) {
            continue;
        }
        let b = bounds_for(node);
        min_x = min_x.min(b.x());
        min_y = min_y.min(b.y());
        max_x = max_x.max(b.x() + b.width());
        max_y = max_y.max(b.y() + b.height());
        any = true;
    }

    if any {
        Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    } else {
        // Empty group → render a small chrome-only chip near origin
        // until the host populates members. Host can override with
        // explicit `bounds` if it wants a different placement.
        Rect::new(0.0, 0.0, 160.0, 40.0)
    }
}

/// Variant of [`compute_group_auto_bounds`] that ignores one member.
///
/// Used by membership detection so a dragged node can escape its
/// own group: the visible auto-bounds expand with the node (good
/// feedback), but the "is the node still inside?" test runs
/// against the *other* members' footprint, which does NOT track
/// the dragged node. Returns `None` when the group has no other
/// members (excluding the dragged node would leave the group
/// empty; that's still a "remove" so the caller handles it).
fn compute_group_auto_bounds_excluding<N, G>(
    group: &Group<G>,
    nodes: &[NodeInstance<N>],
    exclude: &NodeId,
    bounds_for: impl Fn(&NodeInstance<N>) -> Rect,
) -> Option<Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;

    for node in nodes {
        if !group.members.contains(&node.id) || node.id == *exclude {
            continue;
        }
        let b = bounds_for(node);
        min_x = min_x.min(b.x());
        min_y = min_y.min(b.y());
        max_x = max_x.max(b.x() + b.width());
        max_y = max_y.max(b.y() + b.height());
        any = true;
    }
    any.then(|| Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

/// Multi-node variant of [`compute_group_auto_bounds_excluding`].
/// Used by group-drag detach: every node being dragged together must
/// be excluded so the parent group's "stationary baseline" reflects
/// only the non-dragged remainder. If the parent's members are
/// entirely contained in `exclude`, returns `None` — caller treats
/// that as "no baseline, so every dragged member escapes."
fn compute_group_auto_bounds_excluding_set<N, G>(
    group: &Group<G>,
    nodes: &[NodeInstance<N>],
    exclude: &[NodeId],
    bounds_for: impl Fn(&NodeInstance<N>) -> Rect,
) -> Option<Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;

    for node in nodes {
        if !group.members.contains(&node.id) || exclude.contains(&node.id) {
            continue;
        }
        let b = bounds_for(node);
        min_x = min_x.min(b.x());
        min_y = min_y.min(b.y());
        max_x = max_x.max(b.x() + b.width());
        max_y = max_y.max(b.y() + b.height());
        any = true;
    }
    any.then(|| Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

/// Resolve the centre point of a port referenced by [`PortAddress`].
/// Returns `None` if the node or port doesn't exist (stale connection
/// — the editor draws nothing rather than panicking). Takes a
/// `slot_lookup` closure so the caller can use either the cache or
/// a freshly-computed slot table.
fn resolve_port_point<K: PortKind, N>(
    nodes: &[NodeInstance<N>],
    templates: &AHashMap<String, NodeTemplate<K>>,
    addr: &PortAddress,
    theme: &ThemeResolver<'_>,
    slot_lookup: &dyn Fn(&NodeTemplate<K>, &NodeInstance<N>) -> NodeSlots,
) -> Option<blinc_core::layer::Point> {
    let node = nodes.iter().find(|n| n.id == addr.node)?;
    let template = templates.get(&node.component)?;
    let slots = slot_lookup(template, node);
    for (_dir, desc, centre) in
        iter_port_positions(node, &template.inputs, &template.outputs, &slots, theme)
    {
        if desc.id == addr.port {
            return Some(centre);
        }
    }
    None
}

/// Look up the host port-kind for the address by walking the
/// template's input + output lists. Returns `None` if the node /
/// template / port can't be resolved.
fn lookup_port_kind<K, N, C, G>(editor: &NodeEditor<K, N, C, G>, addr: &PortAddress) -> Option<K>
where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    let graph = editor.graph.read().unwrap();
    let templates = editor.templates.read().unwrap();
    let node = graph.nodes.iter().find(|n| n.id == addr.node)?;
    let template = templates.get(&node.component)?;
    template
        .inputs
        .iter()
        .chain(template.outputs.iter())
        .find(|p| p.id == addr.port)
        .map(|p| p.kind.clone())
}

/// Drive the [`DragConnect`] state machine from a per-frame
/// drag-tick over a port hit region. Called from the
/// `on_element_drag` handler when the active region's id starts
/// with `port:`.
///
/// State transitions inside one drag session:
///   * First tick from `Idle` → `begin(from_addr, cursor)`.
///   * Subsequent ticks: hit-test the cursor; if it lands on
///     another port, run the host's `on_validate` callback and
///     transition `Hovering`; otherwise `move_to(cursor)`.
fn update_port_drag<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    from_addr: PortAddress,
    cursor: blinc_core::layer::Point,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    // First tick: bring the FSM out of Idle.
    {
        let mut drag = editor.drag_state();
        if !drag.is_active() {
            drag.begin(from_addr.clone(), cursor);
            editor.drag_state.set(drag);
        }
    }

    // Probe the cursor's current region. Skip the source port
    // (own region) so the FSM doesn't flicker into Hovering
    // against itself.
    let target_region = editor.kit.hit_test(cursor);
    let source_region = RegionId::Port(from_addr.clone()).encode();
    let to_addr = match target_region {
        Some(ref r) if r != &source_region => match RegionId::parse(r) {
            Some(RegionId::Port(addr)) => Some(addr),
            _ => None,
        },
        _ => None,
    };

    let mut drag = editor.drag_state();
    if let Some(to_addr) = to_addr {
        // Look up kinds + run validator.
        let from_kind = lookup_port_kind(editor, &from_addr);
        let to_kind = lookup_port_kind(editor, &to_addr);
        let validation = match (from_kind, to_kind) {
            (Some(fk), Some(tk)) => {
                let cb_slot = editor.on_validate.read().unwrap();
                match cb_slot.as_ref() {
                    Some(cb) => {
                        let req = crate::connection::ConnectRequest {
                            from: &from_addr,
                            from_kind: &fk,
                            to: &to_addr,
                            to_kind: &tk,
                        };
                        cb(&req)
                    }
                    None => {
                        // No validator supplied — fall back to the
                        // host's `PortKind::compatible_with` so
                        // hosts that wired typed ports still get
                        // type-aware preview tinting for free.
                        if fk.compatible_with(&tk) {
                            crate::connection::ValidationOutcome::Accept
                        } else {
                            crate::connection::ValidationOutcome::Reject {
                                reason: format!(
                                    "kinds incompatible: {} → {}",
                                    fk.label(),
                                    tk.label()
                                ),
                            }
                        }
                    }
                }
            }
            _ => crate::connection::ValidationOutcome::Reject {
                reason: "port kind not resolvable".into(),
            },
        };
        drag.hover(to_addr, validation, cursor);
    } else {
        // Cursor left the port region. `move_to` alone would keep the
        // FSM latched in `Hovering { candidate, validation }` against
        // whatever port we last touched — so a release on empty space
        // (or after passing through another port the user changed
        // their mind about) would wrongly land on that stale
        // candidate. Drop back to `Dragging` first, then update the
        // cursor so the rubber-band preview still tracks the pointer.
        drag.unhover();
        drag.move_to(cursor);
    }
    editor.drag_state.set(drag);
}

/// Drag-end side of the port-connect flow. Reads the current
/// `DragConnect` state; if release lands on a candidate the
/// validator already accepted, pushes
/// [`EditorEvent::ConnectionAccepted`] onto the editor's queue
/// with a fully-typed [`crate::connection::ConnectionEvent`].
/// Always resets the FSM to `Idle`.
///
/// `release_cursor` is the cursor position at the moment of release
/// (from the drag-end event). The FSM's existing `Hovering` state
/// from the last per-frame drag tick can be stale (winit doesn't
/// guarantee a final `Drag` event flush at the release coordinate),
/// so we re-run `update_port_drag` against `release_cursor` first.
/// That guarantees the candidate / validation pair read by
/// `release()` reflects the port actually under the cursor at the
/// click-up — not whichever port the user happened to be over one
/// frame earlier.
fn finalise_port_drag<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    release_cursor: blinc_core::layer::Point,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    use crate::interaction::DragRelease;
    // Re-resolve the hover state against the actual release
    // coordinate so the FSM lands in the correct accept / reject
    // arm even on a flick-release that fires before the next
    // drag tick.
    let from_addr_for_replay = editor.drag_state().from_port().cloned();
    if let Some(addr) = from_addr_for_replay {
        update_port_drag(editor, addr, release_cursor);
    }
    let mut drag = editor.drag_state();
    let from_addr = drag.from_port().cloned();
    let release = drag.release();
    editor.drag_state.set(drag);
    let Some(from) = from_addr else {
        return;
    };
    match release {
        DragRelease::Accepted(to) => {
            let from_kind = lookup_port_kind(editor, &from);
            let to_kind = lookup_port_kind(editor, &to);
            if let (Some(fk), Some(tk)) = (from_kind, to_kind) {
                editor.push_event(EditorEvent::ConnectionAccepted(
                    crate::connection::ConnectionEvent {
                        from,
                        from_kind: fk,
                        to,
                        to_kind: tk,
                    },
                ));
            }
        }
        DragRelease::Rejected { candidate, reason } => {
            // Broadcast the validator's reason so the host can surface
            // a toast / banner. The live red preview already conveyed
            // THAT the connection was rejected; this event delivers
            // the textual WHY for hosts that want it.
            editor.push_event(EditorEvent::ConnectionRejected {
                from,
                to: candidate,
                reason,
            });
        }
        DragRelease::Empty => {
            // Released on empty canvas — nothing to broadcast.
        }
    }
}

/// Apply a drag delta to whichever entity owns the registered
/// hit region. Region IDs are produced by the renderer:
///
/// - `node:{node_id}` — drag the single node by `delta`.
/// - `group:{group_id}` — drag every member node of the group by
///   `delta`. Lets the user reposition an entire group by grabbing
///   the empty chrome between nodes (the node hit rects override
///   the group rect when the pointer is on a member, so node drags
///   still work normally).
fn apply_drag_delta<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    region_id: &str,
    delta: blinc_core::layer::Point,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    let touched = {
        let mut graph = editor.graph.write().unwrap();
        match RegionId::parse(region_id) {
            Some(RegionId::Node(node_id)) => {
                if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_id) {
                    node.position.x += delta.x;
                    node.position.y += delta.y;
                    true
                } else {
                    false
                }
            }
            Some(RegionId::Group(group_id)) => {
                let member_ids: Vec<NodeId> = graph
                    .groups
                    .iter()
                    .find(|g| g.id == group_id)
                    .map(|g| g.members.clone())
                    .unwrap_or_default();
                let mut any = false;
                for node in graph.nodes.iter_mut() {
                    if member_ids.contains(&node.id) {
                        node.position.x += delta.x;
                        node.position.y += delta.y;
                        any = true;
                    }
                }
                any
            }
            _ => false,
        }
    };
    if touched {
        editor.bump_graph_rev();
    }
}

/// After a node drag settles, fire `AddToGroupRequested` /
/// `RemoveFromGroupRequested` events if the node crossed a group
/// boundary.
///
/// `shift_held` gates the escape semantics:
///
/// * **Without Shift** (default drag): the dragged node's current
///   group auto-bounds expand with it, so it never registers as
///   "outside." We only fire `AddToGroupRequested` when the node
///   crosses INTO a group it wasn't already in. Membership never
///   shrinks on a plain drag — that prevents accidental drops in
///   large graphs where a member's natural movement could
///   otherwise pull it out.
/// * **With Shift held** at drag-end: the dragged node's CURRENT
///   group is tested against the bounds of its *other* members
///   (the node's own contribution to auto-fit is excluded), giving
///   the user an explicit escape gesture. Cross-group moves emit
///   both legs so hosts can audit each side independently.
///
/// Compute the dragged node's "before" / "after" group membership for
/// the live position. Shared between drag-end (event firing) and per-
/// frame drag (preview tinting). Returns `(before, after,
/// after_inside_any)` where `after_inside_any` flags whether the
/// chosen `after` group was hit via real bounds (vs falling through).
///
/// `shift_held` controls the current-group bounds source — see
/// [`detect_membership_changes`] for the semantics.
fn compute_drag_group_targets<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    dragged: &NodeId,
    shift_held: bool,
) -> Option<(Option<GroupId>, Option<GroupId>)>
where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    let graph = editor.graph.read().unwrap();
    let node = graph.nodes.iter().find(|n| n.id == *dragged)?;
    let before = graph
        .groups
        .iter()
        .find(|g| g.members.contains(dragged))
        .map(|g| g.id.clone());
    let (cx, cy) = {
        let (w, h) = node.size.unwrap_or((180.0, 72.0));
        (node.position.x + w * 0.5, node.position.y + h * 0.5)
    };
    let theme_overrides = editor.theme.read().unwrap();
    let theme = ThemeResolver::new(&theme_overrides);
    let pad = theme.group_padding();
    let mut hit: Option<GroupId> = None;
    for group in &graph.groups {
        let region_bounds = if let Some(explicit) = group.bounds {
            explicit
        } else {
            let auto_bounds = if shift_held && before.as_ref() == Some(&group.id) {
                match compute_group_auto_bounds_excluding(group, &graph.nodes, dragged, |n| {
                    editor.node_bounds_for(n, &theme)
                }) {
                    Some(b) => b,
                    None => continue,
                }
            } else {
                compute_group_auto_bounds(group, &graph.nodes, |n| {
                    editor.node_bounds_for(n, &theme)
                })
            };
            let header_h = editor
                .group_slots
                .read()
                .ok()
                .and_then(|cache| {
                    cache
                        .get(&fingerprint_group(
                            group,
                            &group_inputs_from(group, auto_bounds.width() + pad * 2.0, &theme),
                            editor.theme_revision.load(Ordering::Relaxed),
                        ))
                        .map(|s| s.header.height())
                })
                .unwrap_or(28.0);
            Rect::new(
                auto_bounds.x() - pad,
                auto_bounds.y() - pad - header_h,
                auto_bounds.width() + pad * 2.0,
                auto_bounds.height() + pad * 2.0 + header_h,
            )
        };
        if cx >= region_bounds.x()
            && cx <= region_bounds.x() + region_bounds.width()
            && cy >= region_bounds.y()
            && cy <= region_bounds.y() + region_bounds.height()
        {
            hit = Some(group.id.clone());
            break;
        }
    }
    Some((before, hit))
}

/// Update the editor's live drag-group preview based on the dragged
/// node's current position + the held modifier state. Called per-
/// frame from the drag handler; the renderer reads the preview and
/// tints group borders accordingly.
fn update_drag_group_preview<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    dragged: &NodeId,
    shift_held: bool,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    let Some((before, after)) = compute_drag_group_targets(editor, dragged, shift_held) else {
        return;
    };
    let next_preview = DragGroupPreview {
        dragged_node: Some(dragged.clone()),
        dragged_group: None,
        shift_held,
        // Add hint: the would-be target is a group the node ISN'T
        // already in. Suppresses the highlight when the user is just
        // wiggling within their current group.
        add_target: match (&before, &after) {
            (b, Some(next)) if b.as_ref() != Some(next) => Some(next.clone()),
            _ => None,
        },
        // Remove hint: the node is currently a member of a group but
        // the live position would take it out (cross-group move OR
        // escape to no-group). Only signal while Shift is held —
        // without Shift, drag-end can't actually remove anything.
        remove_target: if shift_held {
            match (&before, &after) {
                (Some(prev), Some(next)) if prev != next => Some(prev.clone()),
                (Some(prev), None) => Some(prev.clone()),
                _ => None,
            }
        } else {
            None
        },
    };
    let mut slot = editor.drag_group_preview.write().unwrap();
    if *slot != next_preview {
        *slot = next_preview;
        // Drop the lock before request_redraw so the next paint can
        // re-acquire it without contention.
        drop(slot);
        blinc_layout::request_redraw();
    }
}

/// Compare two `DragGroupPreview`s by value — derived `PartialEq`
/// equivalent via field equality, used inline above. Saves a derive
/// macro that would otherwise need `GroupId: PartialEq` (already
/// satisfied) plus an explicit derive on the struct.
impl PartialEq for DragGroupPreview {
    fn eq(&self, other: &Self) -> bool {
        self.dragged_node == other.dragged_node
            && self.dragged_group == other.dragged_group
            && self.shift_held == other.shift_held
            && self.add_target == other.add_target
            && self.remove_target == other.remove_target
    }
}

/// Multi-node sibling of [`update_drag_group_preview`] driving the
/// live tint while the user drags a Group container by its chrome.
/// Computes the parent-group escape preview using the dragged
/// group's FULL member set as the exclusion (so the per-node
/// "auto-bounds excluding me" trick — which fails when every member
/// is moving together — is sidestepped). Mirrors the single-node
/// path's policy: `remove_target` only fires while Shift is held.
fn update_drag_group_preview_for_group<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    dragged_group: &GroupId,
    shift_held: bool,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    let (dragged_members, parent_overlaps) = {
        let graph = editor.graph.read().unwrap();
        let Some(dg) = graph.groups.iter().find(|g| g.id == *dragged_group) else {
            return;
        };
        let members: Vec<NodeId> = dg.members.clone();
        if members.is_empty() {
            return;
        }
        let parents: Vec<(GroupId, Vec<NodeId>)> = graph
            .groups
            .iter()
            .filter(|g| g.id != *dragged_group)
            .filter_map(|g| {
                let overlap: Vec<NodeId> = members
                    .iter()
                    .filter(|m| g.members.contains(m))
                    .cloned()
                    .collect();
                if overlap.is_empty() {
                    None
                } else {
                    Some((g.id.clone(), overlap))
                }
            })
            .collect();
        (members, parents)
    };

    // Pick the FIRST parent whose remaining-baseline no longer
    // encloses any overlap member as the remove_target. Same
    // "first hit" policy as compute_drag_group_targets — the
    // renderer only tints one group at a time, and a single
    // wrapping subgraph almost never sits inside multiple parents.
    let remove_target: Option<GroupId> = if shift_held {
        let graph = editor.graph.read().unwrap();
        let theme_overrides = editor.theme.read().unwrap();
        let theme = ThemeResolver::new(&theme_overrides);
        let pad = theme.group_padding();
        let header_h_fallback = 28.0;
        let mut found: Option<GroupId> = None;
        'outer: for (parent_id, overlap_members) in &parent_overlaps {
            let Some(parent) = graph.groups.iter().find(|g| g.id == *parent_id) else {
                continue;
            };
            let region_bounds: Option<Rect> = if let Some(explicit) = parent.bounds {
                Some(explicit)
            } else {
                compute_group_auto_bounds_excluding_set(
                    parent,
                    &graph.nodes,
                    &dragged_members,
                    |n| editor.node_bounds_for(n, &theme),
                )
                .map(|auto_bounds| {
                    Rect::new(
                        auto_bounds.x() - pad,
                        auto_bounds.y() - pad - header_h_fallback,
                        auto_bounds.width() + pad * 2.0,
                        auto_bounds.height() + pad * 2.0 + header_h_fallback,
                    )
                })
            };
            for m in overlap_members {
                let Some(node) = graph.nodes.iter().find(|n| n.id == *m) else {
                    continue;
                };
                let (w, h) = node.size.unwrap_or((180.0, 72.0));
                let cx = node.position.x + w * 0.5;
                let cy = node.position.y + h * 0.5;
                let escapes = match region_bounds {
                    None => true,
                    Some(b) => {
                        !(cx >= b.x()
                            && cx <= b.x() + b.width()
                            && cy >= b.y()
                            && cy <= b.y() + b.height())
                    }
                };
                if escapes {
                    found = Some(parent_id.clone());
                    break 'outer;
                }
            }
        }
        found
    } else {
        None
    };

    let next_preview = DragGroupPreview {
        dragged_node: None,
        dragged_group: Some(dragged_group.clone()),
        shift_held,
        // add_target deliberately unset on group-drag: cross-parent
        // moves aren't part of the current gesture vocabulary
        // (a wrapping subgraph dropped into a brand-new parent
        // would need a separate UX decision). remove_target is
        // sufficient for the escape gesture the user is asking
        // about.
        add_target: None,
        remove_target,
    };
    let mut slot = editor.drag_group_preview.write().unwrap();
    if *slot != next_preview {
        *slot = next_preview;
        drop(slot);
        blinc_layout::request_redraw();
    }
}

fn detect_membership_changes<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    dragged: &NodeId,
    shift_held: bool,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    use crate::interaction::{classify_drag_membership_change, GroupMembershipChange};

    tracing::debug!(
        target: "blinc_node_editor::membership",
        node = dragged.as_str(),
        shift = shift_held,
        "drag-end membership check entered"
    );

    let Some((before, after)) = compute_drag_group_targets(editor, dragged, shift_held) else {
        return;
    };
    let after_inside_any = after.is_some();

    tracing::debug!(
        target: "blinc_node_editor::membership",
        node = dragged.as_str(),
        shift = shift_held,
        before = ?before.as_ref().map(|g| g.as_str()),
        after = ?after.as_ref().map(|g| g.as_str()),
        "classified"
    );

    if let Some(change) =
        classify_drag_membership_change(dragged.clone(), before.clone(), after.clone())
    {
        match change {
            GroupMembershipChange::Add(req) => {
                editor.push_event(EditorEvent::AddToGroupRequested(req));
            }
            GroupMembershipChange::RemoveOut(req) => {
                editor.push_event(EditorEvent::RemoveFromGroupRequested(req));
                // Cross-group case: classify() returns only the
                // RemoveOut leg; follow up with the Add so the host
                // can record both sides of the move atomically.
                if let (Some(_prev), Some(next)) = (before, after) {
                    if after_inside_any {
                        editor.push_event(EditorEvent::AddToGroupRequested(
                            crate::group::AddToGroupRequest {
                                group: next,
                                node: dragged.clone(),
                            },
                        ));
                    }
                }
            }
        }
    }
}

/// Multi-node sibling of [`detect_membership_changes`] for the case
/// where the user drags a Group container by its chrome instead of an
/// individual node. The per-node detect path is unhelpful here: every
/// dragged node moves together, so each one's "auto bounds excluding
/// me" still tracks the other dragged members and the dragged node
/// stays "inside" the parent forever.
///
/// Shift+drag of a group container means: any parent group whose
/// members overlap with the dragged group's members should compute
/// its auto-bounds excluding ALL the dragged-group members. The
/// remaining (non-dragged) members give the stationary baseline; any
/// dragged member whose new centre falls outside that baseline fires
/// a [`crate::group::RemoveFromGroupRequest`] against the parent.
///
/// Without Shift, the gesture is a no-op (same conservative policy as
/// single-node drag: don't accidentally tear members out of a group
/// on a casual nudge).
fn detect_group_drag_membership_changes<K, N, C, G>(
    editor: &NodeEditor<K, N, C, G>,
    dragged_group: &GroupId,
    shift_held: bool,
) where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    use crate::group::{RemoveFromGroupRequest, RemoveSource};

    if !shift_held {
        return;
    }

    let graph = editor.graph.read().unwrap();
    let Some(dg) = graph.groups.iter().find(|g| g.id == *dragged_group) else {
        return;
    };
    let dragged_members: Vec<NodeId> = dg.members.clone();
    if dragged_members.is_empty() {
        return;
    }

    // Any group (other than the dragged one) whose members include
    // any of `dragged_members` is a candidate parent.
    let parent_overlaps: Vec<(GroupId, Vec<NodeId>)> = graph
        .groups
        .iter()
        .filter(|g| g.id != *dragged_group)
        .filter_map(|g| {
            let overlap: Vec<NodeId> = dragged_members
                .iter()
                .filter(|m| g.members.contains(m))
                .cloned()
                .collect();
            if overlap.is_empty() {
                None
            } else {
                Some((g.id.clone(), overlap))
            }
        })
        .collect();
    if parent_overlaps.is_empty() {
        return;
    }

    let theme_overrides = editor.theme.read().unwrap();
    let theme = ThemeResolver::new(&theme_overrides);
    let pad = theme.group_padding();
    let header_h_fallback = 28.0;

    let mut to_remove: Vec<RemoveFromGroupRequest> = Vec::new();
    for (parent_id, overlap_members) in &parent_overlaps {
        let Some(parent) = graph.groups.iter().find(|g| g.id == *parent_id) else {
            continue;
        };

        // Region bounds for the parent: explicit override wins;
        // otherwise auto-bounds excluding the WHOLE dragged set. If
        // all of parent's members are in the dragged set, no
        // baseline remains and every overlap member escapes by
        // definition.
        let region_bounds: Option<Rect> = if let Some(explicit) = parent.bounds {
            Some(explicit)
        } else {
            compute_group_auto_bounds_excluding_set(parent, &graph.nodes, &dragged_members, |n| {
                editor.node_bounds_for(n, &theme)
            })
            .map(|auto_bounds| {
                Rect::new(
                    auto_bounds.x() - pad,
                    auto_bounds.y() - pad - header_h_fallback,
                    auto_bounds.width() + pad * 2.0,
                    auto_bounds.height() + pad * 2.0 + header_h_fallback,
                )
            })
        };

        for m in overlap_members {
            let Some(node) = graph.nodes.iter().find(|n| n.id == *m) else {
                continue;
            };
            let (w, h) = node.size.unwrap_or((180.0, 72.0));
            let cx = node.position.x + w * 0.5;
            let cy = node.position.y + h * 0.5;
            let escapes = match region_bounds {
                None => true,
                Some(b) => {
                    !(cx >= b.x()
                        && cx <= b.x() + b.width()
                        && cy >= b.y()
                        && cy <= b.y() + b.height())
                }
            };
            if escapes {
                to_remove.push(RemoveFromGroupRequest {
                    group: parent_id.clone(),
                    node: m.clone(),
                    source: RemoveSource::DraggedOut,
                });
            }
        }
    }
    drop(graph);

    for req in to_remove {
        editor.push_event(EditorEvent::RemoveFromGroupRequested(req));
    }
}

/// Dispatch a `KEY_DOWN` event against the current selection +
/// drag state.
///
/// * `Esc` — cancel an active drag-connect gesture (FSM → Idle).
/// * `Shift+Delete` / `Shift+Backspace` — push
///   `DeleteConnectionRequested` for each selected edge and
///   `DeleteNodesRequested` for the selected nodes. The Shift
///   modifier is required so plain Delete / Backspace remain
///   available for focused text inputs (search box, inline title
///   editor, port-edit popovers) without those widgets stealing
///   the destructive shortcut — and vice versa, so canvas-side
///   deletes don't fight whatever editable widget currently owns
///   focus. Hosts choose how to confirm (cn dialog, undo
///   bookkeeping, etc.).
///
/// `Cmd-Z` / `Cmd-Shift-Z` (`Ctrl` on non-macOS) — emit
/// [`EditorEvent::UndoRequested`] / [`EditorEvent::RedoRequested`].
/// Hosts wire these to a [`crate::history::History`] (or their own
/// undo stack); the editor itself doesn't keep state.
///
/// `Cmd-D` — emit [`EditorEvent::DuplicateNodesRequested`] carrying
/// the current node selection. `Shift+D` toggles soft-disabled on
/// the selection (the Shift modifier is required so plain `D` is
/// released back to focused text widgets — search box, inline title
/// editor, etc. — and doesn't get swallowed by the canvas-side
/// disable toggle when the user types `D` into a search field).
///
/// `Cmd-A` — emit [`EditorEvent::SelectAllRequested`]. Hosts call
/// [`crate::NodeEditor::select_all`] (or build a tailored selection).
impl<K, N, C, G> NodeEditor<K, N, C, G>
where
    K: PortKind,
    N: Send + Sync + 'static,
    C: Send + Sync + 'static,
    G: Send + Sync + 'static,
{
    /// Route a `KeyDown` event into the editor's keyboard-shortcut
    /// handler. The canvas closure wires this to `on_key_down`
    /// automatically; hosts that own their own canvas div (or want to
    /// rebind shortcuts) can call it directly. Returns `true` if the
    /// editor consumed the key, `false` otherwise so hosts can layer
    /// their own shortcuts on top.
    pub fn handle_key_down(
        &self,
        kc: blinc_core::events::KeyCode,
        mods: blinc_core::events::Modifiers,
    ) -> bool {
        if kc == blinc_core::events::KeyCode::ESCAPE {
            // Reset the drag-to-connect FSM if a gesture is in flight.
            // No-op otherwise so Esc stays available for higher-level
            // host shortcuts (closing dialogs, deselecting, etc.).
            let mut drag = self.drag_state();
            if drag.is_active() {
                drag.cancel();
                self.drag_state.set(drag);
                blinc_layout::request_redraw();
                return true;
            }
            return false;
        }
        let editor = self;
        if (kc == blinc_core::events::KeyCode::DELETE
            || kc == blinc_core::events::KeyCode::BACKSPACE)
            && mods.shift()
        {
            // Snapshot selection ids by category — canvas-kit stores the
            // prefixed region ids, so we split connection ids
            // (`edge:{u64}`), node ids (`node:{str}`), and group ids
            // (`group:{str}`) here. Each category fires its own event so
            // the host can confirm independently (a destructive dialog
            // per type, or skip entirely for groups via
            // `EditorCommand::RemoveGroup`).
            let sel = editor.kit.selection().selected;
            let mut connection_ids: Vec<ConnectionId> = Vec::new();
            let mut node_ids: Vec<NodeId> = Vec::new();
            let mut group_ids: Vec<GroupId> = Vec::new();
            for region in &sel {
                match RegionId::parse(region) {
                    Some(RegionId::Edge(id)) => connection_ids.push(id),
                    Some(RegionId::Node(id)) => node_ids.push(id),
                    Some(RegionId::Group(id)) => group_ids.push(id),
                    _ => {}
                }
            }
            for id in connection_ids {
                editor.push_event(EditorEvent::DeleteConnectionRequested(id));
            }
            for id in group_ids {
                editor.push_event(EditorEvent::DeleteGroupRequested(
                    crate::group::DeleteGroupRequest { group: id },
                ));
            }
            if !node_ids.is_empty() {
                editor.push_event(EditorEvent::DeleteNodesRequested(node_ids));
            }
            return true;
        }
        // Cmd-/Ctrl- combinations route through host-defined events.
        // Tested BEFORE the plain-D arm so Cmd-D doesn't fall through
        // into the disable-toggle.
        if mods.command() {
            if kc == blinc_core::events::KeyCode::A {
                editor.push_event(EditorEvent::SelectAllRequested);
                return true;
            }
            if kc == blinc_core::events::KeyCode::D {
                let sel = editor.kit.selection().selected;
                let node_ids: Vec<NodeId> = sel
                    .iter()
                    .filter_map(|r| match RegionId::parse(r)? {
                        RegionId::Node(id) => Some(id),
                        _ => None,
                    })
                    .collect();
                if !node_ids.is_empty() {
                    editor.push_event(EditorEvent::DuplicateNodesRequested(node_ids));
                }
                return true;
            }
            if kc == blinc_core::events::KeyCode::Z {
                if mods.shift() {
                    editor.push_event(EditorEvent::RedoRequested);
                } else {
                    editor.push_event(EditorEvent::UndoRequested);
                }
                return true;
            }
        }
        if kc == blinc_core::events::KeyCode::D && !mods.command() && mods.shift() {
            // Toggle soft-disabled on every selected node + every
            // selected group. The Shift modifier is required so plain
            // `D` is released back to focused text widgets (search box,
            // inline title editor, port-edit popovers) — otherwise the
            // canvas-side disable toggle hijacks the keystroke before
            // the input can receive it, and the user sees a missing
            // character in the search box every time their query
            // contains `D`. Same policy as Shift+Delete for destructive
            // shortcuts (see the BACKSPACE / DELETE arm above).
            //
            // Mixed-state selections (some disabled, some not) all flip
            // to disabled on the first press, then all enable on the
            // next — the new state derives from the FIRST selected
            // entity's current value (node first, then group) so the
            // action stays deterministic regardless of selection order.
            let sel = editor.kit.selection().selected;
            let node_ids: Vec<NodeId> = sel
                .iter()
                .filter_map(|r| match RegionId::parse(r)? {
                    RegionId::Node(id) => Some(id),
                    _ => None,
                })
                .collect();
            let group_ids: Vec<GroupId> = sel
                .iter()
                .filter_map(|r| match RegionId::parse(r)? {
                    RegionId::Group(id) => Some(id),
                    _ => None,
                })
                .collect();
            if node_ids.is_empty() && group_ids.is_empty() {
                return false;
            }
            let target_disabled = {
                let g = editor.graph.read().unwrap();
                let from_node = node_ids
                    .first()
                    .and_then(|id| g.nodes.iter().find(|n| n.id == *id).map(|n| n.disabled));
                let from_group = group_ids.first().and_then(|id| {
                    g.groups
                        .iter()
                        .find(|gg| gg.id == *id)
                        .map(|gg| gg.disabled)
                });
                !from_node.or(from_group).unwrap_or(false)
            };
            for id in &node_ids {
                editor.set_node_disabled(id, target_disabled);
            }
            for id in &group_ids {
                editor.set_group_disabled(id, target_disabled);
            }
            return true;
        }
        false
    }
}

/// Classify a hit-tested region id into a [`ContextMenuTarget`].
/// `None` / non-matching prefix → [`ContextMenuTarget::Canvas`] so a
/// right-click on empty space, on a non-modelled region (`port:`
/// chrome buttons, `edge_delete:`, `group_title:` for double-click),
/// or on an unknown id falls through to the canvas / background menu
/// — host can still surface select-all / zoom-to-fit / undo from
/// there.
fn resolve_context_menu_target(region_id: Option<&str>) -> crate::event::ContextMenuTarget {
    use crate::event::ContextMenuTarget;
    let Some(region) = region_id else {
        return ContextMenuTarget::Canvas;
    };
    match RegionId::parse(region) {
        Some(RegionId::Node(id)) | Some(RegionId::NodeBadge(id)) => ContextMenuTarget::Node(id),
        Some(RegionId::Edge(id)) | Some(RegionId::EdgeDelete(id)) => ContextMenuTarget::Edge(id),
        // Group sub-regions (group_title / group_desc / group_edit /
        // group_delete / group_collapse / group_badge) all resolve
        // to the parent group's context menu — the user right-
        // clicked group chrome and expects group-level actions.
        Some(other) => match other.as_group() {
            Some(gid) => ContextMenuTarget::Group(gid.clone()),
            None => ContextMenuTarget::Canvas,
        },
        None => ContextMenuTarget::Canvas,
    }
}

/// Re-derive the canvas-kit region id from a [`ContextMenuTarget`].
/// Used to update the selection when the target isn't already
/// selected. `None` for `Canvas` — no region to add.
fn target_region_id(target: &crate::event::ContextMenuTarget) -> Option<String> {
    use crate::event::ContextMenuTarget;
    match target {
        ContextMenuTarget::Node(id) => Some(RegionId::Node(id.clone()).encode()),
        ContextMenuTarget::Edge(id) => Some(RegionId::Edge(*id).encode()),
        ContextMenuTarget::Group(id) => Some(RegionId::Group(id.clone()).encode()),
        ContextMenuTarget::Canvas => None,
    }
}

/// Translate a canvas-kit region id back into a [`HoverTarget`].
/// Returns `None` for ids that don't map to graph entities (e.g.
/// the `edge_delete:` button — we don't want a delete-button hover
/// to count as an edge hover that lights up the curve).
fn parse_hover_target(region_id: &str) -> Option<HoverTarget> {
    match RegionId::parse(region_id)? {
        RegionId::Node(id) => Some(HoverTarget::Node(id)),
        RegionId::NodeBadge(id) => Some(HoverTarget::NodeBadge(id)),
        RegionId::Group(id) => Some(HoverTarget::Group(id)),
        RegionId::GroupBadge(id) => Some(HoverTarget::GroupBadge(id)),
        RegionId::Edge(id) => Some(HoverTarget::Edge(id)),
        RegionId::Port(addr) => Some(HoverTarget::Port(addr)),
        // EdgeDelete + group chrome chips intentionally don't hover-
        // light the underlying entity — they're affordance clicks,
        // not entity hovers.
        _ => None,
    }
}

/// Seconds since the first call. Used as the animation clock for
/// state-driven edge motion (Running shimmer + Pending pulse).
/// `web_time::Instant` works on every target —
/// `std::time::Instant::now` panics on wasm32-unknown-unknown.
fn animation_time_secs() -> f32 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<web_time::Instant> = OnceLock::new();
    let start = EPOCH.get_or_init(web_time::Instant::now);
    start.elapsed().as_secs_f32()
}

/// Which edge of a node a port hangs off. Drives the sliding
/// port layout: ports get grouped per (node, side) so a single
/// connection-facing port lands at the side midpoint while
/// multiple share an even distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PortSide {
    Left,
    Right,
    Top,
    Bottom,
}

/// Project a point onto the closest point on `rect`'s perimeter.
/// Distinct from [`closest_point_on_rect`] which snaps to the
/// LEFT or RIGHT side-centre (used for collapsed-group stubs).
/// Currently dead-code: the sliding-port pass now groups ports
/// by side and evenly distributes them along the chosen edge,
/// which gave better visual results than free-form projection
/// (single ports landed in corners; multi-port nodes crossed
/// each other). Kept around for callers that want raw perimeter
/// projection — e.g. a future "drag-from-edge" interaction.
#[allow(dead_code)]
fn project_to_rect_perimeter(rect: Rect, p: Point) -> Point {
    let left = rect.x();
    let right = rect.x() + rect.width();
    let top = rect.y();
    let bottom = rect.y() + rect.height();
    let inside = p.x > left && p.x < right && p.y > top && p.y < bottom;
    if inside {
        // Find the nearest edge by clamped distance.
        let d_left = p.x - left;
        let d_right = right - p.x;
        let d_top = p.y - top;
        let d_bottom = bottom - p.y;
        let min = d_left.min(d_right).min(d_top).min(d_bottom);
        if min == d_left {
            Point::new(left, p.y)
        } else if min == d_right {
            Point::new(right, p.y)
        } else if min == d_top {
            Point::new(p.x, top)
        } else {
            Point::new(p.x, bottom)
        }
    } else {
        // p is outside — clamp x/y into the rect, the result
        // lies on the perimeter naturally.
        Point::new(p.x.clamp(left, right), p.y.clamp(top, bottom))
    }
}

/// Centroid of a list of points. Returns the origin for an empty
/// list (caller must guard). Kept after the port-sort fix dropped
/// its only call site because the same helper is the natural shape
/// for any future "average counterpart position" feature (e.g.
/// curved-edge anchor smoothing).
#[allow(dead_code)]
fn average_point(points: &[Point]) -> Point {
    if points.is_empty() {
        return Point::new(0.0, 0.0);
    }
    let n = points.len() as f32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for p in points {
        sum_x += p.x;
        sum_y += p.y;
    }
    Point::new(sum_x / n, sum_y / n)
}

/// Pick the side-centre of `rect` (left-centre or right-centre)
/// that faces `p`. Used as a "stub" endpoint for edges incident
/// to a collapsed group: each connection terminates at the
/// vertical centre of the chip's left or right edge, mirroring
/// Zeal's collapsed-group routing. Returns the LEFT centre when
/// `p` lies to the left of the rect's horizontal centre, the
/// RIGHT centre otherwise — so external sources on the right
/// always connect to the right side and sinks on the left to the
/// left side. The cubic-bezier mid-x control points the renderer
/// uses then sweep cleanly outward from the chip.
fn closest_point_on_rect(rect: Rect, p: Point) -> Point {
    let centre_x = rect.x() + rect.width() * 0.5;
    let centre_y = rect.y() + rect.height() * 0.5;
    let x = if p.x < centre_x {
        rect.x()
    } else {
        rect.x() + rect.width()
    };
    Point::new(x, centre_y)
}

/// Union of two AABB rects (smallest enclosing rect).
fn union_rect(a: Rect, b: Rect) -> Rect {
    let min_x = a.x().min(b.x());
    let min_y = a.y().min(b.y());
    let max_x = (a.x() + a.width()).max(b.x() + b.width());
    let max_y = (a.y() + a.height()).max(b.y() + b.height());
    Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
}

/// Theme-pulled workspace surface — the solid fill the canvas dots
/// draw on top of. Pulled from [`ColorToken::Background`] so the
/// editor inherits the host's page-surface tint automatically.
fn workspace_background() -> blinc_core::layer::Color {
    use blinc_theme::tokens::ColorToken;
    use blinc_theme::ThemeState;
    ThemeState::try_get()
        .map(|s| s.color(ColorToken::Background))
        .unwrap_or_else(|| blinc_core::layer::Color::rgb(0.04, 0.05, 0.07))
}

/// Default canvas background used when the host doesn't override
/// via `with_background(...)`. A zoom-adaptive dot pattern.
///
/// We derive the dot tint by *luminance-shifting* the workspace
/// background — for a dark workspace we lighten (additive), for a
/// light workspace we darken (subtractive). This gives a reliable
/// contrast ratio regardless of bundle, where pulling a fixed
/// token like `Border` produced near-invisible dots on bundles
/// whose border was close to the workspace luminance (Universal
/// Hybrid dark + every light scheme).
fn default_canvas_background() -> CanvasBackground {
    use blinc_theme::tokens::ColorToken;
    use blinc_theme::ThemeState;

    let workspace = ThemeState::try_get()
        .map(|s| s.color(ColorToken::Background))
        .unwrap_or_else(|| blinc_core::layer::Color::rgb(0.05, 0.06, 0.08));

    // Luminance-aware shift: dark workspace gets a lighter dot;
    // light workspace gets a darker dot. Magnitudes are large
    // enough that dots remain perceptible at 1.0 zoom (and don't
    // need a microscope to find on light bundles whose `Background`
    // luminance is near-white).
    let luminance = 0.299 * workspace.r + 0.587 * workspace.g + 0.114 * workspace.b;
    let shift: f32 = if luminance < 0.5 { 0.40 } else { -0.45 };
    let tint = blinc_core::layer::Color::rgba(
        (workspace.r + shift).clamp(0.0, 1.0),
        (workspace.g + shift).clamp(0.0, 1.0),
        (workspace.b + shift).clamp(0.0, 1.0),
        1.0,
    );

    // Dot size + spacing — tuned so the pattern reads as
    // workspace texture at 1.0 zoom without the dots becoming
    // visual noise. 3.5 px @ 28 px spacing = ~1.4 % coverage,
    // similar to Figma's canvas dot density. Zoom-adaptive kicks
    // in below 0.3× to thin the pattern at low zoom.
    CanvasBackground::dots(tint)
        .with_size(3.5)
        .with_spacing(28.0)
        .with_zoom_adaptive(0.3, 5)
}

/// Per-port hover-state lookup against the current drag state.
fn drag_port_hover_state<K: PortKind, M>(
    drag: &DragConnect,
    node: &NodeInstance<M>,
    desc: &crate::port::PortDesc<K>,
) -> PortHoverState {
    match drag {
        DragConnect::Idle => PortHoverState::None,
        DragConnect::Dragging { .. } => PortHoverState::None,
        DragConnect::Hovering {
            candidate,
            validation,
            ..
        } => {
            if candidate.node == node.id && candidate.port == desc.id {
                if validation.is_accept() {
                    PortHoverState::Compatible
                } else {
                    PortHoverState::Incompatible
                }
            } else {
                PortHoverState::None
            }
        }
    }
}

/// Pure helper for `align_nodes` — takes a snapshot of
/// `(NodeId, Rect)` pairs and returns the new top-left position for
/// every node so they all share `edge`. Extracted so it's testable
/// without constructing a full `NodeEditor`.
fn compute_align(snapshot: &[(NodeId, Rect)], edge: AlignEdge) -> Vec<(NodeId, Point)> {
    if snapshot.len() < 2 {
        return Vec::new();
    }
    let target = match edge {
        AlignEdge::Left => snapshot
            .iter()
            .map(|(_, b)| b.x())
            .fold(f32::INFINITY, f32::min),
        AlignEdge::Right => snapshot
            .iter()
            .map(|(_, b)| b.x() + b.width())
            .fold(f32::NEG_INFINITY, f32::max),
        AlignEdge::CenterX => {
            let sum: f32 = snapshot.iter().map(|(_, b)| b.x() + b.width() * 0.5).sum();
            sum / snapshot.len() as f32
        }
        AlignEdge::Top => snapshot
            .iter()
            .map(|(_, b)| b.y())
            .fold(f32::INFINITY, f32::min),
        AlignEdge::Bottom => snapshot
            .iter()
            .map(|(_, b)| b.y() + b.height())
            .fold(f32::NEG_INFINITY, f32::max),
        AlignEdge::CenterY => {
            let sum: f32 = snapshot.iter().map(|(_, b)| b.y() + b.height() * 0.5).sum();
            sum / snapshot.len() as f32
        }
    };
    snapshot
        .iter()
        .map(|(id, b)| {
            let p = match edge {
                AlignEdge::Left => Point::new(target, b.y()),
                AlignEdge::Right => Point::new(target - b.width(), b.y()),
                AlignEdge::CenterX => Point::new(target - b.width() * 0.5, b.y()),
                AlignEdge::Top => Point::new(b.x(), target),
                AlignEdge::Bottom => Point::new(b.x(), target - b.height()),
                AlignEdge::CenterY => Point::new(b.x(), target - b.height() * 0.5),
            };
            (id.clone(), p)
        })
        .collect()
}

/// Pure helper for `distribute_nodes` — see [`compute_align`].
fn compute_distribute(snapshot: &[(NodeId, Rect)], axis: DistributeAxis) -> Vec<(NodeId, Point)> {
    if snapshot.len() < 3 {
        return Vec::new();
    }
    let mut sorted: Vec<&(NodeId, Rect)> = snapshot.iter().collect();
    sorted.sort_by(|a, b| {
        let key = |r: &Rect| match axis {
            DistributeAxis::Horizontal => r.x() + r.width() * 0.5,
            DistributeAxis::Vertical => r.y() + r.height() * 0.5,
        };
        key(&a.1)
            .partial_cmp(&key(&b.1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let centre = |r: &Rect| match axis {
        DistributeAxis::Horizontal => r.x() + r.width() * 0.5,
        DistributeAxis::Vertical => r.y() + r.height() * 0.5,
    };
    let first = centre(&sorted.first().unwrap().1);
    let last = centre(&sorted.last().unwrap().1);
    let n = sorted.len();
    let step = (last - first) / (n - 1) as f32;
    let mut out = Vec::with_capacity(n - 2);
    for (i, (id, b)) in sorted.iter().enumerate().skip(1).take(n - 2) {
        let target_centre = first + step * i as f32;
        let p = match axis {
            DistributeAxis::Horizontal => Point::new(target_centre - b.width() * 0.5, b.y()),
            DistributeAxis::Vertical => Point::new(b.x(), target_centre - b.height() * 0.5),
        };
        out.push((id.clone(), p));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::ContextMenuTarget;

    fn n(id: &str, x: f32, y: f32, w: f32, h: f32) -> (NodeId, Rect) {
        (NodeId::from(id), Rect::new(x, y, w, h))
    }

    // ── resolve_context_menu_target — exhaustive mapping ──
    //
    // Pinned so the seven-variant `RegionId::as_group()` collapse
    // can't silently shift behaviour if a future RegionId variant
    // is added to that accessor. Right-clicking any group sub-
    // region (chrome chips / title / desc / badge) surfaces the
    // PARENT group's context menu, not a chrome-specific one;
    // right-clicking the edge-delete chip surfaces the parent
    // edge's context menu.

    #[test]
    fn context_menu_target_for_node_region() {
        let r = RegionId::Node(NodeId::from("n1")).encode();
        assert_eq!(
            resolve_context_menu_target(Some(&r)),
            ContextMenuTarget::Node(NodeId::from("n1"))
        );
    }

    #[test]
    fn context_menu_target_collapses_node_badge_to_node() {
        let r = RegionId::NodeBadge(NodeId::from("n1")).encode();
        assert_eq!(
            resolve_context_menu_target(Some(&r)),
            ContextMenuTarget::Node(NodeId::from("n1"))
        );
    }

    #[test]
    fn context_menu_target_collapses_every_group_subregion_to_group() {
        let gid = GroupId::from("g0");
        for r in [
            RegionId::Group(gid.clone()),
            RegionId::GroupBadge(gid.clone()),
            RegionId::GroupCollapse(gid.clone()),
            RegionId::GroupDelete(gid.clone()),
            RegionId::GroupDesc(gid.clone()),
            RegionId::GroupEdit(gid.clone()),
            RegionId::GroupTitle(gid.clone()),
        ] {
            let encoded = r.encode();
            assert_eq!(
                resolve_context_menu_target(Some(&encoded)),
                ContextMenuTarget::Group(gid.clone()),
                "{encoded} must resolve to Group(g0)"
            );
        }
    }

    #[test]
    fn context_menu_target_for_edge_region() {
        let r = RegionId::Edge(ConnectionId(7)).encode();
        assert_eq!(
            resolve_context_menu_target(Some(&r)),
            ContextMenuTarget::Edge(ConnectionId(7))
        );
    }

    #[test]
    fn context_menu_target_collapses_edge_delete_to_edge() {
        let r = RegionId::EdgeDelete(ConnectionId(7)).encode();
        assert_eq!(
            resolve_context_menu_target(Some(&r)),
            ContextMenuTarget::Edge(ConnectionId(7))
        );
    }

    #[test]
    fn context_menu_target_for_port_falls_through_to_canvas() {
        // Ports don't have a node-level context menu; clicking a
        // port falls through to the Canvas surface so the host
        // can still offer "select all / zoom to fit / undo".
        let r = RegionId::Port(crate::port::PortAddress {
            node: NodeId::from("n1"),
            port: crate::port::PortId::from("p0"),
        })
        .encode();
        assert_eq!(
            resolve_context_menu_target(Some(&r)),
            ContextMenuTarget::Canvas
        );
    }

    #[test]
    fn context_menu_target_for_unknown_or_none_returns_canvas() {
        assert_eq!(
            resolve_context_menu_target(Some("unknown:foo")),
            ContextMenuTarget::Canvas
        );
        assert_eq!(resolve_context_menu_target(None), ContextMenuTarget::Canvas);
    }

    #[test]
    fn align_left_pulls_all_to_leftmost_x() {
        let snap = vec![
            n("a", 10.0, 0.0, 50.0, 50.0),
            n("b", 30.0, 100.0, 80.0, 40.0),
        ];
        let out = compute_align(&snap, AlignEdge::Left);
        assert_eq!(out[0].1.x, 10.0);
        assert_eq!(out[1].1.x, 10.0);
    }

    #[test]
    fn align_right_pushes_all_to_rightmost_edge() {
        let snap = vec![
            n("a", 10.0, 0.0, 50.0, 50.0),
            n("b", 30.0, 100.0, 80.0, 40.0),
        ];
        let out = compute_align(&snap, AlignEdge::Right);
        // Right-most edge in the snapshot: 30 + 80 = 110.
        // `a` width 50 → x = 60; `b` width 80 → x = 30 (unchanged).
        assert_eq!(out[0].1.x, 60.0);
        assert_eq!(out[1].1.x, 30.0);
    }

    #[test]
    fn align_top_pulls_all_to_topmost_y() {
        let snap = vec![n("a", 0.0, 50.0, 40.0, 40.0), n("b", 0.0, 10.0, 40.0, 40.0)];
        let out = compute_align(&snap, AlignEdge::Top);
        assert_eq!(out[0].1.y, 10.0);
        assert_eq!(out[1].1.y, 10.0);
    }

    #[test]
    fn align_center_y_picks_average_centre() {
        // Centres: 50 + 20 = 70 → avg 35. b height 40 → y = 35 - 20 = 15.
        let snap = vec![n("a", 0.0, 30.0, 40.0, 40.0), n("b", 0.0, 0.0, 40.0, 40.0)];
        let out = compute_align(&snap, AlignEdge::CenterY);
        assert_eq!(out[0].1.y, 15.0);
        assert_eq!(out[1].1.y, 15.0);
    }

    #[test]
    fn align_noop_for_single_node() {
        let snap = vec![n("a", 10.0, 10.0, 50.0, 50.0)];
        assert!(compute_align(&snap, AlignEdge::Left).is_empty());
    }

    #[test]
    fn distribute_horizontal_spaces_interior_evenly() {
        // Anchors at x_centre 25 and 175; middle at (25+175)/2 = 100.
        let snap = vec![
            n("a", 0.0, 0.0, 50.0, 50.0),   // centre 25
            n("b", 75.0, 0.0, 30.0, 50.0),  // centre 90  — should move to 100
            n("c", 150.0, 0.0, 50.0, 50.0), // centre 175
        ];
        let out = compute_distribute(&snap, DistributeAxis::Horizontal);
        assert_eq!(out.len(), 1, "only interior nodes move");
        // b width 30 → new x = 100 - 15 = 85.
        assert_eq!(out[0].0, NodeId::from("b"));
        assert!((out[0].1.x - 85.0).abs() < 0.01, "got {}", out[0].1.x);
    }

    #[test]
    fn distribute_noop_below_three() {
        let snap = vec![n("a", 0.0, 0.0, 50.0, 50.0), n("b", 100.0, 0.0, 50.0, 50.0)];
        assert!(compute_distribute(&snap, DistributeAxis::Horizontal).is_empty());
    }
}
