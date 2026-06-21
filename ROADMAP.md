# `blinc_node_editor` Roadmap

Gap analysis against [Zeal](https://github.com/offbit-ai/zeal) plus interaction polish + runtime observability items that any production node editor needs.

Items grouped by priority. Tier 1 changes the data model; tier 2 adds inspector ergonomics; tiers 3+ are pure interaction/widget additions on top of what's already there.

---

## Known upstream blockers

These aren't `blinc_node_editor` issues, but they constrain what hosts can do on top of the editor. Tracked here so we don't keep hitting them.

### U1. Canvas content paints over absolutely-positioned overlay widgets

**Status**: FIXED upstream — hosts mark above-canvas overlays with `.layer(RenderLayer::Foreground)`.

**Symptom (pre-fix)**: An overlay widget (`viewport_hud`, `reset_button`, an editor zoom HUD, a host's minimap, etc.) positioned with `.absolute()` over a `kit.element()` canvas appears to be UNDER the canvas's drawn content — node panels drawn by the canvas closure cover the overlay. Visible in `canvas_kit_demo` (the "Zoom: 56% / Pan: (0,0)" HUD and the "Reset" button get clipped by the demo's draggable node panels).

**Root cause**: The slow paint walker bakes foreground primitives into the static cache alongside bg + glass. On compositor frames, the cache blit included the fg pixels, then the canvas overlay dispatched with `LoadOp::Load` painted on top of them. Net order seen on screen: bg → glass → fg (in cache) → canvas overlay (covers fg).

**Fix**: `BlincApp::render_foreground_overlay` re-dispatches the cached foreground primitives AFTER the canvas overlay dispatch at each `composite_frame` callsite. Restores the intended order: cache (bg + glass + fg) → canvas overlay → fg overlay. Hosts opt in by marking their above-canvas widget with `.layer(RenderLayer::Foreground)`.

**aux_data caveat**: The fg overlay dispatch reuses whatever aux_data the previous pass left in the storage buffer. Foreground primitives that reference aux entries (3D groups, polygon clip-paths emitted from a fg-layer element) may read stale data. Typical HUD / minimap / toolbar overlays are simple SDF + text and don't reference aux. Revisit if a host hits this.

**Where**: `crates/blinc_app/src/context.rs::render_foreground_overlay` + 3 `composite_frame` callsites.

**Related memory entry**: [gotcha_canvas_vs_static_z_ordering](../../.claude/projects/-Users-amaterasu-Vibranium-Blinc/memory/gotcha_canvas_vs_static_z_ordering.md) — same root cause manifesting for canvas-vs-stack-layered-siblings.

---

## Tier 1 — Foundational data-model additions

### 1.1 Port `description` field ✓
Zeal's `Port.description` is a semantic doc string — used both for tooltips and for LLM-driven graph construction.

**Shipped:** `PortDesc<K>.description: Option<String>` + `.with_description(...)` builder. Surfaced by the port tooltip pipeline (`draw_port_tooltip_clamped`) — title + wrapped description chip anchored outside the port, viewport-clamped, inverse-zoom-scaled so it stays a constant on-screen size regardless of canvas zoom. Width measured via `blinc_layout::measure_text` (real font metrics, not heuristic) so multi-line wrap lands inside the chip padding.

### 1.2 Connection state animation ✓
`ConnectionState::{Running, Pending}` now drive per-frame edge motion.

**Shipped:** `draw_edge_with_state` modulates colour per segment when `state == Running` (bright "comet" dot travels along the curve at 0.5 cycles/sec via `flow_tint`) and pulses opacity for `Pending` (sin-wave between 0.35 and 0.80 at 0.7 Hz). Clock uses `web_time::Instant` for cross-target safety (`std::time::Instant` panics on wasm). `any_animated_edge` triggers `request_animation_tick` + `request_redraw` each frame a Running / Pending edge is on-canvas, so the chain self-sustains; quiesces the moment the source stops re-asking. CPU cost capped via `WindowConfig::animation_fps_cap` for hosts that don't need 60 fps for background runtime motion.

### 1.3 Subgraph navigation fields on `NodeInstance`
A node that descends into a subgraph carries a `(graph_id, namespace, workflow_id)` triple. We have `Subgraph` / `ExposedPort` / `NavigationCrumb`, but no first-class fields on `NodeInstance` that say "this node IS a subgraph reference."

**Change:** Optional `subgraph: Option<SubgraphRef>` field on `NodeInstance<M>`, where `SubgraphRef { graph_id, namespace, workflow_id, display_name }`. Host's `on_node_click` (or double-click) can read this and push a `NavigationCrumb`.

---

## Tier 2 — Inspector / config schema

### 2.1 Typed `PropertyDefinition` instead of opaque JSON
Currently `NodeTemplate.config_schema: serde_json::Value`. Zeal defines a rich `PropertyDefinition` enum (text / number / select / boolean / textarea / code-editor / file / rules / dataOperations) with per-type validation (`required`, `defaultValue`, `options`, `language`, `lineNumbers`, etc.).

**Change:** Introduce `PropertyDefinition` enum + `ConfigSchema = Vec<PropertyDefinition>` in a new `config.rs`. `NodeTemplate::config_schema` becomes `ConfigSchema` (opaque-JSON still supported via a `Custom(Value)` variant).

`inspector.rs` (currently a stub) implements form generation per variant. Hosts subscribe to `InspectorPatchRequest`.

### 2.2 Conditional property rules
Zeal's `propertyRules.triggers` + `rules.when/updates` let the schema declare reactive form behaviour (changing port count when a "mode" select changes, etc.).

**Change:** `PropertyRule { triggers: Vec<String>, when: Predicate, updates: HashMap<String, Value> }`. Inspector evaluates against current `NodeInstance.config` and emits patches.

### 2.3 `requiredEnvVars` on templates
Useful for runtime sandboxes (nan8 + dynASB) — declares which env vars the actor needs.

**Change:** `NodeTemplate.required_env_vars: Vec<String>`. Inspector surfaces unset vars as warnings.

---

## Tier 3 — Subgraph + group polish

### 3.1 Group `created_at` / `updated_at`
Useful for audit panels + "recently modified" sorting.

**Change:** Optional timestamps on `Group<G>`. Host sets them; editor never mutates.

### 3.2 Relative member positions inside collapsed group
Zeal's `nodePositions: Record<NodeId, {x,y}>` stores member positions relative to the group, so an unmodified group preserves its layout after a collapse → restore cycle even if member positions changed in between.

**Change:** Optional `member_layout: HashMap<NodeId, Point>` on `Group<G>`, captured on collapse, restored on expand. Editor manages snapshot/restore behind the existing `ToggleCollapseRequest`.

### 3.3 Group header chrome buttons (collapse / delete / edit) ✓
We render the badge in the group header AND now three affordance chips: edit (pencil) / delete (×) / collapse (chevron).

**Shipped:** `GroupSlots` gained `chrome_edit` / `chrome_delete` / `chrome_collapse` flex slots at the right edge of the header. `draw_group_header_chrome` paints each via embedded Tabler outline SVG (cached by `(glyph_kind, quantised stroke color)` in a `OnceLock<Mutex<HashMap>>`) so strokes get rounded caps + joins that close cleanly at the apex. Three hit regions registered AFTER the group body (`group_edit:{id}` / `group_delete:{id}` / `group_collapse:{id}`). Click → `EditorEvent::EditGroupRequested` (combined title + description, opened in a `blinc_cn::dialog` modal with both fields and one history entry covering both) / `DeleteGroupRequested` / `ToggleCollapseRequested`.

**Known issue:** when a group's description wraps to 2+ visual lines the header band grows to fit, but the chrome chips stay anchored to the title baseline — they end up off-centre vertically relative to the full header. See [gotcha_group_chrome_header_height](../../.claude/projects/-Users-amaterasu-Vibranium-Blinc/memory/gotcha_group_chrome_header_height.md).

---

## Tier 4 — Interaction polish

### 4.1 Drag-into-group ✓
Drop a node into a group's footprint → group adds it as a member.

**Shipped:** Drag-end handler classifies via `classify_drag_membership_change` against the pre / post group membership snapshot, fires `EditorEvent::AddToGroupRequested`. Live drag-into preview tints the target group's border before release via `update_drag_group_preview` + `GroupBorderKind::AddTarget` so the user sees the would-be membership change as the gesture is in flight.

### 4.2 Drag-out-of-group ✓
Reverse — node dragged outside its parent group's bounds emits `RemoveFromGroupRequested`.

**Shipped:** Same drag-end classification fires `RemoveFromGroupRequested` with `RemoveSource::DraggedOut`. Live preview shows `GroupBorderKind::RemoveTarget` tint while the gesture is in flight. Shift modifier on drag-end produces "escape" semantics (remove without re-adding) when crossing between groups.

### 4.3 Marquee multi-select ✓
Box-select via canvas-kit's marquee tool, with selection-coherence handling so dragging an outside node doesn't drag still-selected group members.

**Shipped:** `on_element_drag` gates per-frame deltas on `active == evt.region_id` so only the clicked-on element drags; selection survives. `EditorEvent::MultiSelectionSettled { node_ids, anchor_screen }` fires when the marquee finalizes with 2+ nodes so hosts can pop a floating toolbar (Group / Align / Distribute / Delete). `SelectionCleared` fires when the marquee narrows to ≤1 node so any open toolbar can dismiss.

### 4.4 Keyboard shortcuts ✓
Built into the canvas div on focus.

**Shipped:** `Esc` cancels active drag-connect. `Delete` / `Backspace` splits selection by region prefix and emits `DeleteConnectionRequested` / `DeleteGroupRequested` / `DeleteNodesRequested`. Plain `D` toggles soft-disable on selection. `Cmd-A` → `SelectAllRequested`, `Cmd-D` → `DuplicateNodesRequested(node_ids)`, `Cmd-Z` → `UndoRequested`, `Cmd-Shift-Z` → `RedoRequested`. `NodeEditor::handle_key_down(kc, mods) -> bool` is a public method so hosts can intercept or layer. Plus a full **History module** (command-log + inverse, drag coalescing via `CoalesceKey::DragNode`, host-defined transactions via `push_transaction`) — see `History`, `HistoryEntry`, `TransactionFn` in `history.rs`. Demo records every user-driven mutation incl. group creation / rename via cn::dialog.

### 4.5 Context menu ✓ (known issue: invisible until mouse motion)
Right-click any node / edge / group / blank canvas → contextual menu with the relevant action set.

**Known issue:** first right-click opens the menu but its surface is invisible — only after mouse-motion (any cursor delta) does the bg + border + items paint. Same family as the cn::popover-with-cn::input bug (which was resolved by the `hash_composite_scratch must hash layer_commands` fix). Two open paths could each produce it: the [class-animation timing race](../../.claude/projects/-Users-amaterasu-Vibranium-Blinc/memory/gotcha_first_interaction_skips_class_animation.md) on first open, or a composite-cache lock at the opacity-0 sample on context-menu's specific render layer. Tracked in [gotcha_context_menu_invisible_until_mouse_motion](../../.claude/projects/-Users-amaterasu-Vibranium-Blinc/memory/gotcha_context_menu_invisible_until_mouse_motion.md).

**Shipped:**
- New event `EditorEvent::ContextMenuRequested { target: ContextMenuTarget, anchor_screen: Point }`. `ContextMenuTarget = Node(NodeId) | Edge(ConnectionId) | Group(GroupId) | Canvas`. The editor hit-tests on right-click (via `Div::on_right_click`, which filters POINTER_DOWN on `mouse_button == 2`), classifies the region, replaces the selection with the right-clicked target when it isn't already selected (standard editor convention), and emits the event. Hosts decide everything else — entries, callbacks, presentation — keeping `blinc_node_editor` cn-free.
- `NodeEditor::push_event(...)` made public so host context-menu callbacks can re-enter the editor's event loop alongside the keyboard path.
- Group sub-regions (`group_title:` / `group_desc:` / `group_edit:` / `group_delete:` / `group_collapse:`) all resolve to the parent `Group(id)` target — user expects group-level actions when right-clicking group chrome.
- Demo wires `blinc_cn::context_menu()` per target:
  - **Node**: Duplicate (⌘D / Ctrl+D), Focus, Toggle Disable, Delete (Delete).
  - **Edge**: Delete Connection (Delete).
  - **Group**: Edit…, Toggle Collapse, Zoom to Group, Toggle Disable, Delete Group.
  - **Canvas**: Select All (⌘A), Zoom to Fit, Undo (⌘Z), Redo (⌘⇧Z).

### 4.6 Inline node title rename ✓
Double-click title → in-place editor backed by `cn::input` / `cn::textarea` popover; commit re-syncs through the editor's command channel and pushes a history entry.

**Shipped:** Same flow as the group title / description rename — anchored popover with deferred-focus to avoid mid-frame canvas tear, single-press Enter commits, Esc dismisses, click-outside dismisses. History records `InsertGroup(updated)` ↔ `InsertGroup(prev)` so Cmd-Z reverts the rename atomically.

### 4.7 Node disabled state ✓
Disabled nodes render dimmed and skip execution.

**Shipped:** `NodeInstance.disabled: bool` + `.with_disabled(true)` builder. `draw_node_at` wraps the node paint in `push_opacity(theme.node_disabled_alpha())` (0.45 default) so body / header / icon / title / badge all dim together. Editor builds a `disabled_nodes: HashSet<NodeId>` once per frame; any edge whose endpoint is in the set paints via `draw_edge_with_state(ConnectionState::Pending, …)` regardless of the connection's own state.

### 4.8 Click-to-select edge ✓
Edge curves are now clickable along their visible length, with a delete `×` affordance on the selected curve's midpoint.

**Shipped:** `edge:{id}` hit region registered via a 8-sample segment-AABB polyline of the cubic (shares the region id across every segment so the click handler doesn't disambiguate). Selected edge gets a thicker stroke + theme selection outline + delete-button overlay at the curve midpoint. Click on the delete button fires `DeleteConnectionRequested`. Selection survives drag (canvas-kit's `selection` set handles `edge:` prefixes uniformly).

### 4.9 Pan via space-drag / middle-click ✓
`blinc_canvas_kit` already shipped a two-mode tool model (`CanvasTool::Pan` / `Select`); the missing design-tool norms — temporary pan via spacebar-held + middle-click pan — are now in.

**Shipped:**
- `CanvasKit::set_force_pan(bool)` + `force_pan() -> bool` accessor on canvas-kit. Hosts toggle from any source.
- `handle_pointer_down` gate: when `evt.mouse_button == 1` (middle button) OR `force_pan` is set, skip hit-test + marquee setup → `interaction.active = None` so `handle_drag` falls through to the background-pan branch automatically. Selection survives (no clobber).
- Node editor's canvas div now binds `on_key_down` / `on_key_up` for `KeyCode::SPACE` → `kit.set_force_pan(true)` / `false`. Mid-drag pan keeps panning since the flag is consulted only at pointer-down — releasing space mid-pan doesn't abort.

**Out of scope here (canvas-kit policy):** canvas-kit stays keyboard-agnostic per its own design (`element()` builder explicitly doesn't attach key handlers). The spacebar listener lives on the editor's canvas div for now. Hosts that want space-pan on a stand-alone canvas-kit instance attach the same two-line key handler to their canvas wrapper.

---

## Tier 5 — Layout

### 5.1 Layered (Sugiyama) auto-layout
`LayoutStrategy::Layered` is currently `unimplemented!()`. Layout adjacency → longest-path layering → median in-layer ordering → Brandes-Köpfe coordinate assignment.

### 5.2 Force-directed layout ✓
`LayoutStrategy::Force` now runs a deterministic Hooke + Coulomb relaxation against the existing `ForceConfig` knobs.

**Shipped:** `apply_force_layout` in `layout.rs`. Seeds from the nodes' current positions (idempotent re-apply continues converging from where the user left off), iterates up to `max_iterations` ticks with all-pairs Coulomb repulsion + per-edge Hooke attraction. Damped velocity integration; early-exit when total kinetic energy falls below `spring * node_count * 1e-4`. Self-loops are filtered out so they don't produce NaN forces. Duplicate-position nodes get a tiny deterministic index-derived nudge so the first-tick repulsion direction is well-defined. Six unit tests in `layout::tests::force_layout_*` cover empty / single / overlapping / edge-pulled / determinism / self-loop. `NodeEditor::set_layout_strategy(strategy)` is the runtime sibling of `with_layout` so hosts can swap strategies live (demo's canvas context menu picks this up).

### 5.3 Snap-to-grid ✓
`CanvasKit::with_snap(spacing)` wired through the editor: `NodeEditor::with_snap(spacing)` builder, `set_snap_enabled(bool)`, `snap_enabled()`, `snap_point(Point)` accessor. `update_node_position` quantises on the way in; the drag-end handler also snaps before pushing `EditorEvent::NodeDragged` so hosts see the settled-on-grid position.

### 5.4 Align selected ✓
Quick actions on selected nodes — pure-math helpers + thin editor wrappers.

**Shipped:** `align_nodes(&[NodeId], AlignEdge)` + `distribute_nodes(&[NodeId], DistributeAxis)` methods. `AlignEdge::{Left, Right, CenterX, Top, Bottom, CenterY}` resolves the target coordinate from the bundle's existing extent (no node moves toward an arbitrary anchor). `DistributeAxis::{Horizontal, Vertical}` evenly spaces interior nodes between the two outermost anchors by centre coord. Each move routes through `update_node_position` so snap-to-grid + the graph-revision bump apply. `EditorCommand::AlignNodes` / `EditorCommand::DistributeNodes` variants for the dispatch path. Pure helpers `compute_align` / `compute_distribute` are unit-tested.

---

## Tier 6 — Command + Event API (host-as-driver, Blinc-signal-native)

**Palette / inspector / search bar / zoom HUD are host concerns.** Domain-specific UI varies per app (some hosts have a sidebar palette, some a top-down command palette; some want zoom controls in a toolbar, some want them in a status bar). The editor exposes a typed command channel (host → editor) and a SIGNAL channel (editor → host) so any host UI can drive the canvas and observe state reactively. The editor never ships those widgets.

**Exception — the minimap is editor core** (shipped, see §6.6). It is the one navigation surface that is generic across every host (a scaled overview + draggable viewport rect), depends only on the editor's own camera + graph bounds, and benefits from being painted inside the canvas's coordinate frame. So it lives in the editor rather than being re-implemented per host; it is config-toggled and on by default.

**The building blocks already exist in `blinc_canvas_kit`** — hosts compose their own UI on top of them:
- `kit.viewport_signal()` → reactive viewport (zoom, pan)
- `kit.viewport()` → current `CanvasViewport`
- `kit.update_viewport(|vp| ...)` → mutate (with `vp.reset()`, custom zoom, etc.)
- `kit.selection_signal()` → reactive selection set
- `kit.set_selection(...)` / `kit.clear_selection()`
- `kit.interaction_signal()` → reactive interaction state (hover, active region)

The node editor adds graph-level signals on top (`graph_signal`, `drag_state_signal`, events queue) — see §6.3 below.

See [`examples/canvas_kit_demo.rs::viewport_hud`](../../examples/blinc_app_examples/examples/canvas_kit_demo.rs) for the pattern: a `Stateful` widget with `.deps([kit.viewport_signal()])` re-renders on viewport change, reads `kit.viewport()` for the live value. Identical pattern works for zoom controls, a breadcrumb bar, an info HUD, etc. (The minimap used to be cited here as a host pattern; it now ships in core — §6.6.)

Callbacks are NOT the primary surface: Blinc's UI is built around `Signal<T>` + `derived(...)` + `effect(...)`, and threading callbacks into a reactive component is awkward (you end up bridging the callback's `Send + Sync` closure into a `State<T>` manually). State the host observes lives in signals. Synchronous responses (validation) stay as callbacks — they're the only places where the editor needs an answer mid-event, before the host can react.

### 6.1 Granular command methods on `NodeEditor` ✓

**Shipped:** Every granular mutation, selection, viewport, observability, bulk + align/distribute method listed below lives as a public method on `NodeEditor`. Each bumps the appropriate signal exactly once. Pre-fix the only path was `set_graph(...)` (full replace); now hosts can patch single entities without a full re-walk.

**Graph mutations**
- `insert_node(NodeInstance<N>)`
- `remove_node(NodeId)` — also drops incident connections
- `update_node_position(NodeId, Point)` — programmatic move (host-side layout result)
- `insert_connection(Connection<C>)`
- `remove_connection(ConnectionId)`
- `insert_group(Group<G>)` / `remove_group(GroupId)`
- `set_group_members(GroupId, Vec<NodeId>)`
- `set_group_collapsed(GroupId, bool)`

**Selection**
- `select(Vec<NodeId>)` / `select_one(NodeId)` / `add_to_selection(NodeId)`
- `clear_selection()`
- `selected_node_ids()` → `Vec<NodeId>`
- `selected_connection_ids()` → `Vec<ConnectionId>`

**Viewport**
- `focus_on_node(NodeId)` — pan + zoom to centre the node
- `zoom_to_fit()` — frame all nodes
- `zoom_to_selection()`
- `set_viewport(zoom: f32, pan: Point)`
- `viewport()` → `CanvasViewport` (read-back for save/restore)

**Runtime observability shortcuts** (avoid full `set_graph` for state-only updates)
- `set_node_badge(NodeId, Option<StatusBadge>)`
- `set_connection_state(ConnectionId, ConnectionState)`
- `flash_node(NodeId, FlashKind, Duration)` — transient highlight overlay (trace events, debug step-through)

**Bulk**
- `set_graph(...)` (existing) — full replace, used on initial load and CRDT re-sync

### 6.2 `EditorCommand` enum + `dispatch(cmd)` adapter ✓

**Shipped:** `EditorCommand<K, N, C, G>` enum covers every granular method (graph mutations, selection, viewport, observability, bulk, align/distribute) + a `Composite(Vec<EditorCommand>)` variant for atomic compound dispatch (used by `History` for the compound inverse of `RemoveNode`: re-insert node + every incident connection + restore each affected group's membership). `NodeEditor::dispatch(cmd)` matches and delegates to the matching granular method.

### 6.3 Signals (continuous observable state) ✓

Mirror canvas-kit's `selection_signal()` / `viewport_signal()` / `interaction_signal()` pattern. Each piece of editor state the host observes gets a `SignalId` getter so hosts can `derived(|| editor.X)` or `effect_with_deps(...)`:

- `graph_signal()` — bumps on every graph mutation (insert/remove node, connection, group; member changes; node position update). Hosts derived against this to refresh palette counts, breadcrumbs, audit logs.
- `selection_signal()` — already exists via `editor.canvas_kit().selection_signal()`. Re-exported.
- `viewport_signal()` — already exists via the kit. Re-exported.
- `drag_state_signal()` — bumps on `DragConnect` FSM transitions. Hosts can show "connecting…" overlay or block other interactions during a connect-drag.
- `hover_signal()` — bumps when hovered node / port / edge changes. Hosts can show tooltips, status-bar context.

State getters paired with each signal (call from inside a `derived` / `effect`):
- `graph_revision() -> u64`
- `drag_state() -> DragConnect`
- `hovered() -> Option<HoverTarget>` where `HoverTarget = Node(NodeId) | Port(PortAddress) | Edge(ConnectionId) | Group(GroupId)`

### 6.4 Event channel (one-shot events) ✓

**Shipped:** `EditorEvent<K>` enum + `events_signal() -> SignalId` + `drain_events() -> Vec<EditorEvent<K>>`. Hosts drain reactively by depending on `events_signal()` inside an `effect_with_deps` / `stateful_with_key().deps([...])` block. Variants cover every interaction outcome (drag settled, connection accepted / rejected, group create / add / remove / toggle / delete requests, multi-selection settled, selection cleared, node / edge clicks, layout applied, edit-title / edit-description / edit-group dialogs requested, keyboard-originated undo / redo / duplicate / select-all). Discrete events that don't fit a "current value" shape (a connection accepted, a node deleted, a group create request fired) live in this queue:

```rust
pub enum EditorEvent<K: PortKind> {
    NodeDragged { id: NodeId, position: Point },
    NodeClicked { id: NodeId, modifiers: ClickModifiers },
    NodeDoubleClicked { id: NodeId },
    NodeContextMenu { id: NodeId, screen_point: Point },
    CanvasContextMenu { content_point: Point, screen_point: Point },
    EdgeClicked { id: ConnectionId },
    ConnectionAccepted(ConnectionEvent<K>),
    CreateGroupRequested(CreateGroupRequest),
    AddToGroupRequested(AddToGroupRequest),
    RemoveFromGroupRequested(RemoveFromGroupRequest),
    ToggleCollapseRequested(ToggleCollapseRequest),
    DeleteGroupRequested(DeleteGroupRequest),
    DeleteConnectionRequested(ConnectionId),
    LayoutApplied(Vec<(NodeId, Point)>),
}
```

Editor API:
- `events_signal() -> SignalId` — bumps every time `EditorEvent` is pushed
- `drain_events() -> Vec<EditorEvent<K>>` — returns and clears pending events

Host pattern:
```rust
let evts = editor.events_signal();
effect_with_deps([evts], move || {
    for evt in editor.drain_events() {
        match evt {
            EditorEvent::ConnectionAccepted(c) => host_state.add_edge(c),
            EditorEvent::NodeDragged { id, position } => host_state.move_node(id, position),
            EditorEvent::CreateGroupRequested(req) => host_state.materialise_group(req),
            // …
        }
    }
});
```

### 6.5 Validation callbacks (the only callback surface) ✓

Validators need a SYNCHRONOUS response from the host while the user is mid-drag (e.g. should the preview line glow green or red as the user hovers a candidate port?). Signals are asynchronous; the host couldn't answer in time. So validators stay as callbacks:

- `on_connect_request(|req: &ConnectRequest<K>| -> ValidationOutcome)` — **shipped.** Drives the live preview line tint (green = accept, red = reject) and the post-release `ConnectionAccepted` / `ConnectionRejected` event split. `ValidationOutcome::Reject { reason }` carries a host-supplied reason string the editor broadcasts back via `ConnectionRejected` so hosts can surface it textually (toast / banner) — the live red preview already conveyed THAT it was rejected; the event carries WHY.
- (Future) `on_drop_validate(|payload, content_point| DropOutcome)` — for canvas drops from a palette.

### 6.6 What stays in the editor crate as types-only

Helper TYPE definitions stay (so hosts speak a shared vocabulary when implementing palette / inspector / search / zoom HUD):
- `palette::{PaletteQuery, PaletteInsertRequest, filter_templates}` — types + filter helper
- `inspector::InspectorPatchRequest` — patch event shape

What ships in core:
- **`minimap` module — SHIPPED.** Reversed the earlier "delete it" decision: the minimap is now a built-in editor feature (`MinimapConfig`, `Corner`, `with_minimap` / `set_minimap_enabled` / `minimap_config`), painted inside the canvas under the inverse-viewport transform, click/drag to navigate. See §6.6 below. The other host surfaces (palette, inspector, search) still stay host-built from the signals.

What goes away:
- Most event callbacks (`on_node_drag`, `on_create_group_request`, `on_connect_accepted`, …) — replaced by `events_signal` + `drain_events`
- Any "widget" pretense in `palette.rs` / `inspector.rs` — keep as type modules only

### 6.4 Trait surface (optional)

A `NodeEditorController` trait that exposes just the command + event API (without exposing the full `NodeEditor` struct) for hosts that want to abstract over multiple editor variants or test against a mock. Methods + dispatch + callback-registration methods. Lower priority — only if a host actually asks for it.

### 6.5 What types stay in the editor crate

Helper TYPE definitions stay (so hosts speak a shared vocabulary):
- `palette::{PaletteQuery, PaletteInsertRequest, filter_templates}` — types + filter helper
- `inspector::InspectorPatchRequest` — patch event shape
- `minimap` module — SHIPPED in core (`MinimapConfig` / `Corner` / `MinimapHit`); see §6.6. (Earlier this read "no minimap module — out of scope"; that decision was reversed.)

What goes away:
- `Minimap` struct (delete `minimap.rs`)
- Any "widget" pretense in `palette.rs` / `inspector.rs` — keep as type modules only

---

## Tier 7 — Runtime observability hooks

### 7.1 Streaming connection state
Hosts already update `Connection<M>.state` and re-sync via `set_graph`. Could expose an `update_connection_state(id, state)` shortcut that avoids a full re-walk for state-only changes.

### 7.2 Node execution badges
We have `StatusBadge::running()` / `error()` / etc. Hosts manually attach. Could expose `editor.set_node_status(id, badge)` so hosts don't manage their own copy.

### 7.3 Trace event ribbon
A bottom panel showing a stream of node-completion events. Host-rendered, but editor could expose `flash_node(id, kind)` to draw a transient highlight on a node when an event fires.

---

## Tier 8 — Collaboration / CRDT

### 8.1 Per-user presence
Render other users' cursors / selections on the canvas. `editor.set_presence(user_id, Presence { cursor, selected_ids, color })`.

### 8.2 CRDT-friendly mutation API
Currently mutations go through `set_graph` (full replace). For CRDT hosts, a delta-application API (`apply_delta(GraphDelta)`) would be more efficient.

---

## Tier 9 — Performance

### 9.1 Frustum cull ✓
`render_frame` computes `visible_content_rect_padded()` once (visible canvas-content rect inflated 25 % on each axis) and skips any node whose `node_bounds_for` doesn't intersect it. Same predicate culls edges via their endpoint-pair AABB. The 25 % slack keeps hit-region registration for elements at the screen edge + absorbs one frame of camera-pan motion without pop-in.

### 9.2 LOD for nodes at low zoom
Below zoom 0.2× the title text isn't readable. Render a coloured rounded rect with no text / no ports.

### 9.3 Edge bundling / proxy strokes
At zoom <0.2×, replace per-connection bezier with straight-line "edge bundles" that group multi-connection clusters into single thicker strokes.

### 9.4 Static-cache nodes that haven't moved
Once a node's position + state is unchanged across frames, its primitive sub-sequence in the bg batch can be reused without re-emit. Cache by `fingerprint + position hash`.

---

## Tier 11 — Metadata-driven node content slots (Zeal parity)

Zeal nodes embed real widgets inside the node body — code editors, dropdowns, number inputs, color pickers, rules tables — driven by per-template metadata. Our nodes currently render only header chrome; this tier adds the equivalent surface so a `cn::code_editor` / `cn::input` / `cn::select` lives inside each node, sized to its content. Unblocks IIP defaults + inline editing + custom-widget node types.

### 11.1 `NodeTemplate.content` — two surfaces

**Static (compile-time, Rust-typed)**: `NodeTemplate.content_render: Option<Arc<dyn Fn(&NodeInstance<M>, &EditorContext) -> Box<dyn ElementBuilder> + Send + Sync>>`. Hosts hand in a closure that builds Blinc elements. Most flexible; the natural fit for nan8's reflow-typed actor templates.

**Dynamic (runtime, JSON/DSL)**: `NodeTemplate.content_schema: Option<ContentSchema>` where `ContentSchema = Vec<ContentItem>` and `ContentItem` enums the supported widgets (`Text`, `Input { kind, label, default }`, `Select { options }`, `CodeEditor { language }`, `ColorPicker`, `Custom(serde_json::Value)`). The renderer interprets the schema and emits the matching `cn::*` widget bound to the node's `metadata.extra[field]`.

### 11.2 Embedding architecture

Our nodes currently render inside a single canvas closure (primitives only, no layout tree). Embedding requires per-node layout subtrees:

* **Layout-subtree-per-node** (preferred): each node becomes a `Stateful` Div positioned absolutely over the canvas (or wrapped in a custom widget that pans/zooms with the viewport transform). Edges + groups stay in the canvas. The bridge code translates `kit.viewport()` → CSS transform on the node wrapper so nodes pan / zoom with the canvas content.
* **Render-into-canvas adapter**: a `CanvasEmbed` widget rasterises a sub-tree to primitives each frame. More invasive; loses event routing for the embedded widget.

The first is the right path.

### 11.3 Implementation order

1. Refactor `NodeEditor::element()` to mount each node as an absolutely-positioned `Stateful` div above the canvas (canvas keeps edges + groups + ports). Pan/zoom plumbing.
2. Header chrome moves to that div (icon + title + subtitle + status badge); body stays empty for now.
3. Add `NodeTemplate.content_render` (static path) + render its output inside the body.
4. Add `ContentSchema` + renderer mapping (dynamic path).
5. Wire IIP defaults: `PortMetadata.iip_default` becomes an inline `cn::input` next to the port.

### 11.4 Sizing

Each node sizes to `max(intrinsic_content + header, instance.size)`. Drop the empty-body dead space currently visible when `instance.size` exceeds intrinsic — until content slots land, body height = header height.

---

## Tier 10 — Misc

### 10.1 Node size presets
Zeal's `'small' | 'medium' | 'large'`. We have explicit `(f32, f32)` — strictly more flexible but loses convention. Add `NodeSizePreset` helper that resolves to size pairs.

### 10.2 Node pin / favourite state
Per-instance flag for "always visible in palette." Host-defined; editor surfaces a small pin icon when set.

### 10.3 Custom node renderer
`NodeShape::Custom` is reserved but falls back to Rectangle. Closure-on-template that receives `(ctx, bounds, theme)` and paints anything — for fully bespoke node shapes.

### 10.4 Renaming `GlyphInstance.uv_bounds`
Field stores atlas pixel coords now, not UVs. Misleading name. (Upstream `blinc_text`, not editor.)

---

## Implementation order suggestion

**Shipped batches:**

1. ~~Tier 6.1 + 6.3~~ granular methods + signals ✓
2. ~~Tier 4.1 + 4.2~~ drag-into / drag-out group ✓
3. ~~Tier 4.3~~ marquee multi-select ✓
4. ~~Tier 4.4~~ keyboard shortcuts (Esc / Delete / D / Cmd-A / Cmd-D / Cmd-Z / Cmd-Shift-Z) + History module ✓
5. ~~Tier 3.3~~ group header chrome buttons + combined edit dialog ✓
6. ~~Tier 1.1~~ port description ✓
7. ~~Tier 1.2~~ edge animation (Running shimmer / Pending pulse) ✓
8. ~~Tier 4.6~~ inline title rename (group + node) ✓
9. ~~Tier 4.7~~ node disabled state ✓
10. ~~Tier 4.8~~ click-to-select edge ✓
11. ~~Tier 6.2~~ `EditorCommand` enum + dispatch (incl. `Composite`) ✓
12. ~~Tier 6.4~~ event channel (`events_signal` + `drain_events`) ✓
13. ~~Tier 6.5~~ validation callbacks (`on_connect_request`) ✓
14. ~~Tier 9.1~~ frustum cull ✓
15. ~~Tier 5.3 + 5.4~~ snap-to-grid + align/distribute ✓
16. ~~Tier 4.9~~ space-drag + middle-click pan (in canvas-kit + editor key-binding) ✓
17. ~~Tier 4.5~~ context menu (node / edge / group / canvas) via `ContextMenuRequested` + demo `cn::context_menu` ✓
18. ~~Tier 1.3~~ subgraph nav (`SubgraphRef` + diamond chrome + `SubgraphRequested` event + demo expand-into-wrapper save flow) ✓
19. ~~Tier 5.2~~ force-directed layout (Hooke + Coulomb, deterministic, demo-wired via canvas context menu) ✓

**Next:**

1. **Tier 2.1** (typed `PropertyDefinition`) — hosts can already implement inspectors with the opaque-JSON model; formalised schema is an ergonomics upgrade, not a blocker.
2. **Tier 5.1** (layered Sugiyama layout) — the second `unimplemented!()` stub. Multi-pass: cycle break → longest-path layering → median in-layer ordering → Brandes-Köpfe coordinate assignment.
3. **Tier 11.x** (portal-UI content slots polish) — partly done; close out remaining subitems.
5. **Tier 3.1 + 3.2** (group timestamps + relative member positions) — small data-model adds.
6. **Tier 7.3** (trace event ribbon) — flash-node already shipped; ribbon is host-side polish.
7. **Tier 9.2–9.4** (LOD / edge bundling / static-cache) — only when a real graph saturates the existing pipeline.
8. **Tier 10.x** (size presets / pin / custom renderer / GlyphInstance rename) — low priority polish.
9. **Tier 8** (CRDT / presence) — only when nan8 actually needs it.


---

## What we DO have that Zeal doesn't

- **Generic over port-kind type** (`PortKind` trait) — Zeal's port type is implicit `any`; we enforce host-typed ports + compatibility matching at the editor level.
- **Theme-driven chrome with squircle / shadow / typography tokens** — Zeal renders against fixed Tailwind classes; we re-tint instantly with `with_theme(bundle)`.
- **Per-template flex layout via blinc_layout** (`slot.rs` taffy compute) — Zeal hand-codes node geometry.
- **Off-render slot cache warmed via rayon** — Zeal recomputes node layout each render.
- **Pre-built status badges with theme-aware colours** — Zeal's badges are hand-coloured per component.
- **First-class metadata generics** (`<K, N, C, G>`) — Zeal's metadata is `Record<string, any>`.
