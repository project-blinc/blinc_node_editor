//! Auto-layout strategies — Manual (no-op), Layered (Sugiyama-style),
//! Force-directed, and host-supplied Custom closures.
//!
//! `Manual` is the default (positions come from the host's
//! `NodeInstance.position`). `Layered` is the typical pick for
//! DAGs and respects group bounding-boxes via a hierarchical
//! intra-then-super-graph pipeline. `Force` is for organic /
//! cyclic graphs and runs the same hierarchical pattern over
//! Fruchterman-Reingold-style repulsion + edge-spring attraction.

use crate::connection::Connection;
use crate::node::NodeInstance;
use blinc_core::layer::Point;

// ─────────────────────────────────────────────────────────────────────
// LayoutStrategy
// ─────────────────────────────────────────────────────────────────────

/// Strategy for placing nodes on the canvas. Default is `Manual` —
/// the host's `NodeInstance.position` values are used as-is.
#[derive(Clone, Default)]
pub enum LayoutStrategy {
    /// No-op. Positions come from the host's `NodeInstance.position`.
    /// Default.
    #[default]
    Manual,
    /// Sugiyama-style layered layout for DAGs:
    /// 1. Cycle break via DFS back-edge reversal
    /// 2. Longest-path layering from sources → layers
    /// 3. In-layer ordering (median heuristic)
    /// 4. Coordinate assignment
    ///
    /// Respects group containment via a hierarchical pipeline:
    /// each group's members lay out internally first, then a
    /// super-graph of (groups + free nodes) lays out together,
    /// then group-relative positions are translated into the
    /// super-graph slot. Configurable via [`LayeredConfig`].
    Layered(LayeredConfig),
    /// Force-directed layout for cyclic / organic graphs. Suitable
    /// when the graph has cycles that defeat the layered approach.
    /// Runs the same hierarchical (intra → super-graph → assembly)
    /// pipeline as [`LayoutStrategy::Layered`] but uses
    /// Fruchterman-Reingold repulsion + edge-spring attraction at
    /// each level. Configurable via [`ForceConfig`].
    Force(ForceConfig),
    /// Host-supplied layout — the host provides a closure that
    /// rewrites positions however it likes. Useful for hosts that
    /// want a single bespoke layout (e.g. ML inference graphs with
    /// known visualisation conventions).
    Custom(CustomLayoutFn),
}

/// Knobs for `LayoutStrategy::Layered`.
#[derive(Debug, Clone)]
pub struct LayeredConfig {
    /// Horizontal distance between consecutive layers (left → right
    /// flow). Default 240px.
    pub layer_spacing: f32,
    /// Vertical distance between nodes in the same layer. Default 96px.
    pub in_layer_spacing: f32,
    /// Whether to lay out left-to-right (default) or top-to-bottom.
    pub orientation: LayoutOrientation,
}

impl Default for LayeredConfig {
    fn default() -> Self {
        Self {
            layer_spacing: 240.0,
            in_layer_spacing: 96.0,
            orientation: LayoutOrientation::LeftToRight,
        }
    }
}

/// Knobs for `LayoutStrategy::Force`.
///
/// Group boundaries are a HARD constraint: when the input contains
/// any group, the layout runs a two-level pass (members internal,
/// then group super-nodes at the top level) so members never escape
/// their group's bbox and non-members never invade it. No soft-
/// constraint knobs to tune.
#[derive(Debug, Clone)]
pub struct ForceConfig {
    /// Repulsion strength between every pair of nodes (Coulomb).
    pub repulsion: f32,
    /// Spring constant for connected node pairs (Hooke).
    pub spring: f32,
    /// Ideal edge length the spring is centred on.
    pub ideal_edge_length: f32,
    /// Per-tick damping factor. Lower = converges faster but jankier.
    pub damping: f32,
    /// Max iterations before stopping regardless of convergence.
    pub max_iterations: u32,
}

impl Default for ForceConfig {
    fn default() -> Self {
        Self {
            repulsion: 4000.0,
            spring: 0.06,
            ideal_edge_length: 180.0,
            damping: 0.92,
            max_iterations: 400,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LayoutOrientation {
    #[default]
    LeftToRight,
    TopToBottom,
}

/// Host-supplied layout function. Receives the current nodes +
/// connections; returns a new vec of positions in node-id order
/// (`positions[i]` = new position for `nodes[i]`).
///
/// Boxed + `Send + Sync` so the editor can store it cheaply and
/// invoke it from the layout-trigger event.
pub type CustomLayoutFn = std::sync::Arc<
    dyn Fn(&LayoutContext<'_>) -> Vec<Point> + Send + Sync + 'static,
>;

// Manual implementation of Debug because Arc<dyn Fn> isn't Debug.
impl std::fmt::Debug for LayoutStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manual => write!(f, "Manual"),
            Self::Layered(c) => write!(f, "Layered({c:?})"),
            Self::Force(c) => write!(f, "Force({c:?})"),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

/// Input to a [`CustomLayoutFn`]. Both lists are borrowed; the
/// closure returns positions without mutating either.
pub struct LayoutContext<'a> {
    pub nodes: &'a [LayoutNode<'a>],
    pub edges: &'a [LayoutEdge<'a>],
}

/// Minimal node view passed to the layout closure — id + current
/// position + bounding-box size. The closure doesn't need to see
/// host metadata.
pub struct LayoutNode<'a> {
    pub id: &'a crate::node::NodeId,
    pub position: Point,
    pub size: (f32, f32),
}

/// Minimal edge view — endpoint node-ids only.
pub struct LayoutEdge<'a> {
    pub from: &'a crate::node::NodeId,
    pub to: &'a crate::node::NodeId,
}

// ─────────────────────────────────────────────────────────────────────
// apply_layout — driver the editor calls
// ─────────────────────────────────────────────────────────────────────

/// Apply the strategy to a node list + edge list, returning new
/// positions in the same order as `nodes`. The editor's
/// "auto-layout" command calls this and patches each
/// `NodeInstance.position`.
///
/// `Manual` returns the existing positions unchanged. `Layered`
/// and `Force` run their respective hierarchical pipelines.
/// `Custom` calls the host's closure with the snapshot.
pub fn apply_layout<N, M, G>(
    strategy: &LayoutStrategy,
    nodes: &[NodeInstance<N>],
    connections: &[Connection<M>],
    groups: &[crate::group::Group<G>],
) -> Vec<Point> {
    match strategy {
        LayoutStrategy::Manual => nodes.iter().map(|n| n.position).collect(),

        LayoutStrategy::Layered(config) => {
            apply_layered_layout(config, nodes, connections, groups)
        }

        LayoutStrategy::Force(config) => {
            apply_force_layout(config, nodes, connections, groups)
        }

        LayoutStrategy::Custom(f) => {
            let node_views: Vec<LayoutNode<'_>> = nodes
                .iter()
                .map(|n| LayoutNode {
                    id: &n.id,
                    position: n.position,
                    size: n.size.unwrap_or((180.0, 72.0)),
                })
                .collect();
            let edge_views: Vec<LayoutEdge<'_>> = connections
                .iter()
                .map(|c| LayoutEdge {
                    from: &c.from.node,
                    to: &c.to.node,
                })
                .collect();
            let ctx = LayoutContext {
                nodes: &node_views,
                edges: &edge_views,
            };
            f(&ctx)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Force-directed layout (Hooke + Coulomb, deterministic)
// ─────────────────────────────────────────────────────────────────────

/// Iterative Hooke + Coulomb force-directed layout. Starts from the
/// nodes' CURRENT positions (so applying the layout twice converges
/// further from where the user left off, instead of teleporting to
/// a random initial state) and runs `config.max_iterations` ticks
/// or until kinetic energy falls below a quiescence threshold,
/// whichever comes first.
///
/// Forces per tick:
///   * **Repulsion** between every node pair `i, j` — `F = repulsion
///     / d²` along the unit vector from `j` to `i`. Same model as
///     point-charges; the inverse-square falloff lets clusters stay
///     coherent while still separating overlapping pairs.
///   * **Attraction** along each edge `(u, v)` — `F = spring *
///     (d - ideal_edge_length)` along the unit vector. Hooke's law
///     with rest length `ideal_edge_length`; positive when stretched,
///     negative (i.e. pushing apart) when compressed below the rest
///     length so edges can re-expand if a repulsion pulse jammed
///     them together.
///   * **Damping** — velocity is multiplied by `damping` at the end
///     of each tick so the system loses energy and settles.
///
/// Deterministic for a given input: no RNG. Two nodes at exactly the
/// same position get a tiny index-derived nudge on the first tick so
/// the repulsion vector is well-defined.
fn apply_force_layout<N, M, G>(
    config: &ForceConfig,
    nodes: &[NodeInstance<N>],
    connections: &[Connection<M>],
    groups: &[crate::group::Group<G>],
) -> Vec<Point> {
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![nodes[0].position];
    }

    // Build an id→index map once so the edge loop doesn't do linear
    // searches per tick.
    let mut id_to_index: ahash::AHashMap<crate::node::NodeId, usize> =
        ahash::AHashMap::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        id_to_index.insert(node.id.clone(), i);
    }

    // Pre-resolve each group's member indices. Groups with zero
    // members on this graph are skipped (e.g. a group whose every
    // member id is stale).
    let group_members: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            g.members
                .iter()
                .filter_map(|m| id_to_index.get(m).copied())
                .collect()
        })
        .collect();
    let has_groups = group_members.iter().any(|m| !m.is_empty());

    // ── Hierarchical path ────────────────────────────────────────────
    //
    // When the graph has any groups, run a TWO-LEVEL layout so group
    // membership is a hard constraint (not a soft spring):
    //
    //   1. For each group, lay out its members internally with their
    //      intra-group edges. Translate the result so the member
    //      centroid sits at origin; we'll re-anchor in step 3.
    //   2. Build a super-graph where every group becomes ONE node
    //      (with a "size" matching its members' bbox + padding) and
    //      every cross-group / group-to-free / free-to-free edge
    //      becomes a super-edge. Lay this out with the same FR
    //      kernel.
    //   3. For each free node, take its super-position. For each
    //      group member, translate its intra-group position by the
    //      group's super-position so the member's final spot is
    //      `super_pos + intra_offset`. Members never escape the
    //      group's bbox by construction; non-members never invade it.
    //
    // This sidesteps the soft-spring tuning rabbit hole the prior
    // implementation kept failing on: cohesion strong enough to
    // bind members would over-clamp connected non-members, and
    // separation strong enough to keep non-members out would warp
    // the top-level shape. Hierarchical FR is the canonical fix.
    if has_groups {
        return apply_hierarchical_force_layout(
            config,
            nodes,
            connections,
            groups,
            &id_to_index,
            &group_members,
        );
    }

    // ── Single-level path (no groups) ─────────────────────────────────
    let edges: Vec<(usize, usize)> = connections
        .iter()
        .filter_map(|c| {
            let from = id_to_index.get(&c.from.node)?;
            let to = id_to_index.get(&c.to.node)?;
            if from == to {
                None
            } else {
                Some((*from, *to))
            }
        })
        .collect();

    let mut positions: Vec<(f32, f32)> = nodes
        .iter()
        .map(|node| (node.position.x, node.position.y))
        .collect();
    // Per-node node-bbox half-extent (max of width / height halves)
    // so the kernel's "personal space" zone matches the node's
    // rendered footprint. Without this the kernel happily packs
    // nodes whose bboxes overlap — visually unreadable.
    let radii: Vec<f32> = nodes
        .iter()
        .map(|n| {
            let (w, h) = n.size.unwrap_or((180.0, 72.0));
            (w * 0.5).max(h * 0.5)
        })
        .collect();

    nudge_duplicates(&mut positions);
    // Top-level (no groups): no centroid pull — edges + repulsion
    // alone determine the spread. Nodes with no edges and no other
    // forces will be left near their input positions, which matches
    // FR's contract for disconnected components.
    force_kernel(&mut positions, &edges, &radii, config, 0.0);
    positions
        .into_iter()
        .map(|(x, y)| Point::new(x, y))
        .collect()
}

/// Constant referenced everywhere we need to keep distance from
/// being zero (which would cause `1/dist` singularities). Small
/// enough that it's never visually meaningful but large enough
/// that `f32` doesn't round to zero.
const KERNEL_EPSILON: f32 = 1e-3;

/// Spiral-out nudge for any pair of nodes landing on the same
/// coordinate. Deterministic by index so a repeat layout call
/// produces the same nudges. Idempotent — once nodes are 1 px
/// apart, no further work happens.
fn nudge_duplicates(positions: &mut [(f32, f32)]) {
    let n = positions.len();
    for i in 1..n {
        for j in 0..i {
            let dx = positions[i].0 - positions[j].0;
            let dy = positions[i].1 - positions[j].1;
            if dx.abs() < KERNEL_EPSILON && dy.abs() < KERNEL_EPSILON {
                let t = (i as f32) * 0.5;
                positions[i].0 += t.cos();
                positions[i].1 += t.sin();
            }
        }
    }
}

/// Core Fruchterman-Reingold-style force kernel. Drives `positions`
/// in place from their current values toward a settled state.
///
/// * **Repulsion** — every pair gets an inverse-square push,
///   centre-distance based; the radii scale the magnitude linearly
///   so larger nodes / super-nodes hold a proportionally wider
///   neighbourhood.
/// * **Attraction** — each edge's rest length is `ideal_edge_length
///   + r_u + r_v` so connected nodes settle with the configured gap
///   regardless of node size.
/// * **Centroid attraction** (optional) — when `centroid_pull > 0`,
///   every node gets a linear Hooke pull toward the population
///   centroid. This is the "phase-1 cohesion" force the hierarchical
///   layout uses to keep group members tight when they have few or
///   no intra-group edges: pure repulsion with damping has no
///   stable equilibrium for an unconnected pair (they drift apart
///   until energy bleeds out at whatever spread the iteration count
///   allows), so we add a virtual centripetal spring that caps the
///   spread at `~ ideal_edge_length` regardless of edge topology.
/// * **Damping** — velocity is scaled by `damping` each tick so the
///   system loses energy. Early-exit when total kinetic energy
///   falls below `spring * n * 1e-4`.
fn force_kernel(
    positions: &mut [(f32, f32)],
    edges: &[(usize, usize)],
    radii: &[f32],
    config: &ForceConfig,
    centroid_pull: f32,
) {
    let n = positions.len();
    if n < 2 {
        return;
    }
    let mut velocities: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    let quiescence = (config.spring * n as f32).max(1.0) * 1e-4;

    for _ in 0..config.max_iterations {
        let mut forces: Vec<(f32, f32)> = vec![(0.0, 0.0); n];

        // Repulsion (all pairs). Inverse-square on centre-distance.
        // Earlier attempts to measure "distance between bbox edges"
        // here (`dist - r_i - r_j`) created a singularity when bboxes
        // touched — repulsion magnitude exploded to ~10⁹ and the
        // system numerically diverged. Centre-distance is stable; we
        // still account for radii on the ATTRACTIVE side so connected
        // nodes settle at `ideal + r_u + r_v` apart, which gives the
        // intended gap between bboxes anyway.
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                let dist_sq = (dx * dx + dy * dy).max(KERNEL_EPSILON);
                let dist = dist_sq.sqrt();
                // Inflate repulsion by `(1 + r_i + r_j /
                // ideal_edge_length)` so larger super-nodes hold a
                // proportionally wider neighbourhood. Linear in
                // radii — no singularity at touching bboxes because
                // the denominator stays centre-distance-based.
                let r_scale = 1.0
                    + (radii[i] + radii[j]) / config.ideal_edge_length.max(KERNEL_EPSILON);
                let magnitude = config.repulsion * r_scale / dist_sq;
                let fx = (dx / dist) * magnitude;
                let fy = (dy / dist) * magnitude;
                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[j].0 -= fx;
                forces[j].1 -= fy;
            }
        }

        // Attraction. Rest length = ideal_edge_length + r_u + r_v so
        // the gap between bbox edges is `ideal_edge_length` instead
        // of `ideal_edge_length - r_u - r_v` (which would jam big
        // nodes into each other).
        for &(u, v) in edges {
            let dx = positions[v].0 - positions[u].0;
            let dy = positions[v].1 - positions[u].1;
            let dist = (dx * dx + dy * dy).max(KERNEL_EPSILON).sqrt();
            let rest = config.ideal_edge_length + radii[u] + radii[v];
            let displacement = dist - rest;
            let magnitude = config.spring * displacement;
            let inv_dist = 1.0 / dist;
            let fx = dx * inv_dist * magnitude;
            let fy = dy * inv_dist * magnitude;
            forces[u].0 += fx;
            forces[u].1 += fy;
            forces[v].0 -= fx;
            forces[v].1 -= fy;
        }

        // Centroid attraction (phase-1 cohesion). Pulls each node
        // toward the population centroid with a Hooke spring of
        // strength `centroid_pull`. Zero when called from the
        // top-level layout (where edges + repulsion alone settle to
        // a stable spread); non-zero when called from a group's
        // intra-layout, where a tight cluster is the desired
        // outcome regardless of how many edges the members share.
        // Without this, an unconnected pair drifts apart further on
        // every layout invocation — pure repulsion has no rest
        // length, so each click of the auto-layout button compounds
        // the spread.
        if centroid_pull > 0.0 {
            let (sx, sy) = positions.iter().fold((0.0_f32, 0.0_f32), |(sx, sy), &(x, y)| {
                (sx + x, sy + y)
            });
            let inv = 1.0 / n as f32;
            let (cx, cy) = (sx * inv, sy * inv);
            for i in 0..n {
                let dx = cx - positions[i].0;
                let dy = cy - positions[i].1;
                forces[i].0 += dx * centroid_pull;
                forces[i].1 += dy * centroid_pull;
            }
        }

        // Integrate. Damping applies AFTER the force impulse so a
        // tick with strong forces still produces motion before the
        // damping bleeds energy out.
        let mut total_kinetic: f32 = 0.0;
        for i in 0..n {
            velocities[i].0 = (velocities[i].0 + forces[i].0) * config.damping;
            velocities[i].1 = (velocities[i].1 + forces[i].1) * config.damping;
            positions[i].0 += velocities[i].0;
            positions[i].1 += velocities[i].1;
            total_kinetic +=
                velocities[i].0 * velocities[i].0 + velocities[i].1 * velocities[i].1;
        }

        if total_kinetic < quiescence {
            break;
        }
    }
}

/// Two-level layout that treats groups as a HARD constraint.
///
/// Phase 1: for each group, run the kernel on its members + intra-
/// group edges only. The group's footprint is the bbox of the
/// resulting positions; the member offsets are stored relative to
/// the group's local origin (top-left of the bbox).
///
/// Phase 2: build a super-graph — one node per free (no-group) host
/// node + one super-node per group, sized by the group's phase-1
/// footprint. Cross-group + cross-boundary edges become edges in
/// the super-graph. Run the kernel on it.
///
/// Phase 3: assemble final positions. Free nodes get their super-
/// position verbatim; group members get `super_pos + intra_offset`
/// so they sit inside the group's footprint exactly where phase 1
/// placed them, with the whole group translated to wherever phase
/// 2 ended up putting it.
///
/// Net effect: members never escape their group bbox; non-members
/// never invade it. The soft-spring cohesion / separation knobs are
/// unused on this path — they were an earlier (insufficient) attempt
/// at the same constraint and kept producing the "members outside,
/// non-members inside" mess the user reported.
fn apply_hierarchical_force_layout<N, M, G>(
    config: &ForceConfig,
    nodes: &[NodeInstance<N>],
    connections: &[Connection<M>],
    _groups: &[crate::group::Group<G>],
    id_to_index: &ahash::AHashMap<crate::node::NodeId, usize>,
    group_members: &[Vec<usize>],
) -> Vec<Point> {
    let n = nodes.len();
    // For each node, which group claims it (first match wins —
    // multi-group membership picks the smallest-indexed group; the
    // others get the node's primary footprint via the same shared
    // super-position).
    let mut primary_group: Vec<Option<usize>> = vec![None; n];
    for (gi, members) in group_members.iter().enumerate() {
        for &m in members {
            if primary_group[m].is_none() {
                primary_group[m] = Some(gi);
            }
        }
    }

    let node_radius = |i: usize| {
        let (w, h) = nodes[i].size.unwrap_or((180.0, 72.0));
        (w * 0.5).max(h * 0.5)
    };

    // ── Phase 1: lay out each group internally ───────────────────────
    //
    // `group_local` holds the per-group layout: for each member
    // index, the relative offset from the group's origin (top-left
    // of the post-layout bbox). `group_size` is (width, height) of
    // that bbox; used to size the super-node in phase 2.
    let mut group_local: Vec<ahash::AHashMap<usize, (f32, f32)>> =
        vec![ahash::AHashMap::default(); group_members.len()];
    let mut group_size: Vec<(f32, f32)> = vec![(0.0, 0.0); group_members.len()];

    for (gi, members) in group_members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        // Build intra-group edge list mapped to LOCAL indices.
        let mut local_index: ahash::AHashMap<usize, usize> = ahash::AHashMap::default();
        for (li, &gi_member) in members.iter().enumerate() {
            local_index.insert(gi_member, li);
        }
        let intra_edges: Vec<(usize, usize)> = connections
            .iter()
            .filter_map(|c| {
                let from = *id_to_index.get(&c.from.node)?;
                let to = *id_to_index.get(&c.to.node)?;
                let lf = *local_index.get(&from)?;
                let lt = *local_index.get(&to)?;
                if lf == lt {
                    None
                } else {
                    Some((lf, lt))
                }
            })
            .collect();
        let mut local_positions: Vec<(f32, f32)> = members
            .iter()
            .map(|&gi_member| {
                let pos = nodes[gi_member].position;
                (pos.x, pos.y)
            })
            .collect();
        let local_radii: Vec<f32> = members.iter().map(|&m| node_radius(m)).collect();

        if intra_edges.is_empty() {
            // No intra-group edges → pure repulsion would diverge
            // (no spring rest length to balance it against), and a
            // soft centroid-pull spring is numerically unstable
            // because repulsion blows up to ~∞ as nodes converge
            // to the centroid (the integrator overshoots then
            // bounces out, oscillating with amplitude growth that
            // outpaces damping). Fall back to a deterministic
            // compact grid centred on the input centroid — gives a
            // tight cluster every time and idempotent across
            // re-layouts (the same input always returns the same
            // output, so the gap doesn't compound on repeated
            // clicks of "Auto-layout").
            let (sx, sy) = local_positions
                .iter()
                .fold((0.0_f32, 0.0_f32), |(sx, sy), &(x, y)| (sx + x, sy + y));
            let inv = 1.0 / members.len() as f32;
            let (cx, cy) = (sx * inv, sy * inv);
            let max_radius = local_radii.iter().copied().fold(0.0_f32, f32::max);
            // Grouped, unconnected nodes are CONCEPTUALLY clustered
            // — the user grouped them because they belong together,
            // not because they need the connected-node breathing
            // room `ideal_edge_length` describes. Use 30 % of
            // `ideal_edge_length` as the bbox-edge gap (≈ 54 px on
            // default config), with a 20 px floor so radius-only
            // spacing always leaves a visible separator.
            let bbox_gap = (config.ideal_edge_length * 0.3).max(20.0);
            let spacing = max_radius * 2.0 + bbox_gap;
            let cols = (members.len() as f32).sqrt().ceil().max(1.0) as usize;
            let rows = members.len().div_ceil(cols);
            let grid_w = (cols.saturating_sub(1)) as f32 * spacing;
            let grid_h = (rows.saturating_sub(1)) as f32 * spacing;
            for (li, _) in members.iter().enumerate() {
                let col = li % cols;
                let row = li / cols;
                local_positions[li] = (
                    cx - grid_w * 0.5 + col as f32 * spacing,
                    cy - grid_h * 0.5 + row as f32 * spacing,
                );
            }
        } else {
            // With intra-edges, the force kernel converges
            // cleanly — the spring rest length gives the system a
            // stable equilibrium. No centroid pull needed; the
            // edges + repulsion + damping carry the layout.
            nudge_duplicates(&mut local_positions);
            force_kernel(
                &mut local_positions,
                &intra_edges,
                &local_radii,
                config,
                0.0,
            );
        }

        // Translate so the bbox top-left sits at origin; store
        // size for phase 2's super-node footprint.
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (li, &(x, y)) in local_positions.iter().enumerate() {
            let r = local_radii[li];
            min_x = min_x.min(x - r);
            min_y = min_y.min(y - r);
            max_x = max_x.max(x + r);
            max_y = max_y.max(y + r);
        }
        let (w, h) = (max_x - min_x, max_y - min_y);
        group_size[gi] = (w, h);
        for (li, &gi_member) in members.iter().enumerate() {
            let (x, y) = local_positions[li];
            group_local[gi].insert(gi_member, (x - min_x, y - min_y));
        }
    }

    // ── Phase 2: super-graph layout ──────────────────────────────────
    //
    // Super-nodes are indexed as:
    //   `0..free_count`               — free (no-group) host nodes
    //   `free_count..free_count + G`  — one super-node per group
    let free_indices: Vec<usize> = (0..n).filter(|i| primary_group[*i].is_none()).collect();
    let free_count = free_indices.len();
    let group_count = group_members.len();
    let super_n = free_count + group_count;

    // Index lookups:
    //   `host_to_super[i]` = super-index for host node i
    let mut host_to_super: Vec<usize> = vec![usize::MAX; n];
    for (super_idx, &host_idx) in free_indices.iter().enumerate() {
        host_to_super[host_idx] = super_idx;
    }
    for (gi, members) in group_members.iter().enumerate() {
        let super_idx = free_count + gi;
        for &m in members {
            host_to_super[m] = super_idx;
        }
    }

    // Super-positions: seed free nodes from their current host
    // positions; seed group super-nodes from the member centroid so
    // the relaxation continues from where the user left off (group
    // shapes don't teleport across the canvas).
    let mut super_positions: Vec<(f32, f32)> = Vec::with_capacity(super_n);
    for &host_idx in &free_indices {
        let p = nodes[host_idx].position;
        super_positions.push((p.x, p.y));
    }
    for members in group_members.iter() {
        if members.is_empty() {
            super_positions.push((0.0, 0.0));
            continue;
        }
        let (sx, sy) = members.iter().fold((0.0_f32, 0.0_f32), |(sx, sy), &m| {
            let p = nodes[m].position;
            (sx + p.x, sy + p.y)
        });
        let inv = 1.0 / members.len() as f32;
        super_positions.push((sx * inv, sy * inv));
    }

    // Super-radii: free nodes use their bbox half-extent; group
    // super-nodes use HALF the GROUP'S diagonal extent so the
    // kernel's personal-space zone keeps groups from overlapping.
    let mut super_radii: Vec<f32> = Vec::with_capacity(super_n);
    for &host_idx in &free_indices {
        super_radii.push(node_radius(host_idx));
    }
    for (gi, _) in group_members.iter().enumerate() {
        let (w, h) = group_size[gi];
        super_radii.push(((w * w + h * h).sqrt() * 0.5).max(node_radius(0)));
    }

    // Super-edges: rewrite each connection into super-space and
    // drop intra-group and self-edges (intra-group was handled in
    // phase 1; self-edges would NaN the force kernel).
    let mut super_edges_set: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for c in connections {
        let from = match id_to_index.get(&c.from.node) {
            Some(i) => *i,
            None => continue,
        };
        let to = match id_to_index.get(&c.to.node) {
            Some(i) => *i,
            None => continue,
        };
        let sf = host_to_super[from];
        let st = host_to_super[to];
        if sf == st || sf == usize::MAX || st == usize::MAX {
            continue;
        }
        let (a, b) = if sf < st { (sf, st) } else { (st, sf) };
        super_edges_set.insert((a, b));
    }
    let super_edges: Vec<(usize, usize)> = super_edges_set.into_iter().collect();

    nudge_duplicates(&mut super_positions);
    // Super-graph (phase 2): no centroid pull. The top-level layout
    // wants free spread between groups + free nodes — pulling them
    // toward a common centroid would defeat that.
    force_kernel(
        &mut super_positions,
        &super_edges,
        &super_radii,
        config,
        0.0,
    );

    // ── Phase 3: assemble final positions ────────────────────────────
    let mut final_positions: Vec<Point> = Vec::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        let super_idx = host_to_super[i];
        let (sx, sy) = if super_idx != usize::MAX {
            super_positions[super_idx]
        } else {
            (node.position.x, node.position.y)
        };
        let pos = match primary_group[i] {
            None => Point::new(sx, sy),
            Some(gi) => {
                // Translate the group's bbox so its TOP-LEFT lands
                // at the super-position minus half the bbox; the
                // member's intra offset is from that same origin.
                let (w, h) = group_size[gi];
                let origin_x = sx - w * 0.5;
                let origin_y = sy - h * 0.5;
                let (ox, oy) = group_local[gi]
                    .get(&i)
                    .copied()
                    .unwrap_or((0.0, 0.0));
                Point::new(origin_x + ox, origin_y + oy)
            }
        };
        final_positions.push(pos);
    }
    final_positions
}

// ─────────────────────────────────────────────────────────────────────
// Layered (Sugiyama) layout
// ─────────────────────────────────────────────────────────────────────
//
// Four phases:
//   1. Cycle break — DFS to identify back-edges; reverse them in the
//      working adjacency so the remaining graph is a DAG. Original
//      edge identities are unaffected; this only changes which
//      direction the layering pass treats as "downstream."
//   2. Longest-path layering — `layer[v] = max(layer[u] + 1)` over
//      every incoming edge. Sources (in-degree 0) start at layer 0.
//   3. Crossing reduction — alternate up-sweep + down-sweep median-
//      heuristic passes, reordering each layer so each node sits at
//      the median of its neighbours' ranks in the adjacent layer.
//      Stops after a fixed pass count (`CROSSING_REDUCTION_PASSES`)
//      — exact minimum-crossings is NP-hard; the heuristic converges
//      fast on typical editor-scale DAGs.
//   4. Coordinate assignment — `x = layer * layer_spacing`, `y =
//      rank * in_layer_spacing`, both centred on the layer/rank
//      midpoints so the result is symmetric around origin. Swap x/y
//      at the end for `LayoutOrientation::TopToBottom`.
//
// Group respect mirrors the force layout's hierarchical pattern: when
// the input has any group, run the layered pipeline INSIDE each
// group first (intra-layout, members + intra-group edges only), then
// run it AGAIN on the super-graph (free host nodes + one super-node
// per group, sized by phase-1 footprint). Phase 3 translates each
// member into its group's super-position.

/// Crossing-reduction pass count. Median heuristic typically settles
/// within 4–8 alternating sweeps on small graphs; we run extra to be
/// safe on larger inputs without paying much CPU at editor scale.
const LAYERED_CROSSING_PASSES: usize = 12;

fn apply_layered_layout<N, M, G>(
    config: &LayeredConfig,
    nodes: &[NodeInstance<N>],
    connections: &[Connection<M>],
    groups: &[crate::group::Group<G>],
) -> Vec<Point> {
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![nodes[0].position];
    }

    let mut id_to_index: ahash::AHashMap<crate::node::NodeId, usize> =
        ahash::AHashMap::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        id_to_index.insert(node.id.clone(), i);
    }
    let group_members: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            g.members
                .iter()
                .filter_map(|m| id_to_index.get(m).copied())
                .collect()
        })
        .collect();
    let has_groups = group_members.iter().any(|m| !m.is_empty());

    if has_groups {
        return apply_hierarchical_layered_layout(
            config,
            nodes,
            connections,
            &id_to_index,
            &group_members,
        );
    }

    // Single-level (no groups) — straight Sugiyama.
    let edges: Vec<(usize, usize)> = connections
        .iter()
        .filter_map(|c| {
            let from = id_to_index.get(&c.from.node)?;
            let to = id_to_index.get(&c.to.node)?;
            if from == to {
                None
            } else {
                Some((*from, *to))
            }
        })
        .collect();
    let node_sizes: Vec<(f32, f32)> = nodes
        .iter()
        .map(|n| n.size.unwrap_or((180.0, 72.0)))
        .collect();
    let positions = layered_kernel(n, &edges, &node_sizes, config);
    positions
        .into_iter()
        .map(|(x, y)| Point::new(x, y))
        .collect()
}

/// Core Sugiyama pipeline. Returns positions in the input node order.
/// `node_sizes[i]` = `(width, height)` of node `i` — used to size the
/// per-rank slot so larger nodes get more in-layer room.
fn layered_kernel(
    n: usize,
    edges: &[(usize, usize)],
    node_sizes: &[(f32, f32)],
    config: &LayeredConfig,
) -> Vec<(f32, f32)> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(0.0, 0.0)];
    }

    // ── Phase 1: cycle break ─────────────────────────────────────────
    //
    // DFS the graph; any edge from `u` to `v` where `v` is on the
    // current DFS stack is a back-edge — record it so the layering
    // step treats it as reversed. Doesn't actually mutate the input
    // edge list; just produces a side-table of "edges to flip
    // direction on for the layering pass."
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        adj[u].push(v);
    }
    let mut back_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut color: Vec<u8> = vec![0; n]; // 0 = white, 1 = gray (on stack), 2 = black
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (node, next_neighbour_index)
    for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        stack.push((start, 0));
        while let Some(&(u, k)) = stack.last() {
            if k < adj[u].len() {
                let v = adj[u][k];
                stack.last_mut().unwrap().1 += 1;
                if color[v] == 1 {
                    back_edges.insert((u, v));
                } else if color[v] == 0 {
                    color[v] = 1;
                    stack.push((v, 0));
                }
            } else {
                color[u] = 2;
                stack.pop();
            }
        }
    }

    // Working edge list with back-edges reversed.
    let mut work_edges: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(u, v)| if back_edges.contains(&(u, v)) { (v, u) } else { (u, v) })
        .collect();
    work_edges.sort_unstable();
    work_edges.dedup();

    // ── Phase 2: longest-path layering ───────────────────────────────
    //
    // `layer[v] = max(layer[u] + 1)` over every incoming edge.
    // Computed via topological order (Kahn's algorithm on
    // in-degrees). Disconnected components each start at layer 0
    // from their own source.
    let mut in_deg: Vec<usize> = vec![0; n];
    let mut adj_fwd: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &work_edges {
        adj_fwd[u].push(v);
        in_deg[v] += 1;
    }
    let mut layer: Vec<i32> = vec![0; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if in_deg[i] == 0 {
            queue.push_back(i);
        }
    }
    while let Some(u) = queue.pop_front() {
        let lu = layer[u];
        for &v in &adj_fwd[u] {
            if layer[v] < lu + 1 {
                layer[v] = lu + 1;
            }
            in_deg[v] -= 1;
            if in_deg[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    let max_layer = layer.iter().copied().max().unwrap_or(0) as usize;
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_layer + 1];
    for i in 0..n {
        layers[layer[i] as usize].push(i);
    }

    // ── Phase 3: crossing reduction (median heuristic) ───────────────
    //
    // Repeatedly resort each layer by the MEDIAN of each node's
    // neighbours in the adjacent layer. Alternates down-sweeps
    // (median of in-edges from layer above) and up-sweeps (median of
    // out-edges to layer below) so each pass aligns the two
    // adjacent boundaries.
    let mut adj_in: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut adj_out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &work_edges {
        adj_out[u].push(v);
        adj_in[v].push(u);
    }
    for _ in 0..LAYERED_CROSSING_PASSES {
        // Down-sweep — order each layer by median of incoming
        // neighbours' ranks in the previous layer.
        for li in 1..layers.len() {
            let prev_ranks: ahash::AHashMap<usize, usize> = layers[li - 1]
                .iter()
                .enumerate()
                .map(|(rank, &node)| (node, rank))
                .collect();
            let current = layers[li].clone();
            let mut with_keys: Vec<(usize, f32)> = current
                .iter()
                .map(|&v| {
                    let mut ranks: Vec<usize> = adj_in[v]
                        .iter()
                        .filter_map(|u| prev_ranks.get(u).copied())
                        .collect();
                    ranks.sort_unstable();
                    let key = median_key(&ranks);
                    (v, key)
                })
                .collect();
            sort_by_key_stable(&mut with_keys);
            layers[li] = with_keys.into_iter().map(|(v, _)| v).collect();
        }
        // Up-sweep — order each layer by median of outgoing
        // neighbours' ranks in the next layer.
        for li in (0..layers.len().saturating_sub(1)).rev() {
            let next_ranks: ahash::AHashMap<usize, usize> = layers[li + 1]
                .iter()
                .enumerate()
                .map(|(rank, &node)| (node, rank))
                .collect();
            let current = layers[li].clone();
            let mut with_keys: Vec<(usize, f32)> = current
                .iter()
                .map(|&v| {
                    let mut ranks: Vec<usize> = adj_out[v]
                        .iter()
                        .filter_map(|w| next_ranks.get(w).copied())
                        .collect();
                    ranks.sort_unstable();
                    let key = median_key(&ranks);
                    (v, key)
                })
                .collect();
            sort_by_key_stable(&mut with_keys);
            layers[li] = with_keys.into_iter().map(|(v, _)| v).collect();
        }
    }

    // ── Phase 4: coordinate assignment ───────────────────────────────
    //
    // x = sum of half-widths of every prior layer + every prior
    //     layer's `layer_spacing` gap + this layer's half-width.
    // Without per-layer width awareness, a wide super-node (a group
    // with N inner nodes inlined as a single layer slot) would
    // overlap its neighbour layers — fixed `layer_spacing` isn't
    // enough when one layer is 660 px wide. The cumulative-half-
    // widths approach gives every layer the room it actually needs.
    //
    // y = cumulative node perpendicular extents within the layer +
    //     per-node padding, centred on y=0 so the result is
    //     symmetric around origin (easier for "zoom to fit" math +
    //     the editor's viewport tween).
    //
    // For TopToBottom we lay out the same way then swap x/y at emit.
    let layer_max_para: Vec<f32> = layers
        .iter()
        .map(|members| {
            members
                .iter()
                .map(|&i| {
                    let (w, h) = node_sizes[i];
                    match config.orientation {
                        LayoutOrientation::LeftToRight => w,
                        LayoutOrientation::TopToBottom => h,
                    }
                })
                .fold(0.0_f32, f32::max)
        })
        .collect();
    // x of each layer's centre, accumulated left-to-right.
    let mut layer_x: Vec<f32> = Vec::with_capacity(layers.len());
    let mut cursor_x = 0.0_f32;
    for (li, w) in layer_max_para.iter().enumerate() {
        if li == 0 {
            cursor_x = w * 0.5;
        } else {
            cursor_x += layer_max_para[li - 1] * 0.5 + config.layer_spacing + w * 0.5;
        }
        layer_x.push(cursor_x);
    }
    // Centre the whole grid horizontally so origin is the midpoint
    // of the layout's bounding box.
    let total_w = *layer_x.last().unwrap_or(&0.0)
        + layer_max_para.last().copied().unwrap_or(0.0) * 0.5
        - (layer_max_para.first().copied().unwrap_or(0.0) * 0.5);
    let x_shift = -total_w * 0.5
        - (layer_x.first().copied().unwrap_or(0.0)
            - layer_max_para.first().copied().unwrap_or(0.0) * 0.5);

    let mut positions: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    for (li, layer_members) in layers.iter().enumerate() {
        let x = layer_x[li] + x_shift;
        let perp_dims: Vec<f32> = layer_members
            .iter()
            .map(|&i| {
                let (w, h) = node_sizes[i];
                match config.orientation {
                    LayoutOrientation::LeftToRight => h,
                    LayoutOrientation::TopToBottom => w,
                }
            })
            .collect();
        let total_height: f32 = perp_dims.iter().sum::<f32>()
            + (perp_dims.len().saturating_sub(1)) as f32 * config.in_layer_spacing;
        let mut cursor = -total_height * 0.5;
        for (k, &node_idx) in layer_members.iter().enumerate() {
            let half = perp_dims[k] * 0.5;
            let y = cursor + half;
            positions[node_idx] = (x, y);
            cursor += perp_dims[k] + config.in_layer_spacing;
        }
    }

    // ── Phase 5: orientation transform ───────────────────────────────
    if matches!(config.orientation, LayoutOrientation::TopToBottom) {
        for p in positions.iter_mut() {
            std::mem::swap(&mut p.0, &mut p.1);
        }
    }

    positions
}

/// Median rank for the crossing-reduction key. Per Eades + Wormald:
/// odd count → centre element; even count → weighted average of the
/// two centre elements biased toward the side with more outside
/// neighbours. Empty list (no neighbours) returns 0.0 so nodes
/// without any adjacent-layer connections stay where they were.
fn median_key(ranks: &[usize]) -> f32 {
    let n = ranks.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    if n & 1 == 1 {
        return ranks[mid] as f32;
    }
    // Even: weighted between two centres.
    let left = ranks[mid - 1] as f32;
    let right = ranks[mid] as f32;
    let left_span = left - ranks[0] as f32;
    let right_span = ranks[n - 1] as f32 - right;
    if (left_span + right_span).abs() < KERNEL_EPSILON {
        return (left + right) * 0.5;
    }
    (left * right_span + right * left_span) / (left_span + right_span)
}

/// Stable sort by the f32 key alone — preserves the existing
/// relative order on ties so a node with no adjacent-layer
/// neighbours doesn't get shuffled randomly.
fn sort_by_key_stable(vec: &mut [(usize, f32)]) {
    vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
}

/// Three-phase hierarchical wrapper for the Layered pipeline,
/// mirroring `apply_hierarchical_force_layout` exactly. Members are
/// laid out inside their group (Sugiyama on intra-group edges),
/// groups become super-nodes in a top-level Sugiyama pass, then
/// member positions are translated into their group's super-position.
fn apply_hierarchical_layered_layout<N, M>(
    config: &LayeredConfig,
    nodes: &[NodeInstance<N>],
    connections: &[Connection<M>],
    id_to_index: &ahash::AHashMap<crate::node::NodeId, usize>,
    group_members: &[Vec<usize>],
) -> Vec<Point> {
    let n = nodes.len();
    let mut primary_group: Vec<Option<usize>> = vec![None; n];
    for (gi, members) in group_members.iter().enumerate() {
        for &m in members {
            if primary_group[m].is_none() {
                primary_group[m] = Some(gi);
            }
        }
    }

    let node_sizes: Vec<(f32, f32)> = nodes
        .iter()
        .map(|n| n.size.unwrap_or((180.0, 72.0)))
        .collect();

    // ── Phase 1: intra-group layouts ─────────────────────────────────
    let mut group_local: Vec<ahash::AHashMap<usize, (f32, f32)>> =
        vec![ahash::AHashMap::default(); group_members.len()];
    let mut group_size: Vec<(f32, f32)> = vec![(0.0, 0.0); group_members.len()];

    for (gi, members) in group_members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let mut local_index: ahash::AHashMap<usize, usize> = ahash::AHashMap::default();
        for (li, &m) in members.iter().enumerate() {
            local_index.insert(m, li);
        }
        let intra_edges: Vec<(usize, usize)> = connections
            .iter()
            .filter_map(|c| {
                let from = *id_to_index.get(&c.from.node)?;
                let to = *id_to_index.get(&c.to.node)?;
                let lf = *local_index.get(&from)?;
                let lt = *local_index.get(&to)?;
                if lf == lt {
                    None
                } else {
                    Some((lf, lt))
                }
            })
            .collect();
        let local_sizes: Vec<(f32, f32)> = members.iter().map(|&m| node_sizes[m]).collect();
        let local_positions = layered_kernel(members.len(), &intra_edges, &local_sizes, config);

        // Translate so the bbox top-left sits at origin; record
        // (w, h) for phase 2's super-node footprint. Includes node
        // size in the bbox extent.
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (li, &(x, y)) in local_positions.iter().enumerate() {
            let (w, h) = local_sizes[li];
            min_x = min_x.min(x - w * 0.5);
            min_y = min_y.min(y - h * 0.5);
            max_x = max_x.max(x + w * 0.5);
            max_y = max_y.max(y + h * 0.5);
        }
        group_size[gi] = (max_x - min_x, max_y - min_y);
        for (li, &m) in members.iter().enumerate() {
            let (x, y) = local_positions[li];
            group_local[gi].insert(m, (x - min_x, y - min_y));
        }
    }

    // ── Phase 2: super-graph layered layout ──────────────────────────
    let free_indices: Vec<usize> = (0..n).filter(|i| primary_group[*i].is_none()).collect();
    let free_count = free_indices.len();
    let group_count = group_members.len();
    let super_n = free_count + group_count;

    let mut host_to_super: Vec<usize> = vec![usize::MAX; n];
    for (super_idx, &host_idx) in free_indices.iter().enumerate() {
        host_to_super[host_idx] = super_idx;
    }
    for (gi, members) in group_members.iter().enumerate() {
        let super_idx = free_count + gi;
        for &m in members {
            host_to_super[m] = super_idx;
        }
    }

    let mut super_sizes: Vec<(f32, f32)> = Vec::with_capacity(super_n);
    for &host_idx in &free_indices {
        super_sizes.push(node_sizes[host_idx]);
    }
    for (gi, _) in group_members.iter().enumerate() {
        super_sizes.push(group_size[gi]);
    }

    let mut super_edges_set: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for c in connections {
        let from = match id_to_index.get(&c.from.node) {
            Some(i) => *i,
            None => continue,
        };
        let to = match id_to_index.get(&c.to.node) {
            Some(i) => *i,
            None => continue,
        };
        let sf = host_to_super[from];
        let st = host_to_super[to];
        if sf == st || sf == usize::MAX || st == usize::MAX {
            continue;
        }
        super_edges_set.insert((sf, st));
    }
    let super_edges: Vec<(usize, usize)> = super_edges_set.into_iter().collect();

    let super_positions = layered_kernel(super_n, &super_edges, &super_sizes, config);

    // ── Phase 3: assemble ────────────────────────────────────────────
    let mut final_positions: Vec<Point> = Vec::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        let super_idx = host_to_super[i];
        let (sx, sy) = if super_idx != usize::MAX {
            super_positions[super_idx]
        } else {
            (node.position.x, node.position.y)
        };
        let pos = match primary_group[i] {
            None => Point::new(sx, sy),
            Some(gi) => {
                let (w, h) = group_size[gi];
                let origin_x = sx - w * 0.5;
                let origin_y = sy - h * 0.5;
                let (ox, oy) = group_local[gi].get(&i).copied().unwrap_or((0.0, 0.0));
                Point::new(origin_x + ox, origin_y + oy)
            }
        };
        final_positions.push(pos);
    }
    final_positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::Connection;
    use crate::node::NodeInstance;
    use crate::port::PortAddress;

    fn node(id: &str, x: f32, y: f32) -> NodeInstance<()> {
        NodeInstance::<()>::new(id, "test", Point::new(x, y))
    }
    fn edge(from: &str, to: &str) -> Connection<()> {
        Connection::<()>::new(
            PortAddress::new(from.into(), "out"),
            PortAddress::new(to.into(), "in"),
        )
    }

    #[test]
    fn force_layout_empty_input_returns_empty() {
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(ForceConfig::default()),
            &[],
            &[],
            &[],
        );
        assert!(positions.is_empty());
    }

    #[test]
    fn force_layout_single_node_returns_input_position() {
        let nodes = vec![node("a", 100.0, 50.0)];
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(ForceConfig::default()),
            &nodes,
            &[],
            &[],
        );
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].x, 100.0);
        assert_eq!(positions[0].y, 50.0);
    }

    #[test]
    fn force_layout_two_overlapping_nodes_separate() {
        // Two nodes at the same exact position must end up apart.
        let nodes = vec![node("a", 0.0, 0.0), node("b", 0.0, 0.0)];
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(ForceConfig::default()),
            &nodes,
            &[],
            &[],
        );
        let dx = positions[0].x - positions[1].x;
        let dy = positions[0].y - positions[1].y;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist > 1.0, "overlapping nodes did not separate; dist = {dist}");
    }

    #[test]
    fn force_layout_edge_pulls_pair_toward_ideal_length() {
        // Two nodes connected by an edge, started very far apart,
        // should converge toward ~ideal_edge_length of each other.
        let nodes = vec![node("a", -2000.0, 0.0), node("b", 2000.0, 0.0)];
        let conns = vec![edge("a", "b")];
        let cfg = ForceConfig::default();
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(cfg.clone()),
            &nodes,
            &conns,
            &[],
        );
        let dx = positions[1].x - positions[0].x;
        let dy = positions[1].y - positions[0].y;
        let dist = (dx * dx + dy * dy).sqrt();
        // Loose bound — exact convergence depends on damping etc.,
        // but the pair should be much closer than its 4000 px start
        // and within the same order of magnitude as ideal_edge_length.
        assert!(
            dist < 1000.0,
            "edge did not pull nodes together; dist = {dist}, ideal = {}",
            cfg.ideal_edge_length
        );
        assert!(
            dist > 10.0,
            "edge pulled nodes too close (no balancing repulsion); dist = {dist}"
        );
    }

    #[test]
    fn force_layout_is_deterministic() {
        // Same input → identical output, no RNG sneaking in.
        let nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 200.0, 0.0),
            node("c", 0.0, 200.0),
            node("d", 200.0, 200.0),
        ];
        let conns = vec![edge("a", "b"), edge("b", "d"), edge("a", "c")];
        let cfg = ForceConfig::default();
        let r1 = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(cfg.clone()),
            &nodes,
            &conns,
            &[],
        );
        let r2 = apply_layout::<(), (), ()>(&LayoutStrategy::Force(cfg), &nodes, &conns, &[]);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
        }
    }

    #[test]
    fn force_layout_keeps_members_inside_and_nonmembers_outside_group_bbox() {
        // Three nodes in a group, one non-member starting CLOSE to
        // the group's pre-layout centroid. Hierarchical layout must
        // keep members inside the group's post-layout bbox and the
        // non-member outside it.
        let nodes = vec![
            node("a", -800.0, 0.0),
            node("b", 0.0, -800.0),
            node("c", 800.0, 0.0),
            // Non-member positioned at the group's initial centroid
            // (origin) — soft springs would have a hard time
            // evicting it. Hierarchical layout treats it as a free
            // node in the super-graph, so it sits in its own super-
            // position outside the group's super-bbox.
            node("outside", 0.0, 0.0),
        ];
        let group = crate::group::Group::<()>::new(
            crate::group::GroupId::from("g1"),
            "Group 1",
        )
        .add_member(crate::node::NodeId::from("a"))
        .add_member(crate::node::NodeId::from("b"))
        .add_member(crate::node::NodeId::from("c"));
        let groups = vec![group];

        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(ForceConfig::default()),
            &nodes,
            &[],
            &groups,
        );

        // Compute member bbox (with each node's half-extent baked
        // in — same convention as ExpansionBaseline uses).
        const NODE_HALF: f32 = 90.0; // default (180, 72) ⇒ max half = 90
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for i in 0..3 {
            min_x = min_x.min(positions[i].x - NODE_HALF);
            min_y = min_y.min(positions[i].y - NODE_HALF);
            max_x = max_x.max(positions[i].x + NODE_HALF);
            max_y = max_y.max(positions[i].y + NODE_HALF);
        }
        // Members inside their own bbox is trivially true; the
        // interesting assertion is that the non-member sits
        // OUTSIDE.
        let outside = positions[3];
        let inside_bbox =
            outside.x >= min_x && outside.x <= max_x && outside.y >= min_y && outside.y <= max_y;
        assert!(
            !inside_bbox,
            "non-member landed INSIDE the group bbox: outside = ({}, {}), \
             bbox = ({min_x}..{max_x}) × ({min_y}..{max_y})",
            outside.x, outside.y
        );
    }

    #[test]
    fn force_layout_group_without_intra_edges_converges_on_repeat() {
        // Members of a group with NO intra-group edges must NOT
        // spread further on each successive layout call. Pure
        // repulsion with no balancing spring would drift them
        // apart every iteration so clicking auto-layout twice
        // would double the gap; the phase-1 centroid pull
        // provides the missing rest length so the spread
        // stabilises.
        let mut nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 200.0, 0.0),
        ];
        let group = crate::group::Group::<()>::new(
            crate::group::GroupId::from("g1"),
            "Group 1",
        )
        .add_member(crate::node::NodeId::from("a"))
        .add_member(crate::node::NodeId::from("b"));
        let groups = vec![group];
        let cfg = ForceConfig::default();

        // First run.
        let positions1 = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(cfg.clone()),
            &nodes,
            &[],
            &groups,
        );
        let dist1 = {
            let dx = positions1[1].x - positions1[0].x;
            let dy = positions1[1].y - positions1[0].y;
            (dx * dx + dy * dy).sqrt()
        };

        // Re-seed the nodes from the first run's output, then run
        // layout AGAIN — same input → idempotent layout.
        for (n, p) in nodes.iter_mut().zip(positions1.iter()) {
            n.position = *p;
        }
        let positions2 = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(cfg),
            &nodes,
            &[],
            &groups,
        );
        let dist2 = {
            let dx = positions2[1].x - positions2[0].x;
            let dy = positions2[1].y - positions2[0].y;
            (dx * dx + dy * dy).sqrt()
        };

        // The second run must produce a distance within ~10% of
        // the first. Without the centroid pull dist2 would be
        // roughly 2× dist1, so the gap would compound visibly on
        // every successive layout call.
        let ratio = dist2 / dist1.max(1.0);
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "intra-group distance drifted on repeat layout: dist1 = {dist1}, \
             dist2 = {dist2}, ratio = {ratio}"
        );
    }

    // ─── Layered (Sugiyama) ──────────────────────────────────────────

    #[test]
    fn layered_layout_empty_input_returns_empty() {
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &[],
            &[],
            &[],
        );
        assert!(positions.is_empty());
    }

    #[test]
    fn layered_layout_linear_chain_lands_in_order() {
        // a → b → c → d should produce 4 increasing x-coordinates
        // (LTR default) with strictly monotonic ordering.
        let nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 0.0, 0.0),
            node("c", 0.0, 0.0),
            node("d", 0.0, 0.0),
        ];
        let conns = vec![edge("a", "b"), edge("b", "c"), edge("c", "d")];
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &nodes,
            &conns,
            &[],
        );
        // Each successive node sits at one layer_spacing PLUS the
        // adjacent node half-widths. For default 180×72 nodes:
        // 180/2 + 240 + 180/2 = 420 px between centres.
        for w in positions.windows(2) {
            let dx = w[1].x - w[0].x;
            assert!(
                (dx - 420.0).abs() < 1.0,
                "expected width-aware dx ≈ 420 (180/2 + 240 + 180/2), got {dx}"
            );
        }
    }

    #[test]
    fn layered_layout_top_to_bottom_swaps_axes() {
        // The same chain in TopToBottom mode should produce
        // increasing y-coordinates, not x.
        let nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 0.0, 0.0),
            node("c", 0.0, 0.0),
        ];
        let conns = vec![edge("a", "b"), edge("b", "c")];
        let config = LayeredConfig {
            orientation: LayoutOrientation::TopToBottom,
            ..LayeredConfig::default()
        };
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(config),
            &nodes,
            &conns,
            &[],
        );
        // In TopToBottom the perpendicular axis becomes height
        // (72px default), so per-layer dy = 72/2 + 240 + 72/2 = 312.
        for w in positions.windows(2) {
            let dy = w[1].y - w[0].y;
            assert!(
                (dy - 312.0).abs() < 1.0,
                "expected width-aware dy ≈ 312 (72/2 + 240 + 72/2) in TopToBottom, got {dy}"
            );
            // x should be unchanging across a linear chain in TB.
            assert!(
                (w[0].x - w[1].x).abs() < 1.0,
                "expected aligned x in TopToBottom chain, got dx = {}",
                w[1].x - w[0].x
            );
        }
    }

    #[test]
    fn layered_layout_handles_cycle_via_back_edge_reversal() {
        // a → b → c → a is a cycle. Cycle break reverses a back-
        // edge so the layering pass runs on a DAG. Doesn't matter
        // exactly which positions come out — just that they're
        // finite and the algorithm doesn't loop forever.
        let nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 0.0, 0.0),
            node("c", 0.0, 0.0),
        ];
        let conns = vec![edge("a", "b"), edge("b", "c"), edge("c", "a")];
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &nodes,
            &conns,
            &[],
        );
        assert_eq!(positions.len(), 3);
        for p in &positions {
            assert!(p.x.is_finite(), "x not finite: {}", p.x);
            assert!(p.y.is_finite(), "y not finite: {}", p.y);
        }
    }

    #[test]
    fn layered_layout_is_deterministic() {
        let nodes = vec![
            node("a", 10.0, 20.0),
            node("b", 30.0, 40.0),
            node("c", 50.0, 60.0),
            node("d", 70.0, 80.0),
        ];
        let conns = vec![edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d")];
        let r1 = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &nodes,
            &conns,
            &[],
        );
        let r2 = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &nodes,
            &conns,
            &[],
        );
        assert_eq!(r1, r2);
    }

    #[test]
    fn layered_layout_respects_group_bbox() {
        // Three nodes in a group + a free node. The free node must
        // not land inside the group's super-node footprint. Edges
        // make a, b, c a chain inside the group; d is connected to
        // a from outside.
        let nodes = vec![
            node("a", 0.0, 0.0),
            node("b", 0.0, 0.0),
            node("c", 0.0, 0.0),
            node("d", 0.0, 0.0),
        ];
        let conns = vec![edge("a", "b"), edge("b", "c"), edge("d", "a")];
        let group = crate::group::Group::<()>::new(
            crate::group::GroupId::from("g1"),
            "Group 1",
        )
        .add_member(crate::node::NodeId::from("a"))
        .add_member(crate::node::NodeId::from("b"))
        .add_member(crate::node::NodeId::from("c"));

        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Layered(LayeredConfig::default()),
            &nodes,
            &conns,
            &[group],
        );
        const NODE_HALF_W: f32 = 90.0;
        const NODE_HALF_H: f32 = 36.0;
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for i in 0..3 {
            min_x = min_x.min(positions[i].x - NODE_HALF_W);
            min_y = min_y.min(positions[i].y - NODE_HALF_H);
            max_x = max_x.max(positions[i].x + NODE_HALF_W);
            max_y = max_y.max(positions[i].y + NODE_HALF_H);
        }
        let d = positions[3];
        let inside = d.x >= min_x && d.x <= max_x && d.y >= min_y && d.y <= max_y;
        assert!(
            !inside,
            "free node landed INSIDE the group bbox: d = ({}, {}), bbox = \
             ({min_x}..{max_x}) × ({min_y}..{max_y})",
            d.x, d.y
        );
    }

    #[test]
    fn force_layout_ignores_self_loops() {
        // A self-loop (u == v) must not produce NaN forces.
        let nodes = vec![node("a", 0.0, 0.0), node("b", 50.0, 0.0)];
        let conns = vec![edge("a", "a"), edge("a", "b")];
        let positions = apply_layout::<(), (), ()>(
            &LayoutStrategy::Force(ForceConfig::default()),
            &nodes,
            &conns,
            &[],
        );
        for p in &positions {
            assert!(p.x.is_finite(), "x = {} is not finite", p.x);
            assert!(p.y.is_finite(), "y = {} is not finite", p.y);
        }
    }
}
