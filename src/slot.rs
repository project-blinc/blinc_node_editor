//! Slot layout — off-render measurement of node + group interior chrome
//! via `blinc_layout::LayoutTree`, cached by structural fingerprint, warmed
//! in parallel via rayon.
//!
//! ## Why this exists
//!
//! `render.rs` used to hard-code interior geometry: header height as
//! `bounds.height() * 0.34`, title at `header.x() + 10.0`, badge at
//! `bounds.x() + bounds.width() - 14.0`. That works for one shape; the
//! moment a node wants a different chrome layout (icon column, footer
//! strip, denser badge bar) the math breaks.
//!
//! Instead, each unique node / group "shape" is composed as a small
//! `LayoutTree`, measured once, and the resolved per-slot rects are
//! cached. The cache is keyed by a structural fingerprint
//! ([`NodeFingerprint`]) — two instances of the same template with the
//! same size + port counts + badge presence share one cache entry.
//!
//! ## Compute budget
//!
//! Slot tables are computed exactly when the fingerprint set changes
//! ([`set_graph`](crate::NodeEditor::set_graph) calls
//! [`warm_slot_cache`]). The rayon work-pool maps the uncached set in
//! parallel; one [`LayoutTree`] per worker, no shared state. Below
//! [`PARALLEL_THRESHOLD`] the warm path falls back to serial — rayon's
//! setup overhead otherwise dominates for small fingerprint sets.
//!
//! ## Port anchors
//!
//! Ports live on the OUTER edge of the node and are NOT part of the
//! interior layout tree. Their positions come from an even-spacing
//! formula keyed off the per-side port count + node bounds. This keeps
//! layout-tree size small and port placement intrinsically scalable.

use ahash::AHashMap;
use blinc_core::layer::{Point, Rect};
use blinc_layout::tree::{LayoutNodeId, LayoutTree};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use taffy::prelude::{AvailableSpace, Style};
use taffy::{
    Dimension, FlexDirection, LengthPercentage, LengthPercentageAuto, Rect as TaffyRect,
    Size as TaffySize,
};

use crate::group::Group;
use crate::node::{NodeInstance, NodeTemplate};
use crate::port::PortKind;
use crate::theme::ThemeResolver;

/// Above this many uncached fingerprints, [`warm_slot_cache`] uses
/// `par_iter` instead of a serial sweep. Tuned for typical
/// rayon setup overhead at ~32 work-items.
pub const PARALLEL_THRESHOLD: usize = 32;

// ─────────────────────────────────────────────────────────────────────
// Slot tables — what the renderer consumes
// ─────────────────────────────────────────────────────────────────────

/// Per-node interior layout. All rects are in NODE-LOCAL coordinates
/// (origin = top-left of the node body). The renderer translates by
/// `node.position` when painting.
#[derive(Debug, Clone)]
pub struct NodeSlots {
    pub header: Rect,
    /// Optional icon slot — square rect in the header, left of the
    /// title column. Present iff the fingerprint included an icon.
    pub icon: Option<Rect>,
    pub title: Rect,
    pub subtitle: Option<Rect>,
    pub badge: Option<Rect>,
    pub body: Rect,
    /// Total node height as computed by taffy from the slot tree.
    /// `fit-content`: header + body, where header sizes from its
    /// children (icon / title / subtitle) + padding, and body
    /// collapses to 0 when empty. This is the renderer's source of
    /// truth for node height — `render::node_bounds` reads it
    /// rather than re-deriving with hand-math.
    pub total_height: f32,
    /// Total node width — equal to `NodeSlotInputs.width` resolved
    /// from `instance.size`, theme default, and any
    /// `template.content.min_width` floor. Renderer reads this
    /// instead of recomputing from `instance.size` so the
    /// `content.min_width` widening (port placement, body width,
    /// outer rect) is consistent everywhere.
    pub total_width: f32,
}

/// Per-group chrome layout. Same node-local convention.
#[derive(Debug, Clone)]
pub struct GroupSlots {
    pub header: Rect,
    pub title: Rect,
    pub description: Option<Rect>,
    pub badge: Option<Rect>,
    /// Square slot at the FAR right of the header reserved for the
    /// collapse / expand chevron. The renderer paints the chevron
    /// here + registers it as the `group_collapse:{id}` hit region.
    pub chrome_collapse: Rect,
    /// Square slot to the LEFT of `chrome_collapse` reserved for the
    /// delete (×) button. Paints + hits as `group_delete:{id}`.
    pub chrome_delete: Rect,
    /// Square slot to the LEFT of `chrome_delete` reserved for the
    /// edit-title pencil button. Paints + hits as `group_edit:{id}`.
    pub chrome_edit: Rect,
}

// ─────────────────────────────────────────────────────────────────────
// Fingerprints
// ─────────────────────────────────────────────────────────────────────

/// Structural fingerprint of a node — two instances with the same
/// fingerprint share a slot-table cache entry. Position never
/// participates; dragging a node never invalidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeFingerprint(pub u64);

/// Structural fingerprint of a group's header chrome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupFingerprint(pub u64);

/// Inputs to [`compute_node_slots`] — captured by [`fingerprint_node`]
/// so the parallel warm loop can re-derive the same `NodeSlots`
/// without holding the read lock on the graph.
#[derive(Debug, Clone)]
pub struct NodeSlotInputs {
    pub width: f32,
    pub has_subtitle: bool,
    pub has_badge: bool,
    pub has_icon: bool,
    pub icon_size: f32,
    pub title_font_size: f32,
    pub subtitle_font_size: f32,
    pub content_padding: f32,
    /// Body region height. Zero when the template has no content
    /// slot — body collapses and total height = header height.
    /// Non-zero when a [`crate::NodeContent`] is attached: the body
    /// rect grows by this much and a portal paints inside it.
    pub content_height: f32,
}

#[derive(Debug, Clone)]
pub struct GroupSlotInputs {
    pub width: f32,
    pub header_height: f32,
    pub has_description: bool,
    pub has_badge: bool,
    pub title_font_size: f32,
    pub subtitle_font_size: f32,
    pub content_padding: f32,
    /// Pre-computed line count for the description after greedy
    /// word-wrap at the available width. Drives the description
    /// slot's `height = description_lines * description_line_height`
    /// so a long description after an inline edit reflows the
    /// header instead of overflowing into the body. Always `>= 1`;
    /// `1` when `has_description = false` so the cached slot still
    /// hashes consistently for the no-description case.
    pub description_lines: u32,
}

pub fn fingerprint_node<K: PortKind, M>(
    template: &NodeTemplate<K>,
    instance: &NodeInstance<M>,
    inputs: &NodeSlotInputs,
    theme_revision: u64,
) -> NodeFingerprint {
    use std::hash::{Hash, Hasher};
    let mut h = ahash::AHasher::default();
    template.component.hash(&mut h);
    (inputs.width.to_bits()).hash(&mut h);
    inputs.has_subtitle.hash(&mut h);
    inputs.has_badge.hash(&mut h);
    inputs.has_icon.hash(&mut h);
    (inputs.icon_size.to_bits()).hash(&mut h);
    (inputs.title_font_size.to_bits()).hash(&mut h);
    (inputs.subtitle_font_size.to_bits()).hash(&mut h);
    (inputs.content_padding.to_bits()).hash(&mut h);
    (inputs.content_height.to_bits()).hash(&mut h);
    template.inputs.len().hash(&mut h);
    template.outputs.len().hash(&mut h);
    // Per-instance shape override leaks into fingerprint so different
    // shapes don't collide.
    format!("{:?}", instance.shape).hash(&mut h);
    theme_revision.hash(&mut h);
    NodeFingerprint(h.finish())
}

pub fn fingerprint_group<G>(
    group: &Group<G>,
    inputs: &GroupSlotInputs,
    theme_revision: u64,
) -> GroupFingerprint {
    use std::hash::{Hash, Hasher};
    let mut h = ahash::AHasher::default();
    (inputs.width.to_bits()).hash(&mut h);
    (inputs.header_height.to_bits()).hash(&mut h);
    inputs.has_description.hash(&mut h);
    inputs.has_badge.hash(&mut h);
    (inputs.title_font_size.to_bits()).hash(&mut h);
    inputs.description_lines.hash(&mut h);
    group.is_collapsed.hash(&mut h);
    theme_revision.hash(&mut h);
    GroupFingerprint(h.finish())
}

/// Collect the per-template / per-theme inputs the slot compute needs.
/// Pulls font sizes + padding from the resolver so theme changes
/// invalidate cleanly via `theme_revision`. Reads the template to
/// detect `has_icon` (instance icon overrides template icon, but
/// either presence means the slot should be allocated).
pub fn node_inputs_from<K: PortKind, M>(
    template: &NodeTemplate<K>,
    instance: &NodeInstance<M>,
    theme: &ThemeResolver<'_>,
) -> NodeSlotInputs {
    let (def_w, _) = theme.default_node_size();
    // Width resolution: explicit `NodeInstance.size.0` wins; else
    // theme default, but raised to the template's declared
    // `content.min_width` floor if present so portal-ui content
    // (immediate-mode, opaque to taffy) doesn't clip when the
    // template knows it needs more room.
    let content_min_w = template
        .content
        .as_ref()
        .and_then(|c| c.min_width)
        .unwrap_or(0.0);
    let w = instance
        .size
        .map(|(w, _)| w)
        .unwrap_or_else(|| def_w.max(content_min_w));
    NodeSlotInputs {
        width: w,
        has_subtitle: instance.subtitle.is_some(),
        has_badge: instance.badge.is_some(),
        has_icon: instance.icon.is_some() || template.icon.is_some(),
        icon_size: 36.0,
        title_font_size: theme.title_font_size(),
        subtitle_font_size: theme.subtitle_font_size(),
        content_padding: theme.node_content_padding(),
        content_height: template.content.as_ref().map(|c| c.height).unwrap_or(0.0),
    }
}

pub fn group_inputs_from<G>(
    group: &Group<G>,
    width: f32,
    theme: &ThemeResolver<'_>,
) -> GroupSlotInputs {
    let has_description = group.description.is_some() || group.description_placeholder.is_some();
    // Estimate the description's wrapped line count up front so the
    // description slot's height grows to fit. Uses the same
    // per-class advance estimator + greedy wrap as
    // `render::draw_group`, against the same available width, so
    // layout and paint agree on line count. `available_w` mirrors
    // the runtime constraint: full group width minus the header's
    // left/right padding, the collapse-chrome chip on the right
    // (theme-sized clamp 16-22), the chrome's 8 px gap, the badge
    // chip + gap when present. Falls back to a small min width if
    // the math goes negative on a very narrow group.
    let title_line = theme.title_font_size() + 2.0;
    let chrome_size = title_line.clamp(16.0, 22.0);
    // Chrome row holds THREE chips (edit / delete / collapse) with
    // 8 px gutter between the title column and the first chip, 4 px
    // between subsequent chips — must match `compute_group_slots`'s
    // taffy layout exactly. Subtracting only one chip's worth (the
    // old `chrome_size + chrome_gap` calculation) overestimated the
    // description's available width by ~52 px, causing `wrap_text`
    // to report fewer lines than the runtime layout actually
    // produced and the description text to overflow below the
    // header band. Symptom: "Group created from multi-select"
    // landing on a third line *outside* the header chrome.
    let chrome_row_w: f32 = 3.0 * chrome_size + 8.0 + 4.0 + 4.0;
    let badge_size: f32 = 14.0;
    let badge_gap: f32 = 8.0;
    let pad = theme.node_content_padding();
    let mut available_w = width - 2.0 * pad - chrome_row_w;
    if group.badge.is_some() {
        available_w -= badge_size + badge_gap;
    }
    let available_w = available_w.max(40.0);
    let description_text: &str = group
        .description
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(group
            .description_placeholder
            .as_deref()
            .filter(|s| !s.is_empty()))
        .unwrap_or("");
    let description_lines = if has_description && !description_text.is_empty() {
        crate::render::wrap_text(description_text, theme.subtitle_font_size(), available_w)
            .len()
            .max(1) as u32
    } else {
        1
    };
    GroupSlotInputs {
        width,
        header_height: 28.0,
        has_description,
        has_badge: group.badge.is_some(),
        title_font_size: theme.title_font_size(),
        subtitle_font_size: theme.subtitle_font_size(),
        content_padding: theme.node_content_padding(),
        description_lines,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cache
// ─────────────────────────────────────────────────────────────────────

/// Slot cache. Lives on the editor; cleared on theme changes via the
/// theme-revision bump.
pub type NodeSlotCache = Arc<RwLock<AHashMap<NodeFingerprint, NodeSlots>>>;
pub type GroupSlotCache = Arc<RwLock<AHashMap<GroupFingerprint, GroupSlots>>>;

// ─────────────────────────────────────────────────────────────────────
// Default composition — builds the classic header / body layout tree
// ─────────────────────────────────────────────────────────────────────

/// Build the default node-interior layout tree and return the
/// resolved slot rects after `compute_layout`. This is the
/// built-in composition; templates that want custom shapes will
/// override via `NodeTemplate::with_composition(...)` (a future
/// closure-based hook).
pub fn compute_node_slots(inputs: &NodeSlotInputs) -> NodeSlots {
    let mut tree = LayoutTree::new();

    let pad = inputs.content_padding;
    // 10/10 padding — comfortable breathing room without
    // ballooning. Symmetric so the icon centre lines up with the
    // title block centre when the header collapses to a single row.
    let header_pad_top: f32 = 10.0;
    let header_pad_bottom: f32 = 10.0;
    let title_line = inputs.title_font_size + 2.0;
    let subtitle_line = inputs.subtitle_font_size + 2.0;
    let title_subtitle_gap: f32 = 2.0;

    // Root — flex column, width is fixed by `instance.size.0` or the
    // theme default; height is `Auto` so taffy sizes it as
    // `fit-content` from the header + body children.
    let root = tree.create_node(Style {
        flex_direction: FlexDirection::Column,
        size: TaffySize {
            width: Dimension::Length(inputs.width),
            height: Dimension::Auto,
        },
        ..Default::default()
    });

    // Header row — height is `Auto` so taffy sizes it from its
    // children + padding (`fit-content`). Headers with just a title
    // are one-line tall; headers with a subtitle stretch to two
    // lines. The body row's `flex_grow: 1.0` absorbs the remainder
    // of the node height.
    let header = tree.create_node(Style {
        flex_direction: FlexDirection::Row,
        size: TaffySize {
            width: Dimension::Percent(1.0),
            height: Dimension::Auto,
        },
        flex_shrink: 0.0,
        padding: TaffyRect {
            left: LengthPercentage::Length(pad),
            right: LengthPercentage::Length(pad),
            top: LengthPercentage::Length(header_pad_top),
            bottom: LengthPercentage::Length(header_pad_bottom),
        },
        align_items: Some(taffy::AlignItems::FlexStart),
        ..Default::default()
    });

    // Icon slot — FIXED square chip (Zeal parity: nodes use the
    // same icon size regardless of title / subtitle content).
    // Sits LEFT of the title column with a trailing margin so
    // the title doesn't crowd the glyph. Vertically centred
    // against the icon's own height via FlexStart + the
    // text-block centre offset handled in the renderer.
    let icon_node = if inputs.has_icon {
        let icon_render_size = inputs.icon_size;
        let n = tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Length(icon_render_size),
                height: Dimension::Length(icon_render_size),
            },
            margin: TaffyRect {
                right: LengthPercentageAuto::Length(12.0),
                left: LengthPercentageAuto::Length(0.0),
                top: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        });
        tree.add_child(header, n);
        Some(n)
    } else {
        None
    };

    // Title / subtitle column — fills the header row's free width.
    let title_col = tree.create_node(Style {
        flex_direction: FlexDirection::Column,
        flex_grow: 1.0,
        ..Default::default()
    });

    let title_node = tree.create_node(Style {
        size: TaffySize {
            width: Dimension::Auto,
            height: Dimension::Length(title_line),
        },
        ..Default::default()
    });
    tree.add_child(title_col, title_node);

    let subtitle_node = if inputs.has_subtitle {
        let n = tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Auto,
                height: Dimension::Length(subtitle_line),
            },
            margin: TaffyRect {
                top: LengthPercentageAuto::Length(title_subtitle_gap),
                right: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
                left: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        });
        tree.add_child(title_col, n);
        Some(n)
    } else {
        None
    };

    tree.add_child(header, title_col);

    // Badge slot — fixed size, vertically centred against the title
    // line so it never sits flush with the rounded corner.
    let badge_node = if inputs.has_badge {
        let badge_size: f32 = 14.0;
        let badge_top_offset = ((title_line - badge_size) * 0.5).max(0.0);
        let n = tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Length(badge_size),
                height: Dimension::Length(badge_size),
            },
            margin: TaffyRect {
                left: LengthPercentageAuto::Length(8.0),
                top: LengthPercentageAuto::Length(badge_top_offset),
                right: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        });
        tree.add_child(header, n);
        Some(n)
    } else {
        None
    };

    tree.add_child(root, header);

    // Body — explicit height when a content slot (portal-ui) is
    // attached, `Auto` (== 0 without children) otherwise. Either
    // way, taffy adds it to the root's total height so
    // `node_bounds` / port placement / effective-bounds growth
    // all see the right total.
    let body_height = if inputs.content_height > 0.0 {
        Dimension::Length(inputs.content_height)
    } else {
        Dimension::Auto
    };
    let body = tree.create_node(Style {
        size: TaffySize {
            width: Dimension::Percent(1.0),
            height: body_height,
        },
        ..Default::default()
    });
    tree.add_child(root, body);

    tree.compute_layout(
        root,
        TaffySize {
            width: AvailableSpace::Definite(inputs.width),
            // MaxContent for height so the root grows to fit its
            // children rather than being clamped to an outer
            // viewport height.
            height: AvailableSpace::MaxContent,
        },
    );

    // Read the taffy-computed root height — this is the node's
    // natural `fit-content` height and the renderer uses it
    // verbatim instead of running its own padding maths.
    let root_layout = read_rect(&tree, root, Point::new(0.0, 0.0));

    NodeSlots {
        header: read_rect(&tree, header, Point::new(0.0, 0.0)),
        icon: icon_node.map(|n| read_rect_within(&tree, n, &[root, header])),
        title: read_rect_within(&tree, title_node, &[root, header, title_col]),
        subtitle: subtitle_node.map(|n| read_rect_within(&tree, n, &[root, header, title_col])),
        badge: badge_node.map(|n| read_rect_within(&tree, n, &[root, header])),
        body: read_rect(&tree, body, Point::new(0.0, 0.0)),
        total_height: root_layout.height(),
        total_width: inputs.width,
    }
}

/// Build the default group-header composition and return resolved
/// slot rects.
///
/// Mirrors the node-header approach: height is `Auto` so taffy
/// sizes the band from its children + padding, instead of clamping
/// to a fixed input value that would push the description out of
/// the band when both title + description are present.
pub fn compute_group_slots(inputs: &GroupSlotInputs) -> GroupSlots {
    let mut tree = LayoutTree::new();
    let pad = inputs.content_padding;
    let header_pad_top: f32 = 6.0;
    let header_pad_bottom: f32 = 6.0;
    let title_line = inputs.title_font_size + 2.0;
    let description_line = inputs.subtitle_font_size + 2.0;
    let title_description_gap: f32 = 2.0;

    let header = tree.create_node(Style {
        flex_direction: FlexDirection::Row,
        size: TaffySize {
            width: Dimension::Length(inputs.width),
            height: Dimension::Auto,
        },
        padding: TaffyRect {
            left: LengthPercentage::Length(pad),
            right: LengthPercentage::Length(pad),
            top: LengthPercentage::Length(header_pad_top),
            bottom: LengthPercentage::Length(header_pad_bottom),
        },
        align_items: Some(taffy::AlignItems::FlexStart),
        ..Default::default()
    });

    let title_col = tree.create_node(Style {
        flex_direction: FlexDirection::Column,
        flex_grow: 1.0,
        ..Default::default()
    });

    let title = tree.create_node(Style {
        size: TaffySize {
            width: Dimension::Auto,
            height: Dimension::Length(title_line),
        },
        ..Default::default()
    });
    tree.add_child(title_col, title);

    let description = if inputs.has_description {
        let line_count = inputs.description_lines.max(1) as f32;
        let n = tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Auto,
                height: Dimension::Length(description_line * line_count),
            },
            margin: TaffyRect {
                top: LengthPercentageAuto::Length(title_description_gap),
                right: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
                left: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        });
        tree.add_child(title_col, n);
        Some(n)
    } else {
        None
    };

    tree.add_child(header, title_col);

    // Badge sized to a chip + vertically centred against the title
    // line so it doesn't hug the rounded corner.
    let badge = if inputs.has_badge {
        let badge_size: f32 = 14.0;
        let badge_top_offset = ((title_line - badge_size) * 0.5).max(0.0);
        let n = tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Length(badge_size),
                height: Dimension::Length(badge_size),
            },
            margin: TaffyRect {
                left: LengthPercentageAuto::Length(8.0),
                top: LengthPercentageAuto::Length(badge_top_offset),
                right: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        });
        tree.add_child(header, n);
        Some(n)
    } else {
        None
    };

    // Chrome (edit / delete / collapse) — always present. Added
    // AFTER the badge so the flex row places them at the FAR right
    // edge of the header. Each sized to match the badge so all
    // chips sit on the same baseline; if the title line is bigger,
    // the chrome grows with it (clamped to 22px) so the click
    // targets stay generous. Order in the flex row is
    // edit | delete | collapse — collapse stays farthest right so
    // its position remains stable (least surprising) when the
    // delete + edit buttons appear / disappear in future variants.
    let chrome_size: f32 = title_line.clamp(16.0, 22.0);
    let chrome_top_offset = ((title_line - chrome_size) * 0.5).max(0.0);
    let make_chrome = |tree: &mut LayoutTree, left_margin: f32| {
        tree.create_node(Style {
            size: TaffySize {
                width: Dimension::Length(chrome_size),
                height: Dimension::Length(chrome_size),
            },
            margin: TaffyRect {
                left: LengthPercentageAuto::Length(left_margin),
                top: LengthPercentageAuto::Length(chrome_top_offset),
                right: LengthPercentageAuto::Length(0.0),
                bottom: LengthPercentageAuto::Length(0.0),
            },
            ..Default::default()
        })
    };
    // First chrome chip leans against the badge / title; the rest
    // sit shoulder-to-shoulder with a 4px gutter.
    let chrome_edit = make_chrome(&mut tree, 8.0);
    let chrome_delete = make_chrome(&mut tree, 4.0);
    let chrome_collapse = make_chrome(&mut tree, 4.0);
    tree.add_child(header, chrome_edit);
    tree.add_child(header, chrome_delete);
    tree.add_child(header, chrome_collapse);

    tree.compute_layout(
        header,
        TaffySize {
            width: AvailableSpace::Definite(inputs.width),
            // No height constraint — header sizes to content.
            height: AvailableSpace::MaxContent,
        },
    );

    // Read the actual header height from the computed layout so the
    // renderer paints the band at the correct size.
    let header_rect = read_rect(&tree, header, Point::new(0.0, 0.0));

    GroupSlots {
        header: header_rect,
        title: read_rect_within(&tree, title, &[header, title_col]),
        description: description.map(|n| read_rect_within(&tree, n, &[header, title_col])),
        badge: badge.map(|n| read_rect_within(&tree, n, &[header])),
        chrome_edit: read_rect_within(&tree, chrome_edit, &[header]),
        chrome_delete: read_rect_within(&tree, chrome_delete, &[header]),
        chrome_collapse: read_rect_within(&tree, chrome_collapse, &[header]),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Layout read-back helpers
// ─────────────────────────────────────────────────────────────────────

fn read_rect(tree: &LayoutTree, id: LayoutNodeId, parent_origin: Point) -> Rect {
    let Some(l) = tree.get_layout(id) else {
        return Rect::new(0.0, 0.0, 0.0, 0.0);
    };
    Rect::new(
        parent_origin.x + l.location.x,
        parent_origin.y + l.location.y,
        l.size.width,
        l.size.height,
    )
}

/// `read_rect`, but resolves the absolute (node-local) origin by
/// summing parent layout origins. Taffy returns each node's layout
/// position relative to its immediate parent; for the renderer we
/// want positions relative to the node's outer top-left.
fn read_rect_within(tree: &LayoutTree, id: LayoutNodeId, ancestors: &[LayoutNodeId]) -> Rect {
    let mut origin = Point::new(0.0, 0.0);
    for &a in ancestors {
        if let Some(l) = tree.get_layout(a) {
            origin.x += l.location.x;
            origin.y += l.location.y;
        }
    }
    read_rect(tree, id, origin)
}

// ─────────────────────────────────────────────────────────────────────
// Cache warm-up — serial below threshold, rayon above
// ─────────────────────────────────────────────────────────────────────

/// Populate `cache` with `NodeSlots` for every fingerprint in `wanted`
/// not already present. Uses rayon's work-pool above
/// [`PARALLEL_THRESHOLD`]; serial otherwise.
///
/// Returns the count of freshly-computed entries (0 = full cache hit).
pub fn warm_node_slot_cache(
    cache: &NodeSlotCache,
    wanted: &HashSet<NodeFingerprint>,
    input_for: impl Fn(NodeFingerprint) -> NodeSlotInputs + Sync,
) -> usize {
    // Phase A — filter to only uncached fingerprints under a brief
    // read lock.
    let uncached: Vec<NodeFingerprint> = {
        let cache_r = cache.read().unwrap();
        wanted
            .iter()
            .copied()
            .filter(|fp| !cache_r.contains_key(fp))
            .collect()
    };
    if uncached.is_empty() {
        return 0;
    }

    // Phase B — compute, serial or parallel.
    let computed: Vec<(NodeFingerprint, NodeSlots)> = if uncached.len() < PARALLEL_THRESHOLD {
        uncached
            .into_iter()
            .map(|fp| (fp, compute_node_slots(&input_for(fp))))
            .collect()
    } else {
        uncached
            .into_par_iter()
            .map(|fp| (fp, compute_node_slots(&input_for(fp))))
            .collect()
    };

    // Phase C — single write-locked merge.
    let n = computed.len();
    let mut cache_w = cache.write().unwrap();
    for (fp, slots) in computed {
        cache_w.insert(fp, slots);
    }
    n
}

pub fn warm_group_slot_cache(
    cache: &GroupSlotCache,
    wanted: &HashSet<GroupFingerprint>,
    input_for: impl Fn(GroupFingerprint) -> GroupSlotInputs + Sync,
) -> usize {
    let uncached: Vec<GroupFingerprint> = {
        let cache_r = cache.read().unwrap();
        wanted
            .iter()
            .copied()
            .filter(|fp| !cache_r.contains_key(fp))
            .collect()
    };
    if uncached.is_empty() {
        return 0;
    }

    let computed: Vec<(GroupFingerprint, GroupSlots)> = if uncached.len() < PARALLEL_THRESHOLD {
        uncached
            .into_iter()
            .map(|fp| (fp, compute_group_slots(&input_for(fp))))
            .collect()
    } else {
        uncached
            .into_par_iter()
            .map(|fp| (fp, compute_group_slots(&input_for(fp))))
            .collect()
    };

    let n = computed.len();
    let mut cache_w = cache.write().unwrap();
    for (fp, slots) in computed {
        cache_w.insert(fp, slots);
    }
    n
}

// ─────────────────────────────────────────────────────────────────────
// Port anchors — kept out of the layout tree (intrinsically scalable)
// ─────────────────────────────────────────────────────────────────────

/// Evenly-spaced port anchor on a node edge. Port placement is
/// closed-form and doesn't benefit from flex layout. `body` is the
/// node's body-region rect (excluding the header) so L/R ports sit
/// below the header band; for top/bottom ports it's ignored.
pub fn port_anchor(
    bounds: Rect,
    body: Rect,
    side: crate::port::PortPosition,
    index: usize,
    count: usize,
) -> Point {
    crate::render::port_position_on_node(bounds, body, side, index, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_node_slots_fit_within_bounds() {
        let inputs = NodeSlotInputs {
            width: 180.0,
            has_subtitle: true,
            has_badge: true,
            has_icon: true,
            icon_size: 36.0,
            title_font_size: 13.0,
            subtitle_font_size: 11.0,
            content_padding: 10.0,
            content_height: 0.0,
        };
        let slots = compute_node_slots(&inputs);
        assert!(slots.header.height() > 0.0);
        assert!(slots.title.width() > 0.0);
        assert!(slots.subtitle.is_some());
        assert!(slots.badge.is_some());
        assert!(slots.icon.is_some());
        // Title sits inside the header.
        assert!(slots.title.y() >= slots.header.y());
        assert!(
            slots.title.y() + slots.title.height()
                <= slots.header.y() + slots.header.height() + 0.5
        );
        // Total height tracks the icon row: 10 pad + 36 icon + 10 pad.
        assert!(slots.total_height >= 56.0);
    }

    #[test]
    fn icon_drives_height_when_text_shorter() {
        // No subtitle, short title — icon (36px) should still set height.
        let inputs = NodeSlotInputs {
            width: 200.0,
            has_subtitle: false,
            has_badge: false,
            has_icon: true,
            icon_size: 36.0,
            title_font_size: 13.0,
            subtitle_font_size: 11.0,
            content_padding: 10.0,
            content_height: 0.0,
        };
        let slots = compute_node_slots(&inputs);
        // 10 + 36 + 10 = 56
        assert!(
            slots.total_height >= 55.0 && slots.total_height <= 57.0,
            "expected ~56, got {}",
            slots.total_height
        );
    }

    #[test]
    fn content_height_grows_total_height() {
        // 100px content slot under a 56px header (icon-driven) → ~156px total.
        let inputs = NodeSlotInputs {
            width: 200.0,
            has_subtitle: false,
            has_badge: false,
            has_icon: true,
            icon_size: 36.0,
            title_font_size: 13.0,
            subtitle_font_size: 11.0,
            content_padding: 10.0,
            content_height: 100.0,
        };
        let slots = compute_node_slots(&inputs);
        assert!(
            slots.total_height >= 155.0 && slots.total_height <= 157.0,
            "expected ~156, got {}",
            slots.total_height
        );
        // Body should be exactly 100px tall.
        assert!(
            (slots.body.height() - 100.0).abs() < 0.5,
            "body height {} != 100",
            slots.body.height()
        );
    }

    #[test]
    fn warm_cache_idempotent() {
        let cache: NodeSlotCache = Arc::new(RwLock::new(AHashMap::new()));
        let inputs = NodeSlotInputs {
            width: 200.0,
            has_subtitle: false,
            has_badge: false,
            has_icon: false,
            icon_size: 36.0,
            title_font_size: 13.0,
            subtitle_font_size: 11.0,
            content_padding: 10.0,
            content_height: 0.0,
        };
        let fp = NodeFingerprint(42);
        let mut wanted = HashSet::new();
        wanted.insert(fp);
        let inputs_clone = inputs.clone();
        let n1 = warm_node_slot_cache(&cache, &wanted, |_| inputs_clone.clone());
        assert_eq!(n1, 1);
        let inputs_clone = inputs.clone();
        let n2 = warm_node_slot_cache(&cache, &wanted, |_| inputs_clone.clone());
        assert_eq!(n2, 0); // already cached
    }
}
