//! Rendering primitives — node bodies, ports, edges, groups, badges.
//!
//! All drawing routines are pure functions over
//! `&mut dyn DrawContext` plus the resolved [`ThemeResolver`]. The
//! editor's canvas closure drives them in back-to-front order:
//! groups, edges, nodes, drag preview, boundary chrome.
//!
//! ## Coordinate space
//!
//! Everything here works in canvas-content coordinates. The canvas
//! kit handles screen-to-content transforms before the closure
//! fires; callers don't translate.
//!
//! ## Geometry helpers
//!
//! [`port_position_on_node`] is the source of truth for where a port
//! dot lives — every renderer + the interaction layer's drag-preview
//! routing both call it so a port can't be drawn in one spot and
//! hit-tested in another.

use std::sync::Arc;

use blinc_core::draw::{Path, Stroke, TextStyle};
use blinc_core::layer::{Brush, Color, CornerRadius, Point, Rect};
use blinc_core::{DrawContext, FontWeight, TextAlign, TextBaseline};

use crate::bezier::{cubic_point, mid_x_controls};
use crate::connection::{Connection, ConnectionState};
use crate::group::{BadgeKind, Group, StatusBadge};
use crate::node::{NodeInstance, NodeShape};
use crate::port::{Direction, PortDesc, PortKind, PortPosition};
use crate::theme::ThemeResolver;

// ─────────────────────────────────────────────────────────────────────
// Theme-aware draw wrappers
// ─────────────────────────────────────────────────────────────────────
//
// Every rounded-rect fill / stroke in this module routes through
// `themed_fill_rect` / `themed_stroke_rect` so the renderer picks up
// the active theme's `ShapeTokens` (squircle profile + per-radius
// smoothing threshold). Themes that don't opt into squircles fall
// through to `n = 1.0` round arcs — no visible change.

fn max_corner(cr: CornerRadius) -> f32 {
    cr.top_left
        .max(cr.top_right)
        .max(cr.bottom_right)
        .max(cr.bottom_left)
}

fn themed_fill_rect(
    ctx: &mut dyn DrawContext,
    rect: Rect,
    corner_radius: CornerRadius,
    brush: Brush,
    theme: &ThemeResolver<'_>,
) {
    ctx.set_corner_shape(theme.corner_shape_uniform(max_corner(corner_radius)));
    ctx.fill_rect(rect, corner_radius, brush);
    ctx.clear_corner_shape();
}

fn themed_stroke_rect(
    ctx: &mut dyn DrawContext,
    rect: Rect,
    corner_radius: CornerRadius,
    stroke: &Stroke,
    brush: Brush,
    theme: &ThemeResolver<'_>,
) {
    ctx.set_corner_shape(theme.corner_shape_uniform(max_corner(corner_radius)));
    ctx.stroke_rect(rect, corner_radius, stroke, brush);
    ctx.clear_corner_shape();
}

/// Iterate a theme shadow stack and dispatch one `draw_shadow` per
/// layer. The Universal HID variants stack 2-3 layers per slot for
/// depth; single-layer themes pass a one-element slice.
fn draw_shadow_stack(
    ctx: &mut dyn DrawContext,
    rect: Rect,
    corner_radius: CornerRadius,
    shadows: &[blinc_theme::Shadow],
) {
    for s in shadows {
        let core: blinc_core::Shadow = s.into();
        ctx.draw_shadow(rect, corner_radius, core);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Geometry — node bbox + port placement
// ─────────────────────────────────────────────────────────────────────

/// Resolve a node's bounding rect: `position` is the top-left.
///
/// Width and height both come from the slot table — taffy resolves
/// the slot tree's `fit-content` over icon + title + subtitle +
/// padding (and the body region when the template carries a
/// content slot), and the width includes any
/// `template.content.min_width` floor declared at the template
/// level. There is no hand-math here on purpose: this is the SOLE
/// source of truth for the node rect, shared with
/// `iter_port_positions` / `draw_node_at` so the body and the
/// ports never disagree.
pub fn node_bounds<M>(
    _instance: &NodeInstance<M>,
    slots: &crate::slot::NodeSlots,
    _theme: &ThemeResolver<'_>,
) -> Rect {
    Rect::new(
        _instance.position.x,
        _instance.position.y,
        slots.total_width,
        slots.total_height,
    )
}

/// Compute the centre point of a port on a node.
///
/// `bounds` is the full node rect (used for top/bottom edge ports).
/// `body` is the body-region rect — used for left/right ports so
/// they stay BELOW the header band instead of spanning the full
/// node height. Pass `bounds` for both if the node has no header.
///
/// Same routine used by the renderer (to draw the dot) and the
/// interaction layer (to hit-test against the cursor). Keeps the
/// two trivially in sync.
pub fn port_position_on_node(
    bounds: Rect,
    body: Rect,
    position: PortPosition,
    index: usize,
    same_side_count: usize,
) -> Point {
    let count = same_side_count.max(1) as f32;
    let slot = (index as f32 + 0.5) / count; // 0..1 along the edge
    match position {
        // Left / right ride the body's vertical range so the header
        // stays clear of port dots.
        PortPosition::Left => Point::new(bounds.x(), body.y() + body.height() * slot),
        PortPosition::Right => {
            Point::new(bounds.x() + bounds.width(), body.y() + body.height() * slot)
        }
        // Top / bottom sit on the outer edges of the node itself —
        // the header doesn't displace them since they're on the
        // perpendicular axis.
        PortPosition::Top => Point::new(bounds.x() + bounds.width() * slot, bounds.y()),
        PortPosition::Bottom => Point::new(
            bounds.x() + bounds.width() * slot,
            bounds.y() + bounds.height(),
        ),
    }
}

/// Translate a node-local slot rect to absolute (canvas-content)
/// coordinates by offsetting with the node's bounds origin.
fn absolute_body_rect<M>(
    instance: &NodeInstance<M>,
    slots: &crate::slot::NodeSlots,
    bounds: Rect,
) -> Rect {
    let _ = instance;
    Rect::new(
        bounds.x() + slots.body.x(),
        bounds.y() + slots.body.y(),
        slots.body.width(),
        slots.body.height(),
    )
}

/// Geometry of a node's content slot — the inset background that
/// visually distinguishes a portal-content body from the header,
/// plus the inner rect the portal actually paints into.
///
/// Shared by `draw_node_at` (paints the inset background) and the
/// editor's render loop (sizes the `portal.frame` body rect) so
/// the two never disagree about where the content lives.
pub struct ContentSlotRects {
    /// Outer inset rect — the visually distinct background.
    pub inset: Rect,
    /// Inner rect inside the inset, with a small breathing pad —
    /// the area the portal closure paints into.
    pub portal: Rect,
}

/// Compute [`ContentSlotRects`] from the node's outer bounds + slot
/// table. Returns `None` when the node has no content slot
/// (`slots.body.height() == 0`).
pub fn content_slot_rects(
    bounds: Rect,
    slots: &crate::slot::NodeSlots,
) -> Option<ContentSlotRects> {
    if slots.body.height() <= 0.0 {
        return None;
    }
    // Outer padding — distance from the node edge to the inset
    // background. Smaller on top so the inset hugs the header band
    // without a visible gap.
    let pad_x = 10.0_f32;
    let pad_top = 2.0_f32;
    let pad_bot = 10.0_f32;
    let inset = Rect::new(
        bounds.x() + slots.body.x() + pad_x,
        bounds.y() + slots.body.y() + pad_top,
        (slots.body.width() - pad_x * 2.0).max(0.0),
        (slots.body.height() - pad_top - pad_bot).max(0.0),
    );
    // Inner padding inside the inset — gives portal widgets a
    // breathing margin from the inset rect's edge.
    let inner_pad = 8.0_f32;
    let portal = Rect::new(
        inset.x() + inner_pad,
        inset.y() + inner_pad,
        (inset.width() - inner_pad * 2.0).max(0.0),
        (inset.height() - inner_pad * 2.0).max(0.0),
    );
    Some(ContentSlotRects { inset, portal })
}

/// Iterate a node's ports + their resolved centre points, grouped
/// by [`PortPosition`]. Uses `slots.body` (translated to absolute
/// coords) for L/R ports so they sit below the header.
pub fn iter_port_positions<'a, K: PortKind, M>(
    instance: &'a NodeInstance<M>,
    inputs: &'a [PortDesc<K>],
    outputs: &'a [PortDesc<K>],
    slots: &'a crate::slot::NodeSlots,
    theme: &'a ThemeResolver<'a>,
) -> impl Iterator<Item = (Direction, &'a PortDesc<K>, Point)> + 'a {
    let bounds = node_bounds(instance, slots, theme);
    let body = absolute_body_rect(instance, slots, bounds);

    let mut by_side: [Vec<(Direction, &PortDesc<K>)>; 4] = Default::default();
    let bucket = |p: PortPosition| -> usize {
        match p {
            PortPosition::Top => 0,
            PortPosition::Right => 1,
            PortPosition::Bottom => 2,
            PortPosition::Left => 3,
        }
    };
    for desc in inputs {
        by_side[bucket(desc.resolved_position())].push((Direction::Input, desc));
    }
    for desc in outputs {
        by_side[bucket(desc.resolved_position())].push((Direction::Output, desc));
    }

    by_side
        .into_iter()
        .enumerate()
        .flat_map(move |(side_idx, entries)| {
            let position = match side_idx {
                0 => PortPosition::Top,
                1 => PortPosition::Right,
                2 => PortPosition::Bottom,
                _ => PortPosition::Left,
            };
            let total = entries.len();
            entries
                .into_iter()
                .enumerate()
                .map(move |(i, (dir, desc))| {
                    let pt = port_position_on_node(bounds, body, position, i, total);
                    (dir, desc, pt)
                })
        })
}

// ─────────────────────────────────────────────────────────────────────
// Node body
// ─────────────────────────────────────────────────────────────────────

/// Translate a node-local rect by the node's outer origin.
fn translate_rect(rect: Rect, origin: Point) -> Rect {
    Rect::new(
        rect.x() + origin.x,
        rect.y() + origin.y,
        rect.width(),
        rect.height(),
    )
}

/// Draw a node body — background + header + title + subtitle +
/// selection ring + corner badge + optional icon. Ports are drawn
/// separately by [`draw_port`] so callers can interleave port-hover
/// overlays.
///
/// `slots` carries the resolved interior rects in node-local
/// coordinates (computed off-render by [`crate::slot`] and cached by
/// fingerprint). `template` supplies the fallback icon when the
/// instance has none. `is_selected` controls the outline ring; the
/// editor passes its selection set through.
pub fn draw_node<K: PortKind, M>(
    ctx: &mut dyn DrawContext,
    instance: &NodeInstance<M>,
    template: &crate::node::NodeTemplate<K>,
    theme: &ThemeResolver<'_>,
    slots: &crate::slot::NodeSlots,
    is_selected: bool,
) {
    let bounds = node_bounds(instance, slots, theme);
    draw_node_at(
        ctx,
        instance,
        template,
        bounds,
        theme,
        slots,
        is_selected,
        instance.disabled,
    );
}

/// Same as [`draw_node`] but with explicit `bounds` + explicit
/// `disabled`. The editor uses this to:
/// - draw nodes at their effective port-count-driven height (which
///   can exceed `node_bounds(instance, slots, theme)` when L/R
///   sides hold multiple ports), AND
/// - apply group-inherited disable state — `disabled` may be true
///   even when `instance.disabled` is false because the node is a
///   member of a disabled group.
#[allow(clippy::too_many_arguments)] // intrinsic to the per-node paint surface
pub fn draw_node_at<K: PortKind, M>(
    ctx: &mut dyn DrawContext,
    instance: &NodeInstance<M>,
    template: &crate::node::NodeTemplate<K>,
    bounds: Rect,
    theme: &ThemeResolver<'_>,
    slots: &crate::slot::NodeSlots,
    is_selected: bool,
    disabled: bool,
) -> Option<Rect> {
    // Soft-disable: dim every primitive emitted while painting this
    // node by pushing a single opacity frame. Cheaper + more uniform
    // than alpha-mixing every fill colour individually, and keeps the
    // selection ring + badge + icon all dimmed together. The
    // `push_opacity` API multiplies into the parent opacity, so this
    // composes correctly inside a portal clip's existing opacity
    // stack.
    let pushed_opacity = disabled;
    if pushed_opacity {
        ctx.push_opacity(theme.node_disabled_alpha());
    }
    let origin = Point::new(bounds.x(), bounds.y());
    // Subgraph-reference nodes ALWAYS render as a diamond regardless
    // of their template's `default_shape` or any per-instance shape
    // override — the diamond is the editor's "this is navigable
    // subgraph entry" signal, matching Zeal's `SubgraphNode` pattern.
    // Hosts that explicitly want a different shape on a subgraph-ref
    // node would need to clear `subgraph_ref` (and then handle the
    // navigation themselves via their own click listener).
    let shape = if instance.subgraph_ref.is_some() {
        NodeShape::Diamond
    } else {
        instance.shape.unwrap_or(NodeShape::Rectangle)
    };
    let radius = theme.node_corner_radius();
    let cr_body = CornerRadius::uniform(radius);

    // Hoisted: shift that centres the whole header block (icon +
    // title + optional subtitle + badge) against the body bounds.
    // Computed once and shared by the icon / title / badge draw
    // sites so all four elements stay vertically aligned.
    //
    // Centering ONLY kicks in for headerless / content-less nodes
    // whose `effective_bounds` grew taller than the natural header
    // (multi-port pad). When there's a real content body
    // (`slots.body.height() > 0`) the header sits at the TOP and
    // the body fills the rest — otherwise the centring shift
    // pushes the title + icon down INTO the content region.
    let header_h = slots.header.height();
    let header_y_shift = if slots.body.height() > 0.0 {
        0.0
    } else {
        ((bounds.height() - header_h) * 0.5).max(0.0) - slots.header.y()
    };

    // Drop the theme shadow stack under the body BEFORE the fill.
    if matches!(shape, NodeShape::Rectangle | NodeShape::Custom) {
        draw_shadow_stack(ctx, bounds, cr_body, &theme.node_shadow_stack());
    }

    // Subgraph-reference nodes pick up the theme's accent fill +
    // stroke so they read distinct from regular surface-tinted
    // rectangle nodes at a glance. Matches Zeal's `orange-600`
    // convention. Hosts wanting a different colour treatment can
    // override the `node_subgraph_*` theme tokens or wire a
    // per-instance variant resolver (planned follow-up).
    let is_subgraph_ref = instance.subgraph_ref.is_some();
    let body_fill = if is_subgraph_ref {
        theme.node_subgraph_fill()
    } else {
        theme.node_body_fill()
    };

    // Body fill — dispatched per shape so Circle / Diamond differ
    // visually. Custom collapses to Rectangle.
    match shape {
        NodeShape::Circle => {
            let centre = Point::new(
                bounds.x() + bounds.width() * 0.5,
                bounds.y() + bounds.height() * 0.5,
            );
            let r = bounds.width().min(bounds.height()) * 0.5;
            ctx.fill_circle(centre, r, Brush::Solid(body_fill));
        }
        NodeShape::Diamond => {
            let cx = bounds.x() + bounds.width() * 0.5;
            let cy = bounds.y() + bounds.height() * 0.5;
            let path = Path::new()
                .move_to(cx, bounds.y())
                .line_to(bounds.x() + bounds.width(), cy)
                .line_to(cx, bounds.y() + bounds.height())
                .line_to(bounds.x(), cy)
                .close();
            ctx.fill_path(&path, Brush::Solid(body_fill));
        }
        NodeShape::Rectangle | NodeShape::Custom => {
            themed_fill_rect(ctx, bounds, cr_body, Brush::Solid(body_fill), theme);
        }
    }

    // Content-slot inset background. When the node has a content
    // body (`slots.body.height() > 0`), draw a slightly darker
    // rounded rect inside that region so the portal area reads as
    // recessed against the header chrome above it. Geometry shared
    // with the editor's `portal.frame` call site via
    // `content_slot_rects` so paint + portal-bounds stay aligned.
    if matches!(shape, NodeShape::Rectangle | NodeShape::Custom) {
        if let Some(slot) = content_slot_rects(bounds, slots) {
            // The content slot now uses the workspace `Background`
            // token (same as the canvas bg) instead of a darken of
            // the body fill. Reads as a recess sunk through the
            // node body down to the canvas surface — cleaner on
            // light bundles where the previous `darken(body, 0.20)`
            // produced a muddy grey that didn't relate to either
            // the node chrome or the workspace tone. A 1px Border
            // outline on top delineates the slot from the node body
            // on light schemes where fill alone may not carry
            // enough contrast.
            let inset_fill = theme.content_slot_fill();
            let inset_border = theme.content_slot_border();
            let inset_radius = CornerRadius::uniform(radius * 0.7);
            themed_fill_rect(
                ctx,
                slot.inset,
                inset_radius,
                Brush::Solid(inset_fill),
                theme,
            );
            themed_stroke_rect(
                ctx,
                slot.inset,
                inset_radius,
                &Stroke::new(1.0),
                Brush::Solid(inset_border),
                theme,
            );
        }
    }

    // Effective icon — instance override wins, then template.
    let effective_icon = instance.icon.as_ref().or(template.icon.as_ref());

    // Per Zeal's compact-node design: header background MATCHES
    // body background — no separate band tint. The title + icon
    // sit directly on the body surface; content slots will get
    // their OWN backdrop (e.g. a code editor's dark chrome) when
    // wired up, but the node chrome stays uniform.
    // Body fill already covered the whole `bounds` above, so we
    // skip the extra header rect entirely and just place the
    // icon + title.
    if matches!(shape, NodeShape::Rectangle | NodeShape::Custom) {
        if let (Some(icon), Some(slot)) = (effective_icon, slots.icon) {
            let mut icon_rect = translate_rect(slot, origin);
            icon_rect = Rect::new(
                icon_rect.x(),
                icon_rect.y() + header_y_shift,
                icon_rect.width(),
                icon_rect.height(),
            );
            draw_icon(ctx, icon, icon_rect);
        }
        draw_node_title(ctx, instance, slots, origin, bounds, header_y_shift, theme);
    } else {
        // Circle / Diamond: title sits CENTRED INSIDE the shape's
        // widest band (the title fits naturally because it's short
        // — "Subgraph", "Filter", a short component name). The
        // subtitle / namespace, which is typically longer than the
        // diamond's effective width (e.g.
        // "demo-workflow/sample-sub"), renders OUTSIDE the diamond
        // just below it, horizontally centred against the bounds.
        // This matches Zeal's SubgraphNode pattern — the namespace
        // is the secondary chrome and lives below the body without
        // forcing the diamond itself to grow wide enough to fit
        // the longest possible namespace.
        let title_text = &instance.component;
        let subtitle_text = instance.subtitle.as_deref();
        let title_size = theme.title_font_size();
        let subtitle_size = theme.subtitle_font_size();
        let centre_x = bounds.x() + bounds.width() * 0.5;
        let centre_y = bounds.y() + bounds.height() * 0.5;
        let title_style = TextStyle::new(title_size)
            .with_color(theme.node_title_color())
            .with_weight(FontWeight::Medium)
            .with_align(TextAlign::Center)
            .with_baseline(TextBaseline::Middle);
        ctx.draw_text(title_text, Point::new(centre_x, centre_y), &title_style);
        if let Some(sub) = subtitle_text {
            // Subtitle drawn below the shape, horizontally centred
            // against the bounds rect. Top-baseline so the text
            // grows downward from a fixed anchor line.
            let subtitle_anchor_y = bounds.y() + bounds.height() + 6.0;
            let sub_style = TextStyle::new(subtitle_size)
                .with_color(theme.node_subtitle_color())
                .with_align(TextAlign::Center)
                .with_baseline(TextBaseline::Top);
            ctx.draw_text(sub, Point::new(centre_x, subtitle_anchor_y), &sub_style);
        }
    }

    // Body + selection outline — must follow the shape's actual edges,
    // not the bounding rect. Diamond and Circle nodes (used for
    // subgraph navigation entry-points and pin-style chrome) get their
    // outline traced along the inscribed shape; rectangles use the
    // rounded-rect stroke they always have. Subgraph-reference nodes
    // additionally pick up the accent stroke colour so the diamond
    // outline reads as a deliberate "navigable subgraph" affordance.
    let body_stroke_color = if is_subgraph_ref {
        theme.node_subgraph_stroke()
    } else {
        theme.node_body_stroke()
    };
    let outline_brush = if is_selected {
        Brush::Solid(theme.node_selected_outline())
    } else {
        Brush::Solid(body_stroke_color)
    };
    let outline_width = if is_selected { 2.0 } else { 1.0 };
    let outline_stroke = Stroke::new(outline_width);
    match shape {
        NodeShape::Diamond => {
            let cx = bounds.x() + bounds.width() * 0.5;
            let cy = bounds.y() + bounds.height() * 0.5;
            let path = Path::new()
                .move_to(cx, bounds.y())
                .line_to(bounds.x() + bounds.width(), cy)
                .line_to(cx, bounds.y() + bounds.height())
                .line_to(bounds.x(), cy)
                .close();
            ctx.stroke_path(&path, &outline_stroke, outline_brush);
        }
        NodeShape::Circle => {
            let centre = Point::new(
                bounds.x() + bounds.width() * 0.5,
                bounds.y() + bounds.height() * 0.5,
            );
            let r = bounds.width().min(bounds.height()) * 0.5;
            ctx.stroke_circle(centre, r, &outline_stroke, outline_brush);
        }
        NodeShape::Rectangle | NodeShape::Custom => {
            themed_stroke_rect(ctx, bounds, cr_body, &outline_stroke, outline_brush, theme);
        }
    }

    // Status badge — Zeal positions it at the TOP-RIGHT corner of
    // the icon itself, overlapping the icon edge slightly so it
    // reads as a "live indicator" attached to the icon (think OS
    // notification dot on an app). When there's no icon, fall
    // back to the slot's far-right header position.
    let mut badge_rect: Option<Rect> = None;
    if let Some(badge_data) = &instance.badge {
        if let Some(icon_slot) = slots.icon {
            let icon_rect = translate_rect(icon_slot, origin);
            let badge_size = slots
                .badge
                .map(|b| b.width().min(b.height()))
                .unwrap_or(12.0);
            // Overlap = badge_size * 0.30 so ~1/3 of the badge
            // sits over the icon edge, the rest hangs outside.
            let overlap = badge_size * 0.30;
            let icon_top = icon_rect.y() + header_y_shift;
            let chip = Rect::new(
                icon_rect.x() + icon_rect.width() - badge_size + overlap,
                icon_top - overlap,
                badge_size,
                badge_size,
            );
            draw_badge(ctx, badge_data, chip, theme);
            badge_rect = Some(chip);
        } else if let Some(badge_slot) = slots.badge {
            let chip_raw = translate_rect(badge_slot, origin);
            let chip = Rect::new(
                chip_raw.x(),
                chip_raw.y() + header_y_shift,
                chip_raw.width(),
                chip_raw.height(),
            );
            draw_badge(ctx, badge_data, chip, theme);
            badge_rect = Some(chip);
        }
    }
    if pushed_opacity {
        ctx.pop_opacity();
    }
    badge_rect
}

fn draw_node_title<M>(
    ctx: &mut dyn DrawContext,
    instance: &NodeInstance<M>,
    slots: &crate::slot::NodeSlots,
    origin: Point,
    bounds: Rect,
    header_y_shift: f32,
    theme: &ThemeResolver<'_>,
) {
    let title = &instance.component;
    let subtitle = instance.subtitle.as_deref();
    let title_size = theme.title_font_size();
    let subtitle_size = theme.subtitle_font_size();
    let has_subtitle = subtitle.is_some() && slots.subtitle.is_some();

    let title_rect = translate_rect(slots.title, origin);
    if has_subtitle {
        // Stacked layout — title at slot top, subtitle below. The
        // shift offset moves both elements down by the same amount
        // so they stay centred against bounds as a unit.
        let title_style = TextStyle::new(title_size)
            .with_color(theme.node_title_color())
            .with_weight(FontWeight::Medium)
            .with_align(TextAlign::Left)
            .with_baseline(TextBaseline::Top);
        ctx.draw_text(
            title,
            Point::new(title_rect.x(), title_rect.y() + header_y_shift),
            &title_style,
        );

        if let (Some(sub), Some(subtitle_slot)) = (subtitle, slots.subtitle) {
            let subtitle_rect = translate_rect(subtitle_slot, origin);
            let sub_style = TextStyle::new(subtitle_size)
                .with_color(theme.node_subtitle_color())
                .with_align(TextAlign::Left)
                .with_baseline(TextBaseline::Top);
            ctx.draw_text(
                sub,
                Point::new(subtitle_rect.x(), subtitle_rect.y() + header_y_shift),
                &sub_style,
            );
        }
    } else {
        // No subtitle: align the title's vertical centre to the
        // ICON's centre (when there is one) so they read as a
        // single aligned row. Falls back to bounds centre for
        // icon-less nodes. Using icon-centre instead of
        // bounds-centre keeps the title locked to the icon even
        // if the icon SVG draws slightly off-centre within its
        // bounding rect.
        let title_style = TextStyle::new(title_size)
            .with_color(theme.node_title_color())
            .with_weight(FontWeight::Medium)
            .with_align(TextAlign::Left)
            .with_baseline(TextBaseline::Middle);
        let centre_y = if let Some(icon_slot) = slots.icon {
            let icon_rect = translate_rect(icon_slot, origin);
            icon_rect.y() + header_y_shift + icon_rect.height() * 0.5
        } else {
            bounds.y() + bounds.height() * 0.5
        };
        ctx.draw_text(title, Point::new(title_rect.x(), centre_y), &title_style);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Port
// ─────────────────────────────────────────────────────────────────────

/// Draw one port dot at the resolved centre point.
///
/// Ports get a **soft drop shadow** (not a thick stroke ring) so
/// they read as elevated chips off the node body. Hover /
/// compatibility states recolour the shadow itself so the cue
/// blooms outward instead of growing a competing border.
pub fn draw_port<K: PortKind>(
    ctx: &mut dyn DrawContext,
    desc: &PortDesc<K>,
    centre: Point,
    theme: &ThemeResolver<'_>,
    hover_state: PortHoverState,
) {
    let r = theme.port_radius();
    let accent = desc.kind.accent();

    // Hover / compat states inflate the shadow blur and recolour
    // it. Idle = neutral drop from the theme.
    let shadow = match hover_state {
        PortHoverState::None => theme.port_shadow(),
        PortHoverState::Hovered => tinted_glow(theme.port_hover_outline(), 6.0, 0.55),
        PortHoverState::Compatible => tinted_glow(theme.port_compatible_outline(), 7.0, 0.65),
        PortHoverState::Incompatible => tinted_glow(theme.port_incompatible_outline(), 7.0, 0.65),
    };

    ctx.draw_circle_shadow(centre, r, shadow);
    ctx.fill_circle(centre, r, Brush::Solid(accent));
}

/// Width of `s` rendered at `sz` px. Delegates to the global
/// `blinc_layout::measure_text` which uses the registered
/// `FontTextMeasurer` on real apps (desktop / web) and an estimator
/// fallback in unit tests. A hand-rolled per-glyph advance table
/// would underestimate wide glyphs (`M`, `W`, em-dash, non-ASCII
/// Unicode) and leave tooltip chips sized smaller than the
/// rendered text — measure-text-via-the-platform avoids that
/// class of layout bug entirely.
pub(crate) fn estimate_text_width(s: &str, sz: f32) -> f32 {
    blinc_layout::measure_text(s, sz).width
}

/// Greedy word-wrap against `max_w` using [`estimate_text_width`].
/// Splits on whitespace and never breaks a word mid-character — a
/// word longer than `max_w` ends up on a line of its own (URLs /
/// identifiers stay whole). Returns at least one line even for an
/// empty input so callers can iterate without an empty-string check.
pub(crate) fn wrap_text(s: &str, sz: f32, max_w: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w = 0.0_f32;
    let space_w = estimate_text_width(" ", sz);
    for word in s.split_whitespace() {
        let word_w = estimate_text_width(word, sz);
        if current.is_empty() {
            current.push_str(word);
            current_w = word_w;
        } else if current_w + space_w + word_w <= max_w {
            current.push(' ');
            current.push_str(word);
            current_w += space_w + word_w;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
            current_w = word_w;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Render a tooltip chip near a hovered port. Shows the port's
/// name + optional description (from
/// [`crate::port::PortDesc::with_description`]). Anchored OUTSIDE
/// the port so the cursor target stays visible — input ports get
/// the tooltip to the LEFT (since inputs sit on the node's left
/// edge), outputs to the RIGHT. Vertical centring follows the
/// port's centre line.
///
/// Drawn directly inside the canvas closure so the tooltip
/// composites against the same SDF batch as the rest of the
/// editor chrome (no overlay-stack flicker on canvases that
/// re-render every frame).
pub fn draw_port_tooltip<K: PortKind>(
    ctx: &mut dyn DrawContext,
    desc: &PortDesc<K>,
    centre: Point,
    theme: &ThemeResolver<'_>,
) {
    draw_port_tooltip_clamped(
        ctx,
        desc,
        centre,
        theme,
        None,
        blinc_core::layer::Affine2D::IDENTITY,
    );
}

/// Variant of [`draw_port_tooltip`] that clamps the chip horizontally
/// into `viewport` (in screen coordinates) so a port near the edge of
/// the visible canvas doesn't paint its description off-screen. Also
/// caps the description to a max line width and word-wraps when it
/// overflows so a long sentence (e.g. "Numeric cutoff — values above
/// pass through") doesn't render as one runaway line.
///
/// Pass `viewport = None` to keep the legacy "anchor-only" behaviour
/// (no clamp); call sites that have a viewport handy should pass it.
///
/// `canvas_view` is the content→screen affine the canvas surface is
/// painted under (the canvas-kit viewport's `transform()`). The
/// tooltip body is drawn in screen space by pushing the inverse of
/// this affine so the chip stays a constant on-screen size regardless
/// of canvas zoom — without this, the port tooltip would balloon when
/// zoomed in and shrink to unreadable when zoomed out. Pass
/// `Affine2D::IDENTITY` when calling outside a zoomable canvas (the
/// legacy `draw_port_tooltip` shim does this).
pub fn draw_port_tooltip_clamped<K: PortKind>(
    ctx: &mut dyn DrawContext,
    desc: &PortDesc<K>,
    centre: Point,
    theme: &ThemeResolver<'_>,
    viewport: Option<Rect>,
    canvas_view: blinc_core::layer::Affine2D,
) {
    let title = if desc.name.is_empty() {
        desc.id.as_str().to_string()
    } else {
        desc.name.clone()
    };
    let description = desc.description.clone();

    // Move the anchor + the port's visual radius into screen space.
    // After we push the inverse-viewport transform below, the tooltip
    // body draws in screen-pixel logical units, so the anchor and the
    // gap-from-port-edge calculation must also be screen-relative.
    let screen_centre = canvas_view.transform_point(centre);
    let viewport_zoom = canvas_view_zoom(&canvas_view);
    let port_r_screen = theme.port_radius() * viewport_zoom;

    // Padding bumped from 10/6 to 12/8 so even at the edge cases the
    // real text measurer disagrees on by ~1-2 px, there's still a
    // visible gap between the text and the chip border.
    let pad_x = 12.0_f32;
    let pad_y = 8.0_f32;
    let title_size = 12.0_f32;
    let desc_size = 11.0_f32;
    let line_gap = 3.0_f32;
    // Maximum chip width (content-area excluding pad_x*2). Picked
    // empirically: ~220 px reads as a tooltip rather than a small
    // dialog, and at 11pt it fits about 8-9 words per line which is
    // comfortable for the kind of one-sentence descriptions port
    // authors tend to write. Long descriptions wrap onto multiple
    // lines instead of stretching the chip across the viewport.
    let max_content_w: f32 = 220.0;
    let title_w = estimate_text_width(&title, title_size).min(max_content_w);
    let desc_lines: Vec<String> = description
        .as_ref()
        .map(|d| wrap_text(d, desc_size, max_content_w))
        .unwrap_or_default();
    let desc_w = desc_lines
        .iter()
        .map(|l| estimate_text_width(l, desc_size))
        .fold(0.0_f32, f32::max);
    let content_w = title_w.max(desc_w);
    let desc_line_h = desc_size + 2.0;
    let content_h = if !desc_lines.is_empty() {
        title_size + line_gap + desc_lines.len() as f32 * desc_line_h - (desc_line_h - desc_size)
    } else {
        title_size
    };
    let chip_w = content_w + pad_x * 2.0;
    let chip_h = content_h + pad_y * 2.0;

    // Anchor outside the port on the side it sits. 8px gap. All
    // arithmetic is in screen-pixel units now.
    let gap = 8.0_f32;
    let is_input = matches!(desc.direction, crate::port::Direction::Input);
    let (mut chip_x, mut chip_y) = if is_input {
        (
            screen_centre.x - port_r_screen - gap - chip_w,
            screen_centre.y - chip_h * 0.5,
        )
    } else {
        (
            screen_centre.x + port_r_screen + gap,
            screen_centre.y - chip_h * 0.5,
        )
    };

    // Clamp into viewport when one was supplied. Both `vp` and the
    // chip coordinates live in the same (screen-pixel) space so
    // the comparison is correct at every zoom level. Mixing
    // content-space chip coords against a screen-space viewport
    // would silently under-clamp at high zoom and over-clamp at
    // low zoom.
    if let Some(vp) = viewport {
        let inset = 4.0_f32;
        if chip_x + chip_w > vp.x() + vp.width() - inset {
            chip_x = vp.x() + vp.width() - inset - chip_w;
        }
        if chip_x < vp.x() + inset {
            chip_x = vp.x() + inset;
        }
        if chip_y + chip_h > vp.y() + vp.height() - inset {
            chip_y = vp.y() + vp.height() - inset - chip_h;
        }
        if chip_y < vp.y() + inset {
            chip_y = vp.y() + inset;
        }
    }

    let chip = Rect::new(chip_x, chip_y, chip_w, chip_h);
    let cr = CornerRadius::uniform(6.0);

    // Push the inverse of the canvas viewport transform so subsequent
    // emits land in screen-pixel logical units (the viewport's
    // existing `then(&pushed)` composition means
    // `parent_dpi * viewport * inverse(viewport) = parent_dpi` — DPI
    // scaling is preserved, only the zoom is cancelled). Fall back to
    // identity when the affine is singular (zoom = 0, which shouldn't
    // happen in practice but guards against `affine_inverse` returning
    // `None`).
    let inverse = blinc_canvas_kit::affine_inverse(&canvas_view)
        .unwrap_or(blinc_core::layer::Affine2D::IDENTITY);
    ctx.push_transform(blinc_core::draw::Transform::Affine2D(inverse));

    ctx.draw_shadow(chip, cr, theme.port_shadow());
    let bg = theme.tooltip_bg();
    let border = theme.tooltip_border();
    ctx.fill_rect(chip, cr, Brush::Solid(bg));
    let stroke = Stroke::new(1.0);
    ctx.stroke_rect(chip, cr, &stroke, Brush::Solid(border));

    // Text alignment follows the side the chip sits on. After
    // clamping the chip may have moved relative to the port, but we
    // keep the original anchor-side alignment because the chip's own
    // edge is what the text aligns to, not the port.
    let (text_align, text_x) = if is_input {
        (TextAlign::Right, chip_x + chip_w - pad_x)
    } else {
        (TextAlign::Left, chip_x + pad_x)
    };
    let text_color = theme.tooltip_text();
    let secondary = theme.tooltip_text_secondary();
    let title_style = TextStyle::new(title_size)
        .with_color(text_color)
        .with_weight(FontWeight::Medium)
        .with_align(text_align)
        .with_baseline(TextBaseline::Top);
    ctx.draw_text(&title, Point::new(text_x, chip_y + pad_y), &title_style);

    if !desc_lines.is_empty() {
        let desc_style = TextStyle::new(desc_size)
            .with_color(secondary)
            .with_align(text_align)
            .with_baseline(TextBaseline::Top);
        for (i, line) in desc_lines.iter().enumerate() {
            ctx.draw_text(
                line,
                Point::new(
                    text_x,
                    chip_y + pad_y + title_size + line_gap + i as f32 * desc_line_h,
                ),
                &desc_style,
            );
        }
    }

    // Pop the inverse-viewport push so subsequent canvas-kit emits
    // (marquee, etc.) paint under the normal viewport scope again.
    ctx.pop_transform();
}

/// Extract the uniform-zoom factor from a content→screen affine.
/// Assumes the affine is `scale(zoom) * translate(pan)` with no
/// rotation or shear — which is what canvas-kit's `CanvasViewport`
/// produces. Falls back to `sqrt(|det|)` for the general case so
/// future viewport extensions (rotation, non-uniform scale) still
/// give a sane scalar for "how much does a 1-unit content delta
/// expand into screen pixels".
fn canvas_view_zoom(affine: &blinc_core::layer::Affine2D) -> f32 {
    let [a, b, c, d, _, _] = affine.elements;
    let det = a * d - b * c;
    det.abs().sqrt()
}

/// Build a coloured, soft glow shadow for active port states.
fn tinted_glow(color: Color, blur: f32, alpha: f32) -> blinc_core::Shadow {
    blinc_core::Shadow {
        offset_x: 0.0,
        offset_y: 0.0,
        blur,
        spread: 0.0,
        color: Color::rgba(color.r, color.g, color.b, alpha),
    }
}

/// Outline-ring state for [`draw_port`]. Resolved by the interaction
/// layer based on the current drag and per-frame validation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortHoverState {
    None,
    /// Cursor hovering over the port but no drag in progress.
    Hovered,
    /// Drag in progress and this port is a compatible candidate.
    Compatible,
    /// Drag in progress and this port is an incompatible candidate.
    Incompatible,
}

// ─────────────────────────────────────────────────────────────────────
// Edge / connection
// ─────────────────────────────────────────────────────────────────────

/// Draw one connection edge as a cubic bezier from `from` → `to`,
/// using mid-X control points per the canvas_kit_demo convention.
/// Colour comes from the connection's [`ConnectionState`] via the
/// theme.
///
/// `selected` and `hovered` both override the colour with the
/// theme's edge-selection outline so the user gets immediate visual
/// feedback that the curve is interactive.
///
/// `time_secs` drives state-specific motion:
/// * [`ConnectionState::Pending`] → opacity pulses at ~0.7 Hz.
/// * [`ConnectionState::Running`] → bright dot travels along the
///   curve at ~0.5 cycles/sec.
///
/// Pass `0.0` to disable motion (host rendering a snapshot, no
/// per-frame tick).
#[allow(clippy::too_many_arguments)] // intrinsic to the edge paint surface
pub fn draw_edge<M>(
    ctx: &mut dyn DrawContext,
    conn: &Connection<M>,
    from: Point,
    to: Point,
    theme: &ThemeResolver<'_>,
    selected: bool,
    hovered: bool,
    time_secs: f32,
) {
    draw_edge_with_state(
        ctx, conn.state, from, to, theme, selected, hovered, time_secs,
    );
}

/// Same as [`draw_edge`] but accepts the `ConnectionState` explicitly.
/// The editor uses this to downgrade edges incident to a soft-disabled
/// node to `Pending` regardless of the connection's own state — that
/// gives the disabled-node halo a consistent "broken dataflow" cue
/// without mutating the underlying [`Connection`].
#[allow(clippy::too_many_arguments)] // intrinsic to the edge paint surface
pub fn draw_edge_with_state(
    ctx: &mut dyn DrawContext,
    state: ConnectionState,
    from: Point,
    to: Point,
    theme: &ThemeResolver<'_>,
    selected: bool,
    hovered: bool,
    time_secs: f32,
) {
    let base = if selected || hovered {
        theme.edge_selected_outline()
    } else {
        theme.edge_color_for_state(state)
    };
    let thickness = if selected || hovered {
        theme.edge_thickness() + 1.0
    } else {
        theme.edge_thickness()
    };
    // Selection / hover overrides the state animation — we want
    // user interaction to read as priority over background runtime
    // motion.
    if selected || hovered {
        draw_edge_curve(ctx, from, to, thickness, base);
        return;
    }
    match state {
        ConnectionState::Pending => {
            // Slow opacity pulse — keeps the eye drawn but reads as
            // "waiting" rather than "live".
            let phase = (time_secs * std::f32::consts::TAU * 0.7).sin() * 0.5 + 0.5;
            let alpha = 0.35 + 0.45 * phase;
            let mut c = base;
            c.a *= alpha;
            draw_edge_curve(ctx, from, to, thickness, c);
        }
        ConnectionState::Running => {
            // Bright dot travels along the curve at 0.5 cycles/sec.
            // Per-segment tint is computed by `flow_tint` based on
            // the segment's t along the curve relative to the
            // current flow position.
            let flow = (time_secs * 0.5).rem_euclid(1.0);
            draw_edge_curve_tinted(ctx, from, to, thickness, |t| flow_tint(base, t, flow));
        }
        _ => draw_edge_curve(ctx, from, to, thickness, base),
    }
    let _ = time_secs; // silence on no-animation states
}

/// Lighten `base` toward white based on distance from `flow`
/// (both ∈ [0, 1] along the curve). Distance wraps so the bright
/// dot can loop seamlessly. Intensity falls off with a narrow
/// gaussian-ish curve for a tight comet-tail look.
fn flow_tint(base: Color, t: f32, flow: f32) -> Color {
    let raw = (t - flow).abs();
    let dist = raw.min(1.0 - raw);
    // Width of the bright dot — 0.18 of the curve length sits in
    // the "lit" zone with smooth falloff.
    let lit = ((1.0 - dist * 5.5).max(0.0)).powi(2);
    let mix = lit * 0.65;
    Color::rgba(
        base.r + (1.0 - base.r) * mix,
        base.g + (1.0 - base.g) * mix,
        base.b + (1.0 - base.b) * mix,
        base.a,
    )
}

/// Render a small × delete button anchored at `centre`. Returns the
/// AABB the editor should register as a hit region so clicks fire
/// `EditorEvent::DeleteConnectionRequested`.
///
/// The button is a filled circle in the theme's error colour with
/// two diagonal stroke arms forming the ×. Sized so the visual
/// centre matches `centre` exactly (no off-by-half-pixel drift
/// when registering the hit rect).
pub fn draw_edge_delete_button(
    ctx: &mut dyn DrawContext,
    centre: Point,
    theme: &ThemeResolver<'_>,
) -> Rect {
    let radius = theme.edge_delete_button_radius();
    let rect = Rect::new(
        centre.x - radius,
        centre.y - radius,
        radius * 2.0,
        radius * 2.0,
    );

    // Filled circle background.
    let fill = Brush::Solid(theme.edge_delete_button_fill());
    ctx.fill_rect(rect, CornerRadius::uniform(radius), fill);

    // × strokes — two short diagonal rects rotated ±45°, scaled to
    // the button's interior. Drawn via the same translate+rotate
    // trick `draw_edge_curve` uses so the strokes route through the
    // SDF batch (rendered reliably in canvas-closure overlay).
    let arm = radius * 0.55;
    let stroke_w = (radius * 0.18).max(1.0);
    let glyph_brush = Brush::Solid(theme.edge_delete_button_glyph());
    for angle_deg in [45.0_f32, -45.0_f32] {
        ctx.push_transform(blinc_core::draw::Transform::translate(centre.x, centre.y));
        ctx.push_transform(blinc_core::draw::Transform::rotate(angle_deg.to_radians()));
        ctx.fill_rect(
            Rect::new(-arm, -stroke_w * 0.5, arm * 2.0, stroke_w),
            CornerRadius::uniform(0.0),
            glyph_brush.clone(),
        );
        ctx.pop_transform();
        ctx.pop_transform();
    }
    rect
}

/// Draw the drag-preview edge from the source port's centre to the
/// current cursor, tinted by whether the cursor is over a compatible
/// candidate.
pub fn draw_drag_preview(
    ctx: &mut dyn DrawContext,
    from: Point,
    cursor: Point,
    theme: &ThemeResolver<'_>,
    compatible: Option<bool>,
) {
    let colour = match compatible {
        None => theme.edge_drag_preview_color(),
        Some(true) => theme.port_compatible_outline(),
        Some(false) => theme.edge_invalid_drag_color(),
    };
    draw_edge_curve(ctx, from, cursor, theme.edge_thickness(), colour);
}

/// Sample-and-stroke shared between final edges and drag previews.
///
/// **Why a segmented rect chain instead of `stroke_path`:** Blinc's
/// canvas-closure render pipeline currently only re-dispatches the
/// SDF primitive batch in the per-frame composite overlay pass;
/// tessellated path primitives (`stroke_path` / `fill_path` for non-
/// SDF cases) are emitted into a separate batch that the overlay
/// doesn't draw, so cubic strokes from canvas closures vanish on
/// every frame except a full repaint. We work around it by
/// sampling the cubic into short segments and routing each segment
/// through `fill_rect` (SDF batch), with a per-segment transform
/// stack to rotate the rect onto the segment vector. Sub-optimal
/// (more primitives, slight joint aliasing) but renders reliably.
///
/// Sample count of 24 keeps each segment ≤ ~10 px at typical
/// edge lengths, smoothing the curve enough that the joints
/// disappear visually. Raise it for very long edges if needed.
pub fn draw_edge_curve(
    ctx: &mut dyn DrawContext,
    from: Point,
    to: Point,
    thickness: f32,
    colour: Color,
) {
    draw_edge_curve_tinted(ctx, from, to, thickness, |_| colour);
}

/// Variant that lets the caller compute a per-segment colour from
/// the segment's midpoint t ∈ [0, 1] along the curve. Used by
/// state animations: pass a closure that brightens the segment
/// near the current flow position to produce a comet-tail look.
pub fn draw_edge_curve_tinted(
    ctx: &mut dyn DrawContext,
    from: Point,
    to: Point,
    thickness: f32,
    tint: impl Fn(f32) -> Color,
) {
    use crate::bezier::sample_cubic;
    let (c1, c2) = mid_x_controls(from, to);
    let steps = 24usize;
    let samples = sample_cubic(from, c1, c2, to, steps);
    let half = thickness * 0.5;
    for (i, pair) in samples.windows(2).enumerate() {
        let a = pair[0];
        let b = pair[1];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.01 {
            continue;
        }
        let angle = dy.atan2(dx);
        // t = midpoint of this segment along the curve.
        let t = (i as f32 + 0.5) / steps as f32;
        let brush = Brush::Solid(tint(t));
        ctx.push_transform(blinc_core::draw::Transform::translate(a.x, a.y));
        ctx.push_transform(blinc_core::draw::Transform::rotate(angle));
        ctx.fill_rect(
            Rect::new(0.0, -half, len, thickness),
            CornerRadius::uniform(half),
            brush,
        );
        ctx.pop_transform();
        ctx.pop_transform();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Group
// ─────────────────────────────────────────────────────────────────────

/// Draw a group chrome: tinted rounded-rect backdrop + header bar +
/// title + optional description + badge.
///
/// `auto_bounds` is the union of the group's members' bboxes (host-
/// computed and passed in). If the group has an explicit `bounds`,
/// that wins.
/// Border tint variant for [`draw_group`]'s outline. Picked by the
/// editor based on selection state + the live drag-into / drag-out
/// preview, then mapped to the appropriate theme token here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupBorderKind {
    /// Standard idle group border.
    #[default]
    Normal,
    /// Group is selected.
    Selected,
    /// Group is the live drag-into target — positive highlight.
    AddTarget,
    /// Group the dragged node is escaping (Shift held) — warning.
    RemoveTarget,
}

pub fn draw_group<G>(
    ctx: &mut dyn DrawContext,
    group: &Group<G>,
    auto_bounds: Rect,
    slots: &crate::slot::GroupSlots,
    theme: &ThemeResolver<'_>,
    is_selected: bool,
    border_kind: GroupBorderKind,
) -> Option<Rect> {
    // Soft-disable: same opacity push used by `draw_node_at` —
    // dims the group chrome (backdrop, border, header, badge,
    // chevron) as a single unit. Member nodes get their own dim
    // pass via the editor's `disabled_nodes` set, so the group +
    // its members all read as ghosted together without compound
    // alpha math.
    let pushed_opacity = group.disabled;
    if pushed_opacity {
        ctx.push_opacity(theme.node_disabled_alpha());
    }
    let pad = theme.group_padding();
    // When collapsed, the body shrinks to ONLY the header band —
    // member nodes stay where they are but no longer sit inside the
    // group container chrome (group reads as a floating chip above
    // its members). When expanded, the body wraps the full
    // members-union with padding + header.
    let body = group.bounds.unwrap_or_else(|| {
        if group.is_collapsed {
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
        }
    });
    let origin = Point::new(body.x(), body.y());
    let radius = theme.group_corner_radius();
    let cr_body = CornerRadius::uniform(radius);
    let title_size = theme.title_font_size();
    let subtitle_size = theme.subtitle_font_size();

    // Drop the theme group shadow under the backdrop.
    draw_shadow_stack(ctx, body, cr_body, &theme.group_shadow_stack());

    // Backdrop tint (overrides the theme's default if `group.tint`
    // is set).
    let tint = group.tint.unwrap_or_else(|| theme.group_tint());
    themed_fill_rect(ctx, body, cr_body, Brush::Solid(tint), theme);

    // Header bar. When the group is COLLAPSED, the body == the
    // header band, so the header needs all four corners rounded
    // to follow the body's outline — otherwise the header's
    // sharp bottom corners stick out past the body's rounded
    // bottom corners, producing visual artifacts at the edges.
    // When expanded, only the top two corners are rounded so the
    // header reads as a band welded to the body below.
    let header = translate_rect(slots.header, origin);
    let header_corners = if group.is_collapsed {
        CornerRadius::uniform(radius)
    } else {
        CornerRadius::new(radius, radius, 0.0, 0.0)
    };
    // `group.accent` (when set) overrides the theme's header chrome
    // — used by hosts that want to chrome-mark a group with a
    // specific accent (the subgraph-expansion container, "warning"
    // groups in observability dashboards, etc.) without affecting
    // the body fill or member nodes.
    let header_fill_color = group.accent.unwrap_or_else(|| theme.group_header_fill());
    themed_fill_rect(
        ctx,
        header,
        header_corners,
        Brush::Solid(header_fill_color),
        theme,
    );

    // Dashed border — gives groups a distinct "soft container"
    // read versus the solid border on nodes. The dash pattern
    // comes from the theme so hosts can override per-bundle.
    //
    // Border-kind precedence: drag-preview tints win over selection
    // so the user sees the most recent intent (about to add/remove)
    // while the gesture is in flight. `is_selected` falls through
    // when no preview hint is active. `group.accent` (when set)
    // overrides the theme default border colour but yields to
    // selection / drag-preview tints so the user still sees the
    // most-recent intent during a gesture.
    let border = match border_kind {
        GroupBorderKind::AddTarget => theme.group_add_target_border(),
        GroupBorderKind::RemoveTarget => theme.group_remove_target_border(),
        GroupBorderKind::Selected => theme.node_selected_outline(),
        GroupBorderKind::Normal => {
            if is_selected {
                theme.node_selected_outline()
            } else {
                group.accent.unwrap_or_else(|| theme.group_border())
            }
        }
    };
    let outlined = is_selected
        || matches!(
            border_kind,
            GroupBorderKind::AddTarget | GroupBorderKind::RemoveTarget
        );
    let (dash, dash_offset) = theme.group_border_dash();
    let stroke = Stroke::new(theme.group_border_width(outlined)).with_dash(dash, dash_offset);
    themed_stroke_rect(ctx, body, cr_body, &stroke, Brush::Solid(border), theme);

    // When `group.accent` is set, title + description + chrome
    // glyphs all switch to a contrasting foreground colour (dark
    // text on a light accent, light text on a dark accent) so the
    // chrome stays readable against the accent header. Falls back
    // to the theme's standard colours when no accent is configured.
    let accent_fg = group.accent.map(contrasting_foreground);
    let title_color = accent_fg.unwrap_or_else(|| theme.group_title_color());
    let subtitle_color = accent_fg.unwrap_or_else(|| theme.node_subtitle_color());

    // Title.
    let title_rect = translate_rect(slots.title, origin);
    let title_style = TextStyle::new(title_size)
        .with_color(title_color)
        .with_weight(FontWeight::Medium)
        .with_align(TextAlign::Left)
        .with_baseline(TextBaseline::Top);
    ctx.draw_text(
        &group.name,
        Point::new(title_rect.x(), title_rect.y()),
        &title_style,
    );

    // Description — kept visible in collapsed mode too so the
    // chip still reads as informative when the members are folded
    // away. (Taffy's header slot already reserves vertical room
    // for the description row, so the collapsed body height
    // (which we set to slots.header.height()) includes it.) When
    // `description` is None / empty but `description_placeholder`
    // is set, the placeholder paints in a dimmer colour as a hint.
    if let Some(desc_slot) = slots.description {
        let desc_rect = translate_rect(desc_slot, origin);
        let desc_text = group.description.as_deref().filter(|s| !s.is_empty());
        let (label, color) = match desc_text {
            Some(text) => (text, subtitle_color),
            None => match group.description_placeholder.as_deref() {
                Some(p) if !p.is_empty() => {
                    let mut c = subtitle_color;
                    c.a *= 0.5;
                    (p, c)
                }
                _ => ("", subtitle_color),
            },
        };
        if !label.is_empty() {
            let desc_style = TextStyle::new(subtitle_size)
                .with_color(color)
                .with_align(TextAlign::Left)
                .with_baseline(TextBaseline::Top);
            // Greedy word-wrap at the description slot's allocated
            // width. slot.rs::group_inputs_from pre-computed the
            // wrapped line count and grew the slot height to fit,
            // so each line lands inside the reserved rect without
            // overflowing the header into the body. Wrap width is
            // the slot's full width — taffy already accounted for
            // padding + chrome + badge by the time the slot was
            // sized.
            let desc_line_h = subtitle_size + 2.0;
            let lines = wrap_text(label, subtitle_size, desc_rect.width().max(1.0));
            for (i, line) in lines.iter().enumerate() {
                ctx.draw_text(
                    line,
                    Point::new(desc_rect.x(), desc_rect.y() + i as f32 * desc_line_h),
                    &desc_style,
                );
            }
        }
    }

    // Badge.
    let mut badge_rect: Option<Rect> = None;
    if let (Some(badge_data), Some(badge_slot)) = (&group.badge, slots.badge) {
        let chip = translate_rect(badge_slot, origin);
        draw_badge(ctx, badge_data, chip, theme);
        badge_rect = Some(chip);
    }
    if pushed_opacity {
        ctx.pop_opacity();
    }
    badge_rect
}

/// Rectangles for the three group-header chrome buttons in absolute
/// canvas-content coordinates. Returned by
/// [`draw_group_header_chrome`] so the caller can register matching
/// hit regions (`group_collapse:{id}` / `group_delete:{id}` /
/// `group_edit:{id}`).
#[derive(Debug, Clone, Copy)]
pub struct GroupChromeRects {
    pub collapse: Rect,
    pub delete: Rect,
    pub edit: Rect,
}

/// Paint the group-header chrome row at the right edge of the
/// header: edit (pencil) → delete (×) → collapse (chevron). Anchored
/// to `body` (the visible group body rect) + `slots.header`. Called
/// AFTER the header band is filled so the glyphs sit on top of the
/// band tint.
///
/// Returns the AABBs the caller should register as hit regions so
/// the editor's click handler can fire the matching events
/// (`EditorEvent::ToggleCollapseRequested` /
/// `EditorEvent::DeleteGroupRequested` /
/// `EditorEvent::EditGroupTitleRequested`).
///
/// `glyph_color_override` lets hosts switch the glyph + outline
/// colour to a contrasting tone when the header band carries a
/// non-default accent (see `Group::accent`). When `None`, the
/// usual `theme.group_title_color()` / `theme.group_border()`
/// pair is used. Pass a single colour for both pen-stroke and
/// outline so the chrome reads as one cluster.
pub fn draw_group_header_chrome<G>(
    ctx: &mut dyn DrawContext,
    group: &Group<G>,
    body: Rect,
    slots: &crate::slot::GroupSlots,
    theme: &ThemeResolver<'_>,
    glyph_color_override: Option<Color>,
) -> GroupChromeRects {
    let body_origin = Point::new(body.x(), body.y());
    let collapse_rect = translate_rect(slots.chrome_collapse, body_origin);
    let delete_rect = translate_rect(slots.chrome_delete, body_origin);
    let edit_rect = translate_rect(slots.chrome_edit, body_origin);

    let glyph = glyph_color_override.unwrap_or_else(|| theme.group_title_color());
    let outline = glyph_color_override.unwrap_or_else(|| theme.group_border());

    draw_chrome_button_outline_color(ctx, edit_rect, outline);
    draw_chrome_glyph_edit(ctx, edit_rect, glyph);

    draw_chrome_button_outline_color(ctx, delete_rect, outline);
    draw_chrome_glyph_delete(ctx, delete_rect, glyph);

    draw_chrome_button_outline_color(ctx, collapse_rect, outline);
    draw_chrome_glyph_collapse(ctx, collapse_rect, group.is_collapsed, glyph);

    GroupChromeRects {
        collapse: collapse_rect,
        delete: delete_rect,
        edit: edit_rect,
    }
}

/// Pick a foreground colour with adequate contrast against
/// `bg`. Uses the W3C relative-luminance formula
/// (0.299·R + 0.587·G + 0.114·B). Dark text on light bg, light on
/// dark. Sufficient for accent chrome text + glyphs; full WCAG
/// AA / AAA contrast is up to the host's accent picker.
fn contrasting_foreground(bg: Color) -> Color {
    let lum = 0.299 * bg.r + 0.587 * bg.g + 0.114 * bg.b;
    if lum > 0.55 {
        Color::rgba(0.08, 0.09, 0.12, 1.0)
    } else {
        Color::rgba(0.96, 0.96, 0.97, 1.0)
    }
}

/// Outline-only chip — same look across all three chrome buttons so
/// the affordance row reads as a single control cluster. No
/// background fill so the header band's tint shows through.
#[allow(dead_code)]
fn draw_chrome_button_outline(ctx: &mut dyn DrawContext, rect: Rect, theme: &ThemeResolver<'_>) {
    draw_chrome_button_outline_color(ctx, rect, theme.group_border());
}

/// Same as [`draw_chrome_button_outline`] but takes the colour
/// directly — used by [`draw_group_header_chrome`] to thread its
/// `glyph_color_override` through the outline.
fn draw_chrome_button_outline_color(ctx: &mut dyn DrawContext, rect: Rect, color: Color) {
    let btn_radius = (rect.width() * 0.22).max(3.0);
    let stroke = Stroke::new(1.0);
    ctx.stroke_rect(
        rect,
        CornerRadius::uniform(btn_radius),
        &stroke,
        Brush::Solid(color),
    );
}

/// Tabler outline paths (viewBox `0 0 24 24`) embedded directly so
/// the chrome glyphs render via the same SVG pipeline node icons use
/// — strokes get rounded caps and joins that close cleanly, instead
/// of the gap a pair of rotated `fill_rect` arms leaves at the apex.
/// Authored in the canonical Tabler form; if we ever take a direct
/// dep on `blinc_tabler_icons` we can replace these with the
/// generated path constants verbatim.
const TABLER_PENCIL: &str = r#"<path d="M4 20h4l10.5 -10.5a1.5 1.5 0 0 0 -4 -4l-10.5 10.5v4" /><line x1="13.5" y1="6.5" x2="17.5" y2="10.5" />"#;
const TABLER_X: &str =
    r#"<line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />"#;
const TABLER_CHEVRON_UP: &str = r#"<polyline points="6 15 12 9 18 15" />"#;
const TABLER_CHEVRON_DOWN: &str = r#"<polyline points="6 9 12 15 18 9" />"#;

/// Discriminator for [`chrome_glyph_doc`]'s cache. Two-byte enum so
/// the cache key stays small.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ChromeGlyph {
    Pencil,
    Close,
    ChevronUp,
    ChevronDown,
}

impl ChromeGlyph {
    fn path(self) -> &'static str {
        match self {
            Self::Pencil => TABLER_PENCIL,
            Self::Close => TABLER_X,
            Self::ChevronUp => TABLER_CHEVRON_UP,
            Self::ChevronDown => TABLER_CHEVRON_DOWN,
        }
    }
}

/// Cached parsed SVG keyed by `(glyph, quantised stroke colour)`.
/// `SvgDocument::from_str` parses through usvg + tiny_skia_path on
/// every call, so reparsing 3 chips × N groups per frame would be
/// wasteful. Quantising the colour to 8 bits per channel collapses
/// the cache to one entry per (glyph, theme) — exactly what the
/// scene needs.
fn chrome_glyph_doc(glyph: ChromeGlyph, stroke: Color) -> Arc<blinc_svg::SvgDocument> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    // (glyph kind, quantised stroke RGB)
    type CacheKey = (ChromeGlyph, [u8; 3]);
    type ChromeCache = HashMap<CacheKey, Arc<blinc_svg::SvgDocument>>;
    static CACHE: OnceLock<Mutex<ChromeCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let rgb = [
        (stroke.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (stroke.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (stroke.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    ];
    let key = (glyph, rgb);
    {
        let guard = cache.lock().unwrap();
        if let Some(doc) = guard.get(&key) {
            return doc.clone();
        }
    }
    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="rgb({},{},{})" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{}</svg>"#,
        rgb[0],
        rgb[1],
        rgb[2],
        glyph.path()
    );
    let doc = Arc::new(
        blinc_svg::SvgDocument::from_str(&svg).expect("embedded tabler markup is well-formed"),
    );
    cache.lock().unwrap().insert(key, doc.clone());
    doc
}

/// Render a tabler-shaped chrome glyph centred inside `rect`. The
/// glyph is inset by ~22% so the icon doesn't crowd the chip's
/// outline.
fn draw_chrome_tabler_glyph(
    ctx: &mut dyn DrawContext,
    rect: Rect,
    glyph: ChromeGlyph,
    stroke: Color,
) {
    let inset = rect.width() * 0.22;
    let glyph_rect = Rect::new(
        rect.x() + inset,
        rect.y() + inset,
        rect.width() - inset * 2.0,
        rect.height() - inset * 2.0,
    );
    chrome_glyph_doc(glyph, stroke).render_fit(ctx, glyph_rect);
}

/// Chevron glyph: tabler `chevron-up` when expanded ("click to
/// collapse"), `chevron-down` when collapsed ("click to expand").
fn draw_chrome_glyph_collapse(
    ctx: &mut dyn DrawContext,
    rect: Rect,
    is_collapsed: bool,
    color: Color,
) {
    let glyph = if is_collapsed {
        ChromeGlyph::ChevronDown
    } else {
        ChromeGlyph::ChevronUp
    };
    draw_chrome_tabler_glyph(ctx, rect, glyph, color);
}

/// `×` glyph for the delete chip — tabler `x`.
fn draw_chrome_glyph_delete(ctx: &mut dyn DrawContext, rect: Rect, color: Color) {
    draw_chrome_tabler_glyph(ctx, rect, ChromeGlyph::Close, color);
}

/// Pencil glyph for the edit chip — tabler `pencil`.
fn draw_chrome_glyph_edit(ctx: &mut dyn DrawContext, rect: Rect, color: Color) {
    draw_chrome_tabler_glyph(ctx, rect, ChromeGlyph::Pencil, color);
}

// ─────────────────────────────────────────────────────────────────────
// Badge — used by both nodes and groups
// ─────────────────────────────────────────────────────────────────────

/// Render a tooltip chip next to a status badge whose `tooltip` field
/// is set. Anchors below the badge by default; flips above if doing
/// so would clip the chip out of `viewport` (when supplied). Word-
/// wraps long tooltip text at a max line width and clamps the chip
/// horizontally into the viewport using the same shared
/// [`estimate_text_width`] + [`wrap_text`] helpers as the port
/// tooltip path so all tooltip chrome agrees on metrics.
pub fn draw_badge_tooltip(
    ctx: &mut dyn DrawContext,
    text_label: &str,
    badge_rect: Rect,
    theme: &ThemeResolver<'_>,
    viewport: Option<Rect>,
    canvas_view: blinc_core::layer::Affine2D,
) {
    if text_label.is_empty() {
        return;
    }
    // Move badge_rect into screen space — the canvas viewport
    // transform will be cancelled below so chip math (and the clamp
    // against `viewport`'s screen-space rect) needs the anchor at
    // its on-screen position. Use the centre of the screen-space
    // badge rect for the horizontal anchor; for top/bottom we use
    // the bounds' edges as projected through the affine. For the
    // simple scale+translate viewport canvas-kit emits, the rect's
    // shape is preserved (no rotation), so taking opposite corners
    // and recombining is correct.
    let screen_tl = canvas_view.transform_point(Point::new(badge_rect.x(), badge_rect.y()));
    let screen_br = canvas_view.transform_point(Point::new(
        badge_rect.x() + badge_rect.width(),
        badge_rect.y() + badge_rect.height(),
    ));
    let badge_screen_x = screen_tl.x.min(screen_br.x);
    let badge_screen_y = screen_tl.y.min(screen_br.y);
    let badge_screen_w = (screen_br.x - screen_tl.x).abs();
    let badge_screen_h = (screen_br.y - screen_tl.y).abs();

    // Padding kept generous (10/7) so the real-text-measurer's
    // small rounding doesn't push the glyphs against the chip border.
    let pad_x = 10.0_f32;
    let pad_y = 7.0_f32;
    let font = 11.0_f32;
    let line_h = font + 2.0;
    let max_content_w: f32 = 200.0;
    let lines = wrap_text(text_label, font, max_content_w);
    let content_w = lines
        .iter()
        .map(|l| estimate_text_width(l, font))
        .fold(0.0_f32, f32::max);
    let chip_w = content_w + pad_x * 2.0;
    let chip_h = lines.len() as f32 * line_h + pad_y * 2.0 - (line_h - font);
    // Anchor below the badge by default with a 6 px gap. Centre the
    // chip horizontally on the badge so a small dot/triangle badge
    // doesn't pull the tooltip off to one side.
    let gap = 6.0_f32;
    let badge_centre_x = badge_screen_x + badge_screen_w * 0.5;
    let mut chip_x = badge_centre_x - chip_w * 0.5;
    let mut chip_y = badge_screen_y + badge_screen_h + gap;
    if let Some(vp) = viewport {
        let inset = 4.0_f32;
        if chip_y + chip_h > vp.y() + vp.height() - inset {
            // Flip above the badge.
            chip_y = badge_screen_y - gap - chip_h;
        }
        if chip_x + chip_w > vp.x() + vp.width() - inset {
            chip_x = vp.x() + vp.width() - inset - chip_w;
        }
        if chip_x < vp.x() + inset {
            chip_x = vp.x() + inset;
        }
        if chip_y < vp.y() + inset {
            chip_y = vp.y() + inset;
        }
    }
    let chip = Rect::new(chip_x, chip_y, chip_w, chip_h);
    let cr = CornerRadius::uniform(5.0);

    // Push inverse-viewport so the chip body draws in screen-pixel
    // logical units (see draw_port_tooltip_clamped for the full
    // rationale). DPI scale is preserved; only the canvas zoom is
    // cancelled.
    let inverse = blinc_canvas_kit::affine_inverse(&canvas_view)
        .unwrap_or(blinc_core::layer::Affine2D::IDENTITY);
    ctx.push_transform(blinc_core::draw::Transform::Affine2D(inverse));

    ctx.draw_shadow(chip, cr, theme.port_shadow());
    ctx.fill_rect(chip, cr, Brush::Solid(theme.tooltip_bg()));
    let stroke = Stroke::new(1.0);
    ctx.stroke_rect(chip, cr, &stroke, Brush::Solid(theme.tooltip_border()));
    let style = TextStyle::new(font)
        .with_color(theme.tooltip_text())
        .with_align(TextAlign::Left)
        .with_baseline(TextBaseline::Top);
    for (i, line) in lines.iter().enumerate() {
        ctx.draw_text(
            line,
            Point::new(chip_x + pad_x, chip_y + pad_y + i as f32 * line_h),
            &style,
        );
    }
    ctx.pop_transform();
}

/// Draw a status badge inside `chip_rect`. Fills the chip with the
/// kind's colour; if the badge has a count, renders it as small
/// white-on-tint text.
pub fn draw_badge(
    ctx: &mut dyn DrawContext,
    badge: &StatusBadge,
    chip_rect: Rect,
    theme: &ThemeResolver<'_>,
) {
    let colour = theme.badge_color(badge.kind);
    let radius = (chip_rect.height() * 0.5).min(chip_rect.width() * 0.5);

    themed_fill_rect(
        ctx,
        chip_rect,
        CornerRadius::uniform(radius),
        Brush::Solid(colour),
        theme,
    );

    if let Some(count) = badge.count {
        let label = if count > 99 {
            "99+".to_string()
        } else {
            count.to_string()
        };
        let text_style = TextStyle::new((chip_rect.height() * 0.65).max(8.0))
            .with_color(badge_text_color(badge.kind))
            .with_weight(FontWeight::Medium)
            .with_align(TextAlign::Center)
            .with_baseline(TextBaseline::Middle);
        ctx.draw_text(
            &label,
            Point::new(
                chip_rect.x() + chip_rect.width() * 0.5,
                chip_rect.y() + chip_rect.height() * 0.5,
            ),
            &text_style,
        );
    }
}

fn badge_text_color(kind: BadgeKind) -> Color {
    match kind {
        BadgeKind::Info | BadgeKind::Running => Color::rgb(0.04, 0.06, 0.10),
        BadgeKind::Warning => Color::rgb(0.10, 0.06, 0.00),
        BadgeKind::Error | BadgeKind::Success => Color::rgb(1.0, 1.0, 1.0),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bezier midpoint helper — exported for callers that want to anchor
// edge labels on the curve.
// ─────────────────────────────────────────────────────────────────────

/// Midpoint of a default-routed cubic between `from` and `to`. Useful
/// for placing edge labels or contextual chrome (delete button on
/// hover, etc.) at the geometric centre of the curve.
pub fn edge_midpoint(from: Point, to: Point) -> Point {
    let (c1, c2) = mid_x_controls(from, to);
    cubic_point(0.5, from, c1, c2, to)
}

/// Render an SVG-backed [`crate::icon::NodeIcon`] inside the given
/// rect. Centres + scales the SVG via `SvgDocument::render_fit`,
/// which preserves aspect ratio and emits `fill_path` /
/// `stroke_path` (both safe inside canvas closures).
fn draw_icon(ctx: &mut dyn DrawContext, icon: &crate::icon::NodeIcon, rect: Rect) {
    icon.document().render_fit(ctx, rect);
}
