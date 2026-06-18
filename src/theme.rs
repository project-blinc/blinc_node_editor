//! Theme layer — derives editor colours / radii / spacing from
//! [`blinc_theme::ThemeState`], with optional editor-specific
//! overrides.
//!
//! ## Why two-layer
//!
//! The editor renders chrome (nodes, ports, edges, groups, badges)
//! that needs to match the host app's overall theme automatically —
//! a Blinc app shouldn't need to thread its theme through to the
//! node editor by hand. So the editor's defaults derive from
//! [`blinc_theme::ThemeState`] tokens (colours, radii, spacing).
//!
//! On top, [`NodeEditorTheme`] is an optional override layer for
//! editor-specific *semantic* colours that `ThemeState` doesn't
//! name: drag-preview edge tint, port-validation highlight, group
//! tint, badge accents. Hosts customise these without re-skinning
//! the whole app.
//!
//! ## Resolution order (per token)
//!
//! 1. Explicit override on the `NodeEditorTheme` instance.
//! 2. A semantic mapping into a `ThemeState` token if one fits
//!    (e.g. `node_body_fill` → `ColorToken::Surface`).
//! 3. Hardcoded fallback (only when `ThemeState` is uninitialised
//!    — primarily for tests / headless builds).

use blinc_core::layer::Color;
use blinc_theme::tokens::{ColorToken, RadiusToken, ShadowToken, SpacingToken};
use blinc_theme::ThemeState;

use crate::connection::ConnectionState;
use crate::group::BadgeKind;

// ─────────────────────────────────────────────────────────────────────
// NodeEditorTheme — editor-specific override layer
// ─────────────────────────────────────────────────────────────────────

/// Editor-specific theme overrides. Every field is `Option` —
/// `None` means "derive from `blinc_theme::ThemeState`." Pass a
/// `NodeEditorTheme::default()` to inherit fully; set individual
/// fields to override.
///
/// Hosts that want a totally custom palette (decoupled from the
/// app theme) can `set_*` every field; conversely an app that just
/// wants the editor to "look like the rest of the UI" passes the
/// default.
#[derive(Debug, Clone, Default)]
pub struct NodeEditorTheme {
    // ─── Node chrome ───
    pub node_body_fill: Option<Color>,
    pub node_body_stroke: Option<Color>,
    pub node_header_fill: Option<Color>,
    /// Override the fill used for a node's content-slot inset. By
    /// default the slot pulls `ColorToken::Background` so the slot
    /// reads as a recess down to the workspace surface, matching
    /// the canvas dot-pattern bg behind the node. Override to
    /// `Some(c)` if a host wants a custom slot chrome.
    pub content_slot_fill: Option<Color>,
    /// Override the 1px outline stroke painted around the content
    /// slot inset. Defaults to `ColorToken::Border` for a subtle
    /// border that delineates the slot from the node body on light
    /// schemes where fill alone may not carry enough contrast.
    pub content_slot_border: Option<Color>,
    pub node_title_color: Option<Color>,
    pub node_subtitle_color: Option<Color>,
    pub node_selected_outline: Option<Color>,
    pub node_corner_radius: Option<f32>,
    /// Default node width when a `NodeInstance.size` is `None`.
    pub default_node_width: Option<f32>,
    /// Default node height when a `NodeInstance.size` is `None`.
    pub default_node_height: Option<f32>,

    // ─── Port chrome ───
    pub port_radius: Option<f32>,
    pub port_stroke: Option<Color>,
    pub port_hover_outline: Option<Color>,
    pub port_compatible_outline: Option<Color>,
    pub port_incompatible_outline: Option<Color>,

    // ─── Edge / connection ───
    pub edge_default_color: Option<Color>,
    pub edge_pending_color: Option<Color>,
    pub edge_running_color: Option<Color>,
    pub edge_success_color: Option<Color>,
    pub edge_warning_color: Option<Color>,
    pub edge_error_color: Option<Color>,
    pub edge_drag_preview_color: Option<Color>,
    pub edge_invalid_drag_color: Option<Color>,
    pub edge_selected_outline: Option<Color>,
    pub edge_thickness: Option<f32>,

    // ─── Edge delete button (rendered when an edge is selected) ───
    pub edge_delete_button_fill: Option<Color>,
    pub edge_delete_button_glyph: Option<Color>,
    pub edge_delete_button_radius: Option<f32>,

    // ─── Group chrome ───
    pub group_tint: Option<Color>,
    pub group_border: Option<Color>,
    /// Border tint shown when a drag would ADD the dragged node to
    /// this group (positive feedback). Defaults to `Primary`.
    pub group_add_target_border: Option<Color>,
    /// Border tint shown when a Shift-drag would REMOVE the dragged
    /// node from this group (warning feedback). Defaults to
    /// `Warning`.
    pub group_remove_target_border: Option<Color>,
    pub group_header_fill: Option<Color>,
    pub group_title_color: Option<Color>,
    pub group_padding: Option<f32>,
    pub group_corner_radius: Option<f32>,

    // ─── Badge accents ───
    pub badge_info_color: Option<Color>,
    pub badge_warning_color: Option<Color>,
    pub badge_error_color: Option<Color>,
    pub badge_success_color: Option<Color>,
    pub badge_running_color: Option<Color>,

    // ─── Boundary (exposed-port) chrome ───
    pub boundary_port_color: Option<Color>,
    pub boundary_port_unconnected_color: Option<Color>,
}

// ─────────────────────────────────────────────────────────────────────
// ThemeResolver — token resolution against ThemeState + overrides
// ─────────────────────────────────────────────────────────────────────

/// Resolves theme tokens for the renderer. Wraps a
/// [`NodeEditorTheme`] (overrides) and resolves each token by:
/// (1) checking the override, (2) reading from [`ThemeState`], (3)
/// falling back to a hardcoded default if no theme is initialised.
///
/// Stateless beyond its constructor — cheap to construct on every
/// frame. Hosts that want a stable resolver hold an
/// `Arc<ThemeResolver>` in their editor wrapper.
pub struct ThemeResolver<'a> {
    overrides: &'a NodeEditorTheme,
}

impl<'a> ThemeResolver<'a> {
    pub fn new(overrides: &'a NodeEditorTheme) -> Self {
        Self { overrides }
    }

    // ─── Helpers — sampling ThemeState safely ────────────────────────

    fn theme_color(token: ColorToken, fallback: Color) -> Color {
        ThemeState::try_get()
            .map(|t| t.color(token))
            .unwrap_or(fallback)
    }

    fn theme_radius(token: RadiusToken, fallback: f32) -> f32 {
        ThemeState::try_get()
            .map(|t| t.radius(token))
            .unwrap_or(fallback)
    }

    #[allow(dead_code)]
    fn theme_spacing(token: SpacingToken, fallback: f32) -> f32 {
        ThemeState::try_get()
            .map(|t| t.spacing_value(token))
            .unwrap_or(fallback)
    }

    // ─── Node chrome ─────────────────────────────────────────────────

    /// Nodes are visually elevated cards sitting on the workspace
    /// (which uses `ColorToken::Background`). `SurfaceElevated`
    /// matches that elevation tier — `Surface` would tonally
    /// merge with the workspace on some bundles.
    pub fn node_body_fill(&self) -> Color {
        self.overrides.node_body_fill.unwrap_or_else(|| {
            Self::theme_color(ColorToken::SurfaceElevated, Color::rgb(0.16, 0.18, 0.22))
        })
    }

    pub fn node_body_stroke(&self) -> Color {
        self.overrides
            .node_body_stroke
            .unwrap_or_else(|| Self::theme_color(ColorToken::Border, Color::rgb(0.25, 0.27, 0.32)))
    }

    /// Fill the content-slot inset uses inside a node body.
    ///
    /// Defaults to `ColorToken::Background` so the slot reads as a
    /// recess sunk down to the workspace surface — exactly the
    /// canvas dot-pattern bg the node sits on. This keeps the slot
    /// chrome consistent with the workspace regardless of how the
    /// host theme colours the node body itself.
    pub fn content_slot_fill(&self) -> Color {
        self.overrides.content_slot_fill.unwrap_or_else(|| {
            Self::theme_color(ColorToken::Background, Color::rgb(0.04, 0.05, 0.07))
        })
    }

    /// 1px outline stroke around the content-slot inset. Defaults
    /// to `ColorToken::Border` — subtle but enough to read on light
    /// schemes where slot bg and node body sit close in luminance.
    pub fn content_slot_border(&self) -> Color {
        self.overrides
            .content_slot_border
            .unwrap_or_else(|| Self::theme_color(ColorToken::Border, Color::rgb(0.25, 0.27, 0.32)))
    }

    /// Body fill for subgraph navigation nodes (Diamond chrome). Tinted
    /// toward the theme's `Warning` token so they read clearly against
    /// regular `Surface`-fill rectangle nodes — mirrors Zeal's
    /// `orange-600` convention for subgraph instances. Falls back to a
    /// warm accent when the theme has no Warning token configured.
    pub fn node_subgraph_fill(&self) -> Color {
        // The Warning token is tuned bright (orange-500 / amber-500-ish)
        // for icon + edge accents. Diamond body fill needs a tint, not
        // a saturated alarm, so we remap it against the active
        // colour scheme:
        //   • Dark mode → desaturate toward a deep brown that sits one
        //     elevation tier above the canvas bg.
        //   • Light mode → desaturate toward a pale peach (orange-100ish)
        //     so the diamond reads as a soft accent against the lighter
        //     surface instead of the previous dark-mode mud.
        // Without this scheme branch the same RGBA math produced a
        // muddy brown in light mode that visually clashed with the rest
        // of the canvas.
        let base = Self::theme_color(ColorToken::Warning, Color::rgb(0.96, 0.70, 0.32));
        let dark = matches!(
            blinc_theme::ThemeState::get().scheme(),
            blinc_theme::ColorScheme::Dark
        );
        if dark {
            Color::rgba(
                base.r * 0.45 + 0.10,
                base.g * 0.40 + 0.10,
                base.b * 0.35 + 0.08,
                1.0,
            )
        } else {
            // Light-mode tint: keep the hue, blow up the lightness
            // by mixing heavily with white. Equivalent to ~88%
            // white / 12% warning — soft enough to coexist with the
            // brighter Surface canvas yet still recognisably warm.
            Color::rgba(
                base.r * 0.18 + 0.82,
                base.g * 0.18 + 0.78,
                base.b * 0.18 + 0.74,
                1.0,
            )
        }
    }

    /// Body stroke for subgraph navigation nodes — uses the full
    /// `Warning` accent so the diamond outline reads as a deliberate
    /// "this is a navigable subgraph" affordance.
    pub fn node_subgraph_stroke(&self) -> Color {
        Self::theme_color(ColorToken::Warning, Color::rgb(0.96, 0.70, 0.32))
    }

    /// Header sits one elevation tier above the body so the chrome
    /// reads at a glance. `SurfaceOverlay` matches the "popover /
    /// header band" tier on Universal HID and most platform bundles.
    pub fn node_header_fill(&self) -> Color {
        self.overrides.node_header_fill.unwrap_or_else(|| {
            Self::theme_color(ColorToken::SurfaceOverlay, Color::rgb(0.20, 0.22, 0.28))
        })
    }

    /// Group borders are dashed so they read as a "soft container"
    /// hint distinct from nodes' solid borders. Returns
    /// `(dash_lengths, dash_offset)` for `Stroke::with_dash`.
    /// The 6/4 pattern reads at typical zoom levels; renderers
    /// that want zoom-adaptive dashing divide by zoom themselves.
    pub fn group_border_dash(&self) -> (Vec<f32>, f32) {
        (vec![6.0, 4.0], 0.0)
    }

    /// Stroke width for the group border. Slightly thicker than a
    /// node border so the container reads as a heavier visual
    /// grouping even when the dash + tint are subtle.
    pub fn group_border_width(&self, is_selected: bool) -> f32 {
        if is_selected {
            2.0
        } else {
            1.5
        }
    }

    pub fn node_title_color(&self) -> Color {
        self.overrides.node_title_color.unwrap_or_else(|| {
            Self::theme_color(ColorToken::TextPrimary, Color::rgb(0.95, 0.95, 0.96))
        })
    }

    pub fn node_subtitle_color(&self) -> Color {
        self.overrides.node_subtitle_color.unwrap_or_else(|| {
            Self::theme_color(ColorToken::TextSecondary, Color::rgb(0.65, 0.67, 0.72))
        })
    }

    pub fn node_selected_outline(&self) -> Color {
        self.overrides
            .node_selected_outline
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    pub fn node_corner_radius(&self) -> f32 {
        self.overrides
            .node_corner_radius
            .unwrap_or_else(|| Self::theme_radius(RadiusToken::Md, 8.0))
    }

    /// Opacity multiplier applied to every primitive a disabled node
    /// emits — body fill, border, icon, title, badge. 0.45 reads as
    /// "ghosted, not invisible": the node is clearly distinct from
    /// its active siblings but the structure (which port connects
    /// to what) stays legible at a glance.
    pub fn node_disabled_alpha(&self) -> f32 {
        0.45
    }

    pub fn default_node_size(&self) -> (f32, f32) {
        let w = self.overrides.default_node_width.unwrap_or(180.0);
        let h = self.overrides.default_node_height.unwrap_or(72.0);
        (w, h)
    }

    // ─── Port chrome ─────────────────────────────────────────────────

    pub fn port_radius(&self) -> f32 {
        // Default 4 px — tighter Zeal-style dot. 5/6 px reads
        // chunky next to a 36 px icon. Hosts can override via
        // `NodeEditorTheme.port_radius`.
        self.overrides.port_radius.unwrap_or(4.0)
    }

    pub fn port_stroke(&self) -> Color {
        self.overrides
            .port_stroke
            .unwrap_or_else(|| Self::theme_color(ColorToken::Border, Color::rgb(0.30, 0.32, 0.38)))
    }

    pub fn port_hover_outline(&self) -> Color {
        self.overrides
            .port_hover_outline
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    pub fn port_compatible_outline(&self) -> Color {
        self.overrides
            .port_compatible_outline
            .unwrap_or(Color::rgb(0.30, 0.85, 0.40))
    }

    pub fn port_incompatible_outline(&self) -> Color {
        self.overrides
            .port_incompatible_outline
            .unwrap_or(Color::rgb(0.90, 0.30, 0.30))
    }

    // ─── Edge / connection ───────────────────────────────────────────

    /// Resolve an edge colour by runtime [`ConnectionState`]. Used by
    /// the renderer to colour-code edges at-a-glance per Zeal's
    /// pattern.
    pub fn edge_color_for_state(&self, state: ConnectionState) -> Color {
        match state {
            ConnectionState::None => self.edge_default_color(),
            ConnectionState::Pending => self.edge_pending_color(),
            ConnectionState::Warning => self.edge_warning_color(),
            ConnectionState::Error => self.edge_error_color(),
            ConnectionState::Success => self.edge_success_color(),
            ConnectionState::Running => self.edge_running_color(),
        }
    }

    pub fn edge_default_color(&self) -> Color {
        self.overrides.edge_default_color.unwrap_or_else(|| {
            Self::theme_color(ColorToken::TextSecondary, Color::rgb(0.55, 0.60, 0.68))
        })
    }

    pub fn edge_pending_color(&self) -> Color {
        self.overrides
            .edge_pending_color
            .unwrap_or(Color::rgba(0.55, 0.60, 0.68, 0.55))
    }

    pub fn edge_running_color(&self) -> Color {
        self.overrides
            .edge_running_color
            .unwrap_or(Color::rgb(0.40, 0.75, 1.00))
    }

    pub fn edge_success_color(&self) -> Color {
        self.overrides
            .edge_success_color
            .unwrap_or_else(|| Self::theme_color(ColorToken::Success, Color::rgb(0.30, 0.85, 0.40)))
    }

    pub fn edge_warning_color(&self) -> Color {
        self.overrides
            .edge_warning_color
            .unwrap_or_else(|| Self::theme_color(ColorToken::Warning, Color::rgb(0.95, 0.70, 0.20)))
    }

    pub fn edge_error_color(&self) -> Color {
        self.overrides
            .edge_error_color
            .unwrap_or_else(|| Self::theme_color(ColorToken::Error, Color::rgb(0.90, 0.30, 0.30)))
    }

    pub fn edge_drag_preview_color(&self) -> Color {
        self.overrides
            .edge_drag_preview_color
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    pub fn edge_invalid_drag_color(&self) -> Color {
        self.overrides
            .edge_invalid_drag_color
            .unwrap_or(Color::rgb(0.90, 0.30, 0.30))
    }

    pub fn edge_selected_outline(&self) -> Color {
        self.overrides
            .edge_selected_outline
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    pub fn edge_thickness(&self) -> f32 {
        self.overrides.edge_thickness.unwrap_or(2.0)
    }

    pub fn edge_delete_button_fill(&self) -> Color {
        self.overrides
            .edge_delete_button_fill
            .unwrap_or_else(|| Self::theme_color(ColorToken::Error, Color::rgb(0.85, 0.30, 0.30)))
    }

    pub fn edge_delete_button_glyph(&self) -> Color {
        self.overrides
            .edge_delete_button_glyph
            .unwrap_or_else(|| Self::theme_color(ColorToken::TextInverse, Color::WHITE))
    }

    pub fn edge_delete_button_radius(&self) -> f32 {
        self.overrides.edge_delete_button_radius.unwrap_or(9.0)
    }

    // ─── Port tooltip ─────────────────────────────────────────────

    pub fn tooltip_bg(&self) -> Color {
        Self::theme_color(
            ColorToken::SurfaceOverlay,
            Color::rgba(0.10, 0.11, 0.14, 0.96),
        )
    }

    pub fn tooltip_border(&self) -> Color {
        Self::theme_color(ColorToken::Border, Color::rgba(0.60, 0.62, 0.68, 0.45))
    }

    pub fn tooltip_text(&self) -> Color {
        Self::theme_color(ColorToken::TextPrimary, Color::rgb(0.95, 0.95, 0.96))
    }

    pub fn tooltip_text_secondary(&self) -> Color {
        Self::theme_color(ColorToken::TextSecondary, Color::rgb(0.70, 0.72, 0.78))
    }

    // ─── Inline port label ────────────────────────────────────────

    /// Colour of the port-name text drawn beside each port. Picks
    /// the theme's secondary text tint so the label reads as a
    /// supporting annotation rather than competing with the node
    /// title.
    pub fn port_label_color(&self) -> Color {
        Self::theme_color(ColorToken::TextSecondary, Color::rgb(0.70, 0.72, 0.78))
    }

    /// Font size for the inline port label. ~11px keeps the label
    /// readable without crowding the node body.
    pub fn port_label_font_size(&self) -> f32 {
        11.0
    }

    // ─── Group chrome ────────────────────────────────────────────────

    /// Group bodies are translucent containers — workspace dots +
    /// surface should read through them so the group reads as a
    /// "hint of grouping," not a solid card competing with the
    /// nodes on top. Pulls the chroma from `Surface` and overrides
    /// the alpha to keep it subtle.
    pub fn group_tint(&self) -> Color {
        if let Some(c) = self.overrides.group_tint {
            return c;
        }
        let base = Self::theme_color(ColorToken::Surface, Color::rgb(0.10, 0.11, 0.14));
        Color::rgba(base.r, base.g, base.b, 0.18)
    }

    pub fn group_border(&self) -> Color {
        self.overrides.group_border.unwrap_or_else(|| {
            Self::theme_color(
                ColorToken::BorderSecondary,
                Color::rgba(0.60, 0.62, 0.68, 0.45),
            )
        })
    }

    /// Border tint used while a drag-in-progress would ADD the
    /// dragged node to this group. Positive feedback — the
    /// renderer swaps `group_border()` for this on the live
    /// add-target group.
    pub fn group_add_target_border(&self) -> Color {
        self.overrides
            .group_add_target_border
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    /// Border tint used while a Shift-drag would REMOVE the
    /// dragged node from this group. Warning feedback — distinct
    /// from `Error` so the user reads it as "leaving" not
    /// "broken".
    pub fn group_remove_target_border(&self) -> Color {
        self.overrides
            .group_remove_target_border
            .unwrap_or_else(|| Self::theme_color(ColorToken::Warning, Color::rgb(0.95, 0.65, 0.30)))
    }

    /// Group header reads as a band one tier above the group body —
    /// `SurfaceElevated` matches the elevation between `Surface`
    /// (group body) and `SurfaceOverlay` (node header).
    pub fn group_header_fill(&self) -> Color {
        self.overrides.group_header_fill.unwrap_or_else(|| {
            Self::theme_color(ColorToken::SurfaceElevated, Color::rgb(0.14, 0.16, 0.20))
        })
    }

    pub fn group_title_color(&self) -> Color {
        self.overrides.group_title_color.unwrap_or_else(|| {
            Self::theme_color(ColorToken::TextPrimary, Color::rgb(0.95, 0.95, 0.96))
        })
    }

    pub fn group_padding(&self) -> f32 {
        self.overrides.group_padding.unwrap_or(24.0)
    }

    pub fn group_corner_radius(&self) -> f32 {
        self.overrides
            .group_corner_radius
            .unwrap_or_else(|| Self::theme_radius(RadiusToken::Lg, 12.0))
    }

    // ─── Badges ──────────────────────────────────────────────────────

    pub fn badge_color(&self, kind: BadgeKind) -> Color {
        match kind {
            BadgeKind::Info => self.overrides.badge_info_color.unwrap_or_else(|| {
                Self::theme_color(ColorToken::Info, Color::rgb(0.50, 0.70, 0.95))
            }),
            BadgeKind::Warning => self.overrides.badge_warning_color.unwrap_or_else(|| {
                Self::theme_color(ColorToken::Warning, Color::rgb(0.95, 0.70, 0.20))
            }),
            BadgeKind::Error => self.overrides.badge_error_color.unwrap_or_else(|| {
                Self::theme_color(ColorToken::Error, Color::rgb(0.90, 0.30, 0.30))
            }),
            BadgeKind::Success => self.overrides.badge_success_color.unwrap_or_else(|| {
                Self::theme_color(ColorToken::Success, Color::rgb(0.30, 0.85, 0.40))
            }),
            BadgeKind::Running => self
                .overrides
                .badge_running_color
                .unwrap_or(Color::rgb(0.40, 0.75, 1.00)),
        }
    }

    // ─── Boundary (exposed-port) chrome ─────────────────────────────

    pub fn boundary_port_color(&self) -> Color {
        self.overrides
            .boundary_port_color
            .unwrap_or_else(|| Self::theme_color(ColorToken::Primary, Color::rgb(0.40, 0.65, 1.00)))
    }

    pub fn boundary_port_unconnected_color(&self) -> Color {
        self.overrides
            .boundary_port_unconnected_color
            .unwrap_or(Color::rgba(0.55, 0.60, 0.68, 0.45))
    }

    // ─── Shape tokens (continuous-curvature corners) ────────────────
    //
    // The renderer calls `DrawContext::set_corner_shape([n, n, n, n])`
    // before each `fill_rect` / `stroke_rect` using the value
    // returned here, so node bodies / group chrome / badges adopt
    // the active theme's squircle profile automatically. Themes that
    // don't opt into squircle rendering (Catppuccin, platform
    // bundles) leave `ShapeTokens` at its off default, in which
    // case we return `1.0` (round) and the GPU draws normal arcs.

    /// Effective superellipse `n` for a given corner radius, in the
    /// GPU's shape-field encoding (`1.0` = round, `2.0` = classic
    /// squircle). Returns `1.0` when the active theme's
    /// `ShapeTokens` is off OR when `radius` is below the theme's
    /// `smoothing_threshold` (squircle subtlety is imperceptible at
    /// small radii).
    pub fn corner_shape_n(&self, radius: f32) -> f32 {
        let Some(state) = ThemeState::try_get() else {
            return 1.0;
        };
        let shape = state.shape();
        if shape.is_off() || radius < shape.smoothing_threshold {
            1.0
        } else {
            shape.effective_corner_n()
        }
    }

    /// `corner_shape_n` lifted to the 4-element `[tl, tr, br, bl]`
    /// array expected by [`DrawContext::set_corner_shape`].
    pub fn corner_shape_uniform(&self, radius: f32) -> [f32; 4] {
        let n = self.corner_shape_n(radius);
        [n, n, n, n]
    }

    // ─── Shadow tokens ──────────────────────────────────────────────
    //
    // `ShadowTokens` ships a stack (`Vec<Shadow>`) per slot — the
    // Universal HID variants layer 2-3 shadows for depth. The
    // renderer iterates the stack and dispatches one
    // `DrawContext::draw_shadow` per layer.

    /// Node-body shadow stack. Nodes must read as **elevated cards**
    /// floating above the workspace — `ShadowToken::Md` is the
    /// minimum elevation that reliably reads across both light and
    /// dark bundles. Returns an empty `Vec` when the theme isn't
    /// initialised (tests / headless).
    pub fn node_shadow_stack(&self) -> Vec<blinc_theme::Shadow> {
        ThemeState::try_get()
            .map(|s| s.shadows().get(ShadowToken::Md).to_vec())
            .unwrap_or_default()
    }

    /// Group-backdrop shadow stack. Groups sit BENEATH nodes so
    /// their shadow should be lighter than a node's — `Sm` keeps
    /// the layering hierarchy intact (nodes still pop above the
    /// group footprint).
    pub fn group_shadow_stack(&self) -> Vec<blinc_theme::Shadow> {
        ThemeState::try_get()
            .map(|s| s.shadows().get(ShadowToken::Sm).to_vec())
            .unwrap_or_default()
    }

    /// Port-chip shadow — a soft drop that gives every port a sense
    /// of elevation off its node body. Single-layer `Shadow` (not a
    /// stack) since ports are small circles and a per-port loop
    /// over a multi-layer stack adds GPU work without visual
    /// payoff. The drop colour is tinted from `Shadow`-ink with
    /// alpha tuned for both schemes.
    pub fn port_shadow(&self) -> blinc_core::Shadow {
        let alpha = ThemeState::try_get()
            .map(|s| {
                s.shadows()
                    .get(ShadowToken::Sm)
                    .first()
                    .map(|sh| sh.color.a)
                    .unwrap_or(0.35)
            })
            .unwrap_or(0.35);
        blinc_core::Shadow {
            offset_x: 0.0,
            offset_y: 1.0,
            blur: 3.0,
            spread: 0.0,
            color: Color::rgba(0.0, 0.0, 0.0, alpha),
        }
    }

    // ─── Typography tokens ──────────────────────────────────────────
    //
    // Title / subtitle sizes pull from the theme's typography ladder
    // so dense / loose token bundles re-scale node body text without
    // a per-renderer change. Falls back to hardcoded defaults when
    // the theme isn't initialised.

    /// Font size for node + group titles.
    pub fn title_font_size(&self) -> f32 {
        ThemeState::try_get()
            .map(|s| s.typography().text_sm)
            .unwrap_or(13.0)
    }

    /// Font size for node + group subtitles / descriptions.
    pub fn subtitle_font_size(&self) -> f32 {
        ThemeState::try_get()
            .map(|s| s.typography().text_xs)
            .unwrap_or(11.0)
    }

    // ─── Spacing tokens ─────────────────────────────────────────────

    /// Inset of the title / icon / content from the body's edges.
    /// Bumped from 10px to 16px so the slots have visible breathing
    /// room from the ports sitting on the perimeter — without this
    /// the title text can crowd a left-edge input port.
    pub fn node_content_padding(&self) -> f32 {
        Self::theme_spacing(SpacingToken::Space4, 16.0)
    }
}
