//! Minimap — a screen-pinned overview of the whole graph with a
//! draggable viewport indicator.
//!
//! The minimap is editor chrome: a small panel anchored to a screen
//! corner that draws every node (and group) at a reduced scale, plus a
//! rectangle marking the region the camera currently shows. Clicking or
//! dragging inside the panel recenters the camera on that point, so it
//! doubles as a fast pan control for large graphs.
//!
//! It is enabled by default (bottom-right). Disable or relocate it with
//! [`NodeEditor::with_minimap`](crate::NodeEditor::with_minimap) /
//! [`NodeEditor::set_minimap_enabled`](crate::NodeEditor::set_minimap_enabled).

use blinc_core::layer::Rect;

/// Canvas-kit hit-region id for the minimap panel. A raw string (not a
/// [`crate::region::RegionId`]) so `RegionId::parse` returns `None` for
/// it and the typed routing in the editor's pointer handlers ignores it
/// — the minimap branches match this id explicitly, before parsing.
pub const MINIMAP_REGION: &str = "__minimap__";

/// Which screen corner the minimap panel anchors to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Minimap appearance + placement. Visual only — it never affects node
/// layout, so swapping it at runtime needs no slot-cache invalidation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimapConfig {
    /// When `false`, the minimap is neither drawn nor hit-tested (zero
    /// per-frame cost).
    pub enabled: bool,
    /// Screen corner the panel anchors to. Default [`Corner::BottomRight`].
    pub corner: Corner,
    /// Panel size in screen-logical pixels (width, height).
    pub size: (f32, f32),
    /// Gap in screen-logical pixels between the panel and the two
    /// screen edges it hugs.
    pub margin: f32,
    /// Inset in screen-logical pixels between the panel border and the
    /// drawn graph content, so dots near the graph's bbox edge aren't
    /// clipped by the panel stroke.
    pub padding: f32,
}

impl Default for MinimapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            corner: Corner::BottomRight,
            size: (210.0, 150.0),
            margin: 16.0,
            padding: 10.0,
        }
    }
}

impl MinimapConfig {
    /// Compute the panel's screen-space rect for a given canvas size.
    pub fn panel_rect(&self, screen_w: f32, screen_h: f32) -> Rect {
        let (w, h) = self.size;
        let m = self.margin;
        let (x, y) = match self.corner {
            Corner::TopLeft => (m, m),
            Corner::TopRight => (screen_w - m - w, m),
            Corner::BottomLeft => (m, screen_h - m - h),
            Corner::BottomRight => (screen_w - m - w, screen_h - m - h),
        };
        Rect::new(x, y, w, h)
    }
}

/// Per-frame cache written by the render pass and read by the pointer
/// handlers so a click/drag inside the panel maps back to a world
/// coordinate. Both rects are in the same spaces they were computed in:
/// `content_screen` in screen-logical pixels (where the graph is
/// actually drawn inside the panel, letterboxed to preserve aspect),
/// `world_bbox` in canvas-content space.
#[derive(Clone, Copy, Debug)]
pub struct MinimapHit {
    /// Screen-pixel rect the graph maps into inside the panel.
    pub content_screen: Rect,
    /// World-space bounding box the `content_screen` rect represents.
    pub world_bbox: Rect,
}

impl MinimapHit {
    /// Map a screen-pixel point inside the panel to the world point it
    /// represents, clamped to the graph's bounding box.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let cw = self.content_screen.width().max(1.0);
        let ch = self.content_screen.height().max(1.0);
        let nx = ((screen_x - self.content_screen.x()) / cw).clamp(0.0, 1.0);
        let ny = ((screen_y - self.content_screen.y()) / ch).clamp(0.0, 1.0);
        (
            self.world_bbox.x() + nx * self.world_bbox.width(),
            self.world_bbox.y() + ny * self.world_bbox.height(),
        )
    }
}
