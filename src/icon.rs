//! Node icons — SVG-based, icon-pack agnostic.
//!
//! ## Why SVG strings
//!
//! Hosts can ship any SVG markup — tabler, lucide, Material, custom
//! authored SVGs — and the editor renders them via
//! [`blinc_svg::SvgDocument`]. Renders through `fill_path` /
//! `stroke_path` so it works inside canvas closures, unlike
//! `draw_text` (currently blocked by an upstream Blinc gotcha).
//!
//! ## Adapters
//!
//! - **tabler**:
//!   ```ignore
//!   use blinc_tabler_icons::{outline, to_svg};
//!   NodeIcon::from_svg_str(&to_svg(outline::DATABASE, 16.0))
//!       .expect("tabler emits valid SVG")
//!   ```
//! - **lucide / Material / custom**: same shape — pass any SVG
//!   string to `NodeIcon::from_svg_str`.
//!
//! ## Caching
//!
//! Parsing SVG is non-trivial (usvg + tiny_skia_path). Hosts that
//! build many template instances should parse once at template-
//! registration time (via `from_svg_str` returning a `NodeIcon`
//! that holds the parsed [`SvgDocument`]) and reuse the icon
//! across instances — `NodeIcon: Clone` is cheap (`SvgDocument`
//! is `Clone` and shares its internal tree).

use blinc_svg::SvgDocument;

/// A renderable node icon. The default variant carries a parsed
/// [`SvgDocument`]; we keep the enum open so future variants (raw
/// `Path`, raster atlas, etc.) can be added without breaking
/// `NodeTemplate` / `NodeInstance` consumers.
#[derive(Clone)]
pub enum NodeIcon {
    /// Pre-parsed SVG. Use [`Self::from_svg_str`] to construct from
    /// raw SVG markup.
    Svg(SvgDocument),
}

impl NodeIcon {
    /// Parse SVG markup into a [`NodeIcon`]. The parsed document
    /// is stored on the icon; subsequent renders skip the parse
    /// step entirely.
    ///
    /// Returns `Err` if `svg` isn't valid SVG (usvg parse failure).
    /// Hosts wiring icons at template-registration time should
    /// `expect` — well-formed tabler / lucide markup never fails.
    pub fn from_svg_str(svg: &str) -> Result<Self, blinc_svg::SvgError> {
        SvgDocument::from_str(svg).map(NodeIcon::Svg)
    }

    /// Borrow the underlying parsed document — useful for hosts
    /// that want to inspect the icon's intrinsic size or
    /// commands.
    pub fn document(&self) -> &SvgDocument {
        match self {
            NodeIcon::Svg(doc) => doc,
        }
    }
}

impl std::fmt::Debug for NodeIcon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeIcon::Svg(doc) => f
                .debug_struct("NodeIcon::Svg")
                .field("width", &doc.width)
                .field("height", &doc.height)
                .finish(),
        }
    }
}
