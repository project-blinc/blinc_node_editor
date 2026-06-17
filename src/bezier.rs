//! Bezier helpers — cubic curve sampling + distance-to-curve for
//! edge hit-testing.
//!
//! Mirrors the sampling pattern proven in
//! `examples/blinc_app_examples/examples/canvas_kit_demo.rs` (steps =
//! 20, mid-X control points). The crate exposes both the sample
//! primitive (for renderers) and a `distance_squared_to_curve`
//! helper (for edge hit-testing).
//!
//! Coordinate convention: all functions accept + return `Point` in
//! canvas-content coordinates. Callers translate from screen via
//! `CanvasKit::screen_to_content` before passing in.

use blinc_core::layer::Point;

/// 1D cubic Bezier evaluation. Identical to the demo's helper —
/// kept as a standalone function so callers that already have the
/// four scalars don't construct `Point`s.
#[inline]
pub fn cubic_1d(t: f32, p0: f32, p1: f32, p2: f32, p3: f32) -> f32 {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    p0 * uu * u + p1 * 3.0 * uu * t + p2 * 3.0 * u * tt + p3 * tt * t
}

/// 2D cubic Bezier evaluation at parameter `t ∈ [0, 1]`.
#[inline]
pub fn cubic_point(t: f32, p0: Point, p1: Point, p2: Point, p3: Point) -> Point {
    Point::new(
        cubic_1d(t, p0.x, p1.x, p2.x, p3.x),
        cubic_1d(t, p0.y, p1.y, p2.y, p3.y),
    )
}

/// Mid-X control point pair for a "horizontal sweep" edge — the
/// shape canvas_kit_demo.rs uses and the convention this crate
/// adopts for default edge routing. Useful when endpoints are
/// known but control points are derived.
///
/// Returns `(ctrl1, ctrl2)` where:
/// * `ctrl1 = (mid_x, start.y)` — pull start horizontally toward
///   the midpoint
/// * `ctrl2 = (mid_x, end.y)` — pull end horizontally toward the
///   midpoint
///
/// Yields the smooth S-curve typical of node-graph editors.
#[inline]
pub fn mid_x_controls(start: Point, end: Point) -> (Point, Point) {
    let mid_x = (start.x + end.x) * 0.5;
    (Point::new(mid_x, start.y), Point::new(mid_x, end.y))
}

/// Midpoint of a cubic at t = 0.5. Convenience over
/// `cubic_point(0.5, ...)` for renderers that need to anchor
/// decorations (delete buttons, status pips) to the visual centre
/// of an edge.
#[inline]
pub fn cubic_midpoint(p0: Point, p1: Point, p2: Point, p3: Point) -> Point {
    cubic_point(0.5, p0, p1, p2, p3)
}

/// Generate a list of axis-aligned bounding boxes covering a cubic
/// curve, each `thickness` wide along the segment normal. Useful as
/// a coarse hit-region approximation: register one rect per segment
/// and the caller's rect-based hit-test (e.g.
/// [`CanvasKit::hit_rect`]) will pick up clicks within `thickness/2`
/// of the curve.
///
/// `steps` controls polyline subdivision; the returned vec has
/// `steps` rects. Each rect is the axis-aligned bbox of the
/// segment endpoints, inflated by `thickness * 0.5`. False
/// positives near sharp curve corners are accepted for sparse
/// graphs; refine by intersecting with [`distance_squared_to_curve`]
/// at hit time if precision matters.
pub fn segment_bboxes(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    steps: usize,
    thickness: f32,
) -> Vec<blinc_core::layer::Rect> {
    use blinc_core::layer::Rect;
    let pts = sample_cubic(p0, p1, p2, p3, steps);
    let half = (thickness * 0.5).max(1.0);
    let mut out = Vec::with_capacity(pts.len().saturating_sub(1));
    for win in pts.windows(2) {
        let (a, b) = (win[0], win[1]);
        let min_x = a.x.min(b.x) - half;
        let min_y = a.y.min(b.y) - half;
        let max_x = a.x.max(b.x) + half;
        let max_y = a.y.max(b.y) + half;
        out.push(Rect::new(min_x, min_y, max_x - min_x, max_y - min_y));
    }
    out
}

/// Sample a cubic Bezier into `steps + 1` points (inclusive both
/// ends). The renderer walks the result emitting fill_rect /
/// stroke segments; hit-testing walks the same polyline computing
/// per-segment distance.
///
/// `steps = 20` matches the demo's default. Callers that need
/// smoother curves at high zoom can pass higher values; callers
/// rendering many edges at low zoom can drop to 10.
pub fn sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: usize) -> Vec<Point> {
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        pts.push(cubic_point(t, p0, p1, p2, p3));
    }
    pts
}

/// Squared distance from `pt` to the curve, measured as the
/// minimum squared distance to any line segment of a `steps`-step
/// polyline approximation. Squared form to avoid the per-segment
/// sqrt; callers comparing against a tolerance threshold should
/// pre-square it (`tol * tol`).
///
/// Returns `f32::INFINITY` if `steps == 0` (degenerate polyline).
pub fn distance_squared_to_curve(
    pt: Point,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    steps: usize,
) -> f32 {
    if steps == 0 {
        return f32::INFINITY;
    }
    let polyline = sample_cubic(p0, p1, p2, p3, steps);
    let mut best = f32::INFINITY;
    for window in polyline.windows(2) {
        let d = distance_squared_to_segment(pt, window[0], window[1]);
        if d < best {
            best = d;
        }
    }
    best
}

/// Squared point-to-segment distance. Standard formula —
/// projecting `pt` onto the segment and clamping the parameter to
/// `[0, 1]`.
fn distance_squared_to_segment(pt: Point, a: Point, b: Point) -> f32 {
    let ab = (b.x - a.x, b.y - a.y);
    let ap = (pt.x - a.x, pt.y - a.y);
    let ab_len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    if ab_len_sq < f32::EPSILON {
        // Degenerate segment — just point-to-point distance.
        return ap.0 * ap.0 + ap.1 * ap.1;
    }
    let t = ((ap.0 * ab.0 + ap.1 * ab.1) / ab_len_sq).clamp(0.0, 1.0);
    let proj = (a.x + ab.0 * t, a.y + ab.1 * t);
    let dx = pt.x - proj.0;
    let dy = pt.y - proj.1;
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_endpoints_match() {
        let p0 = Point::new(0.0, 0.0);
        let p1 = Point::new(1.0, 0.0);
        let p2 = Point::new(2.0, 0.0);
        let p3 = Point::new(3.0, 0.0);
        let start = cubic_point(0.0, p0, p1, p2, p3);
        let end = cubic_point(1.0, p0, p1, p2, p3);
        assert!((start.x - 0.0).abs() < 1e-5);
        assert!((end.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn distance_zero_on_curve_point() {
        let p0 = Point::new(0.0, 0.0);
        let (c1, c2) = mid_x_controls(p0, Point::new(100.0, 50.0));
        let p3 = Point::new(100.0, 50.0);
        // A point exactly on the curve should have distance ~0.
        let mid = cubic_point(0.5, p0, c1, c2, p3);
        let d2 = distance_squared_to_curve(mid, p0, c1, c2, p3, 20);
        assert!(d2 < 0.5, "expected ~0 distance, got {d2}");
    }
}
