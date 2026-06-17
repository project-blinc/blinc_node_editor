//! Port types — typed, NOT stringly.
//!
//! Per DIRECTIVE.md §2: the editor is generic over the host's port
//! type via the [`PortKind`] trait. nan8 will impl this for reflow's
//! `PortType` and forward to reflow's `is_compatible_with` rules.
//! Other DAG runtimes plug in the same way.

use blinc_core::layer::Color;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// PortKind — the host-typed port surface
// ─────────────────────────────────────────────────────────────────────

/// Host-supplied port-type description. The editor never inspects the
/// concrete shape of the type; it asks the host whether two ports are
/// compatible, what to label them, and what colour-accent to render.
///
/// The bound is `Clone + Send + Sync + 'static` so port descriptors can
/// be stored in shared state and accessed from event handlers across
/// threads.
///
/// Reflow adapter sketch (lives in `nan8_ui`, NOT here):
/// ```ignore
/// impl PortKind for reflow_graph::PortType {
///     fn compatible_with(&self, other: &Self) -> bool {
///         // Forward to reflow's finalised PortType::is_compatible_with.
///         // (See DIRECTIVE.md §2 open question O1.)
///         reflow_graph::is_compatible(self, other)
///     }
///     fn label(&self) -> String { format!("{self:?}") }
///     fn accent(&self) -> Color { match self {
///         reflow_graph::PortType::Flow    => Color::rgb(0.95, 0.55, 0.20),
///         reflow_graph::PortType::Boolean => Color::rgb(0.60, 0.80, 0.30),
///         /* ... per-variant theme mapping ... */
///         _ => Color::rgb(0.50, 0.70, 0.90),
///     } }
///     fn category(&self) -> PortCategory { match self {
///         reflow_graph::PortType::Flow | reflow_graph::PortType::Event => PortCategory::Control,
///         reflow_graph::PortType::Stream => PortCategory::Stream,
///         _ => PortCategory::Data,
///     } }
/// }
/// ```
pub trait PortKind: Clone + Send + Sync + 'static {
    /// Whether `self` (an output) can connect to `other` (an input).
    /// Direction is implicit: `self` is the producer side.
    fn compatible_with(&self, other: &Self) -> bool;

    /// Short human-readable label — shown in port tooltips and the
    /// inspector. Doesn't need to be globally unique.
    fn label(&self) -> String;

    /// Accent colour for the port dot + matching edge tint. Pulled from
    /// the host's type-to-colour mapping; the editor never injects its
    /// own type colours.
    fn accent(&self) -> Color;

    /// Distinguish data ports from control / streaming ports so the
    /// renderer can draw them differently (e.g. solid dot for `Data`,
    /// chevron for `Control`, dashed for `Stream`). Defaults to
    /// `Data` for hosts that don't model the distinction.
    fn category(&self) -> PortCategory {
        PortCategory::Data
    }
}

/// Visual category of a port. The renderer picks a glyph per category;
/// the host's `PortKind::category()` returns this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortCategory {
    /// Carries data values. Drawn as a filled circle.
    Data,
    /// Carries control-flow signals (reflow `Flow` / `Event`). Drawn
    /// as a triangle / chevron to distinguish from data.
    Control,
    /// Carries a streaming handle (reflow `Stream`). Drawn as a dashed
    /// ring or animated outline so users can spot streams visually.
    Stream,
}

// ─────────────────────────────────────────────────────────────────────
// Port identity + addressing
// ─────────────────────────────────────────────────────────────────────

/// Direction of a port relative to its node — data-flow semantic, not
/// geometric placement. See [`PortPosition`] for which edge the port
/// is rendered on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Direction {
    /// Sinks data. Default placement: left edge.
    #[default]
    Input,
    /// Sources data. Default placement: right edge.
    Output,
}

impl Direction {
    pub fn is_input(self) -> bool { matches!(self, Self::Input) }
    pub fn is_output(self) -> bool { matches!(self, Self::Output) }
    pub fn flip(self) -> Self { match self {
        Self::Input => Self::Output,
        Self::Output => Self::Input,
    } }

    /// Default geometric placement: input → left, output → right.
    /// Zeal supports all 4 sides; nodes that want top/bottom ports
    /// override per-port via [`PortDesc::position`].
    pub fn default_position(self) -> PortPosition {
        match self {
            Self::Input => PortPosition::Left,
            Self::Output => PortPosition::Right,
        }
    }
}

/// Which edge of the node a port is rendered on. Borrowed from Zeal's
/// `Port.position` — many real graph designs (looping flows, vertical
/// pipelines, control-flow side-channels) need ports on the top/bottom
/// edges, not just left/right. Defaults derive from [`Direction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortPosition {
    Top,
    Right,
    Bottom,
    Left,
}

/// Stable per-node port identifier. Scoped to the parent node — two
/// different nodes can both have a port with id `"out"`. Combine with
/// a [`crate::node::NodeId`] via [`PortAddress`] for cross-node
/// addressing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId(pub Arc<str>);

impl PortId {
    pub fn new(id: impl Into<Arc<str>>) -> Self { Self(id.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl From<&str> for PortId    { fn from(s: &str)    -> Self { Self(Arc::from(s)) } }
impl From<String> for PortId  { fn from(s: String)  -> Self { Self(Arc::from(s)) } }
impl From<Arc<str>> for PortId { fn from(s: Arc<str>) -> Self { Self(s) } }

/// A port's full address — node + port. Used in [`Connection`](crate::Connection)
/// endpoints and in drag-to-connect events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortAddress {
    pub node: crate::node::NodeId,
    pub port: PortId,
}

impl PortAddress {
    pub fn new(node: crate::node::NodeId, port: impl Into<PortId>) -> Self {
        Self { node, port: port.into() }
    }
}

// ─────────────────────────────────────────────────────────────────────
// PortDesc — the declarative port shape
// ─────────────────────────────────────────────────────────────────────

/// Declarative port descriptor. Lives inside a [`NodeTemplate`](crate::NodeTemplate)
/// (declares what a component HAS) and is what the renderer reads to
/// draw the port dot, place it on the node edge, and resolve hit-test
/// targets.
///
/// `metadata` is a host-supplied [`serde_json::Value`]-shaped bag the
/// editor doesn't inspect — used to carry per-port hints (units,
/// validation rules, doc strings, IIP default values) the inspector
/// might surface.
#[derive(Debug, Clone)]
pub struct PortDesc<K: PortKind> {
    pub id: PortId,
    pub name: String,
    pub direction: Direction,
    /// Which edge of the node this port renders on. `None` =
    /// derive from `direction` (input → left, output → right). Set
    /// explicitly for designs that put control-flow / branching
    /// ports on top or bottom.
    pub position: Option<PortPosition>,
    pub kind: K,
    pub metadata: PortMetadata,
    /// Optional human-readable doc string shown in port tooltips +
    /// inspector. Useful both for end-user discoverability and for
    /// LLM-driven graph construction (Zeal exposes the equivalent
    /// `description` field on every port for prompt context).
    pub description: Option<String>,
}

impl<K: PortKind> PortDesc<K> {
    /// Convenience constructor for the common case (no extra metadata,
    /// default position derived from direction).
    pub fn new(id: impl Into<PortId>, name: impl Into<String>, direction: Direction, kind: K) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            direction,
            position: None,
            kind,
            metadata: PortMetadata::default(),
            description: None,
        }
    }

    /// Resolved position — explicit if set, otherwise derived from
    /// direction.
    pub fn resolved_position(&self) -> PortPosition {
        self.position.unwrap_or_else(|| self.direction.default_position())
    }

    pub fn with_position(mut self, position: PortPosition) -> Self {
        self.position = Some(position);
        self
    }

    /// Attach a human-readable doc string. Shown in the port
    /// tooltip on hover + surfaced by the inspector.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Editor-supplied port hints. Opaque to the editor's render path
/// except for [`PortMetadata::iip_default`] — when an input port
/// carries an IIP default, the editor renders the value inline
/// instead of waiting for an incoming edge (per DIRECTIVE.md §6.5).
///
/// Hosts that want richer metadata (units, schemas, …) embed them in
/// [`PortMetadata::extra`]; the inspector picks them up.
#[derive(Debug, Clone, Default)]
pub struct PortMetadata {
    /// Optional human-readable type / unit string (e.g. "ms",
    /// "pixels"). Renders next to the port label in tooltips.
    pub unit: Option<String>,
    /// IIP default value the editor surfaces inline on an unconnected
    /// input port. `None` means "no default; show an empty slot".
    pub iip_default: Option<serde_json::Value>,
    /// Anything else the host wants to carry. The editor doesn't read
    /// this; downstream widgets (palette, inspector) might.
    pub extra: serde_json::Value,
}
