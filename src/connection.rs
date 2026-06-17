//! Connection types — typed edges, validation requests, events.

use crate::port::{PortAddress, PortKind};

// ─────────────────────────────────────────────────────────────────────
// ConnectionId
// ─────────────────────────────────────────────────────────────────────

/// Stable identifier for a connection. Derived from the
/// `(from, to)` endpoint pair via a fast non-cryptographic hash so
/// the same endpoint pair always produces the same id (idempotent
/// re-creation across `set_graph()` calls), regardless of insertion
/// order in the host's model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(pub u64);

impl ConnectionId {
    /// Hash a `(from, to)` pair into a stable id. Uses `ahash` because
    /// the same crate uses it internally for hit-region indexing — keeps
    /// the dep surface small.
    pub fn from_endpoints(from: &PortAddress, to: &PortAddress) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        from.node.as_str().hash(&mut hasher);
        from.port.as_str().hash(&mut hasher);
        to.node.as_str().hash(&mut hasher);
        to.port.as_str().hash(&mut hasher);
        Self(hasher.finish())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Connection — the placed edge
// ─────────────────────────────────────────────────────────────────────

/// Runtime / lifecycle state of a connection, driving its colour
/// + animation in the renderer. Borrowed from Zeal's
/// `ConnectionState` — the at-a-glance green / red / pulsing
/// distinction is core UX. Hosts update this from their runtime
/// observability layer (for nan8 + reflow this comes from the
/// KyuGraph-projected trace stream).
///
/// `None` = neutral edge (default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConnectionState {
    /// Default — no runtime classification. Rendered in the theme's
    /// neutral edge colour.
    #[default]
    None,
    /// Connection is wired but hasn't carried traffic yet, or is
    /// awaiting a downstream actor. Drawn dimmed.
    Pending,
    /// Recent traffic raised a warning the host wants to surface
    /// (slow response, partial result). Drawn in the theme's
    /// warning colour, no animation.
    Warning,
    /// Recent traffic errored — exception thrown, schema mismatch,
    /// timeout. Drawn in the theme's error colour.
    Error,
    /// Recent traffic completed successfully. Drawn in the theme's
    /// success colour (typically green).
    Success,
    /// Currently carrying traffic. Drawn with the edge-flow particle
    /// animation (canvas_kit's `Sketch + Player`).
    Running,
}

/// A live edge between two ports in the editor. The host provides
/// these from its model (reflow's `GraphConnection`, …); the editor
/// renders them, lets users select and delete them, and routes
/// drag-to-connect events back to the host for materialisation.
///
/// `M` is the host's per-edge metadata (e.g. reflow's
/// `GraphConnection.metadata`). Opaque to the editor.
#[derive(Debug, Clone)]
pub struct Connection<M> {
    pub id: ConnectionId,
    pub from: PortAddress,
    pub to: PortAddress,
    /// Runtime state driving edge colour + animation. Defaults to
    /// `None` (neutral). Hosts update this from their observability
    /// layer.
    pub state: ConnectionState,
    pub metadata: M,
}

impl<M: Default> Connection<M> {
    pub fn new(from: PortAddress, to: PortAddress) -> Self {
        Self {
            id: ConnectionId::from_endpoints(&from, &to),
            from,
            to,
            state: ConnectionState::default(),
            metadata: M::default(),
        }
    }
}

impl<M> Connection<M> {
    pub fn with_state(mut self, state: ConnectionState) -> Self {
        self.state = state;
        self
    }

    pub fn with_metadata(mut self, metadata: M) -> Self {
        self.metadata = metadata;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// ConnectRequest — drag-to-connect validation hand-off
// ─────────────────────────────────────────────────────────────────────

/// In-flight connection request as the user drags from an output port
/// dot toward an input port dot. Handed to the host's validator via
/// `NodeEditor::on_connect_validate`; the host returns `bool` (or
/// richer per the `ValidationOutcome` extension) and the editor uses
/// that to colour the drag-preview edge (compatible → highlight,
/// incompatible → dim/red).
///
/// `K` is the host's port-kind — both endpoints' kinds are surfaced
/// so the host can match without re-looking-them-up.
pub struct ConnectRequest<'a, K: PortKind> {
    pub from: &'a PortAddress,
    pub from_kind: &'a K,
    pub to: &'a PortAddress,
    pub to_kind: &'a K,
}

/// Returned by `on_connect_validate`. The simple `bool` form converts
/// via `From<bool>`; richer hosts can return `Reject { reason }` so
/// the editor surfaces a tooltip with the reason.
#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    Accept,
    /// Reject with a human-readable reason. Shown as a tooltip on the
    /// drag-preview cursor; the editor doesn't materialise the edge.
    Reject { reason: String },
}

impl From<bool> for ValidationOutcome {
    fn from(b: bool) -> Self {
        if b { Self::Accept } else { Self::Reject { reason: String::new() } }
    }
}

impl ValidationOutcome {
    pub fn is_accept(&self) -> bool { matches!(self, Self::Accept) }
}

// ─────────────────────────────────────────────────────────────────────
// ConnectionEvent — successful connection materialisation
// ─────────────────────────────────────────────────────────────────────

/// Fired by the editor after a successful drag-to-connect (validator
/// returned `Accept`). The host's handler creates the corresponding
/// edge in its own model; subsequent `set_graph(...)` calls reflect
/// the new edge back into the editor.
///
/// The editor does NOT auto-add the connection to its own list — the
/// host is the source of truth; the editor is a view.
#[derive(Debug, Clone)]
pub struct ConnectionEvent<K: PortKind> {
    pub from: PortAddress,
    pub from_kind: K,
    pub to: PortAddress,
    pub to_kind: K,
}
