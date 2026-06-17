//! Interaction state machines — drag-to-connect, marquee-to-group,
//! drag-out-of-group detection.
//!
//! All interaction state lives here rather than scattered through
//! the editor so the logic is testable in isolation (without a real
//! `CanvasKit`).

use crate::connection::ValidationOutcome;
use crate::group::{
    AddToGroupRequest, CreateGroupRequest, GroupId, RemoveFromGroupRequest, RemoveSource,
};
use crate::node::NodeId;
use crate::port::PortAddress;
use blinc_core::layer::Point;
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────
// Drag-to-connect state machine
// ─────────────────────────────────────────────────────────────────────

/// State of a drag-to-connect interaction.
///
/// Lifecycle:
/// 1. User pointer-down on an output port dot → `Idle` → `Dragging { from }`.
/// 2. As pointer moves, the editor refreshes the rubber-band preview
///    edge from `from` → current cursor.
/// 3. If pointer enters another port's hit region: `Dragging` →
///    `Hovering { from, candidate, validation }`. The validation
///    outcome (compatible / incompatible) drives the preview tint.
/// 4. Pointer-up on a compatible candidate → emit `ConnectionEvent`,
///    reset to `Idle`.
/// 5. Pointer-up anywhere else → discard, reset to `Idle`.
/// 6. Escape key during drag → also discard.
#[derive(Debug, Clone, Default)]
pub enum DragConnect {
    #[default]
    Idle,
    /// Pointer is dragging from an output port; not currently over a
    /// candidate input.
    Dragging { from: PortAddress, cursor: Point },
    /// Pointer is over a candidate input port; validation has been
    /// run.
    Hovering {
        from: PortAddress,
        candidate: PortAddress,
        validation: ValidationOutcome,
        cursor: Point,
    },
}

impl DragConnect {
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Idle)
    }

    pub fn from_port(&self) -> Option<&PortAddress> {
        match self {
            Self::Idle => None,
            Self::Dragging { from, .. } | Self::Hovering { from, .. } => Some(from),
        }
    }

    pub fn cursor(&self) -> Option<Point> {
        match self {
            Self::Idle => None,
            Self::Dragging { cursor, .. } | Self::Hovering { cursor, .. } => Some(*cursor),
        }
    }

    /// Start a drag from a freshly-clicked output port.
    pub fn begin(&mut self, from: PortAddress, cursor: Point) {
        *self = Self::Dragging { from, cursor };
    }

    /// Pointer moved without entering a candidate port.
    pub fn move_to(&mut self, cursor: Point) {
        match self {
            Self::Idle => {}
            Self::Dragging { cursor: c, .. } | Self::Hovering { cursor: c, .. } => *c = cursor,
        }
    }

    /// Pointer entered (or moved within) a candidate port. Caller
    /// supplies the validation outcome from the host's matcher.
    pub fn hover(&mut self, candidate: PortAddress, validation: ValidationOutcome, cursor: Point) {
        let from = match self {
            Self::Idle => return,
            Self::Dragging { from, .. } | Self::Hovering { from, .. } => from.clone(),
        };
        *self = Self::Hovering {
            from,
            candidate,
            validation,
            cursor,
        };
    }

    /// Pointer left the candidate without releasing — fall back to
    /// `Dragging`.
    pub fn unhover(&mut self) {
        if let Self::Hovering { from, cursor, .. } = self {
            *self = Self::Dragging {
                from: from.clone(),
                cursor: *cursor,
            };
        }
    }

    /// Pointer released. Returns a [`DragRelease`] describing how the
    /// drag ended so the caller can emit the right event:
    ///
    /// - `Empty` — released on empty canvas (or while still in
    ///   `Dragging`, no candidate hovered). Caller emits nothing.
    /// - `Accepted(candidate)` — released on a candidate the validator
    ///   accepted. Caller emits `EditorEvent::ConnectionAccepted`.
    /// - `Rejected { candidate, reason }` — released on a candidate
    ///   the validator rejected. Caller emits
    ///   `EditorEvent::ConnectionRejected` so the host can show a
    ///   reason-bearing toast (the live red-tinted preview line
    ///   already conveyed the rejection visually, but doesn't carry
    ///   the validator's reason string).
    ///
    /// Always resets to `Idle` before returning.
    pub fn release(&mut self) -> DragRelease {
        let result = match self {
            Self::Hovering {
                candidate,
                validation,
                ..
            } => match validation {
                ValidationOutcome::Accept => DragRelease::Accepted(candidate.clone()),
                ValidationOutcome::Reject { reason } => DragRelease::Rejected {
                    candidate: candidate.clone(),
                    reason: reason.clone(),
                },
            },
            _ => DragRelease::Empty,
        };
        *self = Self::Idle;
        result
    }

    /// Cancel without releasing — escape key, click elsewhere, focus
    /// loss.
    pub fn cancel(&mut self) {
        *self = Self::Idle;
    }
}

/// Outcome of a pointer-up during a port-connect drag. See
/// [`DragConnect::release`] for the lifecycle.
#[derive(Debug, Clone)]
pub enum DragRelease {
    /// Released on empty canvas (or while no candidate was hovered).
    /// The editor emits no event.
    Empty,
    /// Released on a candidate the validator accepted. The editor
    /// emits [`crate::EditorEvent::ConnectionAccepted`].
    Accepted(PortAddress),
    /// Released on a candidate the validator rejected. The editor
    /// emits [`crate::EditorEvent::ConnectionRejected`] so the host
    /// can show a toast / banner explaining WHY (the live red
    /// preview already conveyed THAT it was rejected; this carries
    /// the validator's reason string for hosts that want to surface
    /// it textually).
    Rejected {
        candidate: PortAddress,
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Marquee-to-group
// ─────────────────────────────────────────────────────────────────────

/// Build a [`CreateGroupRequest`] from a selection set. The editor
/// calls this from the "group these" action (context menu /
/// keyboard shortcut) when at least two nodes are selected. The
/// host's handler creates the group and re-syncs.
///
/// Returns `None` if fewer than two nodes are selected — a one-node
/// group adds noise without value.
pub fn group_request_from_selection(
    selected_node_ids: HashSet<NodeId>,
    default_name: impl Into<String>,
) -> Option<CreateGroupRequest> {
    if selected_node_ids.len() < 2 {
        return None;
    }
    let mut members: Vec<NodeId> = selected_node_ids.into_iter().collect();
    // Sort for determinism — tests + reproducibility.
    members.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    Some(CreateGroupRequest {
        members,
        default_name: default_name.into(),
    })
}

// ─────────────────────────────────────────────────────────────────────
// Drag-into / drag-out-of group detection
// ─────────────────────────────────────────────────────────────────────

/// After a node drag-end, classify whether the new position changed
/// its group membership relative to the prior membership. Returns:
/// * `Some(Add { group, node })` — node moved INTO a group it wasn't
///   in before.
/// * `Some(RemoveOut { group, node })` — node moved OUT of a group
///   it was in.
/// * `None` — no membership change.
///
/// The editor calls this with the membership snapshots before /
/// after the drag; both must come from the host's model.
pub fn classify_drag_membership_change(
    node: NodeId,
    before_membership: Option<GroupId>,
    after_membership: Option<GroupId>,
) -> Option<GroupMembershipChange> {
    use GroupMembershipChange::*;
    match (before_membership, after_membership) {
        (Some(prev), Some(next)) if prev == next => None,
        (Some(prev), None) => Some(RemoveOut(RemoveFromGroupRequest {
            group: prev,
            node,
            source: RemoveSource::DraggedOut,
        })),
        (None, Some(next)) => Some(Add(AddToGroupRequest { group: next, node })),
        (Some(prev), Some(_next)) => {
            // Moved from prev → next; emit a remove-from-prev. The
            // caller follows up with an add-to-next on the same drag-end
            // hook. (Two events because the host might want distinct
            // confirmations for each.)
            Some(RemoveOut(RemoveFromGroupRequest {
                group: prev,
                node,
                source: RemoveSource::DraggedOut,
            }))
        }
        (None, None) => None,
    }
}

/// Discriminator for [`classify_drag_membership_change`] output.
#[derive(Debug, Clone)]
pub enum GroupMembershipChange {
    Add(AddToGroupRequest),
    RemoveOut(RemoveFromGroupRequest),
}
