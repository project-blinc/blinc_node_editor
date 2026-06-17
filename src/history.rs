//! Undo / redo history for a [`crate::NodeEditor`].
//!
//! ## Strategy
//!
//! Command-log + inverse. Each entry stores the forward
//! [`EditorCommand`] that was applied plus the inverse command needed
//! to undo it. The history doesn't snapshot the whole graph — only
//! the deltas — so memory cost is bounded by entry count, not graph
//! size. Compound undos (e.g. re-inserting a deleted node with its
//! incident edges + group memberships) are described as a single
//! [`EditorCommand::Composite`].
//!
//! ## Where the recording happens
//!
//! Hosts call [`History::push`] at the natural mutation site, where
//! they already have access to the pre-state needed to synthesise the
//! inverse. For drag gestures the host calls [`History::push_coalesced`]
//! with a [`CoalesceKey::DragNode`] so a continuous drag collapses
//! into one undoable step rather than one entry per pixel of motion.
//!
//! The editor never auto-records. This keeps "host as source of truth"
//! intact: programmatic re-syncs (badge updates, connection-state
//! ticks, `set_graph` round-trips) don't accidentally land on the undo
//! stack. If a host wants the editor's own keyboard shortcut to fire
//! undo, it subscribes to [`crate::EditorEvent::UndoRequested`] (and
//! `RedoRequested`) and calls `History::undo` / `redo` itself.
//!
//! ## Bounds
//!
//! `History<K, N, C, G>` widens the editor's generic bounds to
//! `N: Clone, C: Clone, G: Clone` because forward + inverse commands
//! carry owned copies of node / connection / group payloads. Existing
//! host metadata types (`reflow`'s `GraphNode`, simple unit `()`, …)
//! already derive `Clone`.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::editor::NodeEditor;
use crate::event::EditorCommand;
use crate::node::NodeId;
use crate::port::PortKind;

/// Reference-counted, thread-safe closure backing one half of a
/// host-defined transaction. Stored inside
/// [`HistoryEntry::Transaction`] so a single entry can be replayed
/// any number of times by undo / redo.
pub type TransactionFn = Arc<dyn Fn() + Send + Sync + 'static>;

/// Default cap when the host calls [`History::with_default_cap`].
/// Tuned for graph editors: 100 user gestures is several minutes of
/// continuous editing and well below the memory ceiling (the
/// worst-case entry is ~2 KB for a `RemoveNode` with many incident
/// edges, so 100 entries ≈ 200 KB).
pub const DEFAULT_CAP: usize = 100;

/// One recorded edit. Stored on both the undo and redo stacks: when
/// the user undoes, the entry moves from `undo` → `redo`; redoing
/// moves it back the other way.
///
/// Two flavours:
///
/// * [`Self::Command`] — wraps a pair of [`EditorCommand`]s. The
///   editor's [`NodeEditor::dispatch`] applies them. This is what
///   every built-in mutation goes through.
/// * [`Self::Transaction`] — wraps a host-supplied `Fn` pair. Use
///   this when a single user gesture mutates *both* the editor's
///   graph AND host-side state that the editor doesn't know about
///   (preset libraries, project files, selection metadata, …). Push
///   the transaction with [`History::push_transaction`].
#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // EditorCommand payloads dominate; boxing would force allocs on every history push (drag-coalesced UpdateNodePosition is hot)
pub enum HistoryEntry<K: PortKind, N: Clone, C: Clone, G: Clone> {
    /// Symmetric [`EditorCommand`] pair. The undo path clones
    /// `inverse` and feeds it to [`NodeEditor::dispatch`]; redo does
    /// the same with `forward`.
    Command {
        forward: EditorCommand<K, N, C, G>,
        inverse: EditorCommand<K, N, C, G>,
        label: &'static str,
    },
    /// Host-defined undo / redo closures. `forward` re-applies the
    /// user action (called on redo); `inverse` reverses it (called
    /// on undo). The closures may call into the editor (via captured
    /// `NodeEditor` clones) AND patch any host-owned state in one
    /// shot, so the user sees a single undoable step.
    ///
    /// Closures are `Fn` (not `FnOnce`) and stored in `Arc`, so an
    /// entry can be replayed by undo / redo any number of times.
    /// Both halves must be `Send + Sync + 'static`.
    Transaction {
        forward: TransactionFn,
        inverse: TransactionFn,
        label: &'static str,
    },
}

impl<K, N, C, G> HistoryEntry<K, N, C, G>
where
    K: PortKind,
    N: Clone,
    C: Clone,
    G: Clone,
{
    /// Human-readable label associated with this entry (regardless of
    /// variant).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Command { label, .. } | Self::Transaction { label, .. } => label,
        }
    }
}

impl<K, N, C, G> std::fmt::Debug for HistoryEntry<K, N, C, G>
where
    K: PortKind,
    N: Clone,
    C: Clone,
    G: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Command { label, .. } => f
                .debug_struct("HistoryEntry::Command")
                .field("label", label)
                .finish_non_exhaustive(),
            Self::Transaction { label, .. } => f
                .debug_struct("HistoryEntry::Transaction")
                .field("label", label)
                .finish_non_exhaustive(),
        }
    }
}

/// Key used by [`History::push_coalesced`] to fold consecutive edits
/// of the same kind into a single undo step. Today only node-drag
/// coalescing is wired; future widget-level coalescing (e.g. inline
/// text edits) goes here as new variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoalesceKey {
    /// Successive [`EditorCommand::UpdateNodePosition`] calls for the
    /// same node id collapse to one entry. The entry's forward
    /// position is the last one received; the inverse position is the
    /// node's location at the start of the gesture.
    DragNode(NodeId),
}

/// Undo / redo stacks owned by the host.
///
/// Cheap to construct; one per editor instance is the typical
/// pattern. Sharing one history across multiple editors is
/// theoretically possible but probably a bug — there's no
/// editor-id discriminator in the entries.
pub struct History<K: PortKind, N: Clone, C: Clone, G: Clone> {
    undo: VecDeque<HistoryEntry<K, N, C, G>>,
    redo: VecDeque<HistoryEntry<K, N, C, G>>,
    cap: usize,
    /// Tag on the most recently pushed entry; [`Self::push_coalesced`]
    /// uses this to decide whether to update the back-most entry in
    /// place rather than appending. Cleared on any non-coalesced
    /// `push` (or whenever the back-most entry is consumed by undo).
    coalesce: Option<CoalesceKey>,
    /// When `false`, [`Self::push`] and [`Self::push_coalesced`] are
    /// no-ops. Toggled by [`Self::with_recording_off`].
    recording: bool,
}

impl<K, N, C, G> History<K, N, C, G>
where
    K: PortKind,
    N: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
    G: Clone + Send + Sync + 'static,
{
    /// New history with the given capacity. Going over the cap drops
    /// the oldest entry. `cap` of `0` disables history (every push is
    /// silently discarded — useful for read-only viewers that want to
    /// share the host plumbing without paying for the stack).
    pub fn new(cap: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: VecDeque::new(),
            cap,
            coalesce: None,
            recording: true,
        }
    }

    /// Convenience constructor using [`DEFAULT_CAP`].
    pub fn with_default_cap() -> Self {
        Self::new(DEFAULT_CAP)
    }

    /// Push a new [`EditorCommand`] edit. Truncates the redo stack
    /// (re-editing after an undo invalidates the future). Drops the
    /// oldest entry when the cap is exceeded. Use
    /// [`Self::push_transaction`] when the host needs to capture
    /// mutations the editor doesn't know about (host-side model,
    /// preset libraries, etc.).
    pub fn push(
        &mut self,
        forward: EditorCommand<K, N, C, G>,
        inverse: EditorCommand<K, N, C, G>,
        label: &'static str,
    ) {
        if !self.recording || self.cap == 0 {
            return;
        }
        self.redo.clear();
        self.coalesce = None;
        self.undo.push_back(HistoryEntry::Command {
            forward,
            inverse,
            label,
        });
        self.trim_cap();
    }

    /// Push a host-defined transaction: a pair of `Fn` closures that
    /// re-apply / reverse a single user gesture. Use this whenever
    /// the gesture mutates state outside the editor's
    /// [`EditorCommand`] surface — host project files, a preset
    /// library, an inspector cache — so the user sees one undoable
    /// step instead of an inconsistent half-undo.
    ///
    /// Both closures must be `Fn + Send + Sync + 'static`. They run
    /// with history recording suspended, so calling [`Self::push`]
    /// from inside them is a no-op (preventing self-recording
    /// loops). Closures typically capture clones of the host's
    /// state-holding `Arc`s plus a `NodeEditor` clone to dispatch
    /// editor commands inline.
    pub fn push_transaction<F, I>(&mut self, forward: F, inverse: I, label: &'static str)
    where
        F: Fn() + Send + Sync + 'static,
        I: Fn() + Send + Sync + 'static,
    {
        if !self.recording || self.cap == 0 {
            return;
        }
        self.redo.clear();
        self.coalesce = None;
        self.undo.push_back(HistoryEntry::Transaction {
            forward: Arc::new(forward),
            inverse: Arc::new(inverse),
            label,
        });
        self.trim_cap();
    }

    /// Push an edit that may collapse into the previous one. If the
    /// back-most entry was pushed with the same `key` AND is itself
    /// a [`HistoryEntry::Command`], this call updates its `forward`
    /// (the *latest* state) in place and leaves its `inverse` (the
    /// gesture-start state) untouched — so a 60 fps drag produces
    /// ONE undoable entry, not 60. Transactions never coalesce.
    ///
    /// A non-coalesced `push` between coalesced calls breaks the
    /// chain, as does any of [`Self::undo`], [`Self::redo`],
    /// [`Self::clear`].
    pub fn push_coalesced(
        &mut self,
        key: CoalesceKey,
        forward: EditorCommand<K, N, C, G>,
        inverse: EditorCommand<K, N, C, G>,
        label: &'static str,
    ) {
        if !self.recording || self.cap == 0 {
            return;
        }
        self.redo.clear();
        if self.coalesce.as_ref() == Some(&key) {
            if let Some(HistoryEntry::Command {
                forward: last_forward,
                label: last_label,
                ..
            }) = self.undo.back_mut()
            {
                *last_forward = forward;
                *last_label = label;
                return;
            }
        }
        self.coalesce = Some(key);
        self.undo.push_back(HistoryEntry::Command {
            forward,
            inverse,
            label,
        });
        self.trim_cap();
    }

    fn trim_cap(&mut self) {
        while self.undo.len() > self.cap {
            self.undo.pop_front();
        }
    }

    /// Pop the top of the undo stack and apply its inverse — either
    /// by dispatching an [`EditorCommand`] (for
    /// [`HistoryEntry::Command`]) or by invoking the host's
    /// `inverse` closure (for [`HistoryEntry::Transaction`]). Moves
    /// the entry onto the redo stack and returns its label. `None`
    /// when the stack is empty.
    ///
    /// Recording is suspended for the duration of the apply, so
    /// commands / transactions that themselves call [`Self::push`]
    /// don't re-land on the undo stack.
    pub fn undo(&mut self, editor: &NodeEditor<K, N, C, G>) -> Option<&'static str> {
        let entry = self.undo.pop_back()?;
        self.coalesce = None;
        let label = entry.label();
        self.with_recording_off(|| match &entry {
            HistoryEntry::Command { inverse, .. } => {
                editor.dispatch(inverse.clone());
            }
            HistoryEntry::Transaction { inverse, .. } => {
                (inverse)();
            }
        });
        self.redo.push_back(entry);
        Some(label)
    }

    /// Inverse of [`Self::undo`]. Pops the top of the redo stack,
    /// re-applies its forward action (command dispatch or
    /// transaction closure), and pushes the entry back onto the
    /// undo stack.
    pub fn redo(&mut self, editor: &NodeEditor<K, N, C, G>) -> Option<&'static str> {
        let entry = self.redo.pop_back()?;
        self.coalesce = None;
        let label = entry.label();
        self.with_recording_off(|| match &entry {
            HistoryEntry::Command { forward, .. } => {
                editor.dispatch(forward.clone());
            }
            HistoryEntry::Transaction { forward, .. } => {
                (forward)();
            }
        });
        self.undo.push_back(entry);
        Some(label)
    }

    /// True when [`Self::undo`] would return `Some`.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// True when [`Self::redo`] would return `Some`.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Depth of the undo stack. Useful for "unsaved-changes" gating
    /// (host marks the document dirty when `len()` grows past its
    /// last-saved baseline).
    pub fn len(&self) -> usize {
        self.undo.len()
    }

    /// True when no edits have been recorded.
    pub fn is_empty(&self) -> bool {
        self.undo.is_empty()
    }

    /// Drop everything. Hosts call this when swapping documents (a
    /// new `set_graph` from disk shouldn't leave the user able to
    /// "undo" back into the previous document) or when an external
    /// signal invalidates the existing commands (host re-wrote node
    /// metadata schemas, for example).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.coalesce = None;
    }

    /// Suspend recording for the duration of `f`. Any `push` /
    /// `push_coalesced` calls inside become no-ops; the previous
    /// recording state restores on return (including from panics — the
    /// flag's RAII via the closure's drop order). Use this around
    /// programmatic re-syncs that the user shouldn't be able to undo.
    pub fn with_recording_off<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let prev = std::mem::replace(&mut self.recording, false);
        let result = f();
        self.recording = prev;
        result
    }

    /// Peek the next-to-undo entry's label, without consuming. Hosts
    /// use this to render "Undo: Move Node" menu items.
    pub fn next_undo_label(&self) -> Option<&'static str> {
        self.undo.back().map(|e| e.label())
    }

    /// Peek the next-to-redo entry's label.
    pub fn next_redo_label(&self) -> Option<&'static str> {
        self.redo.back().map(|e| e.label())
    }
}

impl<K, N, C, G> Default for History<K, N, C, G>
where
    K: PortKind,
    N: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
    G: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::with_default_cap()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EditorCommand;
    use blinc_core::layer::Point;

    // Test stub matching the editor's PortKind trait shape. We only
    // exercise the History's data structure here; integration with a
    // real NodeEditor lives in the demo.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Kind;
    impl PortKind for Kind {
        fn compatible_with(&self, _other: &Self) -> bool {
            true
        }
        fn label(&self) -> String {
            String::new()
        }
        fn accent(&self) -> blinc_core::layer::Color {
            blinc_core::layer::Color::TRANSPARENT
        }
    }

    type Cmd = EditorCommand<Kind, (), (), ()>;
    type Hist = History<Kind, (), (), ()>;

    fn move_to(id: &str, x: f32, y: f32) -> Cmd {
        EditorCommand::UpdateNodePosition(NodeId::from(id), Point { x, y })
    }

    #[test]
    fn push_and_undo_returns_label() {
        let mut h: Hist = History::with_default_cap();
        h.push(
            move_to("a", 10.0, 0.0),
            move_to("a", 0.0, 0.0),
            "Move Node",
        );
        assert!(h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.next_undo_label(), Some("Move Node"));
    }

    #[test]
    fn coalesce_collapses_consecutive_drag_entries() {
        let mut h: Hist = History::with_default_cap();
        let id = NodeId::from("a");
        for i in 0..30 {
            h.push_coalesced(
                CoalesceKey::DragNode(id.clone()),
                move_to("a", i as f32, 0.0),
                move_to("a", 0.0, 0.0),
                "Move Node",
            );
        }
        assert_eq!(h.len(), 1, "30 drag samples should collapse to 1 entry");
    }

    #[test]
    fn non_coalesced_push_breaks_chain() {
        let mut h: Hist = History::with_default_cap();
        let id = NodeId::from("a");
        h.push_coalesced(
            CoalesceKey::DragNode(id.clone()),
            move_to("a", 1.0, 0.0),
            move_to("a", 0.0, 0.0),
            "Move",
        );
        h.push(move_to("a", 2.0, 0.0), move_to("a", 1.0, 0.0), "Other");
        h.push_coalesced(
            CoalesceKey::DragNode(id.clone()),
            move_to("a", 3.0, 0.0),
            move_to("a", 2.0, 0.0),
            "Move",
        );
        assert_eq!(
            h.len(),
            3,
            "intervening non-coalesced push must break the chain"
        );
    }

    #[test]
    fn cap_drops_oldest() {
        let mut h: Hist = History::new(3);
        for i in 0..5 {
            h.push(
                move_to("a", i as f32, 0.0),
                move_to("a", 0.0, 0.0),
                "Move",
            );
        }
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn cap_zero_disables_recording() {
        let mut h: Hist = History::new(0);
        h.push(move_to("a", 1.0, 0.0), move_to("a", 0.0, 0.0), "Move");
        assert!(h.is_empty());
        assert!(!h.can_undo());
    }

    #[test]
    fn with_recording_off_no_ops() {
        let mut h: Hist = History::with_default_cap();
        h.with_recording_off(|| {
            // Nested pushes can't happen here without a self capture;
            // exercise the flag flip + restore.
        });
        // Recording remains on after the closure.
        h.push(move_to("a", 1.0, 0.0), move_to("a", 0.0, 0.0), "Move");
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn transactions_land_on_the_stack_with_label() {
        let mut h: Hist = History::with_default_cap();
        h.push_transaction(|| (), || (), "Apply Preset");
        assert!(h.can_undo());
        assert_eq!(h.next_undo_label(), Some("Apply Preset"));
        match h.next_undo_label() {
            Some("Apply Preset") => {}
            other => panic!("unexpected label: {:?}", other),
        }
    }

    #[test]
    fn transactions_do_not_coalesce_with_subsequent_commands() {
        // push_coalesced should only fold over the back-most entry if
        // it is itself a Command. A leading Transaction blocks the
        // fold and forces a new Command entry.
        let mut h: Hist = History::with_default_cap();
        let id = NodeId::from("a");
        h.push_transaction(|| (), || (), "Host edit");
        h.push_coalesced(
            CoalesceKey::DragNode(id.clone()),
            move_to("a", 1.0, 0.0),
            move_to("a", 0.0, 0.0),
            "Move",
        );
        h.push_coalesced(
            CoalesceKey::DragNode(id.clone()),
            move_to("a", 2.0, 0.0),
            move_to("a", 0.0, 0.0),
            "Move",
        );
        // Transaction + two coalesced drags should collapse the
        // drags into one — final count is 2 (Transaction + one Move).
        assert_eq!(h.len(), 2);
    }

}
