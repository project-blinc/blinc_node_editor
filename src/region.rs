//! Typed canvas-kit hit-region identifiers.
//!
//! The editor uses canvas-kit's region map (string keys) to address
//! every interactive surface — node bodies, port dots, edge curves,
//! group chrome chips. Each element kind has its own wire prefix
//! (`node:`, `port:`, `group:`, `edge:`, …) and the full region key
//! is `{prefix}{payload}`. Pre-2.x this was stringly-typed at every
//! call site (`format!("node:{}", id.as_str())` + `strip_prefix("node:")`)
//! which doesn't scale: a typo on one side never reaches the other,
//! and adding a new region kind is a needle-in-a-haystack hunt for
//! every `format!` that has to learn the new prefix.
//!
//! [`RegionId`] is the typed enum that holds the entire surface. It
//! round-trips through `encode()` (typed → canvas-kit string) and
//! `parse()` (string → typed); the wire format matches the legacy
//! string layout exactly so existing canvas-kit state, serialised
//! debug dumps, and any host code still constructing region strings
//! by hand continue to work.
//!
//! Migration pattern:
//!
//! ```ignore
//! // before — stringly typed:
//! let region = format!("node:{}", id.as_str());
//! if let Some(s) = region.strip_prefix("node:") {
//!     let id = NodeId::from(s);
//! }
//!
//! // after — typed:
//! let region = RegionId::Node(id.clone()).encode();
//! match RegionId::parse(&region) {
//!     Some(RegionId::Node(id)) => { /* … */ }
//!     _ => { /* … */ }
//! }
//! ```
//!
//! Adding a new region kind becomes a one-line `enum` addition + two
//! lines in `encode` / `parse`; the editor's `match` arms over
//! [`RegionId`] surface every unhandled case as a compile error.

use crate::connection::ConnectionId;
use crate::group::GroupId;
use crate::node::NodeId;
use crate::port::{PortAddress, PortId};

/// Wire prefix written into canvas-kit region keys for every
/// element kind the editor manages.
///
/// The associated payload type per variant:
///
/// | Variant            | Payload          | Wire format                  |
/// | ------------------ | ---------------- | ---------------------------- |
/// | `Node`             | [`NodeId`]       | `node:{node_id}`             |
/// | `NodeBadge`        | [`NodeId`]       | `node_badge:{node_id}`       |
/// | `Port`             | [`PortAddress`]  | `port:{node_id}:{port_id}`   |
/// | `Group`            | [`GroupId`]      | `group:{group_id}`           |
/// | `GroupBadge`       | [`GroupId`]      | `group_badge:{group_id}`     |
/// | `GroupCollapse`    | [`GroupId`]      | `group_collapse:{group_id}`  |
/// | `GroupDelete`      | [`GroupId`]      | `group_delete:{group_id}`    |
/// | `GroupDesc`        | [`GroupId`]      | `group_desc:{group_id}`      |
/// | `GroupEdit`        | [`GroupId`]      | `group_edit:{group_id}`      |
/// | `GroupTitle`       | [`GroupId`]      | `group_title:{group_id}`     |
/// | `Edge`             | [`ConnectionId`] | `edge:{u64}`                 |
/// | `EdgeDelete`       | [`ConnectionId`] | `edge_delete:{u64}`          |
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RegionId {
    Node(NodeId),
    NodeBadge(NodeId),
    Port(PortAddress),
    Group(GroupId),
    GroupBadge(GroupId),
    GroupCollapse(GroupId),
    GroupDelete(GroupId),
    GroupDesc(GroupId),
    GroupEdit(GroupId),
    GroupTitle(GroupId),
    Edge(ConnectionId),
    EdgeDelete(ConnectionId),
}

impl RegionId {
    /// Static wire prefix for this variant (includes the trailing
    /// colon). Exposed for tracing / debug formatting; encoders
    /// should call [`encode`](Self::encode) instead so the payload
    /// is appended consistently.
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::Node(_) => "node:",
            Self::NodeBadge(_) => "node_badge:",
            Self::Port(_) => "port:",
            Self::Group(_) => "group:",
            Self::GroupBadge(_) => "group_badge:",
            Self::GroupCollapse(_) => "group_collapse:",
            Self::GroupDelete(_) => "group_delete:",
            Self::GroupDesc(_) => "group_desc:",
            Self::GroupEdit(_) => "group_edit:",
            Self::GroupTitle(_) => "group_title:",
            Self::Edge(_) => "edge:",
            Self::EdgeDelete(_) => "edge_delete:",
        }
    }

    /// Encode to the canvas-kit string form.
    ///
    /// **Port contract**: a `Port` variant's `PortAddress.port` MUST
    /// NOT contain a colon. The encoder joins on `:` but the decoder
    /// uses `rsplit_once(':')` so the split happens at the LAST
    /// colon — this lets node ids contain colons (reflow's
    /// `"component:instance"` shape) but means a colon in `port`
    /// would silently shift the node/port boundary on decode. A
    /// `debug_assert!` enforces the rule in debug builds; release
    /// builds will produce a region key that round-trips to the
    /// wrong `PortAddress`.
    pub fn encode(&self) -> String {
        match self {
            Self::Node(id) => format!("node:{}", id.as_str()),
            Self::NodeBadge(id) => format!("node_badge:{}", id.as_str()),
            Self::Port(addr) => {
                debug_assert!(
                    !addr.port.as_str().contains(':'),
                    "RegionId::Port: port id must not contain ':' \
                     (port=`{}`, node=`{}`); decode would split at the \
                     wrong colon. See region.rs encode/decode contract.",
                    addr.port.as_str(),
                    addr.node.as_str(),
                );
                format!("port:{}:{}", addr.node.as_str(), addr.port.as_str())
            }
            Self::Group(id) => format!("group:{}", id.as_str()),
            Self::GroupBadge(id) => format!("group_badge:{}", id.as_str()),
            Self::GroupCollapse(id) => format!("group_collapse:{}", id.as_str()),
            Self::GroupDelete(id) => format!("group_delete:{}", id.as_str()),
            Self::GroupDesc(id) => format!("group_desc:{}", id.as_str()),
            Self::GroupEdit(id) => format!("group_edit:{}", id.as_str()),
            Self::GroupTitle(id) => format!("group_title:{}", id.as_str()),
            Self::Edge(id) => format!("edge:{}", id.0),
            Self::EdgeDelete(id) => format!("edge_delete:{}", id.0),
        }
    }

    /// Parse from the canvas-kit string form. Returns `None` when
    /// the string doesn't match a known prefix, or when the payload
    /// is malformed (e.g. `edge:notanumber`, `port:nocolon`).
    ///
    /// Prefix matching is longest-first so `node_badge:` resolves
    /// to [`RegionId::NodeBadge`] rather than a `RegionId::Node`
    /// whose payload starts with `_badge:`.
    pub fn parse(region: &str) -> Option<Self> {
        // node_badge: before node: (the longer prefix wins).
        if let Some(rest) = region.strip_prefix("node_badge:") {
            return Some(Self::NodeBadge(NodeId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("node:") {
            return Some(Self::Node(NodeId::from(rest)));
        }

        if let Some(rest) = region.strip_prefix("port:") {
            // Split at the LAST colon so node ids that contain a
            // colon (reflow's `"component:instance"` shape, host
            // synthesised composite ids) round-trip — the port id
            // is the tail past the final colon.
            let (node_raw, port_raw) = rest.rsplit_once(':')?;
            if node_raw.is_empty() || port_raw.is_empty() {
                return None;
            }
            return Some(Self::Port(PortAddress {
                node: NodeId::from(node_raw),
                port: PortId::from(port_raw),
            }));
        }

        // group_* before group: (the longer prefix wins).
        if let Some(rest) = region.strip_prefix("group_badge:") {
            return Some(Self::GroupBadge(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group_collapse:") {
            return Some(Self::GroupCollapse(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group_delete:") {
            return Some(Self::GroupDelete(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group_desc:") {
            return Some(Self::GroupDesc(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group_edit:") {
            return Some(Self::GroupEdit(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group_title:") {
            return Some(Self::GroupTitle(GroupId::from(rest)));
        }
        if let Some(rest) = region.strip_prefix("group:") {
            return Some(Self::Group(GroupId::from(rest)));
        }

        // edge_delete: before edge:.
        if let Some(rest) = region.strip_prefix("edge_delete:") {
            return Some(Self::EdgeDelete(ConnectionId(rest.parse().ok()?)));
        }
        if let Some(rest) = region.strip_prefix("edge:") {
            return Some(Self::Edge(ConnectionId(rest.parse().ok()?)));
        }

        None
    }

    /// Collapse any node sub-region (`Node` / `NodeBadge`) to its
    /// `NodeId`. Returns `None` for non-node regions. Convenience
    /// for selection / hit-test sites that don't care which surface
    /// of the node was clicked.
    pub fn as_node(&self) -> Option<&NodeId> {
        match self {
            Self::Node(id) | Self::NodeBadge(id) => Some(id),
            _ => None,
        }
    }

    /// Collapse any group sub-region (`Group` / `GroupTitle` /
    /// `GroupDesc` / chrome chips) to its `GroupId`. Returns `None`
    /// for non-group regions.
    pub fn as_group(&self) -> Option<&GroupId> {
        match self {
            Self::Group(id)
            | Self::GroupBadge(id)
            | Self::GroupCollapse(id)
            | Self::GroupDelete(id)
            | Self::GroupDesc(id)
            | Self::GroupEdit(id)
            | Self::GroupTitle(id) => Some(id),
            _ => None,
        }
    }

    /// Collapse any edge sub-region (`Edge` / `EdgeDelete`) to its
    /// `ConnectionId`.
    pub fn as_edge(&self) -> Option<ConnectionId> {
        match self {
            Self::Edge(id) | Self::EdgeDelete(id) => Some(*id),
            _ => None,
        }
    }
}

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.encode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> NodeId {
        NodeId::from(s)
    }
    fn g(s: &str) -> GroupId {
        GroupId::from(s)
    }

    #[test]
    fn round_trips_every_variant() {
        let cases: Vec<RegionId> = vec![
            RegionId::Node(n("n42")),
            RegionId::NodeBadge(n("n42")),
            RegionId::Port(PortAddress {
                node: n("n42"),
                port: PortId::from("out_value"),
            }),
            RegionId::Group(g("g7")),
            RegionId::GroupBadge(g("g7")),
            RegionId::GroupCollapse(g("g7")),
            RegionId::GroupDelete(g("g7")),
            RegionId::GroupDesc(g("g7")),
            RegionId::GroupEdit(g("g7")),
            RegionId::GroupTitle(g("g7")),
            RegionId::Edge(ConnectionId(1234)),
            RegionId::EdgeDelete(ConnectionId(1234)),
        ];
        for r in cases {
            let s = r.encode();
            let parsed = RegionId::parse(&s)
                .unwrap_or_else(|| panic!("failed to parse own encoding: {s:?}"));
            assert_eq!(r, parsed, "round-trip mismatch for {s:?}");
        }
    }

    #[test]
    fn wire_format_matches_legacy_strings() {
        // These ARE the strings the editor used to build via
        // `format!("node:{}", id.as_str())` etc. Anything serialised
        // by older builds, stashed in tests, or constructed by host
        // code must keep parsing identically.
        assert_eq!(RegionId::Node(n("alpha")).encode(), "node:alpha");
        assert_eq!(RegionId::NodeBadge(n("alpha")).encode(), "node_badge:alpha");
        assert_eq!(
            RegionId::Port(PortAddress {
                node: n("alpha"),
                port: PortId::from("p0"),
            })
            .encode(),
            "port:alpha:p0"
        );
        assert_eq!(RegionId::Group(g("g0")).encode(), "group:g0");
        assert_eq!(RegionId::GroupBadge(g("g0")).encode(), "group_badge:g0");
        assert_eq!(
            RegionId::GroupCollapse(g("g0")).encode(),
            "group_collapse:g0"
        );
        assert_eq!(RegionId::GroupDelete(g("g0")).encode(), "group_delete:g0");
        assert_eq!(RegionId::GroupDesc(g("g0")).encode(), "group_desc:g0");
        assert_eq!(RegionId::GroupEdit(g("g0")).encode(), "group_edit:g0");
        assert_eq!(RegionId::GroupTitle(g("g0")).encode(), "group_title:g0");
        assert_eq!(RegionId::Edge(ConnectionId(99)).encode(), "edge:99");
        assert_eq!(
            RegionId::EdgeDelete(ConnectionId(99)).encode(),
            "edge_delete:99"
        );
    }

    #[test]
    fn longer_prefixes_win_over_shorter_substrings() {
        // `node_badge:foo` must NOT parse as `RegionId::Node` with
        // payload `_badge:foo`.
        let r = RegionId::parse("node_badge:foo").unwrap();
        assert!(matches!(r, RegionId::NodeBadge(_)));
        let r = RegionId::parse("group_delete:g1").unwrap();
        assert!(matches!(r, RegionId::GroupDelete(_)));
        let r = RegionId::parse("edge_delete:7").unwrap();
        assert!(matches!(r, RegionId::EdgeDelete(_)));
    }

    #[test]
    fn unknown_prefix_returns_none() {
        assert!(RegionId::parse("unrelated:foo").is_none());
        assert!(RegionId::parse("").is_none());
        assert!(RegionId::parse("nope").is_none());
    }

    #[test]
    fn malformed_payload_returns_none() {
        // edge: payload must be a u64.
        assert!(RegionId::parse("edge:abc").is_none());
        assert!(RegionId::parse("edge_delete:abc").is_none());
        // port: payload must contain at least one colon between
        // node + port id segments, and neither side empty.
        assert!(RegionId::parse("port:onlynode").is_none());
        assert!(RegionId::parse("port::tail").is_none());
        assert!(RegionId::parse("port:head:").is_none());
    }

    #[test]
    #[should_panic(expected = "port id must not contain ':'")]
    fn port_id_with_colon_trips_debug_assert() {
        // Symmetric guard to port_payload_splits_at_last_colon: the
        // decoder splits at the LAST colon to support node ids with
        // colons; the encoder asserts the port id is colon-free so
        // a malformed PortAddress can't silently round-trip to the
        // wrong fields. Only fires in debug builds.
        let _ = RegionId::Port(PortAddress {
            node: n("n0"),
            port: PortId::from("has:colon"),
        })
        .encode();
    }

    #[test]
    fn port_payload_splits_at_last_colon() {
        // Node id may contain colons (reflow's
        // "component:instance" style). The port id is always the
        // tail past the FINAL colon.
        let r = RegionId::parse("port:foo:bar:p0").unwrap();
        match r {
            RegionId::Port(addr) => {
                assert_eq!(addr.node.as_str(), "foo:bar");
                assert_eq!(addr.port.as_str(), "p0");
            }
            _ => panic!("expected Port"),
        }
    }

    #[test]
    fn as_node_collapses_node_variants() {
        assert_eq!(RegionId::Node(n("n0")).as_node(), Some(&n("n0")));
        assert_eq!(RegionId::NodeBadge(n("n0")).as_node(), Some(&n("n0")));
        assert!(RegionId::Group(g("g0")).as_node().is_none());
    }

    #[test]
    fn as_group_collapses_every_group_variant() {
        let gid = g("g0");
        for r in [
            RegionId::Group(gid.clone()),
            RegionId::GroupBadge(gid.clone()),
            RegionId::GroupCollapse(gid.clone()),
            RegionId::GroupDelete(gid.clone()),
            RegionId::GroupDesc(gid.clone()),
            RegionId::GroupEdit(gid.clone()),
            RegionId::GroupTitle(gid.clone()),
        ] {
            assert_eq!(
                r.as_group(),
                Some(&gid),
                "expected group accessor for {r:?}"
            );
        }
        assert!(RegionId::Node(n("n0")).as_group().is_none());
    }

    #[test]
    fn display_emits_encoded_form() {
        let r = RegionId::Node(n("n0"));
        assert_eq!(format!("{r}"), "node:n0");
    }
}
