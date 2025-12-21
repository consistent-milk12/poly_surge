//! # `poly_surge`
//!
//! **Somewhat novel**, fast incremental polytope surgery in Rust. Add and remove
//! halfspaces, **19x-713x faster** than the standard "just reconstruct it" approach.
//!
//! ## What is this?
//!
//! A convex polytope in 3D is a bounded region defined by the intersection of
//! half-spaces (think of it as a 3D shape carved out by planes). This crate lets
//! you **dynamically add and remove** these half-space constraints while preserving
//! the mesh topology (vertices, edges, faces)—no full reconstruction needed.
//!
//! ## Quick Start
//!
//! ```rust
//! use poly_surge::{IncrementalPolytope, HalfSpace};
//! use glam::DVec3;
//!
//! // Create a polytope and add planes to form a cube
//! let mut polytope = IncrementalPolytope::new();
//!
//! // Six faces of a unit cube centered at origin
//! let planes = [
//!     HalfSpace::new(DVec3::X, 0.5),    // +X face
//!     HalfSpace::new(-DVec3::X, 0.5),   // -X face
//!     HalfSpace::new(DVec3::Y, 0.5),    // +Y face
//!     HalfSpace::new(-DVec3::Y, 0.5),   // -Y face
//!     HalfSpace::new(DVec3::Z, 0.5),    // +Z face
//!     HalfSpace::new(-DVec3::Z, 0.5),   // -Z face
//! ];
//!
//! for plane in planes {
//!     polytope.add_plane(plane);
//! }
//!
//! assert_eq!(polytope.vertex_count(), 8);  // Cube has 8 vertices
//! assert!(polytope.is_bounded());
//!
//! // Remove one face - the cube becomes an infinite prism
//! let plane_to_remove = polytope.planes().next().unwrap().0;
//! polytope.remove_plane(plane_to_remove);
//!
//! assert!(!polytope.is_bounded());  // Now unbounded
//! ```
//!
//! ## Key Features
//!
//! - **Incremental construction**: Add planes one at a time with `O(V + E)` clipping
//! - **Topology-based removal**: Remove planes in ~`O(N)` typical time (vs `O(N log N + V)` rebuild)
//! - **Lazy evaluation**: Edges and face orderings built on-demand
//! - **Unbounded support**: Handles infinite polytopes via homogeneous coordinates
//! - **Lazy boundedness check**: `O(1)` query when cached, `O(N log N)` rebuild when dirty
//!
//! ## When to Use
//!
//! - Dynamic Voronoi diagrams (cell modification)
//! - CSG operations on halfspace-defined solids
//! - Robot motion planning (C-space constraint relaxation)
//! - Interactive geometry editors
//! - Any application where planes are added/removed frequently
//!
//! ## When NOT to Use
//!
//! - N > 10,000 planes (consider specialized data structures)
//! - Exact arithmetic required (we use f64 with epsilon tolerance)
//! - Batch deletion of many planes at once (single rebuild may be faster)
//!
//! ## Algorithm
//!
//! The removal algorithm exploits vertex topology: when a plane is removed, vertices
//! with only 2 remaining incident planes lie on a line. We search along that line to
//! find where it hits other constraints, reconstructing the "unclipped" geometry.
//!
//! See the [research proposal](https://gitlab.com/dryad1/demiurge) for details.

#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

mod incremental_hull;
mod incremental_polytope;

pub use incremental_polytope::{
    AddPlaneResult, Edge, EdgeIdx, HalfSpace, IncrementalPolytope, PlaneIdx, RemovePlaneResult,
    TopologyError, Vertex, VertexIdx,
};

pub use incremental_hull::IncrementalHull;

/// Re-export glam types for convenience
pub mod math {
    pub use glam::{DMat3, DVec3, DVec4};
}
