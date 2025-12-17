//! # Incremental Convex Polytope Construction
//!
//! This module implements the **incremental convex polytope** algorithm, which
//! builds a 3D convex polytope by adding half-space constraints one at a time.
//!
//! ## What is a Convex Polytope?
//!
//! A convex polytope in 3D is a bounded region defined by the intersection of
//! half-spaces. Think of it as a 3D shape where every point on a line segment
//! between two points in the shape is also in the shape (convexity).
//!
//! ## Key Concepts
//!
//! - **Half-space**: Region on one side of a plane: `{ x : n·x ≤ d }`
//! - **Vertex**: Point where 3+ half-space boundaries meet
//! - **Edge**: Line segment where exactly 2 half-space boundaries meet
//! - **Face**: Polygon on one half-space boundary
//!
//! ## Algorithm Overview
//!
//! 1. **Initial formation**: With ≥4 planes, find vertices by solving 3-plane
//!    intersections
//! 2. **Clipping**: Each new plane clips the polytope via Sutherland-Hodgman
//!    style algorithm
//! 3. **Lazy construction**: Edges and face orderings are built on-demand
//! 4. **Topology-based removal**: Planes can be removed with `O(V_r × K² × N)`
//!    reconstruction
//!
//! ## Complexity
//!
//! | Operation            | Complexity      | Notes                         |
//! |---------------------|-----------------|-------------------------------|
//! | Add plane (clip)    | O(V + E)        | Classify vertices, clip edges |
//! | Add plane (initial) | O(N³) worst     | Try 3-plane combinations      |
//! | Remove plane        | O(V_r × K² × N) | Topology-based reconstruction |
//! | Duplicate detection | O(1) expected   | Spatial hashing               |

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DVec3, DVec4};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::incremental_hull::IncrementalHull;

const EPSILON: f64 = 1e-7;

/// Estimated number of new vertices per removal (for pre-allocation).
const ESTIMATED_NEW_VERTICES: usize = 8;

/// Minimum vertex count before using edge walking for face vertex enumeration.
/// For small polytopes, direct O(V) filtering is faster than `O(E + V_f × deg)` edge walking
/// due to the overhead of finding the initial edge.
const EDGE_WALK_VERTEX_THRESHOLD: usize = 20;

/// Extract the xyz components from a `DVec4` (homogeneous coordinates).
#[inline]
const fn dvec4_xyz(v: &DVec4) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

/// Grid-based spatial hash for O(1) expected-time duplicate detection.
///
/// # How It Works
///
/// Divides 3D space into a grid of cells. To check if a point is a duplicate:
/// 1. Compute which cell the point falls into
/// 2. Check that cell + 26 neighbors (3×3×3 cube)
/// 3. Compare distances only to points in those cells
///
/// This avoids O(n) comparisons against all existing points.
///
/// Uses `FxHashMap` for faster hashing (non-cryptographic, ~2-3x faster).
#[derive(Clone)]
struct SpatialHash {
    cells: FxHashMap<(i64, i64, i64), Vec<DVec3>>,
    cell_size: f64,
    tolerance: f64,
}

impl SpatialHash {
    fn new(tolerance: f64) -> Self {
        // Cell size = 2× tolerance ensures duplicates are in adjacent cells
        Self {
            cells: FxHashMap::default(),
            cell_size: tolerance * 2.0,
            tolerance,
        }
    }

    /// Create with pre-allocated capacity for expected number of cells.
    fn with_capacity(tolerance: f64, capacity: usize) -> Self {
        Self {
            cells: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            cell_size: tolerance * 2.0,
            tolerance,
        }
    }

    /// Reset tolerance and clear (for reuse with different epsilon).
    fn reset(&mut self, tolerance: f64) {
        self.cell_size = tolerance * 2.0;
        self.tolerance = tolerance;
        self.cells.clear();
    }

    /// Map a point to its grid cell indices.
    #[inline]
    fn cell_coords(&self, p: DVec3) -> (i64, i64, i64) {
        #[expect(clippy::cast_possible_truncation)]
        let discretize = |v: f64| (v / self.cell_size).floor() as i64;
        (discretize(p.x), discretize(p.y), discretize(p.z))
    }

    /// Check 27 neighboring cells for any point within tolerance.
    fn is_duplicate(&self, point: DVec3) -> bool {
        let (cx, cy, cz) = self.cell_coords(point);
        // Check 3×3×3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(pts) = self.cells.get(&(cx + dx, cy + dy, cz + dz))
                        && pts.iter().any(|&p| (p - point).length() < self.tolerance)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline]
    fn insert(&mut self, point: DVec3) {
        self.cells
            .entry(self.cell_coords(point))
            .or_default()
            .push(point);
    }

    /// Insert only if not a duplicate. Returns true if inserted.
    #[inline]
    fn insert_if_unique(&mut self, point: DVec3) -> bool {
        if self.is_duplicate(point) {
            false
        } else {
            self.insert(point);
            true
        }
    }
}

// TYPE-SAFE INDICES - Prevent mixing up different index types at compile time

/// Index into the planes array. Using a newtype prevents accidentally
/// passing a vertex index where a plane index is expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlaneIdx(pub usize);

/// Index into the vertices array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexIdx(pub usize);

/// Index into the edges array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeIdx(pub usize);

// CORE GEOMETRIC TYPES

/// A half-space constraint: `n · x ≤ d`
///
/// Geometrically, this divides 3D space into two regions:
/// - **Inside**: points where `n · x < d` (strictly satisfies constraint)
/// - **Boundary**: points where `n · x = d` (the plane itself)
/// - **Outside**: points where `n · x > d` (violates constraint)
///
/// The normal vector points toward the "outside" (forbidden) region.
#[derive(Clone, Debug)]
pub struct HalfSpace {
    /// Unit normal vector pointing toward the forbidden region.
    pub normal: DVec3,
    /// Signed distance from origin to the plane boundary.
    pub offset: f64,
}

impl HalfSpace {
    /// Create a half-space, normalizing the input normal vector.
    ///
    /// # Panics
    /// Panics if the normal vector has zero length.
    #[must_use]
    pub fn new(normal: DVec3, offset: f64) -> Self {
        let len = normal.length();
        assert!(len > EPSILON, "Normal vector must be non-zero");
        Self {
            normal: normal / len,
            offset: offset / len,
        }
    }

    /// Create from an already-normalized normal (debug-asserts unit length).
    #[must_use]
    pub fn new_normalized(normal: DVec3, offset: f64) -> Self {
        debug_assert!((normal.length() - 1.0).abs() < EPSILON * 100.0);
        Self { normal, offset }
    }

    /// Try to create, returning None if normal is zero.
    #[must_use]
    pub fn try_new(normal: DVec3, offset: f64) -> Option<Self> {
        let len = normal.length();
        (len >= EPSILON).then(|| Self {
            normal: normal / len,
            offset: offset / len,
        })
    }

    /// Classify a point: Inside (satisfies), On (boundary), or Outside
    /// (violates).
    #[must_use]
    pub fn classify(&self, point: DVec3, epsilon: f64) -> Classification {
        let d = self.signed_distance(point);
        if d < -epsilon {
            Classification::Inside
        } else if d > epsilon {
            Classification::Outside
        } else {
            Classification::On
        }
    }

    /// Signed distance: negative = inside, zero = on boundary, positive =
    /// outside.
    #[must_use]
    pub fn signed_distance(&self, point: DVec3) -> f64 {
        self.normal.dot(point) - self.offset
    }
}

/// Classification of a point relative to a half-space boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    /// Strictly satisfies the half-space constraint: `n·x < d - ε`
    Inside,

    /// On the boundary within tolerance: `|n·x - d| ≤ ε`
    On,

    /// Violates the half-space constraint: `n·x > d + ε`
    Outside,
}

/// A vertex: the intersection point of 3+ half-space boundaries.
///
/// In 3D, a vertex requires at least 3 planes to "pin" it in place.
/// More than 3 planes can meet at a vertex (degeneracy), e.g., the apex
/// of a square pyramid has 4 incident planes.
#[derive(Clone, Debug)]
pub struct Vertex {
    /// Homogeneous coordinates (x, y, z, w).
    ///
    /// - `w != 0`: Finite vertex at Euclidean position `(x/w, y/w, z/w)`
    /// - `w == 0`: Ideal vertex (point at infinity) in direction `(x, y, z)`
    pub position: DVec4,

    /// All planes passing through this vertex (≥3 for finite, ≥2 for ideal).
    pub planes: Vec<PlaneIdx>,

    /// Edges incident to this vertex.
    pub edges: Vec<EdgeIdx>,
}

impl Vertex {
    /// Returns `true` if this is a finite vertex (has a Euclidean position).
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.position.w.abs() > EPSILON
    }

    /// Returns the Euclidean position for finite vertices, or `None` for ideal vertices.
    #[must_use]
    pub fn to_euclidean(&self) -> Option<DVec3> {
        if self.is_finite() {
            Some(dvec4_xyz(&self.position) / self.position.w)
        } else {
            None
        }
    }

    /// Returns the direction to infinity for ideal vertices, or `None` for finite vertices.
    #[must_use]
    pub fn direction(&self) -> Option<DVec3> {
        if self.is_finite() {
            None
        } else {
            Some(dvec4_xyz(&self.position).normalize())
        }
    }
}

/// An edge: the line segment where exactly 2 half-space boundaries meet.
///
/// Edges connect two vertices and lie on the intersection line of two planes.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Defining planes (canonical order: planes.0 < planes.1).
    pub planes: (PlaneIdx, PlaneIdx),

    /// Endpoint vertices.
    pub vertices: (VertexIdx, VertexIdx),
}

impl Edge {
    /// Canonical ordering ensures (p1, p2) and (p2, p1) map to the same key.
    #[must_use]
    pub const fn canonical_planes(p1: PlaneIdx, p2: PlaneIdx) -> (PlaneIdx, PlaneIdx) {
        if p1.0 < p2.0 { (p1, p2) } else { (p2, p1) }
    }

    /// Returns `true` if this edge is bounded (both endpoints are finite vertices).
    #[must_use]
    pub fn is_bounded(&self, polytope: &IncrementalPolytope) -> bool {
        let v0 = polytope.vertex(self.vertices.0);
        let v1 = polytope.vertex(self.vertices.1);
        matches!((v0, v1), (Some(v0), Some(v1)) if v0.is_finite() && v1.is_finite())
    }

    /// Returns `true` if this edge is a ray (one finite, one ideal vertex).
    #[must_use]
    pub fn is_ray(&self, polytope: &IncrementalPolytope) -> bool {
        let v0 = polytope.vertex(self.vertices.0);
        let v1 = polytope.vertex(self.vertices.1);
        match (v0, v1) {
            (Some(v0), Some(v1)) => v0.is_finite() != v1.is_finite(),
            _ => false,
        }
    }

    /// Returns `true` if this edge is a line (both endpoints are ideal vertices).
    #[must_use]
    pub fn is_line(&self, polytope: &IncrementalPolytope) -> bool {
        let v0 = polytope.vertex(self.vertices.0);
        let v1 = polytope.vertex(self.vertices.1);
        matches!((v0, v1), (Some(v0), Some(v1)) if !v0.is_finite() && !v1.is_finite())
    }

    /// Returns the direction vector of this edge.
    ///
    /// For segments: direction from v0 to v1.
    /// For rays: direction toward infinity.
    /// For lines: direction computed from defining planes.
    #[must_use]
    pub fn direction(&self, polytope: &IncrementalPolytope) -> Option<DVec3> {
        let v0 = polytope.vertex(self.vertices.0)?;
        let v1 = polytope.vertex(self.vertices.1)?;

        match (v0.is_finite(), v1.is_finite()) {
            (true, true) => {
                // Segment: direction from v0 to v1
                let p0 = v0.to_euclidean()?;
                let p1 = v1.to_euclidean()?;
                Some((p1 - p0).normalize())
            }
            (true, false) => {
                // Ray: direction is v1's ideal direction
                v1.direction()
            }
            (false, true) => {
                // Ray (reversed): direction toward v1
                Some(-v0.direction()?)
            }
            (false, false) => {
                // Line: compute from defining planes
                let p0 = polytope.plane(self.planes.0)?;
                let p1 = polytope.plane(self.planes.1)?;
                let dir = p0.normal.cross(p1.normal);
                if dir.length_squared() < EPSILON * EPSILON {
                    None
                } else {
                    Some(dir.normalize())
                }
            }
        }
    }

    /// Returns the other endpoint of this edge given one endpoint.
    ///
    /// Returns `None` if `v` is not an endpoint of this edge.
    #[inline]
    #[must_use]
    pub fn other_vertex(&self, v: VertexIdx) -> Option<VertexIdx> {
        if self.vertices.0 == v {
            Some(self.vertices.1)
        } else if self.vertices.1 == v {
            Some(self.vertices.0)
        } else {
            None
        }
    }

    /// Returns `true` if this edge lies on the given plane.
    #[inline]
    #[must_use]
    pub fn is_on_plane(&self, plane_idx: PlaneIdx) -> bool {
        self.planes.0 == plane_idx || self.planes.1 == plane_idx
    }
}

// RESULT TYPES - Outcomes of polytope operations

/// Result of adding a plane to the polytope.
#[derive(Clone, Debug)]
pub enum AddPlaneResult {
    /// Need more planes to form a bounded region (< 4 planes).
    StillUnbounded,

    /// Plane doesn't change the polytope (all vertices already satisfy it).
    Redundant,

    /// Plane clipped the polytope, creating/removing vertices.
    Added {
        /// Vertices created by the new plane intersecting existing edges.
        new_vertices: Vec<VertexIdx>,

        /// Vertices that were clipped away (now outside the half-space).
        removed_vertices: Vec<VertexIdx>,

        /// Index of the newly added plane.
        plane_idx: PlaneIdx,
    },

    /// Constraints are infeasible (no region satisfies all planes).
    Empty,
}

/// Result of removing a plane from the polytope.
#[derive(Clone, Debug)]
pub struct RemovePlaneResult {
    /// Vertices removed (had < 3 remaining planes).
    pub removed_vertices: Vec<VertexIdx>,
    /// Vertices kept but modified (plane list updated).
    pub affected_vertices: Vec<VertexIdx>,
}

/// Topology validation errors.
///
/// These indicate inconsistencies in the polytope structure that may result
/// from numerical issues or bugs in the incremental algorithms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TopologyError {
    /// Euler characteristic mismatch: expected χ = 2 (bounded) or χ = 1 (unbounded).
    EulerMismatch {
        vertices: usize,
        edges: usize,
        faces: usize,
        expected: i32,
        actual: i32,
    },
    /// Finite vertex has fewer than 3 incident planes.
    UnderconstrainedVertex {
        vertex: VertexIdx,
        plane_count: usize,
    },
    /// Ideal vertex has fewer than 2 incident planes.
    InvalidIdealVertex {
        vertex: VertexIdx,
        plane_count: usize,
    },
    /// Edge references a non-existent vertex.
    DanglingEdge {
        edge: EdgeIdx,
        missing_vertex: VertexIdx,
    },
    /// Face (plane) has fewer than 3 vertices.
    DegenerateFace {
        plane: PlaneIdx,
        vertex_count: usize,
    },
    /// Duplicate edge key in edge map.
    DuplicateEdge { planes: (PlaneIdx, PlaneIdx) },
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EulerMismatch {
                vertices,
                edges,
                faces,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Euler mismatch: V={vertices}, E={edges}, F={faces}, χ={actual} (expected {expected})"
                )
            }
            Self::UnderconstrainedVertex {
                vertex,
                plane_count,
            } => {
                write!(
                    f,
                    "Vertex {vertex:?} has only {plane_count} incident planes (need ≥3)"
                )
            }
            Self::InvalidIdealVertex {
                vertex,
                plane_count,
            } => {
                write!(
                    f,
                    "Ideal vertex {vertex:?} has only {plane_count} incident planes (need ≥2)"
                )
            }
            Self::DanglingEdge {
                edge,
                missing_vertex,
            } => {
                write!(
                    f,
                    "Edge {edge:?} references non-existent vertex {missing_vertex:?}"
                )
            }

            Self::DegenerateFace {
                plane,
                vertex_count,
            } => {
                write!(
                    f,
                    "Face {plane:?} has only {vertex_count} vertices (need ≥3)"
                )
            }
            Self::DuplicateEdge { planes } => {
                write!(f, "Duplicate edge for planes {planes:?}")
            }
        }
    }
}

impl std::error::Error for TopologyError {}

// MAIN STRUCTURE - The incremental convex polytope

/// An incrementally constructed convex polytope.
///
/// # Design Decisions
///
/// **Sparse storage**: Arrays use `Option<T>` with free lists for O(1) slot
/// reuse when elements are removed. This avoids invalidating indices.
///
/// **Lazy construction**: Edges and face orderings are built on-demand, not
/// eagerly. This means adding N planes without querying edges costs nothing for
/// edges.
///
/// **Dirty tracking**: Instead of rebuilding everything, we track which
/// vertices/faces changed and only rebuild those (incremental updates).
#[derive(Clone)]
#[expect(clippy::struct_excessive_bools)]
pub struct IncrementalPolytope {
    // Core geometry (sparse arrays with free lists)
    planes: Vec<Option<HalfSpace>>,
    vertices: Vec<Option<Vertex>>,
    edges: Vec<Option<Edge>>,

    // Fast lookups (FxHashMap for non-cryptographic speed)
    edge_map: FxHashMap<(PlaneIdx, PlaneIdx), EdgeIdx>, // (plane1, plane2) → edge

    // State
    is_bounded: bool,
    epsilon: f64,

    // Free lists for O(1) slot reuse
    vertex_free_list: Vec<VertexIdx>,
    edge_free_list: Vec<EdgeIdx>,
    plane_free_list: Vec<PlaneIdx>,

    // Active plane indices for O(N_active) iteration instead of O(N_slots)
    active_planes: Vec<PlaneIdx>,

    // Lazy construction flags
    edges_dirty: bool,
    dirty_vertices: FxHashSet<VertexIdx>, // Which vertices need edge rebuild (O(1) insert/contains)
    faces_dirty: bool,
    dirty_faces: FxHashSet<PlaneIdx>, // Which faces need ordering rebuild (O(1) insert/contains)

    // Cached data
    face_orderings: FxHashMap<PlaneIdx, Vec<VertexIdx>>, // Sorted vertices per face

    // Incremental convex hull for bounded check
    normal_hull: IncrementalHull,
    /// True if hull needs batch rebuild (initial construction or after clear).
    /// When false, plane changes use incremental insert/remove.
    hull_dirty: bool,

    // Reusable spatial hash for vertex deduplication (avoids allocation per operation)
    spatial_hash: SpatialHash,
}

impl IncrementalPolytope {
    // CONSTRUCTION & BASIC QUERIES

    /// Create an empty polytope (represents all of ℝ³ - no constraints yet).
    #[must_use]
    pub fn new() -> Self {
        Self::with_epsilon(EPSILON)
    }

    /// Create with custom numerical tolerance.
    #[must_use]
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            planes: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            edge_map: FxHashMap::default(),
            is_bounded: false,
            epsilon,
            vertex_free_list: Vec::new(),
            edge_free_list: Vec::new(),
            plane_free_list: Vec::new(),
            active_planes: Vec::new(),
            edges_dirty: false,
            dirty_vertices: FxHashSet::default(),
            face_orderings: FxHashMap::default(),
            dirty_faces: FxHashSet::default(),
            faces_dirty: false,
            normal_hull: IncrementalHull::new(epsilon),
            hull_dirty: true,
            spatial_hash: SpatialHash::with_capacity(epsilon * 10.0, 32),
        }
    }

    /// Reset to empty state.
    pub fn clear(&mut self) {
        *self = Self::with_epsilon(self.epsilon);
    }

    // Counts

    /// Returns the number of active planes in the polytope.
    #[inline]
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        self.active_planes.len()
    }

    /// Returns the number of active vertices in the polytope.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.iter().flatten().count()
    }

    /// Returns the number of edges (may trigger lazy edge construction).
    pub fn edge_count(&mut self) -> usize {
        self.ensure_edges_valid();
        self.edges.iter().flatten().count()
    }

    /// Returns `true` if the polytope is bounded (finite volume).
    #[must_use]
    pub fn is_bounded(&mut self) -> bool {
        // Once bounded, always bounded (adding planes can't make it unbounded)
        if self.is_bounded {
            return true;
        }

        // Check lazily
        self.is_bounded = self.check_bounded();
        self.is_bounded
    }

    /// Validates the polytope topology and returns any errors found.
    ///
    /// This performs several consistency checks:
    /// - Euler characteristic: V - E + F = 2 (bounded) or 1 (unbounded)
    /// - Vertex degree: finite vertices need ≥3 planes, ideal need ≥2
    /// - Edge validity: all referenced vertices must exist
    /// - Face size: all faces need ≥3 vertices
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the topology is consistent
    /// - `Err(TopologyError)` describing the first inconsistency found
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut poly = IncrementalPolytope::new();
    /// // ... add planes ...
    /// if let Err(e) = poly.validate_topology() {
    ///     eprintln!("Topology error: {}", e);
    /// }
    /// ```
    pub fn validate_topology(&mut self) -> Result<(), TopologyError> {
        // Ensure edges are built
        self.ensure_edges_valid();

        // Check 1: Vertex degree constraints
        for (v_idx, vertex) in self.vertices() {
            let plane_count = vertex.planes.len();
            if vertex.is_finite() {
                if plane_count < 3 {
                    return Err(TopologyError::UnderconstrainedVertex {
                        vertex: v_idx,
                        plane_count,
                    });
                }
            } else if plane_count < 2 {
                return Err(TopologyError::InvalidIdealVertex {
                    vertex: v_idx,
                    plane_count,
                });
            }
        }

        // Check 2: Edge validity (vertices exist)
        for (e_idx, edge) in self.edges_internal() {
            if self.vertex(edge.vertices.0).is_none() {
                return Err(TopologyError::DanglingEdge {
                    edge: e_idx,
                    missing_vertex: edge.vertices.0,
                });
            }
            if self.vertex(edge.vertices.1).is_none() {
                return Err(TopologyError::DanglingEdge {
                    edge: e_idx,
                    missing_vertex: edge.vertices.1,
                });
            }
        }

        // Check 3: Face sizes (need mesh for this)
        let (_, faces) = self.to_mesh();
        for (face_idx, face) in faces.iter().enumerate() {
            if face.len() < 3 {
                // Find corresponding plane
                let plane_idx = self
                    .active_planes
                    .get(face_idx)
                    .copied()
                    .unwrap_or(PlaneIdx(face_idx));
                return Err(TopologyError::DegenerateFace {
                    plane: plane_idx,
                    vertex_count: face.len(),
                });
            }
        }

        // Check 4: Euler characteristic
        let v = self.vertex_count();
        let e = self.edge_count();
        let f = faces.len();
        let euler = v as i32 - e as i32 + f as i32;

        // Expected: 2 for bounded, 1 for unbounded (open polyhedron)
        let expected = if self.is_bounded() { 2 } else { 1 };

        if euler != expected {
            return Err(TopologyError::EulerMismatch {
                vertices: v,
                edges: e,
                faces: f,
                expected,
                actual: euler,
            });
        }

        Ok(())
    }

    // Element access
    /// Returns a reference to the half-space at the given index.
    #[must_use]
    pub fn plane(&self, idx: PlaneIdx) -> Option<&HalfSpace> {
        self.planes.get(idx.0)?.as_ref()
    }

    /// Returns a reference to the vertex at the given index.
    #[must_use]
    pub fn vertex(&self, idx: VertexIdx) -> Option<&Vertex> {
        self.vertices.get(idx.0)?.as_ref()
    }

    /// Returns a reference to the edge at the given index (may trigger lazy construction).
    pub fn edge(&mut self, idx: EdgeIdx) -> Option<&Edge> {
        self.ensure_edges_valid();
        self.edges.get(idx.0)?.as_ref()
    }

    //  Iterators

    /// Iterates over all active planes with their indices.
    pub fn planes(&self) -> impl Iterator<Item = (PlaneIdx, &HalfSpace)> {
        self.planes
            .iter()
            .enumerate()
            .filter_map(|(i, p)| Some((PlaneIdx(i), p.as_ref()?)))
    }

    /// Iterates over all active vertices with their indices.
    pub fn vertices(&self) -> impl Iterator<Item = (VertexIdx, &Vertex)> {
        self.vertices
            .iter()
            .enumerate()
            .filter_map(|(i, v)| Some((VertexIdx(i), v.as_ref()?)))
    }

    /// Iterates over all edges with their indices (may trigger lazy construction).
    pub fn edges(&mut self) -> impl Iterator<Item = (EdgeIdx, &Edge)> {
        self.ensure_edges_valid();
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, e)| Some((EdgeIdx(i), e.as_ref()?)))
    }

    /// Finds an edge by its two defining planes (may trigger lazy construction).
    pub fn edge_by_planes(&mut self, p1: PlaneIdx, p2: PlaneIdx) -> Option<EdgeIdx> {
        self.ensure_edges_valid();
        self.edge_map.get(&Edge::canonical_planes(p1, p2)).copied()
    }

    // Internal (non-rebuilding)

    fn edges_internal(&self) -> impl Iterator<Item = (EdgeIdx, &Edge)> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, e)| Some((EdgeIdx(i), e.as_ref()?)))
    }

    fn edge_internal(&self, idx: EdgeIdx) -> Option<&Edge> {
        self.edges.get(idx.0)?.as_ref()
    }

    fn edge_by_planes_internal(&self, p1: PlaneIdx, p2: PlaneIdx) -> Option<EdgeIdx> {
        self.edge_map.get(&Edge::canonical_planes(p1, p2)).copied()
    }

    // ADD PLANE - The main incremental construction operation

    /// Add a half-space constraint to the polytope.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. If < 4 planes: Store and return StillUnbounded
    /// 2. If no vertices yet: Try forming initial vertices from 3-plane intersections
    /// 3. Classify all vertices as Inside/On/Outside relative to new plane
    /// 4. If all Outside: Return Empty (infeasible)
    /// 5. If all Inside/On: Check if plane creates new vertices (unbounded case)
    /// 6. Otherwise: Clip polytope (Sutherland-Hodgman style)
    /// ```
    pub fn add_plane(&mut self, half_space: HalfSpace) -> AddPlaneResult {
        let plane_idx = self.alloc_plane(half_space);
        // Note: is_bounded check is now lazy - computed when is_bounded() is called

        // ─── Case 1: No vertices yet → try to form initial geometry ───
        if self.vertex_count() == 0 {
            // With < 3 planes, we can't form meaningful geometry
            // (2 planes = infinite line, but clipping doesn't work well incrementally)
            if self.plane_count() < 3 {
                return AddPlaneResult::StillUnbounded;
            }

            // Try 3-plane case: form corner with 1 finite + 3 ideal vertices
            if self.plane_count() == 3 {
                return match self.try_form_corner_from_three_planes() {
                    Some((new_verts, _)) => AddPlaneResult::Added {
                        new_vertices: new_verts,
                        removed_vertices: vec![],
                        plane_idx,
                    },
                    None => AddPlaneResult::StillUnbounded,
                };
            }

            // 4+ planes: try to form bounded initial polytope
            return match self.try_form_initial_vertices() {
                Some((new_verts, _)) => {
                    self.edges_dirty = true;
                    self.faces_dirty = true;
                    // Bounded check is lazy - done when is_bounded() is called
                    AddPlaneResult::Added {
                        new_vertices: new_verts,
                        removed_vertices: vec![],
                        plane_idx,
                    }
                }
                None => AddPlaneResult::StillUnbounded,
            };
        }

        // Classify vertices relative to new plane
        let classifications = self.classify_vertices(plane_idx);
        let all_outside = classifications
            .iter()
            .all(|(_, c)| *c == Classification::Outside);
        let all_inside_or_on = classifications
            .iter()
            .all(|(_, c)| matches!(c, Classification::Inside | Classification::On));

        // Case 2: All outside → infeasible
        if all_outside {
            self.free_plane(plane_idx);
            self.clear();
            return AddPlaneResult::Empty;
        }

        // Case 3: All inside/on → check for new vertices (unbounded case)
        if all_inside_or_on {
            let new_vertices = self.try_form_vertices_with_plane(plane_idx);
            if new_vertices.is_empty() {
                self.free_plane(plane_idx);
                return AddPlaneResult::Redundant;
            }
            // Bounded check is lazy - done when is_bounded() is called
            self.mark_vertices_and_faces_dirty(&new_vertices);
            return AddPlaneResult::Added {
                new_vertices,
                removed_vertices: vec![],
                plane_idx,
            };
        }

        //  Case 4: Mixed → clip polytope (Sutherland-Hodgman style)
        self.ensure_edges_valid();

        // Remember faces of vertices we're about to remove
        let removed_vertex_faces: Vec<PlaneIdx> = classifications
            .iter()
            .filter(|(_, c)| *c == Classification::Outside)
            .filter_map(|(v_idx, _)| self.vertex(*v_idx))
            .flat_map(|v| v.planes.clone())
            .collect();

        let (new_vertices, removed_vertices) = self.clip_by_plane(plane_idx, &classifications);
        // Bounded check is lazy - done when is_bounded() is called
        // Once bounded, always bounded (adding planes can't make it unbounded)

        // If clipping removed all vertices, mark hull dirty for accurate boundedness check
        if self.vertex_count() == 0 {
            self.hull_dirty = true;
        }

        // Mark dirty: faces for new vertices, and faces that lost vertices
        // Note: We use mark_faces_dirty_for_vertices instead of mark_vertices_and_faces_dirty
        // because clip_by_plane already correctly builds edges. Setting edges_dirty would
        // trigger rebuild_all_edges which clears the correct edges we just built.
        self.mark_faces_dirty_for_vertices(&new_vertices);
        self.mark_face_dirty(plane_idx);
        for p_idx in removed_vertex_faces {
            self.mark_face_dirty(p_idx);
        }

        AddPlaneResult::Added {
            new_vertices,
            removed_vertices,
            plane_idx,
        }
    }

    /// Helper to mark new vertices and their incident faces as dirty.
    fn mark_vertices_and_faces_dirty(&mut self, vertices: &[VertexIdx]) {
        for &v_idx in vertices {
            self.mark_vertex_dirty(v_idx);
        }
        let faces: Vec<PlaneIdx> = vertices
            .iter()
            .filter_map(|&v| self.vertex(v))
            .flat_map(|v| v.planes.clone())
            .collect();
        for p in faces {
            self.mark_face_dirty(p);
        }
    }

    /// Helper to mark only faces dirty for vertices (without marking edges dirty).
    /// Used after `clip_by_plane` where edges are already correctly built.
    fn mark_faces_dirty_for_vertices(&mut self, vertices: &[VertexIdx]) {
        let faces: Vec<PlaneIdx> = vertices
            .iter()
            .filter_map(|&v| self.vertex(v))
            .flat_map(|v| v.planes.clone())
            .collect();
        for p in faces {
            self.mark_face_dirty(p);
        }
    }

    /// Try to form new vertices by intersecting the new plane with pairs of
    /// existing planes. Used when all existing vertices satisfy the new
    /// plane (unbounded polytope case).
    fn try_form_vertices_with_plane(&mut self, plane_idx: PlaneIdx) -> Vec<VertexIdx> {
        let mut new_vertices = Vec::new();
        let mut spatial_hash = SpatialHash::new(self.epsilon * 10.0);

        // Seed spatial hash with existing finite vertices
        for (_, v) in self.vertices() {
            if let Some(pos) = v.to_euclidean() {
                spatial_hash.insert(pos);
            }
        }

        // Get indices of all planes except the new one
        let other_planes: Vec<usize> = (0..self.planes.len())
            .filter(|&i| i != plane_idx.0 && self.planes[i].is_some())
            .collect();

        // Try every pair of existing planes with the new plane
        for (&i, &j) in other_planes.iter().tuple_combinations() {
            let (Some(p1), Some(p2), Some(p3)) =
                (&self.planes[i], &self.planes[j], &self.planes[plane_idx.0])
            else {
                continue;
            };

            // Intersect three planes → potential finite vertex (w=1)
            let Some(point_hom) = intersect_three_planes(p1, p2, p3) else {
                continue;
            };
            // Extract Euclidean position (safe since intersect_three_planes returns w=1)
            let point = dvec4_xyz(&point_hom);

            // Must satisfy all constraints and not be a duplicate
            if !self.point_satisfies_all_planes(point) {
                continue;
            }
            if !spatial_hash.insert_if_unique(point) {
                continue;
            }

            // Collect all incident planes (handle degeneracy: >3 planes through point)
            let incident = self.find_incident_planes(point, &[i, j, plane_idx.0]);
            new_vertices.push(self.alloc_vertex(Vertex {
                position: point_hom,
                planes: incident,
                edges: vec![],
            }));
        }

        if !new_vertices.is_empty() {
            self.edges_dirty = true;
        }
        new_vertices
    }

    /// Find all planes passing through a point (within epsilon).
    /// Uses `active_planes` for `O(N_active)` iteration instead of `O(N_slots)`.
    fn find_incident_planes(&self, point: DVec3, known: &[usize]) -> Vec<PlaneIdx> {
        let mut incident: Vec<usize> = known.to_vec();

        for &plane_idx in &self.active_planes {
            if !known.contains(&plane_idx.0)
                && let Some(plane) = &self.planes[plane_idx.0]
                && plane.signed_distance(point).abs() < self.epsilon
            {
                incident.push(plane_idx.0);
            }
        }

        incident.sort_unstable();
        incident.into_iter().map(PlaneIdx).collect()
    }

    // HELPER METHODS

    /// Classify all vertices relative to a plane.
    ///
    /// For finite vertices: uses standard signed distance classification.
    /// For ideal vertices: classifies based on direction (n·d > 0 means Outside).
    fn classify_vertices(&self, plane_idx: PlaneIdx) -> Vec<(VertexIdx, Classification)> {
        let plane = self.planes.get(plane_idx.0).and_then(|p| p.as_ref());
        plane.map_or(vec![], |p| {
            self.vertices()
                .map(|(idx, v)| {
                    let classification = v.to_euclidean().map_or_else(
                        || {
                            // Ideal vertex: classify by direction
                            let dir = v.direction().unwrap();
                            let dot = p.normal.dot(dir);

                            if dot > self.epsilon {
                                Classification::Outside
                            } else if dot < -self.epsilon {
                                Classification::Inside
                            } else {
                                Classification::On
                            }
                        },
                        |pos| p.classify(pos, self.epsilon),
                    );
                    (idx, classification)
                })
                .collect()
        })
    }

    /// Check if the polytope is bounded by verifying the origin is strictly
    /// inside the convex hull of plane normals.
    ///
    /// **Theoretical basis:** A polytope defined by half-spaces {x : nᵢ·x ≤ dᵢ}
    /// is bounded iff the normals {n₁, ..., nₖ} positively span ℝ³. This is
    /// equivalent to the origin being strictly inside conv({n₁, ..., nₖ}).
    ///
    /// **Algorithm:** Hybrid lazy/incremental approach:
    /// - Initial construction: batch build on first query (O(n log n) amortized)
    /// - Subsequent changes: incremental insert/remove (O(h) per operation)
    /// - Query: O(1) cached result
    fn check_bounded(&mut self) -> bool {
        // Need at least 4 planes and 4 vertices to potentially be bounded
        if self.plane_count() < 4 || self.vertex_count() < 4 {
            return false;
        }

        // Check 1: All vertices must be finite (w != 0)
        // If any vertex is at infinity, the polytope is unbounded
        let all_finite = self.vertices().all(|(_, v)| v.is_finite());
        if !all_finite {
            return false;
        }

        // Check 2: Origin must be inside conv{normals}
        // This catches cases like a cube with one face removed - all vertices
        // are finite but the polytope is unbounded in the direction of the removed face.
        if self.hull_dirty {
            self.rebuild_normal_hull();
        }
        self.normal_hull.origin_inside()
    }

    /// Batch rebuild the normal hull from all current planes.
    /// Used for initial construction; subsequent changes use incremental updates.
    fn rebuild_normal_hull(&mut self) {
        let normals = self
            .planes
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|plane| (i as u32, plane.normal)));

        self.normal_hull = IncrementalHull::build(normals, self.epsilon);
        self.hull_dirty = false;
    }

    // INITIAL VERTEX FORMATION - Finding vertices from plane intersections

    /// Try to form initial vertices from current planes (O(N³) worst case).
    ///
    /// Two-phase approach:
    /// 1. Quick O(N) check for 4 non-coplanar planes (early-out for unbounded)
    /// 2. Exhaustive O(N³) search trying all 3-plane combinations
    fn try_form_initial_vertices(&mut self) -> Option<(Vec<VertexIdx>, Vec<EdgeIdx>)> {
        if self.plane_count() < 4 {
            return None;
        }

        let valid_planes: Vec<usize> = (0..self.planes.len())
            .filter(|&i| self.planes[i].is_some())
            .collect();

        // Quick check (may fail for degenerate cases, so we continue anyway)
        let _ = self.find_initial_simplex(&valid_planes);

        self.try_form_initial_vertices_exhaustive(&valid_planes)
    }

    /// O(N) heuristic: find 4 planes with maximally diverse normals.
    /// Greedy selection minimizes parallelism between chosen planes.
    fn find_initial_simplex(&self, valid_indices: &[usize]) -> Option<[usize; 4]> {
        if valid_indices.len() < 4 {
            return None;
        }

        let normals: Vec<(usize, DVec3)> = valid_indices
            .iter()
            .filter_map(|&i| self.planes[i].as_ref().map(|p| (i, p.normal)))
            .collect();
        if normals.len() < 4 {
            return None;
        }

        // Greedily select planes with maximally different normals
        let mut selected = vec![normals[0]];
        while selected.len() < 4 {
            // Find plane most different from all selected (minimize max |dot|)
            let best = normals
                .iter()
                .filter(|(i, _)| !selected.iter().any(|(j, _)| i == j))
                .max_by(|(_, n1), (_, n2)| {
                    let score1 = 1.0
                        - selected
                            .iter()
                            .map(|(_, s)| n1.dot(*s).abs())
                            .fold(f64::NEG_INFINITY, f64::max);
                    let score2 = 1.0
                        - selected
                            .iter()
                            .map(|(_, s)| n2.dot(*s).abs())
                            .fold(f64::NEG_INFINITY, f64::max);
                    score1.partial_cmp(&score2).unwrap()
                });
            selected.push(*best?);
        }

        // Verify normals are linearly independent (det ≠ 0)
        let det = DMat3::from_cols(selected[0].1, selected[1].1, selected[2].1).determinant();
        (det.abs() >= self.epsilon)
            .then(|| [selected[0].0, selected[1].0, selected[2].0, selected[3].0])
    }

    /// O(N³) exhaustive search: try all 3-plane combinations to find vertices.
    fn try_form_initial_vertices_exhaustive(
        &mut self,
        valid_planes: &[usize],
    ) -> Option<(Vec<VertexIdx>, Vec<EdgeIdx>)> {
        let mut new_vertices = Vec::new();
        let mut seen: HashMap<Vec<usize>, VertexIdx> = HashMap::new();

        for (&i, &j, &k) in valid_planes.iter().tuple_combinations() {
            let (Some(p1), Some(p2), Some(p3)) =
                (&self.planes[i], &self.planes[j], &self.planes[k])
            else {
                continue;
            };

            // Intersect three planes → finite vertex (w=1)
            let Some(point_hom) = intersect_three_planes(p1, p2, p3) else {
                continue;
            };
            let point = dvec4_xyz(&point_hom); // Extract Euclidean position

            if !self.point_satisfies_all_planes(point) {
                continue;
            }

            // Find all incident planes (handles degeneracy)
            let mut incident: Vec<usize> = vec![i, j, k];
            for &m in valid_planes {
                if ![i, j, k].contains(&m)
                    && self.planes[m]
                        .as_ref()
                        .is_some_and(|p| p.signed_distance(point).abs() < self.epsilon)
                {
                    incident.push(m);
                }
            }
            incident.sort_unstable();

            // Skip duplicates
            if seen.contains_key(&incident) {
                continue;
            }

            let planes = incident.iter().map(|&p| PlaneIdx(p)).collect();
            let v_idx = self.alloc_vertex(Vertex {
                position: point_hom,
                planes,
                edges: vec![],
            });
            new_vertices.push(v_idx);
            seen.insert(incident, v_idx);
        }

        if new_vertices.is_empty() {
            return None;
        }
        self.edges_dirty = true;
        Some((new_vertices, vec![]))
    }

    /// Try to form a line from exactly 2 non-parallel planes.
    ///
    /// Creates 2 ideal vertices representing the opposite directions along the
    /// line of intersection, plus 1 edge connecting them (an infinite line).
    ///
    /// # Returns
    /// - `Some((vertices, edges))` if 2 non-parallel planes exist
    /// - `None` if planes are parallel or degenerate
    #[allow(dead_code)]
    fn try_form_line_from_two_planes(&mut self) -> Option<(Vec<VertexIdx>, Vec<EdgeIdx>)> {
        let valid_planes: Vec<usize> = (0..self.planes.len())
            .filter(|&i| self.planes[i].is_some())
            .collect();

        if valid_planes.len() != 2 {
            return None;
        }

        let (p0_idx, p1_idx) = (valid_planes[0], valid_planes[1]);
        let (Some(p0), Some(p1)) = (&self.planes[p0_idx], &self.planes[p1_idx]) else {
            return None;
        };

        // Compute line direction (cross product of normals)
        let dir_hom = intersect_two_planes_direction(p0, p1)?;
        let dir = dvec4_xyz(&dir_hom).normalize();

        // Create two ideal vertices in opposite directions
        let v0_idx = self.alloc_vertex(Vertex {
            position: DVec4::new(dir.x, dir.y, dir.z, 0.0),
            planes: vec![PlaneIdx(p0_idx), PlaneIdx(p1_idx)],
            edges: vec![],
        });
        let v1_idx = self.alloc_vertex(Vertex {
            position: DVec4::new(-dir.x, -dir.y, -dir.z, 0.0),
            planes: vec![PlaneIdx(p0_idx), PlaneIdx(p1_idx)],
            edges: vec![],
        });

        // Create edge connecting the two ideal vertices (infinite line)
        let edge_planes = Edge::canonical_planes(PlaneIdx(p0_idx), PlaneIdx(p1_idx));
        let e_idx = self.alloc_edge(Edge {
            planes: edge_planes,
            vertices: (v0_idx, v1_idx),
        });

        // Link edge to vertices
        if let Some(v0) = self.vertex_mut(v0_idx) {
            v0.edges.push(e_idx);
        }
        if let Some(v1) = self.vertex_mut(v1_idx) {
            v1.edges.push(e_idx);
        }

        self.edges_dirty = true;
        self.faces_dirty = true;
        Some((vec![v0_idx, v1_idx], vec![e_idx]))
    }

    /// Try to form a corner from exactly 3 non-parallel planes.
    ///
    /// Creates 1 finite vertex at the intersection point plus 3 ideal vertices
    /// representing the rays emanating from the corner along each pair of planes.
    ///
    /// # Returns
    /// - `Some((vertices, edges))` if 3 non-parallel planes intersect at a point
    /// - `None` if planes are degenerate (parallel or coincident)
    fn try_form_corner_from_three_planes(&mut self) -> Option<(Vec<VertexIdx>, Vec<EdgeIdx>)> {
        let valid_planes: Vec<usize> = (0..self.planes.len())
            .filter(|&i| self.planes[i].is_some())
            .collect();

        if valid_planes.len() != 3 {
            return None;
        }

        let (p0_idx, p1_idx, p2_idx) = (valid_planes[0], valid_planes[1], valid_planes[2]);

        // Clone planes upfront to avoid borrow conflicts
        let p0 = self.planes[p0_idx].clone()?;
        let p1 = self.planes[p1_idx].clone()?;
        let p2 = self.planes[p2_idx].clone()?;

        // Intersect three planes to get finite vertex
        let corner_hom = intersect_three_planes(&p0, &p1, &p2)?;
        let corner_pos = dvec4_xyz(&corner_hom);

        // Pre-compute all ray directions before mutating self
        let plane_data = [
            (p0_idx, p1_idx, p2_idx, &p0, &p1, &p2),
            (p1_idx, p2_idx, p0_idx, &p1, &p2, &p0),
            (p0_idx, p2_idx, p1_idx, &p0, &p2, &p1),
        ];

        let epsilon = self.epsilon;
        let mut ray_directions: Vec<(usize, usize, DVec3)> = Vec::new();

        for (pi_idx, pj_idx, _, pi, pj, pk) in plane_data {
            // Compute ray direction (line of intersection)
            let Some(dir_hom) = intersect_two_planes_direction(pi, pj) else {
                continue;
            };
            let dir = dvec4_xyz(&dir_hom).normalize();

            // Test which direction stays inside the third plane
            let test_point_pos = corner_pos + dir * epsilon * 100.0;
            let test_point_neg = corner_pos - dir * epsilon * 100.0;

            let dist_pos = pk.signed_distance(test_point_pos);
            let dist_neg = pk.signed_distance(test_point_neg);

            // Choose direction that stays inside (more negative = more inside)
            let outward_dir = if dist_pos < dist_neg { dir } else { -dir };
            ray_directions.push((pi_idx, pj_idx, outward_dir));
        }

        // Now create vertices and edges (mutating self)
        let corner_idx = self.alloc_vertex(Vertex {
            position: corner_hom,
            planes: vec![PlaneIdx(p0_idx), PlaneIdx(p1_idx), PlaneIdx(p2_idx)],
            edges: vec![],
        });

        let mut new_vertices = vec![corner_idx];
        let mut new_edges = Vec::new();

        for (pi_idx, pj_idx, outward_dir) in ray_directions {
            // Create ideal vertex
            let ideal_idx = self.alloc_vertex(Vertex {
                position: DVec4::new(outward_dir.x, outward_dir.y, outward_dir.z, 0.0),
                planes: vec![PlaneIdx(pi_idx), PlaneIdx(pj_idx)],
                edges: vec![],
            });
            new_vertices.push(ideal_idx);

            // Create edge (ray) from corner to ideal vertex
            let edge_planes = Edge::canonical_planes(PlaneIdx(pi_idx), PlaneIdx(pj_idx));
            let e_idx = self.alloc_edge(Edge {
                planes: edge_planes,
                vertices: (corner_idx, ideal_idx),
            });
            new_edges.push(e_idx);

            // Link edge to vertices
            if let Some(v) = self.vertex_mut(corner_idx) {
                v.edges.push(e_idx);
            }
            if let Some(v) = self.vertex_mut(ideal_idx) {
                v.edges.push(e_idx);
            }
        }

        self.edges_dirty = true;
        self.faces_dirty = true;
        Some((new_vertices, new_edges))
    }

    /// Build edges between vertices sharing ≥2 planes (edge = 2-plane
    /// intersection).
    fn build_edges_from_vertices(&mut self, vertices: &[VertexIdx]) -> Vec<EdgeIdx> {
        let mut new_edges = Vec::new();

        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let (v1_idx, v2_idx) = (vertices[i], vertices[j]);
                let (Some(v1), Some(v2)) = (self.vertex(v1_idx), self.vertex(v2_idx)) else {
                    continue;
                };

                // Edge exists if vertices share ≥2 planes
                let shared: Vec<_> = v1
                    .planes
                    .iter()
                    .filter(|p| v2.planes.contains(p))
                    .copied()
                    .collect();
                if shared.len() < 2 {
                    continue;
                }

                let edge_planes = Edge::canonical_planes(shared[0], shared[1]);
                if self
                    .edge_by_planes_internal(edge_planes.0, edge_planes.1)
                    .is_some()
                {
                    continue;
                }

                let e_idx = self.alloc_edge(Edge {
                    planes: edge_planes,
                    vertices: (v1_idx, v2_idx),
                });
                new_edges.push(e_idx);

                // Link edge to vertices
                if let Some(v) = self.vertex_mut(v1_idx) {
                    v.edges.push(e_idx);
                }
                if let Some(v) = self.vertex_mut(v2_idx) {
                    v.edges.push(e_idx);
                }
            }
        }
        new_edges
    }

    // CLIPPING - Sutherland-Hodgman style polytope clipping

    /// Clip polytope by a new plane (Sutherland-Hodgman style).
    ///
    /// # Algorithm
    /// ```text
    /// 1. Vertices ON the plane → add plane to their incident list
    /// 2. For each edge:
    ///    - Both endpoints inside/on → keep edge
    ///    - Both outside → remove edge
    ///    - Mixed → create new vertex at intersection, remove old edge
    /// 3. Remove outside vertices
    /// 4. Connect new vertices to form edges on the cutting plane
    /// ```
    #[expect(clippy::too_many_lines)]
    fn clip_by_plane(
        &mut self,
        plane_idx: PlaneIdx,
        classifications: &[(VertexIdx, Classification)],
    ) -> (Vec<VertexIdx>, Vec<VertexIdx>) {
        // Build HashMap for O(1) classification lookups instead of O(n) linear search
        let class_map: HashMap<VertexIdx, Classification> =
            classifications.iter().copied().collect();
        let class_of = |v: VertexIdx| -> Classification {
            class_map.get(&v).copied().unwrap_or(Classification::Inside)
        };

        // Single pass: partition vertices by classification
        let mut on_vertices = Vec::new();
        let mut inside_or_on_vertices = Vec::new();
        let mut removed_vertices = Vec::new();
        for &(v_idx, class) in classifications {
            match class {
                Classification::On => {
                    on_vertices.push(v_idx);
                    inside_or_on_vertices.push(v_idx);
                }
                Classification::Inside => inside_or_on_vertices.push(v_idx),
                Classification::Outside => removed_vertices.push(v_idx),
            }
        }

        // Step 1: Vertices ON the plane become part of the new face
        for &v_idx in &on_vertices {
            if let Some(v) = self.vertex_mut(v_idx)
                && !v.planes.contains(&plane_idx)
            {
                v.planes.push(plane_idx);
                v.planes.sort();
            }
        }

        // Step 2: Process edges - keep, remove, or clip
        // Store (edge_idx, v1, v2) tuples to avoid re-fetching edge data in removal loop
        let mut new_vertices = Vec::new();
        let mut edges_to_remove: Vec<(EdgeIdx, VertexIdx, VertexIdx)> = Vec::new();
        let mut new_vertex_on_edge: HashMap<EdgeIdx, VertexIdx> = HashMap::new();

        let edge_indices: Vec<EdgeIdx> = self.edges_internal().map(|(i, _)| i).collect();

        for e_idx in edge_indices {
            let Some(edge) = self.edge_internal(e_idx) else {
                continue;
            };
            // Extract edge data without cloning the whole struct
            let (ev0, ev1) = edge.vertices;
            let edge_planes = edge.planes;
            let (c1, c2) = (class_of(ev0), class_of(ev1));

            match (c1, c2) {
                // Both inside/on: keep
                (
                    Classification::Inside | Classification::On,
                    Classification::Inside | Classification::On,
                ) => {}

                // Both outside: remove (store vertices to avoid re-fetch)
                (Classification::Outside, Classification::Outside) => {
                    edges_to_remove.push((e_idx, ev0, ev1));
                }

                // Mixed: clip edge at plane intersection
                _ => {
                    let v0 = self.vertex(ev0).unwrap();
                    let v1 = self.vertex(ev1).unwrap();
                    let plane = self.planes[plane_idx.0].as_ref().unwrap();

                    // Handle edge clipping based on vertex types
                    let new_vertex_pos = if v0.is_finite() && v1.is_finite() {
                        // Segment: standard edge-plane intersection
                        let p1 = v0.to_euclidean().unwrap();
                        let p2 = v1.to_euclidean().unwrap();
                        compute_edge_plane_intersection(p1, p2, plane, self.epsilon)
                            .map(|p| DVec4::new(p.x, p.y, p.z, 1.0))
                    } else if v0.is_finite() {
                        // Ray from v0 in direction of v1: find ray-plane intersection
                        let p = v0.to_euclidean().unwrap();
                        let d = v1.direction().unwrap();
                        let result = compute_ray_plane_intersection(p, d, plane, self.epsilon);
                        result.map(|p| DVec4::new(p.x, p.y, p.z, 1.0))
                    } else if v1.is_finite() {
                        // Ray from v1 in direction of v0 (ideal vertex direction = ray direction)
                        let p = v1.to_euclidean().unwrap();
                        let d = v0.direction().unwrap();
                        let result = compute_ray_plane_intersection(p, d, plane, self.epsilon);
                        result.map(|p| DVec4::new(p.x, p.y, p.z, 1.0))
                    } else {
                        // Line (both ideal): intersection is the 3-plane intersection
                        let p0 = self.planes[edge_planes.0.0].as_ref().unwrap();
                        let p1 = self.planes[edge_planes.1.0].as_ref().unwrap();
                        intersect_three_planes(p0, p1, plane)
                    };

                    if let Some(intersection) = new_vertex_pos {
                        let mut planes = vec![edge_planes.0, edge_planes.1, plane_idx];
                        planes.sort();
                        let new_v = self.alloc_vertex(Vertex {
                            position: intersection,
                            planes,
                            edges: vec![],
                        });
                        new_vertices.push(new_v);
                        new_vertex_on_edge.insert(e_idx, new_v);
                    }
                    // Store vertices to avoid re-fetch in removal loop
                    edges_to_remove.push((e_idx, ev0, ev1));
                }
            }
        }

        // Step 3: Remove clipped/outside edges (no re-fetch needed, vertices stored)
        for (e_idx, v1, v2) in edges_to_remove {
            if let Some(v) = self.vertex_mut(v1) {
                v.edges.retain(|&e| e != e_idx);
            }
            if let Some(v) = self.vertex_mut(v2) {
                v.edges.retain(|&e| e != e_idx);
            }
            self.free_edge(e_idx);
        }

        // Step 4: Connect new vertices to kept vertices sharing ≥2 planes
        for &new_v_idx in new_vertex_on_edge.values() {
            let Some(new_v) = self.vertex(new_v_idx) else {
                continue;
            };
            let new_v_planes = new_v.planes.clone();

            // Only check inside/on vertices (already filtered, avoids re-checking classification)
            for &v_idx in &inside_or_on_vertices {
                if v_idx == new_v_idx {
                    continue;
                }
                let Some(v) = self.vertex(v_idx) else {
                    continue;
                };

                // Count shared planes without allocating - we only need first 2
                let mut shared = [PlaneIdx(0), PlaneIdx(0)];
                let mut shared_count = 0;
                for &p in &new_v_planes {
                    if v.planes.contains(&p) {
                        if shared_count < 2 {
                            shared[shared_count] = p;
                        }
                        shared_count += 1;
                        if shared_count >= 2 {
                            break;
                        }
                    }
                }

                if shared_count >= 2 {
                    let canonical = Edge::canonical_planes(shared[0], shared[1]);
                    if self
                        .edge_by_planes_internal(canonical.0, canonical.1)
                        .is_none()
                    {
                        let e_idx = self.alloc_edge(Edge {
                            planes: canonical,
                            vertices: (v_idx, new_v_idx),
                        });
                        if let Some(v) = self.vertex_mut(v_idx) {
                            v.edges.push(e_idx);
                        }
                        if let Some(v) = self.vertex_mut(new_v_idx) {
                            v.edges.push(e_idx);
                        }
                    }
                }
            }
        }

        // Step 5: Remove outside vertices
        for &v_idx in &removed_vertices {
            self.free_vertex(v_idx);
        }

        // Step 6: Try to create new vertices by intersecting the new plane with
        // all pairs of existing planes. This handles cases where:
        // 1. A face (plane) lost all its vertices and needs new ones
        // 2. New 3-plane intersections become valid after removing outside vertices
        // 3. Planes that weren't connected now have valid intersection with new plane
        let mut additional_vertices = Vec::new();
        {
            let mut spatial_hash = SpatialHash::new(self.epsilon * 10.0);
            for (_, v) in self.vertices() {
                if let Some(pos) = v.to_euclidean() {
                    spatial_hash.insert(pos);
                }
            }

            // Try all pairs of existing planes with the new plane
            let other_planes: Vec<PlaneIdx> = self
                .active_planes
                .iter()
                .filter(|&&p| p != plane_idx)
                .copied()
                .collect();

            for i in 0..other_planes.len() {
                for j in (i + 1)..other_planes.len() {
                    let p1_idx = other_planes[i];
                    let p2_idx = other_planes[j];

                    let (Some(p1), Some(p2), Some(p3)) = (
                        &self.planes[p1_idx.0],
                        &self.planes[p2_idx.0],
                        &self.planes[plane_idx.0],
                    ) else {
                        continue;
                    };

                    let Some(point_hom) = intersect_three_planes(p1, p2, p3) else {
                        continue;
                    };
                    let point = dvec4_xyz(&point_hom);

                    if !self.point_satisfies_all_planes(point) {
                        continue;
                    }
                    if !spatial_hash.insert_if_unique(point) {
                        continue;
                    }

                    // Found a valid new vertex
                    let incident =
                        self.find_incident_planes(point, &[p1_idx.0, p2_idx.0, plane_idx.0]);
                    let new_v = self.alloc_vertex(Vertex {
                        position: point_hom,
                        planes: incident,
                        edges: vec![],
                    });
                    additional_vertices.push(new_v);
                    new_vertices.push(new_v);
                }
            }
        }

        if !additional_vertices.is_empty() {
            self.edges_dirty = true;
        }

        // Build edges on the new face (between new vertices AND on vertices)
        let mut face_vertices = new_vertices.clone();
        face_vertices.extend(&on_vertices);

        if face_vertices.len() >= 2 {
            self.build_face_edges(plane_idx, &face_vertices);
        }

        (new_vertices, removed_vertices)
    }

    /// Build edges between vertices on a new face.
    /// Sorts vertices by angle around centroid to get proper ordering.
    #[expect(
        clippy::cast_precision_loss,
        reason = "vertex count is small enough that f64 mantissa is sufficient"
    )]
    fn build_face_edges(&mut self, plane_idx: PlaneIdx, face_vertices: &[VertexIdx]) {
        if face_vertices.len() < 2 {
            return;
        }

        // Extract plane normal early to avoid borrow conflicts with mutable methods
        let plane_normal = match &self.planes[plane_idx.0] {
            Some(plane) => plane.normal,
            None => return,
        };

        // Separate finite and ideal vertices
        let mut finite_pairs: Vec<(VertexIdx, DVec3)> = Vec::new();
        let mut ideal_pairs: Vec<(VertexIdx, DVec3)> = Vec::new();

        for &v_idx in face_vertices {
            if let Some(v) = self.vertex(v_idx) {
                if let Some(pos) = v.to_euclidean() {
                    finite_pairs.push((v_idx, pos));
                } else if let Some(dir) = v.direction() {
                    ideal_pairs.push((v_idx, dir));
                }
            }
        }

        // Handle different cases based on vertex types
        let total_vertices = finite_pairs.len() + ideal_pairs.len();
        if total_vertices < 2 {
            return;
        }

        // Case 1: Only ideal vertices (line at infinity) - create edges between ideal pairs
        if finite_pairs.is_empty() && ideal_pairs.len() >= 2 {
            self.build_ideal_edges(plane_idx, &ideal_pairs);
            return;
        }

        // Case 2: Mixed finite and ideal vertices - create ray edges
        if !ideal_pairs.is_empty() && !finite_pairs.is_empty() {
            self.build_ray_edges(plane_idx, &finite_pairs, &ideal_pairs);
            // Fall through to also build finite edges if we have enough
        }

        // Case 3: Only finite vertices (or continue from case 2)
        if finite_pairs.len() < 2 {
            return;
        }

        // Compute centroid from finite vertices
        let centroid: DVec3 =
            finite_pairs.iter().map(|(_, pos)| *pos).sum::<DVec3>() / finite_pairs.len() as f64;

        // Create local 2D coordinate system on the plane
        let (u_axis, v_axis) = create_plane_basis(&plane_normal);

        // Project vertices to 2D and compute angles
        let mut vertex_angles: Vec<(VertexIdx, f64)> = finite_pairs
            .iter()
            .map(|&(v_idx, pos)| {
                let local = pos - centroid;
                let u = local.dot(u_axis);
                let v = local.dot(v_axis);
                let angle = v.atan2(u);
                (v_idx, angle)
            })
            .collect();

        // Sort by angle
        vertex_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create edges between adjacent vertices in sorted order
        for i in 0..vertex_angles.len() {
            let v1_idx = vertex_angles[i].0;
            let v2_idx = vertex_angles[(i + 1) % vertex_angles.len()].0;

            if v1_idx == v2_idx {
                continue;
            }

            let v1 = self.vertex(v1_idx).unwrap();
            let v2 = self.vertex(v2_idx).unwrap();

            // Find the other shared plane (not the cutting plane)
            let other_shared: Option<PlaneIdx> = v1
                .planes
                .iter()
                .filter(|&&p| p != plane_idx)
                .find(|&p| v2.planes.contains(p))
                .copied();

            if let Some(other_plane) = other_shared {
                let edge_planes = Edge::canonical_planes(plane_idx, other_plane);

                if self
                    .edge_by_planes_internal(edge_planes.0, edge_planes.1)
                    .is_none()
                {
                    let edge = Edge {
                        planes: edge_planes,
                        vertices: (v1_idx, v2_idx),
                    };

                    let e_idx = self.alloc_edge(edge);

                    if let Some(v) = self.vertex_mut(v1_idx) {
                        v.edges.push(e_idx);
                    }

                    if let Some(v) = self.vertex_mut(v2_idx) {
                        v.edges.push(e_idx);
                    }
                }
            }
        }
    }

    /// Build edges between ideal vertices on a face (lines at infinity).
    ///
    /// For unbounded faces with only ideal vertices, we need to create edges
    /// representing the lines where the face extends to infinity.
    fn build_ideal_edges(&mut self, plane_idx: PlaneIdx, ideal_pairs: &[(VertexIdx, DVec3)]) {
        if ideal_pairs.len() < 2 {
            return;
        }

        // For ideal vertices, we connect them based on their directions
        // Sort by angle around the plane normal to get consistent ordering
        let Some(plane) = &self.planes[plane_idx.0] else {
            return;
        };

        let (u_axis, v_axis) = create_plane_basis(&plane.normal);

        let mut sorted: Vec<(VertexIdx, f64)> = ideal_pairs
            .iter()
            .map(|&(v_idx, dir)| {
                let u = dir.dot(u_axis);
                let v = dir.dot(v_axis);
                (v_idx, v.atan2(u))
            })
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create edges between adjacent ideal vertices
        for i in 0..sorted.len() {
            let v1_idx = sorted[i].0;
            let v2_idx = sorted[(i + 1) % sorted.len()].0;

            if v1_idx == v2_idx {
                continue;
            }

            self.try_create_edge_between(plane_idx, v1_idx, v2_idx);
        }
    }

    /// Build ray edges connecting finite vertices to ideal vertices.
    ///
    /// For unbounded faces, rays extend from finite vertices toward ideal vertices
    /// (points at infinity). Each finite vertex connects to the ideal vertices
    /// that share planes with it.
    fn build_ray_edges(
        &mut self,
        plane_idx: PlaneIdx,
        finite_pairs: &[(VertexIdx, DVec3)],
        ideal_pairs: &[(VertexIdx, DVec3)],
    ) {
        // For each ideal vertex, find which finite vertex it should connect to
        // based on shared planes (other than the face plane)
        for &(ideal_idx, _) in ideal_pairs {
            let ideal_planes: Vec<PlaneIdx> = self
                .vertex(ideal_idx)
                .map(|v| v.planes.clone())
                .unwrap_or_default();

            // Find finite vertices that share a plane with this ideal vertex
            for &(finite_idx, _) in finite_pairs {
                let finite_planes: Vec<PlaneIdx> = self
                    .vertex(finite_idx)
                    .map(|v| v.planes.clone())
                    .unwrap_or_default();

                // Check if they share a plane other than plane_idx
                let shared_plane = ideal_planes
                    .iter()
                    .filter(|&&p| p != plane_idx)
                    .find(|&p| finite_planes.contains(p));

                if shared_plane.is_some() {
                    self.try_create_edge_between(plane_idx, finite_idx, ideal_idx);
                }
            }
        }
    }

    /// Try to create an edge between two vertices on a given plane.
    ///
    /// Finds the other shared plane and creates the edge if it doesn't exist.
    fn try_create_edge_between(
        &mut self,
        plane_idx: PlaneIdx,
        v1_idx: VertexIdx,
        v2_idx: VertexIdx,
    ) {
        let v1_planes: Vec<PlaneIdx> = self
            .vertex(v1_idx)
            .map(|v| v.planes.clone())
            .unwrap_or_default();
        let v2_planes: Vec<PlaneIdx> = self
            .vertex(v2_idx)
            .map(|v| v.planes.clone())
            .unwrap_or_default();

        // Find the other shared plane (not the face plane)
        let other_shared = v1_planes
            .iter()
            .filter(|&&p| p != plane_idx)
            .find(|&p| v2_planes.contains(p));

        if let Some(&other_plane) = other_shared {
            let edge_planes = Edge::canonical_planes(plane_idx, other_plane);

            if self
                .edge_by_planes_internal(edge_planes.0, edge_planes.1)
                .is_none()
            {
                let edge = Edge {
                    planes: edge_planes,
                    vertices: (v1_idx, v2_idx),
                };

                let e_idx = self.alloc_edge(edge);

                if let Some(v) = self.vertex_mut(v1_idx) {
                    v.edges.push(e_idx);
                }

                if let Some(v) = self.vertex_mut(v2_idx) {
                    v.edges.push(e_idx);
                }
            }
        }
    }

    /// Check if a point satisfies all plane constraints.
    fn point_satisfies_all_planes(&self, point: DVec3) -> bool {
        self.planes
            .iter()
            .filter_map(|p| p.as_ref())
            .all(|p| p.normal.dot(point) <= p.offset + self.epsilon)
    }

    /// Check if a point satisfies all plane constraints except the specified ones.
    /// Uses `active_planes` for `O(N_active)` iteration instead of `O(N_slots)`.
    fn point_satisfies_planes_except<const N: usize>(
        &self,
        point: DVec3,
        skip: [PlaneIdx; N],
    ) -> bool {
        for &plane_idx in &self.active_planes {
            if skip.contains(&plane_idx) {
                continue;
            }

            if let Some(plane) = &self.planes[plane_idx.0]
                && plane.normal.dot(point) > plane.offset + self.epsilon
            {
                return false;
            }
        }

        true
    }

    // Output: to_mesh

    /// Convert the polytope to a mesh representation.
    /// Returns (vertices, faces) where faces are lists of vertex indices.
    ///
    /// Note: Only finite vertices are included. For unbounded polytopes with
    /// ideal vertices, use `to_mesh_clipped()` to first clip to a bounding box.
    pub fn to_mesh(&mut self) -> (Vec<DVec3>, Vec<Vec<usize>>) {
        // Ensure face orderings are up to date
        self.ensure_faces_valid();

        // Build vertex index mapping (sparse to dense)
        // Only finite vertices are included in the mesh output.
        let mut vertex_positions = Vec::new();
        let mut idx_map: HashMap<VertexIdx, usize> = HashMap::new();

        for (v_idx, vertex) in self.vertices() {
            if let Some(pos) = vertex.to_euclidean() {
                idx_map.insert(v_idx, vertex_positions.len());
                vertex_positions.push(pos);
            }
        }

        // Build faces from cached orderings
        let mut faces = Vec::new();

        for ordered_vertices in self.face_orderings.values() {
            if ordered_vertices.len() < 3 {
                continue;
            }

            let face: Vec<usize> = ordered_vertices
                .iter()
                .filter_map(|v_idx| idx_map.get(v_idx).copied())
                .collect();

            if face.len() >= 3 {
                faces.push(face);
            }
        }

        (vertex_positions, faces)
    }

    /// Convert an unbounded polytope to a mesh by clipping to a bounding box.
    ///
    /// For bounded polytopes, this is equivalent to `to_mesh()`.
    /// For unbounded polytopes (with ideal vertices), this first clips the
    /// polytope to the specified axis-aligned bounding box, then converts to mesh.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum corner of the bounding box
    /// * `max` - Maximum corner of the bounding box
    pub fn to_mesh_clipped(&mut self, min: DVec3, max: DVec3) -> (Vec<DVec3>, Vec<Vec<usize>>) {
        if self.is_bounded() {
            return self.to_mesh();
        }

        // Clone and clip with 6 bounding box planes
        let mut clipped = self.clone();

        // +X face: n = (1, 0, 0), d = max.x → x ≤ max.x
        clipped.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), max.x));
        // -X face: n = (-1, 0, 0), d = -min.x → -x ≤ -min.x → x ≥ min.x
        clipped.add_plane(HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), -min.x));
        // +Y face
        clipped.add_plane(HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), max.y));
        // -Y face
        clipped.add_plane(HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), -min.y));
        // +Z face
        clipped.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), max.z));
        // -Z face
        clipped.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), -min.z));

        clipped.to_mesh()
    }

    /// Convert to mesh without triggering lazy rebuilds (for testing).
    ///
    /// # Panics
    ///
    /// Panics if vertex angles contain NaN values during face ordering.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "vertex count is small enough that f64 mantissa is sufficient"
    )]
    pub fn to_mesh_no_rebuild(&self) -> (Vec<DVec3>, Vec<Vec<usize>>) {
        // Build vertex index mapping (sparse to dense)
        // Only finite vertices are included in the mesh output.
        let mut vertex_positions = Vec::new();
        let mut idx_map: HashMap<VertexIdx, usize> = HashMap::new();

        for (v_idx, vertex) in self.vertices() {
            if let Some(pos) = vertex.to_euclidean() {
                idx_map.insert(v_idx, vertex_positions.len());
                vertex_positions.push(pos);
            }
        }

        // Build faces from planes
        let mut faces = Vec::new();

        for plane_idx in 0..self.planes.len() {
            let Some(plane) = &self.planes[plane_idx] else {
                continue;
            };
            let plane_idx = PlaneIdx(plane_idx);

            // Find all vertices on this plane.
            // Use O(V_f) edge walking for large polytopes, O(V) filtering for small ones.
            let face_vertices: Vec<VertexIdx> = if self.vertices.len() >= EDGE_WALK_VERTEX_THRESHOLD
            {
                self.find_face_vertices_via_edges(plane_idx)
                    .unwrap_or_else(|| {
                        self.vertices()
                            .filter(|(_, v)| v.planes.contains(&plane_idx))
                            .map(|(idx, _)| idx)
                            .collect()
                    })
            } else {
                // Direct O(V) filtering is faster for small polytopes
                self.vertices()
                    .filter(|(_, v)| v.planes.contains(&plane_idx))
                    .map(|(idx, _)| idx)
                    .collect()
            };

            if face_vertices.len() < 3 {
                continue;
            }

            // Collect only finite vertices with their positions (skip ideal vertices)
            let finite_pairs: Vec<(VertexIdx, DVec3)> = face_vertices
                .iter()
                .filter_map(|&v_idx| {
                    self.vertex(v_idx)
                        .and_then(|v| v.to_euclidean().map(|pos| (v_idx, pos)))
                })
                .collect();

            if finite_pairs.len() < 3 {
                continue;
            }

            let centroid: DVec3 =
                finite_pairs.iter().map(|(_, pos)| *pos).sum::<DVec3>() / finite_pairs.len() as f64;
            let (u_axis, v_axis) = create_plane_basis(&plane.normal);

            let mut vertex_angles: Vec<(VertexIdx, f64)> = finite_pairs
                .iter()
                .map(|&(v_idx, pos)| {
                    let local = pos - centroid;
                    let u = local.dot(u_axis);
                    let v = local.dot(v_axis);
                    (v_idx, v.atan2(u))
                })
                .collect();

            vertex_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Check winding order - normal should point outward
            if vertex_angles.len() >= 3 {
                let p0 = self
                    .vertex(vertex_angles[0].0)
                    .and_then(Vertex::to_euclidean);
                let p1 = self
                    .vertex(vertex_angles[1].0)
                    .and_then(Vertex::to_euclidean);
                let p2 = self
                    .vertex(vertex_angles[2].0)
                    .and_then(Vertex::to_euclidean);

                if let (Some(p0), Some(p1), Some(p2)) = (p0, p1, p2) {
                    let face_normal = (p1 - p0).cross(p2 - p0);
                    if face_normal.dot(plane.normal) < 0.0 {
                        vertex_angles.reverse();
                    }
                }
            }

            let face: Vec<usize> = vertex_angles
                .iter()
                .filter_map(|(v_idx, _)| idx_map.get(v_idx).copied())
                .collect();

            if face.len() >= 3 {
                faces.push(face);
            }
        }

        (vertex_positions, faces)
    }

    // Internal: Storage Management

    /// Allocate a new vertex slot, reusing holes if available.
    fn alloc_vertex(&mut self, vertex: Vertex) -> VertexIdx {
        if let Some(idx) = self.vertex_free_list.pop() {
            self.vertices[idx.0] = Some(vertex);
            idx
        } else {
            let idx = VertexIdx(self.vertices.len());
            self.vertices.push(Some(vertex));
            idx
        }
    }

    /// Remove a vertex, adding its slot to the free list.
    fn free_vertex(&mut self, idx: VertexIdx) {
        if self.vertices[idx.0].is_some() {
            self.vertices[idx.0] = None;
            self.vertex_free_list.push(idx);
        }
    }

    /// Allocate a new edge slot, reusing holes if available.
    fn alloc_edge(&mut self, edge: Edge) -> EdgeIdx {
        let key = Edge::canonical_planes(edge.planes.0, edge.planes.1);
        let idx = if let Some(idx) = self.edge_free_list.pop() {
            self.edges[idx.0] = Some(edge);
            idx
        } else {
            let idx = EdgeIdx(self.edges.len());
            self.edges.push(Some(edge));
            idx
        };
        self.edge_map.insert(key, idx);
        idx
    }

    /// Remove an edge, adding its slot to the free list.
    fn free_edge(&mut self, idx: EdgeIdx) {
        if let Some(edge) = &self.edges[idx.0] {
            let key = Edge::canonical_planes(edge.planes.0, edge.planes.1);
            self.edge_map.remove(&key);
            self.edges[idx.0] = None;
            self.edge_free_list.push(idx);
        }
    }

    /// Get mutable reference to a vertex.
    fn vertex_mut(&mut self, idx: VertexIdx) -> Option<&mut Vertex> {
        self.vertices.get_mut(idx.0).and_then(|v| v.as_mut())
    }

    /// Allocate a new plane slot, reusing holes if available.
    #[expect(clippy::cast_possible_truncation, reason = "plane indices fit in u32")]
    fn alloc_plane(&mut self, plane: HalfSpace) -> PlaneIdx {
        let normal = plane.normal;
        let idx = if let Some(idx) = self.plane_free_list.pop() {
            self.planes[idx.0] = Some(plane);
            idx
        } else {
            let idx = PlaneIdx(self.planes.len());
            self.planes.push(Some(plane));
            idx
        };

        // Track active planes for O(N_active) iteration
        self.active_planes.push(idx);

        // Hybrid approach: incremental insert only if hull already built,
        // otherwise defer to batch build on first is_bounded() call
        if !self.hull_dirty {
            self.normal_hull.insert(idx.0 as u32, normal);
        }

        idx
    }

    /// Remove a plane, adding its slot to the free list.
    #[expect(clippy::cast_possible_truncation, reason = "plane indices fit in u32")]
    fn free_plane(&mut self, idx: PlaneIdx) {
        if self.planes[idx.0].is_some() {
            self.planes[idx.0] = None;
            self.plane_free_list.push(idx);

            // Remove from active_planes (swap_remove is O(1), order doesn't matter)
            if let Some(pos) = self.active_planes.iter().position(|&p| p == idx) {
                self.active_planes.swap_remove(pos);
            }

            // Incremental remove if hull is built, otherwise no-op (batch rebuild will skip it)
            if !self.hull_dirty {
                self.normal_hull.remove(idx.0 as u32);
            }
        }
    }

    // Lazy Edge Construction (Incremental)

    /// Ensure edges are valid, rebuilding them if dirty.
    /// Uses incremental updates when possible (only dirty vertices).
    fn ensure_edges_valid(&mut self) {
        if !self.edges_dirty {
            return;
        }

        // If no edges exist yet, or dirty_vertices is empty, do full rebuild
        // (incremental assumes edges already exist for non-dirty vertices)
        let has_edges = self.edges.iter().any(Option::is_some);
        if self.dirty_vertices.is_empty() || !has_edges {
            // Full rebuild needed (initial construction or after remove_plane)
            self.rebuild_all_edges();
        } else {
            // Incremental update: only rebuild edges for dirty vertices
            self.rebuild_edges_incremental();
        }

        self.edges_dirty = false;
        self.dirty_vertices.clear();
    }

    /// Full edge rebuild - O(V²)
    fn rebuild_all_edges(&mut self) {
        // Clear existing edges
        self.edges.clear();
        self.edge_map.clear();
        self.edge_free_list.clear();

        // Clear edge references from all vertices
        for v in self.vertices.iter_mut().filter_map(|v| v.as_mut()) {
            v.edges.clear();
        }

        // Rebuild all edges from vertex pairs
        let vertex_indices: Vec<VertexIdx> = self.vertices().map(|(idx, _)| idx).collect();
        self.build_edges_from_vertices(&vertex_indices);
    }

    /// Incremental edge rebuild - `O(V + D × K × V_p)` instead of `O(D × V)`
    ///
    /// Only rebuilds edges involving dirty vertices, using a plane-to-vertices
    /// index for efficient candidate lookup.
    ///
    /// - V = total vertices (for building index once)
    /// - D = number of dirty vertices
    /// - K = planes per vertex (typically 3-4)
    /// - `V_p` = vertices per plane (typically 4-6)
    fn rebuild_edges_incremental(&mut self) {
        // HashSet already deduplicates, take ownership to avoid clone
        let dirty_vertices = std::mem::take(&mut self.dirty_vertices);

        if dirty_vertices.is_empty() {
            return;
        }

        // BUILD PLANE-TO-VERTICES INDEX ONCE - O(V × K)
        // This allows O(1) lookup of vertices on each plane instead of O(V) scan
        let mut plane_to_verts: HashMap<PlaneIdx, Vec<VertexIdx>> = HashMap::new();
        for (v_idx, vertex) in self.vertices() {
            for &p in &vertex.planes {
                plane_to_verts.entry(p).or_default().push(v_idx);
            }
        }

        for &dirty_v in &dirty_vertices {
            let Some(dirty_vertex) = self.vertex(dirty_v) else {
                continue;
            };

            // Clear existing edges for this vertex
            let old_edges: Vec<EdgeIdx> = dirty_vertex.edges.clone();
            for e_idx in old_edges {
                // Only remove if both endpoints are dirty (otherwise keep for non-dirty
                // vertex)
                if let Some(edge) = self.edge_internal(e_idx) {
                    let other_v = if edge.vertices.0 == dirty_v {
                        edge.vertices.1
                    } else {
                        edge.vertices.0
                    };

                    // If other vertex is also dirty, it will handle its own edges
                    // If other vertex is not dirty, we need to update it
                    if !dirty_vertices.contains(&other_v)
                        && let Some(other) = self.vertex_mut(other_v)
                    {
                        other.edges.retain(|&e| e != e_idx);
                    }
                }
                self.free_edge(e_idx);
            }

            if let Some(v) = self.vertex_mut(dirty_v) {
                v.edges.clear();
            }

            // Get planes of dirty vertex
            let dirty_planes: Vec<PlaneIdx> = match self.vertex(dirty_v) {
                Some(v) => v.planes.clone(),
                None => continue,
            };

            // GATHER CANDIDATES FROM INDEX - O(K × V_p) instead of O(K × V)
            // Use the pre-built index for O(1) plane lookups
            let mut candidates: HashSet<VertexIdx> = HashSet::new();
            for &plane_idx in &dirty_planes {
                if let Some(verts) = plane_to_verts.get(&plane_idx) {
                    candidates.extend(verts.iter().filter(|&&v| v != dirty_v).copied());
                }
            }

            // Now check only candidates for edge connectivity
            for other_v in candidates {
                let Some(other_vertex) = self.vertex(other_v) else {
                    continue;
                };

                // Find shared planes
                let shared: Vec<PlaneIdx> = dirty_planes
                    .iter()
                    .filter(|p| other_vertex.planes.contains(p))
                    .copied()
                    .collect();

                // Edge exists if they share >= 2 planes
                if shared.len() >= 2 {
                    let edge_planes = Edge::canonical_planes(shared[0], shared[1]);

                    // Check if edge already exists
                    if let Some(existing_e_idx) =
                        self.edge_by_planes_internal(edge_planes.0, edge_planes.1)
                    {
                        // Edge exists - just ensure dirty_v has it in its edge list
                        // (it was cleared earlier, and the edge might have been created
                        // when processing another dirty vertex)
                        if let Some(v) = self.vertex_mut(dirty_v)
                            && !v.edges.contains(&existing_e_idx)
                        {
                            v.edges.push(existing_e_idx);
                        }
                        continue;
                    }

                    let edge = Edge {
                        planes: edge_planes,
                        vertices: (dirty_v, other_v),
                    };

                    let e_idx = self.alloc_edge(edge);

                    if let Some(v) = self.vertex_mut(dirty_v) {
                        v.edges.push(e_idx);
                    }
                    if let Some(v) = self.vertex_mut(other_v) {
                        v.edges.push(e_idx);
                    }
                }
            }
        }
    }

    /// Mark a vertex as needing edge rebuild.
    fn mark_vertex_dirty(&mut self, v_idx: VertexIdx) {
        self.dirty_vertices.insert(v_idx); // O(1) insert with HashSet
        self.edges_dirty = true;
    }

    // Incremental Face Ordering

    /// Ensure face orderings are valid, rebuilding them if dirty.
    /// Uses incremental updates when possible (only dirty faces).
    fn ensure_faces_valid(&mut self) {
        if !self.faces_dirty {
            return;
        }

        // If no face orderings exist yet, or dirty_faces is empty, do full rebuild
        // (incremental assumes face orderings already exist for non-dirty faces)
        if self.dirty_faces.is_empty() || self.face_orderings.is_empty() {
            // Full rebuild needed
            self.rebuild_all_face_orderings();
        } else {
            // Incremental update: only rebuild dirty faces
            self.rebuild_faces_incremental();
        }

        self.faces_dirty = false;
        self.dirty_faces.clear();
    }

    /// Full face ordering rebuild - `O(F × V_f × log V_f)`
    fn rebuild_all_face_orderings(&mut self) {
        self.face_orderings.clear();

        // Build face orderings for each plane
        for plane_idx in 0..self.planes.len() {
            let plane_idx = PlaneIdx(plane_idx);
            self.rebuild_face_ordering(plane_idx);
        }
    }

    /// Incremental face rebuild - only rebuilds dirty faces
    fn rebuild_faces_incremental(&mut self) {
        // HashSet already deduplicates, take ownership to avoid clone
        let dirty_faces = std::mem::take(&mut self.dirty_faces);

        for plane_idx in dirty_faces {
            self.face_orderings.remove(&plane_idx);
            self.rebuild_face_ordering(plane_idx);
        }
    }

    /// Find all vertices on a face by walking edges around the face boundary.
    ///
    /// This is `O(V_f)` where `V_f` is the number of vertices on the face,
    /// compared to `O(V)` for filtering all vertices.
    ///
    /// # Algorithm
    /// 1. Find any edge on the face (has `plane_idx` as one of its defining planes)
    /// 2. Start at one vertex of that edge
    /// 3. Walk around the face by following edges that lie on the plane
    /// 4. Stop when we return to the starting vertex
    ///
    /// Returns `None` if no edges exist on this face (degenerate case).
    fn find_face_vertices_via_edges(&self, plane_idx: PlaneIdx) -> Option<Vec<VertexIdx>> {
        // Find any edge on this face
        let (start_edge_idx, start_edge) = self
            .edges_internal()
            .find(|(_, e)| e.is_on_plane(plane_idx))?;

        let start_v = start_edge.vertices.0;
        let mut current_v = start_v;
        let mut prev_edge_idx = start_edge_idx;
        let mut face_verts = Vec::new();

        loop {
            face_verts.push(current_v);

            // Find the next edge on this face incident to current_v (different from prev_edge)
            let current_vertex = self.vertex(current_v)?;
            let next_edge_idx = current_vertex.edges.iter().copied().find(|&e_idx| {
                e_idx != prev_edge_idx
                    && self
                        .edge_internal(e_idx)
                        .is_some_and(|e| e.is_on_plane(plane_idx))
            })?;

            let next_edge = self.edge_internal(next_edge_idx)?;
            let next_v = next_edge.other_vertex(current_v)?;

            // Check if we've completed the loop
            if next_v == start_v {
                break;
            }

            current_v = next_v;
            prev_edge_idx = next_edge_idx;
        }

        Some(face_verts)
    }

    /// Rebuild face ordering for a single plane
    #[expect(
        clippy::cast_precision_loss,
        reason = "vertex count is small enough that f64 mantissa is sufficient"
    )]
    fn rebuild_face_ordering(&mut self, plane_idx: PlaneIdx) {
        let Some(plane) = &self.planes[plane_idx.0] else {
            return;
        };

        // Find all vertices on this plane.
        // Use O(V_f) edge walking for large polytopes, O(V) filtering for small ones.
        let face_vertices: Vec<VertexIdx> = if self.vertices.len() >= EDGE_WALK_VERTEX_THRESHOLD {
            self.find_face_vertices_via_edges(plane_idx)
                .unwrap_or_else(|| {
                    self.vertices()
                        .filter(|(_, v)| v.planes.contains(&plane_idx))
                        .map(|(idx, _)| idx)
                        .collect()
                })
        } else {
            // Direct O(V) filtering is faster for small polytopes
            self.vertices()
                .filter(|(_, v)| v.planes.contains(&plane_idx))
                .map(|(idx, _)| idx)
                .collect()
        };

        if face_vertices.len() < 3 {
            return;
        }

        // Collect only finite vertices with their positions (skip ideal vertices)
        // This fixes the zip mismatch bug where positions was filtered but face_vertices wasn't
        let finite_pairs: Vec<(VertexIdx, DVec3)> = face_vertices
            .iter()
            .filter_map(|&v_idx| {
                self.vertex(v_idx)
                    .and_then(|v| v.to_euclidean().map(|pos| (v_idx, pos)))
            })
            .collect();

        // Need at least 3 finite vertices for a valid face ordering
        if finite_pairs.len() < 3 {
            return;
        }

        let centroid: DVec3 =
            finite_pairs.iter().map(|(_, pos)| *pos).sum::<DVec3>() / finite_pairs.len() as f64;
        let (u_axis, v_axis) = create_plane_basis(&plane.normal);

        let mut vertex_angles: Vec<(VertexIdx, f64)> = finite_pairs
            .iter()
            .map(|&(v_idx, pos)| {
                let local = pos - centroid;
                let u = local.dot(u_axis);
                let v = local.dot(v_axis);
                (v_idx, v.atan2(u))
            })
            .collect();

        vertex_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check winding order - normal should point outward
        if vertex_angles.len() >= 3 {
            let p0 = self
                .vertex(vertex_angles[0].0)
                .and_then(Vertex::to_euclidean);
            let p1 = self
                .vertex(vertex_angles[1].0)
                .and_then(Vertex::to_euclidean);
            let p2 = self
                .vertex(vertex_angles[2].0)
                .and_then(Vertex::to_euclidean);

            if let (Some(p0), Some(p1), Some(p2)) = (p0, p1, p2) {
                let face_normal = (p1 - p0).cross(p2 - p0);
                if face_normal.dot(plane.normal) < 0.0 {
                    vertex_angles.reverse();
                }
            }
        }

        let ordered: Vec<VertexIdx> = vertex_angles.into_iter().map(|(v, _)| v).collect();
        self.face_orderings.insert(plane_idx, ordered);
    }

    /// Mark a face (plane) as needing ordering rebuild.
    fn mark_face_dirty(&mut self, plane_idx: PlaneIdx) {
        self.dirty_faces.insert(plane_idx); // O(1) insert with HashSet
        self.faces_dirty = true;
    }

    // Public API: Remove Plane

    /// Remove a plane from the polytope.
    ///
    /// This operation removes the specified plane and reconstructs the affected
    /// geometry. When a plane is removed, vertices that were created by its
    /// intersection with other planes are removed, and new vertices are
    /// computed by finding intersections of the remaining adjacent planes.
    ///
    /// # Arguments
    /// * `plane_idx` - Index of the plane to remove
    ///
    /// # Returns
    /// * `Some(RemovePlaneResult)` - Information about removed/affected
    ///   vertices
    /// * `None` - If the plane index is invalid
    ///
    /// # Panics
    ///
    /// Panics if internal vertex data is inconsistent (vertex exists in storage
    /// but cannot be retrieved).
    pub fn remove_plane(&mut self, plane_idx: PlaneIdx) -> Option<RemovePlaneResult> {
        // Validate plane exists
        self.planes.get(plane_idx.0).and_then(|p| p.as_ref())?;

        // Ensure edges are valid before removal (we need consistent edge data)
        self.ensure_edges_valid();

        // Find vertices that depend on this plane and collect adjacent planes
        // Single pass: gather affected vertices and adjacent planes together
        let mut adjacent_planes_set: std::collections::HashSet<PlaneIdx> =
            std::collections::HashSet::new();
        let mut removed_vertex_data: Vec<(VertexIdx, DVec4, Vec<PlaneIdx>)> = Vec::new();
        let mut kept_vertex_indices: Vec<VertexIdx> = Vec::new();

        for (v_idx, vertex) in self.vertices() {
            if !vertex.planes.contains(&plane_idx) {
                continue;
            }

            // Collect adjacent planes (O(1) insert via HashSet)
            for &p in &vertex.planes {
                if p != plane_idx {
                    adjacent_planes_set.insert(p);
                }
            }

            // Check remaining plane count (planes.len() - 1 since we're removing one)
            let remaining_count = vertex.planes.len() - 1;
            if remaining_count < 3 {
                // Vertex will be removed - capture data for reconstruction
                let remaining_planes: Vec<PlaneIdx> = vertex
                    .planes
                    .iter()
                    .filter(|&&p| p != plane_idx)
                    .copied()
                    .collect();
                removed_vertex_data.push((v_idx, vertex.position, remaining_planes));
            } else {
                // Vertex will be kept - just record index, update in-place later
                kept_vertex_indices.push(v_idx);
            }
        }

        let adjacent_planes: Vec<PlaneIdx> = adjacent_planes_set.into_iter().collect();

        // Extract removed vertex indices
        let removed_vertices: Vec<VertexIdx> =
            removed_vertex_data.iter().map(|(idx, _, _)| *idx).collect();

        // Update kept vertices in-place using retain (avoids intermediate Vec)
        for &v_idx in &kept_vertex_indices {
            if let Some(v) = self.vertex_mut(v_idx) {
                v.planes.retain(|&p| p != plane_idx);
            }
        }

        // Collect ALL edges to remove in one pass:
        // 1. Edges defined by the removed plane
        // 2. Edges connected to removed vertices
        let removed_vertex_set: std::collections::HashSet<VertexIdx> =
            removed_vertices.iter().copied().collect();

        let edges_to_remove: Vec<(EdgeIdx, VertexIdx, VertexIdx)> = self
            .edges_internal()
            .filter(|(_, e)| {
                e.planes.0 == plane_idx
                    || e.planes.1 == plane_idx
                    || removed_vertex_set.contains(&e.vertices.0)
                    || removed_vertex_set.contains(&e.vertices.1)
            })
            .map(|(idx, e)| (idx, e.vertices.0, e.vertices.1))
            .collect();

        // Remove edges and update vertex edge lists
        for (e_idx, v1_idx, v2_idx) in edges_to_remove {
            // Only update edge lists for vertices that aren't being removed
            if !removed_vertex_set.contains(&v1_idx)
                && let Some(v) = self.vertex_mut(v1_idx)
            {
                v.edges.retain(|&e| e != e_idx);
            }

            if !removed_vertex_set.contains(&v2_idx)
                && let Some(v) = self.vertex_mut(v2_idx)
            {
                v.edges.retain(|&e| e != e_idx);
            }

            self.free_edge(e_idx);
        }

        // Remove the vertices (edges already handled above)
        for &v_idx in &removed_vertices {
            self.free_vertex(v_idx);
        }

        // Remove the plane itself
        self.free_plane(plane_idx);

        // If all vertices were removed, mark hull dirty for fresh rebuild
        // This ensures check_bounded() gets accurate results
        if self.vertex_count() == 0 {
            self.hull_dirty = true;
        }

        // Reconstruct missing vertices (topology-based)
        let reconstructed_vertices =
            self.reconstruct_vertices_topology_based(&removed_vertex_data, &adjacent_planes);

        // Mark affected faces dirty
        for &p in &adjacent_planes {
            self.mark_face_dirty(p);
        }

        // Mark new vertices dirty for edge building
        for &v_idx in &reconstructed_vertices {
            self.mark_vertex_dirty(v_idx);
        }

        // Mark edges dirty since topology changed
        self.edges_dirty = true;
        self.faces_dirty = true;

        // LAZY bounded check: invalidate cache, don't recompute eagerly.
        // Removing planes CAN make a bounded polytope unbounded (e.g., removing
        // one face of a cube creates an infinite prism). We mark the hull dirty
        // and let the next is_bounded() call trigger recomputation.
        // This ensures zero overhead when boundedness is not queried.
        self.is_bounded = false;
        self.hull_dirty = true;

        // Remove face ordering for this plane
        self.face_orderings.remove(&plane_idx);

        Some(RemovePlaneResult {
            removed_vertices,
            affected_vertices: kept_vertex_indices,
        })
    }

    /// Reconstruct vertices after plane removal by finding intersections
    /// of adjacent planes that satisfy all remaining constraints.
    ///
    /// This "closes the hole" left by removing a plane by finding vertices
    /// that would have existed without that plane.
    ///
    /// **Note**: This is the O(A³) brute-force fallback kept for benchmark
    /// comparison. Prefer `reconstruct_vertices_topology_based` for better
    /// performance. Only available in test/benchmark builds.
    pub fn reconstruct_vertices_after_removal(
        &mut self,
        adjacent_planes: &[PlaneIdx],
        _removed_vertices: &[VertexIdx],
    ) -> Vec<VertexIdx> {
        let mut new_vertices = Vec::new();

        // Use spatial hash for O(1) duplicate detection
        let mut spatial_hash = SpatialHash::new(self.epsilon * 10.0);

        // Insert existing finite vertex positions
        for (_, v) in self.vertices() {
            if let Some(pos) = v.to_euclidean() {
                spatial_hash.insert(pos);
            }
        }

        // Try all combinations of 3 adjacent planes to find new vertices
        // These are planes that bordered the removed face
        for i in 0..adjacent_planes.len() {
            for j in (i + 1)..adjacent_planes.len() {
                for k in (j + 1)..adjacent_planes.len() {
                    let p1_idx = adjacent_planes[i];
                    let p2_idx = adjacent_planes[j];
                    let p3_idx = adjacent_planes[k];

                    let Some(p1) = &self.planes[p1_idx.0] else {
                        continue;
                    };
                    let Some(p2) = &self.planes[p2_idx.0] else {
                        continue;
                    };
                    let Some(p3) = &self.planes[p3_idx.0] else {
                        continue;
                    };

                    if let Some(point_hom) = intersect_three_planes(p1, p2, p3) {
                        let point = dvec4_xyz(&point_hom); // Extract Euclidean position

                        // Check if point satisfies all remaining constraints
                        if !self.point_satisfies_all_planes(point) {
                            continue;
                        }

                        // Check if vertex already exists using spatial hash (O(1)
                        // expected)
                        if !spatial_hash.insert_if_unique(point) {
                            continue;
                        }

                        // Find all planes passing through this point (degeneracy
                        // handling)
                        let mut incident_planes = vec![p1_idx.0, p2_idx.0, p3_idx.0];

                        for plane_idx in 0..self.planes.len() {
                            if plane_idx != p1_idx.0
                                && plane_idx != p2_idx.0
                                && plane_idx != p3_idx.0
                                && let Some(plane) = &self.planes[plane_idx]
                            {
                                let dist = plane.signed_distance(point).abs();

                                if dist < self.epsilon {
                                    incident_planes.push(plane_idx);
                                }
                            }
                        }

                        incident_planes.sort_unstable();

                        let plane_indices: Vec<PlaneIdx> =
                            incident_planes.iter().map(|&p| PlaneIdx(p)).collect();

                        let vertex = Vertex {
                            position: point_hom,
                            planes: plane_indices,
                            edges: vec![],
                        };

                        let v_idx = self.alloc_vertex(vertex);
                        new_vertices.push(v_idx);
                    }
                }
            }
        }

        new_vertices
    }

    /// Topology-based vertex reconstruction after plane removal.
    ///
    /// Instead of testing all C(A,3) combinations of adjacent planes (O(A³ ×
    /// N)), this uses the topology of removed vertices to find new vertices
    /// in `O(V_r × K² × N)` where `V_r` = removed vertices, K = planes per
    /// vertex (typically 3-4).
    ///
    /// ## Algorithm
    ///
    /// 1. For each removed vertex with defining planes [Q, R, ...] (excluding
    ///    removed plane P)
    /// 2. For each pair of planes (Q, R), they define a line that was blocked
    ///    by P
    /// 3. From the vertex position, search along Q∩R to find where it hits the
    ///    next plane
    /// 4. The new vertex is at intersection of Q, R, and the hit plane
    ///
    /// This is essentially the inverse of clipping: unclipping edges that were
    /// cut.
    #[expect(clippy::too_many_lines)]
    fn reconstruct_vertices_topology_based(
        &mut self,
        removed_vertex_data: &[(VertexIdx, DVec4, Vec<PlaneIdx>)],
        adjacent_planes: &[PlaneIdx],
    ) -> Vec<VertexIdx> {
        // Pre-allocate with estimated capacity to avoid reallocations
        let mut new_vertices = Vec::with_capacity(ESTIMATED_NEW_VERTICES);

        // Reuse spatial hash for O(1) duplicate detection of finite vertices
        // Reset and reuse to avoid allocation overhead
        self.spatial_hash.reset(self.epsilon * 10.0);

        // Track ideal vertices by their defining plane pair and direction sign
        // Key: (min_plane, max_plane, positive_direction: bool)
        // Use FxHashSet for faster hashing
        let mut ideal_vertices_seen: FxHashSet<(PlaneIdx, PlaneIdx, bool)> = FxHashSet::default();

        // Collect existing vertex data first to avoid borrow conflicts
        // (We need to iterate vertices immutably, then mutate spatial_hash)
        let existing_finite_positions: Vec<DVec3> = self
            .vertices()
            .filter_map(|(_, v)| v.to_euclidean())
            .collect();

        let existing_ideal_keys: Vec<(PlaneIdx, PlaneIdx, bool)> = self
            .vertices()
            .filter_map(|(_, v)| {
                if !v.is_finite() && v.planes.len() >= 2 {
                    let p1 = v.planes[0];
                    let p2 = v.planes[1];
                    let (min_p, max_p) = if p1.0 < p2.0 { (p1, p2) } else { (p2, p1) };
                    let dir = dvec4_xyz(&v.position);
                    let positive = dir.x > 0.0
                        || (dir.x == 0.0 && (dir.y > 0.0 || (dir.y == 0.0 && dir.z > 0.0)));
                    Some((min_p, max_p, positive))
                } else {
                    None
                }
            })
            .collect();

        // Insert existing vertex positions into spatial hash
        for pos in existing_finite_positions {
            self.spatial_hash.insert(pos);
        }

        // Insert existing ideal vertices
        for key in existing_ideal_keys {
            ideal_vertices_seen.insert(key);
        }

        // For each removed vertex, extend its edges along the remaining plane pairs
        for (_v_idx, position_hom, remaining_planes) in removed_vertex_data {
            // Need at least 2 planes to form an edge direction
            if remaining_planes.len() < 2 {
                continue;
            }

            // Extract Euclidean position (skip ideal vertices for reconstruction)
            let Some(position) = (if position_hom.w.abs() > EPSILON {
                Some(dvec4_xyz(position_hom) / position_hom.w)
            } else {
                None
            }) else {
                continue;
            };

            // For each pair of remaining planes, find where the line extends to
            for (i, &p1_idx) in remaining_planes.iter().enumerate() {
                let Some(p1) = self.planes[p1_idx.0].clone() else {
                    continue;
                };

                for &p2_idx in remaining_planes.iter().skip(i + 1) {
                    let Some(p2) = self.planes[p2_idx.0].clone() else {
                        continue;
                    };

                    // Line direction is the cross product of the two plane normals
                    let line_dir = p1.normal.cross(p2.normal);
                    if line_dir.length_squared() < self.epsilon * self.epsilon {
                        // Planes are parallel, no valid line
                        continue;
                    }
                    let line_dir = line_dir.normalize();

                    // Search in both directions along the line
                    for sign in [-1.0, 1.0] {
                        let dir = sign * line_dir;

                        // Find the closest constraining plane in this direction
                        if let Some((hit_plane_idx, new_point)) =
                            self.line_search_next_plane(position, dir, p1_idx, p2_idx)
                        {
                            // Check if vertex already exists
                            if !self.spatial_hash.insert_if_unique(new_point) {
                                continue;
                            }

                            // Collect all incident planes (for degeneracy handling)
                            // Pre-allocate with capacity 4 (typical case: 3 planes + maybe 1 more)
                            let mut incident_planes = Vec::with_capacity(4);
                            incident_planes.extend([p1_idx.0, p2_idx.0, hit_plane_idx.0]);

                            // Check other planes for incidence
                            for (idx, plane_opt) in self.planes.iter().enumerate() {
                                if idx == p1_idx.0 || idx == p2_idx.0 || idx == hit_plane_idx.0 {
                                    continue;
                                }

                                if let Some(plane) = plane_opt
                                    && plane.signed_distance(new_point).abs() < self.epsilon
                                {
                                    incident_planes.push(idx);
                                }
                            }

                            incident_planes.sort_unstable();
                            incident_planes.dedup();

                            let plane_indices: Vec<PlaneIdx> =
                                incident_planes.iter().map(|&p| PlaneIdx(p)).collect();

                            // Create finite vertex (w=1)
                            let vertex = Vertex {
                                position: DVec4::new(new_point.x, new_point.y, new_point.z, 1.0),
                                planes: plane_indices,
                                edges: vec![],
                            };

                            let v_idx = self.alloc_vertex(vertex);
                            new_vertices.push(v_idx);
                        } else {
                            // No blocking plane found - create ideal vertex (point at infinity)
                            // This represents a ray extending to infinity in direction `dir`

                            // Check for duplicate ideal vertex
                            let (min_p, max_p) = if p1_idx.0 < p2_idx.0 {
                                (p1_idx, p2_idx)
                            } else {
                                (p2_idx, p1_idx)
                            };
                            let positive = dir.x > 0.0
                                || (dir.x == 0.0 && (dir.y > 0.0 || (dir.y == 0.0 && dir.z > 0.0)));

                            if !ideal_vertices_seen.insert((min_p, max_p, positive)) {
                                // Duplicate ideal vertex, skip
                                continue;
                            }

                            let plane_indices = vec![p1_idx, p2_idx];

                            // Create ideal vertex (w=0, direction stored in xyz)
                            let vertex = Vertex {
                                position: DVec4::new(dir.x, dir.y, dir.z, 0.0),
                                planes: plane_indices,
                                edges: vec![],
                            };

                            let v_idx = self.alloc_vertex(vertex);
                            new_vertices.push(v_idx);
                        }
                    }
                }
            }
        }

        // CROSS-VERTEX CHECK: The edge-based reconstruction only finds vertices
        // at intersections where at least 2 planes came from the same removed vertex.
        // But some new vertices may be at intersections of 3 planes where each
        // plane came from a DIFFERENT removed vertex. We check for these here.
        //
        // This is O(A³/6) where A = number of adjacent planes, but A is typically
        // small (6-10) so this is acceptable.
        if adjacent_planes.len() >= 3 {
            for i in 0..adjacent_planes.len() {
                let p1_idx = adjacent_planes[i];
                let Some(p1) = self.planes[p1_idx.0].clone() else {
                    continue;
                };

                for j in (i + 1)..adjacent_planes.len() {
                    let p2_idx = adjacent_planes[j];
                    let Some(p2) = self.planes[p2_idx.0].clone() else {
                        continue;
                    };

                    #[expect(clippy::needless_range_loop)]
                    for k in j + 1..adjacent_planes.len() {
                        let p3_idx = adjacent_planes[k];
                        let Some(p3) = self.planes[p3_idx.0].clone() else {
                            continue;
                        };

                        // Try to find intersection point
                        let Some(point_hom) = intersect_three_planes(&p1, &p2, &p3) else {
                            continue;
                        };

                        // Only handle finite vertices here (ideal handled by edge extension)
                        if point_hom.w.abs() < EPSILON {
                            continue;
                        }

                        let point = dvec4_xyz(&point_hom) / point_hom.w;

                        // Check if point satisfies all remaining constraints
                        if !self.point_satisfies_all_planes(point) {
                            continue;
                        }

                        // Check if vertex already exists
                        if !self.spatial_hash.insert_if_unique(point) {
                            continue;
                        }

                        // Find all incident planes (for degeneracy handling)
                        // Pre-allocate with capacity 4 (typical case: 3 planes + maybe 1 more)
                        let mut incident_planes = Vec::with_capacity(4);
                        incident_planes.extend([p1_idx.0, p2_idx.0, p3_idx.0]);
                        for (idx, plane_opt) in self.planes.iter().enumerate() {
                            if idx == p1_idx.0 || idx == p2_idx.0 || idx == p3_idx.0 {
                                continue;
                            }
                            if let Some(plane) = plane_opt
                                && plane.signed_distance(point).abs() < self.epsilon
                            {
                                incident_planes.push(idx);
                            }
                        }

                        incident_planes.sort_unstable();
                        incident_planes.dedup();

                        let plane_indices: Vec<PlaneIdx> =
                            incident_planes.iter().map(|&p| PlaneIdx(p)).collect();

                        let vertex = Vertex {
                            position: DVec4::new(point.x, point.y, point.z, 1.0),
                            planes: plane_indices,
                            edges: vec![],
                        };

                        let v_idx = self.alloc_vertex(vertex);
                        new_vertices.push(v_idx);
                    }
                }
            }
        }

        new_vertices
    }

    /// Search along a line from a starting point to find the next constraining
    /// plane.
    ///
    /// Returns the plane index and intersection point, or None if no plane
    /// constrains the line in this direction (unbounded).
    ///
    /// The search ignores the two planes that define the line direction.
    /// Uses `active_planes` for `O(N_active)` iteration instead of `O(N_slots)`.
    fn line_search_next_plane(
        &self,
        start: DVec3,
        dir: DVec3,
        ignore1: PlaneIdx,
        ignore2: PlaneIdx,
    ) -> Option<(PlaneIdx, DVec3)> {
        let mut best_t = f64::MAX;
        let mut best_plane = None;

        for &plane_idx in &self.active_planes {
            // Skip the planes that define the line
            if plane_idx == ignore1 || plane_idx == ignore2 {
                continue;
            }

            if let Some(plane) = &self.planes[plane_idx.0] {
                // Compute intersection: n·(start + t*dir) = d
                // t = (d - n·start) / (n·dir)
                let n_dot_dir = plane.normal.dot(dir);

                // Skip if line is parallel to plane (or nearly so)
                if n_dot_dir.abs() < self.epsilon {
                    continue;
                }

                let t = (plane.offset - plane.normal.dot(start)) / n_dot_dir;

                // Only consider positive t (forward along direction)
                // and t > epsilon (not at current position)
                if t > self.epsilon && t < best_t {
                    let candidate_point = start + t * dir;

                    // Verify the point satisfies all constraints
                    // (it should, but numerical issues might cause problems)
                    let skip_planes = [ignore1, ignore2, plane_idx];
                    if self.point_satisfies_planes_except(candidate_point, skip_planes) {
                        best_t = t;
                        best_plane = Some((plane_idx, candidate_point));
                    }
                }
            }
        }

        best_plane
    }
}

impl Default for IncrementalPolytope {
    fn default() -> Self {
        Self::new()
    }
}

// Geometric Utilities

/// Intersect three planes to find their common vertex.
///
/// Returns homogeneous coordinates with `w = 1` (finite vertex).
/// Returns `None` if planes are parallel or nearly so.
#[must_use]
pub fn intersect_three_planes(p1: &HalfSpace, p2: &HalfSpace, p3: &HalfSpace) -> Option<DVec4> {
    let constraints = DVec3::new(p1.offset, p2.offset, p3.offset);
    let coeffs = DMat3::from_cols(p1.normal, p2.normal, p3.normal).transpose();

    // Check if the matrix is invertible (non-zero determinant)
    let det = coeffs.determinant();
    if det.abs() < EPSILON {
        return None; // Planes are parallel or nearly so
    }

    let inv = coeffs.inverse();
    let pos = inv * constraints;
    Some(DVec4::new(pos.x, pos.y, pos.z, 1.0))
}

/// Compute the direction of intersection of two planes.
///
/// Returns homogeneous coordinates with `w = 0` (ideal vertex / direction).
/// Returns `None` if planes are parallel.
fn intersect_two_planes_direction(p0: &HalfSpace, p1: &HalfSpace) -> Option<DVec4> {
    let dir = p0.normal.cross(p1.normal);

    if dir.length_squared() < EPSILON * EPSILON {
        return None; // Parallel planes
    }

    let dir = dir.normalize();
    Some(DVec4::new(dir.x, dir.y, dir.z, 0.0))
}

// Check if a point satisfies all half-space constraints.
// #[must_use]
// pub fn point_satisfies_all(point: DVec3, planes: &[HalfSpace], epsilon: f64) -> bool {
//     planes
//         .iter()
//         .all(|p| p.normal.dot(point) <= p.offset + epsilon)
// }

/// Compute the intersection of an edge with a plane.
///
/// Given edge endpoints v1 and v2, computes the point where the edge
/// crosses the plane boundary.
///
/// Returns `None` if:
/// - Either endpoint is on the plane (handled separately by classification
///   logic)
/// - Both endpoints are on the same side of the plane
/// - The edge is parallel to the plane
///
/// This function uses explicit epsilon checks for robustness instead of
/// relying on `signum()` which has edge cases for values near zero.
#[must_use]
pub fn compute_edge_plane_intersection(
    v1: DVec3,
    v2: DVec3,
    plane: &HalfSpace,
    epsilon: f64,
) -> Option<DVec3> {
    let d1 = plane.signed_distance(v1);
    let d2 = plane.signed_distance(v2);

    // Use epsilon for "on plane" checks to match the rest of the file.
    // If a point is on the plane, we return None - the clipping logic
    // handles "On" vertices separately via classification.
    let v1_on = d1.abs() < epsilon;
    let v2_on = d2.abs() < epsilon;

    if v1_on || v2_on {
        // If one point is on the plane, the intersection is that point.
        // If both are on, the edge is on the plane.
        // Either way, return None - this case is handled by classification logic.
        return None;
    }

    // Check if edge actually crosses the plane (strictly different sides)
    // d1 and d2 must have opposite signs (one positive, one negative)
    let crosses = (d1 > epsilon && d2 < -epsilon) || (d1 < -epsilon && d2 > epsilon);
    if !crosses {
        return None;
    }

    // Compute parameter t where edge crosses plane
    // p(t) = v1 + t*(v2 - v1)
    // n · p(t) = d
    // t = (d - n·v1) / (n·(v2 - v1))
    let dir = v2 - v1;
    let denom = plane.normal.dot(dir);

    // Parallel edge check
    if denom.abs() < epsilon {
        return None;
    }

    let t = (plane.offset - plane.normal.dot(v1)) / denom;

    // Clamp t to ensure numerical stability keeps point on segment
    let t = t.clamp(0.0, 1.0);

    Some(v1 + t * dir)
}

/// Compute the intersection of a ray with a plane.
///
/// Given a ray starting at `origin` in direction `dir`, computes where it
/// crosses the plane boundary.
///
/// Returns `None` if:
/// - The ray is parallel to the plane
/// - The ray doesn't intersect the plane in the positive direction
/// - The origin is on the plane
#[must_use]
fn compute_ray_plane_intersection(
    origin: DVec3,
    dir: DVec3,
    plane: &HalfSpace,
    epsilon: f64,
) -> Option<DVec3> {
    let d_origin = plane.signed_distance(origin);

    // Origin on plane
    if d_origin.abs() < epsilon {
        return None;
    }

    let denom = plane.normal.dot(dir);

    // Ray parallel to plane
    if denom.abs() < epsilon {
        return None;
    }

    // Compute parameter t where ray crosses plane
    // p(t) = origin + t * dir
    // n · p(t) = offset
    // t = (offset - n·origin) / (n·dir)
    let t = (plane.offset - plane.normal.dot(origin)) / denom;

    // Only positive t (ray goes forward)
    if t < 0.0 {
        return None;
    }

    Some(origin + t * dir)
}

/// Create an orthonormal basis for a plane given its normal.
fn create_plane_basis(normal: &DVec3) -> (DVec3, DVec3) {
    // Choose a vector not parallel to normal
    let arbitrary = if normal.x.abs() < 0.9 {
        DVec3::new(1.0, 0.0, 0.0)
    } else {
        DVec3::new(0.0, 1.0, 0.0)
    };

    let u = normal.cross(arbitrary).normalize();
    let v = normal.cross(u).normalize();

    (u, v)
}

// Tests

#[cfg(test)]
#[expect(clippy::unreadable_literal)]
mod tests {
    use super::*;

    #[test]
    fn test_halfspace_classification() {
        let hs = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 1.0);

        assert_eq!(
            hs.classify(DVec3::new(0.0, 0.0, 0.0), EPSILON),
            Classification::Inside
        );
        assert_eq!(
            hs.classify(DVec3::new(1.0, 0.0, 0.0), EPSILON),
            Classification::On
        );
        assert_eq!(
            hs.classify(DVec3::new(2.0, 0.0, 0.0), EPSILON),
            Classification::Outside
        );
    }

    #[test]
    fn test_three_plane_intersection() {
        // Unit cube corner at origin
        let p1 = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.0);
        let p2 = HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.0);
        let p3 = HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.0);

        let point_hom = intersect_three_planes(&p1, &p2, &p3).unwrap();
        let point = dvec4_xyz(&point_hom); // Extract Euclidean position
        assert!((point - DVec3::ZERO).length() < EPSILON);
        assert!((point_hom.w - 1.0).abs() < EPSILON); // w=1 for finite vertex
    }

    #[test]
    fn test_parallel_planes_no_intersection() {
        let p1 = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.0);
        let p2 = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 1.0);
        let p3 = HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.0);

        // Two parallel planes → no unique point
        assert!(intersect_three_planes(&p1, &p2, &p3).is_none());
    }

    #[test]
    fn test_edge_plane_intersection() {
        let plane = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5);
        let v1 = DVec3::new(0.0, 0.0, 0.0);
        let v2 = DVec3::new(1.0, 0.0, 0.0);

        let intersection = compute_edge_plane_intersection(v1, v2, &plane, EPSILON).unwrap();
        assert!((intersection.x - 0.5).abs() < EPSILON);
        assert!(intersection.y.abs() < EPSILON);
        assert!(intersection.z.abs() < EPSILON);
    }

    #[test]
    fn test_polytope_storage() {
        let mut poly = IncrementalPolytope::new();

        // Allocate and free vertices (use w=1 for finite vertices)
        let v1 = poly.alloc_vertex(Vertex {
            position: DVec4::new(0.0, 0.0, 0.0, 1.0),
            planes: vec![PlaneIdx(0), PlaneIdx(1), PlaneIdx(2)],
            edges: vec![],
        });

        assert!(poly.vertex(v1).is_some());
        poly.free_vertex(v1);
        assert!(poly.vertex(v1).is_none());

        // Reuse freed slot
        let v2 = poly.alloc_vertex(Vertex {
            position: DVec4::new(1.0, 0.0, 0.0, 1.0),
            planes: vec![PlaneIdx(0), PlaneIdx(1), PlaneIdx(3)],
            edges: vec![],
        });

        assert_eq!(v1.0, v2.0); // Same slot reused
    }

    #[test]
    fn test_edge_map() {
        let mut poly = IncrementalPolytope::new();

        let edge = Edge {
            planes: (PlaneIdx(0), PlaneIdx(1)),
            vertices: (VertexIdx(0), VertexIdx(1)),
        };

        let idx = poly.alloc_edge(edge);

        // Lookup works both ways
        assert_eq!(poly.edge_by_planes(PlaneIdx(0), PlaneIdx(1)), Some(idx));
        assert_eq!(poly.edge_by_planes(PlaneIdx(1), PlaneIdx(0)), Some(idx));
    }

    #[test]
    fn test_cube_construction() {
        let mut poly = IncrementalPolytope::new();

        // Define a unit cube centered at origin: [-0.5, 0.5]^3
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),  // +X
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5), // -X
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),  // +Y
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5), // -Y
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),  // +Z
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5), // -Z
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        assert!(poly.is_bounded());
        assert_eq!(poly.vertex_count(), 8); // Cube has 8 vertices
        assert_eq!(poly.edge_count(), 12); // Cube has 12 edges

        let (vertices, faces) = poly.to_mesh();
        assert_eq!(vertices.len(), 8);
        assert_eq!(faces.len(), 6); // Cube has 6 faces

        // Check all vertices are at corners of unit cube
        for v in &vertices {
            assert!((v.x.abs() - 0.5).abs() < EPSILON);
            assert!((v.y.abs() - 0.5).abs() < EPSILON);
            assert!((v.z.abs() - 0.5).abs() < EPSILON);
        }
    }

    #[test]
    fn test_tetrahedron_construction() {
        let mut poly = IncrementalPolytope::new();

        // Define a regular tetrahedron
        let sqrt3 = 3.0_f64.sqrt();
        let sqrt6 = 6.0_f64.sqrt();

        // Four planes forming a tetrahedron
        let normals = [
            DVec3::new(0.0, -1.0 / sqrt3, sqrt6 / 3.0).normalize(),
            DVec3::new(1.0, 1.0 / sqrt3, sqrt6 / 3.0).normalize(),
            DVec3::new(-1.0, 1.0 / sqrt3, sqrt6 / 3.0).normalize(),
            DVec3::new(0.0, 0.0, -1.0),
        ];

        for normal in &normals {
            poly.add_plane(HalfSpace::new(*normal, 1.0));
        }

        assert!(poly.is_bounded());
        assert_eq!(poly.vertex_count(), 4); // Tetrahedron has 4 vertices
        assert_eq!(poly.edge_count(), 6); // Tetrahedron has 6 edges

        let (vertices, faces) = poly.to_mesh();
        assert_eq!(vertices.len(), 4);
        assert_eq!(faces.len(), 4); // Tetrahedron has 4 faces
    }

    #[test]
    fn test_redundant_plane() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Add a redundant plane (further away than existing)
        let result = poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 1.0));

        assert!(matches!(result, AddPlaneResult::Redundant));
        assert_eq!(poly.vertex_count(), 8); // Still 8 vertices
    }

    #[test]
    fn test_clipping_cube() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Clip with a diagonal plane through the center
        let diagonal = DVec3::new(1.0, 1.0, 1.0).normalize();
        let result = poly.add_plane(HalfSpace::new(diagonal, 0.0));

        assert!(matches!(result, AddPlaneResult::Added { .. }));
        assert!(poly.vertex_count() > 0);
        assert!(poly.is_bounded());
    }

    // NOTE: test_compare_with_vertex_enumeration_cube and test_compare_with_vertex_enumeration_octahedron
    // have been removed. They required nalgebra and the vertex_enumeration module from demiurge.
    // The functionality is tested by other tests (test_cube_basic, test_octahedron_construction, etc.)

    // ========================================================================
    // Tests for New Optimizations
    // ========================================================================

    #[test]
    fn test_lazy_edge_construction() {
        let mut poly = IncrementalPolytope::new();

        // Add cube planes
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Vertices should be created
        assert_eq!(poly.vertex_count(), 8);

        // Edges should be built lazily when queried
        assert_eq!(poly.edge_count(), 12);
    }

    #[test]
    fn test_plane_removal_basic() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5), // 0: +X
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5), // 1: -X
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5), // 2: +Y
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5), // 3: -Y
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5), // 4: +Z
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5), // 5: -Z
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        assert_eq!(poly.vertex_count(), 8);
        assert!(poly.is_bounded());

        // Remove one plane (+X face)
        let result = poly.remove_plane(PlaneIdx(0));
        assert!(result.is_some());

        let result = result.unwrap();
        // Removing +X plane should remove 4 vertices (the ones on the +X face)
        assert_eq!(result.removed_vertices.len(), 4);

        // Should have 8 vertices total:
        // - 4 finite vertices (on the -X face)
        // - 4 ideal vertices (rays extending to +infinity in +X direction)
        assert_eq!(poly.vertex_count(), 8);

        // Count finite vs ideal vertices
        let finite_count = poly.vertices().filter(|(_, v)| v.is_finite()).count();
        let ideal_count = poly.vertices().filter(|(_, v)| !v.is_finite()).count();
        assert_eq!(finite_count, 4); // -X face vertices
        assert_eq!(ideal_count, 4); // Rays to +infinity

        // Should have 5 planes remaining
        assert_eq!(poly.plane_count(), 5);

        // Should be unbounded (missing +X face)
        assert!(!poly.is_bounded());
    }

    #[test]
    fn test_plane_removal_invalid() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Try to remove a non-existent plane
        let result = poly.remove_plane(PlaneIdx(100));
        assert!(result.is_none());

        // Polytope should be unchanged
        assert_eq!(poly.vertex_count(), 8);
    }

    #[test]
    fn test_plane_removal_and_readd() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5), // 0: +X
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5), // 1: -X
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5), // 2: +Y
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5), // 3: -Y
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5), // 4: +Z
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5), // 5: -Z
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Remove a plane (the +X face)
        poly.remove_plane(PlaneIdx(0));
        // After removal: 4 finite vertices (-X face) + 4 ideal vertices (rays to +X infinity)
        let vertex_count_after_remove = poly.vertex_count();
        let finite_before = poly.vertices().filter(|(_, v)| v.is_finite()).count();
        let ideal_before = poly.vertices().filter(|(_, v)| !v.is_finite()).count();

        // Debug: print vertex info
        eprintln!(
            "After remove: vertex_count={vertex_count_after_remove}, finite={finite_before}, ideal={ideal_before}"
        );
        for (v_idx, v) in poly.vertices() {
            eprintln!(
                "  vertex {:?}: finite={}, planes={:?}, edges={:?}",
                v_idx,
                v.is_finite(),
                v.planes,
                v.edges
            );
        }

        assert_eq!(vertex_count_after_remove, 8);
        assert_eq!(finite_before, 4);
        assert_eq!(ideal_before, 4);

        // Ensure edges are valid before re-adding
        let edge_count = poly.edge_count();
        eprintln!("Edge count before re-add: {edge_count}");

        // Check vertices again after edge_count (which calls ensure_edges_valid)
        eprintln!("After ensure_edges_valid:");
        for (v_idx, v) in poly.vertices() {
            eprintln!(
                "  vertex {:?}: finite={}, planes={:?}, edges={:?}",
                v_idx,
                v.is_finite(),
                v.planes,
                v.edges
            );
        }

        // Re-add the same plane - this clips the rays at x=0.5, creating finite vertices
        // First, manually check classifications
        let test_plane = HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5);
        eprintln!(
            "Test plane: normal={:?}, offset={}",
            test_plane.normal, test_plane.offset
        );
        for (v_idx, v) in poly.vertices() {
            if v.is_finite() {
                let pos = v.to_euclidean().unwrap();
                let dist = test_plane.normal.dot(pos) - test_plane.offset;
                eprintln!("  vertex {v_idx:?} (finite): pos={pos:?}, dist={dist:.3}");
            } else {
                let dir = v.direction().unwrap();
                let dot = test_plane.normal.dot(dir);
                eprintln!("  vertex {v_idx:?} (ideal): dir={dir:?}, dot={dot:.3}");
            }
        }

        let result = poly.add_plane(test_plane);
        eprintln!("Re-add result: {result:?}");

        // The plane clips the ideal vertices and creates new finite vertices
        assert!(matches!(result, AddPlaneResult::Added { .. }));

        // Should be back to 8 finite vertices (full cube), no ideal vertices
        assert_eq!(poly.vertex_count(), 8);
        let finite_after = poly.vertices().filter(|(_, v)| v.is_finite()).count();
        let ideal_after = poly.vertices().filter(|(_, v)| !v.is_finite()).count();
        assert_eq!(finite_after, 8);
        assert_eq!(ideal_after, 0);
    }

    #[test]
    fn test_face_ordering_cached() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Call to_mesh twice - second call should use cached orderings
        let (vertices1, faces1) = poly.to_mesh();
        let (vertices2, faces2) = poly.to_mesh();

        // Results should be identical
        assert_eq!(vertices1.len(), vertices2.len());
        assert_eq!(faces1.len(), faces2.len());

        for (f1, f2) in faces1.iter().zip(faces2.iter()) {
            assert_eq!(f1, f2);
        }
    }

    #[test]
    fn test_planes_iterator() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // Should have 6 planes
        assert_eq!(poly.plane_count(), 6);
        assert_eq!(poly.planes().count(), 6);

        // Remove one plane
        poly.remove_plane(PlaneIdx(0));

        // Should have 5 planes now
        assert_eq!(poly.plane_count(), 5);
        assert_eq!(poly.planes().count(), 5);

        // Verify plane indices are correct
        let plane_indices: Vec<PlaneIdx> = poly.planes().map(|(idx, _)| idx).collect();
        assert!(!plane_indices.contains(&PlaneIdx(0))); // Removed plane
        assert!(plane_indices.contains(&PlaneIdx(1)));
        assert!(plane_indices.contains(&PlaneIdx(2)));
    }

    #[test]
    fn test_to_mesh_no_rebuild() {
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        // to_mesh_no_rebuild should work without triggering lazy rebuilds
        let (vertices, faces) = poly.to_mesh_no_rebuild();

        assert_eq!(vertices.len(), 8);
        assert_eq!(faces.len(), 6);
    }

    // ========================================================================
    // Regression Tests for Unbounded Polytope Support
    // ========================================================================

    #[test]
    fn test_three_plane_corner_case() {
        // Test that 3 non-parallel planes form an unbounded corner:
        // 1 finite vertex at intersection + 3 ideal vertices (rays to infinity)
        let mut poly = IncrementalPolytope::new();

        // Three planes meeting at origin, forming an octant corner
        poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.0)); // x <= 0
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.0)); // y <= 0
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.0)); // z <= 0

        // Should have 4 vertices: 1 finite (at origin) + 3 ideal (ray directions)
        assert_eq!(poly.vertex_count(), 4);

        let finite_count = poly.vertices().filter(|(_, v)| v.is_finite()).count();
        let ideal_count = poly.vertices().filter(|(_, v)| !v.is_finite()).count();

        assert_eq!(finite_count, 1, "Should have exactly 1 finite vertex");
        assert_eq!(ideal_count, 3, "Should have exactly 3 ideal vertices");

        // The finite vertex should be at the origin
        let finite_vertex = poly
            .vertices()
            .find(|(_, v)| v.is_finite())
            .map(|(_, v)| v.to_euclidean().unwrap());
        assert!(finite_vertex.is_some());
        let pos = finite_vertex.unwrap();
        assert!(pos.length() < EPSILON, "Finite vertex should be at origin");

        // Polytope should be unbounded (only 3 planes)
        assert!(!poly.is_bounded());

        // Should have 3 edges (rays from corner to infinity)
        assert_eq!(poly.edge_count(), 3);
    }

    #[test]
    fn test_plane_removal_makes_unbounded() {
        // Regression test: removing a plane from a bounded polytope should
        // correctly detect that it becomes unbounded
        let mut poly = IncrementalPolytope::new();

        // Create a cube (bounded)
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),  // +X
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5), // -X
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),  // +Y
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5), // -Y
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),  // +Z
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5), // -Z
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        assert!(poly.is_bounded(), "Cube should be bounded");
        assert_eq!(poly.vertex_count(), 8);

        // Remove the +X face (PlaneIdx(0))
        let result = poly.remove_plane(PlaneIdx(0));
        assert!(result.is_some(), "Plane removal should succeed");

        // After removing one face, the polytope should be unbounded
        // (extends to infinity in +X direction)
        assert!(
            !poly.is_bounded(),
            "Cube with one face removed should be unbounded"
        );

        // Should still have 5 planes
        assert_eq!(poly.plane_count(), 5);
    }

    #[test]
    fn test_clipping_preserves_edges() {
        // Regression test for Bug 1: ensure clip_by_plane edges are not
        // overwritten by subsequent edge rebuilds
        let mut poly = IncrementalPolytope::new();

        // Create a cube
        let planes = [
            HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.5),
            HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 0.5),
        ];

        for plane in planes {
            poly.add_plane(plane);
        }

        let initial_edge_count = poly.edge_count();
        assert_eq!(initial_edge_count, 12, "Cube should have 12 edges");

        // Clip with a diagonal plane (slightly offset to avoid degeneracy)
        let diagonal = DVec3::new(1.0, 1.0, 1.1).normalize();
        let result = poly.add_plane(HalfSpace::new(diagonal, 0.0));

        assert!(matches!(result, AddPlaneResult::Added { .. }));

        // Verify edges are consistent after clipping
        let edge_count_after_clip = poly.edge_count();

        // The clipped polytope should have valid edges
        // (exact count depends on geometry, but should be > 0 and reasonable)
        assert!(
            edge_count_after_clip > 0,
            "Should have edges after clipping"
        );

        // Collect edge data first to avoid borrow issues with mutable iteration
        let edge_data: Vec<_> = poly
            .edges()
            .map(|(e_idx, e)| (e_idx, e.vertices, e.planes))
            .collect();

        // Verify each edge connects valid vertices and planes
        for (e_idx, vertices, planes) in &edge_data {
            assert!(
                poly.vertex(vertices.0).is_some() && poly.vertex(vertices.1).is_some(),
                "Edge {e_idx:?} should connect valid vertices"
            );
            assert!(
                poly.plane(planes.0).is_some() && poly.plane(planes.1).is_some(),
                "Edge {e_idx:?} should reference valid planes"
            );
        }

        // Collect vertex edge references to avoid borrow issues
        let vertex_edges: Vec<_> = poly
            .vertices()
            .map(|(v_idx, v)| (v_idx, v.edges.clone()))
            .collect();

        // Verify vertices reference valid edges
        for (v_idx, edges) in &vertex_edges {
            for &e_idx in edges {
                assert!(
                    poly.edge(e_idx).is_some(),
                    "Vertex {v_idx:?} references invalid edge {e_idx:?}"
                );
            }
        }
    }

    #[test]
    fn test_unbounded_to_mesh_clipped() {
        // Test that unbounded polytopes can be converted to mesh via clipping
        let mut poly = IncrementalPolytope::new();

        // Create an unbounded half-space (single plane)
        // Actually need at least 3 planes to form geometry
        poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 0.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 0.0));

        assert!(!poly.is_bounded(), "3-plane corner should be unbounded");

        // Clip to a bounding box and convert to mesh
        let min = DVec3::new(-2.0, -2.0, -2.0);
        let max = DVec3::new(0.0, 0.0, 0.0);
        let (vertices, faces) = poly.to_mesh_clipped(min, max);

        // Should produce a valid mesh (the corner of the bounding box)
        assert!(!vertices.is_empty(), "Clipped mesh should have vertices");
        assert!(!faces.is_empty(), "Clipped mesh should have faces");

        // All vertices should be within or on the bounding box
        for v in &vertices {
            assert!(
                v.x >= min.x - EPSILON && v.x <= max.x + EPSILON,
                "Vertex x={} outside bounds",
                v.x
            );
            assert!(
                v.y >= min.y - EPSILON && v.y <= max.y + EPSILON,
                "Vertex y={} outside bounds",
                v.y
            );
            assert!(
                v.z >= min.z - EPSILON && v.z <= max.z + EPSILON,
                "Vertex z={} outside bounds",
                v.z
            );
        }
    }

    #[test]
    fn test_icosahedron_construction() {
        // Planes forming an icosahedron (20 face normals)
        let normals = vec![
            DVec3::new(0.934172, 0.356822, 0.000000),
            DVec3::new(0.934172, -0.356822, 0.000000),
            DVec3::new(-0.934172, 0.356822, 0.000000),
            DVec3::new(-0.934172, -0.356822, 0.000000),
            DVec3::new(0.000000, 0.934172, 0.356822),
            DVec3::new(0.000000, 0.934172, -0.356822),
            DVec3::new(0.356822, 0.000000, -0.934172),
            DVec3::new(-0.356822, 0.000000, -0.934172),
            DVec3::new(0.000000, -0.934172, -0.356822),
            DVec3::new(0.000000, -0.934172, 0.356822),
            DVec3::new(0.356822, 0.000000, 0.934172),
            DVec3::new(-0.356822, 0.000000, 0.934172),
            DVec3::new(0.577350, 0.577350, -0.577350),
            DVec3::new(0.577350, 0.577350, 0.577350),
            DVec3::new(-0.577350, 0.577350, -0.577350),
            DVec3::new(-0.577350, 0.577350, 0.577350),
            DVec3::new(0.577350, -0.577350, -0.577350),
            DVec3::new(0.577350, -0.577350, 0.577350),
            DVec3::new(-0.577350, -0.577350, -0.577350),
            DVec3::new(-0.577350, -0.577350, 0.577350),
        ];

        let mut poly = IncrementalPolytope::new();
        for normal in &normals {
            poly.add_plane(HalfSpace::new(normal.normalize(), 2.0));
        }

        assert!(poly.is_bounded(), "Polytope should be bounded");

        // The intersection of these 20 half-spaces produces an icosahedron:
        // - 12 vertices (each at the intersection of 3+ planes)
        // - 30 edges (each at the intersection of 2 planes)
        // - 20 faces (one per plane)
        // Euler: 12 - 30 + 20 = 2 ✓
        assert_eq!(poly.vertex_count(), 12);
        assert_eq!(poly.edge_count(), 30);

        let (vertices, faces) = poly.to_mesh();
        assert_eq!(vertices.len(), 12);
        assert_eq!(faces.len(), 20);
    }

    /// This test has 11 planes forming a complex polytope.
    #[test]
    fn test_edge_case_polytope() {
        // 11 planes: [nx, ny, nz, d] where n·x ≤ d
        #[rustfmt::skip]
        let planes_data: [(f64, f64, f64, f64); 11] = [
            (-0.2840365469455719,   -0.9563649892807007,   -0.0684785395860672,      -4.3069167137146),
            (-0.4269607663154602,   -0.9034146666526794,  0.039323657751083374,    -4.109272480010986),
            (0.4125642478466034,    0.9070128798484802,  -0.08437052369117737,    4.1034626960754395),
            (-0.28315502405166626,    0.8755069375038147,    0.3915492296218872,    3.8051040172576904),
            (-0.958615243434906,   0.28470486402511597,                     0.,     0.425659716129303),
            (-0.019496172666549683,  -0.06564456969499588,    0.9976526498794556,    0.5745928287506104),
            (0.958615243434906,  -0.28470486402511597,                    -0.,  -0.39629310369491577),
            (0.019496172666549683,   0.06564456969499588,   -0.9976526498794556,   -0.5452262163162231),
            (0.28315502405166626,   -0.8755069375038147,   -0.3915492296218872,   -3.7757372856140137),
            (0.9590741395950317,   0.25848281383514404,   0.11560016870498657,    1.9914697408676147),
            (-0.9590741395950317,  -0.25848281383514404,  -0.11560016870498657,    -1.962103009223938),
        ];

        let mut poly = IncrementalPolytope::new();
        for (nx, ny, nz, d) in planes_data {
            let normal = DVec3::new(nx, ny, nz);
            poly.add_plane(HalfSpace::new(normal, d));
        }

        assert!(poly.is_bounded(), "Polytope should be bounded");

        let (vertices, faces) = poly.to_mesh();

        // Verify Euler's formula: V - E + F = 2
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let euler = vertices.len() as i32 - poly.edge_count() as i32 + faces.len() as i32;
        assert_eq!(euler, 2, "Euler's formula should hold");
    }

    #[test]
    fn test_validate_topology_cube() {
        let mut poly = IncrementalPolytope::new();

        // Build a cube
        poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 1.0));
        poly.add_plane(HalfSpace::new(DVec3::new(-1.0, 0.0, 0.0), 1.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 1.0, 0.0), 1.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, -1.0, 0.0), 1.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, 1.0), 1.0));
        poly.add_plane(HalfSpace::new(DVec3::new(0.0, 0.0, -1.0), 1.0));

        // Cube should pass validation
        assert!(
            poly.validate_topology().is_ok(),
            "Cube should have valid topology"
        );
    }

    #[test]
    fn test_validate_topology_after_removal() {
        let mut poly = IncrementalPolytope::new();

        // Build a polytope with enough planes
        let golden = f64::midpoint(1.0, 5.0_f64.sqrt());
        for i in 0..20 {
            let theta = std::f64::consts::TAU * f64::from(i) / golden;
            let phi = (1.0 - 2.0 * (f64::from(i) + 0.5) / 20.0).acos();
            let x = phi.sin() * theta.cos();
            let y = phi.sin() * theta.sin();
            let z = phi.cos();
            poly.add_plane(HalfSpace::new(DVec3::new(x, y, z), 1.0));
        }

        // Should be valid before removal
        assert!(
            poly.validate_topology().is_ok(),
            "Should be valid before removal"
        );

        // Remove a plane
        poly.remove_plane(PlaneIdx(5));

        // Should still be valid after removal (bug was fixed)
        assert!(
            poly.validate_topology().is_ok(),
            "Should be valid after removal"
        );
    }
}
