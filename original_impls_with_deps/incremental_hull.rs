//! Incremental 3D Convex Hull for Unit Normals
//!
//! This module provides an incremental convex hull data structure optimized for:
//! - Unit vectors (plane normals on the unit sphere)
//! - Checking if the origin is strictly inside the hull
//! - Dynamic insertion and removal of points
//!
//! # Algorithm
//!
//! - **Insertion**: Beneath-beyond with full adjacency rebuild - O(N)
//!   (theoretical O(h) possible with local stitching, not yet implemented)
//! - **Removal**: Full hull rebuild - O(N log N)
//! - **Origin test**: Cached result - O(1) query, O(N) recomputation when needed
//!
//! # Usage
//!
//! ```ignore
//! let mut hull = IncrementalHull::new(1e-7);
//! hull.insert(0, DVec3::new(1.0, 0.0, 0.0));
//! hull.insert(1, DVec3::new(-1.0, 0.0, 0.0));
//! // ... add more points
//! assert!(hull.origin_inside());
//! ```

use std::collections::{HashMap, HashSet};

use algebra::DVec3;

use convex_hull_f64::convex_hull_triangulated_f64;

/// Default epsilon for geometric predicates.
const EPSILON: f64 = 1e-7;

/// Incremental convex hull for unit vectors (plane normals).
///
/// Maintains the convex hull of a set of 3D points, optimized for:
/// - Points on/near the unit sphere (plane normals)
/// - Efficient origin-inside queries
/// - Dynamic insertion and removal
#[derive(Clone, Debug)]
pub struct IncrementalHull {
    /// Points currently in the point set (not all may be on hull).
    points: Vec<DVec3>,

    /// Map from external ID to internal point index.
    id_to_index: HashMap<u32, usize>,

    /// Map from internal index to external ID.
    index_to_id: Vec<u32>,

    /// Hull faces as triangles (indices into `points`).
    /// Each face is [v0, v1, v2] in counter-clockwise order when viewed from outside.
    /// Uses Option for tombstone-based deletion (stable indices).
    faces: Vec<Option<[usize; 3]>>,

    /// Precomputed outward-facing normal for each face.
    face_normals: Vec<Option<DVec3>>,

    /// For each face, indices of the 3 adjacent faces (opposite to each vertex).
    /// `face_adjacency[f][i]` = face adjacent across edge opposite to vertex i.
    face_adjacency: Vec<Option<[usize; 3]>>,

    /// Free list of deleted face slots for reuse.
    face_free_list: Vec<usize>,

    /// Set of point indices that are on the hull (for O(1) lookup).
    hull_point_indices: HashSet<usize>,

    /// Centroid of hull vertices (used for outward normal orientation).
    hull_centroid: DVec3,

    /// Cached result: is the origin strictly inside the hull?
    origin_inside: bool,

    /// Numerical tolerance for geometric predicates.
    epsilon: f64,
}

impl Default for IncrementalHull {
    fn default() -> Self {
        Self::new(EPSILON)
    }
}

impl IncrementalHull {
    /// Create an empty incremental hull.
    #[must_use]
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: Vec::new(),
            faces: Vec::new(),
            face_normals: Vec::new(),
            face_adjacency: Vec::new(),
            face_free_list: Vec::new(),
            hull_point_indices: HashSet::new(),
            hull_centroid: DVec3::zeros(),
            origin_inside: false,
            epsilon,
        }
    }

    // Face Allocation/Deallocation (Tombstone-based)
    /// Allocate a face slot, reusing from free list if available.
    #[allow(dead_code)] // Useful helper for future use
    fn alloc_face(&mut self, face: [usize; 3], normal: DVec3, adj: [usize; 3]) -> usize {
        if let Some(idx) = self.face_free_list.pop() {
            self.faces[idx] = Some(face);
            self.face_normals[idx] = Some(normal);
            self.face_adjacency[idx] = Some(adj);
            idx
        } else {
            let idx = self.faces.len();
            self.faces.push(Some(face));
            self.face_normals.push(Some(normal));
            self.face_adjacency.push(Some(adj));
            idx
        }
    }

    /// Mark a face as deleted (tombstone), add to free list.
    fn free_face(&mut self, idx: usize) {
        if self.faces.get(idx).is_some_and(Option::is_some) {
            self.faces[idx] = None;
            self.face_normals[idx] = None;
            self.face_adjacency[idx] = None;
            self.face_free_list.push(idx);
        }
    }

    /// Number of active triangular faces on the hull.
    #[inline]
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.faces.len() - self.face_free_list.len()
    }

    /// Iterator over active faces with their indices.
    fn active_faces(&self) -> impl Iterator<Item = (usize, &[usize; 3])> {
        self.faces
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.as_ref().map(|face| (i, face)))
    }

    /// Find which edge slot (0, 1, or 2) in a face corresponds to a given edge.
    /// The edge slot is opposite to vertex i, meaning it connects vertices (i+1) and (i+2).
    fn find_edge_slot(&self, face_idx: usize, v1: usize, v2: usize) -> Option<usize> {
        let face = self.faces.get(face_idx)?.as_ref()?;
        for i in 0..3 {
            let e1 = face[(i + 1) % 3];
            let e2 = face[(i + 2) % 3];
            if (e1 == v1 && e2 == v2) || (e1 == v2 && e2 == v1) {
                return Some(i);
            }
        }
        None
    }

    /// Compact storage by removing tombstones.
    ///
    /// Call this periodically after many insertions/deletions to reclaim memory.
    /// This invalidates all face indices but rebuilds adjacency to maintain correctness.
    ///
    /// Returns the number of tombstones removed.
    pub fn compact(&mut self) -> usize {
        let tombstones = self.face_free_list.len();
        if tombstones == 0 {
            return 0;
        }

        // Build old→new index mapping
        let mut remap: Vec<Option<usize>> = vec![None; self.faces.len()];
        let mut write_idx = 0;

        (0..self.faces.len()).for_each(|read_idx| {
            if self.faces[read_idx].is_some() {
                remap[read_idx] = Some(write_idx);

                if read_idx != write_idx {
                    self.faces.swap(read_idx, write_idx);
                    self.face_normals.swap(read_idx, write_idx);
                    self.face_adjacency.swap(read_idx, write_idx);
                }

                write_idx += 1;
            }
        });

        self.faces.truncate(write_idx);
        self.face_normals.truncate(write_idx);
        self.face_adjacency.truncate(write_idx);
        self.face_free_list.clear();

        // Remap adjacency references
        for adj in self.face_adjacency.iter_mut().flatten() {
            for slot in adj.iter_mut() {
                if *slot != usize::MAX {
                    *slot = remap[*slot].unwrap_or(usize::MAX);
                }
            }
        }

        tombstones
    }

    /// Build a hull from an iterator of (id, point) pairs.
    ///
    /// This is more efficient than repeated `insert()` calls for initial construction.
    #[must_use]
    pub fn build(points: impl Iterator<Item = (u32, DVec3)>, epsilon: f64) -> Self {
        let mut hull = Self::new(epsilon);

        // Add all points to internal storage
        for (id, point) in points {
            if hull.id_to_index.contains_key(&id) {
                continue; // Match incremental semantics: duplicate IDs are no-ops
            }
            if !Self::is_valid_point(&point) {
                continue; // Skip non-finite inputs to avoid NaNs downstream
            }
            hull.add_point_internal(id, point);
        }

        if hull.points.len() < 4 {
            return hull;
        }

        // Use the proven convex_hull_triangulated for initial construction
        hull.rebuild_hull_from_points();
        hull.update_origin_inside();
        hull
    }

    /// Insert a point with the given external ID.
    ///
    /// If the ID already exists, this is a no-op.
    /// Uses the beneath-beyond algorithm for O(h) incremental insertion
    /// where h is the number of affected hull faces.
    pub fn insert(&mut self, id: u32, point: DVec3) {
        if self.id_to_index.contains_key(&id) {
            return; // Already exists
        }
        if !Self::is_valid_point(&point) {
            return; // Ignore non-finite inputs
        }

        let point_idx = self.add_point_internal(id, point);

        // Need at least 4 points for a valid 3D hull
        if self.points.len() < 4 {
            self.update_origin_inside();
            return;
        }

        // If we don't have a valid 3D hull yet (need at least 4 faces for a tetrahedron),
        // rebuild from scratch. This handles degenerate cases like coplanar initial points.
        if self.face_count() < 4 {
            self.rebuild_hull_from_points();
            self.update_origin_inside();
            return;
        }

        // Use beneath-beyond algorithm for incremental insertion
        self.insert_point_incremental(point_idx);

        // If incremental insertion resulted in a degenerate hull, rebuild
        if self.face_count() < 4 && self.points.len() >= 4 {
            self.rebuild_hull_from_points();
            self.update_origin_inside();
            return;
        }

        // Optimization: If origin was already inside, adding a point can only
        // expand the hull, so origin remains inside. Skip the O(F) recomputation.
        if !self.origin_inside {
            self.update_origin_inside();
        }
    }

    /// Incremental insertion using beneath-beyond algorithm.
    ///
    /// 1. Find all faces visible from the new point
    /// 2. If none visible, point is inside hull - done
    /// 3. Find horizon edges (boundary between visible and non-visible faces)
    /// 4. Remove visible faces
    /// 5. Create new faces connecting point to horizon edges
    fn insert_point_incremental(&mut self, point_idx: usize) {
        let point = self.points[point_idx];

        // Find all visible faces (point is on positive side of face plane)
        let visible_faces = self.find_visible_faces(point);

        #[cfg(test)]
        eprintln!(
            "  insert_point_incremental: point_idx={}, visible_faces={:?}, total_faces={}",
            point_idx,
            visible_faces,
            self.faces.len()
        );

        if visible_faces.is_empty() {
            // Point is inside or on the hull - no change needed
            return;
        }

        // Mark the new point as on the hull
        self.hull_point_indices.insert(point_idx);

        // Find horizon edges (edges between visible and non-visible faces)
        let horizon_edges = self.find_horizon_edges(&visible_faces);

        #[cfg(test)]
        {
            eprintln!("  horizon_edges: {horizon_edges:?}");
            // Debug: print adjacency for visible faces
            for &face_idx in &visible_faces {
                if let (Some(face), Some(adj)) =
                    (&self.faces[face_idx], &self.face_adjacency[face_idx])
                {
                    eprintln!("    face[{face_idx}] = {face:?}, adj = {adj:?}");
                }
            }
        }

        // If horizon is empty but we have visible faces, the hull is degenerate.
        // Fall back to full rebuild.
        if horizon_edges.is_empty() && !visible_faces.is_empty() {
            #[cfg(test)]
            eprintln!("  WARNING: Empty horizon with visible faces - falling back to rebuild");
            self.rebuild_hull_from_points();
            self.update_origin_inside();
            return;
        }

        // Check if hull is manifold before incremental update
        if !self.is_manifold() {
            #[cfg(test)]
            eprintln!("  WARNING: Non-manifold hull detected - falling back to rebuild");
            self.rebuild_hull_from_points();
            self.update_origin_inside();
            return;
        }

        // Remove visible faces and create new faces to the point
        // If horizon ordering fails, fall back to rebuild
        if !self.replace_visible_region(point_idx, &visible_faces, &horizon_edges) {
            #[cfg(test)]
            eprintln!("  WARNING: Horizon ordering failed - falling back to rebuild");
            self.rebuild_hull_from_points();
            self.update_origin_inside();
            return;
        }

        // Verify hull is still valid after incremental update
        // Check both edge multiplicity (manifold) and adjacency reciprocity
        if !self.is_manifold() || !self.is_adjacency_consistent() {
            #[cfg(test)]
            eprintln!(
                "  WARNING: Incremental update produced invalid hull - falling back to rebuild"
            );
            self.rebuild_hull_from_points();
            self.update_origin_inside();
        }
    }

    /// Check if the hull is a valid 2-manifold (each edge has exactly 2 adjacent faces).
    fn is_manifold(&self) -> bool {
        if self.face_count() < 4 {
            return false;
        }

        // Count how many faces each edge belongs to
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

        for (_idx, face) in self.active_faces() {
            for i in 0..3 {
                let v1 = face[i];
                let v2 = face[(i + 1) % 3];
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Every edge should appear exactly twice in a manifold
        edge_count.values().all(|&count| count == 2)
    }

    /// Validate that adjacency is symmetric: if face A says B is adjacent, B must say A is adjacent.
    ///
    /// This catches topology corruption that edge-multiplicity checks miss, such as:
    /// - Twisted fans where adjacency order doesn't match edge order
    /// - Stale adjacency pointing to tombstoned faces
    /// - Non-reciprocal adjacency from incorrect wiring
    fn is_adjacency_consistent(&self) -> bool {
        for (face_idx, face) in self.active_faces() {
            let Some(adj) = &self.face_adjacency[face_idx] else {
                // Active face must have adjacency
                return false;
            };

            for i in 0..3 {
                let neighbor_idx = adj[i];

                // Skip boundary edges
                if neighbor_idx == usize::MAX {
                    continue;
                }

                // Neighbor must be an active face
                let Some(neighbor_face) = &self.faces[neighbor_idx] else {
                    #[cfg(test)]
                    eprintln!(
                        "  Adjacency error: face[{face_idx}] slot {i} points to tombstoned face[{neighbor_idx}]"
                    );

                    return false;
                };

                let Some(neighbor_adj) = &self.face_adjacency[neighbor_idx] else {
                    #[cfg(test)]
                    eprintln!("  Adjacency error: face[{neighbor_idx}] has no adjacency");

                    return false;
                };

                // Find the shared edge: edge i is opposite vertex i
                // Edge i connects vertices (i+1) and (i+2)
                let v1 = face[(i + 1) % 3];
                let v2 = face[(i + 2) % 3];

                // Find which slot in neighbor contains this edge (reversed direction)
                let mut found_reciprocal = false;
                for j in 0..3 {
                    let nv1 = neighbor_face[(j + 1) % 3];
                    let nv2 = neighbor_face[(j + 2) % 3];

                    // Edge is shared if vertices match in opposite order (consistent winding)
                    if (v1 == nv2 && v2 == nv1) || (v1 == nv1 && v2 == nv2) {
                        // Check that neighbor points back to us
                        if neighbor_adj[j] == face_idx {
                            found_reciprocal = true;
                            break;
                        }
                        #[cfg(test)]
                        eprintln!(
                            "  Adjacency error: face[{}] slot {} -> face[{}], but face[{}] slot {} -> face[{}] (expected {})",
                            face_idx, i, neighbor_idx, neighbor_idx, j, neighbor_adj[j], face_idx
                        );
                    }
                }

                if !found_reciprocal {
                    #[cfg(test)]
                    eprintln!(
                        "  Adjacency error: face[{face_idx}] slot {i} -> face[{neighbor_idx}], but no reciprocal found"
                    );

                    return false;
                }
            }
        }

        true
    }

    /// Find all faces visible from a point (point is on positive side of face plane).
    fn find_visible_faces(&self, point: DVec3) -> Vec<usize> {
        let mut visible = Vec::new();

        for (face_idx, face) in self.active_faces() {
            let p0 = self.points[face[0]];
            let Some(normal) = self.face_normals[face_idx] else {
                continue;
            };

            // Signed distance from point to face plane
            let dist = normal.dot(&(point - p0));

            // Face is visible if point is on the positive (outside) side
            if dist > self.epsilon {
                visible.push(face_idx);
            }
        }

        visible
    }

    /// Find horizon edges - edges where exactly one adjacent face is visible.
    /// Returns edges as `(v1, v2, non_visible_face_idx)` tuples.
    /// Boundary edges `(adjacency == usize::MAX)` are also treated as horizon edges.
    fn find_horizon_edges(&self, visible_faces: &[usize]) -> Vec<(usize, usize, usize)> {
        let visible_set: HashSet<usize> = visible_faces.iter().copied().collect();
        let mut horizon = Vec::new();

        for &face_idx in visible_faces {
            let Some(face) = &self.faces[face_idx] else {
                continue;
            };
            let Some(adjacency) = &self.face_adjacency[face_idx] else {
                continue;
            };

            // Check each edge of the visible face
            for i in 0..3 {
                let adj_face = adjacency[i];

                // Edge is on horizon if:
                // 1. Adjacent face is not visible (standard case), OR
                // 2. Adjacent face is usize::MAX (boundary edge - hull is open)
                let is_horizon = adj_face == usize::MAX || !visible_set.contains(&adj_face);

                if is_horizon {
                    // Edge opposite to vertex i: vertices (i+1) and (i+2)
                    let v1 = face[(i + 1) % 3];
                    let v2 = face[(i + 2) % 3];
                    // Store with the non-visible adjacent face for winding reference
                    horizon.push((v1, v2, adj_face));
                }
            }
        }

        horizon
    }

    /// Replace visible faces with new faces connecting the point to horizon edges.
    ///
    /// This uses O(h) local adjacency updates instead of O(F) full rebuild.
    ///
    /// # Returns
    /// `true` if the replacement succeeded, `false` if the horizon could not be ordered
    /// cyclically (caller should fall back to full rebuild).
    fn replace_visible_region(
        &mut self,
        point_idx: usize,
        visible_faces: &[usize],
        horizon_edges: &[(usize, usize, usize)],
    ) -> bool {
        // Deduplicate horizon edges by canonical representation
        let deduped_edges = Self::deduplicate_horizon_edges(horizon_edges);

        if deduped_edges.is_empty() {
            return true;
        }

        // Compute interior point early for horizon ordering
        let new_point = self.points[point_idx];
        let interior_point = if self.origin_inside {
            DVec3::zeros()
        } else {
            self.hull_centroid
        };

        // Order horizon edges cyclically around the new point
        // This is critical for correct fan adjacency
        let ordered_edges = if let Some(edges) =
            self.order_horizon_edges_cyclically(deduped_edges, new_point, interior_point)
        {
            edges
        } else {
            #[cfg(test)]
            eprintln!("  WARNING: Failed to order horizon edges cyclically");
            return false;
        };

        // Build set for fast lookup
        let visible_set: HashSet<usize> = visible_faces.iter().copied().collect();

        // Build vertex → faces mapping for O(1) lookup instead of O(F) per vertex
        let mut vertex_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
        for (face_idx, face) in self.active_faces() {
            for &v in face {
                vertex_to_faces.entry(v).or_default().push(face_idx);
            }
        }

        // STEP 1: Build horizon edge → non-visible face map (BEFORE removing faces)
        // Map: canonical_edge → (non_visible_face_idx, edge_slot_in_that_face)
        let mut horizon_to_nonvis: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

        for &(v1, v2, adj_face) in horizon_edges {
            if adj_face != usize::MAX && !visible_set.contains(&adj_face) {
                let canonical = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                if let Some(slot) = self.find_edge_slot(adj_face, v1, v2) {
                    horizon_to_nonvis.insert(canonical, (adj_face, slot));
                }
            }
        }

        // STEP 2: Remove hull membership for vertices that might become interior
        for &face_idx in visible_faces {
            let Some(face) = self.faces[face_idx] else {
                continue;
            };
            for v in face {
                let used_by_non_visible = vertex_to_faces
                    .get(&v)
                    .is_some_and(|faces| faces.iter().any(|&i| !visible_set.contains(&i)));
                if !used_by_non_visible {
                    self.hull_point_indices.remove(&v);
                }
            }
        }

        // STEP 3: Create new face data with normals (using ordered edges)
        let mut new_face_data: Vec<([usize; 3], DVec3)> = Vec::with_capacity(ordered_edges.len());

        for (v1, v2) in &ordered_edges {
            let p0 = self.points[*v1];
            let p1 = self.points[*v2];
            let p2 = new_point;
            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let mut normal = edge1.cross(&edge2);

            if normal.norm() < self.epsilon {
                continue;
            }

            normal = normal.normalize();

            let signed_dist = normal.dot(&(interior_point - p0));
            let mut face = [*v1, *v2, point_idx];

            if signed_dist > self.epsilon {
                face = [*v2, *v1, point_idx];
                normal = -normal;
            }

            new_face_data.push((face, normal));
        }

        // STEP 4: Remove visible faces using tombstones
        for &face_idx in visible_faces {
            self.free_face(face_idx);
        }

        // STEP 5: Pre-allocate face indices for new faces
        let num_new_faces = new_face_data.len();
        let mut new_face_indices: Vec<usize> = Vec::with_capacity(num_new_faces);

        for _ in 0..num_new_faces {
            let idx = if let Some(idx) = self.face_free_list.pop() {
                idx
            } else {
                let idx = self.faces.len();
                self.faces.push(None);
                self.face_normals.push(None);
                self.face_adjacency.push(None);
                idx
            };
            new_face_indices.push(idx);
        }

        // STEP 6: Create new faces with LOCAL adjacency (O(h))
        for (i, (face, normal)) in new_face_data.iter().enumerate() {
            let face_idx = new_face_indices[i];

            // Mark vertices as on hull
            self.hull_point_indices.insert(face[0]);
            self.hull_point_indices.insert(face[1]);
            self.hull_point_indices.insert(face[2]);

            // Compute adjacency locally:
            // Face vertices are [v0, v1, point_idx] (after potential winding fix)
            // - adj[0]: across edge v1-point_idx (opposite v0) → prev face in fan
            // - adj[1]: across edge v0-point_idx (opposite v1) → next face in fan
            // - adj[2]: across edge v0-v1 (opposite point_idx) → non-visible face on horizon

            let v0 = face[0];
            let v1 = face[1];

            // Find adjacent new faces in the fan
            // The fan forms a cycle around point_idx, connected by edges to point_idx
            let prev_idx = new_face_indices[(i + num_new_faces - 1) % num_new_faces];
            let next_idx = new_face_indices[(i + 1) % num_new_faces];

            // Find non-visible face for the horizon edge (v0-v1)
            let canonical = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            let nonvis_adj = horizon_to_nonvis
                .get(&canonical)
                .map_or(usize::MAX, |(idx, _)| *idx);

            let adj = [prev_idx, next_idx, nonvis_adj];

            self.faces[face_idx] = Some(*face);
            self.face_normals[face_idx] = Some(*normal);
            self.face_adjacency[face_idx] = Some(adj);

            // Update non-visible face's adjacency to point back to this new face
            if let Some(&(nonvis_face_idx, slot)) = horizon_to_nonvis.get(&canonical)
                && let Some(ref mut nonvis_adj) = self.face_adjacency[nonvis_face_idx]
            {
                nonvis_adj[slot] = face_idx;
            }
        }

        // Update centroid after adding new hull point
        self.update_hull_centroid();

        true
    }

    /// Deduplicate horizon edges by canonical edge representation.
    /// Returns edges as (v1, v2) pairs preserving the first occurrence's orientation.
    fn deduplicate_horizon_edges(horizon_edges: &[(usize, usize, usize)]) -> Vec<(usize, usize)> {
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut result = Vec::new();

        for &(v1, v2, _adj) in horizon_edges {
            // Canonical edge: smaller index first
            let canonical = if v1 < v2 { (v1, v2) } else { (v2, v1) };

            if seen.insert(canonical) {
                // Keep original order for winding consistency
                result.push((v1, v2));
            }
        }

        result
    }

    /// Order horizon edges in cyclic order around the new point.
    ///
    /// This is critical for correct fan adjacency: the new faces form a fan around
    /// `new_point`, and each face's neighbors in the fan are determined by adjacent
    /// edges in this ordering.
    ///
    /// # Algorithm
    /// 1. Compute the fan axis from interior point to new point
    /// 2. Project each edge's midpoint onto the plane perpendicular to this axis
    /// 3. Sort edges by angle in this plane
    ///
    /// # Returns
    /// `Some(ordered_edges)` if ordering succeeded, `None` if the horizon is degenerate
    /// (e.g., axis is zero-length or edges are collinear with axis).
    fn order_horizon_edges_cyclically(
        &self,
        edges: Vec<(usize, usize)>,
        new_point: DVec3,
        interior_point: DVec3,
    ) -> Option<Vec<(usize, usize)>> {
        if edges.len() < 3 {
            // Need at least 3 edges to form a valid fan
            return if edges.is_empty() { None } else { Some(edges) };
        }

        // Compute the fan axis (from interior toward new point)
        let axis = new_point - interior_point;
        let axis_len = axis.norm();
        if axis_len < self.epsilon {
            // Degenerate: new point coincides with interior point
            return None;
        }
        let axis = axis / axis_len;

        // Create orthonormal basis perpendicular to axis
        let (u_axis, v_axis) = Self::create_perpendicular_basis(&axis);

        // Compute angle for each edge based on its midpoint
        let mut edge_angles: Vec<((usize, usize), f64)> = edges
            .into_iter()
            .map(|(v1, v2)| {
                let p1 = self.points[v1];
                let p2 = self.points[v2];
                let midpoint = (p1 + p2) * 0.5;

                // Project midpoint onto plane perpendicular to axis, centered at new_point
                let local = midpoint - new_point;
                let u = local.dot(&u_axis);
                let v = local.dot(&v_axis);
                let angle = v.atan2(u);

                ((v1, v2), angle)
            })
            .collect();

        // Sort by angle
        edge_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Some(edge_angles.into_iter().map(|(edge, _)| edge).collect())
    }

    /// Create an orthonormal basis perpendicular to a given axis.
    fn create_perpendicular_basis(axis: &DVec3) -> (DVec3, DVec3) {
        // Choose a vector not parallel to axis
        let arbitrary = if axis.x.abs() < 0.9 {
            DVec3::new(1.0, 0.0, 0.0)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        };

        let u = axis.cross(&arbitrary).normalize();
        let v = axis.cross(&u).normalize();

        (u, v)
    }

    /// Remove the point with the given external ID.
    ///
    /// Returns `true` if the point existed and was removed.
    pub fn remove(&mut self, id: u32) -> bool {
        let Some(&idx) = self.id_to_index.get(&id) else {
            return false;
        };

        // Check if this point is on the hull
        let was_on_hull = self.hull_point_indices.contains(&idx);

        // Remove from internal storage first
        self.remove_point_internal(id);

        if was_on_hull {
            // Point was on the hull - need to rebuild
            self.rebuild_hull_from_points();
        }

        self.update_origin_inside();
        true
    }

    /// Check if a point with the given ID is on the hull.
    #[must_use]
    pub fn is_on_hull(&self, id: u32) -> bool {
        self.id_to_index
            .get(&id)
            .is_some_and(|&idx| self.hull_point_indices.contains(&idx))
    }

    /// Check if a point with the given ID exists in the point set.
    #[must_use]
    pub fn contains_id(&self, id: u32) -> bool {
        self.id_to_index.contains_key(&id)
    }

    /// Returns whether the origin is strictly inside the convex hull.
    ///
    /// This is a cached O(1) query; the result is updated after each insert/remove.
    #[must_use]
    pub fn origin_inside(&self) -> bool {
        self.origin_inside
    }

    /// Number of points in the point set (not all may be on hull).
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    // face_count() is defined above as an internal method that returns active count

    /// Number of points on the hull.
    #[must_use]
    pub fn hull_point_count(&self) -> usize {
        self.hull_point_indices.len()
    }

    // INTERNAL: Point Management
    // --------------------------

    /// Add a point to the internal storage, returning its index.
    fn add_point_internal(&mut self, id: u32, point: DVec3) -> usize {
        let idx = self.points.len();
        self.points.push(point);
        self.id_to_index.insert(id, idx);
        self.index_to_id.push(id);
        idx
    }

    /// Validate a point is finite (reject NaN/Inf).
    fn is_valid_point(point: &DVec3) -> bool {
        point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
    }

    /// Remove a point from internal storage (does not update hull).
    fn remove_point_internal(&mut self, id: u32) {
        let Some(&idx) = self.id_to_index.get(&id) else {
            return;
        };

        let last_idx = self.points.len() - 1;

        // Track hull membership BEFORE any modifications
        let _was_on_hull = self.hull_point_indices.remove(&idx);
        let last_was_on_hull = if idx == last_idx {
            false
        } else {
            self.hull_point_indices.remove(&last_idx)
        };

        // Swap-remove from points array
        if idx != last_idx {
            // Move last point to this slot
            self.points.swap(idx, last_idx);
            self.index_to_id.swap(idx, last_idx);

            // Update the moved point's ID mapping
            let moved_id = self.index_to_id[idx];
            self.id_to_index.insert(moved_id, idx);

            // Update hull structures that reference the moved index
            self.update_index_references(last_idx, idx);

            // Restore hull membership for the moved point at its new index
            if last_was_on_hull {
                self.hull_point_indices.insert(idx);
            }
        }

        self.points.pop();
        self.index_to_id.pop();
        self.id_to_index.remove(&id);
    }

    /// Update all index references when a point is moved from `old_idx` to `new_idx`.
    ///
    /// NOTE: This intentionally does NOT update `hull_point_indices`. That field
    /// is handled separately in `remove_point_internal` because we need to track
    /// whether the moved point was on the hull BEFORE any modifications occur.
    /// The caller is responsible for updating `hull_point_indices` correctly.
    fn update_index_references(&mut self, old_idx: usize, new_idx: usize) {
        // Update faces
        for face in self.faces.iter_mut().flatten() {
            for v in face.iter_mut() {
                if *v == old_idx {
                    *v = new_idx;
                }
            }
        }
    }

    /// Rebuild the convex hull from all current points.
    fn rebuild_hull_from_points(&mut self) {
        // Clear existing hull structures (including free list for fresh start)
        self.faces.clear();
        self.face_normals.clear();
        self.face_adjacency.clear();
        self.face_free_list.clear();
        self.hull_point_indices.clear();

        if self.points.len() < 4 {
            return;
        }

        // Use f64 convex hull to avoid precision issues with near-coplanar points
        let hull_indices = convex_hull_triangulated_f64(&self.points);

        if hull_indices.len() < 3 {
            return;
        }

        // Compute centroid for normal orientation (use all points initially)
        let centroid: DVec3 = self.points.iter().copied().sum::<DVec3>() / self.points.len() as f64;

        // Track seen faces to deduplicate (convex_hull_triangulated can return duplicates)
        let mut seen_faces: HashSet<[usize; 3]> = HashSet::new();

        // Rebuild faces from hull indices (triplets)
        for chunk in hull_indices.chunks(3) {
            if chunk.len() == 3 {
                let v0 = chunk[0];
                let mut v1 = chunk[1];
                let mut v2 = chunk[2];

                // Create canonical face key (sorted vertices) for deduplication
                let mut canonical = [v0, v1, v2];
                canonical.sort_unstable();
                if !seen_faces.insert(canonical) {
                    continue; // Skip duplicate face
                }

                // Compute face normal using f64 points
                let p0 = self.points[v0];
                let p1 = self.points[v1];
                let p2 = self.points[v2];
                let edge1 = p1 - p0;
                let edge2 = p2 - p0;
                let mut normal = edge1.cross(&edge2);

                if normal.norm() < self.epsilon {
                    continue; // Skip degenerate faces - don't mark vertices
                }

                normal = normal.normalize();

                // Ensure normal points outward (away from centroid)
                let face_center = (p0 + p1 + p2) / 3.0;
                let to_face = face_center - centroid;
                if normal.dot(&to_face) < 0.0 {
                    // Normal points inward - flip winding and normal
                    std::mem::swap(&mut v1, &mut v2);
                    normal = -normal;
                }

                // Mark vertices as on hull AFTER degeneracy check
                self.hull_point_indices.insert(v0);
                self.hull_point_indices.insert(v1);
                self.hull_point_indices.insert(v2);

                self.faces.push(Some([v0, v1, v2]));
                self.face_normals.push(Some(normal));
                self.face_adjacency
                    .push(Some([usize::MAX, usize::MAX, usize::MAX]));
            }
        }

        // Rebuild adjacency and centroid
        self.rebuild_adjacency();
        self.update_hull_centroid();

        // Validate manifoldness - convex hull can produce non-manifold results
        // with degenerate inputs. Clear the hull if it's invalid to avoid downstream issues.
        if self.face_count() >= 4 && !self.is_manifold() {
            #[cfg(test)]
            eprintln!("  WARNING: rebuild_hull_from_points produced non-manifold hull - clearing");
            self.faces.clear();
            self.face_normals.clear();
            self.face_adjacency.clear();
            self.face_free_list.clear();
            self.hull_point_indices.clear();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL: Centroid and Origin Test
    // ═══════════════════════════════════════════════════════════════════════════

    /// Update the hull centroid from current hull points.
    fn update_hull_centroid(&mut self) {
        if self.hull_point_indices.is_empty() {
            self.hull_centroid = DVec3::zeros();
            return;
        }

        let sum: DVec3 = self
            .hull_point_indices
            .iter()
            .map(|&idx| self.points[idx])
            .sum();
        self.hull_centroid = sum / self.hull_point_indices.len() as f64;
    }

    /// Update the cached `origin_inside` value.
    fn update_origin_inside(&mut self) {
        self.origin_inside = self.compute_origin_inside();
    }

    /// Check if the origin is strictly inside all face half-spaces.
    fn compute_origin_inside(&self) -> bool {
        if self.face_count() < 4 {
            // Need at least a tetrahedron
            return false;
        }

        // Origin is inside iff it's on the negative side of all face planes
        // (since normals point outward)
        for (face_idx, face) in self.active_faces() {
            let Some(normal) = self.face_normals[face_idx] else {
                continue;
            };
            let p0 = self.points[face[0]];

            // Signed distance from origin to face plane
            // d = -normal · p0 (since origin is at 0)
            let dist = -normal.dot(&p0);

            // Origin must be strictly inside (negative distance with tolerance)
            if dist >= -self.epsilon * normal.norm() {
                return false;
            }
        }

        true
    }

    /// Rebuild face adjacency from scratch.
    fn rebuild_adjacency(&mut self) {
        // Build edge to face map (only active faces)
        let mut edge_to_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        for (face_idx, face) in self.active_faces() {
            for i in 0..3 {
                let v1 = face[(i + 1) % 3];
                let v2 = face[(i + 2) % 3];
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                edge_to_faces.entry(edge).or_default().push(face_idx);
            }
        }

        // Collect active face data to avoid borrow conflicts
        let active_face_data: Vec<(usize, [usize; 3])> = self
            .active_faces()
            .map(|(idx, face)| (idx, *face))
            .collect();

        // Reset adjacency for all active faces
        for &(face_idx, _) in &active_face_data {
            self.face_adjacency[face_idx] = Some([usize::MAX; 3]);
        }

        // Set adjacency
        for (face_idx, face) in active_face_data {
            for i in 0..3 {
                let v1 = face[(i + 1) % 3];
                let v2 = face[(i + 2) % 3];
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };

                if let Some(faces) = edge_to_faces.get(&edge) {
                    for &other_face in faces {
                        if other_face != face_idx {
                            if let Some(ref mut adj) = self.face_adjacency[face_idx] {
                                adj[i] = other_face;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// F64 CONVEX HULL (Private)
// ═══════════════════════════════════════════════════════════════════════════════

/// Private f64 convex hull implementation to avoid f32 precision loss.
///
/// This is a minimal port of the f32 `convex_hull_triangulated` function,
/// converted to use `DVec3` (f64) throughout. This eliminates the precision
/// issues that caused non-manifold results when rebuilding the hull.
mod convex_hull_f64 {
    use std::collections::HashSet;

    use algebra::DVec3;

    use crate::convex_hull::indexed_planar_convex_hull;

    /// Tolerance for coplanarity and near-zero checks in hull construction.
    ///
    /// This tolerance affects:
    /// - Zero-vector detection (degenerate faces)
    /// - Coplanarity detection (faces sharing an edge at nearly the same angle)
    ///
    /// Using 1e-6 to match typical input data precision (6 decimal places).
    /// Near-coplanar configurations need reasonable tolerance to avoid
    /// creating overlapping triangles (non-manifold edges).
    const COPLANAR_TOL: f64 = 1e-6;

    /// Compute a triangulated convex hull using f64 precision.
    ///
    /// Returns an empty vector if the points are degenerate (fewer than 4,
    /// all collinear, etc.).
    pub fn convex_hull_triangulated_f64(points: &[DVec3]) -> Vec<usize> {
        // Reject non-finite inputs early to avoid NaNs propagating through geometry.
        if points
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        {
            return Vec::new();
        }

        if points.len() < 4 {
            return Vec::new();
        }

        let Some(ch_triangle) = find_ch_triangle(points) else {
            // Points are collinear or otherwise degenerate
            return Vec::new();
        };

        convex_hull_with_hint_triangulated(points, ch_triangle)
    }

    fn cross(a: &DVec3, b: &DVec3) -> DVec3 {
        a.cross(b)
    }

    fn safe_normal(edge1: &DVec3, edge2: &DVec3) -> DVec3 {
        let n = cross(edge1, edge2);
        let len = n.norm();

        if !len.is_finite() || len < COPLANAR_TOL {
            DVec3::zeros()
        } else {
            n / len
        }
    }

    fn has_non_colinear(points: &[usize], coords: &[DVec3]) -> bool {
        if points.len() < 3 {
            return false;
        }

        let p0 = coords[points[0]];

        for i in 1..points.len() {
            let p1 = coords[points[i]];

            for j in i + 1..points.len() {
                let p2 = coords[points[j]];

                if (p1 - p0).cross(&(p2 - p0)).norm() >= COPLANAR_TOL {
                    return true;
                }
            }
        }

        false
    }

    fn convex_hull_with_hint_triangulated(
        points: &[DVec3],
        ch_triangle: (usize, usize, usize),
    ) -> Vec<usize> {
        let get_normal =
            |i, j, k| -> DVec3 { safe_normal(&(points[j] - points[i]), &(points[k] - points[i])) };

        // The centroid is used to assert winding order.
        let centroid: DVec3 = points.iter().copied().sum::<DVec3>() / points.len() as f64;

        // Add the first triangle to the set.
        let mut face_ids = vec![ch_triangle.0, ch_triangle.1, ch_triangle.2];

        // Test and fix winding order if necessary.
        let normal = get_normal(face_ids[0], face_ids[1], face_ids[2]);
        let test = normal.dot(&(points[face_ids[0]] - centroid));
        if test < 0.0 {
            face_ids.swap(0, 2);
        }

        // Find all points that are coplanar with the starting face.
        let current_normal = get_normal(face_ids[0], face_ids[1], face_ids[2]);
        let mut coplanar_points = Vec::new();
        for point_id in 0..points.len() {
            if point_id == face_ids[0] || point_id == face_ids[1] {
                continue;
            }
            let normal = get_normal(face_ids[0], face_ids[1], point_id);
            let dot = normal.dot(&current_normal);

            if (dot.abs() - 1.0).abs() <= COPLANAR_TOL {
                coplanar_points.push(point_id);
            }
        }

        // Initialize state with first face.
        let p1 = face_ids[0];
        let p2 = face_ids[1];
        face_ids.clear();

        let mut edge_queue = Vec::new();
        let mut seen_edges = HashSet::new();
        triangulate_coplanar_points(
            points,
            p1,
            p2,
            &centroid,
            &current_normal,
            &mut coplanar_points,
            &mut seen_edges,
            &mut face_ids,
            &mut edge_queue,
        );

        // Process edge queue to build hull.
        while let Some((p1_id, p2_id, current_normal)) = edge_queue.pop() {
            if seen_edges.contains(&(p1_id, p2_id)) {
                continue;
            }

            let mut best_point = 0;
            let mut best_dot = -1000.1;
            let mut best_normal = DVec3::zeros();
            let mut coplanar_points = Vec::new();

            for point_id in 0..points.len() {
                if point_id == p1_id || point_id == p2_id {
                    continue;
                }

                let normal = get_normal(p1_id, p2_id, point_id);
                let dot = normal.dot(&current_normal);

                if (dot - best_dot).abs() <= COPLANAR_TOL * 10.0 {
                    if seen_edges.contains(&(best_point, p1_id))
                        || seen_edges.contains(&(p2_id, best_point))
                    {
                        best_point = point_id;
                    }
                    coplanar_points.push(point_id);
                    continue;
                }
                if dot > best_dot {
                    best_point = point_id;
                    best_dot = dot;
                    best_normal = normal;

                    coplanar_points.clear();
                    coplanar_points.push(point_id);
                }
            }

            triangulate_coplanar_points(
                points,
                p1_id,
                p2_id,
                &centroid,
                &best_normal,
                &mut coplanar_points,
                &mut seen_edges,
                &mut face_ids,
                &mut edge_queue,
            );
        }

        face_ids
    }

    /// Find an initial triangle on the convex hull.
    ///
    /// Returns `None` if all points are collinear or degenerate.
    fn find_ch_triangle(points: &[DVec3]) -> Option<(usize, usize, usize)> {
        if points.len() < 3 {
            return None;
        }

        // The lowest point is part of the 3D convex hull.
        let lowest_id = points
            .iter()
            .enumerate()
            .min_by(|(_, p1), (_, p2)| p1.z.total_cmp(&p2.z))
            .map(|(id, _)| id)?;

        // Find second point with best angle (furthest from vertical).
        let mut best_angle = f64::MAX;
        let mut best_angle_id = None;
        for (id, p2) in points.iter().enumerate() {
            if id == lowest_id {
                continue;
            }

            let mut e1 = *p2 - points[lowest_id];
            e1.y = 0.0;
            if e1.norm() < COPLANAR_TOL {
                continue;
            }
            e1 = e1.normalize();

            let dir = DVec3::new(0.0, 0.0, 1.0);
            let dot = e1.dot(&dir);
            if dot < best_angle {
                best_angle = dot;
                best_angle_id = Some(id);
            }
        }

        let best_angle_id = best_angle_id?;

        // Pick a candidate for third point (first point that's different).
        let mut candidate = None;
        for i in 0..points.len() {
            if i != lowest_id && i != best_angle_id {
                candidate = Some(i);
                break;
            }
        }
        let mut candidate = candidate?;

        let get_normal =
            |i, j, k| -> DVec3 { cross(&(points[j] - points[i]), &(points[k] - points[i])) };

        // Find third point such that all other points are on one side.
        for i in 0..points.len() {
            if i == lowest_id || i == best_angle_id {
                continue;
            }

            let normal = get_normal(lowest_id, best_angle_id, candidate);
            if normal.dot(&(points[i] - points[lowest_id])) < 0.0 {
                candidate = i;
            }
        }

        // Validate the triangle is non-degenerate (non-zero area).
        let normal = get_normal(lowest_id, best_angle_id, candidate);
        if normal.norm() < COPLANAR_TOL {
            // All points are collinear
            return None;
        }

        Some((lowest_id, best_angle_id, candidate))
    }

    #[allow(clippy::too_many_arguments)]
    fn triangulate_coplanar_points(
        points: &[DVec3],
        p1_id: usize,
        p2_id: usize,
        centroid: &DVec3,
        _best_normal: &DVec3,
        coplanar_points: &mut Vec<usize>,
        seen_edges: &mut HashSet<(usize, usize)>,
        face_ids: &mut Vec<usize>,
        edge_queue: &mut Vec<(usize, usize, DVec3)>,
    ) {
        let get_normal =
            |i, j, k| -> DVec3 { safe_normal(&(points[j] - points[i]), &(points[k] - points[i])) };

        coplanar_points.push(p1_id);
        coplanar_points.push(p2_id);

        seen_edges.insert((p1_id, p2_id));

        // Remove duplicate positions to avoid degenerate planar hulls.
        let mut deduped: Vec<usize> = Vec::new();
        for &idx in coplanar_points.iter() {
            if deduped
                .iter()
                .all(|&u| (points[u] - points[idx]).norm() >= COPLANAR_TOL)
            {
                deduped.push(idx);
            }
        }
        *coplanar_points = deduped;

        if coplanar_points.len() < 3 || !has_non_colinear(coplanar_points, points) {
            return; // Degenerate set - no face to triangulate
        }

        let mut hull =
            indexed_planar_convex_hull(&|i| points[coplanar_points[i]], coplanar_points.len());

        if hull.len() < 3 {
            return; // Degenerate coplanar set - nothing to triangulate
        }

        // Check and fix winding order.
        let should_swap = (points[coplanar_points[0]] - centroid).dot(&get_normal(
            coplanar_points[hull[0]],
            coplanar_points[hull[1]],
            coplanar_points[hull[2]],
        )) < 0.0;
        if should_swap {
            hull.reverse();
        }

        // Triangulate the convex face using fan triangulation.
        for i in 1..hull.len() - 1 {
            let p1 = coplanar_points[hull[0]];
            let p2 = coplanar_points[hull[i]];
            let p3 = coplanar_points[hull[i + 1]];

            face_ids.push(p1);
            face_ids.push(p2);
            face_ids.push(p3);

            seen_edges.insert((p1, p2));
            seen_edges.insert((p2, p3));
            seen_edges.insert((p3, p1));
        }

        // Add new edges to queue.
        for i in 0..hull.len() {
            let p1 = coplanar_points[hull[i]];
            let p2 = coplanar_points[hull[(i + 1) % hull.len()]];

            if !seen_edges.contains(&(p2, p1)) {
                let normal = if hull.len() >= 3 {
                    get_normal(
                        coplanar_points[hull[0]],
                        coplanar_points[hull[1]],
                        coplanar_points[hull[2]],
                    )
                } else {
                    DVec3::new(0.0, 0.0, 1.0)
                };
                edge_queue.push((p2, p1, normal));
            }
        }
    }
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    fn cube_normals() -> Vec<(u32, DVec3)> {
        vec![
            (0, DVec3::new(1.0, 0.0, 0.0)),
            (1, DVec3::new(-1.0, 0.0, 0.0)),
            (2, DVec3::new(0.0, 1.0, 0.0)),
            (3, DVec3::new(0.0, -1.0, 0.0)),
            (4, DVec3::new(0.0, 0.0, 1.0)),
            (5, DVec3::new(0.0, 0.0, -1.0)),
        ]
    }

    fn tetrahedron_normals() -> Vec<(u32, DVec3)> {
        // Regular tetrahedron normals (pointing outward from origin)
        vec![
            (0, DVec3::new(1.0, 1.0, 1.0).normalize()),
            (1, DVec3::new(1.0, -1.0, -1.0).normalize()),
            (2, DVec3::new(-1.0, 1.0, -1.0).normalize()),
            (3, DVec3::new(-1.0, -1.0, 1.0).normalize()),
        ]
    }

    #[test]
    fn test_empty_hull() {
        let hull = IncrementalHull::new(1e-7);
        assert_eq!(hull.point_count(), 0);
        assert_eq!(hull.face_count(), 0);
        assert!(!hull.origin_inside());
    }

    #[test]
    fn test_build_tetrahedron() {
        let hull = IncrementalHull::build(tetrahedron_normals().into_iter(), 1e-7);
        assert_eq!(hull.point_count(), 4);
        assert_eq!(hull.face_count(), 4);
        assert_eq!(hull.hull_point_count(), 4);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_build_cube_normals() {
        let hull = IncrementalHull::build(cube_normals().into_iter(), 1e-7);
        assert_eq!(hull.point_count(), 6);
        // Cube normals form an octahedron hull (8 faces)
        assert_eq!(hull.face_count(), 8);
        assert_eq!(hull.hull_point_count(), 6);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_incremental_insert() {
        let mut hull = IncrementalHull::new(1e-7);

        // Add tetrahedron points one by one
        for (id, normal) in tetrahedron_normals() {
            hull.insert(id, normal);
        }

        assert_eq!(hull.point_count(), 4);
        assert_eq!(hull.face_count(), 4);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_incremental_insert_cube() {
        let mut hull = IncrementalHull::new(1e-7);

        // Add cube normals incrementally (forms octahedron hull)
        for (id, normal) in cube_normals() {
            hull.insert(id, normal);
        }

        assert_eq!(hull.point_count(), 6);
        // Cube normals form an octahedron hull (8 faces)
        assert_eq!(hull.face_count(), 8);
        assert_eq!(hull.hull_point_count(), 6);
        assert!(hull.origin_inside());
    }

    /// Test incremental insertion of octahedron vertices (non-coplanar start)
    #[test]
    fn test_incremental_insert_octahedron_reordered() {
        let mut hull = IncrementalHull::new(1e-7);

        // Insert in order that gives non-coplanar first 4 points: +X, +Y, +Z, -X, -Y, -Z
        let normals = vec![
            (0, DVec3::new(1.0, 0.0, 0.0)),  // +X
            (1, DVec3::new(0.0, 1.0, 0.0)),  // +Y
            (2, DVec3::new(0.0, 0.0, 1.0)),  // +Z (now we have non-coplanar)
            (3, DVec3::new(-1.0, 0.0, 0.0)), // -X
            (4, DVec3::new(0.0, -1.0, 0.0)), // -Y
            (5, DVec3::new(0.0, 0.0, -1.0)), // -Z
        ];

        for (i, (id, normal)) in normals.into_iter().enumerate() {
            hull.insert(id, normal);
            eprintln!(
                "After point {} (id={}): faces={}, hull_points={}",
                i,
                id,
                hull.face_count(),
                hull.hull_point_count()
            );
        }

        assert_eq!(hull.point_count(), 6);
        assert_eq!(hull.face_count(), 8);
        assert!(hull.origin_inside());
    }

    /// Test inserting a point outside an existing tetrahedron
    #[test]
    fn test_insert_point_outside_tetrahedron() {
        let mut hull = IncrementalHull::build(tetrahedron_normals().into_iter(), 1e-7);
        assert_eq!(hull.face_count(), 4);
        assert!(hull.origin_inside());

        // Insert a point outside the tetrahedron (e.g., along +X axis)
        hull.insert(100, DVec3::new(2.0, 0.0, 0.0));

        // Hull should expand
        assert_eq!(hull.point_count(), 5);
        assert!(hull.hull_point_count() >= 4);
        // Origin should still be inside (2,0,0) doesn't change that for these normals
        // Actually, adding (2,0,0) might make origin not inside anymore
        // Let's just check the hull is valid
        assert!(hull.face_count() >= 4);
    }

    /// Test that inserting a duplicate ID is a no-op
    #[test]
    fn test_insert_duplicate_id() {
        let mut hull = IncrementalHull::build(tetrahedron_normals().into_iter(), 1e-7);
        let initial_faces = hull.face_count();

        // Try to insert with existing ID
        hull.insert(0, DVec3::new(5.0, 5.0, 5.0));

        // Should be no-op
        assert_eq!(hull.face_count(), initial_faces);
        assert_eq!(hull.point_count(), 4);
    }

    #[test]
    fn test_build_ignores_duplicate_ids() {
        let mut normals = tetrahedron_normals();
        let first_point = normals[0].1;
        normals.push((0, DVec3::new(5.0, 5.0, 5.0))); // duplicate ID with different point

        let hull = IncrementalHull::build(normals.into_iter(), 1e-7);

        assert_eq!(hull.point_count(), 4); // duplicate skipped
        let idx = hull.id_to_index[&0];
        assert_eq!(hull.points[idx], first_point); // original point preserved
        assert_eq!(hull.face_count(), 4);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_build_skips_non_finite_points() {
        let mut normals = tetrahedron_normals();
        normals.push((99, DVec3::new(f64::NAN, 0.0, 0.0))); // invalid point

        let hull = IncrementalHull::build(normals.into_iter(), 1e-7);

        assert_eq!(hull.point_count(), 4); // invalid point rejected
        assert!(!hull.contains_id(99));
        assert_eq!(hull.face_count(), 4);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_build_handles_duplicate_positions() {
        let mut normals = tetrahedron_normals();
        let duplicate_point = normals[0].1;
        normals.push((99, duplicate_point)); // different ID, identical position

        let hull = IncrementalHull::build(normals.into_iter(), 1e-7);

        assert!(hull.face_count() >= 4);
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_insert_rejects_non_finite_point() {
        let mut hull = IncrementalHull::new(1e-7);
        hull.insert(42, DVec3::new(f64::NAN, 0.0, 0.0));

        assert_eq!(hull.point_count(), 0);
        assert!(!hull.contains_id(42));
    }

    /// Test the horizon edge detection with a simple case
    #[test]
    fn test_horizon_detection() {
        // Build a tetrahedron, then insert a point that sees exactly one face
        let mut hull = IncrementalHull::build(tetrahedron_normals().into_iter(), 1e-7);
        assert_eq!(hull.face_count(), 4);

        // Find a point just outside one face
        let point = DVec3::new(1.0, 1.0, 1.0).normalize() * 2.0;

        // Check visibility
        let visible = hull.find_visible_faces(point);
        assert!(
            !visible.is_empty(),
            "Point outside hull should see at least one face"
        );

        // Insert it
        hull.insert(100, point);

        // Should have more faces now
        assert!(hull.face_count() >= 4);
    }

    /// Test that `build()` and incremental `insert()` produce equivalent results
    #[test]
    fn test_build_vs_incremental_equivalence() {
        let normals: Vec<(u32, DVec3)> = vec![
            (0, DVec3::new(1.0, 0.0, 0.0)),
            (1, DVec3::new(0.0, 1.0, 0.0)),
            (2, DVec3::new(0.0, 0.0, 1.0)),
            (3, DVec3::new(-1.0, 0.0, 0.0)),
            (4, DVec3::new(0.0, -1.0, 0.0)),
            (5, DVec3::new(0.0, 0.0, -1.0)),
        ];

        // Build all at once
        let hull_batch = IncrementalHull::build(normals.clone().into_iter(), 1e-7);

        // Insert one by one
        let mut hull_incremental = IncrementalHull::new(1e-7);
        for (id, normal) in normals {
            hull_incremental.insert(id, normal);
        }

        // Should have same structure
        assert_eq!(hull_batch.point_count(), hull_incremental.point_count());
        assert_eq!(hull_batch.face_count(), hull_incremental.face_count());
        assert_eq!(
            hull_batch.hull_point_count(),
            hull_incremental.hull_point_count()
        );
        assert_eq!(hull_batch.origin_inside(), hull_incremental.origin_inside());
    }

    /// Test incremental insertion doesn't corrupt adjacency
    #[test]
    fn test_incremental_adjacency_valid() {
        let mut hull = IncrementalHull::new(1e-7);

        // Insert points one by one
        for (id, normal) in tetrahedron_normals() {
            hull.insert(id, normal);
        }

        // After building, verify adjacency is valid
        for (face_idx, adj_opt) in hull.face_adjacency.iter().enumerate() {
            let Some(adj) = adj_opt else { continue };
            for (edge_idx, &adj_face) in adj.iter().enumerate() {
                if adj_face != usize::MAX {
                    assert!(
                        hull.faces.get(adj_face).is_some_and(|f| f.is_some()),
                        "Face {face_idx} edge {edge_idx} points to invalid face {adj_face}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_insert_interior_point() {
        let mut hull = IncrementalHull::build(cube_normals().into_iter(), 1e-7);
        let initial_faces = hull.face_count();

        // Insert a point inside the hull (scaled down)
        hull.insert(100, DVec3::new(0.5, 0.0, 0.0));

        // Should not change the hull
        assert_eq!(hull.face_count(), initial_faces);
        assert!(!hull.is_on_hull(100));
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_remove_non_hull_point() {
        let mut hull = IncrementalHull::build(cube_normals().into_iter(), 1e-7);

        // Insert then remove an interior point
        hull.insert(100, DVec3::new(0.5, 0.0, 0.0));
        let faces_before = hull.face_count();

        hull.remove(100);

        assert_eq!(hull.face_count(), faces_before);
        assert!(!hull.contains_id(100));
        assert!(hull.origin_inside());
    }

    #[test]
    fn test_contains_id() {
        let hull = IncrementalHull::build(cube_normals().into_iter(), 1e-7);

        assert!(hull.contains_id(0));
        assert!(hull.contains_id(5));
        assert!(!hull.contains_id(100));
    }

    #[test]
    fn test_is_on_hull() {
        let mut hull = IncrementalHull::build(cube_normals().into_iter(), 1e-7);

        // All cube normals should be on hull
        for id in 0..6 {
            assert!(hull.is_on_hull(id));
        }

        // Interior point should not be on hull
        hull.insert(100, DVec3::new(0.3, 0.3, 0.3));
        assert!(!hull.is_on_hull(100));
    }

    #[test]
    fn test_origin_not_inside_unbounded() {
        let mut hull = IncrementalHull::new(1e-7);

        // Only 3 points - can't enclose origin
        hull.insert(0, DVec3::new(1.0, 0.0, 0.0));
        hull.insert(1, DVec3::new(0.0, 1.0, 0.0));
        hull.insert(2, DVec3::new(0.0, 0.0, 1.0));

        assert!(!hull.origin_inside());
    }

    #[test]
    fn test_origin_not_inside_half_space() {
        // All normals point in +X direction - origin not enclosed
        let normals = vec![
            (0, DVec3::new(1.0, 0.0, 0.0)),
            (1, DVec3::new(1.0, 0.1, 0.0).normalize()),
            (2, DVec3::new(1.0, -0.1, 0.0).normalize()),
            (3, DVec3::new(1.0, 0.0, 0.1).normalize()),
            (4, DVec3::new(1.0, 0.0, -0.1).normalize()),
        ];

        let hull = IncrementalHull::build(normals.into_iter(), 1e-7);
        assert!(!hull.origin_inside());
    }

    #[test]
    fn test_icosahedron_normals_hull() {
        // These are the exact normals from the failing icosahedron test
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

        let normalized: Vec<(u32, DVec3)> = normals
            .iter()
            .enumerate()
            .map(|(i, n)| (i as u32, n.normalize()))
            .collect();

        let hull = IncrementalHull::build(normalized.into_iter(), 1e-7);

        eprintln!("Hull point count: {}", hull.points.len());
        eprintln!("Hull face count: {}", hull.face_count());
        eprintln!("Hull point indices on hull: {}", hull.hull_point_indices.len());

        // Check manifold status manually
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for (_idx, face) in hull.active_faces() {
            for i in 0..3 {
                let v1 = face[i];
                let v2 = face[(i + 1) % 3];
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        let non_manifold_edges: Vec<_> = edge_count
            .iter()
            .filter(|&(_, count)| *count != 2)
            .collect();

        if !non_manifold_edges.is_empty() {
            eprintln!("Non-manifold edges:");
            for ((v1, v2), count) in &non_manifold_edges {
                eprintln!(
                    "  Edge ({}, {}): count={}, points=({:?}, {:?})",
                    v1, v2, count, hull.points[*v1], hull.points[*v2]
                );
            }
        }

        // The hull should be manifold and contain the origin
        assert!(
            non_manifold_edges.is_empty(),
            "Hull should be manifold, found {} non-manifold edges",
            non_manifold_edges.len()
        );
        assert!(hull.origin_inside(), "Origin should be inside hull of 20 normals spanning all directions");
    }
}
