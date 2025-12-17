//! Grid-based spatial hash for O(1) expected-time duplicate detection.
//!
//! Divides 3D space into a grid of cells. To check if a point is a duplicate:
//! 1. Compute which cell the point falls into
//! 2. Check that cell + 26 neighbors (3x3x3 cube)
//! 3. Compare distances only to points in those cells
//!
//! This avoids O(n) comparisons against all existing points.

use glam::DVec3;
use hashbrown::HashMap;

/// Grid-based spatial hash for efficient duplicate point detection.
pub struct SpatialHash {
    cells: HashMap<(i64, i64, i64), Vec<DVec3>>,
    cell_size: f64,
    tolerance: f64,
}

impl SpatialHash {
    /// Create a new spatial hash with the given tolerance.
    ///
    /// Points within `tolerance` distance of each other are considered duplicates.
    #[must_use]
    pub fn new(tolerance: f64) -> Self {
        // Cell size = 2x tolerance ensures duplicates are in adjacent cells
        Self {
            cells: HashMap::new(),
            cell_size: tolerance * 2.0,
            tolerance,
        }
    }

    /// Map a point to its grid cell indices.
    #[inline]
    fn cell_coords(&self, p: DVec3) -> (i64, i64, i64) {
        #[allow(clippy::cast_possible_truncation)]
        let discretize = |v: f64| (v / self.cell_size).floor() as i64;
        (discretize(p.x), discretize(p.y), discretize(p.z))
    }

    /// Check if the given point is within tolerance of any existing point.
    #[must_use]
    pub fn is_duplicate(&self, point: DVec3) -> bool {
        let (cx, cy, cz) = self.cell_coords(point);

        // Check 3x3x3 neighborhood
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

    /// Insert a point into the hash (does not check for duplicates).
    pub fn insert(&mut self, point: DVec3) {
        self.cells
            .entry(self.cell_coords(point))
            .or_default()
            .push(point);
    }

    /// Insert only if not a duplicate. Returns true if inserted.
    pub fn insert_if_unique(&mut self, point: DVec3) -> bool {
        if self.is_duplicate(point) {
            false
        } else {
            self.insert(point);
            true
        }
    }

    /// Clear all points from the hash.
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash_basic() {
        let mut hash = SpatialHash::new(1e-6);

        let p1 = DVec3::new(1.0, 2.0, 3.0);
        let p2 = DVec3::new(1.0 + 1e-7, 2.0, 3.0); // Within tolerance
        let p3 = DVec3::new(2.0, 2.0, 3.0); // Outside tolerance

        assert!(hash.insert_if_unique(p1));
        assert!(!hash.insert_if_unique(p2)); // Duplicate
        assert!(hash.insert_if_unique(p3)); // Not duplicate
    }

    #[test]
    fn test_spatial_hash_cell_boundary() {
        let mut hash = SpatialHash::new(0.1);

        // Points on opposite sides of a cell boundary but within tolerance
        let p1 = DVec3::new(0.199, 0.0, 0.0);
        let p2 = DVec3::new(0.201, 0.0, 0.0);

        assert!(hash.insert_if_unique(p1));
        assert!(!hash.insert_if_unique(p2)); // Should still be detected as duplicate
    }
}
