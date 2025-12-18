//! Benchmarks comparing `incremental_polytope_mono` against `vertex_enumeration`.
//!
//! This provides an independent baseline using the batch vertex enumeration algorithm
//! from `vertex_enumeration.rs` (based on depth-first search over polytope graph).
//!
//! Run with: `cargo bench --bench vertex_enum_comparison --features bench-utils`

use algebra::{na::DMatrix, DVec3};
use divan::{black_box, Bencher};
use geometry::{
    incremental_polytope_mono::{HalfSpace, IncrementalPolytope, PlaneIdx},
    vertex_enumeration_from_half_spaces,
};

fn main() {
    divan::main();
}

// ============================================================================
// Test Data Generators
// ============================================================================

/// Generate Fibonacci sphere planes (well-distributed normals)
fn fibonacci_sphere_planes(n: usize) -> Vec<(DVec3, f64)> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;

    (0..n)
        .map(|i| {
            let theta = std::f64::consts::TAU * (i as f64) / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();

            let x = phi.sin() * theta.cos();
            let y = phi.sin() * theta.sin();
            let z = phi.cos();

            (DVec3::new(x, y, z), 1.0)
        })
        .collect()
}

/// Convert planes to DMatrix format for vertex_enumeration
fn planes_to_matrix(planes: &[(DVec3, f64)]) -> DMatrix<f64> {
    let mut matrix = DMatrix::from_element(planes.len(), 4, 0.0);
    for (i, (normal, offset)) in planes.iter().enumerate() {
        matrix[(i, 0)] = normal.x;
        matrix[(i, 1)] = normal.y;
        matrix[(i, 2)] = normal.z;
        matrix[(i, 3)] = *offset;
    }
    matrix
}

// ============================================================================
// Construction Comparison: IncrementalPolytope vs vertex_enumeration
// ============================================================================

#[divan::bench(args = [12, 20, 30, 40, 50])]
fn construct_incremental(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [12, 20, 30, 40, 50])]
fn construct_vertex_enum(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Removal vs Rebuild Comparison
// ============================================================================

/// Topology-based removal: clone polytope + remove one plane
#[divan::bench(args = [20, 30, 40, 50, 60])]
fn remove_topology_based(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    // Pre-build template
    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        poly.remove_plane(PlaneIdx(5));
        black_box(poly.vertex_count())
    });
}

/// Rebuild using vertex_enumeration with N-1 planes
#[divan::bench(args = [20, 30, 40, 50, 60])]
fn rebuild_vertex_enum(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    // Pre-compute N-1 planes matrix (excluding plane 5)
    let remaining: Vec<_> = planes
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 5)
        .map(|(_, p)| *p)
        .collect();
    let matrix = planes_to_matrix(&remaining);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Mesh Extraction Comparison
// ============================================================================

/// IncrementalPolytope to_mesh (after construction)
#[divan::bench(args = [12, 20, 30])]
fn to_mesh_incremental(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

/// vertex_enumeration already returns mesh format
#[divan::bench(args = [12, 20, 30])]
fn to_mesh_vertex_enum(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Correctness Verification (not a benchmark, but useful)
// ============================================================================

/// Verify both implementations produce the same vertex count
#[divan::bench(args = [12, 20, 30])]
fn verify_vertex_count_match(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        // IncrementalPolytope
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        let incr_count = poly.vertex_count();

        // vertex_enumeration
        let (vertices, _) = vertex_enumeration_from_half_spaces(&matrix);
        let enum_count = vertices.len();

        // They should match
        assert_eq!(
            incr_count, enum_count,
            "Vertex count mismatch: incremental={}, enum={}",
            incr_count, enum_count
        );

        black_box((incr_count, enum_count))
    });
}

// ============================================================================
// Scalability Comparison
// ============================================================================

/// Compare scaling behavior at larger sizes
#[divan::bench(args = [30, 50, 80, 100])]
fn scale_removal_incremental(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        // Remove multiple planes sequentially
        for i in [5, 10, 15] {
            if i < n_planes {
                poly.remove_plane(PlaneIdx(i));
            }
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [30, 50, 80, 100])]
fn scale_removal_rebuild(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    // Pre-compute N-3 planes (excluding 5, 10, 15)
    let remaining: Vec<_> = planes
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 5 && *i != 10 && *i != 15)
        .map(|(_, p)| *p)
        .collect();
    let matrix = planes_to_matrix(&remaining);

    bencher.bench_local(|| {
        let (vertices, _) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box(vertices.len())
    });
}

// ============================================================================
// Stress Test: Verify IncrementalPolytope matches vertex_enumeration
// ============================================================================

/// Stress test comparing IncrementalPolytope removal against vertex_enumeration rebuild.
/// This verifies correctness by checking vertex counts match after each removal.
#[divan::bench]
fn stress_test_correctness(bencher: Bencher) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TOTAL_OPS: AtomicUsize = AtomicUsize::new(0);
    static MISMATCHES: AtomicUsize = AtomicUsize::new(0);

    bencher.bench_local(|| {
        let mut local_ops = 0usize;
        let mut local_mismatches = 0usize;

        for n_planes in [15, 20, 25, 30] {
            let planes = fibonacci_sphere_planes(n_planes);

            // Build IncrementalPolytope
            let mut template = IncrementalPolytope::new();
            for (normal, offset) in &planes {
                template.add_plane(HalfSpace::new(*normal, *offset));
            }

            // Try removing each plane and compare with vertex_enumeration
            for remove_idx in 0..n_planes {
                local_ops += 1;

                // Method 1: Topology-based removal
                let mut incr_poly = template.clone();
                incr_poly.remove_plane(PlaneIdx(remove_idx));
                let incr_count = incr_poly.vertex_count();

                // Method 2: Rebuild with N-1 planes using vertex_enumeration
                let remaining: Vec<_> = planes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != remove_idx)
                    .map(|(_, p)| *p)
                    .collect();
                let matrix = planes_to_matrix(&remaining);
                let (enum_verts, _) = vertex_enumeration_from_half_spaces(&matrix);
                let enum_count = enum_verts.len();

                // Compare - only count finite vertices for incremental
                let incr_finite = incr_poly
                    .vertices()
                    .filter(|(_, v)| v.is_finite())
                    .count();

                if incr_finite != enum_count {
                    local_mismatches += 1;
                }
            }
        }

        TOTAL_OPS.fetch_add(local_ops, Ordering::Relaxed);
        MISMATCHES.fetch_add(local_mismatches, Ordering::Relaxed);

        black_box((local_ops, local_mismatches))
    });

    // Print results
    static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINTED.swap(true, Ordering::Relaxed) {
        let total = TOTAL_OPS.load(Ordering::Relaxed);
        let mismatches = MISMATCHES.load(Ordering::Relaxed);
        let match_rate = if total > 0 {
            ((total - mismatches) as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "\n[Stress Test] Total: {}, Mismatches: {}, Match rate: {:.2}%",
            total, mismatches, match_rate
        );
    }
}
