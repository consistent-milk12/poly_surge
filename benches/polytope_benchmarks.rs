//! Benchmarks for `poly_surge` incremental polytope operations.
//!
//! Run with: `cargo bench --bench polytope_benchmarks`
//!
//! These benchmarks test:
//! - Batch construction performance
//! - Incremental plane addition
//! - Mesh extraction (`to_mesh`)
//! - Plane removal performance
//! - Scalability with increasing plane counts

use divan::{Bencher, black_box};
use glam::DVec3;
use poly_surge::{HalfSpace, IncrementalPolytope, PlaneIdx};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    divan::main();
}

// ============================================================================
// Test Data Generators
// ============================================================================

/// Generate cube planes (6 planes)
fn cube_planes() -> Vec<(DVec3, f64)> {
    vec![
        (DVec3::new(1.0, 0.0, 0.0), 0.5),
        (DVec3::new(-1.0, 0.0, 0.0), 0.5),
        (DVec3::new(0.0, 1.0, 0.0), 0.5),
        (DVec3::new(0.0, -1.0, 0.0), 0.5),
        (DVec3::new(0.0, 0.0, 1.0), 0.5),
        (DVec3::new(0.0, 0.0, -1.0), 0.5),
    ]
}

/// Generate octahedron planes (8 planes with diagonal normals)
fn octahedron_planes() -> Vec<(DVec3, f64)> {
    let s = 1.0 / 3.0_f64.sqrt();
    vec![
        (DVec3::new(s, s, s), 1.0),
        (DVec3::new(s, s, -s), 1.0),
        (DVec3::new(s, -s, s), 1.0),
        (DVec3::new(s, -s, -s), 1.0),
        (DVec3::new(-s, s, s), 1.0),
        (DVec3::new(-s, s, -s), 1.0),
        (DVec3::new(-s, -s, s), 1.0),
        (DVec3::new(-s, -s, -s), 1.0),
    ]
}

/// Generate dodecahedron planes (12 planes)
fn dodecahedron_planes() -> Vec<(DVec3, f64)> {
    let phi = f64::midpoint(1.0, 5.0_f64.sqrt()); // Golden ratio
    let inv_phi = 1.0 / phi;

    let normals = [
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(1.0, 1.0, -1.0),
        DVec3::new(1.0, -1.0, 1.0),
        DVec3::new(1.0, -1.0, -1.0),
        DVec3::new(-1.0, 1.0, 1.0),
        DVec3::new(-1.0, 1.0, -1.0),
        DVec3::new(-1.0, -1.0, 1.0),
        DVec3::new(-1.0, -1.0, -1.0),
        DVec3::new(0.0, inv_phi, phi),
        DVec3::new(0.0, inv_phi, -phi),
        DVec3::new(0.0, -inv_phi, phi),
        DVec3::new(0.0, -inv_phi, -phi),
    ];

    normals.into_iter().map(|n| (n.normalize(), 1.5)).collect()
}

/// Generate icosahedron dual planes (20 planes)
fn icosahedron_planes() -> Vec<(DVec3, f64)> {
    let phi = f64::midpoint(1.0, 5.0_f64.sqrt());

    let normals = [
        DVec3::new(0.0, 1.0, phi),
        DVec3::new(0.0, 1.0, -phi),
        DVec3::new(0.0, -1.0, phi),
        DVec3::new(0.0, -1.0, -phi),
        DVec3::new(1.0, phi, 0.0),
        DVec3::new(1.0, -phi, 0.0),
        DVec3::new(-1.0, phi, 0.0),
        DVec3::new(-1.0, -phi, 0.0),
        DVec3::new(phi, 0.0, 1.0),
        DVec3::new(phi, 0.0, -1.0),
        DVec3::new(-phi, 0.0, 1.0),
        DVec3::new(-phi, 0.0, -1.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(1.0, 1.0, -1.0),
        DVec3::new(1.0, -1.0, 1.0),
        DVec3::new(1.0, -1.0, -1.0),
        DVec3::new(-1.0, 1.0, 1.0),
        DVec3::new(-1.0, 1.0, -1.0),
        DVec3::new(-1.0, -1.0, 1.0),
        DVec3::new(-1.0, -1.0, -1.0),
    ];

    normals.into_iter().map(|n| (n.normalize(), 2.0)).collect()
}

/// Generate random planes that form a bounded polytope
fn random_planes(count: usize, seed: u64) -> Vec<(DVec3, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Start with a cube to ensure boundedness
    let mut planes = cube_planes();

    // Add random additional planes
    for _ in 6..count {
        let x: f64 = rng.random_range(-1.0..1.0);
        let y: f64 = rng.random_range(-1.0..1.0);
        let z: f64 = rng.random_range(-1.0..1.0);

        let normal = DVec3::new(x, y, z);
        if normal.length() > 0.1 {
            let normal = normal.normalize();
            let offset = rng.random_range(0.2..0.4);
            planes.push((normal, offset));
        }
    }

    planes
}

/// Generate sphere-like polytope planes using Fibonacci distribution
#[expect(clippy::cast_precision_loss)]
fn fibonacci_sphere_planes(n: usize) -> Vec<(DVec3, f64)> {
    let golden = f64::midpoint(1.0, 5.0_f64.sqrt());

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

// ============================================================================
// Batch Construction Benchmarks
// ============================================================================

#[divan::bench]
fn batch_cube(bencher: Bencher) {
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_octahedron(bencher: Bencher) {
    let planes = octahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_dodecahedron(bencher: Bencher) {
    let planes = dodecahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_icosahedron(bencher: Bencher) {
    let planes = icosahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Scalability Benchmarks
// ============================================================================

#[divan::bench(args = [10, 20, 30, 50])]
fn scale_random(bencher: Bencher, n: usize) {
    let planes = random_planes(n, 0xdead_beef);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [8, 12, 16, 20, 30, 50])]
fn scale_fibonacci(bencher: Bencher, n: usize) {
    let planes = fibonacci_sphere_planes(n);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Mesh Extraction Benchmarks
// ============================================================================

#[divan::bench]
fn to_mesh_cube(bencher: Bencher) {
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_octahedron(bencher: Bencher) {
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Incremental Addition Benchmarks
// ============================================================================

#[divan::bench]
fn add_single_plane(bencher: Bencher) {
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Add a diagonal clipping plane
        let diagonal = DVec3::new(1.0, 1.0, 1.0).normalize();
        poly.add_plane(HalfSpace::new(diagonal, 0.2));

        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [2, 4, 6, 8])]
fn add_multiple_planes(bencher: Bencher, extra_planes: usize) {
    let base_planes = cube_planes();
    let mut rng = StdRng::seed_from_u64(12345);

    let extra: Vec<(DVec3, f64)> = (0..extra_planes)
        .map(|_| {
            let x: f64 = rng.random_range(-1.0..1.0);
            let y: f64 = rng.random_range(-1.0..1.0);
            let z: f64 = rng.random_range(-1.0..1.0);
            let normal = DVec3::new(x, y, z).normalize();
            let offset = rng.random_range(0.2..0.4);
            (normal, offset)
        })
        .collect();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();

        for (normal, offset) in &base_planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        for (normal, offset) in &extra {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Plane Removal Benchmarks
// ============================================================================

#[divan::bench]
fn remove_single_plane(bencher: Bencher) {
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        poly.remove_plane(PlaneIdx(0));

        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn remove_and_readd_plane(bencher: Bencher) {
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        poly.remove_plane(PlaneIdx(0));
        poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5));

        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [20, 30, 40, 50])]
fn remove_plane_from_large_polytope(bencher: Bencher, base_planes: usize) {
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &sphere_planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        let before_vertices = poly.vertex_count();
        poly.remove_plane(PlaneIdx(5));

        black_box((before_vertices, poly.vertex_count()))
    });
}

// ============================================================================
// Edge Query Benchmarks
// ============================================================================

#[divan::bench]
fn edge_count(bencher: Bencher) {
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| black_box(poly.edge_count()));
}

#[divan::bench]
fn edge_iteration(bencher: Bencher) {
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut count = 0;
        for (_, edge) in poly.edges() {
            count += 1;
            black_box(edge);
        }
        black_box(count)
    });
}

// ============================================================================
// Cached vs Uncached Benchmarks
// ============================================================================

#[divan::bench]
fn to_mesh_repeated_cached(bencher: Bencher) {
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    // First call populates cache
    let _ = poly.to_mesh();

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_no_rebuild(bencher: Bencher) {
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh_no_rebuild();
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Clear and Rebuild Benchmarks
// ============================================================================

#[divan::bench(args = [10, 25, 50])]
fn construction_then_clear(bencher: Bencher, n: usize) {
    let planes = random_planes(n, 0xcafe_babe);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();

        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        let v_count = poly.vertex_count();
        poly.clear();

        for (normal, offset) in &planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        black_box((v_count, poly.vertex_count()))
    });
}

// ============================================================================
// Topology-Based Removal vs Full Rebuild Comparison
// ============================================================================

#[divan::bench(args = [30, 50, 80, 100, 150, 200])]
fn remove_vs_rebuild_topology(bencher: Bencher, base_planes: usize) {
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &sphere_planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        poly.remove_plane(PlaneIdx(5));
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [30, 50, 80, 100, 150, 200])]
fn remove_vs_rebuild_full(bencher: Bencher, base_planes: usize) {
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    let remaining_planes: Vec<_> = sphere_planes
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 5)
        .map(|(_, p)| *p)
        .collect();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &remaining_planes {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Real-World Use Case Benchmarks
// ============================================================================

/// Simulate Voronoi cell modification
#[divan::bench]
fn realworld_voronoi_cell_removal(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    let n_neighbors = 14;

    let planes: Vec<(DVec3, f64)> = (0..n_neighbors)
        .map(|_| {
            let dir = DVec3::new(
                rng.random::<f64>() - 0.5,
                rng.random::<f64>() - 0.5,
                rng.random::<f64>() - 0.5,
            )
            .normalize();
            (dir, 0.8 + rng.random::<f64>() * 0.4)
        })
        .collect();

    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut cell = template.clone();
        cell.remove_plane(PlaneIdx(7));
        black_box(cell.vertex_count())
    });
}

/// CSG intersection: two rotated cubes
#[divan::bench]
fn realworld_csg_cube_intersection(bencher: Bencher) {
    let cube1: Vec<(DVec3, f64)> = vec![
        (DVec3::new(1.0, 0.0, 0.0), 1.0),
        (DVec3::new(-1.0, 0.0, 0.0), 1.0),
        (DVec3::new(0.0, 1.0, 0.0), 1.0),
        (DVec3::new(0.0, -1.0, 0.0), 1.0),
        (DVec3::new(0.0, 0.0, 1.0), 1.0),
        (DVec3::new(0.0, 0.0, -1.0), 1.0),
    ];

    let angle = std::f64::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    let cube2: Vec<(DVec3, f64)> = vec![
        (DVec3::new(c, s, 0.0), 1.0),
        (DVec3::new(-c, -s, 0.0), 1.0),
        (DVec3::new(-s, c, 0.0), 1.0),
        (DVec3::new(s, -c, 0.0), 1.0),
        (DVec3::new(0.0, 0.0, 1.0), 1.0),
        (DVec3::new(0.0, 0.0, -1.0), 1.0),
    ];

    let mut template = IncrementalPolytope::new();
    for (normal, offset) in cube1.iter().chain(cube2.iter()) {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        poly.remove_plane(PlaneIdx(3));
        black_box(poly.vertex_count())
    });
}

/// Scalability test for removal
#[divan::bench(args = [50, 100, 150, 200, 300])]
fn scalability_removal(bencher: Bencher, n_planes: usize) {
    let planes = fibonacci_sphere_planes(n_planes);

    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        poly.remove_plane(PlaneIdx(n_planes / 2));
        black_box(poly.vertex_count())
    });
}
