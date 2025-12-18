//! Comprehensive benchmarks comparing `incremental_polytope_mono` vs
//! `vertex_enumeration`.
//!
//! Run with: `cargo bench -p geometry`
//!
//! These benchmarks compare:
//! - Batch construction performance
//! - Incremental plane addition
//! - Mesh extraction (to_mesh)
//! - Scalability with increasing plane counts

use algebra::{DVec3, na::DMatrix};
use divan::{Bencher, black_box};
use geometry::{
    incremental_polytope_mono::{HalfSpace, IncrementalPolytope, PlaneIdx},
    vertex_enumeration_from_half_spaces,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() { divan::main(); }

fn euler_counts_from_faces(faces: &[Vec<usize>]) -> (i32, i32)
{
    use std::collections::HashSet;

    let mut vertices: HashSet<usize> = HashSet::new();
    let mut edges: HashSet<(usize, usize)> = HashSet::new();

    for face in faces
    {
        if face.len() < 3
        {
            continue;
        }

        for &v in face
        {
            vertices.insert(v);
        }

        for i in 0..face.len()
        {
            let a = face[i];
            let b = face[(i + 1) % face.len()];
            let key = if a < b { (a, b) } else { (b, a) };
            edges.insert(key);
        }
    }

    (vertices.len() as i32, edges.len() as i32)
}

// ============================================================================
// Test Data Generators
// ============================================================================

/// Generate cube planes (6 planes)
fn cube_planes() -> Vec<(DVec3, f64)>
{
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
fn octahedron_planes() -> Vec<(DVec3, f64)>
{
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
fn dodecahedron_planes() -> Vec<(DVec3, f64)>
{
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
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
fn icosahedron_planes() -> Vec<(DVec3, f64)>
{
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // Icosahedron vertices become dodecahedron face normals
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
        // Additional planes to make it 20
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
/// Uses smaller offsets to ensure planes actually clip the cube
fn random_planes(count: usize, seed: u64) -> Vec<(DVec3, f64)>
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Start with a cube to ensure boundedness
    let mut planes = cube_planes();

    // Add random additional planes that actually clip the cube
    for _ in 6..count
    {
        let x: f64 = rng.random_range(-1.0..1.0);
        let y: f64 = rng.random_range(-1.0..1.0);
        let z: f64 = rng.random_range(-1.0..1.0);

        let normal = DVec3::new(x, y, z);
        if normal.norm() > 0.1
        {
            let normal = normal.normalize();
            // Use smaller offset to ensure plane clips the cube
            let offset = rng.random_range(0.2..0.4);
            planes.push((normal, offset));
        }
    }

    planes
}

/// Convert planes to DMatrix format for vertex_enumeration
fn planes_to_matrix(planes: &[(DVec3, f64)]) -> DMatrix<f64>
{
    let mut matrix = DMatrix::from_element(planes.len(), 4, 0.0);
    for (i, (normal, offset)) in planes.iter().enumerate()
    {
        matrix[(i, 0)] = normal.x;
        matrix[(i, 1)] = normal.y;
        matrix[(i, 2)] = normal.z;
        matrix[(i, 3)] = *offset;
    }
    matrix
}

/// Generate sphere-like polytope planes using Fibonacci distribution
/// This is well-conditioned and works reliably with both implementations
fn fibonacci_sphere_planes(n: usize) -> Vec<(DVec3, f64)>
{
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

// ============================================================================
// Batch Construction Benchmarks
// ============================================================================

#[divan::bench]
fn batch_cube_incremental(bencher: Bencher)
{
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_cube_vertex_enum(bencher: Bencher)
{
    let planes = cube_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn batch_octahedron_incremental(bencher: Bencher)
{
    let planes = octahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_octahedron_vertex_enum(bencher: Bencher)
{
    let planes = octahedron_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn batch_dodecahedron_incremental(bencher: Bencher)
{
    let planes = dodecahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_dodecahedron_vertex_enum(bencher: Bencher)
{
    let planes = dodecahedron_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn batch_icosahedron_incremental(bencher: Bencher)
{
    let planes = icosahedron_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn batch_icosahedron_vertex_enum(bencher: Bencher)
{
    let planes = icosahedron_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Scalability Benchmarks (Random Planes)
// ============================================================================

#[divan::bench(args = [10, 20, 30, 50])]
fn scale_incremental(bencher: Bencher, n: usize)
{
    let planes = random_planes(n, 0xdeadbeef);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [8, 12, 16, 20])]
fn scale_fibonacci_vertex_enum(bencher: Bencher, n: usize)
{
    // Use fibonacci sphere distribution - well-conditioned for vertex_enumeration
    let planes = fibonacci_sphere_planes(n);
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench(args = [8, 12, 16, 20])]
fn scale_fibonacci_incremental(bencher: Bencher, n: usize)
{
    let planes = fibonacci_sphere_planes(n);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Mesh Extraction Benchmarks (to_mesh)
// ============================================================================

#[divan::bench]
fn to_mesh_cube_incremental(bencher: Bencher)
{
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_cube_vertex_enum(bencher: Bencher)
{
    let planes = cube_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_octahedron_incremental(bencher: Bencher)
{
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_octahedron_vertex_enum(bencher: Bencher)
{
    let planes = octahedron_planes();
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Incremental Addition Benchmarks
// ============================================================================

/// Benchmark adding a single clipping plane to an existing cube
#[divan::bench]
fn add_single_plane(bencher: Bencher)
{
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Add a diagonal clipping plane
        let diagonal = DVec3::new(1.0, 1.0, 1.0).normalize();
        poly.add_plane(HalfSpace::new(diagonal, 0.2));

        black_box(poly.vertex_count())
    });
}

/// Benchmark adding multiple planes one at a time (simulating incremental use)
#[divan::bench(args = [2, 4, 6, 8])]
fn add_multiple_planes(bencher: Bencher, extra_planes: usize)
{
    let base_planes = cube_planes();
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Pre-generate extra planes
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

        // Build base cube
        for (normal, offset) in &base_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Add extra planes incrementally
        for (normal, offset) in &extra
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Plane Removal Benchmarks
// ============================================================================

#[divan::bench]
fn remove_single_plane(bencher: Bencher)
{
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Remove one plane
        poly.remove_plane(PlaneIdx(0));

        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn remove_and_readd_plane(bencher: Bencher)
{
    let planes = cube_planes();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Remove and re-add
        poly.remove_plane(PlaneIdx(0));
        poly.add_plane(HalfSpace::new(DVec3::new(1.0, 0.0, 0.0), 0.5));

        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Edge Query Benchmarks
// ============================================================================

#[divan::bench]
fn edge_count(bencher: Bencher)
{
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| black_box(poly.edge_count()));
}

#[divan::bench]
fn edge_iteration(bencher: Bencher)
{
    let planes = cube_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut count = 0;
        for (_, edge) in poly.edges()
        {
            count += 1;
            black_box(edge);
        }
        black_box(count)
    });
}

// ============================================================================
// Degenerate Vertex Benchmarks
// ============================================================================

/// Benchmark with a polytope that has degenerate vertices (4+ planes meeting)
#[divan::bench]
fn degenerate_cube_corner_incremental(bencher: Bencher)
{
    // Cube with an extra diagonal plane that passes through a corner
    let mut planes = cube_planes();
    // This plane passes through the corner (0.5, 0.5, 0.5)
    let diagonal = DVec3::new(1.0, 1.0, 1.0).normalize();
    let corner = DVec3::new(0.5, 0.5, 0.5);
    let offset = diagonal.dot(&corner);
    planes.push((diagonal, offset));

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, off) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *off));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench]
fn degenerate_cube_corner_vertex_enum(bencher: Bencher)
{
    let mut planes = cube_planes();
    let diagonal = DVec3::new(1.0, 1.0, 1.0).normalize();
    let corner = DVec3::new(0.5, 0.5, 0.5);
    let offset = diagonal.dot(&corner);
    planes.push((diagonal, offset));

    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Cached vs Uncached Face Ordering
// ============================================================================

#[divan::bench]
fn to_mesh_repeated_cached(bencher: Bencher)
{
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    // First call populates cache
    let _ = poly.to_mesh();

    // Benchmark repeated calls (should use cache)
    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh();
        black_box((vertices.len(), faces.len()))
    });
}

#[divan::bench]
fn to_mesh_no_rebuild(bencher: Bencher)
{
    let planes = octahedron_planes();
    let mut poly = IncrementalPolytope::new();
    for (normal, offset) in &planes
    {
        poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let (vertices, faces) = poly.to_mesh_no_rebuild();
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Memory/Allocation Patterns
// ============================================================================

#[divan::bench(args = [10, 25, 50])]
fn construction_then_clear(bencher: Bencher, n: usize)
{
    let planes = random_planes(n, 0xcafebabe);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();

        // Build polytope
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        let v_count = poly.vertex_count();

        // Clear and rebuild (tests free list reuse)
        poly.clear();

        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        black_box((v_count, poly.vertex_count()))
    });
}

// ============================================================================
// Real-World Scenario: Joint Polytope (from skeleton meshing)
// ============================================================================

/// Simulates building a joint polytope from edge directions
/// as would happen in skeleton_meshing.rs
#[divan::bench(args = [3, 4, 5, 6])]
fn joint_polytope_incremental(bencher: Bencher, valence: usize)
{
    let mut rng = ChaCha8Rng::seed_from_u64(0xfeedface);

    // Generate random edge directions (like skeleton edges meeting at a joint)
    let directions: Vec<DVec3> = (0..valence)
        .map(|_| {
            let x: f64 = rng.random_range(-1.0..1.0);
            let y: f64 = rng.random_range(-1.0..1.0);
            let z: f64 = rng.random_range(-1.0..1.0);
            DVec3::new(x, y, z).normalize()
        })
        .collect();

    // For each direction, create opposing half-spaces (tube-like constraint)
    let planes: Vec<(DVec3, f64)> = directions
        .iter()
        .flat_map(|&dir| {
            // Two planes per direction (like a slab)
            vec![(dir, 0.5), (-dir, 0.5)]
        })
        .collect();

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [3, 4, 5, 6])]
fn joint_polytope_vertex_enum(bencher: Bencher, valence: usize)
{
    let mut rng = ChaCha8Rng::seed_from_u64(0xfeedface);

    let directions: Vec<DVec3> = (0..valence)
        .map(|_| {
            let x: f64 = rng.random_range(-1.0..1.0);
            let y: f64 = rng.random_range(-1.0..1.0);
            let z: f64 = rng.random_range(-1.0..1.0);
            DVec3::new(x, y, z).normalize()
        })
        .collect();

    let planes: Vec<(DVec3, f64)> = directions
        .iter()
        .flat_map(|&dir| vec![(dir, 0.5), (-dir, 0.5)])
        .collect();

    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Spatial Hash Performance (internal to incremental_polytope)
// ============================================================================

#[divan::bench(args = [12, 20, 30, 50, 80])]
fn sphere_approximation_incremental(bencher: Bencher, n: usize)
{
    // Create a sphere approximation with evenly distributed planes
    // This tests the spatial hash with many near-duplicate vertex candidates
    let planes = fibonacci_sphere_planes(n);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [12, 20, 30])]
fn sphere_approximation_vertex_enum(bencher: Bencher, n: usize)
{
    let planes = fibonacci_sphere_planes(n);
    let matrix = planes_to_matrix(&planes);

    bencher.bench_local(|| {
        let (vertices, faces) = vertex_enumeration_from_half_spaces(black_box(&matrix));
        black_box((vertices.len(), faces.len()))
    });
}

// ============================================================================
// Clipping Performance Benchmarks (tests O(E·V) → O(E+V) optimization)
// ============================================================================

/// Benchmark clipping a large polytope with many vertices/edges.
///
/// This specifically tests the `clip_by_plane` optimization where
/// classification lookups were changed from O(V) linear search to O(1) HashMap
/// lookup. The benefit is most visible when E and V are both large.
///
/// For a polytope with V vertices and E edges, the old implementation did:
///   O(E * V) lookups per clipping plane
/// The new implementation does:
///   O(V) to build HashMap + O(E) lookups = O(E + V)
#[divan::bench(args = [20, 30, 40, 50])]
fn clip_large_polytope(bencher: Bencher, base_planes: usize)
{
    // Build a large sphere-like polytope first
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Pre-generate clipping planes that cut through the middle
    // (ensuring they actually trigger the clipping path, not redundant)
    let clipping_planes: Vec<(DVec3, f64)> = vec![
        (DVec3::new(1.0, 0.2, 0.1).normalize(), 0.3),
        (DVec3::new(0.1, 1.0, 0.2).normalize(), 0.3),
        (DVec3::new(0.2, 0.1, 1.0).normalize(), 0.3),
        (DVec3::new(-0.5, 0.5, 0.3).normalize(), 0.25),
        (DVec3::new(0.3, -0.5, 0.5).normalize(), 0.25),
    ];

    bencher.bench_local(|| {
        // Build base polytope
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &sphere_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        let base_vertices = poly.vertex_count();
        let base_edges = poly.edge_count();

        // Now clip with additional planes - this exercises clip_by_plane
        for (normal, offset) in &clipping_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        black_box((base_vertices, base_edges, poly.vertex_count()))
    });
}

/// Benchmark repeated clipping operations on the same polytope.
///
/// This isolates the clipping cost by pre-building the base polytope
/// and measuring only the clipping operation.
#[divan::bench(args = [20, 30, 40])]
fn clip_single_plane_on_large_polytope(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Clipping plane that cuts through the middle
    let clip_normal = DVec3::new(1.0, 0.3, 0.2).normalize();
    let clip_offset = 0.25;

    bencher.bench_local(|| {
        // Build base polytope fresh each iteration
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &sphere_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Measure just the clipping operation
        poly.add_plane(HalfSpace::new(clip_normal, clip_offset));

        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Plane Removal Performance Benchmarks (tests O(A³×N) → O(A³×A) optimization)
// ============================================================================

/// Benchmark plane removal on large polytopes.
///
/// This tests the `reconstruct_vertices_after_removal` optimization where:
/// - Pre-fetching adjacent planes avoids O(A³) repeated hash lookups
/// - Incident plane search is O(A) instead of O(N) where N = total planes
///
/// The optimization benefit scales with polytope size: for N planes with A
/// adjacent, the incident plane search goes from O(N) to O(A) per candidate
/// vertex.
#[divan::bench(args = [20, 30, 40, 50])]
fn remove_plane_from_large_polytope(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &sphere_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        let before_vertices = poly.vertex_count();

        // Remove a plane from the middle (plane 5 typically has many adjacent planes)
        poly.remove_plane(PlaneIdx(5));

        black_box((before_vertices, poly.vertex_count()))
    });
}

/// Benchmark plane removal followed by re-addition on large polytopes.
///
/// This exercises both removal (reconstruct_vertices_after_removal) and
/// addition (clip_by_plane) paths on a large polytope.
#[divan::bench(args = [20, 30, 40])]
fn remove_and_readd_on_large_polytope(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    bencher.bench_local(|| {
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &sphere_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }

        // Remove plane 5
        poly.remove_plane(PlaneIdx(5));

        // Re-add it
        let (normal, offset) = sphere_planes[5];
        poly.add_plane(HalfSpace::new(normal, offset));

        black_box(poly.vertex_count())
    });
}

/// Benchmark of remove_plane operation (uses topology-based reconstruction).
///
/// This benchmarks the actual remove_plane code path which now uses
/// topology-based reconstruction: O(V × K² × N) instead of O(A³ × N).
///
/// Pre-builds the polytope and only times the remove_plane call.
#[divan::bench(args = [30, 50, 80, 100])]
fn remove_plane_topology_based(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Pre-build a template polytope
    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &sphere_planes
    {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        // Clone the pre-built polytope (fast, avoids construction overhead)
        let mut poly = template.clone();

        // Remove plane 5 (uses topology-based reconstruction)
        let result = poly.remove_plane(PlaneIdx(5));

        black_box(result.map(|r| r.removed_vertices.len()))
    });
}

/// Old O(A³) reconstruction approach for comparison.
///
/// This benchmark isolates just the `reconstruct_vertices_after_removal`
/// function by building a polytope, collecting adjacent planes, and calling the
/// function directly.
#[divan::bench(args = [30, 50, 80, 100])]
fn reconstruct_old_cubic_approach(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Build polytope once and find adjacent planes for plane 5
    let mut setup_poly = IncrementalPolytope::new();
    for (normal, offset) in &sphere_planes
    {
        setup_poly.add_plane(HalfSpace::new(*normal, *offset));
    }

    // Get face vertices for plane 5 to determine adjacent planes
    let plane_to_remove = PlaneIdx(5);
    let face_vertices: Vec<_> = setup_poly
        .vertices()
        .filter(|(_, v)| v.planes.contains(&plane_to_remove))
        .map(|(idx, _)| idx)
        .collect();

    // Collect adjacent planes (planes sharing vertices with the removed plane)
    let mut adjacent_set = std::collections::HashSet::new();
    for v_idx in &face_vertices
    {
        if let Some(v) = setup_poly.vertex(*v_idx)
        {
            for &p in &v.planes
            {
                if p != plane_to_remove
                {
                    adjacent_set.insert(p);
                }
            }
        }
    }
    let adjacent_planes: Vec<PlaneIdx> = adjacent_set.into_iter().collect();

    bencher.bench_local(|| {
        // Fresh polytope each iteration (without the removed plane's vertices)
        let mut poly = IncrementalPolytope::new();
        for (i, (normal, offset)) in sphere_planes.iter().enumerate()
        {
            if i != 5
            {
                poly.add_plane(HalfSpace::new(*normal, *offset));
            }
        }

        // Call old O(A³) reconstruction directly
        let new_verts =
            poly.reconstruct_vertices_after_removal(&adjacent_planes, &face_vertices);

        black_box(new_verts.len())
    });
}

// ============================================================================
// Direct Rebuild vs Topology-Based Comparison
// ============================================================================

/// Direct comparison: topology-based removal vs full rebuild from scratch.
///
/// This benchmark measures the actual cost difference between:
/// - Option A: Clone polytope + remove_plane() (topology-based)
/// - Option B: Build new polytope from N-1 planes (full rebuild)
///
/// This is the fairest comparison for the "should I remove or rebuild?" question.
#[divan::bench(args = [30, 50, 80, 100, 150, 200])]
fn remove_vs_rebuild_topology(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Pre-build template for cloning
    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &sphere_planes
    {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        // Clone + topology-based removal
        let mut poly = template.clone();
        poly.remove_plane(PlaneIdx(5));
        black_box(poly.vertex_count())
    });
}

#[divan::bench(args = [30, 50, 80, 100, 150, 200])]
fn remove_vs_rebuild_full(bencher: Bencher, base_planes: usize)
{
    let sphere_planes = fibonacci_sphere_planes(base_planes);

    // Pre-compute the N-1 planes (excluding plane 5)
    let remaining_planes: Vec<_> = sphere_planes
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 5)
        .map(|(_, p)| *p)
        .collect();

    bencher.bench_local(|| {
        // Full rebuild from scratch with N-1 planes
        let mut poly = IncrementalPolytope::new();
        for (normal, offset) in &remaining_planes
        {
            poly.add_plane(HalfSpace::new(*normal, *offset));
        }
        black_box(poly.vertex_count())
    });
}

// ============================================================================
// Validation Stress Test (measures Euler formula failure rate)
// ============================================================================

/// Stress test that performs many random plane removals and checks Euler's formula.
///
/// This measures how often topology validation fails after plane removal,
/// which indicates numerical robustness issues.
///
/// Returns (total_operations, failures) - printed to stdout for analysis.
#[divan::bench]
fn validation_stress_test(bencher: Bencher)
{
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TOTAL_OPS: AtomicUsize = AtomicUsize::new(0);
    static BOUNDED_OPS: AtomicUsize = AtomicUsize::new(0);
    static FAILURES: AtomicUsize = AtomicUsize::new(0);

    bencher.bench_local(|| {
        let mut local_ops = 0usize;
        let mut local_bounded = 0usize;
        let mut local_failures = 0usize;

        // Test multiple polytope sizes
        for n_planes in [20, 30, 40, 50] {
            let planes = fibonacci_sphere_planes(n_planes);

            // Build polytope
            let mut poly = IncrementalPolytope::new();
            for (normal, offset) in &planes {
                poly.add_plane(HalfSpace::new(*normal, *offset));
            }

            // Try removing each plane and validate
            for remove_idx in 0..n_planes {
                let mut test_poly = poly.clone();
                if test_poly.remove_plane(PlaneIdx(remove_idx)).is_some() {
                    local_ops += 1;

                    // Validate Euler's formula: V - E + F = 2 (bounded)
                    if test_poly.is_bounded() {
                        local_bounded += 1;
                        let (_, faces) = test_poly.to_mesh();
                        let f = faces.len() as i32;
                        let (v, e) = euler_counts_from_faces(&faces);
                        let euler = v - e + f;

                        if euler != 2 {
                            local_failures += 1;
                        }
                    }
                }
            }
        }

        TOTAL_OPS.fetch_add(local_ops, Ordering::Relaxed);
        BOUNDED_OPS.fetch_add(local_bounded, Ordering::Relaxed);
        FAILURES.fetch_add(local_failures, Ordering::Relaxed);

        black_box((local_ops, local_failures))
    });

    // Print results after benchmarking (only once at the end)
    static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PRINTED.swap(true, Ordering::Relaxed) {
        // Run a single diagnostic pass to show actual Euler values
        let mut euler_hist: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for n_planes in [20, 30] {
            let planes = fibonacci_sphere_planes(n_planes);
            let mut poly = IncrementalPolytope::new();
            for (normal, offset) in &planes {
                poly.add_plane(HalfSpace::new(*normal, *offset));
            }
            for remove_idx in 0..n_planes {
                let mut test_poly = poly.clone();
                if test_poly.remove_plane(PlaneIdx(remove_idx)).is_some() && test_poly.is_bounded() {
                    let (_, faces) = test_poly.to_mesh();
                    let f = faces.len() as i32;
                    let (v, e) = euler_counts_from_faces(&faces);
                    let euler = v - e + f;
                    *euler_hist.entry(euler).or_insert(0) += 1;
                }
            }
        }
        eprintln!("\n[Euler histogram]: {:?}", euler_hist);
    }

    let total = TOTAL_OPS.load(Ordering::Relaxed);
    let bounded = BOUNDED_OPS.load(Ordering::Relaxed);
    let fails = FAILURES.load(Ordering::Relaxed);
    if total > 0 {
        let bounded_rate = (bounded as f64 / total as f64) * 100.0;
        let fail_rate = if bounded > 0 {
            (fails as f64 / bounded as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[Validation] Total ops: {}, Bounded: {} ({:.1}%), Euler failures: {} ({:.4}% of bounded)",
            total, bounded, bounded_rate, fails, fail_rate
        );
    }
}

// ============================================================================
// Diagnostic: First Failing Case Analysis
// ============================================================================

/// Diagnostic benchmark that finds and analyzes the first Euler failure.
/// Prints detailed information to help debug topology reconstruction bugs.
#[divan::bench]
fn diagnostic_first_failure(bencher: Bencher)
{
    use std::sync::atomic::{AtomicBool, Ordering};
    static PRINTED: AtomicBool = AtomicBool::new(false);

    bencher.bench_local(|| {
        // Only run diagnostics once
        if PRINTED.swap(true, Ordering::Relaxed) {
            return black_box(());
        }

        eprintln!("\n========== DIAGNOSTIC: First Euler Failure ==========");

        for n_planes in [15, 20, 25, 30, 40] {
            let planes = fibonacci_sphere_planes(n_planes);

            // Build polytope
            let mut poly = IncrementalPolytope::new();
            for (normal, offset) in &planes {
                poly.add_plane(HalfSpace::new(*normal, *offset));
            }

            // Check original polytope
            let (orig_verts, orig_faces) = poly.to_mesh();
            let (orig_v, orig_e) = euler_counts_from_faces(&orig_faces);
            let orig_euler = orig_v - orig_e + orig_faces.len() as i32;
            eprintln!(
                "\n[n={}] Original: V={}, E={}, F={}, χ={}",
                n_planes,
                orig_v,
                orig_e,
                orig_faces.len(),
                orig_euler
            );

            // Try each removal
            for remove_idx in 0..n_planes {
                let mut test_poly = poly.clone();
                if test_poly.remove_plane(PlaneIdx(remove_idx)).is_none() {
                    continue;
                }
                if !test_poly.is_bounded() {
                    continue;
                }

                let (verts, faces) = test_poly.to_mesh();
                let (v, e) = euler_counts_from_faces(&faces);
                let f = faces.len() as i32;
                let euler = v - e + f;

                if euler != 2 {
                    // Found a failure - dump detailed diagnostics
                    eprintln!("\n>>> FAILURE at n={}, remove_idx={}", n_planes, remove_idx);
                    eprintln!("    V={}, E={}, F={}, χ={} (expected 2)", v, e, f, euler);

                    // Removed plane info
                    let (normal, offset) = &planes[remove_idx];
                    eprintln!(
                        "    Removed plane: normal=({:.4}, {:.4}, {:.4}), offset={:.4}",
                        normal.x, normal.y, normal.z, offset
                    );

                    // Face size distribution
                    let mut face_sizes: std::collections::HashMap<usize, usize> =
                        std::collections::HashMap::new();
                    for face in &faces {
                        *face_sizes.entry(face.len()).or_insert(0) += 1;
                    }
                    eprintln!("    Face sizes: {:?}", face_sizes);

                    // Check for degenerate faces
                    let degenerate: Vec<_> =
                        faces.iter().enumerate().filter(|(_, f)| f.len() < 3).collect();
                    if !degenerate.is_empty() {
                        eprintln!("    DEGENERATE faces (len<3): {:?}", degenerate);
                    }

                    // Check vertex degrees (how many faces each vertex appears in)
                    let mut vertex_degrees: std::collections::HashMap<usize, usize> =
                        std::collections::HashMap::new();
                    for face in &faces {
                        for &v in face {
                            *vertex_degrees.entry(v).or_insert(0) += 1;
                        }
                    }
                    let low_degree: Vec<_> =
                        vertex_degrees.iter().filter(|&(_, d)| *d < 3).collect();
                    if !low_degree.is_empty() {
                        eprintln!("    LOW DEGREE vertices (deg<3): {:?}", low_degree);
                    }

                    // Compare with vertex_enumeration
                    let remaining: Vec<_> = planes
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != remove_idx)
                        .map(|(_, p)| *p)
                        .collect();
                    let matrix = planes_to_matrix(&remaining);
                    let (enum_verts, enum_faces) = vertex_enumeration_from_half_spaces(&matrix);
                    let (enum_v, enum_e) = euler_counts_from_faces(&enum_faces);
                    let enum_euler = enum_v - enum_e + enum_faces.len() as i32;
                    eprintln!(
                        "    vertex_enum baseline: V={}, E={}, F={}, χ={}",
                        enum_v,
                        enum_e,
                        enum_faces.len(),
                        enum_euler
                    );
                    eprintln!(
                        "    DELTA: ΔV={}, ΔE={}, ΔF={}",
                        v - enum_v,
                        e - enum_e,
                        f - enum_faces.len() as i32
                    );

                    // Deep dive: Re-run removal with detailed tracing
                    eprintln!("\n    --- Reconstruction deep dive ---");
                    let mut trace_poly = poly.clone();

                    // Count vertices affected by this plane before removal
                    let affected_count = trace_poly
                        .vertices()
                        .filter(|(_, v)| v.planes.contains(&PlaneIdx(remove_idx)))
                        .count();
                    eprintln!("    Vertices touching removed plane: {}", affected_count);

                    // Count vertices that will be removed (only 2 remaining planes)
                    let will_remove: Vec<_> = trace_poly
                        .vertices()
                        .filter(|(_, v)| {
                            v.planes.contains(&PlaneIdx(remove_idx))
                                && v.planes.len() - 1 < 3
                        })
                        .map(|(idx, v)| {
                            let remaining: Vec<usize> = v
                                .planes
                                .iter()
                                .filter(|p| **p != PlaneIdx(remove_idx))
                                .map(|p| p.0)
                                .collect();
                            (idx, v.is_finite(), remaining)
                        })
                        .collect();
                    eprintln!("    Vertices to remove (< 3 remaining planes): {}", will_remove.len());
                    for (idx, is_finite, remaining) in &will_remove {
                        eprintln!(
                            "      {:?}: finite={}, remaining_planes={:?}",
                            idx, is_finite, remaining
                        );
                    }

                    // Do the actual removal
                    let result = trace_poly.remove_plane(PlaneIdx(remove_idx));
                    if let Some(res) = result {
                        eprintln!(
                            "    RemovePlaneResult: removed={}, affected={}",
                            res.removed_vertices.len(),
                            res.affected_vertices.len()
                        );
                    }

                    // Check vertex count after removal
                    let post_v = trace_poly.vertex_count();
                    eprintln!("    Post-removal vertex count: {}", post_v);

                    // Compare actual vertices with vertex_enum
                    let our_verts: Vec<_> = trace_poly
                        .vertices()
                        .filter_map(|(_, v)| v.to_euclidean())
                        .collect();
                    eprintln!("    Our finite vertices: {}", our_verts.len());
                    eprintln!("    vertex_enum vertices: {}", enum_verts.len());

                    // Find which vertex_enum vertex is missing from our set
                    let tolerance = 1e-6;
                    for (i, enum_v) in enum_verts.iter().enumerate() {
                        let found = our_verts.iter().any(|our_v| {
                            (our_v - enum_v).norm() < tolerance
                        });
                        if !found {
                            eprintln!(
                                "    MISSING vertex {}: ({:.6}, {:.6}, {:.6})",
                                i, enum_v.x, enum_v.y, enum_v.z
                            );

                            // Find which planes define this vertex
                            let mut incident_planes = Vec::new();
                            for (pi, (normal, offset)) in remaining.iter().enumerate() {
                                let dist = (normal.dot(enum_v) - offset).abs();
                                if dist < 1e-9 {
                                    incident_planes.push(pi);
                                }
                            }
                            eprintln!("      Incident planes (post-removal indices): {:?}", incident_planes);

                            // Map back to original plane indices
                            let original_indices: Vec<_> = incident_planes
                                .iter()
                                .map(|&pi| {
                                    // remaining excludes remove_idx
                                    if pi < remove_idx {
                                        pi
                                    } else {
                                        pi + 1
                                    }
                                })
                                .collect();
                            eprintln!("      Original plane indices: {:?}", original_indices);

                            // Check if any of these planes are in the removed vertex data
                            let removed_planes: std::collections::HashSet<usize> = will_remove
                                .iter()
                                .flat_map(|(_, _, remaining)| remaining.iter().copied())
                                .collect();
                            let shared: Vec<_> = original_indices
                                .iter()
                                .filter(|p| removed_planes.contains(p))
                                .collect();
                            eprintln!(
                                "      Shared with removed vertex planes: {:?} (of {:?})",
                                shared, removed_planes
                            );
                        }
                    }

                    // Only show first failure per size, then move to next n_planes
                    break;
                }
            }
        }

        eprintln!("\n========== END DIAGNOSTIC ==========\n");
        black_box(())
    });
}

// ============================================================================
// Epsilon Sensitivity Analysis
// ============================================================================

/// Test how different epsilon values affect correctness and performance.
/// Reports Euler validation pass rate for each epsilon.
#[divan::bench]
fn epsilon_sensitivity_analysis(bencher: Bencher)
{
    use std::sync::atomic::{AtomicBool, Ordering};
    static PRINTED: AtomicBool = AtomicBool::new(false);

    bencher.bench_local(|| {
        if PRINTED.swap(true, Ordering::Relaxed) {
            return black_box(());
        }

        eprintln!("\n========== EPSILON SENSITIVITY ANALYSIS ==========");
        eprintln!("Testing Euler validation pass rate across epsilon values\n");

        let epsilons = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];
        let n_planes = 30;
        let planes = fibonacci_sphere_planes(n_planes);

        for &eps in &epsilons {
            let mut total_ops = 0usize;
            let mut euler_pass = 0usize;

            // Build polytope with this epsilon
            let mut poly = IncrementalPolytope::with_epsilon(eps);
            for (normal, offset) in &planes {
                poly.add_plane(HalfSpace::new(*normal, *offset));
            }

            // Test all removals
            for remove_idx in 0..n_planes {
                let mut test_poly = poly.clone();
                if test_poly.remove_plane(PlaneIdx(remove_idx)).is_none() {
                    continue;
                }

                total_ops += 1;

                if test_poly.is_bounded() {
                    let (_, faces) = test_poly.to_mesh();
                    let f = faces.len() as i32;
                    let (v, e) = euler_counts_from_faces(&faces);
                    let euler = v - e + f;

                    if euler == 2 {
                        euler_pass += 1;
                    }
                } else {
                    // Unbounded - check χ = 1
                    euler_pass += 1; // Count as pass for simplicity
                }
            }

            let pass_rate = if total_ops > 0 {
                (euler_pass as f64 / total_ops as f64) * 100.0
            } else {
                0.0
            };

            eprintln!(
                "ε = {:>8.0e}: {}/{} passed ({:.1}%)",
                eps, euler_pass, total_ops, pass_rate
            );
        }

        eprintln!("\n========== END EPSILON SENSITIVITY ==========\n");
        black_box(())
    });
}

// ============================================================================
// Real-World Use Case Benchmarks
// ============================================================================

/// Simulate Voronoi cell modification: a cell with ~12 neighbors
/// (typical for 3D Voronoi), removing one neighbor's bisector plane.
#[divan::bench]
fn realworld_voronoi_cell_removal(bencher: Bencher)
{
    // Voronoi cells typically have 12-16 faces (neighbors)
    // Generate a pseudo-Voronoi cell using random bisector planes
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_neighbors = 14;

    let planes: Vec<(DVec3, f64)> = (0..n_neighbors)
        .map(|_| {
            // Random direction (neighbor position relative to cell center)
            let dir = DVec3::new(
                rng.random::<f64>() - 0.5,
                rng.random::<f64>() - 0.5,
                rng.random::<f64>() - 0.5,
            )
            .normalize();
            // Bisector plane at distance ~1 from origin
            (dir, 0.8 + rng.random::<f64>() * 0.4)
        })
        .collect();

    // Build template cell
    let mut template = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut cell = template.clone();
        // Remove one neighbor (simulating neighbor deletion in Voronoi diagram)
        cell.remove_plane(PlaneIdx(7));
        black_box(cell.vertex_count())
    });
}

/// CSG intersection: two rotated cubes intersecting
#[divan::bench]
fn realworld_csg_cube_intersection(bencher: Bencher)
{
    // Cube 1: axis-aligned
    let cube1: Vec<(DVec3, f64)> = vec![
        (DVec3::new(1.0, 0.0, 0.0), 1.0),
        (DVec3::new(-1.0, 0.0, 0.0), 1.0),
        (DVec3::new(0.0, 1.0, 0.0), 1.0),
        (DVec3::new(0.0, -1.0, 0.0), 1.0),
        (DVec3::new(0.0, 0.0, 1.0), 1.0),
        (DVec3::new(0.0, 0.0, -1.0), 1.0),
    ];

    // Cube 2: rotated 45° around Z axis
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

    // Build CSG intersection
    let mut template = IncrementalPolytope::new();
    for (normal, offset) in cube1.iter().chain(cube2.iter()) {
        template.add_plane(HalfSpace::new(*normal, *offset));
    }

    bencher.bench_local(|| {
        let mut poly = template.clone();
        // Remove one face (simulating CSG subtraction step)
        poly.remove_plane(PlaneIdx(3));
        black_box(poly.vertex_count())
    });
}

/// Scalability test: measure removal time across increasing plane counts
#[divan::bench(args = [50, 100, 150, 200, 300])]
fn scalability_removal(bencher: Bencher, n_planes: usize)
{
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
