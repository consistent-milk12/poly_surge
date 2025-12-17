//! Profiling binary for removal operation.
//! Run with: cargo flamegraph --profile release-with-debug --example profile_removal

use glam::DVec3;
use poly_surge::{HalfSpace, IncrementalPolytope};
use std::f64::consts::PI;

/// Generate Fibonacci sphere points for evenly distributed plane normals
fn fibonacci_sphere_planes(n: usize) -> Vec<(DVec3, f64)> {
    let phi = PI * (5.0_f64.sqrt() - 1.0); // golden angle
    (0..n)
        .map(|i| {
            let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0;
            let radius = (1.0 - y * y).sqrt();
            let theta = phi * i as f64;
            let normal = DVec3::new(theta.cos() * radius, y, theta.sin() * radius).normalize();
            (normal, 1.0)
        })
        .collect()
}

fn main() {
    const N: usize = 200;
    const ITERATIONS: usize = 10_000;

    // Build the base polytope once
    let planes = fibonacci_sphere_planes(N);
    let mut base = IncrementalPolytope::new();
    for (normal, offset) in &planes {
        base.add_plane(HalfSpace::new(*normal, *offset));
    }

    println!("Built polytope with {} planes, {} vertices", N, base.vertex_count());
    println!("Running {} removal iterations...", ITERATIONS);

    // Profile the removal operation
    for i in 0..ITERATIONS {
        let mut polytope = base.clone();
        let plane_idx = poly_surge::PlaneIdx(i % N);
        let _ = polytope.remove_plane(plane_idx);
    }

    println!("Done.");
}
