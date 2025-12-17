# poly_surge

**Somewhat novel**, fast incremental polytope surgery in Rust. Add and remove halfspaces, **23×–862x faster** than the standard "just reconstruct it" approach.

## What is this?

A convex polytope in 3D is a bounded region defined by the intersection of half-spaces. This crate lets you **dynamically add and remove** these half-space constraints while preserving the mesh topology (vertices, edges, faces)—no full reconstruction required.

## Quick Start

```rust
use poly_surge::{IncrementalPolytope, HalfSpace};
use glam::DVec3;

// Create a polytope and add planes to form a cube
let mut polytope = IncrementalPolytope::new();

// Six faces of a unit cube centered at origin
let planes = [
    HalfSpace::new(DVec3::X, 0.5),    // +X face
    HalfSpace::new(-DVec3::X, 0.5),   // -X face
    HalfSpace::new(DVec3::Y, 0.5),    // +Y face
    HalfSpace::new(-DVec3::Y, 0.5),   // -Y face
    HalfSpace::new(DVec3::Z, 0.5),    // +Z face
    HalfSpace::new(-DVec3::Z, 0.5),   // -Z face
];

for plane in planes {
    polytope.add_plane(plane);
}

assert_eq!(polytope.vertex_count(), 8);  // Cube has 8 vertices
assert!(polytope.is_bounded());

// Remove one face - the cube becomes an infinite prism
let plane_to_remove = polytope.planes().next().unwrap().0;
polytope.remove_plane(plane_to_remove);

assert!(!polytope.is_bounded());  // Now unbounded
```

## Features

- **Incremental construction**: Add planes one at a time with O(V + E) clipping
- **Topology-based removal**: Remove planes in ~O(N) typical time (vs O(N log N + V) rebuild)
- **Lazy evaluation**: Edges and face orderings built on-demand
- **Unbounded support**: Handles infinite polytopes via homogeneous coordinates
- **Lazy boundedness check**: O(1) query when cached, O(N log N) rebuild when dirty

## When to Use

- Dynamic Voronoi diagrams (cell modification)
- CSG operations on halfspace-defined solids
- Robot motion planning (C-space constraint relaxation)
- Interactive geometry editors
- Physics simulation (constraint relaxation)

## When NOT to Use

- N > 10,000 planes (consider specialized data structures)
- Exact arithmetic required (we use f64 with epsilon tolerance)
- Batch deletion of many planes at once (single rebuild may be faster)

## Performance

Benchmarks on Fibonacci-sphere polytopes (release mode, `cargo bench`):

| Planes | Topology-Based | Rebuild (incremental `add_plane`) | vs Rebuild |
|--------|----------------|-----------------------------------|------------|
| 30     | 17 μs          | 376 μs                            | **23×**    |
| 50     | 28 μs          | 1,433 μs                          | **51×**    |
| 80     | 43 μs          | 5,079 μs                          | **119×**   |
| 100    | 52 μs          | 9,638 μs                          | **186×**   |
| 150    | 76 μs          | 33,020 μs                         | **436×**   |
| 200    | 96 μs          | 82,500 μs                         | **862×**   |

**Scalability**: Near-linear O(N) — approximately 0.8 μs per plane.

**Real-world proxies**:

- Voronoi cell update (14 planes): ~20 μs
- CSG cube intersection (12 planes): ~11 μs

## Algorithm

The removal algorithm exploits vertex topology: when a plane is removed, vertices with only 2 remaining incident planes lie on a line. We search along that line to find where it hits other constraints, reconstructing the "unclipped" geometry.

See [ResearchProposal.md](ResearchProposal.md) for the full algorithm documentation.

## Origin & Attribution

This algorithm was developed as a hobby project while working on transforming the vertex enumeration code in the [demiurge](https://gitlab.com/dryad1/demiurge) library. The ideas discussed and extended were directly influenced by the vertex enumeration literature and the practical challenges encountered in that codebase.

Original implementation: <https://gitlab.com/dryad1/demiurge>

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
