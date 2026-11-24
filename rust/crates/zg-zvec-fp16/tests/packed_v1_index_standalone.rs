//! Standalone, dependency-free tests of the experimental V1 boundary.
//! This does not start zvec, create a graph, or touch a registered workspace.

#[path = "../src/packed_sign_index.rs"]
mod packed_sign_index;
#[path = "../src/packed_sign_scale.rs"]
mod packed_sign_scale;
