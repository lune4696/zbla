//! Zig BLAS and LAPACK based (dense) array library.
//! zbla contains some functions
//!     > core: basic array function
//!         > array: column-order/major (dense) array Dense(T, n) and its method
//!         > math: mathematical operations for Dense(T, n)
//!     > graph: graph-related operations (graph laplacian, ...)
//!     > solver: (non-)iterative linear algebra solver
//!         > iterative: iterative solver (CG, MINRES, ...)
//!
const std = @import("std");

pub const core = @import("./core/bind.zig");
pub const graph = @import("./graph/bind.zig");
pub const solver = @import("./solver/bind.zig");

test "src/root" {
    std.testing.refAllDecls(core);
    std.testing.refAllDecls(graph);
    std.testing.refAllDecls(solver);
}
