const std = @import("std");

pub const core = @import("./core/bind.zig");
pub const graph = @import("./graph/bind.zig");
pub const solver = @import("./solver/bind.zig");

test "src/root" {
    std.testing.refAllDecls(core);
    std.testing.refAllDecls(graph);
    std.testing.refAllDecls(solver);
}
