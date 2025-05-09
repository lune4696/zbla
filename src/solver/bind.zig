pub const iterative = @import("iterative/bind.zig");
pub const monitor = @import("monitor.zig");

pub const Error = TypeError;
pub const TypeError = error{
    NotImplemented,
};
