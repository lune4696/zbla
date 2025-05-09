const ar = @import("../../core/array.zig");

pub const Result = struct {
    x: ar.Dense(f64, 1),
    iter: usize,
    is_converged: bool,
    threshold: f64,
    res: f64,

    pub fn deinit(self: @This()) void {
        self.x.destroy();
    }
};
