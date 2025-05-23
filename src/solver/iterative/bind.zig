const core = @import("../../core.zig");

pub const cg = @import("./conjugate_gradient.zig");
pub const minres = @import("./minres.zig");

pub const Error = core.Error;

/// DS: Diagonal Scaling
/// ICD: IncompleteCholeskyDecomposition
pub const Preprocessing = enum {
    None,
    DS,
    ICD,
};

pub const Verbosity = enum {
    Silent,
    Result,
    Full,
};

/// CG: Conjugate Gradient method
/// MINRES: Minimization Residual method
pub const Solver = enum {
    CG,
    MINRES,
};

pub const Result = struct {
    x: core.Vector(f64),
    iter: usize,
    is_converged: bool,
    threshold: f64,
    res: f64,

    pub fn deinit(self: @This()) void {
        self.x.destroy();
    }
};

/// 目的
///     > 連立方程式 Ax=b の係数 A, b を格納する型
pub fn Vars(comptime T: type) type {
    return struct {
        a: core.Matrix(T),
        b: core.Vector(T),
        threshold: T,
        max_iteration: usize,
        verbosity: Verbosity,

        pub fn init(a: core.Matrix(T), b: core.Vector(T), threshold: T, max_iteration: usize, verbosity: Verbosity) @This() {
            switch (@typeInfo(T)) {
                .float => |info| {
                    switch (info.bits) {
                        32, 64 => {},
                        else => @compileError("Invalid type: f32 and f64 is accepted"),
                    }
                },
                else => @compileError("Invalid type: f32 and f64 is accepted"),
            }
            return .{
                .a = a,
                .b = b,
                .threshold = threshold,
                .max_iteration = max_iteration,
                .verbosity = verbosity,
            };
        }

        pub fn deinit(self: @This()) void {
            self.a.destroy();
            self.b.destroy();
        }
    };
}
