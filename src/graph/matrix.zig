const std = @import("std");

const ar = @import("../core/array.zig");
const Dense = ar.Dense;
const Matrix = ar.Matrix;
const Error = ar.Error;
const TypeError = ar.TypeError;

/// 目的
///     edge_index (Matrix(usize), .shape = (2, n), pytorch geometric 等と同じ形状) から 近傍行列 (adjacencyMatrix) を生成
///
pub fn adjacencyMatrix(comptime T: type) fn (edge_index: Matrix(usize), alc: std.mem.Allocator) Error!Matrix(T) {
    return struct {
        fn f(edge_index: Matrix(usize), alc: std.mem.Allocator) Error!Matrix(T) {
            switch (@typeInfo(T)) {
                .int, .float => |info| {
                    switch (info.bits) {
                        32, 64 => {},
                        else => return TypeError.DataTypeNotImplemented,
                    }
                },
                else => return TypeError.DataTypeNotImplemented,
            }

            const mat = Matrix(T).zeros(alc, &.{ edge_index.shape[1], edge_index.shape[1] });
            return mat;
        }
    }.f;
}
