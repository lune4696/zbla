const std = @import("std");

const core = @import("../../core.zig");
const math = @import("../../math.zig");

/// 目的
///     > 不完全コレスキー分解(A' = LDL.T)の実行
/// 返り値
///     > L: 下三角行列
///     > DL.T: 対角行列DとLの転置(L.T)の積
/// 注意点
///     > 厳密には修正不完全コレスキー分解
/// 参考
///     > https://pbcglab.jp/cgi-bin/wiki/
fn incompleteCholeskyDecomposition(T: type) fn (alc: std.mem.Allocator, A: *const core.Matrix(f64), non_zero_indices: []const usize) [2]core.Matrix(f64) {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Invalid type: f32 and f64 are accepted."),
    }
    return struct {
        fn f(tmp_alc: std.mem.Allocator, A: *const core.Matrix(f64), non_zero_indices: []const usize) [2]core.Matrix(f64) {
            const n = A.shape[0];
            var L = core.Matrix(f64).zeros(tmp_alc, &A.shape) catch unreachable;
            var D = core.Matrix(f64).zeros(tmp_alc, &A.shape) catch unreachable;
            defer D.destroy();
            const DL = core.Matrix(f64).any(tmp_alc, &A.shape) catch unreachable;

            // スパース性を用いるバージョン
            var idx: usize = 0;
            for (0..n) |i| {
                while (non_zero_indices[idx] < i * n) : (idx += 1) {}
                while (idx < non_zero_indices.len and non_zero_indices[idx] <= i * n + i) : (idx += 1) {
                    const j = non_zero_indices[idx] - i * n;
                    var ldl: f64 = A.get(&.{ i, j }) catch unreachable;
                    for (0..j) |k| {
                        const l_ik = L.get(&.{ i, k }) catch unreachable;
                        const d_kk = D.get(&.{ k, k }) catch unreachable;
                        const l_jk = L.get(&.{ j, k }) catch unreachable;
                        ldl -= l_ik * d_kk * l_jk;
                    }
                    L.set(&.{ i, j }, ldl) catch unreachable;
                }
                D.set(&.{ i, i }, 1 / (L.get(&.{ i, i }) catch unreachable)) catch unreachable;
            }
            L.tr();

            math.gemm(T)(D, L, DL, 1.0, 0.0);

            return .{ L, DL };
        }
    }.f;
}

/// 目的
///     > 対角スケーリング(a = r/trace(A))の実行
///     > 平方根を用いているので厳密には違うのだが、こっちのほうがずっと高速かつ振動無く収束するので採用
/// 注意点
///     > アロケータはrのアロケータを使用
fn diagonalScale(T: type) fn (r: *const core.Vector(T), A: *const core.Matrix(f64)) core.DataError!core.Vector(T) {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Invalid type: f32 and f64 are accepted."),
    }
    return struct {
        fn f(r: *const core.Vector(T), A: *const core.Matrix(f64)) core.DataError!core.Vector(T) {
            const a = r.copy() catch |e| return e;
            for (0..a.data.len) |i| a.data[i] /= (std.math.sqrt(A.get(&.{ i, i }) catch unreachable));
            return a;
        }
    }.f;
}

fn auxiliaryVector(T: type) fn (alc: std.mem.Allocator, r: *const core.Vector(T), L: *const core.Matrix(f64), DL: *const core.Matrix(f64)) core.ArrayError!core.Vector(T) {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Invalid type: f32 and f64 are accepted."),
    }
    return struct {
        fn f(alc: std.mem.Allocator, r: *const core.Vector(T), L: *const core.Matrix(f64), DL: *const core.Matrix(f64)) core.ArrayError!core.Vector(T) {
            const n = r.shape[0];
            const y = core.Vector(T).zeros(alc, &r.shape) catch |e| return e;
            defer y.destroy();
            const a = core.Vector(T).zeros(alc, &r.shape) catch |e| return e;
            // Ly = r
            for (0..n) |row| {
                y.data[row] = r.data[row];
                for (0..row) |col| y.data[row] -= (L.get(&.{ row, col }) catch unreachable) * y.data[col];
                y.data[row] /= (L.get(&.{ row, row }) catch unreachable);
            }
            // Uz = y (U = DL.T, z = a)
            for (0..n) |i| { // 逆順に走査
                const row = n - 1 - i;
                a.data[row] = y.data[row];
                for (row + 1..n) |col| a.data[row] -= (DL.get(&.{ row, col }) catch unreachable) * a.data[col];
                a.data[row] /= (DL.get(&.{ row, row }) catch unreachable);
            }
            return a;
        }
    }.f;
}

test "ICD" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alc = gpa.allocator();
    const hoge = try core.Dense(f64, 2).from(alc, &.{ 3, 3 }, &.{ 7, 1, 2, 1, 8, 0, 2, 0, 9 });
    defer hoge.destroy();
    const L, const D = incompleteCholeskyDecomposition(f64)(alc, &hoge, &.{ 0, 1, 2, 3, 4, 6, 8 });
    defer L.destroy();
    std.debug.print("L: {d}\n", .{L.data});
    defer D.destroy();
    std.debug.print("D: {d}\n", .{D.data});
}
