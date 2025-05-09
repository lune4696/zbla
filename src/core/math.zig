const std = @import("std");
const blas = @cImport({
    @cInclude("cblas.h");
});
const lapack = @cImport({
    @cInclude("lapack.h");
});

const arr = @import("./array.zig");

/// 目的
///     汎用行列-行列算術演算(GEneral Matrix-Matrix operation (GEMM), level 3 BLAS)
///     https://www.netlib.org/lapack/explore-html-3.6.1/index.html (docs(LAPACK))
///     C = alpha * a @ b + beta * C
/// 引数
///     where T = f32 or f64
///     alpha: T
///     beta: T,
///     a: Dense(T, 2), m by k matrix
///     b: Dense(T, 2), k by n matrix
pub fn gemm(comptime T: type) fn (a: *const arr.Dense(T, 2), b: *const arr.Dense(T, 2), c: *const arr.Dense(T, 2), alpha: T, beta: T) arr.Error!void {
    return struct {
        fn f(a: *const arr.Dense(T, 2), b: *const arr.Dense(T, 2), c: *const arr.Dense(T, 2), alpha: T, beta: T) arr.Error!void {
            switch (T) {
                f32, f64 => {},
                else => return arr.TypeError.DataTypeNotImplemented,
            }
            if (a.shape[1] != b.shape[0]) return arr.ShapeError.DimensionsMismatch;

            const m: i32 = @intCast(a.shape[0]);
            const n: i32 = @intCast(b.shape[1]);
            const k: i32 = @intCast(a.shape[1]);
            const ldc = m;
            const major = blas.CblasColMajor;
            const tr = blas.CblasTrans;
            const no = blas.CblasNoTrans;
            switch (T) {
                f32 => {
                    // 2次元配列であることを利用し、転置されたか否かのみでgemmの挙動を切り替える
                    switch (!a.isAligned()) {
                        true => {
                            switch (!b.isAligned()) {
                                // a: 転置有, b: 転置有
                                true => {
                                    const lda = k;
                                    const ldb = k;
                                    blas.cblas_sgemm(major, tr, tr, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                                // a: 転置有, b: 転置無
                                false => {
                                    const lda = k;
                                    const ldb = n;
                                    blas.cblas_sgemm(major, tr, no, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                            }
                        },
                        false => {
                            switch (!b.isAligned()) {
                                // a: 転置無, b: 転置有
                                true => {
                                    const lda = m;
                                    const ldb = k;
                                    blas.cblas_sgemm(major, no, tr, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                                // a: 転置無, b: 転置無
                                false => {
                                    const lda = m;
                                    const ldb = n;
                                    blas.cblas_sgemm(major, no, no, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                            }
                        },
                    }
                },
                f64 => {
                    // 2次元配列であることを利用し、転置されたか否かのみでgemmの挙動を切り替える
                    switch (!a.isAligned()) {
                        true => {
                            switch (!b.isAligned()) {
                                // a: 転置有, b: 転置有
                                true => {
                                    const lda = k;
                                    const ldb = k;
                                    blas.cblas_dgemm(major, tr, tr, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                                // a: 転置有, b: 転置無
                                false => {
                                    const lda = k;
                                    const ldb = n;
                                    blas.cblas_dgemm(major, tr, no, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                            }
                        },
                        false => {
                            switch (!b.isAligned()) {
                                // a: 転置無, b: 転置有
                                true => {
                                    const lda = m;
                                    const ldb = k;
                                    blas.cblas_dgemm(major, no, tr, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                                // a: 転置無, b: 転置無
                                false => {
                                    const lda = m;
                                    const ldb = n;
                                    blas.cblas_dgemm(major, no, no, m, n, k, alpha, a.data.ptr, lda, b.data.ptr, ldb, beta, c.data.ptr, ldc);
                                },
                            }
                        },
                    }
                },
                else => unreachable,
            }
        }
    }.f;
}

/// 目的
///     汎用行列-ベクトル算術演算(GEneral Matrix-Vector operation (GEMV), level 2 BLAS)
///     https://www.netlib.org/lapack/explore-html-3.6.1/index.html (docs(LAPACK))
///     y := alpha * a @ x + beta * y
/// 引数
///     where T = f32 or f64
///     alpha: T
///     beta: T,
///     a: Dense(T, 2), m by n matrix
///     x: Dense(T, 1), n row vector
///     y: Dense(T, 1), m row vector
pub fn gemv(comptime T: type) fn (a: *const arr.Dense(T, 2), x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1), alpha: T, beta: T) arr.Error!void {
    return struct {
        fn f(a: *const arr.Dense(T, 2), x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1), alpha: T, beta: T) arr.Error!void {
            switch (T) {
                f32, f64 => {},
                else => return arr.TypeError.DataTypeNotImplemented,
            }
            if (a.shape[1] != x.shape[0]) return arr.ShapeError.DimensionsMismatch;

            const m: i32 = @intCast(a.shape[0]);
            const n: i32 = @intCast(a.shape[1]);
            // column-majorなので、x, yは常に列ベクトルであると想定しているためincの修正の必要はない(この場合)
            const inc_x = 1;
            const inc_y = 1;
            const major = blas.CblasColMajor;
            const tr = blas.CblasTrans;
            const no = blas.CblasNoTrans;
            switch (T) {
                f32 => {
                    // 2次元配列であることを利用し、転置されたか否かのみでgemmの挙動を切り替える
                    switch (!a.isAligned()) {
                        true => {
                            const lda = n;
                            blas.cblas_sgemv(major, tr, m, n, alpha, a.data.ptr, lda, x.data.ptr, inc_x, beta, y.data.ptr, inc_y);
                        },
                        false => {
                            const lda = m;
                            blas.cblas_sgemv(major, no, m, n, alpha, a.data.ptr, lda, x.data.ptr, inc_x, beta, y.data.ptr, inc_y);
                        },
                    }
                },
                f64 => {
                    // 2次元配列であることを利用し、転置されたか否かのみでgemmの挙動を切り替える
                    switch (!a.isAligned()) {
                        true => {
                            const lda = n;
                            blas.cblas_dgemv(major, tr, m, n, alpha, a.data.ptr, lda, x.data.ptr, inc_x, beta, y.data.ptr, inc_y);
                        },
                        false => {
                            const lda = m;
                            blas.cblas_dgemv(major, no, m, n, alpha, a.data.ptr, lda, x.data.ptr, inc_x, beta, y.data.ptr, inc_y);
                        },
                    }
                },
                else => unreachable,
            }
        }
    }.f;
}

/// 目的
///     汎用ベクトル-ベクトル算術演算(AXPlusY (AXPY), level 2 BLAS)
///     https://www.netlib.org/lapack/explore-html-3.6.1/index.html (docs(LAPACK))
///     y := alpha * x + y
/// 引数
///     where T = f32 or f64
///     alpha: T
///     x: Dense(T, 1), n row vector
///     y: Dense(T, 1), n row vector
pub fn axpy(comptime T: type) fn (x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1), alpha: T) arr.Error!void {
    return struct {
        fn f(x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1), alpha: T) arr.Error!void {
            switch (T) {
                f32, f64 => {},
                else => return arr.TypeError.DataTypeNotImplemented,
            }
            if (x.shape[0] != y.shape[0]) return arr.ShapeError.DimensionsMismatch;

            const n: i32 = @intCast(x.shape[0]);
            // column-majorなので、x, yは常に列ベクトルであると想定しているためincの修正の必要はない(この場合)
            const inc_x = 1;
            const inc_y = 1;
            switch (T) {
                f32 => blas.cblas_saxpy(n, alpha, x.data.ptr, inc_x, y.data.ptr, inc_y),
                f64 => blas.cblas_daxpy(n, alpha, x.data.ptr, inc_x, y.data.ptr, inc_y),
                else => unreachable,
            }
        }
    }.f;
}

/// 目的
///     ベクトル-ベクトル内積(dot, level 1 BLAS)
///     https://www.netlib.org/lapack/explore-html-3.6.1/index.html (docs(LAPACK))
///     ans := x · y
/// 引数
///     where T = f32 or f64
///     x: Dense(T, 1), n row vector
///     y: Dense(T, 1), n row vector
/// 注意点
///     cblasにはdsdotのような混合精度演算、sdsdotのようなスカラー倍混合演算もあるが、現在は使用していない
pub fn dot(comptime T: type) fn (x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1)) arr.Error!T {
    return struct {
        fn f(x: *const arr.Dense(T, 1), y: *const arr.Dense(T, 1)) arr.Error!T {
            switch (T) {
                f32, f64 => {},
                else => return arr.TypeError.DataTypeNotImplemented,
            }
            if (x.shape[0] != y.shape[0]) return arr.ShapeError.DimensionsMismatch;

            const n: i32 = @intCast(x.shape[0]);
            // column-majorなので、x, yは常に列ベクトルであると想定しているためincの修正の必要はない(この場合)
            const inc_x = 1;
            const inc_y = 1;
            return switch (T) {
                f32 => blas.cblas_sdot(n, x.data.ptr, inc_x, y.data.ptr, inc_y),
                f64 => blas.cblas_ddot(n, x.data.ptr, inc_x, y.data.ptr, inc_y),
                else => unreachable,
            };
        }
    }.f;
}

/// 目的
///     ベクトルのスカラー倍(scal, level 1 BLAS)
///     https://www.netlib.org/lapack/explore-html-3.6.1/index.html (docs(LAPACK))
///     x := alpha * x
/// 引数
///     where T = f32 or f64
///     alpha: T
///     x: Dense(T, 1), n row vector
/// 注意点
///     cblasにはdsdotのような混合精度演算、sdsdotのようなスカラー倍混合演算もあるが、現在は使用していない
pub fn scal(comptime T: type) fn (x: *const arr.Dense(T, 1), alpha: T) arr.DataError!void {
    return struct {
        fn f(x: *const arr.Dense(T, 1), alpha: T) arr.DataError!void {
            switch (T) {
                f32, f64 => {},
                else => return arr.TypeError.DataTypeNotImplemented,
            }
            const n: i32 = @intCast(x.shape[0]);
            // column-majorなので、x, yは常に列ベクトルであると想定しているためincの修正の必要はない(この場合)
            const inc_x = 1;
            switch (T) {
                f32 => blas.cblas_sscal(n, alpha, x.data.ptr, inc_x),
                f64 => blas.cblas_dscal(n, alpha, x.data.ptr, inc_x),
                else => unreachable,
            }
        }
    }.f;
}

test "gemm" {
    std.debug.print("\nTEST: gemm\n", .{});
    std.debug.print("\n", .{});

    const alc = std.testing.allocator;

    {
        std.debug.print("VERIFY: gemm(f32, 2)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 1, -1, -2, 0, 1, 0, 2, 1 };
        const b_data = &.{ -0.5, -0.75, 0.25, 0.5, 0.25, 0.25, -1, -0.5, 0.5 };

        const answer = &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try arr.Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try arr.Dense(f32, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f32)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f32, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f32, 2) (trans, trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 30, 84, 138, 24, 69, 114, 18, 54, 90 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try arr.Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try arr.Dense(f32, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f32)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f32, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f32, 2) (trans, no trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 46, 118, 190, 28, 73, 118, 10, 28, 46 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try arr.Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try arr.Dense(f32, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f32)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f32, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f32, 2) (no trans, trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 54, 72, 90, 42, 57, 72, 30, 42, 54 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try arr.Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try arr.Dense(f32, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f32)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f32, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f64, 2)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 1, -1, -2, 0, 1, 0, 2, 1 };
        const b_data = &.{ -0.5, -0.75, 0.25, 0.5, 0.25, 0.25, -1, -0.5, 0.5 };

        const answer = &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try arr.Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try arr.Dense(f64, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f64)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f64, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f64, 2) (trans, trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 30, 84, 138, 24, 69, 114, 18, 54, 90 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try arr.Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try arr.Dense(f64, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f64)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f64, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f64, 2) (trans, no trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 46, 118, 190, 28, 73, 118, 10, 28, 46 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try arr.Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try arr.Dense(f64, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f64)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f64, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemm(f64, 2) (no trans, trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const b_shape = &.{ 3, 3 };
        const c_shape = &.{ 3, 3 };

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const b_data = &.{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        const answer = &.{ 54, 72, 90, 42, 57, 72, 30, 42, 54 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try arr.Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try arr.Dense(f64, 2).any(alc, c_shape);
        defer c.destroy();

        try gemm(f64)(&a, &b, &c, 1, 0);
        try c.print();

        try std.testing.expect(std.mem.eql(f64, c.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "gemv" {
    std.debug.print("\nTEST: gemv\n", .{});
    std.debug.print("\n", .{});

    const alc = std.testing.allocator;

    {
        std.debug.print("VERIFY: gemv(f32, 2) (no trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const x_shape = &.{3};
        const y_shape = &.{3};

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const x_data = &.{ 1, 2, 3 };

        const answer = &.{ 30, 36, 42 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const x = try arr.Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f32, 1).any(alc, y_shape);
        defer y.destroy();

        try gemv(f32)(&a, &x, &y, 1, 0);
        try y.print();

        try std.testing.expect(std.mem.eql(f32, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemv(f32, 2) (trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const x_shape = &.{3};
        const y_shape = &.{3};

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const x_data = &.{ 1, 2, 3 };

        const answer = &.{ 14, 32, 50 };

        const a = try arr.Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const x = try arr.Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f32, 1).any(alc, y_shape);
        defer y.destroy();

        try gemv(f32)(&a, &x, &y, 1, 0);
        try y.print();

        try std.testing.expect(std.mem.eql(f32, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemv(f64, 2) (no trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const x_shape = &.{3};
        const y_shape = &.{3};

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const x_data = &.{ 1, 2, 3 };

        const answer = &.{ 30, 36, 42 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const x = try arr.Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f64, 1).any(alc, y_shape);
        defer y.destroy();

        try gemv(f64)(&a, &x, &y, 1, 0);
        try y.print();

        try std.testing.expect(std.mem.eql(f64, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: gemv(f64, 2) (trans)...\n", .{});
        const a_shape = &.{ 3, 3 };
        const x_shape = &.{3};
        const y_shape = &.{3};

        const a_data = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const x_data = &.{ 1, 2, 3 };

        const answer = &.{ 14, 32, 50 };

        const a = try arr.Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const x = try arr.Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f64, 1).any(alc, y_shape);
        defer y.destroy();

        try gemv(f64)(&a, &x, &y, 1, 0);
        try y.print();

        try std.testing.expect(std.mem.eql(f64, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "axpy" {
    std.debug.print("\nTEST: axpy\n", .{});
    std.debug.print("\n", .{});

    const alc = std.testing.allocator;

    {
        std.debug.print("VERIFY: axpy(f32, 2)...\n", .{});
        const x_shape = &.{3};
        const y_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const y_data = &.{ 1, 2, 3 };
        const alpha = 2;

        const answer = &.{ 3, 6, 9 };

        const x = try arr.Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f32, 1).from(alc, y_shape, y_data);
        defer y.destroy();

        try axpy(f32)(&x, &y, alpha);
        try y.print();

        try std.testing.expect(std.mem.eql(f32, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: axpy(f64, 2)...\n", .{});
        const x_shape = &.{3};
        const y_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const y_data = &.{ 1, 2, 3 };
        const alpha = 2;

        const answer = &.{ 3, 6, 9 };

        const x = try arr.Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f64, 1).from(alc, y_shape, y_data);
        defer y.destroy();

        try axpy(f64)(&x, &y, alpha);
        try y.print();

        try std.testing.expect(std.mem.eql(f64, y.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "scal" {
    std.debug.print("\nTEST: scal\n", .{});
    std.debug.print("\n", .{});

    const alc = std.testing.allocator;

    {
        std.debug.print("VERIFY: scal(f32, 2)...\n", .{});
        const x_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const alpha = 3;

        const answer = &.{ 3, 6, 9 };

        const x = try arr.Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        try scal(f32)(&x, alpha);
        try x.print();

        try std.testing.expect(std.mem.eql(f32, x.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: scal(f64, 2)...\n", .{});
        const x_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const alpha = 3;

        const answer = &.{ 3, 6, 9 };

        const x = try arr.Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        try scal(f64)(&x, alpha);
        try x.print();

        try std.testing.expect(std.mem.eql(f64, x.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "dot" {
    std.debug.print("\nTEST: dot\n", .{});
    std.debug.print("\n", .{});

    const alc = std.testing.allocator;

    {
        std.debug.print("VERIFY: dot(f32, 2)...\n", .{});
        const x_shape = &.{3};
        const y_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const y_data = &.{ 1, 2, 3 };

        const answer = 14;

        const x = try arr.Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f32, 1).from(alc, y_shape, y_data);
        defer y.destroy();

        const result = try dot(f32)(&x, &y);

        try std.testing.expect(result == answer);

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: dot(f64, 2)...\n", .{});
        const x_shape = &.{3};
        const y_shape = &.{3};

        const x_data = &.{ 1, 2, 3 };
        const y_data = &.{ 1, 2, 3 };

        const answer = 14;

        const x = try arr.Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try arr.Dense(f64, 1).from(alc, y_shape, y_data);
        defer y.destroy();

        const result = try dot(f64)(&x, &y);

        try std.testing.expect(result == answer);

        std.debug.print("...SUCCESS\n", .{});
    }
}
