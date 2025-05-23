const std = @import("std");

const core = @import("../../core.zig");
const math = @import("../../math.zig");
const root = @import("./bind.zig");

const Error = root.Error;
const Result = root.Result;
const Vars = root.Vars;
const Preprocessing = root.Preprocessing;
const Verbosity = root.Verbosity;

// Target
//     > 共役勾配法(Conjugate Gradient Method)に基づく頂点電位計算
///     > 前処理の有無でアルゴリズムが異なるため、前処理付き共役勾配法はsolvePCG()を使う
///
/// Input
///     > alc_obj: std.mem.Allocator
///         >> 関数スコープを抜ける際に解放されないデータに対するアロケータ
///     > vars.A: ar.Dense(T)
///         >> 連立方程式 Ax=b の係数行列 A
///     > vars.b
///         >> 連立方程式 Ax=b の右辺ベクトル b
///
/// Output
///     > res: Result
///         >> 演算結果
pub fn solveCG(alc: std.mem.Allocator, alc_tmp: std.mem.Allocator, vars: Vars(f64)) Error!Result {
    const a = vars.a;
    const b = vars.b;

    // 解xは中身だけが書き換わる
    const x = core.Vector(f64).any(alc, vars.b.shape) catch |e| return e;
    errdefer x.destroy();
    //var prng = std.Random.DefaultPrng.init(42);
    //for (0..x.data.len) |i| x.data[i] = prng.random().floatNorm(f64);

    const r = b.copy() catch |e| return e;
    defer r.destroy();
    // r = b-Ax = -1.0*Ax+b
    math.gemv(f64)(a, x, r, -1, 1) catch |e| return e;

    // p0 = r0
    var p = r.copy() catch |e| return e;
    defer p.destroy();

    var r_sum: f64 = 0;
    var i: usize = 0;
    for (r.data) |item| r_sum += @abs(item);
    while (r_sum > vars.threshold and i < vars.max_iteration) : (i += 1) {
        // kpはループ内のalphaを計算するのに必要
        const y = core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e;
        defer y.destroy();
        // y = a @ p
        math.gemv(f64)(a, p, y, 1, 0) catch |e| return e;

        // alpha = dot(r, r)/dot(y, p)
        const alpha: f64 = (math.dot(f64)(r, r) catch |e| return e) / (math.dot(f64)(y, p) catch |e| return e);

        // x_ = alpha*p + x
        math.axpy(f64)(p, x, alpha) catch |e| return e;

        // r_ = -alpha*y + r
        // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
        const r_ = r.copy() catch |e| return e;
        defer r_.destroy();
        math.axpy(f64)(y, r_, -alpha) catch |e| return e;

        // beta = dot(r_, r_)/dot(r, r)
        const beta: f64 = (math.dot(f64)(r_, r_) catch |e| return e) / (math.dot(f64)(r, r) catch |e| return e);

        // ここで r を r_ で更新
        @memcpy(r.data, r_.data);

        // p_ = r_ + beta*p
        // r_に入ったp_のデータをコピー
        math.axpy(f64)(p, r_, beta) catch |e| return e;
        @memcpy(p.data, r_.data);

        r_sum = 0;
        for (r.data) |data| r_sum += @abs(data);

        //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
        if (vars.verbosity != .Silent) std.debug.print("residual at loop {d}: {d}\n", .{ i, r_sum });
    }

    return .{
        .x = x,
        .iter = i,
        .is_converged = !(vars.max_iteration == i),
        .threshold = vars.threshold,
        .res = r_sum,
    };
}

///// Target
/////     > 前処理(Preprocessing)付き共役勾配法(Conjugate Gradient Method)に基づく頂点電位計算
/////     > 前処理の候補
/////         >> .ICD: 不完全(修正)コレスキー分解(Incomplete Cholesky Decomposition)
/////         >> .DS: 対角スケーリング(Diagonal Scaling)
/////     > 前処理の有無でアルゴリズムが異なるため、前処理のない共役勾配法はsolveCG()を使う
/////
///// Input
/////     > alc_obj: std.mem.Allocator
/////         >> 関数スコープを抜ける際に解放されないデータに対するアロケータ
/////     > vars.A: ar.Dense(T)
/////         >> 連立方程式 Ax=b の係数行列 A
/////     > vars.b
/////         >> 連立方程式 Ax=b の右辺ベクトル b
/////
///// Output
/////     > res: Result
/////         >> 演算結果
//pub fn solvePCG(self: @This(), preprocessing: Preprocessing, alc_obj: std.mem.Allocator, vars: Vars(f64), mesh: *const sys.sim.mesh.FEM) Error!Result {
//    // 電圧vは中身だけが書き換わる
//    const x = ar.Dense(T, 1).ones(alc_obj, &.{vars.b.data.len}) catch |e| return e;
//    errdefer x.destroy();
//
//    // kv = k@v
//    // kvは最初のr0を計算するのに必要なだけ
//    const kv = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
//    defer kv.destroy();
//    blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(x.shape[0]), x.data.ptr, 1, 0.0, kv.data.ptr, 1);
//    // TODO: beta: 1.0 -> 0.0に変更
//
//    // r = b-kv = -1.0*kv+b
//    // 残差rも中身だけが書き換わる
//    // b に r のデータが入るので改めてcopyすることに注意
//    const r = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
//    defer r.destroy();
//    blas.cblas_daxpy(@intCast(kv.shape[0]), -1.0, kv.data.ptr, 1, vars.b.data.ptr, 1);
//
//    // ここで r を j で更新 (必ずaがrをコピってくる後じゃないと動かない)
//    @memcpy(r.data, vars.b.data);
//
//    var non_zero_indices = switch (preprocessing) {
//        .ICD => mesh.nonZeroIndices(self.alc_tmp) catch unreachable,
//        else => undefined,
//    };
//    defer switch (preprocessing) {
//        .ICD => non_zero_indices.deinit(),
//        else => {},
//    };
//    // LUz = r is decomposited to Ly = r & Uz = y
//    const decomposed = switch (preprocessing) {
//        .ICD => incompleteCholeskyDecomposition(T)(self.alc_tmp, &vars.A, non_zero_indices.items),
//        else => undefined,
//    };
//
//    defer switch (preprocessing) {
//        .ICD => for (decomposed) |arr| arr.destroy(),
//        else => {},
//    };
//
//    const a = switch (preprocessing) {
//        .None => unreachable,
//        .DS => diagonalScaledVector(T)(&r, &vars.A) catch |e| return e,
//        .ICD => auxiliaryVector(T)(self.alc_tmp, &r, &decomposed[0], &decomposed[1]) catch |e| return e,
//    };
//    defer a.destroy();
//
//    // 共役勾配法: p <- r
//    // 前処理付き共役勾配法: p <- a
//    var p = a.copy() catch |e| return e;
//    defer p.destroy();
//
//    var r_sum: T = 0;
//    for (r.data) |data| r_sum += @abs(data);
//
//    var iter: usize = 0;
//    while (r_sum > self.threshold and iter < self.max_iter) : (iter += 1) {
//        // kpはループ内のalphaを計算するのに必要
//        const kp = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
//        defer kp.destroy();
//        blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(p.shape[0]), p.data.ptr, 1, 0.0, kp.data.ptr, 1);
//
//        // 共役勾配法: alpha = dot(r, r)/dot(kp, p)
//        // 前処理付き共役勾配法: alpha = dot(r, a)/dot(kp, p)
//        const alpha_numerator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, a.data.ptr, 1);
//        const alpha_denominator = blas.cblas_ddot(@intCast(kp.shape[0]), kp.data.ptr, 1, p.data.ptr, 1);
//        const alpha: T = alpha_numerator / alpha_denominator;
//
//        // v += alpha*p
//        blas.cblas_daxpy(@intCast(x.shape[0]), alpha, p.data.ptr, 1, x.data.ptr, 1);
//
//        // r += -alpha*k@p
//        // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
//        const r_ = r.copy() catch |e| return e;
//        defer r_.destroy();
//        blas.cblas_daxpy(@intCast(r_.shape[0]), -alpha, kp.data.ptr, 1, r_.data.ptr, 1);
//
//        // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
//        const a_ = switch (preprocessing) {
//            .None => unreachable,
//            .DS => diagonalScaledVector(T)(&r_, &vars.A) catch |e| return e,
//            .ICD => auxiliaryVector(T)(self.alc_tmp, &r_, &decomposed[0], &decomposed[1]) catch |e| return e,
//        };
//        defer a_.destroy();
//
//        // beta = dot(r_, r_)/dot(r, r)
//        const beta_numerator = blas.cblas_ddot(@intCast(r_.shape[0]), r_.data.ptr, 1, a_.data.ptr, 1);
//        const beta_denominator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, a.data.ptr, 1);
//        const beta: T = beta_numerator / beta_denominator;
//
//        // ここで r, a を r_, a_ で更新
//        @memcpy(r.data, r_.data);
//        @memcpy(a.data, a_.data);
//
//        // p_ = a_ + beta*p
//        // a_にp_のデータが入るので下でcopyすることに注意
//        blas.cblas_daxpy(@intCast(a_.shape[0]), beta, p.data.ptr, 1, a_.data.ptr, 1);
//        // ここで a_ に入った p_ のデータで p を更新
//        @memcpy(p.data, a_.data);
//
//        r_sum = 0;
//        for (r.data) |data| r_sum += @abs(data);
//
//        //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
//        std.debug.print("residual at loop {d}: {d}\n", .{ iter, r_sum });
//    }
//
//    return .{
//        .x = x,
//        .iter = iter,
//        .is_converged = !(self.max_iter == iter),
//        .threshold = self.threshold,
//        .res = r_sum,
//    };
//}

test "cg" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.print("leak check: {any}\n", .{gpa.deinit()});
    const alc = gpa.allocator();
    const a: core.Matrix(f64) = try .from(alc, .{ 4, 4 }, &.{ 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2 });
    defer a.destroy();
    const b: core.Vector(f64) = try .from(alc, .{4}, &.{ 1, -1, 1, -1 });
    defer b.destroy();
    const x: core.Vector(f64) = try .from(alc, .{4}, &.{ 2, -3, 3, -2 });
    defer x.destroy();
    const vars: Vars(f64) = .init(a, b, 1e-6, 1e5, .Silent);
    const res = try solveCG(alc, alc, vars);
    defer res.deinit();
    std.debug.print("res.x.data: {d}\n", .{res.x.data});
    std.debug.print("iteration: {d}\n", .{res.iter});
    for (res.x.data, x.data) |approx, strict| {
        try std.testing.expectApproxEqAbs(approx, strict, 1e-6);
    }
}
