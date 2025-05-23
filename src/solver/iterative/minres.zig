const std = @import("std");

const core = @import("../../core.zig");
const math = @import("../../math.zig");
const root = @import("./bind.zig");

const Error = root.Error;
const Result = root.Result;
const Vars = root.Vars;
const Preconditioning = root.Preprocessing;

/// 目的
///     > MINRES法
///     > julia IterativeSolvers.jl ( https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl ) のMINRES実装に準ずるが、数値型は実数に固定する
///
/// 入力
///     > alc_obj: std.mem.Allocator
///         >> 関数スコープを抜ける際に解放されないデータに対するアロケータ
///     > vars.A: ar.Dense(T)
///         >> 連立方程式 Ax=b の係数行列 A
///     > vars.b
///         >> 連立方程式 Ax=b の右辺ベクトル b
///
/// 出力
///     > res: Result
///         >> 演算結果
///
/// Target
///     > MINRES method
///     > It conforms to MINRES in julia IterativeSolvers.jl ( https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl ), but only real number is avialable.
///
/// Input
///     > alc_obj: std.mem.Allocator
///         >> Allocator for data which is not released after exiting function
///     > vars.A: ar.Dense(T)
///         >> Coeffcient matrix A in simultaneous equation Ax = b
///     > vars.b
///         >> Answer vector b in simultaneous equation Ax = b
///
/// 出力
///     > res: Result
///         >> Calculation result.
///
fn solveMINRES(alc: std.mem.Allocator, alc_tmp: std.mem.Allocator, vars: Vars(f64)) Error!Result {
    // 変数定義
    // MINRES法では各イテレーション(i)において、3状態(i-1, i, i+1)の変数が要求される
    // メモリコピーを毎回行うのは大変なので、インデックスを変えることでアクセスするデータを変更する
    var prev, var curr, var next = [3]usize{ 0, 1, 2 };

    // 解ベクトル
    var x = core.Vector(f64).ones(alc, vars.b.shape) catch |e| return e;
    errdefer x.destroy();
    //var prng = std.Random.DefaultPrng.init(42); // ランダム化要らない気もする
    //for (0..x.data.len) |i| x.data[i] = prng.random().floatNorm(f64);

    // クリロフ部分空間基底ベクトル群
    const v: [3]core.Vector(f64) = .{
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
    };
    defer for (v) |arr| arr.destroy();

    // 解更新用ベクトル W = R * V^-1
    const w: [3]core.Vector(f64) = .{
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
        core.Vector(f64).any(alc_tmp, vars.b.shape) catch |e| return e,
    };
    defer for (w) |arr| arr.destroy();

    // 三重対角行列 (Hessenberg 行列) の active column
    var H: [4]f64 = .{ 0, 0, 0, 0 };
    // 右辺ベクトルの最初の2成分
    var rhs: [2]f64 = undefined;

    // Givens回転係数
    var c_prev: f64 = 1;
    var s_prev: f64 = 0;
    var c_curr: f64 = 1;
    var s_curr: f64 = 0;

    // 一時保存変数
    var resnorm: f64 = 0;
    var iter: usize = 0;

    // ---------------- Initialization ----------------

    // 初期残差ベクトル(第一クリロフ基底ベクトル)の計算: v[cur] = b-Ax[prev] = -1.0*Ax+b
    @memcpy(v[curr].data, vars.b.data);
    math.gemv(f64)(vars.a, x, v[curr], -1, 1) catch |e| return e;

    // resnorm = norm(v[curr])
    for (v[curr].data) |val| resnorm += val * val;
    resnorm = std.math.sqrt(resnorm);

    // rhs を resnorm で更新
    rhs = .{ resnorm, 0 };

    // 初期残差ベクトルを resnorm で正規化
    math.scal(f64)(v[curr], 1 / resnorm) catch |e| return e;

    // ループ処理

    while (resnorm > vars.threshold and iter < vars.max_iteration) : (iter += 1) {

        // v[next] = A * v[curr] - H[1] * v[prev]
        math.gemv(f64)(vars.a, v[curr], v[next], 1, 0) catch |e| return e;
        if (0 < iter) math.axpy(f64)(v[prev], v[next], -H[1]) catch |e| return e;

        // v[next] の直交化 (?)
        const proj: f64 = math.dot(f64)(v[curr], v[next]) catch |e| return e;
        H[2] = proj;
        math.axpy(f64)(v[curr], v[next], -proj) catch |e| return e;

        // Normalize v[next]
        H[3] = 0;
        for (v[next].data) |val| H[3] += val * val;
        H[3] = std.math.sqrt(H[3]);
        math.scal(f64)(v[next], 1 / H[3]) catch |e| return e;

        // Rotation on H[0] and H[1]
        if (1 < iter) {
            H[0] = s_prev * H[1];
            H[1] = c_prev * H[1];
        }

        // Rotation on H[1] and H[2]
        if (0 < iter) {
            const tmp = -s_curr * H[1] + c_curr * H[2];
            H[1] = c_curr * H[1] + s_curr * H[2];
            H[2] = tmp;
        }

        // Givens rotation [[c, s], [-s, c]] * [a, b] = [r, 0] (だと思われる)
        const r = std.math.sqrt(H[2] * H[2] + H[3] * H[3]);
        const c = H[2] / r;
        const s = H[3] / r;
        H[2] = r;

        // rhs も同様に回転させる
        rhs[1] = -s * rhs[0];
        rhs[0] = c * rhs[0];

        // W = V * inv(R)
        // 実際には w[next] = (v[curr] - H[1] * w[curr] - H[0] * w[prev]) / H[2]
        @memcpy(w[next].data, v[curr].data);
        if (0 < iter) math.axpy(f64)(w[curr], w[next], -H[1]) catch |e| return e;
        if (1 < iter) math.axpy(f64)(w[prev], w[next], -H[0]) catch |e| return e;
        math.scal(f64)(w[next], 1 / H[2]) catch |e| return e;

        // x += rhs[0] * w[next]
        math.axpy(f64)(w[next], x, rhs[0]) catch |e| return e;

        // 次イテレーションへの準備
        prev, curr, next = [3]usize{ curr, next, prev };
        c_prev, s_prev, c_curr, s_curr = [4]f64{ c_curr, s_curr, c, s };
        rhs[0] = rhs[1];
        H[1] = H[3];

        // 近似残差ノルム
        resnorm = @abs(rhs[1]);

        //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
        if (vars.verbosity != .Silent) std.debug.print("residual at loop {d}: {d}\n", .{ iter, resnorm });
    }

    return .{
        .x = x,
        .iter = iter,
        .is_converged = !(vars.max_iteration == iter),
        .threshold = vars.threshold,
        .res = resnorm,
    };
}

test "minres" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.print("leak check: {any}\n", .{gpa.deinit()});
    const alc = gpa.allocator();
    const a: core.Matrix(f64) = try .from(alc, .{ 4, 4 }, &.{ 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2 });
    defer a.destroy();
    const b: core.Vector(f64) = try .from(alc, .{4}, &.{ 1, -1, 1, -1 });
    defer b.destroy();
    const x: core.Vector(f64) = try .from(alc, .{4}, &.{ 2, -3, 3, -2 });
    defer x.destroy();
    //const a: core.Matrix(f64) = try .from(alc, .{ 4, 4 }, &.{ 1, 2, 3, 4, 5, -6, 7, -8, -9, 10, -11, 12, 13, 14, 15, 16 });
    //defer a.destroy();
    //const b: core.Vector(f64) = try .from(alc, .{4}, &.{ 1, -2, 3, -4 });
    //defer b.destroy();
    //const x: core.Vector(f64) = try .from(alc, .{4}, &.{ 0, 2, 1, 0 });
    //defer x.destroy();
    const vars: Vars(f64) = .init(a, b, 1e-6, 1e5, .Silent);
    const res = try solveMINRES(alc, alc, vars);
    defer res.deinit();
    std.debug.print("res.x.data: {d}\n", .{res.x.data});
    std.debug.print("iteration: {d}\n", .{res.iter});
    for (res.x.data, x.data) |approx, strict| {
        try std.testing.expectApproxEqAbs(approx, strict, 1e-6);
    }
}
