const std = @import("std");
const blas = @cImport({
    @cInclude("cblas.h");
});
const petsc = @cImport({
    @cInclude("petscksp.h");
});

const unit = @import("../../units/bind.zig");
const sys = @import("../../systems/bind.zig");

const ar = unit.math.array;

pub const ForwardError = error{
    TypeNotImplemented,
};

pub const Error = sys.Error || ar.AllError || unit.sim.MeshError || ForwardError;

/// DS: Diagonal Scaling
/// ICD: IncompleteCholeskyDecomposition
pub const Preprocessing = enum {
    None,
    DS,
    ICD,
};

/// None: No constraint
/// LM: Lagrange Multiplier for electrode
/// LMS: Lagrange Multiplier for electrode, SchurComplementForm
pub const Constraint = enum {
    None,
    LM,
    LMS,
};

/// CG: Conjugate Gradient method
/// MINRES: Minimization Residual method
pub const Solver = enum {
    CG,
    MINRES,
};

pub const Verbosity = enum {
    Silent,
    Result,
    Full,
};

/// 目的
///     Ax=b
fn Vars(comptime T: type) type {
    return struct {
        A: ar.Dense(T, 2),
        b: ar.Dense(T, 1),

        pub fn init(A: ar.Dense(T), b: ar.Dense(T)) ForwardError!@This() {
            switch (@typeInfo(T)) {
                .Float => |info| {
                    switch (info.bits) {
                        64 => {},
                        else => return ForwardError.TypeNotImplemented,
                    }
                },
                else => return ForwardError.TypeNotImplemented,
            }
            return .{
                .A = A,
                .b = b,
            };
        }

        pub fn deinit(self: @This()) void {
            self.A.destroy();
            self.b.destroy();
        }
    };
}

pub fn Forward(comptime T: type) type {
    return struct {
        alc_tmp: std.mem.Allocator,
        solver: Solver,
        constraint: Constraint,
        pre_processing: Preprocessing,
        threshold: T,
        max_iter: usize,
        verbosity: Verbosity = .Full,

        pub fn init(alc_tmp: std.mem.Allocator, solver: Solver, constraint: Constraint, pre_processing: Preprocessing, threshold: T, max_iter: usize) ForwardError!@This() {
            switch (@typeInfo(T)) {
                .float => |info| {
                    switch (info.bits) {
                        64 => {},
                        else => return ForwardError.TypeNotImplemented,
                    }
                },
                else => return ForwardError.TypeNotImplemented,
            }
            return .{
                .alc_tmp = alc_tmp,
                .solver = solver,
                .constraint = constraint,
                .pre_processing = pre_processing,
                .threshold = threshold,
                .max_iter = max_iter,
            };
        }

        /// 目的
        ///     設定された手法でFEMを実行
        pub fn calculate(self: @This(), alc_obj: std.mem.Allocator, mesh: *const sys.sim.mesh.FEM, impedances: ?[]T, currents: ?[]T) Error!unit.sim.fem.Result {
            const vars = switch (self.constraint) {
                .None => self.varsCG(mesh) catch |e| return e,
                .LM => self.varsCGLM(mesh, impedances.?, currents.?) catch |e| return e,
                .LMS => self.varsCGLM(mesh, impedances.?, currents.?) catch |e| return e,
            };
            defer vars.deinit();
            return switch (self.solver) {
                .CG => switch (self.pre_processing) {
                    .None => self.solveCG(alc_obj, &vars) catch |e| return e,
                    .DS => self.solvePCG(.DS, alc_obj, &vars, mesh) catch |e| return e,
                    .ICD => self.solvePCG(.ICD, alc_obj, &vars, mesh) catch |e| return e,
                },
                .MINRES => switch (self.pre_processing) {
                    .None => self.solveMINRES(alc_obj, &vars) catch |e| return e,
                    .DS => ForwardError.TypeNotImplemented,
                    .ICD => ForwardError.TypeNotImplemented,
                },
            };
        }

        /// Target
        ///     > 連立方程式 Ax=b w/o B.C. の変数 A, b の作成
        ///     > 係数行列は .axes = .{ノード数, ノード数}
        fn varsCG(self: @This(), mesh: *const sys.sim.mesh.FEM) ar.ArrayError!Vars(T) {
            const b = ar.Dense(T, 1).any(self.alc_tmp, &.{mesh.num_vertices}) catch |e| return e;
            // 電流密度をロード
            for (mesh.vertices.items(.id), 0..) |id, idx| b.data[id orelse continue] = mesh.vertices.items(.current)[idx] orelse 0;

            const A = stiffnessMatrix(self.alc_tmp, mesh) catch |e| return e;

            for (0..A.shape[0]) |i| if (0 >= (A.get(&.{ i, i }) catch unreachable)) std.debug.panic("A[{d}, {d}] = {d} is not positive\n", .{ i, i, A.get(&.{ i, i }) catch unreachable });

            return .{
                .A = A,
                .b = b,
            };
        }

        /// Target
        ///     > 連立方程式 Ax=b w/ B.C.(ラグランジュの未定定数法による電極直下電位束縛)の変数 A, b の作成
        ///     > 係数行列は .axes = .{ノード数 + 電極ノード数 + 電極数, ノード数 + 電極ノード数 + 電極数}
        ///     > 導出
        ///         >> Ax = b
        ///         >> A = [
        ///         >>   [K, P.T, 0],
        ///         >>   [P, -M, -Q.T],
        ///         >>   [0, -Q, 0],
        ///         >> ]
        ///         >> x = [V_node, -I_node_electrode, V_electrode]
        ///         >> b = [0, 0, I_electrode]
        ///         >> ここで、各記号は以下の通り
        ///         >> K[σ]: 全体剛性行列 (shape = [ノード(node)数, ノード数]) (導電率は各エレメントの剛性行列に対して乗算される)
        ///         >> P[無次元]: 電極ノード選択行列 (shape = [電極ノード(node_electrode)数, ノード数])
        ///         >> M[Ω]: 接触インピーダンス (shape = [電極ノード数, 電極ノード数]) (対角行列)
        ///         >> Q[無次元]: 電極選択行列 (shape = [電極(electrode)数, 電極ノード数])
        ///         >> V_node[V]: ノード電位 (shape = [ノード数])
        ///         >> I_node_electrode[Q]: 電極ノード電流 (shape = [電極ノード数])
        ///         >> V_electrode[V]: 電極電位 (shape = [電極数])
        ///         >> I_electrode[A]: 電極電流 (shape = [電極数])
        fn varsCGLM(
            self: @This(),
            mesh: *const sys.sim.mesh.FEM,
            impedances: []T,
            currents: []T,
        ) ar.ArrayError!Vars(T) {
            // null != id_node: そのdata_indexにid_node.?のnode(ノード)がある
            const ids_node: []?usize = mesh.vertices.items(.id);
            // null != id_elec: そのdata_indexにid_elec.?のnode_elec(電極ノード)がある
            const ids_elec: []?usize = mesh.vertices.items(.electrode);

            const num_node = mesh.num_vertices;
            const num_node_elec = impedances.len;
            const num_elec = currents.len;
            std.debug.print("num_node: {d}\n", .{num_node});
            std.debug.print("num_node_elec: {d}\n", .{num_node_elec});
            std.debug.print("num_elec: {d}\n", .{num_elec});

            // ---------------- Initialization ----------------

            // A: 係数行列
            const A = ar.Dense(T, 2).zeros(self.alc_tmp, &.{ num_node + num_node_elec + num_elec, num_node + num_node_elec + num_elec }) catch |e| return e;

            // b: 解ベクトル
            const b = ar.Dense(T, 1).zeros(self.alc_tmp, &.{num_node + num_node_elec + num_elec}) catch |e| return e;

            // P: 電極ノード選択行列
            var P = ar.Dense(T, 2).zeros(self.alc_tmp, &.{ num_node_elec, num_node }) catch |e| return e;
            defer P.destroy();

            // PT: 電極ノード選択行列の転置
            var PT = ar.Dense(T, 2).zeros(self.alc_tmp, &.{ num_node, num_node_elec }) catch |e| return e;
            defer PT.destroy();

            // Q: 電極選択行列
            var Q = ar.Dense(T, 2).zeros(self.alc_tmp, &.{ num_elec, num_node_elec }) catch |e| return e;
            defer Q.destroy();

            // QT: 電極選択行列の転置
            var QT = ar.Dense(T, 2).zeros(self.alc_tmp, &.{ num_node_elec, num_elec }) catch |e| return e;
            defer QT.destroy();

            // K: 全体剛性行列 (初期化ではなく生成済みの行列を持ってくる)
            const K = stiffnessMatrix(self.alc_tmp, mesh) catch |e| return e;
            defer K.destroy();

            // ---------------- Load data ----------------

            // b: 電極電流をロード
            for (0.., currents) |idx, I| b.data[num_node + num_node_elec + idx] = I;

            // P: ノードid(列インデックス)と電極ノードid(行インデックス)の対応
            var id_node_elec: usize = 0;
            for (ids_node, ids_elec) |id_node, id_elec| {
                if (null == id_elec) continue;
                P.set(&.{ id_node_elec, id_node.? }, 1) catch |e| return e;
                PT.set(&.{ id_node.?, id_node_elec }, 1) catch |e| return e;
                id_node_elec += 1;
            }

            // Q: ノードid(列インデックス)と電極ノードid(行インデックス)の対応
            id_node_elec = 0;
            for (ids_elec) |id_elec| {
                if (null == id_elec) continue;
                Q.set(&.{ id_elec.?, id_node_elec }, -1) catch |e| return e;
                QT.set(&.{ id_node_elec, id_elec.? }, -1) catch |e| return e;
                id_node_elec += 1;
            }

            // ---------------- Construct A ----------------
            //
            // ブロック行列であることを活かし、ブロック毎にデータをロード

            // 1列目: K, P, 0
            for (0..num_node) |c| {
                // 1行目: K
                @memcpy(
                    A.data[c * A.shape[0] .. c * A.shape[0] + num_node],
                    K.data[c * K.shape[0] .. (c + 1) * K.shape[0]],
                );
                // 2行目: P
                @memcpy(
                    A.data[c * A.shape[0] + num_node .. c * A.shape[0] + num_node + num_node_elec],
                    P.data[c * P.shape[0] .. (c + 1) * P.shape[0]],
                );
                // 3行目: 0
            }

            // 2列目: P.T, -M, -Q
            for (0..num_node_elec) |c| {
                // 1列目: P.T
                @memcpy(
                    A.data[(num_node + c) * A.shape[0] .. (num_node + c) * A.shape[0] + num_node],
                    PT.data[c * PT.shape[0] .. (c + 1) * PT.shape[0]],
                );
                // 2列目: -M (対角成分の更新)
                A.data[(num_node + c) * A.shape[0] + num_node + c] = -impedances[c];
                // 3列目: -Q
                @memcpy(
                    A.data[(num_node + c) * A.shape[0] + num_node + num_node_elec .. (num_node + c + 1) * A.shape[0]],
                    Q.data[c * Q.shape[0] .. (c + 1) * Q.shape[0]],
                );
            }

            // 3行目: 0, -Q.T, 0
            for (0..num_elec) |c| {
                // 1列目: 0
                // 2列目: -Q.T
                @memcpy(
                    A.data[(num_node + num_node_elec + c) * A.shape[0] + num_node .. (num_node + num_node_elec + c) * A.shape[0] + num_node + num_node_elec],
                    QT.data[c * QT.shape[0] .. (c + 1) * QT.shape[0]],
                );
                // 3列目: 0 (or εI)
                //A.data[(num_node + num_node_elec + r) * A.shape[1] + num_node + num_node_elec + r] = 1e-9;
            }

            return .{
                .A = A,
                .b = b,
            };
        }

        // TODO: varsCGLM()同様、以下のドキュメントに従うようにアップデート
        /// Target
        ///     > 連立方程式 Ax=b w/ B.C.(ラグランジュの未定定数法による電極直下電位束縛)の変数 A, b の作成
        ///     > 収束安定化を図る為、係数行列の剛性行列以外の成分はSchur補行列によって畳み込まれる
        ///     > 係数行列は .axes = .{ ノード数, ノード数 }
        ///     > 導出
        ///         >> A'x' = b'
        ///         >> A' = [
        ///         >>   [K, P.T, 0],
        ///         >>   [P, -M, -Q.T],
        ///         >>   [0, -Q, 0],
        ///         >> ]
        ///         >> x = [V_node, -I_node_electrode, V_electrode]
        ///         >> b = [0, 0, I_electrode]
        ///         >> A[Ω^-1] = K - [[P.T, 0]] * [[-M, -Q.T], [-Q, 0]] * [[P], [0]]
        ///         >> x[V] = V_node
        ///         >> b[I] = 0
        ///
        ///     > 各記号の意味
        ///         >> K[σ]: 全体剛性行列 (shape = [ノード(node)数, ノード数]) (導電率は各エレメントの剛性行列に対して乗算される)
        ///         >> P[無次元]: 電極ノード選択行列 (shape = [電極ノード(node_electrode)数, ノード数])
        ///         >> M[Ω]: 接触インピーダンス (shape = [電極ノード数, 電極ノード数]) (対角行列)
        ///         >> Q[無次元]: 電極選択行列 (shape = [電極(electrode)数, 電極ノード数])
        ///         >> V_node[V]: ノード電位 (shape = [ノード数])
        ///         >> I_node_electrode[Q]: 電極ノード電流 (shape = [電極ノード数])
        ///         >> V_electrode[V]: 電極電位 (shape = [電極数])
        ///         >> I_electrode[A]: 電極電流 (shape = [電極数])
        fn varsCGLMS(self: @This(), mesh: *const sys.sim.mesh.FEM, num_electrode: usize, contact_impedances: []T) ar.ArrayError!Vars(T) {
            const ids = mesh.vertices.items(.id);

            const b = ar.Dense(T).zeros(self.alc_tmp, &.{mesh.num_vertices}) catch |e| return e;

            // 電流密度をロード
            for (mesh.vertices.items(.follow_pt_info), ids, 0..) |flag, id, idx| {
                if (null == flag) continue;
                // 電極でないなら(電流が無いなら)0
                b.data[id.?] = mesh.vertices.items(.current)[idx] orelse 0;
            }

            var C = ar.Dense(T).zeros(self.alc_tmp, &.{ num_electrode, mesh.num_vertices }) catch |e| return e;
            defer C.destroy();

            for (0..mesh.vertices.len) |i| {
                const id = ids[i] orelse continue;
                const n_elec = mesh.vertices.items(.electrode)[i] orelse continue;
                C.set(&.{ n_elec, id }, 1) catch |e| return e;
            }

            const A = stiffnessMatrix(self.alc_tmp, mesh) catch |e| return e;

            const M = ar.Dense(T).zeros(self.alc_tmp, &.{ num_electrode, num_electrode }) catch |e| return e;
            defer M.destroy();
            for (0..num_electrode) |i| {
                M.data[i * num_electrode + i] = -contact_impedances[i];
            }

            const CM = ar.Dense(T).any(self.alc_tmp, &.{ mesh.num_vertices, num_electrode }) catch |e| return e;
            defer CM.destroy();
            blas.cblas_dgemm(blas.CblasColMajor, blas.CblasTrans, blas.CblasNoTrans, @intCast(mesh.num_vertices), @intCast(num_electrode), @intCast(num_electrode), 1.0, C.data.ptr, @intCast(mesh.num_vertices), M.data.ptr, @intCast(num_electrode), 1.0, CM.data.ptr, @intCast(num_electrode));

            const CMC = ar.Dense(T).any(self.alc_tmp, &.{ mesh.num_vertices, mesh.num_vertices }) catch |e| return e;
            defer CMC.destroy();
            blas.cblas_dgemm(blas.CblasColMajor, blas.CblasNoTrans, blas.CblasNoTrans, @intCast(mesh.num_vertices), @intCast(mesh.num_vertices), @intCast(num_electrode), 1.0, CM.data.ptr, @intCast(num_electrode), C.data.ptr, @intCast(mesh.num_vertices), 1.0, CMC.data.ptr, @intCast(mesh.num_vertices));

            if (A.data.len != CMC.data.len) unreachable;
            for (0..A.data.len) |i| A.data[i] += CMC.data[i];

            return .{
                .A = A,
                .b = b,
            };
        }

        /// Target
        ///     > MINRES法に基づく頂点電位計算
        ///     > julia IterativeSolvers.jl ( https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl ) のMINRES実装に準ずるが、数値型は実数に固定する
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
        fn solveMINRES(self: @This(), alc_obj: std.mem.Allocator, vars: *const Vars(T)) ar.ArrayError!unit.sim.fem.Result {
            // 変数定義
            // MINRES法では各イテレーション(i)において、3状態(i-1, i, i+1)の変数が要求される
            // メモリコピーを毎回行うのは大変なので、インデックスを変えることでアクセスするデータを変更する
            var prev, var curr, var next = [3]usize{ 0, 1, 2 };

            // 解ベクトル
            var x = ar.Dense(f64, 1).ones(alc_obj, &vars.b.shape) catch |e| return e;
            errdefer x.destroy();
            var prng = std.Random.DefaultPrng.init(42); // ランダム化要らない気もする
            for (0..x.data.len) |i| x.data[i] = prng.random().floatNorm(f64);

            // クリロフ部分空間基底ベクトル群
            const v: [3]ar.Dense(f64, 1) = .{
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
            };
            defer for (v) |arr| arr.destroy();

            // 解更新用ベクトル W = R * V^-1
            const w: [3]ar.Dense(f64, 1) = .{
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
                ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e,
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
            // daxpy() は y に解を代入するので、v を b にコピーしてから計算
            @memcpy(v[curr].data, vars.b.data);
            ar.gemv(f64)(&vars.A, &x, &v[curr], -1, 1) catch |e| return e;

            // resnorm = norm(v[curr])
            for (v[curr].data) |val| resnorm += val * val;
            resnorm = std.math.sqrt(resnorm);

            // rhs を resnorm で更新
            rhs = .{ resnorm, 0 };

            // 初期残差ベクトルを resnorm で正規化
            ar.scal(f64)(&v[curr], 1 / resnorm) catch |e| return e;

            // ループ処理

            while (resnorm > self.threshold and iter < self.max_iter) : (iter += 1) {

                // v[next] = A * v[curr] - H[1] * v[prev]
                ar.gemv(f64)(&vars.A, &v[curr], &v[next], 1, 0) catch |e| return e;
                if (0 < iter) ar.axpy(f64)(&v[prev], &v[next], -H[1]) catch |e| return e;

                // v[next] の直交化 (?)
                const proj: f64 = ar.dot(f64)(&v[curr], &v[next]) catch |e| return e;
                H[2] = proj;
                ar.axpy(f64)(&v[curr], &v[next], -proj) catch |e| return e;

                // Normalize v[next]
                H[3] = 0;
                for (v[next].data) |val| H[3] += val * val;
                H[3] = std.math.sqrt(H[3]);
                ar.scal(f64)(&v[next], 1 / H[3]) catch |e| return e;

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
                if (0 < iter) ar.axpy(f64)(&w[curr], &w[next], -H[1]) catch |e| return e;
                if (1 < iter) ar.axpy(f64)(&w[prev], &w[next], -H[0]) catch |e| return e;
                ar.scal(f64)(&w[next], 1 / H[2]) catch |e| return e;

                // x += rhs[0] * w[next]
                ar.axpy(f64)(&w[next], &x, rhs[0]) catch |e| return e;

                // 次イテレーションへの準備
                prev, curr, next = [3]usize{ curr, next, prev };
                c_prev, s_prev, c_curr, s_curr = [4]f64{ c_curr, s_curr, c, s };
                rhs[0] = rhs[1];
                H[1] = H[3];

                // 近似残差ノルム
                resnorm = @abs(rhs[1]);

                //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
                std.debug.print("residual at loop {d}: {d}\n", .{ iter, resnorm });
            }

            return .{
                .x = x,
                .iter = iter,
                .is_converged = !(self.max_iter == iter),
                .threshold = self.threshold,
                .res = resnorm,
            };
        }

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
        fn solveCG(self: @This(), alc_obj: std.mem.Allocator, vars: *const Vars(T)) ar.ArrayError!unit.sim.fem.Result {
            // 解xは中身だけが書き換わる
            const x = ar.Dense(f64, 1).any(alc_obj, &vars.b.shape) catch |e| return e;
            errdefer x.destroy();
            var prng = std.Random.DefaultPrng.init(42);
            for (0..x.data.len) |i| x.data[i] = prng.random().floatNorm(f64);

            // Axはr0を計算するのに必要なだけ
            const Ax = ar.Dense(f64, 1).any(self.alc_tmp, &vars.b.shape) catch |e| return e;
            defer Ax.destroy();
            blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(x.shape[0]), x.data.ptr, 1, 0.0, Ax.data.ptr, 1);

            // r = b-Ax = -1.0*Ax+b
            // j に r のデータが入るので改めてcopyすることに注意
            const r = ar.Dense(f64, 1).from(self.alc_tmp, &vars.b.shape, vars.b.data) catch |e| return e;
            defer r.destroy();
            blas.cblas_daxpy(@intCast(Ax.shape[0]), -1.0, Ax.data.ptr, 1, r.data.ptr, 1);

            // p0 = r0
            var p = r.copy() catch |e| return e;
            defer p.destroy();

            var r_sum: f64 = 0;
            var i: usize = 0;
            for (r.data) |item| r_sum += @abs(item);
            while (r_sum > self.threshold and i < self.max_iter) : (i += 1) {
                // kpはループ内のalphaを計算するのに必要
                const kp = ar.Dense(f64, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
                defer kp.destroy();
                blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(p.shape[0]), p.data.ptr, 1, 0.0, kp.data.ptr, 1);

                // alpha = dot(r, r)/dot(kp, p)
                const alpha_numerator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, r.data.ptr, 1);
                const alpha_denominator = blas.cblas_ddot(@intCast(kp.shape[0]), kp.data.ptr, 1, p.data.ptr, 1);
                const alpha: f64 = alpha_numerator / alpha_denominator;

                // v += alpha*p
                blas.cblas_daxpy(@intCast(x.shape[0]), alpha, p.data.ptr, 1, x.data.ptr, 1);

                // r += -alpha*k@p
                // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
                const r_ = r.copy() catch |e| return e;
                defer r_.destroy();
                blas.cblas_daxpy(@intCast(r_.shape[0]), -alpha, kp.data.ptr, 1, r_.data.ptr, 1);

                // beta = dot(r_, r_)/dot(r, r)
                const beta_numerator = blas.cblas_ddot(@intCast(r_.shape[0]), r_.data.ptr, 1, r_.data.ptr, 1);
                const beta_denominator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, r.data.ptr, 1);
                const beta: f64 = beta_numerator / beta_denominator;

                // ここで r を r_ で更新
                @memcpy(r.data, r_.data);

                // p_ = r_ + beta*p
                // r_にp_のデータが入るので下でcopyすることに注意
                blas.cblas_daxpy(@intCast(r_.shape[0]), beta, p.data.ptr, 1, r_.data.ptr, 1);
                // ここで r_ に入った p_ のデータで p を更新
                @memcpy(p.data, r_.data);

                r_sum = 0;
                for (r.data) |data| r_sum += @abs(data);

                //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
                std.debug.print("residual at loop {d}: {d}\n", .{ i, r_sum });
            }

            return .{
                .x = x,
                .iter = i,
                .is_converged = !(self.max_iter == i),
                .threshold = self.threshold,
                .res = r_sum,
            };
        }

        /// Target
        ///     > 前処理(Preprocessing)付き共役勾配法(Conjugate Gradient Method)に基づく頂点電位計算
        ///     > 前処理の候補
        ///         >> .ICD: 不完全(修正)コレスキー分解(Incomplete Cholesky Decomposition)
        ///         >> .DS: 対角スケーリング(Diagonal Scaling)
        ///     > 前処理の有無でアルゴリズムが異なるため、前処理のない共役勾配法はsolveCG()を使う
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
        fn solvePCG(self: @This(), preprocessing: Preprocessing, alc_obj: std.mem.Allocator, vars: *const Vars(T), mesh: *const sys.sim.mesh.FEM) ar.ArrayError!unit.sim.fem.Result {
            // 電圧vは中身だけが書き換わる
            const x = ar.Dense(T, 1).ones(alc_obj, &.{vars.b.data.len}) catch |e| return e;
            errdefer x.destroy();

            // kv = k@v
            // kvは最初のr0を計算するのに必要なだけ
            const kv = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
            defer kv.destroy();
            blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(x.shape[0]), x.data.ptr, 1, 0.0, kv.data.ptr, 1);
            // TODO: beta: 1.0 -> 0.0に変更

            // r = b-kv = -1.0*kv+b
            // 残差rも中身だけが書き換わる
            // b に r のデータが入るので改めてcopyすることに注意
            const r = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
            defer r.destroy();
            blas.cblas_daxpy(@intCast(kv.shape[0]), -1.0, kv.data.ptr, 1, vars.b.data.ptr, 1);

            // ここで r を j で更新 (必ずaがrをコピってくる後じゃないと動かない)
            @memcpy(r.data, vars.b.data);

            var non_zero_indices = switch (preprocessing) {
                .ICD => mesh.nonZeroIndices(self.alc_tmp) catch unreachable,
                else => undefined,
            };
            defer switch (preprocessing) {
                .ICD => non_zero_indices.deinit(),
                else => {},
            };
            // LUz = r is decomposited to Ly = r & Uz = y
            const decomposed = switch (preprocessing) {
                .ICD => incompleteCholeskyDecomposition(T)(self.alc_tmp, &vars.A, non_zero_indices.items),
                else => undefined,
            };

            defer switch (preprocessing) {
                .ICD => for (decomposed) |arr| arr.destroy(),
                else => {},
            };

            const a = switch (preprocessing) {
                .None => unreachable,
                .DS => diagonalScaledVector(T)(&r, &vars.A) catch |e| return e,
                .ICD => auxiliaryVector(T)(self.alc_tmp, &r, &decomposed[0], &decomposed[1]) catch |e| return e,
            };
            defer a.destroy();

            // 共役勾配法: p <- r
            // 前処理付き共役勾配法: p <- a
            var p = a.copy() catch |e| return e;
            defer p.destroy();

            var r_sum: T = 0;
            for (r.data) |data| r_sum += @abs(data);

            var iter: usize = 0;
            while (r_sum > self.threshold and iter < self.max_iter) : (iter += 1) {
                // kpはループ内のalphaを計算するのに必要
                const kp = ar.Dense(T, 1).any(self.alc_tmp, &.{vars.b.data.len}) catch |e| return e;
                defer kp.destroy();
                blas.cblas_dgemv(blas.CblasColMajor, blas.CblasNoTrans, @intCast(vars.A.shape[0]), @intCast(vars.A.shape[1]), 1.0, vars.A.data.ptr, @intCast(p.shape[0]), p.data.ptr, 1, 0.0, kp.data.ptr, 1);

                // 共役勾配法: alpha = dot(r, r)/dot(kp, p)
                // 前処理付き共役勾配法: alpha = dot(r, a)/dot(kp, p)
                const alpha_numerator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, a.data.ptr, 1);
                const alpha_denominator = blas.cblas_ddot(@intCast(kp.shape[0]), kp.data.ptr, 1, p.data.ptr, 1);
                const alpha: T = alpha_numerator / alpha_denominator;

                // v += alpha*p
                blas.cblas_daxpy(@intCast(x.shape[0]), alpha, p.data.ptr, 1, x.data.ptr, 1);

                // r += -alpha*k@p
                // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
                const r_ = r.copy() catch |e| return e;
                defer r_.destroy();
                blas.cblas_daxpy(@intCast(r_.shape[0]), -alpha, kp.data.ptr, 1, r_.data.ptr, 1);

                // beta計算のために一時的に r_k+1 = r_ と r_k = r を分ける必要がある
                const a_ = switch (preprocessing) {
                    .None => unreachable,
                    .DS => diagonalScaledVector(T)(&r_, &vars.A) catch |e| return e,
                    .ICD => auxiliaryVector(T)(self.alc_tmp, &r_, &decomposed[0], &decomposed[1]) catch |e| return e,
                };
                defer a_.destroy();

                // beta = dot(r_, r_)/dot(r, r)
                const beta_numerator = blas.cblas_ddot(@intCast(r_.shape[0]), r_.data.ptr, 1, a_.data.ptr, 1);
                const beta_denominator = blas.cblas_ddot(@intCast(r.shape[0]), r.data.ptr, 1, a.data.ptr, 1);
                const beta: T = beta_numerator / beta_denominator;

                // ここで r, a を r_, a_ で更新
                @memcpy(r.data, r_.data);
                @memcpy(a.data, a_.data);

                // p_ = a_ + beta*p
                // a_にp_のデータが入るので下でcopyすることに注意
                blas.cblas_daxpy(@intCast(a_.shape[0]), beta, p.data.ptr, 1, a_.data.ptr, 1);
                // ここで a_ に入った p_ のデータで p を更新
                @memcpy(p.data, a_.data);

                r_sum = 0;
                for (r.data) |data| r_sum += @abs(data);

                //if (0 == std.math.mod(usize, iter, 1000) catch unreachable) std.debug.print("r_sum at loop {d}: {d}\n", .{ iter, resnorm });
                std.debug.print("residual at loop {d}: {d}\n", .{ iter, r_sum });
            }

            return .{
                .x = x,
                .iter = iter,
                .is_converged = !(self.max_iter == iter),
                .threshold = self.threshold,
                .res = r_sum,
            };
        }
    };
}

/// 目的
///     不完全コレスキー分解(A' = LDL.T)の実行
/// 返り値
///     L: 下三角行列
///     DL.T: 対角行列DとLの転置(L.T)の積
/// 注意点
///     厳密には修正不完全コレスキー分解
/// 参考
///     https://pbcglab.jp/cgi-bin/wiki/
fn incompleteCholeskyDecomposition(T: type) fn (alc: std.mem.Allocator, A: *const ar.Dense(T, 2), non_zero_indices: []const usize) [2]ar.Dense(T, 2) {
    return struct {
        fn f(tmp_alc: std.mem.Allocator, A: *const ar.Dense(T, 2), non_zero_indices: []const usize) [2]ar.Dense(T, 2) {
            const n = A.shape[0];
            var L = ar.Dense(T, 2).zeros(tmp_alc, &A.shape) catch unreachable;
            var D = ar.Dense(T, 2).zeros(tmp_alc, &A.shape) catch unreachable;
            defer D.destroy();

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

            //// スパース性を用いないバージョン
            //_ = non_zero_indices;
            //for (1..n) |i| {
            //    for (0..i + 1) |j| {
            //        var ldl: f64 = A.get(&.{ i, j }) catch unreachable;
            //        if (1e-12 > ldl) continue;
            //        for (0..j) |k| {
            //            const l_ik = L.get(&.{ i, k }) catch unreachable;
            //            const d_kk = D.get(&.{ k, k }) catch unreachable;
            //            const l_kj = L.get(&.{ k, j }) catch unreachable;
            //            ldl -= l_ik * d_kk * l_kj;
            //        }
            //        L.set(&.{ i, j }, ldl) catch unreachable;
            //    }
            //    D.set(&.{ i, i }, 1 / (L.get(&.{ i, i }) catch unreachable)) catch unreachable;
            //}

            const DL = ar.Dense(T, 2).any(tmp_alc, &A.shape) catch unreachable;
            blas.cblas_dgemm(blas.CblasColMajor, blas.CblasNoTrans, blas.CblasTrans, @intCast(n), @intCast(n), @intCast(n), 1.0, D.data.ptr, @intCast(n), L.data.ptr, @intCast(n), 0.0, DL.data.ptr, @intCast(n));

            return .{ L, DL };
        }
    }.f;
}

/// 目的
///     対角スケーリング(a = r/trace(A))の実行
///     平方根を用いているので厳密には違うのだが、こっちのほうがずっと高速かつ振動無く収束するので採用
/// 注意点
///     アロケータはrのアロケータを使用
fn diagonalScaledVector(T: type) fn (r: *const ar.Dense(T, 1), A: *const ar.Dense(T, 2)) ar.DataError!ar.Dense(T, 1) {
    return struct {
        fn f(r: *const ar.Dense(T, 1), A: *const ar.Dense(T, 2)) ar.DataError!ar.Dense(T, 1) {
            const a = r.copy() catch |e| return e;
            for (0..a.data.len) |i| a.data[i] /= (std.math.sqrt(A.get(&.{ i, i }) catch unreachable));
            return a;
        }
    }.f;
}

fn auxiliaryVector(T: type) fn (alc: std.mem.Allocator, r: *const ar.Dense(T, 1), L: *const ar.Dense(T, 2), DL: *const ar.Dense(T, 2)) ar.ArrayError!ar.Dense(T, 1) {
    return struct {
        fn f(alc: std.mem.Allocator, r: *const ar.Dense(T, 1), L: *const ar.Dense(T, 2), DL: *const ar.Dense(T, 2)) ar.ArrayError!ar.Dense(T, 1) {
            const n = r.shape[0];
            const y = ar.Dense(T, 1).zeros(alc, &r.shape) catch |e| return e;
            defer y.destroy();
            const a = ar.Dense(T, 1).zeros(alc, &r.shape) catch |e| return e;
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
    const hoge = try ar.Dense(f64, 2).from(alc, &.{ 3, 3 }, &.{ 7, 1, 2, 1, 8, 0, 2, 0, 9 });
    defer hoge.destroy();
    const L, const D = incompleteCholeskyDecomposition(f64)(alc, &hoge, &.{ 0, 1, 2, 3, 4, 6, 8 });
    defer L.destroy();
    std.debug.print("L: {d}\n", .{L.data});
    defer D.destroy();
    std.debug.print("D: {d}\n", .{D.data});
}

/// Input
///     > ξ: f64
///         >> n = 2: ±√(1/3)
///     > η: f64
///         >> n = 2: ±√(1/3)
///     > pos00, pos10, pos11, pos01: unit.math.z.Vec2_f64
///         >> 四角形エレメントの座標、数字は各軸(x, y)での大小関係を表す
///
/// Output
///     > Jacobian(J): [4]f64 = .{∂x/∂ξ, ∂x/∂η, ∂y/∂ξ, ∂y/∂η}
///         >> 2x2 matrix, column-order
///
/// Target
///     > アイソパラメトリック要素を用いた四角形エレメント(order=1, n=2)によるFEMの実装
///     > 形状関数のξ-η座標系への座標変換のヤコビアンJを導出
///     > Jは(ξ, η)の関数である為、数値積分点における(ξ, η)を代入した値を導出する
fn jacobian(xi: f64, eta: f64, pos00: unit.math.z.Vec2_f64, pos10: unit.math.z.Vec2_f64, pos11: unit.math.z.Vec2_f64, pos01: unit.math.z.Vec2_f64) [4]f64 {
    const a1 = 0.25 * (-pos00.x() + pos10.x() + pos11.x() - pos01.x());
    const a2 = 0.25 * (pos00.x() - pos10.x() + pos11.x() - pos01.x());
    const a3 = 0.25 * (-pos00.x() - pos10.x() + pos11.x() + pos01.x());
    const a4 = 0.25 * (pos00.x() - pos10.x() + pos11.x() - pos01.x());
    const a5 = 0.25 * (-pos00.y() + pos10.y() + pos11.y() - pos01.y());
    const a6 = 0.25 * (pos00.y() - pos10.y() + pos11.y() - pos01.y());
    const a7 = 0.25 * (-pos00.y() - pos10.y() + pos11.y() + pos01.y());
    const a8 = 0.25 * (pos00.y() - pos10.y() + pos11.y() - pos01.y());
    const dxdxi = a1 + a2 * eta;
    const dxdeta = a3 + a4 * xi;
    const dydxi = a5 + a6 * eta;
    const dydeta = a7 + a8 * xi;

    return .{ dxdxi, dxdeta, dydxi, dydeta };
}

/// Input
///     > matrix(m): [4]f64 = .{e_00, e_10, e_01, e_11}
///         >> 2x2 matrix, column-order
///
/// Output
///     > det(m): f64 = a*d-b*c
///
/// Target
///     > 2x2行列の行列式を導出
fn det(m: [4]f64) f64 {
    return m[0] * m[3] - m[1] * m[2];
}

/// Input
///     > matrix(m): [4]f64 = .{e_00, e_10, e_01, e_11}
///         >> 2x2 matrix, column-order
///
/// Output
///     > m^-1: [4]f64 = (1/det(m)) * .{e_11, -e_10, -e_01, e_00}
///         >> 2x2 matrix, column-order
///
/// Target
///     > 2x2行列の逆行列を導出
fn inv2x2(m: [4]f64) [4]f64 {
    const det_m = det(m);
    return .{
        m[3] / det_m,
        -m[1] / det_m,
        -m[2] / det_m,
        m[0] / det_m,
    };
}

/// Input
///     > ξ: f64
///         >> n = 2: ±√(1/3)
///     > η: f64
///         >> n = 2: ±√(1/3)
///
/// Output
///     > ∂N_k/∂(ξ, η): [8]f64 = .{∂N00/∂ξ, ∂N00/∂η, ∂N10/∂ξ, ∂N10/∂η, ∂N11/∂ξ, ∂N11/∂η, ∂N01/∂ξ, ∂N01/∂η, }
///         >> 4x2 matrix, column-order
///         >> Order of N_ij is counter-clock-wise
///
/// Target
///     > アイソパラメトリック要素を用いた四角形エレメント(order=1, n=2)によるFEMの実装
///     > .{dV/dxi, dV/deta} の導出には dV/dxi = ΣdN_k/dxi * V_k, dV/deta = ΣdN_k/deta * V_k を用いる
///     > 形状関数 N(x, y)_ij のξ-η座標系への座標変換行列 ∂N_ij/∂(ξ, η) を導出
fn dNdxieta(xi: f64, eta: f64) [8]f64 {
    const N00_xi = -0.25 * (1 - eta);
    const N00_eta = -0.25 * (1 - xi);
    const N10_xi = 0.25 * (1 - eta);
    const N10_eta = -0.25 * (1 + xi);
    const N11_xi = 0.25 * (1 + eta);
    const N11_eta = 0.25 * (1 + xi);
    const N01_xi = -0.25 * (1 + eta);
    const N01_eta = 0.25 * (1 - xi);
    return .{ N00_xi, N00_eta, N10_xi, N10_eta, N11_xi, N11_eta, N01_xi, N01_eta };
}

/// Input
///     > pos00: unit.math.z.Vec2_f64,
///     > pos10: unit.math.z.Vec2_f64,
///     > pos11: unit.math.z.Vec2_f64,
///     > pos01: unit.math.z.Vec2_f64,
///     > sigma: [4]f64,
/// Output
///     > elementStiffnessMatrix: [16]usize = .{}
///         >> 4x4 matrix, column-order
/// Target
///     > アイソパラメトリック要素を用いた四角形エレメント(order=1, n=2)によるFEMの実装
///     > ガウス・ルジャンドル数値積分から局所剛性行列 ∫∇N_i・σ・∇N_j を導出
///     > 局所剛性行列(axes=.{4, 4})は以下の通り数値計算される
///         >> dN/d(xi, eta)^T: (.{4, 2}) @ (J^-1)^T: (.{2, 2}) @ σ: (.{2, 2}) @ J^-1: (.{2, 2}) @ dN/d{xi, eta}: (.{2, 4})
///     > when J = d(x, y)/d(xi, eta)
///     > 高速化のために、エレメント剛性行列の導出はスタック上の操作で完結させる
///     > 内部計算においてオンサーガーの相反定理を用いており、導電率テンソルを対称行列であるとしている
fn elementStiffnessMatrixQuad(
    pos00: unit.math.z.Vec2_f64,
    pos10: unit.math.z.Vec2_f64,
    pos11: unit.math.z.Vec2_f64,
    pos01: unit.math.z.Vec2_f64,
    sigma: [4]f64,
) [16]f64 {
    var ans: [16]f64 = .{0} ** 16;

    inline for ([_][2]f64{
        [_]f64{ -1.0 / std.math.sqrt(3.0), -1.0 / std.math.sqrt(3.0) },
        [_]f64{ -1.0 / std.math.sqrt(3.0), 1.0 / std.math.sqrt(3.0) },
        [_]f64{ 1.0 / std.math.sqrt(3.0), -1.0 / std.math.sqrt(3.0) },
        [_]f64{ 1.0 / std.math.sqrt(3.0), 1.0 / std.math.sqrt(3.0) },
    }) |xi_eta| {
        const xi, const eta = xi_eta;
        const jac = jacobian(xi, eta, pos00, pos10, pos11, pos01);
        const jac_inv = inv2x2(jac);

        const det_jac = det(jac);
        if (1e-12 > det_jac) std.debug.panic("Jacobian determinant: {d} is too small\n", .{det_jac});

        const B = dNdxieta(xi, eta);

        const e = jac_inv[0] * (sigma[0] * jac_inv[0] + sigma[2] * jac_inv[1]) + jac_inv[1] * (sigma[1] * jac_inv[0] + sigma[3] * jac_inv[1]);
        // ここで、sigmaは対称行列なので、Dもまた対称行列になることを利用
        const g = jac_inv[2] * (sigma[0] * jac_inv[0] + sigma[2] * jac_inv[1]) + jac_inv[1] * (sigma[1] * jac_inv[0] + sigma[3] * jac_inv[1]);
        const f = jac_inv[2] * (sigma[0] * jac_inv[2] + sigma[2] * jac_inv[3]) + jac_inv[3] * (sigma[1] * jac_inv[2] + sigma[3] * jac_inv[3]);

        const D: [4]f64 = .{
            det_jac * e,
            det_jac * g,
            det_jac * g,
            det_jac * f,
        };

        // dxieta.T * (detj * )j_sigma_j * dxieta
        // ここも対称性を利用できる
        const _m_00 = D[0] * B[0] + D[2] * B[1];
        const _m_10 = D[1] * B[0] + D[3] * B[1];
        const _m_01 = D[0] * B[2] + D[2] * B[3];
        const _m_11 = D[1] * B[2] + D[3] * B[3];
        const _m_02 = D[0] * B[4] + D[2] * B[5];
        const _m_12 = D[1] * B[4] + D[3] * B[5];
        const _m_03 = D[0] * B[6] + D[2] * B[7];
        const _m_13 = D[1] * B[6] + D[3] * B[7];

        const m_00 = B[0] * _m_00 + B[1] * _m_10;
        const m_10 = B[2] * _m_00 + B[3] * _m_10;
        const m_20 = B[4] * _m_00 + B[5] * _m_10;
        const m_30 = B[6] * _m_00 + B[7] * _m_10;

        const m_01 = m_10;
        const m_11 = B[2] * _m_01 + B[3] * _m_11;
        const m_21 = B[4] * _m_01 + B[5] * _m_11;
        const m_31 = B[6] * _m_01 + B[7] * _m_11;

        const m_02 = m_20;
        const m_12 = m_21;
        const m_22 = B[4] * _m_02 + B[5] * _m_12;
        const m_32 = B[6] * _m_02 + B[7] * _m_12;

        const m_03 = m_30;
        const m_13 = m_31;
        const m_23 = m_32;
        const m_33 = B[6] * _m_03 + B[7] * _m_13;

        const matrix: [16]f64 = .{ m_00, m_10, m_20, m_30, m_01, m_11, m_21, m_31, m_02, m_12, m_22, m_32, m_03, m_13, m_23, m_33 };

        for (0..4) |r| {
            for (0..r) |c| {
                if (1e-6 < @abs(matrix[r * 4 + c] - matrix[c * 4 + r])) {
                    std.debug.print("jac: {d}\n", .{jac});
                    std.debug.print("det_jac: {d}\n", .{det_jac});
                    std.debug.print("jac_inv: {d}\n", .{jac_inv});
                    std.debug.print("detj_j_sigma_j: {d}\n", .{D});
                    std.debug.print("dn_dxieta: {d}\n", .{B});
                    std.debug.panic("stiffness matrix is not symmetric at [{d}, {d}], A[{d}. {d}] = {d} but  A[{d}. {d}] = {d}\n", .{ r, c, r, c, matrix[r * 4 + c], c, r, matrix[c * 4 + r] });
                }
            }
        }

        ans = .{
            ans[0] + matrix[0],
            ans[1] + matrix[1],
            ans[2] + matrix[2],
            ans[3] + matrix[3],
            ans[4] + matrix[4],
            ans[5] + matrix[5],
            ans[6] + matrix[6],
            ans[7] + matrix[7],
            ans[8] + matrix[8],
            ans[9] + matrix[9],
            ans[10] + matrix[10],
            ans[11] + matrix[11],
            ans[12] + matrix[12],
            ans[13] + matrix[13],
            ans[14] + matrix[14],
            ans[15] + matrix[15],
        };
    }

    return ans;
}

/// 目的
///     各三角形エレメントにおける局所剛性行列K_ijをスタック
///     要素のarrayIndexをインデクシングに用いる
///     導出した各エレメントの剛性行列にはその導電率が乗算される
fn elementStiffnessMatricesQuad(alc_tmp: std.mem.Allocator, mesh: *const sys.sim.mesh.FEM) ar.ArrayError!struct { std.ArrayList([2]usize), std.ArrayList([16]f64) } {
    var stack_indices = std.ArrayList([2]usize).init(alc_tmp);
    var stack = std.ArrayList([16]f64).init(alc_tmp);
    stack_indices.ensureTotalCapacity(mesh.num_elements) catch return ar.DataError.AllocationFailed;
    stack.ensureTotalCapacity(mesh.num_elements) catch return ar.DataError.AllocationFailed;
    const sigmas: []?f64 = mesh.elements.items(.sigma);

    for (sigmas, 0..) |sigma, idx| {
        if (sigma == null) continue;
        const array_idx = mesh.arrayElementIndex(idx) catch |e| return e;
        const v00, const v01, const v10, const v11 = mesh.getElementVertices(array_idx);
        const pos00 = v00.pos.?;
        const pos10 = v10.pos.?;
        const pos11 = v11.pos.?;
        const pos01 = v01.pos.?;

        const matrix = elementStiffnessMatrixQuad(pos00, pos10, pos11, pos01, .{ sigma.?, 0, 0, sigma.? });

        for (0..4) |r| {
            for (0..r) |c| {
                if (1e-6 < @abs(matrix[r * 4 + c] - matrix[c * 4 + r])) std.debug.panic("stiffness matrix is not symmetric at [{d}, {d}], A[{d}. {d}] = {d} but  A[{d}. {d}] = {d}\n", .{ r, c, r, c, matrix[r * 4 + c], c, r, matrix[c * 4 + r] });
            }
        }

        stack_indices.appendAssumeCapacity(array_idx);
        stack.appendAssumeCapacity(matrix);
    }

    return .{ stack_indices, stack };
}

/// 目的
///     全体剛性行列を求める
pub fn stiffnessMatrix(alc_tmp: std.mem.Allocator, mesh: *const sys.sim.mesh.FEM) ar.ArrayError!ar.Dense(f64, 2) {
    var matrix = ar.Dense(f64, 2).zeros(alc_tmp, &.{ mesh.num_vertices, mesh.num_vertices }) catch |e| return e;
    const stack_indices, const stack = elementStiffnessMatricesQuad(alc_tmp, mesh) catch |e| return e;
    defer {
        stack_indices.deinit();
        stack.deinit();
    }
    for (stack_indices.items, stack.items) |array_idx, m| {
        const v00, const v01, const v10, const v11 = mesh.getElementVertices(array_idx);
        matrix.data[v00.id.? + v00.id.? * mesh.num_vertices] += m[0];
        matrix.data[v10.id.? + v00.id.? * mesh.num_vertices] += m[1];
        matrix.data[v11.id.? + v00.id.? * mesh.num_vertices] += m[2];
        matrix.data[v01.id.? + v00.id.? * mesh.num_vertices] += m[3];
        matrix.data[v00.id.? + v10.id.? * mesh.num_vertices] += m[4];
        matrix.data[v10.id.? + v10.id.? * mesh.num_vertices] += m[5];
        matrix.data[v11.id.? + v10.id.? * mesh.num_vertices] += m[6];
        matrix.data[v01.id.? + v10.id.? * mesh.num_vertices] += m[7];
        matrix.data[v00.id.? + v11.id.? * mesh.num_vertices] += m[8];
        matrix.data[v10.id.? + v11.id.? * mesh.num_vertices] += m[9];
        matrix.data[v11.id.? + v11.id.? * mesh.num_vertices] += m[10];
        matrix.data[v01.id.? + v11.id.? * mesh.num_vertices] += m[11];
        matrix.data[v00.id.? + v01.id.? * mesh.num_vertices] += m[12];
        matrix.data[v10.id.? + v01.id.? * mesh.num_vertices] += m[13];
        matrix.data[v11.id.? + v01.id.? * mesh.num_vertices] += m[14];
        matrix.data[v01.id.? + v01.id.? * mesh.num_vertices] += m[15];
    }

    for (0..matrix.shape[0]) |r| {
        for (0..r) |c| {
            const rc = matrix.get(&.{ r, c }) catch unreachable;
            const cr = matrix.get(&.{ c, r }) catch unreachable;
            if (1e-6 < @abs(rc - cr)) {
                std.debug.panic("stiffness matrix is not symmetric at [{d}, {d}], A[{d}. {d}] = {d} but  A[{d}. {d}] = {d}\n", .{ r, c, r, c, rc, c, r, cr });
            }
        }
    }

    return matrix;
}

//以下現状使用していないコード
//
///// 目的
/////     各三角形エレメントにおける局所剛性行列K_ijの計算
/////     https://github.com/eitcom/pyEIT/blob/master/pyeit/eit/fem.py
/////     本質的には、連立方程式KV=Jにおける電位Vの係数に、各辺の内積を用いているだけ
//fn localStiffnessMatrixTri1D(alloc: std.mem.Allocator, xy: h.Array(f64), area: f64) h.ArrayError!h.Array(f64) {
//    if (!std.mem.eql(usize, xy.axes.constSlice(), &.{ 3, 2 })) unreachable;
//    const list = std.ArrayList(f64).init(alloc);
//    list.append(area) catch return h.DataError.AllocationFailed;
//    const areas = h.Array(f64).dense(&.{ 1, 1 }, list);
//
//    const xy0 = xy.slice(&.{ &.{0}, &.{} }) catch |e| return e;
//    const xy1 = xy.slice(&.{ &.{1}, &.{} }) catch |e| return e;
//    const xy2 = xy.slice(&.{ &.{2}, &.{} }) catch |e| return e;
//    const a = h.concat(&.{ &xy2, &xy0, &xy1 }, 0) catch |e| return e;
//    const b = h.concat(&.{ &xy1, &xy2, &xy0 }, 0) catch |e| return e;
//
//    const edges = h.sub(f64)(&a, &b) catch |e| return e;
//    const edgesT = edges.transpose(&.{ 1, 0 }) catch |e| return e;
//    const e_eT = h.matMul(f64)(&edges, &edgesT) catch |e| return e;
//
//    return h.div(f64)(&e_eT, &areas);
//}
//
///// 目的
/////     各三角形エレメントにおける局所剛性行列K_ijをスタック
//fn stackLocalStiffnessMatrixTri1D(alloc: std.mem.Allocator, mesh: m.Mesh) h.ArrayError!std.ArrayList(h.Array(f64)) {
//    var stack_indices = std.ArrayList([2]usize).init(alloc);
//    var stack = std.ArrayList(h.Array(f64)).init(alloc);
//
//    stack_indices.appendAssumeCapacity(mesh.num_elements);
//    stack.ensureTotalCapacity(mesh.num_elements);
//
//    for (mesh.elements.items(.id), 0..) |_id, idx| {
//        _ = _id orelse continue;
//        const array_idx = mesh.arrayElementIndex(idx) catch |e| return e;
//        const v00, const v01, const v10, const v11 = mesh.getElementVertices(array_idx);
//
//        var list = std.ArrayList(f64).init(alloc);
//        list.ensureTotalCapacity(3 * 2) catch return h.DataError.AllocationFailed;
//        list.appendNTimesAssumeCapacity(0, 3 * 2);
//        const xy = h.Array(f64).dense(&.{ 3, 2 }, list);
//
//        const area = util.areaQuad2D(v00.pos, v01.pos, v10.pos, v11.pos);
//
//        const local_stiffness_matrix = localStiffnessMatrixTri1D(alloc, xy, area) catch |e| return e;
//
//        stack_indices.appendAssumeCapacity(array_idx);
//        stack.appendAssumeCapacity(local_stiffness_matrix);
//    }
//}

test "stiffness matrix" {}
