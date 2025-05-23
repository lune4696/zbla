const std = @import("std");
const blas = @cImport({
    @cInclude("cblas.h");
});
const lapack = @cImport({
    @cInclude("lapack.h");
});

/// 目的
///     > Dense(T, n) のインデクシング (.shape, .strides) についてのエラー
///
/// Target
///     > Errors about Dense(T, n).shape
pub const IndexingError = error{
    /// 2つの shape の各次元の値が整合しない
    DimensionsMismatch,
    /// shape の size (= Π_i shape[i]) が整合しない
    SizeMismatch,
    /// dimension と stride が整合しない
    DimensionStrideMismatch,
    /// 与えられた形状に対して次元数が整合しない
    NumDimensionsMismatch,
    /// shape から想定される配列インデックスの範囲を上回る値が与えられている
    ArrayindexOutOfRange,
    /// shape から想定される線形インデックスの範囲を上回る値が与えられている
    LinearindexOutOfRange,
    /// shape の次元の何れかにゼロの値がある (shape[i] == 0)
    ZeroDimension,
    /// strides 内にゼロが含まれている
    ZeroStride,
    /// strides が整列されていない
    StridesNotOrdered,
};

/// 目的
///     > Dense(T, n) のデータ (.data) についてのエラー
///
/// Target
///     > Errors about Dense(T, n).data
pub const DataError = error{
    /// データアロケーションに失敗した
    AllocationFailed,
    /// データ型が合わない
    DataShapeMismatch,
    /// データが整列されていない
    DataNotOrdered,
};

/// 目的
///     > Dense(T, n) のデータ中の配列要素についてのエラー
///
/// Target
///     > Errors about an element in Dense(T, n).data
pub const ElementError = error{
    /// 計算オーバーフローが発生した
    Overflow,
    /// ゼロ除算が発生した
    ZeroDivision,
};

pub const Error = IndexingError || DataError || ElementError;

pub fn Vector(comptime T: type) type {
    return Dense(T, 1);
}

pub fn Matrix(comptime T: type) type {
    return Dense(T, 2);
}

/// 目的
///     > 任意型多次元密配列、数値型(T)と次元(n)によって型が定義される
///     > メソッドや関数の実装・動作規則はREADMEを参照
///     > 原則として .data は immutable 、 .shape/.strides は mutable として実装
///     > 一度生成された Dense(T, n) は、 caller が明示的に destroy() する必要がある
///
/// Target
///     > Arbitrary type and dimension dense array, defined as data type(T) and number of dimensions(n)
///     > See README for method/function implementations and regulations.
///     > Basically, .data is immutable and .shape/.strides is mutable in functions.
///     > Once Dense(T, n) is generated, caller must destroy() it.
///
/// 補足
///     > データ順は column-major 、ベクトルは列ベクトル (numpyと同じ)
///         >> [
///         >>  [0, 2],
///         >>  [1, 3],
///         >> ]
///
/// Appendix
///     > Data order is column-major and vector is column-vector (= numpy like).
pub fn Dense(comptime T: type, comptime n: usize) type {
    return struct {
        allocator: std.mem.Allocator,
        shape: [n]usize,
        strides: [n]usize,
        data: []T,

        /// 目的
        ///     > データ型を取得するためのメソッド
        /// 補足
        ///     > これはzig側の実装が間に合ってないからだと思うのだが、
        ///     > 何故かインライン展開してもdtype()をcomptimeディスパッチに使えない
        pub inline fn dataType(self: @This()) type {
            _ = self;
            return T;
        }

        /// 目的
        ///     > 次元数を取得するメソッド
        pub inline fn arrayDim(self: @This()) usize {
            _ = self;
            return n;
        }

        /// 目的
        ///     > Dense(T) を初期値を定めずに初期化
        /// 補足
        ///     > この段階ではデータは確保されていない
        pub fn init(a: std.mem.Allocator, shape: [n]usize) DataError!Dense(T, n) {
            comptime {
                if (n == 0) @compileError("ZeroDimension");
            }

            return .{
                .allocator = a,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = &.{},
            };
        }

        /// 目的
        ///     > Dense(T, n) を初期値を定めずにデータ領域を確保して生成
        /// 補足
        ///     > データ領域は確保されるが、データ自体は明示的にロードする必要がある
        pub fn any(a: std.mem.Allocator, shape: [n]usize) DataError!Dense(T, n) {
            comptime {
                if (n == 0) @compileError("ZeroDimension");
            }

            const data = a.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = data,
            };
        }

        /// 目的
        ///     Dense(T, n) をあるデータをコピーして生成
        /// 補足
        ///     original_data はコピーされ、配列自体のデータとして新しく確保される
        pub fn from(a: std.mem.Allocator, shape: [n]usize, original_data: []const T) DataError!Dense(T, n) {
            comptime {
                if (n == 0) @compileError("ZeroDimension");
            }

            if (size(shape) != original_data.len) return DataError.DataShapeMismatch;

            const data = a.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);
            @memcpy(data, original_data);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = data,
            };
        }

        /// 目的
        ///     Dense(T, n) を任意初期値 (initial_value) で埋めして生成
        pub fn with(a: std.mem.Allocator, shape: [n]usize, initial_value: T) DataError!Dense(T, n) {
            comptime {
                if (n == 0) @compileError("ZeroDimension");
            }

            const data = a.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);
            @memset(data, initial_value);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = data,
            };
        }

        /// 目的
        ///     > Dense(T, n) を initial_value = 0 で生成
        pub fn zeros(a: std.mem.Allocator, array_shape: [n]usize) DataError!Dense(T, n) {
            comptime {
                switch (@typeInfo(T)) {
                    .int, .float => {},
                    else => @compileError("InvalidType"),
                }
            }
            return with(a, array_shape, 0) catch |e| return e;
        }

        /// 目的
        ///     > Dense(T, n) を initial_value = 1 で生成
        pub fn ones(a: std.mem.Allocator, array_shape: [n]usize) DataError!Dense(T, n) {
            comptime {
                switch (@typeInfo(T)) {
                    .int, .float => {},
                    else => @compileError("InvalidType"),
                }
            }
            return with(a, array_shape, 1) catch |e| return e;
        }

        /// 目的
        ///     > Dense(T, n) を破棄するメソッド
        pub fn destroy(self: @This()) void {
            self.allocator.free(self.data);
        }

        /// 目的
        ///     > インスタンス情報のプリント
        ///     > dataは生データではなく、viewが返却される
        pub fn print(self: @This()) Error!void {
            std.debug.print("-" ** 16 ++ "\n", .{});
            std.debug.print("Array({s}, {d}): dense \n", .{ @typeName(self.dataType()), self.shape.len });
            std.debug.print("    shape: {d}\n", .{self.shape});
            std.debug.print("    strides: {d}\n", .{self.strides});
            const data = self.getOrderedData() catch |e| return e;
            defer data.deinit();
            std.debug.print("    data: [ ", .{});
            for (data.items) |item| std.debug.print("{?d}, ", .{item});
            std.debug.print("]\n", .{});
            std.debug.print("-" ** 16 ++ "\n", .{});
        }

        /// 目的
        ///     > Dense(T)をコピーするメンバ関数
        ///     > データは再配列されずに複製される
        /// 補足
        ///     > 複製元のアロケータを使用
        pub fn copy(self: @This()) DataError!@This() {
            const data = self.allocator.alloc(T, size(self.shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);
            return .{
                .allocator = self.allocator,
                .shape = self.shape,
                .strides = self.strides,
                .data = data,
            };
        }

        /// 目的
        ///     > Dense(T, n)をコピーするメンバ関数
        ///     > データは再配列されて複製される
        /// 補足
        ///     > 複製元のアロケータを使用
        pub fn clone(self: @This()) Error!@This() {
            var view_idx: [n]usize = .{0} ** n;
            const data = self.allocator.alloc(T, size(self.shape)) catch return DataError.AllocationFailed;
            const dsize = size(self.shape);
            for (0..dsize) |i| {
                defer incrementViewIndex(self.shape, &view_idx);
                data[i] = self.get(view_idx) catch |e| return e;
            }
            const strides = stridesFromShape(self.shape);
            return .{
                .allocator = self.allocator,
                .shape = self.shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
        ///     > 配列要素を整列してArrayListとして渡す
        pub fn getOrderedData(self: @This()) Error!std.ArrayList(T) {
            var list = std.ArrayList(T).init(self.allocator);
            errdefer list.deinit();
            list.ensureTotalCapacity(size(self.shape)) catch return DataError.AllocationFailed;

            var view_idx: [n]usize = .{0} ** n;
            for (0..size(self.shape)) |_| {
                defer incrementViewIndex(self.shape, &view_idx);
                list.appendAssumeCapacity((self.get(view_idx) catch |e| return e));
            }
            return list;
        }

        /// 目的
        ///     > Denseの配列要素のゲッター
        pub fn get(self: @This(), view_idx: [n]usize) IndexingError!T {
            checkIndexInRange(self.shape, view_idx) catch |e| return e;
            const data_idx = dataIndex(self.strides, view_idx) catch |e| return e;
            return self.data[data_idx];
        }

        /// 目的
        ///     > Denseの配列要素のセッター
        pub fn set(self: *@This(), view_idx: [n]usize, value: T) IndexingError!void {
            checkIndexInRange(self.shape, view_idx) catch |e| return e;
            const data_idx = dataIndex(self.strides, view_idx) catch |e| return e;
            self.data[data_idx] = value;
        }

        /// Target
        ///     配列座標をスライスした新たな配列を取得
        /// 使用例
        ///     const arr2 = try arr.slice(.{ &.{1}, &.{ 0, 2 }, &.{} });
        /// 補足
        ///     引数indices_arr: 配列各軸のスライスする位置を指定、無指定はその軸の全てをスライスする
        ///     data_indicesはview_idx順に入るので、データ再配列がここで起きる
        ///     内部の一時的な配列保存に.dataのアロケータを用いている
        ///         実用的な問題は無いと判断した
        ///         スコープを出たらdeinit()される
        ///         arena_allocatorとかでは若干だが容量を圧迫する事に注意
        /// その他
        ///     各軸について特定の範囲を指定してブロック状にスライスする"block(T, n) fn (arr, rangedindices: [n][2]usize)は実装を検討中
        pub fn slice(self: @This(), indices_array: [n][]const usize) Error!@This() {
            var shape: [n]usize = undefined;
            for (0..n) |ax| shape[ax] = if (0 == indices_array[ax].len) self.shape[ax] else indices_array[ax].len;
            for (self.shape, shape) |x, y| if (x < y) return IndexingError.DimensionsMismatch;
            for (indices_array, 0..) |indices, i| {
                for (indices) |idx| if (idx >= self.shape[i]) return IndexingError.DimensionsMismatch;
            }

            const slice_size = size(shape);
            var data_indices = std.ArrayList(usize).init(self.allocator);
            defer data_indices.deinit();
            data_indices.ensureTotalCapacity(slice_size) catch return DataError.AllocationFailed;
            var sliced_view_idx: [n]usize = .{0} ** n;
            var view_idx: [n]usize = undefined;

            for (0..slice_size) |_| {
                defer incrementViewIndex(shape, &sliced_view_idx);
                for (0..self.shape.len) |i| {
                    view_idx[i] = switch (indices_array[i].len) {
                        0 => sliced_view_idx[i],
                        else => indices_array[i][sliced_view_idx[i]],
                    };
                }
                data_indices.appendAssumeCapacity(dataIndex(self.strides, view_idx) catch |e| return e);
            }

            var data = self.allocator.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            errdefer self.allocator.free(data);

            for (data_indices.items, 0..) |idx, i| data[i] = self.data[idx];
            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = data,
            };
        }

        /// Target
        ///     配列座標を変更した新たな配列を取得
        /// 使用例
        ///     const arr2 = try arr.reshape(4, .{ 4, 2, 3, 2 });
        /// 補足
        ///     データ順序が整った行列についてのみ使用可能
        ///     transpose()などでstridesの降順が乱れている場合、アクセスパターンを確定できない為エラーを出す
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn reshape(self: @This(), comptime m: usize, shape: [m]usize) Error!Dense(T, m) {
            if (size(self.shape) != Dense(T, m).size(shape)) return IndexingError.SizeMismatch;
            if (!self.isAligned()) return IndexingError.StridesNotOrdered;

            const data = self.allocator.alloc(T, Dense(T, m).size(shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = Dense(T, m).stridesFromShape(shape),
                .data = data,
            };
        }

        /// Target
        ///     配列を転置, 二次元配列専用
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 補足
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn tr(self: *@This()) void {
            comptime if (n != 2) @compileError("Invalid number of dimension");
            const dim0 = self.shape[0];
            const dim1 = self.shape[1];
            const stride0 = self.strides[0];
            const stride1 = self.strides[1];
            self.shape[0] = dim1;
            self.shape[1] = dim0;
            self.strides[0] = stride1;
            self.strides[1] = stride0;
        }

        /// Target
        ///     配列を転置
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 補足
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn trans(self: *@This(), order: [n]usize) IndexingError!void {
            comptime if (n < 2) @compileError("Invalid number of dimension");
            var shape = self.shape;
            var strides = self.strides;
            for (0..n) |i| {
                var flag = true;
                for (0..n) |j| {
                    if (j == i) flag = false;
                }
                if (flag) return IndexingError.DimensionsMismatch;
            }
            for (order, 0..) |ax, d| {
                shape[d] = self.shape[ax];
                strides[d] = self.strides[ax];
            }

            self.shape = shape;
            self.strides = strides;
        }

        /// Target
        ///     転置した配列を取得, 2次元配列専用
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 補足
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn getTr(self: @This()) DataError!@This() {
            comptime if (n != 2) @compileError("Invalid number of dimension");
            var shape = self.shape;
            var strides = self.strides;
            shape[0] = self.shape[1];
            shape[1] = self.shape[0];
            strides[0] = self.strides[1];
            strides[1] = self.strides[0];
            const data = self.allocator.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// Target
        ///     転置した配列を取得
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 補足
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn getTrans(self: @This(), order: [n]usize) Error!@This() {
            comptime if (n < 2) @compileError("Invalid number of dimension");
            var shape = self.shape;
            var strides = self.strides;
            for (0..n) |i| {
                var flag = true;
                for (0..n) |j| {
                    if (j == i) flag = false;
                }
                if (flag) return IndexingError.DimensionsMismatch;
            }
            for (order, 0..) |ax, d| {
                shape[d] = self.shape[ax];
                strides[d] = self.strides[ax];
            }
            const data = self.allocator.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// Target
        ///     行列のStrideが昇順(column-major)にAlignされているかを確認
        ///     つまり、transpose()されたかどうかの確認に使える
        pub fn isAligned(self: @This()) bool {
            const strides = self.strides;
            if (0 == strides.len) unreachable;
            for (0..self.strides.len - 1) |d| {
                if (strides[d] > strides[d + 1]) return false;
            }
            return true;
        }

        /// Target
        ///     行列の(疑似)逆行列の導出
        /// https://www.netlib.org/lapack/explore-html-3.6.1/index.html
        pub fn inv(self: @This(), tmp_alc: std.mem.Allocator, obj_alc: std.mem.Allocator) Error!@This() {
            switch (T) {
                f32, f64 => {},
                else => DataError.TypeNotImplemented,
            }
            if (self.shape.len != 2) return IndexingError.NotImplemented;

            const shape: [2]usize = .{ self.shape[1], self.shape[0] };
            // In
            const r: i32 = @intCast(self.shape[0]); // 行
            const c: i32 = @intCast(self.shape[1]); // 列
            const lda = r; // leading dimension, lda <= max(1, r)
            const lwork = c; // nと同じ値

            // Out
            const data = obj_alc.alloc(T, size(shape)) catch return DataError.AllocationFailed;
            errdefer obj_alc.free(data);
            @memcpy(data, self.data);

            const ipiv = tmp_alc.alloc(i32, @min(self.shape[0], self.shape[1])) catch return DataError.AllocationFailed;
            defer tmp_alc.free(ipiv);

            const work = tmp_alc.alloc(T, @intCast(lwork)) catch return DataError.AllocationFailed;
            defer tmp_alc.free(work);

            var info: i32 = undefined; // 計算が成功すれば0を返す

            switch (T) {
                f32 => {
                    // LAPACKのdgetrfサブルーチンを呼んで、行列AをLU分解
                    // 引数は全て参照渡し
                    lapack.sgetrf_(&r, &c, data.ptr, &lda, ipiv.ptr, &info);
                    // LU分解後の行列から逆行列を求める
                    // 逆行列は元の配列Aに入る
                    lapack.sgetri_(&c, data.ptr, &lda, ipiv.ptr, work.ptr, &lwork, &info);
                },
                f64 => {
                    // LAPACKのdgetrfサブルーチンを呼んで、行列AをLU分解
                    // 引数は全て参照渡し
                    lapack.dgetrf_(&r, &c, data.ptr, &lda, ipiv.ptr, &info);
                    // LU分解後の行列から逆行列を求める
                    // 逆行列は元の配列Aに入る
                    lapack.dgetri_(&c, data.ptr, &lda, ipiv.ptr, work.ptr, &lwork, &info);
                },
                else => unreachable,
            }
            return .{
                .allocator = obj_alc,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .data = data,
            };
        }
        /// Target
        ///     配列座標(shape)の最大要素数(size)を計算
        pub fn size(shape: [n]usize) usize {
            var ans: usize = 1;
            for (shape) |ax| ans *= ax;
            return ans;
        }

        /// Target
        ///     配列座標(shape)のstrideを計算
        /// 補足
        ///     column-majorなので、stridesは次元の昇順に計算される
        pub fn stridesFromShape(shape: [n]usize) [n]usize {
            var strides: [n]usize = undefined;
            var unit: usize = 1;
            for (0..shape.len) |i| {
                strides[i] = unit;
                unit *= shape[i];
            }
            return strides;
        }

        /// Target
        ///     getなどで配列要素にアクセスする時に、そのインデックスが妥当かどうか確認
        ///     配列座標(shape)の中に対象点(arrayindex)が収まるかどうか確認
        pub fn checkIndexInRange(shape: [n]usize, arrayidx: [n]usize) IndexingError!void {
            for (shape, arrayidx) |ax, idx| if (ax <= idx) return IndexingError.ArrayindexOutOfRange;
        }

        /// Target
        ///     stridesを利用してビューインデックスからデータインデックスを計算
        pub fn dataIndex(strides: [n]usize, view_index: [n]usize) IndexingError!usize {
            for (strides) |stride| if (0 == stride) return IndexingError.ZeroStride;

            var linearidx: usize = 0;
            for (strides, view_index) |stride, ax| {
                linearidx += ax * stride;
            }
            return linearidx;
        }

        /// Target
        ///     配列座標(shape, order, strides)を利用してデータインデックス(線形)からビューインデックス(アレイ)を計算
        pub fn viewIndex(shape: [n]usize, strides: [n]usize, data_idx: usize) Error![n]usize {
            for (strides) |stride| if (0 == stride) return IndexingError.ZeroStride;
            for (shape) |ax| if (0 == ax) return IndexingError.ZeroDimension;
            if (size(shape) <= data_idx) return IndexingError.LinearindexOutOfRange;

            var view_idx = shape.init(shape.len) catch unreachable;
            var _data_idx = data_idx;

            //std.debug.print("DEBUG: shape: {d}\n\n", .{shape});
            //std.debug.print("DEBUG: strides: {d}\n\n", .{strides});
            for (0..strides.len) |_| {
                const dim_max_stride: usize = std.sort.argMax(usize, strides, {}, std.sort.asc(usize)).?;
                const max_stride: usize = strides[dim_max_stride];

                var ax: usize = dim_max_stride;
                for (0..shape.len) |d| {
                    if (strides[d] == max_stride and shape[d] > shape[ax] and strides[d] != 0) ax = d;
                }

                const idx = @divFloor(_data_idx, max_stride);
                view_idx[ax] = @divFloor(_data_idx, max_stride);
                _data_idx -= idx * max_stride;
                strides[ax] = 0;
            }
            return view_idx;
        }

        /// Target
        ///     view_idxをcolumn-major(昇順)でインクリメント
        /// 補足
        ///     ビューインデックスはmutable
        pub fn incrementViewIndex(shape: [n]usize, view_idx: *[n]usize) void {
            if (shape.len != view_idx.len) unreachable;
            for (0..shape.len) |i| {
                if (shape[i] - 1 < view_idx[i]) unreachable;
                if (shape[i] - 1 > view_idx[i]) {
                    view_idx[i] += 1;
                    break;
                }
                view_idx[i] = 0;
            }
        }
    };
}

// unit tests
test "init / deinit / print" {
    std.debug.print("\nTEST: Dense.init(), .deinit(), .print()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).init() / .deinit() / .print()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();
    }

    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "copy" {
    std.debug.print("\nTEST: Dense.copy()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).copy()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();

        const dense2 = try dense.copy();
        defer dense2.destroy();
        try dense2.print();

        try std.testing.expect(std.mem.eql(f32, dense.data, dense2.data));
    }

    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "clone" {
    std.debug.print("\nTEST: Dense.clone()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: (f32).clone()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();

        const dense2 = try dense.clone();
        defer dense2.destroy();
        try dense2.print();

        try std.testing.expect(std.mem.eql(f32, dense.data, dense2.data));
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "getData" {
    std.debug.print("\nTEST: Dense.getData()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).getData()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();

        const data = try dense.getOrderedData();
        defer data.deinit();

        std.debug.print("data: [ ", .{});
        for (data.items) |item| std.debug.print("{?d}, ", .{item});
        std.debug.print("]\n", .{});

        try std.testing.expect(std.mem.eql(f32, data.items, &.{ 1, 1, 1, 1, 1, 1 }));
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "get" {
    std.debug.print("\nTEST: Dense.get()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).get()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();
        std.debug.print("[0, 0, 0]: {?d}\n", .{(try dense.get(.{ 0, 0, 0 }))});
        std.debug.print("[0, 1, 0]: {?d}\n", .{(try dense.get(.{ 0, 1, 0 }))});
        std.debug.print("[0, 2, 0]: {?d}\n", .{(try dense.get(.{ 0, 2, 0 }))});
        std.debug.print("[1, 0, 0]: {?d}\n", .{(try dense.get(.{ 1, 0, 0 }))});
        std.debug.print("[1, 1, 0]: {?d}\n", .{(try dense.get(.{ 1, 1, 0 }))});
        std.debug.print("[1, 2, 0]: {?d}\n", .{(try dense.get(.{ 1, 2, 0 }))});
        const data0 = try dense.getOrderedData();
        const data1 = try dense.getOrderedData();
        const data2 = try dense.getOrderedData();
        const data3 = try dense.getOrderedData();
        const data4 = try dense.getOrderedData();
        const data5 = try dense.getOrderedData();
        defer data0.deinit();
        defer data1.deinit();
        defer data2.deinit();
        defer data3.deinit();
        defer data4.deinit();
        defer data5.deinit();
        try std.testing.expect((try dense.get(.{ 0, 0, 0 })) == data0.items[0]);
        try std.testing.expect((try dense.get(.{ 0, 1, 0 })) == data1.items[1]);
        try std.testing.expect((try dense.get(.{ 0, 2, 0 })) == data2.items[2]);
        try std.testing.expect((try dense.get(.{ 1, 0, 0 })) == data3.items[3]);
        try std.testing.expect((try dense.get(.{ 1, 1, 0 })) == data4.items[4]);
        try std.testing.expect((try dense.get(.{ 1, 2, 0 })) == data5.items[5]);
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "slice" {
    std.debug.print("\nTEST: Dense.slice()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).slice()...\n", .{});
    {
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const dense = try Dense(f32, 3).from(alloc, shape, list.items);
        defer dense.destroy();
        try dense.print();

        const slice = try dense.slice(.{ &.{}, &.{ 0, 2 }, &.{0} });
        defer slice.destroy();
        try slice.print();

        const data = try slice.getOrderedData();
        defer data.deinit();
        std.debug.print("dense[:, [0, 2], 0]: [ ", .{});
        for (data.items) |item| std.debug.print("{?d}, ", .{item});
        std.debug.print("]\n", .{});

        try std.testing.expect(std.mem.eql(f32, data.items, &.{ 0, 1, 4, 5 }));
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "reshape" {
    std.debug.print("\nTEST: Dense.reshape()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = .{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).reshape()...\n", .{});
    {
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr = try Dense(f32, 3).from(alloc, shape, list.items);
        defer arr.destroy();
        try arr.print();

        const arr2 = try arr.reshape(2, .{ 3, 2 });
        defer arr2.destroy();
        try arr2.print();

        const data = try arr2.getOrderedData();
        defer data.deinit();
        try std.testing.expect(std.mem.eql(f32, data.items, &.{ 0, 1, 2, 3, 4, 5 }));
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "transpose" {
    std.debug.print("\nTEST: transpose\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;

    {
        std.debug.print("VERIFY: transpose(f32)...\n", .{});
        const shape = .{ 3, 2 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr = try Dense(f32, 2).from(alloc, shape, list.items);
        defer arr.destroy();
        try std.testing.expect(arr.isAligned());
        try arr.print();

        const arrT = try arr.getTrans(.{ 1, 0 });
        defer arrT.destroy();
        try std.testing.expect(!arrT.isAligned());
        try arrT.print();

        const arrT_ = try arrT.clone();
        defer arrT_.destroy();
        try std.testing.expect(arrT_.isAligned());
        try arrT_.print();

        const arrTr = try arrT_.getTrans(.{ 1, 0 });
        defer arrTr.destroy();
        try std.testing.expect(!arrTr.isAligned());
        try arrTr.print();

        const arr_ = try arrTr.clone();
        defer arr_.destroy();
        try std.testing.expect(arr_.isAligned());
        try arr_.print();

        try std.testing.expect(std.mem.eql(f32, arr.data, arr_.data));
        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: transpose(f32)...\n", .{});
        const shape = .{ 2, 2, 1, 1, 3, 1 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..12) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr0 = try Dense(f32, 6).from(alloc, shape, list.items);
        defer arr0.destroy();
        try arr0.print();
        const arr1 = try arr0.getTrans(.{ 0, 1, 3, 4, 2, 5 });
        defer arr1.destroy();
        try arr1.print();
        const arr2 = try arr1.getTrans(.{ 0, 1, 3, 4, 2, 5 });
        defer arr2.destroy();
        try arr2.print();
        const arr3 = try arr2.getTrans(.{ 0, 1, 3, 4, 2, 5 });
        defer arr3.destroy();
        try arr3.print();
        try std.testing.expect(std.mem.eql(f32, arr0.data, arr3.data));
        std.debug.print("...SUCCESS\n", .{});
    }
}

test "inv" {
    std.debug.print("\nTEST: inv\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;

    {
        std.debug.print("VERIFY: inv(f32, 2)...\n", .{});
        const shape = .{ 3, 3 };
        const data = &.{ 1, 1, -1, -2, 0, 1, 0, 2, 1 };
        const answer = &.{ -0.5, -0.75, 0.25, 0.5, 0.25, 0.25, -1, -0.5, 0.5 };

        const arr = try Dense(f32, 2).from(alloc, shape, data);
        defer arr.destroy();
        try arr.print();

        const arr_inv = try arr.inv(alloc, alloc);
        defer arr_inv.destroy();
        try arr_inv.print();

        try std.testing.expect(std.mem.eql(f32, arr_inv.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: inv(f64, 2)...\n", .{});
        const shape = .{ 3, 3 };
        const data = &.{ 1, 1, -1, -2, 0, 1, 0, 2, 1 };
        const answer = &.{ -0.5, -0.75, 0.25, 0.5, 0.25, 0.25, -1, -0.5, 0.5 };

        const arr = try Dense(f64, 2).from(alloc, shape, data);
        defer arr.destroy();
        try arr.print();

        const arr_inv = try arr.inv(alloc, alloc);
        defer arr_inv.destroy();
        try arr_inv.print();

        try std.testing.expect(std.mem.eql(f64, arr_inv.data, answer));

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "exception check" {
    std.debug.print("\nTEST: Exception check\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;

    std.debug.print("VERIFY: shape-Data incosistent array cannot be made...\n", .{});
    {
        const shape = .{ 2, 2 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        try std.testing.expectError(DataError.DataShapeMismatch, Dense(f32, 2).from(alloc, shape, list.items));
    }
    std.debug.print("...SUCCESS\n\n", .{});

    std.debug.print("VERIFY: shapeOutOfRange can be detected...\n", .{});
    {
        const shape = .{ 2, 3 };

        const dense = try Dense(f32, 2).ones(alloc, shape);
        defer dense.destroy();

        try std.testing.expectError(IndexingError.ArrayindexOutOfRange, dense.get(.{ 2, 2 }));
    }
    std.debug.print("...SUCCESS\n\n", .{});
}
