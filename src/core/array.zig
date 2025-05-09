const std = @import("std");
const blas = @cImport({
    @cInclude("cblas.h");
});
const lapack = @cImport({
    @cInclude("lapack.h");
});

/// 目的
///     Dense.data中の配列要素についてのエラー
///     初期値で埋めるなどの対応により、関数によっては無視することも想定している
pub const ElementError = error{
    Overflow,
    ZeroDivision,
};

/// 目的
///     Dense.shapeについてのエラー
pub const ShapeError = error{
    /// 2つのshapeの各次元の値が整合しない
    DimensionsMismatch,
    /// shapeの次元数 (= shape.len) が整合しない
    LengthMismatch,
    /// shapeのsize (= Π_i shape[i]) が整合しない
    SizeMismatch,
    /// shape とstrideが整合しない
    ShapeStrideMismatch,
    /// このshapeを使用することは想定されていない
    ShapeNotImplemented,
    /// shapeから想定される配列インデックスの範囲を上回る値が与えられている
    ArrayindexOutOfRange,
    /// shapeから想定される線形インデックスの範囲を上回る値が与えられている
    LinearindexOutOfRange,
    /// shapeの次元数 (= shape.len) がゼロ
    ZeroLength,
    /// shapeの次元の何れかにゼロの値がある(shape[i] == 0)
    ZeroDimension,
};

/// 目的
///     Dense.stridesについてのエラー
///     stridesとshapeが干渉する場合も含む
/// 背景
///     shape/strides間エラーは、shapeからstrideが作られるので、stride側のエラーとみなす
pub const StridesError = error{
    ShapeStrideMismatch,
    StridesNotOrdered,
    ZeroStride,
};

/// 目的
///     HybridArray.dataについてのエラー
///     dataとshape, stridesが矛盾する場合のエラーも含む
/// 背景
///     shape/data, strides/data間エラーは、shapeとstrideでdataにアクセスするので、data側のエラーとみなす
pub const DataError = error{
    AllocationFailed,
    DataShapeMismatch,
    DataNotOrdered,
    TypeNotImplemented,
};

/// 目的
///     HybridArray.shape, .stridesに関係するエラーの総体
pub const ShapeStridesError = ShapeError || StridesError;

/// 目的
///     HybridArrayに関係するエラーの総体
pub const ArrayError = DataError || ShapeStridesError;

/// 目的
///     全エラー
///     上記のエラー型でカバーできない時に用いることを想定
///     必要ない時は使わないことを強く推奨
pub const AllError = ElementError || ArrayError;

/// 目的
///     任意型多次元密配列、数値型(T)と次元(n)によって型が定義される
///     メソッドや関数の実装・動作規則はREADMEを参照
///     原則としてdataはimmutable, shape/stridesはmutableとして実装
///     一度生成されたDense(T, n)は、ユーザーが明示的にdestroy()する必要がある
/// 注意点
///     データ順はcolumn-major 、つまり行列なら列ベクトルが並ぶ形で表現される
pub fn Dense(comptime T: type, comptime n: usize) type {
    return struct {
        allocator: std.mem.Allocator,
        shape: [n]usize,
        strides: [n]usize,
        data: []T,

        /// 目的
        ///     データ型を取得するためのメソッド
        /// 注意点
        ///     これはzig側の実装が間に合ってないからだと思うのだが、
        ///     何故かインライン展開してもdtype()をcomptimeディスパッチに使えない
        pub inline fn dataType(self: @This()) type {
            _ = self;
            return T;
        }

        /// 目的
        ///     次元数を取得するメソッド
        pub inline fn arrayDim(self: @This()) usize {
            _ = self;
            return n;
        }

        /// 目的
        ///     Dense(T)を初期値を定めずに初期化
        /// 注意点
        ///     あくまでアロケータ、配列情報のみが割り当てられる
        ///     データ領域は確保されるが、データ自体はload()で明示的にロードする必要がある
        pub fn any(a: std.mem.Allocator, _shape: []const usize) ArrayError!Dense(T, n) {
            switch (@typeInfo(T)) {
                .int, .float => {},
                else => return DataError.TypeNotImplemented,
            }
            switch (n) {
                0 => return ShapeError.ZeroLength,
                else => if (n != _shape.len) return ShapeError.LengthMismatch,
            }

            var shape: [n]usize = undefined;
            @memcpy(&shape, _shape);

            const strides = stridesFromShape(n)(shape) catch |e| return e;

            const data = a.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
        ///     Dense(T)を初期値を定めずに初期化
        /// 注意点
        ///     _dataはコピーされ、配列自体のデータとして新しく確保される
        pub fn from(a: std.mem.Allocator, _shape: []const usize, _data: []const T) ArrayError!Dense(T, n) {
            switch (@typeInfo(T)) {
                .int, .float => {},
                else => return DataError.TypeNotImplemented,
            }
            switch (n) {
                0 => return ShapeError.ZeroLength,
                else => if (n != _shape.len) return ShapeError.LengthMismatch,
            }

            var shape: [n]usize = undefined;
            @memcpy(&shape, _shape);

            if (size(&shape) != _data.len) return DataError.DataShapeMismatch;

            const strides = stridesFromShape(n)(shape) catch |e| return e;

            const data = a.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);
            @memcpy(data, _data);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
        ///     Dense(T)を任意初期値(initial_value)で埋めして初期化
        pub fn with(a: std.mem.Allocator, _shape: []const usize, initial_value: T) ArrayError!Dense(T, n) {
            switch (@typeInfo(T)) {
                .int, .float => {},
                else => return DataError.TypeNotImplemented,
            }
            switch (n) {
                0 => return ShapeError.ZeroLength,
                else => if (n != _shape.len) return ShapeError.LengthMismatch,
            }

            var shape: [n]usize = undefined;
            @memcpy(&shape, _shape);

            const strides = stridesFromShape(n)(shape) catch |e| return e;

            const data = a.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            errdefer a.free(data);
            @memset(data, initial_value);

            return .{
                .allocator = a,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        pub fn zeros(a: std.mem.Allocator, array_shape: []const usize) ArrayError!Dense(T, n) {
            return with(a, array_shape, 0) catch |e| return e;
        }

        pub fn ones(a: std.mem.Allocator, array_shape: []const usize) ArrayError!Dense(T, n) {
            return with(a, array_shape, 1) catch |e| return e;
        }

        /// 目的
        ///     Dense(T)を破棄するメソッド
        pub fn destroy(self: @This()) void {
            self.allocator.free(self.data);
        }

        /// 目的
        ///     インスタンス情報のプリント
        /// 注意点
        ///     dataは生データではなく、viewが返却される
        pub fn print(self: @This()) ArrayError!void {
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
        ///     Dense(T)をコピーするメンバ関数
        ///     データは再配列されずに複製される
        /// 注意点
        ///     複製元のアロケータを使用
        pub fn copy(self: @This()) DataError!@This() {
            const data = self.allocator.alloc(T, size(&self.shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);
            return .{
                .allocator = self.allocator,
                .shape = self.shape,
                .strides = self.strides,
                .data = data,
            };
        }

        /// 目的
        ///     Dense(T, n)をコピーするメンバ関数
        ///     データは再配列されて複製される
        /// 注意点
        ///     複製元のアロケータを使用
        pub fn clone(self: @This()) ArrayError!@This() {
            var view_idx: [n]usize = .{0} ** n;
            const data = self.allocator.alloc(T, size(&self.shape)) catch return DataError.AllocationFailed;
            const dsize = size(&self.shape);
            for (0..dsize) |i| {
                defer incrementViewIndex(&self.shape, &view_idx);
                data[i] = self.get(&view_idx) catch |e| return e;
            }
            const strides = stridesFromShape(n)(self.shape) catch |e| return e;
            return .{
                .allocator = self.allocator,
                .shape = self.shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
        ///     配列要素を整列してArrayListとして渡す
        pub fn getOrderedData(self: @This()) ArrayError!std.ArrayList(T) {
            var list = std.ArrayList(T).init(self.allocator);
            errdefer list.deinit();
            list.ensureTotalCapacity(size(&self.shape)) catch return DataError.AllocationFailed;

            var view_idx: [n]usize = .{0} ** n;
            for (0..size(&self.shape)) |_| {
                defer incrementViewIndex(&self.shape, &view_idx);
                list.appendAssumeCapacity((self.get(&view_idx) catch |e| return e));
            }
            return list;
        }

        /// 目的
        ///     Denseの配列要素アクセス
        /// 注意点
        ///     ライブラリ外からの使用を想定し、shapeではなくsliceを引数に取る
        pub fn get(self: @This(), view_idx: []const usize) ShapeStridesError!T {
            checkIndexInRange(&self.shape, view_idx) catch |e| return e;
            var view_idx_array: [n]usize = undefined;
            @memcpy(&view_idx_array, view_idx);
            const data_idx = dataIndex(n)(self.strides, view_idx_array) catch |e| return e;
            return self.data[data_idx];
        }

        /// 目的
        ///     Denseの配列要素アクセス
        /// 注意点
        ///     ライブラリ外からの使用を想定し、shapeではなくsliceを引数に取る
        pub fn set(self: *@This(), view_idx: []const usize, value: T) ArrayError!void {
            var view_idx_array: [n]usize = undefined;
            @memcpy(&view_idx_array, view_idx);
            const data_idx = dataIndex(n)(self.strides, view_idx_array) catch |e| return e;
            self.data[data_idx] = value;
        }

        /// 目的
        ///     配列座標をスライスした新たな配列を取得
        /// 使用例
        ///     const arr2 = try arr.slice(&.{ &.{1}, &.{ 0, 2 }, &.{} });
        /// 注意点
        ///     引数indices_arr: 配列各軸のスライスする位置を指定、無指定はその軸の全てをスライスする
        ///     data_indicesはview_idx順に入るので、データ再配列がここで起きる
        ///     内部の一時的な配列保存に.dataのアロケータを用いている
        ///         実用的な問題は無いと判断した
        ///         スコープを出たらdeinit()される
        ///         arena_allocatorとかでは若干だが容量を圧迫する事に注意
        /// その他
        ///     各軸について特定の範囲を指定してブロック状にスライスする"block(T, n) fn (arr, rangedindices: [][2]const usize)は実装を検討中
        pub fn slice(self: @This(), indices_array: []const []const usize) ArrayError!@This() {
            var shape: [n]usize = undefined;
            for (0..shape.len) |ax| shape[ax] = if (0 == indices_array[ax].len) self.shape[ax] else indices_array[ax].len;
            for (self.shape, shape) |x, y| if (x < y) return ShapeError.DimensionsMismatch;
            for (indices_array, 0..) |indices, i| {
                for (indices) |idx| if (idx >= self.shape[i]) return ShapeError.DimensionsMismatch;
            }

            const slice_size = size(&shape);
            var data_indices = std.ArrayList(usize).init(self.allocator);
            defer data_indices.deinit();
            data_indices.ensureTotalCapacity(slice_size) catch return DataError.AllocationFailed;
            var sliced_view_idx: [n]usize = .{0} ** n;
            var view_idx: [n]usize = undefined;

            for (0..slice_size) |_| {
                defer incrementViewIndex(&shape, &sliced_view_idx);
                for (0..self.shape.len) |i| {
                    view_idx[i] = switch (indices_array[i].len) {
                        0 => sliced_view_idx[i],
                        else => indices_array[i][sliced_view_idx[i]],
                    };
                }
                data_indices.appendAssumeCapacity(dataIndex(n)(self.strides, view_idx) catch |e| return e);
            }

            var data = self.allocator.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            errdefer self.allocator.free(data);

            for (data_indices.items, 0..) |idx, i| data[i] = self.data[idx];
            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = stridesFromShape(n)(shape) catch |e| return e,
                .data = data,
            };
        }

        /// 目的
        ///     配列座標を変更した新たな配列を取得
        /// 使用例
        ///     const arr2 = try arr.reshape(4, &.{ 4, 2, 3, 2 });
        /// 注意点
        ///     データ順序が整った行列についてのみ使用可能
        ///     transpose()などでstridesの降順が乱れている場合、アクセスパターンを確定できない為エラーを出す
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn reshape(self: @This(), comptime m: usize, _shape: []const usize) ArrayError!Dense(T, m) {
            if (m != _shape.len) return ShapeError.DimensionsMismatch;
            var shape: [m]usize = undefined;
            @memcpy(&shape, _shape);
            if (size(&self.shape) != size(&shape)) return ShapeError.SizeMismatch;
            if (!self.isAligned()) return StridesError.StridesNotOrdered;

            const data = self.allocator.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = stridesFromShape(m)(shape) catch |e| return e,
                .data = data,
            };
        }

        /// 目的
        ///     配列を転置, 二次元配列専用
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 注意点
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn tr(self: *const @This()) ShapeError!void {
            if (n != 2) return ShapeError.ShapeNotImplemented;
            const dim0 = self.shape[0];
            const dim1 = self.shape[1];
            const stride0 = self.strides[0];
            const stride1 = self.strides[1];
            @constCast(self).shape[0] = dim1;
            @constCast(self).shape[1] = dim0;
            @constCast(self).strides[0] = stride1;
            @constCast(self).strides[1] = stride0;
        }

        /// 目的
        ///     配列を転置
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 注意点
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn trans(self: *const @This(), order: []const usize) ShapeError!void {
            if (n < 2) return ShapeError.ShapeNotImplemented;
            var shape = self.shape;
            var strides = self.strides;
            if (order.len != self.shape.len) return ShapeError.DimensionsMismatch;
            for (0..order.len) |i| {
                var flag = true;
                for (0..order.len) |j| {
                    if (j == i) flag = false;
                }
                if (flag) return ShapeError.DimensionsMismatch;
            }
            for (order, 0..) |ax, d| {
                shape[d] = self.shape[ax];
                strides[d] = self.strides[ax];
            }

            @constCast(self).shape = shape;
            @constCast(self).strides = strides;
        }

        /// 目的
        ///     転置した配列を取得, 2次元配列専用
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 注意点
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn getTr(self: @This()) ArrayError!@This() {
            if (n != 2) return ShapeError.ShapeNotImplemented;
            var shape = self.shape;
            var strides = self.strides;
            shape[0] = self.shape[1];
            shape[1] = self.shape[0];
            strides[0] = self.strides[1];
            strides[1] = self.strides[0];
            const data = self.allocator.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
        ///     転置した配列を取得
        ///     コピーしたshapeとstridesのdimensionを入れ替えることでインデクシングを切り替える
        /// 注意点
        ///     transpose()関数自体は軽量だが、配列のアクセスパターンに変化が生じるので、その後に使用する関数に悪影響が生じうることに注意
        ///     stridesの順序が乱れるため、transpose()した行列はreshape()などの関数が使えない
        ///     transpose()した配列をreshape()したい場合、まずclone()で再配列を明示的に行うこと
        pub fn getTrans(self: @This(), order: []const usize) ArrayError!@This() {
            if (n < 2) return ShapeError.ShapeNotImplemented;
            var shape = self.shape;
            var strides = self.strides;
            if (order.len != self.shape.len) return ShapeError.DimensionsMismatch;
            for (0..order.len) |i| {
                var flag = true;
                for (0..order.len) |j| {
                    if (j == i) flag = false;
                }
                if (flag) return ShapeError.DimensionsMismatch;
            }
            for (order, 0..) |ax, d| {
                shape[d] = self.shape[ax];
                strides[d] = self.strides[ax];
            }
            const data = self.allocator.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
            @memcpy(data, self.data);

            return .{
                .allocator = self.allocator,
                .shape = shape,
                .strides = strides,
                .data = data,
            };
        }

        /// 目的
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

        /// 目的
        ///     行列の(疑似)逆行列の導出
        /// https://www.netlib.org/lapack/explore-html-3.6.1/index.html
        pub fn inv(self: @This(), tmp_alc: std.mem.Allocator, obj_alc: std.mem.Allocator) ArrayError!@This() {
            switch (T) {
                f32, f64 => {},
                else => DataError.TypeNotImplemented,
            }
            if (self.shape.len != 2) return ShapeError.shapeNotImplemented;

            const shape: [2]usize = .{ self.shape[1], self.shape[0] };
            // In
            const r: i32 = @intCast(self.shape[0]); // 行
            const c: i32 = @intCast(self.shape[1]); // 列
            const lda = r; // leading dimension, lda <= max(1, r)
            const lwork = c; // nと同じ値

            // Out
            const data = obj_alc.alloc(T, size(&shape)) catch return DataError.AllocationFailed;
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
                .strides = stridesFromShape(2)(shape) catch |e| return e,
                .data = data,
            };
        }
    };
}

/// 目的
///     配列座標(shape)の最大要素数(size)を計算
pub fn size(shape: []const usize) usize {
    var ans: usize = 1;
    for (shape) |ax| ans *= ax;
    return ans;
}

/// 目的
///     配列座標(shape)のstrideを計算
/// 注意点
///     column-majorなので、stridesは次元の昇順に計算される
pub fn stridesFromShape(comptime n: usize) fn (shape: [n]usize) ShapeError![n]usize {
    return struct {
        fn f(shape: [n]usize) ShapeError![n]usize {
            var strides: [n]usize = undefined;
            var unit: usize = 1;
            for (0..shape.len) |i| {
                strides[i] = unit;
                unit *= shape[i];
            }
            return strides;
        }
    }.f;
}

/// 目的
///     getなどで配列要素にアクセスする時に、そのインデックスが妥当かどうか確認
///     配列座標(shape)の中に対象点(arrayindex)が収まるかどうか確認
pub fn checkIndexInRange(shape: []const usize, arrayidx: []const usize) ShapeError!void {
    for (shape, arrayidx) |ax, idx| if (ax <= idx) return ShapeError.ArrayindexOutOfRange;
}

/// 目的
///     stridesを利用してビューインデックスからデータインデックスを計算
pub fn dataIndex(comptime n: usize) fn (strides: [n]usize, view_index: [n]usize) StridesError!usize {
    return struct {
        fn f(strides: [n]usize, view_index: [n]usize) StridesError!usize {
            for (strides) |stride| if (0 == stride) return StridesError.ZeroStride;

            var linearidx: usize = 0;
            for (strides, view_index) |stride, ax| {
                linearidx += ax * stride;
            }
            return linearidx;
        }
    }.f;
}

/// 目的
///     配列座標(shape, order, strides)を利用してデータインデックス(線形)からビューインデックス(アレイ)を計算
pub fn viewIndex(comptime n: usize) fn (shape: [n]usize, strides: [n]usize, data_idx: usize) ShapeStridesError![n]usize {
    return struct {
        fn f(shape: [n]usize, strides: [n]usize, data_idx: usize) ShapeStridesError![n]usize {
            for (strides) |stride| if (0 == stride) return StridesError.ZeroStride;
            for (shape) |ax| if (0 == ax) return ShapeError.ZeroDimension;
            if (size(&shape) <= data_idx) return ShapeError.LinearindexOutOfRange;

            var view_idx = shape.init(shape.len) catch unreachable;
            var strides_checked = stridesFromShape(n).init(strides.len) catch unreachable;
            @memcpy(&strides_checked, strides);
            var _data_idx = data_idx;

            //std.debug.print("DEBUG: shape: {d}\n\n", .{shape});
            //std.debug.print("DEBUG: strides: {d}\n\n", .{strides});
            for (0..strides.len) |_| {
                const dim_max_stride: usize = std.sort.argMax(usize, strides_checked, {}, std.sort.asc(usize)).?;
                const max_stride: usize = strides_checked[dim_max_stride];

                var ax: usize = dim_max_stride;
                for (0..shape.len) |d| {
                    if (strides[d] == max_stride and shape[d] > shape[ax] and strides_checked[d] != 0) ax = d;
                }

                const idx = @divFloor(_data_idx, max_stride);
                view_idx[ax] = @divFloor(_data_idx, max_stride);
                _data_idx -= idx * max_stride;
                strides_checked[ax] = 0;
            }
            return view_idx;
        }
    }.f;
}

/// 目的
///     view_idxをcolumn-major(昇順)でインクリメント
/// 注意点
///     ビューインデックスはmutable
pub fn incrementViewIndex(shape: []const usize, view_idx: []usize) void {
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
pub fn gemm(comptime T: type) fn (a: *const Dense(T, 2), b: *const Dense(T, 2), c: *const Dense(T, 2), alpha: T, beta: T) ArrayError!void {
    return struct {
        fn f(a: *const Dense(T, 2), b: *const Dense(T, 2), c: *const Dense(T, 2), alpha: T, beta: T) ArrayError!void {
            switch (T) {
                f32, f64 => {},
                else => return DataError.TypeNotImplemented,
            }
            if (a.shape[1] != b.shape[0]) return ShapeError.DimensionsMismatch;

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
pub fn gemv(comptime T: type) fn (a: *const Dense(T, 2), x: *const Dense(T, 1), y: *const Dense(T, 1), alpha: T, beta: T) ArrayError!void {
    return struct {
        fn f(a: *const Dense(T, 2), x: *const Dense(T, 1), y: *const Dense(T, 1), alpha: T, beta: T) ArrayError!void {
            switch (T) {
                f32, f64 => {},
                else => return DataError.TypeNotImplemented,
            }
            if (a.shape[1] != x.shape[0]) return ShapeError.DimensionsMismatch;

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
pub fn axpy(comptime T: type) fn (x: *const Dense(T, 1), y: *const Dense(T, 1), alpha: T) ArrayError!void {
    return struct {
        fn f(x: *const Dense(T, 1), y: *const Dense(T, 1), alpha: T) ArrayError!void {
            switch (T) {
                f32, f64 => {},
                else => return DataError.TypeNotImplemented,
            }
            if (x.shape[0] != y.shape[0]) return ShapeError.DimensionsMismatch;

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
pub fn dot(comptime T: type) fn (x: *const Dense(T, 1), y: *const Dense(T, 1)) ArrayError!T {
    return struct {
        fn f(x: *const Dense(T, 1), y: *const Dense(T, 1)) ArrayError!T {
            switch (T) {
                f32, f64 => {},
                else => return DataError.TypeNotImplemented,
            }
            if (x.shape[0] != y.shape[0]) return ShapeError.DimensionsMismatch;

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
pub fn scal(comptime T: type) fn (x: *const Dense(T, 1), alpha: T) DataError!void {
    return struct {
        fn f(x: *const Dense(T, 1), alpha: T) DataError!void {
            switch (T) {
                f32, f64 => {},
                else => return DataError.TypeNotImplemented,
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

// unit tests
test "init / deinit / print" {
    std.debug.print("\nTEST: Dense.init(), .deinit(), .print()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;

    std.debug.print("VERIFY: Dense(f32).init() / .deinit() / .print()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, &.{ 2, 3, 1 });
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
    const shape = &.{ 2, 3, 1 };

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
    const shape = &.{ 2, 3, 1 };

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
    const shape = &.{ 2, 3, 1 };

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
    const shape = &.{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).get()...\n", .{});
    {
        const dense = try Dense(f32, 3).ones(alloc, shape);
        defer dense.destroy();
        try dense.print();
        std.debug.print("[0, 0, 0]: {?d}\n", .{(try dense.get(&.{ 0, 0, 0 }))});
        std.debug.print("[0, 1, 0]: {?d}\n", .{(try dense.get(&.{ 0, 1, 0 }))});
        std.debug.print("[0, 2, 0]: {?d}\n", .{(try dense.get(&.{ 0, 2, 0 }))});
        std.debug.print("[1, 0, 0]: {?d}\n", .{(try dense.get(&.{ 1, 0, 0 }))});
        std.debug.print("[1, 1, 0]: {?d}\n", .{(try dense.get(&.{ 1, 1, 0 }))});
        std.debug.print("[1, 2, 0]: {?d}\n", .{(try dense.get(&.{ 1, 2, 0 }))});
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
        try std.testing.expect((try dense.get(&.{ 0, 0, 0 })) == data0.items[0]);
        try std.testing.expect((try dense.get(&.{ 0, 1, 0 })) == data1.items[1]);
        try std.testing.expect((try dense.get(&.{ 0, 2, 0 })) == data2.items[2]);
        try std.testing.expect((try dense.get(&.{ 1, 0, 0 })) == data3.items[3]);
        try std.testing.expect((try dense.get(&.{ 1, 1, 0 })) == data4.items[4]);
        try std.testing.expect((try dense.get(&.{ 1, 2, 0 })) == data5.items[5]);
    }
    std.debug.print("...SUCCESS\n", .{});
    std.debug.print("\n", .{});
}

test "slice" {
    std.debug.print("\nTEST: Dense.slice()\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;
    const shape = &.{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).slice()...\n", .{});
    {
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const dense = try Dense(f32, 3).from(alloc, shape, list.items);
        defer dense.destroy();
        try dense.print();

        const slice = try dense.slice(&.{ &.{}, &.{ 0, 2 }, &.{0} });
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
    const shape = &.{ 2, 3, 1 };

    std.debug.print("VERIFY: Dense(f32).reshape()...\n", .{});
    {
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr = try Dense(f32, 3).from(alloc, shape, list.items);
        defer arr.destroy();
        try arr.print();

        const arr2 = try arr.reshape(2, &.{ 3, 2 });
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
        const shape = &.{ 3, 2 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr = try Dense(f32, 2).from(alloc, shape, list.items);
        defer arr.destroy();
        try arr.print();
        const arrT = try arr.getTrans(&.{ 1, 0 });
        defer arrT.destroy();
        try arrT.print();
        const arrT_ = try arrT.clone();
        defer arrT_.destroy();
        try arrT_.print();
        const arrTr = try arrT_.getTrans(&.{ 1, 0 });
        defer arrTr.destroy();
        try arrTr.print();
        const arr_ = try arrTr.clone();
        defer arr_.destroy();
        try arr_.print();
        try std.testing.expect(std.mem.eql(f32, arr.data, arr_.data));
        std.debug.print("...SUCCESS\n", .{});
    }

    {
        std.debug.print("VERIFY: transpose(f32)...\n", .{});
        const shape = &.{ 2, 2, 1, 1, 3, 1 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..12) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        const arr0 = try Dense(f32, 6).from(alloc, shape, list.items);
        defer arr0.destroy();
        try arr0.print();
        const arr1 = try arr0.getTrans(&.{ 0, 1, 3, 4, 2, 5 });
        defer arr1.destroy();
        try arr1.print();
        const arr2 = try arr1.getTrans(&.{ 0, 1, 3, 4, 2, 5 });
        defer arr2.destroy();
        try arr2.print();
        const arr3 = try arr2.getTrans(&.{ 0, 1, 3, 4, 2, 5 });
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
        const shape = &.{ 3, 3 };
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
        const shape = &.{ 3, 3 };
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try Dense(f32, 2).any(alc, c_shape);
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try Dense(f32, 2).any(alc, c_shape);
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try Dense(f32, 2).any(alc, c_shape);
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try Dense(f32, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try Dense(f32, 2).any(alc, c_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try Dense(f64, 2).any(alc, c_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try Dense(f64, 2).any(alc, c_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const b = try Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();

        const c = try Dense(f64, 2).any(alc, c_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const b = try Dense(f64, 2).from(alc, b_shape, b_data);
        defer b.destroy();
        try b.tr();

        const c = try Dense(f64, 2).any(alc, c_shape);
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const x = try Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f32, 1).any(alc, y_shape);
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

        const a = try Dense(f32, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const x = try Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f32, 1).any(alc, y_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();

        const x = try Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f64, 1).any(alc, y_shape);
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

        const a = try Dense(f64, 2).from(alc, a_shape, a_data);
        defer a.destroy();
        try a.tr();

        const x = try Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f64, 1).any(alc, y_shape);
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

        const x = try Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f32, 1).from(alc, y_shape, y_data);
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

        const x = try Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f64, 1).from(alc, y_shape, y_data);
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

        const x = try Dense(f32, 1).from(alc, x_shape, x_data);
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

        const x = try Dense(f64, 1).from(alc, x_shape, x_data);
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

        const x = try Dense(f32, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f32, 1).from(alc, y_shape, y_data);
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

        const x = try Dense(f64, 1).from(alc, x_shape, x_data);
        defer x.destroy();

        const y = try Dense(f64, 1).from(alc, y_shape, y_data);
        defer y.destroy();

        const result = try dot(f64)(&x, &y);

        try std.testing.expect(result == answer);

        std.debug.print("...SUCCESS\n", .{});
    }
}

test "exception check" {
    std.debug.print("\nTEST: Exception check\n", .{});
    std.debug.print("\n", .{});

    const alloc = std.testing.allocator;

    std.debug.print("VERIFY: Dense(bool) cannot be made...\n", .{});
    {
        const shape = &.{ 2, 3, 1 };
        var list = std.ArrayList(bool).init(alloc);
        for (0..6) |_| try list.append(true);
        defer list.deinit();

        try std.testing.expectError(DataError.TypeNotImplemented, Dense(bool, 3).from(alloc, shape, list.items));
    }
    std.debug.print("...SUCCESS\n\n", .{});

    std.debug.print("VERIFY: Dense.dim = 0 cannot be made...\n", .{});
    {
        const shape0 = &.{};

        try std.testing.expectError(ShapeError.ZeroLength, Dense(f32, 0).ones(alloc, shape0));
    }
    std.debug.print("...SUCCESS\n\n", .{});

    std.debug.print("VERIFY: shape-Data incosistent array cannot be made...\n", .{});
    {
        const shape = &.{ 2, 2 };
        var list = std.ArrayList(f32).init(alloc);
        for (0..6) |i| try list.append(@floatFromInt(i));
        defer list.deinit();

        try std.testing.expectError(DataError.DataShapeMismatch, Dense(f32, 2).from(alloc, shape, list.items));
    }
    std.debug.print("...SUCCESS\n\n", .{});

    std.debug.print("VERIFY: shapeOutOfRange can be detected...\n", .{});
    {
        const shape = &.{ 2, 3 };

        const dense = try Dense(f32, 2).ones(alloc, shape);
        defer dense.destroy();

        try std.testing.expectError(ShapeError.ArrayindexOutOfRange, dense.get(&.{ 2, 2 }));
    }
    std.debug.print("...SUCCESS\n\n", .{});
}
