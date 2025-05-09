const ar = @import("../../core/bind.zig").array;
const Error = @import("../bind.zig").Error;
const TypeError = @import("../bind.zig").TypeError;

/// 目的
///     Ax=b
pub fn Vars(comptime T: type) type {
    return struct {
        a: ar.Dense(T, 2),
        b: ar.Dense(T, 1),

        pub fn init(a: ar.Dense(T), b: ar.Dense(T)) TypeError!@This() {
            switch (@typeInfo(T)) {
                .Float => |info| {
                    switch (info.bits) {
                        64 => {},
                        else => return TypeError.NotImplemented,
                    }
                },
                else => return TypeError.NotImplemented,
            }
            return .{
                .a = a,
                .b = b,
            };
        }

        pub fn deinit(self: @This()) void {
            self.a.destroy();
            self.b.destroy();
        }
    };
}
