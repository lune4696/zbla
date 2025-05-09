const std = @import("std");

pub const Status = struct {
    mutex: std.Thread.Mutex = .{},
    stage: Stage,

    pub fn lock(self: *@This()) void {
        self.mutex.lock();
    }

    pub fn unlock(self: *@This()) void {
        self.mutex.unlock();
    }
};

pub const Stage = enum(u8) {
    Running,
    Finished,
    Idle,
};
