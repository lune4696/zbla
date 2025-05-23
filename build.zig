const std = @import("std");

pub fn linkZigLibraries(b: *std.Build, comp: *std.Build.Step.Compile, target: std.Build.ResolvedTarget, libs: []const []const u8, mods: []const []const u8) void {
    // add dependency libraries
    for (libs, mods) |lib, mod| {
        const lib_dep = b.dependency(lib, .{
            .target = target,
        });
        const lib_mod = lib_dep.module(mod);
        comp.root_module.addImport(mod, lib_mod);
    }
}

pub fn linkCLibraries(b: *std.Build, comp: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    // exception check
    switch (target.result.cpu.arch) {
        .x86_64 => {},
        else => @panic("At now, x86_64 is an only supported cpu arch!"),
    }
    switch (target.result.os.tag) {
        .linux => {},
        .windows => {},
        else => @panic("At now, linux and windows is an only supported os!"),
    }

    // link c libraries
    comp.linkSystemLibrary("c");
    switch (target.result.os.tag) {
        .linux => {
            comp.linkSystemLibrary("cblas");
            comp.linkSystemLibrary("lapack");
            //comp.linkSystemLibrary("mpi");
        },
        .windows => {

            // windows
            const path_windows_openblas_lib: std.Build.LazyPath = b.path("./lib/windows/openblas/lib/libopenblas.lib");
            const path_windows_openblas_include: std.Build.LazyPath = b.path("./lib/windows/openblas/include");
            comp.addObjectFile(path_windows_openblas_lib);
            comp.addIncludePath(path_windows_openblas_include);
        },
        else => unreachable,
    }
    comp.linkLibC();
}

pub fn compileOnLinux(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, exe_name: []const u8, root_file: []const u8, zig_libs: []const []const u8, zig_mods: []const []const u8) void {
    const exe = b.addExecutable(.{
        .name = exe_name,
        .root_source_file = b.path(root_file),
        .target = target,
        .optimize = optimize,
    });

    // 自分自身をモジュールに含めないと外側から使えない
    const mod = b.addModule(exe_name, .{
        .root_source_file = b.path(root_file),
    });
    exe.root_module.addImport(exe_name, mod);

    linkZigLibraries(b, exe, target, zig_libs, zig_mods);
    linkCLibraries(b, exe, target);
    b.installArtifact(exe);
}

pub fn testOnLinux(b: *std.Build, target: std.Build.ResolvedTarget, root_file: []const u8, zig_libs: []const []const u8, zig_mods: []const []const u8) void {
    const test_step = b.step("test", "Run unit tests");

    const unit_tests = b.addTest(.{
        .root_source_file = b.path(root_file),
        .target = target,
        //.test_runner = b.path("./test_runner.zig"),
    });

    linkZigLibraries(b, unit_tests, target, zig_libs, zig_mods);
    linkCLibraries(b, unit_tests, target);

    const run_tests = b.addRunArtifact(unit_tests);
    run_tests.has_side_effects = true;
    test_step.dependOn(&run_tests.step);
}

/// 注意
///     現状のcompile for linux on linuxの実装はlinkSystemLibraryに頼っている
///     そのため、.os_tag = .linuxとすると、nixで入れたraylib, cblasが探せずに失敗する
///     おそらく適切なパスを見つけてexe.addLibraryPath("hoge");すれば出来そうなのだが、nix管理下のパスを探すのが面倒...
pub fn build(b: *std.Build) void {
    const exe_name = "zbla";
    const root_file = "src/root.zig";
    const zig_libs: []const []const u8 = &.{};
    const zig_mods: []const []const u8 = &.{};
    //const target = b.standardTargetOptions(.{});
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            //.os_tag = .windows,
        },
    });
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    compileOnLinux(b, target, optimize, exe_name, root_file, zig_libs, zig_mods);
    testOnLinux(b, target, root_file, zig_libs, zig_mods);
}
