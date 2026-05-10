const std = @import("std");

const SCHEMA = "pact.zig_source_needle_scan.v1";

const Match = struct {
    path: []const u8,
    hit_count: usize,
};

const Options = struct {
    root: []const u8,
    dirs: std.ArrayList([]const u8),
    suffixes: std.ArrayList([]const u8),
    needles: std.ArrayList([]const u8),
    require_all: bool,
    max_file_bytes: usize,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    const argv = try init.minimal.args.toSlice(allocator);

    var options = try parseArgs(allocator, argv);
    defer options.dirs.deinit(allocator);
    defer options.suffixes.deinit(allocator);
    defer options.needles.deinit(allocator);

    var root_dir = try std.Io.Dir.cwd().openDir(io, options.root, .{ .iterate = true });
    defer root_dir.close(io);

    var matches = std.ArrayList(Match).empty;
    defer {
        for (matches.items) |row| allocator.free(row.path);
        matches.deinit(allocator);
    }
    var file_count: usize = 0;

    for (options.dirs.items) |dir_rel| {
        try scanDir(io, allocator, root_dir, dir_rel, options, &matches, &file_count);
    }
    std.mem.sort(Match, matches.items, {}, matchLessThan);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll("{");
    try stdout.writeAll("\"schema\":");
    try writeJsonString(stdout, SCHEMA);
    try stdout.writeAll(",\"file_count\":");
    try stdout.print("{}", .{file_count});
    try stdout.writeAll(",\"match_count\":");
    try stdout.print("{}", .{matches.items.len});
    try stdout.writeAll(",\"matches\":[");
    for (matches.items, 0..) |row, idx| {
        if (idx != 0) try stdout.writeAll(",");
        try stdout.writeAll("{\"path\":");
        try writeJsonString(stdout, row.path);
        try stdout.writeAll(",\"hit_count\":");
        try stdout.print("{}", .{row.hit_count});
        try stdout.writeAll("}");
    }
    try stdout.writeAll("]}\n");
    try stdout.flush();
}

fn parseArgs(allocator: std.mem.Allocator, argv: []const []const u8) !Options {
    var root: ?[]const u8 = null;
    var dirs = std.ArrayList([]const u8).empty;
    var suffixes = std.ArrayList([]const u8).empty;
    var needles = std.ArrayList([]const u8).empty;
    var require_all = false;
    var max_file_bytes: usize = 16 * 1024 * 1024;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--root")) {
            i += 1;
            if (i >= argv.len) return usageError();
            root = argv[i];
        } else if (std.mem.eql(u8, arg, "--dir")) {
            i += 1;
            if (i >= argv.len) return usageError();
            try dirs.append(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--suffix")) {
            i += 1;
            if (i >= argv.len) return usageError();
            try suffixes.append(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--needle")) {
            i += 1;
            if (i >= argv.len) return usageError();
            try needles.append(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--require-all")) {
            require_all = true;
        } else if (std.mem.eql(u8, arg, "--max-file-bytes")) {
            i += 1;
            if (i >= argv.len) return usageError();
            max_file_bytes = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else {
            return usageError();
        }
    }

    if (root == null or needles.items.len == 0) return usageError();
    if (dirs.items.len == 0) try dirs.append(allocator, ".");
    if (suffixes.items.len == 0) try suffixes.append(allocator, ".py");

    return Options{
        .root = root.?,
        .dirs = dirs,
        .suffixes = suffixes,
        .needles = needles,
        .require_all = require_all,
        .max_file_bytes = max_file_bytes,
    };
}

fn usageError() error{InvalidUsage} {
    printUsage();
    return error.InvalidUsage;
}

fn printUsage() void {
    std.debug.print(
        \\usage: source_needle_scan --root ROOT [--dir DIR ...] [--suffix .py ...] --needle TEXT ...
        \\
        \\Scans source files for exact substrings and emits deterministic JSON.
        \\
    , .{});
}

fn scanDir(
    io: std.Io,
    allocator: std.mem.Allocator,
    root_dir: std.Io.Dir,
    dir_rel_raw: []const u8,
    options: Options,
    matches: *std.ArrayList(Match),
    file_count: *usize,
) !void {
    const dir_rel = if (std.mem.eql(u8, dir_rel_raw, "")) "." else dir_rel_raw;
    var dir = root_dir.openDir(io, dir_rel, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const rel = if (std.mem.eql(u8, dir_rel, "."))
            try allocator.dupe(u8, entry.path)
        else
            try std.fs.path.join(allocator, &.{ dir_rel, entry.path });
        defer allocator.free(rel);

        if (shouldSkip(rel)) continue;
        if (!hasAcceptedSuffix(rel, options.suffixes.items)) continue;

        const data = root_dir.readFileAlloc(
            io,
            rel,
            allocator,
            .limited(options.max_file_bytes),
        ) catch continue;
        defer allocator.free(data);
        file_count.* += 1;

        const hit_count = countNeedleHits(data, options.needles.items);
        const is_match = if (options.require_all)
            hit_count == options.needles.items.len
        else
            hit_count > 0;
        if (!is_match) continue;

        try matches.append(allocator, .{
            .path = try allocator.dupe(u8, rel),
            .hit_count = hit_count,
        });
    }
}

fn shouldSkip(rel: []const u8) bool {
    return std.mem.indexOf(u8, rel, "__pycache__") != null or
        std.mem.startsWith(u8, rel, ".git/") or
        std.mem.startsWith(u8, rel, ".mypy_cache/") or
        std.mem.startsWith(u8, rel, ".pytest_cache/") or
        std.mem.startsWith(u8, rel, ".omx/cache/") or
        std.mem.indexOf(u8, rel, "/.mypy_cache/") != null or
        std.mem.indexOf(u8, rel, "/.pytest_cache/") != null or
        std.mem.indexOf(u8, rel, "/.omx/cache/") != null or
        std.mem.indexOf(u8, rel, "experiments/results/") != null or
        std.mem.indexOf(u8, rel, "comma_lab_public_export/") != null;
}

fn hasAcceptedSuffix(rel: []const u8, suffixes: []const []const u8) bool {
    for (suffixes) |suffix| {
        if (std.mem.endsWith(u8, rel, suffix)) return true;
    }
    return false;
}

fn countNeedleHits(data: []const u8, needles: []const []const u8) usize {
    var count: usize = 0;
    for (needles) |needle| {
        if (needle.len == 0) continue;
        if (std.mem.indexOf(u8, data, needle) != null) count += 1;
    }
    return count;
}

fn matchLessThan(_: void, lhs: Match, rhs: Match) bool {
    return std.mem.lessThan(u8, lhs.path, rhs.path);
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try writer.print("\\u{x:0>4}", .{ch});
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}
