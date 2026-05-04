---
name: Launcher V5 — auto-discovery tarball builder (124MB / 2.8s, was 962MB / 22s)
description: 2026-04-28 PM rewrote build_tarball() with auto-discovery + explicit file list (tar -T). Replaces fragile include-dir + many-excludes pattern that kept breaking (5.9GB tarballs, missing anchors, scan timeouts). Auto-discovers ANCHOR_* paths from remote_lane_*.sh + auto-enumerates code files. Tarball 7.7× smaller, 8× faster, ZERO manual exclude maintenance.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Before (V4) → After (V5)

| Metric | V4 (excludes-based) | V5 (auto-discovery) |
|--------|---------------------|---------------------|
| Tarball size | 962MB | **124MB** (7.7× smaller) |
| Build time | 22s | **2.8s** (8× faster) |
| Manual exclude additions | Per new bloat dir | **None** (auto-skip) |
| Manual anchor-include adds | Per new lane | **None** (auto-discover) |
| Failure mode if new bloat dir | Tarball balloons silently | N/A — only included files end up in tarball |
| Failure mode if new lane anchor | Lane crashes on remote | N/A — auto-discovered |

## How V5 works

### Auto-discovery of anchor paths
```python
def _discover_anchor_paths_from_lane_scripts() -> list[str]:
    pattern = re.compile(
        r'(?:ANCHOR_\w+|LANE_\w*ARCHIVE\w*|...)='
        r'(?:"|\$\{[^:}]+:-)?(experiments/results/[\w./_-]+|upstream/[\w./_-]+)'
    )
    for sh in scripts_dir.glob("remote_lane_*.sh"):
        for m in pattern.finditer(sh.read_text()):
            anchor_paths.add(m.group(1))
    return sorted(anchor_paths)
```

Result on 2026-04-28: 6 anchor paths discovered automatically:
- `experiments/results/lane_a_landed/archive_lane_a.zip`
- `experiments/results/lane_a_landed/extracted/masks.mkv`
- `experiments/results/lane_a_landed/iter_0/{masks.mkv,optimized_poses.pt,renderer.bin}`
- `experiments/results/lane_a_landed/optimized_poses.pt`

### Auto-enumeration of code files
```python
def _enumerate_python_and_shell(root_subdir: str, max_total_mb: int = 50) -> list[str]:
    # Walks .py/.sh/.json/.toml/.md/.txt files, caps cumulative size
    ...
```

Caps prevent runaway includes if a dir is unexpectedly large.

### Explicit file list via `tar -T`
```python
file_list_path.write_text("\n".join(paths) + "\n")
cmd = ["tar", "-czf", tar, "-C", REPO_ROOT, "-T", file_list_path]
```

Tar packs ONLY the listed paths. No exclude-glob magic. No surprises.

## What this permanently fixes

The user identified the right pattern: "we need a way to automatically identify and exclude things that should be and detect things that aren't but should be" + "this is very fragile and will keep breaking over and over again".

V5 addresses both:
1. **Auto-detect what should be included**: anchor-discovery from lane scripts (positive include, not negative exclude)
2. **Auto-exclude bloat**: only enumerate code files (.py/.sh/.toml/.md/.txt) from key dirs; data artifacts come ONLY via explicit anchor paths
3. **Cap protection**: per-dir size caps prevent runaway includes

Result: a new lane that adds `ANCHOR_FOO="experiments/results/lane_x/foo.bin"` is auto-included. A new bloated dir under `experiments/results/` is auto-skipped (never enumerated since we don't walk that dir for code files).

## Cross-references
- `feedback_canonical_parent_shell_launcher_20260428` — V2 history
- `feedback_launcher_v3_phase2_split_needed_20260428` — V3 split rationale
- `feedback_launcher_v4_split_works_end_to_end_20260428` — V4 atomic phases
- `feedback_check_43_tarball_anchor_parity_20260428` — Check 43 (now auto-passes via V5 auto-discovery)
- 3 destroyed instances on 2026-04-28 PM that motivated this rewrite
