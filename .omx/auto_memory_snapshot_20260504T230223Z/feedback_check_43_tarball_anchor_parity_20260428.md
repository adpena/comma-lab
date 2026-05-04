---
name: Preflight Check 43 — launcher tarball must include lane anchors
description: 2026-04-28 PM after 3 Cycle 1 lanes failed because tarball excluded lane_a_landed/ which contained archive_lane_a.zip anchor. Added Check 43 to scan remote_lane_*.sh for ANCHOR_* paths and verify each is in launcher's tarball includes. STRICT @ 0 violations after explicit anchor includes added. 43 STRICT preflight checks total.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What Check 43 catches

Pattern: any path referenced via `ANCHOR_ARCHIVE`, `LANE_A_ARCHIVE`, `ANCHOR_RENDERER`, etc. in `scripts/remote_lane_*.sh` MUST be in `scripts/launch_lane_on_vastai.py`'s tarball includes (or at least not excluded by a parent `--exclude` pattern).

Example bug it catches:
- Lane SAUG-V2 references `LANE_A_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"`
- Launcher tarball had `--exclude=experiments/results` (3.4G dir)
- Result: lane fails on remote with "FATAL: missing experiments/results/lane_a_landed/archive_lane_a.zip"
- Lane wallclock ~30s before crash → cost burn but no auth eval

## Live state (2026-04-28)

After fix: 6 anchor paths detected in `remote_lane_*.sh`, all 6 in tarball includes:
- `experiments/results/lane_a_landed/archive_lane_a.zip`
- `experiments/results/lane_a_landed/iter_0`
- `experiments/results/lane_a_landed/optimized_poses.pt`
- `experiments/results/lane_a_landed/gt_pose_targets.pt`
- `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
- (1 more)

Check 43 STRICT @ 0 violations.

## Implementation pattern

```python
# Scan all remote_lane_*.sh for ANCHOR_*= references
pattern = re.compile(
    r'(?:ANCHOR_\w+|LANE_\w*ARCHIVE\w*|LANE_\w*POSES\w*|LANE_\w*MASKS\w*|LANE_\w*RENDERER\w*)='
    r'(?:"|\$\{[^:}]+:-)?(experiments/results/[\w./_-]+)'
)
# Scan launcher's build_tarball() for include lines (bare quoted paths)
# Compare: for each anchor path, verify it's in includes OR not excluded by any --exclude pattern
```

## Total preflight checks today: 43 STRICT (was 36 at session start)

Today's additions:
- 37: macOS resource-fork purge
- 38: test imports resolve
- 39: undeployed-producer scanner
- 40: FP4 hardware disclosure
- 41: lane heartbeat loop
- 42: pose-projection train/inference parity
- 43: launcher tarball lane-anchor parity

## Cross-references
- `feedback_canonical_parent_shell_launcher_20260428` — V2 launcher
- `feedback_launcher_v4_split_works_end_to_end_20260428` — V4 split (the one with this bug)
- `feedback_check_42_train_inference_parity_20260428` — sibling check class
