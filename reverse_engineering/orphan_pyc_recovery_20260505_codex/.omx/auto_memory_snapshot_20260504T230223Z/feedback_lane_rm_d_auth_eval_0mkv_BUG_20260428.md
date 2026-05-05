---
name: Lane RM-d crashed at contest_auth_eval — 0.mkv missing in extracted/ — INVESTIGATE
description: 2026-04-28 PM Lane RM-d trained successfully (Stage 1 complete, optimized_poses.pt 15620B produced) then CRASHED at Stage 3 contest_auth_eval. Error: "Error opening input file /workspace/pact/lane_rm_results/eval_work/extracted/0.mkv. No such file or directory". inflate.sh expected GT video at extracted/0.mkv but archive contained only renderer.bin/masks.mkv/optimized_poses.pt. Symlink/copy of upstream/videos/0.mkv → extracted/0.mkv missing somewhere (lane script OR contest_auth_eval pre-inflate setup). Lane RM-d destroyed (couldn't recover). NEXT: investigate canonical pattern + fix across affected lanes.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened

Lane RM-d (instance 35794059, Riemannian SE(3) pose TTO) successfully:
- Stage 0: NVDEC probe ✓
- Stage 1: pose TTO with `--optimizer riemannian-sgd`, warm-start from Lane A, eval_roundtrip + posetto-noise-std=0.5 ✓
- Stage 1 OUTPUT: `optimized_poses.pt` (15620 bytes — Lane RM SE(3))
- Stage 2: built archive (Lane A renderer + Lane A masks + Lane RM poses) ✓
- Stage 3: contest_auth_eval CRASHED with:

```
[in#0 @ 0x...] Error opening input: No such file or directory
Error opening input file /workspace/pact/lane_rm_results/eval_work/extracted/0.mkv.
[inflate] returncode=254 elapsed=0.1s
RuntimeError: [inflate] FAILED with returncode=254
```

## Root cause hypothesis

`submissions/robust_current/inflate.sh` reads:
- `ARCHIVE_DIR` (extracted/) — contains renderer.bin + masks.mkv + optimized_poses.pt
- `VIDEO_NAMES_FILE` (upstream/public_test_video_names.txt) — GT video list

But the GT VIDEO 0.mkv must be present somewhere reachable by inflate.sh. The error suggests inflate.sh expected `extracted/0.mkv` — meaning either:
- (a) contest_auth_eval should symlink `upstream/videos/0.mkv → extracted/0.mkv` BEFORE calling inflate
- (b) Lane RM's archive should INCLUDE 0.mkv (incorrect — bloats archive bytes)
- (c) inflate.sh hardcoded `extracted/0.mkv` but should use `--uncompressed-dir upstream/videos/`

Action: investigate contest_auth_eval._run_inflate to find where the GT video should live + whether a symlink/copy is missing.

## Affected lanes (potentially)

ALL lanes that call contest_auth_eval AND don't include 0.mkv in archive may be affected:
- Lane RM, Lane GP, Lane FL, Lane M-V3 (all pose-replacement lanes that just produce optimized_poses.pt)
- Lane J-NWC, Lane J-NWCS, Lane J-IMP (renderer-encoding lanes)
- Lane EBR (entropy bottleneck)
- Lane EC (engineered corrections)
- Lane Q-FAITHFUL (Quantizr replica)

If this is a pattern bug in canonical lane template + contest_auth_eval, ALL future deploys are affected.

## Successful counter-examples

Lane G v3 (1.05 frontier) auth-eval'd successfully. Lane A 1.15 too. So the pattern WORKS on some lanes. The bug is in WHAT changed between those and Lane RM-d's setup.

## Fix path

After codex:adversarial-review lands + the fixer subagent dispatches:
1. Read contest_auth_eval._run_inflate setup
2. Compare Lane RM deploy script against Lane G v3 deploy script (which works)
3. Identify the missing symlink/copy step
4. Fix in Lane RM's script + add canonical pattern to other lanes
5. Add Check N+1 — every lane script must symlink upstream/videos/0.mkv → extracted/0.mkv before contest_auth_eval

## Cost

Lane RM-d burned ~$0.33/hr × 3.5h training = $1.16 → all wasted because eval crashed. Plus 51 min of stale heartbeat after Stage 3 hung in inflate-error-loop.

## Cross-references
- `submissions/robust_current/inflate.sh` — looks for `extracted/0.mkv`
- `experiments/contest_auth_eval.py:206` — `_run_inflate`
- `scripts/remote_lane_rm_riemannian_pose_tto.sh` — Lane RM script
- `experiments/results/lane_g_v3_landed/contest_auth_eval.json` — successful eval (study Lane G v3 deploy script for canonical pattern)
