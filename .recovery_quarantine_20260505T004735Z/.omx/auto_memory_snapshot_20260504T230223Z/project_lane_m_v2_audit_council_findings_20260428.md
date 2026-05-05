---
name: Lane M-V2 council audit — CRITICAL bug + rank-1 hypothesis NOT yet disproven
description: 2026-04-28 skunkworks council audit of Lane M-V2 1.84 result. Found BUG-1 (CRITICAL): train/inference pose-pad mismatch — optimizer feeds renderer ZERO-padded dims 1-5 while inflate feeds frozen-baseline-padded. The 0.076 PoseNet is signal of THE BUG, NOT of rank-1. Lane M-V3-clean ($0.30, 2h) validates the actual hypothesis with predicted band [1.05, 1.20].
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## BUG-1 (CRITICAL) — train/inference pose-pad mismatch

| Phase | Pose tensor fed to renderer |
|-------|------------------------------|
| **Optimization-time** (`experiments/optimize_poses.py:752-770` `_project_to_renderer_pose`) | `[zoom, 0, 0, 0, 0, 0]` ZERO-padded |
| **Save-time** (lines 2131-2149) | `[zoom, baseline_1..5]` frozen-baseline-padded |
| **Inflate-time** (`submissions/robust_current/inflate_renderer.py:2141`) | Saved tensor (frozen-baseline) |

**The optimizer is solving a different problem than the inflate is evaluating.**

The "V2 fix" only patched the SAVE side. The optimizer still uses zero-pad. Lane V1 (2.35) had save-side bug + zero-pad; Lane V2 (1.84) fixed save-side only. The improvement (-0.51) is from save-side; the underlying mismatch remains.

## Empirical confirmation

- Lane A optimized dims 1-5 have **std up to 2.07, range ±4.9** — these are NOT noise, carry signal
- Lane M-V2's saved dims 1-5 are byte-identical to Lane A's (zero diff) — confirming frozen-pad
- Lane M-V2's saved dim 0 has **std=2.66 vs GT-target std=1.26** — 2.1× inflated, optimizer overshooting because zero-padded dims 1-5 forced dim 0 to compensate

## Other issues found by audit

- **BUG-2** (HIGH): argmax-constraint check uses same broken zero-pad path
- **BUG-3** (MEDIUM): proxy-score block compares apples-to-oranges
- **ENG-1**: `test_optimize_poses_radial_zoom_save_shape.py` (T1-T9) pins SAVE side only; nothing tests train/inference parity
- **ENG-4**: stage-2 sanity check is shape-only, not semantic
- **DES-1**: "Freezing dims 1-5" is incoherent for 6-DOF FiLM-conditioned renderer (FiLM is non-linear — can't isolate dim 0)
- **DES-3**: pose_loss is MSE on all 6 PoseNet OUTPUT dims; if rank-1 commit, only dim 0 should be in the loss
- **ARB-1**: predicted band [1.10, 1.30] was magical thinking; empirical precedent (V1=2.35, baseline=2.29) said [2.20, 2.50]

## Rank-1 hypothesis status

**NOT actually disproven.** V1/V2 both contaminated by save-shape and train-inference bugs. The 0.076 PoseNet is the signal of BUG-1, not of the architectural premise.

## Lane M-V3-clean recommendation

- **Fix**: pass `init_poses[:, 1:6]` through `_project_to_renderer_pose` so train and inference use the SAME frozen-baseline padding
- **Add V3-B option**: `--posenet-loss-dims=1` to commit cleanly to rank-1 (only dim 0 in the loss)
- **Anchor**: Lane A 1.15 (NOT Lane G v3 — keeps ablations clean)
- **Predicted band**: [1.05, 1.20]
- **Cost**: $0.30 / 2h Vast.ai 4090
- **Decision criterion**: ≤1.20 to validate; >1.20 = architectural death (kill all Lane M variants + inform Lane GE / LM-V2 / MOS that rank-1 priors are contaminated)

## Composability verdict

Lane M-V2 has **NO positive composability**. Strictly dominated by Lane A on every wedge component. The "raw score ≠ stack value" rule does NOT apply — that rule requires at least one wedge near or below frontier; M-V2 has none.

## Council vote

- Yousfi: APPROVE V3-clean
- Fridrich: APPROVE V3-clean
- Contrarian: DISSENT (wedge-priority — focus on SegNet attack 43% wedge instead)
- Hotz: APPROVE V3-clean
- Quantizr: APPROVE V3-clean

**4/5 APPROVE; non-conservative charter overrides Contrarian. $0.30 diagnostic justified.**

## 5 transferable patterns extracted

1. **Train/inference parity gate**: every lane's optimizer + saver + inflate-loader must agree on input distribution
2. **Save-side test insufficiency**: shape-only tests miss semantic content drift
3. **Empirical-anchored predictions**: use historical scores, not magical thinking
4. **Semantic-not-shape sanity checks**: stage-2 should compare distortion proxies, not just shapes
5. **Wedge-attribution-before-iteration**: don't iterate on a lane targeting a wedge that's already <10% of gap

## Output

Full audit (484 lines): `/Users/adpena/Projects/pact/.omx/research/lane_m_v2_audit_council_20260428.md`

## Cross-references
- `project_lane_m_v2_landed_1_84_regression_20260428` — the result audited
- `project_posenet_rank1_discovery` — foundational hypothesis (status: NOT disproven, awaiting V3)
- `project_lane_g_v3_landed_1_05_20260428` — current frontier
- `project_lane_g_v3_stacking_skunkworks_20260428` — wedge attribution (SegNet=43%)
- `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` — but Lane M-V2 specifically has no composability path
