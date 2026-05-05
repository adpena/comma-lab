---
name: PHASE 1 dispatch-state corrections — Lane 7 + Lane 8 partial impl discovered
description: 2026-04-29 PM. While doing Phase 1 dispatch greenup, discovered partial implementations that the earlier "❌ NOT STARTED" scoping missed. Corrects project_phases_2_3_4_design_implementation_math_provenance_20260429.md and project_6month_strategic_plan_20260429.md status table for Lane 7 (PSD has profile, missing dispatch script) and Lane 8 (multi-pass exists for postfilter, needs generalization to renderer pipeline).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Lane 7: PSD standard 🟡 PARTIAL (was 🟡 needs profile review)

**What exists:**
- `src/tac/profiles.py:168` `PSD_STANDARD_ADAPTIVE` profile (verified 2026-04-29):
  ```python
  PSD_STANDARD_ADAPTIVE = {
      **PROVEN_BASELINE,
      "variant": "psd",
      "boundary_weight": 50.0,
      "hard_frame_ratio": 0.3,
      "error_replay_every": 200,
      "eval_every": 5,
      "use_swa": True,
  }
  ```
- Registered in `PROFILES` dict at `src/tac/profiles.py:3846`

**What's missing:**
- No `scripts/remote_lane_psd_standard.sh` exists (verified `ls scripts/remote_lane_psd*.sh` returned no matches)
- No controlled experiment run on contest-CUDA verifying PSD baseline reproduces or exceeds Lane G v3 (1.05 contest-CUDA)
- No tests/test_lane_psd*.py

**Dispatch readiness gap:**
1. Adapt `scripts/remote_lane_a_optimized.sh` template → `scripts/remote_lane_psd_standard.sh`
2. Adversarial review (3-clean-pass per protocol)
3. Modal A10G or T4 dispatch (~$5/12h training)
4. Auth-eval against Lane G v3 baseline

**Council prediction band**: per project_grand_council_final_designs_20260429.md PSD has high PoseNet risk — predicted [1.10, 1.40] standalone, but if it lands at <1.20 it bolts onto the SC++/q_faithful base for an additional 0.005-0.015 distortion gain.

## Lane 8: multi-pass inflate 🟡 PARTIAL (was ❌ NOT STARTED)

**What exists:**
- `src/tac/trick_stack.py:377` `_stage_multi_pass()` — runs the postfilter model N times with uint8 rounding between passes
- `submissions/robust_current/inflate_postfilter.py` consumes the trick_stack
- Profiles at `src/tac/profiles.py:1364, 1388, 1405` set `use_multi_pass=2 or 3`

**Current implementation scope:**
- ONLY the POSTFILTER multi-pass (postfilter is a CNN that runs after AV1 decode in the legacy pipeline)
- NOT the full renderer-pipeline multi-pass that the 6-month plan envisioned
- `inflate_postfilter.py` is the LEGACY inflate path; the canonical Lane G v3 / Lane A pipeline uses `inflate_renderer.py` which doesn't go through trick_stack at all

**Lane 8 (full vision per 6-month plan, weeks 13-24) requires:**
- Generalization beyond postfilter: multi-pass over `inflate_renderer.py` decode → render → re-encode loop
- Compress-time iteration with score-feedback (inflate → score → refine archive content → re-inflate)
- Convergence criterion (score plateau, max iterations, wall-clock cap)
- The output archive bytes are deterministic; only the COMPRESS-TIME multi-pass is unbounded
- Inflate at deploy time stays single-pass + scorer-free + within 30-min T4 bound

**Math**: multi-pass at compress time can iteratively refine masks/poses/codebooks via:
```
for iter in 1..max_iter:
    rgb = inflate(archive)
    seg_loss, pose_loss = score(rgb, gt)
    grad = backprop(seg_loss + pose_loss → archive_content)
    archive_content = update_step(archive_content, grad)
    if score_plateau(seg_loss, pose_loss, patience=5): break
```

This is essentially a meta-TTO: existing `experiments/optimize_poses.py` does this for poses ONLY; Lane 8 generalizes to all archive content.

**Dispatch readiness gap:**
1. New module `experiments/multi_pass_inflate_optimizer.py`
2. Wraps `inflate_renderer.py` as a differentiable inner step
3. Outer loop: load archive → inflate → score → backprop → update {masks, poses, codebooks} → re-encode → repeat
4. Adversarial review (3-clean-pass — particularly Tao perspective on convergence criterion + Karpathy on instrumentation)

**Council prediction**: codex 6-month plan Section 3 said unlimited compress-time should buy SCORER-MARGINS (Lane 19 SegNet logit-margin) before byte polish. Multi-pass inflate is the meta-loop that uses scorer-margins; could buy 0.005-0.02 score per pass for the first 3-5 passes.

## Updated Phase 1 status table

| # | Lane | State (was) | State (now) |
|---|---|---|---|
| 1 | SC++/q_faithful base | 🔵 RUNNING | 🔵 RUNNING (6 Modal apps) |
| 2 | Archive diet pose-delta | 🟡 partial | ✅ DISPATCH-READY (commit 2d913687 wired encode side) |
| 3 | STC hybrid CUDA | 🟡 needs CUDA validation | 🟡 unchanged |
| 4 | Ω-W water-filling | ✅ READY | ✅ unchanged |
| 5 | LCT | ✅ READY | ✅ unchanged |
| 6 | GP rerun | 🟡 needs Modal dispatch | 🟡 unchanged (Fix-A landed 8746793e) |
| 7 | PSD standard | 🟡 needs profile review | 🟡 PARTIAL (profile exists, NO dispatch script — implementation gap) |
| 8 | Multi-pass inflate | ❌ NOT STARTED | 🟡 PARTIAL (postfilter multi-pass exists in trick_stack; needs renderer generalization) |

## Bug-class hardening landed today

Permanent fix added in commit `dc44b305`: STRICT preflight check 81 `check_silent_default_audit_clean` extinguishes the KL-distill silent-default override class. The audit (hardened in 4eeb6452) now correctly reports 0 CRITICAL after the 3 real-bug fixes (256c5e42).

## Cross-refs

- project_phases_2_3_4_design_implementation_math_provenance_20260429.md (the original status table)
- project_6month_strategic_plan_20260429.md (Phase 1-4 high-level plan)
- src/tac/trick_stack.py:377 (existing _stage_multi_pass for postfilter)
- src/tac/profiles.py:168 (PSD_STANDARD_ADAPTIVE profile)
- experiments/optimize_poses.py (existing pose-only TTO that Lane 8 generalizes)
