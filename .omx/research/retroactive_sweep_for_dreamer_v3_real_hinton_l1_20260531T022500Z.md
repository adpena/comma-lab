# Retroactive sweep — DreamerV3 RSSM real-Hinton L1 wire-in (2026-05-31)

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
the Catalog #348 event-driven retroactive-verdict-taint discipline (applied to
a wire-in landing, not a new STRICT gate — this sweep is the honest
reactivation check the wire-in triggers).

## Bug-class symptom signature

The wire-in fixes the DreamerV3 RSSM `_full_main` **pose-blind default path**:
pre-fix the trainer routed through the canonical `mlx_score_aware_full_main`
harness with `distillation_weight` only and NO `scorer_teacher` /
`pose_scorer_teacher`, so the default `score_aware_loss` path used the
scorer-BLIND mock SegNet cosine and emitted NO `pose_distill` term. The
DreamerV3 categorical-posterior was therefore trained against a
reconstruction-proxy + scorer-blind seg surrogate, with the dominant-at-frontier
pose axis at 0 = phantom-provenance per Catalog #322. This is the IDENTICAL
class the Z7-Mamba-2 Wave N+11 run hit (commit `8fa8fcfda` landing memo).

## Pre-fix window

DreamerV3 RSSM `_full_main` from the MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27
landing through 2026-05-30 (the L0/L1 scaffold era). No real-teacher long MLX
run had been fired against this substrate; the only canonical-equation anchor
on `categorical_posterior_capacity_vs_continuous_gaussian_v1` (N=1) is the
design-derivation anchor, NOT an empirical real-teacher pose-axis anchor.

## Historical KILL / DEFER / FALSIFY search

Searched memory + research ledgers for DreamerV3 / C6 IBPS / categorical-
posterior KILL or FALSIFIED verdicts that the real-teacher wire-in would
reactivate:

- **C6 IBPS v1 SegNet-collapse @ 105.15 contest-CUDA** (the `c6_e4_mdl_ibps`
  substrate landing): IMPLEMENTATION-LEVEL falsification per Catalog #307 — the
  continuous-Gaussian 24-dim IB bottleneck collapsed segmentation. The
  DreamerV3 categorical posterior (192-bit, cannot mode-collapse) is the
  canonical class-shift RESPONSE to that falsification, NOT a re-attempt of the
  same implementation. The real-Hinton wire-in does NOT reactivate C6 IBPS v1;
  it advances the distinct DreamerV3 substrate. No verdict taint.
- **DreamerV3 RSSM advisory ~95.7** (referenced in the substrate module
  docstring): an EARLIER scorer-blind / reconstruction-proxy advisory number
  PRE the real-teacher wire-in. This wire-in is exactly the fix for why that
  advisory was scorer-blind. The advisory is NOT a KILL verdict; it is the
  pre-real-teacher baseline the long run supersedes. No verdict taint — the
  number was never promoted (research-signal only).

**HONEST verdict: 0 historical KILL/FALSIFY verdicts are tainted by this
wire-in.** The wire-in reactivates the DreamerV3 substrate from
scorer-blind-default to real-scorer-bound, consistent with the Z7-Mamba-2 real-
Hinton precedent. The pre-fix advisory numbers were never promoted, so there is
no false-authority anchor to supersede.

## Per-finding RE-EVAL priority

| Finding | Class | RE-EVAL priority | Rationale |
|---|---|---|---|
| C6 IBPS v1 SegNet-collapse 105.15 | IMPLEMENTATION-LEVEL (#307) | NONE | Distinct substrate; DreamerV3 IS the class-shift response |
| DreamerV3 advisory ~95.7 (pre-real-teacher) | research-signal, never promoted | SUPERSEDED-BY-LONG-RUN | The real-teacher long run is the canonical successor anchor |

## Self-protection already in place

The fail-closed structural protections that prevent the pose-blind class from
recurring are ALREADY canonical (no new gate needed for this wire-in):

- `RendererBundle.__post_init__` fail-closes when `distillation_weight > 0` but
  no real `scorer_teacher` + `learnable_student_head` AND
  `allow_mock_scorer_teacher` is False (the C6 IBPS scorer-blind trap).
- `RendererBundle.__post_init__` fail-closes when a SegNet teacher is bound but
  NO PoseNet teacher (the SegNet-only +10.6 pose-drift trap), unless
  `allow_segnet_only_research` is explicitly set.
- `tools/register_dreamer_v3_rssm_real_hinton_pose_axis_anchor.py` fail-closes
  on `pose[ep0] <= 0` (refuses to register a mock run as the real anchor).

These three fail-closed surfaces are the structural extinction of the
pose-blind-default class for DreamerV3, mirroring the Z7-Mamba-2 protections.

Lane: `lane_dreamer_v3_rssm_real_hinton_l1_long_mlx_20260530`
Commit: `dac9f12e3` (trainer + tests).
