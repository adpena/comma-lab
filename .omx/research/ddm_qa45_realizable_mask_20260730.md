---
schema: ddm_qa45_realizable_mask.v1
date_utc: 2026-07-30
arm: ddm_qa45 (realizable-mask transfer probe; QA45, gate-opener for grammar v4b)
lane_id: "lane_ddm_qa45_realizable_mask_20260730"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver + frozen PoseNet; composed byte-close + n600 evaluate gate OWED]"
operator_binding: "MAIN QA45 dispatch — measure the GT↔realizable mask gap before the v4b build"
tool: "experiments/ddm_qa45_realizable_mask_probe.py"
data: "SSD ddm_qa45_20260730/{realizable_mask_probe.partial.jsonl (112 rows), ddm_qa45_aggregate_receipt.json (sha aa38d6f0bdca43b0)}"
---

# ddm_qa45 — realizable-mask transfer probe: the static horizon mask BEATS the GT upper bound

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`, per-pair realized through the
real receiver + frozen PoseNet. This arm characterizes the pose member of an ADVISORY
vehicle (the pfs1 warp base, S≈2.2566) far from the pointer; it does NOT move the pointer.
The composed byte-close + n600 `evaluate.py` gate is still OWED (v4b build).

## §1 The problem this arm answers (ck1 §2/§5 + qa43 §1)

Every two-plane d_pose measured so far (qa43 selection 41.357, ck1 recovery-parity) routed
far/ground/hood by the **GT `lstars` class mask** (frame_1 SegNet argmax, 384×512
nearest-upsampled to 874×1164). GT masks are **ILLEGAL at inflate** (no scorers at decode,
GT not shipped) → qa43/ck1 correctly labelled their two-plane numbers an **UPPER BOUND**.
A realizable receiver may only use: (a) a STATIC geometric mask (pure code, rule-118 FREE,
0 bytes); (b) the decoded partition the archive already ships (0 marginal bytes, needs the
seg-cell decoder wired); (c) a small shipped mask refinement (counted bytes).

## §2 The derived realizable mask (physics, not semantics)

The receiver's ground plane is `n = [0, −cos(pitch), −sin(pitch)]`, pitch=0 → `n=[0,−1,0]`.
The ground-plane **vanishing line** in the image is `l = K⁻ᵀ n ∝ [0, −1, 437]` → the
**horizon row v = cy = 437** (EON intrinsics fy=910, cy=437). Physics: content ABOVE the
horizon genuinely recedes to infinity (no ground-plane parallax → far → H∞ = K·R·K⁻¹ at
s_t=0); content BELOW can carry ground parallax → full H. This is a **static-global** split
(same row for every pair, every frame) = **pure code, 0 bytes**. Hood (ego car, static) →
identity; the hood region is a tiny video-derived majority-class-4 bitmap (76 B brotli at
384-res, COUNTED if shipped) or a free bottom-rectangle prior.

## §3 STAGE-1 transfer result ($0, 112 tail pairs, ~8 PoseNet fwd/pair, ~1.1 s/pair)

Take each pair's ALREADY-SOLVED `p_two_star` (qa43, GT-optimized) and re-evaluate d_pose
under realizable masks WITHOUT re-solving. Selection = Σ min(d_single_cached, d_candidate).

| candidate | two-total | **selection** | wins | sel/GT |
|---|---:|---:|---:|---:|
| gt_control (illegal UB) | 298.004 | **41.357** | 95/112 | 1.000 |
| static h291 + hood (ledger latent-8 prior) | 1828.2 | 84.640 | 9 | 2.047 |
| static h350 + hood | 1248.4 | 84.991 | 7 | 2.055 |
| static h400 + hood | 280.9 | 53.166 | 75 | 1.286 |
| **static h437 + hood (DERIVED cy)** | 180.9 | **22.464** | **102** | **0.543** |
| static h480 + hood | 838.0 | 84.331 | 10 | 2.039 |
| static h437 + hood_rect (FREE) | 181.6 | 22.734 | 103 | 0.550 |
| static h437 + hood_none (FREE) | 161.2 | 22.734 | 103 | 0.550 |

- **GT-control reproduced the cached d_two_solved to 0.00e+00 on all 112 pairs** (substrate
  identity — the harness IS the qa43 instrument; my realizable numbers are trustworthy).
- **DECISION RULE (static wins if selection ≤ 1.15×GT = 47.56): PASSED, and then some** —
  the realizable static mask lands at **0.543× GT (22.464 vs 41.357), a −45.7% improvement
  OVER the illegal upper bound.** The "GT upper bound" was **not a true bound**: the GT
  argmax is a SEMANTIC mask (noisy jagged class boundaries, below-horizon barriers, argmax
  flicker at the skyline); the physics-derived horizon is a strictly BETTER far/ground
  partition for the parallax warp. The GT-optimized `p_two_star` still scores lower through
  the smooth static mask on most pairs.

## §4 The mechanism is real — the horizon curve is a positive control

Selection vs static horizon row: 291→84.6 · 350→85.0 · 400→53.2 · **437→22.5** · 480→84.3.
A sharp minimum AT the K-derived vanishing row; ±40 rows and it collapses to ~2× GT (assigns
real ground to far, or sky to ground). This is the signature of a REAL geometric mechanism,
and it confirms **v=437 is DERIVED, not tuned** (constants-are-poison clean — the value
falls out of K, no fitting). Hood barely matters on the tail selection (vdmaj 22.464 vs
free rect/none 22.734, a −0.27 / ~−0.005 S edge for 76 B — not worth shipping on the tail).

## §5 Composed pose axis (advisory; direct from P0 = pfs1 D2 6dof f16, all 600)

Contribution = √(10·mean d_pose over 600); non-tail(488) held at P0 (sum 8.6199, mean
0.01766); tail(112) = the selection.

| rung | mean d_pose | contribution | ΔS vs warp |
|---|---:|---:|---:|
| warp base | 0.22144 | 1.4881 | — |
| + P0 6dof f16 (+7.2 KB) | 0.15951 | 1.2630 | −0.2251 |
| + two-plane GT selection (UB, **illegal**) | 0.08330 | 0.9127 | −0.5754 |
| **+ two-plane STATIC selection (REALIZABLE, ~0 B)** | 0.05181 | **0.7198** | **−0.7683** |

**The realizable static mask gives −0.7683 S from warp — −0.1929 S MORE than the illegal
GT upper bound (0.9127→0.7198).** The realizability caveat that ck1/qa43 carried is not a
cost; it is a **gain**. (Non-tail two-plane extension over the 488 pairs is a further cheap
sweep, est. ≤−0.05 S, OWED.)

## §6 Stage-2 (realizable re-solve) — NOT needed for the win

Only 6 pairs degrade >1.5× AND lose to single (46, 156, 175, 244, 445, 485 — all pairs where
GT two-plane hit ~1e-3 via a fine below-horizon boundary the flat static horizon cannot
reproduce; min-selection with single-plane caps their damage). Recovering ALL six to
GT-level improves selection only 22.464→21.296 (−0.010 S). Since static already beats the UB
decisively, the stage-2 re-solve is **skipped** (velocity discipline; win banked at stage 1).
A FULL re-solve of `p_two_star` THROUGH the static mask (all pairs — the pose is currently
GT-optimized, suboptimal for static) would improve further and belongs at the v4b build, not
this probe.

## §7 VERDICT + v4b routing (the composed gate MAIN owns)

**RECOMMENDATION: v4b ships the STATIC-FREE mask (option a), NOT the decoded partition and
NOT a counted refinement.** It is 0 bytes, rule-118 free, physics-derived, and beats the GT
upper bound. Composed pose member (realizable): warp 1.4881 → **0.7198** at ≤7.3 KB marginal
(600×6×f16 field + ~75 B selector + rule-118-free multi-plane receiver code).

**v4b build spec (gap list for MAIN's composed gate):**
1. **Receiver far/ground split:** static horizon at `v = round(cy) = 437` from
   `intrinsics_native()` (derive `l = K⁻ᵀ·[0,−1,0]`, `v = −l₂/l₁`; pure code). Rows `< 437`
   → H∞ (s_t=0, s_r=1); rows `≥ 437` → full H. **s_r=1 receiver amendment required** (grammar
   v4 already prices this; D1 ships s_r=0).
2. **Hood:** OPTIONAL. Default = free bottom-rectangle (or omit; selection identical 22.734).
   The 76 B vdmaj bitmap buys only −0.27 selection on the tail — defer to a non-tail A/B.
3. **Pose field:** ship the qa43 `p_two_star` 6dof f16 (+~7.2 KB, same format as P0/D1).
4. **Selector:** per-pair single-vs-two-plane bit (~75 B, set at encode; always-two-plane
   blows up ~10 pairs — two-total 180.9 vs selection 22.5, the selector is REQUIRED).
5. **Byte-close + n600 evaluate gate** on the composed archive (Knee-A/base tokens + P0 +
   two-plane-static + selector). This measures the REAL composed S (seg + pose + rate); all
   numbers here are frozen-PoseNet advisory until then.
6. **Decoded-partition (option b) is NOT needed** — static-free already wins. It stays a
   fallback only if the s_r=1 static receiver ever regresses at the real gate; wiring the
   seg-cell decoder is deferred (not on the critical path).

## §8 Confounds + discipline

- **`tac` import HIJACK control-guarded:** the shared venv editable-install points to the eg1
  codex worktree (`.omx/tmp/codex_worktrees/ddm_eg1_endgame_chain_.../src/tac`). Forced
  `PYTHONPATH=$PWD/src:...` → resolves to this worktree's `src/tac`. The §3 GT-control
  positive control (0.00e+00 delta vs the qa43 cache) proves the runtime decode is
  byte-equivalent to the qa43 instrument, so the measurements hold. Hygiene → QD ledger.
- **Advisory everywhere:** frozen-PoseNet, macOS-CPU, non-promotable, `score_claim=false`.
  The GT-UB numbers stay labelled UPPER BOUND; my static numbers are REALIZABLE-but-advisory.
- **Verdict scope:** the static-wins finding is at the tail-112 (the pose mass); non-tail
  extension + the composed gate are OWED. The finding is FAMILY-level for the mask-source
  question (static geometric mask is legal AND superior), INSTANCE for the exact selection
  totals (this vehicle, frozen PoseNet, GT-optimized poses).
