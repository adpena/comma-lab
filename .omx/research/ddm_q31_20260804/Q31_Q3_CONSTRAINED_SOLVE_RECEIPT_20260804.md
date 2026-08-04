---
arm: ddm_q31
title: "Q3-constrained Road/Lane solve from the start"
utc: 2026-08-04
axis: "[macOS-CPU advisory / CPU Torch SegNet+PoseNet bounded n32]"
research_only: true
score_claim: false
promotion_eligible: false
n600_run: false
pointer_moved: false
tokens: "[no-triality] [p0-ledger-ok]"
---

# q31 - Q3-Constrained Solve Receipt

## Answer First

**Q3-first did not clear the fork.** On the fixed se2 n32 qo1 surface,
the Q3-constrained solve corrected **2,858 / 12,407** Road/Lane target
cells:

| metric | value |
|---|---:|
| target survival | **0.2303538325** |
| ED1 break-even survival | 0.6964303814 |
| fraction of ED1 bar | **0.3307636178** |
| baseline subset flips | 26,054 |
| Q3-constrained subset flips | 21,703 |
| global net flip reduction | **4,351** |
| d_pose mean before | 0.0006482857 |
| d_pose mean after | 0.0006757690 |
| d_pose mean ratio | **1.0423937344** |
| max per-pair d_pose ratio | 1.4512419594 |
| stop census | 32 / 32 `iteration_cap_best_at_cap` |

Verdict: **`Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE`**.
Scope: this q31 formulation - qo1 fixed n32 Road/Lane targets, Q3 hard
projection every gradient/parameter step, 2x2 block snapping, `lr=2.0`,
`steps=50`, `starts=dec`. It is not a family-wide Q3 negative, because every
row is still cap-best at step 50. It is also not a green row: even ignoring the
non-flat pose residual, target survival is only **33.1% of ED1's break-even**.

No n600 scorer spec was appended to `.omx/research/scorer_batch_20260804.md`
because the pre-registered bar did not clear.

## Comparators

| comparator | q31 reading |
|---|---:|
| se2 `r0_delta_32` target survival 0.263238 | q31 is 0.8751x |
| se2 project-after-Q3 target survival 0.017007 | q31 is 13.5446x |
| se2 `r0_delta_32` global net reduction 1,563 | q31 is 2.7837x |
| sq1 unconstrained eta-25 0.7895095949 | q31 target survival is 0.2918x |
| sq2 unconstrained eta-50 0.8620042644 | q31 target survival is 0.2672x |

The sq1/sq2 eta comparison is only a route comparator: sq1/sq2 used the sq1
band denominator, while q31 uses se2's Road/Lane target denominator.

## Inputs And Controls

Matched base: qo1 `sub_auto_pairbit`
`/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
sha256 `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`,
357,836 B.

Selection: fixed se2 n32, seed `20260804`, pairs:

`31, 43, 62, 82, 94, 118, 147, 165, 167, 182, 185, 200, 237, 241, 247, 259, 272, 286, 288, 292, 296, 306, 327, 382, 390, 419, 473, 488, 525, 555, 560, 581`.

Denominators:

| denominator | value |
|---|---:|
| selected pairs | 32 |
| selected Road/Lane target cells | **12,407** |
| n600 Road/Lane target cells | 235,148 |

Controls:

- qo1 archive SHA matched the se2 baseline.
- For every measured pair, decoded qo1 SegNet argmax matched
  `cx1_argmax_n600.npy`.
- For every measured pair, GT decoded through `frame_utils.yuv420_to_rgb`
  matched `gt_argmax_n600.npy`.
- MPS was not used.
- No archive was built, and no full-n600 scorer slot was consumed.

## Q3/Pose Control

The Q3 projector was applied to every gradient and every parameter step, with
the target mask snapped to whole 2x2 scorer blocks before uint8 realization.
The control was not perfectly flat after integer/clipped realization:

| pose/control field | value |
|---|---:|
| d_pose mean delta | +0.0000274833 |
| d_pose mean ratio | 1.0423937344 |
| max per-pair ratio | 1.4512419594 (pair 200) |
| max `clipped_channel_values_pre_uint8` in eval curves | 212 |
| max yuv6 residual dY | 28.23 |
| max yuv6 residual dU | 7.875 |
| max yuv6 residual dV | 10.5874 |

So this is **not** a clean "d_pose-flat proof" for a solved q31 field. The
pose caveat does not change the routing verdict: the seg reach already fails
the ED1 survival bar by about 3.02x before charging any pose residual.

The high-lr smoke (`lr=6.0`, pair 31) produced target survival 0.3987 but
d_pose ratio 4.0368, demonstrating that clipping can invalidate the pose
control. The landed q31 run used `lr=2.0`; pair 31 then measured target
survival 0.2134 with d_pose ratio 1.0019.

## Stop Census

Every pair reported `iteration_cap_best_at_cap`. This is a cap-bound floor,
not a convergence claim:

| stop reason | rows |
|---|---:|
| `iteration_cap_best_at_cap` | 32 / 32 |

Per #874, the cap is explicit and carried into the verdict. A deeper or
box-constrained Q3 optimizer may improve this floor, but this run does not
license an n600 spend or a positive #934 existence-hinge claim.

## Receipts

| file | bytes | sha256 |
|---|---:|---|
| `experiments/ddm_q31_q3_constrained_solve.py` | 26,275 | `a242fd1fd7c0545987fe0892a7a5fb4a56f2524c3c2bc7615c56a0607149a508` |
| `.omx/research/ddm_q31_20260804/q31_summary.json` | 157,836 | `ab641c046912795c923995175d8302799efb5383f6e663d17bc823884f3c44fe` |
| `/Volumes/VertigoDataTier/pact/ddm_q31_20260804/q31_rows.jsonl` | 112,311 | `9ef63600d488ac6fedb65632552b0602574ed58aad82335971c5275b681916a9` |
| `/Volumes/VertigoDataTier/pact/ddm_q31_20260804/q31_summary.json` | 157,836 | `ab641c046912795c923995175d8302799efb5383f6e663d17bc823884f3c44fe` |

Run command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_q31_q3_constrained_solve.py --resume
```

Syntax check:

```bash
.venv/bin/python -m py_compile experiments/ddm_q31_q3_constrained_solve.py
```

Result: passed.

## NEXT-IF-RESUMED

Do not relaunch this exact q31 run. It completed all 32 prescribed rows.

If q31 is reopened, the next useful unit is a stricter box-constrained Q3
solver that prevents clipping from leaving the null subspace, then runs deeper
than 50 steps with the same se2 denominator and reports both target survival
and pose residual. Fire no n600 scorer step unless the bounded survival clears
ED1's 0.6964303814 bar with d_pose held near-flat.

Own-vehicle frontier: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved.
