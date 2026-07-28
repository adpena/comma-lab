# ddm_rp1 — range(A)-cell realized probe: project → uint8 → real SegNet/PoseNet

**Date:** 2026-07-28 · **Arm:** `ddm_rp1_20260728`
**Evidence axis:** `[macOS-CPU frozen-scorer advisory — real upstream/modules.py SegNet+PoseNet forward]`
**`score_claim=false · promotion_eligible=false · rank_or_kill_eligible=false`.**
Canonical frontier pointer **0.1910828242 UNMOVED**; nothing here is a frontier or submission claim.

**STORES CONSULTED:** `CLAUDE.md` (NO-FAKE supreme rule; measured-scored-quantity axis; SSD-first + `/tmp`-free
evidence; pointer-only frontier; MPS-never-authority) · `MEMORY.md` current-state incl.
`box_retired_min_s_target_warp_family_closed_1273_bytes_per_error_20260728`,
`null_subspace_rate_measure_20260717`, `realization_is_quantization_gated_minimal_writes_die_at_uint8_20260720`,
`frozen_scorer_exact_factorization_20260715`, `objective_is_min_S_over_solution_set_not_box_or_point_20260724`,
`shared_venv_editable_install_hijack_from_arm_worktree_20260724` ·
`.omx/research/r6cal_solved_object_byteclose_eval_20260727.md` (independent custody of the box-solve object) ·
`.omx/research/null_compiler_full_kernel_20260720T163500Z.md` (#580 full-kernel projector, nullity 80.6742%) ·
`src/tac/optimization/resize_full_kernel.py` (`FullResizeKernel`) ·
`src/tac/optimization/resize_null_preimage.py` (`ResizeProjector`, A parity to `F.interpolate`) ·
`upstream/modules.py` + `upstream/frame_utils.py` (authority scorer forward) ·
`ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/stage_checkpoints/{01_rate,03_solve,02_scorers}`.

---

## Verdict first

1. **CUSTODY HEADLINE — the charter's C0-on-1.52e-4 is BLOCKED; probe re-scoped honestly.** The
   "1.52e-4 exact-solve object" (q1) is a **MEASURED SCORER CONTROL with ZERO materialized frame records
   on disk** — independently re-confirmed here (matches r6cal 07-27): `01_rate/` holds only q4/q8 box-solve
   chunks (100 files → 1,200 records, d_seg 1.16e-3), `03_solve/exact_binary_solve.json` is DP metadata
   (`selected_steps`), and `02_scorers/scorer_measurement.json` records `q1_exact_control` as
   `MEASURED_EXACT` (d_seg 1.5200e-4, d_pose 1.018e-4, 17,931 errors) with
   `all_candidate_planes_realized_by_exact_constructor=true` — i.e. **constructor-realizable but never
   persisted**. Per the charter honest-boundary clause the probe therefore runs on the **largest custodied
   real substrate: the GT frames** (`gt_n600.npz`), whose SegNet argmax IS the `lstars` cell partition —
   the **highest-margin / OPTIMISTIC-bound operating point** (a break here is decisive; a hold is
   necessary-not-sufficient, but see result 3, it is also an existence proof).

2. **CELLS HOLD (n600, measured through the real decode).** The range(A)-carrier zero-ker uint8 lift
   `Y = round(clip(project_range(X),0,255))` — the #580 min-norm camera preimage of the scorer image A(X),
   decoder-derivable from the carried range content alone — reaches the GT argmax cells at
   **C1 d_seg = 3.6296e-4 = 2.39× the q1 target 1.52e-4** (inside the pre-registered ~2–3× HOLD band) with
   **C1 d_pose = 3.965e-4, contribution √(10·d_pose) = 0.063** (inside the R1 pose tube of 0.127). Both
   scored through the **real `upstream/modules.py` frozen SegNet/PoseNet** on all 600 pairs, C0 custody
   verified (GT → `lstars`, **0 flips**).

3. **The lift is an EXISTENCE PROOF, not just a proxy — this is what makes result 2 load-bearing.** `Y`
   is a legal range(A)-only + zero-ker + uint8 realization that a decoder synthesizes for free from A(X);
   it lands in the cells at 2.39× q1. So the **range-carrier+uint8 FORMULATION demonstrably reaches the
   cells at ~q1 precision — it is NOT broken at uint8.** Per the charter's pre-registered routing this
   makes the sc1-far seed (W_joint d_seg 0.070519 = 464× q1) an **ENGINE-CAPACITY failure, not a
   formulation break** → the **family-d GN-in-description-coordinates build is the named next arm**.

4. **Mechanism confirmed (margin slack absorbs the uint8 break) + #532 independently reproduced.** The
   scorer-space realization break `max|A(round(clip(project_range X))) − A(X)| = 63.82` **reproduces #532's
   Δ=62.74**; the float range projection is A-exact to `1.9e-11` (cf #532 1.7e-13). Despite this 63.8
   scorer-space break, flips concentrate **entirely at near-zero pre-round margin (0.034 at flipped sites
   vs 5.61 at held sites — a 166× gap)**: the argmax slack absorbs the break everywhere except razor-margin
   boundary pixels. Per-class flip physiology matches the natural d_seg residual (no pathological new mode).

---

## 1. Custody (independently re-derived)

| object | seg errors | d_seg | d_pose | frames byte-closed on disk? |
|---|---:|---:|---:|---|
| **q1 exact control** ("the 1.52e-4 object") | 17,931 | 1.5200e-4 | 1.018e-4 | **NO — measured scorer control, 0 records** |
| box-solve (the shipped 277.7 MB `archive.zip`) | 136,839 | 1.1600e-3 | 1.6633e-2 | yes (277.7 MB, inflate ≈382 s) |
| **GT frames (THIS probe's substrate)** | 0 (defines cells) | 0 (custody-verified) | 0 | yes (`gt_n600.npz`, 4.84 GB) |

Substrate: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
(`gt_f0`/`gt_f1` (600,874,1164,3) uint8; `lstars` (600,384,512) int64 = SegNet argmax cells;
`margins` (600,384,512) f32 = SegNet top1−top2; `gt_poses` (600,6) = PoseNet[:6] targets).
Scorer custody: `upstream/models/{segnet,posenet}.safetensors`, `upstream/modules.py`.
**Operator custody (my A):** `ResizeProjector.project_plane` reproduces SegNet's internal
`F.interpolate(size=(384,512), bilinear, align_corners=False)` to **6.6e-13 in float64** (the 2.6e-3 gap
vs SegNet's live forward is pure torch-float32 roundoff — present in C0, which still yields 0 flips, so the
argmax already absorbs float32 resize noise). Shared-venv hijack noted: `tac` resolves to main's src; only
UNMODIFIED reused modules (`FullResizeKernel`, `ResizeProjector`) are imported, so this is inert here.

## 2. Conditions & the C2-degeneracy structural finding

Per pair, camera-space frame X (uint8). A = the exact bilinear resize (camera 874×1164 → scorer 512×384).
`project_range(X)=Q_h X Q_w` (the 19.33% A sees), `project_kernel(X)=X−project_range(X)` (the 80.67% A kills).

- **C0 (control):** X → real SegNet argmax vs `lstars`. GT: **0 flips** (custody PASS).
- **C1 (THE probe):** `Y = round(clip(project_range(X),0,255))`. `project_range(X) = A⁺A(X)` is the exact
  min-norm (zero-ker) camera preimage of the scorer image; rounding it to uint8 is #532's naive
  range-carrier lift with a generic/minimal ker, **decoder-derivable from A(X) for free.** Scored.
- **C2:** `round(project_range(X)+project_kernel(X)) = round(X) = X ≡ C0`. **Degenerate BY CONSTRUCTION,**
  verified numerically: `max|project_range(X)+project_kernel(X) − X| = 0.000e+00` (float64). Because the
  solve is over uint8 (`exact_binary_solve`), the projection round-trip of an integer frame is float-exact,
  so **all realization damage lives in C1** — the charter's C1/C2 split (which assumed a float solve to
  isolate "projection+rounding" from "ker-swap") collapses: for an integer solved object there is no
  separate projection-rounding damage; the entire realization question is C1 (the ker-swap / range-only
  lift). This is a real structural correction to the charter's mental model, not a measurement gap.

## 3. THE ROW — n600 measured (real SegNet/PoseNet, chunked 5×120, resumable)

| condition | d_seg (n600) | ×q1 | d_pose (n600) | √(10·d_pose) | verdict band |
|---|---:|---:|---:|---:|---|
| C0 (GT control) | 0.0 | — | 0.0 | 0.0 | custody PASS (0 flips) |
| **C1 range-carrier zero-ker uint8 lift** | **3.6296e-4** | **2.39×** | **3.965e-4** | **0.063** | **within ~2–3× HOLD band; pose within tube** |
| C2 (projection round-trip) | ≡ C0 (identity=0) | — | ≡ C0 | — | degenerate (uint8 solve) |

Totals: 42,816 flips / 117,964,800 sites. Per-pair C1 d_seg: min 4.58e-5, median 3.31e-4, p90 5.75e-4,
max 4.20e-3; **41.8% of pairs ≤ 2× q1, 78.3% ≤ 3× q1.** Elapsed ≈ 525 s (0.87 s/pair, SegNet-bound).

### 3.1 Per-class flip physiology (matches the natural d_seg residual — no new failure mode)

| class | flips | share of flips | per-class flip rate |
|---|---:|---:|---:|
| 0 Road | 19,696 | 46.0% | 7.19e-4 |
| 2 Undrivable | 12,902 | 30.1% | 2.21e-4 |
| 3 Movable | 5,708 | 13.3% | **3.91e-3** |
| 1 Lane | 3,076 | 7.2% | **4.45e-3** |
| 4 MyCar (ego-hood) | 1,434 | 3.3% | **4.78e-5** |

Road/Undrivable carry the flip MASS (76%); the thin/boundary classes Lane + Movable carry the highest
per-class RATE (~4e-3); the static MyCar hood core is the most robust (4.78e-5) — exactly the canonical
d_seg flip distribution. The range-carrier lift does **not** manufacture a pathological class-specific
failure; it re-expresses the same boundary long-tail.

### 3.2 Margin-erosion telemetry — DOES SLACK ABSORB IT? (yes)

| site set | pre-round GT margin (mean) | post-round C1 margin (mean) |
|---|---:|---:|
| FLIPPED sites | **0.0337** | 0.0330 (new-winner side) |
| HELD sites | **5.6136** | — |
| ratio held/flipped | **166.5×** | — |

Flips live entirely at the smallest pre-round margins (0.034 vs 5.61). The uint8 realization break
(63.82 in scorer space) is **absorbed by argmax slack at every site whose margin exceeds the induced
logit shift**; only the ~0.036% of sites already at near-zero margin cross. This is the crux chart's
"ker(A) free, communicate range(A)" claim vindicated at uint8: a large camera/scorer-space realization
break stays invisible to the argmax except at razor-margin boundaries.

### 3.3 Realization-break magnitudes (independently reproduces #532)

| quantity | measured | reference |
|---|---:|---|
| A-space float range exactness `max\|A(project_range X)−A(X)\|` | 1.9e-11 | #532: 1.7e-13 (float64 vs their fp) |
| **A-space uint8 break `max\|A(round(clip(project_range X)))−A(X)\|`** | **63.82** | **#532: Δ=62.74 ✓** |
| camera-space uint8 break `max\|round(clip(project_range X))−project_range X\|` (n600 mean) | 113.7 | project_range overshoots [0,255] |

## 4. Typed verdict + verdict_scope

**VERDICT: CELLS HOLD** (pre-registered condition met: C1 d_seg 3.63e-4 = 2.39× q1 ∈ [2×,3×]; d_pose
contribution 0.063 ∈ tube). → **The linearized crux chart survives at uint8.** The range-carrier+uint8
representation demonstrably reaches the SegNet argmax cells at ~q1 precision → the sc1-far seed
(d_seg 0.070519) is an **ENGINE-CAPACITY** failure, not a formulation break → **the family-d
GN-in-description-coordinates build is the named next arm** (the charter's pre-registered HOLD routing).

**`verdict_scope` (honest boundaries — this is a HOLD on the OPTIMISTIC bound, plus an existence proof):**
- Measured on **GT frames** (highest-margin operating point), NOT on the q1 solved frames (unmaterialized).
  The HOLD is therefore, strictly, "the range-carrier+uint8 formulation reaches the GT cells at 2.39× q1"
  — an **existence proof** that the formulation is capable, which is exactly what the charter's HOLD
  routing requires ("if the cells hold under projection, the failure is engine capacity"). It is NOT a
  measurement of the range-carrier lift at the **box-solve's smaller-margin operating point** (that object
  spent its margin budget). Because flips are governed purely by pre-round margin (§3.2), the smaller-margin
  operating point would flip somewhat more — but the mechanism (margin absorbs a 63.8 break) is robust.
- **NAMED NEXT MEASUREMENT (confirms the verdict at the true operating point):** inflate the box-solve
  277.7 MB archive (receiver `v10_production_receiver`, ≈382 s) → run the IDENTICAL C1 probe on the
  real box-solve frame_1's → report C1 d_seg vs the box-solve baseline 1.16e-3 and the margin distribution
  at that operating point. The tool (`tools/measure_ddm_rp1_rangeA_cell_probe.py --substrate boxsolve`) is
  wired with a `NotImplementedError` inflate stub to be closed by that arm.
- All rows `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, pointer UNMOVED. The verdict moves
  no exact score — it **routes the next build** (family-d GN) and retires the "formulation broken at uint8"
  hypothesis for the range-carrier representation.

## 5. Wire-in / hooks
Sensitivity-map: N/A (probe, no new byte allocation). Pareto/bit-allocator: N/A. Cathedral autopilot:
N/A (no dispatch). Continual-learning: this memo + DAG FEED are the anchor; the CELLS-HOLD→engine-capacity
routing feeds the family-d GN arm. Probe-disambiguator: this IS the disambiguator between
"formulation-broken-at-uint8" and "engine-capacity" (result: engine-capacity).

## 6. Artifacts
- Tool: `tools/measure_ddm_rp1_rangeA_cell_probe.py` (this branch).
- Per-chunk receipts: `/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/chunks/chunk_gt_{0000_0120…0480_0600}.json`
  (schema `ddm_rp1_rangeA_cell_probe_chunk.v1`, per-pair flips/d_seg/d_pose/per-class/margins).
- Run log: `/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/gt_run.log`.
