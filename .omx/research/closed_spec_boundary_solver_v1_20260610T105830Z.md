# `closed_spec_boundary_solver.v1` — the SOLVE (not search) + exact-scorer verdict (task #55)

**Authority of every number below:** `[local CPU-torch advisory]` — exact upstream `DistortionNet`
(SegNet + PoseNet) on CPU, GT decoded via `upstream/frame_utils.yuv420_to_rgb` ONLY (PyAV rgb24 ==
~100× phantom pose). **NOT** the contest 600-sample harness → non-promotable per the GOAL authority
ladder. `$0` spend, no GPU, no paid dispatch, **NO MPS**.

**Spec executed:** task #55 (the operator's exact spec for `closed_spec_boundary_solver.v1`) +
`.omx/research/closed_spec_boundary_math_system_of_equations_20260610.md` (§4 the polytope system of
linear inequalities, §10 the water level λ\* = 1.27 B/flip). This is the offensive carrier's
seg-correction core: a **real SOLVE** (Gα≥b autograd + graph-cut + MDL admission) replacing lever G's
killed rule-family SEARCH (NO-FAKE class 6).

---

## 0. What was built (a real solver, three deterministic candidates — NOT a search)

`src/tac/boundary_math/boundary_solver.py` (the three candidates + the Gα≥b solve) + **30 behavior
tests** (`src/tac/boundary_math/tests/test_boundary_solver.py`, all green; ruff clean) +
`tools/closed_spec_boundary_solver_smoke.py` (the exact-scorer smoke).

**The architecture fact (lever F, reconciled):** storing the SegNet argmax partition DIRECTLY loses on
rate (524.8 KB seg-alone vs the 177 KB whole archive — amortization beats it, per
`boundary_math_seg_core_…`). So this is a **CORRECTION on a base**, NOT a partition store.

**The SOLVE (spec §4, the system of equations — NOT a sweep):** for every flip pixel `p` (where the base
argmax `A_b(p)` disagrees with the target `A_s(p) = L*(p)`) the first-order constraint is
`(J_{A_s,p} − J_{A_b,p}) · δ ≥ −m_p`. The correction is parameterised over a LOW-DIM boundary basis
`δ = Σ_k α_k φ_k` (smooth blob atoms, support-localized to the flip components), and the linear-inequality
system `G α ≥ b` is SOLVED for `α`:

- `G_{k} = ∂ m_{p_k}(α φ_k)/∂α` — the **real SegNet input-Jacobian** (one autograd pass; batched
  single-pass for disjoint supports). Measured, not assumed.
- `b_k = −m_p(0) + γ` — the margin gap to flip plus slack, from the real base logits.
- closed form `α_k = b_k/G_k` clipped to a box; infeasible (wrong-sign G) → α_k = 0.

**Verified the SOLVE works (the feasibility probe):** a single closed-form `α* = gap/G` from a
support-localized blob atom on the REAL SegNet **actually flips the predicted argmax toward the target**
(3→2, matching A_s). This is the spec §4 polytope solve, not a parameter search.

**The three candidates (operator's exact spec):**
1. `contour_normal` — flip components → support-localized correction atoms (closest to zero-byte lever G;
   `archive_bytes_delta = 0`, a deterministic decode-time field keyed by the base argmax).
2. `graph_cut` — RAG of the base partition; min-cut SELECTS the repair support at the KKT water level
   (`keep iff flips_fixed·1.27 > bytes`); selected components get closed-form atoms.
3. `mdl_contour` — flip components → real chain-code byte cost; **admit a component iff its score-value
   (flips·1.27 B) > its coded byte cost**; honest `archive_bytes_delta` per admitted component.

---

## 1. THE STRUCTURAL FINDING (the crux this build localizes, on the real scorer)

The frontier-base seg residual (d_seg ≈ 5.6e-4, ~114 flips/frame) is, measured on the exact SegNet:

| structure | value |
|---|---|
| flip components per frame | **104** |
| **single-pixel components** | **99 of 104 = 95%** |
| component size histogram | {1px: 99, 2px: 2, 3px: 1, 4px: 2} |

**The residual is scattered salt-and-pepper single-pixel boundary noise**, NOT contiguous patches. This
is the carrier's bicubic-upsampling boundary speckle at razor-thin SegNet decision boundaries (the
lever-G diagnostic median margin 0.156, 91.8% < 0.5).

**The collateral mechanism (why correction loses here):** the SegNet's deep receptive field COUPLES
neighbours. Perturbing a flip pixel (even a support-localized luma atom solved to flip it exactly) flips
~2 CORRECT neighbours the wrong way for each 1 it repairs. The **GT-snap upper bound** — replacing comp
frame1 with EXACT GT pixels on the dilated flip support (the strongest possible per-pixel data-table
correction) — creates **594 new bad flips for 58 repaired (net −536)**. No frame1-space correction can
fix isolated single-pixel argmax flips without net-negative collateral on the frontier base.

---

## 2. THE TYPED ROWS — `engineered_correction_boundary_solver_smoke.v1` (exact local-CPU-torch, 8 frames)

JSON: `experiments/results/closed_spec_boundary_solver_20260610/smoke_frontier_n8_pose.json`. Base =
`frontier_archive` (the 177,169 B contest-CPU frontier carrier's decoded frame1, d_seg≈5.4e-4). d_seg on
the exact SegNet argmax; d_pose on the exact PoseNet via the SAME correction upsampled to the camera
frame1; score recomputed from components.

| candidate | d_seg before→after | repaired | **new_bad** | **pose_side** | bytes Δ | **Δscore** | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `contour_normal` | 5.41e-4 → 7.73e-4 | 421 | **786** | **+0.0112** | 0 | **+0.0344** | WORSE (collateral) |
| `graph_cut`      | 5.41e-4 → 5.72e-4 | 51  | **100** | +0.0028 | 94 | **+0.0060** | WORSE (collateral) |
| `mdl_contour`    | 5.41e-4 → 5.41e-4 | 0   | 0       | +0.0000 | 0  | **+0.0000** | **DECLINED (correct)** |

**The honesty the operator demanded (without `new_bad_flips_created` + `pose_side_effect` the row lies):**
- `contour_normal` "repaired 421" LOOKS like a win, but **786 new bad flips** + **+0.0112 pose
  side-effect** make it a **net loss of +0.0344**. The `repaired` count alone is a lying number.
- `graph_cut`'s water-level cut dropped most components but the 47 it kept still net-lose
  (51 repaired, 100 new_bad) → +0.0060.
- `mdl_contour` is the **winning candidate at Δscore = 0**: every flip component is ≤4 pixels, so its
  value (≤4·1.27 = ≤5.1 B) is below its coded cost (header 4 B + chain ≥ ~7 B for a 1px component) →
  **all components dropped → zero correction → the solver correctly DECLINES to spend bytes on an
  unrepairable residual.** This is the solve working exactly as designed (it is the MDL/water-level KKT
  gate refusing under-water actions).

**The winning (base, candidate, net Δscore): `(frontier_archive, mdl_contour, +0.0000)`** — i.e. the
best a frame1-space correction can do on the frontier base is **decline** (no improvement, no harm). The
positive-Δscore candidates are dominated by their own collateral.

---

## 3. PRE-REGISTERED PREDICTION + KILL CRITERION (this build's verdict against them)

**PRE-REGISTERED PREDICTION (written from the spec, before the n=8 measurement):** the Gα≥b solve flips
target pixels (CONFIRMED — feasibility probe), but the frontier residual being razor-thin boundary noise,
a frame1-space correction will pay collateral; the MDL/graph-cut candidates that price collateral at the
water level will out-perform the unconstrained contour field.

**Result: CONFIRMED.** `mdl_contour` (the water-level-gated candidate) dominates `graph_cut` which
dominates `contour_normal` — exactly monotone in how strictly each candidate prices the collateral.

**PRE-REGISTERED KILL CRITERION:** if NO candidate achieves Δscore < 0 on the frontier base, the
frame1-space seg-correction lever is **DEFERRED** (not killed) on that base, with the finding that the
residual is irreducible without changing the BASE (the carrier), and the campaign must move to the
`lever_b_argmax_generator` base (bigger, possibly-more-contiguous residual).

**KILL CRITERION TRIGGERED → DEFER (per Forbidden-premature-KILL + Catalog #307 IMPLEMENTATION-LEVEL):**
no candidate beats Δscore=0 on `frontier_archive`. The **paradigm** (the polytope SOLVE) is PROVEN exact
(it flips targets in closed form); the **frontier base** is simply at its seg floor for frame1-space
correction (95% single-pixel flips, GT-snap upper bound is net −536). This is a DEFER of the
frontier-base correction lever, NOT a kill of the solver.

---

## 4. THE HANDOFF (per-component marginals → the waterfilling allocator, task #54)

The solver produces, per flip component, the EXACT marginal the §10 waterfiller needs:
`(comp_id, pixels, target, current, center, repair_value_bytes = flips·1.27, coded_bytes, net_collateral)`.
On the frontier base **every component is under-water** (value < cost OR net collateral > value), so the
allocator correctly funds NONE — the seg-correction marginal does not exceed λ\* = 6.66e-7 score/byte on
this base. The allocator's seg-correction input on `frontier_archive` is **empty**; its non-empty input
must come from a base with a repairable (contiguous, multi-pixel) residual.

**Reactivation criteria (the campaign's next build):**
1. **lever_b base (the campaign target):** the lever_b generator's residual is d_seg=0.00826 (~14× more
   flips than the frontier). Its target argmax (`gt_segnet_argmax.u8`) has ~38 regions/frame; the
   generator's OUTPUT residual must be characterized by running the trained generator forward (the
   campaign integration). IF the generator residual is contiguous multi-pixel patches (a smooth net's
   under-fit regions, NOT salt-and-pepper), the contour/graph-cut/MDL candidates can pay rent. This is
   the highest-value next probe: run the solver on the lever_b base.
2. **change the base, not the correction:** the frontier finding says the right move is a base whose
   residual is structurally repairable. The lever-B score-native generator (which learns the argmax
   directly) is exactly that base.
3. **lever D contour coder:** if a base has contiguous residual, the MDL candidate's chain-code cost can
   be pushed below 1.27 B/flip with an STC/UNIWARD boundary coder, crossing the water level.

---

## 5. The immediate-exact-eval question (operator pre-registration)

The operator's spec: "If frontier_archive base shows a real zero-byte Δscore < 0, that is an IMMEDIATE
exact-eval candidate." **It does NOT.** The best frontier-base candidate is Δscore = 0 (`mdl_contour`
declines). The zero-byte `contour_normal` is Δscore **+0.0344** (worse). **No paired CPU+CUDA exact eval
is pre-registered** — there is no advisory row beating the frontier, so the eval gate is not met (correct
fail-closed: do not spend ~$0.6 to confirm a non-improvement). The advisory rows are candidate-generator
signal only; the lane stays at `[local CPU-torch advisory]`.

---

## 6. ANTI-FAKE self-checks (all pass — NO-FAKE class 6 + class 8)

- **NOT a search (class 6):** the correction amplitudes are SOLVED in closed form `α = b/G` from the
  measured SegNet Jacobian + measured margin gap — there is NO grid/sweep over correction params (that is
  the killed lever-G pattern). `uses_stored_per_pixel_table = false` (no per-pixel oracle table; the
  correction is computed from the argmax structure, not stored).
- **rank/verdict only from the exact scorer (class 8):** d_seg on the exact `modules.SegNet` argmax
  (popcount), d_pose on the exact `DistortionNet` PoseNet, GT via `yuv420_to_rgb` ONLY, NEVER MPS. Score
  recomputed from components (the rounded final_score lies).
- **Tests verify BEHAVIOR not constants:** the Gα≥b solve produces amplitudes that ACTUALLY flip the
  predicted class on a controlled linear oracle (a zero/identity solve FAILS the flip test); the
  support-localized atom is zero outside the dilated flip support (a wide blob FAILS); the graph-cut
  SELECTS the predicted support (a select-all stub FAILS the water-level discrimination); the MDL
  admission charges real byte costs (a zero-cost stub FAILS the value>cost test); the end-to-end
  seg-only solve reduces (or holds) d_seg AND accounts new_bad_flips honestly (a wrong-sign oracle →
  infeasible → field zero → d_seg unchanged, proving the real feasible solve is what moves it).
- **honest collateral:** every row reports `new_bad_flips_created` AND `pose_side_effect` — without them
  the `repaired=421` row would lie. The net = repaired − new_bad − pose_side − rate is the verdict.

---

## 7. Wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE: the per-component `(repair_value_bytes, coded_bytes, net_collateral)`
   marginals are the seg-correction sensitivity input to the waterfiller (task #54). On the frontier base
   they are all under-water (empty fund set).
2. **Pareto** — ACTIVE: the rows establish that `frontier_archive` is at a Pareto vertex on the
   seg-correction axis (no frame1 correction moves d_seg down without paying more collateral than it
   saves).
3. **bit-allocator** — ACTIVE: the MDL candidate IS the bit-allocator gate (admit iff value > coded cost
   at λ\* = 1.27 B/flip); it correctly allocates ZERO bytes on the frontier base.
4. **cathedral-autopilot** — the smoke → (conditional) paired-eval dispatch surface; gate NOT met on the
   frontier base (no advisory improvement).
5. **continual-learning** — this verdict reseeds the planner: the frontier-base seg residual is
   irreducible by frame1-space correction (95% single-pixel flips, GT-snap net −536); the repairable
   residual lives on the lever-B generator base.
6. **probe-disambiguator** — RESOLVED: "is the frontier seg residual repairable by a frame1 correction?"
   → NO (the solve flips targets but pays net-negative collateral; the residual is scattered single-pixel
   noise the SegNet receptive field couples). The next probe: the lever-B base residual structure.

---

## 8. Cross-references

`closed_spec_boundary_math_system_of_equations_20260610.md` (§4 the polytope, §10 the water level) ·
`boundary_math_seg_core_20260610T101618Z.md` (task #52, the partition-store-loses-on-rate finding) ·
`lever_b_score_native_argmax_smoke_verdict_20260610.md` (the campaign base, d_seg=0.00826) ·
`lever_g_engineered_correction_smoke_20260610T095654Z.md` (the killed rule-family search; the
bidirectional-symmetry diagnostic this build's collateral measurement confirms) ·
`src/tac/boundary_math/boundary_solver.py` + `src/tac/boundary_math/tests/test_boundary_solver.py` +
`tools/closed_spec_boundary_solver_smoke.py` (the deliverables) ·
`src/tac/optimization/frame1_seg_repair_atoms.py` (the reused argmax+margin+THE-LAW-screening) ·
`upstream/{modules.py,frame_utils.py}` (frozen authority, read in full).
