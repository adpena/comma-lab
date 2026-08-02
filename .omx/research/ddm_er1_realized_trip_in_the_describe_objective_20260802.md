# ddm_er1 — the trip was already inside the describe objective. The **objective** was the surrogate.

**Arm:** ddm_er1 (#888 × #539) · 2026-08-02 · **no n600 scorer job fired** (MAIN owns the slot; live
`ddm_v4c_resolve.py --mode solve` pid 18732).
**Axis:** every number below is `[macOS-CPU frozen-head advisory]` or `DERIVED`. `score_claim=false`,
`promotion_eligible=false`, **exact pointer UNMOVED**. Nothing here lowered S.
**Denominator, used throughout:** scored pixels n600 = 600 × 512 × 384 = **117,964,800**.

**STORES CONSULTED:** `ddm_sv2_survival_engineering_and_the_rebase_20260802.md` ·
`ddm_is1_directive4_159x_pipeline_confound_20260724.md` ·
`ddm_uv1_ep854_pose_illegibility_reject_20260802.md` · the registered laws
`argmax_of_sdf_is_additively_weighted_power_diagram_v1`
(`witness_measured_findings_20260701.py:680`), `segnet_head_affine_gauge_quotient_v1`,
`pdw2_coefficient_only_spatial_nonidentifiability_v1` (`canonical_equations_registry.jsonl:743,745`;
`operator_p0_ledger.jsonl:257`) · `src/tac/boundary_math/power_diagram_witness.py` ·
`src/tac/optimization/direct_description_joint_descent.py` ·
`src/tac/differentiable_eval_roundtrip.py`.

---

## 0. HEADLINE — the charter's premise is half wrong, and the correction makes the build smaller

The charter states: *"The DESCRIBE/SOLVE path has NO equivalent [of `eval_roundtrip`]; it optimizes the
LABEL, while the trip decides SURVIVAL."*

**The first clause is FALSE for the main describe path, and I can point at the lines.**
`direct_description_joint_descent.py` already realizes the full trip inside its own forward pass:

| stage | file:line |
|---|---|
| paint → clip → **STE round to uint8** | `:2347` `clipped + stop_gradient(round(clipped) - clipped)` |
| **full R** bicubic-up 874×1164 → down 384×512, `ste_round=True` | `:2345–2350` `fused_r_roundtrip(...)` |
| real MLX **SegNet** on the round-tripped pair | `:2352` |
| real MLX **PoseNet** on YUV6 from *the same* round-tripped pair | `:2360–2363` |

**The second clause is TRUE, and it is one line.** `:2359`:

```python
seg = ce_seg_loss_mlx(seg_logits_nchw, targets)      # + optional margin_floor_hinge_mlx at :2361
```

and then, four lines later, in the *same function*, on the *same logits*:

```python
d_seg = mx.mean(mx.not_equal(mx.argmax(seg_logits, axis=-1), targets).astype(mx.float32))   # :2419
```

`_loss` (`:2349`) unpacks `seg, pose_mse, _` — **the realized argmax quantity is computed and
discarded.** The describe loop already pays for the realized trip in full and then optimizes a
cross-entropy surrogate of its output.

So the deliverable is **not** "wire the trip in." It is **"stop discarding the realized quantity"** —
replace the CE surrogate with the exact power-diagram margin on a forward pass that already realizes
the trip. That is a materially smaller and better-targeted build than the charter assumed, and per
`built_new_machinery_instead_of_paying_identified_debt` it is the debt to pay on the **existing**
surface.

---

## 1. THE SPLIT — #539 owns the primitive, #888 owns the objective

sv2 §5 warned explicitly against forking a parallel surface. Honored: everything below landed **inside
`src/tac/boundary_math/power_diagram_witness.py`**, the #539 module, which is the single owner.

### 1.1 What was actually missing (MEASURED by source inspection; denominator = 1 file, 1,435 lines, 37 top-level `def`s)

The module had the power-diagram **algebra** — `power_scores` `:549`, `power_distances` `:538`,
`power_assign` `:560`, `pair_tie_value` `:614`, `is_co_maximum_tie` `:626`, the PDW1/PDW2 codecs — and
**no margin/gradient interface at all**: `0` hits for `def *margin*`, `def *grad*`, `jacobian`,
`signed_dist`. *Positive control:* `grep -c margin` on the same file returns **17**, so the scan was
live. The nearest surface, `pair_tie_value`, is unusable as an objective for three reasons: it
**raises** unless the requested pair is already in `target.adjacency` (the deciding pair at a site is
data-dependent), it returns the **unnormalized** tie value, and it exposes the exact gradient only by
reaching into the `tie_normals` dataclass field.

### 1.2 The typed interface #539 now exposes for #888 to consume

```python
realized_margin_and_gradient(points, target, *, junction_tolerance=0.0) -> RealizedMargin
#   .margin            (...)        signed DISTANCE to the deciding hyperplane, quotient units
#   .gradient          (..., rank)  EXACT d(margin)/dz — the unit normal. No STE, no bias.
#   .top_class         (...)        agrees with power_assign site-for-site (first-max tie rule)
#   .runner_up_class   (...)
#   .junction          (...)        True on the codim-2/3 stratum the registered law leaves UN-COVERED
#   .junction_tolerance
```

Four design choices, each load-bearing:

1. **Top-2 selection**, not a declared pair — the competing pair is a property of the data.
2. **Normalized to a distance** (divide by `‖2(s_t−s_r)‖`) — so *one* scalar floor is comparable across
   different class pairs. This is what makes sv2's fail-closed admission gate expressible.
3. **`junction` is returned, never folded away.** sv2: *"a reformulation that silently drifts is worse
   than an STE known to be biased."*
4. **Fails closed on non-finite margin** — see §2.4, a real defect my own round-1 review found.

---

## 2. MEASURED — the reformulation's algebra legs, on the REAL frozen head, scorer-free

Source: `upstream/models/segnet.safetensors`, `segmentation_head.0.{weight,bias}` = `(5,16,3,3)` /
`(5,)`. Derived quotient: **rank 4**, singular values `[3.128, 2.154, 2.025, 1.796, 3.7e-16]` — the 5th
is numerically zero, confirming the common-row gauge is exactly removed. This needs **no video and no
scorer forward pass**, which is why it was runnable under the occupied-slot constraint.

### 2.1 Label agreement — leg (a) of sv2's falsifier #2

| n sampled quotient points | power-diagram argmax vs direct affine-head argmax |
|---:|---:|
| 200,000 | **1.0000000000 (exact)** |

*Positive control:* perturbing every site by +0.5 drops agreement to **0.805** — the test can fail.

**Honesty limit, stated because it caps the claim:** this verifies the *implementation is faithful to
the registered identity* (the gauge, centering, and basis could each have been wrong). It is **not**
the ≥99.5% agreement-against-the-frozen-SegNet leg sv2 pre-registered — that needs real `z` from a
scorer forward and remains **staged**.

### 2.2 Gradient exactness — leg (b) of falsifier #2

Analytic `d(margin)/dz` vs central finite differences, on sites where the deciding pair is unchanged
across the step (4,000/4,000 stable at ε=1e-5):

| | value | pre-registered bar |
|---|---:|---:|
| max relative error | **7.018e-11** | ≤ 1e-3 |
| mean relative error | **1.000e-11** | — |

**PASSES by ~8 orders of magnitude.** *Positive control:* an axis-reversed gradient scores **1.382**,
failing the same bar as required.

### 2.3 Why this beats the CE leg — and where that argument is weaker than I wanted

**(i) It is the geometric margin.** MEASURED: `margin == (l_t − l_r)/‖w_t − w_r‖` to max **6.4e-8**
absolute. *Control:* against the *unnormalized* `(l_t − l_r)` the error is 7.5e-1.

**(ii) Scale invariance — a provable defect of the CE leg.** `argmax` ignores a common positive rescale
of the head; the scored quantity `d_seg` therefore ignores it. MEASURED under `W→cW, b→cb`:

| c | argmax agreement | max \|Δ margin\| | mean CE |
|---:|---:|---:|---|
| 0.5 | 1.000000 | 1.9e-08 | 0.4276 → 0.7415 (**1.73×**) |
| 2.0 | 1.000000 | 3.3e-08 | 0.4276 → 0.2186 (**0.51×**) |
| 10.0 | 1.000000 | 2.9e-07 | 0.4276 → 0.0427 (**0.10×**) |

The margin is invariant to f32 noise; **CE moves 10×** under a transformation that provably cannot
change the score. This is now an executable assertion in the test suite, not a memo claim.

**(iii) The alignment argument is WEAKER than I expected, and I report it against my own thesis.**
Spearman(per-site CE, realized margin) = **−0.9452** (a perfect surrogate would be −1.0). **CE is a
good *ranker* of which sites are marginal.** Where it fails is *allocation*: the bottom-5%-margin sites
carry only **10.91%** of total CE mass (uniform = 5.00%, a mere 2.2× enrichment) — CE spends ~89% of its
gradient on sites whose survival is not in question. *Control:* Spearman(CE, CE) = 1.0000.

So the honest claim is **"CE is a good ranker but a poor allocator, and is scale-sensitive where the
score is not"** — not "CE is misaligned." **Caveat that caps all of §2.3:** these are isotropic-Gaussian
synthetic features, i.e. **chart geometry, not vehicle numbers.** Real SegNet features are strongly
non-Gaussian (pt1 measures 4.4% of pixels carrying 52% of errors at 24.09× enrichment). The real
alignment must be measured on real `z` and is **staged**.

### 2.4 Two defects my own review found — both real, both fixed

- **NaN reachable from finite input.** At `z = 1e308` the score overflows to ±inf and `inf − inf` = NaN;
  the primitive would have seeded a descent with a silent NaN. Now **fails closed**
  (`PowerDiagramWitnessError`), with a regression test asserting the *input* is finite.
- **A float32 noise floor that constrains every consumer.** My junction fixture used √3/2 (not f32
  representable) and a constructed exact 3-way tie came back **broken by 2.69e-8**; with f32-exact
  coordinates the tie is **exactly 0.0**. Independently corroborated: absolute margin error is a uniform
  **~1e-8 regardless of margin size** (max 8.93e-8), so the 6.9e-3 *relative* figure in the smallest
  decile is only a small denominator, not error growth.
  **DERIVED consumer constraint: any margin floor or `junction_tolerance` must sit well above ~1e-7
  quotient units.** Measured `|margin|` median 0.377, p01 0.0051 — a floor at 1e-3 is ~4 orders clear.
  Both are now tests.

---

## 3. THE STRUCTURAL LIMIT — DERIVED, and it re-scopes what #888 can claim

**The power-diagram chart is exact in FEATURE space, and the describe objective lives in IMAGE space.**

```
∂margin/∂D  =  ∂margin/∂z   ·   ∂z/∂x        ·   ∂x/∂D
               ^EXACT           ^frozen backbone   ^paint
               (this arm)       NOT reformulated   (existing)
```

The reformulation removes the argmax non-differentiability at the **last link only**. It does not make
the trip differentiable and it does not remove the uint8 round. **This is not my inference alone — it
is already MEASURED from the rate side** by the registered
`pdw2_coefficient_only_spatial_nonidentifiability_v1`: the 138 B gauge-fixed packet is genuinely
consumed and replays n600 bit-identically, yet *"is NOT a self-contained spatial generator — needs the
quotient feature field z(x) (uncounted)."* Same gap, two independent directions. It is also the crux
the pantheon line already names: **realization in the IMAGE chart**.

Consequence: `∂z/∂x` must still come from autograd through the frozen backbone, and the uint8 round
still needs an STE. **That is consistent with the operator's "reformulate, don't STE" directive, and
here is why:** `is1` **exonerated quantization** as the binding stage (the exact lattice solve passes
the same uint8 gate at 17,931 errors; the 30-byte amplitude arm moved 0.022448 → 0.008619 through the
same gate). sv2 §3.4 states the rule directly — *reformulate the stage that binds (argmax), skip the
one that does not (quantization).* The STE survives **only at the exonerated stage**. I state this
explicitly rather than shipping an STE quietly.

**The canonical trip helper already exists** and should not be rebuilt:
`apply_eval_roundtrip_during_training` (`src/tac/differentiable_eval_roundtrip.py:213`) — bicubic-up →
bilinear-down → `Uint8STE`, autograd preserved. The MLX describe path uses its own
`fused_r_roundtrip` equivalent at `:2345`.

### 3.1 A correction to sv2's own magnitude

sv2 §5 wrote *"~127 file-hits of argmax reformulation exist and none is wired here."* MEASURED over
`src/ tools/ experiments/ scripts/`, glob `'*.py'`, **denominator 10,266 files**: the union of
`power_diagram_witness|affine_head_to_power_diagram|power_scores|power_assign|laguerre` is 205 lines
over 131 files — but **99 of those files hit only the module *name*, and 81 of them are importing
`open_stored_npy_memmap`**, an unrelated memmap-I/O helper that merely lives in that module. The genuine
reformulation surface is far smaller than 127. **The "unwired" verdict stands and is in fact stronger;
the magnitude was overstated.**

Scoped negative with a live positive control: across the 19 describe-path files
(`src/tac/optimization/direct_description_*.py` ×14, `ddm_ws1_warm_start.py`,
`ddm_continuous_paint_ceiling.py`, `ddm_rg1_receiver_grammar.py`, `through_r/palette_realization.py`,
`through_r/resolution_chain.py`) there are **0** occurrences of `affine_head_to_power_diagram`,
`power_scores`, `power_assign`, or `laguerre`. Positive control on the identical file set: `rg -c 'def '`
returns 98 and 16.

*Instrument note, measured:* `rg` descends `.omx/` when it is passed as an **explicit path argument**
(262 files with and without `--hidden`); `--hidden` matters only for a root-level walk
(`power_diagram_witness`: 0 → 41). sv2 §1's warning is right for root walks and wrong for explicit
paths.

---

## 4. THE POSE FALSIFIER — CONFIRMED AS A LAW, and it binds this build

The charter required this run and pre-registered the outcome. **The predicted failure is already
MEASURED, by `uv1` (#889), with a passing positive control — it does not need re-running.**

The power-diagram chart is **intrinsically argmax-only**: cell membership is invariant to *any*
deformation preserving the affine inequalities. §2.3(ii) is this invariance measured. That invariance is
the very property that makes it the right seg chart — **and it is exactly the pose-blindness uv1 named.**

uv1, same solver, same pairs, same starts, **only the base differs**:

| base | mean d_pose after full re-solve |
|---|---:|
| gr1 `cell_drop50` (pose solved against it) | **0.000709** — control PASSES |
| ep854 (seg-optimized) | **2.138939** — 162× over break-even |

**3,019× separation on identical machinery.** Mechanism: SegNet reads only the **argmax** (invariant to
palette deformation); PoseNet reads dense **photometric** correspondence (destroyed) —
`corr(f1_gr1, f1_ep854) = +0.119` with 99.7% of pixels changed *while d_seg improved*.

**Verdict: CONFIRMED law, reported as such and not as a setback.** A describe objective built naively on
this primitive **will** reproduce uv1's failure *while reporting an improving d_seg*. Binding
consequences for #888:

1. **Never adopt this as THE vehicle chart.** It is the exact chart for the *seg leg only*. Installing
   it as the parametrization is the generic-basis error in a sophisticated disguise.
2. The existing objective is already **joint** (`_loss:2349` = `100·seg + w_pose·√(10·pose_mse)`, real
   PoseNet at `:2360–2363` on the same round-tripped pair). **Swap the seg leg only; leave the pose leg
   carrying real photometric gradient.** Setting `pose_objective_weight = 0` for a "clean seg A/B" is the
   uv1 trap and is forbidden.
3. Every descent reports d_pose on the same slice beside d_seg, against the box value
   `joint_finish_d_pose_max = 0.001610`.

---

## 5. WHAT THIS DID NOT DO — stated plainly

- **The pointer did not move.** `pointer_moved: false`. Per the means/ends firewall this is not goal
  progress: it is a primitive plus a re-scoping, not a lower S.
- **I did not measure the gap the trip closes against the 159× anchor.** The charter asked for it at
  n600; the slot is occupied and the arm is scorer-free. What I *can* say is that the anchor itself is
  wrong for the live base: re-derived independently at source, `0.024125/1.52e-4 = 158.7×` is correct
  **only for W_seg**; against the best measured described base (pt1 `global_amplitude_statistics_match`,
  d_seg 0.008619, 30 B, `geometry_changed: false`) the loss is **56.7×**, and the workload above box is
  **879,900** not 2,709,062 — a **3.079×** overstatement. My arithmetic reproduces sv2's to 0.004%.
- **I did not wire the primitive into `direct_description_joint_descent`.** §0 identifies the exact line
  (`:2359`) and §4 constrains the swap, but the edit is not made: it changes a live descent objective
  while a solve is running in this tree, and its verdict is unmeasurable without the scorer slot.
  **Named, priced, and NOT silently left implied-complete.**
- **sv2's falsifier #1 (AS-arm boundary enrichment ≥8×) is still unfired**, and it is the cheapest
  decisive test available. It gates whether margin-aware description can bite at all.
- **The Movable stratum leg is untested on real data.** Junction fractions measured here (0.025% @1e-3,
  0.875% @1e-2, 7.95% @1e-1) are **isotropic-Gaussian chart geometry**, *not* the codim-2/3 Movable
  junction's share of real residual error mass. Movable is 27.0% of flip mass; sv2 pre-registered
  FALSIFIED-IF that stratum carries >10% of residual. **Genuinely open.**
- **One pre-existing test failure observed, not mine, not touched:**
  `test_chroma_boundary_match.py::test_equation_cites_the_dof_source_and_makes_no_score_claim` asserts
  the literal `"0.19110"` appears in an equation blob; it no longer does. The file is unmodified in my
  working tree. It belongs to another arm's surface — flagged, deliberately not fixed.

---

## 6. ROUND-1 ADVERSARIAL SELF-REVIEW — my own defects

1. **I nearly accepted the charter's premise verbatim.** "The describe path has NO `eval_roundtrip`
   equivalent" is false for the main path — it has the STE round *and* full R with `ste_round=True`. Had
   I built to the charter I would have added a trip that was already there and missed the real
   one-line defect. Reading the code beat reading the brief.
2. **My first exactness probe reported max rel err 6.9e-3 and I almost shipped it as "not exact."** It
   is a small-denominator artifact of a uniform ~1e-8 absolute f32 error. Decomposing by margin decile —
   rather than trusting the headline — produced the actual consumer constraint (§2.4).
3. **My alignment hypothesis was substantially wrong.** I expected CE to be badly misaligned with the
   margin; ρ = −0.945 says it ranks well. I report the weaker true claim (poor *allocator*, scale-sensitive)
   rather than the headline I wanted.
4. **My junction test was wrong and the suite caught it** — a constructed tie broken by f32. That
   failure is the empirical proof this suite can fail, and it turned into the noise-floor finding.
5. **I shipped a NaN path in the first draft.** Found in my own round-1 pass, not by a reviewer.
6. **I corrected sv2's "127 file-hits" magnitude** — an inherited number I could have passed through
   unchecked. Most of those hits are an unrelated I/O helper.
7. **§2 is entirely synthetic-feature.** Every number in it is chart geometry. I say so three times
   because it is the single easiest thing to over-read here.
8. **I did not re-price `menu1` / #366** on the re-based denominator. sv2 owed it, I inherited it, and I
   am also not closing it. Unfinished, not closed.

---

## 7. WHAT THIS OWES NEXT (ordered by distance to an exact row)

1. **Fire sv2 falsifier #1** (AS-arm residual boundary enrichment; ≥8× predicted, <4× falsifies). Reuses
   the existing pt1 instrument on an existing arm. Cheapest decisive test in the queue.
2. **Swap `:2359`'s seg leg** to `realized_margin_and_gradient` under §4's constraints (pose leg live,
   d_pose reported), then the matched A/B against the CE leg from the same start.
3. **Measure §2.1/§2.3 on real `z`** — label agreement vs the frozen SegNet, CE-vs-margin allocation, and
   the Movable codim-2/3 stratum's real share of residual error mass.
4. **Emit per-element survival as typed JSONL** from the existing pt1 path (sv2 §3.4b) — an emission
   path, not a new instrument.
5. Correct the `159×` label at `pp1:132` and the 2.7 M workload at `pantheon:202` / `ar1:38`
   (append-only supersession; do not mutate the historical FEED-603 ledger row).

**CLOSING-ARTIFACT: .omx/research/ddm_er1_realized_trip_in_the_describe_objective_20260802.md**
