# ra3 — the untried cell MEASURED: subspace + trust-regioned per-pair re-fit, accepted on realised measurement

`date_utc: 2026-08-16` · `owner: ddm_ra3` · `axis: [macOS-CPU advisory, upstream chain, n600]`
`score_claim: false` · `promotable: false` · `frontier_moved: false`
receipts: `/Volumes/APDataStore/pact/ddm_ra3/RA3_TRUST_REGION_REFIT_r11.json`,
`/Volumes/APDataStore/pact/ddm_ra3/RA3_RATE_CHECK_r11.json`,
`/Volumes/APDataStore/pact/ddm_jc1/retained/RA3_REPRO_INCUMBENT_r11.json`,
prediction recorded before the rows landed:
`/Volumes/APDataStore/pact/ddm_ra3/RA3_PREDICTION_BEFORE_REALISED.md`

## THE ANSWER, FIRST

**The untried cell WORKS, by the largest margin any carrier arm has produced — and it still
REFUSES by 35.5×. The family closes.**

1. **The re-fit is a real, large effect.** Exact n600, per-pair realised acceptance, scored on the
   authority-tracking GT: the incumbent projection costs `ΔS_pose = +0.136386`; the trust-regioned
   per-pair re-fit costs `+0.043639` — **3.13× better on the score-relevant quantity** (7.76× in
   `d_pose` ratio, 304.0× → 39.2× of base). 553 of 600 pairs improved.
2. **It is still refused by 35.5×.** Net `ΔS = +0.042410` against a rate credit of `+0.001230`.
   Break-even needs another **35.5×**; the whole method just bought 3.13×.
3. **Even PERFECT execution cannot reach the target.** At zero pose damage the rung returns only
   its bytes: **913–1,847 B = 0.00061–0.00123 S = 6.3–12.8% of the 0.0095973 gap.** The ceiling
   is below the goal, so no amount of further refinement on this rung matters.
4. **The 11.5× bar in my charter was an artifact of two compounding level errors.** On the only
   priceable axis the incumbent's pose cost is 110.9× its rate credit, not 11.5×.
5. **Two instrument findings that outlive this arm**, both MEASURED (§7): selecting the trust-region
   radius on the wrong GT costs **1.374×** and moves **346 of 600** pairs; and the additive-floor
   conversion is accurate to 0.45% at catastrophic damage but wrong by **2.64×** near break-even —
   i.e. accurate exactly where the answer does not matter.

**Pointer UNMOVED: hv1 ep0634, S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600].**
This unit produced no lower score.

## 1. Reproduction — re-derived, not inherited

MAIN's charter said re-derive ra2's grid before building on it. Two independent checks:

- **The DERIVED ladder reproduces bit-for-bit.** Re-running `ddm_ra2_pose_metric_rank_ladder.py`
  regenerates all six candidate arrays; its `_retain_array` guard refuses on any byte difference
  and did not fire. Every number in ra2 §3 (errors, bars, the matched-rank race) reproduces
  exactly, including `cond(G_pose) = 144.437` and `cond(G_basis) = 15.711`.
- **The EXACT incumbent reproduces to the last digit.** Re-measuring
  `candidate_pose_subspace_r11.float64.npy` through `ddm_jc1_carrier_pose_jacobian.py --eval-coeff`:
  `d_pose = 0.0022432311489210995`, **identical** to ra2's record, same candidate sha
  `7731cc0f843bc749…`.

**Re-derived by me:** ra2's whole subspace row (both metrics, r=4/8/11) and the exact r=11
incumbent. **Inherited, not re-run:** jc1's keep-set row (235.3× / 238.9×) and jc1's Jacobian
payload, which I consume as an input.

## 2. The cell, and why the trust region is anchored on the incumbent

The two surviving cells optimise the wrong functional in opposite directions. `jc1` minimised the
realised residual `‖r_i + J_i δ_i‖²` — the right objective — but let an unbounded linear model
*design* the step, and measured its own model wrong by up to **1,065×**. `ra2` minimised the
perturbation `‖J_i δ_i‖²` and ignored the base residual `r_i` entirely.

The untried cell is the right objective under an explicit trust region. Whitening by
`G_p = mean_i J_iᵀJ_i` and writing `u = z_i − z_i^proj` for the correction *on top of* ra2's
projection, the problem separates exactly:

    minimise_u  ‖b_i + A_i u‖² + λ_i ‖u‖²        A_i = J_i G_p^{-1/2} V_r,  b_i = r_i − J_i G_p^{-1/2} y_i^⊥
    ⇒ (A_iᵀA_i + λ_i I) u = −A_iᵀ b_i            ‖δ_i‖²_{G_p} = ‖u‖² + ‖y_i^⊥‖²

`λ_i → ∞` returns `u = 0`, i.e. **exactly ra2's projection**; `λ_i → 0` returns jc1's failure
mode. `λ_i = μ · mean(eig(A_iᵀA_i))` makes the grid dimensionless across pairs whose Jacobians
differ in scale.

**Two consequences I state up front because they bound what this arm can claim.** First, the
anchoring means realised acceptance *cannot* return a row worse than ra2's — so "beats 15.211×"
is nearly guaranteed by construction and is **not** evidence about the family (§5). Second, the
separation identity and the closed form were verified against an independent numerical minimiser
of the stated objective (BFGS, 5 random instances, agreement 1e-5; the orthogonality identity
`‖δ‖²_{G_p} = ‖u‖² + ‖y^⊥‖²` holds to 1e-15).

**MODEL-PROPOSED, REALISED-ACCEPTED.** Every `u_i(μ)` is a proposal. For each pair and each μ the
frame is rendered through the shipped chain with hard quantization and scored by the frozen CPU
PoseNet; the μ with the smallest **measured** per-pair squared residual wins, with the projection
always in the candidate set. Shrink-on-ascent is therefore exhaustive rather than greedy: a pair
whose every proposal ascends simply keeps the incumbent step. No number in the verdict comes from
the linear model.

**Instrument: ra2's, unchanged.** `ddm_jc1_carrier_pose_jacobian.py` is imported wholesale — same
render, same sparse frame-0 selector, same upstream preprocess, same frozen weights, same retained
ground truth, same 600 pairs — and the incumbent is re-measured inside this run so every ratio is
within-axis. Centre control: my projection agrees with ra2's retained array to **6.3e-15**, far
below the uint8 render floor.

### The model's proposals (DERIVED — authority is §4, not this table)

| μ | step / coeff RMS | step / incumbent step | 1st-order predicted `d_pose` |
|---:|---:|---:|---:|
| 100 | 0.1351 | 1.0004 | 3.216902e-03 |
| 30 | 0.1354 | 1.0026 | 2.126485e-03 |
| 10 | 0.1361 | 1.0098 | 9.073192e-04 |
| 3 | 0.1373 | 1.0220 | 1.923907e-04 |
| 1 | 0.1380 | 1.0298 | 3.815861e-05 |
| 0.3 | 0.1384 | 1.0339 | 1.328731e-05 |
| 0.1 | 0.1384 | 1.0365 | 1.041642e-05 |
| 0.01 | 0.1385 | 1.0720 | 8.117716e-06 |
| 0.001 | 0.1491 | 1.3141 | 5.592785e-06 |
| 0.0001 | 0.2207 | 2.3246 | 3.416337e-06 |

The model predicts clearing the advisory bar (`1.57e-4`) from μ=3 downward, and beating base by
43× at μ=1e-4. The trust region is genuinely tight: at μ ≥ 0.01 the step is within **7.2%** of the
incumbent's own step, against jc1's keep-set steps of 3.4–43.9× coefficient RMS.

## 3. The byte credit — MEASURED for the first time, and it is 1.11–2.0× optimistic

Every verdict in this family divides its measured pose damage by a byte credit that was never
measured. jc1 named the debt explicitly (`BYTES_PER_DROPPED_DIM = 1853.5` is **ASSUMED-BY-TRANSFER**);
three different values circulate — **1,514** (ra2c), **1,847** (ra2), **1,854** (jc1) — and they
were never reconciled. It is also the DENOMINATOR of every miss, and exactly the quantity a re-fit
can silently spoil, because the shipped coefficient stream is Rice-coded over zigzagged
frame-deltas and higher-entropy values hand back fewer bytes.

`ddm_ra3_refit_rate_check.py` encodes each candidate through **`carrier_codec`, the shipped
encoder**, and measures real blobs. Controls: shipped re-encode is byte-identical to the canonical
CPR1 blob; the zigzag-delta round trip is exact; the quantizer reproduces the shipped codes.

**The third control earned its keep.** Against the Rice-decoded codes it FAILED on exactly 30 of
7,200 coordinates: the receiver applies a **36-byte compensation overlay** to the decoded codes
before scaling (`ra2b` chain[4]), so the shipped coefficients are the POST-overlay codes. Pricing
a re-fit against pre-overlay codes would have mis-stated the grid on 30 coordinates silently. The
overlay is separately MEASURED rate-neutral at the Rice layer (post-overlay codes re-encode to the
same 79,020 bits) and is excluded from both sides of the credit.

MEASURED credit at r=11, from real CPR1 blobs:

| convention | credit | advisory bar |
|---|---:|---:|
| **measured container** (rotated 11-atom basis, max-abs 5-bit) | **918 B** (951 B raw) | 1.0321× |
| **most favourable** (basis pro-rated, coefficient half MEASURED) | **1,658 B** | 1.0583× |
| ra2 assumed (uniform bytes per dimension) | 1,847 B | 1.0651× |

**Mechanism.** A rank-r carrier stores **rotated** atoms, and rotation mixes amplitude
distributions, so the 11-atom basis costs **11,970 B** rather than the pro-rated 11,254 B — 6.4%
*more per atom* than the shipped basis. The coefficient half returns 635 B, not the pro-rated 823 B.

**Recorded, non-refusing:** the shipped basis quantizer is *not* the max-abs-to-15 grid this tool
applies — three of twelve atoms cap at `|code| = 7`, and those three are exactly the lowest-RMS
atoms, so the producer allocated per-atom precision by rate-distortion. That rule is not
reconstructible from the archive, so the rotated-basis size is an **upper bound** and the credit is
reported as a bracket rather than a guess. **The verdict below is computed at ra2's 1,847 B** so
every ratio stays directly comparable to ra2's and jc1's; the tightening is reported separately.

## 4. The measured rows — exact n600, both GTs, 8,400 scorer forwards

The instrument identity is settled empirically, not argued: the in-run projection reproduces ra2's
number with `incumbent_reproduction_relative_error = 0.0` — **bit-for-bit**, so my evaluator *is*
ra2's. And re-scoring the retained `pose6` against the PyAV GT reproduces the parent receipt to
**4.70e-16**, so when the GT is swapped, the GT is the only thing that changed.

| candidate | `d_pose` PyAV (ra2's axis) | ratio | `d_pose` AUTHORITY GT | ratio | `ΔS_pose` |
|---|---:|---:|---:|---:|---:|
| **projection** (ra2's incumbent) | 0.00224323 | 15.212× | 2.09335e-03 | 304.0× | +0.136386 |
| μ=100 | 0.00181147 | 12.284× | 1.69349e-03 | 245.9× | +0.121836 |
| μ=30 | 0.00124893 | 8.469× | 1.18691e-03 | 172.4× | +0.100647 |
| **μ=10 (best single radius)** | 0.00079914 | 5.419× | 8.27670e-04 | 120.2× | +0.082678 |
| μ=3 | 0.00084505 | 5.730× | 9.62200e-04 | 139.7× | +0.089794 |
| μ=1 | 0.00110954 | 7.524× | 1.27189e-03 | 184.7× | +0.104480 |
| μ=0.3 … 0.01 | 0.00128–0.00138 | 8.68–9.38× | 1.464–1.581e-03 | 212.6–229.7× | +0.1127–0.1175 |
| μ=0.001 | 0.00385757 | 26.159× | 4.03434e-03 | 585.9× | +0.192559 |
| μ=0.0001 | 0.05905642 | 400.470× | 5.92623e-02 | 8606.7× | +0.761522 |
| **ACCEPTED** (per-pair, authority-selected) | — | — | **2.69749e-04** | **39.18×** | **+0.043639** |

Base: PyAV `1.4747e-04`; authority GT **`6.88559506e-06`** — **1.00081×** the contest-CUDA
authority `6.88e-06`, exactly as `pi2` measured. The fix lands.

**Three readings.**

- **The μ curve has an interior optimum and the extremes both fail.** μ=10 is the best single
  radius; loosening to μ=1e-4 explodes to 8,607× — **jc1's designer failure reproduced exactly**,
  now with the trust region as the dial that produces it continuously rather than as an accident.
- **No global radius exists.** The accepted-slot histogram is spread across the entire four-decade
  grid (46 at μ=1e-4 … 66 at μ=10 … 47 at the projection); nothing concentrates. Per-pair selection
  buys **3.07×** over the best single radius (8.2767e-04 → 2.6975e-04). This is the mechanism
  behind the whole result: the optimal trust-region radius is a per-pair property.
- **The projection is never the answer for 553 of 600 pairs**, yet the 47 pairs that keep it are
  real — the exhaustive shrink-on-ascent is doing work, not decorating.

**The 12-bit storage tax is small, measured like-for-like on the authority axis.** Quantizing the
stored `z` to the shipped code width moves the projection from `2.09335e-03` to `2.08725e-03`
(**−0.3%**, inside noise) and the accepted row from `3.70508e-04` to `3.78510e-04` (**+2.2%**;
both are the PyAV-selected variant, so the comparison isolates quantization from selection).
Storage precision is not what refuses this rung — it is worth ~2%, and the rung needs 35.5×.

## 5. The falsifier — and a declared pre-registration deviation

ra2's pre-registered falsifier, which I adopted: *"if the trust-regioned per-pair re-fit fails to
beat 15.211×, the carrier rank/refit family closes at FORMULATION scope."*

**It did not fire. The re-fit beat 15.211× by 9.08×** on the axis that number was measured on.

**I nonetheless close the family, and I flag that as a deviation rather than leave it implicit.**
I recorded *before the rows landed* (`RA3_PREDICTION_BEFORE_REALISED.md`) that this threshold is
structurally near-vacuous: the trust region is *anchored on the incumbent*, so `μ → ∞` returns
ra2's projection exactly and realised acceptance can never return a worse row. "Beats the
incumbent" was guaranteed-or-tie by construction and is not evidence about the family. The
informative threshold is break-even, and it is missed by **35.5×**.

The closure rests on three independent grounds, none of which is the falsifier:

1. **Measured.** 35.5× from break-even on the only priceable axis, after the one reformulation
   the 2×2 implied.
2. **Structural ceiling.** Perfect execution returns 913–1,847 B = **6.3–12.8% of the gap** (§3).
   A rung whose *best case* is a tenth of the target does not deserve further slots.
3. **The denominator is optimistic.** The byte credit every arm divided by is 1.11–2.0× too large
   when measured (§3), so grounds 1 and 2 are both stated in the family's own favour.

My own round-2 decision rule (pre-registered) also did not discriminate: it keyed on the histogram
concentrating aggressive or conservative, and the histogram is *spread*. I report that as a rule
that failed to fire rather than reinterpreting it — the spread is the finding, not the trigger.

## 6. d_seg needs no measurement, and that is now a PROOF

`upstream/modules.py:108` is `x = x[:, -1, ...]  # Use only last frame`: SegNet reads **only
frame_1**. The carrier renders **only frame_0**. So `d_seg` is invariant to *any* carrier edit
**by construction**. ra2 recorded this as "MEASURED identically on 4 treatments"; it is stronger
than a tally, and no future carrier arm need re-measure `d_seg` at all. Every carrier question is
a pure (pose, rate) trade — structurally, not empirically.

## 7. THE BAR IS NOT 11.5× — the family needs 956×, and the ratio framing was two level errors

Sister arm `pi2` MEASURED today that the advisory pose instrument is neither broken nor
multiplicatively biased: it carries a **fixed additive floor** of `d_pose = 1.4061e-04`, essentially
all of it GT-decode-path drift (PyAV/`yuv420_to_rgb` GT vs DALI/nvdec GT, same frozen scorer).
Scorer-forward CPU/CUDA drift is `3.57e-12` and is falsified as the cause by nine orders of
magnitude. I re-derived the closure at source: `6.88e-06 + 1.4061e-04 = 1.474900e-04` against my
instrument's measured base `1.4747e-04` — **0.014%**.

**The floor is additive in `d_pose`, so it cancels in the ABSOLUTE DELTA and in nothing else.**
That supersedes the orientation-bracket derivation I had here (it happened to land on the correct
`305.6×` in its orthogonal row, but by a weaker route). The binding rule:

    Δd_pose_abs = d_pose_advisory(candidate) − d_pose_advisory(base)          # floor cancels
    ΔS_pose     = √(10·(6.88e-06 + Δd_pose_abs)) − √(10·6.88e-06)

**Never rescale an advisory ΔS_pose; never quote an advisory `d_pose` ratio as a score.** ra2's four
cells remain valid for *comparison among themselves* — the floor is common to all four, so the
treatment axis (15.7×) and the metric axis (1.88×) are safe. What is **not** safe is their absolute
level: a ratio of floored quantities compared against a ratio bar. My cell is measured on the same
exact instrument, so it belongs in that table; I report its ratio for comparability and its
absolute delta for pricing, and never mix the two.

Re-priced correctly, the rate credit at r=11 (ra2's 1,847 B convention) is **+0.0012297 S**, so the
break-even pose damage is **Δd_pose_abs ≤ 2.191e-06**:

| candidate (all exact n600, same instrument) | advisory `d_pose` | Δd_pose_abs | ΔS_pose | net ΔS | pose cost ÷ rate credit |
|---|---:|---:|---:|---:|---:|
| ra2 pose subspace r=11 (**the incumbent**) | 0.00224323 | 2.0958e-03 | +0.136710 | **+0.135481** | **111.2×** |
| ra2 euclid subspace r=11 | 0.00422005 | 4.0726e-03 | +0.193682 | +0.192453 | 157.5× |
| ra2 pose subspace r=8 | 0.02579991 | 2.5652e-02 | +0.498256 | +0.497026 | 405.2× |
| jc1 euclid keep-set r=11 | 0.03469834 | 3.4551e-02 | +0.579564 | +0.578334 | 471.3× |

**So my charter's premise is falsified.** It said *"the bar is 11.5× and the best measured cell is
15.211×, so the family is close, not dead."* The 11.5× was a ratio-of-floored-quantities compared
against a ratio bar — two level errors compounding in the optimistic direction. On the score, the
incumbent's pose cost is **111× the rate credit it buys**, and to break even the re-fit must cut
absolute pose damage from `2.0958e-03` to `2.191e-06` — a **956× reduction**, not 1.3×.

The fallback rule is apparatus, not prose: `experiments/ddm_ra3_advisory_pose_pricing.py`
implements it, re-derives the floor closure at import time, and **refuses** a below-floor input or
a ratio-as-score. Encoding it is the only thing that stops the level error recurring — it survived
three arms as an unexamined "11.5× bar".

### 7b. But the conversion is the FALLBACK; the FIX is a fixed reference — and I MEASURED why that matters

`pi2`'s final (`ed153d0203`) found the root cause inside our own tooling: the advisory instrument
reads **two ground truths** — seg loads a cached DALI/nvdec-lineage argmax, pose has no cache and
re-decodes with PyAV every run. One instrument, two lineages; that is the whole ~21× offset. The
fix is not a factor (the implied factor is not even constant — 7.55×–11.45× across rn1's rows) but
a fixed reference: `gt_cache_dali.pt["pose"]` (sha `a91d9825…`), which tracks the contest authority
at **1.00081×**. My base re-scores to `6.88559506e-06` against `6.88e-06` — the fix lands to 8e-4.

**This cost me zero recomputation, and only because the payload rule is P0.** The parent tool
persisted all 600×12 generated `pose6` vectors, and `d_pose` is a pure function of (generated, GT),
so swapping the ground truth is arithmetic. Had this arm persisted only the scalar `d_pose` values
— the measure-and-discard defect — the instrument fix would have cost a 40-minute re-run of 8,400
scorer forwards.

Two things the fix exposes that the conversion hides, both MEASURED here:

- **Selecting on the wrong GT costs 1.374×, and moves 346 of 600 pairs.** The per-pair trust-region
  radius chosen against the PyAV GT scores `3.70508e-04` on the authority GT; re-choosing against
  the authority GT gives `2.69749e-04`. More than half the pairs pick a different radius. A
  per-pair selection rule is *doubly* sensitive to the GT: once in the score, once in the choice.
- **The additive-floor conversion is damage-dependent, and wrong precisely where it matters.**
  Comparing `Δd_pose` on both axes per candidate:

  | candidate | Δ PyAV | Δ authority | conversion error |
  |---|---:|---:|---:|
  | projection (catastrophic damage) | 2.09576e-03 | 2.08646e-03 | **0.45%** |
  | μ=0.0001 (catastrophic damage) | 5.89090e-02 | 5.92554e-02 | **0.58%** |
  | μ=10 (moderate) | 6.51671e-04 | 8.20784e-04 | **20.6%** |
  | μ=3 (moderate) | 6.97586e-04 | 9.55315e-04 | **27.0%** |
  | ACCEPTED (near break-even) | 9.95015e-05 | 2.62864e-04 | **2.64×** |

  Mechanism: `Δ_auth = 2⟨e,p⟩ + |p|²` while `Δ_PyAV = 2⟨e+d,p⟩ + |p|²`; they differ by `2⟨d,p⟩`,
  which is negligible only when `|p| ≫ |d|`. **So the conversion is accurate exactly where the
  answer does not matter (a candidate that is obviously catastrophic) and off by 2.6× exactly where
  it does (a candidate near break-even).** That independently vindicates "use the fixed reference,
  not a scale factor" with numbers rather than with caution.

## 8. Where the headroom actually is — and why "redirect to seg" and "pose marginal is 6.03× seg's" are BOTH right

The gap is `S − 0.15 = 0.00959729`. Decomposed against the frontier's own terms:

| axis | term at hv1 | headroom if driven to ZERO | can it close the gap alone? |
|---|---:|---:|---|
| pose | 0.0082946 | 0.0082946 | **NO — structurally impossible.** Perfect pose still leaves **+0.0013027** owed |
| seg | 0.0296110 | 0.0296110 | yes, needs a **32.4%** cut in `d_seg` |
| rate | 0.1216917 | large | yes, needs **14,413 B** = 7.9% of the archive |

**The whole pose term is smaller than the gap.** Driving `d_pose` to exactly zero — a physically
unreachable ideal — still leaves 0.0013 on the table. So pose cannot reach the target alone, ever.
That is not a statement about difficulty; it is arithmetic.

This reconciles the two live claims rather than pitting them against each other. `rn1`'s "pose
marginal is 6.03× seg's" is a statement about the **derivative** (`5/√(10·d_pose) = 602.8` vs seg's
constant `100`) — pose is the cheapest place to buy the *next* increment. My charter's "redirect to
seg" is a statement about **total available headroom** — seg is the only single axis with enough of
it. Both are true at different levels, which is exactly the standing law that an instrument's level
and aggregation are part of the claim. **Operationally: pose is a high-yield refinement, seg and
rate are the only axes that can carry the target.**

And the carrier specifically is finished as a rate lever on arithmetic alone: it is 22,161 B =
0.0147561 S, so even **deleting it for free** would leave −0.0051588 — it *could* close the gap if
it were free, but `α=0` costs 350,428× in pose (ra2c). The five treatments now measured
(α=0, rank truncation, keep-set re-fit, subspace, subspace + trust-regioned re-fit) span the space.

## 9. Honest limits

- **Advisory throughout; no Modal.** $18.62 of the $20 cap was consumed before this arm started, so
  nothing here was dispatched. The authority-GT column tracks the contest axis at 1.00081× but is
  still a local measurement on a retained render — it is not a contest row and nothing is promotable.
- **The `d_pose` verdict is exact; the rate credit is not.** Every `d_pose` is a real render through
  the shipped chain plus the frozen CPU PoseNet. The byte credit is measured through the shipped
  coder but the *container* it prices (an 11-atom CPR1) does not exist, and the shipped per-atom
  quantizer rule is not reconstructible — hence the bracket, not a point.
- **The 12-atom render is the fidelity limit, and it is shared.** The receiver normalises atoms
  before the einsum, so a real 11-atom container would renormalise its own rotated atoms. Every
  exact `d_pose` in this family — ra2c's, jc1's, ra2's and mine — is measured on the 12-atom render,
  so the container change is unmeasured for all of them equally. It is not introduced here, and it
  cannot rescue a 35.5× miss.
- **The averaged pullback is a relaxation** (inherited from ra2). `G_p = mean_i J_iᵀJ_i` chooses the
  subspace; the true objective weights each pair by its own `J_i`, and per-row-weighted low-rank
  approximation is NP-hard. A per-pair-optimal subspace could beat my subspace — but the exact
  `d_pose` I report bounds nothing above it, and it would need 35.5×.
- **`J_i` is measured at the SHIPPED coefficients**, while every candidate sits ~13.5% of coefficient
  RMS away, so the model extrapolates. Realised acceptance is what makes that safe rather than fatal.
  A re-linearised second round is unmeasured (§5 explains why I did not fire it).
- **The trust-region solve is verified, not asserted**: the closed form matches an independent BFGS
  minimiser of the stated objective on 5 random instances (1e-5), and the orthogonality identity
  `‖δ‖²_{G_p} = ‖u‖² + ‖y^⊥‖²` holds to 1e-15.
- **jc1's keep-set row is INHERITED**, not re-run. ra2's subspace row is fully re-derived.

## 10. NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | **Nothing on the carrier.** Five treatments now span the space (α=0, rank truncation, keep-set re-fit, subspace, subspace + trust-regioned re-fit) and the rung's perfect-execution ceiling is 6.3–12.8% of the gap. It should stop consuming slots. | — | do not re-open without a NEW mechanism, not a new radius |
| 2 | **Port the per-pair trust-region + realised-acceptance pattern to a rung with real headroom.** The mechanism is sound and general: it bought 3.13× on a hopeless rung. It is the right shape for any lever where a linear model exists, is untrustworthy as a designer, and the per-object optimum varies — which describes the token stream and the semantic renderer. | unowned | when a rung with ≥ 10% of the gap is in flight |
| 3 | **Re-price every live carrier/pose claim through `ddm_ra3_advisory_pose_pricing.py` or the authority GT.** The 11.5× bar propagated through three arms unchallenged. Anything else quoting an advisory `d_pose` ratio as a score is suspect by up to 21×. | MAIN | $0, immediate |
| 4 | **Measure the byte credit before dividing by it.** §3's method (encode the candidate through the shipped coder, compare real blobs) applies to any section. Three different assumed credits circulated here (1,514 / 1,847 / 1,854 B) and the measured value is 913–1,658 B. | unowned | before any future rate rung is priced |
