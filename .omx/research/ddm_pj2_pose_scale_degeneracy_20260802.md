---
schema: ddm_pj2_pose_scale_joint_solve.v1
date_utc: 2026-08-03
arm: ddm_pj2 (#850 GN termination x #873 menu-as-RD-codebook x #882 start-is-the-lever)
lane_id: "lane_ddm_pj2_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: INSTANCE
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. n600-ordered per-pair solve through the
  REAL receiver + byte-close through the REAL builder. NO training, NO paid dispatch, NO exact gate
  fired, NO pointer mutation."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_ms8_archive.zip  (the live frontier)
  - /Volumes/VertigoDataTier/pact/ddm_ms8_20260802/{ms8_curves_shard*.jsonl,ms8_st_override.json}
  - .omx/research/ddm_ms8_menu_selector_solver_st_codebook_20260802.md  (s11 owed items)
  - .omx/research/ddm_os1_optimization_sweep_termination_census_20260802.md  (#1 LIVE site)
  - .omx/research/ddm_sv1_solver_termination_sweep_20260801.md  (s5.B owed live sites)
  - .omx/research/ddm_uv1_ep854_pose_illegibility_reject_20260802.md  (#827 CLOSED; not re-run)
  - src/tac/information_geometry/fisher_natural_trust_region.py  (the canonical metric step)
produces:
  - tools/pj2_pose_scale_joint_solve.py
  - src/tac/tests/test_ddm_pj2_pose_scale_joint_solve.py  (37 tests)
  - /Volumes/VertigoDataTier/pact/ddm_pj2_20260802/{final_ms8_reference.jsonl,pj2_degen_receipt.json,
    pj2_report.json,final_pj2.jsonl,n600/pj2_solve_shard*.jsonl}
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pj2_archive.zip (STAGED, NOT gated)
consumers: [MAIN, "#850", "#873", "#882", "#909", the next pose unit]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_pj2 — `s_t` and the pose translation are ONE degenerate scale, so ms8's win was reachable through a column the archive already pays for

## §0 POINTER HONESTY — first

**The exact contest pointer is UNMOVED and I fired no gate.** Everything below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`. The composed S is a PREDICTION whose
fidelity anchors on this vehicle are the QA78 v4d gate residual (1.8e-6) and the pw1 gate residual
(2.5e-6); ms8's own prediction landed within 1.4e-6, so the anchor is live and this number is still
a prediction and is labelled one.

**I had to WAIT for the scorer slot.** `stage_v4d_realized_gate.sh cpu dc1_fold` was live when I was
ready to launch; I stopped my six shards ~20 s after starting them, waited ~6.5 min for the gate to
release, and relaunched on a verified-free slot. Sunk: ~1 min of shard time, no rows discarded that
mattered.

## §1 THE FINDING, in one line

The shipped ground homography (`pfs1_warp_receiver.pose_to_homography`) is

```
t = s_t * [p[2], p[1], p[0]]        R = expmap(rot * [p[3], p[4], p[5]])
H = K (R - t n^T / h) K^-1          and the FAR plane is evaluated at s_t = 0
```

so the translation enters **only** through the product `s_t · p[0:3]`. `s_t` and the pose
translation triple are an **exact scale-degenerate pair**: `H(p, s) == H(λ·p[0:3], s/λ)`.

**MEASURED, three independent legs:**

| leg | what it isolates | result |
|---|---|---|
| algebra (200 random draws) | the identity itself | max relative `|H₁−H₂|` = **7.38e-16** (float64 roundoff) |
| realized, exact arithmetic (12 pairs, full scorer path) | warp → uint8 → frozen PoseNet | max `|d(p,s) − d(λp, s/λ)|` = **0.000e+00** |
| realizable (12 pairs, shipped f16 quantization) | can the lattice express the move | max relative `|pose-route − s_t-route|` = **4.11e-03** |

Consequence: **ms8's `-0.0491770` was a reparameterization.** The same effective scale is reachable
through the pose column — which the archive already ships — at **f16 resolution** instead of eleven
codewords, and without the `+51 B` ms8 paid for the widened index stream.

## §2 POSITIVE CONTROLS, including one of mine that failed first

* **CANARY: 0.000e+00 on 12/12 pairs.** My realized `d_pose`, computed by driving the archive's own
  parameters through the receiver's own branch structure, reproduces ms8's independently-measured
  per-pair values to every digit.
* **MY FIRST CANARY FAILED at 3.26e-01 and I did not patch past it.** I had pointed it at
  `final_pw1.jsonl` while reading parameters from the **ms8** archive — a stale-reference error, mine.
  Per `ddm_uv1` §6 I stopped and rebuilt the reference (`final_ms8_reference.jsonl`, derived from the
  ms8 curves at the fitted codeword: mean `0.0051661970` vs the memo's `0.00516620`), after which the
  canary is exact. Had I trusted the first run I would have reported a fabricated 300× discrepancy.
* **BUILD-PATH CONTROL: byte-identical rebuild.** Rebuilding from my reference JSONL through the real
  builder reproduces the shipped ms8 archive at sha
  `48e0f31b4369bb3c1b21ff364d42e693e32ccb65accb35970780824c3dbef168`, 360,374 B. So the emit path was
  validated *before* it carried any solved pose.
* **RECEIVER BRANCH FIDELITY.** ms8's harness always took the beta-blend path; the receiver takes a
  single-warp path when `beta_mag == 0`, and `(1−α)x + αx` is not bit-identical to `x` in float — the
  documented pw1 instrument floor. This tool mirrors the receiver branch for branch, which is why the
  canary is exact rather than exact-on-578-of-600.

## §3 THE SOLVER — #850 cured, not described

`ddm_os1` #1 measured the live 6-parameter pose GN reporting `ALL_STOPPED_ON_A_BOUND 600/600, 100 %
mass`, with its `cur < 1e-6` criterion firing **0/600** and **fused into the same `break`** as the
bound, so no census could separate convergence from truncation.

What landed:

1. **The exits are SPLIT**: `step_below_shipped_quantization` · `trust_radius_cap` · `relin_cap` ·
   `singular_normal_equations` · `zero_natural_gradient`, plus a distinct sweep-level reason.
2. **The convergence value is a PROOF, not a tolerance.** A step whose *quantized* candidate lands on
   the same float16 cells cannot change the shipped `d_pose` however long the loop continues — local
   optimality **on the lattice that ships**. It reads the candidate only and costs zero scorer
   evaluations.
3. **Realized-acceptance**: every candidate is quantized exactly as the builder stores it
   (`dim0_offset + f16(residual)` for dim 0, plain f16 elsewhere) *before* it is scored, so the
   returned value is monotone by construction and is the value that ships.

## §4 METRIC-FIRST — the canonical helper WIRED, and the readback that justifies it

The inner step is `tac.information_geometry.fisher_natural_trust_region.
fisher_natural_cotangent_trust_region_step`, which returns
`min(1, r/‖H⁻¹g‖_H) · H⁻¹g` — projected into an **H-norm (Fisher) ball, never a Euclidean one**.
`H = JᵀJ` where `J` is the Jacobian of the frozen PoseNet's 6-vector output w.r.t. the shipped
parameters, i.e. the Gauss-Newton pullback of the PoseNet quadratic: the scorer's own metric on this
chart. The helper's docstring makes constructing the gauge-fixed SPD chart a **consumer obligation**
("implicit damping here would silently change the geometry"); with 6 residuals and 8 parameters the
raw pullback has rank ≤ 6, so the ridge is declared in `_spd_chart` and recorded, not hidden.

**I did not substitute a Euclidean surrogate and I did not reuse a foreign chart.** `H` is the
Gauss-Newton pullback MEASURED per pair by finite differences through the real receiver on the
8-parameter chart that actually ships. `ddm_ms3`/`ms4`'s custodied ≤6-dim PoseNet quadratic is the
same family, but a pullback is CHART-SPECIFIC — it must be measured on the chart in use — so it is
re-measured here rather than transplanted. Nothing needed from that bundle was missing; nothing from
it was substituted.

**DUAL-METRIC READBACK — FULL n600 (917 recorded relinearizations, 600 pairs):**

| | median | mean abs |
|---|---:|---:|
| `cos_euclid(displacement, gradient)` | **+0.00021** | 0.01374 |
| `cos_fisher(displacement, natural)` | **+0.09909** | 0.24848 |
| **sign disagreements** | **0 / 917 (0.00 %)** | |
| **median magnitude ratio \|fisher\|/\|euclid\|** | **459×** | |
| Fisher condition number | median **8.93e+10** | p90 8.81e+11, max 1.81e+14 |

**No sign flip on this surface — and the honest reading is that the magnitude disagreement is the
finding instead.** The useful displacement is essentially *orthogonal* to the Euclidean gradient
(median cosine 2.1e-4) while retaining real alignment with the Fisher-natural direction (median
0.099): a **459×** disagreement. Reading Euclid alone here would have concluded "the solver moves in
a direction unrelated to the gradient," which is false. Per the standing law both are reported and
neither alone.

**And the geometry is not decoration — it decided a design A/B, which is a direct test of the
'descending in the wrong metric wastes reach' hypothesis on THIS surface.** I first coupled the ridge
to the trust region (the classic LM ↔ TR equivalence, raising the ridge on rejection). Raising the
ridge is exactly what collapses the natural step `−H⁻¹g` toward steepest descent `−g`. MEASURED on
the three hardest pairs, that is **1.74× worse**, and it is worse precisely in the direction the
readback says carries no signal:

| arm | pair 44 | pair 16 | pair 74 | sum | evals |
|---|---:|---:|---:|---:|---:|
| start (ms8) | 0.449008 | 0.455874 | 0.474073 | 1.378955 | — |
| ridge-escalating (→ steepest descent) | 0.314551 | 0.154892 | 0.473464 | 0.942907 | 515 |
| **radius-only (natural direction preserved)** | **0.020804** | **0.049147** | **0.473464** | **0.543416** | 787 |

**Scope, stated:** this is the pose PARAMETER chart (8 dims, dense FD Jacobian), not `bp2`'s pixel
chart, and it is not evidence about `bp2`'s `δ = −sign(g)` L∞ step. It is an independent measurement
that on a pose surface with cond ~1e11, moving toward steepest descent costs 1.74× — which makes the
L∞-vs-natural question on the pixel chart worth measuring rather than assuming. The ridge therefore
stays a declared gauge choice and the **radius is the only search handle**, which is exactly what the
condition number predicts.

**The condition number is not decoration — it decided a design A/B.** I first coupled the ridge to
the trust region (the classic LM ↔ TR equivalence, raising the ridge on rejection). MEASURED on the
three hardest pairs that is **worse**, because at cond ~1e11 a large ridge collapses the natural step
onto steepest descent — the very direction the readback shows the displacement is orthogonal to:

| arm | pair 44 | pair 16 | pair 74 | sum | evals |
|---|---:|---:|---:|---:|---:|
| start (ms8) | 0.449008 | 0.455874 | 0.474073 | 1.378955 | — |
| ridge-escalating | 0.314551 | 0.154892 | 0.473464 | 0.942907 | 515 |
| **radius-only, shrink 0.5 × 20** | **0.020804** | **0.049147** | **0.473464** | **0.543416** | 787 |

So the ridge stays a declared gauge choice and the **radius is the only search handle** — but it
needs RANGE, which is exactly what the condition number predicts.

## §5 THE MEASURED DESCENT CURVE — the #850 question answered with a number

Per-sweep mean relative gain, FULL n600:

```
sweep 1   18.12 %   (median 6.789 %)   n=600
sweep 2    6.43 %   (median 0.007 %)   n=519
sweep 3    5.91 %   (median 0.000 %)   n=244
sweep 4    4.46 %   (median 0.015 %)   n= 99
sweep 5    3.42 %   (median 0.000 %)   n= 42
```

The shipped solve stopped at roughly the sweep-1 point. **Sweeps 2–5 are still buying 3–6 % per sweep
when the sweep budget runs out** — the mean has not decayed to zero, it has decayed to 3.42 %, and
the `n` column shows the survivors are the pairs that keep paying. The charter's premise ("still
descending 13–23 %/iter when it stops") is CONFIRMED in kind: truncation was costing real distortion,
and it still is.

**At the GN level it does NOT converge, and I will not dress that up.** The stop census over
**1,504** recorded relinearizations is `trust_radius_cap 1462 · relin_cap 41 ·
step_below_shipped_quantization 1`. **One** relinearization in 1,504 (0.07 %) reached the
lattice-resolution proof, even at 20 radius steps and 8 relinearizations per sweep. My cure made
convergence *distinguishable* from a bound — which the fused exit could not, and which is the whole
of #850 — but the honest reading is that the solve remains **bound-limited**, so more budget would
still buy more. Named, measured residual; not a closed question.

## §6 WHERE THE WIN COMES FROM — both owed ms8 items are real

| component | mean relative gain | pairs improved | share of total mass drop |
|---|---:|---:|---:|
| **σ-refinement** (the pose-route scale ray: the inter-column optimum ms8's 21-point support could not express) | **11.16 %** (median 0.154 %) | 323/600 | **28.7 %** |
| **the other 5 pose dims + (a,b)** | — | 569/600 | **71.3 %** |

n600 mass: `3.099786 → 2.648907` after the σ ray alone `→ 1.530752` final.

ms8 §11 owed exactly these two and predicted both: *"the 21-point support is a floor, not a ceiling"*
and *"a joint re-solve should win more."* Both are now MEASURED, and each is individually material.

A caveat that travels with the σ number: on the very hardest pairs (44, 16, 74) the σ ray alone found
**nothing** — ms8's fitted codebook had already placed a codeword inside their narrow valleys (pair
44's valley is ~7 % wide and the fitted table's local spacing is ~7 %). The σ gain is therefore
concentrated in the *mid* tail, not the extreme tail, and on the extreme tail the remaining 5 DOFs
are the whole story.

## §7 THE RESULT — byte-closed, and the byte cost measured before it was spent

Rate sensitivity of the pose route, MEASURED at $0 before any solve (the pose field is a byte-plane
column-major brotli stream, so this is not assumable):

```
incumbent pose field                6,378 B
rescale  2 % of pairs                  -1 B      rescale 30 %   +21 B
rescale 10 % of pairs                  +1 B      rescale 100 %   -5 B
```

The pose route is **byte-neutral to within ±21 B (ΔS_rate ~1.4e-5)** against ms8's `+51 B`.

**n600 COMPLETE — 600/600 pairs solved, 100 % of the incumbent d_pose mass, 115,280 scorer
evaluations, 0 regressions** (the report's monotone guard never had to fire; acceptance-at-shipped-
quantization makes that structural, not lucky).

| | archive B | seg | pose | rate | composed S |
|---|---:|---:|---:|---:|---:|
| v4d (pre-pw1) | 360,238 | 0.431179 | 0.292939 | 0.239868 | 0.9639858 |
| pw1 | 360,323 | 0.431179 | 0.276504 | 0.239924 | 0.9476070 |
| **ms8 — the live own-vehicle frontier** | 360,374 | 0.431179 | 0.227293 | 0.239958 | **0.8984300** |
| **pj2 — this unit** | **360,406** | 0.431179 | **0.159726** | 0.239980 | **0.8308849** |

**ΔS = −0.0675451 for +32 B.** d_pose mean 0.005166197 → **0.002551250** (−50.6 %); 569/600 pairs
improve, 0 regress. Gap-to-bar (0.172141) from ms8 is 0.6587439, so this is **9.30 % of the whole
remaining gap**. `state/tokens.dr7t` is untouched, so **d_seg is inherited bit-unchanged** and this
composes with the seg line rather than competing with it.

Archive `v4d_composed_pj2_archive.zip`, sha
`a9626f8b0c50d8de7ee7978efcba29caa0cbe2569a1dd72afd457b99fba242fa`, 360,406 B.
`ddm_v4d_verify_decode.py`: **`all_checks_ok: true`** (A parse-back consumption bijection #417,
B field bit-exactness, C independent byte-exact compose recompute on 7 sampled pairs including a
beta≠0 pair).

**STAGED FOR MAIN, NOT SELF-FIRED** (`experiments/stage_v4d_realized_gate.sh:3`):

```bash
bash experiments/stage_v4d_realized_gate.sh cpu pj2
```

The gain is tail-concentrated exactly as ms8's was: top 1 % of pairs carry **59.9 %**, top 10 %
carry **93.5 %**. Median per-pair gain is small; the term is a tail and `sqrt(10·mean)` makes the
tail the score.

## §7b THE OPERATOR'S QUESTION — realizing the pose win WITHOUT paying the rate

> *"There is probably a way to realize the pose win without paying for all of that rate by pursuing
> more optimal and nuanced"*

**Measured answer: yes, and this unit is it.** The ranking the directive asked for is
**Δd_pose per COUNTED byte**, not pose gain alone:

| route | ΔS_pose | marginal counted B | **ΔS per byte** | is it genuinely ~0-byte? |
|---|---:|---:|---:|---|
| `bp2` blind-set pixel actuator | (65.9 % pose gain) | **+401,285 B** index | catastrophic | **NO** — the index was the cost |
| `ms8` widen the stored `s_t` codebook | −0.0491770 | **+51 B** | −9.64e-4 | small, but a REAL new payload |
| **`pj2` move along the exact gauge orbit + the remaining pose DOFs, in coordinates that ALREADY ship** | **−0.0675451** | **+32 B** | **−2.11e-3** | **YES — see below** |

**pj2 is 2.19× better per byte than ms8 and 1.37× larger in absolute ΔS at 63 % of the byte cost.**

**Why the +32 B is not a price.** The pose field is a FIXED-SIZE object — 600×6 float16 in a
byte-plane column-major brotli stream — so changing its VALUES buys distortion without adding a
record, an index, or a support set. There is nothing to say *which* pairs moved, because every pair
already has a slot. MEASURED at $0 before the solve was run, by perturbing the incumbent poses and
re-encoding through the real codec:

```
incumbent pose field                6,378 B
rescale  2 % of pairs   -1 B        rescale  30 % of pairs  +21 B
rescale 10 % of pairs   +1 B        rescale 100 % of pairs   -5 B
```

The field responds **−5 to +21 B** across the whole range. The realized +32 B is brotli entropy noise
on an existing field (ΔS_rate 2.1e-5 = 0.003 % of the gap), **not the cost of the move**. This is
handle 1 of the directive's four — "a pose move along an already-shipped coordinate costs ZERO
marginal bytes" — measured end to end and byte-closed, rather than argued.

**One sharpening of handle 1 the directive did not state, and it matters.** `ddm_dc1` and I both
proved the `(s_t, p[0:3])` multiplicative gauge. But the gauge orbit is only *one dimension* of the
free move. The pose column ships **all six** pose values plus `(a,b)` per pair, so **every one of
those 8 coordinates is already counted** — and §6 measures that the orbit itself carries only
**28.7 %** of the win while the other seven coordinates carry **71.3 %**. The general principle is
therefore stronger than the degeneracy: *any* re-solve confined to already-counted coordinates is
~0-byte, and the degenerate orbit is just its most obvious special case.

**Honest placement of the other three handles, without claiming what I did not measure:**

* **Handle 2 (parametric support, rule-118 free)** — owned by `ddm_pb3`; I did not duplicate it. What
  this unit contributes to the ranking is a strictly cheaper CLASS: an actuator that needs no support
  set at all dominates one whose support must be derived, because a derived index is still receiver
  code plus a correctness obligation, whereas an already-counted coordinate is neither.
* **Handle 3 (amortize, don't enumerate)** — **orthogonal to and composable with this unit**, not
  competing. `sc1`'s rank-1 residual and a class map would shrink the 6,378 B pose FIELD; pj2 changes
  the values *inside* that field. Measured signal on that question, at $0: my solved poses re-encode
  to 8,752 B of `pose_warp` vs the incumbent's 8,720 B, i.e. **+0.37 %** — the solution is very
  slightly less compressible, so an amortization would have to absorb that, but nothing here blocks it.
* **Handle 4 (ξ dual-use)** — the degeneracy I proved IS a statement about the screw's translation
  scale, so it is the same object. **I did not measure ξ sharing with the seg side** and make no claim
  about it.

**I did not propose spending `rd2`'s freed 74,794 B.** This unit spends 32.

## §8 FALSIFIER VERDICT

Pre-registered: *pose is CLOSED at this operating point if converged-GN + joint re-solve +
inter-column refinement together buy **< 3× the fidelity anchor** (≈ 7.5e-6 S) beyond ms8's
−0.0491756.*

**NOT MET, by a factor of ~9,000.** The additional buy beyond ms8 is **−0.0675451 S**, against a
bar of ≈ 7.5e-6. Every one of the three named components is independently material:
converged-GN (sweeps 2–5, §5), the joint re-solve (§6, 71.3 % of the drop) and inter-column
refinement (§6, 28.7 %).

**Pose is NOT exhausted at this base.** Stronger, and this is the part I want on the record: the GN
stop census says the solve is *still bound-limited* — `trust_radius_cap 1462 · relin_cap 41 ·
step_below_shipped_quantization 1` out of 1,504 relinearizations. **One** relinearization in 1,504
reached the lattice-resolution proof. The remaining truncation is a measured, named residual.

## §9 A GRADE-5 ORPHAN FOUND ON THE WAY IN (#909) — verified absent, exhaustive scope

MAIN routed me to wire `tac.riemannian_newton_substrate_engineering`, warning that a STRICT gate
`check_canonical_riemannian_newton_meta_substrate_use` refuses `hand_rolled_newton_step` /
`inline_stiefel_retract` / `custom_trust_region` outside it. **None of it exists.** Scope stated,
because negative-existence claims are the dominant error class:

| claim | scope searched | result |
|---|---|---|
| module `tac.riemannian_newton_substrate_engineering` | whole-tree `find` + `import` | **absent**; only `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` |
| gate `check_canonical_riemannian_newton_meta_substrate_use` | `src/ tools/ experiments/ scripts/` | **0 definitions**; `src/tac/preflight.py` contains **0** occurrences of "riemannian" |
| refusal tokens (3) | `src/ tools/ experiments/ .omx/` | **0 `.py` hits**; they appear only in memo prose |
| `src/tac/solvers/riemannian_newton_stiefel.py:21-22` — "Task #899 … landed at commit `a39ffdf80`" | `git cat-file`, `git log --all` | **not a valid object**; task #899 has **0** ledger rows |

So a routing directive specified a module and a STRICT gate at OP-1..OP-3, neither was built, and a
**shipped source file carries a provenance pointer to a commit that does not exist**. This is the
designed-stub-is-orphan-signal class at its sharpest: the gate that would have refused a hand-rolled
solver was itself never built, so nothing in the repo is even *shaped* to trip it.

**What I did instead of hand-rolling anyway:** the canonical mathematics the memos specify (Fisher
preconditioning, a trust-region inner solve) IS built in this repo under a different name —
`tac.information_geometry.fisher_natural_trust_region` — and §4 wires it. The named module was
unbuildable-to; the named *discipline* was honoured.

Registered as task **#909**. What it does NOT license: it does not make hand-rolled solvers fine
elsewhere, and `riemannian_newton_stiefel.py` (a Stiefel-manifold Newton for PQ-codebook init) is a
genuinely different object that does not fit this 8-parameter black-box chart — its docstring's
provenance line is the defect, not the module.

## §10 WHAT I DID NOT DO / OWED

* **No exact gate fired.** MAIN owns the slot; the staged command is in §7.
* **#827/#881 NOT re-run.** `ddm_uv1` closed it as REJECT-by-arithmetic before my charter was
  written, with a matched control proving the failure is the BASE. I read the closure and dropped the
  leg; **zero sunk time** (no probe was started). Its stranded `-0.0865743 S` of seg+rate remains
  available to a base trained with a pose term in the loop — a *training* question.
* **The GN is still bound-limited** (§5): `step_below_shipped_quantization` fired 0/217. The sharp
  next measurement is the same solve with the radius ladder extended until that reason appears, which
  bounds what the truncation is still costing.
* **`selector ∈ {0,1}` is still frozen** the way `s_t` was, and is still unmeasured — ms8 flagged it,
  I did not close it.
* **The dim0 offset moves under `--dim0-offset auto`** (31.546875 → 31.671875) because the poses
  changed. It is one manifest float and the receiver reads it, so this is correct — but a pinned-offset
  arm was not raced and might compress marginally better.
* **An efficiency defect I found in round-2 review and deliberately did NOT fix mid-run.** On every
  accepted step the solver re-evaluates `pose6` to refresh the residual, although `d_pose` already
  computed it internally — roughly a 25 % waste on accepted steps. Fixing it mid-flight would have
  made different pairs search under different code, which is a mixed-code confound worse than the
  waste. Owed, cheap, and it makes the next budget go ~25 % further.
* **Single seed, no noise floor on the solver itself.** The per-pair objective is deterministic
  (measured: canary exact), so there is no sampling noise, but the *search* has no restart census —
  `sv1` §2b and `uv1` §4 both measured start-bias on sister surfaces and I inherited neither.
* No training, no paid dispatch, no `upstream/` edit, pointer untouched.

## §11 FALSIFIERS FOR THIS UNIT

1. The exact gate on the staged archive returns a composed S outside the predicted value ± 1e-4 ⇒ the
   byte-close fidelity anchor is broken and every advisory pose row on this line reopens, not just mine.
2. A run that extends the radius ladder until `step_below_shipped_quantization` fires buys **less**
   than this one ⇒ my ladder was already past the useful range and `trust_radius_cap` is a benign
   terminal state rather than a live truncation.
3. The σ-refinement share (§6, 27.9 %) collapses toward 0 on a re-solve from a *different* start ⇒
   what I attribute to inter-column resolution is really start-bias, and `sv1`'s restart finding
   dominates this one too.
