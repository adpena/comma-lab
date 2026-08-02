# ddm_ft1 — the clip-fraction falsifier ran. The prediction is dead, and the stale set is bigger than we thought.

**Date:** 2026-08-02 · **Task:** #891 · **Arm:** ddm_ft1
**Axis:** `[macOS-CPU scorer-free advisory]` NON-PROMOTABLE · `score_claim=false` ·
pointer `0.1910828242 [contest-CPU]` **UNMOVED** (nothing here is a score mover; it is a
measurement that stops a wrong repair from being funded).
**Artifacts:**
`/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/ft1/{ft1_channels.jsonl,ft1_receipt.json}` (beta axis),
`.../ft1_full/{ft1_channels.jsonl,ft1_receipt.json}` (full fit→ship displacement).

---

## Answer first

**The clip-fraction prediction is FALSIFIED, and it retires nothing.** The number the arm asked
for — "how many of the 244 can we retire for free?" — is **ZERO**. Not "few": zero. On the full
fit→ship displacement, **0 of 267** stale pairs have an inert partner.

The criterion the prediction proposed would have retired **199 of 267** — including the pairs
that moved *most*. Its own worst case is the reductio: pair 17 ships `beta = -7.5`, moves 43.9%
of its pixels (`pre_rms` 4.00, the largest drift on the beta axis) and has a clamp channel of
**exactly zero**. The clip criterion puts the worst pair in the population at the top of its
retirement list.

Three further results, in descending order of how much they change what we do next:

1. **The stale partner set is wider than ddm_fs1 counted.** fs1 counted beta drift: 244 pairs.
   But `pw1` moved the **pose** on 89 rows and `mq1` moved **p1/p2** on 128 more, and
   **neither re-solved (a,b)** (MEASURED: zero rows changed `a` or `b` across either stage).
   Counting *any* drifted fitted partner, the stale set is **267**, with **178** pairs shipping
   an `(a,b)` fitted against a pose that no longer exists. This does not contradict fs1's 244 —
   it widens the definition from "beta drifted" to "any fitted partner drifted", which is what
   staleness means.

2. **The hull escape is 101 pairs, not 59.** fs1 tested magnitude against `min/max` of the menu.
   At fit time the *sign* was pinned to the pose yaw (`beta = g·yaw_sign`, `g ≥ 0`), so a shipped
   opposing sign is a direction the fit never sampled either. **63 pairs** ship one; the union
   with the 59 magnitude escapes is **101**, of which **100 are stale**. A magnitude-only hull
   test cannot see them.

3. **The derivation named the smaller quantization channel.** `_to_uint8` is `clip(round(·))`.
   ROUND is non-invertible too and bites every pixel, not just saturated ones. MEASURED:
   sub-quantum moves account for **62.5%** of all uint8 changes; the clamp accounts for
   **0.0015%**.

---

## What was measured, and what was not

Scorer-free, $0, n600 (all 600 pairs, no subset), through the **real receiver primitives**
(`inflate_runner_v4d.Decoder._warp_pair` + `pfs1_warp_receiver._to_uint8`, imported not
re-implemented). No PoseNet forward anywhere — the n600 scorer slot stayed free for the live job
(pid 18732).

### Controls (both executed, both hard-fail)

| Control | Result | Meaning |
|---|---|---|
| `W(a·I + b) == a·W(I) + b` | max rel residual **3.39e-16** | The derivation's algebraic premise is **CONFIRMED**. The warp is linear and constant-preserving; the operators genuinely commute. |
| tool compose vs `Decoder.f0` | **bit-identical** (max abs u8 diff 0) | What was measured is the shipped receiver, not a lookalike. |
| stale-set reproduction | **244 / 59** | Independently reproduces ddm_fs1's census from the same artifact. |

### The three channels, on the stale set

| | beta axis (244 stale) | full fit→ship (267 stale) |
|---|---|---|
| **CLAMP** `clip_sym_diff` mean | 1.265e-06 | 1.736e-06 |
| — median | **0.0** | **0.0** |
| — pairs with exactly zero | **185 / 244** | **199 / 267** |
| — share of total uint8 change | **1.48e-05** | 1.48e-05 |
| **ROUND** share of uint8 change | 0.677 | 0.625 |
| **GEOMETRY** `u8_diff_frac` mean | 0.1067 | 0.1171 |
| — `pre_rms` mean / max | 0.505 / 4.00 | 0.762 / **10.45** |
| pairs with an **inert** partner | 1 | **0** |

### Does the predictor rank anything? (Pearson, stale set)

| | vs `|Δbeta|` | vs `pre_rms` |
|---|---|---|
| `clip_frac_ship` (the prediction) | **−0.019** | **−0.011** |
| `clip_sym_diff` (clamp channel) | +0.051 | +0.196 |
| `pre_rms` (the survivor) | **+0.420** | — |

The predictor is not weakly correlated; it is **degenerate**. Its median is exactly zero across
three quarters of the population it was supposed to rank. A predictor with no variance cannot
order anything, and here it anti-orders slightly.

### What was NOT measured — and it matters

Every number above is **image-domain**. It bounds the channels through which `d²L/d(a,b)dβ` can
travel; it does **not** measure that coupling. Closing this needs one PoseNet job — re-solve
`(a,b)` at the shipped geometry on the top-ranked pairs and difference the realized `d_pose` —
and that needs the n600 scorer slot, which was occupied. **OWED**, and it is the single
measurement that converts a falsified predictor plus a derived mechanism into a measured cost.

---

## Why the prediction failed (DERIVED)

The premise is right and the inference does not follow from it.

Commuting **operators** do not imply a vanishing **mixed partial of the loss**. With
`X_β := W_β(I)` and `g` the frozen-PoseNet MSE, `L(a,b,β) = g(a·X_β + b)`, so

```
d²L/da dβ  =  ⟨∇g, dX/dβ⟩  +  a·⟨Hess_g · dX/dβ, X_β⟩
```

Both terms are generically non-zero for **any** non-zero `dX/dβ`, with no clamp anywhere in the
expression. The coupling is `O(‖dX/dβ‖)` — how far the geometry moved the image, times the
**curvature of the frozen scorer**. Clipping is one additive channel; it was never the mechanism.

This is the whole error in one sentence: *the pipeline being non-invertible at the clamp is what
makes order matter for the OPERATORS; it is not what makes the OPTIMUM move.* The optimum moves
because PoseNet is nonlinear and the picture changed.

The alternative hypothesis the arm named — ISP tone curve — is **not** what the data says either.
There is no residual tone curve in this path: the receiver is `warp → a·(·)+b → clip(round(·))`,
end to end, and the commutation control confirms the only non-linearity is the uint8 knife-edge.
The mechanism is scorer curvature, which is a third answer, and a larger one.

---

## Consequence for the fiber transport

The transport itself is correct and is landed:

```
δ(a,b) = −H_ab⁻¹ · (d²L/d(a,b)dβ) · Δβ
```

`H_ab` is the 2×2 GN Hessian the solver already forms from `GAIN_FD=0.02` / `BIAS_FD=2.0`; the
mixed partial costs **one** extra finite difference in beta per pair and is exactly what
sequential coordinate descent throws away. `fiber_transport_delta_ab` implements it and refuses a
singular or indefinite `H_ab` rather than pseudo-inverting an unidentified direction.

But the retirement that was supposed to shrink the work did not happen, so the population is the
full 267, and it splits:

* **~144 pairs** (244 beta-stale − 100 outside the fitted set) are inside the hull → **linear
  transport is admissible**.
* **100 pairs** are outside it (59 magnitude escapes, 63 opposing sign, union 101 of which 100 are
  stale) → **linear transport has no warrant**; these need the real joint `(a,b,β)` Gauss-Newton.
  `transport_admissible` returns that boundary explicitly so the two do not get blurred.
* The **178** pose-drifted pairs are a *different* fiber question — the `(a,b)` GN linearized at
  one pose and there was no pose menu at all, so the boundary there is the GN trust region, not a
  discrete hull. Not answered here; flagged rather than fudged.

**Ranking that survived:** `pre_rms` (the pre-quantization RMS displacement), encoded as
`staleness_priority`, which also names `clip_frac`/`clip_sym_diff` as a **forbidden key** with a
pointer to this falsification so the dead predictor cannot be rediscovered.

---

## The stamping debt (designed, deliberately NOT landed)

`ddm_fs1.fit_staleness` returns `UNDETERMINED_NO_CONTEXT` on every live row because no solver
stamps what its coefficients were fitted against. The patch is generated and verified to apply
cleanly, and is handed back rather than applied — the v4c file is under a live run (pid 18732):

**`.omx/research/ddm_ft1_fit_context_stamp_20260802.patch`** (`git apply --check` passes)

Three sites, and the third is the one that actually matters:

1. `experiments/ddm_v4c_resolve.py` `run_photo` — stamp `fitted_against={"beta":0.0,"p":[…]}` +
   `fit_menu={"beta":[0.0,0.5,1.0],"beta_sign_pinned_to_yaw":True}` at the rung-B rec.
2. `experiments/ddm_v4d_resolve.py` refine — same stamp at the `_refit_ab` site.
3. `tools/mq1_joint_pose_refine_emit.py` — **carry the inherited context forward verbatim**, at
   both the per-pair rec and the merged emit. A refiner that moves a partner without re-solving
   the dependent must NOT re-stamp: a fresh stamp would claim a freshness that was never
   re-established. Dropping it silently downgrades every replaced row to
   `UNDETERMINED_NO_CONTEXT`, which is how this became hand-archaeology in the first place.

`fit_menu` carries `beta_sign_pinned_to_yaw` because that single flag is what makes the 63
opposing-sign escapes visible to an automated hull test.

---

## Landed

| Artifact | What |
|---|---|
| `src/tac/canonical_equations/ddm_ft1_photometric_beta_commutator_20260802.py` | The law: channel decomposition, both n600 censuses, `staleness_priority`, `transport_admissible`, `fiber_transport_delta_ab`. |
| `src/tac/tests/test_ddm_ft1_photometric_beta_commutator.py` | 34 tests, behaviour not constants (hand-placed saturated pixels, sub-quantum moves, a quadratic with a known stationary point). |
| `tools/ddm_ft1_clip_fraction_falsifier.py` | The n600 measurement, resumable, two hard-fail controls, `--fit-jsonl` for the full displacement. |
| `.omx/research/ddm_ft1_fit_context_stamp_20260802.patch` | The stamping design, verified-applying, deliberately unapplied. |

## Owed

1. **The one scorer job** (n600 slot): re-solve `(a,b)` at the shipped geometry on the top-`pre_rms`
   pairs, difference realized `d_pose`. Converts a bounded channel into a measured cost and tells
   us whether any of this is worth bytes.
2. **Land the stamping patch** once pid 18732 releases the v4c file.
3. **The pose fiber** — 178 pairs, trust-region boundary not hull boundary, unanswered.
