# RIPO categorical Fisher trust-region build specification

**UTC date:** 2026-07-14  
**lane:** `lane_ripo_fisher_isometric_trust_region_500_20260714`  
**task:** `500_ripo_fisher_isometric_trust_region_20260714`  
**status:** `PINNED_BEFORE_IMPLEMENTATION`  
**research_only:** `true`  
**authority ceiling:** `[macOS-CPU advisory]`; no score or pointer claim  
**budget:** `$0`; no remote, paid, GPU, or long-training launch

## 1. Purpose and corrected premise

Build a deterministic, NumPy-authoritative categorical Fisher trust-region clip and a bounded
fixed-trunk experiment for the current V9 EMA-best. The implementation MUST NOT encode the
intake memo's scalar `sqrt(delta / p1)` as a categorical logit law.

RIPO Eq. 10 bounds one sampled action's **importance ratio**. For a categorical softmax logit
update `u`, the coupled local geometry is

```text
F(p) = diag(p) - p p^T
q_F(u; p) = u^T F(p) u
D_KL(p || softmax(log p + u))
    = log(sum_k p_k exp(u_k)) - sum_k p_k u_k.
```

The two delta conventions are explicit and never mixed:

```text
delta_kl   = RIPO Eq. 10 physical KL budget
delta_quad = 2 * delta_kl = RIPO Eq. 11 absorbed convention
q_F <= delta_quad  <=>  0.5 q_F <= delta_kl.
```

For a proposed direction `v`, the local directional clip is

```text
alpha = min(1, sqrt(delta_quad / q_F(v; p))).
```

The exact finite-KL mode solves for the largest sign-preserving `alpha in [0,1]` satisfying
`D_KL <= delta_kl`. All updates are gauge-centred; constant-logit directions are null.

For winner `w`, rival `r`, symmetric margin motion `u=t(e_w-e_r)/2`,

```text
C_wr = p_w + p_r - (p_w - p_r)^2
q_F = t^2 C_wr / 4
|t| <= 2 sqrt(delta_quad / C_wr).
```

The binary reduction is exact only on a declared binary/winner-versus-rest restricted
submanifold and includes `p1(1-p1)`. The top-two reduction is exact only when top-two mass and
all other probabilities are fixed. Neither yields `sqrt(delta / p1)` in general.

## 2. Authority boundary: scorer logits versus witness-head logits

The categorical policy probabilities are frozen SegNet outputs. `out_sdf.weight/bias` produce a
different five-channel witness field upstream of palette rendering, uint8/resize `R`, and SegNet.
Therefore a pixel-aligned clip of witness-head logits using SegNet probabilities is NOT an exact
pullback Fisher trust region. The exact head metric is

```text
G_head = sum_pixels J_pixel^T F(p_pixel) J_pixel,
J_pixel = d(SegNet_logits after R) / d(out_sdf parameters).
```

This landing builds both:

1. the exact **output-space** categorical clip; and
2. a clearly labelled `cross_space_pixel_aligned_reprojection_v1` experimental formulation that
   clips proposed witness-head logit fields and least-squares reprojects them onto the one global
   affine `out_sdf` head.

The cross-space formulation may produce an INSTANCE/FORMULATION verdict, never a categorical-
Fisher FAMILY verdict. Full pullback/block-Fisher is the queued reformulation if it fails.

## 3. File ownership and collision containment

The implementation worker owns only these currently absent paths:

- `src/tac/optimization/ripo_fisher_trust_region.py`
- `src/tac/optimization/ripo_fisher_trust_region_mlx.py`
- `src/tac/optimization/tests/test_ripo_fisher_trust_region.py`
- `tools/probe_ripo_fisher_trust_region_saved.py`
- `src/tac/tests/test_probe_ripo_fisher_trust_region_saved.py`

The worker is not alone in the repository. It MUST preserve all concurrent edits and MUST NOT
edit or stage any other file. In particular it MUST NOT touch `witness_dsl/**`,
`canonical_equations/**`, `preflight.py`, either level-set trainer, V9 config/autoconfig/bijection,
`src/tac/scorer_surrogate/vjp_fidelity.py`, or `src/tac/information_geometry/**`.

## 4. Core interfaces

### NumPy authority

```python
clip_categorical_fisher_step_numpy_fp32(
    probabilities,
    proposed_logit_step,
    *,
    delta,
    delta_convention,
    mode,
    tolerance,
) -> FisherClipResult
```

Required modes:

- `local_directional`: the full `K`-class Fisher quadratic direction clip;
- `exact_kl`: deterministic monotone bisection of exact finite KL on `[0,1]`;
- `local_euclidean_ball`: gauge-fixed `lambda_max(F)` conservative ball;
- `uniform_l2_control`: probability-independent Euclidean clip, never labelled RIPO/Fisher.

The result/receipt includes the centred input/output, `alpha`, clipped flag, `q_before`, `q_after`,
`exact_kl_before`, `exact_kl_after`, `lambda_max`, winner/rival, top-two mass, delta convention,
quadratic approximation error, and top-two approximation error where defined. Accumulations and
root decisions use float64; returned arrays are deterministic float32. A post-cast constraint
check conservatively rescales or fails closed.

Helper functions expose exact KL, Fisher quadratic, winner-rival curvature/radius, and delta
conversion without hidden defaults.

### MLX parity

The MLX file lazy-imports MLX, accepts the same shapes/conventions, and contains no implicit device
selection. NumPy-fp32 is verdict authority. MLX parity is measured at >= 0.9997 and exact-KL bounds
are independently recomputed with NumPy.

## 5. Saved-array probe and experiment contract

The probe is foreground, bounded, resumable, atomic, and writes only to an explicit isolated
output directory. It has two fail-closed tiers:

### Tier A: algebra/calibration receipt

Inputs are full K=5 frozen-CPU SegNet probabilities/logits plus a proposed K=5 update. Scalar GT
margins alone are insufficient. Emit:

- exact source/input/config hashes;
- empirical exact-KL versus quadratic error;
- binary/top-two versus full-K approximation error;
- confidence-band clipping/variance summaries;
- flip-cost delta quantiles; and
- schema/metric IDs:
  `argmax_native_vjp_fidelity_v1` and
  `reachable_decision_geometry_fidelity.v1`.

If those arrays are absent, emit `NO_VERDICT_DATA_CUSTODY`; never synthesize them.

### Tier B: fixed-global-head experiment

The only mutable checkpoint arrays are `out_sdf.weight (5,96)` and `out_sdf.bias (5,)`; all other
arrays remain byte-identical. Current-EMA custody records `__cfg_self_orient=0` and
`__cfg_in_feat=80`, so this instance has no co-evolved self-orientation state to reconstruct.
The shared Fourier features and pair-conditioned fixed-trunk hidden states are deterministically
derived from the sealed checkpoint; their implementation/config custody is hashed. If this probe is
ever applied to a checkpoint with self orientation enabled, it MUST fail closed until a separately
specified, custodied reconstruction policy is supplied. Before any comparison, the unmodified
checkpoint MUST reproduce its separately recorded n600 baseline through the same NumPy deploy ->
actual camera `R` -> frozen CPU-Torch SegNet path. A mismatch blocks the experiment.

The proposed vanilla head step is generated from the same loss/data/order for every arm. For each
pixel, the induced witness-head update is clipped, then streamed normal equations reproject the
clipped field onto the global affine head. Compare:

- `vanilla_unclipped`;
- `uniform_l2_control`; and
- `categorical_fisher_{local_directional,exact_kl}`.

Delta values are not literals copied from the paper. They are derived from analytic symmetric
tie-crossing KL thresholds applied to frozen scorer probabilities that were measured after actual
`R`; they are not witness-head perturb-and-replay flip/spill thresholds. The sweep is made from
registered quantiles of desired-correction thresholds subject to protected-pixel spill thresholds.
`1.273 B/flip` is used only as an after-measurement rate break-even selector; it does not set delta.
Eikonal epsilon has incompatible units and cannot set delta without a measured Jacobian calibration.

Each candidate is measured together in deterministic batches across all 600 pairs through actual
`R` and frozen CPU-Torch SegNet. The receipt reports overall and per-GT-class error, CE, exact-KL,
net flips, harmful spill, confidence-band dispersion, and wall time. Pose and archive bytes remain
explicitly `NOT_MEASURED` unless genuine custodied measurements are added; no pointer move is legal.

Large frame/logit caches require SSD storage preflight. Metadata-only results declare
`large_artifacts_written=false`. Every stage has an atomic checkpoint, progress fingerprint, exact
input/source hashes, contiguous-state validation, and automatic success-only scratch cleanup.

## 6. Acceptance tests

The unit suite MUST cover:

- identity below bound and exact contraction above it;
- `q_after <= delta_quad` and `exact_KL_after <= delta_kl` after float32 cast;
- gauge invariance and constant-direction null space;
- direction preservation in the gauge quotient;
- seeded K=5 random inputs, near-zero probabilities, zero update, and near-bound cases;
- invalid simplex, nonfinite, shape, convention, and negative-delta failures;
- Eq. 10/Eq. 11 factor-of-two equivalence;
- exact K=2 reduction and a K=5 nonzero-tail counterexample;
- asymmetric finite-KL roots/signs;
- near-tie `(0.45,0.45,0.05,0.03,0.02)` versus confident
  `(0.98,0.01,0.005,0.003,0.002)` regression, proving the p1 heuristic reverses the actual radius;
- NumPy/MLX parity when MLX is installed; and
- probe refusal on missing full logits/probabilities or incomplete n600 custody.

Required local command:

```bash
.venv/bin/python -m pytest \
  src/tac/optimization/tests/test_ripo_fisher_trust_region.py \
  src/tac/tests/test_probe_ripo_fisher_trust_region_saved.py -q
```

## 7. Held triality/live wiring

No live DSL or canonical-equation edit is authorized in this lane. The findings memo will specify
the exact held LawRef/equation/Lever/consumer and standalone DAG feed. Until its owner integrates
those hot surfaces, the strongest status is
`BUILT_AND_MEASURED_BUT_NOT_V9_LIVE_WIRED` (or the corresponding custody blocker), never `V9-live`.
