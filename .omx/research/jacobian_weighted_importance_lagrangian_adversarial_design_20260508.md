# Jacobian-Weighted Importance Lagrangian Allocation - Adversarial Design - 2026-05-08

## Verdict

The strongest viable route is not another local tensor-error proxy. It is a
score-pullback allocator:

```text
pixel / boundary / scorer importance -> deterministic decoder VJP/JVP ->
per-weight or per-tensor quantization protection weights -> byte-closed
Lagrangian allocation -> exact CUDA auth eval.
```

The reframe "the pixel is the weight is deterministic" is correct only under
the contest custody constraint: the decoded pixels are a deterministic function
of charged archive bytes and `inflate.sh`; therefore any compress-time
importance map must be pulled through that exact deterministic decoder and then
charged back to the archive bytes that changed. CPU/MPS/proxy maps can route
work, but cannot promote, rank, or kill.

This design is the highest-probability repair path for the current
lossy-coarsening failure mode; it is not itself evidence of repair yet. The
exact PR106 UNIWARD-Lagrangian rms=0.05 packet saved bytes (`150511` vs
`186239`) but scored worse on contest-CUDA (`0.3371617511972341` vs baseline
`0.20454`) because decoder-symbol rel_err did not price downstream SegNet/
PoseNet sensitivity. Jacobian weighting must price that sensitivity before
another lossy PR106 dispatch.

## Primary Sources

- HNeRV stores a video as content-adaptive embeddings plus a decoder; this is
  the deterministic function whose weights/latents create pixels:
  [Chen et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html),
  [arXiv:2304.02633](https://arxiv.org/abs/2304.02633).
- Rate-distortion allocation is the right mathematical lens:
  [Shannon, 1959](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf).
- Second-order/Fisher saliency is the classical weight-importance precedent:
  [LeCun, Denker, Solla, 1989](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
  and [HAWQ-V2, arXiv:1911.03852](https://arxiv.org/abs/1911.03852).
- UNIWARD justifies content-adaptive distortion into difficult-to-model
  texture/noise regions, but it is not score-aware by itself:
  [Holub, Fridrich, Denemark, 2014](https://link.springer.com/article/10.1186/1687-417X-2014-1).
- Detector-informed allocation is plausible but must handle nonlinear detector
  response and detector mismatch:
  [Yousfi, Dworetzky, Fridrich, 2022](https://researchconnect.suny.edu/en/publications/detector-informed-batch-steganography-and-pooled-steganalysis/).
- Boundary weighting is relevant for segmentation, but only after being pulled
  back through the decoder:
  [Boundary IoU, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html).
- Film grain is acceptable only when deterministic and bitstream/accounted;
  AV1 signals grain parameters and seed as syntax and requires decoder support:
  [AOMedia AV1](https://aomedia.org/specifications/av1/),
  [AV1 bitstream spec](https://aomediacodec.github.io/av1-spec/av1-spec.pdf).

## Mathematical Formulation

Let charged archive bytes define decoder parameters and latents
`theta = (theta_w, theta_z)`. The deterministic inflate path emits frames:

```text
x = F(theta)
```

The contest score is:

```text
S(theta, B) = 100 * seg(F(theta))
            + sqrt(10 * pose(F(theta)))
            + 25 * B / 37,545,489
```

For an allocation choice `c_t` on tensor `t`, let `B_t(c_t)` be charged bytes
and `Delta theta_t(c_t)` be the quantization/coarsening perturbation induced
by that choice. The local second-order score model is:

```text
Delta S approx 25 * Delta B / 37,545,489
              + sum_t E[Delta theta_t^T H_t Delta theta_t]
```

where a practical generalized Gauss-Newton/Fisher diagonal is:

```text
H_diag(theta) approx diag(J_F(theta)^T W_x J_F(theta))
```

`W_x` is the pixel-space importance metric. It may include:

- official score VJP terms from PoseNet and SegNet;
- segmentation boundary multipliers from mask boundaries;
- component-specific weights matching `100 * seg + sqrt(10 * pose)`;
- holdout-calibrated stability multipliers.

For per-tensor allocation, reduce to a scalar:

```text
I_t = mean_or_sum_{i in tensor t} H_diag_i
```

Then select one codec/coarsening action per tensor:

```text
min_{c_1..c_T} sum_t B_t(c_t)
             + lambda * sum_t w_t * D_t(c_t)
```

where `D_t(c_t)` is the measured curve distortion, such as squared rel_err,
FP32 reconstruction energy, or candidate-specific finite-difference response.
The protection weight is:

```text
w_t = normalize(I_t / C_t)
```

`C_t` is optional texture/capacity from UNIWARD-like variance or wavelet
residual. Larger `I_t` protects a tensor. Larger `C_t` allows more distortion
only if score-pullback sensitivity does not object. This is the safe ordering:
score importance first, texture capacity second.

Dispatch breakeven must be computed before GPU spend:

```text
allowed_component_worsening < -25 * DeltaBytes / 37,545,489
```

For the exact negative lossy-coarsening result, the rate contribution improved
by about `0.01455`, but SegNet/PoseNet worsened by about `0.13992`; the byte
term was an order of magnitude too small to pay for unpriced component damage.

## Implementable Algorithm

1. Baseline custody.
   Start from a parsed, exact replayed archive. Record archive bytes/SHA,
   member SHA, runtime tree SHA, exact component fields, and dispatch claim
   status. Do not build from detached PR clones except as forensic inputs.

2. Decoder substrate.
   Use existing HNeRV parsers/packers to expose the exact decoder tensors and
   latents. PR101/PR106 code paths already expose 28 decoder tensors and K
   curves in:
   - `tools/pr101_lossy_coarsening_analytical.py`
   - `tools/pr101_omega_opt_uniward_weighted_allocation.py`
   - `tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py`

3. Pixel/scorer importance.
   On CUDA only, render deterministic frame pairs from the baseline decoder.
   Build component maps using the existing sensitivity direction:
   - PoseNet VJP for `sqrt(10 * pose_dist)` with the clamp/epsilon recorded.
   - SegNet VJP or finite-difference proxy for the official disagreement term.
   - Boundary multipliers from mask boundaries for SegNet, never as a
     standalone tensor ranking.
   - Separate `posenet`, `segnet`, and `combined` maps.

4. Pullback without full J.
   Never materialize `J_F`. For each frame-pair microbatch, compute a scalar
   weighted component objective and run one backward pass. Accumulate
   `grad(theta)^2` into per-parameter, per-channel, or per-tensor buffers.
   This estimates `diag(J^T W J)`. For candidate validation, use JVPs in the
   opposite direction: inject a proposed `Delta theta` and measure predicted
   pixel/component movement without storing `J`.

5. Reduction and calibration.
   Reduce per-weight buffers to per-tensor `I_t`, plus optional per-channel
   vectors matching `src/tac/sensitivity_map.py`. Calibrate the top/bottom
   tensors with signed finite differences through the closest available
   official scorer path. Promotion-grade maps must then be validated against
   byte-level perturbations through `archive.zip -> inflate.sh ->
   upstream/evaluate.py`; direct in-process scorer calls remain diagnostic.

6. Allocation.
   Build K/action curves as today, but feed `I_t` into the canonical allocator.
   A small code patch in this turn added:
   - `normalize_importance_weights`
   - `compute_jacobian_importance_weights`
   - `JacobianWeightedAllocator`
   in `src/tac/optimization/lagrangian_per_tensor_allocation.py`.
   These use the existing cost convention: higher weight means stronger
   protection from quantization error.

7. Packet build.
   Emit a byte-closed runtime packet. Record old/new archive SHA-256, old/new
   changed payload SHA-256, action choices, side-info bytes, deterministic
   rebuild proof, and no-op proof. No scorer loads at inflate time.

8. Exact evidence.
   Before dispatch, claim the lane with `tools/claim_lane_dispatch.py claim`.
   Then run canonical CUDA auth eval and adjudicate structured JSON. Only exact
   CUDA full-sample archive evidence can promote, rank, or retire a measured
   config.

## Complexity And Memory Strategy

Full `J_F` is impossible. For 1200 frames at `384 x 512 x 3`, the pixel vector
is roughly `7.08e8` channels. Against roughly `2.29e5` decoder symbols, a dense
Jacobian would have about `1.62e14` entries before scorer state.

Use these strategies:

- VJP accumulation: one weighted scalar objective per microbatch, one backward,
  accumulate `grad^2` into CPU tensors. Memory is model activations plus small
  accumulators, not `J`.
- Pair sharding: use the existing 600 pair plan. Emit shard manifests with
  pair indices, split seed, component, device, model/runtime SHA, and map SHA.
- Layer hooks: if per-weight storage is too large for experiments, reduce
  Conv2d gradients immediately to output-channel or tensor scalars.
- Hutchinson probes: for second-order response beyond `grad^2`, use random
  pixel/scorer probes with VJP or candidate perturbation JVP checks.
- Top-k finite differences: exact archive-level signed perturbation checks
  only for the tensors/actions that allocation would actually change.
- Cache curve bytes separately from sensitivity: K/action byte curves are CPU
  deterministic; score-pullback maps are CUDA artifacts with stricter custody.

## Comparison Against Existing Lanes

### Beta Fisher

Best current adjacent signal. It already matches second-order saliency and the
repo has `src/tac/sensitivity_map.py`,
`experiments/profile_component_sensitivity.py`, and
`experiments/build_sensitivity_map_pr106.py`. Weaknesses: current artifacts can
be direct scorer/Fisher proxies, not necessarily official archive-level
component response; per-channel reductions can hide within-channel risk; and a
flat Fisher tensor weight does not know whether an error moves a boundary,
texture patch, or pose-critical region. Jacobian weighting should consume beta
Fisher as a starting estimator, then calibrate it with component response.

### UNIWARD

Useful as a capacity prior, dangerous as the primary objective. UNIWARD says
textured/noisy regions can hide changes better than smooth edges. That
transfers only after the deterministic decoder maps weight perturbations to
those pixels. A high-variance tensor is not automatically safe if the decoder
routes it to lane boundaries or pose features. The safe formula is
`w_t = score_importance_t / texture_capacity_t`, not `1 / variance_t` alone.

### Boundary-Mass Weighting

Boundary mass is plausible for SegNet because boundary metrics are more
sensitive to boundary errors than mask IoU-style averages. But a boundary map
in pixel space is not an allocator until it is pulled back through `F`. It also
does not price PoseNet. Use boundary mass as a multiplier in `W_x`, not as a
standalone whitelist or blacklist.

### Film-Grain Re-Injection

Treat as suspicious. It is contest-valid only if deterministic, fully
archive-accounted, and consumed by the scored inflate path. Random grain,
external state, uncharged sidecars, nondeterministic PRNG, or post-eval visual
claims are noncompliant. AV1 is a useful design precedent because grain
parameters and seeds are signaled in the bitstream and the decoder process is
specified; that still does not prove this contest scorer likes grain. In this
repo, film grain should be a byte-closed residual candidate, not a substitute
for score-pullback allocation.

### Lossy Coarsening / Lagrangian Allocation

Mechanically strong, empirically chastened. The byte curves and lambda
bisection are the right shell, but the distortion term was wrong. The exact
negative on `budget=0.05` falsifies that measured config as a score
improvement; it does not kill coarsening. Reactivation should require
score-aware per-tensor weights, smaller budget ladder candidates, and exact
CUDA for each byte-closed archive.

## Repo Integration Points

- `src/tac/optimization/lagrangian_per_tensor_allocation.py`: canonical
  lambda-bisection allocator. This memo's small patch adds
  `JacobianWeightedAllocator` for externally computed pullback weights.
- `src/tac/sensitivity_map.py`: existing per-channel map contract, CUDA
  device gate, certification checks, and metadata safety.
- `experiments/profile_component_sensitivity.py`: closest CUDA component
  sensitivity producer; should become the shard runner for VJP pullback maps.
- `experiments/build_sensitivity_map_pr106.py`: beta-Fisher PR106 producer;
  current canonical pair iterator blocker must be resolved before promotion.
- `src/tac/uniward_delta.py`: pixel-space UNIWARD and detector-cost manifest
  concepts. Reuse only as a texture/capacity prior and compliance warning
  precedent.
- `tools/pr101_lossy_coarsening_analytical.py`: CPU byte/curve substrate.
- `tools/pr101_omega_opt_uniward_weighted_allocation.py`: existing UNIWARD
  weighting wrapper; should be superseded or extended to scorer-pullback
  weights.
- `tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py`:
  PR106 cross-substrate byte curves; should be wired to the new allocator.
- `tools/build_pr106_uniward_runtime_packet.py`: runtime packet builder for
  exact eval after protected-file hardening finishes; do not edit casually.

## Tests And Gates Needed

Already added in this turn:

- Unit tests for normalized Jacobian importance weights, texture capacity
  division, all-zero rejection, and high-importance tensor protection:
  `src/tac/tests/test_optimization_lagrangian_per_tensor_allocation.py`.

Next required gates:

- Toy analytic decoder test: small linear/conv decoder with known `J`; compare
  VJP-accumulated per-tensor weights to closed-form `diag(J^T W J)`.
- Finite-difference sanity: for top-k tensors, perturb signed K/action deltas
  and require predicted component ordering to match observed CUDA scorer
  response within a declared tolerance.
- Artifact validation: pullback map must record component, pair split,
  source archive bytes/SHA, runtime tree SHA, device CUDA, scorer commit, map
  SHA, reduction rule, and calibration/holdout statistics.
- Fail-closed map validation: reject CPU/MPS, all-zero, uniform, missing tensor
  entries in promotion mode, stale source archive SHA, or missing component
  calibration.
- No-op proof: candidate packet must prove charged payload bytes changed when
  claiming an optimize mode, and identity/canonicalize modes must not become
  score claims.
- Deterministic rebuild: packet builder rebuilds archive byte-identically.
- Exact evidence gate: pre-submission compliance, dispatch claim, CUDA auth
  eval, structured adjudication, and result-review packet before status change.

## Risks And Counterexamples

- Linearization breaks. SegNet argmax and pose geometry can be discontinuous;
  a small weight perturbation can cross a boundary. Mitigation: signed
  finite-difference checks for selected actions and conservative trust regions.
- Gradient masking. If the scorer has flat or saturated regions, VJP can be
  near zero while finite differences are large. Mitigation: combine VJP with
  finite-difference response curves and holdout stability.
- Pose square-root singularity. Near-zero pose distortion makes
  `sqrt(10 * pose)` locally steep. Clamp/epsilon choice must be recorded and
  calibrated against official component fields.
- Per-tensor scalar loses structure. A tensor can contain both safe and unsafe
  channels. Mitigation: promote per-channel allocation where byte format
  supports it; tensor scalar is phase 1 only.
- Byte interactions are nonseparable. Brotli/range coding couples tensors.
  Mitigation: use joint encoder hooks in the allocator and memoized selection
  vectors; do not trust sum-of-per-tensor bytes when joint bytes are available.
- UNIWARD can point at scorer-sensitive texture. Texture capacity is subordinate
  to score pullback; never allow variance to override high component
  sensitivity.
- Film grain can be visual but score-negative or noncompliant. It must be
  deterministic, archive-accounted, and exact-evaluated as a candidate.
- Archive path drift. Direct scorer/in-process renderer maps can disagree with
  `archive.zip -> inflate.sh -> upstream/evaluate.py`. Promotion requires the
  latter.

## Dispatch And Evidence Requirements

Minimum candidate packet before CUDA spend:

- baseline archive bytes/SHA and exact component fields;
- candidate archive bytes/SHA and member SHA;
- changed payload proof and side-info byte accounting;
- pullback map SHA plus source archive/runtime/scorer bindings;
- allocation manifest with `I_t`, optional `C_t`, final weights, lambda,
  action per tensor, byte curve source, and expected breakeven;
- deterministic rebuild SHA proof;
- `score_claim=false`, `promotion_eligible=false`, and
  `rank_or_kill_eligible=false` until exact CUDA returns;
- active dispatch claim with lane/job ID.

Promotion packet after CUDA:

- full-sample exact CUDA `contest_auth_eval.json`;
- component recomputation from structured fields;
- archive byte term recomputation;
- runtime tree custody;
- adversarial classification: legitimate movement, measured-config negative,
  harness/archive bug, proxy leakage, timeout, or indeterminate;
- scoped reactivation or retirement criteria.

## Recommended Next Implementation Steps

1. Build `experiments/build_jacobian_importance_pr106.py` as a CUDA-only VJP
   producer that emits per-tensor and per-channel pullback maps over the PR106
   pair plan, initially diagnostic-only.
2. Add a toy-decoder closed-form Jacobian test and a top-k finite-difference
   calibration test before any real archive dispatch.
3. Wire PR106 K/action curves into `JacobianWeightedAllocator`, generate a
   small `rms_target` ladder (`0.01`, `0.02`, `0.03`), and dispatch only the
   first byte-closed candidate whose breakeven math can survive the exact
   lossy-coarsening negative.
