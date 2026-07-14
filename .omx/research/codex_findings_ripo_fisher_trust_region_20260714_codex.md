# Codex findings: RIPO categorical Fisher trust region for the V9 witness head

**UTC date:** 2026-07-14  
**lane:** `lane_ripo_fisher_isometric_trust_region_500_20260714`  
**task:** `500_ripo_fisher_isometric_trust_region_20260714`  
**research_only:** `true`  
**status:** `BUILT_CORE_NOT_V9_LIVE_WIRED; FRESH_SEQUENTIAL_N600_CAPTURE_COMPLETE; RAW_SUM_CANDIDATE_V1_FALSIFIED; MEAN_LOSS_CANDIDATE_V3_SOURCE_TOCTOU_BLOCKED; ITERATIVE_V4_BUILDING; GENUINE_PULLBACK_BLOCKED`  
**authority ceiling:** `[macOS-CPU advisory]`; no contest score authority  
**pointer:** `UNCHANGED`; no archive-closed exact CPU/CUDA row

## Result first

The intake transfer `||Delta logit|| <= sqrt(delta/p1)` is falsified for a five-class softmax.
RIPO bounds one action's probability **ratio**.  The correct local categorical logit geometry is
directional:

```text
F(p) = diag(p) - p p^T
q_F(u;p) = u^T F(p)u
D_KL(p || softmax(log p + u))
    = log(sum_k p_k exp(u_k)) - p^T u
alpha_quad = min(1, sqrt(2 delta_KL / q_F(v;p))).
```

For symmetric winner-rival margin motion `u=t(e_w-e_r)/2`,

```text
C_wr = p_w + p_r - (p_w-p_r)^2
q_F = t^2 C_wr/4
|t| <= sqrt(8 delta_KL/C_wr).
```

This depends on both top probabilities and tail mass.  It gives a *wider* absolute Fisher radius
to the confident-interior counterexample than to the near-tie annulus counterexample.  Annulus
pixels are easier to flip because their required margin displacement is smaller, not because the
categorical Fisher ball is wider.

The exact witness-head metric is also not output-space `F` alone:

```text
G_head = sum_pixel J_pixel^T F(p_pixel) J_pixel,
J_pixel = d SegNet_logits(after actual R) / d theta_out_sdf.
```

Any pixel-aligned clip followed by a least-squares projection onto the global 485-parameter
`out_sdf` head is therefore named `cross_space_pixel_aligned_reprojection_v1`.  It can support an
INSTANCE/FORMULATION verdict, never a Fisher/KL trust-region FAMILY verdict.

## Paper-methodology check

The full RIPO v1 methodology/proposition text was read, not just the abstract.  RIPO Eq. 7 uses
the quadratic probability-ratio form.  Eq. 10 gives
`|r_a-1| <= sqrt(2 delta_10/p_a)`; Eq. 11 absorbs the factor two.  Proposition 4.1 concerns the
importance-weighted policy-gradient variance at that action-ratio boundary.  It does not prove
homoscedastic witness-head updates.  The retrieved v1 contains no separate appendix; the
derivation and proposition are in the main paper.

## Delta provenance

The paper's example/default delta is not transferable.  For a measured desired correction,

```text
delta_KL,flip_i = D_KL(p_i || softmax(z_i + t_flip_i d_i)).
```

The registered sweep uses quantiles of desired-correction thresholds and protected-pixel spill
thresholds observed at the sealed EMA after actual `R` and frozen CPU-Torch SegNet.  Its honest
quantity name is `cross_space_output_tie_kl_budget`: the tie threshold is analytic in the frozen
scorer probability vector, not an actual perturb-and-replay measurement of a witness-head update.
It cannot be promoted to a receiver/head flip budget until a parameter perturbation or declared
secant/JVP is replayed through `R`.
Eikonal epsilon is an SDF spatial-gradient scale and cannot set a scorer-logit KL budget without a
measured Jacobian.  The measured `1.2731082 bytes/net flip` is score economics only:

```text
100/(600*384*512) = 8.4771050e-7 score per Seg pixel
25/37,545,489     = 6.6585895e-7 score per archive byte
ratio             = 1.2731082 bytes per realized net flip.
```

## Built implementation and round-1 review

The isolated implementation is:

- `src/tac/optimization/ripo_fisher_trust_region.py`: NumPy-fp32 authority with float64
  decisions; local directional, exact finite-KL, local Euclidean-ball, and uniform-L2 modes.
- `src/tac/optimization/ripo_fisher_trust_region_mlx.py`: lazy MLX parity surface with a
  non-lowerable `0.9997` floor plus absolute/relative, direction, realized-alpha, and constraint
  gates.  Only `local_directional` is recomputed natively in MLX; exact-KL, uniform, and Euclidean
  modes consume the NumPy authority output and are not claimed as native MLX algorithms.
- `tools/probe_ripo_fisher_trust_region_saved.py`: header-only, machine-custodied saved-array
  preflight.  Monolithic full-grid NPZ is refused before materialization because it is not a
  bounded/resumable execution format.

Round 1 initially found and then extincted these bug classes:

1. cancellation erased positive KL for tiny non-null steps;
2. `delta=0` admitted a non-null step through a tolerance;
3. arbitrary `(600,...,5)` random arrays and dummy hashes could masquerade as real n600;
4. Tier-A output algebra used the cross-space formulation ID;
5. start/end source custody had a TOCTOU gap;
6. correlation-only MLX parity could accept a scaled wrong answer; and
7. caller root tolerance could relax simplex validation.

Final combined focused verification currently reports `78 passed, 2 skipped`; the skips are only because
MLX is installed but this sandbox exposes no Metal device.  The adversarial tiny-step checks are:

| input | delta_KL | exact KL before | realized exact KL after | result |
|---|---:|---:|---:|---|
| `(1e-8,-1e-8,0,0,0)`, uniform K=5 | `0` | `2.0e-17` | `0` | zero update |
| `(1e-7,-1e-7,0,0,0)`, uniform K=5 | `1e-20` | `2.0e-15` | `9.99985e-21` | certified clipped update |

## Measured through-R calibration so far

Current source checkpoint:
`experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz`,
SHA-256 `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c`,
epoch 150, `self_orient=0`, `in_feat=80`, mutable head exactly `(5,96)+(5)=485`
parameters.  Two baseline substrates must not be conflated:

- the fp32 EMA training selector row is 4,107,576 errors / 117,964,800 pixels,
  `d_seg=0.03482035319010417`;
- the receiver-closed all-int8 deploy baseline measured twice by Task #336 is 3,970,488 errors,
  `d_seg=0.03365824381510417`, archive SHA-256
  `81a4c5163aa434f61489773a35862dd4b4a733219173c71e6eb8a6ef2b0613b7`.

The fixed-head capture uses the same exact receiver bytes as the second substrate, but round-1
execution falsified reuse of its scorer counts.  The current receiver rebuild reproduces the
Task #336 blob SHA, archive SHA `81a4c516...613b7`, and 63,664 archive bytes exactly.  Nevertheless,
the current sequential SegNet process gives 5,037 errors at pair 13 where Task #336's concurrent
SegNet/PoseNet process recorded 5,039.  Pairs 0--12 agreed.  Batch-16/one-thread also gives 5,037;
an older batch-16/four-thread n16 smoke gives 5,039.  Thus the discrete argmax row depends on the
declared scorer process/batch/thread substrate even when the archive bytes are identical.

The old 3,970,488 count is therefore historical custody, not the expected value of this new
process.  Capture schema v3 pins `sequential_segnet_batch4_threads1`, re-derives exact archive
bytes before scoring, and completed a fresh n600 row at **3,970,482 / 117,964,800**,
`d_seg=0.03365819295247396`.  The six-pixel difference from the historical row is scorer-axis
drift, not a candidate gain.  The ordered fresh pair-error vector is sealed by SHA-256
`71aca359445b6d2e17aa62820e06e0bf19374b3094a366490cc300deaff883a2`; its per-GT-class error
counts are `[2,894,327, 289,308, 748,515, 12,735, 25,597]`.  Candidate fit and evaluation must
independently replay that fresh vector before any arm is admitted.  This remains
`[macOS-CPU advisory]`; upstream's default batch 16 recomputes both GT and candidate and is a
separate axis.  The initial fp32/deploy mixed gate and the later stale-scorer-count gate were both
caught fail-closed before a candidate was constructed.

A fresh pair-0 NumPy deploy -> actual camera `R` -> frozen CPU-Torch SegNet pass reproduced 4,080
errors and measured:

| field | q10 | q25 | q50 | q90 | max |
|---|---:|---:|---:|---:|---:|
| exact correction tie-crossing KL (4,080 baseline errors) | 0.00189249 | 0.0124736 | 0.0597094 | 0.316477 | 1.53296 |
| local `0.5 q_F` for the same move | 0.00188891 | 0.0123452 | 0.0560950 | 0.208264 | 1.81739 |
| exact protected spill tie-crossing KL (192,528 correct pixels) | 0.563519 | 0.672358 | 0.688877 | 0.700866 | 0.717577 |

The local-vs-exact correction relative error is 5.81% at p50, 33.44% at p90, and 52.58% at
p99 on pair 0.  The full-n600 fixed-log-histogram estimates are correction tie-KL q25
`0.0169717`, q50 `0.0769687`, q95 `0.558821`, and protected-spill tie-KL q01 `0.0185720`.
The registered candidate deltas use conservative histogram-bin lower bounds: q10
`0.002570842480640978`, q25 `0.016886937079030943`, and q50 capped by protected q01 at
`0.01847929279424884`.  These are `[macOS-CPU advisory]` and derive the sweep; they do not yet
constitute a candidate verdict.  The control/Fisher evaluation table remains explicitly `OWED`.

The first real candidate fit smoke caught and extincted a geometry bug before fitting: the new
runner incorrectly required scorer-grid `(384,512,3)` bytes before `SegNet.preprocess_input`,
whereas actual `R` produces camera-grid `(874,1164,3)` uint8 bytes and delegates the bilinear
downsample to the frozen scorer.  The corrected runner imports canonical `CAMERA_HW`, refuses the
pre-downsampled surrogate, passed 37 capture/candidate tests, regenerated its source-bound
preflight, and completed a real four-pair frozen-SegNet fit checkpoint.  The capture producer and
its bound core source were not changed.

A second own-review pass then falsified candidate-fit v1 before it could produce a verdict.  The
capture deliberately stores the **sum** CE gradient over 117,964,800 pixels; v1 used that sum
directly as an unscaled parameter step (`||g_sum||=43,927,513.63`).  That makes the vanilla control
meaningless and forces clipped arms toward boundary saturation.  The resumable v1 fit was stopped
at 60/600 and preserved as a falsified instance; it carries no verdict.  Candidate v2 must divide
by the exact recorded CE sample count, state that its unit mean-loss step is a formulation choice
rather than an optimizer-equivalent update, and measure the *final reprojected + int8-deployed*
head's constraint violations.  Pre-reprojection energy alone cannot certify the realized head.

The same review found that v1's `uniform_l2_control` is a logit-L2 geometry control, not the
requested PPO-analog constant action-ratio clip.  V2 therefore owes a separate
`uniform_ratio_ppo` arm with `r_k(alpha)=q_k(alpha)/p_k` constrained uniformly and an explicit
epsilon convention; L2 remains separately labeled.  It also owes exact/local mismatch telemetry:
the saturated v1 prefix showed exact-KL/local-energy ratios of roughly 108--118x for the local
arms, so those arms are approximation probes, never exact-KL-certified trust regions.

RIPO Proposition 4.1 is variance of the importance-weighted policy-gradient estimator at the
ratio boundary.  Confidence-band variance of target-probability movement here is only
formulation-stability telemetry; it is not a reproduction of the proposition's homoscedasticity
result.

An independent round-1 review then found a separate convergence-scope gap.  Candidate v3 clips
and reprojects one frozen unit mean-CE diagnostic proposal; it is not a few-step optimizer loop,
and only the local approximation spans q10/q25/q50 while the exact-KL and uniform controls use
q25.  V3 can therefore measure one `INSTANCE x FORMULATION` response, but it cannot select an
optimal `(delta,g)` or claim convergence.  A new isolated iterative harness is required to
recompute the head CE direction at each fixed-trunk step and sweep exact-KL q10/q25/q50 against
the controls.  Its baseline-policy reuse must remain explicit unless current-arm SegNet
probabilities are genuinely refreshed.

The source-bound v3 n600 fit was then stopped fail-closed after 165/600 pairs when the concurrently
owned `src/tac/boundary_math/localized_basis_frames.py` changed during #502 remediation.  The
durable blocker is
`experiments/results/ripo_fisher_trust_region_500_20260714/candidates_mean_ce_v3_logratio/fit_blocker.json`
with code `INPUT_TOCTOU`.  The prefix is non-verdict and MUST NOT be resumed or combined across
the source boundary.  A fresh pair-zero run is permitted only after the owner publishes a stable
source hash and a new preflight binds it; weakening the source-closure guard is forbidden.

Round-1 dependency tracing also proved that merely deleting the changing localized file from the
guard would be unsafe.  The prior capture/candidate adapters import it before branch dispatch;
the byte-close loader executes one of its semantic helpers even for the polar checkpoint; and the
generic receiver embeds localized source at import time.  The checkpoint itself omits
`__cfg_basis_family`, so `polar_directional_fourier` is a legacy loader default, not a persisted
claim.  Candidate v4 therefore uses a new polar-only runtime contract that must: avoid every
localized/byte-close import; record the absent-key legacy-default resolution; hash the regenerated
polar feature matrix; recompute gradients plus correction/spill histograms; independently replay
all 600 baseline pair counts; and fail before materializing a head if any feature/pair/hist digest
differs from the sealed capture.  This is strict executed-branch isolation, not a source-guard
bypass.  It makes no receiver/archive claim.

The follow-on through-`R` evaluator remains a distinct owed stage.  It must consume only the
versioned v4 final-step raw/deployed head artifact, reuse the same polar-only runtime, score each
arm in sealed groups of four with one Torch thread, independently replay the baseline pair vector,
checkpoint every group, and report per-class/overall fixes/spills plus post-deploy budget audits.
Fit geometry or histogram equivalence alone is never a `d_seg` verdict.

The genuine receiver/head Fisher experiment is separately
`BLOCKED_NO_RECEIVER_PULLBACK`: exact uint8 R and dynamic int8 redeployment are discontinuous, and
there is not yet a declared fixed-scale-int8/round STE or finite-difference secant JVP/VJP with
adjoint and exact-forward secant checks.  Any current pixel-aligned table is therefore only the
named cross-space formulation, not `G_head = J^T F J` custody.

## Saved-custody preflight receipt

`experiments/results/ripo_fisher_trust_region_500_20260714/saved_array_preflight/receipt.json`
records `NO_VERDICT_DATA_CUSTODY`, `materialization_attempted=false`, and
`pair_sharded_streaming_implemented=false`.  This is a DATA-CUSTODY/BOUNDEDNESS blocker only.

## Held V9 DSL/equation/consumer specification

No hot DSL/canonical-equation/trainer file was edited.  The integration owner should add this
only after a favorable complete receipt and after the current provenance seal is stable:

```text
equation_id: categorical_fisher_logit_trust_region_v1
metric_id: argmax_native_vjp_fidelity_v1
state_receipt_schema: reachable_decision_geometry_fidelity.v1

LawRef delta_kl:
  source = exact full-n600 through-R correction/spill receipt
  value = selected registered correction quantile subject to registered spill gate
  no literal copied from RIPO

Lever factory:
  CategoricalFisherTrustRegionLever(
      equation=LawRef(categorical_fisher_logit_trust_region_v1),
      delta=LawRef(full_n600_selected_delta_kl),
      mode in {local_directional, exact_kl},
      formulation in {full_pullback_head, cross_space_pixel_aligned_reprojection_v1},
      measurement_receipt_sha256=<required non-null content hash>,
  )

consumer:
  the compiled seg-head update policy inside the sole V9 DSL path;
  clip the proposed UPDATE, never multiply or precondition the gradient direction;
  reject cross-space receipts when full_pullback_head is declared.
```

No argv flag is invented here.  The owner must extend the typed compiler and provenance bijection
first, then provide a compiled consumer receipt.  Until that exists the status is
`NOT_V9_LIVE_WIRED`.  The current end-to-end V9 seal also fails closed in the exclusive provenance
owner's scientific-declaration table; this lane must not weaken or edit that source-closure gate.

## Launch-ready ticket, conditional only

If and only if the complete n600 receipt beats vanilla and uniform controls with no unacceptable
spill, Pose is subsequently measured, and archive parse-back preserves the gain, open an
operator-GO ticket for a resumable head-only V9 integration run.  Required gates are: exact
receipt hash, source/checkpoint/SegNet/R hashes, per-class table, confidence-band dispersion,
only-head-changed proof, stage checkpoints, SSD preflight, receiver parse-back, Pose, archive
bytes, and then separate contest-CPU/CUDA exact rows.  No such launch is authorized by this memo.

## Verdict scope

- Scalar `sqrt(delta/p1)` transfer: **FALSIFIED FORMULATION**.
- Full-K output-space clip implementation: **BUILT + LOCAL-VERIFIED**.
- Pixel-aligned head projection v1: **FALSIFIED INSTANCE** due to raw-sum proposal scaling; no
  candidate verdict.
- One-step mean-loss proposal + realized post-projection audit v3: **NO VERDICT; source TOCTOU at
  165/600, fresh pair-zero remeasurement required**.
- Few-step exact q10/q25/q50 convergence loop: **BUILDING; not yet measured**.
- Full pullback/block Fisher and KL-proximal head family: **OPEN but current experiment blocked
  on a declared receiver STE/secant JVP/VJP**.
- V9 live wiring, Pose, archive bytes, contest score, pointer: **NOT MEASURED / UNCHANGED**.

## Stores consulted

- RIPO, *Beyond Euclidean Clipping*, arXiv:2607.10169, full v1 methodology/proposition text.
- `.omx/research/paper_warm_start_designs_recent_intake_20260714T160000Z.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` and
  `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`.
- `reports/latest.md`, `.omx/state/canonical_frontier_pointer.json`, lane/task/progress ledgers,
  latest sister findings/design/council/directive memos, and the live inbox through the latest
  checkpoint.
- Current EMA/run receipts, GT cache, frozen SegNet weights, actual-R code, and the canonical
  `argmax_native_vjp_fidelity_v1` contract.
