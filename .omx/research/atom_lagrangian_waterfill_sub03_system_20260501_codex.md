# Atom Lagrangian Water-Fill System - 2026-05-01

Evidence grade: `derivation` plus `external` plus exact local artifacts where
explicitly cited. This is a control-plane design note, not a score claim.

## Objective

The contest score is:

```text
S(A) = 100 * seg(A)
     + sqrt(10 * pose(A))
     + 25 * bytes(A) / 37545489
```

For a current archive `A` and a finite set of candidate atoms `a`, choose a
set `X` that minimizes:

```text
J(X) = S(A + X) + R(X)
```

where `R(X)` is the reproducibility/compliance risk penalty. In the exact
contest loop, `S(A + X)` is only authoritative after CUDA auth eval on the
complete archive bytes.

For planning before exact eval, use a conservative marginal model:

```text
DeltaS_hat(a | X)
  = 100 * DeltaSeg_hat(a | X)
  + [sqrt(10 * pose_after_hat) - sqrt(10 * pose_before_hat)]
  + 25 * bytes(a | X) / 37545489
  + rho(a | X)
```

Accept atom `a` into a build candidate only when:

```text
E[-DeltaS_hat(a | X)] / bytes(a | X)
  > 25 / 37545489 + uncertainty_penalty(a) + interaction_penalty(a | X)
```

This is the water-fill condition: spend bytes first where the expected
component-score reduction per charged byte exceeds the rate slope.

## Atom Types

The allocator should treat all of these as first-class atoms:

- Archive atoms: member names, single-member packing, header bytes, entropy
  coding, deterministic timestamp/permission/layout.
- Pose atoms: velocity, log-zoom, yaw/pitch/roll residuals, pair-local pose
  perturbations, qpose quantization grid, per-pair deltas.
- Mask atoms: frame, pair, class, connected component, boundary band, lane
  region, horizon band, FoE neighborhood, soft-LUT grayscale bin.
- Renderer atoms: channel, layer, tensor block, FP4 group, low-rank adapter,
  per-pair FiLM vector, latent canvas cell.
- Video geometry atoms: native-camera foveation, radial zoom, ego-motion,
  vanishing point, horizon/lane lines, temporal motion basin.
- Training atoms: loss weights, pair sampler, scorer-path rounding, decoder
  architecture, quantization schedule.

An atom record should carry:

```text
atom_id
stream
scope
charged_bytes
archive_member_delta
source_signal
expected_seg_delta
expected_pose_delta
expected_rate_delta
uncertainty
interaction_keys
promotion_allowed=false until exact CUDA
```

## Hard Pair Identification

A hard pair is not simply high error. It is high repair opportunity density:

```text
hardness_i = E[score saved by repairing pair i] / bytes_to_repair_i
```

The signal hierarchy is:

1. Exact finite-difference CUDA response on a complete archive.
2. Pair-local PoseNet/SegNet deltas against the current frontier.
3. Influence estimates from differentiable scorer-path proxies, calibrated
   against exact response.
4. Priors from motion, lane geometry, hard-pair history, and top submissions.

For pose-sensitive pairs, split any winning pair atom into:

```text
pair -> frame -> pose_dim -> class/region -> residual tile -> entropy atom
```

Then solve a small knapsack or beam search under:

```text
bytes <= B
seg <= seg_gate
pose <= pose_gate
inflate_time <= 30 minutes
archive_compliant = true
```

## Manifold And Differential View

Treat archive construction as a trajectory on a constrained manifold:

```text
dA/dt = u_bytes + u_geometry + u_decoder + u_pose + u_training
```

with projections:

```text
P_rate(A): bytes(A) <= B
P_seg(A): seg(A) <= tau_seg
P_pose(A): pose(A) <= tau_pose
P_runtime(A): inflate(A) <= 30 minutes
P_compliance(A): archive closure and deterministic custody
```

Dykstra/ADMM language is useful operationally:

```text
A_{k+1} = P_compliance P_runtime P_pose P_seg P_rate (A_k + atoms_k)
```

but it is not proof of nonconvex composition. Every promising projected point
must become its own archive and pass exact CUDA.

## Top Submission Reverse Engineering

Live public leaderboard on 2026-05-01 shows the floor to beat:

```text
qpose14          0.32
unified_brotli   0.33
quantizr         0.33
fp4_mask_gen     0.37
selfcomp         0.38
```

Public branch-code anatomy:

- Top entries use one semantic control stream plus a tiny charged renderer.
- `qpose14` and `unified_brotli` use single member `p`.
- `qpose14` keeps all pose dimensions on a quantized integer grid.
- `unified_brotli` drops to velocity-only pose for its learned renderer.
- `quantizr` trains through scorer-like uint8, resize, SegNet, and PoseNet
  losses.
- `selfcomp` proves grayscale can work only when train and inflate both use
  the same soft Gaussian LUT distribution.

Local exact evidence so far:

- Single-member p/posecd lossless packing works as a pure rate atom.
- Velocity-only C-044 pose packing collapses PoseNet and is not transferable
  without the learned renderer geometry.
- CRF grayscale plus sparse repair remains in the wrong geometry basin.

## Camera, Ego-Motion, And Foveation

The native contest video is 1164x874 at 20 fps. Local video forensics record
intrinsics approximately:

```text
fx = fy = 910
cx = 582
cy = 437
scorer VP approx = (256, 174)
native VP approx = (582, 396.03125)
```

Use these as priors, not claims. Candidate atoms:

- FoE-centered radial foveation with identity trust-region parameterization.
- Horizon-band preservation around scorer y in `[155, 195]`.
- Lane-line/log-zoom atoms for hard pose pairs.
- Ego-motion-predicted pair conditioning for frame 1 renderer FiLM.

The foveation transform should be parameterized as:

```text
x' = x + alpha * r(x)^2 * (x - c)
y' = y + beta  * r(x)^2 * (y - c)
```

with `alpha=beta=0` as identity and tiny charged side information. Exact CUDA
component response decides whether this manifold is safe.

## Current Decision

Under the Lane 12 retraining gate, the highest-EV legal work is:

1. Treat C-051 as the active exact frontier anchor:
   `0.9867772369277311`, `594047` bytes, SHA-256
   `dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f`.
2. Keep qpose-grid pose perturbation as an active rate/geometry atom family,
   because the RP2 qpose14-style packet beat the lossless posecd RP2 packet
   by `0.0002096727060869` exact T4 score and `64` archive bytes.
3. Do not spend more exact eval on CRF grayscale sparse repair unless the base
   representation changes to a trained soft-LUT/qpose/Quantizr-style decoder.
4. Use the canonical PR #67/#65 reproduction artifact to turn public packer
   tricks into local atoms: QZS3 grouped FP4/QV model packing, QP1/qpose
   delta-VLQ pose coding, single-member blob layout, and PR #65 residual
   postprocess atoms.
5. Prepare the qpose/Quantizr/Selfcomp learned renderer lane so it can launch
   immediately when Lane 12 L2 clearance exists.

The sub-0.3 path is therefore not one clever byte shave. It is a coupled
solution:

```text
one-mask-per-pair semantic control
+ learned tiny renderer
+ pose/log-zoom/FoE conditioning
+ scorer-path training
+ atomized byte allocation
+ exact CUDA archive promotion
```

## Canonical Reproduction Hooks - 2026-05-01T18:50Z

The public-leaderboard floor and branch-code anatomy now have a deterministic
local reproducer:

```bash
.venv/bin/python experiments/reverse_engineer_top_submissions.py \
  --output-dir experiments/results/top_submission_reverse_roundtrip_20260501 \
  --work-dir /tmp/pact_topsubs_repro_20260501
```

Current canonical artifact:

```text
experiments/results/top_submission_reverse_roundtrip_20260501/archive_anatomy.json
schema_version=2
score_claim=false
evidence_grade=external_plus_empirical_byte_anatomy
```

PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` is the packer target:

```text
archive_bytes=276564
archive_sha256=a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765
single_member=p
mask_obu_br=219472 compressed bytes
model_qzs3_br=56093 compressed bytes
pose_qp1_br=899 compressed bytes
QZS3 validation: 111 keys, 87836 params, finite
```

This is not a local score claim. It is a deterministic external-plus-empirical
atom source. A local archive only promotes after its own exact CUDA auth eval.

## New Atom Evidence - 2026-05-01T19:55Z

QP1 velocity-only as an OWV3/C-044 atom:

```text
archive_bytes=588562
archive_sha256=2a080314233011b0f82d20cec304d4931eb9d105d8063cb7c56f1b0a1b11b8b9
score=2.2175001665333225
posenet=0.2441081
segnet=0.00263205
evidence=A-negative scoped forensic, L40S CUDA
```

Interpretation:

- Rate benefit and SegNet benefit are real.
- PoseNet cost is catastrophic for the current OWV3 geometry.
- Therefore the Lagrangian sign for QP1-on-OWV3 is negative:

```text
Delta L ~= +1.21996 score points versus C-044
```

This narrows the atom, not the family. QP1 remains active for learned
JointFrameGenerator/qpose renderers where the model has been trained with that
pose contract.

Q-FAITHFUL packer atom:

```text
raw_QFAI            563206 bytes
QZS3                284486 bytes
QZS3 + RP2 qpose14  276651 bytes
QZS3 + RP2 QP1      273048 bytes
```

The packer atom has already reached public-top-submission byte scale on an
early snapshot. Its score term alone is:

```text
25 * 276651 / 37545489 = 0.184211...
```

The sub-0.3 requirement for that archive is therefore approximately:

```text
100 * seg_dist + sqrt(10 * pose_dist) < 0.115789
```

For a qpose14-like target with PoseNet around `0.00052`, the pose term is
approximately `0.0722`, leaving SegNet below about `0.000436`. For the
published qpose14 SegNet `0.00061261`, the non-rate term is about `0.1335`,
and total score sits near `0.3177`. The mathematical route below `0.3` is
therefore not primarily smaller zip overhead; it is a coupled reduction in
SegNet/PoseNet through scorer-aligned training, hard-pair allocation, and
possibly foveated/ego-motion atoms while preserving the `~276 KB` byte scale.

Queued CUDA checkpoint:

```text
exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z
```

Decision rule after harvest:

- If score is already in or near the public basin, prioritize checkpoint
  cadence and exact-eval every phase gate.
- If score is bad but components are finite and plausible, continue training
  and use the result as a phase-1 baseline.
- If runtime/decoder fails, fix the source-closure or archive contract before
  spending another training hour.

## Exact Per-Sample Trace Formalization - 2026-05-01T21:16Z

The hard-pair problem is now canonicalized as an exact component trace plus
atom-allocation problem.

Let archive candidate `A` have exact per-pair component distances:

```text
p_i(A) = PoseNet distance for contest pair i
s_i(A) = SegNet distance for contest pair i
n = 600
P(A) = (1/n) * sum_i p_i(A)
S(A) = (1/n) * sum_i s_i(A)
B(A) = archive bytes

score(A) = 100*S(A) + sqrt(10*P(A)) + 25*B(A)/37,545,489
```

For a repair atom `a` with charged byte cost `c_a`, exact marginal value is
only known after a CUDA archive eval of `A + a`. Before that exact checkpoint,
the allocator uses first-order ranking:

```text
d score / d s_i = 100 / n
d score / d p_i = 5 / (n * sqrt(10*P(A)))

estimated_value(a | A) =
  sum_{i in support(a)} [
    (100/n) * expected_delta_s_i
    + (5/(n*sqrt(10*P(A)))) * expected_delta_p_i
  ]

rate_cost(a) = 25*c_a/37,545,489
```

An atom is positive-EV only if:

```text
estimated_value(a | A) - rate_cost(a) - risk_penalty(a) > 0
```

This is a Lagrangian water-fill over charged side information, not a raw
hard-pair sort. The pair with the highest distortion is not necessarily the
best repair target; the best target is the highest confidence-adjusted
`score_saved_per_byte`.

The exact trace job queued for C-051 is:

```text
exact_eval_component_trace_c051_l40s_20260501T2116Z
archive_sha256=dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f
archive_bytes=594047
state=.omx/state/component_trace_c051_l40s_batch_jobs_20260501T2116Z.json
```

Once harvested, `component_trace.json` becomes the canonical per-pair `p_i`
and `s_i` source for:

1. C-051 hard-pair diagnosis.
2. Alpha AMR1 sparse repair atom ranking.
3. Candidate-vs-baseline excess ranking once a lossy Alpha/Q-FAITHFUL trace is
   available.
4. Calibration targets for learned atom selectors.

## Planner Consumption Contract - 2026-05-01T21:21Z

The mathematical trace source is now wired into the deterministic Alpha atom
planner.

Accepted trace contract:

```text
score_claim == false
evidence_grade == diagnostic_component_trace
n_samples == 600
pair_index set == {0, ..., 599}
contest_auth_eval_cross_check.all_match == true
```

For each pair `i`, the planner consumes:

```text
p_i = samples[i].posenet_dist
s_i = samples[i].segnet_dist
g_i = samples[i].score_combined_contribution_first_order
```

and ranks repair atoms by:

```text
score_signal_prior_per_byte(a)
  = sum_{i in support(a)} g_i / compressed_bytes(a)
```

before falling back to legacy formula priors. This is still a first-order
water-fill, not proof of additivity. The exact archive built from any selected
policy is the only score evidence.

New CLI knob:

```bash
experiments/alpha_repair_atom_planner.py \
  --pair-weights-meta <component_trace.json> \
  --pair-signal-top-k 100
```

The `pair_signal_top_k` set is only the hard-pair prior. Atom ordering also
uses the full 600-pair `g_i` vector, so atoms outside the top-k can still be
accounted for if they appear in a candidate residual partition.

Next mathematical refinement:

- Replace pair-only support with frame/class/component supports using the
  same `g_i` trace prior as the outer measure.
- Add a confidence shrinkage term:

```text
utility(a) =
  E[score_saved(a)] - lambda_rate*c_a - beta*uncertainty(a) - gamma*interaction_risk(a)
```

- Calibrate `uncertainty(a)` from candidate-vs-baseline trace deltas once the
  first lossy Alpha candidate trace exists.

## Exact Trace Vector Instantiated - 2026-05-01T21:39Z

The C-051 trace landed and is now the first exact CUDA-backed outer measure for
atom selection.

Trace artifact:

```text
component_trace_json=experiments/results/lightning_batch/exact_eval_component_trace_c051_l40s_20260501T2116Z/component_trace.json
component_trace_sha256=dabf29a76ac390c19a83cf77d2487b7ebc39c280b00fdba8c4984ea148fe98b6
n_samples=600
score_claim=false
evidence_grade=diagnostic_component_trace
contest_auth_eval_cross_check.all_match=true
```

Top combined first-order `g_i` atoms:

```text
i=127  g=0.018335940801282723  pose=0.37293943762779236  seg=0.011210124008357525
i=75   g=0.013577676763007545  pose=0.29224878549575806  seg=0.004038493148982525
i=109  g=0.006064904586401161  pose=0.08778192847967148  seg=0.013132731430232525
i=133  g=0.003986590169561404  pose=0.00045639160089194775 seg=0.0237986259162426
i=514  g=0.0038463283708715344 pose=0.03890116140246391  seg=0.0127716064453125
i=125  g=0.0034106770370180804 pose=0.04926970601081848  seg=0.007410685531795025
i=517  g=0.0024364844331000806 pose=0.00023410015273839235 seg=0.014556884765625
i=522  g=0.002324512367893982  pose=0.00030920878634788096 seg=0.013865153305232525
i=177  g=0.002177802265752412  pose=0.0009798650862649083  seg=0.01280721090734005
i=111  g=0.0021315114808380807 pose=0.01450280286371708  seg=0.00894673727452755
```

Planner outputs:

```text
crf63_plan=experiments/results/alpha_repair_atom_plan_c051_trace_crf63_pair_lzma_20260501/atom_plan.json
crf63_plan_sha256=b3493b4c11eab6c4386837b9666c47adb82da018313d0a89cbdd8782c23d72e8
crf62_plan=experiments/results/alpha_repair_atom_plan_c051_trace_crf62_pair_lzma_20260501/atom_plan.json
crf62_plan_sha256=dcaf241d104c12090db300feceb22d4c9941cffd01e7cb502b80f5005e41ffdf
```

Mathematical interpretation:

- Pair 127 and 75 are high-pose-leverage atoms and dominate the first-order
  current-frontier loss, but this trace describes the C-051 in-basin archive.
- The Alpha sparse-repair exact grid showed global out-of-basin PoseNet
  collapse, so C-051 `g_i` is not sufficient by itself to repair Alpha.
- The next rigorous allocator needs a lossy-candidate trace and should optimize
  excess contribution:

```text
g_i_excess(A) = g_i(A) - g_i(C-051)
utility(a | A) =
  sum_{i in support(a)} max(0, g_i_excess(A))
  - lambda_rate * charged_bytes(a)
  - beta * uncertainty(a)
  - gamma * interaction_risk(a)
```

This changes "hard pair" from a static label into a candidate-dependent
measure on the archive manifold.

## Full-Pipeline Atom Water-Fill Extension - 2026-05-02T02:55Z

The PR65/PR67 public-floor trace comparison is now consumed as a deterministic
planning table, not just prose. `experiments/build_frontier_atom_ledger.py`
emits `atom_allocation_table` with:

- measured reference-relative PR65/PR67 trace atoms for postprocess, pose, and
  mask families;
- planning-only full-pipeline opportunity atoms for mask grammar, renderer
  quantization, archive overhead, runtime simplification, and RL/bandit/
  multipass selection;
- synergy and antagonism flags for stack composition review;
- exact-eval stack gate recommendations for closed payload custody, canonical
  CUDA auth eval, component-antagonism review, runtime budget, and T4/equivalent
  promotion.

The opportunity atoms are not score evidence. They are a control-plane map for
which exact archive to build next. Any selected stack still needs its own
`archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA eval on identical
bytes before promotion, ranking, or retirement.

## Public-Floor Reference Measure and Lagrangian System - 2026-05-01T22:09Z

New artifact:

```text
comparison_json=experiments/results/component_trace_comparison_c051_vs_public_floor_20260501/trace_comparison.json
comparison_tool=experiments/compare_component_traces.py
comparison_evidence_grade=diagnostic_trace_comparison
```

The component-trace comparator formalizes the reference-relative version of the
repair problem. For archive `A` and reference `R`, define:

```text
S(A) = 100 * mean_i s_i(A)
     + sqrt(10 * mean_i p_i(A))
     + 25 * bytes(A) / U

U = 37,545,489
lambda_rate = 25 / U = 6.658257e-7 score/byte
```

For a pair `i`, the local differential score mass is:

```text
g_i(A) =
  100 * s_i(A) / n
  + [5 / sqrt(10 * mean_j p_j(A))] * p_i(A) / n
```

The reference-relative hard-pair field is:

```text
h_i(A | R) = max(0, g_i(A) - g_i(R))
```

This is the correct hard-pair definition for optimization. A pair is not hard
because it has large absolute distortion; it is hard when it has positive,
repairable excess score mass versus a stronger reference at acceptable byte
cost.

For repair atoms `a` with support over pairs, frames, classes, connected
components, tiles, latents, pose channels, or decoder parameters:

```text
utility(a | A, R) =
  E[sum_{i in support(a)} h_i(A | R)]
  - lambda_rate * charged_bytes(a)
  - beta * uncertainty(a)
  - gamma * interaction_risk(a)
  + eta * synergy(a | selected)
```

Active atoms in a continuous water-fill satisfy the KKT-style condition:

```text
-d E[distortion_score] / d bytes = lambda_rate
```

with separate nonlinear pose derivative:

```text
d sqrt(10 * mean p) / d p_i =
  5 / (n * sqrt(10 * mean p))
```

and with inequality constraints for:

```text
archive compliance, payload closure, deterministic inflate, CUDA device,
n_samples=600, component gates, runtime budget, source manifest custody
```

C-051 versus public PR63 decomposition:

```text
total_gap=0.6615832118515242
seg_gap=0.3414705165511501
rate_gap=0.20406845679916435
pose_gap=0.11604423850120972
archive_delta_bytes=306474
break_even_bytes_per_0.001_score=1501.81956
```

Top reference-relative excess fields:

```text
combined: 127, 75, 109, 133, 514, 125, 517, 522, 111, 45
pose:     127, 75, 109, 125, 514, 179, 378, 90, 111, 289
seg:      133, 517, 109, 522, 177, 514, 45, 127, 521, 516
```

Mathematical consequence:

- Pair atoms are a high-resolution allocation measure, not the full path to
  sub-0.3. The public-floor gap is mostly global mask/renderer geometry plus
  bytes.
- The next allocator must operate over nested atoms:
  `global topology -> video/mask latent -> pose scalar -> frame pair ->
  class/component/tile -> entropy code`.
- Synergy and antagonism must be measured, not assumed. A pair atom that helps
  C-051 can be useless or harmful on Alpha if the whole archive is out of the
  PoseNet basin.
- The most valuable future exact diagnostic is a component trace on an
  in-basin public-pose-basis Q checkpoint or on a geometry-regenerated Alpha
  candidate. That produces `h_i(A | PR63)` for the archive we actually need to
  repair.

## Q-FAITHFUL Out-of-Basin Counterexample - 2026-05-01T23:16Z

Artifact:

```text
comparison_json=experiments/results/component_trace_comparison_qfaithful_2146_qp1_vs_public_floor_20260501/trace_comparison.json
candidate=qfaithful_2146_pr64_qp1_l40s
reference=pr63_qpose14_public_floor
```

Component gap:

```text
score_delta_vs_pr63=21.74033131907131
pose_delta_vs_pr63=21.417086479367764
seg_delta_vs_pr63=0.3328798187552214
rate_delta_vs_pr63=-0.009634979051677844
archive_delta_bytes_vs_pr63=-14470
```

This is the canonical counterexample for naive atom repair:

- The archive is byte-better than PR63 by `14,470` bytes.
- The rate gain is only `0.00963` score points.
- PoseNet loses `21.417` score points.
- No sparse residual or pair patch can rationally repair this under the
  Lagrangian byte price unless it changes the global renderer/pose manifold.

The allocator must therefore classify candidate archives into two regimes:

```text
if pose_delta_vs_reference >> feasible_atom_budget:
  regime = global_manifold_redesign_required
else:
  regime = local_atom_water_fill
```

For QP1 21:46Z, the regime is `global_manifold_redesign_required`.

## Public-Floor PVL1 Lagrangian Constraint - 2026-05-01T23:36Z

The current public-floor candidate is not a learned-manifold jump; it is a
layout/pose-code atom inside the already-measured PR63 basin:

```text
A0 = public_pr63_qpose14
S(A0)=0.32518843312932477
B(A0)=287573
A1 = pr63_decoded_geometry + pr64_len_table + pvl1_pose
B(A1)=286960
delta_B=-613
```

If exact CUDA confirms:

```text
seg(A1)=seg(A0)
pose(A1)=pose(A0)
```

then:

```text
S(A1)-S(A0)=25*(B(A1)-B(A0))/37545489
            = -0.0004081811801578958
S(A1)=0.3247802519491669
```

This atom has near-zero interaction risk only if the decoded runtime members
are identical. Its hard constraints are:

```text
decoded_pose_sha(A1)=cc99e99c28b2ea686439b226ee504ba3a0d82fd8eb8550f4fed05d35ece5dc40
mask_sha(A1)=a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb
renderer_sha(A1)=d97849d15859ae013ec983de8c1e2f638e63f3876fef658a8b7781bcfaa16a5f
archive_sha(A1)=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
```

Grand Council C adversarial review found a tighter atom:

```text
A2 = public_pr64_mask_first_len_table + bare_public_pr64_velocity_delta_pose
```

Expected gain over PVL1 before Brotli interactions is `8` raw pose bytes,
because PVL1 is `1208` bytes while public PR64 bare velocity-delta pose is
`1200` bytes. The byte gain is small:

```text
25*8/37545489 = 0.000005326486528651984
```

but the public-layout parity gain is valuable. If A1 lands, A2 becomes the
next deterministic patch/eval because it reduces both byte cost and parser
branch distance from `unified_brotli`.

## Public-Floor Atom Update - 2026-05-01T23:59Z

Exact H100 evidence updated the atom table:

```text
atom                         byte_delta_vs_pr63   exact_score_H100         verdict
PVL1                         -613                 0.3246902093443082      keep, T4 promotion running
PVR1_top256                  -373                 0.3248499593443082      valid but dominated by PVL1
PR64_qpose14_repack          -449                 0.33108572602475084     pose-regressed
PR64_PVR1_top64              -562                 0.3310104760247508      pose-regressed
barevel_rendererfirst        -391                 19.534443931576337      implementation mismatch
```

Lagrangian conclusion:

```text
PVL1 dominates all currently measured public-floor pose/packer atoms.
```

The "bare public velocity" atom is not a live byte-saving atom until decoded
pose parity is repaired. Its current marginal distortion penalty is many orders
of magnitude above the byte benefit:

```text
pose_score_penalty ~= 19.21
byte_score_gain_vs_PVL1 ~= 25*(286960-287182)/37545489 = -0.0001478
```

Therefore the next atom search must use exact decoded-pose parity as a hard
constraint before archive CUDA spend:

```text
if decoded_pose_sha(candidate) != decoded_pose_sha(reference):
  run_pose_contract_forensics()
  do_not_submit_exact_eval_unless_the_experiment_is_explicitly forensic
```

## PVL1 Exact T4 Anchor And Next Atom Ranking - 2026-05-02T00:16Z

The active Lagrangian anchor is now exact T4, not H100 diagnostic:

```text
A* = public_floor_pvl1
S(A*) = 0.3247176275031171
B(A*) = 286960
seg(A*) = 0.00061261
pose(A*) = 0.00052391
archive_sha(A*) = 4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
hardware = Tesla T4
evidence = A++ contest T4
```

Marginal byte coefficient is unchanged:

```text
lambda_rate = 25 / 37545489 = 0.000000665858953875
```

Grand Council water-fill conclusion after exact PVL1:

```text
1. PR67/QZS3/QP1 packer parity has the highest byte EV. A PR67-like
   276564-byte archive with PVL1 components would score about 0.31780.

2. PR65/private-layout parity around 284425 bytes would score about 0.32303
   if components held.

3. Hard-pair/AMR1 repairs are currently dominated unless a repair atom costs
   under roughly 500-700 charged bytes for a near-perfect top-pair correction.

4. More PR64/PVR1 pose variants are dominated unless they reduce bytes below
   286960 or improve exact components.
```

Next atom table rows must compare against `A*`, not C-044/C-051.

## QZS3/QP1 Anchor And Pose Line-Search Atom Economics - 2026-05-02T01:08Z

The active A++ Lagrangian anchor moved from PVL1 to QZS3/QP1:

```text
A* = public_floor_qzs3_qp1_fixedslice
S(A*) = 0.3243472585872431
B(A*) = 276296
seg(A*) = 0.00061244
pose(A*) = 0.00062614
archive_sha(A*) = c5260473c26c4d4537d99d4a6a18b8ff0d9d1a901f6db17cd2208559e1010362
hardware = Tesla T4
evidence = A++ contest T4
```

QZS3/QP1 versus PVL1:

```text
delta_B = 276296 - 286960 = -10664
rate_gain = 25*10664/37545489 = 0.007100871908470986
observed_score_gain = 0.00037036891587399756
```

The byte atom is real, but most of the rate win is spent on PoseNet drift. The
line-search objective therefore targets pose recovery while keeping the same
QZS3/QP1 archive basin.

First line-search checkpoint:

```text
A_ls1 = qzs3_qp1_fixedslice_pose_line_search_r2
S_H100(A_ls1) = 0.32114254758178584
B(A_ls1) = 276427
archive_sha(A_ls1) = 8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
pose_H100(A_ls1) = 0.00057865
seg_H100(A_ls1) = 0.00061012
```

Candidate delta versus A++ QZS3/QP1, using H100 as diagnostic only:

```text
delta_B = +131
rate_penalty = 25*131/37545489 = 0.000087227522958424
diagnostic_score_gain = 0.3243472585872431 - 0.32114254758178584
                      = 0.0032047110054572525
benefit_to_rate_penalty_ratio ~= 36.74
```

This is a high-EV atom despite extra bytes because it repairs PoseNet enough to
overpay the charged side information. T4 confirmation is mandatory before it
can replace the anchor.

Active continuation objective:

```text
obj_start = 0.259544306
obj_radius3_pass2 = 0.257372331
obj_radius5_pass2 = 0.255006542
obj_radius8_pass1 = 0.254208609
```

Water-fill implication:

- Pose-stream atoms are currently the dominant marginal-benefit-per-byte
  target within the QZS3/QP1 public-floor basin.
- Continue coordinate/radius line-search while objective improves, but every
  checkpoint becomes score evidence only after exact archive CUDA eval.
- Once pose improvement saturates, next atoms should be selected by exact
  pair/component traces from C-053 and the confirmed line-search T4 result.

## C-054 Exact T4 Anchor And Differentiable Proposal Atoms - 2026-05-02T01:24Z

The active A++ Lagrangian anchor is now the first QZS3/QP1 pose line-search
checkpoint:

```text
A* = qzs3_qp1_pose_line_search_t4_r1
S(A*) = 0.3218613619571356
B(A*) = 276427
seg(A*) = 0.00061244
pose(A*) = 0.00058608
archive_sha(A*) = 8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
hardware = Tesla T4
evidence = A++ contest T4
```

Measured atom economics versus C-053:

```text
delta_B = 276427 - 276296 = +131
rate_penalty = 25*131/37545489 = 0.000087227522958424
observed_score_gain = 0.3243472585872431 - 0.3218613619571356
                    = 0.002485896630107509
benefit_to_rate_penalty_ratio ~= 28.50
```

The exact result confirms that charged pose-stream atoms dominate the next
water-fill layer inside the public-floor QZS3/QP1 basin. The r8 continuation
checkpoint remains diagnostic-only until T4 promotion:

```text
S_H100(A_r8) = 0.3152653422017416
B(A_r8) = 276426
archive_sha(A_r8) = c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
```

New proposal atom families, still non-score until exact eval:

```text
directional_delta_atom = sparse/asymmetric integer deltas over QP1 col0
gradient_delta_atom = sign(-d PoseNetLoss / d col0) * magnitude_set
backtrack_guard = small opposite-direction deltas to hedge soft-gradient error
acceptance_rule = exact rounded archive objective improvement before archive build
promotion_rule = exact CUDA archive eval on final bytes
```

This turns "different radii by direction/vector/manifold" into a concrete
Lagrangian primitive: the atom proposal distribution is differentiable and
scorer-aligned, while the accepted atom remains a charged integer payload.

## Thesis-Review Consolidation - 2026-05-02T02:00Z

The latest writeup/study-material review is now separated into:

```text
.omx/research/meta_lagrangian_scientific_rigor_thesis_review_20260502_codex.md
```

It formalizes the same atom water-fill system as a paper-ready method:

```text
Score(A) = 100*Sg(A) + sqrt(10*P(A)) + 25*B(A)/37,545,489
lambda_rate = 25/37,545,489

utility(a | A, X) =
  E[component_score_saved(a | A, X)]
  - lambda_rate*charged_bytes(a | A, X)
  - beta*uncertainty(a | A, X)
  - gamma*interaction_risk(a | A, X)
  + eta*synergy(a | X)
```

The current paper frontier is the r8 QZS3/QP1 T4 archive:

```text
score=0.3159064496962538
bytes=276426
sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
evidence=A++ exact T4
```

The r13 T4 run remains unclaimed unless terminal local artifacts show otherwise.
Sub-`0.30` remains a hypothesis, not a result. The advisor-grade wording is
that scalar pose line search has produced the current frontier, but likely
needs PR65-style postprocess/side-channel atoms or an equivalent
component-improving atom family to cross `0.30`.

Public source links for this wording: the
[challenge repository](https://github.com/commaai/comma_video_compression_challenge),
[PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67),
and [PR #65](https://github.com/commaai/comma_video_compression_challenge/pull/65).

Paper/custody rules:

1. Public PRs are external signals only, even when their architecture matches
   the local basin.
2. H100/L40S diagnostics may guide atom proposals but cannot replace T4 A++
   wording.
3. KKT/waterline language is a planning relaxation, not a proof that discrete
   archive atoms compose.
4. Every accepted atom set must be rebuilt as a complete deterministic archive,
   copied into custody, hashed, scored through exact CUDA, and reviewed before
   it can enter a result table.
5. Negative results retire the measured implementation/config only unless
   independent exact evidence or a mathematical impossibility proof supports a
   broader claim.

## Supersession Note - 2026-05-02T04:25Z C-059 Anchor And Active Atom Loop

The latest exact anchor is now C-059, not the earlier C-051/r8/r13 anchors
described in historical sections above:

```text
score            0.3157055307844823
archive bytes    276347
archive SHA-256  cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab
evidence grade   A++ exact Lightning T4
```

The active water-fill primitive is the scorer-weighted pose atom policy emitted
by `experiments/plan_scorer_weighted_pose_atoms.py`:

```text
policy       c_059_pose_atoms_top032
output       experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json
expected     formula-only net utility 0.00042796260663844623
actuator     pair-window QP1 velocity line search
promotion    false until complete archive exact CUDA eval and T4 confirmation
```

The immediate mathematical lesson is that low-dimensional subspaces must be
selected by the current archive's actual actuator. Raw PVR1 residual top-K was
byte-regressive because decoded QP1 non-velocity columns are currently zero;
the same ranked pair atoms are being tested through velocity line search
instead.

## Terminology - 2026-05-02 Yousfi-Fridrich Floor

Internal shorthand: `Yousfi-Fridrich floor`.

Meaning: the contest-task MDL/rate-distortion floor that can sit below generic
human-video Shannon intuition because the objective is not perceptual video
reconstruction. The archive only has to preserve sufficient statistics for:

```text
100 * SegNetDistance + sqrt(10 * PoseNetDistance)
```

while charging every representation, decoder, postfilter, latent, pose stream,
mask grammar, and entropy-code bit through:

```text
25 * archive_bytes / 37,545,489
```

This floor is still contest-faithful. It does not permit sidecars, scorer
patches, host-local files, or hidden dependencies. It says the correct
mathematical object is a charged sufficient-statistic program for the fixed
video and fixed evaluator, not a universal human-viewable codec.

Operational consequence:

- Known pixel residuals between a candidate mask grammar and the source mask
  tensor become repair atoms.
- Component traces convert pairs/classes/regions into approximate
  `d score / d distortion` weights.
- The allocator accepts only atoms whose expected component-score reduction
  beats `lambda_rate = 25 / 37,545,489` after uncertainty and interaction
  penalties.
- Learned postfilters, GAN-style refiners, differentiable decoders, and
  inverse-steg style payloads are in scope only when their complete runtime and
  weights are charged inside `archive.zip` and the exact CUDA eval validates
  the final archive bytes.

This is a planning term, not a result claim. No archive is said to reach this
floor without exact T4/equivalent CUDA evidence, component traces, custody, and
adversarial review.
