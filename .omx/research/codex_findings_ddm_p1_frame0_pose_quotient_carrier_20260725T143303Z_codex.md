# Codex findings — DDM P1 frame-0 PoseNet quotient carrier

UTC preregistration: `2026-07-25T14:33:03Z`

Status at preregistration: `PREREGISTERED_NOT_MEASURED`

Research-only: `true`  
Score claim: `false`  
Promotion eligible: `false`  
MAIN landing review required: `true`

## Authority and boundaries

- Delegated authority was read completely before action: 8,938 bytes,
  SHA-256 `a11dfd6795acb6f63e34f32f30578845511d4cb09fa2de55b630640eea55ef24`.
- Lane: `lane_ddm_p1_frame0_pose_quotient_carrier_20260725`, phase 1,
  `research_only=true`.
- PR130 and PR86 code, weights, learned bases, constants, and bytes are
  forbidden. Their quarantine existence is not numerical evidence and is not
  consumed by this construction.
- Live `j11` and `ks1` surfaces remain read-only. No paid dispatch, remote
  launch, GPU launch, score claim, pointer mutation, adoption language, or
  composition authorization is in scope.
- Pointer remains `0.1910828242 [contest-CPU]` and is not moved.

## Preregistered scorer-recursive formulation

The frozen evaluator factorization makes frame 0 invisible to SegNet and
visible to PoseNet. P1 therefore adds a legacy-compatible PC1 subtype:

`frame0' = uint8(parent_frame0 + NN24x32(sum_k c_ik * b_k * 2^e_k))`

`frame1' = parent_frame1` byte-for-byte.

The generic fixed-grid interpreter belongs in free receiver code. Every
video-selected basis value `b_k`, coefficient `c_ik`, and exponent `e_k` is
stored in `pose/pc1_frame0_quotient.ddp`. The existing `pose/pc1.ddp` schema and
two-frame receiver remain unchanged.

Treatment basis derivation is preregistered as:

1. At each exact G4 parent pair, differentiate frozen Pose6 with respect to the
   receiver-realized 24x32x3 frame-0 chart.
2. Form the target-directed damped minimum-norm actuator
   `u_i = J_i^T (J_i J_i^T + lambda I)^-1 (p_target_i - p_parent_i)`.
3. Center the 600 actuator rows and eigendecompose their covariance.
4. Quantize the leading shared actuator directions; solve each pair's
   coefficients with the same realized secant/Gauss-Newton procedure for
   treatment and control.

This is derived from the PoseNet inner Jacobian and exact target residual, not
from a generic disk/global-write/spatial menu.

## Preregistered canonical rank law

For descending actuator-covariance eigenvalues `lambda_j`, parent distortion
`D0`, and rank `r`:

`D_lin(r) = D0 * sum_{j>r}(lambda_j) / sum_j(lambda_j)`.

Select the least `r in {1,...,6}` with `D_lin(r) <= 5e-5`. If none exists, run
all six receiver-realized ranks; the linearized law does not substitute for
the exact reach curve.

Canonical implementation:
`tac.canonical_equations.ddm_p1_frame0_pose_quotient_carrier_20260725`.

## Exact treatment/control preregistration

The treatment and contrarian control are matched before measurement:

- same parent archive and parent raw bytes;
- same rank selected by the treatment rank law;
- same 24x32x3 packet geometry, int8 basis precision, int16 per-pair
  coefficient precision, power-of-two exponent precision, and coefficient
  solver/call budget;
- fixed-width, uncompressed packet and ZIP_STORED composition, so packet and
  archive byte counts must match exactly;
- treatment uses the target-directed PoseNet actuator basis;
- control uses a sealed seed-20260725 untargeted Rademacher basis, stored in
  the same counted basis home;
- both must parse back to byte-identical packet re-emission;
- both must preserve the complete parent frame-1 byte stream and SHA-256.

Before each n600 batch32 scorer pass, `psutil.virtual_memory().available` must
be at least 20 GiB. The scorer is frozen macOS CPU, four threads, batch 32;
these rows are advisory and never contest authority.

## Preregistered hypotheses and exact decision test

The delegated pass test requires all of:

1. treatment exact n600 batch32 `d_pose <= 5e-5`;
2. `d_seg(treatment) - d_seg(control) = 0` within exact replay tolerance,
   backed by frame-1 byte identity and frozen SegNet's frame-1-only
   factorization;
3. treatment counted carrier packet `<= 30,000` bytes;
4. the exact matched-control fence above passes.

GC4's stricter `d_pose <= 2.94e-5` remains a separate contrarian/adoption veto.
Meeting the delegated 5e-5 bar does not clear that stricter veto.

## Falsifier and negative scope

If no rank reaches `5e-5`, emit at least five exact receiver-realized rows of
`rank, d_pose, carrier_bytes`, plus the exact control row and a named
obstruction. The only allowed negative is:

`FORMULATION: shared low-rank, quantized, parent-additive 24x32 frame-0 actuator
basis with per-pair coefficients under the exact G4 parent and frozen
batch32 scorer.`

No negative may close frame-0 quotient carriers, scorer-solved inverse
renderers, larger/shared nonlinear generators, or the optimal family.

## Custody frozen before measurement

- G4 receipt:
  `.omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/g4_receipt.json`,
  2,863 bytes,
  SHA-256 `7f30e8aa765512682276310199b1e08c7d854c5cfbb36b272f79359a39339342`.
- Exact G4 parent archive:
  `/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/e5a_runtime/candidate_packet/archive.zip`,
  130,101 bytes,
  SHA-256 `fb69964da2649c310b7694416ff9863e13f54594af215cd771dcd50f5898a85d`.
- G4 realized state:
  `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`;
  its preserved batch32 raw chunks are read-only scorer inputs.
- MS4d direct metric completion:
  `.omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json`,
  2,799 bytes,
  SHA-256 `d670eff3dd01d61a24bdebedf045fa8cde2528953660dc6d1e64ba9c2fa94e25`.
- Pose6 target rows:
  `/Volumes/VertigoDataTier/pact/ddm_ms4_metric_producers_and_measurement_20260724T042005Z/pose_metric_n600_batch32.json`,
  577,705 bytes,
  SHA-256 `5e06cc78711a6ca6984c907600a25816cdecc6239903f782d85bcf9473a8f1bc`.
- Frozen scorer source SHA-256:
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`.
- Frozen PoseNet weights SHA-256:
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.
- Frozen SegNet weights SHA-256:
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.

## Triality and stores consulted

- DSL: typed config
  `.omx/research/configs/ddm_p1_frame0_pose_quotient_carrier_20260725.json`.
- DAG: `FEED-603-p1` is owed after measurement.
- Equations:
  `ddm_p1_frame0_pose_quotient_rank_law_v1`.
- Stores consulted: full `CLAUDE.md`, full `AGENTS.md`, craft handoff manual,
  `PROGRAM.md`, v7.5 and v8 vehicle specs, current frontier, lane registry,
  subagent checkpoints, per-arm inbox, broadcast inbox through
  `2026-07-24T23:09:25Z`, GC4 council/findings/FEED, optimal-start card,
  PC1/PC2 source and receipts, G4 custody, MS4d custody, and frozen scorer
  source/weights.

## Measurement result

`NOT_RUN_AT_PREREGISTRATION`.
