# Codex findings — DDM P1 frame-0 PoseNet quotient carrier

UTC preregistration: `2026-07-25T14:33:03Z`

Status at preregistration: `PREREGISTERED_NOT_MEASURED`

Final status: `P1_SHARED_LOW_RANK_FRAME0_ACTUATOR_FORMULATION_BLOCKED`

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

Exact local measurement completed after the preregistration above.

Evidence axis: `[macOS-CPU frozen-scorer advisory]`

Pair count / batch geometry: `n600 / batch32`

Delegated pass: `false`

GC4 strict veto cleared: `false`

Pointer moved: `false`

The measured actuator spectrum did not contain a rank at the preregistered
linearized target:

| rank | covariance eigenvalue | predicted `D_lin(r)` |
|---:|---:|---:|
| 1 | 4,866,776.639072087 | 16.20747111074233 |
| 2 | 857,056.6548165884 | 12.810019232551447 |
| 3 | 744,941.1329114469 | 9.857003656593685 |
| 4 | 513,242.86787965405 | 7.822461440923049 |
| 5 | 296,081.68169274926 | 6.6487662599665285 |
| 6 | 274,428.6331218263 | 5.560905766976562 |

The parent distortion was `35.49982400273336`; the derived law selected no
rank. Per the frozen falsifier, all six exact receiver-realized ranks were
nevertheless measured:

| rank | treatment `d_pose` | carrier bytes | archive bytes | carrier SHA-256 |
|---:|---:|---:|---:|---|
| 1 | 19.89493129583306 | 3,520 | 134,991 | `9f7e65d8ccd19e0cb7c4976af4c4fa74bebef6bfff1a58679e323924cf42d7f5` |
| 2 | 23.666537871896537 | 7,025 | 138,496 | `57ee2589529241b4ed64bc3bce6a1b0be51b96ecbe6aa7f445712cfac3dd7e10` |
| 3 | 23.492813182500594 | 10,530 | 142,002 | `3d21a046ce8c217b5ea091a23ea8ca12475a490fb32d45a18eefda36a9b9d1b3` |
| 4 | 26.054884825627166 | 14,035 | 145,507 | `b8ca1fb39c78b31ea3cf563b156b4386f0e3d72ae03f73ed0930c616809c16f2` |
| 5 | 27.36323513050055 | 17,540 | 149,013 | `e899f4bbdf5ad1eaf6fa1f0121de6ff2fc5b9c962bc91c054d8cc1bb42417d6c` |
| 6 | 48.14744629668481 | 21,045 | 152,518 | `38cd2e6327d01b90f732e1d2cb7c08536ac2c9838ae44932668d3b395c574545` |

The exact rank-6 seeded matched control measured
`d_pose=20.31820279520745` at the same `21,045` carrier bytes and `152,518`
archive bytes. Its carrier SHA-256 is
`33c25bb5a455c25c429889c14cae41c119ca770d46416a0574a0645a3c7d5ebe`.
Treatment and control used four Gauss-Newton iterations each.

## Preregistered decision-test disposition

- `d_pose(treatment) <= 5e-5`: **FAIL**; best measured treatment row is rank 1
  at `19.89493129583306`.
- `d_seg(treatment)-d_seg(control) = 0`: **PASS**, measured `0.0`.
- `carrier_bytes <= 30,000`: **PASS** for all six treatment rows.
- exact matched-control fence: **PASS**. Treatment and control have the same
  rank, precision, solver call budget, packet bytes, and archive bytes.
- parse-back byte identity: **PASS** for every treatment and control packet.
- frame-1 identity: **PASS** with shared batch digest-chain SHA-256
  `6da41ce656285d4a88baea9725c2513bcfca653da0d000a20a09ee996b2f5722`.
- Seg-cell identity: **PASS** with shared digest-chain SHA-256
  `d9610fcff842f3d50015d49908020321598f8f3fecaac86d6c3e98fe2346bdcf`.

The all-video-state-counted treatment basis SHA-256 is
`bd6a2c6b599f4106bae402454831cad5b1834f4ea64d6d395de6841a3df47dcf`;
the independently seeded control basis SHA-256 is
`22189a2f7fc515117bd52b18ee658abc83c5f18898ef60399c018f6009313efa`.
The unquantized derived basis SHA-256 is
`3fa757386c918f6772429c7e600582b2eaaaa9199450974631ed0884743971cc`.

## Named obstruction and verdict scope

Named obstruction:
`SHARED_BASIS_TARGET_ACTUATOR_SPECTRAL_TAIL_PLUS_EXACT_UINT8_TRUST_REGION_CROSSING`.

The measured six-dimensional target-directed actuator covariance has a large
tail: even the preregistered linearized rank-6 law predicts `d_pose=5.5609`,
more than five orders of magnitude above the bar. Exact realization then
becomes non-monotone with rank, and the fixed four-step rank-6 solve crosses the
uint8 receiver trust region (`48.1474` treatment versus `20.3182` matched
control). Thus this shared PCA chart does not retain enough pair-specific
target-actuator geometry, and its unconstrained local coefficient update is not
receiver-stable.

This is only a
`FORMULATION: shared low-rank, quantized, parent-additive frame-0 actuator basis
with per-pair coefficients; frame 1 is an exact parent-byte identity`.
It does not close nonlinear, pair-conditioned, higher-rank, or
scorer-solved frame-0 quotient generators.

## Measurement custody

- Full immutable SSD receipt: 21,139 bytes, SHA-256
  `d08bc13fc1cb3962f99560631700f495e059b286c3c364bdf0dcb3f392e68e55`.
- Typed SSD reach curve: 2,257 bytes, SHA-256
  `2bfc82077bada5d60726fea1d98afd071224f79be11c51f6cb31192f7e0e1010`.
- Spectrum receipt / 600-row spectrum SHA-256:
  `eb7761a5fd77cf77e827353b0d31f7ef8209aeb141f479864d82b3f08e49c466`
  / `94ba3b2ae6ae71ab3a8cafe4c36eb60c4afee38deda9c7f5d5166eb1ee1c4f13`.
- Solved-packet receipt SHA-256:
  `e43f8be31b34c01020d614ce3f1f1afe8750f54e5f34e59fca94bad424dffd99`.
- SSD root:
  `/Volumes/VertigoDataTier/pact/ddm_p1_frame0_pose_quotient_carrier_20260725T141713Z`.
- Derivation RAM admission: `91,979,186,176 >= 21,474,836,480` bytes.
  Every exact replay row has its own passing psutil admission receipt.
- Storage admission: `424,198,135,808 >= 21,474,836,480` free bytes.
- All derivation batches, solver batches, packets, archive receipt-bytes, and
  exact replay batches are stage-checkpointed and preserved; no parent bytes
  were deleted.

Canonical compact receipt:
`.omx/research/ddm_p1_frame0_pose_quotient_carrier_receipt_20260725.json`.
Typed reach curve:
`.omx/research/ddm_p1_frame0_pose_quotient_carrier_reach_curve_20260725.jsonl`.
The receiver/packet subtype and generic runner are durable, but the failed
contrarian test forbids composition or adoption.
