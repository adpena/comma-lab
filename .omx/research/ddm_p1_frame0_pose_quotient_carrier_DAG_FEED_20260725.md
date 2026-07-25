# FEED-603-p1 — frame-0 PoseNet quotient carrier

UTC preregistration: `2026-07-25T14:33:03Z`  
State: `MEASURED_FORMULATION_BLOCKED`

Research-only: `true`  
MAIN landing review required: `true`

## Executable DAG

`G4 exact parent archive + preserved batch32 raw chunks`

→ `frame-0 24x32x3 receiver chart; frame 1 held byte-identical`

→ `frozen PoseNet inner Jacobian J_i at exact parent pair`

→ `target-directed actuator u_i = J_i^T(J_iJ_i^T+lambda I)^-1(p*_i-p_i)`

→ `shared actuator covariance spectrum`

→ `least rank r<=6 under D_lin(r)=D0*tail_energy_fraction`

→ parallel exact-budget arms:

- treatment: leading target-directed actuator basis;
- control: seed-20260725 untargeted Rademacher basis.

→ `same solver + int8 basis + int16 per-pair coefficients + power-of-two scale`

→ `fixed-width PC1 counted packet; exact parse-back; packet <=30,000 bytes`

→ `frame-1 byte/SHA identity`

→ `psutil available-RAM >=20 GiB`

→ `frozen macOS-CPU n600 batch32 PoseNet advisory rows`

→ one of:

- delegated row: treatment `d_pose<=5e-5`, zero Seg delta, matched fence passes;
- falsifier: at least five `rank,d_pose,carrier_bytes` rows plus named,
  formulation-scoped obstruction.

GC4's stricter `d_pose<=2.94e-5` stays a separate veto. Neither outcome
authorizes adoption, composition, contest score, dispatch, or pointer movement.

## Triality

- DSL:
  `.omx/research/configs/ddm_p1_frame0_pose_quotient_carrier_20260725.json`.
- Equations:
  `tac.canonical_equations.ddm_p1_frame0_pose_quotient_carrier_20260725`.
- Receiver/packet:
  `tac.optimization.ddm_p1_frame0_pose_quotient_carrier`.
- Findings:
  `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md`.

## Pointer delta

`0.1910828242 [contest-CPU] -> 0.1910828242 [contest-CPU]` (`UNMOVED`).

## Empirical feed

Exact n600/batch32 local macOS-CPU advisory replay completed.

`D0=35.49982400273336`; the derived linearized rank law selected no rank
because `D_lin(1..6)=[16.20747111074233,12.810019232551447,
9.857003656593685,7.822461440923049,6.6487662599665285,
5.560905766976562]`, all above `5e-5`.

Typed receiver-realized reach curve:

`rank:carrier_bytes:d_pose =
1:3520:19.89493129583306,
2:7025:23.666537871896537,
3:10530:23.492813182500594,
4:14035:26.054884825627166,
5:17540:27.36323513050055,
6:21045:48.14744629668481`.

Matched rank-6 seeded control:
`21045 bytes, d_pose=20.31820279520745`.

Treatment/control fence:

- same rank, precision, four-iteration call budget, packet bytes, and archive
  bytes: `PASS`;
- packet parse-back byte identity: `PASS`;
- frame-1 digest-chain identity:
  `6da41ce656285d4a88baea9725c2513bcfca653da0d000a20a09ee996b2f5722`;
- Seg-cell digest-chain identity:
  `d9610fcff842f3d50015d49908020321598f8f3fecaac86d6c3e98fe2346bdcf`;
- measured `d_seg` delta: `0.0`;
- all carrier rows below 30,000 bytes: `PASS`;
- delegated Pose endpoint: `FAIL`.

Named obstruction:
`SHARED_BASIS_TARGET_ACTUATOR_SPECTRAL_TAIL_PLUS_EXACT_UINT8_TRUST_REGION_CROSSING`.
The six-dimensional shared actuator covariance leaves a large target residual,
and the exact rank-6 four-step update crosses the uint8 receiver trust region.
This is formulation-scoped; nonlinear, pair-conditioned, higher-rank, and
scorer-solved quotient generators remain open.

## Reusable system signal

- The canonical rank law now refuses a selected rank when the measured
  covariance tail cannot reach the target.
- The typed runner automatically falls through to the six-point exact
  falsifier, preserving every n600 batch checkpoint.
- The PC1 subtype counts the learned basis, every per-pair coefficient, and
  every exponent; its parser proves byte-identical re-emission and exact
  frame-1 inheritance.
- Exact receiver non-monotonicity is now a strict signal: future variants need
  an explicit uint8 trust-region admission and pair-conditioned/nonlinear
  geometry rather than increasing this shared PCA rank.

Consumer: P3/G5 composition must treat P1 as unavailable and retain
PC1-joint-descent (`j11/#366`) unless a separately preregistered formulation
clears both the delegated and GC4 contrarian bars.
