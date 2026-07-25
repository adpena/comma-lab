# FEED-603-p1 — frame-0 PoseNet quotient carrier

UTC preregistration: `2026-07-25T14:33:03Z`  
State: `PREREGISTERED_NOT_MEASURED`  
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

`NOT_RUN_AT_PREREGISTRATION`.
