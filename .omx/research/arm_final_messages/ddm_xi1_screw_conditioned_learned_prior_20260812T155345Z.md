Leg B completed and landed; Leg A remains honestly blocked before training because this sandbox exposes no Metal device. Commit: `3dc3b791c7`.

Full receipt: [ddm_xi1_screw_conditioned_learned_prior_20260812.md](/Users/adpena/Projects/pact/.omx/research/ddm_xi1_screw_conditioned_learned_prior_20260812.md:1). Resumable runner: [run_ddm_xi1_screw_conditioned_learned_prior.py](/Users/adpena/Projects/pact/tools/run_ddm_xi1_screw_conditioned_learned_prior.py:1209).

### Leg A — learned ξ context

| λ | spatial Range bytes | spatial+ξ Range bytes | verdict |
|---:|---:|---:|---|
| 1.0 | UNMEASURED | UNMEASURED | Metal unavailable |
| 0.5 | UNMEASURED | UNMEASURED | Metal unavailable |

The stratified-random n120 selection and matched context planes are retained. The ξ plane’s deterministic repeat is byte-identical, SHA-256 `8dc5db71…`. CPU substitution was refused.

### Leg B — exact n600 pose-carrier race

| Coding | Bytes | Realized d_pose | CPR1 decode |
|---|---:|---:|---|
| Direct CPR1 | 22,327 | 0.00000688 | exact |
| CAP1 AR(1)+bias | 22,242 | 0.00000688 | exact |
| Counted learned ξ linear prior | 30,072 | 0.00000688 | exact |

All decoded CPR1 payloads share SHA-256 `709ea928…`.

- FA: not adjudicated; no learned Leg A byte rows exist.
- FB: does not fire. CAP1 beats direct by 85 B, while the self-contained learned ξ formulation loses by 7,745 B.
- Frontier: unmoved.
- Verification: self-test passed, Ruff and `py_compile` clean, payload-retention gate found zero violations, and two review passes covered all 46 Python entities.

## NEXT_IF_RESUMED

- ddm_xi1_leg_a_mps: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN Metal executor. Consumer store: /Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_A_RESULT.json. Fire trigger: a governed process reports torch.backends.mps.is_available() == True; execute the pinned resume_command without CPU substitution.

## LIVE-HYPOTHESES

- Joint ξ-conditioned HPAC may still reduce Range bytes because frozen XOR/LZ failures never measured a jointly learned conditional probability model.
- A stronger nonlinear ξ prior could pay when geometric pose is already carried by the target vehicle; on CP135 it must first overcome the honest 7,200-byte context tax.

## DEAD-ENDS

- Self-contained linear geometric-ξ conditioning on current CP135 is closed at formulation scope: 30,072 B versus 22,327 B direct.
- Treating geometric ξ as free on CP135 is closed: CP135 carries a photometric carrier, not that pose plane.
- CPU substitution for the CL1 MPS experiment is closed as invalid.
- Own-vehicle frontier remains LC2 `S = 0.16959899569230852 @ 187,226 B [macOS-CPU advisory, n600]`.