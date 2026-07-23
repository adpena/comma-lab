# DDM Costate Organ Elevation 2 — typed DSL surface

Date: 2026-07-23  
Maturity: `_dev`  
Authority: advisory only; `research_only=true`; `execution_allowed=false`; `actuation=NONE`  
Landing: MAIN review required

## Purpose

This note names the typed control surface consumed by the live DDM costate organ. It does not add
trainer flags or an execution path. Any later actuator must compile these blocks through a reviewed
typed DDM program; stringly invented launcher flags are refused.

## Producer inputs

| Producer | Typed receipt family | Role | Horizon |
|---|---|---|---|
| dv1 | `ddm_description_vocabulary_receipt.v1` | bytes and described-fraction primitives | until a newer dv1 or receiver realization |
| g3 | `ddm_g3_score_atlas_receipt.v1` | pair/site score debt, visibility geometry, byte allocation | until atlas/scorer/reconstruction changes |
| g4 | `ddm_g4_spatial_stationarity_receipt.v1` | recurrence opportunities and cell-space marginal value | until decomposition or realization changes |
| v19 | `ddm_v19_pure_priced_objective_receipt.v1` | exact receiver pair outcomes | one downstream joint-stack mutation |
| v19b | `ddm_v19b_joint_remeasure_stack_receipt.v1` | sequential block outcomes and D2 joint survival | one downstream joint-stack mutation |
| e1 | pending schema-registered producer | J_paint/export realization | re-derive when it lands |
| dv2 | pending schema-registered producer | recursive description grammar | re-derive when it lands |

Selection is latest run-id timestamp within each registered family, followed by receipt schema,
authority-firewall, byte-count, and SHA-256 verification. A partial producer set is not blended
with witness-era telemetry.

## Advisory block vocabulary

Every block has:

`block_id`, `dependencies[]`, `coarse_level`, `frees_bytes`, `lambda_abs`,
`validity_radius`, `lambda_status`, `reason`, and source hashes.

Current live blocks:

1. `j_paint_dv1_persistent_ground`
2. `j_paint_g4_movable_midband`
3. `r6_exact_receiver_rehearsal`
4. `ddm_iteration_curve_instrument`

`source_custody` and `pair_site_lambda` are derivation prerequisites, not actions. A block with
`lambda_status=UPPER_BOUND_ONLY_UINT8_REALIZABILITY_OWED` can be recommended only as an
instrumentation duty. It cannot be admitted as a score-moving update.

The scheduler compiles:

`dependency frontier → freeing before spending → coarse to fine → max |lambda|×validity_radius`.

## Resume surface

`DdmCostateCheckpoint` implements the canonical `Resumable` protocol and registers as
`ddm_live_costate_advisory` with prefix `__ddmcostate_`. The checkpoint contains only:

- the complete receipt/summary/atlas source-hash set;
- completed block IDs;
- advisory cycle index.

Restore fails when any source hash changed. The CLI also refuses to overwrite an existing receipt.

## CLI

```text
python3 tools/ddm_costate_organ.py
python3 tools/ddm_costate_organ.py --json
python3 tools/ddm_costate_organ.py --write-receipt <new-append-only-path>
python3 tools/ddm_costate_organ.py --resume-from <prior-receipt> --write-receipt <new-path>
```

No command launches training, dispatches a provider, changes a run, or promotes a score.

## `_dev` → `_prod` gate

All must hold:

1. required input hashes are closed and consumer-verified;
2. full 600-pair g3 pair/site lambda is available without shared-byte duplication;
3. J_paint receiver realization and R6 exact rehearsal are complete;
4. the DDM iteration instrument passes hash-stable walk-forward `r2 >= 0.5`, or remains retired;
5. three clean independent reviews and MAIN landing review complete;
6. contest-CPU and contest-CUDA evidence remain separate from macOS advisory evidence.

