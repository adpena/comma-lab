# DDM RS1 feature-relay DAG feed

Date: 2026-07-25  
Feed id: `FEED-659-ddm-rs1-feature-relay-input-custody-blocker`  
Research only: `true`  
Canonical append: `DEFERRED-MAIN-REVIEW`

## Proposed append

```text
NODE FEED-659-ddm-rs1-feature-relay-input-custody-blocker
  authority = [macOS-CPU frozen-scorer advisory]
  receipt = .omx/research/ddm_rs1_feature_relay_solve_20260725T030635Z/receipt.json
  receipt_sha256 = 23d074104ba2081fad50bdb00df5a3415cb076f7dee6f0ab18982350d91e7e71
  verdict = BLOCKED_INTERNAL_STATION_DYNAMICS_NOT_CUSTODIED
  verdict_scope = FORMULATION x current SHA-bound #484/AT1/SN1/MS4D/J8F custody
  family_kill = false
  g3_top24 = NOT_RUN_INPUT_ADMISSION_REFUSED
  bounded_n600 = NOT_RUN_G3_FIRST_GATE_REFUSED
  score_claim = false
  pointer_delta = NONE
  reactivation = SHA-bound G3-top24 block2/block3 targets + Fisher Grams +
                 input->block2->block3->rank4 Jacobians + continuity secants
```

## Dependency edges

```text
#580 range(A) projector --------------------------\
#484 block2/block3 locus definitions --------------+--> RS1 input-custody gate
AT1x input/scorer-plane atlas ---------------------+
SN1 aggregate telemetry ---------------------------+
MS4D rank4 final metric ---------------------------+
v17 direct validity law ---------------------------+
J8F receiver-closed n600 realized harness --------/

RS1 input-custody gate --REFUSE--> G3 top24 relay/direct ladder
G3 top24 relay/direct ladder --NOT RUN--> bounded n600
bounded n600 --NOT RUN--> ms2r relay-target law
```

The refusal is intentional: the dependency set establishes station identities
and end-verdict machinery, but not the measured internal dynamics needed by the
multiple-shooting equations.

## Triality

- Typed DSL/config:
  `.omx/research/configs/ddm_rs1_feature_relay_solve_20260725.json`
- Equation surface:
  `.omx/research/ddm_rs1_feature_relay_solve_canonical_equations_20260725.md`
- Evidence:
  `.omx/research/ddm_rs1_feature_relay_solve_20260725T030635Z/receipt.json`

No shared DAG bytes were edited. MAIN decides whether to append this feed after
independent landing review.
