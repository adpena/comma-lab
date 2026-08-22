Implemented and committed the JO1U2 cure as `056433e297`.

Root cause: the Modal image mixed top-level and `experiments.*` module layouts, so `experiments.modal_auth_eval` failed during module import—before the entrypoint’s first volume write. The local `/Volumes` dependency and config-SHA hypotheses were refuted.

The cure adds:

- Complete package-path mounts and explicit imports.
- Immutable remote start/failure receipts with stage, inputs, traceback, and commit-before-reraise ordering.
- Coverage for malformed requests and worker-defined exceptions.
- A CPU-only control-plane probe.

The one authorized probe was attempted, but Modal connectivity failed before creating a provider call. No call ID, GPU, scorer, payload, or remote artifact exists. The durable receipt is [PROBE_LOCAL_RECEIPT.json](/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/jo1u2_probe_r1/PROBE_LOCAL_RECEIPT.json).

The r2 seal is structurally `READY_TO_FIRE`:

- [compiled_config.json](/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r2/compiled_config.json), SHA `9477ad1e…c9c9d6`
- [FIRE_ORDER.json](/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r2/FIRE_ORDER.json), SHA `ab5fd85a…8ee9a`
- Frozen fx5 archive remains unchanged, SHA `4b54fccc…9841`

Verification passed: two review passes, 21 tests, Ruff, compilation, diff checks, retention census with zero findings, seal reload, artifact pins, and CLI parsing. Provider-container imports, remote receipts, T4 materialization, scoring, and pointer movement were not measured.

Full receipt: [ddm_jo1u2_materializer_cure_20260821.md](/Users/adpena/Projects/pact/.omx/research/ddm_jo1u2_materializer_cure_20260821.md).

Own-vehicle frontier: **fx5_e1 — S 0.14823186109359 @ 180,386 B [contest-CUDA T4, n600]**, unchanged by JO1U2.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `comma-auth-eval-cache-artifacts:ddm_jo1u_fx5_e1_n600_r2/`; fire trigger: Modal connectivity restored, storage preflight passes, no active n600 scorer job, and MAIN holds the unique lane claim; action: fire ordinal 1 exactly once.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN harvester; consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest/`; fire trigger: ordinal 1 is terminal with a complete, verified `FINAL_RESULT.json`; action: run ordinal `1H` and reseal the harvested tensors.
- **BLOCKED** — owner: JO1 design plus MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/`; fire trigger: tensors are resealed, matching T4 memory evidence exists, AP has at least 44 GiB free, and the receiver-close backend is implemented; action: only then expose training.

## LIVE-HYPOTHESES

- The r2 image will enter the dispatcher body because every absolute package dependency now exists under one `/workspace/pact/experiments` tree.
- Any subsequent in-body failure will leave a useful immutable traceback receipt before the caller receives a loud built-in exception.
- Materialization will fit AP storage because the sealed requirement is 21,174,283,716 B versus 30,424,694,784 B free at seal time.

## DEAD-ENDS

- Re-diagnosing the #1167 local claim/refusal genus is closed: the failed r3 call dispatched successfully.
- Missing remote `/Volumes` mounts are closed: archive and runtime bytes are uploaded inline.
- Container-side config-SHA normalization is closed: validation occurs locally before spawn.
- The old v1 catch is closed: it cannot observe pre-body imports or guarantee an immutable committed traceback.
- Retrying the probe from this arm is closed: its single authorized attempt was consumed.
- Firing full n600 from this arm is closed: MAIN owns the scorer slot and re-fire.