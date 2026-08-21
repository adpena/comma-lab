# DDM JO1U2 materializer cure — package closure, failure black box, and r2 seal

- Date: 2026-08-21
- Source disposition: **BUILT-AND-REVIEWED**
- Seal disposition: **READY_TO_FIRE for MAIN under the named trigger; NOT FIRED**
- Diagnostic probe disposition: **BLOCKED-EXTERNAL-PRE-DISPATCH after the one authorized attempt**
- Consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/`
- Authority boundary: control-plane diagnosis and component-payload readiness only. This arm produced
  no materialized scorer payload, T4 scorer result, candidate archive, exact score, or pointer movement.

## RESULT

The failed r1 Modal image mixed two incompatible module layouts. Its inherited auth-eval image mounted
`modal_auth_eval` as a top-level module, while the JO1 dispatcher imported
`experiments.modal_auth_eval`. JO1 mounted several other files under `/workspace/pact/experiments`, but
did not mount `experiments/__init__.py` or `experiments/modal_auth_eval.py` there. That import occurs
while Modal imports the dispatcher module, before `run_payload_materializer` enters its body. This
explains the complete evidence triple for call `fc-01M0JH3TK89HVSV03GAN8RHQWJ`: empty
`RemoteError`, no application logs beyond dispatch, and no
`comma-auth-eval-cache-artifacts:ddm_jo1u_fx5_e1_n600_r1/` directory.

The cure gives the image one explicit package topology, imports each dependency through
`importlib.import_module("experiments....")`, and mounts the package initializer, auth module, JO1
modules, and retained JS1B worker under `/workspace/pact/experiments` with copied bytes. A local
isolated-tree negative/positive control reproduced the old topology failure and then resolved the
same package import after the missing package file was supplied.

The dispatcher now also has a stdlib-only remote black box around the materializer body. On entry it
writes and commits an immutable attempt-start record. On any in-body failure it records stage,
inputs seen, error type/message/repr, and the full traceback; commits the immutable failure receipt;
writes a mutable convenience pointer; commits again; and re-raises a nonempty built-in
`RuntimeError`. Malformed requests take the same receipt path. No failure cleanup is allowed, so any
payload bytes already written remain retained. This second landing does not claim to catch a module
import that prevents the dispatcher module itself from loading; the package-closure cure addresses
that earlier boundary.

## ROOT-CAUSE ADJUDICATION

| candidate | disposition | source evidence |
|---|---|---|
| local `/Volumes` dependency inside the container | **REFUTED for this failure** | `_spawn_materializer` reads the frozen archive and runtime tree locally, builds the runtime bundle locally, and passes `archive_bytes` plus `runtime_zip_bytes` to the remote function. The `/Volumes/...` source paths in the request are metadata and are not opened by the remote worker. |
| custom-exception serialization masking | **NOT ESTABLISHED as the initiating defect; cured as a silence amplifier** | The import-closure defect occurs before the remote body can raise a JO1 worker exception. The new wrapper nevertheless converts all body failures to a structured volume receipt plus a loud built-in exception. |
| config-SHA or authorization mismatch | **REFUTED for this failure** | Config loading and expected-SHA validation happen in the local entrypoint before spawn. The remote request consumes the already-validated workload digest and uploaded bytes; it does not recompute the compiled-config digest from container-local path normalization. |
| mixed top-level/package image topology | **ROOT CAUSE, source-reproduced** | The inherited image supplied top-level `modal_auth_eval`, while JO1 required `experiments.modal_auth_eval` and omitted that package-path file. In an isolated tree matching the old mounts, `find_spec("experiments.modal_auth_eval")` returned no module; adding the package-path file made the same import resolve. The failure boundary is pre-body, matching the absent first volume write. |

This is a source-and-topology diagnosis, not a provider-container confirmation. The one permitted
Modal probe could not reach the provider, so the corrected module closure and remote black box remain
untested inside a live Modal container.

## CURE DIFF SUMMARY

- `experiments/ddm_jo1_modal_joint_objective.py`
  - selects `/workspace/pact` when its package tree exists and uses explicit absolute package imports;
  - mounts every dispatcher dependency at the package path, including `experiments/__init__.py` and
    `experiments/modal_auth_eval.py`;
  - records immutable `REMOTE_START` and structured failure receipts with stage, input identities,
    traceback, and two volume commits before re-raising;
  - routes malformed remote requests through a safe receipt directory;
  - adds an explicitly authorized, CPU-only, no-scorer control-plane probe that checks exact module
    paths/source hashes and injects a sentinel failure through the recorder.
- `experiments/tests/test_ddm_jo1_joint_objective.py`
  - seals the explicit package topology;
  - proves the recorder retains traceback/stage/inputs and raises a loud built-in exception;
  - proves the raw entrypoint and malformed-request paths write the receipt before a worker-defined
    exception can cross Modal;
  - proves the probe refuses to run without its diagnostic authorization.

The materializer physics and custody contract did not change: real fx5 decode, real T4 scorers,
full n600, batch 16, chunk limit 120, deterministic repeat, resume state, storage preflight, and all
payload retention remain intact. `memory_preflight` and `train` remain blocked on their existing
training gates.

## PROBE RECEIPT

The arm claimed and terminally closed lane `ddm_jo1u2_materializer_probe`, instance
`jo1u2_control_plane_probe_r1`, then made exactly one charter-authorized attempt:

```text
.venv/bin/modal run experiments/ddm_jo1_modal_joint_objective.py::probe_control_plane --output-receipt /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/jo1u2_probe_r1/PROBE_LOCAL_RECEIPT.json --run-id ddm_jo1u2_control_plane_probe_r1 --diagnostic-authorization
```

The CLI returned 1 with `Could not connect to the Modal server.` before the local entrypoint entered
or a provider call was created. There is therefore no call ID, GPU request, scorer load, payload,
remote volume directory, or call-ledger row. The lane was closed as
`failed_external_provider_connectivity_pre_dispatch`; this arm did not retry.

Durable local receipt:

- path: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/jo1u2_probe_r1/PROBE_LOCAL_RECEIPT.json`
- bytes: 1,352
- SHA-256: `b6fe15fea5bcbee3a1645fbdc873f50138c3910f9d2a9255ffd1c6c5df5816ba`
- measured boundary: **provider connectivity failure only**; not container-import or recorder evidence.

## R2 SEAL AND FIRE ORDER

The frozen fx5 input remains byte-identical: 180,386 B, SHA-256
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`. It was read but not
modified. The new seal is under the existing lineage at
`/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r2/`:

| artifact | bytes | SHA-256 | disposition |
|---|---:|---|---|
| `AUTHOR_CONFIG.json` | 19,261 | `7458e4482f1080456e63a1585b3e76cd0fef5125900ffa0ed876f7d669fa4333` | r2 author input |
| `compiled_config.json` | 15,394 | `9477ad1e1e12b5b70fdc6d6436473751e6aff8675b7b1347264185576cf9c9d6` | **READY_TO_FIRE** |
| `FIRE_ORDER.json` | 1,516 | `ab5fd85a0165573669ca6483a5cd47354d8a3878cb5683c97bb7980a0ed8ee9a` | one MAIN fire plus typed downstream rows |
| `READINESS.json` | 710 | `8e51bf125200696dd00090f79714619825ee1cebbd200541ee4e5876f51abec9` | zero build/storage blockers |

The compiled workload SHA-256 is
`0e5571a73505a87b18d0b08fa5681d00c38ac539535de6ceef9f87f92c9aff69`; the remote volume
run ID is `ddm_jo1u_fx5_e1_n600_r2`. Its dispatcher pin is 32,211 B, SHA-256
`ecdc28c58efa6c9a74bee8c638104e82dd868bc82b6eeb88283238d1ed6ffd05`.

Seal-time storage was `[MEASURED local APDataStore]`: 30,424,694,784 B free; 694,780 B already
retained; 16,880,011,200 B expected total payload; 16,879,316,420 B remaining payload;
4,294,967,296 B reserve; 21,174,283,716 B required free. The 47,244,640,256 B training
requirement was explicitly not applied and still does not authorize training.

The one full materializer re-fire remains MAIN-owned. Exact ordinal-1 argv:

```text
.venv/bin/modal run experiments/ddm_jo1_modal_joint_objective.py::materialize_scorer_payloads --compiled-config /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r2/compiled_config.json --expected-config-sha256 9477ad1e1e12b5b70fdc6d6436473751e6aff8675b7b1347264185576cf9c9d6 --main-owned-dispatch-authorization --detach --provider-detach-ack
```

Operational fire trigger: Modal connectivity is restored; the materializer storage preflight still
passes; no n600 scorer job is active; and MAIN holds a unique `ddm_jo1_payload_unblock` lane claim.
The arm did not fire this command.

## RECALL EVIDENCE

The full-corpus recall searched `.omx/research/` receipts, `CANONICAL_RESEARCH_INDEX*`,
`sub015_DAG_*` FEED blocks, design/SPEC/task-ledger surfaces, `.omx/state/main_hot_state.md`, Git
history, the canonical equation list, and live dispatcher/worker sources. Content queries included
`ddm_jo1`, `materialize_scorer_payloads`, `RemoteError`, `experiments.modal_auth_eval`,
`add_local_python_source`, `remote_volume_run_id`, `runtime_tree`, `call_id_ledger`, and
`modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1`.

Beyond the charter seeds, `.omx/research/ddm_dr1_dispatch_infra_repair_20260814.md` supplied an exact
prior instance of the same mixed top-level/package Modal image class and its optimal cure: explicit
package imports, package-path mounts, and a CPU-only import smoke. That evidence changed the plan from
instrumenting local `/Volumes` opens or editing the worker algorithm to first reproducing and repairing
the dispatcher image closure. The canonical runtime-tree parity equation reinforced the need to prove
local/worker topology rather than infer it. The prior JO1U receipt and JS1B retained worker confirmed
that archive/runtime bytes are uploaded and that the materializer body already owns real decode,
scoring, retention, and resume semantics. No recalled evidence justified changing payload physics,
chunking, storage arithmetic, or the frozen fx5 bytes.

## PRIOR-LAW COUNT

The M2 plumbing-family prediction landed once, but its literal suspect did not: this was one
container import/mount binding defect, **not** a missing upload leg for `/Volumes` inputs. Zero new
physics defects landed. A genuine Modal runtime or exception-serialization limitation was not
established. Custom-exception masking remains a plausible amplifier of future body failures and was
cured regardless, as required.

## VERIFICATION

- Review pass 1: full source review, `ruff`, `py_compile`, 21 focused tests, review scan/diff-scan,
  and file marking. It found and fixed two recorder holes: malformed requests bypassed the wrapper,
  and the immutable failure receipt was not committed before the mutable pointer.
- Review pass 2: independent reread plus fresh `ruff`, `py_compile`, 21 focused tests,
  `git diff --check`, review scan/diff-scan, and file marking: PASS.
- `pytest -q experiments/tests/test_ddm_jo1_joint_objective.py`: **21 passed**. The existing Pydantic
  field-shadow warnings are unchanged and are not failures.
- Exact isolated import topology: old package tree failed to resolve
  `experiments.modal_auth_eval`; cured package tree resolved it: PASS.
- Always-keep-the-payload bounded census: 2 Python files discovered/examined, 1 producer candidate
  parsed, 0 unreadable files, 0 findings: PASS.
- R2 compiled-config reload, all artifact/source pins, storage readiness, frozen archive hash, and
  exact combined `file.py::entrypoint` CLI parse: PASS.
- Provider-container import closure and remote recorder execution: **NOT MEASURED**, because the
  one authorized probe ended before dispatch with external connectivity failure.

## MEASURED / NOT MEASURED

**MEASURED now:** old mount topology from source; isolated negative/positive package resolution;
inline archive/runtime request flow; source and artifact hashes; frozen archive identity; local AP
storage arithmetic; two review passes; 21 focused tests; bounded retention-gate result; r2 config
reload/readiness; exact CLI parse; and the pre-dispatch provider-connectivity failure. These are
source, local-control, and custody measurements on the axes stated above.

**NOT MEASURED:** corrected imports in a Modal container; remote start/failure receipt creation;
T4 startup; exact receiver decode; new base argmax or Pose6; any scorer tuple; materializer wall time
or cost; training; candidate archive; `d_seg`; `d_pose`; contest-CPU/CUDA score; or pointer movement.
`READY_TO_FIRE` describes the reviewed source and byte-closed seal under the operational trigger; it
does not claim that the provider accepted a job or that future payloads exist.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `comma-auth-eval-cache-artifacts:ddm_jo1u_fx5_e1_n600_r2/` with local custody under `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/`; fire trigger: Modal connectivity is restored, r2 storage preflight still passes, no n600 scorer job is active, and MAIN holds the unique `ddm_jo1_payload_unblock` claim; action: fire ordinal 1 exactly once and stop for its terminal receipt.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN harvester; consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest/`; fire trigger: ordinal 1 is terminal and its volume `FINAL_RESULT.json` is complete with every retained batch record verified; action: run ordinal `1H`, bind the harvested fields into a new seal, and do not continue into training in the same step.
- **BLOCKED** — owner: `ddm_jo1_joint_objective_design` plus MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/`; fire trigger: current-base tensors are harvested and resealed, a fresh matching real-T4 peak-memory receipt exists, AP free is at least 44 GiB, and the fresh same-object Schur receiver-close backend is implemented and reviewed; action: only then expose memory-preflight and training commands.

## LIVE-HYPOTHESES

- The r2 image will enter the dispatcher body and write `REMOTE_START.json`, because the old missing
  package module was reproduced in an isolated tree and every absolute package dependency is now
  mounted under the same `/workspace/pact/experiments` root. This remains plausible but untested in
  a provider container.
- If the materializer body fails after entry, the immutable failure receipt will survive with a
  useful stage and traceback before the caller sees a loud built-in exception, because unit tests
  exercise the raw entrypoint, malformed request, custom worker exception, and commit ordering.
- The full materializer will fit the AP tier because the byte-closed retention estimate plus 4 GiB
  reserve requires 21,174,283,716 B while 30,424,694,784 B was free at seal time. Fire-time free
  space can drift and must be rechecked.

## DEAD-ENDS

- Reopening the #1167 local claim/refusal genus is closed: r3 dispatched cleanly and the observed
  failure began remotely.
- Treating `/Volumes/APDataStore` or `/Volumes/VertigoDataTier` as missing remote mounts is closed
  for this failure: the dispatcher sends archive and runtime bytes inline, and the other local paths
  are provenance metadata.
- Recomputing the compiled-config SHA in the container is closed: validation occurs locally before
  spawn, and no container path normalization participates in the remote workload digest.
- Relying on the old v1 in-body catch is closed: it could not observe pre-body import failure and did
  not guarantee a committed immutable traceback before re-raise.
- Retrying the CPU probe from this arm is closed: the charter allowed one attempt, which ended at a
  recorded external-connectivity boundary before a provider call existed.
- Firing full n600 from this arm is closed: MAIN owns ordinal 1, the unique lane claim, and the scorer
  slot.

Own-vehicle frontier: **fx5_e1 — S 0.14823186109359 @ 180,386 B [contest-CUDA T4, n600]**, archive SHA `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`; **UNMOVED by JO1U2**.
