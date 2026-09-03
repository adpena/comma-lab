# DDM WC3 — QBR1 executable EMA law cure and cured fire order

**Verdict:** `CURED-SEAL-BUILT / BURN-NOT-FIRED / RESUME-IDENTITY-OWED-TO-MAIN`.
The silent intervention mismatch is removed at the DSL, construction, resume, and
preflight surfaces. Six configs were regenerated from the exact cured source and every
build payload is retained. This arm ran zero scorer, Metal, Modal, and contest-evaluator
invocations. The exact frontier did not move.

This handoff follows the claim-label and re-derivation discipline in
`docs/operating_manual_craft_handoff.md`: facts below are labeled by how they were obtained,
and the missing scorer-owned proof is not promoted to a pass.

## Cure verdict

| row | old sealed / executable | cured sealed / executable | status |
|---|---|---|---|
| EMA mode | constant decay / implicit warmup | constant decay / constant decay | `DERIVED + TESTED` |
| `warmup` | absent from the typed contract / literal `True` | DSL `execution.mode=constant_decay`, `warmup=false` / constructed from that field | `MEASURED-CONFIG` |
| decay | `0.9990793899844618` / same cap | `0.9990793899844618` / same | `MEASURED-CONFIG` |
| terminal initialized-shadow coefficient at 5,000 updates | `0.010000000000000278` / `1.838001854879489e-27` | `0.010000000000000278` / `0.010000000000000278` | old `DERIVED`; new `TESTED` and independently recomputed at construction |
| mismatch response | silent | JSON `confound_alarm(ema_law_mismatch)` and halt beyond `1e-12` relative | `TESTED` |
| warmup | accidental default | available only as typed `warmup_ablation`, with `ablation_declared=true` | `TESTED` |

`EmaDecayCalibrated` is the single factory. It compiles the execution mode alongside the
`ema_decay_run_geometry_v1` LawRef. QBT1 and QBR1 call
`construct_ema_from_config`; checkpoint restoration calls the same verifier before the
resumed state can continue. Results stamp both `ema_law_sealed` and `ema_law_executed`.
The adversarial review found and fixed one silent second-order defect: construction had
initially trusted the stored sealed coefficient. It now independently recomputes the seal
and catches a stale or tampered coefficient as well as decay, mode, warmup, update-count,
or executable-law drift.

The PR130-lift sibling constructors now read their already-existing typed policy rather
than carrying literal flags; their declared warmup behavior is unchanged.

## Self-protection and verification

- Catalog **#412**, `check_ema_executable_law_matches_sealed_law`, is STRICT. It scans
  maintained EMA/LawRef modules for literal Boolean `warmup` construction and permits only
  a same-line `EMA_WARMUP_ABLATION_OK:<substantive rationale>` waiver; placeholder rationales
  are rejected. The registered positive control reproduces the original QBR1 constructor.
- Live strict count is **0 violations** across **6,812 maintained modules considered, 122 EMA
  candidates parsed, 6 EMA-LawRef modules, and 6 EMA constructors**.
- `experiments/tests/test_ddm_wc3_qbr1_ema_law_cure.py`: **26 passed** on exact cured commit
  `106d0dd0a094dd4c289eba69c8d2c5124e13eb02`. The gate suite includes original-shape,
  literal-false, dynamic-field, genuine-waiver, placeholder-waiver, positive-control,
  strict, and live-zero cases.
- Focused WC2+WC3: **31 passed**. QBT+WC3 after the adversarial fix: **61 passed**. Broad
  relevant shared-tree suite: **333 passed**, with two pre-existing `SyntaxWarning`s. Lane
  gate regression suite: **204 passed**.
- Ruff is green on every owned Python edit. The curriculum file was checked with its
  pre-existing `I001`, `UP037`, and `RUF100` whole-file debts ignored.
- Two real review passes cover the twelve changed Python files: correctness/dataflow, then
  adversarial resume/gate/storage. Receipt:
  `.omx/research/ddm_wc3_qbr1_ema_law_cure_review_receipt_20260903.json`.
- Developer preflight passed, examining **25 of 27** declared developer gates.
- Full `preflight_all --scope all --allow-slow-preflight` is
  `BLOCKED-ENVIRONMENT-CAPABILITY`, not green: the managed sandbox denied the real
  `ps -axo pid=,command=` call in `check_no_live_mcp_processes`. No mock, skip, or gate
  weakening was used.

The family exemplar pin requested by the charter is
`e09ba1f62ceb821b22646510b514588ecbd365ba` (`git log -1 --
src/tac/confound_gates.py` before this arm's unlanded work).

## Serializer and source custody

The serializer could not write the shared repository's Git object store. It emitted the
valid `BUNDLE_READY_MAIN_MUST_LAND` fallback commit
`106d0dd0a094dd4c289eba69c8d2c5124e13eb02`, based on
`97846a07bcaebf4883f0c7c3ca93b33b92eaaccf`. This is a real commit but is **not landed on
main**.

- Bundle:
  `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/serializer/20260903T172957.654787Z-32727/intended-commit.bundle`,
  27,995 B, SHA-256
  `821eab882751a1005b859d9071f91598470eca1e1271c8d665f987415db958de`.
- Format patch: same directory, `intended-commit.format-patch`, SHA-256
  `6670a615e02aaa5614768574aa947ef081eb3d0ad2adb9d8c9089f16357da42d`.
- Exact detached source checkout:
  `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2`.

The exact fallback tree's WC3 suite is green. Its broad confound suite has six unrelated
pre-existing failures that are repaired only by other dirty shared-tree work; those sibling
changes were deliberately excluded rather than absorbed. The staged shared index was not
touched.

## Re-seal and retained payloads

`MEASURED-BUILD`: the cured seal is rooted at source revision `106d0dd0a094...`, with QBT1
source SHA-256 `6eda9c202b3aee008d457373813ae07992e73902438cca114abb4c84bb8d980b`.
The authoritative retained handoff is:

- Build receipt:
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/BUILD_RECEIPT.json`, SHA-256
  `2ae8dab527331fd9b3a4b7b7b2531a01a74038b43521f1aab763f737f2a5562d`.
- Fire order:
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/SEALED_MAIN_FIRE_ORDER.json`, SHA-256
  `b418358157a8302c6d4d2d4091f630f551718aa3ff0697c1a5366741db35f3fa`.
- Initial EMA state:
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/initialized/qbr1_from_r10_ema_state.pt`,
  SHA-256 `991a1cc653c786affb607347def53b9dc91176e6ffd043f500076b5c35bf27b0`.
- Six sealed configs: seeds 20260902, 20260903, and 20260904 crossed with
  `control_native100` and `treatment_zero_native`; their hashes and exact detached launch
  commands are in the fire order.

The GT cache reverified at **5,078,017,610 B**, SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
At seal time AP had **32,131,645,440 B** free; six cells project to
**22,541,950,716 B**, leaving the required **8,589,934,592 B** reserve. Vertigo had
**196,486,438,912 B** free. The retained real B=16 parent measured peak RSS was
**2,572,632,064 B**. The old invalidated live-run directory was not written.

## Resume identity

| equality | result | reason / consumer |
|---|---|---|
| cursor | `NOT-MEASURED-WITH-REASON` | live smoke loads both frozen scorers; MAIN owns the scorer lane |
| live state | `NOT-MEASURED-WITH-REASON` | same boundary |
| EMA state | `NOT-MEASURED-WITH-REASON` | same boundary |
| archive | `NOT-MEASURED-WITH-REASON` | same boundary |

Source inspection disproved the charter's “scorer-free n=1” premise: the executable smoke
loads PoseNet and SegNet and uses a real **B=16** chunk. Per the charter's stop rule, this arm
did not run it. The fire order queues the exact comparison: interrupt after update 1, resume
to update 2, and compare with an uninterrupted two-update run, requiring cursor 2 and
bit-identical live, EMA, and archive hashes. Every checkpoint, history, training payload,
and archive goes under
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/resume_smoke`.

## WC2 unfinished rows

| row | status | finding |
|---|---|---|
| original EMA defect | `MEASURED-STATIC + DERIVED` | WC2-F1 remains burn-invalidating for the old config; machine alarm retained as `WC2_FIRE_ALARM.json` |
| default-off profiler identity | `MEASURED [macOS-CPU retained-tree read-only]` | PASS; 168 files, 257,214,191 B, tree SHA-256 `2ee47df26a0133884ed4aef004f7cbe48cf177a8f427d84df5a6adb8a52e0a8e`; callable identities unchanged |
| history fsync proxy | `MEASURED [macOS-CPU APDataStore scorer-free IO proxy]` | n=1000 mean 0.0086825621 s, p95 0.0349725829 s; not Metal timing |
| target-array loader proxy | `MEASURED [macOS-CPU APDataStore scorer-free IO proxy]` | n=100 mean 0.0044981013 s, p95 0.0105912914 s; not Metal timing |
| whole-cell timing projection | `DERIVED-FROM-R10` | 10,010 updates / 21,387.928 s = 2.1366561439 s/update; six finishes 17.8055 h, six BR2 realizations 0.80795 h, total 18.6134 h before build overhead |
| synchronized per-stage Metal timing | `NOT-MEASURED-WITH-REASON` | needs MAIN's Metal and scorer claims; timing wrapper is default-off and ready for the first cured cell |
| realized expected-flip loss | `VERIFIED-VIA-SOURCE-INSPECTION` | consumes logits from `roundtrip_to_camera_uint8_ste(rgb_pair_01)` followed by frozen `scorer_forward`; it is through the actual public R path |
| native expected-flip loss | `VERIFIED-VIA-SOURCE-INSPECTION` | consumes `outputs[class_logits]` before the public interface and is separately named and weighted |
| milestones / terminal realization | `VERIFIED-VIA-SOURCE-INSPECTION` | evaluate the EMA shadow through the same R+frozen-scorer path, retain realized outputs, and use HT-weighted n32 estimates; no new scorer number was produced |
| prior 78.71% native-render / net-repairer claim | `TRANSFERRED-PRIOR, NOT-REMEASURED` | `mst1` scope only; supports timing priority, not a QBR1 numerical verdict |
| historical 1.157x token-to-argmax multiplier | `DEAD TRANSFER` | later LB1 evidence refuted it by about 12x; it is not used in this seal or projection |
| WC2 staged point patch | `SUPERSEDED` | retained patch SHA-256 `059b663e68727784a71566d5e7b4cf9fa1edfe7b766cb2a764ca8f416b95fb35`; do not apply because Catalog #412's typed class cure replaces it |

WC2 evidence store:
`/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/`.

## MAIN fire order

The exact commands are arrays in `SEALED_MAIN_FIRE_ORDER.json` so shell quoting cannot drift.
MAIN must first reverify source pins and storage, acquire a unique scorer claim, and run the
queued resume smoke. On PASS, each cell requires unique live scorer and Metal claims and an
authorized retained copy of its sealed config. Each >30-minute cell launches through
`tools/launch_detached_process.py`, with measured peak RSS 2.3959503174 GiB, four threads,
an 18,000-second cap, and a durable `DONE.json`.

After all six complete, the fire order's `adjudication_argv` consumes the six exact
`RESULT.json` paths and writes
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/ADJUDICATION_RESULT.json`.
No result can be promoted before that paired three-seed adjudication.

## RECALL EVIDENCE

The bounded full-corpus recall searched `.omx/research/` memos and receipts, the canonical
equation registry, canonical research index/DAG surfaces, design/spec files, and task/lever
ledgers using the content queries `QBR1`, `EMA`, `warmup`,
`ema_decay_run_geometry_v1`, `terminal seed fraction`, `expected flip`, `realizer`,
`78.71`, and `1.157`.

Beyond the charter seeds, `.omx/state/lever_activation_ledger.jsonl` records the generic
`ema_target_seed_fraction` warmup formulation as retired: at target seed fraction 0.01 the
warmup crossover is about 1.954 run lengths. That changed the cure from “choose a safer EMA
default” to “make the registered constant law the canonical executable and isolate warmup as
an explicit ablation.” Recall also found that the 1.157x through-origin transfer was refuted
at LB1, so WC2's realization discussion does not use it. The 78.71% native-render claim is
retained only as an `mst1`-scoped prior. The canonical equation lookup confirmed
`ema_decay_run_geometry_v1` is the governing constant-decay law.

**Own-vehicle frontier:** UNMOVED — no new byte-closed archive or exact evaluator row. The
current effective frontier remains **afr1, S 0.14797617125559104 @ 180,002 B
[contest-CUDA T4, n600]**, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

## NEXT_IF_RESUMED

- `BUNDLE-READY-MAIN-MUST-LAND` — owner **MAIN**; consumer store is the shared Git repository plus the retained serializer directory; fire trigger is a context allowed to import/cherry-pick fallback commit `106d0dd0a094...` from the verified bundle without disturbing unrelated index/worktree state, followed by post-landing hash verification.
- `BLOCKED-ENVIRONMENT-CAPABILITY` — owner **MAIN**; consumer store is `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/preflight_all/`; fire trigger is an environment in which the real `ps -axo pid=,command=` probe is permitted, then rerun full `preflight_all` without mocks or skips and retain its receipt.
- `SEALED-BLOCKED-ON-MAIN-SCORER-LANE` — owner **MAIN**; consumer store is `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/resume_smoke`; fire trigger is a unique live scorer claim, then run `bounded_resume_smoke.argv` from the sealed fire order and require all four equality rows to PASS.
- `QUEUED-WITH-A-FIRE-ORDER` — owner **MAIN**; consumer store is `/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/metal_stage_profile/`; fire trigger is the first cured authorized cell plus unique scorer and Metal claims, then use `experiments/ddm_wc2_qbr1_stage_timing.py run-config AUTHORIZED_CONFIG_PATH --profile-stages --timing-output <consumer-store>/RESULT.json` under the same detached-launch contract.
- `SEALED-AWAITING-MAIN-LIVE-CLAIMS` — owner **MAIN**; consumer store is `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs`; fire trigger is cured resume PASS, source-pin revalidation, AP/Vertigo storage PASS, and unique live scorer and Metal claims; run the six ordered `cells[].launcher_argv` arrays, then the exact `adjudication_argv`.

## LIVE-HYPOTHESES

- The cured resume smoke will pass cursor, live-state, EMA-state, and archive equality because construction and restore now share the typed law and the scorer-free state-machine tests pass. This remains a hypothesis until MAIN executes the real scorer-dependent B=16 smoke.
- Removing the native proxy term may improve the realized fair-form endpoint in at least two of three paired seeds because the treatment removes a competing pre-public-interface objective while retaining the through-R objective. No treatment result exists yet.
- Frozen scorer plus realizer work likely dominates the cured cell's wall time, based on the source call graph and transferred R10/mst1 evidence. Only the synchronized stage profile can quantify that split.

## DEAD-ENDS

- The old literal-`warmup=True` seal is invalid as a constant-decay experiment; its affected cell cannot be promoted under the sealed treatment name.
- Applying WC2's staged one-site warmup patch is closed: it is superseded by the typed DSL, construction/restore alarm, and STRICT class guard.
- Treating resume identity as a scorer-free n=1 check is closed by source inspection; the real path loads both scorers and uses B=16.
- Mocking or skipping process inspection to manufacture a green full preflight is closed.
- Treating current main HEAD as the cured sealed source is closed until MAIN lands the serializer fallback commit and verifies its hashes.
