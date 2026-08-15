# DDM DR1 dispatch infrastructure repair — 2026-08-14

## Disposition

R1 and R3 are repaired with executed receipts. R2 is repaired and passes an exact isolated import-topology simulation, but its required Modal CPU container confirmation is `BLOCKED-EXTERNAL`: two bounded `modal run` attempts failed before the local entrypoint or any worker ran with `Could not connect to the Modal server.` No T4, scorer, or worker call was created. MAIN must not fire the real MT1 T4 order until the CPU import receipt is green.

This infrastructure unit did not run an exact evaluation and did not move either frontier. The live effective frontier remains MC36 Variant C, `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`; the own-vehicle frontier remains LC2, `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## R1 — seal versus signature drift

### Mechanism

The fire-order builder assembled a free-form shell string and never checked it against the function wrapped by Modal as `::main`. That allowed the sealed argv and the live entrypoint options to drift independently.

### Cure

`build_fire_order` now derives the accepted option set from the undecorated Python entrypoint with `inspect.signature`, parses the exact shell argv with `shlex`, rejects unknown/repeated/missing options, and requires both detach acknowledgements before persisting a seal. The local Modal help surface independently showed all six expected options, including required `--output-dir` and the two Boolean acknowledgements.

`reseal-fire-order` rehashes the sealed request and all nine retained payloads, preserves prior fire orders under `superseded/`, and replaces only the fire-order envelope. It does not materialize or alter payloads.

### Receipt

- Sealed request, unchanged: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/SEALED_REQUEST.json`, 9,124 bytes, SHA-256 `c9d6d62c8115f6c209576a57d4cbf7e40c2191c542473fa0df33bc82af91dffc`.
- Repaired fire order: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/SEALED_FIRE_ORDER.json`, 6,918 bytes, SHA-256 `e7527b7c8db2c33b45edb72fca603bb7e151fdf7c6812500b6429470f21ca206`.
- Reseal receipt: same directory, `DR1_RESEAL_RESULT.json`, SHA-256 `5941c0114638e95491bdb528f602d26d271fe70ec1395737dc2eea11ac25dd68`.
- Original fire order retained byte-identically: `superseded/SEALED_FIRE_ORDER.892cf1e0a6d43682dd448b6d58bbefb16fbec64506f8ab79e116e03620fab75c.json`.
- Negative controls: removing `--output-dir` is rejected as a missing required option; removing `--provider-detach-ack` is rejected as a missing acknowledgement.

## R2 — container import drift

### Reproduced mechanism

An isolated tree matching the failed mount topology reproduced the receipt. MT1 was mounted top-level, JS1B was mounted top-level, and only `experiments/__init__.py` existed under `/workspace/pact/experiments`. The top-level JS1B import entered JS1B, whose package import of `experiments.modal_auth_eval` failed; MT1's broad `except ModuleNotFoundError` caught that transitive dependency miss and incorrectly changed routes to `from experiments import ddm_js1b...`, where no package-mounted JS1B file existed. The observed terminal error was therefore a partial mount plus an over-broad fallback, not a broken package initializer.

Commit `88dc45548f` did not change `experiments/__init__.py`; it changed preflight package resolution. The initializer still satisfies ordinary Python submodule import behavior when the submodule file actually exists under the package.

### Cure

MT1 now uses one explicit absolute package route through `importlib.import_module`. Its worker image mounts both `experiments/modal_auth_eval.py` and `experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py` at their package paths and no longer relies on a mixed top-level/package dependency topology. Direct-script execution inserts the real repository root; the worker selects `/workspace/pact` when that mounted package exists.

The image also exposes a CPU-only `smoke_import_and_seal` entrypoint. Its worker imports both package modules, hash-parses the exact sealed request, reports module paths and payload count, requests no GPU, and does not load a scorer.

### Positive controls and boundary

- Local direct import: green.
- Exact isolated mount simulation with only the package-mounted initializer, JS1B, and `modal_auth_eval` plus top-level MT1: green; it resolved `experiments.ddm_js1b_modal_cuda_argmax_field_materializer` and `experiments.modal_auth_eval`, then passed the live fire-argv signature check.
- `modal run ...::main --help`: green; exact current options parsed.
- Required Modal CPU worker smoke: not measured. Two bounded attempts failed before entrypoint execution with provider connectivity failure. Durable blocker receipt: `/Volumes/VertigoDataTier/pact/ddm_dr1_dispatch_infra_repair_20260814/retained/mt1_import_smoke/IMPORT_SEAL_PARSE_SMOKE.json`, SHA-256 `a60717513c0b104366a457b26b99d0a9c3f3d112ce9105a97e2365c2274ec634`.

### Sibling sweep

Bounded census: 28 `experiments/*.py` files containing `.spawn(` or `.remote(`. Twelve contain an `experiments` package import; the eight DDM dispatcher surfaces sharing this family were inspected below. Did not find another dispatcher in this census with MT1's exact combination of a `base_image`, a package initializer mount, top-level-only dependency sources, and a transitive `ModuleNotFoundError` fallback.

| Dispatcher | Import/mount form | Disposition |
|---|---|---|
| `ddm_mt1_modal_multitoken_sign_gate.py` | Formerly mixed top-level/package dependencies on `base_image` | Fixed: explicit package imports and package-path mounts |
| `ddm_ec2_modal_oriented_adapter_trainer.py` | Bare-first fallbacks, but derives `eval_image`; explicitly mounts initializer and top-level JS1B | No matching partial-mount defect found |
| `ddm_js1b_modal_cuda_argmax_field_materializer.py` | Package-first `modal_auth_eval`, top-level fallback; derives `eval_image` | Both named import surfaces are supplied by the parent image |
| `ddm_po1_modal_t4_pose_feedback.py` | Same `modal_auth_eval` fallback; derives `eval_image` | Both named import surfaces are supplied by the parent image |
| `ddm_re1t_modal_t4_sign_gate.py` | Package/bare JS1B and auth fallbacks; derives `eval_image` | Top-level dependency sources are explicitly supplied |
| `ddm_sa1_modal_t4_sign_gate.py` | Same family as RE1T; derives `eval_image` | Top-level dependency sources are explicitly supplied |
| `ddm_qs1_modal_t4_dual_axis.py` | Package/bare substrate fallback; derives RE1T `gate_image` | Inherits RE1T/eval image closure; no `base_image` split |
| `ddm_vd1_modal_batch_event_validator.py` | Package auth import with documented top-level fallback; derives `eval_image` | Parent image supplies the documented top-level module |

## R3 — closer rc=1 at 782 seconds

### Mechanism

The charter's initial timing hypothesis was refuted by the full receipt. The poller did keep polling. At 782.5 seconds it received a terminal provider result with `returncode=0`, then the closer correctly refused mutation because call `fc-01M014B5F4DB6FJ5BSXXRGNB83` had no canonical call-ID ledger row. The source defect was in both canonical auth dispatchers: after `.spawn()` they wrote local spawn metadata but never called the canonical call-ID registration writer.

### Cure

Both `experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py` now call `register_dispatched_call_id_fail_closed` immediately after resolving the call ID and before writing spawn metadata. The record includes lane, axis, recipe, hardware, source commit, archive identity, and pair group.

The canonical closer gained one bounded legacy path for already-completed calls: if and only if the call row is absent, it may authenticate the exact spawn receipt against call ID, lane, instance job, agent, platform, schema, app/tool pair, axis, archive hash, and local request; it then uses the canonical writer to reconstruct the missing dispatched row. A mismatch remains a typed refusal. A separate supplied closure manifest can adapt a pre-AC1 result without weakening the normal manifest requirement.

### Executed positive control

The closer ran against completed F26R call `fc-01M014B5F4DB6FJ5BSXXRGNB83` and returned `CLOSED`, `process_rc=0`:

- authenticated the historical CPU spawn receipt;
- added one `dispatched` row and one `harvested` row to the canonical call-ID ledger;
- observed the lane claim already terminal and made no claim mutation;
- classified the real remote result by its zero return code;
- verified the exact retained 578-byte inflated-output manifest, SHA-256 `181549a1098812a8dc8be3bb2bcaac3ee7fb090bc97a7a9b60a9458a0d925a0f`;
- on an idempotent rerun, kept the call ledger at exactly two rows: `dispatched`, `harvested`.

Receipt: `/Volumes/VertigoDataTier/pact/ddm_dr1_dispatch_infra_repair_20260814/retained/f26r_closer_positive_control/closure/ENDPOINT_CLOSURE.receipt.json`, SHA-256 `8e995dcb30d69b0f9acc5c15e87117134e71acdf261107b7c080651f2a609533`.

The legacy manifest adapter is `.omx/research/ddm_dr1_20260814/F26R_CLOSURE_MANIFEST.json`; it names only the volume-backed artifact already recorded in the returned F26R result.

## Verification

- `51 passed, 1 deselected` across the MT1, endpoint-closer, and call-registration suites.
- Ruff: all changed Python files pass.
- `py_compile`: all changed Python production files pass.
- `git diff --check`: pass.
- The one deselected live-repository call-registration census failure is pre-existing unrelated dirty work in `src/tac/canonical_anti_patterns/pattern_matcher.py:678`; the focused regression proves both canonical auth dispatchers register between `.spawn()` and `write_spawn_metadata`.
- No T4 or scorer work ran. No archive or score was produced.

## RECALL EVIDENCE

Searched `.omx/research/`, arm final messages, state/task stores, the canonical equation list, canonical research index/DAG surfaces, Git history, and the live dispatcher/closer sources with the queries `ddm_mt1`, `SEALED_FIRE_ORDER`, `fc-01M01AJV3VNSZT8V51FCXT4F2G`, `fc-01M014B5F4DB6FJ5BSXXRGNB83`, `modal_endpoint_close`, `call_id_ledger`, `experiments.modal_auth_eval`, `add_local_python_source`, `#936`, and `88dc45548f`.

Beyond the charter seeds, the recall found: FS1 commit `9778fb0679` deliberately moved MT1 to the pre-mount `base_image`; AC1's poll/ledger separation makes local deadlines nonterminal; DT1's dependency closure covered worker imports but not the dispatcher image's Python package topology; and the canonical equation `modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1` records prior local/worker path drift. These changed the plan from editing `experiments/__init__.py` or extending the poll deadline to repairing the image package closure, adding a worker import smoke, registering call IDs at the true spawn boundary, and keeping legacy recovery receipt-authenticated.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_dr1_dispatch_infra_repair_20260814/retained/mt1_import_smoke/IMPORT_SEAL_PARSE_SMOKE.json`; fire trigger: Modal connectivity is restored; exact argv: `.venv/bin/modal run experiments/ddm_mt1_modal_multitoken_sign_gate.py::smoke_import_and_seal --sealed-request /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/SEALED_REQUEST.json --expected-request-sha256 c9d6d62c8115f6c209576a57d4cbf7e40c2191c542473fa0df33bc82af91dffc --output-receipt /Volumes/VertigoDataTier/pact/ddm_dr1_dispatch_infra_repair_20260814/retained/mt1_import_smoke/IMPORT_SEAL_PARSE_SMOKE.json`; require `passed=true`, package module paths under `/workspace/pact/experiments`, `gpu_requested=false`, and `scorer_loaded=false`.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1`; fire trigger: the CPU container receipt above is green, the MT1 T4 lane is free, and `SEALED_REQUEST.json` plus all nine payload hashes and repaired dispatcher source match `SEALED_FIRE_ORDER.json`; then consume that file's exact argv without editing it.

## LIVE-HYPOTHESES

- The repaired MT1 image will pass the real CPU container import smoke once Modal is reachable, because the exact mixed-topology failure was reproduced locally and the repaired isolated tree resolves only the package paths actually mounted in the worker image. This remains untested in a provider container.
- Immediate call-ID registration will eliminate the F26R missing-ledger closer class for future canonical CPU and CUDA auth dispatches, because registration now occurs at the first point after a provider call ID exists and before fallible local metadata writes. This is unit-tested but awaits the next real detached auth dispatch.
- The queued MT1 T4 component sign gate may confirm the locally negative sign without starting the second train, because the unchanged request and nine byte-identical inputs preserve the original experiment while only dispatch infrastructure changed. No T4 evidence was collected here.

## DEAD-ENDS

- Editing `experiments/__init__.py` is closed: commit `88dc45548f` did not modify it, ordinary package submodule imports work when files are mounted, and the failure was the partial mount plus transitive exception capture.
- Extending the closer deadline or changing timeout handling is closed: the poller received a terminal `rc=0` result at 782.5 seconds; missing call-ID registration caused the refusal.
- Treating local import simulation as the required container receipt is closed: it is a useful positive control but not provider-container evidence.
- Repeating Modal connection attempts without an external connectivity change is closed after two identical pre-entrypoint failures; neither attempt issued a call ID.
- Firing the real T4 sign gate in this arm is closed by charter; MAIN owns that fire after the CPU import receipt is green.
