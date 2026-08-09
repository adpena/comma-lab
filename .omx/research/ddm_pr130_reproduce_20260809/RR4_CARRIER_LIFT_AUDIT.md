# RR4 carrier and lift fidelity audit

**Date:** 2026-08-09

**Axis:** scorer-free source/receipt audit, plus previously preserved `[macOS-MPS advisory]` receipts; no scorer forward and no contest score

**Base under audit:** PR130 CPR1 `S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, 191,052 B, archive SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`

**Intake pin:** `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo` at `e34f31bc4969042c0051ac81aa3c56884419a231`

**Lift provenance pin:** `2f94596bb0136d342254022a5c9584756eae0468`

## Conclusion

The lift is not a clean header-only custody copy. Of 13 Python files in the two lifted trees, 0/13 are whole-file byte-identical to the intake, 10/13 have only declared adaptation, and 3/13 contain silent post-lift behavior changes. All 12 files that claim an intake source SHA-256 name the correct original bytes, but those hashes do not authenticate the current vendored bodies. The three silent files still say that our only change is the accounting header even though commit `f06c8493f2` added an admission-guard call to each.

The pose port's dense adapter preserves the borrowed row-local Adam equations in the tested two-step CPU envelope. The active wrapper nevertheless selects that adapter unconditionally on MPS, with no flag, warning, fallback event, result field, or pinned-runtime assertion. This is a silent mechanism selection. It contradicts the preserved native-sparse PASS, which says the reference mechanism is available on MPS under Torch 2.10.0.

The round-1 statement that `2.4437744286842644e-05` had no locatable receipt is now falsified in the bounded scopes searched here. The exact value is present in an SSD result as the step-6 history mean over all 600 pairs. It is a six-step `[macOS-MPS advisory]` row produced by the dense adapter, not the owed 4,000-step training result, not an exact archive measurement, and not score authority. Its result file omits the runtime version and optimizer-path provenance needed to resolve those questions later.

No finding moved the PR130 base or any contest pointer.

## Ranked findings

### 1. High — three lifted trainers have silent drift

**Verdict scope:** INSTANCE, the 13 current files at Pact HEAD versus their named intake originals at `e34f31bc4969042c0051ac81aa3c56884419a231`.

`train_semantic_full.py`, `train_semantic_quantized.py`, and `pose/lifted/train_pose_carrier_full.py` each add `assert_governed_admission(...)` after argument parsing. The edits are real governance guards, not a change to PR130's scientific update equations, but their headers say `ours: This accounting header only` or `vendored custody copy only`. They therefore violate the declared borrowed-substrate boundary. Commit `f06c8493f2` introduced all three edits after the lift.

**Why it matters:** a consumer trusting the header or `source_sha256` cannot tell that executing the vendored file is no longer executing the intake body. The hash names the original source, not the current copy.

**Falsifier/cure:** either restore body identity or declare each admission guard in its accounting header, then add a test that reconstructs the expected body from the pinned intake and permits only an explicit adaptation patch. A manifest-literal test is not a falsifier.

### 2. High — the wrapper silently substitutes the dense mechanism on every MPS run

**Verdict scope:** FORMULATION, current `build_row_local_coefficients` dispatch in `pose/mps_port.py` and its sole carrier-wrapper consumer; not a kill of dense row-local adapters or native sparse MPS.

The adapter body is mechanism-preserving within the tested local envelope: it uniquifies selected row IDs, increments each selected row clock once per step, keeps `exp_avg` and `exp_avg_sq` per row, uses the borrowed bias corrections, refuses gradients on undeclared rows, and requires a fresh declaration each step. The two-step repeated-ID CPU control checks weights and optimizer state against `RowLocalSparseAdam` exactly.

The selection logic is the defect. `use_sparse = device.type != "mps"` makes `RowLocalDenseAdam` mandatory for every MPS execution. There is no CLI choice, fallback-on-exception, warning, or telemetry. The wrapper's result schema does not record optimizer class, sparse/dense mode, fallback environment, Torch version, or the native-probe receipt.

This is now contrary to the stronger receipt `/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json` (SHA-256 `32ce0585d070fd578bea563f94b33fffe6e000b8cc608f827d4fcb5319893ec3`): native sparse `nn.Embedding`, coalesced COO gradient, and borrowed `RowLocalSparseAdam` passed for 2 steps over 4 rows on real MPS at pinned Torch 2.10.0, with row clocks `[1,2,1,2]`, untouched rows bit-identical, CPU/MPS parity within `atol=2e-6, rtol=2e-5`, and zero CPU-fallback warnings.

**Falsifier/cure:** make native sparse the explicit reference mode on the pinned runtime, expose any adapter use as a named opt-in or loud fallback, and persist runtime, optimizer, fallback, and receipt identity in every result and checkpoint. Then compare uninterrupted reference and adapter trajectories beyond the present two-step control.

### 3. High — the missing six-step n600 receipt exists, but it does not close the 4,000-step claim

**Verdict scope:** INSTANCE, one six-step full-population MPS run. Its `FAIL` is a run-threshold outcome, not a pose-carrier-family verdict.

Located receipt:

- Result: `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/reports/METAL_SMOKE_carrier.json`
- Result SHA-256: `0c85e4a31928361e4f3977cd6365569937ea10a78d31e9b7bb5fa740e4d5ec6f`
- Result size/time: 26,152 B; `2026-08-09T08:21:37-0500`
- Step-6 full-state checkpoint: `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/checkpoints/METAL_SMOKE_carrier.step000006.full_state.pt`
- Checkpoint SHA-256: `abd09ca0f030fcfe3e688adcc532ebc34f107774fa3d8d34055effbfa01ca9a3`
- Checkpoint size/time: 362,405 B; `2026-08-09T08:21:34-0500`

The result declares `device=mps`, `steps=4000`, `stop_after_step=6`, `smoke_pairs=null`, and active pair IDs 0 through 599. Its history contains:

| step | phase | denominator | mean d_pose | status |
| ---: | --- | ---: | ---: | --- |
| 3 | `full_quantized` | 600/600 | `2.3431171939591877e-05` | best row retained at top level |
| 6 | `full_quantized` | 600/600 | `2.4437744286842644e-05` | the previously withdrawn exact value |

Because the active wrapper always chooses dense on MPS, this run used `RowLocalDenseAdam`. The JSON does not persist Torch version, optimizer kind, fallback policy/warnings, git SHA, exact argv/command, or safe-run receipt identity. `LOCAL_TRAINING_AUDIT.md` says the run used `tools/safe_run.py`, took 70.72 seconds, and peaked at 3,397 MiB, but I did not find a durable governor receipt or log in the run directory or the searched repository surfaces.

**Boundary:** this restores a six-step, n600, `[macOS-MPS advisory]` training-history measurement only. It does not prove 4,000-step convergence, sparse-reference training, resume equivalence, byte-closed carrier packing, exact `upstream/evaluate.py`, or a contest score.

**Falsifier/cure:** a governed, pinned-runtime 4,000-step run must preserve per-stage/periodic atomic checkpoints and record the optimizer path and runtime in its receipt. An exact archive parse-back and axis-matched evaluator run remain separate gates.

### 4. Medium-high — resumability captures broad state but is not crash-atomic or trajectory-proven

**Verdict scope:** FORMULATION, the current resumable wrapper and its current tests; not a claim that its serialized fields are insufficient in every failure mode.

`_save_full_state` captures model tensors, both optimizer states, both scheduler states, generator state, order/cursor, sampling weights, history, best state, active IDs, and args. It also preserves step-encoded filenames. That is substantive behavior.

The writes are direct `torch.save(payload, path)` and `torch.save(payload, latest)` calls. The periodic/best/final result saves are direct as well. There is no temporary file plus atomic rename, so an interruption can corrupt both a step artifact and the latest pointer. This violates the repository's binding atomic-checkpoint rule.

The state unit test restores values into stock Adam optimizers; it does not exercise the actual row-local optimizer or prove that an interrupted trajectory is bit-identical to an uninterrupted trajectory. The earlier two-step resume smoke proves that continuation executes, not equivalence.

**Falsifier/cure:** atomic writes for every checkpoint/result, followed by an uninterrupted-versus-resumed comparison using the actual row-local optimizer, scheduler, RNG/order/cursor, and a repeated-ID batch sequence.

### 5. Medium — the tests are behavior-heavy, but miss the load-bearing provenance and dispatch seams

**Verdict scope:** FORMULATION, 27 test functions in the five current `src/tac/pr130_lift/tests/test_*.py` modules.

The charter named three modules; the live tree has five, and all five were classified.

| module | tests | behavior | constants/structure | what the behavioral tests actually exercise |
| --- | ---: | ---: | ---: | --- |
| `test_mx1_pr130_lift.py` | 11 | 8 | 3 | lifted forward/curriculum, determinism, journal resume, schedule, selection, tail averaging, EMA, governed wall cap |
| `test_mx2_pose_adapter_caches.py` | 1 | 1 | 0 | master-cache second-frame extraction |
| `test_mx2_pose_lift.py` | 5 | 3 | 2 | CPR1 symbol round-trip, non-compatible rejection, generic lossless round-trip |
| `test_mx2_pose_resumable_state.py` | 3 | 3 | 0 | state value restoration and admission guard behavior |
| `test_pq1_pose_mps_port.py` | 7 | 6 | 1 | dense/reference two-step equality, guard failures, cache dispatch, CPU-first load, sparse CPU worker |
| **Total** | **27** | **21** | **6** | **77.8% behavior / 22.2% constants or structure** |

This is not a constants-only suite: replacing the implementation bodies with markers would fail 21/27 tests. The six constants/structure tests are the two MLX availability/import probes, the live-ticket structure check, the manifest literal check, the mocked MLX-missing probe, and the wrapper source-string check.

Load-bearing omissions remain:

- no test recomputes every source SHA and body diff against the intake, so 3/13 silent drifts pass;
- the manifest test asserts only two literal hashes and never authenticates vendored bodies;
- no test requires native sparse after the pinned MPS PASS or requires an optimizer-mode receipt;
- no consumer test fails closed on Torch other than 2.10.0;
- no crash-atomic write test;
- no uninterrupted-versus-resumed trajectory comparison using the real row-local optimizer;
- no real-frame MLX/PoseNet pose parity or pose training-convergence test.

Current-turn execution did not yield a clean fresh suite verdict. `PYTHONPATH=src .venv/bin/python -m pytest src/tac/pr130_lift/tests -q` reached its test summary with five environment/import-time failures, then was interrupted during teardown (exit 130). Torch/SymPy imports repeatedly exceeded the test time limits, and teardown reported a missing `.venv/.../curlify-3.0.0.dist-info/entry_points.txt`. A standalone Torch/SymPy import also hung until interrupted. Earlier preserved receipts report 27/27 passing before this environment state; I did not relabel that historical pass as current verification.

### 6. Low — the three OURS artifacts are honestly scoped, but remain unraced

**Verdict scope:** INSTANCE, these three current implementations versus the OTS PR130 baseline; no family kill.

| OURS artifact | claim audit | measured evidence | honest status |
| --- | --- | --- | --- |
| `mlx_semantic_renderer.py` | Says “MLX port” and explicitly says no score authority; it does not claim exact equivalence or a win. | Real n4 forward receipt: raw-frame max-abs `0.011199951171875` on `[0,255]`; 180/786,432 scorer argmax pixels differ; one-step loss differs by about 1%; gradient parity not measured. | **UNRACED versus OTS; admitted only as a training substrate.** |
| `pose/mlx_pose_carrier.py` | Says local training/parity surface, not scorer authority; “follows the shape” is not an equivalence claim. | No real-frame MLX/PoseNet parity, gradient parity, training convergence, or OTS A/B. | **UNRACED and parity-blocked.** |
| `pose/repack_race.py` | Claims only an exact, lossless CPR1 applicability gate and records `score_claim=false`; no better-than-OTS claim. | Byte-only race covered 17/17 banked pose-adjacent sections; CPR1 applied to 0/17 because none had the legacy layout. Generic codes round-tripped. | **HONEST BOUNDED NEGATIVE; UNRACED versus PR130's shipped repack.** |

No NO-FAKE #8 surrogate equivalence/better claim was found in these files. The OTS path remains the PR130-base path; OURS has not earned admission through an A/B.

## Per-file drift census

Method: for every Python file in `src/tac/pr130_lift/lifted/` and `src/tac/pr130_lift/pose/lifted/`, resolve the declared `source_path`, read that file from intake commit `e34f31bc4969042c0051ac81aa3c56884419a231`, recompute its SHA-256, and compare both whole bytes and the vendored body after removing the declared accounting header. The intake remained read-only.

`source_head=2f94596...` is a valid lift-time provenance pin, while `SOURCE_REPO_HEAD=e34f31...` is the current executable intake pin. The source files involved are unchanged between those intake points; the separation itself is not a defect.

| lifted file | declared intake source | declared SHA vs source | body comparison | classification |
| --- | --- | --- | --- | --- |
| `lifted/evaluate_semantic_quantization.py` | `code/evaluate_semantic_quantization.py` | MATCH `5bbd2136...` | exact after declared header | drifted-with-declared-adaptation |
| `lifted/semantic_renderer_oracle.py` | `code/semantic_renderer_oracle.py` | MATCH `2bf3a6a8...` | exact after declared header | drifted-with-declared-adaptation |
| `lifted/train_semantic_full.py` | `code/train_semantic_full.py` | MATCH `2d7a3575...` | extra admission import/call | **drifted-SILENTLY** |
| `lifted/train_semantic_quantized.py` | `code/train_semantic_quantized.py` | MATCH `4bcaf8a5...` | extra admission import/call | **drifted-SILENTLY** |
| `pose/lifted/__init__.py` | none; local package initializer | N/A | declared local scaffold | drifted-with-declared-adaptation |
| `pose/lifted/carrier_codec.py` | `code/carrier_codec.py` | MATCH `d2f14402...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/learned_pose_carrier_oracle.py` | `code/learned_pose_carrier_oracle.py` | MATCH `59a574bf...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/pack_semantic_pose.py` | `code/pack_semantic_pose.py` | MATCH `151413b0...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/pose_basis_oracle.py` | `code/pose_basis_oracle.py` | MATCH `90909bdd...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/refine_pose_coeff_codes.py` | `code/refine_pose_coeff_codes.py` | MATCH `8912464b...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/repack_carrier.py` | `code/repack_carrier.py` | MATCH `df6bdaa2...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/search_pose_coeff_cpu.py` | `code/search_pose_coeff_cpu.py` | MATCH `42b0f57c...` | exact after declared header | drifted-with-declared-adaptation |
| `pose/lifted/train_pose_carrier_full.py` | `code/train_pose_carrier_full.py` | MATCH `684a4906...` | extra admission import/call | **drifted-SILENTLY** |

Denominators:

- whole-file byte-identical: **0/13**;
- drifted with declared adaptation: **10/13**;
- drifted silently: **3/13**;
- original-backed source hashes correct: **12/12**;
- original-backed bodies exact after the declared header: **9/12**;
- original-backed bodies with undeclared edits: **3/12**.

## Carrier trainer device-site sweep

The active direct-import surface comprised the 467-line intake `train_pose_carrier_full.py` plus `learned_pose_carrier_oracle.py`, `pose_basis_oracle.py`, `semantic_renderer_oracle.py`, and conditionally imported `evaluate_semantic_quantization.py`: **5 files / 1,415 lines**, supplemented by the actual upstream PoseNet/FastViT modules.

### Direct device findings

| intake site | classification | port consequence |
| --- | --- | --- |
| trainer line 264 `torch.cuda.empty_cache()` | unconditional CUDA API call | real required edit; wrapper correctly replaces it with device-specific cache clearing |
| trainer line 269 `load_file(..., device=str(device))` | external direct-to-device load; becomes MPS when selected | avoidable compatibility/provenance risk; wrapper correctly loads on CPU then moves the model |
| trainer line 220 `--device` default `cuda` | overridable CLI default | not a hard pin; MPS works only when explicitly selected |
| trainer line 266 `.to(..., non_blocking=True)` | CPU masters are not pinned | no correctness change found; likely a performance no-op rather than asynchronous transfer |

Within this 5-file active surface I found **0** `.cuda()` tensor/module calls, literal CUDA tensor allocations, pinned-memory loaders, CUDA streams/events, NCCL, custom CUDA/fused kernels, AMP/autocast/GradScaler in the trainer, or CUDA-only dtype assumptions. Some imported scripts have conditional CUDA defaults or standalone-main autocast, but those `main()` paths are not called by the carrier trainer. This bounded negative does not cover the nine other non-trainer stages in the 24-stage pose leg.

PP2's 60-family census originally left safetensors direct-to-MPS, sparse Embedding backward, and sparse COO optimizer handling unknown. The wrapper structurally removes the first; the pinned native sparse receipt closes the latter two for its 2-step/4-row scope.

### Runtime drift

`tools/probe_sparse_mps.py` fails closed unless Torch is exactly 2.10.0. `train_pose_carrier_full_resumable.py` has no equivalent assertion. The repo environment currently identifies as Torch 2.12.1 from package metadata, so a normal wrapper invocation can silently answer a different runtime question. The six-step result does not record its Torch version. This is a FORMULATION-scoped provenance failure in the current wrapper, not evidence that 2.12.1 changes the numerical answer.

## `source_loader.py` and repack boundaries

`source_loader.py` provides a temporary flat-import path and dynamically loads the vendored pose module. It restores `sys.path` after use, but it does not authenticate the loaded bytes against `vendor_manifest.json` or the intake. Consequently its behavior is compatible with all three silent drifts.

`repack_race.py` is fail-closed on format applicability: it admits CPR1 only at the exact legacy length and verifies every decoded symbol. Its 0/17 result is scoped to the 17 banked sections searched; it is not evidence that CPR1 cannot beat an OTS repack on a genuine PR130 legacy carrier.

## RECALL EVIDENCE

Searched beyond the charter seeds:

- full `.omx/research/` content for `RowLocalSparseAdam`, `train_pose_carrier_full`, `pr130_lift`, `mlx_pose_carrier`, `repack_race`, and the exact value `2.4437744286842644e-05`;
- `.omx/state/main_hot_state.md`, `.omx/state/probe_outcomes.jsonl`, the PR130 ledger, and task `#995` surfaces;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, and the canonical equation registry through `tools/list_canonical_equations.py --json`;
- lift history and current tests by source content, not only file names;
- `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/` and `/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/` for durable receipts.

Findings beyond the charter seeds that changed the plan:

1. `LOCAL_TRAINING_AUDIT.md` and the SSD result contain the supposedly missing exact value, so this audit restores it with a narrower six-step disposition instead of repeating the withdrawal.
2. The native sparse MPS PASS makes the current always-dense MPS dispatch a live silent-selection defect, not merely conservative compatibility code.
3. The MAIN-host MLX semantic receipt narrows its status to training-substrate parity with nonzero argmax disagreement and unmeasured gradient parity.
4. The mx2 receipt closes the repack denominator at 17/17 inspected sections and confirms 0/17 CPR1 applicability.

I did not find a canonical equation or DAG result in the searched registry/index scopes that settles end-to-end adapter trajectory equivalence, the full 4,000-step carrier run, or MLX pose parity. Those are bounded absences, not claims of global nonexistence.

## Could not check

- No Metal device is available to this arm, so I did not execute MPS/MLX. I used preserved real-Metal receipts only within their stated scopes.
- I did not run a scorer, evaluator, archive build, CUDA job, or full-n600 score job. This charter did not own a scorer slot.
- The 4,000-step carrier run has not been completed in the located artifacts; only steps 3 and 6 were measured over n600.
- I did not audit the nine pose-leg stages outside the trainer/import surface, so there is no whole-pose-leg portability verdict here.
- I did not recover a durable safe-run log/receipt for the six-step carrier run in the searched run/repository scopes.
- The current repo virtual environment did not provide a clean fresh test execution, as detailed above. I did not repair or mutate that shared environment.

## Landing status

The required serializer was invoked with the report as the sole explicit file, a post-edit SHA guard, `base=new`, the required `[no-triality] [p0-ledger-ok]` tags, no attribution trailer, and the Markdown-only review override. Git refused before staging with `unable to create temporary file: Operation not permitted` and `failed to insert into database`. This checkout's Git metadata is read-only to the arm. The report remains an untracked workspace artifact, the shared staged index remains untouched, and no commit claim is made.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER — lift custody cure.** Owner: MAIN / PR130 lift custodian. Consumer store: `src/tac/pr130_lift/tests/` and `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md`. Fire trigger: before the next claim that the vendored trainers are header-only or before executing one of the three silent copies. Declare the guards and add a complete pinned-body verifier.
- **QUEUED-WITH-A-FIRE-ORDER — reference optimizer selection and provenance.** Owner: MAIN / task `#995` pose-port owner. Consumer store: `.omx/state/probe_outcomes.jsonl` plus the next SSD carrier result. Fire trigger: before the next MPS carrier launch. Default to the validated native sparse path on pinned Torch 2.10.0; make adapter selection loud and persist runtime/optimizer/fallback fields.
- **QUEUED-WITH-A-FIRE-ORDER — atomic resume equivalence.** Owner: MAIN / task `#995` pose-port owner. Consumer store: `src/tac/pr130_lift/tests/` plus `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/`. Fire trigger: before authorizing the full 4,000-step run. Make every save atomic and compare uninterrupted versus resumed real row-local trajectories.
- **QUEUED-WITH-A-FIRE-ORDER — full carrier training.** Owner: MAIN / task `#995`. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/` and `.omx/state/probe_outcomes.jsonl`. Fire trigger: the three preceding gates pass, storage preflight passes, and governed launcher custody is ready. Resume or launch the 4,000-step n600 run with periodic distinct checkpoints.
- **FOLDED — repeat the native sparse feasibility probe.** Owner: MAIN / task `#995`. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json`. Fire trigger for reactivation: Torch, OS, sparse operator family, or trainer mechanism changes. The existing 2-step/4-row PASS already answers current feasibility.
- **FOLDED — promote any OURS optimization.** Owner: PR130 optimization campaign. Consumer store: the future OTS-versus-OURS A/B ledger. Fire trigger for reactivation: the OTS carrier baseline is reproduced and a named same-object metric/byte race is preregistered. No present OURS artifact has won that race.

## Frontier impact

No score was measured and no archive was changed. The live base remains **PR130 CPR1 `S = 0.172141297491896447` at 191,052 B `[contest-CUDA, DALI GT, n600]`**. This audit does not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN / PR130 lift custodian. Consumer store: `src/tac/pr130_lift/tests/` and the PR130 OTS/port ledger. Fire trigger: before executing or describing any of the three silent lifted trainers; reconcile the headers and enforce complete pinned-body verification.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN / task `#995` pose-port owner. Consumer store: `.omx/state/probe_outcomes.jsonl` and the next SSD carrier receipt. Fire trigger: before any MPS carrier launch; pin Torch 2.10.0, select native sparse by default, and record optimizer/fallback/runtime provenance.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN / task `#995` pose-port owner. Consumer store: `src/tac/pr130_lift/tests/` and `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/`. Fire trigger: before a 4,000-step launch; make all saves atomic and prove uninterrupted/resumed equality with the real optimizer.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN / task `#995`. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/` and `.omx/state/probe_outcomes.jsonl`. Fire trigger: custody, optimizer, runtime, atomic-resume, storage, and governor gates are green; complete the full 4,000-step n600 carrier run.

## LIVE-HYPOTHESES

- The native sparse PR130 mechanism will train on MPS without the dense adapter because the exact sparse Embedding/COO/row-clock path already passes on the pinned runtime; only a full governed trajectory can show whether its accumulated numerics remain acceptable.
- The dense adapter may remain a valid portability fallback because its two-step repeated-ID CPU trajectory is exactly equal to the borrowed optimizer and its body preserves per-row clocks; longer uninterrupted/resumed and MPS comparisons are still needed.
- The step-3 carrier state may be a better early checkpoint than step 6 because its n600 quantized mean is lower, but six steps are far too early and the run lacks enough provenance to infer a training optimum.
- A strict source-body verifier will prevent this drift genus because all three silent changes share the same shape: a valid governance edit landed after custody headers and hashes were frozen.

## DEAD-ENDS

- Do not re-search for `2.4437744286842644e-05` as globally absent. The exact value is in the six-step SSD result; the honest correction is to scope it, not withdraw it.
- Do not repeat the pinned native sparse two-step feasibility probe on unchanged Torch 2.10.0. It already passed; the next unknown is the real training trajectory.
- Do not treat the dense adapter's docstring or the two-step CPU test as end-to-end equivalence. They close only the local update mechanism in the tested envelope.
- Do not infer vendored-body fidelity from `source_sha256` or `vendor_manifest.json`. Those authenticate original intake bytes, while three current bodies contain undeclared edits.
- Do not promote the MLX semantic renderer, MLX pose carrier, or repack helper as better than OTS. None has won a same-object A/B, and the pose MLX path lacks real-frame parity.
- Do not call the six-step result the 4,000-step carrier run, an archive result, or a score. It is `[macOS-MPS advisory]` training history from the silently selected dense path.
