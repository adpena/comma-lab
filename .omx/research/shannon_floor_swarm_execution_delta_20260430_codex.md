# Shannon Floor Swarm Execution Delta - 2026-04-30

Timestamp: 2026-04-30T21:49:11Z
Author: Codex

Source-control plane:

- `grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `council_paradigm_shift_round{1,2,3}_20260430.md`
- `grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- `shannon_floor_execution_readiness_20260430_codex_progress.md`
- `grand_council_live_telemetry_kill_audit_20260430_codex.md`

## Swarm Outputs Integrated

1. OWV3 r4 adjudication:
   - Exact CUDA/T4 score packet exists and is scientifically valuable.
   - It is not promotable because the predeclared SegNet relative component
     gate failed: observed `0.00402120`, reference `0.00400656`, relative
     `1.003654`, cap `1.002`.
   - No retroactive gate relaxation is allowed. Next admissible paths are
     paired same-run PFP16 calibration and/or SegNet-conservative OWV3 R5.

2. H-V3 repair:
   - `segnet_uncertainty_weighted_loss` now preserves a 3-channel BCHW tensor
     before SegNet preprocessing.
   - Regression test pins SegNet preprocess input `(B, 1, 3, H, W)` and
     forward input `(B, 3, H, W)`.

3. HM-S/SA SegMap pack contract:
   - The block-FP packer is explicitly lossy via
     `segmap_block_fp_per_channel_lossy_v1`.
   - HM-S/SA scripts now write `segmap_pack_roundtrip.json`, embed lossy
     metadata, and require archive-level exact CUDA eval before any score
     claim.
   - The old `tol=1e-6` tensor-identity gate is no longer the contract for
     this lossy codec.

4. Lane 19/20 forensic holds:
   - Lane 19 now has deterministic archive build markers, JSON adjudication,
     current PFP16 A++ score/component gates, and corrected provenance.
   - Lane 20 now fails before auth eval unless a real non-static byte win and
     BHv1 archive/inflate integration exist.
   - Launcher holds are conditional: deleted or `cleared: true` hold entries
     still block if lane-specific clearance requirements are unmet.

5. Lightning supply-chain hardening:
   - Local strict scan is clean: `lightning=null`, `pytorch-lightning=null`,
     `lightning_sdk=2026.4.10`, `violation_count=0`.
   - `scripts/launch_lightning_batch_job.py` now sets
     `LIGHTNING_DISABLE_VERSION_CHECK=1` before SDK import paths.
   - PyPI `lightning==2.6.2/2.6.3` remains treated as compromise; use
     `lightning-sdk` only and scan every runner.

6. KL/distill hardening:
   - `train_renderer.py` now blocks positive `kl_distill_weight` unless
     `kl_distill_scope="segnet_aux"` is explicit.
   - `kl_distill_scope="primary_scorer"` is blocked in renderer training.
   - Current positive-KL profiles declare `kl_distill_scope="segnet_aux"`.
   - Strict preflight `check_train_renderer_kl_aux_explicit_scope` landed.

7. External research intake:
   - DeepSeek visual primitives are useful for Alpha geometry diagnostics and
     sparse residual selection, not as a runtime dependency.
   - arXiv 2604.26919 motivates dual-readout sensitivity validation:
     structural Fisher/gradient rank stability plus held-out PoseNet/SegNet
     perturbation response.
   - PufferLib/PPO over exact eval is rejected for now. Use repo-native
     bandit/BO over cheap deterministic surrogates, exact-eval finalists only.
   - Training-free GRPO is useful as an advisory replay/decision protocol, not
     as a repository dependency.

## Verification

- `py_compile`: passed for touched Python files.
- Focused tests: `107 passed in 2.20s`.
- Earlier focused H-V3/SegMap/Lane-hold suite: `68 passed in 1.66s`.
- Shell syntax: remote shell scripts passed `bash -n`.
- Strict local preflights passed:
  - `check_train_renderer_kl_aux_explicit_scope`
  - `check_segmap_hm_sa_lossy_pack_contract`
  - `check_no_active_mcp_server_config` with discovered user/project MCP config
    files.
- Lightning supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_current.json`.

## Current Live State

- Vast.ai: no live instances at the last sweep.
- Modal: app list shows `Tasks=0` on all apps.
- MCP: helper processes killed; discovered MCP config server maps emptied.

## Next Admissible Wall-Clock Order

1. OWV3 R5:
   - Run paired same-run PFP16 calibration on the r4 Lightning runner/toolchain.
   - Generate SegNet-conservative byte-feasible candidates near r4.
   - Exact eval only candidates with deterministic archive SHA/bytes and
     component gates.

2. Component sensitivity:
   - Produce CUDA diagnostic maps for candidate selection.
   - Promote to `component_sensitivity_v1` only after finite-difference
     component response curves, holdout stability, and exact custody exist.

3. NWCS:
   - Build-only only until validated sensitivity artifacts exist.
   - No exact eval or claim from debug/fake/shape-only sensitivity.

4. Alpha:
   - Run visual-primitives geometry diagnostics before retraining/eval spend.
   - Exact eval only after decoded-baseline geometry gates pass.

5. Hidden-gem recovery:
   - H-V3 and SegMap can be rerun as repaired implementations.
   - Lane 19 remains held until operator deliberately clears after exact
     archive/custody readiness.
   - Lane 20 remains held until real BHv1 archive/inflate integration lands.

## 2026-04-30T22:05Z Six-Item Swarm Follow-Up

Additional xhigh subagent workstreams were closed and integrated.

Landed or verified:

- PFP16 is already landed as the current A++ frontier: exact T4 CUDA,
  `686635` bytes, archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  recomputed score `1.043987524793892`. A fresh local build reproduced the
  same SHA and byte count.
- Lane 12/Alpha dispatch now defaults to the canonical Lane G v3 base archive,
  `GT_MASKS_SOURCE=decoded-baseline`, and `RUN_AUTH_EVAL=0`. Retired
  fresh-SegNet targets are blocked unless explicitly forensic, and exact eval
  requires pose-regeneration provenance.
- The Vast retry launcher now enforces the no-new-retraining-before-Lane12-L2
  rule through `.omx/state/lane12_nerv_l2_clearance.json`. Build-only,
  harvest, and exact-eval-only lanes remain allowed.
- J-NWC gained a build-only non-promotable path. J-NWC/NWCS/NWCS-EC exact CUDA
  paths now run `scripts/adjudicate_contest_auth_eval.py` with PFP16 A++
  score/component gates after `contest_auth_eval.py`.
- Sensitivity/OWV3 R5 readiness improved: the component profiler emits
  `perturbation_basis_v1.json` and response-curve prediction calibration
  diagnostics while remaining explicitly non-promotable; the OWV3 byte sweep
  can rank SegNet-conservative R5 neighbors around r4.
- Claim matrix now records OWV3 r4 as exact CUDA/T4 diagnostic evidence that
  failed the predeclared SegNet component gate. The Grand Council source doc
  now says the three shifts are necessary, with sufficiency unproved until
  exact stacked evidence exists.
- Live MCP helper processes are now a strict preflight failure via
  `check_no_live_mcp_processes`.

Telemetry:

- `uv run --no-sync vastai show instances --raw` returned `[]`.
- `uv run --no-sync modal app list` showed every app at `Tasks=0`.
- `check_no_live_mcp_processes(strict=True)` passed after killing respawned
  MCP helpers.

Verification:

- Focused integration suite passed: `69 passed in 2.66s`.
- Additional subagent suites passed: PFP16 `34 passed`; Sensitivity/OWV3
  `83 passed`; Lane 12 `11 passed`; J-NWC/NWCS `29 passed`.
- `bash -n` passed for touched remote shell scripts.
- `py_compile` passed for touched Python files.
- Strict preflights passed for remote auth JSON adjudication, launch retry
  self-protection, and live MCP process absence.

Next admissible actions:

1. Generate or exact-eval the OWV3 R5 SegNet-conservative candidate only with
   predeclared component gates and paired PFP16 calibration.
2. Produce CUDA component-sensitivity diagnostics, then promote to
   `component_sensitivity_v1` only after official finite-difference response,
   holdout stability, and exact custody.
3. Run Lane 12 Alpha only as build-only candidate generation until geometry
   diagnostics and pose-regeneration provenance exist.
4. Keep J-NWC/NWCS build-only unless validated sensitivity/corpus artifacts
   and exact adjudication gates are present.

## 2026-04-30T22:24Z Orchestration Closeout Delta

Swarm closeout and additional hardening:

- All six xhigh workers closed. Their outputs were integrated as an
  implementation/readiness delta, not as new score claims.
- Local OWV3 R5 sweep reproduced the worker's rank-1 SegNet-conservative
  candidate:
  `owv3_0047_bbr0p67_protect0p00135_aggr1em05`, `686468` bytes, archive SHA
  `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`.
  It is a queue candidate only until exact CUDA/T4 evaluation and component
  adjudication pass.
- OWV3 r4 exact CUDA/T4 packet was re-adjudicated locally after fixing the
  adjudicator failure mode. Gate-triggered runs now still write forensic
  artifacts:
  `contest_auth_eval.adjudicated.json` and `adjudication_provenance.json`.
  The r4 lane status is `COMPONENT_GATE_REVIEW_REQUIRED`, not promotable.
- Lightning exact-eval reproducibility advanced to dry-run queue readiness.
  SDK machine discovery works, local supply-chain scan is clean, and dry-run
  plans were written under `.omx/state/`. No spendful R5 job was submitted
  because SSH staging still fails with public-key denial and remote archive
  custody is therefore unproved.
- `src/tac/deploy/lightning/batch_jobs.py` now surfaces adjudication lane
  status, component-gate status, regression status, and `promotion_eligible`
  during artifact validation so failed exact evals remain harvestable without
  being promotable.
- J-NWC/NWCS/NWCS-EC provenance now records adjudicated result/provenance paths,
  score source, adjudication requirement, component-gate requirement, and SHA
  custody for adjudication files.
- Lane 12 jsonfix40 Alpha-Geo-0 comparison against Lane G v3/base masks failed
  gates (`global_disagreement=0.012303928799099393`,
  `boundary_2px=0.14883144511692872`,
  `pair_transition=0.009507171571470149`,
  `missing_component_rate=0.4611606740560512`). Lane 12 stays build-only and
  locked behind L2 clearance, geometry diagnostics, and pose-regeneration
  provenance.
- MCP helper processes were killed again and strict live-process/config
  preflights passed. Provider telemetry remains idle: Vast `[]`, Modal
  `Tasks=0`.

Verification:

- Combined focused regression slice: `137 passed in 3.27s`.
- Python compile passed for the adjudicator, Lightning deployment helpers,
  Lightning exact-eval wrapper, OWV3 sweep, sensitivity profiler, Alpha-Geo
  diagnostic, and focused tests.
- Shell syntax passed for touched remote scripts.
- `git diff --check` passed for the touched tracked implementation/runbook
  paths.
- Strict preflights passed for live MCP process absence, active MCP config
  absence, remote auth-eval JSON adjudication, and launch retry single-flight
  signal safety.

Custody / commit hygiene note:

- The `scripts/lightning_repro_workspace.py` staged-deletion/untracked
  same-path state was normalized with `git restore --staged` after verifying
  the file is tracked in `HEAD`; the file now appears as a normal modified
  tracked path.

Next admissible wall-clock order:

1. Repair Lightning SSH key authorization or switch to a remote-build exact-eval
   path that constructs the R5 archive inside the Studio before submission.
2. Run paired same-run PFP16 calibration, then exact CUDA/T4 adjudication for
   OWV3 R5 rank 1.
3. Generate CUDA component sensitivity/Fisher diagnostics; promote to
   `component_sensitivity_v1` only after finite-difference response curves,
   holdout stability, and manifest custody.
4. Keep Lane 12 build-only until an Alpha-Geo passing candidate and
   pose-regeneration provenance exist.
5. Keep J-NWC/NWCS exact eval disabled until validated sensitivity/corpus
   artifacts and adjudication provenance exist.

## 2026-04-30T22:30Z Lightning Queue Activation Delta

Lightning credentials and exact-eval queue status:

- Re-ran the Lightning SSH setup script; SSH now succeeds against the Studio
  (`ssh ... 'pwd'` returned `/home/zeus`).
- Remote workspace and upstream custody paths exist:
  `/teamspace/studios/this_studio/pact` and
  `/teamspace/studios/this_studio/upstream`.
- The reproducible staging script exposed a Lightning filesystem bug class:
  remote `uv sync` failed while hardlinking Torch files from the Studio cache.
  Fixed permanently by making remote staging use
  `UV_LINK_MODE=${UV_LINK_MODE:-copy} uv sync --locked --extra runtime`.
- Regression coverage added:
  `test_uv_sync_remote_command_forces_copy_link_mode_for_lightning_filesystems`.
- Remote staging then succeeded with a byte-verified manifest:
  `1093` files, `18490969` bytes,
  manifest SHA `3cd7611e6cce9a18e00ef9505f367fa44ee31622841d8a7378a2d360690919f1`.

Submitted Lightning Batch Jobs:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`
  - Status at `2026-04-30T22:29:54Z`: `Pending`.
  - Archive SHA:
    `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
  - Archive bytes: `686635`.
  - Purpose: paired PFP16 calibration/reference on the same Lightning T4 queue.
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`
  - Status at `2026-04-30T22:29:54Z`: `Pending`.
  - Archive SHA:
    `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`.
  - Archive bytes: `686468`.
  - Purpose: OWV3 R5 rank-1 exact CUDA/T4 eval. Metadata requires paired
    re-adjudication against the calibration job before any promotion claim.

Do not promote either job until artifacts are harvested locally, archive
identity is validated, CUDA/T4/sample gates pass, adjudication artifacts exist,
and R5 is re-adjudicated against the paired calibration result.

## 2026-04-30T22:48Z Lightning Exact-Eval Isolation Fix

New finding:

- The first live Lightning exact-eval attempts exposed a harness bug, not a lane
  result. `submissions/robust_current/inflate.sh` invokes `uv run`; inside the
  shared Studio repo this recreated `.venv`, then `upstream/evaluate.py` ran
  under the mutated env and failed on `ModuleNotFoundError: tqdm`.
- Affected jobs:
  - `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`: `Failed`,
    harness/env failure, no score evidence.
  - `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`: stopped after the
    same known-bad path began; no score evidence.

Permanent fix landed:

- Lightning exact-eval commands now set a per-job
  `UV_PROJECT_ENVIRONMENT=<output_dir>/uv_project_env` and
  `UV_LINK_MODE=${UV_LINK_MODE:-copy}` before `contest_auth_eval.py`.
- DALI/shared-venv setup now takes `.omx/state/lightning_exact_eval_venv.lock`
  so parallel exact-eval jobs cannot mutate the shared `.venv` concurrently.
- Tests cover both the copy-mode staging rule and the exact-eval isolation /
  lock command text.

Clean reruns submitted:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv`
  queued at `2026-04-30T22:47:20Z`, status initially `Pending`.
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv`
  queued at `2026-04-30T22:47:39Z`, status initially `Pending`.

Promotion state remains unchanged: PFP16 A++ is the only promotion-grade anchor;
R5 is queue-only until harvested/validated/adjudicated against paired PFP16.

## 2026-04-30T22:53Z Swarm Closeout Status

Current Lightning exact-eval queue state:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv`:
  `Pending`, cost `0.0`, no completion/failure timestamp.
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv`:
  `Pending`, cost `0.0`, no completion/failure timestamp.

Evidence status:

- No score is promoted from Lightning yet.
- The abandoned `r2`/original jobs remain harness failures only, caused by the
  pre-fix shared `.venv` mutation path.
- The clean isolated jobs are admissible queue work, but become evidence only
  after local harvest, archive SHA/size validation, CUDA/T4 provenance,
  adjudication JSON, and paired R5-vs-PFP16 readjudication.

Integrated verification:

- Focused regression suite covering Lightning queue generation, exact-eval
  reproduction, Lightning staging, remote-auth hardening, component
  sensitivity, Lane 12, and J-NWC/NWCS: `177 passed in 5.04s`.
- Python compile, shell syntax, targeted whitespace checks, provider telemetry,
  and strict MCP live/config preflights passed in the same closeout loop.

Next wall-clock-critical action is to poll both Lightning jobs, harvest only
canonical JSON/archive/provenance artifacts after completion, and run paired
adjudication before any R5 claim is allowed into the Grand Council result set.

## 2026-04-30T22:55Z Lightning Running Status

Latest SDK refresh:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv`:
  `Running`, cost `0.0`, no completion/failure timestamp.
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv`:
  `Running`, cost `0.0`, no completion/failure timestamp.

Operational implication: keep polling, but do not touch promotion state until
both jobs reach terminal states and artifacts pass local validation. If either
job fails, classify first as harness/infrastructure/custody/lane evidence only
after log and artifact inspection.

## 2026-04-30T23:10Z Exact CUDA Harvest And R5 Paired Result

Lightning terminal state:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv`:
  SDK status `Failed` because adjudication exited nonzero after a SegNet
  component gate fired, not because CUDA eval failed.
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv`:
  SDK status `Failed` for the same adjudication-gate reason.

Canonical harvest:

- Added `harvest-ssh` to `scripts/launch_lightning_batch_job.py`. It derives
  the SDK-persisted Studio artifact path, copies only canonical top-level
  evidence files, validates archive SHA/bytes, CUDA/T4 provenance,
  supply-chain scans, DALI/bootstrap records, and adjudication JSON, and
  attaches the validation to `.omx/state/lightning_batch_jobs.json`.
- Local artifact mirrors:
  - `experiments/results/lightning_batch/pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv/`
  - `experiments/results/lightning_batch/owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv/`

Exact CUDA/T4 facts:

- PFP16 paired calibration: recomputed score `1.037045485927815`, PoseNet
  `0.00316404`, SegNet `0.00401966`, rate `0.01828808`, archive bytes
  `686635`, SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `promotion_eligible=false`, `COMPONENT_GATE_REVIEW_REQUIRED`.
- OWV3 R5 rank-1: recomputed score `1.0373951773937642`, PoseNet
  `0.0031739`, SegNet `0.0040215`, rate `0.01828363`, archive bytes
  `686468`, SHA
  `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`,
  `promotion_eligible=false`, `COMPONENT_GATE_REVIEW_REQUIRED`.
- Paired delta: R5 is `+0.00034969146594909795` worse than PFP16 despite
  `167` fewer bytes. Do not promote R5.

Scientific classification:

- This is valid exact CUDA/T4 forensic evidence with local JSON custody.
- It is not frontier evidence because the predeclared component gate fired,
  and paired R5 is worse than paired PFP16.
- Next R5 work should be SegNet-conservative or sensitivity-map-backed; do not
  relax `max_segnet_relative=1.002` retroactively.

Additional implementation delta:

- `experiments/profile_component_sensitivity.py` now has an explicit
  `--promotion-finite-difference` mode for official CUDA component-response
  maps. The old Fisher-proxy path remains diagnostic and blocked from manifest
  assembly.
- Closeout verification: focused regression `154 passed in 4.10s`,
  Lightning/sensitivity focused subset `101 passed in 2.24s`, local
  supply-chain scan `OK`, Vast `[]`, Modal `Tasks=0`, and strict MCP
  process/config preflights clean after killing respawned MCP helpers.

## 2026-04-30T23:30Z R6 Queue And Guardrail Greenup

Swarm closeout:

- Worker A selected the shortest admissible R6 probe after the failed R5 exact
  result: `owv3_0076_bbr0p65_protect0p0013_aggr1em05`.
- Worker B found sensitivity-promotion blockers: zero-signal response curves,
  NaN map acceptance, contest/archive custody mismatch, and sample-plan hash
  rewriting.
- Worker C confirmed J-NWC/NWCS must wait for a real
  `build_nwcs_sensitivity_inputs.py`-style artifact builder; fake/uniform
  sensitivity cannot enter promotion mode.
- Worker D confirmed Lane 12 NeRV is blocked: no L2 clearance packet and the
  latest `jsonfix40` Alpha-Geo diagnostic has `overall_pass=false`.
- Worker E recommended component-gate-only outcomes complete as forensic
  Lightning jobs while remaining non-promotable.
- Worker F confirmed C-040 claim hygiene and listed stale public docs to
  regenerate or quarantine.

Implementation landed:

- `experiments/sweep_owv3_byte_plan.py` now has an R6 SegNet-conservative
  selector after failed exact R5. It requires strictly fewer OWV2-low-bit
  channels than the failed R5 reference and records failed-R5 exact CUDA/T4
  metrics in the queue packet.
- `scripts/adjudicate_contest_auth_eval.py`,
  `src/tac/deploy/lightning/batch_jobs.py`, and
  `scripts/launch_lightning_batch_job.py` now support
  `--allow-component-gate-forensic-success`. Default direct adjudication still
  fails closed; Lightning exact-eval jobs can complete with valid forensic
  artifacts when the component gate is the only failure.
- `experiments/profile_component_sensitivity.py` now blocks zero-signal
  finite-difference curves from promotion.
- `experiments/build_component_sensitivity_manifest.py` now rejects NaN/Inf
  tensors, contest JSON/archive SHA or byte mismatches, and sample-plan
  `split_hash` mismatches instead of silently rewriting custody.
- `docs/runbooks/owv3_r5_exact_eval_queue.md` now records R5 as historical
  non-promotable exact evidence and R6 as active queue state.

R6 exact-eval dispatch:

- Remote archive custody checked on Lightning:
  `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`,
  `686531` bytes.
- Submitted Lightning job
  `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`.
- SDK job name:
  `owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1`.
- Latest status at `2026-04-30T23:29:12Z`: `Pending`, cost `0.0`.
- The first submit attempt failed client-side because Lightning needed
  `--user adpena`; the second failed client-side because accelerator alias
  `T4` was not accepted by the AWS cluster. The successful submit used
  `--machine g4dn.2xlarge`, which SDK reports as T4. Those failed attempts did
  not start eval jobs.

Verification:

- `136 passed in 3.40s` across OWV3 sweep, Lightning batch tooling,
  adjudication, profile sensitivity, manifest builder, and artifact validator.
- Python compile passed for all touched Python files.

Next admissible action: poll R6 until terminal, harvest with `harvest-ssh`,
validate canonical artifacts, then classify the result under paired PFP16
component gates. Do not cite R6 as score evidence while it is pending.

## 2026-04-30T23:48Z R6 Exact Harvest Classification

R6 reached terminal `Completed`, was harvested through `harvest-ssh`, and local
artifact validation passed with adjudication required.

Exact CUDA/T4 result:

- Job: `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`
- Candidate: `owv3_0076_bbr0p65_protect0p0013_aggr1em05`
- Archive SHA: `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`
- Archive bytes: `686531`
- Score recomputed: `1.0393166493980681`
- PoseNet: `0.00323147`
- SegNet: `0.00402421`
- Device: `cuda`, GPU: `Tesla T4`, samples: `600`

Adjudication:

- Regression versus paired PFP16: `+0.0022711634702530237` score.
- Byte delta versus paired PFP16: `-104` bytes.
- PoseNet gate failed: relative `1.0213113614240024` > `1.002`.
- SegNet gate passed: relative `1.0011319365319455` <= `1.002`.
- Strict final-deploy adjudication returned exit code `2`.

Grand Council classification:

- R6 is A++ exact CUDA/T4 forensic negative evidence for this
  implementation/config.
- It is not promotable and does not update the frontier.
- It is not an OWV3 family KILL. The failure moved from R5's SegNet gate to
  R6's PoseNet gate, so the next design problem is sensitivity balancing, not
  generic byte-count reduction.
