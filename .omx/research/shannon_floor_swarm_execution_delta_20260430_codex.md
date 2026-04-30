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
