# Codex Progress Ledger - Shannon Floor Execution Readiness

Adjacent source of truth:
`shannon_floor_execution_readiness_20260430.md`

Date: 2026-04-30

## Scope

This ledger records current execution state, parallel lanes, blockers, and the
next-turn command plan for fastest wall-clock progress.

## Landed

1. OWV3 implementation slice:
   - Sensitivity-map contract added.
   - Mixed-channel OWV3 archive added.
   - `OWV3` magic registered.
   - Contest inflate dispatch added.
   - Synthetic and inflate-path tests passing.

2. Lane 12 packaging slice:
   - `.nrv` accepted by auth archive validator.
   - `masks.nrv` discoverable by inflate resolver even with default
     `masks.mkv` setting.

3. Documentation state:
   - Readiness, contest audit, and external research intake docs updated with
     implementation-readiness status and strict evidence labels.

## Ongoing Work

- OWV3 still needs:
  - stack archive builder,
  - sensitivity artifact generator or converter,
  - train/holdout sensitivity CV report,
  - exact archive eval.
- Lane 12 still needs:
  - full CUDA training,
  - clean contest dependency closure for `tac.nerv_mask_codec`,
  - exact `.nrv` archive,
  - exact contest eval.
- PFP16 exact CUDA eval is harvested. It now needs the remote
  provenance/adjudication parser fix and, if promoted to a submission
  candidate, T4/equivalent A++ evidence.
- Lane 17 IMP needs harvest or exact eval once remote result is ready.
- Lane 19, Lane 8, and hidden-gem recovery lanes need dispatch triage.

## Fastest-Wall-Clock Roadmap

1. Parent agent implements OWV3 stack builder and per-weight-to-per-channel
   sensitivity conversion.
2. Swarm audits exact eval dispatch, Lane 12 clean inflate closure, sensitivity
   artifact availability, and hidden-gem recovery ordering in parallel.
3. Fix the PFP16 remote provenance/adjudication parser bug class and preserve
   the harvested `contest_auth_eval.json` as the authoritative score source.
4. Dispatch OWV3 exact eval after builder plus sensitivity artifact exist.
5. Dispatch Lane 12 full CUDA `.nrv` eval after dependency closure is proven.
6. Run IMP and hidden-gem lanes in parallel where GPU slots are available.
7. Run gamma MDL/ADMM/static entropy only after measured component streams
   exist.

## Next Turn Checklist

- Spawn focused workers for:
  - PFP16 parser/adjudication fix and A++ readiness,
  - Lane 12 clean contest dependency closure,
  - sensitivity artifact inventory/conversion,
  - hidden-gem dispatch order.
- Implement:
  - `experiments/build_lane_g_v3_owv3_stack.py`,
  - per-weight Fisher to per-channel sensitivity-map conversion utility.
- Verify:
  - targeted pytest,
  - deterministic zip manifest,
  - no scorer import in OWV3 decode,
  - `.nrv` resolver avoids SegNet fallback.

---

## Update - 2026-04-30 Later

Swarm results incorporated:

1. **PFP16 harvest:** exact CUDA eval has landed and is Grade A score-grade.
   A++ remains blocked by RTX 4090 hardware (`gpu_t4_match=false`) and missing
   contest-budget/T4 evidence.
2. **Hidden-gem ordering:** harvest active Lane 8, Lane 19, Lane 17 first; next
   parallel dispatch wave should be SegMap clone, H-V3 half-frame, and
   Q-FAITHFUL. FL waits for RAFT chunking. MAE-V runs if spare 4090/A10G exists.
3. **Sensitivity inventory:** no usable `hessian_per_weight.pt` exists. Prior
   run crashed because mask grayscale values were used as class IDs. That bug
   is now fixed.
4. **Reproducibility audit:** central archive builder had nondeterministic set
   ordering and source mtimes. That is now patched for deterministic writes.
5. **Paper rigor:** writeup blueprint now exists and must be used for all
   claims and ablations.

Implementation landed after swarm:

- `experiments/build_lane_g_v3_owv3_stack.py`
- `experiments/convert_fisher_to_owv3_sensitivity_map.py`
- `src/tac/tests/test_owv3_sensitivity_conversion.py`
- `src/tac/tests/test_profile_hessian_mask_decode.py`
- `src/tac/submission_archive.py` `masks.nrv` manifest support and deterministic
  archive writes.

Verification:

```text
62 passed:
  profile_hessian_mask_decode
  lane12_nerv_dependency_closure
  owv3_sensitivity_conversion
  sensitivity_map
  owv3_sensitivity_weighted
  runtime_guards_pass_3
  contest_auth_eval
  pfp16_codec

17 passed:
  integration_boundaries
  engineered_corrections
```

Immediate fastest-wall-clock commands:

1. Fix the PFP16 remote provenance/adjudication parser class:
   - `contest_auth_eval.json` is authoritative.
   - `remote_provenance.json` currently says `contest_cuda_score=100.0`,
     `hard_kill_triggered=true`, and `lane_status=HARD_KILL_REGRESSION`; those
     fields are invalid and superseded until the script fix lands.
   - The harvested local evidence directory is
     `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.

2. Rerun CUDA Fisher after mask decode fix:

```bash
PYTHONPATH=src:upstream .venv/bin/python experiments/profile_hessian_per_weight.py \
  --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \
  --video upstream/videos/0.mkv \
  --masks-mkv experiments/results/lane_a_landed/iter_0/masks.mkv \
  --poses experiments/results/lane_a_landed/iter_0/optimized_poses.pt \
  --upstream upstream \
  --pair-weights experiments/results/lane_lane_w_modal/harvested_artifacts/lane_w_results/pair_weights.pt \
  --top-k 30 \
  --device cuda \
  --pair-batch 4 \
  --output experiments/results/owv3_sensitivity/hessian_per_weight.pt
```

3. Convert Fisher and build OWV3:

```bash
PYTHONPATH=src .venv/bin/python experiments/convert_fisher_to_owv3_sensitivity_map.py \
  --checkpoint experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --fisher experiments/results/owv3_sensitivity/hessian_per_weight.pt \
  --output experiments/results/owv3_sensitivity/sensitivity_map.pt \
  --metadata-json experiments/results/owv3_sensitivity/sensitivity_map.meta.json

PYTHONPATH=src .venv/bin/python experiments/build_lane_g_v3_owv3_stack.py \
  --sensitivity-map experiments/results/owv3_sensitivity/sensitivity_map.pt \
  --output experiments/results/lane_g_v3_owv3/archive_lane_g_v3_owv3.zip \
  --provenance-json experiments/results/lane_g_v3_owv3/provenance.json
```

4. Dispatch hidden-gem wave as capacity allows:

```bash
scripts/remote_lane_sa_segmap_clone.sh
scripts/remote_lane_h_v3_jointly_trained_halfframe.sh
scripts/remote_lane_q_faithful_jointgen.sh
```

Q-FAITHFUL remains high-risk because its current script uses KL-distill-like
machinery; it must be labeled prediction/high-risk until exact CUDA proves no
PoseNet collapse.

---

## Update - 2026-04-30 PFP16 Harvest

PFP16 evidence now in hand:

- Directory:
  `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.
- `contest_auth_eval.json`: `final_score=1.04`,
  `score_recomputed_from_components=1.0440481283330025`,
  `avg_posenet_dist=0.0034602`, `avg_segnet_dist=0.0040083`,
  `archive_size_bytes=686635`, `n_samples=600`.
- Archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Hardware: NVIDIA GeForce RTX 4090 CUDA, `gpu_t4_match=false`.

Readiness impact:

- PFP16 is Grade A score-grade and becomes the current verified frontier.
- It is not A++ until the same archive SHA has T4/equivalent hardware,
  contest-budget inflate, and full 1:1 provenance evidence.
- Current targeted bug-class fix is remote score parser/adjudication: invalid
  `contest_cuda_score=100.0` and `hard_kill_triggered=true` fields must be
  recomputed from, or superseded by, `contest_auth_eval.json`.

---

## Update - 2026-04-30 Remote Harness Hardening

Remote score parser/adjudication is now fixed and guarded:

- New canonical helper: `scripts/adjudicate_contest_auth_eval.py`.
- PFP16, Ω-W-V2, and Lane 8 stack scripts now adjudicate from
  `eval_work/contest_auth_eval.json`, not `auth_eval.log` text.
- PFP16 remote script now asserts the exact deterministic archive SHA before
  spending GPU time.
- Adjacent NWC/GP/FL/WC/sweep scripts now require `contest_auth_eval.json` and
  refuse last-object log scraping.
- `scripts/launch_lane_with_retry.py` now uses timeouts compatible with the
  underlying launcher poll windows and returns `UNKNOWN_REMOTE_STATE` on
  phase2-launch timeout to prevent duplicate dispatches.
- Strict preflight check:
  `check_remote_lane_auth_eval_json_adjudication`.
- Test coverage:
  `src/tac/tests/test_remote_auth_eval_hardening.py`.

Verification:

- `6 passed` for remote auth-eval hardening tests.
- `bash -n` clean across all modified lane scripts.
- `py_compile` clean across adjudicator, retry launcher, and preflight.

Readiness impact:

- PFP16 Grade A score-grade evidence remains valid and is now protected from
  false legacy regression provenance in future dispatches.
- Next wall-clock priority returns to CUDA Fisher sensitivity generation,
  OWV3 candidate build/eval, Lane 12 `.nrv` CUDA eval, and hidden-gem harvest.

---

## Update - 2026-04-30 Dispatch Readiness / DX Hardening

Dispatch orchestration is now treated as a correctness surface:

- `scripts/launch_lane_with_retry.py` now fails closed on same-label live Vast
  state through `live_instances_with_label_prefix`.
- A per-label advisory lock under `.omx/state/launch_locks/` prevents two local
  retry wrappers from launching the same logical lane simultaneously.
- Child launcher phases run in their own process group and are terminated on
  timeout/SIGINT/SIGTERM, preventing orphaned phase2 children.
- `phase2-launch` timeout remains `UNKNOWN_REMOTE_STATE`, never blind retry.
- Strict preflight now includes
  `check_launch_retry_wrapper_singleflight_and_signal_safe`.

Remote wave status:

- SegMap clone is live on Vast as instance `35906669`,
  label `lane_sa_segmap_clone_2026-04-30_codex_a2`, RTX 4090, `$0.2539/hr`,
  SSH `root@ssh2.vast.ai:26668`.
- Remote proof at launch:
  - `/workspace/setup.log` contains `SETUP_COMPLETE`.
  - `/workspace/pact/lane_sa_segmap_clone_results/heartbeat.log` exists.
  - `run.log` reached Stage 2 training after NVDEC probe and anchor checks.
- Existing active lanes at the same checkpoint:
  - `35885106` HM-S.
  - `35899850` Lane 19 logit margin.
- H-V3 dispatch added after the SA checkpoint:
  - Live instance `35907873`,
    label `lane_h_v3_joint_halfframe_2026-04-30_codex_a4`, RTX 4090,
    `$0.2731/hr`, SSH `root@ssh5.vast.ai:27872`.
  - The host passed lightweight NVDEC pre-probe.
  - At 2026-04-30T15:21Z it was still in setup Stage 3 installing
    `nvidia-dali-cuda120`; lane training had not started yet.

Still blocked / not promoted:

- PFP16 A++: exact T4/equivalent rerun required.
- OWV3: CUDA Fisher artifact, sensitivity conversion, archive build, and exact
  CUDA eval required.
- Lane 12: full CUDA `.nrv` archive eval required.
- H-V3: dispatched/setup-running only; exact archive CUDA eval still required.
- Q-FAITHFUL: gated because of KL-distill-like risk.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile`: launcher wrapper, preflight, adjudicator.
- Preflight scanners:
  - `launch_self_protect_violations 0`
  - `remote_auth_json_violations 0`
- `git diff --check`: clean.

---

## Update - 2026-04-30T16:16Z Routing And Readiness Delta

Promotion-grade compute routing:

- Lightning AI endpoint recorded for exact CUDA reruns:
  `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- A xhigh worker is validating Lightning access/hardware and will run
  `scripts/pfp16_a_plus_plus_exact_t4_eval.sh --run` only if the machine
  satisfies the T4/equivalent A++ gate.
- Modal credits remain in play for non-promotion smoke, ablation, build-only
  codec work, and Fisher/sensitivity generation.
- Vast should be used conservatively because state tracking and NVDEC
  reliability remain weak; current live lanes are monitored but not trusted
  until they emit lane-local `contest_auth_eval.json`.

Readiness changes:

- Lane 12 NeRV exact CUDA eval completed and retired the current measured
  implementation/config:
  `26.03719330455429` recomputed, PoseNet `49.77849960`, SegNet
  `0.03528685`, archive `296478` bytes.
- The NeRV lane is no longer launch-blocked, but the current technique is not
  frontier-useful without a geometry-preserving target.
- PFP16 remains the cleanest near-term promotion candidate: Grade A
  score-grade evidence exists, A++ is blocked only on exact T4/equivalent run.
- OWV3 remains blocked on a real Fisher artifact and archive eval; Vast NVDEC
  failures make Lightning/Modal the next correct route.
- H-V3 has progressed from setup to Stage 1 training and remains worth
  monitoring because it attacks a real train/inflate distribution mismatch.

Immediate execution priority:

1. Use Lightning for PFP16 A++ exact rerun as soon as hardware is confirmed.
2. Use Modal/Lightning to produce Fisher/sensitivity artifacts for OWV3.
3. Continue harvesting HM-S, Lane 19, SA, and H-V3 only through exact
   `contest_auth_eval.json`.
4. Treat Lane 12 NeRV as failed evidence and redesign alpha around
   pose-preserving masks before more spend.

---

## Update - 2026-04-30T16:25Z Lightning Exact-Eval Home Validated

Lightning AI successfully produced a promotion-grade PFP16 exact eval:

- SSH endpoint used:
  `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- Remote repo for this run was staged separately at
  `/home/zeus/content/pact_pfp16_exact_20260430T1625Z` because the default
  `/home/zeus/content/pact` tree was stale/non-git and missing required eval
  files.
- Remote upstream tree: `/home/zeus/content/upstream`, git `c5e1274`.
- Hardware: Tesla T4; `gpu_t4_match=true`.
- Evidence:
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`.

Readiness implication:

- Use Lightning as the exact-eval home for PFP16, OWV3 candidates, and any
  stack that must be promoted.
- Use separate staged Lightning project/worktree directories for hermetic runs
  instead of mutating a possibly stale default workspace.
- Modal remains the fastest supplementary path for Fisher/build-only smoke and
  ablations, but current Modal wrappers are advisory because they force CPU eval.

---

## Update - 2026-04-30T16:45Z Execution Readiness Supersession

Earlier checklist items that say PFP16 needs A++ or Lane 12 needs first exact
CUDA eval are superseded.

Current readiness state:

- PFP16 A++ is complete and is the deploy baseline pending final bundle polish:
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/`,
  recomputed `1.043987524793892`, SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Lane 12 NeRV `jsonfix40` has exact negative evidence and should not be
  rerun as a frontier candidate without a redesigned alpha objective.
- OWV3/Fisher has a Modal smoke packet, but it is suspicious negative:
  archive `912971` bytes, `+218897` vs Lane G v3, no exact eval. The builder
  now fails closed on non-smoke size regressions unless
  `--allow-size-regression` is explicitly passed.
- Lightning AI remains the exact-eval home; Modal remains build/smoke/ablation
  only until its outputs are rerun through Lightning exact CUDA.
- A read-only xhigh sidecar is auditing
  `https://github.com/Lightning-AI/pytorch-lightning` for reusable
  checkpointing/logging/accelerator primitives that do not compromise
  contest-grade reproducibility.

Lightning repo audit result:

- Use Lightning Fabric patterns selectively; do not migrate canonical lanes to
  full PyTorch Lightning `Trainer`.
- High-value borrow: DataLoader worker seeding, `isolate_rng` around eval,
  rank-zero artifact writing, optional Fabric callback-style hooks, and
  optional full training-state checkpoint interface.
- Avoid for contest-critical paths: Trainer-managed samplers/checkpoints,
  DDP exact eval, LightningCLI sweeps, remote loggers/artifact managers.
- Exact eval and archive construction stay on the current custom deterministic
  path.

Forensic policy update:

- Disappointing/unexpected results now require a mitigation and
  leaderboard-reverse-engineering pass before any scoped retirement language:
  stacking options, hybrid residuals, fallback routing, side-info accounting,
  and likely full-stack archive allocation hypotheses must be recorded.

Lightning AutoResearch / Batch Jobs audit result:

- AutoResearch is useful conceptually, but the horizontal Batch Jobs version is
  not yet the ready dispatcher for this repo. Use official Lightning Batch Jobs
  directly for auditable work.
- Make official `Job.run(...)` the default target for future Lightning
  training/eval queues once wired; keep SSH/tmux for emergency manual work.
- Use one T4 exact-eval queue for `contest_auth_eval.py`; faster GPUs are for
  training/sweeps only.
- Job names must encode lane/profile/seed/git-or-diff hash/attempt; retries use
  new names, never overwritten jobs.
- Harvest `contest_auth_eval.json`, provenance, archive, telemetry, logs,
  `job.link`, `job.snapshot_path`, and `job.artifact_path`.
- Do not delete Lightning jobs until artifacts are mirrored locally.
- Keep large mutable data outside Studio snapshots and content-hash inputs.
- Avoid AutoResearch autonomous code edits on promotion paths because they can
  reintroduce forbidden KL/adaptive/scorer/archive patterns.

---

## Update - 2026-04-30T17:00Z Evidence Vocabulary Greenup

The adjudicator and source docs now use scoped regression language for future
negative evidence:

- New adjudicator flag: `--regression-threshold`.
- New structured fields: `regression_triggered`, `regression_threshold`, and
  `regression_scope`.
- New regression status: `REGRESSION_REVIEW_REQUIRED`.
- Legacy `hard_kill_triggered` fields remain historical-only and must never
  override `contest_auth_eval.json`.

This supports the Grand Council rule that a disappointing exact result retires
only the measured implementation/config until mitigation, stacking,
leaderboard reverse-engineering, and adversarial consensus determine a broader
scope.

---

## Update - 2026-04-30T17:35Z Execution Queue Reopened

The Lightning PyPI incident has been contained locally and converted into
strict preflight/DX controls. Execution is back on the six-item frontier queue.

Immediate ordered queue:

1. **Active harvest:** poll live Vast lanes and harvest only lane-local
   canonical `contest_auth_eval.json` plus exact archive/provenance. Logs,
   CPU/MPS, and proxy numbers remain monitor-only.
2. **Lightning Batch Jobs:** promote the new official Batch Jobs wrapper to the
   primary auditable eval queue by adding status, harvest, mirror, SHA preflight,
   and adjudication wiring.
3. **PFP16 A++ deploy packet:** keep
   `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/` as the
   local deploy bundle and close remaining staged-tree/source-manifest gaps if
   remote evidence becomes recoverable.
4. **OWV3/Fisher:** implement the byte-aware redesign in
   `owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`; do not run
   promotion eval until archive bytes are plausible against PFP16 A++.
5. **Alpha/Lane 12:** implement the pose-preserving redesign in
   `alpha_pose_preserving_redesign_spec_20260430_codex.md`; Alpha-Geo-0
   stale-pose isolation is first.
6. **Paper/writeup:** update only from claim-matrix rows. PFP16 A++ is the
   current deploy baseline; Lane 12 is A-negative scoped evidence; OWV3 smoke
   is empirical design motivation only.

Current subagent allocation:

- Lightning Batch Jobs implementation worker.
- Vast live harvest explorer.
- OWV3/Fisher implementation explorer.
- Alpha/Lane 12 rescue explorer.
- PFP16/paper bundle explorer.

Local Codex coverage:

- KL/DX audit after subagent capacity frees up.
- Integration of returned worker changes.
- Verification sweep before deployment or final report.

---

## Update - 2026-04-30T17:43Z Readiness Queue State

Current readiness by stream:

1. **Active harvest:** watch-only. Four Vast lanes are still running and none
   has lane-local exact artifacts. Do not rsync/promote/retire from logs.
2. **Lightning Batch Jobs:** implementation slice landed. The wrapper now has
   expected archive identity checks, command-hash queue records, JSON-preserving
   exact CUDA command generation, local artifact validation/mirroring, and
   state-attached harvest records. Next gate is local verification plus a dry
   run against the PFP16 archive identity.
3. **PFP16 A++ deploy packet:** score claim remains A++; paper packet remains
   blocked by source custody and stale provenance contradictions inside the
   bundle.
4. **OWV3/Fisher:** blocked for implementation, not science. Required next
   build is ASYM-preserving, byte-aware, missing-policy-error, calibration/
   holdout Fisher, and canonical adjudication before exact eval.
5. **Alpha/Lane 12:** proceed with Alpha-Geo-0 diagnostics, then Alpha-Geo-1
   pose rescue against decoded NeRV masks. Do not broad-kill mask compression.
6. **Paper/writeup:** claim matrix is the source. Public docs still need PFP16
   A++ regeneration and stale CPU/MPS/local claims quarantined.
7. **KL/distill:** primary scorer KL is forensic-only; SegNet-only auxiliary KL
   remains allowed under exact component gates. New fail-closed SegMapTrainer
   guard prevents accidental primary-KL routing.

Next executable wall-clock actions:

- Run focused verification for Lightning Batch Jobs, KL tests, preflight, and
  compile.
- Use the Batch Jobs dry-run CLI with the PFP16 archive SHA/byte identity to
  exercise queue metadata without making a real API call.
- After KL council sidecar returns, patch any remaining guard/test gaps.
- Begin OWV3 ASYM-passthrough implementation or Alpha-Geo-0 diagnostics as the
  next code-heavy slice, depending on GPU lane harvest timing.

---

## Update - 2026-04-30T17:50Z Verification And Next Implementation Fork

The immediate verification gate is green:

- Compile: touched Python files compiled.
- Shell: OWV3/Fisher remote script passed `bash -n`.
- Tests: focused suite passed `291 passed`.
- Whitespace: `git diff --check` passed.
- Lightning dry run: exact-eval Batch Jobs spec generated for PFP16 A++ with
  expected SHA/bytes, adjudication wiring, and queue metadata.

Readiness changes:

- KL/distill is no longer blocked on the known SegMap scope or SNR-controller
  no-op bugs. It is still promotion-gated by high-weight waivers, exact CUDA
  component gates, and scoped provenance.
- OWV3/Fisher conversion now fails closed on missing Fisher tensors. This
  prevents unmeasured layers from becoming hidden FP16-protect smoke artifacts
  in promotion scripts.
- Lightning Batch Jobs wrapper is ready for a real exact-eval queue submission
  once credentials/session context is available and the operator approves API
  spend. No real API call was made in this verification pass.

Next code-heavy slices, in order:

1. OWV3 ASYM-passthrough / byte-aware allocation implementation and tests.
2. Alpha-Geo-0 NeRV geometry diagnostics before any further Lane 12 exact eval.
3. PFP16 deploy-bundle custody cleanup: supersede stale hard-kill provenance,
   produce a tight staged-tree/source manifest, then regenerate public docs.
4. KL high-weight waiver/preflight and component-gated adjudication.
5. Live Vast harvest when lane-local exact artifacts appear.

---

## Update - 2026-04-30T18:08Z Six-Item Swarm Greenup

Five xhigh swarm slices returned and were integrated or reviewed:

1. **Active harvest:** no live Vast lane is harvestable. HM-S `35885106`,
   Lane 19 `35899850`, SA `35906669`, and H-V3 `35907873` still lacked
   lane-local CUDA `contest_auth_eval.json`; watch-only.
2. **Lightning Batch Jobs/adjudication:** component gates landed for PoseNet
   and SegNet. Exact-eval adjudication now rejects before copying accepted
   artifacts when `avg_posenet_dist` or `avg_segnet_dist` breaches absolute or
   relative gates, even if total score is in the predicted band.
3. **PFP16 A++ custody:** final deploy bundle now quarantines stale legacy
   parser fields under `legacy_parser_output_quarantined`; the sole score
   authority is `eval/contest_auth_eval.json`. Archive remained unchanged:
   SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
   `686635` bytes.
4. **OWV3/Fisher:** first promotion gate landed. `fallback_action=keep_asym`
   is now the default for protected/all-protected paths; `diagnostic_fp16` is
   explicit smoke/debug and non-promotable. Builder emits byte plan, deterministic
   ZIP rebuild proof, member manifest, and PFP16 A++ frontier comparator gate.
5. **Alpha/Lane 12:** Alpha-Geo-0 CPU tensor diagnostics landed for mask
   geometry: global disagreement, per-class confusion/F1, boundary-band drift,
   stable-region false flips, transition disagreement/F1, speckle rate,
   component centroid jumps, and worst frame pairs. ZIP member loading was
   hardened against traversal.
6. **J-NWC corpus codec:** xhigh worker remains active. Scope is deterministic
   corpus manifest, fail-closed corpus selection, no fake sensitivity in
   promotable J-NWCS scripts, and deterministic archive writes.

Verification this update:

- `py_compile` passed for Lightning adjudicator/wrapper, Alpha diagnostic,
  OWV3 builder/runtime, KL touched files, and PFP16 bundle builder.
- `bash -n scripts/remote_lane_g_v3_owv3_fisher_stack.sh` passed.
- Focused pytest suite passed: `77 passed in 3.56s`.
- Alpha diagnostic focused test passed: `8 passed in 1.02s`.
- PFP16 custody checks passed: JSON parse, score-authority `jq` assertion,
  bundle run-command `bash -n`, archive SHA/bytes unchanged.
- `git diff --check` and `git diff --check --cached` passed.
- MCP process sweep: exact `chrome-devtools-mcp` / `rbx-studio-mcp` patterns
  killed again; app-level runtime may respawn them, but repo/Codex/Claude config
  remains empty of project MCP servers.

Next wall-clock order:

1. Wait for Bernoulli/J-NWC corpus hardening and run its focused tests.
2. Run Alpha-Geo-0 against Lane 12 `jsonfix40` versus Lane G v3 mask target to
   isolate the actual PoseNet geometry failure.
3. Build an OWV3 archive only after a CUDA authoritative sensitivity map exists
   and the post-ZIP byte plan beats or justifies against PFP16 A++.
4. Submit the PFP16 A++ archive through Lightning Batch Jobs exact-eval dry-run
   to real queued execution when credentials/session context are available.
5. Continue live Vast/Modal harvest only when canonical JSON/archive/provenance
   artifacts exist.

---

## Update - 2026-04-30T18:42Z Readiness Delta

Readiness changes landed:

1. **Lightning SSH and reproducible staging**
   - SSH to `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai` works.
   - `scripts/lightning_repro_workspace.py` now stages source plus explicit
     artifacts with SHA/byte manifests and remote verification.
   - Real sync `owv3_repro_contract_20260430_r1` verified `1074` files and
     `18645277` bytes. This is the current source/artifact custody mechanism
     for Lightning work.
   - Exact eval is not yet armed in the main Lightning tree because this sync
     intentionally used `--no-install`; environment JSON recorded system Python
     and no torch. Next step is locked runtime install or `PYBIN` routed to the
     existing CUDA venv before Batch Jobs/exact eval.

2. **OWV3 Fisher r2**
   - CUDA/T4 Fisher and sensitivity map exist with protected Conv2d coverage.
   - Missing Conv2d policy is fail-closed and clean.
   - Archive bytes do not beat PFP16 A++; exact eval remains intentionally
     blocked. This prevents spending or claiming on a rate-regressing candidate
     without distortion evidence.

3. **Alpha geometry evidence**
   - Lane 12 `jsonfix40` has Alpha-Geo-0 diagnostics against Lane G v3 masks.
   - Bad axes are temporal/component/lane-marking geometry, not simple global
     mask disagreement. Alpha redesign should prioritize pose-preserving
     geometry and temporal continuity before retraining spend.

4. **Sensitivity/Perturbation audit**
   - Existing tools cover pair sensitivity, Fisher, FP4 layer sensitivity,
     scorer saliency, PoseNet sensitivity maps, YUV null-space probes, and
     NeRV geometry diagnostics.
   - Missing production gate is unified `component_sensitivity_v1` with
     PoseNet-only, SegNet-only, combined maps, holdout calibration, perturbation
     response curves, source/eval custody, and CUDA-only promotion checks.

5. **MCP hardening**
   - Live MCP helper processes were killed. Claude/Codex active MCP config is
     empty; OpenCode MCP config was also emptied. Any helper recurrence should
     be treated as outer-runtime respawn and killed by exact process pattern.

Next executable wall-clock order:

1. Install/verify the locked runtime in the Lightning staged tree, or add a
   documented `PYBIN` path to the known CUDA venv, then run exact-eval queue
   submission only for byte-plausible candidates.
2. Sweep OWV3 byte-plan thresholds and entropy/metadata overhead to get below
   PFP16 A++ bytes before exact eval.
3. Implement `component_sensitivity_v1` validator and then wire Fisher and
   response-curve producers into it.
4. Wait for the J-NWC adversarial audit; do not dispatch J-NWCS promotion until
   deterministic corpus manifests and no-fake-sensitivity gates are confirmed.
5. Start Alpha-Geo-1: pose rescue / temporal-component-preserving mask codec
   diagnostics before any full NeRV retraining.

---

## Update - 2026-04-30T19:08Z Dispatch Readiness Changes

J-NWC/NWCS readiness improved materially:

- J-NWC/NWCS remote scripts now fail closed on CPU/MPS auth-eval overrides and
  can no longer mark a downgraded run as `[contest-CUDA]`.
- Anchor archive extraction is zip-slip safe and member allowlisted.
- Corpus manifest replay is relocatable across staged workspaces.
- Codec construction is now inside the seed contract for both NWC and NWCS.
- NWCS sensitivity promotion requires provenance anchored to the exact anchor
  archive, extracted renderer, corpus manifest, block size, parameter shapes,
  and block counts.
- NWCS renderer output now uses a real `NWCS1` container with magic bytes,
  embedded codec checkpoint, tensor metadata, parser validation, renderer
  export dispatch, and inflate-side loader dispatch.
- `component_sensitivity_v1` validator exists; producers still need to emit
  the full PoseNet/SegNet/combined artifact before beta lanes can promote.

Verification:

- `bash -n` for J-NWC/NWCS/J-NWCS-EC scripts: passed.
- Focused suite plus Lightning repro tests:
  `src/tac/tests/test_neural_weight_codec_corpus.py`,
  `test_neural_weight_codec.py`,
  `test_neural_weight_codec_sensitivity.py`,
  `test_neural_weight_codec_sensitivity_renderer_format.py`,
  `test_component_sensitivity_artifact.py`,
  `test_remote_lane_j_nwc_hardening.py`,
  `test_lane_j_nwc.py`,
  `test_lightning_repro_workspace.py`: `64 passed in 2.88s`.
- Lightning staged tree refreshed under
  `shannon_greenup_20260430_jnwcs_r1_manifest.json`: `1078` files,
  `18724610` bytes, remote SHA verification OK. Runtime install still pending.

Next dispatch gates:

1. Produce real anchor and corpus sensitivity artifacts with the new required
   metadata, not debug placeholders.
2. Run a short NWCS build-only smoke that emits an `NWCS1` archive and verifies
   inflate loader dispatch before exact eval spend.
3. Only then submit a CUDA exact eval through Lightning/Vast/Modal with
   canonical JSON/archive/provenance custody.

---

## Update - 2026-04-30T19:14Z Execution Queue Refinement

Readiness delta:

- OWV3 has advanced from byte-blocked to byte-feasible for one candidate:
  `686557` bytes, `-78` versus PFP16 A++ `686635`, SHA-256
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
  This is not a score claim; it is the first candidate eligible for exact CUDA
  eval once a GPU-backed Lightning/Vast/Modal runner is available.
- Lightning staging for that archive is reproducible:
  `.omx/state/owv3_byte_feasible_repro_20260430_r1_manifest.json`, `1081`
  files, `17674947` bytes, remote manifest verification OK.
- Lightning Batch Jobs dry-run exists:
  `owv3_byte_feasible_exact_cuda_20260430_codex_dryrun`, command hash
  `e8551610ddb813ae6d0ee4857c3f110a22affa201ce64333624709bbeee15e89`.
  The current SSH shell is not GPU-ready (`nvidia-smi` absent), so execution is
  intentionally held.
- Local and remote Lightning supply-chain scans are clean for the known
  compromised `lightning==2.6.2/2.6.3` indicators. Use
  `scripts/scan_lightning_supply_chain.py --strict --quiet --json-out <path>`
  before any newly created runner is trusted.
- Alpha-Geo-0 now has both Lane G v3 and Lane A/base comparisons for Lane 12
  `jsonfix40`; both show the same geometry failure profile. Alpha-Geo-1 should
  target temporal/component continuity and pose preservation, not blind
  retraining.

Verification:

- Focused cross-slice suite: `126 passed in 3.64s`.
- `bash -n` passed for J-NWC/NWCS/OWV3/active remote lane scripts.
- `py_compile` passed for touched Python scripts/modules/tests.

Fastest safe wall-clock order from here:

1. Turn on or attach a CUDA-visible Lightning runtime; rerun the supply-chain
   scan; verify `nvidia-smi` and torch CUDA before any exact eval.
2. Submit OWV3 byte-feasible exact eval with expected archive SHA/bytes and
   PoseNet/SegNet component gates. If it regresses components, quarantine as a
   measured config only.
3. In parallel, generate `component_sensitivity_v1` CUDA producers for
   PoseNet-only, SegNet-only, and combined maps with response curves.
4. Run NWCS build-only `NWCS1` smoke with validated sensitivity provenance; only
   then queue exact CUDA eval.
5. Integrate the four active xhigh research/audit outputs into the claim
   matrix before launching new expensive lanes.

---

## Update - 2026-04-30T19:22Z Readiness After Hardening Loop

Execution readiness changes:

- `component_sensitivity_v1` now has both validator and deterministic manifest
  assembler. The remaining blocker is the actual CUDA perturbation producer for
  PoseNet-only, SegNet-only, and combined maps plus response curves.
- Legacy FP4 sensitivity profiling can no longer silently consume encoded mask
  luma as class IDs. CPU profiling is explicitly diagnostic and non-promotable.
- Legacy `segnet_kl` cannot be promotion-capable by default. It is fenced as
  `kl_distill_scope="segnet_aux"` plus `promotion_eligible=False` until a
  reviewed migration to the canonical SegNet-only KL helper exists.
- All four xhigh audit/research outputs have landed; they add no score claims
  but refine the roadmap:
  - use dual-readout sensitivity validation,
  - use visual primitives for Alpha geometry,
  - prefer bandit/BO over direct RL exact-eval control,
  - keep primary KL forensic-only and `segnet_kl` fenced.

Current execution blockers:

- Lightning current SSH shell is still not GPU-ready. Exact eval remains
  blocked on CUDA visibility, not code readiness.
- Component maps/curves still need CUDA computation. The assembler prevents
  hand-built or incomplete promotion manifests but does not generate the maps.

Verification:

- Expanded focused suite: `172 passed in 4.78s`.
- `py_compile`, `bash -n`, and `git diff --check`: passed.

Next fastest wall-clock order:

1. Bring up a CUDA-visible Lightning job/session, then run
   `scripts/scan_lightning_supply_chain.py --strict` and `nvidia-smi`/torch
   CUDA checks inside that runtime.
2. Submit OWV3 byte-feasible exact eval with expected SHA/bytes and component
   gates.
3. In parallel, implement the CUDA component sensitivity producer that emits
   maps and response-curve JSON compatible with
   `experiments/build_component_sensitivity_manifest.py`.
4. Apply visual-primitives Alpha-Geo-1 diagnostics to reject geometry-damaging
   mask candidates before exact eval.
5. Use bandit/BO as the next cheap-search controller; do not put PPO/PufferLib
   on exact eval until surrogate correlation is measured.

---

## Update - 2026-04-30T19:34Z Producer-Ready Sensitivity Queue

Readiness changes:

- The component sensitivity path now has both:
  - `experiments/profile_component_sensitivity.py` for CUDA-authored
    PoseNet/SegNet/combined maps, holdout response curves, stability JSON, and
    sample plan.
  - `experiments/build_component_sensitivity_manifest.py` for exact
    custody/manifest assembly.
- The producer can optionally assemble the manifest in the same run when
  `--archive`, `--contest-auth-eval-json`, and `--manifest-output` are supplied.
- This makes the next CUDA beta/OWV3/NWCS sensitivity run a direct producer
  run, not a hand-crafted artifact exercise.

Current blockers:

- Current Lightning SSH shell remains CPU-only/no torch. It cannot run exact
  eval or the new producer.
- The producer is verified structurally but has not yet produced authoritative
  maps because no CUDA runner is visible in this shell.

Verification:

- Expanded focused suite: `203 passed in 5.14s`.
- `py_compile`, `bash -n`, `git diff --check`: passed.

Next wall-clock command shape once CUDA is visible:

```bash
.venv/bin/python experiments/profile_component_sensitivity.py \
  --checkpoint <candidate renderer/checkpoint> \
  --video upstream/videos/0.mkv \
  --masks-mkv <candidate masks.mkv> \
  --poses <candidate optimized_poses.pt> \
  --upstream upstream \
  --output-dir <evidence/component_sensitivity> \
  --all-pairs \
  --device cuda \
  --archive <candidate archive.zip> \
  --contest-auth-eval-json <exact_cuda/contest_auth_eval.json> \
  --manifest-output <evidence/component_sensitivity_manifest.json>
```

The exact archive eval still comes first for score truth; the sensitivity
producer explains and gates scorer-sensitive allocation, not ranking.

---

## Update - 2026-04-30T19:40Z Promotion Readiness Delta

Readiness changes:

- NWCS Stage 5 export is no longer blocked on hidden architecture metadata:
  both NWCS remote scripts now import `_infer_asymmetric_config` in the exact
  heredoc that exports the `NWCS1` renderer container.
- Component sensitivity sample plans now preserve absolute pair IDs after
  top-k pair selection. This matters for reproducibility: calibration/holdout
  records now identify real dataset frame pairs rather than subset offsets.
- Lightning Batch Jobs OWV3 byte-feasible exact-eval dry-run was regenerated
  with `--studio pact`; prior dry-run with `studio=null` is superseded.

Current promotion blockers:

- No exact CUDA eval exists yet for the OWV3 byte-feasible archive. The archive
  remains byte-only evidence.
- No CUDA-authored `component_sensitivity_v1` artifact exists yet. Producer and
  assembler are ready, but authoritative maps/curves require CUDA.
- Current interactive Lightning shell remains non-authoritative unless it shows
  CUDA; use Lightning Batch Jobs or another verified T4/CUDA runner.

Verification:

- `60 passed in 1.45s` for sensitivity/NWCS/Lightning batch tests.
- `250 passed in 23.53s` for preflight/supply-chain/repro/config tests.
- Strict local supply-chain scan r4: OK, zero violations.
- `py_compile`, `bash -n`, `git diff --check`: passed.

Next wall-clock order:

1. Launch real Lightning Batch Job exact eval from the corrected dry-run spec
   once the Studio-backed job target is confirmed CUDA-visible.
2. Harvest only
   `archive.zip`, `contest_auth_eval.json`, `contest_auth_eval.adjudicated.json`,
   `adjudication_provenance.json`, logs, and custody metadata from that job.
3. Run CUDA component sensitivity profiling on the same candidate/custody pair.
4. Assemble and validate `component_sensitivity_v1`; only then use the maps for
   OWV3/NWCS promotion decisions or paper claims.
5. For NWCS, rerun build-only smoke after the import fix, then exact CUDA eval;
   do not promote tensor-only fallback metadata.

---

## Update - 2026-04-30T19:55Z Execution Readiness Delta

Readiness changes:

- Lightning exact eval runner is now fail-closed at job command level:
  supply-chain scan and CUDA runner preflight execute inside the Batch Job
  before archive copy and `contest_auth_eval.py`.
- Real OWV3 byte-feasible exact eval is submitted and running on Lightning
  Batch Jobs, not merely dry-run queued:
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x`.
- Lightning ops CLI can now refresh status from SDK job attributes and list
  provider machines; this avoids log-derived status and makes provider slugs
  (`g4dn.2xlarge`, etc.) reproducible.
- Component sensitivity readiness was intentionally downgraded from
  "producer-ready" to "diagnostic Fisher-proxy only." The current profiler
  records promotion blockers and refuses promotable manifest assembly.
- NWCS debug/fake sensitivity and build-only paths now stop before exact eval
  and record non-promotable provenance. Corpus replay now rejects unsafe paths
  and direct-dict schema bypasses.
- MCP helper processes were killed and persistent MCP server config entries
  were removed from active Codex/plugin config surfaces.

Current promotion blockers:

- OWV3 byte-feasible job is `Running`, but no exact score is claimable until
  artifacts are harvested and validated:
  `archive.zip`, `lightning_supply_chain_scan.json`,
  `lightning_runner_preflight.json`, `contest_auth_eval.json`,
  `contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
  `auth_eval.log`, and custody metadata.
- Component sensitivity remains non-promotable until a new official
  finite-difference component-response producer lands with
  symmetric/directional curves and exact CUDA custody.
- Alpha/NeRV still needs geometry-preserving redesign; Alpha-Geo-0 2px
  diagnostics fail exploratory gates for Lane 12 `jsonfix40`.
- Live Vast/Modal harvest has no newly score-promotable lane-local exact JSON
  plus archive/custody set.

Verification:

- Focused suite:
  `312 passed in 26.28s`.
- `py_compile`, `bash -n`, `jq empty`, and `git diff --check`: passed.
- Lightning SDK status refresh at `2026-04-30T19:55:00Z`: job `Running`.
- MCP post-cleanup process/config sweep: no live helper except the checking
  `rg`; no active MCP server config definitions found.

Next wall-clock order:

1. Continue Lightning status polling; harvest and validate exact artifacts
   immediately after completion.
2. Update the claim matrix only from validated CUDA JSON/adjudication, never
   from logs or queue metadata.
3. Implement the official component-response sensitivity producer and rerun
   against the exact candidate/custody pair.
4. Rerun NWCS build-only smoke under the new guards; dispatch exact eval only
   after validated sensitivity provenance exists.
5. Keep paper/report edits in lockstep with the claim matrix; GP v3 and UNIWARD
   v8 remain quarantined until CUDA/custody gaps close.

### 2026-04-30T20:00Z Lightning Path Repair

- First real OWV3 byte-feasible job failed before eval because the command
  wrote to `/teamspace/jobs/<underscore-name>/artifacts`, but Lightning SDK
  created `/teamspace/jobs/<hyphen-name>/artifacts`. This is an infrastructure
  failure, not compression evidence.
- `src/tac/deploy/lightning/batch_jobs.py` now normalizes default job artifact
  paths with `lightning_sdk_job_name()`.
- `scripts/launch_lightning_batch_job.py refresh-status` uses the same helper.
- Regression test added for underscore-to-hyphen artifact path selection.
- Rerun `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2`
  is submitted with corrected artifact path and unchanged SHA/byte/component
  gates. Last refresh `2026-04-30T20:04:33Z`: `Running`.
- No score claim exists. Harvest gate remains exact JSON/adjudication/custody
  only.
- Full focused verification after the repair: `313 passed in 24.19s`.

### 2026-04-30T20:15Z Readiness Delta

- Lightning r2 failed before eval because `/teamspace/jobs/<sdk-name>/artifacts`
  is read-only inside the running Studio Batch Job. The r2 failure is
  infrastructure-only and creates no compression evidence.
- R3 is submitted with writable output under the staged repo:
  `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
  Last refresh `2026-04-30T20:15:12Z`: `Pending`.
- Exact-eval queue records now distinguish `remote_output_dir` from
  `sdk_artifact_path`, and validation rejects `/teamspace/jobs/...` as a
  command output target.
- Alpha-Geo-1 readiness improved: NeRV training can use decoded baseline masks
  as ground truth with ZIP-safe custody and geometry/class checks before
  retraining spend.
- Component sensitivity readiness improved at the validator layer: promotable
  artifacts require official response metadata, symmetric/directional curves,
  finite gate specs, readout provenance, and stability thresholds. The current
  profiler remains diagnostic/non-promotable.
- NWCS readiness improved: build-only smoke is now explicit, non-promotable,
  and no-auth-eval; bad sensitivity values and hidden corpus sidecars fail
  closed.
- Strict preflight now includes repo-owned MCP server config detection.

Current blockers:

- R3 is `Running` as of `2026-04-30T20:18:59Z`; no exact artifacts exist yet.
- Component response producer still needs real CUDA finite-difference evidence.
- NWCS exact eval still requires validated sensitivity provenance and CUDA
  archive eval.
- No live Vast/Modal lane has newly promotable exact archive/eval/custody
  evidence beyond PFP16 A++.
- MCP helpers respawned from outside active config and were killed again.
  Active config search remains clean.

### 2026-04-30T21:05Z Readiness Delta

- R3 is now classified infrastructure-only: upstream eval crashed on missing
  `nvidia.dali`; no `contest_auth_eval.json`, no score claim, no kill claim.
- Exact-eval readiness tightened from "runner preflight exists" to "runner
  preflight is content-validated":
  - expected archive SHA/bytes are mandatory,
  - adjudication is mandatory,
  - stale output artifacts are deleted before each run,
  - pre/post supply-chain scans are distinct artifacts,
  - DALI requirements are direct URL + SHA-256 hash pinned,
  - DALI bootstrap/runner-preflight/supply-chain JSON contents are validated,
  - harvest rejects failed scans, non-CUDA runner preflight, bad DALI probes,
    missing adjudication, and missing expected archive metadata.
- Remote lane score-custody readiness tightened: contest-CUDA scripts now use
  literal `--device cuda`; preflight catches unguarded `AUTH_EVAL_DEVICE` and
  requires kept `eval_work` custody.
- Lightning supply-chain readiness tightened: installed bare PyPI `lightning`
  is forbidden entirely. Local strict scan at
  `.omx/state/lightning_supply_chain_scan_20260430_codex_preflight_metabugs.json`
  is clean (`status=OK`, `violation_count=0`).
- R4 is submitted:
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4`,
  command SHA
  `79bf98e80faa762456f5f0d35845a0326bee79d8285e89b871ded8a7d837ca60`,
  expected archive
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  `686557` bytes. Last refresh `2026-04-30T21:01:31Z`: `Pending`.

Current blockers:

- R4 has not produced validated artifacts yet. No OWV3 score claim exists.
- Continue polling Lightning status; harvest only from the writable remote
  output dir, then run `validate-artifacts --require-adjudication`.
- Component-response CUDA producer, NWCS exact eval, and Alpha retraining still
  require their own authoritative CUDA evidence before promotion.

### 2026-04-30T21:26Z Execution Readiness Delta

Live compute state:

- Vast.ai live inventory is empty:
  `.omx/state/vastai_show_instances_live_final_20260430.json` = `[]`.
- Modal has no live tasks (`Tasks=0` on all apps).
- No local `launch_lane_with_retry`, `remote_lane_19`, `remote_lane_20`,
  `modal run`, or rsync processes remain after cleanup.
- MCP helper processes were killed again; the post-kill sweep matched only the
  checking `rg`.

Readiness changes:

- Lane 19/20 duplicate relaunch class is fixed at the launcher layer:
  timestamped queue labels now map to a shared `logical_lane_key()` for both
  the advisory lock and live Vast duplicate check.
- Lane 19/20 forensic relaunch holds are now encoded in
  `.omx/state/dispatch_holds.json`; the launcher exits with
  `FATAL_DISPATCH_HOLD` before creating a Vast instance unless explicitly
  overridden after recorded Grand Council clearance.
- Vast dispatch readiness is stricter: no relaunch should proceed unless the
  logical lane key is clear or the operator explicitly uses
  `--allow-existing-label-prefix` / `--override-dispatch-hold` after
  recovery/destroy and review.
- Lightning r4 is no longer pending. It is an exact CUDA/T4 result packet with
  failed adjudication.

Lightning r4 evidence status:

- Artifact dir:
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4/`.
- Focused artifact harvest includes archive, eval JSON, provenance, DALI
  bootstrap/requirements, pre/post supply-chain scans, CUDA runner preflight,
  auth/adjudication logs, report, and `SHA256SUMS`. Inflated raw frames were
  intentionally not harvested.
- Exact CUDA/T4 score:
  `score_recomputed_from_components=1.0378905176070103`.
- Components:
  PoseNet `0.00319052`, SegNet `0.00402120`, bytes `686557`.
- Adjudication failed the SegNet relative gate:
  reference `0.00400656`, observed `0.00402120`, relative `1.003654`, cap
  `1.002`.
- `validate-artifacts` therefore fails because no adjudication provenance or
  adjudicated JSON exists. This is correct fail-closed behavior.

Current blockers:

- OWV3 r4 cannot enter the claim matrix as a promotion until Grand Council
  adjudicates the SegNet gate and either approves a revised gate with paper
  rationale or lands a mitigation/rerun that passes the existing gate.
- Lane 19 needs deterministic archive build, JSON adjudication, current
  frontier gates, and corrected profile/provenance before rerun.
- Lane 20 needs a non-static byte win before any further GPU eval on this
  anchor; current Ballé path is no-op/static fallback.
- HM-S/SA need SegMap pack/roundtrip contract repair; H-V3 needs channel-shape
  repair.

### 2026-04-30T21:49Z Readiness Delta

Readiness changes:

- H-V3 rerun blocker removed: the SegNet uncertainty loss now feeds RGB BCHW
  into SegNet and is regression-tested.
- HM-S/SA rerun blocker removed at pack-contract level: block-FP payloads are
  declared lossy, roundtrip MSE is recorded, and exact CUDA archive eval is
  the score gate.
- Lane 19/20 relaunch protection strengthened: launcher checks conditional
  clearance requirements even when a hold file is missing or marked cleared.
- Lightning security readiness improved: current strict scan is clean at
  `.omx/state/lightning_supply_chain_scan_20260430_codex_current.json`, and
  batch CLI import paths disable `lightning_sdk` PyPI version checks before
  import.
- Renderer KL readiness improved: positive KL/JBL auxiliary weights are
  explicit-scope only (`kl_distill_scope="segnet_aux"`), with primary/full-
  scorer KL blocked in `train_renderer` and a strict preflight guard.
- MCP readiness: discovered config files now have empty server maps and the
  strict MCP config preflight passes when pointed at those paths.

Current compute state:

- Vast.ai live inventory: empty at the latest sweep.
- Modal: `Tasks=0` on all listed apps.
- No MCP helper process remained after the latest kill/config sweep except the
  checking process itself.

Next wall-clock order:

1. Run OWV3 R5 path only through paired PFP16 calibration or
   SegNet-conservative candidate exact eval. Do not promote r4.
2. Generate CUDA component diagnostics for OWV3/NWCS candidate selection, then
   build `component_sensitivity_v1` only with finite-difference response and
   custody evidence.
3. Rerun H-V3 and SegMap only as repaired implementations with exact archive
   custody. No broad family kill or promotion from prior broken runs.
4. Keep Lane 19/20 on hold until their respective clearance gates are
   satisfied and deliberately reviewed.
5. Keep Lightning supply-chain scans in every exact-eval runner preflight.

Verification:

- `107 passed in 2.20s`.
- `py_compile`, shell syntax for touched shell scripts, targeted
  `git diff --check`, and strict readiness preflights passed.

### 2026-04-30T22:05Z Readiness Delta

Six-item execution status:

1. PFP16 is complete for implementation/readiness and remains the A++ score
   frontier.
2. Sensitivity-map/OWV3 work advanced to R5 readiness, but only as diagnostic
   Fisher-proxy evidence. Promotion still requires CUDA finite-difference
   component response, holdout stability, exact custody, and manifest assembly.
3. OWV3 R4 claim state is corrected: exact CUDA/T4 score packet exists, but it
   failed the SegNet component gate and is not promotable. R5 must be
   SegNet-conservative and pre-adjudicated.
4. Lane 12 NeRV/Alpha is build-only by default. Exact eval is blocked until
   decoded-baseline geometry diagnostics and pose-regeneration provenance are
   present.
5. The no-new-retraining-before-Lane12-L2 policy is now launcher-enforced.
   Build-only, harvest, and exact-eval-only lanes remain permissible.
6. J-NWC/J-NWCS corpus codec path is safer: plain J-NWC has build-only
   provenance, and all J-NWC/NWCS exact paths now use JSON adjudication with
   PFP16 A++ component gates before final records.

Telemetry/readiness:

- Vast: `uv run --no-sync vastai show instances --raw` returned `[]`.
- Modal: `uv run --no-sync modal app list` showed `Tasks=0` for every app.
- MCP: helper processes were killed and live-process preflight passed.
- Direct `modal`/`vastai` are not on this shell PATH; use `.venv/bin/...` or
  `uv run --no-sync ...` for local provider telemetry.

Verification:

- Integrated focused suite: `69 passed in 2.66s`.
- Subagent focused suites: `34`, `83`, `11`, and `29` tests passed for PFP16,
  Sensitivity/OWV3, Lane 12, and J-NWC/NWCS respectively.
- Strict remote-auth, launcher, and live-MCP preflights passed.

Next exact order:

1. Build/eval OWV3 R5 only from SegNet-conservative byte candidates with
   deterministic archive SHA/bytes and PFP16 paired calibration.
2. Run CUDA component-sensitivity producer work, but keep it diagnostic until
   official component response curves and manifest custody exist.
3. Generate Lane 12 Alpha build-only candidates, run Alpha-Geo diagnostics, and
   add pose-regeneration provenance before any auth eval.
4. Keep J-NWC/NWCS exact eval disabled unless validated sensitivity/corpus
   artifacts and adjudication gates are present.

### 2026-04-30T22:24Z Readiness Closeout Delta

Current deploy readiness:

- Provider state is idle: Vast has no live instances, Modal has no active
  tasks. Stale local dispatch rows/locks must be reconciled before any retry.
- MCP is currently clean by strict preflight after killing respawned helper
  processes. Keep `check_no_live_mcp_processes(strict=True)` in pre-submit and
  pre-provider flows.
- Lightning SDK auth is sufficient for machine discovery and dry-run queue
  plans. SSH staging is not yet sufficient for archive custody, even after the
  setup script generated `~/.ssh/lightning_rsa`.
- OWV3 R5 exact eval is ready at the plan/candidate level, not the execution
  level. The remote archive must be staged or built inside Lightning before a
  spendful Batch Job is scientifically admissible.
- Exact-eval artifact validation now supports forensic failures: component
  gate violations produce adjudication artifacts and validate as
  `promotion_eligible=false`.

Readiness blockers:

1. Lightning SSH public-key denial blocks reproducible staging.
2. No official `component_sensitivity_v1` manifest exists.
3. Lane 12 lacks L2 clearance, passing Alpha-Geo diagnostics, and
   pose-regeneration provenance.
4. J-NWC/NWCS lack validated real sensitivity/corpus artifacts for exact eval.
5. `scripts/lightning_repro_workspace.py` has an index/worktree hygiene issue:
   staged deletion plus untracked same-path file. Resolve deliberately before
   commit or deploy packaging.

Verification:

- Focused tests: `137 passed in 3.27s`.
- Shell syntax, Python compile, targeted whitespace checks, and strict
  preflights passed.
