# Codex Memory — Shannon Floor Orchestration, 2026-04-30

## Role

Codex is responsible for implementation orchestration with the Grand Council
and Skunkworks Council: contest-grade rigor, scientific/mathematical audit,
full implementation hardening, extreme parallelism, and shortest-wall-clock
drive toward the Shannon theoretical floor.

## Evidence Discipline

- Do not promote, kill, or rank lanes from predictions, byte-only reports,
  CPU/MPS runs, smoke tests, or memory-only scores.
- Grade A score-grade requires exact archive custody, SHA-256, CUDA eval,
  600 samples, recomputed score, and the canonical
  `archive.zip -> inflate.sh -> upstream/evaluate.py` path.
- Grade A++ 1:1 contest-grade additionally requires T4/equivalent hardware,
  clean manifest/provenance, payload closure, and contest-budget inflate.
- Current verified frontier is Lane G v3 PFP16 A++:
  `1.043987524793892`, archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  bytes `686635`, Lightning AI Tesla T4, `gpu_t4_match=true`.

## Landed By Codex This Turn

- `src/tac/sensitivity_map.py`: per-Conv2d-channel sensitivity artifact
  contract, CUDA-authoritative guard, validation, save/load, CV distance.
- `src/tac/owv3_sensitivity_weighted.py`: OWV3 mixed-channel renderer archive;
  protected high-sensitivity Conv2d output channels stay FP16, lower channels
  use OWV2 water-fill + arithmetic coding.
- `OWV3` registered in `src/tac/codec_magic_registry.py`.
- `submissions/robust_current/inflate_renderer.py`: OWV3 dispatch added;
  `.nrv` mask resolver fallback added for Lane 12.
- `experiments/contest_auth_eval.py`: `.nrv` archive members allowed.
- Tests added for sensitivity maps, OWV3 mixed-channel round trip, inflate
  dispatch, `.nrv` auth validation, and `.nrv` resolver discovery.
- Docs updated:
  - `.omx/research/shannon_floor_execution_readiness_20260430.md`
  - `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
  - `.omx/research/external_research_intake_shannon_floor_20260430.md`

## Verification Already Run

- `.venv/bin/python -m pytest src/tac/tests/test_sensitivity_map.py src/tac/tests/test_owv3_sensitivity_weighted.py src/tac/tests/test_runtime_guards_pass_3.py src/tac/tests/test_contest_auth_eval.py -q`
  - Result: `34 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_owv2_renderer_archive_inflate.py::test_owv2_archive_inflate_renderer_dispatch src/tac/tests/test_imps_renderer_archive.py::test_imps_registered_in_codec_magic_registry_synthetic -q`
  - Result: `2 passed`
- `py_compile` clean for touched runtime files.

## Active Next-Turn Priorities

1. **PFP16 parser/adjudication fix:** `contest_auth_eval.json` is
   authoritative; legacy remote provenance has invalid
   `contest_cuda_score=100.0` and `hard_kill_triggered=true` fields. The
   adjudicator now emits scoped regression fields for future runs.
2. **PFP16 A++ evidence:** if it remains the submission candidate, rerun the
   exact archive SHA on T4/equivalent hardware with contest-budget inflate
   proof.
3. **OWV3 builder:** add Lane G v3 + OWV3 stack archive builder with exact
   provenance and deterministic zip.
4. **Sensitivity conversion:** convert existing per-weight Fisher artifacts into
   OWV3 per-channel sensitivity maps, with CUDA metadata and train/holdout CV.
5. **Lane 12 NeRV closure:** full CUDA training and exact archive eval remain
   blocked until clean contest dependency closure for `tac.nerv_mask_codec` is
   proven; `.nrv` auth whitelist and resolver are now unblocked.
6. **IMP / hidden gems:** harvest Lane 17 IMP and launch high-EV recovery lanes
   in parallel: Q-FAITHFUL, H-V3, SegMap clone, FL chunked, MAE-V.
7. **γ coordinator later:** MDL/ADMM/static entropy/hyperprior-lite only after
   measured α and β/renderer components exist.

## Hard Rules

- Legacy primary KL-distill remains promotion-ineligible except explicitly
  fenced forensic/SegNet-aux experiments that remain promotion-gated until
  exact evidence clears PoseNet.
- Adaptive rebalance remains retired.
- Neural/runtime artifacts used by inflate must be inside `archive.zip` or
  fixed contest code.
- Do not modify upstream scorer files.
- Do not call OWV3, IMP, or recovered lanes Grade A until exact archive CUDA
  eval exists. Current Lane 12 NeRV `jsonfix40` is retired by exact-CUDA
  regression for that implementation/config only. PFP16 now has A++ evidence.

## Later Update — Same Day

Additional landed work:

- `experiments/build_lane_g_v3_owv3_stack.py`: deterministic OWV3 stack archive
  builder with provenance.
- `experiments/convert_fisher_to_owv3_sensitivity_map.py`: converts
  `hessian_per_weight.pt` to `tac_score_sensitivity_map_v1` using per-channel
  Fisher sum; missing layers protect by default.
- `experiments/profile_hessian_per_weight.py`: fixed mask grayscale decoding
  bug (`class * 63` pixels are now converted back to class IDs before renderer
  embedding).
- `src/tac/submission_archive.py`: added `masks_nrv` manifests, `masks.nrv`
  validation, and deterministic ZIP writing.
- `src/tac/tests/test_profile_hessian_mask_decode.py` and
  `src/tac/tests/test_owv3_sensitivity_conversion.py` added.
- Paper/writeup blueprint:
  `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`.

Verification:

- Targeted suite: 62 passed.
- Neighboring archive/integration suite: 17 passed.
- OWV3 builder smoke with synthetic CPU all-protect map succeeded, but archive
  was larger. This is smoke-only proof of builder mechanics, not a candidate.

Known blockers:

- No real `hessian_per_weight.pt` exists locally yet. Run CUDA Fisher after the
  mask-decode fix.
- Superseded: PFP16 now has T4 A++ evidence and the parser bug class has a
  structural fix. Remaining work is final bundle/source-manifest polish.
- Superseded: Lane 12 NeRV `jsonfix40` completed exact CUDA eval and failed
  hard. Remaining alpha work is redesign, not first eval.

## PFP16 Exact CUDA Harvest — Same Day

PFP16 exact CUDA evidence is preserved locally under
`experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.

Authoritative `contest_auth_eval.json` fields:

- `final_score=1.04`
- `score_recomputed_from_components=1.0440481283330025`
- `avg_posenet_dist=0.0034602`
- `avg_segnet_dist=0.0040083`
- `archive_size_bytes=686635`
- archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`

This RTX 4090 run is superseded for final contest wording by the later
Lightning AI Tesla T4 A++ run with the same archive SHA.

Current targeted bug-class fix: remote provenance/adjudication parsing. The
harvested `remote_provenance.json` contains invalid
`contest_cuda_score=100.0`, `hard_kill_triggered=true`, and
`lane_status=HARD_KILL_REGRESSION`; those fields are superseded by
`contest_auth_eval.json`. The parser bug class has since been structurally
fixed in the remote adjudication path.

## Remote Harness Hardening Landed - 2026-04-30

- Added `scripts/adjudicate_contest_auth_eval.py` to adjudicate exact archive
  evals from `contest_auth_eval.json` only.
- Refactored PFP16, Ω-W-V2, and Lane 8 remote stack scripts off fragile
  auth-log regex parsing.
- PFP16 now asserts archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` before
  CUDA eval.
- Removed remaining remote-lane fallbacks that scraped `grep -Eo '{.*}'` from
  auth logs; adjacent scripts now require `contest_auth_eval.json`.
- Hardened `scripts/launch_lane_with_retry.py`: phase2 timeouts now exceed
  launcher poll windows, and phase2-launch timeout returns
  `UNKNOWN_REMOTE_STATE` instead of blindly retrying.
- Added strict preflight:
  `check_remote_lane_auth_eval_json_adjudication`.
- Added `src/tac/tests/test_remote_auth_eval_hardening.py`; targeted suite
  passed 6/6 plus shell syntax and Python compile checks.

## 2026-04-30 Swarm + DX Self-Protection Update

Codex orchestrated the six-item next turn against the Grand Council plan:

- Active harvest monitor: HM-S `35885106` and Lane 19 `35899850` remain live;
  no lane-local `contest_auth_eval.json` was available at the checkpoint.
- PFP16 A++: runbook created at
  `.omx/research/pfp16_a_plus_plus_exact_t4_eval_runbook_20260430.md`; exact
  archive SHA remains
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
  Current PFP16 evidence is Grade A score-grade only, not A++.
- OWV3/Fisher: CUDA host script and runbook created
  (`scripts/remote_lane_g_v3_owv3_fisher_stack.sh`,
  `docs/owv3_fisher_runbook.md`); no CUDA Fisher artifact landed locally.
- Lane 12: dependency-closure tests added for `.nrv` codec/import/archive/auth
  discipline; worker verification was 40 passed. Exact CUDA `.nrv` eval is
  still required.
- Hidden-gem wave: SegMap clone and H-V3 scripts were hardened to use
  JSON-only auth eval adjudication. Q-FAITHFUL remains gated/high-risk due
  KL-distill-like machinery.

DX/harness hardening landed:

- `scripts/launch_lane_with_retry.py` now has per-label advisory lock,
  live Vast label-prefix guard, signal-safe child process-group cleanup, and
  fail-closed `UNKNOWN_EXISTING_LABEL_PREFIX`.
- New strict preflight:
  `check_launch_retry_wrapper_singleflight_and_signal_safe`.
- `src/tac/tests/test_remote_auth_eval_hardening.py` expanded to 9 tests.
- New adjacent progress doc:
  `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`.

Dispatch state:

- Empty duplicate `35905846` was destroyed.
- Staged `35905118` failed NVDEC and auto-destroyed.
- Clean SegMap clone dispatch succeeded on `35906669`
  (`lane_sa_segmap_clone_2026-04-30_codex_a2`,
  `root@ssh2.vast.ai:26668`, RTX 4090, `$0.2539/hr`).
- Remote proof: `SETUP_COMPLETE`, heartbeat present, Stage 2 training reached.
- H-V3 dispatched through the hardened wrapper:
  - Attempts 1/2 hit slow SSH/readiness and were retired.
  - Attempt 3 failed NVDEC and auto-destroyed.
  - Attempt 4 succeeded as `35907873`
    (`lane_h_v3_joint_halfframe_2026-04-30_codex_a4`,
    `root@ssh5.vast.ai:27872`, RTX 4090, `$0.2731/hr`).
  - At the checkpoint it was still in setup installing `nvidia-dali-cuda120`;
    do not mark it training or Grade A until setup/training/auth-eval evidence
    exists.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile`: retry launcher, preflight, adjudicator clean.
- `check_launch_retry_wrapper_singleflight_and_signal_safe`: 0 violations.
- `check_remote_lane_auth_eval_json_adjudication`: 0 violations.
- `git diff --check`: clean.

## 2026-04-30T17:35 Six-Item Resumption After Security Interruption

Lightning PyPI compromise handling landed and execution resumed.

Security state:

- `lightning_sdk==2026.4.10` was audited against Mini Shai-Hulud indicators;
  no payload evidence found.
- Use `lightning-sdk`, never PyPI `lightning`, in this project.
- `src/tac/preflight.py` now blocks PyPI `lightning`, bad pins, unsafe
  `lightning --version` probes, planted repo paths, hidden `_runtime`, and
  known IOC hashes.
- `src/tac/deploy/cloud_deploy.py` checks `lightning-sdk` metadata instead of
  executing `lightning --version`.
- `src/tac/deploy/lightning/batch_jobs.py` sets
  `LIGHTNING_DISABLE_VERSION_CHECK=1` before SDK import.

New adjacent source docs:

- `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`
- `.omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`

Current six-item queue:

1. Harvest live lanes only through lane-local `contest_auth_eval.json`.
2. Finish official Lightning Batch Jobs status/harvest/mirror/adjudication.
3. Maintain PFP16 A++ deploy bundle and paper provenance packet.
4. Redesign OWV3/Fisher around charged bytes and exact eval.
5. Redesign Alpha/Lane 12 around pose-preserving masks, starting Alpha-Geo-0.
6. Continue paper/writeup claim matrix and KL/DX hardening.

Swarm allocation:

- Lightning Batch Jobs worker.
- Vast harvest explorer.
- OWV3/Fisher explorer.
- Alpha/Lane 12 explorer.
- PFP16/paper explorer.

PFP16 A++ remains the deploy baseline until a stacked exact archive beats it
under the same evidence discipline.

Next highest-leverage actions:

1. Monitor/harvest HM-S, Lane 19, and SA only through lane-local
   `contest_auth_eval.json` plus archive/provenance bundle.
2. Finalize the PFP16 A++ provenance bundle.
3. Redesign OWV3 after the Modal size-regression smoke, then run CUDA Fisher ->
   sensitivity map -> archive -> exact CUDA eval.
4. Redesign alpha after Lane 12 NeRV `jsonfix40` exact failure.
5. Monitor H-V3 until `SETUP_COMPLETE`, Stage 1 training, and eventual
   `contest_auth_eval.json`; keep Q-FAITHFUL gated until KL risk is removed or
   exact CUDA evidence clears it.

## 2026-04-30T16:16Z XHigh Swarm Update

User provided the new Lightning AI SSH endpoint:
`ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`. Treat Lightning as the
preferred promotion-grade exact-eval path, especially for PFP16 A++ T4/equivalent
rerun. Modal credits remain available and should be used for build/smoke,
ablation, Fisher/sensitivity generation, and other non-promotion acceleration;
rerun candidates on exact CUDA archive eval before any Grade A/A++ claim.

Lane 12 NeRV is complete and failed hard-kill:

- Evidence:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`.
- Exact CUDA archive eval, 600 samples, RTX 4090 (`gpu_t4_match=false`).
- `score_recomputed_from_components=26.03719330455429`, rounded `26.04`.
- PoseNet `49.77849960`, SegNet `0.03528685`, archive `296478` bytes.
- Archive SHA in nested provenance:
  `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`.
- Verdict: not promotable. Current NeRV mask replacement collapses PoseNet
  geometry and should not receive further spend without a pose-preserving
  redesign.

KL hardening from the xhigh Grand Council subagent:

- Primary `loss_mode="kl_distill"` is forensic-only and promotion-ineligible.
- Ambiguous primary-KL configs are rejected.
- SegNet-only KL auxiliary use is explicitly scoped and temperature-plumbed.
- Root cause remains unit scaling: spatial KL with `batchmean` on `[B,C,H,W]`
  over-pressured the auxiliary by image area, matching the historic collapse.

OWV3/Fisher:

- Two hardened Vast attempts failed NVDEC and auto-destroyed.
- Superseded: a Modal smoke later produced Fisher/sensitivity/build artifacts,
  but the archive was larger and had no exact eval. Next route is encoder/config
  diagnosis, then Lightning exact eval only if the archive is rate-viable.

Vast live state at 2026-04-30T16:16Z:

- HM-S `35885106`: live/training, heartbeat fresh, uses `variant=kl_distill`;
  forensic/high-risk until exact evidence clears it.
- Lane 19 `35899850`: live/training, heartbeat fresh, logit-margin profile.
- SA `35906669`: live/training, Stage 2, heartbeat fresh.
- H-V3 `35907873`: live/training, Stage 1 joint half-frame, heartbeat fresh.
- Lane 12 instance destroyed after harvest; no live Lane 12 remains.

DX guardrails added:

- `scripts/reconcile_vast_dispatch_state.py`.
- `src/tac/tests/test_reconcile_vast_dispatch_state.py`.
- Reconciler observed `live=4`, `tracker=204`, `active_dispatches=3`,
  `tracker_missing_live=200`, `active_missing_live=3`, `live_missing_active=3`;
  state ledgers are stale and must not be trusted over live API/artifacts.
- PPID=1 orphan MCP processes were killed and re-scan is clean.

Verification after the xhigh wave:

- Focused hardening/regression suite: 118 passed.
- `bash -n`: clean for PFP16 A++ helper, OWV3/Fisher remote script, and Lane 12
  NeRV remote script.
- `py_compile`: clean for touched Python scripts/modules.
- `git diff --check`: clean.

## 2026-04-30T16:25Z PFP16 A++ And Backend Routing

PFP16 now has exact T4 A++ evidence:

- Evidence directory:
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/`.
- Exact archive SHA:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Archive bytes: `686635`.
- Eval chain:
  `contest_auth_eval.py --device cuda -> inflate.sh -> upstream/evaluate.py`.
- Hardware: Lightning AI Tesla T4, driver `580.126.09`.
- `gpu_t4_match=true`, `n_samples=600`.
- Rounded score `1.04`, recomputed `1.043987524793892`.
- PoseNet `0.00346442`, SegNet `0.00400656`, rate `0.01828808`.
- Grade: A++.

Lightning operational details:

- User-provided SSH:
  `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- The default remote `/home/zeus/content/pact` was stale/non-git for this eval.
- A hermetic staged tree was used:
  `/home/zeus/content/pact_pfp16_exact_20260430T1625Z`.
- Upstream was `/home/zeus/content/upstream`, git `c5e1274`.
- Use staged Lightning trees for future exact-eval candidates rather than
  mutating stale default workspaces.

Modal routing:

- Modal is installed/authenticated locally with client `1.4.1` and profile
  `adpena`.
- Modal scores remain advisory because current Modal training wrapper forces
  CPU eval.
- Use Modal credits for OWV3 Fisher/build-only smoke/full artifacts and
  ablations; promote only after Lightning exact CUDA archive eval.

## 2026-04-30T16:45Z Latest Supersession

- Source docs and progress ledgers now treat PFP16 A++ as the controlling
  frontier, not merely Grade A.
- OWV3/Fisher Modal smoke landed artifacts but regressed size:
  archive `912971` bytes, `+218897` vs Lane G v3, SHA
  `710cba0c7c490b13db8b0aee897dd0f33cb8b66a6ed229466bf0d1aea392f5a3`.
  This is suspicious negative smoke only, not a method-family kill.
- `experiments/build_lane_g_v3_owv3_stack.py` now fails closed on archive size
  regression by default; `--allow-size-regression` is explicit smoke/debug.
- Dykstra ceiling recorded in paper blueprint: sub-`0.30` requires
  `archive_bytes <= 450545` even with zero distortion. PFP16 A++ is deployable
  but not Shannon-floor by itself.
- All MCP server configs were disabled and all MCP processes killed per user
  request. Active configs edited: Codex, Claude, Cursor, Claude Desktop; backups
  have timestamp `20260430T163944Z`.
- A read-only xhigh sidecar is auditing
  `https://github.com/Lightning-AI/pytorch-lightning` for useful primitives
  that preserve deterministic contest evidence.
- Lightning repo audit returned: borrow Fabric-style seed/rank-zero/callback
  and checkpoint-interface ideas only; avoid full Trainer, DDP exact eval,
  LightningCLI sweeps, and remote loggers in canonical contest paths.
- Every unexpected/disappointing result must now get mitigation/stacking and
  leaderboard-reverse-engineering analysis before scoped retirement language:
  ask whether hybrid residuals, fallback routing, side-info accounting, or
  full-stack archive allocation can rescue the signal.
- Lightning AutoResearch/batch-jobs audit: use official Lightning Batch Jobs
  directly for auditable lane/eval queues; AutoResearch is conceptually useful
  but not the promotion dispatcher. Prefer one T4 exact-eval queue, faster GPUs
  for training, immutable/content-hashed inputs, new job names for retries, and
  local mirroring of all job artifacts before deletion.

## 2026-04-30T17:00Z Kill-Discipline Vocabulary Hardening

- Grand Council audit confirmed the policy: bad evidence can retire a measured
  implementation/config, but broad family/method kills require independent
  exact evidence or a mathematical impossibility argument plus clean consensus.
- `scripts/adjudicate_contest_auth_eval.py` now uses
  `--regression-threshold`, emits `REGRESSION_REVIEW_REQUIRED`, and records
  `regression_triggered`, `regression_threshold`, and
  `regression_scope=measured_implementation_config_only_pending_review`.
- Future provenance no longer writes `hard_kill_triggered` by default; old
  `hard_kill_triggered=true` / `HARD_KILL_REGRESSION` fields are legacy
  parser/adjudicator artifacts and must be ignored when canonical
  `contest_auth_eval.json` exists.
- Active adjudicator-calling remote scripts were moved to
  `--regression-threshold`. Run-abort thresholds are control limits, not
  scientific kill evidence.

## 2026-04-30T17:05Z CUDA Auth Eval Is The Only Score Truth

- User re-emphasized: MPS/local paths materially mess up score/auth-eval signal.
- For GPU-dependent score or signal claims, exact CUDA auth eval on exact
  archive bytes is the only reliable source of truth.
- Canonical path: `archive.zip -> inflate.sh -> upstream/evaluate.py`, normally
  via `experiments/contest_auth_eval.py --device cuda`.
- Canonical artifact: `contest_auth_eval.json`; do not parse human logs when
  structured JSON exists.
- MPS, CPU, local proxy scorers, and non-canonical renderer checks are
  development-only and must never promote, rank, kill, retire, validate a
  stack, or anchor paper claims. CUDA auth eval wins every disagreement.

## 2026-04-30T17:43Z Swarm Return And KL/Lightning Hardening

- Active Vast lanes at swarm return: HM-S `35885106`, Lane 19 `35899850`,
  SA `35906669`, H-V3 `35907873`. None had lane-local `contest_auth_eval.json`,
  adjudication JSON, lane-local archive ZIP, or auth-eval log yet; watch-only.
- PFP16 A++ score evidence remains strong:
  score `1.043987524793892`, archive `686635`, SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
  Paper/deploy bundle still needs source custody cleanup and stale
  hard-kill-provenance quarantine.
- OWV3/Fisher is smoke-only until implementation preserves ASYM bytes and
  proves post-ZIP byte savings. FP16 protected/fallback layers are the current
  byte-regression class.
- Alpha/Lane 12 `jsonfix40` exact negative is scoped to the measured NeRV mask
  replacement. Renderer and pose bytes were identical to base; only `masks.nrv`
  changed. Proceed with geometry diagnostics and pose rescue before retraining.
- Lightning Batch Jobs wrapper now supports expected archive SHA/byte checks,
  command-hash queue metadata, exact CUDA JSON preservation, adjudication
  wiring, local artifact validation/mirroring, and state-attached harvests.
- New KL ledger:
  `.omx/research/kl_distill_hardening_status_20260430_codex.md`.
  Primary scorer KL is forensic-only; SegNet-aux KL remains experimental under
  exact PoseNet/component gates. `SegMapTrainer` now fails closed unless
  `kl_distill_scope=="segnet_aux"`.
- Claim matrix rows C-011/C-012/C-013 encode KL policy, OWV3 byte gate, and
  Alpha/NeRV scoped-redesign status.

## 2026-04-30T17:50Z Verification Green And Additional Hardening

- KL council sidecar also found an `optimize_poses.py` controller-only no-op:
  `--kl-distill-snr-target` could make the effective KL weight positive while
  GT pairs were materialized only for static `--kl-distill-weight > 0`.
  Fixed: `kl_distill_active` is true for static KL or SNR-controller KL, GT
  pairs materialize for both, logging uses effective KL weight, and Lane PS
  warnings understand controller-active KL.
- Generic Trainer checkpoint metadata now records KL scope, weight,
  temperature, forensic opt-in, and promotion eligibility.
- Preflight KL roundtrip scanner now includes `src/tac/segmap_renderer.py`.
- OWV3/Fisher converter default is now `missing_policy="error"`; remote
  OWV3/Fisher script passes `--missing-policy error`. `protect` is smoke/debug
  only.
- Verification at 17:50Z: `py_compile` passed, OWV3 script `bash -n` passed,
  focused pytest suite passed `291 passed`, and `git diff --check` passed.
- Lightning Batch Jobs dry run against PFP16 A++ identity succeeded with
  command hash
  `895eae34fc47a2d3211511f9bea4a3cdbab97a66876cf6dc8f9055d426c8630d`;
  no real Lightning API call was made.

## 2026-04-30T18:08Z Six-Item Swarm Greenup

Latest xhigh swarm landed and was reviewed:

- Lightning/adjudication: `scripts/adjudicate_contest_auth_eval.py` and
  Lightning Batch Jobs now support absolute and relative PoseNet/SegNet
  component gates. They reject before accepting/copying artifacts if component
  collapse is detected even when total score is in band.
- Active harvest: no live Vast lane is harvestable. HM-S `35885106`, Lane 19
  `35899850`, SA `35906669`, and H-V3 `35907873` still had no lane-local CUDA
  `contest_auth_eval.json` at the read-only probe.
- Alpha: `experiments/diagnose_nerv_geometry.py` and
  `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py` landed. Alpha-Geo-0
  diagnostics are CPU empirical only, not score evidence, and include global
  disagreement, class confusion/F1, boundary drift, stable-region false flips,
  transition F1, speckles, component jumps, and worst pairs. ZIP member loading
  was hardened against traversal.
- OWV3: promotion default is now `fallback_action="keep_asym"`. The old FP16
  fallback is explicit `diagnostic_fp16` and non-promotable. Builder records
  byte plan, deterministic ZIP rebuild proof, manifest, and PFP16 A++ frontier
  comparator gate. No CUDA score was run.
- PFP16 custody: final deploy bundle now quarantines stale legacy parser fields
  and makes `eval/contest_auth_eval.json` the only score authority. Archive
  unchanged: SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `686635` bytes.
- J-NWC corpus: Bernoulli worker is still active. Required output is
  deterministic corpus manifest, fail-closed corpus selection, no fake/random
  sensitivity in promotable J-NWCS scripts, and focused tests.
- MCP: exact `chrome-devtools-mcp` / `rbx-studio-mcp` processes were killed
  again after respawn. Project-level MCP configs remain empty/disabled, but the
  outer runtime may respawn helper processes when MCP integrations are
  advertised.

Verification at this checkpoint:

- Focused cross-lane suite: `77 passed in 3.56s`.
- Alpha suite after zip-slip hardening: `8 passed in 1.02s`.
- `py_compile`, `bash -n`, PFP16 custody JSON/`jq`, archive SHA/byte checks,
  and `git diff --check` passed.

Next order:

1. Integrate Bernoulli/J-NWC corpus hardening and run its tests.
2. Run Alpha-Geo-0 against Lane 12 `jsonfix40` versus Lane G v3/base masks.
3. Generate CUDA authoritative Fisher/sensitivity map before any OWV3 exact eval.
4. Use Lightning Batch Jobs for exact PFP16/next-candidate eval queue when
   credentials/session context permit.

## 2026-04-30T19:08Z J-NWC/NWCS And Sensitivity Greenup

- Component-sensitivity validator landed:
  `src/tac/component_sensitivity_artifact.py` with tests in
  `src/tac/tests/test_component_sensitivity_artifact.py`. It validates
  `component_sensitivity_v1` manifests for CUDA-only promotion, required
  PoseNet/SegNet/combined maps, calibration/holdout, response curves, exact
  custody, finite metrics, and rejects debug/smoke/fake/random/proxy markers.
- J-NWC/NWCS hardening landed:
  - `train_neural_weight_codec.py` seeds torch before codec construction.
  - `build_corpus_from_manifest(..., replay_root=...)` supports relocated
    deterministic corpus replay with size/SHA/shape/dtype/block checks.
  - J-NWC/J-NWCS/J-NWCS-EC remote scripts use zip-safe anchor extraction,
    CUDA-only `AUTH_EVAL_DEVICE`, artifact SHA/byte custody, and NWCS
    sensitivity provenance gates.
  - Promotable NWCS sensitivities require anchor archive SHA, anchor renderer
    SHA, corpus manifest SHA, block size, parameter names/shapes/block counts,
    and finite nonnegative values. Raw shape-only sensitivity is debug-only.
- NWCS renderer format landed:
  - `src/tac/neural_weight_codec_sensitivity.py` now has `NWCS1` container
    helpers and strict parser validation.
  - `src/tac/renderer_export.py` detects/loads
    `neural_weight_compression_sensitivity_v1`.
  - `submissions/robust_current/inflate_renderer.py` dispatches `NWCS1`.
  - Remote NWCS scripts emit `NWCS1` containers instead of raw concatenated
    `[name][blob]` streams.
- Verification at this checkpoint:
  - `bash -n` passed for J-NWC/J-NWCS/J-NWCS-EC scripts.
  - Focused J-NWC/NWCS/component suite passed: `60 passed in 2.85s`.
- No J-NWC/NWCS score claim exists yet. Next valid step is real validated
  sensitivity artifacts, a build-only inflate-dispatch smoke, then CUDA exact
  auth eval with canonical JSON/archive/provenance custody.

Update to 19:08Z greenup verification:
- Focused J-NWC/NWCS/component/Lightning suite passed: `64 passed in 2.88s`.
- Lightning source/artifact sync refreshed with
  `.omx/state/shannon_greenup_20260430_jnwcs_r1_manifest.json`: `1078` files,
  `18724610` bytes, remote SHA verification OK. Environment record still used
  `--no-install` system Python with no torch; exact eval needs locked runtime
  install or explicit CUDA `PYBIN`.

## 2026-04-30T19:14Z OWV3 Byte Feasible, Lightning Scan Utility, And New Swarm

- OWV3 byte sweep landed:
  `experiments/sweep_owv3_byte_plan.py` and
  `src/tac/tests/test_sweep_owv3_byte_plan.py`.
- Best byte-only OWV3 candidate:
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip`,
  `686557` bytes, SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  `-78` bytes versus PFP16 A++ `686635`.
- This is not a score claim. It can only promote after exact CUDA auth eval on
  the exact archive bytes with component gates.
- Lightning staging bug fixed in `scripts/lightning_repro_workspace.py`: remote
  generated Python now emits `None` for `python_bin_requested=None`, not JSON
  `null`. Test coverage added in `test_lightning_repro_workspace.py`.
- OWV3 byte-feasible archive staged to Lightning:
  `.omx/state/owv3_byte_feasible_repro_20260430_r1_manifest.json`, `1081`
  files, `17674947` bytes, manifest SHA
  `5fde235b76d19c991d489ce603aa640b391bb46b235ef866d0b7095230c0790e`.
- Lightning Batch Jobs dry-run created:
  `owv3_byte_feasible_exact_cuda_20260430_codex_dryrun`, command hash
  `e8551610ddb813ae6d0ee4857c3f110a22affa201ce64333624709bbeee15e89`.
  Current SSH shell lacks `nvidia-smi`, so do not submit exact eval until a
  CUDA-visible runtime is confirmed.
- Lightning supply-chain scan utility landed:
  `scripts/scan_lightning_supply_chain.py` and
  `src/tac/tests/test_lightning_supply_chain_scan.py`.
  Local strict scan output:
  `.omx/state/lightning_supply_chain_scan_20260430_codex.json`; status OK,
  zero violations, `lightning-sdk==2026.4.10`, no PyPI `lightning` or
  `pytorch-lightning`.
- Remote read-only Lightning Studio IOC scan found no compromised
  `lightning==2.6.2/2.6.3` dist-info, hidden `_runtime`, or known planted repo
  indicators in checked roots.
- Alpha-Geo-0 now ran Lane 12 `jsonfix40` versus both Lane G v3 and Lane A/base
  masks. Both diagnostic outputs match: global disagreement
  `0.012303928799099393`, transition disagreement `0.009507171571470149`,
  transition F1 `0.095099661402374`. Treat as CPU diagnostic only.
- Verification: focused cross-slice suite `126 passed in 3.64s`, `py_compile`
  passed for touched Python, `bash -n` passed for J-NWC/NWCS/OWV3/active
  remote scripts, MCP process sweep clean except the checking `rg`.
- Progress/claim docs updated:
  `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`,
  `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`,
  `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`, and
  `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`.
- AGENTS.md now requires running `scripts/scan_lightning_supply_chain.py
  --strict` and preserving JSON before trusting new exact-eval runners.
- New xhigh agents started:
  Averroes for KL distill hardening review, Feynman for arXiv 2604.26919
  research intake, Volta for PufferLib/RL/visual-primitives applicability, and
  Meitner for PoseNet/SegNet perturbation/profiling tooling audit.

## 2026-04-30T19:22Z Component Sensitivity Assembler, KL Fence, FP4 Fix

- Swarm reports completed:
  - `external_research_arxiv_2604_26919_shannon_floor_20260430_agent.md`:
    no direct codec path; use adaptive warm-ramp, sparse top-k, and dual
    readout validation as diagnostic patterns only.
  - `pufferlib_rl_visual_primitives_shannon_floor_20260430_agent.md`: use
    bandit/BO and local-model triage before PufferLib/PPO; visual primitives
    are useful for Alpha geometry preservation; no score claims.
  - `posenet_segnet_perturbation_tooling_audit_20260430_agent.md`: current
    missing bridge is deterministic component sensitivity production and
    response-curve custody; flagged FP4 mask decode/CPU diagnostic risks.
  - `kl_distill_hardening_grand_council_review_20260430_agent.md`: primary KL
    forensic-only; `segnet_kl` needed fencing.
- Added `experiments/build_component_sensitivity_manifest.py` plus
  `src/tac/tests/test_build_component_sensitivity_manifest.py`. It assembles
  `component_sensitivity_v1` manifests from real maps, response curves,
  stability JSON, exact CUDA eval JSON, archive, checkpoint, video, and
  upstream tree. It materializes SHA/byte custody and fails closed on non-CUDA
  eval JSON, wrong sample count, missing tensors, or missing response holdout
  error.
- Hardened `experiments/profile_fp4_layer_sensitivity.py`: grayscale mask luma
  is remapped to class IDs before renderer use; CPU requires
  `--allow-diagnostic-cpu`; metadata records `promotion_eligible` and evidence
  class. Test coverage added in `test_profile_fp4_layer_sensitivity.py`.
- Hardened KL config: `TrainConfig(loss_mode="segnet_kl")` now requires
  `kl_distill_scope="segnet_aux"` and `promotion_eligible=False`.
  `SEGNET_KL_SMOKE` and `SEGNET_KL_FULL` profiles are explicit forensic-only.
- AGENTS.md updated: use the component sensitivity assembler rather than
  hand-editing promotable manifests; `segnet_kl` is forensic/debug only unless
  revalidated.
- Local Lightning supply-chain scan rerun:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_r2.json`, status OK.
  Current Lightning SSH shell still has no CUDA visible.
- Verification: expanded focused suite `172 passed in 4.78s`; `py_compile`,
  `bash -n`, and `git diff --check` passed.
- Next critical code slice: implement CUDA map/response-curve producer that
  feeds `experiments/build_component_sensitivity_manifest.py`; then exact-eval
  OWV3 byte candidate once CUDA Lightning/Vast/Modal runtime is visible.

## 2026-04-30T19:40Z Swarm Closure, NWCS Export Fix, Corrected Lightning Dry-Run

- Closed the active xhigh swarm:
  - Exact eval ops: OWV3 byte-feasible archive exists locally and on Lightning,
    but no exact CUDA score exists; Lightning Batch Jobs T4 remains fastest safe
    queue.
  - Component sensitivity review: producer/assembler path is structurally
    ready but still requires real CUDA maps/curves.
  - Alpha-Geo-1 design: visual-primitives diagnostics are ready as rejection
    and pretraining gates only.
  - NWCS1 plan/smoke: build-only NWCS1 smoke worked, but promotion was blocked
    by missing `_infer_asymmetric_config` import in Stage 5 export heredocs.
  - Harvest audit: live Vast/Modal artifacts are not newly score-promotable;
    use only canonical archive+JSON+custody harvest.
- Fixed the NWCS blocker in
  `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh` and
  `scripts/remote_lane_j_nwcs_ec_stack.sh`: the Stage 5 export heredoc now
  imports `_infer_asymmetric_config` in the same Python process that writes
  `NWCS1` metadata. Test coverage now checks that specific heredoc.
- Fixed component-sensitivity sample-plan rigor in
  `experiments/profile_component_sensitivity.py`: top-k pair-weighted runs now
  record absolute dataset pair IDs in calibration/holdout records, not
  subset-relative offsets.
- Regenerated OWV3 byte-feasible Lightning Batch Jobs dry-run with
  `--studio pact`: job
  `owv3_byte_feasible_exact_cuda_20260430_codex_studio_pact_dryrun`, command
  SHA `45456318dccbd437e02c4446f7339ad66aaa4e79668c0e69d4707b39c506358f`,
  expected archive SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  bytes `686557`. This is dry-run only; no score claim.
- Strict local Lightning supply-chain scan r4 is OK with zero violations:
  `lightning-sdk==2026.4.10`; no PyPI `lightning` or `pytorch-lightning`.
- Verification: sensitivity/NWCS/Lightning batch suite `60 passed`; security
  preflight/supply-chain/repro/config suite `250 passed`; `py_compile`,
  `bash -n`, and `git diff --check` passed for touched slices.
- Next turn priorities:
  1. Submit/harvest the real corrected Lightning Batch Job only when CUDA is
     visible and strict supply-chain scan passes inside that runtime.
  2. Run CUDA component sensitivity profiling and assemble
     `component_sensitivity_v1` against exact archive/eval custody.
  3. Rerun NWCS build-only smoke after the import fix, then exact CUDA eval only
     with validated sensitivity provenance.
  4. Keep all live harvest/promotions gated on canonical CUDA JSON, exact
     archive bytes, adjudication, and custody.

## 2026-04-30T19:55Z Real Lightning Job, Fail-Closed Sensitivity, MCP Removed

- Integrated the latest xhigh swarm outputs:
  - Lightning exact-eval ops, component sensitivity plan, J-NWC/J-NWCS
    hardening, live harvest triage, Alpha diagnostics, and paper claim hygiene.
- Lightning Batch Jobs exact CUDA eval runner is now hardened:
  - `src/tac/deploy/lightning/batch_jobs.py` runs
    `scripts/scan_lightning_supply_chain.py --quiet --strict` inside the job.
  - It writes `lightning_supply_chain_scan.json` and
    `lightning_runner_preflight.json`.
  - It aborts before eval if torch import, CUDA visibility, or device count
    fail; exact-eval spec validation requires
    `LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK`.
  - Tests cover scan/preflight presence and reject missing preflight.
- `scripts/launch_lightning_batch_job.py` gained:
  - `refresh-status` to update local queue state from SDK Job attributes, not
    logs.
  - `list-machines` to discover provider machine slugs for Lightning Batch
    Jobs.
- Real OWV3 byte-feasible exact-eval job submitted:
  - Local job/spec:
    `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x`.
  - Teamspace/studio/user: `comma-lab`, `lossy-compression-challenge`,
    `adpena`.
  - Machine: `g4dn.2xlarge` after SDK/provider machine discovery showed plain
    `T4` was not accepted by the cluster.
  - Artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x/artifacts`.
  - Expected archive SHA:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
  - Expected archive bytes: `686557`.
  - Command SHA:
    `8333f867324ff9a1fef521d28418b7faad9a3097cf722ce0e581d1d1678ed0e6`.
  - Last refresh at `2026-04-30T19:55:00Z`: status `Running`, no score claim.
- Component sensitivity corrected:
  - Current `experiments/profile_component_sensitivity.py` is diagnostic
    Fisher-proxy evidence only, even on CUDA.
  - It writes `promotion_eligible=false`, diagnostic evidence grade, and
    promotion blockers; `--manifest-output` is blocked.
  - Negative epsilons are accepted for future symmetric/directional
    response-curve probes.
  - AGENTS.md records this fail-closed policy.
- J-NWC/NWCS hardening from Mill:
  - Corpus replay rejects unsafe paths and direct-dict schema bypasses.
  - `NWCS_ALLOW_DEBUG_SENSITIVITY=1` or `NWCS_BUILD_ONLY=1` stops before
    `contest_auth_eval.py` and records non-promotable provenance with
    `score_claim=false`.
- Alpha diagnostics from Anscombe:
  - Alpha-Geo-0 2px diagnostics artifact:
    `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_2px_hashes.json`.
  - Fails exploratory gates:
    `boundary_2px_disagreement=0.14883144511692872`,
    `global_disagreement=0.012303928799099393`, `overall_pass=false`.
  - Diagnostic only; no score claim.
- Paper/report hygiene:
  - `reports/latest.md` quarantines GP v3 and UNIWARD v8 Modal/local rows from
    score/paper claims until lane-local CUDA eval, archive custody, and
    component recomputation exist.
- MCP cleanup:
  - Killed live `rbx-studio-mcp` and `chrome-devtools-mcp` helper processes.
  - Active `/Users/adpena/.codex/config.toml` already had no `[mcp_servers]`.
  - Removed stale MCP backup config and Cloudflare plugin `.mcp.json` cache.
  - Post-cleanup process search matches only the checking `rg`; active config
    search has no MCP server definitions.
- Verification:
  - Focused sensitivity/NWCS/Lightning/Alpha/preflight suite:
    `312 passed in 26.28s`.
  - `py_compile`, `bash -n`, `jq empty`, and `git diff --check` passed.
- Next hard gates:
  1. Poll/harvest the Lightning job; validate `archive.zip`, supply-chain
     scan, runner preflight, `contest_auth_eval.json`, adjudicated JSON,
     adjudication provenance, logs, and custody with expected SHA/bytes.
  2. If exact eval passes, update claim matrix and paper evidence; if it
     fails, treat as engineering/config/math investigation before any family
     conclusion.
  3. Implement official component-response sensitivity producer with
     finite-difference PoseNet/SegNet response validation,
     symmetric/directional curves, calibration/holdout stability, and exact
     CUDA custody.
  4. Rerun NWCS build-only smoke under new guards, then exact CUDA eval only
     with validated sensitivity provenance.

## 2026-04-30T20:00Z Lightning Artifact Path Bug Fixed And R2 Submitted

- First real OWV3 byte-feasible Lightning job
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x` failed
  before eval. SDK logs showed permission denied creating the underscore path:
  `/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x`.
- Root cause: Lightning SDK uses hyphenated job names for job artifact paths,
  while our command defaulted to the local underscore job name.
- This failure is infrastructure-only; no archive eval, score, promotion, or
  kill evidence exists.
- Fixed:
  - Added `lightning_sdk_job_name(name)` to
    `src/tac/deploy/lightning/batch_jobs.py`.
  - Default exact-eval output directory now uses
    `/teamspace/jobs/<hyphenated-sdk-name>/artifacts`.
  - `scripts/launch_lightning_batch_job.py refresh-status` uses the same
    helper.
  - Regression test:
    `test_exact_eval_default_output_dir_matches_sdk_job_artifact_path`.
- Verification:
  - `src/tac/tests/test_lightning_batch_jobs.py`: `19 passed in 0.18s`.
  - `py_compile` passed for Lightning Batch Jobs files.
  - `git diff --check` passed for the repaired code/state slice.
- Rerun submitted:
  - Local job/spec:
    `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2`.
  - Artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts`.
  - Expected SHA/bytes unchanged:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
    `686557`.
  - New command SHA:
    `cf1ab6a1c81b0fa69273007ec4c3efcd54f25bdce1f8f3856bb2bb3bddc21e08`.
  - Last refresh at `2026-04-30T20:04:33Z`: `Running`.
- Full focused suite after repair: `313 passed in 24.19s`.
- Next: poll r2, harvest exact artifacts only after success, and validate with
  expected SHA/bytes plus `--require-adjudication`.

## 2026-04-30T20:15Z Six-Worker Round, R2 Read-Only Failure, R3 Submitted

- Six xhigh workers completed:
  - Goodall: r2 failed infrastructure-only before archive copy/eval; no score
    evidence. Wrote
    `.omx/research/lightning_r2_exact_eval_harvest_20260430_worker.md`.
  - Lovelace: no newly promotable live lane beyond PFP16 A++ and active
    Lightning watch. Omega-W-V2 custody-blocked; Modal rows CPU/invalid.
  - Bacon: Alpha-Geo-1 pre-retraining patch landed. `train_nerv_mask.py`
    supports `--gt-masks-source decoded-baseline`; `remote_lane_nerv.sh`
    exposes `GT_MASKS_SOURCE=decoded-baseline`,
    `DECODED_BASELINE_PATH`, and `DECODED_BASELINE_MEMBER`. Verification:
    `9 passed in 0.82s`.
  - Locke: direct component sensitivity manifest assembly rejects diagnostic
    sources; strict preflight now scans repo-owned MCP server configs.
    Verification: `12 passed`.
  - Avicenna: component sensitivity validator now requires official response
    metadata, passed gates, finite gate specs, official readouts,
    symmetric/directional coverage, and stability thresholds. Verification:
    `48 passed in 1.50s`.
  - Boole: NWCS build-only/provenance hardening landed; CPU-only smoke archive
    SHA `9339fed08deffb25b73803b2e311ec34a93508256e5aff993758d23ec0e9c6fd`,
    `3895` bytes, `auth_eval_skipped=true`,
    `promotion_eligible=false`, `score_claim=false`. Verification:
    `36 passed in 1.48s`.
- R2 Lightning failure:
  - Status `Failed`.
  - Logs show `OSError: [Errno 30] Read-only file system` writing
    `lightning_queue_metadata.json` under
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts`.
  - This is infrastructure-only; no archive eval, score, promotion, or kill
    evidence.
- Lightning queue repair:
  - `src/tac/deploy/lightning/batch_jobs.py` now distinguishes
    `lightning_sdk_artifact_path(name)` from writable
    `default_exact_eval_output_dir(repo_dir, job_name)`.
  - Exact-eval spec records `remote_output_dir` and `sdk_artifact_path`.
  - Validation rejects `/teamspace/jobs/...` as command output because it is a
    read-only SDK artifact view inside Studio jobs.
  - Focused component/Lightning/MCP suite: `72 passed in 1.64s`;
    `py_compile` and `git diff --check` passed for touched slices.
- R3 submitted:
  - Local job/spec:
    `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3`.
  - Writable remote output dir:
    `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
  - SDK artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3/artifacts`.
  - Expected SHA/bytes:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
    `686557`.
  - Command SHA:
    `23bba87e21b56791278cd7b6c8686a8718004f67fbabf72fd07d2a90bc813467`.
  - Last refresh `2026-04-30T20:18:59Z`: `Running`, no artifacts yet.
- Current claims:
  - PFP16 A++ remains the only paper-ready score claim.
  - OWV3 byte-feasible remains active exact-eval watch only; no score claim
    until CUDA JSON/adjudication/preflight/supply-chain/archive/custody
    artifacts validate.
- MCP helpers respawned from an external parent after config cleanup. Killed
  exact `rbx-studio-mcp` and `chrome-devtools-mcp` helpers again. Active Codex
  and plugin config search still has no MCP server definitions; post-kill
  process sweep matched only the checking `rg`.

## 2026-04-30T21:05Z Preflight Metabug Hardening + Lightning r4

- R3 OWV3 byte-feasible Lightning exact eval is infrastructure-only:
  CUDA/archive/inflate setup progressed, then upstream `evaluate.py` failed
  because `nvidia.dali` was missing. No `contest_auth_eval.json`, no score
  claim, no promotion, no regression/kill evidence.
- Exact-eval runner hardening landed:
  - mandatory expected archive SHA-256 + byte count,
  - mandatory adjudication provenance (`exact-eval` CLI rejects missing
    `--adjudicate`),
  - stale output artifacts removed before each run,
  - distinct pre/post supply-chain scan artifacts,
  - hash-pinned direct URL DALI requirements artifact,
  - content validation for DALI bootstrap, runner preflight, supply-chain scans,
    adjudicated JSON copy, archive SHA/bytes, and CUDA provenance.
- PyPI Lightning compromise guard now rejects any installed bare `lightning`
  distribution, not just known compromised `2.6.2/2.6.3`. Local strict scan
  clean: `.omx/state/lightning_supply_chain_scan_20260430_codex_preflight_metabugs.json`,
  `status=OK`, `violation_count=0`, `lightning=null`,
  `lightning-sdk=2026.4.10`.
- Remote lane contest-CUDA hardening landed:
  - mechanical replacement of promotable remote `--device
    "${AUTH_EVAL_DEVICE:-cuda}"` calls with literal `--device cuda`,
  - preflight catches unguarded `AUTH_EVAL_DEVICE` under `[contest-CUDA]`,
  - preflight requires kept `eval_work` custody (`--keep-work-dir` and
    `--work-dir`).
- Verification:
  - focused guardrail suite `270 passed in 24.80s`,
  - broader focused suite `374 passed in 27.14s`,
  - `py_compile` passed,
  - `git diff --check` passed.
- R4 submitted:
  - job/spec `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4`,
  - SDK job `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r4`,
  - expected archive SHA `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  - expected bytes `686557`,
  - command SHA `79bf98e80faa762456f5f0d35845a0326bee79d8285e89b871ded8a7d837ca60`,
  - last refresh `2026-04-30T21:01:31Z`: `Pending`.
- Claim discipline remains unchanged: PFP16 A++ is still the only
  claim-capable score anchor. OWV3 r4 is watch-only until exact CUDA JSON,
  adjudication, archive, DALI bootstrap, runner preflight, pre/post
  supply-chain scans, and custody artifacts validate.
- MCP helpers were killed again after process check found
  `chrome-devtools-mcp` and `rbx-studio-mcp`. Post-kill sweep matched only the
  checking `rg`.

## 2026-04-30T21:26Z Telemetry Cleanup and R4 Result Packet

- Vast.ai live inventory is empty after cleanup:
  `.omx/state/vastai_show_instances_live_final_20260430.json` = `[]`.
- Destroyed current-run/duplicate Vast instances after harvest/snapshot:
  `35885106`, `35906669`, `35907873`, `35899850`, `35925274`, `35925374`,
  `35925475`, `35925801`, `35925825`, `35925916`.
- Modal has no live tasks. MCP helpers were killed again; process sweep after
  cleanup matched only the checking `rg`.
- Scientific classification remains strict: no lane-family kill from these
  Vast results. HM-S/SA are SegMap pack/roundtrip engineering failures; H-V3
  is a tensor-channel engineering failure; Lane 19 is a cost/proxy abort only;
  Lane 20 on the Lane G v3 anchor is static-fallback/no-op until a non-static
  byte win exists.
- Lightning r4 exact CUDA/T4 packet was harvested to
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4/`
  with `SHA256SUMS`.
- R4 result:
  `score_recomputed_from_components=1.0378905176070103`,
  `final_score=1.04`, PoseNet `0.00319052`, SegNet `0.00402120`, bytes
  `686557`, SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  device `cuda`, GPU `Tesla T4`.
- R4 adjudication failed the SegNet relative gate:
  `0.00402120 / 0.00400656 = 1.003654` against cap `1.002`.
  This is high-value exact diagnostic evidence, not promotion-grade evidence.
- Launcher duplicate-spend class fixed:
  `scripts/launch_lane_with_retry.py` now uses `logical_lane_key()` for
  advisory locks and live Vast duplicate detection, collapsing timestamped
  labels such as `_q1_20260430T...`, `_q1c_20260430T...`, and non-numeric
  queue tags.
- Added `.omx/state/dispatch_holds.json` plus launcher `FATAL_DISPATCH_HOLD`
  enforcement. Lane 19 and Lane 20 are held fail-closed until Grand Council
  clearance is recorded.
- Verification: `py_compile` passed for launcher/preflight/test files;
  focused hardening suite passed `22 passed in 1.46s`.

Next mandatory order:

1. Grand Council review OWV3 r4 component gate and mitigation before any claim.
2. Patch H-V3 channel bug before rerun.
3. Repair SegMap pack/roundtrip contract before HM-S/SA reruns.
4. Do not rerun Lane 19 until deterministic archive/adjudication/frontier gates
   are fixed.
5. Keep Vast empty unless a lane has a clear logical key, preflighted exact
   evidence path, and no duplicate live state.

## 2026-04-30T21:49Z Swarm Integration Memory

- New adjacent delta ledger:
  `/Users/adpena/Projects/pact/.omx/research/shannon_floor_swarm_execution_delta_20260430_codex.md`.
- OWV3 r4 Grand Council verdict:
  - exact CUDA/T4 packet is real diagnostic evidence,
  - not promotable under the predeclared SegNet gate,
  - no retroactive gate relaxation,
  - next admissible paths are paired same-run PFP16 calibration and
    SegNet-conservative OWV3 R5 candidates.
- H-V3 channel bug repaired:
  `segnet_uncertainty_weighted_loss` preserves RGB BCHW before SegNet; focused
  regression tests pass.
- HM-S/SA SegMap pack contract repaired:
  explicit lossy `segmap_block_fp_per_channel_lossy_v1`, `1e-3` MSE gate,
  `segmap_pack_roundtrip.json`, and exact CUDA archive-eval gate.
- Lane 19/20 holds hardened:
  launcher checks lane-specific clearance requirements even if the hold file
  is missing or marked `cleared: true`.
- Lightning security:
  strict local scan clean at
  `.omx/state/lightning_supply_chain_scan_20260430_codex_current.json`;
  `scripts/launch_lightning_batch_job.py` sets
  `LIGHTNING_DISABLE_VERSION_CHECK=1` before SDK import.
- KL/distill:
  `train_renderer.py` now blocks positive `kl_distill_weight` without
  `kl_distill_scope="segnet_aux"` and blocks primary/full-scorer KL outright;
  current positive-KL profiles declare scope; strict preflight added.
- MCP:
  helper processes killed; discovered server maps in `.claude`, `.cursor`,
  `.gemini/antigravity`, `.lmstudio`, and `/Users/adpena/Projects/molt/.mcp.json`
  are empty. Plugin caches are inert and were not deleted.
- Verification bundle:
  `107 passed in 2.20s`; shell syntax passed for touched shell scripts;
  targeted `git diff --check` passed; strict KL/SegMap/MCP preflights passed.

Carry forward:

- PFP16 A++ remains the only promotion-grade score anchor.
- R4 OWV3 is exact diagnostic evidence only.
- Bad results stay scoped to measured implementation/config until exact CUDA
  custody and adversarial review prove a broader claim.
- Remote/eval runners must be supply-chain scanned and CUDA-auth exact before
  score claims.

## 2026-04-30T22:05Z Six-Item Follow-Up Memory

- Swarm agents closed:
  - PFP16 verified already landed A++: exact T4 CUDA, archive SHA
    `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
    `686635` bytes, recomputed score `1.043987524793892`; local rebuild
    reproduced SHA/bytes.
  - Sensitivity/OWV3 R5 readiness landed: `perturbation_basis_v1.json`,
    response prediction calibration diagnostics, and R5 neighbor ranking.
    Still non-promotable until CUDA finite-difference component response and
    `component_sensitivity_v1` custody exist.
  - Lane 12/Alpha dispatch is build-only by default with decoded-baseline
    targets; exact eval requires pose-regeneration provenance.
  - J-NWC has build-only non-promotable provenance; J-NWC/NWCS exact paths now
    run adjudication with PFP16 A++ score/component gates.
  - Auditor finding integrated: claim matrix updated for OWV3 r4 exact
    diagnostic/non-promotable state; source doc no longer claims sufficiency.
- New launcher guard:
  no unrelated retraining dispatch before
  `.omx/state/lane12_nerv_l2_clearance.json` records Lane 12 L2 unblock,
  geometry gate pass, and 3 clean Grand Council passes.
- New MCP hardening:
  `check_no_live_mcp_processes(strict=True)` blocks already-running MCP helper
  processes, not just configs.
- Telemetry:
  Vast `[]`; Modal all `Tasks=0`; use `uv run --no-sync` or `.venv/bin`
  commands because direct shell PATH misses provider CLIs.
- Verification:
  integrated focused suite `69 passed in 2.66s`; worker suites passed PFP16
  `34`, Sensitivity/OWV3 `83`, Lane 12 `11`, J-NWC/NWCS `29`.

Carry forward next:

1. OWV3 R5 exact CUDA/T4 eval only from SegNet-conservative candidates with
   predeclared PFP16 component gates.
2. Component-sensitivity promotion only after official finite-difference
   response curves and manifest custody.
3. Lane 12 exact eval only after Alpha-Geo diagnostics plus pose-regeneration
   provenance.
4. J-NWC/NWCS exact eval only with validated sensitivity/corpus artifacts and
   adjudication provenance.

## 2026-04-30T22:24Z Orchestration Closeout Memory

- Six xhigh workers were closed and their deltas integrated. Treat this as
  readiness/hardening work, not as a new promoted score.
- Adjacent source-of-truth progress docs:
  - `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
    records claim-grade Grand Council deltas against the main design doc.
  - `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
    records deploy/readiness status against execution readiness.
  - `.omx/research/shannon_floor_swarm_execution_delta_20260430_codex.md`
    is the operational swarm ledger tying worker results to admissible next
    actions.
- OWV3 R5 rank-1 queue candidate:
  `owv3_0047_bbr0p67_protect0p00135_aggr1em05`, `686468` bytes, SHA
  `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`.
  It has no score claim until paired PFP16 calibration and exact CUDA/T4
  adjudication pass.
- OWV3 r4 remains exact CUDA/T4 diagnostic evidence only; adjudication artifacts
  now write even when component gates fail, and Lightning artifact validation
  reports `promotion_eligible=false`.
- Lightning: SDK machine discovery and dry-run exact-eval plans work; SSH
  staging still fails public-key auth. Do not submit a spendful exact eval
  unless remote archive custody is proven or archive construction happens
  inside the remote job.
- Lane 12 jsonfix40 failed Alpha-Geo-0 gates versus Lane G v3/base masks; keep
  Lane 12 build-only behind L2 clearance and pose-regeneration provenance.
- J-NWC/NWCS exact paths now include stronger adjudication custody, but remain
  blocked until validated real sensitivity/corpus artifacts exist.
- Provider/MCP state: Vast `[]`, Modal `Tasks=0`, strict MCP live/config
  preflights passed after killing respawned helpers.
- Verification closeout: `137 passed in 3.27s`, plus Python compile, shell
  syntax, targeted whitespace, and strict remote-auth/launcher/MCP preflights.
- Commit hygiene: `scripts/lightning_repro_workspace.py` existed with a staged
  deletion plus untracked same-path file; it was normalized with
  `git restore --staged` after confirming the file is tracked in `HEAD`.

## 2026-04-30T22:30Z Lightning Queue Memory

- Lightning SSH is now working after rerunning the setup script.
- Permanent DX hardening landed in `scripts/lightning_repro_workspace.py`:
  remote `uv sync` uses `UV_LINK_MODE=${UV_LINK_MODE:-copy}` to avoid hardlink
  failures on Lightning Studio filesystems. Test coverage added in
  `test_lightning_repro_workspace.py`.
- Reproducible staging succeeded:
  `1093` files, `18490969` bytes, manifest SHA
  `3cd7611e6cce9a18e00ef9505f367fa44ee31622841d8a7378a2d360690919f1`.
- Active Lightning T4 jobs:
  - `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`, Pending,
    archive SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
    bytes `686635`.
  - `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`, Pending,
    archive SHA `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`,
    bytes `686468`.
- R5 was submitted in parallel for wall-clock speed but is explicitly marked
  `requires_paired_readjudication=true`; do not promote until calibration and
  R5 artifacts are harvested and paired adjudication passes.

## 2026-04-30T22:48Z Lightning Exact-Eval Isolation Memory

- First Lightning PFP16/R5 jobs exposed a harness bug: `inflate.sh` uses
  `uv run`, which recreated the shared Studio `.venv`; `upstream/evaluate.py`
  then failed on missing `tqdm`. Treat those attempts as harness failures only.
- Permanent fix landed in `src/tac/deploy/lightning/batch_jobs.py`:
  exact-eval commands export per-job `UV_PROJECT_ENVIRONMENT=<out>/uv_project_env`,
  export `UV_LINK_MODE=${UV_LINK_MODE:-copy}`, and lock shared `.venv`
  DALI/bootstrap setup with `.omx/state/lightning_exact_eval_venv.lock`.
- Contaminated jobs:
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r2` failed;
  `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4` was stopped.
- Clean reruns submitted:
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv` and
  `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv`.
  Both remain queue/eval work until harvested and adjudicated.

## 2026-04-30T22:53Z Queue/Verification Memory

- Latest status refresh: both clean isolated Lightning exact-eval reruns are
  still `Pending` with cost `0.0`; no completion/failure timestamp and no
  harvestable score evidence yet.
- Closeout verification: integrated focused suite `177 passed in 5.04s`, plus
  Python compile, shell syntax, targeted whitespace, provider telemetry, and
  strict MCP preflights.
- Next turn must poll Lightning first, harvest only canonical artifacts after
  terminal status, validate archive SHA/bytes/provenance, then run paired
  PFP16-vs-R5 readjudication before any Grand Council promotion review.

## 2026-04-30T22:55Z Running Eval Memory

- Latest Lightning refresh: both clean isolated exact-eval jobs are `Running`
  with cost `0.0` and no terminal timestamp.
- Provider/MCP closeout: Vast `[]`, Modal listed `Tasks=0`, strict MCP
  process/config preflights clean.
- Do not promote or kill from running status. Next action is polling until
  terminal status, then canonical harvest and paired adjudication.

## 2026-04-30T23:10Z Exact Harvest Memory

- Clean isolated PFP16/R5 Lightning jobs reached terminal `Failed` because
  adjudication failed the predeclared SegNet component gate after exact CUDA/T4
  eval succeeded. This is forensic evidence, not harness failure and not
  promotion.
- Added reproducible `harvest-ssh` support to `scripts/launch_lightning_batch_job.py`
  and `src/tac/deploy/lightning/batch_jobs.py`; it derives the SDK artifact
  mirror path and copies only canonical top-level evidence files.
- Harvested and validated local mirrors for both jobs:
  - PFP16 calibration score `1.037045485927815`, archive
    `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
    `686635` bytes, `promotion_eligible=false`.
  - OWV3 R5 score `1.0373951773937642`, archive
    `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`,
    `686468` bytes, `promotion_eligible=false`.
- Paired delta: R5 is worse than PFP16 by `0.00034969146594909795` despite
  `167` fewer bytes. Next R5 branch must be SegNet-conservative or use the
  new promotion finite-difference sensitivity path.
- Closeout verification: `154 passed in 4.10s`; local Lightning supply-chain
  scan `OK`; Vast `[]`; Modal `Tasks=0`; strict MCP preflights clean after
  killing respawned helper processes.

## 2026-04-30T23:30Z R6 / Guardrail Memory

- R6 exact-eval is queued, not evidence:
  `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`, SDK name
  `owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1`, status `Pending`
  at `2026-04-30T23:29:12Z`.
- R6 archive is
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip`,
  SHA `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`,
  `686531` bytes, `-104` bytes versus paired PFP16, low-bit channels `58`
  versus failed R5's `62`.
- Remote Lightning custody was checked by SSH before submit; the updated
  adjudicator was copied to the Studio and exposes
  `--allow-component-gate-forensic-success`.
- Failed submit attempts were client-side only: missing `--user adpena`, then
  unsupported `T4` alias. Successful submit used `--machine g4dn.2xlarge`.
- Code landed:
  R6 selector in `experiments/sweep_owv3_byte_plan.py`; Lightning forensic
  component-gate success mode; sensitivity zero-signal rejection; manifest
  rejection for NaN/Inf maps, archive custody mismatch, and sample-plan
  split-hash mismatch.
- Verification: `136 passed in 3.40s` plus `py_compile`.
- Next turn must poll/harvest R6 first, then update C-041 from active queue to
  exact forensic/promotable status depending on paired component gates.

## 2026-04-30T23:44Z Swarm Hardening Memory

- R6 exact eval remains active, not evidence: `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1` is `Running` as of `2026-04-30T23:42:42Z`, cost `0.0882`, artifact root `/teamspace/jobs/owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1/artifacts`. Poll with state-aware `refresh-status`; harvest only after terminal status.
- Lightning status-refresh DX fixed: `scripts/launch_lightning_batch_job.py refresh-status` infers SDK job name, teamspace, org, and user from `.omx/state/lightning_batch_jobs.json`.
- Lightning/PyPI compromise hardening extended: strict supply-chain scan covers `tools/`, blocks `.venv/bin/lightning`, bare `lightning <subcommand>`, and `$LIGHTNING`; `tools/lightning_run.sh` and `tools/lightning_monitor.sh` now use SSH/SCP. Latest local scan `.omx/state/lightning_supply_chain_scan_20260430_codex_tools_hardened.json` is OK with no PyPI `lightning` or `pytorch-lightning`, `lightning-sdk==2026.4.10`.
- Component sensitivity direct finite-difference profiler is explicitly non-promotable: no manifest output, `promotion_eligible=false`, `official_component_response=false`, exact 1200-frame diagnostic guard, and `not_canonical_inflate_eval_path` blocker until archive -> inflate.sh -> upstream/evaluate.py custody exists.
- KL hardening landed: high-weight KL in `train_renderer` requires explicit forensic opt-in, legacy 1.0-weight profiles are forensic/non-promotable, FilmCanvas KL is scoped to SegNet aux, corrected Lane D-V3 remains promotion-capable only pending exact gates.
- NWCS builder landed: `experiments/build_nwcs_sensitivity_inputs.py` emits anchor/corpus sensitivity inputs only from promotable `component_sensitivity_v1` and rejects fake/proxy/uniform/stale/incomplete maps. It does not create sensitivity evidence.
- Lane 12 remains no-go: missing `.omx/state/lane12_nerv_l2_clearance.json`; Worker E recorded exact CUDA negative jsonfix40 score `26.03719330455429`, PoseNet `49.7784996`, SegNet `0.03528685`, Alpha-Geo overall false. Do not dispatch Lane 12 until real L2 clearance and three clean reviews.
- MCP helper processes respawned and were killed again. If they reappear, host-level MCP supervisor/config outside repo still needs removal.
- Verification: `372 passed` across NWCS/KL/preflight/Lightning/component-sensitivity/manifest focused suite; `358 passed` prior integrated slice; Python compile, shell syntax, strict supply-chain scan, and scoped diff checks passed.

## 2026-04-30T23:48Z R6 Exact Harvest Memory

- R6 exact eval completed and was harvested/validated locally: `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`, SDK `owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1`, completed refresh `2026-04-30T23:47:45Z`, validation `2026-04-30T23:48:37Z`.
- Exact CUDA/T4 result: score `1.0393166493980681`, final score `1.04`, archive bytes `686531`, archive SHA `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`, PoseNet `0.00323147`, SegNet `0.00402421`, n=600, GPU `Tesla T4`.
- Paired PFP16 baseline: score `1.037045485927815`, bytes `686635`, PoseNet `0.00316404`, SegNet `0.00401966`.
- R6 saved `104` bytes but regressed by `+0.0022711634702530237` score and failed PoseNet component gate: relative `1.0213113614240024` > `1.002`; SegNet passed with relative `1.0011319365319455` <= `1.002`.
- Strict final-deploy adjudication returned exit code `2` with `REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED`.
- Classification: A++ exact CUDA/T4 forensic negative for this R6 implementation/config, not promotable and not OWV3 family KILL. Next OWV work must address PoseNet drift while preserving R6's SegNet behavior, or wait for canonical component sensitivity evidence before another exact-eval spend.

## 2026-04-30T23:58Z Next-Wave Telemetry / Research / DX Memory

- MCP helpers were killed again and verified absent.
- Live provider posture from local credentials: Vast `.venv/bin/vastai show instances --raw` returned `[]`; Modal `.venv/bin/modal app list` showed zero tasks; Lightning bulk SDK refresh found no new harvestable result beyond already harvested R6.
- Landed Lightning telemetry DX: `scripts/launch_lightning_batch_job.py refresh-status --all` refreshes every latest non-dry-run local record through `lightning_sdk.Job` attributes only, skips dry-runs by default, and supports `--fail-on-error`. Do not use the Lightning console script for telemetry.
- Verification: `src/tac/tests/test_lightning_batch_jobs.py` passed `33 passed`; Python compile passed; supply-chain scan `.omx/state/lightning_supply_chain_scan_20260430_codex_nextwave.json` is OK with zero violations.
- Provider audit note: `.omx/research/provider_telemetry_canonical_harvest_audit_20260430_worker_c.md` independently confirms no live Vast/Modal jobs and no Lightning running records in local state. Historic Vast tracker rows are stale and not live spend truth. Keep Lane 19/20 holds in place.
- Research consensus: arXiv:2604.26919v1, PufferLib/RL/LM Studio/visual primitives, and Training-Free GRPO are useful for proposal hygiene, dual-readout sensitivity audits, BOHB/bandit scheduling, and experience memory only. They are not compression evidence, must not add runtime dependencies, and cannot promote/kill lanes. Exact CUDA archive eval with SHA/bytes/components/adjudication remains the only score authority.
- New adjacent ledger: `.omx/research/shannon_floor_nextwave_telemetry_and_research_20260430_codex.md`.

## 2026-04-30T23:59Z OWV3 R7 Guardrail Memory

- Worker B landed a fail-closed OWV3 R7 scalar-threshold selector in `experiments/sweep_owv3_byte_plan.py` with tests in `src/tac/tests/test_sweep_owv3_byte_plan.py`.
- R7 after R6 now requires byte feasibility versus paired PFP16, `fallback_action=keep_asym`, no diagnostic FP16 layers, promotion-eligible metadata, exclusion of exact failed R5/R6 candidate IDs, OWV2-low-bit channels `<=58`, and `bit_budget_ratio>=0.65`.
- Applying the R7 selector to the current R5 sweep rows with R6 candidate `owv3_0076_bbr0p65_protect0p0013_aggr1em05` returns `candidate_count=0`.
- Interpretation: do not spend another exact CUDA eval on blind OWV3 scalar-threshold candidates from this grid. The next admissible OWV3 mitigation is component-balanced PoseNet/SegNet sensitivity, or a materially new action rule with pre-registered gates.
- Verification: `test_sweep_owv3_byte_plan.py` passed `13 passed`; Python compile and scoped whitespace checks passed.

## 2026-05-01T00:02Z Official Component-Response Producer Memory

- Landed canonical official response-curve producer: `experiments/profile_component_sensitivity_official.py` with tests in `src/tac/tests/test_profile_component_sensitivity_official.py`.
- Purpose: consume baseline and perturbation archives, evaluate via `experiments/contest_auth_eval.py` or validate existing exact `contest_auth_eval.json` custody, then emit PoseNet/SegNet/combined official response curves for `experiments/build_component_sensitivity_manifest.py`.
- Boundary: it does not generate perturbation archives, component maps, or stability JSON. Those remain required inputs before assembling promotable `component_sensitivity_v1`.
- Codex added an extra fail-closed guard after review: the baseline archive is rejected if it appears at any nonzero epsilon, avoiding ambiguous response baselines.
- Verification: official-response + manifest/schema suite passed `48 passed`; compile and scoped whitespace checks passed.
- Next canonical sequence: generate deterministic perturbation archives -> run `profile_component_sensitivity_official.py --device cuda --require-passed` on CUDA -> assemble `component_sensitivity_v1` only after official response curves, maps, stability, sample plan, archive SHA/bytes, and exact eval custody all pass.

## 2026-05-01T00:07Z Next Wave 2 Kickoff Memory

- Spawned xhigh wave for: perturbation archive/official response-plan producer, Lightning official-response queue readiness, Alpha visual-primitive geometry diagnostics, NWCS fail-closed readiness, claim-ledger adversarial audit, and DX/preflight self-protection.
- Control-plane telemetry before implementation: MCP helpers killed/absent; Vast live instances `[]`; Modal tasks `0`; Lightning SDK bulk refresh `.omx/state/lightning_batch_jobs_refresh_20260501_codex_nextwave2.json` has `refreshed_count=9`, `skipped_count=13`, `failure_count=0`; supply-chain scan `.omx/state/lightning_supply_chain_scan_20260501_codex_nextwave2.json` is OK with zero violations and no PyPI `lightning`/`pytorch-lightning`.
- No provider kill or harvest action is available from current local credentials. Next deployable path remains official perturbation archives -> CUDA official response curves -> promotable `component_sensitivity_v1` -> OWV3/NWCS/Alpha decisions.

## 2026-05-01T00:22Z Next Wave 2 Integration Memory

- Landed deterministic perturbation archive/plan producer:
  `experiments/build_component_response_perturbation_plan.py` with tests in
  `src/tac/tests/test_build_component_response_perturbation_plan.py`. It emits
  `official_component_response_plan_v1` and bounded archive variants, rejects
  unsafe ZIP members and renderer-magic mutation by default, and records
  `auth_eval_required=cuda`. No score claim.
- Landed Lightning official component-response queue readiness:
  `src/tac/deploy/lightning/batch_jobs.py`,
  `scripts/launch_lightning_batch_job.py`,
  `src/tac/tests/test_lightning_batch_jobs.py`, and
  `docs/runbooks/lightning_official_component_response.md`. Non-dry-run
  submit requires staged manifest plus local plan validation; jobs are
  CUDA-only, supply-chain/DALI guarded, and compact-harvest validated.
- Parent integration added official-response subprocess regression coverage:
  one `--inflate-timeout`, one `--evaluate-timeout`.
- Backfilled R6 `adjudication_provenance.json` as non-promotable forensic:
  `promotion_eligible=false`,
  `paper_claim_grade="A-negative scoped forensic"`,
  `allowed_use=["forensic","no_rank_frontier","no_promotion"]`.
- Integrated NWCS fail-closed hardening, Alpha residual-region ranking,
  Modal CPU advisory guard, and claim-ledger fixes from the first xhigh swarm.
- Final telemetry checkpoint:
  `.omx/state/lightning_batch_jobs_refresh_20260501_codex_nextwave2_final.json`
  reports `refreshed_count=9`, `skipped_count=13`, `failure_count=0`,
  statuses `Completed=1`, `Failed=7`, `Stopped=1`, and no running jobs. Vast
  instances `[]`; Modal tasks `0`; no persistent MCP process found.
- Verification in parent:
  official response / perturbation plan / Lightning suite `52 passed`; Alpha
  diagnostics `11 passed`; remote auth + NWCS slice `87 passed`; Modal CPU
  guard `33 passed`; compile/shell syntax checks passed.
- Closed first six workers and spawned xhigh ongoing research/design swarm:
  arXiv:2604.26919v1, PufferLib/RL + visual primitives, Tencent training-free
  GRPO, and KL-distill architecture hardening.
- Next production path remains:
  reviewed perturbation basis -> deterministic response archives -> Lightning
  `component-response --require-passed` -> validated official curves ->
  promotable `component_sensitivity_v1` -> component-balanced OWV3/NWCS/Alpha
  decisions. Do not exact-eval another blind OWV3 scalar-threshold candidate.

## 2026-05-01T00:34Z Research Intake / MCP Hardening Memory

- External research agents completed:
  - arXiv:2604.26919v1 is methodology only: sparse top-k, warm-ramp updates,
    and dual readout. It is not a compression lane.
  - CI-ICM/channel-importance, S2-CoT entropy co-tuning, TinyNeRV, feedback
    rate control, HAWQ-V3/SCN/Cool-Chic/constriction are design inputs only.
  - DeepSeek visual primitives should become an Alpha diagnostic packet over
    decoded baseline masks: components, boundaries, lanes/road/vehicles,
    temporal tracks, and primitive failures.
  - PufferLib/RL is deferred until a cheap surrogate reward is correlated with
    exact CUDA anchors; start with deterministic bandit/BO/CMA-ES/Optuna/
    Nevergrad/BoTorch loops.
  - LM Studio/local models are allowed only for read-only JSON-schema triage
    and lane cards behind deterministic validators.
  - Tencent Training-Free GRPO is useful as a hashed, read-only
    experience-library protocol for grouped proposals; it cannot dispatch or
    serve as score evidence.
- No-claim rule reaffirmed: none of these external/proxy methods can promote,
  rank, kill, compose deltas, or anchor paper claims. Exact CUDA archive eval
  with SHA/bytes/components/adjudication remains authority.
- MCP cleanup:
  - killed live `rbx-studio-mcp`, `chrome-devtools-mcp`, and child helpers
    spawned by the outer Codex supervisor;
  - disabled `game-studio@openai-curated` and `cloudflare@openai-curated` in
    `/Users/adpena/.codex/config.toml`;
  - removed `.playwright-mcp` and transient `.codex/.tmp` MCP JSON files;
  - verified `/Users/adpena/.claude/mcp.json` has empty `mcpServers` and no
    MCP helper process remains.
- Active remaining subagent: KL-distill architecture/hardening audit.

## 2026-05-01T00:41Z KL Audit / Micro-Hardening Memory

- KL audit completed. Current code mostly preserves primary-KL prohibition,
  but KL-family promotion still needs one typed policy/provenance surface:
  family, scope, weight, temperature, class weights, student/teacher roundtrip
  contract, promotion eligibility, forensic reason, banned-primary opt-in, and
  optional controller telemetry.
- Immediate code hardening landed:
  - `src/tac/losses.py`: `kl_distill_scorer_loss` and
    `kl_distill_segnet_only` now reject non-finite or non-positive
    temperatures before dividing logits.
  - `src/tac/losses_jbl.py`: removed over-strong claim that JBL cannot induce
    PoseNet collapse; JBL is documented as distillation-family and gated by
    exact CUDA component evidence.
  - Tests added in `src/tac/tests/test_losses.py` and
    `src/tac/tests/test_training.py`.
- Verification: KL/training focused suite `70 passed`; py_compile passed.
- Next KL implementation unit:
  add `src/tac/kl_config.py`, normalize legacy CLI/profile flags, serialize
  KL policy into train/archive/remote/adjudication provenance, add preflight
  checks for missing/unsafe KL policy and roundtrip contracts, and make
  adjudication block KL/JBL/distill-active promotion without exact CUDA
  non-collapse evidence.

## 2026-05-01T01:02Z Official Response / KL Policy / Alpha Packet Memory

- Current A++ anchor remains the PFP16 archive at
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/`;
  exact CUDA/T4 eval is the only promotion-grade source of truth for GPU
  scorer claims. CPU/MPS diagnostics remain non-promotable.
- Landed ASYM-safe renderer perturbation basis selection and deterministic
  official component-response plan artifacts under
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r1/`.
  These artifacts are queue-ready but score-neutral until Lightning/CUDA
  exact component-response eval runs and returns JSON custody.
- Lightning SSH setup script was fetched and saved at
  `.omx/tmp/lightning_ssh_setup_20260501.sh`; public key hash recorded as
  `61cee7ad4530683618cef000b17015dfc3d73f2503dc014dea5fabd51ba62568`.
  SSH still fails with `Permission denied (publickey)`, so no non-dry-run
  Lightning staging or Batch Job submission occurred in this pass.
- Landed frozen KL/distillation policy schema in `src/tac/kl_config.py` plus
  focused tests. Policy vocabulary covers inactive, SegNet-aux KL,
  primary-scorer KL, legacy SegNet-KL, and JBL; primary KL is forensic-only,
  legacy SegNet-KL/JBL require non-promotable forensic representation, and
  promotion-capable SegNet-aux KL requires positive weight, temperature >= 2,
  and eval-roundtripped student/teacher scorer-input contract.
- Next KL hardening unit is runtime integration: include
  `distillation_policy_v1` in training/checkpoint/remote/adjudication
  provenance, make preflight validate every profile with positive
  `kl_distill_weight` or KL/JBL loss family through the typed policy, and
  make adjudication block KL/JBL/distill-active promotion without exact CUDA
  non-collapse evidence.
- Alpha visual-primitives packet was added to
  `experiments/diagnose_nerv_geometry.py` as diagnostic-only,
  `promotion_eligible=false`, `score_claim_eligible=false`. Full Lane 12
  visual extraction still needs predecoded mask cache or longer CPU budget;
  existing scalar Alpha JSON remains the current diagnostic evidence.

## 2026-05-01T01:36Z KL Runtime Policy Hardening Memory

- Follow-up xhigh KL audit found that the first policy schema was not yet
  fail-closed at trainer construction and was too sidecar-heavy for artifact
  custody.
- Landed runtime integration:
  `TrainConfig` now validates `distillation_policy_v1` during construction,
  exposes `forensic_reason`, and computes canonical policy SHA-256;
  `Trainer` and `SegMapTrainer` normalize/store the policy before optimizer
  work; active SegNet-aux KL with `kl_distill_temperature < 2.0` fails before
  training; forensic primary KL and legacy SegNet-KL require explicit reasons.
- Embedded `distillation_policy` and `distillation_policy_sha256` into generic
  training state, int8 checkpoint metadata, renderer training state, renderer
  FP4/FP32 `__meta__`, best-renderer meta JSON, and renderer telemetry. Training
  proxy artifacts are stamped non-score/non-promotable pending exact CUDA auth
  eval.
- Strict `check_distillation_policy_schema_clean` is wired into preflight and
  live profiles currently normalize cleanly.
- Verification: KL/config/training/loss suite `99 passed`; train-renderer and
  preflight-adjacent suite `270 passed`; py_compile and scoped diff-check
  passed; MCP helper processes were killed again after respawn and the final
  probe was clean.
- Remaining KL hardening: validate harvested remote/adjudication provenance for
  policy format/schema/hash, exact CUDA device, exact archive SHA/bytes, and
  component non-collapse gates; scan remote scripts with KL/JBL/distill flags
  for policy provenance; wire `optimize_poses.py` into the same policy/hash
  contract.

## 2026-05-01T01:40Z Continuation Swarm Memory

- Closed xhigh workers for KL/distillation promotion hardening, J-NWC/NWCS
  custody, six-item ops telemetry, Lightning security, arXiv/Tencent research,
  PufferLib/visual primitives, and Alpha runtime unblock.
- Landed fail-closed distillation promotion adjudication:
  `scripts/adjudicate_contest_auth_eval.py` now blocks
  KL/JBL/distillation-active promotion unless exact CUDA, archive SHA/bytes,
  `distillation_policy_v1`, matching policy SHA, and PoseNet/SegNet component
  gates are present. Remote preflight now scans for missing distillation
  promotion provenance.
- Landed NWCS manifest-custody rechecks in
  `experiments/build_nwcs_sensitivity_inputs.py` before sensitivity
  projection.
- Hardened Lightning official component-response dry-run closure:
  source manifest and local perturbation plan are paired requirements, and
  plan-listed archives are checked against the staged manifest even in dry-run.
- Lightning security status: active venv has `lightning_sdk==2026.4.10`, no
  PyPI `lightning` or `pytorch-lightning`; strict scan
  `.omx/state/lightning_supply_chain_scan_20260501_codex_ioc_expanded.json`
  is OK. The preflight IOC set now includes additional reported Mini
  Shai-Hulud hashes and scans pip/uv caches for cached `lightning` 2.6.2/2.6.3
  artifacts.
- Research consensus: arXiv:2604.26919v1, Tencent Training-Free GRPO,
  PufferLib, and DeepSeek visual primitives are control-plane/diagnostic
  inputs only. They cannot promote, rank, kill, or enter archive runtime.
- Alpha/Lane 12: bounded visual-primitives artifact exists at
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_visual_primitives_bounded_20260501.json`.
  It covers 1200 scalar and visual frames using a 450M decoded-mask cache, but
  remains empirical CPU diagnostic only. It recommends
  `repair_or_retrain_before_exact_eval_spend` and does not create L2 clearance.
- Verification: parent focused Alpha tests `18 passed`; supply-chain focused
  tests `27 passed`; broad preflight/Lightning/KL/adjudication/J-NWC/NWCS
  regression slice `347 passed`; py_compile and scoped diff-check passed.
- Current blocker: Lightning SSH still rejects public-key auth, so the
  official component-response job is ready but not submitted non-dry-run.

## 2026-05-01T01:58Z Lightning Access Fixed, Exact-Response Jobs Running

- Update/supersession: Lightning SSH public-key auth is fixed through the
  alias `scratch-studio-devbox`. The previous SSH blocker is obsolete.
- The interactive Studio shell is CPU-only right now despite SSH health:
  `torch_cuda_available=false`, `torch_cuda_device_count=0`, `nvidia_smi=null`.
  Evidence path:
  `.omx/state/lightning_ssh_runtime_cuda_preflight_20260501_cpu_only.json`.
- Landed permanent DX guard:
  `scripts/lightning_repro_workspace.py --ssh-check-only --require-cuda`
  performs a remote Python/Torch CUDA probe and fails when the Studio shell has
  no GPU. Use this before interactive Lightning CUDA work. Batch Jobs still
  require their own in-job `lightning_runner_preflight.json`.
- Integrated provider-auth hardening worker output: static preflight now
  rejects disabled host-key checking, `/dev/null` known-hosts, and bare
  `ssh.lightning.ai` usage in Lightning scripts/runbooks. `AGENTS.md` records
  the durable rule.
- Official component-response jobs now active:
  - r1:
    `official_component_response_pfp16_a_plus_plus_20260501_codex_r1`, T4,
    Running, `--require-passed`, command SHA
    `772395f8e71bf67b095f2e36dd56479d52f82b25fab613b0e2dd61ccd71c0c45`.
  - r2:
    `official_component_response_pfp16_a_plus_plus_20260501_codex_r2_t4_small_race`,
    T4_SMALL, Running, `--require-passed`, command SHA
    `c7cc181f924f50df1ba65c10b30c78adef1f5bdb9e4615b75d3c655feb7432fe`.
  - r3:
    `official_component_response_pfp16_a_plus_plus_20260501_codex_r3_t4_no_gate`,
    T4, Pending, no `--require-passed`, command SHA
    `da87a91dc26a68a451a1326b33d234e7a4f77160c3f3cd521efecccaa6f23b5f`.
- R3 was intentionally submitted without `--require-passed` after adversarial
  review noted that the current official response plan lacks nonzero
  `predicted_delta` entries. It is diagnostic official CUDA evidence only and
  should not be promoted directly to `component_sensitivity_v1`.
- Verification in parent after merging hardening:
  py_compile clean; `bash -n` clean; focused Lightning/MCP tests `38 passed`;
  MCP cleanup strict reports no live MCP helper processes.
- Next turn starts with: refresh r1/r2/r3; harvest terminal artifacts; if
  r1/r2 fail only on promotion gates after curves are written, harvest without
  `--require-passed` for forensic official-response curves; assemble
  `component_sensitivity_v1` only after CUDA curves, maps, stability, sample
  plan, and custody are present.

## 2026-05-01T02:15Z Deterministic Lightning Harness And r4 Queue

- Supersession: r1/r2/r3 component-response attempts are not lane evidence.
  r1 failed before CUDA/DALI/input preflight on strict supply-chain scan
  because the snapshot still contained stale `tools/lightning_*` wrappers that
  invoke the PyPI `lightning` console script. r2 failed from the same stale
  snapshot class. r3 was stopped to avoid spend on known-bad provenance.
- Permanent harness fix landed:
  `scripts/launch_lightning_batch_job.py` now supports remote pre-submit
  supply-chain preflight via `--remote-preflight-ssh-target` for exact-eval and
  component-response submissions. Component-response harvest now supports
  `--job-name --state-path` and derives SDK artifact mirrors from recorded
  state, matching exact-eval harvest behavior.
- New client method:
  `LightningBatchJobsClient.harvest_ssh_component_response_artifacts`.
  Tests cover state-derived component-response harvest and remote strict scan
  before submit. Verification: `src/tac/tests/test_lightning_batch_jobs.py`
  45 passed; py_compile clean; scoped `git diff --check` clean; MCP cleanup
  matched zero helper processes.
- Runbooks and `AGENTS.md` now record the non-ad-hoc path:
  `comma-lab` teamspace, `lossy-compression-challenge` Studio,
  `scratch-studio-devbox` SSH alias, remote strict scan before submit, and
  state-derived component-response harvest. Manual `/teamspace/jobs/...` path
  composition is non-promotable.
- r4 was staged through `scripts/lightning_repro_workspace.py` with explicit
  artifacts and remote byte verification. Manifest:
  `.omx/state/official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful_manifest.json`;
  file_count 1114; total_bytes 21307573; manifest SHA
  `80d44b40b4048ee1d2c7ba850e1e98e45025eda65b248b12a494d6e1fdf1928e`.
- r4 submitted as diagnostic no-gate T4 Batch Job:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful`;
  command SHA `d9eec67b70b20b938dc76b66b34e0f498cc7d92e5307348c8798c0aa072a63c0`;
  latest status at first refresh: `Pending`. Harvest terminal r4 with
  `harvest-component-response-ssh --state-path .omx/state/lightning_batch_jobs.json
  --job-name official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful
  --ssh-target scratch-studio-devbox ...` without `--require-passed` unless a
  promotion-grade predicted-delta plan has replaced the current diagnostic plan.

## 2026-05-01T02:38Z r4 Forensics And r5 Active Queue

- r4 supersession: it is a harness/input-plan failure, not lane evidence. It
  passed remote strict supply-chain, hash-pinned DALI, and in-job CUDA T4
  runner preflight, then failed because the perturbation plan resolved
  `baseline_contest_auth_eval_json` to a host-local `/Users/adpena/...` path.
- Permanent fixes landed:
  `profile_component_sensitivity_official.py` and Lightning input preflight
  now let explicit `--baseline-contest-auth-eval-json` override stale plan
  metadata; `build_component_response_perturbation_plan.py` emits repo-internal
  portable relative paths when possible; component-response submit validation
  blocks absolute point archive/per-point JSON paths; Lightning staging and
  harvest transfers reuse noninteractive SSH/SCP/rsync auth options and
  `ConnectTimeout`.
- New portable plan:
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_portable_plan/official_component_response_plan.json`.
- Verification: py_compile clean and focused Lightning/staging/component-plan
  suite `85 passed`.
- Pre-submit doctor:
  `.omx/state/lightning_doctor_20260501_r5_pre_submit.json` passed local and
  remote supply-chain, SSH auth, and T4 machine inventory.
- Active job:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`,
  manifest SHA `ae3028935151c8e8e8f57315fa2a4f54edbfaebf3e6fd6c56064824e36f7e7e4`,
  command SHA `182c287d986a4fce61dbf12871b1e985bf01c4715b8e897e980e44d7e9c6ffa7`,
  latest status `Pending` at `2026-05-01T02:38:30Z`. Refresh until terminal and
  harvest with state-derived component-response SSH without `--require-passed`
  for diagnostic curves.

## 2026-05-01T02:55Z Closure Hardening And Live Queue State

- r6 component-response race job was staged and submitted:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race`.
  Remote manifest SHA:
  `91cd1e8011a7045a3068b0a2a4a74b0f842be6b8be4232da91473e6445780684`;
  command SHA:
  `8cd111eb0b3448c1f9143929a96b0fd990afd361e6488c66a2c7c2ed0086deec`.
- Latest Lightning statuses: r5 `Running` at `2026-05-01T02:48:23Z`; r6
  `Pending` at `2026-05-01T02:48:24Z`. Harvest only terminal artifacts through
  state-derived SSH; current interactive Studio filesystem checks do not show
  live job output dirs and must not be used for score interpretation.
- New permanent bug-class fixes:
  `scripts/launch_lightning_batch_job.py` validates source-manifest paths for
  exact-eval and component-response: no absolute paths, traversal, duplicate
  entries, backslashes, controls, unstable separators, hidden files, or macOS
  resource forks. `_remote_repo_rel` also validates its derived repo-relative
  paths. Exact-eval submit now validates staged custody of queue-metadata
  `baseline_json`/`baseline_contest_auth_eval_json`.
- `src/tac/deploy/lightning/batch_jobs.py` direct SSH harvest helpers now reject
  bare `ssh.lightning.ai` and unsafe targets, not only the CLI wrapper.
- Verification: py_compile clean and `src/tac/tests/test_lightning_batch_jobs.py`
  passed `57` tests.
- Parallel audits:
  Vast live inventory empty; Modal app list shows zero live tasks; no provider
  kill is needed. No promotable local `component_sensitivity_v1.json` exists;
  OWV3 Fisher maps are diagnostic only until official CUDA component-response
  curves and stability/custody gates are assembled.
- Active worker: `Sagan the 2nd` is implementing prebuilt corpus-manifest +
  replay-root custody for J-NWC/NWCS remote scripts to remove corpus
  regeneration drift before CUDA exact eval.
- Latest provider status at `2026-05-01T02:54Z`: r5 component-response remains
  `Running` on T4; r6 component-response is also `Running` on T4_SMALL. Vast
  has zero live instances. Modal app list shows zero tasks; polling stale call
  IDs for `lane_sa_v4`, `lane_sc_plus_plus_v4`, `mae_v_v2`, `q_faithful_v3`,
  `stc_cuda`, and `sz_phase2_v2` returned `STILL RUNNING`, with no artifacts
  harvested.

## 2026-05-01T03:10Z r5 Curves, NWCS Custody, Alpha Gate

- r5 component-response completed and was harvested:
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`.
  It is CUDA/T4 official response evidence with baseline archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  but diagnostic only: `promotion_eligible=false`.
- r5 curve signal:
  PoseNet observed delta max `0.0003012800000000001`, SegNet
  `1.3420000000000099e-05`, combined `0.006991338976567674`. Coverage,
  finite values, signal, and zero repro pass. Promotion blockers:
  `missing_prediction_deltas` and `prediction_error_gate_failed`.
- Fixed harvest mirror bug class: copied read-only validation JSON is replaced
  atomically and chmodded to `0644`; Lightning batch tests remain `57 passed`.
- J-NWC/NWCS custody patch landed and parent-verified: prebuilt
  corpus-manifest and replay-root support in trainer/remote scripts; NWCS
  promotion now requires matching corpus-manifest custody when using
  `CORPUS_SENSITIVITY_PT`. Focused J-NWC/NWCS tests: `42 passed`.
- Alpha-Geo Lane G current rerun:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_current_20260501.json`,
  SHA `6b17b004d238ada62180077aa072f02594954ef02f5a5c610bc70e65619fa80d`,
  `overall_pass=false`, global disagreement `0.012303928799099393`, 2px
  boundary disagreement `0.14883144511692872`, missing-component rate
  `0.4611606740560512`. Lane 12 remains blocked for retrain/exact eval until
  a successor archive passes Alpha-Geo plus pose-regeneration provenance.
- r6 T4_SMALL duplicate diagnostic remains `Running` at `2026-05-01T03:07:35Z`,
  cost `0.048555557`. Harvest when terminal only if cross-machine diagnostic
  reproducibility is useful; do not promote.
- r6 completed and was harvested as cross-machine diagnostic replication:
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race`.
  Same blockers as r5, same non-promotable status. R6-r5 max absolute delta
  differences: PoseNet about `4.0e-7`, SegNet about `6.0e-8`, combined about
  `8.3e-6`.

## 2026-05-01T05:11Z Continuation Memory

- MCP is globally disabled for this project/operator environment. Deleted known
  MCP configs/backups/caches/tool-output/OAuth/state artifacts from Claude,
  Cursor, Gemini, LM Studio, Codex/project-local homes, plus `.playwright-mcp`
  artifacts. Final cleanup command
  `scripts/kill_orphaned_mcp_processes.py --strict --json` reported zero live
  matches and zero remaining PIDs.
- Hardened MCP killer/preflight false positives: audit/search commands that
  mention `mcp` or `model.context` are no longer killed or reported as live
  helpers; actual helper launch forms remain blocked. Focused tests: `6
  passed`.
- r5/r6 component-response runs are diagnostic only. Treat their previous
  zero-repro interpretation as superseded because eps=0 used an external
  copied baseline JSON, not same-run zero. Promotion now requires same-run
  eps=0 under `--require-passed`.
- Added structured pre-response prediction deltas:
  `experiments/build_component_response_prediction_deltas.py`, format
  `official_component_response_prediction_deltas_v1`. Perturbation plan
  promotion ingestion now rejects legacy arbitrary predicted-delta maps and
  observed-response leakage, and ties predictions to baseline archive custody
  plus perturbation-basis atom-set SHA.
- Component-sensitivity manifest validation now requires explicit finite
  `gate_results` all true and no blockers; focused response/sensitivity suite
  passed `73`.
- New research ledger:
  `.omx/research/lightning_ecosystem_repo_intake_20260501_codex.md`.
  Initial Lightning repo decisions: LitModels research only due optional
  `lightning`/`pytorch-lightning` paths; lightning-thunder opt-in performance
  research only until deterministic CUDA parity/provenance audit; utilities
  useful for local doctor/rank-zero/import-helper patterns without adding
  broad dependencies or changing exact-eval custody.

## 2026-05-01T06:01Z Continuation Memory

- R1 Lightning diagnostic component-sensitivity failures were harness bugs:
  generated `--response-epsilons -0.002,...` made argparse think the negative
  value was another option. Fixed generator to emit
  `--response-epsilons=-0.002,...`; regression test landed.
- Added reusable diagnostic component-sensitivity validation/harvest wrappers
  in the Lightning Batch launcher. Harvest through state-derived
  `harvest-component-sensitivity-ssh`; do not hand-compose `/teamspace/jobs`
  paths or copy raw bulky dirs.
- Submitted fixed r2 CUDA diagnostic sensitivity jobs:
  T4 `component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2` and
  L40S
  `component_sensitivity_pfp16_a_plus_plus_cuda_fisher_l40s_20260501_r2`.
  Latest refresh at 06:01Z: both `Pending`, zero cost. Source manifest SHA:
  `8d5eeb9c267c0ee6c3019710b1cdc3b799559833f8c86eebdc3497da6675ad66`.
- J-NWC/NWCS audit: no promotable sensitivity artifacts yet; CPU/Fisher/proxy
  maps are diagnostic. Corpus-manifest/replay custody is in place, but
  promotion waits for real `component_sensitivity_v1`,
  `ANCHOR_SENSITIVITY_PT`, and `CORPUS_SENSITIVITY_PT`.
- Alpha-Geo audit: Lane 12 `jsonfix40` remains blocked by geometry failure and
  absent L2 clearance. Do not spend exact eval/retraining until a successor
  passes Alpha-Geo and pose-regeneration provenance.
- Live provider state: Vast `[]`; Modal has zero live tasks, with harvest
  backlog only. MCP strict cleanup reports zero live helper processes.
- Verification this turn: `py_compile` clean, J-NWC/NWCS shell syntax clean,
  focused pytest `152 passed`, `git diff --check` and cached check clean.
- Local strict Lightning supply-chain scan artifact:
  `.omx/state/lightning_supply_chain_scan_local_20260501T0601Z_codex.json`,
  `status=OK`, `violation_count=0`, `lightning`/`pytorch-lightning` absent,
  `lightning-sdk==2026.4.10`.

## 2026-05-01T06:17Z Continuation Memory

- Added `experiments/build_component_response_plan_from_sensitivity_artifacts.py`
  and tests. It validates harvested diagnostic component-sensitivity artifacts,
  builds pre-response prediction deltas, and emits an official response plan.
  It accepts `--perturbation-basis-json` so fresh basis selection can be used
  instead of any stale/post-hoc basis. Outputs are planning only, not score
  evidence.
- L40S r2 diagnostic sensitivity completed and was harvested locally at
  `experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_l40s_20260501_r2`.
  CUDA device `NVIDIA L40S`, baseline SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  bytes `686635`, 600 pairs, diagnostic Fisher only, non-promotable.
- Built fresh-basis response packet:
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex`.
  Basis SHA `2cf8f8c6940d7fd905068fe67a797c50929f60873951901fb09ad9bbc5bbb3aa`;
  prediction-deltas SHA
  `e9deb2f21fa132e730d692a1fff046e6171dae04621b850696946ebae3089a3d`;
  plan SHA `4f810618bc65ca9f72705cb6afe95f67fbfdf28e52252b1a38a26b6329521c43`.
- Submitted L40S official component-response job with `--require-passed`:
  `official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_l40s`.
  Latest status: `Pending` at 06:17Z.
- T4 sensitivity job still `Running`. Harvest it when terminal; use it as
  promotion-preferred sensitivity source if clean.
- Subagent decisions: plain J-NWCS first after promotable
  `component_sensitivity_v1`; no EC stack before exact plain evidence. Lane
  12 remains blocked by geometry and needs decoded-baseline retraining with
  lane/boundary/transition protection. Current git index has stale rollback
  content; do not commit until index is repaired safely.

## 2026-05-01T06:37Z Continuation Memory

- Landed certified component-sensitivity map path:
  `experiments/certify_component_sensitivity_maps.py`,
  `src/tac/sensitivity_map.py`,
  `experiments/build_component_sensitivity_manifest.py`, and
  `src/tac/component_sensitivity_artifact.py`. Promotion manifests now require
  certified `tac_score_sensitivity_map_v1` maps with
  `component_sensitivity_map_certification_v1` metadata. Raw diagnostic maps
  and clean-but-uncertified maps fail closed.
- Durable protocol added to `AGENTS.md` and research ledger
  `.omx/research/component_sensitivity_map_certification_20260501_codex.md`.
  Never strip diagnostic metadata from maps; certification copies tensors into
  new artifacts and cites source SHA, official response SHA, stability/sample
  SHA, baseline custody, response/stability gates, and >=3 clean reviews.
- Alpha-Geo worker landed `alpha_geo_primitive_contract_v1` emission via
  `experiments/diagnose_nerv_geometry.py --primitive-contract-json`. It is
  empirical/no-claim only and will feed decoded-baseline Lane 12 retraining
  design. Worker tests: `19 passed`.
- T4 diagnostic sensitivity r2 completed and was harvested at
  `experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2`.
  CUDA/T4, 600 pairs, diagnostic Fisher/proxy, non-promotable.
- L40S official response r7 failed and was harvested at
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_l40s`.
  Main blockers: `prediction_error_gate_failed` for all components; also
  same-run L40S eps=0 baseline drifted from supplied PFP16 A++ T4 baseline
  JSON, so this is calibration drift evidence, not certification evidence.
- Hardened `experiments/profile_component_sensitivity_official.py` to gate
  `external_baseline_repro` when an external baseline JSON is supplied.
  Certifier also rejects curves with external-baseline repro false.
- T4 official response r7 is still running as of `2026-05-01T06:36:45Z`,
  cost `0.1431889`. It was queued before external-baseline hardening, so if it
  completes, manually compare eps=0 components against the PFP16 A++ baseline
  JSON before considering any response gate useful.
- Verification this segment: py_compile clean; certification/manifest suite
  `64 passed`; official-response/certifier/artifact suite `58 passed`; Alpha
  contract suite `19 passed`; touched-file `git diff --check` clean; MCP
  strict cleanup zero.

## 2026-05-01T06:55Z Continuation Memory

- T4 official response r7 reached terminal `Failed` and was harvested at
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_t4_parallel`.
  It is diagnostic CUDA official-response evidence only. Supply-chain and T4
  CUDA preflight were clean.
- T4 r7 failed prediction-error gates on all components:
  posenet max relative error `85639.15781135917`, segnet
  `46771.67473424154`, combined `553.2107592118479`. Same-run eps=0 produced
  score `1.0370448266022327` from PoseNet `0.00316423`, SegNet `0.0040196`,
  while the PFP16 A++ anchor JSON records recompute `1.043987524793892`,
  PoseNet `0.00346442`, SegNet `0.00400656`. Treat as prediction-model
  failure plus runner/scorer drift; no promotion/kill/stack claim.
- Lightning status hardening landed: status refresh now records full SDK
  snapshots, `status_anomalies`, name-only identity confidence, and
  reconciliation-required state. The live r7 record backfilled the
  `Running -> Pending` anomaly and then terminal `Failed`; `identity_confidence`
  is `name_only`.
- Alpha contract consumption landed in `experiments/train_nerv_mask.py` and
  `src/tac/nerv_mask_codec.py`: decoded-baseline training requires
  `--alpha-primitive-contract`, validates decoded-mask SHA/shape, builds
  deterministic weighted sampling pools, and remains empirical/no-score.
  `gt_masks_source=segnet` now requires `--allow-forensic-segnet-target`.
- Certifier hardening landed: `experiments/certify_component_sensitivity_maps.py`
  now requires prediction-deltas JSON and perturbation-basis JSON, cross-checks
  atom IDs/epsilon ladders, verifies curve perturbation SHAs, and enforces
  external-baseline repro when supplied. Component-response artifact validation
  also de-promotes missing/false external-baseline repro gates.
- Verification: `py_compile` clean; consolidated focused suite
  `203 passed in 5.79s`; Lightning suite `69 passed`; certifier suite
  `4 passed`; `bash -n scripts/remote_lane_nerv.sh` clean; `git diff --check`
  clean on touched files; MCP strict cleanup zero.
- Next best wall-clock sequence: run direct finite-difference CUDA maps with
  full 600-pair coverage, build R8 prediction deltas from those maps plus a
  fresh archive-byte basis, run official response with external-baseline repro,
  certify maps, then unlock OWV3/NWCS exact eval. Use Alpha contract path for
  build-only NeRV retraining and Alpha-Geo diagnostics before any exact eval.

## 2026-05-01T07:10Z Continuation Memory

- Direct finite-difference component-sensitivity is now wired as a Lightning
  `component-sensitivity` job mode via `--promotion-finite-difference` and
  `--finite-difference-epsilon`. It remains diagnostic/no-score but is the only
  certification handoff source; Fisher/proxy remains planning-only.
- `validate_local_component_sensitivity_artifact_dir` and the remote generated
  validator now inspect summaries, run metadata, input preflight, response
  curves, and `tac_score_sensitivity_map_v1` map metadata for non-score,
  non-promotable, non-official, non-canonical status and allowed
  `sensitivity_source`.
- `experiments/profile_component_sensitivity.py` now emits
  `score_claim=false` in diagnostic summaries and curves.
- `scripts/remote_lane_nerv.sh` now requires and forwards
  `ALPHA_PRIMITIVE_CONTRACT` for decoded-baseline Lane 12 dispatch, records
  contract metadata in provenance, and fails closed on invalid
  `alpha_geo_primitive_contract_v1`.
- Verification: py_compile clean; `bash -n scripts/remote_lane_nerv.sh` clean;
  `src/tac/tests/test_lightning_batch_jobs.py` plus
  `src/tac/tests/test_lane12_nerv_dependency_closure.py` = `97 passed`;
  touched-file `git diff --check` clean; MCP cleanup zero.
- Next highest-EV engineering task: implement deterministic direct-FD
  layer/channel sharding plus merge validation before spending serious T4 time.
  Then restage Lightning source, submit shards, harvest state-derived artifacts,
  build prediction deltas/basis, run official response with external-baseline
  repro, certify maps, and unlock OWV3/NWCS exact eval.
- Telemetry check: local Lightning state has no running jobs; recent response
  and sensitivity records are terminal/harvested/failed, with two old stopped
  records. Vast reconcile reports `live_count=0`; `.omx/state` Vast trackers
  are stale and not live evidence.
