# Agent Onboarding - Comma Video Compression Challenge

This repository is operated as a contest-grade research and engineering system.
The objective is to drive toward the Shannon-floor frontier in the shortest
wall-clock time while preserving exact reproducibility, contest compliance,
scientific rigor, and mathematical rigor.

Do not treat this file as a result ledger. Do not add transient scores,
leaderboard ranks, lane outcomes, or one-off findings here. Store findings and
results in the dated `.omx/research/` ledgers and experiment artifact
directories. This file is for durable protocols, codebase structure, and
non-negotiable operating rules.

## Source-Of-Truth Documents

Use these documents as the research/control plane:

- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `.omx/research/council_paradigm_shift_round{1,2,3}_20260430.md`
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
- `.omx/research/shannon_floor_execution_readiness_20260430.md`
- `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
- `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`
- `.omx/research/codex_recursive_adversarial_greenup_review_20260430.md`
- `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`
- `.omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/kl_distill_hardening_status_20260430_codex.md`
- `.omx/research/component_sensitivity_map_certification_20260501_codex.md`

When these documents disagree, prefer the strictest contest-grade evidence
standard and the newest dated progress addendum. Preserve history by appending
supersession notes instead of erasing old research context.

## Codebase Map

- `src/tac/` - training, codecs, archive helpers, profiles, guards, and tests.
- `experiments/` - canonical training/eval/build entry points and lane tools.
- `scripts/` - remote lane launchers, adjudicators, harvesters, and runbooks.
- `submissions/robust_current/` - contest submission runtime and inflate path.
- `upstream/` - contest evaluator assets. Do not patch scorer files.
- `reports/` - derived reports and non-authoritative summaries.
- `.omx/state/` - dispatch state. Treat as advisory when live API or lane-local
  artifacts disagree.
- `.omx/research/` - dated scientific, mathematical, adversarial, and progress
  ledgers.

## Contest Objective

All score claims reduce to the contest formula:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

Every scored archive must record exact archive bytes, archive SHA-256,
component distances, sample count, recomputed score, eval command, hardware,
manifest, provenance, and logs.

## Evidence Grades

Use evidence grades rigorously:

- `A++`: exact 1:1 contest-grade archive evidence. Requires exact archive
  custody, clean manifest, payload closure, canonical `archive.zip ->
  inflate.sh -> upstream/evaluate.py` path, CUDA, full sample count,
  T4/equivalent or official contest-equivalent hardware, inflate budget proof,
  and adversarial review.
- `A`: exact local CUDA score-grade archive evidence with full component
  recomputation and archive custody, but not necessarily contest-equivalent
  hardware.
- `A-negative`: exact archive CUDA evidence showing a measured implementation
  regresses. This supports diagnosis and redesign, not broad family kills.
- `B`: diagnostic CUDA evidence with incomplete custody, schema, or rerun
  proof.
- `empirical`: byte, smoke, loss, round-trip, partial, or component evidence.
- `derivation`: formula-only conclusion.
- `prediction`: hypothesis or forecast.
- `external`: outside paper, OSS, or leaderboard intake.
- `invalid`: CPU, MPS, proxy, stale, no-op, sidecar, missing archive, or
  unreproducible score evidence.

No lane can promote, rank, kill, or anchor stack math from prediction,
byte-only, CPU, MPS, proxy, smoke, memory-only, or stale-log evidence.

## CUDA Auth Eval Is The Score Truth

MPS, CPU, local proxy scorers, local renderer checks, and non-canonical eval
paths can materially distort SegNet/PoseNet behavior and total score. They are
useful only for development, byte checks, shape checks, smoke tests, and bug
triage.

For any GPU-dependent score or signal claim, the only reliable source of truth
is exact CUDA auth eval on the exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Prefer `experiments/contest_auth_eval.py --device cuda` for this path and use
its `contest_auth_eval.json` as the canonical artifact. Local M-series/MPS or
CPU output must never promote, rank, kill, retire a method, validate a stack,
or anchor paper claims. If a local/MPS result disagrees with CUDA auth eval,
CUDA auth eval wins.

## Contest Compliance

Non-negotiable compliance rules:

- Neural/runtime artifacts required by inflate must be inside `archive.zip` or
  fixed contest code.
- Do not modify upstream scorer files.
- Do not use local renderer shortcuts for score claims.
- Do not load score-affecting sidecars outside the archive.
- Archive manifests must exclude resource forks, hidden files, caches, debug
  payloads, and zip-slip paths.
- Use deterministic archive construction: fixed member ordering, timestamps,
  permissions, compression settings, and manifest records.
- Exact eval path is canonical: `archive.zip -> inflate.sh ->
  upstream/evaluate.py`, preferably through `experiments/contest_auth_eval.py`.
- JSON artifacts are authoritative. Do not parse scores from human logs when a
  structured `contest_auth_eval.json` exists.
- Recompute the score from components before claiming any result.
- Exact adjudication may reject on PoseNet/SegNet component gates even when
  the total score is in band. Component collapse is a first-class failure mode.
- Learned codec/corpus lanes must emit deterministic manifests with checkpoint
  paths, sizes, hashes, selected tensors, block counts, and exclusion reasons.
- No dummy/random sensitivity, hard-pair, Fisher, or scorer-side proxy signal
  is allowed in a promotable dispatch. Smoke/debug modes must be explicit and
  non-promotable in provenance.
- Archive and diagnostic ZIP handling must be zip-slip safe: no absolute paths,
  parent traversal, resource forks, or hidden sidecars.

## Kill Discipline

Do not use broad `KILL`, `dead`, or permanent-retirement language unless the
Grand Council has completed deep adversarial review and reached three clean
consensus passes on scope.

For any bad, surprising, or disappointing result:

1. Preserve the exact archive, JSON, logs, manifest, SHA, source provenance,
   environment, and command before cleanup.
2. Recompute score from components and verify device, sample count, archive
   bytes, eval path, and payload closure.
3. Classify failure mode: legitimate regression, harness bug, archive bug,
   no-op/encode-discard bug, config/dead-flag bug, CPU/MPS/proxy leakage,
   sidecar dependency, codec attribution confound, KL/PoseNet collapse, data
   geometry mismatch, timeout/NVDEC infrastructure, or indeterminate.
4. Run engineering, mathematical, geometry, and optimization review before
   drawing a conclusion.
5. Run mitigation and stacking analysis before retirement: hybrid residuals,
   fallback routing, side-info accounting, per-region gating, PFP16/SA/H-V3/
   OWV3-style composition, or other full-stack rescue paths.
6. Run leaderboard reverse-engineering analysis: archive member sizes,
   representation family, raw-output geometry, stream allocation, and likely
   full-stack strategy.
7. Scope any retirement narrowly to the measured implementation/config unless
   independent exact reproductions or a mathematical impossibility proof support
   a broader claim.

Use these status words precisely:

- `run abort`: budget, timeout, smoke, or control threshold. This is not
  scientific failure evidence.
- `measured-implementation retired`: exact artifact/config failed after custody,
  scorer, archive, and harness checks.
- `family/method killed`: only after independent exact evidence or a
  mathematical impossibility argument plus clean Grand Council consensus.

Every negative result should produce redesign options, not just a verdict.

## Scientific And Mathematical Rigor

- Distinguish measurement, derivation, prediction, and external motivation.
- Additive component deltas are not composable until standalone exact evals and
  a stacked exact eval exist.
- Dykstra/ADMM language is a feasibility and projection discipline, not proof
  that sampled nonconvex codec deltas compose.
- Use Dykstra-style intersection constraints: rate budget, SegNet distortion,
  PoseNet distortion, archive compliance, inflate budget, and reproducibility.
- Treat side information as charged bytes inside the archive.
- Require calibration/holdout stability for sensitivity maps and learned
  allocation rules.
- Keep exact eval single-device unless distributed sampling and aggregation are
  audited for no duplicate/missing samples.
- No Shannon-floor attainment claim is allowed without exact contest-grade
  evidence. The broad mandate is to push aggressively toward the floor, not to
  inflate claims.

## Shannon-Floor Execution Policy

Shortest wall-clock progress comes from parallel independent hypotheses, not
from serial speculation.

Operate these streams in parallel when write sets and scientific failure modes
are independent:

- Alpha representation work: mask/video/latent payloads that preserve scorer
  geometry and temporal information.
- Beta sensitivity work: per-channel/per-region score sensitivity,
  mixed-precision allocation, water-filling, and PoseNet/SegNet protection.
- Renderer compression work: renderer-byte reduction with exact distortion
  gates.
- Pose-byte work: small deterministic archive wins with no scorer side effects.
- Hidden-gem recovery: bugged lanes re-engineered under strict evidence gates.
- Gamma coordination: ADMM/MDL/entropy/hyperprior/range or arithmetic coding
  only after measured components exist.

Stack experiments wait until component archives have exact evidence. A stack is
its own archive and must pass its own exact eval.

Lane 12/Alpha NeRV retraining is build-only until explicit L2 clearance.
Production training targets must use decoded baseline archive masks with a
validated `alpha_geo_primitive_contract_v1`; direct `gt_masks_source=segnet`
is forensic/debug only behind an explicit flag. Contract consumption must
validate decoded-mask SHA and shape, record contract SHA and sampling gates,
and preserve weighted sampling provenance for uniform, critical-box,
boundary-band, and transition-endpoint pools. These trainer artifacts are
empirical/no-score until a later canonical CUDA archive eval is run.

## Backend Routing

- Lightning AI is the preferred exact-eval home for T4/equivalent
  promotion-grade runs. Use hermetic staged trees and preserve source manifests.
- Modal is approved for build-only, smoke, Fisher/sensitivity, ablation, and
  cheap exploratory work. Modal auth evals are advisory unless the wrapper
  proves CUDA exact eval and all contest gates.
- Vast.AI is useful for cheap parallel training but has host/NVDEC and state
  drift hazards. Trust lane-local artifacts and live API over stale trackers.
- Local M-series/MPS work is for development, smoke, and byte/round-trip checks
  only. It cannot rank or promote.
- Do not trust remote state ledgers without reconciliation. Use
  `scripts/reconcile_vast_dispatch_state.py` for non-mutating drift reports.
- No new retraining lane should be dispatched before Lane 12/Alpha has an
  explicit L2 unblock packet at `.omx/state/lane12_nerv_l2_clearance.json`.
  Build-only, harvest, and exact-eval-only lanes may continue. Retraining
  dispatches must fail closed unless the packet records
  `cleared_for_retraining_unblock=true`, `lane12_l2=true`,
  `geometry_gate_passed=true`, `grand_council_clean_passes>=3`, and evidence
  paths. The Vast retry launcher enforces this gate.

## Lightning And PyTorch Lightning Guidance

Use Lightning AI infrastructure aggressively, but keep the contest path
deterministic.

- Use `scripts/lightning_repro_workspace.py` for Lightning Studio staging.
  One-off `rsync` is acceptable only for emergency debugging and must be
  superseded by a source/artifact manifest before promotion. Generated payloads
  must be passed with explicit `--artifact`; bulky experiment outputs,
  checkpoints, videos, and archives are not source.
- Use `scripts/lightning_repro_workspace.py --ssh-check-only --require-cuda`
  before interactive Lightning CUDA work. Plain SSH success only proves the
  Studio is reachable; it does not prove a GPU is attached. If this runtime
  probe reports `torch_cuda_available=false`, do not run or promote
  interactive CUDA work from that Studio shell. Batch Jobs may still run on
  requested GPU machines, but their own `lightning_runner_preflight.json` is
  the authority.
- A Lightning staged tree must preserve a local and remote manifest with file
  count, bytes, SHA-256, source/artifact role, git status, command, and
  environment JSON. Exact eval jobs must cite that manifest in provenance.
- Before non-dry-run Lightning submit, run
  `scripts/launch_lightning_batch_job.py doctor --ssh-target <alias>
  --require-ssh --require-remote-supply-chain --require-machine-inventory`
  and preserve its JSON. Doctor passing is not score evidence, but a failed
  doctor blocks dispatch.
- Component-response perturbation plans used for remote Batch Jobs must not
  contain host-local absolute paths in point archives or per-point eval JSON.
  Plan paths must be relative to the plan file, and the staged source manifest
  must include every resolved file. If a plan contains stale top-level
  `baseline_contest_auth_eval_json`, the explicit CLI
  `--baseline-contest-auth-eval-json` path is the authority and must point
  inside the remote repo.
- Remote Lightning `uv sync` must use copy-mode installs:
  `UV_LINK_MODE=${UV_LINK_MODE:-copy} uv sync --locked --extra runtime`.
  Studio filesystems can fail hardlink installs while materializing Torch; this
  is a permanent DX guard, not an ad hoc workaround.
- Exact-eval Batch Jobs must isolate inflate-side `uv run` environments from
  the scorer environment. Export a per-job `UV_PROJECT_ENVIRONMENT` under the
  job output dir before `contest_auth_eval.py`; otherwise `inflate.sh` can
  recreate the shared repo `.venv` and break `upstream/evaluate.py`.
- Exact-eval Batch Jobs that mutate/check the shared `.venv` for DALI/bootstrap
  must hold `.omx/state/lightning_exact_eval_venv.lock` while doing so. Parallel
  jobs may run the scorer concurrently only after this setup phase is complete.
- Lightning Batch Job artifact paths must be derived from
  `lightning_sdk_job_name()` or an explicit output directory. The SDK normalizes
  underscores to hyphens for `/teamspace/jobs/<job>/artifacts`; do not
  hand-compose artifact paths from the local queue name.
- `scripts/launch_lightning_batch_job.py refresh-status` should be run with
  only `--state-path` and `--job-name` when the job was queued locally; it
  infers the SDK job name, teamspace, org, and user from the state record to
  avoid operator drift.
- Lightning SDK status strings are telemetry, not standalone custody. If
  refresh history shows a nonterminal regression such as `Running -> Pending`,
  the local state must record `status_anomalies`,
  `status_reconciliation_required=true`, and full per-refresh snapshots. For
  non-dry-run exact/component-response/sensitivity jobs, nonterminal status
  regressions must fail closed as `REMOTE_STATUS_RECONCILIATION_REQUIRED`
  unless a terminal SDK status supersedes them; terminal artifacts still need
  harvest validation before scientific use.
- Lightning refreshes that only resolve jobs by name must record
  `identity_confidence=name_only` and
  `identity_reconciliation_required=true`. Prefer stable SDK job ids whenever
  the SDK exposes them; null-id name-only refreshes are not enough for
  promotion custody without state-derived artifact validation.
- Non-dry-run Studio-backed Lightning Batch submissions must use
  `--remote-preflight-ssh-target <alias>` unless a specific auditable
  break-glass reason is recorded. This runs
  `scripts/scan_lightning_supply_chain.py --quiet --strict` on the remote
  Studio tree immediately before SDK submission, so stale snapshots with
  compromised `lightning` CLI wrappers fail before spending GPU time.
- Lightning staging and harvest SSH/SCP/rsync operations must use noninteractive
  auth policy: `BatchMode=yes`, password and keyboard-interactive auth
  disabled, and an explicit `ConnectTimeout`. Preflight-only SSH hardening is
  insufficient if the actual copy/harvest commands do not reuse the same
  policy.
- Studio-backed Lightning jobs persist Studio paths under the SDK artifact
  mirror, e.g. `/teamspace/jobs/<job>/artifacts/pact/...`. Use
  `scripts/launch_lightning_batch_job.py harvest-ssh --job-name ...` for
  terminal harvests; it derives the persisted path from the recorded
  `remote_output_dir`, copies only canonical evidence files, and validates
  archive/adjudication JSON locally. Do not `scp -r` whole eval directories;
  they can contain multi-GB raw frames and break custody.
- Official component-response harvests must also be state-derived:
  `scripts/launch_lightning_batch_job.py harvest-component-response-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name <job> ...`.
  The wrapper maps recorded Studio output dirs into SDK artifact mirrors and
  validates compact response evidence locally. Do not hand-compose
  `/teamspace/jobs/...` paths for promotion-grade claims.
- Diagnostic component-sensitivity harvests must also be state-derived:
  `scripts/launch_lightning_batch_job.py harvest-component-sensitivity-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name <job> ...`.
  These artifacts are non-promotable unless later assembled into a reviewed
  `component_sensitivity_v1` packet through the official CUDA component
  response path.
- After a diagnostic CUDA component-sensitivity harvest, use
  `experiments/build_component_response_plan_from_sensitivity_artifacts.py`
  to validate the harvested artifact directory, build pre-response
  `official_component_response_prediction_deltas_v1`, and emit the deterministic
  official response plan. This remains planning evidence only; score signal
  still requires the subsequent official CUDA component-response Batch Job with
  same-run eps=0 and `--require-passed`.
- Generated Lightning Batch commands must emit option values that may begin
  with `-` using the `--flag=value` form, not `--flag value`. This is required
  for epsilon ladders such as `--response-epsilons=-0.002,...`; otherwise
  argparse can treat the negative value as another option and fail remotely.
- Supply-chain rule: do not install the PyPI package named `lightning` in this
  repo or on remote runners. On 2026-04-30, `lightning==2.6.2` and
  `lightning==2.6.3` were reported compromised with import-time credential
  theft. Use `lightning-sdk` for Lightning AI Batch Jobs/CLI work.
- Run `scripts/scan_lightning_supply_chain.py --strict` before trusting a new
  local or remote runner for exact eval. Preserve the JSON output under
  `.omx/state/` with the runner/date in the filename.
- The Lightning supply-chain scanner must stay current with incident IOCs,
  including reported `router_runtime.js`, `_runtime/start.py`,
  malicious `lightning/__init__.py`, and `lightning` wheel hashes, plus pip/uv
  cache scans for cached `lightning` 2.6.2/2.6.3 artifacts.
- Do not execute `lightning` CLI probes just to discover installation state.
  Inspect package metadata or known console-script targets instead. A poisoned
  `lightning` executable on `PATH` can be an import-time trigger.
- Do not call `.venv/bin/lightning`, bare `lightning`, or `$LIGHTNING` from
  operator scripts. Use SSH-backed wrappers or `lightning-sdk` APIs. The strict
  supply-chain preflight scans `scripts/`, `tools/`, and Lightning deploy
  helpers for these stale console-script paths.
- Treat any environment that installed and imported `lightning==2.6.2` or
  `lightning==2.6.3` as compromised until isolated and credentials are rotated.
- Scan for Mini Shai-Hulud indicators before trusting a runner or repo:
  `.claude/router_runtime.js`, `.claude/setup.mjs`, `.vscode/setup.mjs`,
  `.github/workflows/format-check.yml`, hidden `lightning/_runtime/`, and
  npm `postinstall` hooks that run `setup.mjs`.
- Borrow Lightning Fabric patterns selectively for optional training-lane
  wrappers, seed/rank-zero discipline, callback organization, and optional
  training-state checkpoints.
- Treat Lightning-AI ecosystem repos such as LitModels, lightning-thunder, and
  utilities as research inputs, not promotion dependencies, until they pass the
  local supply-chain scanner, deterministic replay checks, CUDA parity checks,
  and import audit. Copy/adapt small, audited patterns where useful; do not add
  broad dependencies or cloud model-registry custody to contest artifacts.
- LitModels may inform checkpoint/registry ergonomics, but its optional
  Lightning/PyTorch-Lightning integrations are outside promotion environments.
  Do not install extras that pull the PyPI `lightning` package.
- lightning-thunder may be evaluated only behind opt-in profiling/training
  flags. It is not allowed in canonical exact eval or score custody until
  numerical parity, deterministic behavior, compile-cache effects, and CUDA
  runtime provenance are adversarially audited.
- lightning-utilities patterns such as rank-zero logging, import/version
  helpers, and dependency CLI checks may be adapted if they reduce local code
  fragility. Keep local wrappers independent of `lightning` imports.
- When adapting `lightning-utilities` import helpers, do not point any helper
  that imports the target module at `lightning`; use metadata-only package
  inspection for high-risk names. Avoid broad requirement-rewrite helpers for
  Pact because `uv.lock`, upper bounds, and reviewed dependency custody are
  authoritative.
- Do not migrate canonical archive construction or exact eval into a full
  PyTorch Lightning `Trainer` loop.
- Avoid DDP for exact eval unless sampler and aggregation are audited.
- Avoid remote loggers/artifact managers in canonical result custody.
- Keep authoritative artifacts as local JSON, logs, manifests, `.pt` files, and
  ZIP archives harvested into `experiments/results/`.

## Component Sensitivity And OWV3

- Promotion-grade sensitivity artifacts must be CUDA-authored and must separate
  PoseNet, SegNet, and combined scorer signal. A single proxy/Fisher tensor is
  not enough for paper or deployment claims unless the component breakdown,
  calibration split, holdout stability, and response-curve validation are
  recorded.
- The target schema is `component_sensitivity_v1`: manifest, PoseNet map,
  SegNet map, combined map, per-pair metrics, perturbation response curves,
  command, environment, source manifest, input hashes, sample plan, stability
  metrics, and optional exact eval custody.
- Use `experiments/build_component_sensitivity_manifest.py` to assemble
  `component_sensitivity_v1` packets from real CUDA maps, response curves,
  exact eval JSON, and archive artifacts. Do not hand-edit promotable
  sensitivity manifests.
- Current `experiments/profile_component_sensitivity.py` output is
  diagnostic Fisher-proxy evidence only, even on CUDA. It may produce maps,
  response-curve JSON, stability JSON, and sample plans for design/debugging,
  but it deliberately records `promotion_eligible=false` and blocks
  `--manifest-output`. Do not use it as promotion-grade
  `component_sensitivity_v1` evidence until official finite-difference
  component response validation, symmetric/directional response curves, and
  CUDA exact-eval custody are implemented and reviewed. CPU runs additionally
  require `--allow-diagnostic-cpu` and are non-promotable.
- Lightning diagnostic component-sensitivity validation must inspect every map,
  response curve, summary, input-preflight, and run-metadata artifact for
  `score_claim=false`, `promotion_eligible=false`, allowed
  `sensitivity_source`, non-official response status, and
  `canonical_scorer_path=false`. Direct renderer CUDA finite-difference maps
  may be `planning_eligible` and `certification_handoff_eligible`, but remain
  non-promotable until certified with official CUDA component-response
  evidence. Fisher/proxy maps are planning-only and never certification handoff
  eligible.
- Promotion-grade component maps must pass a separate certification stage; do
  not edit or strip diagnostic metadata from source maps. Use
  `experiments/certify_component_sensitivity_maps.py` to copy eligible CUDA
  direct finite-difference tensors into new `tac_score_sensitivity_map_v1`
  files with `component_sensitivity_map_certification_v1` metadata. The
  certification must cite source-map SHA, official response-curve SHA,
  stability SHA, sample-plan SHA, baseline archive SHA/bytes, baseline
  `contest_auth_eval.json` SHA, pre-response prediction-deltas SHA,
  archive-byte perturbation-basis SHA, response gate metrics, stability gate
  metrics, and at least three clean review passes. Fisher/proxy/debug/smoke/
  random maps are never certifiable.
- `experiments/build_component_sensitivity_manifest.py` promotion assembly
  must reject raw diagnostic maps and clean-but-uncertified maps. A promotable
  manifest may only reference certified maps plus official CUDA response curves
  with all promotion gates passed.
- Official component-response jobs that are given an external baseline
  `contest_auth_eval.json` must compare the same-run eps=0 baseline to that
  external JSON. External-baseline component drift is a runner/scorer
  calibration failure and blocks promotion even if same-run zero reproduces
  internally. Local component-response artifact validation and map
  certification must reject or de-promote curves that omit or fail
  `gate_results.external_baseline_repro` when an external baseline was
  supplied.
- Component sensitivity sample plans must identify absolute dataset pair IDs.
  If top-k pair weighting selects a subset, do not record subset-relative
  offsets in calibration/holdout records.
- Fake, random, dummy, CPU, MPS, smoke, debug, proxy-only, or no-holdout
  sensitivity artifacts are non-promotable. They may guide debugging only.
- Official component-response promotion plans must carry pre-response
  prediction deltas from
  `experiments/build_component_response_prediction_deltas.py` in
  `official_component_response_prediction_deltas_v1` format. Ad hoc epsilon
  maps, post-hoc observed-response deltas, copied scorer JSON, or any payload
  containing response-curve/eval leakage are non-promotable and must fail
  closed when `--require-predicted-deltas` is set.
- `experiments/profile_component_sensitivity_official.py --require-passed`
  must use a same-run eps=0 baseline. External baseline JSON may be retained
  as archive custody only; it cannot satisfy zero-repro gates or absorb
  runtime/scorer drift.
- Promotable component-response curves must include explicit finite
  `gate_results` with coverage, same-run zero repro, signal, prediction-error,
  and promotion gates all exactly true, and no nested promotion blockers.
  `experiments/build_component_sensitivity_manifest.py` must preserve those
  gates and `src/tac/component_sensitivity_artifact.py` must reject missing or
  false gates.
- OWV3 Fisher profiling for promotion must include protected Conv2d weights
  with `--include-protected-conv2d`; protected Linear FiLM parameters remain
  excluded unless a new reviewed converter supports them.
- OWV3 Fisher conversion must use `--missing-policy error`. Any protected or
  nonprotected missing Conv2d sensitivity key blocks promotion. The legacy
  protected-missing fallback is smoke/debug only.
- Do not spend exact eval on an OWV3 archive that is larger than the PFP16 A++
  byte frontier unless an exact distortion-reduction justification and review
  tag are present.

## Neural Weight Codec Lanes

- J-NWC and J-NWCS artifacts must be loadable renderer formats, not ad hoc
  concatenated tensor blobs. File magic, schema version, JSON header,
  embedded codec state or explicit loader contract, tensor metadata,
  length-prefixed blobs, and inflate dispatch must be tested end to end before
  promotion.
- NWC/NWCS training seed contracts include codec construction. Set the torch
  seed before constructing codec modules, not only inside the training sampler.
- Corpus manifests must be deterministic and relocatable. Replay may use a
  manifest-relative root, but must always recheck file size, SHA-256, tensor
  shape, dtype, block count, and ordering.
- NWCS sensitivity artifacts must be provenance-anchored for promotion:
  anchor archive SHA-256, anchor renderer SHA-256, corpus manifest SHA-256,
  block size, parameter names, shapes, block counts, and nonnegative finite
  values. Raw shape-only sensitivity dictionaries are debug-only.
- J-NWC/NWCS remote scripts must use zip-slip-safe archive reads, reject hidden
  sidecars, duplicates, absolute paths, traversal, and unexpected members, and
  record SHA-256/bytes for every custody artifact.
- J-NWC/NWCS exact CUDA paths must run `scripts/adjudicate_contest_auth_eval.py`
  after `contest_auth_eval.py`, preserve adjudication provenance and
  adjudicated JSON, and configure component gates against the active frontier.
  Build-only/debug paths must stop before auth eval with `score_claim=false`,
  `promotion_eligible=false`, `auth_eval_skipped=true`, and `result_json=null`.
- NWCS `NWCS1` export heredocs must import and use
  `_infer_asymmetric_config` in the same Python process that writes the
  container metadata. A fallback to `{"tensor_only": true}` is non-promotable.
- `AUTH_EVAL_DEVICE` must be `cuda` for promotable J-NWC/NWCS runs. CPU/MPS
  overrides must fail closed or mark the run explicitly non-promotable before
  any result can be harvested.

## Loss And Training Guardrails

- Primary KL distillation is forbidden for promotion paths unless explicitly
  fenced as forensic. SegNet-only auxiliary KL must be explicitly scoped,
  temperature-plumbed, and promotion-gated by exact PoseNet non-collapse.
- Renderer-training KL/JBL auxiliaries must never activate from
  `kl_distill_weight` alone. Positive `kl_distill_weight` requires
  `kl_distill_scope="segnet_aux"` in CLI/profile, and `primary_scorer` scope
  is blocked in `train_renderer`.
- Legacy `loss_mode="segnet_kl"` is forensic/debug only unless separately
  revalidated; it must set `kl_distill_scope="segnet_aux"` and
  `promotion_eligible=False`.
- `loss_mode="kl_distill"` is never self-explanatory. Promotion-capable KL
  configs must set `kl_distill_scope="segnet_aux"` and record
  `kl_distill_weight`, `kl_distill_temperature`, `eval_roundtrip`, and exact
  component gates. `kl_distill_scope="primary_scorer"` is forensic-only,
  requires `allow_banned_primary_kl_distill=True` and
  `promotion_eligible=False`, and must not be routed through SegMapTrainer.
- Retired/adversarially invalidated formulas or adaptive schemes must remain
  disabled unless a new proof and exact evidence reopens them.
- Loss weights, units, reductions, and temperature factors must be derived or
  empirically justified. No arbitrary constants in promoted lanes.
- Scorer-sensitive objectives need exact post-training archive eval; proxy
  losses do not promote.
- Bad training results are suspected engineering/config/math issues until
  reviewed.
- Component sensitivity promotion requires official CUDA finite-difference
  component response, full 600-pair sample coverage, response-curve gates,
  stability gates, exact contest eval custody, and
  `component_sensitivity_v1` manifest validation. Fisher-proxy sensitivity
  maps are diagnostic only. Use
  `experiments/profile_component_sensitivity.py --promotion-finite-difference`
  only for the promotion path; the default profiler mode must remain
  non-promotable.

## Harness And DX Hardening

- Fix bug classes, not just individual bugs.
- Prefer fail-closed behavior for promotion paths.
- Add preflight checks for known meta-bugs: stale CLI guidance, regex score
  parsing, nondeterministic archives, sidecars, duplicate dispatches, stale
  trackers, non-CUDA evals, and hidden fallbacks.
- Remote scripts must use strict shell mode, deterministic packaging,
  lane-local JSON adjudication, heartbeat/provenance logs, and explicit
  hardware recording.
- Avoid duplicate dispatches: use locks and live-prefix checks.
- Keep all warnings and low-severity DX issues on the hardening backlog.
- MCP servers are disabled for this project unless explicitly re-enabled by the
  user; do not depend on MCP tools for routine work.
- MCP is globally disabled for this operator environment as of 2026-05-01:
  known MCP config files, plugin caches, tool-output caches, OAuth/state files,
  and `.playwright-mcp` artifacts have been removed from the local tool homes.
  Do not recreate, install, sync, or enable MCP server/plugin state in Claude,
  Cursor, Gemini, LM Studio, Codex, or project-local config.
- Repo-owned MCP config files must have empty `mcpServers` objects and no
  active `mcp_servers` TOML sections; preflight blocks accidental reactivation.
- If MCP helper processes respawn from an outer app/runtime, kill only the exact
  MCP command patterns and continue without relying on those integrations.
- Already-running MCP helper processes are a preflight failure during
  contest/eval work. Run `check_no_live_mcp_processes(strict=True)` after
  cleanup and before trusting the local execution environment.
- Use `scripts/kill_orphaned_mcp_processes.py --strict` to clean the known MCP
  helper process class (`chrome-devtools-mcp`, `rbx-studio-mcp`,
  `roblox_studio_mcp`, and `model.context`). Prefer this over broad process
  killing. If helpers keep respawning, continue killing exact matches and treat
  the supervisor as external noise unless the user explicitly re-enables MCP.
- MCP cleanup/preflight must distinguish live helpers from audit commands that
  merely mention MCP tokens. Keep regression coverage so `find`, `rg`, `grep`,
  shell audit commands, and Python one-liners containing these strings are not
  killed or reported as live MCP helpers, while direct binaries, `npm exec`,
  `npx`/package launchers, shell-wrapped launches, and `python -m
  model.context` remain blocked.
- Lightning source manifests are part of promotion custody. Any manifest used
  to submit exact-eval or component-response jobs must fail closed on absolute
  paths, `..` traversal, duplicate entries, empty separators, backslashes,
  control characters, hidden files, macOS resource forks, and `__MACOSX`
  entries. Queue metadata that names a baseline JSON must be covered by the
  staged source manifest.
- Lightning SSH harvest targets must be state-derived where possible and must
  use a configured SSH alias or user-qualified target. Not bare ssh.lightning.ai —
  preflight `check_lightning_ssh_static_policy` is FATAL on regressions to the
  bare provider host (it is not acceptable for reproducible artifact custody).
- Lightning SSH commands used by repo wrappers must set noninteractive auth,
  a finite `ConnectTimeout`, client keepalives (`ServerAliveInterval` and
  `ServerAliveCountMax`), `TCPKeepAlive=yes`, and bounded
  `ConnectionAttempts`. Transient provider key-exchange resets such as
  `kex_exchange_identification` or `Connection reset by peer` should be retried
  only as transport failures; public-key auth failures, disabled host-key
  checking, and supply-chain scan failures must still fail closed.
- Modal component-sensitivity fallback work must use the dedicated lightweight
  direct-FD shard launcher, keep the same deterministic shard topology as the
  Lightning wave it backs up, and remain diagnostic/non-promotable until exact
  CUDA archive custody and official response gates are satisfied. Do not use
  broad Modal training mounts for this path when only the PFP16 archive,
  source, and upstream scorer assets are needed.
- J-NWC/NWCS promotion paths must use exact corpus-manifest custody. When a
  prebuilt corpus manifest or `CORPUS_SENSITIVITY_PT` is involved, remote
  scripts must receive the matching `PREBUILT_CORPUS_MANIFEST` and
  `CORPUS_REPLAY_ROOT`, preserve manifest bytes/SHA, and fail closed on
  missing or mismatched replay custody.

## Verification Before Deployment

Before treating any change as deployable:

```bash
.venv/bin/python -m py_compile <touched python files>
.venv/bin/python -m pytest <focused tests> -q
bash -n <touched shell scripts>
git diff --check
```

For score-affecting changes, also require:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

The final deploy packet must include archive, SHA, manifest, JSON, logs,
hardware provenance, command line, source/staged-tree manifest, upstream hash,
and adversarial review status.

## Agent Workflow

- Read the relevant source docs and current progress ledgers before acting.
- Use subagents when explicitly authorized for parallel research, audit, or
  implementation. Give them disjoint scopes and require file paths changed or
  read-only output.
- Use online research liberally for current papers, OSS, Lightning AI tooling,
  entropy coders, learned compression, and optimization methods, but never use
  external results as contest evidence.
- Record progress in adjacent dated markdown docs and persistent memory.
- End substantial turns with work landed, ongoing work, roadmap, and next
  steps needed to keep pushing toward the floor.
- Never revert unrelated user/agent changes in a dirty worktree.
- If the work is blocked on user action and the user is not present, use the
  installed `imsg` CLI for brief escalation. Use only when intervention is
  genuinely required to unblock an active experiment or security issue; include
  the run/job name, the blocker, and the exact requested action. Do not store
  private phone numbers, emails, or chat IDs in source; pass them via local
  shell history, environment, or the operator's active command context.
