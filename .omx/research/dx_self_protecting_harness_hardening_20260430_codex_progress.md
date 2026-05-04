# DX Self-Protecting Harness Hardening - Codex Progress - 2026-04-30

This note is adjacent to the Grand Council source-of-truth planning docs and
records harness/DX hardening that protects contest-grade evidence generation.

## Source-Of-Truth Cross-References

- Grand Council paradigm-shift design:
  `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- Adversarial reviews:
  `.omx/research/council_paradigm_shift_round{1,2,3}_20260430.md`
- Codex execution progress:
  `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- Contest-grade audit progress:
  `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
- Readiness progress:
  `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`

## Incident

An interrupted SegMap-clone retry dispatch produced duplicate/partial Vast
state:

- `35905846`: duplicate empty instance, destroyed.
- `35905118`: staged repo, no lane; launch reached setup but failed NVDEC and
  auto-destroyed.
- Final clean dispatch: `35906669`,
  `lane_sa_segmap_clone_2026-04-30_codex_a2`.

This is a DX correctness bug class because orchestration state controls spend,
evidence provenance, and which artifacts are safe to harvest.

## Permanent Fixes Landed

- `scripts/launch_lane_with_retry.py`
  - Per-label advisory lock under `.omx/state/launch_locks/`.
  - Live Vast label-prefix guard before each new attempt.
  - Fail-closed `UNKNOWN_EXISTING_LABEL_PREFIX` when a matching live instance
    already exists or live-state cannot be verified.
  - Child phases use `start_new_session=True`.
  - Timeout/SIGINT/SIGTERM kill child process groups with `os.killpg`.
  - `phase2-launch` timeout remains `UNKNOWN_REMOTE_STATE`, not retry.

- `src/tac/preflight.py`
  - Added `check_launch_retry_wrapper_singleflight_and_signal_safe`.

- `src/tac/tests/test_remote_auth_eval_hardening.py`
  - Expanded to cover same-prefix duplicate refusal.
  - Expanded to cover stage timeout cleanup return behavior.

## Current Live Remote State

At the post-dispatch checkpoint:

- `35885106`: HM-S active.
- `35899850`: Lane 19 logit margin active.
- `35906669`: SegMap clone active, RTX 4090, `$0.2539/hr`,
  `root@ssh2.vast.ai:26668`.
- `35907873`: H-V3 active/setup-running, RTX 4090, `$0.2731/hr`,
  `root@ssh5.vast.ai:27872`.

SegMap clone launch proof:

- `SETUP_COMPLETE` present in `/workspace/setup.log`.
- Fresh heartbeat present at
  `/workspace/pact/lane_sa_segmap_clone_results/heartbeat.log`.
- `run.log` reached Stage 2 training.

H-V3 launch proof:

- Attempts 1/2 hit slow SSH/readiness and were retired.
- Attempt 3 failed NVDEC and auto-destroyed.
- Attempt 4 `35907873` launched through the hardened wrapper.
- Remote setup passed lightweight NVDEC pre-probe and reached Stage 3
  `nvidia-dali-cuda120` install; at the checkpoint it had not yet reached
  `SETUP_COMPLETE` or lane training.

## Verification

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile`: clean for `scripts/launch_lane_with_retry.py`,
  `src/tac/preflight.py`, and `scripts/adjudicate_contest_auth_eval.py`.
- `check_launch_retry_wrapper_singleflight_and_signal_safe`: 0 violations.
- `check_remote_lane_auth_eval_json_adjudication`: 0 violations.
- `git diff --check`: clean.

## Remaining DX Hardening Targets

- Build a watcher that harvests only lane-local `contest_auth_eval.json` and
  exact archive/provenance bundles.
- Add a stale active-dispatch reconciler that compares
  `.omx/state/active_dispatches.md`, `.omx/state/vastai_active_instances.json`,
  and live `vastai show instances --raw`.
- Add remote heartbeat grading that distinguishes setup, training, auth-eval,
  crashed, and harvest-ready states without relying on human logs.
- Add T4/equivalent promotion runner for exact PFP16 archive SHA.
- Keep Q-FAITHFUL gated until KL-distill-like risk is removed or exact CUDA
  evidence proves it does not collapse PoseNet.

---

## Update - 2026-04-30T16:16Z Reconciliation And Process Hygiene

New permanent DX guardrail:

- Added `scripts/reconcile_vast_dispatch_state.py`.
- Added `src/tac/tests/test_reconcile_vast_dispatch_state.py`.
- The reconciler compares live `vastai show instances --raw`,
  `.omx/state/vastai_active_instances.json`, and
  `.omx/state/active_dispatches.md`.
- It reports stale tracker entries, active-dispatch records missing from live
  Vast state, live instances missing from the tracker, and normalized label
  prefix drift without mutating state.

Observed drift at 2026-04-30T16:16Z:

- `live=4`, `tracker=204`, `active_dispatches=3`.
- `tracker_missing_live=200`; the JSON tracker is massively stale.
- `active_missing_live=3`; stale records remain for Lane 19 `_a1`, Lane 8, and
  Lane 17.
- `live_missing_active=3`; HM-S, SA, and H-V3 are live but absent from the
  active-dispatch ledger.
- Conclusion: live Vast API plus lane-local artifacts must override state
  ledgers until a non-destructive prune/update workflow is added.

Process hygiene:

- PPID=1 orphan MCP processes were killed.
- A follow-up PPID=1 MCP scan is clean.
- Non-orphan MCP children owned by active parent processes were left alone.

KL bug-class hardening:

- Primary KL distill is now fenced as forensic and non-promotable unless
  explicitly scoped.
- SegNet-only KL auxiliary plumbing now carries explicit temperature/scope.
- Focused KL/config tests passed in the subagent report.

Current remaining DX targets:

- Add a non-destructive state-prune/update command for the Vast tracker.
- Add remote heartbeat classification that emits structured setup/training/eval
  states instead of relying on human log tails.
- Add Lightning as the preferred exact-eval runner and keep Modal as a
  controlled non-promotion acceleration backend.

Verification for this pass:

- Focused suite:
  `src/tac/tests/test_pfp16_a_plus_plus_helper.py`,
  `src/tac/tests/test_reconcile_vast_dispatch_state.py`,
  `src/tac/tests/test_remote_auth_eval_hardening.py`,
  `src/tac/tests/test_remote_lane_g_v3_owv3_fisher_stack_script.py`,
  `src/tac/tests/test_config_validation.py`,
  `src/tac/tests/test_kl_distill_weight_plumbed.py`,
  `src/tac/tests/test_losses.py`,
  `src/tac/tests/test_training.py::TestFitLossModeGuard`,
  `src/tac/tests/test_preflight_meta_bugs.py::TestKlDivReductionCorrect`,
  `src/tac/tests/test_segmap_renderer.py`,
  `src/tac/tests/test_lane12_nerv_dependency_closure.py`, and
  `src/tac/tests/test_nerv_mask_codec.py`.
- Result: `118 passed in 5.58s`.
- `bash -n` clean for PFP16 A++ helper, OWV3/Fisher remote script, and Lane 12
  NeRV remote script.
- `py_compile` clean for touched Python scripts/modules.
- `git diff --check`: clean.

---

## Update - 2026-04-30T16:25Z Lightning And Modal Backend Policy

Lightning:

- Verified usable for exact T4 CUDA eval.
- PFP16 A++ run succeeded with the helper and produced local artifacts under
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/`.
- Operational lesson: use hermetic staged remote trees for exact evidence;
  the default Lightning `/home/zeus/content/pact` tree may be stale and should
  not be mutated blindly.

Modal:

- Modal CLI/auth are configured locally (`modal` client `1.4.1`, profile
  `adpena`).
- Current Modal wrappers are not promotion-grade because
  `experiments/modal_train_lane.py` forces `AUTH_EVAL_DEVICE=cpu`.
- Modal is approved for supplementary build/smoke/Fisher/ablation work whose
  outputs later move to Lightning exact CUDA eval for promotion.

---

## Update - 2026-04-30T16:40Z MCP Shutdown And Config Removal

User directed that no MCP servers are in use and all MCP processes/configs
should be disabled.

Actions landed:

- Killed all visible `chrome-devtools-mcp`, `rbx-studio-mcp`,
  `roblox_studio_mcp`, and `model.context` processes.
- Verified no matching MCP processes remain in `ps`.
- Removed active MCP server entries from:
  - `/Users/adpena/.codex/config.toml`
  - `/Users/adpena/.claude/mcp.json`
  - `/Users/adpena/.cursor/mcp.json`
  - `/Users/adpena/Library/Application Support/Claude/claude_desktop_config.json`
- Left JSON configs valid with empty `mcpServers` objects where applicable.
- Backup files were written with timestamp `20260430T163944Z` beside each
  edited config.

Validation:

- `python3 -m json.tool` passed for the edited JSON configs.
- `rg` over the edited active config files shows no active `mcp_servers` TOML
  sections and only empty `mcpServers` JSON objects.

---

## Update - 2026-04-30T17:00Z Scoped Regression Vocabulary Hardening

Grand Council kill-discipline audit found that the policy was correct but
some live helpers still emitted broad `hard-kill` vocabulary.

Actions landed:

- `scripts/adjudicate_contest_auth_eval.py` now accepts
  `--regression-threshold` and emits `regression_triggered`,
  `regression_threshold`, `regression_scope`, and
  `REGRESSION_REVIEW_REQUIRED`.
- The deprecated `--hard-kill-above` direct-call path remains only as a
  backward-compatible alias.
- Future provenance no longer writes `hard_kill_triggered` by default.
- Active remote scripts that call the adjudicator now pass
  `--regression-threshold`.
- H-V3 and Lane 12/NeRV comments now describe run-abort thresholds rather than
  scientific kills.

Rigor implication:

- A bad exact archive can retire the measured implementation/config only.
  Family/method kills still require independent exact evidence or a
  mathematical impossibility argument plus clean Grand Council consensus.

---

## Update - 2026-04-30T17:05Z CUDA Auth Eval Source-Of-Truth Pin

User re-emphasized that MPS/local paths materially distort score and auth-eval
signal. This is now pinned in `AGENTS.md`:

- MPS, CPU, local proxy scorers, and non-canonical renderer checks are
  development-only.
- For GPU-dependent score or signal claims, the only reliable source of truth
  is exact CUDA auth eval on the exact archive bytes:
  `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- `experiments/contest_auth_eval.py --device cuda` and its
  `contest_auth_eval.json` are the canonical path/artifact.
- Local/MPS evidence must never promote, rank, kill, retire, validate a stack,
  or anchor paper claims. If local/MPS disagrees with CUDA auth eval, CUDA wins.

Verification for this greenup:

- Focused regression suite covering auth-eval hardening, Lightning harvesting,
  OWV3 fail-closed size guard, contest auth eval, PFP16 A++ helper, Vast
  reconcile, and GP smooth-basis retirement guard: `70 passed in 1.71s`.
- `py_compile` clean for touched Python adjudicator/Lightning/OWV3/PFP16/GP
  files.
- `bash -n` clean for touched adjudicator-calling remote scripts.
- `git diff --check` clean.

---

## Update - 2026-05-01 Modal CPU Auth-Eval Advisory Guard

Bug class hardened:

- Modal training wrappers can force `AUTH_EVAL_DEVICE=cpu` to avoid NVDEC/DALI,
  producing useful diagnostic telemetry that must never be mistaken for CUDA
  auth-eval evidence.

Actions landed:

- Added `check_modal_cpu_auth_eval_is_advisory_only` to `src/tac/preflight.py`
  and wired it into codebase preflight.
- Removed stale Modal CPU score-truth wording from
  `experiments/modal_train_lane.py`.
- Added explicit Modal metadata/env markers:
  `MODAL_AUTH_EVAL_ADVISORY_ONLY=1`, `SCORE_CLAIM=false`,
  `PROMOTION_ELIGIBLE=false`, `auth_eval_advisory_only=true`,
  `score_claim=false`, and `promotion_eligible=false`.
- Updated `experiments/modal_recover_lane.py` so recovered non-CUDA auth scores
  print as `ADVISORY AUTH SCORE (NON-PROMOTABLE, device=...)` and prefer
  `score_recomputed_from_components`.
- Added regression tests proving stale CPU-equivalence wording is blocked and
  CPU/CUDA recovery labels differ.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 33 passed.
- Focused Modal advisory tests: 4 passed.
- `py_compile` clean for touched Python files.
- `git diff --check` clean for touched files.

---

## Update - 2026-05-01T06:01Z Lightning Negative-Value CLI And Sensitivity Harvest Guards

Bug classes hardened:

- Generated remote commands that pass option values beginning with `-` can fail
  under `argparse` when emitted as `--flag value`; the safe form is
  `--flag=value`.
- Diagnostic component-sensitivity artifacts need the same state-derived
  harvest discipline as exact eval and official component-response artifacts,
  otherwise operators can hand-compose stale `/teamspace/jobs/...` paths or
  copy bulky raw outputs.

Actions landed:

- `src/tac/deploy/lightning/batch_jobs.py` now emits response epsilon ladders
  as `--response-epsilons=<value>`.
- Added local and SSH diagnostic component-sensitivity harvest/validation
  helpers and CLI subcommands:
  `validate-component-sensitivity-artifacts`,
  `harvest-component-sensitivity-local`, and
  `harvest-component-sensitivity-ssh`.
- `AGENTS.md` now records the permanent equals-form rule for negative-valued
  generated CLI options and the state-derived diagnostic sensitivity harvest
  rule.
- J-NWC/NWCS test fixtures were updated to the stricter
  `component_sensitivity_v1` gate schema so tests fail closed with the current
  promotion validator.

Verification:

- `py_compile` clean for touched sensitivity, Lightning, and NWC/NWCS files.
- J-NWC/J-NWCS shell scripts: `bash -n` clean.
- Focused pytest suite: `152 passed in 3.99s`.
- `git diff --check` and `git diff --cached --check` clean.
- `scripts/kill_orphaned_mcp_processes.py --strict --json` reported zero
  matched or remaining MCP helper processes.
- Local Lightning supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_local_20260501T0601Z_codex.json`,
  strict `status=OK`, `violation_count=0`, with no installed `lightning` or
  `pytorch-lightning` package and `lightning-sdk==2026.4.10`.

---

## Update - 2026-05-01T06:17Z Post-Harvest Response Planning Guard

Bug class hardened:

- Diagnostic sensitivity harvests previously required hand-wiring map files,
  perturbation basis JSON, prediction deltas, archive variants, and official
  response plans. That creates room for stale/post-hoc basis reuse and
  operator path mistakes.

Actions landed:

- Added `experiments/build_component_response_plan_from_sensitivity_artifacts.py`.
  It validates the harvested diagnostic sensitivity artifact directory,
  optionally consumes a fresh explicit `perturbation_basis_v1`, builds
  `official_component_response_prediction_deltas_v1`, and emits the official
  response plan with `score_claim=false` and `promotion_eligible=false`.
- `AGENTS.md` now records this script as the deterministic post-harvest bridge
  before official CUDA component-response dispatch.
- Added regression tests for artifact-dir validation, fresh-basis override,
  prediction semantics propagation, and CLI import/help.

Verification:

- New focused test file path: `14 passed`.
- Expanded focused suite after the first wrapper version: `154 passed`.
- MCP strict cleanup remained zero-live-process.

---

## Update - 2026-05-01T06:37Z Certified Maps And External Baseline Repro Gate

Bug classes hardened:

- Diagnostic CUDA Fisher/proxy maps could be mistaken for promotion-grade
  component sensitivity if a later tool copied tensors while ignoring source
  metadata.
- Official component-response runs with same-run eps=0 could pass internal
  zero reproduction while drifting from a supplied external baseline
  `contest_auth_eval.json`.

Actions landed:

- Added `experiments/certify_component_sensitivity_maps.py`, which certifies
  only eligible CUDA direct finite-difference maps after official response
  curves, stability, sample coverage, baseline custody, and review passes are
  verified.
- `experiments/build_component_sensitivity_manifest.py` now rejects diagnostic
  maps and clean-but-uncertified maps for promotion assembly.
- `src/tac/component_sensitivity_artifact.py` validates
  `component_maps.*.certification` and rejects false external-baseline repro
  gates when present.
- `experiments/profile_component_sensitivity_official.py` now records
  `external_baseline_repro` and blocks promotion when same-run eps=0 does not
  reproduce the supplied baseline JSON.
- `AGENTS.md` now records the certification protocol and external-baseline
  repro requirement.

Verification:

- `py_compile` clean for touched certification and response files.
- Certification/manifest/sensitivity suite: `64 passed`.
- Official-response/certifier/component-artifact suite: `58 passed`.
- Alpha-Geo primitive-contract suite from worker integration: `19 passed`.
- `git diff --check` clean for touched files.
- MCP strict cleanup reports zero live helper processes.

---

## Update - 2026-05-01T06:55Z Lightning Status, Certifier Custody, Alpha Contract DX

Bug classes hardened:

- Lightning SDK status can be non-monotonic or name-only; accepting
  `Running -> Pending` as ordinary truth hides requeue/stale-read/name-collision
  hazards.
- Certified sensitivity maps could otherwise cite official curves without
  mandatory prediction-delta and perturbation-basis custody.
- Lane 12 NeRV training could still use direct SegNet targets unless the
  decoded-baseline Alpha contract path was enforced.

Actions landed:

- `src/tac/deploy/lightning/batch_jobs.py` now records full refresh snapshots,
  status anomalies, name-only identity confidence, and reconciliation-required
  state. Nonterminal regressions on non-dry-run exact/component-response/
  sensitivity jobs fail closed unless superseded by a terminal SDK status.
- `scripts/launch_lightning_batch_job.py refresh-status --all --fail-on-error`
  treats status reconciliation requirements as failures.
- `validate_local_component_response_artifact_dir` now de-promotes official
  response packets that cite an external baseline but whose curves omit or fail
  `gate_results.external_baseline_repro`.
- `experiments/certify_component_sensitivity_maps.py` now requires
  `--prediction-deltas-json` and `--perturbation-basis-json`, validates formats,
  cross-checks atom IDs and epsilon ladders, and verifies curve perturbation
  custody against those SHAs.
- `experiments/train_nerv_mask.py` and `src/tac/nerv_mask_codec.py` now enforce
  Alpha primitive-contract consumption and deterministic weighted sampling for
  decoded-baseline NeRV training; SegNet target training is forensic only.

Verification:

```text
src/tac/tests/test_lightning_batch_jobs.py: 69 passed
src/tac/tests/test_certify_component_sensitivity_maps.py: 4 passed
Consolidated focused suite: 203 passed in 5.79s
bash -n scripts/remote_lane_nerv.sh: clean
git diff --check on touched files: clean
MCP strict cleanup: zero live helper processes
```

---

## Update - 2026-05-01T07:10Z Direct-FD Harvest And Alpha Remote Contract Closure

Bug classes hardened:

- Diagnostic direct finite-difference sensitivity packets could be rejected at
  harvest as non-Fisher even though they are the required certification
  handoff source.
- Conversely, a diagnostic sensitivity packet could have clean summary/curve
  JSON while map `.pt` metadata smuggled score/promotion claims or a mismatched
  source.
- Remote Lane 12 decoded-baseline dispatch could omit
  `ALPHA_PRIMITIVE_CONTRACT` and silently fall back to unweighted/non-contract
  NeRV training even though the trainer itself had a production gate.

Actions landed:

- `scripts/launch_lightning_batch_job.py component-sensitivity` and
  `src/tac/deploy/lightning/batch_jobs.py` now expose/pass
  `--promotion-finite-difference` and `--finite-difference-epsilon` for
  direct renderer CUDA finite-difference sensitivity runs.
- Local and remote diagnostic component-sensitivity validators now whitelist
  only `fisher_proxy` and
  `direct_renderer_cuda_finite_difference_component_response`, enforce
  non-score/non-promotable status on inputs, run metadata, summaries, maps, and
  curves, and return `planning_eligible=true` with
  `certification_handoff_eligible=true` only for direct-FD maps.
- `experiments/profile_component_sensitivity.py` now writes
  `score_claim=false` into diagnostic summaries and response curves so harvest
  validation can enforce the same invariant across all artifacts.
- `scripts/remote_lane_nerv.sh` now records `alpha_primitive_contract`
  metadata in lane provenance, fail-closes decoded-baseline dispatch when the
  contract is missing or invalid, and forwards
  `--alpha-primitive-contract "$ALPHA_PRIMITIVE_CONTRACT"` into
  `experiments/train_nerv_mask.py`.

Verification:

```text
py_compile: clean for touched Python files
bash -n scripts/remote_lane_nerv.sh: clean
src/tac/tests/test_lightning_batch_jobs.py + test_lane12_nerv_dependency_closure.py: 97 passed
git diff --check on touched files: clean
MCP strict cleanup: zero live helper processes
```

---

## Update - 2026-05-02T02:25Z Component-Trace Feedback Parity And Compiler-Loop Protocol

Bug classes hardened:

- Standalone `experiments/contest_component_trace.py` could generate hard-pair
  and water-fill profile feedback through a runtime that did not match
  archive-only exact eval: system `ffmpeg` could lack the explicit inflate
  color-contract options, and inflate-side `uv run` could mutate the shared
  repo environment.
- Compiler-style positive feedback loops were an operator concept, not a
  durable protocol. That risked treating profile feedback as score truth or
  losing the provenance needed to turn profile-guided optimization into a
  reproducible scientific artifact.

Actions landed:

- `src/tac/preflight.py` now includes strict
  `check_contest_component_trace_runtime_parity`, wired into
  `preflight_all()`. It blocks regressions unless component tracing keeps a
  parity `FFMPEG_BIN` resolver, explicit color-contract option checks,
  job-local `UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE=copy`, runtime sidecar
  emission, non-promotable evidence status, and contest-auth cross-check
  support.
- `src/tac/tests/test_contest_component_trace.py` now verifies the real repo
  passes the new guard and that a trace script with removed parity/runtime
  guards is rejected.
- `AGENTS.md` now has a durable Closed-Loop Compiler-Style Optimization
  protocol: every stage should emit typed profile artifacts, exact CUDA archive
  evidence remains the only promotion truth, and synergies/antagonisms must be
  validated by exact stacked archives rather than additive assumptions.
- `AGENTS.md` also records the component-trace runtime parity rule alongside
  remote archive-only eval parity, so hard-pair/component traces cannot poison
  optimization with mismatched runtime feedback.

Verification:

```text
py_compile: clean for src/tac/preflight.py, experiments/contest_component_trace.py,
  submissions/robust_current/apply_qzs3_postprocess.py,
  experiments/build_qzs3_postprocess_candidate.py
focused pytest: 16 passed
  src/tac/tests/test_contest_component_trace.py
  src/tac/tests/test_qzs3_postprocess_candidate.py
  src/tac/tests/test_remote_archive_only_eval_script.py
git diff --check on touched files: clean
```

---

## Update - 2026-05-02T02:40Z Line-Search Scorer Runtime And DALI Preflight

Bug class hardened:

- The H100 PR67-informed active-subspace line-search failed before scoring
  first on missing scorer Python dependencies and then on missing
  `nvidia.dali`. This is a paid-wall-clock preflight bug, not lane evidence:
  `experiments/line_search_pose_refinement.py` imports upstream PoseNet and
  uses `DaliVideoDataset`, so the runner needs both scorer dependencies and
  the hash-pinned DALI wheel set before GPU work starts.

Actions landed:

- `experiments/line_search_pose_refinement.py` now includes `nvidia.dali` in
  `SCORER_RUNTIME_MODULES` and points the failure message at
  `scripts/bootstrap_dali_hash_pinned.py`.
- `src/tac/preflight.py` now has strict
  `check_line_search_scorer_runtime_preflight`, wired into `preflight_all()`,
  so GT-backed line-search tools must keep the scorer/DALI dependency guard.
- `src/tac/tests/test_line_search_pose.py` covers both the runtime helper and
  the static preflight guard.
- `AGENTS.md` now states that compress-time proposal tools touching
  `upstream/modules.py` or `DaliVideoDataset` must install the runtime extra
  and run the hash-pinned DALI bootstrap before paid remote work.

Remote mitigation:

- H100 runner `35985850` was bootstrapped with
  `nvidia-dali-cuda130==1.52.0` through
  `scripts/bootstrap_dali_hash_pinned.py`, recording
  `experiments/results/line_search_qzs3_qp1_pr67_active_subspace_c057_20260502T0230Z/dali_bootstrap.json`.
  The active-subspace run was relaunched as
  `pact_ls_qzs3_pr67_active_subspace_c057_fix2_20260502T0240Z`.

---

## Update - 2026-05-02T03:26Z QZS Repacker Source-Contract And No-Op Guard

Bug classes hardened:

- Frontier archives may already be deployed as a single charged runtime blob
  (`p` or `renderer_payload.bin*`). A repacker that only reads exploded runtime
  members cannot operate on the actual frontier archive and tempts operators
  into untracked manual extraction.
- Existing QZS3 renderer streams can be byte-screened with a new
  `--qzs3-block-size` while accidentally reusing the source stream unchanged.
  This produces duplicate archives that look like format experiments but are
  actually no-op controls.

Actions landed:

- `experiments/repack_quantizr_faithful_qzs3_archive.py` now treats packed
  single-blob archives as a first-class source contract by unpacking them
  through `submissions.robust_current.unpack_renderer_payload` in a temporary
  directory and recording `source_runtime_contract` provenance.
- The QZS3 repack path now decodes and re-encodes existing QZS3 payloads when
  the requested block size differs from the source block size. Provenance
  records source block size, requested block size, and action.
- `src/tac/tests/test_qzs3_packer.py` covers both guardrails: packed-frontier
  source ingestion and block-size re-encode parity for QZS3 sources.

Verification:

```text
focused pytest: 2 passed
  src/tac/tests/test_qzs3_packer.py::test_repack_qzs3_block_size_reencodes_existing_qzs3_source
  src/tac/tests/test_qzs3_packer.py::test_repack_qzs4_accepts_existing_single_blob_frontier_archive
```

Permanent policy:

- Byte-screened format variants must prove that the knob under test changed the
  relevant payload contract, not only archive member ordering. When a source is
  already compressed in the same family, provenance must distinguish `reuse`
  from `decode_reencode`.

---

## Update - 2026-05-02T03:36Z Promotion Submit Guardrail Successes

Guardrails exercised:

- Lightning T4 exact-eval submission refused to launch until inflate-side Torch
  was pinned to a CUDA-12 compatible wheel:
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match`.
- The dispatch-claim guard refused a job-specific Lightning submit when only a
  broad lane claim existed. The active claim row was added manually because the
  helper enforces a single active row per lane while an H100 diagnostic sweep
  was still active.

Why this matters:

- Both blocks occurred before paid score work, converting known remote fragility
  into deterministic operator feedback. This is the intended self-protecting
  stack behavior: exact-eval custody fails closed before cloud spend or
  ambiguous evidence.

Follow-up hardening landed:

- `tools/claim_lane_dispatch.py` now supports controlled same-lane child
  claims with `--allow-parallel --child-of <active-job> --parallel-reason
  <why-disjoint>`. It only allows the claim when `--child-of` matches an active
  same-lane parent row.
- `src/tac/preflight.py` now requires the child-claim flags in the dispatch
  helper, so the regression becomes visible in strict preflight.
- `AGENTS.md` records the rule and forbids routine `--force` use for same-lane
  parallel work.

Verification:

```text
py_compile: clean for tools/claim_lane_dispatch.py and src/tac/preflight.py
focused pytest: 7 passed
  src/tac/tests/test_claim_lane_dispatch.py
git diff --check on touched files: clean
```

---

## Update - 2026-05-02T03:42Z Experiment Script Self-Rooting

Bug class hardened:

- `experiments/repack_quantizr_faithful_qzs3_archive.py` could be imported by
  tests but failed as a direct script from repo root because `sys.path[0]`
  became `experiments/`, hiding the repo package root. Operators had to know
  to set `PYTHONPATH=.` manually.

Action landed:

- The script now inserts the repo root into `sys.path` before importing
  `experiments.*` modules, matching how production-oriented lane tools should
  behave when launched directly from runbooks.
- `src/tac/tests/test_qzs3_packer.py` now asserts that
  `python experiments/repack_quantizr_faithful_qzs3_archive.py --help` works
  from the repo root and exposes `--qzs3-block-size`.

Verification:

```text
py_compile: clean for experiments/repack_quantizr_faithful_qzs3_archive.py
focused pytest: 1 passed
  src/tac/tests/test_qzs3_packer.py::test_repack_script_help_is_directly_executable_from_repo_root
```

---

## Update - 2026-05-02T03:50Z Lightning Status Regression Reconciled By Artifacts

Bug class exercised:

- Lightning SDK telemetry can regress nonterminal status after a job has
  already generated terminal artifacts. The C-059 T4 promotion stream reported
  `Running -> Pending` with cost accrued and was correctly marked
  `REMOTE_STATUS_RECONCILIATION_REQUIRED` in
  `.omx/state/qzs3_b32_maskfirst_qp1_fix1_t4_batch_jobs_20260502T0331Z.json`.

Protective behavior:

- No score claim was made from the SDK status. The state-derived artifact path
  was harvested with `scripts/launch_lightning_batch_job.py harvest-ssh`, then
  local validation checked archive bytes, SHA-256, adjudicated JSON, component
  trace, and artifact mirror integrity.
- The resulting C-059 row is A++ because terminal artifacts validate, not
  because the SDK status became clean.

Permanent rule:

- A nonterminal SDK regression blocks status-only promotion, but it does not
  discard terminal evidence. Promotion is allowed only after state-derived
  artifact harvest validates exact archive custody and canonical JSON locally.

---

## Update - 2026-05-02T04:30Z Long-Running Line-Search Telemetry Guard

Observed issue:

- The active H100 pair-window line search can spend several minutes inside a
  single basis-candidate objective call while `line_search.log` still shows
  only the baseline line. That is not scientific failure evidence, but it is an
  expensive observability metabug: operators cannot distinguish healthy
  long-running CUDA work from a stalled search without manual process/GPU
  checks.

Permanent fix:

- `experiments/line_search_pose_refinement.py` now emits
  `basis_candidate_start ...` progress telemetry before long basis/vector
  objective calls.
- The new `--progress-every-candidates` CLI option controls periodic telemetry
  during long vector stages.
- `src/tac/tests/test_line_search_pose.py` asserts the basis-progress surface
  exists in the smoke path.

Boundary:

- This does not change the search objective, candidate ordering, accepted
  archive bytes, or score evidence. It only makes paid wall-clock runs
  inspectable before they finish.

---

## Update - 2026-05-02T04:19Z Vast Fast-Chip Anchor-Custody Guard

Bug class hardened:

- `scripts/launch_lane_on_vastai.py` recorded `--anchor-dirs` in dispatch
  metadata, but the phase2 tarball builder did not consume those paths. A
  remote archive-only or line-search lane could therefore launch on a fast chip
  with source code present but without the exact archive/policy/evidence inputs
  the metadata claimed were shipped.

Action landed:

- `build_tarball()` now accepts explicit anchor paths and expands them into the
  positive `tar -T` file list before upload.
- Missing, absolute, parent-traversal, or backslash-containing anchors now fail
  closed before SCP/cloud spend.
- The old gzip append helper was removed because appending to `.tar.gz` is not
  portable and could silently continue without anchors.
- The existing `--prefer-fast-chip` Vast path remains opt-in and now has a
  custody-safe path for archive-bearing launches.

Verification:

```text
py_compile: clean for scripts/launch_lane_on_vastai.py,
  scripts/probe_fastest_chip.py, and the focused test module
focused pytest: 3 passed
  test_launcher_default_disk_is_60
  test_launcher_exposes_fast_chip_preference
  test_launcher_includes_explicit_anchor_dirs_in_tarball
git diff --check on touched launcher/test files: clean
```

---

## Update - 2026-05-02T04:29Z CMG1 Archive-Allowlist Closure

Bug class hardened:

- The CMG1 runtime and builder could emit a charged `masks.cmg1` payload, but
  the exact-eval archive validator and local smoke whitelist still rejected the
  suffix. This is the same class as the earlier AMR1 rejection: a valid charged
  runtime artifact becomes impossible to score because validator and runtime
  contracts drift.

Permanent fix:

- Added `.cmg1` to `experiments/contest_auth_eval.py` and
  `experiments/canonical_local_auth_eval_smoke.py`.
- Added focused guard coverage for a `renderer.bin + masks.cmg1 +
  optimized_poses.bin` archive member set.
- Extended the smoke/contest whitelist parity assertion so future suffix drift
  fails in local tests before remote GPU spend.

Verification:

```text
py_compile: clean for contest_auth_eval.py, canonical_local_auth_eval_smoke.py,
  test_runtime_guards_pass_3.py, and test_canonical_local_e2e_smoke.py
focused pytest: 15 passed
  src/tac/tests/test_runtime_guards_pass_3.py
  src/tac/tests/test_canonical_local_e2e_smoke.py::test_smoke_whitelist_parity_with_contest_auth_eval
```

---

## Update - 2026-05-02T05:05Z Dispatch Claim Terminal-Family Guard

Bug class hardened:

- The cross-agent claim helper treated custom statuses such as
  `completed_b32_only_survivor` as active because only a few literal
  completion statuses were terminal. That could block safe follow-up dispatches
  in the same lane after the scientific work had already closed.

Permanent fix:

- `tools/claim_lane_dispatch.py` now treats the `completed_` status family as
  terminal.
- Added a focused regression test so future custom completed statuses do not
  re-open the same block.
- Normalized the old local row to `completed_no_frontier` for clarity.

Verification:

```text
py_compile: clean for tools/claim_lane_dispatch.py and
  src/tac/tests/test_claim_lane_dispatch.py
focused pytest: 8 passed
  src/tac/tests/test_claim_lane_dispatch.py
```

---

## Update - 2026-05-02T05:16Z Lightning Source-Manifest Hidden-Path Guard

Bug class hardened:

- A deadline-pressure Lightning staging command used `--source .`, which
  allowed `.DS_Store` into the source manifest. The downstream exact-eval
  submitter correctly rejected the manifest before GPU spend because hidden
  and macOS resource-fork paths are not custody-safe.

Permanent fix:

- `scripts/lightning_repro_workspace.py` now excludes hidden path components,
  `__MACOSX`, AppleDouble `._*` files, and `.DS_Store` from source manifests
  even when staging from the repo root.
- Explicit `--artifact` paths still remain the only path for generated archives
  and result files, so promotion custody keeps source/artifact roles separate.

Verification:

```text
py_compile: clean for scripts/lightning_repro_workspace.py and
  src/tac/tests/test_lightning_repro_workspace.py
focused pytest: 17 passed
  src/tac/tests/test_lightning_repro_workspace.py
```

---

## Update - 2026-05-02T05:17Z Archive Hash Authority Over Stale Metadata

Bug class hardened:

- The weighted-pair top32 line-search metadata reported SHA
  `cda95e70440e9ef295985a042fda2d74715ef6a0e665a1c37871cddd051cd908`, while
  the harvested archive and exact H100 auth-eval provenance recorded SHA
  `877fc5ac13e9fbd5c4158a9c7fa9dec3354057522b086004a4a28c6822456fe8`.

Permanent rule:

- For promotion custody, `contest_auth_eval.json` archive provenance plus an
  actual archive byte/hash check supersede local search metadata. Search
  metadata is proposal telemetry unless it is independently re-hashed against
  the archive file being submitted.
- Active dispatch claims and T4 expected SHA/bytes were corrected to the
  harvested archive authority before Lightning submit.

Follow-up hardening target:

- Add a reusable post-search verifier that fails closed when `metadata.json`
  archive SHA/bytes disagree with the output `archive.zip`. This should become
  part of the line-search completion path, not an operator-memory rule.

---

## Update - 2026-05-02T05:19Z Remote Archive-Only Source-Coherence Guard

Bug class hardened:

- A remote archive eval can be scientifically invalid even when the archive
  bytes are correct if the remote runtime source is stale relative to the
  candidate contract. This occurred in the MQZ1 path when the candidate used
  compact `fp4_block_sizes` metadata but the remote decoder still expected the
  older verbose `fp4_tensors` schema.

Permanent fix:

- `scripts/remote_archive_only_eval.sh` now accepts
  `REQUIRED_SOURCE_SHA256S` as newline-separated `path=sha256` entries.
- Each path is validated as a safe repo-relative source file, then hashed
  remotely before exact CUDA work begins.
- Missing, malformed, unsafe, or mismatched entries fail closed before GPU
  spend.

Verification:

```text
bash -n scripts/remote_archive_only_eval.sh passed
focused pytest: included in 34 passed batch
  src/tac/tests/test_remote_archive_only_eval_script.py
```

Usage pattern:

```text
REQUIRED_SOURCE_SHA256S='src/tac/quantizr_qzs3_codec.py=<sha256>'
```

Use this for every archive whose runtime contract depends on freshly patched
source rather than only on stable contest runtime files.

---

## Update - 2026-05-02T05:38Z Deterministic Single-Payload Brotli Repack

Bug class / opportunity hardened:

- A quick local Brotli parameter sweep found a 124-byte lossless improvement
  for the C-059 single-member `p` archive, but ad hoc snippets are not durable
  enough for promotion custody.

Permanent fix:

- Added `experiments/repack_single_payload_brotli.py`.
- The tool accepts only safe one-member archives, decompresses the Brotli
  payload, recompresses with recorded parameters, verifies exact raw-payload
  round-trip, writes a deterministic stored ZIP, and emits custody JSON with
  source/output archive SHA, member SHA, raw SHA, byte deltas, and formula-only
  rate delta.
- Non-Brotli single-member archives now fail with a domain-specific
  `RepackError`; this prevents treating raw compact payloads as Brotli streams.

Verification:

```text
py_compile: clean
focused pytest: 3 passed
  src/tac/tests/test_repack_single_payload_brotli.py
```

Boundary:

- This is `empirical_lossless_byte_transform` only until exact CUDA auth eval
  validates the closed archive. It is not a substitute for the higher-EV
  pose/manifold and charged-grammar lanes.

---

## Update - 2026-05-02T06:45Z Public-Adapter Replay And Inflate Script Provenance

Bug class hardened:

- Public leaderboard archive replay needs the public PR's own inflate runtime
  or a deliberately labeled compatibility adapter. A hardened runner that
  silently falls back to `submissions/robust_current/inflate.sh` can produce
  misleading failures or non-compliant fallback behavior.
- The previous F5 `config.env` guard was too broad for direct public-adapter
  replays. It correctly protects renderer-dispatch inflates, but it should not
  reject a simple public `inflate.sh` that never reads `PYTHON_INFLATE` or
  `CONFIG_ENV_PATH`.

Permanent fixes:

- `scripts/remote_archive_only_eval.sh` now accepts an explicit `INFLATE_SH`
  override, resolves it safely inside the repo, hashes it, and passes the
  resolved path to `experiments/contest_auth_eval.py --inflate-sh`.
- Runtime provenance records both `inflate_sh` and `inflate_sh_sha256`.
- `experiments/contest_auth_eval.py` now scopes the F5 `config.env` guard to
  inflate scripts that actually dispatch through `PYTHON_INFLATE` or
  `CONFIG_ENV_PATH`; robust-current remains fail-closed.

Verification:

```text
bash -n scripts/remote_archive_only_eval.sh
py_compile: experiments/contest_auth_eval.py and focused tests
focused pytest: runtime guard and remote archive-only tests passed
```

Evidence boundary:

- Public-adapter exact CUDA replays are `external` reverse-engineering signal,
  not our contest evidence. A score claim still requires our own fixed
  submission runtime, charged payload closure, and exact archive custody.

---

## Update - 2026-05-02T06:48Z Non-Login Remote uv PATH Guard

Bug class hardened:

- Vast/SSH non-login shells can omit `$HOME/.local/bin` even when the standard
  `uv` installer succeeded. Direct archive/component-trace tooling then fails
  before inflate with "`uv` is not on PATH" despite the binary existing at
  `/root/.local/bin/uv`.

Permanent fix:

- `experiments/contest_auth_eval.py::_ensure_uv_available()` now checks the
  standard installer locations (`Path.home()/.local/bin` and
  `/root/.local/bin`), prepends the discovered directory to `PATH`, and only
  raises if no executable `uv` is present.
- `experiments/contest_component_trace.py` inherits this guard through the
  canonical auth-eval helper.

Verification:

```text
py_compile: experiments/contest_auth_eval.py and runtime guard tests
focused pytest: src/tac/tests/test_runtime_guards_pass_3.py -> 17 passed
```

---

## Update - 2026-05-02T06:50Z Atom Ledger Active-Frontier Guard

Bug class hardened:

- `experiments/build_frontier_atom_ledger.py` still defaulted the active
  frontier to `C-051`, even after C-063 became the exact A++ frontier. That
  can poison water-fill tables, external byte-gap calculations, and report
  automation with stale anchors.

Permanent fix:

- Added C-063 to the default exact-candidate set.
- Added an explicit active-frontier preference list so newer exact frontier
  labels supersede older defaults when their adjudicated JSON is present.
- Added a focused regression test that fails if `C-063` and `C-051` are both
  available but the ledger selects the stale row.

Verification:

```text
py_compile: experiments/build_frontier_atom_ledger.py and focused tests
focused pytest: src/tac/tests/test_build_frontier_atom_ledger.py -> 4 passed
```

---

## Update - 2026-05-02T07:20Z Q-FAITHFUL Zoom-Consumption Contract Guard

Bug class hardened:

- Q-FAITHFUL geometry closure treated charged `zoom_scalars.bin` preservation
  as sufficient for a half-frame exact screen. H100 exact eval proved this is
  incomplete: the archive preserved the member, but the QZS3
  JointFrameGenerator runtime did not expose `use_zoom_flow`, so inflate used
  the duplicate fallback and PoseNet collapsed.

Permanent fix:

- `scripts/q_faithful_snapshot_loop.py` now records a
  `renderer_zoom_contract` and fails closed for half-frame masks plus charged
  zoom geometry unless the renderer runtime proves zoom consumption.
- The specific non-promotable reason is
  `zoom_warp_geometry_not_consumed_by_renderer`.
- Export metadata now carries the renderer zoom contract so future manifests
  cannot claim geometry closure from byte preservation alone.

Verification:

```text
py_compile: scripts/q_faithful_snapshot_loop.py and focused tests
focused pytest: src/tac/tests/test_q_faithful_snapshot_loop.py -> 11 passed
```
## 2026-05-02 - C-063 Pose-Regeneration Renderer Loader Hardening

Bug class: pose-regeneration and TTO tools can silently lag behind the contest
runtime renderer grammar. `experiments/optimize_poses.py` handled ASYM, FP4A,
and OWV3 locally, then fell through to `torch.load(..., weights_only=False)`.
The C-063 CRF52 stale-pose isolation run supplied a valid QZS3
`renderer.bin`, so the job failed before scientific evidence with a pickle
loader error. This is the same class as DEN-V2/SHIRAZ, but in an exact-eval
support tool rather than the inflate path.

Permanent fix landed:

- `experiments/optimize_poses.py` now inserts repo root before imports, keeps
  its existing ASYM/FP4A/OWV3 fast paths, and delegates packed contest runtime
  formats (`QZS3`, `MQZ1`, `QFAI`, `OWV2`, `NWC1`, `NWCS1`, `IMPS`, etc.) to
  `submissions/robust_current/inflate_renderer.py:_load_renderer` by file path
  so the upstream `submissions` package cannot shadow it.
- `experiments/optimize_poses.py` refuses to call `torch.load()` unless the
  payload starts with a torch-save pickle/zip magic.
- `src/tac/preflight.py::preflight_check` recognizes current packed runtime
  renderer magics and fails unknown non-pickle bytes instead of warning and
  assuming `.pt`.
- `src/tac/tests/test_optimize_poses_renderer_loader.py` locks the QZS3
  dispatch and unknown-magic rejection. Local focused verification:
  `3 passed` for that file; extended focused loader/preflight surface:
  `5 passed`.

Operational status: the failed H100 run is classified as
`failed_harness_qzs3_loader_bug`; the same CRF52 stale-pose isolation was
relaunched as `c063_mask_crf52_pose_regen_h100sxm_fix1_20260502T0752Z` with
remote source SHA preflight and no score claim until `contest_auth_eval.json`
lands.

Follow-on wall-clock bug in the same lane: once QZS3 loading was fixed, the
remote job entered `load_gt_video(n_frames=1200)` and spent minutes decoding
the full source video before slicing to 1200 frames. That is a harness
metabug for the deadline environment: a caller-supplied contest frame limit
must reach the decoder, not just a post-decode slice.

Permanent fix landed:

- `src/tac/data.py::decode_video` now accepts `max_frames` and breaks the
  decode loop immediately after the requested number of frames is materialized.
- `src/tac/data.py::load_gt_video` passes `n_frames` through to that decoder
  limit.
- `src/tac/tests/test_data_decode_limits.py` locks both behaviors.
- Focused verification: `5 passed` for `test_data_decode_limits.py` plus the
  QZS3 optimize-pose loader tests.

Operational status: the unbounded-decode relaunch is classified as
`stopped_harness_unbounded_gt_decode`; the CRF52 stale-pose isolation was
restarted as `c063_mask_crf52_pose_regen_h100sxm_fix2_20260502T0805Z` with
`src/tac/data.py` in the remote SHA preflight.

Additional wall-clock guard: `experiments/optimize_poses.py` now supports
`--skip-proxy-score`. If precomputed `--gt-pose-targets` are present and KL
distillation is inactive, this flag skips GT-video decode and the final proxy
score entirely. The archive evidence standard is exact CUDA auth eval anyway;
proxy generation was spending wall-clock without changing `optimized_poses.bin`
or the candidate archive. Focused verification now includes parse coverage for
the new flag: `6 passed` across data decode-limit and optimize-pose loader
tests.

Follow-on CUDA executor bug: the `fix2` CRF52 stale-pose isolation got past
loader and decode guards, then failed before score evidence in QZS3
`JointFrameGenerator` Conv2d with PyTorch
`Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(output)`.
The failed run used `--batch-pairs 100`; that micro-batch can exceed CUDA
32-bit indexing limits for the packed QZS3 runtime renderer even when VRAM is
available.

Permanent fix landed:

- `experiments/optimize_poses.py::apply_renderer_cuda_batch_safety` detects
  QZS3/JointFrameGenerator runtime renderers on CUDA and caps
  `--batch-pairs` to 32 before the first renderer call.
- `src/tac/tests/test_optimize_poses_renderer_loader.py` now includes
  regression coverage for the cap and its loud log line.
- `AGENTS.md` records the durable dispatch rule: do not relaunch QZS3 pose
  jobs by only changing a shell driver; keep the code guard and test.
- Focused verification after the batch guard plus Hubble's charged-mask branch:
  `30 passed` across charged-mask candidate tests, optimize-pose loader tests,
  and video decode-limit tests.

Operational status: the `fix2` run is classified as
`failed_cuda_index_math_batch100`. The corrected `fix3` H100 SXM run is active
with source hash `b25f86c0f11dcbea459b7c1e7cff870ea0556ee2dd88ebd43b6444b08ec9b302`
for `experiments/optimize_poses.py` and is using the batch cap in live logs.

## 2026-05-02 - C-063 CRF52 Pose-Regeneration Isolation Closed

Evidence grade: `A-negative` H100 CUDA diagnostic, not T4 promotion.

The `fix3` stale-pose isolation run completed after the permanent loader,
bounded-decode, proxy-skip, and QZS3 batch-safety guards. The archive was
evaluated through the canonical exact path:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Result:

```text
archive_sha256=c1f635775df5e52977eb8ae4714f57fc066df5fd150ad775f3f56eae0447914c
archive_bytes=263605
gpu=NVIDIA H100 80GB HBM3
score_recomputed=1.9196752592671689
avg_posenet_dist=0.19799431
avg_segnet_dist=0.00337047
inflate_elapsed_seconds=54.363056969828904
evaluate_elapsed_seconds=74.542215783149
artifact=experiments/results/vast_harvest/c063_mask_crf52_pose_regen_h100sxm_fix3_20260502/exact_eval/contest_auth_eval.json
```

Interpretation: regenerating poses against the decoded CRF52 mask stream
reduces the CRF52 collapse but remains more than 1.6 score points behind
C-063. The measured confound is therefore mask geometry/topology loss, not
only stale pose side information. This retires the measured CRF52
stale-pose-rescue implementation as a frontier path. It does not retire
pose-regeneration generally when paired with a scorer-safe mask representation.

## 2026-05-02 - Exact-Eval Runtime Custody Hash Guard

Incident: the same archive SHA-256
`226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
appears in two exact CUDA records with different scores:

- H100 diagnostic C-064:
  `experiments/results/vast_harvest/archive_eval_pr67_direct_publicmask_c059_modelpose_h100_20260502/contest_auth_eval.json`
  scored `0.3629464459892586`.
- T4 promotion C-067:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json`
  scored `0.31561703078448233`.

The archive bytes are identical, and both records hashed only
`submissions/robust_current/inflate.sh`, not the repo-local Python runtime
loaded by that shell script. This is a custody metabug: exact-eval comparison
requires archive bytes plus fixed runtime closure. If the runtime Python files
changed between H100 and T4, identical archive bytes are not a pure archive
comparison.

Permanent guard landed:

- `experiments/contest_auth_eval.py` now records
  `provenance.inflate_runtime_manifest`.
- The manifest includes the fixed runtime root, file list, bytes, SHA-256s,
  `upstream/evaluate.py` SHA-256, and
  `inflate_runtime_manifest.runtime_tree_sha256`.
- `src/tac/tests/test_contest_auth_eval.py` verifies that changing
  `inflate_renderer.py` changes the runtime tree hash while archive-like
  payload files such as `renderer.bin` are not mistaken for fixed runtime.
- `AGENTS.md` now requires runtime tree hashes for exact eval provenance and
  marks identical archive SHA with different runtime tree hashes as a
  runtime-custody comparison.

Verification:

- `.venv/bin/python -m py_compile experiments/contest_auth_eval.py src/tac/tests/test_contest_auth_eval.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_probe_cmg2_mask_codecs.py`
- `git diff --check` on the touched hardening files

Status: this guard is prospective. It does not retroactively invalidate C-067
T4 as the current promotion-grade frontier, but future exact-eval artifacts
will expose the missing comparison axis directly.

## 2026-05-02 - Dispatch Claim Terminal-Row Closure Guard

Incident: `.omx/state/active_lane_dispatch_claims.md` is append-first for
cross-agent auditability, but the claim helper's conflict detector treated an
older nonterminal row as active even after a newer terminal row was appended for
the same `lane_id` and `instance/job_id`. That can create phantom conflicts
after completed Lightning/Vast jobs and can push operators toward unsafe
manual edits or `--force` use.

Permanent guard landed:

- `tools/claim_lane_dispatch.py::_active_conflicts` now tracks newer terminal
  rows by matching `instance/job_id` and ignores older nonterminal rows closed
  by that terminal identity.
- `src/tac/tests/test_claim_lane_dispatch.py` covers the regression with a
  newest-first terminal row followed by an older active row for the same job.
- `src/tac/preflight.py::check_dispatch_claim_helper_present` now checks for
  the terminal-row closure guard and for AGENTS documentation.
- `AGENTS.md` now specifies that dispatch completion should be recorded by
  appending a terminal claim row through the helper.

Operational cleanup:

- Completed CMG2 exact-eval claims were closed with terminal rows for the
  downsample2x2, top256 repair, and top512 repair jobs.
- The stale reported Claude Q-FAITHFUL Vast redeploy was marked
  `stale_assumed_dead` after live Vast and Modal checks showed no running
  instance or task.

## 2026-05-02 - C-067 Fixed-Slice Matrix Anchor Guard

Incident: `experiments/build_c063_breakthrough_candidate_matrix.py` defaulted
to the older C-063 outer-Brotli PR64 length-table source layout. When asked to
refresh the candidate matrix against the current C-067 frontier archive, it
failed because C-067 is already a public PR67 fixed-slice `p` payload and is
not itself an outer Brotli stream. The failure was loud, but the previous
successful matrix refresh had therefore been anchored to the superseded C-063
frontier unless explicit C-067 line-search metadata was supplied.

Permanent guard landed:

- The matrix builder now recognizes `public_pr67_qzs3_qp1_fixed_slices` through
  the canonical packed-payload parser and reuses the exact C-067 `p` payload as
  the line-search source.
- The generated metadata records the current mask/model/pose charged slice
  sizes (`219472`, `55965`, `677`) and decoded member hashes.
- `src/tac/tests/test_build_c063_breakthrough_candidate_matrix.py` covers both
  the legacy outer-Brotli conversion and the public fixed-slice reuse path.

Current refreshed planning artifact:

```text
experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/c063_breakthrough_candidate_matrix.json
frontier_score=0.31561703078448233
frontier_archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
line_search_source_payload_format=frontier_public_pr67_fixedslice_reused_for_line_search
score_claim=false
```

## 2026-05-02 - CMG3A Raw-Mask SHA Header Guard

Incident: early CMG3A multimask reconciler archives failed inside inflate
before scorer execution because the archive header recorded a typed-array SHA
in fields named `source_mask_u8_sha256` and `reconstructed_mask_u8_sha256`.
The runtime loader correctly hashes raw uint8 decoded masks, so the guard
rejected otherwise well-formed archives with a deterministic mismatch. Example
pre-fix failure:

```text
ValueError: CMG3 reconstructed mask SHA mismatch:
manifest='5af320c7ff15d299c69f3098acbca82238be039d3a579a97ba4cf64315f3254c'
actual=feb59cab6084da9caecc1669cb127fe8e79df7ddd20515c3b476b1f5a4922bfe
```

Permanent guard landed:

- `experiments/build_c067_multimask_reconciler_candidate.py` now writes raw
  uint8 SHA values into CMG3 header fields and keeps typed-array identity in a
  separate manifest field.
- `src/tac/tests/test_build_c067_multimask_reconciler_candidate.py` adds a
  builder-to-runtime decode regression test that verifies the CMG3 header
  `reconstructed_mask_u8_sha256` matches `_decode_cmg3_nonzero_row_runs()`.
- Pre-fix L40S/T4 jobs are classified as pre-score harness/build failures, not
  score evidence.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_build_c067_multimask_reconciler_candidate.py -q`
  returned `5 passed`.

Status: prospective guard active. Fixed CMG3A archives must still earn their
own exact CUDA evidence before any ranking or promotion.

## 2026-05-02 - CMG3A Policy And NPZ Intake Guards

Incident: the Yousfi-Fridrich field-equation planner could emit class-0
row-run atoms into CMG3A policies even though the nonzero-row-run grammar
cannot encode implicit background class atoms. A separate builder intake path
also assumed decoded masks were stored under one specific `.npz` key, blocking
valid single-array decoded-mask artifacts.

Permanent guards landed:

- `experiments/plan_yousfi_fridrich_field_equations.py` filters unrepresentable
  class-0 atoms from CMG3A policies.
- `experiments/build_cmg3_nonzero_runs_candidate.py` accepts decoded-mask
  `.npz` files with a single array or the preferred keys
  `masks`, `mask`, `decoded_masks`, or `array`.
- Focused tests cover both the class-0 policy filter and `.npz` intake.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_plan_yousfi_fridrich_field_equations.py src/tac/tests/test_build_cmg3_adaptive_runs_candidate.py -q`
  returned `14 passed`.

## 2026-05-02 - CMG3A Pose-Collapse Dispatch Guard

Incident: fixed-header CMG3A multimask reconciler archives now pass runtime
custody, but exact CUDA diagnostics for extra065 and extra072 still showed
catastrophic PoseNet collapse. The bug class is remote dispatch that treats
plain target-body/extra-run byte screens as sufficient after known exact
PoseNet-collapse negatives.

Permanent guard landed:

- `src/tac/preflight.py` now includes
  `check_cmg3a_remote_dispatch_requires_pose_safety`, wired into strict
  `preflight_all`.
- The check scans remote dispatch scripts and rejects CMG3A
  `--target-body-bytes` / `--target-extra-runs` commands unless they include a
  pose-safety selector (`--field-policy-json`, hard-pair/frame selection,
  class weights) or an explicit `CMG3A_POSE_COLLAPSE_REVIEWED:<reason>`
  marker.
- Local builders remain available for planning and byte screens; the guard
  blocks new remote GPU spend on the measured-bad shape.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_preflight_meta_bugs.py -q`
  returned `250 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_build_c067_multimask_reconciler_candidate.py src/tac/tests/test_plan_yousfi_fridrich_field_equations.py src/tac/tests/test_build_cmg3_adaptive_runs_candidate.py -q`
  returned `19 passed`.

## 2026-05-02 - Public Archive Forensics Must Not Abort On Unsupported Grammars

Incident: the ZIP-level archive profiler originally aborted the whole
collection when a public/nonstandard archive could not be parsed by Python's
`zipfile` implementation, and the deeper stream-level byte-accounting profiler
could not account for PR65's compact member `x` bundle. That is a research
fragility class: one malformed or unsupported public archive should not block
byte forensics for the rest of the archive set.

Permanent guards landed:

- `src/tac/archive_byte_profile.py` now supports `--continue-on-error`,
  records invalid/nonstandard archives with bytes, SHA, error type, and
  `valid_profile=false`, and keeps the markdown renderer total-field safe.
- `experiments/profile_archive_byte_accounting.py` now recognizes PR65's
  analysis-only compact `x` grammar and emits stream accounting for the core
  mask/model/pose streams plus charged qpost/control streams. This parser is
  explicitly external byte forensics only; it does not change the contest
  inflate contract.
- Tests cover invalid archive collection and PR65 compact-bundle accounting.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_archive_byte_profile.py src/tac/tests/test_profile_archive_byte_accounting.py -q`
  returned `18 passed, 1 warning`.

## 2026-05-02 - C067 Half-Frame Pair Indices And Non-T4 Promotion Gate

Two fragility classes were exposed during C067 micro-mask diagnostics and are
now guarded.

Half-frame hard-pair expansion:

- Incident: the protected-mask builder interpreted hard-pair indices only as
  full-frame pairs and expanded each pair `i` to frames `2*i,2*i+1`.
  C067/Apogee mask streams are already 600 half-frame masks, so component
  trace pair indices can map directly to mask-frame indices. The old behavior
  rejected valid hard-pair policies with out-of-range protected frames.
- Permanent fix: `experiments/build_protected_mask_reencode_candidate.py`
  now exposes `--hard-pair-frame-mode {full_frames,half_frame_masks,auto}` and
  records requested/resolved mode in the manifest.
- Regression coverage:
  `src/tac/tests/test_build_protected_mask_reencode_candidate.py` now covers
  the half-frame mask-stream path.

Non-T4 exact CUDA promotion gate:

- Incident: `scripts/adjudicate_contest_auth_eval.py` could mark a non-T4
  exact CUDA diagnostic as `promotion_eligible=true` when score and component
  gates passed. This blurred the required distinction between L40S diagnostic
  evidence and T4/equivalent A++ promotion custody.
- Permanent fix: the adjudicator now separates
  `scientific_score_eligible` from `promotion_eligible`. Non-T4 exact CUDA
  can be score-grade diagnostic evidence, but it sets
  `hardware_promotion_gate_triggered=true` and
  `promotion_eligible=false` until identical archive bytes land on
  T4/equivalent hardware.
- Regression coverage:
  `src/tac/tests/test_remote_auth_eval_hardening.py` now asserts that non-T4
  exact CUDA packets are not promotion-eligible.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/adjudicate_contest_auth_eval.py \
  experiments/build_protected_mask_reencode_candidate.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_build_protected_mask_reencode_candidate.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_protected_mask_reencode_candidate.py \
  src/tac/tests/test_plan_c067_micro_mask_reencode.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_archive_byte_profile.py -q

60 passed, 1 warning
```

## 2026-05-03 - Lightning Studio Machine-Class Dispatch Guard

Incident: a Lightning exact-eval submit reached the SDK with
`g4dn.4xlarge`, then failed because the Studio provider cluster did not
exist. This was a paid-wall-clock preflight bug, not lane evidence.

Permanent fix:

- `src/tac/deploy/lightning/batch_jobs.py` now validates Studio-backed Batch
  Job specs against the known local machine/class routes before dry-run or
  submit records can call the SDK.
- Supported concrete routes are recorded as
  `g4dn.xlarge/T4_SMALL`, `g4dn.2xlarge/T4`,
  `g4dn.12xlarge/T4_X_4`, `g6e.4xlarge/L40S`,
  `g7e.4xlarge/RTXP_6000`, and
  `g7e.12xlarge/RTXP_6000_X_2`.
- `scripts/launch_lightning_batch_job.py` reuses the shared validation and
  checks component-response/component-sensitivity submissions before remote
  preflight work.
- Regression coverage asserts `g4dn.4xlarge` fails locally while supported
  current routes still pass.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/launch_lightning_batch_job.py \
  src/tac/deploy/lightning/batch_jobs.py \
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q

101 passed
```

## 2026-05-03 - Half-Frame Mask Training Index Guard

Incident: Modal fixed-renderer retraining burns on A10G and H100 failed before
training signal because `src/tac/experiments/train_renderer.py` treated a
600-frame half-frame mask stream as if it were indexed by 1200 full-frame
starts. The failure class is structural for any fixed-mask/fixed-pose burn
that feeds PR75/C067 half-frame masks through `--mask-noise-mkv`.

Permanent fix:

- `training_mask_pair_from_index(...)` now centralizes pair lookup.
- Noisy mask streams with `noisy_masks.shape[0] * 2 == all_masks.shape[0]`
  resolve to `half_frame_pair_index` instead of truncating the ground-truth
  video/mask tensors.
- Disagreement reporting compares half-frame noisy masks against odd-frame
  ground-truth masks.
- Out-of-range pair indices fail closed.

Regression coverage:

- `src/tac/tests/test_train_renderer_half_frame_noise.py` covers pair-index
  mapping and out-of-range protection.

Verification:

```text
.venv/bin/python -m py_compile \
  src/tac/experiments/train_renderer.py \
  src/tac/tests/test_train_renderer_half_frame_noise.py \
  experiments/modal_train_lane.py

.venv/bin/python -m pytest \
  src/tac/tests/test_train_renderer_half_frame_noise.py \
  src/tac/tests/test_prepare_c067_fixed_renderer_burn.py \
  src/tac/tests/test_modal_auth_eval.py -q

11 passed
```

Operational follow-up:

- Failed old Modal calls:
  `fc-01KQP98P99HHAX82ZS1XCR1PJJ`,
  `fc-01KQP9C0KV34JBNV81YMZC5KWX`.
- Patched fixed-renderer burns relaunched on H100/A100/A10G with explicit
  dispatch claims. These runs remain training evidence only until their
  snapshots pass transplant preflight, pose-safety preflight, and exact
  CUDA/T4 auth eval.

## 2026-05-03 - Lightning Repro Wrapper Dispatch-Claim Forwarding

Incident: `scripts/lightning_exact_eval_repro.py` could stage a reproducible
workspace and construct the exact-eval command, but it did not expose or
forward `--dispatch-lane-id` to `scripts/launch_lightning_batch_job.py`.
That forced operators to either drop down to the lower-level launcher or use a
break-glass path, increasing the chance of unclaimed exact-eval dispatch.

Permanent fix:

- Added `--dispatch-lane-id`, `--dispatch-claims-path`, and
  `--allow-missing-dispatch-claim-reason` to
  `scripts/lightning_exact_eval_repro.py`.
- Non-dry-run queue commands now forward those flags to the exact-eval
  submitter.
- Dry-run planning remains unchanged.

Regression coverage:

- `src/tac/tests/test_lightning_exact_eval_repro.py` asserts the active-claim
  flags and break-glass reason are forwarded into the generated queue command.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/lightning_exact_eval_repro.py \
  src/tac/tests/test_lightning_exact_eval_repro.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_exact_eval_repro.py -q

7 passed
```

## 2026-05-03 - Bounded Lightning Stop Command

Incident: while closing redundant exact-eval spend after the renderer-shrink
T4 result failed PoseNet, `scripts/launch_lightning_batch_job.py stop ...`
did not exist. Operators previously used ad hoc `lightning_sdk.Job.stop()`
snippets, and prior SDK stop calls could block long enough to interfere with
orchestration. This is a control-plane bug class: duplicate GPU jobs must be
terminable through the same stateful launcher that submitted and refreshes
them.

Permanent fix:

- Added `stop` to `scripts/launch_lightning_batch_job.py`.
- The command infers SDK job name, teamspace, org, and user from the local
  state record just like `refresh-status`.
- `Job.stop()` is bounded by `--timeout-seconds` so provider-side blocking does
  not hang local orchestration.
- Each stop request is appended to the state record under `stop_requests`, then
  the command refreshes SDK status into the same JSON.

Regression coverage:

- `src/tac/tests/test_lightning_batch_jobs.py` now verifies that the CLI stop
  command infers SDK context from state, calls the provider `stop()` method,
  records the request reason, and appends a status-history refresh.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/launch_lightning_batch_job.py \
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q

102 passed
```

## 2026-05-03 - Inherited A-negative Repack Dispatch Block

Incident: the fast P6 repacker found a 10-byte-smaller lossless child of the
`bb8d...` renderer-shrink candidate. That child had decoded-stream parity with
its source, but the source exact T4 result then closed as PoseNet-collapse
A-negative. A lossless wrapper repack of an exact-negative source inherits the
same component failure unless the runtime tree or decoded streams intentionally
change.

Permanent guardrail:

- The worker recommendation
  `experiments/results/c082_fast_packer_worker_20260503/dispatch_recommendation.json`
  now marks `shrink_queued_bb8d_p6_stream_resweep` as
  `blocked_inherited_a_negative_source`.
- The candidate manifest
  `experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/manifest.json`
  now carries `remote_dispatch_block`.
- Future lossless repack screens should treat decoded-stream parity with a
  known A-negative source as a dispatch blocker, not as a T4 candidate, unless
  the explicit experiment is runtime-custody differential diagnosis.

Verification:

```text
The source exact T4 artifact reports score `0.5066348615388583`, PoseNet
`0.00685808`, promotion disabled. The child manifest records decoded stream
parity for `masks.mkv`, `renderer.bin`, `optimized_poses.qp1`, and
`seg_tile_actions.bin`, so no score-affecting decoded payload changed.
```

## 2026-05-03 - Lightning Exact-Eval Wrapper Env Forwarding

Incident: the reproducible wrapper `scripts/lightning_exact_eval_repro.py`
successfully staged the QP1 active-subspace exact-eval packet, then failed
before queueing because the lower-level T4 submitter correctly required
inflate-side CUDA-12 torch pins. The wrapper did not expose or forward
`--env`, forcing a direct `launch_lightning_batch_job.py exact-eval` submit.

Permanent fix:

- Added repeatable `--env KEY=VALUE` to
  `scripts/lightning_exact_eval_repro.py`.
- The wrapper now forwards each env override to the lower-level exact-eval
  queue command and records the env list in the plan JSON.
- This preserves the higher-level manifest/staging workflow for T4 jobs that
  need `INFLATE_TORCH_SPEC`, `UV_EXTRA_INDEX_URL`, or other deterministic
  runtime pins.

Regression coverage:

- `src/tac/tests/test_lightning_exact_eval_repro.py` now verifies that repeated
  `--env` values are forwarded in order and preserved in the generated plan.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/lightning_exact_eval_repro.py \
  src/tac/tests/test_lightning_exact_eval_repro.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_exact_eval_repro.py -q

8 passed
```

## 2026-05-03 - Lightning Remote uv.lock Staging Preflight

Incident: a PR65/PR75 qpost interaction exact-eval submit reached remote
manifest verification and then failed before Batch job creation because
`scripts/lightning_repro_workspace.py --requirements-mode uv-sync` ran remote
`uv sync --locked --extra runtime` against a stale local `uv.lock` snapshot.
This burned wall-clock and produced partial remote staging, but no GPU job.

Permanent fix:

- `scripts/lightning_repro_workspace.py` now runs local
  `uv lock --check` before any SSH/rsync when `--requirements-mode uv-sync`
  is selected.
- A stale lockfile now fails locally with a clear remediation:
  run `uv lock`, or use `--requirements-mode no-install` when the remote
  exact-eval environment is already verified.
- `--requirements-mode no-install`, `verify-only`, dry-run, and SSH-check-only
  workflows are left untouched.

Regression coverage:

- `src/tac/tests/test_lightning_repro_workspace.py` covers the successful
  local lock preflight and the stale-lock failure class.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/lightning_repro_workspace.py \
  src/tac/tests/test_lightning_repro_workspace.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_repro_workspace.py -q

19 passed
```

## 2026-05-03 - Lightning Exact-Eval Component Trace Forwarding

Incident: the qpost exact screens completed with clean A++ custody but no
`component_trace.json`, so their exact negative signal was only aggregate
PoseNet/SegNet. That is enough for ranking, but it slows the next atom-level
allocator because it cannot learn which pairs improved or regressed.

Permanent fix:

- `scripts/lightning_exact_eval_repro.py` now exposes `--component-trace` and
  `--component-trace-top-k`.
- The wrapper forwards those flags to
  `scripts/launch_lightning_batch_job.py exact-eval`, preserving the
  manifest/staging/claim path while enabling per-pair traces on future exact
  screens.

Regression coverage:

- `src/tac/tests/test_lightning_exact_eval_repro.py` verifies that component
  trace flags are present in the generated queue command and recorded in the
  plan JSON.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/lightning_exact_eval_repro.py \
  src/tac/tests/test_lightning_exact_eval_repro.py \
  scripts/lightning_repro_workspace.py \
  src/tac/tests/test_lightning_repro_workspace.py

.venv/bin/python -m pytest \
  src/tac/tests/test_lightning_exact_eval_repro.py \
  src/tac/tests/test_lightning_repro_workspace.py -q

28 passed
```

## 2026-05-04 - Lightning Harvest Failure Classification And PR86 Stale-Evidence Guard

Two deadline-critical bug classes were hardened:

- `scripts/launch_lightning_batch_job.py` now refines missing
  `contest_auth_eval.json` harvest failures from `auth_eval.log` instead of
  collapsing every pre-score terminal into a generic missing-score-json class.
  New terminal classes include archive whitelist blocks, inflate returncode
  failures, and PR86 HPAC/constriction invalid entropy-model failures.
- `experiments/plan_pr86_hpac_pr85_contract_port.py` now prefers the stricter
  `pr86_hpac_full_decode_reencode_gate_20260504_codex.json` artifact when it
  exists, falling back to the older bounded-prefix diagnostic only when the
  full gate is absent.

Why this matters:

- Failed exact-eval jobs now preserve the real failure class needed for the
  next engineering action.
- PR86 contract-port planning no longer accidentally ignores the strongest
  local failed-closed evidence and proposes stale next gates.

Regression coverage:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_lightning_batch_jobs.py \
  -k "missing_score_json or refines_missing_score_json or missing_artifacts_bucket or persists_refined_harvest_failure" -q

6 passed, 109 deselected

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_pr86_hpac_pr85_contract_port.py \
  src/tac/tests/test_pr86_hpac_replay_parity.py -q

13 passed
```
