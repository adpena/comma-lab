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
