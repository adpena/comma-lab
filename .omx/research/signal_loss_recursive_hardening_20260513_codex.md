# Recursive Signal-Loss Hardening

Date: 2026-05-13
Author: codex

## Scope

This landing converts the latest recursive adversarial review findings into
code-level fail-closed guards. No GPU dispatch, no remote provider launch, and
no score claim were made.

## Fixed Failure Classes

1. **Adjudication laundering**
   - `scripts/adjudicate_contest_auth_eval.py` now fails closed when raw
     `promotion_blockers` or `rank_or_kill_blockers` remain on an input
     artifact.
   - The adjudicated result rewrites both `allowed_use` and `allowed_uses` to
     the fail-closed view, and sets `score_claim=false`,
     `score_claim_valid=false`, `promotion_eligible=false`, and
     `rank_or_kill_eligible=false` when blocked.
   - Regression coverage:
     `src/tac/tests/test_adjudicate_contest_auth_eval_policy.py`.

2. **PR106 runtime proof signal loss**
   - `src/tac/packet_compiler/pr106_runtime_consumption.py` now binds runtime
     proofs to both archive SHA-256 and runtime source-tree SHA-256.
   - Format `0x02` proofs now probe `framing_meta` consumption, not just
     `sidecar_payload`.
   - `tools/all_lanes_preflight.py` Gate #26 enforces archive/runtime SHA
     custody, section-level runtime consumption, and non-score labels.
   - Separate detailed ledger:
     `.omx/research/pr106_runtime_consumption_sha_guard_20260513_codex.md`.

3. **Vast.ai claim-after-create spend risk**
   - `scripts/launch_lane_on_vastai.py` now opens a pre-provider
     `active_precreate` claim before paid instance creation, then force-writes
     the real instance claim and closes the pre-provider row as superseded.
   - If instance creation or real-claim recording fails, the pre-provider row
     is closed terminally.

4. **Provider terminal-status drift**
   - `src/tac/deploy/claims.py` now matches the canonical terminal-prefix set
     from `tools/claim_lane_dispatch.py`, including `falsified_`, `retired_`,
     `config_retired_`, and `measured_implementation_retired_`.

5. **Kaggle proxy false authority**
   - `src/tac/deploy/kaggle/kaggle_output_ingest.py` preserves source manifest
     proxy flags as `source_manifest_*`, but forces ingest summary
     `score_claim=false`, `ready_for_exact_eval_dispatch=false`, and
     `promotion_eligible=false`.
   - `tools/harvest_kaggle_kernels.py` now closes terminal Kaggle claims even
     when no downloadable kernel summary exists, using the active claim ledger
     for the provider job id when possible.

6. **Modal smoke non-interactive refusal**
   - `tools/operator_authorize.py` adds explicit `--yes` for caller-owned
     operator approval.
   - `tools/run_modal_smoke_before_full.py` passes `--yes` for both smoke and
     full dispatch phases after the wrapper-level route is already approved.

7. **Muon optimizer canonicalization**
   - `src/tac/optimization/muon.py` is now canonical OSS code with ASCII
     docstrings, lazy exports from `tac.optimization`, and focused tests.
   - This preserves the PR95/Muon training signal as code, not chat.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_adjudicate_contest_auth_eval_policy.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_muon_optimizer.py \
  src/tac/tests/test_launch_lane_on_vastai_create_instance.py \
  src/tac/tests/test_deploy_claims_active_row.py \
  src/tac/tests/test_kaggle_output_ingest.py \
  src/tac/tests/test_kaggle_t1_balle_sweep_harness.py \
  src/tac/tests/test_run_modal_smoke_before_full.py
```

Observed: `102 passed`.

```bash
.venv/bin/python -m py_compile \
  scripts/adjudicate_contest_auth_eval.py \
  scripts/launch_lane_on_vastai.py \
  src/tac/deploy/claims.py \
  src/tac/deploy/kaggle/kaggle_output_ingest.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/optimization/muon.py \
  tools/harvest_kaggle_kernels.py \
  tools/harvest_modal_calls.py \
  tools/operator_authorize.py \
  tools/prove_pr106_sidecar_runtime_consumption.py \
  tools/run_modal_smoke_before_full.py \
  tools/all_lanes_preflight.py
```

Observed: passed.

```bash
.venv/bin/python -m ruff check \
  scripts/adjudicate_contest_auth_eval.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  tools/prove_pr106_sidecar_runtime_consumption.py \
  tools/all_lanes_preflight.py \
  src/tac/optimization/muon.py \
  src/tac/optimization/__init__.py \
  tools/operator_authorize.py \
  tools/run_modal_smoke_before_full.py \
  src/tac/tests/test_adjudicate_contest_auth_eval_policy.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_muon_optimizer.py \
  src/tac/tests/test_deploy_claims_active_row.py \
  src/tac/tests/test_kaggle_output_ingest.py \
  src/tac/tests/test_kaggle_t1_balle_sweep_harness.py \
  src/tac/tests/test_run_modal_smoke_before_full.py
```

Observed: passed.

```bash
.venv/bin/python -m ruff check \
  --select E402,I001,C408,RUF022,UP035,B009,UP017 \
  scripts/launch_lane_on_vastai.py \
  src/tac/deploy/claims.py \
  src/tac/deploy/kaggle/kaggle_output_ingest.py \
  tools/harvest_kaggle_kernels.py \
  tools/harvest_modal_calls.py \
  src/tac/tests/test_launch_lane_on_vastai_create_instance.py
```

Observed: passed. The full `scripts/launch_lane_on_vastai.py` legacy ruff
surface still contains older style findings outside this hardening slice; this
landing did not churn that launch script mechanically beyond the claim-order
fix.

## Follow-Up Hardening From Recursive Review

The second review pass found three non-blocking but real signal-loss risks and
they were fixed in code before landing:

- Modal/Vast/Lightning/GHA Linux x86_64 CPU exact-eval anchors now share the
  same approved `[contest-CPU]` posterior substrate policy in
  `tac.continual_learning` and the non-HNeRV drift calibrator. macOS CPU remains
  advisory-only.
- Blocking-mode Modal CPU/CUDA terminal claim notes now carry score axis,
  hardware, archive SHA-256, archive bytes, recomputed score, and output
  directory, matching the recovery path's stale-custody suppression needs.
- `tools/profile_pr106_latent_sidecar_recode.py` now emits
  `best_runtime_decoder_implemented_candidate`; the legacy
  `best_runtime_consumed_candidate` key remains only as a backward-compatible
  alias and the report still carries `score_claim=false` plus dispatch blockers.
- `tools/run_modal_smoke_before_full.py` now honors recipe-level
  `smoke_only: true` and refuses `--full-only` when a recipe declares that
  scaffold guard. This prevents S2SBS from accidentally firing a full Modal run
  before the full trainer implementation exists.
- Gate #10 no-signal-loss source audit was greened by refreshing the
  `experiments/results/` runtime-source baseline in
  `.omx/research/untracked_source_dispositions_20260505_codex.json` after the
  new tracked result artifacts landed.
- `tools/payload_entropy_density_map_local.py` was promoted from untracked
  source-like signal into tracked tooling after py_compile + ruff validation.

Additional verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_continual_learning.py \
  src/tac/tests/test_bulk_backfill_anchors_into_posterior.py \
  src/tac/tests/test_calibrate_non_hnerv_drift_class.py
```

Observed: `124 passed`.

```bash
.venv/bin/python -m ruff check \
  src/tac/continual_learning.py \
  src/tac/tests/test_continual_learning.py \
  src/tac/tests/test_bulk_backfill_anchors_into_posterior.py \
  tools/calibrate_non_hnerv_drift_class.py \
  src/tac/tests/test_calibrate_non_hnerv_drift_class.py \
  experiments/modal_auth_eval.py \
  experiments/modal_auth_eval_cpu.py \
  tools/profile_pr106_latent_sidecar_recode.py
```

Observed: passed.

```bash
.venv/bin/python -m py_compile \
  tools/payload_entropy_density_map_local.py \
  tools/run_modal_smoke_before_full.py \
  src/tac/tests/test_run_modal_smoke_before_full.py

.venv/bin/python -m ruff check \
  tools/payload_entropy_density_map_local.py \
  tools/run_modal_smoke_before_full.py \
  src/tac/tests/test_run_modal_smoke_before_full.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_check_198_operator_authorize_bypass.py

.venv/bin/python tools/audit_untracked_source_artifacts.py \
  --repo-root . \
  --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json \
  --strict --format text
```

Observed: py_compile passed; ruff passed; `29 passed`; untracked source audit
had `0 undispositioned`.

## Evidence Discipline

All affected proof, provider, Kaggle, Modal-smoke, and Muon surfaces remain
non-score by default unless an exact byte-closed `[contest-CUDA]` or
Linux x86_64 `[contest-CPU]` artifact lands through the canonical claimed
dispatch and adjudication path.
