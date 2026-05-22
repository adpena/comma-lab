# Codex Findings: DQS1 Rank019 Observation, Rank023 Queue, Worker Hardening, Drift Guards

Date: 2026-05-22T20:19:51Z

## Verdict

Rank019/pair0151 is now an exact `[contest-CPU]` observed regression in the
same SegNet-penalized one-byte-drop class as ranks 020, 022, 026, 027, and 031.
The canonical component-marginal equation has an eighth empirical anchor for
pair 151, and the local-first queue has moved to rank023/pair0440.

This memo is the append-only rank019/rank023 continuation record. The older
dated pairset and queue memos are preserved as historical provenance rather
than being rewritten for the new current state.

## Rank019 Evidence

- Candidate: `pairset_drop_one_rank019_pair0151`
- Axis: `[contest-CPU]`
- Score: `0.19202928295713673`
- Archive bytes: `178559`
- Archive SHA-256:
  `846f19ba010b79e3c9c47627ff8754a2dd07bb0f10047ad4896c468e802701c6`
- Runtime SHA-256:
  `bb16adef8e0a27b5f56c3534978ad2f8f11700565228da96da872d73313dfe5a`
- Inflated raw aggregate SHA-256:
  `be8edef3aa4b1edc55248d8b5365e6c970170e56952c1998040df6bdb7610186`
- Component deltas versus compact DQS1 top32 CPU: PoseNet `+0.0`,
  SegNet `+0.000001`, rate `-0.00000066585895312`
- Verdict: exact CPU regression versus current rank021 CPU frontier

## Canonical Signal

- Canonical equation:
  `pairset_component_marginal_score_decomposition_v1`
- New empirical anchor:
  `dqs1_pair0151_drop_one_component_cpu_penalty_20260522`
- Source artifact for the rank019 anchor: this memo
- Current helper-generated portfolio:
  `experiments/results/cross_family_candidate_portfolio/20260522T202400Z_pairset_component_rank019_append_only_hardened/portfolio.json`
- Portfolio SHA-256:
  `66dbcd3588d1f19f844cdd73f2b3656b811ce0ff9b0b187efd47b22041a5f267`
- Action summary SHA-256:
  `7a4c658bfa8bcd45d092203b1682cfac73925e322db06b3b7eccfcd144545924`
- Current model summary: CPU-safe observed drop pair `[371]`;
  CPU-protected observed drop pairs `[327, 376, 320, 378, 296, 430, 167, 151]`;
  CUDA-protected observed drop pair `[371]`
- Next recommended local-control candidate:
  `pairset_drop_one_rank023_pair0440`

## Queue And Worker Hardening

- Current queue lane:
  `lane_dqs1_pairset_drop_one_rank023_pair0440_local_first_20260522`
- Queue definition:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`
- Rank023 selected pairs:
  `[26, 59, 68, 98, 109, 112, 134, 151, 167, 229, 242, 257, 259, 296, 320, 327, 371, 376, 378, 412, 430, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`
- `run-worker --execute --max-steps N` now provides bounded local queue
  execution with SIGINT/SIGTERM stop requests, queue-definition reload between
  steps, append-only worker events, and dry-run planning.
- Step claiming is atomic against `status='queued'` and queue control mode
  `running`, so a stale `ReadyStep` cannot bypass `paused` or `frozen`.
- Unknown resource kinds now fail closed during queue normalization instead of
  being treated as local resources.
- Duplicate step ids are rejected before state insertion, preventing SQLite
  primary-key collapse and silent command loss.
- Queue summaries count active-definition steps separately from stale reroute
  rows, reported as `orphaned_steps`.
- Queue SQLite WAL/SHM/journal sidecars are ignored under `.gitignore`.

## Local CPU Drift Hardening

- DQS1/FEC6 drift calibration now has a narrow trust-region assessment for
  same-archive, SegNet-rounding-only local CPU versus contest CPU anchors.
- Out-of-class rows are rejected before calibration and retained as
  rejected-anchor evidence.
- Empty calibrations serialize with a finite fail-closed guard band instead of
  producing non-JSON `Infinity`.
- Calibration JSON loading recomputes from embedded anchors rather than trusting
  stored stats.
- Eureka signals remain false-authority exact-eval spend triggers only and now
  block on candidate trust-region mismatch or non-local-CPU advisory axes.

## Commands

- `.venv/bin/python tools/run_decoder_q_selective_runtime_locality_controls.py`
  on drop-one rank019 pair0151
- `.venv/bin/python tools/recover_modal_auth_eval.py --output-dir experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank019_pair0151_selective_decoderq_cpu_20260522T200223Z`
- `.venv/bin/python tools/append_mlx_dynamic_sweep_observation.py --jsonl experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl --candidate-id pairset_drop_one_rank019_pair0151 --sweep-config-id contest_cpu_exact_candidate --optimization-pass-id exact_cpu_calibration --family decoder_q_pairset_drop_one --observed-axis contest_cpu --evidence-grade contest-CPU --evidence-tag '[contest-CPU]' --observed-score-or-delta 0.19202928295713673 --archive-sha256 846f19ba010b79e3c9c47627ff8754a2dd07bb0f10047ad4896c468e802701c6 --runtime-sha256 bb16adef8e0a27b5f56c3534978ad2f8f11700565228da96da872d73313dfe5a --raw-output-or-cache-sha256 be8edef3aa4b1edc55248d8b5365e6c970170e56952c1998040df6bdb7610186 --segnet-delta 0.000001 --posenet-delta 0.0 --rate-delta -0.00000066585895312 --source-artifact experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank019_pair0151_selective_decoderq_cpu_20260522T200223Z/contest_auth_eval.json --run-id dqs1_pairset_drop_one_rank019_pair0151_cpu_20260522T200223Z --selected-pair-indices 26,59,68,98,109,112,134,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- `.venv/bin/python tools/canonicalize_pairset_component_marginal_signal.py --incumbent-score 0.205330029 --incumbent-score-by-axis contest_cpu=0.19202828295713675 --pairset-acquisition experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition_dense_tail_20260522T1812Z.json --observation-jsonl experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl --output-dir experiments/results/cross_family_candidate_portfolio/20260522T202400Z_pairset_component_rank019_append_only_hardened --top-k 32 --top-actions 8 --register-equation --agent codex --subagent-id pairset_component_marginal_rank019_append_only_20260522 --equation-notes register_pairset_component_marginal_rank019_anchor_after_append_only_source_hardening`

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py tools/experiment_queue.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_experiment_queue.py src/tac/optimization/local_cpu_contest_drift.py tools/calibrate_local_cpu_contest_drift.py src/tac/tests/test_local_cpu_contest_drift.py src/tac/canonical_equations/pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_local_cpu_contest_drift.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_canonical_equations_initial_population.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state .omx/tmp/dqs1_pairset_queue_smoke_rank023.sqlite status`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state .omx/tmp/dqs1_pairset_queue_worker_plan_rank023.sqlite run-worker --max-steps 2`
- `.venv/bin/python tools/lane_maturity.py validate`
- `git diff --check`

## Next Actions

1. Run the rank023 queue plan and materialization locally through `run-worker`
   once the current patch is committed.
2. If locality controls pass, run the local macOS CPU advisory and route
   eureka/drift calibration output as spend-triage only.
3. Dispatch exact contest CPU/CUDA anchors only through the existing claim and
   recovery gates after local controls identify a candidate worth spend.
