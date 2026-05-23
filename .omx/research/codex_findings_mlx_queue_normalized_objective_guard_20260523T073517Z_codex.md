# Codex Findings - MLX Queue Normalized Objective Guard

timestamp_utc: 2026-05-23T07:35:17Z
agent: codex
lane_id: lane_codex_mlx_queue_normalized_objective_guard_20260523
axis_scope: [macOS-MLX research-signal, macOS-CPU advisory, contest-CPU, contest-CUDA]
score_claim: false
promotion_eligible: false
research_only: false

## Review Scope

Adversarial review and hardening pass over the MLX scorer-response, learned
sweep, and candidate-queue surfaces that can consume local MLX/local CPU
signals before exact contest-axis spend. The pass focused on false-authority
escape hatches, normalized-objective signal loss, malformed selected-pair
admission, and queue wiring that could let a local signal influence spend or
planning without the required full-video objective contract.

## Findings And Fixes

1. `score_claim_valid` was not uniformly covered by the false-authority
   contract. MLX scorer-response normalization and effective-spend triage now
   reject truthy `score_claim_valid` when present and emit it as `false` on
   normalized local rows.

2. MLX response rows could fall back to raw scorer fields when the normalized
   full-video objective was missing. Planning now excludes those rows from MLX
   production-contract coverage, effective-spend triage, probe row generation,
   and best-metric selection until the normalized objective is present.

3. Normalized full-video break-even bytes were not independently checked.
   `tac.optimization.normalized_objective` now recomputes and validates
   `break_even_added_bytes_from_normalized_full_video_gain` rather than
   trusting propagated metadata.

4. Decoder-q selective bridge and cross-family portfolio wiring could carry or
   admit MLX-normalized rows without a complete validated objective. Both now
   require the canonical normalized-objective helper at the handoff boundary,
   and manual cross-family candidate rows cannot bypass it by smuggling
   normalized fields.

5. DQS1 local-first queue selected-pair payloads accepted malformed pair lists.
   The queue now rejects duplicate, negative, out-of-range, and unsorted
   selected-pair indices against the canonical FEC6 pair count.

6. Dynamic learned-sweep and optimizer queue admission lacked aggregate guards
   connecting raw window gain to normalized full-video gain. Both now verify
   that the denominator is the canonical 600-sample contest video length and
   that MLX/window/full-video aliases agree before candidate admission.

7. Optimizer scheduler pairings could reference parent queue candidate ids that
   were not accepted from the ranked learned-sweep rows. Pairing admission now
   validates parent membership so downstream combinations cannot silently bind
   to orphaned or rejected ranked candidates.

## Wired Surfaces

- `src/tac/optimization/scorer_response_dataset.py`
- `src/tac/optimization/mlx_effective_spend_triage_selection.py`
- `src/tac/optimization/decoder_q_selective_window_bridge.py`
- `src/tac/optimization/cross_family_candidate_portfolio.py`
- `src/tac/optimization/mlx_dynamic_learned_sweep.py`
- `src/tac/optimizer/candidate_queue.py`
- `src/comma_lab/scheduler/dqs1_local_first_queue.py`

## Verification

Focused pytest stack:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_normalized_objective.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py \
  src/tac/tests/test_decoder_q_selective_window_bridge.py \
  src/tac/tests/test_cross_family_candidate_portfolio.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_dqs1_local_first_queue_builder.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep.py \
  src/tac/tests/test_optimizer_candidate_queue.py
```

Result: `193 passed in 2.29s`.

Lint:

```bash
.venv/bin/ruff check <touched implementation and test files>
```

Result: `All checks passed!`.

Whitespace:

```bash
git diff --check
```

Result: clean.

## Gitignore Review

No `.gitignore` edit was required in this pass. The generated-artifact classes
adjacent to these fixes are already covered:

- `.omx/state/experiment_queue*.sqlite`
- `.omx/state/experiment_queue*.db`
- `.omx/state/experiment_queue_logs/`
- `experiments/results/`
- `.omx/state/*.json`
- `.omx/state/*.jsonl`

The new tests use inline fixtures or pytest temporary paths and do not create
new durable artifact patterns.

## Remaining Work

1. Add a reusable no-clobber/free-space artifact writer and apply it to queue,
   learned-sweep, and materialization paths before the next large MLX/local-CPU
   batch.

2. Add optional constructed-unit validation to selective bridge consumers so
   selected-pair lists, parent ids, normalized objective metadata, and archive
   bytes are validated as one object before dispatch.

3. Extend the same normalized-objective and parent-membership guards to any
   remaining X-ray, master-gradient, PacketIR, and cathedral autopilot consumers
   that accept local MLX/local CPU rows.

4. Run broader preflight and full relevant optimizer/scheduler suites once the
   current focused guard patch is landed and serialized.
