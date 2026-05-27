# Codex findings: 5D extended operator queue wire-in

- timestamp_utc: 2026-05-27T03:40:00Z
- agent: codex
- lane_id: `lane_build_2_3_ext_8_not_built_operators_replace_merge_reorder_frame_level_motion_conditional_temporal_coherence_20260526`
- scope: preserve stale subagent BUILD-2+3-EXT work, add missing cathedral/queue automation, and verify against the live 6bae 5D canvas artifact
- authority: `[predicted]`, false-authority, encoder-side planning only

## Findings

The subagent's 8 extended operators were present but not fully integrated. The module and CLI were syntactically valid, and the comprehensive test file already passed, but ruff failed and there was no auto-discovered cathedral consumer or queue artifact. That left the surface closer to "built plus tested" than "wired and automated."

Codex added:

- `src/tac/cathedral_consumers/pair_frame_5d_extended_operator_consumer/__init__.py`
- `src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py`
- `tools/build_5d_extended_operator_queue.py`
- focused queue tests in `src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`
- a real CLI execution test in `src/tac/tests/test_8_extended_operators_5d_canvas.py`

## Live Execution

Built queue:

```bash
.venv/bin/python tools/build_5d_extended_operator_queue.py \
  --canvas-path experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/populated_5d_canvas.json \
  --output-root experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/extended_operator_queue_outputs_v2 \
  --queue-out experiments/results/pair_frame_5d_canvas_build123_20260527T032000Z/extended_operator_queue_v2.json \
  --queue-id pair_frame_5d_extended_operator_live_6bae0201_v2 \
  --top-n 8 \
  --overwrite
```

Validation:

- queue id: `pair_frame_5d_extended_operator_live_6bae0201_v2`
- experiments: 8
- steps: 8
- `valid=true`

Worker execution:

- `success_count=8`
- `failure_count=0`
- each operator wrote a candidate manifest under `extended_operator_queue_outputs_v2/`
- all live manifests emitted zero candidates, which is a fail-closed no-false-positive result for the sparse live canvas

## Integration Bug Fixed

The first queue worker run failed all 8 steps despite successful CLI execution because `json_false_authority` defaults require top-level false authority keys. Extended candidate manifests store false-authority markers inside candidate rows and omit top-level score-authority fields. The queue postcondition now sets `required_false=[]` and treats the top-level authority keys as `false_or_missing`, which matches the manifest contract and still blocks truthy authority.

## Verification

```bash
.venv/bin/ruff check \
  src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py \
  tools/build_5d_extended_operator_queue.py \
  src/tac/tests/test_pair_frame_5d_extended_operator_queue.py \
  src/tac/tests/test_8_extended_operators_5d_canvas.py \
  src/tac/cathedral_consumers/pair_frame_5d_extended_operator_consumer/__init__.py \
  src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.py \
  tools/apply_8_extended_operators_to_5d_canvas_cli.py
```

`All checks passed.`

```bash
.venv/bin/pytest \
  src/tac/tests/test_pair_frame_5d_extended_operator_queue.py \
  src/tac/tests/test_8_extended_operators_5d_canvas.py -q
```

`60 passed`

## Remaining Gap

This lands queue-owned local automation and cathedral visibility. It does not yet make any operator score-authoritative, exact-eval-ready, or globally optimal. The next integration step is to feed this queue artifact into the frontier autonomous-chain refresh output when a populated 5D canvas artifact exists, then widen the operator parameter sweeps and MLX scorer-response calibration.
