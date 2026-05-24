# Codex Findings - Decoder-Q Lightweight Scheduler Constants

UTC: 2026-05-24T01:37:11Z
Author: Codex
Lane: `codex_decoderq_lightweight_constants_20260524T013711Z`

## Finding

The first bounded `tertiary` SSH setup exposed an import-boundary bug class:
planning-only scheduler startup imported
`tac.optimization.decoder_q_selective_runtime_packet` only to read
`FEC6_PAIR_COUNT`. That runtime packet module imports PR101/FEC6 byte-closed
mutation code and eventually `torch`, so an 8 GB CPU-only remote could not even
start a false-authority materializer planning row without a heavyweight GPU
dependency.

That is not a dependency issue; it is a layering issue. Queue/DAG scheduling
and planning constants must remain lightweight.

## Landing

Added `tac.optimization.decoder_q_constants` for lightweight DQS1 constants:

- `FEC6_PAIR_COUNT`
- `CONTEST_RATE_DENOMINATOR_BYTES`

Scheduler/planning modules that only need these constants now import the
lightweight module. `decoder_q_selective_runtime_packet` reuses the same
constant definitions, preserving one source of truth while keeping runtime
packet imports byte-closed and explicit.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check src/tac/optimization/decoder_q_constants.py src/tac/optimization/decoder_q_selective_runtime_packet.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/dqs1_local_first_queue.py src/tac/optimization/decoder_q_selective_window_bridge.py src/tac/optimization/decoder_q_selective_runtime_feedback.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_experiment_queue.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/tac/optimization/decoder_q_constants.py src/tac/optimization/decoder_q_selective_runtime_packet.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/dqs1_local_first_queue.py src/tac/optimization/decoder_q_selective_window_bridge.py src/tac/optimization/decoder_q_selective_runtime_feedback.py`

Result: all passed.

## Remaining Work

After this patch is pushed, verify on `tertiary` that scheduler/materializer
imports pass with only the minimal planning dependencies installed, then run the
bounded SSH materializer smoke with artifact pullback.
