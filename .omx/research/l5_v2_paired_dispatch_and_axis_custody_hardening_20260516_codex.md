# L5-v2 paired dispatch and axis-custody hardening - 2026-05-16

## Scope

Codex patch set for L5-v2/Cathedral operator-surface hardening after adversarial
audit found paired CPU/CUDA and authority fail-open risks.

## Changes

- Added `tac.deploy.modal.paired_dispatch.paired_auth_eval_dispatch_command_template`
  as the canonical operator-facing paired Modal auth-eval command builder.
- Replaced direct packet-builder command templates for single-axis
  `experiments/modal_auth_eval.py` / `experiments/modal_auth_eval_cpu.py` with
  `tools/dispatch_modal_paired_auth_eval.py`.
- Covered PR106 HLM1, HDM8 film-grain sidecar, PR106 FES, and PR101 FES packet
  builders.
- HLM1 packet artifact now suppresses `submit_contest_cpu` and exposes a single
  paired CPU/CUDA execute path plus a plan-only command.
- CPU-axis custody now rejects mixed CPU+CUDA/GPU strings in hardware,
  inflate-device, eval-device, and auth-eval command surfaces.
- L5-v2 paired-axis validation now delegates device token semantics to the
  shared exact-eval custody classifier instead of duplicating its own token
  rules.
- Operator briefing now records PR106 PacketIR matrix SHA and suppresses L5-v2
  exact-eval target rows when the artifact SHA drifts from the pinned L5-v2
  constant.
- all-lanes preflight now fails on
  `l5_v2_packetir_matrix_artifact_sha_mismatch`.

## Non-authority

All surfaces remain:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

These are planning, custody, and guardrail changes only. No score claim or
promotion claim is made.

## Verification

- `ruff check` on changed Python files: pass.
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py src/tac/tests/test_hdm8_film_grain_sidecar.py src/tac/tests/test_frame_exploit_selector_packet.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py -q`
  -> `156 passed`.
- `PYTHONPATH=src:. .venv/bin/python tools/operator_briefing.py --json --top 3`
  -> PacketIR matrix SHA matches pinned SHA; target rows unsuppressed; 13 full
  target rows; all target commands use paired dispatcher; no direct Modal wrapper
  entrypoint leakage.
- `.venv/bin/python tools/review_tracker.py policy-check --changed-only --base HEAD`
  -> 0 violations.
- Direct wrapper scan over owned packet builders and HLM1 packet:
  no `experiments/modal_auth_eval*.py` operator command leakage.

## Remaining follow-up

- Paired Modal skip/reuse should append explicit terminal claim rows for reused
  anchors so no-spend reuse decisions have a complete claim lifecycle.
- Paper/OSS provenance schema should require source version/commit, retrieved
  date, claim scope, and license evidence for OSS-backed rows.
