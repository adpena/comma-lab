# JCSP rawvideo runtime consumption landing (2026-05-11)

## Classification

This is a score-lowering-enabling runtime bridge, not a score claim.

- `score_claim=false`
- `dispatch_attempted=false`
- exact-eval dispatch remains blocked until a concrete JCSP archive is built,
  claimed, and adjudicated through contest CUDA
- no scorer loads at inflate time

## What changed

The robust-current runtime now has a narrow production consume path for real
JCSP AQ/rawvideo streams:

- `submissions/robust_current/inflate.sh` defaults JCSP handling to
  `consume-real-raw-outputs` when `archive/jcsp.bin` is present.
- `submissions/robust_current/jcsp_runtime_bridge.py` emits `.raw` outputs
  only when every expected contest output is backed by exactly one real
  arithmetic-static AQ stream named as the relative `.raw` path.
- `src/tac/jcsp_stream_builder.py` exposes
  `rawvideo_bytes_to_stream_source()` as the Python oracle for the accepted
  rawvideo stream contract.

Fail-closed cases covered:

- missing `jcsp.bin`
- raw-passthrough fixture streams
- Ballé/hyperprior streams masquerading as rawvideo
- unexpected extra streams
- unsafe or non-`.raw` stream names
- non-RGB24-aligned payload lengths

## Adversarial correction

Worker output initially let `consume-real-raw-outputs` return success when the
JCSP member was absent. That was wrong for a production consume mode. Codex
tightened the CLI semantics so return code `0` requires all three facts:

- `ready_for_submission_runtime_consumption=true`
- `real_raw_outputs_emitted=true`
- `candidate_outputs_from_real_bridge_rawvideo=true`

Absent or rejected members now return the JCSP refusal code with a structured
blocker, preserving no-op protection.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_jcsp_runtime_bridge.py src/tac/tests/test_jcsp_stream_builder.py -q`
  - `51 passed`
- `.venv/bin/python -m ruff check submissions/robust_current/jcsp_runtime_bridge.py src/tac/jcsp_stream_builder.py src/tac/tests/test_jcsp_runtime_bridge.py src/tac/tests/test_jcsp_stream_builder.py`
  - pass
- `bash -n submissions/robust_current/inflate.sh`
  - pass
- `time .venv/bin/python tools/all_lanes_preflight.py`
  - all 29 checks passed
  - wall clock: `2.539s total`

## Score-lowering relevance

This unblocks a byte-closed JCSP rawvideo archive candidate path without
touching scorer code or relying on fixture passthrough. The immediate next
engineering step is to build a minimal JCSP rawvideo candidate archive from a
known raw-output source, run local runtime smoke/no-op proof, then claim and
send exact contest CUDA only if the charged bytes and runtime closure are
credible.
