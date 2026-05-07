# Bug-hunter v3 — Integration + composition seam review

**Date**: 2026-05-07
**Round**: 3 (after v1 module-level + v2 individual-module passes)
**Scope**: integration + composition seams between cathedral modules
**Test count**: 211 baseline → 227 (+16 new regression tests)
**Commits landed**: 5 (sequential per-fix, all via `subagent_commit_serializer.py`)
**Final verdict**: **all integration seams clean, ready for Phase 1 dispatch**

## Summary

Round 3 looked specifically at the SEAMS between cathedral modules — places
where one module's output feeds another's input and a contract drift could
silently corrupt downstream behavior. Five findings landed with regression
tests; one seam was checked and found clean (no fix needed); one seam (Op4
encode delegate's matrix audit) was already covered by v2's existing tests.

## Per-seam findings

### Seam 1 — Cross-op state_dict drift (Op3 → Op1)
**Status**: clean. No fix needed.

The substrate-transform composition `[Op3, Op1]` is empirically correct: the
final decoded state matches Op1's int7-band re-quantization of the int6
substrate (per-tensor delta ~ 1.5e-3, well within Op1's quant grid). This is
the documented composition contract; tests at `test_codec_pipeline_full_stack.py`
already pin the rel-err bound (≤ 5%) and roundtrip integrity. No drift
found.

### Seam 2 — CPL1 wire-format edge cases
**Finding 1 (MEDIUM)**: opaque error on non-JSON-serializable op_state
**Status**: FIXED — commit `245114c5`

The previous `CodecPipeline.encode` called bare `json.dumps(op_state, ...)`.
When an op accidentally embedded a `torch.Tensor` / `numpy.ndarray` / `set`
in `op_state`, the `TypeError` was opaque (`"Object of type Tensor is not
JSON serializable"`) with no indication which op or which key was responsible.
Across 8 canonical stacks and 2-3 ops per stack, operators had to bisect by
hand.

**Fix**: catch the `TypeError`, walk `op_state` recursively to find the
first non-serializable leaf via `_find_non_json_serializable_key`, re-raise
with op_name + dotted key path + JSON-encoding fix hint.

**Tests**: 3 new (`test_pipeline_encode_raises_actionable_error_on_non_json_op_state`,
`test_pipeline_encode_with_numpy_in_op_state_names_offender`,
`test_pipeline_stress_30_op_chain_roundtrips` — also stress-tests CPL1
correctness on long chains).

### Seam 3 — StreamProximalCodec ↔ CodecOp adapter gap
**Status**: clean (Protocols are orthogonal by design, no adapter needed
for current usage). Documented for future reference: a `CodecOp` can NOT
be wrapped to satisfy `StreamProximalCodec` directly — they have
incompatible call signatures (CodecOp expects `state_dict + context`,
StreamProximalCodec expects `target_bytes + dual`). This is correct
separation of concerns; the bridge between them is `Op_GammaJointADMM`,
which builds `StreamSource` instances internally from a `state_dict`.

### Seam 4 — contest_score_marginals ↔ joint_admm_coordinator integration
**Finding 2 (MEDIUM)**: documented oracle but no adapter
**Status**: FIXED — commit `db51e772`

`contest_rate_distortion_system.py` documented `contest_score_marginals` as
"the canonical operating-point-aware sensitivity oracle that the Joint-ADMM
coordinator queries each iteration to update its per-stream dScore/dByte
marginals" — but no adapter wired the contest formula's marginals into the
coordinator's `StreamSource.score_per_byte_marginal`. The coordinator
(`joint_admm_coordinator.py`) does not even import the contest module.
Callers ad-hocced constants (e.g. `Op_GammaJointADMM` defaulted marginal
to `1e-6`).

**Fix**: two new helpers in `contest_rate_distortion_system.py`:
- `joint_admm_marginal_for_stream(stream_role, ...)` — canonical bridge.
  Roles: weights/renderer/rate (return dS/dB), seg_correction, pose_correction
  (with operating-point-aware divergence at low pose_distortion).
- `joint_admm_marginal_from_empirical(delta_score_per_byte)` — pass-through
  with finite + non-negative sign-convention guards.

**Tests**: 6 new in `test_contest_rate_distortion_system.py`
(weights gradient match, pose-correction divergence, unknown role rejection,
empirical-marginal passthrough/zero/negative/non-finite, end-to-end
ProximalStepResult consumability).

### Seam 5 — shannon_h2_loss + contest_rate_distortion_system constants
**Status**: clean. `CONTEST_RAW_VIDEO_BYTES = 37,545,489` aligns with the
shannon_h2_loss implicit assumption (rate = h0_bits × n_symbols / 8). The
PR106 anchor (1.015× H0 ratio) reproduces under both modules' arithmetic.
No drift found.

### Seam 6 — Test-coverage gaps
**Status**: clean. Every cathedral module has a dedicated test file:
- codec_pipeline.py → test_codec_pipeline.py
- codec_pipeline_apogee_int.py → test_codec_pipeline_apogee_int.py
- codec_pipeline_full_stack.py → test_codec_pipeline_full_stack.py
- codec_pipeline_joint_admm.py → test_codec_pipeline_joint_admm.py
- codec_pipeline_sensitivity.py → test_codec_pipeline_sensitivity.py
- codec_pipeline_deltaepszeta_callback.py → test_codec_pipeline_deltaepszeta_callback.py
- codec_pipeline_mask.py → test_codec_pipeline_mask.py
- contest_rate_distortion_system.py → test_contest_rate_distortion_system.py + test_contest_rate_distortion_theorems.py
- shannon_h2_loss.py → test_shannon_h2_loss.py
- joint_admm_coordinator.py → test_joint_admm_coordinator.py
- run_cathedral_autopilot.py → test_run_cathedral_autopilot.py
- run_bilevel_optimization.py → test_run_bilevel_optimization.py
- contest_score_gap_decomposition.py → test_contest_score_gap_decomposition.py

### Seam 7 — Dispatch playbook ↔ lane registry sync
**Status**: clean. The `pr103_pr106_standalone` playbook checks the lane
claim presence at `.omx/state/active_lane_dispatch_claims.md` (verified;
the v2 fix already converted the silent-warn-and-proceed pattern to an
exit-non-zero gate at `deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh:105-117`).
Other playbooks (`top3_replay`, `apogee_int6_int7`) are out of scope.
Lane registry sync is not a v3 finding.

### Seam 8 — Bilevel driver assumptions
**Finding 3 (HIGH)**: /tmp paths in test fixtures persist into atom ledger
**Status**: FIXED — commit `8040d7e2`

The `test_atom_ledger_appends_jsonl` and
`test_atom_ledger_marks_cpu_prep_when_no_score` fixtures passed
`path="/tmp/fake"` / `path="/tmp/cpu"` into `SubstrateCandidate(...)`.
The `append_to_atom_ledger` writes that string into
`experiments/results/bilevel_atom_ledger.jsonl` — a direct violation of
CLAUDE.md "Forbidden /tmp paths in any persisted artifact". Tests are
the authoritative source for what the canonical contract permits;
implicitly blessing /tmp here would have leaked into production usage.

**Fix**: replaced fixture paths with `tmp_path / "fake_substrate.pt"`
under pytest's own scratch dir; both fixtures now also assert the persisted
`substrate_path` does NOT start with `/tmp/`.

**Tests**: 1 new (`test_real_atom_ledger_has_no_tmp_paths` — scans the
real repo's atom ledger; trivially passes on fresh checkouts).

### Additional findings (not from the 8-seam list)

**Finding 4 (MEDIUM)**: silent zero-pose back-solve
**Status**: FIXED — commit `f63e69ef`

`tools/contest_score_gap_decomposition._back_solve_pose` silently returned
0.0 when `seg_term + rate_term > score`, producing misleading gap-decomposition
tables where pose_term=0 looked like a measurement instead of a guess.

**Fix**: emit a `UserWarning` identifying the operating point + reason; the
canonical `seg=0.00067082` case still computes a positive pose value with
no warning.

**Tests**: 2 new (underdetermined warns; canonical case is silent).

**Finding 5 (LOW)**: dead-comment about `context['score_marginals']`
**Status**: FIXED — commit `2d0ae720`

`Op_GammaJointADMM.encode` had a comment claiming
"Caller-supplied real marginals may override per-stream via
context['score_marginals']" but the code never read that key — a
dead-comment that the dead-flag-wiring linter pattern would flag.

**Fix**: the wrap now honors a `Mapping[tensor_name, float]` passed via
`context['score_marginals']`. Missing keys fall back to the tiny non-zero
default. Type / sign-convention guards reject non-Mapping inputs, negative
marginals, and non-finite marginals. Pairs cleanly with the
`joint_admm_marginal_for_stream` adapter.

**Tests**: 4 new (per-tensor accept + decode roundtrip, non-Mapping reject,
negative reject, non-finite reject).

## Test count delta

| Module | Before v3 | After v3 | Δ |
|---|---:|---:|---:|
| test_codec_pipeline.py | 22 | 25 | +3 |
| test_contest_rate_distortion_system.py | 5 | 11 | +6 |
| test_codec_pipeline_joint_admm.py | 18 | 22 | +4 |
| test_contest_score_gap_decomposition.py | 7 | 9 | +2 |
| test_run_bilevel_optimization.py | 13 | 14 | +1 |
| **Total** | **211** | **227** | **+16** |

All 227 tests passing.

## Files modified

- `/Users/adpena/Projects/pact/src/tac/codec_pipeline.py`
- `/Users/adpena/Projects/pact/src/tac/contest_rate_distortion_system.py`
- `/Users/adpena/Projects/pact/src/tac/codec_pipeline_joint_admm.py`
- `/Users/adpena/Projects/pact/tools/contest_score_gap_decomposition.py`
- `/Users/adpena/Projects/pact/src/tac/tests/test_codec_pipeline.py`
- `/Users/adpena/Projects/pact/src/tac/tests/test_contest_rate_distortion_system.py`
- `/Users/adpena/Projects/pact/src/tac/tests/test_codec_pipeline_joint_admm.py`
- `/Users/adpena/Projects/pact/src/tac/tests/test_contest_score_gap_decomposition.py`
- `/Users/adpena/Projects/pact/src/tac/tests/test_run_bilevel_optimization.py`

## Final verdict

**All integration seams clean, ready for Phase 1 dispatch.**

No CRITICAL findings remain after the v3 pass. The cathedral now has:
- Actionable errors at the encode-state JSON boundary (no more bisecting
  across stacks for opaque TypeErrors).
- A canonical bridge between the contest formula's marginals and the
  Joint-ADMM coordinator's per-stream marginal field, so the dezeta /
  gamma paradigm wraps no longer ad-hoc constants.
- The dezeta paradigm's caller-supplied score_marginals contract honored,
  no longer a dead-comment.
- A UserWarning on underdetermined gap-decomposition back-solves so
  operators don't mistake the zero-pose guess for a measurement.
- /tmp paths extincted from the bilevel atom ledger fixture chain.

The Phase 1 dispatch playbook (PR101 canonical-winner replay) is now
unblocked from a v3 perspective; remaining gates (operator GPU billing
+ lane claim) are external.

Cross-references:
- v1 review: `.omx/research/bug_hunter_review_v1_*.md` (module-level)
- v2 review: `.omx/research/bug_hunter_review_v2_*.md` (per-module fixes)
- 4-way stack composition contract: `.omx/research/four_way_stack_composition_contract_20260507_claude.md`
- Phase 1 deferred dispatch playbook: `scripts/deferred_dispatch_playbook_pr101_canonical_winner_replay_20260507.sh`
