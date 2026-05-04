# PR85 QMA9 Alt-Grammar Worker - 2026-05-04

Worker: QMA9-AltGrammar

## Scope

Local-only PR85 QMA9 alternate grammar/table-reduction screen after runtime-supported trim screens failed. No GPU, remote dispatch, scorer invocation, lane claim, or score claim was performed.

## Inputs

- Source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
  - bytes: `236328`
  - sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Source QMA9 mask segment:
  - bytes: `159011`
  - sha256: `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
  - decoded token sha256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- Prior runtime-supported trim screen:
  - `experiments/results/pr85_qma9_native_grammar_candidates_20260504_orchestrator/candidate_summary.json`
  - result: no byte-positive runtime-supported trim candidate.
- Opportunity matrix:
  - `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final2/pr85_full_stack_opportunity_matrix.json`
  - next QMA9 target: native run grammar/table-reduction, not suffix trim.

## Artifact

- Summary: `experiments/results/pr85_qma9_alt_grammar_candidates_20260504/candidate_summary.json`
- Payloads: `experiments/results/pr85_qma9_alt_grammar_candidates_20260504/payloads/`

## Result

Full-stream deterministic replay-codec screen produced no byte-positive alternate grammar:

| mode | magic | bytes | delta vs QMA9 | sha256 |
| --- | --- | ---: | ---: | --- |
| adaptive9bin | QMA9 | 159011 | 0 | `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179` |
| adaptive9up2left2 | QMA9 | 161034 | +2023 | `da17f28d5226d71ebdc060e494ce96167bbe3d3627bde4412100fc88895de173` |
| adaptive8prpdup2 | QMA8 | 163488 | +4477 | `a23d02e7bbfa04e17911a4bf28c16bd6a658d139b1a51e67e96b4a4923f1b2c2` |
| adaptive6pr | QMA6 | 173776 | +14765 | `e39c86a70393803402774a564fbf8f657ae0c17eee15f75c2c912310240ea934` |

The complete mode table is in the summary JSON. The closest alternate was `adaptive9up2left2`, but it is `+2023` bytes versus source QMA9. Rate-only economics if components were unchanged: `+0.0013470326621661526` score points, so it is not a candidate.

## Runtime Custody

Current live runtime contract is `QMA9 adaptive9bin` only in `submissions/robust_current/range_mask_codec.cpp`. Non-reference QMA9 modes collide with the current `QMA9` magic and would be misdecoded as `adaptive9bin`; QMA6/QMA7/QMA8 modes require live runtime magic and inflate/unpack detection changes.

Fail-closed blockers:

- `no_byte_positive_alt_grammar_candidate`
- `no_runtime_supported_byte_positive_alt_grammar_candidate`
- `qma9_adaptive9bin_remains_smallest_observed_full_stream`
- `robust_current_runtime_accepts_only_qma9_adaptive9bin`

Dispatch status: `dispatch_unlocked=false`.

## Verification

- `.venv/bin/python -m py_compile src/tac/qma9_alt_grammar.py experiments/build_pr85_qma9_alt_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_alt_grammar_candidates.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr85_qma9_alt_grammar_candidates.py -q`
- `.venv/bin/python experiments/build_pr85_qma9_alt_grammar_candidates.py --out-dir experiments/results/pr85_qma9_alt_grammar_candidates_20260504`
