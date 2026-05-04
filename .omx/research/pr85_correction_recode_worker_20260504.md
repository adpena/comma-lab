# PR85 Correction Recode Worker - 2026-05-04

## Scope

Worker owned only:

- `experiments/build_pr85_correction_recode_candidates.py`
- `src/tac/tests/test_build_pr85_correction_recode_candidates.py`
- `.omx/research/pr85_correction_recode_worker_20260504.md`

No GPU, remote dispatch, lane claim, scorer run, or upstream scorer mutation was performed.

## Builder

Added a strict local builder for Dalton's
`decoded_parity_recode_all_correction_streams` plan policy. The builder:

- reads `/tmp/pr85_correction_atom_waterfill_plan.json` when present, otherwise rebuilds the plan in-process;
- verifies the PR85 source single-member archive contract for member `x`;
- verifies static replay-runtime support for the PR85 correction grammars before emitting alternatives;
- decodes canonical semantics for `post`, `shift`, `frac`, `frac2`, `frac3`, `bias`, `region`, and `randmulti`;
- re-encodes only runtime-supported legal forms;
- fails closed on semantic mismatch, parser mismatch, non-deterministic ZIP bytes, unexpected archive members, source-plan mismatch, or missing runtime support;
- writes candidate archives only when the archive-level byte delta is negative.

## Real PR85 Screen

Command:

```bash
.venv/bin/python experiments/build_pr85_correction_recode_candidates.py --out-dir /tmp/pr85_correction_recodes_20260504_codex --max-archive-candidates 8
```

Source archive:

- path: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- bytes: `236328`
- SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- member `x` bytes: `236228`
- member `x` SHA-256: `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`

Output:

- summary: `/tmp/pr85_correction_recodes_20260504_codex/candidate_summary.json`
- summary SHA-256: `0fe8d52b381a425cacfb2f7b882ffd5dc4ca1ce8e3c7fc1ad03850de27abecc3`
- archive candidates: `0`
- best archive-level byte delta: `0`
- exact eval unlocked: `false`
- result class: `exact_local_negative_no_byte_winning_recode`

All eight correction streams selected `source_exact` as the best runtime-supported decoded-parity-preserving representation. No byte-winning subset existed, so no candidate archive was written.

## Verification

Commands:

```bash
python3 -m py_compile experiments/build_pr85_correction_recode_candidates.py src/tac/tests/test_build_pr85_correction_recode_candidates.py
.venv/bin/python -m py_compile experiments/build_pr85_correction_recode_candidates.py src/tac/tests/test_build_pr85_correction_recode_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr85_correction_recode_candidates.py -q
```

Results:

- `py_compile`: passed
- focused pytest: `4 passed in 0.72s`

Synthetic tests cover:

- one decoded-parity byte-winning recode that writes an eligible local archive;
- one semantic-mismatch fail-closed path;
- one no-byte-win negative JSON path with no archive output;
- one real PR85 smoke that emits either a byte-winning candidate or an exact local negative.

## Promotion State

This lane did not unlock exact CUDA auth eval because it produced no byte-winning PR85 archive candidate. If future PR85/runtime artifacts change, the builder can be rerun; any emitted candidate manifest will still require a Level-2 lane claim before exact eval dispatch.
