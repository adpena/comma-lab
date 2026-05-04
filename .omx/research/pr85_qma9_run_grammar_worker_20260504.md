# PR85 QMA9 Run-Grammar Worker - 2026-05-04

## Scope

Local-only PR85 QMA9 QRG1 row-run grammar byte screen. No scorer, GPU eval,
remote dispatch, lane claim, runtime edit, or score claim was performed.

Owned files:

- `src/tac/qma9_run_grammar.py`
- `experiments/build_pr85_qma9_run_grammar_candidates.py`
- `src/tac/tests/test_build_pr85_qma9_run_grammar_candidates.py`
- `.omx/research/pr85_qma9_run_grammar_worker_20260504.md`

## Anchor

- Source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- Source bytes: `236328`
- Source SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 QMA9 mask segment bytes: `159011`
- PR85 QMA9 mask segment SHA-256: `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- Expected decoded token bytes: `117964800`
- Expected decoded token SHA-256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`

## Artifact

- Summary: `experiments/results/pr85_qma9_run_grammar_candidates_20260504_worker/candidate_summary.json`

The builder wrote only the root summary JSON. It did not write candidate
archives or QRG1 payload files by default.

## Result

The QRG1 row-run compiler screened seven deterministic local modes:

- `row_rle_zlib9`
- `row_rle_bz2_9`
- `row_rle_lzma6`
- `row_copy_up_rle_zlib9`
- `row_copy_up_rle_bz2_9`
- `row_copy_up_prev_rle_bz2_9`
- `row_copy_up_prev_rle_lzma6`

Best local payload:

- mode: `row_rle_lzma6`
- bytes: `462176`
- delta vs PR85 QMA9 `159011B`: `+303165`
- SHA-256: `7a0b5c716a809b2c0bb80ef236cee8014df6cf6112ede942c871f22b6fe25d2b`

Candidate counts:

- `candidate_count`: `7`
- `byte_positive_candidate_count`: `0`
- `runtime_supported_byte_positive_candidate_count`: `0`
- `dispatch_unlocked`: `false`

No byte-positive candidate exists from this screen. The exact blockers in the
summary are:

- `dispatch_locked_until_qrg1_runtime_output_parity_and_fresh_lane_claim`
- `no_byte_positive_qrg1_row_run_candidate`
- `qma9_adaptive9bin_beats_screened_row_run_payloads`
- `robust_current_runtime_does_not_accept_qrg1_run_grammar`

## Runtime Contract

Current `robust_current` runtime accepts `QMA9` adaptive9bin mask payloads, not
the planning-only `QRG1` row-run grammar. If a future QRG1 variant becomes
byte-positive, the required runtime changes are:

- add a reviewed QRG1 row-run decoder to the robust_current inflate path
- extend mask payload detection to admit QRG1 without changing PR85 fixed-slice custody
- preserve PR85 QMA9 token storage order, `600x512x384` source shape, transpose behavior, and `_half_frame_only` metadata
- add raw-token SHA parity tests against the exact PR85 token source before any exact eval
- add runtime output parity against the baseline PR85 inflate masks tensor before any exact eval
- record the updated inflate runtime tree SHA in `contest_auth_eval` provenance
- open a fresh Level-2 dispatch claim before any later CUDA exact-eval run

## Verification

Commands:

```bash
.venv/bin/python -m py_compile src/tac/qma9_run_grammar.py experiments/build_pr85_qma9_run_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_run_grammar_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr85_qma9_run_grammar_candidates.py -q
.venv/bin/python experiments/build_pr85_qma9_run_grammar_candidates.py --out-dir experiments/results/pr85_qma9_run_grammar_candidates_20260504_worker
```

Results:

- `py_compile`: passed
- focused pytest: `3 passed`
- local PR85 screen: completed, wrote `candidate_summary.json`
