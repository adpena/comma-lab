# PR79 Next Byte Candidates Worker - 2026-05-03

Scope: local-only candidate generation from public PR79/PR77 archive anatomy.
No remote dispatch was performed. No CPU/MPS/scorer result is claimed.

Owned implementation:

- `experiments/build_pr79_next_byte_candidates.py`
- `src/tac/tests/test_build_pr79_next_byte_candidates.py`
- `experiments/results/pr79_next_byte_candidates_20260503_worker/`

Inputs:

- PR79 public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`
  bytes `277388`, SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`.
- PR77 public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip`
  bytes `276551`, SHA
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`.
- Mask-body source:
  `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53`
  from `experiments/results/pr79_mask_body_reduction_20260503_worker/candidate_matrix.json`.
- Current S2 frontier anchor:
  bytes `277321`, SHA
  `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`,
  score `0.31453355357318635`. This tool did not rerun eval.

Candidate matrix:

- `experiments/results/pr79_next_byte_candidates_20260503_worker/candidate_matrix.json`

Top local byte rows:

| candidate | bytes | delta vs S2 frontier | changed decoded members vs PR79 | dispatch flag |
|---|---:|---:|---|---|
| `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53__pr77_public_raw4_action_wire_br__brotli_rpk1_flatpack` | `245979` | `-31342` | `masks.mkv`, `seg_tile_actions` | exact-screenable after lane claim |
| `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53__pr79_public_raw4_action_wire_br__brotli_rpk1_flatpack` | `246788` | `-30533` | `masks.mkv` | exact-screenable after lane claim |
| `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53__pr77_public_raw4_action_wire_br__stored_rpk1` | `258501` | `-18820` | `masks.mkv`, `seg_tile_actions` | exact-screenable after lane claim |
| `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53__pr79_public_raw4_action_wire_br__stored_rpk1` | `259339` | `-17982` | `masks.mkv` | exact-screenable after lane claim |
| `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53__pr79_decoded_actions_bin__stored_rpk1_control` | `260866` | `-16455` | `masks.mkv` | exact-screenable after lane claim |

Important guards:

- PR79 renderer and PR79 QP1 pose are decoded-byte preserved in every ranked
  non-control row.
- `source_pr79_noop_control` is marked as a payload-identical container re-emit
  no-op and is not dispatchable.
- Direct PR79 S2 action-wire-in-RPK1 rows are marked non-dispatchable: the S2
  fixed-slice wire is not a direct `seg_tile_actions.br` inflate-loader payload.
- Score estimates are unchanged-component byte estimates versus the current S2
  frontier only. They are not score evidence.
- Every emitted archive records deterministic rebuild SHA-256, changed members,
  no-op status, action-loader compatibility, and exact CUDA gate text.

Verification:

```bash
.venv/bin/python -m py_compile experiments/build_pr79_next_byte_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr79_next_byte_candidates.py
.venv/bin/python experiments/build_pr79_next_byte_candidates.py \
  --output-dir experiments/results/pr79_next_byte_candidates_20260503_worker \
  --force
```

Result: focused pytest `2 passed`. No remote job was dispatched.

Dispatch rule:

Before any exact CUDA eval, claim a non-conflicting lane with
`tools/claim_lane_dispatch.py claim ...`, then run the identical archive bytes
through `archive.zip -> inflate.sh -> upstream/evaluate.py` via
`experiments/contest_auth_eval.py --device cuda` and preserve
`contest_auth_eval.json`, runtime tree hash, archive SHA, bytes, and component
gates.
