# QMA9 First-Row Context Specialization Byte Screen - 2026-05-03

## Scope

Implemented a deterministic local-only QMF1 prototype for PR81/QMA9 range-mask
first-row specialization. This is a byte-screen artifact only. No GPU eval,
remote dispatch, training, scorer path, or lane claim was used.

The prototype preserves QMA9's base arithmetic model for non-first rows. On
the first row of each frame, it skips the static `up == sentinel` gate and
screens three context specializations:

- `qmf1_first_row_1_skip_static_up_gate_full_context`
- `qmf1_first_row_2_skip_static_up_gate_prev_left_context`
- `qmf1_first_row_3_skip_static_up_gate_left_context`

Every candidate is encoded from raw mask bytes and immediately decoded by the
local QMF1 decoder. Candidates with non-identical raw mask bytes are rejected
before any byte comparison can select them.

## Artifacts

- Profile JSON:
  `experiments/results/qma9_first_row_specialization_20260503_codex/qma9_first_row_specialization_profile.json`
- Byte-screen manifest:
  `experiments/results/qma9_first_row_specialization_20260503_codex/byte_search/qma9_range_mask_byte_search_profile.json`
- Candidate payloads:
  `experiments/results/qma9_first_row_specialization_20260503_codex/byte_search/candidates/`

## Local Screen Result

Command:

```bash
.venv/bin/python experiments/profile_qma9_range_mask_bitstream.py \
  --skip-cpp-full \
  --pure-python-max-pixels 1024 \
  --checkpoint-pixels 0,1,1023 \
  --run-byte-search \
  --byte-search-frames 4 \
  --qmb1-block-widths '' \
  --qmf1-first-row-modes 1,2,3 \
  --output-dir experiments/results/qma9_first_row_specialization_20260503_codex \
  --output-json experiments/results/qma9_first_row_specialization_20260503_codex/qma9_first_row_specialization_profile.json
```

Raw mask source: prefix decode from `archive_range_mask.qma9`.

- Frames screened: 4
- Raw bytes: 786432
- Raw SHA-256:
  `e763e118ec695f667eccda8b987420b167fd2ba263e382ff2d9348c578e51a46`
- Baseline prefix QMA9 bytes: 1221
- Baseline prefix QMA9 SHA-256:
  `fcf143230da1c30885969b7490dba9549397a352bdf347efdfa706f6bcf6ffe3`

Candidate matrix:

| mode_id | bytes | delta_vs_qma9 | payload_sha256 | raw_parity | no_op_status | selectable |
| --- | ---: | ---: | --- | --- | --- | --- |
| qmf1_first_row_1_skip_static_up_gate_full_context | 1224 | +3 | `eb9ac13feb0458de0e06b3ce0786efc7e8115c12fa97b492b8dede5594215731` | true | state_changed | false |
| qmf1_first_row_2_skip_static_up_gate_prev_left_context | 1225 | +4 | `0d0ac243a86cd2f45c853b507d07a81e0f25636accde95b0ec6cbe907d7b9f9f` | true | state_changed | false |
| qmf1_first_row_3_skip_static_up_gate_left_context | 1241 | +20 | `df063a8485eb49995e2e0947339f45de87cee272ac744b2d40102747e9b56f07` | true | state_changed | false |

No candidate is dispatch-selectable from this finite local screen. All QMF1
variants changed the archive-relevant payload state and preserved exact raw
mask bytes, but none beat baseline QMA9 bytes. This is useful negative
evidence: the first-row `up` gate was already cheap enough that skipping it did
not amortize QMF1 header/model-state differences on the tested PR81 prefix.

## Verification

```bash
.venv/bin/python -m py_compile \
  src/tac/qma9_range_mask_contract.py \
  experiments/profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_qma9_range_mask_contract.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py
```

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_qma9_range_mask_contract.py \
  src/tac/tests/test_qma9_range_mask_byte_search.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py \
  -q
```

Result: `19 passed`.
