# ddm_tk1 checkpoints

## Commands

- `.venv/bin/python -m py_compile experiments/ddm_tk1_semantic_stream_race.py experiments/test_ddm_tk1_semantic_stream_race.py`
- `.venv/bin/python experiments/ddm_tk1_semantic_stream_race.py --self-test`
- `.venv/bin/python -m pytest experiments/test_ddm_tk1_semantic_stream_race.py -q`
- `.venv/bin/python experiments/ddm_tk1_semantic_stream_race.py --n 8 --estimate-only --receipt-dir /Volumes/VertigoDataTier/pact/ddm_tk1_20260806/smoke_receipts --ssd-dir /Volumes/VertigoDataTier/pact/ddm_tk1_20260806/smoke_frames`
- `.venv/bin/python experiments/ddm_tk1_semantic_stream_race.py --estimate-only --receipt-dir .omx/research/ddm_tk1_20260806 --ssd-dir /Volumes/VertigoDataTier/pact/ddm_tk1_20260806`
- Exact learned-prior addendum commands decoded/re-encoded the full tq1c frame and encoded/decoded the full GT frame through `experiments.ddm_tk1_semantic_stream_race`.
- `.venv/bin/python tools/review_tracker.py scan`
- `.venv/bin/python tools/review_tracker.py mark-file experiments/ddm_tk1_semantic_stream_race.py --status reviewed --reviewer codex --pass tk1-pass1`
- `.venv/bin/python tools/review_tracker.py mark-file experiments/test_ddm_tk1_semantic_stream_race.py --status reviewed --reviewer codex --pass tk1-pass1`
- `.venv/bin/python tools/review_tracker.py mark-file experiments/ddm_tk1_semantic_stream_race.py --status reviewed --reviewer codex --pass tk1-pass2`
- `.venv/bin/python tools/review_tracker.py mark-file experiments/test_ddm_tk1_semantic_stream_race.py --status reviewed --reviewer codex --pass tk1-pass2`

## Validation Results

- `py_compile`: pass.
- `--self-test`: `{"schema": "ddm_tk1_semantic_stream_race.v1", "self_test": "ok"}`.
- Focused pytest: 4 passed.
- n8 real-source smoke: tq1c best 2872 B KT, GT best 3332 B KT.
- n600 scorer-free run: tq1c best 142001 B KT, GT best 173617 B KT.
- tq1c exact learned-prior frame: 700111 B, decode equality true, canonical
  re-encode equality true.
- GT exact learned-prior frame: 713345 B, decode equality true, canonical
  re-encode equality true.

## Source Checkpoints

- tq1c parent argmax raw sha:
  `a7dd6f4271eedfa877f6499348de5f9dae2d97311f9e98f4f534908eb66e044e`.
- tq1c parent argmax file sha:
  `764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6`.
- tq1c batch checkpoint digest checks: 38/38 pass, no mismatches.
- GT `lstars` raw sha:
  `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557`.
- GT `lstars` file sha:
  `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d`.

## Artifact Locations

- Durable receipt directory: `.omx/research/ddm_tk1_20260806/`.
- SSD exact learned frames: `/Volumes/VertigoDataTier/pact/ddm_tk1_20260806/`.
- No `/tmp` evidence path is used.

## Pointer Delta

No exact composed archive was built, no scorer was fired, and no contest CPU/CUDA
row was produced. Pointer remains unmoved.

