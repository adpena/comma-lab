# PR85 QH0 Record Serializer Worker - 2026-05-04

## Scope

Local-only serializer/repacker for the PR85 QH0/QM0 model segment. No remote
or GPU dispatch was performed, and no score claim is made.

## Source Custody

- source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- source archive bytes: `236328`
- source archive SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- source model segment bytes: `57074`
- source model segment SHA-256: `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc`
- decoded QH0 bytes: `61590`
- decoded QH0 SHA-256: `503d666ec4c1b3e0ce09fa9dcb963b049de185eac0ddd82318578a7732439ed4`

## Implemented Artifact

- serializer: `src/tac/qh0_record_serializer.py`
- builder: `experiments/build_pr85_qh0_serializer_candidates.py`
- tests: `src/tac/tests/test_qh0_record_serializer.py`
- local smoke summary: `experiments/results/pr85_qh0_serializer_candidates_20260504_codex/candidate_summary.json`

The serializer parses the reviewed runtime record order and can re-emit:

- `QH0` canonical bytes with high/low FP4 and even/odd fp16 byte split.
- `QM0` direct bytes with the same tensor records and no split transform.

Both forms are decoded through `tac.qh0_renderer_codec.decode_qh0_state_dict`
for tensor parity proof before any archive is eligible.

## Local Screen Result

Best screened record:

- candidate id: `qh0_canonical_source_passthrough`
- model bytes: `57074`
- byte delta vs source model segment: `0`
- decoded tensor parity: `true`
- dispatch unlocked: `false`
- blocker class: `no_real_byte_win`

Best non-source recompressions were byte-negative:

- `qh0_canonical_brq10_lg18`: `57165` bytes, `+91`
- `qh0_canonical_brq11_lg18`: `57227` bytes, `+153`
- `qm0_direct_brq10_lg18`: `57353` bytes, `+279`

No candidate archive was emitted because every runtime-compatible serialized
model stream was byte-neutral or byte-negative. This is a local exact negative
for the current deterministic QH0/QM0 grammar screen, not a score result and
not a method kill.

## Runtime Compatibility

The public PR85 replay inflater accepts both `QH0` and `QM0` model payloads in
the single-member `x` bundle. The current `robust_current` renderer member
loader also advertises `QH0`/`QM0`, but its packed-payload unpacker does not
yet parse PR85 single-member `x` directly. That robust-current gap did not
block this local PR85 replay-path screen, and no runtime edits were made.

Minimal robust-current runtime implementation if this lane later finds a byte
win: add a no-sidecar PR85 single-member `x` unpacker that materializes the
existing `masks.qma9`, `renderer.bin`, `optimized_poses.bin`, and `qpost.bin`
contracts without changing scorer/upstream files.

## Verification

- `.venv/bin/python -m py_compile src/tac/qh0_record_serializer.py experiments/build_pr85_qh0_serializer_candidates.py src/tac/tests/test_qh0_record_serializer.py`
- `.venv/bin/python -m pytest src/tac/tests/test_qh0_record_serializer.py -q`
- `.venv/bin/python experiments/build_pr85_qh0_serializer_candidates.py`

All focused tests passed. The builder wrote only the local summary JSON and no
candidate archives.
