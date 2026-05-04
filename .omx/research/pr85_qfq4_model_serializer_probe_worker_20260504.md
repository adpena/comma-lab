# PR85 QFQ4 Model Serializer Probe - Worker - 2026-05-04

## Scope

Implement or decisively block a PR90-QFQ4-style model serializer transfer probe
for public PR85. No GPU dispatch was performed.

Owned artifacts:

- `experiments/analyze_or_build_pr85_qfq4_model_serializer_candidate.py`
- `src/tac/tests/test_pr85_qfq4_model_serializer_probe.py`
- `experiments/results/pr85_qfq4_model_serializer_probe_20260504_worker/candidate_summary.json`

## Inputs

- PR85 archive:
  `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
  - bytes: 236328
  - sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 source model segment:
  - bytes: 57074
  - sha256: `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc`
  - decoded QH0 bytes: 61590
  - decoded QH0 sha256: `503d666ec4c1b3e0ce09fa9dcb963b049de185eac0ddd82318578a7732439ed4`
- PR90 source evidence:
  `experiments/results/public_pr90_intake_20260504_worker/payload_probe.json`
  reports a `QFQ4\0` model-body slice of 56385 bytes.
- Worker G planning-only context:
  `pr90_qfq4_style_pr85_model_serializer_probe`, estimated -689 bytes and
  -0.000458776819 rate-score delta, requiring tensor/output parity.

## Probe Result

The local PR85 QFQ4-style lowering is a byte screen win but is not buildable.

Best screened lowering:

- candidate id: `qfq4_pr85_shifted_int8_rows`
- candidate PR85 outer model segment bytes: 56415
- delta vs PR85 source model segment: -659 bytes
- formula-only rate-score delta if components were identical:
  -0.0004388010501075109
- inner QFQ4 Brotli body bytes: 56406
- inner QFQ4 Brotli sha256:
  `e15dd3711ddfbfeb9278f85dac9fef2505ff82d7d88952763051b04440fac952`

Fail-closed blockers:

1. Decoded tensor parity failed.
   - tensor: `frame1_head.block1.film_proj.weight`
   - changed elements: 4726 / 5376
   - max abs diff: 6.103515625e-05
   - cause: PR85 stores this tensor as an int8 row-scale record. The PR90 QFQ4
     row contract reconstructs the special row tensor through an fp16 cast, so
     the source QH0 float32 products are not bit-identical.
2. PR85 no-edit runtime lacks a QFQ4 model loader.
   - `public_pr85_replay_missing_QFQ4_model_loader`
   - `robust_current_missing_QFQ4_renderer_loader`
   - `robust_current_missing_pr85_single_x_unpacker`

No candidate archive was built. Exact eval readiness is false.

## Evidence

- Summary JSON:
  `experiments/results/pr85_qfq4_model_serializer_probe_20260504_worker/candidate_summary.json`
- Focused tests:
  `.venv/bin/python -m pytest src/tac/tests/test_pr85_qfq4_model_serializer_probe.py -q`
  passed 2 tests.
- Compile check:
  `.venv/bin/python -m py_compile experiments/analyze_or_build_pr85_qfq4_model_serializer_candidate.py src/tac/tests/test_pr85_qfq4_model_serializer_probe.py`
  passed.

Evidence grade: empirical/local fail-closed byte and tensor-parity probe.
This is not score evidence and does not unlock GPU dispatch.
