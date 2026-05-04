# C082 PR75 Lossless Repack Hardening Addendum - 2026-05-03

Scope: local-only hardening and validation of the C082/PR75 lossless repack
path. No remote GPU dispatch was performed.

Hardened parser contract:

- `P6` delta-varint action records now reject truncated varints, overlong
  varints, noncanonical ULEB encodings, partial records, trailing bytes, and
  pair indices outside the PR75 action domain.
- The parser remains standalone in
  `submissions/robust_current/unpack_renderer_payload.py` and still emits the
  same decoded runtime `seg_tile_actions.bin` stream for canonical P6 payloads.

Manifest contract:

- Builder manifests now preserve the original `decoded_stream_parity=true`
  flag and add `decoded_stream_parity_detail` with source/candidate decoded
  member byte counts and SHA-256s.
- Builder manifests now record `noop`, `noop_status`, `source_preserving`, and
  `source_preservation`. The current best P6 candidate is explicitly not a
  byte-identical no-op; it is a decoded-stream-preserving repack.

Current best local artifact:

- candidate: `c082_p6_delta_varint_actions_stream_resweep`
- archive: `experiments/results/c082_pr75_lossless_repack_20260503_worker/c082_p6_delta_varint_actions_stream_resweep/archive.zip`
- bytes: `276394`
- sha256: `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
- source archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z/archive.zip`
- source bytes: `276460`
- source sha256: `851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9`
- local evidence grade: `empirical_lossless_byte_transform`
- score claim: `false`

Exact-eval priority when dispatch is allowed:

1. `c082_p6_delta_varint_actions_stream_resweep`: highest byte win, decoded
   streams preserved locally, new P6 parser path.
2. `c082_p6_delta_varint_actions`: isolates the action-codec change from the
   mask/pose Brotli resweep.
3. `c082_p3_stream_resweep`: fallback with no runtime format change.

Failure modes to check before any promotion:

- exact CUDA auth eval must run on the exact archive bytes through
  `archive.zip -> inflate.sh -> upstream/evaluate.py`, preferably via
  `experiments/contest_auth_eval.py --device cuda`;
- runtime tree hash must be recorded because identical archive bytes can score
  differently under runtime changes;
- PoseNet/SegNet component gates can still reject a rate-only byte win;
- any malformed, noncanonical, truncated, or source-preserving/no-op ambiguous
  action stream must fail closed before scoring;
- promotion remains blocked if manifest decoded parity, no-op status, payload
  SHA, or runtime unpack summary is missing or mismatched.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_build_pr75_lossless_repack_candidates.py -q` -> `3 passed`
- `.venv/bin/python -m py_compile submissions/robust_current/unpack_renderer_payload.py experiments/build_pr75_lossless_repack_candidates.py src/tac/tests/test_build_pr75_lossless_repack_candidates.py`
- `.venv/bin/python submissions/robust_current/unpack_renderer_payload.py experiments/results/c082_pr75_lossless_repack_20260503_worker/smoke_unpack_best --summary-json experiments/results/c082_pr75_lossless_repack_20260503_worker/smoke_unpack_best/renderer_payload_unpack_summary.json`
- Rebuilt local candidate matrix with `--force`; the `276394`-byte best P6
  archive remained byte-identical with SHA-256
  `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`.
