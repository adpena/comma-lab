# PR79 S1 Lossless Action Repack - 2026-05-03

## Scope

Ownership: PR79 `seg_tile_actions` stream grammar/compression only. No
remote, GPU, training, or eval job was submitted.

Source archive:
`experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`

- bytes: 277388
- sha256: `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`
- source `seg_tile_actions` Brotli bytes: 1162
- source decoded action bytes: 2688
- source decoded action sha256:
  `a48bd4e49f8928158756610fd8094e8fb1611a2040121611055266f840faf13f`

## Result

Implemented an `S1` split action grammar:

```text
b"S1" + group_count + (tile_delta, count)* + pair_deltas* + actions*
```

The split stream preserves decoded raw4 action records exactly while giving
Brotli separate local structure for tile metadata, pair deltas, and action
ids. Runtime support is in
`submissions/robust_current/unpack_renderer_payload.py`; preflight validation
support is in `src/tac/submission_archive.py`.

Best candidate [evidence:experiments/results/pr79_action_lossless_repack_20260503_codex/pr79_s1_fixed_lossless_actions/manifest.json]:

- archive: `experiments/results/pr79_action_lossless_repack_20260503_codex/pr79_s1_fixed_lossless_actions/archive.zip`
- bytes: 277347
- sha256: `d61527a43218b87871fd869dcb92b6875e99482bc28a8fdaf879caf6d8cfc4eb`
- delta vs PR79: -41 bytes
- `S1` action Brotli bytes: 1121
- `S1` action Brotli sha256:
  `16c7fd9b49a50580342a18108c0093b087828635bf83678e6ac1103e7d9f057a`
- decoded action parity: true
- non-action streams preserved: true
- no-op status: `decoded_action_semantics_preserved_action_bytes_changed`
- evidence grade: empirical lossless byte screen

Secondary self-describing P3 candidate
[evidence:experiments/results/pr79_action_lossless_repack_20260503_codex/pr79_s1_p3_lossless_actions/manifest.json]:

- bytes: 277357
- sha256: `a0d20df571a0373d4f20ba44ae774a53a25bfa476886fbb69957c0a2ac60013f`
- delta vs PR79: -31 bytes
- decoded action parity: true
- non-action streams preserved: true

## Dispatch Gate

This is not score evidence. If promoted, first claim a non-conflicting lane
with `tools/claim_lane_dispatch.py claim ...`, then run exact T4-equivalent
CUDA auth eval on the exact archive SHA/bytes via:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
experiments/contest_auth_eval.py --device cuda
```

The eval command must pin expected archive SHA and size and preserve
`contest_auth_eval.json`, runtime tree hash, component gates, manifest, and
logs.

## Verification

Commands:

```text
.venv/bin/python -m py_compile submissions/robust_current/unpack_renderer_payload.py src/tac/submission_archive.py experiments/profile_pr75_minp_archive.py experiments/build_pr79_action_lossless_repack_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr79_action_lossless_repack_candidates.py src/tac/tests/test_build_pr79_action_subset_candidates.py src/tac/tests/test_profile_pr75_minp_archive.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py src/tac/tests/test_seg_tile_actions_preflight.py -q
.venv/bin/python experiments/build_pr79_action_lossless_repack_candidates.py --force
```

Observed: 28 passed; builder emitted the byte matrix at
`experiments/results/pr79_action_lossless_repack_20260503_codex/candidate_matrix.json`.
