# HNeRV HDC2 Combined Entropy Manifest - 2026-05-06

## Summary

2026-05-08 supersession: the current exact local HNeRV rate frontier is now
PR103-on-PR106 at strict formula score `0.2089810755823297`, `185578` bytes,
SHA-256 `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`.
The PR106x row below is preserved as the historical source/control substrate
for this HDC2 analysis, not the active anchor.

The superseded local HNeRV rate frontier used by this manifest was
`PR106x-lowlevel-brotli`, score `0.20935073680571203`, archive bytes
`186080`, archive SHA-256
`b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`.

The next entropy target in this historical PR106x substrate was the
`decoder_packed_brotli` section. Any current rate-only successor must also
beat the PR103-on-PR106 `185578`-byte archive floor or stack with a
scorer-changing packet. The HDC2
global previous-symbol stream is raw-equal and deterministic, but still
rate-negative until both model overhead and payload gap are reduced together.

## Artifact

- JSON:
  `experiments/results/hnerv_entropy_packet_discovery_20260506_codex/hdc2_combined_entropy_reduction_manifest.json`
- Markdown:
  `experiments/results/hnerv_entropy_packet_discovery_20260506_codex/hdc2_combined_entropy_reduction_manifest.md`
- Tool:
  `tools/build_hnerv_hdc2_combined_entropy_manifest.py`
- Library:
  `src/tac/hnerv_hdc2_combined_entropy.py`

## Byte Accounting

- historical PR106x frontier section bytes: `170127`
- current HDC2 replacement bytes: `221381`
- net delta now: `+51254`
- model overhead target: `40840`
- payload entropy gap target: `23979`
- net after removing model overhead only: `+10414`
- minimum payload reduction after zero model overhead: `10415`
- net after combined known targets: `-13565`
- projected rate score delta after combined known targets:
  `-0.009032376699102255`

## Interpretation

Model-overhead removal alone is insufficient. The next implementation must
combine:

1. actual static context metadata elision or shared codebook implementation,
2. actual range-payload entropy-gap reduction,
3. candidate archive manifest,
4. runtime-tree parity,
5. strict pre-submission compliance,
6. meta-Lagrangian atom export.

This is planning evidence only: `score_claim=false`,
`dispatch_attempted=false`, and `ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_hdc2_combined_entropy.py src/tac/tests/test_hnerv_entropy_candidate_packet.py src/tac/tests/test_hnerv_frontier_entropy_ranking.py -q`
  - `17 passed`
- `.venv/bin/ruff check src/tac/hnerv_hdc2_combined_entropy.py tools/build_hnerv_hdc2_combined_entropy_manifest.py src/tac/tests/test_hnerv_hdc2_combined_entropy.py`
  - passed
