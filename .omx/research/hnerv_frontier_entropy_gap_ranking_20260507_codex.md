# HNeRV Frontier Entropy Gap Ranking - 2026-05-07 Codex

## Scope

Bounded local review of the HNeRV scorecard, PR106x low-level Brotli repack
manifest, and HDC2 entropy-overhead audit. This pass did not dispatch remote
work, did not claim a lane, and did not make a new score claim.

## Durable Artifacts

- [empirical:experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/frontier_entropy_gap_ranking.json]
  Deterministic JSON ranking from
  `tools/rank_hnerv_frontier_entropy_gaps.py`.
- [empirical:experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/frontier_entropy_gap_ranking.md]
  Human-readable rendering of the same manifest.

## Result

The current exact frontier selected from the local scorecard is
`PR106x-lowlevel-brotli`, score `0.20935073680571203`, archive bytes `186080`,
archive SHA-256
`b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`.

The deterministic next rate-only action is review of the existing exact
lossless Brotli control before promotion:

- source archive bytes: `186231`
- candidate archive bytes: `186080`
- byte delta: `-151`
- raw Brotli equivalence: closed
- dispatch allowed: `false`

The entropy follow-up is not an exact-eval packet. HDC2 maps to the current
frontier `decoder_packed_brotli` section through the PR106x low-level repack
section-diff audit. The current HDC2 stream is byte-negative against the exact
frontier decoder section:

- current frontier decoder section: `170127` bytes
- HDC2 stream bytes: `221381`
- net delta now: `+51254` bytes
- HDC2 model-overhead target alone: still `+10414` bytes after reduction
- HDC2 payload-gap target alone: still `+27275` bytes after reduction
- combined known HDC2 targets: `-13565` bytes only if both are actually closed
  with byte-equivalent artifacts

## Next Local Gate

Build a combined HDC2 overhead-reduction manifest only if it records old/new
stream bytes and proves the replacement section can beat `170126` bytes before
archive work. Exact CUDA remains blocked until byte-different archive custody,
runtime parity, strict preflight, a Level 2 lane claim, and exact auth eval.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_frontier_entropy_ranking.py -q
.venv/bin/ruff check src/tac/hnerv_frontier_entropy_ranking.py src/tac/tests/test_hnerv_frontier_entropy_ranking.py tools/rank_hnerv_frontier_entropy_gaps.py
.venv/bin/python -m py_compile src/tac/hnerv_frontier_entropy_ranking.py tools/rank_hnerv_frontier_entropy_gaps.py
```

Results:

- focused pytest: `2 passed`
- ruff: `All checks passed`
- py_compile: passed
