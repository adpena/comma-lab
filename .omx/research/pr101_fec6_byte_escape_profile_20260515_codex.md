# PR101 FEC6 Byte Escape Profile

timestamp_utc: 2026-05-15T00:00:00Z

score_claim: false
dispatch_attempted: false
ready_for_exact_eval_dispatch: false

## Question

Operator asked whether the `0.192` result is confirmed legitimate, whether the
film-grain / selector / dynamic water-bucket path was fully engineered, and
whether the remaining local basin has been exhausted.

## Axis answer

The near-`0.192` artifact is legitimate on `[contest-CPU]`, not on
`[contest-CUDA]`.

- archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- archive bytes: `178517`
- archive sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- `[contest-CPU]` exact score: `0.1920513168811056`
- `[contest-CUDA]` paired exact score: `0.22621002169349796`
- strict byte gap to `<0.192` on `[contest-CPU]` if components are unchanged: `78` bytes

This cannot be promoted as a CUDA frontier result. It is useful CPU-axis
leaderboard reproduction and CPU/CUDA drift evidence.

## New artifact

Generated a deterministic audit:

```bash
.venv/bin/python tools/profile_pr101_fec6_escape_routes.py
```

Outputs:

- `experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.json`
- `experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.md`

The tool is planning/audit only. It does not mutate archives, load scorers,
dispatch evals, or claim score movement.

## Byte-only result

| surface | current bytes | best/floor bytes | realistic saving | verdict |
|---|---:|---:|---:|---|
| PR101 decoder Brotli streams | 162164 | 162162 | 2 | bounded recompress only |
| PR101 latent raw-LZMA | 15387 | 15387 | 0 | filter sweep saturated |
| PR101 latent sidecar | 607 | 603 | 4 | near entropy floor |
| FEC6 selector payload | 249 | 241 | 8 | selector entropy saturated |
| FP11 wrapper | 10 | 0 | 10 | hardcode-only, insufficient |

Same-frame byte-only realistic upper bound is about `16` bytes. That is below
the `78` byte strict gap to `<0.192` on `[contest-CPU]`.

## Engineering verdict

The packet engineering is correct for what it claims:

- selector bytes are archive-charged;
- runtime consumes the selector;
- FEC6 has same-runtime full-frame parity versus the matching FEC3 K16 source;
- exact CPU and exact CUDA results exist for the same archive SHA;
- promotion blockers remain explicit.

The family is not globally exhausted:

- selector/waterfill construction was CPU/MPS/proxy selected, not CUDA-in-loop;
- the waterfill is not yet a true charged KKT/rate-distortion solver;
- CUDA component rows and a charged selector objective are still the right next
  mechanism if this family is reopened;
- new trained substrates / PR95-PR101 control-arm divergence remain open.

## Decision

Retire additional same-frame byte-only PR101/FEC6 polishing unless a new tool
shows at least `78` saved bytes with full-frame parity. Continue only through:

1. CUDA-in-loop per-pair/per-mode component table;
2. charged rate-distortion selector rebuild from those CUDA rows;
3. exact CPU and exact CUDA eval on the resulting byte-closed packet;
4. or a larger substrate change that moves components, not just bytes.

## Verification

```bash
.venv/bin/python -m ruff check tools/profile_pr101_fec6_escape_routes.py
.venv/bin/python tools/profile_pr101_fec6_escape_routes.py
```

Ruff passed. The profile command regenerated the JSON and Markdown artifacts
listed above.
