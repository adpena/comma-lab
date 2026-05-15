# PR101 FEC6 Wrapper Profile

date: 2026-05-15
research_only: true
score_claim: false
dispatch_attempted: false

## Purpose

Add a deterministic no-network forensic helper for the PR101/FEC6 CPU-axis
near-frontier packet. This is byte and parser evidence only. It does not
inflate frames, dispatch auth eval, or claim score movement.

## Command

```bash
.venv/bin/python tools/pr101_fec6_wrapper_profile.py \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --json-out /tmp/pr101_fec6_wrapper_profile.json
```

## Current Facts

- Archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- Archive bytes: `178517`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- ZIP member: `x`, stored, `178417` bytes
- Outer wrapper: `FP11`
- Source payload bytes: `178158`
- Selector payload: `FEC6`, `249` bytes, SHA-256 `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca`
- Selector pairs: `600`
- Selector code bits: `1944`
- Selector average bits per pair: `3.24`
- Selector entropy floor: `241` bytes, current payload gap `8` bytes
- Source PR101 payload match: `true`

## Use

Use this helper before further PR101 selector or PacketIR edits to verify:

- wrapper offsets and section boundaries,
- source-payload parity against the public PR101 archive,
- selector histogram and fixed-Huffman bitstream integrity,
- whether a claimed byte saving is real payload movement rather than ZIP or
  manifest churn.

Promotion still requires the existing paired exact auth-eval custody. The
current paired artifact remains CPU-positive but CUDA-missing because the CUDA
gap is component dominated, especially PoseNet.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_pr101_fec6_wrapper_profile.py
.venv/bin/ruff check tools/pr101_fec6_wrapper_profile.py src/tac/tests/test_pr101_fec6_wrapper_profile.py
```

Result: `4 passed`; ruff passed.
