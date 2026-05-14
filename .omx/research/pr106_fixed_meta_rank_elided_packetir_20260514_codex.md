# PR106 Fixed-Meta Rank-Elided PacketIR Closure - 2026-05-14

## Scope

This is a byte-grammar cleanup artifact, not a frontier score claim. It closes
the interrupted PR106/R2 PR101-sidecar format `0x05` work so follow-on work can
move to divergence lanes without leaving a half-supported archive grammar in
the tree.

## Artifact Custody

- Source archive:
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- Source archive bytes: `186780`
- Source archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- Candidate archive:
  `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/archive.zip`
- Candidate archive bytes: `186771`
- Candidate archive SHA-256:
  `c9d117fd6e9b5ef3fe7c1165d724c572560bfbc7f3b4cb8cc485b9bb4c612af5`
- Rate-only byte delta versus format `0x02`: `-9` bytes

Ignored artifact manifests preserved locally:

- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/packetir_manifest.json`
  SHA-256 `c1cf2f65a9c54f34afb211b026515132785716fcf780801bee69282349645ddd`
- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/runtime_consumption.json`
  SHA-256 `0a9a61eb0dd0aa03f81475a1149469224ce18ef94b08aed08e7114262ec03e39`
- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/prefix_parity_cpu_1pair.json`
  SHA-256 `e54f6ceee07e6125606a096c17a6d1b2a2207c4c4e359c39457845cf45f4fdbb`

## Evidence Axis

- `[packet-ir-parser-local-no-score]`: parse/emit identity and consumed-byte
  accounting for format `0x05`.
- `[runtime-sidecar-decode-local-no-score]`: runtime parser/decoder consumes
  format `0x05` and the semantic mutation changes corrected latents.
- `[macOS-CPU prefix parity no-score]`: one-pair same-runtime raw-output prefix
  parity versus the source `0x02` archive.

No `[contest-CPU]` or `[contest-CUDA]` score is claimed from this artifact.

## Test Evidence

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q
# 40 passed

PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py -q
# 12 passed
```

## Classification

Legitimate byte-grammar cleanup, not a dispatch priority. The main score-lowering
path remains divergence work: Z3 v2 latent replacement, C1/Z5 predictive receiver
surfaces, C6 executable campaign repair, Tier-C real-scorer MDL, and exact
CUDA/CPU axis-labelled candidate evaluation.
