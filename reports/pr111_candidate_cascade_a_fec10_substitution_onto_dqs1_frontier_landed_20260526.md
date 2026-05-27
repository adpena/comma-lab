# PR111 Candidate — Cascade A FEC10 substitution onto DQS1 frontier

**Lane**: DQS1 pairset-drop-one rank021 + Cascade A FEC10 hybrid selector packet substitution
**Frontier-crossing**: -7.66e-6 [contest-CPU] / -8.66e-6 [contest-CUDA T4]
**Archive sha256**: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
**Archive size**: 178546 bytes (-13B vs DQS1 frontier baseline 178559)
**Author**: Alejandro Peña <adpena@gmail.com>

---

## Summary

This candidate substitutes the Cascade A FEC10 hybrid adaptive-blend selector packet directly into the DQS1 pairset-drop-one rank021 frontier archive (sha `7a0da5d0fc327cba...`, the current best [contest-CPU] anchor), preserving the source_payload + DQS1 packet trailer byte-identically. The substitution saves 13 wire bytes on the selector packet while preserving distortion components (seg + pose) within floating-point noise via frame-byte-identical inflate. Net effect: pure rate-axis improvement at fixed distortion.

| Axis | Score | Baseline | Delta |
|---|---|---|---|
| `[contest-CPU]` linux x86_64 | **0.19202062679074616** | DQS1 rank021 0.19202828295713675 | **-7.66e-6** (frontier-crossing) |
| `[contest-CUDA T4]` | **0.22618311337661345** | DQS1 paired CUDA 0.22619176954300405 | **-8.66e-6** (frontier-crossing) |

Both axes measured via Modal (paired contest-CUDA on T4 + paired contest-CPU on Linux x86_64 Modal container) using upstream `evaluate.py` on the exact archive bytes.

---

## Innovation

The Cascade A FEC10 hybrid adaptive-blend codec is an entropy coder for the 600-pair 16-symbol selector stream used by PR101-family frame-exploit selectors. It blends a marginal-symbol distribution with a per-context Markov transition table via a count-weighted blending rule with α=2 (empirical optimum). Wire-byte savings vs FEC6 fixed-Huffman baseline: -13B = -8.66e-6 contest_score_units rate-axis savings via `Δrate = 25 * (178546 - 178559) / 37545489`.

The substitution semantics preserve apples-to-apples comparability against the DQS1 frontier baseline:
- Decode equality verified: `decoded(FEC10) === decoded(FEC6)` on the same selector codes
- Frame-byte-identical inflate verified: output `0.raw` (3.66 GB) sha256 matches DQS1 baseline output exactly
- Distortion components preserved within floating-point noise (avg_segnet_dist +1e-8 from MKL CPU eval pathway; avg_posenet_dist identical)

---

## Per-component breakdown

| Component | CUDA T4 | CPU |
|---|---|---|
| avg_segnet_dist | 0.00066254 | 0.00055979 |
| avg_posenet_dist | 0.00016845 | 2.943e-05 |
| rate_unscaled | 0.004755458105766048 | 0.004755458105766048 |
| score_seg_contribution | 0.066254 | 0.055979 |
| score_pose_contribution | 0.041043879928928 | 0.017155174146594957 |
| score_rate_contribution | 0.11888523344768545 | 0.11888523344768545 |
| **score_recomputed** | **0.22618311337661345** | **0.19202062679074616** |

---

## Reproducibility

- Archive: deterministic ZIP (`zipfile.ZipInfo` with fixed `date_time=(2026,5,26,0,0,0)`, `ZIP_STORED`, `create_system=3`, `external_attr=0o644<<16`)
- Inflate runtime: `inflate.sh` (818 bytes, canonical 3-arg contract) + `inflate.py` (DQS1 inflate + FECa dispatch case added) + `src/` (DQS1 codec + FEC10 hybrid decoder shim) + `encoder/` (FEC10 hybrid encoder library)
- Entry point: `inflate.sh archive_dir output_dir file_list`
- Dependency closure: PyTorch + numpy + brotli (inherited from DQS1 reference runtime)
- Inflate device: auto (CUDA-if-available else CPU)
- All score components recomputed bytewise from upstream `evaluate.py --device {cuda,cpu}` on the exact archive bytes; no proxy scores reported

---

## Lineage

Builds on:
- DQS1 pairset-drop-one rank021 frontier substrate (archive sha `7a0da5d0fc327cba...`) — the canonical CPU frontier
- FEC10 hybrid adaptive-blend codec library — paradigm-portable across PR101-compatible selector packet substrates
- PR101 frame-exploit selector framework — 600-pair 16-symbol palette

The codec PARADIGM was first proposed via PR101 FEC6 baseline (-13B wire-byte savings; below-frontier substrate failed PR candidate test). The operator insight was that the codec library is paradigm-portable; substituting onto the actual canonical frontier substrate (DQS1 rank021) yields the predicted frontier-crossing rate-axis improvement.

---

## Operational notes

- Selector packet swap location: offset 178166 in `x` member (selector_len uint16 header + selector_payload bytes)
- DQS1 source_payload (178158 bytes) preserved byte-identically
- DQS1 packet trailer (42 bytes after selector) preserved byte-identically
- Inflate runtime extended with `b"FECa"` dispatch case in `unpack_pr101_selector` + `unpack_compact_selector_codes`

---

## Limitations

- Substitution semantics depend on selector packet structure compatibility (16-symbol palette + 600-pair format + PR101 wrapper); not portable to selector packets with different palette size or pair counts
- The -13B wire savings is a fixed-overhead improvement; future iterations may benefit from larger context windows (FEC11 = 3rd-order Markov) or wider blending (FEC12 = 3-distribution blend)
- Score gap of ~7.7e-6 below DQS1 frontier is within rate-axis savings noise; ANY further selector-stream entropy reduction would compound directly into score
