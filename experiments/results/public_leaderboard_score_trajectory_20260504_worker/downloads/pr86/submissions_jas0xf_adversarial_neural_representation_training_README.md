# jas0xf_adversarial_neural_representation — Training Pipeline

This directory contains the full training pipeline that produces the 0.27 archive
for the comma video compression challenge. Run via the parent `compress.sh`.

## Files

| File | Role |
|---|---|
| `master.py` | TokenRendererV62 architecture + master pretrain & DALI fine-tune |
| `slave.py` | ShrinkSingleNeRV architecture + slave init / FT / DALI fine-tune |
| `hpac.py` | HPACMini entropy model + arithmetic codec + residual training loop |
| `archive.py` | Bake checkpoints, encode tokens, pack `archive.zip` |
| `data.py` | Shared GT video loader (`preload_rgb_pairs_av`) + image-size constants |

Each feature script is invoked as a subcommand, e.g.
`python master.py pretrain` or `python slave.py dali-ft`.

## Pipeline

The codec has three trained components:

- **Master** (`TokenRendererV62`, ~31 KB compressed) — token→RGB CNN with FiLM frame
  modulation. Reconstructs the second frame of each pair from a 5-class token map.
- **Slave** (`ShrinkSingleNeRV`, ~32 KB compressed) — per-frame NeRV-style latent
  decoder. Reconstructs the first frame of each pair (no token input, just frame index).
- **HPAC entropy model** (`HPACMini`, ~28 KB compressed via PPMd) — patch-grouped
  autoregressive arithmetic coder for the 600 token maps.

Together with the encoded `tokens.bin` (~114 KB) and `meta.pt` (~1 KB), they form
the final ~207 KB `archive.zip`.

## Stages

Total wall time: **~50–60 hours on a Tesla T4**, ~20–25 hours on RTX 4090.

| # | Stage | Command | Output | T4 hr |
|---|---|---|---|---|
| 1 | Master pretrain | `master.py pretrain` | `v62_full_best.pt` | ~25 |
| 2 | Master DALI FT | `master.py dali-ft` | `v62_full_dali_swa.pt` | ~3.5 |
| 3 | Slave init | `slave.py init --tag ultra --channels 24,16,12,8,8,6,6 --d-lat 6` | `slave_v62_shrink_ultra_swa.pt` | ~7 |
| 4 | Slave FT | `slave.py ft` | `slave_v62_shrink_ultra_ft_best.pt` | ~5 |
| 5 | Slave DALI FT | `slave.py dali-ft` | `slave_v62_shrink_dali_best.pt` | ~4.5 |
| 6 | HPAC residual entropy | `hpac.py train --save hpac.pt` | `hpac.pt` | ~7 |
| 7 | PPMd compress HPAC | (inline in `compress.py`, max_order=4) | `archive_build/hpac.pt.ppmd` | <1 min |
| 8 | Archive build | `archive.py` | `archive.zip` | ~10 min |

## Key hyperparameters

Specific tuned values (learning rates, epoch counts, SWA windows,
loss weights, SCN regularization schedules, error-boost factors,
etc.) are intentionally not listed here while the competition is
still active. Public defaults in the argparse interface are off-the-
shelf starting points, not the values that produced 0.27. The full
recipe and ablation table will be released after the deadline.


## Reproducibility notes

- DALI version **1.52.0** was used for both training and archive build, matching
  the official `evaluate.sh` GT pipeline. Mismatched DALI versions cause a
  noticeable score swing (we observed 0.27 → 0.34 on a version mismatch).
- Master FiLM is computed on **CPU FP32** at inflate time (`bake_film_table` in
  `inflate.py`) — the tiny `Linear(8→64)` picks different cuBLAS kernels on
  Ada SM89 vs Turing SM75, causing per-pixel divergence across GPUs.
- HPAC arithmetic coding uses `1/16384`-grid probability quantization for
  cross-hardware portability of the encoder/decoder state machine.
- All scripts share `<submission_dir>/training_workspace/` for inter-stage
  checkpoints. Resume a partial run with `--skip-stage <name>`.

## Caveat

The actual 0.27 archive submitted to the leaderboard was produced via an
iterative development process (multiple FT rounds with varying hyperparameters,
exploration runs, and a final consolidation). The orchestrator in `compress.py`
is a clean reference implementation of the recipe — running it end-to-end will
produce *an archive* in the same family that should score comparably on the
public test set, but exact bit-for-bit reproduction of the validated archive is
not guaranteed.
