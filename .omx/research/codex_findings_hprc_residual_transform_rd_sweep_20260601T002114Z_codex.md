# HPRC Residual Transform Full-Video RD Sweep

Axis: `[macOS-MLX research-signal]`; not contest CPU/CUDA authority.

This sweep converts the residual low-hanging fruit from "delete the section" to
a real pre-entropy residual-token actuator. The previous full-video finding
showed that `residual_rc` is valuable and cannot be removed wholesale. The
new result is stronger: deterministic residual coarsening improves the full
600-pair MLX advisory score while shrinking `archive.zip`.

| Variant | Archive bytes | Delta score | Delta nonrate | Delta rate | Delta pose | Delta seg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold_abs_le=3 | 453,171 | -1.3152033838421247 | -0.8428543620994589 | -0.47234902174266524 | -9.316994195307274 | +0.003626988641917704 |
| quant_step=2 | 704,018 | -1.296610014884351 | -0.9912897139555241 | -0.3053203009288279 | -7.51042821254444 | -0.00022460930980742339 |
| threshold_abs_le=2 | 629,493 | -0.9378472278699448 | -0.5829037884596886 | -0.35494343941025774 | -5.473324539860073 | +0.0012073601720233723 |
| threshold_abs_le=1 | 895,490 | -0.28453827659710385 | -0.10671132114048731 | -0.17782695545661953 | -1.0531068873405331 | +0.0002769046897689448 |
| keep_top_fraction=0.4 | 895,512 | -0.28452362770013906 | -0.10671132114048731 | -0.17781230655965086 | -1.0531068873405331 | +0.0002769046897689448 |

`threshold_abs_le=3` is the current total-score winner and now has archive-bound
receiver proof:

- Archive SHA-256: `bfd2c7a604134ead151ff7e784fb06a7f76ffd66c817610ef997d9dd23814188`
- Archive bytes: `453529`
- Receiver output bytes consumed: `3662409600`
- Receiver output SHA-256: `ed484328d899fe6b7b0e1076db1d5d5ec38ac21429361122f96d7f93c8489e69`
- Receiver wall seconds: `7.216726`

Engineering note: vectorizing the compact receiver frame compute gives a real
compute-only speedup, but the end-to-end writer is dominated by full-resolution
raw/scorer-cache I/O. The next performance target is an HPRC acquisition cache
path that renders directly into scorer input tensors for MLX sweeps, with full
`inflate.sh` receiver proof reserved for winners.

Next action: exact CPU/CUDA gate the receiver-proven `threshold_abs_le=3`
candidate, then refine the residual allocator between `threshold_abs_le=3` and
`quant_step=2` with class/boundary/P19 surfaces.
