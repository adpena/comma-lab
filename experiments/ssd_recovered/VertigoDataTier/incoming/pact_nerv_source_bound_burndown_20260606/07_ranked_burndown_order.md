# 07 — Ranked burndown order

| Rank | Task | Why first | Files | Failing test | Pass target |
|---:|---|---|---|---|---|
| 1 | Parse-back archive selection | PR95 source-faithful and cheap; runs only at export. | `long_training_canonical.py`, adapter hook | fake live better locally but worse after replay | selected candidate follows parse-back proxy |
| 2 | HiNeRV target-region birth actuator | Directly attacks min-ratio=0 failure. | `hi_nerv/mlx_renderer.py`, `target_region_birth.py` | worst target region margin improves but hard ratio still zero | region ratio or frontier margin improves under scoped params |
| 3 | Joint Seg/Pose trust region | Prevents Seg repair from destroying pose and vice versa. | `mlx_score_aware/adapter.py` | positive joint delta accepted | positive joint delta rejected/backtracked |
| 4 | SNeRV full TUB parity closure | Cannot long-run official SNeRV without it. | `snerv_official_tub_source_forward_replay.py` | unmapped TUB keys present | no unmapped keys, full parity true |
| 5 | Section value-per-byte ledger | Makes rate controls measured. | `nerv_section_value_ledger.py`, tools | actuator lacks bytes/delta score | fail closed until row exists |
| 6 | Full-video MLX replay smoke | Sample smokes are not enough. | tools | no full-video row | full-video false-authority replay exists |
| 7 | Long-run launch configs | Only after gates. | experiments runner configs | gates missing | artifact contains gate proof paths |

## Not first

- Muon as default optimizer: PR95 used it at Stage 8; do not use it as the class-birth mechanism.
- More semantic class weights: evaluator prices pixels, not semantic importance.
- More LF/HF storage sweeps for SNeRV before official TUB closure: rate results may be on the wrong graph.
