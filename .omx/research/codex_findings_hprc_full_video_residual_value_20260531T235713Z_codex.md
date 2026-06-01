# HPRC Full-Video Residual Value Finding

Axis: `[macOS-MLX research-signal]`; not contest CPU/CUDA authority.

The 2-pair residual-zero result was a prefix mirage. On the full 600-pair
MLX-GPU scorer pass, deleting `residual_rc` saves 1,118,342 `archive.zip` bytes
but worsens the advisory full-video score by 0.2663341961831449.

| Variant | Archive bytes | Score | Pose dist | Seg dist | Rate term |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1,161,670 | 44.623549912068924 | 154.0151398026943 | 0.046052788592254125 | 0.7735083700734328 |
| residual_zero | 43,328 | 44.88988410825207 | 163.1264383951823 | 0.04472119649561743 | 0.028850336720877442 |

Delta residual_zero minus baseline:

- Score: `+0.2663341961831449`
- Rate term: `-0.7446580333525553`
- Pose distance: `+9.111298592487998`
- Seg distance: `-0.0013315920966366981`

Verdict: residual bytes are valuable at full-video scope, mostly through PoseNet.
The low-hanging fruit is therefore not section deletion; it is scorer-ranked
residual compression with a hard rate price. The next materializer should sweep
significance/bitplane/range-coded residuals and preserve only residual atoms
whose full-video P18/P19 marginal value exceeds `25 / ORIGINAL_VIDEO_BYTES`.

Operational note: the original CPU MLX profile path was stopped after the
GPU full-video run completed because it was still in singleton CPU MLX scoring
after roughly 19 minutes and had been launched before the full-video partial
flag fix. The generated SSD cache and variant artifacts were preserved at
`/Volumes/VertigoDataTier/pact/hprc_mlx_component_neutralization_full600_20260531T235713Z`.

Stored evidence:

- `.omx/research/hprc_full_video_residual_value_gpu_20260531T235713Z_codex/full_video_residual_value_verdict.json`
- `.omx/research/hprc_full_video_residual_value_gpu_20260531T235713Z_codex/mlx_responses_gpu/baseline.json`
- `.omx/research/hprc_full_video_residual_value_gpu_20260531T235713Z_codex/mlx_responses_gpu/neutralize_residual_rc.json`
