# Codex Findings - PR95 MLX Downstream Scorer Drift Boundary

UTC: 2026-05-25T21:08:00Z

## Scope

Reviewed and corrected the PR95 MLX-to-PyTorch full-decoder downstream scorer
drift measurement lane after sibling work produced a useful but over-broad
memo. The tool now defaults to the contest inflate boundary:
`scorer_input_mode=contest_uint8`.

## Finding

The original measurement signal is useful, but it must remain bounded. The
available archive smoke capped at `n_pairs_actual=1`, so it is a sampled
engineering-bridge measurement, not a full-video contest authority result and
not a reason to skip exact CPU/CUDA anchors for promotion or hardware-sensitive
rank/kill decisions.

Fresh Codex smoke after the boundary fix:

- Command: `.venv/bin/python tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py --n-pairs 1 --scorer-input-mode contest_uint8 --output-json experiments/results/pr95_mlx_pytorch_full_decoder_downstream_drift_20260525T_codex_contest_uint8_smoke/results.json`
- Result path: `experiments/results/pr95_mlx_pytorch_full_decoder_downstream_drift_20260525T_codex_contest_uint8_smoke/results.json`
- Result SHA-256: `4874409f6c6ff2d4d5944fad85315ac93790e7ef885891c70114f6d3782bd093`
- Aggregate drift: `7.411371190353894e-05`
- Verdict: `BELOW_SCORER_PRECISION` against the local `0.001` drift threshold
- Evidence grade: `[macOS-MLX research-signal]`

## Landing

`tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py` now:

- measures SegNet/PoseNet on the `contest_uint8` scorer input boundary by
  default;
- keeps `decoder_float` as an explicit diagnostic mode;
- blocks diagnostic-mode manifests from contest-path closure evidence;
- writes top-level `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`
  fields on every result manifest;
- avoids language that could be read as promotion, score, or exact-eval
  authority.

The probe-outcomes row was corrected to point at the contest-uint8 smoke and to
route follow-up toward wider PR95-class batch measurement before spend-triage
use.

## Authority Boundary

This remains local `[macOS-MLX research-signal]` only. It does not claim score,
promote, rank/kill, or authorize exact-eval dispatch. Exact contest CPU/CUDA
anchors remain mandatory before any promotion or hardware-sensitive decision.

## Next Gap

Run the same tool against a wider PR95-class packet or a full-video archive once
available, then feed the result into the MLX production-contract and effective
spend-triage gates as bounded calibration evidence.
