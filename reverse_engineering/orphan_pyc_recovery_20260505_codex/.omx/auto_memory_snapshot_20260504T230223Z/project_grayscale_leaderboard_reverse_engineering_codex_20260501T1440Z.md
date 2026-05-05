# Project Memory: Grayscale Leaderboard Reverse Engineering

Date: 2026-05-01T14:40Z

Scope: durable context for the comma video compression Shannon-floor push.

Key finding:

- The #2/Selfcomp-style grayscale trick is not plain CRF grayscale replacement.
  It is train/inflate parity around an analog soft class-probability map:
  `softmax(exp(-(gray - target)^2 / (2 sigma^2)))` with targets
  `[0, 255, 64, 192, 128]` and `sigma=15.0`.
- Public Selfcomp source feeds this soft probability map directly into the
  renderer and uses frame indices so one decoded grayscale frame can generate a
  two-frame pair. Public Quantizr source uses one charged mask per pair plus
  pose-conditioned joint generation. Both train against the decoded
  representation instead of retrofitting grayscale after training.
- Our CRF60/CRF62 exact L40S CUDA results collapsed PoseNet. These are scoped
  negatives for post-hoc hard/CRF grayscale substitution only, not a family kill
  for learned grayscale, Selfcomp, Quantizr, Alpha, NeRV, INR, or HNeRV.

Code correction landed:

- `src/tac/mask_grayscale_lut.py` now implements the Selfcomp bell-softmax LUT
  and exposes `grayscale_to_probability_map`.
- `experiments/train_segmap.py` and
  `experiments/train_segmap_film_canvas.py` now build pair tensors from the
  exact soft-LUT distribution.
- `submissions/robust_current/inflate_segmap.py` and
  `submissions/robust_current/inflate_segmap_film_canvas.py` now default to
  `SEGMAP_GRAYSCALE_MODE=soft_lut`; `hard_onehot` remains only for forensic
  compatibility.
- `scripts/remote_lane_sa_segmap_clone.sh` and
  `scripts/remote_lane_fc_film_canvas.sh` explicitly write
  `SEGMAP_GRAYSCALE_MODE=soft_lut` into their generated eval `config.env` and
  record the mode in provenance.
- `src/tac/segmap_renderer.py` and `src/tac/optimize_grayscale_canvas.py` now
  use the same bell-softmax math in differentiable paths.

Verification:

- MCP cleanup strict: clean.
- Python compile passed for touched modules.
- Shell syntax passed for `scripts/remote_lane_sa_segmap_clone.sh`,
  `scripts/remote_lane_fc_film_canvas.sh`, and
  `scripts/remote_lane_q_faithful_jointgen.sh`.
- Tests passed: grayscale LUT + analog canvas `26 passed`; SegMap/KL/LCT focus
  `29 passed, 1 skipped`; `git diff --check` passed.

Operational implication:

- Stop primary spending on plain CRF grayscale threshold sweeps. Next highest
  EV action is fast-CUDA training/export/eval of corrected soft-LUT SA or
  FilmCanvas, plus Q-FAITHFUL/Quantizr-like closure if a free worker/GPU is
  available. Exact CUDA auth eval remains the only score truth.
