# Codex Session Summary: F1-as-A2 RGB Invariance

Timestamp: 2026-05-19T02:06:39Z
Actor: codex
Session: `019de465`

## Landed

- Added corrected F1-as-A2 probe helper:
  `src/tac/contest_exploits/f1_as_a2_rgb_invariance.py`
- Added internal-only authority helper:
  `src/tac/contest_exploits/hydra_dim_invariance.py`
- Added operator probe:
  `tools/probe_f1_as_a2_posenet_rgb_invariance.py`
- Preserved routed tool name as a corrected compatibility wrapper:
  `tools/probe_hydra_dim_7_12_score_invariance.py`
- Added focused regression tests:
  `src/tac/tests/test_rate_attack_f1_as_a2_probe.py`
- Registered corrected L0 lane:
  `lane_rate_attack_f1_as_a2_posenet_rgb_invariance_20260519`
- Registered advisory probe outcome:
- Superseded initial too-strong metric row:
  `f1_as_a2_rgb_invariance_20260519T020507Z`
- Registered blocking direct-Hydra outcome:
  `f1_direct_hydra_dim_channel_blocked_20260519T021330Z`
- Superseded corrected advisory terminology row:
  `f1_corrected_a2_rgb_invariance_20260519T021330Z`
- Registered current corrected advisory outcome:
  `f1_as_a2_rgb_invariance_20260519T021450Z`

## Result

The real local CPU probe found a narrow positive mechanism signal:

- corrected metric: `changed_rgb_values_per_pair=110.0`
- recovered payload bits: `0.0`
- `pose_0_5_rmse=4.727051377700551e-06`
- `seg_delta_fraction=0.0`
- evidence JSON SHA-256:
  `6b6a57d98f4c5da6e0824c64ca1c4667c2aa1a530110aaaf03afff767b33b05e`
- axis: `[macOS-CPU advisory]`

This result is only `[local-rgb-capacity-probe]` authority. It does not claim a
score, a CPU/CUDA axis result, dispatch readiness, or promotion eligibility.
The old `accepted_bits_per_pair` wording was superseded before commit because
changed RGB values are not payload bits without a decoder/recovery proof.

## Tests

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_rate_attack_f1_as_a2_probe.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check src/tac/contest_exploits/f1_as_a2_rgb_invariance.py src/tac/contest_exploits/hydra_dim_invariance.py src/tac/contest_exploits/__init__.py tools/probe_f1_as_a2_posenet_rgb_invariance.py tools/probe_hydra_dim_7_12_score_invariance.py src/tac/tests/test_rate_attack_f1_as_a2_probe.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile src/tac/contest_exploits/f1_as_a2_rgb_invariance.py src/tac/contest_exploits/hydra_dim_invariance.py tools/probe_f1_as_a2_posenet_rgb_invariance.py tools/probe_hydra_dim_7_12_score_invariance.py src/tac/tests/test_rate_attack_f1_as_a2_probe.py`
- `git diff --check`

All passed before this memo was written.

## Remaining

- Build charged archive grammar for F1-as-A2 perturbations.
- Prove perturbation bytes are consumed by inflate and are not no-op/provenance
  artifacts.
- Run a capacity sweep across more pairs and trust-region settings.
- Promote only through paired `[contest-CPU]` and `[contest-CUDA]` exact eval.
- Keep original direct Hydra dim 7:12 channel blocked as internal-only unless
  reworked as a self-contained deterministic packet compiler.
