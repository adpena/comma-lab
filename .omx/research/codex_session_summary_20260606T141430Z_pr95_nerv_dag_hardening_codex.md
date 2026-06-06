# Codex Session Summary — PR95 NeRV DAG Hardening

UTC: 2026-06-06T14:14:30Z

## Scope

Converted the PR95 HiNeRV/SNeRV comparison into a fail-closed launch-readiness
surface rather than a memo-only analysis. The implemented/verified surface
models PR95 distortion readiness as an ordered DAG:

1. official non-overlapping seq_len=2 pair geometry;
2. upstream evaluate.py archive byte price;
3. evaluator preprocessing and PoseNet YUV6 roundtrip;
4. dual real SegNet/PoseNet scorer pressure;
5. scorer-domain telemetry;
6. PoseNet marginal/VJP telemetry;
7. family-local scorer-atom actuators;
8. PR95 staged QAT/coder curriculum;
9. live/fake-quant/archive-parseback/inflate/evaluate axis trace.

## Landed Contracts

- Added PR95 eight-stage optimization DAG semantics from the public PR95 intake:
  CE, tau-softplus, smooth disagreement, QAT, L7+C1a, lambda sweep, sigma
  sweep, and stage-8 Muon polish.
- Added archive parse-back distortion axis tracing contract:
  live forward, fake-quant forward, archive parse-back, inflate replay, and
  official evaluate.py.
- Added PoseNet frontier-marginal contract:
  `d/d(d_pose) sqrt(10*d_pose) = 5/sqrt(10*d_pose)`, with required direct-live
  score marginal and VJP telemetry before long-run authority.
- Added family-specific scorer-atom actuator contract:
  HiNeRV uses grid/output-head/pair-adapter controls; SNeRV uses
  MFU/HFR/TUB/output_2 and LF/HF scorer-incidence controls. Cross-family
  evidence remains rejected.
- Added PR95 source inventory proof for 28-D per-pair latents from the recovered
  PR95 source tree.
- Threaded the new contracts through campaign rows, queue launch authority,
  consumer/admission selected rows, and rebuilt row guards.

## Verification

- `uv run ruff check src/tac/analysis/pr95_distortion_practices_guard.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_pr95_distortion_practices_guard.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_admission.py`
- `uv run pytest src/tac/tests/test_pr95_distortion_practices_guard.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_admission.py -q`
- Result: 140 passed in 116.07s.

## Authority Boundary

This is not a score claim, exact-eval result, promotion gate closure, or
receiver-proof closure. It hardens the planner/admission DAG so future
HiNeRV/SNeRV long-run rows fail closed unless they carry the PR95-derived
geometry, scorer math, parse-back axis, PoseNet marginal, and family-actuator
contracts.

## Next Work

- Land the low-level direct-live PoseNet marginal telemetry in the MLX loss path.
- Add the crux trace tool as an executable init-to-replay diagnostic.
- Run the family-local pair-adapter smoke for HiNeRV and the MFU/HFR/TUB
  source-forward parity closure for SNeRV before any promotional long run.
