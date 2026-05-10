# A1 Modal score-gradient exact-CUDA regression

Generated: `2026-05-10T08:31:00Z`

## Verdict

`track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` completed exact
`[contest-CUDA]` custody on Modal T4, but the measured configuration is a
regression and must not be promoted.

Classification: `A-negative / measured-config regression`.

## Custody

- Modal call id: `fc-01KR8D302GXGKGT49ETYMA0BZC`
- Archive SHA-256: `f5d04f22d46bc1c4b863e9e2989c25f9b04e07cb21d54980b5effb654edc127a`
- Archive bytes: `206110`
- Runtime tree SHA-256: `fae77695921cd2a6c948cbd85d0e720b9a08d3a6e64c85f4a79f44dd579e6fa2`
- Hardware: Modal `Tesla T4`, CUDA `12.4`, DALI/NVDEC preflight passed
- Samples: `600`
- Harvest summary: `experiments/results/track1_phase_a1_score_gradient_modal_20260510T0738Z_codex/harvest_summary.json`
- Terminal claim: `completed_modal_contest_cuda_recovered`
- Code-state caveat: this dispatch predates the mounted-code snapshot hardening
  landed in `a3311268`, so archive/runtime custody is exact but source-state
  replay is weaker than future Modal dispatches that record clean mounted-code
  patches.

## Exact-CUDA result

- Score: `0.5447505505333358`
- Seg distance: `0.00336345`
- Pose distance: `0.00050645`
- Rate term: `0.13724025`
- Rate unscaled: `0.00548961`

The run is exact evidence that this A1 score-gradient configuration worsens the
frontier. It is not evidence against score-domain training as a family.

## Adversarial interpretation

Training proxy moved in the intended direction, but exact score did not:

- Initial training proxy seg: `0.016153013333678246`
- Final training proxy seg: `0.0025390908122062683`
- Initial training proxy pose: `0.0002999906428158283`
- Final training proxy pose: `0.0012046925257891417`
- Best proxy epoch: `181`
- Best proxy weighted proxy: `0.27168376521416054`

Likely failure class: proxy/exact mismatch plus rate blow-up after decoder
weight re-encoding. The final EMA archive grew from PR101's `178258` bytes to
`206110` bytes, so any small segmentation improvement had to overcome a large
rate penalty. It did not.

## Reactivation criteria

Do not re-run this exact configuration. Reactivate A1 score-gradient only with
one of:

1. Exact-eval-in-loop or frequent exact-eval anchors, not training proxy alone.
2. Rate-constrained archive builder that rejects decoder payload growth above a
   declared trust region.
3. Best-proxy checkpoint exact eval as a bounded diagnostic, not a promotion
   candidate.
4. Lower-LR / early-stop trust region that caps both pose drift and archive-byte
   growth before CUDA auth eval.
5. A2/T1-style packet compiler path that records code snapshot, runtime closure,
   and no-op proof before dispatch.

## Dispatch implication

Long T1 score-domain training should not launch blindly from this result. The
next T1 run should be a short guarded Modal exact-eval probe with a small target
pair cap and full custody, then promote only if exact CUDA moves in the right
direction.
