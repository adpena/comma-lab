# Codex Findings: Upstream Eval Contract Grounding

UTC: 2026-06-01T16:30:00Z

## What Landed

I made the upstream contest evaluator contract machine-readable instead of
leaving it as chat memory:

- `src/tac/contest_eval_contract.py`
  - records source/model custody for `upstream/evaluate.py`, `evaluate.sh`,
    `modules.py`, `frame_utils.py`, and upstream `.github/workflows/eval.yml`;
  - validates defining source fragments for score, rate, inflate, SegNet,
    PoseNet, raw tensor loading, and CI hardware axes;
  - exposes the actual score geometry:
    `100*d_seg + sqrt(10*d_pose) + 25*archive_zip_bytes/source_video_bytes`;
  - exposes the exact byte water level:
    `25 / 37,545,489 = 6.658589531221714e-7` score per archive byte;
  - records that inflated raw bytes (`3,662,409,600`) are a receiver shape
    requirement, not the rate denominator.
- `tools/build_upstream_eval_contract.py`
  - emits the contract as JSON for operator/runner consumption.
- `src/tac/substrates/hprc/bitstream_grammar.py`
  - now embeds the upstream score allocation contract in the HPRC joint
    P18/P19 saliency contract.

## Score-Relevant Findings

- SegNet scores only frame 1 of each 2-frame pair:
  `modules.py::SegNet.preprocess_input` slices `x[:, -1, ...]`.
  Frame 0 atoms have zero direct SegNet saliency; frame 1 atoms carry SegNet
  plus PoseNet incidence.
- PoseNet scores both frames of the pair after resize to `384x512`, YUV6
  conversion, and first-six pose-head MSE.
- The upstream `rgb_to_yuv6` forward is the eval authority, but it uses
  no-grad/in-place clamp behavior. Any P19 gradient lane must use a verified
  differentiable mirror and remain proposal-only until byte-closed replay and
  exact CPU/CUDA eval.
- "Optimal grammar" is split into two distinct layers:
  - container grammar, already near entropy floor on PR95/fec6-style packets;
  - score grammar, the contest-fixed reverse-waterfill rule that spends bytes
    only when measured P18/P19 value per byte exceeds `25/source_video_bytes`.

## Guardrails

The new `upstream_saliency_verification_contract.v1` requires no-fake numerical
proofs before saliency rows can drive budget spend:

- YUV6 forward parity against upstream.
- Nonzero gradient through the differentiable YUV6 mirror.
- SegNet frame-1-only asymmetry.
- Exact SegNet argmax-flip reduction.
- Exact PoseNet first-six-dim MSE reduction.
- Exact byte-price accounting.

Receiver-side contest compliance remains fail-closed: no scorer loads, no
evaluator-state inspection, no sidecars, no eval-time adaptation.

## Evidence

Generated contract:

`.omx/research/upstream_eval_contract_20260601T162902Z_codex.json`

YUV6 differentiability probe:

`.omx/research/yuv6_differentiability_probe_20260601T163115Z_codex.json`

Summary:

- `contract_valid=true`
- `blockers=[]`
- `live_rate_denominator_bytes=37,545,489`
- `canonical_rate_denominator_bytes=37,545,489`
- source/model SHA-256 custody recorded for evaluator files and scorer weights
- upstream no-grad baseline: `grad_l2=0.0`, forward diff vs mirror `0.0`
- monkey-patch and TAC differentiable routing: `grad_l2=38.03618621826172`,
  forward diff vs mirror `0.0`, both pass

Verification:

- `ruff check src/tac/contest_eval_contract.py src/tac/substrates/hprc/bitstream_grammar.py src/tac/tests/test_contest_eval_contract.py src/tac/tests/test_hprc_bitstream_grammar.py tools/build_upstream_eval_contract.py`
- `pytest src/tac/tests/test_contest_eval_contract.py src/tac/tests/test_hprc_bitstream_grammar.py -q`
  - `10 passed`
- `python tools/probe_yuv6_differentiability_disambiguator.py --output .omx/research/yuv6_differentiability_probe_20260601T163115Z_codex.json`

## Adjacent WIP Not Absorbed

The worktree also contains untracked sibling files with overlapping upstream
scorer-contract and mirror-fidelity ideas:

- `src/tac/eval/upstream_scorer_contract.py`
- `src/tac/local_acceleration/mlx_upstream_scorer_contract.py`
- `tools/verify_upstream_scorer_mirror_fidelity.py`

I did not stage or subsume those because they appear to be partner WIP and have
not been fully reviewed in this slice. The next safe action is to reconcile them
against `tac.contest_eval_contract` so one canonical contract owns custody while
any mirror-fidelity harness becomes a tested consumer, not a duplicate authority
surface.
