---
title: CPU/CUDA DALI/PyAV Xray Tooling Review
date: 2026-05-08
owner: codex
status: diagnostic tooling review; no dispatch
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
---

# CPU/CUDA DALI/PyAV Xray Tooling Review

Scope: inspect the live diagnostics, preflight guards, and docs around eval
loader drift, CPU/CUDA calibration, PoseNet/SegNet layer drift, and stale
FastViT-attention mechanism claims. No remote, GPU, or exact-eval dispatch was
performed. This ledger is review evidence only.

## Surfaces Inspected

- `AGENTS.md` dual-axis auth eval protocol and DALI/PyAV mechanism caveat.
- `upstream/evaluate.py` and `upstream/frame_utils.py` device routing:
  CUDA uses `DaliVideoDataset`; CPU and MPS use `AVVideoDataset`; upstream
  asserts `AVVideoDataset` is not for CUDA.
- `experiments/contest_auth_eval.py` evidence contract:
  Linux x86_64 CPU full-sample eval is tagged `contest-CPU`, but remains
  non-promotable and not rank/kill eligible; T4-equivalent CUDA full-sample
  eval is the promotable `[contest-CUDA]` axis.
- CPU/CUDA planning and public-comment tools:
  `tools/plan_dual_device_auth_eval.py`,
  `tools/plan_public_pr_cpu_auth_eval.py`,
  `tools/public_pr_eval_comment_scorecard.py`, and
  `tools/analyze_cpu_cuda_eval_drift.py`.
- DALI/PyAV mechanism probes:
  `tools/probe_eval_loader_drift.py` and `tools/probe_eval_drift_matrix.py`.
- Xray/introspection tools:
  `tools/probe_posenet_layer_drift.py`,
  `experiments/dump_scorer_activations.py`,
  `src/tac/diagnostics/scorer_introspection.py`, and
  `tools/visualize_scorer_drift.py`.
- Guard surfaces:
  `tools/all_lanes_preflight.py` Gate #22 for eval-loader drift diagnostics,
  and `src/tac/preflight.py::check_no_fastvit_attention_compounding_claim`.
- Current reports and docs:
  `reports/eval_loader_drift_probe_dali_vs_pyav_plan_20260508.json`,
  `reports/posenet_layer_drift_probe_cpu_cuda_plan_20260508.json`,
  `reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json`,
  `docs/findings/cuda_cpu_auth_eval_split_20260508.md`,
  `docs/writeup/cuda_cpu_drift_methodology.md`,
  `.omx/research/public_replay_drift_hypothesis_20260508_codex.md`,
  `.omx/research/cpu_cuda_drift_adversarial_review_20260508_codex.md`, and
  `.omx/research/loader_drift_xray_supersession_20260508_codex.md`.

## Findings

1. Dual-axis calibration is now materially better than the earlier CUDA-only
   posture. The canonical auth-eval path distinguishes `[contest-CUDA]`,
   `[contest-CPU]`, and advisory CPU, and the current dual-device planner has
   a machine-checkable `dual_axis_completion` block. Older saved plan reports,
   including `reports/pr102_dual_device_auth_eval_plan_20260508.json`, predate
   that completion block and should not be treated as current schema examples.

2. The DALI/PyAV discriminator shape is correct but not closed. The live
   loader probe records the intended cells:
   `CPU+AV`, `CUDA+DALI`, `CUDA+AV/shared-input`, and `CPU+DALI`, and its
   forward-cell mode compares both PoseNet heads and SegNet logits/argmax.
   However, the CUDA/DALI path has not produced a filled tensor-custody
   artifact in this repo. The stale report
   `reports/eval_loader_drift_probe_dali_vs_pyav_plan_20260508.json` predates
   the hardened intended-cell schema and should be regenerated or marked
   superseded.

3. Preflight Gate #22 protects the loader-drift diagnostic from becoming a
   score claim. It validates non-promotable fields, axis-custody labels,
   intended 2x2 cells, future dispatch-claim requirements, typed missing
   prerequisites, and decoded-RGB comparison metrics when a comparison exists.
   This is a good local guard, but it still accepts a known missing CUDA/DALI
   prerequisite as pass on local non-CUDA hosts, so it does not prove the
   mechanism has been measured.

4. The xray stack is split. `probe_posenet_layer_drift.py` is a focused
   PoseNet/preprocess shared-input tracer; it is not a SegNet layer tracer.
   `dump_scorer_activations.py` and `scorer_introspection.py` can capture both
   PoseNet and SegNet records, but the real-video CUDA path in
   `dump_scorer_activations.py` currently calls `AVVideoDataset(...,
   device=cuda)`, which upstream explicitly rejects. That makes the
   introspection CLI unsafe as the canonical CUDA xray runner until it can
   consume an already-dumped shared input tensor or select DALI for the CUDA
   loader cell.

5. The stale FastViT-attention story is documented as rejected, but cleanup is
   incomplete. `check_no_fastvit_attention_compounding_claim(strict=False)`
   found 15 live docs/research violations in this checkout. Some are genuinely
   old claims; some are modern falsification text that the +/-3-line marker
   window does not recognize because "false" is not one of the accepted
   supersession markers. The guard is warn-only in `preflight_all()`, and it
   does not scan stale source-code comments such as older OOM explanations.

6. The 25 percent decoder contribution in the calibration registry is correctly
   labelled as a prior, not a measurement. `cuda_cpu_axis_profile_registry.py`
   keeps `DEFAULT_DECODER_POSE_DRIFT_FRACTION=0.25` as planning-only pending
   the shared-tensor DALI/PyAV x CPU/CUDA matrix. This is acceptable if
   downstream consumers preserve the non-score semantics.

## Rigor Gaps

- No current filled CUDA host artifact with AV and DALI input tensor SHA-256s,
  frame/video SHA, library versions, GPU/driver metadata, per-cell output
  hashes, and repeat-run jitter.
- No strict `forward_matrix_complete` field for
  `tools/probe_eval_loader_drift.py --run-forward-cells`; a runtime-error cell
  can remain buried inside per-cell rows instead of failing the whole xray
  matrix explicitly.
- No integrated SegNet layer-drift xray paired with the PoseNet tracer on the
  same shared input tensors.
- Saved reports are partly stale relative to the live schemas.
- FastViT-attention stale-claim cleanup is warn-only and still noisy enough
  that it cannot be promoted to strict.

## Top Integration Needed Next

Make `tools/probe_eval_loader_drift.py --run-forward-cells` the canonical
mechanism xray runner. It should dump and hash the exact shared AV and DALI
input tensors, run CPU and CUDA scorer forwards from those tensors, optionally
emit PoseNet and SegNet `ScorerIntrospector` records for each cell, and add
machine-checkable `tensor_custody` plus `forward_matrix_complete` fields.
Then wire Gate #22 to require that schema when a CUDA/DALI host is available,
while still accepting typed missing prerequisites on local non-CUDA hosts.

As part of that integration, `experiments/dump_scorer_activations.py` should
accept `--input-tensor` or an explicit loader-cell mode and must never
instantiate `AVVideoDataset` with a CUDA device.
