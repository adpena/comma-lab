# Phase A1 Score-Gradient Modal Advisory Negative - 2026-05-08

Scope: classify the recovered Modal A1 score-gradient PR101 fine-tune without
promoting it as a contest score.

## Remote harvest

- Lane claim: `track1_phase_a1_score_gradient`
- Modal call: `fc-01KR4X449H99NB1MQXJC0QN52S`
- Harvest path:
  `experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/`
- Remote stage: `completed_training_build_cuda_eval_skipped_nvdec_preflight`
- Remote return code: `0`
- Evidence grade: `[cuda-training-build-only]`
- Score claim: `false`
- Candidate archive bytes: `205879`
- Candidate archive SHA-256:
  `cb9de2b71133929b0c2df00b0e511b9c306939d62438ffb348e947aef719e185`

The remote Modal run trained and built an archive, but exact CUDA auth eval was
skipped because the DALI/NVDEC probe failed with `PROBE_CLASSIFICATION:DALI_BUILD`
and `nvml error (999)`. This is not a score and cannot promote, rank, or kill.

## Runtime portability fix

The first local advisory eval failed before scoring because the generated A1
`inflate.sh` used bare `python`:

```text
RuntimeError: [inflate] FAILED with returncode=127
```

`tools/build_pr101_finetuned_archive.py` now emits:

```bash
"${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"
```

The harvested runtime was patched the same way for the local advisory
classification below. The archive bytes and archive SHA did not change.

## Local macOS CPU advisory classification

Claim rows:

- `failed_pre_score_inflate_python_missing` for
  `a1_score_gradient_macos_cpu_advisory_20260508T233329Z`
- `completed_macos_cpu_advisory_score_3p721654` for
  `a1_score_gradient_macos_cpu_advisory_20260508T233431Z`

Command:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/harvested_artifacts/finetuned_archive/archive.zip \
  --inflate-sh experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/harvested_artifacts/finetuned_archive/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/macos_cpu_advisory_work_after_python_patch \
  --json-out experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/contest_auth_eval.macos_cpu_advisory.after_python_patch.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Result:

- Evidence grade: `[macOS-CPU advisory]`
- Evidence semantics: `non_contest_cpu_auth_eval_advisory`
- Canonical score: `3.7216542390470915`
- PoseNet distortion: `0.17846388`
- SegNet distortion: `0.02248664`
- Rate contribution: `0.1370865`
- Archive bytes: `205879`
- Runtime tree SHA-256:
  `dcdc5b995993fb455989e743bd55eba9987648675a9cb2c5da6e3975c451bc7c`
- Durable JSON:
  `experiments/results/track1_phase_a1_score_gradient_20260508T230020Z_modal/contest_auth_eval.macos_cpu_advisory.after_python_patch.json`

Disposition:

- The measured A1 config
  `(epochs=200, steps_per_epoch=20, lr=1e-4, aux_kl_weight=1.0,
  aux_pixel_l1_weight=0.01, max_frames=1200)` is retired as a
  measured-configuration negative.
- A1 as a family is not killed. Reactivation requires a constrained fine-tune
  that keeps the PR101 reconstruction basin intact, smaller learning rate or
  tighter early-stop/custody checks, and a local advisory eval before any
  exact CUDA or contest-CPU spend.
- Do not remote exact-eval this archive unless the operator explicitly wants a
  formal negative.
