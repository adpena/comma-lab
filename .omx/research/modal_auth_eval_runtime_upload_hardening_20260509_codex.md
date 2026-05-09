# Modal CUDA Auth Eval Runtime Upload Hardening - 2026-05-09

## Verdict

`experiments/modal_auth_eval.py` is now usable for arbitrary candidate packets,
not only archives whose `inflate.sh` already exists inside one of the fixed
Modal image mounts.

The wrapper now uploads a deterministic transport ZIP of a local
`submission_dir` when `--submission-dir` is provided, extracts it fail-closed
with `tac.submission_archive.safe_extract_zip`, and runs the canonical CUDA
path against the uploaded runtime:

```text
archive.zip -> uploaded submission_dir/inflate.sh -> upstream/evaluate.py --device cuda
```

It also applies the Modal DALI workaround `DALI_DISABLE_NVML=1` in both image
and subprocess environments.

## Why This Matters

The A1 latent-aligned archive has a real `[contest-CPU]` score but still lacks a
paired `[contest-CUDA]` result. Its runtime lives in the harvested
`finetuned_archive/submission_dir`, which the old Modal CUDA auth-eval wrapper
could not see unless the path was hard-mounted into the image.

This hardening makes the exact-CUDA attempt runtime-complete without editing the
scorer or relying on a pre-mounted experiment-result path.

## Validation

```bash
.venv/bin/python -m pytest src/tac/tests/test_modal_auth_eval.py -q
.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings
```

Focused coverage checks:

- T4 wrapper signature accepts uploaded runtime ZIP custody.
- Local `--submission-dir` requests preserve `score_claim=false` until remote
  validation succeeds.
- Runtime transport ZIP is deterministic and strips `.pyc` / `.DS_Store`.
- Source contains `DALI_DISABLE_NVML=1`, `safe_extract_zip(...)`, and CUDA-only
  canonical eval wiring.

## Next Command Shape

After claiming the lane:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_auth_eval.py \
  --archive experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip \
  --submission-dir experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/a1_latentalign_importpathfix_modal_cuda_eval_20260509 \
  --gpu T4 \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400
```

The output remains adjudication-required and not promotion-eligible until the
harvested `contest_auth_eval.json` is reviewed and component recomputation
matches the archive custody.
