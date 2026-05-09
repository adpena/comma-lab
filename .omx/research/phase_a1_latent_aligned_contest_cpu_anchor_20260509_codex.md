# Phase A1 Latent-Aligned Contest-CPU Anchor - 2026-05-09

## Verdict

The constrained A1 score-gradient refire is a **real `[contest-CPU]` positive**
on Linux x86_64:

- Recomputed CPU score: `0.19284757743677347`
- PoseNet distortion: `0.00003286`
- SegNet distortion: `0.00056023`
- Archive bytes: `178262`
- Archive SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Runtime tree SHA-256 from macOS advisory packet: `d40ce2732bbf2c9539ef1d833619b073bc2abc622f83d70ab9bdce362bf1a618`

This is public-axis evidence only. Internal promotion still requires a paired
`[contest-CUDA]` score. Modal T4 training/build succeeded, but its CUDA auth
eval path was skipped because the DALI/NVDEC preflight failed with NVML error
`999`.

## Candidate

- Lane: `track1_phase_a1_score_gradient`
- Modal label:
  `track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal`
- Modal call id: `fc-01KR55GH98QW3J3QDGQB1EG4CR`
- GHA submission name: `a1_latentalign_importpathfix_20260509`
- Fork PR: `#3`
- GHA workflow run: `25588422622`
- GHA URL:
  `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25588422622`
- Release tag:
  `cpu-eval-a1_latentalign_importpathfix_20260509-20260509T015557Z`

## Axis Comparison

| Axis | Score | Pose | Seg | Bytes | Evidence |
|---|---:|---:|---:|---:|---|
| macOS CPU advisory | `0.19286357743677346` | `0.00003286` | `0.00056039` | `178262` | non-promotable dev signal |
| GHA Linux x86_64 CPU | `0.19284757743677347` | `0.00003286` | `0.00056023` | `178262` | `[contest-CPU]` |

macOS vs Linux CPU delta is `1.6e-05`, consistent with using macOS CPU as a
fast advisory screen while preserving Linux x86_64 as the public leaderboard
authority.

## Commands

Modal recovery:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal
```

macOS advisory:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip \
  --inflate-sh experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/macos_cpu_advisory_work \
  --json-out experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

GHA Linux x86_64 CPU:

```bash
.venv/bin/python tools/dispatch_cpu_eval_via_github_actions.py \
  --archive-path experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip \
  --archive-sha 87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5 \
  --submission-name a1_latentalign_importpathfix_20260509 \
  --submission-dir experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir \
  --auto-create-fork-pr \
  --output-dir experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509
```

## Review Notes

The GHA helper originally copied the rounded report display value (`0.19`) into
`canonical_score`. This was fixed in `tools/dispatch_cpu_eval_via_github_actions.py`
so `canonical_score` now equals `score_recomputed_from_components`, with the
rounded line stored separately as `reported_final_score_display_rounded`.

The result-review packet also now distinguishes exact `[contest-CPU]` evidence
from generic non-CUDA proxy evidence:

- Review packet:
  `.omx/research/artifacts/a1_latentalign_importpathfix_result_review_20260509_codex.json`
- Evidence row:
  `reports/a1_latentalign_importpathfix_contest_cpu_evidence_row_20260509.json`

## Next Required Work

1. Run a paired `[contest-CUDA]` eval on the same archive/runtime packet.
2. Repair or avoid the Modal DALI/NVDEC preflight failure before using Modal for
   exact CUDA scoring.
3. If dual-axis custody remains green, scale A1 beyond the conservative `40 x 8`
   schedule and test whether longer score-gradient fine-tuning improves the CPU
   public axis without CUDA collapse.

## 2026-05-09 CUDA Path Hardening

The A1 Modal dispatcher now applies the known Modal DALI workaround
`DALI_DISABLE_NVML=1` at both levels that matter:

- Modal image environment, via `run_image.env(...)`.
- Runtime subprocess/probe environment, via the env passed into
  `scripts/probe_nvdec.sh` and `experiments/contest_auth_eval.py`.

It also sets the current worker process environment before the CUDA/DALI
preflight so the in-process DALI import and pipeline construction see the same
setting. Regression coverage:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py \
  tests/test_modal_phase_a1_recover_paths.py -q
```

This does not retroactively promote the existing A1 archive. It only clears the
known `nvml error (999)` bug class for the next claimed Modal refire or exact
CUDA eval attempt.
