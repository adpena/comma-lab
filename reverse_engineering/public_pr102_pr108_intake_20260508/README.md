# Public PR102/PR108 Intake - 2026-05-08

Evidence grade: external plus empirical custody. No score claim.

This directory is a source-sized manifest surface for two public contest PRs
that can otherwise confuse frontier bookkeeping:

- PR102: `hnerv_lc_v2_scale095_rplus1`, merged. The correct archive is the
  maintainer-comment attachment at
  `https://github.com/user-attachments/files/27369164/archive.zip`, not the
  stale `public_pr102_intake_20260505_auto` asset. It is byte-identical to
  PR100 and changes behavior through runtime constants and a decode-side
  frame-0 red-channel nudge.
- PR108: `andimin01`, open and non-frontier by its own CPU report. The archive
  is a single `0.mkv` member produced by AV1 plus ROI preprocessing and
  sharpened CPU inflation.

Raw archives are intentionally not stored here. Current local custody paths:

- PR102 archive:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- PR108 archive:
  `experiments/results/public_pr108_andimin01_intake_20260508_codex/archive.zip`

Use `manifest.json` for PR number, URL, head SHA, archive URL/path/bytes/SHA,
ZIP member hashes, runtime file hashes, compliance risks, and the fastest
exact replay path. Any future score-bearing claim still requires exact CUDA
replay through:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <archive.zip> \
  --inflate-sh <public-runtime-adapter/inflate.sh> \
  --upstream-dir upstream \
  --device cuda
```

Do not promote PR102 or PR108 from this manifest alone. PR102 explains a
public CPU/CUDA drift and a local wrong-asset gap; PR108 is intake/classification
evidence unless exact CUDA unexpectedly changes its status.
