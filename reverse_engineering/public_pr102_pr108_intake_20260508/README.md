# Public PR102/PR108 Intake - 2026-05-08

Evidence grade: PR102 has local A++ contest T4 replay evidence for the
corrected archive/runtime pair; PR108 remains external plus empirical custody.

This directory is a source-sized manifest surface for two public contest PRs
that can otherwise confuse frontier bookkeeping:

- PR102: `hnerv_lc_v2_scale095_rplus1`, merged. The correct archive is the
  maintainer-comment attachment at
  `https://github.com/user-attachments/files/27369164/archive.zip`, not the
  stale `public_pr102_intake_20260505_auto` asset. It is byte-identical to
  PR100 and changes behavior through runtime constants and a decode-side
  frame-0 red-channel nudge. Hardened local Lightning T4 replay landed at
  `0.22839372989108092` `[contest-CUDA]`, confirming the public CUDA comment
  band and not the public CPU/leaderboard band.
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

Do not promote PR108 from this manifest alone. For PR102, use the tracked
result summary at
`.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json` plus the
local harvested artifact directory for exact replay custody; remaining PR102
work is wire-grammar/decode parity, not another same-archive replay.
