# judges one-pager

## headline (current era — neural renderer)

- Best contest-CUDA score: **`1.05`** (Lane G v3, 694KB archive)
- Modal T4 reproduction: **`1.04`** (within 0.01 noise of Vast.ai)
- Recipe: dilated-h64 neural renderer + KL distill weight=0.002 + pose TTO retry on Lane A anchor
- PoseNet 0.0034 / SegNet 0.0040 / Rate 0.0185
- Live work targets sub-0.30 via the Selfcomp paradigm portfolio

## historical headline (Era 1 — kept for the writeup arc)

- Best honest Track B **current_workflow** score: **`1.73`** (h64 long-horizon QAT+EMA learned int8 post-filter)
- Bytes: `864,167`
- This was the codec + post-filter era. Still the best score in that paradigm; the renderer paradigm displaced it after Era 2 launched.

## why we landed at 1.05

- **prior contest-CUDA floors**: 0.90 (pinned dilated h64 + CRF=50 baseline, 2026-04-25), 1.15 (Lane A pose TTO from baseline poses, 2026-04-27)
- **change**: KL distill on the SegNet logits (T=2.0, weight=0.002) during renderer training, paired with a fresh pose-TTO retry on the Lane A anchor
- **result**: `1.05` [contest-CUDA] (2026-04-28 verified, 2026-04-29 reproduced 1.04 on Modal T4)
- **reflection**: distillation weight matters more than distillation choice — earlier KL attempts at weight≥0.01 collapsed PoseNet; weight=0.002 sustains the SegNet boundary signal without overwhelming the renderer's pose path

## proof points

- contest-CUDA inflate.sh → upstream/evaluate.py on the EXACT submission archive bytes
- Modal T4 reproduction independently confirms the score within 0.01
- canonical local E2E smoke passes (10 stages, 0.02s)
- 78 STRICT preflight checks gate every measurement against known catastrophic-failure classes
- the proxy-auth gap is closed: eval_roundtrip non-negotiable everywhere

## leaderboard context (fetched 2026-04-29)

- Quantizr 0.33 (#1) — FiLM CNN 88K + KL-T2 + AV1
- Selfcomp 0.38 (#2) — self-compression ~1.017 bpw + analytical-pose affine
- Mask2mask 0.60 (#3) — obfuscated arch
- our 1.05 would rank ~4th if submitted today; live work targets sub-0.30
