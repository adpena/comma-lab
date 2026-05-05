# PR106 vs PR101 — the +0.017 gap is training-recipe, not codec (2026-05-04)

## Discovery

The fixed dashboard surfaced a curious gap: PR106 frontier (0.209451) is
**+0.017 ahead of PR101** (0.226353), the next-best public score. Both are
HNeRV-family architectures. Audited the components to find the mechanism.

## Component breakdown

```
PR106 (belt_and_suspenders adapter, t4 cuda):
  archive_bytes:  186,239
  pose_avg:       0.000034  → contribution: 0.018306
  seg_avg:        0.000671  → contribution: 0.067142
  rate:           0.004960  → contribution: 0.124009
  TOTAL:          0.209457

PR101 (hnerv_ft_microcodec, t4 cuda):
  archive_bytes:  178,258  (-7,981 vs PR106)
  pose_avg:       0.000171  (5.0× WORSE than PR106)
  seg_avg:        0.000663  (essentially identical to PR106)
  rate:           0.004748
  TOTAL:          0.226353

PR106 vs PR101 component delta:
  bytes:  -7,981 (PR106 pays MORE rate)
  pose:   +0.023049 (PR106 wins HUGELY)
  seg:    -0.000838 (PR106 marginally worse, irrelevant)
  rate:   -0.005314 (PR101 wins because smaller)
  TOTAL Δ: +0.016897 (PR106 wins by 0.017)
```

The win is **entirely in pose** — PR106 trades +0.005 rate (8KB more bytes)
for **-0.023 PoseNet** via 5× lower per-pair pose distortion. The
square-root-law gives that 5× pose-dist reduction → sqrt(5) ≈ 2.24× pose
contribution reduction.

## The mechanism: 8-stage training pipeline

```
PR106 source/submissions/belt_and_suspenders/src/stages/:
  codec_stage.py
  common.py
  stage1_v328_ce.py        — cross-entropy loss
  stage2_v331_softplus.py  — softplus smoothing
  stage3_v332_smooth.py    — additional smoothness regularization
  stage4_v332_qat.py       — quantization-aware training
  stage5_c1a_l7.py         — c1a loss config (architecture-conditional)
  stage6_lambda_sweep.py   — lambda hyperparameter sweep
  stage7_sigma_sweep.py    — sigma hyperparameter sweep
  stage8_muon_finetune.py  — Muon-optimizer final fine-tune
```

PR101 is just `hnerv_ft_microcodec` — a single-stage fine-tune. The
"belt_and_suspenders" name is literal — PR106 layers redundant training
techniques (8 sequential stages, each refining the previous).

## Implications for our roadmap

**This is VALIDATING signal, not a new lane:**

1. We CAN'T close the 0.017 gap by changing PR106's codec — we already use
   PR106's decoder as our anchor. The training is what makes it good.

2. The apogee_intN bit-width reduction OPERATES ON the already-well-trained
   PR106 decoder. We're getting "PR106's training quality + smaller bytes"
   — the best of both worlds.

3. The sidechannel paradigm (latent + yshift sidecars) ALSO operates on
   the already-well-trained decoder. Same composition.

4. To improve the TRAINING (not the codec), we'd need to:
   - Replicate or improve PR106's 8-stage pipeline (multi-week effort)
   - Run on H100 SXM (GPU compute >> $0.30)
   - That's research-grade lane work, not /loop polish

**Decision: NOT pursuing the training-side improvement.** The codec-side
$0.30-cost paths (apogee_intN dispatch + sidechannel proposals) remain
the highest-EV moves at /loop tick scale. The 0.017 gap is genuinely
"PR106 won the training game"; our job is to win the codec game ON TOP
of that.

## Bonus: PR101 already uses the x-trick

PR101 archive member is named `x` (1 byte), not `0.bin` (5 bytes). This
saves 8 bytes vs the standard ZIP-header layout — same trick used by
the PR106 xrepack variant (which scores 0.209451 vs adapter's 0.209457,
the documented 5.5e-6 Δ). PR101's smaller archive bytes are partly
explained by this rate-side optimization being shared.

So PR101 = standard codec polish + standard fine-tune.
   PR106 = standard codec polish + 8-stage training recipe.

The 8-stage training recipe is the entire 0.017 score gap.

## Cross-refs

- PR106 staged training source: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/stages/`
- PR101 single-stage source: `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/`
- Component breakdown discovery: enabled by dashboard log-parser fix (commit dbb0032d)
- Operator handoff snapshot: `docs/operator_handoff_snapshot_20260504.md`
- Score-aware sidechannel paradigm thread: `docs/INDEX_score_aware_sidechannel_thread_20260504.md`
