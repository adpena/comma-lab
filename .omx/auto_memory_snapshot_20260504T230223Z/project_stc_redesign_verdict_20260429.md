---
name: STC redesign codex verdict — Hybrid AV1+residual is top redesign (78% endorse), 280-450KB reachable
description: 2026-04-29 PM. Max-rigor STC redesign brainstorm with 22-voice council. Verdict 16/22 say STC-as-primary is dead. Top redesign is Hybrid AV1+residual (use AV1 as predictor, STC encodes corrections only). Reachable 280-450KB vs current implementation's 21MB. Hard abandon rule: 24h smoke must yield <380KB total OR <100-150KB residual + scorer-improvement-worth-rate-cost.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Council top 3 redesigns

| Rank | Redesign | Endorse | Reachable bytes | Implementation cost |
|---|---|---:|---|---|
| 1 | **Hybrid AV1 + residual STC** | 78% | 280-450KB | 300-600 LOC, 1 day |
| 2 | **NeRV / Cool-Chic / codebook** | 44-46% | 150-450KB lossy / 250KB-1.2MB exact-ish | 600-1200 LOC, 2-3 days |
| 3 | **Stored-flow residual** (homography → RAFT) | 35% | 300KB-1.2MB | 700-1200 LOC, 2-4 days. flow_compress.py already at 76.8KB for 32 DCT coeffs |

Lower-ranked but documented:
- Wavelet-domain (28% endorse, multi-region topology dominates)
- Telescope foveation (30% alone, 55% stacked with AV1+residual)
- CLADE (12% — helps synthesis conditioning not exception positions)
- openpilot seeding (18% — public weights don't shrink shipped bytes)
- CDC / Codec Avatars (2% — wrong domain)

## Top redesign math (Hybrid AV1+residual)

- Base: AV1 monochrome at appropriate CRF gives 250-340KB
- Residual: STC encodes corrections where AV1 misclassifies (~1-5% of pixels)
- If AV1 misclassifies 1% with avg 1 byte/correction → +30KB residual
- Total: 280-370KB → **beats AV1's 421KB by 50-141KB → -0.033 to -0.094 score from rate alone**
- Distortion: lossless after residual application; same as clean argmax

Strict-scorer compliance: BOTH decoders are CPU.
- AV1 decode: ffmpeg CPU
- Residual decode: pure integer arithmetic
- No scorer load at inflate

## Use auth-eval scorer/PoseNet/SegNet directly at COMPRESS time

User direction 2026-04-29 PM: "we can also use the auth eval scorer and posenet and segnet implementations directly in the STC redesign and all others"

This unlocks:
- Hybrid AV1+residual: encode corrections ONLY where SegNet's logit gradient says score loss is largest (not every pixel difference). Massively reduces residual byte count.
- NeRV: train the tiny model with scorer-aware loss (not just MSE on argmax)
- Flow residual: use PoseNet to identify important regions for compression budget allocation

CLAUDE.md compress-time scorer use is permitted (it's the inflate-time use that's forbidden). All compress-time SegMap/PoseNet pipelines already do this. Apply systematically to all lanes including the STC redesign.

## Hard abandon rule (codex council)

Abandon STC entirely if 24h smoke does NOT yield ANY of:
- Total mask layer < 380KB
- Scorer improvement worth the rate cost with residual < 100-150KB
- A redesign retaining `>10M` exception pixels, `>1MB` mask payload, or eval-time scorer weights

If abandoned, spend remaining deadline on PSD / stacking / proven baselines.

## Implementation order (if proceeding)

1. Hybrid AV1+residual smoke ($0 local, 4h):
   - Build base AV1 at CRF {35, 45, 50, 55} on Lane A masks → measure base bytes
   - Compute residual = clean_argmax - av1_decoded_argmax (sparse)
   - Encode residual via RLE + arithmetic coder (existing `arithmetic_qint_codec.py`)
   - Total bytes vs 421KB AV1 baseline
2. If smoke shows <380KB total, dispatch CUDA validation on Modal T4 (~$0.20)
3. If 380KB target met, full lane integration with strict-scorer-rule compliant inflate path

## Cross-refs

- /tmp/codex_runs/stc_redesign_brainstorm.log (full transcript)
- project_lane_stc_clean_source_FALSIFIED_20260429.md (the now-undetermined original lane)
- project_grand_council_final_designs_20260429.md (top-5 EV ordering)
- src/tac/flow_compress.py:1 (existing 76.8KB DCT flow estimate)
- experiments/contest_auth_eval.py (compress-time scorer reference impl)
