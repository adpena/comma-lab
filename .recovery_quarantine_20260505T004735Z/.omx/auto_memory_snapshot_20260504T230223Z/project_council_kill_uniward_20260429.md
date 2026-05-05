---
name: Council 5/5 unanimous — KILL Lane UNIWARD standalone
description: 2026-04-29 extreme-rigor council debate after UNIWARD v8 = 1.14 ≈ Lane A noise. UNIWARD encoder pipeline is a no-op on the archive bitstream — no stacking can move the score without an SLI1 inflate-time decoder, which is dominated by Lane Ω cost.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
UNIWARD v8 LANDED 1.14 [Modal-T4-CPU] with anchor fix (full-res 384x512 masks). Same archive bytes (694KB) as Lane A 1.15; pose 0.0045 vs 0.0034 (worse), seg 0.0046 vs 0.0040 (worse), rate identical. The encoder pipeline is a **no-op on the bitstream** without the SLI1 inflate-time decoder.

Council debate 2026-04-29 (extreme rigor, all 5 voices on record):

- **Yousfi**: UNIWARD targets DCT-domain SRM detectors; our SegNet is pixel-domain CNN with stride-2 stem that loses fine textures UNIWARD treats as cheap. Wrong blind spot. KILL standalone.
- **Fridrich**: UNIWARD philosophy correct ("spend bits where detector is blind"), but right CNN adaptation is adversarial cost = 1/|∂logit/∂pixel|, which TTO already does implicitly. UNIWARD-as-prior on top of TTO is redundant.
- **Quantizr**: Doesn't use UNIWARD; uses kl_on_logits(T=2.0). Selfcomp #56 at 0.38 also doesn't use UNIWARD ("same trick as Quantizr" = KL distillation). UNIWARD empirically dominated by KL distill + smaller models + weight self-compression.
- **Hotz**: v8 archive identical bitstream to Lane A. Iterating on a no-op cannot move the score. SLI1 build cost ≈ 1 week + compliance review; Lane Ω is the cheaper path to similar score gain.
- **Contrarian**: Steelman lives in Selfcomp's "PoseNet trained with affine-transformed learned image" trick — UNIWARD philosophy in PoseNet domain. Fork to NEW Lane LI; UNIWARD-as-mask-prior dies.

**Verdict (5/5 unanimous)**: KILL Lane UNIWARD standalone. No v9/v10 dispatches.

**Stacking analysis**:
- UNIWARD + Lane G v3: redundant (both TTO). Predicted: 1.04 (no change).
- UNIWARD + Lane W: encoder still no-op. Predicted: ≈ Lane W alone.
- UNIWARD + Lane Ω: encoder still no-op. Predicted: ≈ Lane Ω alone.
- UNIWARD + SLI1 inflate decoder: only path to score movement; cost ≈ Lane Ω alone but with compliance risk.

**How to apply**:
- Reject any future "Lane UNIWARD vN" without an SLI1 decoder spec attached.
- Lane LI (Learned-Image PoseNet fork) is the live successor — defer dispatch until Lane W lands to avoid double-spending on rate-attack work.
- The lesson: a measurement showing "score within noise of baseline" with "archive bytes identical to baseline" is **not a UNIWARD measurement** — it's a pipeline measurement. Future encoder-only experiments should declare expected bitstream delta as a precondition.
