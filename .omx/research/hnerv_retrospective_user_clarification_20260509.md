# HNeRV retrospective — operator clarification (2026-05-09)

<!-- generated_at: 2026-05-09T05:25:00Z, from_state_hash: hnerv_user_reframe -->

## Operator clarification (verbatim)

> "like the hnerv implementation or archive or other artifacts were just trained or engineered right, when we had explored hnerv or nerv and other stuff but never got such low scores, they were doing something right and differently that we haven't understood or accomplished yet"

## Sharper framing for subagent a1a9359d

The framing operator wants is **NOT** "what's the integration-discipline meta-lesson." That framing is half-right but it's a research-process answer. The operator is asking the harder forensic-technical question:

**WHAT SPECIFIC TRAINING / ARCHITECTURE / HYPERPARAMETER / DATA-PREP / LOSS-FORMULATION THING DID THEY DO RIGHT THAT WE STILL HAVEN'T REPLICATED?**

This is a "competitor reverse-engineering" question, not a "process improvement" question. Distinct from the codex `representation_integration_gap_audit` (which is the process answer) — the operator wants the THING.

## Required deliverables (refined)

In the diagnostic memo (`feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`), please add a new top-level section:

### "The Thing" — the specific technical secret(s)

Bullet list of concrete differences between leaderboard HNeRV (PR100/101/102/103/105) and our internal attempts (Lane 12, Phase A class-level), ranked by score-impact magnitude. For each:

1. **WHAT** the leaderboard did concretely (cite exact file:line in the public-PR clone if possible, e.g., `experiments/results/public_pr100_intake_*/source/hnerv/model.py:142`)
2. **WHAT** we did instead (cite our file:line, e.g., `src/tac/lane_12_nerv_mask_codec.py:73`)
3. **WHY** their version produces lower score (mathematical / engineering reason)
4. **HOW** to replicate it concretely (specific code change or training config)
5. **CONFIDENCE** that this single change moves our score by N (with bound)

Examples of the SPECIFIC THING to look for (NOT exhaustive — investigate empirically):

- **Training-corpus split**: did they train on the entire 1200-frame contest video, or specific subset (e.g., key frames only)?
- **Per-frame conditioning vector**: what dim, what initialization (random vs. positional encoding vs. learned), what schedule?
- **Encoder architecture**: ConvNeXt vs. their custom hybrid? Stride/dilation pattern?
- **Decoder NeRV cell**: specific channel mixing pattern, specific activation (Sine/SwiGLU/GELU)?
- **Bit-allocation**: did they hand-tune per-tensor precision, or use a learned hyperprior?
- **Quantization scheme**: per-tensor symmetric int8? per-channel asymmetric int4? With what calibration corpus?
- **Loss function**: pure pixel MSE? VGG perceptual? Differentiable scorer surrogate? KL distill of SegNet logits at T=2.0?
- **Optimizer + schedule**: AdamW with what β1/β2/eps/weight_decay? Cosine? With warmup?
- **EMA decay**: 0.999, 0.9995, 0.997?
- **Number of training steps / epochs**: 100K? 1M? With what eval interval?
- **Random seed selection**: did they cherry-pick a good seed across N runs?
- **Data augmentation**: any temporal shifting, random masks, dropout?
- **Distillation source**: did they use our stronger Q-faithful renderer as a teacher signal, or pure pixel reconstruction?
- **Postprocessing**: any test-time refinement / TTO / post-hoc bit-rate optimization?
- **Archive layout**: specific tensor-quantization-then-brotli order? Brotli quality 11? Custom AC? Specific section ordering?
- **Pose handling**: did they predict poses jointly with frames, or use teacher poses?
- **Mask handling**: did they predict masks jointly, or use a separate mask codec?
- **Inflate-time runtime**: what specific PyTorch ops? Did they avoid certain expensive ops?

## How to investigate

1. **Read the actual public PR clones** under `experiments/results/public_pr100_intake_*/source/`, `public_pr101_intake_*/source/`, `public_pr102_intake_*/source/`, `public_pr103_intake_*/source/`, `public_pr105_intake_*/source/`. These are the source of truth.
2. **Compare to our internal attempts** — particularly `src/tac/lane_12_nerv_mask_codec.py`, `experiments/train_score_gradient_pr101_finetune.py` (A1), and any HNeRV/NeRV experiments under `experiments/`.
3. **Look for "obvious" things we never tried** — even if they look uninteresting in isolation. E.g., a specific weight init scheme, a specific normalization, a specific BN/LN/GN choice.
4. **Pay particular attention to PR101/103 (silver/bronze)** since they share the HNeRV substrate and bracket the medal band on both axes. The DELTA between PR101 (gold) and PR103 (silver) at +0.002 score might encode the marginal "what makes a HNeRV better" signal.
5. **Match against our R_pose ≈ 5.04 calibration** — every leaderboard HNeRV submission has CUDA-CPU drift in the same band. The stale FastViT-T12 attention-compounding story is superseded: PoseNet's FastViT path is RepMixer/conv-style, and the active hypotheses are loader-byte drift (DALI vs PyAV), conv-kernel accumulation drift, and Hydra/head sensitivity. If our internal NeRV attempts had different R_pose, that's diagnostic.

## Why the operator is asking this RIGHT NOW

The substrate-vs-codec meta-pattern says "stop building codec stacks on score-naive substrates." But the operator is going one level deeper: **even on the right substrate, the leaderboard implementations have a TRAINING-TIME secret we haven't extracted.** Maybe it's a single hyperparameter. Maybe it's a 50-line architectural detail. Whatever it is, identifying it lets us:

1. Apply it directly to T1 (Ballé hyperprior end-to-end) before $80 dispatch
2. Apply it to T6 (Ballé+UNIWARD cross-paradigm) before $80 dispatch
3. Apply it to T10 (IB-Lagrangian co-trained scorer) before $40 dispatch
4. Apply it to Phase 3 (joint scorer-renderer-codec) before the $600-1200 dispatch

A single missed training-time secret could be the difference between Phase 1 landing 0.155 vs. 0.180. The operator wants to know the THING before any new GPU dispatch.

## Permanence requirements (unchanged from original prompt)

- New memory file PLUS CLAUDE.md NON-NEGOTIABLE section
- 3-clean-pass adversarial greenup
- Tag claims `[empirical: ...]` / `[diagnostic: ...]`
- DEFERRED-pending-research, NEVER killed

## References

- Original HNeRV-lessons subagent prompt: this memo, plus all original prompt context
- Codex review findings (cross-bug-class connections): `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
- Substrate-vs-codec meta: `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- Codex representation-integration gap: `~/Projects/pact/.omx/research/representation_integration_gap_audit_20260508_codex.md`
