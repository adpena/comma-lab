---
name: Nitrobrew (Tilde Research) lossless distillation — NOT-APPLICABLE to our portfolio
description: 2026-04-28 evaluated https://blog.tilderesearch.com/blog/nitrobrew per user mandate. Verdict NOT-APPLICABLE across all 5 of our distillation lanes. Mechanism is fused chunked-V softmax-KL kernel for transformer LMs with V≥50k vocab. Our distillation targets are pose regression / continuous features / 5-class spatial KL / pixel MSE — none touch large softmax. $0 cost delta. Documented to prevent re-evaluation in future sessions.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Mechanism (verbatim from blog + nitrobrew-release Apache-2.0 code)

Fused chunked-V softmax-KL kernel that streams over the vocabulary axis V in tiles of `chunk_V`, accumulating online softmax stats. Hidden states `h ∈ ℝ^d_model` are sent across devices instead of full logits `z ∈ ℝ^V`. Logits reconstructed locally via `z = W_U h` only inside the kernel and discarded after each tile.

## "Lossless" definition (narrow)

Bit-identical full-vocabulary KL vs **top-k-truncated KL** approximations. NOT bit-identical model outputs, NOT score parity, NOT weight parity. It is a memory/comms-side primitive, not a distillation algorithm.

## Hard requirements for any benefit

- `[V, D]` linear unembedding matrix
- Softmax over a large vocabulary axis (V ≥ 50k for the wins to be visible)
- Multi-device training where logit traffic dominates
- Without these: kernel simplifies to a regular matmul-then-KL with NO speedup

## Why NOT applicable to our portfolio

Verdict scores (1-10):
- Lane M-V3-clean (PoseNet-embedding distill): 1/10 — 6-D pose regression, no V axis
- Lane DI (openpilot supercombo): 1/10 — continuous feature L2/cosine
- Lane G v3 KL distillation: 2/10 — V=5 classes, KL tensor is 15MB at fp32, nothing to fuse
- Diffusion Teacher (1192 LOC, never deployed): 1/10 — pixel-space MSE/score-matching, not categorical
- DMD2 / Lane V-DMD: 1/10 — VSD + adversarial critic, no vocabulary KL

Our largest "vocabulary" is 5 (SegNet classes). Nitrobrew's chunk_V=2048 is meaningless when V=5. We train single-GPU on a 4090 — no inter-device logit traffic. Distillation memory footprints are 10-100MB, not the 30+GB regime where nitrobrew matters.

## Cost delta: $0

No applicable training-time delta across any current or planned distillation lane.

## When to revisit

If we ever:
- Distill from a transformer LM with a giant softmax (e.g., a CLIP/SigLIP text head)
- Use a VQ-codebook quantized scene token over V=8k-64k codes
- Train on multi-GPU where logit traffic dominates

None of these are in any current or planned lane. Researched-and-rejected.

## Cross-references
- `.omx/research/comprehensive_council_eval.md` — Diffusion Teacher REVIVED architecture (1192 LOC)
- `project_lane_m_v2_audit_council_findings_20260428` — Lane M lineage
- `project_lane_g_v3_landed_1_05_20260428` — current frontier with KL distill weight=0.002
