# B1 carrier crux — CONSOLIDATED: diverse-but-blurry decoder (HF-fidelity), NOT dead latents / bridge / codec

UTC 2026-06-09 · claude · `[macOS-CPU advisory]` / `exact_pair_scorer` + `mechanism`-only (NOT score
roadmap). The consolidated decisive diagnosis chain for the HiNeRV carrier, scoping the deep-review
subagent's PR95-fidelity manifest (`deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md`, in flight).

## The empirical chain (each step authority-classified, each rules out a hypothesis)
1. **clean PR95 ep1000 archive eval (exact_evaluate, advisory):** d_seg=0.5048, d_pose=168.8, score=91.74.
   The CLEAN recipe (zero novelty) ALSO mean-fields → off-spec-amplification is NOT the whole story.
2. **live-MLX-render exact-d_seg trace, N=48 (exact_pair_scorer, advisory):** live d_seg=0.5054 ≈ archive
   0.5048; SegNet collapses to one class. **CATEGORY A confirmed: the model renders evaluator-bad frames;
   the export/inflate/quant BRIDGE is EXONERATED** (consistent with fp16≈int8 + roundtrip atol 5e-2).
3. **latent-drive probe, ep999:** latents WOKE (cross-pair var coarse 6.4e-2 / mid 1.6e-2 / fine 7.2e-3,
   all alive); renders DIVERSE + non-fixed (f0 std 0.295, f1 0.405). **NOT dead latents.**
4. **PSNR context (prior):** pure-recon PSNR plateaus ~21.7 dB (blurry; good video 35-40 dB), and 21.7 dB
   PSNR is NOT low d_seg (PSNR ≠ argmax fidelity).

## The crux (what survives the rule-outs)
The decoder maps the (alive, diverse) per-pair latents to **diverse-but-BLURRY frames that lack the
high-frequency structure SegNet's argmax keys on** → SegNet sees ~uniform → one class → d_seg≈0.50; and
the two-frame motion/luma structure PoseNet needs is absent → d_pose≈170-206. The crux is **decoder
high-frequency fidelity**, NOT: dead latents (alive+diverse), the bridge (Category A), quantization
(fp16≈int8), or rate (rate term 0.17 of 91.74).

## What this scopes the PR95-fidelity manifest to (the deep-review subagent's job)
The diverse-but-blurry signature points at a MISSING / WEAK high-frequency path in our decoder vs the
PR95/HiNeRV reference. Prime suspects (the subagent verifies term-by-term, file:line):
- **bilinear-skip residual** — CLAUDE.md L18 says the PR95 block is `PixelShuffle + BILINEAR-SKIP + sin`.
  A missing/weak skip is the textbook cause of low-HF blur (the skip carries the sharp residual).
- **hierarchical positional encoding** — HiNeRV's defining feature injects multi-scale HF coordinates; a
  coordinate-poor decoder spectral-biases to the DC/mean (exactly the blur we see).
- **objective** — PR95 stage-1 is CE-Seg (drives SegNet logits directly); our boundary-hinge surrogate may
  permit the mean-field minimum the renderer fell into.
- **undertraining is NOT the sole fix** — diverse-but-blurry at a 21.7 dB PSNR *plateau* is a
  representational/HF ceiling, not just "needs more epochs" (though more epochs may sharpen if the HF path exists).

## Route (authority-disciplined; nothing preempts the manifest)
- This whole chain is `exact_cpu_advisory` / `mechanism_update_eligible` → it localizes the bug and directs
  the next experiment; it does NOT update the score roadmap (only a contest-axis `exact_evaluate` row does).
- **NEXT = the PR95-fidelity manifest** (subagent): if a HF component (bilinear-skip / hierarchical PE) is
  MISSING, that is the fix → restore it, re-trace live d_seg. If topology is faithful, the objective
  (CE-Seg vs hinge) + far more epochs are the levers.
- **DO NOT** branch to codebooks / PR110++ atoms / adaptive codec / optimizer exotica — all pay rent only
  against a carrier that reaches evaluator-fidelity, which this one does not.
- **DO NOT** train another long run before the manifest names the missing HF component.

## Artifacts (durable on the SSD run dir)
`b1_clean_ep1000_authority_trace.v1.json` (N=48 live vs archive) · `b1_clean_ep1000_latent_drive_probe.json`
(latents woke + render diversity) · `candidate_action_evaluation_b1_clean_ep1000.v1.json` +
`campaign_decision_b1_clean_ep1000.v1.json` (V3 rows). Cross-ref
`b1_clean_pr95_ep1000_verdict_psnr_is_not_d_seg_20260609.md` (the PSNR≠d_seg correction).
