# B1-R2 clean PR95 baseline — FLAT TREND verdict (answers the burning question)

UTC 2026-06-09 · claude · run `b1_229k_clean_20260609T085348Z` (COMPLETED ep3000, TRAIN_EXIT rc=0).
Axis: [macOS-CPU advisory] backend-only (promotion_eligible=false; authority needs dual Linux-x86_64
CPU + T4 CUDA). Frontier (pointer): 0.19199 [contest-CPU].

## Burning question (relayed): does a clean grad-clipped B1-R2 baseline show stable stage-1 SegNet
## chamber entry AND an ep-250 exact-eval trajectory that justifies continuing to ep-3000?

**ANSWER: PARTIALLY YES on stability, NO on the trajectory.**
- Stage-1 SegNet chamber entry STABLE: yes — grad-clip held 100% of steps, nan_inf=0, SEG proxy-loss
  stable ~1.1–1.6 (vs R1's divergent 18→400). The kitchen-sink fix WORKED for divergence.
- ep-250 → ep-3000 exact trajectory justifies continuing: **NO.** The run ALREADY completed ep3000
  (it finished while away); the COMPLETE trend is FLAT: d_seg stuck ~0.50, d_pose ~160, S ~90, across
  ALL 8 stages incl. QAT + Muon. R2 fixed R1's DIVERGENCE but not the NON-LEARNING beneath it.

Trend (12 backend-only exact points): see `MASTER_ROADMAP_v3_to_theoretical_floor_20260609.md §0`.
ep250=90.12 ⋯ ep3000=90.36; d_seg 0.5048→0.5048; d_pose 155.7→157.7; bytes 256072→254012.

## Strict scrutiny on the negative (operator directive: suspect ALL negative results)
REAL training failure, NOT a bridge/measurement bug:
- bytes vary across checkpoints (export reads real distinct weights, not a stub);
- d_seg varies 0.5041–0.6386 (inflate responds to weights);
- ep2750 (Muon) transiently WORSE 0.6386 then recovers — consistent with Muon orthogonalizing in a bad
  basin (not a fixed garbage output);
- d_seg≈0.50 is the signature of DEGENERATE frames → SegNet maps to one dominant class → ~50% of source
  pixels disagree. The renderer is not fitting the video.
- It is OUR HiNeRV implementation, NOT the PR95 paradigm (PR95 = 0.193 with this recipe).

## Verdict: INSPECT_BINDING_CONSTRAINT (binding=seg), auto_kill=False (Forbidden premature KILL)
The clean baseline is DEFERRED-pending-diagnosis, NOT killed. R1→R2 fixed divergence; R3 must fix
NON-LEARNING. The divergence was a symptom; the non-learning is the disease.

## Root-cause hypotheses (Phase-0 diagnosis, ordered by likelihood)
1. **No RGB-reconstruction anchor (MOST LIKELY).** B1 trained scorer-DISTILLATION only. HiNeRV must
   first MEMORIZE the video via RGB reconstruction, THEN scorer-aware fine-tune. Distillation-only
   through frozen argmax-hinge + YUV6 → weak gradients → degenerate collapse.
2. Inflate-resolution mismatch (historical 48×64 catastrophe class).
3. Differentiable-YUV6 / eval-roundtrip gradient path severed.
4. Architecture mis-sized / coordinate-latent wiring wrong.

## Phase-0 diagnosis (NEXT; $0 local MLX, MVP-first, falsifiable)
Train HiNeRV with a DIRECT RGB-reconstruction loss (L2 vs `upstream/videos/0.mkv` frames) ~300 ep;
measure exact d_seg. PREDICTION: if hypothesis 1, d_seg drops sharply (renderer fits the video) →
add RGB anchor as the curriculum BASE, scorer-distillation as fine-tuning refinement. If d_seg stays
~0.50 → architecture/inflate bug → inspect inflate output resolution + single-frame overfit test.

## What R2 DID prove (the infrastructure win)
The full self-driving loop is REAL + validated end-to-end on real checkpoints: checkpoint → EMA export
→ backend-only HIV1 archive → inflate → 600-pair exact eval → CandidateActionEvaluation → machine-
readable decision; the sequential trajectory harvester built the complete trend autonomously (12
points, idempotent, fail-closed, disk-hygiene clean ~3.6 GB peak). The machinery is ready; it now
needs a renderer that LEARNS to feed it.
