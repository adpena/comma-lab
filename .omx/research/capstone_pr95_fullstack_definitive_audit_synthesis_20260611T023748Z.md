# Capstone + PR95 full-stack DEFINITIVE audit synthesis (2026-06-11) — 5 audits converged

**Source:** the operator's "need another pass of all capstone + pr95 full stack." Five independent audits
this session converged: (1) #76 inert-loop fix, (2) subagent B per-step optimizer audit, (3) the adversarial
results review, (4) subagent A's curriculum port + its QAT-vjp bug fix, (5) the Quantizr-pose audit
(`ffca58a92`), and (6) the comprehensive full-stack auditor. This memo is the converged verdict + re-plan.
**Two binding conclusions; one strategic decision.**

## CONCLUSION 1 — the d_seg floor (~0.008) is a POISONED RECIPE (fixable), not capacity
Confirmed by 2 audits, file:line. The dominant poison is **[B1] NO cosine LR schedule** anywhere in the MLX
stack (PR95 cosine-anneals both LRs per stage; ours is constant `muon_lr=3e-2`, 150× PR95) → d_seg dithers
at the noisy floor instead of annealing into the argmax-correct basin. Compounded by **[A1] the capstone
trains NO weight-EMA and exports LIVE weights** (EMA non-negotiable violated; `ema_decay`/`use_ema_for_eval`
are dead fields) + **[B2] optimizer bias-correction/cosine never restart per stage**. The loss-form
curriculum (A landed) is the other half. **FIXABLE — B1+A1 are ~30 LOC + tests.**

## CONCLUSION 2 — the pure-VQ-NeRV carrier is STRUCTURALLY WRONG for pose (the big finding)
The Quantizr-pose audit (`ffca58a92`) settled it with a real A/B + the Quantizr source. **Quantizr's actual
mechanism** (`pr81 qzs3 intake`): a MASK-CONDITIONED generator (`JointFrameGenerator`) — stores the 6-d GT
pose (FROZEN), renders a `SharedMaskDecoder` trunk from a **FULL stored 384×512 mask per pair**, applies
FiLM **inside a conv residual block on the MOVING frame only**, trains with plain AdamW + cosine + EMA +
FP4-QAT. **Achieved d_pose ≈ 0.00051 (the tube).** OUR capstone's per-pair carrier is an **8-bit VQ index
(600 bytes total)** + a single per-channel affine FiLM. **A FiLM over a CONTENT-FREE latent cannot
synthesize the spatially-structured ego-motion flow the FastViT PoseNet reads** — synthetic evidence: it
holds d_pose only to ~1e-2 even when reachable. The FiLM→Muon hypothesis (B3) was **empirically REFUTED** as
the oscillation cause (Muon-FiLM reaches LOWER d_pose than AdamW-FiLM) — the binding constraint is **carrier
geometry (D3)**, not the optimizer. **The capstone needs the Quantizr-faithful `[mask-blob]⊕[pose-store]`
carrier (mask-conditioned trunk + FiLM-in-conv-block on the moving frame + frozen pose), NOT a pure-VQ-NeRV
latent.** This is the reviewer's "factorized carrier" + #83 representation-audit lesson, now with the
SPECIFIC proven mechanism. (Note: a stored 384×512 mask per pair is kilobytes — the byte budget must be
re-derived for the mask carrier; this is the same representation Quantizr shipped at 0.33.)

## CONCLUSION 3 (gate) — the ADVISORY is DECOUPLED from a real contest eval [CRITICAL E1]
Three independent decouplings: **[A1]** export live not EMA weights; **[A2]** advisory d_seg/d_pose measured
on the LIVE bicubic render, but the archive is int8 (quant loss never scored — PR95 re-parses + evaluates
the RELOADED decoder); **[A3]** inflate uses BILINEAR camera upscale while training/eval use BICUBIC (the
parity test uses bilinear on both sides so it CANNOT catch this). ⟹ **no capstone advisory number is a
trustworthy `inflate.sh→evaluate.py` predictor today.** A funded CUDA port now would chase a number the
local advisory does not represent. **Plus [C1]:** the bridge silently depends on a global yuv6 monkey-patch
done by a different module; no fail-closed assertion → a future caller with an un-patched scorer gets a
SILENTLY pose-inert loss. (Pose pixel-grad is also ~200× weaker than seg at 100:1 weights — pose is a
structurally small lever on this carrier even when flowing.)

## THE 5 HIGHEST-EV FIXES (carrier-INDEPENDENT — needed for any learned carrier)
1. **B1 cosine LR schedule** (both LRs, per-epoch, per-stage restart) — dominant d_seg-floor fix.
2. **A1 weight-EMA in the capstone** (build/update/snapshot-restore-eval/EXPORT the shadow, decay 0.997).
3. **A2+A3 score the RELOADED int8 archive + fix inflate to BICUBIC** — makes the advisory predict the real
   eval (closes E1). Add an `inflate.sh→evaluate.py` smoke on a tiny real archive.
4. **C1 bridge fail-closed yuv6 assertion** — cheap insurance vs a silent pose-inert regression.
5. **B2 per-stage optimizer-state/bias-correction reset** (couples with B1).
Then B4 (EMA 0.997) / B5 (clip retune post-B1) / B6 (FiLM in C1a-set) / D1 (per-stage re-eval cost) / A4
(delete dead `_exact_d_pose`) / C2 (reconcile the two FiLM impls) integrate on top.

## LAUNCH VERDICT (from the comprehensive auditor — binding)
**Do NOT launch a long base_ch=24 curriculum daemon yet.** Minimum gate: land **B1 + A1**, then **A2/A3**,
then a **200ep seeded constant-vs-cosine A/B** to learn whether d_seg crosses below ~0.003 (and an honest
capacity re-classification if not). C1 is a cheap same-batch add. THEN decide the carrier (Conclusion 2).

## WHAT IS GENUINELY SOUND (don't re-audit — confirmed across audits)
The 4 stage seg-losses + C1a + fake_quant are bit-parity-tested vs torch (real behavioral tests); VQ STE +
codebook EMA correct; bit-pack/zigzag/int8 codec exact-invertible; Muon Newton-Schulz coeffs + partition
match PR95; eval_roundtrip faithful in the LOSS path; numpy↔MLX render parity real; the d_pose-roundtrip fix
(audit #4) landed; authority tags clean; `force_film_to_adamw` correctly default-OFF on a refuted
hypothesis; the QAT-vjp primal-restore is correct. #79/lever-G/the-8-no-moves/pointer-hygiene all sound.

## STRATEGIC DECISION FOR THE OPERATOR
The recipe/honesty fixes (B1/A1/A2/A3/C1) are unconditional — integrate them regardless. But **Conclusion 2
is a CARRIER PIVOT**: the pure-VQ-NeRV capstone can (with the recipe fixes) become a trustworthy d_seg
machine, but it **cannot reach pose-tube** — pose needs the Quantizr-faithful **mask-conditioned
`[mask-blob]⊕[pose-store]`** carrier. Options: (a) fix the recipe → get the trustworthy d_seg verdict on the
VQ-NeRV (resolves "is the small basis d_seg-walled") AND in parallel design the mask-conditioned pose
carrier; (b) pivot the whole capstone to the mask-conditioned carrier now (closer to Quantizr's proven 0.33
substrate). Recommend (a): the recipe fixes are needed either way + give the honest d_seg verdict cheaply,
and the mask-carrier is the bigger build to design deliberately.
