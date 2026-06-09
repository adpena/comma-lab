# B1 clean-PR95 ep1000 verdict — the 21 dB "capacity" signal was PSNR, NOT d_seg (correction)

UTC 2026-06-09 · claude · `[macOS-CPU advisory]` / `exact_cpu_advisory` / `metric_family=exact_evaluate`.
NON-promotable; `score_roadmap_update_eligible=False`. The ep1000 clean-PR95 B2 exact eval, ingested
through V3, overturns a prior conclusion. Raw row: d_seg=0.504824, d_pose=168.83, bytes=255,262,
advisory_score=91.74 (vs off-spec R3 89.57). Candidate/decision rows on the SSD run dir.

## The finding
The CLEAN PR95-faithful baseline (zero novelty, no amplification — the contract-blessed, leaderboard-
proven recipe) ALSO mean-fields at ep1000: **d_seg ≈ 0.505**, identical to R1/R2/R3, int8/fp16, and
pact_nerv_vq. So "R1/R2/R3 failed only because of off-spec amplification" is FALSIFIED as the whole
story — even the clean recipe sits at d_seg≈0.50 at this epoch.

## The correction (overturns the "capacity is not the crux" read)
Arm A's **21.74 dB** was measured by the recon-fit probe as **PSNR-vs-source**, NOT as d_seg. These are
different metrics:
- 21.7 dB PSNR is MEDIOCRE/blurry (good video reconstruction is 35–40 dB; 21 dB ≈ 6% per-pixel error,
  visibly smooth — it captures the low-frequency mean and loses high-frequency detail).
- d_seg = SegNet argmax disagreement, which keys on **high-frequency structure at class boundaries**.
- A blurry 21 dB reconstruction flips SegNet argmax on ~half the boundary pixels → d_seg≈0.50; and
  PoseNet's motion/temporal detail is gone → d_pose explodes.
- Arm A PLATEAUED at 21.7 dB (flat ep50→ep100), so 21.7 dB is the carrier's intrinsic PSNR ceiling
  at this config — and that ceiling is INSUFFICIENT for the evaluator.

**Revised conclusion:** the 229K carrier is capable of a blurry ~21 dB mean, NOT of the evaluator-
fidelity the score requires. This single fact reconciles the universal d_seg≈0.50 across every config
(it was never "capacity is fine" — the capacity probe measured the wrong metric).

## Why PR95 reached 0.193 with a similar 229K decoder (the open question)
Two non-exclusive hypotheses, each testable:
- **H-undertrain:** PR95 used 29,650 epochs; our clean run is 3000 (10× short) and we read ep1000
  (33%). d_seg may descend with vastly more epochs. Test: let clean PR95 reach ep3000, then extend.
- **H-fidelity-ceiling:** the carrier's PSNR/evaluator-fidelity ceiling at 229K + this config is too
  low regardless of epochs; PR95's score-aware curriculum reached low d_seg DIRECTLY (not via PSNR) by
  optimizing the argmax/pose terms to convergence — which our run hasn't reached at ep1000.

## The decisive next probe (splits the hypotheses; mechanism, not score)
**Score the LIVE MLX render through the EXACT `evaluate.py` d_seg (argmax disagreement), NOT PSNR.**
The recon-fit probe measured PSNR-vs-source; it never measured the live render's d_seg. Add d_seg
scoring (the exact `DistortionNet` already loaded in the sanity ladder) to the live render:
- live-render d_seg ≈ 0.50 → the carrier genuinely cannot reach evaluator fidelity (the 21 dB is too
  blurry); the fix is capacity/fidelity (architecture, much more training, or direct-d_seg objective).
- live-render d_seg GOOD → the archive/inflate/export path corrupts good frames (bridge bug) — but the
  fp16/int8 authority trace already argues against this (both 0.50; roundtrip atol 5e-2).

The prior weight is now on H-fidelity-ceiling + H-undertrain, not the bridge.

## Route (authority-disciplined)
- This is `exact_cpu_advisory` → `mechanism_update_eligible=True` (directs the next experiment),
  `score_roadmap_update_eligible=False` (advisory ≠ score). No roadmap branch.
- NEXT: (1) the live-render exact-d_seg probe (decisive); (2) let clean PR95 reach ep3000 (undertrain
  check) — the heavy slot continues. Do NOT branch to codebook/optimizer/atoms — they all pay rent only
  against a carrier that reaches evaluator fidelity, which none has yet.
- The recon-fit capacity probe is HARDENED in hindsight: a capacity probe must measure the EVALUATOR
  metric (d_seg), not a proxy (PSNR) — else "capable" is a false read (the lr/clip lesson, recurring at
  the metric layer). Per the dual-optimization + non-arbitrariness discipline: measure the thing the
  objective actually charges.

## Cross-refs
`dual_optimization_principle_..._20260609.md` · `pact_evidence_constitution_20260609.md` (this row is a
textbook exact_cpu_advisory: real number, mechanism-only) · `adversarial_review_..._20260609.md` (the
metric-provenance discipline that should have caught PSNR≠d_seg earlier).
