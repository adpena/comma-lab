# Carrier-pivot correction: the pose problem is the 8-bit VQ-index, not "content-free" (2026-06-11)

**Context:** the Quantizr-pose audit concluded the pure-VQ-NeRV capstone is structurally wrong for pose
("content-free latent") and recommended the Quantizr-faithful `[mask-blob]⊕[pose-store]` carrier. This note
SHARPENS that — the mask carrier may over-correct.

## The sharper diagnosis
- **An amortized decoder CAN hold pose:** the frontier (PR101-class amortized HNeRV decoder) reaches
  **d_pose ≈ 2.9e-5 (the tube)**. So "amortized decoder can't do pose" is FALSE.
- **Our capstone's actual impoverishment:** the bundle has a 28-d per-pair latent (`latent_dim=28`) but
  **VQ-quantizes it to an 8-bit codebook index** (codebook 256; the #67 free-inflate exploit: fixed
  codebook is free code, only the index is budgeted). **8 bits/pair cannot encode 600 distinct
  ego-motions** → the per-pair content the FiLM/decoder can express is ~256 buckets → pose wanders 0.06-0.34.
- **The frontier's carrier:** PR95 stores the **28-d per-pair latent DIRECTLY** (temporal-delta + raw-LZMA,
  ~6% of the 177 KB archive ≈ 10 KB), which is BOTH rate-efficient AND content-rich (28 floats/pair ≫ 8 bits).
- **Quantizr's stored 384×512 mask** also reaches pose-tube but is **rate-WORSE**: the LOWER scoreboard
  measured partition-direct seg = 253,413 B (rate 0.169) > the amortized decoder's 162 KB (rate 0.118).
  Direct storage LOSES on rate; amortization wins.

## The corrected pivot (cheaper than the mask carrier)
The likely fix is **NOT stored masks** — it is: **drop the VQ quantization; store the rich 28-d per-pair
latent directly** (temporal-delta-coded like PR95 L25), which is the frontier's own proven carrier — both
rate-efficient (~10 KB for 600 pairs) AND pose-capable (the frontier reaches d_pose 2.9e-5 with it). This
keeps the capstone an amortized NeRV (rate-efficient) while giving the per-pair carrier enough bits for pose.

## The A/B to settle it (pose-half lever, after the d_seg daemon)
On the now-trustworthy stack (cosine LR + EMA + honest advisory), at base_ch=20/24, 48-100 pairs:
- **Arm 1 (current):** 8-bit VQ index per pair (codebook 256).
- **Arm 2 (corrected):** store the 28-d latent directly (no VQ; temporal-delta + LZMA), ~10 KB/600 pairs.
Predict: Arm 2's d_pose descends toward the tube (rich per-pair content) where Arm 1 oscillates; rate cost
~10 KB (still sub-0.15 budget). If Arm 2 holds pose AND the curriculum d_seg daemon breaks the ~0.008 floor,
the sub-0.15 path is the amortized NeRV with a stored-28-d-latent carrier — NOT the mask carrier, NOT the
8-bit VQ index.

## Open question for the operator's carrier decision
Three carrier options now on the table, ranked by my read of EV: (1) **store-28-d-latent** (frontier's
carrier; cheapest pivot; this note's recommendation); (2) mask-conditioned `[mask-blob]⊕[pose-store]`
(Quantizr-faithful; rate-worse; bigger build); (3) keep 8-bit VQ index (pose-walled — rejected). The d_seg
daemon (pid 1717) resolves the seg half; this latent-vs-index A/B resolves the pose half on the corrected
carrier. Authority: all `[macOS advisory]`, non-promotable.
