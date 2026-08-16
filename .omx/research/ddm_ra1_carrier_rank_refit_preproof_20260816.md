# ddm_ra1 — rfo2 rung 2 (carrier rank/refit) MEASURED: rate side is real, distortion side misses by 3–4 orders

Date: 2026-08-16 · Owner: ddm_ra1 (rate axis) · Frontier UNMOVED by this arm:
**hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, sha `80d9c8c6…`.
Axis of this arm: **exact coded bytes [MEASURED]** + **exact carrier-field MSE [MEASURED, closed
form]**. No scorer ran. `score_claim=false`. Cost: $0, no dispatch, no heavy launch.

STORES CONSULTED: `ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md` (route order + the rung) ·
`ddm_1058_composition_campaign_close_20260816.md` (rung-2 fire-condition) ·
`ddm_mp2_mixed_precision_receiver_close_20260815.md` (Stage-2 sealed design + exact Brotli race) ·
`ddm_pk2_pose_carrier_representation_20260809/RESULTS.md` (the 49-row prior) ·
`ddm_fd135_fractal_decomposition_20260810.md` (CAP1 anatomy) ·
`ddm_wd3_n120_family_disposition_20260816.md` · `ddm_gx1_gap_closure_composition_table_20260816.md` ·
`upstream/evaluate.py:63-64` · the shipped F26 receiver at
`generations/hv1_base_control/{cpr1,runtime}/`.

## Conclusion

**The rung has no supplier, and rung 2 is now the reason why — measured, not projected.**

The carrier rank cut is the one lever whose **rate credit alone exceeds the gap**, and that half is
now confirmed: **rank-4 saves 14,709 B**, clearing the 14,414 B rung by 295 B. gx1's projection
(14,774 B) was right to 0.44%. But the same cut destroys **30.6% of the carrier field energy** —
an RMS error of **12.53 grey levels** on frame 0 — under the **least-squares optimal refit**, which
is a strict lower bound on every rank-4 carrier that keeps the receiver's linear synthesis. PK2
already measured that a **13.6× smaller** perturbation moves `d_pose` by 2,700×–19,000×. The pose
budget at rank-4 permits **1.06×**. Rung 2 misses by three to four orders of magnitude.

This closes the last un-measured rung on rfo2's route. Rungs 1 and 3 were closed on 08-16
(`ddm_1058_…`, `ddm_wd3_n120_…`); rung 4 was closed the same day
(`ddm_td1_token_drop_schur_arithmetic_20260816.md`). **No rung on the route supplies −14,414 B.**

## Corrections to the charter and the relay

1. **The rung is 14,414 B, not 15,157 B and not 14,413 B.** 15,157 B was computed against the
   183,502 B e480b base, one pointer move ago. On the live 182,759 B base the continuous
   requirement is 14,413.402 B — but **archive bytes are integral and the target is strict**:
   saving 14,413 B leaves `S = 0.15000026786363613` (NOT sub-0.15); 14,414 B gives
   `0.14999960200468301`. Both readings agree the archive must be **≤ 168,345 B**; 182,759 −
   168,345 = **14,414**. rfo2 §"Derive-first gap allocation" made exactly this integral/strict
   argument for its own base and reached 168,345 B too. MEASURED.
2. **The denominator is verified.** `upstream/evaluate.py:64` computes `uncompressed_size` as an
   `rglob` sum, not a constant — but `upstream/videos/` holds only `0.mkv` at exactly
   **37,545,489 B**, so the constant is correct for this eval config. `25/37,545,489 =
   6.658589531221714e-7` S/B confirmed. The Catalog #812 hazard is real but not live here.
3. **gx1's pose budget needs reconciliation (does not change the verdict).** gx1 states the CUDA
   budget at 14,774 B permits `d_pose` to rise to **3.37× base**. I derive, from the frontier
   components: rate after the cut = 0.1216917 − 0.009837 = 0.1118547, so sub-0.15 requires pose
   term < 0.0085343 ⇒ `d_pose` < 7.283e-6 = **1.059× base**; the looser "no worse than the
   incumbent" bar gives **4.73× base**. 3.37× matches neither. Immaterial here — the measured
   damage is 3–4 orders past every candidate bar — but the number should be re-derived before it
   is reused.

## Instrument (and why its numbers are real)

The receiver builds **frame 0 of every pair** from the carrier
(`fx1_runtime_tree/inflate.py:621-672`):

```
basis  = normalized_basis(basis_codes.reshape(12,3,24,32) * basis_scales)   # interp→384×512, mean-sub, RMS-norm
carrier = einsum("bk,kchw->bchw", coeff, basis) / sqrt(12)
slave   = (127.5 + 64.0*carrier).clamp(0,255).round()   →  output[2*i]
master  = semantic(tokens, idx)                          →  output[2*i + 1]
```

Two facts follow, and both are load-bearing.

**(1) The carrier cannot touch seg — structurally, not empirically.** `master` depends only on
`semantic` and `tokens`; SegNet reads `x[:, -1, ...]`, the last frame of the pair, which is
`output[2i+1]` = `master`. The carrier writes only `output[2i]`. The paths are disjoint. PK2
confirms empirically: `d_seg` is **identical to all 12 digits (0.000289620308) across all 49 of its
scorer rows.** This resolves gx1's falsifier (b) by proof rather than by a byte check.

**(2) The field is bilinear in the coefficients**, so the exact error of any rank-r approximation
is a closed form in the 12×12 Gram matrix `G` of the RMS-normalised basis at evaluation
resolution — no frames need materialising, and no scorer is involved:

```
MSE(Δc) = (64²/12) · mean_b[ Δcᵦᵀ G Δcᵦ ]        (units: grey level², 0–255 scale)
```

`G` measured: diagonal exactly 1.0 (atoms are unit-RMS by construction), condition number 15.71.
Carrier signal energy = **512.75 grey²**, i.e. the whole carrier is an **RMS 22.64 grey-level**
modulation of a flat 127.5 field.

**The rank-r refit is the least-squares optimum** `c_r = (Gᵣᵣ)⁻¹ Gᵣ,₁₂ c`, so each MSE below is a
**lower bound** over every rank-r refit heuristic. A cut that misses under this bound misses under
all of them.

**Byte custody.** Candidates are encoded through the shipped CPR1 encoder and the shipped Brotli
q11 cell. The encoder is validated: it re-encodes the shipped carrier **byte-identically**
(22,307 B, sha `709ea928c2d73c59…`). Decode chain is the real one: F0C1 split → CAP1 →
`materialize_cpr1` → `decode_compact_carrier`. Pricing domain is canonical CPR1 + Brotli
(baseline 22,278 B); the shipped F0C1/CAP1 container is **117 B** tighter, recorded and immaterial
at this scale.

## The measured curve

Baseline 22,278 B in the pricing domain. `MSE` is the least-squares-optimal refit (lower bound).

| rank | coded B | saved B | archive if adopted | rate credit S | MSE (grey²) | RMS grey | energy lost | supplies 14,414 B? |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 12 | 22,257 | +21 | 182,738 | 0.000014 | 2.2e-27 | 0.000 | 0.0% | no |
| 11 | 20,611 | +1,667 | 181,092 | 0.001110 | 4.921 | 2.218 | 1.0% | no |
| 10 | 18,655 | +3,623 | 179,136 | 0.002412 | 13.700 | 3.701 | 2.7% | no |
| 9 | 16,666 | +5,612 | 177,147 | 0.003737 | 20.219 | 4.497 | 3.9% | no |
| 8 | 14,691 | +7,587 | 175,172 | 0.005052 | 36.196 | 6.016 | 7.1% | no |
| 7 | 12,914 | +9,364 | 173,395 | 0.006235 | 81.080 | 9.004 | 15.8% | no |
| 6 | 11,229 | +11,049 | 171,710 | 0.007357 | 119.079 | 10.912 | 23.2% | no |
| 5 | 9,269 | +13,009 | 169,750 | 0.008662 | 133.874 | 11.570 | 26.1% | no |
| **4** | **7,569** | **+14,709** | **168,050** | **0.009794** | **156.926** | **12.527** | **30.6%** | **YES** |
| 3 | 5,708 | +16,570 | 166,189 | 0.011033 | 177.683 | 13.330 | 34.7% | YES |
| 2 | 3,884 | +18,394 | 164,365 | 0.012248 | 259.770 | 16.117 | 50.7% | YES |
| 1 | 1,940 | +20,338 | 162,421 | 0.013542 | 339.277 | 18.419 | 66.2% | YES |

Rank-12 returns MSE 2.2e-27 — the identity check the instrument owes.

## Falsifier disposition (gx1's, adopted verbatim)

- **(a) Byte falsifier — DOES NOT FIRE.** Threshold was "< 4,000 B saved at every rank". Measured:
  ranks 10 through 1 all exceed 4,000 B; rank-4 reaches 14,709 B. **The rate side passes.** Byte
  scaling is near-proportional in atom count, as projected — unlike PK2's coefficient-only low rank,
  which saved 8 B at rank 11 because it left all 12 basis atoms in place. Dropping *atoms* removes
  basis symbols and coefficients together; that is why this rung was worth measuring.
- **(b) Seg falsifier — PASSES, structurally.** Proven at source above, plus PK2's identical
  `d_seg` across 49 rows.
- **(c) Pose falsifier — REFUSED AT THE PRE-GATE. Do not buy the advisory row.** The pre-gate is
  #1058's ("projected pose ΔS within ~2× of the bar") and PK2's (`carrier-product MSE < 2.5e-6`).
  Rank-4's MSE is **156.9**, i.e. **6.3e7×** PK2's gate. Even rank-11 — one atom, optimally
  refitted — is 4.92, about **2e6×** the gate. Calibration against measured pose: PK2's
  single-dimension drops carry naive MSE ≈ 11.6–30.6 in these units and measured `d_pose`
  **0.0556–0.388** against a base of 2.01e-5, i.e. **2,700×–19,000×**. Rank-4's error is **13.6×
  larger than the smallest of those**. The rank-4 budget permits **1.06× base** (sub-0.15) or
  **4.73×** (merely not worse). Projected `d_pose` at rank-4 is O(0.05–0.5) ⇒ a pose term of
  **0.7–2.2 S** against a rate credit of **0.0098 S**.

`verdict_scope`: **FAMILY** — every rank-r carrier that preserves the receiver's linear synthesis
`einsum(coeff, normalized_basis)`, refit included, because the least-squares optimum bounds them
all. **NOT** covered: a carrier retrained from scratch at lower rank with pose in the training loop
(rate-aware QAT), which is PK2's own reactivation path and remains untested by construction. That
form must clear the same pre-gate before it costs anything.

## Two exact side findings

1. **`basis_scales` is a gauge — 48 B of the carrier carry no information.** The receiver
   RMS-normalises each atom *after* applying its scale, so any positive per-atom scale cancels
   exactly. Measured: all 12 signs are `+1`, and replacing every magnitude with 1.0 perturbs the
   normalised basis by **2.24e-7** (float32 noise). Byte value, MEASURED: **−7 B** through Brotli
   with no format change (22,278 → 22,271); ceiling **48 B = 3.196e-5 S** if the field is dropped
   from the format, which is legal because the receiver lives in the free `inflate.py`. Above the
   #1044 1e-5 naming bar, far below the rung. Owner: whoever next opens the carrier format.
2. **The shipped CPR1 encoder in-repo is byte-exact against the frontier archive.** Anyone pricing
   a carrier candidate can use it directly and get real coded lengths.

## Honest status of the −14,414 B rung

**NO SUPPLIER.** All four rfo2 rungs are now measured, and the route is exhausted on this base:

| rung | mechanism | status | receipt |
|---|---|---|---|
| 1 | mixed precision / post-hoc semantic edits | **MEASURED DEAD** (family; pose 3.8–5.0×) | `ddm_1058_composition_campaign_close_20260816.md` |
| 2 | **carrier rank/refit** | **MEASURED DEAD** (family, this memo): rate passes at 14,709 B, pose misses by 3–4 orders | this memo |
| 3 | nested-width distillation | **MEASURED DEAD** (family @65ep, parked with ladder) | `ddm_wd3_n120_family_disposition_20260816.md` |
| 4 | token drop + coder refit | **MEASURED DEAD** (≥16-bit structural, ≥8-bit by 1.2%) | `ddm_td1_token_drop_schur_arithmetic_20260816.md` |

The recurring mechanism across all four is now one sentence: **on this vehicle every byte that can
be removed post-hoc is load-bearing for pose, and by two to four orders of magnitude more than its
rate value.** Rung 2 is the sharpest instance because it is the only one where the rate side
actually delivered — 14,709 B, enough on its own — and the vehicle still refuses it.

That points where rfo2, #1058 and gx1 already pointed: the remaining routes run through **joint
descent**, not post-hoc editing. A carrier retrained at rank 4 with pose in the loop is the one
form this memo does not close, and it now has a cheap, exact, scorer-free pre-gate to clear first.

## Receipts

- `experiments/ddm_ra1_carrier_rank_refit_preproof.py` — the instrument (ruff clean).
- `/Volumes/APDataStore/pact/ddm_ra1_carrier_rank_refit_20260816/retained/CARRIER_RANK_REFIT_PREPROOF.json`
  — full arithmetic, custody block, Gram diagonal, environment.
- `…/retained/payloads/rank{01..12}_refit.{raw.bin,br}` — **all 24 candidate payloads retained**
  with sha256 + byte count per row (ALWAYS-KEEP-THE-PAYLOAD).
- Custody verified at run time, fail-closed: `outer_carrier.bin` 22,242 B `196f0e51…` ·
  `carrier.raw.bin` 22,219 B `065fce08…` · `carrier.br` 22,161 B `fd14aabc…` ·
  `archive.repeat.zip` 182,759 B `80d9c8c6…` (**the frontier archive**).
