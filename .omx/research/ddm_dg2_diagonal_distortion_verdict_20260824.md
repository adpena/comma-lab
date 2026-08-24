# The diagonal is ENTERED and REFUSED — 686× overwhelmed, and POSE is 93% of the damage

**Date:** 2026-08-24
**Owner:** MAIN (dg2's named next measurement; the arm's byte leg was scorer-free)
**Axis:** `[macOS-CPU advisory / n600 / cpu_env_mismatch_advisory]` — **NOT a score claim.**
`score_claim: false` · `promotion_eligible: false` · `promotable: false`

STORES CONSULTED: `.omx/research/ddm_dg2_diagonal_reentry_20260823.md` (commit `8606e28442`,
including my `CORRECTION-20260824`) · `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate
**CITED, not re-derived**) · `/Volumes/VertigoDataTier/pact/ddm_mst1_manufactured_stage_split/advisory_r1/work/contest_auth_eval.json`
(the matched base) · task rows `#1215` (the empty 2×2 cell), `#1221` (jf1's failed control),
`#1222` (renderer-carries-pose), `#1237` (the half-updated pin), `#1238` (the break-even
correction) · memory `[[perfect-localization-is-worthless-the-address-is-the-tax]]`.

---

## Verdict first

**REFUSED. The diagonal cell — HPAC model and field moving together — is entered with a passing
control and measures net ΔS +0.720160, which is 686× the rate credit it buys.**

dg2's own PRIOR-LAW PREDICTION was *"REFUSED — the diagonal measures net-positive ΔS (worse),
consistent with a genuine local optimum rather than an axis-probing artifact."* **CONFIRMED**, and
not narrowly.

`verdict_scope`: **INSTANCE** — the k060000 rung (178,792 B, archive sha
`59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a`) of dg2's e0060 diagonal on
the dx2 body. The FAMILY verdict is not claimed here: the k040000 rung (179,426 B, sha
`31d99f0b…`) is LIVE as I write (pid 38282) precisely so the family is not closed from one point.

---

## 1. The measurement, matched-instrument

Both legs are macOS-CPU advisory, n600, the same firer, the same `upstream_eval_mirror_20260815`.

| | base (dx2) | candidate (k060000) | Δ |
|---|---:|---:|---:|
| archive bytes | 180,368 | **178,792** | **−1,576** |
| `avg_segnet_dist` | 0.0003474 | **0.00083204** | **+4.84640e-04** |
| `avg_posenet_dist` | 0.00014701 | **0.05056445** | **+5.04174e-02** |
| canonical S (recomputed from components) | 0.19318153076125097 | **0.9133410981506166** | **+0.720160** |

Base receipt: mst1 `advisory_r1`, archive `976f706d…` (dx2 EXACT), `device=cpu`, `n_samples=600`.
Candidate receipt: `.omx/tmp/main_advisory/dg2_k060000/attempt2/work/contest_auth_eval.json`
(35,321 B), rc=0, 1,042 s.

## 2. Where the damage is — POSE, not seg

| term | ΔS | share of damage |
|---|---:|---:|
| **pose** √(10·d_pose): 0.038342 → 0.711087 | **+0.672745** | **93.3%** |
| seg 100·d_seg | +0.048464 | 6.7% |
| rate 25·ΔB/37,545,489 | −0.001049 | (the credit) |
| **net** | **+0.720160** | **686× the credit** |

Against the CORRECTED bar (`#1238`): break-even Δd_seg = **1.049394e-05**; measured
**+4.846400e-04** = **46.183× over**. It fails on the seg leg alone, before pose is counted.

**This is `#1222` measured directly.** PoseNet scores the FRAMES, so the *renderer* carries pose —
and a joint field+model move is, by construction, a rewrite of the rendered frames. d_pose went to
**344×** its base. The diagonal's whole appeal was that it pays its addressing in model bits rather
than address bits ([[perfect-localization-is-worthless-the-address-is-the-tax]]); the measurement
says it pays instead in pose, at 686× the byte credit.

## 3. Why the instrument caution turned out not to bind (stated, not assumed)

dg2's byte leg was scorer-free, so no same-instrument seg baseline existed; comparing a macOS-CPU
advisory d_seg to dx2's **contest-CUDA** 0.00020139 would have been the `#1034` cross-instrument
genus — measured CPU-vs-CUDA seg gap ~1.43× against a 5.2% signal. I therefore adjudicated against
the matched CPU base 0.0003474 throughout, and did **not** rescale by the 1.4425× lineage ratio
(a level ratio applied to a delta is its own wrong-object move).

Having done that: **at a 46× seg miss and a 344× pose miss, no instrument gap of order 1.4× could
change the sign.** The discipline was correct to apply and the verdict is robust to it. That is
worth saying plainly rather than presenting the care as if it were load-bearing.

## 4. What this does and does not close

**CLOSES:** the k060000 diagonal rung, on a control that PASSED bidirectionally (dg2's positive
control reproduced the shipped 113,777 B stream byte-identically; its negative control detected a
one-token perturbation at 820.5×). `#1215`'s empty cell is now populated at one point, and
`#1221`'s "the cell was never entered" is discharged for this rung.

**DOES NOT CLOSE:** the diagonal FAMILY. k040000 is live. Its bars, derived exactly from its
−942 B credit (= 6.2724e-04 S):

- needs **Δd_seg < 6.272391e-06** — k060000 delivered 77× that;
- needs **d_pose ≤ 1.518592e-04**, i.e. a **+3.3%** degradation ceiling — k060000 delivered +34,295%.

The prior is overwhelmingly negative. It is nonetheless UNDECIDED, because the measured
amplification exponent ~16.7 **forbids interpolating distortion between rungs** — and closing a
family from a single rung is the exact defect `ny1` caught in my own work (`#1225`, `#1226`).
17 minutes and $0 buys the honest family verdict instead of an inferred one.

## 5. NOT CLAIMED

- No score. This is an advisory axis (`cpu_env_mismatch_advisory`, `evidence_grade: auth-eval env
  mismatch advisory`, GT lineage `PYAV_YUV420_TO_RGB` not the authority `DALI_NVDEC`). The pointer
  is untouched and no contest-CUDA row was bought — correctly: a 686×-over candidate does not earn
  a T4 fire.
- No claim about the diagonal at OTHER (model-direction × field-direction) pairs, other epochs, or
  other bodies. dg2 sampled e0060 at two k thresholds.
- No claim that the joint-move *idea* is dead. What is measured dead here is this construction of
  it — a token-threshold drop with model refit — on this body. The mechanism that motivated it
  (addressing absorbed into stored parameters) is untouched as a principle; it is the pose cost of
  rewriting frames that kills this realization.
- dg2's byte-leg numbers are unaffected and stand.

---

## 6. Own-vehicle frontier

dx2 — **S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`** — **UNMOVED.**
Gap to 0.12 = 0.028220 ⇒ shed 42,382 B at fixed distortion, or 150 B at zero distortion.
