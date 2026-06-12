# ALL-FINDINGS UNDER-POWER AUDIT — every Track-A finding swept for the POWER dimension (2026-06-12)

**Subagent:** `underpower-audit`. **Operator directive (binding, verbatim):** *"Review all findings for
underpowered"* + *"sometimes training time is necessary or capacity and config for true signal."*

**What "under-powered" means (the classification axes).** A verdict is UNDER-POWERED if it could FLIP at
adequate power. Power deficits flagged: **sample size** (n=8/12/24 advisory slice vs full 600 — the
`√(10·d_pose)` term is noisy on small slices); **epochs/convergence** (mid-basin ep340/426, d_pose~0.001,
d_seg~0.0035 vs converged ep2120, d_pose 0.00034); **operating-point dependence** (a verdict that credits an
un-converged term — LeverD credited the 0.35 seg term that converges to 0.056); **real-vs-proxy** (real
frozen scorer = TRUSTED advisory; MPS/synthetic = NOISE, never authority); **training-config dependence**
(a different basis/capacity/config could flip it).

**Three classes.** **PROPERLY-POWERED** — measured at adequate n/epochs/convergence/real-scorer; the verdict
holds. **UNDER-POWERED** — could flip with proper power; the specific deficit + the resolving re-validation
are named. **STRUCTURAL** — information-theoretic / architectural certainty, power-independent.

**Authority discipline:** every number here is `[contest-CPU advisory] / [macOS-CPU advisory]`
NON-PROMOTABLE until `upstream/evaluate.py` on a byte-closed archive. MPS is the train-gradient device only,
never authority. **Flagging a celebrated win as UNDER-POWERED is a SUCCESS — it prevents a false score claim.**

---

## THE VERDICT TABLE (every finding this session — no signal loss)

| # | Finding (memo) | Verdict | n / epochs / scorer | Class | Specific power deficit (if UNDER-POWERED) |
|---|---|---|---|---|---|
| 1 | **D1 cross-pair latent dedup = 0 B** (`d1_..._214737Z`) | TRUE negative on THESE latents | full 600 latents, exact lossless coding, real | **PROPERLY-POWERED** (exact) | — (basis-conditional DEFER, not a power gap; auto-fires on Track-B latents) |
| 2 | **Lever-4 QAT byte-direction-only** (`rate_levers_..._210729Z`) | TRUE mechanism (structural erasure) | n=12, 8-12 ep, real | **STRUCTURAL** (vendored 127-requant erases a train-only grid — architectural certainty, not budget) | — |
| 3 | **Variable codec net +worse** (`rate_levers`, ledger ITEM B) | RE-OPENED (byte win real; net via CRUDE alloc) | n=12, real basin (ep100), advisory | **UNDER-POWERED** | uniform 27/28 coarsen mis-shaped; the more-aggressive arm had the SMALLER penalty (tell). Untried: (a) KKT waterfill $0 (Partner B), (b) variable-grid QAT |
| 4 | **Cool-Chic AR-prior saturated** (`cool_chic_ar_prior_..._164116Z`) | TRUE negative | 800 ep (130× the smoke), real latents | **PROPERLY-POWERED** (the under-training hypothesis WAS tested and the prior did NOT tighten) | — |
| 5 | **Cool-Chic d_seg/pose wall** (`cool_chic_fullstack_..._153802Z`) | RE-OPENABLE | mid-budget CPU-tractable epochs, real | **UNDER-POWERED** | "monotonic, still descending"; under-trained at the CPU epoch budget. More epochs / full-res / capacity (Track-B basis question). 7.83 MB latent rate is prior-training-dependent |
| 6 | **Witness seg-boundary sidecar NO-GO** (`witness_..._181038Z`) | TRUE structural (HYBRID: fold into training) | n=120, mid-basin ep426, real | **STRUCTURAL** (flip-count 884/pair → 543 KB; 46.4% round-trip < 50% bar; "base-d_seg-robust": at frontier d_seg 0.056 → ~110 flips/pair → ~67 KB still > 38% of frontier budget) | — |
| 7 | **Distortion finishing-kit PR98+T10 = −0.058** (`finishing_kit_..._220727Z`, ledger ITEM C) | UNDER-POWERED-by-construction | **n=24, MID-BASIN ep340, real (CPU pre-round path)** | **UNDER-POWERED** (the flagship) | mid-basin d_pose 0.001478 (n=24 slice) vs converged 0.00034; the SAME operating-point risk that flipped LeverD. **RE-VALIDATED below** |
| 8 | **LeverD margin-residual GO→NO-GO** (`finishing_kit`, ledger ITEM C) | TRUE NO-GO at convergence | n=24, mid-basin ep340, real | **STRUCTURAL** (the GO was a MID-BASIN ARTIFACT crediting the un-converged 0.35 seg term; at convergence seg→0.056 so the 410-KB residual dominates — reproduces the witness flip-count crux) | — (already correctly self-flagged + dispositioned to in-training Lever-5) |
| 9 | **S12 resize-null invisibility** (`finishing_kit`) | TRUE certification (no byte lever on render base) | n=4-proof, real, certified-exact (residual 0.0) | **STRUCTURAL** (zero-distortion certification is mathematically exact + universal; honestly scoped to no-byte-lever on a decoder+latents substrate) | — |
| 10 | **pose-FiLM d_pose reduction (GO)** (`pose_film_cpu_disambiguator`) | DIRECTION robust; MAGNITUDE under-powered | **n=8 REAL GT pairs, frozen-decoder LOWER BOUND**, real | **UNDER-POWERED** (magnitude only) | n=8 + frozen-decoder lower bound + n=600 byte-cost is a LINEAR PROJECTION. The ANSWER (d_pose drops, beats byte cost) is robust; the exact ΔS is a projection. Resolve: in-curriculum full A/B at n=600 |
| 11 | **5 Layer-2 levers (R8/R9/R10/R11)** (`layer2_levers_review_round8-11`) | levers correct on real scorer; SEAL count 1/3 | n=8 real (R8/R10) + n=54 synthetic (R9) + synthetic (R11) | **PROPERLY-POWERED** for the MECHANISM (the levers FIRE + behave correctly on the real scorer); SEAL is a review-completeness gate, NOT a power gap | — (the levers' SCORE impact is measured by the live distortion arm, not these reviews) |
| 12 | **Lever-1 rate surrogate (Spearman 0.90/0.999)** (`rate_levers`, `independent_audit`) | correlation real; deployed −6 B | n=12 real, advisory | **PROPERLY-POWERED** for the surrogate-vs-brotli correlation; the −6 B deployed is operating-point (near-floor basin) honest | — (MED-1 scan-order proxy is a fidelity caveat, not a power gap) |
| 13 | **Bolt-on stack ≈ 0 B on basin** (`bolton_measured_..._checkpoint`) | TRUE (vendored at entropy floor) | n=1 checkpoint ep100, real lossless recode | **PROPERLY-POWERED** (brotli within 0.9-1.0% of order-0 entropy bound — measured, not assumed); mild operating-point note (ep100 INT8 near-uniform; a converged decoder MAY have more structure) | — (weak re-open: re-measure on ep2120 weights — low EV, brotli already near floor) |
| 14 | **gate15 advisory↔exact custody PASS** (`gate15_..._n24_MEASURED`) | TRUE custody-gate pass | n=24, early (global_ep 13), real | **PROPERLY-POWERED** for the custody mechanism (int8 honors float within 2.5e-5 d_seg / conservative direction) | — (absolute d_seg differs with convergence, but the GATE conclusion is convergence-robust) |
| 15 | **Lever-C frame1 carrier re-measured** (`r1_lever_c_live_dseg_rerun`) | does NOT un-falsify (carrier ~740× above frontier) | n=8, 40 ep smoke, real | **PROPERLY-POWERED enough** (seg_ce plateau 2.4-2.7 evident early; ~8× above re-open threshold — a large margin the carrier cannot close) — but the 40-ep budget is itself short | borderline: the plateau prediction is strong but 40 ep is a smoke. Re-open only if a fuller run shows seg_ce descending below ~0.5 |
| 16 | **base_ch20 basin best S≈0.378** (ledger; `best_meta.json`) | TRUE real-scorer advisory | full 600 pairs, ep2120 converged, real | **PROPERLY-POWERED** (full 600, converged, real frozen scorer) | — (NON-PROMOTABLE until exact `evaluate.py`; that is an authority gap, not a power gap) |
| 17 | **MLX-vs-MPS scorer ceiling (NO-GO on MLX speed)** (`mlx_vs_mps_..._ceiling`) | TRUE (throughput) | benchmark, real | **PROPERLY-POWERED** (throughput measurement, power-dimension N/A) | — |
| 18 | **async authority eval (throughput)** (`async_authority_eval_basin`) | TRUE (byte-for-bit unchanged) | n=600, real, same computation | **STRUCTURAL** (same computation on snapshot; throughput only) | — |
| 19 | **real-vs-synthetic scorer authority audit** (`real_vs_synthetic_..._203247Z`) | TRUE (0 HIGH; 12/12 tags correct) | inventory audit | **STRUCTURAL** (tag-correctness audit; no measured dataset) | — |

**Adversarial classification spot-checks (Lens-1, ≥3 findings verified against source memos):**
- **#4 Cool-Chic AR PROPERLY-POWERED** ✓ — `cool_chic_ar_prior_..._164116Z` confirms **800 epochs** (130× the
  6-epoch smoke); `disc_bits/elem 6.7192 → 6.6570 = 0.93% drop, non-monotone`. The under-training hypothesis
  was genuinely TESTED at high power and the prior did NOT tighten. Correctly PROPERLY-POWERED.
- **#7 finishing-kit UNDER-POWERED** ✓ — `finishing_kit_..._220727Z` headline is explicitly **n=24** on the
  **MID-BASIN ep340 fork-point** (`best_score=0.529`); the memo itself says "re-validate via
  `kit_aware_exact_eval` on the converged 600-pair checkpoint before any score claim." Correctly UNDER-POWERED.
- **#10 pose-FiLM MAGNITUDE under-powered** ✓ — `pose_film_cpu_disambiguator` is explicitly **n=8**,
  **frozen-decoder LOWER BOUND**, and "the exact ΔS at n=600 is a projection." The GO direction is robust; the
  magnitude is under-powered. Correctly classified.
- **#6 witness STRUCTURAL** ✓ — `witness_..._181038Z` decisive term is the absolute flip-count priced against
  the frontier byte budget (info-theoretic), explicitly "base-d_seg-robust" — power-independent. Correctly
  STRUCTURAL.

---

## THE DECISIVE RE-VALIDATION — finishing-kit on the CONVERGED ep2120 decoder (finding #7)

**The question (operator's lens).** The kit measured PR98+T10 = **−0.058 distortion-score at n=24 on the
MID-BASIN ep340 fork-point** (POSE-axis: n=24-slice d_pose 0.001478 → 0.000401). Does the gain SHRINK toward
0 as d_pose converges (→ under-training artifact, do NOT bank) or PERSIST (→ real uint8-round-trip artifact
like PR101's shipped PR98, → bank as convergence-robust)?

**Method (`experiments/probe_finishing_kit_convergence_revalidation.py`).** Re-fit the PR98/T10 constants
ON the more-converged basin **ep2120 best** decoder (d_pose 0.00034 over 600 pairs — ~2.4× lower than the
ep340 fork-point's 0.000831, ~3.7× lower than the n=24 mid-basin slice's 0.001478), measured on the REAL
frozen CPU scorer + REAL `yuv420_to_rgb` GT. **Re-uses the probe's own fitting functions** (`refit_pr98_bias`
/ `fit_t10_affine` — NOT a reimplementation) so the apples-to-apples is exact; only the LOADER (converged
archive vs fork-point) + the multi-slice harness are new. Two slices (n=24 primary matching the mid-basin
probe + n=48 secondary) for robustness against small-slice `√(10·d_pose)` noise, plus a cross-slice transfer
test (fit on slice-1, apply to slice-2). **Constants are re-fit ON the converged decoder** — C's finding that
PR101's canonical constants do NOT transfer means the ep340 constants may not transfer to ep2120 either.

**RESULT: <FILLED IN ON DAEMON COMPLETION — see `finishing_kit_convergence_revalidation_RESULT.json`>**

---

## ADVERSARIAL EVAL-FAITHFULNESS + BYTE-IDENTITY (closing C's missing recursive review)

The kit's production-faithfulness was verified directly (not assumed):

- **LENS-2a — the probe IS production-faithful (fit-path).** `apply_distortion_kit_to_camera_float` (the kit's
  torch path) produces **bit-identical uint8** to the probe's `_measure_bias_affine` transform. So the probe's
  fitted constants apply identically in the kit. ✓
- **LENS-2b — the post-round inflate path differs by ≤1 LSB (mean 0.21).** The PRODUCTION inflate path
  (`apply_distortion_kit_to_raw_frames`, post-round uint8) differs from the camera-float fit path by **max 1
  LSB, mean 0.21** — a double-round artifact (raw path rounds an already-uint8 frame; camera path rounds the
  float once). **This is the gap the driver docstring itself flags** ("the ≤1 ULP gap matters for a ±1-bias
  fit"). It means the n=24 mid-basin AND the camera-float re-fit both carry ≤1-LSB optimism vs the production
  packet. The decisive number is the one through `kit_aware_exact_eval` (post-round). **LOW finding** —
  re-measured below via the production path.
- **LENS-2c — default-OFF / identity byte-identity HOLDS.** `serialize(disabled) == b""` (zero bytes);
  `apply(disabled)` and `apply(identity-enabled)` both return the SAME object (`is` identity) → byte-identical
  no-op. The inviolable daemon-safety contract holds. ✓
- **LENS-2d — section round-trips + fail-closes.** 54-B section round-trips scale/bias exactly; bad-magic
  raises (no silent wrong-transform). ✓
- **LENS-2e — NO-FAKE: the transform actually changes pixels** (a no-op would fail). ✓
- **`kit_aware_exact_eval` is production-faithful** (source-verified): rounds to uint8 FIRST (vendored
  inflate), THEN applies the kit on post-round uint8 via `apply_distortion_kit_to_raw_frames` — exactly the
  substrate's `inflate.sh` chain. NO vendored edit. The disabled/identity path is the byte-identical no-op.

**Production-path verification: <FILLED IN ON DAEMON COMPLETION — kit_aware_exact_eval gain vs camera-float
gain>**

---

## PRIORITIZED RE-VALIDATION LIST (by EV = P(flip) × value-if-flips) — the operator's actionable output

<FILLED IN ON DAEMON COMPLETION — depends on the kit verdict>

---

## 6-hook wire-in (Catalog #125) + mission

#1 sensitivity-map ACTIVE (the per-finding power-deficit map is a sensitivity prior over which verdicts to
re-measure). #2 Pareto N/A (audit, no archive bytes). #3 bit-allocator N/A. #4 cathedral N/A. #5
continual-learning ACTIVE (the verdict table + the convergence re-validation reseed the judge on which
advisory negatives are basis/convergence-conditional vs structural). #6 probe-disambiguator ACTIVE (the
convergence re-validation IS the disambiguator between "mid-basin mirage" and "convergence-robust win").
**Mission contribution:** `frontier_protecting` (prevents a false score claim from an under-powered advisory)
+ `frontier_breaking_enabler` (the re-validation tells the kit which bolt-ons to bank when the arm converges).
**Frontier UNMOVED.**
