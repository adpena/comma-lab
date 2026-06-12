# ALL-FINDINGS UNDER-POWER AUDIT — every Track-A finding swept for the POWER dimension (2026-06-12)

**Subagent:** `underpower-audit`. **Operator directive (binding, verbatim):** *"Review all findings for
underpowered"* + *"sometimes training time is necessary or capacity and config for true signal."*

## TL;DR (data, not prose) — `[contest-CPU advisory] NON-PROMOTABLE`, frontier UNMOVED

1. **19 findings classified** (PROPERLY-POWERED / UNDER-POWERED / STRUCTURAL). **4 are UNDER-POWERED** (#3
   variable codec, #5 Cool-Chic wall, #7 finishing-kit, #10 pose-FiLM magnitude); the rest hold.
2. **DECISIVE RE-VALIDATION — the flagship finishing-kit (−0.058) SHRINKS to −0.003 at convergence (5.3%
   retained, a 19× collapse) → it was an UNDER-TRAINING ARTIFACT; do NOT bank the −0.058.** The mid-basin
   probe credited a pose-axis correction of the under-trained decoder (d_pose 0.001478); the converged ep2120
   decoder already has d_pose **0.0000954** (15× lower) and ANY fixed PR98 bias REGRESSES it (+0.028 / +0.003).
   Three independent n=24 probes on the converged decoder agree. This is the SAME operating-point trap that
   flipped LeverD — caught BEFORE it entered the stack. (Flagging a celebrated win as under-powered is a
   SUCCESS: it prevents a 19×-overstated false score claim.)
3. **Kit infrastructure is SOUND** (default-OFF byte-identical through `kit_aware_exact_eval`; production-faithful
   post-round path verified; ≤1-LSB pre-round optimism is the only gap, and it makes the gain SMALLER). Keep
   it WIRED; RE-FIT on the FINAL n=600 converged decoder; bank ONLY if a robust gain then survives.
4. **Highest-EV $0-local re-validation still open: #3 variable codec** (KKT waterfill, Partner B running).
   #5/#10 are real but training-time-gated (in-curriculum / Track-B).

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
| 7 | **Distortion finishing-kit PR98+T10 = −0.058** (`finishing_kit_..._220727Z`, ledger ITEM C) | **RESOLVED → SHRINKS to −0.003 (under-training artifact; do NOT bank)** | **n=24, MID-BASIN ep340** → RE-VALIDATED at **n=24, CONVERGED ep2120** | **UNDER-POWERED (CONFIRMED + RESOLVED)** | the −0.058 was a pose-axis correction of the under-trained decoder; at convergence d_pose is 15× lower so the gain collapses 19× to −0.003. **RE-VALIDATION below** |
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

**RESULT: SHRINKS — the −0.058 was overwhelmingly an UNDER-TRAINING ARTIFACT. Do NOT bank it.**

`[contest-CPU advisory] NON-PROMOTABLE`. Two independent measurements agree exactly (cross-validated baseline
dscore 0.27772): the full re-validation daemon's fresh re-fit (`finishing_kit_convergence_revalidation_RESULT.json`,
SLICE-1) + a fast 3-point transfer check (`finishing_kit_converged_fast_check.json`).

| quantity | MID-BASIN ep340 (n=24, the −0.058 claim) | CONVERGED ep2120 (n=24, this re-validation) |
|---|---:|---:|
| baseline d_pose | 0.001478 | **0.0000954** (15.5× lower) |
| baseline d_seg | 0.003532 | 0.002468 |
| baseline distortion-score | 0.47479 | 0.27772 |
| **full kit (PR98+T10) Δ, FRESH re-fit** | **−0.058** | **−0.003063** |
| PR98-only Δ, fresh re-fit | −0.047981 | −0.002493 |
| **retained fraction of the mid-basin gain** | (100%) | **5.3%** |

**Transfer test (do the mid-basin / canonical constants help the converged decoder?) — NO, they REGRESS it:**
- mid-basin best bias `[[+1,+1,+1],[−1,−1,−1]]` on the converged decoder → **Δ = +0.0284 (WORSE)**; d_pose
  goes UP 0.0000954 → 0.000345 (the fixed bias OVER-corrects the already-balanced converged decoder).
- canonical PR101 bias `[[+1,0,+1],[0,+1,0]]` → **Δ = +0.0029 (WORSE)**.
- the converged decoder's FRESH best PR98 bias is `[[1,0,1],[0,0,0]]` (a tiny frame_0-only nudge) — a
  DIFFERENT operating point than the mid-basin's `[[+1,+1,+1],[−1,−1,−1]]`, confirming the constants are
  decoder-STATE-specific (C's "PR101 constants don't transfer" extended: even the base_ch20 mid-basin re-fit
  doesn't transfer to the converged base_ch20 decoder).

**Mechanism (why it shrinks).** PR98+T10 was almost entirely a POSE-axis lever (the mid-basin gain was
d_pose 0.001478 → 0.000401). It worked by correcting the UNDER-TRAINED decoder's systematic per-frame
brightness/temporal-offset imbalance. The CONVERGED decoder — trained against the real scorer — has ALREADY
learned that balance: its baseline d_pose is 0.0000954, already **4× below** where the mid-basin kit ended up
(0.000401). There is almost no pose-axis distortion left to correct, so the fresh re-fit finds only −0.003
(within small-slice `√(10·d_pose)` noise of the floor) and any FIXED bias over-corrects. **This is exactly the
operating-point risk that flipped LeverD** (GO mid-basin → NO-GO converged): the mid-basin probe credited an
un-converged term.

**Disposition.** The finishing kit's PR98/T10 sections are NOT a banked −0.058 win. At convergence they are a
~−0.003 advisory at n=24 (sign-positive but tiny and noise-dominated). The kit's INFRASTRUCTURE is sound
(default-OFF byte-identical, production-faithful) and should remain WIRED so it can be RE-FIT on the FINAL
converged decoder at n=600 and applied IF the re-fit then shows a robust gain — but the **−0.058 headline must
NOT enter the stack**. Flagging this is a SUCCESS: it prevents a 19×-overstated false score claim.

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

**Production-path verification (`finishing_kit_production_path_verify_RESULT.json`, n=24).** Re-measured the
fresh-fit constants through `kit_aware_exact_eval` (the production-faithful POST-ROUND path):
- **OFF (production no-op): dscore 0.27772 — byte-identical to the baseline through the production path** (the
  default-OFF contract holds end-to-end, not just in the unit test).
- **FULL kit via production path: Δ = −0.002793** vs the camera-float fit's −0.003063. The
  **production-vs-camera-float gap is +0.00027** — the post-round path is marginally WEAKER, exactly the
  ≤1-LSB optimism LENS-2b predicted. The ±1-bias fit carries a small pre-round bonus that the production
  packet does NOT fully realize. This makes the converged gain even smaller (~−0.0028), reinforcing SHRINKS.
- NO-FAKE: the OFF arm being byte-identical-to-baseline through `kit_aware_exact_eval` proves the eval is not
  secretly applying the kit; the FULL arm differing proves the kit is not a no-op. ✓

---

## PRIORITIZED RE-VALIDATION LIST (by EV = P(flip) × value-if-flips) — the operator's actionable output

Only the UNDER-POWERED rows are re-validation candidates (PROPERLY-POWERED + STRUCTURAL verdicts hold).
Ranked by EV = P(verdict flips with proper power) × value-if-it-flips.

| Rank | Finding | P(flip) | value-if-flips | Cheapest resolving re-validation | $0-local or training-time? |
|---|---|---|---|---|---|
| ~~1~~ **RESOLVED** | **#7 finishing-kit PR98+T10** | **RESOLVED: SHRINKS to −0.003 (5.3% retained) — under-training artifact** | was LARGE (−0.058); now ~0 banked | DONE: re-fit + measured on converged ep2120 via 3 agreeing probes. Disposition: keep WIRED, RE-FIT on the FINAL n=600 converged decoder, bank ONLY if robust then | **$0-local (done)** |
| **2** | **#3 variable codec net** | moderate-high (the more-aggressive arm had the SMALLER penalty = the uniform 27/28 allocation is mis-shaped — a flippable tell) | moderate (decoder-weight rate is the LAST rate headroom after D1+Cool-Chic exhaust the latent rate) | (a) KKT reverse-waterfill — coarsen only lowest-sensitivity tensors, hold pose-sensitive at 127 ($0, no retrain) **— Partner B running**; (b) variable-grid QAT (decoder learns coarse-grid robustness) | (a) **$0-local**; (b) training-time (folds into a QAT stage) |
| **3** | **#10 pose-FiLM magnitude** | LOW that the DIRECTION flips (GO is robust); the MAGNITUDE/net-ΔS is the under-powered part | high (a real d_pose lever at ~1 KB — pose is the binding MARGINAL term below the crossover per CLAUDE.md SegNet-vs-PoseNet operating-point rule) | in-curriculum FULL A/B at n=600 (un-freeze the decoder; the frozen-decoder n=8 is a LOWER BOUND, full co-adaptation expected BETTER) | training-time (Modal/CUDA paired A/B) |
| **4** | **#5 Cool-Chic d_seg/pose wall** | moderate ("monotonic, still descending" = genuinely under-trained) | high IF it flips (Track-B latent-floor-below-HNeRV thesis) but the 7.83 MB latent rate is ~44× above HNeRV's floor — a long way to go | more epochs / full-res / capacity at the CPU-tractable budget ceiling | training-time (NOT $0; the binding question is convergence + capacity) |
| **5** (low) | **#13 bolt-on stack on ep2120 weights** | LOW (brotli already within 0.9-1.0% of order-0 entropy bound at ep100; a converged decoder may have marginally more structure) | small (≤ tens of bytes) | re-run `bolton_measured_on_basin_checkpoint` recode on the ep2120 best weights | $0-local (low EV — likely confirms the floor) |
| **6** (low) | **#15 Lever-C Config B (250k/214KB)** | LOW (d_seg 0.414 is 8× above the re-open bar; seg_ce plateau is the mechanism, not lag) | low (the carrier is seg-blind; ~740× above frontier) | re-run Config B to completion (the "second effect" config not run to completion) | training-time (low EV — the 8× margin is large) |

**The operator's actionable take:** rank 1 (the finishing kit) is the ONLY high-value $0-local re-validation,
and it is being resolved now (below). Rank 2 (the variable codec) has a $0-local arm (Partner B's KKT
waterfill) already running — that is the right next byte-lever attack. Ranks 3-4 (pose-FiLM magnitude,
Cool-Chic wall) are genuine but training-time-gated — they belong in the in-curriculum / Track-B campaigns,
not a $0 sprint. Ranks 5-6 are low-EV completeness items (do NOT spend the distortion arm's CPU on them).

---

## MY RECURSIVE ADVERSARIAL REVIEW (owner-run, 3 clean lenses)

**Lens 1 — the classification is correct (≥3 spot-checks against source memos).** Verified #4 Cool-Chic AR
(800 ep tested → PROPERLY-POWERED ✓), #7 finishing-kit (n=24 mid-basin ep340, memo self-flags re-validate →
UNDER-POWERED ✓), #10 pose-FiLM (n=8 frozen-decoder lower bound, n=600 projection → MAGNITUDE under-powered
✓), #6 witness (convergence-projected flip-count stays NO-GO → STRUCTURAL ✓). Every PROPERLY-POWERED call
cites adequate n/epochs in its source memo. **CLEAN.**

**Lens 2 — the ep2120 re-validation is REAL (NO-FAKE; would FAIL if the kit were a no-op).** (a) REAL frozen
CPU scorer (`load_frozen_distortion_net`) + REAL `yuv420_to_rgb` GT (no MPS, no PyAV-rgb24, no synthetic
fixture). (b) REAL converged decoder loaded via the SAME vendored `parse_archive`/`HNeRVDecoder` the driver
uses (baseline dscore 0.27772 cross-validated by TWO independent probes — the daemon's fresh re-fit AND the
fast 3-point check — to 5 decimals). (c) constants RE-FIT on the converged decoder (not assumed to transfer).
(d) the OFF arm through `kit_aware_exact_eval` is byte-identical-to-baseline → a no-op kit produces ZERO delta
→ the −0.058 claim CANNOT be reproduced by a no-op (it requires the under-trained decoder). The kit's
transform actually changes pixels (LENS-2e) and the FULL arm differs from OFF. **A result that would hold if
the kit were a no-op FAILS this test — and the no-op produces exactly 0, while the real kit produces −0.003.
CLEAN.**

**Lens 3 — robustness (stable across slices, given small-slice d_pose noise).** The SHRINKS verdict is
corroborated by THREE independent measurements at n=24 on the converged decoder: (i) the daemon's fresh
re-fit (−0.003063), (ii) the fast-check's fixed mid-basin/canonical biases (BOTH regress, +0.028/+0.003),
(iii) the production-path eval (−0.002793). All three agree the converged gain is ~−0.003 to 0 (vs −0.058
mid-basin) — a 19× collapse far exceeding any plausible small-slice `√(10·d_pose)` noise (the d_pose floor
itself is 0.0000954, and the SHRINKAGE is structural: 15× lower baseline d_pose = 15× less headroom). The
SLICE-2 (n=48) + cross-slice transfer arms of the daemon add a second independent slice; they were still
in-flight at write time (the no-early-break refine on the converged decoder is ~50 min CPU/slice under sibling
contention — durably checkpointed, no signal lost) but the n=24 verdict is already over-determined by the
three agreeing measurements + the mechanism (no pose headroom at convergence). **CLEAN.**

**3/3 CLEAN.** Counter never reset (no finding inside my own conclusions). The ONE finding I surfaced (the
≤1-LSB post-round gap, LENS-2b) is in the KIT's measurement path, not my audit's conclusions, and it
REINFORCES (not contradicts) the SHRINKS verdict.

## 6-hook wire-in (Catalog #125) + mission

#1 sensitivity-map ACTIVE (the per-finding power-deficit map is a sensitivity prior over which verdicts to
re-measure). #2 Pareto N/A (audit, no archive bytes). #3 bit-allocator N/A. #4 cathedral N/A. #5
continual-learning ACTIVE (the verdict table + the convergence re-validation reseed the judge on which
advisory negatives are basis/convergence-conditional vs structural). #6 probe-disambiguator ACTIVE (the
convergence re-validation IS the disambiguator between "mid-basin mirage" and "convergence-robust win").
**Mission contribution:** `frontier_protecting` (prevents a false score claim from an under-powered advisory)
+ `frontier_breaking_enabler` (the re-validation tells the kit which bolt-ons to bank when the arm converges).
**Frontier UNMOVED.**
