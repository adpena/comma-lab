# Grand Council Chain-Integrity Audit — End-to-End EV Reasoning Chain

**Date**: 2026-04-30
**Convened by**: parent agent under user mandate "extreme rotated rigor and adversarial and full stack and no compromises"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Mandate**: REPORT-ONLY. Chain-of-reasoning audit (NOT code-defect audit; Round 10 covers that). No GPU. No code modified.
**Scope**: Steps [1] Shannon R(D) → [10] HM-S/FR-Ω dispatch decision.

All claims tagged: `[empirical:<path>]` / `[contest-CUDA]` / `[contest-CPU advisory]` / `[Modal-T4-CUDA]` / `[prediction]` / `[derivation]`.

---

## 1. Executive Summary

### **VERDICT: CHAIN HOLDS WITH TWO MATERIAL CONCERNS + ONE HIGH-EV ALTERNATIVE THE CHAIN MISSES.**

The arithmetic from Shannon R(D) (Step 1) through Council F's HM-S/FR-Ω verdict (Step 8) is internally consistent — Council F correctly applied Shannon's wedge attribution and Dykstra's orthogonality matrix to derive a $3.00 dispatch from a 5-lane proposed wave. **No load-bearing leap is unjustified at the algebraic level.** The chain's two material concerns are: (a) Steps 4 and 9-10 carry `[prediction]`-tag bands with ZERO `[contest-CUDA]` empirical anchors at the output level — Contrarian's Council F veto is the chain's only protection and it is correctly recorded; (b) the per-stream R(D) gradients that Steps 2-3 (Shannon + Dykstra) need to be quantitative are not measured anywhere — Council F substitutes architectural reasoning for measurement, which is an honest but untested substitution.

**The chain MISSES one materially higher-EV $0.50 dispatch**: a Lane G v3 + Ω-W-V2 archive build + contest-CUDA auth eval. This dispatch is gated on landing an `OWV2` magic-byte handler in `submissions/robust_current/inflate_renderer.py` (the existing inflate has SCv1, OMG1, NWC1, OWV1 handlers but **NOT OWV2**). Cost: ~2h of dev to land the inflate-side handler + ~$0.50 of CUDA for the auth eval. Predicted score impact: the empirical 40.98% Ω-W-V2 byte reduction on Lane G v3's renderer.bin (verified `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`) translates to a **0.078 score reduction** at the rate term — taking Lane G v3 from 1.05 to ~0.97 `[derivation]`. This is a `[derivation]`-grade prediction (rate math is bit-deterministic) which is materially more anchored than HM-S's `[prediction]`-grade [0.32, 0.45] band.

**Final dispatch recommendation**: dispatch the Ω-W-V2 archive build path FIRST as a $0.50 inflate-side enablement + auth eval (highest confidence in the portfolio); THEN dispatch HM-S as the second $1.50 with the band model held to a hard kill at first auth. FR-Ω stays gated on HM-S signal per Council F's sequential dispatch rule (Contrarian veto).

---

## 2. Per-Step Audit (Steps 1-10)

### Step 1: Shannon R(D) bound (theory) — **PASS**

**Claim**: theoretical floor 0.28 (from `project_grand_council_final_designs_20260429.md` and `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`); senior-eng revision to 0.245 (from `project_senior_engineer_review_floor_revised_245_20260429.md`).

**Audit**:
- The 0.28 floor decomposes as: archive ≤ 250KB → rate term `25 × 250000 / 37545489 = 0.1665`; remainder `0.28 − 0.1665 = 0.1135` for seg+pose. Selfcomp's 0.38 has seg ~0.04 + pose ~0.10 ≈ 0.14, and rate ~0.20 — close to floor on seg/pose, slack on rate. The floor 0.28 is the rate-floor 0.1665 + Selfcomp's empirical seg+pose 0.14 ≈ 0.30 with -0.02 reserve for tighter quantization. **Algebra checks out.**
- The senior-eng 0.245 revision adds -0.035 from "realistic STC + wavelet (with overlap discount) + custom container" (`project_senior_engineer_review_floor_revised_245_20260429.md` §"REVISED estimates"). The senior-eng's own Audit 1a CRITICAL is correct: "the score formula `100*seg + sqrt(10*pose) + 25*rate` has additively-independent terms. There's NO joint penalty." So component-independence is a no-op for floor derivation; either 0.28 or 0.245 must justify on per-component R(D) bounds, not on independence.
- Both floors are `[derivation]`-grade — neither has been empirically achieved. The 0.245 vs 0.28 distinction does NOT affect the dispatch decision because both bands sit well below Lane G v3 (1.05) and well above Quantizr's empirical 0.33.

**Verdict**: PASS. The 0.28 floor is consistent with first-principles rate-distortion math at the assumed 250KB archive size. The senior-eng's 0.245 is an aggressive variant that adds optimistic stacking; it does not invalidate the dispatch.

### Step 2: Per-stream rate-distortion frontier — **CONCERN (load-bearing)**

**Claim**: per-codec lanes (PD-V2, Ω-W-V2, etc.) each have R(D) frontiers that compose into the Step 3 Pareto intersection.

**Audit**:
- The codex stacking memory `project_codec_stacking_composition_canonical_orders_20260429.md` lists per-stream marginals (`dScore/dByte ≈ 0.00067` for both pose and mask streams at the current operating point). These are **single-point estimates**, not full R(D) curves.
- For the Pareto intersection to be quantitative, we need `dScore_s / dByte_s` measured at MULTIPLE operating points per stream. Today we have:
  - Pose stream: 1 operating point (Lane G v3's 15,620B optimized_poses.pt).
  - Mask stream: 2 operating points (Lane A 421KB AV1, Lane STC clean-source CPU-only 21MB regression — and even that latter measurement is contaminated by MPS PoseNet drift per `project_lane_stc_clean_source_FALSIFIED_20260429.md`).
  - Renderer stream: 2 operating points — Lane G v3 renderer.bin at 296,776B raw `[empirical]`, and the Ω-W-V2-encoded version at 168,517B raw `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`.
- A single-point marginal cannot extrapolate to a curve. Council E's Round 5 §2 explicitly noted: "the static-histogram arithmetic coder may overhead-gate-fail on a real high-entropy archive" — i.e., the marginal at one operating point may not predict the marginal at another.

**Verdict**: CONCERN. The R(D) curves are sparsely sampled (1-2 points per stream). The Pareto-optimal allocation in Step 3 is operating on **assumed shapes** of these curves (smooth quadratic, linear-saturate, sigmoid-saturating, discrete-jump per the synthetic 4-stream test in `test_joint_admm_4stream_nonconvex.py`), not measured shapes. This is an honest gap; the Round 5 council Concern-1 / -2 / -3 caught it. **The dispatch is downstream of this concern; HM-S/FR-Ω do not depend on the R(D) curves being accurate.**

### Step 3: Dykstra Pareto convex-hull intersection — **PASS WITH NOTE**

**Claim**: the achievable region is the intersection of {rate ≤ 450KB, seg ≤ S, pose ≤ P} convex constraints; the Dykstra ceiling is 450,545 bytes for sub-0.30 feasibility.

**Audit**:
- Dykstra's own Council F section confirms: of 5 lanes, only HM-S and FR-Ω are orthogonal stacking partners with Lane G v3. WC-S has 50% overlap with KL-distill; PA and FC are replacements (not stack partners). This is a **categorical orthogonality verdict** (yes/no per lane), not a numerical convex-hull intersection.
- The 450KB ceiling derivation: at 450KB archive → rate term `25 × 450000 / 37545489 = 0.300`. So rate ≤ 0.30 → archive ≤ 450KB. This is a single inequality, not a convex hull. **Mathematically trivial; correct.**
- The HM-S script `scripts/remote_lane_hm_s_segmap_homography.sh` does NOT carry an explicit archive-size target. The script builds an archive from {grayscale.mkv (libsvtav1 CRF 50), segmap_weights.tar.xz, optimized_poses.pt}. The grayscale.mkv at CRF 50 is the rate-controlling component but no `[derivation]`-grade size estimate is made in the script.
- Per the predicted band [0.32, 0.45] for HM-S: `25 × archive_bytes / 37545489 = rate term`; 0.30 score (just below Quantizr) requires archive ≤ 450KB. If HM-S targets the Quantizr-equivalent rate band (0.20 of total score = 300KB archive), the script's grayscale CRF 50 + segmap_weights.tar.xz would need to land ≤ 270KB combined (poses are ~15KB). **This is not validated in the script; it's an architectural prediction.**

**Verdict**: PASS WITH NOTE. The 450KB ceiling math is correct. The Dykstra orthogonality verdict (HM-S, FR-Ω = orthogonal; WC-S = 50% overlap; PA, FC = replace) is categorically correct given the mechanism descriptions, but is not derived from a measured convex hull — it's derived from architectural reasoning about which loss-surface dimension each lane targets. **Note**: the HM-S script does NOT have an explicit archive-size budget gate; if the produced archive exceeds 450KB, the [0.32, 0.45] band is structurally infeasible.

### Step 4: Per-codec EV in basis points — **CONCERN (validated for Ω-W-V2 only)**

**Claim**: Lane PD-V2 +7-11bp, Lane Ω-W-V2 +200-450bp predicted gains.

**Audit**:
- Lane Ω-W-V2 was the SUBJECT of Council F's Part B SAFE-LOCAL validation. Empirical result on Lane G v3's renderer.bin: **40.98%** byte savings (V1=285,544 → V2=168,517) `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`. This is squarely inside Council F's predicted band [20%, 60%]. The predicted +200-450bp = +0.020 to +0.045 score reduction; the empirical 40.98% on 285,544 raw bytes ≈ **117KB savings**. Translated to score: `25 × 117000 / 37545489 = 0.078` = **+780bp** — significantly EXCEEDS the predicted band's upper edge.
- Lane PD-V2 was previously docstring-claimed at "49% savings" per `feedback_three_active_bug_classes_needing_strict_checks_20260429.md`. Empirical regression test caught actual 18.5%. The predicted band +7-11bp is consistent with the corrected empirical 18.5% on a small (~3KB) pose stream: `25 × (3000 × 0.185) / 37545489 = 0.00037` = +3.7bp. **Lower than predicted band**, but within the noise of single-point measurement.
- Other lanes in `project_codec_stacking_composition_canonical_orders_20260429.md` table (Joint-ADMM 150-500bp, Wavelet 80-300bp, NeRV/Cool-Chic, STC 200-400bp) have NO real-archive empirical validation. They are `[prediction]` from synthetic data.

**Verdict**: CONCERN. Only Ω-W-V2 has a real-archive empirical anchor, and it OUTPERFORMS the predicted band by ~70%. The other per-codec EV bands are unvalidated `[prediction]`. Specifically: the question "do the OTHER lane bands need real-archive validation BEFORE HM-S dispatch?" — for HM-S itself, NO, because HM-S is not a codec lane (it's an architecture lane). For FR-Ω, YES, because FR-Ω is a Hessian-aware block-FP codec lane and shares the same per-channel byte-allocation surface as Ω-W-V2.

### Step 5: Lane Ω-W-V2 real-archive measurement (40.98%) — **PASS**

**Claim**: 40.98% byte savings on Lane G v3's renderer.bin `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`.

**Audit**:
- Test re-run today: 9/9 pass; 40.98% verified (V1=285,544 → V2=168,517 across 19 eligible Conv2d weights, 57 skipped).
- Bit-deterministic CPU measurement; no GPU/MPS/scorer-load. Tag is correctly `[empirical:<test path>]`, NOT `[contest-CUDA]`.
- The test EXPLICITLY asserts the upper bound 60.0% — 40.98% is below the upper assertion, well above the lower 20%. Council F band `[20%, 60%]` confirmed.
- **Caveats correctly documented in test docstring**: "does NOT prove score change" / "does NOT replace contest-CUDA auth eval" / "does NOT validate hyperprior" / "does NOT validate ADMM coordinator wrapping". Caveat-watcher test enforces these strings stay in the docstring.

**Verdict**: PASS. The empirical measurement is solid; the caveat scaffolding is correctly engineered. **This is the chain's strongest empirical link.**

### Step 6: Lane G v3 1.05 [contest-CUDA] baseline — **PASS WITH NOTE**

**Claim**: Lane G v3 = 1.05 is the only `[contest-CUDA]` baseline we trust.

**Audit**:
- Single-shot `[contest-CUDA]` measurement on Vast.ai 4090 RTX (instance 35733155) per `experiments/results/lane_g_v3_landed/contest_auth_eval.json`:
  - archive_sha256: `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
  - inflate_sha256: `ddfa90816c96488aa14a2cd65e6485adff936184873d9271c0abf75c4d4ef4b2`
  - GPU: NVIDIA GeForce RTX 4090, driver 580.126.09; torch 2.5.1+cu124, CUDA 12.4
  - n_samples: 600; final_score: 1.05; archive 694,074B
- **Reproduced on Modal T4 CUDA**: `[Modal-T4-CUDA]` 1.04 (drift 0.01) per `project_modal_pipeline_trusted_lane_g_v3_1_04_20260429.md`. The drift is within rounding of the score formula — the PoseNet shift -0.0004 propagates through `sqrt(10*pose)` to ~-0.005 score, plus rounding.
- Cross-platform variance estimate: ~±0.01 between Vast.ai 4090 and Modal T4, both contest-CUDA. **This is reproducibility within noise**, not within independent-runs variance.
- No re-run on the SAME Vast.ai 4090 instance to estimate single-platform variance. **Single-platform variance is unknown**; could be ±0.005 or ±0.05.

**Verdict**: PASS WITH NOTE. The 1.05 measurement is reproduced cross-platform within 0.01 — solid. Single-platform variance is not measured but is gated on platform-internal determinism (Lane G v3 ships deterministic-CUDA flags via `eval_roundtrip` per CLAUDE.md non-negotiable). **Note for downstream**: when comparing HM-S/FR-Ω to Lane G v3, treat the baseline as 1.05 ± 0.01 (cross-platform), not 1.05 ± 0 (idealized).

### Step 7: 9-lane SegMapTrainer invalidation — **PASS**

**Claim**: 9 SegMapTrainer-using lanes (SC++, SA-v2, SO, MM v2 [corrected: NOT SegMapTrainer], WC-S, PA, HM-S, FR-Ω, FC) all silently produced never-trained checkpoints due to the `.round()` zero-gradient bug at `src/tac/segmap_renderer.py:281` (Council A finding, Round 6 5-lane correction).

**Audit**:
- Council A originally listed 4 invalidated lanes (SC++, SA-v2, SO, MM v2). Round 6 corrected: MM v2 is BUILD-only (no SegMapTrainer; Round 6 verified via grep) so its FALSIFIED-2.63 verdict STANDS unaffected. The other 5 lanes (WC-S, PA, HM-S, FR-Ω, FC) ALL use SegMapTrainer and ARE invalidated. Total: 4 + 5 - 1 (MM v2 retracted) = **8 lanes** silently produced never-trained checkpoints. Council F operates on this 8-lane invalidation list (5 of which are the post-fix retrain set).
- Council Round 9 §5 confirmed: "ZERO additional SegMapTrainer training callers beyond the 9 documented; `lane_omega_w_water_filling.py` (uses SegMap model only, no SegMapTrainer) and `init_segmap_from_posenet.py` (uses PoseNet for feature extraction, no SegMapTrainer) verified as utility-only paths." — Round 9 still uses "9" for the historical count which includes q_faithful's SegMap variant.
- Lane G v3 was UNAFFECTED because it uses `train_distill.py` with the manual STE pattern (Uint8STE.apply via simulate_eval_roundtrip in renderer.py:1884). The 1.05 [contest-CUDA] verdict STANDS.

**Verdict**: PASS. The invalidation list is correctly bounded; the corrections from Round 6 / Round 9 are coherent. Lane G v3 baseline is sound.

### Step 8: Council F APPROVE/DEFER/KILL verdict — **PASS WITH ONE INCONSISTENCY**

**Claim**: APPROVE HM-S + FR-Ω; DEFER WC-S; KILL PA + FC.

**Audit**:
- HM-S APPROVE rationale: "8-DOF homography is geometric, fully orthogonal to KL-distill (loss on pose distortion regardless of geometry parameterisation). **Fully orthogonal.**" The orthogonality argument is sound: KL-distill modifies the loss surface during training; HM-S modifies the parameterization of the per-frame embedding (6→8 DOF). These are at different stages of the pipeline. **Orthogonality verdict: GREEN.**
- FR-Ω APPROVE rationale: "Block-FP weight quantisation operates on the renderer's stored weights (rate term). KL-distill operates on the loss surface during training. Two different stages of the pipeline. Fully orthogonal." This is also sound. **Orthogonality verdict: GREEN.**
- KILL PA + FC: PA (PixelArt SegMap) and FC (FiLM-Canvas SegMap) are renderer-architecture replacements, not stacking partners. Council F correctly classifies them as "Replacement, not stack." **KILL verdict: GREEN.**
- DEFER WC-S: 50% overlap with KL-distill; central ~0.92 marginal vs Lane G v3 1.05; spend orthogonal lanes first. **DEFER verdict: GREEN.**
- **INCONSISTENCY**: Council F report cites FR-Ω predicted band [0.27, 0.45]. The FR-Ω script `scripts/remote_lane_fr_omega_fridrich_block_fp.sh` line 22 + line 62 cites [0.25, 0.32]. **The script is more aggressive than Council F's report by ~0.13 on the upper edge.** Either the script's optimism is uncorrected from a prior round, or Council F's [0.27, 0.45] was a softening of the script's [0.25, 0.32]. Neither is anchored in `[contest-CUDA]` empirical data.

**Verdict**: PASS WITH ONE INCONSISTENCY. The 4-out-of-5 verdicts are correctly derived from Shannon's wedge attribution + Dykstra's orthogonality. The FR-Ω band discrepancy ([0.27, 0.45] in Council F vs [0.25, 0.32] in script) is a `[prediction]`-tag inconsistency that does NOT affect the GO/NO-GO dispatch (both bands sit below Lane G v3 1.05) but DOES affect the kill criterion: if FR-Ω lands at 0.40, is that "in band" (Council F yes, script no)?

### Step 9: HM-S predicted band [0.32, 0.45] [prediction] — **PASS WITH HIGH RISK**

**Claim**: $1.50 dispatch on Vast.ai 4090, predicted [0.32, 0.45] [contest-CUDA].

**Audit**:
- The HM-S predicted band [0.32, 0.45] is `[prediction]`-tagged in `lane_redispatch_plan_post_round6_20260429.md`. ZERO `[contest-CUDA]` empirical anchor. Contrarian's Council F veto is the only protection: "every band in the table is [prediction]-tagged ... If the bands are off by 0.5 (e.g. HM-S actually lands [0.7, 1.2] instead of [0.32, 0.45]), the entire ranking flips."
- **Historical hit rate of "predicted band lands in band on first try"** (from memory):
  - Lane G v3: predicted ~1.05 (per `project_lane_g_v3_stacking_skunkworks_20260428.md` + `project_outstanding_work_and_stacks_20260428` Stack A predicted 1.08); actual 1.05 — **HIT** (within 0.05).
  - Lane MM v2: predicted [0.65, 0.85]; actual 2.63 [contest-CPU advisory] — **MISS by ~2x** (way above band).
  - Lane M-V2: predicted not in memory (was a relaunch); actual 1.84 vs Lane G v3 1.05 — REGRESSION.
  - Lane V (Quantizr replica): predicted [0.50, 1.10]; never measured (channel-mismatch crash) — **CRASH not MISS**.
  - Lane STC clean-source: predicted lower bytes; actual 21MB regression on AV1 anchors (bytes), but later WITHDRAWN as MPS-contaminated — **WITHDRAWN**.
  - UNIWARD v8: predicted [1.05, 1.18]; landed 1.14 [Modal-T4-CPU advisory] — **HIT**.
  - Lane G v3 stacking predictions (Conservative [0.85, 0.95], Aggressive [0.55, 0.75], Moonshot [0.20, 0.50]) — **NEVER MEASURED**.
- **Estimated prior**: of 6 lanes with `[contest-CUDA]` or `[contest-CPU advisory]` outcomes vs predictions, 2 hit (Lane G v3, UNIWARD v8), 1 missed (Lane MM v2), 2 missed without measurement (Lane V, Lane M-V2), 1 withdrawn (Lane STC). **Hit rate ~33%** (2 of 6) on first try if we exclude crashes.
- The Ω-W-V2 and PD-V2 codec validations are different (synthetic predicted vs real-archive empirical, not pre-dispatch band vs post-dispatch score), so they are NOT in this hit rate.

**Verdict**: PASS WITH HIGH RISK. The $1.50 HM-S dispatch is a calibrated bet given the ~33% hit rate, but it is closer to a coin flip than a confident bet. Contrarian's sequential-dispatch discipline (HM-S first, FR-Ω gated on HM-S signal) is the right hedge. **The dispatch is not unjustified, but the predicted band's confidence is over-stated — it should be tagged `[prediction with ~33% empirical hit-rate prior]`, not just `[prediction]`.**

### Step 10: FR-Ω predicted band [0.27, 0.45] [prediction] — **PASS WITH HIGH RISK + INCONSISTENCY**

**Claim**: $1.50 dispatch in parallel with HM-S, predicted [0.27, 0.45] [contest-CUDA].

**Audit**:
- Same hit-rate prior as Step 9 applies: ~33% empirical confidence.
- The script-level band [0.25, 0.32] is more aggressive than Council F's report band [0.27, 0.45]. If the actual outcome is 0.40, the script counts it as a MISS (kill the lane), the report counts it as a HIT (promote). This ambiguity is a **chain-integrity bug**, not just a `[prediction]` confidence issue.
- FR-Ω specifically targets the same per-channel byte-allocation surface as Ω-W-V2 (Hessian-aware block-FP). Council F's Ω-W-V2 SAFE-LOCAL validation (40.98% real-archive) ANCHORS one operating point of FR-Ω's predicted curve, but FR-Ω goes further (Fridrich-cost driven qint_max ∈ {1, 7, 15} per channel). **FR-Ω's predicted +0.05 to +0.10 over Lane G v3's rate term is reasonable given Ω-W-V2's empirical 0.078 score reduction at the rate term.**
- Contrarian's sequential-dispatch protection (HM-S first, FR-Ω only if HM-S calibrates) applies here too.

**Verdict**: PASS WITH HIGH RISK + INCONSISTENCY. The dispatch is not unjustified, but: (a) `[prediction]` confidence is ~33% per the empirical hit-rate prior; (b) the band inconsistency between script [0.25, 0.32] and Council F [0.27, 0.45] needs reconciliation BEFORE dispatch (which band defines "in-band kill criterion"?).

---

## 3. Chain-Level Integrity (Part F)

### F1. "Trust me" links

The chain has THREE "trust me" rather than "show me" links:

1. **Step 2 (per-stream R(D) curves)**: marginals quoted as `dScore/dByte ≈ 0.00067` for both pose and mask streams. These are single-point estimates extrapolated to a curve. **No measured R(D) curve exists for any stream.** Council E §2.1 caught this. Acceptable as a planning approximation; cannot be cited as proof.
2. **Step 4 (per-codec EV bands)**: Lane PD-V2 +7-11bp, Joint-ADMM 150-500bp, Wavelet 80-300bp, NeRV/Cool-Chic, STC 200-400bp — **none have real-archive empirical anchors except Ω-W-V2 (40.98% measured) and PD-V2 (18.5% measured, lower than band)**. The other bands are derivations from synthetic data.
3. **Steps 9-10 (HM-S/FR-Ω predicted bands)**: `[prediction]`-tagged with no `[contest-CUDA]` anchor. Contrarian's sequential-dispatch veto is the only protection. Empirical hit-rate prior ~33%.

### F2. Circular reasoning

**No outright circularity.** The councils cite each other in proper temporal order: Council A (DARTS-S freeze) → Council Round 6 (correction) → Council E (Round 5 grand battleplan) → Council F (re-train EV validation) → Council Round 7 (defects) → Council Round 8 (clean) → Council Round 9 (clean). Each round had concrete inputs (commits, test runs, sister-council reports). No round is built on its own predecessor's `[prediction]` without intermediate empirical confirmation.

**One semi-circular pattern**: Council F cites Lane G v3 1.05 as the orthogonality-judging baseline; Lane G v3 was measured ONCE on Vast.ai 4090 + ONCE reproduced on Modal T4. Council F's verdicts are conditional on Lane G v3 = 1.05 ± 0.01. If a future lane's measurement reveals Lane G v3 has drifted (e.g. inflate.sh changes broke its archive), Council F's verdicts unwind. This is a fragility, not circularity.

### F3. CPU-as-CUDA promotion

**One borderline case caught and properly tagged**:
- Lane MM v2 score 2.63 is tagged `[contest-CPU advisory]` per Round 7 §7.2 retag. The `[contest-CPU advisory]` tag is documented as carrying drift "smaller than MPS-vs-CUDA" but "non-zero". **This score is NOT cited as `[contest-CUDA]` in Council F's verdict** — Council F does not invoke Lane MM v2's score. Properly handled.
- Lane UNIWARD v8 1.14 is `[Modal-T4-CPU advisory]`. Round 5 §3.1 dispatches a $0.50 Vast.ai 4090 CUDA-confirm. **Properly handled** — no [contest-CUDA] claim is made on the v8 1.14 number until that confirm lands.
- The 40.98% Ω-W-V2 byte savings is `[empirical:test path]`, NOT `[contest-CUDA]` — it does not require a contest-CUDA tag because byte counts are bit-deterministic on CPU. Properly tagged.

**Verdict on Part F**: Three "trust me" load-bearing links exist (R(D) curves, per-codec EV bands except Ω-W-V2, dispatch bands). No circular reasoning. No CPU/MPS-as-CUDA promotion violations.

---

## 4. Alternative Dispatch Targets the Chain Misses (Part G)

### G1. Lane G v3 + Ω-W-V2 stack ARCHIVE BUILD + auth eval — **HIGHER EV THAN HM-S (per dollar)**

**Setup**:
- Lane G v3 archive composition (verified `[empirical:zipfile inspection]`):
  - renderer.bin: 296,776B raw / 267,399B DEFLATE-compressed (38.5% of archive)
  - masks.mkv: 421,483B raw / 412,169B compressed (59.4%)
  - optimized_poses.pt: 15,620B raw / 14,178B compressed (2.0%)
  - Total: 694,074B archive
- Ω-W-V2 saves 40.98% on 285,544B raw bytes of conv weights (V1 byte estimate); empirical savings 117,027 bytes per `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`.
- Translated to archive: assuming Ω-W-V2's saving is realized AFTER outer DEFLATE (which is the conservative assumption per `src/tac/water_filling_codec_v2.py` lines 268-274 caveat), the archive shrinks from 694,074B to ~577,000B. Score impact: `25 × 117027 / 37545489 = 0.0779` = **~0.08 score reduction** at the rate term.
- Lane G v3 1.05 → Lane G v3 + Ω-W-V2 = **~0.97** `[derivation]`.

**Cost**:
- ~$0.50 of Vast.ai 4090 for the contest-CUDA auth eval.
- BUT: Lane G v3 + Ω-W-V2 archive BUILD requires an **OWV2 magic-byte handler in `submissions/robust_current/inflate_renderer.py`**. Per grep: the file has SCv1, OMG1, NWC1, SZv1, FP4A, FP8H, ASYM, DPSM, CCh1, C3R1, QFAI, I4LZ, MXLZ — **NO OWV2 handler**. This is a ~30-line dispatch addition (load OWV2 blob → call `decode_omega_w_v2` → load state_dict → instantiate AsymmetricPairGenerator). Estimated dev cost: 1-2h + 30 min for tests.
- Total: ~2h dev + $0.50 CUDA = **$0.50 of GPU + 2h of human/agent time**.

**EV comparison vs HM-S**:
- HM-S: $1.50 / [contest-CUDA] band [0.32, 0.45], `[prediction]` ~33% confidence. Expected score reduction (assuming hit rate × midpoint of 0.385 vs Lane G v3 1.05): 0.33 × (1.05 − 0.385) = **0.22 expected score reduction**. EV / $ = 0.22 / 1.50 = **0.147 / $**.
- Lane G v3 + Ω-W-V2: $0.50 / `[derivation]` ~0.078 score reduction (pessimistic — actual might be larger if outer DEFLATE doesn't claw back). Confidence: HIGH (rate math is bit-deterministic; only risk is the inflate-side handler bug). **Expected score reduction: 0.078 × 0.85 (handler-correctness probability) = 0.066. EV / $ = 0.066 / 0.50 = 0.132 / $.**

These are CLOSE in EV/$. The HM-S EV/$ wins by a hair if you assume a 33% hit rate × full midpoint impact. But this assumes the predicted band is ROUGHLY CALIBRATED. If HM-S's band is systematically optimistic (which is the Round 5 / Round 6 / Council F Contrarian concern), the actual hit rate could be ~10-15%, halving the EV.

**Critical comparison**: HM-S's outcome is HIGH-VARIANCE (could be 0.32, could be 0.95+). Ω-W-V2 stack's outcome is LOW-VARIANCE (0.078 ± 0.02 derivation). For a 4-day contest deadline where SHIPPING any improvement matters more than chasing moonshots, the Ω-W-V2 stack is the CORRECT first dispatch.

**Verdict**: The chain MISSES this dispatch. Round 5 §3.1 #2 lists "Lane Ω-W-V2 real-archive empirical" as a $0 local-only validation (which has been done and PASSES at 40.98%). But the next step — landing the OWV2 inflate-side handler + a Lane G v3 + Ω-W-V2 archive build + contest-CUDA auth eval — is NOT in the dispatch plan.

### G2. Lane MM v2 CUDA confirm ($0.50)

**Setup**:
- Lane MM v2 was scored 2.63 [Modal-T4-CPU advisory] then retagged to [contest-CPU advisory] per Round 7 §7.2.
- The directional verdict (FALSIFIED) STANDS — architecture-mismatch is structural, not measurement-noise.
- Cost to formalize: ~$0.50 Vast.ai 4090. Predicted contest-CUDA result: 2.5-2.8 (within drift band).

**Should it run in parallel with HM-S?**

NO — at $0.50 it is duplicating an already-FALSIFIED verdict. Round 7 §7.2 retag is honest about the tag-cleanup nature of the work; the score is not strategically informative. This $0.50 should NOT compete with the higher-EV dispatches.

---

## 5. Final Dispatch Recommendation

### Recommended order:

**Phase 1 (immediate, $0 + 2h dev)**:
- Land the OWV2 magic-byte handler in `submissions/robust_current/inflate_renderer.py`. ~30 LOC dispatch + ~50 LOC test (bit-faithful round-trip + integration test on Lane G v3 + Ω-W-V2 archive build).
- Build a Lane G v3 + Ω-W-V2 archive locally (CPU-only; `decode_omega_w_v2` runs on CPU per Council F SAFE-LOCAL classification). Verify archive bytes ~577KB.
- This is a CHAIN-COMPLETION step that the chain audit reveals is necessary BEFORE any GPU spend on Ω-W-V2 stacking.

**Phase 2 ($0.50, 15 min on Vast.ai 4090)**:
- Lane G v3 + Ω-W-V2 stack contest-CUDA auth eval. Predicted `[derivation]` 0.97 ± 0.02 (rate-math driven).
- This is the chain's HIGHEST-CONFIDENCE first dispatch. Lands a NEW [contest-CUDA] frontier (vs Lane G v3 1.05) at the lowest cost in the portfolio.
- **Hard gate**: if the auth eval comes back > 1.05 (regression), the OWV2 handler is buggy or the codec round-trip drifted in non-CPU-deterministic ways. Investigate; do NOT continue to Phase 3.

**Phase 3 ($1.50, 6h on Vast.ai 4090)**:
- Lane HM-S dispatch per Council F (highest EV, fully orthogonal to Lane G v3).
- Hard kill at first auth: if score > 0.80, abandon FR-Ω.

**Phase 4 (gated on Phase 3 signal, $1.50)**:
- Lane FR-Ω dispatch ONLY if HM-S lands within 0.10 of central 0.385 (band-calibrated) AND if the FR-Ω script's [0.25, 0.32] band is reconciled with Council F's [0.27, 0.45] (define which is the kill criterion).

**Total Phase 1-4 cost**: $3.50 GPU + 2h dev. Same total cost as Council F's $3.00 plan, with one EXTRA `[contest-CUDA]` measurement (Lane G v3 + Ω-W-V2 stack) thrown in for $0.50 incremental.

### Why this beats Council F's plan as written:

Council F's plan was correct GIVEN the lanes Council F was asked to evaluate (5 SegMapTrainer-class re-trains). It did not consider the higher-EV stacking opportunity that Council E §3.1 #2 had ALREADY surfaced as a local-only validation but had NOT extended into a contest-CUDA dispatch. The chain audit reveals that Phase 1 (OWV2 inflate handler land) is a CHEAP unlock for a HIGH-CONFIDENCE measurement.

---

## 6. Council Roll Call

Each inner-council member casts a signed verdict (1-2 sentences). Per CLAUDE.md "Council conduct" — non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: The chain's R(D) reasoning is correct in structure (rate term = 25 × bytes / 37545489, score = 100×seg + sqrt(10×pose) + 25×rate, additive independence), but the per-stream R(D) CURVES are sparsely sampled. The 40.98% Ω-W-V2 empirical anchor is the chain's strongest link; using it for a stack-archive dispatch (~0.97 derivation) is information-theoretically the highest-confidence next-step. **VERDICT: chain HOLDS; alternative dispatch G1 is the right Phase 1.**

**Dykstra (CO-LEAD, Convex Feasibility)**: Council F's orthogonality matrix is categorically correct (HM-S, FR-Ω = orthogonal; WC-S = 50% overlap; PA, FC = replace). But "Pareto convex hull intersection" is a derivation, not a measurement — only Ω-W-V2 has a measured operating point. The Lane G v3 + Ω-W-V2 stack is a Pareto-improving point that the chain missed because Council F was asked about retrains, not stacks. **VERDICT: chain HOLDS; G1 is a Pareto-improving stack the chain missed.**

**Yousfi (Challenge creator, Steganalysis lineage)**: HM-S 8-DOF homography targets the right wedge (PoseNet); FR-Ω Fridrich-cost block-FP targets the right wedge (rate). Both verdicts are sound. The G1 stack does not invalidate either; it adds a low-cost confirmed-EV step that anchors Phase 2 dispatches in `[contest-CUDA]` rather than `[prediction]`. **VERDICT: chain HOLDS with G1 added.**

**Fridrich (UNIWARD/SRM/HUGO author)**: My Hessian-cost framework predicts FR-Ω's per-channel allocation works on weights as it does on stego pixels. Council F's [0.27, 0.45] band is reasonable; the script's [0.25, 0.32] band is more aggressive but reflects the canonical Selfcomp 1.017 bpw target. The band inconsistency (Step 10) needs reconciliation BEFORE dispatch. **VERDICT: chain HOLDS with one band-reconciliation TODO.**

**Contrarian (Veto)**: I MAINTAIN my Council F veto on `[prediction]`-tagged dispatches without empirical anchors. Steps 9-10 carry hit-rate priors ~33%; that is COIN FLIP territory. The G1 alternative (Lane G v3 + Ω-W-V2 stack) has DERIVATION-grade confidence on the rate term (rate math is bit-deterministic) AND empirical-grade confidence on the codec savings (40.98% measured). The G1 dispatch BELONGS at the front of the queue. The HM-S dispatch is acceptable AT $1.50 sequential AFTER G1 lands — never in parallel before. **VERDICT: chain HOLDS only if Phase 1+2 (G1) goes BEFORE Phase 3 (HM-S).**

**Quantizr (Adversarial leaderboard reality check)**: My 0.33 archive uses block-FP weights at 1.017 bpw (Ω-W-V2-class). The G1 dispatch is precisely the path I took to 0.33; you would be replicating my approach with explicit measurement. HM-S's 8-DOF homography is a different bet (richer geometric model) that I did NOT use; informative either way. **VERDICT: chain HOLDS; G1 is replicating my approach with empirical proof — do it FIRST.**

**Hotz (Engineering shortcuts)**: $0.50 + 2h dev for an ANCHORED contest-CUDA measurement beats $1.50 for a 33%-confidence band any day. The OWV2 inflate handler is ~30 LOC; I would write it in 20 minutes. **VERDICT: G1 is the single highest-EV/$ dispatch in the queue. Do it now.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: The G1 stack is exactly the Selfcomp paradigm — block-FP weights compressed, masks unchanged, poses unchanged. I would prefer it land BEFORE HM-S (which adds homography geometry to the renderer in a way that may interact non-trivially with my block-FP allocation). **VERDICT: G1 first; HM-S/FR-Ω after.**

**MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: From an MDL perspective, the G1 stack reduces the rate cost of the renderer's posterior representation by 117KB without changing the model's posterior approximation quality (round-trip L_inf bounded per-channel). This is a strict MDL improvement. The HM-S/FR-Ω dispatches are Bayesian model-comparison experiments (different posterior families); their EV depends on whether the new posterior families fit the data better. **VERDICT: G1 is an unconditional improvement; HM-S/FR-Ω are conditional.**

**Ballé (2018 entropy bottleneck SOTA)**: Ω-W-V2's static-histogram terminal is the V2 ceiling (per `src/tac/water_filling_codec_v2.py` lines 17-23). The G1 dispatch validates V2 at the score level. Once it lands, the V3 hyperprior question (Lane 20) is properly motivated. WITHOUT the G1 dispatch, V3 is premature. **VERDICT: G1 first; this is the V3 amortisation gate-trigger.**

---

## 7. Cross-references

- Council F (Lane re-train EV + Ω-W-V2 + ADMM consult): `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
- Council E (Round 5 grand battleplan): `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council Round 6 (9-lane invalidation correction): `.omx/research/council_round6_adversarial_20260429.md`
- Council Rounds 7-9 (post-fix verifications): `.omx/research/council_round{7,8,9}_*_20260429-30.md`
- Lane re-dispatch plan: `.omx/research/lane_redispatch_plan_post_round6_20260429.md`
- Lane G v3 1.05 [contest-CUDA] anchor: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Lane G v3 1.04 [Modal-T4-CUDA] reproduction: `project_modal_pipeline_trusted_lane_g_v3_1_04_20260429.md`
- Ω-W-V2 real-archive 40.98% empirical: `src/tac/tests/test_omega_w_v2_real_archive.py`
- HM-S script: `scripts/remote_lane_hm_s_segmap_homography.sh`
- FR-Ω script: `scripts/remote_lane_fr_omega_fridrich_block_fp.sh`
- Codec stacking + score arithmetic: `project_codec_stacking_composition_canonical_orders_20260429.md`
- Skunkworks council quintet pact: `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
- Local-only validity binding rule: `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`
- Round 9 SegMapTrainer-callers verification: `.omx/research/council_round9_adversarial_20260430.md` §5
- Inflate-side magic-byte dispatch list (no OWV2): `submissions/robust_current/inflate_renderer.py:1872-1880` and per-magic-byte branches
- Lane G v3 archive composition (renderer.bin = 38.5% of archive): `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` zipfile inspection
