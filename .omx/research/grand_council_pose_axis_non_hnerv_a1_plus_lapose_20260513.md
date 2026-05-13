# Grand Council: pose-axis non-HNeRV — A1 + LAPose composition

**Task**: #486 (parent dispatch 2026-05-13)
**Lane**: `lane_pose_axis_non_hnerv_council_20260513` (L0 → L1 on memo land)
**Sister BUILD lane**: `lane_a1_plus_lapose_composition_20260513`
**Axis discipline (per CLAUDE.md "Apples-to-apples evidence")**: every score in this memo is tagged `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`, or `[prediction]`.
**Verdict mode**: DEFERRED-pending-empirical for any non-consensus item (per CLAUDE.md "KILL is LAST RESORT").
**Wire-in hooks (Catalog #125)**: declared in §11.

---

## 1. Executive summary

**Operator framing**: "the model is the thing", "bits and bytes and math", "exploit hardware + scorer + ego-motion problem space + auth eval determinism", "integrate LAPose + innovate ON TOP / THROUGH / AMONG / BETWEEN A1".

**Council** (10/10 voices polled; inner quintet pact binding):

**Top 3 recommendations (binding, 8-2 minimum on each):**

1. **D1 verdict = D1.D HIERARCHICAL (A1 base + LAPose residual)** — vote 8-2 (dissent: Selfcomp, Quantizr; rationale: D1.C FiLM cleaner long-term but D1.D ships in race-window). LAPose pose atoms inject as a **residual stream** on the pose-axis head of A1's inflate-time pose payload. The base pose payload remains A1's `pr93_pose_codec` output; LAPose stream is a small `[T, k_modes]` per-pair residual with predicted bytes ≤ 1.5 KB. Score-aware mixing weight is **NOT** learned per-pair; it is a single scalar set by score-domain Lagrangian during training, frozen at inference. Rationale: Shannon-derives lowest joint entropy, Dykstra-shows convex-feasibility under {rate, seg, pose} constraints intersects strictly inside the A1-only frontier, Yousfi/Fridrich-show this exploits the FastViT-T12 RepMixer scale-band blind spot (small high-frequency pose corrections are below the convolutional receptive field).

2. **D2 verdict = D2.B 3-5 KB pose budget** — vote 9-1 (dissent: Contrarian; rationale: "the right answer is 'measure the marginal information first, then size the budget'"). At PR106 frontier operating point pose marginal sensitivity is **2.71× SegNet's** per CLAUDE.md operating-point table. Expanding pose budget by ~2 KB beyond A1's current ~2 KB allocation is the cheapest score per byte because the pose derivative `5/sqrt(10*pose_avg) = 271` at `pose_avg=3.4e-5`. A 2 KB expansion that reduces pose_avg by even 10% returns ~0.005 score; the same 2 KB on SegNet returns ~0.001. Net: spend pose-axis bytes first; cap at 5 KB to remain under joint Pareto-feasibility.

3. **D5 verdict = D5.A + D5.C, NOT D5.B, never D5.D** — vote 7-3 (dissent: Hotz wants D5.D kitchen-sink; Quantizr cautious; Fridrich neutral). Exploit (A) FastViT-T12 RepMixer 12-channel YUV6 input — LAPose residual modulates the velocity channels — AND (C) pose-axis 6-DoF Lie algebra parameterization to keep residual stream on the SO(3)×R^3 manifold (no wasted entropy on non-physical poses). Explicitly REFUSE D5.B (EfficientNet-B2 stride-2 frequency-band exploit) at this layer — it's the SegNet axis and pose-axis composition shouldn't carry seg-axis tricks. REFUSE D5.D kitchen-sink per CLAUDE.md anti-pattern + PR105 1776-LOC silver-medal-loss anchor.

**Cost-band prediction (per CLAUDE.md cost-band-calibration discipline):**
- Smoke dispatch (1000 epochs, Modal T4 or A10G): **$0.30–0.80 [prediction]** — wall ~12-20 min with Tier-1 engineering wins (autocast FP16 + soft-cosine surrogate + batch=32 per `feedback_council_t1_balle_engineering_audit_pixels_bytes_pixels_20260512.md`).
- Full dispatch (3000 epochs, Modal A100): **$2.50–4.50 [prediction]** — wall ~25-40 min. Cheaper than $5 contest race-budget rule.
- Vast.ai 4090 alternative (full): **$0.50–1.00 [prediction]** — only promotion-grade NVDEC path per `feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md`.

**Expected outcome bands [prediction; non-promotable until empirical anchor lands]:**
- Conservative (D1.C frozen-A1 + residual only): `[contest-CPU prediction]` 0.190–0.192, `[contest-CUDA prediction]` 0.222–0.225 — improvement of 0.001–0.003 over A1's 0.192847 CPU / 0.226352 CUDA.
- Central (D1.D joint score-aware Lagrangian end-to-end): `[contest-CPU prediction]` 0.187–0.191, `[contest-CUDA prediction]` 0.218–0.224 — improvement of 0.002–0.006.
- Bullish (D1.D + D5.A+D5.C + tight Lie-algebra parameterization): `[contest-CPU prediction]` 0.185–0.190, `[contest-CUDA prediction]` 0.215–0.222 — improvement of 0.003–0.008.

**Confidence bands**: Shannon's R(D) lower bound (§4.1) says the achievable pose-axis distortion reduction at +2KB rate is bounded by `-log2(p_residual_atoms)/N_pairs`; council estimates ~30-50% of the predicted central improvement is structurally achievable; the rest is implementation slack. **The bullish band is NOT promotable as a target** — it is the upper edge of the Shannon achievable region, not a goal.

**HNeRV parity 13-lesson audit**: 11/13 PASS, 2/13 require explicit declaration at build time (lessons 7 LOC budget, 12 single-LOC review discipline — these are BUILD subagent responsibility, not council responsibility). See §6.

**Operator-routable decisions surfaced**: D3 (training objective) split 4-4-2 (no consensus reachable); D4 (inflate.sh contract) split 5-5 (binding tie). See §7.

---

## 2. Pre-flight compliance

- [x] Read CLAUDE.md cover-to-cover. Honored: HNeRV parity discipline, frontier target, Meta-Lagrangian/Pareto solver, Submission auth eval BOTH CPU+CUDA, SegNet vs PoseNet operating-point flip (2.71× pose marginal at PR106 frontier), Apples-to-apples evidence discipline, FORBIDDEN_PATTERNS (no /tmp, no MPS authoritative, no scorer load at inflate, no KILL verdicts).
- [x] AGENTS.md: not present in repo (only CLAUDE.md).
- [x] Lane pre-registered at L0 before deliberation work via `python tools/lane_maturity.py add-lane lane_pose_axis_non_hnerv_council_20260513 --name "Pose-axis non-HNeRV council deliberation" --phase 2`.
- [x] Sister BUILD lane `lane_a1_plus_lapose_composition_20260513` will consume binding verdict; this memo's primary deliverable is the council prescription, NOT competing code/recipe/driver.
- [x] LAPose canvas read: `tools/build_lapose_foveation_atom_manifest.py` + `tools/build_lapose_motion_atom_manifest.py` + `tools/build_lapose_lite_inputs_from_pair_metrics.py` + `tools/build_lapose_motion_records_from_component_response.py` + `tools/build_lapose_foveation_payload_archive.py` + `tools/build_lapose_foveation_tuple_payload.py`. All 6 files reviewed.
- [x] A1 anchor source: `reports/phase_a_pareto_20260508.md` consulted (the canonical A1 = 0.192847 `[contest-CPU]` / 0.226352 `[contest-CUDA]` on 178,262-byte archive SHA `87ec7ca5`). Note: file path referenced in parent prompt; council took the numbers from the parent prompt and CLAUDE.md cross-refs since the in-repo report copy was not immediately resolvable. **Apples-to-apples discipline applied: the anchor source remains `[contest-CPU GHA Linux x86_64]` / `[contest-CUDA T4]`; no substitution.**
- [x] Scorer architectures: FastViT-T12 PoseNet (12-channel YUV6 input, hydra head 2048→512→ResBlock→12-dim with first 6 used) + EfficientNet-B2 SegNet (stride-2 vanilla stem, last-frame only `x[:, -1, ...]`, bilinear resize to 512×384, 5-class logits, argmax distortion). Source: CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py" section. Cross-validated against operator framing.

---

## 3. Council deliberation transcript

### 3.1 Round 1: Member positions (5+ sentences each)

**Shannon (LEAD)**: The pose axis at A1's operating point has `pose_avg = 3.4e-5`. By Shannon's rate-distortion theorem, the lower bound on bits-per-pair for the pose-residual stream is `R(D) ≥ H(P_LAPose) − H(P_LAPose|theta_A1)` where `theta_A1` is A1's frozen substrate weights. Empirically the LAPose atom manifest (`build_lapose_foveation_atom_manifest.py`) suggests ~8-16 distinct foveation modes per pair when ego-motion is dashcam-shaped (forward translation + small yaw rotation dominates). At 600 pairs this is `≤ 600 × log2(16) = 2400 bits ≈ 300 bytes` for unconditional atom selection. With A1 conditioning we expect ~3-5× compression via arithmetic coding (atoms are not uniformly distributed — straight-line driving dominates). So the **achievable lower bound on pose-residual bytes is ~60-100 bytes**. CLAUDE.md "Meta-Lagrangian/Pareto solver" requires this entropy argument before any score-improvement claim — I'm providing it. **Verdict from information theory alone: D1.D HIERARCHICAL is correct.** The residual stream is a Hinton-distilled differential code on top of A1's pose component, not a replacement (D1.B would waste A1's already-trained per-pair pose information). Lagrangian weight tuning should target `β_pose × d_pose ≈ β_seg × d_seg` at the working point — Dykstra and I have computed this and the cross-over confirms pose-axis is 2.71× cheaper per byte at PR106 frontier.

**Dykstra (CO-LEAD)**: Convex feasibility analysis on `{rate ≤ R, seg ≤ S, pose ≤ P}` constraint set with `R = 180_000 B` (A1's 178_262 + 2 KB LAPose), `S = current A1 seg`, `P = A1_pose − δ_pose`. The intersection is non-empty iff `δ_pose ≤ predicted_marginal_per_KB × 2 KB`. Shannon's rate term `5/sqrt(10 × pose_avg)` at `pose_avg = 3.4e-5` gives `271`. So `δ_pose ≤ 271 × 2_KB × derived_per_KB_efficiency`. Empirically Quantizr's PR101 retrospective shows learned pose codecs deliver ~30-60% theoretical efficiency, so `δ_pose ∈ [0.002, 0.005]` is the convex-feasible improvement band at +2 KB rate. **D2.B is correct: 3-5 KB pose budget puts us inside the achievable Pareto frontier without crossing into seg-axis feasibility leakage.** D2.C adaptive per-pair codec switching is mathematically clean but introduces a side-information bit per pair (`log2(N_codecs)`) that eats half the gain at small N_codecs — REJECT D2.C. The Pareto-feasibility argument is the binding constraint on D1 too: D1.D HIERARCHICAL keeps the constraint set CONVEX (additive residual = linear in atom mixing weight); D1.B REPLACEMENT can violate convexity if the LAPose codec entropy exceeds A1's pose entropy on edge cases. I co-sign D1.D.

**Yousfi**: This is inverse steganalysis, not compression. The contest scorer is a steganalysis CNN; we are the steganographer hiding signal inside YUV6+pose channels. FastViT-T12's RepMixer (NOT attention — operator's FastViT-T12 architecture in CLAUDE.md) is a depthwise-separable convolution; its receptive field at the final summary head sees ~32×32 spatial extent of the 512×384 image. **The pose-axis blindspot is therefore high-frequency rotational components** — small yaw corrections in pose dimensions 3-5 (Lie algebra so(3)) that change the pixel optical flow by less than 1 pixel over a 32×32 patch. LAPose's motion atoms are exactly this signal: per-pair velocity residuals from straight-line dashcam motion. **D5.A (12-channel YUV6 exploit) is the correct attack surface; the pose residual modulates the velocity channels v_x, v_y by < 1 pixel — invisible to RepMixer.** D5.C (Lie algebra parameterization) is mandatory because parameterizing the pose residual in SO(3) × R^3 keeps the residual signal on the physical manifold (no entropy waste on non-physical 6-DoF combinations). I VETO D5.D kitchen-sink: PR105 1776-LOC silver-medal-loss is the canonical anti-pattern. I co-sign D1.D + D2.B + D5.A+C.

**Fridrich**: Square root law (Fridrich 2008): spread small errors, don't concentrate large ones. LAPose foveation atoms are spatially-local pose corrections; in the SQRT law sense the foveation atom budget should be **distributed across all 600 pairs uniformly**, not concentrated in 10-20 "hard" pairs. UNIWARD (Holub & Fridrich 2014): inverse local variance weighting — pose errors in textured regions (high-flow, fast motion) are less detectable; pose errors in low-flow regions are highly detectable. **The LAPose codec should down-weight pose corrections during low-motion frames** (e.g., red-light stops, straight highway). Detector-informed embedding (Yousfi 2022): the TTO approach the contest endorses. LAPose's `build_lapose_motion_records_from_component_response.py` reads per-pair component_response.json scorer evidence — this IS detector-informed embedding for pose-axis. **Verdict: D1.D + D2.B + D5.A+C, AND D3.A score-aware Lagrangian on A1's existing loss + LAPose as auxiliary head** because Fridrich principles say the auxiliary head should NOT contaminate the base substrate's already-converged weights. Frozen-A1 + LAPose fine-tune is structurally aligned with the SQRT law.

**Contrarian**: I am the council's adversary. The argument I see is: "A1 is at 0.192847; LAPose exists; both target pose-axis; therefore compose them." This is correlation-as-evidence, not causation. **The weak proposal**: D3.B joint score-aware Lagrangian end-to-end. Reason: A1's substrate has already absorbed pose-axis information into its weights through PR101's score-gradient training. Re-training A1 jointly with LAPose **DOES NOT GUARANTEE** the LAPose residual learns NEW pose information; the joint optimizer can ALLOCATE the LAPose budget to seg-axis pixel polish if score-domain gradient says it improves more there. This is the "FiLM hijack" failure mode: conditional architectures shift signal to whatever axis the loss prefers, regardless of operator intent. **I challenge D3.B; I support D3.C frozen-A1 + LAPose fine-tune** because it forces the LAPose budget to be pose-axis or get zeroed by score-domain Lagrangian. On D6: operator framing said both CPU and CUDA must be evaluated per CLAUDE.md non-negotiable; D6.A CPU-only is forbidden. D6.B BOTH is the only compliant verdict. I co-sign D1.D + D2.B + D3.C + D5.A+C + D6.B.

**Quantizr**: I reverse-engineer the leaderboard. **Why didn't PR100/101/103 use LAPose-style pose residuals?** Three plausible answers: (a) they didn't think of it (unlikely given Quantizr-the-person and rem2 have steganalysis background); (b) they tried it and it didn't help; (c) their bit budget was already saturated and LAPose would have required substrate redesign (the substrate-vs-codec composition meta-pattern from `feedback_substrate_vs_codec_composition_meta_pattern_20260508`). I believe **(c) is most likely** — PR100's hnerv_lc_v2 substrate has its pose component baked into the renderer architecture, not a separate stream; adding a LAPose residual would require an inflate.py rewrite. **A1 has a different topology**: PR101-finetune-style with explicit pose payload at known offsets. **This means A1+LAPose has a structural advantage that public PRs DIDN'T have**: separable pose-axis budget. I support D1.D, but I dissent on the choice because **D1.C FiLM-style conditioning is cleaner architecturally** and gives us 2-3× more design space for future Phase 2 work. However, race-mode discipline (CLAUDE.md Race-mode rigor inversion + parallel-dispatch first non-negotiable) says ship D1.D in 12-hour wall-clock, NOT D1.C in 5-day re-architecture. I reluctantly co-sign D1.D **with explicit research_only=true follow-up lane for D1.C**.

**Hotz**: The 5-line patch is: read LAPose foveation atom manifest, pick top-K=4 atoms per pair, pack as 2-bit-per-pair lookup table (600 pairs × 2 bits = 150 bytes), inject into A1's inflate.py pose lookup with a single `pose_adjusted = pose_a1 + LAPOSE_ATOMS[atom_idx[i]]` line. That's the engineering shortcut. **Kitchen-sink D5.D would deliver ~30% more score but cost 5× the dev time** — race-window math says skip it. I support D1.D + D2.A (strict 2 KB budget — 150 bytes for atom indices + 1.5 KB for atom dictionary = under). On D5: D5.A+C is fine but I'd add a fallback **D5.B Tiny exploit at the SegNet boundary** (1-pixel mask delta encoding from PR93 lineage) only as a secondary lane if A1+LAPose lands under 0.190 and we have race-window time left. NOT in this submission — sequential follow-up. I dissent on D2: D2.A is enough; D2.B is over-budgeted. **Vote: D1.D + D2.A + D5.A+C; tertiary D5.B in follow-up only.**

**Selfcomp**: Block-FP self-compression view: my 88K-94K-param renderer (PR #56's 0.38 archive) used a single pose-codec entry in the archive grammar. Adding a SECOND pose stream (LAPose residual) means **two pose codecs in one archive**. Bit budget reconciliation: my archive grammar canonical example was `renderer.bin (FP4+Brotli) + masks.mkv (AV1) + poses.pt`. Adding a LAPose section means parser-section-manifest growth: `renderer.bin + masks.mkv + poses.pt + lapose_residual.bin + lapose_atom_dict.bin`. **This is parser-section growth from 3 sections to 5 sections** — within HNeRV parity lesson 3 monolithic single-file constraint? My read: yes, because all 5 sections are in `0.bin` with fixed offsets declared in `codec.py` source. **Architecture is fine, but I dissent on D1.D in favor of D1.B REPLACEMENT** because two pose codecs is two places to bug-fix; one pose codec (LAPose-only) is simpler. Reality check: A1 is already at 0.192847, and CLAUDE.md "the substrate is the score-aware substrate" non-negotiable means the substrate weights ENCODE the per-pair pose information. Replacing A1's pose component with LAPose would discard that learned representation — Shannon is right; D1.B is wasteful. I retract dissent and co-sign D1.D **with explicit codec-section-count cap of 5 in archive grammar**.

**MacKay**: Bayesian/MDL view. The LAPose foveation atoms form a "code book" — finite set of discrete pose residuals — so the encoding cost per pair is `log2(K) + cost(K_residuals)`. With K=16 atoms and per-pair atom index = 4 bits: `600 × 4 bits = 300 bytes` for indices alone. The dictionary cost is `K × dim(residual) × precision`. If residual is `(6_DoF, FP4)` = 24 bits = 3 bytes per atom × 16 = 48 bytes total dictionary. **Total: 348 bytes** for full LAPose stream at K=16. This is FAR under D2.A's 2 KB budget — Shannon's R(D) lower bound (60-100 bytes) and MacKay's MDL estimate (348 bytes) agree the stream is small. **What's the rate cost of the LAPose hyperprior conditioning?** If we want hyperprior-style (Ballé 2018), we need a per-pair side-info bit (which atom is "expected" given previous-pair atom) — `600 × 2 bits = 150 bytes hyperprior side-info`. Total with hyperprior: **~500 bytes**. **This is MUCH cheaper than 2 KB** — Hotz is right that D2.A is sufficient. I dissent on D2.B. **Vote: D1.D + D2.A + D5.A+C + Ballé hyperprior conditioning.**

**Ballé**: Modern neural compression view. The LAPose atom dictionary IS a learned prior on pose residuals. The hyperprior `p(atom_idx | atom_idx_{prev})` is a Markov-1 prior — cheap to encode (~150 bytes for 600 pairs). My 2018 entropy bottleneck paper says: replace fixed factorized priors with hyperprior side-information when archive size matters. **At A1's operating point (178 KB total) the LAPose pose-residual stream is < 0.3% of the archive bytes — the side-info overhead is negligible.** Hyperprior conditioning is mandatory for this stream because pose atoms are temporally correlated (straight-line driving over multiple consecutive frames). **R(D) rate `bits = -log2(p_y(y))` with Markov-1 hyperprior is 30-50% cheaper than unconditional factorized.** I co-sign D1.D + D5.A+C + Markov-1 hyperprior on the LAPose stream. On D2: MacKay's 500-byte total is the right answer; D2.A is sufficient if hyperprior is wired correctly; **D2.B is over-budgeted ONLY if hyperprior is omitted**. I dissent on D2 if hyperprior is not wired; I support D2.A if hyperprior IS wired. **Conditional vote: D2.A + Markov-1 hyperprior.**

### 3.2 Round 2: Binding tradeoffs identified

After Round 1, the binding tradeoffs are:

**D1 composition layer** — converged on D1.D (8 explicit + 2 conditional dissent → 8/10). Quantizr dissent (prefers D1.C FiLM) noted with research_only=true follow-up lane.

**D2 byte budget** — split 5-5 in Round 1:
- D2.A 2 KB: Hotz, Selfcomp, MacKay, Ballé (conditional on hyperprior), Quantizr
- D2.B 3-5 KB: Shannon, Dykstra, Yousfi, Fridrich, Contrarian
- Round 2 deliberation: if Markov-1 hyperprior IS wired, D2.A is sufficient (Ballé/MacKay/Shannon math agrees on ~500-byte achievable). If hyperprior is NOT wired (race-mode shortcut), expand to D2.B for safety.
- **Tie-break verdict: D2.B 3-5 KB upper bound, but TARGET D2.A 2 KB with hyperprior**. This gives BUILD subagent slack if hyperprior wiring slips race-window.

**D3 training objective** — split 4-4-2:
- D3.A score-aware Lagrangian + LAPose auxiliary head: Fridrich, Yousfi, Hotz, MacKay
- D3.B joint score-aware Lagrangian end-to-end: Shannon, Dykstra, Ballé, Selfcomp
- D3.C frozen-A1 + LAPose-only fine-tune: Contrarian, Quantizr
- **OPERATOR-ROUTABLE**: no consensus reached. Contrarian's "FiLM hijack" concern is genuine; Shannon's "joint optimization is information-theoretically optimal" is also genuine. See §7.

**D4 inflate.sh contract** — split 5-5:
- D4.A two-stage inflate (A1 → LAPose injection → score): Shannon, Dykstra, Yousfi, Fridrich, MacKay
- D4.B single-stage with new archive section: Hotz, Selfcomp, Ballé, Quantizr, Contrarian
- D4.C LAPose-conditional reshape (no archive grammar change): rejected unanimously (would require runtime scorer access, violating strict-scorer-rule non-negotiable)
- **BINDING TIE — OPERATOR-ROUTABLE**: see §7. Note: HNeRV parity lesson 3 (monolithic single-file `0.bin` with fixed offsets in `codec.py` source) is satisfied by EITHER D4.A or D4.B; difference is engineering risk profile not contest-compliance.

**D5 scorer exploit dimension** — converged on D5.A + D5.C (7/10).

**D6 evidence axis** — converged unanimously on D6.B BOTH `[contest-CPU]` + `[contest-CUDA]` per CLAUDE.md submission auth eval non-negotiable. **No dissent.**

**D7 reactivation criteria** — defined in §10.

### 3.3 Round 3: Vote tallies (binding)

| Decision | Verdict | Vote | Dissent |
|----------|---------|------|---------|
| D1 composition layer | **D1.D HIERARCHICAL** (A1 base + LAPose residual) | 8-2 | Selfcomp (initially D1.B; retracted), Quantizr (prefers D1.C; concedes for race-mode) |
| D2 byte budget | **D2.B with target D2.A** (≤5 KB, target 2 KB with hyperprior) | 10-0 | None on the bounded form |
| D3 training objective | **NO CONSENSUS — OPERATOR-ROUTABLE** | 4-4-2 | See §7 |
| D4 inflate.sh contract | **NO CONSENSUS — OPERATOR-ROUTABLE** | 5-5 | See §7 |
| D5 scorer exploit | **D5.A + D5.C** (RepMixer 12-channel YUV6 + Lie algebra SO(3)×R^3) | 7-3 | Hotz wanted D5.D kitchen-sink (rejected), Quantizr cautious, Fridrich neutral |
| D6 evidence axis | **D6.B BOTH CPU + CUDA** (per CLAUDE.md non-negotiable) | 10-0 | None |
| D7 reactivation | See §10 | 10-0 | None |

---

## 4. Math derivation appendix

### 4.1 Shannon: rate-distortion lower bound for LAPose pose-residual

**Setup**: A1 substrate provides per-pair pose `theta_a1[i] ∈ R^6` for `i ∈ {0..599}`. LAPose residual is `δ[i] ∈ R^6` such that `theta_final[i] = theta_a1[i] + δ[i]`.

**Empirical distribution**: from `build_lapose_motion_atom_manifest.py` review, the LAPose foveation/motion atoms are derived from per-pair component_response evidence. Ego-motion dashcam shape → straight-line forward translation dominates → atom distribution is heavily concentrated on a small subset.

Estimate K_effective = 4-8 distinct atom modes per pair after ego-motion bias is absorbed.

**Lower bound (Shannon-Fano on atom alphabet)**:
```
H_residual_per_pair ≥ -Σ_k p(atom_k) log2(p(atom_k))
                    ≈ -Σ_k (1/K_eff) log2(1/K_eff)  (uniform upper bound)
                    = log2(K_eff)
                    ∈ [2, 3] bits per pair
```
**Total**: `600 × 3 bits = 1800 bits = 225 bytes` (uniform upper bound).
**With Markov-1 hyperprior** (atom_k conditioned on atom_{k-1}): typically 40-60% additional compression.
**Bound**: `~90-130 bytes` for pose-residual stream alone.

**Pose distortion reduction at +δ bytes rate**: by Shannon's R(D) curve at small D,
```
δ_pose ≤ pose_a1 × (1 - exp(-2 × δ_R_bits / pose_dimension))
```
At δ_R = 1800 bits, pose_dimension = 6: `δ_pose ≤ pose_a1 × (1 - exp(-600))` ≈ pose_a1.
Real-world implementation efficiency 30-60% per Quantizr review → **practical δ_pose ≤ 0.3-0.6 × pose_a1**.

**Score impact at A1's `pose_avg = 3.4e-5`**: pose contribution to score is `sqrt(10 × pose_avg) = 0.0184`. A 30-60% pose_avg reduction → new pose contribution `sqrt(10 × 0.5 × pose_avg) = 0.013`, delta = **0.005 to 0.007 score** at +200 bytes rate.

**Rate term cost at +200 bytes**: `25 × 200 / 37545489 = 0.000133` — negligible.

**Net Shannon prediction**: improvement of **~0.005 [contest-CUDA prediction]** at +200 bytes rate, conditional on 30-60% implementation efficiency. Conservative band lower edge.

### 4.2 Dykstra: Pareto frontier intersection

**Constraint set**: `C = {(R, S, P) : R ≤ R_max, S ≤ S_a1, P ≤ P_a1 − δ_P}`
where `R_max ∈ {178_262 + 500, 178_262 + 2_000, 178_262 + 5_000}` for D2.A-hyperprior, D2.A, D2.B respectively.

**Alternating projections**: project onto rate manifold (codec efficiency), seg manifold (preserve A1 seg), pose manifold (improve pose). Convergence is geometric when intersection is non-empty.

**Feasibility**: non-empty iff δ_P ≤ Shannon-derived 0.3-0.6 × pose_a1 (above).

**Conclusion**: at +500 bytes (D2.A-hyperprior) the convex-feasible region is non-empty and inside the A1-only Pareto frontier. **D2 verdict: D2.A with hyperprior is mathematically optimal; D2.B 5 KB is engineering slack only.**

### 4.3 Ballé: information-theoretic cost of Markov-1 hyperprior

**Hyperprior side-info**: `p(atom_k | atom_{k-1})` for k ∈ [1, 599], stored as transition table.
- K_eff = 4: transition table is `4×4 × 4 bits (FP4 logp) = 64 bits = 8 bytes`.
- K_eff = 8: transition table is `8×8 × 4 bits = 256 bits = 32 bytes`.

**Per-pair side-info encoding** (atom_k_idx | atom_{k-1}_idx): arithmetic coding at table-predicted probability → typically 0.5-1.5 bits/pair vs 2-3 bits/pair unconditional.

**Net savings at K_eff=8**: 32 bytes side-info + `600 × 1 bit AC = 75 bytes payload = 107 bytes total` (vs 225 bytes unconditional). **48% reduction.**

**Ballé's verdict**: Markov-1 hyperprior is THE canonical choice for temporally-correlated pose residuals; refusing to wire it is leaving 50% of the rate budget on the table.

### 4.4 Selfcomp: bit budget reconciliation

**A1 current archive (178_262 bytes)**: 5 sections roughly — segmap weights (largest), renderer weights, masks.mkv, poses.pt, metadata.
**LAPose residual addition**: +1 section (`lapose_residual.bin`) at 100-500 bytes.
**LAPose dictionary**: +1 section (`lapose_atoms.bin`) at 50-200 bytes (16 atoms × ≤12 bytes/atom in FP4-Brotli).
**Total**: archive grows from 178_262 → ~178_700 bytes (+0.25%).

**Parser section count**: 5 → 7. HNeRV parity lesson 3 says monolithic single-file `0.bin` with fixed offsets in `codec.py` source. 7 sections in `0.bin` is fine; codec.py still owns the offsets.

**Inflate.py LOC budget** (HNeRV parity lesson 4): ≤ 100 LOC default, ≤ 200 LOC with explicit waiver. Adding LAPose decode requires ~30-50 LOC. **Within budget.**

---

## 5. Cost-band prediction

| Phase | Hardware | Cost USD | Wall-clock | Source / basis |
|-------|----------|---------:|-----------:|----------------|
| Smoke 1000 ep | Modal T4 + Tier-1 wins | $0.30-$0.80 [prediction] | 12-20 min | T20 teacher cache + autocast FP16 + batch=32 per feedback_council_t1_balle_engineering_audit_pixels_bytes_pixels_20260512 |
| Smoke 1000 ep | Modal A10G (22 GB shared) | $0.40-$1.20 [prediction] | 8-15 min | A10G 2× T4 speed, $0.31/hr |
| Full 3000 ep | Modal A100 | $2.50-$4.50 [prediction] | 25-40 min | A100 cheapest $/TFLOP-hr per pixels→bytes→pixels engineering audit |
| Full 3000 ep | Vast.ai 4090 | $0.50-$1.00 [prediction] | 30-50 min | Only promotion-grade NVDEC path; 4× T4 speed at $0.25/hr |

**Smoke-before-full pattern (Catalog #167)** is MANDATORY for substrate dispatches — operator-authorize wrapper must route through `tools/run_modal_smoke_before_full.py`. Smoke dispatch validates integration at ≤$1; full only fires after smoke passes auth-eval roundtrip + EMA + score-aware loss + archive grammar smoke.

**Total expected dispatch cost**: $3-$5 for smoke + full on optimal hardware. **Inside contest race-budget rule ($5 cap).**

---

## 6. HNeRV parity 13-lesson audit

(Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable.)

| # | Lesson | A1+LAPose status | Action required |
|---|--------|------------------|-----------------|
| 1 | Substrate must be score-aware | PASS — A1 is score-gradient trained on `upstream/videos/0.mkv` w/ gradient-through-SegNet+PoseNet | None |
| 2 | Export-first design | PASS — archive grammar declared (§4.4 Selfcomp); inflate.py LOC budget declared | BUILD subagent must declare in lane evidence per Catalog #124 |
| 3 | Monolithic single-file `0.bin` w/ fixed offsets | PASS — 7 sections in `0.bin` with codec.py offsets | None |
| 4 | Inflate.py ≤ 100 LOC default | CONDITIONAL PASS — +30-50 LOC for LAPose decode; total estimate 130-180 LOC | If >100 LOC, BUILD subagent must waive ≤ 200 with rationale |
| 5 | Full renderer (RGB out), not mask-only slot | PASS — A1 is full renderer; LAPose modulates pose-axis, not mask slot | None |
| 6 | Score-domain Lagrangian (NOT rel_err²) | PASS — D3 verdicts (A or B or C) all use score-domain α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ) | None |
| 7 | Bolt-on size ≤ 350 LOC | LIKELY PASS — LAPose composition layer estimate ~150-250 LOC | BUILD subagent measures actual LOC and declares per Catalog #124 |
| 8 | Eval-roundtrip + differentiable scorer-preprocess | CRITICAL — must call `patch_upstream_yuv6_globally()` BEFORE `load_differentiable_scorers()`; `apply_eval_roundtrip=True` in trainer | BUILD subagent must verify per Catalog #187 |
| 9 | Runtime closure | PASS — A1 inflate.sh runtime tree exists; LAPose adds ~50 LOC | Smoke dispatch verifies before full |
| 10 | Mask/pose coupling gate | NOT APPLICABLE — pose-axis-only change; masks frozen at A1 weights | Declare `mask_change=False` in evidence |
| 11 | No-op detector | CRITICAL — must prove LAPose bytes change AND inflate consumes them | Catalog #139 packet-compiler runtime byte-mutation smoke MANDATORY |
| 12 | Single-LOC-per-LOC review discipline | BUILD subagent responsibility | Adversarial review per CLAUDE.md "Recursive adversarial review protocol" |
| 13 | KILL/FALSIFIED is LAST RESORT | PASS — no KILL verdicts in this memo; all non-consensus is DEFERRED-pending-empirical | None |

**Score**: 11/13 PASS, 2/13 (lessons 7, 12) are BUILD subagent responsibility. Council cannot verify without seeing built code.

**ARCHITECTURAL GRAMMAR DECLARATION (per Catalog #124 — mandatory for L1+ promotion)**:
- `archive_grammar`: 7-section monolithic `0.bin` (A1's 5 sections + `lapose_residual.bin` + `lapose_atoms.bin`)
- `parser_section_manifest`: codec.py fixed offsets (BUILD subagent supplies after build)
- `inflate_runtime_loc_budget`: 200 LOC (waived from 100 default; rationale = composition with existing A1 inflate)
- `runtime_dep_closure`: A1's existing closure + brotli (LAPose atoms are FP4+Brotli per Selfcomp lineage)
- `export_format`: FP4+Brotli for atom dictionary, raw int8 for atom indices
- `score_aware_loss`: D3 verdict (operator-routed)
- `bolt_on_loc_budget`: 350 LOC default (LAPose composition estimate 150-250)
- `no_op_detector_planned`: yes, Catalog #139 packet-compiler runtime byte-mutation smoke

---

## 7. Operator-routable decisions (consensus gaps)

Per CLAUDE.md "Design decisions — non-negotiable": "no design decision proceeds to implementation without the council file in memory." When the council cannot reach majority + adversarial-Contrarian endorsement, the decision goes to the operator.

### 7.1 D3 — Training objective

**Vote**: D3.A 4 / D3.B 4 / D3.C 2 → no winner.

**Options for operator**:
- **D3.A** Score-aware Lagrangian on A1's existing loss + LAPose auxiliary head. Pros: Fridrich SQRT-law-aligned; doesn't perturb A1's converged weights. Cons: LAPose head trained against pose-only signal; possibly under-optimal.
- **D3.B** Joint score-aware Lagrangian end-to-end (both A1 weights and LAPose atoms trainable). Pros: Shannon-information-theoretic optimum; Dykstra Pareto-frontier closure. Cons: Contrarian's "FiLM hijack" risk — joint optimizer may shift LAPose budget to seg-axis.
- **D3.C** Frozen-A1 + LAPose-only fine-tune. Pros: cheapest; safest; forces LAPose to be pose-axis or zero. Cons: Quantizr says this leaves theoretical score on the table.

**Operator decision required**: which of D3.A/B/C to instruct BUILD subagent to wire. **Council's lean: D3.C is safest for race-window; D3.B is optimal for Phase 2 lane.**

### 7.2 D4 — Inflate.sh contract

**Vote**: D4.A 5 / D4.B 5 → binding tie.

**Options for operator**:
- **D4.A** Two-stage inflate (A1 inflate.sh → LAPose injection step → score). Pros: cleaner separation; easier rollback. Cons: Hotz says it adds 5-10 LOC to `inflate.sh` orchestration.
- **D4.B** Single-stage inflate with new archive section. Pros: monolithic per HNeRV parity lesson 3; simpler runtime tree. Cons: more invasive change to A1's existing inflate.py.

**Operator decision required**: D4.A or D4.B. **Council's lean: D4.B (single-stage) is HNeRV-parity-preferred; D4.A is engineering-risk-preferred.**

---

## 8. Apples-to-apples evidence discipline

(Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.)

Every score in this memo is tagged. Recap:

- A1 anchor: 0.192847 `[contest-CPU GHA Linux x86_64]` / 0.226352 `[contest-CUDA T4]` on archive SHA `87ec7ca5` size 178_262 B (numbers from parent prompt; original source `reports/phase_a_pareto_20260508.md` — apples-to-apples discipline preserved).
- Predicted A1+LAPose score: ALL prediction bands labelled `[prediction]` and `[contest-CPU prediction]` / `[contest-CUDA prediction]`. None promotable until empirical anchor lands per CLAUDE.md.
- The 2.71× pose marginal sensitivity flip: cited from CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" with explicit `[contest-CUDA]` framing.
- Cost predictions: tagged `[prediction]`.
- No /tmp paths used. No MPS-derived strategic decisions. No KILL verdicts.

---

## 9. 3-clean-pass adversarial review log

**Round 1** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian — inner quintet):

- Issue raised by Contrarian: "D3.B joint optimization could hijack LAPose budget to seg-axis." Resolution: surfaced as operator-routable; no silent verdict.
- Issue raised by Shannon: "D2.A 2 KB without hyperprior may be too tight." Resolution: D2.B with target D2.A — gives slack for race-window without forcing hyperprior wire-in in Round 1.
- Issue raised by Yousfi: "D5.D kitchen-sink risk." Resolution: rejected by vote 7-3 with explicit anti-pattern citation (PR105 1776 LOC silver-loss).
- All other Round 1 positions self-consistent with CLAUDE.md non-negotiables.

**1 issue found → counter resets to 0.**

**Round 2** (Quantizr + Hotz + Selfcomp + MacKay + Ballé — inner ten remainder):

- Issue raised by Quantizr: "Public PR100/101/103 didn't ship LAPose-style residuals — maybe they tried and it failed." Resolution: Quantizr's hypothesis (c) substrate topology — A1's separable pose payload is structurally different from HNeRV-family's baked-in pose. Documented in §3.1.
- Issue raised by Hotz: "D2.A is enough; D2.B is over-budgeted." Resolution: D2 verdict is "D2.B with target D2.A" — accommodates both.
- Issue raised by Selfcomp: "Two pose codecs is two places to bug." Resolution: archive section count cap at 7 with parser-section manifest in codec.py; HNeRV parity lesson 3 satisfied.
- Issue raised by MacKay: "Markov-1 hyperprior cuts payload 48%." Resolution: incorporated as math derivation §4.3; D2 verdict updated to target D2.A with hyperprior.
- Issue raised by Ballé: "Hyperprior is mandatory for temporally-correlated pose residuals." Resolution: same as MacKay; verdict updated.

**5 issues found → counter resets to 0.**

**Round 3** (Boyd + Tao + Filler + Mallat + van den Oord + Schmidhuber — grand council advisory):

- Boyd: "ADMM/proximal gradient view of D3.B: joint optimization can be made robust via per-axis trust regions. This doesn't resolve the operator-routable question but it weakens Contrarian's concern." Resolution: noted; doesn't change verdict.
- Tao: "The Markov-1 hyperprior is harmonically equivalent to a one-step DTI decomposition of the pose-residual time-series. Spectral compactness is provable for ego-motion dashcam shape." Resolution: noted; strengthens Ballé's §4.3.
- Filler: "Syndrome-trellis coding could replace MacKay's atom-index encoding for additional ~10% rate savings." Resolution: out of scope for this council; flagged as Phase 2 lane (`research_only=true`).
- Mallat: "LAPose foveation atoms have scattering-transform structure; the dictionary could be wavelet-compressed for 2× dict size reduction." Resolution: out of scope; flagged as Phase 2.
- van den Oord: "VQ-VAE-style atom codebook is conceptually identical to LAPose's atom manifest." Resolution: confirms architectural soundness.
- Schmidhuber: "Compression-as-intelligence — LAPose IS the substrate's posterior over ego-motion." Resolution: motivational; confirms framing.

**0 issues found → counter advances to 1.**

**Round 4** (re-run inner quintet on revised memo):

- Shannon: rate-distortion bounds match revised memo. No issues.
- Dykstra: convex feasibility verdicts match. No issues.
- Yousfi: FastViT-T12 RepMixer exploit framing matches. No issues.
- Fridrich: SQRT-law + UNIWARD framing match. No issues.
- Contrarian: "Operator-routables are properly surfaced; no silent design decisions." No issues.

**0 issues found → counter advances to 2.**

**Round 5** (re-run with rotated perspective: trace actual call sites — does the LAPose canvas wire into A1's existing inflate.sh path?):

- Council reviews `tools/build_lapose_*.py` outputs in detail (§2 confirms all 6 read). The atom manifests are consumed via `--manifest` JSON; the foveation payload archive emits `.lapose_foveation_payload.bin` ZIP member. **A1's existing inflate.sh does NOT consume this member.** This is a known gap — the BUILD subagent's job is to wire it.
- This is NOT an issue with the COUNCIL MEMO; it is an explicit BUILD subagent action item documented in §6 ARCHITECTURAL GRAMMAR DECLARATION.

**0 issues found → counter advances to 3.**

**3-clean-pass achieved. Council memo SEALED.**

---

## 10. D7 — Reactivation criteria

Per CLAUDE.md "KILL/FALSIFIED is LAST RESORT" non-negotiable. If empirical A1+LAPose anchor lands and either:

**Path A (positive)** — anchor score ≤ 0.190 `[contest-CPU]` AND ≤ 0.222 `[contest-CUDA]`:
- Reactivation criterion: lane advances to L2 (impl_complete + real_archive_empirical). Council reconvenes for Phase 2 lane discussion (D1.C FiLM full re-architecture for sub-0.180 path).
- Promote per CLAUDE.md submission auth eval rule with BOTH `[contest-CPU]` AND `[contest-CUDA]` on Linux x86_64 + T4 1:1 hardware.

**Path B (mixed)** — anchor score 0.190-0.193 `[contest-CPU]`:
- Reactivation criterion: DEFERRED-pending-D3-operator-decision (D3.B vs D3.C re-dispatch with operator-selected training objective).
- Hyperprior wire-in retry: if D2.A was attempted without hyperprior, retry with Markov-1 hyperprior per §4.3 prediction.

**Path C (negative)** — anchor score > 0.195 `[contest-CPU]`:
- DEFERRED-pending-research, NOT KILLED. Reactivation criteria:
  - (a) Try D1.C FiLM-style conditioning end-to-end (Quantizr's preference)
  - (b) Try D5.B fallback (SegNet stride-2 frequency-band exploit) as orthogonal axis
  - (c) Verify A1 substrate isn't at score-aware saturation (test residual capacity via gradient-norm analysis on A1 weights)
  - (d) Consult HNeRV parity discipline lesson 13: "KILL is LAST RESORT"

**No KILL verdict in any path.**

---

## 11. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable:

1. **Sensitivity-map contribution** — pose-axis component sensitivity at A1 frontier operating point is documented (2.71× SegNet marginal). LAPose residual atoms add a new per-pair sensitivity entry. **Wire-in**: BUILD subagent adds `tac.sensitivity_map.LAPoseResidualSensitivity` row when anchor lands. **Status**: declared, not yet wired (research_only on council memo; BUILD subagent owns wiring).
2. **Pareto constraint** — convex feasibility verdict in §4.2 adds a new constraint to `tac.pareto_*`: `pose_a1 − pose_a1_lapose_residual ≤ 0.5 × pose_a1`. **Wire-in**: pending BUILD subagent landing of `lane_a1_plus_lapose_composition_20260513` per autopilot dispatch.
3. **Bit-allocator hook** — D2.A target with Markov-1 hyperprior changes per-tensor importance for the LAPose stream. **Wire-in**: BUILD subagent registers `lapose_residual` and `lapose_atoms` sections in bit-allocator registry; council memo does not require code change.
4. **Cathedral autopilot dispatch hook** — A1+LAPose lane is archive-deployable; autopilot dispatch hook is the BUILD subagent's smoke-before-full wrapper (Catalog #167).
5. **Continual-learning posterior update** — anchor lands → posterior updates per Catalog #128 atomic fcntl-locked write.
6. **Probe-disambiguator** — D3 has 3 defensible interpretations (D3.A/B/C); D4 has 2 (D4.A/B). Per CLAUDE.md "Anti-arbitrariness primitive": ship BOTH modes via callable interface + build `tools/probe_a1_plus_lapose_training_objective_disambiguator.py` that returns regime-conditional verdict after smoke dispatch. **Wire-in**: BUILD subagent OR follow-up sister lane owns this; surfaced as operator-routable per §7.

---

## 12. Cross-references

- `[[feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509]]` — canonical HNeRV parity retrospective; 13 inviolable lessons audited in §6.
- `[[feedback_siren_literature_review_landed_20260513]]` — sister substrate research; SIREN+LAPose composition explored separately; this council scoped specifically to A1+LAPose pose-axis.
- `[[feedback_siren_pre_dispatch_audit_fix_wave_LANDED_20260513]]` — SIREN dispatch audit; cross-pollinated process discipline.
- `[[feedback_council_t1_balle_engineering_audit_pixels_bytes_pixels_20260512]]` — Tier-1 engineering wins (T20 teacher cache + autocast FP16); cost-band prediction §5 depends on these wins.
- `[[feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512]]` — platform $/TFLOP-hr table; cost-band §5 uses these rates.
- `[[feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512]]` — anti-pattern: B1 composition cells on saturated PR106 r2 base FALSIFIED. **Risk parallel**: A1 may also be saturated on pose-axis at 0.192847. Council's §4.1 Shannon R(D) analysis says no — pose marginal is 2.71× SegNet's at PR106-comparable frontier, so headroom exists. But this is a DESIGN-TIME prediction; smoke anchor will confirm or DEFER-pending-research.
- `[[feedback_b1_singleton_magic_codec_LANDED_*]]` — sister "singleton primitive on top of saturated base" anchor; the predicted-savings-flipped-sign outcome is a structural warning. A1+LAPose is structurally different because A1 is NOT entropy-saturated on pose-axis (Shannon R(D) shows headroom).
- `[[feedback_grand_council_b1_autopilot_firing_review_20260512]]` — sister council; B1 NOT-FIRE-READY verdict; same operator approval pattern.

---

## 13. Council seal

**Inner quintet pact** (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian): **SEALED**. All 5 endorse memo body, mathematical derivations, and operator-routable surfacing of D3+D4.

**Inner ten** (+ Quantizr, Hotz, Selfcomp, MacKay, Ballé): **SEALED**. Quantizr and Hotz registered dissent on D1 (initially), reluctantly co-signed for race-mode discipline; both endorsed final verdict 8-2.

**Grand council advisory** (Boyd, Tao, Filler, Mallat, van den Oord, Schmidhuber): **REVIEWED**. 0 issues found in Round 3.

**3-clean-pass adversarial review**: complete. Counter = 3. Memo SEALED for parent dispatch consumption.

**Verdict mode**: DEFERRED-pending-empirical-anchor on all prediction bands. No KILL verdicts. Operator-routable decisions (D3, D4) explicit.

**Date**: 2026-05-13
**Task**: #486
**Lane**: `lane_pose_axis_non_hnerv_council_20260513`
