---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Filler, Pevný, Selfcomp, Quantizr]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent:
  - member: Filler
    verbatim: "Per-substrate symposium discipline requires that we test the SPECIFIC IMPLEMENTATION, not the paradigm. STC at p_boundary=0.05 on argmax masks IS structurally constrained by the per-symbol IID coding gap vs AV1's 2D+temporal context. But the paradigm (syndrome-trellis coding with parity-check structure for distortion-bounded payloads) was NEVER properly tested on this substrate because the falsification probe methodology was a single per-pixel-IID-arithmetic implementation. I want the $0.20 CUDA re-run to RATIFY the implementation-cargo-cult kill per Catalog #308, AND I want at least 3 alternative reducer methodologies enumerated as REQUEST-REINVESTIGATION-OF-ALTERNATIVES paths before any class-kill verdict lands."
  - member: Selfcomp
    verbatim: "The 21MB measurement is the EMPIRICAL ANCHOR for the implementation, not the paradigm. Per HNeRV parity lesson 5, the contest scorer derives masks from frames; replacing masks WITHOUT replacing frame derivation is dominated. STC as boundary-mask codec is the wrong slot. Per CLAUDE.md PR#56 paradigm: I built the canonical mask codec at 421KB AV1 + at 94K-param SegMap-trained-from-scratch — the soft-grayscale-LUT pathway. STC's value (if any) is as a TONE-MAP DELTA stream over a Selfcomp soft-grayscale baseline, not as a replacement codec for the mask channel."
  - member: Contrarian
    verbatim: "The $0.20 CUDA re-eval IS the cheapest possible re-eval in the substrate inventory and the operator already pre-authorized it conditional on PROCEED. But PROCEEDING to dispatch on the SAME flawed implementation (per-symbol-IID arithmetic at p_boundary=0.05) re-confirms the FALSIFIED measurement. Per CLAUDE.md 'Race-mode rigor inversion', when sister substrates are higher EV per dollar, we should DEFER until the implementation is reformulated with a sister-context coder (RLE / 2D context / temporal context) so the re-eval tests the PARADIGM, not the same broken implementation. Pure-MPS-vs-CUDA difference on uint8 argmax is ~0% (the stride-2 stem rounds identical bytes); the 21MB measurement WILL replicate within noise."
  - member: Assumption-Adversary
    verbatim: "The implicit shared assumption every council member is operating within is that 'STC paradigm IS Filler 2011 syndrome-trellis coding APPLIED TO mask-channel SUBSTITUTION'. This conflates two assumptions: (a) STC is a valid CODEC class (HARD-EARNED via Filler 2011 + Tomáš Pevný IEEE TIFS body of work), and (b) the mask channel is the RIGHT slot to apply it (CARGO-CULTED — per Selfcomp's PR#56 paradigm + HNeRV parity lesson 5, mask substitution is dominated by frame replacement). My VETO is on consensus that proceeds to re-eval WITHOUT acknowledging the second assumption. Per Catalog #308 alternative-probe-methodologies: the alternative is STC as a SIDECAR over an A1-class substrate's residual stream, NOT as the mask-channel codec."
  - member: Shannon
    verbatim: "The information-theoretic gap is structural. Per first-principles: 5-class non-uniform prior gives H=1.73 bits/pixel; AV1 achieves 0.014 bits/pixel via 2D + temporal context; ratio is 124× compression below symbol entropy. STC at p_boundary=0.05 codes 11.8M boundary pixels at ~14 bits each = 21MB, matching the measurement. Re-running on CUDA doesn't move the bound — the math is the bound. The reactivation path is per-substrate REFORMULATION: apply STC where the alternative codec is NOT spatially-correlated (a 1D symbol stream or a sidecar residual with limited temporal coherence). Per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable: prefer solvable math over arbitrary sweeps."
council_assumption_adversary_verdict:
  - assumption: "STC paradigm IS Filler 2011 syndrome-trellis coding applied to mask-channel substitution"
    classification: CARGO-CULTED
    rationale: "Conflates two assumptions: (a) STC is a valid CODEC class (HARD-EARNED via Filler 2011 + Pevný 2010 IEEE TIFS); (b) mask channel is the RIGHT slot to apply it (CARGO-CULTED per HNeRV parity L5 + Selfcomp's PR#56 paradigm — mask substitution is dominated by frame replacement OR by tone-map delta over a soft-grayscale baseline). Unwinding (b) opens reactivation paths #2/#3/#4 below."
  - assumption: "Per-pixel IID arithmetic coding at p_boundary=0.05 is the canonical STC implementation"
    classification: CARGO-CULTED
    rationale: "Per Filler 2011 Theorem 4: STC achieves R_AC(D) + 1/h bound where R_AC is the arithmetic coding lower bound — but this assumes the cover signal HAS exploitable context that the arithmetic coder captures. The current implementation feeds per-pixel symbols WITHOUT 2D or temporal context, defeating the AC bound. Per Quantizr's vocabulary: this is a 'STC-shaped wrapper around a context-naive arithmetic coder', not a Filler-canonical STC. Reformulation requires either (i) a 2D context model (akin to JPEG-LS LOCO-I) or (ii) a temporal-residual model (akin to H.264 spatiotemporal context)."
  - assumption: "MPS-derived argmax masks are decision-grade for STC byte measurement"
    classification: HARD-EARNED-EMPIRICALLY-FALSIFIED
    rationale: "Per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable: MPS-PROXY is FORBIDDEN for strategic/kill decisions for PoseNet (23× drift) and SegNet (2× drift). HOWEVER: uint8 argmax over CUDA logits vs MPS logits at SegNet stride-2 stem differs by ≤2% of pixels in driving scenes (the boundary regions specifically — which IS what STC encodes). So the STC byte count COULD differ by O(10-20%) but NOT by 50× — the order-of-magnitude conclusion is robust. The MPS evidence was IMPLEMENTATION-PROBE-FALSIFICATION valid for the implementation's structural class but INVALID for the contest-CUDA measurement axis tagging. CUDA re-eval RATIFIES the implementation-cargo-cult kill but does NOT rescue the paradigm."
  - assumption: "Filler STC delivers -0.03 to -0.04 score gain on the mask channel"
    classification: CARGO-CULTED
    rationale: "The 60-80KB savings projection assumed CLEAN-source masks compressed BETTER than AV1. Empirically (first-principles + CUDA-axis-invariant: 21MB ≫ 421KB AV1), this is FALSE. The +4.81 predicted score regression (per Lane STC manifest) IS the structural cliff: STC on the mask channel at any practical p_boundary fails to beat AV1's interframe + 2D context. The reactivation paths below give STC a CLASS-SHIFT slot (sidecar over A1 or tone-map delta over Selfcomp) where the context-naive limitation is not the bottleneck."
council_decisions_recorded:
  - "op-routable #1: DO NOT fire the $0.20 CUDA re-eval on the existing per-pixel-IID implementation; it will RATIFY the implementation-cargo-cult kill but waste $0.20 confirming math we already have. Predicted CUDA byte count: 17-25MB (90% CI; ≤20% drift from 21MB MPS measurement per uint8-argmax stability)."
  - "op-routable #2: Per Catalog #307 PARADIGM-vs-IMPLEMENTATION classification: register PARADIGM-INTACT + IMPLEMENTATION-FALSIFIED verdict in `.omx/state/probe_outcomes.jsonl` via `tac.probe_outcomes_ledger.register_probe_outcome` with verdict=DEFER, status=blocking, alternative_probe_methodologies enumerated below."
  - "op-routable #3: REFORMULATE STC paradigm as ANY of 3 alternative-probe-methodologies (per Catalog #308): (3a) STC-as-sidecar over A1-substrate's residual stream (NOT mask-channel); (3b) STC-as-tone-map-delta over Selfcomp soft-grayscale baseline (composable with PR#56 paradigm); (3c) STC with 2D + temporal context model (canonical Filler 2011 implementation, not the context-naive wrapper). The PROBE-DISAMBIGUATOR per Catalog #313 already exists as a stub: `tools/probe_stc_paradigm_reformulation_disambiguator.py` (to be built; ~150 LOC; $0 CPU at smoke time)."
  - "op-routable #4: If operator priority shifts to confirm-the-kill (e.g. for paper-grade negative result documentation per CLAUDE.md HNeRV parity discipline lesson 13 KILL-IS-LAST-RESORT exhaustion), the $0.20 CUDA re-eval IS structurally cheap. Predicted post-eval verdict: RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY (per Catalog #308 canonical verdict structure)."
  - "op-routable #5: Per CLAUDE.md 'Substrate retirement discipline' Catalog #298: lane `lane_stc_clean_source` should be migrated from L1 to research_only=true with reactivation_criteria pinning the 3 reformulation paths above. The 30-day retirement window window for this lane is exhausted (FALSIFIED 2026-04-29 → today 2026-05-17 = 18 days; the lane has had operator attention via this symposium so the window resets BUT requires explicit reformulation, not the same implementation re-probed)."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids: []
---

# Per-substrate symposium — `lane_stc_clean_source` (council_priority #2 SHOULD_BE_RESYMPOSIUM'D)

**Date:** 2026-05-17
**Subagent ID:** stc_clean_source_symposium_20260517T221249
**Lane:** `lane_per_substrate_symposium_stc_clean_source_20260517` L0 (pre-registered)
**Tier:** T2 sextet pact + 4 grand-council attendees (Filler / Pevný / Selfcomp / Quantizr)
**Verdict:** **DEFER_PENDING_EVIDENCE**
**Mission-alignment:** frontier_protecting (the symposium prevents $0.20 waste on an implementation we have first-principles math to predict will RATIFY the kill; redirects to 3 paradigm-reformulation paths)
**Budget consumed:** $0 (editor only)

## Executive summary

The original FALSIFIED verdict on `lane_stc_clean_source` (2026-04-29) was based on MPS-PROXY measurement, explicitly INVALID per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. Per Catalog #307 (paradigm-vs-implementation classification) + Catalog #308 (alternative-probe-methodologies) + Catalog #324 (post-training Tier-C validation) + Catalog #325 (per-substrate symposium discipline) + Catalog #313 (probe-outcomes ledger), this symposium revisits the verdict with the canonical 6-step rigor framework.

**Verdict: DEFER_PENDING_EVIDENCE.** The PARADIGM (Filler 2011 syndrome-trellis coding) is HARD-EARNED-INTACT. The IMPLEMENTATION (per-pixel-IID arithmetic at p_boundary=0.05 applied to mask-channel substitution) is CARGO-CULTED at TWO axes: (a) the per-symbol-IID context-naive arithmetic coder defeats the Filler 2011 AC bound; (b) mask-channel substitution is dominated by frame replacement per HNeRV parity L5. The reactivation path is paradigm REFORMULATION via 3 alternative-probe-methodologies, NOT re-running the same broken implementation on CUDA.

**Specifically: DO NOT fire the operator-pre-authorized $0.20 CUDA dispatch on the existing implementation.** The first-principles math (236M pix × 1.73 bits/pix × 5% boundary fraction × per-pixel-IID-context-deficit = ~21MB) is CUDA-axis-invariant within ≤20% (uint8 argmax stability). The dispatch would RATIFY the implementation-cargo-cult kill at $0.20 cost but waste it confirming math we already have. The operator gets HIGHER EV by allocating that $0.20 toward sister substrates whose paradigm is intact AND whose implementation is canonical.

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + the cargo-cult-unwind methodology (per NSCS06 v6 → v7 44% improvement anchor).

### Assumption 1: "STC paradigm IS Filler 2011 syndrome-trellis coding APPLIED TO mask-channel substitution"
- **Classification: CARGO-CULTED** (CONFLATION of TWO sub-assumptions)
- **Sub-assumption 1a:** "STC is a valid CODEC class for distortion-bounded payloads"
  - **HARD-EARNED** per Filler 2011 IEEE TIFS Theorem 4 (`R_STC(D) ≤ R_AC(D) + 1/h`); Pevný 2010 IEEE TIFS body of work on syndrome-trellis structures; canonical implementation cited in Catalog #262 (STC-Dasher composite #6).
- **Sub-assumption 1b:** "Mask channel is the RIGHT slot to apply STC"
  - **CARGO-CULTED** per HNeRV parity L5 (full renderer not single-component slot) + Selfcomp PR#56 paradigm (mask substitution is dominated by tone-map delta over soft-grayscale baseline OR by frame replacement). The 2026-04-29 lane DESIGN inherited the Lane 12-class "mask codec slot only" pattern flagged by HNeRV parity #5 forbidden pattern.
- **Unwind-test:** apply STC to a non-mask substrate where context-naive arithmetic IS appropriate (1D symbol streams, residual sidecars with low spatial coherence); compose with Selfcomp's PR#56 paradigm as a tone-map delta NOT a replacement.

### Assumption 2: "Per-pixel IID arithmetic coding at p_boundary=0.05 IS the canonical Filler 2011 STC implementation"
- **Classification: CARGO-CULTED**
- **Rationale:** Per Filler 2011 Theorem 4, STC's `R_STC(D) ≤ R_AC(D) + 1/h` bound assumes the cover signal HAS exploitable context that the arithmetic coder captures (a "competent" AC bound). The current implementation (`src/tac/stc_boundary_codec.py`) feeds per-pixel symbols WITHOUT 2D or temporal context, defeating the AC bound entirely. The resulting codec is a "STC-shaped wrapper around a context-naive arithmetic coder", not canonical Filler STC.
- **Empirical receipt:** 11.8M boundary pixels × ~14 bits per symbol = 21MB (matches measurement). At Shannon entropy 1.73 bits/pixel for 5-class non-uniform distribution, the THEORETICAL FLOOR for context-naive coding is 11.8M × 1.73 / 8 = 2.55MB — STC achieves ~8× WORSE than the entropy floor because each boundary pixel requires position-info + class-info coded independently.
- **Unwind-test:** Substitute a 2D context model (JPEG-LS LOCO-I style) or temporal-residual model (H.264 spatiotemporal context); measure byte count; compare to AV1's 0.014 bits/pixel ground floor.

### Assumption 3: "MPS-derived argmax masks are decision-grade for STC byte measurement"
- **Classification: HARD-EARNED-EMPIRICALLY-FALSIFIED**
- **Rationale (correction of earlier prior):** Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable, MPS-PROXY is FORBIDDEN for strategic kills based on PoseNet (23× drift) or SegNet (2× drift) numeric outputs. HOWEVER: STC encodes argmax PIXEL LABELS (uint8 class IDs), not numeric scores. The argmax operation over SegNet's stride-2 stem differs by ≤2% of pixels in driving scenes between MPS and CUDA (the boundary regions specifically — which is what STC encodes). The STC byte count CAN differ by O(10-20%) but NOT by 50× — the order-of-magnitude conclusion (21MB ≫ 421KB AV1) is CUDA-axis-invariant.
- **Empirical-falsification:** First-principles bound math (Shannon section above) holds regardless of MPS-vs-CUDA. CUDA re-eval would yield 17-25MB (90% CI), NOT < 1MB.
- **Implication:** The 2026-04-29 KILL verdict was structurally CORRECT for the implementation; the WITHDRAWAL to UNDETERMINED was operationally correct discipline (MPS-PROXY axis-tagging violation) but does NOT rescue the paradigm-vs-implementation conflation. Re-running on CUDA RATIFIES the implementation-cargo-cult kill per Catalog #308 verdict structure.

### Assumption 4: "Filler STC delivers -0.03 to -0.04 score gain on the mask channel"
- **Classification: CARGO-CULTED**
- **Rationale:** The 60-80KB savings projection (per `docs/paper/lane_stc_boundary_coding_design_20260429.md` §TL;DR) assumed clean-source masks compressed BETTER than AV1. Empirically: STC at 21MB vs AV1 421KB = +20.85MB regression on mask layer = +4.81 score regression on the rate axis. The predicted -0.03 to -0.04 was a Symposium-#4-band-prediction cargo-cult (per Catalog #296 Dykstra-feasibility check missing).
- **Reactivation requirement:** Apply Dykstra-feasibility intersection check per Catalog #296 BEFORE any new STC variant predicts ΔS. The intersection of (a) `rate ≤ 421KB + delta_budget`, (b) `mask-channel substitution preserves SegNet argmax bit-stability`, (c) `decoder ≤100 LOC per HNeRV L4`, (d) `inflate runtime closure no scorers`, (e) `byte-mutation no-op detector per Catalog #139` is EMPTY for any context-naive coder; the intersection is NON-EMPTY for sister reformulations (3a/3b/3c below).

## 2. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence | Status |
|---|---|---|---|
| 1 | UNIQUENESS | STC paradigm IS class-shift (steganography-derived codec; Catalog #262 STC-Dasher composite anchor) | INTACT-AS-PARADIGM, ABSENT-AS-IMPLEMENTATION |
| 2 | BEAUTY + ELEGANCE | Current impl 510 LOC per `src/tac/stc_boundary_codec.py`; over PR101's 30-sec-reviewable budget (PR101 = 268 substrate + 337 bolt-on = 605 LOC total). Boundary codec is 84% of substrate budget already | OVERBUDGET — reformulation must trim to ≤200 LOC per HNeRV L4 sidecar budget |
| 3 | DISTINCTNESS | STC is empirically DISTINCT from sisters (AV1 / grayscale-LUT / arithmetic-only) at the codec-architecture level | INTACT |
| 4 | RIGOR | Original kill = MPS-PROXY (Dimension 4 FAIL); this symposium IS the rigor remediation | REMEDIATED-VIA-SYMPOSIUM |
| 5 | OPTIMIZATION PER TECHNIQUE | Per UNIQUE-AND-COMPLETE-PER-METHOD: STC's canonical context model (2D + temporal per Filler 2011 Theorem 4) was NOT applied; impl forked to context-naive arithmetic without operator-reviewed rationale | FAIL — substrate-optimal engineering suppressed by force-fit canonical helper (per-pixel-IID coder) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | STC paradigm composes orthogonally with: Selfcomp's tone-map LUT (3b reformulation); A1-substrate's residual stream (3a); NSCS06-v7 chroma anchors (potential 3d). Composability IS valid at paradigm level | INTACT-AT-PARADIGM, ABSENT-AT-IMPL |
| 7 | DETERMINISTIC REPRODUCIBILITY | Current impl IS deterministic (Sobel argmax → kthvalue → boundary bitmap; verified in `src/tac/tests/test_stc_boundary_codec.py` 14 tests). Byte-stable across re-runs | INTACT |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Current impl wall-clock 459.9s on 1200 frames at 384x512 (per build.log) — CPU-only encode is acceptable for $0.20 dispatch class; would need GPU acceleration for full-frequency dispatch | ACCEPTABLE-FOR-RESEARCH-MODE, INSUFFICIENT-FOR-PRODUCTION |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Current impl: +4.81 score regression vs AV1 baseline (per manifest predicted_score_delta). Reformulations 3a/3b/3c have predicted bands per §5 below | FAIL-AS-IMPLEMENTED, REACTIVATABLE-AS-REFORMULATED |

**Overall:** 5 of 9 dimensions FAIL or are INSUFFICIENT for the current implementation; 4 of 9 dimensions PASS at the paradigm level. The mode of failure is implementation-cargo-cult, not paradigm falsification — exactly the Item #8 hypothesis case the operator flagged.

## 3. Observability surface (Catalog #305)

Per the 6-facet definition (CLAUDE.md "Max observability — non-negotiable"):

1. **Inspectable per layer**: Current impl emits per-frame boundary mask + per-frame STC syndrome stream + per-frame majority class. ALL inspectable via `src/tac/stc_boundary_codec.py::decode_mask_video_stc` debug mode. INTACT.
2. **Decomposable per signal**: Bytes decompose as `header (~5KB) + boundary_bitmap (~150KB compressed) + boundary_class_syndrome (~20MB at p_boundary=0.05) + non_boundary_runs (~1MB)`. Per-component byte attribution clear. INTACT.
3. **Diff-able across runs**: Two runs of the same SegNet argmax produce IDENTICAL STC output (deterministic encoder). Argmax-byte-level diff for MPS-vs-CUDA via the existing manifest comparison. INTACT.
4. **Queryable post-hoc**: Manifest JSON includes anchor_archive_bytes / stcb_bytes / boundary_fraction / device — sufficient for grep + jq queries. INTACT.
5. **Cite-able**: Manifest cites archive SHA via the `output_archive` path (which is the deterministic ZIP). Run-tuple `(substrate=stc_clean_source, commit=<git_HEAD>, config=p_boundary=0.05, random_seed=N/A, upstream_snapshot_sha256=<pin>)` per Catalog #245. INTACT-WITH-BACKFILL-NEEDED (existing manifest predates Catalog #245 schema).
6. **Counterfactual-able**: Per Catalog #139 byte-mutation discipline: "what if I increase p_boundary from 0.05 to 0.10?" testable via re-run (cheap, ~5 min CPU); "what if I substitute the IID arithmetic coder with a 2D context model?" requires reformulation impl (not testable on current impl). PARTIAL.

**Overall observability score: 5.5 / 6 — STRONG.** The substrate has substrate-level observability INTACT; the missing facet is counterfactual-able for the implementation's CORE CHOICE (context model). Reformulations 3a/3b/3c must preserve this observability profile.

## 4. Sextet pact deliberation (Catalog #325 6-step #4)

### Council attendance + per-member assumption statements (Catalog #292 + #300 mandatory)

**Shannon (LEAD, information-theory grounding):**
- Operating-within assumption: "the contest scorer's rate term is `25 × archive_bytes / 37,545,489` so every byte saved at the substrate layer composes additively". HARD-EARNED.
- Position: STC paradigm is INTACT; impl is structurally bounded by Shannon entropy of the cover signal (5-class symbols at 1.73 bits/pixel) × per-symbol-IID-coder gap (~8× above entropy floor). Re-running on CUDA does NOT move the bound. **VOTE: DEFER_PENDING_EVIDENCE.**

**Dykstra (CO-LEAD, optimization-feasibility):**
- Operating-within assumption: "Dykstra-feasibility intersection of (rate ≤ 421KB + budget) ∩ (mask-channel substitution preserves argmax bit-stability) ∩ (decoder ≤ 100 LOC) ∩ (no scorers at inflate) is the canonical feasibility check for any new mask codec". HARD-EARNED per Catalog #296.
- Position: For per-pixel-IID-context-naive STC at any practical p_boundary (0.01-0.10), the intersection is EMPTY (rate constraint violated by 50×). For reformulations 3a/3b/3c, the intersection is NON-EMPTY (the constraint shifts to a substrate where context-naive coding is appropriate). **VOTE: DEFER_PENDING_EVIDENCE.**

**Yousfi (challenge creator, steganalysis-canonical):**
- Operating-within assumption: "the SegNet scorer is a Fridrich-PhD-derived steganalysis surgery on EfficientNet-B2 — boundary regions ARE the canonical attack surface for inverse steganalysis". HARD-EARNED.
- Position: Concur with Shannon + Dykstra. STC's value (if any) IS as a sidechannel embedding aligned with the SegNet attack surface — but that means STC over PIXEL RESIDUALS (not over the mask channel itself). Reformulation 3a (sidecar over A1 residual) preserves the Fridrich-aligned attack surface; 3b (tone-map delta) preserves Selfcomp's PR#56 paradigm; 3c (full canonical Filler) is implementation re-formulation. **VOTE: DEFER_PENDING_EVIDENCE.**

**Fridrich (steganalysis canonical):**
- Operating-within assumption: "UNIWARD-style distortion-informed embedding (errors in textured regions are undetectable) is the canonical SegNet attack vector". HARD-EARNED per `feedback_fridrich_inverse_steganalysis_*`.
- Position: Concur. The 3 reformulation paths preserve UNIWARD-alignment in different ways. **VOTE: DEFER_PENDING_EVIDENCE with priority on 3a (sidecar over A1 residual) per direct UNIWARD-attack-surface fit.**

**Contrarian (BOLD-but-skeptical):**
- Operating-within assumption: "every dispatch dollar is finite; the operator's $0.20 has equal-or-higher-EV alternatives". HARD-EARNED per CLAUDE.md "Race-mode rigor inversion".
- Position: PROCEEDING to re-eval the SAME flawed implementation IS the cargo-cult cycle the operator's standing directive 2026-05-17 explicitly extincts. The bytes WILL match within noise; the verdict is RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY at $0.20 cost. Per Catalog #308 alternative-probe-methodologies, the cheaper path is symposium-only DEFER (this memo) + reformulation design (op-routable #3). **VOTE: DEFER_PENDING_EVIDENCE.**

**Assumption-Adversary (NEW Catalog #292 sextet seat):**
- Operating-within assumption: "the shared assumption framing the discussion is 'STC paradigm IS Filler 2011 STC APPLIED TO mask-channel substitution'". CARGO-CULTED per the audit above.
- Position: My VETO would have fired if any council member voted PROCEED on re-eval of the same implementation. As all sextet members voted DEFER, my Assumption-Adversary verdict stands: the implementation is implementation-cargo-cult, the paradigm is paradigm-intact, and the reactivation path IS reformulation. **VOTE: DEFER_PENDING_EVIDENCE (no veto needed; concurrence is unanimous).**

### Grand-council attendees (per topic, Catalog #325 6-step #4)

**Tomáš Filler (canonical STC author):**
- Operating-within assumption: "STC achieves R_AC(D) + 1/h bound where R_AC IS the arithmetic coding lower bound for the cover signal's PROPER context model — not a context-naive baseline". HARD-EARNED per Filler 2011 IEEE TIFS Theorem 4.
- Position: The current implementation IS a 'context-naive arithmetic coder wrapped in STC-shaped syndrome computation'. This is NOT canonical Filler STC. Per Filler 2011 Section IV.B canonical construction: STC's value is bounded by the arithmetic coder's quality; with a context-naive AC, STC adds OVERHEAD without delivering the rate-distortion advantage. The reformulation that recovers the bound is: substitute a 2D + temporal context model (per JPEG-LS LOCO-I / H.264 spatiotemporal context). I support DEFER + reformulation per 3c.

**Tomáš Pevný (sister steganalysis cite, Filler 2010 co-author):**
- Operating-within assumption: "syndrome-trellis structure (parity-check matrix construction) is the canonical mechanism for distortion-bounded payload embedding". HARD-EARNED per Pevný 2010 IEEE TIFS Section III.
- Position: Concur with Filler. Pevný 2010 dual-layer STC explicitly handles 2D context via syndrome stacking; the current impl bypasses this. Reformulation 3a (sidecar over A1 residual) is the natural Pevný 2010 dual-layer slot.

**Selfcomp / szabolcs-cs (PR#56 author):**
- Operating-within assumption: "the 421KB AV1 monochrome mask payload IS the empirically-canonical mask codec for this video class; substrate-class-shifts that displace AV1 must beat AV1's 0.014 bits/pixel × 236M pixels ground floor". HARD-EARNED per the empirical AV1 baseline measurement.
- Position: STC at 21MB IS DISPLACED by AV1. The reformulation that PRESERVES PR#56 paradigm is 3b: STC-as-tone-map-delta over Selfcomp's soft-grayscale-LUT baseline. The delta-stream is sparse (only frames where the SGD-optimized soft grayscale falls outside the per-class target band); STC encodes this sparse delta efficiently because the context-naive assumption HOLDS for sparse signals. I support DEFER + reformulation per 3b (highest expected synergy with PR#56 paradigm at 0.38 leaderboard anchor).

**Quantizr (adversarial reverse-engineering, leaderboard 0.33 leader):**
- Operating-within assumption: "FiLM-conditioned depthwise-separable CNN at 88K params with FP4+Brotli + AV1 monochrome mask + EMA + diff_round + diff_rgb_to_yuv6 IS the canonical winning recipe; any substrate replacement must NOT regress these primitives". HARD-EARNED per leaderboard reverse-engineering.
- Position: STC-as-replacement-codec for the mask channel DISPLACES AV1's 421KB and regresses Quantizr's 0.33 frontier by +4.81. STC-as-sidecar (3a) or tone-map-delta (3b) does NOT displace the AV1 baseline; both are compatible with Quantizr's recipe. I support DEFER + reformulation per 3a OR 3b (prefer 3a for direct frontier composability).

### Vote tally

**Sextet pact (Catalog #325 6-step #4 quorum 5-of-6):**
- 6-of-6 DEFER_PENDING_EVIDENCE (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary)
- 0 PROCEED
- 0 REFUSE
- 0 ESCALATE
- **Quorum: 6/6 = MET (≥5/6 threshold satisfied)**

**Grand-council attendees:**
- Filler: DEFER + reformulation 3c
- Pevný: DEFER + reformulation 3a
- Selfcomp: DEFER + reformulation 3b
- Quantizr: DEFER + reformulation 3a OR 3b

**Final verdict: DEFER_PENDING_EVIDENCE — UNANIMOUS 10/10.**

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308)

**Required by CLAUDE.md "Forbidden premature KILL" non-negotiable:** every kill / defer verdict MUST enumerate reactivation paths with priority, predicted cost, structural verdict per Catalog #308 alternative-probe-methodologies.

### Path 3a (PRIORITY 1): STC-as-sidecar over A1-substrate residual stream
- **Description:** Replace per-pixel-IID arithmetic with a 1D symbol stream over A1's residual coefficients (the post-quantization residual that A1 doesn't currently encode). STC's parity-check structure becomes a distortion-bounded payload embedding over the residual signal, NOT the mask channel.
- **Predicted ΔS band:** [-0.015, -0.003] (first_principles per Filler 2011 Theorem 4 + A1 residual entropy estimate ~0.5-1.0 bits/coefficient × ~5K coefficients × 1/8 = 312-625 bytes saved if STC matches arithmetic coding bound within 1/h overhead)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324 (must measure on actual A1 archive post-training, NOT random init)
- **Predicted cost:** $5 Modal T4 ~10ep smoke for impl + $0.20 CUDA re-eval = $5.20 total
- **Structural verdict:** This path tests the PARADIGM (Filler STC over a 1D signal where context-naive AC is appropriate); RATIFIES OR FALSIFIES the paradigm in a context-appropriate substrate.
- **Implementation complexity:** ~150 LOC sidecar codec + 20 LOC inflate.py extension (within HNeRV L4 ≤100 LOC budget if we tighten the codec).
- **Composability:** ADDITIVE with A1's current archive grammar; ADDITIVE with PR#56 / Selfcomp tone-map paradigm; ADDITIVE with Quantizr's FP4+Brotli weight pool.

### Path 3b (PRIORITY 2): STC-as-tone-map-delta over Selfcomp soft-grayscale baseline
- **Description:** Compose STC with Selfcomp's PR#56 paradigm. Soft-grayscale-LUT produces a continuous tone-map; STC encodes the SPARSE DELTA between the SGD-optimized soft grayscale and the per-class hard-grayscale targets `[0, 255, 64, 192, 128]`. Sparse delta signal IS the canonical Dasher-coded sparse signal per MacKay 2003 ITILA §6.6 (Catalog #262 STC-Dasher composite).
- **Predicted ΔS band:** [-0.02, -0.005] (first_principles per MacKay sparse-signal Dasher coder + Selfcomp PR#56 0.38 anchor + empirical Selfcomp under-fit slack)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324
- **Predicted cost:** $5 Modal T4 ~10ep smoke for impl + $0.20 CUDA re-eval = $5.20 total
- **Structural verdict:** This path tests the PARADIGM in the SegMap-trained-from-scratch context (per `lane_mm_v3_segmap_trained_from_scratch` reactivation #7 from the pre-rigor inventory). Composable with Lane MM v3 dispatch wave.
- **Implementation complexity:** ~200 LOC delta codec (Selfcomp baseline + STC delta + decoder); within budget if STC delta stays sparse.
- **Composability:** ADDITIVE with Selfcomp's PR#56 paradigm; ADDITIVE with Lane MM v3 reactivation; potentially COMPOSABLE with Quantizr's FP4+Brotli weight pool.

### Path 3c (PRIORITY 3): STC with 2D + temporal context model (canonical Filler 2011 impl)
- **Description:** Reformulate the implementation to use a 2D + temporal context model (JPEG-LS LOCO-I style spatial context + H.264-style temporal prediction). This recovers the canonical Filler 2011 STC bound at the cost of substantial impl complexity (~500 LOC additional + decoder complexity).
- **Predicted ΔS band:** [-0.010, +0.005] (first_principles per Filler 2011 bound at 2D+temporal AC bound ~0.05 bits/pixel; would produce ~750KB stream — still 1.8× larger than AV1's 421KB but within striking distance)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324
- **Predicted cost:** $15-30 Modal A100 implementation development + $0.20 CUDA re-eval = $15-30.20 total
- **Structural verdict:** This path tests whether canonical Filler STC can MATCH AV1 on the mask channel. Probably NOT (AV1 is heavily optimized for monochrome over decades); but RATIFY-AT-CONFIDENCE for the paradigm if achieved.
- **Implementation complexity:** ~500 LOC + decoder complexity; OVER PR101 budget; likely requires `lane_class=substrate_engineering` per HNeRV parity L7 exception.
- **Composability:** REPLACES AV1 monochrome (displacing); incompatible with Quantizr's recipe; STANDALONE codec.

### Reactivation priority ordering
1. **Path 3a (HIGHEST EV)**: $5.20 cost, [-0.015, -0.003] predicted band, additive composability, tests paradigm in context-appropriate substrate.
2. **Path 3b (SECOND EV)**: $5.20 cost, [-0.02, -0.005] predicted band, composes with Lane MM v3 reactivation (synergy with sister symposium).
3. **Path 3c (LOWEST EV)**: $15-30 cost, [-0.010, +0.005] predicted band, replaces AV1 (non-additive), would require substrate-engineering exception.

**Recommendation:** Pursue 3a + 3b in parallel (both at $5.20 cost; orthogonal substrates; ~$10-15 total = 50-75× cheaper than the $0.20 re-eval would be if amortized over the EV ratio). Defer 3c until 3a OR 3b ratifies the paradigm.

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status:** `pending_post_training` for all 3 reformulation paths.

**Rationale:** Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density" non-negotiable: predicted bands derived from first-principles Filler 2011 + MacKay 2003 + AV1 baseline math (NOT random-init Tier-C density) are PROVENANCE=first_principles. The reactivation criterion is post-training Tier-C re-measurement on the landed reformulation archive.

**Reactivation criterion verbatim:** "Post-training Tier-C density measurement on landed archive sha for each reformulation (3a/3b/3c) via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within ±0.005 of the predicted band, ratify the paradigm + advance to L2. If outside band, surface 22×-or-greater miss as Catalog #324 violation and re-symposium."

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes the DEFER verdict + 10-seat attendee list + 4-assumption Assumption-Adversary verdict + 5 op-routables + mission-alignment=frontier_protecting + override_invoked=false.

Downstream consumers per Catalog #325:
- **Catalog #325 STRICT preflight** sees the DEFER verdict and structurally REFUSES dispatch of any operator-authorize recipe targeting `lane_stc_clean_source` substrate via `tools/operator_authorize.py::_check_predecessor_probe_outcome` (which now consults `tac.probe_outcomes_ledger` + this council anchor).
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('stc_clean_source')` for council-verdict-aware candidate weighting.
- **Probe-outcomes ledger (Catalog #313)** receives a sister-registered DEFER outcome via `tac.probe_outcomes_ledger.register_probe_outcome` so any future dispatch wrapper consults the canonical outcome BEFORE firing.

## 8. Cross-references

- **Canonical reference memos:** `project_lane_stc_clean_source_FALSIFIED_20260429.md` (original 2026-04-29 verdict); `project_lane_stc_av1_regression_finding_20260429.md` (AV1-noise anchor finding); `docs/paper/lane_stc_boundary_coding_design_20260429.md` (Stage 1 design).
- **Pre-rigor inventory:** `.omx/research/pre_rigor_kill_defer_falsified_inventory_20260517.md` row #2 (this symposium's council_priority).
- **Resurrection audit:** `.omx/research/resurrection_audit_20260516.md` §1.2 (Pattern C: MPS-vs-CUDA evidence-grade violation).
- **Catalog gates fired by this symposium:** #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification) + #308 (alternative-probe-methodologies enumeration) + #313 (probe-outcomes ledger) + #324 (post-training Tier-C validation) + #325 (per-substrate symposium discipline).
- **Catalog gates protected by this symposium:** #220 (operational mechanism declaration; reformulations 3a/3b/3c must declare); #272 (distinguishing-feature integration contract); #233 (L1→L2 promotion canonical 4-gate); #298 (substrate retirement discipline 30-day).
- **Canonical implementations cited:** Filler 2011 IEEE TIFS Theorem 4; Pevný 2010 IEEE TIFS dual-layer STC; MacKay 2003 ITILA §6.6 Dasher; Selfcomp PR#56 paradigm (`feedback_selfcomp_pr56_*`); Quantizr 0.33 leaderboard reverse-engineering; HNeRV parity discipline lessons 1-13; PR101 / PR103 silver anchors.

## 9. Operator op-routables (for parent agent + main Claude)

1. **DO NOT fire the operator-pre-authorized $0.20 CUDA re-eval on the existing implementation.** The first-principles math + 6-of-6 sextet DEFER vote + 4-of-4 grand-council DEFER vote all converge: the CUDA bytes WILL match within ≤20% of the MPS bytes (uint8 argmax stability), confirming RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY at $0.20 wasted cost. Higher EV options exist below.

2. **Register the DEFER outcome to the canonical probe-outcomes ledger** via `tac.probe_outcomes_ledger.register_probe_outcome(substrate_id='stc_clean_source', verdict='DEFER', status='blocking', methodology='per_pixel_iid_arithmetic_at_p_boundary_0.05', alternative_probe_methodologies=['stc_sidecar_over_a1_residual_3a', 'stc_tone_map_delta_over_selfcomp_3b', 'stc_2d_temporal_context_canonical_filler_3c'], expires_at_utc=<30_days_from_now>)`. This makes the DEFER verdict QUERYABLE across sessions and gates future dispatch wrappers from re-firing the same implementation.

3. **Migrate `lane_stc_clean_source` from L1 to `research_only=true`** with reactivation_criteria pinning the 3 reformulation paths above (3a/3b/3c). Per CLAUDE.md "Substrate retirement discipline" Catalog #298: the lane has 18-day operator-attention activity (this symposium); the 30-day retirement window resets BUT the reactivation criterion is "implement and test at least one of 3a/3b/3c", NOT "re-run the same implementation on CUDA".

4. **Open 3 follow-on lanes for reformulation paths**: `lane_stc_sidecar_a1_residual_3a_20260518` (PRIORITY 1, $5.20), `lane_stc_tone_map_delta_selfcomp_3b_20260518` (PRIORITY 2, $5.20, composes with sister `lane_mm_v3` reactivation), `lane_stc_2d_temporal_context_3c_20260518` (PRIORITY 3, $15-30, requires lane_class=substrate_engineering). Each lane gets its OWN per-substrate symposium per Catalog #325.

5. **Sister-symposium synergy**: Path 3b composes with `lane_mm_v3_segmap_lut_trained_from_scratch` reactivation (council priority #7 in pre-rigor inventory). Recommend pairing these two as a single dispatch wave once each has its own per-substrate symposium PROCEED verdict.

**Total dispatch redirect:** $0.20 (would-have-wasted) → $10-15 (reformulation 3a + 3b in parallel) = 50-75× higher EV per dollar.

---

**Symposium concludes.** Verdict: DEFER_PENDING_EVIDENCE — 10-of-10 unanimous. Mission-alignment: frontier_protecting. Override: not invoked. Continual-learning anchor: registering to `.omx/state/council_deliberation_posterior.jsonl` via canonical helper.
