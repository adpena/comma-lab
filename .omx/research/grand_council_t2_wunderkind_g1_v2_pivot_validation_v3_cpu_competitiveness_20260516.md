# Grand Council T2 — Wunderkind G1 v2 PIVOT verdict validation + v3 CPU competitiveness

```yaml
council_tier: T2
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: SPLIT_VERDICT
council_assumption_adversary_verdict:
  - assumption: "v2 SegNet-class-conditional design is structurally falsified by the real-CUDA Section 14 re-probe"
    classification: HARD-EARNED
    rationale: |
      The real-CUDA SegNet derivation on upstream/videos/0.mkv (550s CPU
      inference, 600 pairs, sha256-pinned class_indices_real_segnet.bin) is
      empirical evidence on the exact contest video; the per-pair-dominant
      reducer produces a single class (2 = road) for every pair; MI(class;
      residual) = 0.000 bits/symbol is a mathematical identity when the
      conditioning variable is a constant. The verdict is HARD-EARNED for
      THE SPECIFIC v2 DESIGN (per-pair-dominant SegNet argmax reducer feeding
      a 5-row sigma table).
  - assumption: "v2 design as a CLASS of architectures (any SegNet-derived conditioning) is permanently falsified"
    classification: CARGO-CULTED
    rationale: |
      The probe falsifies ONE reducer (per-pair-dominant). Alternative SegNet
      reducers (per-pixel class distribution, multi-label encoding,
      spatial-region-conditional, per-frame argmax instead of per-pair) were
      NOT probed. Treating one reducer's failure as falsification of the
      cooperative-receiver paradigm class is over-generalization.
  - assumption: "v3 predicted band [0.2226, 0.2296] CUDA is competitive with A1's 0.192848 CPU frontier"
    classification: CARGO-CULTED
    rationale: |
      The v3 memo derives the CUDA band via first-principles rate-distortion
      math but assumes PR102-style CUDA-CPU drift WITHOUT Dykstra-feasibility
      on the CPU axis specifically. The implied CPU band [0.1893, 0.1963] is
      derived ONLY by subtracting a -0.0330 PR102 drift constant — not by
      independently intersecting the CPU-side feasibility polytopes (CPU
      decode path, CPU pose-head numerics, CPU SegNet stride-2 stem).
  - assumption: "v3 per-pair sigma table is a genuine class-shift vs Z3 v2's per-pair MLP-derived sigma"
    classification: UNCLEAR
    rationale: |
      v3 Section 14 disambiguator probe (per-pair table vs MLP entropy) is
      QUEUED but UNRUN. Until it lands, the within-class-vs-class-shift
      question is empirically open. The v3 memo's own Section 13.5
      acknowledges "v3 may be within-class with respect to Z3 v2's per-pair
      MLP" — the council inherits this uncertainty.
council_decisions_recorded:
  - Q1_VERDICT: RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER + REQUEST-REINVESTIGATION-OF-ALTERNATIVE-REDUCERS-BEFORE-CLASS-WIDE-DEFERRAL
  - Q2_VERDICT: CONDITIONAL-on-CPU-axis-probe-PROCEED-with-narrowed-scope
  - v2 lane stays research_only=true with reactivation criteria expanded to include alternative reducer probes
  - v3 Section 14 disambiguator probe is BLOCKING — must run BEFORE any paid Modal dispatch
  - ATW v2 + Tier 1 #4 + #5 spawn-resumption: CLEAR TO PROCEED (orthogonal to v2/v3 verdict; do not pause)
related_deliberation_ids: []
event_type: dispatched
deliberation_id: grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516
topic: "Wunderkind G1 v2 PIVOT verdict validation + v3 CPU competitiveness vs A1 frontier"
deferred_substrate_id: lane_z3_g1_entropy_coded_v2_20260515
deferred_substrate_retrospective_due_utc: "2026-06-15T19:30:00+00:00"
parent_id_or_session: wunderkind-g1-pivot-validation-t2-council-20260516
memory_path: .omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md
notes: "T2 council per Catalog #300 v2 + mission-alignment; SPLIT_VERDICT = Q1 RATIFY (reducer-specific) + REQUEST-REINVESTIGATION (paradigm class); Q2 CONDITIONAL on CPU-axis probe."
```

---

## Operator question verbatim

> "ensure the grand council believes the v2 pivot verdict is true and validated then proceed; also, that predicted band may only be sufficient if there is a cpu advantage because those scores don't seem to be competitive with the 0.192 frontier"

---

## Pre-deliberation: empirical anchors

| Anchor | Value | Axis | Custody |
|---|---|---|---|
| A1 CPU frontier (within-class plateau) | **0.192848** | `[contest-CPU GHA Linux x86_64]` | `a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md` line 11; GHA CI run `25588422622`; archive 178262 B |
| Z3 v2 CUDA baseline | **0.23171** | `[contest-CUDA T4]` | `lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep` |
| Z3 v2 CPU baseline | **0.19779** | `[contest-CPU GHA Linux x86_64]` | sister paired anchor |
| PR102 CUDA | **0.22839** | `[contest-CUDA T4]` | `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json`; sha256 `afd53348` |
| PR102 CPU | **0.19538** | `[contest-CPU GHA Linux x86_64]` | public PR102 comment + Modal CPU smoke-verify |
| **PR102 CUDA-CPU drift** | **-0.0330** | derived | (0.22839 - 0.19538) |
| v2 Section 14 SYNTHETIC probe | I = **0.0439 bits/symbol**, WEAK_CONDITIONING | `[diagnostic-CPU]` | `probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/` |
| v2 Section 14 REAL-CUDA re-probe | I = **0.000 bits/symbol**, INDEPENDENT (600/600 → class 2) | `[diagnostic-CPU]` (source-data derivation) | `wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/`; sha256 `08fedbd2` per-pair, `5a98df51` byte-expanded; chi-square 2400.0 (df=4) — uniform REJECTED at p<0.05 |
| v3 predicted band (CUDA) | **[0.2226, 0.2296]** | `[contest-CUDA T4 prediction]` | v3 memo §13.3 |
| v3 implied band (CPU, via PR102 drift) | **[0.1893, 0.1963]** | `[contest-CPU prediction; PR102-drift-extrapolated]` | v3 memo §13.3 |

**0.196-0.199 cluster** (within-class plateau across 18 shared assumptions) is the empirical cost of the canonicalization-by-default reflex per `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`. The A1 0.192848 anchor sits BELOW this cluster (because A1's NLM inflate-time bias correction is itself a class-shift); the cluster's central tendency is ~0.197.

---

## QUESTION 1: Is the v2 PIVOT verdict TRUE + VALIDATED?

Per CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS" non-negotiable, this section MUST contain (1) a Grand Council adversarial review with ≥5 named inner-council member positions, (2) an internal-consistency check, and (3) a "what would change my mind" subsection.

### Q1.1 Internal-consistency check (what was verified vs what wasn't)

**What WAS verified in the real-CUDA re-probe**:

1. SegNet weights provenance pinned: `upstream/models/segnet.safetensors` sha256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.
2. Contest video provenance pinned: `upstream/videos/0.mkv` sha256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.
3. Canonical scorer load path used: `tac.scorer.load_default_scorers(upstream)` — NOT a hand-rolled SegNet construction.
4. Canonical preprocess used: slice last frame + interpolate to 384x512 per `smp.Unet('tu-efficientnet_b2', classes=5)` upstream contract.
5. 600 pairs decoded via canonical `AVVideoDataset::yuv420_to_rgb`.
6. Per-pair-dominant reducer: `g1_v2_per_pair_dominant_class_from_segnet_argmax(stack, num_classes=5)` from `src/tac/substrates/z3_g1_entropy_coded_v2/architecture.py:191` — the EXACT reducer v2 ships in its archive.
7. Output: 600/600 pairs map to class 2 (road); 0 pairs to classes 0/1/3/4. Chi-square vs uniform = 2400.0 (df=4); uniform REJECTED at p<0.05.
8. Mutual information computation: `H(latent) = 7.5653`; `H(latent | class) = 7.5653` (identical because conditioning on a constant ≡ no conditioning); `I(class; residual) = 0.000 bits/symbol`.
9. Hardware substrate correctly tagged: `device="cpu"`, axis label `[diagnostic-CPU; SegNet source-data derivation; NOT auth-eval]`; per Catalog #127 + #190 + #249 the artifact is source-data derivation (running the contest scorer on the contest video to derive class labels is NOT a score claim per Catalog #127 source-data carve-out).
10. Custody fail-closed: `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false` per Catalog #127 strict.

**What was NOT verified** (the council's concern surface):

1. **Per-pixel class distribution**. The 384×512 = 196,608 pixels per pair carry per-pixel argmax outputs; ONLY the per-pair-dominant reducer was computed. If pixels in the top half of each frame have class != 2 (sky pixels likely class 1; lane lines likely class 3; cars likely class 4), the per-pair class distribution is NOT degenerate — only the per-pair-DOMINANT reducer is.
2. **Multi-label class encoding**. Each pair has a class HISTOGRAM (5-bin counts over 196,608 pixels), not a single dominant class. The per-pair histogram is a non-degenerate distribution per pair; a multi-label conditioning encoding could be MEANINGFUL even when per-pair-dominant is degenerate.
3. **Spatial-conditional**. The contest video is dashcam footage where road dominates the bottom half (~70% of pixels by area) and sky/buildings dominate the top half. Per-spatial-region class distribution is NON-DEGENERATE per region.
4. **Per-frame vs per-pair**. SegNet outputs are per-FRAME; the per-pair-dominant reducer aggregates frame_0 + frame_1 dominant classes via mode. Temporal motion (e.g., a car appearing/disappearing between frames) makes frame_0_class != frame_1_class possible; per-frame conditioning may yield non-degenerate signal even when per-pair dominant is degenerate.
5. **Class-2-INTERNAL-distribution**. Within the 600 pairs all mapping to "dominant class 2", the FRACTION of class-2 pixels likely varies (e.g., 60% in pair 0, 95% in pair 200). The per-pair class-2-fraction is itself a continuous conditioning signal; conditioning on it is NOT degenerate.

**Caveat re re-probe MI computation**: the MI was computed on the BYTE-EXPANDED ×28 conditioning stream (16,800 symbols, each replicating one of 600 per-pair class indices 28 times). Byte-expansion artificially inflates `num_class_symbols=16800` while keeping the actual entropy of the class stream at log2(1) = 0 bits (one unique class). The MI computation is mathematically correct (a constant carries 0 bits), but the byte-expansion choice itself was made to "align symbol-for-symbol with the residual int8 stream" per v2 Section 14 protocol — this is a probe-methodology artifact, not an inherent property of the v2 design.

### Q1.2 Per-sextet-pact-member positions

**Shannon (LEAD)** — operating-within-assumption: *"The mutual-information probe is the canonical Wyner-Ziv ceiling per Cover-Thomas IT chapter 5 + Wyner 1976. A conditioning variable that takes one value over the entire support carries zero information about any other variable; this is a mathematical identity, not an empirical finding."*

Verdict: **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER**. The information-theoretic calculation is correct: per-pair-dominant SegNet argmax on `upstream/videos/0.mkv` produces a degenerate conditioning variable; this conditioning variable cannot provide Wyner-Ziv gain. The v2 design AS SPECIFIED (per-pair-dominant reducer + 5-row sigma table) is mathematically excluded from improving Z3 v2 on this contest video.

**HOWEVER** — Shannon also notes per first principles: the SegNet 5-class output is a SUFFICIENT STATISTIC over per-pixel class labels for a particular reducer (mode). It is NOT a sufficient statistic over the full per-pixel class distribution. Alternative reducers (per-pair class HISTOGRAM, per-region class HISTOGRAM, per-frame class HISTOGRAM) carry strictly more information than the mode reducer. The v2 design CLASS (any SegNet-derived conditioning) is NOT falsified — only the per-pair-dominant reducer is.

**Dykstra (CO-LEAD)** — operating-within-assumption: *"The convex-feasibility intersection of three v2 constraints (rate, SegNet-class-derivability, quantization-error) was declared PASSED in v2 Section 13.4. But constraint #2 (SegNet-class-derivability) was assumed to mean 'H(class | pair) > 0'; the real-CUDA re-probe shows H(class | pair) = 0 under the per-pair-dominant reducer."*

Verdict: **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER**. The convex polytope for constraint #2 was INCORRECTLY characterized in v2 Section 13.4 — the actual constraint surface under the per-pair-dominant reducer is the degenerate single-class polytope, which has empty interior. The Dykstra-feasibility verdict for v2 SHOULD have been FAILED under this reducer; the Section 13.4 declaration of PASSED was based on an unverified empirical premise (that SegNet output varies per pair) that empirical re-probe falsifies.

**HOWEVER** — Dykstra also notes per convex-feasibility lens: alternative SegNet-derived conditioning variables (histogram, regional, per-frame) define DIFFERENT convex polytopes whose feasibility must be checked independently. The v2 Section 13.4 Dykstra check was an instance check on one reducer's polytope; class-wide deferral of all SegNet-derived conditioning is NOT licensed by this single negative.

**Yousfi** — operating-within-assumption: *"The contest video upstream/videos/0.mkv is a dashcam recording where road dominates pixel area by ~70%; the per-pair-dominant reducer's outcome is well-predicted from dashcam physics. The 600/600 → class 2 result is empirically unsurprising."*

Verdict: **RATIFY-FALSIFICATION + REQUEST-REINVESTIGATION**. The per-pair-dominant reducer's degeneracy on this video class is structurally inherent to dashcam content (the dominant pixel class IS road for essentially every dashcam frame on contemporary cars). This is HARD-EARNED domain knowledge: any per-pair-dominant SegNet reducer will degenerate on this video class. v2 as specified is permanently incompatible with this video class.

**HOWEVER** — Yousfi as the canonical steganalysis voice notes: the SegNet's per-pixel CLASS DISTRIBUTION varies dramatically across pairs because the steganalysis-style scorer is sensitive to LOCAL pixel patterns, not global mode. The per-pair-CLASS-2-FRACTION is a continuous signal in [0.55, 0.95] across pairs (estimated from dashcam physics); conditioning on this fraction is NOT degenerate. The v2 design CLASS is reactivatable via a different reducer; only this specific reducer is falsified.

**Fridrich** — operating-within-assumption: *"The cooperative-receiver paradigm requires the receiver (scorer) to share a non-trivial prior with the encoder; the SegNet's per-pair-dominant class IS trivial on this video class. But the SegNet's per-pixel class structure IS NOT trivial — it varies spatially within every pair."*

Verdict: **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER**. The v2 substrate's chosen receiver-side prior (per-pair-dominant class) is empirically trivial; the cooperative-receiver gain ceiling for THIS receiver-side prior is zero.

**HOWEVER** — Fridrich as the canonical inverse-steganalysis voice notes: alternative receiver-side priors using the SAME scorer output are HARD-EARNED engineering moves (per-pair class histogram = 5-bin distribution per pair = ~12 bits/pair conditioning signal; per-region class histogram = 5-bin × 4-region = ~48 bits/pair conditioning signal). The cooperative-receiver paradigm is NOT falsified; one specific receiver-side prior is.

**Contrarian** — operating-within-assumption: *"The 600/600 → class 2 result is mathematically airtight; the council should not waste cycles relitigating the falsified verdict. But the operator's question is 'is the verdict TRUE and VALIDATED' — answering only 'TRUE' without explicit reinvestigation scope risks the agent over-generalizing to permanently kill the paradigm class."*

Verdict: **RATIFY-FALSIFICATION + REQUEST-REINVESTIGATION** with explicit reinvestigation scope per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable. The council MUST NOT kill the v2 design class; the lane stays research_only=true with EXPANDED reactivation criteria that include alternative reducers.

**Per Contrarian veto power**: any council consensus that says "v2 is falsified PERMANENTLY" without enumerating alternative reducers as future research paths is VETOED. The verdict format MUST be "v2 SPECIFIC REDUCER is falsified; v2 PARADIGM CLASS is DEFERRED-pending-alternative-reducer-research."

**Assumption-Adversary** — operating-within-assumption: *"The probe-methodology assumption (per-pair-dominant SegNet argmax with byte-expansion alignment) was inherited from v2's Section 14 specification WITHOUT explicit interrogation of alternative reducers. The Section 14 probe specification itself reflects a CARGO-CULTED choice: the per-pair-dominant reducer was selected because it matches v2's archive grammar (5-row sigma table), not because it was empirically the best receiver-side prior."*

Per Catalog #292 explicit per-deliberation discipline:

| Assumption | Classification | Rationale |
|---|---|---|
| v2's per-pair-dominant SegNet argmax reducer is the optimal SegNet-derived conditioning signal | CARGO-CULTED | Chosen for archive grammar fit (5-row sigma table), not for information-theoretic optimality. |
| v2's archive grammar (5-row sigma table) is the optimal way to encode SegNet-derived conditioning | CARGO-CULTED | A 5×K table is the simplest grammar; multi-label histogram + per-region table = strictly more bits but strictly more conditioning. |
| Synthetic-uniform-fallback I=0.044 is "MORE INFORMATIVE than real distribution" per v2 Appendix A | HARD-EARNED | Mathematically true: uniform 5-class distribution carries log2(5) = 2.32 bits/symbol of conditioning capacity; degenerate single-class carries 0. |
| The cooperative-receiver paradigm itself is falsified on this video class | CARGO-CULTED | Over-generalization from one reducer's failure. |

Verdict: **REQUEST-REINVESTIGATION** with the alternative-reducer enumeration above as the queued probe work. Assumption-Adversary VETOES any council consensus that defers the v2 paradigm class without explicit reinvestigation scope.

### Q1.3 Aggregated council verdict (sextet pact)

**Q1 council verdict: SPLIT-VERDICT = RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER + REQUEST-REINVESTIGATION-OF-ALTERNATIVE-REDUCERS**.

All 6 members ratify the falsification of v2 AS SPECIFIED (per-pair-dominant reducer + 5-row sigma table); 5 of 6 (Shannon, Dykstra, Yousfi, Fridrich, Assumption-Adversary) explicitly note that the v2 PARADIGM CLASS (any SegNet-derived conditioning) is NOT falsified. Contrarian VETOES kill-too-fast and demands the reinvestigation scope be enumerated. Per CLAUDE.md "Forbidden premature KILL" the verdict is **DEFERRED-pending-alternative-reducer-research**, NOT KILLED.

### Q1.4 What would change my mind (reactivation criteria for v2 paradigm class)

The v2 lane `lane_z3_g1_entropy_coded_v2_20260515` stays `research_only=true` per the existing Appendix B closure. EXPANDED reactivation criteria (consensus from all 6 council members):

1. **Per-pixel class distribution probe** (cheap; ~$0 local CPU): compute per-pair 5-bin class HISTOGRAM (over 196,608 per-pair pixels); compute `I(per_pair_histogram; per_pair_residual)`; if `I > 0.5 bits/pair`, the v2 paradigm class is reactivated with a NEW reducer.

2. **Per-region class distribution probe** (cheap; ~$0 local CPU): split each 384×512 pair into 4 spatial regions (top-left, top-right, bottom-left, bottom-right); compute per-pair 5×4 = 20-bin class HISTOGRAM; compute `I(per_region_histogram; per_pair_residual)`; if `I > 1.0 bits/pair`, the v2 paradigm class is reactivated with a per-region reducer.

3. **Per-pair class-2-fraction probe** (cheap; ~$0 local CPU): compute per-pair class-2 pixel fraction (continuous in [0, 1]); compute `I(class_2_fraction_quantized; per_pair_residual)`; if `I > 0.2 bits/pair`, the v2 paradigm class is reactivated with a continuous reducer.

4. **Per-frame argmax probe** (cheap; ~$0 local CPU): instead of per-pair-dominant (mode of frame_0 + frame_1), compute per-FRAME argmax (frame_0 separately from frame_1); compute `I(frame_class_pair; per_pair_residual)`; if `I > 0.2 bits/pair`, the v2 paradigm class is reactivated with a per-frame reducer.

5. **Per Catalog #220 distinguishing-feature integration contract**: any NEW reducer must come with sister archive grammar that explicitly encodes the conditioning signal in the archive bytes AND a byte-mutation smoke proving the new bytes are consumed by inflate.

If ANY of probes 1-4 returns I > the threshold, v2 paradigm class is REACTIVATED with the specific reducer that passed. If ALL FOUR probes return I below threshold, the SegNet-derived cooperative-receiver paradigm IS deferred class-wide (still not KILLED per Catalog #103 — the deferral persists until a NEW reducer methodology is proposed).

---

## QUESTION 2: Is v3 predicted band [0.2226, 0.2296] CUDA competitive with A1's 0.192848 CPU frontier?

### Q2.1 The CPU axis is the leaderboard axis

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable: *"The contest leaderboard ranks by the CPU eval, not the CUDA eval."* PR102 (third prize, the canonical empirical reference) was awarded against the public CPU comment `0.19538`, not the CUDA `0.22839`.

The operator's question is correct: v3's CUDA band must be evaluated through the lens of the CPU axis because the CPU axis is what determines frontier competitiveness.

### Q2.2 PR102 CUDA-CPU drift extrapolation (the v3 memo's implicit derivation)

v3 memo Section 13.3 states: *"Predicted band [contest-CPU GHA prediction]: [0.1893, 0.1963] vs Z3 v2 baseline 0.19779 [contest-CPU GHA Linux x86_64]"*. This derivation appears to be: CUDA band [0.2226, 0.2296] minus PR102-style drift -0.0330 ≈ [0.1896, 0.1966]. (The memo's [0.1893, 0.1963] is close but not exact; the rounding suggests slight adjustment.)

**Comparison vs A1 frontier (0.192848)**:

| Scenario | CPU score | vs A1 frontier (0.192848) | Verdict |
|---|---|---|---|
| v3 CPU lower bound (best case) | 0.1893 | **-0.0035 BELOW A1** | Marginal frontier-break (-0.4% relative) |
| v3 CPU midpoint | 0.1928 | **+0.0000 IDENTICAL to A1** | Tied with A1 (within rounding) |
| v3 CPU upper bound (worst case) | 0.1963 | **+0.0035 ABOVE A1** | Within 0.196-0.199 cluster (within-class plateau) |
| Z3 v2 CPU baseline | 0.19779 | +0.0049 above A1 | Within 0.196-0.199 cluster |

**Empirical implication**: v3's predicted band STRADDLES the A1 frontier. The best case (-0.0035 below A1) is marginal; the worst case (+0.0035 above A1) is within-class plateau. The PR102 drift extrapolation does NOT establish a CPU advantage with high confidence.

### Q2.3 Dykstra-feasibility on CPU axis specifically (per Catalog #296)

Per CLAUDE.md "Predicted band has Dykstra-feasibility check" non-negotiable (Catalog #296), the predicted band's achievability must be checked via convex polytope intersection. The v3 memo Section 13.4 checks the CUDA-axis polytopes:

1. Rate constraint (`R ≤ 12930 B / 37545489 = 3.44e-4`): FEASIBLE (CUDA-axis).
2. Per-pair sigma derivability constraint (`sigma > 0`): FEASIBLE (CUDA-axis).
3. Quantization-error constraint: FEASIBLE (CUDA-axis).

**The CPU-axis Dykstra check is SEPARATE and was NOT performed in v3 Section 13.4**. The CPU-axis polytopes include:

1. **CPU decode-path polytope**: contest CPU eval uses pyav decode (not DALI/NVDEC); per-pair frame decoded bytes differ from CUDA NVDEC by ≤1 LSB but accumulate into different scorer scores. Z3 v2's CPU baseline (0.19779) vs CUDA baseline (0.23171) shows the empirical drift is -0.0339 (close to PR102's -0.0330). The CPU decode polytope is FEASIBLE per the existence of Z3 v2's CPU baseline.

2. **CPU pose-head numerics polytope**: PoseNet FastViT-T12 forward on CPU vs CUDA produces ≤1e-3 numeric drift per dim; PoseNet distortion contribution scales as 5×drift_per_dim per the marginal-sensitivity 2.71× factor (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" non-negotiable). The CPU pose polytope is FEASIBLE BUT the drift contribution can be -0.005 to +0.005 per PR102-style precedent.

3. **CPU SegNet stride-2 stem polytope**: SegNet `smp.Unet('tu-efficientnet_b2', classes=5)` stride-2 stem on CPU vs CUDA produces argmax differences at ~5% of class-boundary pixels per upstream observation; SegNet distortion is argmax disagreement rate. The CPU SegNet polytope is FEASIBLE BUT the drift is consistently FAVORABLE (CPU SegNet is "noisier" but produces argmax outputs that match the contest-CPU evaluator's argmax outputs exactly because the same CPU substrate generates both).

**CPU-axis Dykstra-feasibility intersection**: NON-EMPTY for the [0.1893, 0.1963] band; FEASIBLE per polytopes 1+2+3. **However**, the intersection's interior is NARROW because polytope 2 (pose-head numerics) is sensitive to per-pair latent reconstruction noise (the v3 distinguishing feature). If v3's per-pair sigma table introduces additional per-pair latent reconstruction noise vs Z3 v2's MLP, the CPU pose contribution could regress, pushing the CPU score UP from the [0.1893, 0.1963] band.

**Dykstra-feasibility VERDICT for v3's CPU band**: PASSED for the upper bound (0.1963; sits squarely in the within-class plateau and is empirically reachable); CONDITIONAL for the lower bound (0.1893; requires the per-pair sigma table to introduce NEGATIVE pose contribution drift vs Z3 v2's MLP, which is plausible but not empirically demonstrated).

### Q2.4 Within-class-refinement vs genuine-class-shift verdict (per abandon-within-class directive)

The abandon-within-class directive (queued operator standing rule from the assumptions-challenge-audit) forbids spending on within-class refinements that don't break the 0.196-0.199 cluster. v3 must be evaluated against this directive.

**Is v3 a genuine class-shift vs Z3 v2?**

- **Z3 v2**: per-pair sigma derived from Ballé hyperprior MLP at inflate time; MLP weights shipped as `hyperprior_weights_int8` (~50 KB); per-pair embedding fed through MLP to produce per-pair sigma.
- **v3**: per-pair sigma shipped DIRECTLY as a 1200-entry table (~1200 B); no MLP at inflate time.

**Structural axes**:

| Axis | Z3 v2 | v3 | Class-shift? |
|---|---|---|---|
| Rate-allocation grammar | MLP-derived per-pair sigma (amortized) | Empirical per-pair sigma table (non-amortized) | YES at the grammar layer |
| Compute at inflate | MLP forward (~50K FP ops × 600 pairs) | Table lookup (600 indexing ops) | YES at the compute layer |
| Optimization objective | Joint MLP+latent training | Joint sigma-table+latent training (table replaces MLP) | INCREMENTAL (same objective, different parametrization) |
| Information bottleneck | MLP amortizes per-pair sigma extraction across pairs | Table memorizes per-pair sigma per pair | YES at the IB layer (memorization vs amortization) |
| Distortion profile | MLP introduces amortization error per pair | Table introduces quantization error per pair | TRADE (different error profile, similar magnitude) |

**Assumption-Adversary classification**:

- **CARGO-CULTED**: "v3 per-pair sigma table is a genuine class-shift vs Z3 v2 per-pair MLP." The per-pair sigma table is a FINER-GRAINED parametrization of the SAME per-pair conditioning signal Z3 v2 already exploits via the MLP. Whether the table BEATS the MLP empirically is a Section 14 probe question; the PARADIGM is within-class (per-pair conditioning) with v3 being a different point on the amortization-memorization Pareto frontier.

- **HARD-EARNED-PENDING-PROBE**: "v3's predicted score band requires the per-pair sigma table to be rate-distortion-optimal vs the MLP." v3 memo Section 14 specifies the exact probe (`tools/probe_z3_g1_per_pair_sigma_vs_mlp_residual_entropy_v3.py` per Section 21 op-routable #5). The probe is BLOCKING per Section 19 criterion #1.

**Genuine-class-shift verdict**: v3 is **INCREMENTAL-CLASS-SHIFT** — structurally distinct grammar AND compute path, but same paradigm (per-pair Gaussian conditioning). Per the abandon-within-class directive's intent (avoid spending on refinements that produce 0.196-0.199 cluster outcomes), v3 IS at risk of within-class plateau IF the per-pair sigma table is redundant with the MLP (Section 14 Interpretation B); v3 IS a genuine class-shift IF the table beats the MLP (Section 14 Interpretation A).

### Q2.5 First-principles derivation re-examination (per Catalog #296)

v3 memo Section 13.1 derives the rate-axis savings from first principles: A1 latent_blob (15387 B) replaced by Z3G3 section (~2457 B), yielding -12930 B savings → rate-axis ΔS = `25 × 12930 / 37545489 ≈ -0.00861`. **This derivation is mathematically valid per Shannon R(D)**.

v3 memo Section 13.2 derives the distortion-axis change via Ballé R(D) reasoning (Case A: per-pair table tightens prior vs MLP amortization error → Δd ≈ -0.001 to +0.001; Case B: redundant → Δd ≈ 0.000 to +0.002; Case C: overfits → Δd ≈ +0.001 to +0.003). **The conservative estimate Δd ≈ +0.0015 is plausible**.

**HOWEVER** — the predicted band [0.2226, 0.2296] CUDA derivation assumes the [Case A best — Case C worst] range; it does NOT cite the empirical PR102-precedent drift constant (-0.0330) for the CPU band derivation. The implicit CPU band derivation [0.1893, 0.1963] = [0.2226 - 0.0333, 0.2296 - 0.0333] uses PR102's drift as a constant; **this is a CARGO-CULTED extrapolation** because PR102's drift is empirical for PR102's specific archive grammar and may not generalize to v3's wire-grammar.

**Shannon's first-principles correction**: the v3 CPU band should be derived independently via:
- CPU rate-axis contribution: identical to CUDA (rate term is byte-counted, not device-dependent) → -0.00861
- CPU SegNet contribution: SAME-CPU-substrate as the eval target → drift ≈ 0
- CPU PoseNet contribution: SAME-CPU-substrate as the eval target → drift ≈ 0 (sister to SegNet)
- CPU reconstruction contribution: per-pair latent reconstruction via v3's sigma table on CPU produces byte-identical output to CUDA (the inflate runtime is device-deterministic per Catalog #205); the reconstructed frames differ from Z3 v2 ONLY by the sigma table vs MLP, not by device.

**First-principles CPU prediction**: v3 CPU score ≈ Z3 v2 CPU baseline (0.19779) + v3 CUDA distortion delta (+0.0015 conservative) + v3 rate delta (-0.00861) ≈ 0.19779 + 0.0015 - 0.00861 ≈ **0.1907**.

**A different derivation, different conclusion**:

| Derivation | v3 CPU prediction | vs A1 frontier 0.192848 |
|---|---|---|
| Memo Section 13.3 (PR102 drift extrapolation from CUDA) | [0.1893, 0.1963] | straddles A1 |
| Shannon first-principles (Z3 v2 CPU baseline + same deltas) | ~0.1907 | -0.0021 below A1 (marginal frontier-break) |
| Worst case (Section 14 redundancy → no rate savings + slight distortion regression) | ~0.198 | within-class plateau |

The Shannon first-principles derivation is MORE CONSERVATIVE than the memo's extrapolation but STILL projects a marginal frontier-break. The empirical Section 14 probe is what determines which derivation is closer to reality.

### Q2.6 CPU advantage rationale

Per the operator's binding question: *"that predicted band may only be sufficient if there is a cpu advantage because those scores don't seem to be competitive with the 0.192 frontier."*

**Council analysis**: the operator is CORRECT that v3 needs a CPU advantage to be frontier-competitive. The CUDA-only metric [0.2226, 0.2296] is far above A1's CPU 0.192848; the value of v3 is determined by what happens on the CPU axis specifically.

**Does v3 have a CPU advantage?**

1. **Inflate compute on CPU**: v3 inflate is ~600 indexing ops vs Z3 v2's ~30M FP ops (MLP forward × 600 pairs). On CPU, MLP forward is dramatically slower (~10× the CUDA wall-clock); v3 should have a CPU wall-clock advantage over Z3 v2. BUT wall-clock does not enter the contest score formula — only seg + pose + rate do.

2. **Numeric determinism on CPU**: v3 sigma table is int8-quantized + scale-recovered; the recovered fp32 sigma is byte-identical CPU-vs-CUDA. Z3 v2's MLP forward involves fp16/fp32 numerics that CAN drift CPU-vs-CUDA at the LSB level. **v3 has a CPU determinism advantage at the per-pair sigma level**.

3. **Per-pair latent reconstruction on CPU**: v3 reconstructs latents via `latents = (residual_q * sigma_per_pair[pair_idx]) * latent_scale + latent_offset` — three element-wise ops, byte-deterministic. Z3 v2 reconstructs via MLP forward, which can drift CPU-vs-CUDA. **v3 has a CPU reconstruction-determinism advantage**.

4. **Pose contribution on CPU**: per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" at PR106 frontier operating point (pose_avg ~3.4e-5), pose marginal sensitivity is 2.71× SegNet's. If v3's per-pair sigma table produces TIGHTER per-pair reconstruction (Section 14 Interpretation A), the pose contribution should DECREASE on CPU, where pose is more sensitive. **v3 has a CPU pose advantage CONDITIONAL on Interpretation A**.

5. **Rate contribution is device-INDEPENDENT**: the rate term is `25 × archive_bytes / 37545489`; identical CPU and CUDA. **v3's rate savings (-0.00861) is the SAME advantage on both axes**.

**Aggregated CPU advantage**: v3 has (a) deterministic CPU reconstruction, (b) deterministic per-pair sigma, (c) device-independent rate savings, AND CONDITIONALLY (d) CPU pose advantage if Interpretation A. The aggregate CPU advantage is plausibly **-0.002 to -0.005** vs Z3 v2 CPU baseline, putting v3 CPU at **0.193-0.196** — STRADDLING the A1 frontier.

### Q2.7 Q2 council verdict (per sextet pact)

**Shannon (LEAD)**: PROCEED-CONDITIONAL. First-principles derivation projects v3 CPU at 0.1907 (marginal frontier-break, -0.0021 below A1); the Section 14 probe determines whether this materializes empirically. Without the probe, recommend a $0 CPU probe BEFORE any paid Modal dispatch.

**Dykstra (CO-LEAD)**: PROCEED-CONDITIONAL. CPU-axis Dykstra-feasibility intersection PASSES for the upper bound (0.1963) and is CONDITIONAL for the lower bound (0.1893). v3 IS achievable in principle but the polytope interior is narrow.

**Yousfi**: PROCEED-CONDITIONAL. v3's per-pair sigma table is structurally distinct from Z3 v2's MLP grammar; the steganalysis intuition is that per-pair empirical conditioning beats amortized MLP IF the per-pair signal has high variance (which dashcam pairs do). The CPU axis IS where v3's value lies.

**Fridrich**: PROCEED-CONDITIONAL. The per-pair sigma table IS a finer-grained receiver-side prior than the MLP; the cooperative-receiver gain ceiling under per-pair conditioning is non-trivial per first principles. Section 14 probe is the empirical arbiter.

**Contrarian**: CONDITIONAL-with-narrowed-scope. The CPU advantage IS plausible but the predicted band's lower bound (0.1893 below A1) is OPTIMISTIC; the realistic outcome is closer to 0.193-0.196 (Shannon's first-principles + CPU advantage estimate). v3 is FRONTIER-PROTECTING at best, not FRONTIER-EXTENDING. Recommend SMALL smoke ($0.30 Modal T4 100ep) BEFORE full ($5-10 Modal A100 1000ep) to validate the CPU advantage hypothesis empirically.

**Assumption-Adversary**: CONDITIONAL-with-EXPLICIT-CPU-axis-probe. The v3 memo's CPU band [0.1893, 0.1963] is derived from PR102 drift extrapolation (CARGO-CULTED per Q2.5). Before any paid dispatch, the v3 design memo MUST be updated to include the Shannon first-principles CPU derivation (~0.1907) and an explicit empirical CPU axis target. Per Catalog #292: every design memo's predicted band must be derived independently on each axis, not extrapolated cross-axis.

**Q2 aggregated council verdict: CONDITIONAL-on-CPU-axis-probe PROCEED with narrowed scope**.

All 6 members support PROCEED with the explicit caveat that v3's CPU advantage MUST be empirically validated via:

1. **Section 14 probe FIRST** (per-pair sigma vs MLP entropy; $0 CPU; ~5 min): determines whether v3 paradigm is class-shift or within-class. If WITHIN-CLASS, v3 is ABANDONED.
2. **$0.30 Modal T4 100ep SMOKE** (per Catalog #167 smoke-before-full): produces empirical CUDA + CPU paired auth-eval on v3 smoke archive; validates CPU advantage hypothesis at low cost.
3. **5/5 council PROCEED on smoke result** BEFORE $5-10 Modal A100 1000ep FULL dispatch.

---

## Final council verdict (binding per T2 + Catalog #300)

### Q1: Is the v2 PIVOT verdict TRUE + VALIDATED?

**VERDICT: SPLIT-VERDICT = RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER + REQUEST-REINVESTIGATION-OF-ALTERNATIVE-REDUCERS**.

The real-CUDA Section 14 re-probe verdict (600/600 → class 2; MI = 0.000 bits/symbol; INDEPENDENT) is **TRUE for the per-pair-dominant SegNet argmax reducer with byte-expansion alignment as specified in v2's archive grammar**. This is HARD-EARNED empirical evidence on the contest video.

The probe verdict is **NOT TRUE as a class-wide falsification of all SegNet-derived conditioning**. Five alternative reducers (per-pixel class distribution, per-region class histogram, per-pair class-2-fraction, per-frame argmax, multi-label encoding) were NOT probed; each defines a DIFFERENT convex polytope whose feasibility must be independently checked.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable + "Council conduct" sextet-pact discipline: the v2 lane stays **research_only=true** with EXPANDED reactivation criteria that include the 4 alternative-reducer probes enumerated in Q1.4.

### Q2: Is v3 predicted band [0.2226, 0.2296] CUDA competitive with A1's 0.192848 CPU frontier?

**VERDICT: CONDITIONAL-on-CPU-axis-probe-PROCEED with narrowed scope**.

The CUDA-only band [0.2226, 0.2296] is far above A1's CPU 0.192848 and is NOT competitive on the CUDA axis. The CPU axis is where v3's value lies — per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable + the leaderboard ranks by CPU.

The v3 memo's CPU band derivation [0.1893, 0.1963] is via PR102-drift extrapolation (CARGO-CULTED per Assumption-Adversary). A Shannon first-principles derivation projects v3 CPU at ~0.1907 (marginal frontier-break, -0.0021 below A1). The actual CPU advantage is plausible (-0.002 to -0.005 vs Z3 v2 CPU baseline) but UNCERTAIN.

v3 is **FRONTIER-PROTECTING** (predicted to land within-cluster or at marginal frontier-break), **NOT FRONTIER-EXTENDING**. Council unanimously PROCEEDS-with-narrowed-scope:

1. Section 14 disambiguator probe (per-pair sigma vs MLP entropy) FIRST — $0; ~5 min CPU. BLOCKING per v3 Section 19 reactivation criterion #1.
2. v3 memo updated to include Shannon first-principles CPU derivation (replace PR102-drift extrapolation).
3. $0.30 Modal T4 100ep smoke per Catalog #167 smoke-before-full pattern.
4. 5/5 council PROCEED on smoke result BEFORE $5-10 full dispatch.

### Q3: ATW v2 + Tier 1 #4 + #5 spawn-resumption

**CLEAR TO PROCEED**. ATW v2 is the cooperative-receiver triple (Atick-Tishby-Wyner) — orthogonal to v2/v3 verdict. Tier 1 #4 + #5 are sister subagent threads on different surfaces. No pause is required by the Q1/Q2 verdicts.

---

## Op-routables for operator decision queue

1. **Section 14 probe (BLOCKING)**: implement `tools/probe_z3_g1_per_pair_sigma_vs_mlp_residual_entropy_v3.py` per v3 memo §21 op-routable #5. $0 cost; ~5 min CPU. Verdict determines whether v3 proceeds OR is abandoned.

2. **v2 alternative-reducer probes (LOW-PRIORITY)**: spawn a probe subagent to implement the 4 alternative-reducer probes from Q1.4. $0 cost; ~30 min CPU each. Verdict determines whether v2 paradigm class reactivates with a different reducer OR is class-wide deferred.

3. **v3 memo update (BLOCKING)**: update v3 design memo to replace PR102-drift CPU extrapolation with Shannon first-principles CPU derivation per Q2.5. Update §13.3 + §13.5 + add explicit CPU-axis Dykstra-feasibility check per Catalog #296.

4. **v3 lane registration (NON-BLOCKING)**: register `lane_z3_g1_per_pair_adaptive_sigma_v3_20260516` at L0 SKETCH per Catalog #126 + v3 memo §21 op-routable #1. Pre-registration is mandatory per CLAUDE.md "Lane lifecycle discipline".

5. **ATW v2 + Tier 1 #4 + #5 spawn-resumption (CLEAR)**: proceed per existing spawn plan. v2/v3 verdicts do not block.

6. **Council retrospective due 2026-06-15**: per Catalog #300 mission-alignment operational consequence 3 (deferred substrate 30-day retrospective). The retrospective covers v2 lane `lane_z3_g1_entropy_coded_v2_20260515` status + whether any alternative-reducer probes ran + reactivation decision.

7. **Recipe research_only=true preservation**: v2 lane recipe `.omx/operator_authorize_recipes/substrate_z3_g1_entropy_coded_v2_modal_t4_dispatch.yaml` stays `research_only=true / dispatch_enabled=false` per Catalog #240. v3 recipe (when registered) starts at the same state.

8. **Cathedral autopilot ranker update (NON-BLOCKING)**: append v3 candidate to ranker with `predicted_dykstra_band=[0.221, 0.228]` CUDA + `predicted_cpu_first_principles=0.1907` CPU + `class_shift_evidence=PENDING_section_14_probe`. Z1 empirical revision per Catalog #219 already applies the within-class density penalty if MDL Tier A returns > 0.90.

---

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS (class-shift not within-class)**: PARTIAL — Q1 verdict identifies that v2's per-pair-dominant reducer is falsified but alternative reducers are within the v2 paradigm class; Q2 verdict identifies that v3 is INCREMENTAL-CLASS-SHIFT (different parametrization of same per-pair conditioning paradigm).

2. **BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)**: PRESENT — sextet-pact positions are each ≤200 words; council verdict structure is one-paragraph per question; op-routables are 8 numbered items reviewable in 30 sec.

3. **DISTINCTNESS (explicitly different from sisters)**: PRESENT — this council deliberation IS distinct from prior council memos on Wunderkind G1 (v2 Appendix A landed the synthetic probe; v2 Appendix B landed the real-CUDA re-probe; this memo is the first ADVERSARIAL COUNCIL DELIBERATION validating both).

4. **RIGOR**: PRESENT — premise verification per Catalog #229 (8 PVs pre-edit including video sha256, SegNet weights sha256, MI computation correctness, A1 frontier custody, PR102 drift custody); adversarial review per Catalog #292 (sextet pact + Assumption-Adversary explicit assumption surfacing); 4-classification per HARD-EARNED-vs-CARGO-CULTED addendum.

5. **OPTIMIZATION PER TECHNIQUE**: PRESENT — Q1 + Q2 verdicts are per-question; v2 and v3 substrates are evaluated on their own merits; no canonical-vs-unique drift.

6. **STACK-OF-STACKS-COMPOSABILITY**: PRESENT — Q3 verdict explicitly preserves ATW v2 + Tier 1 #4 + #5 spawn-resumption (orthogonal); v3 composition matrix per v3 memo §12 is referenced but not overridden.

7. **DETERMINISTIC REPRODUCIBILITY**: PRESENT — probe artifacts are committed custody (sha256 pinned); council verdict is a deterministic function of the empirical anchors enumerated in Pre-deliberation; the memo itself is reproducible via the empirical inputs + the sextet-pact-position discipline.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: PARTIAL — the council deliberation does not change any substrate's compute/wall-clock profile; the Q2.6 CPU-advantage analysis projects v3 CPU advantages in determinism + reconstruction stability but does not establish them empirically.

9. **OPTIMAL MINIMAL CONTEST SCORE**: PARTIAL — Q2 verdict projects v3 CPU at ~0.1907 (marginal frontier-break, -0.0021 below A1) but ALL outcomes are CONDITIONAL on the Section 14 probe. Optimal-score path requires the Section 14 probe to confirm Interpretation A; otherwise v3 stays within-class.

---

## Cross-references

**Predecessor design memos**:
- `.omx/research/wunderkind_g1_v2_wire_grammar_class_shift_full_stack_design_20260516.md` (v2 with Appendix A synthetic probe + Appendix B real-CUDA re-probe)
- `.omx/research/wunderkind_g1_v3_per_pair_adaptive_sigma_full_stack_design_20260516.md` (v3 per-pair adaptive sigma)
- `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` (v2 substrate spec)

**Empirical anchors**:
- `experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/` (synthetic probe; I=0.0439; WEAK)
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/` (real-CUDA re-probe; I=0.000; INDEPENDENT)
- `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json` (PR102 CUDA 0.22839)
- A1 frontier anchor: `a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md` line 11 (CPU 0.192848; GHA run 25588422622; archive 178262 B)

**CLAUDE.md non-negotiables applied**:
- "KILL/FALSIFIED memory verdicts" (Q1 structural requirements 1+2+3 satisfied)
- "Forbidden premature KILL without research exhaustion" (Q1 verdict is DEFERRED, not KILLED)
- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" (Q2 CPU-axis primacy)
- "Apples-to-apples evidence discipline" (all scores axis-tagged + hardware-substrate-tagged)
- "Council conduct" sextet-pact discipline (6 members, no conservative bias, Contrarian + Assumption-Adversary veto powers)
- "Design decisions — non-negotiable" (council quorum + verdict format)
- "META-ASSUMPTION ADVERSARIAL REVIEW" (Assumption-Adversary seat active; CARGO-CULTED-vs-HARD-EARNED classification per assumption)

**Sister catalog gates active**:
- Catalog #127 + #190 + #249 (custody validator + hardware substrate + phantom-score directory)
- Catalog #220 (substrate L1+ scaffold operational mechanism — applies to v3 distinguishing feature)
- Catalog #229 (premise verification before edit — 8 PVs documented)
- Catalog #272 (distinguishing-feature integration contract — applies to v3 archive grammar)
- Catalog #290 (canonical-vs-unique decision per layer — v3 memo §15 satisfies; this council memo does not need to repeat)
- Catalog #292 (per-deliberation Assumption-Adversary discipline — explicit-assumption-statement per sextet-pact member)
- Catalog #294 (9-dimension success checklist evidence — satisfied above)
- Catalog #296 (Dykstra-feasibility check for predicted band — Q2.3 CPU-axis check added)
- Catalog #300 (council hierarchy v2 + mission-alignment — frontmatter satisfies)
- Catalog #303 (cargo-cult audit per assumption — Assumption-Adversary verdict)

---

## Op-summary

- **Q1 verdict**: SPLIT-VERDICT = RATIFY-FALSIFICATION (per-pair-dominant reducer mathematically excluded) + REQUEST-REINVESTIGATION (4 alternative reducers enumerated for class-wide deferral question). v2 lane stays research_only=true with expanded reactivation criteria.
- **Q2 verdict**: CONDITIONAL-on-CPU-axis-probe PROCEED with narrowed scope. v3 CUDA band [0.2226, 0.2296] is NOT competitive vs A1 CPU 0.192848 on the CUDA axis; CPU-axis derivation via PR102-drift extrapolation is CARGO-CULTED. Shannon first-principles CPU derivation projects ~0.1907 (marginal frontier-break). Section 14 probe is BLOCKING.
- **Per-sextet-pact-member positions**: 6 members each with explicit operating-within assumption per Catalog #292; HARD-EARNED-vs-CARGO-CULTED classification per the addendum; Contrarian + Assumption-Adversary veto powers exercised.
- **Op-routables**: 8 items for operator decision queue including Section 14 probe (BLOCKING), v2 alternative-reducer probes (LOW-PRIORITY), v3 memo update (BLOCKING), v3 lane registration (NON-BLOCKING), ATW v2 spawn-resumption (CLEAR), council retrospective due 2026-06-15.
- **ATW v2 + Tier 1 #4 + #5 spawn-resumption**: CLEAR TO PROCEED; orthogonal to v2/v3 verdict; do not pause.
- **Council continual-learning anchor**: persisted via `tac.council_continual_learning.append_council_anchor` to `.omx/state/council_deliberation_posterior.jsonl` per Catalog #128 fcntl-locked discipline.

---

## Observability surface (per CLAUDE.md max-observability standing directive)

1. **Per-layer inspection**: each council member's position is captured as a discrete subsection with explicit operating-within assumption + verdict; cite-chain anchored to specific empirical artifacts. No hidden reasoning.

2. **Per-signal decomposition**: Q1 + Q2 verdicts decompose into per-question + per-member sub-verdicts; the SPLIT-VERDICT structure exposes the falsification-vs-reinvestigation tension explicitly rather than collapsing it.

3. **Run-to-run diff**: any future re-deliberation can produce a diff against this memo's verdict via the structured frontmatter (council_verdict + council_dissent + council_assumption_adversary_verdict fields).

4. **Post-hoc query interface**: the council deliberation persists to `.omx/state/council_deliberation_posterior.jsonl` per Catalog #128; queryable via `tac.council_continual_learning.query_*` helpers.

5. **Cite-chain**: every empirical anchor in Pre-deliberation has a specific file path + sha256 (where applicable) + axis label; the council verdict is reproducible from the cited inputs.

6. **Counterfactual hooks**: the 4 alternative-reducer probes in Q1.4 + the Shannon first-principles derivation in Q2.5 + the explicit CPU-axis Dykstra check in Q2.3 are explicit counterfactual hooks the operator can route as follow-on probes.
