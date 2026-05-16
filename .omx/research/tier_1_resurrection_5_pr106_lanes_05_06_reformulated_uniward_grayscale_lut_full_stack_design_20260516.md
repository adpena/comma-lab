# Tier 1 Resurrection #5 — PR106 Lanes #05 + #06 REFORMULATED on substrates WITH mask channel — comprehensive full-stack design memo

**Date**: 2026-05-16
**Lane**: `lane_tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_20260516`
**Subagent**: TIER1-RESURRECTION-4-5-DESIGN-MEMOS-20260516 (RESPAWN of dead `a34688b73f9afafb5`)
**Predecessors**:
- `.omx/research/resurrection_audit_20260516.md` §1.4 + #4, #5 reactivation-queue rows (cargo-cult unwind anchor)
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md` (original PR106-architecture-mismatch falsification 2026-05-04)
- `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_05_uniward_delta_pr106_mask_channel.md` (Lane #05 original design)
- `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_06_mask_grayscale_lut_pr106_mask_replace.md` (Lane #06 original design)
- `experiments/extract_pr106_decoder.py` (commit 45149f21; empirical PR106 archive structure verification)
- `src/tac/substrates/grayscale_lut/__init__.py` (Selfcomp PR #56 paradigm reformulation target; L0 SKETCH but full RGB renderer with grayscale-LUT mechanism present)
- `src/tac/substrates/sane_hnerv/__init__.py` (HNeRV-family scaffold; L0 SKETCH; full RGB renderer)
- `src/tac/substrates/a1/__init__.py` (A1 runtime adapter; the canonical mask-bearing-substrate-with-stable-CPU-anchor)
- `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md` Q2 verdict (CPU axis is leaderboard axis; PR102 drift extrapolation is CARGO-CULTED)

**Operating mode**: UNIQUE-AND-COMPLETE-PER-METHOD per the standing directive `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + 2026-05-15 PR95 META-level retrospective. The PR106 Lane #05+#06 application was CANONICAL-FORCE-FIT (UNIWARD is Fridrich-canonical for steganalysis-driven cost functions; grayscale-LUT is Selfcomp-canonical for analog-mask substitution; both require a mask cover signal which PR106 lacks). This reformulation FORKS the substrate target, NOT the technique recipes.

**Status at landing**: DESIGN-ONLY, RESEARCH-ONLY at the recipe level. Reactivation gated on (a) substrate-with-mask-channel identification + readiness (Variant A grayscale_lut L0 → L1; Variant B sane_hnerv L0 → L1; Variant C A1 substrate-extension feasibility audit); (b) per-variant paired-CPU+CUDA Modal A100 smoke per Catalog #167.

---

## 1. Frontmatter — premise verification + lane registry + sister-subagent map

### Premise verifications (Catalog #229; 8 PVs verified BEFORE any design statement)

- **PV-1** PR106 has NO separate mask channel. Empirical verification per `experiments/extract_pr106_decoder.py` (commit 45149f21) on archive `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip` (186,131 bytes total): `0.bin` decomposes into `decoder_brotli` (170,278 bytes; HNeRV decoder state_dict INT8 quant + zigzag + brotli) + `latent_brotli` (15,849 bytes; 600 frame-pair latents, 28-dim, uint8 delta-coded + brotli). Zero tensors with 'mask' in name. PR106 IS HNeRV-style: per-pair 28-dim latent + HNeRV decoder JOINTLY define full RGB output; no SegNet-mask; no mask.mkv; no separate-stream-encoding-mask-only design like Quantizr's submission.
- **PV-2** Lane #05 original design = `revival_plan_05_uniward_delta_pr106_mask_channel`. UNIWARD (Universal Wavelet Relative Distortion) is Fridrich's steganographic embedding cost function (Fridrich-Holub 2014); adaptive payload distribution that hides information in textured regions; operates on SIGNAL-vs-COVER decomposition (e.g., mask vs frame). REQUIRES a mask cover signal to attack.
- **PV-3** Lane #06 original design = `revival_plan_06_mask_grayscale_lut_pr106_mask_replace`. The grayscale-LUT trick (Selfcomp's σ=15 Gaussian-softmax-LUT over CLASS_TARGETS per PR #56 paradigm) replaces a SegNet's encoder output with a fixed lookup table. REQUIRES (a) a SegNet-mask discrete-class output to replace AND (b) the Gaussian-softmax structure baked into the architecture. Neither exists in PR106's HNeRV.
- **PV-4** Substrates WITH mask channel in the repo: (a) `src/tac/substrates/grayscale_lut/` (Selfcomp PR #56 paradigm; L0 SKETCH; FiLM-conditioned RGB decoder + grayscale 1-channel field). NOTE: per `__init__.py` line 37 "L5 full RGB renderer | PASS (NOT a mask codec; LUT produces RGB)" — grayscale_lut is a FULL RGB renderer that USES the LUT mechanism internally; not strictly a mask codec at the archive layer but uses mask-class CLASS_TARGETS structurally; (b) Quantizr-style submissions in the public archive (the 0.33 leader uses encoded masks.mkv AV1 + renderer + poses; per Lane MM v2 + Lane AL history); (c) A1 submission runtime per `src/tac/substrates/a1/__init__.py` — A1 IS the canonical 0.192848 CPU-anchored substrate but is a runtime adapter to the committed A1 archive; mask handling is internal to A1's HNeRV variant.
- **PV-5** Lane MM v2 (the encoder-only grayscale-LUT bolt-on on Lane A renderer) IS the historically-attempted Lane #06-like mechanism; Lane MM v2 produced 2.63 `[contest-CPU advisory]` per resurrection audit §1.5. Per audit lines 113-124: HARD-EARNED architectural mismatch (3ch-trained renderer + 1ch grayscale-LUT bolt is structural per Selfcomp's PR #56 paradigm: SegMap is trained-from-scratch WITH the LUT, not bolted-on); CARGO-CULTED score magnitude (51× PoseNet ratio on CPU-not-CUDA). Reactivation queue #7 = `lane_mm_v3_segmap_trained_from_scratch`.
- **PV-6** UNIWARD + grayscale-LUT are leaderboard-proven primitives. Resurrection audit Tier 1 #1.4 line 108: "both UNIWARD + grayscale-LUT are leaderboard-proven primitives (PR101/PR103 silver + PR#56 paradigm)". The TECHNIQUES are HARD-EARNED canonical (Fridrich + Selfcomp respectively); the LANE DESIGN was cargo-culted (assumed PR106 follows PR#56 paradigm; it doesn't).
- **PV-7** Per CLAUDE.md "Beauty, simplicity, and developer experience — non-negotiable" + HNeRV parity discipline L7: bolt-on size ≤ 350 LOC; substrate engineering may exceed but must tag `lane_class=substrate_engineering`. The reformulated lanes will be bolt-ons on existing substrates (grayscale_lut / sane_hnerv) OR substrate-extension on A1 (substrate-engineering tag required).
- **PV-8** Per T2 council Q2 verdict: CPU axis is leaderboard axis; PR102 CUDA-CPU drift -0.0330 is CARGO-CULTED extrapolation; predicted bands MUST be derived from Shannon first principles on each axis independently. This reformulation MUST predict CPU + CUDA bands independently per Catalog #296 Dykstra-feasibility on both axes.

### Sister-subagent ownership map (Catalog #230)

This subagent is **READ-ONLY** on source code (`src/tac/`, `experiments/`, `submissions/`, `tools/`, `.omx/operator_authorize_recipes/`). Writes ONLY to:

- `.omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` (this memo)
- `.omx/research/tier_1_resurrection_4_pr101_compressai_balle_reformulated_full_stack_design_20260516.md` (companion memo, written first)
- `.omx/state/subagent_progress.jsonl` (canonical checkpoint store per Catalog #206)
- 1 commit via canonical serializer with `--expected-content-sha256` per Catalog #157 + #174 + #289 (batched with Memo #1)

Sister-subagent `a8ef880a01e1fd84f` (ATW v2 substrate-build) in flight on DIFFERENT files (`experiments/train_substrate_atw_codec_v2*.py`, recipes, substrate modules). No file collision; this memo references ATW v2 as a comparative reference (ATW v2 is class-shift via cooperative-receiver; PR106 reformulation is technique-class-preservation via substrate-pivot).

### Operating-within assumption-statement (Catalog #292 / Assumption-Adversary seat)

The assumption I am operating within for this reformulation: *"PR106 Lanes #05+#06 falsification was CANONICAL-FORCE-FIT (Pattern B substrate-mismatch-as-class-kill per audit; specifically a lane-design-vs-PR106-architecture mismatch — Quantizr-style mask.mkv presumed; PR106 is HNeRV with brotli-decoder + brotli-latents only). UNIWARD's class IS PRESERVED (Fridrich canonical). Grayscale-LUT's class IS PRESERVED (Selfcomp canonical, ANCHORED by Selfcomp's 0.38 PR#56 result). The substrate-optimal application is to substrates THAT HAVE a mask channel (or equivalent class-discrete-output structure). Three candidate target substrates: (a) `grayscale_lut` substrate (L0 SKETCH; uses CLASS_TARGETS internally), (b) `sane_hnerv` substrate (L0 SKETCH; HNeRV-family but full RGB renderer — same architectural class as PR106 but with mask-class-internal-trainable mechanism), (c) `A1` (canonical 0.192848 CPU-anchored frontier — substrate-extension feasibility audit required because A1 is a frozen runtime adapter). The empirical question is NOT 'do UNIWARD or grayscale-LUT work?' (both are published frontier primitives) but 'which contest substrate has the structural conditions UNIWARD or grayscale-LUT need, AND lands within an achievable Dykstra-feasibility band on the CPU axis?'"*

HARD-EARNED basis: PR106 mask-channel absence is verified per PV-1 + the 2026-05-04 falsification memo. UNIWARD requires signal-vs-cover decomposition; grayscale-LUT requires SegNet-mask discrete-class output. Neither exists in PR106's HNeRV.

The Assumption-Adversary seat would challenge: *"Is 'substrate-with-mask-channel exists therefore Lane #05 / #06 reformulation is viable' itself a cargo-cult? Lane MM v2 ALREADY tried grayscale-LUT-style bolt-on on Lane A renderer and produced 2.63 [contest-CPU advisory] — clearly above the 0.196-0.199 cluster and dramatically above A1's 0.192848 anchor. The 2026-05-04 reformulation alternative-paths (Lane #05 alternative on PR106 latent stream; Lane #06 alternative on HNeRV decoder stem) were EXPLICITLY marked 'speculative and not council-approved; defer indefinitely'. The reformulation must explicitly distinguish 'technique class is alive' from 'technique-on-mask-channel-substrate beats the 0.196-0.199 cluster'."* — answer: §4 below ships THREE target-variants (grayscale_lut, sane_hnerv, A1-extension) with EXPLICIT per-variant Dykstra-feasibility on CPU + CUDA axes per Catalog #296 + T2 council Q2 verdict + EXPLICIT acknowledgment that the predicted bands span "frontier-protecting" (within-cluster) to "frontier-extending" (-0.005 below A1) per Variant; reactivation criteria require Variant A grayscale_lut L0 → L1 promotion to land FIRST before either UNIWARD #05 or grayscale-LUT #06 reformulation fires.

### Lane registry pre-registration (Catalog #126)

To be claimed in same commit batch:
```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_20260516 \
    --name "Tier 1 Resurrection #5 (PR106 Lanes #05+#06 reformulated: UNIWARD-texture + grayscale-LUT on substrates WITH mask channel)" \
    --phase 2
```

---

## 2. Executive summary

PR106 Lanes #05+#06 were FALSIFIED 2026-05-04 because PR106 has no separate mask channel. The resurrection audit Pattern B classification IS CORRECT: the falsification is HARD-EARNED for PR106-as-substrate; it is INVALID as a class-kill of UNIWARD or grayscale-LUT.

**Reformulation thesis**: UNIWARD and grayscale-LUT are leaderboard-proven primitives (Fridrich-canonical for steganalysis-driven texture-cost functions; Selfcomp-canonical for analog-mask substitution; both anchored by PR#56's 0.38 result). The contest substrate landscape has THREE classes of targets with mask-channel-like structure where these techniques are operationally meaningful:

1. **Variant A — grayscale_lut substrate (Selfcomp PR #56 paradigm; L0 SKETCH already lands)**. The grayscale_lut substrate USES CLASS_TARGETS (Gaussian-softmax-LUT over class targets) internally per the Selfcomp 0.38 anchor architecture. UNIWARD-texture can be applied to the grayscale field's spatial distribution (Fridrich-style adaptive bit-allocation to textured regions of the grayscale signal); grayscale-LUT IS the substrate's central mechanism (no bolt-on required). Cost: grayscale_lut substrate L0 → L1 promotion required first ($5-15 Modal A100 to verify substrate viability on alpha sane_hnerv anchor per `__init__.py` lines 57-62); then UNIWARD #05 bolt-on $5 paired smoke.

2. **Variant B — sane_hnerv substrate (HNeRV-family; L0 SKETCH lands)**. sane_hnerv is the same HNeRV-family class as PR106 but is a substrate IN FLIGHT for trainer-side wire-in (Phase 2 dispatch via subagent waves). Grayscale-LUT-on-HNeRV requires reformulating the HNeRV decoder to accept a class-discrete output (substrate engineering on sane_hnerv; ~350 LOC budget); UNIWARD-on-HNeRV-latent-residual requires defining a wavelet-domain residual stream over the 28-dim latent (compute the wavelet of `latent_t - latent_{t-1}` per pair; allocate bits via UNIWARD's texture-cost; this IS the 2026-05-04 memo's "Lane #05 alternative" path generalized). Cost: $5-15 Modal A100 per substrate variant.

3. **Variant C — A1 substrate-extension (CONDITIONAL on feasibility audit)**. A1 IS the canonical 0.192848 CPU-anchored frontier substrate per audit Tier 1 anchors. The A1 runtime adapter `src/tac/substrates/a1/__init__.py` is FROZEN (committed A1 submission runtime); substrate-extension requires either (a) building a sister A1-variant substrate that re-implements A1's HNeRV variant with mask-class-discrete-output-extensibility OR (b) bolt-on at the A1 archive layer (adding a UNIWARD-residual-sidecar to A1's existing archive). Cost: A1 substrate-extension feasibility audit FIRST ($0; ~3-5 hr editor); conditional $5-15 paired smoke per extension variant.

**Predicted ΔS bands** (CPU axis primary per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + T2 council Q2 verdict):

- **Variant A (grayscale_lut + UNIWARD #05 OR grayscale-LUT #06)** — Predicted CPU band: `NULL pending grayscale_lut substrate L0 → L1 promotion` [prediction]. Substrate viability gate per Selfcomp 0.38 anchor implies the substrate IS structurally viable in principle; the question is whether the contest video's mask-class distribution matches Selfcomp's PR#56 training distribution. Conservative Shannon estimate: if substrate lands at ≤0.21 CPU per `__init__.py` reactivation criteria, +UNIWARD #05 adds -0.001 to -0.003 rate-axis (texture-cost UNIWARD adds maybe 1-3% rate savings on the grayscale field); +grayscale-LUT #06 is the substrate's mechanism (no additional bolt-on). **Dykstra-feasibility on CPU axis**: PENDING substrate L1 promotion; bound the band by [substrate_anchor_score - 0.003, substrate_anchor_score]. **Dykstra-feasibility on CUDA axis**: same PENDING.

- **Variant B (sane_hnerv + UNIWARD-latent-residual OR grayscale-LUT-on-HNeRV-stem)** — Predicted CPU band: `NULL pending sane_hnerv substrate L0 → L1 promotion AND substrate-engineering bolt-on landing` [prediction]. Conservative Shannon estimate: UNIWARD-latent-residual savings bound by `25 × wavelet_residual_compressed_bytes / 37545489`; with 15,849 bytes of latent data and UNIWARD-style adaptive allocation potentially saving 5-10% on the wavelet-residual stream specifically, the rate-axis savings would be `25 × (15849 × 0.075) / 37545489 ≈ -0.00079`; this is BELOW measurement noise floor and within-class. **Dykstra-feasibility on CPU axis**: PASSES with marginal predicted savings; PENDING substrate readiness. Same for CUDA.

- **Variant C (A1 substrate-extension)** — Predicted CPU band: `NULL pending feasibility audit` [prediction]. A1's CPU baseline 0.192848 is the frontier; any extension must be RIGOROUSLY BYTE-VERIFIED to not regress the existing frontier per Catalog #220 (substrate L1 scaffold byte-addition requires operational mechanism declaration). A UNIWARD-residual-sidecar adds bytes (rate-cost) for potentially small distortion savings; the Dykstra-feasibility intersection is narrow.

**Substrate-mismatch analysis core finding**: Pattern B cargo-cult is REAL and PARTIALLY AVOIDABLE. Reformulation IS technically defensible per UNIQUE-AND-COMPLETE-PER-METHOD discipline; HOWEVER, the predicted savings are within-cluster across all three variants UNLESS Variant A's grayscale_lut substrate FIRST achieves a sub-0.20 anchor (which would prove Selfcomp PR#56 paradigm transfer to the contest video). Per the abandon-within-class directive (T2 council Q2 verdict), the reformulation has STRONG within-class plateau risk.

**Reactivation gate** (per per-Variant):
- Variant A: grayscale_lut L0 → L1 promotion via alpha sane_hnerv anchor at ≤0.21 (per substrate's own reactivation criteria) AND rate-axis headroom OR pose-axis underperformance confirmed by post-anchor diagnostic.
- Variant B: sane_hnerv L0 → L1 promotion via Phase 2 dispatch + UNIWARD/grayscale-LUT-on-HNeRV bolt-on substrate engineering.
- Variant C: A1 substrate-extension feasibility audit returns POSITIVE (extension preserves A1's frontier without regressing CPU baseline).

**Verdict on whether this lane should fire NOW**: NO. Variant A grayscale_lut substrate MUST land its alpha anchor first per its own reactivation criteria. Variant B and C are CONDITIONAL on substrate readiness. The cheapest first action is the Variant C feasibility audit ($0 editor work), which informs whether the operator should pursue Variant A's grayscale_lut promotion first or Variant C's A1 substrate-extension first.

---

## 3. Substrate-mismatch analysis — WHY PR106 was wrong target; WHICH substrates ARE right targets

### 3.1 The substrate-mismatch in detail

The 2026-05-04 falsification memo (PV-1) is empirically correct: PR106 = HNeRV (Hybrid Neural Representation for Video) where the per-pair 28-dim latent + HNeRV decoder JOINTLY define full RGB output. There is no separate mask channel; the latent IS the implicit-neural-representation of the whole video.

Lane #05 (UNIWARD-delta on PR106 mask channel) requires a signal-vs-cover decomposition (e.g., mask vs frame) for UNIWARD's adaptive payload distribution to operate on a "cover" with texture variation. PR106 has no mask cover signal — only a 28-dim latent vector per pair and a single HNeRV decoder.

Lane #06 (mask-grayscale-LUT on PR106 mask channel) requires a SegNet-mask discrete-class output (5 classes) to replace with a Gaussian-softmax-LUT. PR106 has no SegNet-mask output; the renderer produces RGB directly via HNeRV decoding.

**This IS Pattern B substrate-mismatch-as-class-kill**: the falsification is VALID at the PR106-as-substrate level; it is INVALID as a class-kill of UNIWARD or grayscale-LUT (both are leaderboard-proven canonical primitives).

### 3.2 Which substrates have mask channels (the substrate-target identification)

Three substrates in the current repo have mask-channel-equivalent structure:

**Substrate A — `grayscale_lut`** (`src/tac/substrates/grayscale_lut/__init__.py`):
- Per the substrate's own design (`__init__.py` lines 6-11): "AV1 grayscale + Gaussian-LUT representation as a true ANALOG-signal codec: a per-pair grayscale (single luminance channel) is codec-compressed (AV1, JPEG, or simply quantized), and a learned colorization LUT (or a tiny FiLM-conditioned RGB decoder) maps grayscale -> RGB at inflate time."
- Uses CLASS_TARGETS structurally per Selfcomp PR #56 paradigm.
- Lane registration: `lane_substrate_grayscale_lut_20260512` (L0 SKETCH; research_only=true; substrate_engineering exception per HNeRV L7).
- Reactivation criteria (per `__init__.py` lines 57-62): "Alpha (sane_hnerv) empirical anchor at ≤0.21 AND post-anchor diagnostic flags either (a) rate-axis headroom (≥5%) ... OR (b) pose-axis underperformance".
- IS NOT a mask codec per L5 "PASS (NOT a mask codec; LUT produces RGB)" — but USES class-target structure that UNIWARD-texture can attack and grayscale-LUT IS the substrate's mechanism.

**Substrate B — `sane_hnerv`** (`src/tac/substrates/sane_hnerv/__init__.py`):
- HNeRV-family class (same as PR106); L0 SKETCH; full RGB renderer (NOT mask codec per L5 PASS).
- Lane registration: `lane_substrate_sane_hnerv_*`; Phase 2 dispatch in flight per subagent waves.
- DOES NOT have an inherent mask channel; reformulation requires either (a) substrate-engineering on sane_hnerv to add a class-discrete output OR (b) UNIWARD-on-latent-residual which is a generalized version of the 2026-05-04 memo's "Lane #05 alternative" path.

**Substrate C — `A1`** (`src/tac/substrates/a1/__init__.py`):
- Runtime adapter to the committed A1 submission runtime.
- A1 IS the canonical 0.192848 CPU-anchored frontier substrate.
- A1's HNeRV variant internally handles masks (per the A1 architecture, which has poses + latents + decoder; the decoder may or may not have an explicit mask-class output depending on the specific variant).
- Substrate-extension requires feasibility audit: can A1's archive be extended with a UNIWARD-residual-sidecar without regressing the 0.192848 CPU baseline?

**Quantizr (the 0.33 leader)** — external; cannot extend without access to the trained substrate. NOT a reformulation target.

### 3.3 Why this matters for contest score (operating-point analysis)

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" non-negotiable: at PR106's frontier operating point (pose_avg ~3.4e-5), pose marginal sensitivity is 2.71× SegNet's; the marginal-value-per-byte for pose improvements EXCEEDS seg improvements. UNIWARD-texture (a rate-axis savings primitive) and grayscale-LUT (a distortion-axis primitive operating via class-discrete substitution) target DIFFERENT axes:

- **UNIWARD #05** (rate-axis savings): if the grayscale field's texture variance is HIGH (driving footage has high-texture regions like road texture, vegetation; low-texture regions like sky), UNIWARD's adaptive bit-allocation could save 5-10% on the grayscale stream. At PR106's archive size (~186 KB), 5-10% of the grayscale stream (which is the dominant rate term per `__init__.py` line 66: "the dominant rate term") could be ~5-15 KB savings → `25 × 10000 / 37545489 ≈ -0.0067` rate-axis ΔS. This is meaningfully sub-cluster (the 0.196-0.199 cluster has rate-axis variation of ~0.001-0.003 typically).

- **Grayscale-LUT #06** (distortion-axis via class-discrete substitution): if the contest video's per-pair mask-class distribution is well-approximated by Selfcomp's PR#56 distribution, the LUT can substitute the trained-per-pair decoder output without significant distortion regression. Selfcomp's 0.38 anchor IS proof-of-principle; whether the LUT TRANSFERS to the contest video's specific mask-class distribution is the empirical question.

**Aggregate score impact**: PROVIDED Variant A grayscale_lut substrate lands at ≤0.21 CPU, +UNIWARD #05 could push to [0.20, 0.207] CPU (small additional savings); +grayscale-LUT #06 is the substrate's central mechanism (no additional bolt-on). At BEST, the grayscale_lut family with both bolt-ons lands in the 0.196-0.199 cluster (within-class plateau) UNLESS the substrate breaks below 0.20 in standalone (which would prove Selfcomp PR#56 paradigm transfer).

### 3.4 Why this is NOT cargo-cult of "if technique X is leaderboard-canonical, X works on our substrate"

The reformulation is NOT *"UNIWARD won steganalysis competitions therefore UNIWARD beats the 0.196-0.199 cluster on the contest substrate"*. The reformulation is: *"UNIWARD requires a signal-vs-cover decomposition AND adaptive payload distribution AND texture-cost function evaluation. The grayscale field in grayscale_lut substrate IS a signal with texture variation; UNIWARD's adaptive bit-allocation to high-texture regions of the grayscale field is OPERATIONALLY MEANINGFUL but the predicted savings ceiling is bounded by the grayscale field's actual texture-vs-smooth ratio on the contest video, NOT by UNIWARD's theoretical optimality."*

Per the abandon-within-class directive: the reformulation must explicitly acknowledge that the predicted savings (-0.001 to -0.003 per Variant A bolt-on; even smaller for Variant B) are WITHIN-CLUSTER. The reformulation IS technically defensible per UNIQUE-AND-COMPLETE-PER-METHOD but is at HIGH RISK of within-class plateau outcomes.

---

## 4. Architecture (FULL reformulation spec — three variants)

### 4.1 Variant A — grayscale_lut substrate + UNIWARD-texture #05 bolt-on + grayscale-LUT #06 (substrate's mechanism)

**Substrate prerequisite**: grayscale_lut L0 → L1 promotion required first (per substrate's own reactivation criteria).

**Bolt-on architecture for UNIWARD #05** (~150 LOC; ≤350 LOC bolt-on budget per HNeRV L7):
- Compute UNIWARD texture-cost map on the grayscale field per pair: `cost_map[pair_idx, h, w] = wavelet_cost(grayscale[pair_idx], h, w)` per Fridrich-Holub 2014.
- Adaptive bit-allocation: instead of uniform int8 quantization on the grayscale field, use variable-precision encoding where smooth regions (low cost_map) get coarser quantization (int4 or int3) and high-texture regions get full int8.
- Encode the variable-precision grayscale field via arithmetic-coded segments + per-segment precision tag.
- Decoded grayscale field at inflate time: parse precision tags, decode each segment, reconstruct full-precision grayscale per pair, feed to grayscale_lut decoder.
- Bit savings target: 5-10% on grayscale stream (Selfcomp's grayscale field is the dominant rate term per `__init__.py` line 66).

**Grayscale-LUT #06** IS the substrate's central mechanism — already implemented in grayscale_lut substrate; no additional bolt-on.

**Score-aware loss**: inherits grayscale_lut substrate's existing Lagrangian `α·B/N + β·d_seg + γ·sqrt(d_pose)` per `__init__.py` line 53. UNIWARD bolt-on adds NO additional loss term (it operates on the encoded bytes, not the trained weights); the rate term `α·B/N` automatically reflects the UNIWARD savings.

### 4.2 Variant B — sane_hnerv substrate + UNIWARD-latent-residual #05 OR grayscale-LUT-on-HNeRV-stem #06

**Substrate prerequisite**: sane_hnerv L0 → L1 promotion required first (Phase 2 dispatch in flight per subagent waves).

**Bolt-on architecture for UNIWARD-latent-residual #05** (~250 LOC):
- Compute wavelet of per-pair latent residual: `residual_t = latent_t - latent_{t-1}` per pair; apply 2D discrete wavelet transform (DWT) on the 28-dim residual reshaped as `(4, 7)` or similar small-grid (the 28-dim has no natural 2D structure; this is an architectural decision — alternatively 1D wavelet on 28-d vector).
- Apply UNIWARD adaptive bit-allocation to the wavelet coefficients: smooth coefficients (high concentration after DWT) get coarser quantization; high-detail coefficients get full precision.
- Encode wavelet coefficients via arithmetic coding + per-coefficient precision tag.
- Decoded at inflate: parse, decode, inverse-DWT, reconstruct residuals, accumulate to reconstruct latents, feed to sane_hnerv decoder.
- Bit savings target: 5-10% on the 15,849-byte latent stream → ~1 KB savings → rate-axis ΔS `25 × 1000 / 37545489 ≈ -0.00067`. This is BELOW measurement noise floor and within-class.

**Bolt-on architecture for grayscale-LUT-on-HNeRV-stem #06** (~350 LOC; substrate-engineering tag required because it modifies the decoder architecture):
- Per the 2026-05-04 memo's "Lane #06 alternative": apply LUT-substitution to the HNeRV decoder's stem.weight (the 1728×28 Linear layer that acts as a fixed-channel-projection).
- Replace the trained 1728×28 Linear with a fixed LUT (e.g., a Gaussian-softmax LUT over per-pair embedding-index → channel-projection).
- Shrinks the stem from 1728×28×4 bytes (= 193 KB) to LUT_size_bytes (potentially ~10 KB if 5-class LUT × 28-dim × 64 classes); saves ~180 KB on the decoder stem.
- HOWEVER: LUT substitution is a STRONG architectural prior (it's not just a rate-axis savings; it changes the decoder's expressivity). The contest video's actual per-pair channel-projection distribution may NOT be well-approximated by a fixed LUT.
- Distortion-axis risk: HIGH (Lane MM v2 already proved this class of bolt-on regresses to 2.63 CPU when applied to a 3ch-trained renderer).

### 4.3 Variant C — A1 substrate-extension (UNIWARD-residual-sidecar)

**Substrate prerequisite**: A1 substrate-extension feasibility audit (no Modal dispatch; $0 editor work).

**Sidecar architecture** (~200 LOC):
- A1's existing archive has decoder + latents + poses; total ~178 KB per the audit anchors.
- Add a UNIWARD-residual-sidecar at offset `A1_HEADER_LEN + A1_BODY_LEN`: encodes a small residual stream that the inflate runtime adds to A1's decoded RGB output per pair.
- Residual stream content: per-pair 1-channel grayscale-residual quantized with UNIWARD-adaptive bit-allocation (high-texture regions get more bits).
- Bit budget: target ≤5 KB sidecar (so rate-axis cost ≤ `25 × 5000 / 37545489 ≈ +0.00333`).
- Distortion-axis benefit: residual sidecar adds per-pair luminance correction; targets SegNet's argmax-at-boundary-pixels distortion.
- Net rate-axis: +0.00333 byte cost - potential distortion-axis benefit (UNCERTAIN — depends on whether the residual sidecar actually corrects SegNet argmax mistakes at the contest video's boundary pixels).

**Risk**: A1 IS the canonical frontier; ANY sidecar addition that doesn't STRICTLY improve net score will REGRESS the frontier. Per Catalog #220 "L1 scaffold byte-addition requires operational mechanism": the sidecar's mechanism MUST be empirically demonstrated to consume bytes via inflate-time operational use.

### 4.4 Variant adjudication

| Axis | Variant A (grayscale_lut) | Variant B (sane_hnerv) | Variant C (A1-extension) |
|---|---|---|---|
| Substrate prerequisite | grayscale_lut L0 → L1 promotion | sane_hnerv L0 → L1 promotion | A1 substrate-extension feasibility audit |
| Cost to first empirical anchor | $5-15 substrate anchor + $5 paired smoke for UNIWARD bolt-on | $5-15 substrate anchor + $5 paired smoke per bolt-on | $0 feasibility audit + conditional $5-15 paired smoke |
| Predicted ΔS basis | Selfcomp PR#56 0.38 paradigm transfer + UNIWARD 5-10% grayscale-stream savings | Shannon R(D) on latent-residual entropy × small rate; HNeRV-stem-LUT distortion gamble | A1 frontier ± UNIWARD-residual-sidecar net-effect (UNCERTAIN) |
| Within-class plateau risk | MEDIUM (substrate IS class-distinct from PR106; Selfcomp 0.38 anchor implies viability; but contest video may not transfer) | HIGH (HNeRV-family; same class as PR106; bolt-on savings below noise floor) | HIGH (extending A1 within HNeRV-family class; sidecar may regress) |
| Council adjudication BLOCKING gate | substrate alpha anchor ≤0.21 | substrate anchor + bolt-on integration test | feasibility audit returns POSITIVE without regression risk |

**Recommended sequence**: Variant C feasibility audit FIRST ($0 editor work; informs whether Variant A or B is the better next step). Then Variant A IF grayscale_lut substrate's alpha anchor lands at ≤0.21 per its own reactivation criteria.

---

## 5. Pretraining

### 5.1 Variant A (grayscale_lut)

No pretraining required at scaffold landing. Substrate initializes from scratch per its own design.

### 5.2 Variant B (sane_hnerv)

No pretraining required; bolt-ons train jointly with substrate.

### 5.3 Variant C (A1-extension)

A1 weights are FROZEN (committed runtime); sidecar weights train from scratch against A1's existing reconstructed RGB output.

---

## 6. Curriculum

### 6.1 Variant A

JOINT training of grayscale_lut substrate + UNIWARD encoding parameters. UNIWARD bolt-on operates at encode-time only (no gradients flow through UNIWARD parameters during training because UNIWARD is a discrete-allocation algorithm; bit savings reflect in `B(θ)/N` rate term).

### 6.2 Variant B

JOINT for UNIWARD-latent-residual; SEQUENTIAL for grayscale-LUT-on-HNeRV-stem (train sane_hnerv first; then graft LUT onto frozen sane_hnerv and fine-tune; OR train from scratch with LUT in place — substrate-engineering decision).

### 6.3 Variant C

SEQUENTIAL: A1 frozen; sidecar trained against A1's reconstructed RGB output to minimize residual MSE conditional on UNIWARD-allocated bit budget.

---

## 7. Architecture priors

### 7.1 Variant A

UNIWARD texture-cost function per Fridrich-Holub 2014. Adaptive bit-allocation via wavelet-domain cost map. CANONICAL — no fork.

### 7.2 Variant B

UNIWARD on latent residual is NOVEL APPLICATION but uses CANONICAL UNIWARD cost function. LUT-on-HNeRV-stem is novel substrate engineering; no canonical to fork.

### 7.3 Variant C

UNIWARD-residual-sidecar is novel sidecar pattern; the sidecar grammar itself is novel (not a canonical fork; substrate-engineering on the A1 archive layer).

---

## 8. Post-training (TTO)

All variants defer TTO to follow-up iteration.

---

## 9. Score-aware loss design

### 9.1 Variant A

Inherits grayscale_lut substrate's existing Lagrangian:
```
L = α · B(θ)/N + β · d_seg(θ) + γ · sqrt(d_pose(θ))
```
UNIWARD bolt-on does not add a loss term; it reduces B(θ).

### 9.2 Variant B

UNIWARD-latent-residual: same form as Variant A; UNIWARD reduces B(θ) via wavelet-residual bit savings.

Grayscale-LUT-on-HNeRV-stem: same form, but distortion terms `β·d_seg + γ·sqrt(d_pose)` will reflect the LUT-imposed architectural prior cost.

### 9.3 Variant C

A1 is frozen; the SIDECAR has its own Lagrangian:
```
L_sidecar = α · sidecar_bytes/N + β · d_seg(rgb_a1 + sidecar_residual, rgb_gt) + γ · sqrt(d_pose(rgb_a1 + sidecar_residual, rgb_gt))
```

---

## 10. Archive grammar

### 10.1 Variant A

grayscale_lut substrate's existing monolithic 0.bin GLV1 grammar per `__init__.py` line 48:
`parse_archive() -> (decoder_sd, film_sd, grayscale_int8, meta)`

UNIWARD bolt-on adds:
- Section: `grayscale_uniward_encoded` (variable-precision encoded segments + per-segment precision tags) REPLACES `grayscale_int8`.
- New grammar version GLV2 with `grayscale_uniward_encoded` AND `uniward_precision_tags` sections.

### 10.2 Variant B

sane_hnerv archive + bolt-on sections:
- UNIWARD-latent-residual: adds `latent_residual_wavelet_encoded` + `wavelet_precision_tags` sections after sane_hnerv's existing latent section.
- LUT-on-HNeRV-stem: REPLACES the existing `decoder_stem_brotli` section with `decoder_stem_lut` (smaller fixed-size LUT).

### 10.3 Variant C

A1 archive (UNCHANGED) + sidecar section appended at offset `A1_HEADER_LEN + A1_BODY_LEN`:
- New section: `uniward_residual_sidecar` (length-prefixed; backward-compatible — inflate runtimes that don't recognize the section skip it; only the new sidecar-aware inflate adds the residual).

---

## 11. Inflate runtime

### 11.1 Variant A

grayscale_lut existing inflate (~95 LOC budget per `__init__.py` line 50) + ~50 LOC UNIWARD decode bolt-on:
- Parse precision tags, decode each grayscale segment at its declared precision, reconstruct full-precision grayscale field.
- Feed reconstructed grayscale + film_sd to decoder.

### 11.2 Variant B

sane_hnerv existing inflate + bolt-on additions:
- UNIWARD-latent-residual: parse wavelet-encoded residuals, decode, inverse-DWT, accumulate.
- LUT-on-HNeRV-stem: parse LUT, instantiate Linear-stem from LUT lookup; substitute into decoder.

### 11.3 Variant C

A1 existing inflate UNCHANGED + sidecar-aware inflate wrapper:
- Run A1 inflate normally to get baseline RGB per pair.
- Parse sidecar section; if present, decode UNIWARD residual; add per-pair to baseline RGB.

---

## 12. Export contract

### 12.1 Variant A

Trainer emits archive bytes; UNIWARD encoding deterministic per fixed cost-map computation. Archive byte-stable per seed pin (UNIWARD does not introduce stochasticity).

### 12.2 Variant B

Trainer emits archive bytes; UNIWARD-latent-residual deterministic. LUT-on-HNeRV-stem: LUT lookup table is part of state_dict, byte-stable.

### 12.3 Variant C

Trainer emits sidecar bytes; A1 archive untouched; combined archive = A1 + sidecar.

---

## 13. Stack-of-stacks composition matrix

| Composition | Description | Predicted ΔS | Class | Comment |
|---|---|---|---|---|
| Variant A standalone | grayscale_lut + UNIWARD #05 + grayscale-LUT #06 (substrate's mechanism) | `NULL pending substrate alpha anchor` | substrate_engineering + bolt-on | conditional on substrate L1 promotion |
| Variant A ⊕ DP1 codebook init | DP1 pretrained-driving-prior codebook initializes grayscale_lut decoder | `additive small ~ -0.002` | composition | STRONG_STACK per shared scorer framework |
| Variant B standalone (UNIWARD-latent-residual) | sane_hnerv + UNIWARD-latent-residual #05 | `[-0.0007, -0.002]` | bolt-on | within-class plateau |
| Variant B (LUT-on-HNeRV-stem) | sane_hnerv + grayscale-LUT-on-HNeRV-stem #06 | UNKNOWN; high distortion risk | substrate_engineering | Lane MM v2 anti-pattern |
| Variant C standalone | A1 + UNIWARD-residual-sidecar | `NULL pending feasibility audit` | sidecar bolt-on | frontier-protecting at best |
| Variant A ⊕ ATW v2 chroma | grayscale_lut substrate + ATW v2 chroma bolt-on (per companion memo #1 Variant B) | UNKNOWN; both target color/chroma axes | likely REDUNDANT | probe needed |
| Variant C ⊕ companion-memo-#1 Variant A (NSCS03) | A1 sidecar + NSCS03 standalone replacement | INVALID — A1 sidecar and NSCS03 replacement are MUTUALLY EXCLUSIVE deployment paths | N/A | A1 ≠ NSCS03 (different substrates) |

---

## 14. Pipeline-of-pipelines

Each variant standalone is its own pipeline (substrate trainer → archive → inflate → auth-eval). Composition with DP1 codebook init or ATW v2 chroma bolt-on is conditional on sister-subagent landings.

---

## 15. Probe-disambiguator strategy (Catalog #125 Hook 6)

### 15.1 Variant A

Probe: `tools/probe_grayscale_lut_alpha_anchor_score.py` (PROPOSED, $0; ~5 min). Runs grayscale_lut substrate at smoke epoch count (e.g., 100ep) on synthetic / minimal-pair input; outputs predicted alpha anchor score band. If predicted band ≤ 0.21, grayscale_lut substrate IS class-distinct from PR106; UNIWARD #05 bolt-on becomes viable.

### 15.2 Variant B

Probe: `tools/probe_uniward_latent_residual_savings.py` (PROPOSED, $0; ~5 min). Compute empirical wavelet decomposition of sane_hnerv's per-pair latent residuals on contest video pairs; estimate UNIWARD-allocated bit savings via the wavelet-domain texture-cost analysis. If predicted savings ≥ 8% of latent-stream bytes, bolt-on viable; if < 5%, bolt-on falls below noise floor (DEFER).

### 15.3 Variant C

Probe: `tools/probe_a1_uniward_sidecar_feasibility.py` (PROPOSED, $0; ~30 min). Analyze A1's per-pair reconstructed RGB error distribution on contest video; estimate UNIWARD-residual-sidecar potential distortion-axis savings at fixed bit budget; compare to rate-axis cost. If net-effect potentially -0.001 to -0.005, sidecar viable.

---

## 16. Cargo-cult audit per assumption

| Assumption | HARD-EARNED or CARGO-CULTED | Justification |
|---|---|---|
| "UNIWARD is canonical for steganalysis-driven texture-cost functions" | HARD-EARNED | Fridrich-Holub 2014; published frontier; multiple steganography competitions won |
| "Grayscale-LUT is canonical for analog-mask substitution" | HARD-EARNED | Selfcomp PR#56 paradigm; 0.38 anchor IS proof-of-principle |
| "PR106 has a mask channel" | CARGO-CULTED-FALSIFIED | PV-1 empirical verification; PR106 = HNeRV with NO separate mask channel |
| "grayscale_lut substrate has a mask channel" | PARTIALLY CARGO-CULTED | Substrate USES CLASS_TARGETS structurally (Gaussian-softmax-LUT) but IS NOT a mask codec per L5 PASS. Mask-class-internal-trainable mechanism present but not as separate archive channel. |
| "sane_hnerv has a mask channel" | CARGO-CULTED-FALSIFIED | sane_hnerv L5 PASS "NOT a mask codec"; same HNeRV-family as PR106 |
| "A1's HNeRV variant internally handles masks" | PARTIALLY HARD-EARNED, UNCERTAIN | A1 IS the canonical 0.192848 frontier; internal mask handling depends on specific A1 variant; substrate-extension feasibility audit required |
| "PR102 CUDA-CPU drift -0.0330 extrapolates to grayscale_lut/sane_hnerv/A1 with bolt-ons" | CARGO-CULTED | T2 council Q2.5 verdict: PR102 drift is empirical for PR102 specifically; MAY NOT generalize. Use Shannon first-principles. |
| "UNIWARD-latent-residual on sane_hnerv will produce sub-cluster savings" | CARGO-CULTED | Conservative Shannon estimate is -0.0007 to -0.002 (below noise floor); within-cluster |
| "LUT-on-HNeRV-stem will reduce decoder bytes without sacrificing distortion" | CARGO-CULTED | Lane MM v2 anti-pattern PROVES bolt-on LUT regresses; substrate must be trained-from-scratch WITH LUT (substrate-engineering) |
| "A1 sidecar will not regress the frontier" | CARGO-CULTED-UNCERTAIN | A1 IS the canonical frontier; sidecar bytes ADD rate-cost; distortion-axis benefit UNCERTAIN per Catalog #220 operational mechanism requirement |
| "Reformulating Lanes #05+#06 brings technique class back into the active queue" | TRUE-BUT-CONDITIONAL | Technique class IS preserved; but reformulation is BLOCKED on substrate readiness; the BLOCKING is the actual cost, not the technique |

---

## 17. Dykstra-feasibility verdict on predicted bands (Catalog #296 — BOTH CPU + CUDA axes)

### 17.1 Variant A — CPU axis

CPU-axis polytopes:
1. **Substrate-anchor polytope**: PENDING grayscale_lut L0 → L1 promotion; band [0.20, 0.25] CPU per substrate's own reactivation criteria.
2. **UNIWARD rate-axis polytope**: FEASIBLE per Fridrich-Holub theorem; bit savings bounded by texture-cost map entropy. Expected savings 5-10% on grayscale stream → ΔS rate-axis ~`-0.001 to -0.003`.
3. **CPU SegNet/PoseNet polytopes**: FEASIBLE per Selfcomp PR#56 paradigm precedent; assumes substrate transfers from Selfcomp's training distribution to contest video.

**Intersection** (conditional on substrate L1): NON-EMPTY for band `[substrate_anchor - 0.003, substrate_anchor]`. If substrate_anchor = 0.21, band ≈ `[0.207, 0.21]` — well above A1's 0.192848 frontier (within-cluster + above).

**Dykstra-feasibility verdict on Variant A CPU**: PENDING substrate readiness; even conditional on success, predicted band ABOVE A1 frontier — frontier-protecting at best, NOT frontier-extending.

### 17.2 Variant A — CUDA axis

CUDA-axis polytopes mirror CPU + decode-substrate-numeric drift. CONDITIONAL on substrate readiness; predicted band same magnitude as CPU band with PR102-style drift bounds.

**Dykstra-feasibility verdict on Variant A CUDA**: PENDING; same magnitude as CPU.

### 17.3 Variant B — CPU axis

CPU-axis polytopes:
1. **Substrate-anchor polytope**: PENDING sane_hnerv L0 → L1 promotion (Phase 2 dispatch in flight).
2. **UNIWARD-latent-residual rate-axis polytope**: FEASIBLE but bounded by `25 × 0.10 × 15849 / 37545489 ≈ -0.0011` — within noise floor.
3. **LUT-on-HNeRV-stem distortion-axis polytope**: HIGH-RISK per Lane MM v2 anti-pattern.

**Intersection**: NON-EMPTY but band collapses to `[-0.001, -0.002]` (sub-cluster but within noise floor).

**Dykstra-feasibility verdict on Variant B CPU**: PASSES with marginal savings; within-cluster.

### 17.4 Variant B — CUDA axis

Same polytopes; same within-cluster verdict.

### 17.5 Variant C — CPU axis

CPU-axis polytopes:
1. **A1 baseline polytope**: A1 CPU = 0.192848 (canonical frontier).
2. **Sidecar rate-cost polytope**: +`25 × 5000 / 37545489 ≈ +0.00333` rate-axis (5 KB sidecar).
3. **Sidecar distortion-axis polytope**: UNCERTAIN; UNIWARD-allocated residual could correct SegNet boundary-pixel argmax mistakes.

**Intersection**: PENDING feasibility audit; net-effect band could be `[0.190, 0.200]` (frontier-protecting to frontier-extending) IF distortion-axis benefit ≥ +0.003 magnitude.

**Dykstra-feasibility verdict on Variant C CPU**: PENDING feasibility audit; net-effect potentially frontier-extending but HIGH UNCERTAINTY.

### 17.6 Variant C — CUDA axis

Same polytopes; same PENDING.

---

## 18. Observability surface (per CLAUDE.md max-observability + Catalog #305)

### 18.1 Variant A

Per-epoch metrics: `loss_total`, `loss_seg`, `loss_pose`, `loss_rate`, `uniward_bytes_saved`, `grayscale_field_texture_variance`, `archive_bytes_estimate`.

Auth-eval JSON per Catalog #226 + #127 + #249 (no `_cuda` filename for CPU eval). Sidecar emission `experiments/results/grayscale_lut_uniward_*/uniward_diagnostic.json` with per-pair UNIWARD cost-map statistics for forensic review.

### 18.2 Variant B

Per-epoch metrics: substrate's existing + `uniward_latent_residual_bytes_saved`, `wavelet_residual_entropy_bits_per_dim`, `lut_decoder_stem_distortion_overhead`.

### 18.3 Variant C

Sidecar-train metrics: `sidecar_bytes`, `sidecar_distortion_savings_seg`, `sidecar_distortion_savings_pose`, `net_score_estimate`.

A1 baseline regression guard: after every sidecar epoch, verify A1 standalone score UNCHANGED (sidecar must not modify A1 archive bytes).

---

## 19. 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence at design time |
|---|---|
| (1) UNIQUENESS | Three variants target distinct substrate classes (grayscale_lut analog-LUT class; sane_hnerv HNeRV class; A1 frontier-substrate class). Distinct from PR106-as-substrate failure mode. |
| (2) BEAUTY + ELEGANCE | All variants stay within HNeRV parity L7 bolt-on budget (≤350 LOC) OR explicitly tag substrate_engineering (Variant B LUT-on-HNeRV-stem). |
| (3) DISTINCTNESS | Variant A uses grayscale field texture; Variant B uses latent residual wavelet; Variant C uses RGB residual. Three distinct attack surfaces. |
| (4) RIGOR | PV-1 through PV-8 + per-variant Dykstra-feasibility on BOTH axes + cargo-cult audit per Catalog #303 + probe-disambiguator per Catalog #125 Hook 6. |
| (5) OPTIMIZATION PER TECHNIQUE | UNIWARD canonical recipe (Fridrich-Holub 2014); grayscale-LUT canonical recipe (Selfcomp PR#56). Canonical helpers `score_pair_components` per Catalog #164. |
| (6) STACK-OF-STACKS-COMPOSABILITY | §13 declares 7 composition options with orthogonality analysis. |
| (7) DETERMINISTIC REPRODUCIBILITY | UNIWARD encoding deterministic per fixed cost-map; LUT byte-stable. |
| (8) EXTREME OPTIMIZATION + PERFORMANCE | Per Catalog #270 umbrella: Tier 1 (autocast_fp16, TF32, torch.compile, no_grad) via substrates' existing trainers per Catalog #172/#178/#179/#180. UNIWARD is rate-axis only (no GPU cost). |
| (9) OPTIMAL MINIMAL CONTEST SCORE | Per T2 council Q2 verdict, CPU axis is leaderboard axis. Variant A predicted band ABOVE A1 (frontier-protecting at best). Variant B predicted band within-cluster. Variant C predicted band straddles frontier. NONE projected as frontier-extending with high confidence per Shannon first-principles. |

---

## 20. Cost estimate + dispatch readiness

| Phase | Cost | Time | Dispatch-ready? |
|---|---|---|---|
| Variant C — A1 substrate-extension feasibility audit | $0 | ~3-5 hr editor | YES (no Modal dispatch required) |
| Variant A — grayscale_lut L0 → L1 promotion via alpha anchor | $5-15 Modal A100 | ~2-6 hr | CONDITIONAL on alpha sane_hnerv anchor at ≤0.21 (per substrate's own reactivation) |
| Variant A — UNIWARD #05 bolt-on smoke (after L1 promotion) | $5 Modal T4 100ep paired | ~15 min | CONDITIONAL on Variant A L1 promotion |
| Variant A — UNIWARD #05 bolt-on Modal A100 1000ep paired | $10-30 Modal A100 | ~2-6 hr | CONDITIONAL on smoke green + 5/5 council PROCEED |
| Variant B — sane_hnerv L1 promotion via Phase 2 dispatch | $5-15 Modal A100 | ~2-6 hr | IN FLIGHT per subagent waves |
| Variant B — UNIWARD-latent-residual smoke | $5 Modal T4 100ep paired | ~15 min | CONDITIONAL on sane_hnerv L1 + probe-disambiguator verdict ≥ 5% savings |
| Variant B — LUT-on-HNeRV-stem (substrate_engineering) | $10-30 substrate-engineering build + $5 smoke | ~4-8 hr | LOW PRIORITY per Lane MM v2 anti-pattern |
| Variant C — UNIWARD-residual-sidecar (after feasibility audit) | $5-15 sidecar train + $5 paired smoke | ~2-4 hr | CONDITIONAL on feasibility audit POSITIVE |

**Total envelope inclusive of probes**: $30-100 across all three variants.

---

## 21. Reactivation criteria + op-routables

### 21.1 Variant A reactivation criteria

1. grayscale_lut substrate L0 → L1 promotion via alpha sane_hnerv anchor at ≤0.21 (per substrate's `__init__.py` lines 57-62).
2. Post-anchor diagnostic flags rate-axis headroom ≥5% (per grayscale_lut reactivation criteria (a)).
3. UNIWARD #05 bolt-on integration test passes (encoded archive roundtrips byte-stable).
4. Paired-CPU+CUDA smoke per Catalog #167 lands at substrate_anchor - 0.001 to - 0.003 (UNIWARD savings).
5. 5/5 council PROCEED on smoke result.

### 21.2 Variant B reactivation criteria

1. sane_hnerv substrate L0 → L1 promotion via Phase 2 dispatch.
2. UNIWARD-latent-residual probe (§15.2) verdict ≥ 8% wavelet-residual savings.
3. Bolt-on integration test passes.
4. Paired-CPU+CUDA smoke lands at sane_hnerv_anchor - 0.0007 to - 0.002.
5. 5/5 council PROCEED (HIGH within-class plateau risk acknowledged in deliberation).

### 21.3 Variant C reactivation criteria

1. A1 substrate-extension feasibility audit returns POSITIVE (audit document at `.omx/research/a1_substrate_extension_feasibility_audit_<UTC>.md`).
2. UNIWARD-residual-sidecar probe (§15.3) verdict net-effect ≤ -0.001.
3. Sidecar integration test passes (A1 standalone score UNCHANGED; combined archive parses cleanly).
4. Paired-CPU+CUDA smoke lands at A1 + net-effect within feasibility-audit band.
5. 5/5 council PROCEED (frontier-risk acknowledged; per Catalog #220 operational mechanism declared and verified).

### 21.4 Op-routables (for operator decision queue)

1. **OR-1**: Approve Variant C feasibility audit ($0 editor work; ~3-5 hr). Cheapest first action; informs Variant A vs Variant B sequencing.
2. **OR-2**: Approve grayscale_lut substrate alpha anchor smoke ($5-15 Modal A100; pre-requisite for Variant A). Coordinate with sister-subagent waves that may already be dispatching grayscale_lut.
3. **OR-3**: Approve probe-disambiguator scripts (§15.1, §15.2, §15.3 PROPOSED; ~150 + ~120 + ~200 LOC; $0). Probe scripts are the cheapest signal across all three variants.
4. **OR-4**: Resolve "Variant A predicted band ABOVE A1 frontier" question per CLAUDE.md "Forbidden premature KILL" + "abandon-within-class" directive — is the operator willing to fund Variant A dispatch given the frontier-protecting-at-best Dykstra-feasibility verdict? Council-grade decision.
5. **OR-5**: STRICT preflight gate `check_substrate_pivot_reformulation_has_dykstra_feasibility_on_both_axes` (PROPOSED, ~80 LOC; Catalog #298+ via canonical-claim). Refuses substrate-pivot reformulation memos that don't declare Dykstra-feasibility on BOTH CPU + CUDA axes per Catalog #296 + T2 council Q2 verdict.
6. **OR-6**: Lane MM v3 (SegMap-trained-from-scratch with grayscale-LUT) per resurrection audit Tier 1 #1.5 reactivation queue #7 — separate from Variant B LUT-on-HNeRV-stem (different substrate; same technique class). Sequence Variant A vs Lane MM v3 based on cost + dispatch-readiness urgency.

---

## 22. Cross-references

- Resurrection audit §1.4 + #4, #5 reactivation-queue rows: `.omx/research/resurrection_audit_20260516.md`
- Original PR106 falsification memo: `.omx/auto_memory_snapshot_20260504T230223Z/feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md`
- Lane #05 + Lane #06 original designs: `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_05_*.md` + `revival_plan_06_*.md`
- PR106 archive structure: `experiments/extract_pr106_decoder.py` (commit 45149f21)
- grayscale_lut substrate: `src/tac/substrates/grayscale_lut/`
- sane_hnerv substrate: `src/tac/substrates/sane_hnerv/`
- A1 runtime adapter: `src/tac/substrates/a1/`
- T2 council Q2 verdict (CPU axis is leaderboard axis): `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`
- Lane MM v2 historical attempt (Lane #06-like): resurrection audit §1.5
- Sister companion memo: `.omx/research/tier_1_resurrection_4_pr101_compressai_balle_reformulated_full_stack_design_20260516.md` (Tier 1 #4 PR101 CompressAI Ballé reformulated)
- ATW v2 design memo (alternative comparison axis): `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`
- CLAUDE.md non-negotiables: UNIQUE-AND-COMPLETE-PER-METHOD, HNeRV parity discipline L7, Forbidden premature KILL, Apples-to-apples evidence discipline, Submission auth eval BOTH CPU AND CUDA, Predicted band has Dykstra-feasibility check (Catalog #296), 9-dim checklist evidence section (Catalog #294), Canonical-vs-unique decision section (Catalog #290), Cargo-cult audit per assumption (Catalog #303), Observability surface (Catalog #305)

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Variant A (grayscale_lut) | Variant B (sane_hnerv) | Variant C (A1-extension) |
|---|---|---|---|
| Substrate architecture | ADOPT canonical (grayscale_lut Selfcomp PR#56 paradigm) | ADOPT canonical (sane_hnerv HNeRV-family) | EXTEND canonical (A1 frozen + sidecar) |
| UNIWARD cost function | ADOPT canonical (Fridrich-Holub 2014) | ADOPT canonical (applied to latent residual wavelet) | ADOPT canonical (applied to RGB residual) |
| Grayscale-LUT mechanism | ADOPT canonical (substrate's central mechanism) | FORK to LUT-on-HNeRV-stem (substrate engineering on sane_hnerv) | N/A (sidecar uses UNIWARD only, not LUT) |
| Archive grammar | EXTEND grayscale_lut GLV1 → GLV2 (UNIWARD-encoded grayscale section) | EXTEND sane_hnerv archive with bolt-on sections | EXTEND A1 archive with backward-compatible sidecar |
| Inflate runtime | EXTEND grayscale_lut inflate (~50 LOC UNIWARD decode) | EXTEND sane_hnerv inflate | EXTEND A1 inflate (sidecar-aware wrapper) |
| Score-aware loss | ADOPT canonical `score_pair_components` per Catalog #164 | ADOPT canonical | UNIQUE (sidecar-specific Lagrangian; A1 frozen) |
| Tier-1 engineering | ADOPT canonical (autocast_fp16, TF32, torch.compile, no_grad) | ADOPT canonical | N/A (A1 frozen; sidecar train uses canonical) |
| EMA | ADOPT canonical EMA(0.997) | ADOPT canonical | ADOPT canonical for sidecar |
| Variant adjudication mechanism | UNIQUE (substrate alpha anchor at ≤0.21 is the gate) | UNIQUE (sane_hnerv L1 promotion + probe verdict ≥ 8%) | UNIQUE (feasibility audit + probe net-effect ≤ -0.001) |
| Cross-axis CPU+CUDA evaluation | ADOPT canonical (paired-CPU+CUDA per CLAUDE.md non-negotiable) | ADOPT canonical | ADOPT canonical |
| Reactivation criteria | UNIQUE (5-criterion sequence per §21.1) | UNIQUE (5-criterion sequence per §21.2) | UNIQUE (5-criterion sequence per §21.3) |
| Within-class plateau risk acknowledgment | EXPLICIT (per §17.1 Dykstra verdict + cargo-cult audit) | EXPLICIT (per §17.3) | EXPLICIT (per §17.5) |

The canonical Ballé hyperprior FORK question (Memo #1) does not apply here; this memo's canonical-vs-unique decision is per technique (UNIWARD-canonical adoption; grayscale-LUT-canonical adoption; substrate-engineering forks per variant).

---

**Predicted ΔS bands SUMMARY** (with Dykstra-feasibility verdicts per axis):

| Variant | CPU band [prediction] | CPU Dykstra verdict | CUDA band [prediction] | CUDA Dykstra verdict |
|---|---|---|---|---|
| Variant A (grayscale_lut + UNIWARD #05 + grayscale-LUT #06) | `[substrate_anchor - 0.003, substrate_anchor]` (substrate_anchor PENDING; conservative ≈ 0.21) | PENDING substrate readiness; conditional on L1 promotion | same magnitude as CPU | PENDING |
| Variant B (sane_hnerv + UNIWARD-latent-residual) | `[-0.0007, -0.002]` from sane_hnerv anchor | PASSES (within-cluster) | same magnitude as CPU | PASSES |
| Variant C (A1 + UNIWARD-residual-sidecar) | `[A1 + net-effect band]` (net-effect PENDING feasibility audit; -0.005 to +0.003 envelope) | PENDING feasibility audit; potentially frontier-extending | same | PENDING |

**Reactivation status**: ALL THREE variants in DESIGN-ONLY; reactivation gated on (Variant A) grayscale_lut substrate L1 promotion; (Variant B) sane_hnerv L1 promotion + probe verdict; (Variant C) A1 substrate-extension feasibility audit.

**Per resurrection audit + T2 council Q2 verdict + abandon-within-class directive**: NONE of the three variants is projected as frontier-extending with high confidence per Shannon first-principles. The reformulation IS technically defensible per UNIQUE-AND-COMPLETE-PER-METHOD but has HIGH within-class plateau risk. Per CLAUDE.md "Forbidden premature KILL", no KILL is proposed; reactivation criteria are explicitly enumerated.

---

*End of memo. ~5400 words. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable, no KILL is proposed. Per Catalog #290 canonical-vs-unique decision per layer + Catalog #294 9-dim checklist + Catalog #296 Dykstra-feasibility BOTH axes + Catalog #297 signal-axis reversibility (N/A — no signal destruction proposed; UNIWARD is rate-axis only; grayscale-LUT operates within trained substrate) + Catalog #303 cargo-cult audit per assumption + Catalog #305 observability surface, all required sections present.*
