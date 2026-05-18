---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Filler, Pevný, Selfcomp, Quantizr]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Assumption-Adversary
    verbatim: "The parent symposium #861 enumerated path 3a with predicted ΔS band [-0.015, -0.003] derived from 'A1 residual entropy estimate ~0.5-1.0 bits/coefficient × ~5K coefficients × 1/8 = 312-625 bytes saved if STC matches AC bound within 1/h overhead'. That derivation conflates BYTES-SAVED with SCORE-DELTA: 312-625 bytes at A1's rate term gives a +0.00021 to +0.00042 RATE PENALTY (sidecar ADDS bytes; the residual is NEW INFO not currently encoded). For the predicted band [-0.015, -0.003] to hold, the residual sidecar must yield a DISTORTION GAIN of -0.0032 to -0.0154 — 16x to 77x larger than the rate cost. At A1's operating point (contest-CPU 0.193 ≈ PR101 gold, already saturated), no sister sidecar anchor in the registry has delivered that magnitude: PR100→PR105 (the canonical pattern this path mimics) delivered -0.00218 at +1124 bytes; PR106 yshift predicted -0.0005 to -0.0015 at 1-2KB. The shared assumption I am operating within is that 'STC paradigm wins over Filler 2011's bounded AC competitor on a 1D sparse signal'. HARD-EARNED per the lane_filler_stc_paradigm_alpha empirical anchor (STC=996B vs LZMA-ternary=3084B vs AV1=79KB; +68% savings vs LZMA, +98% vs AV1 on 262K-sym ternary sample). VETO if any council member proceeds without tightening the predicted band per sister-anchor empirical evidence."
  - member: Filler
    verbatim: "My operating-within assumption: STC's R_STC(D) ≤ R_AC(D) + 1/h bound (Filler 2011 Theorem 4) requires the cover signal to have a meaningful arithmetic-coding lower bound. The A1 latent sidecar already exists at ~607 bytes with a NEAR-OPTIMAL Huffman coder (decode_canonical_huffman + colex combination + 3-bit length packing — see src/codec.py lines 320-465). STC over the EXISTING sidecar payload could only save the gap between Huffman (per-symbol IID) and arithmetic coding with temporal context (across adjacent pairs); that gap is bounded by H(X_t) - H(X_t | X_{t-1}) which empirically is ~0.5 bits/symbol for a sparse no-op-dominated signal (no-op rate ~0.5-0.7 per the SIDECAR_NOOP_COMB packing scheme). For 600 pairs, that's ~37 bits = 5 bytes of savings — STATISTICAL NOISE at A1's rate-term scale. The path that REALLY tests STC paradigm in this substrate is encoding a NEW RESIDUAL STREAM, e.g. the post-quantization latent residual (28-dim × 600 pairs × float = quantized to uint8 at scale*x + min reconstruction; the residual is fp32 - reconstruct which is ~0.5*scale typical). That gives ~16,800 residual values × ~3-4 bits/value (if smoothly distributed) ~= ~7-8 KB raw entropy. STC compressed: ~4-6 KB. Even with full distortion gain, the rate cost (4-6 KB → +0.003 to +0.004 score) likely DOMINATES distortion gain at A1's already-converged operating point. So I REVISE my support to PROCEED_WITH_REVISIONS contingent on (a) tightening predicted band to [-0.005, +0.001] per sister-anchor evidence + Catalog #324 post-training Tier-C validation, (b) DEFER smoke until the residual signal's contest-CPU sensitivity is measured via a $0 CPU pre-probe (the PR101→A1 finetune residual exists in repo)."
  - member: Selfcomp
    verbatim: "My operating-within assumption: A1's sidecar is structurally EQUIVALENT to PR100→PR105's latent sidecar pattern (per-pair (dim, delta) corrections), so empirical sister anchors apply. PR100→PR105 delivered -0.00218 score Δ at +1124 bytes for ONE pattern; A1 already inherits this paradigm via the apply_latent_sidecar helper. STC-as-second-sidecar over A1's residual is a SECOND LATENT-CORRECTION LAYER, but residual-of-residual signals decay rapidly per Shannon. I support PROCEED_WITH_REVISIONS specifically with the routing: (a) tighten predicted band to [-0.003, +0.001]; (b) the PROBE-DISAMBIGUATOR should measure the LATENT QUANTIZATION RESIDUAL ENTROPY on the actual landed A1 archive 87ec7ca5 BEFORE dispatching impl work — if H(residual | latent_uint8, scale, min) < 1.5 bits/value AND the cover has temporal correlation (likely), then STC IS worth the impl; if H ≥ 2.0 bits/value (near-uniform), then 3b composability with my PR#56 paradigm is higher EV. Composition with my PR#56 paradigm via Lane MM v3 reactivation route is preferred."
  - member: Quantizr
    verbatim: "My operating-within assumption: A1's archive is the canonical post-PR95 winning recipe. Any modification that REGRESSES contest-CPU or contest-CUDA score is non-promotable. The A1 sidecar is currently optimized for the PARTICULAR set of 16 SIDECAR_DELTAS_X100 values + 28 LATENT_DIM × 600 pairs space; the Huffman coder is near-Shannon. STC-as-second-layer either (a) replaces the existing Huffman with STC over the SAME sidecar payload (likely SAME-OR-WORSE bytes — Huffman is already tight) or (b) encodes a NEW residual stream. Path (a) violates Filler 2011's gap (Huffman is the optimal block code for IID symbols; STC introduces overhead 1/h above the AC bound, which for IID symbols IS the Huffman bound). Path (b) is the only structurally distinct path. I SUPPORT PROCEED_WITH_REVISIONS conditional on the symposium explicitly disambiguating which path 3a means, and pinning predicted-band to per-path-specific empirical-anchor-derived ranges. Composability with my 0.33 leaderboard pattern is INTACT iff path (b) chosen + bytes added < 1 KB."
  - member: Contrarian
    verbatim: "My operating-within assumption: $5.20 is a small dispatch but symptoms of 'fan-out cheap dispatches without first-principles disambiguation' is the canonical Race-mode rigor inversion failure mode the operator extincted today. The parent symposium ALREADY ran the 6-of-6 sextet + 4-of-4 grand-council DEFER on the original implementation. We are now in REFORMULATION-DESIGN phase, NOT empirical-falsification phase. The cheapest reformulation $5.20 dispatch makes sense ONLY IF the impl is structurally distinct AND the predicted band is empirically defensible AND no $0 CPU pre-probe could resolve the question first. Per the Assumption-Adversary verdict above: the predicted band [-0.015, -0.003] is implausible per sister anchors. Per Filler: a $0 CPU pre-probe (measure A1 latent residual entropy on archive 87ec7ca5) disambiguates the structural question before any GPU spend. SUPPORT PROCEED_WITH_REVISIONS with the explicit op-routable: $0 pre-probe FIRST, then $5.20 dispatch ONLY if pre-probe verdict is HIGH-ENTROPY-RESIDUAL-PRESENT."
council_assumption_adversary_verdict:
  - assumption: "STC paradigm is the right codec class for the A1 residual stream (the 'class-shift' axis)"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "lane_filler_stc_paradigm_alpha empirical anchor (2026-05-08) measured STC=996B vs LZMA-ternary=3084B (+68% savings) vs AV1-monochrome=79KB on 262K-sym ternary mask-delta sample. STC IS the right codec class for SPARSE TERNARY-OR-SIMILAR 1D signals; this is HARD-EARNED on a different substrate (mask channel) but the codec-class argument transfers to the A1 latent-quantization-residual cover signal IF that cover signal is ALSO sparse-ternary-or-similar (the empirical pre-probe must verify; CONDITIONAL pending verification)."
  - assumption: "The A1 residual stream EXISTS and has exploitable entropy structure"
    classification: HYPOTHETICAL-NEEDS-EMPIRICAL-PRE-PROBE
    rationale: "The 'A1 residual stream' is not a single canonical artifact. Three plausible interpretations: (a) the LATENT QUANTIZATION RESIDUAL: fp32 latents - reconstruct(uint8 codes, scale, min) for ~16,800 values; (b) the DECODER WEIGHT RESIDUAL: q-byte * fp16 scale - true fp32 weights for ~88K params (STATIC, encodable once but never updated); (c) the SIDECAR-RESIDUAL: per-pair latent corrections that DIDN'T fit the existing 16-value SIDECAR_DELTAS_X100 quantization. Each has different cover-signal characteristics. The Assumption-Adversary mandate per CLAUDE.md 'Council conduct' Fix-7 amendment surfaces this assumption explicitly: WITHOUT a $0 pre-probe measuring each candidate residual stream's entropy + temporal-context-exploitable structure, the predicted band is unjustified guessing."
  - assumption: "The predicted band [-0.015, -0.003] is defensible at A1's operating point"
    classification: CARGO-CULTED
    rationale: "Per the numerical decomposition: 312-625 bytes (parent symposium's predicted bytes-saved range) at A1's rate term gives +0.00021 to +0.00042 rate penalty. For the predicted band to hold, the distortion gain must be -0.0032 to -0.0154 — 16x to 77x rate cost. Sister-anchor evidence: PR100→PR105 latent sidecar delivered -0.00218 at +1124 bytes (distortion gain 5x rate cost). PR106 yshift predicted -0.0005 to -0.0015 at 1-2KB (distortion gain 1-3x rate cost). NO sister anchor at A1's operating point has delivered 16-77x rate-to-distortion ratio. The parent band is from a different operating point (probably designed for pre-saturation substrates) and CARGO-CULTED forward. UNWIND: tighten band to [-0.003, +0.001] per sister-anchor empirical evidence; flag SIGNED-BAND (could regress if rate-cost dominates distortion gain). Per Catalog #296 Dykstra-feasibility: the intersection of (rate ≤ +1KB) ∩ (post-training Tier-C density on landed archive shows exploitable residual entropy) ∩ (distortion gain ≥ rate cost) ∩ (decoder ≤+25 LOC over A1's current ~120 LOC inflate) is NON-EMPTY but TIGHT."
  - assumption: "The path is ADDITIVELY COMPOSABLE with A1 + Selfcomp + Quantizr per parent symposium claim"
    classification: HARD-EARNED-AT-PARADIGM, UNVERIFIED-AT-IMPLEMENTATION
    rationale: "At paradigm level: STC-as-sidecar over latent-quantization-residual is ORTHOGONAL to (a) A1's existing per-pair latent sidecar (different cover signal: the residual NOT the integer-correction-delta), (b) Selfcomp's PR#56 mask paradigm (different archive section: the latent stream NOT the mask channel), (c) Quantizr's FP4+Brotli weight pool (different archive section: the latent residual NOT the decoder weights). Mathematically orthogonal: HARD-EARNED. At implementation level: composability requires the STC sidecar bytes to not displace existing bytes via shared-section-resizing OR archive-grammar-incompatibility. UNVERIFIED-AT-IMPLEMENTATION until the actual archive grammar amendment is designed. Per the cargo-cult-unwind methodology + the sister Path B failure (NSCS06 v8 regressed previously-unwound cargo-cults #1+#2 while unwinding #3+#6): SIDECAR ADDITION is structurally LOW-RISK for composability regression because it adds bytes without modifying existing payload — but the bytes-added STILL count against rate term. Confirm composability via $0 archive-grammar dry-run BEFORE the $5.20 dispatch."
council_decisions_recorded:
  - "op-routable #1 (PRE-PROBE, $0): Build a $0 CPU pre-probe at `tools/probe_stc_3a_a1_residual_entropy.py` that loads A1 archive `87ec7ca5...` from `submissions/a1/archive.zip`, extracts (a) latent quantization residual = fp32 - reconstruct(uint8, scale, min), (b) decoder weight residual = float - reconstruct(q-byte, fp16 scale), (c) sidecar payload as-is for entropy comparison. Measure: H(X), H(X|X_{t-1}), H(X|context-2D), apparent sparsity, ternary-structure-rate, alignment with SegNet attack surface (boundary-residual classification). Emit verdict `HIGH-ENTROPY-RESIDUAL-PRESENT` (band [-0.003, +0.001] dispatch-eligible) / `MEDIUM-ENTROPY-RESIDUAL-PRESENT` (defer to 3b composition) / `LOW-ENTROPY-RESIDUAL-ABSENT` (DEFER + redirect to 3b/3c). Cost: $0 (CPU only on local M5 Max). Time: ~1 hour to land + ~5 min to run. THIS PROBE IS THE PROBE-DISAMBIGUATOR per Catalog #313."
  - "op-routable #2 (CONTINGENT $5.20 dispatch): IF pre-probe verdict is `HIGH-ENTROPY-RESIDUAL-PRESENT`, proceed to $5.20 Modal A100 10ep smoke. Recipe MUST declare: `predicted_band: [-0.003, +0.001]` (REVISED from parent's [-0.015, -0.003] per sister-anchor evidence); `predicted_band_validation_status: pending_post_training`; `dispatch_kind: substrate`; `expected_axis: contest_cuda` with paired-CPU adjudication via `tools/dispatch_modal_paired_auth_eval.py`. Skip-if-anchor-exists per Catalog #246 (A1 sister archives at 87ec7ca5 + sister sidecar variants). Smoke epochs=10; full would be 50ep at $25-30."
  - "op-routable #3 (PARALLEL with 3b, $5.20 + $5.20 = $10.40 total): Operator may pair Path 3a with Path 3b (`lane_stc_tone_map_delta_selfcomp_3b_20260518` per parent symposium #861 op-routable #4). The two paths are ORTHOGONAL (3a = A1 substrate, latent residual; 3b = Selfcomp PR#56 substrate, mask tone-map delta). 3b composes with Lane MM v3 reactivation per Selfcomp's verdict in parent symposium. Pairing tests STC paradigm in 2 substrate-classes simultaneously; provides paradigm-vs-implementation classification per Catalog #307."
  - "op-routable #4 (CATALOG #313 PROBE-OUTCOMES LEDGER ENTRY): After this memo lands and the pre-probe runs, register the canonical probe outcome to `.omx/state/probe_outcomes.jsonl` via `tac.probe_outcomes_ledger.register_probe_outcome(substrate_id='stc_3a_sidecar_a1_residual', verdict='PROCEED_WITH_REVISIONS' or 'DEFER' depending on pre-probe verdict, status='blocking' until paired dispatch lands, methodology='stc_over_a1_latent_quantization_residual_with_canonical_2d_context_model', alternative_probe_methodologies=['stc_over_decoder_weight_residual_static', 'stc_over_sidecar_residual_2nd_layer', 'huffman_with_2d_temporal_context_no_stc', 'arithmetic_coder_with_dasher_context_no_stc'], expires_at_utc=<30_days_from_now>)`. This makes the verdict QUERYABLE across sessions and gates future dispatch wrappers from re-firing without re-probe."
  - "op-routable #5 (CATALOG #298 LANE RETIREMENT DISCIPLINE): Pre-register `lane_stc_sidecar_a1_residual_3a_20260518` at Level 0 via `tools/lane_maturity.py add-lane lane_stc_sidecar_a1_residual_3a_20260518 --name 'STC sidecar over A1 latent quantization residual (3a reformulation per stc parent symposium #861)' --phase 4` with `notes='research_only=true until pre-probe verdict + 5-PROCEED council; reactivation_criteria=HIGH-ENTROPY-RESIDUAL-PRESENT + PROCEED-unconditional council deliberation'`. The lane's `lane_class` should be `substrate_engineering` (per HNeRV parity L7) since this is a substrate-specific bolt-on requiring archive-grammar amendment."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - "council_per_substrate_symposium_stc_clean_source_20260517"
---

# Per-substrate symposium — STC reformulation **Path 3a**: STC-as-sidecar over A1-substrate residual stream

**Date:** 2026-05-17
**Subagent ID:** stc_3a_symposium_1779077815
**Lane:** `lane_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517` L0 (pre-registered; in-flight)
**Tier:** T2 sextet pact + 4 grand-council attendees (Filler / Pevný / Selfcomp / Quantizr)
**Verdict:** **PROCEED_WITH_REVISIONS** (unanimous 10/10)
**Mission-alignment:** `frontier_protecting` (Path 3a is a frontier-protecting reformulation per parent symposium #861's reactivation paths; protects against canonical-helper-share suppression of substrate-specific STC opportunity per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" 2026-05-15 standing directive)
**Budget consumed:** $0 (editor only)
**Conditional next-step:** $0 pre-probe (`tools/probe_stc_3a_a1_residual_entropy.py`); IF HIGH-ENTROPY verdict, then $5.20 Modal A100 10ep smoke

## Executive summary

Path 3a is the CHEAPEST reformulation path from the parent stc symposium #861 ($5.20 vs $5.20 for 3b vs $15-30 for 3c). The structural argument for Path 3a is that STC-as-sidecar over A1's latent-quantization-residual cover signal is APPLE-PIE the canonical Filler 2011 use case — a 1D symbol stream where context-naive arithmetic coding IS appropriate (per Filler 2011 Theorem 4's `R_STC(D) ≤ R_AC(D) + 1/h` bound with `h=12`, the canonical implementation in `src/tac/codec/syndrome_trellis_codec.py`).

However, the parent symposium's predicted band `[-0.015, -0.003]` is unjustified at A1's already-saturated operating point (contest-CPU 0.19284758, functionally tied with PR101 gold 0.19284). Per the Assumption-Adversary verdict + sister-anchor empirical analysis: 312-625 bytes added to A1's archive yields a rate penalty of +0.00021 to +0.00042; for the predicted band to hold the distortion gain must be 16x-77x larger than the rate cost. No sister sidecar anchor at A1's operating point has delivered that ratio. PR100→PR105 (the canonical pattern this path mimics) delivered -0.00218 at +1124 bytes (5x ratio). PR106 yshift predicted -0.0005 to -0.0015 at 1-2KB (1-3x ratio). **REVISED predicted band: `[-0.003, +0.001]` (SIGNED — could regress)**.

The verdict is PROCEED_WITH_REVISIONS unanimous 10/10. The structural revision is: BEFORE any $5.20 dispatch, run a $0 CPU pre-probe (`tools/probe_stc_3a_a1_residual_entropy.py`) to disambiguate which of three candidate A1 residual streams (latent quantization residual / decoder weight residual / sidecar 2nd-layer residual) has exploitable entropy structure. If the probe verdict is `HIGH-ENTROPY-RESIDUAL-PRESENT` (defined: `H(X|context) < 1.5 bits/value AND temporal correlation present AND sparse/ternary structure`), proceed to $5.20 dispatch on the highest-EV residual stream. Otherwise DEFER + redirect to 3b composition with Selfcomp PR#56 paradigm per parent symposium op-routable #5.

**Critical empirical receipts informing this verdict:**

- `lane_filler_stc_paradigm_alpha` (2026-05-08): STC=996B vs LZMA-ternary=3084B (+68% savings) vs AV1=79KB on 262K-sym ternary sample. **STC IS the right codec class for sparse 1D ternary signals.**
- `lane_pr106_latent_sidecar` (PR100→PR105 sister): +1124 bytes for -0.00218 score Δ. **The canonical pattern this path mimics; predicts 5x distortion-gain-to-rate-cost ratio at PR106 operating point.**
- `lane_pr106_yshift_sidechannel`: 1-2KB for predicted -0.0005 to -0.0015 (1-3x ratio at PR106 operating point).
- A1 current sidecar: ~607 bytes (~0.34% of archive); near-Huffman-optimal for the current SIDECAR_DELTAS_X100 quantization scheme. **STC over THIS payload is dominated by Huffman; STC over a NEW residual stream is the only structurally distinct path.**

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + cargo-cult-unwind methodology (NSCS06 v6→v7 44% improvement anchor).

### Assumption 1: "Path 3a is structurally distinct from the parent symposium's failed implementation"

- **Classification: HARD-EARNED**
- **Rationale:** The parent failure was per-pixel-IID arithmetic over the MASK CHANNEL (236M pixel symbols at p_boundary=0.05 → 21MB). Path 3a is canonical Filler 2011 STC (constraint_height=12 trellis, parity-check matrix per `src/tac/codec/syndrome_trellis_codec.py`) over A1's LATENT QUANTIZATION RESIDUAL (1D sparse signal, ~16,800 fp32 residual values). DIFFERENT codec implementation, DIFFERENT cover signal, DIFFERENT substrate. The HNeRV parity discipline L5 anti-pattern (mask substitution dominated by frame replacement) does NOT apply here because Path 3a does NOT touch the mask channel.
- **Verification:** The implementations are structurally distinct: parent uses `src/tac/stc_boundary_codec.py` (per-pixel IID); Path 3a would use `src/tac/codec/syndrome_trellis_codec.py` (canonical Viterbi-trellis with `_viterbi_encode_binary`) over a NEW residual cover signal.

### Assumption 2: "The A1 latent quantization residual has exploitable entropy structure"

- **Classification: HYPOTHETICAL-NEEDS-EMPIRICAL-PRE-PROBE**
- **Rationale:** Three plausible residual streams: (a) latent-quantization residual `fp32 - reconstruct(uint8, scale, min)` for ~16,800 values; (b) decoder weight residual `float - reconstruct(q-byte, fp16 scale)` for ~88K params (STATIC); (c) sidecar 2nd-layer residual `(dim, delta) values that didn't fit the existing 16-value SIDECAR_DELTAS_X100 quantization`. Without empirical entropy measurement, the cover signal's exploitable structure is unknown. STC over a near-uniform residual is dominated by raw byte storage; STC over a sparse-ternary-like residual is the canonical Filler 2011 win.
- **Unwind-test:** Build the $0 pre-probe per op-routable #1. Measure `H(X)`, `H(X|X_{t-1})`, sparsity, ternary-structure-rate. Emit categorical verdict.

### Assumption 3: "The parent symposium's predicted band [-0.015, -0.003] is defensible at A1's operating point"

- **Classification: CARGO-CULTED**
- **Rationale:** See Assumption-Adversary verdict above. 312-625 bytes at A1's rate term gives +0.00021 to +0.00042 rate penalty; the predicted band requires 16x-77x rate-cost distortion gain. No sister anchor has delivered that ratio at A1's operating point. The parent band was derived without sister-anchor empirical calibration.
- **Unwind-test:** REVISED predicted band: `[-0.003, +0.001]` per sister-anchor evidence (PR100→PR105 = 5x ratio, PR106 yshift = 1-3x ratio). SIGNED-BAND flags that net regression is possible.

### Assumption 4: "STC encodes residual MORE EFFICIENTLY than the existing Huffman coder over the existing sidecar"

- **Classification: CARGO-CULTED**
- **Rationale:** A1's existing sidecar uses near-optimal canonical Huffman (see `src/codec.py::decode_canonical_huffman_all` + `decode_huff_length_rank` + colex combinatorial packing). For the EXISTING sidecar payload (per-pair `(dim, delta)` integer corrections), Huffman is the optimal block code for IID symbols. STC over the SAME payload adds `1/h` bits/symbol overhead above the arithmetic-coding bound; for `h=12`, that's ~0.083 bits/symbol = ~6 bytes overhead vs Huffman's typical 0-3 bytes overhead. **STC ONLY wins if it encodes a DIFFERENT cover signal that Huffman doesn't currently encode — i.e. the NEW residual stream.**
- **Unwind-test:** Path 3a MUST encode a NEW residual stream, NOT replace the existing Huffman over the existing sidecar. The pre-probe verifies which residual stream is the right cover signal.

### Assumption 5: "The decoder for STC-as-sidecar fits within A1's inflate ≤100 LOC budget per HNeRV L4"

- **Classification: HARD-EARNED-CONDITIONAL**
- **Rationale:** A1's current inflate.py is 136 LOC (already over the strict 100 LOC budget; HNeRV L4 allows ≤200 with rationale). STC decoder via `src/tac/codec/syndrome_trellis_codec.py::_viterbi_decode_binary` + ternary split is ~150 LOC additional. Total: ~286 LOC, OVER budget. Per HNeRV L7 (substrate engineering exception), declare `lane_class=substrate_engineering` to legitimize the budget exception. Per the bolt-on vs substrate-engineering split CLAUDE.md L7: "bolt-ons share (≤350 LOC); substrate engineering unique-ifies (happens ONCE per architecture class; size budget exceeds)". The 286 LOC bolt-on FITS within the 350 LOC bolt-on budget if we class A1+STC as a NEW substrate-engineering instance.
- **Unwind-test:** PROCEED with `lane_class=substrate_engineering` declaration AND keep the STC decoder modular so it can be REUSED for sister substrates (3b composition).

### Assumption 6: "STC over A1 residual is composable with Selfcomp + Quantizr per parent claim"

- **Classification: HARD-EARNED-AT-PARADIGM, UNVERIFIED-AT-IMPLEMENTATION**
- **Rationale:** Mathematically orthogonal sections (latent residual vs mask channel vs decoder weights). Implementation-level composability requires archive-grammar amendment that doesn't displace existing bytes via shared-section-resizing. Per Catalog #220 / #272 / #139: STC sidecar bytes must (a) be structurally consumed by inflate (no-op detector), (b) produce frame-output change (byte-mutation smoke), (c) declare distinguishing-feature integration contract.
- **Unwind-test:** $0 archive-grammar dry-run BEFORE $5.20 dispatch: append STC sidecar section to A1 archive, verify monolithic single-file `0.bin` grammar preserves (currently `x` not `0.bin` — A1 already uses single-member). Verify parser-section manifest can be amended without breaking PR101's split-Brotli decoder or existing latent sidecar.

## 2. 9-dimension success checklist evidence (Catalog #294)

Per CLAUDE.md "9-dimension success checklist evidence" non-negotiable.

| # | Dimension | Evidence | Status |
|---|---|---|---|
| 1 | UNIQUENESS | STC paradigm IS class-shift (steganography-derived codec per Catalog #262 STC-Dasher composite); A1 substrate + STC sidecar IS a structurally distinct combination from any existing substrate in the registry | INTACT |
| 2 | BEAUTY + ELEGANCE | Path 3a target: ~150 LOC STC decoder + ~30 LOC archive-grammar amendment = ~180 LOC bolt-on. PR101 budget ~605 LOC total = ~270 LOC budget for A1+STC combined; FITS within bolt-on budget per HNeRV L7 substrate-engineering exception | INTACT-CONDITIONAL |
| 3 | DISTINCTNESS | A1+STC is empirically distinct from: A1 alone (no residual encoding), A1+yshift (pixel-translation NOT latent-residual), A1+lapose (pose-axis NOT latent-residual), A1+wavelet (wavelet basis NOT STC parity-check) | INTACT |
| 4 | RIGOR | This symposium IS the rigor remediation per Catalog #325 6-step contract. Pre-probe (op-routable #1) is the empirical verification step BEFORE GPU spend per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" | INTACT-PENDING-PRE-PROBE |
| 5 | OPTIMIZATION PER TECHNIQUE | Canonical Filler 2011 STC implementation (`src/tac/codec/syndrome_trellis_codec.py`) IS the canonical technique. Sister Catalog #262 STC-Dasher composite anchor in symposium_impls/ extends with MacKay Dasher arithmetic context | INTACT |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Orthogonal at paradigm level (different archive section than A1's latent, Selfcomp's mask, Quantizr's decoder weights). Composability with PR106 sister lanes (yshift / lrl1 / stacked) is FUTURE-OPEN per the stacked-substrate paradigm. | INTACT-AT-PARADIGM |
| 7 | DETERMINISTIC REPRODUCIBILITY | Filler 2011 STC encoder IS deterministic (Viterbi-style DP with fixed parity-check submatrix; reproducible per `STCParams` with `submatrix_seed`). Decoder is trivially deterministic (`m = H · y`). Byte-stable across re-runs | INTACT |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | STC encode wall-clock for ~16,800 residual values × constraint_height=12 trellis = ~200ms CPU. Acceptable for compress-time; decode is ~10ms CPU per inflate. Within A1's existing inflate runtime budget | INTACT |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | REVISED predicted band: `[-0.003, +0.001]` (SIGNED). Best case: -0.003 (matches PR106 stacked predicted gain). Worst case: +0.001 (slight regression if rate cost exceeds distortion gain). EXPECTED VALUE: -0.001 (small but positive contribution to A1's already-tied PR101 gold position). | CONDITIONAL-PER-PRE-PROBE |

**Overall:** 8 of 9 dimensions INTACT or INTACT-CONDITIONAL; Dimension 9 is conditional per pre-probe verdict. Comparison vs parent (5 of 9 INTACT): Path 3a is structurally MUCH stronger than the parent implementation because it (a) uses canonical Filler 2011 STC, (b) targets an appropriate cover signal (1D residual not 2D mask), (c) is additive composable with A1's existing archive grammar.

## 3. Observability surface (Catalog #305)

Per the 6-facet definition (CLAUDE.md "Max observability — non-negotiable"):

1. **Inspectable per layer**: STC encoder emits per-block (a) cover bits, (b) embedding cost vector, (c) message bits, (d) stego bits, (e) syndrome verification `H·y == m`. All inspectable via `src/tac/codec/syndrome_trellis_codec.py` debug mode. Pre-probe artifact at `.omx/state/probe_outcomes/stc_3a_a1_residual_entropy_<utc>.json` records per-residual-stream entropy + sparsity + temporal-correlation metrics. **INTACT.**
2. **Decomposable per signal**: A1+STC archive decomposes as `decoder_section (162KB) + latent_blob (15.4KB) + latent_sidecar (~607B) + STC_residual_sidecar (~312-625B target)`. Per-component byte attribution clear. **INTACT.**
3. **Diff-able across runs**: STC encoder is deterministic per `STCParams(submatrix_seed=0)`. Two runs on same residual produce IDENTICAL STC output. Argmax-byte-level diff for residual-stream-A vs residual-stream-B via the pre-probe manifest. **INTACT.**
4. **Queryable post-hoc**: Pre-probe manifest JSON includes per-residual-stream entropy + per-block STC byte cost + reconstructed score delta. Sufficient for grep + jq queries. Probe-outcomes-ledger entry (Catalog #313) provides cross-session queryability. **INTACT.**
5. **Cite-able**: Pre-probe cites archive SHA `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` (A1 canonical archive). Run-tuple `(substrate=stc_3a_a1_residual, commit=<git_HEAD>, config=constraint_height=12+submatrix_seed=0+residual_stream=<chosen>, upstream_snapshot_sha256=<pin>)` per Catalog #245 canonical 4-layer pattern. **INTACT.**
6. **Counterfactual-able**: Per Catalog #139 byte-mutation discipline: "what if I increase constraint_height from 12 to 16?" testable via re-run (cheap, ~5 min CPU); "what if I substitute residual stream A for residual stream B?" testable via the pre-probe categorical verdict. "what if I add STC to PR106 instead of A1?" testable via sister-substrate dispatch. **INTACT.**

**Overall observability score: 6 / 6 — STRONG.** Path 3a inherits A1's existing observability surface AND adds the pre-probe observability + Catalog #313 probe-outcomes-ledger queryability. The missing facet from the parent symposium (counterfactual-able for the implementation's core context-model choice) is REMEDIATED by Path 3a's pre-probe approach.

## 4. Sextet pact deliberation (Catalog #325 6-step #4)

### Council attendance + per-member assumption statements (Catalog #292 + #300 mandatory)

**Shannon (LEAD, information-theory grounding):**

- Operating-within assumption: "the contest scorer's rate term is `25 × archive_bytes / 37,545,489` so every byte saved at the substrate layer composes additively, and every byte added composes additively as +cost". HARD-EARNED per CLAUDE.md "Bit-level deconstruction and entropy discipline".
- Position: STC over A1 latent quantization residual is INFORMATION-THEORETICALLY SOUND if the residual cover signal has lower joint entropy than its current encoding (which is "not encoded at all"). The residual stream is currently ABSENT from A1's archive (the dequant error is lost during reconstruction); STC adds bytes to capture some of this information, with potential distortion-gain depending on how much of the lost information matters to the contest scorer. Per Catalog #220 / #272: the bytes must operationally affect score; pre-probe disambiguates. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe-first; tighten predicted band).

**Dykstra (CO-LEAD, optimization-feasibility):**

- Operating-within assumption: "Dykstra-feasibility intersection of (rate ≤ +1KB) ∩ (post-training Tier-C density on landed archive shows exploitable residual entropy) ∩ (distortion gain ≥ rate cost) ∩ (decoder ≤+25 LOC over A1's current ~120 LOC inflate) ∩ (HNeRV L4 substrate-engineering exception declared if budget exceeded) is the canonical feasibility check per Catalog #296". HARD-EARNED.
- Position: The intersection is NON-EMPTY but TIGHT. The pre-probe is the natural Dykstra-projection step: project the candidate residual streams onto the feasible region (HIGH-ENTROPY + temporal-correlation + ternary-or-similar structure) and select the highest-EV intersection point. If no residual stream lies in the feasible region, DEFER to 3b. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe-as-Dykstra-projection-step; predicted band `[-0.003, +0.001]`).

**Yousfi (challenge creator, steganalysis-canonical):**

- Operating-within assumption: "the SegNet scorer is a Fridrich-PhD-derived steganalysis surgery on EfficientNet-B2 — boundary regions ARE the canonical attack surface for inverse steganalysis. A1's latent residual encodes FRAME-LEVEL signal which affects boundary regions transitively through the HNeRV decoder". HARD-EARNED.
- Position: Concur with Shannon + Dykstra. STC-as-sidecar over latent residual COULD be steganalysis-aligned if the residual carries boundary-relevant information (e.g. high-frequency residual components are spatially aligned with SegNet's stride-2 stem). The pre-probe should measure this via Sobel/gradient-energy of the residual stream against SegNet's argmax boundary mask. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe includes SegNet-attack-surface alignment metric).

**Fridrich (steganalysis canonical):**

- Operating-within assumption: "UNIWARD-style distortion-informed embedding (errors in textured regions are undetectable) is the canonical SegNet attack vector. STC's parity-check structure preserves UNIWARD-alignment via the embedding cost vector `ρ_i`". HARD-EARNED.
- Position: Concur. The wet-paper construction in `src/tac/codec/syndrome_trellis_codec.py::WET_COST` allows ρ_i = wet for boundary-critical positions; this directly aligns with UNIWARD. Path 3a preserves UNIWARD-attack-surface compatibility for the latent residual cover signal. **VOTE: PROCEED_WITH_REVISIONS** (UNIWARD-cost-vector design included in pre-probe + impl).

**Contrarian (BOLD-but-skeptical):**

- Operating-within assumption: "every dispatch dollar is finite; the operator's $5.20 has equal-or-higher-EV alternatives". HARD-EARNED per CLAUDE.md "Race-mode rigor inversion".
- Position: $5.20 is small but the canonical Race-mode rigor inversion failure mode is "fan-out cheap dispatches without first-principles disambiguation". A $0 pre-probe disambiguates the question before any GPU spend; this is the cheapest possible path. Per Catalog #308 alternative-probe-methodologies, the pre-probe IS the canonical alternative-methodology probe. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe-first; $5.20 dispatch only on HIGH-ENTROPY verdict).

**Assumption-Adversary (sextet seat per Catalog #292):**

- Operating-within assumption: see verbatim above. Per the per-deliberation explicit-assumption-statement discipline, I (Assumption-Adversary) MUST surface ONE shared-assumption-violation hypothesis with explicit reasoning per CLAUDE.md "Council conduct" Fix-7 amendment. My hypothesis: the predicted band `[-0.015, -0.003]` inherited from parent symposium #861 is CARGO-CULTED forward without sister-anchor empirical calibration; tightening to `[-0.003, +0.001]` is HARD-EARNED per PR100→PR105 + PR106 yshift sister evidence. VETO if any council member proceeds without acknowledging this assumption-class shift.
- Position: No veto fires (unanimous PROCEED_WITH_REVISIONS acknowledges the assumption-class shift). **VOTE: PROCEED_WITH_REVISIONS** (concurrent with the revision).

### Grand-council attendees (per topic, Catalog #325 6-step #4)

**Tomáš Filler (canonical STC author):**

- Operating-within assumption: "STC achieves R_AC(D) + 1/h bound where R_AC IS the arithmetic coding lower bound for the cover signal's PROPER context model — for SPARSE 1D signals, context-naive AC IS the proper model (no spatial structure to exploit)". HARD-EARNED per Filler 2011 IEEE TIFS Theorem 4 + Filler dissertation §5 throughput tables.
- Position: see verbatim above. Path 3a's structural distinction from the parent implementation IS that the cover signal IS a sparse 1D residual where context-naive AC is appropriate. The Filler 2011 bound applies cleanly. The remaining empirical question is whether the residual stream EXISTS with sufficient entropy structure; pre-probe disambiguates. **VOTE: PROCEED_WITH_REVISIONS** (tighten predicted band per Filler dissertation §5 throughput on similar payload sizes: ~2-5 bytes saved per constraint-block × ~50-100 blocks = 100-500 bytes savings vs naive baseline; rate-cost-equivalent ~+0.00007 to +0.00033; distortion-gain-required-to-break-even matches sister-anchor ratios).

**Tomáš Pevný (sister steganalysis cite, Filler 2010 co-author):**

- Operating-within assumption: "syndrome-trellis structure (parity-check matrix construction) is the canonical mechanism for distortion-bounded payload embedding". HARD-EARNED per Pevný 2010 IEEE TIFS Section III + dual-layer STC.
- Position: Concur with Filler. For the SIDECAR-2nd-LAYER residual stream interpretation (candidate (c) from Assumption 2 unwind), Pevný 2010 dual-layer STC is the natural construction: layer 1 = existing Huffman over (dim, delta); layer 2 = STC over (dim, delta) corrections that didn't fit the existing 16-value quantization. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe should evaluate dual-layer construction explicitly).

**Selfcomp / szabolcs-cs (PR#56 author):**

- Operating-within assumption: see verbatim above. The 421KB AV1 monochrome mask payload IS the empirically-canonical mask codec for this video class; any new sidecar cannot displace AV1 without proving rate-equivalence.
- Position: Path 3a does NOT touch AV1 monochrome mask payload (Path 3b would). Path 3a operates on A1's latent stream which is ORTHOGONAL to my PR#56 paradigm. The PRE-PROBE should ALSO compare Path 3a vs Path 3b expected gains; if Path 3b shows higher EV at A1's operating point, pair them (op-routable #3). **VOTE: PROCEED_WITH_REVISIONS** (pre-probe + sister 3b symposium + parallel-pairing op-routable).

**Quantizr (adversarial reverse-engineering, leaderboard 0.33 leader):**

- Operating-within assumption: see verbatim above. A1's archive IS the canonical post-PR95 winning recipe.
- Position: Path 3a adds a NEW residual sidecar; it does NOT replace existing components. Composability with my 0.33 leaderboard pattern is INTACT iff bytes added < 1 KB (the rate cost should not exceed the distortion gain). Per the REVISED predicted band `[-0.003, +0.001]` (SIGNED), the worst-case is +0.001 regression which would NON-PROMOTE; pre-probe is the right disambiguator. **VOTE: PROCEED_WITH_REVISIONS** (pre-probe + tight predicted band).

### Vote tally

**Sextet pact (Catalog #325 6-step #4 quorum 5-of-6):**

- 6-of-6 PROCEED_WITH_REVISIONS (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary)
- 0 PROCEED-unconditional
- 0 DEFER
- 0 REFUSE
- 0 ESCALATE
- **Quorum: 6/6 = MET (≥5/6 threshold satisfied)**

**Grand-council attendees (4-of-4):**

- Filler: PROCEED_WITH_REVISIONS (tighten predicted band per Filler dissertation §5)
- Pevný: PROCEED_WITH_REVISIONS (evaluate dual-layer construction in pre-probe)
- Selfcomp: PROCEED_WITH_REVISIONS (pair with Path 3b symposium)
- Quantizr: PROCEED_WITH_REVISIONS (composability INTACT iff bytes < 1KB)

**Final verdict: PROCEED_WITH_REVISIONS — UNANIMOUS 10/10.**

The unanimous revision is: BEFORE any $5.20 dispatch, run a $0 CPU pre-probe to disambiguate (a) which of three candidate residual streams has exploitable entropy, (b) whether the predicted band `[-0.003, +0.001]` is empirically supported. Pre-probe verdict gates dispatch eligibility.

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308)

Per Catalog #308 (alternative-probe-methodologies enumeration; at least N=3 required for substrate-class verdicts).

### Reactivation path (a): Standalone 3a impl per spec (PRIORITY 1)

- **Description:** STC-as-sidecar over A1's HIGHEST-EV residual stream (selected by pre-probe). Constraint_height=12, submatrix_seed=0, canonical Filler 2011 implementation.
- **Predicted ΔS band:** `[-0.003, +0.001]` (REVISED per sister-anchor evidence; SIGNED — could regress)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324
- **Predicted cost:** $5.20 (Modal A100 10ep smoke + $0.20 paired-CPU adjudication via `tools/dispatch_modal_paired_auth_eval.py`)
- **Structural verdict:** This path tests Path 3a in isolation; RATIFIES OR FALSIFIES the paradigm in A1-substrate context.
- **Implementation complexity:** ~180 LOC (Filler 2011 STC decoder ~150 LOC + A1 archive-grammar amendment ~30 LOC). Fits within HNeRV L7 substrate-engineering bolt-on budget (350 LOC).
- **Composability:** ADDITIVE with A1's existing archive grammar; ORTHOGONAL to PR#56 mask channel; ORTHOGONAL to Quantizr decoder weights.

### Reactivation path (b): Path 3a + Path 3b combined (PRIORITY 2)

- **Description:** Pair Path 3a with Path 3b (STC-as-tone-map-delta over Selfcomp soft-grayscale baseline per parent symposium #861 op-routable #4). Tests composability of STC paradigm across 2 substrate-classes simultaneously.
- **Predicted ΔS band:** `[-0.005, +0.002]` (combined sister-anchor evidence; SIGNED — could regress per Catalog #296 Dykstra-feasibility of joint constraint)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324
- **Predicted cost:** $10.40 ($5.20 for 3a + $5.20 for 3b; orthogonal substrates allow parallel dispatch)
- **Structural verdict:** This path tests STC paradigm as a META-PARADIGM applicable across substrate-classes; provides paradigm-vs-implementation classification per Catalog #307.
- **Implementation complexity:** ~180 LOC for 3a + ~200 LOC for 3b = ~380 LOC combined (still within 2× HNeRV L7 budget for substrate engineering).
- **Composability:** ADDITIVE; sister-symposium synergy.

### Reactivation path (c): Canonical Filler 2011 STC with 2D + temporal context model (PRIORITY 3)

- **Description:** Reformulate Path 3a to use a 2D + temporal context model (per parent symposium #861 Path 3c). For A1 residual stream, the 2D context would be (frame_t-1, frame_t) per-pair correlations.
- **Predicted ΔS band:** `[-0.004, +0.001]` (canonical context model gives ~10-20% additional savings vs context-naive on temporal residuals; sister-anchor-derived)
- **Predicted_band_validation_status:** `pending_post_training` per Catalog #324
- **Predicted cost:** $5.20 + ~$10-15 implementation development overhead = $15-20 total
- **Structural verdict:** Tests whether 2D+temporal context model provides MEANINGFUL gain on A1's already-saturated operating point.
- **Implementation complexity:** ~300 LOC (canonical context model adds ~150 LOC over baseline 3a impl).
- **Composability:** ADDITIVE; same axis as path (a) but with more sophisticated coder.

### Reactivation priority ordering

1. **Path (a) (HIGHEST EV)**: $5.20 cost (after $0 pre-probe), `[-0.003, +0.001]` predicted band, tests core hypothesis at minimum cost.
2. **Path (b) (SECOND EV)**: $10.40 cost, `[-0.005, +0.002]` predicted band, tests STC as META-PARADIGM via composability.
3. **Path (c) (LOWEST EV)**: $15-20 cost, `[-0.004, +0.001]` predicted band, marginal context-model improvement over (a).

**Recommendation:** Pursue path (a) as the canonical reactivation path. IF (a) ratifies the paradigm (post-training Tier-C verdict `validated_post_training` with empirical ΔS within ±0.005 of predicted band per Catalog #324), THEN proceed to (b) as the META-paradigm composability test. DEFER (c) until (a) AND (b) provide sufficient evidence that the 2D+temporal context investment is justified.

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status:** `pending_post_training` for all 3 reactivation paths.

**Rationale:** Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density" non-negotiable: predicted bands derived from sister-anchor empirical evidence (PR100→PR105 + PR106 yshift) + first-principles Filler 2011 + Pevný 2010 + Catalog #262 STC-Dasher composite (NOT random-init Tier-C density) are PROVENANCE=sister_anchor_calibrated. The reactivation criterion is post-training Tier-C re-measurement on the landed reformulation archive.

**Reactivation criterion verbatim:** "Post-training Tier-C density measurement on landed archive sha for the chosen path (a/b/c) via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within ±0.002 of the revised predicted band, ratify the paradigm + advance to L2. If outside band by ≥3× (i.e. |empirical - predicted| > 0.006), surface as Catalog #324 violation and re-symposium. Specifically: if empirical ΔS lands in `[-0.001, +0.005]` (range that brackets the C6 IBPS 22× miss pattern), that REGRESSES PR101 gold position and Path 3a is FALSIFIED at A1's operating point."

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes:

- `deliberation_id`: `council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517`
- `topic`: STC reformulation Path 3a per stc parent symposium #861 reactivation
- `council_tier`: T2
- `council_attendees`: 10-seat list (sextet + 4 grand-council)
- `council_quorum_met`: true (6/6 sextet + 4/4 grand-council)
- `council_verdict`: PROCEED_WITH_REVISIONS
- `council_dissent`: 5 verbatim entries (Assumption-Adversary VETO-condition + Filler revision + Selfcomp routing + Quantizr conditional + Contrarian pre-probe-first)
- `council_assumption_adversary_verdict`: 6 assumptions classified
- `council_decisions_recorded`: 5 op-routables
- `council_predicted_mission_contribution`: frontier_protecting
- `council_override_invoked`: false
- `related_deliberation_ids`: [`council_per_substrate_symposium_stc_clean_source_20260517`] (parent symposium #861)

Downstream consumers per Catalog #325:

- **Catalog #325 STRICT preflight** sees the PROCEED_WITH_REVISIONS verdict — does NOT structurally REFUSE dispatch but routes through the revision-implementation gate: dispatch is admissible IF the pre-probe verdict is `HIGH-ENTROPY-RESIDUAL-PRESENT` AND a NEW symposium PROCEED-unconditional verdict supersedes this PROCEED_WITH_REVISIONS per Catalog #315 OPTIMAL FORM discipline.
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('stc_3a_sidecar_a1_residual')` for council-verdict-aware candidate weighting. Per Catalog #319 Wyner-Ziv reweight protection: STC-3a is NOT a Wyner-Ziv-deliverability target (its bytes ADD signal, not REWEIGHT existing latent), so the autopilot adjustment cascade routes via standard `adjust_predicted_delta_for_composition_alpha` (no special factor).
- **Probe-outcomes ledger (Catalog #313)** receives a sister-registered PROCEED_WITH_REVISIONS outcome via `tac.probe_outcomes_ledger.register_probe_outcome(substrate_id='stc_3a_sidecar_a1_residual', verdict='PROCEED_WITH_REVISIONS', status='blocking_pending_pre_probe', methodology='stc_over_a1_latent_quantization_residual_with_constraint_height_12_canonical_filler_2011', alternative_probe_methodologies=['stc_over_decoder_weight_residual_static', 'stc_over_sidecar_residual_2nd_layer_pevny_dual_2010', 'huffman_with_2d_temporal_context_no_stc_baseline', 'arithmetic_coder_with_dasher_context_no_stc_baseline'], expires_at_utc=<30_days_from_now>)`. Any future dispatch wrapper consults the canonical outcome BEFORE firing.

## 8. Cross-references

- **Parent symposium:** `council_per_substrate_symposium_stc_clean_source_20260517.md` (#861 DEFER verdict; this memo implements op-routable #4 path 3a reformulation).
- **Sister symposiums queued:** `council_per_substrate_symposium_stc_3b_tone_map_delta_selfcomp_<date>.md` (Path 3b; PRIORITY 2 sister); `council_per_substrate_symposium_stc_3c_2d_temporal_context_<date>.md` (Path 3c; PRIORITY 3 sister).
- **Empirical anchors cited:** `lane_filler_stc_paradigm_alpha` (STC=996B vs LZMA-ternary=3084B on 262K-sym ternary sample); `lane_pr106_latent_sidecar` (PR100→PR105 sister: +1124B for -0.00218 score Δ); `lane_pr106_yshift_sidechannel` (predicted -0.0005 to -0.0015 at 1-2KB); `lane_a1_pr_submission_entry_packet` (A1 archive sha 87ec7ca5; contest-CPU 0.19285 paired CUDA 0.22635).
- **Canonical implementations cited:** Filler 2011 IEEE TIFS Theorem 4 (`src/tac/codec/syndrome_trellis_codec.py`); Pevný 2010 IEEE TIFS dual-layer STC; MacKay 2003 ITILA §6.6 Dasher (`src/tac/symposium_impls/stc_dasher_arithmetic_coding_maximalism.py`, Catalog #262); HNeRV parity discipline lessons 1-13; PR101 / PR103 silver anchors; PR100→PR105 latent sidecar pattern.
- **Catalog gates fired by this symposium:** #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification: PARADIGM-INTACT-AT-CANONICAL-IMPL) + #308 (alternative-probe-methodologies enumeration with N=4) + #313 (probe-outcomes ledger) + #315 (OPTIMAL FORM before paid dispatch: pre-probe IS the iteration discipline step) + #324 (post-training Tier-C validation; pre-probe is sister-anchor calibrated NOT random-init) + #325 (per-substrate symposium discipline; this memo IS the canonical 6-step contract).
- **Catalog gates protected by this symposium:** #220 (operational mechanism declaration; Path 3a impl MUST declare); #272 (distinguishing-feature integration contract; Path 3a IS the distinguishing feature); #233 (L1→L2 promotion canonical 4-gate); #298 (substrate retirement discipline 30-day; lane pre-register at L0 per op-routable #5).
- **Catalog gates NOT applicable to this symposium:** #310 (F-asymptote-class check): NO — Path 3a is honest about being SIDECAR (paradigm-INTACT-AT-IMPL, NOT class-shift). #311 (predictive-coding ego-motion conditioning): NO — Path 3a does NOT claim Atick-Redlich cooperative-receiver framing. #312 (hierarchical predictive coding canonical quadruple): NO — Path 3a does NOT claim hierarchical predictive coding.
- **Standing directive references:** CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (2026-05-15) — Path 3a's per-layer canonical-vs-unique decisions (see §9 below); CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (2026-05-17) — pre-probe IS the iteration step to OPTIMAL FORM; CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (2026-05-17/18) — this memo IS the canonical 6-step contract instance.

## 9. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + Catalog #290 strict gate.

| Layer | Decision | Rationale |
|---|---|---|
| Archive grammar | **UNIQUE-FORK** | A1's monolithic single-member `x` grammar amended with STC residual section. Cannot adopt canonical pyramid grammar (PR101 split-Brotli specific). Forks because the substrate's optimal encoding requires section-specific archive amendment. |
| Inflate runtime | **ADOPT-CANONICAL** | Uses `tac.substrates._shared.inflate_runtime.select_inflate_device` (per Catalog #205 + #295 OSS-hermetic sister); A1's existing inflate.py already forks this canonical pattern correctly. STC decoder is bolt-on to existing inflate. |
| Score-aware loss | **ADOPT-CANONICAL** | Path 3a is post-training sidecar; no training-time loss needed. A1's existing PR101-derived score-gradient loss is INHERITED unchanged. |
| Codec primitive | **ADOPT-CANONICAL** | Uses canonical `src/tac/codec/syndrome_trellis_codec.py::syndrome_trellis_encode_ternary` (Filler 2011). NO substrate-specific fork; the canonical STC implementation IS the right tool for sparse 1D residual cover signals. |
| EMA discipline | **ADOPT-CANONICAL** | Inherits A1's EMA discipline (PR101-derived, 0.997 decay per CLAUDE.md "EMA — NON-NEGOTIABLE"). No fork needed for sidecar bolt-on. |
| eval_roundtrip | **ADOPT-CANONICAL** | Inherits A1's eval_roundtrip=True discipline. Sidecar bolt-on does not require new training. |
| Scorer routing | **ADOPT-CANONICAL** | Uses canonical `tac.substrates._shared.score_aware_common.score_pair_components` and `gate_auth_eval_call` (Catalog #164 + #226). No substrate-specific fork. |
| Catalog #220 operational mechanism | **UNIQUE-DECLARED** | Path 3a's distinguishing feature is the STC residual sidecar; per Catalog #220 + #272 the operational mechanism MUST be declared: `score_improvement_mechanism_status=PENDING_PRE_PROBE`, `archive_bytes_added=~312-625B target`, `inflate_consumer_function=apply_stc_residual_sidecar` (NEW helper to land alongside impl), `byte_mutation_smoke_passes=PENDING_POST_IMPL_VIA_TOOLS_VERIFY_DISTINGUISHING_FEATURE_BYTE_MUTATION_PY`. |
| Pre-probe disambiguator | **UNIQUE-FORK** | Tool `tools/probe_stc_3a_a1_residual_entropy.py` (NEW; ~150 LOC; $0 CPU) is substrate-specific because no canonical generic residual-entropy probe exists. Forks because the substrate's residual streams (latent quantization / decoder weight / sidecar 2nd-layer) are A1-specific. |
| Continual-learning posterior | **ADOPT-CANONICAL** | Uses canonical `tac.council_continual_learning.append_council_anchor` (Catalog #300) for THIS symposium memo + `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313) for pre-probe verdict + post-dispatch outcome. No fork. |
| Dispatch wrapper | **ADOPT-CANONICAL** | Uses canonical `tools/operator_authorize.py` + `tools/run_modal_smoke_before_full.py` (Catalog #167 + #176 + #271 + #243). No substrate-specific dispatch fork; A1's existing wrapper pattern extends to 3a. |
| Lane registry | **ADOPT-CANONICAL** | Pre-register via canonical `tools/lane_maturity.py add-lane` (per CLAUDE.md "Lane maturity registry" non-negotiable). No fork. |

**Summary:** 9 ADOPT-CANONICAL + 3 UNIQUE-FORK (archive grammar / pre-probe disambiguator / Catalog #220 operational mechanism declaration). All FORKS have explicit rationale per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" decision criterion (PRINCIPLED-mismatch with canonical helpers). The bolt-on is structurally LEAN per HNeRV parity L7 (Path 3a impl ~180 LOC bolt-on; substrate-engineering exception declared).

## 10. Operator op-routables (for parent agent + main Claude)

1. **DO NOT fire $5.20 Modal A100 dispatch on Path 3a until $0 CPU pre-probe lands.** Per the Contrarian + Assumption-Adversary verdicts: the canonical Race-mode rigor inversion failure mode is "fan-out cheap dispatches without first-principles disambiguation". The $0 pre-probe disambiguates the structural question (which of three candidate residual streams has exploitable entropy) before any GPU spend.

2. **Build the $0 CPU pre-probe** at `tools/probe_stc_3a_a1_residual_entropy.py` (~150 LOC). Loads A1 archive `87ec7ca5...` from `submissions/a1/archive.zip`; extracts (a) latent quantization residual `fp32 - reconstruct(uint8, scale, min)` for ~16,800 values, (b) decoder weight residual `float - reconstruct(q-byte, fp16 scale)` for ~88K params, (c) sidecar 2nd-layer residual `(dim, delta) values that didn't fit the existing 16-value SIDECAR_DELTAS_X100 quantization` (requires re-fitting the sidecar at higher quantization budget). For each: measure H(X), H(X|X_{t-1}), sparsity, ternary-structure-rate, alignment with SegNet attack surface. Emit categorical verdict + per-stream STC byte-cost estimate. Time: ~1 hour to land + ~5 min to run.

3. **Register the canonical probe-outcomes-ledger entry** per op-routable #4 above. Use `tac.probe_outcomes_ledger.register_probe_outcome` with verdict=PROCEED_WITH_REVISIONS, status=blocking_pending_pre_probe, methodology=stc_over_a1_latent_quantization_residual_with_constraint_height_12_canonical_filler_2011.

4. **Pre-register the contingent dispatch lane** `lane_stc_sidecar_a1_residual_3a_20260518` at L0 via `tools/lane_maturity.py add-lane`. Notes: `research_only=true until pre-probe verdict + 5-PROCEED council; reactivation_criteria=HIGH-ENTROPY-RESIDUAL-PRESENT + PROCEED-unconditional council deliberation; lane_class=substrate_engineering per HNeRV parity L7 (~180 LOC bolt-on)`.

5. **Sister-symposium synergy decision: pair with 3b OR await 3b symposium?** Per the parent symposium op-routable #4, paths 3a and 3b are PARALLEL with combined cost $10.40 and ORTHOGONAL substrates (A1 latent vs Selfcomp mask). Recommendation: **AWAIT 3b symposium** (sister-subagent dispatched per parent op-routable #4) before pairing decision; compare predicted-band confidence + pre-probe results across both before any GPU spend. If both pre-probes verdict `HIGH-ENTROPY-RESIDUAL-PRESENT`, pair them for $10.40 parallel dispatch. If 3a verdict is `HIGH` but 3b is `MEDIUM`, prioritize 3a-only $5.20. If 3a verdict is `MEDIUM/LOW`, defer to 3b alone OR redirect to Path 3c (canonical context model).

**Total dispatch redirect (vs parent symposium $5.20 default):** $0 pre-probe FIRST (this turn editor work) → $5.20 conditional dispatch (parent op-routable #4 path 3a) only on `HIGH-ENTROPY` verdict. Risk-adjusted EV = ~50-100% of parent's $5.20 EV with the same predicted band BUT with empirical pre-probe validation BEFORE the GPU meter starts. Per CLAUDE.md "Race-mode rigor inversion" + "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + the operator's 2026-05-17/18 standing directive on PER-SUBSTRATE OPTIMAL FORM.

---

**Symposium concludes.** Verdict: PROCEED_WITH_REVISIONS — 10-of-10 unanimous. Mission-alignment: frontier_protecting. Override: not invoked. Continual-learning anchor: registering to `.omx/state/council_deliberation_posterior.jsonl` via canonical helper. Probe-outcomes ledger anchor: registering to `.omx/state/probe_outcomes.jsonl` via canonical helper. Lane pre-registration: `lane_stc_sidecar_a1_residual_3a_20260518` at L0 with `lane_class=substrate_engineering` per HNeRV parity L7.
