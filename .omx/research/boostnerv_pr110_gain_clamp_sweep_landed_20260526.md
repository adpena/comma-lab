<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 gain_clamp × epochs 9-cell sweep landing memo. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: empirical 9-cell sweep landed end-to-end with canonical artifact at .omx/research/boostnerv_pr110_gain_clamp_sweep_results_20260526/sweep_heatmap.json. -->
<!-- # FORMALIZATION_PENDING:boostnerv_gain_clamp_sweep_extends_l1_empirical_anchor_for_canonical_equation_residual_hybrid_boosting_savings_v1_candidate_per_catalog_344_operator_decision_protocol_no_unilateral_registration -->
---
council_tier: T1
council_attendees: [Carmack, Shannon, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "Empirical sweep VALIDATES my analytical dissent on the loss/recon axis: loosening gain_clamp from 0.05 to 0.20 cuts training loss 7.8% → 34.6% and recon-proxy MSE reduction 16.2% → 52.6%. THIS IS HEADROOM. The L1 EMPIRICAL anchor at gain_clamp=0.05 WAS clamp-bound. But the BPR1 sidecar stays 42 bytes across all 9 cells — that's because the int8 quantization step `residual_int8 = round(residual_clamped / gain_clamp * 127)` normalizes the residual to the same [-127, 127] range regardless of gain_clamp; brotli sees the same near-saturated distribution either way. So the rate axis is SCALE-INVARIANT under int8 quantization — the loss improvement does NOT translate to fewer bytes. The substrate paradigm has signal-quality headroom but the BPR1 codec design caps the byte savings at the header floor (28B + ~14B brotli). FOR T3 PR110-STACKING ORDERING: BoostNeRV-PR110 enters as candidate #6 with the gain_clamp=0.20 config; predicted Δrate stays +0.000028 (negligible); predicted distortion reduction is UNMEASURED on contest axes (recon-proxy improvement is advisory per Catalog #192). Operator-routable: paired CUDA dispatch on best-cell config (gain_clamp=0.20, epochs=30) to measure true d_seg+d_pose reduction."
council_assumption_adversary_verdict:
  - assumption: "gain_clamp=0.05 is the binding constraint on residual learning"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "p99 of pre-clamp |residual| at gain_clamp=0.05 is 0.7759 — 15× larger than the clamp; the residual learner WANTS to produce magnitudes far exceeding 0.05 but is being clipped. At gain_clamp=0.10 p99=0.90; at 0.20 p99=0.96. Loss reductions track: 7.8% → 13% → 34.6%. Carmack's hypothesis fully confirmed at the loss axis."
  - assumption: "Loss improvement from loosening gain_clamp translates to byte savings"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED
    rationale: "BPR1 sidecar stays exactly 42 bytes across all 9 cells. The int8 quantization step normalizes residuals to [-127, 127] using gain_clamp as the scale factor — same relative quantization granularity regardless of gain_clamp absolute value. Brotli sees same near-saturated quantized distribution. The rate axis is structurally scale-invariant under the current BPR1 codec design. This is an UNEXPECTED CARGO-CULTED implicit assumption that the sweep extincted."
  - assumption: "Epochs is a meaningful free variable in this regime"
    classification: CARGO-CULTED-EMPIRICALLY-REFUTED
    rationale: "Loss spread across epochs ∈ {30, 100, 300} at fixed gain_clamp is < 0.0003 (e.g. 0.10697 vs 0.10725 at gain_clamp=0.05). Training converges at ~30 epochs; additional epochs over-fit slightly to the 50-pair fixture. Epochs=30 is sufficient; longer schedules waste compute without learning."
council_decisions_recorded:
  - "Carmack-dissent VALIDATED on loss/recon axis: gain_clamp=0.05 → 0.20 yields 7.8% → 34.6% loss reduction + 16.2% → 52.6% recon MSE reduction"
  - "Carmack-dissent REFUTED on byte axis: BPR1 sidecar stays 42 bytes across all 9 cells (int8 quantization normalization is scale-invariant)"
  - "Best cell: gain_clamp=0.20, epochs=30 — final_loss=0.0605, recon_red=52.6%, sidecar=42B, Δrate=+0.000028"
  - "Operator-routable #1 (HIGHEST EV): refactor BPR1 codec to break int8 quantization scale-invariance — variable bit-width per gain_clamp OR signed-exponent variable-precision encoding"
  - "Operator-routable #2: paired CUDA dispatch on best-cell config to measure true d_seg+d_pose contest-axis reduction"
  - "Operator-routable #3: fold gain_clamp=0.20, epochs=30 BoostNeRV-PR110 config into T3 PR110-stacking ordering memo as candidate #6"
  - "Operator-routable #4 (DEFER): canonical equation residual_hybrid_boosting_savings_v1 registration AWAITS operator decision per Catalog #344 — empirical anchors extend the proposed equation but BPR1 byte-savings invariance complicates the predicate Δrate_residual_hybrid formula"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boostnerv_pr110_l1_empirical_landed_20260526
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
  - t3_council_pr110_stacking_pivot_ordering_landed_20260526
  - comprehensive_roadmap_synthesis_landed_20260526
---

# BoostNeRV-PR110 gain_clamp × epochs 9-cell sweep LANDED 2026-05-26

**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` (extends sister L1 EMPIRICAL via Carmack-dissent disambiguator)
**Subagent**: `boostnerv-pr110-gain-clamp-sweep-20260526`
**Predecessor**: BoostNeRV L1 EMPIRICAL respawn (commit `b2fd3e587`)
**Operator authority**: Carmack analytical dissent in L1 landing memo + ROADMAP TOP-EV cascade approval 2026-05-26
**Wallclock**: 15.0 seconds (M5 Max MLX-local; 9 cells sequential)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Drift surface declaration (per NEW MLX↔CUDA bidirectional drift directive 2026-05-26)

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`, the sweep reuses the L1 EMPIRICAL probe's canonical helpers verbatim (identical NHWC convention, identical Conv2d padding, identical tanh+clip ordering, identical brotli quality 9, identical AdamW β₁/β₂ defaults, identical fp32 throughout). No new drift surface introduced. The L1 declared mitigations apply unchanged to all 9 sweep cells. Paired CUDA verification per Catalog #1265 contest-equivalence gate remains deferred per "Remember all on MLX".

## Canonical-vs-frontier-push decision (per NEW pushing-the-frontier directive 2026-05-26)

Per `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`:

- **Sweep apparatus level**: CANON-APPLICATION (3×3 cartesian-product sweep is the canonical experimental-design pattern; no novel algorithm contribution at the orchestration layer).
- **Empirical interpretation level**: FRONTIER-PUSH (the sweep is the disambiguator probe per Catalog #313 probe-outcomes ledger between (a) "L1 anchor is clamp-bound" vs (b) "L1 anchor is at architectural ceiling" — empirically the answer is (a) on the loss/recon axis and (b) on the byte axis; this dual-finding is itself a novel insight per the dual-purpose framing).
- **Codec-design level**: explicit CANON-APPLICATION decision to reuse int8 quantization with gain_clamp scale factor (per L1 BPR1 codec design); the sweep EXTINCTED the cargo-culted assumption that "loss improvement → byte savings". A future Variant B residual codec (signed-exponent or variable-bit-width quantization) would be FRONTIER-PUSH; this sweep does not pursue it (out of scope per Carmack operator-routable).

## Empirical findings — 9-cell heatmap

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| **gain_clamp=0.05** (L1 anchor) | loss=0.10697 / sidecar=42B / recon_red=16.2% / p99=0.776 | loss=0.10709 / 42B / 16.2% / 0.771 | loss=0.10725 / 42B / 16.2% / 0.758 |
| **gain_clamp=0.10** | loss=0.08905 / 42B / 30.2% / 0.903 | loss=0.08915 / 42B / 30.2% / 0.899 | loss=0.08929 / 42B / 30.2% / 0.889 |
| **gain_clamp=0.20** | **loss=0.06054 / 42B / 52.6% / 0.960** | loss=0.06061 / 42B / 52.6% / 0.958 | loss=0.06070 / 42B / 52.6% / 0.953 |

(p99 = p99 of pre-clamp `|residual|` magnitude on 50-pair × 96×128 × 3 = 1.84M elements; tanh-bounded ∈ [0, 1]; sister diagnostic for clamp-bound-vs-unbound regime)

### Sister-anchor coherence check

The (gain_clamp=0.05, epochs=30) cell is the L1 EMPIRICAL anchor RE-RUN; final loss reproduced as **0.10697** vs L1 anchor `0.10697495...` per `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json`. Match within `1e-5` per MLX AdamW seed determinism on M5 Max. Sister-anchor coherence VERIFIED (Catalog #305 observability facet: diff-able across runs).

## Carmack-dissent verdict per Catalog #307

**PARADIGM-LEVEL FALSIFICATION**: NONE. The BoostNeRV-against-PR110 residual-correction-hybrid stacking paradigm is INTACT.

**IMPLEMENTATION-LEVEL DUAL FINDING**:

1. **LOSS/RECON AXIS — Carmack-dissent VALIDATED**: gain_clamp=0.05 IS clamp-bound; loosening to 0.20 yields 4.4× larger loss reduction (7.8% → 34.6%) and 3.2× larger recon-proxy MSE reduction (16.2% → 52.6%). The L1 anchor was structurally under-using the available signal capacity.

2. **BYTE AXIS — Carmack-dissent REFUTED (unexpected discovery)**: BPR1 sidecar stays 42 bytes across all 9 cells. The L1 BPR1 codec's int8 quantization step `residual_int8 = round(residual_clamped / gain_clamp * 127)` is **scale-invariant** by construction — the residual magnitude is always normalized to [-127, 127] regardless of gain_clamp absolute value. Brotli sees the same near-saturated quantized distribution either way. The rate axis is structurally locked at ~42 bytes by the BPR1 header (28B) + brotli-of-near-saturated-int8 (~14B) floor.

**Net contest-axis implication**: at the current BPR1 codec design, loosening gain_clamp moves loss/recon BUT does not move bytes. Whether this translates to net positive contest-ΔS depends on UNMEASURED d_seg+d_pose reduction (Catalog #164/#226 canonical MLX scorer routing OR paired CUDA dispatch required).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the substrate is DEFERRED-pending-(a)-paired-CUDA-d_seg+d_pose-measurement-OR-(b)-Variant-B-residual-codec-redesign-that-breaks-quantization-scale-invariance. NOT KILLED.

## Catalog #344 candidate equation `residual_hybrid_boosting_savings_v1` — empirical anchor extension

Per L1 EMPIRICAL landing memo §"Canonical equation #344 candidate (proposed, awaits operator approval)": the proposed equation predicate is `Δrate_residual_hybrid = 25 × sidecar_bytes / 37545489 + Δdistortion_residual = bounded_below_by_(d_seg_per_pixel × pixels_affected) - bounded_above_by_(p_residual_clamp × max_per_pixel_distortion)`.

**This sweep adds empirical anchors that COMPLICATE the proposed predicate**:

- At fixed BPR1 codec, `sidecar_bytes` is INDEPENDENT of `gain_clamp` (sweep shows 42B at all 9 cells).
- Empirical Δrate is therefore CONSTANT at +0.000028 contest units across all 9 cells.
- The `p_residual_clamp` term in the upper bound does NOT scale linearly with gain_clamp because the int8 quantization normalizes.
- The proposed `bounded_above_by_(p_residual_clamp × max_per_pixel_distortion)` predicate is FALSIFIED at the int8-quantized BPR1 codec surface; would need replacement with `bounded_above_by_(quantization_fidelity × max_per_pixel_distortion)` where quantization_fidelity is itself scale-invariant ~1.0 at int8/127.

**Recommendation per Catalog #344 operator-decision protocol**: DEFER registration of `residual_hybrid_boosting_savings_v1` pending either (a) Variant B codec design that produces gain_clamp-scaled sidecar bytes; (b) paired CUDA dispatch that empirically measures `Δdistortion_residual` and tests whether the proposed predicate generalizes. Per Catalog #359 sister discipline: avoid mis-application to residual-hybrid context that the equation does not predict cleanly.

## 6-hook wire-in declaration (per Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map contribution**: ACTIVE — the 9-cell residual histogram + per-cell p99 + boundary-fraction-at-clamp distributions can feed sister `tac.sensitivity_map.*` ranker. WIRE-IN: deferred to operator-routable; current artifact JSON carries per-cell `residual_stats` block.
2. **Pareto constraint**: ACTIVE — the 9 cells empirically map the (gain_clamp, loss) Pareto frontier at fixed BPR1 codec; the byte axis is degenerate (constant 42B). WIRE-IN: documented in this memo; cathedral_consumer at sister candidate may emit explicit Pareto-row.
3. **Bit-allocator hook**: N/A at this sweep (uniform int8 quantization across all cells; no per-element allocation).
4. **Cathedral autopilot dispatch hook**: ACTIVE — `.omx/research/boostnerv_pr110_gain_clamp_sweep_results_20260526/sweep_heatmap.json` carries canonical Provenance per Catalog #323 (`axis_tag="[macOS-MLX research-signal]"` + `promotion_eligible=false` + `score_claim=false` + `ready_for_exact_eval_dispatch=false`). Auto-discoverable per Catalog #335 cathedral_consumers Protocol contract.
5. **Continual-learning posterior update**: ACTIVE — the 9-cell anchors extend the BoostNeRV-PR110 evidence for cathedral_consumers query helpers. Per operator-routable: register the sweep anchor via `tac.continual_learning.posterior_update_locked`.
6. **Probe-disambiguator**: ACTIVE — THIS sweep IS the canonical Carmack-dissent disambiguator probe per Catalog #313 probe-outcomes ledger between (a) "L1 anchor was clamp-bound" → CONFIRMED on loss/recon axis vs (b) "L1 anchor was at architectural ceiling" → CONFIRMED on byte axis under current BPR1 codec.

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (predicted PLATEAU-ADJACENT to FRONTIER-PURSUIT lower band per design memo §"9-dimension success checklist evidence" 9.OPTIMAL MINIMAL CONTEST SCORE)

Current empirical position: rate axis at +0.000028 (constant across all 9 cells; well-inside roadmap-predicted [-0.010, +0.0045] band); distortion axis UNMEASURED on contest axes. The sweep does NOT change the HORIZON-CLASS verdict; it disambiguates the internal mechanism behind the L1 anchor's loss/recon performance.

## Cross-pollination with sister substrates

- **NSCS06 v8 chroma_lut MLX L1** (different substrate class; in flight parallel slot): zero scope collision. NSCS06 v8 is per-class deterministic LUT codec (Carmack-Hotz strip-everything paradigm); BoostNeRV is gradient-trained MLX residual (boosting paradigm). Per Catalog #294 dim 6 stack-of-stacks-composability: a future composition might apply NSCS06 v8 chroma LUT FIRST then BoostNeRV residual SECOND.
- **PR110-OPT-3 Variant B Markov context coder** (selector-stream codec): zero collision (orthogonal axis per Catalog #356 per-axis decomposition discipline).
- **T3 PR110 stacking ordering memo** (`t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`): currently 5-row ordering NSCS06 v8 #1 / grayscale_lut #2 / VQ-VAE #3 / ATW V2 REMOVAL #4 / DP1 DEFERRED #5. THIS sweep produces evidence that BoostNeRV-PR110 (best cell: gain_clamp=0.20, epochs=30) is a candidate for #6 — pending operator-routable d_seg+d_pose CUDA measurement that confirms net negative contest-ΔS.

## Operator-routable next steps (priority-ordered)

1. **HIGHEST EV** (codec-design FRONTIER-PUSH): refactor BPR1 codec to break int8 quantization scale-invariance. Two candidate designs:
   - **Variant B-a (signed-exponent variable-precision)**: store `(sign, exponent, mantissa)` 8-bit triplet per residual; exponent encodes gain_clamp-scaled magnitude. Predicted byte savings: gain_clamp=0.05 cells produce many zero-residual entries → highly compressible; gain_clamp=0.20 cells produce wide-distribution residuals → less compressible. Sweep validates whether codec dynamic range exploitation actually yields byte savings.
   - **Variant B-b (gain_clamp-dependent bit-width)**: int4 quantization at gain_clamp=0.05 (4× smaller), int8 at gain_clamp=0.10, int16 at gain_clamp=0.20. Direct test of "more clamp → more bits → more savings" hypothesis at codec level.
   - Sister subagent can iterate on $0 MLX-local in <30 min wallclock.

2. **NEXT** (Carmack operator-routable #2): paired CUDA dispatch on best-cell config `(gain_clamp=0.20, epochs=30)` to measure true d_seg+d_pose reduction. Cost: $0.20-0.50 if MLX→PyTorch export bridge (Catalog #1251) yields matching state_dict. The 52.6% recon-proxy MSE reduction is advisory; only paired CUDA SegNet+PoseNet routing per Catalog #164/#226 produces contest-axis truth.

3. **FOLD-INTO-STACKING** (if Carmack op-routable #2 confirms net negative ΔS): add BoostNeRV-PR110 best-cell config to T3 PR110 stacking ordering memo as candidate #6 (after NSCS06 v8 #1 / grayscale_lut #2 / VQ-VAE #3 / ATW V2 REMOVAL #4 / DP1 DEFERRED #5). Composition order TBD per sister stacking analysis.

4. **DEFER** (Catalog #344 operator-decision): registration of `residual_hybrid_boosting_savings_v1` canonical equation. The sweep's empirical anchors EXTENDED the proposed equation's domain knowledge but ALSO COMPLICATED its predicate (byte-axis scale-invariance under int8 quantization). Operator decision required whether to (a) register equation with revised predicate accounting for codec quantization regime; (b) defer pending Variant B codec landing; (c) reject candidate registration as too codec-design-specific for canonical-equation framework.

5. **DEFER** (real PR110-archive latent extraction per L1 operator-routable #2): the sweep used seeded-RNG cargo-culted z_latent; real PR110-archive-extracted per-pair latent would disambiguate "synthetic-latent artifact" from "real-residual artifact" but is orthogonal to the gain_clamp-axis question this sweep answers.

## Cross-references

- Pre-execution gate report: `.omx/research/boostnerv_pr110_gain_clamp_sweep_pre_execution_gate_report_20260526.md` (sister; same commit batch)
- Sister L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (commit `b2fd3e587`)
- Sister L1 EMPIRICAL pre-execution gate: `.omx/research/boostnerv_pr110_l1_empirical_pre_execution_gate_report_20260526.md`
- Design memo: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
- T3 PR110-stacking-ordering memo: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- Sweep harness script: `.omx/tmp/boostnerv_pr110_gain_clamp_sweep.py`
- Sweep heatmap artifact: `.omx/research/boostnerv_pr110_gain_clamp_sweep_results_20260526/sweep_heatmap.json`
- Bidirectional MLX↔CUDA drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Pushing-the-frontier directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`
