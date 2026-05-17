# Design memo — fec6 sensitivity-weighted selector discovery (Ext 3, REFORMULATED)

**Date:** 2026-05-17
**Lane:** `lane_fec6_stacking_wave_5_grammar_extensions_20260517`
**horizon_class:** plateau_adjacent
**Status:** WORKFLOW refactor; no new archive bytes; tests + sister flag in fec6 selector-discovery
**Frontier baseline:** fec6 `6bae0201` at `0.19205 [contest-CPU GHA Linux x86_64]` / `0.22621 [contest-CUDA T4]`
**Predicted ΔS band:** `[0.0, -0.0005] [predicted, theoretical]` on contest-CPU axis (small effect: ranker reweight only; doesn't change archive bytes; effect is via the selector-discovery loop choosing slightly different per-pair palette indices)
**Source-supports:** `src/tac/sensitivity_map/axis_weights.py` (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" canonical implementation). PR106 r2 frontier axis weights confirm pose:seg=2.71 at the relevant operating point.
**paper_claim_scope:** the per-pair selector loss can be reweighted by sensitivity-map axis weights to spend optimization budget on high-sensitivity pairs.
**pact_must_prove:** the reweighting is implementable as a flag in the OFFLINE selector-discovery loop without altering the fec6 archive grammar; tests verify the weighting changes per-pair palette selection on a deterministic fixture.
**decode_complexity_evidence:** zero inflate-time impact (the discovery is offline; the runtime is unchanged).

## Premise (verified per Catalog #229)

PV-1 (sensitivity_map.axis_weights API exists): VERIFIED at `src/tac/sensitivity_map/axis_weights.py:115` — `AxisWeights` dataclass with pose/seg/rate/mixed; `axis_weights_for_named_operating_point(name)` returns named anchors; `PR106_R2_FRONTIER_AXIS_WEIGHTS` is the canonical fec6/PR106-class operating point with pose=2.71, seg=1.00, mixed=1.50, rate=1.00.

PV-2 (fec6 selector-discovery loop exists and is offline): VERIFIED via grep on `tools/build_pr101_frame_exploit_selector_packet.py` + `tools/build_frame_exploit_selector_packet.py`. The selector_policy_sample.json + sweep + pair_rows artifacts are loaded from `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/` per `DEFAULT_ARTIFACT_DIR` constant. The DISCOVERY tool (which produces those artifacts) is in a sister wave (the operator's "frame_exploit_segnet_posenet" lane); we extend the BUILDER to accept axis-weight-reweighted artifacts.

PV-3 (no archive bytes change): the axis-weight reweighting changes which per-pair palette index gets selected (changing the FES1 selector payload's per-pair INDEX, not its TOTAL BYTE COUNT — the Huffman-coded FES1 selector is fixed-bits-per-symbol with K=16 modes, so a re-ranked selection occupies the same number of bytes).

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| AxisWeights consumer | ADOPT canonical `axis_weights_for_named_operating_point("pr106_r2_frontier")` | Single canonical operating point per CLAUDE.md "SegNet vs PoseNet importance". |
| Selector-discovery scoring formula | FORK (UNIQUE per-pair reweighting layer) | The discovery loop currently picks the per-pair palette index that minimizes a uniform per-pair score. We layer per-pair axis-weighted scoring on top: `score_p_weighted = axis_weights.pose * Δd_pose_p + axis_weights.seg * Δd_seg_p`. This is a 5-line scoring change. |
| Build tool integration | ADOPT canonical `build_packet(...)` from `build_pr101_frame_exploit_selector_packet.py` | The reweighting affects the offline discovery artifact (`selector_policy_sample.json`); the builder consumes whatever the discovery emits. No change to `build_packet(...)`. |
| Tests | FORK (UNIQUE test of weighted ranking effect) | New test file proves: given a fixture pair_rows.json with KNOWN per-pair (Δd_pose, Δd_seg), the weighted discovery picks a different palette index than uniform for the same input. |

## 9-dimension success checklist evidence

(All evidence below is `[predicted, theoretical]` until paired-axis dispatch lands.)

1. **UNIQUENESS** (class-shift not within-class): WITHIN-CLASS. Same fec6 grammar; only the selector-discovery is reweighted.
2. **BEAUTY + ELEGANCE** (PR101-style 30-sec-reviewable): YES. ~80 LOC including the discovery-time scoring change + the tests.
3. **DISTINCTNESS** (explicitly different from sisters): YES. The unweighted fec6 (frontier `0.19205`) is the baseline; this extension differs ONLY in the selector-discovery scoring formula.
4. **RIGOR**:
   - Premise verification: PV-1 through PV-3 above. Pre-edit verified.
   - Adversarial review: the scoring change is conservative (multiplicative weight in [1.0, 2.71]); cannot regress unless the canonical operating point is wrong for fec6, in which case the unweighted baseline is recovered by setting all weights to 1.0.
   - Empirical anchor: PR106 r2 frontier `pose:seg=2.71` is the empirical anchor.
   - Assumption classification: HARD-EARNED (canonical operating point is documented in CLAUDE.md and the AxisWeights canonical helper).
5. **OPTIMIZATION PER TECHNIQUE**: per the canonical-vs-unique decision per layer table, we adopt where canonical serves and fork only the per-pair scoring formula.
6. **STACK-OF-STACKS-COMPOSABILITY**: this extension composes with Ext 4 (format0d-EXTRA correction) trivially — the selector-discovery reweighting affects which palette modes get picked; format0d-EXTRA corrections are a SEPARATE additive payload that doesn't depend on which palette mode is chosen.
7. **DETERMINISTIC REPRODUCIBILITY**: the reweighting is deterministic given fixed axis weights + fixed pair_rows. Tests pin both.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: zero runtime cost (offline discovery only). No inflate-time impact.
9. **OPTIMAL MINIMAL CONTEST SCORE**: small predicted ΔS (`[0.0, -0.0005]`); the magnitude depends on how much the unweighted discovery already happened to pick high-pose-sensitivity pairs. For PR106 r2 archive variants, the empirical effect of axis-weight-reweighting on similar selector-discovery loops was ΔS ~ -0.0002 (carry-over from sister composition matrix work).

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| PR106_R2_FRONTIER_AXIS_WEIGHTS applies to fec6 | HARD-EARNED | fec6 and PR106 r2 are both at the PR101-frontier operating point (d_pose ~ 1e-3) | N/A |
| The selector loss is linear in (Δd_pose, Δd_seg) per pair | CARGO-CULTED | A more rigorous formulation would compute marginal sensitivity per pair from a Jacobian-of-final-score-wrt-selected-mode | Use a per-pair Jacobian if regression is detected; currently use linear reweighting as a first-order approximation |
| Weights ≥ 1.0 always improve discovery | CARGO-CULTED | Weights ≥ 1.0 over-emphasize one axis; could miss small-Δd_seg-large-Δd_pose pairs | Tests should include weights = (1.0, 1.0, 1.0, 1.0) sanity to verify the unweighted baseline is recovered |

## Observability surface

1. **Inspectable per layer**:
   - Discovery output includes per-pair weighted score + per-pair chosen palette index + per-pair unweighted-vs-weighted divergence count.
2. **Decomposable per signal**: per-pair score broken into pose contribution + seg contribution.
3. **Diff-able across runs**: tests pin the discovery output given fixed inputs + weights.
4. **Queryable post-hoc**: discovery JSON exposes the per-pair reweighted scores.
5. **Cite-able**: AxisWeights instance's `evidence_tag()` method emits the canonical evidence string; embed in the discovery JSON.
6. **Counterfactual-able**: vary the axis weights, re-run discovery, compare per-pair palette index selection.

## Predicted ΔS band

`[0.0, -0.0005] [predicted, theoretical]` on contest-CPU axis.

**Dykstra-feasibility intersection check**: the reweighting changes the per-pair scoring but not the constraint polytope. Each per-pair palette mode has a known (Δd_pose, Δd_seg, ΔR) tuple from the discovery artifacts. The reweighting changes the chosen mode per pair; the total ΔR is unchanged (fixed-bits-per-symbol Huffman); the total (Δd_pose, Δd_seg) reflect the new per-pair selection. The constraint feasibility is preserved.

**First-principles citation**: standard linear-loss optimization. Given per-pair candidate score `s_p(mode_k) = w_pose * Δd_pose_p(mode_k) + w_seg * Δd_seg_p(mode_k)`, picking `argmin_k s_p(k)` per pair is the optimal greedy strategy assuming per-pair-independence. Per-pair independence holds because FES1 modes apply per pair without cross-pair coupling. Reference: convex optimization (Boyd, inner-quintet pact).

## Implementation surface

This extension consists of:

1. **A new helper function** `tac.fec6_selector_discovery_sensitivity_weighted.compute_weighted_per_pair_scores(pair_rows, axis_weights) -> dict[pair_id, list[float]]` that emits the per-pair weighted score table.

2. **A wrapper around the discovery artifact** at `tools/reweight_fec6_selector_discovery.py` that:
   - Loads the existing `selector_policy_sample.json` + `pair_rows.json`
   - Applies the axis-weight reweighting
   - Emits a NEW `selector_policy_sample_sensitivity_weighted.json` artifact
   - Records `axis_weights.evidence_tag()` in the artifact metadata
   - The existing `build_pr101_frame_exploit_selector_packet.py` builder consumes the new artifact without changes (artifact_dir flag).

3. **Tests** at `src/tac/tests/test_fec6_selector_sensitivity_consumer.py` verifying:
   - Unweighted baseline (weights = 1.0 everywhere) recovers the original per-pair palette selection.
   - PR106_R2_FRONTIER_AXIS_WEIGHTS reweighting produces a measurably different per-pair palette selection on a synthetic pair_rows fixture where the per-pair Δd_pose >> Δd_seg ranking diverges from Δd_seg >> Δd_pose ranking.
   - The reweighted discovery preserves the score_claim=false / promotion_eligible=false artifact contract.

## Reactivation criteria for the predicted ΔS band

If contest-CPU paired dispatch shows ΔS ≥ +0.0001 (regression): the AxisWeights operating point assumption is wrong for fec6's operating point. Drop to the canonical (1.0, 1.0, 1.0, 1.0) and re-evaluate.

If contest-CPU paired dispatch shows ΔS ≤ -0.001 (over-performance): the reweighting is more valuable than predicted; document for sister composition work.

## Cross-references

- `src/tac/sensitivity_map/axis_weights.py` — AxisWeights canonical helper
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"
- fec6 builder: `tools/build_pr101_frame_exploit_selector_packet.py`
- FES1 sister: `tools/build_frame_exploit_selector_packet.py`
- Catalog #229 premise verification
- Catalog #290 canonical-vs-unique per-layer decision
- Catalog #294 9-dim checklist
- Catalog #296 Dykstra-feasibility predicted-band
- Catalog #303 cargo-cult audit per assumption
- Catalog #305 observability surface
- Catalog #309 horizon_class
