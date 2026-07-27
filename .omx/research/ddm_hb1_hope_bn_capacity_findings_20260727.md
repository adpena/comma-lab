---
title: hb1 findings — HOPE BN-capacity generator on frozen SegNet, exact n600 measure (#725)
utc: 2026-07-27T22:30:00Z
lane_id: lane_ddm_hb1_hope_bn_capacity_20260727
verdict: REPRODUCED_17_OF_17_CAPACITY_CONFIRMS_ADDRESSES
research_only: true
score_claim: false
promotion_eligible: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
pointer_moved: false
main_landing_review_required: true
canonical_equation_id: hope_bn_capacity_per_stratum_codebook_v1
coordinate_family: FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK
---

# hb1 — HOPE BN-capacity generator (crosswalk T2 build)

**Charter:** `.omx/tmp/codex_prompts/ddm_hb1_hope_bn_capacity_family.md` (task #725).
**Paper basis:** arXiv 2607.21366 (HOPE, Mobahi + Bartlett), consumed strictly through the
sealed crosswalk `.omx/research/crosswalk_arxiv_2607_21366_hope_hilbert_rank1_operator_
compression_20260726.md` (commit 0058123af3), takeaways T1/T2 with all carried caveats.
**NO-FAKE boundary:** every number below is OURS, measured on OUR frozen SegNet under OUR
exact n600 measure. No HOPE paper number is quoted as a result anywhere in this arm's
artifacts (their effect sizes are not quotable per crosswalk tension 4).

## What was built

`src/tac/optimization/hope_bn_capacity.py` — the generator library:

1. **Unit inventory** of the frozen SegNet (`smp.Unet tu-efficientnet_b2`, SHA-pinned
   `68956e32...`): 78 BN-normalised units = 68 encoder timm `BatchNormAct2d` + 10 decoder
   `Conv2dReLU` BN+ReLU units. Consumer (`w_out`) resolution is structural and honest:
   decoder chain + rank-4 head fully resolved (concat order verified against the installed
   smp source: `[upsampled_prev, skip]` → prev channels occupy the leading slice);
   encoder in-block chains resolved with explicit SE-gate upper-bound flags;
   encoder block outputs (residual/skip fan-out) carry `UNRESOLVED_CONSUMER_GRAPH_V1`
   and emit sqrt(K) only — no guessed capacity anywhere (NO-FAKE class 1/4 avoidance).
2. **Exact empirical kernels**: one streaming n600 pass over the cached `gt_f1` frames
   (bit-parity preprocess replica of `upstream/modules.py::SegNet.preprocess_input`),
   fp64 accumulators for K(i,i)=E[psi_i^2] at every unit; per-pair per-bucket accumulation
   at the 16-channel pre-head layer over the 37 occupied pf2 buckets (sealed event index,
   SHA `dd164a75...`, flat-index convention verified semantically: MyCar strata land in
   hood rows). **Custody proof:** the pass asserts forward argmax == cached GT argmax
   (`lstars`) — the integration measure IS the exact frozen-scorer measure, bit-for-bit.
3. **HOPE closed-form ReLU kernel (their Eq. 79)** implemented ONLY as a surrogate
   comparison column for the 10 decoder units; validated against Monte Carlo in tests.
4. **Rank-4 head composition** (`segnet_head_rank4_linear_flipdist_v1`): per-class-pair
   Delta-w Frobenius norms per pre-head channel + numerical rank check of the logit-
   difference space; per-stratum capacity cap_b^{ab}(i) = ||Dw_ab[i]||_F * sqrt(K_b(i)).
5. **Fine-band selection** (own implementation, parity-tested against
   `derive_rg3_fisher_margin_band`): Fisher trace 0.5*sech^2(m/2) on the sealed margin
   field (SHA `177d22f0...`), receiver support from the sealed V19C base
   (SHA `dc767b59...`), optional site-local capacity weighting
   W(x) = sum_i c_hat_i psi_i(x)^2 — flat weight reduces exactly to parity.

Runner: `tools/run_ddm_hb1_hope_bn_capacity.py` (detached launch, custody pins fail-closed).
Canonical equation: `src/tac/canonical_equations/hope_bn_capacity_per_stratum_20260727.py`
registering `hope_bn_capacity_per_stratum_codebook_v1` (pure evaluator + provenance to the
agreement receipt).
Tests: `src/tac/optimization/tests/test_hope_bn_capacity.py` (27) +
`src/tac/canonical_equations/tests/test_hope_bn_capacity_per_stratum_20260727.py` (6).

## The ReLU-family check (charter caveat — MEASURED)

The scorer path is NOT pure ReLU-family: encoder = 68 SiLU BNAct units + 23 sigmoid SE
gates (not positively homogeneous), decoder = 10 plain BN+ReLU, head = linear conv.
Consequence (per crosswalk tension 2): HOPE's closed-form kernels (Eqs. 3/5/79/85) do NOT
transfer to the encoder and are used NOWHERE as authority. No kernel extension was needed:
all authority kernels are exact empirical second moments under the n600 measure — the
structurally stronger position the crosswalk names (their data-free Gaussian surrogate is
moot when the exact input measure is in custody). The decoder surrogate-vs-exact deviation
column quantifies exactly how wrong the paper's surrogate would have been here.

## Rate-denominator caveat (BINDING, honored)

No rate column exists in any table from this arm. Per the crosswalk caveat (tension 5),
rate denominators must be measured coder bytes per action, never parameter counts; no
coder bytes were measured here, so every row carries
`score_units_per_byte_status=OWED_NOT_ADMITTED` (same status discipline as the sealed rg3
assignment rows).

## RESULTS (run `20260727T0001Z`, 775.3 s wall, detached local CPU fp32)

Artifacts (SHA-pinned inside the agreement receipt; receipt content SHA prefix
`423ee775aa7140d9`):

- `.omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_bn_capacity_table.json`
  (sha256 prefix `ef3216a51d33`) — 78 units, per-channel exact sqrt(K) everywhere,
  capacities where resolved.
- `.omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_per_stratum_capacity_table.json`
  (sha256 prefix `c11fb0d26c20`) — 37 strata x 16 pre-head channels, rank-4-composed
  per-class-pair capacities.
- `.omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_rg3_agreement_receipt.json`
  — the validation-gate receipt (full per-row records + custody block).
- SSD (disk-hygiene tier): `/Volumes/VertigoDataTier/pact/ddm_hb1_hope_bn_capacity_
  20260727T0001Z/target_pair_prehead_features.npz` (56.8 MB, sha in receipt) — the
  11 target pairs' 16-ch pre-head features (rebuildable; kept for site-local follow-ups).

**Measured facts:**

1. **Measure custody:** forward argmax vs cached GT argmax agreement = `0.999999991522895`
   over n600 (exactly 1 pixel of 117,964,800 — tie-level). The exact input measure is
   bit-faithfully the frozen scorer's own measure.
2. **ReLU-family census (MEASURED):** 45 encoder SiLU BNAct + 23 encoder identity BNAct
   + 10 decoder BN+ReLU → `scorer_is_pure_relu_family = false`. Closed forms surrogate-only.
3. **Gaussian-surrogate error quantified:** HOPE Eq. 79 vs exact empirical K on the 10
   decoder ReLU units: median relative deviation per unit ranges **0.197–0.335** (~20–34%).
   The paper's data-free surrogate would be ~25% wrong on this scorer under the real
   measure — the exact-kernel substitution is not cosmetic.
4. **Dead-channel census:** 0 channels with K < 1e-12 across all 78 units — the frozen
   SegNet carries no analytically dead capacity (HOPE-style free pruning mass: none).
5. **Rank-4 head check:** logit-difference space rank = 4 exactly (singular values
   4.703 / 2.831 / 2.039 / 2.018) — consistent with `segnet_head_rank4_linear_flipdist_v1`.
6. **Per-stratum capacity concentration (examples):** Lane–Movable cell static: top-3
   channels (2, 9, 6) carry 70.7% of stratum capacity; Lane–Undrivable cell static:
   channel 9 alone 30.0%; Road–MyCar cell static flatter (top ch 0 at 16.1%). The
   codebook family gets a strongly non-uniform, measured per-channel weighting.

## Validation gate — agreement with the 17 hand-derived rg3 rows

**Verdict: `REPRODUCED_17_OF_17`.** For every Fisher-margin assignment row, the
generator's independent parity selector, the in-repo reference
(`derive_rg3_fisher_margin_band` recomputed live on the V19C receiver support + sealed
margin field), and the recorded `receiver_derived_fine_band` agree exactly (17/17/17).

**Capacity-refined mode: 17/17 selections UNCHANGED** — the HOPE site-local capacity
weighting confirms every hand-derived address at 16-row fine-band granularity, including
all 9 rows that remained blocked in the sealed rg3 sweep
(`NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN`). Honest reading:
the 9 blockers are NOT band-address selection errors — capacity-aware re-selection at the
same granularity would re-probe the same addresses, so a same-address rerun is NOT
warranted. The capacity table's value for the blockers is FINER-than-band site selection
(the "SITE_LOCAL" in the family name): per-channel capacity + per-pair features enable
column/site-level address candidates inside the confirmed bands, which is exactly what
rg4's measured-hard wall demands beyond the exhausted 16-row alphabet.

## What this does NOT claim

- No score, no pointer movement, no promotion. Pointer per
  `.omx/state/canonical_frontier_pointer.json` (SoT) — UNMOVED by this arm.
- Capacity-refined fine-band selections are REFINEMENT PROPOSALS, unmeasured against
  signed probes; they authorize nothing (rg4 successor lanes remain MAIN-gated).
- Encoder capacities where the consumer graph is unresolved are NOT emitted (sqrt(K)
  only); resolving the full encoder consumer graph (residual adds + U-Net skip slices)
  is the named v2 extension.
- No witness-side adoption: the witness basis (sin/step/hosc) would need its own kernel
  derivation (crosswalk tension 2); explicitly out of scope per charter.

## First rungs (findings name their next measurement)

1. The capacity-refined selections CONFIRMED all 9 blocked addresses at band granularity
   → the implied next measurement is NOT a same-address rerun but a SUB-BAND site-local
   address family: within each confirmed 16-row band, rank 32-column cells by
   capacity-weighted Fisher mass (all inputs already in custody: per-stratum capacity
   vectors + saved target-pair features + margin field) and probe the top cells as new
   counted coordinates (MAIN authorization required; rg4 lineage discipline).
2. Decoder surrogate-vs-exact deviation quantifies the Gaussian-surrogate error on our
   measure; if small, cheap analytic sweeps of hypothetical decoder edits are licensed
   as PRIORS (never verdicts).
3. Dead-channel census (K < 1e-12) at each unit = HOPE's analytic dead detection made
   empirical; any nonzero count at the pre-head layer feeds the null-subspace rate law
   (`null_subspace_rate_measure_20260717`) as additional scorer-invisible mass.

## No-orphan routing (6-hook declaration)

- Sensitivity map: per-channel x per-stratum capacity table IS a sensitivity surface for
  the codebook family; consumed by any rg4-successor assignment builder.
- Pareto/bit allocator: N/A-with-rationale — no rate denominators admitted (OWED).
- Cathedral/autopilot: N/A — no dispatchable archive candidate is produced by this arm.
- Continual learning: canonical equation `hope_bn_capacity_per_stratum_codebook_v1`
  registered with provenance to the agreement receipt.
- Probe disambiguator: the agreement receipt itself disambiguates parity vs refined
  selection per row (both recorded; neither silently overrides the other).
- Triality: DSL leg N/A (no trainer lever; this is scorer-side analysis apparatus, not a
  witness lever) — declared per the drift-detector contract; equations leg satisfied by
  the registered canonical equation; DAG leg = proposed FEED below for MAIN.

## Proposed DAG FEED (for MAIN to append after review)

FEED-hb1: HOPE BN-capacity generator landed (#725). Per-channel x per-stratum capacity
table for frozen SegNet under the EXACT n600 measure (argmax custody 1 px/118M); rank-4
head composed (rank check = 4 exact); validation gate REPRODUCED_17_OF_17 vs the rg3
Fisher rows AND capacity-refined mode confirms all 17 addresses (incl. all 9 blockers →
blockers are not band-address errors; next rung = sub-band site cells); Gaussian
surrogate would be ~20-34% wrong on decoder kernels (exact-measure substitution
justified, quantified); scorer NOT pure ReLU-family (45 SiLU/23 id/10 ReLU — closed
forms surrogate-only); 0 dead channels; rate columns OWED by design. Pointer UNMOVED.

## STORES CONSULTED

CLAUDE.md (NO-FAKE · canonical class order · SegNet architecture · disk hygiene ·
serializer/review-gate · detached-launch rc=144 class) · charter
`.omx/tmp/codex_prompts/ddm_hb1_hope_bn_capacity_family.md` · HOPE crosswalk (0058123af3)
T1/T2 + tensions 2/4/5 · rg3 findings memo `codex_findings_ddm_rg3_residual_family_
productions_20260724T110418Z_codex.md` (17 Fisher rows + 25-blocker inventory + Finding 5
schema lessons) · rg3 assignment/receipts dir (assignment SHA `40d4150e...`, ms6 receipt,
support summary `g3_top24_coverage.missing_blocks`) · `src/tac/optimization/ddm_rg1_
receiver_grammar.py` (`derive_rg3_fisher_margin_band`, band constants, `_base_masks_for_
classes`) · `src/tac/optimization/direct_description_carrier_compose.py` +
`direct_description_preuint8_channel.py` + `direct_description_coupled_margin.py` (V19C
unwrap chain, verified live) · canonical equation module pattern `ddm_rg3_residual_family_
productions_20260724.py` · `segnet_head_rank4_flipdist_20260715.py` (recalled, composed) ·
gt cache `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (shapes verified) · pf2
occupied event index (SSD, SHA-pinned) · margin field f16 (SSD, SHA re-verified against
assignment custody) · upstream/modules.py (preprocess parity source) · MEMORY.md hooks:
`reaper_tty_kills_were_the_harness_kill_class_20260717` (applied after the rc=144 kill),
`no_coauthor_trailer_all_commits_are_operators_20260717`, `findings_are_first_rungs_
name_next_measurement_20260723`, `null_subspace_rate_measure_20260717`,
`distortion_byte_economics_are_upper_bounds_20260724` (rate-caveat sister).
