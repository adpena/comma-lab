# VEH vs CAP — the instrument-matched n32 verdict pair (correction fork CLOSED)

Tags: [no-triality] [p0-ledger-ok]. Axis for every number here:
**[macOS-CPU advisory torch upstream SegNet]** — `score_claim=false`, `promotion_eligible=false`,
n32 arm-instrument scope, NEVER a contest row.

## The A/B (what each arm tests)

Same w96 receiver architecture, same repro-repo init (`semantic_renderer_w96_b4_qat4_12k.pt`),
same probe config (lr 2e-7, n32), same 32 pairs, same exact-path roundtrip
(bilinear↑874×1164 → uint8 STE → bilinear↓384×512), same frozen CPU-torch upstream SegNet,
same facets instrument (`--mode torch-facets`, batch 32, repro head `2f94596bb013…`):

- **ARM-CAP** (gt→gt): trains toward GT labels FROM GT labels — the receiver-class
  **capacity control** (how well can this class render the partition when the input carries
  full information).
- **ARM-VEH** (tq1c→gt): trains toward GT labels FROM tq1c tokens — the **recoverability
  test** (can decode-side training resurrect information the tq1c encode destroyed).

## The measured pair (both through the SAME instrument, facets_v2 protocol)

| arm | K=8 tail d_seg | final-step d_seg | steps | receipt |
|---|---:|---:|---:|---|
| CAP (gt→gt) | **0.0010862350** | 0.0010890961 @6000 | 6000 | `endpoint_facets_v2/facets_result.json`, commit 9ff40f2f6a |
| VEH (tq1c→gt) | **0.0045445760** | 0.0045092901 @5000 | 5000 (adjudicated early stop) | `endpoint_facets_veh/facets_result.json` (this commit) |

Instrument identity verified by MAIN: pair_ids byte-equal (n=32), segnet_batch_size 32 both,
contest_faithful_roundtrip string equal, source_repo_head equal. VEH determinism anchor:
step-1500 reproduced `0.004514535268147786` at **abs_diff 0.0** (tolerance 1e-12); instrument
correctness is carried by CAP's mx1h-anchored pass through the identical code path. Run-1's
fail-closed blocker (lineage-bound ARM-CAP anchor default fired on VEH — the positive control
working) preserved as `MX1T_ANCHOR_BLOCKER.run1_wrong_lineage_anchor.json`.

## Verdict

**VEH/CAP = 4.18×.** VEH's whole 5000-step trajectory wanders 0.004397–0.004678 and ends at
0.004509 ≈ its own step-1500 entry error (0.004515) — FLAT at the input's own error while the
surrogate loss fell ~10.6%. The mismatch composition barely moves (near-margin fraction rises
0.126→0.161, churn/current falls to 0.006 — `low_churn_stable_residual`); the tool's own
iteration verdict: `do_not_assume_more_steps_pay_without_new_objective_or_capacity_change`.

**Correction fork CLOSED — verdict_scope: FORMULATION.** Under this configuration (6k-step
lr-2e-7 probe from the repro 12k init, n32 advisory, convenience-shared config per the
naive-first-pass law), decode-side training does NOT recover information destroyed at encode:
the data-processing reading stands, and the composed vehicle carries BETTER TOKENS instead of
hoping the receiver repairs bad ones. NOT a FAMILY verdict: a reference-form corrective
receiver (proper training length, flip-targeted losses, n≥120 stratified) was not raced — but
the CAMPAIGN routing does not wait on that race, because the same experiment's CAP arm shows
the capacity is real (0.00109 from good input) and PR130's shipped renderer proves the class
reaches 2.97e-4 at n600. **The binding constraint is carriage quality + receiver training
budget, not decode-side correction.**

## Consumers

- **#984 arithmetic sheet, receiver term:** receiver-class capacity 0.00109 (n32 advisory,
  6k-step probe) vs PR130's converged 2.97e-4 — the gap is TRAINING/CONFIG DEBT, closed by
  the n120 reference-form build (PR130 renderer form off-the-shelf grant + our flip-targeted
  losses), not by architecture search.
- **lx1 term-3 row** (receiver): cite this pair as the CAP/VEH receipt named in the charter.
- **gt-HPAC line (#982):** carriage of GT-quality labels at HPAC price (112,044 B measured on
  tq1c labels; gt-arm training, ep60 ~1-2d) is the vehicle's label stream; VEH's flatness is
  the measured reason the vehicle does NOT budget a correction stage.

## Instrument debt (filed, #977 leg)

1. `torch-facets` findings template hardcodes "ARM-CAP" in verdict/scope lines (corrected by
   hand in this run's findings; writer should take the arm name from cache provenance).
2. `--facet-anchor-d-seg` default is lineage-bound to the CAP/mx1h arm — anchor
   expected-values are per-arm config; the registry row should record that defaults of
   positive-control anchors must be arm-explicit at launch (this is what run-1's blocker
   caught, correctly).
