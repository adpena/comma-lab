# Codex findings: DDM v18b common-master rebaseline and pricing

UTC: 2026-07-23T10:12:37Z  
Lane ID: `ddm_v18_column_generation_vocabulary`  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
Score claim: `false`  
Pointer: `0.1910828242 [contest-CPU]`, **UNMOVED**

## Verdict

`FALSIFIED_FORMULATION_THREE_CLEAN_PRICING_ROUNDS`

Verdict scope: `FORMULATION:COLUMN_FAMILIES_AND_THREE_ROUND_GLOBAL_SELECTION`.
This closes the preregistered Probe B formulation on this common exact-R master.
It does not close residual, G1, template-DOF, shearlet, or direct-description
families under a different formulation.

The immutable authority is
`.omx/research/ddm_v18b_common_master_pricing_20260723T050800Z/ddm_v18b_common_master_pricing_receipt.json`,
SHA-256
`0d7e3535905cd48d42d7caeb6cfa8f56486a781bf16bbcb58cbe34afab014f55`.
Its producer is Git
`9e2f20be3f2752a587d95571b43eae097680b196`; post-campaign hardening is
separate at `817feaf972e99da4f456ada2c3c298f7c0aff7d8`.

## Common receiver and rebased v12 control

The post-solve-only common master is 103,629 bytes, SHA-256
`50332acf742717f463111cc0ead2878c33a9e5d4fa7cc15dee9329bdafca8714`.
It carries no PREDICT productions, scorer weights, or ground-truth argmax
table. The current camera-resolution uint8 receiver precedes evaluator R.

The exact fixed v12 inventory remained 4,096 atoms in 353 immutable bundles.
Sequential greedy resolved all 353 rows: 43 exact admissions, 294 exact
non-wins, 6 address conflicts, and 10 exact receiver-output no-op refusals.
The latter are per-row verdicts only.

| Added-byte cap | Selected bundles | Realized added bytes | Archive bytes | d_seg | Delta vs legacy `0.034003668891` | d_pose | Joint objective |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 43 | 4,091 | 107,720 | 0.034075563219 | +0.000071894328 | 163.032252598150 | 43.856535229213 |
| 49,152 | 43 | 4,091 | 107,720 | 0.034075563219 | +0.000071894328 | 163.032252598150 | 43.856535229213 |
| 98,304 | 43 | 4,091 | 107,720 | 0.034075563219 | +0.000071894328 | 163.032252598150 | 43.856535229213 |
| 147,456 | 43 | 4,091 | 107,720 | 0.034075563219 | +0.000071894328 | 163.032252598150 | 43.856535229213 |

Every control row reconstructs archive SHA-256
`60bc8c026bf74652dd8a5ba81078d37f7f0319b3325a1b3d9377c222ab993cb6`.
The cap rows are flat because the fixed pool was exhausted after only 4,091
realized added bytes. The empty common master itself measured d_seg
0.034717763265, which is +0.000714094374 versus the legacy scorer-grid value;
the 43 admitted bundles recover most, but not all, of that receiver-contract
sensitivity.

## Probe B pricing rounds

Each round priced 64 previously unseen columns on the n64 screen and used
exact receiver replay as authority. Frozen full-chain Jacobian prediction was
excluded per the 2026-07-23T04:10:16Z directive.

| Round | New columns | Negative reduced cost | Global selector | Accepted columns | n600 accepted-set archive |
|---:|---:|---:|---|---:|---|
| 1 | 64 | 0 | `beam_width_32` | 0 | 103,629 bytes / `50332acf...` |
| 2 | 64 | 0 | `beam_width_32` | 0 | 103,629 bytes / `50332acf...` |
| 3 | 64 | 0 | `beam_width_32` | 0 | 103,629 bytes / `50332acf...` |

The exact n600 generated-set replay is therefore the common base:
d_seg 0.034717763265, d_pose 163.037949029624, joint objective
43.918736599760.

## Same-receiver byte-cap comparison

| Cap | v12 realized added bytes / objective | Generated realized added bytes / objective | Exact gap (generated - v12) | Generated beats v12 |
|---:|---:|---:|---:|---|
| 16,384 | 4,091 / 43.856535229213 | 0 / 43.918736599760 | -4,091 | false |
| 49,152 | 4,091 / 43.856535229213 | 0 / 43.918736599760 | -4,091 | false |
| 98,304 | 4,091 / 43.856535229213 | 0 / 43.918736599760 | -4,091 | false |
| 147,456 | 4,091 / 43.856535229213 | 0 / 43.918736599760 | -4,091 | false |

The generated side could not spend up to the rebased control because all
three pricing rounds found zero admissible negative-reduced-cost columns.
This exact under-cap gap is reported rather than pretending equal utilization.

Coder dispositions:

- `unstructured_explicit_indices`: `MEASURED_RECEIVER_CLOSED`, 103,629 bytes.
- `structured_nm_2_of_4_coding_order`:
  `NOT_APPLICABLE_TYPED_ATOM_STREAM_NOT_DENSE_2_OF_4`.
- `mx_block_shared_scale_int4`:
  `QUEUED_NOT_MEASURED_NO_COMMON_RECEIVER_DECODER`.
- `permutation_gauge_canonical_address_order`:
  `MEASURED_TRIVIAL_EMPTY_SELECTED_PAYLOAD`, canonical 103,629 bytes versus
  as-is 103,629 bytes, delta 0. The candidate pool is not counted payload;
  no training or merging claim is made. Supplemental receipt:
  `.omx/research/ddm_v18b_common_master_pricing_20260723T050800Z/ddm_v18b_permutation_gauge_coder_receipt.json`.

## Falsifier derivation

The preregistered conjunction is true:

1. exactly three complete pricing rounds;
2. negative-reduced-cost counts `[0, 0, 0]`;
3. exact accepted-set n600 replay complete for all three rounds;
4. no same-receiver byte-cap row beats the rebased v12 control.

Therefore the formulation falsifier triggers. No score, promotion, or pointer
claim follows from this macOS-CPU advisory evidence.

## Apparatus hardening

Ten fixed-pool bundles legitimately compiled to exact receiver-output no-ops.
During the custodied campaign, each was recovered with an explicit typed
rejection checkpoint while the immutable producer remained unchanged.
Commit `817feaf972` makes that path automatic for future runs, but catches
only `DirectDescriptionError` messages containing the exact
`receiver-output no-op` token; every other compiler error still raises.
It also records the later permutation-gauge coder entrant. Verification:
8 focused tests, 28 integrated tests, Ruff, pycompile, diff check, and three
consecutive clean review passes.

## Triality and pointer delta

- DSL: `DDMA1ColumnGeneratedCorrectionConfigV1`, typed config SHA-256
  `631379cfc261e90a91af0fe3c7523293184c38656b5e5a93efca8b7e0a48083c`.
- DAG: `.omx/research/ddm_v18_column_generation_vocabulary_DAG_FEED_20260723.md`,
  keyed by lane ID `ddm_v18_column_generation_vocabulary`.
- Equations: `tac.canonical_equations.ddm_v18_column_pricing_law_20260723`;
  exact replay, not frozen-Jacobian prediction, remains admission authority.
  No new pricing law stabilized, so no equation mutation was warranted.
- Pointer delta: exactly zero; `pointer_moved=false`.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
the delegated authority SHA-256 `86526d862ed35afb2aaa363b2eb91f0eab103d581f7a7d6f4223fed24b722fd2`;
the a1 preregistration; the v18 blocker memo and DAG; `reports/latest.md`;
`.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`;
the fixed v12 candidate/checkpoint inventory; v15/v16 receipts and source
archives; the frozen scorer/target custody surfaces; the per-arm and broadcast
inboxes. The nested-Jacobian and permutation-gauge directives were consumed.
The aborted `T040000Z`, `T042300Z`, and `T043000Z` directories are retained
as fail-closed forensic inputs and excluded from the authoritative result.

## Re-derivation

```bash
.venv/bin/pytest -q \
  src/tac/optimization/tests/test_ddm_v18_common_exact_r_master.py \
  src/tac/optimization/tests/test_ddm_column_generation.py \
  tools/tests/test_probe_ddm_a1_column_generated_correction.py \
  tools/tests/test_run_ddm_v18b_common_master_pricing.py

shasum -a 256 \
  .omx/research/ddm_v18b_common_master_pricing_20260723T050800Z/ddm_v18b_common_master_pricing_receipt.json

jq '{v12_rebased_control_rows,pricing_round_history,equal_byte_rows,falsifier,verdict,verdict_scope,pointer_moved,score_claim}' \
  .omx/research/ddm_v18b_common_master_pricing_20260723T050800Z/ddm_v18b_common_master_pricing_receipt.json
```

MAIN landing review is required before merge. Review the common-master
receiver boundary, the exact no-op classifier scope, the equal-byte underfill
semantics, and the formulation-only falsifier scope.
