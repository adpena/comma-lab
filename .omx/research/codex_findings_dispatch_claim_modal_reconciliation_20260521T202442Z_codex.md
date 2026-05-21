# Codex Findings: Dispatch Claim × Modal Ledger Reconciliation

**UTC:** 2026-05-21T20:24:42Z  
**Lane:** `dispatch_claim_modal_reconciliation_20260521`  
**Verdict:** PROCEED — active Modal claim blocker reduced from 13 to 1 without touching partner WIP or live submission files.

## What landed

- Added `tools/reconcile_dispatch_claims_with_modal_ledger.py`.
- Added `src/tac/tests/test_reconcile_dispatch_claims_with_modal_ledger.py`.
- Registered lane `dispatch_claim_modal_reconciliation_20260521`.
- Ran the reconciler against the live dispatch claim ledger and Modal call-id ledger.

## Empirical result

Initial reconciliation artifact:

- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation.json`
  - SHA-256: `07cd649f5a4bc15a8cee72b8ec1f359404861d1a73b3c3d0acf302a0ae407f81`
- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation.md`
  - SHA-256: `f5f7c90ad10a4a923a9daf44700bde3cbbbf04442f0187b42df2113c3b44b548`

Executed-closure artifact:

- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation_after_execute.json`
  - SHA-256: `dbed774b80e5c4515543fd16cebed47fd7578dee10b5d40fac7332feb41416ff`
- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation_after_execute.md`
  - SHA-256: `631dafd89b26d6b6a91a2b24103e3800e148f186fb755aadbf0e7e393276378a`

Final post-closure artifact:

- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation_final.json`
  - SHA-256: `fce1a5577e4fb2a55196e2ffaf519239a42645e08ca3ddb4071255301ab4d079`
- `experiments/results/dispatch_claim_modal_reconciliation_20260521T202327Z/reconciliation_final.md`
  - SHA-256: `a70fff947361936b0c26697961c8019c26976798e393e9728cf29d02f6744f48`

## Terminal closures appended

The tool found and appended 8 high-confidence terminal closures where an active claim had an exact `call_id` or exact `(lane_id, label)` Modal ledger match:

| closed lane | job | terminal status |
|---|---|---|
| `lane_dp1_baseline_paired_auth_eval_post_gg361_20260521_contest_cpu` | `dp1_path_a_baseline_post_gg361_4_arm_paired_modal_auth_20260521T181112Z_cpu` | `completed_modal_auth_eval_recovered` |
| `lane_dp1_baseline_paired_auth_eval_post_gg361_20260521_contest_cuda` | `dp1_path_a_baseline_post_gg361_4_arm_paired_modal_auth_20260521T181112Z_cuda` | `completed_modal_auth_eval_recovered` |
| `lane_dp1_original_baseline_first_paired_anchor_20260520` | `fc-01KS5RSNWQCYF5PR3KYPM8S9J9` | `completed_modal_training_recovered_no_score_claim` |
| `lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520` | `fc-01KS5RV15HVMFF39CHR2BJHKQ8` | `completed_modal_training_recovered_no_score_claim` |
| `lane_dp1_procedural_paired_auth_eval_post_gg361_20260521_contest_cpu` | `dp1_path_a_procedural_post_gg361_4_arm_paired_modal_auth_20260521T181215Z_cpu` | `completed_modal_auth_eval_recovered` |
| `lane_dp1_procedural_paired_auth_eval_post_gg361_20260521_contest_cuda` | `dp1_path_a_procedural_post_gg361_4_arm_paired_modal_auth_20260521T181215Z_cuda` | `completed_modal_auth_eval_recovered` |
| `lane_overnight_vv_nscs06_v8_phase_4_retry_with_catalog_202_bypass_20260521` | `fc-01KS5XN8WF9JF15KVX3GPCFAE7` | `failed_modal_call_recovered_rc_1` |
| `lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521` | `substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T184408Z` | `failed_modal_call_recovered_rc_1` |

Then 4 pending/meta claim rows were closed as stale/superseded after manual evidence review:

- `pending-spawn-overnight-dd`: same-lane real dispatch had already terminalized with `failed_pre_spawn_init_fatal_rc22`.
- `pending-spawn-qq`: same-lane real dispatch had already terminalized with `failed_recipe_vs_driver_state_divergence_rc22_diagnosed_overnight_rr`.
- `pending-spawn-vv`: spawned `call_id=fc-01KS5XN8WF9JF15KVX3GPCFAE7` terminalized as `failed_modal_call_recovered_rc_1`.
- `pending-spawn-baseline-then-procedural`: child DP1 training and paired auth-eval claims terminalized.

## Current dispatch state

After reconciliation:

```text
CLAIM_SUMMARY active=1 stale_nonterminal=1 terminal_latest=1054 unparsable_timestamp=0 invalid_lane_id=4
ACTIVE lane_id=lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521 job=substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185859Z platform=modal status=active_dispatch agent=claude:operator_authorize
STALE_NONTERMINAL lane_id=lane_master_gradient_post_decompress_grain_multi_archive_extension_20260519 job=multi_archive_post_decompress_grain_extension_20260519 platform=local status=active_phase1 agent=claude
```

Selfcomp A100 call remains live/unharvested:

```text
call_id=fc-01KS5YG9W26T72D6Z8Y3N44JEN
lane=lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521
gpu=A100
dispatched_at=2026-05-21T13:59:26.288439
```

## Verification

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/reconcile_dispatch_claims_with_modal_ledger.py \
  src/tac/tests/test_reconcile_dispatch_claims_with_modal_ledger.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_reconcile_dispatch_claims_with_modal_ledger.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/claim_lane_dispatch.py summary

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/harvest_modal_calls.py \
  --from-ledger --call-id fc-01KS5YG9W26T72D6Z8Y3N44JEN
```

Result:

- Unit test: `1 passed`.
- Active Modal dispatch blockers: `13 -> 1`.
- Remaining live dispatch is not closed because the Modal ledger latest status is still `dispatched`.

## Next action

Poll or harvest `fc-01KS5YG9W26T72D6Z8Y3N44JEN` when terminal. Do not launch the HFV exact-eval candidate while this Selfcomp A100 job remains live unless the operator explicitly accepts parallel dispatch against the active claim.
