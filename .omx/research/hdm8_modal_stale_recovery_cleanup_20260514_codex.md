# HDM8 Modal stale recovery cleanup - 2026-05-14

## Reason

Four detached Modal HDM8 postfilter sweeps were still active in the dispatch
ledger more than three hours after their predicted ETA. A default recovery poll
and a 60-second blocking recovery on the oldest call both returned
`status=pending`, with no terminal result materialized.

Per the provider fail-fast rule, these are treated as stale provider calls, not
method negatives. They were cancelled to stop stale spend and unblock the live
claim ledger.

## Cancelled calls

| lane_id | job_id | call_id | final claim status |
|---|---|---|---|
| `hdm8_modal_t4_postfilter_policy_sweep_20260514` | `hdm8_modal_t4_policy_palette_v1_20260514T113221Z` | `fc-01KRK453W4GT5A99XFTMQ3KXMF` | `stopped_stale_pending_recovery_timeout` |
| `hdm8_local_first_postfilter_cuda_confirm_20260514` | `hdm8_local_first_cuda_confirm_20260514T115726Z` | `fc-01KRK5KKSQYCCMAG95CN4FCE0K` | `stopped_stale_pending_recovery_timeout` |
| `hdm8_multiplicative_cuda_probe_20260514` | `hdm8_multiplicative_cuda_probe_20260514T121530Z` | `fc-01KRK6M42CB2VGZCZASGHBSV72` | `stopped_stale_pending_recovery_timeout` |
| `hdm8_tile_chroma_cuda_probe_20260514` | `hdm8_tile_chroma_cuda_probe_20260514T123320Z` | `fc-01KRK7KYQVJH4BCPBBDF93BHRR` | `stopped_stale_pending_recovery_timeout` |

## Evidence

Recover summaries are preserved under:

- `experiments/results/modal_hdm8_postfilter_sweep/hdm8_modal_t4_policy_palette_v1_20260514T113221Z/modal_hdm8_postfilter_sweep_recover_summary.json`
- `experiments/results/modal_hdm8_postfilter_sweep/hdm8_local_first_cuda_confirm_20260514T115726Z/modal_hdm8_postfilter_sweep_recover_summary.json`
- `experiments/results/modal_hdm8_postfilter_sweep/hdm8_multiplicative_cuda_probe_20260514T121530Z/modal_hdm8_postfilter_sweep_recover_summary.json`
- `experiments/results/modal_hdm8_postfilter_sweep/hdm8_tile_chroma_cuda_probe_20260514T123320Z/modal_hdm8_postfilter_sweep_recover_summary.json`

After terminal claims, the live dispatch summary has only:

- `lane_c6_ibps1_decoder34858_byte_patch_exact_cuda_20260514` active on Modal
- `lane_z3_balle_hyperprior_bolton_campaign_20260514` local pending research build

## Classification

- score_claim: `false`
- promotion_eligible: `false`
- failure class: stale provider call / pending recovery timeout
- reactivation criteria: rerun HDM8 selector sweep with smaller mode batches,
  bounded provider timeout, and checkpointed per-mode artifact streaming.
