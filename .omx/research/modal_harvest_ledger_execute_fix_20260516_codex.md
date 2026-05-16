# Modal Harvest And Ledger Execute Fix - 2026-05-16

## Scope

No new provider work was launched. This pass harvested existing Modal call IDs
still inside the result-cache window, fixed the harvest tool path that could
silently no-op, and reconciled the canonical call-ID ledger so the
`--from-ledger` view stops reporting already-harvested calls as live.

## Harvest Results

Freshly recovered terminal rows:

| Lane | Call ID | Result | Evidence |
| --- | --- | --- | --- |
| `lane_d4_wyner_ziv_frame_0_substrate_20260514` | `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C` | `rc=0`, `elapsed_seconds=6914.784993056`, `score_claim=false` | `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T194805Z__smoke__100ep_modal/harvested_artifacts/_harvest_summary.json` |
| `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515` | `fc-01KRPKCXARWP7NBGJCXB2P9QEP` | `rc=0`, `elapsed_seconds=868.24924665`, `score_claim=false` | `experiments/results/lane_substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch_20260515T195556Z__smoke__100ep_modal/harvested_artifacts/_harvest_summary.json` |
| `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515` | `fc-01KRPMDRH2VNVKFFAYTWD4X0SD` | `rc=1`, guard refused full run because G1 bytes are not consumed by runtime without the explicit research-only direct-residual acknowledgement | `experiments/results/lane_substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch_20260515T201408Z__full__1000ep_modal/harvested_artifacts/_harvest_summary.json` |
| `lane_z4_cooperative_receiver_loss_step2_20260514` | `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93` | `rc=0`, `elapsed_seconds=2930.851619733`, `score_claim=false` | `experiments/results/lane_substrate_z4_cooperative_receiver_loss_modal_t4_dispatch_20260515T194645Z__smoke__100ep_modal/harvested_artifacts/_harvest_summary.json` |

Result-review classification: training artifacts only. None of the four rows
is a promotion result or rank/kill authority because all terminal claims carry
`score_claim=false` and `promotion_eligible=false`.

## Bug Class Fixed

`tools/harvest_modal_calls.py --from-ledger --execute` was documented as an
executable path, but the implementation returned immediately after printing the
ledger view. The result was a false operator surface: the canonical ledger could
show unharvested call IDs even after the metadata harvest path had recovered
terminal artifacts and claims.

Fix landed:

- `--from-ledger --execute` now prints the canonical ledger view, then runs the
  normal harvest flow.
- terminal harvests append a terminal event back to
  `.omx/state/modal_call_id_ledger.jsonl` via the canonical locked helper.
- repeated harvests avoid duplicate terminal call-ID events when the latest
  ledger status is already terminal.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_modal_call_id_ledger.py src/tac/tests/test_modal_training_harvest_summary.py -q`
- `.venv/bin/python -m py_compile tools/harvest_modal_calls.py`
- `.venv/bin/python tools/harvest_modal_calls.py --from-ledger --repo-root .` now reports `unharvested call_ids: 0`.

## Next Actions

1. Classify D4 and Z4 artifacts through an exact result-review packet before
   any dispatch promotion.
2. Do not re-run Z3-G1 full as a score-bearing job until an inflate-time G1
   byte consumer and byte-mutation smoke are landed, or until the run is
   explicitly tagged as a research-only direct-residual control.
3. Keep the call-ID ledger as the first harvest status surface; a nonzero
   unharvested count should be treated as a cache-loss risk until closed.
