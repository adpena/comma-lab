# ddm_tq1 NEXT_IF_RESUMED

Status: `SATURATED_ACCEPTED_PREFIX`. Phase B tested 26 queued moves and accepted 6.

Current held advisory archive:
`/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`
sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`, 357,837 B.

Current held advisory row:
`S=0.7534578126155775`, `d_seg=0.004305419922`, `d_pose=0.000716508925`, bytes `357837`, axis `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.

- Receipt JSON: `.omx/research/ddm_tq1_20260805/tq1c/phase_b_realized_acceptance_receipt.json`
- Phase A receipt JSON: `.omx/research/ddm_tq1_20260805/tq1c/phase_a_receipt.json`
- Candidate menu: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form_tq1c/tq1_phase_a_candidate_menu.jsonl` (1,133 rows)
- Priced prefix: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form_tq1c/tq1_phase_a_candidate_prices.jsonl` (26 rows)
- Realized measurement JSONL: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/tq1_phase_b_realized_measurements.jsonl`
- Accepted-move ledger: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/tq1_phase_b_accepted_move_ledger.jsonl`
- Resume checkpoint root: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/stage_checkpoints`

NEXT_IF_RESUMED:
Regenerate or extend the Phase A price ledger from new-base menu index 27, candidate `snap_r01_c13_L12`, starting from the held final archive sha above. Do not claim full-menu closure until all 1,133 generated rows are priced and realized or explicitly folded.

The next launch must restage/decode tokens from the held final archive, use that archive as `--base-archive`, set `--base-archive-sha256 b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`, and preserve the score axis as advisory unless an exact contest CPU/CUDA eval is actually run.
