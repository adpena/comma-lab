# Feedback: G1 CPU-axis rerank landed

Date: 2026-05-18
Agent: Codex
Lane: `lane_rate_attack_g1_cpu_axis_specific_20260518`

## Durable lesson

G1 is valuable as a zero-cost operator probe, but it must be phrased as
existing-anchor evidence routing, not as a new score claim. The canonical helper
now lives in `tac.frontier_scan` and the operator-facing probe lives at
`tools/probe_g1_cpu_axis_re_rank.py`.

The live probe scanned canonical frontier anchors and found no existing
qualifying Linux x86_64 CPU anchor below PR101/fec6:

- best CPU anchor: `0.1920513168811056`
- best archive: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- result report: `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`
- probe outcome: `.omx/state/probe_outcomes.jsonl`, `probe_id=g1_cpu_axis_re_rank_20260518`
- evidence base: 194 canonical anchors, 55 qualifying CPU anchors, including
  live `accepted_anchor_history` posterior rows
- metadata-bucket method: `metadata_token_match_no_archive_sha_guess`; this avoids the
  substring trap where `pr106_component_prefix16_pr101grammar` was previously
  at risk of being mis-bucketed as PR101

## Design consequence

Keep CPU-axis-specific optimization alive, but only as:

1. Existing-anchor rerank when new paired CPU anchors land.
2. Paired CPU/CUDA xray when a candidate has an unexplained axis gap.
3. A production dispatch criterion for leaderboard-axis exact eval.

Do not use the G1 reranker to create or imply a score claim. The payload carries
`new_score_claim_valid=false` and
`score_claim_kind=existing_anchor_rerank_no_new_score_claim` by construction.
