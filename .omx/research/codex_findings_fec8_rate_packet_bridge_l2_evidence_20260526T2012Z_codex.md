# FEC8 Rate Packet Bridge L2 Evidence Preservation

UTC: 2026-05-26T20:12Z

## Why This Exists

The lane registry was advanced to L2 using a generated refresh directory under
`.omx/research/frontier_rate_attack_feedback_refresh_*`. That directory is
intentionally ignored by `.gitignore`, so the durable registry evidence must be
mirrored into a tracked memo instead of relying on a local-only JSON bundle.

## Preserved Evidence

Ignored source bundle:

` .omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/ `

Key artifact SHA-256 values:

- `receiver_closed_correction_budget.json`: `14af4565960642d5e5ba8e085fbdc1df50618508fba65d33f2e2eebfc9669d24`
- `rate_budget_preservation_plan.json`: `547b3dd111f8edb1b6e1c68928b5f7cd6f4be5f215502a321285f14bafc2964f`
- `targeted_component_correction_acquisition.json`: `4725b7680de41097c6c1dc56f40825e27a57a6a3f8c681bb4643761de052c9f4`
- `repair_budget_waterfill_queue.json`: `3c2f5237f5e89d1fb9191da02f3988ed129e0bd6bf2e44a34668925d11263e6f`

Receiver-closed rate packet row:

- schema: `frontier_rate_attack_receiver_closed_rate_packet_materialization_signal.v1`
- candidate codec: `fec8_static_second_order_markov_k16`
- parent codec: `fec6_fixed_huffman_k16`
- candidate archive bytes: `178507`
- parent archive bytes: `178517`
- saved bytes at risk: `10`
- archive delta vs parent: `-10`
- selector payload wire delta: `-10`
- receiver closed: `true`
- ready for budget spend: `false`
- candidate archive SHA-256 verified on disk: `true`
- candidate archive bytes verified on disk: `true`
- parent archive SHA-256 verified on disk: `true`
- parent archive bytes verified on disk: `true`
- candidate submission dir verified: `true`
- parent submission dir verified: `true`

Rate-budget preservation:

- rate-only saved bytes total: `170`
- FEC8 packet row saved bytes: `10`
- FEC8 packet row entropy position:
  `at_entropy_coder_integer_codeword_boundary`
- cumulative ledger remains false-authority:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`

Targeted correction acquisition:

- active: `true`
- receiver-closed saved bytes total: `10`
- row count: `5`
- queue-actionable acquisition count: `5`
- top correction family: `segnet_posenet_waterfill_region_repair`
- FEC8 row context carries candidate/parent packet manifests, codec identities,
  entropy position, and `receiver_closed_saved_bytes=10`.

## Authority Boundary

This is L2 integration evidence, not score authority. The 10 bytes can drive
MLX/local targeted repair acquisition and waterfill planning only. Budget spend,
promotion, rank/kill, and any score claim still require component response and
exact auth-axis evaluation.
