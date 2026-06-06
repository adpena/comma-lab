schema: frontier_rate_attack_consolidation_audit.v1
status: PASS
canonical_surface: frontier_final_rate_attack_materializer_stack
formal_name: score_program_compiler_over_frozen_evaluator_quotient
rule: Extend the existing final-rate/byte-shaving/materializer/inverse-steganalysis stack; do not add parallel score-program, action-byte, or action-atlas compiler trees.
registry: 22 adapter(s), 13 executable, 2 planning-only
layers:
  - action_candidates: registered-only; exec_archive=2; planning_only=2; receiver_contracts=4; receiver_proofs=1
  - entropy_grammar: registered-only; exec_archive=7; planning_only=0; receiver_contracts=8; receiver_proofs=6
  - payload_and_residual_basis: registered-only; exec_archive=3; planning_only=0; receiver_contracts=6; receiver_proofs=3
dag: 7 node(s), 7 edge(s)
production_action: blocked; blockers=17
machine_vision_source_code_lineage: 4 signal(s); blockers=0
  - Quantizr/PR55: consumed_by_canonical_stack; artifacts=13; consumers=3; layers=payload_and_residual_basis,action_candidates
  - qrepro/PR90: consumed_by_canonical_stack; artifacts=17; consumers=4; layers=action_candidates,entropy_grammar,payload_and_residual_basis
  - PR95: consumed_by_canonical_stack; artifacts=200; consumers=4; layers=payload_and_residual_basis,action_candidates
  - PR110: consumed_by_canonical_stack; artifacts=84; consumers=4; layers=action_candidates,entropy_grammar
state:
  - frontier_final_rate_attack_queues: 216
  - post_feedback_chain_compiler_queues: 27
  - post_feedback_repair_budget_queues: 15
