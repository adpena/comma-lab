# CodecOp Entropy-Floor Ledger Adapter - Worker G - 2026-05-07

## Scope

This ledger records the Worker G hardening pass for converting PR101 CodecOp entropy-floor reports into meta-Lagrangian planning atoms. It is not a score claim, dispatch claim, promotion record, or candidate archive manifest.

## Inputs

| path | sha256 | atoms | negative deltas | schemas |
| --- | --- | ---: | ---: | --- |
| reports/pr101_provable_optimal_floor.json | 42e4f458157d26d8... | 5 | 4 | pr101_compression_floor_ladder.v2 |
| reports/pr101_context_transform_floor_probe.json | 7053fdb71aec6e69... | 21 | 11 | pr101_context_transform_floor_probe.v1 |

## Output

- JSONL atom rows: `experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/codec_op_entropy_floor_atoms.jsonl`
- Atom count: `26`
- Negative byte-delta atom count: `15`

## Policy

- planning-only rows: `26`
- proxy rows: `26`
- score-claim rows: `0`
- score-affecting payload changed rows: `0`
- charged bits changed rows: `0`
- dispatchable rows: `0`
- exact-eval-ready rows: `0`
- promotion-eligible rows: `0`
- fail-closed policy: `True`

No training, eval, or remote-GPU job was dispatched by this adapter run; no lane-dispatch claim is required for this planning-only ledger emission.

## Best Negative Byte Deltas

| atom | label | byte_delta | target_archive_bytes | policy |
| --- | --- | ---: | ---: | --- |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:delta_mod255:markov2` | delta_mod255:markov2 | -107115 | 71029 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:identity:markov2` | identity:markov2 | -64037 | 114107 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:signed_zigzag:markov2` | signed_zigzag:markov2 | -64037 | 114107 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_provable_optimal_floor:markov2_per_tensor` | markov2_per_tensor | -64037 | 114107 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:zero_mask_nonzero_value:markov2` | zero_mask_nonzero_value:markov2 | -62438 | 115706 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:abs_sign_split:markov2` | abs_sign_split:markov2 | -27988 | 150156 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:identity:markov1` | identity:markov1 | -9944 | 168200 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:signed_zigzag:markov1` | signed_zigzag:markov1 | -9944 | 168200 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_provable_optimal_floor:markov1_per_tensor` | markov1_per_tensor | -9944 | 168200 | planning_only/proxy/non_dispatchable/non_promotable |
| `codec_op_entropy_floor:pr101_context_transform_floor_probe:zero_mask_nonzero_value:markov1` | zero_mask_nonzero_value:markov1 | -9830 | 168314 | planning_only/proxy/non_dispatchable/non_promotable |

## Blockers

- `codec_op_entropy_floor_atom_is_planning_only`
- `entropy_floor_is_model_class_lower_bound_not_candidate_archive`
- `requires_joint_admm_codecop_materialization`
- `requires_byte_closed_archive_manifest_before_dispatch`
- `requires_noop_or_roundtrip_decode_proof`
- `requires_exact_cuda_auth_eval`
- `not_promotion_eligible`

## Row Custody Check

- rows inspected: `26`
- all rows non-promotable: `True`
- all rows non-dispatchable: `True`
- all rows not exact-eval-ready: `True`
