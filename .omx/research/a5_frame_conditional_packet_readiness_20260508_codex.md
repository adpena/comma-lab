# A5 Frame-Conditional Packet Readiness Planner

Date: 2026-05-08

Scope: local-only A5 packet-readiness planning. No remote dispatch, no GPU eval,
no score promotion.

## Artifact

Command:

```bash
.venv/bin/python tools/build_pr101_frame_conditional_packet_readiness.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --json-out experiments/results/pr101_frame_conditional_packet_readiness_20260508_codex/readiness.json
```

Generated artifact:

- `experiments/results/pr101_frame_conditional_packet_readiness_20260508_codex/readiness.json`
- Bytes: `7596`
- SHA-256: `faaeb16ab5fce09bbd48a6bfafe2397e983ed0cd2a13316b32a1365eb9c7f60a`
- Canonical payload SHA-256 without tool-run wrapper:
  `786a13c9bba216b3e9f6299590d1f7f8492e932e9bc63499d3ba70ab0ce54944`

The artifact is ignored under `experiments/results/`; this ledger is the tracked
control-plane pointer.

## Status

- Schema: `pr101_frame_conditional_packet_readiness.v1`
- Input A5 manifest:
  `experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json`
- Input A5 manifest SHA-256:
  `acd8aee2879d1f56432eec9bafe1bea81f5f6f0fcb3160e77d5ce4ea538d785c`
- A5 best byte proxy: `eta=4.0`, `archive_delta_bytes=-4098`
- A5 side-info SHA-256:
  `2685786664c499e41e768259823c91e8b4164cd95845e5765aa658620fee05d4`
- A5 variable-width latent payload SHA-256:
  `7b222b242523d7f86f4e4e420da6467f1a7916d91577bf6fa0912a469a3c0f13`

Evidence semantics:

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`

## Current Blockers

The planner consumes the A5 manifest and fails closed until these exact packet
artifacts are supplied and validated:

1. `candidate_archive_manifest`
2. `packet_local_runtime_patch_manifest`
3. `frame_conditional_runtime_consumption_proof`
4. `per_pair_score_marginal_manifest`
5. `strict_pre_submission_compliance_json`

The planner also keeps exact-eval dispatch closed even after local prerequisites
are valid; a Level-2 dispatch claim and exact CUDA auth eval remain separate
operator-gated steps.

## Focused Tests

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_frame_conditional_bit_budget.py \
  src/tac/tests/test_pr101_frame_conditional_packet_readiness.py
```

Result: `14 passed`.
