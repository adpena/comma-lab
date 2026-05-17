# Master-Gradient Operator Response Plan Landed - 2026-05-17

## Verdict

The raw archive-byte master-gradient campaign is superseded by a packet-valid
operator-response plan. The valid object is an
`(N_valid_mutation_operators, 3)` matrix, not an `(N_archive_bytes, 3)` byte or
bit finite-difference derivative.

## Landed Surface

- Reusable module: `src/tac/master_gradient_operator_plan.py`
- Operator CLI: `tools/build_master_gradient_operator_plan.py`
- Focused tests: `src/tac/tests/test_master_gradient_operator_plan.py`
- Feasibility guard reused: `tac.master_gradient_feasibility`

The planner consumes either a single `tac_frontier_archive_layout_v1` manifest
or a `tac_frontier_archive_layout_batch_v1` report, skips grammar headers and
magic bytes, and emits only grammar-aware mutation-operator rows for
parser-proven logical sections such as decoder, latent, sidecar, and byte-map
sections.

Each row carries:

- `mutation_grain=grammar_aware_operator`
- score-response columns:
  `seg_dist_delta`, `pose_dist_delta`, `rate_bytes_delta`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- packet proofs required before probe readiness:
  `repacked_archive`, `updated_zip_headers`, `updated_zip_crc`,
  `inflate_success_proof`, and `byte_consumption_noop_detector`

## Why This Matters

The May 17 master-gradient WIP docs had two partially conflicting framings:
raw byte finite differences, then autograd/FP4 projection. Both become false
authority if they are described as raw archive-byte derivatives. A contest
archive is a discrete packet with ZIP metadata and entropy-coded inner streams.
The coordinate system that matters for search is therefore the set of valid
codec/grammar mutations that can rebuild a packet and survive inflate, not the
outer byte array.

This landing gives L5, Rule #6, and CPU-frontier search a concrete artifact to
route through without retreading the invalid derivative assumption.

## Operator Command

```bash
python tools/build_master_gradient_operator_plan.py \
  --layout-json reports/frontier_monolithic_archive_layout_20260508.json
```

Rows remain blocked until a concrete mutation builder proves packet closure.
Only then may the operator intentionally pass `--packet-proofs-available`, and
even then the resulting manifest remains a probe plan rather than a score,
promotion, rank, kill, or dispatch authority.

## Next Highest-EV Build Step

Implement one mutation builder for the highest-EV Rule #6/FEC6 logical section:

1. Parse the source archive with `tac.frontier_archive_layout`.
2. Apply exactly one grammar-aware operator to a parser-proven logical section.
3. Rebuild ZIP local and central headers plus CRC.
4. Prove inflate success.
5. Prove changed bytes are consumed by the runtime.
6. Emit one row of `{seg_dist_delta, pose_dist_delta, rate_bytes_delta}` under
   explicit `[contest-CPU]`, `[contest-CUDA]`, or paired custody after exact
   result review.

Until those six steps exist, master-gradient rows are planning artifacts only.
