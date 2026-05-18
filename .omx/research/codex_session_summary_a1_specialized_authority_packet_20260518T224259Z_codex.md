# Codex Session Summary: A1-Specialized Authority Packet

Timestamp: 2026-05-18T22:42:59Z
Actor: codex

## Landed

- Claimed `codex_routing_directive_a1_specialized_deterministic_packet_compiler_20260518::PHASE_0_FEASIBILITY`.
- Added `tac.contest_exploits.a1_specialized_inverter`, a deterministic
  VQ/FP4/sparse/Brotli prototype builder for fixed feature/pattern matrices.
- Added `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py`, a thin
  operator CLI that builds the A1 blob and, by default, wraps it through the
  canonical deterministic packet compiler in identity mode.
- Added tests proving deterministic output, fail-closed authority flags, proxy
  feature labeling, and canonical packet-compiler byte-custody manifest output.
- Persisted findings memos for the A1 authority chain and the broader
  rate-attack assumption/authority review.

## Authority Split

The produced A1 report is deliberately non-authoritative for score:

- `receiver_classification=RECLAIMABLE_VIA_PACKET_COMPILER`
- `legal_status=SANCTIONED_FEASIBILITY`
- `evidence_grade=prediction_only`
- canonical packet compiler evidence: `byte_custody_only`
- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `full_frame_inflate_output_parity_missing=true`
- `unmeasured_accuracy=true`
- `net_score_delta_claim=null`

Score authority remains blocked until a self-contained contest archive consumes
the packet bytes, exact CUDA auth eval lands, and paired Linux CPU evidence is
available for frontier/submission claims.

## Verification

```bash
.venv/bin/python -m pytest src/tac/contest_exploits/tests/test_a1_specialized_inverter.py src/tac/tests/test_deterministic_compiler.py::test_identity_mode_writes_manifest_and_no_op_proof
.venv/bin/python -m ruff check src/tac/contest_exploits/a1_specialized_inverter.py src/tac/contest_exploits/__init__.py src/tac/contest_exploits/tests/test_a1_specialized_inverter.py tools/build_a1_per_pattern_vq_vae_inverter_prototype.py
```

Results:

- 8 passed
- Ruff: all checks passed

## Next

Continue with the highest canonical queue item after push. A natural follow-up
is an A1 per-pattern-vs-generic disambiguator or a tiny fail-closed validator
that prevents stale F1/G1 predicted deltas from entering aggregate
rate-attack composition.
