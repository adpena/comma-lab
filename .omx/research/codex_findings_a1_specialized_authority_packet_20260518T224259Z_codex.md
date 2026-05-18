# Codex Findings: A1-Specialized Authority Packet

Timestamp: 2026-05-18T22:42:59Z
Actor: codex
Task: codex_routing_directive_a1_specialized_deterministic_packet_compiler_20260518::PHASE_0_FEASIBILITY

## Finding

A1-SPECIALIZED can establish authority only as an evidence chain, not as a
design-claim shortcut. The prototype packet compiler must emit a charged byte
artifact, a provenance-bearing report, and hard fail-closed labels until exact
CUDA auth eval validates a self-contained contest archive.

Implemented authority chain:

1. Reusable compiler surface: `src/tac/contest_exploits/a1_specialized_inverter.py`
2. Operator CLI: `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py`
   wraps its blob through `tac.packet_compiler.deterministic_compiler.compile_packet`
   in identity mode by default.
3. Tests: `src/tac/contest_exploits/tests/test_a1_specialized_inverter.py`
4. Report fields:
   - `score_claim=false`
   - `score_claim_valid=false`
   - `promotion_eligible=false`
   - `ready_for_exact_eval_dispatch=false`
   - `dispatch_attempted=false`
   - `rank_or_kill_eligible=false`
   - `full_frame_inflate_output_parity_missing=true`
   - `receiver_classification=RECLAIMABLE_VIA_PACKET_COMPILER`
   - `legal_status=SANCTIONED_FEASIBILITY`
   - `research_only=true`
   - `unmeasured_accuracy=true`
   - `byte_rate_cost_if_shipped`
   - `net_score_delta_claim=null`
   - `provenance` from `tac.provenance.build_provenance_for_predicted`
   - canonical deterministic-packet compiler manifest path/evidence grade once
     the byte-custody wrapper is emitted.

## Guarded Interpretation

This landing does not prove A1 score movement. It proves a deterministic,
small-packet build path, routes the produced bytes through the canonical packet
compiler, and makes the non-authority explicit. The artifact
becomes score-authoritative only after all of the following are true:

1. The blob is integrated into a self-contained contest archive.
2. Every packet byte is consumed by `inflate.sh archive_dir output_dir file_list`.
3. The runtime/archive packet has custody hashes.
4. Exact CUDA auth eval produces component distances and score.
5. The measured distortion benefit exceeds `25 * compressed_blob_bytes / 37_545_489`.

## Assumption Corrections Absorbed

Sidecar assumption review flagged the key A1 pitfall: predicted feasibility rows
must not imply measured accuracy or net score improvement. The report therefore
forces `unmeasured_accuracy=true` and `net_score_delta_claim=null` while still
recording charged byte-rate cost.

## Verification

```bash
.venv/bin/python -m pytest src/tac/contest_exploits/tests/test_a1_specialized_inverter.py
.venv/bin/python -m pytest src/tac/tests/test_deterministic_compiler.py::test_identity_mode_writes_manifest_and_no_op_proof
.venv/bin/python -m ruff check src/tac/contest_exploits/a1_specialized_inverter.py src/tac/contest_exploits/__init__.py src/tac/contest_exploits/tests/test_a1_specialized_inverter.py tools/build_a1_per_pattern_vq_vae_inverter_prototype.py
```

Results:

- 8 passed
- Ruff: all checks passed
