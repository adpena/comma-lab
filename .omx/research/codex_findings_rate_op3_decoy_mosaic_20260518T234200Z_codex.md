# Codex Findings: RATE-OP-3 Decoy/Mosaic Residual-Basis Probe

timestamp_utc: 2026-05-18T23:42:00Z
agent: codex
task_id: rate_attack_op_3_decoy_mosaic_residual_basis
source_design_memo: .omx/research/rate_attack_novel_vectors_design_memo_20260518.md
canonical_consumer: tools/cathedral_autopilot_autonomous_loop.py
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

RATE-OP-3 now has a bounded, reusable route-entropy planning artifact. The
current fec6 packet does not yet provide score authority for decoy/mosaic
routing: the 50-pair cheap probe produces a near-uniform 4-route split
(`route_entropy_bits_per_pair=1.9988455359952018`, route counts 13/12/13/12),
but the route labels are not bound to charged archive bytes, no specialist-head
payload is measured, and no single-monolith control exists.

The correct OP3 state is therefore:

- route-table math: promising enough to keep the lane alive
- current-packet authority: prediction-only
- score delta claim: none
- next real proof: export-first route-table grammar plus monolith control and
  specialist residual entropy

## Artifact

Generated artifact:

`experiments/results/rate_attack_op3_decoy_mosaic_probe_20260518T233642Z/route_entropy_report.json`

Content hashes:

- `route_entropy_report.json`: `f91e70279574a3028d0c23ae9cf360913e3d70d8273b5860daaf87ce92279075`
- `cathedral_autopilot_candidates.jsonl`: `7ac82301afec64d9e64dadb7bf75187f33680961a98fe15a4d785ff4cf35f79a`
- `route_table_candidates.jsonl`: `549acc7184c6c1825c4289b547b020946c5c4c6b4ed8f91b2ebebb408a70eb6c`
- `mosaic_routes.csv`: `401e1b84b254e12ce1e2b12d2f58e6b2904c99902e9507a0c753c18515217cfc`

Input archive:

- path: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- axis tag: `[contest-CPU GHA Linux x86_64]`
- sha256 prefix: `6bae0201fb08`

Feature sidecar:

- path: `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`
- evidence: advisory per-pair routing features only
- cheap probe: first 50 pairs, deterministic truncation

## Adversarial Findings

1. Route table bytes are cheap but not sufficient.

The canonical 2-bit, 600-pair route table is only 278 bytes under the current
model (`150` raw route bytes + `64` header + `64` route metadata). That is
structurally small, but OP3's actual cost is route table plus specialist heads
plus runtime/pack overhead. The helper therefore refuses a negative
`predicted_score_delta` unless charged labels, monolith control, measured
specialist-head bytes, and overhead materiality are all present.

2. Fec6 does not currently expose a charged OP3 route-label payload.

The current archive inspection found no route/mosaic/specialist/head/residual
section. Any route labels derived from `.omx`, CSV, JSON, or analysis sidecars
remain research-only until embedded inside `archive.zip`/`0.bin` and proven
consumed by inflate.

3. Decoy statistics remain uncalibrated.

The OP3 source memo warned that expected-statistics decoys can become a vague
baseline. This implementation treats decoy routing as a probe target, not as a
score-bearing result. Residual entropy, monolith-control residual entropy, and
specialist-head bytes are required before any mosaic-vs-monolith method verdict.

4. Cathedral integration is intentionally rank-only.

`cathedral_autopilot_candidates.jsonl` loads through the real
`load_candidates_from_jsonl` path with all rows carrying:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `predicted_score_delta=0.0`

That makes the signal visible to the canonical consumer without polluting
exact-ready queues or promotion surfaces.

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A - OP3 emits route-entropy planning only;
   no component sensitivity tensor is measured.
2. Pareto constraint: active as Dykstra fields in `route_entropy_report.json`
   (`C_rate`, `C_seg`, `C_pose`, `C_inflate`, `C_runtime`, `C_custody`,
   `C_consumer`).
3. Bit-allocator hook: active through `mosaic_routes.csv` and route table byte
   accounting; no mutation authority yet.
4. Cathedral autopilot dispatch hook: active through
   `cathedral_autopilot_candidates.jsonl`, with false authority flags.
5. Continual-learning posterior: active via this findings memo and canonical
   task-status completion row.
6. Probe-disambiguator: active; OP3 now separates cheap route entropy from
   charged route-label authority and monolith-control authority.

## Required Next Proofs

- Build an export-first `0.bin` route-label grammar and prove route bytes are
  charged and consumed.
- Add a single-monolith control on the same pair set, archive family, and byte
  budget.
- Measure specialist-head bytes and residual entropy after final compression.
- Emit byte-different archive, runtime consumption proof, full-frame parity,
  and exact CUDA auth eval before any score claim.
