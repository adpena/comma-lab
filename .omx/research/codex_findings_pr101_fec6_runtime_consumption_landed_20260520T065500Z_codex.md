# Codex Findings - PR101/FEC6 Runtime Consumption Closure

Timestamp UTC: 2026-05-20T06:55:00Z

## Verdict

`operator_packetir_compiler_pr101_fec6_20260519::IDENTITY_AND_QUEUE` is locally closed as a non-promotional PacketIR/compiler artifact. The prior blocker `runtime_byte_consumption_noop_detector_missing` is cleared by a deterministic runtime-consumption proof, then propagated into the PR101/FEC6 candidate queue and frontier matrix.

This is not a score claim, not a promotion claim, and not a dispatch claim.

## Artifacts

| artifact | sha256 | authority |
|---|---|---|
| `.omx/research/pr101_fec6_runtime_consumption_proof_20260520T065500Z_codex.json` | `7d03b58b9a20a3b198cbabbedc42e6323deec89a9fc1e586e63b0d8f9b4b2bf9` | runtime-consumption proof only |
| `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packetir_candidate_queue.json` | `567512b67cc3412149c402c86e665bc100da3ba5b46b8c035d1c269bf60531f2` | candidate queue, non-promotional |
| `.omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.json` | `0e28a195e58916fedc2f51331797053249f5bbfdeadc549eebca6aedb64205ba` | authority matrix, non-promotional |
| `.omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.md` | `34410b89594e9d6d87e7432f519dc333198458479c7a86392ffb9e6b26dacc3c` | rendered operator matrix |

## Proof facts

- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- ZIP member: `x`
- Member payload bytes: `178417`
- Runtime bytes consumed: `178417`
- `no_op_detector_passed=true`
- Mutation probes: `4`
- Blockers: `[]`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The proof imports the frozen `submission_dir/inflate.py`, runs `parse_pr101_frame_selector_archive`, runs `parse_archive` on the source payload, and confirms source/selector bytes are runtime-visible through deterministic mutation probes.

## Matrix closure

`tools/build_pr101_fec6_packetir_candidate_queue.py --runtime-consumption-proof ...` now exits `0` and emits:

- `runtime_consumption_proven=true`
- `blockers=[]`
- `candidate_count=33`
- `operator_candidate_count=29`

`tools/build_pr101_frontier_packetir_matrix.py` now marks:

- `run_compile_packet_identity_closure=done`
- `generate_fec6_packetir_candidate_queue=done`
- `prove_parser_consumption_and_byte_accounting=done`
- `prove_runtime_byte_consumption_noop_detector=done`
- `local_identity_profile_smoke=done`
- `paired_exact_eval_after_candidate_queue=blocked_until_candidate_queue_and_operator_authorization`

Canonical task status was updated from blocked to in-progress to completed because the writer forbids direct `blocked -> completed` transitions. The durable proof lives under `.omx/research/`; the ignored `experiments/results/.../runtime_consumption_proof_20260520_codex.json` copy is not the queue authority.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_pr101_frontier_packetir_matrix.py \
  src/tac/tests/test_pr101_fec6_runtime_consumption.py \
  src/tac/tests/test_pr101_fec6_candidate_queue.py \
  -p no:cacheprovider
```

Result: `30 passed in 0.76s`

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr101_frontier_packetir_matrix.py \
  src/tac/tests/test_pr101_frontier_packetir_matrix.py \
  src/tac/packet_compiler/pr101_fec6_runtime_consumption.py \
  tools/prove_pr101_fec6_runtime_consumption.py \
  src/tac/tests/test_pr101_fec6_runtime_consumption.py
```

Result: `All checks passed!`

```bash
.venv/bin/python tools/canonical_task_status.py --validate
```

Result: `{"rows": 251, "status": "valid"}`

## Follow-up

The VQ diagnostic Modal call `fc-01KS21XSVGM2KJ5ET0ET3YCCFN` was polled at 2026-05-20T06:53Z and was still running. Re-run:

```bash
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS21XSVGM2KJ5ET0ET3YCCFN
```

Do not claim score from that lane; it is a fixed-int16 VQV1 quality diagnostic with `score_claim=false`, `promotion_eligible=false`, and no contest-axis authority.
