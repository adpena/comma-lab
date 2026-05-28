# Codex Findings - PSV3 Decoder Quant Exact Handoff

UTC: 2026-05-28T13:01Z
Agent: Codex
Lane: lane_psv3_decoder_quant_a8180e2f85db

## Verdict

PSV3 decoder quantization is now wired from local byte transform into exact-ready handoff instead of stopping at an advisory repack manifest.

The repacker emits:

- byte-closed candidate archive
- deterministic receiver runtime adapter
- family-agnostic runtime consumption proof
- optimizer_candidate_queue.v1 source queue row
- serialized archive delta contract
- materializer submission closure packet
- exact-readiness queue
- dry-run exact-eval dispatch plan

All score authority remains false until contest CPU/CUDA eval signs the result.

## Live Evidence

Source archive:

- path: `experiments/results/pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_20260528T080350Z/archive.zip`
- bytes: 137351
- sha256: `ef5a087ff6301dbff630de4ce65dabd5039c35b6b3902ad5084f67b6029223b2`

Candidate archive:

- path: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/archive.zip`
- bytes: 96030
- sha256: `a8180e2f85db4a03981c61c7548d4961c38d2d69b19c42ac5fd33d03cb35255c`

Realized archive-byte movement:

- archive delta: -41321 bytes
- realized saved bytes: 41321
- source 0.bin bytes: 130210
- candidate 0.bin bytes: 87822
- candidate 0.bin delta: -42388 bytes

Local decoder proof:

- proof pairs: 8
- max abs drift in sigmoid output: 0.005030632019042969
- mean abs drift in sigmoid output: 0.0005091930506750941
- axis: `[macOS-CPU archive-proof]`

Exact-handoff artifacts:

- source queue: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/optimizer_candidate_queue.json`
- runtime proof: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/runtime_consumption_proof.json`
- closure report: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/submission_closure/submission_closure_report.json`
- exact-readiness bridge: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/exact_readiness_bridge_report.json`
- exact-ready queue: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/exact_readiness/pact_nerv_selector_v3_decoder_quant_repack_int8_per_channel_brotli_q11_a8180e2f85db.exact_ready_queue.json`
- dry dispatch plan: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/exact_eval_dispatch_plan.json`
- dry experiment queue: `.omx/research/psv3_decoder_quant_exact_handoff_live_20260528T1301Z/exact_eval_experiment_queue.json`

## Integration Fix

The exact-readiness bridge initially produced an exact-ready queue, but the dry dispatch planner blocked because promoted rows had stripped the family-agnostic runtime-proof metadata required by `exact_dispatch_authority`.

Fixed in `src/tac/optimizer/exact_readiness.py`: promoted rows now preserve `target_kind`, `materializer_id`, `receiver_contract_kind`, and runtime-adapter identity fields.

Regression added in `src/tac/tests/test_optimizer_exact_readiness.py`.

## Verification

- `ruff check` on touched files: passed
- `pytest src/tac/substrates/pact_nerv_selector_v3/tests/test_pact_nerv_selector_v3.py -q`: 17 passed
- `pytest src/tac/tests/test_optimizer_exact_readiness.py -q`: 66 passed
- live repack: passed
- live materializer submission closure: passed
- live exact-readiness bridge: 1 ready, 0 blocked
- live dry dispatch plan: 1 authorized, 0 blocked
- recursive adversarial review bundle `52400d6a77184c05`: 3 clean passes, 0 unresolved critical findings

## Remaining Authority Boundary

This is exact-ready dispatch input, not a score claim. The next authoritative step is lane claim plus contest CPU/CUDA exact eval on the closed archive/runtime packet.
