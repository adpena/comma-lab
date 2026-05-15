# PR106 HDM8 Headerless Format 0x07 PacketIR - 2026-05-15

## Scope

Close the remaining PR106/HDM8 PacketIR wrapper-byte leak after format `0x06`.
This is a pure rate transform: remove the non-score-affecting `0xFE` magic and
format byte, and let the runtime infer the fixed 526-byte sidecar tail when the
member starts with the PR106 inner payload byte `0xFF`.

This is not a score claim until exact CUDA recovery lands.

## Implementation

Added sidecar format:

- format id: `0x07`
- kind: `pr101_ranked_no_op_headerless_implicit_len_fixed_meta_rank_elided`
- runtime contract: `pr106_payload || fixed_526_byte_sidecar`
- parser discriminator: first byte is not `0xFE` and equals `0xFF`
- consumed score-affecting sections: `pr106_payload`, `sidecar_payload`

Changed files:

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
- `src/tac/packet_compiler/pr106_runtime_consumption.py`
- `src/tac/packet_compiler/__init__.py`
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py`
- `tools/profile_pr106_latent_sidecar_recode.py`
- `src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py`
- `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`

## Candidate

Source format `0x06` archive:

- path: `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/emitted_candidates/pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.archive.zip`
- bytes: `186382`
- sha256: `44eb228ab2ae58f267d91c53db9119db3beb6a6c8488c7c903d9bb78e8798844`
- exact CUDA score already known: `0.2063530088582316`

Format `0x07` archive:

- path: `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/emitted_candidates/pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07.archive.zip`
- bytes: `186380`
- sha256: `d09550244c8f12aca9bb958f9dfddeca6ec559e8c1b726fc639ca61c49198258`
- archive delta vs `0x06`: `-2`
- rate delta vs `0x06`: `-0.0000013317179062443428`

Profile evidence:

- JSON: `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/recode_profile.json`
- Markdown: `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/recode_profile.md`
- row fields include archive bytes, archive delta, and archive-rate delta.

## Runtime Proofs

Runtime consumption proof:

- path: `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/runtime_consumption_0x07.json`
- blockers: `[]`
- `runtime_sidecar_decode_consumption_claim`: `true`
- `runtime_sidecar_apply_consumption_claim`: `true`
- `runtime_all_score_affecting_sections_consumed`: `true`
- runtime source tree sha256: `a229d77c0c1101c8113bce0994c80bf3870285b86f6c12442870dd492de52fc7`
- inflate.py sha256: `323b3f8893f90a9c4de7e09518a933d18bb19283ded1142e9b78960047d78c4f`

Same-runtime full-frame CPU parity against `0x06`:

- path: `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/full_frame_parity_cpu_0x07.json`
- `full_frame_inflate_output_parity_claim`: `true`
- `streaming_output_sha256_equal`: `true`
- streaming raw sha256: `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- total frames: `1200`
- total raw bytes per side: `3662409600`

## Exact CUDA Dispatch

Dispatched exact `[contest-CUDA]` Modal T4 eval:

- lane_id: `lane_pr106_hdm8_headerless_fmt07_20260515`
- instance/job id: `modal_pr106_hdm8_fmt07_t4_20260515T025608Z`
- call_id: `fc-01KRMS110S6SNA7KSG8BFD3RPY`
- output_dir: `experiments/results/modal_auth_eval/pr106_hdm8_fmt07_t4_20260515T025608Z`
- uploaded runtime tree sha256: `f1890b6c91215778f6344aeaf089ef64abecfb023c6473f06f304004552d0b08`
- runtime content tree sha256: `f0d0b9cad9333a6c602aec7640e66b619e7d117dfeb11db67b1289052c0e2c4a`
- recovery status: `recovered`
- avg_segnet_dist: `0.0006426`
- avg_posenet_dist: `0.00003236`
- archive bytes: `186380`
- canonical score: `0.20635167714032537`
- delta vs format `0x06`: `-0.0000013317179062443428`
- result review: `.omx/research/pr106_hdm8_headerless_format07_exact_cuda_result_review_20260515_codex.json`
- evidence row: `.omx/research/pr106_hdm8_headerless_format07_exact_cuda_evidence_row_20260515_codex.json`
- exact closure JSON: `.omx/research/pr106_hdm8_headerless_format07_packetir_exact_closure_20260515_codex.json`
- exact closure markdown: `.omx/research/pr106_hdm8_headerless_format07_packetir_exact_closure_20260515_codex.md`

Recovery command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py --output-dir /Users/adpena/Projects/pact/experiments/results/modal_auth_eval/pr106_hdm8_fmt07_t4_20260515T025608Z
```

## Verification

```bash
.venv/bin/ruff check src/tac/packet_compiler/pr106_sidecar_packet.py src/tac/packet_compiler/pr106_runtime_consumption.py src/tac/packet_compiler/__init__.py submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py tools/profile_pr106_latent_sidecar_recode.py src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_pr106_latent_sidecar_recode.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_pr106_latent_sidecar_recode.py -q
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_result_review_packet.py --auth-eval-json experiments/results/modal_auth_eval/pr106_hdm8_fmt07_t4_20260515T025608Z/contest_auth_eval.json --technique pr106_hdm8_headerless_format07_packetir_cuda_review --lane-id lane_pr106_hdm8_headerless_fmt07_20260515 --job-id modal_pr106_hdm8_fmt07_t4_20260515T025608Z --baseline-score 0.2063530088582316 --reactivation-criteria 'Use format 0x07 as the canonical PR106/HDM8 sidecar wrapper only after exact CUDA closure and compliance review; do not spend further on wrapper-byte shaving unless paired with component movement.' --evidence-row-output .omx/research/pr106_hdm8_headerless_format07_exact_cuda_evidence_row_20260515_codex.json --output .omx/research/pr106_hdm8_headerless_format07_exact_cuda_result_review_20260515_codex.json
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_pr106_r2_packetir_exact_closure.py --lane-id lane_pr106_hdm8_headerless_fmt07_20260515 --candidate-result experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/emitted_candidates/pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07.manifest.json --candidate-archive experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/emitted_candidates/pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07.archive.zip --cuda-eval experiments/results/modal_auth_eval/pr106_hdm8_fmt07_t4_20260515T025608Z/contest_auth_eval.json --source-cuda-eval experiments/results/modal_auth_eval/pr106_hdm8_fmt06_t4_20260515T000000Z/contest_auth_eval.json --current-best-cuda-eval experiments/results/modal_auth_eval/pr106_hdm8_fmt06_t4_20260515T000000Z/contest_auth_eval.json --recode-profile experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/recode_profile.json --runtime-consumption-proof experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/runtime_consumption_0x07.json --full-frame-parity-proof experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/full_frame_parity_cpu_0x07.json --no-cpu-eval --output-json .omx/research/pr106_hdm8_headerless_format07_packetir_exact_closure_20260515_codex.json --output-md .omx/research/pr106_hdm8_headerless_format07_packetir_exact_closure_20260515_codex.md
```

Result:

- `ruff`: pass
- pytest: `58 passed in 1.55s`
- result review: `exact_cuda_result_reviewed score=0.20635167714032537`
- exact closure: `classification=exact_measured_improves_packetir_source_cuda`

## Classification

`score_claim=true` for the exact `[contest-CUDA]` result only.

This is legitimate byte-closed PacketIR/rate work, but the effect size is tiny:
two archive bytes, or about `0.0000013317` score if exact CUDA components match
format `0x06`. Exact CUDA confirmed the predicted rate-only movement:
`0.2063530088582316 -> 0.20635167714032537`.

This should be closed for parser/runtime correctness and exact custody, but it
is not a local-minimum escape. The selector/film-grain frontier needs
CUDA-conditioned component movement, not more wrapper-byte shaving.
