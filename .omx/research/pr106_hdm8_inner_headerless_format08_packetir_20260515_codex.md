# PR106 HDM8/HLM2 PacketIR format 0x08 exact closure - 2026-05-15

## Summary

Implemented and exact-evaluated a PR106/HDM8-specific PacketIR wrapper:
`PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED`
(`0x08`). The format elides the 4-byte PR106 inner header for fixed
HDM8/HLM2 payloads and reconstructs it in the submission runtime before
calling the normal decoder.

This is a legitimate pure-rate improvement over format `0x07`, not a score
frontier break:

- Source format `0x07`: 186,380 bytes, exact [contest-CUDA] score
  `0.20635167714032537`.
- Candidate format `0x08`: 186,376 bytes, exact [contest-CUDA] score
  `0.2063490137045129`.
- Delta: `-4` bytes, score delta `-0.0000026634358124832946`.
- SegNet/PoseNet unchanged: `avg_segnet_dist=0.0006426`,
  `avg_posenet_dist=0.00003236`.

## Code landing

- Commit: `fc168ffad` (`Add PR106 HDM8 inner-headerless PacketIR format`)
- Pushed: `origin/main`
- Runtime touched: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py`
- Core module: `src/tac/packet_compiler/pr106_sidecar_packet.py`
- Closure hardening: `src/tac/packetir_exact_closure.py`

## Candidate artifacts

- Candidate archive:
  `experiments/results/pr106_hdm8_inner_headerless_fmt08_20260515_codex/emitted_candidates/pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x08.archive.zip`
- Archive SHA-256:
  `20b91d7283dbb63f3846a019dba4909adc8849e73c8d5669345de3be0a8c36f9`
- Archive bytes: `186376`
- Source archive SHA-256:
  `d09550244c8f12aca9bb958f9dfddeca6ec559e8c1b726fc639ca61c49198258`
- Runtime content tree SHA-256:
  `fe4f56a54adddfc1b52d23014c3d4e78c3133015814197392a38ab794a2b323b`
- Inflated output aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`

## Proofs

- Runtime sidecar decode/apply proof:
  `experiments/results/pr106_hdm8_inner_headerless_fmt08_20260515_codex/runtime_consumption_0x08.json`
- Same-runtime full-frame parity proof:
  `experiments/results/pr106_hdm8_inner_headerless_fmt08_20260515_codex/full_frame_parity_cpu_0x08.json`
- Exact [contest-CUDA] eval:
  `experiments/results/modal_auth_eval/pr106_hdm8_fmt08_t4_20260515T033731Z/contest_auth_eval.json`
- Result review:
  `.omx/research/pr106_hdm8_inner_headerless_format08_exact_cuda_result_review_20260515_codex.json`
- Exact closure:
  `.omx/research/pr106_hdm8_inner_headerless_format08_packetir_exact_closure_20260515_codex.json`
  and `.md`

## Verification commands

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  src/tac/packetir_exact_closure.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packetir_exact_closure.py

PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_packetir_exact_closure.py -q
```

Observed: `75 passed`; ruff clean.

## Interpretation

This closes the HDM8/HLM2 header-elision opportunity. It confirms the remaining
wrapper-only PacketIR surface is very small: the current measured gain is four
bytes, and the next known wrapper-only candidates are lower EV than component
movement, scorer-aware correction, or substrate shift work.

