# PR106 HDM9 PacketIR format09 byte-closed recode

Date: 2026-05-15

## Operator question

The `0.192` result is a `[contest-CPU]` near-miss, not a confirmed
`[contest-CUDA]` or leaderboard-class result. The exact CPU artifact is:

- archive: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- bytes: `178517`
- score: `0.1920513168811056` `[contest-CPU]`
- artifact:
  `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/modal_cpu_auth_eval_result.json`

The same archive/runtime on `[contest-CUDA]` scored `0.22621002169349796`, so
it is not a confirmed CUDA frontier result:

- artifact:
  `experiments/results/modal_auth_eval/archive_6bae0201fb08/modal_cuda_auth_eval_result.json`

Conclusion: legitimate CPU-axis evidence, not a CUDA promotion.

## Film grain / selector / waterfill status

The current film-grain and selector family has not been globally exhausted, but
the tested promotion path is not trustworthy enough to keep spending on without
a CUDA-in-loop gate. MPS/proxy positives did not transfer:

- HDM8 format08 exact CUDA baseline:
  `0.2063490137045129`, `186376` bytes,
  `20b91d7283dbb63f3846a019dba4909adc8849e73c8d5669345de3be0a8c36f9`
- HDM8 charged selector exact CUDA:
  `0.2161099173824375`, `187366` bytes,
  `793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798`
- HDM8 sparse top001 exact CUDA:
  `0.2064796628814009`, `186518` bytes,
  `98fd0bd779404970f11ca616b5c98dcb3ec41f74fb0a4ffe6d4ce613684d1223`
- HDM8 sparse budget128 exact CUDA:
  `0.20787717836935493`, `186760` bytes,
  `b0645ee705cf7fe34300f5b4586efe03d5c3262c66ac72a503b90599a996d004`
- HDM8 full aggressive selector exact CUDA:
  `0.2095197967107254`, `187226` bytes,
  `34dc94644f5619ea7e6254079e3e4d3bbf0952f8a0ad287f675f7a249f359071`

Verdict: retire proxy-ranked selector promotion for now. Reactivation requires
CUDA-prefix scorer evidence, fixed-mode positive controls, or a transfer model
that predicts PoseNet regressions before paid exact eval.

## New byte-closed work landed locally

HDM9 recodes the HDM8 decoder scale tail. It keeps HDM8 q-Brotli chunks
unchanged and stores each fp32 scale as low-three bytes plus a 0x3B/0x3C
high-byte bitmask.

- source archive:
  `experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/archive.zip`
- emitted HDM8 format08 archive:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x08.archive.zip`
- emitted HDM9 format09 archive:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip`
- HDM9 bytes: `186352`
- HDM9 SHA-256:
  `09bcd867c2778d38d5ac04b648d44cb3bdfcfd3e3db402beb8886826cced50e9`
- delta vs fixed-meta source: `-34` bytes
- delta vs HDM8 format08: `-24` bytes
- expected pure-rate delta vs HDM8 format08:
  `-0.00001598061487493211`
- expected score if exact CUDA components match format08:
  `0.20633303308963796`

This does not escape the `<0.192` local minimum; it is a small exact compiler
recode and a cleaner launch point for PacketIR/range-coder work.

## Exact CUDA closure

After this ledger was opened, the HDM9 format09 packet was dispatched through
Modal T4 exact CUDA auth eval:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_auth_eval.py \
  --archive experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval/pr106_hdm9_fmt09_t4_20260515T043733Z \
  --expected-runtime-tree-sha256 cfdfb0ab801162207222c43f3701e8232e89f80e739d27bf52f35bee76a81291 \
  --detach --provider-detach-ack \
  --lane-id lane_pr106_hdm9_packetir_format09_20260515 \
  --instance-job-id pr106_hdm9_fmt09_t4_20260515T043733Z
```

Recovered result:

- call id: `fc-01KRMYVF8A08TKG09HSV5Q8R0D`
- output dir:
  `experiments/results/modal_auth_eval/pr106_hdm9_fmt09_t4_20260515T043733Z`
- result:
  `experiments/results/modal_auth_eval/pr106_hdm9_fmt09_t4_20260515T043733Z/modal_cuda_auth_eval_result.json`
- status: `recovered`
- score axis: `[contest-CUDA]`
- score: `0.20633303308963796`
- bytes: `186352`
- archive SHA-256:
  `09bcd867c2778d38d5ac04b648d44cb3bdfcfd3e3db402beb8886826cced50e9`
- avg SegNet dist: `0.0006426`
- avg PoseNet dist: `0.00003236`
- dispatch claim:
  `completed_contest_cuda_modal_auth_eval_recovered`

The score exactly matches the pure rate prediction from format08:
`0.2063490137045129 - 25*24/37545489 = 0.20633303308963796`.

Verdict: confirmed legitimate `[contest-CUDA]` byte-only improvement over
PR106 HDM8 format08, but not a frontier escape.

## Proofs

- profile:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/profile.with_proofs.json`
- runtime consumption:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/runtime_consumption_hdm9.json`
- same-runtime full-frame parity:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm8_vs_hdm9.json`

Runtime consumption result:

- `blockers=[]`
- `runtime_sidecar_decode_consumption_claim=true`
- `runtime_sidecar_apply_consumption_claim=true`
- `runtime_all_score_affecting_sections_consumed=true`
- `score_claim=false`
- `promotion_eligible=false`

Same-runtime parity result:

- source archive: format08, `186376` bytes,
  `20b91d7283dbb63f3846a019dba4909adc8849e73c8d5669345de3be0a8c36f9`
- candidate archive: format09, `186352` bytes,
  `09bcd867c2778d38d5ac04b648d44cb3bdfcfd3e3db402beb8886826cced50e9`
- device axis: `local-cpu-streaming-runtime`
- pairs hashed: `600`
- frames hashed: `1200`
- bytes hashed: `3662409600`
- streaming SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- `full_frame_inflate_output_parity_claim=true`
- `score_claim=false`

## Tests

```bash
.venv/bin/python -m ruff check \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py \
  src/tac/packetir_exact_closure.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_submission_pr101_decoder_adapter.py \
  src/tac/tests/test_pr106_fixed_latent_recode.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  -q --tb=short
```

Observed: `112 passed`.

## Next

Do not spend more on MPS-only selector positives. Next score-moving work should
be one of:

1. PR101 FEC6 byte-shave below `<0.192` on `[contest-CPU]` if at least `77`
   more pure bytes can be removed without component changes.
2. CUDA-in-loop selector calibration with a tiny prefix gate before any full
   exact eval.
3. PacketIR/range-coder continuation where each delta is decoded-output
   preserving and byte-closed before exact eval.
