# PR106 HDM8 PacketIR Format 0x06 - Implicit Length

Date: 2026-05-15
Agent: codex:gpt-5.5
Score claim: true, axis `[contest-CUDA]`

## Objective

Continue real score-lowering PacketIR work from the HDM8 fixed-meta `0x05`
reference by removing the explicit four-byte `pr106_len_le_u32` field. The
sidecar tail is fixed at 526 bytes for this grammar, so the runtime can split:

```text
format 0x06 payload = magic(1) | format_id(1) | pr106_payload | fixed_526_byte_sidecar
```

This is a pure rate transform. It must preserve full-frame runtime output and
requires exact contest CUDA eval before any score claim.

## Implementation

- Added `PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED = 0x06`
  to the PR106 PacketIR compiler.
- Exported the format and fixed sidecar-tail byte constant from
  `tac.packet_compiler`.
- Updated `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py` to
  decode `0x06` by deriving the inner PR106 payload length from the fixed
  526-byte sidecar tail.
- Updated the local runtime-consumption proof helper to recognize `0x06`.
- Updated payload profiling to account for the missing length field.

## Candidate

Source archive:

- path: `experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip`
- bytes: `186395`
- sha256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- format: `0x02`

Existing exact-CUDA reference:

- path: `experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/archive.zip`
- bytes: `186386`
- sha256: `9a8e7c4e09572586bac0c1ae425400267ae65b70c918a1c9d13a4ddb08f05adc`
- format: `0x05`
- exact CUDA score: `0.20635567229404411`

New candidate:

- path: `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/emitted_candidates/pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.archive.zip`
- bytes: `186382`
- sha256: `44eb228ab2ae58f267d91c53db9119db3beb6a6c8488c7c903d9bb78e8798844`
- format: `0x06`
- bytes vs source `0x02`: `-13`
- bytes vs reference `0x05`: `-4`
- projected rate-only delta vs `0x05`: `-0.0000026634358125`
- projected score if exact CUDA components match `0x05`: `0.20635300885823163`

## Proofs

Runtime decode/apply proof:

- path: `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/runtime_consumption_0x06.json`
- blockers: `[]`
- runtime decode claim: `true`
- runtime apply claim: `true`
- runtime all score-affecting sections consumed: `true`
- score claim: `false`

Full same-runtime CPU frame parity:

- path: `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/full_frame_parity_cpu_0x06.json`
- source format: `0x02`
- candidate format: `0x06`
- n_pairs_hashed: `600`
- total_frames: `1200`
- streaming_raw_sha256: `b272a1a4841f8fcc9fe843e0544ea0bb46b8359fe5f8cc9d81acf8bd3b7baf99`
- streaming output SHA equal: `true`
- full frame parity claim: `true`
- contest axis claim: `false`
- score claim: `false`

Linked profile and manifest:

- `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/recode_profile.json`
- `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/emitted_candidates/pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.manifest.json`

After proof linking, the only candidate blockers are:

- `exact_cuda_auth_eval_missing`
- `contest_auth_eval_adjudication_missing`

## Exact CUDA Dispatch

Dispatched detached Modal T4 exact auth eval and recovered successfully:

- lane_id: `lane_pr106_hdm8_implicit_len_fmt06_20260515`
- instance/job id: `modal_pr106_hdm8_fmt06_t4_20260515T000000Z`
- Modal call id: `fc-01KRMJFQ38NNFV83YBS4W94672`
- output dir: `experiments/results/modal_auth_eval/pr106_hdm8_fmt06_t4_20260515T000000Z`
- expected uploaded runtime tree sha256:
  `518eebba504811cf53c3a82e70eb8f9ca5361781fa6608132624766134d48b7b`
- recovery status: `recovered`
- result JSON:
  `experiments/results/modal_auth_eval/pr106_hdm8_fmt06_t4_20260515T000000Z/modal_cuda_auth_eval_result.json`
- archive bytes: `186382`
- archive sha256:
  `44eb228ab2ae58f267d91c53db9119db3beb6a6c8488c7c903d9bb78e8798844`
- avg_segnet_dist: `0.0006426`
- avg_posenet_dist: `0.00003236`
- score recomputed from components: `0.2063530088582316`
- evidence grade: `contest-CUDA`
- lane claim terminal status: `completed_contest_cuda_modal_auth_eval_recovered`
- result review packet:
  `.omx/research/pr106_hdm8_implicit_len_format06_exact_cuda_result_review_20260515_codex.json`
- evidence row:
  `.omx/research/pr106_hdm8_implicit_len_format06_exact_cuda_evidence_row_20260515_codex.json`
- autopilot evidence append:
  `reports/cathedral_autopilot_evidence.jsonl`

Harvest command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/pr106_hdm8_fmt06_t4_20260515T000000Z
```

Recovery result:

```json
{
  "status": "recovered",
  "passed": true,
  "score_axis": "contest_cuda",
  "evidence_grade": "contest-CUDA",
  "score_recomputed_from_components": 0.2063530088582316
}
```

## Verification

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  experiments/profile_hnerv_frontier_payloads.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_profile_hnerv_frontier_payloads.py
```

Result: `All checks passed!`

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_profile_hnerv_frontier_payloads.py -q
```

Result: `57 passed in 1.54s`

## Interpretation

This does not solve the `0.192` CPU-only vs CUDA-transfer problem and does not
exhaust film grain or selector search. It is a small but confirmed
byte-closed compiler win inside the current HDM8 basin: a four-byte exact
archive reduction against the current HDM8 `0x05` exact-CUDA reference, with
runtime consumption, full-frame same-runtime parity, terminal dispatch-claim
closure, and exact `[contest-CUDA]` score recovery.

## PacketIR Exact Closure

The generic PacketIR exact-closure verifier now accepts the real HDM8 format-06
manifest shape:

- `candidate_packet_ir_consumed_byte_proof` is normalized to the canonical
  consumed-byte proof field.
- missing source bytes are inferred from the supplied source `[contest-CUDA]`
  eval artifact.
- runtime proofs may carry non-score placeholders such as `framing_meta: null`
  without forcing them into the score-affecting section set.

Tracked closure artifacts:

- JSON:
  `.omx/research/pr106_hdm8_implicit_len_format06_packetir_exact_closure_20260515_codex.json`
- Markdown:
  `.omx/research/pr106_hdm8_implicit_len_format06_packetir_exact_closure_20260515_codex.md`

Closure result:

```json
{
  "classification": "exact_measured_improves_packetir_source_cuda",
  "blockers": [],
  "byte_delta_vs_packetir_source": -13,
  "delta_vs_source_cuda": -0.00000865616639061928,
  "delta_vs_current_best_cuda": -0.0000026634358125110502,
  "runtime_consumption_valid": true,
  "same_runtime_full_frame_parity_valid": true
}
```

This is still not a new promotion claim. It is a closed evidence packet for an
already measured exact CUDA candidate, and it blocks duplicate dispatch of the
same archive/runtime/axis key.
