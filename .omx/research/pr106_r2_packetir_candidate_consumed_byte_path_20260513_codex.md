# PR106/R2 PacketIR Candidate Consumed-Byte Path (2026-05-13)

## Scope

Strengthen the canonical PR106/R2 PacketIR path for latent-sidecar compression
candidates. This landing is local custody/tooling only: no dispatch, no score
claim, no promotion, and no SIREN trainer/provider edits.

## Code Surfaces

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
  - Added `decode_pr106_sidecar_packet_dim_delta(...)`.
  - Added `build_pr106_sidecar_recode_candidate_packet(...)`.
  - Added `emit_pr106_sidecar_recode_candidate_archive(...)`.
  - Added `pr106_sidecar_recode_candidate_manifest(...)`.
  - Extended consumed-byte section rows with explicit `offset_start`,
    `byte_count`, and `offset_end_exclusive` aliases while preserving the
    existing `offset` / `bytes` / `end_offset` contract.
- `tools/profile_pr106_latent_sidecar_recode.py`
  - Candidate rows now carry PacketIR identity and consumed-byte proof status
    when the source is a full PR106 sidecar archive.
  - Added `--emit-runtime-candidates-dir` for deterministic archive plus JSON
    manifest emission for candidates whose runtime decoder already exists.
- `src/tac/tests/test_pr106_latent_sidecar_recode.py`
  - Covers candidate PacketIR consumed-byte proofs and emitted non-promotable
    runtime-candidate archives/manifests.

## Proof Status

Identity proof:

- `tools/prove_pr106_packetir_identity.py` passes on
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`.
- Expected archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`.
- `packet_ir_identity_passed=true`.
- `score_claim=false`.
- `ready_for_exact_eval_dispatch=false`.

Candidate proof:

- The PR106/R2 `0x01` source archive can deterministically emit the existing
  PR101 ranked/no-op `0x02` sidecar candidate.
- The emitted candidate member payload matches
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip` member bytes.
- Candidate manifests include:
  - source and candidate packet payload SHA-256s;
  - semantic dim/delta SHA-256s;
  - source and candidate PacketIR consumed-byte proofs;
  - contiguous section offsets and byte counts;
  - per-section SHA-256s;
  - `runtime_consumption_claim=false`;
  - `score_claim=false`;
  - `promotion_eligible=false`;
  - `ready_for_exact_eval_dispatch=false`.

## Claim Boundary

The new candidate path proves lossless sidecar semantic equivalence and
PacketIR parse->reemit identity for supported sidecar formats. It does not
prove full-frame inflate parity for a newly emitted candidate, and it does not
replace exact auth eval.

Required next proof before score or promotion language:

1. Runtime decode/apply proof for the emitted candidate archive and exact
   runtime surface.
2. Full-frame same-runtime parity or same-runtime auth eval where equivalence
   language is needed.
3. Claimed exact `[contest-CUDA]` auth eval with archive/runtime custody,
   dispatch claim closure, and adjudication.

## Solver Wire-In

- `sensitivity-map contribution`: N/A. This landing changes byte-custody and
  candidate manifest plumbing only; no component-distance empirical anchor.
- `pareto constraint`: Non-binding until exact CUDA. Candidate rows expose byte
  deltas and exact blockers, but `ready_for_exact_eval_dispatch=false`.
- `bit-allocator hook`: N/A. The path consumes already selected sidecar
  corrections; it does not alter per-pair importance allocation.
- `cathedral autopilot dispatch hook`: Not registered for dispatch. The emitted
  manifests are explicitly non-promotable and list exact-eval blockers.
- `continual-learning posterior update`: N/A. No empirical anchor was produced.
- `probe-disambiguator`: Existing recode profile keeps multiple candidate
  grammars side-by-side; unsupported grammars remain parser-only until a runtime
  decoder exists.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py -q
```

Result: `28 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py -q
```

Result: `21 passed`.

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/__init__.py \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Result: `All checks passed!`.

```bash
.venv/bin/python tools/prove_pr106_packetir_identity.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --expected-archive-sha256 c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383
```

Result: exit `0`.

## 2026-05-14 Rank-Elided Format 0x04 Candidate

Implemented the next consumed-byte PacketIR candidate for PR106/R2:
format `0x04` elides the fixed PR101 Huffman length-rank byte and the sidecar
length prefix because the single ZIP member payload delimits the packet.

Artifacts:

- Candidate archive:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/pr106_sidecar_rank_elided_format04_candidate.zip`
- Candidate SHA-256:
  `bf83c2ffc559dd42eec131e283bf106b789c0debfe7c3323e10ce1b5d8aa9a70`
- Source archive: `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- Source bytes: `186,780`
- Candidate bytes: `186,776`
- Byte delta: `-4`

Proofs:

- Runtime consumption:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/runtime_consumption_format04.json`
  has `blockers=[]` and
  `runtime_all_score_affecting_sections_consumed=true`.
- Prefix parity:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/same_runtime_prefix_parity_format04.json`
  has `prefix_parity_claim=true` and
  `full_frame_inflate_output_parity_claim=false`.
- Full-frame same-runtime parity:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/same_runtime_full_frame_parity_format04.json`
  has `full_frame_inflate_output_parity_claim=true`,
  `streaming_output_sha256_equal=true`, and matching full-600-pair raw SHA
  `b272a1a4841f8fcc9fe843e0544ea0bb46b8359fe5f8cc9d81acf8bd3b7baf99`.

Interpretation: this is a tiny rate-only win, not a new architecture. It is
valuable because it proves the PacketIR/compiler path can produce consumed,
score-affecting, full-frame-parity-preserving byte reductions. Exact CUDA eval
is mechanically safe but low EV unless batched with a larger PacketIR/repack
candidate or used as a custody canary.

Focused verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py
# 39 passed
```

Combined guard verification after proxy-to-CUDA hardening:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_postdecode_selector_waterfill.py \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_frame_exploit_cuda_transfer_audit.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py
# 60 passed
```

## 2026-05-14 Exact CUDA Dispatch: Rank-Elided Format 0x04

Dispatched the rank-elided `0x04` candidate to Modal T4 exact CUDA as a
byte-closed custody canary. This is not a new architecture and the expected
movement is only the archive-rate delta from `-4` bytes, but it tests the full
PacketIR compiler path under the official CUDA scorer.

Dispatch custody:

- Lane: `pr106_r2_rank_elided_format04_exact_cuda_20260514`
- Job: `pr106_r2_rank_elided_format04_modal_t4_20260514T165500Z`
- Modal call id: `fc-01KRKK6PSAVADS04VAH6F6VTBY`
- Output directory:
  `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z`
- Archive:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/pr106_sidecar_rank_elided_format04_candidate.zip`
- Archive bytes: `186,776`
- Archive SHA-256:
  `bf83c2ffc559dd42eec131e283bf106b789c0debfe7c3323e10ce1b5d8aa9a70`
- Runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh`
- Expected uploaded runtime tree SHA-256:
  `bbe489ac9f82dcd1a24246c7931cf173748d97fd6011d253a201871fced53b67`
- Axis: `[contest-CUDA]`
- Initial recovery:
  `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/modal_auth_eval_recover_summary.json`
  recorded `status=pending` at `2026-05-14T15:55:55Z`.

Dispatch command:

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/pr106_sidecar_rank_elided_format04_candidate.zip \
  --output-dir experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh inflate.sh \
  --gpu T4 \
  --expected-runtime-tree-sha256 bbe489ac9f82dcd1a24246c7931cf173748d97fd6011d253a201871fced53b67 \
  --detach --provider-detach-ack \
  --lane-id pr106_r2_rank_elided_format04_exact_cuda_20260514 \
  --instance-job-id pr106_r2_rank_elided_format04_modal_t4_20260514T165500Z \
  --claim-agent codex:gpt-5.5
```

Next command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z
```

## 2026-05-14 Exact CUDA Harvest: Rank-Elided Format 0x04

Recovered successfully at `2026-05-14T15:57:18Z`.

- Result JSON:
  `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/modal_cuda_auth_eval_result.json`
- Auth-eval JSON:
  `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/contest_auth_eval.json`
- Inflated output manifest:
  `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/inflated_outputs_manifest.json`
- Status: `passed=true`
- Evidence grade: `[contest-CUDA]`
- Score: `0.20661535728576175`
- SegNet distance: `0.0006426`
- PoseNet distance: `3.236e-05`
- Archive bytes: `186,776`
- Inflated aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`

Apples-to-apples baseline:

- Source PR106/R2 PR101 grammar `[contest-CUDA T4]`:
  `0.2066181354574151`, archive bytes `186,780`, SHA-256
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
  (recorded in
  `.omx/research/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex.md`).

Interpretation:

- Delta vs matching source baseline: `-0.0000027781716533392675`.
- Rate-only expectation from `-4` bytes:
  `-25 * 4 / 37,545,489 = -0.0000026634358124886856`.
- The measured delta is within `1.15e-7` of the byte-only expectation and the
  component distances match the source baseline. This confirms format `0x04` is
  a real consumed-byte PacketIR win, not a proxy-only or no-op artifact.
- It is not a frontier move. It is a validated compiler/byte-path canary for
  larger PR106/R2 PacketIR transforms.

## 2026-05-14 Runtime Hardening: Exact Format 0x04 Payload Closure

Adversarial review found a real silent-failure class in the format `0x04`
runtime grammar: a malformed future packet could append surplus sidecar payload
bytes that the PR101 Huffman decoder would ignore after reading the expected
600 symbols. That did not invalidate the harvested exact CUDA result above,
because the candidate archive has full same-runtime parity and the exact CUDA
delta matches the expected `-4` byte rate term. It did mean the runtime and
PacketIR helper needed fail-closed exact payload-length checks before future
byte compiler work builds on this grammar.

Fix landed:

- `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py` now validates
  format `0x04` as:
  `dim_bytes + ceil((600 - noop_count) * huff_min_len / 8) + noop_rank_bytes`.
- `src/tac/packet_compiler/pr106_sidecar_packet.py` enforces the same exact
  length in parser, re-expander, decoder, and candidate builder paths.
- `tools/build_pr106_sidecar_rank_elided_candidate.py` uses the same closure
  rule, preventing malformed no-op byte wins from being emitted by the builder.
- `src/tac/packet_compiler/pr106_runtime_consumption.py` now records
  `reviewed_archive_sha256` and clarifies that `source_archive_sha256` means
  the archive under review, not the source baseline.

Hardened runtime custody:

- Hardened runtime source tree SHA-256:
  `11d93d9e4cbff3dd5ef10bc0a6daa736771dc44431b4c006845667c79a7e0498`
- Hardened `inflate.py` SHA-256:
  `3896775b0328f2b2e15004941028a4f53b3aba6d217caad37fc70f5b97954d72`
- Runtime consumption proof refreshed:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/runtime_consumption_format04.json`
- Full-frame same-runtime parity refreshed with hardened runtime:
  `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/same_runtime_full_frame_parity_format04_hardened_runtime.json`
- Hardened same-runtime full-frame raw SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- Source format `0x02` and candidate format `0x04` both render `1200` frames,
  `3,662,409,600` bytes, with `streaming_output_sha256_equal=true`.

Important evidence-axis caveat:

- The harvested `[contest-CUDA]` score above was run against uploaded runtime
  tree SHA-256
  `bbe489ac9f82dcd1a24246c7931cf173748d97fd6011d253a201871fced53b67`.
- The hardening patch changes the runtime tree. Promotion with the hardened
  runtime would require a fresh exact auth eval using runtime tree
  `11d93d9e4cbff3dd5ef10bc0a6daa736771dc44431b4c006845667c79a7e0498`.
- Because this is a non-frontier `-4` byte canary, the right operational use is
  to carry the hardening forward into the next larger PacketIR/repack candidate,
  not to spend more GPU on this isolated canary.

Verification after hardening:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py
# 41 passed

.venv/bin/python -m pytest -q \
  src/tac/tests/test_postdecode_selector_waterfill.py \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_frame_exploit_cuda_transfer_audit.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py
# 62 passed

.venv/bin/ruff check \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  tools/build_pr106_sidecar_rank_elided_candidate.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py
# All checks passed
```
