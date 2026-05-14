# HNeRV HDM6 Tuned-Brotli Candidate - 2026-05-14

## Verdict

HDM6 is a byte-closed, rate-only HLM2 follow-up candidate. It preserves the
HDM4 fixed decoder order and split recipe, but hardcodes per-chunk Brotli
encoder parameters that save 3 charged archive bytes on the current HLM2 exact
CUDA anchor.

No score claim is made here. Promotion still requires a lane claim and exact
CUDA auth eval on the exact archive/runtime packet.

## Candidate

- Source archive:
  `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip`
- Source archive bytes: `186411`
- Source decoder section codec: `hdm4_q_brotli_split`
- Candidate archive:
  `experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/pr106_r2_hdm4_hlm2_xmember_hdm6_archive_candidate.zip`
- Candidate archive SHA-256:
  `f3941568035d40bc7cb9e6fc0a108a5ec8bedf33f7ae14f6c060e92f7f247593`
- Candidate archive bytes: `186408`
- Decoder section bytes: `169990 -> 169987`
- Rate-only candidate delta, axis label `[pending-contest-CUDA; rate-term-only;
  not a score claim]`, if exact CUDA components are equal:
  `-0.000001997577`

## Mechanism

HDM4 q chunks were:

```text
[130887, 2770, 4398, 31806]
```

An exhaustive local Brotli parameter grid over the same HDM4 order/splits found
HDM6 q chunks:

```text
[130887, 2769, 4397, 31805]
```

The decoder pays no extra metadata for these parameters because Brotli streams
are self-describing at decode time. The only HDM6 metadata is the same 17-byte
fixed runtime header shape as HDM4: magic, recipe id, and four len24 prefixes.

The fixed recipe is:

```text
chunk 0: quality=11, lgwin=18
chunk 1: quality=11, lgwin=16
chunk 2: quality=11, lgwin=16
chunk 3: quality=10, lgwin=16
```

## Custody

Archive build:

```bash
.venv/bin/python tools/build_hnerv_hdm3_archive_candidate.py \
  --source-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --output-dir experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex \
  --source-label PR106-R2-HDM4-HLM2-XMEMBER \
  --decoder-recode-variant hdm6 \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/manifest.json \
  --fail-if-blocked
```

Runtime parity proof:

```bash
.venv/bin/python tools/prove_hnerv_hdm3_runtime_adapter.py \
  --source-archive experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip \
  --candidate-archive experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/pr106_r2_hdm4_hlm2_xmember_hdm6_archive_candidate.zip \
  --output-dir experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/runtime_adapter_proof.json \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar
```

The proof now distinguishes exact payload identity from decoder-raw equivalence:

- `inflate_output_parity_proven_by_payload_identity=false`
- `inflate_output_parity_proven_by_lossless_decoder_equivalence=true`
- `submission_runtime_candidate_parse_claim=true`
- `full_frame_inflate_output_parity_claim=false`

Static compliance:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/exact_eval_static_release_surface \
  --archive experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip \
  --archive-manifest-json experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/exact_eval_static_release_surface/archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 f3941568035d40bc7cb9e6fc0a108a5ec8bedf33f7ae14f6c060e92f7f247593 \
  --expected-archive-size-bytes 186408 \
  --public-scan-path experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/exact_eval_static_release_surface \
  --json-out experiments/results/pr106_r2_hdm4_hlm2_hdm6_candidate_20260514_codex/pre_submission_compliance.static.json \
  --strict
```

Static packet readiness:

- `ready_for_archive_preflight=true`
- `static_packet_ready=true`
- Remaining blockers:
  - `lane_dispatch_claim_missing`
  - `exact_cuda_auth_eval_missing`

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_hnerv_decoder_recode.py \
  src/tac/tests/test_hnerv_hdm3_archive_candidate.py \
  src/tac/tests/test_hnerv_hdm3_runtime_adapter.py \
  src/tac/tests/test_submission_pr101_decoder_adapter.py \
  src/tac/tests/test_hnerv_lowlevel_packer.py
```

Result after adversarial hardening: `76 passed in 31.55s`.

## Dispatch Gate

Next step, if spending is allowed in this cycle:

```bash
tools/claim_lane_dispatch.py claim \
  --lane-id hnerv_hdm6_q_brotli_tuned_exact_eval \
  --platform modal \
  --instance-job-id hnerv_hdm6_hlm2_modal_t4_20260514T<HHMMSS>Z \
  --agent codex \
  --status eval \
  --notes "HDM6 tuned Brotli exact CUDA eval; archive_sha256=f3941568035d40bc7cb9e6fc0a108a5ec8bedf33f7ae14f6c060e92f7f247593"
```

Then run Modal exact CUDA auth eval through the reviewed PR106-R2 PR101-grammar
runtime. Until that JSON exists, this remains a byte-closed candidate, not a
score claim.
