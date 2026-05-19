# Codex Findings: PR107 Apogee CD1 Master-Gradient Projector

Timestamp UTC: 2026-05-19T06:37:00Z
Agent: codex
Scope: OP_SYN_1 master-gradient six-archive extension
Evidence axis: `[diagnostic-CPU]` smoke only; no contest score claim.

## Finding

PR107 Apogee is no longer just a detectable xray grammar. Its stock `0.bin`
payload exposes a Brotli-compressed `CD1` decoder section with architecture-
ordered HNeRV tensors:

- `meta_brotli`: 80 bytes; JSON declares `n_pairs=600`, `latent_dim=28`,
  `base_channels=36`, `eval_size=[384,512]`.
- `decoder_blob`: 162,343 bytes; Brotli inflates to `CD1`, `scale_bits=16`,
  `n_tensors=28`, raw length 229,022 bytes.
- `latents_brotli`: 15,849 bytes.

The decoder section is projector-eligible because the source codec and model
define a deterministic `CD1` layout:

```text
CD1 magic(3), scale_bits(1), n_tensors(u32),
then for each HNeRVDecoder state_dict tensor:
  scale(fp16 or fp32), zigzagged INT8 mantissa bytes[numel]
```

## Landing

Implemented a registered projector:

```text
pr107_apogee_cd1_decoder_jacobian_camera_offset_roundtrip_latents_zero_grad_v1
```

This projector:

- parses PR107 metadata and validates required dimensions;
- inflates and validates the `CD1` decoder payload;
- maps each architecture-ordered tensor's scale and mantissa offsets;
- stamps the decoded PR107 state dict through the differentiable scorer path
  with PR107's runtime camera-space channel offsets applied before the STE
  roundtrip (`frame0 R/B -= 1`, `frame1 G -= 1`);
- projects per-parameter gradients back onto the compressed decoder-byte
  region using the same conservative Brotli-spread approximation as the
  existing packed-HNeRV projectors.

Metadata and latent sections remain explicit zero-gradient v1 surfaces. Rate
authority remains packet-valid operator-response only; byte-count effects are
not inferred from this diagnostic tensor.

## Verification

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_extract_master_gradient.py \
  src/tac/tests/test_master_gradient_archive_parsers.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  tools/extract_master_gradient.py \
  src/tac/tests/test_extract_master_gradient.py \
  src/tac/master_gradient_archive_parsers.py \
  src/tac/tests/test_master_gradient_archive_parsers.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip \
  --detect-grammar-only

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip \
  --inflate-py experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/source/submissions/apogee/inflate.py \
  --upstream-dir upstream \
  --axis '[diagnostic-CPU]' \
  --output-npy .omx/state/master_gradient_pr107_apogee_diagnostic_1pair_20260519_codex.npy \
  --device cpu \
  --n-pairs-used 1 \
  --n-pairs-total 600 \
  --no-anchor-write \
  --compute-dtype float32 \
  --storage-dtype float32
```

Results:

- focused pytest: `56 passed`;
- touched-file Ruff: passed;
- detect-only JSON reports `gradient_projection_supported=true` and
  `required_projector=pr107_apogee_cd1_decoder_jacobian_camera_offset_roundtrip_latents_zero_grad_v1`;
- one-pair no-anchor smoke wrote finite sidecar:
  `.omx/state/master_gradient_pr107_apogee_diagnostic_1pair_20260519_codex.npy`
  with SHA-256 `f66cf7f11c897c3fb90e7cf9b65bd3a5c0bf670179992a24b80f07f41331f0a7`,
  shape `(178284, 3)`, dtype `float32`, nonzero entries `324246`, absmax
  `3.178710903739557e-05`.

## Authority

This is a diagnostic projector landing, not a score claim. A contest-axis
anchor still requires the normal authoritative hardware, sample-count, custody,
and ledger path. Candidate score-lowering mutations must still be emitted as
typed `CandidateModificationSpec` rows and validated through exact auth eval.

OP_SYN_1 remains open: DP1 renderer/codebook/residual projector and PR106
format0d sidecar operator-response rows are still separate blockers.
