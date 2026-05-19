# Codex Findings - PR106 Format0d Primary Projector

Date: 2026-05-19T06:29:00Z
Author: Codex
Task: `codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518::OP_SYN_1`
Score claim: false

## Finding

`pr106_format0d` no longer needs to remain fully xray-only. The PacketIR
wrapper proves exact boundaries for the primary PR106 payload, and the primary
payload is the packed-HNeRV `0xff + decoder_len24 + decoder_blob + latent_blob`
grammar consumed by the PR106 runtime. That decoder blob can be projected
through a grammar-aware packed-HNeRV per-parameter Jacobian.

This landing therefore promotes `pr106_format0d` to anchor-ready for the
primary packed-HNeRV decoder bytes, while keeping the discrete base/extra
ranked sidecar streams at explicit zero-gradient v1 semantics. Sidecar
score-response still belongs in packet-valid `CandidateModificationSpec`
operator rows, not raw byte-gradient authority.

## Patch

- `tools/extract_master_gradient.py`
  - registered
    `pr106_format0d_primary_packed_hnerv_decoder_jacobian_sidecar_zero_grad_v1`
    as the PR106 format0d projector;
  - added a packed-HNeRV decoder layout parser for format0d primary payloads;
  - decodes and applies format0d base+extra sidecar correction passes before
    scorer forward so the measured operating point matches runtime behavior;
  - emits a PR106-specific measurement method string for aggregate and per-pair
    sidecars.
- `src/tac/master_gradient_archive_parsers.py`
  - marks `pr106_format0d` anchor-emitting in the public parser facade.
- Tests now pin the live archive's `gradient_projection_supported=true`, the
  projector contract, synthetic packed-decoder byte mapping, and runtime-order
  sidecar correction application.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_master_gradient_archive_parsers.py`
  - `54 passed in 0.73s`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py src/tac/master_gradient_archive_parsers.py src/tac/tests/test_master_gradient_archive_parsers.py`
  - `All checks passed`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py list-grammars | .venv/bin/python -m json.tool >/dev/null`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py --archive experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip --detect-grammar-only | .venv/bin/python -m json.tool >/dev/null`
- One-pair no-anchor diagnostic smoke:
  - command used `--axis '[diagnostic-CPU]' --n-pairs-used 1 --no-anchor-write`
  - archive SHA `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
  - subject SHA `15a5dccba352838df7a8dd190a8782e51afa53b475bcf07921f05ce88c10785e`
  - output `.omx/state/master_gradient_pr106_format0d_diagnostic_1pair_20260519_codex.npy`
  - sidecar SHA `0eb941ede9fae5a02386ac49fd4e3ca198dce7cfae058da53bc60bdc2ac3b87a`
  - shape `(186776, 3)`, dtype `float32`, finite `true`, nonzero `339370`,
    absmax `3.714456033776514e-05`

## Residual

OP-SYN-1 remains open. DP1 and PR107 are still detection-only, and PR106
format0d sidecar-byte operator response is still not a raw byte-gradient claim.
The next projector slice should either add PR107 length-prefixed schema
projection or DP1 renderer/codebook/residual projection with the same
fail-closed authority boundary.
