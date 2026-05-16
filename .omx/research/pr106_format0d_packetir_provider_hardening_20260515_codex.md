# PR106 Format0D PacketIR And Provider False-Authority Hardening

Date: 2026-05-15

## Scope

This ledger records a hardening tranche produced from adversarial review. It
does not claim a contest score or promotion.

## Fixes

### Format0D Runtime Consumption

Format `0x0D` carries two additive correction passes:

1. `base_format0c_sidecar_payload`
2. `extra_pr101_ranked_no_op_payload`

The runtime-consumption proof now:

- decodes `0x0D` through the submission runtime's `decode_format0d_sidecar()`;
- applies base corrections before extra corrections;
- records `runtime_apply_order`;
- records section identities for `pr106_payload`,
  `base_format0c_sidecar_payload`, `extra_pr101_ranked_no_op_payload`, and
  `extra_framing_meta`;
- probes `extra_framing_meta` fail-closed through the runtime parser;
- mutates the extra PR101 payload for semantic consumption proof.

The materialized archive
`experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`
with SHA-256
`9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
now produces a non-score runtime-consumption manifest instead of crashing on
`unsupported PR106 sidecar format_id=0x0D`.

### PacketIR Exact Closure

`src/tac/packetir_exact_closure.py` now treats `0x0D` as a special multi-pass
closure mode. It requires:

- PacketIR parser sections for the four score-affecting sections above;
- matching runtime section SHA/length identities;
- base-then-extra runtime apply order.

Parser identity alone is insufficient.

### Provider False Authority

Azure remains scaffold-only in the provider contract. The Azure launcher now
refuses `--no-dry-run` while the contract lacks an execution flag or exact-CUDA
support. Cloud-provider readiness text now derives exact-CUDA destinations from
provider contracts instead of naming scaffold providers as exact targets.

## Verification

Focused local verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packetir_exact_closure.py \
  tests/test_cloud_provider_readiness.py \
  tests/test_launch_lane_azure.py \
  src/tac/tests/test_provider_deploy_contracts.py \
  src/tac/tests/test_azure_dispatch.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py -k packetir_closure
```

Observed: 136 tests passed in the combined focused suite; 3 packet-closure
tests passed in the HMD9 recode slice.

Static checks:

```bash
.venv/bin/python -m ruff check <touched Python files>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile <touched Python files>
git diff --check -- <touched files>
```

Observed: all passed.

## Remaining Boundary

PR106/R2 and Format0D are still non-promotional until exact same-runtime
auth-eval artifacts exist with archive/runtime custody, component
recomputation, and explicit `[contest-CPU]` or `[contest-CUDA]` labels.
