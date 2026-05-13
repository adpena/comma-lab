# PacketIR PR106 identity proof operator surface (2026-05-13)

## Scope

This landing promotes the existing PR106/R2 sidecar PacketIR identity check
from test-local logic into an operator-facing, reusable proof surface.

It does **not** claim score movement. It is a byte-custody and parser-accounting
artifact that must precede PR106/R2 PacketIR transforms before exact eval spend.

## Code changes

- Added `tac.packet_compiler.prove_pr106_sidecar_packet_ir_identity(...)`.
- Added `tools/prove_pr106_packetir_identity.py` as the thin CLI wrapper.
- Rewired `tools/all_lanes_preflight.py` Gate #26 to consume the canonical
  identity proof instead of duplicating parse/emit logic.
- Added regression tests for both release archives, fail-closed SHA mismatch,
  CLI manifest output, and all-lanes identity drift detection.

## Empirical artifact

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/prove_pr106_packetir_identity.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --expected-archive-sha256 c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383 \
  --output-json experiments/results/pr106_r2_pr101_packetir_identity_20260513_codex.json
```

Observed manifest:

- schema: `pr106_sidecar_packet_ir_identity_proof_v1`
- archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- manifest SHA-256:
  `a4e3a26607835fa30665dc2ca1481ec226edd41cdfcb574404c66b8c894d7c79`
- format_id: `0x02`
- `packet_ir_identity_passed=true`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py -q
```

Result: `29 passed in 1.09s`.

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/__init__.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py \
  tools/all_lanes_preflight.py \
  tools/prove_pr106_packetir_identity.py
```

Result: `All checks passed!`.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m tac.preflight --scope dev \
  --timings-json reports/preflight_dev_timing_20260513_packetir_identity_final.json
```

Result: `PREFLIGHT PASSED`.

Pre-stage all-lanes check:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/all_lanes_preflight.py \
  --jobs 8 --timings \
  --timings-json reports/all_lanes_preflight_timing_20260513_packetir_identity.json
```

Result: Gate #26 passed; wall-clock was `2.15s` with 8 workers. The only
failure was Gate #10 because the new tool/test and a throwaway stdout JSON were
still untracked during the development run. The throwaway stdout file was
deleted; the failed pre-stage timing artifact was not retained.

Post-stage all-lanes check:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/all_lanes_preflight.py \
  --jobs 8 --timings \
  --timings-json reports/all_lanes_preflight_timing_20260513_packetir_identity_poststage.json
```

Result: `ALL 30 PREFLIGHT CHECKS PASSED`; wall-clock was `2.15s`, serial sum
was `12.54s`, and estimated speedup was `5.84x`. Gate #26 passed with
`format_ids=0x01,0x02`; it still prints `score_claim=false` and
`ready_for_exact_eval_dispatch=false`.

## Score-lowering implication

The next PR106/R2 compiler transform should start from this proof:

1. identity proof from `tools/prove_pr106_packetir_identity.py`;
2. runtime decode/apply proof from `tools/prove_pr106_sidecar_runtime_consumption.py`;
3. full-frame same-runtime parity when claiming equivalence;
4. exact `[contest-CUDA]` / `[contest-CPU]` eval only after archive/runtime
   custody and lane claim are closed.

This keeps PR106/R2 byte transforms from becoming no-op or proxy-score claims
while preserving the path to real sidecar/compression score lowering.
