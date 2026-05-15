# Modal auth-eval device axis guard

Date: 2026-05-15

## Failure Class

D1 Modal smoke `fc-01KRNBXVYS345TDFW33BA4E77P` completed the trainer/export path
and produced byte-closed artifacts, then failed in inline auth eval because the
canonical substrate helper ignored the provider wrapper's explicit
`AUTH_EVAL_DEVICE=cpu` setting and hardcoded `--device cuda`.

Evidence:

- Artifact directory:
  `experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260515T082638Z__smoke__50ep_modal`
- Archive: `archive.zip`, 185676 bytes.
- D1 payload: `d1_polytope.bin`, 7390 bytes.
- Modal log: `modal_lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260515T082638Z__smoke__50ep.log`
- The failed auth-eval command used `--device cuda` despite the Modal wrapper
  exporting `AUTH_EVAL_DEVICE=cpu`.
- Upstream `evaluate.py` then entered the CUDA DALI/NVDEC path and failed with
  `nvml error (999): A nvml internal driver error occurred`.

## Fix

`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` now separates
the trainer device from the auth-eval device:

- `auth_eval_device=...` or `AUTH_EVAL_DEVICE` explicitly selects the auth-eval
  device.
- CPU and CUDA auth-eval results remain separate evidence axes.
- CPU auth eval can run on contest-compliant Linux x86_64 provider hardware and
  produce `contest_cpu` evidence, but it is never returned through
  `auth_eval_cuda_score`.
- Legacy CUDA-claim callers receive `None` for non-CUDA auth-eval results unless
  they opt in with `return_non_cuda_result=True`.

## Guard Proof

Focused tests:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_smoke_auth_eval_gate.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_probe_pr106_format0b_sidecar_compression.py -q
```

Result: `88 passed`.

Lint:

```bash
.venv/bin/python -m ruff check \
  src/tac/substrates/_shared/smoke_auth_eval_gate.py \
  src/tac/tests/test_smoke_auth_eval_gate.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/__init__.py \
  src/tac/packetir_exact_closure.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_probe_pr106_format0b_sidecar_compression.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  tools/probe_pr106_format0b_sidecar_compression.py
```

Result: `All checks passed`.

## Reactivation

Re-dispatch D1 or any Modal training lane after this commit. The expected
behavior is:

- Modal wrapper sets `AUTH_EVAL_DEVICE=cpu`.
- Inline auth eval runs `contest_auth_eval.py --device cpu`.
- The result is recorded as CPU-axis evidence, not a CUDA claim.
- CUDA ranking still requires a separate CUDA auth-eval packet on a provider
  whose CUDA video path is healthy.
