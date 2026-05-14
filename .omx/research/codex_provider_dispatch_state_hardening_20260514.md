# Codex Provider Dispatch State Hardening - 2026-05-14

## Scope

Follow-up hardening after PacketIR/xmember closure, focused on provider
dispatch-state authority before further score-lowering work.

## Landed engineering

- Hardened `tools/trigger_gha_cpu_eval.py` run binding. The trigger now refuses
  to fire if the pre-dispatch workflow-run snapshot cannot be parsed, and it
  fails closed when the post-dispatch run-list delta contains more than one new
  workflow run. It no longer binds `dispatch_metadata.json` to the largest run
  id under concurrent dispatch ambiguity.
- Hardened `tools/operator_authorize.py` smoke cost metadata. A smoke
  authorization now has an explicit `CostBandRequest` context, resolves the
  smoke cost bucket (`platform/gpu/epochs`), and uses a smoke-scaled fallback
  p50 only when the posterior has no usable smoke bucket. The full-run A100
  fallback is preserved as reference metadata rather than being recorded as the
  smoke p50.
- Hardened `tools/run_modal_smoke_before_full.py` so smoke dispatches pass both
  `--cost-band-epochs-override` and `--cost-band-gpu-override`. This makes
  operator authorization and lane-claim notes match the actual smoke GPU and
  epoch budget.
- Added focused regressions covering ambiguous GHA run deltas, unparseable GHA
  pre-snapshots, smoke-scaled cost fallback, smoke authorization banner/claim
  notes, and smoke wrapper GPU override threading.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_gha_cpu_eval_harness.py src/tac/tests/test_operator_authorize_canonical_tool.py src/tac/tests/test_run_modal_smoke_before_full.py src/tac/tests/test_check_167_smoke_before_full_pattern.py src/tac/tests/test_cost_band_calibration.py`
  - `120 passed`
- `MODAL_GPU=T4 .venv/bin/python tools/operator_authorize.py --recipe substrate_sane_hnerv_modal_a100_dispatch --dry-run --yes --label-suffix __smoke__100ep --timeout-hours-override 1 --cost-band-epochs-override 100 --cost-band-gpu-override T4`
  - Banner shows `platform: modal (T4)` and `cost context: smoke override
    modal/T4 x 100 epochs`.
- `.venv/bin/python tools/operator_authorize.py --recipe substrate_sane_hnerv_modal_a100_dispatch --dry-run --yes`
  - Full-run banner remains `platform: modal (A100)`.
- `git diff --check`
  - clean.

## Frontier routing note

Fresh byte-target review after xmember closure ranks the next meaningful
exact-evaluable byte target as the `inner_decoder_packed_brotli` decoder
section (`169990` bytes). Any decoder-section successor must remain
hypothetical until it clears section equality, same-runtime full-frame parity,
terminal lane-claim custody, and exact CUDA auth eval. HLM2 latent microcodec
is the lower-risk follow-up after the provider-state hardening commit.
