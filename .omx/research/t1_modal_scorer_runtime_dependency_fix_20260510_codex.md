# T1 Modal scorer-runtime dependency fix — Codex ledger

Date: 2026-05-10

## Result classified

Dispatch `t1_balle_modal_guard_740778ba_20260510T1000Z` closed terminally as
`failed_t1_modal_recovered_no_score_claim`.

- Modal call id: `fc-01KR8HFZPJRQHWTXAX7TT72D04`
- Stage: `remote_script_failed`
- Return code: `1`
- Score claim: `false`
- Promotion eligible: `false`
- Rank/kill eligible: `false`
- Harvest summary:
  `experiments/results/t1_balle_modal_guard_740778ba_20260510T1000Z/harvest_summary.json`

Root failure:

```text
ModuleNotFoundError: No module named 'segmentation_models_pytorch'
```

The run reached Stage 5 score-domain training and loaded CUDA/NVDEC correctly,
so the previous canonical-A1-payload mount bug was cleared. This failure is an
infrastructure/runtime dependency-closure bug, not a T1 model result.

## Fix

Implemented shared Modal scorer-runtime dependency closure in
`src/tac/deploy/modal/runtime.py` and changed
`experiments/modal_t1_balle_endtoend.py` to use it instead of lane-local package
lists.

The shared runtime now owns:

- apt/runtime package list for contest CUDA Modal workers;
- pip scorer/runtime package list including `safetensors` and
  `segmentation-models-pytorch`;
- DALI/NVML value;
- scorer import probe modules;
- Modal image builder helper.

T1-specific Modal logic now remains lane-actuator logic: claim lifecycle,
canonical A1 payload mount, T1 params, recovery, and score evidence gating.

Additional hardening:

- mount `tools/tool_bootstrap.py`, which `tools/build_phase1_packet_compiler.py`
  imports;
- add `t1_modal_import_probe` before score-domain training so missing scorer
  packages fail in a distinct `remote_import_probe_failed` stage;
- require labels containing `guard` to use bounded guard params
  (`epochs<=100`, `batch_size<=8`, `max_target_pairs<=64`,
  `train_timeout_hours<=3`);
- reduce default train timeout from `23.0h` to `22.5h` and require a 15-minute
  artifact-collection buffer beyond the post-train eval buffer.

## AGENTS.md protocol update

Added `Provider Runtime Architecture — NON-NEGOTIABLE`:

- provider/runtime contracts belong in `src/tac/deploy/<provider>/`;
- experiment/provider files should be thin lane adapters;
- Modal scorer dependency closure must use the canonical deploy helper;
- missing scorer deps classify as infrastructure failures, not model results;
- provider actuators must preserve deterministic reproducibility, mounted-code
  custody, claims, terminal rows, and exact evidence boundaries.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/modal_t1_balle_endtoend.py \
  src/tac/deploy/modal/runtime.py \
  src/tac/tests/test_modal_t1_balle_endtoend.py \
  src/tac/preflight.py \
  src/tac/tests/test_preflight_cli_timeout.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_preflight_cli_timeout.py \
  src/tac/tests/test_codex_round6_medium2_preflight_cli_default_scope.py \
  src/tac/tests/test_modal_t1_balle_endtoend.py

.venv/bin/python -m tac.preflight --scope dev \
  --timings-json experiments/results/preflight_dev_timing_provider_runtime_refactor_20260510_codex.json
```

Observed:

- focused tests: `37 passed`
- dev preflight: `PREFLIGHT PASSED`
- timing: `wall_elapsed_s=9.804019`, `timeout_s=30.0`
- claim summary after T1 and Kaggle harvests:
  `active=0`, `stale_nonterminal=0`, `terminal_latest=568`

## Relaunch criteria

Do not relaunch T1 under a `guard` label without bounded guard params. A
reasonable next guard command should use:

```text
--epochs 50 --batch-size 8 --max-target-pairs 64 --train-timeout-hours 2
```

Full T1 runs should use a non-guard label and should only launch after the
bounded guard passes the import probe, reaches training, emits packet artifacts,
and proves the packet compiler/auth-eval path.
