# ddm_mx1b Receipt - MX1 Load-Phase Memory Hardening

Date: 2026-08-06
Axis: CPU-side load telemetry in sandbox; MLX/Metal allocator blocked locally
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

borrowed_substrate_accounting: PR130 semantic renderer mechanism is unchanged. This landing is OUR engineering around load order, memory telemetry, allocator limits, launch-ticket scheduling, and tests.

## Verdict

Implemented in this working tree; the final serializer commit is reported by the operator-facing closeout rather than embedded in this pre-commit receipt snapshot.

The pre-fix allocator class was real: the mlx-train path held both full 600-pair label caches while it then created selected NumPy/MLX arrays and loaded the scorer. In the baseline blob, this was `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 382-385 (`torch.load(... )["seg"].long()` for both caches, followed by slicing). The fix routes all three modes through selected clone/free helpers and adds load-stage RSS/MLX telemetry.

The local mem-probe is a BLOCKED receipt, not Metal clearance: this sandbox has the MLX package but no accessible Metal device, so it measures the torch/cache side through selected clone/free and stops before MLX initialization. MAIN must run the generated `mem_probe_command` on the Metal host and consume only a `status=passed` receipt before firing any arm.

## Measured CPU-Side Memory Table

Command:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mem-probe --device cpu --pairs 32 --mem-probe-steps 3 --run-dir .omx/research/ddm_mx1b_20260806/mem_probe_cpu --out .omx/research/ddm_mx1b_20260806/mem_probe_cpu_result.json
```

Exit: `2` because MLX/Metal is unavailable in this sandbox after the CPU load stages.

Receipt: `.omx/research/ddm_mx1b_20260806/mem_probe_cpu/mem_probe_receipt.json`
SHA-256: `79fc2fb2ee2a430caf48695d73ed4497771d4c347a4b6bf4d02200cb336a8ae3`

| stage | RSS GiB | delta RSS GiB | available GiB | delta available GiB | MLX active/cache |
|---|---:|---:|---:|---:|---|
| start | 0.200165 | 0.000000 | 105.173141 | 0.000000 | unavailable |
| after_init_checkpoint_torch_load | 0.201294 | 0.001129 | 105.173294 | 0.000153 | unavailable |
| before_selected_cache_load | 0.201309 | 0.001144 | 105.173294 | 0.000153 | unavailable |
| after_input_cache_selected_clone | 1.175095 | 0.974930 | 104.202820 | -0.970322 | unavailable |
| after_target_cache_selected_clone | 1.221985 | 1.021820 | 104.151031 | -1.022110 | unavailable |
| after_selected_cache_numpy_copy_and_torch_free | 1.268951 | 1.068787 | 104.105133 | -1.068008 | unavailable |

Peak CPU-side RSS in this sandbox: `1.268951 GiB`.
Clearance checks: `required_stage=after_train_step_000003`, `has_required_stage_sample=false`, `has_mlx_allocator_telemetry_at_required_stage=false`, `metal_fire_clearance=false`.
Metal-only unknowns: MLX active/cache/peak, model conversion, scorer conversion, R round-trip, and 3 training steps are not measured here because `require_mlx(device="cpu")` raises `MlxUnavailableError: [metal::load_device] No Metal device available`.

## What Changed

- Added typed RSS/system-available/MLX allocator sampling with `LoadPhaseMemoryProbe`.
- Replaced full-cache retained tensors with `_load_selected_seg_tokens`: `torch.load`, `index_select`, selected `.clone().contiguous()`, delete full payload, and `gc.collect()`.
- Reordered `run_mlx_train` so selected CPU cache arrays are prepared before MLX setup, then torch checkpoint/scorer tensors are deleted immediately after MLX conversion.
- Added setup `mx.eval` barriers after model init, model weight conversion, selected token conversion, scorer conversion, and each mem-probe step.
- Added `--mem-budget-gb` with default 50% of available memory at process start and guarded calls to `set_memory_limit` / `set_cache_limit` on both direct and `metal.*` MLX APIs.
- Added `--mode mem-probe`, which runs the load path plus three steps when MLX is available and writes `mem_probe_receipt.json`.
- Regenerated the v2 two-arm launch ticket shape with `mem_probe_receipt_required: true`, `mem_probe_receipt_path`, `mem_probe_command`, and sequential one-Metal-arm scheduling while preserving `argv_n32_arm_cap`, `argv_n32_arm_veh`, `argv_n120_arm_cap`, and `argv_n120_arm_veh`.

Generated probe artifact:

- `.omx/research/ddm_mx1b_20260806/probe_result.json`

## RECALL EVIDENCE

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `.omx/tmp/codex_runs/mx1b_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, and incident memory | Live board has own-vehicle `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved; ET4 owns scorer slot. | Kept this arm scorer-free and did not claim pointer movement. |
| Prior PR130/MX1 reviews | `rg "ddm_mx1|mx1|mlx-train|mem-probe|memory spike|subset-before|launch ticket|ticket schema|two_arm|v2_two_arm"` over research/state/docs/experiments/src | RR4 verified the `v2_two_arm` ticket and four argv keys; RR5 required fire-time regeneration and current cache/init hash checks. | Preserved the v2 two-arm structure and only added receipt/scheduling fields. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` with PR130/MX1/authority terms | No PR130-lift equation superseded the authority split; score-axis equations reinforced no score claim. | Receipt remains load telemetry only. |
| Full corpus/state | `rg "PR130|lift wave|ddm_rr[234]|ET4|HB1|MX1|Row-1|v2_two_arm"` over research index, DAG, state, docs, experiments | Campaign order remains ET4 evaluate before Row-1; MLX telemetry is research-signal. | Added one-Metal-fire scheduling and no n120 before n32 CPU-torch verdict selection. |
| Source inspection | `git show HEAD:experiments/ddm_mx1_pr130_semantic_renderer.py` and current `nl -ba` | Baseline retained full 600-pair caches before slicing; current helper drops full payload after selected clone. | Diagnosed the allocator at file:line and made the fix mechanism-preserving. |

## Verification

```bash
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
.venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py -q
```

Results: ruff passed, py_compile passed, focused pytest `3 passed`.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
