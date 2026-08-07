# ddm_mx1b Next If Resumed

## Fire Protocol

1. Regenerate the ticket from the current source on MAIN:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode probe --run-dir .omx/research/ddm_mx1b_20260806/probe_ticket --out .omx/research/ddm_mx1b_20260806/probe_result.json
```

2. Run the generated `launch_ticket.mem_probe_command` on MAIN Metal before any `mlx-train` arm. The receipt must satisfy all of:

- `schema == "ddm_mx1_load_phase_peak_receipt.v1"`
- `status == "passed"`
- `metal_fire_clearance == true`
- samples include `after_train_step_000003`
- MLX active/cache/peak fields are present
- composed peak projection leaves headroom under the 116 GiB host ceiling

3. If mem-probe blocks or lacks MLX allocator samples, do not fire ARM-CAP or ARM-VEH. The blocked sandbox receipt here is not clearance.

4. Fire one Metal training process at a time. ARM-CAP fires first. ARM-VEH fires only after ARM-CAP completes, unless a composed measured-peak projection from passed receipts proves concurrent headroom under 116 GiB.

5. Preserve the v2 two-arm structure:

- `argv_n32_arm_cap`
- `argv_n32_arm_veh`
- `argv_n120_arm_cap`
- `argv_n120_arm_veh`

No bare `argv_n32` or `argv_n120` is valid.

6. Do not dispatch n120 until both n32 arms have CPU-torch verifier verdicts and the scaled arm is explicitly selected. MLX telemetry remains `[macOS-MLX research-signal]`.

7. This arm does not own the n600 scorer slot. Queue any scorer step behind the live slot owner and label every result by axis.

## If The Metal Mem-Probe Still Spikes

- Read the passed/blocked receipt's per-stage peak.
- If the spike occurs before `after_require_mlx_and_memory_limits`, the remaining class is torch deserialization / OS cache / checkpoint load.
- If the spike occurs between `after_require_mlx_and_memory_limits` and `after_model_weight_mlx_conversion`, inspect MLX model init and torch-state conversion.
- If the spike occurs at `after_segnet_mlx_conversion`, split scorer conversion from renderer training or require a separate scorer preload cap.
- If the spike occurs at `after_train_step_000001`, treat it as the #205 lazy-graph genus and add narrower `mx.eval`/cache-clear barriers inside the first loss/roundtrip/scorer call.

## Current Evidence Paths

- CPU-side blocked mem-probe: `.omx/research/ddm_mx1b_20260806/mem_probe_cpu/mem_probe_receipt.json`
- CPU-side driver result: `.omx/research/ddm_mx1b_20260806/mem_probe_cpu_result.json`
- Regenerated probe ticket artifact: `.omx/research/ddm_mx1b_20260806/probe_result.json`
- Diagnosis: `.omx/research/ddm_mx1b_20260806/MEM_DIAGNOSIS.md`
- Receipt: `.omx/research/ddm_mx1b_20260806/RECEIPT.md`

Own-vehicle frontier remains unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
