# Sub015 DAG FEED — int8 training rungs (2026-07-13)

`research_only=true` · `lane_id=lane_int8_training_rungs_20260713` · local probes only · no training/eval/paid dispatch

## Executable dependency graph

```text
FEED-int8-0  exact v7.5.2 EMA custody (ef2c097f…c965) + real n600 GT cache
    |
    +--> FEED-int8-A1  inspect installed MLX 0.31.2 quantized layer/API surface
    |       `-- SOURCE_VERIFIED: quantized_matmul + QuantizedLinear/Embedding; no QuantizedConv/to_quantized
    |
    +--> FEED-int8-A2  CoreML W8A8 conversion readiness
    |       `-- BLOCKED_NOT_MEASURED: coremlcompiler present; coremltools absent
    |              |
    |              `--> TICKET: n600 calibration -> activation quant -> weight quant -> CPU_AND_NE compile
    |                           -> persistent warm latency -> concurrent GPU-witness overlap
    |
    +--> FEED-int8-A3  W8A8 QDQ/fp32-accum/STE scorer instrumentation
    |       |-- Metal preflight
    |       |     `-- BLOCKED_NOT_MEASURED in contained process (no Metal device)
    |       `-- terminal gate = n600 AND global cosine>=.99 AND min-pair cosine>=.99 AND speedup>=1.5
    |              `-- if fail: operator-wise fp32 precision waterfill; never kill int8 family globally
    |
    `--> FEED-int8-B0  build actual LVLS1 per-tensor int8 payload
            |-- canonical parser round-trip == direct int8_dequant_params (required)
            |-- fp32 EMA counterfactual and parsed-int8 treatment
            |-- same NumPy receiver -> real R -> frozen CPU SegNet, n600
            `-- FEED-int8-B1  signed post-hoc gap Δd_seg = d_seg(int8)-d_seg(fp32)
                    |-- MEASURED_N600: 0.0370965152 - 0.0375266266 = -0.0004301114
                    |-- DERIVED: -0.0430111 Seg score units; positive QAT recovery ceiling = 0
                    |-- premise "post-hoc int8 costs aggregate d_seg" FALSIFIED on this checkpoint
                    `-- FEED-int8-B2  default-OFF finishing-stage QAT A/B ticket
                            control: fp32 train -> post-hoc LVLS1 int8
                            treatment: same stage budget -> FakeQuantSTE at exact LVLS1 grid
                            admission: parsed receiver n600 + exact bytes, never proxy loss
                            scope: confirmatory/local-basin arm, not promised recovery of a positive gap
```

## Gradient boundary for FEED-int8-A2

Frozen scorer weights do not need gradients, but the training loss needs the scorer-input VJP
`J_S(x)^T dL/dS`. Therefore ANE forward-only is exact for monitoring/verdict work but cannot replace the
gradient-bearing teacher. A heterogeneous training treatment needs a measured custom VJP:

`z_ANE = S_ANE,W8A8(x)` and `g_x = J_S_MLX,QDQ(x)^T * (dL/dz evaluated at declared logits)`.

The ideal overlap equation `T=max(T_GPU-witness,T_ANE-forward)+T_sync` applies only to independent
forward-only monitoring. It is not a license to omit or double-count the MLX VJP.

## Triality + six-hook wire-in

- Equations: `tac.canonical_equations.int8_training_rungs_20260713`.
- DSL: `tac.witness_dsl.int8_training_rungs_policy`; both stubs default OFF, unwired, argv `[]`.
- DAG: this file.
- Sensitivity map: A3 per-pair cosine/flip rows become operator-waterfill input after the Metal ticket runs.
- Pareto constraint: A3 uses the conjunction of quality and speed; B uses receiver d_seg plus exact bytes.
- Bit allocator: QAT grid is exactly the existing LVLS1 per-tensor absmax/127 grid; no #336 file is consumed or changed.
- Autopilot dispatch: none while OFF_UNWIRED; future dispatch requires parser-backed wiring and a lane claim.
- Continual learning: terminal A3/B receipts are the empirical anchors; blocked receipts remain blocker signal, not zeros.
- Probe disambiguator: ANE monitoring-only and ANE+custom-VJP are separate modes; their timing/quality cannot be merged.

## Verdict scopes and reformulation queue

- A1 `NO_NATIVE_QUANTIZED_CONV` is scoped to the installed MLX 0.31.2 public Python/API surface.
- A2 `BLOCKED_NOT_MEASURED` is scoped to this contained interpreter; it is not an ANE capability verdict.
- A3 `BLOCKED_NOT_MEASURED` is scoped to the no-Metal contained process. Re-run the exact resumable tool on a
  Metal-entitled M5 host, then waterfill any failing operators.
- B is scoped to the exact v7.5.2 EMA, first real n600 pairs, Seg-only macOS CPU advisory path; it says nothing
  about d_pose, another checkpoint, achieved QAT recovery, or contest score.

Pointer delta: ZERO. No frontier/promotion/run-dir mutation.
