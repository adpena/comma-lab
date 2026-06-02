# HiNeRV/SNeRV Blocker Sweep - 2026-06-02T20:41:03Z

Axis: [planning/control]

Verdict: blockers are not all burned down.

HiNeRV has 10 remaining method blockers plus 1 current execution blocker:
the lr2.7e-5 planner-row long worker returned 143/SIGTERM before
`compact_renderer_mlx_spine_runner_report.json`.

SNeRV has 20 remaining method blockers and no new execution blocker from this
pass.

This pass burned down the pre-artifact launch class enough to relaunch with
better custody: planner rows now request GPU distillation/scorer surfaces,
admitted local MLX rows carry an explicit 43200 second timeout, the compact
family runner writes a startup marker before heavy work, and the canonical
training harness can flush epoch telemetry every epoch.

Next attack order:

1. Relaunch the first HiNeRV planner-row from the 20:39 fixed queue and verify
   the startup marker exists before long training proceeds.
2. Attach a candidate-scoped HiNeRV decoder-weight waterfill plan; current
   waterfill plumbing is real, but the selected row still says the plan is
   missing.
3. For SNeRV, materialize the native scorer-loop best packet as receiver-closed
   SNAR1, then attach SNAR1 byte feedback and section-value profiling.
4. Bind SNeRV PR95 staged curriculum, eval-roundtrip STE, EMA selection, and
   Muon/AdamW partition to the native MLX adapter before treating SNeRV rows as
   source-grade.
5. Promote neither family until full-video MLX prefilter, byte-closed receiver
   proof, local CPU replay, and paired exact CPU/CUDA gates are satisfied.

False-authority: no score, rank, promotion, exact CPU, or exact CUDA claim.
