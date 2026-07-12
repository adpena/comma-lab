# V9 CGauge warm-restart ticket — micro-batch B2

Date: 2026-07-12  
Status: **PREPARED, NOT FIRED**  
Axis: `[macOS-MLX training-gradient]`; advisory only; `score_claim=false`  
Vehicle: `v9_cgauge_432`  
Change from stopped arm: `--micro-batch-pairs 2` only

## Canonical checkpoint custody

The operator-cited ep251 stage boundary remains preserved, but the canonical rolling resume
checkpoint advanced to ep275 before the process stopped. Resuming ep251 would discard 24 completed,
checkpointed epochs, so this ticket uses the freshest disk state.

| artifact | encoded epoch | bytes | SHA-256 | disposition |
|---|---:|---:|---|---|
| `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_resume_state.npz` | 275 | 1,499,516 | `17c4b4765370ee39d34919805a1e45f6c88ce8b213c70aaaf18100cbf58881e2` | **resume source** |
| `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_resume_stageOctave1_ep251.npz` | 251 | 1,499,196 | `792525515bbf55ae354d32cd218e84c8f63408317c2a90efebfa65e68c06e0a2` | preserved fallback |
| `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz` | 251 | 380,136 | `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758` | preserved deploy checkpoint |
| `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz` | 150 | 379,776 | `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c` | preserved best, not resume source |

## Governed launch command — do not fire from this ticket

Run only after the main loop reviews the real-Metal parity/timing receipt and the landed main SHA.
The launcher re-derives the sealed V9 argv, validates all trainer flags, runs memory/system/throughput
admission, emits per-stage and periodic checkpoints, and launches through the governed daemon.

```bash
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --config v9_cgauge_432 \
  --extra-trainer-flags "--resume-from experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_resume_state.npz --micro-batch-pairs 2" \
  --out-dir experiments/results/v9_cgauge_432_coherent_arm_microbatch_resume_20260712 \
  --purpose "warm restart of stopped V9 CGauge arm from latest preserved resume checkpoint; throughput-only micro-batch B2 delta"
```

No epoch, schedule, loss weight, scorer, optimizer, checkpoint cadence, or V9 lever override is added.
The named config retains sealed `epochs=3000`, `--ckpt-every 25`, and `--stage-checkpoints`.

## Measured preflight projection

Dry-run command: the command above plus `--dry-run --skip-throughput-gate`. It exited 0, wrote and
validated the launch script, and did **not** spawn a trainer or daemon.

- Trainer argparse: **199/199 flags valid**; exactly one `--micro-batch-pairs 2`.
- Expected-active-lever manifest: **OK** (10 pinned V9 levers).
- Typed DSL gate: **OK**; schedule-provenance gate: **OK**.
- Projected peak RSS: **24.48 GiB** = fixed 15.0 + cf_mx cache 0.07 + GT 3.41 + verdict 6.0.
- Physical memory: **128.0 GiB**; observed available: **92.3 GiB**.
- Policy safe ceiling: **89.6 GiB** (70% concurrent policy branch).
- System-aware admission projection: **ADMIT**; projected used 60.2 GiB <= adaptive ceiling
  103.7 GiB, with 43.5 GiB headroom.
- Safe-compile fingerprint: **OK** for `hosc_activation`.
- Fast-path environment: `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` and
  `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1` emitted.

The dry-run observed the governed `levelset_dash_supervisor` control-plane process; the
system-admission calculation reported zero active heavy training jobs. The real launch must rerun
the same preflight against then-current system state and fail closed on any governor refusal.

## Fire gate

**HOLD.** Required before launch: real Aqua-session Metal execution of the complete focused tests and
the faithful equal-step B1-vs-B2 full-V9 receipt, including functional loss/gradient tolerance,
end-to-end speedup, and peak RSS. A headless/sandboxed MLX process is not evidence: it currently
fails at `metal::load_device`, so no throughput number is inferred from the dry-run or historical
component benchmarks.

The reviewed code must also be serialized and pushed from a Git-writable session. This managed
session exposes `.git` and `.git/objects` read-only. A post-edit-SHA serializer attempt on the
17-file frozen-SegNet/ELM unit failed closed at `git add` with `unable to create temporary file:
Operation not permitted`; the index remained empty. Do not claim a landed SHA from this session.

Pointer delta: zero. This ticket authorizes no score claim and no launch by itself.
