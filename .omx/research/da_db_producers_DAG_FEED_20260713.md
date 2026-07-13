# DAG FEED — D-A/D-B telemetry producers and #408 resume-boundary batch — 2026-07-13

`research_only=false` · `score_claim=false` · `pointer_moved=false` · `$0 local`  
Lane: `lane_da_db_producers_20260713` · checkpoint: `da_db_producers`

## Terminal build state

Both telemetry-producer build blockers are cleared in the trainer and in the two canonical launch
tickets. The governed dry-runs remain `REFUSE_RC11_NO_SPAWN` because two independent system blockers
remain: `MEMORY_WATERFILL_B2_UNMEASURED_N600` and `SSD_WORKLOAD_ROOT_MISSING`. The full variant also
fails the current memory-governor snapshot; the trimmed variant is admitted by that snapshot. No
trainer, GPU, paid dispatch, archive mutation, or score evaluation was launched.

## Executable dependency graph

```text
typed WitnessProgram
  + TelemetryCadence(default ON, score-neutral)
  + VerdictLiveGap(default OFF, score-affecting advisory inference cadence)
       -> compiled trainer argv / parser ownership
       -> canonical resume registry: da_db_telemetry
       -> legacy checkpoint load with absent telemetry state => deterministic defaults
       -> trainer epoch loop
            |- D-A update-free one-pair component decomposition
            |    teacher fwd / teacher bwd / witness fwd / witness bwd / R
            |    + real verdict worker/sync duration
            |    + real checkpoint I/O duration
            |    + monotonic epoch total
            |       -> witness_component_wallclock.v1 (8 exact fields, fcntl JSONL)
            |
            |- D-B nominal and actual engagement observer
            |    ep450 screw +/-2 and ep726 phase +/-2; deterministic n600 K=4 strata
            |    separate seg / pose / temporal trunk gradients; no update or EMA mutation
            |       -> sps_gradient_role_conflict_engagement.v1 (fcntl JSONL)
            |
            `- #408 Q1-Q7 additive rows
                 clip activation / term inertness / optional live gap / tail endpoint /
                 would-fire / ladder completion / uniform lever engagement
       -> causal-manifest no-score boundary rows (compose; do not duplicate)
       -> resume-safe producer latch/checkpoint state
       -> full + trimmed composer
       -> governed launcher dry-run
            D-A clear + D-B clear
            remaining system blockers only; no spawn
```

## Equation leg

No empirical law is minted by this apparatus landing. It operationalizes the standalone SPS probe's
already preregistered conflict test and a monotonic timing identity:

```text
T_component = sum_j (perf_counter_ns_after_j - perf_counter_ns_before_j) / 1e9
cos(g_p, g_t) = <g_p,g_t> / (||g_p||_2 ||g_t||_2)
conflict = [cos <= -0.05] AND [negative-product scalar fraction >= 0.10]
```

The NumPy reducer preserves the standalone probe's key names, coactivity epsilon, per-tensor rows,
and zero-norm `null` cosine. These rows are mechanism evidence only until a real engaged n600 run
emits them.

## Triality and six-hook disposition

- **DSL:** `TelemetryCadence` owns the default-ON read-only producer cadence. `VerdictLiveGap` owns
  `--verdict-live-gap-every`; cadence zero is byte-identical/default-OFF and remains a tracked duty.
- **DAG:** this standalone FEED is the dependency and promotion boundary. The shared hot DAG was not
  edited while sibling agents were live.
- **Equations:** reuse the exact gradient-cosine/conflict rule above; no score or measured law added.
- **Sensitivity map:** D-B supplies engaged-regime gradient-role geometry for the SPS reformulation;
  no live sensitivity map is mutated before a real row exists.
- **Pareto constraint:** observer work must be score-neutral, update-free, and bounded to deterministic
  K=4 strata. A row that perturbs optimizer/EMA/RNG state is inadmissible.
- **Bit allocator:** N/A-with-reason; the telemetry ships zero archive bytes.
- **Cathedral/autopilot:** the existing `sps_weight_space_gradient_role_separation` pool row remains
  `reformulation-queue`; its gate now names this engaged n600 producer. Q3 is registered
  `built-never-fired` via canonical `record_candidate`.
- **Continual learning:** component wall split and SPS conflict become typed, queryable run rows; no
  historical 95/5 estimate is copied into a measured field.
- **Probe disambiguator:** D-B emits seg-vs-temporal, pose-vs-temporal, and fully armed combined views,
  preserving all defensible role partitions in one boundary observation.

## Artifact edges

- Build/read surface: `src/tac/witness_control/telemetry_producers.py`
- Trainer producer: `experiments/train_levelset_witness_realized_through_R_mlx.py`
- Resume registry: `src/tac/witness_control/resume_registry.py`
- DSL: `src/tac/witness_dsl/curriculum_dsl.py`
- Tests: `src/tac/witness_control/tests/test_telemetry_producers.py` and
  `src/tac/tests/test_spec_next_launch_all_levers_20260713.py`
- Full receipt: `experiments/results/next_launch_all_levers_ticket_20260713/da_db_preflight_receipt.json`
- Trimmed receipt:
  `experiments/results/next_launch_all_levers_ticket_trimmed_20260713/da_db_preflight_receipt.json`
- Full narrative: `.omx/research/da_db_producers_20260713.md`
- Pool staging/readback: `.omx/research/da_db_producers_candidate_rows_20260713.jsonl`

