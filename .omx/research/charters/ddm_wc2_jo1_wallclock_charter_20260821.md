# ddm_wc2_jo1_wallclock — measured ETA instrument + determinism-preserving wall-clock levers for the jo1 critical path

## MISSION (operator 2026-08-21: "Wall clock optimization is always on the table too")
The jo1 r8 solve is THE campaign critical path: the entire end-state route (es1 portrait,
nr1 body-rebase #1187, the 9-stage composed route) is serialized behind its endpoint.
Projection 21.69–36.22 h; LIVE measured state at charter time: worker pid 14801 at 238%
CPU / 5.9 GB RSS, resumed from target_birth step 600, full 600/600 frame-0 materialization
completed in ≤78 min — but the RECEIVER WORKLOAD (jo5 LIVE-HYPOTHESIS: "the complete
receiver workload remains unexecuted") is the unmeasured majority. Deliver: (A) a
read-only ETA instrument on the live run, (B) a measured step-cost profile, (C) built
determinism-preserving speedup levers with swap economics, (D) wall budgets for the es1
route stages. DO NOT TOUCH the live run (pid 14801, run dir sacred). Bounded offline
probes on COPIES only; no heavy launch; MAIN owns any swap decision.

## WORK ORDER
1. ETA INSTRUMENT (read-only, ships first): a small tool reading the run dir's artifact
   timestamps + cursors (RESUME_LATEST.json, FX5_FRAME0_CURSOR.json, stages/*/receipts)
   → per-phase measured rates → live endpoint ETA band, re-derivable on demand. Wire a
   one-line summary MAIN can invoke. This converts the 21.7–36.2 h band into a live curve.
2. STEP-COST PROFILE: on a COPY of 1–3 pairs offline (never the live process), decompose
   the 1.3895 s/step (r6 receipt) into {PoseNet forward · Schur/compensation · carrier
   solve · coder race · retention/cert IO}. The dominant term names the lever.
3. LEVERS (build to READY, fire = MAIN): candidates in priority order, each adjudicated
   against the jo5 DETERMINISM LAW (batch SHAPE is part of accumulation identity —
   the cure replays the exact exploration batch; any lever must preserve PER-PAIR batch
   integrity and the cert regeneration tuples):
   a. PAIR-LEVEL PROCESS PARALLELISM — pairs are independent within a stage; N workers
      each owning whole pairs preserves every pair's batch shape exactly. Expected ≈
      min(N, perf-cores)× on the dominant phase. Requires: deterministic merge order,
      per-worker retention certs, re-proof (3 pairs × 3 repeats byte-identity), r9 reseal.
   b. THREAD/BLAS TUNING within the pinned mechanism — free ONLY if batch shapes and the
      seal config are untouched; a changed thread count is a config change → reseal.
   c. MLX/Metal port of the heavy forwards — runtime-lift grant stands, but adjudicate
      HONESTLY against the L70 bit-identity wall: the determinism gate + regeneration
      tuples must reproduce on the SAME substrate; if MLX-GPU cannot guarantee it,
      say so and stop (a lever that breaks the gate is not a lever). ONE Metal fire max
      (governor law); Metal controls are MAIN-fire-only (#999).
4. SWAP ECONOMICS (the binding deliverable): reseal-r9-hot-swap fires ONLY if
   measured-ETA(remaining, r8) − [build+re-proof+reseal+resume cost] − measured-ETA(r9)
   ≥ 2× the swap cost — arithmetic on MEASURED rates from (1)+(2), never projections
   (#1087: a smoke became a 4.9×-wrong cost model). If the live ETA tracks the low end,
   the honest recommendation may be RIDE r8 — say so plainly. r9 resumes from r8's
   latest checkpoint (resumability P0 — no restart).
5. ROUTE WALL BUDGETS: one line per es1 stage (memo ddm_es1_end_state_characterization_20260821.md)
   with a derived wall budget + its dominant cost driver — so the route is wall-clock-
   aware from birth (min-wall-clock law m33).

## OPTIMAL FORM
Family reference form + receipt: the wall-clock family reference is the #509 epoch-time
burn-down campaign (measured per-lever wall attribution, telemetry-first) + the r6 probe
receipt (1.3895 s/step, MEMORY_PREFLIGHT.json 09e5affa lineage) + jo5's measured
determinism proof (9/9, memo .omx/research/ddm_jo5_determinism_cure_reseal_20260821.md).
Provenance pin: experiments/ddm_jo3_joint_objective_entrypoint.py=766d3494751b27343df8904db2b74fd21e3d7804274a7e3931316ae11736bcdd
SCOPE reductions (legal): 1–3 pair offline profiles on copies. MECHANISM reductions:
NONE — a lever that changes the solve mechanism or loosens the determinism gate is the
fake this family refuses; parallelism must preserve per-pair batch identity EXACTLY.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- jo5 batch-shape law (memo above): singleton ≠ batch even single-threaded — any
  restructuring of WHAT is batched changes identity; only WHO computes whole pairs is free.
- L70 MLX-GPU bit-identity wall (#348 lineage): GPU accumulation nondeterministic without
  fixed-order reduction — the MLX candidate inherits this burden fully.
- 1-thread 2.96× law (config †D): single-threading is deterministic but 2.96× slower —
  not a speedup lever; recorded so it is not re-proposed.
- #1087 cost-model trap: a 50-step smoke priced windows 4.9× wrong — swap economics must
  use rates measured from the LIVE run's own artifacts, never probe extrapolation alone.
- first_attempt_wall_clock_is_not_a_family_verdict (memory): a slow first profile of a
  lever is not a family kill; tune before verdict.
- Concurrent Metal fires OOMed the machine (memory 20260806): ONE Metal fire, governed.

## CONTEXT ANCHORS (memo-associated)
- Campaign #1182 sub-0.12; live run dir (SACRED):
  experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/
- nr1 fire-order #1187 (fires at the r8/r9 endpoint — the consumer of every hour saved).
- es1 route memo .omx/research/ddm_es1_end_state_characterization_20260821.md.

## CONTRACT
upstream/ READ-ONLY; live run untouched; serializer commits; .py = 2 genuine review
passes; memo .omx/research/ddm_wc2_jo1_wallclock_20260821.md; final message = live ETA
band + step-cost table + per-lever verdict {READY/REFUTED/RIDE-R8} + the exact swap
arithmetic + GESTALT-DELTA line.
