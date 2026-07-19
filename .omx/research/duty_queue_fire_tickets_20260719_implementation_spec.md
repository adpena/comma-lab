# Duty-Queue Fire Tickets — Implementation Specification

Date: 2026-07-19

Lane: `duty_queue_fire_tickets_20260719`

Authority: delegated prompt SHA-256 `c8718f7d30b38d639d635b53578b4271a37986205ac2da0c1d7bc7fbd3cc6aa0`

Execution authority: **NEVER LAUNCH**; build, inspect, and dry-verify only.

## Objective

Materialize four ordered, durable measurement-ticket packages:

1. `DsegAwareTaper` A/B (`78.9%` duty share)
2. `HorizonWeightedMargin` A/B (`47.3%` duty share)
3. `StepNativeActivation` A/B (`34.2%` duty share)
4. Catalog #497 curvelet matched-bytes A/B audit

The contest-CPU pointer remains `0.1910828242` and is not moved by this lane.

## Outputs

- `.omx/research/duty_queue_fire_tickets_20260719_codex.md`
- `.omx/research/duty_queue_fire_tickets_20260719/01_dseg_aware_taper/`
- `.omx/research/duty_queue_fire_tickets_20260719/02_horizon_weighted_margin/`
- `.omx/research/duty_queue_fire_tickets_20260719/03_step_native_activation/`
- `.omx/research/duty_queue_fire_tickets_20260719/04_curvelet_matched_bytes_p0_497/`

Each ticket directory must contain:

- `launch.sh`
- a compiled-config or audited-config artifact
- provenance with hashes and custody facts
- a no-training dry-start receipt
- a verdict card
- a per-ticket confound self-audit

## Ticket 1–3 contract

Each signed control-to-treatment contrast is expressed through the typed
`WitnessProgram` plus exactly one named `Lever` declaration. For
`DsegAwareTaper`, the declaration belongs to the ON control and the canonical
treatment removes the complete Lever; `HorizonWeightedMargin` adds a treatment
Lever; `StepNativeActivation` replaces the treatment endpoint. The paired OFF
twin must be compiled from the same warm-start checkpoint, #518 recipe, seed,
n600 ordering, observer cadence, and checkpoint cadence. A pre-existing banked
c2 control is not admissible unless byte-level evidence proves it used the
identical #518 recipe; absent that proof, the paired OFF twin is mandatory.

The warm start is the v9c2 BEST EMA at epoch 725, reported d_seg
`0.003457972208658854`, loaded weights-only into a fresh optimizer. The package
must record the checkpoint path, byte count, and SHA-256 without modifying the
sacred source run.

The compiled candidate must account for every #518 geometry obligation:

- widened LR-rewarm window derived from beta2 law;
- beta2 warmup `LawRef`;
- fork head-solve;
- a measured/provenance-backed margin trust-region cap;
- v0 schedule positioning;
- resume-event reanchor.

No ticket is fireable if any obligation lacks executable provenance. In
particular, a configured float or prose-only citation is not a derived
`MarginStepCap` value. Missing geometry, response-time, NCDE/costate-noise, or
power evidence must produce an explicit `BLOCKED` ticket, never an invented
constant or permissive launch script.

The measurement window and FIRED-PAYS / NEUTRAL / HURTS slope thresholds must
be derived from the beta2 relaxation, lever response time, predicted term share,
and measured NCDE/costate noise. If those quantities cannot be joined from
authoritative artifacts, the verdict card records the exact missing edge and
remains non-authorizing.

Every ticket must additionally verify:

- n600 memory projection and freshness rc6;
- resumability and preserved end-of-stage checkpoints;
- `safe_run` containment;
- liveness (`ep_loss > 0`), spike guard, and positive-control sentinel;
- treatment activity through `lever_engage` and term-share telemetry;
- live-vs-EMA verdict source, with EMA-lag handling;
- resume-event reanchor;
- identical RNG and data order across the paired arms;
- observer and checkpoint cadence equality, avoiding a 27-min/epoch
  `--ckpt-every` confound.

The no-training dry-start receipt may be GREEN only for the bounded properties
it actually checks (imports, typed compilation, hash gates, custody, and
fail-closed behavior). It must not imply launch readiness when a blocker exists.

## Ticket 4 contract

Audit the existing `tools/fire_curvelet_matched_bytes_ab_p0_497.py` rather than
recomposing it. Verify:

- `--arm control` and `--arm treatment` behavior;
- import closure;
- checkpoint/input/output paths;
- compiled-config freshness;
- refusal before training in this delegated lane.

Materialize an audit receipt and verdict card. Any stale path, missing import,
or unmatched arm contract is a named blocker.

## Hash and governor gates

All materialized executable surfaces must carry their post-edit SHA-256 and a
hash of the typed compiled payload. Launch scripts are non-authorizing in this
lane and must fail closed on unresolved prerequisites. Catalog #506 launcher /
governor hashes must be checked against repository truth rather than copied
from historical prose.

## Verification and landing

1. Run static import/compile and focused tests only; do not start training.
2. Run each ticket's dry-start/refusal surface and preserve stdout/stderr in the
   receipt.
3. Obtain independent fresh-eyes review of all four packages, including the
   per-ticket confound audit.
4. Commit through `tools/subagent_commit_serializer.py` with expected post-edit
   SHA-256 values.
5. Report the resulting commit for mandatory MAIN landing review. Do not claim
   that the worktree commit is on MAIN.

## Stop conditions

- No paid dispatch, GPU actuation, evaluator run, training epoch, or pointer
  update.
- No modification of the v9c2 run or bank.
- No invented flags or direct argv-only lever wiring.
- No launch-ready verdict while a #518, custody, freshness, confound, or power
  prerequisite remains unresolved.
