# GO PACKET — guarded K=2 exact-costate reuse in-loop A/B

**Status:** `OPERATOR-GO REQUIRED` — governed training/timing launch; no launch was made in this
landing.  
**Lane:** `lane_p0_backward_closer_20260713`  
**Cached fidelity authority:** `PENDING_SEALED_N600_RECEIPT`  
**Purpose:** convert exact-call and diagnostic teacher-slice accounting into real in-loop component
and whole-step wall clock without weakening the full-facet or resume contracts.

## Why a GO remains required

The cached n600 replay can decide whether one stale costate is a safe changed formulation and can
derive exact calls saved. It cannot establish current trainer wall time. The prior `f=0.1784755863`
forward share and `82.15%` backward share came from a diagnostic harness whose total cost was about
12x the live path; the sparse flagship observed the opposite component ordering in another
diagnostic substrate. Neither ratio has in-loop authority.

The hot trainer also lacks a reviewed current-costate provider seam. The landed DSL deliberately
compiles no invented flag and refuses live activation. Operator GO authorizes a bounded main-owned
integration and paired treatment; it does not authorize bypassing the governor or running directly
from an argv invented in this packet.

The separate D-A component-timer surface is now **ONE-GO-READY** (`163/163` parser/DSL/telemetry
checks in the latest main landing). That closes generic timer availability, not K2 actuation: it
still cannot expose, persist, restore, or reuse the current exact costate without the prerequisite
provider patch below.

## Prerequisite landing — main-owned hot-file patch

Before launch, main must review and serializer-land a minimal patch that:

1. exposes the exact input-costate payload already produced by the frozen teacher backward;
2. atomically persists payload bytes, SHA-256, exact anchor metrics, step/event/stage identity, and
   `ControllerState` beside every periodic and stage-end checkpoint;
3. restores and SHA-verifies that payload before resume; missing/tampered bytes force full refresh;
4. calls the pure K2 controller for **one** changed-frame attempt only, then requires a new anchor;
5. performs the exact forward-only CE/d_seg/d_pose guard before optimizer admission; failure restores
   the byte-exact candidate/optimizer state and performs a full exact teacher refresh;
6. forces refresh on event, stage, scorer/objective, control-scope, or payload custody change;
7. emits monotonic `_teacher_fwd`, `_teacher_bwd`, guard, rollback, exact-call, full-step, and
   checkpoint I/O timers plus accepted/fallback counters;
8. registers every additive state field in the canonical resume registry with a legacy default that
   selects the full teacher; and
9. compiles only through the typed DSL after the provider exists. The current argv-inert lever is a
   refusal surface, not permission to invent `--costate-reuse-*` flags.

The shared trainer, resume registry, and curriculum DSL are hot. Use serializer patch-file/SHA guards,
never absorb sibling state, and rerun the three-clean-pass seal after reconciliation.

## Paired governed treatment

After the prerequisite commit, compile two tickets from the same sealed base spec, seed, source
checkpoint, n-pair treatment, stage schedule, hardware, and scorer/objective custody:

- **A — exact baseline:** provider enabled for telemetry, reuse disabled; exact backward every step.
- **B — guarded K2:** same provider and telemetry; event-controlled Kmax=2 policy enabled with the
  fixed full-facet guard and full-teacher fallback.

The typed tickets, not this prose, own exact argv. Launch both only through the governed launcher
after storage waterfall and memory preflight. Preserve all periodic and stage-end checkpoints and
keep run directories sacred. Start with the cheapest governed timing treatment that exercises the
real loop; extend only if warmup/variance diagnostics say the paired estimate is unresolved. No score
claim is needed, and no contest evaluator dispatch is part of this packet.

## Required emitted receipt

The paired timer receipt must bind:

- post-edit git/source/config/argv hashes, seed, exact source checkpoint, objective/scorer hashes,
  hardware and evidence axis;
- resumed and uninterrupted path parity, including anchor payload and controller-state hashes;
- per-step and aggregate exact backward call counts, forward-only guard counts, accepted reuse,
  fallback refresh, rollbacks, forced boundaries, and failed-custody refusals;
- median, dispersion, warmup exclusion, and paired delta for `_teacher_fwd`, `_teacher_bwd`, guard,
  checkpoint I/O, full teacher slice, and full optimizer step;
- `d_seg`, `d_pose`, CE, rate, island birth, and pose-need facets at every stage boundary; and
- explicit labels: `MEASURED_IN_LOOP`, `DERIVED_FROM_MEASURED`, or `UNKNOWN`.

## Admission / stop rules

`FIRE` requires all of:

1. cached n600 receipt admitted and its expected hash is code-reviewed/pinned;
2. every in-loop accepted reuse satisfies the exact full-facet guard; every rejection is charged;
3. crash-resume reproduces treatment state and loses at most one intra-stage checkpoint interval;
4. no stage/event/custody boundary reuses an old anchor;
5. measured exact-backward call reduction agrees with `p/2` within accounting tolerance;
6. measured full-step wall clock improves with paired uncertainty excluding zero; and
7. no holistic facet regresses at stage boundary.

Otherwise `REFUSE` and retain the exact full teacher. A timing miss kills only the current provider,
guard, hardware, and treatment formulation. It does not reopen K>2, blind cadence, checkpoint/adjoint-
ODE/reversible mechanisms, or the sparse formulation already closed by separate evidence.

## What the operator authorizes

Reply `GO P0-K2 TIMER` to authorize the prerequisite hot-file integration plus the bounded paired
governed timing treatment. Main then lands the patch, compiles typed tickets, claims the dispatch
lane, runs storage/memory/governor preflights, launches, harvests the receipt, and updates the
canonical equation/DAG/posterior. Until that explicit GO, the measured cached-data landing remains
default-off and no training launch is permitted.

---

## 2026-07-14 CORRECTION / SUPERSESSION — FIDELITY_BLOCKED_FUTURE_TEMPLATE

The 2026-07-13 packet above is retained byte-for-byte as **HISTORICAL_PROVENANCE**. Its pending
receipt premise and `GO P0-K2 TIMER` solicitation are superseded. The shared DAG feed is **NOW
LANDED** as `FEED-p0-backward-closer-20260713`.

**Status:** `FIDELITY_BLOCKED_FUTURE_TEMPLATE`. **No current GO is requested or authorized.**
**verdict_scope=FORMULATION:** the blocked formulation is direct raw-input-costate zero-order-hold
K2 with strict all-accepted stale-minus-exact `d_seg<=0`, three V9 checkpoint states (200 rows each),
and `[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]`. It closed as
`K2_CORRECTED_NOT_ADMITTED_DEFAULT_OFF`; the family remains intact under a newly preregistered
guard/tolerance formulation.

The sealed breakdown is 456 accepts, 67 actual guard fallbacks, and 77 terminal/blocked rows;
total nonaccept is `144=67+77`, not 144 guard fallbacks. Only `308/456` accepted rows met the strict
`d_seg` regret rule, so a timer cannot repair or waive the fidelity gate. Baseline cost is `2.0` and
the diagnostic guarded cost is `2+alpha-p=1.4184755862999998`. The `1.6129032258064517x` exact-
backward factor and `0.38` reduction are separate **DERIVED COUNTERFACTUALS behind the failed gate**.
Admitted K2 factor is `1.0x`; admitted sparse factor is `1.0x`; admitted bulk reduction is `0%`;
whole-epoch effect is `UNKNOWN`.

Commit `e59f69a79cb2d974ec29fcaf75c6c855bd782a7a` and
`.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json` provide a **MEASURED** n96
`0.621 s/pair` on `[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE`; `6.21 min/n600`
is a **DERIVED linear extrapolation**. That forward-only authority-verdict result makes this timer
diagnostic rather than the global bottleneck authority.

This template becomes actionable only after a distinct formulation/provider is preregistered and
fidelity-admitted by a new reviewed receipt. Then, and only then, an operator may authorize a
bounded diagnostic in-loop timer under the original resumability, per-stage-checkpoint, custody,
governor, and holistic-facet rules. The present K2 formulation remains default-off and cannot be
activated by timing evidence.
