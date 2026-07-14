# OPERATOR-GO packet — whole-teacher distilled student in-loop integration

**Status:** `NOT_FIRED`; `OPERATOR-GO REQUIRED`  
**Lane:** `lane_whole_teacher_distilled_student_20260713`  
**Purpose:** convert a sealed cached-data student receipt into a governed,
matched-window training-gradient and throughput treatment. This packet is not
launch authority by itself.

## Fail-closed prerequisites

Main must refuse the patch and launch until all of the following are true:

1. A real n600 receipt authenticates exactly 600 unique V9 replay assignments,
   the rendered frames, centered teacher quotients, labels, and full exact
   teacher input VJPs.
2. The decisive heldout/n600 VJP thresholds pass; a forward-only pass is
   insufficient.
3. NumPy-fp32 reference parity passes and MLX measurements come from a
   Metal-visible process. MPS/MLX remains local advisory, never a score axis.
4. At least one fully charged training-gradient economics row pays after
   student forward+VJP, exact teacher anchor, and update cost are included.
5. The source receipt, student parameter bytes, policy, equations, and harness
   are serializer-landed and content-hash pinned.
6. The current forward/incremental-VJP production split is either measured by
   the held D-A packet or kept explicitly `UNKNOWN`; diagnostic 82% is not
   promoted to in-loop authority.

The current landing does **not** satisfy items 1-5. Consequently this packet is
one-GO-ready as an integration specification, not fire-ready as a run ticket.

## Preregistered admission budget for the first governed treatment

The first candidate is **DERIVED/CONDITIONAL**, not measured: training-gradient
tier, `small` (468 parameters), `K_student=64`. The diagnostic-slice inclusive-
95 budget requires
`C_student,VJP + U/64 <= 103.437 ms/pair`; strict pay alone permits
`< 2962.053 ms/pair`. Those limits come from the inherited
`C_teacher,VJP=3009.069611 ms/pair` diagnostic and must be recomputed from the
current in-loop timer before fire if that authority changes. A positive point
estimate is insufficient: the confidence interval for fully charged whole-step
improvement must remain above zero.

The same candidate has **DERIVED lower bounds**, not measured peak memory:
1,872 parameter bytes; 7,488 bytes for parameters + Adam `m/v` + preserved best
parameters; 9,360 bytes with one parameter-gradient buffer; and 36.75 MiB for
one activation copy. The admission preflight must measure Metal peak including
autograd/VJP workspaces, allocator fragmentation, renderer, exact-anchor
tensors, and checkpoint staging. Exceeding the admitted budget refuses or
selects a preregistered smaller size; it never silently degrades checkpoint
custody.

## Main-owned hot-file integration

After prerequisites close, main owns a minimal serializer `--patch-file`
landing against the then-current trainer SHA. It must:

1. add a typed provider seam at the actual `adapter.segnet(f1)` loss call in
   `make_loss_fn`, preserving the full exact teacher as the default/fallback;
2. select the student only from the typed DSL—never an invented flag—and keep
   the OFF path byte-identical;
3. compute the scalar CE/active loss on the student's five-class zero-sum
   quotient representative and obtain one differentiable input backward;
4. force an exact teacher anchor at the typed cadence and whenever stage,
   event, scorer/objective, render/R, source, policy, or trust-region custody
   changes;
5. type the student's exact-anchor cadence separately from #487's
   event-sensitive stale-costate `K_max=2` cap when the controllers compose;
   no blind cadence reuse or inherited speed credit is allowed;
6. compare exact-anchor value, argmax/margins, full input VJP, renderer-gradient
   direction, exact CE descent, d_seg, d_pose, rate, island birth, class-anchor,
   and pose-need facets; any failure falls back to the exact teacher;
7. atomically persist student parameters/optimizer, exact-anchor tensors and
   hashes, controller state, stage/epoch, policy/source hashes, and fallback
   counters in every periodic and stage-end checkpoint, preserving all stages;
8. restore and SHA-verify every additive field before resume; missing legacy
   fields select the exact teacher, never a partly initialized student;
9. emit monotonic component timers for student forward, student input VJP,
   anchor teacher forward, anchor teacher VJP, anchor update, renderer VJP,
   checkpoint I/O, and whole step; and
10. register additive state in the canonical resume registry and complete the
    three-clean-pass seal after reconciling all hot files.

## Governed matched treatment

Compile same-SHA typed tickets only after the integration commit:

- **A — exact teacher:** provider telemetry enabled; student activation OFF.
- **B — student:** same seed/source/stage/objective/scorer/hardware, student ON,
  exact-anchor fallback ON and the optional sibling stale-costate controller
  typed independently at `K_max=2`; its cap does not constrain the student's
  exact-anchor cadence and transfers no speed credit.

Run the cheapest real-state matched window that exercises refresh and resume,
then extend only if timing uncertainty overlaps zero. Launch sequentially via
the governed launcher after lane claim, storage waterfall, memory admission,
and resume dry-run. Preserve every periodic/stage checkpoint and sacred run
directory. No paid/evaluator dispatch is part of this GO.

## Future offline architecture rung — not part of this GO

Operator inbox directive `2026-07-14T00:23:02Z` adds a recurrent-local/foveated
whole-teacher student as a named **matched offline rung**. It is `NOT_BUILT` and
`NOT_FIRED`; it does not reopen the closed single-shot fixed-feature localizer
and carries no visual-generalization claim. Its only admissible questions are
whether sequential local aggregation improves real n600 quotient/full-VJP
fidelity and whether total recurrent wall time still pays.

Run it only after the baseline raw-cache/storage/Metal gates close, at matched
target, parameters/teacher calls, exact-anchor cadence, split, and custody. A
separate typed recurrence/glimpse policy and probe-disambiguator are owed if
two or more recurrence interpretations survive. It must clear the same
fully-charged economics and decisive VJP gates before it can enter a later
in-loop GO packet.

## Required receipt and stop rules

The receipt binds git/config/argv/source/scorer/R/student/policy hashes, seed,
hardware/axis, uninterrupted-versus-resumed parity, per-component timers,
anchor/update/fallback counts, all fidelity metrics, and holistic facets.

`FIRE` requires a statistically positive fully charged whole-step improvement,
all VJP/full-facet gates, deterministic resume parity, and zero custody
refusals. Otherwise `REFUSE` and retain the exact teacher. A negative scopes
only the measured student size/cadence/tier/hardware/objective. It does not
reopen the closed feature-localizer family and does not close whole-teacher
distillation.

## Explicit authorization string

After a sealed n600 receipt passes, reply `GO WHOLE-TEACHER-STUDENT TIMER` to
authorize the main-owned hot-file patch and bounded matched timing treatment.
Until then the provider remains default-OFF and no training loop is fired.
