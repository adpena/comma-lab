# Whole-teacher distilled student — cached-only build and fail-closed measurement

**Date:** 2026-07-13  
**Lane:** `lane_whole_teacher_distilled_student_20260713`  
**Mode:** `$0` local cached-only build/fit/measure; `research_only=true`  
**Authority:** NumPy-fp32 reference; `[macOS-MLX advisory]`; no score authority  
**Pointer delta:** unchanged

> **MEANS caveat:** this student is throughput apparatus for replacing a
> frozen SegNet teacher slice. It is not the pointer, not a score claim, and
> not promotion evidence. Only a receiver-closed, byte-closed exact evaluator
> row on the exact archive bytes can move the score or frontier pointer.

> **One-line scoped verdict:** `{STUDENT-PAYS-at-(tier,K,size)=NONE-DEMONSTRATED
> (n=0, not a no-pay family verdict) · VJP-FIDELITY-GATE=FAIL-CLOSED/
> UNMEASURED_BLOCKED (not empirical FAIL) · NO-GO-scope=INSTANCE x INPUT-CACHE
> + INSTANCE x STORAGE-PREFLIGHT + INSTANCE x CURRENT-HOST; whole-teacher
> family remains OPEN}`.

## Executive result

The differentiable whole-teacher student, NumPy/Torch/MLX parity surfaces,
strict real-n600 cache contract, cached-only fit/measure harness, default-OFF
DSL, canonical equation, DAG feed, storage waterfall, and governed integration
packet were built. The fit and measurement were correctly refused before any
optimizer step because the required raw rendered-state n600 bundle does not
exist in current cache custody. The current managed process also cannot
evaluate an MLX array on Metal, and the required SSD workload tier is not
writable from this process.

**MEASURED preflight:** `n_pairs=0`, `fit_steps=0`, `teacher_calls=0`,
`bulk_output_root_touched=false`. Therefore the requested forward worst-pair
QoI, exact-teacher VJP cosine/relative-L2, student/update timings, and paying
economics cells are `UNMEASURED_BLOCKED`. They are not zeros, and the decisive
VJP gate is fail-closed rather than empirically failed.

**Does any operating point pay? `N — not demonstrated`.** This statement has
`verdict_scope=INSTANCE x INPUT-CACHE x CURRENT-HOST x STORAGE-PREFLIGHT`; it
does not say that no whole-teacher student can pay. Diagnostic-slice algebra
shows that `K=20` cannot achieve an inclusive 95% kill with any positive
student/update cost, while `K in {32,64,128}` has positive algebraic headroom.
No cadence or size earns `STUDENT_PAYS` until its own measured cost and tier
fidelity gate pass.

## Why this is not the closed feature-localizer family

The cheap pre-SE feature-localizer family is already family-closed under its
recorded req-R. RFF/prefix, support ranking, deeper/nonlinear, single-source
tileability, and cheap-global/multi-source reopenings all failed the retained
boundary-mass bar. Those experiments asked whether a fixed cheap feature
source already contains enough boundary localization.

**MEASURED-INHERITED, not rerun:** the recorded best retained-mass values span
`0.11–0.32`, below the preregistered `0.47` bar across those closed rounds.

This arm asks a different structural question. It maps the realized-through-R
rendered RGB frame directly to the full frozen teacher's four-dimensional
centered-logit decision quotient and learns the scalar-composed input VJP. It
has no fixed feature-source admission premise and no tileability requirement.
The two walls that closed the localizer are therefore inapplicable. No closed
round was rerun, and this blocked receipt does not close the student family.

## Target and decisive gate

The five-class teacher logits `u_T` are projected into the softmax/argmax
quotient with the fixed orthonormal Helmert basis `H in R^(5x4)`:

```text
q_T = H^T (u_T - mean_class(u_T))
u_hat_S = H q_S
g_S = d CE(u_hat_S, y) / d x
```

The student value target is `q_T`, not hard labels or an unscoped raw costate.
Penultimate features remain a matched comparator, not the default. Value
matching alone cannot control the RGB Jacobian; the binding training-gradient
gate is the full-vector comparison between `g_S` and the exact teacher costate
`g_T` on every real n600 state.

The primary forward and VJP fidelity rows compare the exact teacher tensors to
the **NumPy-fp32 student reference**. MLX outputs cannot move either primary
gate. MLX is retained only for separately named NumPy/MLX parity, matched-device
student/update timing, and deterministic-repeat custody on the advisory host.

The preregistered thresholds are **ASSUMED_AWAITING_VERIFICATION**, not tuned
results:

| Gate | Preregistered threshold | Actual n600 result |
|---|---:|---|
| worst-pair forward quotient cosine | `>= 0.995` | `UNMEASURED_BLOCKED`, `n=0` |
| worst-pair forward quotient relative-L2 | `<= 0.05` | `UNMEASURED_BLOCKED`, `n=0` |
| worst-pair argmax disagreement | `<= 0.005` | `UNMEASURED_BLOCKED`, `n=0` |
| worst-pair full input-VJP cosine | `>= 0.95` | `UNMEASURED_BLOCKED`, `n=0` |
| worst-pair full input-VJP relative-L2 | `<= 0.25` | `UNMEASURED_BLOCKED`, `n=0` |
| worst-pair NumPy/MLX forward cosine | `>= 0.9997` | structural/unit surfaces only; no real n600 MLX row |
| worst-pair NumPy/MLX input-VJP cosine | `>= 0.9997` | structural/unit surfaces only; no real n600 MLX row |

**VJP-FIDELITY-GATE: FAIL-CLOSED / UNMEASURED_BLOCKED.** The word “fail” here
means the activation guard refused missing evidence. It is not an empirical
fidelity failure and cannot support a family-negative verdict.

## Student and harness built

The student is a deterministic coordinate-conditioned depthwise-separable
convolutional map from `(1,3,384,512)` realized RGB to `(1,4,384,512)` quotient
coordinates. The layout is explicit and framework-independent:

All parameter and byte counts in this table are **DERIVED by deterministic
structural enumeration/serialization**; they are not runtime-memory
measurements.

| Size | Widths | Parameters | Raw fp32 parameter bytes | Deterministic serialized blob bytes |
|---|---:|---:|---:|---:|
| `tiny` | `8,8` | 236 | 944 | 2,373 |
| `small` | `12,16` | 468 | 1,872 | 3,312 |
| `medium` | `16,24` | 764 | 3,056 | 4,498 |

For the eventual in-loop path, persistent fp32 student custody
`parameters + Adam m + Adam v + best parameters` is **DERIVED** as four times
the raw parameter bytes: 3,776 / 7,488 / 12,224 bytes for tiny/small/medium.
One additional parameter-gradient buffer raises that to 4,720 / 9,360 /
15,280 bytes. The one-copy activation lower bound
`4*H*W*(9 + 2*w1 + w2)` is 24.75 / 36.75 / 48.75 MiB. This excludes autograd
retention, VJP transform workspaces, allocator fragmentation, and the renderer;
actual Metal peak is `UNMEASURED_BLOCKED` and remains a governed memory-admission
gate. No lower-bound number is mislabeled as peak memory.

The implementation includes:

- deterministic NumPy-fp32 initialization, forward, analytic input VJP,
  quotient lift, CE costate, serialization, and worst-pair aggregation;
- lazy Torch-CPU formula/parity helpers that are never relabeled MLX or score;
- lazy differentiable MLX forward and input-VJP surfaces;
- a strict manifest validator for exactly 600 unique real V9 assignments over
  `{ep150,ep251,ep275}`, with rendered frame, quotient, labels, full exact
  teacher costate, per-tensor dtype/shape/bytes/SHA-256, and sealed 480/120
  split;
- a cached-only MLX fit driver with deterministic boundary-mask training,
  full-vector n600 VJP measurement, typed optimizer state, hash-bound resume,
  atomic periodic and distinct stage checkpoints, and no teacher-call path;
- fully charged timing/economics fields and an exact-teacher fallback contract;
- storage preflight and success-only scratch cleanup with no `/tmp` evidence.

The four-neighbour boundary mask is derived deterministically from cached
labels and is used for fit terms. The decisive heldout/cohort VJP is still
computed on the complete frame-shaped vector; boundary-only VJP is diagnostic
and cannot substitute for it.

## The input-custody premise was falsified

The launch snapshot said reusable raw n600 rendered quotients and exact
costates already existed in the prior round caches. Artifact bytes say
otherwise:

| Artifact | Current fact | SHA-256 |
|---|---|---|
| Round 3 measurement receipt | `raw_training_costates_preserved=false` | `83704e64d1e5a70c00cf96c19330ff8453459e1024f957bceb48f99972157d75` |
| Round 4 receipt | `raw_exact_costates_preserved=false` | `6ccbf0e10691dc39c94b77aaefdfe7d9ac3a38b32962bfa5eefcb1107f627222` |
| Round 5 receipt | `raw_exact_costates_preserved=false` | `38033922bd39cb48f72a154ddd622c41b18be0f137ede56fe4c76873e7bfe98f` |
| sparse-adjoint measurement | full-grid arrays were rebuildable scratch, not retained training custody | `52a22f4b60367fc27ca0fca7293b0741da4b809724479cd3ef7e92291c250cef` |
| sparse-adjoint cleanup manifest | 120 arrays / 283,130,880 bytes certified and success-cleaned | `c55a59d64fbb3ab8cd30bad517438849f07bb1017cbb73a62cd8a0ace17fad75` |

Per-pair hashes, compact support/mass statistics, and source-video SSD logits
cannot be substituted for raw V9 rendered frames and frame-shaped teacher
costates. Even self-consistent tensor hashes are insufficient without
content-bound actual-R, post-R 0..255 differentiation units, frozen teacher
source/weights, scalar objective/reduction, Helmert quotient basis, and
renderer/source custody. Deterministic teacher regeneration is technically
possible but is a different, separately governed producer action; cached-only
containment and the absent live-inbox authorization prohibited it here.

The quant-tail reliability lane likewise records that its official heldout raw
features/arrays were not preserved. No ridge-reliability number is borrowed.
The student is dense depthwise-separable rather than sparse, so no sparse-adjoint
kernel speed or fidelity credit is inherited.

## Preflight blocker classification

| Surface | MEASURED state | `verdict_scope` | `req-R` |
|---|---|---|---|
| input cache | manifest absent; no raw n600 tensor bundle; zero teacher calls attempted | `INSTANCE x INPUT-CACHE` | seal exactly 600 real rendered-state frame/quotient/label/full-input-VJP rows plus R/post-R-units, teacher source/weights, objective/reduction, quotient-basis, renderer/source, and timing-receipt semantic custody, or separately authorize deterministic reconstruction |
| storage | Vertigo tier has capacity but is not writable in this managed process; APDataStore absent; local fallback not authorized | `INSTANCE x STORAGE-PREFLIGHT` | select a writable SSD workload root with the requested reserve |
| backend | MLX `0.31.2` imports and reports Metal, but an isolated evaluated-array canary fails `RuntimeError: [metal::load_device] No Metal device available` | `INSTANCE x CURRENT-HOST` | rerun on a process where device-info plus evaluated MLX array prove Metal usable |

No bulk task output directory was created. The durable receipt contains the
exact source hashes, policy, blocker-specific scopes, and blocker-specific
req-R. Its final SHA-256 is recorded in the verification section after source
sealing.

## Economics — measured inputs versus derived feasibility

The only teacher times used here are **MEASURED-INHERITED diagnostic-harness
values**, not current in-loop authority. Their direct source custody is
`.omx/research/onpolicy_surrogate_95kill_20260713.md`
(`SHA-256=78609829dc10204155f8be547555c86b2b610aaef2db249fea2d0290b77d6a90`),
cross-referenced by `.omx/research/per_epoch_detailed_accounting_20260713.md`
(`SHA-256=967d68a46827d5674cc3f4a723457930b860a46ecdae27cc85b561cb4195def2`):

- forward-advisory teacher: `C_T,fwd = 537.045463 ms/pair` (`n=9`);
- training-gradient teacher forward+input-backward:
  `C_T,VJP = 3009.069611 ms/pair` (`n=6`).

The current production forward/incremental-VJP split remains unknown. The
real-n600 current vehicle is measured only at whole-epoch level; the held D-A
component treatment is one-GO-ready but was not fired. Therefore neither the
diagnostic 82% incremental-VJP ratio nor a conflicting forward-heavy replay
ratio is promoted here.

For tier `t`, student size `s`, and exact-anchor cadence `K_student`:

```text
C_bar_t(s,K) = C_student,t(s) + (C_teacher,t + U(s)) / K
STUDENT_PAYS iff tier gate passes and C_bar_t < C_teacher,t
INCLUSIVE_95 iff tier gate passes and C_bar_t <= 0.05 C_teacher,t
```

The sibling #487 `K_max=2` controls only its own event-sensitive stale-costate
attempt. It neither caps `K_student` nor transfers speed credit.

### Cadence headroom, diagnostic-slice conditional

`Pay headroom` is the maximum allowable `C_student + U/K` for any strict
teacher-slice improvement. `95% headroom` is the maximum for an inclusive 95%
kill. All values below are **DERIVED** from inherited diagnostic timings.

| Tier | K | exact-anchor floor ms | pay headroom ms | inclusive-95 headroom ms |
|---|---:|---:|---:|---:|
| forward | 2 | 268.523 | 268.523 | -241.670 |
| forward | 4 | 134.261 | 402.784 | -107.409 |
| forward | 8 | 67.131 | 469.915 | -40.278 |
| forward | 20 | 26.852 | 510.193 | 0.000 |
| forward | 32 | 16.783 | 520.263 | 10.070 |
| forward | 64 | 8.391 | 528.654 | 18.461 |
| forward | 128 | 4.196 | 532.850 | 22.657 |
| training-gradient | 2 | 1,504.535 | 1,504.535 | -1,354.081 |
| training-gradient | 4 | 752.267 | 2,256.802 | -601.814 |
| training-gradient | 8 | 376.134 | 2,632.936 | -225.680 |
| training-gradient | 20 | 150.453 | 2,858.616 | 0.000 |
| training-gradient | 32 | 94.033 | 2,915.036 | 56.420 |
| training-gradient | 64 | 47.017 | 2,962.053 | 103.437 |
| training-gradient | 128 | 23.508 | 2,985.561 | 126.945 |

Thus `K<=20` is algebraically impossible for an inclusive 95% kill when
student/update cost is positive. `K in {32,64,128}` is merely algebraically
eligible; it does not pay until measured costs fit the corresponding headroom.

### Required pays-at-(tier,K,size) table

| Tier | Size | Gate authority needed | Measured `C_student,U` | Measured paying K | Inclusive-95 candidate K | Disposition |
|---|---|---|---|---|---|---|
| forward-advisory | tiny | n600 worst-pair forward | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |
| forward-advisory | small | n600 worst-pair forward | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |
| forward-advisory | medium | n600 worst-pair forward | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |
| training-gradient | tiny | n600 full exact input-VJP decisive gate | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |
| training-gradient | small | n600 full exact input-VJP decisive gate | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |
| training-gradient | medium | n600 full exact input-VJP decisive gate | `UNMEASURED_BLOCKED` | none demonstrated | 32/64/128 conditional | `NO_PAY_AUTHORITY` |

The forward tier is useful for an advisory teacher-forward substitution only.
It cannot authorize witness training-gradient replacement. The decisive route
remains the training-gradient tier regardless of which side of the unresolved
current forward/backward split eventually binds.

## Corpus and organ disposition

No comma10k/openpilot datum was used and no corpus-generalization claim is
made. That corpus and the #431/#433/#434 regime model remain admissible only as
initialization or a weak typed prior. Any such initialization still returns to
the same real on-policy n600 value/VJP gates; it is never frozen authority.

## Later inbox architecture signal — recurrent-local matched rung

**Consumed operator directive `2026-07-14T00:23:02Z`.** The cited paper studies
visual tasks requiring sequential aggregation of local information and reports
that strictly local recurrent policies can avoid global shortcuts that harm
length/compositional generalization. That result does **not** transfer a
generalization claim to this one-video witness problem: memorizing a useful
global shortcut is legal here, and real n600 value/VJP fidelity plus charged
cost remain the only authorities. Primary source:
[Madan et al., arXiv 2607.09061](https://arxiv.org/abs/2607.09061).

The **SPECULATIVE** transferable hypothesis is narrower: recurrence may supply the sequential
aggregation missing from the already-closed **single-shot** localizer while
retaining small local kernels and differentiability. A recurrent-local,
boundary-annulus/foveated whole-teacher student is therefore
**WORTH-A-MATCHED-RUNG**, not adopted and not dominated a priori. It is
structurally a student variant, not a reopening of fixed pre-SE
feature-localizer rounds.

The rung remains `NOT_BUILT`, `NOT_FIRED`, and downstream of the same cache,
storage, and Metal gates. When unblocked, compare it to the global single-shot
student at matched quotient target, parameter/teacher-call budget, `K_student`,
real n600 split, and exact full-vector VJP gate. Report recurrent step count,
glimpse geometry, total sequential wall time, and memory. It survives only if
its measured fidelity/cost point improves the Pareto frontier; “generalizes
better” is neither a metric nor an admission reason here.

## One-GO-ready integration surface — not fired

The integration specification is prepared in
`operator_authorizations/GO_PACKET_whole_teacher_distilled_student_20260713.md`.
It is one-GO-ready at the seam/telemetry/checkpoint level, but not fire-ready
while the n600 receipt and a paying VJP row are absent. No trainer, live run,
resume registry, witness-control file, or governed ticket was edited or fired.

After all prerequisites close, main owns a serializer `--patch-file` change at
the then-current trainer SHA:

1. preserve the exact teacher as default/fallback and add a typed provider at
   the actual SegNet loss seam;
2. take one differentiable student backward between exact anchors;
3. force exact refresh at cadence `K_student` and on event/stage/objective/R/
   source/policy/trust-region change;
4. measure exact-anchor value, argmax/margins, full input VJP,
   renderer-gradient direction, CE descent, d_seg, d_pose, rate, islands,
   class anchors, and pose need; fail back to the teacher;
5. persist parameters, full optimizer, anchors, hashes, cadence/event state,
   fallback counters, stage/epoch, and additive legacy-compatible resume fields
   atomically at periodic and every stage boundary; and
6. emit disjoint student/anchor/update/renderer/whole-step timers and run a
   matched exact-teacher versus student treatment through the governed
   launcher.

The explicit later authorization string is:
`GO WHOLE-TEACHER-STUDENT TIMER`. It does not become valid until the fail-closed
prerequisites in the packet pass.

## Triality and unified-stack disposition

- **DSL:** `tac.witness_dsl.whole_teacher_distilled_student_policy` is default
  OFF, `research_only=true`, emits no trainer argv, preregisters sizes/cadences/
  gates, and types optional #487 `K_max=2` separately.
- **Equation:**
  `tac.canonical_equations.whole_teacher_distilled_student_20260713` owns the
  quotient, VJP errors, and fully charged cadence law. The shared registry was
  not edited because it is a live sibling surface.
- **DAG:** `whole_teacher_distilled_student_DAG_FEED_20260713.md` keeps custody
  -> fit -> value -> decisive VJP -> economics -> operator GO -> governed
  treatment -> receiver/evaluator edges durable.
- **Sensitivity map:** no empirical residual is appended until sealed n600
  student-minus-teacher VJPs exist.
- **Pareto/autopilot:** missing custody, fidelity, economics, or hashes refuses
  activation and routes to the full teacher.
- **Bit allocator:** non-binding because this is training apparatus, not an
  archive payload. Any future shipped payload owes counted-byte marginal value
  and receiver survival.
- **Continual learning:** no posterior row is emitted from `n=0`; doing so would
  manufacture evidence. A later sealed n600 receipt must append the residuals.

## Verification and receipt

**MEASURED structural verification:** the five-file focused suite below passed
`40 passed in 1.16s`; Ruff passed all nine source/test files, `py_compile`
passed, and `git diff --check` passed.

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_whole_teacher_distilled_student.py \
  src/tac/tests/test_whole_teacher_distilled_student_vjp.py \
  src/tac/tests/test_probe_whole_teacher_distilled_student.py \
  src/tac/witness_dsl/tests/test_whole_teacher_distilled_student_policy.py \
  src/tac/canonical_equations/tests/test_whole_teacher_distilled_student_20260713.py
```

Final sealed implementation hashes:

The 15-file implementation/triality landing is commit
`65be3f5a1e5fb439896b017155ab628bc340f1e0`; the containment receipt below was
regenerated after that commit so its `git_head` names the landed source state.

| Surface | SHA-256 |
|---|---|
| student/core | `f2bf229dc67ce78b05c03be8c49e41e33021083cafd7ecda720ffc3baf9c48ec` |
| fit/measure probe | `23b37ad74f9069985fa54593fc2298499730ee08f76be6ab7941636d403b77fa` |
| typed DSL policy | `9d63c763af1f515cebf8ff63f654911d063f71b9e382d61a4ac2de59d3769577` |
| canonical equation | `97d5f65e598cd5c865616acc2d2cca4fc826d83951a3b67106114439e953c496` |

**MEASURED containment verification:** the final source-bound preflight exited
with expected `rc=2`, `status=blocked`, `blocker_count=5` over the three unique
surfaces `{cache,storage,backend}`, `n_pairs=0`, `fit_steps=0`,
`teacher_calls=0`, and `bulk_output_root_touched=false`. The requested bulk
root remained absent. This blocker exit is a successful containment verdict,
not a failed structural test.

The durable receipt is
`.omx/research/whole_teacher_distilled_student_blocker_receipt_20260713.json`,
13,471 bytes,
`SHA-256=81a7f8db9e2e3e41fe37b9751de1cf1bca0077e31da27bd6bd46f2709a6fd69a`.
It is byte-identical to the experiment receipt and binds the exact source
hashes, current typed policy, separate blocker scopes/req-R, zero teacher calls,
and no bulk-root mutation. It contains no n600 fidelity or timing claim.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`;
- v7.5 operating-contract and v8 decomposition specs;
- `reports/latest.md`, lane registry, subagent progress, gradient anchors,
  dispatch ledger, cost/continual-learning posteriors, probe-outcomes surface;
- latest sister Codex findings/session summary, council/design memos, and all
  current directive files;
- `jepa_latent_surrogate_20260713.md` and its canonical equation;
- Round 3/4/5, pre-SE, sparse-adjoint, #487, quant-tail reliability, and
  throughput-fresh-eyes receipts/memos.

The two live inbox files were checked at every checkpoint. The operator
architecture directive at `2026-07-14T00:23:02Z` was consumed and is reflected
in the recurrent-local matched-rung section; no later directive was present
through this memo's sealing checkpoint.
