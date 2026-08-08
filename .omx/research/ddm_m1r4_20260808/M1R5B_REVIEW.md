VERDICT: FINDINGS_RESET_COUNTER

# ddm_m1r5b — M1 fire-gate science review after the m1c1 amendment

Tags: [no-triality] [p0-ledger-ok]
Axis: [apparatus/source review; scorer-free]
Scope: the four content-pinned subject files plus transitive reads; no Metal, scorer, ticket mutation,
trainer mutation, archive build, or score claim.

## Answer first

The amended M1 artifact is not scientifically ready to fire. It now contains a real in-process
stopping evaluator, durable journal, EMA shadow, and typed cap machinery, but six defects still make
the proposed burn capable of stopping for the instrument rather than the receiver, or of producing a
terminal state that the preregistered CPU authority path cannot adjudicate. Any one finding resets all
three concurrent passes; the counter remains 0/3.

No score was measured and no pointer moved.

## Frozen-subject re-verification

Recomputed with `shasum -a 256` before source review and again before writing this receipt:

| file | expected prefix | recomputed sha256 | verdict |
|---|---|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `90cf28d390999ef9` | `90cf28d390999ef9cda47340d9ec01bc65a15fb9ab3f88c60625abc29b414ec9` | MATCH |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `8bad6a6b8be1b201` | `8bad6a6b8be1b20189791283f64638d01a1b406482a1a725bab712c77f27894c` | MATCH |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` | MATCH |
| `tools/ddm_seal_orchestrator.py` | `11c4368f009afc31` | `11c4368f009afc31aca67da37d7c9b2a9b109f3a00d20ca903800c26f3897a79` | MATCH |

The content hashes, not live HEAD, define the subject. The reviewed content is the m1c1-amended tree
at `393d67d016`; later charter/receipt commits do not move the subject.

## Findings, most severe first

| id | severity | mechanism | concrete failure scenario | smallest correct cure |
|---|---|---|---|---|
| M1R5B-F1 | CRITICAL / LAUNCH-BLOCKING | The timeout-surviving controller exists, but both the ticket's FIRE step and the seal orchestrator nominate the raw safe-run child. The ticket says `command_ref=argv_m1_n120_cap_saturated` (`launch_ticket...json:649-652`). With no top-level `fire_argv_key`, the orchestrator defaults to that same child and reports it as FIRE-ready (`ddm_seal_orchestrator.py:237-259`). Only `run_m1_controlled_train` survives return code 124 and writes the wall-cap decision/terminal receipt (`renderer.py:4320-4401`). | Fresh training normally halts at the step-250 calibration boundary, but a pre-existing passed schedule receipt, a later direct resume, or any route that reaches the eight-hour wall cap runs as the child. `safe_run` kills it; the controller is not alive to type the censored exit. The advertised scientific cap receipt becomes route-dependent. | Make the seal/FIRE surface report the controller argv while separately guarding its child argv; refuse any M1 FIRE plan that names a child key directly. Add a control that drives the exact orchestrator-reported command to a synthetic return-124 status receipt and proves the terminal cap receipt appears. |
| M1R5B-F2 | HIGH / LAUNCH-BLOCKING | The staircase patience is an uncalibrated five-eval literal. `event_free_horizon_evals=5` (`launch_ticket...json:888`) becomes 250 steps (`renderer.py:208-220`). A six-row flat trace is explicitly accepted as `STOP_CONVERGED` (`test_trajectory_stopping.py:199-210`). The plateau-then-drop control invokes the evaluator only after the drop (`test_trajectory_stopping.py:168-181`); it never evaluates every online prefix, so it does not test whether the controller would have stopped before a later drop. | Six flat eligible rows certify convergence at the exact patience boundary; a lattice drop on the next eval is never observed. This is the known event-punctuated failure class wearing an event-aware label. | Derive patience from a same-vehicle inter-event-gap or conservative censored-horizon receipt, not `5`. Test the controller sequentially at every prefix and place the positive-control drop strictly after the current horizon; no earlier prefix may return `STOP_CONVERGED`. Until calibrated, cap exits must remain typed censored outcomes, not convergence. |
| M1R5B-F3 | HIGH / LAUNCH-BLOCKING | The base LR remains transferred from PR130, and the proposed step-250 adoption rule is internally weaker than M1's own value bar. One n120 flip over 250 steps is `1.6954210069444444e-08 S/step`, exactly 0.2 times the stop bar `8.477105034722223e-08 S/step`. Nevertheless one flip admits the schedule (`renderer.py:4285-4315`). The safety cap is also reused as the cosine and curriculum horizon (`renderer.py:2927-2966,3323-3367`; `launch_ticket...json:921-928`). | A transferred LR can pass by five times less marginal value than the burn later calls worth continuing. The first 250 completed updates test LR near its start (`1.971443484195934e-7` on update 250), not the decay to `2e-9` at step 3249. At the cap, the resume holds `2e-9`; schedule-induced freezing can look like receiver convergence. | Separate safety bounds from scientific schedule/curriculum horizons. Re-derive the LR/schedule at M1 batch geometry with a same-object comparison whose admission rate is at least consistent with the registered marginal bar; a nonzero one-flip result is only a calibration signal, not schedule adoption. Do not invent values: derive candidate values from a registered gradient/batch-geometry law or an explicitly scoped bracket. |
| M1R5B-F4 | HIGH / LAUNCH-BLOCKING | The sigma batch cannot exercise the event classifier it claims to validate. Each sigma argv runs only 5 steps with one eval at step 5 (`launch_ticket...json:222-604`), while the stop requires at least five rows and a 250-step event-free horizon (`launch_ticket...json:888,936-955`). Sigma keys are not M1 child keys, so `_load_m1_executor_policy` returns no controller (`renderer.py:192-231`), no stop journal, and no decision. The recorded repeat sigma is for final-step training loss, not a decision-rate series (`launch_ticket...json:821-847`). | The five repeats are byte-identical and prove deterministic short-horizon execution, but `event classification flips inside the envelope` is never evaluated. Seal can pass without any fp16/fp32 decision comparison at the one-flip-per-50-step bar. | Run a cadence-compatible fp16/fp32 calibration long enough to produce the minimum online decision history and compare the actual typed decisions in common S/step units. Bind every falsifier to a named metric and unit. Short deterministic repeats may remain a backend check, but must not be called stop-rule resolution. |
| M1R5B-F5 | HIGH / OUTCOME-INTERPRETABILITY | Terminal CPU authority is preregistered only for step 3250, although the event controller may halt at any 50-step eval or at a wall cap. The three commands hard-code live/EMA/K8 step-003250 paths (`launch_ticket...json:908-918`). The adoption rule also requires a selector receipt naming the minimum (`:895-896`), but the pinned subject has no terminal selector command or consumer; the only implemented selector compares step 0 to step 250 for schedule admission (`renderer.py:4285-4317`). | An event stop at step 300, 1000, or a wall-cap checkpoint cannot run the listed CPU commands. Even at step 3250, three verdict files exist without the promised selector, and no executable check decides whether CPU authority refutes the MLX stop trend. The result is not adoptable or interpretable under its own rule. | Generate terminal CPU argv from the actual halt receipt's live/EMA/tail paths and add a fail-closed selector that checks same pair IDs and same step, names the minimum CPU d_seg, and records MLX-vs-CPU stop agreement. Queue that exact selector from every terminal mode. |
| M1R5B-F6 | MEDIUM / VALUE-PROVENANCE | `d=0.999` is algebraically reproduced from the canonical law, but the law input is not scientifically derived for M1. The ticket equates `K=8` checkpoint averaging with a 2,000-update two-time-constant warmup (`launch_ticket...json:878-886`). Eight checkpoint states 250 steps apart span 1,750 updates, and a uniform K8 mean does not identify an EMA warmup fraction. The same decay is then carried into a possible 6,500-update resume even though `updates_per_run` remains 3,250 (`renderer.py:234-251,2927-2940`). | The LawRef verifies arithmetic around an assumed `phi`; it does not authorize the assumption. At 3,250 updates seed retention is `0.999^3250 = 0.0387112`; at 6,500 it is `0.00149856`, a different run geometry. A winner/loser result is confounded with an arbitrarily chosen shadow horizon. | Declare EMA exploratory until a target seed fraction or warmup fraction is derived for each allowed total-update horizon. Re-derive on extension, or pre-register one horizon-invariant scientific target. Keep live and K8 as independent candidates; do not infer EMA calibration from K8's n32 win. |

## B1 — stopping rule after the threshold cure

### Re-derivation

`[DERIVED]` From the ticket's own geometry (`N=120`, `H=384`, `W=512`):

```text
sites                    = 120 * 384 * 512 = 23,592,960
one d_seg lattice flip   = 1 / 23,592,960
                         = 4.238552517361111e-8 d_seg
one S lattice flip       = 100 / 23,592,960
                         = 4.238552517361111e-6 S
marginal bar             = one S flip / 50 steps
                         = 8.477105034722223e-8 S/step
```

The ticket values at `launch_ticket...json:936-955` are arithmetically correct. Source also re-derives
the one-flip quantum and refuses drift (`renderer.py:201-220`).

### Scientific adequacy

`[SOURCE-VERIFIED]` The amended evaluator is materially better than round 3's reviewed object. It
requires event-free time, flat loss, a trajectory-own slope-noise upper bound below the marginal bar,
weight-update liveness, and no sustained erosion (`trajectory_stopping.py:775-900`). It is not merely
the old four-row smooth-fit predicate.

`[UNDETERMINED]` No source in the searched M1/MX1/GC21/canonical-equation scopes derives five evals as
the longest credible inter-event gap for this vehicle. The amendment derives 250 only by multiplying
the chosen literal by cadence; that is dimensional conversion, not scientific calibration.

`[EXECUTED scorer-free control / SOURCE-VERIFIED failure]` Calling the production evaluator on six flat
rows at steps 0..250 returned `STOP_CONVERGED`; appending a drop at step 300 returned `CONTINUE`. Thus
the existing plateau-then-drop test asks only what happens after a drop that the live controller might
never reach. This yields finding F2.

Conclusion: the threshold is correct; the patience/staircase model is not yet adequate.

## B2 — safety cap, censoring, and continuation

`[SOURCE-VERIFIED]` A step cap is now typed. At a step boundary, unresolved evidence becomes
`QUEUE_RESUME` (`trajectory_stopping.py:876-899`), and the trainer writes a terminal receipt with the
live/EMA/tail checkpoints and exact resume key (`renderer.py:3561-3613`). A wall cap is also typed when
the controlled wrapper survives safe-run rc 124 (`renderer.py:4320-4401`). These are real cures relative
to round 3.

`[SOURCE-VERIFIED failure]` The fire plan does not consistently invoke the wrapper that makes the wall
cap observable. The ticket and seal orchestrator both nominate the child, producing F1.

`[DERIVED]` The cap is also a plan in disguise because `horizon_steps=3250` controls both cosine LR and
curriculum coordinate. The LR reaches `2e-9`, then remains there through a 6,500-step extension. Thus
`QUEUE_RESUME` after the cap does not preserve a still-live schedule; it queues a near-frozen extension.
That is censored scientifically even though it has a typed apparatus receipt. This contributes to F3.

Conclusion: cap receipts exist, but the fired route and scientific continuation plan are not closed.

## B3 — n120 population

`[SOURCE-VERIFIED]` `_select_stratified_indices` seeds NumPy, splits `0..599` into 120 buckets, samples
one member from each bucket, and sorts the result (`renderer.py:909-913`). The training path calls it
with the recorded seed (`renderer.py:2922-2926`). Because 600/120=5 exactly, this is a seeded one-of-five
draw in every contiguous stratum, not a prefix and not fixed stride.

`[MEASURED prior population audit, not remeasured here]` The reason this matters is supported by the
existing audit: prefixes are 0.95-0.97x easier on seg and 2.54-4.21x harder on pose; the first 120 pairs
are the two hardest pose blocks and the block range reaches 79x (`ddm_na2_negative_audit_20260803.md:
329-342`).

`[SOURCE/ARTIFACT-VERIFIED]` The existing sigma result records 120 ids spanning the clip rather than
`0..119`. No scorer was run in this review.

Conclusion: CLEAN for B3. The population is seeded stratified-random. It remains n120 sampled research
evidence, not n600 authority.

## B4 — objective fit

`[SOURCE-VERIFIED]` The burn optimizes the frozen SegNet target-margin curriculum after the contest-
faithful round trip (`renderer.py:3335-3367`), and the stopping objective is `100*d_seg_batch_mlx`
(`renderer.py:3487-3519`; ticket `:951`). It has no pose or rate term.

`[DERIVED from the standing decomposition]` Against PR130, the live gap is approximately:

| term | gap S | share of 0.5813 gap | M1 directly optimizes it? |
|---|---:|---:|---|
| seg | 0.4010 | 69.0% | yes, via a differentiable seg surrogate and discrete d_seg stop |
| pose | 0.0693 | 11.9% | no |
| rate | 0.1110 | 19.1% | no |

Perfectly closing only the seg gap would still leave about `0.1803 S` of pose+rate gap. Therefore M1 is
aimed at the dominant component, but it is not a composed-score burn and cannot itself establish PR130
closure. The objective choice is CLEAN; the surrogate-to-discrete divergence remains a known
interpretability risk and is why terminal CPU authority must be executable.

## B5 — LR, EMA, and schedule

### LR and schedule

`[SOURCE-VERIFIED/TRANSFERRED]` The ticket accurately labels `2e-7` as a PR130 source constant from a
batch-size-2 regime and `BORROWED_CANDIDATE_NOT_ADOPTED` (`launch_ticket...json:767-775,921-928`). It
does not derive an M1-optimal LR.

`[DERIVED]` The step-250 gate is inconsistent with the stopping economics:

```text
admission gain             = 1 flip / 250 steps
                           = 1.6954210069444444e-8 S/step
registered marginal bar    = 8.477105034722223e-8 S/step
admission/bar              = 0.2
```

`[DERIVED]` The schedule values are `1.971443484195934e-7` on the 250th completed update and `2e-9` at step 3249; the
calibration observes only the near-base-LR head. The safety cap, not a measured knee or stage law,
defines that decline. This produces F3.

### EMA

`[SOURCE-VERIFIED]` The EMA is now real: it is initialized from live weights, updated after every
optimizer update, checkpointed distinctly, and restored on resume (`renderer.py:3044-3063,
3231-3308,3453-3457`). This cures round 4's missing-shadow defect.

`[DERIVED arithmetic / ASSUMED scientific input]` `ema_decay_run_geometry_v1` correctly maps
`phi=2000/3250` to `d=0.999`, but M1's choice of `phi` is not derived by the n32 K8 result. K8 is a
uniform mean of eight stored states, not a measurement of two EMA time constants. The allowed 6,500-
update continuation also changes the geometry without changing the law inputs. This produces F6.

### Curriculum

`[SOURCE-VERIFIED]` `ce_fraction=0` and `softplus_fraction=-999` keep the ticket on the expected-flip
tail branch, but `total_steps=schedule_horizon_steps` still supplies 3,250 as the curriculum coordinate
(`renderer.py:3359-3367`). No current-vehicle derivation for that boundary was found in the searched
scope. It is coupled to F3 rather than a separate finding.

## B6 — sigma-calibration resolution

`[MEASURED by pre-existing ticket artifacts]` Five same-seed fp16 checkpoints are byte-identical, so
the backend repeatability floor for checkpoint-derived metrics is zero in that exact five-step scope.
The CPU d_seg endpoint is identical for fp16 and fp32: `0.0010835435655381944`
(`launch_ticket...json:800-847`). This is useful determinism evidence.

`[DERIVED]` A single lattice flip over the five-step calibration is
`4.238552517361111e-6 / 5 = 8.477105034722222e-7 S/step`, ten times the marginal bar. More importantly,
each run has one d_seg endpoint and no controller decision, so the protocol has no empirical resolution
for a one-flip-per-50-step classifier.

`[SOURCE-VERIFIED]` The production decision's slope-noise guard is estimated from its own tail
(`trajectory_stopping.py:843-872`), which is better than importing the training-loss sigma. But this
does not make the five-step seal protocol capable of the fp16/fp32 event-classification comparison it
claims. F4 is launch-blocking under B6's fail-closed instruction.

## B7 — likely scientifically unusable outcomes

| likely unusable outcome | mechanism | cheap preregistration now |
|---|---|---|
| Schedule-induced false convergence | LR/curriculum decay is tied to the safety cap and freezes at 1% of base; a flat objective may measure the schedule rather than receiver capacity. | Separate cap and schedule provenance; align the schedule-admission rate with the marginal bar and record the LR/curriculum state in every terminal interpretation. |
| Staircase event censored by online patience | A future drop can occur after the uncalibrated 250-step event-free horizon; the current positive control never checks the pre-drop prefixes. | Prefix-by-prefix test with a drop after the horizon; derive patience or type the endpoint as censored. |
| Terminal result cannot be adjudicated on CPU authority | Event/wall halts can occur away from 3250, terminal CPU argvs are static, and no minimum-selector exists. | Generate commands from the actual terminal receipt and execute one fail-closed same-step selector with MLX/CPU trend agreement. |

All three are INSTANCE/PROTOCOL scoped. None kills the semantic-receiver family.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | effect on this review |
|---|---|---|---|
| Governing state | `PROGRAM.md`, identical-content `CLAUDE.md`/`AGENTS.md`, craft handoff, common contract, live hot state | The live board already recorded the prior round-4 defect class and m1c1 activity, but the four content hashes still defined this independent subject. | Did not inherit the board's conclusions as proof; used source traces against the pinned hashes. |
| Round 3 first | `.omx/research/ddm_m1r3_20260808/M1R3_REVIEW.md` | Round 3 accepted ticket declarations for event stopping, transferred LR, sigma, and EMA absence boundaries without an executable amended object. | Attacked the amendment's new mechanisms rather than repeating its checklist. |
| Full M1/MX1 corpus | queries for `ddm_m1`, `MX1`, `plateau-then-drop`, `event_free_horizon`, `same_object_cpu_selection`, `3250`, `2e-7`, `EMA`, and `sigma` over `.omx/research`, source, tools, state, index, and DAG | Found M1C1's cure receipt, the n32 MX1T trajectory/averaging receipt, the online test shape, GC14's boundary-step precedent, and the old R4B failure analysis. | Exposed that the new patience test is not online, schedule admission is weaker than the stop bar, and terminal selection is absent. |
| Canonical equations | `tools/list_canonical_equations.py --json` plus direct read of `ema_decay_run_geometry_v1` and `trajectory_stopping.py` | The EMA law derives decay only after a target seed/warmup fraction is supplied; it does not derive that scientific target. The staircase evaluator estimates decision noise from the live tail. | Kept EMA arithmetic separate from input authority; did not misstate the five-step repeat sigma as the production slope-noise estimate. |
| Population law | `_select_stratified_indices`, sigma pair IDs, `ddm_na2_negative_audit_20260803.md` | Confirmed true seeded stratification and the opposite-sign prefix bias. | B3 stays clean. |
| Task/ledger/index | bounded searches over canonical research index, long DAG, hot state, task status, and P0 ledger | Did not find in those searched scopes a later terminal selector, a derived five-eval M1 event horizon, or an M1 batch-geometry LR derivation. | Reported bounded absence; these missing inputs remain named rather than guessed. |
| Dirty worktree/index | `git status --short`, `git diff --cached --name-only` | Many unrelated user changes exist; staged index was empty before this receipt. | Touched only this review file and will use the serializer with its post-edit hash. |

## Round-3 attack accounting

I deliberately did not re-check round 3's memory projection arithmetic, mem-probe load telemetry,
guard equivalence tuple, checkpoint NPZ atomicity, CPU verdict batching, HPAC byte alternative, or
Metal-window counterfactual ranking. Those are primarily mechanics/arith/counterfactual surfaces and
were not needed to decide this pass.

I checked things round 3 could not check on its older object:

1. traced the amended cap survivor all the way to the ticket's and seal orchestrator's reported FIRE
   command;
2. evaluated the staircase control as an online-prefix problem rather than recognizing a post-drop
   expected answer;
3. compared the LR calibration gate numerically with the stop rule's own marginal bar;
4. separated the safety cap from its hidden role as LR and curriculum horizon;
5. checked the authority of the EMA law inputs, including the 6,500-update continuation geometry;
6. proved the sigma argvs cannot produce a stop decision at all;
7. traced terminal CPU commands to arbitrary event-stop steps and searched for the promised selector.

## Recommendation and boundaries

Recommendation: cure F1-F6, repin all changed content, and restart three independent reviews at 0/3.
Do not fire M1 from this subject.

This review measured no new model, d_seg, d_pose, bytes, or S; it ran no Metal or scorer work and did
not modify the ticket, trainer, guard, orchestrator, upstream, or staged index.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved at `0.19108`.
