VERDICT: FINDINGS_RESET_COUNTER

# M1R4B science/objective-fit review

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free derivation; no Metal; no new scorer forward]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run_by_this_review=0`,
`metal_runs_by_this_review=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer first

This is not a clean pass. Five science findings block launch. Per the recursive-review
contract, this pass and the two concurrent passes are void as clean-pass credit and the
counter remains `0/3`. The two Round-2 cures are present; none of the findings below is a
restatement of them.

| id | severity | finding | smallest cure before a fresh review sequence |
|---|---|---|---|
| M1R4B-F1 | CRITICAL / LAUNCH-BLOCKING | The event-driven burn is declarative, not executable: the pinned trainer emits no atomic per-eval JSONL, calls no trajectory-stop evaluator, and the ticketed `safe_run` timeout has neither a durable status receipt nor a ticketed resume argv. | Wire one fail-closed monitor/executor that consumes atomic eval rows, emits a typed decision at every eligible row and at every boundary, and provides ticketed fresh/resume argv plus durable timeout/cap receipts. |
| M1R4B-F2 | HIGH / LAUNCH-BLOCKING | Even if wired, `evaluate_trajectory_stop` treats a flat four-row tail as `marginal_below_bar`; it has no event-gap, temporal-noise, loss/margin, or sustained-window input. That is unsafe for the already-observed loss-down/d_seg-staircase regime. | Add a preregistered staircase-aware stop gate: a plateau cannot certify convergence until a rederived event-free horizon and score-relevant facet/loss tests are also flat; add plateau-then-drop and flat-loss-down positive controls. |
| M1R4B-F3 | HIGH / LAUNCH-BLOCKING | The `2e-7` LR and cosine horizon are transferred, not rederived. The cited n32 run used a 6,000-step horizon; its loss fell 18.6% while CPU d_seg worsened 2.1%. At the n120 3,250-step cap the cosine reaches `2e-9`, and extending to 6,500 steps would jump it about 50.5x. | Rederive the LR/schedule on the n120 accumulated-batch geometry, or preregister a same-object checkpoint decision that selects a derived schedule before continuation; make extension preserve schedule state without a horizon-induced LR jump. |
| M1R4B-F4 | HIGH / LAUNCH-BLOCKING | The MLX loop has no EMA shadow and checkpoints only live parameters, despite the canonical run-geometry law and an existing n32 A/B where the last-eight-checkpoint mean beat the final checkpoint by `5.8809916e-6 d_seg`. | Implement and checkpoint a LawRef-derived EMA shadow, or at minimum bind the already-measured symmetric tail-average selection protocol to n120; compare live/EMA/average on the same CPU facet before adoption. |
| M1R4B-F5 | HIGH / LAUNCH-BLOCKING | The sigma protocol proves five-step same-seed determinism, not decision resolution at the marginal bar. It has one d_seg endpoint, no temporal residual estimate, a 47.19-lattice-flip fp16 guard around a one-flip-per-eval stop bar, and a unit-ambiguous falsifier. | Calibrate decision noise from the eval trajectory in d_seg/S units, bind the CPU/MLX agreement envelope to that same unit, and refuse convergence while its upper bound is at or above the one-flip interval bar. |

## Content-pin reverification

`[SOURCE-VERIFIED]` All charter pins matched before the review:

| artifact | required first 16 | reverified sha256 |
|---|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `9c8373b5b352cacc` | `9c8373b5b352cacc2456a21eac0deb53e32f445eb942e4675043825a1d896500` |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `1ef18faf37e2f171` | `1ef18faf37e2f171d480b4e8073c453185f9ae00a1b3200b46d5bb258cd60895` |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` |
| `tools/ddm_seal_orchestrator.py` | `e592cb36fb00d502` | `e592cb36fb00d502693cf17ef43da0f01c7f7c7aecc7d59a3e25e6efeb36e2dc` |

The immutable subject tree is commit `1381ac84cb`. Per the charter's provenance-clause v2,
live HEAD is observational rather than a pin: it was `1381ac84cb` at initial preflight and
advanced to `0d6df1d7ad` through charter/review-note commits while the four content hashes above
remained unchanged. The amended charter was re-read in full. The four content pins must be checked
again at handoff; drift in any one is finding number one.

## RECALL EVIDENCE

| scope | query / source | finding beyond the charter seeds | effect on this review |
|---|---|---|---|
| Governing surfaces | Read `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/tmp/codex_runs/_common_contract.md`, `.omx/state/main_hot_state.md`, and the amended charter in full. | The live own-vehicle pointer is `0.7534578126155775 @ 357,837 B`; M1 is unsealed; any finding resets all concurrent passes. Provenance-clause v2 freezes the four content hashes, not live HEAD. | Kept the work scorer-free and read-only toward the ticket/trainer; only this receipt is written, and expected charter/review-note commits were not misclassified as subject drift. |
| Round 3 first | Read `.omx/research/ddm_m1r3_20260808/M1R3_REVIEW.md` before tracing B1-B7. | Round 3 accepted the declared event predicate, transferred LR, recorded JSONL field list, checkpoint mechanics, and sigma endpoint as sufficient (`M1R3_REVIEW.md:44-53,84-104,157-165`). | Attacked the declarations at their executable consumers instead of repeating the checklist. |
| Round-2 cure state | Read `.omx/research/ddm_m1r2_20260808/M1R2_REVIEW.md` and the current ticket/guard. | The n120 lattice/bar arithmetic and flag-classification guard were cured. | Did not reopen either cured finding. |
| Actual burn source | `rg`/line reads over the pinned trainer, guard, orchestrator, and `trajectory_stopping.py`. | Bounded search did not find `evaluate_trajectory_stop` or `adjudicate_tail_slope` in the pinned trainer/guard/orchestrator. The trainer keeps eval rows only in in-memory `history` and writes checkpoints every 250 steps. | Produced F1/F2 rather than accepting ticket prose as an actuator. |
| MX1 trajectory and averaging | Read `.omx/research/ddm_mx1t_20260807/CHARTER.md`, `MX1T_FINDINGS.md`, and `mx1t_facets_result.json`. | `[MEASURED macOS-CPU advisory, n32]` loss fell while d_seg stayed flat/oscillatory; the final checkpoint was worse than step 250; average-K=8 beat final. The charter explicitly records that the lifted MLX loop has no EMA. | Produced F2/F3/F4 and changed the verdict. |
| Censored-cap precedent | Read `src/tac/optimization/trajectory_stopping.py`, `.omx/research/ddm_tj1_20260805/recall_inventory.md`, and the plateau-then-drop record in `.omx/research/ddm_gc14_first_descent_20260731.md`. | `adjudicate_tail_slope` already carries typed `censored_still_descending` / `ascending_past_min` / `converged_plateau` outcomes because a short-window label previously censored a 6.2-sigma descent (`trajectory_stopping.py:224-373`). | Reused the typed-receipt principle; did not import its 20/40-step spans because M1 evaluates every 50 steps. |
| EMA law | Searched `ema_decay_run_geometry_v1`; opened `src/tac/canonical_equations/ema_decay_run_geometry_20260717.py` and its evaluator. | The canonical law is executable and derives decay from actual optimizer-update horizon; flat legacy constants do not transfer (`ema_decay_run_geometry_20260717.py:4-37,123-169`). | F4 requires a LawRef-derived shadow, not a guessed decay. |
| Selection/bias | Opened `_select_stratified_indices`; searched m88/m96 and opened `.omx/research/ddm_na2_negative_audit_20260803.md`. | The source samples one seeded random member from each of 120 five-index buckets; the prior prefix law measures seg prefixes 0.95-0.97x easier and pose prefixes 2.54-4.21x harder (`ddm_na2...md:329-337`). | B3 is clean; no prefix/strided inference was used. |
| Canonical objective | Read `upstream/evaluate.py:63-65,89-100`, `src/tac/contest_score.py:1-59`, and `score_marginal_lagrange_multipliers_v1`. | The exact composition is `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`; the trainer uses only expected flip margin. | B4 separates the correct seg-axis attack from a composed-score objective claim. |
| Live task/ledger | Searched `.omx/state/main_hot_state.md`, the harness bridge, the canonical research index, the long DAG, and bounded M1/MX1 task strings. | Did not find in those searched scopes a newer authority that makes the pinned M1 object sealed or gives the missing stop/EMA/sigma actuators. | Kept the review on the pinned object and described bounded absence, not global nonexistence. |

## Round-3 attack accounting

I intentionally did not re-audit Round 3's load-phase memory projection, guard-argv bijection,
atomic NPZ replace mechanics, CPU scorer batching, HPAC component bytes, or competing Metal-lane
ranking. Those are not load-bearing to the findings here and are primarily mechanics/counterfactual
surfaces.

Checks added that Round 3 did not perform:

1. traced the event predicate to an executable caller and telemetry sink;
2. evaluated the flat-window behavior of the selected stop implementation;
3. traced the 3,250-step cap through `safe_run`, resume argv, and cosine-horizon semantics;
4. checked the actual n32 score-relevant trajectory rather than the phrase
   `MEASURED-descending`;
5. traced EMA state through the live MLX update and checkpoint payload;
6. compared sigma/envelope units to the n120 lattice quantum and marginal bar;
7. executed the exact seeded selector formula scorer-free and compared its shape to prefix/stride.

## B1. Marginal bar and event-punctuated stopping

### Derivation

`[DERIVED]` The cured threshold is arithmetically correct:

```text
one lattice-site flip in d_seg = 1 / (120*384*512)
                               = 4.238552517361111e-8
one lattice-site flip in S     = 100 times that
                               = 4.238552517361111e-6 S
bar per optimizer step         = one-flip S / 50
                               = 8.477105034722223e-8 S/step
```

The ticket records these values at `launch_ticket_v5_event_driven.json:759-781`.

The selected implementation does not encode the rest of the advertised event logic.
`TrajectoryStopConfig` has no `window_rows`, `sustained_erosion_windows`, loss, margin, event-gap,
or noise-envelope field (`trajectory_stopping.py:48-75`). If no smooth decay fit clears R2, the
fallback uses the last four points and clips negative gain to zero (`trajectory_stopping.py:547-563`).
Therefore a flat four-row tail deterministically has marginal gain zero and becomes
`marginal_below_bar` (`trajectory_stopping.py:566-623`), even if the smooth loss is still falling
or a discrete flip event would occur just after the window.

`[MEASURED macOS-CPU advisory, n32, not transferred as n120 magnitude]` The retained MX1 series
shows exactly the relevant shape: loss falls from `0.0002796522` at step 250 to `0.0002276980` at
step 3250 while d_seg moves from `0.0010511080` to `0.0010732015` and oscillates across the saved
series (`MX1T_FINDINGS.md:22-38`; `mx1t_facets_result.json`, checkpoint rows). The MX1T charter
describes the observed loss-down/d_seg-flat divergence at `CHARTER.md:3-11`.

### Conclusion

`FINDING M1R4B-F2.` The bar is correct but not a scientifically safe stopping rule for this
trajectory class. A below-bar window is not evidence that a later discrete event is absent.

### Honesty boundary

`[INFERRED mechanism, source-verified predicate]` This review did not run an n120 trajectory and
does not assert that a plateau-then-drop will occur. It proves that the current rule would stop on
the flat prefix of such a trajectory and cites an existing same-family n32 divergence as the risk
anchor. The negative is INSTANCE/PREDICATE scoped, not a kill of trajectory stopping.

## B2. The 3,250-step safety cap and typed exit

### Source trace

The ticket says MAIN or a monitor evaluates the predicate at every eligible eval row and boundary,
and queues a resume on a bound (`launch_ticket_v5_event_driven.json:759-783`). The executable path
does not implement that statement:

- the fire argv is one `--steps 3250` process under an 8-hour timeout, with no `--status-receipt`,
  no `--child-pidfile`, and no `--resume-from` (`launch_ticket...json:60-123`);
- the trainer evaluates d_seg at cadence, appends only to in-memory `history`, and saves NPZ files
  only every 250 steps (`ddm_mx1_pr130_semantic_renderer.py:3079-3187`);
- bounded search did not find a call to either trajectory adjudicator in the pinned trainer,
  fire guard, or seal orchestrator;
- `safe_run` kills the process group when the timeout binds and writes a durable status only when
  `--status-receipt` is supplied (`safe_run.py:95-117,390-425,509-547`); the ticket does not supply it.

`[DERIVED]` At the ticket's measured/projection cadence of roughly 29.3 seconds per step, 8 hours
binds around step 982, well before 3,250. The last periodic checkpoint would generally be step 750;
the exact last step is not asserted because runtime cadence varies. No typed scientific exit record
is named for that timeout.

The cap continuation is also undefined by the optimizer schedule. The trainer computes cosine LR
from absolute `step / (args.steps-1)` (`ddm_mx1...py:2939-2944`). Thus:

```text
lr(step=3249, total=3250) = 2.0e-9
lr(step=3250, total=6500) = 1.0097607188e-7
ratio on a naive doubled-horizon resume = 50.488x
```

Resuming with the same `--steps 3250` performs zero updates; increasing `--steps` changes the
schedule and creates the jump above.

### Conclusion

`FINDING M1R4B-F1.` The cap/timeout is a censored experiment with no executable typed adjudication
or schedule-preserving continuation. Ticket prose is not a receiver for the telemetry it names.

### Smallest scientific receipt

At every eval/boundary, persist: ticket/argv/checkpoint hashes; step and wall time; d_seg/S and loss
trajectory; selected fit plus uncertainty; best and endpoint checkpoint identities; typed outcome
`STOP_CONVERGED`, `CENSORED_STILL_DESCENDING`, `ASCENDING_PAST_MIN`, or `QUEUE_RESUME`; and the exact
next argv/fire order. Window spans and noise thresholds must be rederived for M1's 50-step cadence,
not copied from the 5-step TP1 precedent.

## B3. n120 selection and prefix bias

### Verification

`[SOURCE-VERIFIED]` `_select_stratified_indices` seeds NumPy, partitions `range(600)` into 120
contiguous buckets, chooses one random member per bucket, and sorts the result
(`ddm_mx1_pr130_semantic_renderer.py:728-732`). The train path calls it with the ticket seed
(`ddm_mx1...py:2774-2777`). Since 600/120 is exact, every stratum has five indices.

`[DERIVED scorer-free]` Re-executing that exact formula with seed `20260808` produced 120 unique
indices spanning 2 through 596, list SHA-256
`45e7489093320845600a05d30ab59c9b1404584153f2a65e4f686b1e134e3f63`; it is neither
`range(120)` nor `range(0,600,5)`.

The bias motivation is real: `[MEASURED prior population audit]` prefix subsets were 0.95-0.97x
easier on seg but 2.54-4.21x harder on pose (`ddm_na2_negative_audit_20260803.md:329-337`).

### Conclusion

`CLEAN for B3.` The actual selection algorithm is seeded stratified-random, not prefix or evenly
strided. This says nothing about n600 authority; n120 remains a sampled research trajectory.

## B4. Objective fit to the composed score

### Gap decomposition

Using the rounded live components and PR130 reference components cited by the board, `[DERIVED]`:

| component | own | PR130 reference | gap | fraction of `0.5813165` gap |
|---|---:|---:|---:|---:|
| `100*d_seg` | `0.4305420` | `0.0296600` | `0.4008820` | `68.96%` |
| `sqrt(10*d_pose)` | `0.08464685` | `0.01526761` | `0.06937924` | `11.93%` |
| `25*bytes/37,545,489` | `0.23826897` | `0.12721368` | `0.11105529` | `19.10%` |

The exact live pointer uses higher-precision inputs; this rounded recomputation is decision
arithmetic, not a new score row. It agrees with the live board's `~0.5813`, seg `~0.4010` and 69%
statement (`main_hot_state.md:5-20`). Even perfect closure of M1's seg gap leaves about `0.1804345 S`
of pose plus rate gap.

### Actual optimized quantity

The MLX loss is a mean sigmoid of negative target-class margin after the real round-trip and frozen
SegNet; with the ticket fractions it stays in the expected-flip branch
(`mlx_semantic_renderer.py:292-333`; `ddm_mx1...py:2975-2989`). It contains no pose term and no rate
term. The discrete stop metric `100*d_seg_batch_mlx` is correctly aligned to the seg component, but
the differentiable training loss is only a convenient seg surrogate. Existing n32 evidence shows
that surrogate can fall while CPU d_seg does not.

### Conclusion

`CLEAN axis choice, NOT a composed-score objective.` Seg is the majority gap, so a receiver burn is
the right axis. Scientifically valid conclusions are limited to this sampled receiver/config's seg
trajectory and facet response. M1 alone cannot claim composed-score progress, PR130 closure, pose/rate
closure, byte closure, or n600/exact authority.

## B5. LR, EMA, and schedule

### LR and schedule

The ticket honestly says `2e-7` came from PR130 batch-size 2 and that M1 accumulates all pairs
(`launch_ticket...json:668-676`). It then calls the pair `(accumulated-batch, 2e-7)`
`MEASURED-descending` at n32. That support is not score-relevant at the current artifact:

- `[SOURCE-VERIFIED]` the n32 ticket used `--steps 6000`, seed `20260806`, eval every 250, and
  `2e-7` (`launch_ticket_mx1g_from_regen2.json:234-289`), while M1 uses 3,250, seed `20260808`, eval
  every 50 (`launch_ticket_v5_event_driven.json:60-123`);
- `[MEASURED macOS-CPU advisory, n32]` from step 250 to 3250, loss improved 18.58% but d_seg worsened
  2.10% (`mx1t_facets_result.json`, checkpoint rows; summary at `MX1T_FINDINGS.md:22-38`).

Therefore `2e-7` is IMPORTED/TRANSFERRED with proxy-loss support, not rederived and not shown
descending on the score-relevant quantity under the M1 schedule.

### EMA

The MLX loop constructs only model plus AdamW, updates live parameters, and checkpoints
`model.parameters()` plus optimizer state (`ddm_mx1...py:2831-2844,3072-3074,3139-3187`;
`mlx_semantic_renderer.py:358-400`). There is no EMA shadow in this path. The prior MX1T charter
states the same at `CHARTER.md:39-46`.

This is not an abstract doctrine objection. `[MEASURED macOS-CPU advisory, n32]` average-K=8 improved
d_seg by `5.880991617838397e-6` versus the final checkpoint (`MX1T_FINDINGS.md:40-58`). The canonical
EMA law derives constant decay from actual optimizer updates and a declared seed/warmup horizon
(`ema_decay_run_geometry_20260717.py:4-37,123-169`); the ticket supplies neither an EMA nor the law
inputs.

### Conclusion

`FINDINGS M1R4B-F3 and F4.` The LR/schedule are a transferred formulation whose n32 support is loss
descent, and the no-EMA path discards an already-measured same-vehicle selection lever. A negative
M1 endpoint would confound receiver capacity, LR/schedule, and live-vs-average selection.

## B6. Sigma protocol versus the bar

### What the protocol proves

`[MEASURED, ticketed scope]` Five identical-seed fp16 five-step checkpoints are byte-identical;
same-checkpoint-derived metrics therefore have repeat sigma zero. The fp16 and fp32 CPU d_seg
endpoints are both `0.0010835435655381944` (`launch_ticket...json:701-748`). This is a useful
short-horizon determinism/fp16 sanity receipt.

### What it cannot prove

It samples one d_seg endpoint per run, so it has no time-series residual or event-gap distribution.
Repeating one deterministic trajectory cannot estimate the within-trajectory flip churn that the
stop rule must distinguish from descent.

`[DERIVED]` On the n120 lattice:

```text
one d_seg lattice quantum                = 4.238552517361111e-8
fp16 guard 2.0e-6 d_seg                  = 47.18592 quanta
plateau_eps*window = 2.5e-6 d_seg        = 58.98240 quanta
stop bar over one 50-step eval interval  = 1 quantum
```

Thus the admission envelope is much coarser than the scientific stop event. The ticket also records
an fp16/fp32 training-loss delta `2.0381150534376502e-5` but phrases its falsifier as delta greater
than `max(2e-6,3*sigma)` without binding the metric/unit (`launch_ticket...json:710-719,750-755`).
If read as the recorded training-loss delta, it exceeds `2e-6` by 10.19x; if read as d_seg, the
zero endpoint delta passes but still does not estimate temporal decision noise.

### Conclusion

`FINDING M1R4B-F5.` The noise floor relevant to the one-flip marginal decision remains
`UNMEASURED`, not zero. Under the charter's fail-closed rule, convergence cannot be admitted until a
same-unit upper bound is below the bar. This does not reject fp16 training; it rejects the claim that
the current sigma probe resolves the event predicate.

## B7. Scientific unusability modes and preregistrations

| mode | mechanism | scenario | cheap preregistration / cure |
|---|---|---|---|
| False convergence on a staircase | Four flat d_seg rows force zero fallback marginal, while smooth margin/loss can continue to prepare a later flip. | M1 stops after five eligible rows, just before a class-boundary event. | Synthetic plateau-then-drop positive control; convergence requires a rederived event-free horizon plus flat score-relevant facets/loss, with a typed censored outcome otherwise. |
| Uninterpretable cap/continuation | Timeout kills before the forecast cap; no durable decision exists; changing total steps changes cosine LR. | The next operator sees only a last checkpoint and manually resumes with a 50x LR jump. | Durable safe-run receipt plus atomic eval JSONL, typed boundary adjudication, and schedule-state-preserving resume argv named before fire. |
| Optimization-constant confound | Imported LR, a changed cosine horizon, and no EMA can dominate endpoint selection. | d_seg is flat/worse and the result is misread as receiver-family capacity. | Same-object CPU-facet comparison at registered checkpoints; LawRef-derived EMA/live rows; verdict scope fixed to INSTANCE/CONFIG until the LR/schedule/selection leg is deconfounded. |
| Precision/decision-unit mismatch | Five-step deterministic repeats report sigma zero while the stop envelope is 47-59 lattice flips. | MLX/CPU or temporal churn changes a one-flip stop decision inside an accepted envelope. | Report every envelope in d_seg and S quanta; estimate temporal residuals from eligible rows; refuse stop when the confidence bound overlaps one flip. |

## Final recommendation and authority boundary

Recommendation: `FINDINGS_RESET_COUNTER`. MAIN should cure the five findings, repin the changed
ticket/transitive sources, and start three fresh independent passes. This review did not fire a
burn, run Metal, consume a scorer slot, modify the sealed artifact, create an archive, or move a
pointer.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
