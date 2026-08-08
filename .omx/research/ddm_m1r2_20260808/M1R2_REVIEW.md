# M1R2 Review - M1 n120 Fire Ticket Pass 2

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free review]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run_by_this_review=0`,
`metal_runs_by_this_review=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer First

Recommendation: `FINDINGS_RESET_COUNTER`.

This is not `CLEAN_PASS_2_OF_3`. I found two review findings in the sealed M1
ticket surface:

| id | severity | verdict | finding | required action |
|---|---|---|---|---|
| M1R2-F1 | HIGH | FINDING | The ticket's `one_sample_flip_S` and `marginal_bar_S_per_step` do not follow from its own `N=120`, `H=384`, `W=512`, `eval_every_steps=50` constants or from GC21's predicate. | Recompute and reseal the stop predicate before any fire. Counter resets to 0/3. |
| M1R2-F2 | MEDIUM | FINDING | The fire guard's receipt-equivalence tuple omits several `mlx-train` flags consumed by the burn path (`steps`, `seed`, `eval_every`, `checkpoint_every`, `run_dir`, and `resume_from`). | Either add them to the receipt/equivalence contract, or explicitly document why each omitted flag is outside the mem-probe equivalence claim. Counter resets to 0/3. |

Clean surfaces remain clean within scope: the orchestrator derives gates from the
ticket, the entrypoint re-evaluates the in-process guard for GPU `mlx-train`,
the sigma result is scoped to the calibration horizon, the fp16/fp32 d_seg
verdict is real and not a checkpoint collision, stale receipts fail closed, and
3250 is written as a safety cap rather than convergence.

## Provenance Pins

| artifact | sha256 |
|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `dafb9e7792564de2ad3b68875792c5b4e75eb91b72360e99db78d6bed441fd47` |
| `tools/ddm_seal_orchestrator.py` | `e592cb36fb00d502693cf17ef43da0f01c7f7c7aecc7d59a3e25e6efeb36e2dc` |
| `tools/mx1_fire_guard.py` | `cbbad2371673210b20c932e8a3a87fdd9972c2bf492b494068afbcf62f20b1b6` |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `faf45fce38d9062f38b3979f182dc5dc2118127a0e5486e0d3c6948a1fe015d6` |
| `.omx/research/ddm_m1_20260808/sigma/sigma_harvest_receipt.json` | `bfdd921982eef458b90c53567bb60abffacff68a558f7a31cee838629663fe6d` |
| `.omx/research/ddm_m1_20260808/run/n120_metal/mem_probe/mem_probe_receipt.json` | `91ad0bee7e16827205b5baff82de9087b261aec74df49f01f7e377cb59709ef9` |
| `.omx/research/ddm_m1_20260808/run/fp32_mem_probe/mem_probe_receipt.json` | `12efb06fa41423f77e82beb5935375fb7eaf202b077264d2ee53fc66e5ccccd1` |
| `.omx/research/ddm_gc21_20260808/GC21_CONVOCATION.md` | `15f6d2febc23e7eb779ebaa93d902d7470aec612a9a4c6bba54cd9f6de1d06ee` |
| `.omx/research/ddm_ng1_20260808/NG1_CROSSWALK.md` | `26f76a4496ad55a2a15af69889f0a60d28a724c372c4f94ea59375b96ce28845` |

Read-time repository HEAD: `0259e8904a7ad3d880ddd9bb3c1b1482c4a29f91`.
Ticket source head at compose: `d14391b1d4`
(`launch_ticket_v5_event_driven.json:767-768`).

Frontier boundary: own-vehicle pointer remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
`0.19108` remains borrowed and unmoved (`main_hot_state.md:5-17`).

## Recall Evidence

| scope | result |
|---|---|
| Governing files | `PROGRAM.md`, the local agent instructions, the operating manual, and the common contract were read before artifact work. No protected file was edited. |
| Live authority | The hot-state pointer line was read (`main_hot_state.md:5-17`). The retained live-process prose was older than the receipts, so gate truth came from the ticket and seal orchestrator. |
| GC21 | GC21 defines the event predicate: `objective_S = 100 * d_seg_batch_mlx`, `one_sample_flip_S = 100 / (N * H * W)`, and `marginal_bar_S_per_step = one_sample_flip_S / eval_every_steps` (`GC21_CONVOCATION.md:107-161`). |
| NG1 | NG1 was used as the sigma-methodology antecedent only; no NG1 constants were imported as M1 constants. |
| Cap-censoring genus | OD2 records the open cap-bound blocker: 29/32 best-at-cap and 25/32 `safety_bound_REPORTED` (`OD2_STAGE12_RECEIPT.md:17-25`). OD3 records the clean contrast: 32/32 rows stopped before the derived ceiling and 0/32 `safety_bound_REPORTED` (`OD3_TERMINALITY_RECEIPT.md:3-21`). |
| Source inspection | The guard tuple, orchestrator, entrypoint guard, trainer flag consumers, sigma harvester, and trajectory stopper were inspected at the pinned hashes above. |

Finding that changed the recommendation: M1R2-F1. The stop-threshold arithmetic
is not a stylistic issue; the ticket embeds the threshold consumed by
`TrajectoryStopConfig`, so the numeric mismatch is a predicate drift.

## Q1 - Guard Versus Ticket Argv Bijection

Verdict: `FINDING` due M1R2-F2.

Clean part: the launch path is ticket-bound. The M1 burn argv is recorded under
`argv_m1_n120_cap_saturated` with `--steps 3250`, `--seed 20260808`,
`--checkpoint-every 250`, `--eval-every 50`, guard path, ticket path, and argv
key (`launch_ticket_v5_event_driven.json:60-123`). The orchestrator runs ticket
commands verbatim for train/verdict gates and never auto-executes FIRE
(`ddm_seal_orchestrator.py:448-477`). The GPU `mlx-train` entrypoint refuses
without `--fire-guard-verdict`, `--launch-ticket-path`, and `--fire-argv-key`,
then re-runs `evaluate_guard` in-process and checks the verdict identity
(`ddm_mx1_pr130_semantic_renderer.py:3822-3897`).

Finding: the receipt-equivalence tuple is narrower than the consumed training
surface. `_parsed_fire_config` records many core flags but not `steps`, `seed`,
`eval_every`, `checkpoint_every`, `run_dir`, or `resume_from`
(`mx1_fire_guard.py:140-173`). `_receipt_config` has the same omission
(`mx1_fire_guard.py:176-219`), and `_validate_config_match` compares only the
listed tuple plus `lr`, `ce_fraction`, `softplus_fraction`, and `mem_budget_gb`
(`mx1_fire_guard.py:310-342`). The trainer consumes the omitted values:
`seed` selects pair ids (`ddm_mx1_pr130_semantic_renderer.py:2774-2776`),
`steps` enters the config and LR/loss schedule
(`ddm_mx1_pr130_semantic_renderer.py:2781-2788`,
`ddm_mx1_pr130_semantic_renderer.py:2939-2943`,
`ddm_mx1_pr130_semantic_renderer.py:2981-2988`), `resume_from` changes start
state and history (`ddm_mx1_pr130_semantic_renderer.py:2839-2850`),
`eval_every` controls d_seg rows
(`ddm_mx1_pr130_semantic_renderer.py:3090-3138`), and
`checkpoint_every`/`run_dir` control persisted checkpoints
(`ddm_mx1_pr130_semantic_renderer.py:3139-3170`).

Bounded non-findings: mode-specific/non-training flags such as `train_exact_path`,
`scorer`, `facet_*`, `coreml_*`, and `mem_probe_steps` are outside the M1 burn's
`mlx-train` semantic path. `out` is an evidence-path flag. The guard identity
flags are checked separately by the entrypoint.

## Q2 - Sigma Methodology

Verdict: `CLEAN`, scoped.

The sigma receipt is honest about scope and units. It reports axis
`[macOS-CPU advisory]` for d_seg rows and `[macOS-MLX research-signal]` for loss
rows (`sigma_harvest_receipt.json:1-2`), d_seg fp16=fp32
`0.0010835435655381944` with abs_delta `0.0` in d_seg units
(`sigma_harvest_receipt.json:3-10`), and all falsifiers as not fired
(`sigma_harvest_receipt.json:11-32`). The repeat proof is bit-identical fp16
checkpoint bytes over five same-seed runs (`sigma_harvest_receipt.json:44-70`).
The harvester implements the intended proof as a deterministic checkpoint-byte
argument, not as an unmeasured stochastic transfer (`ddm_seal_orchestrator.py:551-583`).

Scope boundary: the receipt explicitly says the measurement is at the
calibration horizon, not the full burn horizon, and that the in-loop fp16
fallback guard remains live for the burn (`sigma_harvest_receipt.json:75-78`).

## Q3 - fp16 Admission

Verdict: `CLEAN`.

The fp16/fp32 d_seg equality is not a cache collision or same-checkpoint replay.
The fp16 CPU-torch verdict reports d_seg `0.0010835435655381944` and checkpoint
sha `56047d059595b36887a77b2940ebfd15f607413ee82cbd09f2eb946e50eba55c`
(`dseg_fp16/verdict_result.json:14-16`,
`dseg_fp16/verdict_result.json:63-71`). The fp32 CPU-torch verdict reports the
same d_seg and a different checkpoint sha
`9f5ec7ef3ee5cb6b376e1cfbc201a9ffd950870f2969c42c3a08884d06236302`
(`dseg_fp32/verdict_result.json:14-16`,
`dseg_fp32/verdict_result.json:63-71`). The shared input/target cache is
intended GT cache reuse; both verdicts record selected n120 cache loading and
CPU-torch axis (`dseg_fp16/verdict_result.json:17-62`,
`dseg_fp32/verdict_result.json:17-62`).

## Q4 - Stop Predicate And 3250 Safety Cap

Verdict: `FINDING` due M1R2-F1.

The 3250-step value is correctly written as a safety cap, not convergence. The
ticket states that `--steps 3250` is the per-run safety bound and forecast input,
not a stop rule, and that safety-bound hits queue/resume rather than converge
(`launch_ticket_v5_event_driven.json:769-793`). The stopper code preserves that
distinction: `safety_bound_REPORTED` sets `bound_reported=True` and is separate
from `converged_projected` and `marginal_below_bar`
(`trajectory_stopping.py:566-623`). This also matches GC21's cap rule
(`GC21_CONVOCATION.md:145-156`).

The numeric threshold is wrong. GC21 requires:

```text
one_sample_flip_S = 100 / (N * H * W)
marginal_bar_S_per_step = one_sample_flip_S / eval_every_steps
```

with `N=120`, `H=384`, `W=512`, and `eval_every_steps=50`
(`GC21_CONVOCATION.md:113-140`). Re-derived:

```text
100 / (120 * 384 * 512) = 4.238552517361111e-06 S
4.238552517361111e-06 / 50 = 8.477105034722223e-08 S/step
```

The ticket instead records `one_sample_flip_S = 4.4228e-06` and
`marginal_bar_S_per_step = 8.8456e-08`, and passes that latter value into
`TrajectoryStopConfig(marginal_score_gain_per_compute=8.8456e-8)`
(`launch_ticket_v5_event_driven.json:772-789`). Since
`TrajectoryStopConfig` validates and then consumes this field directly
(`trajectory_stopping.py:49-67`, `trajectory_stopping.py:583-620`), this is an
actual predicate mismatch. It is stricter by about 4.34 percent, so it is not a
fake improvement, but it is not the GC21/ticket-derived predicate.

## Q5 - Memory Projection And Step-Count Growth

Verdict: `CLEAN`, with explicit boundary.

The 16.0 GiB projection is defensible for the measured immediate saturated
configuration: the ticket records measured peak `8.493787 + 2.021515 =
10.515302 GiB`, multiplied by 1.5 to `15.772953`, rounded to `16.0 GiB`
(`launch_ticket_v5_event_driven.json:691-702`). The fp16 receipt reports
software-cap enforcement, required-stage sample, and peak values
(`mem_probe_receipt.json:25-34`, `mem_probe_receipt.json:46-99`,
`mem_probe_receipt.json:225-235`). The fp32 mem probe is larger at
`13.707321 + 2.019913 GiB` but belongs to the fp32 reference route
(`fp32_mem_probe_receipt.json:225-235`), not the fp16 first burn.

Unmeasured-but-bounded growth channels: checkpoint files accumulate on disk at
`--checkpoint-every 250` and final step (`ddm_mx1_pr130_semantic_renderer.py:3139-3170`);
history rows grow once per eval row
(`ddm_mx1_pr130_semantic_renderer.py:3079-3090`,
`ddm_mx1_pr130_semantic_renderer.py:3138`); long-run
allocator fragmentation can only be proven by a longer probe or the live
software cap. These are not shown to defeat the 16.0 GiB admission, but they
remain burn-monitor responsibilities rather than clean n3250 memory proof.

## Q6 - Freshness And Fail-Closed Behavior

Verdict: `CLEAN`.

The guard has a six-hour freshness window, validates receipt freshness, and
returns failure on stale receipt (`mx1_fire_guard.py:23-27`,
`mx1_fire_guard.py:287-301`, `mx1_fire_guard.py:355-386`). The seal orchestrator
treats stale mem-probe receipts as `PENDING` with a re-probe reason
(`ddm_seal_orchestrator.py:288-314`) and derives review/guard/FIRE dependencies
from the ticket (`ddm_seal_orchestrator.py:127-260`). A stale receipt therefore
blocks or pends; it does not silently pass.

## Q7 - Additional Adversarial Checks

Verdict: `NO ADDITIONAL FINDINGS` beyond M1R2-F1 and M1R2-F2.

Seal-orchestrator dry status, without execution, currently reports mem-probes,
sigma runs, d_seg verdicts, and sigma harvest satisfied; review counter remains
1/3; burn guard has no verdict on disk; FIRE is manual/held. This review did not
write any guard verdict, run Metal, mutate the ticket, or launch a scorer job.

The local targeted test suite for the seal orchestrator passed:

```text
.venv/bin/python -m pytest -q tools/tests/test_ddm_seal_orchestrator.py
30 passed in 0.22s
```

The non-mutating fresh guard evaluation over the seven existing train keys
passed while receipts were fresh:

```text
argv_sigma_fp16_run1 through argv_sigma_fp16_run5: passed fire_guard_passed
argv_sigma_fp32_ref: passed fire_guard_passed
argv_m1_n120_cap_saturated: passed fire_guard_passed
```

These checks do not override the findings above. A clean pass cannot be counted
until the stop arithmetic is corrected and the guard-equivalence gap is resolved
or explicitly scoped.

## Final Recommendation

`FINDINGS_RESET_COUNTER`.

Do not fire this ticket as sealed. Reseal with the GC21-derived threshold:

```text
one_sample_flip_S = 4.238552517361111e-06
marginal_bar_S_per_step = 8.477105034722223e-08
```

Then either extend the fire-guard receipt-equivalence tuple to cover the omitted
consumed `mlx-train` flags, or record a narrow formal reason each omitted flag
is outside the mem-probe equivalence claim. After that, restart the consecutive
independent review counter from 0/3.

No frontier movement: own-vehicle pointer remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
`0.19108` remains borrowed and unmoved.
