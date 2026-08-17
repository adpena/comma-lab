# The q3q4 lever's contribution is within noise — the owed control is paid, and it de-confounds every ce1 number

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal). `Q3Q4OFF`, 3,000 steps, rc=0, 1,441 s.
**Axis:** `[macOS-MPS training-signal]` `quantized_exact_seg`, EMA shadow, n600, eval-batch 8.
**No score claim. Frontier untouched** (hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4]`).
Payload `/Volumes/APDataStore/pact/ddm_ce1/Q3Q4OFF/`.

## ANSWER FIRST

`--weight-qat-q3q4` (F2) was **ON in every `ce1` arm with no OFF control** — the defect FINDING 3 of
`ddm_drain_vehicle_split_and_lever2_payoff_20260817` named. The control is now run.

⚠ **This memo was corrected before landing**, after `tools/recall_neighborhood_check.py` flagged
`ddm_b2e_edit_replay_admission_verdict_20260816` as an uncited same-topic artifact. It had already
measured F2's *other* channel and it changes two claims below. See §"What b2e already settled".

Init = 2.861616347e-04 = **33,757 flips**.

| arm | `--weight-qat-q3q4` | endpoint flips | vs init | `best_step` |
|---|---|---:|---:|---:|
| `EF3000` | **ON** | 31,471 | **−2,286** | 3000 (at cap) |
| `Q3Q4OFF` | OFF | 32,034 | **−1,723** | 2800 (interior) |

**Lever contribution = −563 flips (ON better) = 0.66σ** against the 855-flip two-run band (605×√2).
**Not distinguishable from noise.** The point estimate favors ON; one seed cannot separate that from
scatter. In S units the contribution is **−4.772610e-04** seg — bounded, not established.

`packed_parameter_bytes` = **40,252 on BOTH arms.** The lever is byte-neutral by itself; it changes
the training-time grid, not the deployed packing. Wall clock 1,441 s vs 1,445 s (−0.3%).

## What this de-confounds

The worry was not that F2 confounded the *comparisons* — it was held constant across `EF3000`/`FRD077`
and across the whole allocation ladder, so those orderings were never at risk. The worry was that an
**unmeasured contribution was baked into every absolute `ce1` number**. That is now bounded:
|contribution| ≲ 855 flips. Small enough that no `ce1` conclusion moves.

**A control that finds nothing is still worth its 24 minutes** when the alternative is quoting numbers
with an unknown additive term inside them.

## The caveat, stated rather than buried

The two arms differ by exactly one argv token (verified by set-diff: `--weight-qat-q3q4`, plus the
per-arm output paths). But `train_semantic_quantized_resumable.py:1215-1229` branches on
`lever_config.any_active`, and `Q3Q4OFF` has **no** active lever, so it takes a different call:

| arm | `any_active` | render call |
|---|---|---|
| `EF3000` | True | `with levers.applied(model, base_bits=4): qat.render_float(...)` |
| `Q3Q4OFF` | False | `qat.render_quantized(model, ..., bits=4)` |

The trainer asserts these are equivalent-modulo-parameter-values in a **comment** (`:1223-1227`:
"The render tail … is identical in both helpers, so only the parameter values differ"). Per this
repo's own rule, a comment promising behavior is not a contract. So:

* **What the lever is designed to do** — swap uniform q4 for mixed q3/q4 (3 bits on
  `SELECTED_MIXED_Q3_NAMES` via `editability_levers.mixed_bit_allocation`, 4 elsewhere) — is real and
  wired (`:373`, `:505`). This is not a config orphan; I checked the consumers, not the argparse.
* **What is UNVERIFIED** — that the two render helpers agree bit-for-bit at identical parameters. If
  they do not, part of the −563 is path, not lever. Since the result is inside the noise band either
  way, this does not change the verdict; it would matter if a future arm reads the sign as real.

**The cheap cure** (not applied here, it is a training-path change): a byte-identity assert at
`base_bits` with `any_active=False` proving `levers.applied + render_float == render_quantized`. That
turns the comment into a contract and costs one test.

## Scope

**verdict_scope: INSTANCE** — `Q3Q4OFF` vs `EF3000`, one seed (20260715), the semantic renderer
(`blocks.{0..3}`, 66,339 params), the trainer's advisory `quantized_exact_seg`. **NOT** the `hv1`
frontier vehicle (37 tensors, 39,375 params, no FiLM — a different architecture, not a different
checkpoint), **NOT** byte-closed, **NOT** exact-eval'd.

## What b2e already settled — and the 138× my arms add to it

`ddm_b2e_edit_replay_admission_verdict_20260816` (2026-08-16, `[macOS-CPU advisory n600]`) ran an
**F2-alone window** and measured the *other* channel: does training with the mixed grid make the
mz2 edits cheaper to APPLY? Its bar asked each edit's pose-damage excess to collapse ≥50×.

| edit | verdict | collapse | required |
|---|---|---:|---:|
| `mixed_q3q4` | **REFUSED** | 0.945098 | 50.0 |
| `film_row_prune_keep87` | REFUSED | 1.058953 | 50.0 |
| `film_row_prune_keep75_minus_keep87` | REFUSED | 0.747891 | 50.0 |

So my draft's speculation — "the recode's distortion blocker is not obviously a training-time debt" —
is **replaced by a measurement**: it is not a training-time debt, and b2e proved it by trying. The
`mixed_q3q4` edit costs ~4× pose excess on BOTH the F2-trained base (4.188) and the calibration base
(3.959). Training through the grid does not buy the edit anything.

**And the 138× the two windows disagree by is the real new signal.** Same lever, same 3,000 steps,
same trainer:

| window | lr | `ce_fraction` / `softplus_fraction` | Δd_seg |
|---|---:|---|---:|
| b2e burn-2 F2-alone | 2e-7 | 0.5 / 0.85 | **+1.4e-7** (flat, slightly worse) |
| my `EF3000` | 2e-5 | **0.0 / 0.0** | **−1.938e-5** (−2,286 flips) |

Two variables differ (100× lr AND the objective allocation), so this pair does not attribute — but it
is exactly the shape task #1091 predicted (*"the seg wall is 92.7% CONFIGURATION"*) and #1089 named
(*81.19% of the LR budget goes to the worst-aligned objective*). b2e's own §open-question asks
literally this: *"does any lr / step budget move the burn-2 base d_seg or d_pose beyond the noise
floor"*. **Answer: yes — measured twice (−2,286 at 3k, −2,620 at 6k).** That question is now closed;
which of the two variables carries it is not.

`mz2`'s banked **−823 B mixed-q3/q4 recode** is a post-hoc recode of the frontier archive — a
different vehicle and a different mechanism from either window. Neither this control nor b2e prices
its RATE half; b2e priced its distortion half and refused it.

## Drain status (#1092)

| # | lever | status |
|---|---|---|
| 1 | `--film-row-dropout` | MEASURED seg-neutral (0.18σ); payoff path INSTANCE-blocked (frd077) |
| 2 | `--carrier-rank-penalty` | NOT FIRED — factorization channel negative by arithmetic at every fidelity |
| 3 | `--weight-qat-q3q4` | **MEASURED here — contribution within noise, byte-neutral** |
| 4–12 | perturb-robustness · perturb-shape · film-critical-multiplier · distill-weight/max-seg · film-row-dropout-protect-top · fixed-zero-mask | pending |

**Next:** `--weight-perturb-robustness` (F1) + `--weight-perturb-shape` + `--film-critical-multiplier`
are one coupled family (`f1_active = sigma > 0`), so they are one A/B, not three.
