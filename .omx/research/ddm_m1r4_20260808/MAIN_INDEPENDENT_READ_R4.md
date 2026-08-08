# MAIN's independent read, SEALED before the round-4 verdicts land

**Purpose.** I read the sealed ticket's own numbers while the three round-4 arms were running and
formed a view. Writing it down NOW, before their receipts exist, so that when they land I am
comparing two independent derivations instead of agreeing with theirs after the fact. This file is
NOT in any charter's pinned scope and was NOT shared with the arms.

Written 2026-08-08, after spawning `ddm_m1r4a/b/c`, against ticket sha `9c8373b5b352cacc`,
HEAD `1381ac84cb`.

---

## R0-M1 — the σ gate's falsifier fires in the units it is written in, and does not fire in the unit that matters

The ticket carries BOTH measurements. Read literally, they disagree about whether the seal passes.

**Falsifier 2 as written** (`sigma_calibration.seal_falsifiers[1]`):

> fp16 vs fp32 delta > max(2.0e-6, 3*sigma) OR event classification flips inside the envelope
> -> fall back to fp32

**The measured inputs:**

| quantity | value | unit |
|---|---|---|
| `sanity_sigma_measured.sigma` | `0.0` | training_loss |
| `sanity_sigma_measured.n` | 5 | runs |
| `fp16_fp32_delta_measured.abs_delta` | `2.0381150534376502e-05` | **training_loss** |
| `fp16_fp32_delta_measured.delta_in_sigma` | `None` | — |
| `dseg_unit_measurement.abs_delta` | `0.0` | **d_seg** |
| `dseg_unit_measurement.fp16` | `0.0010835435655381944` | d_seg |
| `dseg_unit_measurement.fp32` | `0.0010835435655381944` | d_seg |
| falsifier bar `2.0e-6` | described in `seal_falsifiers[0]` as "the gc21 fp16 guard bar" | **d_seg** |

**Arithmetic.** `3*sigma = 0`, so the bar reduces to `max(2.0e-6, 0) = 2.0e-6`.
`2.0381e-05 > 2.0e-6` by **10.19×**. Taken at face value the falsifier is **MET** and the
prescribed action is **fall back to fp32** — which would change the fire config.

**But the comparison is a unit mismatch.** `abs_delta` is `metric: final_step_training_loss`;
the `2.0e-6` bar is introduced one line above as a **d_seg** bar. In d_seg — the unit the score is
actually denominated in — fp16 and fp32 are **bit-identical**: `abs_delta = 0.0`, both `0.0010835435655381944`.
So the fp16 training-loss difference does not propagate to the scored quantity at all.

**My read (DERIVED, to be checked against the arms):** the seal is substantively fine and fp16
stands, because the falsifier's *intent* is "does precision change the SCORE" and in the scored
unit the answer is exactly zero. But the falsifier's *text* compares a training-loss delta to a
d_seg bar, so as literally written it fires. Two defects follow, and they are both real:

1. **UNIT-MISMATCHED FALSIFIER** (MEDIUM). The predicate cannot be evaluated as written. It must
   either name the d_seg quantity explicitly (then it passes, 0.0 < 2.0e-6) or carry its own
   training-loss bar (which nobody has derived). Cure = make the falsifier name its metric, and
   evaluate it against `dseg_unit_measurement`, not `fp16_fp32_delta_measured`.
2. **`delta_in_sigma = None` HID IT** (MEDIUM, and the more interesting one). That field is the
   one a reader would scan to see whether the delta is significant. It is `None` because σ=0 makes
   the ratio a division by zero. So the single field designed to surface this comparison silently
   rendered nothing, and the gate list was never updated. **A derived comparison that cannot be
   computed must not degrade to `None` — it must degrade to a REFUSAL or an explicit
   `UNDEFINED_BECAUSE_SIGMA_ZERO` token.** Silent `None` is the vacuity genus (`skip == green`).

## R0-M2 — σ = 0 is a determinism receipt, not a noise measurement

All five repeat runs produced **bit-identical checkpoints** (`distinct_sha_count = 1`,
sha `56047d05…`), so `sigma = 0.0` exactly and `relative_sigma = 0.0`.

This is honestly scoped in the ticket (`scope: same-seed backend nondeterminism floor`) and it is
a genuinely good result — it proves the Metal backend is bit-deterministic at this config, which
means any change observed during the burn is signal, not backend jitter. **It does not, however,
measure what a stopping rule needs**, which is the variation of the descent estimate itself. With
same-seed determinism, five repeats of the same seed are five copies of one sample; the effective
n for any noise question is **1**, not 5. Whether that matters depends entirely on whether the
stopping rule's window can be fooled by within-run structure (plateau-then-drop), which is exactly
lens B's B1. I expect B to reach this independently; if it does not, that is a gap in B.

## R0-M3 — `seal_gates_remaining` is stale, and it is the frozen-literal genus again

`seal_gates_remaining` reads:

```
[0] DONE 2026-08-08: mem-probe PASSED ...
[1] sigma calibration: 5x repeated ... -> sanity_si...
[2] independent review pass 2 of 3 (codex arm, sealed-ticket scope)
[3] independent review pass 3 of 3 (codex arm)
```

Two staleness defects, both LOW severity but both the same shape as M1R2-F1:

- **[1] lists σ calibration as REMAINING** while `sigma_calibration.sanity_sigma_measured` is fully
  populated (n=5, per-run values, per-run checkpoint shas). The status string is a literal; the
  evidence that would derive it sits in the same file.
- **[2]/[3] say "pass 2 of 3" and "pass 3 of 3"** — that numbering was true when round 2 was
  expected to be the second clean pass. Round 2 returned findings, the counter reset to 0, and
  three FRESH passes are now required. `review_passes` was reset to `[]` in the amendment but this
  human-readable list was not.

**Cure (structural, not textual):** `seal_gates_remaining` should be DERIVED at read time from the
evidence fields — a gate is remaining iff its evidence is absent — and `review_passes_required = 3`
should be compared against `len(review_passes)`, not narrated. I am deliberately NOT editing it
while the arms hold the sha; it goes into the same amendment as whatever they find.

---

## Why I am not acting on any of this yet

Editing the ticket now would break the pinned sha under three live reviewers and void all three
passes — the exact failure I told them to report as finding #1. These go into ONE amendment after
their verdicts land, together with their findings, and then three fresh passes run against the
cured artifact. Same discipline as the M1R2 cure cycle.

## What I will check when their receipts arrive

- Does **C** find the unit mismatch independently? (C6 recomputes the arithmetic; it should.)
- Does **C** classify `delta_in_sigma = None` as a finding, or only note the literal?
- Does **B** reach the "n=5 repeats of one seed is effective n=1" point from B6?
- Does **anyone** catch the stale `seal_gates_remaining`? If all three miss it, that is a real gap
  in the round-4 lens set and I should say so rather than quietly adding it to the amendment.
- Any finding of theirs that I did NOT anticipate is the most valuable output of the round, and I
  should say plainly that I missed it.

---

# APPENDED (same turn, before any arm verdict landed): relative-significance debt paid

The magnitude-dismissal Stop hook fired on this document. It pointed at R0-M2, which I judge a
**false positive on location** — R0-M2 dismisses nothing; it routes the question to lens B's B1
with the deciding condition named. But the hook's *demand* was substantively right and unpaid
elsewhere: R0-M1 and R0-M3 both carried magnitude judgements ("does not propagate", "LOW severity")
with **no denominator**. Appending rather than rewriting so the pre-verdict state stays verbatim.

Operating point: S_current **0.7534578126155775** (tq1c, own-vehicle `[macOS-CPU advisory]`),
bar **0.1721417** (PR130 measured), **gap = 0.5813161126155775**.

## R0-M1 relative significance (this STRENGTHENS the finding, it does not soften it)

| quantity | value | as % of remaining gap |
|---|---|---|
| falsifier bar `2.0e-6` **d_seg** | 2.000000e-04 S | **0.034405%** |
| MEASURED fp16−fp32 delta, **d_seg** | 0.0 exactly | **0.00000000%** |
| the delta the falsifier text compares, **training_loss** | 2.038115e-05 | **no S conversion exists** |

So the precision choice is worth **exactly zero percent of the remaining gap in the scored unit** —
and that is a MEASUREMENT (bit-identical d_seg, `0.0010835435655381944` both sides), not an eyeball.
That is the un-recoverability citation the discipline asks for: there is no recoverable ΔS hiding
in fp16-vs-fp32, because the two produce the *same scored value*.

The third row is the actual defect and it is **not** a magnitude question at all: a training-loss
quantity has no conversion into S, so comparing it to a d_seg bar is undefined in either direction.
Severity is unchanged (MEDIUM); the cure is unchanged (make the falsifier name its metric).

## R0-M3 re-graded: "LOW severity" was the wrong axis, not the wrong value

`seal_gates_remaining` staleness has **ΔS = 0 by construction** — it is a human-readable status
string that changes no argv, no config, and no scored quantity. Grading it "LOW" implied it is a
small version of the same thing the other findings are. It is not; it is a **different currency**:
its cost is *review-integrity*, i.e. the probability that a reviewer or a future MAIN reads a gate
as open/closed against the evidence in the same file. That cost is not expressible in S and must
not be traded against S. Re-graded: **ΔS-NEUTRAL / REVIEW-INTEGRITY-RELEVANT**, cure unchanged
(derive the list from the evidence fields).

## What paying the debt actually found (the hook earned its keep)

Computing the bar in gap-relative terms exposed its derivation, which the ticket never states:

```
marginal_bar_S_per_step / one_sample_flip_S = 8.477105034722223e-08 / 4.238552517361111e-06
                                            = 0.02000000  EXACTLY  = 1/50
```

**The stopping bar is "one argmax flip per 50 steps."** That is a derived law, not a tuned constant
— and it explains why M1R2-F1 mattered so much: the bar is a fixed 1/50 of `one_sample_flip_S`, so a
stale flip constant propagates into the stopping threshold with gain 1/50 and nothing downstream
would reveal it. The chain is n=120 → `one_sample_flip_S` → ÷50 → bar. **The ticket should state
that ÷50 and its derivation; if it is a chosen constant rather than a derived one, it is a
BORROWED constant and lens C should catch it.** I did not find its derivation in the ticket.

Two more gap-relative facts for lens B's B1/B4, stated so the arms can check me:

- one flip at n=120 = `4.238553e-06` S = **0.000729%** of gap.
- a full 3250-step safety-cap run sitting **exactly at** the stopping bar yields
  `2.755059e-04` S = **0.047393%** of gap. That is the stopping rule's *tolerance floor*, not a
  prediction — real descent starts well above the bar and decays toward it. But it bounds what the
  rule is willing to keep paying wall-clock for, and B4 should compare that against the burn's cost.

## Adversarial review of my own R0-M2 verdict (verdict_scope: INSTANCE)

R0-M2 claims five same-seed repeats give effective n=1 for a noise question. Attacking it: if the
backend is bit-deterministic, then σ_run-to-run is genuinely 0 and the stopping rule faces **no**
run-to-run noise — so "effective n=1" understates the result, because the quantity the rule needs
is not estimated from repeats at all; it is *identically zero by determinism*. The residual risk is
therefore NOT sampling noise but **within-run structure** (plateau-then-drop fooling a marginal-rate
window). That is exactly B1, and R0-M2's routing stands. Scope: INSTANCE — one document's grading,
not a verdict on the σ protocol, which lens B owns.
