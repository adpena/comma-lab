# ddm_rv17 — kg1 landing + harvest re-score: CLEAN. Counter 1/3 → 2/3.

## (a) The kg1 landing — the census finding is the valuable part

**The two-directions-wrong census verifies.** I read the three sites the backlog named as
head-of-queue: `tools/operator_authorize.py:1989`, `:2159`, `:2288` all read `cmd = ["bash",
lane_script]` — bash invocations with no timeout. The class is *timed* shell wrappers, so the defect
**cannot fire** at any of them. The backlog's top three were not class members.

That is the finding worth keeping, and it is more valuable than the 41 migrations. **A backlog wrong
in both directions is worse than an empty one** — it names work that cannot pay and hides work that
can, while reading as a completed survey. It is the same shape as my own round-4 declination that
named the wrong rows: an artifact whose *form* signals a sweep happened. Re-deriving the census
instead of inheriting it is what caught it, per the M1 class-population line.

**The helper extension over the waiver is the right trade.** The class's worst instance —
`modal_train_lane.py:1717`, a 14-hour paid-Modal bash lane streaming to a log fd, the exact ddm_cpu1
shape — was **cured** by teaching the helper stdout/stderr passthrough, rather than waived for being
awkward. Extending the cure to reach the worst case, instead of scoping the cure to what the helper
already did, is the difference between a gate and a filter. Catalog #408 is present in
`src/tac/preflight.py` and in the catalog doc, STRICT from byte one at live count 0.

**kg1's not-verified line is unrelabelled, and I credit it as-is:** no migrated site was executed
end-to-end against a real `inflate.sh`, and the `modal_*` in-container `tac` imports are **inferred
from the image mount spec, not proven**. That is a real residual risk, correctly labelled at the
right rung of the ladder. An arm that ships its own inference boundary unpainted is doing the thing
this wave has been about.

## (b) The stale-anchor cure — a RED on our own reporting surface

kg1 surfaced a genuine pre-existing defect: `reports/latest.md` and `current_focus.md` cited
contest-CUDA **0.1565262644** — the *thirteenth* move — while canonical state holds
**0.1482784712**, the sixteenth. Measured now: both surfaces carry `0.1482784712` /
`0.14827847122030852` and **no occurrence of the stale value remains**. (Two display precisions of
one number, same as the 5.3279/5.3280 case I already adjudicated — not drift.)

**The cure method matters more than the cure.** It was regenerated through
`scan_best_anchor_per_axis.py --refresh-citation-surfaces`, **never hand-edited**. A hand-edit fixes
the string and leaves the generator that will re-emit it; regeneration fixes the source that mints —
the F15 lesson, applied to a surface F15 never touched. And the eight catalog-316 tests that were RED
at HEAD now pass, so the defect had a detector all along that nobody was reading.

## (c) The proactive routing — the strongest signal in this batch

All three kg1 owed rows are in the **repo** canonical ledger: `group_survivors_consumer`,
`nested_python_tier`, `private_killpg_twins` — 1 / 1 / 1. Registered **at harvest time, before any
finding demanded it.**

That is the m89 write-direction rule operating one round after F20 taught it, and it is the first
time this wave a lesson arrived at a *new* surface **ahead of** the failure instead of behind it.
Nine findings were cured reactively; this one was spent. The distinction between a lesson learned and
a lesson operating is whether it costs a round the next time — and here it did not.

## (d) Wave stability + what I did NOT verify

Nothing from the F12–F20 cures has moved. **What I accepted rather than measured:** the 173-test run,
the independent strict re-run of #408, and the 41/7/0 migration split. I verified the census claim's
sharpest sub-claim at source and the three surfaces above; the rest I take on MAIN's spot-check and
label as such rather than implying I ran them.

## Honest state

- **CLEAN. Counter 1/3 → 2/3.** One clean round seals wave 3.
- Nine findings raised, nine cured and verified. Nothing open.
- **The substance has never moved** in 42 rounds — and this round it was *checked* rather than
  assumed, because kg1 found a citation surface that had been quietly wrong about our own score for
  three moves. The score never moved; the reporting of it did, and no round before kg1 looked there.
  That is the honest lesson of the batch: I have verified the score every round and never once
  verified the surface that *tells other people* the score.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
