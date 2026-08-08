VERDICT: FINDINGS_RESET_COUNTER

# M1R4A mechanics review — provenance stop

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source-provenance inspection; scorer-free review]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run_by_this_review=0`,
`metal_runs_by_this_review=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer first

This pass is void and the review counter remains `0/3`. The four content pins match, but the
charter's fifth provenance pin does not: it requires HEAD `1381ac84cb`, while read-time HEAD was
`fbce171a5a8599e6e3c5ebb725f351965123dfd7`. The charter makes the frozen state binding, requires
a stop on a differing pin, and identifies the four file hashes plus HEAD as the provenance pins
(`CHARTER_r4a_mechanics.md:35-45`, `108-117`). I therefore did not inspect the sealed file
contents or adjudicate A1-A6.

## Provenance re-verification

| pin | required | read-time value | result |
|---|---|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `9c8373b5b352cacc` | `9c8373b5b352cacc2456a21eac0deb53e32f445eb942e4675043825a1d896500` | MATCH |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` | MATCH |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `1ef18faf37e2f171` | `1ef18faf37e2f171d480b4e8073c453185f9ae00a1b3200b46d5bb258cd60895` | MATCH |
| `tools/ddm_seal_orchestrator.py` | `e592cb36fb00d502` | `e592cb36fb00d502693cf17ef43da0f01c7f7c7aecc7d59a3e25e6efeb36e2dc` | MATCH |
| repository HEAD | `1381ac84cb` | `fbce171a5a8599e6e3c5ebb725f351965123dfd7` | **MISMATCH** |

Read-time `git show -s` established that `fbce171a5a` is a direct child of `1381ac84cb` and
adds the three round-4 charter files. A bounded `git diff` found no differences in the four
pinned files between those commits. That does not satisfy the charter as written: the HEAD pin
is separately repeated in the OPTIMAL FORM provenance clause (`CHARTER_r4a_mechanics.md:108-117`),
and the concurrency clause requires every pass to review the same frozen state
(`CHARTER_r4a_mechanics.md:16-19`).

## RECALL EVIDENCE

| scope | query / source | found beyond the charter seeds | plan impact |
|---|---|---|---|
| Prior M1 reviews | Content search for `M1R2-F1`, `M1R2-F2`, `mx1_fire_guard`, `one_sample_flip_S`, and `_EQUIVALENCE_EXCLUSIONS`; opened `ddm_m1r2_20260808/M1R2_REVIEW.md` and `ddm_m1r3_20260808/M1R3_REVIEW.md`. | Round 2 really reset the counter with the stale-threshold and incomplete-equivalence findings (`M1R2_REVIEW.md:9-25`); round 3 explicitly says a cure voids concurrent clean passes (`M1R3_REVIEW.md:8-18`). | Preserved the counter reset; no prior clean receipt licenses ignoring a new provenance mismatch. |
| Guard genus | Content search found RR10's stale/forged-verdict bypass and RR11's later in-process revalidation (`ddm_rr10_20260807/ROUND10_FINDINGS.md:48-86`; `ddm_rr11_20260807/ROUND11_FINDINGS.md:95-109`). | The guard has prior silent-bypass history, but those are cure antecedents, not authority to bypass this review's stop gate. | Kept the finding narrowly scoped to the round-4 provenance instance; did not import an old guard finding as a current-source verdict. |
| Equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` filtered for `mx1`, `fire`, `seal`, `resume`, `checkpoint`, `memory`, and `peak`. | No equation-registry result found in that filtered scope overrides the charter's explicit SHA/HEAD stop rule. | No change: provenance must be repaired before mechanics adjudication. |
| Research index / DAG / ledger | Content search over `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and `operator_p0_ledger.jsonl` for `mx1`, `M1R2`, `launch_ticket_v5`, `fire guard`, and `seal orchestrator`. | No scoped result found an exception permitting a moving live HEAD under a frozen-pass claim. | No change: stop rather than infer an exception. |
| Live board | `.omx/state/main_hot_state.md`. | The live board says M1 is cured, the counter is `0/3`, and three fresh passes are running (`main_hot_state.md:22-28`). | A false clean would advance a live fire gate, so the mismatch is launch-blocking. |

## Finding M1R4A-F1 — LAUNCH-BLOCKING

**Verdict scope:** INSTANCE — the round-4 review-provenance contract, not the M1 mechanism family.

**Mechanism.** The charter declares a live repository HEAD value as part of the immutable review
subject while the charter itself is introduced by the next commit. At read time, the charter-only
commit `fbce171a5a` had parent `1381ac84cb`. Thus the review cannot simultaneously read the committed
charter and observe the stated live HEAD. In a shared worktree, later sister-receipt commits would
move live HEAD again even when every sealed subject file remains byte-identical. This makes live HEAD
an unstable freeze key and defeats the stated same-frozen-state predicate.

**Exact failure scenario.** A reviewer either (1) follows the stop rule and every pass is void before
A1-A6, or (2) silently reinterprets the pin and may issue `CLEAN_PASS` without satisfying the written
provenance contract. Under the concurrency rule, option (2) can let separately based reviews count as
one clean sequence (`CHARTER_r4a_mechanics.md:16-19`, `35-45`, `108-117`).

**Smallest correct cure.** MAIN should replace the moving-live-HEAD assertion with an immutable
`review_subject_commit`/tree pin (here, apparently `1381ac84cb`) and state explicitly that charter-only
and receipt-only descendant commits are outside the reviewed subject. The resealed charter must also
pin or hash every transitive dependency admitted by its bounded scope. Then restart all three passes
against that same immutable subject. Merely changing the required live HEAD to `fbce171a5a` is not a
durable cure because the first sister receipt commit moves it again.

Follow-on disposition: **QUEUED-WITH-A-FIRE-ORDER** — MAIN repairs and reseals the provenance clause;
only then rerun R4A/R4B/R4C from pass 1 with counter `0/3`.

## A1 — fire argv consumers

`UNDETERMINED`. A1 requires a full flag-to-`add_argument`-to-consumer trace
(`CHARTER_r4a_mechanics.md:51-53`). The earlier provenance stop (`:44-45`) forbids reading the
sealed content for this pass. Needed: a corrected immutable review-subject pin, then a fresh full-depth
trace over the re-pinned ticket and trainer.

## A2 — probe/fire peak-memory equivalence

`UNDETERMINED`. A2 requires every exact/float key and every exclusion reason to be re-derived
(`CHARTER_r4a_mechanics.md:55-61`). Needed: the corrected immutable pin and a fresh source trace of
both argv configurations through their memory-bearing consumers.

## A3 — flag-classification gate coverage

`UNDETERMINED`. A3 requires adversarial analysis of alternate argv spellings and wrappers
(`CHARTER_r4a_mechanics.md:63-66`). Needed: the corrected immutable pin, followed by static source
inspection and bounded non-writing parser exercises against that subject.

## A4 — resume and per-stage checkpoint durability

`UNDETERMINED`. A4 requires proof of distinct stage-encoded, EMA-bearing, atomic, byte-close-loadable
checkpoints for the actual fire config (`CHARTER_r4a_mechanics.md:68-72`). Needed: the corrected
immutable pin and a fresh source/config trace. No loop or checkpoint path was inspected in this pass.

## A5 — measured saturated memory preflight

`UNDETERMINED`. A5 requires the real-scale saturated mem-probe path and measured-peak receipt consumer
to be traced (`CHARTER_r4a_mechanics.md:74-77`). Needed: the corrected immutable pin and a fresh
ticket/guard/trainer trace. No memory formula or receipt was accepted as evidence in this stopped pass.

## A6 — seal-orchestrator fail-closed behavior

`UNDETERMINED`. A6 requires malformed and missing receipt paths to be re-derived from orchestrator
source (`CHARTER_r4a_mechanics.md:79-80`). Needed: the corrected immutable pin, source inspection, and
bounded non-writing tests on that subject.

## Lens boundary

This pass covered only governance, content-hash verification, commit ancestry, and the provenance
stop mechanism. It did **not** review A1-A6 mechanics; science; arithmetic/authority; ticket semantics;
current guard correctness; checkpoint implementation; receipt contents; or orchestrator behavior.
It ran no Metal job, scorer, guard invocation, burn, or new measurement and modified none of the four
sealed files. The exact contest pointer did not move.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`
(`.omx/state/main_hot_state.md:5-17`).
