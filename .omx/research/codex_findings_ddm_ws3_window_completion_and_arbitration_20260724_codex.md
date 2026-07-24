---
research_only: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
promotion_eligible: false
main_review_required: true
---

# Codex findings: DDM WS3 window completion and arbitration

## Verdict

The registered falsifier selects `W_joint`, but this branch is
`BLOCKED_NEEDS_SCORER_RECURSIVE_PROPOSAL_CONSTRUCTION`, not fire-ready. The
selection is an INSTANCE arbitration. W_seg's stop is scoped to the reformed
seg-lexicographic opening FORMULATION under campaign acceptance. No family or
paradigm is closed.

The terminal blocker comes from the later operator directive at
`2026-07-24T14:45:16Z`: generic proposal-menu results must be labeled
`[naive-menu upper bound]` and cannot authorize firing until construction is
derived from the recursive scorer factorization. FIRING remains with MAIN.

## Custody and execution

- Delegated authority SHA-256:
  `b36872d5ce619dc741cf619542878dd87ff244defb94a03eeb16a68fe1bbd6c7`.
- Threads were pinned to 4. Bulk artifacts were written to
  `/Volumes/VertigoDataTier/pact`; no paid, remote, GPU, E4-adapter, or
  contest-axis execution occurred.
- W_seg fresh memory receipt SHA-256:
  `dab4be10bc1fcb30e50500d611f4132e23dd1ff0a0f9f874a4e991c16af32463`;
  measured peak `16.9360046387 GiB`, projected `21.3232055664 GiB`, admitted.
- W_joint fresh memory receipt SHA-256:
  `14bde86fe3c597a34b1c475f5962c7e4256d97a64100ab98b33594abbecf770b`;
  measured peak `16.9584503174 GiB`, projected `21.3501403809 GiB`, admitted.
- Current source custody:
  launcher `fbcd0f1cc6e7c61e6a5db87ab0094c4a756ac30105e8bac1a1f8c321941da93c`;
  consumer `06c4520dc66251937d9dc4dd2cac09828c79e4ec54f4e85126a8b836f371db6f`.

## W_seg reformed opening

Exact full-run receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_ws3_w_seg_reformed_four_step_abort_shrink_20260724T122200Z/full_run_receipt.json`,
SHA-256 `e367f60c8d8a64ba8b5aa8710bbe0f8115f37be8664eb2520f7d6081c09d0ab3`.

Stage-0 exact state was `d_seg=0.024124510023328993`,
`d_pose=146.36493245487776`, `138031 B`, archive
`264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9`.
The exact receiver-visible proposals at the first effective rung were:

| proposal | Seg term | Pose term | Rate term | joint delta S | disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| y- | +0.015455457899305608 | +0.028469392927583215 | +0.000001997576859367 | +0.04392684840374819 | reject |
| local exact gradient | n/a | n/a | n/a | n/a | structural `REJECT_AND_SHRINK`: G1 movable polygon escaped scorer geometry |
| x+ | +0.002464294433593764 | -0.002892069938759789 | -0.000001997576859367 | -0.000429773082025392 | reject: d_seg component regression |
| x- | +0.050560845269097265 | +0.014136416160212661 | -0.000001331717906244 | +0.06469592971140369 | reject |

The x+ proposal removed priced debt but added 2,907 exact Seg errors; exact
campaign acceptance therefore rejected it. Nominal smaller ladder rungs did
not produce distinct receiver states after realization, so they are not
claimed as effective quarter-quantum measurements. The preserved rollback is
global step 0. The receipt's legacy umbrella string
`BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER` is less precise
than the decomposition: one pure-priced descent existed but failed the
component gate. The exact proposal receipts and
`latest_realized_stage_decision=BLOCKED_REALIZED_DSEG_REGRESSION` are the
authority.

## W_joint exact history fill

Exact full-run receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_ws3_w_joint_exact_history_20260724T132200Z/full_run_receipt.json`,
SHA-256 `dca72cdc8c15a46be1a1d2e053bf813a79491d99c2793e57ec247f698be87e62`.

| exact step | d_seg | d_pose | bytes | archive SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.07051923116048177 | 36.618184751411334 | 138801 | `5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e` |
| 1 | 0.07030889723036024 | 36.48107420610205 | 138804 | `4487754bf1517946eb7b604817f99c5623ec0320aad3287edc67b436bae793f5` |
| 2 | 0.07030889723036024 | 36.48107420610205 | 138804 | `4487754bf1517946eb7b604817f99c5623ec0320aad3287edc67b436bae793f5` |
| 3 | 0.0702156745062934 | 36.40313537672988 | 138804 | `15e31a50f33ed5712c4616c35759b915397aade590d15f4efca28d998dee9e20` |
| 4 | 0.0702156745062934 | 36.37587755493872 | 138804 | `9601e777010b1dc45ed0841e118fcf34c58452324f8730fe9958a3440502e3a4` |

History `[0,1,2,3,4]` is exact. Step 2 is a measured receiver-identical
plateau, not a proxy. Distinct per-step checkpoints are preserved, including
the blocked step-2 checkpoint.

## Registered arbitration

The unchanged registered callable at `R*=4.1215446777965665` returns:

- `decision=KEEP_WJOINT`, `reason=SEG_REGRESSION`;
- observed W_seg ratio `1.1735894458608507`;
- predicted W_seg pose repayment `6611.801236822376` steps versus Seg-advantage
  exhaustion `1882.677674578264` steps;
- W_joint full-window distortion-term delta `-0.09377302357444606`.

The W_seg input is the exact terminal rejected proposal expressly allowed by
the preregistered formulation-stop path. The receipt does not pretend W_seg
completed four accepted steps.

Durable arbitration:
`.omx/research/ddm_ws3_warm_start_slope_arbitration_receipt_20260724.json`.

## Reseal and terminal re-smoke

The selected W_joint ticket was resealed to current source/config custody. Its
governed dry-run at
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_ws3_selected_w_joint_dryrun_20260724T145000Z/governed_dry_run.json`
was green with `projected_peak_gib=21.350140380859376`,
`score_claim=false`, and the pointer unchanged.

The final bounded exact re-smoke result is recorded in the machine-readable
readiness receipt landed beside this memo. Under the late operator directive,
that measurement is a `[naive-menu upper bound]`: exact negative action is
measurement evidence, not fire authority.

The selected y- proposal independently reproduced
`delta S=-0.05689051019463004`: Seg term
`-0.021033393012152846`, Pose term `-0.03585911475933656`, rate
term `+0.000001997576859366514`. Its exact receiver state is
`d_seg=0.07030889723036024`, `d_pose=36.48107420610205`, `138804 B`,
archive SHA-256
`4487754bf1517946eb7b604817f99c5623ec0320aad3287edc67b436bae793f5`.
The component and cumulative residual gates were green.

## Round-1 adversarial review

1. The first W_seg attempt exposed an uncaught structural G1 proposal failure.
   The launcher now records a deterministic structural rejection and
   abort-shrinks; a regression test covers the path.
2. Source-hash drift correctly invalidated the prior memory receipt after the
   hardening edit. Fresh memory preflight was required and obtained; custody
   was not weakened.
3. Nominal quarter rungs below the first receiver-visible quantum collapsed.
   This memo does not call them effective trials or claim a completed W_seg
   window.
4. Bounded exit code 4 classifications are not silently promoted to campaign
   failure or success; exact receipts and checkpoints are the evidence.
5. The arbitration consumes the registered terminal-proposal path and states
   its FORMULATION-vs-INSTANCE scope. It does not invent a substitute slope
   rule.
6. W_joint step 2 is explicitly exact even though it is numerically identical
   to step 1.
7. The W_seg full-run blocker string is an imprecise legacy umbrella. Changing
   executable bytes after measurement would break source custody, so the
   landed finding corrects interpretation rather than rewriting history.
8. The 14:45 UTC directive arrived after the main window measurements. It
   was already present when this arm launched the terminal re-smoke but had not
   yet been consumed. The bounded run was retained only as an upper-bound
   measurement. The directive supersedes a fire-ready reading: the generic
   coordinate/menu construction must be replaced with scorer-recursive,
   resize-footprint and stem-lattice-derived construction.

## Directive consumption

| UTC | source | directive | disposition |
| --- | --- | --- | --- |
| launch | delegated authority | cure both instance blockers, run registered falsifier, reseal and one bounded re-smoke; no firing | consumed; exact evidence and scoped arbitration landed |
| 2026-07-24T14:45:16Z | operator broadcast | candidate construction must derive from recursive scorer analysis; generic menus are naive upper bounds | consumed; terminal readiness fails closed and scorer-derived replacement is named |

No applicable per-arm directive was present through the last checkpoint.

## Verification

The frozen focused suite passed cleanly three times:

- pass 1: `48 passed in 92.06s`;
- pass 2: `48 passed in 91.51s`;
- pass 3: `48 passed in 94.49s`.

Ruff, JSON parsing, and `git diff --check` were green. One attempted pass
between passes 2 and 3 was discarded after it hit the repository's 60-second
timeout in an unchanged archive parser while the exact scorer process held
CPU; it produced no assertion failure and was rerun clean after the bounded
scorer process exited.

## Required MAIN review

MAIN must review the full branch diff and independently verify:

1. source/ticket/memory/full-run SHA closure;
2. the structural abort-shrink behavior and its tests;
3. exact W_joint history `[0,1,2,3,4]`;
4. registered-callable inputs and FORMULATION-vs-INSTANCE scope;
5. the late directive's fail-closed effect on readiness;
6. the required scorer-recursive replacement before any fire decision.
