# ddm_fa1 Next If Resumed

status: QUEUE-DISPOSITION-ONLY
budget: USD 0 unless a future charter explicitly changes it

## Resume Boundary

Resume this arm only as a paper-crosswalk follow-on. Do not start training, remote dispatch, full-n600 scoring, archive construction, or GPU work from FA1 alone.

Before any code touch:

1. Re-read `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, and the current trainer/DSL files.
2. Check the live lane/serializer state and staged index.
3. Reconfirm no protected path from the common contract is in the edit set.
4. If a scorer or launch is proposed, obtain a new charter; FA1 does not own a scorer slot.

## Follow-On Dispositions

| ID | Disposition | Fire order |
|---|---|---|
| fa1.F1 | QUEUED-WITH-FIRE-ORDER | Build a design-only or default-off `$0` replay for `StageTransitionSoftVelocityBlend` after confirming the current trainer reset owner. First reuse or capture one stage-boundary checkpoint plus deterministic mini-batch/gradient sequence. Then compare against existing `v <- v_prev` and bias-corrected reset controls at matched update RMS. Report effective-LR spike, descent alignment, and component replay metrics. Do not run a scorer or continue training. |
| fa1.F2 | FOLDED | Direct FlowAdam optimizer import is folded. The paper limitations and lack of Pact-local update/curvature custody block adoption. |
| fa1.F3 | QUEUED-WITH-FIRE-ORDER | Add an observer-only replay comparing the FlowAdam EMA difficulty detector to existing Pact event gates, but only after at least 10 logged comparable trajectories are available. It may report prediction quality; it must not control the schedule. |
| fa1.F4 | QUEUED-WITH-FIRE-ORDER | For coupled-parameter regimes, first collect or locate local gradient/curvature receipts for one named block: QA83 head, QA84 rowband, FiLM/rank-collapse surface, sc1 rank-1 `e_p`, #140 pose codec, or a coupled JD1-style finisher. Only then design the smallest update-RMS-matched replay. |
| fa1.F5 | FOLDED | Implicit-regularization rhetoric remains lesson-only. It becomes actionable only if a later arm names a DE-derived consumer and a receiver/scorer-measured falsifier. |
| fa1.F6 | FOLDED | Paper benchmark percentages are not Pact forecasts and must not be converted to contest-score projections. |

## Minimal Packet To Carry Forward

- `RECEIPT.md` contains the rigor-triage grades and recall evidence.
- `CROSSWALK.md` contains the ranked adoption table and the single highest-value row.
- No code was changed, so a future implementation arm must create fresh base/post shas for any touched file and commit through `tools/subagent_commit_serializer.py`.

## Own-Vehicle Frontier Line

FA1 moved no pointer. The live hot-state line observed here remains: `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; the contest pointer remains borrowed and unmoved until an exact receiver-closed archive under our custody beats it.
