# ddm_ec2 collateral-suppressed conditioner — retained-field result and no-fire seal

Date: 2026-08-22  
Task: `ddm_ec2_collateral_suppressed_conditioner`  
Verdict: **BUILT AND MEASURED; QUEUED-WITH-FIRE-ORDER, BLOCKED; NO FIRE**  
Axis: `[contest-CUDA T4 retained argmax fields, n600; macOS-CPU scorer-free analysis] COMPONENT-ONLY`

## Result

The EC1 harmful and beneficial populations are separable by a receiver-visible
decoded-semantic context, but not without substantial loss of beneficial mass.
Across five held-out pair folds, the selected 3,125-state oriented-context gate
kept **6,670 B and 1,017 H**, for **5,653 net flips** and **86.7699% B/(B+H)**.
Relative to the full EC1 endpoint, it retained **55.2381% of B** while retaining
only **1.9242% of H**. Thus H fell much faster than B: the charter's exact
“B and H are not separable” falsifier did **not** fire. The stronger prior-law
prediction did fail on this retained endpoint, because reaching the high
beneficial rate still removed **44.7619% of B**, close to the gross-benefit
shrinkage the charter forbade treating as success.

This is a cross-fit retained-field projection, not a realized gated candidate.
The built proposer consumes only decoded semantic context plus a counted gate;
it has no scorer, GT, logit, or margin input. Its deterministic 54-context gate
is **97 B raw payload**, repeat-identical and parse-back exact, but it has no
measured archive delta, pose result, public-receiver render, or exact score.
No scorer, Modal call, candidate compile, or evaluator ran. The pointer did not
move.

## H versus B characterization

All denominators are the complete retained B/H populations: EC1
`12,075 B + 52,854 H = 64,929` cells and QS3
`108 B + 76 H = 184` B/H cells. No prefix or sampled negative was used.
Distances are scorer-grid pixels. `q25/median/q75` are per-cell distributions.

| object / population | cells | distance to nearest realized B | distance to nearest base-error target | distance to decoded-token boundary | pre-edit margin |
|---|---:|---:|---:|---:|---:|
| EC1 B | 12,075 | `0 / 0 / 0` | `0 / 0 / 0` | `0 / 0 / 0` | `0/12,075` retained |
| EC1 H | 52,854 | `7.28 / 19.24 / 41.05` | `3.16 / 8.25 / 20.00` | `0 / 0 / 0` | `0/52,854` retained |
| QS3 B | 108 | `0 / 0 / 0` | `0 / 0 / 0` | `0 / 1.00 / 1.10` | `0/108` retained |
| QS3 H | 76 | `1.00 / 3.16 / 7.02` | `1.00 / 1.00 / 2.24` | `0 / 0 / 1.00` | `0/76` retained |

For QS3, distance to the nearest listed edit was additionally
`1.00 / 1.21 / 2.24` for B and `1.00 / 3.80 / 8.28` for H. Those populations
separate in hindsight, but QS4 already showed that hard nearest-site trimming
does not preserve the modeled benefit, and QS5 repeated the 17-flip ceiling
after fresh compensation. It is not an admissible reason to rerun blind trim.

### What the proposal could know

| field | proposal-time status | use here |
|---|---|---|
| decoded center + left/right/up/down semantic classes | **known** | shippable 3,125-state gate; selected |
| distance to decoded semantic boundary | **known** | measured, but the best strict radius preserved 99.37% B and 97.29% H, leaving net `-39,422`; rejected |
| QS3 named source→target proposal class pair | **known** | all three named classes remained net-positive; no class-pair veto exists |
| QS3 distance to listed edit coordinates | **known** | diagnostic only; prior exact-object trim negative consumes this route |
| scorer argmax transition class pair | **unknown** | diagnostic only; never passed to proposer |
| nearest realized B or base-error target | **unknown** | hindsight/GT-derived diagnostic only |
| pre-edit top-1/top-2 margin | **unavailable** | coverage `0/64,929` EC1 and `0/184` QS3; margin floor blocked |

The dominant EC1 outcome transition was Road→Lane: `B=6,024`, `H=31,542`,
`B/(B+H)=16.04%`. It is not proposal-visible because it requires the retained
base/GT/candidate scorer fields. Using it as a runtime veto would be a scorer
leak. For QS3, the proposal-visible named class pairs were:

| named proposal class | B | H | net | B/(B+H) |
|---|---:|---:|---:|---:|
| Undrivable→Movable | 33 | 31 | +2 | 51.56% |
| Road→Movable | 8 | 4 | +4 | 66.67% |
| MyCar→Road | 67 | 41 | +26 | 62.04% |

Pair 532 is harmful, but it shares Undrivable→Movable with useful pairs 176
and 178. A class-pair veto cannot identify it without collapsing positive mass.

## Collateral-priced objective

For a proposal support `A`, with expected beneficial and harmful flips
`E[B(A)]` and `E[H(A)]`, archive-byte delta `Δb`, and pose change already in
contest-score units `ΔS_pose`, the proposer prices

```text
E[ΔS(A)] = (100 / 117,964,800) * (E[H(A)] - E[B(A)])
          + ΔS_pose
          + (25 / 37,545,489) * Δb.
```

The exact marginals are `8.477105034722222e-7 S/net flip` and
`6.658589531221714e-7 S/archive byte`, hence
`0.7854791823326633 net flips/B` or `1.273108215332031 B/net flip`.
H and B use the same full-field denominator and opposite signs. A proposal
cannot receive gross-B credit while hiding H in an endpoint diagnostic.
Actual admission still requires a fresh full-field receiver measurement,
fresh pose, and actual ZIP bytes; the proposal objective is a ranker only.

## Built proposer

`src/tac/optimization/ec2_collateral_suppressed_proposer.py` provides:

- exact B/H/rate/pose pricing with strict negative-delta admission;
- offline context-count fitting with explicit `+1=B`, `-1=H` labels;
- proposal-time support from decoded oriented semantic context only;
- a deterministic counted bit-gate grammar with strict parse-back;
- input-domain checks that reject non-integer/out-of-vocabulary context.

The selected gate was learned with five folds by `pair_id mod 5`; each held-out
cell was decided by a table fitted on the other four folds. The final full-data
payload is video-derived and therefore counted. It keeps 54/3,125 contexts and
measures **97 B**, with byte-identical repeat SHA-256
`786732f59250aa63c974fa68a309358beaa3d8e814aa93ab081b95c333045b13`.
This module does not integrate the gate into the old pre-TokenBlock EC1
receiver: doing so post hoc would reintroduce the propagation and stale-
compensation defects. The fire trigger instead requires in-training use on a
reviewed output-side actuation with fresh same-object compensation.

## Prediction against sub-0.12

The selected held-out projection has `5,653` net flips. At the measured law it
could tolerate `7,196.88 B` total before rate-only break-even. Pricing the
97 B raw gate beside the `1,176 B` rc2 repin adapter anchor gives projected
`ΔS=-0.00394447`; pricing it beside EC1's original `1,471 B` CP135 archive
delta gives `ΔS=-0.00374804`. Both are **projections only**: the gate's actual
archive delta, gated receiver field, pose, and dx2 composition are unmeasured.

Even the optimistic lower-rate bracket maps the current dx2 score only from
`0.14821987563243377` to `0.14427540660362984`, before composability. Even the
impossible favorable bound that keeps all 12,075 EC1 B and drives H to zero,
while charging no added rate or pose, reaches only `0.13798377130300668`.
Therefore EC2 collateral suppression alone cannot reach sub-0.12 and cannot
replace the required new rate representation.

## Fire disposition

Disposition: **QUEUED-WITH-FIRE-ORDER, BLOCKED**. MAIN owns the consumer and
no lane was claimed or fired. The single seal is
`/Volumes/APDataStore/pact/ddm_ec2_collateral_suppressed_conditioner/seal_r4/SEALED_FIRE_ORDER.json`
(SHA-256 `a0380a60aebbce24b9d10c8c2332ad37a35665627806d24543c9d3c6c8703518`).
Its dispatch argv is deliberately null; this blocked order is not authority to
invent one.

Fire trigger: MAIN owns an idle unique contest-CUDA scorer lane; a reviewed
from-scratch conditioner consumes the counted decoded-context gate inside
training, prices H with the same full-field denominator as B, retains at least
95% of gross B while raising `B/(B+H)` strictly above `108/189` on a held-out
stage, integrates fresh same-object compensation, and real-coder parse-back
proves negative complete projected delta before scorer dispatch. MAIN must then
reseal exact source/payload/archive/runtime hashes and a non-null command.

## Retained custody

Seal manifest:
`/Volumes/APDataStore/pact/ddm_ec2_collateral_suppressed_conditioner/seal_r4/MANIFEST.json`
(SHA-256 `78716bbe42da8655f7ee3234b6c3e99a88ce6d8d1f394e27067b406f2583d53c`).

| payload | bytes | SHA-256 |
|---|---:|---|
| `retained/ec1_hb_cells.npz` | 439,515 | `008e63644ea7fb6d70efd9167dbf09fd233c7ca3ab52646c1e3977492c5215ac` |
| `retained/qs3_hb_cells.npz` | 6,379 | `41c265867d2743a6de42ef3a325d2db9b36609872b343e338966d114ac070be5` |
| `retained/ec2_context_counts.npz` | 2,304 | `4d7ae3dd8121c5ce8262f55b05ea2d70e93620ef306bb28a0988d10c7110d63e` |
| `retained/ec2_context_gate.br` | 97 | `786732f59250aa63c974fa68a309358beaa3d8e814aa93ab081b95c333045b13` |
| `retained/ec2_context_gate.repeat.br` | 97 | `786732f59250aa63c974fa68a309358beaa3d8e814aa93ab081b95c333045b13` |
| `H_B_CHARACTERIZATION.json` | 14,844 | `00c359398b99dbd7abd02b25380b25bb145a638f28ac66ab6ac1a658fe930a45` |
| `SUPPRESSION_VARIANTS.json` | 6,406 | `999582236091146d5c1e778c070b9b1260b6fc25856ebf9ffdda2b1d6a005aa1` |
| `SUB012_PROJECTION.json` | 1,481 | `6677a9fc265e13ba742eb6e6272b620ece61c111db5c392ddffb41f61d40a345` |

Superseded `seal_r1` and the interrupted resumable `seal_r2` were preserved;
`seal_r3` was superseded after adding full adapter-plus-gate rate brackets.
No evidence bytes were deleted or overwritten.

## RECALL EVIDENCE

Searches covered the exact charter seeds across `.omx/research`, the canonical
research indexes/DAG, `main_hot_state.md`, EC1/EC2 workers/runtime, QS3/QS4/QS5
receipts, BG2 decomposition, MP1 instruments, JO1's exact objective, and the
canonical-equation registry. The exact `ddm_ec2`/seg-mechanism memory hook was
not found in the bounded memory registry search, so no memory claim was
fabricated; primary retained receipts were used.

Findings beyond the charter seeds that changed the result:

1. BG2 retained exact EC1 B and H masks, and the EC1 fire input retained the
   full decoded semantic-token plane. Together they expose a legal
   proposal-visible context without another scorer run.
2. BG2's existing 8D frame state had negative incremental held-out R2, whereas
   class/edge/location exposure explained substantial variance. This moved the
   build from a frame-state conditioner to decoded local context.
3. Neither EC1 nor QS3 retained base logits. MP1's margin instrument is a
   different object; importing it would be borrowed-margin fake evidence.
4. QS3's three named proposal class pairs are all net positive. Pair 532's
   negative row is pair-specific, not class-pair-specific, so the apparent
   class veto disappears under the actual proposal vocabulary.
5. JO1 already owns the exact full-field collateral objective and future joint
   training route. EC2 therefore built a proposer primitive and a blocked
   handoff rather than duplicating or touching the live JO2 run.

## Verification

- Pin checks passed for TL1
  `d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15`
  and IG1
  `8ec60069b33f2d19d9a39ea30c94acee66ac299d800b5e739f411a48aa42ce8b`.
- Seal verification rehashed all 13 declared outputs and reparsed the gate.
- Focused proposer suite: **5 passed**.
- `py_compile`: passed for proposer, analyzer, and tests.
- `ruff check`: passed for proposer, analyzer, and tests.
- Developer preflight: **18/25 gates green, 7 red**. Bounded follow-up placed
  every red outside the four EC2 landing files: one existing strict state
  writer, one existing custody-tag bypass, one existing shared-state writer,
  25 legacy ad-hoc launchers, one AGENTS claim-closure documentation gap, 124
  old landing memos, and five existing substrate scorer-contract violations.
  No waiver or unrelated repair was added.
- Review pass 1 caught and fixed non-resumable checkpoint writes plus stale
  imports. Review pass 2 caught and fixed filesystem-sidecar manifest capture,
  the distinction between QS3's named proposal edge and nearest realized token
  transition, and underpriced adapter-plus-gate projection arithmetic.
- Scorer forwards: **0**. Modal dispatches: **0**. Candidate archives: **0**.

## What was not measured

No pre-edit margins, gated receiver output, fresh compensation, gate archive
delta, gated PoseNet result, contest-CPU row, contest-CUDA candidate row, or
`upstream/evaluate.py` score was measured. The 5,653-net row is cross-fit
field selection on an existing endpoint, not a receiver-realized candidate.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER, BLOCKED** — owner: MAIN sole contest-CUDA scorer-lane and exact-row owner; consumer store: `/Volumes/APDataStore/pact/ddm_ec2_collateral_suppressed_conditioner/main_fire/`; fire trigger: after the live JO2 run is harvested and the lane is idle, integrate the counted context gate into a reviewed from-scratch output-side conditioner with the full-field B/H objective and fresh same-object compensation, then proceed only if a retained held-out stage keeps at least 95% of B, exceeds `108/189` beneficial activity, and a real-coder parse-back projection is complete-objective negative.

## LIVE-HYPOTHESES

- A soft or dual-priced use of the 54 high-value contexts during training may
  preserve more than the hard gate's 55.24% B while retaining its 98.08% H
  suppression, because the held-out separation is large and not a same-frame
  fit artifact.
- Moving the gated actuation after the renderer TokenBlocks may keep decoded-
  context targeting while avoiding the pre-TokenBlock receptive spread that
  produced EC1's Road→Lane collateral.
- Full stage-boundary fields may contain an earlier checkpoint with higher B
  retention than the endpoint; EC1's end-only receipt could not test it.

## DEAD-ENDS

- A pre-edit margin floor on these retained objects: base logits were not
  retained, and candidate post-edit margins are not substitutes.
- Blind semantic-boundary support radii: every tested radius left the EC1 net
  harmful; radius zero still had `B=11,999`, `H=51,421`.
- QS3 named class-pair veto: every named proposal class is net positive, and
  the harmful pair 532 shares its class with useful pairs.
- Hindsight scorer-transition vetoes and distances to realized B/base errors:
  they require scorer/GT outcome and are not legal proposal-time inputs.
- Reusing QS4's trimmed edit set, QS5's three-pair support, or stale Schur
  compensation: the retained exact-object negatives already close those
  instances, and fresh compensation is mandatory for any new support.
- Claiming the 97 B gate or 5,653-net cross-fit row as a candidate: neither is
  receiver-closed, pose-measured, real-archive-priced, or exactly evaluated.

Own-vehicle frontier: **dx2 S 0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]**, archive `976f706d…`; **UNMOVED**.
