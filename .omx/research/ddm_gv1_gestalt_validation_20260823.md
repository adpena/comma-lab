# ddm_gv1 gestalt validation — the joint-move routing law is refuted

Date: 2026-08-23  
Cost: $0  
Verdict: **`REFUTED`**  
`verdict_scope: FORMULATION:DX2_LINEAGE_SUB012_CAMPAIGN`  
Score claim: false  
Pointer movement: none

## Answer first

The census does not support “single-axis loses, joint wins.” It contradicts the descriptive premise
that produced that causal story:

- **114 measured moves** across **50 families** met the declared population rule.
- **13 of 18 pointer moves were single-axis**. Only 5 were joint.
- Counting `POINTER_MOVE` and `ADMITTED_SUB_BAND` as successes, joint moves succeeded **8/49
  (16.33%)** and single-axis moves succeeded **14/65 (21.54%)**.
- Fisher's exact test gives **odds ratio 0.7108, two-sided p=0.6326**. The risk difference is
  **-5.21 percentage points** for joint moves and phi is **-0.0654**. The sign is opposite the memo's
  proposed law, but the effect is small and statistically indistinguishable from zero.
- Time is the dominant confounder: early moves succeeded **20/46 (43.48%)**; the 08-22/23 late window
  succeeded **2/68 (2.94%)**. Late-versus-early Fisher odds ratio is **0.03939**, two-sided
  **p=7.286e-08**.
- After stratifying by time, there is still no axis-count signal: early joint **7/15** versus early
  single **13/31** (OR 1.2115, p=1.0); late joint **1/34** versus late single **1/34** (OR 1.0,
  p=1.0).

The correct plain-language conclusion is: **the campaign became harder as it converged; axis count did
not predict success.** Jointness is not a routing privilege. A charter still has to name the actual
causal interaction it expects and test it.

## Population and classifier

The bounded population is every retained CP135 -> rr4 -> dx2 shipping-lineage move found in the
searched campaign corpus that has:

1. an actual byte or evaluator-distortion delta on a shipping-lineage body;
2. a receipt-supported `delta_S` rather than a rounded `Final score` display;
3. a terminal disposition representable as `POINTER_MOVE`, `ADMITTED_SUB_BAND`, `REFUSED`, or
   `NET_POSITIVE_COST`.

The campaign clock is day 0 = 2026-08-12 and day 10 = 2026-08-23. The confounder split is early
`campaign_day < 9` versus late `campaign_day >= 9`, which is exactly the 08-22/23 window named by the
memo. Repeated rungs are separate moves because the charter says every measured move, not one row per
family.

Axes were classified from the mutation implemented by code:

- `field`: stored or receiver-realized semantic field values;
- `model`: the causal probability law, not every set of learned weights;
- `coder`: the coding algorithm or symbol/receiver grammar;
- `order`: traversal or serialization order;
- `renderer`: renderer architecture or renderer parameters, treated as one economic axis;
- `carrier`: frame-0 carrier coefficients or basis;
- `pose`: an explicit pose payload or pose-target predictor.

A consequent stream-byte change is not a second coder axis when the coder is unchanged. This is why
`rr4` and `to1` are single-axis `model` moves, and why `fs2`/`fs3` are single-axis `field` moves even
though their changed fields were re-encoded. Conversely, `js7` is joint because it changes the field
and introduces a distinct EC1 overlay codec. The original prose labels were not used as the
classifier.

## Full census table

The complete, uncapped table is the adjacent typed JSONL:

- `.omx/research/ddm_gv1_gestalt_validation_20260823.jsonl`
- schema: `ddm_gv1_census.v1`
- rows: **114**
- families: **50**
- SHA-256: `483ca46dc3063a477e60a48886e6f0bcab753a51903cee94848f810e26d88ba5`

Every line contains `move_id`, `axes_touched`, `axis_count`, `outcome`, receipt-derived `delta_S`,
authority, the actual code mutation, and an evidence path with its full SHA-256. The JSONL is the full
census table; no qualifying row was capped or elided from it.

Validation re-read all 114 lines, rejected duplicate `move_id` values, checked the axis and outcome
enums, checked `axis_count == len(axes_touched)`, required finite `delta_S`, opened every evidence
path, and matched every recorded evidence SHA-256.

## Contingency tables

Success means `POINTER_MOVE` or `ADMITTED_SUB_BAND`. Failure means `REFUSED` or
`NET_POSITIVE_COST`.

| classifier | success | failure | n | success rate |
|---|---:|---:|---:|---:|
| single axis | 14 | 51 | 65 | 21.54% |
| joint, 2+ axes | 8 | 41 | 49 | 16.33% |
| **total** | **22** | **92** | **114** | **19.30%** |

Two-sided Fisher exact: **OR 0.7108013937, p=0.6326319332**. Effect sizes: joint-minus-single
risk difference **-0.0521193** and phi **-0.0653805**.

The uncapsed k x 2 view is:

| axis count | success | failure | n |
|---:|---:|---:|---:|
| 1 | 14 | 51 | 65 |
| 2 | 8 | 38 | 46 |
| 3 | 0 | 1 | 1 |
| 4 | 0 | 2 | 2 |

The four-outcome audit is:

| axis count | POINTER_MOVE | ADMITTED_SUB_BAND | REFUSED | NET_POSITIVE_COST |
|---:|---:|---:|---:|---:|
| 1 | 13 | 1 | 18 | 33 |
| 2 | 5 | 3 | 7 | 31 |
| 3 | 0 | 0 | 1 | 0 |
| 4 | 0 | 0 | 2 | 0 |
| **total** | **18** | **4** | **28** | **64** |

The memo's strongest descriptive sentence is therefore false on its own population: **72.2% of
pointer moves were single-axis**, and the late window contains **33 joint failures and 33 single-axis
failures**, not exclusively single-axis losses.

## Confounder adjudication

### Bytes

The exact-archive-byte subset has n=75: 24 joint and 51 single-axis moves.

| classifier | n | median delta bytes | positive-byte moves | non-positive-byte moves |
|---|---:|---:|---:|---:|
| joint | 24 | +55.5 B | 16 | 8 |
| single | 51 | -105 B | 17 | 34 |

Joint moves did spend more bytes on this subset. Mann-Whitney U is **834.0, p=0.0118552**; the
positive-versus-non-positive Fisher table gives **OR 4.0, p=0.0118581**. This does not rescue the
routing law: despite buying more byte headroom, joint moves still did not succeed more often. It is a
real confounder against interpreting raw joint outcomes as evidence for co-location.

### Early-easy versus late-hard

| campaign phase | success | failure | n | success rate |
|---|---:|---:|---:|---:|
| early, day 0-8 | 20 | 26 | 46 | 43.48% |
| late, day 9-10 | 2 | 66 | 68 | 2.94% |

Late-versus-early Fisher exact: **OR 0.0393939, p=7.286e-08**. This is the dominant pattern. Within
each phase, axis count has no detectable association:

| phase | joint success/failure | single success/failure | OR | p |
|---|---:|---:|---:|---:|
| early | 7 / 8 | 13 / 18 | 1.2115 | 1.0 |
| late | 1 / 33 | 1 / 33 | 1.0 | 1.0 |

The late-hard alternative fully explains the visual pattern that MAIN recalled. It leaves no evidence
for a generic co-location mechanism.

### Repeated-sweep sensitivity

The main test correctly counts every rung, but sweep siblings are correlated. A sensitivity pass that
drops the repeated-rung families `ld1`, `ae1`, `oe1`, `ap1`, `to2`, `ad2`, `ef1`, `mp2`, and `wd3`
leaves n=43. Joint succeeds 7/17; single succeeds 13/26; **OR 0.7, p=0.7556**, risk difference
**-8.82 percentage points**. The conclusion does not depend on large negative sweeps.

## Exclusion and deduplication ledger

- `jf1` is excluded exactly as chartered: it was still running and this arm does not adjudicate it.
- Apparatus, hygiene, custody-only, and scoreless build receipts are excluded because they carry no
  terminal byte/distortion move.
- Only one terminal row is counted per actual move. Intermediate projections are not counted again.
- CP5V's initial projection is excluded, but its later exact contest-CUDA VD1 row is included.
- The re1 full-auth signed result is excluded because its own erratum retracts the sign under 8-decimal
  component uncertainty. The separate stable re1 component admission row is included.
- SA1's T4 sign gate is excluded because its quoted joint delta mixes an exact T4 field/rate leg with a
  local-linearized pose leg; no realized complete `delta_S` was retained for that move.
- JG1 calls itself “not a row,” JG2 stops after its sparse rate-price stage, and JG3 labels its archive
  delta unproven and does not reach its n600/byte-close stages. I did not find a terminal complete-S
  move for those three in the searched memos, receipts, DAG, index, and task-ledger scope. JG4's
  terminal field-only refusal and JG5's terminal joint pointer move are included.

These exclusions are outcome-blind: each follows the population rule above, not whether the row would
help or hurt the hypothesis.

## RECALL EVIDENCE

I searched beyond the charter's seed list before freezing the population:

- `.omx/research/` memos and arm receipts by content with queries including
  `dx2|rr4|cp135|POINTER_MOVE|ADMITTED_SUB_BAND|REFUSED|NET_POSITIVE_COST`, and then by concrete
  archive sizes, score deltas, and candidate SHAs;
- all `ddm_jg*`, `ddm_rc4*`, `ddm_fs*`, `ddm_to*`, `ddm_qs*`, `ddm_re1*`, and the named late-loss
  families;
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC surfaces,
  `.omx/state/canonical_task_status.jsonl`, and the live board;
- the complete canonical-equation registry from
  `.venv/bin/python tools/list_canonical_equations.py --json`, with score/rate/marginal terms checked;
- implementation files for the disputed axes, including the rr2/rr4 probability recode, to1 tail
  override, fs2/fs3 field edits, rc4 receiver-drop contract, PZ4R carrier predictor, IV1 SD1M path,
  PS1U carrier overlay, MP2 renderer edits, and WD2/WD3 renderer replacements.

Beyond the seeds, recall found CP5V, JS7, PO1, PZ4R, WD2, WD3, MP2, PS1U, IV1, and the earlier pointer
chain (E480B through DX2). It also found that several moves described as “joint” in synthesis prose
only mutate one economic axis in code: rr4 and to1 change the probability model while retaining RC64;
fs2/fs3 change the field while retaining the coder and carrier; rc4 changes the coder/receiver omission
contract while retaining the carrier. These findings widened the preliminary census and changed the
classification before the denominator was frozen.

## Verdict

**`VERDICT: REFUTED`**

`verdict_scope: FORMULATION:DX2_LINEAGE_SUB012_CAMPAIGN`

This is stronger than `DESCRIPTIVE_ONLY_CAUSAL_REFUTED`: the causal claim fails, and both supporting
descriptive universals fail as well. It is not `UNDERPOWERED`; n=114 is ample to reject the universal
claim, while the exact test gives no positive association to promote as a weaker routing rule.

The verdict does **not** say joint moves are useless. It says jointness by itself is not evidence of a
win. Particular joint mechanisms such as field edit plus carrier compensation can still work when a
receipt proves the interaction.

## Exact append-only correction owed to the gestalt memo

The following text is appended verbatim to
`.omx/research/ddm_gestalt_the_three_laws_20260823.md`:

> ## APPEND-ONLY CORRECTION — GV1 full census (2026-08-23)
>
> **§4 and the “only winning move class” premise in §6 are retracted.** GV1 classified 114 retained
> shipping-lineage moves from code, not prose. Thirteen of 18 pointer moves were single-axis. Counting
> pointer moves plus admitted sub-band rows as successes, joint moves succeeded 8/49 (16.33%) and
> single-axis moves succeeded 14/65 (21.54%): Fisher OR 0.7108, two-sided p=0.6326, joint-minus-single
> risk difference -5.21 percentage points. The late 08-22/23 window contains 33 joint failures and 33
> single-axis failures.
>
> Campaign phase, not axis count, carries the signal: early success was 20/46 versus late success 2/68
> (late/early Fisher OR 0.03939, p=7.286e-08). Within the early and late strata, axis count has no
> association with success. Joint moves also spent more archive bytes in the eligible subset (median
> +55.5 B versus -105 B; Mann-Whitney p=0.01186), so raw joint outcomes are byte-confounded.
>
> **Corrected verdict: `REFUTED`.** `verdict_scope: FORMULATION:DX2_LINEAGE_SUB012_CAMPAIGN`.
> Jointness is not a
> routing privilege and does not justify `ddm_rj1` or any other fire. A future joint move must name and
> measure its own causal interaction. Full table and tests:
> `.omx/research/ddm_gv1_gestalt_validation_20260823.{md,jsonl}`; JSONL SHA-256
> `483ca46dc3063a477e60a48886e6f0bcab753a51903cee94848f810e26d88ba5`.

## Measurement and authority boundaries

- **Measured this arm:** no new candidate and no scorer output. This arm measured only the retained
  receipt population, classifications, counts, exact contingency tests, effect sizes, and confounders.
- **Not measured:** no SegNet/PoseNet pass, no inflate, no archive build, no Modal/GPU/CPU contest eval,
  no JF1 outcome, and no new score.
- Advisory and contest authority labels remain separate in the JSONL. No advisory row is promoted.
- Negative scope is this bounded lineage campaign, not all possible joint mechanisms.
- No new fire or follow-on action is created. MAIN continues to own all fires.

## Ending frontier

**dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, archive SHA-256
`976f706d…` — **UNMOVED**. Gap to 0.12: **0.028220**.

## LIVE-HYPOTHESES

- A joint move can still win when its axes are demonstrably complementary, such as a field edit whose
  pose damage is re-solved through the carrier. This remains plausible because QS2 and JG5 are real
  successes, but the interaction must be measured rather than inferred from axis count.
- Campaign phase is a useful prior for expected yield. The early-to-late collapse is large and exact,
  so a genuinely new representation may be more valuable than another local perturbation of dx2.
- Rate-neutral or rate-negative joint moves may behave differently from the byte-spending joint pool.
  That subgroup remains plausible because joint rows spent significantly more bytes, but it was not a
  pre-registered stratum and is not promoted here.

## DEAD-ENDS

- The generic rule “single-axis moves lose and joint moves win” is closed for this campaign: the full
  census has the opposite point estimate and no significant association.
- The claim “every pointer move was joint” is closed: 13 of 18 pointer moves were single-axis.
- Treating a consequential re-encode as a second coder axis is closed: rr4/to1 retained RC64, while
  fs2/fs3 retained their coder; their code mutates one economic axis.
- Using the late 08-22/23 loss cluster as causal evidence for jointness is closed: campaign phase
  explains the cluster, and late joint and single losses are exactly tied 33 to 33.
- Re-running current-body fixed-coder residue allocation, the NI1/RI1 whole-body replacements, or the
  generic PPMd/ZPAQ sweeps is closed by their retained family/formulation receipts; GV1 adds no reason
  to reopen them.
