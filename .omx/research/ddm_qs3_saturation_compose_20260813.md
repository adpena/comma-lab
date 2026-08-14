# ddm_qs3 saturation compose — retained-field blocker, scale race, and no-fire

**Status:** `QUEUED-WITH-A-FIRE-ORDER` for the missing authority-field recovery;
no dual-axis candidate was sealed or fired. The effective frontier is **UNMOVED**.

## Result first

QS3 did not produce a valid composed candidate. The required download receipt eventually
landed, and all 38 batches on each of the four downloaded n600 surfaces form exact 0–599
partitions, but the download does **not** contain the referenced matched T4 GT argmax field
(`91d3ff11…`). Two direct `modal volume get` recovery attempts failed with `Could not connect
to the Modal server`. Without that field, the aggregate worker facts—189 changed pixels and
32 net beneficial flips—do not identify which changed pixels helped, hurt, or changed from one
wrong class to another. An exact per-pixel mechanism taxonomy and exact per-pair flips/B
waterfill would therefore be fabricated.

The key correction is mathematical: **157 = 189 − 32 is aggregate arithmetic, not a set of
157 identifiable “reverted pixels.”** If `B`, `H`, and `R` are beneficial, harmful, and
wrong-to-different-wrong changed pixels, then `B-H=32`, `B+H+R=189`, and therefore
`157=2H+R`, not `H+R`. The GT field is required to solve those counts and identities.

The scorer-free work did land three usable facts:

1. QS2's worker result is complete and deterministic. Recomputed from its full-precision
   components, ΔS is **−4.374913965108395e-6**, so it is genuinely beneficial but sub-band.
2. The exact nine-pair four-bit codebook rung costs **6.1111 B/pair**, worse than QS2's
   5.67 B/pair. Deadzone step 2 reaches **5.0 B/active pair**, but remains a toy-bracket:
   Q3C1 has no contest-runtime consumer and its nine-pair Pose effect is unmeasured.
3. The complete 200-row calibrated screen admits **0/200** at the current 16.9312%
   realization calibration plus conservative Schur-pose price. This is a formulation-scoped
   projection, not a family negative; improved realization remains capable of changing it.

## Measured component rows

| row | Seg ΔS | Pose ΔS | rate ΔS | total ΔS | axis / verdict |
|---|---:|---:|---:|---:|---|
| QS2 R2, 34 B | −2.712673611111111e-5 | +1.126177398488859e-7 | +2.263920440615383e-5 | **−4.374913965108395e-6** | `[contest-CUDA T4 frozen-SegNet field + PoseNet first6 vectors, n600] COMPONENT-ONLY`; SUB-BAND |

The worker measured 34,970 base flips, 34,938 candidate flips, candidate d_pose
`6.885829861857928e-6` versus matched base `6.885642960696714e-6`, and a 186,286 B
archive. Pose repeated bit-identically at the component value. This is not an
`upstream/evaluate.py` score and is not a promotion row.

## Retained-field post-mortem

The exact base/candidate T4 fields differ at only these six pairs and at 189 pixels total.
The transition census below is GT-free and therefore does not label benefit or harm.

| pair | proposal | edge | changed | source→target | target→source | off-edge | changed in named target component | exact marginal flips/B |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 105 | `js6_0000_9fbf75d81c43` | Road→Movable | 12 | 7 | 0 | 5 | 7 | BLOCKED: matched GT absent |
| 176 | `js6_0072_f790b6493122` | Undrivable→Movable | 9 | 5 | 4 | 0 | 5 | BLOCKED: matched GT absent |
| 178 | `js6_0006_92685b3e3e44` | Undrivable→Movable | 41 | 40 | 0 | 1 | 18 | BLOCKED: matched GT absent |
| 517 | `js6_0004_06fc74e20d9e` | MyCar→Road | 36 | 35 | 0 | 1 | 18 | BLOCKED: matched GT absent |
| 523 | `js6_0001_da319a6b65d0` | MyCar→Road | 77 | 44 | 0 | 33 | 37 | BLOCKED: matched GT absent |
| 532 | `js6_0118_83f376603d6e` | Undrivable→Movable | 14 | 5 | 7 | 2 | 4 | BLOCKED: matched GT absent |
| **total** |  |  | **189** | **136** | **11** | **42** | **89** | **BLOCKED** |

### Requested mechanism taxonomy

| mechanism | exact count among the claimed ~157 denominator | available evidence | verdict |
|---|---:|---|---|
| sub-quantum amplitude vs js5 uint8 floor | unmeasured | All 140/140 pixels in the six named target components are inside JS6's ≥1-scorer-value strong support. Across the full 200-row bank, 1,299/1,321 are above and only 22/1,321 are below. | Quantum-floor dominance is **not supported** on the available target-component proxy; exact reverted-pixel attribution blocked. |
| AA/resize washout through shared D | unmeasured | Candidate pre-R, camera, and scorer-input payloads are retained, but the missing GT prevents identifying the failed authority set. | BLOCKED |
| tie-margin failure | unmeasured | Candidate logits are retained; the matched T4 base/GT attribution surface is not local. | BLOCKED |
| other | unmeasured | Aggregate `189−32` cannot identify pixels. | BLOCKED |

`verdict_scope=INSTANCE`: the exact matched QS1 T4 retained-field post-mortem is blocked by
one missing authority field. This is not a negative on survival engineering.

## Survival-engineered recipes

The runner retained 600 recipes—three variants for each of all 200 JS6 rows—at
`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/survival_engineered_recipes.jsonl`
(SHA-256 `40337b152d8969b57ff2a83b10b5cc7e03920022c582013afcdce929520e8e29`).
Each carries a curve at efficiencies 16.9312%, 25%, 33.3%, and 50%.

- `amplitude_above_js5_quantum`: recipe generated from measured strong-support coverage.
- `d_aware_precompensation`: recipe generated, but no receiver payload was invented; the exact
  D-response needs the authority-pixel assignment.
- `margin_targeted_placement`: explicitly blocked because matched T4 base logits are absent.

These are `TOY-BRACKET` proposal recipes, not semantic-token payloads and not candidate rows.
GCA1 changed the plan by forbidding a fake finite heat-kernel edit radius: graph distance/heat
is retained only as a prioritizer. HR1/RVS1 changed the generator contract to quantize first,
then compensate in nearby legal integer DOF, with a whole-quantum survival guard.

## Complete-bank calibrated screen

Selection was the full 200-row bank, not a prefix or sample. The calibration used exact QS2
realization `32/189 = 0.1693121693`, the exact nine-pair codebook cost, and the median retained
QS1 conservative Schur-pose S per semantic cell. The full JSONL is
`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/full_bank_screen.jsonl`
(SHA-256 `39dc688d75de33b628a051fca6a1b0c9c29679d2b4df7463f8b41f1af2b678c3`).

| rank | proposal | pair | target-mass upper bound | calibrated flips/B | pose-adjusted bar | toy-bracket pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `js6_0000_9fbf75d81c43` | 105 | 58 | 1.60693 | 2.87428 | no |
| 2 | `js6_0001_da319a6b65d0` | 523 | 52 | 1.44069 | 2.39945 | no |
| 3 | `js6_0002_721cb5eefb29` | 510 | 35 | 0.96970 | 2.58938 | no |
| 4 | `js6_0003_56b961be310e` | 176 | 34 | 0.94199 | 2.39945 | no |
| 5 | `js6_0072_f790b6493122` | 176 | 33 | 0.91429 | 1.25984 | no |

No row reached the charter's `0.785 × (1 + pose_S/rate_S)` bar. The best row requires roughly
30.3% realization efficiency at the measured 6.111 B/pair price, so the charter's 50% survival
hypothesis remains numerically sufficient if a real generator achieves it. Rows are queued for
exact Schur/authority evaluation only after the post-mortem is unblocked; none is admitted now.

## Shared codebook and deadzone race

QS2 Q2C1 cannot represent the three additional unique retained rows: their exact code deltas
reach ±6, outside its `[-3,4]` alphabet. QS3 therefore implemented and round-trip-tested Q3C1,
a strict four-bit `[-8,7]` payload. All 24 q0–q11 candidate payload sets were retained.
The adapted reference is provenance-pinned to QS2 commit
`d77fb69efc390bf9cbb41dab90d10400300180e5`: the current rate runner and overlay hashes match
that commit byte-for-byte (`8654e6d3…` and `7e5d905d…`).

| rung | active pairs | overlay bytes | best q | archive bytes | ΔB vs CP135 | B/active pair | closure |
|---|---:|---:|---:|---:|---:|---:|---|
| exact step 1 | 9 | 57 | 10 | 186,307 | +55 | **6.1111** | payload parse-back only; no contest-runtime consumer |
| deadzone step 2 | 8 | 39 | 11 | 186,292 | +40 | **5.0** | payload parse-back only; nine-pair Pose unmeasured |

The exact rung does not beat 5.67 B/pair. Step 2 does, but this is only a bounded coding result;
it cannot be used for candidate admission until a receiver consumes Q3C1 and the changed lattice
gets matched Pose/Seg measurement.

## Compile and dispatch disposition

No waterfilled HP3/RC64 composed candidate was built. The ordered Step-1 authority gate failed,
the complete calibrated screen admitted zero rows, and Q3C1 is not receiver-closed. Building or
firing a candidate would violate the pre-encode requirement `net realized ΔS < 0` on the matched
34,970-flip / `6.885642960696714e-6` base.

The sealed no-fire/recovery receipt is
`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/SEALED_NO_FIRE_ORDER.json`. It assigns MAIN and
contains the exact GT recovery argv plus the exact QS3 resume argv. No Modal scorer job was fired.
The canonical evaluate.py follow-on is deliberately unnamed until a worker verifies a super-band
candidate.

## RECALL EVIDENCE

Search scope and queries:

- Full `.omx/research/`, `docs`, `prompts`, `src`, and `experiments` content searches for
  `ddm_qs1`, `ddm_qs2`, `js6`, `realization efficiency`, `survival`, `reverted`,
  `edit-propagation`, `quantum`, `deadzone`, `Schur`, `waterfill`, `HP3`, and `RC64`.
- `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` searched for survival,
  quantization, edit propagation, and waterfill.
- `.venv/bin/python tools/list_canonical_equations.py --json` searched for score, rate, pose,
  Seg, waterfill, quantization, margin, and Schur equations.
- Live authority re-read from `.omx/state/main_hot_state.md`; it supersedes the common
  contract's stale frontier paragraph.

Beyond the charter seeds, the search found: GCA1's explicit no-theorem boundary for heat/edit
radius; RVS1's camera-lattice quantize-then-compensate requirement; the canonical research
index's established D/round-trip survival split; and the now-landed QS2 worker result. These
changed the plan by (1) refusing a graph-radius admission claim, (2) widening the integer coder
instead of clipping ±6 deltas, (3) separating structural quantum coverage from exact authority
taxonomy, and (4) consuming QS2 as the measured sub-band calibration rather than its projection.

## Boundaries

Measured here: exact retained base/candidate T4 field difference census; full 200-row structural
screen; real Q3C1 payload bytes and real Brotli/ZIP bytes for 24 toy-bracket archives; QS2 worker
component recomputation. Not measured: matched per-pixel GT attribution, exact per-pair flips/B,
base-logit margins, new proposal realization, nine-pair Q3C1 Pose, receiver parse-back for Q3C1,
an HP3/RC64 composed candidate, contest-CPU, or `upstream/evaluate.py`.

Own-vehicle frontier: **lc2 S 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600], UNMOVED.**

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/`; fire trigger: Modal connectivity is available, then execute the sealed GT-field recovery argv, verify SHA-256 `91d3ff11…`, and resume QS3 before any exact waterfill or candidate compile.

## LIVE-HYPOTHESES

- Raising realization from 16.9% past ~30.3% can admit the best current JS6 row at the exact-rung price; this is plausible because all 140 selected target-component pixels already clear the structural quantum proxy, so the dominant loss is not visibly the uint8 floor.
- Receiver-closing deadzone step 2 may restore scale economics: its real archive bracket is 5.0 B/active pair and QS2's six-pair local Pose direction was favorable, but the nine-pair Pose and T4 Seg effects remain unmeasured.
- Exact GT attribution may reveal a profitable subset of the six measured pairs; pair-local scorer effects are separable, so dropping harmful marginals can improve complete S without another SegNet forward once the field is local.

## DEAD-ENDS

- Treating `189−32=157` as a literal set of reverted pixels is closed: the aggregate equations show it equals `2H+R`, so only the matched GT field can identify the actual harmful/retargeted set.
- The exact nine-pair Q3C1 rung does not beat QS2's 5.67 B/pair: it measured 6.1111 B/pair. Do not cite scale amortization on the exact lattice.
- A three-bit QS2 overlay for all nine retained Schur rows is impossible: three rows contain ±6/−4 deltas outside Q2C1's `[-3,4]` domain.
- Quantum-floor dominance on the six selected target components is not supported: 140/140 pixels clear JS6's ≥1 scorer-value strong-support proxy. Do not use the floor as the explanation without the missing authority attribution.
