# ddm_mc1 — motion-compensated previous-plane ceiling verdict

## Verdict

**CEILING-REFUSED** at **FORMULATION** scope.

On the exact AFR1 n600 decoded field, all three decoder-derived constant-velocity
motion planes made the held-out MI1/DDS1 conditional codelength worse than the
co-located previous-plane baseline. The least harmful candidate was the global
translation: **-136.861 SCREEN bits**, or **-17.108 B** after the charter's
refusal-only division by eight. It misses the pre-registered +5,000 B retraining
gate by **5,017.108 B** and represents **-0.040717%** of the 42,016 B rate demand.
SCREEN bits are not physical coder bytes.

The gate therefore forbids the 60-epoch HPAC retrain and all downstream RC64,
receiver-copy, archive, decode-timing, scorer, Modal, and Metal work. MC1 did not
move the frontier.

Evidence axis: **[macOS-CPU scorer-free conditional-codelength SCREEN, n600]**.

## Inputs and selection

- Exact AFR1 field: 117,964,800 B, SHA-256
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- Exact retained archive anchor: 180,002 B, SHA-256
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
- DF1 coding argmax: 117,964,800 B, SHA-256
  `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e`.
- DF1 coding pmax: 471,859,200 B, SHA-256
  `f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b`.
- Selection: all 600 pairs, never a prefix; pair-level seeded two-fold split,
  seed 20260903. There were 50,009,121 live sites, 67,955,679 exact-pmax-one
  sites excluded by the MI1 family, and 227,671 wrong live sites.
- Motion alignment covers pairs 2 through 599: 598 pairs and 117,571,584 sites.
  Pairs 0 and 1 use the declared co-located fallback.
- APDataStore preflight passed with 32,127,713,280 B free against the required
  1.5 GiB minimum.

Each transform is estimated only from decoded `field[t-2] -> field[t-1]` and
then extrapolated onto `field[t-1]`; no motion parameter is carried. Global and
64-row-band translations use class-balanced, edge-weighted integer agreement.
The affine candidate fits an integer Q12 affine field to 64x64 local integer
translations. The bounded search recorded 0 global, 23 row-band, and 875 affine
local-window boundary hits.

## Ceiling table

The baseline context is `{co-located previous class}` (5 cells). Each candidate
nests it as `{co-located previous class, MC previous class}` (25 cells). Both
are fit independently with the same pair-level two-fold MI1/DDS1 log-odds
offset family on top of `q = 1 - position_coding_pmax`. Positive gain means the
MC context helped. `refusal-only B = gain bits / 8`; it is not a physical byte
measurement.

| model | class | co-located IoU | MC IoU | baseline SCREEN bits | MC SCREEN bits | gain bits | refusal-only B |
|---|---|---:|---:|---:|---:|---:|---:|
| global translation | Road | 0.952221014 | 0.949708109 | 350,831.459 | 350,936.022 | -104.563 | -13.070 |
| global translation | Lane | 0.252516614 | 0.244989559 | 303,864.072 | 303,825.759 | +38.314 | +4.789 |
| global translation | Undrivable | 0.994232868 | 0.993947919 | 99,516.813 | 99,556.638 | -39.825 | -4.978 |
| global translation | Movable | 0.852746080 | 0.850444958 | 89,520.054 | 89,513.515 | +6.539 | +0.817 |
| global translation | MyCar | 0.993029710 | 0.991600846 | 46,448.841 | 46,486.165 | -37.325 | -4.666 |
| global translation | **overall** | macro 0.808949257; agreement 0.987514968 | macro 0.806138278; agreement 0.986880563 | 890,181.238 | 890,318.099 | **-136.861** | **-17.108** |
| row-band translation | Road | 0.952221014 | 0.939425026 | 350,831.459 | 351,226.608 | -395.149 | -49.394 |
| row-band translation | Lane | 0.252516614 | 0.229966556 | 303,864.072 | 303,837.652 | +26.420 | +3.302 |
| row-band translation | Undrivable | 0.994232868 | 0.993585996 | 99,516.813 | 99,529.607 | -12.794 | -1.599 |
| row-band translation | Movable | 0.852746080 | 0.844082270 | 89,520.054 | 89,616.222 | -96.168 | -12.021 |
| row-band translation | MyCar | 0.993029710 | 0.982804513 | 46,448.841 | 46,475.916 | -27.076 | -3.384 |
| row-band translation | **overall** | macro 0.808949257; agreement 0.987514968 | macro 0.797972872; agreement 0.984193970 | 890,181.238 | 890,686.006 | **-504.767** | **-63.096** |
| affine block fit | Road | 0.952221014 | 0.913188991 | 350,831.459 | 350,866.991 | -35.532 | -4.441 |
| affine block fit | Lane | 0.252516614 | 0.140520162 | 303,864.072 | 303,848.805 | +15.267 | +1.908 |
| affine block fit | Undrivable | 0.994232868 | 0.990441644 | 99,516.813 | 99,542.686 | -25.873 | -3.234 |
| affine block fit | Movable | 0.852746080 | 0.815655939 | 89,520.054 | 89,568.052 | -47.998 | -6.000 |
| affine block fit | MyCar | 0.993029710 | 0.967840179 | 46,448.841 | 46,502.345 | -53.504 | -6.688 |
| affine block fit | **overall** | macro 0.808949257; agreement 0.987514968 | macro 0.765529383; agreement 0.977129219 | 890,181.238 | 890,328.879 | **-147.640** | **-18.455** |

The physical alignment result has the same sign as the coding screen. Even the
best global transform reduced overall label agreement by 0.000634405 and macro
IoU by 0.002810979. Lane and Movable gained only +4.789 B and +0.817 B of
refusal-only screen value under that model; Road, Undrivable, and MyCar more
than erased it. Extra geometric flexibility worsened alignment rather than
recovering the near-field classes.

## Retrain receipt

| item | result |
|---|---|
| warm start from shipped integer HPAC | **NOT RUN — prohibited by CEILING-REFUSED** |
| seed / epochs / per-epoch checkpoints / EMA | **NOT CREATED** |
| integer export / model bytes / export SHA | **NOT CREATED** |

One-line retrain reason: an added HPAC context plane is a trained object by
construction, so a passing screen would require the declared warm-start
retrain; this screen did not pass, and the closed-form-first contract forbids
paying for that retrain.

## Exact RC64 and receiver rows

| row | stream B | model B | archive B | identity | repeated encode | CPU decode delta | fraction of 42,016 B demand |
|---|---:|---:|---:|---|---|---|---:|
| AFR1 custody anchor | 113,411 | 13,515 | 180,002 | existing exact field custody | not re-run by MC1 | not re-run by MC1 | 0% baseline |
| MC1 global translation | **NOT RUN** | **NOT RUN** | **NOT BUILT** | **NOT TESTED** | **NOT RUN** | **NOT MEASURED** | -0.040717% SCREEN only |
| MC1 row-band translation | **NOT RUN** | **NOT RUN** | **NOT BUILT** | **NOT TESTED** | **NOT RUN** | **NOT MEASURED** | -0.150171% SCREEN only |
| MC1 affine block fit | **NOT RUN** | **NOT RUN** | **NOT BUILT** | **NOT TESTED** | **NOT RUN** | **NOT MEASURED** | -0.043924% SCREEN only |

The baseline exact receipt binds the 113,411 B stream and 180,002 B archive;
13,515 B is the shipped physical-model component fixed by the charter/JF1
custody. No MC1 candidate has physical stream, model, archive, identity, repeat,
or decode-time numbers. Consequently there is no FIRE ORDER or READY-FOR-T4
row and no score claim.

## Retained evidence and verification

All generated payloads are under
`/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/`.

- `RESULT.json`: 15,366 B, SHA-256
  `478aae86968ca2e6926cb9ad8374ba0920d175fed9b59cd4bae21da74cb1f291`.
- `MANIFEST.json`: 8,515 B, SHA-256
  `892025fef7561537d742ea916d320f1d75c1b036f5631e70b5e78bc43198c1d9`;
  all 34 listed entries re-hashed cleanly, totaling 590,033,890 B before the
  self-excluded manifest.
- `DETERMINISM.json`: 1,193 B, SHA-256
  `7433bfbec37e6fb1a81de58444f3a24809fbdd7d8d8c883a378cdb111e93c09b`.
- Selected global field primary/repeat: both 117,964,800 B and SHA-256
  `785277ffc9360addd459d4c6d50eeb8f3314037814136895593c0282ee6dba90`.
- Selected global motion parameters primary/repeat: both 2,503 B and SHA-256
  `176961454165dc306e13e820a38efbf6f5aa7ea9ad0b7435f5a76e280ba5a3c5`.
- Payload-retention static gate: 1 file examined, 0 findings.
- Runner tests: 4 passed; Ruff and `py_compile` passed. Runner and tests landed
  in commit `2c32e2767b` after two recorded review passes.

## RECALL EVIDENCE

I searched `.omx/research/` and arm receipts by content for
`motion.compens|motion aligned|previous field|previous plane|warp context`,
`constant.velocity|row-dependent shift|affine|temporal alignment`,
`INTER-CAE|Wyner-Ziv|decoder side information|field_geometry_temporal`, and
`previous decoded class|full previous frame|temporal IoU`. I also queried the
canonical equations registry, research index/DAG FEED surfaces, design docs,
the task ledgers, the live hot state, and the actual HPAC/MI1/DDS1 sources.

Beyond the charter seeds:

- `generator_description_online_survey_20260719.md` records the classical
  MPEG-4 INTER-CAE precedent for a motion-compensated previous alpha plane.
  This supported measuring a full decoder-derived plane rather than another
  carried motion packet.
- `ddm_dv3_divergent_weird_ideas_20260818.md` says the full-resolution semantic
  field dictionary/context leg never ran after QA39. This confirmed that the
  exact field-level object remained unmeasured.
- `ddm_qbw2_temporal_bound_verdict_20260827.md` excluded a decoded-carrier shift
  from its gate because it was not geometric Pose6. That distinction kept MC1
  on the lossless rate axis and prevented an unnecessary scorer run.
- The prior xi1 and d3b negatives remained mechanism-scoped: count-table
  dilution and from-zero online mixing are not a trained second HPAC branch.
  They did not pre-refuse this screen, but the new n600 result now supplies the
  charter's own falsifier.

The process followed
[`docs/operating_manual_craft_handoff.md`](../../docs/operating_manual_craft_handoff.md):
source custody was hash-checked before computation, the exact field rather
than the null overlay was used, all materialized payloads and stage checkpoints
were retained, and authority labels were kept separate.

## LIVE-HYPOTHESES

- **None within MC1's adjudicated formulation.** Constant-velocity global,
  64-row-band, and integer-affine extrapolation all worsened physical alignment
  and held-out codelength. Non-constant-velocity flow, scene-cut gating, or a
  learned motion estimator would be a different family and has no fire order
  from this arm.

## DEAD-ENDS

- **FORMULATION closed:** adding a decoder-derived constant-velocity MC plane
  from any of the three tested transforms to the co-located previous-class
  context. Best screen value was -17.108 B versus the +5,000 B gate.
- **Global-shift-only rescue closed:** it was the best candidate, but still
  reduced alignment and lost held-out codelength overall.
- **Ground-plane row flexibility closed:** the row-band model lost 63.096
  refusal-only bytes and degraded Lane and Movable alignment.
- **Integer-affine rescue closed:** additional local/affine flexibility lost
  18.455 refusal-only bytes and produced the largest alignment degradation.
- **Retrain/RC64/archive path folded at the mandatory gate:** retrying it would
  violate closed-form-first and cannot be presented as unfinished evidence.

Own-vehicle frontier: **AFR1 — S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]**; MC1 left it unchanged.
