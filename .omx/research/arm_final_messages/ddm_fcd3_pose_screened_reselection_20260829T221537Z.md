Executed the charter in full and committed the deliverables as `a78ac3729c`.

The selected `tau_1e-6` rung passed fresh pose publication but failed the realized Seg admission gate:

| Metric | jt21 base | Published candidate | Delta |
|---|---:|---:|---:|
| Bytes | 180,192 | 177,252 | −2,940 |
| `d_seg` | 0.000347400259 | 0.000387463049 | +4.00628e-5 |
| `d_pose` | 0.000147010913 | 0.000146204315 | −8.06598e-7 |
| Recomputed S | 0.1930644845 | 0.1950078089 | **+0.0019433244** |

Disposition: `INSTANCE-REFUSED-SEG-BAND`. The rate reduction and small pose improvement did not compensate for the diffuse Seg regression. No candidate seal, READY order, Modal dispatch, or pointer mutation was created.

Key outcomes:

- Fresh publication passed at `d_pose=5.8495951113985735e-6`, below the `6.375684516735624e-6` ceiling.
- Both close passes produced byte-identical 177,252-byte archives, SHA `a4913f44…`.
- Base and candidate scorers completed sequentially as 38 resumable stages each.
- The formulation-level `<45 B` falsifier did not fire: the publishable rung retained 2,940 bytes.
- The scorer lane is terminally closed as `refused_seg_band`.
- Tests: lint and compilation passed; targeted regressions passed 6/6.
- Unrelated worktree changes and `upstream/` were untouched.

Artifacts: [final memo](/Users/adpena/Projects/pact/.omx/research/ddm_fcd3_pose_screened_reselection_20260829.md), [screen/re-encode implementation](/Users/adpena/Projects/pact/experiments/ddm_fcd3_pose_screened_reselection.py).

Own-vehicle frontier remains unchanged: S `0.14811799921260607` at `180,215 B` `[contest-CUDA T4, n600]`, gb1 SHA `ba1f3830…`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned fcd successor; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd3_pose_screened_reselection/`; fire trigger: MAIN reopens task #1320 with the scorer lane free. Fresh-solve, publish, and sequentially score the retained strict `tau_1e-8` rung.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned position-selector successor; consumer store: a new subtree under `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: the remaining pair-level rung is refused or MAIN folds its expected value. Build a position-level joint Seg/Pose selector with real re-encoding and the same ordered gates.

## LIVE-HYPOTHESES

- The stricter `tau_1e-8` rung may remove enough Seg spill while retaining useful rate credit. It already preserves 2,336 real bytes but has not received fresh compensation or full scoring.
- Position-level joint Seg/Pose selection may separate useful edits from harmful effects hidden within kept pairs. Pair-level damage was diffuse across 404 pairs, while within-pair structure remains untested.

## DEAD-ENDS

- Published `tau_1e-6` is closed for sealing and dispatch because its advisory S worsened by `0.0019433243907622244`.
- Entropy, average-price, and additive-credit estimates are closed; every rung was genuinely re-encoded.
- Carried fcd2 compensation is closed as publication evidence; fresh exact-object compensation was necessary.
- B/H token labels cannot stand in for realized SegNet effects; the full scorer directly falsified that transfer here.
- Another worst-pair-only shortcut is unsupported because the worst ten pairs explain only 6.75% of positive Seg harm.
- Int16 carrier widening remains dominated: int12 already cleared pose for only +25 bytes, leaving no pose deficit worth the estimated ~3,600-byte widening cost.