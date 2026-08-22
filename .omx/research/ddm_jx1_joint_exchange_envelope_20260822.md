# DDM JX1 joint exchange envelope — measured prefix stops at DX2; fixed-distortion residual is 42,382 B

**No prefix of retained, receiver-closed measured moves reaches `S <= 0.12`.** The selected
prefix is already fully consumed by DX2 and ends at **S = 0.14821987563243377 @ 180,368 B**.
There is no measured row left to fire on that body. At the measured distortion the new
representation must remove **42,382 B**; at fixed rate the equivalent distortion request is
**0.028219875632433777 S**, which is impossible because the vehicle has only
**0.028120227975693968 S** of distortion to remove. Even the unphysical zero-distortion endpoint
still needs **150 B** of new rate representation.

`date_utc: 2026-08-22` · `arm: ddm_jx1_joint_exchange_envelope` ·
`axis: [retained-receipt exact arithmetic at the contest-CUDA DX2 operating point]` ·
`score_claim: false` · `promotion_eligible: false` · `Modal/scorer/Metal spend: $0`

No scorer, local advisory, Metal, Modal, archive build, or payload materialization ran in this
arm. `upstream/` was read-only. The live JO r9 directory was neither read nor written.

## 1. Provenance and live arithmetic

All charter pins were re-hashed before adjudication and matched exactly:

| source | required and observed SHA-256 |
|---|---|
| EC2 | `466d75ad05b7cd3489c7c345ba32a5cf7a92a91386be6fa03fe39658dbdb9715` |
| XT1 | `6437bc53d96e527049c3fd6cd60b91af220305881a7bcc68195fece15a728867` |
| TK1 receipt | `5519cce5a986ffd1536233c2f0865a1ce2f95996293f230cb8a0da0f30e09861` |
| TL1 | `d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15` |
| NL1 | `a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a` |
| RB1 exemplar | `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09` |

The VF1 sibling landed during JX1's review at commit `1fe403e7186d9c649a95b36e9d7c3d7603040259`,
memo SHA-256 `a793f90e58f8137bb44287fcde1cc202674f5a4ac16458ee279180281a795ca4`.
It independently proves the same strict threshold: 137,986 B scores
`0.11999944148120990`, while 137,987 B scores `0.12000010734016302`. VF1 stopped before
its token census, so it confirms JX1's one-byte rounding and contributes **zero** invented
load-bearing credit.

DX2 is archive SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
The exact recomputation agrees with the charter; there is no arithmetic disagreement:

| quantity | value |
|---|---:|
| archive bytes | 180,368 B |
| `d_seg` / score term | 0.00020139 / 0.020139 |
| `d_pose` / score term | 0.00000637 / 0.007981227975693965 |
| distortion total | **0.028120227975693968** |
| rate term, `25*180368/37545489` | **0.1200996476567398** |
| recomputed S | **0.14821987563243377** |
| gap to 0.12 | **0.028219875632433777** |

The live tangent prices required by the charter are:

| coordinate | live marginal | useful equivalence |
|---|---:|---:|
| one archive byte | `25/37545489 = 6.658589531221714e-7 S/B` | 1 S = 1,501,819.56 B |
| one unit `d_seg` | `100 S` | `0.001 S = 1.502 B` is wrong; `0.001 S = 1,502 B` |
| one unit `d_pose` | `5/sqrt(10*6.37e-6) = 626.4700137907352 S` | `1e-7 d_pose = 6.264700e-5 S = 94.086 B` |

These are local tangent prices. For large adverse moves such as WD4 or MP2 the exact nonlinear
score from the source receipt is the verdict authority; the tangent value below is still reported
because this charter requires every triple re-priced at the same live operating point.

## 2. Measured-move ledger

Signs are candidate minus its named base, so negative is favorable. `Delta S_live` is the linear
re-price `lambda_B*Delta B + 100*Delta d_seg + 626.4700137907352*Delta d_pose`, never the source
memo's baseline-dependent score. A row is not composable merely because it appears here.

Noise labels are deliberately strict:

- `0, identity` means a retained repeat or decoded-output identity makes the changed score legs
  exactly zero.
- `0, scorer repeat` means the retained scorer repeat itself was identical.
- `UNMEASURED (Q8 only)` means no independent scorer repeat was found; an 8-decimal component
  report is a quantization bound, not a repeat-noise measurement.

### 2.1 Selected ancestry and consumed micro-moves

| move and named base | retained measured triple `(Delta B, Delta d_seg, Delta d_pose)` | repeat-noise floor | `Delta S_live` | status on DX2 |
|---|---:|---|---:|---|
| cp135 lossless recompose vs PR135 | `(-472, 0, 0)` | `0, identity` | `-3.1428543e-4` | carried ancestry; already consumed |
| qs2 vs cp135 | `(+34, -32/117964800, +1.8690116e-10)` | `0, scorer repeat` | `-4.3704437e-6` | five useful events consumed by MC36; pair 532 dropped; never re-fire |
| re1 vs cp135 | `(0, -2/117964800, +8.1081453e-10)` | `0, scorer repeat` | `-1.1874700e-6` | both events consumed by MC36; never re-fire |
| MC36 fresh joint union vs cp135 | `(+17, -37/117964800, +1.0913936e-10)` | `0, scorer repeat` | `-1.9977314e-5` | carried; this is the joint receipt, not `qs2+re1` |
| HV1 HPAC harvest vs MC36 | `(-3510, 0, 0)` | `0, decoded-state identity` | `-2.3371649e-3` | carried; already consumed |
| RR4/free-corrector re-encode vs HV1 | `(-1598, 0, 0)` | `0, decoded-token identity` | `-1.0640426e-3` | carried; already consumed |
| FX1 fixed-point mixer vs RR4 | `(-560, 0, 0)` | `0, decoded-token identity` | `-3.7288101e-4` | superseded alternative; do not add to FX2 |
| FX2 selected model axis vs RR4 | `(-711, 0, 0)` | `0, decoded-token identity` | `-4.7348572e-4` | selected replacement for FX1; carried |
| SZ1 split on FX2 | `(-520, 0, 0)` | `0, decoded-state identity` | `-3.4624666e-4` | later dropped by CK1; not a remaining credit |
| FX2+SZ1 jointly measured generation 3 vs RR4 | `(-1231, 0, 0)` | `0, raw identity` | `-8.1967237e-4` | joint receipt; do not also sum its two legs; only FX2 persists |
| SA3 vs generation 3 | `(-790, +2.04e-6, +4.50e-7)` | `UNMEASURED (Q8 only)` | `-4.0117067e-5` | superseded alternative to keep01 |
| keep01 vs generation 3 | `(-2354, +5.24e-6, +8.40e-7)` | `UNMEASURED (Q8 only)` | `-5.1719716e-4` | carried into CK1; already consumed |
| CK1 vs keep01 | `(-394, +1.74e-6, +5.00e-8)` | `UNMEASURED (Q8 only)` | `-5.7024927e-5` | carried; already consumed |
| CK2 plane2 vs CK1 | `(-657, 0, 0)` | `0, decoded-state identity` | `-4.3746933e-4` | carried; container axis exhausted |
| TO1 tail override vs CK2 | `(-105, 0, 0)` | `0, decoded-token identity` | `-6.9915190e-5` | carried; already consumed |
| UP3 pose splice vs TO1 | `(0, 0, -1.20237e-7)` | `UNMEASURED (Q8 authority row)` | `-7.5324875e-5` | carried; exact-object pose solve consumed |
| BR1 GN pose solve vs UP3 | `(+9, 0, -6.56090e-7)` | `UNMEASURED (Q8 authority row)` | `-4.0502798e-4` | carried; exact-object pose solve consumed |
| JG5 joint waterfill vs BR1 | `(+4196, -1.0170e-4, -6.30e-7)` at T4 report precision | payload repeat `0`; scorer repeat unmeasured; two-row Q8 bound `6.969573e-6 S` | `-7.7707319e-3` | carried; the joint selected object |
| RR5/RC2 rider vs JG5 | `(-169, 0, 0)` | `0, contest raw identity` | `-1.1253016e-4` | carried; already consumed |
| FX5 model-axis coder vs RC2 | `(-70, 0, 0)` | `0, decoded-field identity` | `-4.6610127e-5` | carried; already consumed |
| DX2 CABAC fold vs FX5 | `(-18, 0, 0)` | `0, archive repeat + contest raw identity` | `-1.1985461e-5` | **current endpoint; no residual old-body rate headroom** |

The FX1, FX2, and generation-3 rows are alternatives/joint receipts, not three additive credits.
Likewise SA3 and keep01 share the same generation-3 base. The table retains them because each has a
measured triple, while the reachability column prevents double counting.

### 2.2 Retained refused triples

| refused move and named base | retained measured triple `(Delta B, Delta d_seg, Delta d_pose)` | repeat-noise floor | `Delta S_live` | narrow disposition |
|---|---:|---|---:|---|
| T1H pass 2 vs RR4 | `(+8, 0, +3.6574e-5)` | archive repeat `0`; scorer repeat unmeasured | `+2.2917841e-2` | `CLOSED(FORMULATION)`: CPU-selected carrier moves anti-transfer on this vehicle |
| MP2 mixed q3/q4 vs HV1 | `(-823, +1.14e-6, +5.8376e-4)` | archive repeat `0`; scorer repeat unmeasured | `+3.6527413e-1` | `REFUSED(INSTANCE)`: pose dominates |
| MP2 keep87 vs HV1 | `(-130, +6.40e-7, +5.3643e-4)` | archive repeat `0`; scorer repeat unmeasured | `+3.3603475e-1` | `REFUSED(INSTANCE)`: pose dominates |
| MP2 keep75 vs HV1 | `(-471, +1.07e-6, +4.9212e-4)` | archive repeat `0`; scorer repeat unmeasured | `+3.0809180e-1` | `REFUSED(INSTANCE)`: pose dominates |
| MP2 differential vs HV1 | `(-25, +5.60e-7, +4.0804e-4)` | archive repeat `0`; scorer repeat unmeasured | `+2.5566418e-1` | `REFUSED(INSTANCE)`; the supposedly safe marginal subset was not safe |
| FS3 drop137 vs JG5 | `(-664, +3.33e-6, +4.0424e-4)` | payload controls exact; scorer repeat unmeasured | `+2.5313511e-1` | `CLOSED(FORMULATION)`: stale carrier; exact rescue demand 696x vs measured 8x |
| WD4 trained width64 vs FX5 | `(-13927, +0.03161884, +13.43292362)` | archive repeat `0`; scorer repeat unmeasured | `+8.4184765e3` tangent; exact nonlinear source row `S=14.8829` | `GATE-FAIL(INSTANCE)`: width-only warm slice |

The large tangent losses are not exact-score estimates; they are the required common-operating-point
prices and expose how far these rows lie outside the local regime.

### 2.3 Incomplete rows excluded from the measured envelope

| row | retained evidence | missing coordinate / reason it cannot enter a fire table |
|---|---|---|
| EC2 selected conditioner | 6,670 B benefit, 1,017 H, 97 B gate; projected `Delta S=-0.00394447` with adapter | no receiver archive and no measured pose; projection only |
| EC1 perfect zero-collateral endpoint | measured teacher counts; `S=0.1379837713` arithmetic | impossible zero-collateral assumption, no built receiver candidate |
| FS2 rung 4 | exact `-1022 B`; Seg damage measured/model-bound | no measured candidate pose triple; already loses with pose set free |
| TK1 Route S | 142,001 B label stream and projected 168,892 B route | no composed archive; both distortion coordinates borrowed projections |
| XT1/PK4/DW1 | retained heldout and KD negatives | no complete DX2 archive triple; closed at their named formulations |
| DB1 Family A/boundary variants | retained complete token payloads, decode proofs, deterministic coder repeats | no contest archive/scorer triple; every measured payload is larger than 113,777 B |
| DB1 Family B/C | structural quotient specification only | unimplemented and unmeasured; cannot receive borrowed credit |
| RC1 K=2,048 temporal program | late uncommitted sibling memo reports a retained 59,884 B token payload, 113,006 B shadow container, strict research-token decode, and 98.795970% token agreement | no shipping full-RGB receiver, no measured `Delta d_seg` or `Delta d_pose`, and no exact evaluator archive; cannot receive score credit |

This exclusion is the NO-FAKE boundary: byte-only, scorer-only, or projected rows remain useful
constraints but are not measured exchange moves.

## 3. The joint envelope

### 3.1 Measured, receiver-closed envelope

The selected receiver-closed ancestry is already inside DX2. Every negative `Delta S_live` row not
inside DX2 is either a superseded alternative, a duplicate view of a joint receipt, or measured on a
retired body. Every retained unconsumed complete triple is adverse. Therefore the feasible measured
set on the live body is the singleton:

`E_measured(DX2) = {(180368 B, 0.00020139, 0.00000637, S=0.14821987563243377)}`.

Its minimum is the pointer itself. The exact residual is `0.028219875632433777 S`.

The campaign's `3.705x` receipt confirms **non-additivity**, but its measured direction needs a
correction: MC36's fresh joint compensation was about **3.705x better** than the naive
`qs2+re1` sum, not an under-delivery. The charter's phrase that this union “under-delivered” is
refuted by BU1/MC36 at event-ID scope. That favorable interaction is already consumed, and its
direction does not transfer to a new body. Thus it cannot be used as a multiplier on EC2 or any
other projection.

### 3.2 UPPER-BOUND PROJECTIONS — not measurements

| row, labelled on its face | construction | projected S | exact residual to 0.12 |
|---|---|---:|---:|
| **UPPER-BOUND PROJECTION: EC2 selected field** | subtract EC2's `0.00394447 S` projected benefit from DX2 | `0.14427540563243377` | `0.024275405632433772 S` = 36,458 B |
| **UPPER-BOUND PROJECTION: perfect EC1 zero collateral** | all 12,075 fixes, zero harm, fixed DX2 rate | `0.1379837713` | `0.0179837713 S` = 27,009 B |
| **UPPER-BOUND PROJECTION: perfect Seg removal** | set current `d_seg=0`, retain pose and rate | `0.12808087563243378` | `0.008080875632433787 S` = 12,137 B |
| **PHYSICS BOUND, not a candidate: all distortion removed** | set both distortion terms to zero | `0.1200996476567398` | `9.964765673980969e-5 S` = **150 B** |

None reaches 0.12. EC2 and EC1 may not be added to each other: EC2 selects a subset of EC1's
field and has no joint receiver receipt. The all-distortion endpoint is not an achievable move; it
is the proof that rate representation is mandatory under every possible distortion composition.

## 4. Ranked fire table

The **prospective measured fire table is empty**. Adding an unmeasured NR1/EC2/DB1 row would be the
borrowed-number fake prohibited by the charter. The only honest ordered prefix is the already-fired
post-BR1 composition below. Dependency order is binding; within each available stage the retained
joint price, not a borrowed leg sum, selected the row.

| order | move | joint price on its actual object | receiver closure | owner | fire trigger / disposition |
|---:|---|---:|---|---|---|
| 1 | JG5 joint edit+carrier waterfill | exact row `S 0.1561524295 -> 0.1483910014`, `Delta S=-0.0077614281` | exact n600 T4 row; joint re-solve on own renders | MAIN | trigger satisfied; **FIRED/CARRIED** |
| 2 | RR5/RC2 lossless rider | `-169 B`, exact `Delta S=-1.1253016e-4` | contest raw byte-identical | MAIN | trigger satisfied; **FIRED/CARRIED** |
| 3 | FX5 model-axis lossless coder | `-70 B`, exact `Delta S=-4.6610127e-5` | decoded token field identical; exact T4 row | MAIN | trigger satisfied; **FIRED/CARRIED** |
| 4 | DX2 CABAC lossless fold | `-18 B`, exact `Delta S=-1.1985461e-5` | archive repeat and contest raw byte-identical | MAIN | trigger satisfied; **FIRED/CURRENT** |

This prefix ends at `0.14821987563243377`, not 0.12. There is no fifth measured move. RB1 proves
the tested fixed-representation rate headroom is zero after FX5+DX2; DB1 independently shows that
moving sparse support/width metadata across the token packet boundary makes the complete payload
341,855–477,133 B rather than smaller. A late uncommitted RC1 sibling artifact reports enough
new-representation byte mass to cross the bar in a shadow container, but its missing shipping RGB
receiver and scorer coordinates keep it outside this complete-triple fire table.

## 5. The residual, three ways

### 5.1 Fixed distortion — the actual measured target

The largest strict archive at the current distortion is:

`floor((0.12 - 0.028120227975693968) * 37545489 / 25) = 137,986 B`.

Therefore the fixed-distortion demand is **180,368 - 137,986 = 42,382 B**. The continuous gap is
42,381.161 B; strict integer scoring requires the next byte, hence 42,382.

### 5.2 Fixed rate — impossible by 150 B-equivalent

At 180,368 B the required distortion reduction is the entire score gap,
**0.028219875632433777 S**. That is larger than all current distortion
`0.028120227975693968 S` by **9.964765673980969e-5 S**, or 149.6528 byte-equivalent.

At the live tangent this would mean `Delta d_seg=-0.0002821987563243378` if Seg alone paid it,
which exceeds the entire current `d_seg` by `8.08087563243378e-5`; or
`Delta d_pose=-4.504585217363698e-5` if Pose alone paid it, which exceeds the entire current
`d_pose` by `3.867585217363698e-5`. Exact nonlinear arithmetic makes the same point more simply:
zeroing both axes still leaves the rate term above 0.12.

### 5.3 Joint frontier — measured point and minimum-byte physics point

For a realized distortion credit `c` in score units, the strict byte cut is
`ceil((0.028219875632433777-c)/(25/37545489))`. The joint boundary is:

| realized distortion credit `c` | required new-representation cut |
|---:|---:|
| 0, the measured envelope | **42,382 B** |
| 25% of all current distortion | 31,824 B |
| 50% | 21,266 B |
| 75% | 10,708 B |
| 100%, unphysical perfect removal | **150 B** |

Thus the point minimizing **new rate byte mass** is the physics endpoint
`(distortion=0, archive<=180,218 B)`, but it is not a measured candidate. The point minimizing
claims under the retained measured set is DX2 itself, with **42,382 B** still owed. There is no
unit-free scalar notion of “total mass” that can add bytes and distortion without these score
prices; the table is the complete priced Pareto boundary.

## 6. Required axis from RB1 anatomy

RB1's exact physical member anatomy constrains the residual:

| counted stream | DX2 bytes | can alone return 42,382 B at fixed distortion? |
|---|---:|---|
| semantic tokens | **113,777** | **yes in mass**; replacement must be <=71,395 B |
| semantic renderer | 30,856 | no |
| carrier | 22,010 counted physical | no |
| HPAC | 13,515 | no |
| fixed residual / semantic small fields / headers / ZIP | 210 combined around the named rows | no |

The named axis is therefore the **semantic token field / scorer-cell quotient representation**, or
a new cross-stream task-cell body that jointly replaces token+semantic+carrier mass. RC1's late
uncommitted shadow result confirms that this axis can expose enough byte mass, but does not show that
its 1,420,331 changed labels remain in the same evaluator cells. It is not a pose-only route: the
whole current pose term is only 0.00798123 S. It is not another incumbent coder race: RB1's
within-representation ceiling is fully harvested, and DB1 closes fixed-grid dense support plus
one-width-per-group formulations. Family B remains open only as an actual receiver-checkable
quotient whose counted `QPARAM+QCTX+QPAIR+QEVENT` payload is measured, not as a free theoretical
label.

## 7. Prior-law verdict

`verdict_scope: INSTANCE x CENSUS` — the retained complete-triple census named in this memo, on the
DX2/RC2 ancestry through 2026-08-22; not a family nonexistence claim.

- **CONFIRMED qualitatively:** no composition of currently retained measured moves reaches 0.12,
  and none comes within one admissible measured move. The live fire table is empty.
- **REFUTED quantitatively:** the registered “20,000–40,000 B” residual band misses the measured
  fixed-distortion residual. The answer is **42,382 B**, 2,382 B / 5.96% above its upper edge.
- **REFUTED in the cited union direction:** the 3.705x MC36 receipt was favorable joint
  re-compensation versus the naive sum. What survives is non-additivity, not a universal adverse
  multiplier.
- The 27,009 B perfect-EC1 projection falls inside the prior band, but it is an impossible
  zero-collateral upper-bound projection and cannot rescue the prediction about the **measured**
  envelope.

## RECALL EVIDENCE

The recall pass went beyond the charter seeds before the envelope was priced:

- Full-corpus research queries covered `joint`, `exchange`, `waterfill`, `union`, `banked`,
  `d_seg`, `d_pose`, `delta_bytes`, `receiver-close`, `repeat_noise`, `180625`, `180456`,
  `180386`, `180368`, `42,382`, `113777`, `token drop`, `carrier re-solve`, `pose re-solve`,
  `semantic edit`, and the named arm IDs across `.omx/research/`, the canonical research index,
  and `sub015_DAG_*` FEED blocks.
- The canonical equation registry was queried with
  `.venv/bin/python tools/list_canonical_equations.py --json`; relevant surviving laws were
  `score_marginal_lagrange_multipliers_v1`, `compensated_semantic_edit_exchange_v1`,
  `section_coding_axis_closure_v1`, `token_rate_model_direction_dependence_v1`,
  `greedy_set_average_vs_marginal_price_v1`, and `carrier_rate_credit_pose_affordance_v1`.
- Live/task stores were searched for JX1, NR1, DX2, RC1, DB1, VF1, JO, quotient, and the dynamic
  archive ceiling. The live JO r9 run directory itself remained sacred and unread.
- Primary receipt expansion beyond the charter seeds found QS2, RE1, MC36/BU1, the packet
  generation log, SA3, keep01, CK1/CK2, TO1, UP3, BR1, JG5, RC2, FX5, DX2, T1H, MP2, FS2/FS3,
  WD4, R012, PW2, GS2, the landed DB1 and VF1 siblings, and RC1's late uncommitted sibling memo.

That recall changed the plan in four material ways. It replaced the charter's adverse reading of
the 3.705x union with the actual favorable joint-compensation receipt; proved QS2/RE1 and all three
post-JG5 rate moves are already consumed; admitted the large refused triples so the tangent envelope
could not cherry-pick winners; and added DB1's newly landed boundary negative while keeping its
unmeasured Family-B quotient outside the fire table. VF1 then landed during review and independently
confirmed the strict 137,986 B / 42,382 B arithmetic, but it explicitly stopped before a token census,
so JX1 booked no extra credit. RC1 appeared untracked during the final review and reported a
113,006 B shadow container, but not a shipping receiver or scorer triple; JX1 therefore queued its
sealed route without treating it as landed evidence or measured score credit.

## Boundaries

- **Measured here:** only exact arithmetic over retained receipts, pin verification, body
  attribution, reachability, and the residual.
- **Not measured here:** no new candidate quality, coder payload, receiver output, scorer value,
  runtime, noise floor, or exact contest score.
- A missing scorer repeat is written `UNMEASURED`; deterministic archive repetition is not passed
  off as scorer repetition.
- “Did not find” claims are bounded to the searched corpus and timestamp. The absence of a measured
  live row is not a claim that no future representation exists.
- No payload existed in memory, so this arithmetic-only arm created no bulky receipt and had no
  payload to discard. The durable artifact is this memo.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER, BLOCKED** — owner: MAIN exact-row and unique n600 scorer-lane owner; consumer store: `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v3/main_fire/`; fire trigger: the live JO solve is terminal and harvested, RC1's code and memo are reviewed and landed, and MAIN owns an idle unique n600 lane; action: integrate exact retained payload `eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164` into a fresh MAIN-owned full-RGB DX2 receiver, retain exact repeats and every-paid-section mutation controls, then score only that exact archive and fold unless it improves the canonical pointer and remains at or below the then-current strict byte ceiling.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / NR1; consumer store: `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/retained/`; fire trigger: live JO r9 is terminal, its frozen endpoint/provenance are harvested without JX1 reading the sacred live directory, and a reviewed executable supplies a real receiver-checkable counted quotient; action: actual-code `QPARAM + QCTX + QPAIR + QEVENT`, retain every candidate payload and repeat, and admit only a complete receiver-closed archive measured against the then-recomputed fixed-distortion ceiling (`137,986 B` on DX2). Fold DB1 Family C into this action unless a receiver-runnable non-delta posterior exists.

## LIVE-HYPOTHESES

- A scorer-cell quotient can replace enough of the 113,777 B token member because exact token
  identity is stronger than evaluator-cell equivalence; this is plausible in mass because a
  <=71,395 B replacement closes the measured fixed-distortion gap, but no executable quotient is
  measured yet.
- Jointly training the quotient with the renderer/carrier may move left along the priced boundary
  and lower the required byte cut; this is plausible because JG5 and MC36 both showed that
  exact-object joint solving can outperform separately finished legs, but their interaction factor
  is not transferable.
- A receiver-runnable non-delta posterior could make REC/bits-back useful after a Family-B
  quotient exists; this is plausible algebraically because it prices the cell rather than the
  exact token answer, but a delta posterior on the incumbent field refunds zero bits.
- RC1's temporal-program representation may retain enough evaluator behavior despite changing
  1,420,331 token labels; this is plausible because its 113,006 B shadow container clears the
  fixed-distortion ceiling by 24,980 B, leaving `0.0166337 S` of distortion room, but its class-1
  IoU of 0.146 makes the shipping receiver and exact scorer load-bearing.

## DEAD-ENDS

- Re-firing QS2, RE1, RR5, FX5, or DX2: closed because their useful content is already present in
  DX2; treating them as a bank would double count.
- Summing EC2, EC1, micro-edit, or pose legs into a measured composition: closed because no joint
  receiver receipt exists and the campaign measured non-additivity.
- Incumbent coder/rider search: closed on this body by the fully harvested 88 B FX5+DX2 ceiling.
- FS2/FS3 confidence-threshold token drop: closed at formulation scope by adverse real rate and
  stale-carrier pose respectively.
- T1H CPU-selected carrier solve, MP2 unscreened semantic cuts, and WD4 width64: closed at their
  named scopes by measured wrong-sign authority/advisory triples, not by lack of rate credit.
- DB1 fixed-grid dense support and one-width-per-group metadata: closed at formulation scope at
  372,049 B and 477,133 B complete payloads; relocating support is not deriving it.
- Promoting RC1 from token agreement or shadow bytes alone: closed as a claim because neither is an
  evaluator-visible score, and the current sibling artifact has no full-RGB shipping receiver.
- Literal EC1 perfection, perfect Seg, or zero distortion as fire rows: closed as candidates because
  they are counterfactual bounds without retained receiver-closed payloads.

Own-vehicle frontier remains **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, DX2 archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; JX1 did not move it.
