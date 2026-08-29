# ddm_bhw1 win-win cone re-screen — terminal byte-only verdict

## Verdict

**The required full LD1 cone is a new real rate opening, but the proposed cross-family law is
falsified on the only family that was re-screenable at zero cost.** On
`lane2road_topcost_k060000`, all 5,282 B-labelled edits produce a 176,913 B archive: **−3,476 B
versus the 180,389 B LD1 base, −3,302 B versus the 180,215 B gb1 pointer, and −5.264672473
bits/edit**. However, 5,268/5,282 cells are the already-known DX2 B cone. LD1 itself introduces
only 14 B cells, 0.008325% of its 168,159 disagreements; coding those 14 alone produces 180,390 B,
**+1 B versus the LD1 base and +0.571428571 bits/edit**. This meets the charter's `<0.1%` falsifier
and is non-negative in real bytes.

The full 176,913 B archive is a typed live **rate-only** opening. Its realized `d_seg` and `d_pose`
are **UNMEASURED** because this arm did not own the scorer lane. B/H token labels are not SegNet or
PoseNet outcomes. Axis: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`;
`promotable=false`. The own-vehicle frontier is unmoved.

## Typed byte result

| row | exact B/H/W edits | real archive | marginal vs LD1 base | marginal vs gb1 | real bits/edit | distortion | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| LD1 k060 full top cone | 5,282 / 0 / 0 | 176,913 B, `abc9aa1584c447cf3228b636e5717cfc5627a8c543033882bea7318da50f4aed` | **−3,476 B** | **−3,302 B** | **−5.264672473** | `d_seg`/`d_pose` UNMEASURED | `NEW-LIVE-RATE-OPENING`, mostly inherited DX2 support |
| LD1-induced subset only | 14 / 0 / 0 | 180,390 B, `ad89cf1b1b679a841fc80e879c558ec8344dcf978614e0c0d3ef03f6d84eafd2` | **+1 B** | +175 B | **+0.571428571** | `d_seg`/`d_pose` UNMEASURED | `NO-LD1-SPECIFIC-OPENING`; prior-law falsifier satisfied |

The n600 inverse-coder control was byte-identical: 113,798 emitted token-stream bytes, SHA-256
`6e180f6a7ac0354f1a021ccba6832859fa5d1787a50dcff731857b84bf28892f`. The full and LD1-only
candidate receipts both report `delta_trustworthy=true`; no entropy, average-price, or additive
projection is used in either row.

## Family inventory and re-screenability

The complete machine-readable inventory, including every receipt path, byte count, and full
SHA-256, is `PREPARE.json` at
`/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/PREPARE.json` (47,438 B,
`1b5d0386e47c9db1af723455615d1c65802a757e147e5b41ae20b266751374f3`). The primary retained
objects are enumerated below; path roots are explicit and every digest is full-length.

| family | source-verified search | retained object / coding-argmax status | $0 B/H/W? | scope / honest re-derivation price |
|---|---|---|---|---|
| dg2 diagonal | joint field/model rate↔realized-distortion trade; not win-win-aware | k040/k060 JF2 fields, archives, and refit models retained; family final coding argmax absent | no | `REDUCED`; adapting DX2-hardwired df1 is a mechanism extension. DF1-reference projection: 1,847.018898 s and 4,050,362,934 stage bytes; these rows overlap JF2 |
| jf1/jf2 terminal diagonal | terminal byte winners followed by same-axis Seg/Pose refusal; not win-win-aware | seven fields, archives, and refit models retained; seven final coding argmax fields absent | no | `REDUCED`; DF1-reference projection: 6,464.566144 s and 14,176,270,269 stage bytes, plus model-specific producer adaptation |
| oe1 zero-stored causal escape | lossless causal rate redistribution; not win-win-aware | five members, streams, and decoded fields retained; online-mixture final coding argmax absent | no | `REDUCED`; original five-rung run measured 1,869.164006 s; new argmax payload floor 589,824,000 B; producer/checkpoint extension required |
| ld1 lossy Lane | explicit rate↔token-truth trade across six nested fields; not win-win-aware | six fields/archives retained; model remained DX2, so retained DX2 final coding argmax is the exact family surface | **yes** | `FULL` for all six registered fields; completed here |
| ae1 anti-predicted excess | gross allocation and static-overlay rate accounting; not win-win-aware | result/manifest retained; FS2 predictor argmax is pre-corrector and semantically unsuitable; no finite static-overlay RC64 object exists | no | `REDUCED`; not honestly replay-costable until a physical receiver/coder is built, then its final argmax is persisted |

### Primary object SHA ledger

- DG2 is the k040000/k060000 subset of
  `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/retained/<tag>/retained/`:
  k040 archive `31d99f0beab5d0d665b76cdde66e3e5fb795183b7ac729385af6acb2a1ee4122`, field
  `03ce7bd8a8498ea2a1fc61a0191d0c9eeab3e5ff729e7d522dc07f64add08093`, model
  `04a065b95ee4260f08081d34b15e35f1ddd449fd79cb262111bcde6cee29ecda`; k060 archive
  `59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a`, field
  `15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878`, model
  `98b96ee585f16250b14a05c2202c67541f7717e01c63dfbf068f0af7a714ddc0`.
- JF2 uses the same root. In `(archive, field, selected-model)` order: k002500
  (`32b30b835321e5661613f11756dafba4b0639efafdab4720917bda20f120b152`,
  `c45979acb7a87bdae41fe23d67c9efd10661d5320e5e0c84f9d863a743b3831e`,
  `e40d9b8f30efe52bab4f9866f0e8859b35e96a326afd27a16848958900e20043`); k005000
  (`56447c871b8f7d33f2bfb2edb0e290a24420763b7e15a8b8c4be24a1721c30fb`,
  `6c210dd19eefb2b67dad5c5f93ee8008a625b8aea50e685553ee5335f179f000`,
  `1a317b4ac7bf8f9b97f8c6cd4b7d0a897b670327c9b1bccdd344a3080d31c2c0`); k010000
  (`a06021ba9ca6c4abb893023b664b56d0e063ecfadda08059d8c39b7a93736970`,
  `297cee64f3e1438b985f9b242d6405ad5521b5cf320865390bc0ca105fe8351d`,
  `b628feccf37b66b5007b4cfd13040b2bd313ea1e16d429b0f1b9891c99e86bc9`); k020000
  (`db07e110b4b8f774b9e25811ab9643f39f5b5ec15f504bb31d11bd65c514df7f`,
  `7251367a078796a12c2302d726d2d5b1941c9d35d5755745fc664f29de0344fb`,
  `6ebf74adf9f4180c79c2d7431783b0c87cde19b1433a34b503eec961d09e2bd3`); k040/k060 as
  above; null (`f7b075662d5486382403a3a8a1afa0ff810ecd0b8ca44ba43e1ce0d6be06bb27`,
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`,
  `af8bde55040a9d2843a8db3416f5facd6bf3e3aa45c592a7615869d7e24df177`).
- OE1 root is
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member/retained/rungs/<tag>/`.
  In `(member, stream)` order: control_w0
  (`365f1b8d70463b250a2fe95e3599318ac90b31875cce5d66a767819404431c7a`,
  `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`); escape_w1
  (`21728e895ecd9111c90fc09766c3058e5bd5401585aa2e02827129681b304911`,
  `3c9996561b50b12e59fd7cb24f225434506f0f5d292ac966a16c46b35d171d99`); escape_w4
  (`583dd80139ee94dfaecf6452998dba1489aa088ac3c76ab113b43e203a324544`,
  `d37806b03736e9e86cb874be753db6dc196e7cad5c09272c8b2bc4fa59f00a68`); escape_w16
  (`de427eafff180542ec68e7bc10f1cef0820cd3a1afee60af0a84a0a3f02f7cef`,
  `c8d4f445b507147997e1d802c446113eb4ed4deed339a0933131d88dc0ce190d`); escape_w64
  (`cec655f5715bd6b1174c8d01a8d799cf1d934f9e1ae1951791c24d1525ada4af`,
  `1f45a4bdd59dfec6e75c5ed052e2e26600514e3ed461abd73924e92f6ae2ef3b`). All five decoded
  fields are byte-identical, SHA-256
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- LD1 root is
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/`.
  In `(archive, field)` order: k002500
  (`e5b08a80d1b6cdabfb2fe6c8dd03349cb8d0c6225b488f2201cb9549ebef8f7d`,
  `c45979acb7a87bdae41fe23d67c9efd10661d5320e5e0c84f9d863a743b3831e`); k005000
  (`ce61232b7cc640f14025414d3a294e01fe3f65255824fe4e241b7e903c5b91c1`,
  `6c210dd19eefb2b67dad5c5f93ee8008a625b8aea50e685553ee5335f179f000`); k010000
  (`6df3ac608f8f01d512bfcea8176f46814601c79db69eea937bcd24d64b629e00`,
  `297cee64f3e1438b985f9b242d6405ad5521b5cf320865390bc0ca105fe8351d`); k020000
  (`7d768eb115bc827c4777f258b97a412ef1d0eb6586edfd5d7efe3a523b28ff1e`,
  `7251367a078796a12c2302d726d2d5b1941c9d35d5755745fc664f29de0344fb`); k040000
  (`a83bea6cb435fbeb7f3ab13c5c999a16ab3fcbee4917a9719db958782143b29e`,
  `03ce7bd8a8498ea2a1fc61a0191d0c9eeab3e5ff729e7d522dc07f64add08093`); k060000
  (`2f0891c589afc46c4be4cf04c2d1becfc0206b0e92e72d66f722f81548b67150`,
  `15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878`). Exact unchanged-model
  coding argmax:
  `/Volumes/APDataStore/pact/ddm_df1_dddb_field/measurement_v1/retained/fields/position_coding_argmax.u8.bin`,
  `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e`.
- AE1 result:
  `/Volumes/VertigoDataTier/pact/ddm_ae1_anti_predicted_excess/measurement_v2/RESULT.json`,
  `0554bbad599be921651dcde7174772527dfd801e388bb072a08448330b5d6a6e`; manifest at the same root,
  `20c160b7eb78d8ee4806adca10bb9176da84b25d82034bf6f96eccfdf935a9d3`. The unsuitable FS2
  pre-corrector predictor is
  `/Volumes/APDataStore/pact/ddm_fs2/retained/token_rd/argmax_field.npy`,
  `93cdf71daedd39505c5031aca7cf8524a6358fc862ce838acfbcc1cc73dcae33`.

## Exact B/H/W screen

`B = token wrong and coding argmax == GT`; `H = token == GT and coding argmax wrong`; `W = both
wrong`. These are the landed `classify_pool` definitions, evaluated over all 600×384×512 cells
against DALI-lineage GT SHA-256
`91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`.

| family / row | B | H | W | B share of disagreements | typed outcome |
|---|---:|---:|---:|---:|---|
| dg2 k040/k060 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | `REDUCED-UNSCREENED`: family argmax absent |
| jf1/jf2 seven rows | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | `REDUCED-UNSCREENED`: seven family argmax fields absent |
| oe1 five rows | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | `REDUCED-UNSCREENED`: online-mixture argmax absent |
| LD1 k002500 | 5,268 | 219,362 | 579 | 2.340% | full n600 screen |
| LD1 k005000 | 5,268 | 216,862 | 600 | 2.365% | full n600 screen |
| LD1 k010000 | 5,268 | 211,862 | 648 | 2.419% | full n600 screen |
| LD1 k020000 | 5,268 | 201,862 | 753 | 2.535% | full n600 screen |
| LD1 k040000 | 5,268 | 181,862 | 894 | 2.801% | full n600 screen |
| LD1 k060000 | 5,282 | 161,876 | 1,001 | 3.141% | top full cone; 5,268 B inherited from DX2 |
| LD1-induced part of k060000 | 14 | 0 | 0 | **0.008325% of k060 disagreements** | real price +1 B; `WIN-WIN-VERIFIED-CLOSED` at family-induced scope |
| ae1 static overlays | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | `REDUCED-UNSCREENED`: no physical coder/final argmax |

The six LD1 fields form nested Lane-to-Road trades. The first five add no B cell beyond DX2. Only
k060 adds 14, all transition `0→1`, across 12 pairs and five 60-frame blocks. An independent
full-field implementation reproduced `(top_B=5282, base_B=5268, shared_B=5268,
LD1_induced_B=14)`.

## Per-family exit and revised law

- **DG2 — `REDUCED-UNSCREENED`, trade closure stands only in its original scope.** No family-final
  argmax was found in the bounded JF2 store census. It is not win-win-verified; using DX2's argmax
  would be a fake object. Its two rows are already contained in JF2, so a separate re-derivation
  would duplicate work.
- **JF1/JF2 — `REDUCED-UNSCREENED`, trade closure stands only in its original scope.** The retained
  fields and refit models are real, but the seven matching final coding-argmax fields were not
  persisted. The cheapest high-information extension is one k060 model-specific argmax trajectory,
  not all seven at once.
- **OE1 — `REDUCED-UNSCREENED`, original causal-rate closure stands.** The final online-mixture
  argmax was never retained. DX2 df1 does not contain OE1's expert, so this is an explicit producer
  and checkpoint-schema extension, not replay-only work.
- **LD1 — `WIN-WIN-VERIFIED-CLOSED` at family-induced scope.** Its 14 new B cells are below the
  charter's 0.1% mass threshold and cost +1 real byte. The 5,282-cell full cone is a separate live
  composite opening because it imports 5,268 DX2 opportunities.
- **AE1 — `REDUCED-UNSCREENED`, formulation incomplete for this question.** Its FS2 predictor
  argmax is not a final coding surface, and no static-overlay RC64 object exists. Building that
  object precedes any B/H/W claim.

**Revised law scope:** the prior prediction is **falsified over the complete $0-re-screenable
scope**. The fcd1 win-win cone remains an **INSTANCE result on the DX2 field×model pair**, not a
demonstrated cross-family law. This is not a global nonexistence claim: DG2/JF2/OE1/AE1 remain
reduced-scope unknowns because their semantically matching final coding argmax was not retained.

## RECALL EVIDENCE

I searched beyond the charter seeds by content across `.omx/research/*.md`,
`CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED surface, `main_hot_state.md`, and task-ledger
surfaces with the queries `field-for-coder|coding argmax|B/H/W|win-win cone|surprise premium`, plus
the canonical equation registry using
`.venv/bin/python tools/list_canonical_equations.py --json` filtered for
`token|marginal|direction|surprise|rate`. I also read the actual retained receipts and the producers
`ddm_df1_drop_field.py`, `ddm_oe1_online_escape_member.py`, the JF2 harvest, LD1 rungs/rate curve,
AE1 source binding, and the real joint re-encoder.

Beyond the seeds, recall found:

- `token_rate_model_direction_dependence_v1` and `greedy_set_average_vs_marginal_price_v1`, which
  forbid average/additive pricing and required full-stream real re-encoding;
- DF1's 30-checkpoint n600 reference cost (923.509449 s, 2,025,181,467 stage bytes) and its hardwired
  DX2 constants, which changed four families from apparently cheap replay to declared mechanism
  extensions;
- OE1's final `coding` state exists transiently in its producer but was never persisted, which made
  its re-derivation costable without pretending the object already existed;
- AE1's apparent argmax is FS2 pre-corrector state and its static rows lack physical RC64 objects,
  which removed them from the $0 screen;
- JT23's coder-axis closure and fcd1/fcd2's compensation/pose negatives, which prevented a coder
  rerace and prevented the 176,913 B row from being called a score win;
- the LD1 fields share the DX2 model and exact final coding argmax, which made LD1 the sole honest $0
  screen and motivated the additional 14-cell family-induced control.

No semantically suitable family-final argmax was found in the bounded DG2/JF2, OE1, or AE1 stores.
That is a scoped absence over the listed roots, not a claim that such arrays cannot be regenerated.

## Custody, reproducibility, and boundaries

- Driver: `experiments/ddm_bhw1_winwin_cone_rescreen.py`, SHA-256
  `bf31e6043b0524efaafead0a663e66743da33b4a4f5ae4e81f96d5c772fe072b`.
- Result: `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/REAL_REENCODE_RESULT.json`, 3,282 B,
  SHA-256 `a7e87777b5b7fe66fe67680e3c7b83e17f29abc4c05797032dcbed31f9227f48`.
- Complete artifact manifest: `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/MANIFEST.json`,
  59,826 B, SHA-256 `5bd824f90616f29fbb89e00d0ff46a3103f6d3db402da6a2dcb56ae9c2716c9e`.
  All 231 listed artifacts (270,961,872 B total) were independently rehashed with zero mismatches;
  repeated summarization reproduced the result and final manifest hashes.
- Full-cone field payload: 117,964,800 B,
  `d0f289afae165a8f42ea6289862e6f3dd531109e1278e427f2607634fbae0870`; edit NPZ 768,715 B,
  `74d62675e36e84f0aeb29b699e0e8c061481cde20a12f74c17c171b71670623f`; coordinate NPZ 25,530 B,
  `b5a0ac7a4ad9206f9761c9fe5efe30cceb9cff9c643b93d23c71a5d79b5976f8`.
- LD1-induced field payload: 117,964,800 B,
  `98a98ceefd5b7cbe0ad14ff576c3f4e212648bfd0b363fc0db8dd0f2da56c5f6`; edit NPZ 17,748 B,
  `719cb50344ad0cb6e9e68340b2d950484214cd69f427ffa4111800aa9822fac5`; coordinate NPZ 674 B,
  `3f74aba220f201ad070876f4d1d27dffec317583199fa3e5b39de55549ad6af5`.
- Storage preflight selected APDataStore and passed with more than the 8 GiB minimum before every
  materialization/dispatch. Checkpoints were written atomically every 20 frames and retained; all
  commands/configs and logs are in the manifested store.
- `upstream/` remained read-only. No Modal job, scorer job, MPS authority claim, fcd1/fcd3 consumer
  write, fcd3 lane mutation, or frontier mutation occurred.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN as exclusive scorer-lane owner; consumer store:
  `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/scorer_n600/`; fire trigger: fcd3 reaches a
  terminal state, relinquishes the sole n600 scorer lane, and MAIN explicitly claims it. Run the
  exact retained LD1 base and 176,913 B full-cone archive through the real receiver plus frozen
  SegNet/PoseNet in chunks no larger than 120; retain component outputs and recompute S. Do not
  promote unless the exact distortion legs make joint delta-S negative.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN or a named bhw2 successor; consumer store:
  `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/jf2_k060_rescreen/`; fire trigger: the full
  cone scorer row above is terminal and APDataStore has at least 8 GiB free. Adapt DF1 to persist the
  matching JF2 k060 refit-model final coding argmax with two genuine review passes, then run exact
  B/H/W and a real re-encode if its B mass is at least 0.1%; do not substitute DX2 argmax.
- **FOLDED** — owner: MAIN; consumer store:
  `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/oe1_rescreen/`; fire trigger: JF2 k060
  independently reaches at least 0.1% B with negative real marginal bytes. Only then extend OE1's
  producer/checkpoint schema to retain its five final online-mixture argmax fields; otherwise the
  measured 1,869 s re-derivation is lower expected value than the scorer decision.

## LIVE-HYPOTHESES

- The 176,913 B full cone may survive joint scoring on the LD1 composition: its 3,476 B rate credit
  is real and every edited token moves to DALI GT. This is plausible but weak because fcd2 already
  showed that token-benefit support can be catastrophically pose-hostile; only the queued scorer row
  can decide it.
- A refit JF2 model may create a genuinely new B cone even though LD1 did not: JF2 changes the model
  paired with each field, whereas LD1 kept DX2 fixed. This is why one k060 matching-argmax recovery is
  still worth doing after the live distortion row, despite the current $0 falsifier.

## DEAD-ENDS

- LD1 as independent confirmation of the win-win law is closed at `FORMULATION/FAMILY-INDUCED`
  scope: only 14/168,159 disagreements are new B cells (0.008325%), and their real marginal is +1 B.
- Treating all 5,282 LD1 B cells as second-family evidence is closed: 5,268 are byte-identical DX2
  opportunities already present before the LD1 trade.
- Entropy, average-price, and additive byte estimates are closed for this decision; the two real
  full-stream encodes give −3,476 B and +1 B directly.
- Reusing DX2's coding argmax for DG2/JF2 refit models is closed as a fake-object shortcut; their
  matching argmax surfaces must be regenerated.
- Using AE1's FS2 predictor argmax as a final coding argmax is closed; it is pre-corrector, and the
  retained static formulations have no physical RC64 receiver object.
- Inferring realized SegNet/PoseNet changes from B/H token labels is closed. Distortion remains
  explicitly unmeasured, and the prior fcd2 pose refusal is a warning, not a substitute score.
- Re-racing coders is closed by JT23 and out of scope: this is a field result.

Own-vehicle frontier: gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600], UNMOVED.
