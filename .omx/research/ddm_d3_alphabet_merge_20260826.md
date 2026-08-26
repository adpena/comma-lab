# ddm_d3 alphabet merge — real four-symbol rate win, carrier composition refused

## Verdict

The D3 rate premise is real and substantially stronger than the `lm1` bound: a true four-symbol
RC64 stream, driven by the shipped GB1 five-output HPAC and F26 corrector with Road+Lane probability
pooled at the entropy boundary, is **49,696 B**.  The receiver reproduced all **117,964,800** merged
symbols byte-identically.  Against GB1 this removes **63,928 archive bytes** before Lane carriage.

The selected self-contained Lane carrier does not compose.  Its counted payload is **52,531 B**;
the actual research container is **168,826 B**, only **11,389 B** below GB1.  The full n600 retained
render measured `d_seg = 0.006373087565104167` and `d_pose = 0.7545628308192127`, for advisory
`S = 3.496653766972336`.  The mixed-axis route-triage delta against GB1 is **+3.34853576775973**.
This is strongly uphill, so no seal, Modal job, public runtime, or authority claim was made.

| typed row | stream B | model W B | stream+model B | carrier/framing B | actual archive B | disposition |
|---|---:|---:|---:|---:|---:|---|
| `lm1`/DX2 accounting reference | 113,777 | 13,515 | 127,292 | — | — | recalled bar, not re-encoded here |
| GB1 pinned base | 113,624 | 13,515 | 127,139 | 53,076 other archive bytes | 180,215 | measured base |
| D3 rate-only | 49,696 | 13,515 | 63,211 | 53,076 other archive bytes | 116,287 | receiver-closed rate leg |
| D3 + `block_s3_t3` | 49,696 | 13,515 | 63,211 | 52,531 carrier + 8 D3 framing + 53,076 other | 168,826 | n600 scorer-refused instance |

`W = 13,515 B` is deliberately unchanged.  This arm did **not** train or claim a four-output HPAC.
It changed the real RC64 alphabet to four, pooled `p(Road)+p(Lane)`, and mapped decoded dense symbols
back to canonical `{0,2,3,4}` before every HPAC/F26 feedback edge.  The corrected no1 demand-closing
bar is `stream' <= 85,064 - W = 71,549 B`; the measured 49,696 B stream clears it by **21,853 B**.
Against the actual GB1 stream rather than DX2's 153 B larger stream, the exact saved amount is 63,928 B.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus, research index/DAG feeds, design/tool/experiment code,
canonical task ledger, and canonical equation registry.  Content queries included
`Lane.*Road|Road.*Lane`, `alphabet.*merge|quotient4|five.class|four.symbol`,
`lane_render_band|lane_band_counted_bytes|per.class.*Lane.*crop`, `HPAC.*binary|argmax.right`, and
`lossy.*Lane|boundary.*zero.byte`.  I also ran `tools/list_canonical_equations.py --json` and searched
its output for the same quotient, Lane-carriage, and score-derivative surfaces.

Beyond the charter seeds, the search found four plan-changing facts:

- `hc1`'s 97.80% binary question is **“is HPAC argmax right?”**, not Lane membership.  D3 removes the
  Road/Lane entropy boundary but does not delete the dominant correctness indicator.
- `dg2`/`jf1` partial Lane refits retained a five-class alphabet.  They are evidence about a frozen
  five-slot object, not implementations of this quotient.
- CB1's 2,052 B program+knots are marginal mutations of an inherited video-derived Lane chart that
  GB1 does not contain.  Transplanting only 2,052 B would hide a required parent payload, so the
  claimed analytic-first transplant was refused rather than fake-priced.
- RL1's real per-class Lane crop total is 272,869 B; the often-recalled ~26.5 KB number was margin
  under an older parent representation, not the carrier size.  The source-local exact masks here
  measured 163,304 B best losslessly and therefore cannot fit the D3 credit.

Those findings changed the build from a nominal CB1/crop transplant into (1) a true four-symbol
mechanism, and (2) a self-contained real-coded carrier race whose entire payload is counted.
`msr1` additionally kept zero-byte boundary repair closed; no zero-byte Lane restoration was tried.

## Real coder and receiver proof

| fact | measured value | retained evidence |
|---|---:|---|
| source five-class field | 117,964,800 B, sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | `/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/retained/fields/decoded_tokens_instrumented.u8` |
| merged canonical field | 117,964,800 B, sha `deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07` | `retained/fields/tokens_lane_to_road_canonical.u8` |
| dense four-symbol field | 117,964,800 B, sha `bac40058b83b734f6ca402b22ea8b2c887ac67a7e7c6a38e8c50a8333ab3a6d8` | `retained/fields/tokens_lane_to_road_dense4.u8` |
| true-four RC64 stream | 49,696 B, sha `84fa2f499fb6c052cf6a43f8cae98c227ac32412ce1495cc715aa5af94b8692d` | `retained/encode/token_stream_alphabet4_n600.bin` |
| rate-only archive | 116,287 B, sha `34d4e598136e43d0c162c6ab21ebee65e34c8a72294d8e80b0338d2e3f24e2c6` | `retained/candidates/candidate_d3_rate_only.zip` |
| independent receiver field | 117,964,800 B, sha `deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07` | `retained/decode/tokens_lane_to_road_receiver.u8` |
| generated RC64 source | 14,825 B, sha `15307665764bbf3bce37758cc7f2507e4980ae5d0aeb91f8f9baa38693d2db70` | `retained/build/rc64_alphabet4/rc64_backend_alphabet4.c` |
| generated RC64 library | 34,696 B, sha `1f6f86576de10cb6eae6cec3ef9b0e40b9823c0aa2ef3368cab0c58f84a1e4f2` | `retained/build/rc64_alphabet4/librc64_alphabet4.dylib` |

Encoder and decoder checkpoints preserve the native RC64 state, every live F26 corrector array,
the previous frame, and the partial decoded field at 20-frame boundaries.  The encode resumed from
frame 420; the final independent decode consumed exactly n600 and matched every symbol.

The `lm1` crude KT price of 73,061 B and the real 49,696 B are consistent rather than contradictory:
KT was a deliberately crude alphabet-conditional bound, whereas the real row retains the learned
F26/HPAC causal prior and adaptive corrector.  The real machinery gains another 23,365 B.

## Lane carriage race

All 20 payloads below were actually serialized, Brotli-q11 coded, decoded, and retained.  The screen
metric is exact Lane-mask IoU after the receiver-derived Road-support gate; it is a selection metric,
not a scorer proxy or a negative verdict.  Projected archives omit the 8-byte D3 tag/length framing;
the selected row's actual packed archive is reported in the scorer section.

| carrier | counted B | projected archive B | delta vs GB1 B | Road-gated IoU | recall | precision |
|---|---:|---:|---:|---:|---:|---:|
| `lossless_xor` | 273,464 | 389,751 | +209,536 | 1.000000 | 1.000000 | 1.000000 |
| `lossless_pixel_major` | 163,304 | 279,591 | +99,376 | 1.000000 | 1.000000 | 1.000000 |
| `block_s2_t1` | 94,308 | 210,595 | +30,380 | 0.625093 | 1.000000 | 0.625093 |
| `block_s2_t2` | 87,882 | 204,169 | +23,954 | 0.680515 | 0.912520 | 0.728009 |
| `block_s2_t3` | 67,916 | 184,203 | +3,988 | 0.594412 | 0.633661 | 0.905630 |
| `block_s2_t4` | 55,938 | 172,225 | -7,990 | 0.434881 | 0.434881 | 1.000000 |
| `block_s3_t1` | 60,266 | 176,553 | -3,662 | 0.444384 | 1.000000 | 0.444384 |
| `block_s3_t2` | 56,903 | 173,190 | -7,025 | 0.499728 | 0.957138 | 0.511168 |
| **`block_s3_t3`** | **52,531** | **168,818** | **-11,397** | **0.525530** | **0.895045** | **0.560043** |
| `block_s3_t5` | 42,244 | 158,531 | -21,684 | 0.499509 | 0.617834 | 0.722853 |
| `block_s3_t7` | 21,764 | 138,051 | -42,164 | 0.279164 | 0.288286 | 0.898190 |
| `block_s3_t9` | 11,017 | 127,304 | -52,911 | 0.112738 | 0.112738 | 1.000000 |
| `block_s4_t1` | 43,400 | 159,687 | -20,528 | 0.345200 | 1.000000 | 0.345200 |
| `block_s4_t2` | 41,028 | 157,315 | -22,900 | 0.385655 | 0.974429 | 0.389597 |
| `block_s4_t4` | 35,893 | 152,180 | -28,035 | 0.435269 | 0.876987 | 0.463572 |
| `block_s4_t6` | 30,905 | 147,192 | -33,023 | 0.446906 | 0.711070 | 0.546068 |
| `block_s4_t8` | 22,755 | 139,042 | -41,173 | 0.382350 | 0.479206 | 0.654186 |
| `block_s4_t10` | 13,985 | 130,272 | -49,943 | 0.271467 | 0.294232 | 0.778206 |
| `block_s4_t12` | 9,758 | 126,045 | -54,170 | 0.191608 | 0.197922 | 0.857267 |
| `block_s4_t16` | 2,508 | 118,795 | -61,420 | 0.026601 | 0.026601 | 1.000000 |

Only `block_s3_t3`, selected by the declared highest-IoU-in-rate rule, was promoted to an n600
renderer/scorer row.  Every other distortion and score cell is therefore **NOT MEASURED**, not a
negative.  In particular, the precision-1 `block_s2_t4` row remains an untested alternative.

The selected carrier is 52,531 B, sha
`db0ca0e4f7469ec670c3529c62f5058f032c4b820e8ec0d2252efe6a92792a51`; its Brotli body is 52,529 B,
sha `016a398b5391821a78e3be59548e959ae5feb75552fa4bfb9fadc29629657edb`.  The research renderer
reparsed those exact counted bytes and proved its decoded mask byte-identical to the retained mask
sha `8367c42bfddc0d12b9708cc7afbbd9d069b72b370c70fc877e9a9cb053c7bf2a` before applying paint.

## Realized n600 row and composed arithmetic

Axis: **[macOS-CPU advisory; DALI-GT pinned n600]**.  The candidate raw was produced by the retained
GB1 renderer, then read through the frozen CPU SegNet/PoseNet scorer.  The GT tables are DALI-pinned.
This is a route-triage row; GB1's reference is contest-CUDA, so the delta is explicitly mixed-axis
and non-promotable.

| component | GB1 reference | D3 `block_s3_t3` | delta in S units |
|---|---:|---:|---:|
| archive | 180,215 B | 168,826 B | -0.0075834676171084095 |
| `d_seg` | 0.00020139 | 0.006373087565104167 | +0.6171697565104167 |
| `d_pose` | 0.00000637 | 0.7545628308192127 | +2.738949478866422 |
| recomputed `S` | 0.14811799921260607 | 3.496653766972336 | **+3.34853576775973** |

The candidate distortion-only score is **3.3842394633525323**, so it also fails no1's `<0.12`
distortion-only screen independently of rate.  Its 751,800 SegNet flips decompose as 473,775 Road,
257,285 Lane, 7,040 Undrivable, 7,814 Movable, and 5,886 MyCar.  That composition explains why
highest mask IoU was not a safe score-ordering rule: the false-positive Road paint dominates the
SegNet damage and coincides with catastrophic pose damage.

Retained scored object:

- actual research archive: 168,826 B, sha
  `2c7db149a3f2cf0a1d9a839853d950d20f8e30e290b9e9fefa386ee35aeef5fb`;
- rendered raw: 3,662,409,600 B, sha
  `6c78f63fcdd0f4f5b21751b4c21c1105a571dc22c129ea915a3e9f2715496501`;
- n600 argmax: 117,964,928 B, sha
  `b34feb96fc0d215f6eac9898ecdfa5f9844f9a5d0cf3ba0d175185ee327c2f49`;
- n600 pose6: 14,528 B, sha
  `216559eb1588124fdfb06531575d7979cedfcd98004e057b95bbb1ec40cf90fe`;
- Seg GT: sha `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`;
- Pose GT: sha `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff`.

The archive has a counted, parseable research container, but no public `inflate.py` was emitted.
That work was correctly folded after the n600 row failed; this is not a public-runtime-closed candidate.

## Reconciliation and binary-question identification

`ld1` and this result test different mechanisms.  `ld1` relabelled Lane while retaining five RC64
frequency slots and a frozen five-class probability interface, so every lossy rung grew by 21–1,528 B.
D3 removes the fifth entropy symbol, pools Road+Lane probability, and maps feedback back into the
canonical field.  The exact receiver-closed result saves 63,928 B.  Thus `ld1` remains dead at its
frozen-five formulation while D3's rate leg is verified.  The contradiction disappears at the
alphabet interface; the new blocker is Lane carriage, not entropy rate.

The `hc1` binary variable is **HPAC argmax correctness**.  It is not “is this pixel Lane?”  D3 attacks
one spatially embedded class boundary and therefore benefits the stream, but it does not directly
remove the 97.80% correctness/not-correct decomposition.

## Ledger receipt and routing

The canonical row is `ddm_no1_row3_alphabet_merge`, actor/session `ddm_d3`.  It is completed as an
executed falsifier with an INSTANCE-scoped negative for `block_s3_t3`; the broad D3 formulation is
not declared dead.  Registration, `in_progress`, and completion rows were appended at
`2026-08-26T14:14:43.484269Z`, `2026-08-26T14:14:43.521940Z`, and
`2026-08-26T14:14:43.560528Z`; the completion row has `test_status=green`.  The post-append ledger
snapshot sha is `6e157cd1843852a4922643a21d08fd0048619be80d8cd2fdcada4aa20c1483e0`.

- **FOLDED — authority dispatch/public runtime.** Owner MAIN.  Fire only after a counted carrier has
  an n600 composed delta below zero; this arm produced the opposite result, so no Modal job or seal.
- **FOLDED — trained four-output HPAC refit.** Owner MAIN.  The unchanged five-output source already
  leaves 63,928 B gross credit; refit is not the binding action until a Lane carrier is score-viable.
- **QUEUED-WITH-A-FIRE-ORDER — precision-1 carrier check.** Owner MAIN.  Consumer store
  `/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/scorer/block_s2_t4/`.  Fire when the local
  n600 scorer slot is free: parse the already-retained 55,938 B payload, build the actual D3 container,
  render, and score.  Stop unless its full composed delta is below the `block_s3_t3` row; do not infer
  it from mask metrics.
- **QUEUED-WITH-A-FIRE-ORDER — self-contained analytic carrier.** Owner MAIN.  Consumer store
  `/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/carrier_analytic_next/`.  Fire only when
  the complete source-local Lane chart/program is counted and independently parseable with an actual
  archive below 180,215 B.  Then render and n600-score once; marginal CB1 bytes alone are forbidden.

## GESTALT-DELTA

**D3 is no longer a 38.65 KB rate question: the true quotient creates 63.93 KB of gross rate credit,
but Lane identity is coupled to photometric and pose survival.  The selected self-contained raster
spends 52.53 KB and still destroys Road/Lane segmentation and pose.  The object is rate-positive and
carrier-negative; the next discriminator is precision-controlled or analytic carriage, not more
entropy characterization.**

## Payload custody

Root: `/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/`.

- `PREPARE_RESULT.json`: sha `32a05677a570a2e67ff9868cda23125e1cd35df71d88d035403d73c3aaeb1f5b`.
- `CARRIER_RACE_RESULT.json`: sha `d1ce4dba8f334f3a88547061cc1a5c46a47a8360bd259911e7aea552ef47af6a`.
- `ENCODE_RESULT.json`: sha `b6ad58341891421a678ea537bb0ab5e49ee67bef425a84017ccad33fa4c7e6ec`.
- `DECODE_RESULT.json`: sha `b5c71ad772aa9c3c0d18d22a84a13a69906fb528f629b8937fe7ba1d8899c728`.
- `retained/render/block_s3_t3/RENDER_RESULT.json`: sha
  `5f1cc3e5897da050d3a1068af7207990029e3c2e4a39471fa4523def36a8e4c9`.
- `retained/scorer/block_s3_t3/RESULT.json`: sha
  `9960634bf755f80900df7d96460179e1d0436a72ae3d9e823ba599e94b2fb98e`.

Own-vehicle frontier: **S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**, GB1 archive
sha `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`; **UNMOVED** by D3.
