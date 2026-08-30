# ddm_bhw2 JF2/OE1 argmax screen — rank-1 byte admission

## Verdict

**JF2 k060 is a receiver-closed rate opening, so the charter's rank-1 STOP fired and OE1 was not
run.** The full model-matched JF2 screen found 8,301 GT-benefit cells, 207,809 GT-harm cells, and
920 wash cells over 217,030 coding disagreements. Re-encoding every B cell through the real JF2
RC64 trajectory produced a 174,582 B archive, **−4,210 B** versus its own 178,792 B base and
**−4.057342489 real bits per edit**. The shipped production decoder reproduced the exact
117,964,800-byte benefit field.

This is a byte-only result on `[macOS-CPU frozen-scorer advisory]`:
`score_claim=false`, `promotable=false`, and `d_seg`/`d_pose` are **UNMEASURED**. B/H/W token labels
are not SegNet or PoseNet outcomes. No scorer, Modal job, seal, or frontier mutation occurred.

## Typed byte result

| rank / family / row | selection | exact B/H/W | B share of disagreements | real archive transition | real bits/edit | distortion | verdict |
|---|---|---:|---:|---:|---:|---|---|
| 1 / JF2 / k060000 | all 600×384×512 cells; edit every B cell | 8,301 / 207,809 / 920 | 3.824816846% | 178,792 B → **174,582 B**, **−4,210 B** | **−4.057342489** | `d_seg`/`d_pose` UNMEASURED | `BYTE-ADMITTED-FIRE-MAIN` |
| 2 / OE1 / five rungs | strictly gated after rank 1 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | `FOLDED-RANK1-STOP`; not run |

The base archive is SHA-256
`59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a`; the candidate is
`5e217b85bae8687ea37f80f151a5ee68f3df0a2927609510d0dc58272b61287c`. Its exact receiver output
and retained benefit target are byte-identical at SHA-256
`da09731f140a0ddbd79520004a41cc4a77eab5efd85f3b3d012bf1e48756553a`. The real token stream
transition is 112,318 B (`90d85c19…aa424`) → 108,108 B (`ab7e327b…736bb`); archive framing adds
the same fixed overhead on both sides.

## Prior-law prediction

The prior prediction is **CONFIRMED by rank 1**. LD1's family-induced B share was 0.008325%; the
model-moved JF2 share is 3.824816846%, **459.437× LD1** and 45.944× the prediction's 10× threshold
of 0.08325%. Its real byte marginal is also negative. The stated falsifier cannot hold because JF2
is neither within 2× of LD1 nor non-negative in bytes. OE1 is unnecessary to decide that law and
was stopped rather than spent after admission.

This refutes the proposed model-invariant/LD1-like closure at the measured JF2 k060 instance. It
does not prove every JF2 row or another family has the same cone.

## Producer adaptation and resumability

Commit `74b14a9fe6` added
`experiments/ddm_bhw2_jf2_oe1_argmax_screen.py` (SHA-256
`6521cf2db98ac09b1b1a4b4c9a2c5ce8160ae77db1769c84437fe83b7601d461`) after two genuine review
passes. The adapter:

- replays the JF2 k060 model and FreeCorrector on the exact target-field receiver trajectory;
- persists model-matched final coding argmax, target streams, corrector state, encoder state, and
  distinct atomic checkpoints every 20 frames;
- applies the landed B/H/W classifier against pinned DALI-lineage GT, then performs a full real
  benefit-field re-encode;
- packs the real archive and runs the shipped production decoder to exact field identity;
- contains the separately reviewed OE1 five-rung producer/checkpoint/argmax and candidate-decoder
  path, but did not execute it because JF2's typed admission closed rank 2.

The JF2 original-stream control regenerated 112,318 bytes at SHA-256
`90d85c19df03d35c055aaf68e559910104485de9257c9662b1a764067f4aa424`, exactly matching its family
source. Base replay, candidate replay, and production decode took 679.569 s, 668.621 s, and 660.501
s respectively.

## Storage and custody

The rank-1 storage preflight passed with 17,063,739,392 free bytes against 10,615,116,059 required,
including the 8 GiB reserve; shortfall was zero and no cleanup was attempted. Bulky artifacts are
under `/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/`.

- Result: `jf2/JF2_RESULT.json`, 13,909 B, SHA-256
  `11b69b1bec08b7bd3042a3a9da12d1fac31aada098d692c9975cbfc77e8368b5`.
- Row JSONL: `jf2/JF2_ROWS.jsonl`, 1,814 B, SHA-256
  `c1ce727b32eace2f65a21c08bac312201353bdff67983bf99cad8f43284c42bf`.
- MAIN fire order: `jf2/MAIN_FIRE_ORDER.json`, 2,855 B, SHA-256
  `96c39ccfd045c511251317291864294716913a19aefa9749a37346a6e5b07ca1`.
- Full argmax: 117,964,800 B, SHA-256
  `d3dfc1c8b3816fcd609d4b64682e8d9a1c73825d5a91ba703af2255502e424ab`.
- Run log: 19,588 B, SHA-256
  `7a885275354f1eb0e91d58152d1f84b99471f778dfb44c29988b232718be2548`.
- Manifest: 820 artifacts, 871,265,534 B total, 208,031 B manifest, SHA-256
  `f0fdbec7a511a916b8af59968d1170ca9b7568916e55d947490a548642384c30`; an independent full rehash
  found zero size or digest mismatches.

The source binding re-derived the four BHW1 pins before consumption, pinned the JF2 archive/field/
model, all five OE1 source members/streams/decoded fields, GT, charter, common contract, producer
result/manifest, and implementation. `upstream/` remained read-only.

## RECALL EVIDENCE

I searched the full local corpus by content across `.omx/research/`, arm receipts,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `main_hot_state.md`, source producers, and task-ledger
surfaces using `B/H/W|coding argmax|win-win cone|JF2|OE1|model refit|online mixture|marginal price`,
and queried the canonical equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json` filtered for
`token|rate|marginal|direction|surprise`.

Beyond the charter seeds, recall found:

- `token_rate_model_direction_dependence_v1` and
  `greedy_set_average_vs_marginal_price_v1`; these required a full real re-encode and forbade
  additive or average-bit pricing;
- DF1's exact float32/RC64 winner tie rule; the adapter therefore uses the same `coding_prediction`
  path rather than a new argmax convention;
- JG2's structurally complete corrector checkpoint and native RC64 mirror; this supplied the exact
  resumable producer rather than a proxy encoder;
- the JF2 terminal k060 physical row, including its candidate runtime/model and byte-identical
  112,318-byte refit stream; this narrowed rank 1 to one own-model surface;
- OE1's uniform-mixture ordering law and transient final coding rows; this made its five-rung
  argmax producer implementable, but did not override the charter's rank-1 STOP;
- NA11's MAIN adjudication and DG2 containment: DG2 is already inside JF2, so it was not rerun as a
  duplicate third rank;
- fcd1/fcd2's prior pose refusal: moving tokens to DALI GT plus saving bytes is still not a realized
  score claim, so the scorer follow-on remains mandatory.

This changed the plan by requiring native full-stream encoding, exact tie semantics, production
decode closure, and one grouped-but-gated OE1 implementation. No additional family-final argmax
payload beyond the listed JF2/OE1 sources was used.

## Boundaries and dispositions

- JF2 k060 is `BYTE-ADMITTED-FIRE-MAIN` at **INSTANCE / byte-only** scope.
- OE1 rank 2 is **FOLDED by the mandatory rank-1 STOP**, not negatively measured.
- DG2 remains folded into JF2; AE1 remains out of scope because its physical prerequisite does not
  exist.
- `d_seg`, `d_pose`, realized delta-S, and exact contest score are UNMEASURED. The rate-only
  component would change by −0.002803266193 if distortion were held fixed, but it is not a score.
- The canonical pointer is unchanged because no exact evaluator ran.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / exclusive n600 scorer-lane custodian; consumer
  store: `/Volumes/APDataStore/pact/ddm_bhw2_jf2_oe1_argmax_screen/jf2/scorer_n600`; fire trigger:
  MAIN explicitly owns the sole idle n600 scorer lane after revalidating the base archive
  `59428f07…8c54a`, candidate `5e217b85…1287c`, storage, and runtime. Run the real receiver plus
  frozen SegNet/PoseNet in chunks no larger than 120, retain components, and recompute S from the
  components. Do not promote from the B/H/W labels.

## LIVE-HYPOTHESES

- The JF2 B cone may improve joint score because all 8,301 edits move to DALI GT while the archive
  saves 4,210 bytes. This is plausible on the token/rate axes, but fcd2 showed that GT-benefit token
  edits can still be pose-hostile, so only the queued scorer row can decide it.
- OE1's five final coding argmax fields may be byte-identical because each causal escape row is a
  positive affine contraction toward uniform and therefore preserves class ordering. The exact
  producer path is landed, but this remains untested at n600 because rank 1 required STOP.

## DEAD-ENDS

- The LD1-like/model-invariant prediction is closed at the measured JF2 k060 instance: its B share
  is 459.437× LD1 and its real marginal is −4,210 B.
- Average, entropy-only, and additive byte prices are closed for this decision; the native full
  re-encode supplies the real archive transition.
- Substituting DX2 argmax for JF2 is closed as a fake-object shortcut; the retained model-matched
  field is materially different and source-bound.
- Running OE1 after JF2 admission is closed for this charter by the explicit sequential STOP.
- Rerunning DG2 separately is closed because its rows are contained in JF2; AE1 is closed here
  because its physical coder prerequisite is absent.
- Inferring SegNet, PoseNet, or contest-score movement from B/H/W labels is closed. Those axes are
  explicitly unmeasured.

Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.

---

## MAIN ADJUDICATION (appended 2026-08-29, append-only per Catalog #110/#113)

This section is MAIN's verification and routing of the arm's landed verdict above. It does not
mutate a single measurement. Adjudicated against `ddm_fcd1`, `ddm_fcd2`, `ddm_fcd3`, `ddm_jf1`,
`ddm_jf2`, and `ddm_bhw1` read at source.

### 1. Verification — every number re-derived independently, all reproduce

| claim | re-derived | agrees |
|---|---|---|
| B+H+W = disagreements | 8,301 + 207,809 + 920 = 217,030 | ✓ exact |
| B share 3.824816846% | 8,301 / 217,030 = 0.03824816845597383 | ✓ |
| 459.437× LD1 | 0.03824816845597383 / 8.325e-05 = 459.4374589306165 | ✓ |
| archive Δ −4,210 B | 174,582 − 178,792 | ✓ |
| **stream Δ == archive Δ** | 108,108 − 112,318 = −4,210 | ✓ framing byte-identical both sides |
| bits/edit −4.057342489 | −4,210 × 8 / 8,301 | ✓ |
| rate-only S −0.002803266193 | −4,210 × 25 / 37,545,489 | ✓ |

**The NO-FAKE boundary held.** The charter forbade substituting DX2's argmax for a family that
lacks its own, calling that "a fake object." The base replay regenerated JF2 k060's native stream
at SHA-256 `90d85c19df03d35c055aaf68e559910104485de9257c9662b1a764067f4aa424` — byte-identical to
the value `ddm_jf2_terminal_diagonal_harvest_20260826.md` recorded for that arm in its own payload
table. The 117,964,800-byte coding argmax is therefore JF2's OWN model's, regenerated, not borrowed.
`scorer_ran_here=False`, `seal_created_here=False`, `d_seg`/`d_pose` UNMEASURED — all three honest.

### 2. The prediction is CONFIRMED, and the law it establishes is real

Bar was ≥10× LD1's 0.008325% share. Measured 459.437× — 45.9× past the bar. **Moving the coding
model changes the B/H/W labels by two orders of magnitude.** fcd1's win-win cone is NOT a
model-invariant property of the DX2 field; it is a property of the field×model PAIR. That is a
genuine law and it is the arm's real product.

### 3. The fire order is REFUSED AS WRITTEN — three measured reasons

The arm queued `JF2_TERMINAL_WINNERS_REALIZED_COMPONENTS_N600` asking MAIN for the sole n600
scorer lane. MAIN declines to fire it in that form. Not because the arm erred — it followed its
charter's STOP exactly — but because the routing question was already answered on the live body:

**(a) This exact edit class has been realized TWICE on the pointer body, and refused BOTH times,
on two DIFFERENT axes.**

| arm | what it did | outcome |
|---|---|---|
| `fcd1` | B/H/W screen on DX2 field×model | 5,268 B-cells, **−3,756 B** — byte-positive |
| `fcd2` | realize that union | `INSTANCE-REFUSED-POSE-GATE`: uncompensated d_pose 1.6055e-3 (252× the jt21 base 6.3657e-6); even with **full in-compile Schur compensation** d_pose_after 2.7348e-4 = **42.96× base** |
| `fcd3` | pose-SCREEN the population, then re-solve | pose FIXED (d_pose 5.8496e-6 ≤ base ✓) and −2,940 B kept — **but realized d_seg REGRESSED +4.006279e-5**, net **+0.0019433 S**, `INSTANCE-REFUSED-SEG-BAND` |

Priced on the live lb1 pointer (d_seg 0.00020139, d_pose 6.37e-6, 180,083 B):
fcd2's compensated 42.96× pose ratio costs **+0.044332 S** against a −0.002803 S rate gain —
**15.81× underwater.** fcd3 then bought the pose back and the seg bill arrived instead: a seg
regression **2.05×** the rate gain it purchased.

**(b) The B/H/W label is a token-space GT-agreement label, and fcd3 measured that it does not
predict realized seg.** fcd3's 4,194 exact B positions — 100% GT-benefit by label — produced a
seg regression. bhw2's own boundaries section says this correctly ("B/H/W token labels are not
SegNet or PoseNet outcomes"); fcd3 is the measurement that shows the gap is not merely unmeasured
but **adverse in sign**. That is the campaign-level finding of this arc.

**(c) JF2's base has itself never been distortion-measured.** `ddm_jf2` (2026-08-26) queued
`JF2_TERMINAL_WINNERS_REALIZED_COMPONENTS_N600` and it has never fired. So a scorer row on the
candidate would cost **two** n600 runs (base + candidate) to produce a delta on an object that is
not the pointer, whose lineage's positive control failed by 7,554 B at epoch-2 scope (`ddm_jf1`),
and whose edit class the pointer body has already refused twice.

For completeness, the byte arithmetic that would tempt a fire: JF2's candidate at 174,582 B sits
5,501 B under the lb1 pointer = **13.07% of the 42,097 B sub-0.12 demand** — but only if its
distortion were free, and the two rows above measure that it is not.

### 4. OE1 is NOT closed — na11's rank 2 remains genuinely unscreened

The arm stopped OE1 per the charter's explicit STOP-on-admit, and said so honestly: "OE1 is
unnecessary to decide that law and was stopped rather than spent after admission." Correct for the
LAW. But na11's rank-2 row asked for a SCREEN, and OE1 has none. Its disposition is
`FOLDED-RANK1-STOP`, not measured. Given §3, the value of buying it is now low: the law it would
test is confirmed, and the realization price of the whole class is measured and refused. **Recorded
as: OE1 SCREEN OUTSTANDING, priced ~1,869 s + 590 MB, LOW priority, no fire order.** It is not
signal loss; it is a measurement whose consumer just closed.

### 5. What this arm is worth, stated plainly

The rate opening is real and byte-verified. The score claim does not exist and the arm never made
one. What the campaign gets is a **law**, not a route: the win-win cone survives a model change and
grows 459×, while the realization of that cone has now been refused three times on the live body
(fcd2 pose, fcd3 seg, and by inheritance here). The cone is a property of the *coder*, and the
score is a property of the *scorer* — and this arc measured that those two are not the same object.

### 6. GESTALT-DELTA

The 2×2 of #1215 (field moves × model moves) is now fully populated on the byte axis and the
diagonal is byte-positive in both its measured instances (DX2 −3,756 B, JF2 −4,210 B). But the
**distortion axis of that same 2×2 is refused in every cell that has been realized.** Adding to the
gestalt: *a byte-positive edit set defined by agreement with GT in TOKEN space is not a
distortion-safe edit set in SCORER space, and the sign of the seg error is not predicted by the
token label.* That joins `m144` (dx2 lossless remaining ≈2,009 B, all model-axis) and the
sharp-optimum law as the third independent statement that the sub-0.12 gap is not reachable by
re-coding this object — it requires a different object.

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**
Sub-0.12 gap 0.028030; demand 42,097 B at current distortion.
