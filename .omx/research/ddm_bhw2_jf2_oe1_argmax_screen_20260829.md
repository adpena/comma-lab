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
