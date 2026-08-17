# DDM RX1 rate-representation attack: frozen tq1c prior does not transfer to MC36

**Status:** COMPLETE; no exact fire

**Axis:** `[macOS-CPU advisory, scorer-free lossless composition]`

**Verdict scope:** `INSTANCE/FROZEN-TRANSFER` — the exact tq1c IHS1 probability object, MC36 labels, RC64, the MC36 correction table ON/OFF, and the 13 lossless model representations measured here

**Frontier:** unchanged at **S = 0.1619344578804448 @ 186,269 B `[contest-CUDA] n600 T4`**

## Result first

The frozen tq1c HPAC prior is lossless on MC36 but moves rate in the wrong direction. The best RX1 archive is **191,746 B**, or **+5,477 B** versus MC36. It reproduces all 117,964,800 token symbols and the complete 3,662,409,600-byte raw output exactly, but its transplanted probability object costs **+5,462 token bytes** and **+15 model/container bytes**. Its score at unchanged distortion would be **0.16558136736669493** by arithmetic projection, not an evaluator result.

No RX1 candidate is smaller than MC36. The sealed MAIN order is therefore `DO_NOT_FIRE`; this arm ran no scorer, Modal job, GPU job, or `upstream/evaluate.py`, and it did not move a pointer.

## Per-lever measurements

Every archive delta is against MC36 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de` at 186,269 B. Distortion is zero only where exact token and full-raw identity were proved.

| Lever / exact object | Model delta | Token delta | Archive delta | Distortion receipt | Disposition |
|---|---:|---:|---:|---|---|
| tq1c HPAC + MC36 residual table ON + Brotli q10 | +15 B | +5,462 B | **+5,477 B** | all tokens exact; full raw exact | FOLDED; best frozen transfer still loses |
| tq1c HPAC + neutral residual table OFF + Brotli q10 | +15 B | +5,799 B | **+5,814 B** | all tokens exact; no full-raw replay needed after the larger exact stream | FOLDED; table removal loses another 337 B |
| tq1c model representation, custodied XZ -> Brotli q10/q11 | **-204 B within RX1** | 0 B | **-204 B within RX1** | exact IHS1 parse-back | BANKED in every RX1 comparison; insufficient to beat MC36 |
| MC36 residual table ON -> neutral OFF on tq1c | 0 B | +337 B | **+337 B** | all tokens exact | FOLDED; the stored 96-B table pays for itself on this transfer |
| PZ4A sensitivity-allocated precision formulation | not rerun | not rerun | prior bounded result: +500 B gross, -2,232 B net after a 2,732-B map | no RX1 distorted candidate | FOLDED at its prior INSTANCE/FORMULATION scope; no byte credit transferred |
| FD135 same-state ANS/RC64 and CAP1 fixed-field rows | not rerun | not rerun | already closed at +6/+9 B for ANS and -79 B for packed CAP1 | existing exact receipts only | FOLDED as already banked on the unchanged F26 objects |

The model-column comparison uses MC36's 70,835-byte outer model and RX1's 70,850-byte q10 outer model. Residual storage stays 96 B in both. Thus the best archive delta closes exactly: `+15 + 5,462 + 0 = +5,477 B`.

The historical tq1c stage-4 stream is 97,928 B and its XZ model is 14,116 B, but those bytes encode tq1c's own parent-argmax labels. They are not a prediction for MC36. On MC36's exact labels, the same frozen IHS1 produces 120,700 B with the correction table and 121,037 B without it. This measured transfer gap is the mechanism behind the negative.

## Composed least-losing candidate

- archive: `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/retained/candidates/tq1c_table_on/brotli_q10/archive.zip`
- bytes: **191,746**
- SHA-256: `817d3758df5ab9f7ebc0f2668b0d071e315e27738ca57bb4033f79e2987276c0`
- repeat archive: byte-identical, same bytes and SHA-256
- stored member: 191,646 B, SHA-256 `8531e837cd85a16065e3dda3c57e2be2d832139610580517ca9526451010470c`
- RX1 model: 70,850 B, SHA-256 `8a8d0afc92da27b56cbdcb2817688746d5bf64105e90d254a9967bc0b9f43b68`
- RC64 token payload: 120,700 B, SHA-256 `c62ad88df8ec71f082c10488d25820e3bfb752bbf78c1aac36ec2aeee8c5b679`
- compact residual table: 96 B, SHA-256 `db0e56acc6ee6b14c41de7702b9b6d4119b51d782a0bd47a98f9a5dbfac11a63`
- projected score if MC36 distortion is held: **0.16558136736669493 `[arithmetic projection; no scorer]`**

The shipped RX1 receiver restores the exact canonical tq1c IHS1 and the exact MC36 semantic, carrier, residual, and token fields. The receiver smoke caught and fixed a missing inverse for MC36's packed CAP1-plus-compensation carrier before the n600 build; the final copied receiver performs that inverse and fails closed on malformed or trailing RX1 model data.

## Decode and identity receipt

The least-losing candidate was replayed through the lifted F26 CPU receiver with four Torch CPU threads:

- decoded spatial tokens: 117,964,800 B, SHA-256 `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`, exact MC36 identity;
- decoded event order: 117,964,800 B, SHA-256 `f4149ab66096e9de8771d5cf9be1058c543177acc0041fed6c361b73e0820be8`, exact source identity;
- full raw: 3,662,409,600 B, SHA-256 `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`, exact MC36 CPU-raw identity;
- decode-and-render time: 650.509 s; wrapper wall time: 653.620 s; inside the 30-minute constraint;
- durable receipt: `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/retained/cpu_decode/best_rx1/receipts/CPU_DECODE_RESULT.json`.

The n600 probability export was frame-checkpointed for both arms. Native RC64 made 25 durable state checkpoints per arm. The table-ON export took 429.113 s, encode 8.036 s, and strict decode 14.722 s. The table-OFF export took 426.301 s, encode 7.973 s, and strict decode 14.554 s.

## Payload custody and resumability

All materialized payloads remain under `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/`. This includes the decompressed IHS1, XZ and Brotli q0-q11 representations, 1,200 per-frame probability lattices and their receipts, both RC64 payloads, 50 RC64 state checkpoints, both decoded event-order fields, both decoded spatial fields, all 26 candidate members/models/residuals/tokens/archives/repeats, the adapted runtimes, the full raw output, and every result JSON.

No retained payload was deleted, moved, or reduced to a scalar-only receipt. The full raw is intentionally retained because it is the final zero-distortion proof; certify-or-block does not authorize deleting it. The top-level machine verdict is `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/FINAL_RESULT.json`.

The exhaustive retained-tree inventory covers **2,781 files / 6,638,286,404 B**. It is `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/RETENTION_INVENTORY.json`, 691,558 B, SHA-256 `39987df201474ec6e8fb7dcc8c0b9503206b54d3e357dac4b2e5f29c352b7e3e`.

## Sealed exact-fire order

- disposition: `DO_NOT_FIRE`
- owner: `MAIN`
- consumer store: `.omx/state/main_hot_state.md` plus the canonical frontier pointer
- fire trigger: a receiver-valid archive is strictly smaller than MC36 and passes lifted CPU raw identity
- command: none
- reason: every RX1 archive is larger than MC36, so an exact evaluation cannot improve the frontier

## RECALL EVIDENCE

The bounded recall searched `.omx/research/`, arm-final receipts, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG, canonical equations, live hot state, and task/probe ledgers by content for `HPAC self compress tq1c IHS1 label transfer`, `probability object prior exact token identity`, `precision waterfill variable precision PZ4A`, `FD135 section decomposition`, `same-state ANS RC64`, `CAP1 metadata pack`, and `MC36 115238 f0ba4bb4`.

Beyond the charter's seeds, HB2's machine receipt established that tq1c's 97,928-B stream is bound to a different label payload and that deploy-bound-exact IHS1 restoration is mandatory. LP135 established that the named same-state ANS and CAP1 rows were already executed at full form, so rerunning them would duplicate settled work. PZ4A established that its tested sensitivity-allocation formulation cannot clear even a zero-map 2,000-B pre-gate. MC36's memo supplied the exact 115,238-B current token stream and final seven-pair receiver shape. These findings changed the plan from a broad coder/precision rerun into the two highest-value unmeasured objects: exact frozen-prior transfer onto MC36 labels, and table-ON versus table-OFF representation closure, followed by the lossless model-representation race.

The bounded search did not find a retained current-MC36-label tq1c-capacity checkpoint or a prior n600 frozen-tq1c-on-MC36 encode in those scopes. That scoped absence justified this run; it is not a claim that no such object exists elsewhere.

## Verification and boundaries

MEASURED here: exact input pins; 13 lossless tq1c model representations; two full n600 probability exports; two native RC64 payloads; full symbol equality; 26 complete archives and repeats; shipped-receiver parse-back; one full CPU raw identity replay; runtime; and retained-file SHA-256/byte receipts.

NOT measured here: any d_seg or d_pose scorer component, exact contest score, CUDA runtime, adaptive HPAC retraining on MC36 labels, a lossy HPAC precision candidate, a new semantic/carrier representation, or a scorer-aware cross-section change. No result from another label state is promoted onto MC36. The negative closes only this frozen-transfer instance, not the HPAC-retraining family.

Ten focused tests pass. They cover deterministic ZIP framing, RX1 XZ parse-back, malformed/trailing rejection, neutral residual encoding, event/spatial inversion, probability normalization, RC64 state framing, and the payload-retention gate. Both owned Python files pass the strict measure-and-discard scan.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN rate-lane router; consumer store: a new `/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/` retained store plus `.omx/state/main_hot_state.md`; fire trigger: a seeded, resumable current-MC36-label HPAC pre-proof predicts a complete receiver-valid archive below 186,269 B after counted model bytes, with per-stage checkpoints and every payload retained; action: train the probability object on MC36's exact labels, then repeat this whole-container identity race.
- `FOLDED-INTO-RX2` — owner: the RX2 trainer owner; consumer store: RX2's per-stage candidate ledger; fire trigger: the first current-label HPAC checkpoint passes exact IHS1 round-trip and is within 5,477 B of MC36 on the joint model-plus-token accounting; action: fit and race a checkpoint-specific residual correction table rather than transplanting MC36's table or assuming neutral is optimal.

## LIVE-HYPOTHESES

- A probability object trained directly on MC36's labels can reverse the 5,462-B token penalty. This is plausible because tq1c reached 97,928 token bytes on its own training distribution, while RX1 proves the frozen object fails specifically at cross-label transfer rather than at receiver integration.
- Joint current-label HPAC training plus a checkpoint-specific correction table may approach the sub-0.15 rate target. The historical tq1c model-plus-token total was 112,044 B on its own labels, close to the roughly 18-KiB reduction demanded from MC36, and the transplanted MC36 table already saves 337 B versus neutral on RX1.
- A future probability state may change the ANS/RC64 ordering. Existing exact states favor RC64 by only single-digit bytes, so the coder family is closed only on those states and should be reconsidered only after the prior changes.

## DEAD-ENDS

- Frozen tq1c IHS1 on MC36 labels: closed at INSTANCE/FROZEN-TRANSFER scope because the best complete archive is 5,477 B larger despite exact token and raw identity.
- Removing the residual table from the frozen transfer: closed because the neutral table makes the token stream another 337 B larger while retaining the same 96-B wire.
- More lossless compression of this tq1c IHS1 as a standalone cure: closed because q10/q11 save only 204 B versus its XZ and the resulting RX1 model is still 15 B larger than MC36 before the 5,462-B token loss.
- Treating tq1c's historical 97,928-B token stream as transferable: closed because it encodes a different parent-argmax label payload; the real MC36-label stream is 120,700 B.
- Re-running same-state F26 ANS/RC64 or CAP1 packing: closed as duplicate work by LP135's exact +6/+9-B ANS losses and already-banked -79-B CAP1 archive win.
- Re-running PZ4A's tested sensitivity-allocation formulation: closed at its prior INSTANCE/FORMULATION scope because its best gross saving is 500 B and counted net is -2,232 B.
- Firing this losing archive on T4: closed because exact token/raw identity fixes distortion at MC36 while +5,477 B strictly worsens rate, so no evaluator outcome can improve the frontier.

---

## REBASE NOTE (appended 2026-08-16 by `ddm_fb1`) — APPEND-ONLY, nothing above is changed

**The body above was CORRECT WHEN WRITTEN. This note exists so the bar is not consumed stale.**
Per Catalog #110/#113 HISTORICAL_PROVENANCE no line above is rewritten; this is a superseding row.

At the time of writing, the frontier was `S = 0.1619344578804448 @ 186,269 B` (MC36 Variant C).
**It has since moved twice:** `MC36 -> e480b v2 (183,502 B) -> hv1 ep0634`.

**LIVE BASE as of 2026-08-16:**
`S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`,
sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(`.omx/state/canonical_frontier_pointer.json`, `effective_frontier`).

**WHY THIS MATTERS — the staleness runs in the dangerous direction.** The `186,269 B` bar sits
**3,510 B ABOVE what we already ship**. A candidate can PASS the bar written above while scoring
**+0.002337165 WORSE** than the incumbent — 233.7x the 1e-5 naming bar.

**USE THIS INSTEAD — a bar that does not go stale.** `seg + pose` is decode-identical across the
whole `cp135 -> MC36 -> e480b v2 -> hv1` lineage (measured to 1e-15), so only rate moves:

```
sub-0.15  <=>  archive <= 168,345.5977 B      (from the live 182,759 B: cut 14,413.4 B)
beat the incumbent  <=>  archive <  182,759 B  (at equal-or-better distortion)
```

Caveat that travels with the invariant: it is a PURE-RATE target, valid only while distortion is
held. Any candidate that CHANGES `d_seg` or `d_pose` must re-measure against the live pointer.

Full derivation, the repo-wide sweep with its denominator, and the bank-union verdict:
`.omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md`.
