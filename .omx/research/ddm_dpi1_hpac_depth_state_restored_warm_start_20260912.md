# ddm_dpi1 — the HPAC prior's BIT-DEPTH warm-start state, restored and priced on move 48

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · axis
`[macOS-CPU advisory; exact bytes, scorer-free]` for every byte figure and
`[macOS-MPS research-signal]` for every trainer estimate.
`# FORMALIZATION_PENDING: this arm restores a lost INPUT STATE to an existing law and prices the result with the landed rail; it proposes no new equation. The governing relation is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489 with d_seg and d_pose held by construction (output-lossless decode), so the only term that moves is the rate term and the exchange constant 6.658589531221714e-7 S/B already carries it. A depth-restoration law is registrable only once a SECOND refit under a different lambda or budget confirms the sign of its byte effect; one row is an instance, not a law.`

**STATE (Opus, 2026-09-12) — COMPLETE.** Every deliverable is measured. **The headline is a
negative and it is worth more than the rung it kills: the depth state was NOT a defect.**
Restoring it works exactly as designed — the warm start round-trips to the shipped bytes and the
refit ends at 4.052 bits/row instead of 5.890 — and the exact archive comes out **37 B BIGGER**
(§5). The model leg falls 400 B and the tail pays 437 B: a secant of **−1.0925** against the −1
break-even, so the trade loses. DO NOT FIRE. No seal inputs staged, no Modal, no scorer run, no
candidate claim. This file is crash-resumable.

Frontier line: `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` —
UNMOVED by this arm.

---

## 1. The defect, with receipts — MEASURED

`tools/train_ddm_cl1_hpac_capacity.py` builds the model, calls
`enable_self_compression(model, args.init_bits)` — which registers every compressible module's
`bit_depth` parameter at **8.0 bits** — and then loads the warm start with `strict=False`,
explicitly tolerating every missing `.bit_depth` key (`allowed_missing`, ~line 1193).

| object | `*.bit_depth` tensors in `state_dict` | receipt |
|---|---:|---|
| `ddm_hpr1/train_inputs/init.pt` (the law's warm start) | **0** | sha `cf411127…f383`, 162,409 B |
| `ddm_hv1_harvest_compose/retained/epoch_0634.pt` | **9** | sha `5007beae…47ec`, 1,103,503 B |

So every refit under the reference law starts its 517 row depths at 8 bits and has 30 QAT epochs
to descend. hpr1 measured the consequence without naming the cause: retrained priors land at mean
row depth **5.888–5.890** against the shipped prior's **4.116**, "identical across geometries,
a 60-epoch training-budget effect" (hpr1 §4d). This is the confound signature exactly — a silent
default in the harmful direction.

**Correction to the charter, first of two.** The charter says epoch_0634.pt carries **10**
`bit_depth` tensors. MEASURED: it carries **NINE** — `conv_a`, `conv_b1`, `conv_b2`, `conv_past`,
`frame_scale`, `frame_shift`, `head`, `spm_dw`, `spm_pw`. There is no tenth; `frame_embed` is a
plain embedding and is not a `COMPRESSIBLE_TYPE`. Nine tensors × (8 × 64 + 5) = **517 rows**, which
is exactly the packed body's row count.

---

## 2. The charter's SOURCE is falsified, and the exact ancestor is proven — MEASURED

The charter's mechanism story is "the state was dropped when the EMA init was cut [from
epoch_0634]". MEASURED, from init.pt's own metadata:

```
init.pt["source"] == "move45 shipped hpac member"
```

init.pt was never cut from epoch_0634. It was reconstructed by unpacking move 45's shipped `hpac`
member. And epoch_0634's depths do not restore the shipped state:

| candidate depth source | rows | deployed mean row depth | equals the shipped multiset? |
|---|---:|---:|---|
| `epoch_0634.pt["state_dict"]` (= its EMA shadow) | 517 | **4.2186** bits | **no** |
| `epoch_0634.pt["live_state_dict"]` | 517 | 4.2650 bits | no |
| **cl2 `rungs/lambda_1p0/retained/terminal_checkpoint.pt["state_dict"]`** | 517 | **4.1160541586073505** bits | **YES, exactly** |
| move 45's shipped IHS1 body, read with the receiver | 517 | **4.1160541586073505** bits | — (it IS the target) |

epoch_0634 is 60 epochs UPSTREAM of init.pt. The charter's own acceptance test — "prove the depths
you loaded reproduce the shipped mean row depth (~4.116 bits)" — is met exactly by cl2's λ=1.0
terminal EMA state and **missed by 0.1026 bits** by the source the charter names. The two halves of
the charter disagree; the measurement settles it, and this arm follows the acceptance test.

**The ancestry is proven, not inferred.** `experiments/ddm_dpi1_build_init_depths.py` refuses unless,
for all nine modules, deploying cl2's terminal weights under cl2's OWN bit depths reproduces
init.pt exactly:

| check | result |
|---|---|
| `round(clip(w, ±2^(bits−1))) == init.pt weight`, all 9 modules | **EXACT** |
| `round(clip(bias, ±32768)) == init.pt bias`, all 9 modules | **EXACT** |
| `round(clip(exponent, −6, 0)) == init.pt exponent`, all 9 modules | **EXACT** |
| `round(frame_embed.weight) == init.pt frame_embed.weight` | **EXACT** |

cl2's terminal checkpoint sha `90f7bc38785c71f9bb4e6531318747f4cc3feff2ae7d7b5b31c019b75a905d1e`,
1,051,959 B, epoch 60. Nothing about init.pt's weights moves; only the nine depth tensors return.

---

## 3. The restored initializer, and the flag-vs-constant trap resolved — MEASURED

`init_depths.pt` = init.pt's 28-entry `state_dict` + the nine restored `bit_depth` tensors.

| | value |
|---|---|
| path | `/Volumes/VertigoDataTier/pact/ddm_dpi1/train_inputs/init_depths.pt` |
| sha256 | `f05bae5b2696b1e96817c214d9b1d94ecb64ca18f1c6c30b6e398cd5d4b22072` |
| bytes | 169,056 (mode 0444) |
| receipt | `train_inputs/INIT_DEPTHS.json` |

**The trainer's OWN `bit_depth_histogram` at epoch 0**, from the live run log
(`launch_depths/run.log`, `{"epoch": 0, "resume": false, …}`):

```
{"0": 61, "1": 4, "2": 27, "3": 46, "4": 111, "5": 160, "6": 75, "7": 26, "8": 7}
```

— 517 rows, mean **4.1160541586073505** bits, **identical to move 45's shipped body histogram**,
with `missing_keys == []` and `unexpected_keys == []` (the `allowed_missing` tolerance is no longer
exercised at all).

**The flag-vs-constant trap (rp1 r2), resolved by measurement.** `--init-bits 8.0` is still passed —
it is the law's constant and this arm did not move it. The epoch-0 histogram proves which object the
depths descend from: the STATE, through the checkpoint, not the flag. The flag is now inert for all
nine compressible modules.

**The strongest form of the proof — the restored warm start round-trips to the shipped bytes.**
Packing this run's own epoch-0 checkpoint through the landed IHS1 packer
(`ddm_rx2_mc36_identity_race._pack_terminal_ihs1`) yields:

| | bytes | sha256 |
|---|---:|---|
| packed epoch-0 body | 17,770 | `817281908d993d89f62349fa466ad53f204b7e83d36e29ba0edc30cf2cb8f085` |
| move 45's shipped IHS1 body | 17,770 | `817281908d993d89f62349fa466ad53f204b7e83d36e29ba0edc30cf2cb8f085` |

**BYTE-IDENTICAL.** Receipt `probe/epoch0_pack/EPOCH0_PACK.json`. The trainer's own epoch-0 model
estimate says the same thing in its own units: **17,765 B** here against hpr1's **27,026 B** for the
identical weights without the depth state — a 9,261 B head start the law was throwing away. The
epoch-0 TOKEN estimate is **124,038 B**, byte-for-byte hpr1's control's epoch-0 value, which
independently confirms that no weight moved: only the depths returned.

---

## 4. The rail extension and the `control48` falsifier

`experiments/ddm_hpr1_shape_price.py` gains three things and loses none:

- `PROMOTED48` + `ON_MOVE48` — the move-48 promoted tree as a pricing base, asserted equal to the
  LIVE pointer at call time (never a constant), so a move underneath refuses instead of pricing a
  stale base.
- `control48` — the move-48 falsifier: the shipped prior re-encoded unchanged, which must reproduce
  move 48's archive byte-identically through three independent gates
  (`HPAC_CONTAINER_CONTROL_FAILED`, the no-op check, `LIVE_LOOP_CONTROL_FAILED`).
- `retrain_depths_frame_even` — this arm's candidate. The frame-even rounding is factored out of the
  move-47 composition branch into `frame_rounded_body` so a prior that comes from a RETRAINED
  checkpoint receives the identical edit, and `main` refuses any move-48 treatment until
  `control48` has reproduced the live pointer in this store.

Payloads live under `/Volumes/VertigoDataTier/pact/ddm_dpi1/price` (`--store-root dpi1`); a
successor arm writes under its OWN store and never into hpr1's.

**`control48` model leg — MEASURED** (`price/control48/INPUTS.json`), before the 600-frame encode:

| quantity | control48 | move 48 | Δ |
|---|---:|---:|---:|
| `hpac` archive member | 11,629 B | 11,629 B | **0** |
| IHS1 body | 18,929 B | 18,929 B | **0** |
| rows / stored values | 517 / 20,416 | 517 / 20,416 | 0 / 0 |
| mean row depth | 5.889748549323017 | 5.889748549323017 | 0 |
| `receiver_change` | false | — | — |

The full-archive leg lands when the encode finishes.


**`control48` PASSES — MEASURED** (`price/control48/PRICE.json`, 600 frames, real coder, real
receiver loop):

| object | this rail's re-encode | move 48 | verdict |
|---|---|---|---|
| archive sha256 | `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c` | same | identical |
| archive bytes | 179,111 | 179,111 | **Δ = 0** |
| `hpac` member | 11,629 | 11,629 | identical |
| RLC1 token stream | 118,896 | 118,896 | identical |
| decoded field sha256 | `a92e7d902a449896…` | the shipped field | output-lossless |
| twin encodes | agree | — | deterministic |

**The rail is real on the move-48 base; the price below is admissible.**

---

## 5. The candidate's EXACT PRICE, and the fire verdict — MEASURED

`price/retrain_depths_frame_even/PRICE.json`. 600 frames, real coder, real receiver loop, twins
agreeing, `receiver_change: false`, decoded field equal to `a92e7d90…` — output-lossless by
in-loop assertion, so **no scorer was needed and none ran**. The frame-even rounding moved 2,344 of
the 4,800 `frame_embed.weight` values (`COMPOSITION.json`), so the row is a genuine composition of
the two edits, not one of them.

| leg | candidate | move 48 | Δ |
|---|---:|---:|---:|
| IHS1 body | 17,565 B | 18,929 B | **−1,364 B** |
| mean row depth | **4.052224371373308** | 5.889748549323017 | **−1.8375 bits** |
| rows / stored values | 517 / 20,416 | 517 / 20,416 | 0 / 0 |
| `hpac` archive member | 11,229 B | 11,629 B | **Δmodel = −400 B** |
| RLC1 token stream | 119,333 B | 118,896 B | **Δtail = +437 B** |
| **archive** | **179,148 B** | 179,111 B | **ΔB = +37 B** |
| archive sha256 | `a4dab2375af3d5c5cd35bf388868691ef7786520dbc2a721304d700bcd4731f5` | — | — |

**FIRE VERDICT — DO NOT FIRE.** The bar is net ΔS < −2e-5 ⇔ ΔB ≤ −30.04 B ⇔ archive ≤ 179,080.96 B.
The candidate is **+37 B**, ΔS **+2.4637e-5** — a LOSS, and **67.04 B** the wrong side of the bar.

In the container-break lottery's own units (sd **34.8 B**): the loss is **1.06 σ** and the distance
to the fire bar is **1.93 σ**. So the *sign* of the total is barely outside the noise floor and I
will not over-read it. **The legs are not**: −400 B and +437 B are 11.5 σ and 12.6 σ, and they are
where the finding lives.

**The secant.** Δtail/Δmodel = 437 / (−400) = **−1.0925**. For a model REDUCTION the break-even is a
secant magnitude of exactly 1 — the tail must rise by less than the model falls. At 1.0925 it does
not, and the 37 B is the gap. Compare cl2's **+0.446** for capacity and hpr1's **−1.2066** (naive)
for shape: this axis is the first to land *on* the break-even and fall just past the wrong side.

### 5a. What this settles — the premise this arm was built on is FALSIFIED

The charter, and hpr1's §4a/§4d before it, read the refit's 5.888–5.890 bits/row as an
under-training artefact — "a silent default in the harmful direction" costing **+351 B**. That
reading is now MEASURED FALSE.

Returning the depth state does everything it promised, provably: epoch 0 packs to the shipped bytes
exactly, and the terminal prior compresses to **4.052 bits/row — below even the shipped prior's
4.116**. The archive still gets bigger. **The 5.89 bits was not waste. It was where the λ=1.0
objective puts the model/tail split on the CURRENT token field**, and the split is flat to within
about one container-break σ across a 1.84 bit/row swing. The prior was not badly initialised; it was
sitting near the top of a broad optimum, and the depth axis has no byte left in it at this operating
point.

**The estimator disagrees with the price, in SIGN.** The trainer's terminal ideal-code surrogate
made this run look BETTER than hpr1's dil1 by 772 B (joint 139,943 = token 122,382 + model 17,561,
against dil1's 140,715 = 121,791 + 18,924; both pre-rounding, so the comparison is matched). The
exact encode, with the identical frame-even rounding applied to both, says **+37 B the other way**.
A successor may not rank two priors on this axis with the trainer's estimate — it does not even get
the sign right. Only the encode counts.

**Verdict scope: FORMULATION.** One λ (1.0), one 60-epoch budget, one restoration source, one
rounding composition, one field. The depth axis is measured FLAT-to-slightly-negative here; it is
not closed as a family. What IS closed, and should not be re-opened without new evidence, is the
proposition that the refit's depth inflation is a recoverable defect worth bytes.

### 5b. Held-out honesty — the law exposes none, and the risk tmx1 names does not apply here

MEASURED: `train/depths/result.json` carries **no held-out, validation or split field** — the
trainer's `evaluate()` runs over all 600 frames, and its own `selection_warning` says the surrogate
best (epoch 58) is excluded and that "the preregistered row is the full-state terminal epoch-60 QAT
checkpoint, which must be real-packed, real-encoded, and exactly decoded". That is exactly what was
done here. **So: the law exposes no held-out estimate, and this arm reports none.**

tmx1's risk is fitting on a SAMPLE and losing the gain in transfer to the FIELD. It does not bite
this row, for a reason worth stating precisely rather than waving away: the fit and the price are
both on the SAME full 600-frame field, and the number reported is not an estimate on held-out data —
it is the byte count of an archive that decodes to the shipped field byte-for-byte. The archive
ships this video's code and is scored on this video, so in-sample fitting IS the objective. The
honest residual risk is the container-break lottery (§5), not generalization.

---

## 6. Boundaries honoured

- **No Modal, no paid dispatch, no `authorize_*`, no `fire_modal_auth_eval.py`, no scorer run, no
  seal call, no packet.** Output-losslessness is proved by the in-loop field assertion, so no
  scorer was needed.
- **No seal inputs staged** — the row does not net, and the charter conditions staging on netting.
- Nothing edited under `upstream/`, `submissions/semantic_joint_ctxmix/`, any sealed tree,
  `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, or the receiver. The candidate's
  `receiver_change` is `false` by receipt.
- **No storage reserve lowered.** The rail's 40 GiB fail-closed reserve and the builder's own are
  untouched; 53 GiB free at the end.
- **Every payload on the SSD tier** under `/Volumes/VertigoDataTier/pact/ddm_dpi1/` with sha-verified
  retention (`RETENTION.json`, 19 payload rows, 0 missing, **0.3137 GiB** against the 8 GiB cap).
  Local disk was used for source only.
- **`ddm_ren1` and tmx1 untouched.** The trainer is Metal; the only CPU-heavy job this arm added was
  the single-threaded rail encode, run one at a time at `nice 5` beside ren1's three resolves.
- Every heavy launch went through `tools/launch_detached_process.py` with an armed done-receipt, and
  the trainer additionally through `tools/safe_run.py` (the P0 memory admission gate refused the raw
  launch, correctly, and the refusal is in `launch_depths/run.log`).
- The law's other constants did not move. `--init-bits 8.0`, λ 1.0, seed 20260716, 60 epochs, QAT
  0.5, `past_dilation` 1, `conv_a_dilation` 1, cache pinned by content sha — all cl2's, all as hpr1
  ran dil1. The single delta was `--init`.

## 7. What this does NOT claim

Not claimed: any distortion change (the decoded field is `a92e7d90…` in both rows, byte for byte);
that the +37 B is a confident magnitude (it is 1.06 σ of the container lottery — the LEGS are the
robust part, at 11.5 σ and 12.6 σ); that the depth-axis family is closed (verdict scope FORMULATION:
one λ, one budget, one source, one composition); that the trainer's joint estimate ranks priors (it
predicted a 772 B improvement and the price delivered a 37 B regression — an 809 B swing, sign
included); that a public decode was run (none was — this is a producer row,
not a candidate identity proof); any CPU-axis or contest-CUDA claim.

## 8. Next

1. **Do not re-open "the refit's depths are inflated" as a byte opportunity** without a new
   mechanism. It is measured flat here across a 1.84 bit/row swing.
2. The one live question this leaves is the **λ axis**, not the depth axis: both priors are
   λ=1.0 optima with different model/tail splits and nearly identical joint cost, which is what a
   broad optimum looks like. A λ sweep priced through this rail would map the split directly — but
   cl2 already measured capacity at **+0.446** and this arm measures depth at **−1.0925**, so the
   prior expectation for any pure model/tail re-split on this object should now be "flat", and the
   burden is on a proposal to say why its lever is different.
3. `init_depths.pt` and the rail's `control48` / `ON_MOVE48` base are reusable by any successor that
   wants the move-48 base or a depth-faithful warm start; both are retained with shas.
