# ddm_dpi1 — the HPAC prior's BIT-DEPTH warm-start state, restored and priced on move 48

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · axis
`[macOS-CPU advisory; exact bytes, scorer-free]` for every byte figure and
`[macOS-MPS research-signal]` for every trainer estimate.
`# FORMALIZATION_PENDING: this arm restores a lost INPUT STATE to an existing law and prices the result with the landed rail; it proposes no new equation. The governing relation is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489 with d_seg and d_pose held by construction (output-lossless decode), so the only term that moves is the rate term and the exchange constant 6.658589531221714e-7 S/B already carries it. A depth-restoration law is registrable only once a SECOND refit under a different lambda or budget confirms the sign of its byte effect; one row is an instance, not a law.`

**STATE (Opus, 2026-09-12) — IN FLIGHT.** Deliverables 1 and 3's inputs are MEASURED and the
restoration is PROVEN byte-for-byte (§1–§3). The 60-epoch Metal refit and the `control48`
falsifier are running; §4 and §5 land when they do. This file is crash-resumable.

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
| bytes | 165,109 (mode 0444) |
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

_(§5 fire verdict, §6 boundaries: pending the two runs.)_
