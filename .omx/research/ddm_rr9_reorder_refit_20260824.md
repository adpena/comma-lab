# ddm_rr9 — R3 adjudicated: the only lossless reorder this coder admits is byte-neutral, and the other case is not a reorder at all

`verdict_scope`: **FORMULATION** for R3 case (a) on the shipped DX2 object; case (b) is ruled
**OUT OF R3'S SCOPE** by the shipped source, not by measurement. Axis
`[macOS-CPU advisory / scorer-free EXACT byte measurement]`. `score_claim=false`. No scorer ran, and
none was needed.

---

## 0. The premise, first — it HELD

MAIN's charter rests on a premise I was told to verify independently before building anything: that
`to2` and `ad2` *replaced* the HPAC model with generic coders rather than *refitting* it, so their
headlines (+196.07% loss, +34.5% win) are not verdicts on R3.

**Measured, over the four named artifacts in full, case-insensitive:**

| artifact | refit tokens | generic-coder tokens |
|---|---:|---:|
| `experiments/ddm_to2_token_ordering_race.py` | **0** | 35 |
| `.omx/research/ddm_to2_token_ordering_race_20260822.md` | **0** | 10 |
| `experiments/ddm_ad2_addressing_cost_decomposition.py` | **0** | 32 |
| `.omx/research/ddm_ad2_addressing_cost_decomposition_20260822.md` | **0** | 14 |
| **total** | **0** | **91** |

`refit tokens` = {refit, re-fit, retrain, re-train, fine-tune, finetune}. `generic-coder tokens` =
{brotli, lzma, zlib, zstd}. **SEARCH SCOPE: exactly those four artifacts.** No claim is made about
any artifact outside that list.

**The premise HELD.** Neither arm refitted; both replaced. R3 was genuinely unmeasured, and this arm
was correctly chartered. Receipt: `.omx/tmp/arm_receipts_local/ddm_rr9_reorder_refit/PREMISE.json`.

---

## 1. The headline

**R3 is not one cell. It is two, and neither is live.**

- **Case (a) — within-group reorder.** The only permutation this coder admits without breaking
  decodability. **MEASURED byte-neutral at full n600: 113,777 B → 113,777 B, 0 B, 0.000000%,
  ΔS 0.0** on the real RC64 coder, over all 117,964,800 tokens, with the round trip **proven
  lossless by digest**, not asserted.
- **Case (b) — cross-group reorder.** Not a reorder of a fixed coder. The group index expression is
  *simultaneously* the coding partition **and** the causal mask baked into the trained convolution
  weights. Changing it means training a different model — a **MECHANISM** change, and a different
  cell from R3.

**`tokens × HPAC` therefore closes as a family**: R1 refused (dg2, 687×/792×), R2 refused (tba1 D3,
21.62× on the seg leg), R3 case (a) is byte-neutral by measurement, R3 case (b) is out of scope by
construction.

---

## 2. The control — reused, not rebuilt

Per charter I drove `dg2`'s instrument rather than building a second one. Its control **passes in
both directions** and I re-verified its receipt against the shipped pin:

- **POSITIVE**: the unedited field re-encodes to the shipped stream **byte-identically** —
  113,777 B, sha `e2af55e6…`.
- **NEGATIVE**: **one** flipped token out of **117,964,800** is detected, prefix agreement collapsing
  **820.5×** (1641 B → 2 B).

Receipt: `.omx/tmp/arm_receipts_local/ddm_rr9_reorder_refit/CONTROL.json`.

**I also ran my own positive control.** The measurement is a read-only side channel on the real
encode (§4), so its faithfulness is not free — it must be earned. The native-order side encoder
reproduced the primary encode **byte-identically**, same sha256 `3e7676a5…`. A side channel that
did not reproduce the primary would have made every row below inadmissible; the runner refuses on
exactly that condition.

---

## 3. What the coder actually is — and why that settles R3

The four facts below are from the shipped source, and each is load-bearing.

**(i) The group plan.** `cpr1/inflate.py:275-287`:

```python
grid = columns + HPAC_DELTA * rows
for group in range((1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA):
```

With `HPAC_PATCH = 64`, `HPAC_DELTA = 2` (`cpr1/inflate.py:33-34`) that is
`(1+2)·64 − 2 = **190 groups**` per plane. **Confirmed empirically by this arm**: 190 groups/frame,
with first-group sizes `[48, 48, 96, 96, 144, 144, 192, 192]` — exactly one wavefront diagonal tiled
over the 6×8 = 48 patches.

**Verified twice, by independent instruments.** (1) The live encode reported 190 groups/frame with
those sizes. (2) Recomputing the partition from the formula alone in plain numpy — `col + 2·row` over
a 64×64 tile, tiled over 6×8 patches — independently yields **190 groups**, head sizes
`[48, 48, 96, 96, 144, 144, 192, 192]`, max 1,536, total **196,608 = 384×512**, every group
non-empty. The runtime and the formula agree without sharing code.

Incidentally this retires a loose end: to2's `rc64_event` order was built as
`(xx % 64) + 2*(yy % 64)` — **the same expression**. to2 did test the coder's own native order, but
against brotli/lzma/zlib.

**(ii) The traversal IS the trained weight mask.** `cpr1/hpac_integer.py:73-84`:

```python
offset = column - center + delta * (row - center)
if offset < 0 or (type_ == "B" and offset == 0):
    mask[row, column] = 1.0
```

`offset` is the group-index delta of a neighbour. The same `col + delta·row` expression that
partitions the coding groups is the causal mask convolved into the weights. The model is
masked-autoregressive (PixelCNN-class), teacher-forced in training
(`tools/train_ddm_cl1_hpac_capacity.py:1319-1320`).

**(iii) Every within-group surface is order-blind by construction.** In the encode loop
(`ddm_jg2_tail_reencode.py:703-721`) a group's logits are produced one-shot; `corrector.group_state`
snapshots the corrector *before* any symbol in the group is coded; `coding_row` is computed for the
whole group at once; `observe` folds the group in afterwards; the write-back is a scatter. The
corrector's accumulator is fixed-point *specifically* so this holds —
`runtime/free_corrector.py:103-105`:

> "Fixed-point resolution for the expected-mass accumulator, so `np.add.at` sums integers and the
> result cannot depend on summation order."

**(iv) The stream length is a SUM.** `_row_bits` is `−Σ log2 coding[i, symbol_i]`, and the real RC64
stream tracks it to ~1 B. Three independent `jf1` rows:

| row | real stream (B) | ideal `−Σ log2 p` (B) | gap |
|---|---:|---:|---:|
| k002500 | 120,607 | 120,606.4 | 0.6 |
| k010000 | 121,704 | 121,703.7 | 0.3 |
| k040000 | 118,917 | 118,916.7 | 0.3 |

A sum is invariant under permutation of its terms. Combined with (iii), **case (a) must be
byte-neutral** — and §4 measures it rather than resting on the argument.

---

## 4. The measurement — a faithful side channel on the real encode

`jg2.encode_tail` computes `frame_bits += _row_bits(coding, symbols)` immediately before
`encoder.encode(symbols, coding)`, with the *same two arrays*. Hooking `_row_bits` therefore observes
the true coding rows and true symbols and returns the true value, leaving the primary encode
bit-for-bit unperturbed. The hook drives two additional **real RC64 encoders** — one native-order
(my positive control), one permuted **within each group** by an independent seeded permutation.

**Why permuting the encoder's input is the faithful reduction.** A real within-group reorder permutes
the whole aligned triple — `flat_positions`, the logit rows, and the symbols — together. Trace each
consumer: `feature = boundary[flat_positions]·NUM_CLASSES + predicted` is per-row, so it permutes
with the triple; `corrector.group_state(probability, predicted, flat_positions)` looks up causal
neighbours per site, so a consistent permutation yields the same per-site state in a different row
order; `observe` folds via `np.add.at` (order-independent by design, §3(iii)); the write-back is a
scatter. **`encoder.encode` is the only consumer that reads the sequence as a sequence.** So a
consistent permutation of the full triple reduces exactly to permuting the rows handed to the coder —
which is what is measured here. The reduction is the finding, not a shortcut around it.

Byte counts below are **raw RC64 bodies**, not `finish()` payloads: `finish()` returns
`TOKEN_MAGIC(4) + body + zero-pad to a 4-byte multiple` (`route_b_rc64.py:272-283`), and that padding
would quantize the delta to 4 B and could **hide a 1–3 B difference**. The runner extracts the body
exactly as `encode_tail` retains it.

### n600 (the full clip) — the authoritative row

| stream | bytes | sha256 (head) |
|---|---:|---|
| primary encode | 113,777 | `e2af55e641c4f2d3` |
| side, native order | **113,777** | `e2af55e641c4f2d3` |
| side, within-group permuted | **113,777** | `73fc4e68fb0f8a9d` |

- **`reorder_delta` = 0 B = 0.000000% = ΔS 0.0**, against the shipped baseline of **113,777 B**
- **`primary_reproduces_shipped_stream = true`** — the harness re-derived the shipped stream
  **byte-identically**, sha `e2af55e6…`. This is a third, end-to-end positive control on top of
  dg2's.
- side native sha **==** primary sha → side channel faithful at full scale
- permuted sha **differs** → the coding really was rearranged; it cost the same
- ideal bits: native `910209.432142536`, permuted `910209.4321425362` — delta **1.16e-10 bits**,
  float64 ULP
- **114,000 groups permuted (600 × 190), 117,964,800 symbols — the ENTIRE field — with 0 identity
  permutations drawn.** Every token in the clip was moved within its group and the stream length did
  not change by one byte.
- elapsed 698.2 s

### n8 (first 8 pairs) — the validating row

| stream | bytes | sha256 (head) |
|---|---:|---|
| primary encode | 1,642 | `3e7676a558d75fbe` |
| side, native order | **1,642** | `3e7676a558d75fbe` |
| side, within-group permuted | **1,642** | `97d08b0fdbd1f65c` |

- **`reorder_delta` = 0 B = 0.000000% = ΔS 0.0**
- side native sha **==** primary sha → **side channel faithful**
- permuted sha **differs** → the permutation genuinely rearranged the coding; it simply cost the
  same
- ideal bits: native `13129.642601177717`, permuted `13129.642601177718` — delta **1.82e-12 bits**,
  pure float64 summation-order ULP
- 1,520 groups permuted (8 × 190), 1,572,864 symbols, **0 identity permutations drawn** (no
  degenerate no-ops)

### Losslessness — PROVEN, not asserted

The permuted stream was decoded by the **real RC64 decoder** using the permuted coding rows, the
inverse permutation applied, and the field compared by digest:

- 1,520 groups decoded, **1,572,864 symbols checked, 0 mismatched groups**
- recovered sha `e8fd1bb3d5e3f387…` **==** expected sha `e8fd1bb3d5e3f387…`
- `field_bit_identical = true`, `lossless = true`

---

## 5. Pricing the permutation

**0 counted bytes.** The permutation is a seeded deterministic rule computed by generic code, so it
is rule-118 free — the decoder derives it, nothing is stored. It is priced at 0 B, not omitted.

This matters because `ddm_tba1` measured that *naming* any subset of positions costs more than the
subset holds; a **stored** permutation over 117,964,800 sites would be fatal on its own. A
decoder-derivable ordering rule was the only plausible shape, and it is the shape measured here. It
still wins nothing, because the thing it buys is worth zero.

---

## 6. Why case (b) is out of scope, and the ceiling on it

Cross-group reorder is not available as a reorder. By §3(ii) the group index is the trained weight
mask, so permuting groups desynchronises the decoder outright — the codebase records this exact
failure class (`residual_archive.py:582`: *"ddm_rr2 scored S = 27.83 on a desynchronised decoder and
it read as a model failure rather than what it was."*). To make it lossless you must retrain with a
different `delta`/`patch`. That is a different trained model, i.e. a **mechanism** change.

**And there is a ceiling argument that applies to any such move.** For any ordering π, the coder's
cost is

```
cost(π) = H(X) + Σ_k KL( p_k || q_k )
```

By the chain rule `Σ_k H(X_π(k) | X_π(<k)) = H(X)` for **every** π — the joint entropy is
order-invariant. So reordering cannot touch the information content at all; it can only change the
model's **approximation error**. That is exactly why the substitution law holds — *reordering is a
substitute for a context model* (MAIN's charter cites this as `#1201`; I could not resolve that id in
`.omx/state/canonical_task_status.jsonl`, so I cite the law by its content and by the `to2`/`ad2`
rows that anchor it, not by the bare id). Reordering pays only where the model is weak, which is why
it swung ±196% for
brotli/lzma and buys nothing here, where the conditional structure is already fitted. to2's own
measured coherence differential across orders was **+0.0131 pp**, consistent with this.

---

## 7. The prediction, adjudicated

> **Prediction**: a decoder-derivable lossless reorder + refit lands within ±2% of the shipped
> baseline bytes. **Falsifier**: absolute bytes move >2% in either direction.

**CONFIRMED — measured 0.000000%, against a ±2% band.**

But I record plainly that it is confirmed **for a sharper reason than I predicted**. I predicted
±2% because "the refit absorbs exactly what the reorder gave away" — a story about two effects
cancelling. The measured mechanism is stricter: **the model never sees within-group order at all**,
so nothing is given away and there is nothing to absorb. There is no refit to run for case (a),
because a refit fits to a factorization the permutation does not change. My reasoning was the right
shape and the wrong mechanism, and the mechanism is the part that closes the cell.

---

## 8. NOT CLAIMED

- **No claim that a re-architected causal schedule cannot win.** Varying `delta`/`patch` and
  retraining is a *different, untouched* cell. `--delta` (default 2) and `--patch` (default 64) are
  live trainer flags (`tools/train_ddm_cl1_hpac_capacity.py:850-851`), and `ddm_cl1_capacity`'s
  preregistration records that it held **"C64/P64/delta2/D8 fixed"** — the wavefront geometry has
  never been swept. §6's chain-rule ceiling applies to it, but it is a mechanism change, not R3.
- **No claim about `d_seg` or `d_pose`.** The scorer did not run and did not need to: a lossless
  reorder leaves the decoded field bit-identical, so neither can move.
- **No claim that to2's or ad2's numbers are wrong** on their own terms — only that they are not
  verdicts on a refit.
- **No claim about any artifact outside the four named in §0.**
- **The byte row is full-clip n600; the losslessness proof is an 8-pair prefix.** The standing
  prefix-bias law (a prefix of a skewed population is a different population, anchored by `bp2`'s
  5.1× sign-flip; the prefix is a *scene block*) does **not** touch the byte verdict, which is n600
  over all 117,964,800 tokens. It does scope the round trip, which was run at 8 pairs because a
  full-clip round trip would require retaining ~4.7 GB of coding rows. Losslessness is a structural
  property of a consistently-applied bijection, not a population statistic — but the scope is stated,
  not hidden, and a full-clip round trip is the obvious hardening if anyone wants it.

---

## 9. Verdict

**R3 case (a): MEASURED byte-neutral. R3 case (b): out of R3's scope by the shipped source.**
`tokens × HPAC` closes as a family. dx2 remains **distortion-starved, not byte-starved** — tri1's
reading survives this arm intact, and the 42,382 B demand cannot be met from token reordering.

`verdict_scope`: FORMULATION for case (a) on the shipped DX2 object at the measured frame budgets;
case (b) is not closed by measurement but excluded by construction; the re-architected causal
schedule is untouched and explicitly not claimed.

---

## STORES CONSULTED

- `.omx/research/ddm_tri1_triple_composition_and_pair_closure_20260824.md` (commit `da6255c46a`) —
  §3.5 R3 charter, §3.6 pair verdict, the 127,292 B pool
- `.omx/research/ddm_dg2_diagonal_distortion_verdict_20260824.md` + `ddm_dg2_diagonal_reentry_20260823.md` — R1's 687×/792× refusal; the reused control
- `.omx/research/ddm_tba1_token_bit_attribution_20260823.md` — R2/D3; the stored-address cost law
- `.omx/research/ddm_to2_token_ordering_race_20260822.md` + `ddm_ad2_addressing_cost_decomposition_20260822.md` — the premise targets
- `.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` §0 — exchange rate **6.658590e-07 S/B**, cited, not re-derived
- `.omx/research/ddm_cl1_capacity_20260809/PREREGISTRATION.md` — "C64/P64/delta2/D8 fixed"
- Source: `cpr1/inflate.py`, `cpr1/hpac_integer.py`, `cpr1/hpac_integer_sparse.py`,
  `runtime/free_corrector.py`, `runtime/fx1_logistic_mixer_corrector.py`,
  `runtime/fx2_model_axis_corrector.py`, `runtime/residual_archive.py`,
  `experiments/ddm_jg2_tail_reencode.py`, `experiments/ddm_jf1_joint_field_model_refit.py`,
  `experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py`,
  `tools/train_ddm_cl1_hpac_capacity.py`

## PAYLOAD

Retained with sha256 + byte counts under
`.omx/tmp/arm_receipts_local/ddm_rr9_reorder_refit/` and mirrored to
`/Volumes/APDataStore/pact/ddm_rr9_reorder_refit/` (Vertigo is at 100% and was never written).
Both streams are persisted as bytes — native and permuted — never reduced to a scalar.

---

`dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]` — gap to 0.12 = 0.028220 ⇒ shed
42,382 B at fixed distortion, or 150 B at zero distortion.
