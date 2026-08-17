# ddm_hm1 — d(token bytes)/d(counted model bytes), measured on hv1 at $0

Date: 2026-08-16 · Owner: `ddm_hm1` · Axis: `[macOS-CPU advisory / scorer-free byte measurement]`
`score_claim=false` · `promotable=false` · pointer UNMOVED at **0.15959729295498598 @ 182,759 B**.
Workspace + payloads: `/Volumes/APDataStore/pact/ddm_hm1_20260816/`.

## Verdict

**The correction-table branch of the model-byte derivative is CLOSED, and the knee is exactly
where the vehicle already stands.** Across nine realized rungs spanning 1 → 9,000 cells and five
context families, measured on the full n600 field against the shipped decoder's own logits,
**exactly one rung clears break-even — the rung already inside the archive.**

The shipped 100-byte RCF1 table returns **335.5 token bytes**, a slope of **−3.355 B per counted
byte**. The very next rung returns **2.2 B for +145 B**, a slope of **−0.016**. The derivative
collapses by **210× in one step**, and no rung after it recovers: every adjacent slope from r2 to
r8 lies in **[−0.469, −0.000]**, all inside break-even.

**The ceiling is the decisive number.** Take the best token count anywhere on the ladder and
give the table away for **free**: the most any additive correction can buy is
**356.1 B = ΔS 2.371e-4**, against a required cut of **14,413.4 B = ΔS 9.597e-3**. That is
**2.47% of the bar**. This family cannot reach sub-0.15 even with zero byte cost.

**What this does NOT close:** a retrained or widened HPAC network. That branch stays open and
`ddm_cl1`'s built-but-never-fired ladder is still the right instrument for it. My result re-aims
it rather than replacing it — see the consequence below.

## The charter premise I was given, and what is actually true

My charter said the 3.81× return is measured at one prior size and that **nobody has measured the
derivative**. Recall refutes that in four places, and every refutation changed what I built.

**1. There is a PRIOR `ddm_hm1` arm (2026-08-10) that already measured the SHRINK direction.**
`.omx/research/ddm_hm1_20260810/FINAL_REPORT.md` traced a post-hoc D8→D7→D6 frame-conditioning
curve on PR130 and byte-closed the winner at n600. Its dead-end reads: *"Post-hoc removal of
coordinate 5 is closed as a rate win: model −420 B, tokens +484 B, joint +64 B, archive +60 B."*
That is a measured derivative point: **removing 420 model bytes costs 484 token bytes, slope
−1.15** — just past break-even, i.e. the shipped model is marginally worth its bytes at the
shrink margin. Its n120 screen also **reversed sign at n600** (−40 B projected → +60 B actual),
which is why everything below is full-field.

**2. `ddm_ec2` measured a grow-direction point and it lost — but it is CONFOUNDED.**
`/Volumes/VertigoDataTier/pact/ddm_ec2/FULL_SCALE_RESULT.json`, `decode_exact: true`:

| term | banked CL1 control | EC2 | Δ |
|---|---:|---:|---:|
| model (XZ) | 15,088 | 15,168 | **+80** |
| tokens (real Range) | 116,716 | 116,436 | **−280** |
| counted coordinates | 0 | 413 | **+413** |
| container framing | — | 20 | +20 |
| **complete container** | **131,804** | **132,037** | **+233 WORSE** |

Control figures from `.omx/research/ddm_ec2_sparse_event_hpac_20260812.md:21-22`; the +233 B is my
subtraction. The receipt's own `delta_vs_preregistered_bar_bytes: 15321` compares a complete
container to a tokens-only bar and should not be quoted. **EC2 spent +80 B of model AND +413 B of
counted coordinates**, so its −280 B cannot be attributed: 0.55 B/B against the whole spend,
3.5 B/B against the model alone. It closes "add a channel that needs a counted side-payload." It
does not price model capacity, and `ddm_dc1` does not cite it at all.

**3. `ddm_cl1` preregistered this ladder in August and it never fired.**
`.omx/research/ddm_cl1_capacity_20260809/BLOCKED_RECEIPT.md`: `CL1TrainingError: local Metal is
unavailable in this process; CPU substitution is forbidden`. Only the two λ=1.0 controls exist;
**no λ=0.5 or λ=0.25 rung ran and no slope was fitted.** Its break-even is the same one I use.
Re-designing it would have been duplicate work, so I did not.

**4. The charter's literal Stage 0 was STALE.** It asked me to measure the token field's
conditional entropy under richer context. `ddm_dc1` ran precisely that today
(`retained/deep_context_ladder.json`): the best table-free ORACLE at 72,235 contexts is
**144,167 B**, i.e. **+32,057 B WORSE than the shipped 112,110 B stream**. Count-based context is
already dominated by the shipped learned model, so refining it bounds nothing about the learned
family. Re-running it would have produced a number that cannot answer the question. I re-aimed at
the model's own residual instead.

## Why this is a PURE-RATE axis — proved from source

From hv1's own prepared receiver, `runtime/residual_archive.py::decode_production_tokens`:

```
corrected   = base_logits + parts.table.values[feature]
probability = _probability_table(corrected, runtime.HPAC_LOGIT_PRECISION)
symbols     = decoder.decode(probability)
```

The table and the HPAC weights enter **only** the probability handed to rc64. A range decoder
returns what the encoder put in; a shared model change moves the stream LENGTH, never the symbols.
So the decoded token field is bit-identical under any model or table change, the rendered frames
are identical, and **d_seg and d_pose do not move at all**. `feature` is built from the previous
frame's boundary map and `base_logits.argmax` — already-decoded state, so the encoder computes the
same index causally. Every extra feature I add (margin bin, runner-up, previous class, finer
boundary) is likewise a function of already-decoded state.

`ddm_rc4` measured the last link independently: ZIP member `p` is **STORED, not deflated**, so a
byte removed from the token stream is a byte removed from `archive.zip` **1:1**
(`ddm_rc4_rung4_token_drop_verdict_20260816.md:86-88`), and its NEXT #4 states the same
conclusion — *"pure-rate at fixed decoded field."* Therefore `ΔS = 25·Δjoint / 37,545,489` exactly,
and `pure_rate_byte_bar_from_pointer()` **does** apply here, unlike rc4's own rung 4.

Bar read live from the pointer, not copied: **168,345.5977 B, required cut 14,413.402 B**, off
base `0.15959729295498598 @ 182,759 B [contest-CUDA]`.

## Instrument — and its byte-identity gate

`experiments/ddm_hm1_hpac_logit_replay.py` replays the shipped decoder with **teacher forcing**:
the decode is exact, so feeding the decoded token field back as the causal state reproduces the
identical logits with no range coder in the loop. It retains the raw pre-correction logits, which
are exact multiples of 1/8 and therefore lossless in int16 (1,179,648,000 B for n600).

The gate is byte identity against the shipped decode, not a smoke:

| digest | mine | wc1 / dc1 receipt |
|---|---|---|
| `corrected_quantized_logit_sha256` | `562ac652…` | `562ac652…` ✓ |
| `corrected_cdf_input_sha256` | `dd48843b…` | `dd48843b…` ✓ |
| HPAC cross-entropy, n600 | 112,109.57757858819 B | 112,109.57757858819 B ✓ |
| decoded token field sha | `9ba2e52b…` | `9ba2e52b…` ✓ |

The n8 prefix matched dc1's n8 receipt independently (`3212b80d…`, `546a434f…`,
1611.2580243605626 B), and the ladder's `shipped_actual_table` row re-derives dc1's n600
cross-entropy to the byte from the retained logits. `ddm_rc4` reports the same two digests from a
separate instrument.

## The ladder — realized, not oracle

Every `r*` row is a table that is fitted to the global optimum of a convex objective (per-cell
intercepts, damped Newton), quantized to real RCF1 6-bit codes with a searched fp16 scale, and
evaluated through the receiver's own int16-logit-then-softmax pipeline. **Counted model bytes are
`min(raw RCF1 packing, brotli-q11 of the same body) + the fp16 margin-bin thresholds**, because
those thresholds are quantiles of THIS video and are therefore counted under rule 118. Break-even
is slope `< −1`.

| rung | context | cells | samples/cell | counted model B | token B | joint B | slope | pays? |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `r0_no_table` | no correction table at all | 1 | 117,964,800 | 10 | 112,445.1 | 112,455.1 | — | no |
| `r1_shipped_context` | **SHIPPED context** — prev-frame boundary (5) x argmax (5) | 25 | 4,718,592 | 100 | 112,137.1 | 112,237.1 | -3.422 | **YES** |
| `r2_margin4` | + model confidence margin, 4 bins | 100 | 1,493,225 | 245 | 112,134.9 | 112,379.9 | -0.016 | no |
| `r3_margin16` | + margin, 16 bins | 400 | 411,027 | 504 | 112,013.3 | 112,517.3 | -0.469 | no |
| `r4_bucket8_margin16` | finer boundary (9) x argmax x margin16 | 720 | 233,594 | 764 | 112,013.3 | 112,777.3 | -0.000 | no |
| `r5_bucket8_margin32` | finer boundary x argmax x margin32 | 1,440 | 121,114 | 1,261 | 111,924.2 | 113,185.2 | -0.179 | no |
| `r6_prevclass_margin16` | + co-located previous class | 3,600 | 109,126 | 1,527 | 111,834.1 | 113,361.1 | -0.339 | no |
| `r7_prevclass_margin32` | + previous class, margin32 | 7,200 | 58,370 | 2,434 | 111,753.5 | 114,187.5 | -0.089 | no |
| `r8_second_choice` | + runner-up class | 9,000 | 83,015 | 2,289 | 111,807.1 | 114,096.1 | -0.370 | no |
| `shipped_actual_table` | the RCF1 table actually inside hv1 | 25 | 4,718,592 | 100 | 112,109.6 | 112,209.6 | **−3.355** | **YES** |

**Read the slope column.** One rung pays. It is the one already shipped.

Two secondary facts worth carrying:

- **Placement beats quantity.** The shipped HPAC network returns **3.810 B per counted byte**
  (13,515 B buys 51,484 B of tokens, dc1). The best correction table returns **0.146 B per counted
  byte** (2,434 B buys 356 B). At comparable counted spend the network is **26× more byte-efficient
  than the table.** The counted byte is not fungible — where you spend it dominates how many.
- **My fit is slightly PESSIMISTIC and it does not matter.** At the shipped context my fitted table
  lands 27.5 B above the shipped one, because I optimise the un-rounded objective while the
  receiver rounds corrected logits to 1/8 before the softmax. That is a ~30 B inefficiency against
  a 14,413 B requirement; crediting every rung +30 B changes no verdict.

## The oracle rows refuted my own design — reported as such

I added four free-table oracle rows intending them as an asymmetric kill instrument bounding *any*
post-hoc function of the model's output. **They do not do that, and my own n600 numbers show why.**

| oracle row | cells | samples/cell | free-table token B | vs shipped |
|---|---:|---:|---:|---:|
| `o1_oracle_model_output_coarse` | 62 | 1,902,658 | 205,561.0 | **+93,451.4** |
| `o2_oracle_model_output_fine` | 770 | 153,201 | 137,703.9 | **+25,594.3** |
| `o3_oracle_model_output_finest` | 1,994 | 59,160 | 115,557.4 | **+3,447.9** |
| `o4_oracle_output_plus_context` | 20,548 | 5,741 | 114,207.0 | **+2,097.5** |

A cell index derived from the model's output is a **replacement** for that output, not a
refinement: the oracle predicts from the cell and throws the base logits away. Even the finest
summary tested — argmax × 256 margin bins × runner-up × boundary × previous class, 20,548 occupied
cells, with a completely free table — costs **+2,097 B more than the shipped model**. So these rows
bound nothing about additive corrections; the realized `r*` rungs are the valid instrument.

What they DO establish is worth keeping: **the shipped model's 5-vector output is not summarizable
by those hand-designed features.** Replacing it with any function of them costs ≥ 2,097 B even for
free. That independently explains the realized ladder — keying a richer correction table on more
hand-designed features cannot be the route, because those features do not carry the information.

## The derivative, bracketed

Three measured points now sit either side of the shipped operating point. They come from two
vehicles and two mechanisms, so this is a **DERIVED bracket, not one continuous curve**:

| direction | mechanism | base | Δ model B | Δ token B | slope | reading |
|---|---|---|---:|---:|---:|---|
| shrink | frame_dim coordinate removal (prior `ddm_hm1`) | PR130 | −420 | +484 | **−1.15** | just past break-even |
| — | **shipped RCF1 correction table** | hv1 | +100 | −335.5 | **−3.355** | pays, already banked |
| grow | next correction rung | hv1 | +145 | −2.2 | **−0.016** | 210× collapse |

**The shipped model sits at the knee, bracketed from both sides by measurement.**

## Decode headroom, re-derived on the hv1 base with each machine named

| number | machine | archive | what it is |
|---|---|---|---|
| **516.836 s** | local M5 Max, macOS arm64, `resolved_device=cpu` | hv1 `80d9c8c6…` | wc1 `base_optimized_n600_r3` decode+render. Learned model 127.653 s (24.70%), coder 4.545 s (0.88%). |
| **1,907.241 s** | local M5 Max, `.venv` py3.13.12 | hv1 `80d9c8c6…` | `contest_auth_eval.json`, `score_axis: cpu_env_mismatch_advisory`. Pre-optimization inflate path. NOT contest hardware. |
| **831.535 s** | Linux x86_64, Modal CPU | **MC36** `f0ba4bb4…`, 186,269 B | the only real `[contest-CPU]` decode row (task #1054). ~2.165× headroom vs 1,800 s. |

**Correction to `ddm_dc1`:** its §"What I did NOT measure" #2 calls the 831.5 s anchor
*"UNVERIFIABLE from any artifact I can reach."* It is verifiable — primary receipt
`experiments/results/ddm_f26r_mc36_contest_cpu_20260814/modal_cpu_auth_eval_result.json`
(`inflate_elapsed_seconds: 831.5345255450001`), surfaced in
`.omx/research/ddm_pq1_submission_packet_prep_20260815.md:68-70`. dc1's *decision* to build its
budget axis on the wc1 M5 ladder is right; its claim that the artifact does not exist is wrong.
The caveat my charter already made survives: it is **MC36 bytes, not hv1**.

**hv1 has no contest-CPU decode row at all.** Honest reading:

- For the correction-table branch, decode cost is a table gather. Decode does **not** bind before
  bytes do, at any rung.
- For the neural branch, INFERRED only: if hv1's contest-CPU decode resembled MC36's 831.5 s and
  the learned-model share held at the M5-measured 24.70%, the model would cost ~205 s with ~968 s
  of slack, allowing roughly **5.7×** compute. That chains two unverified transfers (different
  archive, assumed share). Stated as INFERRED. Not the binding constraint either way.

## OPTIMAL FORM

REFERENCE form for a learned entropy model on a stream of this density is a full neural
autoregressive context model. **The vehicle already ships one** — integer HPAC, 7×7 Type-A causal
conv + dilated depthwise + full previous-frame `conv_past` + per-frame FiLM, C64/P64/delta2/D8,
13,515 counted bytes. That is the reference and it is already at it.

My rungs are **not a mechanism reduction of that model.** They are an at-its-own-optimum treatment
of a different, complementary counted-capacity axis — the correction table the archive already
carries. Deltas from reference are **SCOPE only**: n600 full field, no reduction, no prefix.
The `o*` rows are labelled FREE-TABLE ORACLES and are not achievable rows.

**Not bounded by this arm:** a retrained or widened HPAC. Also note `ddm_cl1`'s ordering gate — the
fixed-topology `rate_lambda` slope is preregistered to run BEFORE any width rung, and
`hpac_integer.py:186` carries `channels * weight_bound * activation_bound + 32768 >= 2**24`, which
may trip if `channels` rises past 64.

## Falsifier — pre-registered, and it fired

*If any correction-table rung richer than the shipped one clears `Δtokens/Δmodel < −1` at n600 with
counted bytes priced honestly, the correction-capacity axis is open and the ladder should continue.*

**FIRED — no rung clears it.** Eight richer rungs, all in [−0.469, −0.000]. And the stronger form
also holds: with the table priced at **zero**, the family's whole ceiling is 356.1 B = 2.47% of the
required cut.

## What I did NOT measure

1. **A retrained or widened HPAC.** Needs training; Modal is at $18.62/$20 and I may not dispatch,
   and I may not touch Metal. `ddm_cl1`'s ladder is the built instrument.
2. **A receiver parse-back** for the larger tables. Indexing logic is generic decoder code (free
   under rule 118) and only table values are counted, but I did not land it. Moot — no rung pays.
3. **hv1's contest-CPU decode time.** No such row exists.
4. **A joint fit of table AND network.** My tables are fitted post-hoc against frozen logits. A
   jointly-trained table could differ; the shipped one is jointly trained and beats my post-hoc fit
   by 27.5 B, which is direct evidence that joint training helps a little — and 27.5 B is 0.19% of
   the bar.

## Consequence for the campaign

`ddm_dc1` named the live cell: *"the model-vs-code exchange rate is the unmeasured, gap-sized
question."* This arm measures one of its two branches and closes it.

The remaining branch is **in-network reallocation**, and my result argues it is the right one: the
network converts counted bytes to token bytes 26× more efficiently than a table does, so if any
counted byte is still worth moving it must be moved INSIDE the network. That is exactly what
`ddm_cl1`'s fixed-topology `rate_lambda` rung varies. Nothing in this arm licenses spending a Metal
slot on it — it licenses *not* spending one on any table, sidecar, or hand-designed-feature
conditioning proposal, all of which are now closed at $0.

Set against the bar honestly: the correction family's entire free ceiling is 2.47% of the required
cut, and the one paying rung is already banked. **This axis does not reach sub-0.15.**

## NEXT_IF_RESUMED

| # | row | owner | fire condition | cost |
|---|---|---|---|---|
| 1 | **Retire the correction-table / hand-designed-conditioning family from the arm queue.** No arm should propose a bigger RCF1 table, a second residual table, or extra hand-designed context features on hv1. Point them at this memo's ladder. | MAIN | on read | $0 |
| 2 | `ddm_cl1`'s fixed-topology `rate_lambda` ladder is the ONLY live rung on this axis. It is built, preregistered, and blocked on Metal. My measurement says in-network reallocation is 26× the right place to spend, and the prior `ddm_hm1` shrink point says the model is only −1.15 past break-even, so the expected prize is small — fire it only when a Metal slot is otherwise idle. | `ddm_cl1` MAIN Metal executor | Metal slot free + existing CL1 guard passes | GPU |
| 3 | **The retained logits are a reusable asset.** `base_logits_int16_n600.i16` is the shipped decoder's own pre-correction output, byte-identity-gated. Any future model-capacity arm can price a probability-side change in seconds instead of re-running a 19-minute HPAC forward. | any rate arm | on need | $0 |
| 4 | The oracle rows show the model's output is not summarizable by hand-designed features. If anyone still wants a post-hoc probability-side lever, the only untested shape is a correction keyed on the FULL quantized logit vector rather than a summary of it. My rows do not bound that. Low prior. | rate owner | unowned | $0 |

**No fire-order is emitted. No candidate archive was produced. Nothing here is worth an exact row.**

## Receipts

- Workspace `/Volumes/APDataStore/pact/ddm_hm1_20260816/` with `RETENTION_MANIFEST.json`.
- `retained/base_logits_int16_n600.i16` — the shipped decoder's own pre-correction logits.
- `retained/ladder_n600/ladder.json`, `repriced_ladder.json`, every fitted table as `.f32`, and
  every priced candidate as `.rcf1` AND `.rcf1.br` (both retained, not only the `min()` winner).
- Instruments: `experiments/ddm_hm1_hpac_logit_replay.py`,
  `experiments/ddm_hm1_correction_capacity_ladder.py`, `experiments/ddm_hm1_reprice_tables.py`,
  `experiments/ddm_hm1_retention_manifest.py`,
  `experiments/tests/test_ddm_hm1_correction_capacity_ladder.py` (17 tests).

## Own-vehicle frontier

Unchanged: **S = 0.15959729295498598 @ 182,759 B `[contest-CUDA T4, n600]`**, archive sha
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`. This arm produced no score,
claimed no scorer slot, spent $0, and did not move the pointer.
