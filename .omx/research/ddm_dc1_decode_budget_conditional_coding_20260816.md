# ddm_dc1 — the coder axis is CLOSED on hv1, and the closure is now MEASURED on hv1

Date: 2026-08-16 · Owner: ddm_dc1 · Axis: `[macOS-CPU advisory / scorer-free byte measurement]`
`score_claim=false` · `promotable=false` · pointer UNMOVED at 0.15959729295498598 @ 182,759 B.
Workspace + payloads: `/Volumes/APDataStore/pact/ddm_dc1_20260816/` (61 artifacts,
`RETENTION_MANIFEST.json`, every re-coded candidate payload kept with sha256).

## Verdict

**Decode compute is a sound currency with no market.** The decode budget is not the binding
constraint — the wc1 ladder already bought 3.69× headroom — and the entropy coder is only
**4.55 s of a 516.8 s decode (0.88%)**. But every coder move I raced on the real hv1 payloads
returns **0 to +3,482 bytes WORSE**. The best measured coder-family move is **0 B** against a
**−14,413 B** target.

**The charter's crux premise is FALSIFIED at source.** The charter supposed the conditional-entropy
axis might be unmeasured because #996 used a *memoryless* bound. Two things are wrong with that:

1. #996's token row was **never** a memoryless bound — at source
   (`ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`) it reads
   `114,852 (model cross-entropy)`. Only semantic/pose/hpac were order-0.
2. More decisively: **the shipped vehicle already IS a learned autoregressive context coder.**
   `residual_archive.decode_production_tokens` runs a learned integer HPAC network per symbol —
   spatial causal conv over already-decoded tokens, previous-frame context via `conv_past`, per-frame
   FiLM, plus a 25×5 boundary-residual correction table — and feeds its softmax to an rc64 range
   coder. Conditional coding is not an unexplored axis here; it is 61.4% of the archive.

**The two numbers that decide it.**

1. **The range coder is 7.80 bytes from optimal.** I replayed the shipped native decode and
   accumulated `−log2 p[actual symbol]` from the very buffer rc64 consumes. HPAC's own cross-entropy
   over the full n600 field is **112,109.578 B**; rc64 consumed **112,117.375 B**. Overhead
   **+7.80 B = +0.00696%**, and the 8-frame prefix showed +7.74 B — so it is a **constant flush
   cost, not a rate**. That 7.80 B is the CONSERVATIVE reading: `bit_position` includes the range
   decoder's end-of-stream read-ahead, which is why it exceeds the 112,110 B stream itself. Measured
   against the shipped stream length the gap is only **112,110 − 112,109.578 = 0.42 B**. Take the
   larger: the maximum possible prize from any coder swap on 61.4% of the archive is **≤ 7.8 bytes**.
   The replay is byte-identical to the shipped decode: `decoded_token_sha256`,
   `corrected_cdf_input_sha256` and `corrected_quantized_logit_sha256` all match the wc1 receipt.
2. **The learned prior is worth 3.8× its counted bytes.** A table-free adaptive context model — zero
   counted table bytes, so it could DELETE the 13,515 B HPAC section — needs **177,109 B** for the
   token field. The shipped token subsystem costs **125,625 B** (112,110 stream + 13,515 model).
   Dropping the learned prior would **cost +51,484 B**.

## What I measured, and on which base

Everything below is MEASURED by me on the **hv1 frontier archive**
(sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`, 182,759 B) and its real
decoded token field (sha `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`,
600×384×512 = 117,964,800 symbols). The #996 closure table exists on the **PR130** base only
(191,052 B) and had never been pointed at hv1. This is the first hv1 measurement.

### Section census — independently re-derived from the RX1 header

| section | shipped B | raw B | share |
|---|---:|---:|---:|
| token stream (rc64) | 112,110 | 117,964,800 symbols | 61.34% |
| semantic | 34,763 | 36,040 | 19.02% |
| carrier | 22,161 | 22,219 | 12.13% |
| HPAC model | 13,515 | 17,952 | 7.40% |
| residual table (RCF1) | 96 | — | 0.05% |
| RX1 header | 14 | — | |
| ZIP framing | 100 | — | |
| **total** | **182,759** | | |

Agrees with ra2's independent parse section-for-section. The archive is one stored ZIP member `p`.

### Samples-per-symbol — the governing variable, per section

ra2's relay named the mechanism; here is the table across the whole archive.

| section | symbols | alphabet | **samples/symbol** |
|---|---:|---:|---:|
| token stream | 117,964,800 | 5 | **23,592,960** |
| semantic | 36,040 | 256 | 141 |
| carrier | 22,219 | 256 | 87 |
| HPAC model | 17,952 | 256 | 70 |
| carrier coefficients (ra2's object) | 7,200 | 4,096 | **1.76** |

A **13.4-million-fold spread**, and it predicts every measured outcome in the record. The token
stream is the only section dense enough to amortize a learned conditional model — which is exactly
where the vehicle put one. Every adaptive-model loss on record (ra2's 415–840 B on the carrier;
rc2's PPMd +441…+4,618 B; my own +3,482 B below) sits on the sparse side of that table.

### The coder race on hv1 — real payloads, all retained

| section | shipped | brotli-q11 lgwin24 | LZMA2 xtreme | LZMA1 raw xtreme | best Δ |
|---|---:|---:|---:|---:|---:|
| token stream | 112,110 | 112,115 | 112,117 | 113,734 | **+5** |
| semantic | 34,763 | **34,763** | 35,327 | 35,326 | **+0** |
| carrier | 22,161 | **22,161** | 22,223 | 22,332 | **+0** |
| HPAC | 13,515 | 13,555 | 13,569 | 13,568 | **+40** |

The semantic and carrier recodes are **byte-identical** to the shipped sections
(sha `4099eab6…` and `fd14aabc…`), so the shipped encoder already *is* brotli-q11 lgwin24 there.
Racing them is a definitional no-op. The HPAC section is 40 B *smaller* than my best generic recode.

### Order-0 bound on hv1 — three of four sections are already below it

| section | shipped | order-0 byte entropy | shipped − order-0 |
|---|---:|---:|---:|
| HPAC | 13,515 | 14,961 | **−1,446** |
| semantic | 34,763 | 35,853 | **−1,090** |
| carrier | 22,161 | 22,143 | +18 (+0.08%) |

Same shape as #996 found on PR130: the model sections beat memoryless because brotli buys **LZ match
structure**, not symbol rank (sv2's mechanism). The carrier is 0.08% off its own floor.

### The order-1 mirage — and why it is not a prize

The order-1 conditional byte entropy across the three model sections is 57,298 B against 70,439 B
shipped — an apparent **−13,141 B**, almost exactly the size of the target. It is not real. That
oracle assumes a 65,536-cell context table is free. Measuring what an adaptive coder actually
achieves (exact sequential Krichevsky-Trofimov code length, zero transmitted table):

| section | shipped | order-1 ORACLE | order-1 ADAPTIVE | learning cost |
|---|---:|---:|---:|---:|
| HPAC | 13,515 | 10,487 | 14,550 | **+4,062** |
| semantic | 34,763 | 29,832 | 36,796 | **+6,964** |
| carrier | 22,161 | 16,978 | 22,575 | **+5,597** |
| **total** | **70,439** | 57,298 | **73,921** | **+16,623** |

The **+13,141 B apparent prize becomes a −3,482 B realizable loss.** At 70–141 samples per symbol the
learning cost swamps the conditional gain. This is the density trap, measured.

### The table-free context ladder on the token field — where it turns around

Table-free means zero counted table bytes, so a winner here could delete the 13,515 B HPAC section.
ORACLE = empirical conditional entropy (a strict lower bound for *any* coder using that context,
free table). KT = exact sequential code length an actual adaptive coder achieves.

| context | contexts used | samples/context | ORACLE B | KT B | KT − shipped stream |
|---|---:|---:|---:|---:|---:|
| 4 spatial + 1 prev | 705 | 167,326 | 208,179 | 209,165 | +97,055 |
| 4 spatial + 5 prev | 5,541 | 21,289 | 183,625 | 188,421 | +76,311 |
| 6 spatial + 5 prev | 11,398 | 10,350 | 171,274 | 179,896 | +67,786 |
| **8 spatial + 9 prev** | 40,839 | 2,889 | 152,187 | **177,109** | **+64,999** |
| 10 spatial + 11 prev | 72,235 | 1,633 | 144,167 | 184,337 | +72,227 |

Two facts do the work:

- **The KT curve turns around at 40,839 contexts (2,889 samples/context).** Adding conditioning
  after that point *costs* bytes — the learning cost of the new contexts exceeds the entropy they
  save. Conditioning multiplies the effective alphabet, so richer context makes density worse. This
  is ra2's mechanism firing on the densest object we own.
- **Even the ORACLE bottoms at 144,167 B — still +32,057 B above the shipped 112,110 B.** The oracle
  is a valid lower bound for *any* coder conditioned on those 21 taps, including a logistic-mixing
  CM/PAQ-class coder. So no reweighting of that information can reach the shipped stream. HPAC wins
  because it conditions on strictly more (7×7 Type-A + dilated depthwise + full previous frame +
  patch FiLM), and that extra receptive field is only affordable as *learned weights*.

For reference, order-0 on the same field is 23,821,786 B — 212× the shipped stream.

## The exchange-rate curve

Decode cost measured on the wc1 optimized native path (`base_optimized_n600_r3`, M5-CPU advisory):
token decode 140.54 s, split **`native_sparse_hidden_and_logits` 108.13 s + `native_incremental_conv_update`
16.86 s + `native_frame_context_int16` 2.32 s + `native_conv_state_initialization` 0.34 s = 127.65 s
of LEARNED MODEL**, versus **`native_probability_and_rc64` 4.55 s of CODER**. Full decode 516.8 s.

| move | decode Δs | counted bytes Δ | B per decode-second | status |
|---|---:|---:|---:|---|
| **shipped learned HPAC prior (already banked)** | **+127.65** | **−51,484** | **+403** | BANKED |
| drop the prior → best table-free KT coder | −127.65 | **+51,484** | — | LOSS |
| **a PERFECT coder replacing rc64 (measured ceiling)** | any | **−7.8** | ~0 | **CEILING** |
| rc64 → brotli-q11 on the token stream | ~0 | +5 | 0 | LOSS |
| rc64 → ANS (measured on the F26 ancestor, lp135) | ~0 | +6 … +9 | 0 | LOSS (INHERITED) |
| recode semantic / carrier | 0 | +0 (byte-identical) | 0 | NO-OP |
| recode HPAC section | 0 | +40 | 0 | LOSS |
| adaptive order-1 byte coder, model sections | small + | +3,482 | negative | LOSS |
| PPMd orders 2–16 (rc2, PR130 base) | + | +441 … +4,618 | negative | LOSS (INHERITED) |
| LDPC/BP syndrome (rc2, PR130 base) | +22 … +75 | +540,909 | negative | LOSS (INHERITED) |
| **TARGET LINE** | any | **−14,413** | — | **unreached** |

**The curve has exactly one point with positive slope and we are already standing on it.** The
budget half of the charter's thesis is sound — 4.55 s of coder inside a 516.8 s decode inside a
1,800 s budget means we could afford a coder **100× more expensive** — but there is nothing to buy.
The binding constraint on the token stream is **model class and sample density**, not compute.

## Falsifier — pre-registered, and it fired

The charter's falsifier: *if the conditional model's cross-entropy is within a few percent of the
shipped bytes on every section, the coder axis really is closed at the conditional level too.*

**FIRED, in the sharper form the evidence supports.** I could not compute HPAC's own cross-entropy
directly (see owed measurement below), so I tested the stronger claim instead: whether *any*
table-free conditional model can beat the shipped bytes. On the token stream the best is +64,999 B
worse and even its oracle is +32,057 B worse; on the three model sections the best adaptive
conditional coder is +3,482 B worse and generic recodes are +0/+40 B. Every section is at or below
its own bound. **The coder axis is closed on hv1, at the conditional level, measured on hv1.**

## What I did NOT measure — stated plainly

1. ~~HPAC's own cross-entropy on hv1.~~ **NOW MEASURED — this item is CLOSED** (see the verdict).
   The concern was real and worth closing: on PR130 the *range* coder sat +1.85% above its model and
   ANS harvested 2,120 B, so if hv1's rc64 were similarly loose ~2,075 B would have been available.
   It is not: hv1's rc64 is +0.00696%, a constant 7.80 B flush. The PR130 looseness did not transfer.
2. **The charter's "831.5 s contest-CPU decode" anchor is UNVERIFIABLE from any artifact I can
   reach** — no such figure exists in `.omx/research/` or the canonical task ledger. What I can
   verify: the retained hv1 CPU receipt is stamped `score_axis: cpu_env_mismatch_advisory` with
   `inflate_elapsed_seconds: 1907.24`, matching wc1's pre-optimization baseline of 1,905 s. My
   budget axis is built on the wc1 M5-CPU advisory ladder, not on 831.5 s.
3. **A logistic-mixing CM coder was not built.** It is dominated by the oracle argument above, not
   by a race — I am relying on the bound, and I say so.

## Consequence for the campaign

The rate axis does not close — but the **coding** half of it does. Every remaining rate byte must
come from a **better representation** (ddm_rc4 owns this: which parameters ship) or a **better
model** (the learned prior). Our own record already names the live cell, verbatim from
`RATE_AXIS_LOSSLESS_RACE.md` §5: *"The model-vs-code exchange rate is the unmeasured, gap-sized
question."* My measurement prices its current operating point for the first time: the prior costs
13,515 counted bytes and 127.65 decode-seconds, and returns 51,484 bytes — **+3.81 bytes returned
per byte spent on the model.** Whether that marginal rate survives a larger prior is genuinely open;
I have no measured floor below the shipped 112,110 B, so I cannot bound the upside either way.
The one measured attempt on a different object (hp1's learned AR prior on tq1c IX2TOK01) was
byte-negative at +114,870 B, with `verdict_scope: FAMILY` for ≤10K static-context priors — which
does not bind the HPAC-class prior on this object.

## NEXT_IF_RESUMED

| # | row | owner | fire condition | cost |
|---|---|---|---|---|
| 1 | ~~Measure HPAC's own cross-entropy on hv1.~~ **FIRED AND CLOSED this unit** — 215.8 s local, byte-identical replay, rc64 overhead +7.80 B (+0.00696%). Receipt `retained/hpac_cross_entropy_n600.json`. | ddm_dc1 | done | $0 |
| 2 | **d(tokens)/d(model): does a larger HPAC prior keep returning >1 B per counted byte?** The only live cell in the rate axis. Needs training, so it is blocked while Modal is at $18.62/$20. | model owner | Modal budget reopens, or a local MLX training path lands | GPU |
| 3 | Carrier is +18 B (+0.08%) above its own order-0 floor and ra2's arithmetic coder already found +263 B raw / ~230 B realized there. That is the only section with any measured slack, and it is 1.6% of the gap. | ddm_ra2 | already owned by ra2 | $0 |
| 4 | Retire the coder family from the arm queue: no arm should propose a coder race on any hv1 section. Point them at this memo's race table. | MAIN | on read | $0 |

**No fire-order is emitted.** No candidate archive was produced; nothing here is worth an exact row.

## System-intelligence wire-in

The structural bug here is not the closure — it is that the closure was measured once on PR130 and
then *cited* on four downstream bases without re-measurement. `tools/audit_archive_coder_axis.py`
makes re-pointing it a one-command operation on any RX1 archive: section census, samples-per-symbol,
order-0 bound, order-1 oracle **and** order-1 adaptive (so the density trap is impossible to
misread), plus a real coder race with every candidate payload retained. It reproduces every number
in this memo independently and prints a `coder_axis_closed` / `coder_axis_open` verdict with the
best available saving in bytes. **No arm should ever again inherit this closure — run the tool.**

## Receipts

- `/Volumes/APDataStore/pact/ddm_dc1_20260816/RETENTION_MANIFEST.json` — 61 artifacts, 524,710,243 B,
  every payload with sha256 (raw sections, shipped sections, all 12 re-coded candidates, context
  count tables).
- `retained/token_conditional_entropy.json` · `retained/deep_context_ladder.json` ·
  `retained/sections/section_census.json` · `retained/sections/byte_density_trap.json`.
- Instruments: frontier archive `80d9c8c6…` (182,759 B); decoded tokens `9ba2e52b…` (117,964,800 B),
  binding-verified against `tokens_cpu_stage_complete.json` whose `archive_sha256` is the frontier.
- Decode timings from `ddm_wc1_advisory_decode_wallclock_20260815/runs/base_optimized_n600_r3/result.json`.
