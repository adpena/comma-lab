# THE EXCHANGE CURVE ON LB1'S OWN BODY — it has the wrong SIGN, so there is no rate to quote

`axis: [macOS-CPU scorer-free EXACT byte measurement, lb1's OWN RC64 + logistic-mixer coder]`
`score_claim: false` · `promotable: false`
`verdict_scope: MEASURES the bytes-vs-accuracy exchange on the lb1 frontier body under three`
`perturbation policies, real coder, 7 real-encode rungs. Opens nothing, closes nothing, escalates`
`nothing. It does NOT bound policies that rank by lb1's own per-position probabilities (§6, OWED).`
Date: 2026-08-31 · Owner: ddm_lbx1 · Consumers: THE CHASM · THE PINCER · [[m144]] · [[m124]] · [[m166]]

## STORES CONSULTED

`ddm_gestalt_the_chasm_not_the_cross_20260831.md` · `ddm_gestalt_generate_vs_serialize_pincer_20260831.md`
· `ddm_rd2_hg1_rate_distortion_curve/retained/rd2_phaseA_byte_curve.json` (all READ AT SOURCE, [[m44]])
· `ddm_bz2d_distortion_verdict_20260830.md` + `[[token-error-amplifies-to-argmax-error-no-attenuation]]`
(read at source — and it corrected a constant I was about to use, §2) · `ddm_lb1_banked_lossless_joint_collect`
retained store · `experiments/ddm_jg2_tail_reencode.py` argparse (never-invent-flags).

## 1. METHOD — ADAPTED, and the reason is that the two bodies have opposite structure

rd2's method: born-small emits an approximate field; an explicit **correction residual** is coded on
top; sweep the fraction of corrections applied; code each prefix with real coders; read bytes vs
corrections. It builds the curve by **adding** accuracy to a lossy body.

That does not transfer to lb1, and the reason is structural, not incidental. **lb1 has no correction
residual to withhold.** It is a *lossless* context-mixing coder over the exact token field
(`tokens_changed=0`, verified §2). There is no dial marked "apply fewer corrections." Its payload is:

| section | bytes | share of 179,983 B payload |
|---|---:|---:|
| `token_stream` | 113,492 | 63.1% |
| `compressed_models` (semantic 36,130 + carrier 22,316 + hpac 17,952, jointly coded) | 66,395 | 36.9% |
| `residual_payload` | 100 | 0.1% |

So the adaptation: **lb1's curve must be built by degrading the TARGET FIELD and re-encoding.** Pick a
field `Y'` at Hamming distance k from lb1's exact field `Y`, run it through **lb1's own encoder**, read
the real coded size. That is the same question rd2 asked — what does accuracy cost in bytes on this
body — asked in the only direction this body admits.

This is a **curve, not a family attempt**. It proposes no mechanism for choosing edits and adopts no
representation. It measures the exchange rate that any future trade on this body must clear. The
adjacent measured-closed refusals stay closed and own their ground: **semantic quantization** and
**section coding** own the 66,395 B model side (not touched here — §6); **token drop**, **residue
purchase**, and **tolerance** own specific token-side mechanisms. I ran no mechanism from any of them.

`--tokens` accepts any 117,964,800 B field and `--edits` splices arbitrary frames, so the real encoder
takes a perturbed field directly. `--frames 32` gives a real, bit-faithful partial encode in 54 s.

## 2. THE CONTROL LINE — 7 controls, all green, and one caught a bad constant

| # | control | result |
|---|---|---|
| 1 | retained archive sha256 vs frontier pointer | `5b856e66…8ad28c9` **exact** |
| 2 | S recomputed from components vs reported | `0.14803010583079396`, delta **0.0** |
| 3 | token-field class areas vs canonical n600 | Road 23.23 / Lane 0.586 / Undriv 49.52 / Mov 1.238 / MyCar 25.43 — **match** |
| 4 | decode output vs encode target field | both `cc10a7b0…92636efb` — **byte-identical, lossless confirmed** |
| 5 | retained per-frame ledger vs shipped stream | 113,491.28 B vs 113,492 B |
| 6 | my `--frames 32` control encode vs retained ledger | `48142.38951434754`, max per-frame diff **0.0 bits** |
| 7 | 32-frame window representativeness | holds 5.302% of stream vs 5.333% of frames — ratio **0.9942** |

**Control 6 is the one that matters**: my harness reproduces the shipped encoder's per-frame bit ledger
exactly, so every byte number below is lb1's real coder, not a model of it.

**A constant failed re-derivation before I used it.** I was about to convert token errors to argmax
flips with the **1.157×** ratio from [[token-error-amplifies-to-argmax-error-no-attenuation]]. Read at source, that ratio was **corrected the same day
it was published**: it is not a law (it moves 1.7× between two points). The real relation is affine —
`argmax ≈ 17,241 + 1.1435 × tokens` — so the correct **marginal** is the slope **1.1435**, and the
17,241 intercept is a render-manufactured floor already inside lb1's baseline. All arithmetic below uses
1.1435. ([[m143]], cross-regime constant transfer.)

## 3. THE DEMAND, PRICED AS AN EXCHANGE RATE

At fixed distortion (`100·d_seg + √(10·d_pose) = 0.028120`), the sub-0.12 byte cap is **137,986.84 B**
and the demand is **42,096.16 B**. Converting:

```
dS/dB                                        = 6.6586e-7 per archive byte
break-even per ARGMAX flip                   = 10.185 bits   (1.2731 B)
break-even per TOKEN error  (x1.1435 slope)  = 11.646 bits
budget: at most 28,916 corrupted tokens      = 0.0245% of the field
those tokens must carry 37.1% of the ENTIRE real 907,930-bit stream
required concentration                       = 1,513x the field average of 0.007697 bits/position
```

Structurally, lb1's field is **98.75% temporally static** — only 1,470,519 positions (1.2466%) change
frame to frame, and the whole stream is 0.6174 bits per changing position. So the demand restated:
**1.97% of the changing population must hold 37.1% of the bits — an 18.9× concentration inside the only
population that costs anything.**

## 4. THE CURVE — real lb1 coder, 7 rungs, three policies

Baseline frames 0–31 = **48,142.39 bits**. Negative "bits saved" means the stream **grew**.

| policy | k tokens | real code_bits | bits saved | bits/token | marginal | stream growth |
|---|---:|---:|---:|---:|---:|---:|
| static-oracle snap | 500 | 55,617.9 | **−7,475.5** | **−14.951** | −14.951 | 1.16× |
| static-oracle snap | 2,000 | 73,187.9 | **−25,045.5** | **−12.523** | −11.713 | 1.52× |
| static-oracle snap | 8,000 | 126,006.7 | **−77,864.3** | **−9.733** | −8.803 | 2.62× |
| static-oracle snap | 32,000 | 236,309.0 | **−188,166.6** | **−5.880** | −4.596 | 4.91× |
| snap to previous frame | 2,000 | 63,323.4 | **−15,181.0** | **−7.591** | −7.591 | 1.32× |
| snap to previous frame | 32,000 | 138,639.2 | **−90,496.8** | **−2.828** | −2.511 | 2.88× |
| random → random class | 2,000 | 109,820.7 | **−61,678.3** | **−30.839** | −30.839 | 2.28× |

**Break-even requires +11.646 bits/token. All 7 rungs are negative.** The sign is wrong across the
entire measured range, under all three policies, and no crossing was observed.

The three policies bracket the question cleanly at k=2,000: **random −30.839 · static-oracle −12.523 ·
temporal-snap −7.591**. The ordering is the expected one — my ranking is **2.46× better than random**, so
it is genuinely informative and not anti-correlated — and the whole **4.06× policy spread sits on the
wrong side of zero.**

Two independent cross-checks that the sign is real, not a harness artifact:

- **Real LZMA2-extreme on the same field, same edits**: predicted +3,172 B saved at k=2,000, measured
  **−1,238 B** (real/predicted = −0.39). A second real coder, same inversion.
- **The static model predicted saving 49,709 bits at k=8,000 — more than the entire 48,142-bit real
  stream.** Impossible on its face; it is why I did not report a static-model byte number as a result.

This is **[[m166]]** firing at full strength: *−log2 p is direction-dependent; price token levers by
REAL re-encode.* A static-model curve on this body would have reported a knee at k*≈16–19k saving
~26,600 B. It does not exist. The real coder loses bytes there.

**Mechanism.** lb1 ships **66,395 B of models fitted to this exact field**, and its coder already spends
0.0077 bits/position. The predictability has already been absorbed into the model. An edit is therefore
a *departure from the fitted model*, and it is charged twice — once at the edited position, and again
downstream as the adaptive mixer's context degrades. Making the field "more predictable" in the abstract
makes it **less** predictable to the predictor that ships with it.

## 5. THE DIRECT ANSWER

> **At lb1's operating point, the first 42,097 B costs unbounded distortion: it cannot be bought at any
> distortion by corrupting tokens, because the byte side never moves the right way.** The measured
> exchange rate is **−2.8 to −30.8 bits per corrupted token** (a cost) across three policies, against a
> break-even that requires **+11.6** (a saving). To shed 42,096 B the token stream must shrink 37.1%;
> measured, it **grows** — 1.16× at k=500, 4.91× at k=32,000.

> **Is the curve convex like born-small's, or is there a usable knee? Neither.** Born-small's curve
> *descends* at a rising price — 0.220 → 3.639 bits/correction, a 16.5× convex wall, but it descends, so
> a knee exists and rd2 could quote one. **lb1's does not descend at all.** Bytes and distortion move in
> the *same* direction: you pay bytes *and* lose accuracy. There is no knee because there is no descent.
>
> The marginal cost does decline monotonically toward an eventual sign change — static-oracle −14.951 →
> −4.596, temporal-snap −7.591 → −2.511 — so a crossing presumably exists under heavy corruption. But
> k=32,000 in a 32-frame window is already **20.7× the entire n600 distortion budget** (603,497 tokens
> scaled, against a budget of 28,916) and both policies are still negative there. That crossing is not a
> knee; it is the far side of the field being destroyed. At the budget itself the relevant rung is
> k≈1,542 in-window, where the cost is steepest.

**This verdict is scorer-independent.** It fails on the byte axis alone, so it does not depend on the
1.1435 slope, on the d_seg instrument, or on the #1142 GT fork. Those only affect *how badly* it fails.

**What this does to the standing picture.** `ddm_gestalt_the_chasm_not_the_cross_20260831.md` said lb1 is the one object inside the accuracy
half and misses the byte half by 1.305×, and framed the remaining problem as "shed 42,097 B at fixed
distortion, **or trade accuracy for bytes**." This measures that second clause and closes it *as an
exchange*: on this body there is no accuracy-for-bytes trade to make. The 1.305× must be paid on the
representation, not bought with lb1's own accuracy slack — which is what [[m124]]'s "42,382 B at fixed
distortion OR 150 B at zero" already implied and this now measures directly.

## 6. THE DENOMINATOR — and the one thing this does NOT bound

**Rungs planned 7 · rungs measured 7 · controls run 7, all green.** Plus 3 supporting proxy sweeps
(static ctx-125 oracle curve at 9 k-values; a 5-model strength sweep; a 5-rung real-LZMA cross-check) —
all reported as proxies, none used for a byte verdict. Nothing planned was dropped.

Policies swept: 3 (static-oracle snap, temporal snap, random). Frames: 32 of 600, window
representativeness 0.9942 (control 7), n600 scaling by stream share ×18.86 vs by position count ×18.75 —
within 0.6%.

**The honest limit, and it is the real one.** My ranking used a *static ctx-3125 model's* argmax, not
**lb1's own model's** argmax. So this is a strong-but-mismatched oracle, and a true oracle ranked by
lb1's own per-position probabilities would do better than every row above. Three things bound how much
that can matter, and none of them rescues the sign:

1. The **temporal-snap** policy — the one most aligned with what any video model predicts, on a field
   that is 98.75% temporally static, and therefore already close to lb1's own argmax at most positions —
   is **1.65× cheaper** than my static oracle (−7.591 vs −12.523 bits/token) **and still negative.**
   Better policy improved the cost and did not flip the sign.
2. A true oracle's per-position saving is bounded above by that position's own code cost, so total
   savings ≤ 907,930 bits; the demand is 37.1% of that, requiring the 1,513× concentration of §3.
3. The **model-strength sweep** (five static models, 235,319 → 206,008 B) showed achievable |ΔS| moving
   **0.001959 → 0.001729 (0.88×)** as the model improves. The trend runs *against* the trade, and lb1's
   real coder is 1.82× stronger again than the best model in that sweep.

**How much room is left, stated honestly.** Going from the worst policy to the best bought **23.25
bits/token** (−30.839 → −7.591). Reaching break-even needs a further **19.24** (−7.591 → +11.646) —
**83% of the entire spread again, and in a direction no policy tested moved at all.** I cannot exclude
that a true oracle flips the sign, and I am not claiming it cannot. I am reporting that three policies
spanning 4.06× did not, that the two structural bounds above run against it, and that the OWED run below is
the measurement that settles it.

**OWED (not scorer-gated — local instrumentation):** rank by lb1's own `coding_row` probabilities and
re-run this ladder. `runtime/hpac_inference.py::optimize_sparse_evaluator` + `sparse.selected_logits`
expose the per-position distribution, so an inference-only pass over 32 frames is ~1 encode of cost. If
that also fails to flip the sign, the token-side exchange on this body is closed on its own evidence.

**Not measured here, and owned elsewhere:** the **66,395 B model side**. Cutting model bytes makes the
token stream more expensive — a real second axis, but it is the ground of the **semantic-quantization**
and **section-coding** refusals, and re-opening it by renaming is exactly what I was told not to do.
This memo is silent on it rather than dressed up as new.

## 7. FIRE ORDERS

**None.** Nothing in this result is scorer-gated. The verdict fails on the exact byte axis, measured on
lb1's own coder with a bit-identical control, so no `d_seg` on a perturbed body would change it — and
asking MAIN to spend scorer time confirming the distortion cost of a trade whose byte side never goes
the right way would be spending it on a foregone conclusion. If the OWED true-oracle run (§6) flips the
sign, *that* result would earn a fire order; this one does not.

## 8. RECEIPTS (retained, P0)

Store `/Volumes/APDataStore/pact/ddm_lbx1_lb1_exchange_curve/`:
`retained/LBX1_EXCHANGE_CURVE.json` (curve + controls + all sha256) · `work/edits_manifest.json`
(every perturbation, sha256 + exact k) · `work/edits_{k500,k2000,k8000,k32000,prev2000,prev32000,rand2000}.npz`
· `work/bits_per_frame_lbx1_*.npy` (real per-frame ledgers) · `work/*.log` (encoder receipts) ·
`work/baseline_bits_per_frame.npy`. Every encode carries its own `code_bits`; no payload was measured
and discarded.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B,`
`d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9;`
`this memo did not move the pointer and made no attempt to.`
