# ddm_sa1 — REPRESENTATION attack on the frozen semantic+carrier block

**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement + renderer-field probe]`
`score_claim=false` · `promotable=false` · no Modal, no n600 scorer run, no Metal, no launches.

**Base (verified at receipt, not from memory):** rr4 `archive.zip` **181,161 B**, sha
`35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`, S **0.15853325034789678**
(seg 0.00029611, pose 6.88e-06) `[contest-CUDA T4 n600]`.
Retained: `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip`.

**Headline:** 18 byte-closed candidates, every one parse-back-verified through the **shipping**
receiver. Best credit **−2,889 B = ΔS_rate −1.92367e-3** (22.5% of the 0.00853325 gap). Distortion
is unmeasured and is exactly what the sealed fire-orders buy. **The frontier did not move.**

**The finding that matters most is a negative one about my own method:** a weight-MSE screen
inverted the true ranking by four orders of magnitude and would have shipped a catastrophic
candidate. See §5.

---

## STORES CONSULTED

- `.omx/research/ddm_gs1_gestalt_convocation_20260817.md` §9 (charter origin)
- `ddm_mz2_frozen_section_representation_attack_20260815.md` + commit `5c073e915` (banked candidates)
- `ddm_wc2_hpac_mps_port_20260814.md` (13,619/34,763/22,161 decomposition)
- `ddm_ra1_carrier_rank_refit_preproof_20260816.md`, `ddm_ra2c_alpha0_verdict_*`,
  `ddm_ra2c_rank4_verdict_*`, `ddm_ra2crr_priced_pose_null_*`, `ddm_ra2_charter_stale_family_closed_*`,
  `ddm_ra3_subspace_trust_region_refit_20260816.md` (carrier closure — read in full)
- `ddm_pv1_provenance_lineage_citation_audit_20260817.md` (36,051 vs 36,040 discrepancy)
- `.omx/state/main_hot_state.md` (live bar, stale-constant genus)
- Live source: rr4 `candidate_runtime/` (`residual_archive.py`, `f26_inflate.py`,
  `cpr1/ddm_mp2_semantic_receiver.py`, `cpr1/inflate.py`),
  `pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/renderer_weight_codec.py`

---

## 1. Deliverable 1 — the section accounting, reconciled THREE ways

The charter asked for the "1,288 B discrepancy" to be explained or declared a finding. It is
explained: **it is a packed-vs-decoded conflation across two distinct steps**, and there is a third
object nobody had named alongside the other two.

| object | semantic | carrier | what it is |
|---|---:|---:|---|
| **packed stream** (RX1 header field) | **34,763** | **22,161** | brotli-q11 stream — *what the archive charges* |
| **wire body** (pre-brotli) | **36,040** | — | F12-reordered WANS body |
| **canonical blob** (post-F12-decode) | **36,051** | **22,242** | `parts.semantic_blob` |

- `34,763 + 22,161 + 14 B RX1 header = **56,938 B**` — exactly gs1's "56,938 B frozen block".
- Semantic decoded−packed = 1,288 B = **F12 reorder (11 B) + brotli (1,277 B)**. Carrier's
  analogue is only **81 B**.
- **This settles the ra2-vs-pv1 11 B contradiction** the recall flagged as unreconciled: ra2's
  "36,040 raw body B" is the *wire body*; pv1's "36,051 B" is the *canonical blob*. Both correct,
  different objects. Neither is wrong; the pair was never wrong to begin with.

Full container: zip 100 + RX1 header 14 + hpac 13,515 + semantic 34,763 + carrier 22,161 +
tail 110,608 = **181,061 B member / 181,161 B archive**. Note **hpac packed = 13,515 B**, not
wc2's 13,619 B (−104 B); the semantic and carrier halves match wc2 exactly.

**Carrier compresses by 0.36%; semantic by 3.54%.** The carrier is already entropy-coded, so
decoded-byte removals there convert ~1:1 to charged bytes; semantic converts at 0.965:1.

## 2. Instrument controls (both passed before any candidate was built)

1. **Verbatim reassembly** of parsed parts through `pack_rx1_model` + `deterministic_zip`
   reproduces rr4's archive **byte-identically** (sha `35ac2b9b…`). The pack path is the real one.
2. **brotli q11 (homebrew 1.2.0)** on the wire body reproduces the shipped 34,763 B stream
   **exactly**. My encoder is bit-identical to the shipping one.
3. **`encode_wans1 ∘ decode_wans1`** reproduces the canonical blob, the 36,040 B wire body *and*
   the 34,763 B stream byte-for-byte, on both `global` and `per_tensor` strategies.

Because `pack_rx1_model` consumes already-compressed streams, a semantic-only edit has
`archive_delta == packed-semantic delta` exactly. Every byte number below is a real
`archive.zip` on disk, not an entropy estimate.

## 3. Dead families — verdicts inherited, verified at source, NOT re-run

- **R1 (carrier low-rank/refit) is CLOSED at FAMILY scope on the DISTORTION axis.** Two
  independent bounds (ra1's least-squares optimum; ra2crr's sphere-wide minimum over all
  directions, 292/292 within 1% of optimum) plus ra3's realised refit missing by 35.5×. The
  *rate* side passes outright (rank-4 returns 14,709 B). Only escape named by all three: a
  carrier **retrained with pose in the loop**. I did not rebuild it. Charter rung R1 is retired.
- **R3 (joint rank×precision)** is moot: its rank leg is inside R1's closure.
- **Lossless recoding of the semantic tensors** stays dead, and this arm gives it a *price*
  rather than a refusal — see §4 `V0_all_q4_control`.
- **mz2's "38/38 receiver-required" is e480b-scoped** (its own bar was 15,153 B). I did not cite
  it as an rr4 result. The schema facts are receiver properties and do hold, but the memo asserts
  them only as an e480b instance.

## 4. Deliverable 2 — R0 re-derived, and the blocker that had already dissolved

**The two banked mz2 archives cannot be scored.** They were built on **e480b (183,502 B)**; their
carrier/token/residual members are e480b's. Their −823/−2,051 B figures are stale arithmetic
against a base two pointer-moves old. I refused to score them as specified and re-derived the
transforms onto rr4 instead. **The byte deltas transfer exactly** (rr4's semantic section is
byte-identical to e480b's at 34,763 B); only the S-conversion moved.

**mz2's stated blocker was already satisfied and nobody had noticed.** mz2 queued its candidates
behind "a shipping SD1M receiver adapter parse-backs this exact archive". The rr4
`candidate_runtime` **already ships** `cpr1/ddm_mp2_semantic_receiver.py`, whose own docstring
names "the SM3R row-prune packet used by the six MZ2 candidates", and `f26_inflate.py:428`
accepts `WANS1 | SD1M | SM3R`. Every candidate below was decoded through that module with
**deviation 0.0** against the builder's expected state. **No receiver change is required.**

**The all-q4 control mz2 never ran.** `V0_all_q4_control` re-quantizes the shipped state at q4 in
the SD1M container: **+321 B** at renderer-field RGB rms **0.0013** — functionally lossless and a
*net cost*. So mz2's −823 B decomposes as **+321 B format penalty − 1,144 B from the q3 drop**,
and the lossless axis on this section now carries a measured price, not just a refusal.

**A WANS1-native prune was built and is worse.** Zeroing FiLM rows in the *shipped* format
(no receiver change at all) gives −1,144 B at keep25 against SM3R's −2,051 B *at identical
distortion* (mse 1.052e-5 both). SM3R physically compacts pruned rows and drops their scales;
explicit zeros still cost ANS symbols. SM3R's +321 B format penalty is more than repaid.

## 5. Deliverable 2 (cont.) — R2, and the proxy inversion that nearly shipped a fake win

I first ranked the precision waterfall by **weight-space MSE**. That table said the most efficient
targets by far were `blocks.2.pw.weight` and `blocks.3.pw.weight` (ΔMSE ~2.5e-8 for ~1,100 B each,
five orders of magnitude better than `frame_embed.weight` at 1.21e-5 MSE/B). A structural probe
appeared to confirm it: blocks 2/3 carry `pw.weight` rms **41× smaller** than blocks 0/1,
`pw.bias` 90× smaller, `film.bias` 20× smaller, and their `norm.weight` is **≈1.0003** — i.e. not
rescaled. The reading was "the renderer trained itself into using 2 of its 4 channel-mixing
blocks; quantizing the other two is nearly free." I built it: **V2 = −6,234 B, ΔS_rate −4.15e-3,
48.6% of the gap, at the lowest weight-MSE of any candidate.**

**It is catastrophically wrong.** Running the *real* renderer on the *real* decoded token field
(retained at `parseback/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8`, 600×384×512)
over a seeded **random** 24-pair sample:

| candidate | Δbytes | weight-MSE | **renderer RGB rms** | % pixels Δ>4 |
|---|---:|---:|---:|---:|
| V1 dead-pw q2 | −5,374 | 3.89e-07 (best) | **21.24** | 81.99% |
| V2 dead-pw+film q2 | −6,234 | 4.55e-07 | **21.24** | 81.99% |
| sm3r_keep25 | −2,051 | 1.05e-05 (27× worse) | **2.40** | 9.06% |

The weight-MSE screen **inverted the ordering**. Small absolute magnitude is not small function:
at q2 the *relative* error on those weights is ~40%, and they do precise work in the residual
stream. Every V-family candidate (V1–V8) is **REFUSED on the renderer field**, including my own
headline. Had I not built the forward probe I would have handed MAIN a −6,234 B "best candidate"
that destroys the image, with a clean-looking distortion number attached.

**Two claims I made earlier in this unit are WITHDRAWN**, both drawn from the same bad proxy:
1. "mz2 picked the worst tensor available (`frame_embed.weight`)" — **false**. On the renderer
   metric `frame_embed` is the **3rd most tolerant** of 16 (568 B per unit rms). mz2's q3 set is
   4 of the true top-5.
2. "mz2's mixed q3/q4 is dominated by sm3r_keep62" — **false**. C1 is −823 B at rms 1.301;
   keep62 is −748 B at rms 1.595. C1 is on the Pareto front.

The corrected per-tensor sensitivity order (bytes per unit renderer RGB rms at q3, higher =
better): `blocks.2.film` 6311 · `blocks.3.film` 2933 · `frame_embed` 568 · `blocks.0.film` 310 ·
`blocks.1.film` 244 · `blocks.2.pw` 181 · `blocks.3.pw` 108 · `blocks.1.pw` 89 · `blocks.0.pw` 59 ·
`coord_mix` 18 · `blocks.2.dw` 17 · `head` 15 · `blocks.0.dw` 13 · `blocks.3.dw` 8.5 ·
`blocks.1.dw` 6.3 · `token_embed` 4.5. **The FiLM weights are the tolerant surface; the depthwise
convs, `token_embed`, `head` and `coord_mix` are not.**

That reordering produced one genuinely new candidate that beats the mz2 family:
**`S2_film23_q2_top3_q3`** (`blocks.{2,3}.film` at q2, `blocks.{0,1}.film` + `frame_embed` at q3)
= **−1,333 B at rms 1.634**, which **Pareto-dominates `sm3r_keep50`** (−1,065 B at rms 1.774):
268 more bytes *and* less distortion.

## 6. The candidate ladder (all byte-closed, all parse-back PASS, deviation 0.0)

`ΔS_rate = 25·Δbytes/37,545,489`. RGB rms = renderer-field delta, n=24 seeded random pairs
(rng 20260817) — **advisory screen, not a score**. Pareto-front rows in **bold**.

| candidate | fam | archive B | Δbytes | ΔS_rate | RGB rms | B/rms |
|---|---|---:|---:|---:|---:|---:|
| **sm3r_keep01** | SM3R | 178,272 | **−2,889** | **−1.92367e-03** | 2.800 | 1032 |
| sm3r_keep03 | SM3R | 178,487 | −2,674 | −1.78051e-03 | 2.815 | 950 |
| **sm3r_keep06** | SM3R | 178,570 | −2,591 | −1.72524e-03 | 2.682 | 966 |
| **sm3r_keep09** | SM3R | 178,682 | −2,479 | −1.65066e-03 | 2.673 | 927 |
| sm3r_keep12 | SM3R | 178,861 | −2,300 | −1.53148e-03 | 2.596 | 886 |
| **sm3r_keep15** | SM3R | 178,899 | −2,262 | −1.50617e-03 | 2.455 | 921 |
| **sm3r_keep20** | SM3R | 179,053 | −2,108 | −1.40363e-03 | 2.357 | 894 |
| sm3r_keep25 | SM3R | 179,110 | −2,051 | −1.36568e-03 | 2.397 | 856 |
| **sm3r_keep37** | SM3R | 179,637 | −1,524 | −1.01477e-03 | 2.112 | 722 |
| **S2_film23_q2_top3_q3** | SD1M | 179,828 | −1,333 | −8.87590e-04 | 1.634 | 816 |
| sm3r_keep50 | SM3R | 180,096 | −1,065 | −7.09140e-04 | 1.774 | 600 |
| S1_top5_q3 = S3 | SD1M | 180,319 | −842 | −5.60653e-04 | 1.601 | 526 |
| **C1_mz2_mixed_q3q4** | SD1M | 180,338 | −823 | −5.48002e-04 | 1.301 | 632 |
| sm3r_keep62 | SM3R | 180,413 | −748 | −4.98062e-04 | 1.595 | 469 |
| **sm3r_keep75** | SM3R | 180,690 | −471 | −3.13620e-04 | 1.144 | 412 |
| **sm3r_keep87** | SM3R | 181,031 | −130 | −8.65617e-05 | 1.103 | 118 |
| V0_all_q4_control | SD1M | 181,482 | **+321** | +2.13741e-04 | 0.0013 | — |
| V1…V8 (dead-pw family) | SD1M | — | −5,374…−14,443 | — | **21.2…63.3** | **REFUSED §5** |

For every candidate `hpac_blob`, `carrier_blob`, `token_stream` and `residual_payload` are
**byte-identical to base** (asserted at build). Payloads: `/Volumes/APDataStore/pact/ddm_sa1/retained/`
(26 archives, 28 MB), shas in `CANDIDATES.json`.

## 7. Deliverable 2 (cont.) — sealed fire-orders for MAIN

Admit rule: `net ΔS = ΔS_rate + 100·Δd_seg + (√(10·d_pose_new) − √(10·d_pose_base)) < −3.5e-6`.
Base terms: seg 0.029611, pose 0.008295, rate 0.120628. **Derive the bar from
`canonical_frontier_pointer.json` at fire time — do not latch these literals.**

| # | candidate | Δbytes | rate credit | admits if d_seg ≤ (pose held) | admits if d_pose ≤ (seg held) |
|---|---|---:|---:|---:|---:|
| 1 | `sm3r_keep01` | −2,889 | 1.92367e-03 | 0.000315312 (+6.48%) | 1.04341e-05 (+51.7%) |
| 2 | `S2_film23_q2_top3_q3` | −1,333 | 8.87590e-04 | 0.000304951 (+2.99%) | 8.42479e-06 (+22.5%) |
| 3 | `sm3r_keep87` | −130 | 8.65617e-05 | 0.000296941 (+0.28%) | 7.01848e-06 (+2.0%) |

Fire 1 first (max credit). If it refuses, fire 2. Fire 3 only as a slope anchor — its credit is
too small to matter alone, but two admitted points let MAIN interpolate the whole 16-row ladder
without further fires. Advisory n600 first per mz2's own trigger; T4 only if a row beats the bar.
Full machine-readable order: `/Volumes/APDataStore/pact/ddm_sa1/FIRE_ORDERS.json`.

## 8. PRIOR-LAW PREDICTION — verdict

The charter pre-registered: lossless stays dead; lossy precision/rank yields **1–5 KB gross**;
rfo2's −15,157 B rung is **UNLIKELY** from post-hoc refit.

- **Lossless stays dead — CONFIRMED and priced.** Not merely "+bytes": the functionally-lossless
  reformat costs a *measured* +321 B.
- **1–5 KB gross — CONFIRMED at the low end.** Best byte-closed credit is **2,889 B**, inside the
  band but in its lower third.
- **−15,157 B unreachable — CONFIRMED.** Best is 19% of it. And that constant is itself stale:
  the live strict bar on rr4 is **12,816 B** (archive ≤ 168,345 B). The ceiling form is
  base-invariant and reproduces hv1's: 182,759 − 14,414 = 181,161 − 12,816 = **168,345 B**.
- **"If R0's two banked candidates both fail their own gate, say so plainly."** They can neither
  pass nor fail yet — distortion is unmeasured. What I can say plainly is stronger: **the banked
  archives are unusable** (e480b members), and one of the two (C1) survives re-derivation onto rr4
  while the *other* framing I built to replace it (V-family) is refused outright.

## 9. Honest limits — what this arm does NOT establish

1. **No distortion measurement.** RGB rms is a renderer-field screen on 24 of 600 pairs. It is not
   d_seg, not d_pose, and not a score. §5 is the standing proof that screens on this object can
   invert. The ladder's *ordering* by renderer rms is far better warranted than any *magnitude*.
2. **The renderer probe is n=24 random, not n=600.** Seeded random (never a prefix) per the
   prefix-bias law, because pose prefixes measure 2.54–4.21× harder than the population.
3. **d_seg vs d_pose are not separated.** Semantic edits move *both* (the renderer paints both
   frames); the carrier-side proofs that d_seg is invariant do **not** transfer here.
4. **SD1M and SM3R cannot be composed** on the shipping receiver — they are alternative packets.
   A combined prune×precision packet would need a new receiver (legal and free under rule 118,
   inflate.py is unsized) and is not built.
5. **`blocks.{1,2,3}.film` is the only prunable set** — `ROW_PRUNE_NAMES` is hardcoded and the
   mask is validated, so extending to `blocks.0.film` needs a receiver change.
6. **Full-frame inflate was not run.** Parse-back is verified at the semantic-section level
   through the shipping receiver module; the token stream is byte-identical to base so the
   ~25-min token decode path is unchanged by construction.
7. **The 8,284 B fixed-metadata block (23% of the section) is untouched.** It is exactly 2 B per
   scalar (1,739 fp16 scales + 2,403 fp16 params) — no float32 waste to reclaim — so sub-fp16
   metadata is a *new receiver format*, not an allocation change. Named, not built.
8. The `blocks.2/3` near-identity structure in §5 is **real and measured**; only the inference
   "therefore cheap to quantize" is falsified. Why a 41×-smaller weight is functionally
   load-bearing is not explained here and is a live question.

## 10. NEXT_IF_RESUMED

1. MAIN fires order #1 (`sm3r_keep01`). Two admitted points close the ladder.
2. **Deep prune below keep01 is exhausted** (keep03/keep01 rms is flat at ~2.80 while bytes still
   move) — the next rate on this section needs the metadata block (§9.7) or a composed
   prune×precision receiver (§9.4), both new-format work.
3. The ~278 B lossless carrier row (ra2 CPR1 inner coder ~230 B + ra1 `basis_scales` 48 B) remains
   live, unfired and unowned, blocked by a gate ra2 itself measured vacuous. Not mine; still open.
4. Explain §9.8 — a 41×-smaller weight that is functionally load-bearing constrains what the
   renderer is doing, and that is worth more than the bytes.
