# ddm_rc1 — the two MODEL sections were never entropy-coded. Coding them is −1,733 B at zero distortion.

**Arm:** `ddm_rc1` · **Base:** the cl2 frontier archive
`08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e` @ **179,982 B**,
**S 0.14781744131049854** `[contest-CUDA T4, n600]`.
**Tokens:** `[no-triality] [p0-ledger-ok]` · **Axis:** `[macOS-CPU advisory / scorer-free EXACT byte
measurement]` · `score_claim=false` until MAIN fires a T4 row.

`verdict_scope`: **INSTANCE** for the credit (these two bodies, as exact byte counts);
**FORMULATION** for the negative in §6 (the order-1 context designs I tried, on these bodies).
The distortion legs are **zero by construction**, not by measurement: the receiver restores both
section bodies byte-for-byte before any parsing runs, so every downstream length check, magic
check, and field offset sees exactly the bytes it sees today.

---

## 1. The candidate

| | value |
|---|---|
| archive | `1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122` @ **178,249 B** |
| Δ bytes vs cl2 | **−1,733 B** |
| rate ΔS | **−1.153933565760723e−03** (exact by construction) |
| d_seg ΔS | **0** — decoded state bit-identical |
| d_pose ΔS | **0** — decoded state bit-identical |
| **net ΔS** | **−1.153933565760723e−03** |
| projected S | **0.14666350774473783** |
| vs the −300 B admit bar | **5.78×** |
| twin encode | byte-identical (same sha) |

---

## 2. PRIOR-LAW PREDICTION vs MEASURED

The charter's predictions were written before any measurement. Both are stated on the **container**
basis (the bytes `archive.zip` actually holds), which is how the charter stated them.

| line | PREDICTED | MEASURED | verdict |
|---|---|---|---|
| SM3R body: adaptive coder reaches 2.4–2.9 bits/param vs Brotli's realized 3.72 | container **26,500–28,800 B** = **−2,050 … −4,350 B** | container **30,246 B** = **−610 B** | **MISSED LOW by 1,440–3,740 B.** The premise was wrong twice over: the code stream's own order-0 entropy is **3.281 bits/param** (MEASURED), not 2.4–2.9, and the "3.72 bits/param Brotli realized" figure divided the WHOLE body's container bytes by the param count, so it charged the codes for the fp16 scales, the fp16 tensors, and the prune masks as well. |
| IHS1 body: same coder | **−600 … −1,300 B** | container **12,343 B** = **−1,123 B** | **INSIDE the band.** |
| total | **−2,650 … −5,650 B** = −0.0018…−0.0038 S | **−1,733 B** = **−1.153934e−03 S** | **BELOW the band**, and still **5.78×** the admit bar. |
| FALSIFIER: best adaptive total within −300 B of shipped ⇒ Brotli already sits at these bodies' empirical entropy ⇒ CLOSE at family scope | — | **DID NOT FIRE** (−1,733 B) | the family stays **OPEN**; see §6 for what is closed. |

**The prediction's error is one identifiable mistake, not noise.** It priced the semantic win off a
bits-per-param figure computed on the wrong denominator. Divide the container bytes by the param
count and you get 3.72; divide only the CODE region's raw bytes by the param count and you get
3.893 packed, against a measured order-0 entropy of 3.281. The real headroom on that section was
0.61 bits/param, not 0.8–1.3.

---

## 3. Identity first: the shipped container streams re-pack byte-for-byte

Before any race, both raw bodies were restored and pushed back through the shipped packers. This is
the control that makes every later number a comparison against the object we actually ship.

| section | raw body | container transform | repack | shipped | byte-identical |
|---|---:|---|---:|---:|:--:|
| semantic | 36,130 B | ck2 whole-body 2-plane + Brotli q11 lgwin24 | 30,856 B | 30,856 B | **yes** |
| hpac | 17,770 B | Brotli q11 lgwin24 | 13,466 B | 13,466 B | **yes** |

RX1 member census (MEASURED): header 14 · hpac 13,466 · semantic 30,856 · carrier 22,031 ·
tail 113,515 (= residual table + the 113,419 B token stream) = 179,882 B member, 179,982 B
archive. The charter's recall said carrier 22,010 B; the measured section is **22,031 B**. It is
out of scope either way, but the charter number is corrected here rather than carried. `reserved = 0x1A`
(ck2-semantic 0x02 · RR5 0x08 · DX2 0x10 set; sz1 0x01 and ck2-carrier 0x04 clear).

The structural walks also re-serialize to identity, which is what licenses unpacking the codes:

* **SM3R v1 mode 6**, keep 1%, 60 fields, walk tiles the body exactly:
  18 B header/mask/depth-table · 4,806 B fp16 tensors · 2,338 B fp16 scales · 72 B prune masks ·
  **28,896 B packed codes carrying 59,376 params**. Per-tensor depths: 3 bits for
  `frame_embed.weight` and `blocks.0.film.weight`, 4 bits for the other fourteen.
* **IHS1**: 517 channels · 259 B depth table · **11,156 B packed weights carrying 20,416 params**
  (89,241 bits) · 6,351 B tail. Depth histogram
  `{0: 61, 1: 4, 2: 27, 3: 46, 4: 111, 5: 160, 6: 75, 7: 26, 8: 7}` — 61 channels are entirely
  pruned to zero width. Re-packing the unpacked rows reproduces the shipped weight bytes exactly.

---

## 4. The entropy table (RAW basis — bits per code, bytes over the code region only)

This is the bound the adaptive coder had to approach. `H0` is the plug-in order-0 entropy; `H1` the
plug-in first-order conditional entropy. For the semantic codes the alphabet is 8 or 16 and the
samples run to tens of thousands, so `H1` is trustworthy. For the hpac rows at depth ≥ 5 the
alphabet reaches 123 symbols on 1,011 samples, so the plug-in `H1` is **overfit**; the
Miller–Madow-corrected `H1` is given for those and is the honest reading.

### SM3R codes — 59,376 params, 28,896 B packed (3.893 bits/param)

| tensor | bits | count | packed B | zero % | H0 | H1 | H0 bytes | H1 bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token_embed.weight | 4 | 480 | 240 | 7.71 | 3.805 | 3.453 | 228.3 | 207.2 |
| frame_embed.weight | 3 | 4,800 | 1,800 | 52.19 | 1.889 | 1.884 | 1,133.2 | 1,130.4 |
| coord_mix.weight | 4 | 9,600 | 4,800 | 16.71 | 3.242 | 3.174 | 3,890.7 | 3,809.3 |
| blocks.0.dw.weight | 4 | 864 | 432 | 6.37 | 3.893 | 3.697 | 420.4 | 399.3 |
| blocks.0.pw.weight | 4 | 9,216 | 4,608 | 12.98 | 3.506 | 3.487 | 4,038.7 | 4,016.7 |
| blocks.0.film.weight | 3 | 1,536 | 576 | 29.43 | 2.659 | 2.631 | 510.5 | 505.1 |
| blocks.1.dw.weight | 4 | 864 | 432 | 5.90 | 3.892 | 3.710 | 420.3 | 400.7 |
| blocks.1.pw.weight | 4 | 9,216 | 4,608 | 13.43 | 3.489 | 3.467 | 4,019.1 | 3,994.5 |
| blocks.1.film.weight | 4 | 16 | 8 | 18.75 | 3.203 | 0.717 | 6.4 | 1.4 |
| blocks.2.dw.weight | 4 | 864 | 432 | 7.75 | 3.878 | 3.690 | 418.9 | 398.6 |
| blocks.2.pw.weight | 4 | 9,216 | 4,608 | 16.84 | 3.403 | 3.386 | 3,920.1 | 3,900.7 |
| blocks.2.film.weight | 4 | 16 | 8 | 6.25 | 3.203 | 0.850 | 6.4 | 1.7 |
| blocks.3.dw.weight | 4 | 864 | 432 | 6.83 | 3.884 | 3.671 | 419.5 | 396.4 |
| blocks.3.pw.weight | 4 | 9,216 | 4,608 | 15.58 | 3.425 | 3.407 | 3,946.0 | 3,925.0 |
| blocks.3.film.weight | 4 | 16 | 8 | 0.00 | 3.250 | 0.400 | 6.5 | 0.8 |
| head.weight | 4 | 2,592 | 1,296 | 21.10 | 2.967 | 2.906 | 961.4 | 941.5 |
| **TOTAL** | | **59,376** | **28,896** | **18.19** | **3.281** | **3.238** | **24,346.3** | **24,029.4** |

**The order-1 gain on this section is 317 B — 1.3%.** Quantized renderer weights are close to
memoryless in storage order. The three `blocks.N.film.weight` rows show a huge H0→H1 drop, but they
carry 16 params each: that is small-sample overfit, and the total involved is 20 B.

### IHS1 rows — 20,416 params, 11,156 B packed

| rows | bits | count | packed B | zero % | H0 | H1 (plug-in) | H1 (Miller–Madow) | H0 bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depth 0 | 0 | 3,355 | 0 | 100.00 | 0 | 0 | 0 | 0 |
| depth 1 | 1 | 88 | 11 | 60.23 | 0.970 | 0.972 | 0.988 | 10.7 |
| depth 2 | 2 | 948 | 237 | 10.76 | 1.658 | 1.569 | 1.578 | 196.4 |
| depth 3 | 3 | 1,461 | 547 | 9.45 | 2.701 | 2.646 | 2.673 | 493.4 |
| depth 4 | 4 | 3,206 | 1,603 | 8.89 | 3.846 | 3.710 | 3.764 | 1,541.2 |
| depth 5 | 5 | 3,734 | 2,333 | 8.41 | 4.692 | 4.343 | 4.497 | 2,189.9 |
| depth 6 | 6 | 2,999 | 2,249 | 8.27 | 5.302 | 4.322 | 4.613 | 1,987.5 |
| depth 7 | 7 | 3,614 | 3,162 | 9.10 | 5.495 | 4.132 | 4.410 | 2,482.5 |
| depth 8 | 8 | 1,011 | 1,011 | 8.11 | 5.588 | 3.221 | 3.588 | 706.2 |
| **TOTAL** | | **20,416** | **11,156** | | | | | **9,607.8** |

Order-0 total 9,607.8 B; Miller–Madow order-1 total **≈ 8,469 B**. So the hpac rows carry
**≈ 1,139 B of real first-order structure** on top of the order-0 win — measured, and (see §6) not
reachable by any context set I could afford.

---

## 5. The race (CONTAINER basis — the bytes `archive.zip` holds)

Every coded body was decoded by a **fresh decoder** and asserted byte-identical to the raw body
before its byte count was recorded. `id` below is that assertion.

### semantic — shipped 30,856 B

| coder | rider raw B | container B | Δ | id |
|---|---:|---:|---:|:--:|
| **Brotli q11 lgwin24 over ck2 — SHIPPED** | 36,130 | **30,856** | 0 | yes |
| xz −9e | 36,130 | 32,608 | +1,752 | yes |
| xz −9e over ck2 | 36,130 | 31,988 | +1,132 | yes |
| zstd --ultra −22 (measurement only) | 36,130 | 33,224 | +2,368 | yes |
| zstd --ultra −22 over ck2 (measurement only) | 36,130 | 32,624 | +1,768 | yes |
| RC1 adaptive tree shift 4 + Brotli | 32,121 | 31,097 | +241 | yes |
| RC1 adaptive tree shift 4 + Brotli over ck2 | 32,121 | 30,590 | −266 | yes |
| RC1 adaptive tree shift 5 + Brotli | 31,843 | 30,867 | +11 | yes |
| RC1 adaptive tree shift 5 + Brotli over ck2 | 31,843 | 30,342 | −514 | yes |
| RC1 adaptive tree shift 6 + Brotli | 31,792 | 30,863 | +7 | yes |
| **RC1 adaptive tree shift 6 + Brotli over ck2 — WINNER** | 31,792 | **30,246** | **−610** | yes |
| RC1 adaptive tree shift 7 + Brotli | 31,863 | 30,813 | −43 | yes |
| RC1 adaptive tree shift 7 + Brotli over ck2 | 31,863 | 30,406 | −450 | yes |

### hpac — shipped 13,466 B

| coder | rider raw B | container B | Δ | id |
|---|---:|---:|---:|:--:|
| **Brotli q11 lgwin24 — SHIPPED** | 17,770 | **13,466** | 0 | yes |
| xz −9e | 17,770 | 13,536 | +70 | yes |
| xz −9e over ck2 | 17,770 | 13,720 | +254 | yes |
| zstd --ultra −22 (measurement only) | 17,770 | 13,992 | +526 | yes |
| zstd --ultra −22 over ck2 (measurement only) | 17,770 | 14,143 | +677 | yes |
| RC1 adaptive tree shift 4 + Brotli | 16,268 | 12,377 | −1,089 | yes |
| RC1 adaptive tree shift 4 + Brotli over ck2 | 16,268 | 12,383 | −1,083 | yes |
| RC1 adaptive tree shift 5 + Brotli | 16,267 | 12,372 | −1,094 | yes |
| **RC1 adaptive tree shift 5 + Brotli over ck2 — WINNER** | 16,267 | **12,343** | **−1,123** | yes |
| RC1 adaptive tree shift 6 + Brotli | 16,314 | 12,398 | −1,068 | yes |
| RC1 adaptive tree shift 6 + Brotli over ck2 | 16,314 | 12,404 | −1,062 | yes |

**Three readings.**

1. **Neither generic coder beats the shipped Brotli on either body.** xz costs +1,752 B / +70 B;
   zstd costs +2,368 B / +526 B. The measurement baselines are a control, not a candidate — and
   they confirm the shipped container choice was already the best generic one.
2. **The ck2 plane transform survives the recode and still pays −617 B on the semantic section**
   (30,863 → 30,246), essentially the −613 B ck2 measured on the un-recoded body. It pays because
   the fp16 scales and fp16 tensors, which is where its whole credit lives, are still plain bytes
   in the rider's metadata prefix; the arithmetic payload it splits is incompressible either way.
   On hpac the transform is worth only −29 B, which is why the RC1 hpac stream carries it inside
   its own format rather than asking for a reserved bit of its own.
3. **The adaptation shift matters more than any context.** Semantic wants a slow shift (6, rate
   1/64); hpac wants a fast one (5). The semantic code distribution is stationary across 59,376
   symbols, so slow adaptation buys lower steady-state noise; the hpac rows switch depth every few
   hundred symbols, so the model has to keep re-converging.

---

## 6. What is CLOSED: the order-1 context designs (verdict_scope: FORMULATION)

The hpac rows carry ≈ 1,139 B of measured first-order structure. **Five conditioned designs
converted 10.3 B of it — 0.9%** — all measured as exact adaptive-model code length on the raw rows:

| design | hpac raw B | vs the order-0 tree (9,639.5 B) |
|---|---:|---:|
| binary tree, context = (depth, node) — **the winner** | **9,639.5** | — |
| + previous-magnitude bucket, 2/4 buckets | 9,640.1 – 9,667.2 | +0.6 … +28 |
| + previous bit-length bucket on depths ≥ 6, 2/3/4 buckets | 9,629.2 – 9,681.9 | −10 … +42 |
| sign + adaptive unary magnitude prefix (cap 4/8/16) + bypass | 10,163.4 – 10,797.1 | +524 … +1,158 |
| sign + adaptive bit-length prefix + bypass mantissa (exp-Golomb shape) | 10,289.4 – 10,382.7 | +650 … +743 |
| delta from the previous row element | 9,835.1 – 9,932.4 | +196 … +293 |

The single best-conditioned variant — previous bit-length, 4 buckets, shift 4 — beat the
unconditioned tree by **10.3 B**. That is 0.9% of the 1,139 B available and well inside the spread
of the shift sweep itself, so it does not survive as a design. On the semantic codes every
conditioned variant lost outright: adding a previous-magnitude bucket cost +69 to +213 B against
the unconditioned tree at every shift tried.

**The mechanism is context dilution, and it is measurable, not speculative.** Depth-7 rows hold
3,614 symbols over an alphabet of 120; the plug-in H1 needs 1,508 conditional cells, and
Miller–Madow attributes 0.277 of the 1.363-bit apparent gain to overfit. An adaptive coder has to
LEARN each cell from the same 3,614 symbols, and the learning cost **is** the bias. So the
structure is real **and unaffordable at this sample size**. That is a FORMULATION closure on the
five conditioned designs above, not a family closure — a coder with a genuinely cheaper prior (a
shared mixing weight across depths, a two-pass semi-static table paying explicit counted bytes for
the model) is untested and is registered as ITEM 1.

---

## 7. Where the bytes actually came from

| | semantic | hpac |
|---|---:|---:|
| packed codes, raw | 28,896 B | 11,156 B |
| order-0 entropy of those codes | 24,346 B | 9,608 B |
| RC1 adaptive coder realized | 24,544 B | 9,639 B |
| coder gap to its own order-0 bound | **198 B (0.81%)** | **31 B (0.32%)** |
| non-code metadata carried verbatim | 7,234 B | 6,614 B |
| container credit realized | **−610 B** | **−1,123 B** |

**The coder is essentially at its bound; the sections are not.** Both riders land within 1% of the
order-0 entropy of the very stream they code. The reason the semantic credit is only −610 B while
the raw code region drops 4,352 B is that **Brotli was already extracting most of that**: it sees
packed 4-bit nibbles as a 256-symbol byte alphabet with previous-byte context, which recovers the
order-0 nibble statistics almost fully. On hpac it cannot, because the row depths change every few
hundred symbols and no byte boundary aligns with a code boundary — which is exactly why the hpac
credit is 1.84× the semantic credit off a body 2.0× smaller.

---

## 8. The receiver

Five additive edits; every prior archive takes exactly the path it takes today.

| file | edit |
|---|---|
| `runtime/rc1_adaptive_model_sections.py` | NEW. Encoder reference and receiver implementation in one file, on the `dx2_cabac_coefficients.py` precedent. Integer-only: the same carryless 32-bit range coder and the same 12-bit binary probability model the shipped DX2 rider uses. No float, no device path, no transmitted table — every bin starts at 2048 and updates only from already-decoded bins. |
| `runtime/residual_archive.py` | Two NEW reserved bits: `RC1_RESERVED_SEMANTIC_ADAPTIVE = 0x20`, `RC1_RESERVED_HPAC_ADAPTIVE = 0x40`; `SZ1_RESERVED_KNOWN_BITS` 0x1F → 0x7F. The unknown-bit guard stays fail-closed. With the hpac bit set the body is un-interleaved and must then carry `RC1H`; with the semantic bit set the body must carry `RC1S` and is passed through as tagged. |
| `runtime/ihs2.py` | `materialize_ihs1` gains an `RC1H` branch that restores the packed weight bitstream using `layout_from_runtime(runtime).row_counts` — the same value-free model shell IHS2 already builds. |
| `cpr1/ddm_mp2_semantic_receiver.py` | `unpack_variant_semantic_or_none` restores an `RC1S` blob to its SM3R bytes before dispatch. This is the only seam that holds the renderer template the code geometry needs, which is why the restore lives here and not in the container. |
| `cpr1/inflate.py` | `RC1S` added to the tagged-semantic prefix tuple. |

**Counted bytes added by the receiver: zero.** `inflate.py` is a free, unsized interpreter
(`upstream/evaluate.py:63` charges `archive.zip` only). **Rule 118 is clean**: the context set is
`(tensor index, tree node)` and `(bit depth, tree node)` — pure geometry, derived at decode time
from the archive's own depth table and the model's own shapes. Nothing video-derived crosses into
runtime code, and the RC1 stream transmits no model.

---

## 9. Receiver proof

**(a) Section census — only the two MODEL sections and the reserved byte moved (MEASURED).**

| section | shipped | candidate | identical |
|---|---:|---:|:--:|
| RX1 `reserved` | `0x1A` | `0x7A` | no — RC1 bits `0x20` + `0x40` set |
| hpac | 13,466 B | 12,343 B | no — the credit |
| semantic | 30,856 B | 30,246 B | no — the credit |
| carrier | 22,031 B, sha `801a4445956ca5e9…` | 22,031 B, sha `801a4445956ca5e9…` | **yes** |
| tail (residual table + token stream) | 113,515 B, sha `04fefd6225d70ee4…` | 113,515 B, sha `04fefd6225d70ee4…` | **yes** |

**(b) No-op detector — the changed bytes are CONSUMED and are lossless (MEASURED).**
Parsing the candidate archive independently: both MODEL streams Brotli-decompress and
ck2-un-interleave to bodies carrying the `RC1S` and `RC1H` magics, and restoring them **through
the staged receiver's own modules** reproduces the shipped raw bodies byte-for-byte
(`semantic_body.sm3r.bin` sha `17e0fd0b…`, `hpac_body.ihs1.bin` sha `81728190…`). The receiver's
`ihs2.layout_from_runtime` row geometry equals the encoder's independently-derived geometry
(517 channels, 20,416 weights), and the patched semantic receiver's dispatch returns a state dict
equal tensor-for-tensor to the shipped body's.

**(c) Receiver decode identity, wall-clock, and the twin (MEASURED, `BUILD.json`).**

| gate | result |
|---|---|
| decode identity | **PASS** — the staged receiver decoded the candidate archive to token sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`, the shipped field, `np.array_equal` true over all 600×384×512 bytes |
| decoder bit position | 907,409 — identical to cl2's |
| corrected CDF input sha | `82dc3e3f067d651a…` — identical to cl2's |
| corrected quantized logit sha | `c913e20ff8e60a67…` — identical to cl2's |
| twin encode | **PASS** — a second build from the same inputs produced a byte-identical archive |
| candidate archive | `1438049e3655fbcf…` @ 178,249 B |

**Wall-clock — read the caveat before the number.** The candidate decoded in **2,172.5 s**
(36.2 min) on this Mac at `OMP_NUM_THREADS=4`. That is **NOT a contest-budget measurement and
must not be compared to cl2's 1,299 s**: three sibling arms were running concurrently
(`uptime` load average 19.4–23.6 throughout), so the machine was contended by roughly 5×. The
honest number is the RC1 **delta**, which I measured in isolation on the same tree:

| step | median of 3 |
|---|---:|
| semantic RC1 restore (59,376 symbols) | **0.077 s** |
| hpac RC1 restore (20,416 symbols) | **0.034 s** |
| the shipped path it replaces (Brotli + ck2 only) | 0.0001 s each |
| **RC1's addition to inflate** | **≈ 0.11 s** |

0.11 s of pure Python against a decode measured in the low thousands of seconds is **under
0.01%** of the budget. The two range-decoder passes are bounded by symbol count (79,792
symbols), not by video length, so nothing here scales with the clip. The 30-minute contest
budget is decided by the token decode and the renderer, exactly as before this change.

---

**(d) Seal.** `SEAL_ddm_rc1_model_section_adaptive_recode_contest_cuda.json`, seal sha
`b7811614b860dbbb6c42b52308dacd74c5b75089c47dad5430d01476a6af6aec`, verdict **SEAL_VALID**.
Runtime tree 43 files / 921,033 B, digest `c68a7871a7257940…`; eight receiver files pinned
individually; admit bar net ΔS < −2e−05 against contest_cuda 0.14781744 at zero tolerance; five
falsifiers pre-registered. Path:
`/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/SEAL_ddm_rc1_model_section_adaptive_recode_contest_cuda.json`.
**One ordering note, stated because it is a real gap:** the decode in (c) ran BEFORE
`tac.candidate_seal.repin_receiver` updated `inflate.py`'s two archive-pin constants from the
cl2 values to the candidate's. Those two constants are read by `inflate.py`'s own guard, not by
the `read_residual_archive` → `decode_production_tokens` path the identity used, and the re-pin
helper re-parses and refuses unless the result is CONSISTENT (it reported
`MISMATCH → CONSISTENT`). So the sealed tree differs from the decoded tree in exactly two
constant lines that the decode does not consult. **MAIN fires T4; I did not dispatch Modal.**

---

## 10. Verdict

**ADMIT.** −1,733 B at zero distortion, 5.78× the −300 B admit bar, decoded state bit-identical,
two encodes byte-identical, receiver decode identity proven on the exact field. The rate ΔS is
exact by construction; the seg and pose legs carry no exposure because there is no seg or pose leg.

The FALSIFIER did not fire: Brotli does **not** already sit at these bodies' empirical entropy. It
sits within 1.6% of it on the semantic body (which is why that credit is small) and 12% away on the
hpac body (which is where the credit is).

`score_claim=false`. The T4 row is MAIN's to fire.

---

## 11. Owed

## ITEM 1 — price a semi-static or depth-mixing model for the hpac rows
The measured Miller–Madow order-1 bound on the IHS1 rows is ≈ 8,469 B against the 9,639 B the
adaptive tree realized — **≈ 1,139 B of real structure, of which the five conditioned designs in
§6 converted 10.3 B (0.9%)**.
Two untested forms could pay for it with a cheaper prior: (a) a two-pass semi-static conditional
table whose counted bytes are paid explicitly and traded against the gain, and (b) a logistic
mixture that shares one weight vector across depths so each cell is not learned alone. Owner:
unassigned. Blocker: none — this is a pure local byte measurement on a retained body.

## ITEM 2 — the same coder on the token tail (113,419 B, out of ddm_rc1's scope)
The token stream is 113,419 B — 63.0% of the 179,982 B base archive — and is already RC64-coded,
so the mechanism does not transfer as stated. What DOES transfer is the finding in §7: a generic byte coder recovers most of a packed
code stream's order-0 statistics but loses the boundary structure when the code width changes.
Whether the RC64 tail has an analogous width-change blind spot is unmeasured.

## ITEM 3 — carry RC1 through a fresh container search
The winner used the shipped `(ck2, q11, lgwin24)` container shape unchanged. The rider changes what
Brotli sees, so the argmax over `(ck2, quality, lgwin)` may have moved. Measured cost: one sweep.
Expected size: tens of bytes, not hundreds.

---

**Frontier:** `cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]`
**Candidate (advisory, scorer-free exact byte arithmetic; `score_claim=false`):**
`rc1 projected S 0.14666350774473783 @ 178,249 B, archive sha 1438049e3655fbcf… [macOS-CPU advisory]`
