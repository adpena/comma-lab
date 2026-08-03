# ddm_cb2 — the token codebook race, the ECVQ crossover, and the free-codebook price

- arm: `ddm_cb2` (codebook · coder · selector, rate axis)
- date: 2026-08-02
- axis label: **`[macOS-CPU advisory, rate-only]`** — every byte figure is a real
  `archive.zip` stat or a real coder frame. `score_claim=false`,
  `promotion_eligible=false`, `pointer 0.1910828242 UNMOVED`. **No scorer ran** (the
  single n600 slot is held by another arm); every d_seg/d_pose consequence below is
  explicitly OWED, never asserted.
- operator directives honoured: rate is top priority; **the pose veto is lifted** — rate
  wins are reported in the decision column with pose debt named separately, never composed
  into a joint reject; every sub-FAMILY negative carries its named follow-on.

---

## Verdict first

**The token codebook family is worth at most 5.4% of the gap. Token GRANULARITY is
already measured at 9.7–16.8%, byte-closed, receiver-closed, at ZERO counted bytes.**

| candidate | archive.zip (MEASURED) | ΔS_rate | % of gap | counted cost | receiver |
|---|---:|---:|---:|---|---|
| shipped `dc1_fold` | **360,309** | — | — | — | live |
| **L=8** | **254,652** | **−0.07035** | **9.69%** | **0 B** | **VERIFIED** |
| **L=6** | **207,711** | **−0.10161** | **13.99%** | **0 B** | built |
| **L=5** | **177,404** | **−0.12179** | **16.77%** | **0 B** | built |
| best lossless recode (33 raced) | 346,478 | 0.00000 | 0.00% | — | incumbent wins |
| whole VQ family ceiling (m→∞) | ≥301,647 | ≥−0.03906 | ≤5.38% | 0 B (lattice) | DERIVED |
| selector minimised to its floor | 360,019 | −0.00019 | 0.027% | — | irrelevant |

At **L=5 the rate term is 0.118126 — below PR130's 0.127214.** Rate stops being the
binding axis at a token coarseness the receiver already supports today.

**The one thing owed:** the n600 scorer gate. Break-even is computed per row below; the
rate win survives any d_seg rise smaller than it, and pose debt is now explicitly
acceptable and separately repairable.

---

## §1 What we actually ship (re-derived from the bytes, not recalled)

`/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_dc1_fold_archive.zip`,
360,309 B, sha256 `9fb9f4e9…90d3cb`:

| member | stored B | share |
|---|---:|---:|
| `state/tokens.dr7t` | **346,478** (STORED) | **96.16%** |
| `state/pose_warp.stp` | 8,654 | 2.40% |
| `state/renderer.sec` | 3,341 | 0.93% |
| `manifest.json` | 753 (deflated from 1,450) | 0.21% |
| `state/selector.sec` | 314 (deflated from 535) | 0.09% |
| `state/pose_stub.sec` | 83 | 0.02% |
| ZIP structure | 686 | 0.19% |

Decoded payload: **(600, 24, 32, 4) = 1,843,200 tokens, alphabet 16** →
**1.5038 bits/token**. The receiver contract
(`ddm_tr1_runtime._token_codes` / `tokens_for_pair`) is a **uniform scalar lattice**:
`code = rint((x+1)/2·(L−1))`, `x̂ = code/(L−1)·2 − 1`, with `L = token_quant_levels`
carried as **one integer in the 535 B selector**.

**Consequence that reframes the whole codebook question:** the incumbent codebook is
already rule-118 FREE. It costs zero archive bytes because the receiver generates it
from one selector integer. "Make the codebook free" is not available as a lever — it
already is.

---

## §2 The lossless family — 33 formulations, SMEVR undefeated (MEASURED NEGATIVE)

`ddm_r7` raced its coders on the checkpoint fields (endpoint frame **557,238 B**). The
shipped `dc1_fold` field is a **different, far better-conditioned object** (346,478 B at
the same geometry), so r7's verdict did not automatically transfer. It was re-raced here
on the shipped bytes.

### 2a. r7's nine coders, re-raced on the SHIPPED payload

| codec | bytes | vs shipped |
|---|---:|---:|
| **smevr** | **346,478** | **0** (reproduces the shipped bytes exactly) |
| brotli11 | 396,442 | +49,964 |
| lzma1 | 398,024 | +51,546 |
| kt_o8_prev5_backoff | 398,517 | +52,039 |
| kt_prev1 | 406,153 | +59,675 |
| cae_inspired_identity_inter | 421,864 | +75,386 |
| rans_o0 | 475,083 | +128,605 |
| huffman_nibble | 519,729 | +173,251 |
| rans_o0_on_adjacent_innovation | 536,456 | +189,978 |

Every row round-trips (`np.array_equal` asserted). SMEVR re-encodes the shipped member
byte-for-byte — so the shipped archive **is** the canonical SMEVR encoding.

### 2b. QA08 context-MIX — eighteen KT-charged context models, all lose

Code length `Σ_c [ n_c·H_c + ((K−1)/2)·log₂ n_c ]` (Rissanen/KT two-part; achievable
within O(1) by an adaptive KT coder). Best six of eighteen:

| context | contexts | plug-in | +model | KT total | vs SMEVR |
|---|---:|---:|---:|---:|---:|
| P,P2,ch | 949 | 359,592 | 7,100 | 366,692 | **+20,214** |
| P,bin(L),bin(U),ch | 989 | 374,195 | 8,193 | 382,388 | +35,910 |
| P,L | 249 | 383,281 | 2,278 | 385,558 | +39,080 |
| P,base | 227 | 386,275 | 2,130 | 388,405 | +41,927 |
| P,L,U | 3,523 | 370,923 | 19,115 | 390,038 | +43,560 |
| P,base,L,U,ch | 89,377 | **292,018** | **158,574** | 450,592 | +104,114 |

The last row is the whole story. There **is** 54,460 B of real conditional structure in
the token field — and **no two-part or KT coder can reach it**, because resolving it costs
158,574 B of model. That is the context-dilution wall, measured exactly.

### 2c. QA08's named follow-on (P0 per operator directive 3) — parameter sharing, also loses

The classical escape from dilution is to share parameters across contexts. Both forms were
built and measured:

| shared model | entropy | model | **assignment (COUNTED)** | total | vs SMEVR |
|---|---:|---:|---:|---:|---:|
| 16 clustered pmfs | 352,649 | 236 | **44,688** | 397,574 | +51,096 |
| 64 clustered pmfs | 338,596 | 846 | **67,033** | 406,475 | +59,997 |
| 256 clustered pmfs | 325,166 | 2,859 | **89,377** | 417,402 | +70,924 |
| 1024 clustered pmfs | 314,373 | 9,266 | **111,721** | 435,361 | +88,883 |
| sign × magnitude (free ctx) | 306,948 + 86,505 | — | 0 | 393,452 | +46,974 |

**The mechanism, and it is rule-118 itself:** clustering collapses the model cost to
almost nothing (236 B at k=16) but the *cluster assignment* — which context maps to which
shared distribution — is **fitted to our token field**, therefore video-derived, therefore
**COUNTED**. It costs 44,688–111,721 B, more than the entropy it buys. The one form with a
zero-byte assignment (sign × magnitude, contexts built only from already-decoded state)
still loses by 46,974 B.

**SMEVR wins for a reason that is now stated rather than observed: its model is derived
entirely from already-decoded state, so it ships nothing.** Any coder that must ship its
adaptation loses on this payload.

**QA08 verdict: MEASURED NEGATIVE.** verdict_scope: **FORMULATION** as a race outcome
(32 alternatives on one token field), rising to **FAMILY for FITTED models** — the
rule-118 mechanism is not field-specific. It does **not** close decoder-derived models.

**QA08's remaining live follow-on:** **QA39 carried-ξ token INTER-prediction** — warp the
previous decoded token field by the pose the archive already carries (`pose_warp.stp`,
8,654 B) and code only the innovation. That predictor is decoder-derived, so it ships
nothing and sits on the correct side of the boundary this section just measured.
`ddm_gc8` already named it "THE WINNER" and it remains **UNRACED**. It is the single
highest-value unraced lossless row and this memo does not close it.

---

## §3 The ECVQ crossover (`ddm_gc6` seat T4) — ANSWERED

> *"Fixed lattice + entropy coder vs entropy-constrained codebook at the 0.004-distortion
> operating point — which side of the crossover are we on?"*

The source is discrete on 16 lattice points, so the MSE-optimal / rate-constrained
M-level scalar quantizer is an **optimal contiguous partition — solvable exactly by DP**.
No Lloyd-Max local minima; this is the true ECVQ optimum, then coded with the live SMEVR.

| fidelity (rmse) | uniform lattice | ECVQ | winner |
|---:|---|---|---|
| 0.036 | L=15 → 344,579 (rmse 0.0388) | **M=11 → 314,336 (rmse 0.0360)** | ECVQ **−30,243 B at better fidelity** |
| 0.043 | L=12 → 311,620 | **M=10 → 297,342** | ECVQ **−14,278 B** |
| 0.052 | L=10 → 281,720 | **M=9 → 275,822 (rmse 0.0512)** | ECVQ **−5,898 B at better fidelity** |
| 0.065 | L=8 → 254,652 (rmse 0.0648) | **M=8 → 254,671 (rmse 0.0576)** | ECVQ: **+19 B for 11% better fidelity** |
| 0.089 | **L=6 → 207,711** | M=6λ → 202,617 (rmse 0.0991) | uniform (ECVQ ~+5,300 at matched) |
| 0.118 | **L=5 → 177,404** | M=6λ/M=4λ interp ≈ 188,000 | uniform **−10,600 B** |
| 0.151 | L=4 → 149,539 (rmse 0.1575) | M=4λ → 149,547 (rmse 0.1514) | tie; ECVQ 4% better fidelity |

**ANSWER: the crossover sits at rmse ≈ 0.075–0.085, archive ≈ 210–235 KB. We ship at
rmse 0 / 360,309 B — deep on the ECVQ-favourable side.** Finer than the crossover, ECVQ
wins by 6–30 KB; coarser, the uniform lattice wins by 5–11 KB, because the uniform
partition is more entropy-friendly to SMEVR's contexts than the MSE-optimal one.

**Counted cost of the ECVQ codebook: 2M bytes** (M fp16 reconstruction values, M ≤ 16)
— **at most 32 B, 0.003% of the gap.** This is the load-bearing pricing result of §3 and
§4 together: at scalar dimension the free-vs-counted codebook question is **moot**. Pay
the 32 bytes.

**Practical consequence at the operating point that matters:** if L=8 is gated, ship
**ECVQ M=8 instead of uniform L=8** — same bytes (+19 B), **11% lower reconstruction
error** (rmse 0.05759 vs 0.06480, max deviation 0.12212 vs 0.13333). That is free
d_seg-risk insurance on the largest rate move. It needs a non-uniform dequant table in
the receiver (free code) plus 16 counted bytes in the selector.

---

## §4 QA13 — the seed-generated free VQ codebook, PRICED and DOMINATED

### 4a. The derived ceiling

SMEVR already codes the memory (its contexts are mode-base + temporal + spatial +
renewal-age). The **memory gain of VQ is therefore already spent**; the only gain left is
the **space-filling (granular) gain**, bounded by the best lattice in dimension m
(Conway–Sloane / Forney):

| lattice | granular gain | b/token | bytes | ΔS | % of gap |
|---|---:|---:|---:|---:|---:|
| m=2 A2 (hex) | 0.1671 dB | 0.02775 | 6,395 | −0.00426 | 0.59% |
| m=4 D4 | 0.3657 dB | 0.06074 | 13,995 | −0.00932 | 1.28% |
| m=8 E8 | 0.6590 dB | 0.10946 | 25,219 | −0.01679 | 2.31% |
| m=24 Leech | 1.0329 dB | 0.17156 | 39,528 | −0.02632 | 3.62% |
| **m→∞ sphere bound** | **1.5329 dB** | **0.25461** | **58,662** | **−0.03906** | **5.38%** |

**The entire VQ family — every dimension, every codebook, perfectly realised — is capped
at 5.38% of the gap. The already-measured scalar coarsening at L=8 is 9.69%, at L=6
13.99%, at L=5 16.77%.** Quantizer SHAPE is second-order to quantizer COARSENESS by
1.8×–3.1×.

### 4b. The measured check — m=2 VQ loses everywhere

Weighted Lloyd over the 256 lattice pairs (6 restarts, deterministic seed), indices coded
with the live SMEVR:

| M | rmse | archive | matched-fidelity scalar | VQ penalty |
|---:|---:|---:|---|---:|
| 128 | 0.0327 | 333,525 | ~328,000 | +5,500 |
| 96 | 0.0436 | 316,526 | L=12 → 311,620 (better rmse) | +4,900 |
| 64 | 0.0600 | 293,186 | ~259,000 | +34,000 |
| 48 | 0.0715 | 277,794 | ~239,000 | +39,000 |
| 32 | 0.0892 | 248,845 | **L=6 → 207,711** | **+41,134** |
| 16 | 0.1344 | 169,486 | ~162,000 | +7,500 |

*(M > 16 rows are generous to VQ: SMEVR's alphabet caps at 16, so those rows were credited
with SMEVR's full conditional gain ratio applied to their order-0 entropy. The negative is
therefore conservative.)*

**m=2 VQ is worse than scalar+SMEVR at every matched fidelity, by 4.9–41.1 KB** — an order
of magnitude more than the 6,395 B it could theoretically win. The mechanism: merging two
channels into one symbol **destroys the coder's context structure**, and SMEVR had already
captured that dependency. The VQ memory gain is double-counted.

### 4c. QA13's premise, corrected

QA13 proposed a codebook `~N(0, I_m)` expanded from a counted seed. Three measured/derived
facts dominate it on both sides:

1. At scalar dimension the codebook costs **≤32 counted bytes** — free-vs-counted is
   irrelevant, and a *trained* scalar codebook is strictly better than a random one.
2. At m=2 the trained codebook costs 64–512 B while the VQ **loses by 4,900–41,134 B**.
   Making the codebook free saves ≤512 B against a ≥4,900 B loss — **free does not rescue
   it.**
3. A random Gaussian codebook has **no packing gain**; the free structure that actually
   captures the space-filling gain is a **generic mathematical LATTICE** (E8, Leech) — also
   rule-118 free (defined by mathematics, not by our video), and strictly better than
   random at the same size.

**QA13 verdict: MEASURED NEGATIVE** (m=2, trained + free variants). verdict_scope:
**FORMULATION** for the m=2 channel-adjacent measurement; **FAMILY-CEILING DERIVED** at
5.38% of the gap for all m.

**QA13's named follow-on (P0):** replace the seed-random-Gaussian formulation with a
**free generic lattice** (E8, ceiling 2.31% of gap; Leech, 3.62%) applied so that the
index stream **preserves the token grid** SMEVR conditions on — the failure mode measured
in 4b was the destruction of that structure, not the lattice idea. **Gate it behind §5:**
its whole ceiling is smaller than the granularity move already in hand.

---

## §5 The lever that is actually large — token granularity, BYTE-CLOSED and RECEIVER-CLOSED

Real `archive.zip` files, written with the v4d writer and `stat`-ed. The control rebuild
of the shipped archive is **byte-identical (360,309 B)**, so these are true archive stats,
not arithmetic.

| L | **archive.zip** | ΔS_rate | **% of gap** | rate term | rmse | break-even d_seg rise | rel. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 (shipped) | 360,309 | — | — | 0.239915 | 0 | — | — |
| 12 | 311,620 | −0.03242 | 4.46% | 0.207495 | 0.0430 | +0.000324 | +7.5% |
| 10 | 281,720 | −0.05233 | 7.20% | 0.187586 | 0.0517 | +0.000523 | +12.1% |
| 9 | 264,890 | −0.06354 | 8.75% | 0.176379 | 0.0567 | +0.000635 | +14.7% |
| **8** | **254,652** | **−0.07035** | **9.69%** | 0.169562 | 0.0648 | **+0.000704** | **+16.3%** |
| 7 | 230,392 | −0.08651 | 11.91% | 0.153409 | 0.0775 | +0.000865 | +20.1% |
| **6** | **207,711** | **−0.10161** | **13.99%** | 0.138306 | 0.0891 | **+0.001016** | **+23.6%** |
| **5** | **177,404** | **−0.12179** | **16.77%** | **0.118126** | 0.1182 | **+0.001218** | **+28.2%** |

Denominators from `tac.canonical_equations.gap_decomposition_against_floor_20260802`:
total gap **0.7263015**, 1% = **10,908 B**, shares seg 55.28% / pose 29.19% / rate 15.53%.

**Counted cost of every row: ZERO.** `token_quant_levels` is one integer already in the
535 B selector; the JSON re-serialises to the same 535 B.

### Receiver closure and the consumption proof (NO-FAKE #1)

`experiments/ddm_v4d_verify_decode.py` on `cb2_levels08_archive.zip` — the real vendored
receiver, no `tac` import:

```
A_ok true · B_pose_reconstruct_exact true · B_ab_bit_exact true · B_selector_exact true
B_beta_exact true · C_recompute_byte_exact true · C_two_plane_does_work true
D_archive_bytes 254652 · D_archive_sha256 a9e99a69a6abf166785e342c568a21e367c23c81238d9364d5dd964f5fc9d67a
```

Mutation control, both archives instantiated through `Decoder` + `render_frame1_camera_uint8`:

| | shipped | L=8 |
|---|---|---|
| `selector["token_quant_levels"]` seen by receiver | 16 | **8** |
| decoded code alphabet | max 15, 16 symbols | **max 7, 8 symbols** |
| rendered camera frame_1 | — | **differs: 76.32% of pixels, mean\|Δ\| 2.41/255, max 61** |

The bytes are consumed and the render moves. This is not an inert-section saving.

### The scorer-free risk ladder (5 pairs, full camera resolution)

| L | mean\|Δ\| /255 | p99 | px changed | px \|Δ\|≥8 |
|---:|---:|---:|---:|---:|
| 12 | 1.558 | 11.6 | 66.2% | 2.95% |
| 10 | 1.852 | 13.8 | 70.2% | 4.22% |
| 9 | 2.132 | 15.2 | 73.6% | 5.18% |
| 8 | 2.399 | 16.8 | 76.1% | 6.27% |
| 7 | 2.781 | 20.6 | 77.9% | 8.38% |
| 6 | 3.174 | 23.8 | 79.8% | 10.22% |
| 5 | 4.341 | 32.2 | 84.3% | 15.52% |

Monotone and smooth — no cliff between L=12 and L=5. This is a **proxy**, not d_seg:
SegNet reads regions and its argmax flips at boundaries, so photometric mean is a weak
predictor. It orders the rows; it does not gate them.

### Two caveats that travel with every rmse in this memo

1. **The ladder re-quantises the SHIPPED reconstruction**, i.e. a cascade of two
   quantizers. Quantizing the true latent directly to L levels would give **lower**
   distortion at the same bytes. The rmse column is an **UPPER BOUND**.
2. **The token field was trained with L=16 STE.** A field retrained at L=8 would be a
   better L=8 field. The ladder is therefore a **conservative** estimate of the achievable
   rate/distortion — the "granularity re-race from birth" (`ddm_gc6` row 10) can only
   improve on it.

Both point the same way: the measured numbers understate the lever.

---

## §6 The selector, re-derived under the inverted ratio

535 B raw, **314 B deflated in the archive** (0.087%). Twenty keys, every one pinned by
`_validate_selector`. A binary re-encoding of the ~10 genuinely varying scalars would be
~24 B + framing ⇒ **minimisation ceiling ≈ 290 B = 0.027% of the gap.** Irrelevant.

**The historical tension ("selector weights compete with mask bytes") is not merely
inverted — it is inverted by three orders of magnitude, and the correct conclusion is the
opposite of shrinking it:**

| selector change | its own cost | what it buys | lever arm |
|---|---:|---:|---:|
| `token_quant_levels` 16 → 8 | **0 B** (same 535 B JSON) | −105,657 B | **∞** |
| add ECVQ M=8 recon table | +16 B counted | −105,638 B at 11% better fidelity | **~6,600×** |
| minimise the whole selector | −290 B | −290 B | 1× |

**Policy, derived:** the selector is the free-interpreter control surface, and the
operator's 2026-07-21 deletion of the `inflate.py` LOC cap applies to it directly. It
should be **grown** — every additional decode program it can name (non-uniform dequant,
per-channel levels, per-region levels, a lattice index rule) is paid in free interpreter
work and returns ~10³–10⁴× its counted cost. The only bytes that must stay counted are
genuinely video-derived parameters, and at this scale those are tens of bytes.

**Immediately available and unmeasured:** `token_quant_levels` is currently **one global
integer**. Making it **per-channel** (4 integers, ~12 counted bytes) or **per-region**
allows spending fidelity where d_seg reads it and coarsening where it does not — a
waterfill on the same axis. This is the natural composition with `#766`.

---

## §7 Round-2 pool disposition

| row | verdict | scope | named follow-on (P0) |
|---|---|---|---|
| **QA08** context-MIX | **MEASURED NEGATIVE** — 18 KT models (+20,214 best) + 5 shared models (+51,096 best) + 9 r7 coders | FORMULATION; **FAMILY for FITTED models** (rule-118 mechanism) | **QA39 carried-ξ INTER-prediction** — decoder-derived, ships nothing, UNRACED, named "THE WINNER" by `gc8` |
| **QA13** VQ / seed codebook | **MEASURED NEGATIVE** (m=2: +4,900…+41,134 B) + **FAMILY CEILING DERIVED 5.38%** | FORMULATION (m=2 channel-adjacent); ceiling FAMILY | **free generic LATTICE (E8/Leech)** instead of seed-random Gaussian, structured to preserve the token grid — gate behind §5 |
| **QA12** token-LOTTO | **DERIVED-DOMINATED, not raced** | — | a seed basis reduces *representation*, not the field's *information*; it wins only by being a better PREDICTOR, which is exactly SMEVR's decoder-derived mode-base. Falsifier: a seed basis whose selection stream codes below 346,478 B |
| **QA09** Cl(2) | **NOT REACHED** | — | its definition was not located in the corpus within budget; it is the only round-2 row this memo does not touch |
| **ECVQ crossover** (`gc6` T4) | **ANSWERED** — crossover at rmse ≈0.075–0.085 / ≈210–235 KB; we are on the ECVQ side | MEASURED | ship ECVQ M=8 rather than uniform L=8 (+19 B, 11% better fidelity) |
| **`gc6` row 10** granularity re-race | **MEASURED at $0**, byte-closed and receiver-closed | MEASURED | the n600 gate below |

---

## §8 What is OWED — the exact measurement, blocked on the scorer slot

**BLOCKED-ON-SLOT.** One n600 realized-through-R gate on
`/Volumes/VertigoDataTier/pact/ddm_cb2_20260802/cb2_levels08_archive.zip`
(sha256 `a9e99a69…9d67a`, 254,652 B) returning `d_seg` and `d_pose`.

**Pre-registered decision rule (operator directive 2 — decision column is RATE):**

- **ADOPT** if `d_seg < 0.00501579` (= 0.00431179 + 0.000704). The rate win pays for the
  seg cost on its own, before any pose accounting.
- Any `d_pose` rise is **recorded as named, repairable debt**, not a reject. The pose
  carrier (`pose_warp.stp`, 8,654 B) is re-solvable against the new base — that is exactly
  the `#889` / `cr2r` mechanism, which says the pose must be solved against the base it
  ships with. A transplanted pose onto a coarsened base is expected to degrade and is not
  evidence against the rate move.
- **If ADOPT:** re-run at L=6 (break-even +0.001016) and L=5 (+0.001218). The ladder is
  monotone and smooth, so a binary search over three gates finds the knee.
- **If REFUSE at L=8:** gate L=12 (break-even +0.000324, 4.46% of gap) before concluding
  anything about the family — a refusal at L=8 falsifies L=8, not granularity.

Falsifiers for this memo's own claims:
1. The gate returns `d_seg` outside `[0.00431179, 0.0060]` at L=8 ⇒ the photometric proxy
   ladder in §5 is uninformative and should not be used to order future rows.
2. Any coder beats 346,478 B losslessly on this exact field ⇒ §2's FAMILY-for-fitted-models
   scope is over-claimed and drops back to FORMULATION.
3. An m≥4 VQ beats scalar at matched fidelity by more than 5.38% of the gap ⇒ the
   granular-gain derivation in §4a is misapplied (most likely because SMEVR does *not*
   capture the memory as fully as assumed).

---

## §9 Verdict-scope ladder

| claim | scope | basis |
|---|---|---|
| tokens = 96.16% of the shipped archive | **MEASURED EXACT** | ZIP infolist |
| shipped bytes are the canonical SMEVR encoding | **MEASURED EXACT** | re-encode delta 0 |
| L=8 → 254,652 B, receiver-closed, render changes | **MEASURED EXACT** | real `archive.zip` stat + `ddm_v4d_verify_decode` + mutation control |
| SMEVR beats 32 lossless alternatives on this field | **MEASURED** | FORMULATION (one field); FAMILY for fitted models |
| fitted models must ship their fit and lose | **MEASURED + rule-118 DERIVED** | §2c assignment cost 44,688–111,721 B |
| ECVQ crossover at rmse ≈0.075–0.085 | **MEASURED** | DP-exact partitions + live SMEVR |
| VQ family ceiling 5.38% of gap | **DERIVED** (Conway–Sloane/Forney granular gains) | validated in direction by the m=2 measurement |
| m=2 VQ worse than scalar everywhere | **MEASURED** | FORMULATION (channel-adjacent m=2) |
| rmse columns are upper bounds | **DERIVED** | double-quantisation cascade + L=16 STE training |
| d_seg / d_pose at any L≠16 | **UNMEASURED — OWED** | no scorer slot |
| photometric ladder predicts d_seg | **ASSUMED** | proxy only; SegNet reads regions |

---

## §10 Custody

`/Volumes/VertigoDataTier/pact/ddm_cb2_20260802/`

| archive | bytes | sha256 |
|---|---:|---|
| `cb2_levels16_archive.zip` (control) | 360,309 | `274cfea8c906af9d7681073ae83b8f6638102850066933df67da551e49ea15cf` |
| `cb2_levels12_archive.zip` | 311,620 | `45eacae88c7d279871859040eebf00d056c77d11c0d9fbf9ec47cadc2b47c413` |
| `cb2_levels10_archive.zip` | 281,720 | `827c13d6329ddded98664e9f43d9632a494d626115adbb6831b5de6ef781e3c2` |
| `cb2_levels09_archive.zip` | 264,890 | `cfd183fac16b8c9db98b27246ee58742cbca35381c445bdcbcb78e0b3f8cf097` |
| **`cb2_levels08_archive.zip`** | **254,652** | `a9e99a69a6abf166785e342c568a21e367c23c81238d9364d5dd964f5fc9d67a` |
| `cb2_levels07_archive.zip` | 230,392 | `3004d8ffd3a1ee864d07340e1fde0cb2f3e56464677c9a9fd31ce7c015a5dcbc` |
| `cb2_levels06_archive.zip` | 207,711 | `84a516bf33c9123dfe5ed9c6147f7f38f7213164bf7bc1ece3ed667c8ff9ba27` |
| `cb2_levels05_archive.zip` | 177,404 | `dc54fecf2cd276fccb8ebd8eeffcc62494875da9e097e0d55f39cdcf160c2de3` |

The L=16 control rebuild is byte-identical to the shipped
`v4d_composed_dc1_fold_archive.zip` (`9fb9f4e9…90d3cb`), which is what licenses reading
the other seven as true archive stats.

Sources re-checked at the artifact, not recalled: `ddm_r7_token_coder_race_20260729.md`
(+ receipt `a6503c69…`), `ddm_cv1_seven_surface_convocation_20260802.md` §5/§6,
`ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802.md`,
`ddm_deferral_queue_ledger_20260729.md` (QA08/QA12/QA13/QA39 definitions),
`ddm_gc8_postreversal_convocation_20260729.md`, `src/tac/optimization/ddm_tr1_runtime.py`,
`experiments/ddm_r7_token_coder.py`,
`/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/inflate_runner_v4d.py`.

---

## §11 Own round-1 adversarial review

**Attacked and held:**
- *Is the zip writer faithful?* The L=16 control rebuild is byte-identical to the shipped
  archive. Without that check every other row would be a prediction.
- *Does the receiver ignore the changed selector?* No — proved by reading
  `packet.selector` out of the instantiated `Decoder` (16 vs 8), the decoded alphabet
  (16 vs 8 symbols) and the rendered camera frame (76.32% of pixels move).
- *Do the coder rows actually round-trip?* Every encode is `np.array_equal`-asserted
  against its decode; SMEVR additionally re-encodes the shipped member byte-for-byte.
- *Is the KT charge fair to the alternatives?* It is the standard two-part/Rissanen bound
  that an adaptive KT coder achieves asymptotically — the same class SMEVR belongs to. The
  plug-in column is reported alongside so the dilution is visible rather than hidden.
- *Is the VQ comparison rigged?* It is rigged **for** VQ: the M>16 rows were credited with
  SMEVR's full conditional gain ratio they cannot actually get, and they still lose.

**Attacked and conceded — three real weaknesses:**
1. **The ECVQ crossover is interpolated at two of seven fidelity points** (rmse 0.089 and
   0.118) rather than measured at matched rmse. The measured points bracket them and the
   sign is stable across the bracket, but the ±5,300 B and ±10,600 B figures at those two
   rows are interpolations, not measurements.
2. **m=2 VQ was tested on one pairing axis** (channel-adjacent). A spatial 2×1 or 1×2
   pairing might interact better with SMEVR's contexts. The scope is written as
   FORMULATION(channel-adjacent) for exactly this reason, and the derived ceiling
   (0.59% of gap at m=2) means the correction cannot be large.
3. **The photometric ladder is 5 pairs, not 600.** It orders the rows; it must not be read
   as an estimate of d_seg. It is labelled ASSUMED in §9.

**What I did not do:** QA09 Cl(2) was not located or raced; no scorer ran, so nothing here
is a score claim; and the memo's largest number (L=5 at 16.77% of the gap) rests on a
distortion the scorer has not yet priced. The rate side is closed and byte-closed; the
seg/pose side is one gate away and that gate is named in §8.

---

## NEXT-IF-RESUMED

1. **Fire the §8 gate on `cb2_levels08_archive.zip` the moment the scorer slot frees.**
   Everything else in this memo is subordinate to that one number.
2. **Race QA39** (carried-ξ token INTER-prediction). It is the only unraced lossless row
   sitting on the correct side of the rule-118 boundary §2 measured, and `gc8` already
   called it the winner. Scorer-free; uses `pose_warp.stp` which is already in the archive.
3. **Build the ECVQ M=8 receiver path** (non-uniform dequant from a 16-byte selector
   table) — 11% fidelity insurance on the largest rate move for +19 B.
4. **Per-channel / per-region `token_quant_levels`** — ~12 counted bytes, turns the single
   global lever into a waterfill; composes directly with `#766`.
5. **Granularity re-race from birth** (`gc6` row 10): retrain with L=8 STE from the start.
   §5's caveat 2 says the measured ladder is a conservative bound on what that would give.
