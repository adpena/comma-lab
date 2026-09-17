# ddm_pd8 — the PRICE-FIRST generator, pass 3, on the move-54 field, and the 16 "dead" carrier bytes

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd8_price_first_pass3_20260916`. Base: **move 54**, S 0.13605599532783202 @ 179,266 B,
archive sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`, re-derived from
`reports/latest.md` and `.omx/state/canonical_frontier_pointer.json` at bind time. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container. **No
score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.** No
Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` at PoseNet batch 1 on the moved render,
with `outer_rounds=40, max_gn_iterations=400`; every n600 number comes from
`ddm_up2_shipping_pose_solve.measure_pose` at batch 8.

---

## 1. THE DEAD BYTES ARE NOT DEAD — the lever is CLOSED, and the projection was 2× too big

The second fresh reader of the minimal receiver found 16 bytes in the carrier section at offsets
123–138 that `submissions/mrs6`'s `decode_carrier` never reads, and header flag bit 0x20 set but
unnamed, and priced the pair at **−16 B (0.64 bar)** with an identical decode. The charter asked
whether the SEALED receiver — the tree that actually decodes the shipped archive — also never
reads them. **It reads BOTH. No archive change was produced, and this arm ships no container
change.** Two independent reasons, each sufficient:

**(a) The 16 bytes are a SENTINEL, and the sealed receiver consumes them.** MEASURED by reading
`base/move54_runtime/runtime/residual_archive.py:134`:

```python
lengths = _unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
```

— the 32 basis-alphabet Huffman code lengths, packed at 4 bits each = exactly 16 B. The
arithmetic-basis rider names the same span at `runtime/rr5_arith_basis.py:44`
(`PACKED_LENGTHS_SPAN = (123, 139)  # 32 symbols x 4 bits = 16 B`), and its
`restore_carrier_body` (lines 510–519) reads it and branches on it:

```python
lengths = packed_lengths(bytes(fields["metadata"]))
if not int(np.asarray(lengths, dtype=np.int64).sum()):
    lengths = huffman_lengths_from_histogram(np.bincount(symbols, minlength=BASIS_ALPHABET))
```

**An all-zero span is the rider's signal to REGENERATE the lengths from the decoded symbol
histogram.** MEASURED on move 54's own archive: header flags 250 = `0b11111010`, so
`RR5_RESERVED_ARITH_BASIS` (0x08) is set, and `car[123:139] == b"\x00" * 16`. The zeros are not
slack — **the rider has ALREADY taken this byte saving**, by replacing 16 bytes of real code
lengths with a 16-zero run. And the span is positionally load-bearing four times over:
`residual_archive.py:128` (`if len(packed) < 142: raise`), `:152`
(`if len(result) != len(packed) + 40: raise`), `rr5_arith_basis.py:45–47`
(`PACKED_METADATA_BYTES = 40; BASIS_OFFSET = 102 + 40 = 142`), and `residual_archive.py:218`
(`_packed_portion = 102 + 40 + ...`). Removing it needs a receiver edit — forbidden, and it
would change the receiver behaviour digest and void move 54's inherited `t4_direct` leg.

**(b) Header flag bit 0x20 is `RC1_RESERVED_SEMANTIC_ADAPTIVE`**
(`residual_archive.py:788`), read at `:247` to select the adaptive semantic-plane restore path.
`SZ1_RESERVED_KNOWN_BITS = 0xFF` at `:790`, so every bit is a known bit and an unnamed bit in a
DIFFERENT receiver is no evidence of a dead bit here. Clearing a bit saves 0 bytes in any case:
the flag BYTE is present regardless of which bits are set.

**And the projection was 2× too big even as a hypothetical.** MEASURED by re-brotli-ing the
carrier body WITH and WITHOUT the 16 bytes under identical parameters: the body is
18,526 B against
18,510 B raw, and the
best compressed difference over a (quality 9–11) × (lgwin 20/22/24) sweep is
**8 B, not 16** —
**5.327e-06 S = 0.27 bar**. Brotli
already codes a 16-zero run in a handful of bits. **LAW: the rate term charges the COMPRESSED
extent of a span, never its raw extent. A byte projection taken on a decompressed section and
multiplied by the archive exchange rate over-states the lever by the section's own compression
ratio.** (No swept parameter reproduced the shipped 18,487 B, so the sweep is used only as a
matched-parameter DIFFERENCE, which is what the question needs.)

`verdict_scope: INSTANCE` — this closes the lever for THIS receiver family on THIS archive. It
does not claim the container has no unread bytes anywhere, and it does not close the mrs6 packet
receiver's own question: in mrs6 those 16 bytes really are unread, and mrs6's own format could
drop them if mrs6's offsets moved with them. That is the packet arm's call, not the frontier
line's. Receipt: `/Volumes/APDataStore/pact/ddm_pd8/DEAD_BYTES.json`.

**Cost of asking: about twenty minutes of code reading and one brotli sweep, before any wave ran.
Cost of not asking: a container change that the sealed receiver would have refused, discovered at
the cold parse-back after the whole ladder had been built on it.**

## 2. The binding

`pd4` binds move 52 in module globals evaluated at import, and every pd6/pd7 stage calls
`pd4.bind_move52`. This arm re-points those constants at move 54 PROCESS-LOCALLY
(`experiments/ddm_pd8_price_first_pass3.py::rebind_pd4_to_move54`) and leaves pd4's own
verification path in charge of the work: archive bytes and sha, argmax sha, field sha, raw bytes
and sha, the pointer's score re-derived from its three components, and a re-assertion THROUGH
pp1/sj1/pd1/pd2/pd3 rather than through this module's copy of the facts. No pd4, pd6, pd7 or
shipped-runtime source was edited.

| bound object | value |
|---|---|
| archive | `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`, 179,266 B |
| token field | pd7's `setprice/set_03.npz`, sha `e1ec64e528df6200…` — 66 tokens over 53 pairs vs move 53 |
| decoded token plane | `ed69d961fe0b98c3…` — reproduced INDEPENDENTLY here: the pricer's own `field_to_u8` of that npz hashes to the same value pd7's own set-price absorb recorded for the SHIPPED archive, so the field bound here IS the plane the receiver decodes |
| SegNet argmax | move 54's OWN cold decode (pd7 `seg_cand/argmax_n600.npy`), sha `6682b93da43c33b8…`, 12,127 flips |
| per-pair pose base | move 54's OWN cold decode at batch 8, sha `d46a971ef24ca41c…`, mean 4.099227789357173e−06 |
| score re-derived | 100·0.00010287 + √(10·4.1e−06) + 25·179,266/37,545,489 = **0.13605599532783202**, equal to the pointer's field to all digits |
| receiver behaviour digest | **`9f6e71680a13d859…`** on this arm's copy AND on pd7's sealed tree — the charter's `9f6e7168…` pin |
| inherited decode leg | `609addb9acb938a9…`, mode **`t4_direct`**, **1112.2 s** against the 1260 s limit |

**The leg is move 54's OWN `t4_direct` measurement, minted at its harvest** — not an inherited leg
re-inherited, so `[[an_inherited_decode_leg_blocks_its_own_successor…]]` does not bite here. That
memory's cure (mint `t4_direct` AT HARVEST) is why this arm has a leg to inherit at all.

Receipt: `/Volumes/APDataStore/pact/ddm_pd8/BASE_CONTROLS.json` (all three controls pass) and
`BIND54.json` (pd4's own receipts, with `--verify-raw` over the rebuilt 3,662,409,600 B decode).

## 3. What the lever needed, DERIVED before any credit existed

`HEADROOM.json`, written before the wave. A pair can credit at most its own d_pose; the gain is
`base[pair]·frac·pose_unit` and the cost of one token at a given price is `(bits/8)·S_per_byte`.

| recovered fraction of the pair's own d_pose | 3 bits/token | 6 bits/token | 8 bits/token |
|---|---|---|---|
| 5 % | 68 pairs pay, **-3.3115e-05 S** | 34 pairs, -2.1381e-05 | 24 pairs, -1.6523e-05 |
| 10 % | 105 pairs, **-8.7420e-05** | 68 pairs, -6.6230e-05 | 52 pairs, -5.6235e-05 |
| 20 % | 147 pairs, -2.0506e-04 | 105 pairs, -1.7484e-04 | 93 pairs, -1.5844e-04 |

**The horizon law now has a THIRD data point, and the shrink is ACCELERATING.** pd7 measured its
headroom 1.4–2.9 % thinner than pd6's, cell for cell. This arm measures **1.65–4.57 % thinner
than pd7's**, on the same nine cells:

| recovered / price | pd7 (move 53) | pd8 (move 54) | shift | pd6→pd7 shift |
|---|---:|---:|---:|---:|
| 5% @ 3 bits | 68 pairs, -3.3957e-05 | 68 pairs, -3.3115e-05 | **+2.48 %** | — |
| 5% @ 6 bits | 34 pairs, -2.2170e-05 | 34 pairs, -2.1381e-05 | **+3.56 %** | — |
| 5% @ 8 bits | 24 pairs, -1.7314e-05 | 24 pairs, -1.6523e-05 | **+4.57 %** | — |
| 10% @ 3 bits | 105 pairs, -8.9028e-05 | 105 pairs, -8.7420e-05 | **+1.81 %** | — |
| 10% @ 6 bits | 68 pairs, -6.7914e-05 | 68 pairs, -6.6230e-05 | **+2.48 %** | — |
| 10% @ 8 bits | 53 pairs, -5.7848e-05 | 52 pairs, -5.6235e-05 | **+2.79 %** | — |
| 20% @ 3 bits | 150 pairs, -2.0850e-04 | 147 pairs, -2.0506e-04 | **+1.65 %** | — |
| 20% @ 6 bits | 105 pairs, -1.7806e-04 | 105 pairs, -1.7484e-04 | **+1.81 %** | — |
| 20% @ 8 bits | 93 pairs, -1.6163e-04 | 93 pairs, -1.5844e-04 | **+1.98 %** | — |

The mechanism is unchanged and MEASURED: the base d_pose the successor must recover from fell
again, 4.136441e-06 → **4.099228e-06** (−0.90 %), only
partly offset by the pose unit rising 1.2957 → **1.3016** (+0.45 %).
**The price-first generator makes its own successor harder by exactly the amount it succeeds, and
the second derivative is against us**: at 5 % recovery and 8 bits/token the whole cell is now
-1.6523e-05 — below the −2e−05 admit bar on its own, as it already was for pd7,
and 4.57 % further below than pd7's.

The per-pair seg budget (`floor(base[pair]·pose_unit / seg_cell)`) over the 588 non-floor pairs:
**411 pairs can pay for no cells at all,
56 for one,
121 for two or more**
(pd7: 407 / 58 / 123).
The seg screen tightened too.

Operating point, all three EXPIRING at the next pointer move: S per archive byte
**6.658589531221714e-07**, S per seg cell **8.482724499051702e-07** (through move 54's own
instrument ratio), S per unit of one pair's d_pose **1.3015705930547656**.

## 4. The capture, and the two controls that make it credible

`ddm_pd8_price_first_pass3 capture` wraps `LaneMixer.coding` and `.end_frame` as pure observers
— each calls the shipped method and returns its value unchanged — and runs the pricer's own
control encode of move 54's field. Capture wall **1305.0 s**.

| control | outcome |
|---|---|
| **the pricer re-packs move 54's own archive** | **179,266 B sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`**, byte-identical to the live pointer, **twins identical in-process**, stream 119,014 B (ideal 119013.28), decoded plane sha `ed69d961fe0b98c3…` — **with the capture wrappers installed**. **F1 does not fire** |
| **the captured rows ARE the charged rows** | per frame, the sum of −log2 p over the TRUE symbols must equal the pricer's own `per_frame_bits`. MEASURED over all 600 frames: **max absolute 0.000394 bits, max relative 2.644e-07** against the charter's 1e−06 bar. **F2 does not fire** |
| **move 54's cold decode, re-run by this arm** | **3,662,409,600 B sha `ff43a9c97c72d0917ac4c2b856315648eddf3c0d3f69717eca132bfecf37a324`** — BIT-IDENTICAL to the decode pd7 reported for these bytes, **1127.2 s** on CPU at the contest thread count (4), `decoded_field_matches_admitted: true`. pd7 hashed this decode and then removed the payload under its retention cap; this arm rebuilt the same bytes rather than inheriting a number |
| the field IS the decoded plane | the pricer's `field_to_u8` of `field_move54.npz` hashes to `ed69d961fe0b98c3…`, the plane sha pd7's own set-price absorb recorded for the SHIPPED archive — two independent paths to the same bytes |
| **the batch-1 / n600 pose-base band, MEASURED over ALL 600 pairs** | max gap **5.464e−09** (0.13 % of the base mean), median **2.929e−10**, p99 3.414e−09; the wave's gate is 2× the non-floor max = **1.0929e−08**. batch-1 mean 4.099207e−06 against the n600 batch-8 mean 4.099228e−06 — the two instruments agree to six figures |

## 5. The prices expired again

pd7 retained its own capture rows on move 53's field, so the two are comparable cell for cell.

| head-8 overlap (the 8 cheapest (position, symbol) pairs) | median | mean | min | max |
|---|---:|---:|---:|---:|
| all 588 pairs | 8 | 7.55 | 3 | 8 |
| **the 53 RE-RENDERED pairs** | **6** | **5.49** | 3 | **7** |
| the 535 unchanged pairs | 8 | 7.75 | 5 | 8 |

**pd7's law replicates on a third field and a different edit set.** Not one of the 53 re-rendered
pairs keeps all eight of its cheapest changes (max 7), the median edited pair loses two of
eight, and most unchanged pairs lose none. 37 of 588 pairs lost their move-53 rank-0 cell
out of the move-54 head-1024 entirely (pd7 measured 64 of 588 across 101 edited pairs).

Deeper in the list the pool is stable and the edited/unchanged split nearly vanishes: head-1024
cell overlap is **956.4/1024** at the mean, **948.3** on edited pairs and
**957.1** on unchanged ones. **An edit re-ranks the TOP of a pair's price list
without moving the list.** Two arms, two different fields, the same shape — this is now a law
about the coder's response to a token edit, not a fact about one field.

### Where the cheap half lives — the same geometry a third time

| SegNet class | pd8 (move 54) | pd7 (move 53) | pd6 (move 52) | area share |
|---|---:|---:|---:|---:|
| **Road** | **90.97%** | 90.81 % | 90.99 % | 23.2 % |
| Undrivable | 6.41% | 6.55 % | 6.40 % | 49.5 % |
| Movable | 2.48% | 2.49 % | 2.46 % | 1.24 % |
| MyCar | 0.13% | 0.13 % | 0.14 % | 25.4 % |
| Lane | 0.013% | 0.014 % | 0.01 % | 0.59 % |

602,112 cells over 588 pairs. **The geometry of the cheap half is a property of the coder and
the scene, not of the field's edited tokens** — three fields, three near-identical splits.

### The pool, MEASURED (first-order, a RANKING never a charge)

| quantity | pd8 on move 54 | pd7 on move 53 |
|---|---:|---:|
| pairs with candidates | **588** of 600 | 588 |
| candidates emitted | **4,291** (3,233 single, 1,058 two-token) | 4,291 (3,233 / 1,058) |
| pairs whose cheapest change is NEGATIVE-priced | **213 of 588 (36.2%)** | 209 (35.5 %) |
| cheapest change per pair, first-order bits | median **0.792**, min **-21.565**, max **3.553** | 0.793 / −21.565 / 3.550 |
| two-token partner is a neighbour | **98.39%** | 98.20 % |
| held-fixed price-control pairs | **59** | 59 |

The twelve pairs without candidates are pp1's floor pairs, excluded by construction; they are the
twelve largest-d_pose pairs and hold 56.7 % of the n600 pose mass, so every number in this memo is
drawn on the remaining 43.3 % over 588 pairs.

## 6. THE CHARGE — the decay curve

Every candidate was charged before any credit was read. Each sheet carries one proposal per
pair, so one encode charges the whole sheet frame-locally; twins byte-identical inside every
encode, every encode output-lossless.

| sheet | family | tokens | delta bytes vs move 54 | **bits/token** | pd7 (move 53) | pd6 (move 52) | vs pd7 |
|---|---|---:|---:|---:|---:|---:|---:|
| **00** | single rank 0 | 588 | **+241** | **3.279** | 3.075 | 2.163 | **+6.63 %** |
| **01** | single rank 1 | 588 | **+457** | **6.218** | 5.864 | 5.837 | **+6.03 %** |
| **02** | single rank 2 | 588 | **+530** | **7.211** | 7.061 | 7.088 | **+2.12 %** |
| **03** | single rank 3 | 588 | **+541** | **7.361** | 7.415 | 7.497 | **-0.73 %** |
| **04** | single rank 4 | 588 | **+579** | **7.878** | 7.673 | 7.782 | **+2.67 %** |
| **05** | single rank 5 | 588 | **+579** | **7.878** | 7.878 | 7.973 | **-0.01 %** |
| **06** | two-token rank 6 | 1,117 | **+499** | **3.574** | 3.560 | 3.237 | **+0.39 %** |
| **07** | two-token rank 7 | 1,117 | **+566** | **4.054** | 4.047 | 3.724 | **+0.17 %** |

**THE DECAY CURVE, three passes, and it does NOT do what pd7's single delta suggested.**

| rung | pd6 (move 52) | pd7 (move 53) | pd8 (move 54) | pd6 -> pd7 | **pd7 -> pd8** |
|---|---:|---:|---:|---:|---:|
| **rank 0** | 2.163 | 3.075 | **3.279** | **+42.2 %** | **+6.63 %** |
| **rank 1** | 5.837 | 5.864 | **6.218** | +0.5 % | **+6.03 %** |
| rank 2 | 7.088 | 7.061 | 7.211 | -0.4 % | +2.12 % |
| rank 3 | 7.497 | 7.415 | 7.361 | -1.1 % | -0.73 % |
| two-token (6) | 3.237 | 3.560 | 3.574 | +10.0 % | +0.39 % |

pd7 read its own table as "the head of the price list is what a pass spends; the body is not",
because rank 0 moved +42 % while ranks 1-5 stayed inside 1.5 %. **On a third field that reading
is too narrow.** Here rank 0 moves +6.63 % and **rank 1 moves +6.03 % -- the same magnitude** --
while ranks 3 and 5 are flat or marginally cheaper. **The dearness did not only grow; it MOVED
DOWN THE LIST.** What accumulates across passes is the DEPTH of the affected region, not the top
price alone. The head-8 / head-1024 census in section 5 says the same thing from the other side:
the re-ranking is concentrated in the top 8 and the pool below it is stable.

The two deltas also do not scale with the tokens spent: move 52->53 spent 118 tokens and cost the
rank-0 sheet +42 %; move 53->54 spent 66 tokens (56 % as many) and cost it +6.6 % (16 % as much).
**Two deltas are not a rate**, and this memo does not fit one.

### The cheap half, CHARGED, and the 6/8 split

| threshold | proposals | pairs | fraction of charged | median bits/token |
|---|---:|---:|---:|---:|
| <= 3 | 950 | 386 | 22.1 % | 1.072 |
| <= 5 | 1,680 | 504 | 39.2 % | 2.602 |
| **<= 6 (pd6/pd7)** | 2,101 | 544 | 49.0 % | 3.305 |
| **<= 8 (this arm)** | 2,873 | 575 | 67.0 % | 4.431 |
| <= 12 | 3,843 | 585 | 89.6 % | 5.615 |

Over all 4,291 charged proposals the median is 6.088 bits/token (pd7: 6.025), the range is -22.615 to
24.887, and **316 proposals (7.4 %) carry a NEGATIVE real charge** -- the coder spends fewer bits
on the changed field than on move 54's own (pd7: 341, 7.9 %).

## 7. CREDIT AFTERWARDS

7 shards, **1,115 realized rows over 398 pairs**, no prefix stop, every one of the 4,291 proposals
screened.

| quantity | pd8 (move 54) | pd7 (move 53) | pd6 (move 52) |
|---|---|---|---|
| credit improves the pair | **475 of 1,115 (42.6 %)** -- inside the pre-registered 0.35-0.70 | 44.7 % | 47.5 % |
| best single credit | **-2.9878e-06** d_pose | -8.509e-06 | -4.064e-06 |
| seg cells on realized rows | **29 repaid, 627 neutral, 253 cost one, 206 cost two** | 32 / 645 / 267 / 206 | 51 / 687 / 272 / 193 |
| **pairs with ANY positive resolved credit** | **248** -- F4's bar is 30 | 264 | 279 |
| **pairs whose best row pays at its OWN real price** | **44** | 72 | 128 |
| modelled net at the sheet price | **-2.8753e-05 S** | -5.172e-05 | -1.246e-04 |

**The paying population is halving every pass: 128 -> 72 -> 44.** The modelled net follows it:
-1.246e-04 -> -5.172e-05 -> -2.875e-05. Neither the generator nor the credit shape changed; what
changed is that the cheapest rung costs more and the pose left to recover is smaller.

**The carrier re-solve is doing the work, and this is the control that says so.** Before the
re-solve the composed field's pose is **8.4x base** (3.4461e-05 against 4.0992e-06); the per-pair re-solve
recovers **100.1 %** of that damage. Every credit reported here is what survives AFTER the
re-solve, never the stale number.

## 8. THE SET PRICE

| iteration | pairs / tokens | ledger bits | real frame-local bits | **residual bias** | measured pairs used | encoder archive |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44 / 51 | -70.06 | +34.38 | **-149.08 %** | 0 | 179,271 (+5) |
| 1 | 28 / 32 | -83.30 | -73.65 | **-11.58 %** | 25 | 179,255 (-11) |
| 2 | 26 / 29 | -83.71 | -83.83 | **+0.15 %** | 26 | 179,254 (-12) |
| 3 | 26 / 29 | -83.83 | -- | -- | 26 | -- |

**The ledger's over-credit is growing pass over pass, and on this field it FLIPS SIGN.** pd6
measured -23.2 %% at iteration 0, pd7 -59.6 %%, and this arm **-149.08 %%**: the ledger predicted a
70.06-bit SAVING and the real 600-frame encode charged **+34.38 bits**. The first-order price is
a worse predictor of the real frame-local charge every pass, which is what you would expect once
the genuinely cheap cells are spent -- what is left prices cheap to the first-order model and
dear to the adaptive coder.

**That is exactly why pd7's `setabsorb` law is load-bearing rather than hygiene.** The absorb
collapsed the bias to -11.58 %% in one iteration and to **+0.15 %%** in two, and it carried the set
from **+5 B to -12 B** on the encoder member. Iteration 3 reproduces iteration 2's set **with
`measured_pairs_used: 26`** -- a genuine fixed point, not pd7's no-op. **Had this arm repeated
pd7's defect it would have closed on iteration 0's 44-pair set, which the ladder below measures
at 0.912 bars -- BELOW the admit bar.** The defect pd7 recorded rather than hid is the reason
this pass has a candidate at all. F6's bar is 10 %% and the final residual is 0.15 %%: **F6 does
not fire.**

## 9. THE COMPOSITION

A 600-pair overlay rendered from the 44-pair `set_00` field, a carrier re-solve over all 600
pairs starting from the SHIPPED codes, and an n600 pose leg on the result.

| leg | value |
|---|---:|
| per-pair sum of credits | **-1.956474e-05** |
| **composed, re-verified (n600 mean shift x 600)** | **-1.956474e-05** |
| **realized fraction** | **1.0000** |
| **spill onto the 556 unkept pairs** | **0.000000e+00** |
| stale pose (candidate renders, shipped carrier) | 3.4461e-05 = **8.4x base** |
| carrier coordinates changed | **155** over 26 spliced pairs |

### pd7's LAW fires again -- and a free `array_equal` caught it before any close

pd7's correction to pd6 was: *a composed overlay is valid only for the pairs whose token plane
in the SHIPPED field equals the plane the overlay was rendered from, and a set-price iteration
RE-RANKS as well as drops, so compare the planes and re-render when a KEPT pair's edit changed.*

This arm built that comparison as a tool (`plane_diff.py`) and **validated it against pd7's own
answer before trusting it here**: on pd7's retained `set_00` and `set_03` fields it reports
exactly the two pairs pd7 found by hand (238 and 419), with 53 kept and 19 dropped. Two defects
in the tool were caught by that self-test -- reading the npz keys LEXICALLY (which compares one
frame, not a field, and reports every stale overlay as valid) and counting a DROPPED pair as
stale (it is not: it reverts to the base plane, which is what the ladder already prices it at).
**A new check is not evidence until it reproduces a known answer.**

Applied here, before any close:

| overlay field vs shipped field | kept | differ | **STALE (kept, different edit)** | dropped |
|---|---:|---:|---:|---:|
| set_00 vs set_01 | 28 | 19 | **3: 409, 487, 597** | 16 |
| set_00 vs set_02 | 26 | 20 | **2: 409, 487** | 18 |
| set_00 vs set_03 | 26 | 20 | **2: 409, 487** | 18 |

So the overlay was RE-RENDERED from the winner's own field, the carrier re-solved over all 600
pairs against that overlay, and the pose re-measured. The corrected composition is
**4.075124862005007e-06** where the stale one read **4.074230995148548e-06** -- the stale number
was **optimistic by 8.939e-10** in the n600 mean (pd7's was optimistic by 2.12e-09). Pair by
pair, the `set_00` edit credited 409 at 5.675e-08 and 487 at 8.673e-08; the SHIPPED edit reaches
only 3.458e-07 and 3.340e-07. Closing on the stale composition would have over-claimed both.

## 10. THE CANDIDATE — every leg on the shipped bytes' own decode

| leg | value | how |
|---|---:|---|
| rate | **-7.324448e-06** | **-11 B EXACT** -- the closed archive, re-solved carrier included |
| seg | **+1.696545e-06** | **2 cells COST** (12,129 flips against move 54's 12,127), MEASURED by `step0 --raw` on the candidate's own `0.raw`; carried to T4 by move 54's own same-instrument ratio 1.0006628989857342 |
| pose | **-1.945377e-05** | **4.075124862005007e-06** MEASURED by `up2.measure_pose` at batch 8 on the candidate's own `0.raw` with its own carrier |
| **S projected** | **0.13603091365293757** | 100 * 0.00010288696544900 + sqrt(10 * 4.075124862005007e-06) + 25 * 179,255 / 37,545,489 |
| **net vs move 54** | **-2.508167e-05** | **1.254 bars** * **1.082x the 34.8 B container-break sd** |

**Two of the three legs are negative and all three are measured on the decode.** The candidate
is 11 bytes SMALLER than move 54 and lowers pose by 0.61 %; it costs 2 SegNet cells,
which the rate and pose legs outweigh by 15.8x. It sits inside the pre-registered band
[-6e-05, -1e-05] and clears the -2e-05 admit bar at 1.254 bars.

### The SIZE ladder, on real CLOSED archives -- and the row the ledger would have chosen LOSES

| candidate | pairs / tokens | **exact closed archive** | rate | seg | pose | **net** | **bars** |
|---|---:|---|---:|---:|---:|---:|---:|
| set 00 | 44 / 51 | 179,274 B (+8) | 5.327e-06 | +2.545e-06 | -2.612e-05 | -1.824707e-05 | 0.912 |
| set 01 | 28 / 32 | 179,257 B (-9) | -5.993e-06 | +3.393e-06 | -2.160e-05 | -2.419873e-05 | 1.210 |
| **set 02** | 26 / 29 | **179,256 B (-10)** | -6.659e-06 | +1.697e-06 | -2.015e-05 | **-2.511597e-05** | **1.256** |
| set 03 | 26 / 29 | 179,256 B (-10) | -6.659e-06 | +1.697e-06 | -2.015e-05 | -2.511597e-05 | 1.256 |

set 00 is the converged ledger's iteration-0 favourite and it is the WORST row: 44 pairs buy the
largest pose gain (-2.61e-05) and pay +8 bytes for it, netting 0.912 bars -- **below the admit
bar**. set 02 keeps 26 of those pairs, gives up 23 % of the pose gain, and buys -10 bytes and two
fewer SegNet cells with it. **The ladder on closed archives chose the row, exactly as pd6's
handoff said it must** -- and here the gap between the ledger's choice and the ladder's is the
difference between a candidate and no candidate.

(The winning rung was then re-closed from the RE-RENDERED composition, which is the row reported
above: 179,255 B -- one byte smaller than the stale close, because the corrected carrier moves
155 coordinates rather than the stale one's.)

### Every control on the shipped bytes

| control | outcome |
|---|---|
| **cold n600 public parse-back** | **1052.1 s** CPU at the contest thread count (4), raw 3,662,409,600 B sha `9116e6c3486377e74cc92192cedbfaabe0714cfd7541fae67edbe692620bebcd`, `decoded_field_matches_admitted: true` |
| **the decode reproduces the OVERLAY's own odd frame, pair by pair** | **26 of 26 IDENTICAL, max abs difference = 0 grey levels** |
| **d_seg on the DECODE** | **12,129 flips against move 54's 12,127 -- 2 cells COST**, exactly what the per-pair census predicted (`d_cells_total = +2`); 19 argmax cells moved in all |
| **d_pose on the DECODE** | **4.075124862005007e-06 -- IDENTICAL TO ALL DIGITS to the admission**; max per-pair difference 0.0, ratio 1.0. **F7 does not fire** |
| **twins, cross-process, encoder member** | primary and repeat both **179,254 B sha `0e4d3324bdcee219...`**, member sha `39221ad70802004c...`, each built by its own natively-compiled rc64 backend |
| **twins, cross-process, CLOSED archive** | both **179,255 B sha `ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a`**, staged in separate directories |
| (bonus) | set 03's set-price iteration reproduced set 02's field, so its INDEPENDENT encode is a third process agreeing on the same member |
| literal census (rule 118) | **CLEAR** -- the only files differing from move 54 are `archive.zip`, the two archive pins inside `inflate.py`, and the derived `MANIFEST.sha256` that restates them |
| **the RLC1 rider is present** | both the control and the candidate tail suffixes begin `RLC1` (`524c 4331 01fe 1aec`) and both tail sections are exactly **96 B** longer than their suffix -- no archive here is the 64-byte-optimistic undecodable kind |
| receiver behaviour digest | **`9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890`** on the candidate AND on move 54 -- the charter's `9f6e7168...` pin |
| manifest (Catalog #420) | **51 rows**, all hashes re-verified from OUTSIDE the tree, all runtime dependencies listed, derived listing present and excluded from the behaviour digest |

## 11. Falsifiers, as pre-registered

| falsifier | outcome |
|---|---|
| F1 the pricer does not re-pack move 54's own archive | **does not fire** -- 179,266 B sha `5c6bf403...`, twins identical, **with the capture wrappers installed** |
| F2 the captured rows are not the charged rows | **does not fire** -- max relative **2.644e-07** against the 1e-06 bar (max absolute 0.000394 bits), over all 600 frames |
| F3 the per-proposal price does not resolve above its noise | **reported, not decisive** -- the SET re-encode, not the ledger, is what every byte claim here rests on, and the ledger's own bias (-149 % at iteration 0) is the sharper statement about first-order price resolution on this field |
| F4 fewer than 30 pairs carry a cheap proposal with positive resolved-pose credit | **does not fire** -- **248** pairs do, and **44** pay at their own real price |
| F5 the admitted set nets worse than -1e-05 S | **does not fire** -- **-2.5081674894456008e-05**, 1.254 bars, on legs measured on the shipped bytes' own decode. It also clears the tighter -2e-05 admit bar |
| F6 the SET re-price differs from the ledger by >10 % after iteration | **does not fire** -- **-149.08 % -> -11.58 % -> +0.15 %** over three real absorbed iterations |
| F7 the projection's pose leg and the decode's differ by more than 1.5x | **does not fire** -- **ratio 1.0, identical to all digits**, max per-pair difference 0.0 |

**No falsifier fires.** The band was [-6e-05, -1e-05] and the measured net is -2.51e-05.

## 12. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires. No Modal call, no
   authorization, no completion, no packet, no pointer write.
2. **The distortion legs are `[macOS-CPU advisory]`** -- frozen CPU-torch PoseNet and SegNet
   against DALI-lineage GT. The seg leg is the jg1 instrument's count carried to T4 by move 54's
   own same-instrument ratio 1.0006628989857342, the same carry move 54's own leg uses, not an
   independent T4 measurement.
3. **The decode wall clock is INHERITED, not re-measured on this candidate.** It is move 54's own
   `t4_direct` leg (1,112.2 s against the 1,260 s limit, sidecar sha `609addb9acb938a9...`),
   inherited on the strength of an identical receiver behaviour digest. The candidate's
   payload-dependent time was not measured on T4. Its CPU decode did run, in 1,052.1 s.
4. **pp1's twelve floor pairs stay excluded** on pp1's measurement. They are the twelve
   largest-d_pose pairs and hold 56.7 % of the n600 pose mass, so every number here is drawn on
   the remaining 43.3 % over 588 pairs.
5. **The first-order price is a RANKING, never a charge.** Every candidate was charged by a real
   600-frame sheet encode before any credit was read, and the selected SET was re-encoded as one
   field before any byte claim. On this field that distinction is no longer a nicety: the ledger
   and the real encode disagree in SIGN at iteration 0.
6. **The bits/token numbers are THIS pool at THIS edit shape on THIS field**, measured once. The
   three-pass rank-0 series (2.163 / 3.075 / 3.279) is three points, not a rate, and this memo
   does not fit one to them.
7. **The 16 "dead" carrier bytes were never dead.** The verdict in section 1 is instance-scoped:
   it settles the lever for the SEALED receiver on THIS archive by reading the code that consumes
   the span. It does not claim the container has no unread bytes anywhere, and it does not close
   the mrs6 packet receiver's own version of the question.
8. **The `plane_diff` staleness check is a set identity, not a re-decode.** What is MEASURED is
   which pairs' planes differ; the cure (re-render, re-solve, re-measure) was then run and the
   decode reproduced all 26 kept pairs exactly.

## 13. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd8/`**. **Vertigo was never opened for writing.**
Retained **0.95 GiB** against the charter's 2 GiB cap -- 2,322 files,
896,170,206 B hashed in `RETENTION_MANIFEST.json`. APDataStore free space MEASURED at every
heavy step: **24 GiB at start**, 20 GiB through the capture and the base decode, 18 GiB through
the sheets and the wave, 16 GiB through the composition, **12 GiB at the tightest point** (the
candidate's own cold decode), **22 GiB after the certified prune**.

Every bulk payload removed was hashed first, with the exact command that rebuilds it, in
`BULK_CERTIFICATE.json` -- the record lands on disk BEFORE the bytes leave it:

| payload | bytes | sha256 |
|---|---:|---|
| `parseback_base/0.raw` | 3,662,409,600 | `ff43a9c97c72d091...` |
| `parseback_candidate/0.raw` | 3,662,409,600 | `9116e6c3486377e7...` |
| `pose/overlay` (directory) | 1,831,211,386 | 2 files, each hashed |

Plus the pricer's 12 per-field u8 planes and encoder states (1,460,465,566 B over two certified
prunes, `PRUNE_SHEETS.json` and `PRUNE_SETS.json`), each naming its source npz and its own
recorded sha in the pricer's `INPUTS.json`. **Nothing was deleted before its identity was
recorded, and the prune ran only after the LAST rlc1 call** (pd7 defect 2).

| path | what |
|---|---|
| `PREREGISTRATION.json` * `HEADROOM.json` * `BASEBAND.json` * `BASE_CONTROLS.json` | the prediction and its falsifiers, the DERIVED headroom, the wave's own MEASURED gate, and the three binding controls -- all written before any credit |
| `DEAD_BYTES.json` | the charter's container question, settled by reading the sealed receiver and by a matched-parameter brotli difference |
| `capture/frames/frame_????.npz` * `capture/CAPTURE.json` | the coder's own price rows, 600 frames, and the byte-identity control |
| `plan/` * `CHEAP_GEOMETRY.json` * `DISJOINTNESS.json` * `PRICE_EXPIRY.json` * `PRIOR_OVERLAP.json` * `PRIOR_REFUSED.json` | the cheap half, where it lives, and the censuses that say what is new about it |
| `sheets/` * `SHEET_CHARGES.json` * `CHEAP_HALF.json` | the eight sheets, the 4,291 real-encode charges, and the decay curve |
| `search/wave/realized_*.jsonl` * `screen_*.jsonl` * `CREDIT.json` | **every realized row and every screened proposal** -- 1,115 realized, 4,291 screened, winners and losers |
| `setprice/STATE.json` * `set_0?.npz` * `LADDER.json` * `PICK_WINNER.json` * `close/*/CLOSE.json` * `close/*/candidate_archive.zip` | the four set-price iterations with their absorbed residuals, the size ladder, the winner pick with its staleness verdict, and every closed archive including both twins |
| `pose/pose_stale.npy` * `pose_resolved.npy` * `pose_resolved_c.npy` * `POSE_ON_DECODE_set02c.npy` * `refine/` * `refinec/` | the pose vectors, including the decode's own, and both carrier re-solves |
| `parseback_*/PARSEBACK_RESULT.json` * `RENDER_AGREEMENT_set02c.json` * `seg_candidate/argmax_n600.npy` * `PUBLIC_SMOKE.json` * `LITERAL_CENSUS.json` * `seal_inputs/` | the decode receipts, the render-agreement census, the decode's own SegNet argmax, the smokes, the rule-118 census and the seal inputs |
| `BULK_CERTIFICATE.json` * `RETENTION_MANIFEST.json` * `PRUNE_*.json` | the custody record |

Producer: `experiments/ddm_pd8_price_first_pass3.py` -- a thin process-local rebinding of pd4's
move-52 constants onto move 54, plus a MEASURED base-band gate, a move-54 headroom derivation, a
`run` passthrough and a `price-merge` that reads this arm's own sheet encodes. Every stage is
pd6's (through pd7), imported and called unchanged; downstream, pd4's merge/carry/assemble,
pd5's setprice/setabsorb, `ddm_sj1_rlc1_price` and `ddm_sj1_joint_admission` are all unchanged.
Nothing under `ddm_pd1`-`ddm_pd7`, `ddm_sj1`, `ddm_jr*`, `ddm_psa*`, `ddm_mrs*` or
`/Volumes/VertigoDataTier/` was written.

## 14. What this hands the next arm

1. **The dead-byte lever is closed, and the way it closed is the transferable part.** The span
   is READ -- as a SENTINEL whose all-zero value means "regenerate the Huffman lengths from the
   decoded histogram", which is the rider having ALREADY taken the saving. And the projection was
   2x too big regardless. **LAW: the rate term charges the COMPRESSED extent of a span, never its
   raw extent. A byte projection taken on a decompressed section and multiplied by the archive
   exchange rate over-states the lever by that section's own compression ratio.**
2. **The dearness moves DOWN the price list, it does not just grow at the top.** pd7 measured
   rank 0 at +42 % with ranks 1-5 inside 1.5 % and read that as "the head is what a pass spends."
   On a third field rank 0 moves +6.6 % and rank 1 moves +6.0 %. What accumulates is the DEPTH of
   the affected region. A fourth pass should expect rank 2 to move next.
3. **The first-order ledger is losing its sign, and the absorb is now load-bearing.** The
   iteration-0 residual bias went -23.2 % (pd6) -> -59.6 % (pd7) -> **-149.1 %** here: the ledger
   predicted a 70-bit saving and the encode charged +34 bits. Without pd5's `setabsorb` between
   iterations this arm would have shipped iteration 0's 44-pair set, which the ladder measures at
   **0.912 bars -- below the admit bar**. pd7's recorded defect is the reason this pass landed.
4. **The paying population is halving per pass: 128 -> 72 -> 44**, and the modelled net with it
   (-1.25e-04 -> -5.17e-05 -> -2.88e-05). The realized net follows the same curve
   (-1.04e-04 -> -4.59e-05 -> **-2.51e-05**). A fourth pass on this generator should be
   pre-registered against roughly -1.2e-05, which is BELOW the 2e-05 admit bar: **pass 4 is the
   one that has to change something, not just run again.**
5. **The seg screen is still the binding gate, and it tightened.** 411 of 588 pairs can pay for
   no SegNet cell at all (pd7: 407), and the wave refused most of its proposals on that budget.
   An actuator that paid the seg debt elsewhere (psa1/psa2's pair-selective renderer bias is the
   live candidate) is still the lever that would widen this pool without touching the price.
6. **An identical-outcome comparison is only evidence when both sides REACHED the gate the claim
   names.** This arm's first public-entrypoint smoke reported candidate and frontier behaving
   IDENTICALLY -- and both had exited in a third of a second on `usage: inflate.sh <archive-dir>
   <output-dir> <file-list>`, because the probe called the entrypoint with no arguments. A second
   version reached a different wrong place: both raised in five seconds on a missing
   `CPR1_RC64_LIBRARY`, because the direct `f26_inflate` probe did not build the two native
   libraries `inflate.sh` builds. Both versions would have been reported as "identical behaviour."
   The cure is in `public_smoke.py`: it STAGES ITS OWN INPUTS (a deleted scratch directory must
   not be able to turn the smoke into a usage message) and it REFUSES to write a receipt whose
   outcomes are not `REACHED_CUDA_GATE` and `REACHED_TOKEN_DECODE`. The seal caught a third
   version on its own -- a probe timed out AT the declared bound measures bound + spawn overhead
   and refuses; the probe timeout must sit strictly under the declared bound.
7. **What is still unmeasured:** whether a fifth set-price iteration would move the ladder (the
   residual is +0.15 %, so the estimates have converged, and iteration 3 is a true fixed point --
   this is the first pass where the ladder stopped moving); and whether the 2 SegNet cells this
   candidate costs could be repaid by a different rung at the same byte count.

## 15. verdict_scope

**verdict_scope: n/a for the pass** -- no negative verdict is drawn about the price-first family.
Every pre-registered falsifier was evaluated and none fired; the candidate is sealed and MAIN fires.

**verdict_scope: INSTANCE, for the dead-byte lever.** What is MEASURED is that the SEALED
receiver READS carrier offsets 123-138 (`runtime/residual_archive.py:134`, named at
`runtime/rr5_arith_basis.py:44`, consumed as a regeneration sentinel at `:510-519`) and READS
header flag bit 0x20 (`residual_archive.py:247`, `RC1_RESERVED_SEMANTIC_ADAPTIVE`), on move 54's
own archive. This closes the lever for this receiver family on this archive. It does NOT close
the same question for `submissions/mrs6`, where those 16 bytes really are unread by that
receiver's own `decode_carrier`, and it does not claim the container has no unread bytes
elsewhere.

**verdict_scope: FORMULATION, for the price-decay reading.** pd7's "the head of the price list is
what a pass spends; the body is not" is REFINED, not overturned: on a third field the dearness
reaches rank 1 at the same magnitude as rank 0. The head-8 / head-1024 census still holds
exactly (re-ranking concentrated in the top 8, pool stable below), so what moves is the depth of
the re-pricing, not the locality of the re-ranking.

**Nothing here closes or reopens pd5's multi-token formulation.** This arm changed the FIELD, not
the run length; its admitted set is 23 single-token and 3 two-token proposals.

<!-- # FORMALIZATION_PENDING: a measurement pass and (if it nets) a byte-closed candidate; the
score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) +
25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm -- MAIN fires):
**S 0.13605599532783202 @ 179,266 B [contest-CUDA T4 n600]** (move 54).
