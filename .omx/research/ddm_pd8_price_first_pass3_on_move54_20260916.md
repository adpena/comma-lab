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

<!-- PD8_SECTION_6 -->

## 7. CREDIT AFTERWARDS

<!-- PD8_SECTION_7 -->

## 8. THE SET PRICE

<!-- PD8_SECTION_8 -->

## 9. THE COMPOSITION

<!-- PD8_SECTION_9 -->

## 10. THE CANDIDATE — every leg on the shipped bytes' own decode

<!-- PD8_SECTION_10 -->

## 11. Falsifiers, as pre-registered

<!-- PD8_SECTION_11 -->

## 12. What this does NOT claim

<!-- PD8_SECTION_12 -->

## 13. Custody

<!-- PD8_SECTION_13 -->

## 14. What this hands the next arm

<!-- PD8_SECTION_14 -->

## 15. verdict_scope

<!-- PD8_SECTION_15 -->

<!-- # FORMALIZATION_PENDING: a measurement pass and (if it nets) a byte-closed candidate; the
score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) +
25*B/37,545,489. -->
