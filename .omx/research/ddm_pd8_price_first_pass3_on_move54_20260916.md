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

<!-- PD8_SECTION_2 -->

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

<!-- PD8_SECTION_4 -->

## 5. The prices expired again

<!-- PD8_SECTION_5 -->

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
