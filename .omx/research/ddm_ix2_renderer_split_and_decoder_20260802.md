# ddm_ix2 — the bytes were never inside the members; they were BETWEEN them

**UTC** 2026-08-03 · **arm** `ddm_ix2_renderer_split_and_decoder` · **axis** `[byte-closed rate, scorer-free]`
`score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**

**Vehicle:** `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_dc1_fold_archive.zip`,
**360,309 B** (`dc1_fold`, our own-vehicle best, S = 0.8983775).
**Denominator:** gap to the PR130 demonstrated floor **0.7262358**; 1% of gap = **10,907 B**
(`tac.canonical_equations.gap_decomposition_against_floor_20260802`; PR130 = 191,052 B per `ddm_na1`).
**Code:** `src/tac/optimization/ddm_ix2_archive_container.py` + 53 tests (commit `241f0f4657`).

---

## The answer first

1. **`renderer.sec` is REFUTED as a structural cut.** `ddm_ix1` flagged it as *"potentially a larger
   structural cut than everything above."* It is not. Field-split: 3,341 B = 56 frame + 8 header +
   **2,781 B lottery mask** + **492 B f16 gain/bias**. The mask is 22,248 bits at density **0.496584**,
   whose order-0 entropy is **2,780.9 B** and whose combinatorial floor is **2,780.0 B** — we ship
   **2,781 B, one byte above its own floor.** It is a maximum-entropy binary mask and there is nothing
   in it. Total realizable across the whole member: **−75 B.** The generic half (the lottery bank) was
   already free: it is regenerated from `selector.lotto_seed` and never shipped.
2. **The bytes are BETWEEN the members, not inside them.** Six independently framed ZIP members pay
   **686 B** of headers, central-directory entries and *filenames counted twice*, plus six independent
   coder warm-ups. **One member = 108 B**, and one shared coder over the small group saves a further
   **156 B**. That is the firmware move the charter named, and it is the only rung of this arm that was
   not already taken.
3. **The adaptive `cell+prev` coder — `ddm_ix1`'s own rank-3, its largest open number — is REFUTED.**
   Its "310–338 KB plausible realizable band" was an ORACLE. Realized sequentially (which is what a
   coder actually does): cell-id **352,862 B**, cell+prev **371,490 B**, same-flag+value **370,740 B**
   — **12.9 to 31.5 KB WORSE** than the layout+brotli 339,970 B already banked. Mechanism, measured:
   62.6% of the residual is zero and 60.9% is same-as-previous-pair, so in cell-major order brotli
   **LZ-copies runs** at a fraction of a bit, while a per-cell histogram must pay ~1.5 bits/symbol no
   matter how well it is fitted. The structure is MATCH structure, not distribution structure.
4. **BYTE-CLOSED RESULT: −6,604 B → ΔS −0.0043973 → −0.6055% of the gap. S 0.8983775 → 0.8939802.**
   Lossless: token codes, renderer mask, f16 table, `selector.sec` and `pose_warp.stp` all verified
   **bit-identical from the new archive bytes alone**, so `d_seg` and `d_pose` are invariant by
   construction. Receiver-consumption bijection closes exactly; rebuild is byte-deterministic.
5. **Round-1 self-review found a rule-118 defect in the inherited plan and a second false custody
   field.** `ddm_ix1`'s 27 B minimal manifest would have migrated `rs_beta_mags` — a **13-entry
   magnitude codebook fitted to this clip** — into `inflate.py` as if generic. That is hide-data-in-code.
   It is counted here. Separately, auditing all six manifest hashes found **`tr1_packet_sha256` is also
   wrong**, not just `tokens_sha256` (`ddm_gd3`'s finding). §5.

---

## 1. THE MATRIX — every member × every technique, including the zeros and the not-runs

Six members, 360,309 B. A blank cell would read as "assessed and clean"; there are none — every cell is
`win` / `0` / `n/a` / `not run`.

| technique ↓ / member → | `tokens.dr7t` 346,478 | `renderer.sec` 3,341 | `pose_warp.stp` 8,654 | `selector.sec` 535 | `manifest.json` 1,450 | `pose_stub.sec` 83 |
|---|---|---|---|---|---|---|
| **coder race** (deflate/brotli/lzma/stored) | **0** — brotli +5 B WORSE than stored (ix1) | **0** — brotli +4 B worse than stored on the raw payload | **0** — kl1 brotli already best | −221 B (deflate→brotli, ix1) | folded into migration | folded into migration |
| **SoA vs AoS** | **WIN −6,508 B** (cell-major, ix1) + 3 more perms raced here, all lose | n/a — two heterogeneous fields, not an array | **0** — kl1 dim-major byte-plane already beats AoS by 642 B (tp) / 157 B (ab) | n/a | n/a | n/a |
| **sub-byte / nibble packing** | **WIN**, −1.0…−2.0% at every layout (ix1) | n/a — mask is already 1 bit/symbol | n/a — f16 is the width | n/a | n/a | n/a |
| **bit-plane transposition** | **LOSES +20%** (ix1) | not run — 1-bit alphabet, no planes to split | **already shipped** (kl1 byte-plane) | not run | n/a | n/a |
| **field split** (heterogeneous fields → own coder) | n/a — homogeneous | **WIN −28 B** (mask stored, f16 byte-plane+lzma 492→464) | **0** — already split into 5 sections | n/a | n/a | n/a |
| **delta + zigzag + varint** | **LOSES** (temporal delta mod-16: 366,141 vs 339,970) | not run | **LOSES** (tp 6,921 vs 6,349; ab 1,972 vs 1,822) | not run | n/a | n/a |
| **fixed-point Q instead of float** | n/a — already int4 | **not run** — lossy, needs the scorer slot | **not run** — the f16 mantissa-width solve is LOSSY, §6 | n/a | n/a | n/a |
| **base-N packing** | not run — 16 states packs exactly into a nibble, no slack | not run | not run | not run | n/a | n/a |
| **run-length / same-flag** | **LOSES** (353,512 vs 339,970) | not run | not run | n/a | n/a | n/a |
| **adaptive context coder** | **LOSES** −12.9…−31.5 KB, §3 | not run | not run | not run | n/a | n/a |
| **combinatorial index** (colex) | n/a | **0** — mask is 1 B above `log2 C(22248,11048)` | −8 B on `sel_coded` (ix1) | n/a | n/a | n/a |
| **generic→`inflate.py` migration** | **0** — the payload | **0** — the bank is *already* free from `lotto_seed` | **0** | **0** — the decode program is video-derived | **−1,418 B**, §5 | **−83 B** (a constant string) |
| **specialized straight-line decoder** | not run — no byte effect; the LOC cap is deleted and there is no time term, so codegen buys nothing on the rate axis | same | same | same | same | same |

**CROSS-MEMBER** (the row that does not fit in the table, and the one that paid):

| rung | measured |
|---|---:|
| per-member ZIP framing, 6 members (30 B local + 46 B central + **filename twice** + 22 B EOCD) | **686 B** |
| same, consolidated to one member named `0.bin` | **108 B** |
| five small members coded independently (each at its own best) | 13,157 B |
| the same five under **one shared coder state** | **13,001 B** |
| **cross-member total** | **−738 B** |

`tokens` is deliberately kept OUT of the shared stream: at 96% of the archive and already at entropy it
would only dilute the model. That is CLAUDE.md L23's split-stream lesson read from the other direction —
**split when the distributions differ, share when the payloads are small.**

---

## 2. `renderer.sec` — the field split, and why the member is closed

```
3,341 B  =  56 B brotli frame (8 magic + 8 ulen + 8 clen + 32 sha256)
         +   8 B  >II (mask_count, float_count)
         + 2,781 B  packbits(22,248 binary lottery-mask bits)
         +   492 B  246 × f16 big-endian per-channel gain/bias
```

| field | shipped | floor / best | verdict |
|---|---:|---:|---|
| lottery mask, 22,248 bits @ density 0.496584 | 2,781 | order-0 **2,780.9** · colex **2,780.0** | **AT FLOOR** — max-entropy, keep stored |
| f16 gain/bias table | 492 | byte-plane + raw-LZMA1 **464** | −28 B |
| brotli frame header | 56 | 0 (container supplies length; the 32 B sha is unread, §5) | −56 B |
| the weight bank itself | **0** | — | **already free** — regenerated from `selector.lotto_seed` |

Running one coder over `mask ++ floats` is worse than either separately: joint brotli is **3,285 B vs
3,281 B stored** (+4). The incompressible field was poisoning the compressible one — which is the same
"put unlike things in different streams" argument as SoA, one level up. **`IX2REN01` = 3,266 B.**

**Answer to ix1's open question:** `regenerate_bank_and_apply_mask_mods` does mean bank-from-seed, the
generic half **is** already free, and the video-derived half is a maximum-entropy mask. There is no
structural cut here. **Item closed, −75 B, 0.007% of gap.**

---

## 3. `pose_warp.stp` — never assessed before; assessed now; ZERO

```
8,654 B  =  32 B framing (8 magic + 4 n_pairs + 5 × u32 lengths)
         + 6,365 tp_member (kl1 byte-plane f16 (600,6) pose)
         + 1,838 ab_member (kl1 byte-plane f16 (600,2) exposure)
         +   189 st_coded  +  151 beta_coded  +  79 sel_coded
```

Five layouts raced against the shipped `kl1` byte-plane, on both f16 members:

| layout | `tp_member` | `ab_member` |
|---|---:|---:|
| **shipped: byte-plane dim-major + brotli** | **6,349** | **1,822** |
| AoS f16 raw | 6,991 | 1,979 |
| byte-plane + temporal delta (mod 256) | 6,580 | 1,919 |
| byte-plane pair-major | 6,756 | 1,854 |
| int16 delta + zigzag + varint | 6,921 | 1,972 |

**The shipped layout wins outright on both.** `kl1` already took this rung. Remaining headroom is the
~35 B of index work `ddm_ix1` measured (colex on `sel_coded` −8, order-0 arithmetic on `beta_coded` −27)
plus ~16 B of framing. **This member is an honest ZERO** and the next arm should not re-race it.

---

## 4. The adaptive coder — ix1's largest open number, REFUTED

Sequential (KT / Dirichlet-α) code length is exactly what an arithmetic coder achieves, and it charges
the model cost inline instead of excluding it the way an oracle does.

| model | bytes | bits/sym |
|---|---:|---:|
| adaptive H0 | 504,308 | 2.1888 |
| adaptive \| cell-id, α = 0.5 | 359,573 | 1.5606 |
| **adaptive \| cell-id, α = 0.10 (best)** | **352,862** | 1.5315 |
| adaptive \| cell+prev, α = 0.20 (best) | 371,490 | 1.6124 |
| adaptive \| same-flag(cell) then value(cell) | 370,740 | 1.6091 |
| **cell-major nibble + brotli (BANKED)** | **339,970** | **1.4756** |

Additional layouts raced here, all losing to 339,970: `(K,R,C,P)` 340,326 · `(C,R,K,P)` 340,308 ·
`(R,C,P,K)` 360,241 · byte lane 346,919 · same-flag RLE 353,512 · temporal delta mod-16 366,141 ·
per-channel independent streams 343,722.

**The token member is CLOSED at 341,294 B** across 6 layouts, 4 coders, 3 transforms and 5 context
models. The lesson generalises: *an oracle conditional entropy is a bound on a TWO-PASS coder with the
model given away free. Do not quote it as realizable headroom.*

---

## 5. Manifest custody — the six-hash trichotomy, and a rule-118 defect in the inherited plan

**All six `sha256` fields are UNREAD.** `inflate_runner_v4d.py` reads exactly five manifest keys —
`frame0_policy`, `pose_dim0_offset`, `rs_beta_mags`, `tr1_metadata`, `st_grid` — and zero hash fields.

| field | read? | correct? | verdict |
|---|---|---|---|
| `renderer_sha256` | no | ✓ | unread + correct — rate with no function |
| `selector_sha256` | no | ✓ | unread + correct |
| `pose_stub_sha256` | no | ✓ | unread + correct |
| `pose_warp_sha256` | no | ✓ | unread + correct |
| `tokens_sha256` | no | **✗** | **unread + WRONG** (matches neither the member, the TR1-framed payload, nor the raw codes) — `ddm_gd3` |
| `tr1_packet_sha256` | no | **✗** | **unread + WRONG** — **NEW.** `build_packet` verified deterministic, so this is not a rebuild artifact: the field is false |

**read+correct: 0 of 6.** The right move is neither to migrate them (that launders two false fields into
the generic side) nor to keep paying for them: **delete all six**, and keep build-time provenance in a
repo-side receipt at zero archive cost. A custody hash worth its bytes is one the receiver *reads* and
*fails closed on*; none of these did.

### The rule-118 defect I inherited

`ddm_ix1`'s 27 B minimal manifest kept only `pose_dim0_offset` and classified `rs_beta_mags` as
*"GENERIC — quantization GEOMETRY, not assignment."* **Measured: it is not.** The shipped table has 13
entries; the vendored `DEFAULT_BETA_MAGS` has 3. It matches no vendored table and `ddm_cp1` measured that
the builder derives it from this clip's solve on every v4d archive. **Migrating it into `inflate.py`
would be hide-data-in-code.** It is counted here, in `pack_config_section`. The byte cost of being
honest is **10 B**. The byte cost of being wrong would have been a fake headline.

`st_grid` is the sharper case, and `ddm_cp1` is right about it: it is byte-equal to the vendored
`ST_GRID` **on `dc1_fold`** (verified here: `classify_against_vendored → GENERIC`) and FITTED on `ms8` /
`pj2`. **Same key, same slot, opposite rule-118 verdict.** So the module does not classify by field
name: `classify_against_vendored(value, vendored)` computes the verdict per archive, and
`pack_config_section` **refuses to decide without the reference table** rather than guessing. All
migration claims in this memo are stated for `dc1_fold` and do not transfer.

**Counted config section: 32 B** — `pose_dim0_offset` (f32, exact) + the 13-entry fitted β codebook
(f16, exactness enforced, not silently rounded) + one flag byte recording that `st_grid` was proved
generic *on this base*. Migrated to `inflate.py`: `frame0_policy`, `tr1_metadata`, `schema`, `st_grid`,
and the deleted `beta_idx_counts` / `selector_num_two` (derivable from the coded sections).
**Manifest 1,450 → 32 B counted.**

---

## 6. BYTE-CLOSED RESULT

| stack | archive B | Δ B | ΔS | % of gap |
|---|---:|---:|---:|---:|
| baseline `dc1_fold` (rebuild parity verified exact) | 360,309 | 0 | 0 | 0 |
| C1 consolidate 6 members → 1, shared small-group coder | 359,571 | −738 | −0.0004913 | −0.068% |
| C2 + `IX2TOK01` tokens | 354,388 | −5,921 | −0.0039424 | −0.543% |
| C3 + binary config section, `pose_stub` deleted | 353,766 | −6,543 | −0.0043565 | −0.600% |
| **C4 + `IX2REN01` renderer** | **353,705** | **−6,604** | **−0.0043973** | **−0.6055%** |

**S: 0.8983775 → 0.8939802.** Archive sha256 `3e2d2416801db6e8fc6d50dee3fdeeddd1959daba5b3759fef7b832b9bebace9`.

**Why this is an exact ΔS with zero scorer time.** Every transform is LOSSLESS and the decode was proved
bit-identical **from the new archive bytes alone** (8/8 checks), so `d_seg` and `d_pose` cannot move; the
only term that changes is `25 · bytes / 37,545,489`, recomputed from components on a `stat()`-able
archive whose framing parity was verified. Not an estimate, and not the rounded `final_score` field.

**This exemption is the whole advantage of the line, and it is not free to spend.** `ddm_cp1` measured
that a re-representation costs a *tighter* solve relatively more — `ms8` −0.009% (a gain), `pj2` +0.099%
(11× worse) — because the f16 storage lattice is not invariant even where the algebra is (homography
agreement 3.945e-16, yet 188 of 255 pairs got worse). We are exempt **only** because our decode is
bit-identical. Any rung of this line that goes lossy or re-quantizes an already-solved value inherits
that law, and it lands hardest on our best-solved members. Stated in the module docstring so the next
caller cannot miss it.

### 6b. THE RECEIVER IS LANDED AND PARITY-VERIFIED (commit `bed39893b4`)

`experiments/inflate_runner_v4d.py` now reads the single-member container. The path is **additive**:
it selects on the existence of `0.bin`, so every legacy 6-member archive keeps decoding unchanged.

**FULL n600 frame parity: 0 mismatched pairs of 600.** Every `frame_0` and `frame_1` rendered from the
ix2 container is **bit-identical** to the one rendered from the legacy archive, through the real
receiver — not a spot check, the whole clip. Every decoder state array also compares equal
(`p_best`, `st_idx`, `sel`, `ab`, `beta_idx`, `st_vals`, `beta_mags`, `dim0_offset`, `n_pairs`).

Migrated to `inflate.py` as generic constants: `frame0_policy`, `tr1_metadata`, the inert `pose_stub`
string, the section order, and `st_grid` **only because the encoder proved it byte-equal to the vendored
ladder on this base**. Counted in the config section: `pose_dim0_offset` and the fitted β codebook.

**What is still OWED before promotion:** vendor `ddm_ix2_archive_container.py` into the archive's runtime
tree next to `ddm_tr1_runtime.py` (free generic code, no counted bytes), then one confirmation exact eval
on the rebuilt `archive.zip`. **Pointer UNMOVED until then.**

---

## 7. ROUND-1 ADVERSARIAL REVIEW OF MY OWN RESULT

Four attacks, all run, all closed:

1. **"Is the −6,604 a framing artifact — was the BASELINE badly built?"** Re-zipping the six untouched
   members with minimal fixed-timestamp framing reproduces **360,309 B exactly**. The baseline framing is
   already minimal for six members, so the saving is a true delta, not a comparison against a bloated
   control. *(This is the attack that would have inflated the headline, and it is the first one I ran.)*
2. **"The migration is DERIVED, not measured."** It was — so I measured it: `build_packet` fed the
   `tr1_metadata` dict as a migrated `inflate.py` **constant** produces a **byte-identical packet** to the
   one built from the manifest, and it parses (`num_pairs = 600`). Now measured.
3. **"Is `tr1_packet_sha256` wrong, or is my rebuild nondeterministic?"** `build_packet` verified
   deterministic across two calls. The field is false.
4. **"Would the tests pass if the code were broken?"** `test_cell_major_layout_must_beat_aos` builds an
   array with the structure measured on the live lattice and asserts AoS loses — an identity transpose
   fails it. `test_decode_needs_only_the_frame` is the direct regression for ix1's borrowed-`base` fake:
   drop the base block and decode must RAISE, never guess. `test_two_tier_payload_beats_independent_coding`
   fails if the shared-coder grouping becomes a no-op. `test_zip_framing_overhead_matches_a_real_zip`
   compares against an actual `zipfile` build rather than a remembered constant.

**Where I was wrong, before review:** my first `code_block`-based archive measured **−6,404 B**, not
−6,627, because the module stored the whole container raw and silently lost both the selector's deflate
and the shared-coder gain. The fix (`build_payload`'s two-tier bulk/joint split) is now the structural
form of the finding rather than a scratchpad accident.

**Remaining incomplete coverage, named:** (a) the reader IS now landed and n600 parity-verified (§6b),
but the module is not yet vendored into the archive's runtime tree and no confirmation exact eval has
run, so the bytes are measured, decodable, and NOT yet banked; (b) `pj2 × ix1` (`ddm_cp1`: 354,331 B,
ΔS −0.0715 = 9.85% of gap) is DERIVED and I did not run the composition — I unblocked it; (c) the f16 mantissa-width solve on the 8,203 B of pose fields is LOSSY and was not
touched, per §6's law; (d) codegen/straight-line decoders were **not run** — with the LOC cap deleted and
no time term, they have no rate effect and I do not believe they are worth an arm.

---

## 8. NEXT-IF-RESUMED (ranked by measured addressable bytes)

1. **Run `ddm_cp1`'s `pj2 × ix1` composition — 354,331 B, ΔS −0.0715 = 9.85% of gap — through this
   receiver.** It was blocked on exactly this decoder and the decoder now exists and is n600
   parity-verified. **This is ~16× larger than this arm's own row and is the single highest-value item
   on the board.** Two cautions carried from §5 and §6: (a) `st_grid` is FITTED on `pj2`, so
   `classify_against_vendored` will return `VIDEO_DERIVED` there and the encoder will correctly COUNT
   it — do not reuse this memo's `dc1_fold` migration number; (b) `pj2` is the tighter solve, so if any
   part of that composition re-quantizes rather than re-frames, cp1's +0.099% law applies.
2. **Vendor `ddm_ix2_archive_container.py` into the runtime tree and take one confirmation exact eval**
   to bank the −6,604 B on `dc1_fold`. The reader is landed (§6b); this is packaging plus one eval.
3. **Do NOT re-race the token member.** 6 layouts × 4 coders × 3 transforms × 5 context models, all
   measured, all lose to cell-major nibble + brotli. Closed at 341,294 B.
4. **Do NOT re-race `renderer.sec` or `pose_warp.stp`.** Measured zeros: the lottery mask is 1 B above its
   combinatorial floor, and the shipped `kl1` byte-plane beats every alternative layout on both f16 fields.
5. **Do NOT quote an oracle conditional entropy as realizable headroom** (§4). The next arm that wants a
   coder win must measure the *sequential* code length, which charges the model cost the way a real coder
   does.
6. **Delete the six manifest hashes rather than migrating them** (§5), and if custody verification is
   wanted, make it a receiver-read fail-closed check that earns its bytes — two of the six are currently
   false.
7. **The remaining lossless headroom on this vehicle is ~50 B** (colex on `sel_coded`, order-0 arithmetic
   on `beta_coded`, `pose_warp` framing). **The lossless axis is now essentially exhausted at
   353,705 B.** Everything past this point is LOSSY and inherits §6's tighter-solve-costs-more law.

---

*STORES CONSULTED:* `ddm_ix1_index_compaction_ladder_20260802.md` (the charter and the three items it
ranked — items 2 and 3 are answered here, item 1 is the owed landing), `ddm_gd3_grid_downsample_gate_20260802.md`
(`tokens_sha256`, extended here to all six), `ddm_cp1` (the `(field, archive)` custody law, the
`pj2 × ix1` row, and the tighter-solve-costs-more law — all three consumed, none re-derived),
`experiments/inflate_runner_v4d.py` + `src/tac/optimization/ddm_tr1_runtime.py` +
`experiments/ddm_r7_token_coder.py` (read at source: the renderer field split came from
`_encode_renderer`, not from its docstring), `src/tac/optimization/pfs1_warp_receiver.py` (the vendored
`ST_GRID` that makes the `dc1_fold` verdict computable), CLAUDE.md L20–L32 (intake intelligence, raced
not adopted — L20 monolithic `0.bin` and L23 split streams both measured here rather than assumed),
`tac.canonical_equations.gap_decomposition_against_floor_20260802` (the denominator).
