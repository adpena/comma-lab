# ddm_sz1 — the semantic-blob fp16 metadata split: receiver-closed, byte-closed, composed

**Arm:** ddm_sz1 (Opus). **Date:** 2026-08-18. **Charter:**
`.omx/research/charters/ddm_sz1_semantic_blob_serialization_split_20260818.md`.
**Spend:** $0. No Modal, no paid dispatch, no GPU, no scorer job. Byte-level + bounded local decode only.
**Fires:** nothing. MAIN fires.
**Craft:** `docs/operating_manual_craft_handoff.md`.

---

## 1. The headline, in the order that matters

1. **The candidate is real and receiver-closed.** Six archives built, all byte-closed, all
   determinism-repeat identical, all decoding **bit-exact** to their base through a patched
   receiver carrying the free un-split.
2. **Best composed archive: 179,930 B — 671 B under the fx1 pointer (180,601 B), projected
   S 0.15771358, ΔS −4.468e-4.** Distortion is unchanged *by construction* (the decode is
   bit-exact), so this is a pure rate move.
3. **The standalone split does NOT beat the pointer, and MAIN must not read it as if it does.**
   On the rr4 base it lands 180,641–180,663 B, which is **+40 to +62 B WORSE than fx1**, because
   rr4 (181,161 B) is already 560 B behind the pointer. The split is a shipping win **only
   composed** with fx2's token stream.
4. **fx2's −515 B is real as a byte count but was mis-attributed as a mechanism.** It is the
   same transform applied at offset **41** — `_HEADER_BYTES`, the *canonical* WANS1 header —
   to a body whose real prefix is `_OFFSET_BYTES` = **30**. The region is misaligned against
   the fp16 metadata by 11 bytes. The mechanism-derived split is **−498 B**; the extra 17 B is
   Brotli alignment noise. Full derivation in §3.
5. **The composition is exactly additive: 0 B interaction**, measured, not assumed (§5).
6. **The extension is a negative:** only the semantic blob pays. `carrier_blob` best is **+12 B
   (no win)**; `hpac_blob` is **NOT MEASURABLE** on this instrument (§7).

---

## 2. What the transform is

The RX1M `semantic` section is a Brotli stream whose body is the F12 stream-ordered WANS1 payload:

```
offsets(30 B) || fixed metadata(8,284 B) || stream_area(27,726 B)   = 36,040 B
```

The fixed metadata is 4,142 little-endian `<f2` values — 16 W4 scale arrays and 22 fp16 tensors —
and **nothing entropy-codes it**. It rides the outer Brotli as raw interleaved fp16, so an
exponent byte alternates with a mantissa byte. Grouping the two byte planes (all high bytes, then
all low bytes) before the container's Brotli is a pure **serialization** change: decoded values
identical, only the on-disk layout moves.

This is exactly the mechanism sv2's **IX2TOK01 law (#859)** predicts a win for — the live coder is
paid for **match structure**, and the split turns a near-constant exponent plane into long runs. It
is measured through the real coder at the container's own parameters, so it is not the
entropy-surrogate class (#862).

The receiver un-split is the **exact inverse permutation**: free receiver code under rule 118 (no
transmitted table, a single O(n) byte scatter), running *before* any parsing, restoring the F12 body
byte-for-byte so every downstream check sees today's bytes.

**Versioning costs ZERO bytes.** The RX1M 14-byte header already carries a `reserved` byte that both
sides validate as strictly zero. Bit 0 is the split flag. `reserved == 0` takes the identical path it
takes today — inactive is byte-identity by construction — and unknown bits still refuse, so the check
stays fail-closed. (The pre-existing `reserved != 0` refusal is widened to
`reserved & ~SZ1_RESERVED_KNOWN_BITS`, not removed.)

---

## 3. Correcting the record on −515 B (the charter's optimal-form gate)

The charter required reproducing **−515 B ±2** on the real section through the real Brotli before
receiver work counted. My instrument reproduced the shipped section at **delta +0** (control
re-Brotli = 34,763 B = shipped), then measured the split at **−498 B**, not −515.

Rather than absorb a 17 B miss, I scanned the prefix-cut offset. `cut=41` reproduces **−515 exactly**.
41 is `_HEADER_BYTES` — the length of the *canonical* WANS1 header (`F11_FIXED_PREFIX`(11) + offsets(30)).
The shipped F12 body has **no** 11-byte prefix, so its metadata begins at 30. fx2's r5c applied the
canonical offset to the F12 body: the split region is shifted 11 bytes, skipping 11 bytes of real
metadata and pulling in 11 bytes of stream area. (r5b's retained script confirms the lineage — it
slices `sem_raw[_HEADER_BYTES : _HEADER_BYTES + meta_n]` on the *canonical* blob, where 41 is correct.)

The landscape, measured on the real section (control +0 throughout, q=11 lgwin=24):

| offset | section B | Δ | what it is |
|---:|---:|---:|---|
| 30 | 34,265 | **−498** | `_OFFSET_BYTES` — the actual metadata start. **Zero fitted parameters.** |
| 41 | 34,248 | **−515** | `_HEADER_BYTES` — fx2's r5c. Misaligned by 11 B. |
| 45 | 34,244 | −519 | |
| 49 | 34,243 | **−520** | argmax over offsets 0..400 |
| 15 | 34,247 | −516 | |

**The ~22 B spread is Brotli alignment noise, not mechanism.** Adjacent offsets swing ±20 B, a joint
(offset, length) sweep does **not** beat (49, 8284), and the coarse length sweep at step 50 only
reached −505/−514. So: the mechanism is worth **−498 B**; everything above it is noise fitted to one
frozen payload.

All three offsets are equally correct and equally safe — the un-split is a pure byte permutation, so a
straddling region restores byte-exactly just as an aligned one does. All three are built, retained,
and offered. `SHIPPED_PROFILE` is frozen to **DERIVED** because a decoder takes no arguments and the
defensible constant is the one the format supplies.

---

## 4. The candidates (all MEASURED, all bit-exact)

Rate term 25/37,545,489 = 6.6586e-7 per byte. `projected_S` inherits the fx1 `[contest-CUDA]`
distortion terms — legitimate because the decode is bit-exact, but it is a **PROJECTION** until an
exact eval row exists.

| id | base | offset | archive B | Δ vs rr4 | Δ vs fx1 ptr | projected S | ΔS vs ptr |
|---|---|---:|---:|---:|---:|---:|---:|
| `rr4__derived` | rr4 | 30 | 180,663 | −498 | **+62** | 0.15820165 | +4.13e-5 |
| `rr4__fx2_r5c` | rr4 | 41 | 180,646 | −515 | **+45** | 0.15819033 | +3.00e-5 |
| `rr4__tuned` | rr4 | 49 | 180,641 | −520 | **+40** | 0.15818700 | +2.66e-5 |
| `fx2_a__derived` | fx2 A | 30 | 179,952 | −1,209 | −649 | 0.15772823 | **−4.321e-4** |
| `fx2_a__fx2_r5c` | fx2 A | 41 | 179,935 | −1,226 | −666 | 0.15771691 | **−4.435e-4** |
| `fx2_a__tuned` | fx2 A | 49 | **179,930** | **−1,231** | **−671** | **0.15771358** | **−4.468e-4** |

Every row: control delta **+0**, determinism repeat **identical**, decoded fields **bit-exact**,
base **still decodes unchanged** through the patched receiver.

**The three rr4 rows are NOT shipping candidates.** They are retained as the clean standalone
measurement of the lever. rr4 is 560 B behind the pointer, so a −498/−520 B win on it still lands
above fx1.

---

## 5. The composition, measured rather than assumed

fx2's candidate A **byte-closed after its fire-order was sealed as BLOCKED** — the artifact exists at
`/Volumes/APDataStore/pact/ddm_fx2/byteclose_a/retained/archive.zip`, **180,450 B**, sha
`9de0f6db3ca7ae4efcd9237752b7c95ed1119d9285f8aadd92fee7c8c18547ef`, exactly its projected 180,450 B
with a 109,801 B token stream. So the composition is a **real archive**, not a projection.

The two changes touch **disjoint** regions: fx2 re-encodes the token stream (member tail), sz1
rewrites the semantic section (RX1M prefix). Additivity measured:

| composed | expected `181,161 − 711 − split` | measured | miss |
|---|---:|---:|---:|
| fx2 A + derived (−498) | 179,952 | **179,952** | **0 B** |
| fx2 A + fx2_r5c (−515) | 179,935 | **179,935** | **0 B** |
| fx2 A + tuned (−520) | 179,930 | **179,930** | **0 B** |

The charter's falsifier (b) was "composed archive misses the additive projection by >10 B ⇒ report
the interaction". **Miss = 0 B on all three.** The archive is a single `ZIP_STORED` member, so
archive delta equals member delta equals section delta exactly — no container discount, which the
three rr4 rows independently confirm (slope 1 across three different section lengths).

fx2's token stream is carried **verbatim** in every composed member (109,801 B, sha
`5b09fd784a7c80cfe69b49d098acaee60276bba56b837d32c7654e1ac7798a43`), checked at build time.

**Decoded token field.** The charter asks that
`9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` be unchanged. MEASURED: the token
stream *section bytes* are bit-identical and the corrector is untouched. DERIVED (not measured here):
the decoded field therefore cannot change, since it is a pure function of those bytes and that
corrector. A full parse-back to re-derive the field sha is ~25 min of decode and is listed as a
pre-fire step in the fire-order rather than claimed here.

---

## 6. What was proven, and how it could still be wrong

Proven by execution:

- **Bit-exact decode.** Nine decoded fields (`semantic_blob`, `carrier_blob`, `hpac_blob`,
  `token_stream`, `residual_payload`, `table`, `token_codec`, `compensation_blob`, `schema`) hash
  identically between base and candidate through the patched receiver, by a deep structural digest.
  Only `compressed_models` — the raw container bytes we deliberately rewrote — differs, and the
  build **refuses** if it does *not* differ (an inert transform is a fake).
- **Backward compatibility.** The unmodified base still decodes unchanged through the patched
  receiver: `reserved == 0` is byte-identity.
- **Encoder/receiver agreement.** A test compiles and executes the *rendered receiver helper* and
  checks it inverts the encoder's own split, for all three profiles. A decoder takes no arguments,
  so a constant mismatch cannot be caught at decode time — it would silently corrupt every weight.
- **Non-vacuity.** Four injected mutations were **all caught**: inverse planes swapped (5 failures),
  split made the identity (11), receiver offset off-by-one (3), header patch dropped (2).

Found and fixed by the suite: `set_rx1_reserved` wrote byte index **8** (the low byte of
`hpac_bytes`, a section length) instead of index **7**. The shipped archives were never affected —
the builder packs the header positionally — but the helper would have corrupted a length. It now
re-reads the header and refuses if anything but the reserved byte moved.

Honest residual risk:

- `projected_S` assumes fx1's distortion terms carry. That is sound *given* bit-exact decode, but no
  exact eval row exists for these archives. **Only a T4 row makes it a score.**
- The receiver is validated by `read_residual_archive`, not by a full `inflate.sh` parse-back. The
  un-split runs before parsing and restores the body byte-exactly, so the risk is low, but it is
  **not zero** and is a pre-fire step.
- `tuned` and `fx2_r5c` carry an offset fitted to this frozen payload. Safe, but not mechanism.

---

## 7. Bounded extension — a negative

pd1's prior says weight serialization is corpus-wide unsaturated. On **our** sections it is not:

| section | shipped | control Δ | best split Δ | verdict |
|---|---:|---:|---:|---|
| `semantic_blob` | 34,763 | **+0** | **−520** @ off 49 | **PAYS** |
| `carrier_blob` | 22,161 | **+0** | **+12** @ off 261 | **NO WIN** |
| `hpac_blob` | 13,515 | **+40** | — | **NOT MEASURABLE** |

`hpac_blob` gets no number on purpose: re-Brotli does not reproduce its shipped length, so the
instrument is not calibrated there and any delta would be measuring the parameter mismatch, not the
transform. Reporting a number there would be the uncalibrated-instrument fake.

**The semantic blob is the only section of the three carrying un-entropy-coded fp16 metadata.** The
axis is not corpus-wide on this vehicle.

---

## 8. Adjudication against the charter's prior-law prediction (m38)

| prediction | outcome |
|---|---|
| Reproduce −515 B ±2 on the real section | **HIT, and explained.** −515 reproduced exactly at offset 41; the mechanism-derived value is −498. The mis-attribution is reported, not absorbed. |
| Standalone −515 B ⇒ ΔS ≈ −3.43e-4 | **HIT** (−3.429e-4 vs rr4) — but with a **material caveat the prediction omitted**: measured vs the *fx1 pointer* it is **+3.0e-5**, i.e. worse. The standalone lever does not ship. |
| Composed ≈ −1,226 B vs rr4 | **HIT exactly.** 179,935 B = −1,226 B, 0 B interaction. |
| Composed ΔS ≈ −8.16e-4 vs rr4 | **HIT** (−8.163e-4). |
| Composed ≈ −4.4e-4 vs the fx1 pointer | **HIT** (−4.435e-4; −4.468e-4 for `tuned`). |
| Falsifier (a): un-split fails bit-exact round-trip | **Did not fire.** Bit-exact on all six. |
| Falsifier (b): composed misses additive projection by >10 B | **Did not fire.** Miss = 0 B. |
| Falsifier (c): decode wall-clock delta >5 s | **Not measured** — an O(n) 8 KB scatter, but see the fire-order's pre-fire step. |

---

## 9. Custody

Root `/Volumes/APDataStore/pact/ddm_sz1/`, manifest `RETENTION_MANIFEST.json`:
six archives (each with `archive.zip`, `archive.repeat.zip`, `member.bin`, `BUILD_REPORT.json`),
three patched receiver trees, and the extension probe — every one sha256'd. **Per-candidate**
payloads retained, not only the winner's.

Code: `experiments/ddm_sz1_semantic_metadata_split.py` (transform + receiver patch),
`experiments/ddm_sz1_build_split_archive.py` (builder),
`experiments/ddm_sz1_extension_probe.py` (extension),
`experiments/test_ddm_sz1_semantic_metadata_split.py` (25 tests).

Fire-order: `.omx/research/ddm_sz1_sealed_fire_order_20260818.json`.

---

## 10. What MAIN should do

Fire **`fx2_a__tuned`** (179,930 B) or **`fx2_a__derived`** (179,952 B) — 22 B apart; `derived` is the
defensible-constant choice, `tuned` the lowest-byte one. Both need fx2 candidate A's own distortion
row, which has never been measured on T4, so the fire buys **two** answers at once: fx2's token model
and sz1's split. Pre-fire steps and falsifiers are in the fire-order.

Do **not** fire any `rr4__*` row: they are above the pointer.
