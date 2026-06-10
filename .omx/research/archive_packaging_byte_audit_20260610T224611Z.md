# Archive-packaging byte audit + embedded bit-packing (#79) — 2026-06-10

**Verdict (lead):** The packaging/bit-packing lever is **CLOSED** on the current frontier archive
(`b46897267…`, 177,169 B). **Zero recoverable bytes. No pointer move.** The scorer counts ONLY
`archive.zip`'s on-disk size; the container is already at its theoretical floor and the payload is at
its entropy floor. The rate term (0.118) is irreducible by packaging — it only falls with a **smaller
payload** (the capstone's smaller learned basis). This NEGATIVE closes the lever and confirms the
content path is the only rate path.

## 1. What the scorer actually counts (exact, from source)
`upstream/evaluate.py:63`:
```python
compressed_size = (args.submission_dir / 'archive.zip').stat().st_size
```
- **ONLY** `archive.zip`'s file size (a single `.stat().st_size`). `rate = compressed_size / 37_545_489`
  (`uncompressed_size = sum(... uncompressed_dir.rglob('*'))` is a fixed constant for the public set).
- **NOT counted:** `inflate.py`, `inflate.sh`, `README`, any metadata, or anything else in the
  submission dir. They sit *beside* `archive.zip`, not inside it.
- ⟹ **The operator's "minify the scripts / remove human-readable labels" hypothesis pays off ONLY for
  bytes INSIDE `archive.zip`.** Minifying `inflate.py` is free-but-useless here. (A competitor PR that
  "gained by minifying scripts" would only have gained if those scripts were *bundled inside* the
  scored zip — not the case for the comma format, where `inflate.*` are sibling files.)

The packaging lever therefore reduces to exactly two surfaces: **(A) ZIP-container overhead** and
**(B) sub-byte bit-packing of the payload inside the single zip member.**

## 2. ZIP-container anatomy (surface A) — already at the floor
`archive.zip` = 177,169 B decomposes EXACTLY (zero unexplained bytes):
| component | bytes |
|---|---:|
| payload (single member `"x"`, **STORED**, 177,069 uncomp = comp) | 177,069 |
| local file header (30 + 1-char name) | 31 |
| central directory header (46 + 1-char name) | 47 |
| end-of-central-directory record (no comment) | 22 |
| data descriptors / padding / zip comment | 0 |
| **total** | **177,169** |

- Member name is already **1 character** (`"x"`) — minimal.
- Method is **STORED** — no double-compression overhead (correct: the payload is already compressed).
- Overhead is exactly **100 B**, which is the **theoretical minimum** for a valid single-member ZIP
  (31 + 47 + 22). You cannot build a smaller compliant `archive.zip` for this payload. recoded-R3
  already minimized the container. **Surface A slack = 0 B.**

## 3. Payload bit-packing / residual compressibility (surface B) — at the entropy floor
The 177,069 B payload `"x"` (the renamed `0.bin`, PR#112-class entropy-coded substrate, R3 = "payload
entropy recode"):
- **order-0 byte entropy = 7.9990 bits/byte** (flat histogram; floor estimate = 177,047 B = 100.0% of
  payload).
- **Every general compressor makes it LARGER** (lossless re-compression test):
  | coder | result | Δ |
  |---|---:|---:|
  | zlib -9 | 177,130 | **+61** |
  | bz2 -9 | 178,255 | **+1,186** |
  | lzma/xz -9e | 177,136 | **+67** |
  | lzma raw -9e | 177,079 | **+10** |
  | brotli q11 | 177,074 | **+5** |

  Every Δ is positive (bigger) ⟹ the payload is **incompressible** = at the entropy floor.
- **Why sub-byte bit-packing yields nothing here:** bit-packing only beats byte-alignment when the
  stream is NOT entropy-coded. An arithmetic/range-coded stream already packs symbols below
  ⌈log₂N⌉ bits each. The recoded-R3 entropy recode already did this — there are no byte-aligned
  fields left to bit-shift tighter. **Surface B slack = 0 B.**

## 4. The actionable lesson (carries to the capstone #78)
The packaging lever is closed *on the borrowed/entropy-recoded frontier* — but the audit yields a
binding **design constraint for our own basis (#78):**
- **Budget EVERY payload stream through an entropy coder, not fixed bit-packing.** A VQ-NeRV with
  K=16 selector indices: bit-packing to 4 bits/index is good, but range-coding the (non-uniform)
  index histogram beats it. The capstone's `0.bin` must be entropy-coded end-to-end (like R3), then
  STORED in a 1-char-member ZIP with the 100 B floor container — so its rate term is at *its* floor.
- The rate term only comes down by a **smaller payload** (fewer/cheaper learned parameters + indices),
  NOT tighter packaging. This re-confirms the capstone (smaller basis → ~60-72 KB) as the sole rate path.

## 5. NO-FAKE / firewall compliance
- Measured against the EXACT scorer law (`evaluate.py:63`), not assumed.
- The negative is honest: no slack found ⟹ **no `scorer_quotient_candidate_row`, no pointer move**
  (`pointer_update_eligible=False` by construction — there is no candidate). The frontier pointer
  is UNMOVED (0.19109982 → 0.19109982).
- Axis: `[exact: evaluate.py source + on-disk archive bytes]`. No GPU, no dispatch, $0.

## 6. Disposition
- **#79 → DONE (negative, lever closed).** Reactivation criterion: a NON-entropy-coded payload
  (e.g. the capstone's first byte-closed archive before its entropy-coding pass) — re-run §3 on it;
  if it shows residual compressibility, that IS free rate. The audit tooling (zip anatomy + entropy +
  lossless re-compression sweep) is reusable for every future byte-close.
