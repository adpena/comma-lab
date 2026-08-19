# ddm_rr5 — the CPR1 lossless rider is now APPLICABLE, and it is worth 183 B, not 278 B

`date_utc: 2026-08-19` · `owner: ddm_rr5 (rider pre-staging)` ·
`axis: [byte-exact, lossless — no scorer, no advisory row, d_seg and d_pose unchanged by construction]` ·
`score_claim: false` · `promotable: false` · `frontier_moved: false`

**Own-vehicle frontier, UNMOVED by this unit.** This arm produced no score and attempted none.
Cost: $0 — no Modal, no dispatch, no GPU, no scorer forward, no render of a scored frame.

Receipts: `tools/ddm_rr5_rider_apply.py` · `tools/ddm_rr5_rider_compose_probe.py` ·
`src/tac/rr5_arith_basis.py` · `src/tac/tests/test_ddm_rr5_rider_apply.py` (34 tests, all executed) ·
`/Volumes/APDataStore/pact/ddm_rr5/retained/pointer_rider/RR5_RIDER_RECEIPT.json` ·
`/Volumes/APDataStore/pact/ddm_rr5/retained/compose/RR5_COMPOSE_REPORT.json`.

---

## THE ANSWER, FIRST

**The rider works, it is lossless, and it is now a one-command compose step. It is worth
183 B on the pointer body — `ΔS = −1.2185e-4`, which is 66% of the −1.85e-4 the chain
budgeted for it.** The shortfall is not a defect in the rider; it is that the chain's figure
was assembled from a PROVISIONAL number plus a second leg that ra2 never measured.

1. **Charter-recall correction #1 — the −1.85e-4 is not a measured figure.**
   `ddm_ra2`'s own receipt says **+263 B raw / ~230 B realised → `ΔS ≈ −1.53e-4`**, and labels
   the realised half **PROVISIONAL** in as many words. `ddm_tx1` then added **ra1's 48 B
   `basis_scales` leg** to reach ~278 B ⇒ −1.85e-4, and `jg2` clause 6 / `jg3` inherited that
   sum as "MEASURED". The 48 B leg has no round-trip receipt in ra2's artifact. The
   **LOSSLESS claim is sound** (round-trip exact, 27,648/27,648 symbols); only the SIZE was
   over-stated. This is the [[cross-regime constant transfer]] genus: a provisional number
   hardened into a headline one memo downstream.
2. **Charter-recall correction #2 — the "32 B dead table" is 16 B in the archive.**
   ra2 counted the Huffman length table in the *canonical CPR1* layout (32 × u8). The archive
   ships it **packed at 4 bits per symbol = 16 B**
   (`residual_archive.py:134`, `_unpack_unsigned(packed[123:139], 32, 4)`). Half the table
   saving never existed at the layer that is charged.
3. **The transfer to the pointer body is exact, and cost nothing to establish.** The pointer
   body's basis stream is **byte-identical to hv1's** — 12,277 B, 98,213 bits,
   `sha256 a9d037a59950…`, same 32-length table `sha256 48083e5a75ad…`. Only the coefficients
   moved (79,020 → 79,027 bits) when up2/up3 re-solved the carrier. `tx1` predicted exactly
   this and asked for it to be re-measured rather than assumed; it now is.
4. **Order of operations: the rider and jg2's token edits COMMUTE — MEASURED, 0 B apart.**
   Both orders produce **byte-identical archives** and both parse through the real receiver to
   the same `carrier_blob` and `token_stream` sha256. Composition is **exactly additive**.
5. **Losslessness is PROVEN by decode identity, not argued.** Three controls run on every
   apply and any failure refuses the write; a fourth (end-to-end frame render) is available
   behind `--full-inflate`.

---

## 1. What the rider actually does

The shipped F26 carrier codes its 27,648 five-bit basis symbols under a **static order-0
canonical Huffman code with ONE table shared across all 12 basis atoms**. ra2 measured that
the atoms have different code distributions, so a per-atom context pays. The rider replaces
that code with an **adaptive arithmetic coder contexted on the atom index** — the model is
driven identically on both sides, so no probability table is transmitted at all.

**The coefficient stream is NOT touched.** ra2 measured every adaptive model LOSING to Rice
there (−415 B and monotonically worse with finer contexts) because 7,200 samples over a
4,096-symbol alphabet is 1.76 samples/symbol — the model cannot amortise its own learning
cost. Rice's single parameter per dimension is the correct model class. The rider leaves it
alone, and the tool has no switch to change that.

**Measured on the pointer body** (`7ce46fd7…`, 176,420 B):

| quantity | shipped | rider | delta |
|---|---:|---:|---:|
| basis payload | 12,277 B | 12,046 B | **−231 B** |
| basis bits | 98,213 | 96,368 | −1,845 |
| packed Huffman table | 16 B | zeroed | (Brotli absorbs) |
| carrier body (raw) | 22,187 B | 21,956 B | −231 B |
| carrier stream (Brotli) | 22,143 B | 21,960 B | −183 B |
| **archive.zip** | **176,420 B** | **176,237 B** | **−183 B** |

`ΔS = −183 × 25/37,545,489 = −1.2185218842e-4`.

**Why 183 and not 231.** The carrier ships Brotli-compressed, so the raw saving is discounted
by whatever redundancy Brotli was already removing from the Huffman payload — an arithmetic
stream is closer to incompressible, so Brotli gives less back. This is `ddm_up3`'s lesson
("archive ΔB ≠ payload ΔB") landing in our favour's opposite direction, and it is exactly the
gap ra2 flagged as PROVISIONAL. **Measured at the archive layer, which is the only layer the
rate term charges.**

**Where the table went.** The 16 B packed table is zeroed rather than removed, so every packed
offset downstream is unchanged and the receiver patch stays a two-line insertion. The decoder
rebuilds the table from the decoded symbol histogram. That is admissible only because the
reconstruction is EXACT — the encoder rebuilds the table, re-encodes the whole basis under it,
and **refuses to drop the table unless the replay reproduces the shipped basis bytes
byte-for-byte**. On this body it does (test
`test_shipped_huffman_table_reconstructs_exactly_from_the_histogram`).

---

## 2. Losslessness — the controls, and what each one rules out

`d_seg` and `d_pose` are unchanged **by construction**, so no scorer runs and no advisory row
is created. But "by construction" is a claim about code, so it is checked against bytes:

| control | what it proves | what it rules out |
|---|---|---|
| **C1** arithmetic round-trip | the coded stream decodes to the exact input symbols, 27,648/27,648 | a coder bug that silently changes a symbol |
| **C2** carrier-body identity | `restore_carrier_body(rider) == shipped`, byte-for-byte | a framing/packing bug; makes every stage below the carrier bit-identical *by construction* |
| **C3** receiver decode identity | the REAL `read_residual_archive` on both archives returns byte-identical parts | a container edit that parses but decodes differently; also proves the rider archive PARSES |
| **C4** frame identity (`--full-inflate`) | both archives render to the same frame sha256 | anything the parts-level comparison could not see |

**Identity control, run before any of them.** The tool first re-emits the *input* archive from
its own parts and requires `sha256 7ce46fd7…` back. Without that, a container difference could
masquerade as a rider saving. It passes at `(CK2 carrier plane OFF, quality 10, lgwin 16)`.

**One field is excluded from C3, and it is reported rather than hidden.**
`compressed_models` is `outer[:model_end]` — the raw *compressed container bytes*. Any
container edit changes it by definition. It is admissible to exclude ONLY because it is inert:
in the shipped tree the name appears exactly twice (`residual_archive.py:338` declares the
dataclass field, `:498` populates it) and **nothing reads it**. The tool re-measures that
inertness on every run, and a test plants a read site to prove the check can FAIL
(`test_inertness_check_detects_a_planted_read_site`). A vacuous gate would have been the
[[VACUITY==PASS]] failure.

**C4 status.** The first `--full-inflate` attempt returned **rc=2 on BOTH archives** — the
harness had no bare `python` on PATH — and the control correctly reported **FAIL** rather than
passing silently. Fixed with a per-launch exec-wrapper shim (never a symlink, per
[[python-shim-must-be-exec-wrapper-never-symlink]]); the re-run was still in flight at
memo time. **C4 is therefore OWED, not claimed.** C1–C3 stand on their own: C2 already
establishes that the restored carrier body is the shipped carrier body byte-for-byte, so the
render is a pure function of identical inputs.

---

## 3. Order of operations — MEASURED, and it commutes

The chain applies both the rider (carrier section) and jg2's token edits (tail section). The
two orders were built on the real bytes and compared:

* **edits → rider**: rider applied to jg2's retained `candidate_jg1_3pair.zip` (176,450 B).
* **rider → edits**: jg2's edited tail spliced into the rider-applied pointer.

The second order needed no re-run of jg2's expensive encode, because jg2's tail is a pure
function of (token field, edits) and never reads the carrier. **That was not assumed** — the
probe first verifies jg2's candidate differs from the pointer in the `tail` section and
nothing else, and refuses otherwise.

| | archive bytes | sha256 |
|---|---:|---|
| order A (edits → rider) | 176,267 | `2032330303d43931…` |
| order B (rider → edits) | 176,267 | `2032330303d43931…` |

**`orders_commute: true`, delta 0 B.** Both parse through the real receiver to identical
`carrier_blob` and `token_stream` sha256. **Either order is correct for the harvest chain.**

**Composition is exactly additive** (`additive: true`): jg2's 3-pair edits COST 30 B, the
rider SAVES 183 B, composed = **−153 B vs the pointer**, `ΔS = −1.0188e-4`.

---

## 4. What this does to the sub-0.15 margin — stated plainly

`jg3` rung n=12 projects `S = 0.149987` and adds the rider to reach `0.149802`. With the
measured rider that becomes **`0.149987 − 0.000122 = 0.149865`**. Still sub-0.15, and the
rider is still decisive — the n=12 projection sits only `1.3e-5` under the line, so the rider
is most of the margin. But **the chain has `6.3e-5` less headroom than it was planning with**,
and any downstream memo quoting −1.85e-4 should be corrected to **−1.2185e-4** before it is
used to price a stopping rule. Per [[a_delta_without_its_baseline_is_unanchored]], quote the
rider against the body it was measured on: 183 B on `7ce46fd7…`.

**The rider re-measures itself on whatever body it is handed.** The 183 B is not a constant to
carry forward — it is the archive-layer delta on this body, and the tool prints the realised
figure for the body it actually ran on. On the edited candidate it also measured 183 B, which
is evidence the figure is stable across a tail edit, not proof it is stable across a carrier
re-solve.

---

## 5. The receiver change, and why it costs zero bytes

The rider needs a decoder. `inflate.py` and the runtime tree are **FREE** under contest rule
118 — the rate term charges `archive.zip` bytes only, and the rider decoder is generic
algorithm carrying no video-derived content. The tool emits a complete rider runtime tree:

* `runtime/rr5_arith_basis.py` — a **byte-identical copy** of `src/tac/rr5_arith_basis.py`
  (sha256 recorded in the receipt, asserted by a test), so encoder and decoder run the same
  code and the adaptive models cannot drift apart.
* `runtime/residual_archive.py` — two anchored insertions: `SZ1_RESERVED_KNOWN_BITS`
  `0x07 → 0x0F`, and a `reserved & 0x08` branch that restores the shipped carrier body
  immediately after the CK2 hook and before the packed-CAP1 framing arithmetic. Both patches
  fail closed if their anchor is missing.
* `inflate.py` — `ARCHIVE_SHA256` / `ARCHIVE_BYTES` re-pinned to the rider archive.

Reserved bit `0x08` follows the CK2/SZ1 precedent exactly. Old bodies are unaffected; a body
carrying the bit is refused by an unpatched receiver, which is the correct fail-closed
direction.

---

## 6. Custody

All under `/Volumes/APDataStore/pact/ddm_rr5/retained/`:

| artifact | bytes | sha256 (16) |
|---|---:|---|
| `pointer_rider/rider_runtime/archive.zip` | 176,237 | `eeef9e521ed75a30…` |
| `pointer_rider/RR5_RIDER_RECEIPT.json` | 3,477 | `8f448c98cab55fd4…` |
| `compose/order_b_rider_then_edits.zip` | 176,267 | `2032330303d43931…` |
| `compose/RR5_COMPOSE_REPORT.json` | 2,375 | `e9e63edde75a3cbe…` |

Container search ran the **SEALED** `UP3_DECLARED_OPTIONS` space
(seal `7fee75026d8a300e…`, 8 configs) rather than an ad-hoc list, so the seal still describes
what ran. The search is nearly flat — 6 of 8 options tie at 176,237 B, two at 176,238 — so
**ties go to the incumbent** container shape. Without that rule the rider would have flipped
the CK2 carrier plane ON for a 0 B "gain"; the first draft of this tool did exactly that.

---

## 7. The one-command harvest invocation

```
.venv/bin/python tools/ddm_rr5_rider_apply.py apply \
    --archive <candidate>.zip \
    --runtime /Volumes/APDataStore/pact/ddm_up3/candidate_runtime \
    --out-dir /Volumes/APDataStore/pact/ddm_rr5/retained/<tag> \
    --expect-sha256 <sha256 of that candidate>
```

It refuses unless the input hashes to `--expect-sha256`, refuses unless the identity control
reproduces that input, refuses on any losslessness control, and writes both the rider archive
and a complete rider runtime tree. Add `--full-inflate` for the end-to-end frame proof when
there is wall-clock to spend. Since the orders commute, run it **last**, after the token
edits, so it sees final carrier bytes.

---

## 8. What is OWED

1. **C4 end-to-end frame identity** — the `--full-inflate` re-run was in flight at memo time.
   Not claimed until its two frame sha256 match.
2. **ra1's 48 B `basis_scales` leg** — the half of the chain's −1.85e-4 this arm did not
   reproduce. It is a separate claim on a separate field and needs its own round-trip receipt
   before anyone adds it back to the rider's price.
3. **Correct the inherited figure** in `jg2` clause 6, `jg3`, and `tx1` §#3 from −1.85e-4 to
   **−1.2185e-4 (183 B on `7ce46fd7…`)**. Per
   [[corrections_land_in_bodies_headlines_keep_the_stale_number]], the headline needs the
   rewrite as much as the body.
