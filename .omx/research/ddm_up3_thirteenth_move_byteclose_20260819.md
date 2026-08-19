# ddm_up3 — the byte-close was never blocked. Both blockers were one missing transform.

- **arm** `ddm_up3` (task #1136, up2 owed-item 7 / the byte-close of the converged
  thirteenth-move pose candidate)
- **date** 2026-08-19
- **axis** `[macOS-CPU advisory, frozen CPU-torch PoseNet + DALI GT]` · `score_claim=false` ·
  `promotable=false`. **Pointer UNMOVED** at contest-CUDA `0.15659459685822907`.
  This arm fired no Modal job. MAIN owns the T4 slot.
- **cost** $0.
- **code** `experiments/ddm_up3_carrier_splice.py` · `experiments/ddm_up3_byteclose_gate.py`
  · `src/tac/tests/test_ddm_up3_carrier_splice.py` (32 tests) · commit `17c801b134`
- **store** `/Volumes/APDataStore/pact/ddm_up3/` (`UP3_RETENTION_MANIFEST.json`, 101 files)
- **seal** `retained/CANDIDATE_SEAL_up3_r1.json`
  sha `89de991f84e64a27155ba1e67c455c82cbc72c7d58703fdb3817818d49ee82eb` — **SEAL_VALID**

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at seal time) ·
`ddm_up2_shipping_object_pose_solve_20260819.md` (§4, §7, §8 owed-7 — the arm I inherit) ·
`/Volumes/APDataStore/pact/ddm_up2/{identity_control/BYTE_CLOSE_BLOCKER.json,retained/**,
UP2_RETENTION_MANIFEST.json}` · `ddm_up1_uncapped_pose_solve_20260819.md` ·
`/Volumes/APDataStore/pact/ddm_to1/{TO1_RETENTION_MANIFEST.json,compile/TO1_TAIL_OVERRIDE.json}` ·
the SHIPPED receiver at `.../to1_tail_override_r1/runtime/{residual_archive.py,carrier_repack.py,
entropy/coefficient_ar1_codec.py,entropy/coefficient_predictor.py}` and `upstream/modules.py`
(read at source, not quoted from memory) · `experiments/ddm_sa2_compile_candidate.py`,
`ddm_sa3_rebase_sz1.py`, `ddm_rx1_rate_representation_attack.py`, `ddm_t1h_compose_pass1.py`
(the canonical write path, mapped independently and agreeing).

---

## ANSWER FIRST

1. **The candidate is byte-closed, sealed, and validated.** `archive.zip` sha
   `7ce46fd7a845d598…`, **176,420 B — exactly the pointer's byte count, ΔB = 0.**
2. **Both of `ddm_up2`'s blockers were the SAME missing transform, and neither was a
   container limit.** The stored carrier is 2-plane byte-interleaved (`reserved` bit
   `0x04`); the receiver un-interleaves it at `residual_archive.py:188` *before* it reads
   any offset. `ddm_up2`'s borrowed tool never did. On the interleaved buffer byte 139
   reads **177** (up2's "body-specific literal") and the Rice payload is genuinely not
   locatable (up2's "two representations"). After the un-interleave, byte 139 reads
   **8** — the true `k_base`, giving exactly the shipped `[9,9,9,8,8,9,9,9,9,9,9,9]` — and
   the payload sits at the derived offset `6+96+40+basis_bytes`. **`packed[139]` was the
   RECEIVER'S OWN offset and it was right; the buffer was wrong.** A second defect
   compounded it: `PACKED_CAP1_SECTION_BYTES = 22_183` is a stale pin, and this body's
   packed portion is **22,178**.
3. **The identity control that up2's tool failed now passes byte-identically.** Splicing
   the SHIPPED codes back through my chain reproduces the pointer archive
   `50e561454b23026d…`, 176,420 B, bit for bit. Parse-back of the candidate recovers the
   970 solved coefficients with **max |Δ| = 0**, and double compile is byte-identical.
4. **The realized d_pose ON THE RECOMPILED BYTES is `7.649246787072966e-06`** (n600, DALI
   GT, frozen CPU PoseNet) — `ddm_up2`'s predicted `7.649247e-06` to 8 significant
   figures. The base control on the same instrument reads `7.76948388629175e-06` against
   the T4 receipt's `7.77e-06`. **The pose gain survives the byte-close exactly.**
5. **CORRECTION to `ddm_up2` §4/§ANSWER-4: ΔB = 0 was NOT free, and as up2 shipped it the
   candidate would have COST 48 bytes.** up2 priced the *uncompressed* Rice payload
   (9,759 B, unchanged) and concluded `delta_bytes = 0`. The score charges `archive.zip`,
   which is downstream of brotli — and brotli compresses the perturbed payload **48 bytes
   worse** at the shipped settings, at every `lgwin`. That is `+3.196e-05` of rate,
   **47% of the entire pose gain.** I recovered it (§3) but the general claim "the CAP1
   Rice stream absorbs small perturbations for free" is false as stated: it is free in the
   *payload*, not in the *archive*. A one-coefficient flip already costs +3 B.
6. **A defect that would have wasted the T4 fire, caught by the seal validator, not by
   me.** The staged `inflate.py` pins `ARCHIVE_SHA256` and refuses any archive that does
   not match. My candidate would have been refused by its own receiver at decode time.
   Fixed with the canonical `tac.candidate_seal.repin_receiver`. **The seal is not
   paperwork — it caught a real failure.**

---

## 1. The layout, read from the receiver rather than guessed

The receiver decodes the shipped body every day, so its parse code IS the specification.
The whole chain is exact and total, so it runs backwards:

```
archive.zip                 ZIP_STORED, one member "p"     residual_archive.py:449-452
  -> RX1 header + 3 streams                                 :161-178
  -> brotli(carrier_stream)                                 :185
  -> CK2 2-plane un-interleave      if reserved & 0x04      :188-189, :674
  -> packed CAP1 section, portion DERIVED from its own u24s  :195-202
     _restore_packed_cap1_metadata()                        :124-149
  -> canonical CAP1 (_restore_cap1, CAP_FIELDS reorder)     :361-381
  -> CPR1 (decode_cap1)                    coefficient_ar1_codec.py:86
  -> Rice -> unzigzag -> restore_ar1_bias -> int12 codes
```

Measured field map of the packed section on this body (every offset **derived** from the
format's own widths, then asserted against the shipped bytes):

| field | span | value |
|---|---|---|
| basis_bits u24 | `[0:3]` | 98,213 (12,277 B) |
| residual_bits u24 | `[3:6]` | **78,065 (9,759 B)** |
| scales | `[6:102]` | 96 B |
| factor_base | `[102]` | 146 |
| factors 12x7b | `[103:114]` | |
| biases 12x6b | `[114:123]` | |
| lengths 32x4b | `[123:139]` | |
| **k_base** | `[139]` | **8** (up2 read 177) |
| ks 12x1b | `[140:142]` | -> `[9,9,9,8,8,9,9,9,9,9,9,9]` |
| basis | `[142:12419]` | |
| rice payload | `[12419:22178]` | 9,759 B |
| selector tail | `[22178:22187]` | 9 B |

The 9,759-byte payload up2 found "verbatim in the CAP1 blob at offset 12467" is the same
object seen after the `CAP_FIELDS` reorder adds the 8-byte `CAP1\x01\x00\x00\x00` prefix
and un-packs the metadata (+40 B): `12419 + 8 + 40 = 12467`. There were never two
payloads — one payload, two framings.

**Nothing in my splicer is a positional literal.** `PACKED_METADATA_OFFSET` is
`6 + 8*dimensions`; the 40-byte metadata block's spans come from the field widths
`(1,8) (12,7) (12,6) (32,4) (1,8) (12,1)`; the derivation independently lands `k_base` at
**139**, which a test asserts. The forward packer is checked against the shipped bytes
before it is trusted.

## 2. The controls

Every one is executable (`src/tac/tests/test_ddm_up3_carrier_splice.py`, 32 tests, no
skips — I checked the denominator, because a skipped control reads as a pass).

| control | result |
|---|---|
| **identity** — shipped codes -> archive | **byte-identical to `50e561454b23026d…`**, 176,420 B |
| forward packer -> shipped `packed_metadata` | identical |
| `forward_ar1_bias` inverts `restore_ar1_bias` | exact, on the shipped codes AND on random int12 |
| re-encode shipped codes -> Rice payload | identical bytes, identical 78,065 bits, identical ks |
| **roundtrip** — solved codes -> archive -> codes | **exact, max \|Δ\| = 0** |
| **double compile** | byte-identical |
| brotli q11/lgwin24 -> shipped carrier stream | identical (parameters DISCOVERED, not assumed) |
| CK2 forward/inverse | round-trips at every parity incl. 0, 1, odd |
| receiver `read_residual_archive` on the candidate | parses; semantic/hpac/token **byte-identical to the pointer** |

**Mutation controls** (does the detector bite?): a one-coefficient flip breaks the sha and
the payload identity; `ks+1` breaks the packed metadata; skipping CK2 breaks the stream;
parse-back rejects the wrong codes; a corrupted AR(1) model makes `build_archive` refuse
its own output. All five fire.

The builder now **verifies itself** — it parse-backs before returning and raises rather
than hand out bytes it has not proven. That was a second-review finding, not a design
I got right first time.

## 3. The rate leg — where up2 was wrong, and how I got the 48 bytes back

Re-encoding the solved codes gives **identical ks, identical 9,759-byte payload length**,
78,065 -> 78,072 bits. So the uncompressed section is the same length, exactly as up2
priced. But the archive stores `brotli(section)`, and:

| container | base stream | solved stream | Δ archive |
|---|---:|---:|---:|
| shipped shape (CK2 on, q11/lgwin24) | 22,143 | 22,191 | **+48 B** |
| CK2 on, q10/lgwin16 | 22,162 | 22,167 | +24 B |
| **CK2 off, q10/lgwin16** | 22,162 | **22,143** | **0 B** |

+48 B is `+3.196e-05` of score — nearly half the pose gain. The receiver calls a bare
`brotli.decompress` (`residual_archive.py:91`) and reads the CK2 interleave from
`reserved` bit 2 (`:188`), so **both are encoder-only choices**: the decoded bytes are
identical either way. The splicer therefore searches a DECLARED, ORDERED option list and
keeps the FIRST minimum, so the shipped shape wins every tie and the identity control
stays exact. The candidate ships **CK2 carrier OFF, brotli q10/lgwin16, `reserved` 0x2**.

Honest labelling: this is a container choice, not a smaller payload. The candidate
differs from the pointer in two ways — the pose codes (the point) and the container shape
(the rate recovery). Both are receiver-supported; the second is proven by the candidate
parsing through `read_residual_archive` and by the seal's own validation.

**The general lesson, against my own convenience:** never price a byte delta on the
uncompressed section when the score charges the compressed archive. This is the
denominator genus — up2's pricer was sensitive and correct about the payload, and still
gave the wrong archive answer.

## 4. The measured legs

| leg | pointer (T4 receipt) | candidate (measured) | Δ |
|---|---:|---:|---:|
| d_pose (n600, DALI GT) | 7.77e-06 | **7.649246787072966e-06** | −6.876e-05 in S |
| d_seg | 0.00030309 | 0.00030309 (unchanged) | 0.0 |
| archive bytes | 176,420 | **176,420** | 0 |
| rate term | 0.11747083650981346 | 0.11747083650981346 | 0.0 |

**Pose.** Measured from the carrier state loaded back out of the CANDIDATE archive, not
from the solver's in-memory codes. Base control on the same instrument:
`7.76948388629175e-06` (T4 prints `7.77e-06`), so the instrument reproduces the shipping
row independently at full field.

**Seg.** Structural: `upstream/modules.py:108` is `x = x[:, -1, ...]`, and the candidate's
semantic/hpac/token sections are byte-identical to the pointer's, so SegNet's input is the
same bytes. Measured anyway through the REAL `DistortionNet.compute_distortion` on 48
seeded-random pairs (never a prefix, per `ddm_na2`/`ddm_bp2`): seg leg **exactly 0.0**, **0
of 9,437,184 argmax pixels changed** — while frame 0 really moved (1,777,321 pixels, max
|Δ| = 2). The control is not vacuous: the input changed and the output did not.

**Score arithmetic** (all three legs held or measured, none assumed):

* net ΔS vs the T4 8dp receipt: **−6.876309991788766e-05**
* net ΔS vs the realized base: −6.847033967330614e-05 (up2 predicted −6.846805e-05)
* projected pointer: **0.1565258337583112**
* the T4 8dp report should print `d_pose 0.00000765` (from `0.00000777`), implying
  S = `0.15652626435208142`
* clears the **−3.5e-06** admit bar by **19.6x**; clears the summed two-row 8dp pose bound
  (`5.694696952803294e-06`, and bounds ADD) by **12.1x**. Resolvable.

## 5. On the GT-lineage question my charter raised

My charter asked whether seg GT is lineage-stable between the DALI and PyAV decodes before
trusting an advisory `d_seg` check. **I did not need to answer it, and I should say so
rather than assert a convenient answer.** The seg leg here is measured as candidate-vs-
pointer on the SAME decode, which is a difference of two quantities that share whatever GT
they are compared against — so the lineage cancels by construction. Whether the seg GT
itself differs between decoders is still **OPEN and unowned**, sister of up2's §8 item 1.

## 6. My own round-1 adversarial review

1. **Is the identity control circular?** It could have been: if I had derived the brotli
   parameters *from* the shipped stream and then "proved" I reproduce it. I did search for
   them — but the search is over a 3x7 grid of standard settings and exactly one member
   reproduces 22,143 B *and* the exact bytes, and the independent map of the canonical
   write path (`ddm_sa2_compile_candidate.py:64-65`) names q11/lgwin24 for the same
   reason. Two routes, same answer.
2. **Does the container search launder a rate claim?** It is the honest risk. Mitigation:
   the option list is a fixed declared constant, the search is deterministic with a
   documented tie-break, every option must decompress back to the exact bytes, and I
   report BOTH numbers (+48 like-for-like, 0 searched). A reader who distrusts the
   container change can price the candidate at +48 B and it still clears the bar by 10.4x.
3. **Am I quoting a prefix?** No. Pose is full-field n600. The seg sample is
   seeded-random n48 via `up2.select_pairs`, which refuses a prefix below n600.
4. **Does the T4 CUDA decode change the answer?** Unaddressed here and still live —
   up2 §7 flagged it and it stands: the solve exploits fine sub-LSB rounding structure and
   the CUDA decode differs from the CPU decode in exactly that structure. The aggregate
   bound (~5e-9 in d_pose) is ~20x smaller than the gain, so it should survive, but
   **"gain does not transfer" remains a live falsifier of the T4 row, not a formality.**
5. **Is the CK2-off path exercised?** Yes — it is the `reserved == 0` branch the rr4
   lineage uses, and the candidate parses through `read_residual_archive` on it. But the
   full local advisory (inflate + evaluate) was still running when this memo landed;
   its receipt is the last end-to-end proof and is owed below.
6. **What did I get wrong?** Two things, both caught by review rather than by design: the
   builder could return unverified bytes, and my first seg-leg implementation fed the
   scorer the wrong tensor layout with no weights loaded. Neither reached a number.

## 7. Owed, with owners

1. **Fire the T4 row.** MAIN owns the slot. The seal is validated and names its own
   command. Nothing else is needed from this arm.
2. **Finish the local advisory receipt** (inflate + `evaluate.py --device cpu` on the
   candidate). Launched here via `tools/fire_local_advisory.py`, attempt
   `/Volumes/APDataStore/pact/ddm_up3/advisory/attempt_0001`. Expected per up2 §4d: seg
   leg identical, and CPU-axis `d_pose` slightly WORSE (the two GT lineages pull opposite
   ways) — which is a quantitative prediction the receipt can falsify.
3. **Fix `ddm_t1h_compose_pass1.py` and `ddm_t1h_build_candidate_archive.py`.** Both miss
   the CK2 un-interleave and both use the stale `PACKED_CAP1_SECTION_BYTES = 22_183` pin.
   They are CORRECT for the rr4 body (`reserved = 0`, packed portion 22,174 + 9) and
   **silently wrong** for every ck2/to1-lineage body. Same defect in
   `compose_pass1.packed_rice_bit_budget`. **Unowned.**
4. **The stale pin is a class, not an instance.** `residual_archive.py:81` still defines
   `PACKED_CAP1_SECTION_BYTES` even in the to1 runtime, where the live path derives the
   portion instead. Any future tool that reaches for the constant inherits the bug. A
   gate that refuses a fixed-length read of the packed CAP1 section is the structural fix.
5. **Re-price every prior "the Rice stream absorbs this for free" claim** against the
   compressed archive rather than the payload (§ANSWER-5). up2 §4b's overlay-vs-re-encode
   verdict used payload bytes on BOTH sides, so its ordering survives; its magnitudes do
   not.
6. **Richer solve neighbourhood** (up2 §8 item 3) is now cheap to close the loop on: the
   byte-close is a function call, so any improved code set can be priced end-to-end in
   seconds. Still unowned.

## 8. Retained payload

`/Volumes/APDataStore/pact/ddm_up3/` — `UP3_RETENTION_MANIFEST.json` (101 files, sha256 +
bytes each). Headline: `retained/candidate_up3_pose_solved.zip`
(`7ce46fd7a845d598…`, 176,420 B) · `retained/candidate_up3_pose_solved_shippedshape.zip`
(`a5be29aad5e19e4b…`, 176,468 B, the +48 B like-for-like variant, kept because a
per-candidate payload is not just the winner's) · `candidate_runtime/` (the sealed,
re-pinned tree) · `retained/CANDIDATE_SEAL_up3_r1.json` · `retained/GATE_pose_{candidate,
base}_n600.json` + per-pair values · `retained/GATE_seg_n48.json` ·
`retained/CONTROLS_{container_search,shipped_shape}.json` · `retained/up2_{solved,base}_codes.npy`
· both tools. The 3.66 GB advisory raw decode is excluded as deterministically rebuildable
from `archive.zip` + `inflate.sh`, per the certify-or-block rule.
