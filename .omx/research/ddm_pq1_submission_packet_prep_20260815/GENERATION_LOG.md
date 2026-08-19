# Packet generation log

This packet's public artifacts are refreshed in place when the candidate changes,
per step 4 of `SWAP_PROCEDURE.md`. Prior generations are preserved in git history
and in retained custody; they are never overwritten in their retained stores.
This log exists so a reader can tell, at a glance, which candidate the files in
this directory currently describe.

| Gen | Candidate | Archive bytes | Archive SHA-256 | Measured `[contest-CUDA]` | Status |
|---|---|---:|---|---|---|
| 0 | `e480b_v2_s1p25_c1p0_brotli_q10` | 183,502 | `e3e6f440b45bbb92…` | 0.1600920261571558 | superseded, retained |
| 1 | `hv1_ep0634` | 182,759 | `80d9c8c6fdc72caa…` | 0.15959729295498598 | superseded, retained |
| 2 | `rr4_free_corrector_v2_reencode` | 181,161 | `35ac2b9beb7e6fa8…` | 0.15853325034789678 | superseded, retained |
| 3 | `fx2_a__tuned` (sz1 composed split) | 179,930 | `debb025f45bb42e3…` | 0.15771357797660338 | superseded, retained |
| 4 | `ck1_composed_rebased_r4` (SM3R mode-6 row-prune + frame-0 pose compensation) | 177,182 | `35c318d541d70370…` | **0.15710198138050818** | **ACTIVE, HOLD** |

## What changed at generation 2

Generation 2 is a lossless entropy re-encode of generation 1. Seven of the eight
parsed sections are byte-identical to generation 1; only the RC64 token stream
changed, 112,110 to 110,512 bytes. The decoded token field is unchanged
(SHA-256 `9ba2e52b3096…`), so `d_seg` and `d_pose` carry over exactly and the
whole delta is rate: −1,598 bytes, ΔS −0.0010640426070892.

## Custody note carried into generation 2

`GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` inside the generation-2
runtime tree were inherited from the generation-1 source tree and describe a
182,759-byte archive, not the 181,161-byte archive they sit beside. They are
stale **labels**, not stale proofs. This packet binds `RESULT_build.json`,
`RESULT_receiver_build.json`, and `RESULT_parseback_v2.json` instead, and a
`CUSTODY_SUPERSEDED.json` was written beside the inherited pair naming the real
sha and pointing at the three authoritative receipts. The inherited files were
deliberately NOT regenerated: the pinned runtime-tree hash is computed over that
directory, so regenerating them would break replay against the value recorded in
the exact-authority row.

## Reproduction added at generation 2

`experiments/ddm_pq2_compress_e2e.py` rebuilds the archive from the retained
checkpoint and fails closed unless the bytes hash to the pinned value. Verified
2026-08-17: token stream and archive hashes both matched, determinism repeat
byte-identical.

## What changed at generation 3

Generation 3 composes two lossless rate moves over generation 2's decoded state:

1. **Token probability model** (fx2 candidate A): the RC64 token stream is
   re-encoded by a 13-context fixed-point integer log-odds mixer,
   110,512 → 109,801 bytes. The decoded token field is unchanged
   (SHA-256 `9ba2e52b3096…`).
2. **Semantic serialization split** (sz1): 8,284 bytes of raw interleaved fp16
   metadata in the semantic section are byte-planed (high-byte plane, then
   low-byte plane) before the container Brotli, −520 bytes. The receiver
   un-splits deterministically before parsing; decoded values are unchanged.
   Versioned in bit 0 of an existing reserved header byte — zero transmitted
   bytes; reserved == 0 keeps the exact prior code path.

The two moves touch disjoint byte ranges and composed with measured 0 B
interaction. Decode identity was proven at the byte level: worker output 0.raw
hashes identically between this row and the fx2 candidate-A row
(`9a6b75e5…`), so seg and pose carry over exactly. Net vs generation 2:
−1,231 bytes, ΔS −0.00081967237 → **0.15771357797660338** at 179,930 bytes.

## Custody note carried into generation 3

The generation tree's `GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json`
are inherited hv1-lineage labels (they describe a 182,759-byte archive). Per
the generation-2 precedent they are deliberately NOT regenerated — the
exact-authority row validated runtime-tree hash `0d0fc008d6a3…` over the sealed
tree containing those exact bytes. `CUSTODY_SUPERSEDED.json` beside them names
the real candidate and the authoritative receipt chain (sealed fire order +
`FIRE_MANIFEST.json` stage3b seal + r3 auth-eval JSONs).

## What changed at generation 4

**Generation 4 is the first candidate in this packet that is not a lossless
re-encode.** Generations 2 and 3 held the decoded state constant and provably
identical, so their entire improvement was rate and `d_seg`/`d_pose` carried
over exactly. This one changes decoded values on purpose and pays for the bytes
in both distortion terms:

1. **Semantic section, SM3R mode 6.** Three FiLM weight tensors keep only their
   two highest-L2 rows (row bitmask + compact kept-rows block); a per-tensor
   4-bit depth table drops `frame_embed.weight` and `blocks.0.film.weight` to
   3-bit codes while the rest stay at 4. Semantic stream 31,469 B, body
   36,130 B. The receiver recomputes the tensor selection mask and refuses on
   mismatch.
2. **Frame-0 pose compensation.** The re-quantization damages PoseNet (which
   reads the pair) while the semantic renderer produces only frame 1, so the
   frame-0 carrier lattice is re-solved at compile time. 6,713 of 7,200
   signed-int12 coordinates change; the compensation costs 41 archive bytes and
   cancels 99.98% of the leakage energy in the local solve.
3. **The generation-3 serialization split is OFF** (`reserved = 0`). The
   row-prune changes the semantic body length, and re-measured on the edited
   body the split is negative — two credits over the same redundancy do not add.
   Its receiver support ships and is inert.

The HPAC stream (13,515 B) and the tail (fx2 token stream + residual + table
codes, 109,897 B) are spliced byte-identically from generation 3.

Net vs generation 3: −2,748 bytes, ΔS −0.0006115966 → **0.15710198138050818** at
177,182 bytes. Against the intermediate keep01 row (177,576 B, 0.1571619225142182)
the legs are rate −2.6235e-04, seg +1.7400e-04, pose +2.8407e-05, net
−5.994113e-05 — about 23% of the rate credit retained.

## Custody note carried into generation 4 — the inherited receipt pair is GONE

Generations 1–3 all shipped `GENERATION_RECEIPT.json` and
`RECEIVER_PARSEBACK.json` inside the runtime tree as hv1-lineage stale labels,
deliberately unedited because the evaluated runtime-tree hash was computed over
them. Round-11 F2 then found that those two files carried **63 absolute local
`/Volumes/…` paths** onto a public surface, and that sanitizing them in place
would ship bytes the T4 row never evaluated under an unchanged green tree hash.

**The ck1 lineage does not contain either file.** The 32-row runtime manifest
does not declare them and they are not on disk, so the exposure is closed by
construction rather than by an edit that would have cost a T4 row. Verified with
the compliance checker's own `PRIVATE_SURFACE_RE` (text) and binary markers over
all 37 staged files: **0 hits**, and `public_scan_has_no_private_surface` is
GREEN in the generation-4 receipt at 38 files scanned.

## Reproduction at generation 4 — NOT re-verified

Status: **NOT_RE_VERIFIED_FOR_THESE_BYTES.** Generation 3's end-to-end
single-entry-point rebuild is a property of generation 3's bytes and does not
transfer. `experiments/ddm_pq2_compress_e2e.py` has not been re-run for this
candidate. What does exist: the compile receipt `SA3_REBASE.json`, which asserts
decoded-state identity **before** building (refusing rather than carrying a
compensation onto a changed lattice — this repository shipped that bug once),
and receiver parse-back `PASS` at `max_abs_code_deviation = 0` over the shipped
runtime. The owed item is an end-to-end rebuild of `35c318d5…` from pinned
retained inputs through one entry point.

## Reproduction at generation 3

Status: VERIFIED (2026-08-18). `experiments/ddm_pq2_compress_e2e.py` was
re-bound via a recipe-declared split stage (`RECIPE_sz1_composed.json`): the fx2
token re-encode runs through the generic encoder (corrector selected by the
recipe), the pre-split archive is asserted against `9de0f6db…`/180,450 B, the
canonical sz1 split builder then repacks the container with its own decode
bit-identity, base-unchanged and token-verbatim proofs, and the final archive
is asserted against `debb025f45bb42e3…`/179,930 B. End-to-end run from the four
sha-verified retained inputs: all assertions green, determinism repeat
byte-identical. Receipt: `ddm_pq2/e2e_sz1_composed/RESULT_pq2_e2e.json`.
