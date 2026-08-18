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
| 3 | `fx2_a__tuned` (sz1 composed split) | 179,930 | `debb025f45bb42e3…` | **0.15771357797660338** | **ACTIVE, HOLD** |

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

## Reproduction at generation 3

Status: PENDING_REBIND. `experiments/ddm_pq2_compress_e2e.py` asserts the
generation-2 hashes; the sz1 chain (fx2 byte-close driver + split builder)
must be re-bound under the same fail-closed assertions before the strict chain
can claim reproduction green. Build-time determinism repeat: byte-identical.
