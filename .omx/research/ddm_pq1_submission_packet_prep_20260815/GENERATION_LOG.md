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
| 2 | `rr4_free_corrector_v2_reencode` | 181,161 | `35ac2b9beb7e6fa8…` | **0.15853325034789678** | **ACTIVE, HOLD** |

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
