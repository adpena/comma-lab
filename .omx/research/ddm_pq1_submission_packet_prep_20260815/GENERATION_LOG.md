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
| 4 | `ck1_composed_rebased_r4` (SM3R mode-6 row-prune + frame-0 pose compensation) | 177,182 | `35c318d541d70370…` | 0.15710198138050818 | superseded, retained |
| 5 | `jg5_joint_waterfill_455` (joint admission waterfill + carrier re-solve on own renders) | 180,625 | `f3bce5d259a08183…` | 0.14839100138338618 | superseded, retained |
| 6 | `ddm_rc2_object_b_clean_port_rr5_rider` (generation-5 body + RR5 lossless carrier rider + clean C port of the free corrector) | 180,456 | `df7fd266e1b7488c…` | 0.14827847122030852 | superseded, retained |
| 7 | `ddm_afr1_tile48_groupbin8` (five lossless coder/container moves after rc2; final move is tile48×groupbin8) | 180,002 | `cbb8d928a8ccdd3f…` | **0.14797617125559104** | **ACTIVE, HOLD** |

## What changed at generation 7

Generation 7 moves the packet from rc2 to AFR1 through five exact-pointer
states, all lossless re-encodes of the same decoded CUDA object:

| Pointer state | Archive bytes | Exact `[contest-CUDA]` score | Change from prior |
|---|---:|---:|---:|
| rc2 | 180,456 | 0.14827847122030852 | packet generation 6 |
| fx5 e1 | 180,386 | 0.14823186109359 | −70 B |
| dx2 | 180,368 | 0.14821987563243377 | −18 B |
| gb1 | 180,215 | 0.14811799921260607 | −153 B |
| lb1 | 180,083 | 0.14803010583079396 | −132 B |
| **AFR1** | **180,002** | **0.14797617125559104** | **−81 B** |

The total change is −454 B and −0.00030229996471747844 score units.
Both distortion contributions remain `0.020139` and
`0.007981227975693965`; the entire delta is the rate change
`25 × (−454) / 37,545,489`. AFR1's own T4 authority row emits the same
3,662,409,600-byte raw CUDA output as rc2, SHA-256 `6bf8acf8…`.

The runtime expands from 36 to 38 authority rows. Its enumerated tree pin is
`6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58`.
No AFR1 `[contest-CPU]` score exists: the leg is RECORD-WITH-REASON and no
score is inherited from another archive. The end-to-end rebuild VERIFIED
label also remains scoped to generation 3; AFR1 is explicitly not re-verified
by that entry point.

## What changed at generation 6

Two changes, both **decode-identical**, and that is the whole story of this
generation.

1. **The RR5 lossless rider.** The carrier body is re-encoded under an adaptive
   arithmetic basis; reserved header flag `0x08` engages `restore_carrier_body` on
   the receiver, restoring a 22,316 B carrier blob. Worth **−169 bytes**.
2. **The clean C port of the free corrector.** `runtime/f26_corrector_native.c` and
   `runtime/native_free_corrector.py` replace the Python implementation. Worth
   **zero bytes** and **961.2 s** of inflation wall.

Neither touches a decoded value, and this is measured rather than asserted: on the
contest-CUDA T4 axis, generation 5 and generation 6 emit **byte-identical** n600
inflated output, both hashing to
`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883` at 3,662,409,600
bytes. Both distortion legs are therefore exactly equal — `d_seg 0.00020139`,
`d_pose 0.00000637` on both rows — and the entire score delta is rate:
`25 × (−169) / 37,545,489 = −1.1253016e-04`, which is exactly what the two
recomputed scores differ by.

The runtime tree changed shape as well as content: 33 rows became **36**
(`fdd57749…`), because the rider and the native corrector add receiver modules.
That is why generation 5's authority row could not be carried and this generation
bought its own.

**The decode budget is the part that is not marginal.** Generation 5 charged
1,471.3 s on the T4 axis, over both ends of the projected CI residual window
[822, 1302] s. Generation 6 charges **498.5 s** — a PASS at the binding cold-cache
corner with 323.5 s of margin. On the CPU axis both generations are infeasible;
generation 6's own measurement is a kill at the 1,800 s wall with token decode alone
at 2,427.2 s.

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


## What changed at generation 5 — the first sub-0.15 row, and the first that SPENDS rate

Generation 5 is the first candidate in this packet to measure **below 0.15** on
`[contest-CUDA]`, and the first to move in the opposite direction on bytes.

**The archive is LARGER: 180,625 B, +3,443 B against generation 4.** A reader
comparing byte counts alone would read that as a regression. It is not. The legs
against generation 4 are rate **+2.2926e-03**, seg **−1.0170e-02**, pose
**−8.3353e-04**, for a net of **−8.7110e-03**. Rate is spent; both distortion legs
are bought.

**The mechanism.** The prior composition applied all 573 seg token edits and then
re-solved the pose carrier. Measured, that direction is seg-descending but not
pose-null: the edits cost roughly 13× more pose than they bought in seg, and the
composed result scored far worse than either part suggested. Generation 5 stops
composing two finished candidates and solves the admission jointly:

1. Edit admission is swept over a Lagrange multiplier on pose damage. **455 of the
   573 edits are admitted**; the other 118 are dropped and those pairs keep the
   prior carrier's codes.
2. The frame-0 pose carrier is re-solved against the **candidate's own edited
   renders** rather than the base renders, under a derived materiality stop rule.
   **600 of 600 pairs stopped on `no_improving_step` with zero budget hits**, so
   the stopping criterion was never the binding constraint.

**Sign determinacy, stated the correct way.** The net is a delta between two
independently 8dp-rounded rows, so both bounds apply and ADD: 3.336608e-06 +
3.632965e-06 = 6.969573e-06. The net is **1249.86×** that summed bound. (Dividing
by one row's bound alone — the round-12 F1 defect — would overstate by about 2×;
at this magnitude the conclusion is unaffected, but the arithmetic is stated
correctly regardless.)

**The display trap.** The evaluator prints `Final score: … = 0.15`. That is a
2-decimal display that rounds UP across exactly the boundary this candidate sits
on. Generation 5's claim was the value recomputed from components,
`0.14839100138338618`, with a worst-case 8dp bound of `3.633e-06` — about 443×
clear of 0.15. (Generation 6 inherits the trap and clears it by 474×; see its own
section above.)

## Custody note carried into generation 5

The source runtime tree carried **27 `__pycache__/*.pyc` files and 2 AppleDouble
sidecars** that are not in the 33-row runtime manifest. None of them reached the
packet: generation 5 was staged with `tools/stage_contest_submission_packet.py`,
which selects files **by the manifest** rather than by globbing, so contamination
is excluded by construction rather than by a cleanup pass. The excluded classes
are reported with exact paths in `STAGING_RECEIPT.json` — a census that drops a
class silently is the generation-4 defect, and this one names its denominator.

A second, separate contamination did occur and was caught: writing the four public
docs onto the ExFAT volume caused macOS to create **51 new AppleDouble sidecars**
across the generations tree. `tools/packet_census_guard.py` caught all 51 with
exact paths; they were purged and the census re-run clean. **The ordering law this
establishes: purge AppleDouble immediately before the census and the compliance
re-buy, because any write to the volume re-creates them.**

The harvested authority receipts were persisted as **Python `bytes` reprs** — the
files literally begin with `b'` and carry `\n` as two characters. They were decoded
and each decode was proved round-trip exact before use
(`HARVEST_DECODE_RECEIPT.json`). The shipped `report.txt` therefore carries the
evaluator's own text, not a re-authoring.

## Reproduction at generation 5 — NOT re-verified

`experiments/ddm_pq2_compress_e2e.py` has **not** been re-run for these bytes. The
generation-3 VERIFIED label belongs to generation 3 and does not transfer. What
exists for this candidate: the candidate seal binding archive to receiver
(`SEAL_VALID`), the staging proof that this directory is byte-identical to the
evaluated tree with the tree hash **re-derived from the staged rows**, and the
authority receipt itself.

## Generation 7 doc regeneration — 2026-09-02 (F1)

The frozen generation-7 packet carried four documents written in the
refusal era: `compress.py` was the pre-ce1 snapshot whose
`refuse_if_not_expressible` path rejected AFR1 by exact SHA, and
`COMPRESS.md`, `README.md`, `report.txt`, and
`BORROWED_SUBSTRATE_ACCOUNTING.md` all described that refusal as the honest
reproduction result. That description became FALSE on 2026-09-01 when
`ddm_ce1` ran two complete `AFR1_CHAIN` runs from the retained pinned base
and each rebuilt the exact 180,002-byte archive byte-identically
(receipt `RESULT_pq2_e2e.json`; memo
`ddm_ce1_afr1_compress_chain_20260901.md`).

**What changed.** `compress.py` was replaced with the live repo entry point
`experiments/ddm_pq2_compress_e2e.py` (byte-identical copy, sha
`5fe3a8ca94c4…`; the superseded file retained beside it as
`.compress_py_pre_ce1_superseded_20260902`). `COMPRESS.md` was rewritten to
document the five-stage chain with its pinned per-stage output SHAs and the
real CLI (`--stage chain --chain afr1 --chain-repeats 2`, verified against
the entry point's argparse, never invented). `README.md`'s reproduction
boundary and verify-locally block now describe the chain; `report.txt`'s
reproduction paragraph likewise. The accounting file was amended
APPEND-ONLY per Catalog #110/#113 — both "not re-verified" qualifications
keep their original text with a dated supersession note beneath.

**What did NOT change.** `archive.zip` (sha `cbb8d928a8cc…`, 180,002 B) and
the 38-file runtime tree are untouched; `MANIFEST.sha256` verifies 49/49 OK,
and it covers the runtime tree and cpr1 rows only — no packet doc is inside
its scope, so custody is unaffected by construction.

**Receipt re-bought same batch** (r5 freshness law — an edit to a scanned
surface invalidates the prior receipt):
`receipts/pre_submission_compliance.gen7.r3.json` (partial flag set) then
`…r4.json` with the r2-comparable flags. r4 is **79 GREEN / 6 RED** against
r2's 80 / 7: **zero new reds**, one red cleared
(`hosted_archive_manifest_supplied`, because r4 passed
`--archive-manifest-json`). The six survivors are the known set — no CPU
auth-eval row, the Brotli network-install property, the `tac` import in
`compress.py`, runtime-tree-vs-auth-eval, raw-promotion policy, and the
public-surface scan.

Publication state unchanged: **PREPARED HOLD, NOT PUBLISHED.**
