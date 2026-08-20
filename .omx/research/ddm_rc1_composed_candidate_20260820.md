# ddm_rc1 — the COMPOSED candidate: native port × CPR1 rider, merged and decode-proven

`date_utc: 2026-08-20` · `owner: MAIN` · `axis: [byte-exact + merge-proof — no scorer run]`
· `score_claim: false` · `promotable: false` · `frontier_moved: false` · cost **$0**

## THE ANSWER, FIRST

The rr8 native corrector and the rr5 CPR1 rider **compose cleanly into ONE shipping object**, and
the composed tree is decode-proven. Assembled at
`/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed` (37 files, archive
`df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` @ 180,456 B).

Expected on the exact axis: **S 0.14827847122030854** (−1.125302e-04 vs the jg5 pointer, rate-only)
at an inflate wall near **464 s** (vs jg5's 1,419.9 s). One T4 row buys both.

## THE COLLISION, AND WHY IT WAS NOT A BLOCKER

My rr8 memo said the two transforms are "orthogonal in mechanism and compose." Orthogonal in
mechanism does NOT imply disjoint in files, and the diff proved it: **both edit
`runtime/residual_archive.py`.** The port also owns `inflate.sh`; the rider owns `inflate.py`.

Measured at line level, the edits are **disjoint**:

| | base-line regions touched |
|---|---|
| PORT (rr8) | 550 (insert) · 569–570 (replace) · 659 (insert) |
| RIDER (rr5) | 189 (insert) · 670–671 (replace) |
| overlap | **NONE** |

`git merge-file` → **rc=0, zero conflict markers**. Both edit sets fully present (PORT 30/30 added
lines, RIDER 11/11). The byte arithmetic is exactly additive and confirms disjointness
independently: 27,520 (base) + 1,908 (port) + 630 (rider) = **30,058** (merged), to the byte.

Composed provenance — every non-jg5 file:

| file | source | sha256[:16] |
|---|---|---|
| `inflate.sh` | PORT | `971eaa12b78e7168` |
| `inflate.py` | RIDER | `8a950b0d0266a1a1` |
| `runtime/residual_archive.py` | **3-way merge** | `e62489099c6d6d23` |
| `runtime/f26_corrector_native.c` | PORT (new) | `01a6e9557f969215` |
| `runtime/native_free_corrector.py` | PORT (new) | `70d2073b4235ba6e` |
| `runtime/rr5_arith_basis.py` | RIDER (new) | `c44758dfa6b530b0` |
| `archive.zip` | RIDER | `df7fd266e1b7488c` |

`__pycache__` is excluded from the composed tree — stale env artifacts, and the #1122 AppleDouble
hazard class. (Noted separately: the SHIPPED jg5 tree *contains* `__pycache__`. That is a packet
hygiene question, filed, not touched here.)

## DECODE IDENTITY — PROVEN, and the wrong-object trap I nearly fell into

Running the COMPOSED receiver on the composed archive vs the BASE receiver on the base archive,
comparing every parsed field:

| field | composed | base | match |
|---|---|---|---|
| `carrier_blob` | 22,316 | 22,316 | **YES** |
| `hpac_blob` | 17,952 | 17,952 | YES |
| `semantic_blob` | 36,130 | 36,130 | YES |
| `token_stream` | 113,847 | 113,847 | YES |
| `residual_payload` | 100 | 100 | YES |
| `table` / `schema` / `token_codec` / `compensation_blob` | — | — | YES |
| `compressed_models` | 66,413 | 66,582 | **NO (−169 B)** |

**`carrier_blob` is the field the rider restores, and it is byte-identical.** The rider's
`restore_carrier_body` runs inside the parse (composed `residual_archive.py`:195–198) and
reproduces the shipped body exactly, so every stage downstream is unchanged by construction.

`compressed_models` is the **raw on-disk container**, and its −169 B *is the saving* — the same
−169 B as the archive delta. A raw comparison of that field reads as "DIFFERS"; treating that as a
losslessness failure would be a wrong-object error. Confirmed by construction: restoring
`compressed_models` directly RAISES `RiderError: carrier body is truncated against its own bit
counts` — because it is not the carrier body. The carrier body is what matched.

**ERRATUM to `ddm_rr5_jg5_rider_remeasure_20260820.md`:** that memo's C3 line says "all **10**
parsed parts compared byte-for-byte." Measured precisely: **9 of 10 are byte-identical and the
10th is the intended container delta.** The losslessness verdict is UNCHANGED and now rests on the
right object (`carrier_blob`), but the count was overstated. Corrected here.

## WHY d_seg AND d_pose ARE EXACTLY UNCHANGED

Two independent legs, each proven on its own object:

1. **Rider leg** — `carrier_blob` byte-identical through the composed parse (above), plus rr5's C1
   (27,648/27,648 arithmetic symbols round-trip) and C2 (`restore_carrier_body` byte-identity).
2. **Port leg** — rr6 proved token bit-identity; the rr8 T4 row measured the score
   **bit-identical** (0.14839100138338618, d_seg 0.00020139, d_pose 6.37e-06) with
   `free_corrector=NativeFreeCorrector` in the receipt.

The legs act at different stages (basis-section entropy recode vs token-decode corrector) and the
merge is line-disjoint, so composition preserves both. Only the rate term moves.

## WHAT IS STILL OWED BEFORE THE ROW

1. **The port tree used here is the CLEAN one** (`candidate_runtime_jg5_native_corrector`), not the
   instrumented variant the rr8 row measured. The instrumentation emitted per-stage timing; the
   clean tree must be shown byte-neutral in behaviour (it carries `CANDIDATE_SEAL_rr8.json`, so it
   was sealed — re-verify at seal time).
2. **A composed-tree decode smoke** through the real `inflate.sh` (not just the parse) — the parse
   proof covers the rider leg; the port leg needs its native compile to succeed in the target env.
3. **Seal** via `make_candidate_seal.py` (dual-axis; the seal digest is content-only and
   sanitize-invariant) and ONE T4 fire behind single-flight.

## Own-vehicle frontier

**S 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600] — UNMOVED.** This unit produced a
merged, decode-proven shipping object and an exact rate arithmetic. The row is owed.
