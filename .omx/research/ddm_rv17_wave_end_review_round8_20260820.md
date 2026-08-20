# ddm_rv17 — wave-end adversarial review, ROUND 8: one MED finding; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · eighth sibling of `ddm_rv17_wave_end_review_round1-7_20260820.md`.

## THE ANSWER, FIRST

**Counter RESETS to 0/3.** One MED finding — and it is the exact mirror of round 6's.

R6-F1 was *a cure that landed in the record but not at the behavioral surface*. **R8-F1 is a cure
that landed at the behavioral surface and the record did not follow.** The R6 cure edited
`MANIFEST.sha256` (adding the script to the enumeration), moving the repo publish-source sha from
`31540dad…` to **`fea2dc47…`** — and no successor divergence receipt was appended. The chain stops
at R5, still claiming `31540dad…`. A reconciler comparing the frozen copy against the repo copy now
finds an unexplained third value, which is precisely the confusion the chain exists to prevent.

**Everything else re-derived clean from primary bytes**, including two things no prior round had
actually done: I hashed the frozen `archive.zip` itself rather than reading its sha from a receipt,
and I opened the archive to check its internals against the packet's published claims.

**Honest scope note:** this does not reverse round 7. Round 7's scope was the two R6 cures and the
executor surfaces; the receipt chain was not in it. A clean pass is only clean for what was
examined, and item 4 of this round is the first look at the chain since the R6 edit.

---

## PER-ITEM VERDICT ROWS

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 1 | the score | hashed the **frozen archive.zip bytes**; recomputed S from receipt components | `df7fd266…`, **180,456 B**, `S = 0.14827847122030852` | **CLEAN** |
| 2 | the digest chain | **my own construction** from the 36 frozen files + `upstream/evaluate.py`, then the script as cross-check | both `e8dcbc65…`; script `PASS` rc=0 | **CLEAN** |
| 3 | the four published documents | first-time read; every published expectation tested against a primary | all four expectations reproduce exactly | **CLEAN** |
| 4 | receipt chain r3→R4→R5 | every sha against disk | **`MANIFEST.sha256` claim is STALE** | **FINDING R8-F1** |
| 5 | free sweep — archive internals | opened `archive.zip`, compared to `archive_manifest.json` | member/size/method/sha all match | **CLEAN** |

### 1 — the score, from the bytes rather than from a receipt

```
FROZEN archive.zip bytes  : 180456
FROZEN archive.zip sha256 : df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080
receipt expected sha      : df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080   MATCH
S = 100(0.00020139) + sqrt(10 × 6.37e-06) + 25(180456)/37,545,489
  = 0.020139 + 0.007981227975693965 + 0.12015824324461455
  = 0.14827847122030852
```

The `final_score` field still reads `0.15` and was not used (Catalog #877). Eight rounds, and the
value has never moved.

### 2 — the digest, built from first principles

I read the `files_payload` construction fresh from `experiments/contest_auth_eval.py` and rebuilt
it myself — walking the 36 frozen files, asserting each row's sha against the manifest in-walk
(**0 mismatches**), hashing the compact sorted JSON. Result `e8dcbc65…`. Only then did I run
`verify_files_digest.py` as a cross-check: same digest, `PASS`, rc=0. Independent route first,
tool second.

### 3 — the four documents tell one story, and every expectation they publish is true

Read as a first-time reader, then each published expectation tested against a primary artifact:

| the packet says | I measured |
|---|---|
| `shasum -a 256 -c MANIFEST.sha256` → "expect 36 lines ending in: OK" | **36 OK** |
| `python3 verify_files_digest.py` → "expect: PASS: matches the packet's pinned runtime_files_sha256 (36 rows verified)" | that exact line, rc=0 |
| `shasum -a 256 archive.zip` → "expect df7fd266…" | `df7fd266…` |
| `wc -c < archive.zip` → "expect 180456" | 180,456 |
| zip assert `("p", 180356, 0)` | name `p`, 180,356, compress_type 0 |

No document contradicts a primary. `fdd57749…` is correctly labelled the authority pin,
`e8dcbc65…` the reviewer-reproducible digest, `ccd9f7ab…` axis-equal-but-not-files-only, and the
script correctly enumerated as a non-runtime document. MANIFEST's "from this directory", 4A's "from
the published tree root", and FREEZE (d)'s wording all denote the same place.

### 5 — free sweep: the archive's internals versus its published claims

Chosen because seven rounds swept labels, notation, and executor surfaces but **never opened the
scored payload itself.** Every declared property holds:

```
members: 1
  name='p'  file_size=180356  compress=stored  sha256=83fa979c1118499b…
archive_manifest.json: members[0] = {name p, file_size 180356, compress_size 180356,
                        compression_method stored, sha256 83fa979c1118499b7dd6083c…}
archive_bytes 180456 · archive_sha256 df7fd266…    ← both independently confirmed above
```

ZIP overhead is 100 B over the single stored member, consistent with one stored entry. The manifest
also declares `portable_runtime_content_tree_sha256 = ccd9f7ab…` — labelled *portable*, not
reviewer-reproducible, which is the accurate framing per round 3.

---

## RV17-R8-F1 — MED — the divergence receipt chain did not follow the R6 cure

MEASURED:

```
repo   MANIFEST.sha256 on disk : fea2dc4709b2247bdca0872cd2327526bd4c28cdcb05ce656bdaab6bd6b8cd1c
R5 receipt's recorded claim    : 31540dad1a1148e6d74dbdf08617f00828606dd0d3bb57b53d2c44774113b75a   ← STALE
successor receipt (R6)         : does not exist (0 in gen6_receipts/)

PR_BODY_DRAFT.md               : 284d619d… — unchanged, R5's claim still true
verify_files_digest.py         : 52108a66… — unchanged, R5's claim still true
frozen gen6 MANIFEST.sha256    : ba6bbb45… — unchanged
36 data rows                   : byte-identical, 36/36 OK
```

The R6 cure commit touched `FREEZE_CHECKLIST.md`, `MANIFEST.sha256`, and `SWAP_PROCEDURE.md`. Only
the MANIFEST is a chain-tracked document, and only its claim went stale; the other two tracked shas
remain accurate.

**Why it matters.** The r3 receipt defines `authoritative_source` as the repo copy "at the commit
landing this receipt," and r3 → R4 → R5 each superseded the MANIFEST sha as it moved — three times,
correctly. The chain's whole function is to let a future reader reconcile two divergent copies and
know which is authoritative and what it should hash to. Its latest entry now names a value that no
longer exists on either side.

**No score, digest, or row impact:** frozen bytes untouched, 36 rows byte-identical, 36/36 verify,
`e8dcbc65…` and `df7fd266…` unchanged. This is record completeness, not correctness of the packet.

**CURE:** append `DOC_DIVERGENCE_RECEIPT_R6.json` recording `MANIFEST.sha256` repo
`fea2dc4709b2247b…` (superseding R5's `31540dad…`), frozen `ba6bbb45…` unchanged, delta class
comments-only, data rows byte-identical, and the curing commit. **Better, and the reason this keeps
recurring:** the chain is maintained by hand, so it goes stale every time a cure touches a tracked
document — four times now. A tiny check that recomputes the three tracked shas and refuses when the
latest receipt disagrees would end the class, exactly as `verify_files_digest.py` ended the recipe
class.

---

## COUNTER

**0 / 3 — reset.** Per the canonical rule, a round that finds any issue resets the counter; round
7's clean pass does not carry.

I want the shape of this on the record, because it is the eighth instance of one pattern and the
first time it has run in the opposite direction. Every prior finding was *the record is right, the
behavior surface is stale*. This one is *the behavior surface is right, the record is stale*. The
genus is not "documents drift" — it is **hand-maintained coupling between two artifacts that must
agree**. Every instance the campaign cured by hand came back; the one instance cured by machine
(`verify_files_digest.py`, round 6) has not. That is the actionable lesson, and it points at the
same fix here.

What eight rounds have still not found: a wrong number, a wrong pin, an unverifiable claim, a
mis-scoped receipt, or a score error. Re-derived today from the frozen bytes themselves — including
the archive's own sha and its internal member, neither previously checked — everything substantive
holds.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round8_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
