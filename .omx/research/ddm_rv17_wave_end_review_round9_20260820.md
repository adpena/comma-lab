# ddm_rv17 — wave-end adversarial review, ROUND 9: one MED finding — the fifth coupling surface is real and already diverged; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · ninth sibling of `ddm_rv17_wave_end_review_round1-8_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — one MED finding, and it is the answer to the question you asked in item 3.**

The R6 receipt is correct: **all six shas re-derived against disk, all match.** The chain checker is
**sound** — it survived four probes you did not run, including the one I most expected to break it.

But the class question has a real answer, and it is *no*. **The chain check closes the genus only
for documents the receipts happen to track, and that tracking list is itself hand-maintained.** The
genus was displaced one level up, not eliminated. I proved it by enumerating the documents that
actually exist in both trees:

```
MANIFEST.sha256                    DIVERGED   tracked ✓
archive_manifest.json              DIVERGED   NOT TRACKED   ← fifth coupling surface, ACTIVE
BORROWED_SUBSTRATE_ACCOUNTING.md   identical  NOT TRACKED   ← uncovered, currently clean
```

`archive_manifest.json` is diverged **right now** between the repo publish-source and frozen custody,
recorded nowhere, and invisible to `verify_receipt_chain.py`. The r3 receipt's founding claim —
that "the divergence covers exactly `MANIFEST.sha256`" — is measurably incomplete.

The divergence itself is benign: the frozen copy is a strict **subset** (23 repo-only keys, zero
frozen-only keys, zero differing shared keys), and `archive_sha256` / `archive_bytes` agree. No
score, digest, or row impact. The finding is the **coverage gap**, not the instance.

---

## ITEM 1 — the R6 receipt, re-derived — **CLEAN**

Every sha it tracks, hashed from disk by me:

| document | leg | MEASURED | claim |
|---|---|---|---|
| `MANIFEST.sha256` | repo | `fea2dc4709b2247b…` | OK |
| `MANIFEST.sha256` | frozen | `ba6bbb45d499e43f…` | OK |
| `PR_BODY_DRAFT.md` | repo-only | `284d619d95cf1475…` | OK |
| `verify_files_digest.py` | repo-only | `52108a66eb70467d…` | OK |
| `SWAP_PROCEDURE.md` | repo-only | `acc3b26fd9debb5e…` | OK |
| `FREEZE_CHECKLIST.md` | repo-only | `f1e3639a13b787c5…` | OK |

**6/6 verify.** Newly chain-tracking the two executor documents is the right instinct — it is what
would have caught R8-F1 at the source.

---

## ITEM 2 — breaking the checker — **CLEAN, it held**

Live: `PASS: 6 tracked document shas match the latest receipt (DOC_DIVERGENCE_RECEIPT_R6.json)`,
rc=0. Then four probes of my own design, none of which you ran:

| # | my probe | why I tried it | MEASURED |
|---|---|---|---|
| C2 | planted `_R10` with a deliberately wrong sha | **the classic lexicographic-vs-numeric rank bug**: if ranks were string-sorted, `"R10" < "R6"` and the checker would silently validate a *stale* receipt and PASS — a silent wrong answer | picked `_R10`, `FAIL: PR_BODY_DRAFT.md (repo): sha mismatch`, **rc=1** — ranking is numeric (`int(match.group(1))`), bug **disproved by execution** |
| C3 | chain with a gap (unsuffixed + `_R6`, no R4/R5) | a missing link could break "latest" selection | correctly picked `_R6`, `PASS`, rc=0 |
| C4 | receipt tracking a file that does not exist | might crash instead of failing cleanly | `FAIL: NO_SUCH_FILE.md (repo): MISSING on disk (<path>)` — **typed FAIL naming the path**, rc=1 |
| C5 | receipt tracking zero documents | your round-6 vacuity lesson | `FAIL: latest receipt … tracks zero documents`, rc=1 |

**My own methodology error, disclosed.** My first pass at C3–C5 read `rc` through a pipe into
`tail`, so `$?` was `tail`'s status — the exact `PIPESTATUS` bash-ism you disclosed hitting. I
caught it, re-ran pipeless, and the numbers above are the true exit codes. Recording it because I
flagged that same trap in someone else's work and then walked into it.

---

## RV17-R9-F1 — MED — the checker's coverage is a hand-maintained list, and a fifth surface is already diverged

MEASURED — the documents that exist in **both** the repo prep tree and the frozen gen6 tree:

```
repo prep ∩ frozen gen6 = { MANIFEST.sha256, archive_manifest.json, BORROWED_SUBSTRATE_ACCOUNTING.md }
                                 tracked            NOT tracked              NOT tracked

archive_manifest.json   repo 5b948c9032ec…   frozen 9349837fd9f6…   ← DIVERGED, unrecorded
```

Content of the divergence (so this is precise, not alarming): the frozen copy is a strict subset —
**23 keys present only in the repo copy, 0 keys only in frozen, 0 shared keys differing**, and
`archive_sha256` (`df7fd266…`) and `archive_bytes` (180,456) agree. The repo copy simply accumulated
provenance keys (`score_components`, `runtime_files_sha256`, the axis notes, …) after the freeze.
Nothing contradicts; nothing affects the score, the digest, or the 36 rows.

**Why it is still a finding.** `verify_receipt_chain.py` validates *the shas a receipt lists*. It
never asks *which documents could diverge*. So its coverage equals whatever a human remembered to
add to the last receipt — which is the same hand-maintained coupling the check was built to end.
The proof is that the very first receipt in the chain asserted a scope ("exactly `MANIFEST.sha256`")
that was already wrong when written, and nine rounds of review, four receipts, and a machine check
did not surface it until the surface was enumerated from the filesystem.

**CURE — make coverage derived, not declared.** Have the checker enumerate
`set(prep_tree_files) ∩ set(frozen_tree_files)` and refuse when any member is absent from the latest
receipt's tracked set. That converts the tracking list from hand-maintained to machine-derived, and
it is exactly the move that made `verify_files_digest.py` end the recipe class. Then append an R7
receipt tracking `archive_manifest.json` (both legs) and `BORROWED_SUBSTRATE_ACCOUNTING.md`.

---

## ITEM 4 — standing substance — **CLEAN**

```
frozen archive.zip : df7fd266e1b7488c… / 180,456 B      (hashed from bytes)
S                  : 0.14827847122030852
pointer            : 0.14827847122030852 · df7fd266e1b7488c… · contest_cuda   MATCH
manifest rows      : identical repo↔frozen, n=36        · shasum -c: 36 OK
```

---

## COUNTER

**0 / 3.** One MED finding (R9-F1).

Nine rounds, and the pattern has now completed its arc. Every cure has been correct at the level it
addressed, and each one revealed the same defect one level up: a stale heading over a corrected body
→ a pin sentence over rows it did not hash → a prose recipe that prose could not specify → an
obligation filed away from the executor → a record that did not follow the behavioral surface → and
now **a machine check whose coverage is still declared by hand.** The genus was never "documents
drift." It is: *wherever two artifacts must agree and a human decides they agree, they eventually
do not.* Each cure moved the human decision somewhere new; the fix that finally holds is the one
that derives the coupling from the filesystem instead of from memory.

Still not found in nine rounds: a wrong number, a wrong pin, an unverifiable claim, a mis-scoped
receipt, or a score error. `S = 0.14827847122030852` has recomputed identically every time it was
checked, today from the frozen archive's own bytes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round9_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
