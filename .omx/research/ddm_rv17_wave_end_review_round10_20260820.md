# ddm_rv17 — wave-end adversarial review, ROUND 10: the case question bit, and pulling the thread found a stale accounting citation; counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · tenth sibling of `ddm_rv17_wave_end_review_round1-9_20260820.md`.

## THE ANSWER, FIRST

**Counter stays 0/3 — one MED finding, and it came directly out of the question you asked in (c).**

Your derived-coverage cure is **correct and complete**: the R7 receipt's **11/11 shas re-derive**,
the top-level-only scope is **honest** (the prep tree has **zero subdirectories**, so a top-level
intersection *is* the complete intersection), gen-suffixed copies are correctly outside it, and the
live checker reports `PASS: 11 tracked document shas … all 3 derived two-copy pairs covered`.

**Your case question: YES, it bites — one published surface.** The shipped
`BORROWED_SUBSTRATE_ACCOUNTING.md:135` cites `` `ARCHIVE_MANIFEST.json:21` `` while the published
directory ships lowercase `archive_manifest.json`. Every other published surface is clean
(PR body, README_PUBLIC, REPORT_PUBLIC, MANIFEST header: **0** uppercase references).

**Then the thread kept coming.** Chasing that one citation, the case mismatch turned out to be the
*least* wrong thing about it:

```
cited:   "70,453 B, `e35d12371fa79747…`; base-identity from `ARCHIVE_MANIFEST.json:21`"

case     shipped tree has archive_manifest.json (lowercase) — file not found on Linux
line     shipped copy is 20 lines long — line 21 does not exist
line     prep copy line 21 is `"magic": "RX1M",` — not a base-identity row
sha      e35d12371fa79747 appears in NEITHER manifest copy, at any line
bytes    70,453 appears in NEITHER copy; the manifest's own container figure is 66,413 B
```

The citation is a **stale inherited reference**, almost certainly carried from a generation whose
container was 70,453 B and never re-derived when the composed candidate's became 66,413 B. It ships,
and it sits on the borrowed-substrate accounting surface that CLAUDE.md NO-FAKE #7 makes
load-bearing.

---

## ITEM 1 — the R7 receipt, re-derived — **CLEAN (11/11)**

| document | leg | MEASURED |
|---|---|---|
| `MANIFEST.sha256` | repo / frozen | `fea2dc4709b2…` / `ba6bbb45d499…` |
| `archive_manifest.json` | repo / frozen | `5b948c9032ec…` / `9349837fd9f6…` |
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | repo / frozen | `e49c14bf90b7…` / `e49c14bf90b7…` (identical) |
| `PR_BODY_DRAFT.md` · `verify_files_digest.py` · `SWAP_PROCEDURE.md` · `FREEZE_CHECKLIST.md` · `verify_receipt_chain.py` | repo-only | `284d619d…` · `52108a66…` · `acc3b26f…` · `f1e3639a…` · `f8693fc8…` |

**Strict-subset claim re-derived independently: confirmed.** 23 repo-only keys, **0** frozen-only,
**0** differing shared keys; `archive_sha256` (`df7fd266…`) and `archive_bytes` (180,456) agree.

## ITEM 2 — attacking the derived-coverage function — **CLEAN, the scope is honest**

The docstring's "top-level only" reads like a hole. It is not, and the reason is structural rather
than lucky:

```
prep tree subdirectories : 0        ← measured
frozen tree subdirectories: cpr1, runtime
```

A two-copy document must be in **both** trees. With the prep tree structurally flat — a property
`tools/packet_census_guard.py` enforces per `SWAP_PROCEDURE.md` — nothing below top level can be in
both, so the top-level intersection **is** the complete intersection. The scope is honest, and it is
honest *because* of an invariant enforced elsewhere. Worth stating in the docstring, since the
guarantee would silently lapse if the prep tree ever gained a subdirectory.

Gen-suffixed copies: `archive_manifest.gen5.json` and `archive_manifest.gen6.json` exist only in
prep, never in frozen, so they fall outside the intersection — **correct**, and correct for the
right reason rather than by name-filtering.

Live: `PASS: 11 tracked document shas match the latest receipt (DOC_DIVERGENCE_RECEIPT_R7.json);
all 3 derived two-copy pairs covered`, rc=0.

## ITEM 4 — standing substance — **CLEAN**

```
frozen archive.zip : df7fd266e1b7488c… / 180,456 B      S : 0.14827847122030852
pointer            : match, contest_cuda                rows : 36 OK
```

---

## RV17-R10-F1 — MED — a shipped accounting row cites evidence that exists in neither manifest copy

`BORROWED_SUBSTRATE_ACCOUNTING.md:135` — present byte-identically in **both** the prep tree and the
**published** frozen tree (`e49c14bf…`).

Row 3 reads: *"Compressed model container | `inherited-substrate` (unchanged from base; PR-level sha
equality **not** independently verified) | 70,453 B, `e35d12371fa79747…`; base-identity from
`ARCHIVE_MANIFEST.json:21`"*.

MEASURED against both copies of the manifest:

| element | status |
|---|---|
| filename case | shipped dir has `archive_manifest.json`; on the contest's case-sensitive Linux the cited name resolves to nothing |
| line 21 (shipped) | file is **20 lines** — the line does not exist |
| line 21 (prep) | `"magic": "RX1M",` — not a base-identity row |
| sha `e35d12371fa79747` | **absent from both copies**, every line; in the whole shipped tree it appears only in the accounting document asserting it |
| bytes `70,453` | **absent from both copies**; the prep manifest's own `sections.compressed_models_bytes` is **66,413** |

**Being fair to the row:** the 70,453 B / `e35d1237…` values may well be correct *for the base PR's
container* — the row describes inherited substrate, not ours, and our 66,413 B figure need not
match. The defect is not necessarily the values; it is that **the citation offered as their backing
does not back them.** A reader sent to `ARCHIVE_MANIFEST.json:21` finds a different file, a
nonexistent line, and neither value.

**Why MED rather than LOW.** The row's own caveat covers only the *PR130/135-byte-identical* label
("**overstates**"); the byte count and the citation read as asserted fact. And this is the
borrowed-substrate accounting document, which CLAUDE.md NO-FAKE #7 designates as the required
backing for every originality claim — an unbacked citation there is a defect on precisely the
surface where backing is mandatory. No score, digest, archive-byte, or row impact.

**CURE:** re-derive the row from a receipt that actually holds the base container's identity and
cite that receipt by path + sha; or, if the figure cannot be sourced, mark it `UNSOURCED` rather
than citing a manifest that does not contain it. Fix the case to `archive_manifest.json` in the same
pass. **Class note:** this is a *stale inherited citation* — the fifth distinct instance of a value
carried forward without re-derivation when its source moved. The `verify_files_digest.py` /
`verify_receipt_chain.py` pattern applies again: a check that resolves every `` `FILE:LINE` ``
citation in the packet's shipped documents and refuses when the file, line, or quoted value is
absent would end this class the way the other two ended theirs.

---

## COUNTER

**0 / 3.** One MED finding (R10-F1).

Your question in (c) was the right one to ask, and it was worth more than its own answer: the case
mismatch was real but cosmetic, and following it exposed a stale citation that no case-fix would
have touched. That is the tenth round and the tenth instance of one genus — **a value carried
forward after its source moved** — now appearing in citations rather than in labels, receipts, or
recipes.

What ten rounds have still not found: a wrong score, a wrong pin, a wrong digest, a mis-scoped
receipt, or an unverifiable archive claim. `S = 0.14827847122030852` recomputed again today from the
frozen archive's own bytes; 36/36 rows verify; the chain checker passes with derived coverage over
all three two-copy pairs.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round10_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
