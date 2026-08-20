# ddm_rv17 — wave-end adversarial review, ROUND 6: the class cure HOLDS; two findings on what it left behind; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [source + receipt review, scorer-free]` ·
`score_claim: false` · cost $0 · sixth sibling of `ddm_rv17_wave_end_review_round1-5_20260820.md`.

## THE ANSWER, FIRST

**The class cure worked. The recipe defect class is dead.** I executed
`verify_files_digest.py` myself from the frozen gen6 root and it produced
`e8dcbc65…` with `PASS`, rc=0, 36 rows verified. R5-F1 is not clarified but
**structurally eliminated** — I proved it by running the script against a file literally named
`eval_copy.py` and still landing the digest, because the entry's `relative_path` is hard-coded.
The retired prose no longer carries a single derivation detail a reader could follow into a wrong
value. Three rounds of under-specified corners are closed in one move.

**Counter 0/3 — two findings, and neither is in the script's math.** Both are about what the cure
left behind: it added a file the packet now depends on, and the obligation to *ship* that file
landed in a receipt inside frozen SSD custody rather than in the checklist the swap executor
actually follows.

**R6-F1 (MED) is the one that bites at publish.** Two published surfaces now tell a contest
reviewer to run `verify_files_digest.py`. The requirement to include it in the published submission
dir appears **only** in `DOC_DIVERGENCE_RECEIPT_R5.json` on the SSD. `SWAP_PROCEDURE.md` — where
step 4A lives — and `FREEZE_CHECKLIST.md` **do not mention the file at all**. If the stager misses
it, the packet's own verification instruction points at a script that isn't there.

This is the wave's signature genus one more time: the correction landed in the record, not in the
surface that drives behavior.

---

## ITEM 1 — the script, executed and read adversarially — **CLEAN**

| # | checked | method EXECUTED | MEASURED |
|---|---|---|---|
| 1a | happy path | ran from the frozen gen6 root with the public `evaluate.py` | `runtime_files_sha256: e8dcbc65…` · `PASS (36 rows verified)` · **rc=0** |
| 1b | R5-F1 structurally dead | ran against a file named **`eval_copy.py`** | still `e8dcbc65…` — the hard-coded `"evaluate.py"` value defeats the path ambiguity |
| 1c | **vacuity** | stripped every data row from a scratch manifest | `FAIL: expected 36 manifest rows, found 0` — **the VACUITY==PASS class is defeated by the `EXPECTED_ROWS` pin** |
| 1d | tampered row sha | flipped one row's declared sha | `FAIL: sha mismatch`, both values printed |
| 1e | duplicate row | appended a 37th row | caught at sha-verify, before the count check |
| 1f | no `evaluate.py` found | bare run from the frozen root | `FAIL: evaluate.py not found. Pass its path explicitly` — rc=1, actionable |

All probes are mine, run in a scratch copy; the frozen tree was never modified and the scratch was
deleted. The three defences that matter are all present and all fire: **every row's sha re-verified
in-walk**, **the row count pinned**, and **the final digest pinned**. A reviewer cannot reach a
false PASS through a mutated manifest, a missing file, or an empty one.

The candidate-path search is fail-loud by construction: any wrong `evaluate.py` changes the digest,
so it exits non-zero rather than passing on the wrong input.

**One NOTE, deliberately not raised as a finding.** A malformed manifest row with a single token
(`"loneword"`) reaches `stripped.split(None, 1)` and raises an uncaught
`ValueError: not enough values to unpack`. That is loud — non-zero exit, visible traceback — but a
reviewer sees a Python stack trace instead of a typed `FAIL:` line, and every other error path in
this script is typed. It is unreachable with the shipped manifest. Same threshold I have applied
since round 4: **a silent wrong value is a finding; a loud failure is a note.** A `try/except
ValueError` around the split would make the script's error contract uniform whenever it is next
touched.

---

## ITEM 2 — the retired prose does not half-specify — **CLEAN**, with one omission

The paragraph shrank rather than growing a fourth clarification, and **every under-specifiable
element is gone**: no payload shape, no object-vs-array, no `sort_keys`, no `separators`, no
`relative_path` value. What remains is a pointer ("run the shipped `verify_files_digest.py` from
this directory — the script IS the recipe"), a statement of inputs ("exactly these 36 files plus
the public upstream evaluate.py"), and a construction-source reference. **A reader cannot follow
the remaining text into a wrong value, because it no longer describes a construction.** That is the
correct shape for a retired recipe.

### RV17-R6-F2 — LOW — the non-runtime document enumeration was not updated for the file the cure added

`MANIFEST.sha256` (header) enumerates the packet's non-runtime documents — *"this file, README.md,
report.txt, LICENSE, THIRD_PARTY_NOTICES.md, compress.py, COMPRESS.md, archive_manifest.json,
BORROWED_SUBSTRATE_ACCOUNTING.md"* — explicitly so a reader doing a fresh suffix walk can account
for files that are not manifest rows.

`verify_files_digest.py` is a `.py` file, ships in the packet, is **not** a manifest row, and **is
absent from that list**. It appears exactly once in the whole header (line 10, the pointer).
A reconciler walking the published directory finds one file the enumeration does not explain —
which is the precise confusion the list exists to prevent.
**CURE:** add `verify_files_digest.py` to the enumeration in the same pass as R6-F1.

---

## ITEM 3 — receipt chain coherent; the obligation is in the wrong place

**The chain is sound.** `DOC_DIVERGENCE_RECEIPT_R5.json` (sha `420c3afb28370f2c…`) declares itself
successor to R4, scopes what it supersedes, and every sha it claims verifies against disk:

| claim | MEASURED |
|---|---|
| `MANIFEST.sha256` repo `31540dad1a1148e6…` | `31540dad1a1148e6d74dbdf08617f00828606dd0d3bb57b53d2c44774113b75a` ✓ |
| `PR_BODY_DRAFT.md` repo `284d619d95cf1475…` | `284d619d95cf1475596e442678abcfa79a77df1f1a9630f3ea780dc0c81bcd99` ✓ |
| `verify_files_digest.py` repo `52108a66eb70467d…` | `52108a66eb70467d8e7d83749d20b4508dade46fa627d3f74e9c31e7a77b8e74` ✓ |
| frozen gen6 copy `ba6bbb45…` | unchanged ✓ |
| 36 data rows byte-identical | **True**, 36 = 36 ✓ |

r3 → R4 → R5 compose into one true story with each superseded value explicitly retired.

### RV17-R6-F1 — MED — the step-4A obligation is recorded where the swap executor will not read it

MEASURED — `verify_files_digest.py` appears in exactly three repo-side files:

```
MANIFEST.sha256        (the pointer telling a reviewer to run it)
PR_BODY_DRAFT.md       (the published invocation)
verify_files_digest.py (itself)

SWAP_PROCEDURE.md      -> 0 occurrences   ← step 4A lives here
FREEZE_CHECKLIST.md    -> 0 occurrences   ← the freeze-time gate
```

The requirement — *"the stager must include it in the published submission dir at step 4A"* — is
stated only in the `reason` field of `DOC_DIVERGENCE_RECEIPT_R5.json`, which lives at
`/Volumes/APDataStore/pact/…/gen6_receipts/`. A swap executor works from `SWAP_PROCEDURE.md` and
`FREEZE_CHECKLIST.md` in the repo; nothing routes them to a receipt in frozen SSD custody, and the
receipt is not a checklist.

**The failure is concrete, not theoretical.** The packet now *depends* on this file from two
published surfaces: the MANIFEST header instructs a reviewer to run it, and the PR body invokes it
after `shasum -c`. If step 4A omits the copy, the public packet ships an instruction pointing at a
missing script — a broken verification path in the one document written for the maintainer.

Note the asymmetry that makes this easy to miss: the 36 runtime rows are protected by the manifest
and by `verify_files_digest.py` itself, but `verify_files_digest.py` is protected by **nothing** —
it is not a manifest row, so no integrity check covers its presence or its bytes.

**CURE:** add the file to `SWAP_PROCEDURE.md` step 4A's copy list and to `FREEZE_CHECKLIST.md` as a
pre-publication existence check ("the published submission dir contains `verify_files_digest.py`
and it exits 0 there"). The receipt can keep its record; the checklist is what fires.

---

## ITEM 4 — standing substance — **CLEAN**

Digest `e8dcbc65…` unchanged and re-confirmed by execution; 36 data rows byte-identical between
repo and frozen copies; 36/36 rows verify from the frozen root; runtime tree pin `fdd57749…`,
archive `df7fd266…`/180,456 B, and the score untouched.

---

## COUNTER

**0 / 3.** Two findings — R6-F1 (MED), R6-F2 (LOW). One note declined (the untyped `ValueError`,
loud not silent), applying the same threshold in both directions as in rounds 4 and 5.

The verdict I want recorded, because it is the honest one: **the class cure succeeded.** The thing
five rounds kept re-finding — an under-specified byte-exact recipe in prose — is gone, and gone
structurally rather than by another clarification. Six rounds have still found no wrong number, no
wrong pin, no unverifiable claim, and no mis-scoped receipt.

What the cure created is a new dependency, and the obligation to satisfy it was filed in the
archive rather than in the checklist. That is the same genus as this wave's very first finding
back in round 1: a correction that lands in the record while the surface that drives behavior goes
unchanged. Two lines in `SWAP_PROCEDURE.md` and `FREEZE_CHECKLIST.md` close it, and then — on the
evidence of six rounds — this genuinely closes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round6_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
