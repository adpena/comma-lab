# ddm_rv17 — wave-end adversarial review, ROUND 4: one LOW finding, counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [receipt + source review, scorer-free]` ·
`score_claim: false` · cost $0 · fourth sibling of `ddm_rv17_wave_end_review_round1/2/3_20260820.md`.
Counter authority for the #1157 wave-end 3-pass cycle (the packet's `ADVERSARIAL_REVIEW_SCAFFOLD.md`
0/5 counter is separate and untouched).

## THE ANSWER, FIRST

**Counter 0/3 — one LOW finding, and I want to be careful not to oversell it.**

The R3-F1 cure is **substantively correct and complete**. Every value in it is right, both surfaces
agree, and MAIN's independent re-derivation is confirmed by my own: walking the 36 **frozen** gen6
files, re-verifying each row's sha against the manifest during the walk (**0 mismatches**), plus the
public `upstream/evaluate.py`, I get
`e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f` — exact. The owed receipt is
landed and every one of its claims verifies against disk.

The finding is narrow: **the recipe is written in Python notation, and read as Python it produces
the wrong digest.** `[(relative_path, bytes, sha256) for these 36 rows]` is literally a list of
*tuples*; `json.dumps` renders tuples as JSON **arrays**, and the array form hashes to
`37d17eb40eba0b5d…`, not `e8dcbc65…`. The code builds JSON **objects**. A reader who takes the
notation at face value — in a document whose entire purpose is to let a stranger reproduce that
number — lands a wrong value and concludes the pin is broken.

That is a one-line notation fix, not a defect in the cure's substance.

---

## ITEM 1 — independent verification of MAIN's re-derivation — **CLEAN**

Symmetric to the rule I stated in round 3: I did not adopt the match on MAIN's word.

| step | method EXECUTED | MEASURED result |
|---|---|---|
| read construction | `experiments/contest_auth_eval.py` "Environment-free custody digest" block | `files_payload = {files: sorted 3-field rows, upstream_evaluate_py}`, `sort_keys=True`, `separators=(",",":")` |
| walk the **frozen** tree | read all 36 files at `…/generations/gen6_rc2_composed/` as bytes, hashing each | **0 rows mismatched** their manifest-declared sha during the walk |
| measure the public input | `upstream/evaluate.py` read as bytes | 6,005 B, `7da71a84ce24286b…` |
| rebuild + hash | `sha256(json.dumps(payload, sort_keys=True, separators=(",",":")))` | **`e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f`** |

Exact match to MAIN's value, to `PACKET_TARGET.json /auth_runtime/runtime_files_sha256`, and to my
own round-3 derivation — which used the *repo-prep* row list as its path source, where this run
walked the frozen tree and re-verified every sha inline. Two independent routes, one value.

**The `ba713a25…` attribution checks out.** It sits at
`/generation_2_superseded/auth_runtime/runtime_files_sha256` — the generation-2 (rr4) superseded
block. The live `/auth_runtime/runtime_files_sha256` is `e8dcbc65…`. Retained history, not a
conflict, exactly as described.

---

## ITEM 2 — two-surface consistency, and the recipe-as-written test

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 2a | mutual consistency | read both cured surfaces | both name `e8dcbc65…` as reviewer-reproducible; both re-attribute `ccd9f7ab…` as axis-equal-but-not-files-only | **CLEAN** |
| 2b | `ccd9f7ab` re-attribution accuracy | compare against round-3 measurement | "folds in a repo-local import scan, so NOT files-only reproducible" — correct | **CLEAN** |
| 2c | recipe reproduces the digest **as written** | implement the comment text literally, both readings | object reading → `e8dcbc65…` ✓ · **tuple reading → `37d17eb4…` ✗** | **FINDING R4-F1** |

**2a.** `MANIFEST.sha256` header and `PR_BODY_DRAFT.md:485-490` say the same thing with the same
value. The PR-body version adds "derivable from exactly these 36 files plus the public upstream
evaluate.py" and defers the recipe to the MANIFEST header — a clean single-source-of-truth split,
no drift between them.

### RV17-R4-F1 — LOW — the recipe's Python notation, read as Python, yields the wrong digest

`.omx/research/ddm_pq1_submission_packet_prep_20260815/MANIFEST.sha256` (header comment block).

The recipe reads:

> `{"files": [(relative_path, bytes, sha256) for these 36 rows, sorted by path],`
> `"upstream_evaluate_py": (relative_path, bytes, sha256)} with sort_keys=True and`
> `separators=(",",":")`

Everything around the row shape is exact — the two payload keys, the field set, the sort order, the
serializer flags. The row *shape* is the gap. MEASURED, from the same 36 frozen files:

```
A  rows as JSON objects  (what the code builds)   = e8dcbc65…   ← MATCHES the declared digest
B  rows as JSON arrays   (literal Python tuples)  = 37d17eb4…   ← does NOT match
C  array files + object evaluate.py               = 2ec3e841…   ← does NOT match
```

The notation is Python-flavoured throughout — a comprehension, `sort_keys=True`,
`separators=(",",":")` — which invites a Python reading, and in Python `[(a, b, c) for …]` is
unambiguously a list of tuples that `json.dumps` emits as arrays. The reader gets a clean-looking
wrong answer with no error to warn them.

**In fairness, and stated plainly so this is not oversold:** `sort_keys=True` is a real hint that
the rows are keyed objects, the construction site is named, and I judge the object reading the more
likely one. This is a precision nit, not a substantive error — the cure's values are all correct.
I report it because the bar set for this round was executable and explicit — *a reader following
ONLY the comment text must land the digest* — and I measured a natural reading that does not. The
audience makes it worth fixing: this recipe is aimed at an external reviewer who cannot check our
source, so the notation is the entire specification.

**CURE (one line):** state the row shape as objects rather than tuples —
`{"files": [{"relative_path": …, "bytes": …, "sha256": …}, … sorted by relative_path],
"upstream_evaluate_py": {"relative_path": "evaluate.py", "bytes": …, "sha256": …}}` — i.e. "JSON
objects with keys `relative_path`, `bytes`, `sha256`", not `(a, b, c)`.

Round-1 `RV17-F7` remains open in the same header: `Verify: sha256sum -c MANIFEST.sha256` still
omits the working directory it must be run from, which for an external reader is the *other* half
of the same reproducibility problem. Worth curing in the same one-line pass.

---

## ITEM 3 — the landed divergence receipt — **CLEAN, and the scope correction is right**

`…/generations/gen6_receipts/DOC_DIVERGENCE_RECEIPT.json`, sha
`b9b94daff8e58f71845b7f992ac1c31c47360b9922009935ebab65813fef9345` — matches the claimed
`b9b94daf…`. Every substantive claim verified against disk:

| claim | method EXECUTED | MEASURED |
|---|---|---|
| repo `MANIFEST.sha256` final sha `ec906231…` | `shasum -a 256` on the repo copy | `ec906231f80e0f8259ae16220466e1ba246045154a64fb63556c4774d057852d` ✓ |
| frozen gen6 copy `ba6bbb45…` | same on the frozen copy | `ba6bbb45d499e43f074b01feb863ddc55c4fc6a25fad28eaa056a27f25442b9c` ✓ |
| "36 rows byte-identical, comments-only delta" | strip `#` lines, compare lists | 36 = 36, **byte-identical: True** ✓ |
| repo `PR_BODY_DRAFT.md` `ec2f8c1b…` | `shasum -a 256` | `ec2f8c1b421007c78942f2c598f3078b77b708f7ab77dc9bb14024cdc5cfdde5` ✓ |
| `PR_BODY_DRAFT.md` has no frozen counterpart | `test -f` in the frozen tree | **NO** — `frozen_counterpart: null` is correct ✓ |
| frozen tree carries only the packet candidate files | `ls \| wc -l` | **14** ✓ |
| rows still verify post-cure | `shasum -a 256 -c` from the frozen tree | **36 OK** ✓ |

The scope correction made during writing is the right call and materially better than what I named
in round 3: I proposed a row covering "the publish-source document," but the frozen gen6 tree holds
only the 14 packet candidate files, so `MANIFEST.sha256` is the *only* document that can diverge.
Recording `PR_BODY_DRAFT.md` as repo-only with an explicit `null` counterpart is more precise than
my prescription, and it forecloses a future reader hunting for a frozen copy that never existed.

---

## COUNTER

**0 / 3.** One LOW finding (R4-F1). The falsifier — a genuinely clean pass — is still unclaimed,
and I came close to claiming it this round.

Stating the judgment openly, because the pressure runs the other way: three prior rounds each found
something, and there is a real failure mode where a reviewer manufactures a fourth finding rather
than declare clean. I weighed R4-F1 against that. It survives because it is **measured, not
argued** — I have a concrete wrong digest (`37d17eb4…`) from a reading the notation actively
invites, against a bar this round stated in executable form. Had the only complaint been stylistic,
the honest answer would have been 1/3.

Everything substantive in this round passed: the digest is right and independently re-derived two
ways, both surfaces agree, the historical `ba713a25…` is correctly filed, and the receipt's seven
claims each verify against disk. The remaining defect is one line of notation in a document written
for a stranger.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round4_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
