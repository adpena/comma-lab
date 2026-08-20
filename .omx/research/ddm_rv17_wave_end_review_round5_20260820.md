# ddm_rv17 — wave-end adversarial review, ROUND 5: one LOW finding, counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [receipt + source review, scorer-free]` ·
`score_claim: false` · cost $0 · fifth sibling of `ddm_rv17_wave_end_review_round1-4_20260820.md`.
Counter authority for the #1157 wave-end 3-pass cycle.

## THE ANSWER, FIRST

**Counter 0/3 — one LOW finding, and I owe a class-level observation more than I owe another nit.**

Both round-4 cures **work**. I executed the corrected recipe from only the final comment text and
landed `e8dcbc65…` exactly. RV17-F7 is genuinely closed on both surfaces. The supersession chain is
coherent — r3 and R4 together tell one true story, and every sha in both verifies against disk.

The item-4 sweep found one more silent wrong-value branch: **the recipe never states the
`relative_path` *value* for the `upstream_evaluate_py` entry.** `"evaluate.py"` gives `e8dcbc65…`;
`"upstream/evaluate.py"` — the spelling this same header uses on line 6 — gives `697d1146…`, with
no error to warn the reader.

**The class-level point, which matters more than this instance.** Three rounds running, the finding
has been a different under-specified corner of the same prose paragraph: tuple-vs-object (R4-F1),
now path-value (R5-F1). That is not bad writing; it is a structural property. **A prose recipe for
a byte-exact digest is inherently under-specifiable** — every clarification adds sentences that each
admit a reading. The durable cure is to stop writing the recipe and ship it: a ~10-line
`verify_files_digest.py` in the packet that a reviewer runs. That removes the entire defect class in
one move, and it is the last thing this header needs.

---

## ITEM 1 — the notation cure, executed from only the final text — **CLEAN**

| # | checked | method EXECUTED | MEASURED |
|---|---|---|---|
| 1a | recipe as written | implemented objects with keys `relative_path`/`bytes`/`sha256`, `files` sorted by `relative_path`, `sort_keys=True`, `separators=(",",":")` | **`e8dcbc65…` — MATCH** |
| 1b | tuple branch closed | header now says "never an array/tuple, which hashes to a different value" | no tuple notation remains |
| 1c | rows still verify | `shasum -a 256 -c` from the frozen gen6 root | **36 OK** |

R4-F1 is **cured**. The header now names the row shape explicitly and even warns about the failing
branch, which is the right shape of fix — it tells the reader what *not* to do, not just what to do.

---

## ITEM 2 — RV17-F7 on both surfaces — **CLOSED**

`MANIFEST.sha256:31-33` now reads: *"run from the runtime tree root — the directory containing
inflate.py — since the rows are relative to it."* `PR_BODY_DRAFT.md:480` establishes *"From the
submission directory in a checkout of the contest repository."*

**Is "submission directory" unambiguous to a contest reviewer? Yes.** It is the contest's own term
for the directory holding `inflate.sh` / `inflate.py` / `archive.zip`, and it matches the harness's
recorded `runtime_root_name` for this very row — literally `submission_dir`. A reviewer working in a
contest-repo checkout lands the right directory without inference. The two surfaces are consistent.

**One note, deliberately NOT raised as a finding.** The gloss "the directory containing inflate.py"
is not unique — the frozen tree holds both `<root>/inflate.py` and `<root>/cpr1/inflate.py`. But the
primary identifier ("the runtime tree root") is correct and standard, and I executed the wrong
branch: running from `cpr1/` produces **35 missing-file errors + 1 FAILED**, a loud and immediate
refusal. No wrong value and no wrong belief is reachable. I am recording the threshold I applied
explicitly, so it can be checked: **a branch that yields a clean-looking wrong value is a finding; a
branch that fails loudly is a note.** That is the same rule I used in round 4, applied here in the
direction of *not* raising something.

---

## ITEM 3 — the supersession chain — **COHERENT**

`DOC_DIVERGENCE_RECEIPT_R4.json` (sha `db5efe4977c002f7…`) declares itself an append-only successor
and — importantly — **scopes the supersession**: *"the prior receipt's shas were accurate when
written and are superseded for `MANIFEST.sha256` only."* Every claim verified against disk:

| claim | MEASURED |
|---|---|
| R4 `MANIFEST.sha256` repo sha `ab07f6f3b2ce…` | `ab07f6f3b2ce7118ed330b25c12ff1a8eb212d91452a40cd5c1a72a6b715f51c` ✓ |
| frozen gen6 copy still `ba6bbb45…` | unchanged ✓ |
| 36 data rows byte-identical after the edit | **True**, 36 = 36 ✓ |
| r3's `PR_BODY_DRAFT.md` sha still standing | `ec2f8c1b…` unchanged — r3's row remains true ✓ |
| rows verify post-cure | **36 OK** ✓ |
| `review_lineage` | points at my round-4 memo ✓ |

The two receipts compose without contradiction: r3 establishes both documents and the repo-only
status of the PR body; R4 supersedes exactly one field of one of them and says so. A future reader
taking them together gets one true story, with each superseded value explicitly retired rather than
silently overwritten.

---

## ITEM 4 — final sweep for readings that produce a clean-looking wrong value

Swept five candidate readings against the frozen files. **Four cleared, one did not.**

| reading | MEASURED | verdict |
|---|---|---|
| rows sorted vs. manifest order | manifest order **is already sorted** — the sort clause is a no-op | CLEARED (no wrong branch exists) |
| `ensure_ascii=False` | identical digest (paths and hex are ASCII) | CLEARED |
| `bytes` as string, not int | differs, but "bytes" for a file size is not a plausible string | CLEARED (not a reachable reading) |
| row shape object vs array | closed by the R4-F1 cure | CLEARED |
| **`upstream_evaluate_py.relative_path` value** | **`"evaluate.py"` → `e8dcbc65…` · `"upstream/evaluate.py"` → `697d1146…`** | **FINDING R5-F1** |

### RV17-R5-F1 — LOW — the recipe specifies the evaluate.py entry's keys but not its path value

`.omx/research/ddm_pq1_submission_packet_prep_20260815/MANIFEST.sha256:10-16`.

The header says every entry is an object with *exactly the keys* `relative_path`, `bytes`,
`sha256`, and that the inputs are "these files plus the public upstream evaluate.py." For the 36
file rows, `relative_path` is supplied by the manifest itself — unambiguous. For the
`upstream_evaluate_py` entry there is **no source in the document for that field's value**, and the
reader must invent one. Measured:

```
relative_path = "evaluate.py"           -> e8dcbc65…   ← the declared digest
relative_path = "upstream/evaluate.py"  -> 697d1146…   ← silent wrong value
```

The competing spelling is not hypothetical: **line 6 of this same header writes the file as
`upstream/evaluate.py`.** The payload key `upstream_evaluate_py` is a weak hint toward the short
form, but the document never states it, and the wrong branch produces a clean-looking sha with no
error signal — the exact class this item asks me to sweep for.

**Instance cure (one clause):** `"upstream_evaluate_py": {"relative_path": "evaluate.py", …}` —
state the literal value, noting it is relative to the `upstream/` directory, not to the repo root.

**Class cure, which I recommend instead:** ship the derivation as code. A ~10-line
`verify_files_digest.py` in the packet — walk the manifest rows, hash each file, hash
`upstream/evaluate.py`, emit the digest — is unambiguous by construction, testable in CI, and ends a
sequence of three rounds each finding a different under-specified corner of the same paragraph. The
prose can then shrink to one sentence plus a command. If that lands, R5-F1 needs no separate fix.

---

## COUNTER

**0 / 3.** One LOW finding (R5-F1).

Holding the bar in both directions, as agreed. This round I **declined** to raise the non-unique
`inflate.py` gloss, because I executed its wrong branch and it fails loudly — no reader can reach a
false conclusion through it. I raised R5-F1 because its wrong branch is silent and its competing
reading is printed in the same header. Same threshold, applied twice, once in each direction. Had
the sweep turned up only the loud-failure item, the honest verdict would have been **1/3** and I
would have given it.

What I want on the record for whoever runs round 6: the substance has been correct since round 3.
Five rounds of review have not found a wrong number, a wrong pin, an unverifiable claim, or a
mis-scoped receipt — the digest is right, the rows verify, the chain is coherent, and the score is
untouched. What they have found is that a byte-exact recipe written in prose keeps admitting one
more reading. Fix the class, not the sentence, and this closes.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round5_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
