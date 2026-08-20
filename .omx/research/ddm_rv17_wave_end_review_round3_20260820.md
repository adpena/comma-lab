# ddm_rv17 — wave-end adversarial review, ROUND 3: one finding, counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [receipt + source review, scorer-free]` ·
`score_claim: false` · cost $0 · third sibling of `ddm_rv17_wave_end_review_round1/round2_20260820.md`.
Counter authority for the #1157 wave-end 3-pass cycle (the packet's `ADVERSARIAL_REVIEW_SCAFFOLD.md`
0/5 counter is separate and untouched).

## THE ANSWER, FIRST

**Counter 0/3 — one MED finding, and it is a defect in MY OWN round-2 recommendation.**

R2-F1 is fully cured. R2-F2's cure is *half* right, and the wrong half is mine: I told the packet
to publish `runtime_content_tree_sha256 ccd9f7ab…` as "the digest a reviewer can reproduce from
exactly these files." **It is not.** I executed the derivation: `ccd9f7ab…` reproduces only when
the **repo-local tac import manifest** is folded in — the very input the corrected sentence two
lines above names as disqualifying for `fdd57749…`. The cured header therefore contradicts itself,
and it does so because it followed my advice.

The digest that IS reviewer-reproducible is **`runtime_files_sha256 = e8dcbc6542d6f4752559726a…`**,
which the harness's own source comment calls the *"Environment-free custody digest."* I proved it
end-to-end by measuring all 37 inputs myself — the 36 staged files plus `upstream/evaluate.py` —
and reproducing the declared value exactly. It was already recorded in `PACKET_TARGET.json`; the
cure simply picked the wrong one of the two portable digests sitting side by side.

**The labels-first sweep — my own round-2 recommendation for where to look — came back CLEAN.**
Every heading, one-line summary, table caption, and JSON `*_note` field across the five 08-20
landing memos and four packet JSONs checks out against its body. The residual defect of this wave
was not in a label after all; it was in a *cited value*.

---

## ITEM 1 — did the two R2 cures hold?

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 1a | R2-F1 heading | read committed text at `:27` | now reads **TOKEN level (decoded_token_sha256 match)** | **CLEAN — cured** |
| 1b | R2-F2 `fdd57749` clause | read both surfaces; cross-axis measurement | accurate, and **validated**: CPU axis yields `90244e17…` ≠ `fdd57749…` | **CLEAN — cured** |
| 1c | R2-F2 two-surface consistency | diff MANIFEST header vs PR-body appendix | both state the same thing | **CLEAN — consistent** |
| 1d | R2-F2 `ccd9f7ab` framing | **executed both canonical derivations** | `ccd9f7ab` NOT reproducible from the 36 files | **FINDING R3-F1** |

**1a.** `.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md:27` now reads
`## DECODE IDENTITY — confirmed at the TOKEN level (decoded_token_sha256 match); the rr2 desync did
NOT occur`. Heading and body agree. Cured.

**1b is better than a read — it is now measured.** The corrected clause says `fdd57749…`
additionally binds the root directory name, the tac import manifest, and `upstream/evaluate.py`,
so it is environment-coupled. The CPU-leg receipt confirms it empirically: the *same 36 files*
produce `runtime_tree_sha256 = 90244e171009aac9291847cfeb7d1d77b949f165b4bbebcae037afe9cc2ef519`
on the CPU axis versus `fdd57749…` on CUDA, while **both** portable digests are identical across
axes. The env-coupling claim is not merely argued; it is demonstrated.

**1c.** The two surfaces are consistent — they say the same thing. Noting honestly that
consistency and correctness are different tests: both surfaces agree, and both are wrong in the
same way.

### RV17-R3-F1 — MED — the cure names a digest that is not reviewer-reproducible (my error, inherited)

`.omx/research/ddm_pq1_submission_packet_prep_20260815/MANIFEST.sha256` (header comments) and
`PR_BODY_DRAFT.md` (verification appendix).

Both now say the reviewer-reproducible digest is `runtime_content_tree_sha256 ccd9f7ab…`.
MEASURED, by executing the canonical payload construction from
`experiments/contest_auth_eval.py:515-548`:

```
content_tree recomputed (files + ext_roots + repo_local_tac + evaluate.py) = ccd9f7ab…  ← MATCHES declared
files-only variant                                                        = 50d86f20…  ← does NOT match
runtime_files_sha recomputed (files + evaluate.py)                        = e8dcbc65…  ← MATCHES declared
```

`ccd9f7ab…` reproduces **only** with `repo_local_tac_import_manifest` included — a nine-key scan of
*our* repository layout (`tac_root_relative_path`, `discovery`, `files`, `parse_errors`,
`root_import_modules`, …) that a public reviewer does not possess. So the header disqualifies
`fdd57749…` for binding the tac import manifest, then in the next sentence nominates a digest that
binds it too. That is self-contradiction inside a single comment block.

**The correct digest, proven reviewer-side end-to-end.** I measured every input myself — the 36
staged files read as bytes, plus `upstream/evaluate.py` (6,005 B, `7da71a84…`) — with no value
taken from any receipt:

```
REVIEWER-SIDE files_sha = e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f
DECLARED runtime_files  = e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f   MATCH
```

The harness source agrees by construction — the comment at `contest_auth_eval.py:549-556` describes
`runtime_files_sha256` as *"Environment-free custody digest: ONLY the runtime files'
(relative_path, bytes, sha256) plus the upstream evaluate.py identity. Deliberately excludes
runtime_root_name, absolute paths, external dependency roots, and the repo-local tac import scan."*
That comment sits beside `runtime_files_sha`, not beside `content_tree_sha`; my round-2 reading
attached it to the wrong digest and the cure inherited the error.

**Two mitigating facts, stated so the cure is not over-punished.** First, `PACKET_TARGET.json`
`/auth_runtime/cross_axis_tree_note` is **accurate**: it names *both* portable digests and claims
only that they are "equal on both axes" — which my CPU-leg measurement confirms. The overclaim was
introduced by the two cured surfaces, not by the packet's own record, and `e8dcbc65…` was already
present there (2 occurrences). Second, the cured header cites that note as its support, but the
note's claim is *portability*, not *reviewer-reproducibility* — a weaker and different property.
The citation does not carry the sentence built on it.

**CURE:** in both surfaces, replace `runtime_content_tree_sha256 ccd9f7ab…` with
`runtime_files_sha256 e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f`, and state
its inputs explicitly: *the 36 rows' (relative_path, bytes, sha256) plus `upstream/evaluate.py`'s
identity — both of which the reviewer already has.* `ccd9f7ab…` may still be quoted as a portable
cross-axis identity, never as reviewer-reproducible. **Owner: mine to route, since the error is
mine.** Round-1 `RV17-F7` remains open in the same header: `Verify: sha256sum -c MANIFEST.sha256`
still omits its required working directory.

---

## ITEM 2 — the labels-first sweep — **CLEAN**

Ran per my own round-2 recommendation: read every label first, then test it against its body.

| surface | scope | method EXECUTED | result |
|---|---|---|---|
| 5 landing memos (rc2, pq11, sw2, rv17 r1, rv17 r2) | 47 headings | extract all `^#{1,3}` lines, check each against its section body | **CLEAN** |
| `PACKET_TARGET.json` | 23 `*_note` fields | recursive walk, read each against my own measurements | **CLEAN** |
| `ARCHIVE_MANIFEST.json` | 7 `*_note` fields | same | **CLEAN** |
| `GAP_REPORT.json` · `archive_manifest.gen6.json` | 2 · 0 | same | **CLEAN** |

Spot-checks that could have failed and did not:

- `ARCHIVE_MANIFEST.json /decoded_state_note` — "both rows' n600 inflated output hashes to
  `6bf8acf8d4412e43`" — matches my round-2 measurement exactly.
- `/cuda_axis_note` — "inflate 458.752594349 s, evaluate 39.72359129999995 s, charged
  498.47618564899994 s" — matches the receipt to the digit.
- `PACKET_TARGET /auth_runtime/cross_axis_tree_note` — the `90244e17…` CPU value is correct;
  I re-derived it from the CPU receipt independently.
- Every `reproduction_note` is honestly **negative** ("has NOT been re-run", "no prior VERIFIED
  label transfers") rather than quietly inheriting a prior generation's label.
- All five `generation_N_superseded/note` fields carry explicit APPEND-ONLY supersession markers.

**One apparent hit, honestly a false positive of my own method.** A grep-based outline extraction
flags `ddm_rv17_wave_end_round2:51` as a heading reading *"confirmed at the component level."* It is
inside a fenced code block — a quotation of the defect, attributed to
`ddm_rc2_t4_row_sixteenth_move_20260820.md:27`. Renders as code, not a heading. Recording it so a
future outline-based sweep does not re-raise it.

**Two round-1 findings remain correctly open, not regressed:** sw2's `RV17-F3` (title says "16
hits" — correct at row level — while the body still says "The 3 JWTs" against 4 measured jwt rows,
with no reconciling sentence) and `RV17-F7`. Both routed, neither silently patched.

**And one round-1 finding was properly ROUTED rather than quietly fixed:** `RV17-F6` now appears in
`PACKET_TARGET.json /generation_5_superseded/auth_runtime/harvest_decode_note` — *"The harvested
receipts were persisted as Python bytes-reprs (files literally begin `b'`)"* — with
`generations/gen5_receipts/HARVEST_DECODE_RECEIPT.json` existing on disk. That is the correct
handling for a defect in sealed custody: record it, do not edit it.

---

## ITEM 3 — does the SSD-vs-repo comment divergence need a receipt row?

**Adjudication: YES — and it is cheap, because the proof is already measured.**

MEASURED scope of the divergence:

```
repo prep MANIFEST.sha256   sha d041a57feecf5c6a…   36 data rows
gen6 frozen MANIFEST.sha256 sha ba6bbb45d499e43f…   36 data rows
DATA ROWS byte-identical : True          whole-file sha differs : True (comments only)
MANIFEST.sha256 is itself a runtime row : False
archive.zip is a runtime row            : False
```

The divergence **cannot** move any digest or the score: the 36 data rows are byte-identical, and
`MANIFEST.sha256` is not among the 36 runtime rows, so it is hashed into none of
`runtime_tree_sha256` / `runtime_content_tree_sha256` / `runtime_files_sha256`. I re-confirmed the
frozen tree still verifies **36/36 OK** in round 2 and the row set is unchanged.

But a row is still owed, for one reason: two copies of a publish-source document now differ in
text, one of them inside frozen custody, and **nothing on disk tells a future reader which is
authoritative.** That is the stale-sibling class the packet's own `SWAP_PROCEDURE.md` names — *"a
stale present-tense sibling is a review finding by definition."* Frozen custody is append-only, so
the cure is a new receipt, never an edit to the frozen file.

**Named row: `gen6_receipts/DOC_DIVERGENCE_RECEIPT.json`**, recording:

| field | value |
|---|---|
| `schema` | `pact.packet_doc_divergence.v1` |
| `document` | `MANIFEST.sha256` |
| `publish_source` | the repo prep copy (authoritative) |
| `frozen_copy` | the gen6 custody copy (pre-fix comments, retained as evaluated) |
| `publish_source_sha256` / `frozen_copy_sha256` | `d041a57f…` / `ba6bbb45…` |
| `delta_class` | `comments_only` |
| `data_rows_identical` | `true` (36 = 36, byte-identical) |
| `document_is_runtime_manifest_row` | `false` — hashed into no digest |
| `score_effect` | `none` — proven by the two rows above |
| `curing_commit` | `197f5b19d9` |
| `reason_frozen_copy_not_edited` | append-only custody; the evaluated tree ships unmodified |

I did not create it — writing into frozen custody is the packet owner's action, and R3-F1 should
land in the same batch so the receipt records the *final* comment text rather than an interim one.

---

## COUNTER

**0 / 3.** One MED finding (R3-F1). The falsifier — a genuinely clean pass — remains unclaimed.

The honest shape of this round: the labels-first hypothesis I proposed in round 2 was **wrong about
where the residual defect lived.** Every label checked out; the defect was a *cited value* inside a
correctly-structured, internally-consistent, two-surface cure. And its origin was my own round-2
recommendation, which named the wrong one of two adjacent digests because I attached a source
comment to the function above it instead of the function beside it. A review arm's recommendations
are unreviewed new claims, exactly like a fix is unreviewed new code — round 4 should re-derive
R3-F1's proposed cure (`e8dcbc65…`) rather than adopt it on my word, as I failed to do for my own.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round3_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
