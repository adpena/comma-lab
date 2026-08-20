# ddm_pq11 — the typed candidate swap EXECUTED: packet identity jg5 → rc2 composed

`date_utc: 2026-08-20` · `owner: ddm_pq11` · `score_claim: false (this arm measured no score;
it consumed two retained authority receipts)` · `publish_action: none` · `counter_at_exit: 0/5`

## THE ANSWER, FIRST

**SWAP_COMPLETE_FROZEN.** The packet now binds the selected shipping object. Generation 6 is
staged, its identity is proved from measured bytes at every step, every publication surface was
refreshed against it, and the arm stopped at the `SWAP_PROCEDURE.md` step-4 boundary. No push,
hosting action, PR opening, Modal fire, scorer launch, `upstream/` edit, or edit under
`submissions/robust_current/jg5_sub015_runtime/` occurred.

| identity field | before (gen 5) | after (gen 6) |
|---|---|---|
| archive SHA-256 | `f3bce5d259a08183…` | **`df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`** |
| archive bytes | 180,625 | **180,456** |
| member | `p`, 180,525 B | **`p`, 180,356 B, stored, `83fa979c1118499b…`** |
| runtime rows | 33 | **36** |
| runtime tree | `2103073d…` | **`fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2`** |
| recomputed CUDA score | 0.14839100138338618 | **0.14827847122030852** |
| decode budget verdict | over both ends of the CI window | **PASS, 498.476 s vs an 822 s cold ceiling** |
| CPU axis | "pending" | **MEASURED WALL-INFEASIBLE** |

Frozen generation: `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/`
(46 files) · receipts `…/gen6_receipts/` (11 files).

## The charter's PRIOR-LAW PREDICTION, tested

Predicted: zero old-identity references on live surfaces after the swap, historical rows with
explicit supersession markers exempt. **HELD.** 42 occurrence-lines remain in the prep tree and
11 in the staged tree; every one was classified by reading its context, and all sit in one of
four exempt classes:

| class | example | count of lines |
|---|---|---:|
| preserved prior banner / superseded-marked row | `ADVERSARIAL_REVIEW_SCAFFOLD.md` round-13 row, `archive_manifest.gen5.json` `status: HISTORICAL_SUPERSEDED` | 12 |
| append-only historical section under a READ-§10-FIRST banner | `BORROWED_SUBSTRATE_ACCOUNTING.md` §9, §10.1 lineage table | 8 |
| explicit inter-generation comparison, both rows named | `report.txt` "Prior packet generation 5 measured S = …" | 9 |
| nested under a `*_superseded` key or a labelled comparison key | `PACKET_TARGET.json` `/generation_5_superseded/*`, `ARCHIVE_MANIFEST.json` `composition_vs_bases.packet_generation_5` | 13 |

**Three live-surface violations were found by this census and fixed before freeze**, which is
the falsifier doing its job rather than the prediction being trivially true:

1. `COMPLIANCE_RUNBOOK.md` still carried `— ACTIVE` on its generation-4 header and
   `## Generation 5 (ACTIVE)` on its generation-5 header. Both retired to `HISTORICAL`, and the
   generation-5 block gained a banner saying its invocations will refuse against the shipping
   packet, which is the tool behaving correctly.
2. `PACKET_TARGET.json` `/gpu_routing_decision/measured_finding` quoted the
   `2103073d… → 75a1aeef…` hash pair with no scope. That pair was measured on the 33-row tree.
   Scoped explicitly: the MECHANISM transfers (a tree hash is a function of its row set, so any
   edit to a pinned file moves it — arithmetic, not an empirical finding); the CONSTANT does not.
3. `GENERATION_LOG.md` stated the display-trap claim in the present tense with generation 5's
   number. Re-scoped to name the generation.

## Execution log — step · tool run · MEASURED identity proof

### VERIFY_SOURCE (swap step 1)

Four trigger properties read at swap time, not assumed:

1. **Live pointer candidate.** `.omx/state/canonical_frontier_pointer.json` `effective_frontier`
   carries `df7fd266…` at 0.14827847122030852, `last_refreshed_utc 2026-08-20T18:53:07Z`, which
   post-dates the candidate's own T4 row (started 18:43:06Z). Read from the pointer.
2. **One exact retained archive plus its receiver, from disk.** 180,456 B hashing to
   `df7fd266…`; one member `p`, `file_size == compress_size == 180,356`, method `stored`,
   CRC 3771533310, sha `83fa979c1118499b…`; member-name safety asserted (no absolute path, no
   `..` component, not a directory).
3. **Complete `candidate_seal.v1`.** `CANDIDATE_SEAL_rc2_composed.json`, seal sha
   `2e32079c5de2cff9…`, `SEAL_VALID` at fire time per `FIRE_MANIFEST.json`.
4. **MAIN selection**, by charter, after MAIN retired the obsolete `2103073d… UNCHANGED` guard.

All 36 manifest rows re-hashed from the source tree: **36/36 byte-identical, 0 missing,
0 mismatched.**

### STAGE_NEW_GENERATION (swap step 2)

`tools/stage_contest_submission_packet.py`, run three times: once runtime-only as an identity
probe, then twice more after document edits (the tool refuses to overwrite an existing
generation dir, so each re-stage deliberately removed the prior one — rebuildable from the
source runtime, with the prior receipt retained as its certificate).

Final run, MEASURED:

```
STAGED_TREE_PROVED_IDENTICAL_TO_EVALUATED_TREE
  runtime files verified : 36/36
  runtime_tree_sha256    : fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2
  re-derived (measured)  : fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2
  archive                : df7fd266…e9e2080 (180456 B)
  source census          : 37 real files, 0 undeclared, 15 excluded by class
```

The re-derivation input is freshly measured staged bytes, never the manifest's own claimed
digests — deriving from the rows would be the tautology `rv15` F2 named.

Independent post-stage checks on the FINAL tree: archive re-hashed from the staged copy
(match), member re-parsed (match), all 36 rows re-hashed (36/36), `shasum -a 256 -c
MANIFEST.sha256` → **36 OK, 0 FAILED**.

### RESET_AUTHORITY (swap step 3)

No receipt transferred. The CUDA row is this object's own
(`fc-01M0G7QCQPACVJV29D7AAQSXAA`); the CPU row is this object's own
(`fc-01M0G8AQVBZBEZ5GWVZM5YVX53`); the compliance receipt is **not** carried and is recorded as
OWED. Every derived figure was written from
`gen6_receipts/SWAP_FACTS.json`, built by extracting the two retained receipts — no hand-typed
sha, size or timestamp anywhere in this swap.

### REFRESH_PUBLIC_PACKET (swap step 4)

Sixteen prep documents plus one `.py` refreshed. The consumer is a DIRECTORY, so the list below
is the directory, not a curated subset:

| file | what changed |
|---|---|
| `REPORT_PUBLIC.txt` | full rewrite: identity, components, MEASURED timings, exact inter-generation delta, budget PASS, CPU wall-infeasible |
| `README_PUBLIC.md` | title/target renamed `joint_waterfill_rider`; identity table; three-mechanism framing; new "Runtime and decode budget" section; verification block rewritten; the 183 B rider and "native port does not ship" rows superseded |
| `PR_BODY_DRAFT.md` | embedded `report.txt` replaced byte-verbatim; download field blanked with its reason; eval-host and build-cost blocks filled from the receipt; competitive claim, known limits and public-source disclosure re-derived |
| `ARCHIVE_MANIFEST.json` | rewritten from gen-4 to gen-6 (this was pq10 F3) |
| `archive_manifest.gen6.json` | NEW, staged as the packet's `archive_manifest.json` |
| `archive_manifest.gen5.json` | marked `HISTORICAL_SUPERSEDED` with a pointer to gen 6 |
| `MANIFEST.sha256` | regenerated: 36 rows, hashed from the staged copies |
| `PACKET_TARGET.json` | gen-5 state nested under `generation_5_superseded` (append-only); new active candidate, auth runtime, CPU axis, reproduction and compliance blocks |
| `FREEZE_CHECKLIST.md` | rewritten; (c) and (c2) moved from "optional, do not fold" to "FOLDED, it ships" with the objections answered rather than waived |
| `COMPLIANCE_RUNBOOK.md` | stale `ACTIVE` labels retired; new "Generation 6" section with the exact census + strict-chain invocations and the four expected reds |
| `GPU_ROUTING_VARIANTS.md` | wall-clock section rewritten to the measured PASS; the old variant-(a) hash pair scoped to the tree it was measured on |
| `GENERATION_LOG.md` | gen-5 row retired; gen-6 row added; "What changed at generation 6" written |
| `ADVERSARIAL_REVIEW_SCAFFOLD.md` | new candidate-changed banner (prior banners preserved verbatim); round-13 row added; "Round 14 — what to examine" written, round-13 list preserved and scoped |
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | §10 appended: rider row, native-port row, claim arithmetic, PR #130/#135/#133/#138 boundaries restated |
| `SWAP_PROCEDURE.md` | generation-6 adjudication note appended |
| `GAP_REPORT.md` / `.json` | superseded banners re-pointed at generation 6 |
| `CPU_AXIS_SEALED_FIRE_ORDER.json` | banner extended: the question it was written to answer is closed by measurement; do not fire it |
| `experiments/ddm_pq2_compress_e2e.py` | `NOT_EXPRESSIBLE` entry for `df7fd266…` (see below) |
| `…/ddm_pq7…/packet_staging/COMPRESS.md` | headline re-pinned to the shipping archive |

**The `.py` change was not cosmetic — it repaired a claim that had silently become false.** The
README and PR body both promise that `compress.py`, run against the shipping archive, "refuses
by name, before doing any work". `refuse_if_not_expressible` is a registry lookup keyed by
archive sha; the composed sha was absent, so the script would have fallen through and failed
deep inside the encode stage — the exact over-promise its own docstring warns against. Verified
after the fix: the composed sha now refuses by name, unknown shas still return `None`, and both
pre-existing entries still refuse identically. `ruff check` clean; the `ruff format` drift is
pre-existing at HEAD and untouched by these lines. Two genuine review passes run (key
consumption + receipt-path existence + key-set uniformity; then lint + behaviour invariance),
then `review_tracker mark-file`.

## MANIFEST 36/36 verification

```
$ shasum -a 256 -c MANIFEST.sha256      # from the frozen generation dir
36 lines ': OK'   0 lines 'FAILED'
```

Independently, all 36 authority rows re-hashed from the staged bytes match the receipt's
digests, and the tree hash re-derived from those measured bytes equals `fdd57749…`.

## Census

```
census: 48 declared (36 runtime + 12 non-runtime - 0 in both) | undeclared 0 | missing 2 | CENSUS_CLEAN
prep census: 27 flat document(s) | nested 0 | dot-entries 0 | PREP_CLEAN
rc=0
```

**The AppleDouble recurrence fired again and was caught by the guard, not by memory.** The first
census found **60** `._*` sidecars across the generation and receipts trees — written by macOS
on the ExFAT volume during the staging writes themselves. Purged, re-run, clean. The two
`MISSING` rows are `GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json`, the two hv1-lineage
provenance sidecars this lineage deliberately does not contain (round-11 F2(a), closed by
construction); the guard reports them as information at rc=0.

## What the CORRECTION changed, and what it bought

MAIN's mid-task correction (rv17 F1) was right and material. The charter and the rc2 memo both
said the receipt "does not emit a raw sha or per-stage timing split" and that the inflate wall
was "not separately measured". **Read from the receipt, all three exist.** Verified myself
before using them:

| field | value | source |
|---|---|---|
| `inflate_elapsed_seconds` | 458.752594349 | `contest_auth_eval.json` |
| `evaluate_elapsed_seconds` | 39.72359129999995 | same |
| `contest_budget_verdict` | `PASS`, charged 498.47618564899994 s vs an 822 s cold-cache ceiling | same |
| `raw_sha256` | `6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883` @ 3,662,409,600 B | inflate report in the stdout log + `inflated_outputs_manifest.json` |
| `stage_seconds` | setup 0.5648, selector/IO 3.6088, render 41.9503, token decode 397.8766 | inflate report |
| `decoded_token_sha256` | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`, bit position 910837, `NativeFreeCorrector` | inflate report, **identical on the CPU leg** |

Had those stayed `GATED-ON-RC2`, the packet would have shipped four placeholders over measured
values. Instead the only thing still labelled a projection is the CI residual window
[822, 1302] s, which the receipt genuinely does not measure — its own
`false_authority_warning` says so — and which is now labelled as a projection on every surface
that quotes it.

## The finding this arm made that the charter did not anticipate

**The inter-generation score delta is EXACT, not bound-limited, and the proof was sitting
unread in the two receipts.** Generation 5's T4 row and generation 6's T4 row emit
**byte-identical** n600 inflated output: both `0.raw` streams hash to
`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883` at 3,662,409,600 B **on the
contest-CUDA axis**. Identical scorer input through the same deterministic scorer gives
identical scorer output, so both distortion legs cancel exactly rather than merely agreeing at
8dp, and the whole delta is rate: `25 × (−169) / 37,545,489 = −1.1253016307764696e-04` against a
measured difference of `−1.1253016307766206e-04`.

Two consequences the packet now states:

- The rc2 seal's open falsifier — *"contest-CUDA T4 n600 `0.raw` must equal `6bf8acf8…`; this
  separate-axis condition is untested locally"* — is **MEASURED PASS**.
- The generation-5 body's distortion prose transfers verbatim, and it is *proved* to transfer
  rather than assumed, which is what the charter asked for and what pq10's LIVE-HYPOTHESES row
  flagged as needing a fresh audit.

Note the boundary: the two axes' `raw_sha256` values are **not** equal to each other
(`6bf8acf8…` on CUDA, `2fc5dd3d…` on CPU). The neural render is device-dependent; the token
decode is not. Only the same-axis, cross-generation comparison is an identity claim.

## Measurement boundary

Measured: source and staged archive bytes, member grammar and name safety, all 36 runtime rows
on both source and staged trees, the tree re-derivation, `MANIFEST.sha256` verification, both
retained receipts' timing/identity/component fields, cross-axis token identity, cross-generation
raw identity, the score arithmetic from components, the version-control coverage of the runtime
rows (33/36), the census on both trees, and the `.py` refusal behaviour.

Not measured: any new score, any compliance receipt for these bytes, live HTTP availability of
anything, the CI residual window, and the flipped tree hash for the GPU-routing variant on this
36-row tree.

## NEXT_IF_RESUMED

- **OPERATOR-GATED** — owner: repository operator + MAIN packet owner; consumer store: public
  source commit, commit-pinned archive/receipt URLs, final freeze receipt; fire trigger: the
  operator authorizes the public push; action: `SWAP_PROCEDURE.md` step 4A — push the exact
  180,456 archive bytes plus the three runtime files not yet in version control
  (`inflate.py`, `inflate.sh`, `runtime/residual_archive.py`), derive raw URLs pinned to that
  40-character commit, download fresh, and require HTTP 200 plus SHA-256 and byte-count
  equality. Then fill `PR_BODY_DRAFT.md`'s blank download field and the README's block-3 URLs.
- **QUEUED-FOR-COMPLIANCE-REBUY** — owner: MAIN compliance owner; consumer store:
  `gen6_receipts/pre_submission_compliance.gen6.r1.json` + `COMPLIANCE_RUNBOOK.md`; fire
  trigger: hosted pins and all checker-scanned surfaces final; action: run the census guard on
  both directories (rc must be 0), then the strict chain with the exact invocation already
  written in the runbook's "Generation 6" section, preserving every red with a typed
  disposition. Note the freshness law: any edit to a scanned surface after the buy invalidates
  it.
- **QUEUED-FOR-INDEPENDENT-REVIEW** — owner: a reviewer who did not author this swap; consumer
  store: `ADVERSARIAL_REVIEW_SCAFFOLD.md` five-pass counter; fire trigger: compliance re-buy
  complete; action: execute every appendix command against the frozen generation and begin
  round 14. The nine-item examination list is written; item 3 is the one that matters most
  (attack the raw-identity claim, because several sentences depend on it).
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `GPU_ROUTING_VARIANTS.md` +
  `PACKET_TARGET.json`; fire trigger: the operator chooses variant (a); action: measure the
  flipped `runtime_tree_sha256` on THIS 36-row tree and buy the new T4 row. The old
  `75a1aeef…` is scoped to generation 5 and must not be reused.
- **OPERATOR-OWNED** — owner: repository operator; consumer store: the final public PR
  description; fire trigger: the packet reaches the release bar; action: write the LLM/policy
  answer in the operator's own words and explicitly authorize publication.

## Retained custody

- Frozen generation: `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/`
  — 46 real files, per-file sha/bytes inventory in `gen6_receipts/RETAINED_CUSTODY_MANIFEST.json`.
- Receipts: `…/generations/gen6_receipts/` — `contest_auth_eval.json`, `provenance.json`,
  `report.txt`, `inflated_outputs_manifest.json`, `modal_cuda_preflight.json`,
  `modal_cuda_auth_eval_validation.json`, both stdout/stderr logs, `STAGING_RECEIPT.json`,
  `STAGING_RECEIPT.runtime_only.json`, `SWAP_FACTS.json`, `RETAINED_CUSTODY_MANIFEST.json`.
- Authority receipts remain in their own custody and were not copied into the packet:
  `/Volumes/APDataStore/pact/ddm_rc2/t4_row_r2/MODAL_REMOTE_RESULT.json` (CUDA) and
  `/Volumes/APDataStore/pact/ddm_rc2/cpu_row_r1/MODAL_REMOTE_RESULT.json` (CPU).
- Prior generations untouched in their retained stores, as the procedure requires.

OWN-VEHICLE FRONTIER: unchanged by this arm at
**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**, archive
`df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`. This unit moved the PACKET
onto that object; it did not move the score.
