# Review pass 6 — fresh eyes, adversarial

**Verdict: 6 FINDINGS. The counter stays at `0/5`.**

Reviewer: fresh-eyes arm. I had no part in building any generation of this
packet, and no part in the round-5 fixes (commit `71e6d0a076`). Date:
2026-08-18. Candidate: gen-3 `fx2_a__tuned` (sz1 composed),
`debb025f45bb42e3…`/179,930 B. Read-only: I edited, fixed, staged, committed,
and deleted nothing.

**The bytes are clean and all eight round-5 fixes landed correctly.** I
re-derived the hash chain and the score from disk; both match to the last
digit. I diffed every round-5 fix and each one is correct — including the two
hardest to fake, the verbatim `report.txt` copy and the `.pyc` purge.

Every finding below is a **stale statement in a governing document**. They share
one root cause, and I state it plainly because it will keep firing otherwise:
`SWAP_PROCEDURE.md` step 4 (`REFRESH_PUBLIC_PACKET`) names a closed consumer
list — `README.md`, `report.txt`, `archive_manifest.json`, PR body, accounting
table, packet target, CPU fire order. Five documents in the same directory sit
outside that list: `ARCHIVE_MANIFEST.json`, `GAP_REPORT.md/.json`,
`COMPLIANCE_RUNBOOK.md`, `CONTRIBUTION_ETIQUETTE.md`, and the scaffold's own
swap note. The refresh step **structurally cannot reach them**, so they still
describe generation 0 or the pre-fix state. Round 5 fixed the surfaces it was
pointed at; nobody swept the siblings.

---

## Findings

| # | Axis | Finding | Evidence | Severity |
|---|---|---|---|---|
| F1 | 2 — round-5 F1 regression at a sibling surface | `ARCHIVE_MANIFEST.json` still declares **`"reproduction_verified": false`** and **`"reproduction_entry_point": "PENDING_REBIND"`**. Every other surface says VERIFIED. The file even points the reader at the block that contradicts it ("see PACKET_TARGET.json reproduction block"). This is exactly round-5 F1 — a stale `PENDING_REBIND` — surviving at a machine-readable surface. Fix commit `71e6d0a076` did not touch this file; it has been stale since `d6aabb4b5e` landed the VERIFIED status. | `ARCHIVE_MANIFEST.json:50-51` vs `PACKET_TARGET.json` `.reproduction.status = VERIFIED` (`verified_at_utc` 2026-08-18T05:12:00Z) · packet `report.txt:43` · `REPORT_PUBLIC.txt:43` · `README_PUBLIC.md:108` · `GENERATION_LOG.md` "Reproduction at generation 3" · `PR_BODY_DRAFT.md:299` · `git log -- ARCHIVE_MANIFEST.json` → last touched `f4a3882345` | **MEDIUM-HIGH** |
| F2 | 2 — round-5 F6 regression inside the counter authority | The scaffold's **Generation-3 swap note contradicts the scaffold's own header**. It states the counted passes require "a green strict compliance chain — which additionally requires the reproduction re-bind (`PENDING_REBIND` in `PACKET_TARGET.json`) to land first," and pins the packet documents to commits `f4a3882345 / e54bdfa37e / 9fe84725f5`. All three clauses are stale: the re-bind landed (status VERIFIED), the green-chain precondition is superseded by the same document's adjudication clause reading it against the terminal 82/86, and `71e6d0a076` changed the packet documents after those three commits. A reader who reaches the swap note concludes no pass can be counted at all. | `ADVERSARIAL_REVIEW_SCAFFOLD.md:89-95` vs `:3` and `:16-28` · `PACKET_TARGET.json` `.reproduction.status` · `git show --stat 71e6d0a076` | **MEDIUM** |
| F3 | 7 — self-imposed publication gate vs public answer | **Two governing documents instruct answering the compression-source question "no"; the PR body answers "Yes."** The runbook requires "a sanitized, seeded, documented reproduction bundle that consumes public/pinned inputs… Until that real proof exists, answer the template question 'no' and do not imply reproducibility," and states the gate "remains open for the current packet." `GAP_REPORT.md` repeats it ("until then the PR answer remains 'no'"). The verified e2e chain consumes **retained private checkpoint state**, not public inputs — the PR body says so itself ("Stage A — provenance (documented, not re-run)"). No adjudication note reconciles the two, though `SWAP_PROCEDURE.md` received exactly such a note in round 3 for its unsatisfiable literals. | `COMPLIANCE_RUNBOOK.md:56-59, 61-62` · `GAP_REPORT.md` "Additional publication gap" · `PR_BODY_DRAFT.md:121-143` · `PR_BODY_DRAFT.md:125-128` (Stage A) | **MEDIUM** |
| F4 | 4 — compliance-receipt pointer freshness | **The runbook names the r4 receipt nowhere.** Its "Generation 3" section cites `pre_submission_compliance.gen3.r2.json` with the headline "80 GREEN / 6 RED"; the correction block cites r3. The receipt that actually binds the packet on disk is **r4** (`f13030171df65100…`), run after the round-5 fixes changed `report.txt`, staged two documents, and corrected the manifest key. Round 5 declined to file the r2-header version because "the correction is immediately below" — that rationale no longer holds for a receipt named nowhere. Same section, lines 32-39 assert in the present tense that "the packet's executable runtime still matches… full tree `77b94b5c…` and portable content tree `26c7d418…`" — those are generation-0 values, unlabelled; gen-3's are `0d0fc008…` and `994f8aaa…`. | `COMPLIANCE_RUNBOOK.md:64-71, 79-80, 32-39` vs `gen3_receipts/pre_submission_compliance.gen3.r4.json` · `ADVERSARIAL_REVIEW_SCAFFOLD.md:14` (which does cite r4) | **LOW-MEDIUM** |
| F5 | 4 — the gap report states generation 0's gaps as current | `GAP_REPORT.md` and `GAP_REPORT.json` were last touched at the original gen-0 landing (`dec5402577`) and were never refreshed across three candidate swaps. They carry **no generation label** and a present-tense disposition: "The final real strict check evaluated 86 checks: **78 passed and 8 failed**," binding archive `e3e6f440…` and runtime tree `77b94b5c…`. Three of the listed reds (`auth_eval_schema_metric_consistency`, `dispatch_claim_successful_exact_eval_terminal_row`, and the two `dispatch_claim_*_sha_bound` rows) are **GREEN at r4**. The document whose sole job is to state the current gaps states the wrong ones. | `GAP_REPORT.md:5-6` and red table vs `pre_submission_compliance.gen3.r4.json` (82/4, four named reds) · `git log -- GAP_REPORT.md` | **LOW-MEDIUM** |
| F6 | 7 — stale CPU-axis framing in the etiquette gate | `CONTRIBUTION_ETIQUETTE.md` states "the exact-byte CPU row is **marked pending**." The gen-3 packet does not mark it pending. `PACKET_TARGET.json` records `MEASURED_INFEASIBLE_WITHIN_CONTEST_BUDGET`, and its adjudication says "this receipt **replaces the pending-CPU-row debt**"; `SWAP_PROCEDURE.md` says "the axis is adjudicated MEASURED-INFEASIBLE, **not pending**." The runbook directs a reviewer to apply this document "against the active packet generation" before recommending a PR. | `CONTRIBUTION_ETIQUETTE.md:21` vs `PACKET_TARGET.json` `.cpu_axis` · `SWAP_PROCEDURE.md:72-77` · `COMPLIANCE_RUNBOOK.md:43-44` | **LOW** |

**Considered and NOT filed.** The accounting is titled "generation 3 (rr4
re-encode candidate)" while the packet calls rr4 generation 2, and §2 is written
against the 181,161-byte candidate. §7's numbering note discloses both schemes
explicitly and §7 amends every changed row append-only. It is self-consistent
and disclosed, and it now ships with that disclosure inside it. Not a finding —
I reached the same conclusion round 5 did, independently.

Also not filed: the CPU sub-timings do not sum to the total (3,108.7 + 299.3 =
3,408.0 against 3,422.7). The 14.7 s remainder is unattributed harness overhead,
the receipt's own floats reproduce both figures exactly, and no claim rests on
the sum.

---

## Per-axis verification notes

### Axis 1 — hash chain from disk: **CLEAN**

Recomputed on this machine. Nothing taken on trust:

```
debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a  archive.zip  179930 B
e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62  inflate.sh     2203 B
5c5baf88bd3e0b9860c80d496333bba902ee986ef5ee1dbc0a7724ba948ad6bd  inflate.py     2282 B
6c41f7faa5a951d905e23651d615b88fe8f1cbbeef2f330627b22d227885203f  report.txt     3716 B
```

ZIP internals read through `zipfile`:

```
name='p'  compress_type=0 (stored)  size=179830  csize=179830
crc=3747474564 (0xdf5dec84)
member sha256: be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8
members total: 1
```

All four pinned values match. Single member, stored, no duplicates.

Score re-derived from the report's 8-dp components and the exact archive bytes:

```
seg  = 100 * 0.00029611          = 0.029611
pose = sqrt(10 * 0.00000688)     = 0.008294576541331089
rate = 25 * 179930 / 37545489    = 0.11980800143527229
S                                 = 0.15771357797660338
```

Float-equal to the receipt's `canonical_score`, to `PACKET_TARGET.json`, to both
manifests, to `report.txt`, and to the PR body. The rate term inverts cleanly:
`rate * 37545489 / 25 = 179930.0`.

Derived figures also check. `1800 / 1143.270127967 = 1.5744` ("1.574x" /
"1.57x"). `3422.711146813 / 1800 = 1.9015` ("1.90x over budget"). CPU
sub-timings `3108.7304750509998 → 3,108.7 s` and `299.2728009090001 → 299.3 s`.
Byte lineage is coherent: `182,759 − 1,598 = 181,161 − 711 − 520 = 179,930`, and
the claimed cumulative `2,829 = 1,598 + 711 + 520`. Versus generation 2:
`181,161 − 179,930 = 1,231` B and `ΔS = 0.15853325034789678 − 0.15771357797660338
= 0.00081967237`.

Source pins resolve. `provenance.pact_commit` in the receipt is
`2e0af59966c4a1405bad342de5969d0de4d99f7a`, exactly as the PR body claims. All
cited commits resolve in git (`31c64e4ce0`, `85880c77a6`, `6449c7cdd5`).
`upstream/evaluate.py` on disk hashes to `7da71a84ce24286b…`, matching
`report.txt`. Document line citations resolve too: `inflate.sh:27` is the
`uv pip install` line the compliance red names, and `inflate.sh:32` is the
`${CC:-cc}` compile the accounting §1.3 names.

### Axis 2 — round-5 fix regression: **F1, F2**

Eight fixes, checked one at a time against disk.

| Fix | Verdict | Proof |
|---|---|---|
| F1 — `report.txt` Reproduction says VERIFIED | **LANDED** | `report.txt` `=== Reproduction ===` reads "Status: VERIFIED", with pre-split `9de0f6db`/180,450 B and final `debb025f…`/179,930 B. Both shas correct. |
| F2 — PR body block is a verbatim copy | **LANDED, exactly** | Extracted the ```` ```text ```` fence programmatically. `fence + "\n"` hashes to `6c41f7fa…` — **string-equal to the shipped file**. 3,715 + 1 = 3,716 B. |
| F3 — no `constriction`; NumPy declared | **LANDED** | Zero `constriction` hits in the packet and in every public document. Survives only in the three review-history files that record the finding. NumPy now appears in the PR body, `README_PUBLIC.md:147`, and accounting §1.3 — and the README's count "four things" matches its four bullets. |
| F4 — `README.md` + accounting staged | **LANDED** | Both present. Both **byte-identical** to their repo copies (`eabc8564…`, `ffe913e2…`). |
| F5 — no `.pyc` / `__pycache__` / `._*` | **LANDED** | `/usr/bin/find` returns **zero** matches. 39 files total = 34 runtime + `archive.zip` + `archive_manifest.json` + `report.txt` + the two staged documents. |
| F6 — single counter authority | **PARTIAL → F2** | `SWAP_PROCEDURE.md:84-86` now defers correctly ("the SINGLE counter authority — this document does not carry its own count"). But the scaffold's own swap note still carries stale preconditions. |
| F7 — manifest portable-tree key | **LANDED** | `archive_manifest.json` `portable_runtime_content_tree_sha256 = 994f8aaab28ec1ff…`. I checked the semantics rather than the string: the r4 receipt's `portable_runtime_tree_sha256_without_custody_files` and the auth-eval's `runtime_content_tree_sha256` are **both** `994f8aaa…`, over the same 34 files. The key now holds the value both sides agree on. |
| F8 — offset alignment-noise qualification | **LANDED, all three** | PR body accounting row ("~22 B of the win is Brotli alignment noise… adjacent offsets swing ±20 B"), `README_PUBLIC.md:54-56`, accounting §7.2:215-217. |

Adding the two `.md` files did not disturb the pinned tree: the r4 receipt was
written at 04:56, one minute after they were staged at 04:55, and still reports
`runtime_file_count = 34` with the portable tree unchanged. The runtime manifest
walks executable and receipt files, not documentation.

### Axis 3 — public-text hygiene: **CLEAN**

Scanned `PR_BODY_DRAFT.md`, `README_PUBLIC.md`, `REPORT_PUBLIC.txt`, the packet
`report.txt`, and **both newly staged packet files** for `/Users/`, `/Volumes/`,
`/tmp/`, `/root/`, `/workspace/`, Tailscale and RFC-1918 addresses, credentials
and bearer tokens, Modal call ids (`fc-01…`), provider hostnames, and any
Claude/Anthropic/AI attribution or co-author trailer.

**Zero hits in all seven files.** The two staged documents are as clean as the
five that were already reviewed.

`GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` still carry generator
absolute paths. That is disclosed in the public README, superseded by
`CUSTODY_SUPERSEDED.json`, and unfixable without breaking the pinned tree hash.
Not a finding, on the same reasoning round 5 gave.

### Axis 4 — compliance receipt r4: **F4, F5**

`pre_submission_compliance.gen3.r4.json` parses to **86 checks, 82 passed, 4
failed** — exactly as pinned. Receipt sha `f13030171df65100…`. It binds the
current packet: `archive.sha256 = debb025f…`, `bytes = 179930`,
`runtime_file_count = 34`,
`portable_runtime_tree_sha256_without_custody_files = 994f8aaab28ec1ff…`.

The four reds are exactly the four named, each with a documented route:

| Failed check | Route |
|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | runbook red 2 — documented structural; running the adjudicator would downgrade the payload |
| `contest_cpu_auth_eval_exists` | runbook red 3 — CPU axis measured infeasible, structural |
| `submission_runtime_has_no_network_install_or_local_paths` | runbook red 4 — `inflate.sh:27` pinned-wheel Brotli bootstrap, by design, e4 precedent |
| `hosted_archive_manifest_supplied` | runbook red 5 — operator-gated hosting |

One-to-one: no red without a route, no route without a red. I read the checker's
own detail strings, not the prose. The `inflate.sh:27` hit is the literal
`uv pip install` line; the structural red is the unconditional blocker trio.

I also confirmed the run used the right pins rather than trusting the totals:
`expected_archive_sha256_matches`, `expected_archive_size_bytes_matches`,
`auth_eval_runtime_tree_expected_match`, `submission_runtime_tree_matches_auth_eval`,
all ten `archive_manifest_*` checks, `auth_eval_score_recomputes`,
`public_text_has_no_unresolved_template_placeholders`, and
`frontier_no_regression_on_submitted_axis` all pass.

The receipt is sound. F4 and F5 are about the two documents that describe it and
have not been refreshed to match.

### Axis 5 — `REPORT_PUBLIC.txt` byte-identity: **CLEAN**

```
6c41f7faa5a951d905e23651d615b88fe8f1cbbeef2f330627b22d227885203f  REPORT_PUBLIC.txt   3716 B
6c41f7faa5a951d905e23651d615b88fe8f1cbbeef2f330627b22d227885203f  packet report.txt   3716 B
```

`diff` reports identical. Both now carry the VERIFIED reproduction text, so the
round-5 fix propagated to both copies rather than one.

### Axis 6 — borrowed-substrate accounting: **CLEAN**

The accounting holds up, and it now ships beside the archive where its two
public references said it would.

Lineage is section-level and byte-bound across PR #130, #133, #135, #136, and
#138, including the `rc64_backend.c` rows. The closed 4-value class set is
stated and obeyed. The document is repeatedly unflattering to its author: it
**withdraws** the `ours-original` label on the residual payload for want of a
receipt (§6.1), flags the compressed-model container as "PR-level equality not
independently verified" (§6.2), and records that the shipped RC64 receiver
backend `05839d14…` **differs** from PR135's source `5c75e2c7…`. §7.2 declines
to call a standard shuffle filter original.

No whole-vehicle originality claim exists. Both documents say the opposite in
plain words: "This remains a lossless re-encode program on a PR130/PR135 learned
substrate, not a claim that the learned vehicle is original." The PR #138
concurrency disclosure concedes that PR #138 published the mechanism class
first, gives the timeline to the minute, and makes no priority claim.

The PR body's inline table and the accounting agree row for row, including the
offset qualification added in round 5.

### Axis 7 — template conformity: **F3, F6**

All six upstream headings from `upstream/.github/pull_request_template.md` are
present, in order, and answered in plain language: submission name, upload,
`report.txt`, GPU-required, compression script, additional comments. Four extra
headings (eval host, build cost, changes from upstream, competitive or
innovative) come from the newer template `CONTRIBUTION_ETIQUETTE.md` requires. A
superset is fine.

The template instruction "copy the report.txt content here" is now honoured
literally — verified byte-for-byte above. That was round-5 F2 and it is the
fix I checked hardest, because it is the one that hides other defects.

Other etiquette gates hold:

- **One PR, one archive URL.** No URL is claimed; the draft says hosting is
  pending operator authorization, and `hosted_archive_manifest_supplied` is red
  by design. No archive is checked into the tree.
- **CPU and CUDA labelled separately.** Both axes carry their own tags. The CPU
  entry reports a measured boundary, not a promise. No score crosses bytes or
  axes; the generation-2 comparison names the other archive and its own score.
- **No dependency-file mutation, no provider transcripts, no machine
  attribution.** Verified by scan.

The two failures are F3 (the packet's own publication gate says answer "no";
the body answers "Yes", with no adjudication note) and F6 (the etiquette
document still describes the CPU row as pending).

---

## What I would fix before pass 7

Small, and mostly mechanical:

1. Set `ARCHIVE_MANIFEST.json` `reproduction_verified: true` and replace
   `PENDING_REBIND` with the verified entry point (F1).
2. Rewrite the scaffold's Generation-3 swap note to defer to its own header and
   adjudication clause, and drop the superseded commit pins (F2).
3. Decide F3 deliberately, then record the decision. Either add an adjudication
   note explaining why "Yes" is right when the bundle consumes retained inputs,
   or change the answer. Do not leave the gate and the body disagreeing.
4. Point the runbook at r4 and label the generation-0 tree hashes as
   generation-0 (F4).
5. Give `GAP_REPORT.md/.json` a generation label, or refresh it to the r4 red
   set (F5).
6. Update the etiquette document's CPU line to "measured infeasible" (F6).

Then fix the cause, not just the six symptoms: **extend `SWAP_PROCEDURE.md` step
4's consumer list** to name every document in the packet-prep directory, so the
next refresh cannot miss the same five files again. A refresh step with a closed
consumer list is a gate that cannot see part of what it governs.

None of this touches `archive.zip`, the runtime tree, the compliance receipt, or
the measured row. The score, the bytes, and the CUDA authority are sound, and
the eight round-5 fixes are real.

---

*Read-only review. No file in the packet, the receipts, or the repository was
modified by this pass. MAIN records the scaffold row.*

## Erratum — the F1 row's manifest citation is stale (rv17 round 14, R14-F1 derived-set find)

The F1 row cites `ARCHIVE_MANIFEST.json:50-51` for the then-live `reproduction_verified: false`
defect. That citation was generation-correct (the gen-5-era working manifest carried those
fields at those lines) and is stale against the SHIPPED gen-6 packet three ways: the archive
manifest is regenerated per candidate (the shipped lowercase `archive_manifest.json` is 20
lines), the uppercase working name does not resolve on the contest's case-sensitive Linux, and
the F1 defect itself was subsequently CURED (reproduction status VERIFIED). The finding and its
resolution stand as history; only the citation dangles.

covered-citation: `ARCHIVE_MANIFEST.json:50`
