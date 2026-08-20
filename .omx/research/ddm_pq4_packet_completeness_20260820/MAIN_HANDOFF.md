# ddm_pq4 → MAIN handoff — what to re-stage, what is owed

`date_utc: 2026-08-20` · `owner: ddm_pq4` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.**
Unmoved by this arm.

---

## 1. The hash-safety check I ran BEFORE editing anything

pq3's finding #2 is that **any** edit to a file in the 33-row runtime manifest moves
`runtime_tree_sha256` and voids the exact row. So I verified membership from the authority
receipt rather than assuming it:

```
runtime manifest rows: 33
  README.md                        IN MANIFEST: False
  report.txt                       IN MANIFEST: False
  BORROWED_SUBSTRATE_ACCOUNTING.md IN MANIFEST: False
  archive_manifest.json            IN MANIFEST: False
  archive.zip                      IN MANIFEST: False
  inflate.sh                       IN MANIFEST: True
  inflate.py                       IN MANIFEST: True
```

Source: `gen5_receipts/provenance.json → inflate_runtime_manifest.files`.

**The three documents I edited are outside the pin. Re-staging them cannot move
`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` and cannot invalidate
the 0.14839100138338618 row.** I touched nothing inside the manifest, nothing inside
`gen5_jg5_waterfill/`, and nothing under `submissions/robust_current/jg5_sub015_runtime/`.

## 2. What MAIN re-stages at freeze

The packet copies were byte-identical to the repo bundle at HEAD before my edits — verified by
sha, all three pairs matched — so the bundle is the source and re-staging is a straight copy
through the canonical stager.

| Repo bundle file (edited) | Packet destination |
|---|---|
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/BORROWED_SUBSTRATE_ACCOUNTING.md` | `gen5_jg5_waterfill/BORROWED_SUBSTRATE_ACCOUNTING.md` |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/README_PUBLIC.md` | `gen5_jg5_waterfill/README.md` |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/REPORT_PUBLIC.txt` | `gen5_jg5_waterfill/report.txt` |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT.md` | not staged — it is the PR body itself |

Use `tools/stage_contest_submission_packet.py`, then re-run `tools/packet_census_guard.py`.
Per pq3's finding #4 the ordering law applies: **purge, then census, then buy the receipt,
with no writes in between.**

**Mirror invariant, checked and currently TRUE.** The `” ```text ”` block in `PR_BODY_DRAFT.md`
is byte-identical to `REPORT_PUBLIC.txt`. I edited both in lockstep and verified equality
programmatically. If either is edited again, re-check it — a drifted mirror is the kind of
small incoherence that makes a reviewer stop trusting the numbers.

## 3. What changed in each document

**`BORROWED_SUBSTRATE_ACCOUNTING.md` — APPEND-ONLY respected.** Nothing in §§1–9.4 was
altered. A new **§9.5** appends eleven mechanisms of ours the section-scoped ledger had no row
for: nine `ours-original` build-chain instruments and solves (jg2 tail re-encoder, edit-cost
superposition law, ck2 plane2 container transform −657 B, to1 tail override −105 B, ma1
within-miss corrector, up2 uncapped GN, up3 un-interleave + Rice splice, br1 damped GN, jg4
checkpoint fix), the custody apparatus, and the level-set witness line marked research-only.
It also lands **two corrections against us**: the 12-dim basis re-orientation is a MEASURED
NULL that ships nothing, and the three-way `{edit, drop, keep}` solve shipped only two
branches. **No classification above §9.5 moves, and §9.5 adds zero counted archive bytes.**

**`PR_BODY_DRAFT.md` — three edits.** (a) The compression-script answer now says the entry
point *cannot* rebuild these bytes, not merely that it was not re-run. (b) A new
"What else in this work is ours" section, placed before Credits so the borrow disclosure still
reads first. (c) The budget section gains the cause — token decode is 95.72% of inflation —
plus the two-band disagreement.

**`README_PUBLIC.md` — two edits.** The runtime-risk bullet gains the same cause and the
native-port disclosure; the Reproduction section gains the cannot-rebuild statement and the
version-control disclosure below.

**`REPORT_PUBLIC.txt` — one edit**, mirrored into the PR body block.

## 4. Owed to MAIN — decisions I did not take

1. **The residual-band disagreement is unreconciled and now disclosed.** pq3 and the report use
   `[890.6, 1430.6] s` and grade WARN; wc2 uses `[822, 1302] s` and grades the same body
   REFUSE. Both are ours. I disclosed both rather than quoting the kinder one, which is the
   honest interim state but not a resolution. **One of these bands is wrong and MAIN should
   decide which before the packet is published.**
2. **24 of the 34 files in the jg5 candidate tree have no source in version control** —
   including `runtime/f26_inflate.py`, `runtime/residual_archive.py`, `runtime/free_corrector.py`,
   all of `cpr1/`, and `inflate.py` / `inflate.sh`. Measured by wc2c's census, not by me. I
   added a conservative disclosure to the README's Reproduction section. **Whether that is the
   right level of disclosure, and whether the receiver should be committed before publication,
   is MAIN's call, not an arm's.**
3. **The harvested-receipt `bytes`-repr defect** pq3 routed to MAIN is still open; I did not
   touch it.
4. **`tools/stage_contest_submission_packet.py` has no tests.** Named by pq3, still owed.
5. **The rr2 FreeCorrector native port (L6)** is the submission critical path and unbuilt.
6. **The true cause of the rr2 T4 refusal remains OPEN** — the chartered CPU-vs-CUDA
   explanation was falsified and the differential test refused to convict.

## 5. Five charter premises this arm falsified

Recorded because four of them would have produced an over-claim if I had trusted the charter
over the corpus.

| Charter said | Measured |
|---|---|
| The repo bundle "PREDATES the sub-0.15 row … parts are stale" | False — pq3 rebased all four documents onto jg5 on 2026-08-20 |
| The compression script's rc64 pin is stale; cure is `05839d14` | False, **and the cure would have been a defect** — `05839d14` is the shipped decoder-only body; the pin correctly names the encoder role `5c75e2c7`. Pinning the decoder would break the encode stage |
| "wc2 split-native port with **NEON**" | NEON is not receipted anywhere in scope. The forced-scalar twin parity control is |
| "joint `{edit, drop, keep}` n600 solve" as shipped | `drop` is NOT shipped — it needs a receiver change this body has no path for |
| "container transforms plane2" grouped with the dropped split | Two different mechanisms — ck2's plane2 IS shipped (−657 B); sz1's semantic split is DROPPED |

**Nothing here is a submission action.** No push, no hosting, no PR, no publication. Freezing
and publishing remain the operator's one-line confirm per `FREEZE_CHECKLIST.md`.
