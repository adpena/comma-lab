# Pin census (#1237 residue) + the reaper-argv finding + SSD orphan re-classification (MAIN, 2026-08-29)

**Axis:** `[local apparatus/custody measurement]` · no score claim · `score_claim=false` · `promotable=false`.
Nothing here touches archive bytes or the frontier. Receipts retained at
`/Volumes/APDataStore/pact/ddm_pin_census_20260829/` (shas in §5).

## 1. THE PIN CENSUS — #1237's residue numbers do not reproduce; the mechanism does

Task #1237 recorded: *"SSD census 11/23 in-scope runtimes MISMATCHED (jf2 + po1 clusters; 4 po1
confirmed beyond dg2/jf2) — each must be re-pinned+rechecked or explicitly retired BEFORE
consumption; live hypothesis = bespoke materializers outside the canonical runtime assembler."*

I re-derived it at source over the **full** SSD population using the landed canonical predicate
`tac.candidate_seal.check_pin_consistency` (the same one hd1's preflight calls), not a re-implementation.

**Population: 322 runtime dirs** (`archive.zip` + `inflate.py` co-located) across both SSD tiers —
188 at depth ≤4, 134 at depth 5–7. The residue's denominator of 23 was a narrower in-scope set.

| verdict | count | share |
|---|---:|---:|
| CONSISTENT | 214 | 66.5% |
| PIN_ABSENT (unpinned tree; check is vacuous, reported not passed) | 88 | 27.3% |
| **MISMATCH** | **20** | **6.2%** |

**The 20 mismatches are 2 arm families + 1 intentional control, nothing else:**
- 14 × `ddm_jf2_terminal_diagonal_harvest/retained/k*/retained/{candidate,model_only}_runtime`
- 4 × `ddm_po1_20260813/…/candidate_runtime`
- 1 × `ddm_ps1u_uncapped_pose_20260816/retained/candidate_generation` (182,759 measured vs 183,347 pinned)
- 1 × `ddm_hd1_apparatus_two_landings/controls/jf2_k002500_broken_original_copy` — **hd1's deliberately
  broken positive control**. Mismatched *by design*; it is the executed proof the gate fires. Counting
  it as a defect would be counting a passing control as a failure.

So "11 must be re-pinned" is not the live state. **The honest count of genuine, unintentional
mismatches is 19**, all inside two arms' retained-evidence trees.

## 2. THE MECHANISM — measured at full sha width, two distinct signatures

Split of the 18 deep mismatches by full 64-hex comparison (not prefix):

- **7 rows: sha IDENTICAL, bytes DIFFER.** For one file that is logically impossible as content
  drift — if the sha matches, the bytes match. It is a **pure bookkeeping defect**: `ARCHIVE_SHA256`
  was correctly re-pinned per variant while `ARCHIVE_BYTES` was left at the template value.
  This is literally the "HALF-UPDATED PIN" the task is named for, now with a field-level mechanism.
  (All 7 are jf2 `candidate_runtime`; all carry pinned bytes `180,368`.)
- **11 rows: sha differs.** Of these, 7 jf2 `model_only_runtime` all pin the *same* template
  (`976f706d5af6…` / 180,368) and 4 po1 all pin the *same* template (`6eb1a3b79cb1…` / 186,252) —
  i.e. **one template pin copied verbatim into every variant**, never re-pinned at all.

**Named culprit.** `experiments/ddm_po1_t4_error_feedback_pose_compensation.py:527-530` rewrites the
pin by **string substitution against a hardcoded expected literal**:

```python
old_sha_line   = f'ARCHIVE_SHA256 = "{CP135_SHA256}"'
old_bytes_line = f"ARCHIVE_BYTES = {CP135_BYTES:_}"
```

If the staged receiver's line differs from that expected literal by anything — including the `:_`
underscore-separator formatting — the replace **silently no-ops** and the template pin survives.
The canonical writer (`tac.candidate_seal.repin_receiver`, AST-anchored via `read_receiver_pin`)
fails closed instead. This is the #1093 genus exactly: *correct by expected NAME/LITERAL rather than
by measured VALUE*, one level down.

**Adoption decay (the #936 genus).** Only 5 non-test files call `repin_receiver`/`write_receiver_pin`
— and 3 of the 5 are arms written today (fcd1, fcd1_incompile_schur, bhw1). The canonical writer
exists and is barely adopted; the bespoke substitution path is what actually ran.

## 3. DISPOSITION — do NOT re-pin these; the cure is already landed

The residue said *"re-pinned+rechecked or explicitly retired BEFORE consumption."* Re-pinning is the
**wrong** action here and I am not taking it:

- All 19 live in `retained/` evidence trees. Under the artifact-lifecycle rules HISTORICAL_PROVENANCE
  artifacts are append-only; editing a retained receiver would mutate forensic custody and change the
  runtime tree digest that other receipts cite.
- The consumption hazard the residue was protecting against is **already structurally closed** by
  hd1's landed preflight (f3d6aba3e1): `fire_local_advisory` refuses rc=2 and `make_candidate_seal`
  refuses rc=3 on a mismatched pin, with the real jf2 broken runtime as the executed positive control.
  Every one of the 19 is refused at consumption today.

**So the residue is dischargeable as: census DONE (322 scanned, denominator published), mechanism
NAMED at file:line, population BOUNDED (19 + 1 control, 2 arms, all retained evidence), consumption
GATED by an already-landed fail-closed preflight.** What remains is forward-looking only: the
materializer substitution pattern should be migrated to the canonical AST writer so new runtimes are
not born half-pinned. That is a two-landing leg, recorded below, not done here.

## 4. THE REAPER-ARGV FINDING (new instance of #1189)

`tools/launch_detached_process.py` REFUSED my scan launch with
`matched_tokens: ["claude"]`, `matching_argv_parts: [<the session scratchpad path>]`.

**The session scratchpad directory the harness mandates for temp files
(`/private/tmp/claude-501/…`) contains the literal token `claude`, which matches the fleet reaper's
own kill predicate.** Any long-running detached job that names a scratchpad path in its argv is
therefore SIGTERMed by our own reaper.

This is the measured explanation for the earlier `rc=144` death of the background manifest scan
(task `bat7sc2ru`, zero artifacts). It is #1189's genus ("my own shim PATH is what made the fleet
reaper kill three jo1 r9 daemons") at a **new and more general site**: not a shim we installed, but
the harness-provided scratchpad itself.

Cure applied here: route detached-job output to a repo-local path (`.omx/tmp/…`). The scan then ran
clean — rc=0, 330 s, counter 699. **Standing consequence: never pass a scratchpad path in the argv of
a detached job.** The launch guard already catches it, which is how this was found.

## 5. SSD ORPHAN CLASS — re-classified, not what the headline said

The SessionStart hook flags **96 authored blobs living on the SSD alone** (fresh manifest: **102**
rows — 83 `.py`, 15 `.sh`, 3 `.md`, 1 `.c`; scan rc=0, 140,108 files, `absent_distinct_blobs` 1,348).

I tested the class rather than routing it blind. Measured:
- Blob absence is real — `git hash-object` on two top rows returns blobs absent from the object store,
  so they were never committed at any point.
- **But all six top producers HAVE tracked counterparts** (`experiments/ddm_ad2_*.py`,
  `ddm_tb2_*`, `ddm_bl1_*`, `ddm_lq1_*`, `ddm_xs1_*`, `ddm_mst1_*` are all `git ls-files`-present).

So the class is **not** "producer unrecovered". It is *"the exact code that produced a retained
receipt is an uncommitted variant of a tracked producer."* Divergence is heterogeneous and must not be
treated as one number: ad2 diverges 311/1,269 lines (24.5%), tb2 only 25/1,049 (2.4%). The signal loss
is real for the high-divergence rows (the receipt is not reproducible from git) and near-nil for the
low ones. Routing all 102 identically would be the #821 fan-out error.

Receipts (`/Volumes/APDataStore/pact/ddm_pin_census_20260829/`):
`PIN_CENSUS_SHALLOW.json` sha 596d0bdc634af95a (124,919 B) ·
`PIN_CENSUS_DEEP.json` sha 7e796e3a7ff39f9a (111,714 B) ·
`SSD_AUTHORED_MANIFEST.json` sha d2f1f63018c6d157 (41,636 B) ·
`SSD_AUTHORED_SUMMARY.json` sha 890bd6ccc0591b01 (2,151 B).

## 6. NEXT_IF_RESUMED

- **CLOSED (adjudicated)** — #1237's census residue. 322 scanned · 19 genuine mismatches + 1
  intentional control · mechanism named at `ddm_po1_t4_error_feedback_pose_compensation.py:527-530` ·
  consumption already refused by the landed preflight · retained trees deliberately NOT mutated.
- **QUEUED-W-FIRE-ORDER** — owner MAIN; trigger: the next landing that touches a runtime materializer.
  Migrate literal-substitution pin writers to `tac.candidate_seal.repin_receiver` (fails closed) and
  add the two-landing guard over `ARCHIVE_BYTES = ` string-replace outside the canonical writer.
  Consumer: this memo §2. Cost to falsify: one grep + one test.
- **QUEUED-W-FIRE-ORDER** — owner MAIN; trigger: next consolidation boundary. Disposition the 102
  SSD-only authored blobs by DIVERGENCE BAND (commit the high-divergence variants that receipts
  depend on; certify-in-place the near-identical ones). Consumer: `SSD_AUTHORED_MANIFEST.json`.
  Explicitly NOT a uniform sweep — see §5.

## LIVE-HYPOTHESES

- The 88 PIN_ABSENT trees are a larger latent class than the 19 mismatches and the check is *vacuous*
  for them by construction. Whether unpinned retained runtimes should be refused or merely reported at
  consumption is an undecided policy question, not a measured defect.
- The canonical-writer adoption count (5 non-test callers, 3 of them written today) suggests the pin
  writer is on the same adoption-decay curve #936 measured for `tac.verdicts.emit`.

## DEAD-ENDS

- Re-pinning the 19 retained receivers: refused — mutates append-only forensic custody, and the
  consumption hazard is already gated.
- Reading the mismatch table from 12-char sha prefixes: would have missed that 7 of 18 are
  sha-IDENTICAL/bytes-differ, which is the entire mechanism.
- Hand-rolled `nohup … & disown` for the scan: refused by the launch guard, correctly (§4).

Own-vehicle frontier UNMOVED this turn: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**
(gb1, archive sha `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`).
