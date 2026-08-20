# ddm_vh2 — 3,001 artifacts, 15 harvested rows. Build the ROUTING LEDGER, then drain partition 1.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free · `[macOS-CPU advisory]` ·
`score_claim=false`

## OPTIMAL FORM (read first)

Reference form: a DURABLE, QUERYABLE routing ledger over the full vehicle corpus in which every
harvested row exits **OWNED** — {owner · consumer · fire-order} or {deferred · named blocker ·
fire-condition} — plus a real drain of the first partition. Declared reductions: SCOPE only — you
drain ONE partition, not all. MECHANISM reductions are TOY-BRACKET: another ranked markdown table
with no consumer (that is the disease, not the cure); a "harvest" that samples without stating its
denominator; rows that exit UNOWNED.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. Cite path + commit.

## WHY — measured, not asserted

Operator 2026-08-09: *"There's more stuff from more vehicles besides those that you have already
harvested. Also, all of that needs to be routed and followed up on."*

MEASURED this session (reproduce it; do not trust this line):
- `.omx/research/*.md` = **7,092** · `ddm_*` artifacts = **3,001**
- `ddm_wl1` harvested v1–v10 → **15 rows**. `ddm_vp1` re-scored those same 15.
- Lineage doc-counts with NO dedicated harvest: `ddm_v*` 111 (v11–v19) · `ddm_j*` 35 ·
  `ddm_ms*` 28 · `ddm_m*` 24 · `ddm_ws*` 18 · `ddm_rg*` 18 · `ddm_e*` 16 · `ddm_ic*` 13 ·
  `ddm_is*` 13 · `ddm_ar*` 13 · `ddm_pb*` 11 · `ddm_dm*` 11 · `ddm_menu*` 10 · `ddm_od*` 9 …

So the campaign's transfer inventory covers a SAMPLE and has been cited as if it covered the
corpus. That is the scope error this arm exists to fix — and the fix is NOT "harvest harder,"
it is **an instrument that makes un-routed signal visible and drainable without depending on
anyone's memory.**

## WHAT TO BUILD (in this order)

1. **PARTITION the corpus by lineage**, from the filesystem, with counts. Emit the partition table
   with its DENOMINATOR stated. A partition scheme nobody can reproduce is not an instrument.
2. **The ROUTING LEDGER — a typed, append-only JSONL**, one row per harvested finding:
   `{lineage, source_doc, sha, finding, evidence_class, vehicle_scope, status, owner, consumer,
   fire_order | blocker, fire_condition}`. `status ∈ {ROUTED-FIRED · ROUTED-QUEUED ·
   DEFERRED-WITH-BLOCKER · DEAD-ON-THIS-BASE · NEEDS-REMEASURE}`. **UNOWNED is not a legal status.**
   Reuse the existing ledger machinery — `probe_outcomes_ledger` has 23 callers and 662 live rows
   and WON the last consolidation (#936); do NOT build a parallel store, and do NOT route toward
   `tac.verdicts` (0 production callers). Cite which you extend and why.
3. **A COVERAGE QUERY.** `coverage()` returns, per lineage: artifacts · harvested · routed ·
   un-harvested. This is the standing answer to "what is still un-routed?" — the thing that has
   been living in my head and therefore getting lost. Wire ONE real reader (costate digest or the
   MAIN hot-state surface); a ledger with no reader is the #936 write-only-API failure repeated.
4. **DRAIN PARTITION 1 for real.** Pick the highest-value un-harvested lineage and justify the pick
   from measured campaign state (rate is 73.9% of S; seg 16.6% with the decomposition blocked;
   pose closed by `pk2`). Every row lands in the ledger OWNED. If a row's honest status is
   DEAD-ON-THIS-BASE, name the closure that kills it (`pk2` pose · #996 coder — **note #996's own
   scope is under review by `ddm_rc2`, so cite it as "coder axis, scope-under-review"** ·
   `113b52fdb1` gauge · #917 retired-vehicle instruments).

## HARD RULES

- Bulk → `/Volumes/APDataStore/pact/ddm_vh2_20260810/` (tier-2, ~997 GiB; tier-1 VertigoDataTier
  is 98% full). No `/tmp` in evidence. Operator 2026-08-09: *"keep all archives and outputs and
  everything"* — persist every intermediate, sha256'd. Run `tac.payload_retention_gate`.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`.
- Write ONLY under `ddm_vh2` paths + the ledger you extend. Sister arms' artifacts are APPEND-ONLY.
- `upstream/` IMMUTABLE. No scorer, no Modal. Every number carries its axis.
- **DECODE TIME IS NOT A DISQUALIFIER** (operator binding, `tac.subagent_contract`, `e2700086f2`).
- Do NOT re-harvest what `wl1`/`fh1`/`vp1` already covered — LOAD their rows into the ledger as the
  first entries, with their existing routing preserved. Rediscovery is the cardinal sin.

## DELIVERABLE

The partition table with denominators · the routing ledger with `wl1`+`fh1`+`vp1` loaded and
partition 1 drained, every row OWNED · the `coverage()` query with one live reader · the honest
count of what remains un-harvested per lineage. If the corpus turns out to contain far less
*routable* signal than 3,001 artifacts suggests — much of it superseded, duplicated, or
vehicle-dead — that is a fine and valuable answer, but it must come with the measured
classification, not an impression.
