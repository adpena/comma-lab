# ddm_hv4_recovery_consumption_sweep — the RECOVERY half: drain the 357-row plan surface + the local-disk-only memo orphans to owned exits (task #1323; owning memos `ddm_hv2_harvest_consumption_sweep_20260827.md` (the prior sweep, which produced the fcd1 chain) + task #1190's orphan-signal finding)

## MANDATE

Operator 2026-08-29 (standing GO, verbatim): *"Recover and respawn and continue with all."*
The RECOVER half has a measured surface. `tools/codex_arm_queue.py next` reports **357 live
NEXT_IF_RESUMED plan rows** across the arm corpus, and task #1190 recorded **42 research memos
that existed only on local disk — invisible to git, the graph, and the corpus** (so invisible to
every recall step, which is the m44 never-recall-from-working-memory failure made structural).
The precedent is exact and it PAID: `ddm_hv2_harvest_consumption_sweep_20260827.md` ran this sweep
once and its FIRE-NOW head produced the jf2→fcd1 chain — i.e. today's only live rate opening came
out of a consumption sweep, not out of a new mechanism. This arm runs it again over the
post-fcd2 state, loop-until-dry, and hands MAIN a ranked FIRE-NOW head.

## SCOPE

1. **Orphan recovery (the #1190 class)**: re-measure the population — memos/receipts present on
   local disk or the SSD tiers but absent from git AND absent from the corpus/graph index. For
   each: {path, sha, bytes, birth, is it signal or scratch?} → commit-or-certify per the
   certify-or-block rule (never delete; MOVE to the SSD tier with a machine-readable cert if
   rebuildable). Report the count honestly even if it is now 0 — a drained class is a finding.
2. **Plan-surface drain**: over the 357 live NEXT_IF_RESUMED rows, assign every row ONE typed exit
   — {FIRED (already done, cite the row) · FOLDED (subsumed, name the subsumer) · QUEUED-W-FIRE-ORDER
   (owner + trigger + consumer store) · CLOSED (verdict + scope) · STALE (precondition moved, name it)}.
   Zero rows may exit UNOWNED (the m45 law). Use the landed extractor
   (`tools/codex_arm_queue.py next` / the NEXT_IF_RESUMED surface), not a re-grep.
3. **Cross-check against the task ledger** (the #880 join defect: the follow-on detector's corpus
   is MEMOS while the backlog is TASK ROWS — they were never joined). Report rows that exist on
   one surface and not the other; that gap is where signal hides.
4. **Rank a FIRE-NOW head**: the ≤5 rows with a real measured opening, a live object, and a $0-or-cheap
   first step — ranked by (measured ΔS or B) ÷ (cost to falsify). Each head row gets a named
   consumer and a fire trigger. This is the arm's PRIMARY deliverable.
5. **Do NOT execute the heads** — this is a routing arm. Exception: any head that is literally a
   $0 read of a RETAINED receipt may be executed inline and reported with its receipt sha.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO heavy launch. NO scorer job (fcd3 owns the scorer lane).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD — recovery here means MOVE-and-certify, NEVER delete (P0 DEF CON 1000).
  Receipts to `/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/`.
- Read RECEIPTS, not memo headlines — the #1191/#1224/#953 stale-headline genus is measured and
  live (a headline survives a corrected body; the corrections index at
  `.omx/research/ddm_au1_20260805/` is the instrument, rebuilt 2026-08-29).
- Every claimed count carries its denominator (the #1197/vacuity law: a census with no denominator
  is not a measurement).
- Axis honesty on any number that touches score: `[macOS-CPU frozen-scorer advisory]`.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_hv2_harvest_consumption_sweep_20260827.md`: the prior sweep — do NOT re-drain rows it
  already exited; start from its exit table and sweep the DELTA plus everything landed since.
- Task #1210: a stale capacity note in a routing memo KILLED three arms; routing claims must be
  re-derived at consumption time, never carried (the m143 cross-regime transfer law).
- Task #1191: the verdict-scope detector produced 194 false positives from ONE template header
  counted 194 times — count DISTINCT facts, never occurrences (the #821 fan-out law).
- Task #1226 (`ny1`): a prior wrongly-closed-rows audit found the first verdict-time toy was the
  auditor's OWN — apply the no-toy law to this arm's own screening rows.
- Task #1085: a census leg that produced 60% of advisories and could not be made precise was
  RETIRED fail-closed — if a screening leg here cannot be made precise, retire it, do not tune it.

## OPTIMAL FORM

- Family exemplar: `ddm_hv2_harvest_consumption_sweep_20260827.md` is the reference form (its
  FIRE-NOW rank 1 became ddm_jf2 → the fcd1 chain → today's −3,729 B opening); reference receipts
  in its consumer store. Run that landed form over the post-fcd2 delta.
- SCOPE reductions declared per row (a row whose object is off-disk is screened at REDUCED scope
  and labelled). MECHANISM reductions FORBIDDEN: no sampling the plan surface — all 357 rows get
  an exit, or the uncovered remainder is named with its exact count.
- **PRIOR-LAW PREDICTION (falsifiable):** hv2's precedent (a consumption sweep produced the
  campaign's best live opening) plus the measured 357-row backlog predicts ≥1 FIRE-NOW head with
  a measured opening ≥30 B or ≥1e-5 S that no arm currently owns. FALSIFIER: every one of the 357
  rows exits FIRED/FOLDED/CLOSED/STALE with zero QUEUED-W-FIRE-ORDER heads carrying a measured
  opening — that would mean the plan surface is genuinely drained and the campaign's constraint is
  purely mechanism-invention, not consumption. That is a real and useful finding; count it plainly.

## DELIVERABLE

`.omx/research/ddm_hv4_recovery_consumption_sweep_20260829.md` — typed rows: (1) orphan-recovery
table w/ per-file disposition + certs; (2) the full 357-row exit table (or the named uncovered
remainder w/ count); (3) the memo↔task join gap; (4) the ranked ≤5 FIRE-NOW head w/ owner, trigger,
consumer, and cost-to-falsify; (5) NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Commit via the
serializer. End with the own-vehicle frontier line.
