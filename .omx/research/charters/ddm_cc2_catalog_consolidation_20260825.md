# ddm_cc2_catalog_consolidation — Catalog consolidation REVIEW (harness task #1272, this charter file is its owning memo): the #299 quota-waiver debt census + ranked retire/consolidate recommendation table

## MANDATE

Operator 20260825: *"Codex and opus available now"* (arm-capacity grant) consuming the routed finding
in task #1272 "Catalog consolidation review (quota #299 waiver debt)", filed during the #842
preflight-enumeration window. CLAUDE.md's "Gate consolidation discipline" (Catalog #299) binds: no new
STRICT gate past **#400** without retirement, replacement, or an explicit file-level
`# CATALOG_QUOTA_EXCEEDED_OK:` waiver — and gate numbers far past 400 exist in live use (e.g. #812
dynamic-rglob, #842-window rows). Nobody has ever run the "stop and consolidate" pause the discipline
promises. This arm runs the CENSUS and produces the ranked RECOMMENDATION table — it does NOT mutate
the catalog, preflight.py, or CLAUDE.md (MAIN + operator adjudicate; recommendation-only by design).

## SCOPE

1. CENSUS: enumerate the claimed catalog numbers — authority = `docs/meta_bug_class_catalog.md` (the
   pointer target CLAUDE.md names) + `tools/claim_catalog_number.py` state + grep of
   `src/tac/preflight.py` for `Catalog #<N>` markers. Report: max claimed #, total rows, rows > #400,
   presence/absence of `CATALOG_QUOTA_EXCEEDED_OK` waivers in the first 200 lines of CLAUDE.md (the
   #299 contract's named location).
2. ORPHAN/DUP CROSS-CHECK (static, cheap): `preflight_all()` call sites vs catalog rows per the #176
   discipline — call sites with no row, rows with no call site, duplicate numbers. Report counts with
   file:line evidence; do not fix.
3. CONSOLIDATION FAMILIES: cluster the past-#400 gates (plus any obviously sister-redundant earlier
   gates surfaced by the cross-check) into umbrella candidates per #299's bar — an umbrella must
   subsume ≥3 sister cases or REPLACE an existing gate. Each cluster row: member gates, shared
   bug-class signature, proposed umbrella shape, risk of coverage loss.
4. RETIREMENT CANDIDATES: gates that have fired 0 findings since landing AND whose bug class is
   structurally extinct at the source (cite the structural cure commit/receipt) — per the
   Mission-alignment "annual gate audit by empirical score contribution" clause. Where fire-history is
   not cheaply obtainable, say NOT-MEASURED for that row rather than guessing (#821 counting trap).
5. RECOMMENDATION TABLE: ranked {RETIRE / CONSOLIDATE-INTO-<umbrella> / KEEP / QUOTA-WAIVER-NEEDED}
   per affected gate, with the exact #299-compliant landing shape for each recommendation. NO code or
   catalog mutation in this arm.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_cc2_catalog_consolidation/`.
- RECOMMENDATION-ONLY: do NOT edit `src/tac/preflight.py`, `CLAUDE.md`, or
  `docs/meta_bug_class_catalog.md`. Two parallel Opus arms are editing `.omx/research/*.md` memos
  (Catalog #287/#300/#305 backfills) — treat all `.omx/research/*.md` as READ-ONLY except your own
  deliverable memo.
- POPULATION-STRUCTURE FIRST (#821): before ranking, measure whether violations/overage rows are few
  facts fanned out (templates, sister-copies) — report the decomposition.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Task #821 (`.omx/research` corrections index lineage): "the 184 violations are ONE FACT counted 184
  times" — naive per-row counting overstates populations 205×; measure fan-out first.
- Task #1085 (memo `ddm_*_census` lineage, commit 03b02ddc99): a census INVERTED the filing finding —
  one advisory leg produced 60% of rows; expect the same concentration here.
- Task #1149 (canonical_equations_registry roundtrip red): registry-file edits by sweep are fragile —
  another reason this arm is recommendation-only.
- Catalog #299's own text: "pure-additive gate landings are the slow death" — the failure mode this
  review exists to reverse; a recommendation table that only says KEEP-everything is the null result
  and must be justified row-by-row if reached.

## OPTIMAL FORM

- Family exemplar: the ca1 censored-caps census (commit 57c87898c2) is the reference form — an
  executed repo-wide census with per-site evidence, a typed disposition table, and zero speculative
  fixes; receipt path `.omx/research/` ca1 lineage. This charter reduces SCOPE only (catalog rows
  instead of cap sites); the census+disposition MECHANISM is unchanged.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN.
- **PRIOR-LAW PREDICTION (falsifiable):** per #821/#1085 (fan-out concentration law), the past-#400
  overage decomposes into a SMALL number (≤6) of consolidation families each subsuming ≥3 sisters,
  making #299-compliant consolidation feasible. FALSIFIER: the census shows past-#400 gates are
  predominantly singleton bug-classes with no ≥3-sister cluster — then consolidation is refuted at
  this population and the honest recommendation is an operator quota decision (raise #400 or
  per-gate waivers); count it plainly if it lands.

## DELIVERABLE

`.omx/research/ddm_cc2_catalog_consolidation_20260825.md` — typed rows: §1 census numbers, §2
orphan/dup table (file:line), §3 consolidation-family clusters, §4 retirement candidates
(fired-count or NOT-MEASURED), §5 ranked recommendation table with per-row #299-compliant landing
shape, §6 prior-law-prediction verdict. Commit via the serializer. End with the own-vehicle frontier
line.
