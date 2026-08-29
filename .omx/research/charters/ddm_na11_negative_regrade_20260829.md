# ddm_na11_negative_regrade — re-grade the whole closed-verdict corpus against the laws measured SINCE each closure (task #1329, owning memo `ddm_gestalt_delta_interior_seam_20260829.md`)

## MANDATE

Operator 2026-08-29: *"Audit all for other prematurely killed or useful negative or mixed signal"*

Three closures landed in the last hour (`ddm_bz2` capacity ceiling, `ddm_oc2` model-axis drain,
`ddm_qbz1` honest-blocked fit) and between them they put **nine** re-opening laws on the table
that most of the closed corpus was written before. Two of those laws are wrong-object findings —
`#1222` (PoseNet scores the FRAMES, so the RENDERER carries pose, not the pose carrier; refuted
by 1,356× in the wrong direction) and `#1253` (seg on BOUNDARIES, pose in INTERIORS, near-disjoint
supports) — which means a negative can be wrong not because its number was wrong but because it
measured the wrong object. That class does not surface by re-reading verdict lines; it surfaces
by joining each verdict to the law that now contradicts its premise. CLAUDE.md's own rule is that
KILL is a last resort and that a prototype-grade falsification falsifies the IMPLEMENTATION, never
the PARADIGM (Catalog #307) — and `ny1` already proved the corpus contains wrongly-closed rows,
with the first verdict-time toy it found being MAIN's own.

## SCOPE

1. **Enumerate the closed corpus from the STORES, not from memory** (m44 binds: never recall from
   working memory alone). Sweep `.omx/research/**` verdict memos, `.omx/state/probe_outcomes_ledger*`,
   the task ledger's closed rows, and the graph memory (`tools/graph_memory_recall.py`) for
   KILLED / FALSIFIED / REFUTED / DEFERRED / NO-GO / DOMINATED / CLOSED / DEAD / TERMINAL /
   RATE-DEAD / PARKED. Report the denominator you swept and what you could not reach (the vacuity==PASS law, canonical body in `docs/meta_bug_class_catalog.md`: a sweep that does not publish its denominator is not a sweep).
2. **Apply the fan-out law BEFORE counting** (count DISTINCT facts, never occurrences; memory `m84`/task #821, owning memo `ddm_na7_negative_signal_audit_20260814.md`). The verdict-scope detector's false-positive census (task #1191, owning memo `ddm_ny1_live_lineage_toy_and_reactivation_audit_20260823.jsonl`) measured 194 of 198 "violations" to be
   ONE template header counted 194 times. Collapse to DISTINCT facts and say what the collapse
   ratio was; a headline population that does not survive de-duplication is not a finding.
3. **Join each surviving row to the nine laws below.** For each, decide which (if any) applies,
   and whether it touches the verdict's PREMISE (wrong-object → re-open) or only its MAGNITUDE
   (re-scope, verdict may stand).
4. **Honest non-reactivations are REQUIRED.** A sweep that re-opens everything is exactly as
   useless as one that re-opens nothing. State plainly how many rows STAND and why.
5. **Rank the re-opens by cost-to-falsify**, cheapest first, each with a NAMED consumer and the
   single measurement that resolves it. Rows whose resolving measurement needs a scorer run get a
   typed fire order for MAIN, not a launch.

### THE NINE LAWS (each with its receipt — re-derive at source before consuming, per m143)

| # | law | receipt | what it re-opens |
|---|---|---|---|
| L1 | **Holdout signature, n=2** — born representations generalize WITHIN-frame (spatial gap 0.04–0.9%) and NOT ACROSS PAIRS (pair gap 44–64%) | `ddm_bz2_capacity_ceiling_free_closure_20260829.md` + `ddm_qbz1_descent_rate_configuration_20260829.md` (`4ee9199529`) | any negative that assumed cross-pair generalization, or that only tested within-frame |
| L2 | **Wrong-object pose** — PoseNet scores the FRAMES; the RENDERER carries pose, not the carrier (1,356×) | task #1222 | every pose negative that credited/debited the carrier |
| L3 | **τ-inversion** — seg on boundaries, pose in interiors, near-disjoint supports | task #1253 (tv1/tv2) | negatives assuming shared support between the two terms |
| L4 | **The demand reads two ways** — 42,382 B at fixed distortion vs **150 B** at ZERO distortion; seg alone 30,248 B | memory `m124`, task #1203 | anything priced against exactly one reading |
| L5 | **Sharp-optimum has a measured WIN-WIN exception** — fcd1's B/H/W split found 5,268 GT-benefit edits that SHRANK the archive −3,756 B | memory `sharp-optimum-law-has-a-win-win-exception-class`, task #1319 | every family closed on "the optimum is sharp in every direction" (#1214, five arms) owes a B/H/W re-screen |
| L6 | **In-compile Schur compensation is PROVEN** (d_pose landed BELOW base, repeat identical) | `ddm_qs5_verdict_and_no_toy_enforcement_20260813` | every family closed by UNCOMPENSATED pose damage carries a named reactivation |
| L7 | **Pose is an ABSOLUTE budget ≤1.25e-4**, not a ratio (∂S/∂d_pose 626.5) | memory `m110` | ratio-priced pose negatives are mis-scoped |
| L8 | **The model axis is CLOSED on this body** (oc2: 3 orthogonal charts, best 2 B vs a 30 B bar; paid rescue 47.4× under) | `ddm_oc2_orthogonal_conditioning_charts_20260829.md` (`ed5f20b6de`) | negatives that DEFERRED to future conditioning headroom have lost the escape hatch — re-grade terminal, not deferred |
| L9 | **Catalog #307** — prototype-grade falsification kills the IMPLEMENTATION, never the PARADIGM | CLAUDE.md; `ny1` (`301f2b4770`) found wrongly-closed rows incl. MAIN's own | any verdict drawn from a toy/naive/under-scoped run |

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_na11_negative_regrade/`.
  AP is the tier (16 GiB free); Vertigo is at 8.3 GiB — check before materializing.
- **This is an AUDIT, not a build.** Do not re-run the measurements you are grading; grade them.
  If a row's resolution needs a new measurement, NAME it and its cost — do not perform it.
- Axis honesty: any local number is `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`,
  `promotable=false`. Only `upstream/evaluate.py` on exact bytes is authority. Recompute S FROM
  COMPONENTS (#877), never a rounded display field.
- **File ownership:** the fleet is idle at spawn, but re-check `status` at start; if a sister arm
  has claimed a memo you intend to amend, append a disposition rather than editing its body
  (Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE, canonical body in `docs/meta_bug_class_catalog.md` — historical verdicts are never mutated, they are superseded).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ny1_live_lineage_toy_and_reactivation_audit_20260823.jsonl` (`301f2b4770`), task #1226 —
  verdict `WRONGLY_CLOSED_ROWS_FOUND`. **This charter must not re-find ny1's rows and count them
  as new.** De-duplicate against it explicitly and report the overlap.
- `ddm_na10_negative_audit_fresh_laws_20260819.md` (`3225e3a880`, task #1140) — the same
  re-grade-against-fresh-laws move, run 10 days ago under a DIFFERENT law set (GT-lineage fork,
  composition law, realized-acceptance, 95.9% render-loss, pose variance 13.4×). Rows na10 already
  re-graded under those laws are NOT re-openable by those same laws; only L1–L9 above are new.
- `ddm_na9_gestalt_negative_audit_20260818.md` (`4b63faf200`) and
  `ddm_na7_negative_signal_audit_20260814.md` (`cba8813caf`) — na7 landed a CLEAN round (2 routing
  errors, no reopenings). A clean prior round is evidence the corpus is not trivially full of
  errors; a large reopening count here needs a mechanism, not just a list.
- The verdict-scope detector's false-positive class (task #1191, owning memo `ddm_ny1_live_lineage_toy_and_reactivation_audit_20260823.jsonl`): 194 of 198 were one template header. Any count this arm publishes must survive that collapse.
- The vacuity==PASS law (task #1097; canonical body in `docs/meta_bug_class_catalog.md`) — report the DENOMINATOR.

## OPTIMAL FORM

- Family exemplar: `ddm_na10_negative_audit_fresh_laws_20260819.md` (commit `3225e3a880`) is the
  **reference** form for this family — store-seeded enumeration, per-row law join, typed re-grade
  with named resolving measurement, honest non-reactivation count. Run that landed shape; do not
  invent a new audit format. `ny1`'s JSONL (`301f2b4770`) is the reference for the machine-readable
  half — emit rows in a schema a later arm can join on, not prose only.
- SCOPE reductions declared per row (if you sweep a subset of the corpus, state its size and
  selection rule). MECHANISM reductions FORBIDDEN: real store reads, real law joins, real
  de-duplication — an audit that grades from a summary of the corpus rather than the corpus is
  the exact defect it exists to find.
- **PRIOR-LAW PREDICTION (falsifiable):** `ny1` found wrongly-closed rows under ONE law
  (Catalog #307 toy-grading). L1–L9 add eight more independent joins, two of which (L2, L3) are
  WRONG-OBJECT findings that invalidate premises rather than magnitudes — the strongest re-opening
  class there is. So the prediction is **≥3 DISTINCT rows re-open under L1–L8** (i.e. under laws
  other than the toy-grading L9 that ny1 already swept), with **≥1 under a wrong-object law**.
  FALSIFIER: every row either STANDS or re-opens only under L9 — meaning the corpus was already
  correctly graded against the current physics and the last ten days of laws changed no prior
  verdict. **That is a genuinely useful negative: it would say the closure discipline is sound and
  that the campaign's problem is the absence of live routes, not the mis-grading of dead ones.**
  Count it plainly if it lands; do not manufacture reopenings to beat the prediction.

## DELIVERABLE

`.omx/research/ddm_na11_negative_regrade_20260829.md` + a machine-readable sibling
`.jsonl` — typed rows: `{verdict_id · owning memo + sha · original scope on the verdict ladder ·
law(s) joined (L1–L9) · premise-vs-magnitude · re-grade ∈ {STANDS, RE-OPEN, RE-SCOPE,
WRONG-OBJECT} · cheapest resolving measurement + its cost · named consumer}`; plus the swept
DENOMINATOR, the fan-out collapse ratio, the ny1/na10 overlap count, the STANDS count, and the
prediction's outcome stated plainly. Loop-until-dry: seal on a round that produces zero new
DISTINCT rows, and say how many rounds it took. Commit via the serializer. End with the
own-vehicle frontier line.
