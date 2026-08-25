# Catalog #287 evidence-tag gate burn-down (task #1271)

**Date:** 2026-08-25 · **Axis:** apparatus only — no score claim, no pointer movement.
**Frontier line (unchanged by this work):** gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600].

## What this is

`check_no_docstring_overstatement_without_evidence_tag` (Catalog #287) was strict-flipped
2026-05-19 at live-count 0. 162 violations then accrued during the #842 window, because the
gate was skipped on normal commits until 2026-08. It was demoted to warn-only on 2026-08-25
per the strict-flip atomicity rule. This memo records the burn-down that lets it re-flip.

## Population decomposition — measured, not assumed

The 162 rows were **not** one fact counted N times, but they were not 162 independent facts
either. Measured structure:

| Sub-scope | Rows | Distinct files | Distinct root causes |
|---|---:|---:|---|
| A — `src/tac/**/*.py` percentage/multiplier claims | 9 | 8 | 9 (each an independent claim) |
| B — `.omx/research/**/*.md` phantom module citations | 153 | 83 | 85 distinct dotted prefixes |

Sub-scope B clustered heavily: the top 3 program-eras produced 64 of 153 rows.

## Finding 1 — 19 of 162 were detector false positives, cured at the detector

Three prose forms match the sub-scope-B regex while making **no module claim at all**. Each
was cured in `preflight.py` with an *evidence-gated* predicate, not with waivers:

```
1. package dunder attribute   tac.__file__ , tac.__version__      (9 rows)
2. family-glob truncation     tac.pareto_* , tac.lane_mark_{a,b}  (9 rows)
3. metasyntactic placeholder  from tac.X import ClassName         (1 row)
```

The regex stops at the trailing `_` of a glob (the glob char is not a word char), yielding a
truncated non-module; and a dunder is never a submodule, so `find_spec` is False for a name
that genuinely resolves. Case 1 matters most: the flagged token is the canonical venv-hijack
check, so every future memo recording that check would have accrued fresh debt forever.

Each cure requires **positive evidence** before it suppresses — the attribute must exist on
the real package, the glob family must have real members, the placeholder must be a bare
capital letter. Genuine phantom citations still fire; 15 tests pin this, including a dunder
that does not resolve on the package, a glob over a family that was never built, and a
trailing underscore with no glob char, all of which must remain flagged.

(This memo tripped its own gate on first write, on a non-resolving dunder spelled out in
prose. That is the cure behaving correctly, and it is left recorded here rather than hidden.)

Curing this at the detector rather than with 19 waivers follows the structural-beats-
procedural discipline: a waiver would have hidden a detector bug and let it re-accrue.

## Finding 2 — sub-scope A: 8 real citations, 1 false label

Every number was checked against the artifact before it was tagged. All four cited artifacts
exist and contain the exact figures (17.96x / 11,149 ms -> 621 ms; 25.35s vs 23.44s; 5.5x;
6.14% / 13.55% / 8.53%).

- 5 rows got a real `[empirical:<path>]` tag naming the artifact that actually holds the number.
- 3 rows were derivations, not measurements, and were tagged `[prediction]` with the derivation
  stated (a backward-call-count collapse, and a complexity ratio).
- 1 row was a **false label**: a substrate archive docstring claimed an "Empirical archive size
  ... ~45% reduction" with no retained artifact behind it. A repo-wide search for a measured
  byte receipt found none. The text was corrected to "Projected", tagged `[prediction]`, and
  annotated with what would be needed to re-label it MEASURED.

## Finding 3 — sub-scope B: waived in the manifest, memo bytes preserved

The remaining rows are citations of modules that were designed and never built, in dated
memos. Per Catalog #110/#113 append-only historical provenance, the memo bytes are **not**
mutated; waiver authority lives in the append-only ledger
`.omx/state/catalog_287_phantom_api_waivers.jsonl`, which is the channel the gate author built
for exactly this case and which already carried 388 rows from the 2026-05-19 wave.

Every row carries `line_sha256`, so a waiver **fails closed** — if the line changes, the
violation re-fires rather than silently waiving different text. This mattered during this
task: a concurrent agent was committing frontmatter into council memos while it ran, shifting
line numbers underneath the scan.

Rationales are per-cluster and name the era and the reason, never a generic stamp:

```
C1  slot GG/HH RL-substrate design proposal (2026-05-29)
C2  g-series taskspace inverse-witness SPEC proposal (2026-07-25/27)
C3  ActionEffect / evaluator-inverse IR design lineage (2026-06/07)
C4  retired-era historical research prose, name never built or long superseded
```

The manifest is exact to `(relpath, line, dotted)`. It cannot waive a file wholesale, so a
future phantom citation added to any of these memos still fires.

## Honest limits

- The manifest waives **citations**, it does not resurrect the designs. Nothing here claims any
  of the named helpers should exist or will be built.
- Line-keyed waivers die on line drift. That is deliberate, but it means a large edit to a
  waived memo will surface its rows again for re-adjudication.
- Sub-scope A's `[prediction]` tags record that no timing artifact exists. They are not a
  promise that one will be produced.
