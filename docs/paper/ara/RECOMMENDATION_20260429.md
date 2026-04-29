# Ara Pivot Recommendation — 2026-04-29

**Author**: parallel research subagent (NOT subagent E, who is updating the
existing prose drafts with last week's findings).
**Reference paper**: arXiv 2604.24658 — "The Last Human-Written Paper:
Agent-Native Research Artifacts".
**Audience**: human operator deciding writeup strategy for the May 3
deadline.

---

## TL;DR

**Recommendation: PARTIAL PIVOT.** Keep the human-readable prose drafts
(`docs/writeup_draft.md`, `docs/paper_outline.md`,
`reports/graphs/site/*`) as the primary artifact for the comma.ai
writeup-prize judges. Add a parallel Ara-paradigm package under
`docs/paper/ara/` as a *second*, machine-queryable artifact that
strengthens the submission in three specific ways:

1. **Reproducibility seal.** Every claim points to evidence and code; an
   external reviewer (or the comma.ai judges' agents) can verify any claim
   without reading prose.
2. **Differentiation.** Most challenge submissions will be a PDF + a repo.
   An Ara-native artifact that explicitly cites and adopts the paradigm
   from arXiv 2604.24658 is unusual and signals methodological seriousness.
3. **Future-proofing.** The lab's exploration graph (78 STRICT preflight
   checks, 30+ negative results across 3 eras) is the most valuable
   long-tail artifact. Encoding it in Ara format preserves it for the
   followup arXiv preprint.

Estimated cost: **3-5 hours** of additional work beyond what subagent E is
doing today, distributed across two compilation passes. The skeleton is
already landed in this commit.

---

## Section 1 — Paper paradigm summary

The Ara paper recasts research artifacts from "narrative document" to
"machine-executable knowledge package". Concretely:

- **Four layers per artifact**:
  - `logic/` — scientific reasoning, falsifiable claims, related-work
    dependencies (typed: import / bound / baseline)
  - `src/` — physical implementation: kernel mode (small typed I/O modules)
    or repository mode (full source tree with index.md)
  - `trace/` — exploration DAG: decisions, dead-ends, pivots, with
    `also_depends_on` cross-references at convergence points
  - `evidence/` — outputs only; no prose. Every claim chain ends here.
- **Live Research Manager** — captures research events (decision, claim,
  experiment, heuristic, dead_end, observation, pivot) during normal
  development workflow; routes them into the right layer.
- **Ara Compiler** — translates legacy PDFs and repos into Ara format via
  four stages: semantic deconstruction, cognitive mapping, physical
  grounding, exploration-graph extraction.
- **Ara Seal** — three-level verification:
  - L1: schema conformance + cross-layer reference resolution (seconds)
  - L2: argumentative rigor along 6 anchored 1-5 dimensions
  - L3: scaled-down execution-reproducibility check, with the verifying
    agent isolated from `evidence/` to prevent fabrication
- **Reported gains**: question-answering accuracy 72.4% -> 93.7%;
  reproduction success 57.4% -> 64.4% on PaperBench (8,921 reproduction
  requirements across 23 papers).

The paper's central thesis: "narrative compilation systematically erases
research process knowledge" (the *Storytelling Tax*) and "reviewer-sufficient
documentation is not agent-sufficient" (the *Engineering Tax*). Ara fixes
both.

## Section 2 — Gap analysis

What we already have that aligns with Ara:

| Ara concept                    | Our existing artifact                                        | Gap                                                                                |
|--------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------|
| `logic/problem.md`             | `docs/paper_outline.md` Section 2 + `PROGRAM.md`             | scattered across files; no single normalized entry                                 |
| `logic/claims.md` + falsifiers | implicit in prose; explicit in council reviews               | not aggregated as a falsifiable list with stable IDs                                |
| `logic/experiments.md`         | `docs/paper_outline.md` Section 6 + `experiment_journal.md`  | tied to claims by prose proximity, not by ID-binding                               |
| `logic/related_work.md`        | `docs/paper_outline.md` Section 8                            | dependencies untyped (no import / bound / baseline distinction)                    |
| `src/index.md`                 | `experiments/pipeline.py` + `src/tac/profiles.py`            | no claim->module mapping                                                           |
| `trace/exploration_tree.yaml`  | `.ralph/run_log.md` + memory/`project_*` files (~408 events) | linear log, not a DAG; no parent/child or convergence edges                        |
| `trace/dead_ends_to_revisit`   | scattered across memory feedback files                       | not consolidated; no revisit protocols                                             |
| `evidence/`                    | `experiments/results/**` + `reports/raw/**`                  | no normalized index; no per-claim provenance                                       |
| Live Research Manager          | manual: `tools/review_tracker.py` + memory file conventions  | not automated; events are flat markdown not typed records                          |
| Ara Compiler                   | none                                                         | `tools/ara_compile.py` scaffold landed in this commit                              |
| Ara Seal                       | `tools/check_*.py`, 78 STRICT preflight checks               | preflight is for code, not for the *artifact*. Seal Level 1 implemented; L2/L3 TBD |

What we have that the paper does NOT cover (our differentiation):

- **78 STRICT static checks** that prevent specific measurement-bug
  classes. The paper's "verifier isolation" mitigates fabrication; our
  preflight catalog *prevents* the bugs upstream. This is publishable as
  an extension of the Ara methodology.
- **Lane-tag discipline** (`[contest-CUDA]` / `[Modal-T4-CUDA]` /
  `[advisory only]` / `[MPS-PROXY]`). The paper assumes a single eval
  surface; we have a four-class taxonomy because we burned weeks on
  MPS-vs-CUDA drift.
- **Strict-scorer rule**. Our compliance posture (no scorer weights at
  inflate time) is enforced by Check 6; the paper does not consider
  competition-style compliance.

## Section 3 — Pivot recommendation

**Choice: PARTIAL PIVOT.** Reasons:

- **Cost of a full pivot is too high for May 3.** A full pivot would mean
  rewriting `docs/writeup_draft.md` (1127 lines) and the Cloudflare site
  in Ara form. That is 12-20 hours and wastes subagent E's parallel work
  on the existing drafts.
- **Cost of "keep + cite paper as method ref" is too low.** The judges
  are humans who want a clear narrative AND researchers who want
  reproducibility. Citing the paper without using it leaves the
  reproducibility prize on the table.
- **Partial pivot is the dominant move.** Keep the prose for human readers,
  add the Ara layers for agent reviewers and reproducibility.
  Each layer is independently useful.

### Cost estimate (post-skeleton)

| Pass | Effort | Output                                                                           |
|------|--------|----------------------------------------------------------------------------------|
| 1    | ~30 min | DONE — skeleton committed (this commit)                                         |
| 2    | ~90 min | Copy/normalize `experiments/results/lane_*_landed/contest_auth_eval.json` into `evidence/results/` with per-claim provenance JSON |
| 3    | ~60 min | Wire `tools/review_tracker.py` post-commit hook to emit a `trace/events.jsonl` line for every commit touching `src/tac/` or `experiments/` |
| 4    | ~30 min | Add an "Ara reviewer triage" 500-token block to the top of `reports/graphs/site/judges_one_pager.md` so a judge agent can navigate the artifact |
| 5    | ~30 min | Run `tools/ara_compile.py --validate-only` in CI; gate any commit that breaks the seal at Level 1 |

Total post-skeleton: **3-4 hours** of additional work, parallelizable
across the next two days.

### What this preserves vs what it adds

- Preserved: every prose draft (`docs/writeup_draft.md`,
  `docs/paper_outline.md`, `reports/writeup_working.md`,
  `reports/graphs/final_writeup_draft.md`,
  `reports/graphs/site/*.md`). Subagent E continues to update them
  in-place. NO overwrite.
- Added: `docs/paper/ara/` (this commit) — a parallel,
  agent-queryable artifact pointing at the same underlying evidence.
- Added: `tools/ara_compile.py` (this commit) — the compiler that
  validates the Ara structure.

### Disclosure-policy compliance

- Public-facing files (`reports/graphs/site/*`) are NOT modified by this
  commit.
- Lane W / Lane Omega / Lane DARTS-S details are NOT mentioned in the
  Ara skeleton (verified by grepping the new files).
- Cloudflare URL is NOT mentioned.
- All Era 3 Selfcomp paradigm scores are tagged `live; no scores reported
  here per CLAUDE.md non-negotiable`.

## Section 4 — Concrete first actions taken (this commit)

Files written (all under `docs/paper/ara/` unless noted):

- `PAPER.md` — root manifest with YAML frontmatter, layer index,
  disclosure policy, lane-tag declarations
- `logic/problem.md` — gap definition, key insights, two-era summary
- `logic/claims.md` — 10 falsifiable claims (C1-C10) with forensic
  bindings to evidence and code, public_safe flags
- `logic/experiments.md` — 10 experiment declarations (E1-E10) binding
  claims to commands and artifacts
- `logic/related_work.md` — typed dependencies (import / bound /
  baseline) including the Ara paper itself as the methodological import
- `src/index.md` — kernel-mode physical layer: claim->module map,
  entry points by use case, environment summary
- `trace/exploration_tree.yaml` — 3-era exploration DAG with decisions,
  dead-ends, pivots, and the meta-observation that the most consequential
  discovery (MPS-CUDA drift) was a measurement bug
- `trace/dead_ends_to_revisit.md` — Lane M-V3, Lane GP DCT/B-spline,
  Lane UNIWARD + SLI1 with revisit protocols
- `trace/compilation_log.md` — provenance trail
- `evidence/index.md` — pointers to existing
  `experiments/results/**/contest_auth_eval.json` files
- `tools/ara_compile.py` — compiler scaffold:
  - walks `experiments/results/` and emits `evidence/results_index.json`
    (12 records detected)
  - walks the topic-indexed memory and emits `trace/events.jsonl`
    (408 events classified into Ara event types)
  - implements Ara Seal Level 1 structural integrity check; current
    state: 10 dangling evidence pointers (expected; resolved by Pass 2)

The compiler runs end-to-end on the current repo:

```bash
.venv/bin/python tools/ara_compile.py
# [ara_compile] evidence: 12 records -> docs/paper/ara/evidence/results_index.json
# [ara_compile] trace events: 408 -> docs/paper/ara/trace/events.jsonl
# [ara_compile] seal level 1: 0 ok, 0 warn, 10 error -> docs/paper/ara/trace/seal_report.json
```

The 10 errors are by design — they identify exactly which evidence files
need to be normalized into the Ara `evidence/` tree (Pass 2).

## Section 5 — Open questions for the user

1. **Should the Ara artifact ship at the comma.ai PR, or only on arXiv
   afterward?** The judges may not know what an Ara artifact is; explaining
   it eats word budget. Recommendation: include a brief "this artifact
   follows the Ara paradigm (arXiv 2604.24658)" footer in the comma.ai PR
   description and link to `docs/paper/ara/PAPER.md` as a curiosity. Save
   the full Ara-native pitch for the arXiv preprint.
2. **Disclosure policy on `evidence/index.md`.** Should we strip the
   `(memory: ...)` references that point to internal feedback files? They
   reveal the council-review structure. Recommendation: replace with
   `(internal review record)` for any public publication.
3. **Should subagent E know about this work before they finish their
   prose pass?** Coordinating now would let them add a single
   "see `docs/paper/ara/PAPER.md` for the agent-native version" line to
   the prose; doing it later requires a small follow-up edit.
4. **Should `tools/ara_compile.py` be in the review-tracker gate?**
   Per CLAUDE.md, `.py` files need the review-tracker mark before
   commit. The file has no test yet; we should either ship it as
   reviewed-by-author or add a basic smoke test.
5. **Does the `EVENT_TYPE_RULES` regex classification meet your bar?**
   I used a small ruleset to map memory filenames to Ara event types. A
   better classification would parse the file contents; that is a 30-min
   follow-up.

## Section 6 — Hand-off list for follow-up implementation passes

- [ ] **Pass 2 — evidence normalization.** Extend `tools/ara_compile.py`
  to copy each detected `contest_auth_eval.json` into
  `docs/paper/ara/evidence/results/<lane>/eval.json` with a
  `provenance.json` sidecar. Update `logic/claims.md` evidence pointers.
- [ ] **Pass 3 — Live Research Manager hook.** Add a post-commit hook
  (`tools/post_commit_hook.sh` already exists) that calls
  `ara_compile.py --evidence-only --quiet` after any commit touching
  `src/tac/` or `experiments/`.
- [ ] **Pass 4 — Ara Seal Level 2.** Score each claim along 6 dimensions
  (evidence relevance, falsifiability, methodological rigor, etc.). This
  needs a small LLM-driven evaluator; consider running the existing
  skunkworks council recursively.
- [ ] **Pass 5 — Ara Seal Level 3.** Pick the lowest-cost claim (probably
  C4 width-scaling) and run the experiment at scaled-down resolution
  (h=8, 100 epochs) inside an isolated Modal job; verify the success
  criterion holds.
- [ ] **Pass 6 — judges' triage block.** Add ~500 tokens at the top of
  `reports/graphs/site/judges_one_pager.md` that orient an Ara agent
  reviewer.
- [ ] **Pass 7 — review-tracker gate.** Add `tools/ara_compile.py` to
  the review tracker; mark reviewed; gate future commits on
  `--validate-only` returning 0 (or only warnings).
- [ ] **Pass 8 — coordinate with subagent E.** Ensure their prose updates
  reference `docs/paper/ara/PAPER.md` for agent-readers and that the
  prose claims align with `logic/claims.md` falsifiers.
