# trace/compilation_log.md

Provenance trail for this Ara artifact. Each entry records when a
section of the artifact was compiled, by what tool, from what sources.

---

## 2026-04-29 — initial Ara skeleton compiled

- **compiled by**: hand (subagent in parallel session, not subagent E)
- **compiled from**:
  - `docs/paper_outline.md`
  - `docs/writeup_draft.md`
  - `reports/writeup_working.md`
  - `reports/graphs/final_writeup_draft.md`
  - `reports/graphs/site/judges_one_pager.md`
  - `reports/graphs/site/lab_notebook.md`
  - `reports/graphs/site/experiment_journal.md`
  - `AGENTS.md`
  - `PROGRAM.md`
  - 7 most recent `project_*_20260429*.md` project-memory files
- **paradigm reference**: arXiv 2604.24658 ("The Last Human-Written Paper:
  Agent-Native Research Artifacts")
- **scope**: layer skeleton (PAPER.md root manifest; logic/ with problem,
  claims, experiments, related_work; src/index.md; trace/exploration_tree
  + dead_ends_to_revisit + this log; evidence/index.md placeholder)
- **out of scope**: actual evidence-file generation. The `evidence/`
  directory is a placeholder index pointing to existing
  `experiments/results/**` JSON files. Subsequent compilation passes by
  `tools/ara_compile.py` will populate it.
- **disclosure policy adherence**: yes. Public-safe surface only includes
  [contest-CUDA] tagged scores; Lane W / Lane Omega / Lane DARTS-S
  recipes are NOT mentioned in this skeleton.
- **review status**: skeleton; not yet reviewed. Pending: review by
  user before any of these layers is published on Cloudflare, a public PR,
  an optional preprint, or another public release surface.

## next compilation passes

1. **Evidence ingestion** — `tools/ara_compile.py` walks
   `experiments/results/**/contest_auth_eval.json` and emits a normalized
   index under `evidence/results/`. Done in 2026-04-29 second pass below.
2. **Live Research Manager hook** — wire into the standard
   `tools/review_tracker.py` flow so research events (decisions,
   dead-ends, pivots) land in `trace/exploration_tree.yaml` automatically.
3. **Ara Seal Level 1** — schema validator that confirms every claim has
   a forensic binding to evidence and code.
4. **Ara Seal Level 3** — execution-reproducibility check that runs each
   experiment in `logic/experiments.md` at scaled-down resolution and
   confirms the artifact's success criterion.
