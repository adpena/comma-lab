# Interactive Notebooks — canonical links + forward plan

## Published molab entry (marimo Notebook Competition #2, task #347)

- **Canonical URL (do NOT change):**
  <https://molab.marimo.io/github/adpena/witness-machine/blob/main/notebooks/witness_machine_v12.py>
- **What it is:** "The Witness Machine" — the paper-REPLICATION entry (implement-a-paper track),
  a self-bootstrapping marimo notebook (42 cells) living in the public repo
  `adpena/witness-machine` at `notebooks/witness_machine_v12.py` on `main`.
- **URL mechanics:** this is a molab GitHub-proxy URL — it renders whatever is at that exact
  path+branch. Fixing/iterating the notebook = pushing to that path; the URL never changes as
  long as the path and branch stay identical.
- **Runtime contract:** sealed-bundle bootstrap (sha256-pinned download + cache; no repo-checkout
  imports — the pre-`f111248` breakage was exactly a repo-checkout import assumption that does
  not exist on molab's mirror). Verified e2e 2026-07-10: anonymous render clean, zero console
  errors, bundle 3,704,001 bytes sha256-match, molab-faithful isolated run-all 0 tracebacks.
  Server-side run-all requires molab sign-in (the only auth-gated step).
- **Evidence:** `.omx/research/marimo_linkfix_20260710.md` +
  `.omx/research/marimo_linkfix_evidence_20260710/` (screenshots, local-only). Sister record:
  `paper/README.md` (the contest-entry surface).
- **Timeline honesty:** published pre-deadline 2026-07-09 (then broken-on-molab at runtime);
  bootstrap repair `f111248` + verification landed post-deadline 2026-07-10, same URL.

## Forward plan (operator direction, 2026-07-10)

1. **Continued iteration on the published notebook** is expected — same URL discipline applies
   (push to `adpena/witness-machine:main:notebooks/witness_machine_v12.py`, never rename/move).
2. **A separate OWN-PAPER notebook is planned**: a from-our-own-research interactive notebook
   for OUR paper (`docs/paper/00_abstract.md` … `07_discussion.md`), distinct from the
   paper-replication competition entry above. Not started; when it starts, it gets its own repo
   path + a row in this file so the URL is never disk-orphaned again (the root bug the
   2026-07-10 link-fix closed).
3. **The paper itself needs substantial work later** — `docs/paper/` sources are drafts; the
   own-paper notebook and the paper revision should be planned together so figures/results are
   generated from one source of truth.

## Discipline

Every published notebook URL MUST be recorded here (and in the entry's own README) the moment
it exists. A URL that lives only in a signed-in dashboard is orphaned signal — the 2026-07-10
incident (link unfindable on disk, task blocked) is the anchor.
