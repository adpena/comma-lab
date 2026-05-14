# Long-form writeup aspiration notes

This document tracks what would be needed to turn the current `docs/paper/`
notes into a long-form paper or technical report someday. It is not a public
promise, not a submission checklist, and not evidence that a real manuscript
has been drafted. For now the practical purpose is community documentation,
continued iteration, and historical record.

## Status as of 2026-05-05

The repo has substantial notes, figures, and postmortem material, but no
finished paper draft yet. The missing work is drafting, synthesis,
bibliography, figure wiring, and final claims audit.

### Source files (paper-note content)

| File | Status | Notes |
|---|:---:|---|
| `00_abstract.md` | NOTES | Abstract-shaped notes; not final claim language |
| `01_introduction.md` | NOTES | Intro-shaped material; needs synthesis and citations |
| `02_method.md` | NOTES | Method material; needs pruning and a coherent paper spine |
| `03_gradient_bug.md` | NOTES | Strong section candidate; needs validation figure and references |
| `04_results.md` | NOTES | Results material; needs final evidence table and claims audit |
| `05_production.md` | NOTES | Production implications; likely appendix or blog material |
| `06_related_work.md` | NOTES | Related-work notes; needs real bibliography conversion |
| `07_discussion.md` | NOTES | Discussion material; needs tightening and scope control |
| `99_appendix_postmortem.md` | NOTES | Postmortem appendix material; useful but not final manuscript copy |

### Figures

See `figures/MISSING.md`. Three PNGs are present
(`leaderboard_comparison.png`, `score_decomposition.png`,
`step_curve_comparison.png`) but **not yet wired into the markdown source**
with `![](figures/...)` calls. Action item before submission:

- [ ] Wire the three present PNGs into the appropriate sections
- [ ] Generate the 6-panel diagnostic visualization (or document its absence
      with a TODO in §3.6)
- [ ] (Optional) Generate the gradient-flow validation chart and Lagrangian
      annealing curve

### Cross-reference audit

Performed 2026-05-05. All in-paper section cross-references resolve except
for one (now fixed): `02_method.md` previously referenced "Section 4.19"
for an Int4+LZMA2 scheme; that section did not exist. The reference now
points at §4.0 frontier table + §7.4 future work.

### Missing manuscript pieces if we later pursue a paper

- **Actual draft synthesis** — the current files are useful notes, not a
  coherent manuscript.
- **Thesis and contribution boundary** — decide whether the paper is about the
  gradient bug, contest archive compilers, public-frontier forensics, or the
  broader game-theoretic methodology. Trying to publish all of them at once
  will read unfocused.

- **Author block / affiliations** — needed for any formal front matter. The
  user must decide on attribution format (single-author? lab affiliation?
  acknowledgement of LLM collaboration in the byline vs. in §7.1?).
- **References / bibliography** — paper currently uses inline-citation
  format (e.g., `[Fridrich 2009]`, `[Yousfi et al. 2020]`). For any formal
  publication this should be promoted to a proper bibliography file
  (`references.bib`) and linked. ~20 unique inline citations to convert.
- **PDF build** — the current source is markdown. A future formal version
  would need a real build path, likely pandoc → LaTeX/PDF.

### Pre-submission compliance

- [ ] Run a final pass to ensure no `[contest-CUDA]` claim exists without
      the corresponding archive SHA-256 + JSON path
- [ ] Run a claims-language pass for `prediction`, `proxy`, `MPS`, `CPU`,
      `stub`, and byte-only rows. These may appear only as roadmap or
      diagnostic evidence, never as ranked scores, promoted lanes, family kills,
      or paper empirical anchors.
- [ ] Confirm every `A-negative`, `invalid`, `killed`, `dead`, `retired`, or
      `falsified` phrase names the exact measured archive/runtime/config and a
      concrete reactivation criterion. Single-config failures must not read as
      broad method-family claims.
- [ ] Confirm every Omega/Omega-OPT or intN predicted score is labeled
      non-ranking unless an exact CUDA `contest_auth_eval*.json` path, archive
      SHA-256, runtime tree SHA-256, and component recomputation are cited.
- [ ] Verify no transient scratch paths (the forbidden-pattern class described
      in `CLAUDE.md`) appear in any persisted artifact. Use repo-relative
      `experiments/results/<lane>_<timestamp>/`, `.omx/state/`, or `.omx/tmp/`
      instead.
- [ ] Confirm all archive SHA-256 values in §4 are reproducible from disk
- [ ] Confirm the public PR / leaderboard position figures (PR #107 0.2293,
      ~11th place; PR #106 0.20945 frontier) reflect the leaderboard at
      submission time

### Disclosure decisions for the user if we later publish a paper

- **Submission timing.** The public-release policy notes that the official PR
  is itself a disclosure moment. Any future paper would publish additional
  methodology detail beyond the contest PR. The user must decide timing.
- **Cloudflare site coupling.** The Cloudflare site is approved for
  detail. A future paper may either link to it or stand alone.
- **Future-work disclosure.** §7.4 names specific architectural directions
  (mask2mask joint pair generation, self-compressing weights, meta-learning
  for cross-clip generalization). Some of these may be future research the
  user wants to keep private.

## If we later pursue a formal paper

0. Decide whether a paper is worth pursuing at all, and if so choose the
   single main thesis.
1. Wire figures into markdown source.
2. Generate the 6-panel diagnostic (or accept its absence).
3. Draft a real 6-10 page manuscript from the notes.
4. User reviews abstract + appendix postmortem.
5. User decides on author block + LLM-collaboration acknowledgement format.
6. Build a PDF only after the draft is coherent and claims are audited.
7. Open the contest write-up PR (Tier 2 Item 3, `submissions/apogee/WRITEUP.md`)
   and link any long-form public writeup only if one is actually published.

## Out of scope for this checklist

- Submitting any formal paper.
- Choosing a venue or subject category.
- Press / blog coordination.
