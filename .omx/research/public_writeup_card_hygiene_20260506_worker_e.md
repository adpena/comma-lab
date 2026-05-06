# Public Writeup And Card Hygiene - 2026-05-06 Worker E

Scope: OSS/writeup/public-card wording only. No score code, archive builders,
eval scripts, dispatch tooling, or runtime files were changed.

## Changes Landed

- Root README now frames the repository as a community/historical-record
  workspace and removes the stale proxy "current best" line.
- README and writeup playbook now include explicit evidence-grade placement
  rules for ranked rows, roadmap rows, external context, and invalid evidence.
- HF dataset card now states that public PR metadata is `external` context
  until exact replay evidence exists, and removes the promised DOI wording.
- Paper/writeup/Ara surfaces now avoid committing to arXiv/preprint release.
  They use optional long-form/public-release language until the human chooses a
  venue.
- Paper introduction no longer claims current state of the art from stale
  historical scores; it points ranked performance wording to exact evidence
  rows.
- C-067 attribution and atomic-decomposition addendum now use conditional
  public-release/paper language instead of assuming a future paper surface.

## Remaining Boundary

Several deep historical ledgers still contain older "winner", "SOTA",
"current best", and arXiv-reference wording as archived research context. They
were not bulk-rewritten because the task scope was public/release/writeup
surfaces and because those ledgers preserve dated reasoning. Any future public
export should run strict public-release hygiene and should use only the
evidence-graded docs as source text.

## Focused Verification

Focused markdown/release-view tests passed:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_materialize_pr_archive_release_view.py \
  src/tac/tests/test_materialize_pr_archive_kaggle_mirror.py \
  src/tac/tests/test_materialize_comma_lab_public_export.py \
  src/tac/tests/test_audit_public_publish_links.py \
  src/tac/tests/test_audit_staged_public_release_hygiene.py \
  src/tac/tests/test_oss_publish_staging.py \
  -q
```

Result: `31 passed in 1.52s`.

Static public-link audit over the changed publish/writeup docs passed:

```bash
.venv/bin/python tools/audit_public_publish_links.py --repo-root . --format json \
  README.md \
  docs/comma_pr_archive_dataset_card.md \
  docs/writeup_draft.md \
  docs/writeup_playbook.md \
  docs/paper_outline.md \
  docs/research_roadmap.md \
  docs/paper/00_abstract.md \
  docs/paper/01_introduction.md \
  docs/paper/07_discussion.md \
  docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md \
  docs/paper/methodology_addendum_atomic_decomposition_yf_floor_20260502.md \
  docs/paper/ara/RECOMMENDATION_20260429.md \
  docs/paper/ara/PAPER.md \
  docs/paper/ara/trace/compilation_log.md
```

Result: `link_count=10`, `violation_count=0`.
