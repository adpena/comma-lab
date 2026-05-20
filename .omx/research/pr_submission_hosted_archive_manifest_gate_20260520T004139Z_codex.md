# PR Submission Hosted Archive Manifest Gate - Codex Round

UTC: 2026-05-20T00:41:39Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Gate Change

`scripts/pre_submission_compliance_check.py --contest-final --strict` now
requires:

```bash
--hosted-archive-manifest-json <path>
```

The manifest schema is `hosted_archive_manifest_v1`. It must bind:

- a real `https://github.com/adpena/.../releases/download/...` URL;
- the exact `archive.zip` SHA-256;
- the exact `archive.zip` byte size;
- the same URL appearing in the public packet text.

The gate also rejects leftover hosted URL placeholders in public packet text.
After tightening, legitimate upstream placeholders such as `<archive_dir>` are
not treated as hosted URL placeholders.

Round extension: the hosted-archive check now scans both the submission
directory public text and any extra `--public-scan-path` surfaces, including
the PR body draft. Placeholder diagnostics are location-aware so the fail-closed
JSON identifies the exact public file/line carrying the blocker.

## Tests

Focused:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pre_submission_compliance_check.py::test_strict_contest_final_requires_hosted_archive_manifest \
  src/tac/tests/test_pre_submission_compliance_check.py::test_strict_contest_final_accepts_hosted_archive_manifest \
  src/tac/tests/test_pre_submission_compliance_check.py::test_strict_contest_final_rejects_hosted_archive_placeholder \
  src/tac/tests/test_pre_submission_compliance_check.py::test_strict_contest_final_rejects_hosted_archive_placeholder_in_public_scan_path \
  -q
```

Result:

```text
4 passed in 0.90s
```

Full file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
48 passed in 3.37s
```

## Strict Gate Rerun

Command used PR101 GOLD CPU baseline threshold, runtime-equivalence proof,
public-source remote-ref manifest, and paired Modal CPU lane/job evidence.

Remaining failures:

```text
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pinned_revision_publicly_visible
```

Hosted details:

```text
hosted_archive_manifest_supplied=false
hosted_archive_public_text_has_no_placeholder=false
placeholders=['<HOSTED_URL_PLACEHOLDER>']
locations=[
  '.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:3:<HOSTED_URL_PLACEHOLDER>',
  'experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:99:<HOSTED_URL_PLACEHOLDER>'
]
```

Source details are unchanged from the prior round: public `origin/main` is still
`e0e7d239b1c330449d9b799a67ad727a8737e789`, while the local source-sync
candidate starts at `31c5fa2a9`.

## Packet Text Update

Updated the PR body draft, submission README, and archive manifest to require
`--hosted-archive-manifest-json` before publication. No hosted archive URL was
invented and no release was created.

No push, PR, release, or hosted archive publication was performed.
