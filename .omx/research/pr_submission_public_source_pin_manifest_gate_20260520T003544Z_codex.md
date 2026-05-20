# PR Submission Public Source Pin Manifest Gate - Codex Round

UTC: 2026-05-20T00:35:44Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Gate Change

`scripts/pre_submission_compliance_check.py` now accepts:

```bash
--public-source-ref-manifest-json <path>
```

The manifest schema is `public_source_ref_manifest_v1`. It is a deterministic
remote-ref snapshot, intended to be generated from a command such as:

```bash
git ls-remote origin refs/heads/main
```

When supplied, the gate checks:

- `public_source_ref_manifest_exists`
- `public_source_ref_manifest_json_object`
- `public_source_ref_manifest_schema`
- `public_source_ref_manifest_has_refs`
- `public_source_pinned_revision_publicly_visible`

Pinned `/commit/<sha>` URLs and bare 40-character git SHAs must match an
observed remote ref tip. Pinned release tags and `tree/<ref>` URLs must match
an observed tag or branch ref. This deliberately prevents a local-only SHA from
clearing the public source blocker.

Round extension: public-source reproducibility now scans the same public packet
surface set as the hosted-archive gate: submission public files plus any
`--public-scan-path` inputs. This closes the split-brain case where a PR body
contains the only source pin but the checker only reads the submission
directory.

## Current Remote Snapshot

Landed manifest:

`.omx/research/pr_submission_public_source_refs_20260520T003544Z_codex.json`

Observed public remote:

```text
refs/heads/main e0e7d239b1c330449d9b799a67ad727a8737e789
```

Local source-sync candidate:

```text
31c5fa2a9c6b98a8d532d6ba02b23f14fa20480f
```

Verdict: the local candidate is not publicly visible in this snapshot.

## Tests

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
```

Result:

```text
49 passed in 3.45s
```

New regression coverage:

- visible pinned commit accepted when the manifest ref points at it;
- visible pinned commit accepted when it appears only in a `--public-scan-path`
  PR body draft;
- pinned-looking local SHA rejected when the manifest ref does not contain it;
- bare 40-character git SHA visibility uses git-SHA width, not archive SHA-256 width.

## Strict Gate Rerun

Command used the PR101 GOLD CPU baseline threshold, all existing runtime and
lane/job evidence, and the current public-source remote-ref manifest.

Remaining failures:

```text
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pinned_revision_publicly_visible
```

The CPU threshold check clears under the PR101 baseline policy. The source
section now confirms it scanned all current public packet surfaces:

```text
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/report.txt
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json
.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md
```

Public source pinning is still genuinely blocked until the operator authorizes
a public push or tag and a refreshed `public_source_ref_manifest_v1` proves the
pinned ref is visible. Hosted archive publication is also still blocked until a
real release URL replaces the placeholders and is bound by
`hosted_archive_manifest_v1`.

No push, PR, release, or hosted archive publication was performed.

## Round Extension - Source Pin Placeholder Diagnostics

UTC: 2026-05-20T00:59:21Z

`inspect_public_source_reproducibility()` now fails closed on unresolved source
pin placeholders such as `<PINNED_COMMIT>` across the merged public surface set.
The new check is:

```text
public_source_pin_text_has_no_placeholder
```

Regression added:

```text
test_contest_final_rejects_source_pin_placeholder_in_public_scan_path
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "public_source or source_pin"
4 passed in 0.99s

.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "hosted or axis_labels"
6 passed in 1.10s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
51 passed in 4.09s
```

Real strict gate still exits `1` / `passed=false`. The public-source blocker is
now location-aware:

```text
public_source_pinned_revision_present
public_source_pin_text_has_no_placeholder
public_source_pinned_revision_publicly_visible
```

Source placeholder locations:

```text
.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:68:<PINNED_COMMIT>
.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:71:<PINNED_COMMIT>
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:71:<PINNED_COMMIT>
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:95:<PINNED_COMMIT>
```

The remaining source action is unchanged: replace these placeholders only after
an operator-authorized public push or tag, then refresh
`public_source_ref_manifest_v1` so the pinned ref is publicly visible.
