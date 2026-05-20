# PR Submission Source Pin + Threshold Matrix - Codex Round

UTC: 2026-05-20T00:33:40Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Source-Pin State

Local main:

```text
31c5fa2a9c6b98a8d532d6ba02b23f14fa20480f council: T3 grand council symposium PROCEED_WITH_REVISIONS on PR submission corrected draft
```

Observed public remote:

```text
e0e7d239b1c330449d9b799a67ad727a8737e789 refs/heads/main
```

Remote containment checks:

```text
git branch -r --contains HEAD
# no output

git ls-remote origin 31c5fa2a9c6b98a8d532d6ba02b23f14fa20480f
# no output
```

Runtime custody at local `HEAD`:

```text
submission_dir/inflate.py
submission_dir/inflate.sh
submission_dir/src/codec.py
submission_dir/src/codec_sidecar.py
submission_dir/src/frame_selector.py
submission_dir/src/model.py
```

`git diff --name-only HEAD -- submission_dir runtime files` returned no runtime
file differences; only packet text files are dirty. Therefore the source-sync
candidate exists locally, but it is not public yet. Do not replace
`<PINNED_COMMIT>` or cite the local full SHA in public text until the operator
authorizes a public push or tag.

## Threshold Matrix

Strict gate command was run with all current runtime-equivalence and CPU
lane/job evidence.

Default policy:

```text
--max-submission-score 0.192
```

Failures:

```text
contest_cpu_auth_eval_score_at_or_below_submission_threshold
public_source_pinned_revision_present
```

Threshold detail:

```text
selected_axis=contest_cpu score=0.1920513168811056 source=strict_formula threshold=0.192
```

PR101 GOLD CPU baseline policy:

```text
--max-submission-score 0.1928450127024255
```

Failures:

```text
public_source_pinned_revision_present
```

Threshold detail:

```text
selected_axis=contest_cpu score=0.1920513168811056 source=strict_formula threshold=0.1928450127024255
```

## Operator Decision

NEEDS-OPERATOR-DECISION:

- A: Keep the absolute `0.192` final-packet threshold. This remains stricter
  than the cited PR101 GOLD CPU baseline and blocks this packet at
  `0.1920513168811056` `[contest-CPU]`.
- B: For this post-deadline PR packet, set the threshold to the current top
  merged PR101 GOLD CPU baseline `0.1928450127024255`. This clears the CPU
  threshold check while preserving the evidence-axis labels and still blocks
  publication until source pinning is real.

## Packet Text Update

Updated the PR body draft, submission README, and archive manifest to reflect:

- the local source-sync candidate starts at `31c5fa2a9`;
- observed public `origin/main` is still `e0e7d239b`;
- no public source pin should be cited until push/tag authorization;
- the CPU threshold check clears under the PR101 GOLD CPU baseline policy but
  not under the default absolute `0.192` policy.

No push, PR, release, or hosted-archive publication was performed.
