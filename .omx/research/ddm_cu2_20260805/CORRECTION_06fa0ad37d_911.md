# DDM CU2 Correction Memo: commit 06fa0ad37d / task #911

## Correction

Commit `06fa0ad37d663cc95bf6b805f3304a5f26e56552` is not accurately described by its message tail:

> No behaviour change: every edit is one % -> %% inside a help string.

The commit also recorded a 208 insertion / 7 deletion change to
`experiments/train_tr1_partition_renderer_mlx.py`. That trainer change belongs to the `ddm_bs2`
trainer wire-in stream, not the `ddm_df1` CLI-help repair body.

## Evidence

- `git show --stat --name-status 06fa0ad37d` records 9 files, including
  `experiments/train_tr1_partition_renderer_mlx.py`.
- `git show --numstat --format= 06fa0ad37d` records
  `208  7  experiments/train_tr1_partition_renderer_mlx.py`.
- `.omx/research/ddm_rs2_orphan_resumption_20260802.md:246-249` says the trainer wiring
  "was absorbed into `06fa0ad37d` (`ddm_df1`'s unrelated CLI fix)" and has no commit of its
  own.

## Disposition

Do not revert the trainer change merely because it was misattributed. The RS2 recovery memo
classifies the trainer wire-in as real recovered work, and this CU2 unit did not find a content
correctness reason to back it out.

The structural cure landed in CU2 is attribution-side hardening:

- the serializer now refuses staged files not declared by `--files` before commit, including the
  repair/no-stage real-index path;
- post-commit recorded-but-not-requested files are now hard rc=15, while requested-but-not-recorded
  remains warn-only because no-op requested files are legitimate.

Same-file hunk authorship remains governed by the existing base/expected SHA and patch-file
surfaces; this memo corrects the false commit-body claim and the CU2 code closes the undeclared-file
class.
