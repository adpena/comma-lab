# PR104 Exact Replay Adapter

Source-sized, fail-closed adapter for public PR #104 `qhnerv_ft_best`.

The adapter does not contain archive payload bytes and does not install
packages at inflate time. It wraps the public `inflate.py` from
`experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best`
through the repo-managed Python environment, then maps the canonical extracted
member for `0.mkv` to `0.bin` before invoking the public decoder.

CUDA exact replay has not been run by this adapter. A GPU dispatch must first
claim a lane with `tools/claim_lane_dispatch.py`, then run the canonical
`experiments/contest_auth_eval.py` path with this `inflate.sh`.
