# ddm_fx1 Next If Resumed

1. Commit is still pending because the managed sandbox refused Git writes (`git apply --cached` backing-store temp files and `.git/COMMIT_EDITMSG`). Retry from a Git-write-capable shell using serializer intent mode; the working tree contains unrelated same-file PG1/Q3-projector hunks in `experiments/train_tr1_partition_renderer_mlx.py`, so do not whole-file stage that trainer unless those hunks are intentionally owned by the landing.
2. After the commit, verify `git show --stat HEAD` contains only the ddm_fx1 trainer/test/receipt hunks and does not absorb unrelated same-file PG1 work. The last observed pre-landing `HEAD` was `dd5d0c7e06`.
3. All three TP1-named trainer telemetry debts are cleared in the working tree: pose+birth loss itemization, parent-active boundary decay, and post-restore resume decay display.
4. The regenerator value-ledger debt is verified cleared in the current checkout by `src/tac/tests/test_ddm_jd1_ticket_regenerate.py`; do not mutate already-emitted hash-custodied tickets.
5. BP1 full-suite status is environment-blocked on local MLX Metal availability; the new pure BP1 regressions passed independently.
6. No scorer or launch step is queued by ddm_fx1. This landing is telemetry/provenance plumbing only.
