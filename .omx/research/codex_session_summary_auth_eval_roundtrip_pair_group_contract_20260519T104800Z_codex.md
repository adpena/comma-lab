# Codex Session Summary: Auth-Eval Roundtrip Pair-Group Contract

Codex continued ITEM_7 from the PR101 OP-7 score-response matrix into the
exact-eval launch step. The generated Modal wrapper commands failed closed
before provider spawn because `auth_eval_roundtrip_matrix` still emitted direct
single-axis wrapper commands without the paired-by-default metadata that the
Modal auth-eval wrappers now require.

Landed in this slice:

- contest CPU/CUDA rows from `tac.auth_eval_roundtrip_matrix` share a
  deterministic `--pair-group-id`;
- diagnostic Modal rows carry an explicit non-promotional
  `--single-axis-waiver-reason`;
- focused regression tests cover the shared matrix helper and the PR101 OP-7
  matrix consumer;
- the PR101 OP-7 matrix artifact was regenerated with pair-grouped commands.

Regenerated artifact hashes:

- `score_response_matrix.json`:
  `36a28e32f70d5dec053287f397829cb6ecd0f1a663718cb27a793262ee5561a2`
- `score_response_matrix.md`:
  `654fac6497fcaec2b9b9701a2684bb65d7123b8606741059441e28d1cdeae90d`

Next ITEM_7 step:

1. rerun paired PR101 OP-7 baseline/candidate exact eval commands;
2. recover the four Modal auth-eval outputs;
3. run the generated contest CPU/CUDA score-response probes;
4. review the paired result packet before any score, promotion, rank, or kill
   language.

No score claim was made in this slice.

