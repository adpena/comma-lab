# PR106 format0D Modal Runtime-Hash Contract Hardening - 2026-05-16

## Scope

This ledger records the fail-fast PR106 format0D paired Modal auth-eval attempt
and the launcher hardening that prevents the same provider-custody mismatch
from recurring.

Candidate:

- archive: `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`
- archive bytes: `186876`
- archive sha256: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- submission runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
- pair group: `pair_pr106_format0d_latent_score_table_20260516`

## Fail-Fast Attempt

Command class:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh \
  --label pr106_format0d_latent_score_table \
  --pair-group-id pair_pr106_format0d_latent_score_table_20260516 \
  --lane-id-base lane_pr106_format0d_latent_score_table_20260516 \
  --expected-runtime-tree-sha256 5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c \
  --execute
```

Outcome:

- classification: `refused_dispatch_runtime_tree_hash_contract_mismatch`
- score claim: `false`
- promotion eligible: `false`
- no score work was performed
- no rows matching `lane_pr106_format0d_latent_score_table_20260516` were found in `.omx/state/active_lane_dispatch_claims.md`
- no matching Modal call-ledger row was found in `.omx/state/modal_call_id_ledger.jsonl`

The CUDA wrapper rejected the supplied runtime hash before remote score work:

```text
expected Modal-uploaded CUDA runtime_tree_sha256:
e596b379c4978ad8ec892d2e7dae8f585941437d698a072cc6e1300fcab56db5

supplied runtime_tree_sha256:
5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c

runtime_content_tree_sha256:
8790ec81e5153a8fe3cb250e82b522763ae82b052b48655556be94acb05d5d51
```

## Root Cause

The supplied hash `5e667424...` came from the PR106 no-score
runtime-consumption proof:

```json
runtime_source_manifest.runtime_content_tree_sha256 =
5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c
```

That proof hashes the five PR106 runtime source files used by the local sidecar
decoder-consumption proof:

- `inflate.sh`
- `inflate.py`
- `src/codec.py`
- `src/model.py`
- `src/pr101_grammar.py`

Modal auth eval has a different custody contract. The wrappers call
`experiments.contest_auth_eval._runtime_dependency_manifest()` and project the
uploaded submission directory through
`tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest()`.
That broader exact-eval manifest includes the uploaded runtime tree as Modal
will extract it, custody files, `upstream/evaluate.py`, and the path-bound
remote root.

Current correct hashes for this runtime:

- Modal CUDA runtime tree:
  `e596b379c4978ad8ec892d2e7dae8f585941437d698a072cc6e1300fcab56db5`
- Modal CPU runtime tree:
  `6b336e17784e6c793ce6638f6c5a0eb5beba8fb3d974d1aa286503702dd84b07`
- shared Modal-uploaded runtime content tree:
  `8790ec81e5153a8fe3cb250e82b522763ae82b052b48655556be94acb05d5d51`

This was a hash-contract bug in the paired launcher, not a candidate result
and not evidence against PR106 format0D.

## Hardened Behavior

`tools/dispatch_modal_paired_auth_eval.py` now computes Modal-uploaded
runtime-tree expectations before provider startup whenever `--submission-dir`
is present.

New behavior:

- default with `--submission-dir`: compute per-axis Modal-uploaded hashes
  locally and pass each wrapper the correct value;
- `--expected-runtime-tree-sha256 auto`: explicit equivalent of the default;
- `--expected-cuda-runtime-tree-sha256` and
  `--expected-cpu-runtime-tree-sha256`: allow explicit per-axis hashes;
- stale legacy single-hash input fails locally with a clear `FATAL` and exit
  code `2`, before Modal app startup or claim/spawn work.

Focused verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_modal_paired_auth_eval.py \
  src/tac/tests/test_modal_auth_eval.py -q
```

Result: `44 passed`.

The stale-hash command now fails locally:

```text
FATAL: expected runtime tree does not match the Modal uploaded --submission-dir runtime for contest_cuda
```

The corrected plan was regenerated at:

```text
experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_auth_eval_plan_autohash_20260516.json
```

That plan is non-score, non-promotional until the paired CPU/CUDA Modal jobs are
actually dispatched, harvested, and adjudicated.

