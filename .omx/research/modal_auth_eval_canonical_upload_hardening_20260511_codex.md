# Modal auth-eval canonical upload hardening (2026-05-11)

Status: landed as reusable cleanup; no score claim.

## Change

The Modal CUDA and CPU auth-eval entry points previously duplicated local
archive custody and runtime-upload preparation:

- archive read/hash/byte count;
- `--submission-dir` validation;
- `--inflate-sh` relative-path normalization and traversal rejection;
- deterministic runtime transport zip creation;
- default artifact directory naming.

That duplication is now centralized in
`tac.deploy.modal.auth_eval.prepare_modal_auth_eval_request()`.
`experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py`
both consume the same prepared request object, so CPU/CUDA wrappers cannot
silently drift on path validation or upload custody.

## Score-lowering relevance

This does not lower score directly. It hardens the score-lowering pipeline:
device-axis evals, PR replay, and packet candidates now share one local custody
normalizer before provider submission. That reduces the chance of false
CPU/CUDA conclusions caused by wrapper differences rather than archive/runtime
behavior.

## Verification target

- `src/tac/tests/test_modal_auth_eval.py` includes a direct conformance test for
  the centralized request shape.
- Existing Modal CPU/CUDA local-entrypoint tests continue to exercise the
  wrapper-specific claim, detach, and diagnostic-axis behavior.
