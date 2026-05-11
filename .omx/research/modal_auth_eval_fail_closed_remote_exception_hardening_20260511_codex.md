# Modal auth-eval fail-closed remote exception hardening (2026-05-11)

## Summary

After `modal_auth_eval_detached_claimed_recovery_20260511_codex.md` landed, Codex
launched the first claimed detached PR103-on-PR106 CUDA raw-output-manifest run:

```text
output_dir=experiments/results/modal_auth_eval/pr103_pr106_cuda_raw_manifest_modal_20260511T055005Z
call_id=fc-01KRASC7KPWKD0JW09KCTZN062
archive_sha256=ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce
archive_bytes=185578
runtime_zip_sha256=9645c2836e5a1843556cbb3614e0603ba8060200f8e61ffa6715bce58aec90ad
source_commit=4730f71102f86418609a82cf20a5f8432855bb80
```

Modal returned a provider-level `RemoteError` with an empty message and no
remote artifact payload. This is an infrastructure failure only: no score was
claimed, no lane was promoted, and `tools/recover_modal_auth_eval.py` wrote a
terminal `failed_modal_auth_eval_no_score_claim` dispatch row.

## Permanent fix

The CUDA and CPU Modal auth-eval functions now wrap their remote body in the
shared helper `tac.deploy.modal.auth_eval.fail_closed_remote_exception_result`.
Unexpected remote exceptions return a structured non-promotable result with:

- `passed=false`;
- `score_claim=false`;
- `promotion_eligible=false`;
- `returncode=98`;
- exception type, message, and traceback;
- a validation JSON artifact harvested by the canonical recovery tool.

This keeps provider failures from becoming blank claims while preserving enough
evidence to debug the actual root cause on the next dispatch.

## Files

- `src/tac/deploy/modal/auth_eval.py`
- `experiments/modal_auth_eval.py`
- `experiments/modal_auth_eval_cpu.py`
- `tools/recover_modal_auth_eval.py`
- `src/tac/tests/test_modal_auth_eval.py`
- `src/tac/tests/test_modal_auth_eval_recovery.py`

## Verification

```bash
.venv/bin/python -m py_compile \
  src/tac/deploy/modal/auth_eval.py \
  experiments/modal_auth_eval.py \
  experiments/modal_auth_eval_cpu.py \
  tools/recover_modal_auth_eval.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_modal_auth_eval.py \
  src/tac/tests/test_modal_auth_eval_recovery.py \
  src/tac/tests/test_plan_dual_device_auth_eval.py
```

Result: `34 passed`.

## Next score-lowering use

Rerun the PR103-on-PR106 CUDA raw-output-manifest dispatch with the same archive
and uploaded runtime. If the remote path fails again, the recovered result must
contain the traceback. If it passes, pair it with the Modal CPU axis using the
same archive/runtime to distinguish raw renderer-output drift from
SegNet/PoseNet device drift.

## Addendum: detached launch semantics

The first post-hardening detached rerun still returned a blank `RemoteError`.
A blocking debug invocation with `--inflate-timeout 1 --evaluate-timeout 1`
entered the remote body, executed `inflate.sh`, and failed closed at the
expected timeout:

```text
lane_id=pr103_pr106_modal_bootstrap_debug
job=pr103_pr106_cuda_bootstrap_debug_modal_20260511T055852Z
classification=modal_wrapper_detach_semantics_bug_not_codec_runtime_bug
```

Conclusion: wrapper `--detach` alone is insufficient for ephemeral Modal apps.
The provider-level command must also be detached:

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py \
  ... \
  --detach --provider-detach-ack
```

`experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py` now
fail before claim/spend when wrapper `--detach` is used without
`--provider-detach-ack`. `AGENTS.md` records this as durable operator protocol.
