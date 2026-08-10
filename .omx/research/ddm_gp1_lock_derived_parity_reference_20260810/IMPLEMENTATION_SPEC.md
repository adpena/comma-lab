# ddm_gp1 implementation spec — lock-derived auth-eval parity reference

## Objective

Replace only the missing-`upstream/.venv/bin/python` branch in
`experiments/contest_auth_eval.py` with a fail-closed reference derived by asking `uv` to
export the pinned `upstream/uv.lock` for an explicitly declared dependency group. Preserve
the existing declaration-vs-measurement hardening: `--upstream-python` selects the interpreter
that actually runs `upstream/evaluate.py`, but never asserts parity.

## Source facts already resolved

- `_record_provenance()` stores the selected `auth_eval_python`, and `main()` passes that exact
  value to `_run_upstream_evaluate(..., python_executable=...)`; the selected interpreter is the
  actual scorer interpreter, not wrapper-only metadata.
- The exact supported command is:
  `uv export --frozen --no-emit-project --no-hashes --format requirements-txt --directory upstream --group <group>`.
- With group `cu128`, the real lock exports `numpy==2.3.4`, `timm==1.0.22`,
  `torch==2.9.0+cu128`, and two marker-split torchvision rows. On Linux x86_64 the surviving
  row is `torchvision==0.24.0+cu128`; on Linux aarch64 it is `torchvision==0.24.0`.
- `upstream/uv.lock` currently hashes to
  `eca4542ad8d21354fd1f2bada74e8659329c0176b17f1ae808e04e023674231f` and remained unchanged
  across the reproduced export.

## Required behavior

1. Add an explicit CLI declaration for the upstream uv group. Use the name
   `--upstream-uv-group`. It has no default that silently chooses an axis.
2. If `upstream/.venv/bin/python` exists, keep the existing reference branch and identity
   behavior unchanged. A uv group is not required in that branch.
3. If that interpreter is missing, derive the reference by executing the exact frozen `uv
   export` command above for `args.upstream_uv_group`.
4. Parse the export and evaluate every requirement marker inside the evaluation interpreter,
   using that interpreter's `packaging.markers.default_environment()`. The wrapper/host marker
   environment is not authority. A test-only marker-environment override may be supported so
   both Linux architectures can be executed deterministically.
5. Resolve exactly `AUTH_EVAL_ENV_PACKAGES` (`torch`, `torchvision`, `timm`, `numpy`). Each must
   have exactly one surviving exact `==` requirement. Zero is missing; more than one is
   ambiguous. Never take first-match.
6. Preserve the existing comparison semantics and report shape: exact `python_version` plus
   per-package version comparison. The lock is universal across supported Python patch
   versions, so use the evaluation interpreter's marker environment `python_full_version` as
   the reference Python version while deriving package rows for that same environment. Record
   that source explicitly; do not imply the lock pins a Python patch version.
7. Hash `upstream/uv.lock` before and after the export/marker-resolution operation. Any change
   is a refusal even if package resolution otherwise succeeds.
8. Every failure stays an `env_mismatch` and must carry a precise machine-readable reason:
   no group, uv absent, uv export nonzero/timeout, parity package missing, marker evaluation or
   parsing failure, marker ambiguity (including both surviving rows), non-exact requirement,
   lock missing/unreadable, or lock SHA change.
9. Record enough provenance in `auth_eval_environment.upstream_reference` to reproduce the
   derivation: source kind, declared group, exact argv, uv path/version if cheaply available,
   lock path and before/after SHA, marker environment, selected requirement rows, and failure
   detail. Do not add version literals.
10. Update both `experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py` so their
    existing locked-venv dispatch commands also pass `--upstream-uv-group` with the already
    declared `UPSTREAM_UV_GROUP_CUDA` / `UPSTREAM_UV_GROUP_CPU` constants. Do not invent a second
    group constant.

## Tests and controls

Extend `src/tac/tests/test_contest_auth_eval.py` and, where useful,
`src/tac/tests/test_modal_auth_eval.py` with behavioral tests for:

- Existing `upstream/.venv` identity branch remains unchanged.
- Real lock export for group `cpu` plus the real `upstream/.venv/bin/python` (against a synthetic
  upstream root containing only copied `pyproject.toml` and `uv.lock`, so `.venv` is absent)
  passes with no mismatch on this host.
- The root repo `.venv/bin/python` against that same derived `cpu` reference produces
  `env_mismatch`; the evidence contract keeps `score_claim=False`.
- The real cu128 export resolves torchvision to `0.24.0+cu128` for simulated Linux x86_64 and
  `0.24.0` for simulated Linux aarch64, with marker evaluation executed by the selected
  interpreter.
- Missing group, missing uv, export rc != 0, a missing parity package, ambiguous surviving
  marker rows, and lock mutation all refuse with distinct reasons.
- CUDA and CPU Modal command construction passes the existing group constant to the new flag.
- The real `upstream/uv.lock` SHA is unchanged across the entire focused test run.

Run at minimum:

```text
.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_locked_env_probe.py -q
```

Also run `py_compile` for every touched Python file and `git diff --check`.

## Do not touch

- Anything under `upstream/`.
- The staged index.
- `.omx/research/ddm_cr1_composition_row_827_20260801.md`.
- `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`.
- `src/tac/optimization/direct_description_carrier_compose.py`.
- Any unrelated dirty file.

Do not launch Modal, a scorer, or an exact eval. Do not claim the refused candidate score. Do not
commit; the parent session owns review tracking and serializer commit.
