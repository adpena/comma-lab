# Cathedral Authorized Dispatch Custody Hardening - 2026-05-16

## Summary

Integrated a dispatch-authority hardening patch for
`tools/cathedral_autopilot_autonomous_loop.py` so the Cathedral autopilot fails
closed before ranking, claim creation, or journal writes when operator-authorized
mode sees malformed custody or cost fields.

## Bug Classes Closed

- `estimated_dispatch_cost_usd` must be finite and positive. Zero, negative,
  `NaN`, and infinities are no longer converted into privileged
  `eig_per_dollar` rank authority.
- Operator-authorized caps must be finite and positive, and cumulative spend
  must be finite and non-negative before the loop considers dispatch authority.
- Authorized-mode journal paths must be durable repo-local paths under
  `.omx/state/` or `reports/`; transient paths such as `/tmp`,
  `/private/tmp`, `/var/tmp`, or the platform temp directory are refused before
  any dispatch claim is written.
- `dispatch_packet_sha256`, `archive_sha256`, and `runtime_tree_sha256` must be
  real 64-character hex SHA-256 values when used for self-authorization.
  Placeholder strings such as `dispatch_packet_sha256_for_a` do not satisfy
  custody.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `git diff --check`

## Follow-Up

Kant's read-only prediction-band audit found that literature/source scope is now
guarded, but numeric prediction bands can still influence rank authority without
their own axis, baseline, uncertainty, supersession, and empirical-anchor
custody. Next concrete patch should add a reusable prediction-band custody model
and wire it into the composition matrix, ranking JSON, Cathedral loader, and
preflight.
