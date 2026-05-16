# L5 v2 TT5L Dykstra Preflight Gate - 2026-05-16

## Scope

Follow-on hardening for the L5 v2 TT5L cargo-cult unwind. The prior landing
made `l5_v2_dispatch_readiness()` put the TT5L Dykstra feasibility artifact
before side-info proof, timing smoke, paired anchor planning, and
stack-of-stacks work. This landing wires that ordering into the visible
`tools/all_lanes_preflight.py` operator-briefing gate so the rule cannot exist
only inside the L5 helper.

## Landed Change

`tools/all_lanes_preflight.py` now validates the nested
`l5_v2_frontier_readiness.tt5l_campaign_readiness` payload:

- TT5L campaign schema must be present.
- TT5L campaign authority flags must stay false.
- Dykstra status schema must be present and match the top-level Dykstra
  validity bit.
- Dykstra status authority flags must stay false.
- TT5L next action authority flags must stay false.
- TT5L next action must not route back to PR106.
- If Dykstra is missing, the first action must be
  `run_tt5l_dykstra_feasibility_polytope`.
- If Dykstra exists but side-info proof is missing, the first action must be
  `materialize_tt5l_contest_full_frame_sideinfo_consumption_proof`.
- `first_anchor_timing_smoke_allowed=true` is refused unless both Dykstra and
  side-info proof evidence are valid.

This is a preflight/discoverability guard, not a score claim.

## Verification

```text
.venv/bin/ruff check \
  tools/all_lanes_preflight.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py
# All checks passed

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
# 22 passed

PYTHONPATH=src .venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path("tools/all_lanes_preflight.py")
spec = importlib.util.spec_from_file_location("all_lanes_preflight_live_gate", path)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
ok, output = mod._run_operator_briefing_dispatch_gate()
print("ok=", ok)
print(output)
PY
# ok=True
# operator briefing dispatch routing: PASS
```
