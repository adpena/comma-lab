# L5 v2 TT5L Dykstra Archive-Size Basis - 2026-05-16

## Scope

While materializing the TT5L Dykstra planning artifact, the tool output was
found to record `rate_contribution` but not the exact `archive_size_bytes`
input that produced it. That made the artifact unnecessarily arbitrary: a
future reviewer could not recover which candidate archive byte size was used
without shell history.

## Landed Change

`tools/check_substrate_dykstra_feasibility.py` now includes
`archive_size_bytes` in `DykstraFeasibilityVerdict` and in emitted JSON.

`src/tac/optimization/l5_staircase_v2.py` now refuses a TT5L Dykstra artifact
that omits a positive integer `archive_size_bytes`, using blocker:

```text
tt5l_dykstra_feasibility_archive_size_bytes_missing
```

This preserves the Dykstra artifact as planning-control evidence only:

```text
score_claim=false
promotion_eligible=false
ready_for_exact_eval_dispatch=false
```

## Materialized Local Artifact

Candidate archive basis:

```text
path=experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/submission_dir/archive.zip
archive_size_bytes=34603
archive_sha256=2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a
axis=planning-only candidate-size feasibility, not score authority
```

Command:

```text
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
  --substrate-id time_traveler_l5_5move \
  --predicted-band-lo 0.150 \
  --predicted-band-hi 0.170 \
  --archive-size-bytes 34603 \
  --output-json .omx/state/dykstra_feasibility_time_traveler_l5.json
```

Result:

```text
substrate_id=time_traveler_l5_5move
archive_size_bytes=34603
verdict=FEASIBLE
feasibility_band_lo=0.15
feasibility_band_hi=0.17
```

Live operator-briefing state after materialization:

```text
next_non_pr106_l5_action=materialize_tt5l_contest_full_frame_sideinfo_consumption_proof
dykstra_feasibility_artifact_valid=true
dykstra_archive_size_bytes=34603
first_anchor_timing_smoke_allowed=false
```

## Verification

```text
.venv/bin/ruff check \
  tools/check_substrate_dykstra_feasibility.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py
# All checks passed

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_dykstra_artifact_unblocks_sideinfo_next_action \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_dykstra_artifact_requires_archive_size_basis \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_timing_requires_dykstra_and_sideinfo_evidence -q
# 11 passed

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
