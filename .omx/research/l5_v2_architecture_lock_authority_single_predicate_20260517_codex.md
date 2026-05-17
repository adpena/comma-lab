# L5 v2 architecture-lock authority single predicate - 2026-05-17

## Context

Operator asked to check markdown outside `.omx/research` because fresh L5 /
cargo-cult directives may sit in the `.omx` parent/control-plane area.

Checked:

- `.omx/notepad.md` - stale April AV1/Track-B notebook, not authoritative for
  TT5L v2.
- `.omx/release_manifest_v0.2.0-rc1.md` - release recipe; useful public-hygiene
  context, not an L5 dispatch authority.
- `.omx/state/current_focus.md` - stale 2026-05-15 but still explicitly names
  Time-Traveler L5 staircase as the active rebaseline priority.
- `.omx/state/next_experiments.md` / `.omx/state/dispatch_queue.md` - older
  queue surfaces, not current TT5L v2 actuation authority.
- `.omx/state/active_lane_dispatch_claims.md` - confirms recent TT5L paired
  CPU/CUDA diagnostic anchors are terminal; no new dispatch was made in this
  patch.

## Finding

The TT5L readiness surface and architecture-lock packet already produced the
same live boolean in the current repo state, but they reconstructed the
architecture-lock threshold independently. That is exactly the kind of quiet
authority drift the L5/cargo-cult work is trying to eliminate: one future check
could be added to the packet and forgotten in readiness, or vice versa.

## Change

- Added `_L5_V2_ARCHITECTURE_LOCK_CHECK_BLOCKERS` as the ordered checklist.
- Added `_l5_v2_architecture_lock_authority(...)` as the only predicate that
  translates checks into `architecture_lock_allowed` and blockers.
- `tt5l_campaign_readiness` now emits:
  - `architecture_lock_authority`
  - `architecture_lock_required_checks`
  - `architecture_lock_blockers`
- `l5_v2_architecture_lock_packet()` consumes that same authority payload
  instead of rebuilding the threshold locally.
- Regenerated `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.{json,md}`.

## Current lock state

`architecture_lock_allowed=false`

Current blockers:

- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`

No score, rank, promotion, or dispatch authority is created by this patch.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_requires_timing_and_anchor src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_allows_only_after_full_custody src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_artifact_tracks_live_payload src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_cli_writes_no_lock_packet -q`
  - `4 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
  - `134 passed`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py tools/build_l5_v2_architecture_lock_packet.py`
  - clean
- `.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py tools/build_l5_v2_architecture_lock_packet.py`
  - clean
