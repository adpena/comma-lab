# L5 v2 Architecture Lock Refresh After TT5L Materialization - 2026-05-17

## Purpose

Refresh the derived L5 v2 architecture-lock packet after the TT5L materialized
paired work unit moved from the prior all-zero trained/source archive to the
full-contest `random_lsb` side-info control.

## Command

```bash
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py
```

Result:

```text
architecture_lock_allowed=false
blockers=[
  'requires_all_l5_v2_gate_evidence_valid',
  'requires_c1_z5_tt5l_probe_gate_evidence',
  'requires_paired_cpu_cuda_axis_plan',
  'requires_paired_cpu_cuda_sideinfo_effect_curve',
  'requires_tt5l_first_anchor_timing_smoke_artifact',
  'requires_exact_or_diagnostic_anchor_pair'
]
score_claim=false
```

## Artifact Changes

- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.md`

The packet now surfaces:

- `next_action=review_and_execute_l5_v2_tt5l_materialized_paired_measurement`
- archive:
  `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- archive sha256:
  `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`

The old `l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_all_zero`
derived blocker is gone from the refreshed architecture-lock packet. The
architecture lock itself remains blocked because paired CPU/CUDA side-info
effect-curve evidence has not been executed or harvested.

## No-Authority Statement

No dispatch, no score claim, no promotion, no rank/kill authority. This is a
derived-control-plane freshness update so downstream readers do not see stale
all-zero materialized-work-unit state after the random-LSB work unit landed.
