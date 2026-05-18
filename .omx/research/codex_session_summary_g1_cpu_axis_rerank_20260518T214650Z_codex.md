# Codex session summary: G1 CPU-axis rerank

Date: 2026-05-18T21:46:50Z
Session: 019de465

## Landed

- Registered missing G1/F1/B1 routing directives in canonical task status.
- Claimed G1 Phase 1 local CPU-axis rerank.
- Added `tac.frontier_scan` G1 CPU-axis helper payload and canonical scanner
  text block.
- Added `tools/cpu_axis_optimal_archive_selector.py`.
- Added directive-literal `tools/probe_g1_cpu_axis_re_rank.py`.
- Materialized
  `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`.
- Registered Catalog #313 probe outcome
  `g1_cpu_axis_re_rank_20260518` as advisory `PARTIAL`.
- Registered the missing G1 lane
  `lane_rate_attack_g1_cpu_axis_specific_20260518`.
- Recorded F1 reframing note in canonical task status after subagent review
  and operator correction.
- Fixed the G1 family classifier from substring matching to token matching
  after Huygens caught the `pr106_component_prefix16_pr101grammar` trap.
- Registered A1-SPECIALIZED deterministic packet compiler as a live Phase 0
  feasibility task.

## Result

G1 existing-anchor rerank did not move the CPU frontier:

- current CPU frontier: `0.1920513168811056`
- best CPU anchor: PR101/fec6
- delta vs current frontier: `+0.0000000000`
- verdict: `FRONTIER_STABLE_VIA_RE_RANK`
- evidence base: 194 canonical anchors; 55 qualifying CPU anchors
- metadata-bucket method: `metadata_token_match_no_archive_sha_guess`

## Remaining

- G1 remains useful only when a new paired Linux x86_64 CPU anchor appears.
- F1 should be rewritten as A2-style scorer-blind RGB perturbation capacity,
  not as PoseNet dims 7-12 archive-channel capacity.
- A1-SPECIALIZED should stay live under the deterministic packet compiler:
  naive full-scorer/full-PoseNet inflate remains rejected, but a tiny
  self-contained per-pattern transducer/fixed table/generated code path is
  canonical if exact CUDA auth eval validates it.
- A1 byte-rate arithmetic must stay explicit: 5-20 KiB costs roughly
  `+0.0034` to `+0.0136` score, so specialized transducers are viable only when
  measured savings exceed their own byte-rate cost.
- B1 remains pending.
