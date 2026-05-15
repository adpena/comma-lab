# PR101/FEC6 CPU Component Selector Sub-0.192 Profile

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- reference archive: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- reference bytes: `178517`
- exact CPU score: `0.1920513168811056`
- exact CUDA score: `0.22621002169349796`
- strict CPU byte target at unchanged components: `78`
- unchanged-component archive byte limit: `178439`

## Rate-Only Selector Compression

- FEC6 selector payload bytes: `249`
- FEC6 global entropy floor bytes: `241`
- best charged FEC7 saving vs FEC6: `-19`
- blocked: `true`

## Component-Moving Selector Scan

| packet | bytes | saved vs FEC6 | byte-closed | kind | rate-only CPU score | proxy-est CPU score | verdict |
|---|---:|---:|---|---|---:|---:|---|
| experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k4_cpu_overlay_20260515_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138689213 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec3_compact_greedy_k4_cpu_overlay_20260515_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138689213 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec3_greedy_k4_base_overlay_20260514_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138698299 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec3_greedy_k4_cpu_top16_overlay_20260514_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138689213 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec3_greedy_k4_cpu_top64_overlay_20260514_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138738944 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec3_greedy_k4_lattice_overlay_20260514_codex/packet_manifest.json | 178434 | 83 | true | component_moving_selector_policy | 0.191996050588 | 0.192138689213 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_compact_k4_20260514T1916Z_codex/packet_manifest.json | 178425 | 92 | false | component_moving_selector_policy | 0.191990057857 | 0.192132705568 | component_moving_rate_feasible_proxy_blocks_gate |
| experiments/results/pr101_frame_exploit_selector_fec5_fixed_huffman_k8_cpu_overlay_20260515_codex/packet_manifest.json | 178477 | 40 | true | component_moving_selector_policy | 0.192024682523 | 0.192070681717 | component_moving_no_gate_evidence |
| experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/packet_manifest.json | 178517 | 0 | true | component_moving_selector_policy | 0.192051316881 | 0.192097316075 | component_moving_no_gate_evidence |
| experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex/packet_manifest.json | 178517 | 0 | true | component_moving_selector_policy | 0.192051316881 | 0.192104699147 | component_moving_no_gate_evidence |
| experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex/packet_manifest.json | 178517 | 0 | true | component_moving_selector_policy | 0.192051316881 | 0.192094498758 | component_moving_no_gate_evidence |
| experiments/results/pr101_frame_exploit_selector_fec3_compact_greedy_k8_cpu_overlay_20260515_codex/packet_manifest.json | 178517 | 0 | true | component_moving_selector_policy | 0.192051316881 | 0.192097316075 | component_moving_no_gate_evidence |

## Best Rate-Feasible Component-Moving Row

- packet: `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k4_cpu_overlay_20260515_codex/packet_manifest.json`
- bytes: `178434`
- saved vs FEC6: `83`
- rate-only CPU score if components unchanged: `0.19199605058799646`
- allowable component delta vs FEC6: `3.949412003545483e-06`
- proxy component delta vs FEC6: `0.00014263862536600946`
- verdict: `component_moving_rate_feasible_proxy_blocks_gate`

## Conclusion

- verdict: `hard_blocker_under_current_component_rows`
- found_feasible_sub0192_candidate_path: `false`
- hard_blocker: `true`
- reason: Rate-only selector compression is blocked; the smallest rate-feasible byte-closed component-moving packet changes selector codes and its proxy component delta (0.00014263862536600946) exceeds the strict CPU gate allowance (3.949412003545483e-06).

## 6-Hook Wire-In

1. Sensitivity-map contribution: `N/A - no new empirical score anchor; selector rows preserve score_claim=false`.
2. Pareto constraint: `CPU-axis gate only; CUDA axis recorded separately and remains non-promotional`.
3. Bit-allocator hook: strict byte target and per-candidate archive deltas are machine-readable.
4. Cathedral autopilot dispatch hook: `ready_for_exact_eval_dispatch=false`; no dispatch row emitted.
5. Continual-learning posterior update: blocker/candidate classification is recorded here; no score posterior update.
6. Probe-disambiguator: rate-only and component-moving selector mechanisms are reported as separate verdict spaces.

