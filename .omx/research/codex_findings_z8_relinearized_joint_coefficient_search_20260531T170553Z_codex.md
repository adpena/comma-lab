# Codex Findings: Z8 relinearized joint coefficient search

- Timestamp UTC: 2026-05-31T17:05:53Z
- Scope: Z8 joint P18/P19 coefficient water-fill, iterative relinearization contract, CLI smoke.
- Authority: `[macOS-CPU advisory]` / local archive materialization only. No contest CPU/CUDA score claim, promotion, rank/kill, or exact dispatch authority.

## What Landed

The Z8 coefficient rate attack is no longer only a single frozen-surface materializer. It now has a bounded relinearized search API and CLI:

- `run_joint_p18_p19_relinearized_deadzone_search(...)`
- `materialize_joint_p18_p19_relinearized_deadzone_search(...)`
- `tools/run_z8_joint_p18_p19_relinearized_deadzone_search.py`

Each iteration consumes a fresh joint P18/P19 surface, evaluates a deterministic dead-zone/quantization grid, accepts the best candidate under cumulative distortion guard, and measures cumulative rate/distortion versus the original archive. Duplicate surface reuse fails closed by default so an iterative run cannot masquerade as MLX scorer-VJP relinearization.

## Live Smoke

Input archive:
`experiments/results/z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_20260530T161526Z/submission/archive/0.bin`

Output manifest:
`.omx/research/z8_joint_p18_p19_relinearized_search_smoke_20260531T170553Z/candidate/z8_joint_p18_p19_relinearized_search_manifest.json`

Observed local advisory deltas:

- Z8HPC1 archive bytes: `92,408 -> 7,692` (`archive_rate_ratio=0.08324`)
- Iterations accepted: `2`
- Candidate grid rows: `4`
- Cumulative small-receiver distortion: `mse=0.001073921`, `mae=0.0239236`, `max_abs_delta=0.204564`
- Exact blocker preserved: `receiver_proof_and_contest_cpu_cuda_eval_not_executed`

## Guardrails

- Still false authority across all materializer/search rows.
- Fresh-surface guard is strict by default.
- The STE boundary is explicit as local proxy only: `straight_through_deadzone_quantization_proxy`.
- Pose protection is explicit through the consumed `rate_attack_deadzone_mask`; real MLX scorer-VJP surface production remains the next upstream work item.
