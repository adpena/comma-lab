# Codex Findings: MLX Parent Contract Auth-Cache Guard

- UTC: 20260522T102055Z
- Lane: mlx_parent_production_contract_planning
- Evidence grade: macOS-MLX research-signal only
- Score authority: false
- Promotion eligible: false

## What changed

The MLX parent-production-contract planner now accepts repeatable cache/auth audit inputs via both `--cache-auth-audit` and `--auth-cache-audit`. It uses those audits to compare each required parent group's candidate cache identity against known auth-axis scorer-input identity for the same archive.

This specifically prevents the stale-cache failure mode where an MLX response dataset points at the correct archive SHA-256 but a non-auth-faithful local inflated/raw/tensor cache.

## Live 600-row result

Refreshed plan:

- `experiments/results/mlx_parent_contract_plan_20260522T102015Z_failed_auth_audit_v2/parent_production_contract_plan.json`
- `experiments/results/mlx_parent_contract_plan_20260522T102015Z_failed_auth_audit_v2/parent_production_contract_plan.md`

Summary:

- MLX rows: 600
- Required parent groups: 2
- Covered parent groups: 0
- Cache/auth audits supplied: 1
- Cache/auth matched groups: 0
- Cache/auth mismatched groups: 1
- Cache/auth missing groups: 1
- Status: blocked

FEC6 group `mlx_parent_contract_d2615878d356bd1d` is now correctly blocked against the known contest-CPU auth identity for archive `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`:

- `inflated_outputs_aggregate_sha256` mismatch: dataset/local `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1` vs auth `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d`
- `raw_sha256` mismatch: dataset/local `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c` vs auth `fef02ccd53ad4355f2dbb8e0b9cd4efb847daa243bd35a8411c5260d584fda8b`
- candidate `posenet_yuv6_pair` array mismatch: local `aae96b7cb270059174d987740a95e9fd0d9f4474142fd77ed1c1fce6a4124ed0` vs auth `04687540fa97209157b2ab9bcb200d098169f826db54aa4ced00c48c312bca91`
- candidate `segnet_last_rgb` array mismatch: local `ea4cf2c4879fcdf4cd177cc4e3c762433aa076b631ce252947372cda4da37536` vs auth `59ea5240178801774d59314a2de98764e3dba2d33c7ce3acc995bd2e87e6806d`

Decoder-q group `mlx_parent_contract_f5391bf78f60224c` remains blocked because no auth-axis cache audit has been supplied for archive `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`, and no strict parent production contract covers it.

## Decoder-q parity prerequisite

The decoder-q candidate cache completed a full 300-window singleton CPU parity sweep:

- Artifact: `experiments/results/mlx_parent_contract_prereqs_20260522T1006Z/decoderq_candidate_torch_parity_sweep_cpu_singleton_pairs0_300.json`
- Verdict: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- Covered pair window: `[0, 300]`
- Failed windows: 0
- Max PoseNet output abs diff: `7.62939453125e-06`
- Max SegNet logit abs diff: `0.00011682510375976562`
- SegNet argmax mismatch pixels total: 1

This is useful for a future decoder-q parent production contract, but it is not sufficient without cache/auth identity audit, profile stability, reference parity, score calibration, and contract bundle coverage.

## Final status

MLX local acceleration is still not production-ready for spend triage or training target selection. The next highest-EV action is to rebuild/regenerate FEC6 scorer-response rows from the auth-faithful tensor cache identity, or materialize/download the auth tensor payloads if they are not locally available. Decoder-q needs its own contest-CPU auth eval/cache audit before it can become a strict production-contract parent.
